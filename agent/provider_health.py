"""Small persistent provider-health guard for Baldr/Hermes.

The gateway keeps long-lived agents, but usage limits are account-level.  This
file gives every process a shared, non-secret signal that a provider should be
avoided for a while after quota/auth/billing failures.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


SCHEMA = "baldr-provider-health-v1"
DEFAULT_STATE_PATH = "/srv/agent/state/provider-health.json"


_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s'\"]+", re.I),
    re.compile(r"(OPENAI_API_KEY|OPENROUTER_API_KEY|GITHUB_TOKEN|GH_TOKEN|MATRIX_ACCESS_TOKEN)\s*=\s*[^\s'\"]+", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
]


def _state_path() -> Path:
    return Path(os.environ.get("BALDR_PROVIDER_HEALTH_STATE", DEFAULT_STATE_PATH))


def _provider_key(provider: str | None) -> str:
    return (provider or "").strip().lower()


def _reason_value(reason: Any) -> str:
    value = getattr(reason, "value", reason)
    return str(value or "unknown").strip().lower() or "unknown"


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _safe_env_slug(provider: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_") or "PROVIDER"


def _utc(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _redact(text: str) -> str:
    redacted = text or ""
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", redacted)
    if len(redacted) > 500:
        redacted = redacted[:500] + "...[truncated]"
    return redacted


def _read_state() -> dict[str, Any]:
    path = _state_path()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault("schema", SCHEMA)
    providers = state.get("providers")
    if not isinstance(providers, dict):
        state["providers"] = {}
    return state


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    old_umask = os.umask(0o177)
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        path.chmod(0o600)
    finally:
        os.umask(old_umask)
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def cooldown_seconds_for(provider: str | None, reason: Any) -> int:
    provider_key = _provider_key(provider)
    reason_key = _reason_value(reason)
    slug = _safe_env_slug(provider_key)

    if provider_key == "openai-codex":
        if reason_key == "billing":
            return _env_int("BALDR_CODEX_BILLING_COOLDOWN_SECONDS", 86400)
        if reason_key == "rate_limit":
            return _env_int("BALDR_CODEX_RATE_LIMIT_COOLDOWN_SECONDS", 3600)
        if reason_key in {"auth", "auth_permanent"}:
            return _env_int("BALDR_CODEX_AUTH_COOLDOWN_SECONDS", 3600)

    if reason_key == "billing":
        return _env_int(f"BALDR_{slug}_BILLING_COOLDOWN_SECONDS", 3600)
    if reason_key == "rate_limit":
        return _env_int(f"BALDR_{slug}_RATE_LIMIT_COOLDOWN_SECONDS", 300)
    if reason_key in {"auth", "auth_permanent"}:
        return _env_int(f"BALDR_{slug}_AUTH_COOLDOWN_SECONDS", 600)
    return _env_int(f"BALDR_{slug}_COOLDOWN_SECONDS", 300)


def provider_cooldown(provider: str | None, *, now: float | None = None) -> dict[str, Any]:
    provider_key = _provider_key(provider)
    if not provider_key:
        return {"active": False, "provider": provider_key}

    state = _read_state()
    entry = (state.get("providers") or {}).get(provider_key)
    if not isinstance(entry, dict):
        return {"active": False, "provider": provider_key}

    now_ts = time.time() if now is None else now
    try:
        blocked_until = float(entry.get("blocked_until") or 0)
    except (TypeError, ValueError):
        blocked_until = 0.0
    seconds_remaining = max(0.0, blocked_until - now_ts)
    result = dict(entry)
    result.update(
        {
            "active": seconds_remaining > 0,
            "provider": provider_key,
            "blocked_until": blocked_until,
            "seconds_remaining": seconds_remaining,
        }
    )
    return result


def is_provider_blocked(provider: str | None) -> bool:
    return bool(provider_cooldown(provider).get("active"))


def record_provider_failure(
    *,
    provider: str | None,
    model: str | None = None,
    reason: Any = None,
    message: str = "",
    status_code: int | None = None,
    cooldown_seconds: int | None = None,
) -> dict[str, Any]:
    provider_key = _provider_key(provider)
    if not provider_key:
        return {"active": False, "provider": provider_key}

    reason_key = _reason_value(reason)
    cooldown = cooldown_seconds_for(provider_key, reason_key) if cooldown_seconds is None else max(0, int(cooldown_seconds))
    now_ts = time.time()
    entry: dict[str, Any] = {
        "status": "cooldown" if cooldown > 0 else "failed",
        "reason": reason_key,
        "model": str(model or "").strip(),
        "status_code": status_code,
        "last_failure_utc": _utc(now_ts),
        "cooldown_seconds": cooldown,
        "blocked_until": now_ts + cooldown if cooldown > 0 else 0,
        "blocked_until_utc": _utc(now_ts + cooldown) if cooldown > 0 else None,
    }
    if message:
        entry["message"] = _redact(str(message))

    state = _read_state()
    state.setdefault("providers", {})[provider_key] = entry
    state["updated_utc"] = _utc(now_ts)
    _write_state(state)
    return provider_cooldown(provider_key)


def record_provider_success(*, provider: str | None, model: str | None = None) -> None:
    provider_key = _provider_key(provider)
    if not provider_key:
        return
    state = _read_state()
    providers = state.setdefault("providers", {})
    entry = providers.get(provider_key)
    if not isinstance(entry, dict):
        entry = {}
    entry.update(
        {
            "status": "ok",
            "reason": None,
            "model": str(model or entry.get("model") or "").strip(),
            "last_success_utc": _utc(),
            "blocked_until": 0,
            "blocked_until_utc": None,
            "seconds_remaining": 0,
        }
    )
    providers[provider_key] = entry
    state["updated_utc"] = _utc()
    _write_state(state)
