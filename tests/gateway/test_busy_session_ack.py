"""Tests for busy-session acknowledgment when user sends messages during active agent runs.

Verifies that users get an immediate status response instead of total silence
when the agent is working on a task. See PR fix for the @Lonely__MH report.
"""
import asyncio
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can import gateway code without heavy deps
# ---------------------------------------------------------------------------
import sys, types

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    Platform,
    SessionSource,
    build_session_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text="hello", chat_id="123", platform_val="telegram"):
    """Build a minimal MessageEvent."""
    platform = platform_val if hasattr(platform_val, "value") else MagicMock(value=platform_val)
    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    evt = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )
    return evt


def _make_runner():
    """Build a minimal GatewayRunner-like object for testing."""
    os.environ["HERMES_GATEWAY_BUSY_ACK_ENABLED"] = "true"
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter(platform_val="telegram"):
    """Build a minimal adapter mock."""
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_val)
    return adapter


@pytest.fixture(autouse=True)
def _isolated_baldr_route_events(monkeypatch, tmp_path):
    monkeypatch.setenv("BALDR_ROUTE_EVENTS_PATH", str(tmp_path / "baldr-route-events.jsonl"))
    monkeypatch.setenv("BALDR_ROUTE_EVENT_ENV", "test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBusySessionAck:
    """User sends a message while agent is running — should get acknowledgment."""

    @pytest.mark.asyncio
    async def test_handle_message_queue_mode_queues_without_interrupt(self):
        """Runner queue mode must not interrupt an active agent for text follow-ups."""
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        adapter = _make_adapter()

        event = _make_event(text="follow up in queue mode")
        sk = build_session_key(event.source)

        running_agent = MagicMock()
        runner._busy_input_mode = "queue"
        runner._running_agents[sk] = running_agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_message(runner, event)

        assert result is None
        assert sk in adapter._pending_messages
        assert adapter._pending_messages[sk] is event
        assert sk not in runner._pending_messages
        running_agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_ack_when_agent_running(self):
        """First message during busy session should get a status ack."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="Are you working?")
        sk = build_session_key(event.source)

        # Simulate running agent
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 21,
            "max_iterations": 60,
            "current_tool": "terminal",
            "last_activity_ts": time.time(),
            "last_activity_desc": "terminal",
            "seconds_since_activity": 1.0,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 600  # 10 min ago
        runner.adapters[event.source.platform] = adapter

        result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True  # handled
        # Verify ack was sent
        adapter._send_with_retry.assert_called_once()
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        if not content and call_kwargs.args:
            # positional args
            content = str(call_kwargs)
        assert "Interrupting" in content or "respond" in content
        assert "/stop" not in content  # no need — we ARE interrupting

        # Verify agent interrupt was called
        agent.interrupt.assert_called_once_with("Are you working?")

    @pytest.mark.asyncio
    async def test_queue_mode_suppresses_interrupt_and_updates_ack(self):
        """When busy_input_mode is 'queue', message is queued WITHOUT interrupt."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="Add this to queue")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        # VERIFY: Agent was NOT interrupted
        agent.interrupt.assert_not_called()

        # VERIFY: Ack sent with queue-specific wording
        adapter._send_with_retry.assert_called_once()
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Queued for the next turn" in content
        assert "respond once the current task finishes" in content
        assert "Interrupting" not in content

    @pytest.mark.asyncio
    async def test_steer_mode_calls_agent_steer_no_interrupt_no_queue(self):
        """busy_input_mode='steer' injects via agent.steer() and skips queueing."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="also check the tests")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            await runner._handle_active_session_busy_message(event, sk)

        # VERIFY: Agent was steered, NOT interrupted
        agent.steer.assert_called_once_with("also check the tests")
        agent.interrupt.assert_not_called()

        # VERIFY: No queueing — successful steer must NOT replay as next turn
        mock_merge.assert_not_called()

        # VERIFY: Ack mentions steer wording
        adapter._send_with_retry.assert_called_once()
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Steered" in content or "steer" in content.lower()
        assert "Interrupting" not in content

    @pytest.mark.asyncio
    async def test_steer_mode_falls_back_to_queue_when_agent_rejects(self):
        """If agent.steer() returns False, fall back to queue behavior."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="empty or rejected")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=False)  # rejected
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            await runner._handle_active_session_busy_message(event, sk)

        agent.steer.assert_called_once()
        agent.interrupt.assert_not_called()
        # Fell back to queue semantics: event was merged into pending messages
        mock_merge.assert_called_once()

        # Ack uses queue-mode wording (not steer, not interrupt)
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Queued for the next turn" in content
        assert "Steered" not in content

    @pytest.mark.asyncio
    async def test_steer_mode_falls_back_to_queue_when_agent_pending(self):
        """If agent is still starting (sentinel), steer mode falls back to queue."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="arrived too early")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        # Agent is still being set up — sentinel in place
        runner._running_agents[sk] = sentinel

        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            await runner._handle_active_session_busy_message(event, sk)

        # Event was queued instead of steered
        mock_merge.assert_called_once()

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Queued for the next turn" in content

    @pytest.mark.asyncio
    async def test_debounce_suppresses_rapid_acks(self):
        """Second message within 30s should NOT send another ack."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event1 = _make_event(text="hello?")
        # Reuse the same source so platform mock matches
        event2 = MessageEvent(
            text="still there?",
            message_type=MessageType.TEXT,
            source=event1.source,
            message_id="msg2",
        )
        sk = build_session_key(event1.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 5,
            "max_iterations": 60,
            "current_tool": None,
            "last_activity_ts": time.time(),
            "last_activity_desc": "api_call",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 60
        runner.adapters[event1.source.platform] = adapter

        # First message — should get ack
        result1 = await runner._handle_active_session_busy_message(event1, sk)
        assert result1 is True
        assert adapter._send_with_retry.call_count == 1

        # Second message within cooldown — should be queued but no ack
        result2 = await runner._handle_active_session_busy_message(event2, sk)
        assert result2 is True
        assert adapter._send_with_retry.call_count == 1  # still 1, no new ack

        # But interrupt should still be called for both (since we are in interrupt mode)
        assert agent.interrupt.call_count == 2

    @pytest.mark.asyncio
    async def test_ack_after_cooldown_expires(self):
        """After 30s cooldown, a new message should send a fresh ack."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="hello?")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 10,
            "max_iterations": 60,
            "current_tool": "web_search",
            "last_activity_ts": time.time(),
            "last_activity_desc": "tool",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 120
        runner.adapters[event.source.platform] = adapter

        # First ack
        await runner._handle_active_session_busy_message(event, sk)
        assert adapter._send_with_retry.call_count == 1

        # Fake that cooldown expired
        runner._busy_ack_ts[sk] = time.time() - 31

        # Second ack should go through
        await runner._handle_active_session_busy_message(event, sk)
        assert adapter._send_with_retry.call_count == 2

    @pytest.mark.asyncio
    async def test_includes_status_detail(self):
        """Ack message should include iteration and tool info when available."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="yo")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 21,
            "max_iterations": 60,
            "current_tool": "terminal",
            "last_activity_ts": time.time(),
            "last_activity_desc": "terminal",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 600  # 10 min
        runner.adapters[event.source.platform] = adapter

        await runner._handle_active_session_busy_message(event, sk)

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")
        assert "21/60" in content  # iteration
        assert "terminal" in content  # current tool
        assert "10 min" in content  # elapsed

    @pytest.mark.asyncio
    async def test_draining_still_works(self):
        """Draining case should still produce the drain-specific message."""
        runner, sentinel = _make_runner()
        runner._draining = True
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="hello")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        # Mock the drain-specific methods
        runner._queue_during_drain_enabled = lambda: False
        runner._status_action_gerund = lambda: "restarting"

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is True

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")
        assert "restarting" in content

    @pytest.mark.asyncio
    async def test_pending_sentinel_no_interrupt(self):
        """When agent is PENDING_SENTINEL, don't call interrupt (it has no method)."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="hey")
        sk = build_session_key(event.source)

        runner._running_agents[sk] = sentinel
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is True
        # Should still send ack
        adapter._send_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_adapter_falls_through(self):
        """If adapter is missing, return False so default path handles it."""
        runner, sentinel = _make_runner()

        event = _make_event(text="hello")
        sk = build_session_key(event.source)

        # No adapter registered
        runner._running_agents[sk] = MagicMock()

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is False  # not handled, let default path try


class TestBusySessionOnboardingHint:
    """First-touch hint appended to the busy-ack the first time it fires."""

    @pytest.mark.asyncio
    async def test_first_busy_ack_appends_interrupt_hint(self, tmp_path, monkeypatch):
        """First busy-while-running message gets an extra hint about /busy."""
        import gateway.run as _gr

        monkeypatch.setattr(_gr, "_hermes_home", tmp_path)
        # mark_seen imports utils.atomic_yaml_write; make sure it resolves
        # against a writable dir by pointing _hermes_home at tmp_path.
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="ping")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 3, "max_iterations": 60,
            "current_tool": None, "last_activity_ts": time.time(),
            "last_activity_desc": "api", "seconds_since_activity": 0.1,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 5
        runner.adapters[event.source.platform] = adapter

        await runner._handle_active_session_busy_message(event, sk)

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")

        # Normal ack body
        assert "Interrupting" in content
        # First-touch hint appended
        assert "First-time tip" in content
        assert "/busy queue" in content

        # The flag is now persisted to tmp_path/config.yaml
        import yaml
        cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert cfg["onboarding"]["seen"]["busy_input_prompt"] is True

    @pytest.mark.asyncio
    async def test_second_busy_ack_omits_hint(self, tmp_path, monkeypatch):
        """Once the flag is marked, the hint never appears again."""
        import gateway.run as _gr
        import yaml

        monkeypatch.setattr(_gr, "_hermes_home", tmp_path)
        # Pre-populate the config so is_seen() returns True from the start.
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "onboarding": {"seen": {"busy_input_prompt": True}},
        }))
        monkeypatch.setattr(
            _gr, "_load_gateway_config",
            lambda: yaml.safe_load((tmp_path / "config.yaml").read_text()),
        )

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="ping again")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 3, "max_iterations": 60,
            "current_tool": None, "last_activity_ts": time.time(),
            "last_activity_desc": "api", "seconds_since_activity": 0.1,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 5
        runner.adapters[event.source.platform] = adapter

        await runner._handle_active_session_busy_message(event, sk)

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")

        assert "Interrupting" in content
        assert "First-time tip" not in content
        assert "/busy queue" not in content

    @pytest.mark.asyncio
    async def test_queue_mode_hint_points_to_interrupt(self, tmp_path, monkeypatch):
        """In queue mode the hint should suggest /busy interrupt, not /busy queue."""
        import gateway.run as _gr

        monkeypatch.setattr(_gr, "_hermes_home", tmp_path)
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="queue me")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Queued for the next turn" in content
        assert "First-time tip" in content
        assert "/busy interrupt" in content
        # Must NOT tell the user to /busy queue when they're already on queue.
        assert "/busy queue" not in content

    @pytest.mark.asyncio
    async def test_baldr_matrix_busy_status_answers_immediately(self, monkeypatch):
        """Bell's Matrix status/control questions should not wait in queue."""
        import gateway.run as _gr

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter("matrix")
        event = _make_event(text="а что именно ты щас делаешь чтоя жду", platform_val=Platform.MATRIX)
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = MagicMock()

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                return {"lane": "control", "execution": "answer_now", "priority": "high"}
            if args == ("status", "--json"):
                return {"tasks": [{"id": "apk", "lane": "code", "status": "active", "summary": "собираю Android APK"}]}
            return None

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)
        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        mock_merge.assert_not_called()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "сейчас:" in content
        assert "собираю Android APK" in content

    @pytest.mark.asyncio
    async def test_baldr_matrix_busy_chotam_answers_in_priority_guard(self, monkeypatch):
        """Matrix `чотам` follow-ups must bypass _handle_message's direct busy queue."""
        import gateway.run as _gr

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter("matrix")
        event = _make_event(text="чотам", platform_val=Platform.MATRIX)
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        agent = MagicMock()
        agent.get_activity_summary.return_value = {"seconds_since_activity": 0.0}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                return {"lane": "control", "execution": "answer_now", "priority": "normal"}
            if args == ("status", "--json"):
                return {"tasks": [{"id": "matrix-parallelism-busy-control", "lane": "ops", "status": "active", "summary": "чиню quick replies"}]}
            return None

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)
        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await _gr.GatewayRunner._handle_message(runner, event)

        assert result is None
        mock_merge.assert_not_called()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "сейчас:" in content
        assert "quick replies" in content

    @pytest.mark.asyncio
    async def test_baldr_matrix_busy_reply_chotam_anchors_status_to_replied_topic(self, monkeypatch):
        """Reply-context `чотам` should use the replied topic, not the global task dump."""
        import gateway.run as _gr

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter("matrix")
        event = _make_event(text="чотам", platform_val=Platform.MATRIX)
        event.reply_to_text = "чотам по более дешевым за крипту для селфхоста"
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = MagicMock()

        routed_text = '[Replying to: "чотам по более дешевым за крипту для селфхоста"]\n\nчотам'

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                assert args[1] == routed_text
                return {"lane": "control", "execution": "answer_now", "priority": "high"}
            if args == ("status-for-text", routed_text, "--json"):
                return {
                    "task_id": "selfhost-llm-crypto-vps",
                    "reply_text": "сейчас: selfhost-llm-crypto-vps [infra/completed] — live Vast shortlist ready\nпоследнее: Vast offer 20299950 shortlisted\nдальше: дождаться approve на реальный deploy",
                }
            if args == ("status", "--json"):
                return {
                    "tasks": [
                        {"id": "matrix-parallelism-busy-control", "lane": "ops", "status": "active", "summary": "чиню quick replies"}
                    ]
                }
            return None

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)
        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        mock_merge.assert_not_called()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "selfhost-llm-crypto-vps" in content
        assert "Vast offer 20299950" in content
        assert "matrix-parallelism-busy-control" not in content

    @pytest.mark.asyncio
    async def test_baldr_matrix_busy_reply_chotam_anchors_status_in_priority_guard(self, monkeypatch):
        """Reply-context `чотам` should stay anchored even on _handle_message's direct busy path."""
        import gateway.run as _gr

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter("matrix")
        event = _make_event(text="чотам", platform_val=Platform.MATRIX)
        event.reply_to_text = "чотам по более дешевым за крипту для селфхоста"
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = MagicMock()

        routed_text = '[Replying to: "чотам по более дешевым за крипту для селфхоста"]\n\nчотам'

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                assert args[1] == routed_text
                return {"lane": "control", "execution": "answer_now", "priority": "high"}
            if args == ("status-for-text", routed_text, "--json"):
                return {
                    "task_id": "selfhost-llm-crypto-vps",
                    "reply_text": "сейчас: selfhost-llm-crypto-vps [infra/completed] — live Vast shortlist ready\nпоследнее: Vast offer 20299950 shortlisted\nдальше: дождаться approve на реальный deploy",
                }
            if args == ("status", "--json"):
                return {
                    "tasks": [
                        {"id": "matrix-parallelism-busy-control", "lane": "ops", "status": "active", "summary": "чиню quick replies"}
                    ]
                }
            return None

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)
        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await _gr.GatewayRunner._handle_message(runner, event)

        assert result is None
        mock_merge.assert_not_called()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "selfhost-llm-crypto-vps" in content
        assert "Vast offer 20299950" in content
        assert "matrix-parallelism-busy-control" not in content

    @pytest.mark.asyncio
    async def test_baldr_matrix_busy_correction_answers_immediately(self, monkeypatch):
        """Parallelism/routing corrections should be answered immediately, not queued."""
        import gateway.run as _gr

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter("matrix")
        event = _make_event(
            text="тебе опять надо ревьювить твой параллелизм. ты не отвечаешь в других линиях",
            platform_val=Platform.MATRIX,
        )
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = MagicMock()

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                return {"lane": "ops", "execution": "answer_now", "priority": "urgent"}
            return None

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)
        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        mock_merge.assert_not_called()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "не должно ждать очереди" in content
        assert "busy-control" in content

    @pytest.mark.asyncio
    async def test_baldr_matrix_busy_correction_interrupts_active_turn(self, monkeypatch):
        """Urgent Matrix corrections should ack immediately and stop stale work."""
        import gateway.run as _gr

        runner, _sentinel = _make_runner()
        adapter = _make_adapter(platform_val=Platform.MATRIX)
        event = _make_event(
            text="запиши себе: не лезь в код, ты наделал косяк",
            chat_id="!room:matrix.org",
            platform_val=Platform.MATRIX,
        )
        sk = build_session_key(event.source)
        running_agent = MagicMock()
        runner._busy_input_mode = "queue"
        runner._running_agents[sk] = running_agent
        runner.adapters[event.source.platform] = adapter

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                return {"lane": "ops", "execution": "answer_now", "priority": "urgent"}
            return None

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)

        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        adapter._send_with_retry.assert_awaited_once()
        running_agent.interrupt.assert_called_once_with(event.text)
        mock_merge.assert_not_called()

    def test_append_baldr_route_event_marks_test_env(self, tmp_path):
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        path = tmp_path / "route-events.jsonl"
        os.environ["BALDR_ROUTE_EVENTS_PATH"] = str(path)
        os.environ["BALDR_ROUTE_EVENT_ENV"] = "test"

        GatewayRunner._append_baldr_route_event(runner, {"source": "matrix-busy-router", "message_id": "msg1"})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        item = __import__("json").loads(lines[0])
        assert item["env"] == "test"
        assert item["source"] == "matrix-busy-router"

    @pytest.mark.asyncio
    async def test_baldrctl_json_rejects_non_allowlisted_command(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        create_subprocess = AsyncMock()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

        result = await GatewayRunner._baldrctl_json(runner, "workers", "--json")

        assert result is None
        create_subprocess.assert_not_called()

    @pytest.mark.asyncio
    async def test_baldr_enqueue_link_review_passes_matrix_metadata(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        event = _make_event(text="смотри https://example.com/tool", chat_id="!room:matrix.org", platform_val=Platform.MATRIX)
        event.source.user_name = "bell"
        captured = {}

        class FakeProc:
            returncode = 0

            async def communicate(self):
                return (b'{"queued": true, "item": {"id": "link-review-123"}}', b"")

        async def fake_create_subprocess_exec(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

        item = await GatewayRunner._baldr_enqueue_link_review(
            runner,
            event,
            reply_anchor_text="чотам по ссылке",
        )

        assert item == {"id": "link-review-123"}
        assert captured["args"][:10] == (
            "/srv/agent/scripts/baldr-link-review-queue.py",
            "enqueue",
            "--text",
            "смотри https://example.com/tool",
            "--source",
            "matrix",
            "--message-id",
            "msg1",
            "--chat-id",
            "!room:matrix.org",
        )
        assert "bell" in captured["args"]
        assert "чотам по ссылке" in captured["args"]
        assert captured["kwargs"]["stdout"] == asyncio.subprocess.PIPE
        assert captured["kwargs"]["stderr"] == asyncio.subprocess.DEVNULL

    @pytest.mark.asyncio
    async def test_baldr_matrix_busy_route_appends_durable_route_event(self, monkeypatch):
        """Busy-path routing should persist a durable event with reply anchor and target."""
        import gateway.run as _gr

        runner, _sentinel = _make_runner()
        adapter = _make_adapter(platform_val=Platform.MATRIX)
        event = _make_event(text="чотам", chat_id="!room:matrix.org", platform_val=Platform.MATRIX)
        event.reply_to_text = "чотам по selfhost-llm-crypto-vps"
        event.reply_to_message_id = "$prev"
        sk = build_session_key(event.source)
        runner._busy_input_mode = "queue"
        runner._running_agents[sk] = MagicMock()
        runner.adapters[event.source.platform] = adapter

        captured = []
        routed_text = '[Replying to: "чотам по selfhost-llm-crypto-vps"]\n\nчотам'

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                return {
                    "lane": "control",
                    "execution": "answer_now",
                    "priority": "high",
                    "task_id": "selfhost-llm-vast-deploy",
                    "target_worker": "planner/selfhost",
                }
            if args == ("status-for-text", routed_text, "--json"):
                return {"reply_text": "сейчас: selfhost-llm-vast-deploy [infra/blocked] — ждёт approve"}
            return None

        def fake_append(self, payload):
            captured.append(payload)

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)
        monkeypatch.setattr(_gr.GatewayRunner, "_append_baldr_route_event", fake_append)
        monkeypatch.setattr(
            _gr.GatewayRunner,
            "_baldr_enqueue_link_review",
            AsyncMock(return_value={"id": "link-review-123"}),
        )

        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        mock_merge.assert_not_called()
        assert len(captured) == 1
        route_event = captured[0]
        assert route_event["source"] == "matrix-busy-router"
        assert route_event["session_key"] == sk
        assert route_event["chat_id"] == "!room:matrix.org"
        assert route_event["reply_to_message_id"] == "$prev"
        assert route_event["reply_anchor_text"] == "чотам по selfhost-llm-crypto-vps"
        assert route_event["route_text"] == routed_text
        assert route_event["lane"] == "control"
        assert route_event["priority"] == "high"
        assert route_event["execution"] == "answer_now"
        assert route_event["target_task_id"] == "selfhost-llm-vast-deploy"
        assert route_event["target_worker"] == "planner/selfhost"
        assert route_event["should_interrupt"] is False
        assert route_event["link_review_queue_id"] == "link-review-123"

    @pytest.mark.asyncio
    async def test_baldr_matrix_busy_background_route_preserves_queue_order(self, monkeypatch):
        """Background/delegate busy routes must not emit visible /background noise."""
        import gateway.run as _gr

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter("matrix")
        event = _make_event(text="https://github.com/example/repo что тут", platform_val=Platform.MATRIX)
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        agent = MagicMock()
        agent.get_activity_summary.return_value = {"api_call_count": 2, "max_iterations": 90}
        runner._running_agents[sk] = agent

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                return {"lane": "research", "execution": "background", "priority": "normal"}
            return None

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)
        runner._handle_background_command = AsyncMock(return_value="SHOULD NOT SEND BACKGROUND ACK")

        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        runner._handle_background_command.assert_not_called()
        mock_merge.assert_called_once()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Queued for the next turn" in content
        assert "Background task started" not in content
        assert "Background task complete" not in content

    @pytest.mark.asyncio
    async def test_baldr_unknown_control_answer_now_has_no_generic_filler(self):
        """Unknown control/answer_now messages should not get empty meta replies."""
        runner, _sentinel = _make_runner()

        content = await runner._baldr_control_reply_text(
            "ок",
            lane="control",
            priority="normal",
            execution="answer_now",
        )

        assert content is None

    @pytest.mark.asyncio
    async def test_run_agent_pending_baldr_answer_now_skips_queued_followup(self, monkeypatch, tmp_path):
        """Pending Matrix control/ops answer_now must not wait behind queued follow-up delivery."""
        import gateway.run as _gr
        import hermes_cli.tools_config as tools_config

        class FakeAgent:
            call_count = 0

            def __init__(self, *args, **kwargs):
                self.tools = []
                self.model = "gpt-5.4"

            def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
                type(self).call_count += 1
                return {
                    "final_response": "first response",
                    "messages": [],
                    "api_calls": 1,
                    "completed": True,
                }

        class FakeAdapter:
            edit_message = BasePlatformAdapter.edit_message

            def __init__(self):
                self._pending_messages = {}
                self._active_sessions = {}
                self._post_delivery_callbacks = {}
                self.config = MagicMock()
                self.config.extra = {}
                self.platform = Platform.MATRIX
                self.send = AsyncMock()
                self._send_with_retry = AsyncMock(return_value=True)

            def get_pending_message(self, session_key):
                return self._pending_messages.pop(session_key, None)

            def has_pending_interrupt(self, session_key):
                return False

        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
        monkeypatch.setattr(_gr, "_hermes_home", tmp_path)
        monkeypatch.setattr(_gr, "_env_path", tmp_path / ".env")
        monkeypatch.setattr(_gr, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(_gr, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
        monkeypatch.setattr(
            _gr,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://example.invalid",
                "api_key": "***",
            },
        )
        monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

        runner, _sentinel = _make_runner()
        runner._ephemeral_system_prompt = ""
        runner._prefill_messages = []
        runner._reasoning_config = None
        runner._show_reasoning = False
        runner._provider_routing = {}
        runner._fallback_model = None
        runner._service_tier = None
        runner._background_tasks = set()
        runner._session_db = None
        runner._session_model_overrides = {}
        runner._session_reasoning_overrides = {}
        runner._pending_model_notes = {}
        runner._pending_approvals = {}
        runner._agent_cache = {}
        runner._agent_cache_lock = threading.Lock()
        runner._queued_events = {}
        runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
        runner._enrich_message_with_vision = AsyncMock(return_value="initial")
        runner._draining = True

        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:matrix.org",
            chat_type="dm",
            user_id="@bell:matrix.org",
            thread_id="thread-1",
        )
        pending_event = MessageEvent(
            text="чотам",
            message_type=MessageType.TEXT,
            source=source,
            message_id="$pending",
        )
        session_key = build_session_key(source)
        adapter = FakeAdapter()
        adapter._pending_messages[session_key] = pending_event
        runner.adapters[source.platform] = adapter

        async def fake_json(self, *args, timeout=1.5):
            if args[:1] == ("route",):
                return {"lane": "control", "execution": "answer_now", "priority": "urgent"}
            if args == ("status", "--json"):
                return {"tasks": [{"id": "matrix-parallelism-busy-control", "lane": "ops", "status": "active", "summary": "quick replies verified"}]}
            return None

        monkeypatch.setattr(_gr.GatewayRunner, "_baldrctl_json", fake_json)
        FakeAgent.call_count = 0

        result = await runner._run_agent(
            message="initial",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key=session_key,
        )

        assert result["final_response"] == "first response"
        assert FakeAgent.call_count == 1
        assert session_key not in adapter._pending_messages
        adapter._send_with_retry.assert_awaited_once()
        adapter.send.assert_not_awaited()
        content = adapter._send_with_retry.await_args.kwargs.get("content", "")
        assert "сейчас:" in content
        assert "quick replies verified" in content
        assert adapter._send_with_retry.await_args.kwargs.get("reply_to") == "$pending"
