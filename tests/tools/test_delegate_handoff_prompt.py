from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _TIER2_HANDOFF_PACKET_TEMPLATE,
    _build_child_system_prompt,
)


def test_child_prompt_includes_tier2_handoff_packet_guidance():
    prompt = _build_child_system_prompt(
        "Inspect the failing worker handoff path.",
        "repo=/opt/hermes-agent",
        workspace_path="/opt/hermes-agent",
    )

    assert "Tier-2 handoff rule" in prompt
    assert _TIER2_HANDOFF_PACKET_TEMPLATE in prompt
    assert "current_artifacts" in prompt
    assert "needs_user" in prompt
    assert "last_verified" in prompt


def test_delegate_task_context_schema_mentions_handoff_packet():
    context_description = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["context"]["description"]

    assert "Tier-2" in context_description
    assert "handoff packet" in context_description
    assert "next_action" in context_description
    assert "last_verified" in context_description
