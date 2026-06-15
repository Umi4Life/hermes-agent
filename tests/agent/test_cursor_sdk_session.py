"""Tests for Cursor SDK delegated runtime session adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _FakeRun:
    def __init__(self, text: str = "hello from cursor") -> None:
        self._text = text
        self._cancelled = False

    def iter_text(self):
        yield self._text

    def wait(self):
        return SimpleNamespace(status="finished", id="run-1", result=self._text)

    def text(self):
        return self._text

    def supports(self, op: str) -> bool:
        return op == "cancel"

    def cancel(self) -> None:
        self._cancelled = True


class _FakeSdkAgent:
    def __init__(self) -> None:
        self.agent_id = "local-agent-123"
        self._runs: list[_FakeRun] = []

    def send(self, prompt: str) -> _FakeRun:
        run = _FakeRun(f"reply:{prompt}")
        self._runs.append(run)
        return run

    def close(self) -> None:
        return None


class _FakeAgentCM:
    def __init__(self, agent: _FakeSdkAgent) -> None:
        self._agent = agent

    def __enter__(self) -> _FakeSdkAgent:
        return self._agent

    def __exit__(self, *args) -> None:
        return None


@pytest.fixture
def cursor_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    agent = SimpleNamespace(
        api_key="cursor_test_key",
        model="composer-2.5",
        provider="cursor-sdk",
        api_mode="cursor_sdk",
        session_id="sess-1",
        session_cwd=str(tmp_path),
        platform="discord",
        ephemeral_system_prompt="",
        _session_db=MagicMock(),
        _interrupt_requested=False,
    )
    agent._session_db.get_meta.return_value = None
    return agent


def test_run_turn_creates_agent_and_returns_text(cursor_agent):
    fake_sdk = _FakeSdkAgent()

    with patch("cursor_sdk.Agent.create", return_value=_FakeAgentCM(fake_sdk)):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        result = session.run_turn(user_input="hi there")

    assert result.final_text == "reply:hi there"
    assert result.error is None
    assert result.projected_messages == [
        {"role": "assistant", "content": "reply:hi there"}
    ]
    cursor_agent._session_db.set_meta.assert_called()


def test_mcp_entry_matches_codex_helper():
    from hermes_cli.codex_runtime_plugin_migration import _build_hermes_tools_mcp_entry
    from hermes_cli.cursor_sdk_config import build_hermes_tools_mcp_entry

    assert build_hermes_tools_mcp_entry() == _build_hermes_tools_mcp_entry()


def test_fallback_chain_skips_cursor_duplicate():
    fallback = [
        {"provider": "cursor-sdk", "model": "composer-2.5"},
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
    ]
    primary_model = "composer-2.5"
    filtered = [
        entry for entry in fallback
        if not (
            (entry.get("provider") or "").strip().lower()
            in {"cursor-sdk", "cursor_sdk", "cursor"}
            and (entry.get("model") or "").strip() == primary_model
        )
    ]
    assert filtered == [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}]


def test_runtime_provider_resolves_cursor_sdk(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "cursor_test_key")
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="cursor-sdk")
    assert runtime["api_mode"] == "cursor_sdk"
    assert runtime["provider"] == "cursor-sdk"
    assert runtime["api_key"] == "cursor_test_key"


def test_fallback_wrapper_sets_delivery_messages():
    from agent.cursor_sdk_runtime import run_cursor_sdk_turn_with_fallback

    agent = SimpleNamespace(
        api_key="k",
        model="composer-2.5",
        provider="cursor-sdk",
        api_mode="cursor_sdk",
        session_id="sess-1",
        session_cwd=".",
        platform="discord",
        ephemeral_system_prompt="",
        _session_db=MagicMock(),
        _interrupt_requested=False,
        _iters_since_skill=0,
        session_api_calls=0,
        _sync_external_memory_for_turn=MagicMock(),
        _try_activate_fallback=MagicMock(return_value=True),
        _cursor_session=None,
    )
    agent._session_db.get_meta.return_value = None

    turn = SimpleNamespace(
        final_text="",
        projected_messages=[],
        tool_iterations=0,
        interrupted=False,
        error="boom",
        should_retire=True,
        cursor_agent_error=True,
        run_status_error=False,
    )

    def _fake_run_conversation(_agent, user_message, **kwargs):
        return {"final_response": "fallback answer", "messages": kwargs.get("conversation_history", []), "completed": True}

    with patch("agent.transports.cursor_sdk_session.CursorSDKSession") as session_cls:
        session_cls.return_value.run_turn.return_value = turn
        session_cls.return_value.close = MagicMock()
        result = run_cursor_sdk_turn_with_fallback(
            agent,
            user_message="hello",
            original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}],
            effective_task_id="t1",
            run_conversation_fn=_fake_run_conversation,
        )

    assert result["delivery_messages"] == ["⚠️ Cursor: boom", "fallback answer"]
    assert result["final_response"] is None
    assert agent._cursor_fallback_replay is True
