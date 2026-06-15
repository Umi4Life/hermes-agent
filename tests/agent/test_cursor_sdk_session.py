"""Tests for Cursor SDK delegated runtime session adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _FakeRun:
    def __init__(self, text: str = "hello from cursor", *, status: str = "finished") -> None:
        self._text = text
        self._status = status
        self._cancelled = False
        self._wait_calls = 0
        self._messages_called = False

    def iter_text(self):
        yield self._text

    def messages(self):
        self._messages_called = True
        yield {"type": "assistant", "message": {"content": [{"type": "text", "text": self._text}]}}

    def wait(self):
        self._wait_calls += 1
        return SimpleNamespace(status=self._status, id="run-1", result=self._text)

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
    assert fake_sdk._runs[0]._messages_called is False
    assert fake_sdk._runs[0]._wait_calls == 1
    assert result.projected_messages == [
        {"role": "assistant", "content": "reply:hi there"}
    ]
    cursor_agent._session_db.set_meta.assert_called()


def test_run_turn_prepends_identity_on_first_send_not_agent_create(cursor_agent):
    fake_sdk = _FakeSdkAgent()
    captured_options: list = []

    def _capture_create(options):
        captured_options.append(options)
        return _FakeAgentCM(fake_sdk)

    with (
        patch("cursor_sdk.Agent.create", side_effect=_capture_create),
        patch(
            "agent.transports.cursor_sdk_session.build_identity_prefix",
            return_value="You are Sky Feather.",
        ),
        patch(
            "agent.transports.cursor_sdk_session.compute_identity_hash",
            return_value="identity-hash-1",
        ),
    ):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        result = session.run_turn(user_input="ping")

    assert len(captured_options) == 1
    opts = captured_options[0]
    assert not isinstance(opts, dict) or "instructions" not in opts
    assert result.final_text.startswith("reply:You are Sky Feather.")
    assert "ping" in result.final_text


def test_run_turn_create_uses_agent_options_for_mcp(cursor_agent):
    fake_sdk = _FakeSdkAgent()
    captured_options: list = []

    def _capture_create(options):
        captured_options.append(options)
        return _FakeAgentCM(fake_sdk)

    with (
        patch("cursor_sdk.Agent.create", side_effect=_capture_create),
        patch(
            "agent.transports.cursor_sdk_session.build_cursor_mcp_servers",
            return_value={"hermes-tools": {"command": "python", "args": ["-m", "mcp"]}},
        ),
    ):
        from cursor_sdk import AgentOptions
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        session.run_turn(user_input="ping")

    assert len(captured_options) == 1
    opts = captured_options[0]
    assert isinstance(opts, AgentOptions)
    assert opts.mcp_servers == {
        "hermes-tools": {"command": "python", "args": ["-m", "mcp"]}
    }


def test_run_turn_retries_transient_wait_failure(cursor_agent):
    class _FlakyRun(_FakeRun):
        def __init__(self, prompt: str) -> None:
            super().__init__(f"reply:{prompt}")
            self._attempt = 0

        def wait(self):
            self._attempt += 1
            if self._attempt == 1:
                raise RuntimeError(
                    "Bridge request failed: RemoteProtocolError: peer closed connection"
                )
            return super().wait()

    fake_sdk = _FakeSdkAgent()
    create_calls = {"n": 0}

    def _capture_create(options):
        create_calls["n"] += 1
        return _FakeAgentCM(fake_sdk)

    def flaky_send(self, prompt: str):
        run = _FlakyRun(prompt)
        self._runs.append(run)
        return run

    with (
        patch("cursor_sdk.Agent.create", side_effect=_capture_create),
        patch.object(_FakeSdkAgent, "send", flaky_send),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={
                "runtime": "delegated",
                "timeout_seconds": 180,
                "max_retries": 1,
                "retry_backoff_seconds": 0,
                "fast": False,
                "hermes_tools_mcp": False,
                "inject_identity": False,
            },
        ),
    ):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        result = session.run_turn(user_input="ping")

    assert result.error is None
    assert result.final_text == "reply:ping"
    assert create_calls["n"] == 2


def test_run_turn_connection_refused_on_retry_does_not_retry_again(cursor_agent):
    class _FlakyRun(_FakeRun):
        def wait(self):
            raise RuntimeError(
                "Bridge request failed: RemoteProtocolError: peer closed connection"
            )

    fake_sdk = _FakeSdkAgent()
    create_calls = {"n": 0}

    def _capture_create(options):
        create_calls["n"] += 1
        if create_calls["n"] == 1:
            return _FakeAgentCM(fake_sdk)
        raise RuntimeError("Bridge request failed: ConnectError: [Errno 111] Connection refused")

    def flaky_send(self, prompt: str):
        run = _FlakyRun(f"reply:{prompt}")
        self._runs.append(run)
        return run

    with (
        patch("cursor_sdk.Agent.create", side_effect=_capture_create),
        patch.object(_FakeSdkAgent, "send", flaky_send),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={
                "runtime": "delegated",
                "timeout_seconds": 180,
                "max_retries": 1,
                "retry_backoff_seconds": 0,
                "fast": False,
                "hermes_tools_mcp": False,
                "inject_identity": False,
            },
        ),
    ):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        result = session.run_turn(user_input="ping")

    assert create_calls["n"] == 2
    assert result.transient_error is False
    assert "bridge unavailable" in (result.error or "").lower()


def test_run_turn_status_error_retires_and_clears_persisted_agent(cursor_agent):
    fake_sdk = _FakeSdkAgent()
    create_calls = {"n": 0}
    send_calls = {"n": 0}

    def _capture_create(options):
        create_calls["n"] += 1
        return _FakeAgentCM(fake_sdk)

    def alternating_send(self, prompt: str):
        send_calls["n"] += 1
        if send_calls["n"] == 1:
            return _FakeRun(f"reply:{prompt}", status="error")
        return _FakeRun(f"reply:{prompt}")

    with (
        patch("cursor_sdk.Agent.create", side_effect=_capture_create),
        patch.object(_FakeSdkAgent, "send", alternating_send),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={
                "runtime": "delegated",
                "timeout_seconds": 180,
                "max_retries": 0,
                "retry_backoff_seconds": 0,
                "fast": False,
                "hermes_tools_mcp": False,
                "inject_identity": False,
            },
        ),
    ):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        failed = session.run_turn(user_input="heavy turn")
        ok = session.run_turn(user_input="k")

    assert failed.run_status_error is True
    assert failed.should_retire is True
    assert failed.error is not None
    assert ok.error is None
    assert ok.final_text == "reply:k"
    assert create_calls["n"] == 2
    cleared_calls = [
        c for c in cursor_agent._session_db.set_meta.call_args_list if c.args[1] == ""
    ]
    assert cleared_calls


def test_model_selection_defaults_to_standard(cursor_agent):
    from hermes_cli.cursor_sdk_config import build_cursor_model_selection

    sel = build_cursor_model_selection(cursor_agent)
    assert sel == {
        "id": "composer-2.5",
        "params": [{"id": "fast", "value": "false"}],
    }


def test_model_selection_fast_when_configured(cursor_agent):
    from hermes_cli.cursor_sdk_config import build_cursor_model_selection

    sel = build_cursor_model_selection(
        cursor_agent,
        {"fast": True, "runtime": "delegated"},
    )
    assert sel["params"] == [{"id": "fast", "value": "true"}]


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
