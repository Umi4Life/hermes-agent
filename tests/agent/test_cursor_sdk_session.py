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
        self._send_options: list = []

    def send(self, prompt: str, options=None) -> _FakeRun:
        self._send_options.append(options)
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


@pytest.fixture(autouse=True)
def _implicit_bridge():
    """Default every test to the implicit-bridge path (no owned client).

    Stubbing ``_acquire_client`` to ``(None, 0)`` means ``Agent.create`` is
    called with the same ``(options)`` signature the existing mocks expect and
    no real ``launch_bridge`` subprocess is spawned.  Tests that exercise the
    owned bridge re-patch ``_acquire_client`` themselves; the bridge-manager
    tests drive ``cursor_bridge_manager`` directly and are untouched by this.
    """
    with patch(
        "agent.transports.cursor_sdk_session.CursorSDKSession._acquire_client",
        return_value=(None, 0),
    ):
        yield


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

    with (
        patch("cursor_sdk.Agent.create", return_value=_FakeAgentCM(fake_sdk)),
        # Hermetic: don't inject the ambient SOUL/identity (it varies by host).
        patch(
            "agent.transports.cursor_sdk_session.build_identity_prefix",
            return_value="",
        ),
    ):
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
    """A transient wait failure is retried: the retry recreates the agent and
    sends a fresh run, which succeeds (modeling the blip having cleared)."""

    class _FailingRun(_FakeRun):
        def wait(self):
            raise RuntimeError(
                "Bridge request failed: RemoteProtocolError: peer closed connection"
            )

    fake_sdk = _FakeSdkAgent()
    create_calls = {"n": 0}
    send_calls = {"n": 0}

    def _capture_create(options):
        create_calls["n"] += 1
        return _FakeAgentCM(fake_sdk)

    def flaky_send(self, prompt: str, options=None):
        send_calls["n"] += 1
        # First send's run fails on wait (transient); the retry's send is healthy.
        run = _FailingRun(f"reply:{prompt}") if send_calls["n"] == 1 else _FakeRun(
            f"reply:{prompt}"
        )
        self._runs.append(run)
        self._send_options.append(options)
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
    # The retried send force-expires any stuck prior run.
    assert fake_sdk._send_options[-1] == {"local": {"force": True}}


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

    def flaky_send(self, prompt: str, options=None):
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
    # Connection-refused is bridge-down, not a plain transient.
    assert result.transient_error is False
    assert result.bridge_down is True
    assert "bridge unavailable" in (result.error or "").lower()


def test_run_turn_status_error_retires_and_clears_persisted_agent(cursor_agent):
    fake_sdk = _FakeSdkAgent()
    create_calls = {"n": 0}
    send_calls = {"n": 0}

    def _capture_create(options):
        create_calls["n"] += 1
        return _FakeAgentCM(fake_sdk)

    def alternating_send(self, prompt: str, options=None):
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
        _fallback_chain=[{"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}],
        _fallback_index=1,
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


def test_fallback_wrapper_hints_when_chain_empty():
    from agent.cursor_sdk_runtime import run_cursor_sdk_turn_with_fallback

    agent = SimpleNamespace(
        _try_activate_fallback=MagicMock(return_value=False),
        _fallback_chain=[],
        _cursor_session=None,
        _interrupt_requested=False,
        _iters_since_skill=0,
        session_api_calls=0,
        _sync_external_memory_for_turn=MagicMock(),
        _session_db=MagicMock(),
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

    with patch("agent.transports.cursor_sdk_session.CursorSDKSession") as session_cls:
        session_cls.return_value.run_turn.return_value = turn
        session_cls.return_value.close = MagicMock()
        result = run_cursor_sdk_turn_with_fallback(
            agent,
            user_message="hello",
            original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}],
            effective_task_id="t1",
            run_conversation_fn=MagicMock(),
        )

    assert "fallback_providers" in result["final_response"]
    agent._try_activate_fallback.assert_not_called()


def test_fallback_wrapper_delivery_on_fallback_error():
    from agent.cursor_sdk_runtime import run_cursor_sdk_turn_with_fallback

    agent = SimpleNamespace(
        _try_activate_fallback=MagicMock(return_value=True),
        _fallback_chain=[{"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}],
        _fallback_index=1,
        _cursor_session=None,
        _interrupt_requested=False,
        _iters_since_skill=0,
        session_api_calls=0,
        _sync_external_memory_for_turn=MagicMock(),
        _session_db=MagicMock(),
    )
    agent._session_db.get_meta.return_value = None

    turn = SimpleNamespace(
        final_text="",
        projected_messages=[],
        tool_iterations=0,
        interrupted=False,
        error="peer closed",
        should_retire=True,
        cursor_agent_error=False,
        run_status_error=False,
    )

    def _failing_fallback(_agent, user_message, **kwargs):
        return {"final_response": "", "error": "payment required", "completed": False}

    with patch("agent.transports.cursor_sdk_session.CursorSDKSession") as session_cls:
        session_cls.return_value.run_turn.return_value = turn
        session_cls.return_value.close = MagicMock()
        result = run_cursor_sdk_turn_with_fallback(
            agent,
            user_message="hello",
            original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}],
            effective_task_id="t1",
            run_conversation_fn=_failing_fallback,
        )

    assert result["delivery_messages"] == [
        "⚠️ Cursor: peer closed",
        "⚠️ Fallback failed: payment required",
    ]


def test_cap_channel_context_block_truncates_over_limit():
    from hermes_cli.cursor_sdk_config import cap_channel_context_block

    backfill = "[Recent channel messages]\n" + ("x" * 20000)
    message = f"{backfill}\n\n[New message]\nping"
    capped = cap_channel_context_block(message, 16000)
    assert "[New message]\nping" in capped
    assert len(capped.split("\n\n[New message]\n")[0]) <= 16000
    assert "...[channel context truncated for Cursor]" in capped


def test_cap_channel_context_block_unchanged_under_limit():
    from hermes_cli.cursor_sdk_config import cap_channel_context_block

    message = "[Recent channel messages]\n[Alice] hi\n\n[New message]\nping"
    assert cap_channel_context_block(message, 16000) == message


def test_cap_channel_context_block_zero_disables():
    from hermes_cli.cursor_sdk_config import cap_channel_context_block

    message = "[Recent channel messages]\n" + ("x" * 20000) + "\n\n[New message]\nping"
    assert cap_channel_context_block(message, 0) == message


def test_agent_rotation_on_max_turns_uses_create_not_resume(cursor_agent):
    fake_sdk = _FakeSdkAgent()
    meta_store = {
        "cursor_sdk.agent_id.sess-1": "stored-agent",
        "cursor_sdk.identity_hash.sess-1": "identity-hash-1",
        "cursor_sdk.turn_count.sess-1": "3",
    }
    cursor_agent._session_db.get_meta.side_effect = lambda key: meta_store.get(key)

    with (
        patch("cursor_sdk.Agent.create", return_value=_FakeAgentCM(fake_sdk)) as create_mock,
        patch("cursor_sdk.Agent.resume") as resume_mock,
        patch(
            "agent.transports.cursor_sdk_session.compute_identity_hash",
            return_value="identity-hash-1",
        ),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={
                "max_turns_per_agent": 3,
                "max_agent_age_seconds": 0,
                "timeout_seconds": 180,
                "max_retries": 0,
            },
        ),
    ):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        session.run_turn(user_input="ping")

    resume_mock.assert_not_called()
    create_mock.assert_called_once()


def test_agent_rotation_on_max_age_uses_create_not_resume(cursor_agent, monkeypatch):
    fake_sdk = _FakeSdkAgent()
    meta_store = {
        "cursor_sdk.agent_id.sess-1": "stored-agent",
        "cursor_sdk.identity_hash.sess-1": "identity-hash-1",
        "cursor_sdk.turn_count.sess-1": "1",
        "cursor_sdk.agent_created_at.sess-1": "1000",
    }
    cursor_agent._session_db.get_meta.side_effect = lambda key: meta_store.get(key)
    monkeypatch.setattr("agent.transports.cursor_sdk_session.time.time", lambda: 5000)

    with (
        patch("cursor_sdk.Agent.create", return_value=_FakeAgentCM(fake_sdk)) as create_mock,
        patch("cursor_sdk.Agent.resume") as resume_mock,
        patch(
            "agent.transports.cursor_sdk_session.compute_identity_hash",
            return_value="identity-hash-1",
        ),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={
                "max_turns_per_agent": 0,
                "max_agent_age_seconds": 3600,
                "timeout_seconds": 180,
                "max_retries": 0,
            },
        ),
    ):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        session.run_turn(user_input="ping")

    resume_mock.assert_not_called()
    create_mock.assert_called_once()


def test_successful_turn_increments_turn_count(cursor_agent):
    fake_sdk = _FakeSdkAgent()
    meta_store: dict[str, str] = {}

    def _set_meta(key, value):
        meta_store[key] = value

    cursor_agent._session_db.get_meta.side_effect = lambda key: meta_store.get(key)
    cursor_agent._session_db.set_meta.side_effect = _set_meta

    with (
        patch("cursor_sdk.Agent.create", return_value=_FakeAgentCM(fake_sdk)),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={"timeout_seconds": 180, "max_retries": 0},
        ),
    ):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        session.run_turn(user_input="ping")

    assert meta_store.get("cursor_sdk.turn_count.sess-1") == "1"


def test_failed_turn_does_not_increment_turn_count(cursor_agent):
    class _ErrorRun(_FakeRun):
        def wait(self):
            raise RuntimeError("peer closed")

    fake_sdk = _FakeSdkAgent()

    def _send(prompt: str, options=None):
        run = _ErrorRun(f"reply:{prompt}")
        fake_sdk._runs.append(run)
        return run

    fake_sdk.send = _send
    meta_store: dict[str, str] = {}

    def _set_meta(key, value):
        meta_store[key] = value

    cursor_agent._session_db.get_meta.side_effect = lambda key: meta_store.get(key)
    cursor_agent._session_db.set_meta.side_effect = _set_meta

    with (
        patch("cursor_sdk.Agent.create", return_value=_FakeAgentCM(fake_sdk)),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={"timeout_seconds": 180, "max_retries": 0},
        ),
    ):
        from agent.transports.cursor_sdk_session import CursorSDKSession

        session = CursorSDKSession(cursor_agent)
        result = session.run_turn(user_input="ping")

    assert result.error
    assert meta_store.get("cursor_sdk.turn_count.sess-1") in (None, "0")


# ── Exception classification ────────────────────────────────────────────────


def test_classify_bridge_down_transient_and_retry_after():
    from agent.transports.cursor_sdk_session import _classify_cursor_exception

    bridge = _classify_cursor_exception(
        RuntimeError("ConnectError: [Errno 111] Connection refused")
    )
    assert bridge.is_bridge_down is True
    assert bridge.is_transient is False
    assert bridge.fail_fast is False

    transient = _classify_cursor_exception(
        RuntimeError("RemoteProtocolError: peer closed connection")
    )
    assert transient.is_transient is True
    assert transient.is_bridge_down is False

    class _Retryable(Exception):
        is_retryable = True
        retry_after = "2.5"

    retryable = _classify_cursor_exception(_Retryable("slow down"))
    assert retryable.is_transient is True
    assert retryable.retry_after == 2.5


def test_classify_typed_fail_fast():
    from agent.transports.cursor_sdk_session import _classify_cursor_exception

    try:
        from cursor_sdk.errors import AuthenticationError
    except Exception:
        pytest.skip("cursor_sdk not installed")
    try:
        exc = AuthenticationError("invalid api key")
    except Exception:
        pytest.skip("AuthenticationError constructor differs")

    info = _classify_cursor_exception(exc)
    assert info.fail_fast is True
    assert info.is_cursor is True
    assert info.is_transient is False


def test_fail_fast_error_is_not_retried(cursor_agent):
    from agent.transports import cursor_sdk_session as mod

    fake_sdk = _FakeSdkAgent()
    create_calls = {"n": 0}

    def _capture_create(options):
        create_calls["n"] += 1
        return _FakeAgentCM(fake_sdk)

    class _Boom(_FakeRun):
        def wait(self):
            raise RuntimeError("permanent: malformed request")

    def _send(self, prompt, options=None):
        run = _Boom(f"reply:{prompt}")
        self._runs.append(run)
        return run

    real_classify = mod._classify_cursor_exception

    def _classify(exc):
        info = real_classify(exc)
        if "malformed" in str(exc):
            info.fail_fast = True
            info.is_cursor = True
            info.is_transient = False
        return info

    with (
        patch("cursor_sdk.Agent.create", side_effect=_capture_create),
        patch.object(_FakeSdkAgent, "send", _send),
        patch.object(mod, "_classify_cursor_exception", _classify),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={
                "timeout_seconds": 180,
                "max_retries": 2,
                "retry_backoff_seconds": 0,
                "hermes_tools_mcp": False,
                "inject_identity": False,
            },
        ),
    ):
        session = mod.CursorSDKSession(cursor_agent)
        result = session.run_turn(user_input="ping")

    assert result.fail_fast is True
    assert result.error
    assert create_calls["n"] == 1  # no retry on a permanent error


# ── Owned bridge: relaunch + resume force ───────────────────────────────────


def test_bridge_down_relaunches_then_succeeds(cursor_agent):
    from agent.transports.cursor_sdk_session import CursorSDKSession

    fake_sdk = _FakeSdkAgent()
    create_calls = {"n": 0}
    send_calls = {"n": 0}
    relaunch_calls = {"n": 0}

    class _BridgeDownRun(_FakeRun):
        def wait(self):
            raise RuntimeError(
                "Bridge request failed: ConnectError: [Errno 111] Connection refused"
            )

    def _capture_create(options, client=None):
        create_calls["n"] += 1
        return _FakeAgentCM(fake_sdk)

    def _send(self, prompt, options=None):
        send_calls["n"] += 1
        run = (
            _BridgeDownRun(f"reply:{prompt}")
            if send_calls["n"] == 1
            else _FakeRun(f"reply:{prompt}")
        )
        self._runs.append(run)
        self._send_options.append(options)
        return run

    def _relaunch(cwd, gen, **kwargs):
        relaunch_calls["n"] += 1
        return ("client-gen2", 2)

    with (
        patch("cursor_sdk.Agent.create", side_effect=_capture_create),
        patch.object(_FakeSdkAgent, "send", _send),
        # Owned bridge present: hand back a fake client + generation.
        patch.object(CursorSDKSession, "_acquire_client", return_value=("client-gen1", 1)),
        patch(
            "agent.transports.cursor_sdk_session.cursor_bridge_manager.relaunch",
            side_effect=_relaunch,
        ),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={
                "timeout_seconds": 180,
                "max_retries": 2,
                "retry_backoff_seconds": 0,
                "own_bridge": True,
                "hermes_tools_mcp": False,
                "inject_identity": False,
            },
        ),
    ):
        session = CursorSDKSession(cursor_agent)
        result = session.run_turn(user_input="ping")

    assert result.error is None
    assert result.final_text == "reply:ping"
    assert relaunch_calls["n"] == 1
    assert create_calls["n"] == 2
    # Owned client is threaded into Agent.create.
    assert fake_sdk._send_options[-1] == {"local": {"force": True}}


def test_resume_forces_stuck_run_expiry(cursor_agent):
    from agent.transports.cursor_sdk_session import CursorSDKSession

    fake_sdk = _FakeSdkAgent()
    meta = {
        "cursor_sdk.agent_id.sess-1": "stored-agent",
        "cursor_sdk.identity_hash.sess-1": "h1",
    }
    cursor_agent._session_db.get_meta.side_effect = lambda k: meta.get(k)
    captured = {}

    def _resume(agent_id, options, client=None):
        captured["agent_id"] = agent_id
        return _FakeAgentCM(fake_sdk)

    with (
        patch("cursor_sdk.Agent.resume", side_effect=_resume),
        patch("cursor_sdk.Agent.create") as create_mock,
        patch(
            "agent.transports.cursor_sdk_session.compute_identity_hash",
            return_value="h1",
        ),
        patch(
            "agent.transports.cursor_sdk_session.get_cursor_sdk_settings",
            return_value={
                "timeout_seconds": 180,
                "max_retries": 0,
                "max_turns_per_agent": 0,
                "max_agent_age_seconds": 0,
                "hermes_tools_mcp": False,
                "inject_identity": False,
            },
        ),
    ):
        session = CursorSDKSession(cursor_agent)
        session.run_turn(user_input="ping")

    create_mock.assert_not_called()
    assert captured["agent_id"] == "stored-agent"
    assert fake_sdk._send_options[-1] == {"local": {"force": True}}


# ── Bridge manager pool / generation guard ──────────────────────────────────


def test_bridge_manager_get_client_falls_back_on_launch_failure(monkeypatch):
    from agent.transports import cursor_bridge_manager as mgr

    monkeypatch.setattr(mgr, "_bridges", {})

    def _boom(cwd, max_retries):
        raise RuntimeError("no bridge available")

    monkeypatch.setattr(mgr, "_launch", _boom)
    client, gen = mgr.get_client("/tmp/ws-a")
    assert client is None
    assert gen == 0


def test_bridge_manager_relaunch_without_owned_bridge_is_noop(monkeypatch):
    from agent.transports import cursor_bridge_manager as mgr

    monkeypatch.setattr(mgr, "_bridges", {})
    client, gen = mgr.relaunch("/tmp/ws-b", 1)
    assert client is None
    assert gen == 0


def test_bridge_manager_relaunch_generation_guard(monkeypatch):
    from agent.transports import cursor_bridge_manager as mgr

    class _Client:
        def with_options(self, **kwargs):
            return self

    launches = {"n": 0}

    def _fake_launch(cwd, max_retries):
        launches["n"] += 1
        cm = SimpleNamespace(__exit__=lambda *a: None)
        return cm, _Client()

    monkeypatch.setattr(mgr, "_bridges", {})
    monkeypatch.setattr(mgr, "_launch", _fake_launch)

    _, gen1 = mgr.get_client("/ws")
    assert gen1 == 1 and launches["n"] == 1

    # Stale observed generation → another thread already relaunched → no-op.
    _, gen_stale = mgr.relaunch("/ws", 0)
    assert gen_stale == 1 and launches["n"] == 1

    # Current observed generation → real relaunch, generation bumps.
    _, gen_new = mgr.relaunch("/ws", 1)
    assert gen_new == 2 and launches["n"] == 2

    mgr.shutdown_all()
    assert mgr._bridges == {}
