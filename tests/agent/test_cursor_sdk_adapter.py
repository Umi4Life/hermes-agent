import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeAgentResult:
    def __init__(self, *, status="finished", result="pong", duration_ms=123, run_id="run-1", agent_id="agent-1"):
        self.status = status
        self.result = result
        self.duration_ms = duration_ms
        self.id = run_id
        self.run_id = run_id
        self.agent_id = agent_id


def _install_fake_cursor_sdk(monkeypatch, prompt_impl, *, streaming_impl=None):
    calls = []
    stream_calls = []

    class FakeModelParameterValue:
        def __init__(self, id, value):
            self.id = id
            self.value = value

    class FakeModelSelection:
        def __init__(self, id, params):
            self.id = id
            self.params = params

    class FakeLocalAgentOptions:
        def __init__(self, cwd):
            self.cwd = cwd

    class FakeAgentOptions:
        def __init__(self, api_key, model, local):
            self.api_key = api_key
            self.model = model
            self.local = local

    class FakeRun:
        def __init__(self, result):
            self._result = result

        def iter_text(self):
            yield "po"
            yield "ng"

        def supports(self, name):
            return name == "cancel"

        def cancel(self):
            return None

        def wait(self):
            return self._result

    class FakeStreamingAgent:
        def __init__(self, **kwargs):
            stream_calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def send(self, prompt):
            stream_calls.append({"prompt": prompt})
            if streaming_impl:
                return streaming_impl(prompt, stream_calls)
            return FakeRun(_FakeAgentResult(result="pong"))

    class FakeAgent:
        @staticmethod
        def prompt(prompt, options):
            calls.append((prompt, options, dict(os.environ)))
            return prompt_impl(prompt, options, calls)

        @staticmethod
        def create(**kwargs):
            return FakeStreamingAgent(**kwargs)

    fake = types.SimpleNamespace(
        Agent=FakeAgent,
        AgentOptions=FakeAgentOptions,
        LocalAgentOptions=FakeLocalAgentOptions,
        ModelParameterValue=FakeModelParameterValue,
        ModelSelection=FakeModelSelection,
    )
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake)
    return calls, stream_calls


def test_cursor_sdk_success_returns_openai_shaped_response_with_metadata(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    calls = _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="pong", run_id="run-ok", agent_id="agent-ok"),
    )[0]

    response = run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
        cursor_mode="workspace",
        session_id="session-1",
        timeout_seconds=5,
    )

    assert response.choices[0].message.content == "pong"
    assert response.choices[0].finish_reason == "stop"
    assert response.model == "composer-2.5"
    assert response.cursor_metadata["status"] == "finished"
    assert response.cursor_metadata["model_id"] == "composer-2.5"
    assert response.cursor_metadata["model_params"] == {"fast": "false"}
    assert response.cursor_metadata["agent_id"] == "agent-ok"
    assert response.cursor_metadata["run_id"] == "run-ok"
    assert response.cursor_metadata["retry_count"] == 0
    assert response.cursor_metadata["latency_ms"] >= 0
    assert calls[0][1].model.id == "composer-2.5"
    assert [(p.id, p.value) for p in calls[0][1].model.params] == [("fast", "false")]
    assert Path(calls[0][1].local.cwd).is_relative_to(tmp_path)


def test_cursor_sdk_empty_result_is_failure_and_retried_once(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    def prompt_impl(prompt, options, calls):
        if len(calls) == 1:
            return _FakeAgentResult(status="finished", result="", run_id="run-empty")
        return _FakeAgentResult(status="finished", result="pong", run_id="run-retry")

    _install_fake_cursor_sdk(monkeypatch, prompt_impl)

    response = run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
        max_retries=1,
        timeout_seconds=5,
    )

    assert response.choices[0].message.content == "pong"
    assert response.cursor_metadata["retry_count"] == 1
    assert response.cursor_metadata["run_id"] == "run-retry"


def test_cursor_sdk_failure_after_retry_returns_structured_error_without_choices(monkeypatch, tmp_path, caplog):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(status="error", result="", run_id=f"run-{len(calls)}"),
    )

    with caplog.at_level("WARNING", logger="agent.cursor_sdk_adapter"):
        response = run_cursor_sdk_chat_completion(
            messages=[{"role": "user", "content": "Reply exactly pong"}],
            api_key="cursor-test-key",
            workspace_root=tmp_path,
            max_retries=1,
            timeout_seconds=5,
        )

    assert response.choices == []
    assert response.cursor_metadata["status"] == "error"
    assert response.cursor_metadata["sdk_status"] == "error"
    assert response.cursor_metadata["retry_count"] == 1
    assert response.cursor_metadata["error"] == {
        "type": "CursorSDKCallError",
        "message": "Cursor SDK returned status='error'",
    }
    assert "status=error" in caplog.text
    assert "Cursor SDK call failed before fallback" in caplog.text


def test_cursor_sdk_empty_final_result_logs_empty_result(monkeypatch, tmp_path, caplog):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(status="finished", result="", run_id=f"run-{len(calls)}"),
    )

    with caplog.at_level("WARNING", logger="agent.cursor_sdk_adapter"):
        response = run_cursor_sdk_chat_completion(
            messages=[{"role": "user", "content": "Reply exactly pong"}],
            api_key="cursor-test-key",
            workspace_root=tmp_path,
            max_retries=0,
            timeout_seconds=5,
        )

    assert response.choices == []
    assert response.cursor_metadata["status"] == "empty_result"
    assert response.cursor_metadata["sdk_status"] == "finished"
    assert response.cursor_metadata["raw_error"] == "Cursor SDK returned empty result"
    assert response.cursor_metadata["error"] == {
        "type": "CursorSDKCallError",
        "message": "Cursor SDK returned empty result",
    }
    assert "status=empty_result" in caplog.text


def test_cursor_sdk_sanitizes_hermes_secrets_from_sdk_process_environment(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    monkeypatch.setenv("OPENAI_API_KEY", "sk-sho...leak")
    monkeypatch.setenv("HERMES_SECRET_THING", "should-not-leak")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    calls, _ = _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="pong"),
    )

    run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
        timeout_seconds=5,
    )

    sdk_env = calls[0][2]
    assert sdk_env.get("CURSOR_API_KEY") == "cursor-test-key"
    assert "OPENAI_API_KEY" not in sdk_env
    assert "HERMES_SECRET_THING" not in sdk_env


def test_cursor_sdk_runtime_provider_resolves_selectable_non_default(monkeypatch):
    from hermes_cli import runtime_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-test-key")
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: None)

    runtime = resolve_runtime_provider(requested="cursor-sdk", target_model="composer-2.5")

    assert runtime["provider"] == "cursor-sdk"
    assert runtime["api_mode"] == "cursor_sdk"
    assert runtime["api_key"] == "cursor-test-key"
    assert runtime["base_url"] == "cursor-sdk://local"
    assert runtime["requested_provider"] == "cursor-sdk"
    assert runtime["request_overrides"]["cursor_model_id"] == "composer-2.5"
    assert runtime["request_overrides"]["cursor_model_params"] == {"fast": "false"}


def test_gateway_runtime_kwargs_preserve_cursor_request_overrides(monkeypatch):
    import gateway.run as gateway_run
    from hermes_cli import runtime_provider

    expected_overrides = {
        "cursor_model_id": "composer-2.5",
        "cursor_model_params": {"fast": "false"},
        "cursor_timeout_seconds": 90.0,
        "cursor_max_retries": 1,
    }

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda: {
            "api_key": "cursor-test-key",
            "base_url": "cursor-sdk://local",
            "provider": "cursor-sdk",
            "api_mode": "cursor_sdk",
            "request_overrides": expected_overrides,
        },
    )
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {})

    runtime_kwargs = gateway_run._resolve_runtime_agent_kwargs()

    assert runtime_kwargs["provider"] == "cursor-sdk"
    assert runtime_kwargs["api_mode"] == "cursor_sdk"
    assert runtime_kwargs["request_overrides"] == expected_overrides


def test_gateway_turn_config_merges_cursor_request_overrides_without_fast_mode():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._service_tier = None
    runtime_kwargs = {
        "api_key": "cursor-test-key",
        "base_url": "cursor-sdk://local",
        "provider": "cursor-sdk",
        "api_mode": "cursor_sdk",
        "request_overrides": {
            "cursor_model_id": "composer-2.5",
            "cursor_model_params": {"fast": "false"},
        },
    }

    route = runner._resolve_turn_agent_config("Reply exactly pong", "composer-2.5", runtime_kwargs)

    assert route["runtime"]["provider"] == "cursor-sdk"
    assert route["runtime"]["api_mode"] == "cursor_sdk"
    assert route["request_overrides"] == {
        "cursor_model_id": "composer-2.5",
        "cursor_model_params": {"fast": "false"},
    }


def test_agent_init_admits_cursor_sdk_api_mode_without_downgrading_to_chat(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-test-key")
    agent = AIAgent(
        model="composer-2.5",
        provider="cursor-sdk",
        base_url="cursor-sdk://local",
        api_mode="cursor_sdk",
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        quiet_mode=True,
    )

    assert agent.provider == "cursor-sdk"
    assert agent.api_mode == "cursor_sdk"
    assert agent.api_key == "cursor-test-key"


def test_cursor_sdk_streaming_dispatch_uses_adapter_not_openai_client(monkeypatch, tmp_path):
    from agent import cursor_sdk_adapter
    from agent.chat_completion_helpers import interruptible_streaming_api_call

    calls = []

    def fake_cursor_call(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="pong", tool_calls=None, reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            model=kwargs["model_id"],
            cursor_metadata={"status": "finished"},
        )

    monkeypatch.setattr(cursor_sdk_adapter, "run_cursor_sdk_chat_completion", fake_cursor_call)

    agent = SimpleNamespace(
        api_mode="cursor_sdk",
        provider="cursor-sdk",
        model="composer-2.5",
        api_key="cursor-test-key",
        base_url="cursor-sdk://local",
        session_id="20260612_182716_3fd6e7ae",
        _interrupt_requested=False,
        stream_delta_callback=None,
        reasoning_callback=None,
        _has_stream_consumers=lambda: False,
        _fire_stream_delta=lambda text: None,
        _fire_reasoning_delta=lambda text: None,
        _fire_tool_gen_started=lambda name: None,
        _touch_activity=lambda message: None,
        _buffer_status=lambda message: None,
        _current_streamed_assistant_text="",
        _is_provider_stream_parse_error=lambda exc: False,
        _create_request_openai_client=lambda *a, **k: pytest.fail(
            "cursor-sdk streaming dispatch must not build an OpenAI client"
        ),
        _abort_request_openai_client=lambda *a, **k: None,
        _close_request_openai_client=lambda *a, **k: None,
        _replace_primary_openai_client=lambda *a, **k: None,
    )

    response = interruptible_streaming_api_call(
        agent,
        {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "Reply exactly pong"}],
            "cursor_model_id": "composer-2.5",
            "cursor_model_params": {"fast": "false"},
            "cursor_workspace_root": str(tmp_path),
            "cursor_timeout_seconds": 90.0,
            "cursor_max_retries": 1,
            "session_id": "20260612_182716_3fd6e7ae",
        },
    )

    assert response.choices[0].message.content == "pong"
    assert calls == [
        {
            "messages": [{"role": "user", "content": "Reply exactly pong"}],
            "api_key": "cursor-test-key",
            "model_id": "composer-2.5",
            "model_params": {"fast": "false"},
            "workspace_root": str(tmp_path),
            "session_id": "20260612_182716_3fd6e7ae",
            "timeout_seconds": 90.0,
            "max_retries": 1,
            "cursor_mode": "chat",
            "prompt_mode": "slim",
            "on_text_delta": None,
            "interrupt_check": pytest.ANY,
        }
    ]


def test_cursor_sdk_default_timeout_is_180():
    from agent.cursor_sdk_adapter import DEFAULT_CURSOR_TIMEOUT_SECONDS

    assert DEFAULT_CURSOR_TIMEOUT_SECONDS == 180.0


def test_cursor_sdk_slim_prompt_omits_system_and_tools(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    calls, _ = _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="pong"),
    )

    run_cursor_sdk_chat_completion(
        messages=[
            {"role": "system", "content": "You are Hermes"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "middle"},
            {"role": "tool", "content": '{"ok": true}'},
            {"role": "user", "content": "Reply exactly pong"},
        ],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
        prompt_mode="slim",
        timeout_seconds=5,
    )

    prompt = calls[0][0]
    assert "[HERMES CHAT]" in prompt
    assert "You are Hermes" not in prompt
    assert '{"ok": true}' not in prompt
    assert "Reply exactly pong" in prompt


def test_cursor_sdk_chat_mode_uses_terminal_cwd(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    chat_cwd = tmp_path / "chat-cwd"
    chat_cwd.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(chat_cwd))
    calls, _ = _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="pong"),
    )

    run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path / "workspace-root",
        session_id="session-abc",
        cursor_mode="chat",
        timeout_seconds=5,
    )

    assert Path(calls[0][1].local.cwd) == chat_cwd


def test_cursor_sdk_call_shape_logged(monkeypatch, tmp_path, caplog):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="pong"),
    )

    with caplog.at_level("INFO", logger="agent.cursor_sdk_adapter"):
        run_cursor_sdk_chat_completion(
            messages=[{"role": "user", "content": "Reply exactly pong"}],
            api_key="cursor-test-key",
            workspace_root=tmp_path,
            timeout_seconds=5,
        )

    assert "Cursor SDK call:" in caplog.text
    assert "cursor_mode=chat" in caplog.text
    assert "prompt_mode=slim" in caplog.text
    assert "local_workspace_enabled=False" in caplog.text


def test_cursor_sdk_workspace_mode_uses_session_subdirectory(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    calls, _ = _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="pong"),
    )

    run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
        session_id="session-abc",
        cursor_mode="workspace",
        timeout_seconds=5,
    )

    workspace_path = Path(calls[0][1].local.cwd)
    assert workspace_path.is_relative_to(tmp_path)
    assert workspace_path.name == "session-abc"


def test_cursor_sdk_coding_mode_alias_uses_workspace(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    calls, _ = _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="pong"),
    )

    run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
        session_id="coding-session",
        cursor_mode="coding",
        timeout_seconds=5,
    )

    assert Path(calls[0][1].local.cwd).name == "coding-session"


def test_cursor_sdk_streaming_uses_agent_create_and_deltas(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    _, stream_calls = _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="unused"),
    )
    deltas = []

    response = run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
        timeout_seconds=5,
        on_text_delta=deltas.append,
    )

    assert response.choices[0].message.content == "pong"
    assert deltas == ["po", "ng"]
    assert stream_calls
    assert stream_calls[0]["api_key"] == "cursor-test-key"


def test_switch_model_to_cursor_sdk_does_not_build_openai_client():
    from agent.agent_runtime_helpers import switch_model

    class Cache(dict):
        def clear(self):
            self["cleared"] = True

    agent = SimpleNamespace(
        model="gpt-5.5",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        api_key="old-key",
        client=object(),
        _anthropic_client=None,
        _anthropic_api_key=None,
        _anthropic_base_url=None,
        _is_anthropic_oauth=False,
        _config_context_length=None,
        _client_kwargs={},
        _transport_cache=Cache(),
        _create_openai_client=lambda *a, **k: pytest.fail("cursor-sdk switch must not build OpenAI client"),
        _anthropic_prompt_cache_policy=lambda **kwargs: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
        context_compressor=None,
        _cached_system_prompt="cached",
    )

    switch_model(
        agent,
        new_model="composer-2.5",
        new_provider="cursor-sdk",
        api_key="cursor-test-key",
        base_url="cursor-sdk://local",
        api_mode="cursor_sdk",
    )

    assert agent.model == "composer-2.5"
    assert agent.provider == "cursor-sdk"
    assert agent.api_mode == "cursor_sdk"
    assert agent.base_url == "cursor-sdk://local"
    assert agent.client is None
    assert agent._client_kwargs == {}
    assert agent._transport_cache["cleared"] is True
