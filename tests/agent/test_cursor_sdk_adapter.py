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


def _install_fake_cursor_sdk(monkeypatch, prompt_impl):
    calls = []

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

    class FakeAgent:
        @staticmethod
        def prompt(prompt, options):
            calls.append((prompt, options, dict(os.environ)))
            return prompt_impl(prompt, options, calls)

    fake = types.SimpleNamespace(
        Agent=FakeAgent,
        AgentOptions=FakeAgentOptions,
        LocalAgentOptions=FakeLocalAgentOptions,
        ModelParameterValue=FakeModelParameterValue,
        ModelSelection=FakeModelSelection,
    )
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake)
    return calls


def test_cursor_sdk_success_returns_openai_shaped_response_with_metadata(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    calls = _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(result="pong", run_id="run-ok", agent_id="agent-ok"),
    )

    response = run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
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


def test_cursor_sdk_failure_after_retry_returns_structured_error_without_choices(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    _install_fake_cursor_sdk(
        monkeypatch,
        lambda prompt, options, calls: _FakeAgentResult(status="error", result="", run_id=f"run-{len(calls)}"),
    )

    response = run_cursor_sdk_chat_completion(
        messages=[{"role": "user", "content": "Reply exactly pong"}],
        api_key="cursor-test-key",
        workspace_root=tmp_path,
        max_retries=1,
        timeout_seconds=5,
    )

    assert response.choices == []
    assert response.cursor_metadata["status"] == "error"
    assert response.cursor_metadata["retry_count"] == 1
    assert "empty result" in response.cursor_metadata["raw_error"]


def test_cursor_sdk_sanitizes_hermes_secrets_from_sdk_process_environment(monkeypatch, tmp_path):
    from agent.cursor_sdk_adapter import run_cursor_sdk_chat_completion

    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("HERMES_SECRET_THING", "should-not-leak")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    calls = _install_fake_cursor_sdk(
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
