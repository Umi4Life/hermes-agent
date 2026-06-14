"""Cursor SDK adapter for Hermes model calls.

This module wraps the official Python ``cursor-sdk`` package and returns an
OpenAI Chat Completions-shaped response so the existing Hermes conversation loop
can consume Cursor agent output without a Node bridge.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

DEFAULT_CURSOR_MODEL_ID = "composer-2.5"
DEFAULT_CURSOR_MODEL_PARAMS = {"fast": "false"}
DEFAULT_CURSOR_TIMEOUT_SECONDS = 180.0
DEFAULT_CURSOR_MAX_RETRIES = 1
DEFAULT_CURSOR_WORKSPACE_ROOT = Path("/srv/hermes-cursor/workspaces")
DEFAULT_CURSOR_MODE = "chat"
DEFAULT_CURSOR_PROMPT_MODE = "slim"
_HERMES_CHAT_PREFIX = (
    "[HERMES CHAT] Reply directly. Do not browse files or run tools unless "
    "the user explicitly asks."
)

logger = logging.getLogger(__name__)

_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
_ALLOWED_ENV_NAMES = {
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "LOGNAME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "NO_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
}


class CursorSDKCallError(RuntimeError):
    """Raised internally for Cursor SDK failures that should trigger retry."""


def _cursor_failure_message(metadata: dict[str, Any]) -> str:
    error = metadata.get("error")
    if isinstance(error, dict):
        err_type = error.get("type") or metadata.get("error_type") or "CursorSDKError"
        err_msg = error.get("message") or metadata.get("error_message") or metadata.get("raw_error") or "unknown"
        return f"{err_type}: {err_msg}"
    return str(metadata.get("raw_error") or metadata.get("error_message") or "unknown")


def _log_cursor_sdk_failure(metadata: dict[str, Any]) -> None:
    """Log structured Cursor SDK failure metadata before caller fallback can run."""
    logger.warning(
        "Cursor SDK call failed before fallback: status=%s sdk_status=%s error=%s "
        "raw_error=%s timeout_seconds=%s latency_ms=%s retry_count=%s model=%s "
        "run_id=%s agent_id=%s workspace=%s",
        metadata.get("status"),
        metadata.get("sdk_status"),
        metadata.get("error"),
        metadata.get("raw_error"),
        metadata.get("timeout_seconds"),
        metadata.get("latency_ms"),
        metadata.get("retry_count"),
        metadata.get("model_id"),
        metadata.get("run_id"),
        metadata.get("agent_id"),
        metadata.get("workspace"),
    )


def _preview_prompt(prompt: str, limit: int = 300) -> str:
    collapsed = " ".join(str(prompt or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def _log_cursor_sdk_call_shape(
    *,
    prompt: str,
    workspace: Path | None,
    cursor_mode: str,
    prompt_mode: str,
    model_id: str,
    model_params: dict[str, str],
    timeout_seconds: float,
    streaming_enabled: bool,
    retry_count: int,
    local_workspace_enabled: bool,
) -> None:
    logger.info(
        "Cursor SDK call: prompt_length=%s prompt_preview=%r workspace=%s "
        "cursor_mode=%s prompt_mode=%s model_id=%s model_params=%s "
        "timeout_seconds=%s streaming_enabled=%s local_workspace_enabled=%s retry_count=%s",
        len(prompt or ""),
        _preview_prompt(prompt),
        workspace,
        cursor_mode,
        prompt_mode,
        model_id,
        model_params,
        timeout_seconds,
        streaming_enabled,
        local_workspace_enabled,
        retry_count,
    )


def _log_cursor_sdk_timing(
    *,
    phase: str,
    prep_ms: int | None = None,
    sdk_call_ms: int | None = None,
    post_ms: int | None = None,
    total_ms: int | None = None,
    sdk_duration_ms: int | None = None,
) -> None:
    logger.info(
        "Cursor SDK timing: phase=%s prep_ms=%s sdk_call_ms=%s post_ms=%s "
        "total_ms=%s sdk_duration_ms=%s",
        phase,
        prep_ms,
        sdk_call_ms,
        post_ms,
        total_ms,
        sdk_duration_ms,
    )


@contextmanager
def _sanitized_cursor_environment(api_key: str):
    """Temporarily expose only minimal non-Hermes environment to Cursor SDK."""
    original = dict(os.environ)
    sanitized: dict[str, str] = {}
    for key, value in original.items():
        upper = key.upper()
        if upper.startswith("HERMES_"):
            continue
        if any(marker in upper for marker in _SECRET_ENV_MARKERS):
            continue
        if upper in _ALLOWED_ENV_NAMES:
            sanitized[key] = value
    sanitized["CURSOR_API_KEY"] = api_key
    try:
        os.environ.clear()
        os.environ.update(sanitized)
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                if item:
                    parts.append(str(item))
                continue
            typ = item.get("type")
            if typ in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif typ in {"image_url", "input_image"}:
                parts.append("[image content omitted: Cursor SDK text adapter does not pass images]")
        return "\n".join(part for part in parts if part)
    return str(content)


def _messages_to_cursor_prompt(messages: Iterable[dict[str, Any]]) -> str:
    sections: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").upper()
        if role == "TOOL":
            role = "TOOL RESULT"
        text = _content_to_text(msg.get("content")).strip()
        if not text:
            continue
        sections.append(f"[{role}]\n{text}")
    return "\n\n".join(sections).strip()


def _is_tool_only_assistant(msg: dict[str, Any]) -> bool:
    if str(msg.get("role") or "") != "assistant":
        return False
    text = _content_to_text(msg.get("content")).strip()
    return not text and bool(msg.get("tool_calls"))


def _build_cursor_prompt(
    messages: Iterable[dict[str, Any]],
    *,
    prompt_mode: str,
) -> str:
    mode = (prompt_mode or DEFAULT_CURSOR_PROMPT_MODE).strip().lower()
    if mode != "slim":
        return _messages_to_cursor_prompt(messages)

    msg_list = [m for m in (messages or []) if isinstance(m, dict)]
    if not msg_list:
        return ""

    tail: list[dict[str, Any]] = []
    for msg in reversed(msg_list):
        role = str(msg.get("role") or "")
        if role == "system":
            continue
        if role == "tool":
            continue
        if role == "assistant" and _is_tool_only_assistant(msg):
            continue
        tail.append(msg)
        if len(tail) >= 3:
            break

    sections: list[str] = []
    for msg in reversed(tail):
        role = str(msg.get("role") or "user").upper()
        text = _content_to_text(msg.get("content")).strip()
        if text:
            sections.append(f"[{role}]\n{text}")

    prompt_body = "\n\n".join(sections).strip()
    if len(prompt_body) > 8000:
        prompt_body = prompt_body[-8000:]
    if not prompt_body:
        return ""
    return f"{_HERMES_CHAT_PREFIX}\n\n{prompt_body}"


def _normalize_cursor_mode(cursor_mode: str) -> str:
    mode = (cursor_mode or DEFAULT_CURSOR_MODE).strip().lower()
    if mode in {"workspace", "coding", "code"}:
        return "workspace"
    return "chat"


def _workspace_for_call(
    *,
    cursor_mode: str,
    workspace_root: str | Path | None,
    session_id: str | None = None,
) -> Path | None:
    mode = _normalize_cursor_mode(cursor_mode)
    if mode == "workspace":
        root = Path(
            workspace_root
            or os.getenv("HERMES_CURSOR_WORKSPACE_ROOT")
            or DEFAULT_CURSOR_WORKSPACE_ROOT
        )
        label = session_id or f"cursor-sdk-{uuid.uuid4().hex}"
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in label)[:96]
        workspace = root / safe
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    # Chat mode: no session workspace — SDK uses default local runtime cwd.
    return None


def _failure_response(*, metadata: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=metadata.get("run_id") or f"cursor-sdk-failed-{uuid.uuid4().hex}",
        model=metadata.get("model_id") or DEFAULT_CURSOR_MODEL_ID,
        choices=[],
        usage=None,
        error=metadata,
        cursor_metadata=metadata,
    )


def _success_response(*, text: str, metadata: dict[str, Any]) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=text, tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        id=metadata.get("run_id") or f"cursor-sdk-{uuid.uuid4().hex}",
        model=metadata.get("model_id") or DEFAULT_CURSOR_MODEL_ID,
        choices=[choice],
        usage=None,
        cursor_metadata=metadata,
    )


def _build_agent_options(
    *,
    api_key: str,
    model_id: str,
    model_params: dict[str, str],
    workspace: Path | None,
    cursor_mode: str,
):
    from cursor_sdk import (  # type: ignore[import-not-found]
        AgentOptions,
        LocalAgentOptions,
        ModelParameterValue,
        ModelSelection,
    )

    model = ModelSelection(
        id=model_id,
        params=[ModelParameterValue(id=k, value=v) for k, v in model_params.items()],
    )
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
    }
    mode = _normalize_cursor_mode(cursor_mode)
    if mode == "workspace" and workspace is not None:
        kwargs["local"] = LocalAgentOptions(cwd=str(workspace))
    elif mode == "chat":
        terminal_cwd = os.getenv("TERMINAL_CWD", "").strip()
        chat_cwd = terminal_cwd or os.getcwd()
        kwargs["local"] = LocalAgentOptions(cwd=chat_cwd)
    return AgentOptions(**kwargs)


def _run_sdk_prompt_blocking(
    *,
    prompt: str,
    api_key: str,
    model_id: str,
    model_params: dict[str, str],
    workspace: Path | None,
    cursor_mode: str,
    timeout_seconds: float,
):
    timing: dict[str, int | None] = {
        "prep_ms": None,
        "sdk_call_ms": None,
        "post_ms": None,
        "sdk_duration_ms": None,
    }
    call_started = time.monotonic()

    def _call():
        prep_started = time.monotonic()
        try:
            from tools.lazy_deps import ensure as _ensure_lazy_dep

            _ensure_lazy_dep("provider.cursor_sdk", prompt=False)
        except Exception:
            pass

        from cursor_sdk import Agent  # type: ignore[import-not-found]

        options = _build_agent_options(
            api_key=api_key,
            model_id=model_id,
            model_params=model_params,
            workspace=workspace,
            cursor_mode=cursor_mode,
        )
        timing["prep_ms"] = int((time.monotonic() - prep_started) * 1000)
        sdk_started = time.monotonic()
        with _sanitized_cursor_environment(api_key):
            result = Agent.prompt(prompt, options)
        timing["sdk_call_ms"] = int((time.monotonic() - sdk_started) * 1000)
        timing["sdk_duration_ms"] = getattr(result, "duration_ms", None)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        try:
            result = future.result(timeout=timeout_seconds)
        finally:
            timing["post_ms"] = int((time.monotonic() - call_started) * 1000) - (
                (timing["prep_ms"] or 0) + (timing["sdk_call_ms"] or 0)
            )
            _log_cursor_sdk_timing(
                phase="blocking",
                prep_ms=timing["prep_ms"],
                sdk_call_ms=timing["sdk_call_ms"],
                post_ms=timing["post_ms"],
                total_ms=int((time.monotonic() - call_started) * 1000),
                sdk_duration_ms=timing["sdk_duration_ms"],
            )
        return result


def _run_sdk_prompt_streaming(
    *,
    prompt: str,
    api_key: str,
    model_id: str,
    model_params: dict[str, str],
    workspace: Path | None,
    cursor_mode: str,
    timeout_seconds: float,
    on_text_delta: Callable[[str], None] | None,
    interrupt_check: Callable[[], bool] | None = None,
):
    timing: dict[str, int | None] = {
        "prep_ms": None,
        "sdk_call_ms": None,
        "post_ms": None,
        "sdk_duration_ms": None,
    }
    call_started = time.monotonic()

    def _call():
        prep_started = time.monotonic()
        try:
            from tools.lazy_deps import ensure as _ensure_lazy_dep

            _ensure_lazy_dep("provider.cursor_sdk", prompt=False)
        except Exception:
            pass

        from cursor_sdk import Agent  # type: ignore[import-not-found]

        options = _build_agent_options(
            api_key=api_key,
            model_id=model_id,
            model_params=model_params,
            workspace=workspace,
            cursor_mode=cursor_mode,
        )
        timing["prep_ms"] = int((time.monotonic() - prep_started) * 1000)
        sdk_started = time.monotonic()
        create_kwargs: dict[str, Any] = {
            "api_key": options.api_key,
            "model": options.model,
        }
        if getattr(options, "local", None) is not None:
            create_kwargs["local"] = options.local
        with _sanitized_cursor_environment(api_key):
            with Agent.create(**create_kwargs) as agent:
                run = agent.send(prompt)
                for chunk in run.iter_text():
                    if interrupt_check and interrupt_check():
                        if run.supports("cancel"):
                            run.cancel()
                        break
                    if chunk and on_text_delta:
                        on_text_delta(str(chunk))
                result = run.wait()
        timing["sdk_call_ms"] = int((time.monotonic() - sdk_started) * 1000)
        timing["sdk_duration_ms"] = getattr(result, "duration_ms", None)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        try:
            result = future.result(timeout=timeout_seconds)
        finally:
            timing["post_ms"] = int((time.monotonic() - call_started) * 1000) - (
                (timing["prep_ms"] or 0) + (timing["sdk_call_ms"] or 0)
            )
            _log_cursor_sdk_timing(
                phase="streaming",
                prep_ms=timing["prep_ms"],
                sdk_call_ms=timing["sdk_call_ms"],
                post_ms=timing["post_ms"],
                total_ms=int((time.monotonic() - call_started) * 1000),
                sdk_duration_ms=timing["sdk_duration_ms"],
            )
        return result


def run_cursor_sdk_chat_completion(
    *,
    messages: list[dict[str, Any]],
    api_key: str,
    model_id: str = DEFAULT_CURSOR_MODEL_ID,
    model_params: dict[str, str] | None = None,
    workspace_root: str | Path | None = None,
    session_id: str | None = None,
    timeout_seconds: float = DEFAULT_CURSOR_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_CURSOR_MAX_RETRIES,
    cursor_mode: str = DEFAULT_CURSOR_MODE,
    prompt_mode: str = DEFAULT_CURSOR_PROMPT_MODE,
    on_text_delta: Callable[[str], None] | None = None,
    interrupt_check: Callable[[], bool] | None = None,
) -> SimpleNamespace:
    """Run a Cursor SDK prompt and return an OpenAI-shaped response."""
    model_params = dict(model_params or DEFAULT_CURSOR_MODEL_PARAMS)
    prompt = _build_cursor_prompt(messages, prompt_mode=prompt_mode)
    workspace = _workspace_for_call(
        cursor_mode=cursor_mode,
        workspace_root=workspace_root,
        session_id=session_id,
    )
    local_workspace_enabled = _normalize_cursor_mode(cursor_mode) == "workspace"
    timeout_seconds = float(timeout_seconds or DEFAULT_CURSOR_TIMEOUT_SECONDS)
    max_retries = max(0, int(max_retries))
    streaming_enabled = on_text_delta is not None
    started = time.monotonic()
    last_metadata: dict[str, Any] | None = None

    if not str(prompt).strip():
        metadata = {
            "status": "error",
            "latency_ms": 0,
            "model_id": model_id,
            "model_params": dict(model_params),
            "agent_id": None,
            "run_id": None,
            "retry_count": 0,
            "raw_error": "Cursor SDK prompt is empty after message flattening",
            "error": {
                "type": "CursorSDKCallError",
                "message": "Cursor SDK prompt is empty after message flattening",
            },
            "error_type": "CursorSDKCallError",
            "error_message": "Cursor SDK prompt is empty after message flattening",
            "sdk_status": None,
            "timeout_seconds": timeout_seconds,
            "workspace": str(workspace),
            "streaming": streaming_enabled,
        }
        _log_cursor_sdk_failure(metadata)
        return _failure_response(metadata=metadata)

    for attempt in range(max_retries + 1):
        attempt_started = time.monotonic()
        metadata: dict[str, Any] = {
            "status": "unknown",
            "latency_ms": 0,
            "model_id": model_id,
            "model_params": dict(model_params),
            "agent_id": None,
            "run_id": None,
            "retry_count": attempt,
            "raw_error": None,
            "error": None,
            "error_type": None,
            "error_message": None,
            "sdk_status": None,
            "timeout_seconds": timeout_seconds,
            "workspace": str(workspace),
            "cursor_mode": cursor_mode,
            "prompt_mode": prompt_mode,
            "streaming": streaming_enabled,
        }
        _log_cursor_sdk_call_shape(
            prompt=prompt,
            workspace=workspace,
            cursor_mode=cursor_mode,
            prompt_mode=prompt_mode,
            model_id=model_id,
            model_params=model_params,
            timeout_seconds=timeout_seconds,
            streaming_enabled=streaming_enabled,
            retry_count=attempt,
            local_workspace_enabled=local_workspace_enabled,
        )
        try:
            if not api_key:
                raise CursorSDKCallError("missing CURSOR_API_KEY")
            if streaming_enabled:
                result = _run_sdk_prompt_streaming(
                    prompt=prompt,
                    api_key=api_key,
                    model_id=model_id,
                    model_params=model_params,
                    workspace=workspace,
                    cursor_mode=cursor_mode,
                    timeout_seconds=timeout_seconds,
                    on_text_delta=on_text_delta,
                    interrupt_check=interrupt_check,
                )
            else:
                result = _run_sdk_prompt_blocking(
                    prompt=prompt,
                    api_key=api_key,
                    model_id=model_id,
                    model_params=model_params,
                    workspace=workspace,
                    cursor_mode=cursor_mode,
                    timeout_seconds=timeout_seconds,
                )
            status = str(getattr(result, "status", "") or "").strip() or "unknown"
            text = getattr(result, "result", "") or ""
            metadata.update(
                {
                    "status": status,
                    "sdk_status": status,
                    "agent_id": getattr(result, "agent_id", None),
                    "run_id": getattr(result, "id", None) or getattr(result, "run_id", None),
                    "duration_ms": getattr(result, "duration_ms", None),
                }
            )
            if status == "finished" and not str(text).strip():
                metadata["status"] = "empty_result"
                raise CursorSDKCallError("Cursor SDK returned empty result")
            if status != "finished":
                raise CursorSDKCallError(f"Cursor SDK returned status={status!r}")
            metadata["latency_ms"] = int((time.monotonic() - started) * 1000)
            return _success_response(text=str(text).strip(), metadata=metadata)
        except concurrent.futures.TimeoutError:
            metadata["status"] = "timeout"
            metadata["error_type"] = "TimeoutError"
            metadata["error_message"] = f"Cursor SDK call exceeded {timeout_seconds:.1f}s timeout"
            metadata["raw_error"] = metadata["error_message"]
            metadata["error"] = {"type": "TimeoutError", "message": metadata["error_message"]}
        except Exception as exc:  # noqa: BLE001 - preserve raw SDK errors in metadata
            metadata["status"] = metadata.get("status") or "error"
            metadata["error_type"] = exc.__class__.__name__
            metadata["error_message"] = str(exc)
            metadata["raw_error"] = str(exc)
            metadata["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        finally:
            metadata["latency_ms"] = int((time.monotonic() - attempt_started) * 1000)
            last_metadata = metadata

    if last_metadata is None:
        last_metadata = {
            "status": "error",
            "latency_ms": int((time.monotonic() - started) * 1000),
            "model_id": model_id,
            "model_params": dict(model_params),
            "agent_id": None,
            "run_id": None,
            "retry_count": max_retries,
            "raw_error": "Cursor SDK call failed before execution",
            "error": {"type": "CursorSDKCallError", "message": "Cursor SDK call failed before execution"},
            "error_type": "CursorSDKCallError",
            "error_message": "Cursor SDK call failed before execution",
            "sdk_status": None,
            "timeout_seconds": timeout_seconds,
            "workspace": str(workspace),
            "streaming": streaming_enabled,
        }
    last_metadata["retry_count"] = max_retries
    if last_metadata.get("error") is None:
        last_metadata["error"] = {
            "type": last_metadata.get("error_type") or "CursorSDKCallError",
            "message": _cursor_failure_message(last_metadata),
        }
    _log_cursor_sdk_failure(last_metadata)
    return _failure_response(metadata=last_metadata)
