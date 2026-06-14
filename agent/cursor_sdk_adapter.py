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
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

DEFAULT_CURSOR_MODEL_ID = "composer-2.5"
DEFAULT_CURSOR_MODEL_PARAMS = {"fast": "false"}
DEFAULT_CURSOR_TIMEOUT_SECONDS = 90.0
DEFAULT_CURSOR_MAX_RETRIES = 1
DEFAULT_CURSOR_WORKSPACE_ROOT = Path("/srv/hermes-cursor/workspaces")

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


@contextmanager
def _sanitized_cursor_environment(api_key: str):
    """Temporarily expose only minimal non-Hermes environment to Cursor SDK.

    The Cursor SDK may spawn its own local bridge/agent subprocesses. Since
    subprocesses inherit ``os.environ`` by default, scrub Hermes/OpenAI/etc.
    secrets during the SDK call and pass only ``CURSOR_API_KEY`` plus a small
    allowlist of runtime variables.
    """
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


def _workspace_for_call(workspace_root: str | Path | None, session_id: str | None = None) -> Path:
    root = Path(workspace_root or os.getenv("HERMES_CURSOR_WORKSPACE_ROOT") or DEFAULT_CURSOR_WORKSPACE_ROOT)
    label = session_id or f"cursor-sdk-{uuid.uuid4().hex}"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in label)[:96]
    workspace = root / safe
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


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


def _run_sdk_prompt_once(
    *,
    prompt: str,
    api_key: str,
    model_id: str,
    model_params: dict[str, str],
    workspace: Path,
    timeout_seconds: float,
):
    def _call():
        try:
            from tools.lazy_deps import ensure as _ensure_lazy_dep
            _ensure_lazy_dep("provider.cursor_sdk", prompt=False)
        except Exception:
            # Fall through to the import below so the actual missing-package or
            # SDK error is preserved in structured metadata.
            pass

        from cursor_sdk import (  # type: ignore[import-not-found]
            Agent,
            AgentOptions,
            LocalAgentOptions,
            ModelParameterValue,
            ModelSelection,
        )

        model = ModelSelection(
            id=model_id,
            params=[ModelParameterValue(id=k, value=v) for k, v in model_params.items()],
        )
        with _sanitized_cursor_environment(api_key):
            return Agent.prompt(
                prompt,
                AgentOptions(
                    api_key=api_key,
                    model=model,
                    local=LocalAgentOptions(cwd=str(workspace)),
                ),
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        return future.result(timeout=timeout_seconds)


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
) -> SimpleNamespace:
    """Run a Cursor SDK prompt and return an OpenAI-shaped response.

    Empty text, non-``finished`` statuses, SDK exceptions, and timeout are
    treated as failures. The adapter retries boundedly (default: one retry),
    then returns an invalid OpenAI-shaped response with structured metadata so
    the existing Hermes fallback machinery can escalate.
    """
    model_params = dict(model_params or DEFAULT_CURSOR_MODEL_PARAMS)
    prompt = _messages_to_cursor_prompt(messages)
    workspace = _workspace_for_call(workspace_root, session_id=session_id)
    timeout_seconds = float(timeout_seconds or DEFAULT_CURSOR_TIMEOUT_SECONDS)
    max_retries = max(0, int(max_retries))
    started = time.monotonic()
    last_metadata: dict[str, Any] | None = None

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
        }
        try:
            if not api_key:
                raise CursorSDKCallError("missing CURSOR_API_KEY")
            result = _run_sdk_prompt_once(
                prompt=prompt,
                api_key=api_key,
                model_id=model_id,
                model_params=model_params,
                workspace=workspace,
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
        }
    last_metadata["retry_count"] = max_retries
    if last_metadata.get("error") is None:
        last_metadata["error"] = {
            "type": last_metadata.get("error_type") or "CursorSDKCallError",
            "message": _cursor_failure_message(last_metadata),
        }
    _log_cursor_sdk_failure(last_metadata)
    return _failure_response(metadata=last_metadata)
