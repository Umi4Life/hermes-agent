"""Session adapter for the Cursor SDK delegated runtime.

Owns one Cursor ``Agent`` per Hermes session.  Drives ``send`` / ``wait``,
optional ``iter_text()`` streaming, MCP wiring, cancellation, and returns a
turn result that ``run_cursor_sdk_turn`` splices into ``messages``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from hermes_cli.cursor_sdk_config import (
    build_cursor_mcp_servers,
    build_cursor_model_selection,
    build_identity_prefix,
    compute_identity_hash,
    cursor_sdk_agent_meta_key,
    cursor_sdk_identity_hash_key,
    get_cursor_sdk_settings,
    resolve_cursor_sdk_cwd,
)

logger = logging.getLogger(__name__)

_IDENTITY_SEPARATOR = "\n\n---\n\n"


@dataclass
class CursorTurnResult:
    """Result of one user turn through the Cursor SDK."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    should_retire: bool = False
    cursor_agent_error: bool = False
    run_status_error: bool = False
    transient_error: bool = False


def _classify_cursor_exception(exc: BaseException) -> tuple[bool, bool]:
    """Return ``(is_cursor_error, is_transient)`` for retry decisions."""
    try:
        from cursor_sdk import CursorAgentError
    except ImportError:
        CursorAgentError = Exception  # type: ignore[misc,assignment]

    network_error_types: tuple[type, ...] = ()
    try:
        from cursor_sdk.errors import NetworkError

        network_error_types = (NetworkError,)
    except ImportError:
        pass

    msg = str(exc).lower()
    transient_markers = (
        "peer closed connection",
        "incomplete chunked",
        "bridge request failed",
        "remoteprotocol",
    )
    is_transient = isinstance(exc, network_error_types) or any(
        marker in msg for marker in transient_markers
    )
    is_cursor = isinstance(exc, CursorAgentError) or is_transient
    return is_cursor, is_transient


def _coerce_turn_input_text(user_input: Any) -> str:
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str) and item.strip():
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
        return "\n".join(parts)
    if user_input is None:
        return ""
    return str(user_input)


class CursorSDKSession:
    """One Cursor SDK agent per Hermes session."""

    def __init__(self, agent) -> None:
        self._agent = agent
        self._cwd = resolve_cursor_sdk_cwd(agent)
        self._sdk_agent: Any = None
        self._sdk_agent_cm: Any = None
        self._interrupt_event = threading.Event()
        self._pending_identity_prefix: Optional[str] = None

    def request_interrupt(self) -> None:
        self._interrupt_event.set()

    def close(self) -> None:
        self._release_sdk_agent()

    def _release_sdk_agent(self) -> None:
        if self._sdk_agent_cm is not None:
            try:
                self._sdk_agent_cm.__exit__(None, None, None)
            except Exception:
                logger.debug("cursor_sdk agent context exit failed", exc_info=True)
        elif self._sdk_agent is not None:
            close_fn = getattr(self._sdk_agent, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.debug("cursor_sdk agent close failed", exc_info=True)
        self._sdk_agent = None
        self._sdk_agent_cm = None

    def _session_db(self):
        return getattr(self._agent, "_session_db", None)

    def _meta_get(self, key: str) -> Optional[str]:
        db = self._session_db()
        sid = getattr(self._agent, "session_id", None)
        if db is None or not sid:
            return None
        try:
            return db.get_meta(key)
        except Exception:
            return None

    def _meta_set(self, key: str, value: str) -> None:
        db = self._session_db()
        sid = getattr(self._agent, "session_id", None)
        if db is None or not sid:
            return
        try:
            db.set_meta(key, value)
        except Exception:
            logger.debug("cursor_sdk meta write failed for %s", key, exc_info=True)

    def _clear_persisted_agent(self) -> None:
        sid = getattr(self._agent, "session_id", None) or ""
        if not sid:
            return
        self._meta_set(cursor_sdk_agent_meta_key(sid), "")
        self._meta_set(cursor_sdk_identity_hash_key(sid), "")

    def _persist_agent(self, agent_id: str, identity_hash: str) -> None:
        sid = getattr(self._agent, "session_id", None) or ""
        if not sid or not agent_id:
            return
        self._meta_set(cursor_sdk_agent_meta_key(sid), agent_id)
        self._meta_set(cursor_sdk_identity_hash_key(sid), identity_hash)

    def _build_agent_options(self) -> Any:
        from cursor_sdk import AgentOptions, LocalAgentOptions

        settings = get_cursor_sdk_settings()
        mcp_servers = build_cursor_mcp_servers(settings)
        opts: dict[str, Any] = {
            "api_key": getattr(self._agent, "api_key", "") or "",
            "model": build_cursor_model_selection(self._agent, settings),
            "local": LocalAgentOptions(cwd=self._cwd),
        }
        if mcp_servers:
            opts["mcp_servers"] = mcp_servers
        return AgentOptions(**opts)

    def _ensure_sdk_agent(self) -> None:
        if self._sdk_agent is not None:
            return

        from cursor_sdk import Agent, CursorAgentError

        settings = get_cursor_sdk_settings()
        sid = getattr(self._agent, "session_id", None) or ""
        stored_id = (self._meta_get(cursor_sdk_agent_meta_key(sid)) or "").strip()
        stored_hash = (self._meta_get(cursor_sdk_identity_hash_key(sid)) or "").strip()
        identity_hash = compute_identity_hash(self._agent, settings)
        identity_prefix = build_identity_prefix(self._agent, settings)
        if stored_id and stored_hash and stored_hash != identity_hash:
            self._clear_persisted_agent()
            self._release_sdk_agent()
            stored_id = ""
        resume = bool(stored_id) and stored_hash == identity_hash

        try:
            if resume:
                from cursor_sdk import Agent

                opts = self._build_agent_options()
                cm = Agent.resume(stored_id, opts)
            else:
                from cursor_sdk import Agent

                opts = self._build_agent_options()
                if identity_prefix:
                    self._pending_identity_prefix = identity_prefix
                cm = Agent.create(opts)
            self._sdk_agent_cm = cm
            self._sdk_agent = cm.__enter__()
            agent_id = str(getattr(self._sdk_agent, "agent_id", "") or stored_id)
            if agent_id:
                self._persist_agent(agent_id, identity_hash)
        except CursorAgentError as exc:
            self._clear_persisted_agent()
            raise exc
        except Exception:
            self._clear_persisted_agent()
            raise

    def _should_stop(self) -> bool:
        return bool(
            self._interrupt_event.is_set()
            or getattr(self._agent, "_interrupt_requested", False)
        )

    def run_turn(
        self,
        *,
        user_input: Any,
        stream_callback: Optional[Callable[[str], Any]] = None,
    ) -> CursorTurnResult:
        settings = get_cursor_sdk_settings()
        max_retries = max(0, int(settings.get("max_retries", 1) or 1))
        last_result: Optional[CursorTurnResult] = None

        for attempt in range(max_retries + 1):
            result = self._run_turn_once(
                user_input=user_input,
                stream_callback=stream_callback,
                settings=settings,
            )
            if not result.error or not result.transient_error:
                return result
            last_result = result
            if attempt >= max_retries:
                break
            logger.warning(
                "cursor_sdk transient bridge error (attempt %d/%d): %s — retrying",
                attempt + 1,
                max_retries + 1,
                result.error,
            )
            self.close()
            self._clear_persisted_agent()

        return last_result or CursorTurnResult(error="unknown Cursor error")

    def _run_turn_once(
        self,
        *,
        user_input: Any,
        stream_callback: Optional[Callable[[str], Any]],
        settings: dict[str, Any],
    ) -> CursorTurnResult:
        timeout = float(settings.get("timeout_seconds", 180) or 180)
        result = CursorTurnResult()
        prompt = _coerce_turn_input_text(user_input)
        if not prompt.strip():
            result.error = "empty user input"
            return result

        self._interrupt_event.clear()

        try:
            self._ensure_sdk_agent()
        except Exception as exc:
            is_cursor, is_transient = _classify_cursor_exception(exc)
            if is_cursor:
                result.cursor_agent_error = True
            if is_transient:
                result.transient_error = True
            result.error = f"Cursor startup failed: {exc}" if is_cursor else f"Cursor session failed: {exc}"
            result.should_retire = True
            return result

        assert self._sdk_agent is not None
        deadline = time.monotonic() + timeout

        send_prompt = prompt
        if self._pending_identity_prefix:
            send_prompt = f"{self._pending_identity_prefix}{_IDENTITY_SEPARATOR}{prompt}"
            self._pending_identity_prefix = None

        try:
            run = self._sdk_agent.send(send_prompt)
        except Exception as exc:
            is_cursor, is_transient = _classify_cursor_exception(exc)
            if is_cursor:
                result.cursor_agent_error = True
            if is_transient:
                result.transient_error = True
            result.error = f"Cursor send failed: {exc}"
            result.should_retire = True
            return result

        chunks: list[str] = []
        if stream_callback is not None and hasattr(run, "iter_text"):
            try:
                for text in run.iter_text():
                    if self._should_stop() or time.monotonic() >= deadline:
                        break
                    if text:
                        chunks.append(text)
                        try:
                            stream_callback(text)
                        except Exception:
                            logger.debug("cursor_sdk stream callback failed", exc_info=True)
            except Exception as exc:
                is_cursor, is_transient = _classify_cursor_exception(exc)
                if is_transient:
                    result.transient_error = True
                logger.warning(
                    "cursor_sdk iter_text failed, falling back to wait(): %s",
                    exc,
                    exc_info=is_transient,
                )
        if self._should_stop() and hasattr(run, "supports") and run.supports("cancel"):
            try:
                run.cancel()
                result.interrupted = True
            except Exception:
                logger.debug("cursor_sdk run.cancel failed", exc_info=True)

        try:
            terminal = run.wait()
        except Exception as exc:
            is_cursor, is_transient = _classify_cursor_exception(exc)
            if is_cursor:
                result.cursor_agent_error = True
            if is_transient:
                result.transient_error = True
            result.error = f"Cursor wait failed: {exc}"
            result.should_retire = True
            if chunks:
                result.final_text = "".join(chunks).strip()
                if result.final_text:
                    result.projected_messages.append(
                        {"role": "assistant", "content": result.final_text}
                    )
            return result

        status = str(getattr(terminal, "status", "") or "")
        if status == "error":
            result.run_status_error = True
            result.error = f"Cursor run failed (status=error, id={getattr(terminal, 'id', '?')})"
            result.should_retire = True
            self._clear_persisted_agent()
            self._release_sdk_agent()

        final_text = ""
        if hasattr(run, "text"):
            try:
                final_text = str(run.text() or "")
            except Exception:
                final_text = ""
        if not final_text:
            final_text = "".join(chunks)
        if not final_text and status == "finished":
            final_text = str(getattr(terminal, "result", "") or "")

        result.final_text = final_text.strip()
        if result.final_text:
            result.projected_messages.append(
                {"role": "assistant", "content": result.final_text}
            )
            if not result.error:
                result.transient_error = False

        if self._should_stop() and not result.interrupted:
            result.interrupted = True

        if time.monotonic() >= deadline and not result.final_text:
            result.error = result.error or f"Cursor turn timed out after {int(timeout)}s"
            result.should_retire = True
            result.transient_error = True

        return result

