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

from agent.transports import cursor_bridge_manager
from hermes_cli.cursor_sdk_config import (
    build_cursor_mcp_servers,
    build_cursor_model_selection,
    build_identity_prefix,
    compute_identity_hash,
    cursor_sdk_agent_created_at_key,
    cursor_sdk_agent_meta_key,
    cursor_sdk_identity_hash_key,
    cursor_sdk_turn_count_key,
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
    # Owned bridge died (connection refused / unreachable) — relaunch + retry
    # rather than treat as a plain transient or a fatal error.
    bridge_down: bool = False
    # Permanent error (auth, bad config, bad request) — do not retry; surface now.
    fail_fast: bool = False
    # Server-requested backoff (e.g. rate limit), in seconds, when provided.
    retry_after: Optional[float] = None


@dataclass
class _ClassifiedError:
    """Outcome of classifying a Cursor SDK exception for retry decisions."""

    is_cursor: bool = False
    is_transient: bool = False
    is_bridge_down: bool = False
    fail_fast: bool = False
    retry_after: Optional[float] = None


_BRIDGE_DOWN_MARKERS = (
    "connection refused",
    "errno 111",
    "connecterror",
    "failed to establish a new connection",
)
_TRANSIENT_MARKERS = (
    "peer closed connection",
    "incomplete chunked",
    "remoteprotocol",
)


def _load_error_types() -> Optional[dict[str, tuple[type, ...] | type | None]]:
    """Load the cursor_sdk typed-exception hierarchy, or ``None`` if absent.

    Import-safe on checkouts without ``cursor_sdk`` installed (e.g. Windows
    dev) — callers fall back to string-marker classification.
    """
    try:
        from cursor_sdk import errors as e
    except Exception:
        return None

    def _pick(*names: str) -> tuple[type, ...]:
        out: list[type] = []
        for name in names:
            t = getattr(e, name, None)
            if isinstance(t, type):
                out.append(t)
        return tuple(out)

    return {
        # Permanent — retrying cannot help; surface immediately.
        "fail_fast": _pick(
            "AuthenticationError",
            "PermissionDeniedError",
            "ConfigurationError",
            "BadRequestError",
            "NotFoundError",
            "AgentNotFoundError",
            "UnsupportedRunOperationError",
        ),
        # Transient — worth a bounded retry.
        "transient": _pick(
            "NetworkError",
            "APITimeoutError",
            "InternalServerError",
            "AgentBusyError",
            "RateLimitError",
        ),
        "base": getattr(e, "CursorAgentError", None),
    }


def _coerce_retry_after(exc: BaseException) -> Optional[float]:
    raw = getattr(exc, "retry_after", None)
    if raw is None:
        return None
    try:
        val = float(raw)
        return val if val > 0 else None
    except (TypeError, ValueError):
        return None


def _classify_cursor_exception(exc: BaseException) -> _ClassifiedError:
    """Classify a Cursor SDK exception using the SDK's typed hierarchy.

    Prefers ``isinstance`` checks against ``cursor_sdk.errors`` plus the
    exception's own ``is_retryable`` / ``retry_after`` fields, and falls back
    to string markers when the typed hierarchy is unavailable or unmatched.
    """
    info = _ClassifiedError(retry_after=_coerce_retry_after(exc))
    msg = str(exc).lower()

    # Bridge-down is checked first: a refused/unreachable owned bridge should
    # be relaunched, regardless of how the SDK happens to type it.
    if any(marker in msg for marker in _BRIDGE_DOWN_MARKERS):
        info.is_cursor = True
        info.is_bridge_down = True
        return info

    errs = _load_error_types()
    if errs:
        fail_fast_types = errs.get("fail_fast") or ()
        transient_types = errs.get("transient") or ()
        if fail_fast_types and isinstance(exc, fail_fast_types):
            info.is_cursor = True
            info.fail_fast = True
            return info
        if transient_types and isinstance(exc, transient_types):
            info.is_cursor = True
            info.is_transient = True
            return info
        base = errs.get("base")
        if isinstance(base, type) and isinstance(exc, base):
            info.is_cursor = True
            info.is_transient = bool(getattr(exc, "is_retryable", False))
            return info

    if any(marker in msg for marker in _TRANSIENT_MARKERS):
        info.is_cursor = True
        info.is_transient = True
    if getattr(exc, "is_retryable", False):
        info.is_cursor = True
        info.is_transient = True
    return info


def _format_cursor_startup_error(exc: BaseException) -> str:
    msg = str(exc).lower()
    if any(
        marker in msg
        for marker in ("connection refused", "errno 111", "connecterror")
    ):
        return (
            "Cursor bridge unavailable (connection refused). "
            "The local Cursor agent bridge may have stopped; retry shortly or use fallback."
        )
    return f"Cursor startup failed: {exc}"


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
        # Owned bridge client (None => fall back to SDK's implicit bridge) and
        # the bridge generation we acquired, for relaunch race-detection.
        self._client: Any = None
        self._bridge_generation: int = 0
        # Force-expire a stuck prior run on the next send (set on resume/retry).
        self._force_next_send: bool = False

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
        # The bridge client is owned by cursor_bridge_manager, not by us — drop
        # our reference but never close the pooled bridge here.
        self._client = None

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
        self._meta_set(cursor_sdk_turn_count_key(sid), "")
        self._meta_set(cursor_sdk_agent_created_at_key(sid), "")

    def _get_turn_count(self, session_id: str) -> int:
        raw = (self._meta_get(cursor_sdk_turn_count_key(session_id)) or "").strip()
        try:
            return max(0, int(raw))
        except ValueError:
            return 0

    def _init_agent_rotation_meta(self, session_id: str) -> None:
        self._meta_set(cursor_sdk_turn_count_key(session_id), "0")
        self._meta_set(cursor_sdk_agent_created_at_key(session_id), str(int(time.time())))

    def _increment_turn_count(self, session_id: str) -> None:
        count = self._get_turn_count(session_id) + 1
        self._meta_set(cursor_sdk_turn_count_key(session_id), str(count))

    def _should_rotate_agent(self, settings: dict[str, Any], session_id: str) -> bool:
        max_turns = int(settings.get("max_turns_per_agent", 0) or 0)
        if max_turns > 0 and self._get_turn_count(session_id) >= max_turns:
            return True
        max_age = float(settings.get("max_agent_age_seconds", 0) or 0)
        if max_age > 0:
            raw_created = (self._meta_get(cursor_sdk_agent_created_at_key(session_id)) or "").strip()
            try:
                created_at = float(raw_created)
            except ValueError:
                return False
            if created_at > 0 and (time.time() - created_at) >= max_age:
                return True
        return False

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

    def _bridge_timeouts(self, settings: dict[str, Any]) -> dict[str, Any]:
        return {
            "unary_timeout": float(
                settings.get("client_unary_timeout_seconds", 90) or 90
            ),
            "stream_timeout": float(
                settings.get("client_stream_timeout_seconds", 120) or 120
            ),
            "max_retries": int(settings.get("client_max_retries", 2) or 0),
        }

    def _acquire_client(self, settings: dict[str, Any]) -> tuple[Any, int]:
        """Acquire an owned bridge client, or ``(None, 0)`` to use the implicit one."""
        if not settings.get("own_bridge", True):
            return None, 0
        try:
            return cursor_bridge_manager.get_client(
                self._cwd,
                pool_max=int(settings.get("bridge_pool_max", 4) or 4),
                **self._bridge_timeouts(settings),
            )
        except Exception:
            logger.debug("cursor_sdk: owned bridge acquire failed", exc_info=True)
            return None, 0

    def _relaunch_bridge(self, settings: dict[str, Any]) -> None:
        """Relaunch a dead owned bridge so the next attempt gets a fresh one."""
        if self._client is None:
            return  # implicit-bridge mode — the SDK manages it
        try:
            self._client, self._bridge_generation = cursor_bridge_manager.relaunch(
                self._cwd,
                self._bridge_generation,
                **self._bridge_timeouts(settings),
            )
        except Exception:
            logger.debug("cursor_sdk: bridge relaunch failed", exc_info=True)

    def _ensure_sdk_agent(self) -> None:
        if self._sdk_agent is not None:
            return

        from cursor_sdk import Agent, CursorAgentError

        settings = get_cursor_sdk_settings()
        self._client, self._bridge_generation = self._acquire_client(settings)
        sid = getattr(self._agent, "session_id", None) or ""
        stored_id = (self._meta_get(cursor_sdk_agent_meta_key(sid)) or "").strip()
        stored_hash = (self._meta_get(cursor_sdk_identity_hash_key(sid)) or "").strip()
        identity_hash = compute_identity_hash(self._agent, settings)
        identity_prefix = build_identity_prefix(self._agent, settings)
        if stored_id and stored_hash and stored_hash != identity_hash:
            self._clear_persisted_agent()
            self._release_sdk_agent()
            stored_id = ""
        if stored_id and self._should_rotate_agent(settings, sid):
            logger.info(
                "cursor_sdk: rotating agent (turn_count=%s, max_turns=%s, max_age=%ss)",
                self._get_turn_count(sid),
                settings.get("max_turns_per_agent", 0),
                settings.get("max_agent_age_seconds", 0),
            )
            self._clear_persisted_agent()
            self._release_sdk_agent()
            stored_id = ""
            if identity_prefix:
                self._pending_identity_prefix = identity_prefix
        resume = bool(stored_id) and stored_hash == identity_hash

        # Pass the owned bridge client only when we have one; omitting the
        # kwarg keeps the implicit-bridge call signature unchanged.
        client_kwargs: dict[str, Any] = {}
        if self._client is not None:
            client_kwargs["client"] = self._client

        try:
            if resume:
                from cursor_sdk import Agent

                opts = self._build_agent_options()
                cm = Agent.resume(stored_id, opts, **client_kwargs)
                created_new = False
                # A resumed agent may carry a stuck prior run on the bridge;
                # force-expire it on the next send.
                self._force_next_send = True
            else:
                from cursor_sdk import Agent

                opts = self._build_agent_options()
                if identity_prefix:
                    self._pending_identity_prefix = identity_prefix
                cm = Agent.create(opts, **client_kwargs)
                created_new = True
            self._sdk_agent_cm = cm
            self._sdk_agent = cm.__enter__()
            agent_id = str(getattr(self._sdk_agent, "agent_id", "") or stored_id)
            if agent_id:
                self._persist_agent(agent_id, identity_hash)
                if created_new:
                    self._init_agent_rotation_meta(sid)
        except CursorAgentError as exc:
            # A dead bridge leaves the server-side agent_id valid; keep it so a
            # relaunch + resume can recover.  Only drop it on permanent errors.
            classified = _classify_cursor_exception(exc)
            if not classified.is_bridge_down:
                self._clear_persisted_agent()
            raise exc
        except Exception as exc:
            classified = _classify_cursor_exception(exc)
            if not classified.is_bridge_down:
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
        max_retries = max(0, int(settings.get("max_retries", 2) or 0))
        backoff_base = float(settings.get("retry_backoff_seconds", 5) or 0)
        backoff_cap = float(settings.get("retry_backoff_cap_seconds", 30) or 0)
        last_result: Optional[CursorTurnResult] = None

        for attempt in range(max_retries + 1):
            result = self._run_turn_once(
                user_input=user_input,
                stream_callback=stream_callback,
                settings=settings,
            )
            if (
                not result.error
                and not result.interrupted
                and result.final_text
            ):
                sid = getattr(self._agent, "session_id", None) or ""
                if sid:
                    self._increment_turn_count(sid)
            # Success, permanent failure, or a non-retryable error all return now.
            if not result.error or result.fail_fast:
                return result
            if not (result.transient_error or result.bridge_down):
                return result
            last_result = result
            if attempt >= max_retries:
                break
            logger.warning(
                "cursor_sdk %s (attempt %d/%d): %s — retrying",
                "bridge down" if result.bridge_down else "transient error",
                attempt + 1,
                max_retries + 1,
                result.error,
            )
            # A dead owned bridge is relaunched before the next attempt
            # re-acquires it; otherwise the pool would hand back the corpse.
            if result.bridge_down:
                self._relaunch_bridge(settings)
            sleep_s = self._retry_sleep(
                result.retry_after, attempt, backoff_base, backoff_cap
            )
            if sleep_s > 0:
                time.sleep(sleep_s)
            self.close()
            self._clear_persisted_agent()
            # The next send may collide with a stuck run from this attempt.
            self._force_next_send = True

        return last_result or CursorTurnResult(error="unknown Cursor error")

    @staticmethod
    def _retry_sleep(
        retry_after: Optional[float],
        attempt: int,
        backoff_base: float,
        backoff_cap: float,
    ) -> float:
        """Server-requested backoff wins; otherwise capped exponential backoff."""
        if retry_after and retry_after > 0:
            return retry_after if backoff_cap <= 0 else min(retry_after, backoff_cap)
        if backoff_base <= 0:
            return 0.0
        delay = backoff_base * (2 ** attempt)
        return delay if backoff_cap <= 0 else min(delay, backoff_cap)

    @staticmethod
    def _apply_classification(
        result: CursorTurnResult, info: _ClassifiedError
    ) -> None:
        """Fold a classification into the turn result's retry flags."""
        if info.is_cursor:
            result.cursor_agent_error = True
        if info.is_transient:
            result.transient_error = True
        if info.is_bridge_down:
            result.bridge_down = True
            result.cursor_agent_error = True
        if info.fail_fast:
            result.fail_fast = True
        if info.retry_after is not None:
            result.retry_after = info.retry_after

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
            info = _classify_cursor_exception(exc)
            self._apply_classification(result, info)
            result.error = (
                _format_cursor_startup_error(exc)
                if info.is_cursor
                else f"Cursor session failed: {exc}"
            )
            result.should_retire = True
            return result

        assert self._sdk_agent is not None
        deadline = time.monotonic() + timeout

        send_prompt = prompt
        if self._pending_identity_prefix:
            send_prompt = f"{self._pending_identity_prefix}{_IDENTITY_SEPARATOR}{prompt}"
            self._pending_identity_prefix = None

        # Force-expire a stuck prior run when resuming or retrying.
        send_kwargs: dict[str, Any] = {}
        if self._force_next_send:
            send_kwargs["options"] = {"local": {"force": True}}
        self._force_next_send = False

        try:
            run = self._sdk_agent.send(send_prompt, **send_kwargs)
        except Exception as exc:
            info = _classify_cursor_exception(exc)
            self._apply_classification(result, info)
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
                info = _classify_cursor_exception(exc)
                if info.is_transient:
                    result.transient_error = True
                logger.warning(
                    "cursor_sdk iter_text failed, falling back to wait(): %s",
                    exc,
                    exc_info=info.is_transient,
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
            info = _classify_cursor_exception(exc)
            self._apply_classification(result, info)
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
            # Cancel a still-running run so a hung bridge turn is cleaned up
            # rather than left draining behind us.
            if str(getattr(run, "status", "") or "") == "running":
                try:
                    run.cancel()
                except Exception:
                    logger.debug(
                        "cursor_sdk run.cancel on deadline failed", exc_info=True
                    )
            result.error = result.error or f"Cursor turn timed out after {int(timeout)}s"
            result.should_retire = True
            result.transient_error = True

        return result

