"""Cursor SDK runtime — delegated Composer turn driver."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _format_cursor_error(turn) -> str:
    detail = getattr(turn, "error", None) or "unknown Cursor error"
    if getattr(turn, "cursor_agent_error", False):
        return f"⚠️ Cursor: {detail}"
    if getattr(turn, "run_status_error", False):
        return f"⚠️ Cursor: {detail}"
    if getattr(turn, "interrupted", False):
        return f"⚠️ Cursor: turn interrupted ({detail})"
    return f"⚠️ Cursor: {detail}"


def run_cursor_sdk_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
    stream_callback=None,
) -> Dict[str, Any]:
    """Hand one turn to the Cursor SDK delegated runtime."""
    from agent.transports.cursor_sdk_session import CursorSDKSession

    if getattr(agent, "_interrupt_requested", False):
        if hasattr(agent, "_cursor_session") and agent._cursor_session is not None:
            agent._cursor_session.request_interrupt()

    if not hasattr(agent, "_cursor_session") or agent._cursor_session is None:
        agent._cursor_session = CursorSDKSession(agent)

    try:
        turn = agent._cursor_session.run_turn(
            user_input=user_message,
            stream_callback=stream_callback,
        )
    except Exception as exc:
        logger.exception("cursor_sdk turn failed")
        try:
            agent._cursor_session.close()
        except Exception:
            pass
        agent._cursor_session = None
        return {
            "final_response": f"⚠️ Cursor: turn failed: {exc}",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
            "cursor_fallback_eligible": True,
        }

    if getattr(turn, "should_retire", False):
        try:
            agent._cursor_session.close()
        except Exception:
            pass
        agent._cursor_session = None

    if turn.projected_messages:
        messages.extend(turn.projected_messages)

    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    agent.session_api_calls = getattr(agent, "session_api_calls", 0) + 1
    api_calls = 1

    failed = bool(turn.error) or turn.interrupted
    if not failed and turn.final_text:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
            )
        except Exception:
            logger.debug("cursor_sdk external memory sync failed", exc_info=True)

    if failed:
        return {
            "final_response": _format_cursor_error(turn),
            "messages": messages,
            "api_calls": api_calls,
            "completed": False,
            "partial": True,
            "error": turn.error,
            "cursor_fallback_eligible": True,
        }

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": api_calls,
        "completed": True,
        "partial": False,
        "error": None,
    }


def run_cursor_sdk_turn_with_fallback(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
    stream_callback=None,
    run_conversation_fn,
) -> Dict[str, Any]:
    """Cursor turn with optional two-message fallback orchestration."""
    result = run_cursor_sdk_turn(
        agent,
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        effective_task_id=effective_task_id,
        should_review_memory=should_review_memory,
        stream_callback=stream_callback,
    )
    if not result.get("cursor_fallback_eligible"):
        return result

    error_text = result.get("final_response") or ""
    if not agent._try_activate_fallback():
        return result

    agent._cursor_fallback_replay = True
    fb_result = run_conversation_fn(
        agent,
        user_message,
        conversation_history=messages,
        stream_callback=stream_callback,
    )
    fallback_text = fb_result.get("final_response") or ""
    if fallback_text:
        result["delivery_messages"] = [error_text, fallback_text]
        result["final_response"] = None
        result["messages"] = fb_result.get("messages", messages)
        result["completed"] = fb_result.get("completed", False)
        result["partial"] = True
        result["cursor_fallback_eligible"] = False
    return result
