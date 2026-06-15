"""Cursor SDK runtime — delegated Composer turn driver."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from hermes_constants import display_hermes_home

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
    from hermes_cli.cursor_sdk_config import (
        cap_channel_context_block,
        get_cursor_sdk_settings,
    )

    settings = get_cursor_sdk_settings()
    user_message = cap_channel_context_block(
        user_message,
        int(settings.get("max_channel_context_chars", 0) or 0),
    )

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
            if agent._cursor_session is not None:
                agent._cursor_session._clear_persisted_agent()
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
    chain = list(getattr(agent, "_fallback_chain", None) or [])
    if not chain:
        logger.warning(
            "cursor_sdk: Cursor failed but fallback_providers is empty — "
            "add entries in config.yaml or `hermes fallback add`"
        )
        if error_text and "fallback_providers" not in error_text.lower():
            result["final_response"] = (
                f"{error_text}\n\n"
                "_(No fallback_providers configured — add one in "
                f"{display_hermes_home()}/config.yaml.)_"
            )
        return result

    if not agent._try_activate_fallback():
        logger.warning(
            "cursor_sdk: fallback chain exhausted or providers not configured "
            "(%d entries, index=%s)",
            len(chain),
            getattr(agent, "_fallback_index", "?"),
        )
        if error_text and "fallback" not in error_text.lower():
            result["final_response"] = (
                f"{error_text}\n\n"
                "_(Fallback chain could not activate — check API keys and "
                f"{display_hermes_home()}/config.yaml fallback_providers.)_"
            )
        return result

    fb_entry = chain[min(getattr(agent, "_fallback_index", 1) - 1, len(chain) - 1)]
    logger.info(
        "cursor_sdk: activating fallback %s/%s",
        fb_entry.get("provider"),
        fb_entry.get("model"),
    )

    agent._cursor_fallback_replay = True
    fb_result = run_conversation_fn(
        agent,
        user_message,
        conversation_history=messages,
        stream_callback=stream_callback,
    )
    fallback_text = fb_result.get("final_response") or ""
    if not fallback_text:
        fb_error = fb_result.get("error")
        if fb_error:
            fallback_text = f"⚠️ Fallback failed: {fb_error}"
        else:
            logger.warning(
                "cursor_sdk: fallback returned no response (completed=%s)",
                fb_result.get("completed"),
            )

    if fallback_text:
        result["delivery_messages"] = [error_text, fallback_text]
        result["final_response"] = None
        result["messages"] = fb_result.get("messages", messages)
        result["completed"] = fb_result.get("completed", False)
        result["partial"] = True
        result["cursor_fallback_eligible"] = False
        logger.info("cursor_sdk: two-message fallback delivery prepared")
    return result
