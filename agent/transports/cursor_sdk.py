"""Cursor SDK transport.

The Cursor SDK adapter returns OpenAI Chat Completions-shaped responses, so
normalization is intentionally lightweight. The dedicated transport exists to
make ``api_mode='cursor_sdk'`` a first-class, validated mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, Usage
from agent.transports import register_transport


class CursorSDKTransport(ProviderTransport):
    @property
    def api_mode(self) -> str:
        return "cursor_sdk"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        return messages

    def convert_tools(self, tools: List[Dict[str, Any]]) -> list:
        # Cursor SDK Agent.prompt does not expose Hermes-compatible tool-calling.
        return []

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": self.convert_messages(messages),
            "cursor_model_id": params.get("cursor_model_id") or model or "composer-2.5",
            "cursor_model_params": params.get("cursor_model_params") or {"fast": "false"},
            "cursor_workspace_root": params.get("cursor_workspace_root"),
            "cursor_timeout_seconds": params.get("cursor_timeout_seconds"),
            "cursor_max_retries": params.get("cursor_max_retries", 1),
            "session_id": params.get("session_id"),
        }

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        choice = response.choices[0]
        msg = choice.message
        return NormalizedResponse(
            content=getattr(msg, "content", "") or "",
            tool_calls=[],
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            usage=Usage(),
            provider_data={"cursor_metadata": getattr(response, "cursor_metadata", {})},
        )

    def validate_response(self, response: Any) -> bool:
        if response is None:
            return False
        choices = getattr(response, "choices", None)
        if not choices:
            return False
        try:
            content = choices[0].message.content
        except Exception:
            return False
        return isinstance(content, str) and bool(content.strip())


register_transport("cursor_sdk", CursorSDKTransport)
