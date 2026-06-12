"""Focused regression tests for Cursor SDK provider identity resolution.

The Discord /model picker can list the canonical ``cursor-sdk`` provider from the
model catalog, but selecting it routes through ``switch_model(...,
explicit_provider="cursor-sdk")``. That path first calls
``resolve_provider_full("cursor-sdk")``; if provider identity resolution does not
know the slug, the picker fails before validation, before the Cursor SDK
transport, and before gateway session overrides are written.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model
from hermes_cli.providers import resolve_provider_full


def test_resolve_provider_full_accepts_cursor_sdk():
    pdef = resolve_provider_full("cursor-sdk")

    assert pdef is not None
    assert pdef.id == "cursor-sdk"
    assert pdef.name == "Cursor SDK"
    assert pdef.transport == "cursor_sdk"
    assert pdef.base_url == "cursor-sdk://local"
    assert pdef.auth_type == "api_key"


def test_switch_model_explicit_cursor_sdk_composer_is_not_unknown_provider():
    with (
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
    ):
        result = switch_model(
            raw_input="composer-2.5",
            current_provider="openai-codex",
            current_model="gpt-5.5",
            current_base_url="https://chatgpt.com/backend-api/codex",
            current_api_key="",
            is_global=False,
            explicit_provider="cursor-sdk",
        )

    assert result.success is True
    assert result.error_message == ""
    assert result.target_provider == "cursor-sdk"
    assert result.new_model == "composer-2.5"
    assert result.base_url == "cursor-sdk://local"
    assert result.api_mode == "cursor_sdk"
