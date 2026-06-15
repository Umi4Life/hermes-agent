"""Cursor SDK provider profile — delegated Composer runtime."""

from providers import register_provider
from providers.base import ProviderProfile

cursor_sdk = ProviderProfile(
    name="cursor-sdk",
    aliases=("cursor_sdk", "cursor"),
    display_name="Cursor SDK",
    description="Cursor Composer via delegated SDK runtime (local bridge + Hermes MCP tools)",
    signup_url="https://cursor.com/dashboard/integrations",
    env_vars=("CURSOR_API_KEY",),
    base_url="cursor-sdk://local",
    auth_type="api_key",
    api_mode="cursor_sdk",
    fallback_models=("composer-2.5",),
)

register_provider(cursor_sdk)
