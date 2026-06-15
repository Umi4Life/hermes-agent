"""Cursor SDK runtime configuration and identity helpers."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.config import load_config

logger = logging.getLogger(__name__)

_DEFAULT_CURSOR_SDK: Dict[str, Any] = {
    "runtime": "delegated",
    "timeout_seconds": 180,
    "max_retries": 1,
    "cwd": None,
    "local": True,
    # Composer 2.5 fast toggle — false = standard (cheaper); true = low-latency fast tier.
    # Hermes defaults to standard; Cursor IDE defaults to fast when unset on their side.
    "fast": False,
    "inject_identity": True,
    "identity_mode": "full",
    "identity_max_chars": 4000,
    "hermes_tools_mcp": True,
}


def get_cursor_sdk_settings(config: Optional[dict] = None) -> dict[str, Any]:
    """Return merged ``cursor_sdk`` settings from config.yaml."""
    cfg = config if config is not None else load_config()
    raw = cfg.get("cursor_sdk") if isinstance(cfg, dict) else None
    if not isinstance(raw, dict):
        raw = {}
    out = dict(_DEFAULT_CURSOR_SDK)
    out.update(raw)
    return out


def build_cursor_model_selection(
    agent,
    settings: Optional[dict] = None,
) -> dict[str, Any]:
    """Build Cursor SDK model selection with explicit fast/standard param."""
    settings = settings or get_cursor_sdk_settings()
    model_id = (getattr(agent, "model", "") or "composer-2.5").strip()
    use_fast = bool(settings.get("fast", False))
    return {
        "id": model_id,
        "params": [{"id": "fast", "value": "true" if use_fast else "false"}],
    }


def resolve_cursor_sdk_cwd(agent, settings: Optional[dict] = None) -> str:
    """Resolve working directory for the local Cursor agent."""
    settings = settings or get_cursor_sdk_settings()
    configured = settings.get("cwd")
    if isinstance(configured, str) and configured.strip():
        return str(Path(configured).expanduser().resolve())
    session_cwd = getattr(agent, "session_cwd", None)
    if isinstance(session_cwd, str) and session_cwd.strip():
        return session_cwd
    terminal_cwd = os.environ.get("TERMINAL_CWD", "").strip()
    if terminal_cwd:
        return terminal_cwd
    return os.getcwd()


def build_hermes_tools_mcp_entry() -> dict[str, Any]:
    """Stdio MCP entry for Hermes tools (mirrors codex migration helper)."""
    from hermes_cli.codex_runtime_plugin_migration import _build_hermes_tools_mcp_entry

    return _build_hermes_tools_mcp_entry()


def build_cursor_mcp_servers(settings: Optional[dict] = None) -> Optional[dict[str, Any]]:
    """Build inline MCP server map for Cursor SDK Agent.create/resume."""
    settings = settings or get_cursor_sdk_settings()
    if not settings.get("hermes_tools_mcp", True):
        return None
    entry = build_hermes_tools_mcp_entry()
    return {"hermes-tools": entry}


def _soul_mtime() -> str:
    try:
        from hermes_constants import get_hermes_home

        soul_path = get_hermes_home() / "SOUL.md"
        if soul_path.exists():
            return str(int(soul_path.stat().st_mtime))
    except Exception:
        pass
    return ""


def compute_identity_hash(agent, settings: Optional[dict] = None) -> str:
    """Hash identity inputs so resume can skip re-injection when unchanged."""
    settings = settings or get_cursor_sdk_settings()
    ephemeral = str(getattr(agent, "ephemeral_system_prompt", "") or "")
    mode = str(settings.get("identity_mode", "full"))
    use_fast = bool(settings.get("fast", False))
    digest = hashlib.sha256(
        f"{mode}|{_soul_mtime()}|{ephemeral}|fast={use_fast}".encode("utf-8")
    ).hexdigest()[:16]
    return digest


def build_identity_prefix(agent, settings: Optional[dict] = None) -> str:
    """Build SOUL + /personality overlay for Cursor agent creation."""
    settings = settings or get_cursor_sdk_settings()
    if not settings.get("inject_identity", True):
        return ""
    mode = str(settings.get("identity_mode", "full")).strip().lower()
    if mode == "off":
        return ""

    parts: list[str] = []
    if mode in {"full", "compact"}:
        try:
            from agent.prompt_builder import load_soul_md

            soul = load_soul_md()
            if soul:
                if mode == "compact":
                    max_chars = int(settings.get("identity_max_chars", 4000) or 4000)
                    soul = soul[:max_chars]
                parts.append(soul)
        except Exception as exc:
            logger.debug("cursor_sdk identity: SOUL load failed: %s", exc)

    ephemeral = str(getattr(agent, "ephemeral_system_prompt", "") or "").strip()
    if ephemeral:
        parts.append(ephemeral)

    platform = str(getattr(agent, "platform", "") or "").strip().lower()
    if platform == "discord":
        parts.append(
            "You are replying on Discord. Keep responses readable in chat; "
            "use concise paragraphs."
        )

    return "\n\n".join(p for p in parts if p)


def sanitize_cursor_sdk_env() -> dict[str, str]:
    """Child env for Cursor bridge: keep CURSOR_* and essentials, drop provider keys."""
    keep_prefixes = ("CURSOR_", "PATH", "HOME", "USER", "TMP", "TEMP", "SYSTEMROOT")
    drop_prefixes = (
        "OPENAI_",
        "ANTHROPIC_",
        "OPENROUTER_",
        "GEMINI_",
        "GMI_",
        "DEEPSEEK_",
        "HERMES_",
    )
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if not isinstance(value, str):
            continue
        upper = key.upper()
        if upper.startswith(keep_prefixes):
            out[key] = value
            continue
        if upper.startswith(drop_prefixes):
            continue
        if upper in {"LANG", "LC_ALL", "LC_CTYPE", "PYTHONPATH", "PYTHONUTF8"}:
            out[key] = value
    if "PYTHONUTF8" not in out:
        out["PYTHONUTF8"] = "1"
    return out


def cursor_sdk_agent_meta_key(session_id: str) -> str:
    return f"cursor_sdk.agent_id.{session_id}"


def cursor_sdk_identity_hash_key(session_id: str) -> str:
    return f"cursor_sdk.identity_hash.{session_id}"
