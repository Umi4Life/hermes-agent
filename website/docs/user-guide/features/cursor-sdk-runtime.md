---
title: Cursor SDK Runtime
sidebar_label: Cursor SDK Runtime
---

# Cursor SDK Runtime

Hermes can run turns through the [Cursor SDK](https://cursor.com/docs/sdk/python) (`cursor-sdk` Python package) when `model.provider` is `cursor-sdk`. Composer owns the tool loop locally against your configured working directory; Hermes exposes its richer tools (`web_search`, `browser_*`, skills, vision, etc.) via the same stdio MCP callback used by the Codex app-server runtime.

## Enable

```yaml
model:
  provider: cursor-sdk
  default: composer-2.5

providers:
  cursor-sdk:
    key_env: CURSOR_API_KEY

cursor_sdk:
  runtime: delegated
  timeout_seconds: 180
  fast: false            # false = standard (default, cheaper); true = fast tier
  hermes_tools_mcp: true
  inject_identity: true
  identity_mode: full   # full | compact | off
  max_channel_context_chars: 16000  # truncate Discord mention backfill; 0 = off
  max_turns_per_agent: 0             # rotate Agent.create after N successes; 0 = off
  max_agent_age_seconds: 0          # rotate after wall-clock age; 0 = off

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

Set `CURSOR_API_KEY` in `~/.hermes/.env` (Dashboard → Integrations).

Install the optional package:

```bash
pip install 'hermes-agent[cursor-sdk]'
```

## What tools the model has

### Cursor bridge (local `cwd`)

Shell, read, grep, and patch-like edits against the working directory — Cursor's own agent tools.

### Hermes MCP callback (`hermes_tools_mcp: true`)

Same surface as [Codex app-server runtime](./codex-app-server-runtime.md#3-hermes-tool-callback-mcp-server-registered-in-codexconfigtoml): `web_search`, `web_extract`, `browser_*`, `vision_analyze`, `image_generate`, `skill_view`, `skills_list`, `text_to_speech`, and kanban worker tools when env-gated.

Prefer Hermes MCP for web/browser when configured; use the Cursor bridge for fast cwd file operations.

### Not available on this runtime

`delegate_task`, `memory`, `session_search`, and `todo` require the Hermes agent loop. Use `fallback_providers` to switch to a native provider when you need them.

## Composer 2.5: standard vs fast

`composer-2.5` is one model with a latency/cost toggle. Hermes defaults to **standard** (`cursor_sdk.fast: false`) — cheaper per token, better for gateway/agent loops. Set `fast: true` for lower latency at higher cost (Cursor IDE's interactive default).

Changing `fast` rotates the stored Cursor `agent_id` on the next turn (included in the identity hash).

## SOUL, `/personality`, and Discord

Delegated runtimes do not receive Hermes' full system prompt stack. With `inject_identity: true`:

- **`identity_mode: full`** (default) — injects `load_soul_md()` (same 20k cap as default Hermes) plus `/personality` (`ephemeral_system_prompt`) on `Agent.create`.
- **`identity_mode: compact`** — head-truncated SOUL to `identity_max_chars` plus personality overlay.
- **`identity_mode: off`** — Composer default voice only.

Identity is injected on create (or on resume when the identity hash changes after `/personality` or SOUL edits). Changing personality clears the stored Cursor `agent_id` so the next turn starts fresh.

## Errors and fallback

When Cursor fails, Hermes sends **two Discord messages** when `fallback_providers` is configured:

1. `⚠️ Cursor: …` — explicit error relay
2. The fallback provider's answer on the native Hermes loop

The fallback chain skips entries that duplicate the active `cursor-sdk` + model pair.

## Session continuity

`SessionDB.state_meta` stores `cursor_sdk.agent_id.<session_id>` for `Agent.resume()`. Inline MCP servers are re-passed on every resume (not persisted by the SDK).

On startup failure, the stored id is cleared and the next turn calls `Agent.create` again.

## Context length

Hermes resolves `composer-2.5` to a **200K** context window in the model catalog (not the generic 256K unknown-model fallback). Override anytime with `model.context_length: 200000` in `config.yaml`.

## Bridge hardening

### Channel context cap

Discord `@mention` backfill prepends recent channel history before your message. On the cursor path, Hermes truncates that block to `cursor_sdk.max_channel_context_chars` (default **16000**) before `Agent.send()`. Set `0` to disable. Native providers are unaffected.

### Agent rotation (opt-in)

Long-lived `Agent.resume()` sessions can stress the local bridge. Optional rotation forces a fresh `Agent.create`:

| Key | Default | Meaning |
|-----|---------|---------|
| `max_turns_per_agent` | `0` | Rotate after this many **successful** cursor turns (`0` = disabled) |
| `max_agent_age_seconds` | `0` | Rotate when the stored agent is older than this (`0` = disabled) |

Recommended starting points for heavy Discord use: `max_turns_per_agent: 25`–`40` or `max_agent_age_seconds: 3600`. Use `/new` after bridge errors even when rotation is off.

## Interrupt and restart

`/stop` maps to `run.cancel()` when supported. Gateway restart with in-flight Cursor turns may need up to `timeout_seconds` (default 180s) to drain — use `/stop` or `/new` before restarting when possible.
