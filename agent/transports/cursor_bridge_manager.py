"""Owned Cursor bridge pool for the delegated runtime.

When you call ``cursor_sdk.Agent.create()`` without a ``client=``, the SDK
spins up an *implicit, process-wide* bridge that is shared by every session
and cannot be given a request timeout or a transport-retry budget.  When that
single bridge hiccups or dies, *every* session fails (``connection refused`` /
``peer closed connection``).  That shared bridge is the main flakiness source
for the Hermes delegated runtime.

This module **owns** the bridge explicitly via
``CursorClient.launch_bridge(workspace=...)``, keyed by workspace cwd, so we
can:

  * set per-request (``unary_timeout``) and streaming (``stream_timeout``)
    timeouts plus a transport ``max_retries`` budget — bounding the otherwise
    unbounded ``run.wait()`` / ``iter_text()`` blocks;
  * **relaunch** a dead bridge in place instead of failing the user's turn;
  * cap the number of concurrent bridges (LRU eviction).

Why a cwd-keyed pool rather than one global bridge: ``resolve_cursor_sdk_cwd``
falls back to the per-session ``agent.session_cwd``, and ``launch_bridge``
binds a bridge to one workspace.  A process-wide bridge would force every
session into one cwd and silently break per-session file operations.  When all
sessions share a cwd (the common case) the pool collapses to a single bridge.

Design mirrors :mod:`agent.lsp` — module-level singleton state, a
``threading.Lock``, and a lazily-registered ``atexit`` teardown.  The lock
guards bridge **lifecycle only** (launch / relaunch / pool mutation); it is
never held across ``send`` / ``wait``.  Serializing the waits would reintroduce
head-of-line blocking across the gateway's ``ThreadPoolExecutor`` worker
threads.  The SDK rides on httpx (sync httpx clients are concurrency-safe and
``with_options`` returns a shallow copy sharing the transport), so a single
configured client can be shared across worker threads.
"""
from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

logger = logging.getLogger("agent.cursor_bridge")

_DEFAULT_POOL_MAX = 4


@dataclass
class _ManagedBridge:
    cwd: str
    client: Any
    cm: Any  # context-manager handle returned by launch_bridge(...)
    generation: int
    created_at: float
    last_used: float


_bridges: dict[str, _ManagedBridge] = {}
_lock = threading.Lock()
_atexit_registered = False


def _launch(cwd: str, max_retries: int) -> Tuple[Any, Any]:
    """Spawn a fresh bridge for ``cwd``; return ``(cm, client)``.

    ``launch_bridge`` is a context manager; we enter it and keep the handle so
    teardown can ``__exit__`` it.  Raises on failure (caller decides fallback).
    """
    from cursor_sdk import CursorClient

    cm = CursorClient.launch_bridge(workspace=cwd, max_retries=max_retries or 0)
    client = cm.__enter__()
    return cm, client


def _configure(
    client: Any,
    *,
    unary_timeout: float,
    stream_timeout: float,
    max_retries: int,
) -> Any:
    """Return a client bound to our timeout / retry budget.

    ``with_options`` returns a shallow copy sharing the underlying transport,
    so this is cheap and thread-safe to call per turn.
    """
    try:
        return client.with_options(
            unary_timeout=unary_timeout,
            stream_timeout=stream_timeout,
            max_retries=max_retries,
        )
    except Exception:
        logger.debug(
            "cursor bridge with_options failed; using base client", exc_info=True
        )
        return client


def _close_bridge(mb: _ManagedBridge) -> None:
    """Best-effort teardown of one bridge's process + client."""
    closed = False
    cm = getattr(mb, "cm", None)
    if cm is not None:
        try:
            cm.__exit__(None, None, None)
            closed = True
        except Exception:
            logger.debug("cursor bridge cm exit failed", exc_info=True)
    if not closed:
        shutdown = getattr(mb.client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                logger.debug("cursor bridge shutdown failed", exc_info=True)


def _evict_if_needed(pool_max: int) -> None:
    """Evict least-recently-used bridges while the pool is over capacity.

    Caller must hold ``_lock``.
    """
    if pool_max <= 0:
        return
    while len(_bridges) > pool_max:
        victim_key = min(_bridges, key=lambda k: _bridges[k].last_used)
        victim = _bridges.pop(victim_key)
        logger.info("cursor bridge: evicting LRU bridge for %s", victim_key)
        _close_bridge(victim)


def get_client(
    cwd: str,
    *,
    unary_timeout: float = 90.0,
    stream_timeout: float = 120.0,
    max_retries: int = 2,
    pool_max: int = _DEFAULT_POOL_MAX,
) -> Tuple[Optional[Any], int]:
    """Return ``(configured_client, generation)`` for an owned bridge at ``cwd``.

    Lazily launches a bridge under the lifecycle lock on first use for a cwd.
    Returns ``(None, 0)`` when an owned bridge cannot be launched — the caller
    then falls back to the SDK's implicit module-level bridge (``client=None``),
    preserving the pre-hardening behavior rather than failing the turn.

    The returned ``generation`` lets the caller detect, on a later
    :func:`relaunch`, whether another thread already replaced a dead bridge.
    """
    global _atexit_registered
    key = str(cwd)
    with _lock:
        mb = _bridges.get(key)
        if mb is None:
            try:
                cm, client = _launch(key, max_retries)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "cursor bridge launch failed for %s: %s — "
                    "falling back to implicit SDK bridge",
                    key,
                    exc,
                )
                return None, 0
            if not _atexit_registered:
                atexit.register(_atexit_shutdown)
                _atexit_registered = True
            now = time.time()
            mb = _ManagedBridge(
                cwd=key,
                client=client,
                cm=cm,
                generation=1,
                created_at=now,
                last_used=now,
            )
            _bridges[key] = mb
            logger.info("cursor bridge: launched owned bridge for %s", key)
            _evict_if_needed(pool_max)
        mb.last_used = time.time()
        configured = _configure(
            mb.client,
            unary_timeout=unary_timeout,
            stream_timeout=stream_timeout,
            max_retries=max_retries,
        )
        return configured, mb.generation


def relaunch(
    cwd: str,
    observed_generation: int,
    *,
    unary_timeout: float = 90.0,
    stream_timeout: float = 120.0,
    max_retries: int = 2,
) -> Tuple[Optional[Any], int]:
    """Replace a dead bridge for ``cwd`` and return ``(client, generation)``.

    Guards against a relaunch stampede: if another thread already relaunched
    (the live generation no longer matches ``observed_generation``), this
    returns the *current* bridge instead of spawning another.  Returns
    ``(None, 0)`` if no owned bridge exists for this cwd (implicit-bridge mode —
    nothing to relaunch) or if the relaunch itself fails.
    """
    key = str(cwd)
    with _lock:
        mb = _bridges.get(key)
        if mb is None:
            # Implicit-bridge mode (get_client returned None): nothing owned to
            # relaunch.  The SDK manages the implicit bridge itself.
            return None, 0
        if mb.generation != observed_generation:
            # Another thread already relaunched; reuse the live bridge.
            logger.debug(
                "cursor bridge: relaunch superseded for %s (gen %s != %s)",
                key,
                mb.generation,
                observed_generation,
            )
            mb.last_used = time.time()
            configured = _configure(
                mb.client,
                unary_timeout=unary_timeout,
                stream_timeout=stream_timeout,
                max_retries=max_retries,
            )
            return configured, mb.generation

        new_gen = mb.generation + 1
        _close_bridge(mb)
        try:
            cm, client = _launch(key, max_retries)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cursor bridge relaunch failed for %s: %s", key, exc)
            _bridges.pop(key, None)
            return None, 0
        now = time.time()
        mb = _ManagedBridge(
            cwd=key,
            client=client,
            cm=cm,
            generation=new_gen,
            created_at=now,
            last_used=now,
        )
        _bridges[key] = mb
        logger.info("cursor bridge: relaunched bridge for %s (gen=%s)", key, new_gen)
        configured = _configure(
            mb.client,
            unary_timeout=unary_timeout,
            stream_timeout=stream_timeout,
            max_retries=max_retries,
        )
        return configured, new_gen


def shutdown_all() -> None:
    """Tear down every owned bridge.  Safe to call multiple times."""
    with _lock:
        bridges = list(_bridges.values())
        _bridges.clear()
    for mb in bridges:
        _close_bridge(mb)


def _atexit_shutdown() -> None:
    try:
        shutdown_all()
    except Exception as exc:  # noqa: BLE001
        logger.debug("atexit cursor bridge shutdown failed: %s", exc)


__all__ = ["get_client", "relaunch", "shutdown_all"]
