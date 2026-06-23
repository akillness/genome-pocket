"""Custom implementation of PocketIndex v1 API."""
import asyncio
import contextvars
import hashlib
import inspect
import os
import time
from typing import Any, Callable, Dict, Generic, List, TypeVar, AsyncIterator

try:
    from cocoindex.connectorkits.fingerprint import fingerprint_bytes as _fp_bytes
    _HAVE_COCOINDEX_FP = True
except ImportError:  # cocoindex not installed; fall back to hashlib
    _HAVE_COCOINDEX_FP = False

from pocketindex.stats import ComponentStats, UpdateStats


T = TypeVar("T")

class ContextKey(Generic[T]):
    def __init__(self, name: str):
        self.name = name

# Global context registry for the current run
_CONTEXT: Dict[str, Any] = {}

# Tracks which source item (e.g. a file) is currently being processed, so that
# declare_row can attribute emitted target rows to their originating source.
# This is the backbone of incremental memoization and deletion propagation.
_current_source_key: contextvars.ContextVar = contextvars.ContextVar(
    "pocket_current_source_key", default=None
)

# Tracks the UpdateStats accumulator for the active App run, so mount_each can
# record per-component processing counters without the caller threading them
# through every layer of the pipeline.
_ACTIVE_STATS: contextvars.ContextVar = contextvars.ContextVar(
    "pocket_active_stats", default=None
)

# When truthy, mount_each bypasses the memo fast-path so every source item is
# reprocessed regardless of its stored fingerprint. Backs
# ``App.update_blocking(full_reprocess=True)`` — a force-clean-rebuild on demand
# (e.g. after a schema change the logic fingerprint can't observe). The per-row
# state-diff in the target layer still dedups physical writes, so a full
# reprocess re-runs every transform without needlessly rewriting unchanged rows.
_FULL_REPROCESS: contextvars.ContextVar = contextvars.ContextVar(
    "pocket_full_reprocess", default=False
)

# Collects the source connectors enumerated during a run so live mode can watch
# exactly what was scanned. App.run_async seeds this with an empty list before
# each run; connectors self-register via ``register_source`` from their item
# enumeration. This backs W2 push-style live mode: instead of blindly re-running
# the whole pipeline every interval, the live loop watches these sources' cheap
# change signatures and only re-runs when an actual add/edit/delete is observed.
_SCANNED_SOURCES: contextvars.ContextVar = contextvars.ContextVar(
    "pocket_scanned_sources", default=None
)


def register_source(source: Any) -> None:
    """Register a source connector scanned during the current run.

    Connectors call this from their enumeration (e.g. ``LocalFS.items``) so that
    live mode knows what to watch. A no-op outside an active run (when no
    collection bag has been seeded). Sources that expose a ``signature()``
    method participate in push-based change detection; others are ignored by the
    watcher and fall back to interval polling.
    """
    bag = _SCANNED_SOURCES.get()
    if bag is not None:
        bag.append(source)


def _sources_signature(sources: "List[Any]") -> Dict[Any, Any]:
    """Combine the change signatures of watchable sources into one mapping.

    Keys are namespaced by source position so two sources can't collide. Sources
    lacking a ``signature()`` method contribute nothing (the caller treats an
    all-empty signature with at least one such source as "unwatchable" and keeps
    interval polling)."""
    sig: Dict[Any, Any] = {}
    for index, src in enumerate(sources):
        getter = getattr(src, "signature", None)
        if callable(getter):
            for key, val in getter().items():
                sig[(index, key)] = val
    return sig


def _any_watchable(sources: "List[Any]") -> bool:
    return any(callable(getattr(src, "signature", None)) for src in sources)

def get_current_source_key():
    """Return the source key of the item currently being processed, if any."""
    return _current_source_key.get()

def use_context(key: ContextKey[T]) -> T:
    if key.name not in _CONTEXT:
        raise KeyError(f"Context key '{key.name}' not found in current environment context.")
    return _CONTEXT[key.name]

class EnvironmentBuilder:
    def provide(self, key: ContextKey[T], value: T) -> None:
        _CONTEXT[key.name] = value

    def provide_with(self, key: ContextKey[T], value: Any) -> None:
        # Store the value (which may be an async context manager); App.run_async
        # is what actually enters/exits any __aenter__-capable value later.
        _CONTEXT[key.name] = value

_lifespan_func: Callable[[EnvironmentBuilder], AsyncIterator[None]] = None

def lifespan(func: Callable[[EnvironmentBuilder], AsyncIterator[None]]):
    global _lifespan_func
    _lifespan_func = func
    return func

def fn(func_or_memo=None, **kwargs):
    if func_or_memo is None:
        # Used as @fn(memo=True)
        def decorator(f):
            f._pix_fn = True
            f._memo = kwargs.get("memo", False)
            return f
        return decorator
    elif isinstance(func_or_memo, bool):
        # Used as @fn(True)
        def decorator(f):
            f._pix_fn = True
            f._memo = func_or_memo
            return f
        return decorator
    else:
        # Used as @fn
        func_or_memo._pix_fn = True
        func_or_memo._memo = False
        return func_or_memo

async def map(func: Callable, items: Any, *args) -> List[Any]:
    """Run func concurrently on each item (mirrors cocoindex.map semantics).

    Uses asyncio.gather so all items are in-flight simultaneously within the
    current component — unlike the old sequential for-loop.
    """
    async def _call(item):
        if inspect.iscoroutinefunction(func):
            return await func(item, *args)
        return func(item, *args)

    return list(await asyncio.gather(*(_call(item) for item in items)))

def _logic_fingerprint(func: Callable) -> str:
    """Stable fingerprint of a transform function's *logic*.

    cocoindex keys its persistent memo on a logic fingerprint so editing a
    transform's code invalidates stale memos and forces reprocessing. We mirror
    that by fingerprinting the function source (falling back to bytecode, then
    to the qualified name) so a change to e.g. chunking or extraction re-runs
    every source item instead of serving output produced by the old code.
    """
    raw = None
    try:
        raw = inspect.getsource(func).encode("utf-8")
    except (OSError, TypeError):
        code = getattr(func, "__code__", None)
        if code is not None:
            raw = code.co_code + repr(code.co_consts).encode("utf-8")
    if raw is None:
        raw = f"{getattr(func, '__module__', '')}.{getattr(func, '__qualname__', repr(func))}".encode("utf-8")
    if _HAVE_COCOINDEX_FP:
        return _fp_bytes(raw).hex()
    return hashlib.sha256(raw).hexdigest()


async def _compute_memo_hash(value: Any, logic_sig: str = "") -> str:
    """Stable content fingerprint for a source item.

    Uses cocoindex.connectorkits.fingerprint.fingerprint_bytes when available
    (same algorithm as the real cocoindex engine) and falls back to SHA-256.
    File-like inputs are hashed by content so any edit changes the fingerprint.
    The active embedding signature (``POCKET_EMBED_SIG``) and the transform
    ``logic_sig`` are folded in so that switching the embedding model or editing
    the transform code invalidates every memo and forces a clean reprocess.
    """
    try:
        if getattr(value, "is_image", False) and hasattr(value, "read_bytes"):
            # Binary image source: fingerprint raw bytes (read_text would fail on
            # non-UTF-8 content) so an edited image re-embeds.
            payload = value.read_bytes()
        elif hasattr(value, "read_text"):
            text = value.read_text()
            if inspect.isawaitable(text):
                text = await text
            payload = text.encode("utf-8")
        else:
            payload = repr(value).encode("utf-8")
    except Exception:
        return ""
    # Fold the embedding signature and the transform's logic fingerprint into
    # the payload so switching the embedding model OR editing the transform code
    # invalidates every memo and forces a clean reprocess at the new logic/dim.
    prefixes = [p for p in (logic_sig, os.getenv("POCKET_EMBED_SIG", "")) if p]
    if prefixes:
        payload = b"\x00".join(p.encode("utf-8") for p in prefixes) + b"\x00" + payload
    if _HAVE_COCOINDEX_FP:
        return _fp_bytes(payload).hex()
    return hashlib.sha256(payload).hexdigest()


def _find_target(args: tuple):
    """Locate the lineage-aware target among the mounted function's arguments."""
    for a in args:
        if getattr(a, "_is_pix_target", False):
            return a
    return None


async def mount_each(func: Callable, items: Any, *args, stats: "UpdateStats" = None) -> "UpdateStats":
    """Mount each source item and process it incrementally.

    Implements the declarative ``Target = F(Source)`` contract:
      * memoization  -> unchanged items (matching stored hash) are skipped;
      * reconciliation -> reprocessed items drop chunks no longer emitted;
      * deletion      -> items that vanished from the source are swept from
                         the target along with their lineage state.

    Returns an :class:`UpdateStats` snapshot (creating one if not supplied) so
    callers can monitor adds/reprocesses/unchanged/deletes/errors per run.
    """
    if hasattr(items, "items"):
        iterable = list(items.items())
    else:
        iterable = list(items)

    target = _find_target(args)
    memo_enabled = bool(getattr(func, "_memo", False))
    # Force-rebuild switch (App.update_blocking(full_reprocess=True)): when set,
    # the memo fast-path below is skipped so every source item re-runs its
    # transform even if its fingerprint is unchanged.
    full_reprocess = bool(_FULL_REPROCESS.get())
    seen_keys = set()
    # Fingerprint the transform's *logic* once per run and fold it into every
    # memo key below. cocoindex keys its persistent memo on a logic fingerprint
    # so that editing a transform (e.g. chunking/extraction) invalidates stale
    # memos and forces reprocessing; without this, an unchanged source file
    # would keep output produced by the *old* code after a pipeline edit.
    logic_sig = _logic_fingerprint(func)

    if stats is None:
        stats = _ACTIVE_STATS.get() or UpdateStats()
    component = stats.component(getattr(func, "__name__", "component"))

    # Snapshot the target's lifetime row tallies so we can attribute just this
    # component's physical writes vs. state-diff skips to its stats.
    rows_written_before = getattr(target, "num_row_writes", 0) if target is not None else 0
    rows_skipped_before = getattr(target, "num_row_skips", 0) if target is not None else 0

    for key, value in iterable:
        source_key = str(key)
        seen_keys.add(source_key)

        new_hash = await _compute_memo_hash(value, logic_sig)
        had_prior_state = (
            target is not None and target.get_memo(source_key) is not None
        )
        if target is not None and memo_enabled and new_hash and not full_reprocess:
            stored_hash = target.get_memo(source_key)
            if stored_hash is not None and stored_hash == new_hash:
                # Unchanged source: skip all work (incremental fast path).
                component.num_unchanged += 1
                continue

        component.num_execution_starts += 1
        token = _current_source_key.set(source_key)
        if target is not None:
            target.begin_source(source_key)
        try:
            if inspect.iscoroutinefunction(func):
                await func(value, *args)
            else:
                func(value, *args)
        except BaseException:
            # Roll back any uncommitted rows this source emitted so a failure
            # mid-source never leaks partial data into the next commit.
            component.num_errors += 1
            if target is not None:
                target.abort_source(source_key)
            raise
        finally:
            _current_source_key.reset(token)
        if target is not None:
            target.end_source(source_key, new_hash)
        if had_prior_state:
            component.num_reprocesses += 1
        else:
            component.num_adds += 1

    # Garbage-collect target rows whose source items no longer exist.
    if target is not None:
        removed = target.sweep(seen_keys)
        component.num_deletes += int(removed or 0)
        component.num_row_writes += target.num_row_writes - rows_written_before
        component.num_row_skips += target.num_row_skips - rows_skipped_before

    return stats

class App:
    def __init__(self, name: str, main_func: Callable, **kwargs):
        self.name = name
        self.main_func = main_func
        self.kwargs = kwargs
        # Most recent run statistics, for monitoring/log cross-checking.
        self.last_stats: "UpdateStats" = None

    def update_blocking(
        self,
        live: bool = False,
        report_to_stdout: bool = True,
        live_interval: float = 2.0,
        full_reprocess: bool = False,
    ) -> "UpdateStats":
        return asyncio.run(
            self.run_async(
                live=live,
                report_to_stdout=report_to_stdout,
                live_interval=live_interval,
                full_reprocess=full_reprocess,
            )
        )

    async def run_async(
        self,
        live: bool = False,
        report_to_stdout: bool = True,
        live_interval: float = 2.0,
        full_reprocess: bool = False,
    ) -> "UpdateStats":
        # 1. Run lifespan to set up context
        builder = EnvironmentBuilder()
        active_managers = []
        gen = None

        if _lifespan_func:
            # _lifespan_func is an async generator
            gen = _lifespan_func(builder)
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

            # For any provided values that are async context managers, enter them
            for key_name, val in list(_CONTEXT.items()):
                if hasattr(val, "__aenter__"):
                    entered_val = await val.__aenter__()
                    _CONTEXT[key_name] = entered_val
                    active_managers.append((val, entered_val))

        # Sources scanned by the most recent run; live mode watches these for
        # change events (W2 push). Refreshed by every _run_once invocation.
        watched_sources: List[Any] = []

        async def _run_once(force_reprocess: bool = False) -> "UpdateStats":
            stats = UpdateStats()
            _ACTIVE_STATS.set(stats)
            _FULL_REPROCESS.set(force_reprocess)
            scan_token = _SCANNED_SOURCES.set([])
            started = time.monotonic()
            try:
                if inspect.iscoroutinefunction(self.main_func):
                    await self.main_func(**self.kwargs)
                else:
                    self.main_func(**self.kwargs)
            finally:
                _ACTIVE_STATS.set(None)
                _FULL_REPROCESS.set(False)
                watched_sources[:] = _SCANNED_SOURCES.get() or []
                _SCANNED_SOURCES.reset(scan_token)
            self.last_stats = stats
            if report_to_stdout:
                elapsed = time.monotonic() - started
                print(
                    f"[pocketindex] run complete in {elapsed:.2f}s\n{stats}",
                    flush=True,
                )
            return stats

        try:
            # 2. Run main function (once for catch-up, repeatedly for live mode)
            # full_reprocess applies to the catch-up pass; subsequent live polls
            # revert to incremental so we don't re-run everything every interval.
            stats = await _run_once(force_reprocess=full_reprocess)
            if live:
                # Push-style live mode: when the scanned sources expose change
                # signatures, the loop only re-runs the pipeline after an actual
                # add/edit/delete — idle periods cost just a cheap stat scan
                # instead of a full re-embedding pass. Sources without a
                # signature() fall back to the original interval polling so no
                # change is ever silently missed.
                watchable = _any_watchable(watched_sources)
                last_sig = _sources_signature(watched_sources)
                if report_to_stdout:
                    mode = "watching for changes" if watchable else "polling"
                    print(
                        f"[pocketindex] entering live mode "
                        f"({mode} every {live_interval:.1f}s; Ctrl+C to stop)",
                        flush=True,
                    )
                try:
                    while True:
                        await asyncio.sleep(live_interval)
                        if watchable:
                            current_sig = _sources_signature(watched_sources)
                            if current_sig == last_sig:
                                continue  # no change event — stay idle
                        stats = await _run_once()
                        watchable = _any_watchable(watched_sources)
                        last_sig = _sources_signature(watched_sources)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    if report_to_stdout:
                        print("[pocketindex] live mode stopped.", flush=True)
            return stats
        finally:
            # 3. Clean up context managers
            for mgr, entered_val in reversed(active_managers):
                try:
                    await mgr.__aexit__(None, None, None)
                except Exception as e:
                    print(f"Error exiting context manager: {e}")

            if gen is not None:
                try:
                    await gen.__anext__()
                except StopAsyncIteration:
                    pass
