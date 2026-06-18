"""Custom implementation of PocketIndex v1 API."""
import asyncio
import contextvars
import hashlib
import inspect
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

async def _compute_memo_hash(value: Any) -> str:
    """Stable content fingerprint for a source item.

    Uses cocoindex.connectorkits.fingerprint.fingerprint_bytes when available
    (same algorithm as the real cocoindex engine) and falls back to SHA-256.
    File-like inputs are hashed by content so any edit changes the fingerprint.
    """
    try:
        if hasattr(value, "read_text"):
            text = value.read_text()
            if inspect.isawaitable(text):
                text = await text
            payload = text.encode("utf-8")
        else:
            payload = repr(value).encode("utf-8")
    except Exception:
        return ""
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
    seen_keys = set()

    if stats is None:
        stats = _ACTIVE_STATS.get() or UpdateStats()
    component = stats.component(getattr(func, "__name__", "component"))

    for key, value in iterable:
        source_key = str(key)
        seen_keys.add(source_key)

        new_hash = await _compute_memo_hash(value)
        had_prior_state = (
            target is not None and target.get_memo(source_key) is not None
        )
        if target is not None and memo_enabled and new_hash:
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
    ) -> "UpdateStats":
        return asyncio.run(
            self.run_async(
                live=live,
                report_to_stdout=report_to_stdout,
                live_interval=live_interval,
            )
        )

    async def run_async(
        self,
        live: bool = False,
        report_to_stdout: bool = True,
        live_interval: float = 2.0,
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

        async def _run_once() -> "UpdateStats":
            stats = UpdateStats()
            _ACTIVE_STATS.set(stats)
            started = time.monotonic()
            try:
                if inspect.iscoroutinefunction(self.main_func):
                    await self.main_func(**self.kwargs)
                else:
                    self.main_func(**self.kwargs)
            finally:
                _ACTIVE_STATS.set(None)
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
            stats = await _run_once()
            if live:
                if report_to_stdout:
                    print(
                        f"[pocketindex] entering live mode "
                        f"(polling every {live_interval:.1f}s; Ctrl+C to stop)",
                        flush=True,
                    )
                try:
                    while True:
                        await asyncio.sleep(live_interval)
                        stats = await _run_once()
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
