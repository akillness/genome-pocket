import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional
from pocketindex.stats import UpdateStats
from .context import (
    _SCANNED_SOURCES,
    _ACTIVE_STATS,
    _FULL_REPROCESS,
    _current_source_key,
)
from .memo import _logic_fingerprint, _compute_memo_hash


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


def _sources_signature(sources: List[Any]) -> Dict[Any, Any]:
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


def _any_watchable(sources: List[Any]) -> bool:
    return any(callable(getattr(src, "signature", None)) for src in sources)


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


def _find_target(args: tuple):
    """Locate the lineage-aware target among the mounted function's arguments."""
    for a in args:
        if getattr(a, "_is_pix_target", False):
            return a
    return None


async def mount_each(
    func: Callable, items: Any, *args, stats: Optional[UpdateStats] = None
) -> UpdateStats:
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
    rows_written_before = (
        getattr(target, "num_row_writes", 0) if target is not None else 0
    )
    rows_skipped_before = (
        getattr(target, "num_row_skips", 0) if target is not None else 0
    )

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
