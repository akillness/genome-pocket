"""PocketIndex compatibility facade.

The implementation is split across focused modules (context, runtime, memo,
app), but callers still import the public API from ``pocketindex`` as ``pix``.
Private symbols used by the in-tree tests are re-exported here as compatibility
shims while the engine internals settle.
"""
from .context import (
    ContextKey,
    EnvironmentBuilder,
    get_current_source_key,
    use_context,
    lifespan,
    _CONTEXT,
    _current_source_key,
    _ACTIVE_STATS,
    _FULL_REPROCESS,
    _SCANNED_SOURCES,
    _lifespan_func,
)
from .runtime import (
    fn,
    map,
    mount_each,
    register_source,
    _sources_signature,
    _any_watchable,
    _find_target,
)
from .memo import _logic_fingerprint, _compute_memo_hash
from .app import App
from .stats import ComponentStats, UpdateStats

__all__ = [
    "ContextKey",
    "EnvironmentBuilder",
    "get_current_source_key",
    "use_context",
    "lifespan",
    "fn",
    "map",
    "mount_each",
    "register_source",
    "App",
    "ComponentStats",
    "UpdateStats",
    "_CONTEXT",
    "_current_source_key",
    "_ACTIVE_STATS",
    "_FULL_REPROCESS",
    "_SCANNED_SOURCES",
    "_lifespan_func",
    "_sources_signature",
    "_any_watchable",
    "_logic_fingerprint",
    "_compute_memo_hash",
    "_find_target",
]
