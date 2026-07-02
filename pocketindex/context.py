import contextvars
from typing import Any, Dict, Generic, TypeVar, Callable, AsyncIterator

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


def get_current_source_key():
    """Return the source key of the item currently being processed, if any."""
    return _current_source_key.get()


def use_context(key: ContextKey[T]) -> T:
    if key.name not in _CONTEXT:
        raise KeyError(
            f"Context key '{key.name}' not found in current environment context."
        )
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
