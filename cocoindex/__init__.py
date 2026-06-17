"""Custom implementation of CocoIndex v1 API."""
import asyncio
import contextvars
import hashlib
import inspect
from typing import Any, Callable, Dict, Generic, List, TypeVar, AsyncIterator


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
    "coco_current_source_key", default=None
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
        # value can be a context manager or direct value
        if hasattr(value, "__aenter__"):
            # We will handle async context managers in the App lifecycle
            _CONTEXT[key.name] = value
        else:
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
            f._coco_fn = True
            f._memo = kwargs.get("memo", False)
            return f
        return decorator
    elif isinstance(func_or_memo, bool):
        # Used as @fn(True)
        def decorator(f):
            f._coco_fn = True
            f._memo = func_or_memo
            return f
        return decorator
    else:
        # Used as @fn
        func_or_memo._coco_fn = True
        func_or_memo._memo = False
        return func_or_memo

async def map(func: Callable, items: Any, *args) -> List[Any]:
    # Run the function on each item in the items list/iterable
    results = []
    for item in items:
        if inspect.iscoroutinefunction(func):
            res = await func(item, *args)
        else:
            res = func(item, *args)
        results.append(res)
    return results

async def _compute_memo_hash(value: Any) -> str:
    """Derive a stable content fingerprint for a source item.

    For file-like inputs we hash the actual text content so that any edit
    changes the fingerprint. For everything else we fall back to repr().
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
        # If we cannot read the content, treat it as always-changed.
        return ""
    return hashlib.sha256(payload).hexdigest()


def _find_target(args: tuple):
    """Locate the lineage-aware target among the mounted function's arguments."""
    for a in args:
        if getattr(a, "_is_coco_target", False):
            return a
    return None


async def mount_each(func: Callable, items: Any, *args) -> None:
    """Mount each source item and process it incrementally.

    Implements the declarative ``Target = F(Source)`` contract:
      * memoization  -> unchanged items (matching stored hash) are skipped;
      * reconciliation -> reprocessed items drop chunks no longer emitted;
      * deletion      -> items that vanished from the source are swept from
                         the target along with their lineage state.
    """
    if hasattr(items, "items"):
        iterable = list(items.items())
    else:
        iterable = list(items)

    target = _find_target(args)
    memo_enabled = bool(getattr(func, "_memo", False))
    seen_keys = set()

    for key, value in iterable:
        source_key = str(key)
        seen_keys.add(source_key)

        new_hash = await _compute_memo_hash(value)
        if target is not None and memo_enabled and new_hash:
            stored_hash = target.get_memo(source_key)
            if stored_hash is not None and stored_hash == new_hash:
                # Unchanged source: skip all work (incremental fast path).
                continue

        token = _current_source_key.set(source_key)
        if target is not None:
            target.begin_source(source_key)
        try:
            if inspect.iscoroutinefunction(func):
                await func(value, *args)
            else:
                func(value, *args)
        finally:
            _current_source_key.reset(token)
        if target is not None:
            target.end_source(source_key, new_hash)

    # Garbage-collect target rows whose source items no longer exist.
    if target is not None:
        target.sweep(seen_keys)

class App:
    def __init__(self, name: str, main_func: Callable, **kwargs):
        self.name = name
        self.main_func = main_func
        self.kwargs = kwargs

    def update_blocking(self, live: bool = False, report_to_stdout: bool = True) -> None:
        asyncio.run(self.run_async(live))

    async def run_async(self, live: bool = False) -> None:
        # 1. Run lifespan to set up context
        builder = EnvironmentBuilder()
        active_managers = []
        
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
        
        try:
            # 2. Run main function
            if inspect.iscoroutinefunction(self.main_func):
                await self.main_func(**self.kwargs)
            else:
                self.main_func(**self.kwargs)
        finally:
            # 3. Clean up context managers
            for mgr, entered_val in reversed(active_managers):
                try:
                    await mgr.__aexit__(None, None, None)
                except Exception as e:
                    print(f"Error exiting context manager: {e}")
            
            if _lifespan_func:
                try: 
                    await gen.__anext__()
                except StopAsyncIteration:
                    pass
