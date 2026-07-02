import inspect
import os
import hashlib
from typing import Any

try:
    from cocoindex.connectorkits.fingerprint import fingerprint_bytes as _fp_bytes
    _HAVE_COCOINDEX_FP = True
except ImportError:  # cocoindex not installed; fall back to hashlib
    _HAVE_COCOINDEX_FP = False
    _fp_bytes = None


def _logic_fingerprint(func: Any) -> str:
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
