"""Deterministic ID generator for PocketIndex."""
import hashlib

class IdGenerator:
    async def next_id(self, text: str) -> int:
        # Generate a stable 64-bit integer ID from the text hash
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Convert first 8 bytes to a signed 64-bit integer (SQLite compatible)
        val = int.from_bytes(h[:8], byteorder="big", signed=True)
        return val
