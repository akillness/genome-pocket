"""Chunk resource for PocketIndex."""
from dataclasses import dataclass

@dataclass
class Position:
    char_offset: int

@dataclass
class Chunk:
    text: str
    start: Position
    end: Position
