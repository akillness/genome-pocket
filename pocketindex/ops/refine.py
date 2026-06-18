"""Data refinement operations for PocketIndex.

PocketIndex's incremental model is ``Target = F(Source)``. ``F`` is rarely a
single step -- in real pipelines the raw source is first *refined* (cleaned and
normalized) before it is chunked, embedded, and loaded. This module provides a
deterministic, dependency-free refinement stage so the engine can implement the
full Source -> Refine -> Load contract instead of feeding raw bytes straight
into the embedder.

Refinement is intentionally pure and deterministic: the same input always
produces the same output. That property is what lets the memoization layer in
``pocketindex`` treat a refined document as a stable function of its source.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List


# Matches 3+ consecutive blank lines (with optional trailing spaces).
_EXCESS_BLANK_LINES = re.compile(r"\n[ \t]*\n[ \t]*(\n[ \t]*)+")
# Matches trailing whitespace at the end of each line.
_TRAILING_WS = re.compile(r"[ \t]+(?=\n)")
# Matches runs of 2+ spaces/tabs inside a line.
_INLINE_RUNS = re.compile(r"[ \t]{2,}")


@dataclass
class RefinedDocument:
    """The output of the refinement stage.

    ``text`` is the cleaned content used for chunking/embedding. ``offset_map``
    maps every character index in the refined text back to its character index
    in the *original* source, so downstream lineage (start/end offsets) keeps
    pointing at the real bytes the user can open in an editor.
    """

    text: str
    offset_map: List[int] = field(default_factory=list)

    def source_offset(self, refined_offset: int) -> int:
        """Translate a refined-text offset back to the original source offset."""
        if not self.offset_map:
            return refined_offset
        if refined_offset < 0:
            return self.offset_map[0]
        if refined_offset >= len(self.offset_map):
            # End-of-text: point one past the last mapped original character.
            return self.offset_map[-1] + 1
        return self.offset_map[refined_offset]


class TextRefiner:
    """Clean and normalize raw text while preserving source lineage.

    Steps (all deterministic):
      * Unicode NFC normalization so visually identical text compares equal.
      * Normalize CRLF / CR line endings to LF.
      * Strip trailing whitespace on each line.
      * Collapse 3+ blank lines down to a single blank line.
      * Collapse inline runs of spaces/tabs to a single space.
      * Strip leading/trailing whitespace of the whole document.
    """

    def refine(self, text: str) -> RefinedDocument:
        if not text:
            return RefinedDocument(text="", offset_map=[])

        # 1. Unicode NFC normalization. NFC must run over a base character
        #    together with its trailing combining marks (a grapheme starter +
        #    its combining cluster); normalizing each code point in isolation
        #    would never compose decomposed sequences (e.g. "e"+U+0301 -> "é").
        #    Each resulting character is mapped back to the start of the
        #    original cluster it derived from so offsets stay meaningful.
        normalized_chars: List[str] = []
        norm_offsets: List[int] = []
        src_idx = 0
        text_len = len(text)
        while src_idx < text_len:
            cluster_end = src_idx + 1
            while cluster_end < text_len and unicodedata.combining(text[cluster_end]):
                cluster_end += 1
            nfc = unicodedata.normalize("NFC", text[src_idx:cluster_end])
            for nch in nfc:
                normalized_chars.append(nch)
                norm_offsets.append(src_idx)
            src_idx = cluster_end

        # 2. Normalize line endings (CRLF -> LF, lone CR -> LF) on the
        #    normalized stream, carrying the offset map along.
        nl_chars: List[str] = []
        nl_offsets: List[int] = []
        i = 0
        n = len(normalized_chars)
        while i < n:
            ch = normalized_chars[i]
            if ch == "\r":
                nl_chars.append("\n")
                nl_offsets.append(norm_offsets[i])
                if i + 1 < n and normalized_chars[i + 1] == "\n":
                    i += 1  # consume the LF half of a CRLF pair
            else:
                nl_chars.append(ch)
                nl_offsets.append(norm_offsets[i])
            i += 1

        # 3..6. Whitespace collapsing. We rebuild the string char-by-char so the
        #       offset map stays exact through every deletion.
        out_chars: List[str] = []
        out_offsets: List[int] = []
        m = len(nl_chars)
        i = 0
        while i < m:
            ch = nl_chars[i]
            if ch in (" ", "\t"):
                # Look ahead: is the rest of this run only whitespace up to a
                # newline or end? If so it's trailing whitespace -> drop it.
                j = i
                while j < m and nl_chars[j] in (" ", "\t"):
                    j += 1
                trailing = (j >= m) or (nl_chars[j] == "\n")
                if trailing:
                    i = j
                    continue
                # Inline whitespace run -> collapse to a single space.
                out_chars.append(" ")
                out_offsets.append(nl_offsets[i])
                i = j
                continue
            if ch == "\n":
                # Collapse runs of blank lines: keep at most one blank line,
                # i.e. at most two consecutive newlines.
                j = i
                newline_count = 0
                while j < m and nl_chars[j] in ("\n", " ", "\t"):
                    if nl_chars[j] == "\n":
                        newline_count += 1
                    j += 1
                keep = min(newline_count, 2)
                for _ in range(keep):
                    out_chars.append("\n")
                    out_offsets.append(nl_offsets[i])
                i = j
                continue
            out_chars.append(ch)
            out_offsets.append(nl_offsets[i])
            i += 1

        # 6. Strip leading / trailing whitespace of the whole document.
        start = 0
        end = len(out_chars)
        while start < end and out_chars[start] in (" ", "\t", "\n"):
            start += 1
        while end > start and out_chars[end - 1] in (" ", "\t", "\n"):
            end -= 1

        refined_text = "".join(out_chars[start:end])
        refined_offsets = out_offsets[start:end]
        return RefinedDocument(text=refined_text, offset_map=refined_offsets)
