"""Text splitting operations for PocketIndex.

This mirrors the public surface of upstream ``cocoindex.ops.text``
(:func:`detect_code_language`, :class:`SeparatorSplitter`,
:class:`CustomLanguageConfig`, and a syntax-aware :class:`RecursiveSplitter`)
with a dependency-free, pure-Python implementation. Upstream backs its splitter
with a Rust/tree-sitter core; PocketIndex stays self-contained, so we approximate
syntax awareness with ordered, per-language regex separators (a recursive
character splitter in the spirit of LangChain's language-aware splitter).

All splitters return :class:`Chunk` objects whose start/end
:class:`Position` carry exact character offsets into the *input* text, so the
engine's lineage/memoization layer keeps pointing at real source bytes.
"""
import re
from typing import Dict, List, Optional, Tuple

from pocketindex.resources.chunk import Chunk, Position

# A contiguous text segment: (text, abs_start_offset, abs_end_offset). Segments
# produced for one input always partition it exactly, so concatenating their
# text reproduces the original slice -- which is what keeps offsets exact when
# we merge them back into chunks.
_Segment = Tuple[str, int, int]


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #

# File extension (lower-case, with dot) -> canonical language name.
_EXTENSION_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
}


def detect_code_language(*, filename: str) -> Optional[str]:
    """Detect a (programming) language from a filename.

    Returns the canonical language name if the extension is recognized,
    otherwise ``None``. Mirrors ``cocoindex.ops.text.detect_code_language``.

    >>> detect_code_language(filename="main.py")
    'python'
    >>> detect_code_language(filename="unknown.xyz") is None
    True
    """
    if not filename:
        return None
    # Take the final extension, case-insensitively.
    dot = filename.rfind(".")
    if dot == -1:
        return None
    ext = filename[dot:].lower()
    return _EXTENSION_LANGUAGE.get(ext)


# --------------------------------------------------------------------------- #
# Per-language separator tables (highest-priority boundary first)
# --------------------------------------------------------------------------- #
#
# Separators are regexes. They are tried in order: the splitter cuts on the
# first separator that appears, then recurses into any still-too-large piece
# with the remaining (lower-priority) separators. The separator is kept at the
# START of the following segment ("keep right"), so a `def`/`class`/`fn` line
# stays attached to its body instead of being orphaned. The empty string is the
# terminal fallback: a hard character-count split.

_GENERIC_SEPARATORS: List[str] = [r"\n\n+", r"(?<=[.!?])\s+", r"\n", r" ", r""]

_LANGUAGE_SEPARATORS: Dict[str, List[str]] = {
    "python": [
        r"\nclass ", r"\n[ \t]*def ", r"\n[ \t]*async def ",
        r"\n\n+", r"\n", r" ", r"",
    ],
    "javascript": [
        r"\nclass ", r"\nfunction ", r"\nconst ", r"\nlet ", r"\nvar ",
        r"\n\n+", r"\n", r" ", r"",
    ],
    "typescript": [
        r"\nclass ", r"\ninterface ", r"\nfunction ", r"\nconst ",
        r"\nlet ", r"\nvar ", r"\ntype ", r"\n\n+", r"\n", r" ", r"",
    ],
    "rust": [
        r"\npub fn ", r"\nfn ", r"\nimpl ", r"\nstruct ", r"\nenum ",
        r"\ntrait ", r"\nmod ", r"\n\n+", r"\n", r" ", r"",
    ],
    "go": [
        r"\nfunc ", r"\ntype ", r"\nvar ", r"\nconst ",
        r"\n\n+", r"\n", r" ", r"",
    ],
    "java": [
        r"\n[ \t]*(?:public|private|protected)[^\n]*class ",
        r"\n[ \t]*(?:public|private|protected)[^\n]*\(",
        r"\nclass ", r"\n\n+", r"\n", r" ", r"",
    ],
    "c": [r"\n\w[^\n]*\([^\n]*\)[ \t]*\{", r"\n\n+", r"\n", r" ", r""],
    "cpp": [
        r"\nclass ", r"\nstruct ", r"\n\w[^\n]*\([^\n]*\)[ \t]*\{",
        r"\n\n+", r"\n", r" ", r"",
    ],
    "csharp": [
        r"\n[ \t]*(?:public|private|protected|internal)[^\n]*class ",
        r"\n[ \t]*(?:public|private|protected|internal)[^\n]*\(",
        r"\n\n+", r"\n", r" ", r"",
    ],
    "ruby": [r"\n[ \t]*class ", r"\n[ \t]*def ", r"\n[ \t]*module ",
             r"\n\n+", r"\n", r" ", r""],
    "markdown": [r"\n#{1,6} ", r"\n\n+", r"\n", r" ", r""],
    "html": [r"<(?:div|section|article|header|footer|p)\b", r"\n\n+",
             r"\n", r" ", r""],
}


def _language_separators(language: Optional[str]) -> List[str]:
    if not language:
        return list(_GENERIC_SEPARATORS)
    return list(_LANGUAGE_SEPARATORS.get(language.lower(), _GENERIC_SEPARATORS))


# --------------------------------------------------------------------------- #
# Core recursive separator splitter (offset-preserving)
# --------------------------------------------------------------------------- #


def _split_keep_right(text: str, base: int, separator: str) -> List[_Segment]:
    """Split ``text`` on ``separator`` regex, keeping each separator attached to
    the start of the following segment. Returns contiguous segments with
    absolute offsets (``base`` is the offset of ``text[0]`` in the source).
    """
    matches = [m.start() for m in re.finditer(separator, text) if m.start() > 0]
    cuts = sorted(set([0, *matches, len(text)]))
    segments: List[_Segment] = []
    for a, b in zip(cuts, cuts[1:]):
        if b > a:
            segments.append((text[a:b], base + a, base + b))
    return segments


def _split_chars(text: str, base: int, size: int) -> List[_Segment]:
    """Terminal fallback: hard split into ``size``-character segments."""
    if size <= 0:
        size = len(text) or 1
    segments: List[_Segment] = []
    for i in range(0, len(text), size):
        segments.append((text[i:i + size], base + i, base + i + len(text[i:i + size])))
    return segments


def _merge(segments: List[_Segment], chunk_size: int, chunk_overlap: int) -> List[_Segment]:
    """Greedily merge contiguous segments into chunks up to ``chunk_size``,
    re-seeding each new chunk with trailing segments worth ~``chunk_overlap``
    characters so consecutive chunks overlap. Offsets stay exact because the
    merged text is the concatenation of contiguous pieces.
    """
    chunks: List[_Segment] = []
    current: List[_Segment] = []
    total = 0

    def flush(window: List[_Segment]) -> None:
        if not window:
            return
        merged = "".join(p[0] for p in window)
        if merged.strip():
            chunks.append((merged, window[0][1], window[-1][2]))

    for seg in segments:
        length = len(seg[0])
        if current and total + length > chunk_size:
            flush(current)
            # Drop leading segments until the retained tail fits the overlap
            # budget (and leaves room for the incoming segment).
            while current and (total > chunk_overlap or total + length > chunk_size):
                total -= len(current[0][0])
                current.pop(0)
        current.append(seg)
        total += length
    flush(current)
    return chunks


def _recursive_split(
    text: str,
    base: int,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[_Segment]:
    """Recursively split ``text`` using the ordered ``separators`` so every
    emitted chunk is <= ``chunk_size`` where the separators allow. Returns
    offset-bearing segments.
    """
    # Pick the highest-priority separator that actually occurs (or the empty
    # terminal fallback).
    chosen = ""
    remaining: List[str] = []
    for i, sep in enumerate(separators):
        if sep == "":
            chosen, remaining = "", []
            break
        if re.search(sep, text):
            chosen, remaining = sep, separators[i + 1:]
            break

    if chosen == "":
        pieces = _split_chars(text, base, chunk_size)
    else:
        pieces = _split_keep_right(text, base, chosen)
        # Degenerate separator (e.g. matched only at offset 0): fall through to
        # the next separator so we don't loop forever on the same text.
        if len(pieces) <= 1 and remaining:
            return _recursive_split(text, base, remaining, chunk_size, chunk_overlap)

    out: List[_Segment] = []
    good: List[_Segment] = []
    for piece in pieces:
        if len(piece[0]) <= chunk_size:
            good.append(piece)
            continue
        if good:
            out.extend(_merge(good, chunk_size, chunk_overlap))
            good = []
        if remaining:
            out.extend(
                _recursive_split(piece[0], piece[1], remaining, chunk_size, chunk_overlap)
            )
        else:
            # No finer separator left: hard-split the oversized piece.
            out.extend(_merge(_split_chars(piece[0], piece[1], chunk_size),
                              chunk_size, chunk_overlap))
    if good:
        out.extend(_merge(good, chunk_size, chunk_overlap))
    return out


def _to_chunks(segments: List[_Segment]) -> List[Chunk]:
    return [
        Chunk(text=t, start=Position(char_offset=s), end=Position(char_offset=e))
        for (t, s, e) in segments
    ]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class CustomLanguageConfig:
    """Configuration for a custom language with regex-based separators.

    Mirrors ``cocoindex.ops.text.CustomLanguageConfig``. ``separators_regex``
    are listed highest-priority first; ``aliases`` (e.g. file extensions) also
    resolve to this language.
    """

    def __init__(
        self,
        language_name: str,
        separators_regex: List[str],
        aliases: Optional[List[str]] = None,
    ) -> None:
        self.language_name = language_name
        # Always provide a terminal fallback so recursion can bottom out.
        self.separators_regex = list(separators_regex)
        if not self.separators_regex or self.separators_regex[-1] != "":
            self.separators_regex = [*self.separators_regex, r" ", r""]
        self.aliases = list(aliases or [])


class SeparatorSplitter:
    """Split text by a fixed list of regex separators (OR-joined).

    Mirrors ``cocoindex.ops.text.SeparatorSplitter``. The splitter is stateless
    after construction and can be reused across many inputs.
    """

    def __init__(
        self,
        separators_regex: List[str],
        *,
        keep_separator: Optional[str] = None,
        include_empty: bool = False,
        trim: bool = True,
    ) -> None:
        if not separators_regex:
            raise ValueError("separators_regex must contain at least one pattern")
        self._pattern = "|".join(f"(?:{s})" for s in separators_regex)
        if keep_separator not in (None, "left", "right"):
            raise ValueError("keep_separator must be 'left', 'right', or None")
        self.keep_separator = keep_separator
        self.include_empty = include_empty
        self.trim = trim

    def split(self, text: str) -> List[Chunk]:
        if not text:
            return []
        segments: List[_Segment] = []
        prev = 0
        for m in re.finditer(self._pattern, text):
            s, e = m.start(), m.end()
            if e == s:
                continue  # ignore zero-width matches
            if self.keep_separator == "left":
                segments.append((text[prev:e], prev, e))
                prev = e
            elif self.keep_separator == "right":
                segments.append((text[prev:s], prev, s))
                prev = s
            else:
                segments.append((text[prev:s], prev, s))
                prev = e
        segments.append((text[prev:], prev, len(text)))
        return _to_chunks(self._finalize(segments))

    def _finalize(self, segments: List[_Segment]) -> List[_Segment]:
        out: List[_Segment] = []
        for t, s, e in segments:
            if self.trim:
                lead = len(t) - len(t.lstrip())
                trail = len(t) - len(t.rstrip())
                s2, e2 = s + lead, e - trail
                t2 = t[lead:len(t) - trail] if trail else t[lead:]
            else:
                t2, s2, e2 = t, s, e
            if not t2 and not self.include_empty:
                continue
            out.append((t2, s2, e2))
        return out


class RecursiveSplitter:
    """A recursive, optionally syntax-aware text splitter.

    Mirrors ``cocoindex.ops.text.RecursiveSplitter``. With no ``language`` it
    behaves like a general recursive character splitter (paragraph -> sentence
    -> line -> word -> char); with a recognized ``language`` it prefers that
    language's structural boundaries (class/def/fn/...). ``custom_languages``
    supplement the built-in language table.

    The ``split`` signature stays backward compatible with PocketIndex's
    original character splitter: ``split(text, chunk_size=1000,
    chunk_overlap=200)``.
    """

    def __init__(self, *, custom_languages: Optional[List[CustomLanguageConfig]] = None) -> None:
        self._custom: Dict[str, List[str]] = {}
        for cfg in custom_languages or []:
            self._custom[cfg.language_name.lower()] = cfg.separators_regex
            for alias in cfg.aliases:
                self._custom[alias.lower().lstrip(".")] = cfg.separators_regex

    def _separators_for(self, language: Optional[str]) -> List[str]:
        if language:
            key = language.lower().lstrip(".")
            if key in self._custom:
                return list(self._custom[key])
            # Allow a file extension to resolve to a built-in language.
            detected = detect_code_language(filename=f"x.{key}") if "." not in language else \
                detect_code_language(filename=language)
            if detected:
                return _language_separators(detected)
        return _language_separators(language)

    def split(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        *,
        min_chunk_size: Optional[int] = None,
        language: Optional[str] = None,
    ) -> List[Chunk]:
        if not text:
            return []
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        separators = self._separators_for(language)
        segments = _recursive_split(text, 0, separators, chunk_size, chunk_overlap)
        if not segments:
            return []
        return _to_chunks(segments)
