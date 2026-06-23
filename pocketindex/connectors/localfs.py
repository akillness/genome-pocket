"""Local filesystem connector for PocketIndex."""
import pathlib
from typing import Dict, Optional, Set
from pocketindex.resources.file import FileLike, is_image_path
from pocketindex.ops.text import detect_code_language

# Plain-text/document extensions that are always indexable.
_TEXT_EXTENSIONS: Set[str] = {".md", ".markdown", ".txt", ".rst"}


def _is_indexable(path: pathlib.Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        return True
    # Image files are ingested by the opt-in multimodal embedding path. They are
    # always listed here; the pipeline only routes them when the active embedder
    # advertises image support (otherwise they are simply skipped).
    if is_image_path(path):
        return True
    # Recognized source-code files (python, rust, js/ts, go, ...) are indexable
    # too so the code-aware refine + splitting path has something to chew on.
    return detect_code_language(filename=path.name) is not None


class LocalFS:
    def __init__(self, sourcedir: pathlib.Path):
        self.sourcedir = sourcedir

    def items(self) -> Dict[str, FileLike]:
        # Walk the source directory and return a dict of relative paths to FileLike objects
        from pocketindex import register_source

        # Self-register so live mode can watch this directory for change events.
        register_source(self)
        res = {}
        if not self.sourcedir.exists():
            return res
        for p in self.sourcedir.rglob("*"):
            if p.is_file() and _is_indexable(p):
                rel_path = p.relative_to(self.sourcedir)
                res[str(rel_path)] = FileLike(p)
        return res

    def signature(self) -> Dict[str, tuple]:
        """Cheap change signature of the indexable files under ``sourcedir``.

        Maps each indexable file's relative path to ``(mtime_ns, size)``. Live
        mode compares successive signatures to detect adds, edits, and deletes
        without re-running the full pipeline — a stat scan is orders of magnitude
        cheaper than re-embedding every file."""
        sig: Dict[str, tuple] = {}
        if not self.sourcedir.exists():
            return sig
        for p in self.sourcedir.rglob("*"):
            if p.is_file() and _is_indexable(p):
                st = p.stat()
                rel_path = p.relative_to(self.sourcedir)
                sig[str(rel_path)] = (st.st_mtime_ns, st.st_size)
        return sig

def walk_dir(sourcedir: pathlib.Path, recursive: bool = True, live: bool = False) -> LocalFS:
    return LocalFS(sourcedir)
