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
        res = {}
        if not self.sourcedir.exists():
            return res
        for p in self.sourcedir.rglob("*"):
            if p.is_file() and _is_indexable(p):
                rel_path = p.relative_to(self.sourcedir)
                res[str(rel_path)] = FileLike(p)
        return res

def walk_dir(sourcedir: pathlib.Path, recursive: bool = True, live: bool = False) -> LocalFS:
    return LocalFS(sourcedir)
