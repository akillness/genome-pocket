"""Local filesystem connector for PocketIndex."""
import pathlib
from typing import Dict
from pocketindex.resources.file import FileLike

class LocalFS:
    def __init__(self, sourcedir: pathlib.Path):
        self.sourcedir = sourcedir

    def items(self) -> Dict[str, FileLike]:
        # Walk the source directory and return a dict of relative paths to FileLike objects
        res = {}
        if not self.sourcedir.exists():
            return res
        for p in self.sourcedir.rglob("*"):
            if p.is_file() and p.suffix in (".md", ".txt"):
                rel_path = p.relative_to(self.sourcedir)
                res[str(rel_path)] = FileLike(p)
        return res

def walk_dir(sourcedir: pathlib.Path, recursive: bool = True, live: bool = False) -> LocalFS:
    return LocalFS(sourcedir)
