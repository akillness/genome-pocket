"""File resource for PocketIndex."""
import pathlib

class FilePath:
    def __init__(self, path: pathlib.Path):
        self.path = path

class FileLike:
    def __init__(self, path: pathlib.Path):
        self.file_path = FilePath(path)

    async def read_text(self) -> str:
        # Read file content asynchronously (or synchronously wrapped)
        with open(self.file_path.path, "r", encoding="utf-8") as f:
            return f.read()
