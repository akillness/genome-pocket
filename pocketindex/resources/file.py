"""File resource for PocketIndex."""
import pathlib

# Raster/vector image extensions that the multimodal embedding path can ingest.
# Kept here (next to FileLike) so the connector and the memo layer agree on what
# counts as an image without importing each other.
IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
)


def is_image_path(path: pathlib.PurePath) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


class FilePath:
    def __init__(self, path: pathlib.Path):
        self.path = path


class FileLike:
    def __init__(self, path: pathlib.Path):
        self.file_path = FilePath(path)
        # Binary image files take the multimodal embedding path instead of the
        # text refine/split path; everything downstream branches on this flag.
        self.is_image = is_image_path(path)

    async def read_text(self) -> str:
        # Read file content asynchronously (or synchronously wrapped)
        with open(self.file_path.path, "r", encoding="utf-8") as f:
            return f.read()

    def read_bytes(self) -> bytes:
        # Raw bytes for binary sources (images). Used by the image embedding pass
        # and by the memo fingerprint so an edited image re-embeds.
        with open(self.file_path.path, "rb") as f:
            return f.read()
