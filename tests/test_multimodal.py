"""Multimodal (SigLIP2) embedding path — network-free unit tests.

These cover the routing/ingestion plumbing that the opt-in image path adds,
without downloading any weights: the SiglipEmbedder is constructed lazily, so we
assert on type/flags/fingerprints rather than real embeddings (the live
text<->image alignment is covered by the manual end-to-end smoke).
"""
import asyncio
import pathlib
import tempfile

import pocketindex as pix
from pocketindex.connectors import localfs
from pocketindex.ops.sentence_transformers import build_embedder
from pocketindex.ops.siglip_embedder import SiglipEmbedder, is_siglip_model
from pocketindex.resources.file import FileLike, is_image_path


def test_is_siglip_model():
    assert is_siglip_model("google/siglip2-base-patch16-224")
    assert is_siglip_model("google/siglip-so400m-patch14-384")
    assert not is_siglip_model("Qwen/Qwen3-Embedding-0.6B")
    assert not is_siglip_model("all-MiniLM-L6-v2")


def test_build_embedder_routes_siglip_without_loading_weights():
    emb = build_embedder("google/siglip2-base-patch16-224")
    assert isinstance(emb, SiglipEmbedder)
    assert emb.supports_image is True
    # Lazy: constructing the embedder must not have loaded the model yet.
    assert "_runtime" not in emb.__dict__


def test_build_embedder_text_model_has_no_image_support():
    # In the test session SentenceTransformerEmbedder is patched to MockEmbedder,
    # so just assert the non-siglip branch does NOT return a SiglipEmbedder.
    emb = build_embedder("all-MiniLM-L6-v2")
    assert not isinstance(emb, SiglipEmbedder)
    assert getattr(emb, "supports_image", False) is False


def test_image_extension_detection():
    assert is_image_path(pathlib.PurePath("a/b/diagram.png"))
    assert is_image_path(pathlib.PurePath("photo.JPG"))
    assert not is_image_path(pathlib.PurePath("notes.md"))


def test_localfs_lists_images_with_flag():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "note.md").write_text("hello")
        (root / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n binary not utf8 \xff")
        items = localfs.walk_dir(root).items()
        assert items["note.md"].is_image is False
        assert items["pic.png"].is_image is True


def test_memo_hash_uses_bytes_for_images_and_is_stable():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "pic.png"
        # Non-UTF-8 bytes: read_text would raise, so the image branch (read_bytes)
        # must be taken for a stable, non-empty fingerprint.
        p.write_bytes(b"\x89PNG\xff\xfe\x00\x01")
        f = FileLike(p)
        h1 = asyncio.run(pix._compute_memo_hash(f))
        h2 = asyncio.run(pix._compute_memo_hash(FileLike(p)))
        assert h1 and h1 == h2
        # Editing the image changes the fingerprint.
        p.write_bytes(b"\x89PNG\xff\xfe\x00\x02")
        h3 = asyncio.run(pix._compute_memo_hash(FileLike(p)))
        assert h3 != h1
