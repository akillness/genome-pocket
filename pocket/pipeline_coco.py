"""Phase-4 PoC: pipeline using real cocoindex ops while keeping pocketindex engine.

This module is a drop-in alternative to pocket/pipeline.py that replaces the
pocketindex ops (RecursiveSplitter, SentenceTransformerEmbedder,
detect_code_language) with their real cocoindex equivalents.  The engine layer
(pocketindex App / fn / map / mount_each / lifespan) is intentionally kept
unchanged for this iteration because the real cocoindex.App requires an LMDB
backing store and a running Tokio runtime — a heavier migration that belongs in
Phase 4 full (see docs/architecture/cocoindex-gap.md, Phase 4).

Migration checklist:
    [x] ops.text.RecursiveSplitter        → cocoindex.ops.text.RecursiveSplitter
    [x] ops.text.detect_code_language     → cocoindex.ops.text.detect_code_language
    [x] ops.sentence_transformers         → cocoindex.ops.sentence_transformers
    [ ] pocketindex.App                   → cocoindex.App + LMDB settings  (Phase 4 full)
    [ ] pocketindex.fn / map / mount_each → cocoindex.fn / map / mount_each (Phase 4 full)
    [ ] pocketindex.lifespan              → cocoindex.lifespan              (Phase 4 full)

To run a one-off comparison against the standard pipeline::

    pocket update                        # uses pocket/pipeline.py
    POCKET_PIPELINE=coco pocket update   # uses pocket/pipeline_coco.py (PoC)

The env var routing is wired in pocket/cli.py (see _get_app_main()).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Annotated, AsyncIterator, List, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Engine: pocketindex (unchanged for Phase-4a) ────────────────────────────
import pocketindex as pix
from pocketindex.connectors import localfs, sqlite
from pocketindex.resources.file import FileLike
from pocketindex.resources.id import IdGenerator
from pocketindex.resources.chunk import Chunk

# ── Ops: real cocoindex (the migration payload) ──────────────────────────────
from cocoindex.ops.text import RecursiveSplitter as CocoRecursiveSplitter
from cocoindex.ops.text import detect_code_language as coco_detect_code_language
from cocoindex.ops.sentence_transformers import (
    SentenceTransformerEmbedder as CocoSentenceTransformerEmbedder,
)

# ── Ops: pocketindex-only (not yet in cocoindex, kept as-is) ─────────────────
from pocketindex.ops.refine import TextRefiner
from pocketindex.ops.extract import build_extractor, ExtractedEntity
from pocketindex.ops.entity_resolution import resolve_entities, normalize

import pocket.config as config


# ── Context keys (same as pipeline.py) ───────────────────────────────────────
EMBEDDER = pix.ContextKey[CocoSentenceTransformerEmbedder]("embedder")
SQLITE_DB = pix.ContextKey[sqlite.ManagedConnection]("sqlite_db")


@dataclass
class ChunkEmbedding:
    id: int
    file_path: str
    text: str
    embedding: Annotated[NDArray, EMBEDDER]
    start_offset: int
    end_offset: int


@dataclass
class EntityNode:
    id: int
    name: str
    type: str
    aliases: str
    embedding: Annotated[NDArray, EMBEDDER]
    summary: str
    confidence: float
    source_file: str
    source_chunk_ids: str


@dataclass
class RelationEdge:
    id: int
    subject_id: int
    predicate: str
    object_id: int
    evidence: str
    confidence: float
    source_file: str
    source_chunk_id: int


# ── Ops instances (cocoindex versions) ────────────────────────────────────────
_splitter = CocoRecursiveSplitter()
_refiner = TextRefiner()  # pocketindex — no cocoindex equivalent yet


@pix.lifespan
async def pocket_coco_lifespan(
    builder: pix.EnvironmentBuilder,
) -> AsyncIterator[None]:
    """Lifespan: provide cocoindex embedder + pocketindex SQLite connection."""
    import os

    # Real cocoindex SentenceTransformerEmbedder (thread-safe, VectorSchemaProvider)
    builder.provide(EMBEDDER, CocoSentenceTransformerEmbedder(config.EMBEDDING_MODEL))
    db_path = os.getenv("POCKET_SQLITE_DB") or str(config.POCKET_SQLITE_DB)
    builder.provide_with(SQLITE_DB, sqlite.managed_connection(db_path, load_vec=True))
    yield


def _chunk_file(raw_text: str, filename: pathlib.PurePath) -> List[Chunk]:
    """Refine + split using cocoindex ops.

    detect_code_language and RecursiveSplitter come from cocoindex.ops.text;
    TextRefiner remains pocketindex (no cocoindex equivalent).
    """
    language = coco_detect_code_language(filename=filename.name)
    is_code = language is not None and language not in ("markdown", "html")
    refined = _refiner.refine(raw_text, code=is_code)
    # cocoindex RecursiveSplitter has the same split() contract
    chunks = _splitter.split(
        refined.text, chunk_size=1000, chunk_overlap=200, language=language
    )
    for chunk in chunks:
        chunk.start.char_offset = refined.source_offset(chunk.start.char_offset)
        chunk.end.char_offset = refined.source_offset(chunk.end.char_offset)
    return chunks


@pix.fn
async def process_chunk(
    chunk: Chunk,
    embedder: CocoSentenceTransformerEmbedder,
    file_path: str,
) -> ChunkEmbedding:
    """Embed a single chunk with the cocoindex embedder."""
    # CocoSentenceTransformerEmbedder exposes .embed() just like the pocketindex wrapper
    vector = await embedder.embed(chunk.text)
    return ChunkEmbedding(
        id=IdGenerator.from_text(chunk.text),
        file_path=file_path,
        text=chunk.text,
        embedding=vector,
        start_offset=chunk.start.char_offset,
        end_offset=chunk.end.char_offset,
    )


async def app_main(
    sourcedir: pathlib.Path,
    db_path: pathlib.Path,
    graph: bool = False,
) -> None:
    """Pipeline main — cocoindex ops, pocketindex engine.

    Signature matches pipeline.py so cli.py can route to this transparently.
    """
    embedder = pix.use_context(EMBEDDER)
    db = pix.use_context(SQLITE_DB)

    target = sqlite.TableTarget(
        conn=db,
        table_name="embeddings",
        schema=await sqlite.TableSchema.from_class(
            ChunkEmbedding, primary_key=["id"]
        ),
        fts_text_column="text",
    )

    fs = localfs.walk_dir(sourcedir)

    @pix.fn(memo=True)
    async def process_file(file: FileLike) -> None:
        raw_text = file.read_text()
        filename = pathlib.PurePath(file.path)
        chunks = _chunk_file(raw_text, filename)
        rel_path = str(pathlib.Path(file.path).relative_to(sourcedir))
        embeddings = await pix.map(
            lambda c: process_chunk(c, embedder, rel_path), chunks
        )
        for emb in embeddings:
            target.declare_row(emb.id, emb)

    await pix.mount_each(process_file, fs.items(), target)

    if graph:
        from pocketindex.ops.extract import build_extractor
        from pocketindex.ops.entity_resolution import resolve_entities
        extractor = build_extractor()

        entity_target = sqlite.TableTarget(
            conn=db,
            table_name="entities",
            schema=await sqlite.TableSchema.from_class(
                EntityNode, primary_key=["id"]
            ),
        )
        relation_target = sqlite.TableTarget(
            conn=db,
            table_name="relations",
            schema=await sqlite.TableSchema.from_class(
                RelationEdge, primary_key=["id"]
            ),
        )

        @pix.fn(memo=True)
        async def extract_graph(file: FileLike) -> None:
            raw_text = file.read_text()
            filename = pathlib.PurePath(file.path)
            chunks = _chunk_file(raw_text, filename)
            rel_path = str(pathlib.Path(file.path).relative_to(sourcedir))
            all_entities: list = []
            all_relations: list = []
            for chunk in chunks:
                result = extractor.extract(chunk.text)
                all_entities.extend(result.entities)
                all_relations.extend(result.relations)
            resolved = resolve_entities(all_entities)
            for ent in resolved:
                vec = await embedder.embed(ent.name)
                entity_target.declare_row(
                    IdGenerator.from_text(ent.name + ent.type),
                    EntityNode(
                        id=IdGenerator.from_text(ent.name + ent.type),
                        name=ent.name,
                        type=ent.type,
                        aliases=json.dumps(list(ent.aliases)),
                        embedding=vec,
                        summary="",
                        confidence=ent.confidence,
                        source_file=rel_path,
                        source_chunk_ids="[]",
                    ),
                )

        await pix.mount_each(extract_graph, fs.items(), entity_target)
