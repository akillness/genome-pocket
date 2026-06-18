import pathlib
from dataclasses import dataclass
from typing import Annotated, AsyncIterator
import numpy as np
from numpy.typing import NDArray

import pocketindex as pix
from pocketindex.connectors import localfs, sqlite
from pocketindex.ops.sentence_transformers import SentenceTransformerEmbedder
from pocketindex.resources.file import FileLike
from pocketindex.ops.text import RecursiveSplitter
from pocketindex.ops.refine import TextRefiner
from pocketindex.resources.id import IdGenerator
from pocketindex.resources.chunk import Chunk

import pocket.config as config


# Context key for the embedding model
EMBEDDER = pix.ContextKey[SentenceTransformerEmbedder]("embedder")
SQLITE_DB = pix.ContextKey[sqlite.ManagedConnection]("sqlite_db")

@dataclass
class ChunkEmbedding:
    id: int                                   # Stable unique ID generated from chunk text
    file_path: str                            # Relative path to the source file
    text: str                                 # The text content of the chunk
    embedding: Annotated[NDArray, EMBEDDER]   # Vector embedding (dimensions inferred from model)
    start_offset: int                         # Character start offset in the source file
    end_offset: int                           # Character end offset in the source file

_splitter = RecursiveSplitter()
_refiner = TextRefiner()

@pix.lifespan
async def pocket_lifespan(builder: pix.EnvironmentBuilder) -> AsyncIterator[None]:
    import os
    # Provide the SentenceTransformerEmbedder
    builder.provide(EMBEDDER, SentenceTransformerEmbedder(config.EMBEDDING_MODEL))
    # Provide the SQLite ManagedConnection
    db_path = os.getenv("POCKET_SQLITE_DB") or str(config.POCKET_SQLITE_DB)
    builder.provide_with(SQLITE_DB, sqlite.managed_connection(db_path, load_vec=True))
    yield


@pix.fn
async def process_chunk(
    chunk: Chunk,
    filename: pathlib.PurePath,
    id_gen: IdGenerator,
    table: sqlite.TableTarget[ChunkEmbedding],
) -> None:
    embedder = pix.use_context(EMBEDDER)
    embedding = await embedder.embed(chunk.text)
    
    table.declare_row(row=ChunkEmbedding(
        id=await id_gen.next_id(chunk.text),
        file_path=str(filename),
        text=chunk.text,
        embedding=embedding,
        start_offset=chunk.start.char_offset,
        end_offset=chunk.end.char_offset,
    ))

@pix.fn(memo=True)
async def process_file(file: FileLike, table: sqlite.TableTarget[ChunkEmbedding]) -> None:
    raw_text = await file.read_text()
    # Refinement stage: normalize/clean the raw source before chunking. The
    # refined document keeps an offset map so chunk lineage still points at the
    # original source bytes the user can open.
    refined = _refiner.refine(raw_text)
    chunks = _splitter.split(refined.text, chunk_size=1000, chunk_overlap=200)
    # Translate each chunk's offsets from refined-text space back to original
    # source space so stored lineage references real file positions.
    for chunk in chunks:
        chunk.start.char_offset = refined.source_offset(chunk.start.char_offset)
        chunk.end.char_offset = refined.source_offset(chunk.end.char_offset)
    id_gen = IdGenerator()

    await pix.map(process_chunk, chunks, file.file_path.path, id_gen, table)

@pix.fn
async def app_main(sourcedir: pathlib.Path, db_path: pathlib.Path) -> None:
    # 1. Mount SQLite target table
    target_table = await sqlite.mount_table_target(
        SQLITE_DB,
        table_name="embeddings",
        table_schema=await sqlite.TableSchema.from_class(ChunkEmbedding, primary_key=["id"]),
        # Mirror chunk text into an FTS5 index so the same load supports both
        # vector (sqlite-vec) and lexical (BM25) retrieval.
        fts_text_column="text",
    )
    
    # 2. Walk source directory
    files = localfs.walk_dir(sourcedir, recursive=True, live=True)
    
    # 3. Mount file processing component
    await pix.mount_each(process_file, files.items(), target_table)
