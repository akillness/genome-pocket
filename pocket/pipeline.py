import pathlib
from dataclasses import dataclass
from typing import Annotated, AsyncIterator
import numpy as np
from numpy.typing import NDArray

import cocoindex as coco
from cocoindex.connectors import localfs, sqlite
from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder
from cocoindex.resources.file import FileLike
from cocoindex.ops.text import RecursiveSplitter
from cocoindex.resources.id import IdGenerator
from cocoindex.resources.chunk import Chunk

from pocket.config import POCKET_SOURCE_DIR, POCKET_SQLITE_DB, EMBEDDING_MODEL

# Context key for the embedding model
EMBEDDER = coco.ContextKey[SentenceTransformerEmbedder]("embedder")
SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection]("sqlite_db")

@dataclass
class ChunkEmbedding:
    id: int                                   # Stable unique ID generated from chunk text
    file_path: str                            # Relative path to the source file
    text: str                                 # The text content of the chunk
    embedding: Annotated[NDArray, EMBEDDER]   # Vector embedding (dimensions inferred from model)
    start_offset: int                         # Character start offset in the source file
    end_offset: int                           # Character end offset in the source file

_splitter = RecursiveSplitter()

@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    # Provide the SentenceTransformerEmbedder
    builder.provide(EMBEDDER, SentenceTransformerEmbedder(EMBEDDING_MODEL))
    # Provide the SQLite ManagedConnection
    builder.provide_with(SQLITE_DB, sqlite.managed_connection(POCKET_SQLITE_DB, load_vec=True))
    yield

@coco.fn
async def process_chunk(
    chunk: Chunk,
    filename: pathlib.PurePath,
    id_gen: IdGenerator,
    table: sqlite.TableTarget[ChunkEmbedding],
) -> None:
    embedder = coco.use_context(EMBEDDER)
    embedding = await embedder.embed(chunk.text)
    
    table.declare_row(row=ChunkEmbedding(
        id=await id_gen.next_id(chunk.text),
        file_path=str(filename),
        text=chunk.text,
        embedding=embedding,
        start_offset=chunk.start.char_offset,
        end_offset=chunk.end.char_offset,
    ))

@coco.fn(memo=True)
async def process_file(file: FileLike, table: sqlite.TableTarget[ChunkEmbedding]) -> None:
    text = await file.read_text()
    chunks = _splitter.split(text, chunk_size=1000, chunk_overlap=200)
    id_gen = IdGenerator()
    
    await coco.map(process_chunk, chunks, file.file_path.path, id_gen, table)

@coco.fn
async def app_main(sourcedir: pathlib.Path, db_path: pathlib.Path) -> None:
    # 1. Mount SQLite target table
    target_table = await sqlite.mount_table_target(
        SQLITE_DB,
        table_name="embeddings",
        table_schema=await sqlite.TableSchema.from_class(ChunkEmbedding, primary_key=["id"]),
    )
    
    # 2. Walk source directory
    files = localfs.walk_dir(sourcedir, recursive=True, live=True)
    
    # 3. Mount file processing component
    await coco.mount_each(process_file, files.items(), target_table)
