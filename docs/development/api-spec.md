# API & Interface Specifications

This document defines the core Python interfaces and data schemas used throughout the **Pocket** codebase.

---

## Data Schemas

### 1. `ChunkEmbedding`
Represents a single text chunk, its vector embedding, and its source lineage. This schema is used for the target database table.

```python
from dataclasses import dataclass
from typing import Annotated
from numpy.typing import NDArray
import cocoindex as coco

# Context key for the embedding model
EMBEDDER = coco.ContextKey["SentenceTransformerEmbedder"]("embedder")

@dataclass
class ChunkEmbedding:
    id: int                                   # Stable unique ID generated from chunk text
    file_path: str                            # Relative path to the source file
    text: str                                 # The text content of the chunk
    embedding: Annotated[NDArray, EMBEDDER]   # Vector embedding (dimensions inferred from model)
    start_offset: int                         # Character start offset in the source file
    end_offset: int                           # Character end offset in the source file
```

### 2. `ConceptRelation`
Represents a relationship between two concepts extracted from notes. This schema is used for the graph database target.

```python
@dataclass
class ConceptRelation:
    from_concept: str                         # Source concept name (e.g., "CocoIndex")
    to_concept: str                           # Target concept name (e.g., "Incremental ETL")
    relation_type: str                        # Type of relationship (e.g., "is_a", "depends_on")
    source_file: str                          # File path where the relationship was found
```

---

## Core Interfaces

### 1. Pipeline Main Function
The entry point for the CocoIndex application. It mounts the target tables and starts the filesystem walker.

```python
import pathlib
import cocoindex as coco
from cocoindex.connectors import localfs, sqlite

@coco.fn
async def app_main(sourcedir: pathlib.Path, db_path: pathlib.Path) -> None:
    # 1. Mount SQLite target table
    conn = sqlite.connect(db_path, load_vec="auto")
    sqlite_db = coco.ContextKey[sqlite.ManagedConnection]("sqlite_db")
    # Provide connection in the current context
    # (Note: usually provided in lifespan, but can be mounted directly)
    
    target_table = await sqlite.mount_table_target(
        sqlite_db,
        table_name="embeddings",
        table_schema=await sqlite.TableSchema.from_class(ChunkEmbedding, primary_key=["id"]),
    )
    
    # 2. Walk source directory
    files = localfs.walk_dir(sourcedir, recursive=True, live=True)
    
    # 3. Mount file processing component
    await coco.mount_each(process_file, files.items(), target_table)
```

### 2. File Processing Component
Processes a single file, splits it into chunks, and maps them to the target table.

```python
from cocoindex.resources.file import FileLike
from cocoindex.ops.text import RecursiveSplitter
from cocoindex.resources.id import IdGenerator

_splitter = RecursiveSplitter()

@coco.fn(memo=True)
async def process_file(file: FileLike, table: sqlite.TableTarget[ChunkEmbedding]) -> None:
    text = await file.read_text()
    chunks = _splitter.split(text, chunk_size=1000, chunk_overlap=200)
    id_gen = IdGenerator()
    
    await coco.map(process_chunk, chunks, file.file_path.path, id_gen, table)
```

### 3. Chunk Processing Component
Generates the embedding and declares the row in the target table.

```python
from cocoindex.resources.chunk import Chunk

@coco.fn
async def process_chunk(
    chunk: Chunk, filename: pathlib.PurePath,
    id_gen: IdGenerator, table: sqlite.TableTarget[ChunkEmbedding],
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
```
