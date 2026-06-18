# Sprint 1 Action Plan (Ready for Execution)

**Sprint Goal:** Build a fully functional local-first incremental indexing pipeline that reads markdown files from a local directory, chunks them, generates embeddings using a local model, and stores them in a local SQLite/LanceDB database with full lineage tracking.

---

## Sprint Backlog Items

| Task ID | Title | Estimate (SP) | Assignee | Status | Dependencies |
|---|---|---|---|---|---|
| **POCKET-101** | Project Initialization & CLI Skeleton | 1 | Developer | Ready | None |
| **POCKET-102** | Local Filesystem Source Connector | 2 | Developer | Ready | POCKET-101 |
| **POCKET-103** | SQLite / LanceDB Target Setup | 2 | Developer | Ready | POCKET-101 |
| **POCKET-104** | Incremental Chunking & Embedding Pipeline | 3 | Developer | Ready | POCKET-102, POCKET-103 |
| **POCKET-105** | Lineage Metadata Storage | 1 | Developer | Ready | POCKET-104 |

---

## Detailed Task Breakdown

### POCKET-101: Project Initialization & CLI Skeleton
- **Objective:** Set up the Python project structure using `pyproject.toml` and configure dependencies.
- **Deliverables:**
  - `pyproject.toml` with `pocketindex`, `sqlite-vec` (or `lancedb`), `sentence-transformers`, and `click`/`typer` for CLI.
  - `.env` file containing `POCKET_SQLITE_DB=./.pocket/pocket_data.db`.
  - `pocket/cli.py` with commands: `pocket init`, `pocket update`, and `pocket search`.

### POCKET-102: Local Filesystem Source Connector
- **Objective:** Configure CocoIndex to watch a local directory (e.g., `./notes`) for changes.
- **Deliverables:**
  - Integration of `localfs.walk_dir` in the main CocoIndex app.
  - Support for both catch-up mode (`pocket update`) and live mode (`pocket update -L`).

### POCKET-103: SQLite / LanceDB Target Setup
- **Objective:** Define the target schema for storing chunk embeddings and metadata.
- **Deliverables:**
  - Dataclass `ChunkEmbedding` representing the target schema:
    ```python
    @dataclass
    class ChunkEmbedding:
        id: int
        file_path: str
        text: str
        embedding: Annotated[NDArray, EMBEDDER]
        start_offset: int
        end_offset: int
    ```
  - Database connection setup in `@pix.lifespan`.
  - Table target mounting using `sqlite.mount_table_target` or `lancedb.mount_table_target`.

### POCKET-104: Incremental Chunking & Embedding Pipeline
- **Objective:** Implement the core transformation function that chunks files and generates embeddings.
- **Deliverables:**
  - `@pix.fn(memo=True)` decorated `process_file` function.
  - Use of `RecursiveSplitter` to split markdown text into chunks.
  - Use of `SentenceTransformerEmbedder` (e.g., `all-MiniLM-L6-v2`) to generate embeddings.
  - Deterministic ID generation using `IdGenerator` to ensure stable IDs across runs.

### POCKET-105: Lineage Metadata Storage
- **Objective:** Ensure that every chunk stored in the database contains precise lineage information.
- **Deliverables:**
  - Store `file_path`, `start_offset`, and `end_offset` in the `ChunkEmbedding` row.
  - Verify that querying the database returns the exact source file and character range for any chunk.

---

## Definition of Done (DoD)
1. All code is written in Python and adheres to the CocoIndex v1 API.
2. The pipeline can be executed via the CLI: `pocket update`.
3. Modifying a file and running `pocket update` only reprocesses the modified file (verified via logs/memoization).
4. Deleting a source file automatically removes its corresponding chunks from the database.
5. A basic search command `pocket search "query"` returns matching chunks along with their source file path and character offsets.
