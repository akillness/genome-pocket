# ADR 002: Vector & Graph Database Selection

## Status
Proposed

## Context
Pocket requires two types of storage to support hybrid retrieval:
1. **Vector Storage:** To store chunk embeddings and perform semantic similarity search.
2. **Graph Storage:** To store concepts (nodes) and relationships (edges) extracted from notes and code, enabling relational queries.

We need to select database technologies that align with our local-first, low-overhead, and zero-configuration goals.

## Decision
We will use the following database technologies for Pocket:

1. **Vector Database: SQLite with `sqlite-vec`**
   - *Why:* SQLite is pre-installed on almost all systems, requires zero configuration, and stores data in a single file. The `sqlite-vec` extension adds fast, lightweight vector search capabilities directly to SQLite.
   - *Alternative:* LanceDB. We will support LanceDB as an alternative target for larger datasets or cloud-native storage (S3/GCS) use cases.

2. **Graph Database: SurrealDB (Local Mode)**
   - *Why:* SurrealDB is a multi-model database that supports both document and graph models. It can run in-memory or persist to a local file (using RocksDB or SpeeDB) without requiring a separate server process. It also integrates well with CocoIndex via the `surrealdb` connector.
   - *Alternative:* Neo4j. While Neo4j is a powerful graph database, running it locally requires Docker or a Java runtime, which increases setup complexity for a personal tool.

## Consequences

### Pros
- **Zero Configuration:** Users do not need to install or manage database servers (like PostgreSQL or Neo4j).
- **Single-File Storage:** Both SQLite and SurrealDB can persist to local files in the `.pocket/` directory, making backups and migrations simple.
- **Lightweight:** Minimal memory and CPU overhead when idle.

### Cons
- **Scalability Limits:** SQLite and local SurrealDB are optimized for single-user workloads. For enterprise-scale deployments, we will allow users to configure external PostgreSQL (with pgvector) and Neo4j databases.
- **Extension Support:** `sqlite-vec` requires loading a dynamic library, which can fail on locked-down environments or older Python installations. We will provide a fallback to in-memory numpy-based vector search if the extension cannot be loaded.
