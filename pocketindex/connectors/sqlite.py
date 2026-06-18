"""SQLite target connector for PocketIndex."""
import sqlite3
import sqlite_vec
import pathlib
from typing import Any, Dict, List, Type, get_type_hints, Annotated
import numpy as np

class ManagedConnection:
    def __init__(self, db_path: str, load_vec: bool = True):
        self.db_path = db_path
        self.load_vec = load_vec
        self.conn = None

    async def __aenter__(self):
        # Ensure parent directory exists
        pathlib.Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        if self.load_vec:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            self.conn = None

    def execute(self, sql: str, params: tuple = ()):
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq_of_params):
        return self.conn.executemany(sql, seq_of_params)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

def managed_connection(db_path: str, load_vec: bool = True) -> ManagedConnection:
    return ManagedConnection(db_path, load_vec)

class TableSchema:
    def __init__(self, fields: Dict[str, Type], primary_key: List[str]):
        self.fields = fields
        self.primary_key = primary_key

    @classmethod
    async def from_class(cls, klass: Type, primary_key: List[str]) -> "TableSchema":
        # Extract fields and types from the dataclass's type hints.
        hints = get_type_hints(klass, include_extras=True)
        return cls(dict(hints), primary_key)

class TableTarget:
    # Marker so the pocketindex engine recognises this as a lineage-aware target.
    _is_pix_target = True

    def __class_getitem__(cls, item):
        return cls

    def __init__(self, conn: ManagedConnection, table_name: str, schema: TableSchema,
                 fts_text_column: str = None):
        self.conn = conn
        self.table_name = table_name
        self.schema = schema
        # Single-column primary key is assumed for lineage tracking.
        self._pk = schema.primary_key[0]
        # Per-source set of primary-key values emitted during the current run.
        self._emitted: Dict[str, set] = {}
        # Optional lexical (FTS5) companion index for hybrid retrieval. When a
        # text column is named, every declared row is mirrored into an external
        # FTS5 table keyed by the primary key, so the same target supports both
        # vector (sqlite-vec) and lexical (BM25) search from one declaration.
        self._fts_text_column = fts_text_column if fts_text_column in schema.fields else None
        self._create_table()
        self._create_state_tables()
        if self._fts_text_column is not None:
            self._create_fts_table()


    def _create_table(self):
        # Create the table if it doesn't exist
        cols = []
        for name, hint in self.schema.fields.items():
            # Determine SQLite type
            origin = getattr(hint, "__origin__", None)
            if origin is Annotated:
                # Check if it's an embedding/vector
                # e.g., Annotated[NDArray, EMBEDDER]
                # We store it as a BLOB or float32 vector
                # sqlite-vec uses float32 vector (which is a BLOB in SQLite)
                col_type = "BLOB"
            elif hint is int:
                col_type = "INTEGER"
            elif hint is str:
                col_type = "TEXT"
            elif hint is float:
                col_type = "REAL"
            else:
                col_type = "BLOB"
            
            if name in self.schema.primary_key:
                cols.append(f"{name} {col_type} PRIMARY KEY")
            else:
                cols.append(f"{name} {col_type}")
        
        sql = f"CREATE TABLE IF NOT EXISTS {self.table_name} ({', '.join(cols)})"
        self.conn.execute(sql)
        self.conn.commit()

    # ----- Lineage / memoization state (self-contained, no external deps) -----
    @property
    def _lineage_table(self) -> str:
        return f"_pocket_lineage_{self.table_name}"

    @property
    def _memo_table(self) -> str:
        return f"_pocket_memo_{self.table_name}"

    @property
    def _fts_table(self) -> str:
        return f"_pocket_fts_{self.table_name}"

    def _create_fts_table(self):
        # External-content-free FTS5 index: we store the primary key as an
        # unindexed column alongside the searchable text so we can join back to
        # the main table (and its embeddings/lineage) after a BM25 match.
        try:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts_table} "
                f"USING fts5(row_id UNINDEXED, content)"
            )
            self.conn.commit()
        except sqlite3.OperationalError as exc:
            # FTS5 missing in this SQLite build: degrade to vector-only loading
            # rather than breaking the whole pipeline.
            self._fts_text_column = None
            print(f"[pocketindex.sqlite] FTS5 unavailable, lexical index disabled: {exc}")

    def _fts_delete_rows(self, row_ids: set) -> None:
        if self._fts_text_column is None or not row_ids:
            return
        self.conn.executemany(
            f"DELETE FROM {self._fts_table} WHERE row_id = ?",
            [(rid,) for rid in row_ids],
        )

    def _fts_upsert(self, row_id, content: str) -> None:
        if self._fts_text_column is None:
            return
        # FTS5 has no UPSERT; delete-then-insert keeps it idempotent.
        self.conn.execute(
            f"DELETE FROM {self._fts_table} WHERE row_id = ?", (row_id,)
        )
        self.conn.execute(
            f"INSERT INTO {self._fts_table} (row_id, content) VALUES (?, ?)",
            (row_id, content),
        )

    def _create_state_tables(self):
        # Maps each source item to the primary-key values it produced, so that
        # rows can be reconciled on edit and removed when the source disappears.
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._lineage_table} ("
            f"source_key TEXT NOT NULL, row_id NOT NULL, "
            f"PRIMARY KEY (source_key, row_id))"
        )
        # Stores the content fingerprint of each source item for memoization.
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._memo_table} ("
            f"source_key TEXT PRIMARY KEY, content_hash TEXT)"
        )
        self.conn.commit()

    def get_memo(self, source_key: str):
        cur = self.conn.execute(
            f"SELECT content_hash FROM {self._memo_table} WHERE source_key = ?",
            (source_key,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def begin_source(self, source_key: str) -> None:
        # Start accumulating the ids emitted for this source during this run.
        self._emitted[source_key] = set()

    def abort_source(self, source_key: str) -> None:
        # Discard uncommitted rows emitted for a source that failed mid-run.
        self._emitted.pop(source_key, None)
        self.conn.rollback()

    def _old_row_ids(self, source_key: str) -> set:
        cur = self.conn.execute(
            f"SELECT row_id FROM {self._lineage_table} WHERE source_key = ?",
            (source_key,),
        )
        return {r[0] for r in cur.fetchall()}

    def _delete_rows(self, row_ids: set) -> None:
        if not row_ids:
            return
        self.conn.executemany(
            f"DELETE FROM {self.table_name} WHERE {self._pk} = ?",
            [(rid,) for rid in row_ids],
        )
        # Keep the lexical index in lockstep with the primary table.
        self._fts_delete_rows(row_ids)

    def end_source(self, source_key: str, content_hash: str) -> None:
        """Reconcile this source: drop stale chunks, persist lineage + memo."""
        new_ids = self._emitted.pop(source_key, set())
        old_ids = self._old_row_ids(source_key)

        # Chunks that existed before but were not re-emitted are now orphans.
        stale = old_ids - new_ids
        if stale:
            self._delete_rows(stale)

        # Rewrite lineage for this source to exactly the new id set.
        self.conn.execute(
            f"DELETE FROM {self._lineage_table} WHERE source_key = ?", (source_key,)
        )
        if new_ids:
            self.conn.executemany(
                f"INSERT OR IGNORE INTO {self._lineage_table} (source_key, row_id) "
                f"VALUES (?, ?)",
                [(source_key, rid) for rid in new_ids],
            )

        # Persist the memo fingerprint for the incremental fast path.
        if content_hash:
            self.conn.execute(
                f"INSERT INTO {self._memo_table} (source_key, content_hash) "
                f"VALUES (?, ?) ON CONFLICT(source_key) DO UPDATE SET "
                f"content_hash = excluded.content_hash",
                (source_key, content_hash),
            )
        self.conn.commit()

    def sweep(self, live_source_keys: set) -> None:
        """Remove all target rows whose source items no longer exist."""
        cur = self.conn.execute(
            f"SELECT DISTINCT source_key FROM {self._lineage_table}"
        )
        known = {r[0] for r in cur.fetchall()}
        removed = known - {str(k) for k in live_source_keys}
        for source_key in removed:
            self._delete_rows(self._old_row_ids(source_key))
            self.conn.execute(
                f"DELETE FROM {self._lineage_table} WHERE source_key = ?",
                (source_key,),
            )
            self.conn.execute(
                f"DELETE FROM {self._memo_table} WHERE source_key = ?",
                (source_key,),
            )
        self.conn.commit()

    def declare_row(self, row: Any) -> None:
        # Upsert the row into the table
        fields = self.schema.fields
        cols = []
        vals = []
        placeholders = []
        
        for name, hint in fields.items():
            val = getattr(row, name)
            # If it's a numpy array (embedding), serialize it for sqlite-vec
            if isinstance(val, np.ndarray):
                val = sqlite_vec.serialize_float32(val)
            cols.append(name)
            vals.append(val)
            placeholders.append("?")
            
        # SQLite UPSERT syntax
        pk_cols = ", ".join(self.schema.primary_key)
        update_cols = [f"{c} = excluded.{c}" for c in cols if c not in self.schema.primary_key]
        
        if update_cols:
            sql = f"""
                INSERT INTO {self.table_name} ({', '.join(cols)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT({pk_cols}) DO UPDATE SET {', '.join(update_cols)}
            """
        else:
            sql = f"""
                INSERT OR IGNORE INTO {self.table_name} ({', '.join(cols)})
                VALUES ({', '.join(placeholders)})
            """
            
        self.conn.execute(sql, tuple(vals))

        # Mirror the searchable text into the lexical (FTS5) index so the same
        # declared row is retrievable by both vector and keyword search.
        if self._fts_text_column is not None:
            self._fts_upsert(getattr(row, self._pk), getattr(row, self._fts_text_column))

        # Attribute this row to the source item currently being processed so
        # the engine can reconcile/sweep it later.
        from pocketindex import get_current_source_key

        source_key = get_current_source_key()
        if source_key is not None:
            self._emitted.setdefault(source_key, set()).add(getattr(row, self._pk))

async def mount_table_target(
    sqlite_db_key: Any,
    table_name: str,
    table_schema: TableSchema,
    fts_text_column: str = None,
) -> TableTarget:
    # Retrieve connection from context
    from pocketindex import use_context
    conn = use_context(sqlite_db_key)
    return TableTarget(conn, table_name, table_schema, fts_text_column=fts_text_column)
