"""SQLite target connector for CocoIndex."""
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

    def commit(self):
        self.conn.commit()

def managed_connection(db_path: str, load_vec: bool = True) -> ManagedConnection:
    return ManagedConnection(db_path, load_vec)

class TableSchema:
    def __init__(self, fields: Dict[str, Type], primary_key: List[str]):
        self.fields = fields
        self.primary_key = primary_key

    @classmethod
    async def from_class(cls, klass: Type, primary_key: List[str]) -> "TableSchema":
        # Extract fields and types from dataclass
        hints = get_type_hints(klass, include_extras=True)
        fields = {}
        for name, hint in hints.items():
            fields[name] = hint
        return cls(fields, primary_key)

class TableTarget:
    # Marker so the cocoindex engine recognises this as a lineage-aware target.
    _is_coco_target = True

    def __class_getitem__(cls, item):
        return cls

    def __init__(self, conn: ManagedConnection, table_name: str, schema: TableSchema):
        self.conn = conn
        self.table_name = table_name
        self.schema = schema
        # Single-column primary key is assumed for lineage tracking.
        self._pk = schema.primary_key[0]
        # Per-source set of primary-key values emitted during the current run.
        self._emitted: Dict[str, set] = {}
        self._create_table()
        self._create_state_tables()


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
        return f"_coco_lineage_{self.table_name}"

    @property
    def _memo_table(self) -> str:
        return f"_coco_memo_{self.table_name}"

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

    def _old_row_ids(self, source_key: str) -> set:
        cur = self.conn.execute(
            f"SELECT row_id FROM {self._lineage_table} WHERE source_key = ?",
            (source_key,),
        )
        return {r[0] for r in cur.fetchall()}

    def _delete_rows(self, row_ids: set) -> None:
        for rid in row_ids:
            self.conn.execute(
                f"DELETE FROM {self.table_name} WHERE {self._pk} = ?", (rid,)
            )

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
        for rid in new_ids:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {self._lineage_table} (source_key, row_id) "
                f"VALUES (?, ?)",
                (source_key, rid),
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
        self.conn.commit()

        # Attribute this row to the source item currently being processed so
        # the engine can reconcile/sweep it later.
        from cocoindex import get_current_source_key

        source_key = get_current_source_key()
        if source_key is not None:
            self._emitted.setdefault(source_key, set()).add(getattr(row, self._pk))

async def mount_table_target(
    sqlite_db_key: Any,
    table_name: str,
    table_schema: TableSchema,
) -> TableTarget:
    # Retrieve connection from context
    from cocoindex import use_context
    conn = use_context(sqlite_db_key)
    return TableTarget(conn, table_name, table_schema)
