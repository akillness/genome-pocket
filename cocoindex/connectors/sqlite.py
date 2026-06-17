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
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, conn: ManagedConnection, table_name: str, schema: TableSchema):
        self.conn = conn
        self.table_name = table_name
        self.schema = schema
        self._create_table()


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

async def mount_table_target(
    sqlite_db_key: Any,
    table_name: str,
    table_schema: TableSchema,
) -> TableTarget:
    # Retrieve connection from context
    from cocoindex import use_context
    conn = use_context(sqlite_db_key)
    return TableTarget(conn, table_name, table_schema)
