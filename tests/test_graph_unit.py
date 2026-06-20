"""Graph ops unit tests — DeterministicExtractor + in-memory SQLite.

Covers T4 from cocoindex-gap.md: graph tests that do NOT require a real
pipeline run (`app.update_blocking`).  All extraction, resolution, and
target-table logic is exercised directly against an in-memory SQLite
database, no disk I/O, no model download.
"""
import asyncio
import sqlite3
import tempfile
import pathlib
import unittest

from pocketindex.ops.extract import (
    DeterministicExtractor,
    ExtractedEntity,
    build_extractor,
    parse_extraction_json,
)
from pocketindex.ops.entity_resolution import resolve_entities


class TestGraphExtraction(unittest.TestCase):
    """Pure-unit tests for extraction ops — no DB, no pipeline."""

    def test_deterministic_extractor_finds_entities_and_relations(self):
        ex = DeterministicExtractor()
        out = ex.extract(
            "Pocket uses SQLite for storage. SQLite powers the Pocket index."
        )
        names = {e.name for e in out.entities}
        self.assertIn("Pocket", names)
        self.assertIn("SQLite", names)
        self.assertTrue(
            any(
                r.predicate == "mentioned_with"
                and {r.subject, r.object} == {"Pocket", "SQLite"}
                for r in out.relations
            )
        )
        for e in out.entities:
            self.assertGreaterEqual(e.confidence, 0.0)
            self.assertTrue(e.evidence)

    def test_build_extractor_defaults_to_deterministic(self):
        self.assertIsInstance(build_extractor(), DeterministicExtractor)
        self.assertIsInstance(
            build_extractor(provider="nope"), DeterministicExtractor
        )

    def test_parse_extraction_json_validates_and_normalizes(self):
        ext = parse_extraction_json(
            '```json\n{"entities":[{"name":" Foo ","type":"Tool","confidence":2}],'
            '"relations":[{"subject":"Foo","predicate":"Depends On",'
            '"object":"Bar","confidence":0.7}]}\n```',
            evidence="ctx",
        )
        self.assertEqual(ext.entities[0].name, "Foo")
        self.assertEqual(ext.entities[0].confidence, 1.0)
        self.assertEqual(ext.relations[0].predicate, "depends_on")

    def test_parse_extraction_json_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_extraction_json("not json at all", evidence="ctx")

    def test_entity_resolution_blocks_and_propagates(self):
        ents = [
            ExtractedEntity(name="SQLite", type="Tool", confidence=0.9),
            ExtractedEntity(name="sqlite", type="Tool", confidence=0.5),
            ExtractedEntity(name="Postgres", type="Tool", confidence=0.8),
        ]
        embeds = [[1.0, 0.0, 0.0], [0.99, 0.0, 0.0], [0.0, 1.0, 0.0]]
        resolved = resolve_entities(ents, embeds)
        self.assertEqual(len(resolved), 2)
        sqlite_cluster = next(r for r in resolved if r.name.lower() == "sqlite")
        self.assertEqual(sqlite_cluster.name, "SQLite")
        self.assertIn("sqlite", sqlite_cluster.aliases)

    def test_entity_resolution_adjudicator_is_optional_and_used(self):
        ents = [
            ExtractedEntity(
                name="Knowledge Graph", type="Concept", confidence=0.7
            ),
            ExtractedEntity(name="KG", type="Concept", confidence=0.7),
        ]
        embeds = [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]]
        self.assertEqual(len(resolve_entities(ents, embeds)), 2)
        merged = resolve_entities(ents, embeds, adjudicator=lambda a, b: True)
        self.assertEqual(len(merged), 1)


class TestGraphTargetUnit(unittest.TestCase):
    """Graph target wiring with in-memory SQLite — no real pipeline run."""

    def test_extract_graph_file_in_memory(self):
        """DeterministicExtractor → entities/relations targets, in-memory DB."""
        import sqlite_vec
        import pocketindex as pix
        from pocketindex.connectors import sqlite
        from pocketindex.resources.file import FileLike
        from pocket.pipeline import EntityNode, RelationEdge, extract_graph_file, EMBEDDER
        from tests.conftest import MockEmbedder

        async def _run():
            # In-memory DB with sqlite-vec loaded.
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            mconn = sqlite.ManagedConnection(":memory:", load_vec=False)
            mconn.conn = conn

            # Provide context keys.
            db_key = pix.ContextKey[sqlite.ManagedConnection]("sqlite_db")
            pix._CONTEXT[db_key.name] = mconn
            pix._CONTEXT[EMBEDDER.name] = MockEmbedder()

            try:
                entities_target = await sqlite.mount_table_target(
                    db_key,
                    table_name="entities",
                    table_schema=await sqlite.TableSchema.from_class(
                        EntityNode, primary_key=["id"]
                    ),
                    fts_text_column="name",
                )
                relations_target = await sqlite.mount_table_target(
                    db_key,
                    table_name="relations",
                    table_schema=await sqlite.TableSchema.from_class(
                        RelationEdge, primary_key=["id"]
                    ),
                )

                # Write a temp file for FileLike to read.
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False
                ) as f:
                    f.write(
                        "Pocket uses SQLite for storage. "
                        "SQLite powers the Pocket index."
                    )
                    temp_path = pathlib.Path(f.name)

                try:
                    file_like = FileLike(temp_path)
                    pix._current_source_key.set(str(temp_path))
                    entities_target.begin_source(str(temp_path))

                    await extract_graph_file(
                        file_like, entities_target, relations_target
                    )

                    entities_target.sweep({str(temp_path)})
                    relations_target.sweep({str(temp_path)})

                    # Assertions.
                    names = {
                        r[0]
                        for r in conn.execute(
                            "SELECT name FROM entities"
                        ).fetchall()
                    }
                    assert "Pocket" in names, f"expected Pocket in {names}"
                    assert "SQLite" in names, f"expected SQLite in {names}"
                    rel_count = conn.execute(
                        "SELECT COUNT(*) FROM relations"
                    ).fetchone()[0]
                    assert rel_count > 0, "expected at least one relation"
                finally:
                    temp_path.unlink(missing_ok=True)
                    conn.close()
            finally:
                pix._CONTEXT.pop(db_key.name, None)
                pix._CONTEXT.pop(EMBEDDER.name, None)

        asyncio.run(_run())

class TestExtractionPromptAndMemo(unittest.TestCase):
    """POCKET-404b: hardened JSON prompt + per-(chunk, model, prompt) memo."""

    def test_prompt_version_and_hardened_prompt(self):
        from pocketindex.ops.extract import PROMPT_VERSION, _EXTRACTION_PROMPT

        self.assertTrue(isinstance(PROMPT_VERSION, str) and PROMPT_VERSION)
        # The hardened prompt must pin the strict-JSON schema, grounding, evidence,
        # and calibrated confidence directives the 2026 literature calls for.
        for token in ("entities", "relations", "evidence", "confidence", "ONLY"):
            self.assertIn(token, _EXTRACTION_PROMPT)

    def test_memoizing_extractor_caches_by_text(self):
        from pocketindex.ops.extract import (
            Extraction,
            ExtractedEntity,
            MemoizingExtractor,
        )

        class CountingExtractor:
            name = "counting"
            model = "fake-model"

            def __init__(self):
                self.calls = 0

            def extract(self, text):
                self.calls += 1
                return Extraction(entities=[ExtractedEntity(name=text[:5])])

        inner = CountingExtractor()
        memo = MemoizingExtractor(inner)

        first = memo.extract("Pocket indexes notes locally.")
        second = memo.extract("Pocket indexes notes locally.")
        self.assertEqual(inner.calls, 1)          # second call served from cache
        self.assertEqual(memo.misses, 1)
        self.assertEqual(first.entities[0].name, second.entities[0].name)

        memo.extract("A different chunk of text.")
        self.assertEqual(inner.calls, 2)          # new text => cache miss

    def test_memo_key_separates_model_and_prompt_version(self):
        from pocketindex.ops.extract import extraction_cache_key

        base = extraction_cache_key("chunk", "model-a", "2026.1")
        self.assertNotEqual(base, extraction_cache_key("chunk", "model-b", "2026.1"))
        self.assertNotEqual(base, extraction_cache_key("chunk", "model-a", "2026.2"))
        self.assertEqual(base, extraction_cache_key("chunk", "model-a", "2026.1"))

    def test_sqlite_extraction_store_persists_across_instances(self):
        from pocketindex.ops.extract import SqliteExtractionStore

        conn = sqlite3.connect(":memory:")
        try:
            store = SqliteExtractionStore(conn)
            store.set("k", '{"entities": [], "relations": []}')
            # A fresh store on the same connection sees the persisted row.
            reopened = SqliteExtractionStore(conn)
            self.assertEqual(reopened.get("k"), '{"entities": [], "relations": []}')
            self.assertIsNone(reopened.get("missing"))
        finally:
            conn.close()

    def test_build_extractor_wraps_llm_backends_only(self):
        from pocketindex.ops.extract import (
            DeterministicExtractor,
            MemoizingExtractor,
            OllamaExtractor,
            build_extractor,
        )

        self.assertIsInstance(build_extractor(provider="ollama"), MemoizingExtractor)
        self.assertIsInstance(
            build_extractor(provider="ollama", memo=False), OllamaExtractor
        )
        # Deterministic stays unwrapped (pure, cheap, offline default).
        self.assertIsInstance(build_extractor(provider="deterministic"), DeterministicExtractor)
