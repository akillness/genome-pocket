"""Tests for the hybrid retrieval layer, refinement op, and REST API server."""
import importlib
import os
import pathlib
import sqlite3
import sys
import tempfile
import unittest

import pocketindex as pix


class TestRetrievalAndApi(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = pathlib.Path(self.temp_dir.name) / "notes"
        self.source_dir.mkdir()
        self.db_path = pathlib.Path(self.temp_dir.name) / "pocket_data.db"

        self.old_db_env = os.environ.get("POCKET_SQLITE_DB")
        os.environ["POCKET_SQLITE_DB"] = str(self.db_path)
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
        # retrieval/api_server import config at module load; reload so they see
        # the test DB path.
        for mod in ("pocket.retrieval", "pocket.api_server"):
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])

        # Two notes: one about vectors, one about deletion/lineage, with messy
        # whitespace to exercise the refinement stage.
        (self.source_dir / "vectors.md").write_text(
            "# Vector Search\r\n\r\n\r\n"
            "Semantic   embeddings   power cosine similarity   ranking.\r\n"
        )
        (self.source_dir / "lineage.md").write_text(
            "# Lineage\n\nDeletion propagation removes orphaned chunks automatically.\n"
        )
        # A Python source file to exercise code-aware refine + splitting.
        (self.source_dir / "widget.py").write_text(
            "import os\n\n\n"
            "class Widget:\n"
            "    def render(self):   \n"
            "        return os.getcwd()\n\n"
            "    def reset(self):\n"
            "        self.state  =  0\n\n"
            "def make_widget():\n"
            "    return Widget()\n"
        )

    def tearDown(self):
        if self.old_db_env is not None:
            os.environ["POCKET_SQLITE_DB"] = self.old_db_env
        else:
            os.environ.pop("POCKET_SQLITE_DB", None)
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
        self.temp_dir.cleanup()

    def _run(self):
        from pocket.pipeline import app_main
        app = pix.App(
            "pocket_test",
            app_main,
            sourcedir=self.source_dir,
            db_path=self.db_path,
        )
        app.update_blocking(live=False, report_to_stdout=False)

    def test_fts_index_populated(self):
        """The lexical FTS5 companion table is created and mirrors chunk text."""
        self._run()
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='_pocket_fts_embeddings'"
            )
            self.assertIsNotNone(cur.fetchone(), "FTS5 index table must exist")
            n_main = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            n_fts = conn.execute(
                "SELECT COUNT(*) FROM _pocket_fts_embeddings"
            ).fetchone()[0]
            self.assertEqual(n_main, n_fts, "FTS rows must mirror main rows")
        finally:
            conn.close()

    def test_lexical_search_matches_keyword(self):
        from pocket import retrieval
        importlib.reload(retrieval)
        self._run()
        hits = retrieval.search("Deletion", limit=5, mode="lexical", db_path=self.db_path)
        self.assertTrue(hits, "lexical search must return a keyword match")
        self.assertTrue(any("lineage.md" in h.file_path for h in hits))
        self.assertIsNotNone(hits[0].lexical_rank)

    def test_hybrid_search_fuses_results(self):
        from pocket import retrieval
        importlib.reload(retrieval)
        self._run()
        hits = retrieval.search(
            "embeddings similarity", limit=5, mode="hybrid", db_path=self.db_path
        )
        self.assertTrue(hits, "hybrid search must return results")
        self.assertTrue(any("vectors.md" in h.file_path for h in hits))
        self.assertGreater(hits[0].score, 0.0)

    def test_lineage_offsets_point_at_source(self):
        """After refinement, stored offsets still index into the raw source."""
        self._run()
        conn = sqlite3.connect(str(self.db_path))
        fp = conn.execute(
            "SELECT file_path FROM embeddings WHERE file_path LIKE '%lineage.md' LIMIT 1"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT start_offset, end_offset FROM embeddings WHERE file_path = ?",
            (fp,),
        ).fetchall()
        conn.close()
        raw = (self.source_dir / "lineage.md").read_text()
        self.assertTrue(rows)
        for start, end in rows:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(raw))

    def test_code_file_lineage_and_boundaries(self):
        """Code files index with exact source offsets and structural chunks."""
        self._run()
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute(
            "SELECT text, start_offset, end_offset FROM embeddings "
            "WHERE file_path LIKE '%widget.py' ORDER BY start_offset"
        ).fetchall()
        conn.close()
        self.assertTrue(rows)
        raw = (self.source_dir / "widget.py").read_text()
        # Every stored chunk's offsets must index into the raw source and the
        # stored text must round-trip to a substring of the raw source (offsets
        # were translated from refined space back to source space).
        for text, start, end in rows:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(raw))
            self.assertIn(text.split("\n", 1)[0].strip(), raw)
        # Indentation-preserving refine keeps method bodies intact in the index.
        all_text = "\n".join(r[0] for r in rows)
        self.assertIn("def render", all_text)
        self.assertIn("def make_widget", all_text)

    def test_api_health_and_search(self):
        from starlette.testclient import TestClient
        from pocket.api_server import create_app
        self._run()
        client = TestClient(create_app())

        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["index_ready"])

        r = client.get("/search", params={"q": "deletion", "mode": "lexical"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["mode"], "lexical")
        self.assertTrue(body["results"])

        r = client.post("/search", json={"query": "embeddings", "mode": "hybrid"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["results"])

        r = client.get("/search", params={"q": ""})
        self.assertEqual(r.status_code, 400)

class TestTextRefiner(unittest.TestCase):
    """Unit tests for the deterministic refinement stage."""

    def test_nfc_composes_decomposed_sequences(self):
        """NFD input (base char + combining mark) must compose to NFC."""
        from pocketindex.ops.refine import TextRefiner
        import unicodedata

        refiner = TextRefiner()
        # "cafe" + COMBINING ACUTE ACCENT + "naive" with COMBINING DIAERESIS.
        raw = "cafe\u0301 nai\u0308ve"
        doc = refiner.refine(raw)
        self.assertEqual(doc.text, "caf\u00e9 na\u00efve")
        self.assertTrue(unicodedata.is_normalized("NFC", doc.text))
        # Offset map stays one-to-one with the refined text and points into
        # the original source range so lineage remains valid.
        self.assertEqual(len(doc.offset_map), len(doc.text))
        self.assertTrue(all(0 <= o < len(raw) for o in doc.offset_map))

    def test_refinement_is_idempotent(self):
        """Refining already-clean NFC text is a no-op (stable function)."""
        from pocketindex.ops.refine import TextRefiner

        refiner = TextRefiner()
        once = refiner.refine("# Title\n\nHello world.")
        twice = refiner.refine(once.text)
        self.assertEqual(once.text, twice.text)


class TestCodeAwareSplitting(unittest.TestCase):
    """Unit tests for the code-aware splitter and refine path (POCKET-403)."""

    def test_detect_code_language(self):
        from pocketindex.ops.text import detect_code_language

        self.assertEqual(detect_code_language(filename="main.py"), "python")
        self.assertEqual(detect_code_language(filename="lib.rs"), "rust")
        self.assertEqual(detect_code_language(filename="app.tsx"), "typescript")
        self.assertIsNone(detect_code_language(filename="notes.xyz"))
        self.assertIsNone(detect_code_language(filename="noext"))

    def test_chunk_offsets_are_exact(self):
        """Every chunk's text must equal the source slice at its offsets."""
        from pocketindex.ops.text import RecursiveSplitter

        source = (
            "import os\n\n"
            "class Foo:\n"
            "    def alpha(self):\n        return 1\n\n"
            "    def beta(self):\n        return 2\n\n"
            "def top_level():\n    return Foo()\n"
        )
        splitter = RecursiveSplitter()
        for language in (None, "python"):
            chunks = splitter.split(
                source, chunk_size=40, chunk_overlap=8, language=language
            )
            self.assertTrue(chunks)
            for chunk in chunks:
                self.assertEqual(
                    chunk.text,
                    source[chunk.start.char_offset:chunk.end.char_offset],
                    f"offset mismatch (language={language})",
                )

    def test_python_splits_on_structural_boundaries(self):
        """Function/class definitions should head their own chunks."""
        from pocketindex.ops.text import RecursiveSplitter

        source = (
            "import os\n\n"
            "class Foo:\n"
            "    def alpha(self):\n        return 1\n"
            "    def beta(self):\n        return 2\n\n"
            "def top_level():\n    return Foo()\n"
        )
        chunks = RecursiveSplitter().split(
            source, chunk_size=40, chunk_overlap=0, language="python"
        )
        heads = [c.text.lstrip() for c in chunks]
        self.assertTrue(any(h.startswith("def alpha") for h in heads), heads)
        self.assertTrue(any(h.startswith("def beta") for h in heads), heads)
        self.assertTrue(any(h.startswith("def top_level") for h in heads), heads)

    def test_backward_compatible_signature(self):
        """The original positional (text, chunk_size, chunk_overlap) call works."""
        from pocketindex.ops.text import RecursiveSplitter

        text = "Para one.\n\nPara two is here.\n\nPara three ends it."
        chunks = RecursiveSplitter().split(text, 1000, 200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text[chunks[0].start.char_offset:chunks[0].end.char_offset])

    def test_separator_splitter(self):
        from pocketindex.ops.text import SeparatorSplitter

        text = "Para1\n\nPara2\n\nPara3"
        chunks = SeparatorSplitter([r"\n\n+"]).split(text)
        self.assertEqual([c.text for c in chunks], ["Para1", "Para2", "Para3"])
        for chunk in chunks:
            self.assertEqual(
                chunk.text, text[chunk.start.char_offset:chunk.end.char_offset]
            )

    def test_custom_language_config(self):
        from pocketindex.ops.text import CustomLanguageConfig, RecursiveSplitter

        config = CustomLanguageConfig("myformat", [r"---"], aliases=[".mf"])
        splitter = RecursiveSplitter(custom_languages=[config])
        chunks = splitter.split(
            "AAAA---BBBB---CCCC", chunk_size=6, chunk_overlap=0, language="myformat"
        )
        joined = "".join(c.text for c in chunks)
        self.assertIn("AAAA", joined)
        self.assertIn("BBBB", joined)
        self.assertIn("CCCC", joined)

    def test_code_refine_preserves_indentation(self):
        """Code-mode refine keeps indentation and inline spacing intact."""
        from pocketindex.ops.refine import TextRefiner

        code = (
            "class Foo:\n"
            "    def a(self):   \n"
            "        x  =  1\n\n\n"
            "    def b(self):\n        return x\n"
        )
        refiner = TextRefiner()
        doc = refiner.refine(code, code=True)
        # Indentation preserved.
        self.assertIn("\n    def a", doc.text)
        self.assertIn("\n        x", doc.text)
        self.assertIn("\n    def b", doc.text)
        # Inline double-spaces in code preserved.
        self.assertIn("x  =  1", doc.text)
        # Trailing whitespace still stripped.
        self.assertNotIn("):   \n", doc.text)
        # Excess blank lines still collapsed.
        self.assertNotIn("\n\n\n", doc.text)
        # Offset map stays valid.
        self.assertEqual(len(doc.offset_map), len(doc.text))
        self.assertTrue(all(0 <= o < len(code) for o in doc.offset_map))

    def test_prose_refine_still_collapses(self):
        """Default (prose) refine still collapses inline whitespace runs."""
        from pocketindex.ops.refine import TextRefiner

        doc = TextRefiner().refine("a  =  1")
        self.assertEqual(doc.text, "a = 1")


class TestLifecycleCommands(unittest.TestCase):
    """POCKET-405: ls / show / drop lifecycle commands."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = pathlib.Path(self.temp_dir.name) / "notes"
        self.source_dir.mkdir()
        self.db_path = pathlib.Path(self.temp_dir.name) / "pocket_data.db"

        self.old_db_env = os.environ.get("POCKET_SQLITE_DB")
        os.environ["POCKET_SQLITE_DB"] = str(self.db_path)
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
        for mod in ("pocket.retrieval", "pocket.admin"):
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])

        (self.source_dir / "alpha.md").write_text(
            "# Alpha\n\nThe alpha note about vectors and search.\n"
        )
        (self.source_dir / "beta.md").write_text(
            "# Beta\n\nThe beta note about lineage and deletion.\n"
        )

    def tearDown(self):
        if self.old_db_env is not None:
            os.environ["POCKET_SQLITE_DB"] = self.old_db_env
        else:
            os.environ.pop("POCKET_SQLITE_DB", None)
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
        self.temp_dir.cleanup()

    def _run(self):
        from pocket.pipeline import app_main
        app = pix.App(
            "pocket_test",
            app_main,
            sourcedir=self.source_dir,
            db_path=self.db_path,
        )
        app.update_blocking(live=False, report_to_stdout=False)

    def _count(self, where=""):
        conn = sqlite3.connect(str(self.db_path))
        try:
            sql = "SELECT COUNT(*) FROM embeddings" + (
                f" WHERE {where}" if where else ""
            )
            return conn.execute(sql).fetchone()[0]
        finally:
            conn.close()

    def test_list_sources_reports_each_file(self):
        from pocket import retrieval
        importlib.reload(retrieval)
        self._run()
        sources = retrieval.list_sources(db_path=self.db_path)
        paths = [s["file_path"] for s in sources]
        self.assertEqual(len(sources), 2)
        self.assertTrue(any(p.endswith("alpha.md") for p in paths))
        self.assertTrue(any(p.endswith("beta.md") for p in paths))
        for s in sources:
            self.assertGreater(s["chunks"], 0)
            self.assertGreaterEqual(s["first_offset"], 0)

    def test_target_stats_summarizes_index(self):
        from pocket import retrieval
        importlib.reload(retrieval)
        # Before any run the DB does not exist yet.
        empty = retrieval.target_stats(db_path=self.db_path)
        self.assertFalse(empty["exists"])
        self._run()
        stats = retrieval.target_stats(db_path=self.db_path)
        self.assertTrue(stats["exists"])
        self.assertEqual(stats["sources"], 2)
        self.assertGreater(stats["chunks"], 0)
        self.assertTrue(stats["fts_enabled"])

    def test_drop_source_removes_only_that_file(self):
        from pocket import admin
        importlib.reload(admin)
        self._run()
        conn = sqlite3.connect(str(self.db_path))
        beta_path = conn.execute(
            "SELECT file_path FROM embeddings WHERE file_path LIKE '%beta.md' "
            "LIMIT 1"
        ).fetchone()[0]
        conn.close()

        result = admin.drop_source(beta_path, db_path=self.db_path)
        self.assertGreater(result["removed"], 0)
        self.assertEqual(self._count("file_path LIKE '%beta.md'"), 0)
        self.assertGreater(self._count("file_path LIKE '%alpha.md'"), 0)
        # FTS mirror is kept in lockstep.
        conn = sqlite3.connect(str(self.db_path))
        try:
            n_main = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            n_fts = conn.execute(
                "SELECT COUNT(*) FROM _pocket_fts_embeddings"
            ).fetchone()[0]
            self.assertEqual(n_main, n_fts)
            # The dropped source's memo fingerprint is forgotten so a later
            # update re-adds it instead of memo-skipping it.
            memo = conn.execute(
                "SELECT COUNT(*) FROM _pocket_memo_embeddings"
            ).fetchone()[0]
            self.assertEqual(memo, 1)
        finally:
            conn.close()

    def test_dropped_source_is_reindexed_on_next_update(self):
        from pocket import admin
        importlib.reload(admin)
        self._run()
        conn = sqlite3.connect(str(self.db_path))
        beta_path = conn.execute(
            "SELECT file_path FROM embeddings WHERE file_path LIKE '%beta.md' "
            "LIMIT 1"
        ).fetchone()[0]
        conn.close()
        admin.drop_source(beta_path, db_path=self.db_path)
        self.assertEqual(self._count("file_path LIKE '%beta.md'"), 0)
        # Re-running must bring the dropped source back (memo was cleared).
        self._run()
        self.assertGreater(self._count("file_path LIKE '%beta.md'"), 0)

    def test_drop_target_resets_everything(self):
        from pocket import admin
        importlib.reload(admin)
        self._run()
        result = admin.drop_target(db_path=self.db_path)
        self.assertTrue(result["existed"])
        self.assertEqual(result["sources"], 2)
        self.assertGreater(result["chunks"], 0)
        self.assertIn("embeddings", result["dropped"])
        conn = sqlite3.connect(str(self.db_path))
        try:
            remaining = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='embeddings'"
            ).fetchone()
            self.assertIsNone(remaining, "embeddings table must be dropped")
        finally:
            conn.close()
        # A fresh update rebuilds the whole index from scratch.
        self._run()
        self.assertEqual(self._count(), self._count())
        self.assertGreater(self._count(), 0)

    def test_cli_commands_run(self):
        import pocket.cli as cli_module
        importlib.reload(cli_module)
        from click.testing import CliRunner
        cli = cli_module.cli
        self._run()
        runner = CliRunner()

        res = runner.invoke(cli, ["ls"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("alpha.md", res.output)
        self.assertIn("source(s) indexed", res.output)

        res = runner.invoke(cli, ["show"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Sources:", res.output)

        res = runner.invoke(cli, ["drop", "--yes"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Dropped", res.output)


class TestGraphExtraction(unittest.TestCase):
    """POCKET-404: offline graph ops (extraction + entity resolution) and the
    end-to-end graph target built with the deterministic (no-LLM) backend."""

    def test_deterministic_extractor_finds_entities_and_relations(self):
        from pocketindex.ops.extract import DeterministicExtractor

        ex = DeterministicExtractor()
        out = ex.extract(
            "Pocket uses SQLite for storage. SQLite powers the Pocket index."
        )
        names = {e.name for e in out.entities}
        self.assertIn("Pocket", names)
        self.assertIn("SQLite", names)
        # Co-occurrence relations are emitted between sentence-mates.
        self.assertTrue(
            any(
                r.predicate == "mentioned_with"
                and {r.subject, r.object} == {"Pocket", "SQLite"}
                for r in out.relations
            )
        )
        # Every fact carries confidence and evidence.
        for e in out.entities:
            self.assertGreaterEqual(e.confidence, 0.0)
            self.assertTrue(e.evidence)

    def test_build_extractor_defaults_to_deterministic(self):
        from pocketindex.ops.extract import build_extractor, DeterministicExtractor

        self.assertIsInstance(build_extractor(), DeterministicExtractor)
        # Unknown provider falls back to deterministic, never crashes.
        self.assertIsInstance(
            build_extractor(provider="nope"), DeterministicExtractor
        )

    def test_parse_extraction_json_validates_and_normalizes(self):
        from pocketindex.ops.extract import parse_extraction_json

        ext = parse_extraction_json(
            '```json\n{"entities":[{"name":" Foo ","type":"Tool","confidence":2}],'
            '"relations":[{"subject":"Foo","predicate":"Depends On",'
            '"object":"Bar","confidence":0.7}]}\n```',
            evidence="ctx",
        )
        self.assertEqual(ext.entities[0].name, "Foo")
        # Confidence is clamped to [0,1].
        self.assertEqual(ext.entities[0].confidence, 1.0)
        # Predicate is lower_snake_cased.
        self.assertEqual(ext.relations[0].predicate, "depends_on")

    def test_parse_extraction_json_rejects_garbage(self):
        from pocketindex.ops.extract import parse_extraction_json

        with self.assertRaises(ValueError):
            parse_extraction_json("not json at all", evidence="ctx")

    def test_entity_resolution_merges_duplicates(self):
        from pocketindex.ops.extract import ExtractedEntity
        from pocketindex.ops.entity_resolution import resolve_entities

        ents = [
            ExtractedEntity(name="SQLite", type="Tool", confidence=0.9),
            ExtractedEntity(name="sqlite", type="Tool", confidence=0.6),
            ExtractedEntity(name="Pocket", type="Concept", confidence=0.8),
        ]
        resolved = resolve_entities(ents)
        names = {r.name for r in resolved}
        # The two SQLite surface forms collapse into one cluster.
        self.assertEqual(len(resolved), 2)
        self.assertIn("Pocket", names)
        sqlite_cluster = next(r for r in resolved if r.name.lower() == "sqlite")
        # Canonical is the higher-confidence surface form; the other is an alias.
        self.assertEqual(sqlite_cluster.name, "SQLite")
        self.assertIn("sqlite", sqlite_cluster.aliases)

    def test_entity_resolution_adjudicator_is_optional_and_used(self):
        from pocketindex.ops.extract import ExtractedEntity
        from pocketindex.ops.entity_resolution import resolve_entities

        # Two embeddings that are similar (cos within the ambiguous band) but not
        # identical, with different surface forms.
        ents = [
            ExtractedEntity(name="Knowledge Graph", type="Concept", confidence=0.7),
            ExtractedEntity(name="KG", type="Concept", confidence=0.7),
        ]
        embeds = [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]]  # cosine ~0.71, ambiguous band
        # No adjudicator: stays unmerged (conservative).
        self.assertEqual(len(resolve_entities(ents, embeds)), 2)
        # With an adjudicator that says "merge": collapses to one.
        merged = resolve_entities(ents, embeds, adjudicator=lambda a, b: True)
        self.assertEqual(len(merged), 1)


class TestGraphTarget(unittest.TestCase):
    """POCKET-404a: the end-to-end graph target built offline with --graph."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = pathlib.Path(self.temp_dir.name) / "notes"
        self.source_dir.mkdir()
        self.db_path = pathlib.Path(self.temp_dir.name) / "pocket_data.db"

        self.old_db_env = os.environ.get("POCKET_SQLITE_DB")
        os.environ["POCKET_SQLITE_DB"] = str(self.db_path)
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
        for mod in ("pocket.retrieval", "pocket.admin", "pocket.pipeline"):
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])

        (self.source_dir / "a.md").write_text(
            "# Pocket\n\nPocket uses SQLite for storage. "
            "SQLite powers the Pocket index.\n"
        )

    def tearDown(self):
        if self.old_db_env is not None:
            os.environ["POCKET_SQLITE_DB"] = self.old_db_env
        else:
            os.environ.pop("POCKET_SQLITE_DB", None)
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
        self.temp_dir.cleanup()

    def _run(self, graph=False):
        from pocket.pipeline import app_main

        app = pix.App(
            "pocket_test",
            app_main,
            sourcedir=self.source_dir,
            db_path=self.db_path,
            graph=graph,
        )
        app.update_blocking(live=False, report_to_stdout=False)

    def _conn(self):
        import sqlite_vec

        conn = sqlite3.connect(str(self.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _table_exists(self, conn, name):
        return (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            ).fetchone()
            is not None
        )

    def test_graph_off_creates_no_graph_tables(self):
        self._run(graph=False)
        conn = self._conn()
        try:
            self.assertFalse(self._table_exists(conn, "entities"))
            self.assertFalse(self._table_exists(conn, "relations"))
            # The vector/lexical pipeline is unaffected.
            self.assertGreater(
                conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 0
            )
        finally:
            conn.close()

    def test_graph_on_materializes_entities_and_relations(self):
        self._run(graph=True)
        conn = self._conn()
        try:
            names = {
                r[0] for r in conn.execute("SELECT name FROM entities").fetchall()
            }
            self.assertIn("Pocket", names)
            self.assertIn("SQLite", names)
            # SQLite mentioned twice but resolved to a single node.
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM entities WHERE name='SQLite'"
                ).fetchone()[0],
                1,
            )
            self.assertGreater(
                conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0], 0
            )
            # Edges reference real entity ids.
            rel = conn.execute(
                "SELECT subject_id, object_id FROM relations LIMIT 1"
            ).fetchone()
            ids = {
                r[0] for r in conn.execute("SELECT id FROM entities").fetchall()
            }
            self.assertIn(rel[0], ids)
            self.assertIn(rel[1], ids)
        finally:
            conn.close()

    def test_graph_extraction_is_idempotent(self):
        self._run(graph=True)
        conn = self._conn()
        try:
            before = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        finally:
            conn.close()
        self._run(graph=True)
        conn = self._conn()
        try:
            after = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(before, after)

    def test_deleting_source_sweeps_its_subgraph(self):
        self._run(graph=True)
        (self.source_dir / "a.md").unlink()
        self._run(graph=True)
        conn = self._conn()
        try:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0], 0
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0], 0
            )
        finally:
            conn.close()

    def test_graph_neighborhood_retrieval(self):
        from pocket import retrieval

        importlib.reload(retrieval)
        self._run(graph=True)
        node = retrieval.graph_neighborhood("Pocket", db_path=self.db_path)
        self.assertEqual(node["name"], "Pocket")
        self.assertTrue(node["neighbors"])
        self.assertEqual(node["neighbors"][0]["neighbor"], "SQLite")
        rendered = retrieval.format_neighborhood(node)
        self.assertIn("Pocket", rendered)
        self.assertIn("SQLite", rendered)

    def test_drop_removes_graph_tables(self):
        from pocket import admin

        importlib.reload(admin)
        self._run(graph=True)
        result = admin.drop_target(db_path=self.db_path)
        self.assertTrue(result["existed"])
        self.assertIn("entities", result["dropped"])
        self.assertIn("relations", result["dropped"])
        conn = sqlite3.connect(str(self.db_path))
        try:
            self.assertFalse(self._table_exists(conn, "entities"))
            self.assertFalse(self._table_exists(conn, "relations"))
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main()
