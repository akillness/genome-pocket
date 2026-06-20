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

    def test_routing_trace_annotates_strategies_and_contributors(self):
        from pocket import retrieval

        importlib.reload(retrieval)
        self._run()

        trace = retrieval.routing_trace("deletion propagation", mode="hybrid")
        self.assertEqual(trace["mode"], "hybrid")
        by_name = {s["name"]: s for s in trace["strategies"]}
        self.assertEqual(set(by_name), {"vector", "lexical", "graph"})
        # Hybrid activates all three; lexical is available (FTS built), graph is
        # not (this index has no --graph entities table).
        self.assertTrue(by_name["vector"]["active"])
        self.assertTrue(by_name["lexical"]["active"])
        self.assertTrue(by_name["graph"]["active"])
        self.assertTrue(by_name["lexical"]["available"])
        self.assertFalse(by_name["graph"]["available"])
        self.assertEqual(by_name["graph"]["candidates"], 0)
        self.assertTrue(by_name["vector"]["candidates"] > 0)

        self.assertTrue(trace["results"])
        # Every hit names the strategies that surfaced it, and no hit claims a
        # graph contribution since the graph strategy never ran.
        for hit in trace["results"]:
            self.assertTrue(hit["contributors"])
            self.assertNotIn("graph", hit["contributors"])
            for c in hit["contributors"]:
                self.assertIn(c, {"vector", "lexical"})
        contributing = {c for hit in trace["results"] for c in hit["contributors"]}
        self.assertIn("vector", contributing)

    def test_routing_trace_lexical_mode_routes_only_lexical(self):
        from pocket import retrieval

        importlib.reload(retrieval)
        self._run()

        trace = retrieval.routing_trace("deletion", mode="lexical")
        by_name = {s["name"]: s for s in trace["strategies"]}
        self.assertTrue(by_name["lexical"]["active"])
        self.assertFalse(by_name["vector"]["active"])
        self.assertFalse(by_name["graph"]["active"])
        # Inactive strategies produce no candidates even though vector is
        # otherwise available.
        self.assertEqual(by_name["vector"]["candidates"], 0)
        self.assertTrue(by_name["lexical"]["candidates"] > 0)
        self.assertTrue(trace["results"])
        for hit in trace["results"]:
            self.assertEqual(hit["contributors"], ["lexical"])

    def test_routing_trace_missing_index_returns_empty(self):
        from pocket import retrieval

        importlib.reload(retrieval)
        # No _run(): the DB does not exist yet.
        trace = retrieval.routing_trace("anything", mode="hybrid")
        self.assertEqual(trace["results"], [])
        for s in trace["strategies"]:
            self.assertFalse(s["available"])
            self.assertEqual(s["candidates"], 0)

    def test_api_ui_and_trace_endpoints(self):
        from starlette.testclient import TestClient
        from pocket.api_server import create_app

        self._run()
        client = TestClient(create_app())

        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("Query Tracing", r.text)

        r = client.get("/trace", params={"q": "deletion", "mode": "hybrid"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["mode"], "hybrid")
        self.assertTrue(body["results"])
        self.assertEqual(
            {s["name"] for s in body["strategies"]},
            {"vector", "lexical", "graph"},
        )

        r = client.get("/trace", params={"q": ""})
        self.assertEqual(r.status_code, 400)

        r = client.get("/trace", params={"q": "x", "mode": "bogus"})
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
        self.assertIn("SQLite", rendered)

    def test_graph_mode_search_returns_anchored_chunks(self):
        from pocket import retrieval

        importlib.reload(retrieval)
        self._run(graph=True)
        hits = retrieval.search(
            "SQLite storage", limit=5, mode="graph", db_path=self.db_path
        )
        self.assertTrue(hits, "graph mode must surface entity-anchored chunks")
        self.assertTrue(any(h.file_path.endswith("a.md") for h in hits))
        # Graph hits carry the third-list rank, not the vector/lexical ranks.
        self.assertIsNotNone(hits[0].graph_rank)
        self.assertIsNone(hits[0].vector_rank)
        self.assertIsNone(hits[0].lexical_rank)
        # Every graph hit resolves back to a real source chunk (lineage intact).
        self.assertTrue(all(h.end_offset > h.start_offset for h in hits))

    def test_graph_mode_empty_without_graph_tables(self):
        from pocket import retrieval

        importlib.reload(retrieval)
        self._run(graph=False)  # no entities/relations tables materialized
        hits = retrieval.search(
            "SQLite", limit=5, mode="graph", db_path=self.db_path
        )
        self.assertEqual(hits, [])

    def test_hybrid_fuses_graph_signal_when_graph_present(self):
        from pocket import retrieval

        importlib.reload(retrieval)
        self._run(graph=True)
        hits = retrieval.search(
            "SQLite storage", limit=5, mode="hybrid", db_path=self.db_path
        )
        self.assertTrue(hits)
        # The third (graph) list participates: at least one hit was reinforced
        # by graph traversal on top of the vector/lexical fusion.
        self.assertTrue(any(h.graph_rank is not None for h in hits))

    def test_traverse_graph_mcp_tool(self):
        from pocket import retrieval, mcp_server

        importlib.reload(retrieval)
        importlib.reload(mcp_server)
        self._run(graph=True)
        rendered = mcp_server.traverse_graph("Pocket")
        self.assertIn("Pocket", rendered)
        self.assertIn("SQLite", rendered)
        # The traversal renders the one-hop relations, not just the node header.
        self.assertIn("Relations", rendered)
        self.assertIn("->", rendered)

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

    # --- POCKET-302: human-in-the-loop confidence gate -------------------
    def _set_min_conf(self, value):
        """Set the staging threshold and reload the graph modules so the
        pipeline gate, retrieval filters, and admin review all see it."""
        os.environ["POCKET_GRAPH_MIN_CONFIDENCE"] = str(value)
        self.addCleanup(os.environ.pop, "POCKET_GRAPH_MIN_CONFIDENCE", None)
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
        for mod in ("pocket.pipeline", "pocket.retrieval", "pocket.admin"):
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])

    def test_facts_above_threshold_are_committed(self):
        # Default threshold (0.0): every extracted fact is committed, not staged.
        self._run(graph=True)
        conn = self._conn()
        try:
            self.assertEqual(
                {r[0] for r in conn.execute("SELECT DISTINCT status FROM entities")},
                {"approved"},
            )
            self.assertEqual(
                {r[0] for r in conn.execute("SELECT DISTINCT status FROM relations")},
                {"approved"},
            )
        finally:
            conn.close()

    def test_low_confidence_facts_are_staged_not_committed(self):
        # Threshold above the deterministic extractor's confidence stages
        # everything: the rows exist but never surface in retrieval.
        self._set_min_conf("0.9")
        self._run(graph=True)
        conn = self._conn()
        try:
            self.assertGreater(
                conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0], 0
            )
            self.assertEqual(
                {r[0] for r in conn.execute("SELECT DISTINCT status FROM entities")},
                {"pending"},
            )
        finally:
            conn.close()
        from pocket import retrieval, admin

        # Pending facts are invisible to graph reads.
        self.assertEqual(
            retrieval.graph_neighborhood("Pocket", db_path=self.db_path), {}
        )
        self.assertEqual(
            retrieval.list_graph_concepts(db_path=self.db_path), []
        )
        # But they are listed for review.
        pending = admin.list_pending(db_path=self.db_path)
        self.assertTrue(pending["entities"])
        self.assertIn(
            "Pocket", {e["name"] for e in pending["entities"]}
        )

    def test_approve_pending_commits_facts(self):
        self._set_min_conf("0.9")
        self._run(graph=True)
        from pocket import retrieval, admin

        counts = admin.approve_pending(db_path=self.db_path)
        self.assertGreater(counts["entities"], 0)
        # Now retrievable, and nothing left pending.
        node = retrieval.graph_neighborhood("Pocket", db_path=self.db_path)
        self.assertTrue(node)
        self.assertEqual(node["name"], "Pocket")
        self.assertEqual(admin.list_pending(db_path=self.db_path)["entities"], [])

    def test_approve_specific_id_leaves_others_pending(self):
        self._set_min_conf("0.9")
        self._run(graph=True)
        from pocket import admin

        pending = admin.list_pending(db_path=self.db_path)["entities"]
        self.assertGreaterEqual(len(pending), 2)
        target = pending[0]["id"]
        counts = admin.approve_pending(ids=[target], db_path=self.db_path)
        self.assertEqual(counts["entities"], 1)
        remaining = {
            e["id"] for e in admin.list_pending(db_path=self.db_path)["entities"]
        }
        self.assertNotIn(target, remaining)
        self.assertTrue(remaining)  # the others stay staged

    def test_reject_pending_discards_facts(self):
        self._set_min_conf("0.9")
        self._run(graph=True)
        from pocket import admin

        before = admin.list_pending(db_path=self.db_path)["entities"]
        self.assertTrue(before)
        counts = admin.reject_pending(db_path=self.db_path)
        self.assertEqual(counts["entities"], len(before))
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
        self.assertEqual(admin.list_pending(db_path=self.db_path)["entities"], [])

    def test_cli_graph_review_lists_and_approves(self):
        import pocket.cli as cli_module

        self._set_min_conf("0.9")
        importlib.reload(cli_module)
        from click.testing import CliRunner

        self._run(graph=True)
        runner = CliRunner()

        res = runner.invoke(cli_module.cli, ["graph", "review"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Pending entities", res.output)
        self.assertIn("Pocket", res.output)

        res = runner.invoke(cli_module.cli, ["graph", "review", "--approve-all"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Approved", res.output)

        res = runner.invoke(cli_module.cli, ["graph", "review"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("No facts are pending", res.output)

    def test_cli_graph_show_still_routes_to_neighborhood(self):
        # Backward compat: `pocket graph <entity>` works without the `show` verb.
        import pocket.cli as cli_module

        importlib.reload(cli_module)
        from click.testing import CliRunner

        self._run(graph=True)
        runner = CliRunner()
        res = runner.invoke(cli_module.cli, ["graph", "Pocket"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Pocket", res.output)
        self.assertIn("Relations", res.output)

    # --- POCKET-301: interactive review during `pocket update --graph` ----
    @staticmethod
    def _scripted(answers):
        """A click.prompt stand-in returning queued answers in order."""
        it = iter(answers)

        def _p(*args, **kwargs):
            return next(it)

        return _p

    def test_interactive_review_approve_all_commits(self):
        import pocket.cli as cli_module

        self._set_min_conf("0.9")
        importlib.reload(cli_module)
        self._run(graph=True)
        from pocket import admin, retrieval

        out = []
        cli_module._interactive_graph_review(
            echo=out.append, prompt=self._scripted(["a"])
        )
        joined = "\n".join(out)
        self.assertIn("staged by the confidence gate", joined)
        self.assertIn("Approved", joined)
        # Everything committed: nothing pending, and now retrievable.
        self.assertEqual(admin.list_pending(db_path=self.db_path)["entities"], [])
        node = retrieval.graph_neighborhood("Pocket", db_path=self.db_path)
        self.assertTrue(node)

    def test_interactive_review_each_mode_routes_per_fact(self):
        import pocket.cli as cli_module

        self._set_min_conf("0.9")
        importlib.reload(cli_module)
        self._run(graph=True)
        from pocket import admin

        pending = admin.list_pending(db_path=self.db_path)
        items = pending["entities"] + pending["relations"]
        self.assertGreaterEqual(len(items), 3)

        # Top choice "e" (each), then approve / reject / leave-pending per fact.
        answers = ["e"]
        expect_approve, expect_reject, expect_skip = [], [], []
        for i, item in enumerate(items):
            if i == len(items) - 1:
                answers.append("s")
                expect_skip.append(item["id"])
            elif i % 2 == 0:
                answers.append("y")
                expect_approve.append(item["id"])
            else:
                answers.append("n")
                expect_reject.append(item["id"])

        out = []
        cli_module._interactive_graph_review(
            echo=out.append, prompt=self._scripted(answers)
        )

        # Skipped facts stay pending; approved/rejected leave the queue.
        remaining = admin.list_pending(db_path=self.db_path)
        remaining_ids = {e["id"] for e in remaining["entities"]} | {
            r["id"] for r in remaining["relations"]
        }
        self.assertEqual(remaining_ids, set(expect_skip))

        conn = self._conn()
        try:
            # Approved entity ids are committed (status flipped).
            for table in ("entities", "relations"):
                rows = {
                    r[0]: r[1]
                    for r in conn.execute(f"SELECT id, status FROM {table}").fetchall()
                }
                for rid, status in rows.items():
                    if rid in expect_approve:
                        self.assertEqual(status, "approved")
                # Rejected ids are deleted outright.
                for rid in expect_reject:
                    self.assertNotIn(rid, rows)
        finally:
            conn.close()
        self.assertIn("still pending", "\n".join(out))

    def test_interactive_review_quit_stops_each_loop(self):
        import pocket.cli as cli_module

        self._set_min_conf("0.9")
        importlib.reload(cli_module)
        self._run(graph=True)
        from pocket import admin

        before = admin.list_pending(db_path=self.db_path)
        total_before = len(before["entities"]) + len(before["relations"])
        # each-mode, approve the first fact, then quit: the rest stay pending.
        out = []
        cli_module._interactive_graph_review(
            echo=out.append, prompt=self._scripted(["e", "y", "q"])
        )
        after = admin.list_pending(db_path=self.db_path)
        total_after = len(after["entities"]) + len(after["relations"])
        self.assertEqual(total_after, total_before - 1)

    def test_interactive_review_skip_leaves_everything_pending(self):
        import pocket.cli as cli_module

        self._set_min_conf("0.9")
        importlib.reload(cli_module)
        self._run(graph=True)
        from pocket import admin

        before = admin.list_pending(db_path=self.db_path)
        out = []
        cli_module._interactive_graph_review(
            echo=out.append, prompt=self._scripted(["s"])
        )
        after = admin.list_pending(db_path=self.db_path)
        self.assertEqual(len(after["entities"]), len(before["entities"]))
        self.assertEqual(len(after["relations"]), len(before["relations"]))
        self.assertIn("Skipped", "\n".join(out))

    def test_interactive_review_no_pending_does_not_prompt(self):
        import pocket.cli as cli_module

        importlib.reload(cli_module)
        # Default threshold: every fact is committed, so nothing is pending.
        self._run(graph=True)

        def _boom(*args, **kwargs):
            raise AssertionError("prompt should not be called when nothing pending")

        out = []
        cli_module._interactive_graph_review(echo=out.append, prompt=_boom)
        self.assertIn("No graph facts are pending review.", "\n".join(out))

    def test_cli_update_graph_review_end_to_end(self):
        import pocket.cli as cli_module
        from click.testing import CliRunner

        old_source = os.environ.get("POCKET_SOURCE_DIR")
        os.environ["POCKET_SOURCE_DIR"] = str(self.source_dir)
        self.addCleanup(
            lambda: os.environ.__setitem__("POCKET_SOURCE_DIR", old_source)
            if old_source is not None
            else os.environ.pop("POCKET_SOURCE_DIR", None)
        )
        self._set_min_conf("0.9")
        importlib.reload(cli_module)
        from pocket import admin, retrieval

        runner = CliRunner()
        res = runner.invoke(
            cli_module.cli, ["update", "--graph", "--review"], input="a\n"
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Approved", res.output)
        self.assertEqual(admin.list_pending(db_path=self.db_path)["entities"], [])
        self.assertTrue(retrieval.graph_neighborhood("Pocket", db_path=self.db_path))

    def test_cli_update_review_without_graph_is_ignored(self):
        import pocket.cli as cli_module
        from click.testing import CliRunner

        old_source = os.environ.get("POCKET_SOURCE_DIR")
        os.environ["POCKET_SOURCE_DIR"] = str(self.source_dir)
        self.addCleanup(
            lambda: os.environ.__setitem__("POCKET_SOURCE_DIR", old_source)
            if old_source is not None
            else os.environ.pop("POCKET_SOURCE_DIR", None)
        )
        importlib.reload(cli_module)

        runner = CliRunner()
        res = runner.invoke(cli_module.cli, ["update", "--review"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("no effect without --graph", res.output)
class TestRetrievalEvaluation(unittest.TestCase):
    """POCKET-303: automated retrieval evaluation & regression guard."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = pathlib.Path(self.temp_dir.name) / "notes"
        self.source_dir.mkdir()
        self.db_path = pathlib.Path(self.temp_dir.name) / "pocket_data.db"

        self.old_db_env = os.environ.get("POCKET_SQLITE_DB")
        os.environ["POCKET_SQLITE_DB"] = str(self.db_path)
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
        for mod in ("pocket.retrieval", "pocket.evaluation"):
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])

        # Three notes with deliberately disjoint vocabulary so distinctive-token
        # synthetic queries have exactly one correct source under lexical search.
        (self.source_dir / "biology.md").write_text(
            "# Biology\n\nPhotosynthesis converts sunlight via chlorophyll into "
            "glucose inside chloroplasts.\n"
        )
        (self.source_dir / "geology.md").write_text(
            "# Geology\n\nVolcanic eruption ejects magma forming basalt across "
            "tectonic boundaries.\n"
        )
        (self.source_dir / "crypto.md").write_text(
            "# Cryptography\n\nEncryption ciphers maximize entropy protecting "
            "asymmetric keypairs.\n"
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
            "pocket_test", app_main, sourcedir=self.source_dir, db_path=self.db_path
        )
        app.update_blocking(live=False, report_to_stdout=False)

    def test_metric_primitives(self):
        from pocket import evaluation as ev

        retrieved = ["a.md", "b.md", "c.md", "d.md"]
        relevant = ["c.md"]
        # First relevant hit is at rank 3.
        self.assertAlmostEqual(ev.reciprocal_rank(retrieved, relevant), 1 / 3)
        self.assertEqual(ev.reciprocal_rank(retrieved, ["zzz.md"]), 0.0)
        # 1 of the top-4 is relevant; recall is 1 of 1 relevant file.
        self.assertAlmostEqual(ev.precision_at_k(retrieved, relevant, 4), 1 / 4)
        self.assertAlmostEqual(ev.recall_at_k(retrieved, relevant, 4), 1.0)
        # Cutoff below the hit drops both precision and recall to zero.
        self.assertEqual(ev.recall_at_k(retrieved, relevant, 2), 0.0)
        # AP: single relevant at rank 3 -> precision 1/3 averaged over 1 relevant.
        self.assertAlmostEqual(ev.average_precision(retrieved, relevant, 4), 1 / 3)
        # Two relevant files ranked 1 and 2 -> perfect AP.
        self.assertAlmostEqual(
            ev.average_precision(["c.md", "d.md", "a.md"], ["c.md", "d.md"], 3), 1.0
        )
        # Lenient path matching: basename / relative-suffix counts as a hit.
        self.assertEqual(
            ev.reciprocal_rank(["/abs/path/crypto.md"], ["crypto.md"]), 1.0
        )

    def test_synthesize_and_evaluate_self_retrieves(self):
        from pocket import evaluation as ev

        self._run()
        cases = ev.synthesize_cases(db_path=self.db_path, mode="lexical", per_file=1)
        # One self-labeled case per indexed source file.
        self.assertEqual(len(cases), 3)
        for c in cases:
            self.assertEqual(len(c.relevant_files), 1)
            self.assertTrue(c.query.strip())
            self.assertEqual(c.mode, "lexical")

        metrics = ev.evaluate(cases, db_path=self.db_path, k=5)
        # Distinctive-token queries must each retrieve their own source first,
        # so a healthy lexical index scores a perfect hit rate and MRR.
        self.assertEqual(metrics.n_cases, 3)
        self.assertEqual(metrics.hit_rate, 1.0)
        self.assertEqual(metrics.mrr, 1.0)
        self.assertEqual(metrics.recall_at_k, 1.0)
        for cr in metrics.cases:
            self.assertTrue(cr.hit)
            self.assertTrue(
                any(
                    os.path.basename(cr.relevant_files[0]) == os.path.basename(f)
                    for f in cr.retrieved_files
                )
            )

    def test_synthesize_empty_when_no_index(self):
        from pocket import evaluation as ev

        # No _run(): the DB does not exist yet.
        self.assertEqual(ev.synthesize_cases(db_path=self.db_path), [])
        # evaluate() over no cases yields zeroed metrics, not a crash.
        metrics = ev.evaluate([], db_path=self.db_path, k=5)
        self.assertEqual(metrics.n_cases, 0)
        self.assertEqual(metrics.hit_rate, 0.0)

    def test_load_cases_parsing_and_errors(self):
        import json

        from pocket import evaluation as ev

        good = pathlib.Path(self.temp_dir.name) / "cases.json"
        good.write_text(
            json.dumps(
                {
                    "cases": [
                        {"query": "encryption keys", "relevant_files": ["crypto.md"]},
                        {
                            "query": "magma basalt",
                            "relevant_files": ["geology.md"],
                            "mode": "lexical",
                        },
                    ]
                }
            )
        )
        cases = ev.load_cases(good)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].mode, "hybrid")  # default
        self.assertEqual(cases[1].mode, "lexical")

        # Top-level list form is accepted too.
        listform = pathlib.Path(self.temp_dir.name) / "list.json"
        listform.write_text(
            json.dumps([{"query": "q", "relevant_files": ["a.md"]}])
        )
        self.assertEqual(len(ev.load_cases(listform)), 1)

        bad = pathlib.Path(self.temp_dir.name) / "bad.json"
        bad.write_text(json.dumps([{"relevant_files": ["a.md"]}]))  # no query
        with self.assertRaises(ValueError):
            ev.load_cases(bad)
        bad.write_text(json.dumps([{"query": "q", "relevant_files": []}]))  # empty rel
        with self.assertRaises(ValueError):
            ev.load_cases(bad)

    def test_baseline_roundtrip_and_regression_detection(self):
        from pocket import evaluation as ev

        self._run()
        cases = ev.synthesize_cases(db_path=self.db_path, mode="lexical")
        metrics = ev.evaluate(cases, db_path=self.db_path, k=5)

        baseline_path = pathlib.Path(self.temp_dir.name) / "baseline.json"
        ev.save_baseline(baseline_path, metrics)
        loaded = ev.load_baseline(baseline_path)
        self.assertEqual(loaded["hit_rate"], metrics.hit_rate)
        self.assertNotIn("cases", loaded)  # baselines store aggregates only

        # Identical run -> no regression.
        self.assertEqual(ev.compare_to_baseline(metrics, loaded), [])

        # A stricter baseline that the run no longer meets -> regression flagged.
        harder = dict(loaded)
        harder["hit_rate"] = loaded["hit_rate"] + 0.5
        regs = ev.compare_to_baseline(metrics, harder)
        names = {r.metric for r in regs}
        self.assertIn("hit_rate", names)
        reg = next(r for r in regs if r.metric == "hit_rate")
        self.assertLess(reg.delta, 0.0)
        # Tolerance can absorb the same drop.
        self.assertEqual(ev.compare_to_baseline(metrics, harder, tolerance=1.0), [])
        # Metrics absent from the baseline never fail.
        self.assertEqual(ev.compare_to_baseline(metrics, {"unknown": 1.0}), [])

    def test_cli_eval_synthetic_and_baseline(self):
        import pocket.cli as cli_module
        from click.testing import CliRunner

        importlib.reload(cli_module)
        self._run()
        runner = CliRunner()

        baseline_path = pathlib.Path(self.temp_dir.name) / "cli_baseline.json"
        res = runner.invoke(
            cli_module.cli,
            ["eval", "--mode", "lexical", "--save", str(baseline_path), "--show-cases"],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Synthesized 3 case(s)", res.output)
        self.assertIn("Hit@5:", res.output)
        self.assertTrue(baseline_path.exists())

        # Re-running against the just-saved baseline must pass (no regression).
        res = runner.invoke(
            cli_module.cli,
            ["eval", "--mode", "lexical", "--baseline", str(baseline_path)],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("No regression versus baseline", res.output)

        # A doctored baseline the run can't meet must fail the command (exit 1).
        import json

        data = json.loads(baseline_path.read_text())
        data["hit_rate"] = 1.5
        baseline_path.write_text(json.dumps(data))
        res = runner.invoke(
            cli_module.cli,
            ["eval", "--mode", "lexical", "--baseline", str(baseline_path)],
        )
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertIn("REGRESSION", res.output)


if __name__ == "__main__":
    unittest.main()
