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


if __name__ == "__main__":
    unittest.main()
