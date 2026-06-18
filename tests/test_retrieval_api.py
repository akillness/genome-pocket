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


if __name__ == "__main__":
    unittest.main()
