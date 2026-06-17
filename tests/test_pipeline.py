import unittest
import tempfile
import pathlib
import sqlite3
import sqlite_vec
import cocoindex as coco
import os
import sys
import importlib

class TestPocketPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = pathlib.Path(self.temp_dir.name) / "notes"
        self.source_dir.mkdir()
        self.db_path = pathlib.Path(self.temp_dir.name) / "pocket_data.db"
        self.coco_db_path = pathlib.Path(self.temp_dir.name) / "cocoindex.db"
        
        # Set environment variables for the pipeline to pick up
        self.old_db_env = os.environ.get("POCKET_SQLITE_DB")
        self.old_coco_env = os.environ.get("COCOINDEX_DB")
        os.environ["POCKET_SQLITE_DB"] = str(self.db_path)
        os.environ["COCOINDEX_DB"] = str(self.coco_db_path)
        
        # Reload only config to pick up the new env vars
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
            
        # Create a sample note
        self.note_file = self.source_dir / "test_note.md"
        self.note_file.write_text("# Test Note\n\nThis is a test note for Pocket Knowledge Ops.")

    def tearDown(self):
        if self.old_db_env is not None:
            os.environ["POCKET_SQLITE_DB"] = self.old_db_env
        else:
            os.environ.pop("POCKET_SQLITE_DB", None)
            
        if self.old_coco_env is not None:
            os.environ["COCOINDEX_DB"] = self.old_coco_env
        else:
            os.environ.pop("COCOINDEX_DB", None)
            
        # Reload config to restore original env vars
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
            
        self.temp_dir.cleanup()

    def test_pipeline_and_search(self):
        from pocket.pipeline import app_main
        # Run the pipeline
        app = coco.App(
            "pocket_test",
            app_main,
            sourcedir=self.source_dir,
            db_path=self.db_path,
        )
        app.update_blocking(live=False, report_to_stdout=False)
        
        # Verify database exists and has data
        self.assertTrue(self.db_path.exists())
        
        conn = sqlite3.connect(str(self.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        self.assertGreater(count, 0)
        
        # Verify lineage metadata
        cursor = conn.execute("SELECT file_path, text, start_offset, end_offset FROM embeddings")
        row = cursor.fetchone()
        self.assertTrue(row[0].endswith("test_note.md"))
        self.assertIn("Test Note", row[1])
        self.assertEqual(row[2], 0)
        self.assertGreater(row[3], 0)
        
        conn.close()

    def test_mcp_tools(self):
        from pocket.pipeline import app_main
        from pocket.mcp_server import search_knowledge, get_file_lineage
        
        # Run the pipeline to populate the DB
        app = coco.App(
            "pocket_test",
            app_main,
            sourcedir=self.source_dir,
            db_path=self.db_path,
        )
        app.update_blocking(live=False, report_to_stdout=False)
        
        # Test search_knowledge tool
        search_result = search_knowledge("Pocket")
        self.assertIn("test_note.md", search_result)
        self.assertIn("Test Note", search_result)
        
        # Test get_file_lineage tool
        # We need to pass the exact file path stored in the DB
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT file_path FROM embeddings LIMIT 1")
        db_file_path = cursor.fetchone()[0]
        conn.close()
        
        lineage_result = get_file_lineage(db_file_path)
        self.assertIn("Lineage for", lineage_result)
        self.assertIn("Chunk 1", lineage_result)

    def _run(self):
        from pocket.pipeline import app_main
        app = coco.App(
            "pocket_test",
            app_main,
            sourcedir=self.source_dir,
            db_path=self.db_path,
        )
        app.update_blocking(live=False, report_to_stdout=False)

    def _count_rows(self, where=""):
        conn = sqlite3.connect(str(self.db_path))
        try:
            sql = "SELECT COUNT(*) FROM embeddings" + (f" WHERE {where}" if where else "")
            return conn.execute(sql).fetchone()[0]
        finally:
            conn.close()

    def test_incremental_memoization(self):
        """DoD #3: re-running with no edits skips reprocessing; editing one
        file reprocesses only that file."""
        import pocket.pipeline as pipeline

        second = self.source_dir / "second.md"
        second.write_text("# Second\n\nAnother note about pocket knowledge.")

        # Count how many times process_file actually executes per run.
        original = pipeline.process_file
        calls = []

        async def counting(file, table):
            calls.append(str(file.file_path.path))
            return await original(file, table)
        counting._coco_fn = True
        counting._memo = True
        pipeline.process_file = counting
        try:
            # First run: both files processed.
            self._run()
            self.assertEqual(len(calls), 2)
            base_rows = self._count_rows()
            self.assertGreater(base_rows, 0)

            # Second run, no changes: memoization skips everything.
            calls.clear()
            self._run()
            self.assertEqual(calls, [], "unchanged files must be skipped")
            self.assertEqual(self._count_rows(), base_rows)

            # Edit only one file: only that file is reprocessed.
            calls.clear()
            self.note_file.write_text(
                "# Test Note\n\nHeavily edited content for pocket knowledge ops."
            )
            self._run()
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0].endswith("test_note.md"))
        finally:
            pipeline.process_file = original

    def test_deletion_propagates(self):
        """DoD #4: deleting a source file removes its chunks from the DB."""
        second = self.source_dir / "second.md"
        second.write_text("# Second\n\nAnother note about pocket knowledge.")

        self._run()
        self.assertGreater(self._count_rows("file_path LIKE '%second.md'"), 0)
        self.assertGreater(self._count_rows("file_path LIKE '%test_note.md'"), 0)

        # Delete one source file and re-run.
        second.unlink()
        self._run()

        self.assertEqual(self._count_rows("file_path LIKE '%second.md'"), 0,
                         "deleted source must have no chunks")
        self.assertGreater(self._count_rows("file_path LIKE '%test_note.md'"), 0,
                           "surviving source must keep its chunks")

    def test_abort_source_discards_uncommitted_rows(self):
        """Transaction safety (unit): abort_source must roll back rows a
        failed source emitted before they leak into a later source's commit.

        Without abort_source the partial rows of the failed source remain in
        the connection's pending transaction and get persisted by the next
        successful end_source commit; this test fails in that case.
        """
        from dataclasses import dataclass
        from cocoindex.connectors import sqlite
        import cocoindex as coco

        @dataclass
        class Row:
            id: int
            val: str

        async def build():
            conn = sqlite.ManagedConnection(str(self.db_path), load_vec=False)
            await conn.__aenter__()
            try:
                schema = await sqlite.TableSchema.from_class(Row, primary_key=["id"])
                target = sqlite.TableTarget(conn, "t", schema)

                # Source A starts, emits a row, then "fails" before end_source.
                tokA = coco._current_source_key.set("A")
                target.begin_source("A")
                target.declare_row(Row(id=1, val="from-A"))
                coco._current_source_key.reset(tokA)
                target.abort_source("A")  # failure path

                # Source B processes cleanly and commits.
                tokB = coco._current_source_key.set("B")
                target.begin_source("B")
                target.declare_row(Row(id=2, val="from-B"))
                coco._current_source_key.reset(tokB)
                target.end_source("B", "hashB")

                ids = {r[0] for r in conn.execute("SELECT id FROM t").fetchall()}
                return ids
            finally:
                await conn.__aexit__(None, None, None)

        import asyncio
        ids = asyncio.run(build())
        self.assertNotIn(1, ids, "aborted source's row must not persist")
        self.assertIn(2, ids, "committed source's row must persist")

if __name__ == "__main__":
    unittest.main()
