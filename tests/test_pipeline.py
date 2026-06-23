import unittest
import tempfile
import pathlib
import sqlite3
import sqlite_vec
import pocketindex as pix
import os
import sys
import importlib

class TestPocketPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = pathlib.Path(self.temp_dir.name) / "notes"
        self.source_dir.mkdir()
        self.db_path = pathlib.Path(self.temp_dir.name) / "pocket_data.db"
        
        # Set environment variables for the pipeline to pick up
        self.old_db_env = os.environ.get("POCKET_SQLITE_DB")
        os.environ["POCKET_SQLITE_DB"] = str(self.db_path)
        
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
            
        # Reload config to restore original env vars
        if "pocket.config" in sys.modules:
            importlib.reload(sys.modules["pocket.config"])
            
        self.temp_dir.cleanup()

    def test_pipeline_and_search(self):
        from pocket.pipeline import app_main
        # Run the pipeline
        app = pix.App(
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
        app = pix.App(
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
        app = pix.App(
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

    def _count_fts_rows(self, where=""):
        """Count rows in the FTS5 lexical companion index.

        The lexical (BM25) index is a separate virtual table that must stay in
        lockstep with the main ``embeddings`` table; any row left behind here is
        an orphan that keyword search could surface as a dangling hit.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            sql = (
                "SELECT COUNT(*) FROM _pocket_fts_embeddings"
                + (f" WHERE {where}" if where else "")
            )
            return conn.execute(sql).fetchone()[0]
        finally:
            conn.close()

    def test_fts_index_reconciles_on_edit_and_delete(self):
        """The FTS5 lexical index must not accumulate orphans.

        On every run the lexical companion (`_pocket_fts_embeddings`) must hold
        exactly the same row set as `embeddings`. If `_fts_delete_rows` is not
        called when chunks are reconciled away (on edit) or swept (on delete),
        stale BM25 rows linger and keyword search returns dangling hits.
        """
        second = self.source_dir / "second.md"
        second.write_text("# Second\n\nAnother note about pocket knowledge.")

        # First run: FTS index mirrors the main table exactly.
        self._run()
        base_main = self._count_rows()
        self.assertGreater(base_main, 0)
        self.assertEqual(
            self._count_fts_rows(), base_main,
            "FTS index must mirror the main table after the first run",
        )

        # Edit one file so its chunk set changes; orphaned FTS rows must be
        # removed, not merely shadowed by the main-table reconciliation.
        self.note_file.write_text(
            "# Test Note\n\nWholly rewritten body so the chunk ids differ entirely."
        )
        self._run()
        self.assertEqual(
            self._count_fts_rows(), self._count_rows(),
            "FTS index must stay in lockstep after an edit reshapes chunks",
        )

        # Delete a source file: its lexical rows must be swept along with it.
        second.unlink()
        self._run()
        self.assertEqual(
            self._count_fts_rows("row_id IN (SELECT id FROM embeddings "
                                 "WHERE file_path LIKE '%second.md')"),
            0,
            "deleted source must leave no lexical rows",
        )
        self.assertEqual(
            self._count_fts_rows(), self._count_rows(),
            "FTS index must stay in lockstep after a delete sweep",
        )

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
        counting._pix_fn = True
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

    def test_run_reports_stats(self):
        """Monitoring: a run returns UpdateStats with per-component counters that
        reflect adds, then unchanged, then reprocesses, then deletes."""
        from pocket.pipeline import app_main

        second = self.source_dir / "second.md"
        second.write_text("# Second\n\nAnother note about pocket knowledge.")

        def make_app():
            return pix.App(
                "pocket_test",
                app_main,
                sourcedir=self.source_dir,
                db_path=self.db_path,
            )

        # First run: both files are brand-new adds.
        stats = make_app().update_blocking(live=False, report_to_stdout=False)
        self.assertIsNotNone(stats)
        total = stats.total
        self.assertEqual(total.num_adds, 2)
        self.assertEqual(total.num_unchanged, 0)
        self.assertEqual(total.num_deletes, 0)
        self.assertEqual(total.num_errors, 0)
        # Stats are bucketed by the processor name.
        self.assertIn("process_file", stats.by_component)

        # Second run, no edits: everything is unchanged (memoized fast path).
        stats = make_app().update_blocking(live=False, report_to_stdout=False)
        total = stats.total
        self.assertEqual(total.num_unchanged, 2)
        self.assertEqual(total.num_adds, 0)
        self.assertEqual(total.num_reprocesses, 0)

        # Edit one file: it is reprocessed, the other stays unchanged.
        self.note_file.write_text("# Test Note\n\nEdited content for stats check.")
        stats = make_app().update_blocking(live=False, report_to_stdout=False)
        total = stats.total
        self.assertEqual(total.num_reprocesses, 1)
        self.assertEqual(total.num_unchanged, 1)
        self.assertEqual(total.num_adds, 0)

        # Delete one file: it is swept and counted as a delete.
        second.unlink()
        stats = make_app().update_blocking(live=False, report_to_stdout=False)
        total = stats.total
        self.assertEqual(total.num_deletes, 1)

    def test_live_mode_picks_up_new_file(self):
        """Live mode: a file created after the first pass is indexed by a later
        polling pass, then the watcher stops cleanly."""
        import asyncio
        from pocket.pipeline import app_main

        app = pix.App(
            "pocket_test",
            app_main,
            sourcedir=self.source_dir,
            db_path=self.db_path,
        )

        async def drive():
            run = asyncio.create_task(
                app.run_async(
                    live=True, report_to_stdout=False, live_interval=0.2
                )
            )
            # Let the first pass index the existing note, then add a new one.
            await asyncio.sleep(0.3)
            (self.source_dir / "late.md").write_text(
                "# Late\n\nA note added while live mode is running."
            )
            # Give the poller time to catch the new file.
            await asyncio.sleep(0.6)
            run.cancel()
            try:
                await run
            except asyncio.CancelledError:
                pass

        asyncio.run(drive())
        self.assertGreater(
            self._count_rows("file_path LIKE '%late.md'"),
            0,
            "live mode must index files created after startup",
        )

    def test_abort_source_discards_uncommitted_rows(self):
        """Transaction safety (unit): abort_source must roll back rows a
        failed source emitted before they leak into a later source's commit.

        Without abort_source the partial rows of the failed source remain in
        the connection's pending transaction and get persisted by the next
        successful end_source commit; this test fails in that case.
        """
        from dataclasses import dataclass
        from pocketindex.connectors import sqlite
        import pocketindex as pix

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
                tokA = pix._current_source_key.set("A")
                target.begin_source("A")
                target.declare_row(Row(id=1, val="from-A"))
                pix._current_source_key.reset(tokA)
                target.abort_source("A")  # failure path

                # Source B processes cleanly and commits.
                tokB = pix._current_source_key.set("B")
                target.begin_source("B")
                target.declare_row(Row(id=2, val="from-B"))
                pix._current_source_key.reset(tokB)
                target.end_source("B", "hashB")

                ids = {r[0] for r in conn.execute("SELECT id FROM t").fetchall()}
                return ids
            finally:
                await conn.__aexit__(None, None, None)

        import asyncio
        ids = asyncio.run(build())
        self.assertNotIn(1, ids, "aborted source's row must not persist")
        self.assertIn(2, ids, "committed source's row must persist")

    def test_diff_action_uses_statediff_semantics(self):
        """POCKET-P4: the per-row write decision is driven by cocoindex's
        ``statediff.diff`` (insert for a new key, None when already converged,
        replace when the stored row differs)."""
        from dataclasses import dataclass
        from pocketindex.connectors import sqlite
        import asyncio

        self.assertTrue(
            sqlite._HAVE_STATEDIFF,
            "cocoindex statediff must back the delta-write path in this env",
        )

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
                return (
                    target._diff_action("fp", None),     # no stored row yet
                    target._diff_action("fp", "fp"),      # identical -> converged
                    target._diff_action("fp", "other"),   # differs -> rewrite
                )
            finally:
                await conn.__aexit__(None, None, None)

        new_key, unchanged, changed = asyncio.run(build())
        self.assertEqual(new_key, "insert")
        self.assertIsNone(unchanged)
        self.assertEqual(changed, "replace")

    def test_statediff_skips_unchanged_rows_on_redeclare(self):
        """POCKET-P4: re-declaring a source rewrites only the rows that changed.

        Reprocessing must not blindly re-UPSERT every emitted row. Rows whose
        content is byte-identical to what is stored converge to a no-op write,
        so an edit touches just the changed chunk while the unchanged rows (and
        the lexical index) are left alone — yet they are still attributed to the
        source so end_source does not sweep them as orphans.
        """
        from dataclasses import dataclass
        from pocketindex.connectors import sqlite
        import pocketindex as pix
        import asyncio

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

                # First pass: three brand-new rows are all written.
                tok = pix._current_source_key.set("S")
                target.begin_source("S")
                for i in range(3):
                    target.declare_row(Row(id=i, val=f"v{i}"))
                pix._current_source_key.reset(tok)
                target.end_source("S", "h1")
                self.assertEqual(target.num_row_writes, 3)
                self.assertEqual(target.num_row_skips, 0)

                # Second pass: same ids, but only row 1 is edited.
                w0, s0 = target.num_row_writes, target.num_row_skips
                tok = pix._current_source_key.set("S")
                target.begin_source("S")
                target.declare_row(Row(id=0, val="v0"))      # unchanged
                target.declare_row(Row(id=1, val="EDITED"))  # changed
                target.declare_row(Row(id=2, val="v2"))      # unchanged
                pix._current_source_key.reset(tok)
                target.end_source("S", "h2")

                self.assertEqual(
                    target.num_row_writes - w0, 1,
                    "only the edited row is physically written",
                )
                self.assertEqual(
                    target.num_row_skips - s0, 2,
                    "byte-identical rows converge to a no-op skip",
                )

                rows = dict(conn.execute("SELECT id, val FROM t").fetchall())
                return rows
            finally:
                await conn.__aexit__(None, None, None)

        rows = asyncio.run(build())
        self.assertEqual(
            rows, {0: "v0", 1: "EDITED", 2: "v2"},
            "the edit lands and the unchanged rows survive (not swept as orphans)",
        )

    def test_delta_writes_skip_unchanged_rows_in_pipeline(self):
        """POCKET-P4 end to end: forcing a reprocess of unchanged content writes
        zero rows and skips them all, and the engine reports those state-diff
        deltas in its per-component stats."""
        import pocket.pipeline as pipeline

        def make_app():
            return pix.App(
                "pocket_test",
                pipeline.app_main,
                sourcedir=self.source_dir,
                db_path=self.db_path,
            )

        # First run: every chunk is a brand-new write.
        stats = make_app().update_blocking(live=False, report_to_stdout=False)
        base_rows = self._count_rows()
        self.assertGreater(base_rows, 0)
        self.assertEqual(stats.total.num_row_writes, base_rows)
        self.assertEqual(stats.total.num_row_skips, 0)

        # Force a reprocess of every file *without editing it* by disabling memo,
        # so process_file actually re-declares its rows on the second run.
        original = pipeline.process_file

        async def no_memo(file, table):
            return await original(file, table)

        no_memo._pix_fn = True
        no_memo._memo = False
        pipeline.process_file = no_memo
        try:
            stats = make_app().update_blocking(live=False, report_to_stdout=False)
        finally:
            pipeline.process_file = original

        # Unchanged content -> identical chunk ids/text/embeddings -> every
        # re-declared row converges to a no-op skip; nothing is rewritten.
        self.assertEqual(
            stats.total.num_row_writes, 0,
            "reprocessing unchanged content must not rewrite any row",
        )
        self.assertEqual(stats.total.num_row_skips, base_rows)
        self.assertEqual(self._count_rows(), base_rows, "row set is unchanged")

if __name__ == "__main__":
    unittest.main()
