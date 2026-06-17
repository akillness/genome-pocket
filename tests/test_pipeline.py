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

if __name__ == "__main__":
    unittest.main()
