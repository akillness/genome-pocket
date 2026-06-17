import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Configuration values
POCKET_SOURCE_DIR = Path(os.getenv("POCKET_SOURCE_DIR", str(BASE_DIR / "notes")))
POCKET_SQLITE_DB = Path(os.getenv("POCKET_SQLITE_DB", str(BASE_DIR / ".pocket" / "pocket_data.db")))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Ensure directories exist
POCKET_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
POCKET_SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
