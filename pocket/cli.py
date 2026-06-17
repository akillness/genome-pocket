import click
import pathlib
import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer

import cocoindex as coco
from pocket.config import POCKET_SOURCE_DIR, POCKET_SQLITE_DB, EMBEDDING_MODEL
from pocket.pipeline import app_main

@click.group()
def cli():
    """Pocket Knowledge Ops CLI."""
    pass

@cli.command()
def init():
    """Initialize the notes directory and a welcome note."""
    POCKET_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    welcome_file = POCKET_SOURCE_DIR / "welcome.md"
    if not welcome_file.exists():
        welcome_file.write_text(
            "# Welcome to Pocket\n\n"
            "Pocket is a local-first personal Knowledge Ops runtime with a self-contained incremental ETL engine.\n"
            "It indexes your markdown notes and provides semantic search capabilities.\n"
        )
        click.echo(f"Initialized welcome note at {welcome_file}")
    else:
        click.echo(f"Welcome note already exists at {welcome_file}")

@cli.command()
@click.option("-L", "--live", is_flag=True, help="Run in live mode (watch for changes)")
def update(live):
    """Run the indexing pipeline to process notes."""
    click.echo(f"Starting indexing pipeline (live={live})...")
    
    # Create the app using the default environment (which has the lifespan registered)
    app = coco.App(
        "pocket",
        app_main,
        sourcedir=POCKET_SOURCE_DIR,
        db_path=POCKET_SQLITE_DB,
    )
    
    # Run the update
    app.update_blocking(live=live, report_to_stdout=True)
    click.echo("Indexing pipeline completed.")

@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Number of results to return")
def search(query, limit):
    """Search the indexed notes using semantic vector search."""
    click.echo(f"Searching for: '{query}'...")
    
    # Embed the query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_vector = sqlite_vec.serialize_float32(query_embedding)
    
    # Connect to the database
    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Please run 'pocket update' first.")
        return
        
    conn = sqlite3.connect(str(POCKET_SQLITE_DB))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cursor = conn.execute("""
        SELECT file_path, text, start_offset, end_offset, vec_distance_cosine(embedding, ?) AS distance
        FROM embeddings
        ORDER BY distance ASC
        LIMIT ?
    """, (query_vector, limit))
    
    rows = cursor.fetchall()
    if not rows:
        click.echo("No results found.")
        return
        
    for idx, (file_path, text, start_offset, end_offset, distance) in enumerate(rows, 1):
        similarity = 1.0 - distance
        click.echo(f"\n[{idx}] File: {file_path} (chars {start_offset}-{end_offset}) [Similarity: {similarity:.4f}]")
        click.echo("-" * 40)
        click.echo(text.strip())
        click.echo("-" * 40)

if __name__ == "__main__":
    cli()
