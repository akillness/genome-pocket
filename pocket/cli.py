import click
import pathlib

import pocketindex as pix
from pocket.config import POCKET_SOURCE_DIR, POCKET_SQLITE_DB, EMBEDDING_MODEL
from pocket.pipeline import app_main
from pocket import retrieval

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
    app = pix.App(
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
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "vector", "lexical"]),
    default="hybrid",
    help="Retrieval strategy: hybrid (vector+lexical RRF), vector, or lexical.",
)
def search(query, limit, mode):
    """Search the indexed notes using hybrid (vector + lexical) retrieval."""
    click.echo(f"Searching for: '{query}' (mode={mode})...")

    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Please run 'pocket update' first.")
        return

    hits = retrieval.search(query, limit=limit, mode=mode)
    click.echo(retrieval.format_hits(hits))


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the API server to.")
@click.option("--port", default=8000, type=int, help="Port to bind the API server to.")
def serve(host, port):
    """Serve the knowledge base over a REST API (Starlette + uvicorn)."""
    import uvicorn
    from pocket.api_server import create_app

    click.echo(f"Starting Pocket API server on http://{host}:{port} ...")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    cli()