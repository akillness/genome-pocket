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
@click.option(
    "--interval",
    default=2.0,
    type=float,
    help="Polling interval (seconds) between live-mode passes.",
)
def update(live, interval):
    """Run the indexing pipeline to process notes."""
    click.echo(f"Starting indexing pipeline (live={live})...")

    # Create the app using the default environment (which has the lifespan registered)
    app = pix.App(
        "pocket",
        app_main,
        sourcedir=POCKET_SOURCE_DIR,
        db_path=POCKET_SQLITE_DB,
    )

    # Run the update. The engine prints per-component stats after each pass.
    stats = app.update_blocking(
        live=live, report_to_stdout=True, live_interval=interval
    )
    if stats is not None and not live:
        total = stats.total
        click.echo(
            "Indexing pipeline completed: "
            f"{total.num_adds} added, {total.num_reprocesses} reprocessed, "
            f"{total.num_unchanged} unchanged, {total.num_deletes} deleted, "
            f"{total.num_errors} errors."
        )
    else:
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


@cli.command(name="ls")
def ls_cmd():
    """List indexed source files (stable paths) with their chunk counts."""
    from pocket import retrieval

    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Please run 'pocket update' first.")
        return
    sources = retrieval.list_sources()
    if not sources:
        click.echo("No indexed sources found.")
        return
    click.echo(f"{'CHUNKS':>7}  {'OFFSETS':>15}  SOURCE")
    for s in sources:
        offsets = f"{s['first_offset']}-{s['last_offset']}"
        click.echo(f"{s['chunks']:>7}  {offsets:>15}  {s['file_path']}")
    click.echo(f"\n{len(sources)} source(s) indexed.")


@cli.command()
@click.argument("file_path", required=False)
def show(file_path):
    """Show target state. With no argument, summarize the whole index; with a
    FILE_PATH, show that source's chunk lineage (ids and offsets)."""
    from pocket import retrieval

    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Please run 'pocket update' first.")
        return

    if file_path is None:
        stats = retrieval.target_stats()
        click.echo(f"Database:     {POCKET_SQLITE_DB}")
        click.echo(f"Sources:      {stats['sources']}")
        click.echo(f"Chunks:       {stats['chunks']}")
        click.echo(
            f"Lexical (FTS): {'enabled' if stats['fts_enabled'] else 'disabled'}"
        )
        return

    lineage = retrieval.get_lineage(file_path)
    if not lineage:
        click.echo(f"No chunks found for source: {file_path}")
        return
    click.echo(f"Lineage for {file_path} ({len(lineage)} chunk(s)):")
    for idx, c in enumerate(lineage, 1):
        click.echo(
            f"  Chunk {idx} [id={c['chunk_id']}] "
            f"chars {c['start_offset']}-{c['end_offset']}: {c['snippet']}"
        )


@cli.command()
@click.argument("file_path", required=False)
@click.option(
    "--yes", is_flag=True, help="Skip the confirmation prompt."
)
def drop(file_path, yes):
    """Drop materialized target state. With no argument, reset the entire
    index; with a FILE_PATH, evict only that source's chunks and lineage."""
    from pocket import admin

    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Nothing to drop.")
        return

    if file_path is not None:
        if not yes and not click.confirm(
            f"Drop all chunks for source '{file_path}'?"
        ):
            click.echo("Aborted.")
            return
        result = admin.drop_source(file_path)
        click.echo(f"Removed {result['removed']} chunk(s) for {file_path}.")
        return

    if not yes and not click.confirm(
        "Drop the ENTIRE index (all sources and lineage)?"
    ):
        click.echo("Aborted.")
        return
    result = admin.drop_target()
    if not result["existed"]:
        click.echo("Index was already empty.")
        return
    click.echo(
        f"Dropped {result['chunks']} chunk(s) across {result['sources']} "
        f"source(s). Tables removed: {', '.join(result['dropped'])}."
    )


if __name__ == "__main__":
    cli()