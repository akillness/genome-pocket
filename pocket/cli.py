import json
import os
import click
import pathlib

import pocketindex as pix
from pocket.config import POCKET_SOURCE_DIR, POCKET_SQLITE_DB, EMBEDDING_MODEL
from pocket import retrieval
from pocket.evaluation import METRIC_NAMES



def _get_app_main():
    """Return the pipeline main function.

    Set ``POCKET_PIPELINE=coco`` to use the Phase-4 PoC that runs real
    cocoindex ops (RecursiveSplitter, SentenceTransformerEmbedder) while
    keeping the pocketindex engine.  Default is the standard pipeline.
    """
    if os.environ.get("POCKET_PIPELINE", "").lower() == "coco":
        from pocket.pipeline_coco import app_main as _main
        return _main
    from pocket.pipeline import app_main as _main
    return _main

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
@click.option(
    "--graph",
    is_flag=True,
    help="Also extract a knowledge graph (entities/relations) from notes. "
    "Opt-in; uses the local extractor selected by POCKET_LLM_PROVIDER.",
)
@click.option(
    "--review",
    is_flag=True,
    help="After a --graph build, interactively approve/reject facts the "
    "confidence gate staged as pending (HITL). Ignored without --graph or "
    "in live mode; use 'pocket graph review' for the non-interactive flow.",
)
def update(live, interval, graph, review):
    """Run the indexing pipeline to process notes."""
    click.echo(f"Starting indexing pipeline (live={live}, graph={graph})...")

    if review and not graph:
        click.echo("--review has no effect without --graph; ignoring it.")
    if review and graph and live:
        click.echo("--review is skipped in live mode; run 'pocket graph review' later.")

    # Create the app using the default environment (which has the lifespan registered)
    app = pix.App(
        "pocket",
        _get_app_main(),
        sourcedir=POCKET_SOURCE_DIR,
        db_path=POCKET_SQLITE_DB,
        graph=graph,
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

    if review and graph and not live:
        _interactive_graph_review()


def _interactive_graph_review(echo=click.echo, prompt=click.prompt):
    """Walk the operator through facts the confidence gate staged as pending.

    Reuses the same ``pocket.admin`` review API as ``pocket graph review`` so
    the inline (POCKET-301) and post-hoc flows stay consistent. Offers a bulk
    approve/reject/each/skip choice; ``each`` prompts per fact (approve / reject
    / leave-pending). Anything left unresolved stays pending for later review.
    """
    from pocket import admin

    pending = admin.list_pending()
    ents, rels = pending["entities"], pending["relations"]
    if not ents and not rels:
        echo("No graph facts are pending review.")
        return

    echo(f"\n{len(ents) + len(rels)} fact(s) staged by the confidence gate:")
    for e in ents:
        echo(
            f"  [{e['id']}] {e['name']} ({e['type']}) "
            f"conf={e['confidence']:.2f}  {e['source_file']}"
        )
    for r in rels:
        echo(
            f"  [{r['id']}] {r['subject']} -{r['predicate']}-> {r['object']} "
            f"conf={r['confidence']:.2f}  {r['source_file']}"
        )

    choice = (
        prompt(
            "Review staged facts? [a]pprove all / [r]eject all / [e]ach / [s]kip",
            default="s",
            show_default=False,
        )
        .strip()
        .lower()
    )
    if choice in ("a", "approve"):
        c = admin.approve_pending(ids=None)
        echo(f"Approved {c['entities']} entit(y/ies), {c['relations']} relation(s).")
        return
    if choice in ("r", "reject"):
        c = admin.reject_pending(ids=None)
        echo(f"Rejected {c['entities']} entit(y/ies), {c['relations']} relation(s).")
        return
    if choice not in ("e", "each"):
        echo("Skipped. Run 'pocket graph review' later to resolve pending facts.")
        return

    approve_ids, reject_ids = [], []
    aborted = False
    for label, items in (("entity", ents), ("relation", rels)):
        if aborted:
            break
        for item in items:
            ans = (
                prompt(
                    f"  [{item['id']}] {label}: [y]approve / [n]reject / "
                    "[s]kip / [q]uit",
                    default="s",
                    show_default=False,
                )
                .strip()
                .lower()
            )
            if ans in ("q", "quit"):
                aborted = True
                break
            if ans in ("y", "yes"):
                approve_ids.append(item["id"])
            elif ans in ("n", "no"):
                reject_ids.append(item["id"])
            # anything else: leave pending

    approved = admin.approve_pending(ids=approve_ids) if approve_ids else {"entities": 0, "relations": 0}
    rejected = admin.reject_pending(ids=reject_ids) if reject_ids else {"entities": 0, "relations": 0}
    left = (len(ents) + len(rels)) - len(approve_ids) - len(reject_ids)
    echo(
        f"Approved {approved['entities'] + approved['relations']}, "
        f"rejected {rejected['entities'] + rejected['relations']}, "
        f"{left} still pending."
    )

@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Number of results to return")
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "vector", "lexical", "graph"]),
    default="hybrid",
    help="Retrieval strategy: hybrid (vector+lexical+graph RRF), vector, "
    "lexical, or graph (entity-anchored multi-hop traversal).",
)
@click.option(
    "--mmr/--no-mmr",
    "mmr",
    default=None,
    help="Re-rank results for diversity with Maximal Marginal Relevance "
    "(overrides POCKET_MMR; default follows config).",
)
@click.option(
    "--rerank/--no-rerank",
    "rerank",
    default=None,
    help="Apply a cross-encoder reranker after RRF fusion for higher precision "
    "(overrides POCKET_RERANKER; default follows config).",
)
@click.option(
    "--hyde/--no-hyde",
    "hyde",
    default=None,
    help="Expand the query via HyDE (Hypothetical Document Embeddings) before "
    "vector search (overrides POCKET_HYDE; requires a running Ollama daemon).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit results as a JSON array on stdout (agent-native; "
    "human status lines go to stderr).",
)
def search(query, limit, mode, mmr, rerank, hyde, as_json):
    """Search the indexed notes using hybrid (vector + lexical) retrieval."""
    if as_json:
        # stdout stays pure JSON so a calling agent can parse it; status and
        # error context go to stderr.
        if not POCKET_SQLITE_DB.exists():
            click.echo(
                "Database does not exist. Please run 'pocket update' first.",
                err=True,
            )
            click.echo("[]")
            return
        hits = retrieval.search(query, limit=limit, mode=mode, use_mmr=mmr, use_reranker=rerank, use_hyde=hyde)
        click.echo(
            json.dumps(
                {
                    "query": query,
                    "mode": mode,
                    "count": len(hits),
                    "hits": [h.to_dict() for h in hits],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    click.echo(f"Searching for: '{query}' (mode={mode})...")

    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Please run 'pocket update' first.")
        return

    hits = retrieval.search(query, limit=limit, mode=mode, use_mmr=mmr, use_reranker=rerank, use_hyde=hyde)
    click.echo(retrieval.format_hits(hits))



@cli.command(name="eval")
@click.option("--k", default=5, type=int, help="Cutoff rank for the metrics (top-k).")
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "vector", "lexical", "graph"]),
    default="lexical",
    help="Retrieval mode for synthetic cases (loaded cases keep their own mode). "
    "Defaults to lexical: distinctive-token probes are deterministic there.",
)
@click.option(
    "--cases",
    "cases_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON file of hand-written {query, relevant_files[, mode]} cases. "
    "Without it, cases are synthesized from the index.",
)
@click.option(
    "--per-file", default=1, type=int, help="Synthetic queries to build per source."
)
@click.option(
    "--baseline",
    "baseline_file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Compare this run against a saved baseline; exit non-zero on regression.",
)
@click.option(
    "--save",
    "save_file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write this run's aggregate metrics as a new baseline.",
)
@click.option(
    "--tolerance",
    default=0.0,
    type=float,
    help="Allowed drop versus baseline before a metric counts as a regression.",
)
@click.option("--show-cases", is_flag=True, help="Print per-case hit/miss detail.")
@click.option(
    "--tune",
    is_flag=True,
    help="Grid-search per-strategy RRF weights (POCKET-502) over these cases "
    "and report the best vs the equal-weight baseline instead of a plain run.",
)
@click.option(
    "--tune-metric",
    type=click.Choice(list(METRIC_NAMES)),
    default="mean_average_precision",
    help="Metric the --tune grid search maximizes.",
)

@click.option(
    "--save-weights",
    "weights_file",
    type=click.Path(dir_okay=False),
    default=None,
    help="With --tune, write the winning weights here (point "
    "POCKET_RRF_WEIGHTS_FILE at it to make search/eval use them).",
)
@click.option(
    "--with-judge",
    "with_judge",
    is_flag=True,
    help="After standard IR metrics, score each retrieved chunk's relevance with "
    "a local Ollama LLM judge (RAGAS-style context relevance). "
    "Requires a running Ollama daemon at OLLAMA_HOST.",
)
@click.option(
    "--judge-model",
    default=None,
    help="Ollama model to use as judge (default: POCKET_HYDE_OLLAMA_MODEL).",
)
def eval_cmd(k, mode, cases_file, per_file, baseline_file, save_file, tolerance, show_cases, tune, tune_metric, weights_file, with_judge, judge_model):

    """Evaluate retrieval quality and guard against regressions (POCKET-303).

    Runs standard IR metrics (Hit@k, MRR, Precision/Recall@k, MAP) over either a
    hand-written gold set (--cases) or query/context pairs synthesized from the
    current index. With --baseline it fails (exit 1) if any metric regressed; with
    --save it records the run as a new baseline. With --tune it grid-searches the
    per-strategy fusion weights (POCKET-502) and reports/persists the best.

    """
    from pocket import evaluation

    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Please run 'pocket update' first.")
        raise SystemExit(1)

    if cases_file:
        cases = evaluation.load_cases(pathlib.Path(cases_file))
        click.echo(f"Loaded {len(cases)} case(s) from {cases_file}.")
    else:
        cases = evaluation.synthesize_cases(mode=mode, per_file=per_file)
        click.echo(f"Synthesized {len(cases)} case(s) from the index (mode={mode}).")

    if not cases:
        click.echo("No evaluation cases available (empty index or no cases).")
        raise SystemExit(1)


    if tune:
        result = evaluation.tune_weights(cases, k=k, metric=tune_metric)
        click.echo(
            f"Tuned RRF weights over {len(result.trials)} combination(s) "
            f"(varying {', '.join(result.tuned_strategies) or 'nothing'}), "
            f"maximizing {result.metric}:"
        )
        click.echo(
            f"  baseline (equal weights): {result.baseline_score:.4f}"
        )
        weights_str = ", ".join(
            f"{s}={result.best_weights[s]:g}"
            for s in ("vector", "lexical", "graph")
        )
        click.echo(
            f"  best     ({weights_str}): {result.best_score:.4f} "
            f"({result.delta:+.4f})"
        )
        if not result.improved:
            click.echo("  No weighting beat the equal-weight baseline.")
        if weights_file:
            evaluation.save_weights(pathlib.Path(weights_file), result.best_weights)
            click.echo(
                f"\nSaved weights to {weights_file}. Set "
                f"POCKET_RRF_WEIGHTS_FILE={weights_file} to apply them."
            )
        return

    metrics = evaluation.evaluate(cases, k=k)
    click.echo(evaluation.format_report(metrics, show_cases=show_cases))

    if with_judge:
        judge = evaluation.evaluate_with_judge(
            cases, k=k, ollama_model=judge_model
        )
        click.echo(evaluation.format_judge_report(judge))

    if save_file:
        evaluation.save_baseline(pathlib.Path(save_file), metrics)
        click.echo(f"\nSaved baseline to {save_file}.")

    if baseline_file:
        if not pathlib.Path(baseline_file).exists():
            click.echo(f"\nBaseline {baseline_file} not found; nothing to compare.")
            raise SystemExit(1)
        baseline = evaluation.load_baseline(pathlib.Path(baseline_file))
        regressions = evaluation.compare_to_baseline(metrics, baseline, tolerance)
        if regressions:
            click.echo("\nREGRESSION versus baseline:")
            for r in regressions:
                click.echo(
                    f"  {r.metric}: {r.baseline:.4f} -> {r.current:.4f} "
                    f"({r.delta:+.4f})"
                )
            raise SystemExit(1)
        click.echo("\nNo regression versus baseline.")


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the API server to.")
@click.option("--port", default=8000, type=int, help="Port to bind the API server to.")
def serve(host, port):
    """Serve the knowledge base over a REST API (Starlette + uvicorn)."""
    import uvicorn
    from pocket.api_server import create_app

    click.echo(f"Starting Pocket API server on http://{host}:{port} ...")
    click.echo(f"  Tracing & lineage UI: http://{host}:{port}/")
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


class _DefaultShowGroup(click.Group):
    """`pocket graph <entity>` keeps working: a leading token that is not a
    registered subcommand (and not an option) is routed to `graph show`."""

    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["show"] + list(args)
        return super().resolve_command(ctx, args)


@cli.group(cls=_DefaultShowGroup, invoke_without_command=False)
def graph():
    """Knowledge-graph inspection and review.

    `pocket graph <entity>` prints a node's neighborhood; `pocket graph review`
    manages facts staged by the HITL confidence gate. Requires a graph built
    with `pocket update --graph`.
    """


@graph.command("show")
@click.argument("entity")
@click.option("--limit", default=10, help="Max number of relations to show.")
def graph_show(entity, limit):
    """Print a knowledge-graph entity's neighborhood (relations)."""
    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Please run 'pocket update --graph' first.")
        return
    node = retrieval.graph_neighborhood(entity, limit=limit)
    click.echo(retrieval.format_neighborhood(node))


@graph.command("review")
@click.option("--approve", "approve_ids", multiple=True, help="Approve fact id(s).")
@click.option("--reject", "reject_ids", multiple=True, help="Reject fact id(s).")
@click.option("--approve-all", is_flag=True, help="Approve every pending fact.")
@click.option("--reject-all", is_flag=True, help="Reject every pending fact.")
def graph_review(approve_ids, reject_ids, approve_all, reject_all):
    """Review facts staged by the confidence gate (POCKET-302).

    With no options, lists the pending facts. Use --approve/--reject <id>
    (repeatable) for specific facts, or --approve-all/--reject-all in bulk.
    """
    from pocket import admin

    if not POCKET_SQLITE_DB.exists():
        click.echo("Database does not exist. Please run 'pocket update --graph' first.")
        return

    def _ids(raw):
        out = []
        for r in raw:
            try:
                out.append(int(r))
            except ValueError:
                click.echo(f"Ignoring non-numeric id: {r}")
        return out

    if approve_all and reject_all:
        click.echo("Choose one of --approve-all / --reject-all, not both.")
        return

    acted = False
    if approve_all:
        c = admin.approve_pending(ids=None)
        click.echo(f"Approved {c['entities']} entit(y/ies), {c['relations']} relation(s).")
        acted = True
    elif approve_ids:
        c = admin.approve_pending(ids=_ids(approve_ids))
        click.echo(f"Approved {c['entities']} entit(y/ies), {c['relations']} relation(s).")
        acted = True
    if reject_all:
        c = admin.reject_pending(ids=None)
        click.echo(f"Rejected {c['entities']} entit(y/ies), {c['relations']} relation(s).")
        acted = True
    elif reject_ids:
        c = admin.reject_pending(ids=_ids(reject_ids))
        click.echo(f"Rejected {c['entities']} entit(y/ies), {c['relations']} relation(s).")
        acted = True
    if acted:
        return

    pending = admin.list_pending()
    ents, rels = pending["entities"], pending["relations"]
    if not ents and not rels:
        click.echo("No facts are pending review.")
        return
    click.echo(f"Pending entities ({len(ents)}):")
    for e in ents:
        click.echo(
            f"  [{e['id']}] {e['name']} ({e['type']}) "
            f"conf={e['confidence']:.2f}  {e['source_file']}"
        )
    click.echo(f"Pending relations ({len(rels)}):")
    for r in rels:
        click.echo(
            f"  [{r['id']}] {r['subject']} -{r['predicate']}-> {r['object']} "
            f"conf={r['confidence']:.2f}  {r['source_file']}"
        )
    click.echo(
        "\nApprove with: pocket graph review --approve <id> (or --approve-all); "
        "reject with --reject <id> / --reject-all."
    )

if __name__ == "__main__":
    cli()