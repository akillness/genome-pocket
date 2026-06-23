"""REST API server for Pocket.

Exposes the same hybrid-retrieval layer used by the CLI and MCP server over
HTTP, so external services and UIs can query the knowledge base without
speaking the Model Context Protocol. Built on Starlette + uvicorn (already
pulled in transitively by the MCP dependency) to avoid adding heavy deps.

Endpoints:
  GET  /                              -> local tracing & lineage web UI (HTML)
  GET  /health                       -> liveness + index status
  GET  /search?q=...&limit=&mode=    -> auto/hybrid/vector/lexical/graph retrieval

  POST /search  {query, limit, mode} -> same, JSON body
  GET  /trace?q=...&limit=&mode=     -> query-routing trace + contributing chunks
  GET  /lineage?file_path=...        -> per-file chunk lineage
  GET  /pending                      -> graph facts staged for HITL review
  POST /pending/{approve|reject}     -> commit/discard staged facts (JSON {ids?})
"""
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

import pocket.config as config
from pocket import admin, retrieval
from pocket.web_ui import INDEX_HTML

# "auto" engages the semantic query router (POCKET-504): the mode is picked from
# the query's shape (code -> lexical, relationship -> graph, else hybrid).
_VALID_MODES = {"auto", "hybrid", "vector", "lexical", "graph"}



def _index_exists() -> bool:
    return config.POCKET_SQLITE_DB.exists()


async def health(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "index_ready": _index_exists(),
            "db_path": str(config.POCKET_SQLITE_DB),
            "embedding_model": config.EMBEDDING_MODEL,
        }
    )


def _coerce_limit(raw, default: int = 5) -> int:
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(limit, 50))


def _run_search(query: str, limit: int, mode: str) -> JSONResponse:
    if not query or not query.strip():
        return JSONResponse({"error": "query must not be empty"}, status_code=400)
    if mode not in _VALID_MODES:
        return JSONResponse(
            {"error": f"mode must be one of {sorted(_VALID_MODES)}"}, status_code=400
        )
    if not _index_exists():
        return JSONResponse(
            {"error": "index not built; run 'pocket update' first"}, status_code=503
        )
    hits = retrieval.search(query, limit=limit, mode=mode)
    return JSONResponse(
        {
            "query": query,
            "mode": mode,
            "count": len(hits),
            "results": [h.to_dict() for h in hits],
        }
    )


async def search_endpoint(request: Request) -> JSONResponse:
    """Handle both GET (query string) and POST (JSON body) searches."""
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        query = body.get("query", "")
        limit = _coerce_limit(body.get("limit"))
        mode = body.get("mode", "hybrid")
    else:
        query = request.query_params.get("q", "")
        limit = _coerce_limit(request.query_params.get("limit"))
        mode = request.query_params.get("mode", "hybrid")
    return _run_search(query, limit, mode)


async def lineage(request: Request) -> JSONResponse:
    file_path = request.query_params.get("file_path", "")
    if not file_path:
        return JSONResponse({"error": "file_path is required"}, status_code=400)
    if not _index_exists():
        return JSONResponse(
            {"error": "index not built; run 'pocket update' first"}, status_code=503
        )
    chunks = retrieval.get_lineage(file_path)
    return JSONResponse(
        {"file_path": file_path, "chunk_count": len(chunks), "chunks": chunks}
    )


async def index(request: Request) -> HTMLResponse:
    """Serve the local tracing & lineage web UI (POCKET-301)."""
    return HTMLResponse(INDEX_HTML)


async def trace(request: Request) -> JSONResponse:
    """Explain how a query is routed and which chunks each strategy surfaced."""
    query = request.query_params.get("q", "")
    if not query or not query.strip():
        return JSONResponse({"error": "query must not be empty"}, status_code=400)
    mode = request.query_params.get("mode", "hybrid")
    if mode not in _VALID_MODES:
        return JSONResponse(
            {"error": f"mode must be one of {sorted(_VALID_MODES)}"}, status_code=400
        )
    if not _index_exists():
        return JSONResponse(
            {"error": "index not built; run 'pocket update' first"}, status_code=503
        )
    limit = _coerce_limit(request.query_params.get("limit"))
    return JSONResponse(retrieval.routing_trace(query, limit=limit, mode=mode))


# ids are 64-bit hashes (> 2**53) — stringify for JSON so JS doesn't corrupt them
def _stringify_pending(pending: dict) -> dict:
    return {
        key: [{**row, "id": str(row["id"])} for row in pending.get(key, [])]
        for key in ("entities", "relations")
    }


def _parse_ids(body: dict):
    """Return ``(ids, error)``: ``ids`` is ``None`` (=all) or a list of ints."""
    raw = body.get("ids")
    if raw is None:
        return None, None
    if not isinstance(raw, list):
        return None, "ids must be a list of integer-valued strings or null"
    ids = []
    for value in raw:
        if isinstance(value, bool):
            return None, "ids must be integer-valued strings or integers"
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            return None, "ids must be integer-valued strings or integers"
    return ids, None


async def pending(request: Request) -> JSONResponse:
    """List graph facts the confidence gate staged for HITL review (POCKET-505)."""
    if not _index_exists():
        return JSONResponse(
            {"error": "index not built; run 'pocket update' first"}, status_code=503
        )
    return JSONResponse(_stringify_pending(admin.list_pending()))


async def pending_review(request: Request) -> JSONResponse:
    """Approve or reject staged facts; ``{ids: [...]}`` or omit/``null`` for all."""
    action = request.path_params.get("action")
    if action not in ("approve", "reject"):
        return JSONResponse(
            {"error": "action must be 'approve' or 'reject'"}, status_code=404
        )
    if not _index_exists():
        return JSONResponse(
            {"error": "index not built; run 'pocket update' first"}, status_code=503
        )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    ids, err = _parse_ids(body)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    review = admin.approve_pending if action == "approve" else admin.reject_pending
    counts = review(ids)
    return JSONResponse({"action": action, **counts})


def create_app() -> Starlette:
    """Build the Starlette ASGI application."""
    routes = [
        Route("/", index, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/search", search_endpoint, methods=["GET", "POST"]),
        Route("/trace", trace, methods=["GET"]),
        Route("/lineage", lineage, methods=["GET"]),
        Route("/pending", pending, methods=["GET"]),
        Route("/pending/{action}", pending_review, methods=["POST"]),
    ]
    return Starlette(routes=routes)


app = create_app()


def main() -> None:
    """Console-script entry point: run the API server with uvicorn."""
    import os
    import uvicorn

    host = os.getenv("POCKET_API_HOST", "127.0.0.1")
    port = int(os.getenv("POCKET_API_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
