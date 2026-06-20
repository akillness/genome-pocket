"""REST API server for Pocket.

Exposes the same hybrid-retrieval layer used by the CLI and MCP server over
HTTP, so external services and UIs can query the knowledge base without
speaking the Model Context Protocol. Built on Starlette + uvicorn (already
pulled in transitively by the MCP dependency) to avoid adding heavy deps.

Endpoints:
  GET  /health                       -> liveness + index status
  GET  /search?q=...&limit=&mode=    -> hybrid/vector/lexical/graph retrieval
  POST /search  {query, limit, mode} -> same, JSON body
  GET  /lineage?file_path=...        -> per-file chunk lineage
"""
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import pocket.config as config
from pocket import retrieval

_VALID_MODES = {"hybrid", "vector", "lexical", "graph"}


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


def create_app() -> Starlette:
    """Build the Starlette ASGI application."""
    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/search", search_endpoint, methods=["GET", "POST"]),
        Route("/lineage", lineage, methods=["GET"]),
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
