from mcp.server.fastmcp import FastMCP

import pocket.config as config
from pocket import retrieval

mcp = FastMCP("pocket")

@mcp.tool()
def search_knowledge(query: str, limit: int = 5, mode: str = "hybrid") -> str:
    """Search the personal knowledge base using hybrid (vector + lexical) retrieval.

    Args:
        query: The search query.
        limit: Maximum number of results to return.
        mode: Retrieval strategy - 'hybrid' (default), 'vector', or 'lexical'.
    """
    if not config.POCKET_SQLITE_DB.exists():
        return "Database does not exist. Please run 'pocket update' first."
    if mode not in ("hybrid", "vector", "lexical"):
        return "mode must be one of: hybrid, vector, lexical."

    hits = retrieval.search(query, limit=limit, mode=mode)
    return retrieval.format_hits(hits)

@mcp.tool()
def get_file_lineage(file_path: str) -> str:
    """Retrieve the indexing history and lineage details for a specific source file.

    Args:
        file_path: The path to the source file (e.g., 'notes/welcome.md').
    """
    if not config.POCKET_SQLITE_DB.exists():
        return "Database does not exist. Please run 'pocket update' first."

    chunks = retrieval.get_lineage(file_path)
    if not chunks:
        return f"No lineage found for file: {file_path}"

    results = [f"Lineage for {file_path}:\nTotal Chunks: {len(chunks)}\n"]
    for idx, chunk in enumerate(chunks, 1):
        results.append(
            f"Chunk {idx}: ID={chunk['chunk_id']}, "
            f"Range={chunk['start_offset']}-{chunk['end_offset']}\n"
            f"Snippet: {chunk['snippet']}...\n"
        )
    return "\n".join(results)


@mcp.tool()
def list_concepts(concept: str = None) -> str:
    """List key concepts and relationships extracted from the knowledge graph.
    
    Args:
        concept: Optional concept name to filter by.
    """
    return "Graph search and concept extraction are not yet implemented in Sprint 1 (scheduled for Sprint 2)."

def main():
    mcp.run()

if __name__ == "__main__":
    main()
