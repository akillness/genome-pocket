import os

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
        mode: Retrieval strategy - 'hybrid' (default), 'vector', 'lexical', or 'graph'.
    """
    if not config.POCKET_SQLITE_DB.exists():
        return "Database does not exist. Please run 'pocket update' first."
    if mode not in ("hybrid", "vector", "lexical", "graph"):
        return "mode must be one of: hybrid, vector, lexical, graph."

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
    """List key concepts and relationships from the knowledge graph.

    Queries the graph built by ``pocket update --graph``.  Requires the graph
    tables to be present (i.e. ``POCKET_GRAPH=1`` must have been set during
    indexing).  Returns up to 20 highest-confidence entities with their type
    and top relation.

    Args:
        concept: Optional prefix to filter entity names (case-insensitive).
    """
    if not config.POCKET_SQLITE_DB.exists():
        return "Database does not exist. Please run 'pocket update --graph' first."

    concepts = retrieval.list_graph_concepts(concept=concept, limit=20)
    if not concepts:
        graph_env = os.environ.get("POCKET_GRAPH", "")
        if not graph_env:
            return (
                "No graph data found. Re-index with POCKET_GRAPH=1 enabled: "
                "'POCKET_GRAPH=1 pocket update --graph'"
            )
        return "No concepts found" + (f" matching '{concept}'" if concept else "") + "."

    lines = [f"Concepts ({len(concepts)} found):"]
    for c in concepts:
        rel_str = f"  -> {c['top_relation']}" if c["top_relation"] else ""
        lines.append(
            f"  {c['name']} [{c['type']}, conf={c['confidence']:.2f}, "
            f"src={c['source_file']}]{rel_str}"
        )
    return "\n".join(lines)


@mcp.tool()
def traverse_graph(entity: str, limit: int = 10) -> str:
    """Traverse the knowledge graph from an entity and list its relations.

    Resolves ``entity`` to the nearest node (exact/alias match first, then
    vector similarity over entity-name embeddings) and returns its one-hop
    neighborhood: each relation with direction, predicate, neighbor, evidence
    confidence, and source file. Requires a graph built with
    ``pocket update --graph`` (``POCKET_GRAPH=1``).

    Args:
        entity: The entity name to anchor the traversal on.
        limit: Maximum number of relations to return.
    """
    if not config.POCKET_SQLITE_DB.exists():
        return "Database does not exist. Please run 'pocket update --graph' first."

    node = retrieval.graph_neighborhood(entity, limit=limit)
    if not node:
        graph_env = os.environ.get("POCKET_GRAPH", "")
        if not graph_env:
            return (
                "No graph data found. Re-index with POCKET_GRAPH=1 enabled: "
                "'POCKET_GRAPH=1 pocket update --graph'"
            )
        return f"No matching entity found for '{entity}'."
    return retrieval.format_neighborhood(node)

def main():
    mcp.run()


if __name__ == "__main__":
    main()
