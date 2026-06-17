import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

from pocket.config import POCKET_SQLITE_DB, EMBEDDING_MODEL

mcp = FastMCP("pocket")

@mcp.tool()
def search_knowledge(query: str, limit: int = 5) -> str:
    """Search the personal knowledge base using semantic vector search.
    
    Args:
        query: The search query.
        limit: Maximum number of results to return.
    """
    if not POCKET_SQLITE_DB.exists():
        return "Database does not exist. Please run 'pocket update' first."
        
    # Embed the query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_vector = sqlite_vec.serialize_float32(query_embedding)
    
    # Connect to the database
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
        return "No results found."
        
    results = []
    for idx, (file_path, text, start_offset, end_offset, distance) in enumerate(rows, 1):
        similarity = 1.0 - distance
        results.append(
            f"[{idx}] File: {file_path} (chars {start_offset}-{end_offset}) [Similarity: {similarity:.4f}]\n"
            f"Content:\n{text.strip()}\n"
            f"{'='*40}"
        )
    return "\n\n".join(results)

@mcp.tool()
def get_file_lineage(file_path: str) -> str:
    """Retrieve the indexing history and lineage details for a specific source file.
    
    Args:
        file_path: The path to the source file (e.g., 'notes/welcome.md').
    """
    if not POCKET_SQLITE_DB.exists():
        return "Database does not exist. Please run 'pocket update' first."
        
    conn = sqlite3.connect(str(POCKET_SQLITE_DB))
    cursor = conn.execute("""
        SELECT id, start_offset, end_offset, SUBSTR(text, 1, 100) AS snippet
        FROM embeddings
        WHERE file_path = ?
        ORDER BY start_offset ASC
    """, (file_path,))
    
    rows = cursor.fetchall()
    if not rows:
        return f"No lineage found for file: {file_path}"
        
    results = [f"Lineage for {file_path}:\nTotal Chunks: {len(rows)}\n"]
    for idx, (chunk_id, start_offset, end_offset, snippet) in enumerate(rows, 1):
        results.append(
            f"Chunk {idx}: ID={chunk_id}, Range={start_offset}-{end_offset}\n"
            f"Snippet: {snippet.strip()}...\n"
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
