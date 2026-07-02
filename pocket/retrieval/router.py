import re
import sqlite3
import pocket.config as config
from .db import _graph_available

# Which strategies each retrieval mode activates. Single source of truth for the
# router so the CLI, REST API, and tracing UI agree on what "hybrid" means.
_MODE_STRATEGIES = {
    "hybrid": ("vector", "lexical", "graph"),
    "vector": ("vector",),
    "lexical": ("lexical",),
    "graph": ("graph",),
}

# Conservative: only unambiguous code shapes — a false "lexical" route drops vector strategy
_CODE_SHAPE_RE = re.compile(
    r"""
      \b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*\b   # snake_case identifier (parse_payload)
    | \b[a-z]+[A-Z][A-Za-z0-9]*\b              # camelCase identifier (parsePayload)
    | \b[A-Za-z_][A-Za-z0-9_]*\s*\(            # function/method call: foo(
    | ::[A-Za-z_]                              # C++/Rust scope: ns::sym
    | \b[A-Za-z_][A-Za-z0-9_]*\.(py|js|ts|tsx|jsx|go|rs|rb|java|c|cpp|h|hpp|sql|sh|md|json|yaml|yml|toml)\b  # filename.ext
    | [{}\[\];]                                # code punctuation
    | `[^`]+`                                  # an explicit `code span`
    """,
    re.VERBOSE,
)

# Concept/relationship phrasings → graph multi-hop; flat vector/lexical answers them poorly
_CONCEPT_PHRASES = (
    "related to",
    "relationship between",
    "relation between",
    "connection between",
    "connected to",
    "linked to",
    "links between",
    "associated with",
    "depends on",
    "depend on",
    "dependency between",
    "how does",
    "how do",
    "impact of",
    "interact with",
    "interaction between",
    "difference between",
)


def _route_query(query: str) -> str:
    """Classify a query's shape into a concrete retrieval mode (POCKET-504).

    Returns one of ``"lexical"``, ``"graph"``, or ``"hybrid"``. Pure and
    deterministic (regex + keyword shape only, no I/O) so it is unit-testable and
    its routing decision is reproducible.

    Priority: relationship/concept phrasing → ``graph`` first, because a question
    like *"how does write_ahead_log relate to recovery"* is a relationship query
    even though it embeds a code token; then unambiguous code shape → ``lexical``;
    otherwise the ``hybrid`` blend, which is the safe default for prose.
    """
    text = query.strip()
    lowered = text.lower()
    if any(phrase in lowered for phrase in _CONCEPT_PHRASES):
        return "graph"
    if _CODE_SHAPE_RE.search(text):
        return "lexical"
    return "hybrid"


def _resolve_mode(query: str, mode: str, conn: sqlite3.Connection) -> str:
    """Resolve ``mode`` to a concrete strategy, applying the router (POCKET-504).

    ``"auto"`` always routes; a plain ``"hybrid"`` routes only when
    ``config.POCKET_QUERY_ROUTER`` is enabled (opt-in upgrade for existing
    hybrid callers). Any other mode is returned unchanged. A routed ``"graph"``
    falls back to ``"hybrid"`` when the target has no graph tables, so routing
    can never silently return zero results on a graph-less database.
    """
    if mode == "auto" or (mode == "hybrid" and config.POCKET_QUERY_ROUTER):
        routed = _route_query(query)
        if routed == "graph" and not _graph_available(conn):
            return "hybrid"
        return routed
    return mode
