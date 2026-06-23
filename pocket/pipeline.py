import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Annotated, AsyncIterator, List, Tuple
import numpy as np
from numpy.typing import NDArray

import pocketindex as pix
from pocketindex.connectors import localfs, sqlite
from pocketindex.ops.sentence_transformers import SentenceTransformerEmbedder, build_embedder
from pocketindex.resources.file import FileLike
from pocketindex.ops.text import RecursiveSplitter, SemanticSplitter, detect_code_language
from pocketindex.ops.refine import TextRefiner
from pocketindex.ops.extract import build_extractor, ExtractedEntity, SqliteExtractionStore
from pocketindex.ops.entity_resolution import resolve_entities, normalize
from pocketindex.resources.id import IdGenerator
from pocketindex.resources.chunk import Chunk

import pocket.config as config


# Context key for the embedding model
EMBEDDER = pix.ContextKey[SentenceTransformerEmbedder]("embedder")
SQLITE_DB = pix.ContextKey[sqlite.ManagedConnection]("sqlite_db")

@dataclass
class ChunkEmbedding:
    id: int                                   # Stable unique ID generated from chunk text
    file_path: str                            # Relative path to the source file
    text: str                                 # The text content of the chunk
    embedding: Annotated[NDArray, EMBEDDER]   # Vector embedding (dimensions inferred from model)
    start_offset: int                         # Character start offset in the source file
    end_offset: int                           # Character end offset in the source file


@dataclass
class EntityNode:
    """A resolved knowledge-graph entity (node). Derived from chunks, so a
    chunk's source file remains the ultimate provenance anchor."""

    id: int                                   # Stable id from canonical name + type
    name: str                                 # Canonical surface form (post-resolution)
    type: str                                 # Proposed type (schema-agnostic)
    aliases: str                              # JSON list of merged surface forms
    embedding: Annotated[NDArray, EMBEDDER]   # Vector of `name` (powers blocking + lookup)
    summary: str                              # Optional one-line description
    confidence: float                         # Max extraction confidence
    source_file: str                          # Primary chunk's source file (lineage anchor)
    source_chunk_ids: str                     # JSON list of chunk ids mentioning this entity
    resolution: str = "[]"                    # JSON merge audit trail (POCKET-404c)
    status: str = "approved"                  # "approved" | "pending" (HITL gate, POCKET-302)


@dataclass
class RelationEdge:
    """A knowledge-graph relation (edge) between two entities."""

    id: int                                   # Stable id from (subject_id, predicate, object_id)
    subject_id: int                           # -> EntityNode.id
    predicate: str                            # Relation type (schema-agnostic, lower_snake)
    object_id: int                            # -> EntityNode.id
    evidence: str                             # Verbatim source span supporting the edge
    confidence: float                         # Extraction confidence
    source_file: str                          # Lineage anchor
    source_chunk_id: int                      # Chunk the edge was extracted from
    status: str = "approved"                  # "approved" | "pending" (HITL gate, POCKET-302)


_splitter = RecursiveSplitter()
_refiner = TextRefiner()

@pix.lifespan
async def pocket_lifespan(builder: pix.EnvironmentBuilder) -> AsyncIterator[None]:
    import os
    # Provide the embedder backend for the active model (text-only
    # SentenceTransformer, or multimodal SigLIP2 when a siglip2 id is selected).
    builder.provide(EMBEDDER, build_embedder(config.EMBEDDING_MODEL))
    # Provide the SQLite ManagedConnection
    db_path = os.getenv("POCKET_SQLITE_DB") or str(config.POCKET_SQLITE_DB)
    builder.provide_with(SQLITE_DB, sqlite.managed_connection(db_path, load_vec=True))
    yield


def _chunk_file(
    raw_text: str,
    filename: pathlib.PurePath,
    embedder=None,
) -> List[Chunk]:
    """Refine + split a file's raw text into offset-exact chunks.

    Shared by the embedding pass and the graph-extraction pass so both attribute
    facts to the same chunk ids and the same source offsets.

    When *embedder* is supplied and ``POCKET_SEMANTIC_SPLIT`` is enabled, prose
    and markdown files are split by embedding-guided semantic boundaries instead
    of fixed character counts (SemanticSplitter).  Code files (is_code=True)
    always use the language-aware RecursiveSplitter regardless — sentence
    similarity boundaries are meaningless for source code.
    """
    language = detect_code_language(filename=filename.name)
    is_code = language is not None and language not in ("markdown", "html")
    refined = _refiner.refine(raw_text, code=is_code)
    if config.POCKET_SEMANTIC_SPLIT and not is_code and embedder is not None:
        _sem_splitter = SemanticSplitter(
            model=getattr(embedder, "model", None),
            breakpoint_threshold=config.POCKET_SEMANTIC_SPLIT_THRESHOLD,
        )
        chunks = _sem_splitter.split(refined.text, language=language)
    else:
        chunks = _splitter.split(
            refined.text, chunk_size=1000, chunk_overlap=200, language=language
        )
    for chunk in chunks:
        chunk.start.char_offset = refined.source_offset(chunk.start.char_offset)
        chunk.end.char_offset = refined.source_offset(chunk.end.char_offset)
    return chunks


@pix.fn
async def process_chunk(
    chunk: Chunk,
    filename: pathlib.PurePath,
    id_gen: IdGenerator,
    table: sqlite.TableTarget[ChunkEmbedding],
) -> None:
    embedder = pix.use_context(EMBEDDER)
    embedding = await embedder.embed(chunk.text)
    
    table.declare_row(row=ChunkEmbedding(
        id=await id_gen.next_id(chunk.text),
        file_path=str(filename),
        text=chunk.text,
        embedding=embedding,
        start_offset=chunk.start.char_offset,
        end_offset=chunk.end.char_offset,
    ))

@pix.fn(memo=True)
async def process_file(file: FileLike, table: sqlite.TableTarget[ChunkEmbedding]) -> None:
    # Image files take the multimodal path when the active embedder supports it;
    # otherwise (text-only model) the image source is seen for lineage but emits
    # no rows. Routing here (not as a second mount_each) keeps a single sweep over
    # the shared `embeddings` target so neither modality garbage-collects the other.
    if getattr(file, "is_image", False):
        embedder = pix.use_context(EMBEDDER)
        if not getattr(embedder, "supports_image", False):
            return
        filename = file.file_path.path
        # An image is one atomic unit (no refine/split) -> exactly one row whose
        # vector is the SigLIP2 image embedding. `text` is the relative path so the
        # hit still has a lexical handle and a lineage citation.
        embedding = await embedder.embed_image(filename)
        id_gen = IdGenerator()
        path_str = str(filename)
        table.declare_row(row=ChunkEmbedding(
            id=await id_gen.next_id(path_str),
            file_path=path_str,
            text=path_str,
            embedding=embedding,
            start_offset=0,
            end_offset=0,
        ))
        return

    raw_text = await file.read_text()
    # Detect whether this is source code from its filename. Code files get an
    # indentation-preserving refine pass and language-aware (structural)
    # splitting; prose/markdown keeps the original whitespace-collapsing path.
    # Pass the embedder so SemanticSplitter can use the already-loaded model
    # when POCKET_SEMANTIC_SPLIT=1 (avoids a second model load).
    _embedder = pix.use_context(EMBEDDER) if config.POCKET_SEMANTIC_SPLIT else None
    chunks = _chunk_file(raw_text, file.file_path.path, embedder=_embedder)
    id_gen = IdGenerator()

    await pix.map(process_chunk, chunks, file.file_path.path, id_gen, table)


@pix.fn(memo=True)
async def extract_graph_file(
    file: FileLike,
    entities_target: sqlite.TableTarget[EntityNode],
    relations_target: sqlite.TableTarget[RelationEdge],
) -> None:
    """Extract a file's knowledge subgraph (entities + relations).

    Runs as its own memoized pass (primary lineage target = entities) so it is
    independent of the embedding pass: enabling ``--graph`` on an already-indexed
    corpus still extracts, and an unchanged file is skipped by the entities memo.

    The relations target shares the same ``_current_source_key`` (set by the
    engine for this file), so we drive its ``begin_source`` / ``end_source``
    manually here; ``app_main`` sweeps it after the pass.
    """
    source_key = pix.get_current_source_key()
    relations_target.begin_source(source_key)

    raw_text = await file.read_text()
    filename = file.file_path.path
    chunks = _chunk_file(raw_text, filename)

    # Persist the extraction cache (POCKET-404b) in the same SQLite DB for the
    # LLM backends so an unchanged chunk under an unchanged prompt is never
    # re-sent across runs. The deterministic backend is pure/cheap and stays
    # unwrapped, so default runs gain no new table.
    provider = (config.POCKET_LLM_PROVIDER or "deterministic").lower()
    store = None
    if provider in ("ollama", "airllm"):
        store = SqliteExtractionStore(pix.use_context(SQLITE_DB).conn)
    extractor = build_extractor(provider, config.POCKET_LLM_MODEL, store=store)
    id_gen = IdGenerator()

    # Collect every extracted entity across the file's chunks, remembering which
    # chunk each came from so resolved nodes can carry their source_chunk_ids and
    # edges can reference the originating chunk.
    all_entities: List[ExtractedEntity] = []
    entity_chunk_id: dict = {}      # id(entity) -> chunk id
    relations: list = []            # (ExtractedRelation, chunk_id)
    for chunk in chunks:
        chunk_id = await id_gen.next_id(chunk.text)
        extraction = extractor.extract(chunk.text)
        for ent in extraction.entities:
            all_entities.append(ent)
            entity_chunk_id[id(ent)] = chunk_id
        for rel in extraction.relations:
            relations.append((rel, chunk_id))

    if not all_entities:
        relations_target.end_source(source_key, "")
        return

    embedder = pix.use_context(EMBEDDER)
    # Embed each entity name to drive blocking-based resolution and vector lookup.
    embeddings = [list(map(float, await embedder.embed(e.name))) for e in all_entities]

    resolved = resolve_entities(all_entities, embeddings)

    # Map every surface form (normalized) -> canonical entity id so edges can be
    # rewired to canonical endpoints before they are committed.
    surface_to_id: dict = {}
    id_to_node: dict = {}
    min_conf = config.POCKET_GRAPH_MIN_CONFIDENCE
    for r in resolved:
        ent_id = await id_gen.next_id(f"{r.name}\x00{r.type}")
        chunk_ids = sorted(
            {entity_chunk_id[id(m)] for m in r.members if id(m) in entity_chunk_id}
        )
        # Average the canonical name's embedding for the node vector.
        node_vec = await embedder.embed(r.name)
        node = EntityNode(
            id=ent_id,
            name=r.name,
            type=r.type,
            aliases=json.dumps(r.aliases),
            embedding=node_vec,
            summary="",
            confidence=r.confidence,
            source_file=str(filename),
            source_chunk_ids=json.dumps(chunk_ids),
            resolution=json.dumps([asdict(m) for m in r.merges]),
        )
        id_to_node[ent_id] = node
        for member in r.members:
            surface_to_id[normalize(member.name)] = ent_id

    # Commit entity nodes. The HITL gate (POCKET-302) does NOT drop low-confidence
    # facts; it stages them as ``status="pending"`` so ``pocket graph review`` can
    # approve or reject them. Retrieval defaults to approved-only, so pending facts
    # stay out of search results until a human accepts them.
    for node in id_to_node.values():
        node.status = "approved" if node.confidence >= min_conf else "pending"
        entities_target.declare_row(row=node)

    # Rewire and commit edges to canonical entity ids. Edges below the gate, or
    # whose endpoints are themselves staged, are written as ``status="pending"``
    # (not dropped) so the whole low-confidence fact is reviewable together.
    seen_edge_ids = set()
    for rel, chunk_id in relations:
        subj_id = surface_to_id.get(normalize(rel.subject))
        obj_id = surface_to_id.get(normalize(rel.object))
        if subj_id is None or obj_id is None or subj_id == obj_id:
            continue
        # Both endpoints must actually be extracted nodes.
        if subj_id not in id_to_node or obj_id not in id_to_node:
            continue
        edge_id = await id_gen.next_id(f"{subj_id}\x00{rel.predicate}\x00{obj_id}")
        if edge_id in seen_edge_ids:
            continue
        seen_edge_ids.add(edge_id)
        endpoints_pending = (
            id_to_node[subj_id].status == "pending"
            or id_to_node[obj_id].status == "pending"
        )
        edge_status = (
            "approved"
            if rel.confidence >= min_conf and not endpoints_pending
            else "pending"
        )
        relations_target.declare_row(row=RelationEdge(
            id=edge_id,
            subject_id=subj_id,
            predicate=rel.predicate,
            object_id=obj_id,
            evidence=rel.evidence[:500],
            confidence=rel.confidence,
            source_file=str(filename),
            source_chunk_id=chunk_id,
            status=edge_status,
        ))

    # The entities target's memo (set by the engine) is the incremental key; the
    # relations target only needs its lineage reconciled, so pass an empty hash.
    relations_target.end_source(source_key, "")

@pix.fn
async def app_main(
    sourcedir: pathlib.Path,
    db_path: pathlib.Path,
    graph: bool = False,
) -> None:
    # 1. Mount SQLite target table
    target_table = await sqlite.mount_table_target(
        SQLITE_DB,
        table_name="embeddings",
        table_schema=await sqlite.TableSchema.from_class(ChunkEmbedding, primary_key=["id"]),
        # Mirror chunk text into an FTS5 index so the same load supports both
        # vector (sqlite-vec) and lexical (BM25) retrieval.
        fts_text_column="text",
    )
    
    # 2. Walk source directory
    files = localfs.walk_dir(sourcedir, recursive=True, live=True)
    
    # 3. Mount file processing component (vector/lexical pass).
    await pix.mount_each(process_file, files.items(), target_table)

    # 4. Optional graph branch (POCKET-404): extract entities/relations. Opt-in
    #    via `--graph` or POCKET_GRAPH so default runs are byte-for-byte unchanged.
    if graph or config.POCKET_GRAPH:
        entities_target = await sqlite.mount_table_target(
            SQLITE_DB,
            table_name="entities",
            table_schema=await sqlite.TableSchema.from_class(EntityNode, primary_key=["id"]),
            fts_text_column="name",
        )
        relations_target = await sqlite.mount_table_target(
            SQLITE_DB,
            table_name="relations",
            table_schema=await sqlite.TableSchema.from_class(RelationEdge, primary_key=["id"]),
        )
        # Graph extraction is text-only; images have no read_text() path, so they
        # are excluded from the entity/relation pass.
        graph_items = {
            k: v for k, v in files.items().items()
            if not getattr(v, "is_image", False)
        }
        # The graph pass's primary lineage/memo target is `entities`; relations are
        # driven manually inside the component and swept here afterwards.
        await pix.mount_each(
            extract_graph_file, graph_items, entities_target, relations_target
        )
        relations_target.sweep({str(k) for k in graph_items.keys()})
