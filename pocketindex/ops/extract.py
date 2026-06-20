"""Knowledge-graph extraction op for PocketIndex.

Turns a chunk of text into ``(entities, relations)`` — the building blocks of
the local-first knowledge graph described in ``docs/architecture/graph-target.md``.

Design stance (see the spec):

* **Local-first, no hosted proxy.** Where the upstream ``cocoindex.ops.litellm``
  proxies a *hosted* API, Pocket re-homes extraction onto **local** engines. Three
  backends implement one ``ExtractionModel`` protocol:

  - ``DeterministicExtractor`` — the default. No LLM, no network, no heavy deps.
    A noun-phrase + co-occurrence extractor so the whole graph pipeline is
    offline-testable (the POCKET-404a slice).
  - ``OllamaExtractor`` — talks to a local Ollama daemon over HTTP.
  - ``AirLLMExtractor`` — runs a large model in-process via ``airllm`` (layer-sharded
    inference: a 70B-class model on a single 4GB GPU, no quantization accuracy loss).
    airLLM/torch are an optional extra and imported lazily.

* **Schema-agnostic.** Backends propose entity types and predicates; we do not pin
  an ontology. Every entity/relation carries a ``confidence`` and a verbatim
  ``evidence`` span for the HITL gate and lineage.

* **Degrade, don't crash.** Malformed LLM output drops that chunk's extraction and
  logs, mirroring the FTS5 fallback posture elsewhere in the engine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# Extraction result shape (backend-agnostic).
# --------------------------------------------------------------------------- #
@dataclass
class ExtractedEntity:
    name: str
    type: str = "Concept"
    confidence: float = 1.0
    evidence: str = ""


@dataclass
class ExtractedRelation:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    evidence: str = ""


@dataclass
class Extraction:
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)


@runtime_checkable
class ExtractionModel(Protocol):
    """Every backend turns a chunk of text into an :class:`Extraction`."""

    def extract(self, text: str) -> Extraction:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Shared text helpers (used by the deterministic backend and JSON validation).
# --------------------------------------------------------------------------- #
# Stop words kept tiny and dependency-free; this is a heuristic extractor, not a
# linguistics engine. It exists to exercise the graph plumbing offline.
_STOP_WORDS = frozenset(
    """
    a an and are as at be but by for from has have in into is it its of on or
    that the their then there these this to was were will with not no can may
    """.split()
)

# A "noun phrase" heuristic: runs of Capitalized words, or quoted/backticked
# spans, or snake/camel identifiers — the kinds of tokens that name things in
# notes and code.
_CAPITAL_PHRASE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[ \-][A-Z][A-Za-z0-9]*)*)\b")
_CODE_IDENT = re.compile(r"`([^`]+)`|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def _is_meaningful(token: str) -> bool:
    low = token.lower()
    return len(token) > 2 and low not in _STOP_WORDS


# --------------------------------------------------------------------------- #
# Backend 1: deterministic (default, offline, no LLM).
# --------------------------------------------------------------------------- #
class DeterministicExtractor:
    """A dependency-free, deterministic extractor.

    Pulls candidate entities from capitalized phrases and code identifiers, then
    proposes a ``mentioned_with`` relation between entities that co-occur in the
    same sentence. Confidence is fixed and modest (0.5) to reflect that this is a
    heuristic, so by default these facts land *below* the HITL threshold and are
    staged rather than committed when the gate is enabled.

    This backend makes the entire graph pipeline testable with no network, no GPU,
    and no model download — the POCKET-404a slice.
    """

    name = "deterministic"

    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence

    def _candidate_entities(self, sentence: str) -> List[ExtractedEntity]:
        found: dict[str, ExtractedEntity] = {}
        for match in _CAPITAL_PHRASE.finditer(sentence):
            phrase = _normalize_name(match.group(1))
            if _is_meaningful(phrase) and phrase.lower() not in found:
                found[phrase.lower()] = ExtractedEntity(
                    name=phrase,
                    type="Concept",
                    confidence=self.confidence,
                    evidence=sentence.strip(),
                )
        for match in _CODE_IDENT.finditer(sentence):
            ident = _normalize_name(match.group(1) or match.group(2) or "")
            if ident and _is_meaningful(ident) and ident.lower() not in found:
                found[ident.lower()] = ExtractedEntity(
                    name=ident,
                    type="Symbol",
                    confidence=self.confidence,
                    evidence=sentence.strip(),
                )
        return list(found.values())

    def extract(self, text: str) -> Extraction:
        entities: dict[str, ExtractedEntity] = {}
        relations: List[ExtractedRelation] = []
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            sent_entities = self._candidate_entities(sentence)
            for ent in sent_entities:
                entities.setdefault(ent.name.lower(), ent)
            # Co-occurrence edges between distinct entities in this sentence.
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    a, b = sent_entities[i], sent_entities[j]
                    relations.append(
                        ExtractedRelation(
                            subject=a.name,
                            predicate="mentioned_with",
                            object=b.name,
                            confidence=self.confidence,
                            evidence=sentence,
                        )
                    )
        return Extraction(entities=list(entities.values()), relations=relations)


# --------------------------------------------------------------------------- #
# Strict-JSON extraction prompt (POCKET-404b).
# --------------------------------------------------------------------------- #
# PROMPT_VERSION is part of the extraction memo key: bumping it invalidates every
# cached extraction so a prompt change forces re-extraction (see MemoizingExtractor).
# Bump this whenever _EXTRACTION_PROMPT changes in a way that should re-run models.
PROMPT_VERSION = "2026.1"

# Hardened against the 2026 GraphRAG / KG-construction literature surveyed in
# docs/architecture/graph-target.md §1:
#   * STRICT JSON-only output — small local (7-9B) models reach high task accuracy
#     yet emit *invalid* JSON without an explicit system contract: naive prompting
#     scores 0% valid-output in the small-model structured-output reliability
#     benchmark (arXiv:2605.02363), so we demand "no fence, no prose". Local-LLM
#     GraphRAG is viable at all on consumer hardware per arXiv:2605.20815.
#   * Schema-agnostic types/predicates (arXiv:2606.01208) — propose, don't pin;
#     descriptive lower_snake_case keys also act as an instruction channel that
#     steers generation (arXiv:2604.14862).
#   * Calibrated confidence (uncertainty-guided KG construction, arXiv:2605.26835)
#     so the HITL gate (POCKET-302) gets a real signal instead of a flat 1.0.
#   * Mandatory verbatim evidence span (verifiability, arXiv:2606.01210) so every
#     fact can be audited against its source before it is committed.
#   * A single grounded few-shot exemplar to lift JSON validity on small models
#     (arXiv:2605.02363) at low token cost. Caveat: any format demand carries a
#     "format tax" on reasoning (arXiv:2604.03616) — mitigated here by a tight
#     schema + tolerant parser; a freeform-then-reformat two-pass is a future option.
_EXTRACTION_PROMPT = """You are a precise knowledge-graph extraction engine. \
Read the TEXT and return ONLY a single JSON object — no prose, no markdown, no \
code fence — matching EXACTLY this schema:
{"entities":[{"name":str,"type":str,"confidence":number,"evidence":str}],
 "relations":[{"subject":str,"predicate":str,"object":str,"confidence":number,"evidence":str}]}
Rules:
1. GROUNDING: extract only facts the TEXT explicitly supports; never invent. If \
the TEXT supports nothing, return {"entities":[],"relations":[]}.
2. EVIDENCE: every entity and relation MUST carry a short verbatim "evidence" \
span copied from the TEXT that justifies it.
3. CONFIDENCE: set "confidence" in [0,1] calibrated to how unambiguously the TEXT \
states the fact (1.0 = explicit, lower = inferred). Do not default everything to 1.0.
4. SCHEMA-AGNOSTIC: propose entity "type" labels and "predicate" names freely; do \
not force a fixed ontology. Write predicates in lower_snake_case.
5. Each relation's "subject" and "object" MUST also appear as an entity "name".
Example
TEXT: Pocket stores embeddings in SQLite.
JSON: {"entities":[{"name":"Pocket","type":"Tool","confidence":1.0,"evidence":"Pocket stores embeddings in SQLite."},{"name":"SQLite","type":"Tool","confidence":1.0,"evidence":"Pocket stores embeddings in SQLite."}],"relations":[{"subject":"Pocket","predicate":"stores_embeddings_in","object":"SQLite","confidence":0.9,"evidence":"Pocket stores embeddings in SQLite."}]}
TEXT:
"""


def _coerce_confidence(value, default: float = 0.8) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, c))


def parse_extraction_json(raw: str, evidence: str) -> Extraction:
    """Validate an LLM's JSON reply into an :class:`Extraction`.

    Tolerant of a leading/trailing code fence; raises ``ValueError`` on anything
    that isn't the expected object shape so the caller can degrade gracefully.
    """
    text = raw.strip()
    # Grab the outermost JSON object — this alone tolerates a leading/trailing
    # code fence or prose wrapper, so no brittle pre-strip pass is needed.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output was not a JSON object")

    entities: List[ExtractedEntity] = []
    for e in data.get("entities", []) or []:
        if not isinstance(e, dict) or not e.get("name"):
            continue
        entities.append(
            ExtractedEntity(
                name=_normalize_name(str(e["name"])),
                type=str(e.get("type") or "Concept"),
                confidence=_coerce_confidence(e.get("confidence")),
                evidence=str(e.get("evidence") or evidence)[:500],
            )
        )
    relations: List[ExtractedRelation] = []
    for r in data.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        subj, pred, obj = r.get("subject"), r.get("predicate"), r.get("object")
        if not (subj and pred and obj):
            continue
        relations.append(
            ExtractedRelation(
                subject=_normalize_name(str(subj)),
                predicate=re.sub(r"\s+", "_", str(pred).strip().lower()),
                object=_normalize_name(str(obj)),
                confidence=_coerce_confidence(r.get("confidence")),
                evidence=str(r.get("evidence") or evidence)[:500],
            )
        )
    return Extraction(entities=entities, relations=relations)
# --------------------------------------------------------------------------- #
# Extraction memoization (POCKET-404b).
# --------------------------------------------------------------------------- #
# Extraction is the one genuinely expensive step in the graph branch. The engine
# already memoizes at the *file* level, but any edit re-extracts a file's whole
# chunk set. A per-(chunk-text, model, prompt-version) cache means only the chunks
# that actually changed — and only when the prompt changed — are re-sent to the
# model. The key folds in PROMPT_VERSION so a prompt edit transparently invalidates.
def extraction_to_dict(ext: Extraction) -> dict:
    """Serialize an :class:`Extraction` to a JSON-safe dict (for caching)."""
    return asdict(ext)


def extraction_from_dict(data: dict) -> Extraction:
    """Rebuild an :class:`Extraction` from :func:`extraction_to_dict` output.

    Inputs come only from our own cache (written by :func:`extraction_to_dict`),
    so a plain ``**`` splat is safe; a corrupt/legacy row raises and the caller
    in :class:`MemoizingExtractor` falls through to a fresh extraction.
    """
    return Extraction(
        entities=[ExtractedEntity(**e) for e in data.get("entities", [])],
        relations=[ExtractedRelation(**r) for r in data.get("relations", [])],
    )


def extraction_cache_key(text: str, model_id: str, prompt_version: str) -> str:
    """Deterministic cache key over (chunk text, model, prompt version)."""
    # NUL joins fields so no value can bleed into the next (lengths vary freely).
    payload = "\x00".join((prompt_version, model_id, text)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@runtime_checkable
class ExtractionStore(Protocol):
    """A tiny key→JSON store backing :class:`MemoizingExtractor`."""

    def get(self, key: str) -> Optional[str]:  # pragma: no cover - protocol
        ...

    def set(self, key: str, value: str) -> None:  # pragma: no cover - protocol
        ...


class InMemoryExtractionStore:
    """Process-lifetime cache — dedupes repeated chunks within a single run."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value


class SqliteExtractionStore:
    """Persistent extraction cache in the Pocket SQLite DB.

    Survives restarts so an unchanged chunk under an unchanged prompt is never
    re-sent to the model across runs. Keyed by a content hash, it can never
    return stale data: a changed chunk, model, or PROMPT_VERSION yields a new key.
    """

    _TABLE = "_pocket_extract_memo"

    def __init__(self, conn) -> None:
        self.conn = conn
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._TABLE} "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            f"SELECT value FROM {self._TABLE} WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self._TABLE} (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()


class MemoizingExtractor:
    """Wraps any :class:`ExtractionModel`, caching results by content/model/prompt.

    ``misses`` counts how many times the wrapped model was actually invoked, which
    is what tests assert on and what the run stats surface as real model calls.
    """

    def __init__(
        self,
        inner: ExtractionModel,
        store: Optional[ExtractionStore] = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.inner = inner
        self.store = store if store is not None else InMemoryExtractionStore()
        self.prompt_version = prompt_version
        self.model_id = str(
            getattr(inner, "model", None)
            or getattr(inner, "model_id", None)
            or getattr(inner, "name", "unknown")
        )
        self.name = f"memoized:{getattr(inner, 'name', 'extractor')}"
        self.misses = 0

    def extract(self, text: str) -> Extraction:
        key = extraction_cache_key(text, self.model_id, self.prompt_version)
        cached = self.store.get(key)
        if cached is not None:
            try:
                return extraction_from_dict(json.loads(cached))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass  # corrupt/legacy cache entry → fall through and re-extract
        self.misses += 1
        result = self.inner.extract(text)
        try:
            self.store.set(key, json.dumps(extraction_to_dict(result)))
        except Exception as exc:  # noqa: BLE001 - cache write must never break a run
            print(f"[pocketindex.extract] extraction cache write failed: {exc}")
        return result





# --------------------------------------------------------------------------- #
# Backend 2: Ollama (local HTTP daemon).
# --------------------------------------------------------------------------- #
class OllamaExtractor:
    """Extract via a local Ollama daemon (``/api/generate``).

    Uses only the stdlib so the base install stays dependency-light. Any network,
    HTTP, or JSON error degrades to an empty extraction (logged), never a crash.
    """

    name = "ollama"

    def __init__(self, model: str = "llama3", host: str | None = None, timeout: float = 120.0):
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.timeout = timeout

    def extract(self, text: str) -> Extraction:
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": _EXTRACTION_PROMPT + text,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return parse_extraction_json(body.get("response", ""), evidence=text[:500])
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[pocketindex.extract] Ollama extraction failed, skipping chunk: {exc}")
            return Extraction()


# --------------------------------------------------------------------------- #
# Backend 3: airLLM (local in-process, layer-sharded inference).
# --------------------------------------------------------------------------- #
class AirLLMExtractor:
    """Extract via a large model run locally with ``airllm``.

    airLLM streams a model layer-by-layer from disk so a 70B-class model fits on a
    single 4GB GPU without quantization accuracy loss — letting a privacy-conscious
    user run a capable extractor entirely on their own hardware (no hosted proxy).

    ``airllm`` (and its torch/transformers stack) is an **optional** dependency,
    imported lazily here so the base install carries no ML weight. The model is
    loaded once and reused across chunks.
    """

    name = "airllm"

    def __init__(
        self,
        model: str = "garage-bAInd/Platypus2-70B-instruct",
        max_new_tokens: int = 512,
        max_input_tokens: int = 2048,
        compression: str | None = None,
        device: str | None = None,
    ):
        self.model_id = model
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.compression = compression
        self.device = device
        self._model = None  # lazily constructed on first extract()

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from airllm import AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AirLLMExtractor requires the optional 'airllm' extra. "
                "Install it with: pip install 'genome-pocket[airllm]' "
                "(or: pip install airllm torch transformers)."
            ) from exc
        kwargs = {}
        if self.compression:
            kwargs["compression"] = self.compression
        if self.device:
            kwargs["device"] = self.device
        self._model = AutoModel.from_pretrained(self.model_id, **kwargs)

    def _generate(self, prompt: str) -> str:
        self._ensure_model()
        model = self._model
        input_tokens = model.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=self.max_input_tokens,
            padding=False,
        )
        input_ids = input_tokens["input_ids"]
        # Move to the model's device when one is configured / available.
        to = getattr(input_ids, "to", None)
        if to is not None and getattr(model, "device", None) is not None:
            input_ids = to(model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
        return model.tokenizer.decode(output.sequences[0])

    def extract(self, text: str) -> Extraction:
        try:
            raw = self._generate(_EXTRACTION_PROMPT + text)
            # The decoded sequence echoes the prompt; parse_extraction_json grabs
            # the JSON object regardless of surrounding text.
            return parse_extraction_json(raw, evidence=text[:500])
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash
            print(f"[pocketindex.extract] airLLM extraction failed, skipping chunk: {exc}")
            return Extraction()


# --------------------------------------------------------------------------- #
# Factory: pick a backend from config / env.
# --------------------------------------------------------------------------- #
def build_extractor(
    provider: str | None = None,
    model: str | None = None,
    *,
    memo: bool = True,
    store: Optional[ExtractionStore] = None,
) -> ExtractionModel:
    """Construct an :class:`ExtractionModel` from a provider name.

    ``provider`` defaults to ``$POCKET_LLM_PROVIDER`` or ``"deterministic"`` so the
    pipeline is fully functional and offline-testable with no configuration. The
    deterministic backend is also the safe fallback for an unknown provider.

    The LLM backends (``ollama`` / ``airllm``) are wrapped in a
    :class:`MemoizingExtractor` when ``memo`` is set (default), keyed on
    (chunk text, model, ``PROMPT_VERSION``) so unchanged chunks under an unchanged
    prompt are never re-sent to the model. Pass ``store`` to persist the cache
    (e.g. a :class:`SqliteExtractionStore`); otherwise it is process-lifetime.
    The deterministic backend is pure and cheap, so it is returned unwrapped.
    """
    provider = (provider or os.getenv("POCKET_LLM_PROVIDER") or "deterministic").lower()
    model = model or os.getenv("POCKET_LLM_MODEL")

    backend: ExtractionModel
    if provider == "ollama":
        backend = OllamaExtractor(model=model or "llama3")
    elif provider == "airllm":
        backend = AirLLMExtractor(model=model or "garage-bAInd/Platypus2-70B-instruct")
    else:
        if provider != "deterministic":
            print(
                f"[pocketindex.extract] unknown POCKET_LLM_PROVIDER='{provider}', "
                f"falling back to deterministic extractor."
            )
        return DeterministicExtractor()

    if memo:
        return MemoizingExtractor(backend, store=store)
    return backend
