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

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable


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
# Strict-JSON parsing shared by the LLM backends.
# --------------------------------------------------------------------------- #
_EXTRACTION_PROMPT = """You are a knowledge-graph extractor. Read the TEXT and \
return STRICT JSON (no prose, no code fence) of the form:
{"entities":[{"name":"...","type":"...","confidence":0.0-1.0}],
 "relations":[{"subject":"...","predicate":"...","object":"...","confidence":0.0-1.0}]}
Propose entity types and predicates freely (schema-agnostic). Use lower_snake_case \
predicates. Only include facts the TEXT supports.
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
    if text.startswith(""):
        text = re.sub(r"^[a-zA-Z]*\n?|\n?$", "", text).strip()
    # Grab the outermost JSON object if the model wrapped it in prose.
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
        max_input_tokens: int = 1024,
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
    provider: str | None = None, model: str | None = None
) -> ExtractionModel:
    """Construct an :class:`ExtractionModel` from a provider name.

    ``provider`` defaults to ``$POCKET_LLM_PROVIDER`` or ``"deterministic"`` so the
    pipeline is fully functional and offline-testable with no configuration. The
    deterministic backend is also the safe fallback for an unknown provider.
    """
    provider = (provider or os.getenv("POCKET_LLM_PROVIDER") or "deterministic").lower()
    model = model or os.getenv("POCKET_LLM_MODEL")

    if provider == "ollama":
        return OllamaExtractor(model=model or "llama3")
    if provider == "airllm":
        return AirLLMExtractor(model=model or "garage-bAInd/Platypus2-70B-instruct")
    if provider != "deterministic":
        print(
            f"[pocketindex.extract] unknown POCKET_LLM_PROVIDER='{provider}', "
            f"falling back to deterministic extractor."
        )
    return DeterministicExtractor()
