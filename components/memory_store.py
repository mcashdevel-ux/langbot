"""Long-term semantic memory store (embeddings + ChromaDB).

Search is deliberately more than a raw k-NN lookup, because a bare nearest-neighbour
query returns the *closest* rows whether or not any of them is relevant:

* every hit carries a cosine similarity, and anything below ``memory.min_similarity``
  is dropped rather than handed to the model as if it were a fact;
* candidates are over-fetched, collapsed by normalized text, then selected with MMR, so
  ``n`` results are ``n`` *distinct* facts;
* a lexical leg (substring match on the normalized document) runs alongside the dense
  one and is fused by reciprocal rank, because sentence embeddings are weak on exactly
  what this agent stores most — paths, env-var names, error codes, command names;
* writes drop duplicates, so one repeated fact cannot occupy every result slot.

Canonical, importable home for the memory state that used to live as globals in
``langbot.py``. Both the main thread (``remember``/``recall``, ``/save``) and the
background memory worker read and write the same collection through here, so
writes are serialized behind one lock rather than relying on assumptions about
ChromaDB's internal concurrency guarantees.

Reads deliberately stay outside the lock: the worst case is a ``recall`` issued
mid-batch-write missing the newest facts, which is the accepted eventual
consistency of asynchronous distillation.

The embedding model and the Chroma collection are built on first use (the agent
warms them at startup via ``get_embeddings()``), so importing this module is
cheap and its state is injectable in tests.
"""

import logging
import math
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings

from . import console as ui
from .config import config
from .utils import suppress_native_output

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = config.get("paths.chroma_dir", "./memory/agent_memory_chroma",
                                env="AGENT_CHROMA_DIR")
COLLECTION_NAME = config.get("memory.collection_name", "agent_longterm_memory")
EMBEDDING_MODEL = config.get("memory.embedding_model",
                             "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = config.get("memory.embedding_device", "cpu")

# Similarity below which a hit is noise rather than a memory. all-MiniLM puts
# unrelated short sentences around 0.1-0.25, so 0.3 is a conservative floor.
MIN_SIMILARITY = config.get("memory.min_similarity", 0.3)
# Candidates fetched per requested result before dedup/MMR narrows them down.
RECALL_OVERFETCH = config.get("memory.recall_overfetch", 4)
# MMR relevance/diversity tradeoff: 1.0 is pure relevance, 0.0 pure diversity.
MMR_LAMBDA = config.get("memory.mmr_lambda", 0.7)
# A write is a duplicate when it is this close to an existing fact, shares this
# much of its vocabulary, and names exactly the same identifiers. Embeddings
# alone are not enough: "port 8080" and "port 8081" are near-identical vectors
# but different facts.
DEDUP_SIMILARITY = config.get("memory.dedup_similarity", 0.95)
DEDUP_TOKEN_OVERLAP = config.get("memory.dedup_token_overlap", 0.6)
LEXICAL_SEARCH = config.get("memory.lexical_search", True)
# Tags per fact, and whether untagged writes get deterministic fallback tags.
MAX_TAGS = config.get("memory.max_tags", 5)
AUTO_TAGS = config.get("memory.auto_tags", True)
# Reciprocal-rank-fusion constant; 60 is the value from the original RRF paper.
RRF_K = 60

_write_lock = threading.Lock()
_init_lock = threading.Lock()
_embeddings = None
_client = None
_collection = None


def get_embeddings(announce: bool = True):
    """Return the embedding model, loading it (quietly) on first use.

    The heavy transformers/tqdm progress output goes to the raw stderr fd, so it
    is muted at the fd level while the weights load. Pass ``announce=False``
    when loading off the main thread, so the progress notes cannot land in the
    middle of the REPL's prompt.
    """
    global _embeddings
    with _init_lock:
        if _embeddings is not None:
            return _embeddings
        if announce:
            ui.info("Loading embedding model...")
        try:
            import transformers  # noqa: WPS433 (optional, only to quiet it)

            transformers.logging.set_verbosity_error()
            transformers.logging.disable_progress_bar()
        except Exception:
            pass
        from langchain_huggingface import HuggingFaceEmbeddings

        with suppress_native_output():
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': EMBEDDING_DEVICE},
                encode_kwargs={'normalize_embeddings': True},
            )
        if announce:
            ui.success("Embedding model ready.")
        return _embeddings


def get_collection():
    """Return the persistent Chroma collection, creating it on first use."""
    global _client, _collection
    with _init_lock:
        if _collection is None:
            _client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            _collection = _client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return _collection


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------
# Text normalization — one canonical form used for dedup, the lexical index, and
# collapsing near-identical hits out of a result set.
# ------------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
# Tokens keep the punctuation that makes identifiers identifiers, so
# "~/code/myapp" and "OPENAI_API_KEY" survive as single searchable units.
_TOKEN_RE = re.compile(r"[\w~./#][\w./~:@+-]*")


def normalize(text: str) -> str:
    """Case- and whitespace-insensitive form of a fact."""
    return _WS_RE.sub(" ", (text or "").strip()).casefold()


def _tokens(text: str) -> "set[str]":
    return set(_TOKEN_RE.findall(normalize(text)))


def _overlap(a: str, b: str) -> float:
    """Jaccard overlap of two texts' tokens (0.0 when either is empty)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _identifiers(text: str) -> "set[str]":
    """Tokens that carry a specific value: numbers, paths, env vars, versions.

    These are what makes two otherwise-identical sentences different facts, and
    embeddings are almost blind to them.
    """
    return {
        t for t in _tokens(text)
        if any(c.isdigit() for c in t) or any(c in t for c in "./~:@_-")
    }


def _cosine(a, b) -> float:
    if a is None or b is None or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _vector(raw) -> "list[float] | None":
    """Chroma hands embeddings back as lists or numpy arrays; normalize to list."""
    if raw is None:
        return None
    try:
        return [float(x) for x in raw]
    except TypeError:
        return None


# ------------------------------------------------------------------------------
# Tags — stored two ways: comma-joined in metadata (Chroma metadata values must be
# scalars) for display, and as ``#tag`` tokens inside the document so the lexical
# leg searches them for free. They are never embedded, so they cannot distort
# semantic similarity, and they stay out of ``norm``, so they cannot block dedup.
# ------------------------------------------------------------------------------
_TAG_CLEAN_RE = re.compile(r"[^a-z0-9-]+")


def clean_tags(tags) -> "list[str]":
    """Canonical tag list: lowercase ``[a-z0-9-]``, deduped, capped at MAX_TAGS."""
    out: "list[str]" = []
    for raw in tags or []:
        if not isinstance(raw, str):
            continue
        tag = _TAG_CLEAN_RE.sub("-", raw.strip().casefold()).strip("-")
        if tag and tag not in out:
            out.append(tag)
    return out[:MAX_TAGS]


_AUTO_TAG_RULES = (
    ("preference", re.compile(r"\b(prefers?|likes?|favou?rite|wants?)\b", re.I)),
    ("credentials", re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"
                               r"|\b(api.?key|token|password|secret|credential)\b", re.I)),
    ("web", re.compile(r"https?://", re.I)),
    ("filesystem", re.compile(r"(?:^|[\s'\"(=])(?:~|\.{1,2})?/[\w.@-]+(?:/[\w.@-]+)+")),
)


def auto_tags(text: str) -> "list[str]":
    """Deterministic coarse tags for what the fact visibly contains.

    A fallback, not a classifier: it guarantees every fact carries at least the
    obvious categories even when no tags were supplied.
    """
    return [tag for tag, pattern in _AUTO_TAG_RULES if pattern.search(text or "")]


def _split_tags(joined: str) -> "list[str]":
    return [t for t in (joined or "").split(",") if t]


def _document(norm: str, tags: "list[str]") -> str:
    return norm + "".join(f" #{t}" for t in tags)


@dataclass
class Memory:
    """One retrieved fact, with everything needed to judge it."""

    id: str
    text: str
    score: float                    # cosine similarity to the query, 0.0-1.0
    timestamp: str = ""
    source: str = ""                # "manual" | "distilled" | "supabase" | ""
    tags: "list[str]" = field(default_factory=list)
    matched: "list[str]" = field(default_factory=list)   # "dense" and/or "lexical"


# ------------------------------------------------------------------------------
# Writes
# ------------------------------------------------------------------------------
# Confidence defaults. Manual facts (/save, remember tool) are 1.0; distilled
# facts get a lower default because the distiller operates on truncated context.
DEFAULT_MANUAL_CONFIDENCE = config.get("memory.default_manual_confidence", 1.0)
DEFAULT_DISTILLED_CONFIDENCE = config.get("memory.default_distilled_confidence", 0.7)

# Pruning: facts older than this many days with confidence below the threshold
# are eligible for automatic removal on startup (see components/housekeeping.py).
PRUNE_AGE_DAYS = config.get("memory.prune_age_days", 90)
PRUNE_CONFIDENCE_THRESHOLD = config.get("memory.prune_confidence_threshold", 0.8)


def _metadata(text: str, timestamp: str, source: str, tags: "list[str]",
              confidence: float = DEFAULT_DISTILLED_CONFIDENCE) -> dict:
    return {
        "text": text,
        "norm": normalize(text),
        "timestamp": timestamp,
        "source": source,
        "tags": ",".join(tags),
        "confidence": str(round(confidence, 3)),
    }


def _effective_tags(text: str, tags) -> "list[str]":
    supplied = clean_tags(tags)
    if AUTO_TAGS:
        return clean_tags(supplied + auto_tags(text))
    return supplied


def _merge_tags(collection, mem_id: str, tags: "list[str]") -> None:
    """Fold new tags into an existing row (a duplicate write may know more)."""
    if not tags:
        return
    row = collection.get(ids=[mem_id], include=["metadatas", "embeddings"])
    metas = row.get("metadatas") or []
    if not metas or metas[0] is None:
        return
    meta = dict(metas[0])
    merged = clean_tags(_split_tags(meta.get("tags", "")) + tags)
    if merged == _split_tags(meta.get("tags", "")):
        return
    meta["tags"] = ",".join(merged)
    norm = meta.get("norm") or normalize(meta.get("text", ""))
    embeds = row.get("embeddings")
    vector = _vector(embeds[0]) if embeds is not None and len(embeds) else None
    # The stored vector is passed back unchanged: without it, Chroma re-embeds
    # the (tag-carrying) document with its own default model.
    collection.update(
        ids=[mem_id],
        embeddings=[vector] if vector is not None else None,
        metadatas=[meta],
        documents=[_document(norm, merged)],
    )


def _duplicate_of(collection, text: str, vector) -> "str | None":
    """Id of an existing fact this text duplicates, else None.

    Exact matches are caught by the ``norm`` metadata field. A near-duplicate must
    clear all three of similarity, vocabulary overlap, and identical identifiers,
    so paraphrases collapse while facts that merely look alike ("port 8080" vs
    "port 8081") stay separate. Rows written before ``norm`` existed are only
    reachable via the near-duplicate leg.
    """
    norm = normalize(text)
    try:
        exact = collection.get(where={"norm": norm}, limit=1)
    except Exception:  # noqa: BLE001 - an unsupported filter must not block writes
        logger.debug("memory_store: exact-duplicate lookup failed", exc_info=True)
        exact = None
    if exact and exact.get("ids"):
        return exact["ids"][0]

    if vector is None or DEDUP_SIMILARITY > 1.0 or not collection.count():
        return None
    nearest = collection.query(
        query_embeddings=[vector],
        n_results=1,
        include=["metadatas", "distances"],
    )
    ids = (nearest.get("ids") or [[]])[0]
    metas = (nearest.get("metadatas") or [[]])[0]
    dists = (nearest.get("distances") or [[]])[0]
    if not ids or not metas:
        return None
    similarity = 1.0 - float(dists[0]) if dists else 0.0
    existing = metas[0].get("text", "")
    if (similarity >= DEDUP_SIMILARITY
            and _overlap(norm, existing) >= DEDUP_TOKEN_OVERLAP
            and _identifiers(norm) == _identifiers(existing)):
        return ids[0]
    return None


def store_memory(text: str, source: str = "manual", tags: "list[str] | None" = None,
                 confidence: "float | None" = None) -> str:
    """Store one fact and return its id — the *existing* id if it is a duplicate.

    Callers get an id either way, so "already known" is not an error; only the
    store stays free of the repeated facts that would otherwise crowd out every
    recall result.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("cannot store an empty memory")
    tag_list = _effective_tags(text, tags)
    if confidence is None:
        confidence = (DEFAULT_MANUAL_CONFIDENCE if source in ("manual", "supabase")
                      else DEFAULT_DISTILLED_CONFIDENCE)
    vector = get_embeddings().embed_query(text)
    collection = get_collection()
    mem_id = str(uuid.uuid4())
    with _write_lock:
        duplicate = _duplicate_of(collection, text, vector)
        if duplicate is not None:
            logger.debug("memory_store: skipped duplicate of %s: %r", duplicate, text[:80])
            _merge_tags(collection, duplicate, tag_list)
            return duplicate
        collection.add(
            ids=[mem_id],
            embeddings=[vector],
            documents=[_document(normalize(text), tag_list)],
            metadatas=[_metadata(text, _now_iso(), source, tag_list, confidence)],
        )
    return mem_id


def store_memories_batch(
    texts: list[str],
    timestamps: "list[str] | None" = None,
    source: str = "distilled",
    tags_list: "list[list[str]] | None" = None,
    confidence: "float | None" = None,
) -> list[str]:
    """Store many facts with a single ``embed_documents`` call.

    ``timestamps`` preserves original times for facts pulled from elsewhere
    (e.g. Supabase); missing or blank entries fall back to now. ``tags_list``
    supplies per-fact tags, aligned with ``texts``. The returned ids line up
    with ``texts``; a duplicate yields the id of the fact it duplicates, and
    blank entries yield ``""``.
    """
    if not texts:
        return []
    vectors = get_embeddings().embed_documents(texts)
    now = _now_iso()
    stamps = list(timestamps or [])
    stamps += [now] * (len(texts) - len(stamps))
    if confidence is None:
        confidence = (DEFAULT_MANUAL_CONFIDENCE if source in ("manual", "supabase")
                      else DEFAULT_DISTILLED_CONFIDENCE)
    all_tags = list(tags_list or [])
    all_tags += [[] for _ in range(len(texts) - len(all_tags))]
    collection = get_collection()
    ids: "list[str]" = []
    pending_ids, pending_vectors, pending_docs, pending_metas = [], [], [], []
    seen: "dict[str, str]" = {}
    with _write_lock:
        for text, vector, stamp, tags in zip(texts, vectors, stamps, all_tags):
            text = (text or "").strip()
            if not text:
                ids.append("")
                continue
            norm = normalize(text)
            tag_list = _effective_tags(text, tags)
            if norm in seen:                      # duplicate within this batch
                ids.append(seen[norm])
                continue
            duplicate = _duplicate_of(collection, text, vector)
            if duplicate is not None:
                seen[norm] = duplicate
                ids.append(duplicate)
                _merge_tags(collection, duplicate, tag_list)
                continue
            mem_id = str(uuid.uuid4())
            seen[norm] = mem_id
            ids.append(mem_id)
            pending_ids.append(mem_id)
            pending_vectors.append(vector)
            pending_docs.append(_document(norm, tag_list))
            pending_metas.append(_metadata(text, stamp or now, source, tag_list, confidence))
        if pending_ids:
            collection.add(
                ids=pending_ids,
                embeddings=pending_vectors,
                documents=pending_docs,
                metadatas=pending_metas,
            )
    skipped = len(ids) - len(pending_ids)
    if skipped:
        logger.debug("memory_store: stored %d fact(s), skipped %d duplicate/blank",
                     len(pending_ids), skipped)
    return ids


# ------------------------------------------------------------------------------
# Search
# ------------------------------------------------------------------------------
def _query_tokens(query: str) -> "list[str]":
    """Query tokens worth a literal lookup: identifier-ish or simply long.

    Short, common words are exactly where substring matching produces junk, and
    exactly where the dense leg already works well.
    """
    scored = []
    for token in _TOKEN_RE.findall(normalize(query)):
        identifier = (token.startswith("#")            # explicit tag lookup
                      or any(c in token for c in "./~:@_-")
                      or any(c.isdigit() for c in token))
        if identifier or len(token) >= 5:
            scored.append(token)
    # Longest first: the most specific token is the most selective filter.
    return sorted(dict.fromkeys(scored), key=len, reverse=True)[:3]


def _candidate(mem_id, meta, vector, score) -> "dict":
    return {
        "id": mem_id,
        "text": (meta or {}).get("text", ""),
        "timestamp": (meta or {}).get("timestamp", ""),
        "source": (meta or {}).get("source", ""),
        "tags": _split_tags((meta or {}).get("tags", "")),
        "vector": vector,
        "score": score,
        "rrf": 0.0,
        "matched": [],
    }


def _dense_candidates(collection, query_vec, k: int, total: int) -> "list[dict]":
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k, total),
        include=["metadatas", "distances", "embeddings"],
    )
    ids = (results.get("ids") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    embeds = (results.get("embeddings") if results.get("embeddings") is not None else [[]])[0]
    out = []
    for i, mem_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        distance = float(dists[i]) if i < len(dists) else 1.0
        vector = _vector(embeds[i]) if embeds is not None and i < len(embeds) else None
        out.append(_candidate(mem_id, meta, vector, max(0.0, 1.0 - distance)))
    return out


def _lexical_candidates(collection, query: str, query_vec, k: int) -> "list[dict]":
    """Rows literally containing a distinctive query token, ranked by similarity.

    Dense retrieval routinely misses these: a MiniLM embedding of "myapp path"
    is not close to "project located at ~/code/myapp", but the substring is
    right there. Documents are stored normalized, so the match is
    case-insensitive.
    """
    found: "dict[str, dict]" = {}
    for token in _query_tokens(query):
        try:
            rows = collection.get(
                where_document={"$contains": token},
                limit=k,
                include=["metadatas", "embeddings"],
            )
        except Exception:  # noqa: BLE001 - lexical search is an optimization, not a contract
            logger.debug("memory_store: lexical lookup failed for %r", token, exc_info=True)
            return []
        ids = rows.get("ids") or []
        metas = rows.get("metadatas") or []
        embeds = rows.get("embeddings")
        for i, mem_id in enumerate(ids):
            if mem_id in found:
                continue
            meta = metas[i] if i < len(metas) else {}
            vector = _vector(embeds[i]) if embeds is not None and i < len(embeds) else None
            found[mem_id] = _candidate(mem_id, meta, vector, _cosine(query_vec, vector))
    return sorted(found.values(), key=lambda c: c["score"], reverse=True)[:k]


def _fuse(legs: "dict[str, list[dict]]") -> "list[dict]":
    """Reciprocal-rank fusion of the retrieval legs, best first."""
    merged: "dict[str, dict]" = {}
    for leg, candidates in legs.items():
        for rank, candidate in enumerate(candidates):
            existing = merged.setdefault(candidate["id"], candidate)
            if existing is not candidate:
                # Keep whichever leg computed a real similarity, and its vector.
                existing["score"] = max(existing["score"], candidate["score"])
                existing["vector"] = existing["vector"] or candidate["vector"]
            existing["rrf"] += 1.0 / (RRF_K + rank + 1)
            existing["matched"].append(leg)
    return sorted(merged.values(), key=lambda c: (c["rrf"], c["score"]), reverse=True)


def _mmr(candidates: "list[dict]", n: int) -> "list[dict]":
    """Pick ``n`` candidates trading relevance off against redundancy.

    Without this, three phrasings of one fact can fill every slot — which is
    precisely what an unfiltered store used to return.
    """
    selected: "list[dict]" = []
    pool = list(candidates)
    while pool and len(selected) < n:
        best, best_value = None, None
        for candidate in pool:
            penalty = max(
                (_cosine(candidate["vector"], chosen["vector"]) for chosen in selected),
                default=0.0,
            )
            value = MMR_LAMBDA * candidate["score"] - (1.0 - MMR_LAMBDA) * penalty
            if best_value is None or value > best_value:
                best, best_value = candidate, value
        selected.append(best)
        pool.remove(best)
    return selected


def search_memories(
    query: str,
    n: int = 3,
    min_similarity: "float | None" = None,
    lexical: "bool | None" = None,
) -> "list[Memory]":
    """Search long-term memory, returning scored, distinct, above-threshold hits.

    Dense k-NN and a lexical leg are fused by reciprocal rank; results below
    ``min_similarity`` are dropped unless a query token literally appears in
    them, near-identical texts are collapsed, and MMR picks the final ``n``.
    An empty list means "nothing relevant", which is a useful answer — far more
    useful than the nearest three rows whatever their distance.
    """
    if not (query or "").strip() or n <= 0:
        return []
    collection = get_collection()
    total = collection.count()
    if total == 0:
        # Chroma rejects n_results < 1, so guard the empty-store case explicitly.
        return []
    floor = MIN_SIMILARITY if min_similarity is None else min_similarity
    use_lexical = LEXICAL_SEARCH if lexical is None else lexical
    query_vec = get_embeddings().embed_query(query)
    k = max(n * max(int(RECALL_OVERFETCH), 1), n)

    legs = {"dense": _dense_candidates(collection, query_vec, k, total)}
    if use_lexical:
        legs["lexical"] = _lexical_candidates(collection, query, query_vec, k)

    ranked = _fuse(legs)
    kept, seen = [], set()
    for candidate in ranked:
        if not candidate["text"]:
            continue
        if candidate["score"] < floor and "lexical" not in candidate["matched"]:
            continue
        norm = normalize(candidate["text"])
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(candidate)

    return [
        Memory(
            id=c["id"],
            text=c["text"],
            score=round(c["score"], 4),
            timestamp=c["timestamp"],
            source=c["source"],
            tags=c["tags"],
            matched=sorted(set(c["matched"])),
        )
        for c in _mmr(kept, n)
    ]


def recall_memories(query: str, n: int = 3) -> list[str]:
    """``search_memories`` reduced to fact text, for callers that want strings."""
    return [m.text for m in search_memories(query, n=n)]


def count() -> int:
    return get_collection().count()
