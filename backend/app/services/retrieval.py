"""
Document retrieval pipeline for RAGLab.

Provides hybrid retrieval (dense + BM25), cross-encoder reranking,
query expansion, numeric grounding validation, and confidence scoring.

Key design decisions
--------------------
- **Hybrid search**: Dense embeddings (BAAI/bge-small-en-v1.5) are combined
  with BM25 keyword scores via Reciprocal Rank Fusion (RRF).
- **Two-pass reranking**: A cross-encoder (BAAI/bge-reranker-base) re-ranks
  the top candidates, first on a wider pool then on a narrower set.
- **Adaptive top-k**: Numeric or complex questions automatically increase
  the number of retrieved chunks.
- **Query expansion**: The LLM generates up to 3 reformulations of the
  question to improve recall.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from sentence_transformers import CrossEncoder

logger = logging.getLogger("raglab")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = PROJECT_ROOT / "backend" / "data" / "samples"
UPLOAD_DIR = Path("/tmp/rag_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
INDEX_STORE: dict[str, VectorStoreIndex] = {}
SESSION_META: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Lazy-loaded cross-encoder
# ---------------------------------------------------------------------------
_RERANKER: CrossEncoder | None = None


def _load_env_file(path: Path) -> None:
    """Load environment variables from a ``.env`` file."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

Settings.llm = Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


# ===================================================================
# Indexing
# ===================================================================
def _build_index(
    documents: list[Document],
    session_id: str,
) -> VectorStoreIndex:
    """Build an in-memory vector index from the given documents.

    Parameters
    ----------
    documents:
        A list of llama-index ``Document`` objects.
    session_id:
        Unique identifier for the new session.

    Returns
    -------
    VectorStoreIndex
        The built index, also stored in ``INDEX_STORE``.
    """
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(
        documents
    )
    index = VectorStoreIndex(nodes, show_progress=False)
    INDEX_STORE[session_id] = index
    return index


def _get_index(session_id: str) -> VectorStoreIndex:
    """Retrieve the index for the given session.

    Raises
    ------
    HTTPException 404
        If the session does not exist.
    """
    if session_id not in INDEX_STORE:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please index documents first.",
        )
    return INDEX_STORE[session_id]


# ===================================================================
# Utilities
# ===================================================================
def _tokenize(text: str) -> list[str]:
    """Simple whitespace-aware tokenizer (lowercased)."""
    return re.findall(r"\b\w+\b", (text or "").lower())


def _is_numeric_question(q: str) -> bool:
    """Check whether the question likely expects a numeric answer."""
    patterns = r"\b(cost|price|amount|total|investment|profit|revenue|emi|rs|₹|percent|%)\b"
    return bool(re.search(patterns, q.lower()))


def _adaptive_top_k(question: str, requested_top_k: int) -> int:
    """Adjust the number of chunks to retrieve based on question type.

    - Numeric questions → tap more chunks to improve grounding.
    - Long / explanatory questions → expand scope slightly.
    """
    q = question.lower()
    if _is_numeric_question(q):
        return max(3, min(6, requested_top_k))
    if len(q.split()) >= 12 or re.search(
        r"\bcompare|difference|summarize|overview|explain\b", q
    ):
        return max(5, min(8, requested_top_k + 2))
    return max(3, min(8, requested_top_k))


def new_session_id() -> str:
    """Generate a fresh, unique session identifier."""
    return str(uuid.uuid4())


# ===================================================================
# LLM helpers
# ===================================================================
async def _llm_complete(prompt: str, timeout_seconds: float = 35.0) -> str:
    """Call the configured LLM with a prompt, with retry logic.

    Raises
    ------
    RuntimeError
        After exhausting retries or on authentication failures.
    """
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(Settings.llm.complete, prompt),
                timeout=timeout_seconds,
            )
            return str(result).strip()
        except Exception as exc:  # noqa: BLE001
            err = str(exc).lower()
            if "invalid_api_key" in err or "401" in err or "unauthorized" in err:
                raise RuntimeError(
                    "Groq authentication failed. "
                    "Check GROQ_API_KEY and ensure it belongs to the active project."
                ) from exc
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(0.4)
    raise RuntimeError(f"LLM call failed after retries: {last_error}")


async def _expand_queries(question: str) -> list[str]:
    """Generate up to 3 query rewrites to improve retrieval recall."""
    prompt = (
        "Generate up to 3 short search rewrites for better document retrieval.\n"
        "Return strict JSON with key rewrites (array of strings).\n"
        f"Question: {question}\nJSON:"
    )
    try:
        raw = await _llm_complete(prompt, timeout_seconds=15.0)
        data = json.loads(raw)
        rewrites: list[str] = [question]
        for r in data.get("rewrites", []):
            r = str(r).strip()
            if r and r.lower() not in {x.lower() for x in rewrites}:
                rewrites.append(r)
        return rewrites[:4]
    except Exception:
        return [question, f"{question} exact value", f"{question} from document"]


# ===================================================================
# BM25
# ===================================================================
def _bm25_retrieve(
    index: VectorStoreIndex,
    query: str,
    top_k: int,
) -> list:
    """Perform BM25 (Okapi) retrieval on the in-memory document store."""
    all_nodes = list(index.docstore.docs.values())
    if not all_nodes:
        return []

    docs_tokens = [_tokenize(n.text) for n in all_nodes]
    doc_freq: dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            doc_freq[t] = doc_freq.get(t, 0) + 1

    n_docs = len(all_nodes)
    avgdl = sum(len(t) for t in docs_tokens) / max(n_docs, 1)
    q_terms = _tokenize(query)
    if not q_terms:
        return []

    k1, b = 1.5, 0.75
    scored: list[tuple[float, Any]] = []
    for node, toks in zip(all_nodes, docs_tokens):
        dl = len(toks) or 1
        term_counts: dict[str, int] = {}
        for t in toks:
            term_counts[t] = term_counts.get(t, 0) + 1

        bm25_score = 0.0
        for term in q_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            numer = tf * (k1 + 1)
            denom = tf + k1 * (1 - b + b * (dl / max(avgdl, 1e-9)))
            bm25_score += idf * (numer / denom)
        if bm25_score > 0:
            scored.append((bm25_score, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:top_k]]


# ===================================================================
# Hybrid retrieval (dense + BM25 via RRF)
# ===================================================================
def _hybrid_retrieve(
    index: VectorStoreIndex,
    query: str,
    dense_k: int,
    alpha_dense: float,
    alpha_bm25: float,
) -> list:
    """Hybrid retrieval combining dense embedding search and BM25.

    Scores are fused via Reciprocal Rank Fusion (RRF) with configurable
    per-strategy weights.
    """
    dense_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=dense_k,
    )
    dense_nodes = dense_retriever.retrieve(query)
    bm25_nodes = _bm25_retrieve(index=index, query=query, top_k=dense_k)

    rrf_k = 60
    fused_scores: dict[str, float] = {}
    node_map: dict[str, Any] = {}

    for rank, node_with_score in enumerate(dense_nodes, start=1):
        node = node_with_score.node
        nid = node.node_id
        fused_scores[nid] = (
            fused_scores.get(nid, 0.0)
            + alpha_dense * (1.0 / (rrf_k + rank))
        )
        node_map[nid] = node

    for rank, node in enumerate(bm25_nodes, start=1):
        nid = node.node_id
        fused_scores[nid] = (
            fused_scores.get(nid, 0.0)
            + alpha_bm25 * (1.0 / (rrf_k + rank))
        )
        node_map[nid] = node

    fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [node_map[nid] for nid, _ in fused[: dense_k * 2]]


# ===================================================================
# Cross-encoder reranking
# ===================================================================
def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker model."""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder("BAAI/bge-reranker-base")
    return _RERANKER


def _rerank_nodes(
    question: str,
    nodes: list,
    top_k: int,
) -> list[tuple[float, Any]]:
    """Re-rank nodes with a cross-encoder and return the top-k.

    Parameters
    ----------
    question:
        The original user question.
    nodes:
        Candidate nodes to re-rank.
    top_k:
        Number of highest-scoring nodes to return.

    Returns
    -------
    list[tuple[float, Any]]
        Sorted list of ``(score, node)`` pairs.
    """
    if not nodes:
        return []
    reranker = _get_reranker()
    seen: set[str] = set()
    unique_nodes: list = []
    for n in nodes:
        if n.node_id not in seen:
            seen.add(n.node_id)
            unique_nodes.append(n)

    pairs = [(question, (n.text or "")[:3000]) for n in unique_nodes]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(scores, unique_nodes),
        key=lambda x: float(x[0]),
        reverse=True,
    )
    return [(float(s), n) for s, n in ranked[:top_k]]


# ===================================================================
# Numeric grounding
# ===================================================================
def _extract_numeric_tokens(text: str) -> set[str]:
    """Extract numeric tokens (including INR-format amounts) from text."""
    matches = re.findall(
        r"(?:rs\.?\s*)?\d[\d,]*(?:\.\d+)?",
        (text or "").lower(),
    )
    return {m.strip() for m in matches}


def _validate_numeric_grounding(
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    """Check what fraction of numeric values in the answer appear in sources.

    Returns
    -------
    dict
        A dictionary with keys:
        - ``has_numbers`` — whether the answer contains any numeric tokens.
        - ``grounded_ratio`` — fraction of answer numbers found in context.
        - ``unsupported_numbers`` — list of ungrounded numeric tokens.
    """
    answer_nums = _extract_numeric_tokens(answer)
    if not answer_nums:
        return {
            "has_numbers": False,
            "grounded_ratio": 1.0,
            "unsupported_numbers": [],
        }

    context_nums: set[str] = set()
    for c in contexts:
        context_nums.update(_extract_numeric_tokens(c))

    unsupported = sorted(n for n in answer_nums if n not in context_nums)
    grounded = len(answer_nums) - len(unsupported)
    ratio = grounded / max(len(answer_nums), 1)
    return {
        "has_numbers": True,
        "grounded_ratio": round(ratio, 3),
        "unsupported_numbers": unsupported,
    }


# ===================================================================
# Confidence scoring
# ===================================================================
def _build_confidence_score(
    rerank_scores: list[float],
    grounding: dict[str, Any],
) -> float:
    """Combine reranker scores and numeric grounding into a single confidence.

    The formula is:
        confidence = 0.7 × rerank_component + 0.3 × grounding_component

    Both components are normalised to [0, 1].
    """
    if not rerank_scores:
        return 0.0

    avg_rerank = sum(rerank_scores) / len(rerank_scores)
    rerank_component = max(0.0, min(1.0, (avg_rerank + 2.0) / 8.0))
    numeric_component = grounding.get("grounded_ratio", 1.0)

    confidence = 0.7 * rerank_component + 0.3 * numeric_component
    return round(max(0.0, min(1.0, confidence)), 3)