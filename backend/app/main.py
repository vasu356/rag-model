"""
RAGLab - FastAPI Backend
~~~~~~~~~~~~~~~~~~~~~~~~~

Provides endpoints for document indexing (sample & upload), hybrid retrieval
with cross-encoder reranking, and SSE-streamed answer generation.

Endpoints
---------
- GET  /health           — Backend health & LLM status
- GET  /samples          — List bundled sample documents
- POST /index/samples    — Index selected sample documents
- POST /index/upload     — Upload & index files (TXT, MD, PDF)
- POST /query/stream     — Stream a grounded answer via SSE
- DELETE /session/{id}   — Clear in-memory session
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.core import Document

from .schemas import HealthResponse, QueryRequest, SessionResponse
from .services.answering import (
    _clean_answer_text,
    _expand_answer_if_needed,
    _normalize_answer_length,
    _self_check_answer,
    _structured_answer,
)
from .services.retrieval import (
    GROQ_API_KEY,
    GROQ_MODEL,
    INDEX_STORE,
    SESSION_META,
    SAMPLES_DIR,
    UPLOAD_DIR,
    _adaptive_top_k,
    _build_confidence_score,
    _build_index,
    _expand_queries,
    _get_index,
    _hybrid_retrieve,
    _is_numeric_question,
    _rerank_nodes,
    _validate_numeric_grounding,
    new_session_id,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("raglab")

# Suppress noisy multipart debug logs.
logging.getLogger("python_multipart").setLevel(logging.ERROR)
logging.getLogger("python_multipart.multipart").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAGLab API",
    description="Full-stack Retrieval-Augmented Generation demo with hybrid search, "
    "cross-encoder reranking, and SSE streaming.",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:4173",  # Vite preview
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Return backend health status and LLM configuration."""
    return {
        "status": "ok",
        "version": "2.1.0",
        "llm_model": GROQ_MODEL,
        "groq_key_present": bool(GROQ_API_KEY),
    }


# ---------------------------------------------------------------------------
# Sample listing
# ---------------------------------------------------------------------------
@app.get("/samples")
def list_samples() -> dict[str, list[dict[str, Any]]]:
    """List bundled sample documents available for indexing."""
    samples: list[dict[str, Any]] = []
    if SAMPLES_DIR.exists():
        for f in sorted(SAMPLES_DIR.iterdir()):
            if f.suffix in {".txt", ".pdf", ".md"}:
                samples.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                })
    return {"samples": samples}


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
@app.post("/index/samples", response_model=SessionResponse)
def index_samples(names: list[str] | None = None) -> SessionResponse:
    """Index one or more bundled sample documents.

    If *names* is ``None`` or omitted, all available samples are indexed.
    """
    if not SAMPLES_DIR.exists() or not any(SAMPLES_DIR.iterdir()):
        raise HTTPException(status_code=404, detail="No sample documents found.")

    # Determine which files to index.
    if names:
        paths = [SAMPLES_DIR / n for n in names if (SAMPLES_DIR / n).exists()]
    else:
        paths = list(SAMPLES_DIR.glob("*.txt")) + list(SAMPLES_DIR.glob("*.md"))

    if not paths:
        raise HTTPException(
            status_code=400,
            detail="No matching sample files found.",
        )

    docs = [
        Document(
            text=p.read_text(encoding="utf-8", errors="ignore"),
            metadata={"source": p.name, "type": "sample"},
        )
        for p in paths
    ]

    session_id = new_session_id()
    index = _build_index(docs, session_id)
    SESSION_META[session_id] = {
        "doc_names": [p.name for p in paths],
        "type": "sample",
    }

    return SessionResponse(
        session_id=session_id,
        doc_count=len(docs),
        chunk_count=len(index.docstore.docs),
        doc_names=[p.name for p in paths],
    )


@app.post("/index/upload", response_model=SessionResponse)
async def index_upload(files: list[UploadFile] = File(...)) -> SessionResponse:
    """Upload and index TXT, MD, or PDF files."""
    allowed = {".txt", ".md", ".pdf"}
    docs: list[Document] = []
    names: list[str] = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}",
            )

        dest = UPLOAD_DIR / file.filename
        content = await file.read()
        dest.write_bytes(content)

        if suffix == ".pdf":
            from llama_index.readers.file import PDFReader

            loaded = PDFReader().load_data(file=dest)
            for d in loaded:
                d.metadata["source"] = file.filename
                d.metadata["type"] = "upload"
            docs.extend(loaded)
        else:
            docs.append(
                Document(
                    text=content.decode("utf-8", errors="ignore"),
                    metadata={"source": file.filename, "type": "upload"},
                )
            )
        names.append(file.filename)

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No content extracted from uploaded files.",
        )

    session_id = new_session_id()
    index = _build_index(docs, session_id)
    SESSION_META[session_id] = {
        "doc_names": names,
        "type": "upload",
    }

    return SessionResponse(
        session_id=session_id,
        doc_count=len(files),
        chunk_count=len(index.docstore.docs),
        doc_names=names,
    )


# ---------------------------------------------------------------------------
# Query / streaming
# ---------------------------------------------------------------------------
@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """Stream a grounded answer as Server-Sent Events.

    Returns a sequence of SSE events:
        - ``type: phase``        — lifecycle phase updates
        - ``type: sources``      — retrieved & reranked source chunks
        - ``type: quality``      — confidence & grounding metadata
        - ``type: token``        — streamed answer tokens
        - ``type: structured``   — structured answer metadata
        - ``type: metrics``      — latency
        - ``type: done``         — stream complete
        - ``type: error``        — error occurred
    """
    # --- Validation --------------------------------------------------------
    index = _get_index(req.session_id)

    if req.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
    if req.alpha_dense < 0 or req.alpha_bm25 < 0:
        raise HTTPException(status_code=400, detail="alpha values cannot be negative.")
    if not (0 <= req.confidence_threshold <= 1):
        raise HTTPException(
            status_code=400,
            detail="confidence_threshold must be between 0 and 1.",
        )
    if not (0 <= req.numeric_grounding_threshold <= 1):
        raise HTTPException(
            status_code=400,
            detail="numeric_grounding_threshold must be between 0 and 1.",
        )

    answer_length = _normalize_answer_length(req.answer_length)
    effective_conf_threshold = min(req.confidence_threshold, 0.30)
    total_alpha = req.alpha_dense + req.alpha_bm25

    if total_alpha == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one alpha must be > 0.",
        )

    dense_weight = req.alpha_dense / total_alpha
    bm25_weight = req.alpha_bm25 / total_alpha

    # --- SSE generator -----------------------------------------------------
    async def event_generator() -> AsyncGenerator[str, None]:
        started = time.time()
        try:
            # 1. Phase: retrieval
            yield _sse("phase", "retrieval_started")

            effective_top_k = _adaptive_top_k(req.question, req.top_k)
            dense_k = max(effective_top_k * 3, 8)

            # 2. Expand queries & retrieve candidates
            rewrites = await _expand_queries(req.question)
            all_candidates: list[Any] = []
            for q in rewrites:
                all_candidates.extend(
                    _hybrid_retrieve(
                        index=index,
                        query=q,
                        dense_k=dense_k,
                        alpha_dense=dense_weight,
                        alpha_bm25=bm25_weight,
                    )
                )

            # 3. Two-pass reranking
            ranked_first = _rerank_nodes(
                question=req.question,
                nodes=all_candidates,
                top_k=max(effective_top_k * 2, 8),
            )
            second_pass_nodes = [n for _, n in ranked_first]
            ranked_final = _rerank_nodes(
                question=req.question,
                nodes=second_pass_nodes,
                top_k=effective_top_k,
            )

            # 4. Build source list & context blocks
            sources: list[dict[str, Any]] = []
            context_blocks: list[str] = []
            rerank_vals: list[float] = []

            for score, node in ranked_final:
                rerank_vals.append(score)
                sources.append({
                    "source": node.metadata.get("source", "unknown"),
                    "score": round(score, 3),
                    "snippet": (node.text or "")[:220].replace("\n", " "),
                })
                context_blocks.append(
                    f"[Source: {node.metadata.get('source', 'unknown')}]\n{node.text}"
                )

            yield _sse("sources", sources)
            yield _sse("phase", "sources_ready")

            # 5. Generate answer
            if not context_blocks:
                answer = (
                    "I could not find strong supporting evidence "
                    "in the indexed documents."
                )
                structured: dict[str, Any] = {
                    "direct_answer": answer,
                    "evidence_points": [],
                    "assumptions": [],
                    "cannot_find": True,
                }
                grounded_ok = False
            else:
                numeric_safe = _is_numeric_question(req.question)
                structured = await _structured_answer(
                    req.question,
                    context_blocks,
                    numeric_safe=numeric_safe,
                    answer_length=answer_length,
                    chat_history=req.chat_history,
                )
                answer = structured.get("direct_answer") or (
                    "I could not find a direct answer in the indexed documents."
                )
                answer = _clean_answer_text(answer)
                answer = await _expand_answer_if_needed(
                    req.question,
                    answer,
                    context_blocks,
                    answer_length=answer_length,
                )
                structured["direct_answer"] = answer
                grounded_ok = True

            # 6. Confidence & grounding checks
            grounding = _validate_numeric_grounding(answer, context_blocks)
            confidence = _build_confidence_score(rerank_vals, grounding)

            low_conf_precheck = confidence < effective_conf_threshold
            low_numeric_precheck = (
                grounding.get("has_numbers", False)
                and grounding.get("grounded_ratio", 1.0)
                < req.numeric_grounding_threshold
            )

            if context_blocks and (low_conf_precheck or low_numeric_precheck):
                grounded_ok = await _self_check_answer(
                    req.question, answer, context_blocks
                )

            quality_payload = {
                "confidence": confidence,
                "grounding": grounding,
                "self_check_grounded": grounded_ok,
                "effective_top_k": effective_top_k,
                "query_rewrites": rewrites,
            }
            yield _sse("quality", quality_payload)

            # 7. Attach warnings for low confidence / poor grounding
            low_conf = confidence < effective_conf_threshold
            low_numeric = (
                grounding.get("has_numbers", False)
                and grounding.get("grounded_ratio", 1.0)
                < req.numeric_grounding_threshold
            )

            if low_conf or low_numeric or (not grounded_ok):
                reasons: list[str] = []
                if low_conf:
                    reasons.append(f"confidence {confidence:.2f} is low")
                if low_numeric:
                    reasons.append(
                        "numeric values are not fully grounded "
                        "in retrieved context"
                    )
                if not grounded_ok:
                    reasons.append(
                        "self-check found potential grounding issues"
                    )
                warning_note = (
                    "Note: " + "; ".join(reasons)
                    + ". Please verify with source context."
                )
                answer = f"{answer}\n\n{warning_note}"

            # 8. Stream answer tokens
            yield _sse("phase", "answer_stream")
            for i, word in enumerate(answer.split(" ")):
                token = word if i == 0 else " " + word
                yield _sse("token", token)

            # 9. Final metadata
            yield _sse("structured", structured)
            yield _sse(
                "metrics",
                {"latency_sec": round(time.time() - started, 3)},
            )
            yield _sse("done", None)

        except Exception as exc:
            logger.warning("query_stream failed: %s", exc)
            yield _sse("error", str(exc))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    """Delete an in-memory session and its index."""
    INDEX_STORE.pop(session_id, None)
    SESSION_META.pop(session_id, None)
    return {"deleted": session_id, "status": "ok"}


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------
def _sse(event_type: str, data: Any) -> str:
    """Format a Server-Sent Event payload."""
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"