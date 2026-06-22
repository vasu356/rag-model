"""
Pydantic schemas for the RAGLab API.

All request/response models defined here ensure type safety,
input validation, and clear API contract documentation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """A single turn in the conversation history."""

    role: str = Field(
        ...,
        description="Role of the speaker: 'user', 'ai', or 'assistant'.",
    )
    content: str = Field(
        ...,
        description="Text content of the chat message.",
    )


class QueryRequest(BaseModel):
    """Request payload for the streaming query endpoint."""

    session_id: str = Field(
        ...,
        description="Active session ID returned by the indexing endpoint.",
    )
    question: str = Field(
        ...,
        description="The user's question to answer against indexed documents.",
        min_length=1,
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of top chunks to retrieve (after reranking).",
    )
    alpha_dense: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Weight for dense (semantic) retrieval in hybrid search.",
    )
    alpha_bm25: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 (keyword) retrieval in hybrid search.",
    )
    answer_length: str = Field(
        default="medium",
        pattern=r"^(short|medium|detailed)$",
        description="Target answer verbosity: 'short', 'medium', or 'detailed'.",
    )
    confidence_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score threshold. Answers below this "
        "trigger a self-check and warning.",
    )
    numeric_grounding_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable grounding ratio for numeric values. "
        "If numeric answers are less grounded than this threshold, "
        "a self-check is triggered.",
    )
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior conversation context (last ~12 turns).",
    )


class SessionResponse(BaseModel):
    """Response returned after indexing documents."""

    session_id: str = Field(
        ...,
        description="Unique session ID for subsequent queries.",
    )
    doc_count: int = Field(
        ...,
        ge=0,
        description="Number of documents indexed.",
    )
    chunk_count: int = Field(
        ...,
        ge=0,
        description="Total number of text chunks created from documents.",
    )
    doc_names: list[str] = Field(
        ...,
        description="Filenames of the indexed documents.",
    )


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    status: str = Field(
        ...,
        description="Server status ('ok' if healthy).",
    )
    version: str = Field(
        ...,
        description="API version string.",
    )
    llm_model: str = Field(
        ...,
        description="LLM model identifier (e.g. 'llama-3.1-8b-instant').",
    )
    groq_key_present: bool = Field(
        ...,
        description="Whether a Groq API key has been configured.",
    )


class IndexingStatusResponse(BaseModel):
    """Optional: snapshot of current indexing pipeline status."""

    active_sessions: int = Field(
        default=0,
        ge=0,
        description="Number of active in-memory sessions.",
    )
    total_chunks: int = Field(
        default=0,
        ge=0,
        description="Total chunks across all active sessions.",
    )