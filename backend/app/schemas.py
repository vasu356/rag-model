from pydantic import BaseModel


class ChatTurn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 4
    alpha_dense: float = 0.65
    alpha_bm25: float = 0.35
    answer_length: str = "medium"  # short | medium | detailed
    confidence_threshold: float = 0.45
    numeric_grounding_threshold: float = 0.8
    chat_history: list[ChatTurn] = []


class SessionResponse(BaseModel):
    session_id: str
    doc_count: int
    chunk_count: int
    doc_names: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_model: str
    groq_key_present: bool
