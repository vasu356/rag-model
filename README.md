<div align="center">
  <br />
  <h1>🧪 RAGLab</h1>
  <p>
    <strong>A Production-Quality Retrieval-Augmented Generation Demo</strong>
  </p>
  <p>
    <em>Hybrid Search · Cross-Encoder Reranking · SSE Streaming · Confidence Scoring</em>
  </p>
  <br />
  <p>
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#api-reference">API</a> •
    <a href="#project-structure">Structure</a> •
    <a href="#testing">Testing</a> •
    <a href="#docker">Docker</a>
  </p>
  <br />
</div>

---

## Overview

**RAGLab** is a full-stack **Retrieval-Augmented Generation** application that demonstrates how to ground LLM responses in your own documents. It combines **dense semantic search** (embeddings) with **BM25 keyword search**, re-ranks results with a **cross-encoder**, and streams answers via **Server-Sent Events** — all with transparent source tracing and confidence scoring.

---

## Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | Combines dense embeddings (`BAAI/bge-small-en-v1.5`) with BM25 keyword search via Reciprocal Rank Fusion (RRF) |
| **Two-Pass Reranking** | Cross-encoder (`BAAI/bge-reranker-base`) re-ranks candidates in two passes for high precision |
| **Query Expansion** | LLM generates up to 3 query rewrites to improve recall |
| **Adaptive Top-K** | Automatically adjusts the number of retrieved chunks based on question type (numeric, explanatory, etc.) |
| **Numeric Grounding** | Validates that numeric values in answers appear in source documents |
| **Confidence Scoring** | Combines reranker scores + numeric grounding into a single [0,1] confidence metric |
| **Self-Check** | LLM-based verification when confidence is low |
| **SSE Streaming** | Real-time token-by-token answer streaming |
| **Source Tracing** | Every answer includes source snippets with relevance scores |
| **Chat History** | Maintains conversation context across turns |
| **Multi-Format Support** | Index TXT, MD, and PDF files |
| **Configurable Pipeline** | Adjust top-K, dense/keyword balance, answer length, and confidence thresholds from the UI |

---

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐
│   Browser   │     │                   FastAPI Backend                    │
│  (React +   │     │                                                     │
│   Vite)     │     │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│             │     │  │  Index   │  │ Retrieve │  │    Generate      │  │
│  ┌───────┐  │     │  │          │  │          │  │                  │  │
│  │Setup  │──┼─────┼─>│• Chunk   │  │• Dense   │  │• Prompt build    │  │
│  │Panel  │  │     │  │• Embed   │  │• BM25    │  │• LLM call (Groq) │  │
│  └───────┘  │     │  │• Store   │  │• RRF fuse│  │• Clean & expand  │  │
│             │     │  └──────────┘  │• Rerank  │  │• Self-check      │  │
│  ┌───────┐  │     │                └──────────┘  └──────────────────┘  │
│  │Chat   │<─┼─────┼──────────────── SSE Stream ───────────────────────│  │
│  │Panel  │  │     │                                                     │
│  └───────┘  │     └─────────────────────────────────────────────────────┘
└─────────────┘
```

### Pipeline Flow

1. **Indexing** — Documents are chunked (512 tokens, 64 overlap), embedded with `bge-small-en-v1.5`, and stored in an in-memory vector index.
2. **Query Expansion** — The LLM generates up to 3 reformulations of the user's question.
3. **Hybrid Retrieval** — Each query variant is searched via dense embeddings AND BM25. Results are fused with RRF.
4. **Two-Pass Reranking** — A cross-encoder scores candidates: first a wide pass (2× top-K), then a narrow pass (top-K).
5. **Answer Generation** — Retrieved chunks are injected into a structured prompt. The LLM returns JSON with `direct_answer`, `evidence_points`, and `assumptions`.
6. **Quality Checks** — Numeric grounding is validated, confidence is scored, and a self-check is triggered if thresholds aren't met.
7. **Streaming** — The answer is streamed token-by-token over SSE, along with source cards and quality metadata.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11) |
| **LLM** | [Groq](https://groq.com/) — `llama-3.1-8b-instant` |
| **Embeddings** | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (local, ~33 MB) |
| **Reranker** | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) (local) |
| **RAG Framework** | [LlamaIndex](https://www.llamaindex.ai/) |
| **Frontend** | [React 18](https://react.dev/) + [Vite 5](https://vitejs.dev/) |
| **Streaming** | Server-Sent Events (SSE) |
| **Containerization** | Docker + Docker Compose |
| **Testing** | pytest |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm 10+
- A [Groq API key](https://console.groq.com) (free tier available)

### 1. Clone & Configure

```bash
git clone https://github.com/vasu356/rag-model.git
cd rag-model

# Set up environment
cp .env.example .env
```

Edit `.env` with your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open the app at `http://localhost:5173`.

### 4. Use the App

1. Select one or more sample documents (or upload your own TXT/MD/PDF files).
2. Click **Index Selected Documents**.
3. Ask questions in the chat panel.
4. Explore source cards, confidence scores, and retrieval metadata.

---

## Docker

Run the entire stack with Docker Compose:

```bash
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Backend health & LLM status |
| `GET` | `/samples` | List bundled sample documents |
| `POST` | `/index/samples` | Index selected sample documents |
| `POST` | `/index/upload` | Upload & index files (TXT, MD, PDF) |
| `POST` | `/query/stream` | Stream a grounded answer (SSE) |
| `DELETE` | `/session/{id}` | Clear in-memory session |

### Query Request

```json
{
  "session_id": "uuid-from-indexing",
  "question": "What is RAG?",
  "top_k": 4,
  "alpha_dense": 0.65,
  "alpha_bm25": 0.35,
  "answer_length": "medium",
  "confidence_threshold": 0.45,
  "numeric_grounding_threshold": 0.8,
  "chat_history": []
}
```

### SSE Event Types

| Event Type | Description |
|---|---|
| `phase` | Lifecycle updates (`retrieval_started`, `sources_ready`, `answer_stream`) |
| `sources` | Retrieved & reranked source chunks with scores |
| `quality` | Confidence score, grounding metadata, query rewrites |
| `token` | Streamed answer tokens |
| `structured` | Structured answer metadata (evidence, assumptions) |
| `metrics` | Latency information |
| `done` | Stream complete |
| `error` | Error occurred |

---

## Project Structure

```
rag-model/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Package metadata
│   │   ├── main.py              # FastAPI application & endpoints
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── services/
│   │       ├── retrieval.py     # Hybrid retrieval, reranking, BM25, confidence
│   │       └── answering.py     # LLM prompt building, answer cleaning, self-check
│   ├── data/
│   │   └── samples/             # Bundled sample documents
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_services.py     # Unit tests (40+ test cases)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.jsx             # React entry point
│   │   ├── App.jsx              # Main application component
│   │   ├── index.css            # Global styles
│   │   └── components/
│   │       ├── Header.jsx       # App header with session info
│   │       ├── SetupPanel.jsx   # Document selection/upload UI
│   │       ├── ChatPanel.jsx    # Chat interface with source cards
│   │       ├── SidebarPanel.jsx # Session stats & tech stack
│   │       ├── MdText.jsx       # Simple markdown renderer
│   │       └── Icons.jsx        # SVG icon components
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── Dockerfile.backend           # Backend container
├── Dockerfile.frontend          # Frontend container (nginx)
├── docker-compose.yml           # Multi-service orchestration
├── .env.example                 # Environment template
├── .gitignore
├── LICENSE                      # MIT License
└── README.md
```

---

## Testing

```bash
cd backend
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

The test suite covers:
- Tokenization & numeric detection
- Adaptive top-K logic
- Numeric grounding validation
- Confidence scoring
- Answer cleaning & normalisation
- History serialisation
- List/bullet formatting utilities

---

## Key Design Decisions

### Why Hybrid Search?
Dense embeddings capture semantic meaning but miss exact keyword matches. BM25 excels at keyword precision but ignores semantics. RRF fusion gives you the best of both worlds.

### Why Two-Pass Reranking?
Cross-encoders are expensive (O(n) per pair). By first narrowing candidates with a wide pass, then re-ranking the top with a narrow pass, we get high precision without excessive compute.

### Why In-Memory Indexing?
For a demo, in-memory storage keeps setup trivial (no vector database required) and makes session cleanup automatic. For production, swap in Pinecone, Weaviate, or Qdrant.

### Why SSE over WebSockets?
SSE is simpler for unidirectional streaming (server → client), works over standard HTTP, and integrates naturally with FastAPI's `StreamingResponse`.

---

## What I Learned

Building RAGLab deepened my understanding of:

- **Information Retrieval**: Dense vs. sparse retrieval, RRF fusion, BM25 scoring
- **Reranking Strategies**: Cross-encoder vs. bi-encoder trade-offs, two-pass pipelines
- **LLM Integration**: Structured JSON prompting, fallback handling, rate-limit management
- **Real-Time Systems**: SSE streaming, async generators, backpressure
- **Full-Stack Architecture**: FastAPI + React integration, CORS, proxy configuration
- **Quality Assurance**: Numeric grounding validation, confidence calibration, self-check mechanisms

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>
    Built with ❤️ by <a href="https://github.com/vasu356">Vasu Agrawal</a>
  </p>
  <p>
    <a href="https://github.com/vasu356/rag-model">GitHub</a> •
    <a href="https://www.linkedin.com/in/vasu-agrawal-m26a2003y">LinkedIn</a>
  </p>
</div>