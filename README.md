# RAG Demo — FastAPI + React

A full-stack Retrieval-Augmented Generation (RAG) demo with:
- Python FastAPI backend (`backend/`)
- React + Vite frontend (`frontend/`)
- Document indexing from sample files or uploads
- Hybrid dense + BM25 retrieval
- Groq LLM streaming answer generation
- Numeric grounding and answer quality checks

## Project Structure

```text
rag-demo/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── services/
│   │       ├── retrieval.py
│   │       └── answering.py
│   ├── data/
│   │   └── samples/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── components/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── .env.example
└── README.md
```

## Features

- Index sample documents and user-uploaded `.txt`, `.md`, `.pdf` files
- Stream answer tokens from the backend to the frontend
- Combine embeddings and BM25 retrieval for more robust search
- Use a persistent session ID for each indexing session
- Validate numeric answers against retrieved document context
- Local embeddings with `BAAI/bge-small-en-v1.5` and reranking with `BAAI/bge-reranker-base`

## Requirements

- Python 3.10+
- Node.js 18+ / npm 10+
- A valid `GROQ_API_KEY`

## Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY
uvicorn app.main:app --reload --port 8000
```

### Backend environment

- `GROQ_API_KEY` is required for the Groq LLM integration.
- `GROQ_MODEL` can be overridden in `.env`; default is `llama-3.1-8b-instant`.
- Embeddings are computed locally via Hugging Face.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open the UI at:
- `http://localhost:5173`

## API endpoints

- `GET /health` — API health and model status
- `GET /samples` — list available sample documents
- `POST /index/samples` — index selected sample documents
- `POST /index/upload` — upload and index files
- `POST /query/stream` — stream an answer from the indexed session
- `DELETE /session/{session_id}` — clear session data from backend memory

## Usage notes

- The UI supports sample indexing and file uploads.
- The backend keeps in-memory session indexes; restarting the backend clears sessions.
- The `.env` file is ignored by `.gitignore`, so secrets are safe from commits.
- There is no `run-python-backend.sh` file in this repository; use the commands above.

## Git / repository state

- `.gitignore` already excludes:
  - `venv/`
  - `.env`
  - `node_modules/`
  - `dist/`
  - `.vite/`
  - `__pycache__/`
  - `/tmp/rag_uploads/`
  - `.DS_Store`
- This workspace currently does not appear to have a valid Git repository initialized in the environment I checked.
- Initialize git in the project root before pushing if needed:

```bash
git init
git add .
git commit -m "Initial commit"
```

## Notes

- Make sure `backend/.env` contains your `GROQ_API_KEY` before starting the backend.
- If you want local embeddings only, the Hugging Face model will download on first run.
- The frontend and backend are separate apps; start both to use the full UI.
