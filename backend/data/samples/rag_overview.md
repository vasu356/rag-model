# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that improves large language model (LLM) outputs by grounding them in external, up-to-date knowledge retrieved at inference time.

Proposed by Lewis et al. (2020) at Facebook AI Research, RAG addresses a key limitation of LLMs: their knowledge is frozen at training time. By pairing an LLM with a retrieval system, RAG enables models to answer questions about documents they never saw during training.

## The RAG Pipeline

A standard RAG pipeline has three stages:

### 1. Indexing (Offline)
- **Chunking**: Source documents are split into smaller passages (chunks), typically 256-1024 tokens with overlap to preserve context across boundaries.
- **Embedding**: Each chunk is converted to a dense vector using an embedding model, for example text-embedding-3-small.
- **Storage**: Vectors are stored in a vector database such as FAISS, Pinecone, Weaviate, or Chroma.

### 2. Retrieval (Online)
When a user submits a query:
- The query is embedded using the same embedding model.
- A similarity search, such as cosine similarity or MIPS, retrieves the top-k most relevant chunks.
- Optional reranking improves precision, for example with cross-encoders.

### 3. Generation
- Retrieved chunks are injected into the LLM prompt as context.
- The LLM generates an answer grounded in the retrieved evidence.
- Sources can be cited for transparency.

## Why RAG vs. Fine-Tuning?

| Aspect | RAG | Fine-Tuning |
|---|---|---|
| Knowledge updates | Instant, swap docs | Requires retraining |
| Cost | Low | High, GPU compute |
| Hallucination | Reduced via grounding | Still possible |
| Interpretability | Sources are explicit | Black box |

RAG is generally preferred for knowledge-intensive tasks where accuracy and freshness matter.

## Advanced Techniques

- **HyDE** (Hypothetical Document Embeddings): Generate a hypothetical answer, embed it for retrieval.
- **Query decomposition**: Break complex questions into sub-queries.
- **Contextual chunking**: Add document-level summaries to each chunk's metadata.
- **Hybrid search**: Combine dense semantic search and sparse BM25 retrieval.

## LlamaIndex

LlamaIndex is an open-source framework for building RAG pipelines. It provides:
- Document loaders for 100+ formats, including PDF, HTML, Notion, and Slack
- Flexible chunking strategies such as sentence, token, and hierarchical splitting
- Integrations with major LLMs and vector stores
- High-level abstractions and low-level control over retrievers and post-processors
