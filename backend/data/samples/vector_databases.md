# Vector Databases: The Memory Layer for AI

## Introduction

A vector database is a specialized data store designed to efficiently store, index, and search high-dimensional vectors — the numerical representations (embeddings) that encode the semantic meaning of text, images, audio, and more.

As AI applications proliferate, vector databases have become the critical "long-term memory" layer for LLM-powered systems like RAG pipelines, semantic search engines, and recommendation systems.

## How Vector Search Works

Traditional databases search by exact match or range queries on structured fields. Vector databases search by **similarity** — finding the nearest neighbors to a query vector in high-dimensional space.

### Approximate Nearest Neighbor (ANN)

Exhaustive search over millions of vectors is impractical. ANN algorithms trade a small accuracy loss for massive speed gains:

- **HNSW** (Hierarchical Navigable Small World): Graph-based index. Excellent recall/speed tradeoff. Used by Weaviate, Qdrant.
- **IVF** (Inverted File Index): Clusters vectors into buckets, searches only nearby clusters. Used by FAISS.
- **LSH** (Locality Sensitive Hashing): Hash-based bucketing for sub-linear search.
- **ScaNN** (Google): Production-grade ANN with learned quantization.

## Similarity Metrics

- **Cosine similarity**: Angle between vectors. Best for text embeddings (magnitude-invariant).
- **Dot product**: Fast, used when embeddings are normalized.
- **Euclidean (L2) distance**: Geometric distance. Used for image and audio embeddings.

## Popular Vector Databases

### FAISS (Meta AI)
Open-source library (not a full DB). Extremely fast, GPU-accelerated. Best for in-memory or research use cases.

### Chroma
Open-source, developer-friendly. Excellent for local development and small-to-medium RAG apps. Persistent storage with SQLite backend.

### Pinecone
Managed cloud service. Serverless pricing, production-grade scalability. Supports metadata filtering alongside vector search.

### Weaviate
Open-source with cloud option. Supports hybrid search (BM25 + vector). Schema-based with GraphQL API.

### Qdrant
Rust-based, high performance. Rich filtering. Excellent for self-hosting.

## Metadata Filtering

Modern vector DBs support filtering on metadata fields alongside vector similarity:
```
query: "machine learning basics"
filter: {source: "textbook", year: {$gte: 2020}}
```
This enables precise retrieval without sacrificing semantic search.

## Choosing the Right Store

| Use Case | Recommendation |
|---|---|
| Local dev / prototype | Chroma, FAISS |
| Production cloud app | Pinecone, Weaviate Cloud |
| Self-hosted production | Qdrant, Weaviate |
| Billions of vectors | Pinecone, custom FAISS |
