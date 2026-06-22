# Vector Databases: The Memory Layer for AI

## Introduction

A vector database is a specialized data store designed to efficiently store, index, and search high-dimensional vectors, the numerical representations that encode semantic meaning in text, images, audio, and more.

As AI applications proliferate, vector databases have become the long-term memory layer for RAG pipelines, semantic search engines, and recommendation systems.

## How Vector Search Works

Traditional databases search by exact match or range queries on structured fields. Vector databases search by similarity, finding the nearest neighbors to a query vector in high-dimensional space.

### Approximate Nearest Neighbor (ANN)

Exhaustive search over millions of vectors is impractical. ANN algorithms trade a small accuracy loss for major speed gains:

- **HNSW** (Hierarchical Navigable Small World): Graph-based index with an excellent recall-speed tradeoff.
- **IVF** (Inverted File Index): Clusters vectors into buckets and searches only nearby clusters.
- **LSH** (Locality Sensitive Hashing): Hash-based bucketing for sub-linear search.
- **ScaNN** (Google): Production-grade ANN with learned quantization.

## Similarity Metrics

- **Cosine similarity**: Angle between vectors. Best for text embeddings because it is magnitude invariant.
- **Dot product**: Fast and commonly used when embeddings are normalized.
- **Euclidean (L2) distance**: Geometric distance, often used for image and audio embeddings.

## Popular Vector Databases

### FAISS (Meta AI)
Open-source library, not a full database. Extremely fast and GPU-accelerated. Best for in-memory or research use cases.

### Chroma
Open-source and developer-friendly. Good for local development and small-to-medium RAG apps.

### Pinecone
Managed cloud service with serverless pricing and production-grade scalability.

### Weaviate
Open-source with a cloud option. Supports hybrid search with BM25 plus vector search.

### Qdrant
Rust-based, high performance, and rich filtering. Excellent for self-hosting.

## Metadata Filtering

Modern vector databases support filtering on metadata fields alongside vector similarity:

```text
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
