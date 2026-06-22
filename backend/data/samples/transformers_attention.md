# The Attention Mechanism in Transformers

## Overview

The attention mechanism is the foundation of modern large language models (LLMs). Introduced in the 2017 paper Attention Is All You Need by Vaswani et al., it replaced recurrent neural networks as the dominant architecture for sequence modeling.

## How Self-Attention Works

Self-attention allows each token in a sequence to attend to every other token, computing a weighted sum of value vectors based on query-key dot products.

### Step-by-step

1. **Linear projections**: Each input embedding is projected into three vectors: Query (Q), Key (K), and Value (V), using learned weight matrices.
2. **Score computation**: The attention score between token i and j is score(i, j) = Q_i · K_j / sqrt(d_k), where d_k is the key dimension.
3. **Softmax normalization**: Scores are normalized with softmax to produce attention weights that sum to 1 across all positions.
4. **Weighted sum**: The output for each position is a weighted sum of all value vectors: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V.

## Multi-Head Attention

Rather than computing a single attention function, Transformers use multi-head attention: running h parallel attention heads, each with its own Q/K/V projections, then concatenating and linearly projecting the outputs.

This allows the model to attend to different representation subspaces simultaneously. For example, one head might focus on syntactic relationships, while another focuses on semantic similarity.

## Positional Encoding

Since attention is permutation-invariant, positional encodings, either sinusoidal or learned, are added to input embeddings to inject sequence order information.

## Key Properties

- **Parallelism**: Unlike RNNs, all positions are processed simultaneously during training.
- **Long-range dependencies**: Any two positions can interact in O(1) operations regardless of distance.
- **Interpretability**: Attention weights can be visualized to understand what the model focuses on.

## Applications

The Transformer architecture powers GPT, BERT, T5, LLaMA, Claude, and nearly every state-of-the-art NLP system today.
