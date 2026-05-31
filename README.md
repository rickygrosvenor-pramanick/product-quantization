# Product Quantization

A from-scratch implementation of **Product Quantization (PQ)** for compressing high-dimensional embeddings and performing approximate nearest-neighbour search — no vector database required.

## What is Product Quantization?

PQ splits each D-dimensional vector into **M sub-vectors**, then independently quantizes each sub-space using k-means with **K centroids**. The result is a compact code of M integers (each 1–2 bytes) instead of D floats (D×4 bytes).

```
Original vector (384 floats = 1536 bytes)
┌────────────────────────────────────────────────────┐
│  sub_0  │  sub_1  │  ...  │  sub_7  │  ← M=8 splits
└────────────────────────────────────────────────────┘
     ↓          ↓                ↓
   idx_0      idx_1      ...   idx_7     ← K=256 → uint8
┌────────────────────────────────────────┐
│ 8 bytes = 192× compression             │
└────────────────────────────────────────┘
```

Retrieval uses **Asymmetric Distance Computation (ADC)**: the query stays in full precision while database vectors are represented by their PQ codes. Distances are computed via pre-built lookup tables — one table lookup per sub-space per candidate — making search sub-millisecond even over tens of thousands of vectors.

## Features

- **ProductQuantizer** — train codebooks, encode/decode vectors, compute asymmetric distances
- **PQIndex** — HNSW graph over reconstructed vectors with PQ re-ranking for sub-linear search
- **SimpleRetrievalSystem** — end-to-end text retrieval with PQ compression, optional exact-distance re-ranking, and metadata filtering (e.g. by sentiment)
- **EmbeddingGenerator** — Cohere API wrapper (`embed-english-light-v3.0`, 384-dim) with batching and progress bars
- **Evaluation** — recall@K, latency benchmarks, compression-vs-error analysis, and matplotlib visualizations

## Project Structure

```
├── pq.py                 # ProductQuantizer, PQIndex, evaluation utilities
├── embeddings.py          # Cohere embedding generation, sample data creation
├── data_processor.py      # Data pipeline, SimpleRetrievalSystem
├── demo.ipynb             # Interactive Jupyter notebook walkthrough
├── data.csv               # ~5.8k financial sentiment sentences
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone and install
git clone <repo-url> && cd product-quantization
pip install -r requirements.txt

# (Optional) For Cohere embeddings, create a .env file:
echo "COHERE_API_KEY=your-key-here" > .env
```

Without a Cohere key, the system falls back to random normalized embeddings for demonstration.

## Quick Start

### 1. Train and search in Python

```python
from pq import ProductQuantizer, generate_random_embeddings

# Generate sample data
embeddings = generate_random_embeddings(10_000, dimension=384)

# Train PQ with 8 sub-spaces × 256 centroids
pq = ProductQuantizer(M=8, K=256)
pq.fit(embeddings)

# Encode → 192× compression
codes = pq.encode(embeddings)        # (10000, 8) uint8
print(pq.get_memory_usage(10_000))   # compression_ratio ≈ 192x

# Search using asymmetric distance
query = embeddings[0]
distances = pq.asymmetric_distance(query, codes)
top_5 = distances.argsort()[:5]
```

### 2. HNSW + PQ index (sub-linear search)

```python
from pq import ProductQuantizer, PQIndex, generate_random_embeddings

embeddings = generate_random_embeddings(50_000, 384)

pq = ProductQuantizer(M=8, K=256)
pq.fit(embeddings)

index = PQIndex(pq, max_elements=100_000)
index.add_vectors(embeddings)

# Search: HNSW shortlists candidates, PQ re-ranks with asymmetric distance
dists, ids = index.search(embeddings[0], k=10)
```

### 3. End-to-end text retrieval

```python
from data_processor import DataProcessor, SimpleRetrievalSystem

# Load data and embeddings
processor = DataProcessor("data.csv")
processor.load_data()
processor.load_processed_data()       # or processor.create_embeddings()

# Build retrieval system
system = SimpleRetrievalSystem(M=8, K=256)
system.train_quantizer(processor.embeddings)
system.index_documents(processor.embeddings, processor.texts, processor.sentiments)

# Search with optional re-ranking and sentiment filtering
results, dists, meta = system.search(
    query_embedding,
    k=5,
    rerank=True,            # PQ shortlist → exact L2 re-rank
    sentiment_filter="positive"
)
```

### 4. Run the full pipeline

```bash
python data_processor.py
```

This loads `data.csv`, generates (or loads cached) Cohere embeddings, trains PQ, evaluates recall@K, and starts an interactive search prompt.

### 5. Jupyter notebook

```bash
jupyter notebook demo.ipynb
```

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `M` (sub-spaces) | 8 | Higher → more compression, lower recall |
| `K` (centroids) | 256 | Higher → better recall, slower training |
| `rerank` | False | Exact L2 re-ranking on PQ shortlist for higher precision |
| `rerank_factor` | 4 | Fetch `k × rerank_factor` PQ candidates before re-ranking |
| `ef` (HNSW) | 50 | Higher → better recall, slower search |

## How It Works

1. **Training** — k-means clusters each sub-space independently, producing M codebooks of K centroids each.
2. **Encoding** — each vector is replaced by M centroid indices (1–2 bytes each).
3. **Search (ADC)** — for a query, pre-compute distances to all centroids in each codebook (M×K table), then sum looked-up distances across sub-spaces for each candidate.
4. **Re-ranking** (optional) — shortlist top candidates via PQ, then score with exact L2 for higher precision.
5. **PQIndex** — builds an HNSW graph on reconstructed (decoded) vectors for sub-linear candidate retrieval, then re-ranks with ADC.

## Dataset

The included `data.csv` contains ~5,842 financial news sentences labeled with sentiment (positive / negative / neutral), sourced from the FinancialPhraseBank dataset.

## License

MIT
