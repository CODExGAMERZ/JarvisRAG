# Chapter 7: Retrieval-Augmented Generation (RAG) Systems Engineering

This guide covers retrieval mechanics, vector databases, indexing structures, and system memory optimizations.

---

## Part 1: Beginner: TF-IDF vs. Semantic Search

### 1. What is Document Search?
- **Keyword Search (TF-IDF)**: Matches exact words in a text. If you search for "automobile", it cannot find documents containing "car" because the exact letters do not match.
- **Semantic Search**: Converts text into vector embeddings that represent meanings. It matches "automobile" and "car" because their vectors point in similar directions in space.

### 2. Retrieval-Augmented Generation (RAG)
RAG optimizes LLM outputs by retrieving matching context blocks from local databases before prompting:
`User Query -> Vector Search -> Context Packaging -> LLM Prompt -> Answer`

---

## Part 2: Intermediate: Semantic Ingestion and Chunking

Instead of splitting text by fixed character limits, we split documents where the topic shifts.

### Semantic-Distance Chunking
1. Segment the document into sentences: \(S = (s_1, \dots, s_T)\).
2. Compute embeddings for each sentence: \(v_t = \text{Embed}(s_t)\).
3. Compute cosine distances between adjacent sentences:
   \[d_t = 1 - \frac{v_t \cdot v_{t+1}}{\|v_t\| \|v_{t+1}\|}\]
4. Set the split threshold \(\tau\) based on standard deviation offsets:
   \[\tau = \mu_d + k \cdot \sigma_d\]
   where \(\mu_d\) is the mean distance, \(\sigma_d\) is the standard deviation, and \(k\) is a tuning parameter (typically 1.2).
5. Execute splits at any index \(t\) where \(d_t > \tau\).

---

## Part 3: Expert: IVF and HNSW Indexing

Vector databases index embeddings to perform fast nearest neighbor queries.

### 1. Inverted File (IVF) Indexes
IVF clusters the vector space into \(K\) Voronoi cells using K-Means.
- The index stores centroids \(c_1, \dots, c_K\).
- During queries, the query vector is compared only against the centroids to find the closest clusters (nprobe). The search is restricted to vectors inside those clusters.
- **Search Complexity**: Reduces complexity from \(\mathcal{O}(N)\) to \(\mathcal{O}(\text{nprobe} \times \frac{N}{K})\).

### 2. Hierarchical Navigable Small World (HNSW)
HNSW constructs a multi-layer graph where:
- Top layers contain long-range edges (skips) between distant nodes.
- Bottom layers contain short-range edges linking near neighbors.
- Nodes are assigned to layers based on an exponential decay probability distribution:
  \[l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor\]
- **Search Complexity**: Routes queries in \(\mathcal{O}(\log N)\) search time.

---

## Part 4: OP Level: Re-Ranking and Memory Optimizations

### 1. Re-Ranking: Bi-Encoders vs. Cross-Encoders
- **Bi-Encoder**: Computes query and document embeddings independently:
  \[\text{score} = \cos(Q, D)\]
  Low-latency, suitable for the first-stage retrieval.
- **Cross-Encoder**: Feeds the concatenated query and document into a single encoder:
  \[\text{Input} = [\text{CLS}] \, q_1 \dots q_n \, [\text{SEP}] \, d_1 \dots d_m\]
  Self-attention layers compute cross-correlations directly across query and document tokens simultaneously. High accuracy, but computationally expensive.

---

### 2. Memory-Mapped Vector Loading (`mmap`)
To keep system RAM consumption completely flat when loading large vector databases, we load matrices via `mmap_mode='r'`. The operating system kernel maps the virtual memory address space of the process directly to the file offsets on disk.
- Data is paged into physical RAM on-demand using page faults.
- Unused pages are automatically evicted by the OS page cache manager, avoiding python heap allocations.

```python
import numpy as np

# Load vectors from storage via memory-map
embeddings = np.load("index.npy", mmap_mode='r')
```
