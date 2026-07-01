# Local Knowledge Base RAG

A local-first Retrieval-Augmented Generation pipeline: ingest `.txt` / `.md` / `.pdf`
files, embed them locally (sentence-transformers or Ollama), store vectors in
FAISS (or a pure-NumPy fallback), and synthesize grounded answers with Gemini.

## Files

| File | Purpose |
|---|---|
| `rag_pipeline.py` | Library: `DocumentIngestor`, `EmbeddingGenerator`, `VectorStorageEngine`, `GeminiRAGOrchestrator`. |
| `build_index.py` | CLI to (re)build/sync the vector store from a knowledge-base folder. |
| `query.py` | CLI to ask questions, interactively or one-shot (`-q`). |
| `demo.py` | Self-contained demo — writes to `./demo_knowledge_base` / `./demo_vector_store` only. |
| `test_rag.py` | Unit tests, including regression tests for the bugs listed below. |

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # optional — retrieval still works without it
```

If `sentence-transformers` isn't installed, the pipeline automatically falls back
to a local [Ollama](https://ollama.com) embedding endpoint (`nomic-embed-text` by default).

## Usage

```bash
# 1. Index your knowledge base (defaults to ./my_knowledge_base -> ./vector_store)
python build_index.py --kb-dir ./my_knowledge_base --store-dir ./vector_store

# 2. Ask questions
python query.py                              # interactive
python query.py -q "What temperature does the stabilizer operate at?"

# Try it without any setup
python demo.py
```

Re-running `build_index.py` is incremental: unchanged files are skipped (via
SHA-256 checksum), modified files are re-chunked and re-embedded, and deleted
files are purged from the store automatically.

## What changed from the original version

**Correctness fixes**
- `VectorStorageEngine.remove_files` no longer leaves a file "stuck": it used to
  skip clearing `processed_files` bookkeeping whenever a tracked file happened to
  contribute zero stored chunks (e.g. it previously failed to parse), which meant
  the file could never be re-ingested on a later sync. It's now always cleared.
- `EmbeddingGenerator` no longer silently zero-fills failed Ollama embeddings with
  a hardcoded 384-dim vector (correct for MiniLM, wrong for `nomic-embed-text`'s
  768 dims) — the real dimension is now detected from a successful call.
- `VectorStorageEngine.add_vectors` now validates embedding dimensions and raises
  a clear error on mismatch (e.g. after switching embedding models) instead of
  failing obscurely inside `np.concatenate`.
- The one-off demo (`__main__` block in the old `rag_pipeline.py`) used to
  overwrite files inside `./my_knowledge_base` every time the script ran — which
  could clobber real content. Demo content now lives in its own
  `./demo_knowledge_base` / `./demo_vector_store`, isolated from real data.
- `save_index` writes metadata via a temp-file-plus-`os.replace` swap so a crash
  mid-write can't corrupt the store.
- PDF parsing now handles encrypted PDFs (tries an empty password) and per-page
  extraction failures without aborting the whole file.
- An oversized single line (e.g. PDF text with no newlines) that alone exceeds
  the chunk size limit is now hard-split instead of silently producing one
  oversized chunk.

**Retrieval quality**
- Chunking now carries a configurable character-overlap between consecutive
  chunks so context isn't lost right at a chunk boundary.
- `similarity_search`'s NumPy fallback uses `argpartition` instead of a full
  `argsort` for top-k selection (faster on large stores).

**Efficiency**
- Ollama embedding requests are issued concurrently (thread pool) instead of
  one-by-one, which is the dominant cost when embedding a large knowledge base
  without a local sentence-transformers install.
- Network calls to Ollama and Gemini use exponential-backoff retries instead of
  failing (or, for Ollama, silently corrupting the index) on the first transient
  error.

**Security**
- The Gemini system prompt now explicitly instructs the model to treat retrieved
  document text as untrusted data, not instructions, mitigating prompt injection
  from ingested files.
- Retrieved chunk text is sanitized to neutralize any literal `</retrieved_context>`
  (or similar) sequences that could otherwise be used to break out of the context
  wrapper.

**Usability**
- `build_index.py` and `query.py` are proper `argparse` CLIs (configurable
  store/kb dirs, embedding mode, k, threshold, batch size) instead of hardcoded
  paths buried in `if __name__ == "__main__"` blocks.
- `sync_directory` returns a summary (files ingested/removed/failed, chunks
  added) instead of only logging.

## Configuration

All tunables are environment-variable overridable (see the top of
`rag_pipeline.py`): `RAG_CHUNK_MAX_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_DEFAULT_K`,
`RAG_DEFAULT_THRESHOLD`, `RAG_EMBED_BATCH_SIZE`, `RAG_OLLAMA_MAX_WORKERS`,
`RAG_NETWORK_MAX_RETRIES`, `GEMINI_MODEL`.

## Tests

```bash
python -m unittest test_rag.py -v
```
