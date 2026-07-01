# jarvisRAG — Local Knowledge Base RAG System

A local-first Retrieval-Augmented Generation app: a FastAPI backend that ingests
`.txt` / `.md` / `.pdf` files into a FAISS vector store, retrieves relevant
chunks with local embeddings, and synthesizes grounded answers with Gemini —
plus a full browser dashboard (command console, retrieval stream, knowledge
base manager, and config panel) to drive it.

## ⚠️ Before you do anything else

This project previously had a **live Gemini API key committed in
`backend/config.json`**. It has been removed from this copy, but if that key
was ever pushed to a public or shared repo, **rotate/revoke it now** in
[Google AI Studio](https://aistudio.google.com/apikey) — treat it as
compromised. `backend/config.json` is now git-ignored (see `.gitignore`) so
this can't happen again; use `backend/config.example.json` as the template
and set your real key through the Config page or the `GEMINI_API_KEY`
environment variable instead of hand-editing the file.

## Project layout

```
.
├── main.py                  # FastAPI app: serves the frontend + REST API
├── requirements.txt          # Full app dependencies (web server + RAG pipeline)
├── frontend/                 # Static dashboard (command / stream / knowledge / config)
└── backend/
    ├── rag_pipeline.py       # Core library: ingest, embed, store, retrieve, synthesize
    ├── build_index.py        # CLI: (re)build the vector store
    ├── query.py               # CLI: ask questions from the terminal
    ├── demo.py                 # Self-contained demo (isolated sandbox data)
    ├── test_rag.py             # Unit tests
    ├── my_knowledge_base/     # Your source documents (.txt / .md / .pdf)
    ├── vector_store/            # Generated FAISS index + metadata (git-ignored)
    └── config.example.json     # Template — copy to config.json if you want a starting file
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Local embeddings use `sentence-transformers` (CPU) by default. If it isn't
installed, the app automatically falls back to a local
[Ollama](https://ollama.com) endpoint (`nomic-embed-text` by default) — make
sure Ollama is running if you go that route.

## Running the app

```bash
python main.py
```

This starts the FastAPI server at **http://127.0.0.1:8000** and serves the
dashboard at `/command.html` (root `/` redirects there). Set your Gemini API
key from the **Config** page in the UI, or export it before starting:

```bash
export GEMINI_API_KEY=your_key_here   # optional — retrieval still works without it
python main.py
```

Without a key, `/api/query` still runs retrieval and returns the matched
context chunks, just skips the Gemini synthesis step (`status:
"SYNTHESIS_SKIPPED"`).

### Dashboard pages

| Page | Purpose |
|---|---|
| `/command.html` | Landing/overview: live system vitals, sync task queue, quick query box. |
| `/stream.html` | Ask questions and see the synthesized answer plus every retrieved chunk with similarity scores. |
| `/knowledge.html` | Upload/delete/view documents in the knowledge base and trigger re-indexing. |
| `/config.html` | Model + sampling parameters, API key, live log tail, terminal-style admin commands. |

### REST API

| Method & Path | Purpose |
|---|---|
| `GET /api/config` | Current generation settings (API key is write-only, never returned). |
| `POST /api/config` | Update settings / API key. |
| `GET /api/documents` | List knowledge-base files with indexed/pending status. |
| `POST /api/documents/upload` | Upload a `.txt` / `.md` / `.pdf` file (25 MB limit). |
| `GET /api/documents/{filename}/raw` | Fetch a document's raw contents (used by the "view" button). |
| `DELETE /api/documents/{filename}` | Remove a file and its vectors. |
| `POST /api/documents/sync` | Re-index the knowledge base in the background. |
| `GET /api/documents/sync/status` | Poll the background sync job. |
| `POST /api/query` | Run retrieval (+ synthesis if a key is configured). |
| `GET /api/system/vitals` | CPU/memory/uptime/latency/vector-store stats. |
| `GET /api/logs` | Recent server log lines (ring buffer, last 500). |
| `POST /api/config/flush` | Clear the log buffer. |
| `POST /api/config/reboot` | Reload the vector index from disk without restarting the process. |

### CLI tools (backend/)

You can also drive the pipeline entirely from the terminal, without the web
app:

```bash
cd backend
python build_index.py                          # (re)build the vector store
python query.py -q "What temperature does X operate at?"
python demo.py                                    # sandboxed end-to-end demo
python -m unittest test_rag.py -v                 # unit tests
```

See `backend/README.md` for CLI details and the pipeline's design notes.

## Security notes

- **Prompt injection**: retrieved document text is treated as untrusted data
  in the Gemini system prompt (documents can't override instructions), and
  literal `</retrieved_context>`-style breakout sequences are neutralized
  before being sent to the model.
- **Path traversal**: upload/delete/view endpoints resolve filenames to a
  bare basename and verify the result stays inside `backend/my_knowledge_base`
  before touching disk.
- **XSS**: all user- or model-controlled strings (filenames, synthesized
  answers, error messages) are HTML-escaped before being inserted into the
  DOM in the dashboard.
- **Secrets**: the Gemini API key lives only in `backend/config.json` (now
  git-ignored) or the `GEMINI_API_KEY` environment variable, and is never
  echoed back by `GET /api/config`.

## Known limitations

- The vector store records each chunk's **absolute** source file path. If you
  move the project to a different machine or directory, the next sync will
  detect every file as "changed" (path mismatch) and do a one-time full
  re-embed — this is self-healing (not a data-corruption bug), just an
  efficiency cost on first run in a new location.
- `streaming_mode` and `memory_persistence` toggles are stored in config but
  not yet wired into `/api/query` (answers are always returned as a single
  non-streamed response, and there's no conversation memory across queries).
