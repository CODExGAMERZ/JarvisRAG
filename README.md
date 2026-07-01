# jarvisRAG — Local Knowledge Base RAG System

A local-first Retrieval-Augmented Generation application: a FastAPI backend that ingests `.txt` / `.md` / `.pdf` files into a FAISS vector store, retrieves relevant chunks with local embeddings, and synthesizes grounded answers with Gemini. Features a stunning, futuristic browser dashboard to run queries, manage files, and trace performance metrics in real-time.

---

## Key Features

- **Inline RAG Console (`/command.html`)**: Submit questions directly from Central Command. The main panel toggles between the listening core orb and a holographic terminal displaying the answers inline.
- **Rich LaTeX Math & Markdown Rendering**: Formulas (inline and display LaTeX blocks like `\[...\]` and `\(...\)`) and markdown text are rendered with MathJax and Marked.js.
- **Tailwind Typography styling**: RAG answers and source contexts are styled using Tailwind's `prose-invert` and `prose-cyan` typography theme.
- **Retrieval Stream Dashboard (`/stream.html`)**: Watch real-time RAG operations, search the index, and analyze retrieved source chunks with cosine similarity telemetry.
- **Knowledge Base Manager (`/knowledge.html`)**: Drag-and-drop file ingestion, document raw content previewer, and asynchronous background database synchronization.
- **Pointer-Events Navigation Fix**: Navigation overlay issue resolved to enable click-through access to all panels.

---

## Project Layout

```
.
├── main.py                  # FastAPI app: serves the frontend + REST API
├── requirements.txt          # Full app dependencies (web server + RAG pipeline)
├── frontend/                 # Static dashboard (command / stream / knowledge / config)
└── backend/
    ├── rag_pipeline.py       # Core library: ingest, embed, store, retrieve, synthesize
    ├── build_index.py        # CLI: (re)build the vector store
    ├── query.py              # CLI: ask questions from the terminal
    ├── demo.py               # Self-contained demo (isolated sandbox data)
    ├── test_rag.py           # Unit tests
    ├── my_knowledge_base/    # Your source documents (.txt / .md / .pdf)
    ├── vector_store/         # Generated FAISS index + metadata (git-ignored)
    └── config.example.json   # Template — copy to config.json if you want a starting file
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Local embeddings use `sentence-transformers` (CPU) by default. If it isn't installed, the app automatically falls back to a local [Ollama](https://ollama.com) endpoint (`nomic-embed-text` by default) — make sure Ollama is running if you go that route.

---

## Running the App

```bash
python main.py
```

This starts the FastAPI server at **http://127.0.0.1:8000** and serves the dashboard at `/command.html` (root `/` redirects there). Set your Gemini API key from the **Config** page in the UI, or export it before starting:

```bash
export GEMINI_API_KEY=your_key_here   # optional — retrieval still works without it
python main.py
```

Without a key, `/api/query` still runs retrieval and returns the matched context chunks, but skips the Gemini synthesis step (`status: "SYNTHESIS_SKIPPED"`).

---

## Dashboard Pages

| Page | Purpose |
|---|---|
| `/command.html` | Central Command: Live vitals, sync task queue, and inline RAG query terminal (with MathJax and Markdown). |
| `/stream.html` | Retrieval Stream: Search index and analyze the RAG output and matching chunks side-by-side. |
| `/knowledge.html`| Knowledge base manager: Upload, delete, and view raw files. |
| `/config.html` | Config panel: Settings, API keys, live logger, and CLI administrative tools. |

---

## REST API

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

---

## CLI Tools (backend/)

You can also drive the pipeline entirely from the terminal, without the web app:

```bash
cd backend
python build_index.py                          # (re)build the vector store
python query.py -q "What temperature does X operate at?"
python demo.py                                 # sandboxed end-to-end demo
python -m unittest test_rag.py -v              # unit tests
```

See `backend/README.md` for CLI details and the pipeline's design notes.

---

## Security Notes

- **Prompt Injection**: retrieved document text is treated as untrusted data in the Gemini system prompt (documents can't override instructions), and literal `</retrieved_context>`-style breakout sequences are neutralized before being sent to the model.
- **Path Traversal**: upload/delete/view endpoints resolve filenames to a bare basename and verify the result stays inside `backend/my_knowledge_base` before touching disk.
- **XSS**: all user- or model-controlled strings (filenames, synthesized answers, error messages) are parsed cleanly with Marked.js and MathJax or escaped safely to prevent DOM-injection vulnerabilities.
- **Secrets**: the Gemini API key lives only in `backend/config.json` (git-ignored) or the `GEMINI_API_KEY` environment variable, and is never echoed back by `GET /api/config`.

## Known limitations

- The vector store records each chunk's **absolute** source file path. If you
  move the project to a different machine or directory, the next sync will
  detect every file as "changed" (path mismatch) and do a one-time full
  re-embed — this is self-healing (not a data-corruption bug), just an
  efficiency cost on first run in a new location.
- `streaming_mode` and `memory_persistence` toggles are stored in config but
  not yet wired into `/api/query` (answers are always returned as a single
  non-streamed response, and there's no conversation memory across queries).
