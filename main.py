import os
import sys
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque
from contextlib import asynccontextmanager

import psutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

log_buffer = deque(maxlen=500)

class RingBufferLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            log_buffer.append(log_entry)
        except Exception:
            self.handleError(record)

logger_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_handler = RingBufferLogHandler()
log_handler.setFormatter(logger_formatter)
logging.getLogger().addHandler(log_handler)
logging.getLogger("KnowledgeBaseRAG").addHandler(log_handler)

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
KB_DIR = BACKEND_DIR / "my_knowledge_base"
STORE_DIR = BACKEND_DIR / "vector_store"
CONFIG_FILE = BACKEND_DIR / "config.json"

KB_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
from rag_pipeline import DocumentIngestor, EmbeddingGenerator, VectorStorageEngine, GeminiRAGOrchestrator

DEFAULT_CONFIG = {
    "model_name": "gemini-2.5-flash",
    "temperature": 0.0,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 2048,
    "streaming_mode": False,
    "memory_persistence": False
}

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_config(config_data: dict):
    tmp_file = CONFIG_FILE.with_suffix(".json.tmp")
    with open(tmp_file, "w") as f:
        json.dump(config_data, f, indent=2)
    os.replace(str(tmp_file), str(CONFIG_FILE))

index_lock = asyncio.Lock()

sync_job = {
    "status": "IDLE", # IDLE, RUNNING, SUCCESS, FAILED
    "started_at": 0.0,
    "duration": 0.0,
    "result": None,
    "error": None
}

query_latency_history = deque(maxlen=20)

def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Initializing embedding generator...")
    try:
        try:
            import sentence_transformers
            embedding_mode = "local"
        except ImportError:
            embedding_mode = "ollama"
        
        app.state.embedding_generator = EmbeddingGenerator(mode=embedding_mode)
        logging.info(f"Embedding generator initialized in '{embedding_mode}' mode.")
    except Exception as e:
        logging.error(f"Failed to initialize embedding generator: {e}")
        app.state.embedding_generator = None
        
    logging.info("Initializing vector storage engine...")
    app.state.vector_store = VectorStorageEngine(store_dir=str(STORE_DIR), use_faiss=True)
    app.state.vector_store.load_index()
    app.state.ingestor = DocumentIngestor()
    app.state.start_time = time.time()
    
    cfg = load_config()
    if cfg.get("gemini_api_key"):
        os.environ["GEMINI_API_KEY"] = cfg["gemini_api_key"]
        logging.info("Initialized GEMINI_API_KEY from config.json")
    
    psutil.cpu_percent(interval=None)
        
    yield
    logging.info("Shutting down API server...")

app = FastAPI(title="JarvisRAG RAG System API", lifespan=lifespan)

class ConfigUpdate(BaseModel):
    model_name: Optional[str] = Field(None, pattern="^(AETHER-1|GPT-4.CORE|gemini-2.5-flash|gemini-2.5-pro)$")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    streaming_mode: Optional[bool] = None
    memory_persistence: Optional[bool] = None
    gemini_api_key: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = Field(None, ge=1, le=20)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

@app.get("/api/config")
def get_api_config():
    cfg = load_config()
    api_key_set = bool(cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))
    return {
        "model_name": cfg.get("model_name"),
        "temperature": cfg.get("temperature"),
        "top_p": cfg.get("top_p"),
        "frequency_penalty": cfg.get("frequency_penalty"),
        "presence_penalty": cfg.get("presence_penalty"),
        "max_tokens": cfg.get("max_tokens"),
        "streaming_mode": cfg.get("streaming_mode"),
        "memory_persistence": cfg.get("memory_persistence"),
        "api_key_configured": api_key_set
    }

@app.post("/api/config")
def update_api_config(data: ConfigUpdate):
    cfg = load_config()
    update_dict = data.model_dump(exclude_unset=True)
    
    if "gemini_api_key" in update_dict:
        key = update_dict.pop("gemini_api_key")
        if key:
            cfg["gemini_api_key"] = key
            os.environ["GEMINI_API_KEY"] = key
            logging.info("GEMINI_API_KEY updated in environment variables.")
        elif key == "":
            cfg.pop("gemini_api_key", None)
            os.environ.pop("GEMINI_API_KEY", None)
            logging.info("GEMINI_API_KEY cleared from environment variables.")
    
    for k, v in update_dict.items():
        cfg[k] = v
        
    save_config(cfg)
    return {"status": "SUCCESS", "config": {k: v for k, v in cfg.items() if k != "gemini_api_key"}}

@app.get("/api/documents")
def list_documents():
    chunks_by_file = {}
    if app.state.vector_store and app.state.vector_store.chunks:
        for chunk in app.state.vector_store.chunks:
            path = os.path.abspath(chunk["source_file_path"])
            chunks_by_file[path] = chunks_by_file.get(path, 0) + 1
            
    files_list = []
    if KB_DIR.exists():
        for p in KB_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".txt", ".md", ".pdf"):
                abs_path = os.path.abspath(p)
                rel_path = str(p.relative_to(KB_DIR))
                size = p.stat().st_size
                
                current_checksum = app.state.ingestor._compute_file_checksum(str(p))
                stored_checksum = app.state.vector_store.processed_files.get(abs_path)
                
                if stored_checksum and stored_checksum == current_checksum:
                    status_str = "INDEXED"
                else:
                    status_str = "PENDING"
                    
                chunk_count = chunks_by_file.get(abs_path, 0)
                files_list.append({
                    "filename": p.name,
                    "relative_path": rel_path.replace("\\", "/"),
                    "size_bytes": size,
                    "size_formatted": format_size(size),
                    "chunks": chunk_count,
                    "status": status_str
                })
    return files_list

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), overwrite: bool = Form(False)):
    MAX_SIZE = 25 * 1024 * 1024 # 25 MB
    
    filename = os.path.basename(file.filename)
    if not filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".txt", ".md", ".pdf"):
        raise HTTPException(status_code=400, detail="Unsupported file format. Only .txt, .md, .pdf allowed.")
        
    dest_path = KB_DIR / filename
    if dest_path.exists() and not overwrite:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="File already exists.")
        
    async with index_lock:
        try:
            total_size = 0
            with open(dest_path, "wb") as f:
                while chunk := await file.read(65536):
                    total_size += len(chunk)
                    if total_size > MAX_SIZE:
                        f.close()
                        dest_path.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail="File exceeds 25 MB size limit.")
                    f.write(chunk)
        except Exception as e:
            dest_path.unlink(missing_ok=True)
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
            
    return {"status": "SUCCESS", "filename": filename, "size_bytes": total_size, "size_formatted": format_size(total_size)}

@app.get("/api/documents/{filename}/raw")
def get_document_raw(filename: str):
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    target_path = (KB_DIR / safe_name).resolve()
    if KB_DIR.resolve() not in target_path.parents or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found in knowledge base.")
        
    ext = target_path.suffix.lower()
    media_types = {".txt": "text/plain", ".md": "text/markdown", ".pdf": "application/pdf"}
    media_type = media_types.get(ext, "application/octet-stream")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(target_path),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'}
    )

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    filename = os.path.basename(filename)
    if not filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    dest_path = KB_DIR / filename
    if not dest_path.exists():
        raise HTTPException(status_code=404, detail="File not found in knowledge base.")
        
    async with index_lock:
        try:
            dest_path.unlink()
            abs_path = os.path.abspath(dest_path)
            app.state.vector_store.remove_files([abs_path])
            app.state.vector_store.save_index()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
            
    return {"status": "SUCCESS", "filename": filename}

def run_sync_background(ingestor, kb_dir, store, embed_gen):
    global sync_job
    sync_job["status"] = "RUNNING"
    sync_job["started_at"] = time.time()
    sync_job["error"] = None
    sync_job["result"] = None
    
    try:
        logging.info("Starting background knowledge base synchronization...")
        summary = ingestor.sync_directory(str(kb_dir), store, embed_gen)
        store.save_index()
        sync_job["status"] = "SUCCESS"
        sync_job["result"] = summary
        logging.info(f"Sync complete: {summary}")
    except Exception as e:
        logging.error(f"Sync failed: {e}")
        sync_job["status"] = "FAILED"
        sync_job["error"] = str(e)
    finally:
        sync_job["duration"] = time.time() - sync_job["started_at"]

@app.post("/api/documents/sync")
async def sync_documents(background_tasks: BackgroundTasks):
    global sync_job
    async with index_lock:
        if sync_job["status"] == "RUNNING":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Sync already in progress.")
        
        sync_job["status"] = "RUNNING"
        background_tasks.add_task(
            run_sync_background,
            app.state.ingestor,
            KB_DIR,
            app.state.vector_store,
            app.state.embedding_generator
        )
    return {"status": "RUNNING", "message": "Sync job triggered."}

@app.get("/api/documents/sync/status")
def get_sync_status():
    return sync_job

@app.post("/api/query")
async def execute_query(req: QueryRequest):
    cfg = load_config()
    api_key = cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    
    model_map = {
        "AETHER-1": "gemini-2.5-flash",
        "GPT-4.CORE": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro"
    }
    
    raw_model_name = cfg.get("model_name", "gemini-2.5-flash")
    mapped_model = model_map.get(raw_model_name, "gemini-2.5-flash")
    
    k_val = req.k if req.k is not None else cfg.get("k", 4)
    threshold_val = req.threshold if req.threshold is not None else cfg.get("threshold", 0.55)
    
    if app.state.embedding_generator is None:
        raise HTTPException(status_code=500, detail="Embedding generator not initialized.")
        
    def search_task():
        orchestrator = GeminiRAGOrchestrator(
            vector_store=app.state.vector_store,
            embedding_generator=app.state.embedding_generator,
            api_key=api_key,
            model_name=mapped_model
        )
        
        start_time = time.time()
        res = orchestrator.search_and_synthesize(
            query=req.query,
            k=k_val,
            threshold=threshold_val,
            temperature=cfg.get("temperature"),
            top_p=cfg.get("top_p"),
            frequency_penalty=cfg.get("frequency_penalty"),
            presence_penalty=cfg.get("presence_penalty"),
            max_tokens=cfg.get("max_tokens")
        )
        duration = time.time() - start_time
        return res, duration
        
    try:
        res, duration_sec = await asyncio.to_thread(search_task)
        latency_ms = duration_sec * 1000
        query_latency_history.append(latency_ms)
        
        contexts = []
        for chunk, similarity in res.get("retrieved_contexts", []):
            contexts.append({
                "filename": os.path.basename(chunk["source_file_path"]),
                "page_number": chunk.get("page_number", 1),
                "text": chunk["text"],
                "similarity": similarity,
                "chunk_id": chunk.get("chunk_id", "N/A")[:8]
            })
            
        status_flag = "OK"
        if not api_key:
            status_flag = "SYNTHESIS_SKIPPED"
        elif res.get("answer") == "Insufficient local context to verify answer.":
            status_flag = "INSUFFICIENT_CONTEXT"
            
        return {
            "query": req.query,
            "answer": res.get("answer"),
            "status": status_flag,
            "retrieved_contexts": contexts,
            "latency_ms": round(latency_ms, 1),
            "model_used": mapped_model,
            "k": k_val,
            "threshold": threshold_val
        }
    except Exception as e:
        logging.error(f"Query API failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

@app.get("/api/system/vitals")
def get_system_vitals():
    cpu_usage = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mem_usage = mem.percent
    uptime = time.time() - app.state.start_time
    
    if query_latency_history:
        avg_latency = sum(query_latency_history) / len(query_latency_history)
    else:
        avg_latency = 12.0
        
    total_vectors = len(app.state.vector_store.chunks) if app.state.vector_store else 0
    total_files = len(os.listdir(KB_DIR)) if KB_DIR.exists() else 0
    
    embedding_dim = app.state.vector_store.embedding_dim if app.state.vector_store else None
    embedding_mode = getattr(app.state.embedding_generator, "mode", None)
    
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": mem_usage,
        "uptime_seconds": int(uptime),
        "latency_ms": round(avg_latency, 1),
        "memory_allocated_gb": round(mem.used / (1024**3), 2),
        "memory_total_gb": round(mem.total / (1024**3), 2),
        "total_vectors": total_vectors,
        "total_files": total_files,
        "embedding_dim": embedding_dim,
        "embedding_mode": embedding_mode
    }

@app.get("/api/logs")
def get_api_logs():
    return list(log_buffer)

@app.post("/api/config/flush")
def flush_logs():
    log_buffer.clear()
    logging.info("Logs ring buffer flushed.")
    return {"status": "SUCCESS", "message": "Log buffer cleared."}

@app.post("/api/config/reboot")
async def force_reboot():
    async with index_lock:
        try:
            logging.info("Force reloading vector index state...")
            app.state.vector_store.load_index()
            logging.info("Index reload complete.")
            return {"status": "SUCCESS", "message": "Vector index reloaded successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reboot reload failed: {str(e)}")

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/command.html")

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
