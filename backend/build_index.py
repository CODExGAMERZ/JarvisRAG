"""
CLI entry point for (re)building / syncing the vector store from a
knowledge-base folder.

Usage:
    python build_index.py
    python build_index.py --kb-dir ./my_knowledge_base --store-dir ./vector_store
    python build_index.py --embedding-mode ollama --ollama-model nomic-embed-text
"""

import argparse
import sys

from rag_pipeline import (
    DocumentIngestor,
    EmbeddingGenerator,
    VectorStorageEngine,
    logger,
    EMBED_BATCH_SIZE,
)

def detect_default_embedding_mode() -> str:
    try:
        import sentence_transformers  # noqa: F401
        return "local"
    except ImportError:
        return "ollama"

def main():
    parser = argparse.ArgumentParser(description="Build/sync the local RAG vector store.")
    parser.add_argument("--kb-dir", default="./my_knowledge_base", help="Knowledge base directory to ingest.")
    parser.add_argument("--store-dir", default="./vector_store", help="Where to persist the vector store.")
    parser.add_argument("--embedding-mode", choices=["local", "ollama"], default=None,
                         help="Embedding backend. Defaults to 'local' if sentence-transformers is installed, else 'ollama'.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="sentence-transformers model name.")
    parser.add_argument("--ollama-model", default="nomic-embed-text", help="Ollama embedding model name.")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/embeddings", help="Ollama embeddings endpoint.")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--no-faiss", action="store_true", help="Force the pure-NumPy vector engine instead of FAISS.")
    args = parser.parse_args()

    embedding_mode = args.embedding_mode or detect_default_embedding_mode()
    if embedding_mode == "ollama" and args.embedding_mode is None:
        logger.warning("sentence-transformers not found; defaulting to Ollama embeddings. "
                        "Make sure Ollama is running locally.")

    vector_store = VectorStorageEngine(store_dir=args.store_dir, use_faiss=not args.no_faiss)
    vector_store.load_index()

    try:
        embedding_gen = EmbeddingGenerator(
            mode=embedding_mode,
            model_name=args.embedding_model,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
        )
    except Exception as e:
        logger.error(f"Failed to initialize embedding generator: {e}")
        logger.error("Install sentence-transformers, or run Ollama locally and pass --embedding-mode ollama.")
        sys.exit(1)

    ingestor = DocumentIngestor()
    logger.info(f"Syncing knowledge base '{args.kb_dir}' -> vector store '{args.store_dir}'...")
    summary = ingestor.sync_directory(args.kb_dir, vector_store, embedding_gen, batch_size=args.batch_size)

    logger.info(
        "Sync complete: "
        f"{summary['ingested_files']} file(s) ingested, "
        f"{summary['added_chunks']} chunk(s) added, "
        f"{summary['removed_files']} file(s) purged, "
        f"{summary['failed_files']} file(s) failed."
    )
    logger.info(f"Vector store now holds {len(vector_store.chunks)} chunk(s).")

if __name__ == "__main__":
    main()
