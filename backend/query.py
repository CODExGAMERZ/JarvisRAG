"""
CLI for querying an already-built vector store.

Usage:
    python query.py                              # interactive mode
    python query.py -q "What temperature does X operate at?"
    python query.py -q "..." --k 6 --threshold 0.5
"""

import os
import sys
import argparse
import logging
from typing import Optional

from rag_pipeline import (
    VectorStorageEngine,
    EmbeddingGenerator,
    GeminiRAGOrchestrator,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
)

logging.basicConfig(level=logging.WARNING)

def detect_default_embedding_mode() -> str:
    try:
        import sentence_transformers  # noqa: F401
        return "local"
    except ImportError:
        return "ollama"

def print_result(result: dict):
    print("\n" + "-" * 60)
    print(f"ANSWER:\n{result['answer']}")
    print("-" * 60)
    print("Retrieved Context Chunks:")
    if not result["retrieved_contexts"]:
        print(" - [No context chunks met the similarity threshold]")
    else:
        for chunk, sim in result["retrieved_contexts"]:
            file_name = os.path.basename(chunk["source_file_path"])
            print(f" - {file_name} (Page {chunk['page_number']}) | Similarity: {sim:.4f}")
    print("=" * 60 + "\n")

def build_orchestrator(store_dir: str, embedding_mode: Optional[str]):
    vector_store = VectorStorageEngine(store_dir=store_dir, use_faiss=True)
    if not vector_store.load_index():
        print(f"Error: no vector store found at '{store_dir}'. Run 'python build_index.py' first.")
        sys.exit(1)

    mode = embedding_mode or detect_default_embedding_mode()
    if mode == "ollama" and embedding_mode is None:
        print("sentence-transformers not found; falling back to Ollama embeddings.")

    try:
        embedding_gen = EmbeddingGenerator(mode=mode)
    except Exception as e:
        print(f"Error loading embedding generator: {e}")
        sys.exit(1)

    return GeminiRAGOrchestrator(vector_store=vector_store, embedding_generator=embedding_gen)

def main():
    parser = argparse.ArgumentParser(description="Query the local RAG knowledge base.")
    parser.add_argument("-q", "--query", default=None, help="One-shot query. Omit for interactive mode.")
    parser.add_argument("--store-dir", default="./vector_store", help="Vector store directory.")
    parser.add_argument("--embedding-mode", choices=["local", "ollama"], default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum cosine similarity.")
    args = parser.parse_args()

    orchestrator = build_orchestrator(args.store_dir, args.embedding_mode)

    if args.query:
        result = orchestrator.search_and_synthesize(args.query, k=args.k, threshold=args.threshold)
        print_result(result)
        return

    print("\n" + "=" * 60)
    print("Local Knowledge Base RAG - interactive query")
    print("Type your question and press Enter. (Type 'exit' to quit)")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Query > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            result = orchestrator.search_and_synthesize(query, k=args.k, threshold=args.threshold)
            print_result(result)
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error during query processing: {e}\n")

if __name__ == "__main__":
    main()
