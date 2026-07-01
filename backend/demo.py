"""
Self-contained demo of the RAG pipeline.

Writes a couple of mock files into ./demo_knowledge_base and builds a
throwaway vector store at ./demo_vector_store, so it never touches a real
knowledge base or vector store you may already have.

Usage:
    python demo.py
"""

import os
import shutil

from rag_pipeline import (
    DocumentIngestor,
    EmbeddingGenerator,
    VectorStorageEngine,
    GeminiRAGOrchestrator,
    logger,
)

DEMO_KB_DIR = "./demo_knowledge_base"
DEMO_STORE_DIR = "./demo_vector_store"

def seed_demo_files():
    os.makedirs(DEMO_KB_DIR, exist_ok=True)

    with open(os.path.join(DEMO_KB_DIR, "quantum_system.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Quantum System Specifications\n\n"
            "The experimental stabilizer system uses a core fluid consisting of highly "
            "pressurized Helium-3. The system operates at a critical temperature threshold "
            "of exactly 1.8 Millikelvin. Running the stabilizer above 2.1 Millikelvin triggers "
            "an automatic thermodynamic shutdown sequence to prevent vacuum containment leakage.\n\n"
            "## Safety Protocols\n\n"
            "If thermal runout is detected, standard recovery protocols require injecting "
            "liquid argon into the heat shield within 12 seconds."
        )

    with open(os.path.join(DEMO_KB_DIR, "network_topology.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Security network topology documentation.\n\n"
            "The backend administration node for the sandbox runs exclusively on subnet "
            "10.144.9.0/24. The database server sits at IP address 10.144.9.89 and operates "
            "on non-standard port 9876. All incoming requests outside this range are silently "
            "blackholed at the router."
        )

def main():
    seed_demo_files()
    logger.info(f"Demo knowledge base seeded at '{DEMO_KB_DIR}'.")

    embedding_mode = "local"
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.warning("sentence-transformers not found; falling back to Ollama (ensure it's running).")
        embedding_mode = "ollama"

    vector_store = VectorStorageEngine(store_dir=DEMO_STORE_DIR, use_faiss=True)
    vector_store.load_index()

    try:
        embedding_gen = EmbeddingGenerator(mode=embedding_mode)
    except Exception as e:
        logger.error(f"Failed to initialize embedding generator: {e}")
        return

    ingestor = DocumentIngestor()
    ingestor.sync_directory(DEMO_KB_DIR, vector_store, embedding_gen)

    orchestrator = GeminiRAGOrchestrator(vector_store=vector_store, embedding_generator=embedding_gen)

    print("\n" + "=" * 50)
    print("DEMO: RAG RETRIEVAL & SYNTHESIS")
    print("=" * 50)

    query_a = "What is the critical operating temperature of the Helium-3 stabilizer?"
    result_a = orchestrator.search_and_synthesize(query_a, threshold=0.55)
    print(f"\n[QUERY A]: {query_a}")
    print(f"[ANSWER]: {result_a['answer']}")
    print("Retrieved sources:")
    for chunk, sim in result_a["retrieved_contexts"]:
        print(f" - {os.path.basename(chunk['source_file_path'])} (Page {chunk['page_number']}) | Similarity: {sim:.4f}")

    query_b = "Who was the prime minister of the UK in 1995?"
    result_b = orchestrator.search_and_synthesize(query_b, threshold=0.55)
    print(f"\n[QUERY B - unanswerable from context]: {query_b}")
    print(f"[ANSWER]: {result_b['answer']}")
    print("=" * 50)
    print(f"\nDemo artifacts written to '{DEMO_KB_DIR}' and '{DEMO_STORE_DIR}' — delete them any time.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
