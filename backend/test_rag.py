import unittest
import os
import shutil
import numpy as np
from rag_pipeline import DocumentIngestor, VectorStorageEngine

class TestDocumentIngestor(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.abspath("./test_knowledge")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_semantic_chunking_basic(self):
        ingestor = DocumentIngestor()
        text = (
            "# Main Header\n\n"
            "This is paragraph one. It has multiple sentences to test splits. "
            "But it is relatively short, so it shouldn't be split on sentences.\n\n"
            "## Sub Header\n\n"
            "This is paragraph two. This is another sentence. "
            "And a third sentence in the sub-header section."
        )
        file_path = os.path.join(self.test_dir, "chunk_test.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        chunks = ingestor.parse_file(file_path)

        self.assertTrue(len(chunks) >= 2)
        for chunk in chunks:
            self.assertIn("chunk_id", chunk)
            self.assertEqual(chunk["source_file_path"], os.path.abspath(file_path))
            self.assertEqual(chunk["page_number"], 1)
            self.assertEqual(chunk["character_count"], len(chunk["text"]))
            self.assertIsNotNone(chunk["checksum"])

    def test_chunk_overlap_preserves_boundary_context(self):
        ingestor = DocumentIngestor(chunk_max_size=200, chunk_overlap=50)
        paragraph = "The quick brown fox jumps over the lazy dog. " * 20
        file_path = os.path.join(self.test_dir, "overlap_test.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(paragraph)

        chunks = ingestor.parse_file(file_path)
        self.assertGreaterEqual(len(chunks), 2)
        tail_of_first = chunks[0]["text"][-30:]
        self.assertIn(tail_of_first[-10:], chunks[1]["text"][:80])

    def test_oversized_single_line_is_split(self):
        ingestor = DocumentIngestor(chunk_max_size=100, chunk_overlap=0)
        huge_line = "x" * 500  # single line, no newlines, larger than chunk_max_size
        file_path = os.path.join(self.test_dir, "huge_line.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(huge_line)

        chunks = ingestor.parse_file(file_path)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk["character_count"], 100)

    def test_unsupported_extension_returns_empty(self):
        ingestor = DocumentIngestor()
        file_path = os.path.join(self.test_dir, "image.png")
        with open(file_path, "wb") as f:
            f.write(b"\x00\x01")
        self.assertEqual(ingestor.parse_file(file_path), [])

class TestVectorStorageEngine(unittest.TestCase):
    def setUp(self):
        self.store_dir = os.path.abspath("./test_vector_store")

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def _make_chunks_and_embeddings(self):
        chunks = [
            {"chunk_id": "1", "source_file_path": os.path.abspath("file1.txt"), "page_number": 1, "text": "Helium-3 temperature"},
            {"chunk_id": "2", "source_file_path": os.path.abspath("file2.txt"), "page_number": 1, "text": "UK Prime Minister"},
        ]
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return chunks, embeddings

    def test_numpy_mode_roundtrip(self):
        store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=False)
        self.assertFalse(store.use_faiss)
        self.assertFalse(store.load_index())

        chunks, embeddings = self._make_chunks_and_embeddings()
        store.add_vectors(chunks, embeddings)
        store.save_index()

        new_store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=False)
        self.assertTrue(new_store.load_index())
        self.assertEqual(len(new_store.chunks), 2)
        self.assertEqual(new_store.embeddings.shape, (2, 3))

        query = np.array([1.0, 0.1, 0.0], dtype=np.float32)
        results = new_store.similarity_search(query, k=1, threshold=0.70)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["chunk_id"], "1")
        self.assertGreaterEqual(results[0][1], 0.9)

    def test_faiss_mode_roundtrip(self):
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("FAISS is not installed in the environment.")

        store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=True)
        self.assertTrue(store.use_faiss)

        chunks, embeddings = self._make_chunks_and_embeddings()
        store.add_vectors(chunks, embeddings)
        store.save_index()

        new_store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=True)
        self.assertTrue(new_store.load_index())

        query = np.array([1.0, 0.1, 0.0], dtype=np.float32)
        results = new_store.similarity_search(query, k=1, threshold=0.70)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["chunk_id"], "1")

    def test_remove_files(self):
        store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=False)
        chunks, embeddings = self._make_chunks_and_embeddings()
        store.add_vectors(chunks, embeddings)
        store.processed_files = {chunks[0]["source_file_path"]: "abc", chunks[1]["source_file_path"]: "def"}

        store.remove_files([chunks[0]["source_file_path"]])
        self.assertEqual(len(store.chunks), 1)
        self.assertEqual(store.chunks[0]["chunk_id"], "2")
        self.assertEqual(store.embeddings.shape, (1, 3))
        np.testing.assert_array_almost_equal(store.embeddings[0], [0.0, 1.0, 0.0])
        self.assertNotIn(chunks[0]["source_file_path"], store.processed_files)

    def test_remove_files_pops_bookkeeping_even_with_zero_matching_chunks(self):
        """
        Regression test: a file that is tracked in processed_files but currently
        contributes zero chunks to the store (e.g. it previously failed to parse)
        must still have its processed_files entry cleared on removal, or it can
        never be re-ingested on a later sync.
        """
        store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=False)
        ghost_path = os.path.abspath("ghost_file.txt")
        store.processed_files = {ghost_path: "old_checksum"}

        store.remove_files([ghost_path])
        self.assertNotIn(ghost_path, store.processed_files)

    def test_add_vectors_dimension_mismatch_raises(self):
        store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=False)
        chunks, embeddings = self._make_chunks_and_embeddings()  # dim=3
        store.add_vectors(chunks, embeddings)

        bad_chunk = [{"chunk_id": "3", "source_file_path": os.path.abspath("file3.txt"), "page_number": 1, "text": "x"}]
        bad_embedding = np.array([[1.0, 0.0]], dtype=np.float32)  # dim=2, mismatched
        with self.assertRaises(ValueError):
            store.add_vectors(bad_chunk, bad_embedding)

    def test_similarity_search_k_larger_than_store(self):
        store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=False)
        chunks, embeddings = self._make_chunks_and_embeddings()
        store.add_vectors(chunks, embeddings)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.similarity_search(query, k=10, threshold=0.0)
        self.assertEqual(len(results), 2)

class TestSyncDirectory(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.abspath("./test_sync_kb")
        self.store_dir = os.path.abspath("./test_sync_store")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        for d in (self.test_dir, self.store_dir):
            if os.path.exists(d):
                shutil.rmtree(d)

    def test_modified_file_is_reingested(self):
        class FakeEmbeddingGenerator:
            def generate(self, texts, batch_size=32):
                return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

        file_path = os.path.join(self.test_dir, "note.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("original content")

        ingestor = DocumentIngestor()
        store = VectorStorageEngine(store_dir=self.store_dir, use_faiss=False)
        store.load_index()
        ingestor.sync_directory(self.test_dir, store, FakeEmbeddingGenerator())
        self.assertEqual(len(store.chunks), 1)
        original_chunk_id = store.chunks[0]["chunk_id"]

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("completely different content after edit")

        ingestor.sync_directory(self.test_dir, store, FakeEmbeddingGenerator())
        self.assertEqual(len(store.chunks), 1)
        self.assertNotEqual(store.chunks[0]["chunk_id"], original_chunk_id)
        self.assertEqual(store.chunks[0]["text"], "completely different content after edit")

if __name__ == "__main__":
    unittest.main()
