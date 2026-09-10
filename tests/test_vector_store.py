import os
import tempfile
import numpy as np
import pytest
from embedder import DenseEmbedder, EMBEDDING_DIMENSION
from vector_store import VectorStore


def test_embedder_normalization_and_dimension():
    texts = [
        "def add(a, b): return a + b",
        "class AuthenticationService: pass",
        "# README Documentation",
    ]
    vectors = DenseEmbedder.embed_texts(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, EMBEDDING_DIMENSION)
    assert vectors.dtype == np.float32

    # Verify unit L2 norm
    norms = np.linalg.norm(vectors, axis=1)
    for n in norms:
        assert pytest.approx(n, 1e-4) == 1.0


def test_vector_store_crud_and_search():
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    assert store.total_vectors == 0

    # Create dummy unit vectors
    np.random.seed(42)
    raw_vecs = np.random.randn(5, EMBEDDING_DIMENSION).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    unit_vecs = raw_vecs / norms

    assigned_ids = store.add_vectors(unit_vecs)
    assert assigned_ids == [0, 1, 2, 3, 4]
    assert store.total_vectors == 5

    # Query with vector 0 -> should return vector 0 with score close to 1.0
    query_vec = unit_vecs[0]
    results = store.search(query_vec, top_k=3)

    assert len(results) == 3
    assert results[0][0] == 0
    assert pytest.approx(results[0][1], 1e-4) == 1.0


def test_vector_store_persistence():
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    raw_vecs = np.random.randn(4, EMBEDDING_DIMENSION).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    unit_vecs = raw_vecs / norms
    store.add_vectors(unit_vecs)

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "faiss_index.bin")
        store.save_index(index_path)
        assert os.path.exists(index_path)

        # Reload in new instance
        new_store = VectorStore(dimension=EMBEDDING_DIMENSION)
        new_store.load_index(index_path)
        assert new_store.total_vectors == 4

        # Search should match
        results = new_store.search(unit_vecs[2], top_k=1)
        assert results[0][0] == 2
