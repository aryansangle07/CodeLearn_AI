import os
from typing import List, Optional, Tuple
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        """
        Adds normalized dense vectors to the FAISS index.
        Returns the assigned 0-indexed vector IDs.
        """
        if vectors is None or len(vectors) == 0:
            return []

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors of shape (N, {self.dimension}), got shape {vectors.shape}"
            )

        start_id = self.total_vectors
        vecs_contiguous = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vecs_contiguous)
        end_id = self.total_vectors

        return list(range(start_id, end_id))

    def save_index(self, file_path: str) -> None:
        """Serializes the FAISS index to a binary file on disk."""
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path: str) -> None:
        """Loads a FAISS index from a binary file on disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FAISS index file not found at: {file_path}")
        self.index = faiss.read_index(file_path)
        self.dimension = self.index.d

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Executes a Top-K nearest neighbor search against the indexed vectors.
        Returns a list of (vector_id, similarity_score) tuples sorted descending by score.
        """
        if self.total_vectors == 0:
            return []

        if query_vector.ndim == 1:
            q_vec = np.expand_dims(query_vector, axis=0)
        else:
            q_vec = query_vector

        if q_vec.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension {q_vec.shape[1]} does not match index dimension {self.dimension}"
            )

        # Ensure unit L2 normalization
        norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        norm[norm == 0] = 1e-12
        q_vec = q_vec / norm

        q_contiguous = np.ascontiguousarray(q_vec, dtype=np.float32)
        actual_k = min(top_k, self.total_vectors)

        distances, indices = self.index.search(q_contiguous, actual_k)

        results: List[Tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((int(idx), float(dist)))

        return results
