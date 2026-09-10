from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class DenseEmbedder:
    _model: Optional[SentenceTransformer] = None

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
        """Loads and caches SentenceTransformer model on CPU."""
        if cls._model is None:
            cls._model = SentenceTransformer(model_name, device="cpu")
        return cls._model

    @classmethod
    def embed_texts(
        cls,
        texts: List[str],
        batch_size: int = 16,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_chars: int = 2000,
    ) -> np.ndarray:
        """
        Computes 384-dimensional L2-normalized dense embeddings for input texts.
        Returns a float32 numpy array of shape (len(texts), 384).
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        # Truncate text chunks to max_chars to avoid OOM
        sanitized_texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]

        model = cls.get_model(model_name)
        embeddings = model.encode(
            sanitized_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Built-in L2 unit normalization
        )

        embeddings_f32 = np.asarray(embeddings, dtype=np.float32)

        # Ensure unit L2 norm as safety guarantee
        norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normalized_embeddings = embeddings_f32 / norms

        return normalized_embeddings.astype(np.float32)

    @classmethod
    def embed_query(cls, query: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
        """Embeds a single query string and returns a 1D float32 numpy array of shape (384,)."""
        res = cls.embed_texts([query], model_name=model_name)
        return res[0]
