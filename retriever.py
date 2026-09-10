import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from chunker import CodeChunk
from embedder import DenseEmbedder
from vector_store import VectorStore
from bm25_retriever import BM25Indexer


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_indexer: BM25Indexer,
        chunks_metadata: List[Dict[str, Any]],
        manifest: Optional[Dict[str, Any]] = None,
        index_dir: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.bm25_indexer = bm25_indexer
        self.chunks_metadata = chunks_metadata
        self.manifest = manifest or {}
        self.index_dir = index_dir

    @classmethod
    def index_repository(
        cls,
        chunks: List[CodeChunk],
        output_dir: str,
        owner: str = "local",
        name: str = "repo",
        commit_sha: str = "latest",
    ) -> Dict[str, Any]:
        """
        Builds and saves the dual dense-sparse index alongside chunk metadata on disk.
        """
        os.makedirs(output_dir, exist_ok=True)

        if not chunks:
            # Handle empty chunks cleanly with an empty manifest
            manifest = {
                "owner": owner,
                "name": name,
                "commit_sha": commit_sha,
                "total_chunks": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path = os.path.join(output_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            return manifest

        corpus_texts = [c.content for c in chunks]

        # 1. Dense Embedding Generation & FAISS Index
        dense_vectors = DenseEmbedder.embed_texts(corpus_texts)
        vector_store = VectorStore(dimension=dense_vectors.shape[1])
        vector_ids = vector_store.add_vectors(dense_vectors)
        faiss_path = os.path.join(output_dir, "faiss_index.bin")
        vector_store.save_index(faiss_path)

        # 2. Sparse Lexical BM25 Index
        bm25_indexer = BM25Indexer()
        bm25_indexer.fit(corpus_texts)
        bm25_path = os.path.join(output_dir, "bm25_index.pkl")
        bm25_indexer.save(bm25_path)

        # 3. Canonical Chunk Metadata Records (1-to-1 mapped by vector_id)
        chunks_metadata: List[Dict[str, Any]] = []
        for v_id, chunk in zip(vector_ids, chunks):
            content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            meta_record = {
                "chunk_id": chunk.chunk_id,
                "vector_id": v_id,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "symbol_type": chunk.symbol_type,
                "symbol_name": chunk.symbol_name,
                "parent_symbol": None,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
                "content_hash": content_hash,
                "token_count": chunk.token_count,
            }
            chunks_metadata.append(meta_record)

        chunks_meta_path = os.path.join(output_dir, "chunks_metadata.json")
        with open(chunks_meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks_metadata, f, indent=2)

        # 4. Ingestion Manifest
        manifest = {
            "owner": owner,
            "name": name,
            "commit_sha": commit_sha,
            "total_chunks": len(chunks),
            "faiss_file": "faiss_index.bin",
            "bm25_file": "bm25_index.pkl",
            "metadata_file": "chunks_metadata.json",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    @classmethod
    def load_repository_index(cls, index_dir: str) -> "HybridRetriever":
        """
        Loads the FAISS index, BM25 model, and chunk metadata records from a disk directory.
        """
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"Index directory not found: {index_dir}")

        faiss_path = os.path.join(index_dir, "faiss_index.bin")
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")
        chunks_meta_path = os.path.join(index_dir, "chunks_metadata.json")
        manifest_path = os.path.join(index_dir, "manifest.json")

        vector_store = VectorStore()
        if os.path.exists(faiss_path):
            vector_store.load_index(faiss_path)

        bm25_indexer = BM25Indexer()
        if os.path.exists(bm25_path):
            bm25_indexer.load(bm25_path)

        chunks_metadata = []
        if os.path.exists(chunks_meta_path):
            with open(chunks_meta_path, "r", encoding="utf-8") as f:
                chunks_metadata = json.load(f)

        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        return cls(
            vector_store=vector_store,
            bm25_indexer=bm25_indexer,
            chunks_metadata=chunks_metadata,
            manifest=manifest,
            index_dir=index_dir,
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_limit: int = 20,
        sparse_limit: int = 20,
        symbol_boost_multiplier: float = 1.5,
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel dense semantic and sparse lexical searches, fuses results
        via Reciprocal Rank Fusion (RRF), and applies exact symbol matching boost.
        """
        if not self.chunks_metadata or not query:
            return []

        # 1. Dense Semantic Retrieval
        query_vec = DenseEmbedder.embed_query(query)
        dense_results = self.vector_store.search(query_vec, top_k=dense_limit)
        dense_ranks: Dict[int, int] = {
            vec_id: rank for rank, (vec_id, _) in enumerate(dense_results, start=1)
        }

        # 2. Sparse Lexical Retrieval
        sparse_results = self.bm25_indexer.search(query, top_k=sparse_limit)
        sparse_ranks: Dict[int, int] = {
            doc_id: rank for rank, (doc_id, _) in enumerate(sparse_results, start=1)
        }

        # Union of candidate IDs
        candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        if not candidate_ids:
            return []

        scored_candidates: List[Tuple[int, float, Optional[int], Optional[int]]] = []
        query_lower = query.lower()

        for doc_id in candidate_ids:
            if doc_id >= len(self.chunks_metadata):
                continue

            r_dense = dense_ranks.get(doc_id)
            r_sparse = sparse_ranks.get(doc_id)

            rrf_score = 0.0
            if r_dense is not None:
                rrf_score += 1.0 / (rrf_k + r_dense)
            if r_sparse is not None:
                rrf_score += 1.0 / (rrf_k + r_sparse)

            # Exact symbol match boost
            chunk_meta = self.chunks_metadata[doc_id]
            sym_name = chunk_meta.get("symbol_name", "").lower()
            if sym_name and sym_name not in ("<module>", "file_block", "section") and sym_name in query_lower:
                rrf_score *= symbol_boost_multiplier

            # Exact file path match boost
            file_path_raw = chunk_meta.get("file_path", "").lower()
            file_basename = os.path.basename(file_path_raw)
            if file_basename and (file_basename in query_lower or file_path_raw in query_lower):
                rrf_score *= 1.5

            # Root-level configuration priority boost
            if "/" not in file_path_raw and "\\" not in file_path_raw:
                if any(k in query_lower for k in ["pyproject", "setup", "license", "readme", "version", "build", "dependencies"]):
                    rrf_score *= 1.35

            scored_candidates.append((doc_id, rrf_score, r_dense, r_sparse))

        # Sort by fused score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for doc_id, score, r_dense, r_sparse in scored_candidates[:top_k]:
            meta = dict(self.chunks_metadata[doc_id])
            meta["rrf_score"] = score
            meta["dense_rank"] = r_dense
            meta["sparse_rank"] = r_sparse
            results.append(meta)

        return results

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Alias for hybrid_search."""
        return self.hybrid_search(query=query, top_k=top_k, **kwargs)

