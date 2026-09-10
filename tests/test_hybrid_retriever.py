import os
import tempfile
from chunker import CodeChunk
from retriever import HybridRetriever


def test_hybrid_retriever_dual_index_lifecycle():
    chunks = [
        CodeChunk(
            chunk_id="auth.py#login_1_10",
            file_path="src/auth.py",
            symbol_name="login_user",
            symbol_type="function",
            start_line=1,
            end_line=10,
            content="def login_user(username, password):\n    # authenticates user\n    return authenticate(username, password)",
            token_count=20,
            language="Python",
        ),
        CodeChunk(
            chunk_id="db.py#query_1_8",
            file_path="src/db.py",
            symbol_name="execute_sql_query",
            symbol_type="function",
            start_line=1,
            end_line=8,
            content="def execute_sql_query(query_str):\n    # runs sql\n    return db.session.execute(query_str)",
            token_count=18,
            language="Python",
        ),
        CodeChunk(
            chunk_id="models.py#User_1_15",
            file_path="src/models.py",
            symbol_name="User",
            symbol_type="class",
            start_line=1,
            end_line=15,
            content="class User:\n    id: int\n    username: str\n    email: str",
            token_count=15,
            language="Python",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build index
        manifest = HybridRetriever.index_repository(chunks, tmpdir, owner="test_owner", name="test_repo", commit_sha="abc1234")
        assert manifest["total_chunks"] == 3
        assert os.path.exists(os.path.join(tmpdir, "faiss_index.bin"))
        assert os.path.exists(os.path.join(tmpdir, "bm25_index.pkl"))
        assert os.path.exists(os.path.join(tmpdir, "chunks_metadata.json"))
        assert os.path.exists(os.path.join(tmpdir, "manifest.json"))

        # Load index
        retriever = HybridRetriever.load_repository_index(tmpdir)
        assert len(retriever.chunks_metadata) == 3

        # Search for authentication query
        results = retriever.search("how does login and user authentication work", top_k=2)
        assert len(results) == 2
        top_match = results[0]
        assert top_match["file_path"] in ("src/auth.py", "src/models.py")
        assert "rrf_score" in top_match
        assert top_match["rrf_score"] > 0


def test_hybrid_retriever_exact_symbol_boost():
    chunks = [
        CodeChunk(
            chunk_id="math.py#calc_1_5",
            file_path="math.py",
            symbol_name="calculate_standard_deviation",
            symbol_type="function",
            start_line=1,
            end_line=5,
            content="def calculate_standard_deviation(values): return np.std(values)",
            token_count=12,
            language="Python",
        ),
        CodeChunk(
            chunk_id="stats.py#mean_1_5",
            file_path="stats.py",
            symbol_name="compute_average_mean",
            symbol_type="function",
            start_line=1,
            end_line=5,
            content="def compute_average_mean(values): return np.mean(values)",
            token_count=12,
            language="Python",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        HybridRetriever.index_repository(chunks, tmpdir)
        retriever = HybridRetriever.load_repository_index(tmpdir)

        # Exact symbol query
        results = retriever.search("where is calculate_standard_deviation defined?", top_k=1)
        assert len(results) == 1
        assert results[0]["symbol_name"] == "calculate_standard_deviation"
