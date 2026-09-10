import os
import tempfile
from bm25_retriever import BM25Indexer


def test_bm25_tokenization():
    sample_code = "def calculateTotalAmount(user_id, item_price): return user_id + item_price"
    tokens = BM25Indexer.tokenize(sample_code)

    assert "def" in tokens
    assert "calculatetotalamount" in tokens
    assert "calculate" in tokens
    assert "total" in tokens
    assert "amount" in tokens
    assert "user_id" in tokens
    assert "user" in tokens
    assert "id" in tokens


def test_bm25_search_exact_keyword():
    corpus = [
        "def compute_fibonacci(n): if n <= 1: return n; return compute_fibonacci(n-1)",
        "class JWTAuthenticationService: def verify_token(token): pass",
        "def render_dashboard_chart(data, title): plt.plot(data)",
    ]

    indexer = BM25Indexer()
    indexer.fit(corpus)

    # Search for Fibonacci
    results = indexer.search("compute_fibonacci", top_k=2)
    assert len(results) >= 1
    assert results[0][0] == 0  # Doc 0 must rank first
    assert results[0][1] > 0.0

    # Search for JWT authentication
    jwt_results = indexer.search("JWTAuthenticationService verify_token", top_k=2)
    assert jwt_results[0][0] == 1  # Doc 1 must rank first


def test_bm25_persistence():
    corpus = [
        "SELECT * FROM users WHERE active = 1",
        "import torch.nn as nn; class Transformer(nn.Module): pass",
        "function handleHttpRequest(req, res) { res.send('OK'); }",
    ]
    indexer = BM25Indexer()
    indexer.fit(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = os.path.join(tmpdir, "bm25_index.pkl")
        indexer.save(pkl_path)
        assert os.path.exists(pkl_path)

        loaded_indexer = BM25Indexer()
        loaded_indexer.load(pkl_path)
        assert loaded_indexer.corpus_size == 3

        res = loaded_indexer.search("Transformer Module", top_k=1)
        assert len(res) >= 1
        assert res[0][0] == 1
