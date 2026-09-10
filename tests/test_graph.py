import os
import tempfile
import pytest
from chunker import CodeChunk
from retriever import HybridRetriever
from intent_analyzer import IntentAnalyzer
from schemas import IntentTypeEnum
from graph import build_codelearn_graph, execute_query_pipeline


def test_intent_classification():
    assert IntentAnalyzer.classify_intent("Where is UserRepository defined?") == IntentTypeEnum.SIMPLE_LOOKUP
    assert IntentAnalyzer.classify_intent("Give me an architecture overview of the project") == IntentTypeEnum.ARCHITECTURE
    assert IntentAnalyzer.classify_intent("How does token verification work in authentication?") == IntentTypeEnum.EXPLANATION


def test_graph_pipeline_execution():
    chunks = [
        CodeChunk(
            chunk_id="auth.py#jwt",
            file_path="src/auth.py",
            symbol_name="verify_jwt_token",
            symbol_type="function",
            start_line=1,
            end_line=10,
            content="def verify_jwt_token(token: str):\n    # verifies token validity\n    return decode(token)",
            token_count=18,
            language="Python",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        HybridRetriever.index_repository(chunks, tmp_dir)

        result_state = execute_query_pipeline(
            repository_id="repo-test-123",
            index_dir=tmp_dir,
            query="Where is verify_jwt_token located?",
        )

        assert result_state["intent_type"] == IntentTypeEnum.SIMPLE_LOOKUP
        assert len(result_state["retrieved_chunks"]) == 1
        assert "answer" in result_state
        assert len(result_state["answer"]) > 0
        assert "src/auth.py" in result_state["citations"]
        assert result_state["is_grounded"] is True
        assert result_state["execution_time_seconds"] > 0
