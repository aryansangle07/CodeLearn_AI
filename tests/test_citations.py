import pytest
from citation_validator import CitationValidator


def test_citation_extraction():
    text = (
        "The model is defined in `src/models/user.py` and handled by `src/api/routes.py:L10-L25`.\n"
        "Config is in `config.yaml`."
    )
    citations = CitationValidator.extract_citations(text)
    assert "src/models/user.py" in citations
    assert "src/api/routes.py" in citations
    assert "config.yaml" in citations


def test_citation_validation_grounded():
    retrieved_chunks = [
        {"file_path": "src/models/user.py", "content": "class User: pass"},
        {"file_path": "src/api/routes.py", "content": "def get_users(): pass"},
    ]
    answer = "The user model is in `src/models/user.py`."

    val_answer, verified, is_grounded = CitationValidator.validate_citations(answer, retrieved_chunks)
    assert is_grounded is True
    assert verified == ["src/models/user.py"]
    assert val_answer == answer


def test_citation_validation_hallucinated():
    retrieved_chunks = [
        {"file_path": "src/models/user.py", "content": "class User: pass"},
    ]
    answer = "The user model is in `src/fake_service.py`."

    val_answer, verified, is_grounded = CitationValidator.validate_citations(answer, retrieved_chunks)
    assert is_grounded is False
    assert "insufficient to answer" in val_answer.lower()
