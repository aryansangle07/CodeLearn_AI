import os
import tempfile
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from db import Base, get_db, Repository, IngestionStatus, RunnabilityStatus
from chunker import CodeChunk
from retriever import HybridRetriever

# In-memory SQLite for FastAPI tests
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["app"] == "CodeLearn AI"


def test_submit_invalid_github_url():
    response = client.post(
        "/api/repositories",
        json={"github_url": "https://gitlab.com/invalid/repo", "branch": "main"},
    )
    assert response.status_code == 400
    assert "Invalid GitHub URL" in response.json()["detail"]


def test_submit_valid_repository_and_status():
    response = client.post(
        "/api/repositories",
        json={"github_url": "https://github.com/fastapi/fastapi", "branch": "main"},
    )
    assert response.status_code == 202
    data = response.json()
    assert "repository_id" in data
    repo_id = data["repository_id"]

    # Poll status endpoint
    status_resp = client.get(f"/api/repositories/{repo_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["repository_id"] == repo_id
    assert status_data["progress_percentage"] >= 0


def test_query_nonexistent_repository():
    response = client.post(
        "/api/query",
        json={"repository_id": str(uuid.uuid4()), "query": "Where is the router defined?"},
    )
    assert response.status_code == 404


def test_query_indexed_repository():
    # Seed an indexed repository and its dual-index directory
    session = TestingSessionLocal()
    repo_id = str(uuid.uuid4())

    tmp_dir = tempfile.mkdtemp(prefix="codelearn_test_api_")
    test_chunks = [
        CodeChunk(
            chunk_id="app.py#create_app",
            file_path="src/app.py",
            symbol_name="create_app",
            symbol_type="function",
            start_line=1,
            end_line=10,
            content="def create_app():\n    # initializes application\n    return FastAPI()",
            token_count=15,
            language="Python",
        )
    ]
    HybridRetriever.index_repository(test_chunks, tmp_dir, owner="test", name="api_repo")

    repo = Repository(
        id=repo_id,
        url="https://github.com/test/api_repo",
        owner="test",
        name="api_repo",
        branch="main",
        commit_sha="commit12345",
        status=IngestionStatus.INDEXED,
        index_storage_path=tmp_dir,
        total_files=1,
        total_lines=10,
        total_size_bytes=100,
        language_distribution={"Python": 100.0},
        runnability_score=RunnabilityStatus.RUNNABLE,
        health_details={
            "has_readme": True,
            "has_dependency_manifest": True,
            "dependency_files": ["requirements.txt"],
            "has_entry_point": True,
            "entry_point_files": ["src/app.py"],
            "empty_implementations_count": 0,
            "todo_comment_count": 0,
            "summary_assessment": "Fully runnable repository.",
        },
    )
    session.add(repo)
    session.commit()
    session.close()

    # Query summary
    summary_resp = client.get(f"/api/repositories/{repo_id}/summary")
    assert summary_resp.status_code == 200
    summary_data = summary_resp.json()
    assert summary_data["owner"] == "test"
    assert summary_data["runnability_score"] == "RUNNABLE"

    # Query Q&A
    query_resp = client.post(
        "/api/query",
        json={"repository_id": repo_id, "query": "Where is create_app defined?"},
    )
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert "answer" in query_data
    assert "src/app.py" in query_data["citations"]
    assert query_data["is_grounded"] is True


def test_delete_repository_success():
    session = TestingSessionLocal()
    repo_id = str(uuid.uuid4())
    tmp_storage = tempfile.mkdtemp(prefix="codelearn_del_test_")

    repo = Repository(
        id=repo_id,
        url="https://github.com/delete-test/repo",
        owner="delete-test",
        name="repo",
        branch="main",
        commit_sha="abcdef123456",
        status=IngestionStatus.INDEXED,
        index_storage_path=tmp_storage,
        runnability_score=RunnabilityStatus.RUNNABLE,
    )
    session.add(repo)
    session.commit()
    session.close()

    assert os.path.exists(tmp_storage)

    # Perform DELETE
    del_resp = client.delete(f"/api/repositories/{repo_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Verify storage removed
    assert not os.path.exists(tmp_storage)

    # Verify DB record removed
    session = TestingSessionLocal()
    deleted_check = session.query(Repository).filter(Repository.id == repo_id).first()
    session.close()
    assert deleted_check is None


def test_delete_repository_mid_ingestion_blocked():
    session = TestingSessionLocal()
    repo_id = str(uuid.uuid4())

    repo = Repository(
        id=repo_id,
        url="https://github.com/active-test/repo",
        owner="active-test",
        name="repo",
        branch="main",
        commit_sha="1234567890ab",
        status=IngestionStatus.PARSING,
        index_storage_path="",
        runnability_score=RunnabilityStatus.ANALYSIS_ONLY,
    )
    session.add(repo)
    session.commit()
    session.close()

    # Attempt DELETE while mid-ingestion
    del_resp = client.delete(f"/api/repositories/{repo_id}")
    assert del_resp.status_code == 409
    assert "ingestion is actively in progress" in del_resp.json()["detail"]

    # Verify repo is still intact in DB
    session = TestingSessionLocal()
    intact_check = session.query(Repository).filter(Repository.id == repo_id).first()
    session.delete(intact_check)
    session.commit()
    session.close()


def test_delete_nonexistent_repository():
    del_resp = client.delete(f"/api/repositories/{uuid.uuid4()}")
    assert del_resp.status_code == 404
