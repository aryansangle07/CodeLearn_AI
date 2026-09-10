import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from db import (
    IngestionStatus,
    RunnabilityStatus,
    Repository,
    get_engine,
    get_session_factory,
    get_db,
    init_db,
    check_db_health,
)
from schemas import (
    RepositoryRead,
    HealthDetailsDTO,
    IngestionStatusEnum,
    RunnabilityEnum,
)


@pytest.fixture
def sqlite_engine():
    """Provides an isolated in-memory SQLite database engine."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    return engine


@pytest.fixture
def db_session(sqlite_engine):
    """Provides a transactional database session over the in-memory SQLite database."""
    session_factory = get_session_factory(sqlite_engine)
    session = session_factory()
    yield session
    session.close()


def test_db_engine_creation():
    """Verify SQLite in-memory and PostgreSQL connection engine creates successfully."""
    sqlite_eng = get_engine("sqlite:///:memory:")
    assert sqlite_eng is not None

    pg_eng = get_engine("postgresql://user:pass@localhost:5432/test_db")
    assert pg_eng is not None
    assert pg_eng.pool.size() == 10


def test_init_db_creates_tables(sqlite_engine):
    """Verify init_db() creates the repositories table with expected columns and unique constraints."""
    inspector = inspect(sqlite_engine)
    tables = inspector.get_table_names()
    assert "repositories" in tables

    columns = {col["name"]: col for col in inspector.get_columns("repositories")}
    expected_columns = [
        "id",
        "url",
        "owner",
        "name",
        "branch",
        "commit_sha",
        "status",
        "error_message",
        "total_files",
        "total_lines",
        "total_size_bytes",
        "language_distribution",
        "runnability_score",
        "health_details",
        "index_storage_path",
        "created_at",
        "updated_at",
    ]
    for col_name in expected_columns:
        assert col_name in columns, f"Missing column: {col_name}"


def test_repository_crud_lifecycle(db_session):
    """Create a Repository row, query by ID, update status through lifecycle, and verify commit."""
    repo = Repository(
        id="repo_12345",
        url="https://github.com/fastapi/fastapi",
        owner="fastapi",
        name="fastapi",
        branch="main",
        commit_sha="a1b2c3d4e5f6789012345678901234567890abcd",
        status=IngestionStatus.PENDING,
        total_files=50,
        total_lines=12000,
        total_size_bytes=1048576,
        language_distribution={"Python": 95.5, "Dockerfile": 4.5},
        runnability_score=RunnabilityStatus.RUNNABLE,
        health_details={
            "has_readme": True,
            "has_dependency_manifest": True,
            "dependency_files": ["requirements.txt"],
            "has_entry_point": True,
            "entry_point_files": ["main.py"],
            "empty_implementations_count": 0,
            "todo_comment_count": 2,
            "summary_assessment": "High-quality runnable project",
        },
        index_storage_path="./data/indices/fastapi_fastapi_a1b2c3d4/",
    )
    db_session.add(repo)
    db_session.commit()

    # Query back
    retrieved = db_session.query(Repository).filter_by(id="repo_12345").first()
    assert retrieved is not None
    assert retrieved.name == "fastapi"
    assert retrieved.status == IngestionStatus.PENDING

    # Test status transitions: PENDING -> FETCHING -> PARSING -> EMBEDDING -> INDEXED
    transitions = [
        IngestionStatus.FETCHING,
        IngestionStatus.PARSING,
        IngestionStatus.EMBEDDING,
        IngestionStatus.INDEXED,
    ]
    for target_status in transitions:
        retrieved.status = target_status
        db_session.commit()
        db_session.refresh(retrieved)
        assert retrieved.status == target_status


def test_repository_unique_constraint(db_session):
    """Attempt to insert duplicate (owner, name, commit_sha) and assert IntegrityError is raised."""
    repo1 = Repository(
        id="repo_001",
        url="https://github.com/psf/requests",
        owner="psf",
        name="requests",
        branch="main",
        commit_sha="1111222233334444555566667777888899990000",
        status=IngestionStatus.PENDING,
        index_storage_path="./data/indices/repo_001",
    )
    db_session.add(repo1)
    db_session.commit()

    repo2 = Repository(
        id="repo_002",
        url="https://github.com/psf/requests",
        owner="psf",
        name="requests",
        branch="main",
        commit_sha="1111222233334444555566667777888899990000",  # Duplicate (owner, name, commit_sha)
        status=IngestionStatus.PENDING,
        index_storage_path="./data/indices/repo_002",
    )
    db_session.add(repo2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_invalid_status_enum_rejected():
    """Verify that assigning an invalid status string fails schema validation."""
    with pytest.raises(ValidationError):
        HealthDetailsDTO(summary_assessment="OK")
        RepositoryRead(
            id="test_id",
            url="https://github.com/test/repo",
            owner="test",
            name="repo",
            branch="main",
            commit_sha="1234567890123456789012345678901234567890",
            status="INVALID_STATUS_NAME",  # Invalid enum value
            total_files=0,
            total_lines=0,
            total_size_bytes=0,
            language_distribution={},
            runnability_score=RunnabilityEnum.ANALYSIS_ONLY,
            health_details=HealthDetailsDTO(),
            index_storage_path="./data/indices/test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )


def test_db_health_check_success(sqlite_engine):
    """Assert check_db_health(engine) returns True when database is active."""
    assert check_db_health(sqlite_engine) is True


def test_db_health_check_failure():
    """Assert check_db_health(bad_engine) returns False without throwing an unhandled exception."""
    bad_engine = get_engine("postgresql://invalid:invalid@localhost:9999/nonexistent")
    assert check_db_health(bad_engine) is False


def test_session_rollback_on_exception(sqlite_engine):
    """Verify session rolls back on error and allows subsequent valid operations."""
    session_gen = get_db(sqlite_engine)
    session = next(session_gen)

    # Insert a valid repository
    valid_repo = Repository(
        id="repo_valid",
        url="https://github.com/test/valid",
        owner="test",
        name="valid",
        branch="main",
        commit_sha="1234567890123456789012345678901234567890",
        status=IngestionStatus.PENDING,
        index_storage_path="./data/indices/valid",
    )
    session.add(valid_repo)
    session.commit()

    # Trigger a failure (duplicate primary key)
    dup_repo = Repository(
        id="repo_valid",
        url="https://github.com/test/valid",
        owner="test",
        name="valid",
        branch="main",
        commit_sha="different_sha_123456789012345678901234",
        status=IngestionStatus.PENDING,
        index_storage_path="./data/indices/valid2",
    )
    session.add(dup_repo)
    with pytest.raises(IntegrityError):
        session.commit()

    session.rollback()

    # Verify session is healthy after rollback for a new insert
    another_valid_repo = Repository(
        id="repo_valid_2",
        url="https://github.com/test/valid2",
        owner="test",
        name="valid2",
        branch="main",
        commit_sha="9999999990123456789012345678901234567890",
        status=IngestionStatus.PENDING,
        index_storage_path="./data/indices/valid2",
    )
    session.add(another_valid_repo)
    session.commit()

    count = session.query(Repository).count()
    assert count == 2


def test_pydantic_orm_serialization(db_session):
    """Validate that a SQLAlchemy Repository instance converts into RepositoryRead DTO with from_attributes=True."""
    repo = Repository(
        id="repo_serialize_test",
        url="https://github.com/django/django",
        owner="django",
        name="django",
        branch="main",
        commit_sha="abcdef1234567890abcdef1234567890abcdef12",
        status=IngestionStatus.INDEXED,
        error_message=None,
        total_files=100,
        total_lines=45000,
        total_size_bytes=5242880,
        language_distribution={"Python": 98.0, "HTML": 2.0},
        runnability_score=RunnabilityStatus.PARTIALLY_RUNNABLE,
        health_details={
            "has_readme": True,
            "has_dependency_manifest": True,
            "dependency_files": ["setup.py"],
            "has_entry_point": True,
            "entry_point_files": ["manage.py"],
            "empty_implementations_count": 5,
            "todo_comment_count": 12,
            "summary_assessment": "Partially runnable framework repo",
        },
        index_storage_path="./data/indices/django_django_abcdef12",
    )
    db_session.add(repo)
    db_session.commit()
    db_session.refresh(repo)

    # Validate with Pydantic V2 from_attributes
    dto = RepositoryRead.model_validate(repo)
    assert dto.id == "repo_serialize_test"
    assert dto.owner == "django"
    assert dto.name == "django"
    assert dto.status == IngestionStatusEnum.INDEXED
    assert dto.runnability_score == RunnabilityEnum.PARTIALLY_RUNNABLE
    assert dto.health_details.has_readme is True
    assert dto.health_details.todo_comment_count == 12
    assert dto.language_distribution["Python"] == 98.0
