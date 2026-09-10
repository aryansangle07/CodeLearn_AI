import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings
from db import (
    get_db,
    get_engine,
    get_session_factory,
    init_db,
    check_db_health,
    Repository,
    IngestionStatus,
    RunnabilityStatus,
)
from schemas import (
    IngestionStatusEnum,
    RunnabilityEnum,
    IntentTypeEnum,
    HealthDetailsDTO,
    RepositorySubmissionRequest,
    RepositorySubmissionResponse,
    RepositoryStatusResponse,
    RepositorySummaryResponse,
    RepositoryRead,
    QueryRequest,
    QueryResponse,
    RetrievedChunkDTO,
)
from github_service import GitHubService
from graph import execute_query_pipeline
from worker import ingest_repository_worker
from seed_service import run_demo_seed_pipeline


def resolve_index_path(repo: Repository) -> str:
    """Resolves index directory path reliably across Windows and Linux deployments."""
    settings = get_settings()
    expected_name = f"{repo.owner}_{repo.name}_{repo.commit_sha}"
    local_path = os.path.abspath(os.path.join(settings.INDEX_STORAGE_DIR, expected_name))
    if os.path.exists(local_path):
        return local_path
    if repo.index_storage_path and os.path.exists(repo.index_storage_path):
        return repo.index_storage_path
    return local_path


def purge_expired_user_repositories(session: Session, max_age_hours: int = 2) -> int:
    """Purges non-demo user-added repositories older than max_age_hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    expired_repos = session.query(Repository).filter(
        Repository.is_demo == False,
        Repository.created_at < cutoff,
    ).all()
    count = 0
    settings = get_settings()
    data_base_dir = os.path.dirname(os.path.abspath(settings.INDEX_STORAGE_DIR))
    for r in expired_repos:
        index_p = resolve_index_path(r)
        if index_p and os.path.exists(index_p):
            shutil.rmtree(index_p, ignore_errors=True)
        repo_extract_dir = os.path.join(data_base_dir, "repos", f"{r.owner}_{r.name}_{r.commit_sha}")
        if os.path.exists(repo_extract_dir):
            shutil.rmtree(repo_extract_dir, ignore_errors=True)
        session.delete(r)
        count += 1
    if count > 0:
        session.commit()
    return count


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    settings = get_settings()
    engine = get_engine()
    init_db(engine)

    SessionLocal = get_session_factory(engine)
    session = SessionLocal()
    try:
        # Clean orphaned in-flight ingestion jobs from previous server run
        orphans = session.query(Repository).filter(
            Repository.status.in_([
                IngestionStatus.PENDING,
                IngestionStatus.FETCHING,
                IngestionStatus.PARSING,
                IngestionStatus.EMBEDDING,
            ])
        ).all()
        for orphan in orphans:
            orphan.status = IngestionStatus.FAILED
            orphan.error_message = "Ingestion interrupted by server restart."
            orphan.updated_at = datetime.now(timezone.utc)
        session.commit()

        # Purge any old temporary user-added repositories (> 2 hours old)
        purged = purge_expired_user_repositories(session, max_age_hours=2)
        if purged > 0:
            import logging
            logging.getLogger("main").info(f"[JANITOR] Cleaned up {purged} expired user-submitted repositories on startup.")
    except Exception:
        session.rollback()
    finally:
        session.close()

    # Fast-load pre-indexed demo repositories synchronously (<0.05s) or trigger background
    run_demo_seed_pipeline(background=False)

    yield


settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Deterministic codebase comprehension and semantic retrieval engine.",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root():
    """Root landing endpoint providing system overview and API documentation links."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "online",
        "description": "CodeLearn AI - Deterministic Codebase Comprehension & Semantic Retrieval Engine",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "endpoints": {
            "health": "/health",
            "repositories": "/api/repositories",
            "query": "/api/query",
        },
    }


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint validating service and database connectivity."""
    db_connected = check_db_health(get_engine())
    return {
        "status": "healthy" if db_connected else "degraded",
        "database": "connected" if db_connected else "disconnected",
        "app": settings.APP_NAME,
        "version": "1.0.0",
    }


@app.post(
    "/api/repositories",
    response_model=RepositorySubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Repositories"],
)
def submit_repository(
    request: RepositorySubmissionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Submits a public GitHub repository for ingestion, parsing, and dual-indexing.
    Guarantees idempotency by returning cached index if commit SHA is already processed.
    """
    try:
        owner, name = GitHubService.parse_github_url(request.github_url)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    branch = request.branch or "main"
    commit_sha = GitHubService.resolve_latest_commit_sha(owner, name, branch)

    # Check for existing repository record by owner, name, and commit SHA
    existing_repo = db.query(Repository).filter(
        Repository.owner == owner,
        Repository.name == name,
        Repository.commit_sha == commit_sha,
    ).first()

    if existing_repo:
        if existing_repo.status == IngestionStatus.INDEXED and os.path.exists(existing_repo.index_storage_path):
            return RepositorySubmissionResponse(
                repository_id=existing_repo.id,
                status=IngestionStatusEnum.INDEXED,
                message="Repository commit is already indexed and ready for queries.",
                is_cached=True,
            )
        elif existing_repo.status in (
            IngestionStatus.PENDING,
            IngestionStatus.FETCHING,
            IngestionStatus.PARSING,
            IngestionStatus.EMBEDDING,
        ):
            return RepositorySubmissionResponse(
                repository_id=existing_repo.id,
                status=IngestionStatusEnum(existing_repo.status.value),
                message=f"Ingestion is already active for this repository (current status: {existing_repo.status.value}).",
                is_cached=False,
            )
        else:
            # Re-trigger failed ingestion
            existing_repo.status = IngestionStatus.PENDING
            existing_repo.error_message = None
            existing_repo.updated_at = datetime.now(timezone.utc)
            db.commit()

            factory = sessionmaker(bind=db.get_bind())
            background_tasks.add_task(ingest_repository_worker, existing_repo.id, session_factory=factory)

            return RepositorySubmissionResponse(
                repository_id=existing_repo.id,
                status=IngestionStatusEnum.PENDING,
                message="Retrying repository ingestion in background.",
                is_cached=False,
            )

    # Create new repository record
    repo_id = str(uuid.uuid4())
    index_storage_path = os.path.abspath(
        os.path.join(settings.INDEX_STORAGE_DIR, f"{owner}_{name}_{commit_sha}")
    )

    new_repo = Repository(
        id=repo_id,
        url=request.github_url,
        owner=owner,
        name=name,
        branch=branch,
        commit_sha=commit_sha,
        status=IngestionStatus.PENDING,
        index_storage_path=index_storage_path,
        language_distribution={},
        health_details={},
        runnability_score=RunnabilityStatus.ANALYSIS_ONLY,
    )
    db.add(new_repo)
    db.commit()

    # Dispatch asynchronous background worker using matching session bind
    factory = sessionmaker(bind=db.get_bind())
    background_tasks.add_task(ingest_repository_worker, repo_id, session_factory=factory)

    return RepositorySubmissionResponse(
        repository_id=repo_id,
        status=IngestionStatusEnum.PENDING,
        message="Repository submission accepted. Ingestion worker started in background.",
        is_cached=False,
    )


@app.get(
    "/api/repositories",
    response_model=List[RepositoryRead],
    tags=["Repositories"],
)
def list_repositories(db: Session = Depends(get_db)):
    """Lists all submitted repositories in the system."""
    repos = db.query(Repository).order_by(Repository.created_at.desc()).all()
    return repos


@app.get(
    "/api/repositories/{repository_id}/status",
    response_model=RepositoryStatusResponse,
    tags=["Repositories"],
)
def get_repository_status(repository_id: str, db: Session = Depends(get_db)):
    """Polls real-time ingestion status and progress percentage for a given repository."""
    repo = db.query(Repository).filter(Repository.id == repository_id).first()
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repository_id}",
        )

    progress_map = {
        IngestionStatus.PENDING: (5, "Queued for ingestion worker"),
        IngestionStatus.FETCHING: (25, "Streaming and extracting GitHub archive"),
        IngestionStatus.PARSING: (55, "Applying security filters and AST code parsing"),
        IngestionStatus.EMBEDDING: (80, "Generating embeddings and building dual index"),
        IngestionStatus.INDEXED: (100, "Dual FAISS & BM25 Index Active"),
        IngestionStatus.FAILED: (0, "Ingestion Failed"),
    }

    pct, step = progress_map.get(repo.status, (0, "Unknown state"))

    return RepositoryStatusResponse(
        repository_id=repo.id,
        status=IngestionStatusEnum(repo.status.value),
        error_message=repo.error_message,
        progress_percentage=pct,
        current_step=step,
    )


@app.get(
    "/api/repositories/{repository_id}",
    response_model=RepositorySummaryResponse,
    tags=["Repositories"],
)
@app.get(
    "/api/repositories/{repository_id}/summary",
    response_model=RepositorySummaryResponse,
    tags=["Repositories"],
)
def get_repository_summary(repository_id: str, db: Session = Depends(get_db)):
    """Retrieves repository metadata, language distribution, and runnability assessment."""
    repo = db.query(Repository).filter(Repository.id == repository_id).first()
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repository_id}",
        )

    health_dto = HealthDetailsDTO.model_validate(repo.health_details or {})

    return RepositorySummaryResponse(
        repository_id=repo.id,
        url=repo.url,
        owner=repo.owner,
        name=repo.name,
        commit_sha=repo.commit_sha,
        status=IngestionStatusEnum(repo.status.value),
        total_files=repo.total_files,
        total_lines=repo.total_lines,
        total_size_bytes=repo.total_size_bytes,
        language_distribution=repo.language_distribution or {},
        runnability_score=RunnabilityEnum(repo.runnability_score.value),
        health_details=health_dto,
        created_at=repo.created_at.isoformat(),
    )


@app.delete(
    "/api/repositories/{repository_id}",
    tags=["Repositories"],
)
def delete_repository(repository_id: str, db: Session = Depends(get_db)):
    """
    Deletes an ingested repository, removing its vector index files, metadata,
    and database records. Protects default demo repositories and rejects
    deletion if ingestion is actively in progress.
    """
    repo = db.query(Repository).filter(Repository.id == repository_id).first()
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repository_id}",
        )

    # Edge Case: Protect default demo repositories from deletion
    if getattr(repo, "is_demo", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Default demo repositories are protected and cannot be deleted.",
        )

    # Edge Case: Block deletion if mid-ingestion
    if repo.status in (
        IngestionStatus.PENDING,
        IngestionStatus.FETCHING,
        IngestionStatus.PARSING,
        IngestionStatus.EMBEDDING,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete repository while ingestion is actively in progress.",
        )

    # 1. Clean up index directory on storage
    if repo.index_storage_path and os.path.exists(repo.index_storage_path):
        shutil.rmtree(repo.index_storage_path, ignore_errors=True)

    # 2. Clean up raw repo extraction folder if present
    data_base_dir = os.path.dirname(os.path.abspath(settings.INDEX_STORAGE_DIR))
    repo_extract_dir = os.path.join(data_base_dir, "repos", f"{repo.owner}_{repo.name}_{repo.commit_sha}")
    if os.path.exists(repo_extract_dir):
        shutil.rmtree(repo_extract_dir, ignore_errors=True)

    # 3. Remove DB record
    db.delete(repo)
    db.commit()

    return {
        "success": True,
        "repository_id": repository_id,
        "message": f"Repository '{repo.owner}/{repo.name}' and all associated vector indices have been deleted.",
    }


@app.post(
    "/api/cleanup-session",
    tags=["Repositories"],
)
def cleanup_user_session(db: Session = Depends(get_db)):
    """
    Deletes all temporary user-added repositories while keeping the 3 default
    permanent demo repositories intact.
    """
    user_repos = db.query(Repository).filter(Repository.is_demo == False).all()
    deleted_count = 0
    data_base_dir = os.path.dirname(os.path.abspath(settings.INDEX_STORAGE_DIR))

    for repo in user_repos:
        if repo.index_storage_path and os.path.exists(repo.index_storage_path):
            shutil.rmtree(repo.index_storage_path, ignore_errors=True)
        repo_extract_dir = os.path.join(data_base_dir, "repos", f"{repo.owner}_{repo.name}_{repo.commit_sha}")
        if os.path.exists(repo_extract_dir):
            shutil.rmtree(repo_extract_dir, ignore_errors=True)
        db.delete(repo)
        deleted_count += 1

    if deleted_count > 0:
        db.commit()

    return {
        "success": True,
        "deleted_count": deleted_count,
        "message": f"Purged {deleted_count} temporary user repositories. Default demo repositories remain active.",
    }


@app.post(
    "/api/query",
    response_model=QueryResponse,
    tags=["Query & Q&A"],
)
def query_repository(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Executes a grounded question-answering query against an indexed repository using
    the deterministic 4-node LangGraph pipeline.
    """
    repo = db.query(Repository).filter(Repository.id == request.repository_id).first()
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {request.repository_id}",
        )

    if repo.status != IngestionStatus.INDEXED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Repository is not ready for queries. Current status: {repo.status.value}",
        )

    index_path = resolve_index_path(repo)
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Index storage files are missing on disk.",
        )

    # Execute deterministic LangGraph pipeline
    final_state = execute_query_pipeline(
        repository_id=repo.id,
        index_dir=index_path,
        query=request.query,
    )

    retrieved_dtos = [
        RetrievedChunkDTO(
            chunk_id=c.get("chunk_id", ""),
            file_path=c.get("file_path", ""),
            symbol_name=c.get("symbol_name", ""),
            symbol_type=c.get("symbol_type", ""),
            start_line=c.get("start_line", 1),
            end_line=c.get("end_line", 1),
            content=c.get("content", ""),
            similarity_score=c.get("rrf_score", 0.0),
        )
        for c in final_state.get("retrieved_chunks", [])
    ]

    return QueryResponse(
        query=request.query,
        intent_type=final_state.get("intent_type", IntentTypeEnum.EXPLANATION),
        answer=final_state.get("answer", "No answer generated."),
        citations=final_state.get("citations", []),
        retrieved_chunks=retrieved_dtos,
        is_grounded=final_state.get("is_grounded", True),
        execution_time_seconds=final_state.get("execution_time_seconds", 0.0),
    )
