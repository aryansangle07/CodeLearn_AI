"""
Demo Repository Auto-Seed Service for CodeLearn AI.

Provides an opt-in seeding mechanism that automatically pre-indexes 2-3 small,
high-quality, developer-friendly public repositories on initial startup.
Uses the existing core ingestion pipeline (worker.py) with full fault-tolerance.
"""
import os
import logging
import uuid
from typing import List, Dict
from datetime import datetime, timezone
import concurrent.futures

from config import get_settings
from db import (
    get_engine,
    get_session_factory,
    Repository,
    IngestionStatus,
    RunnabilityStatus,
)
from github_service import GitHubService
from worker import ingest_repository_worker

logger = logging.getLogger(__name__)

# Curated List of Demo Repositories:
# 1. encode/starlette - Python async framework with clean class & middleware AST hierarchies.
# 2. pallets/click - Composable CLI framework with decorators and command trees.
# 3. psf/requests - Gold-standard Python HTTP library with modular architecture.
DEMO_REPOSITORIES: List[Dict[str, str]] = [
    {
        "url": "https://github.com/encode/starlette",
        "branch": "master",
        "description": "Lightweight ASGI framework/toolkit showcasing async AST & middleware structures.",
    },
    {
        "url": "https://github.com/pallets/click",
        "branch": "main",
        "description": "Composable CLI package showcasing decorators, command trees, and docstring chunks.",
    },
    {
        "url": "https://github.com/psf/requests",
        "branch": "main",
        "description": "Standard Python HTTP library showcasing clean modular architecture and sessions.",
    },
]


def seed_single_repository(repo_info: Dict[str, str], engine=None) -> bool:
    """
    Seeds a single demo repository through the standard ingestion pipeline.
    Ensures idempotency and graceful error handling.
    """
    if engine is None:
        engine = get_engine()

    SessionLocal = get_session_factory(engine)
    session = SessionLocal()
    url = repo_info["url"]
    branch = repo_info.get("branch", "main")

    try:
        owner, name = GitHubService.parse_github_url(url)
    except Exception as e:
        logger.warning(f"[SEED WARNING] Invalid demo repository URL '{url}': {e}")
        session.close()
        return False

    try:
        commit_sha = GitHubService.resolve_latest_commit_sha(owner, name, branch)
        settings = get_settings()

        # Check if already indexed or pending
        existing = session.query(Repository).filter(
            Repository.owner == owner,
            Repository.name == name,
        ).order_by(
            (Repository.status == IngestionStatus.INDEXED).desc(),
            Repository.created_at.desc(),
        ).first()

        if existing and existing.status == IngestionStatus.INDEXED:
            logger.info(f"[SEED] Demo repository '{owner}/{name}' is already indexed. Skipping.")
            return True

        if existing:
            repo_id = existing.id
            existing.status = IngestionStatus.PENDING
            existing.error_message = None
            existing.updated_at = datetime.now(timezone.utc)
            session.commit()
        else:
            repo_id = str(uuid.uuid4())
            index_storage_path = os.path.abspath(
                os.path.join(settings.INDEX_STORAGE_DIR, f"{owner}_{name}_{commit_sha}")
            )
            new_repo = Repository(
                id=repo_id,
                url=url,
                owner=owner,
                name=name,
                branch=branch,
                commit_sha=commit_sha,
                status=IngestionStatus.PENDING,
                index_storage_path=index_storage_path,
                language_distribution={},
                health_details={},
                runnability_score=RunnabilityStatus.ANALYSIS_ONLY,
                is_demo=True,
            )
            session.add(new_repo)
            session.commit()

        logger.info(f"[SEED] Starting automated ingestion for demo repository '{owner}/{name}' (ID: {repo_id})...")
        
        # Execute using standard ingestion pipeline
        ingest_repository_worker(repo_id, session_factory=SessionLocal)

        # Verify result
        session.refresh(existing if existing else new_repo)
        final_status = (existing if existing else new_repo).status
        if final_status == IngestionStatus.INDEXED:
            logger.info(f"[SEED SUCCESS] Demo repository '{owner}/{name}' successfully indexed.")
            return True
        else:
            logger.warning(f"[SEED WARNING] Demo repository '{owner}/{name}' ended with status: {final_status}")
            return False

    except Exception as exc:
        logger.warning(f"[SEED WARNING] Failed to seed demo repository '{url}': {exc}. Server startup continues normally.")
        session.rollback()
        return False
    finally:
        session.close()


def run_demo_seed_pipeline(background: bool = True) -> None:
    """
    Executes the demo repository seeding pipeline if enabled by configuration (SEED_DEMO_REPOS=True).
    Safe, optional, and non-blocking.
    """
    settings = get_settings()
    if not getattr(settings, "SEED_DEMO_REPOS", False):
        logger.info("[SEED] Demo repository auto-seeding is disabled (SEED_DEMO_REPOS=False). Skipping.")
        return

    logger.info(f"[SEED] Demo repository auto-seeding is ENABLED. Pre-loading {len(DEMO_REPOSITORIES)} repositories...")

    def _execute_all():
        for repo_info in DEMO_REPOSITORIES:
            try:
                seed_single_repository(repo_info)
            except Exception as e:
                logger.warning(f"[SEED WARNING] Uncaught error seeding '{repo_info.get('url')}': {e}")

    if background:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="DemoSeeder")
        executor.submit(_execute_all)
    else:
        _execute_all()
