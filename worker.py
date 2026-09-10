import os
import traceback
from typing import Dict, List
from datetime import datetime, timezone

from config import get_settings
from db import (
    get_session_factory,
    Repository,
    IngestionStatus,
    RunnabilityStatus,
)
from github_service import GitHubService
from file_filter import FileFilter
from security import SecurityScanner
from parser import CodeParser
from chunker import CodeChunker, CodeChunk
from health_analyzer import HealthAnalyzer
from retriever import HybridRetriever


def ingest_repository_worker(repository_id: str, session_factory=None) -> None:
    """
    Asynchronous background ingestion pipeline:
    1. FETCHING: Streams and extracts GitHub tarball.
    2. PARSING: Filters safe files, redacts secrets, extracts AST symbols, chunks code, and evaluates health.
    3. EMBEDDING: Generates dense embeddings and builds FAISS + BM25 dual index.
    4. INDEXED: Persists metrics to database and cleans up temporary extraction folder.
    """
    settings = get_settings()
    if session_factory is None:
        session_factory = get_session_factory()
    session = session_factory()

    repo = session.query(Repository).filter(Repository.id == repository_id).first()
    if not repo:
        session.close()
        return

    extracted_repo_dir = None

    try:
        # --- STEP 1: FETCHING ---
        repo.status = IngestionStatus.FETCHING
        repo.updated_at = datetime.now(timezone.utc)
        session.commit()

        extracted_info = GitHubService.download_and_extract_tarball(
            owner=repo.owner,
            name=repo.name,
            branch=repo.branch,
        )
        extracted_repo_dir = extracted_info.repo_dir

        # Update commit sha if resolved and clean any stale duplicate records
        if extracted_info.commit_sha:
            if extracted_info.commit_sha != repo.commit_sha:
                stale_dupes = session.query(Repository).filter(
                    Repository.owner == repo.owner,
                    Repository.name == repo.name,
                    Repository.commit_sha == extracted_info.commit_sha,
                    Repository.id != repo.id,
                ).all()
                for sd in stale_dupes:
                    session.delete(sd)
                session.flush()
                repo.commit_sha = extracted_info.commit_sha

        # Determine index storage directory
        index_dir = os.path.abspath(
            os.path.join(settings.INDEX_STORAGE_DIR, f"{repo.owner}_{repo.name}_{repo.commit_sha}")
        )
        repo.index_storage_path = index_dir
        session.commit()

        # --- STEP 2: PARSING & SECURITY ---
        repo.status = IngestionStatus.PARSING
        repo.updated_at = datetime.now(timezone.utc)
        session.commit()

        safe_files, language_distribution = FileFilter.scan_repository_tree(
            root_dir=extracted_repo_dir,
            max_repo_size_mb=settings.MAX_REPOSITORY_SIZE_MB,
            max_files=settings.MAX_FILES_PER_REPOSITORY,
        )

        file_contents: Dict[str, str] = {}
        all_chunks: List[CodeChunk] = []
        total_lines = 0
        total_size_bytes = 0

        for rel_path in safe_files:
            full_path = os.path.join(extracted_repo_dir, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_content = f.read()
            except Exception:
                continue

            file_size = len(raw_content.encode("utf-8"))
            total_size_bytes += file_size
            lines = raw_content.splitlines()
            total_lines += len(lines)

            # Redact secrets
            sanitized_content, _ = SecurityScanner.redact_secrets(raw_content)
            file_contents[rel_path] = sanitized_content

            # Parse symbols and create chunks
            lang = FileFilter.get_file_language(rel_path)
            parsed_file = CodeParser.parse_file(rel_path, sanitized_content, lang)
            chunks = CodeChunker.create_chunks(parsed_file)
            all_chunks.extend(chunks)

        # Runnability health analysis
        runnability, health_dto = HealthAnalyzer.evaluate_repository(safe_files, file_contents)

        # --- STEP 3: EMBEDDING & INDEXING ---
        repo.status = IngestionStatus.EMBEDDING
        repo.updated_at = datetime.now(timezone.utc)
        session.commit()

        HybridRetriever.index_repository(
            chunks=all_chunks,
            output_dir=index_dir,
            owner=repo.owner,
            name=repo.name,
            commit_sha=repo.commit_sha,
        )

        # --- STEP 4: INDEXED & PERSISTENCE ---
        repo.status = IngestionStatus.INDEXED
        repo.total_files = len(safe_files)
        repo.total_lines = total_lines
        repo.total_size_bytes = total_size_bytes
        repo.language_distribution = language_distribution
        repo.runnability_score = runnability
        repo.health_details = health_dto.model_dump()
        repo.error_message = None
        repo.updated_at = datetime.now(timezone.utc)
        session.commit()

    except Exception as e:
        session.rollback()
        err_trace = traceback.format_exc()
        repo = session.query(Repository).filter(Repository.id == repository_id).first()
        if repo:
            repo.status = IngestionStatus.FAILED
            repo.error_message = f"{str(e)}: {err_trace[-300:]}"
            repo.updated_at = datetime.now(timezone.utc)
            session.commit()
    finally:
        if extracted_repo_dir:
            GitHubService.cleanup_repo_dir(extracted_repo_dir)
        session.close()
