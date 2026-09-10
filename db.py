import enum
from datetime import datetime, timezone
from typing import Generator
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    BigInteger,
    Boolean,
    Text,
    DateTime,
    Enum as SQLEnum,
    JSON,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()


class IngestionStatus(str, enum.Enum):
    PENDING = "PENDING"
    FETCHING = "FETCHING"
    PARSING = "PARSING"
    EMBEDDING = "EMBEDDING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class RunnabilityStatus(str, enum.Enum):
    RUNNABLE = "RUNNABLE"
    PARTIALLY_RUNNABLE = "PARTIALLY_RUNNABLE"
    ANALYSIS_ONLY = "ANALYSIS_ONLY"


def utc_now():
    return datetime.now(timezone.utc)


class Repository(Base):
    __tablename__ = "repositories"

    id = Column(String(36), primary_key=True)  # UUIDv4
    url = Column(String(512), nullable=False)  # GitHub URL
    owner = Column(String(100), nullable=False)  # e.g., "fastapi"
    name = Column(String(100), nullable=False)  # e.g., "fastapi"
    branch = Column(String(100), nullable=False, default="main")  # Target branch
    commit_sha = Column(String(40), nullable=False)  # 40-char git commit SHA
    status = Column(
        SQLEnum(IngestionStatus, name="ingestion_status_enum", create_type=True),
        nullable=False,
        default=IngestionStatus.PENDING,
        index=True,
    )
    error_message = Column(Text, nullable=True)  # Failure trace / details
    total_files = Column(Integer, nullable=False, default=0)
    total_lines = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(BigInteger, nullable=False, default=0)
    language_distribution = Column(JSON, nullable=False, default=dict)  # {"Python": 85.0}
    runnability_score = Column(
        SQLEnum(RunnabilityStatus, name="runnability_status_enum", create_type=True),
        nullable=False,
        default=RunnabilityStatus.ANALYSIS_ONLY,
    )
    health_details = Column(JSON, nullable=False, default=dict)  # Health details metrics
    is_demo = Column(Boolean, nullable=False, default=False)  # True for pre-seeded permanent demo repos
    index_storage_path = Column(String(512), nullable=False)  # On-disk directory
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    __table_args__ = (
        UniqueConstraint("owner", "name", "commit_sha", name="uq_repo_commit"),
    )


def get_engine(database_url: str = None):
    """Creates a SQLAlchemy engine configured for PostgreSQL or SQLite (testing)."""
    if database_url is None:
        from config import get_settings
        database_url = get_settings().DATABASE_URL

    if database_url and database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )
    return create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_pre_ping=True,
    )


def get_session_factory(engine=None) -> sessionmaker:
    """Returns a configured sessionmaker instance for a given engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db(engine=None) -> Generator[Session, None, None]:
    """
    Database session generator yielding a managed session with automatic rollback on error.
    Can be used as a FastAPI dependency or a standalone generator.
    """
    if engine is None:
        from config import get_settings

        settings = get_settings()
        engine = get_engine(settings.DATABASE_URL)

    session_factory = get_session_factory(engine)
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine) -> None:
    """Creates all database tables and enums defined in Base.metadata."""
    Base.metadata.create_all(bind=engine)


def check_db_health(engine) -> bool:
    """Safely executes a health-check query (SELECT 1) without raising unhandled exceptions."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return result.scalar() == 1
    except (SQLAlchemyError, Exception):
        return False
