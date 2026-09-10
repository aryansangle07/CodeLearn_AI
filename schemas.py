from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class IngestionStatusEnum(str, Enum):
    PENDING = "PENDING"
    FETCHING = "FETCHING"
    PARSING = "PARSING"
    EMBEDDING = "EMBEDDING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class RunnabilityEnum(str, Enum):
    RUNNABLE = "RUNNABLE"
    PARTIALLY_RUNNABLE = "PARTIALLY_RUNNABLE"
    ANALYSIS_ONLY = "ANALYSIS_ONLY"


class IntentTypeEnum(str, Enum):
    SIMPLE_LOOKUP = "simple_lookup"
    EXPLANATION = "explanation"
    ARCHITECTURE = "architecture"
    COMPARATIVE = "comparative"
    UNRELATED = "unrelated"
    FACTUAL = "factual"
    CROSS_FILE = "cross_file"
    CONFIGURATION = "configuration"
    CODE_DEEP_DIVE = "code_deep_dive"


class HealthDetailsDTO(BaseModel):
    has_readme: bool = False
    has_dependency_manifest: bool = False
    dependency_files: List[str] = Field(default_factory=list)
    has_entry_point: bool = False
    entry_point_files: List[str] = Field(default_factory=list)
    empty_implementations_count: int = 0
    todo_comment_count: int = 0
    summary_assessment: str = "Analysis pending"

    model_config = ConfigDict(from_attributes=True)


class RepositoryBase(BaseModel):
    url: str
    owner: str
    name: str
    branch: str = "main"


class RepositoryCreate(RepositoryBase):
    id: str
    commit_sha: str
    index_storage_path: str


class RepositoryRead(RepositoryBase):
    id: str
    commit_sha: str
    status: IngestionStatusEnum
    error_message: Optional[str] = None
    total_files: int = 0
    total_lines: int = 0
    total_size_bytes: int = 0
    language_distribution: Dict[str, float] = Field(default_factory=dict)
    runnability_score: RunnabilityEnum = RunnabilityEnum.ANALYSIS_ONLY
    health_details: HealthDetailsDTO = Field(default_factory=HealthDetailsDTO)
    is_demo: bool = False
    index_storage_path: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Phase 4 & 5 Query Request/Response DTOs ---

class RepositorySubmissionRequest(BaseModel):
    github_url: str = Field(..., description="Public GitHub repository URL (e.g., https://github.com/fastapi/fastapi)")
    branch: Optional[str] = Field("main", description="Target Git branch or tag name")


class QueryRequest(BaseModel):
    repository_id: str = Field(..., description="Target repository UUID")
    query: str = Field(..., min_length=3, max_length=1000, description="User question about the codebase")


class RepositorySubmissionResponse(BaseModel):
    repository_id: str
    status: IngestionStatusEnum
    message: str
    is_cached: bool = Field(False, description="True if repository commit was already indexed")


class RepositoryStatusResponse(BaseModel):
    repository_id: str
    status: IngestionStatusEnum
    error_message: Optional[str] = None
    progress_percentage: int = Field(..., ge=0, le=100)
    current_step: str


class RepositorySummaryResponse(BaseModel):
    repository_id: str
    url: str
    owner: str
    name: str
    commit_sha: str
    status: IngestionStatusEnum
    total_files: int
    total_lines: int
    total_size_bytes: int
    language_distribution: Dict[str, float]
    runnability_score: RunnabilityEnum
    health_details: HealthDetailsDTO
    is_demo: bool = False
    created_at: str


class RetrievedChunkDTO(BaseModel):
    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    similarity_score: float = 0.0

    model_config = ConfigDict(from_attributes=True)


class QueryResponse(BaseModel):
    query: str
    intent_type: IntentTypeEnum
    answer: str
    citations: List[str] = Field(default_factory=list, description="List of verified relative file paths cited in answer")
    retrieved_chunks: List[RetrievedChunkDTO] = Field(default_factory=list)
    is_grounded: bool = True
    execution_time_seconds: float = 0.0

    model_config = ConfigDict(from_attributes=True)
