from typing import Any, Dict, List, Optional, TypedDict
from schemas import IntentTypeEnum


class GraphState(TypedDict, total=False):
    repository_id: str
    index_dir: str
    query: str
    intent_type: IntentTypeEnum
    retrieved_chunks: List[Dict[str, Any]]
    raw_response: str
    answer: str
    citations: List[str]
    is_grounded: bool
    execution_time_seconds: float
    error: Optional[str]
