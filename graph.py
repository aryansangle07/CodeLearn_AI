import os
import time
from typing import Any, Dict
from langgraph.graph import StateGraph, END

from graph_state import GraphState
from intent_analyzer import IntentAnalyzer
from retriever import HybridRetriever
from llm_service import LLMService
from citation_validator import CitationValidator


def intent_analyzer_node(state: GraphState) -> Dict[str, Any]:
    """Node 1: Classifies user query intent deterministically via rule-based matcher."""
    query = state.get("query", "")
    intent = IntentAnalyzer.classify_intent(query)
    return {"intent_type": intent}


def hybrid_retriever_node(state: GraphState) -> Dict[str, Any]:
    """Node 2: Executes dual dense + sparse hybrid retrieval with RRF fusion."""
    query = state.get("query", "")
    index_dir = state.get("index_dir", "")

    if not index_dir:
        return {"retrieved_chunks": [], "error": "No index directory provided."}

    try:
        retriever = HybridRetriever.load_repository_index(index_dir)
        chunks = retriever.hybrid_search(query=query, top_k=5)
        return {"retrieved_chunks": chunks}
    except Exception as e:
        return {"retrieved_chunks": [], "error": f"Retrieval failed: {str(e)}"}


def response_generator_node(state: GraphState) -> Dict[str, Any]:
    """Node 3: Generates a strictly grounded technical answer using single LLM call budget."""
    query = state.get("query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    raw_response = LLMService.generate_response(query, retrieved_chunks)
    return {"raw_response": raw_response}


def citation_validator_node(state: GraphState) -> Dict[str, Any]:
    """Node 4: Validates file citations and enforces grounding integrity."""
    raw_response = state.get("raw_response", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    validated_answer, citations, is_grounded = CitationValidator.validate_citations(
        raw_response, retrieved_chunks
    )

    return {
        "answer": validated_answer,
        "citations": citations,
        "is_grounded": is_grounded,
    }


def unrelated_handler_node(state: GraphState) -> Dict[str, Any]:
    """Handles out-of-domain / unrelated queries by generating ungrounded helper response with disclaimer."""
    query = state.get("query", "")
    index_dir = state.get("index_dir", "")
    repo_name = os.path.basename(os.path.dirname(index_dir)) if index_dir else "the indexed repository"

    unrelated_answer = LLMService.generate_unrelated_response(query, repo_name=repo_name)
    return {
        "raw_response": unrelated_answer,
        "answer": unrelated_answer,
        "citations": [],
        "is_grounded": False,
    }


def route_by_intent(state: GraphState) -> str:
    """Conditional edge router branching between in-scope code pipeline and unrelated handler."""
    intent = state.get("intent_type", "explanation")
    if intent == "unrelated":
        return "unrelated_handler"
    return "hybrid_retriever"


def build_codelearn_graph():
    """
    Constructs and compiles the deterministic LangGraph workflow:
    [Intent Analyzer] ──(unrelated)──> [Unrelated Handler] ───────────────> END
            │
        (in-scope)
            ▼
    [Hybrid Retriever] -> [Response Generator] -> [Citation Validator] -> END
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("intent_analyzer", intent_analyzer_node)
    workflow.add_node("unrelated_handler", unrelated_handler_node)
    workflow.add_node("hybrid_retriever", hybrid_retriever_node)
    workflow.add_node("response_generator", response_generator_node)
    workflow.add_node("citation_validator", citation_validator_node)

    workflow.set_entry_point("intent_analyzer")
    workflow.add_conditional_edges(
        "intent_analyzer",
        route_by_intent,
        {
            "unrelated_handler": "unrelated_handler",
            "hybrid_retriever": "hybrid_retriever",
        },
    )
    workflow.add_edge("unrelated_handler", END)
    workflow.add_edge("hybrid_retriever", "response_generator")
    workflow.add_edge("response_generator", "citation_validator")
    workflow.add_edge("citation_validator", END)

    return workflow.compile()


def execute_query_pipeline(
    repository_id: str,
    index_dir: str,
    query: str,
) -> GraphState:
    """Convenience helper to run the compiled LangGraph pipeline end-to-end with execution timing."""
    start_time = time.perf_counter()
    app = build_codelearn_graph()

    initial_state: GraphState = {
        "repository_id": repository_id,
        "index_dir": index_dir,
        "query": query,
        "retrieved_chunks": [],
        "raw_response": "",
        "answer": "",
        "citations": [],
        "is_grounded": False,
        "execution_time_seconds": 0.0,
        "error": None,
    }

    final_state = app.invoke(initial_state)
    elapsed = round(time.perf_counter() - start_time, 3)
    final_state["execution_time_seconds"] = elapsed

    return final_state
