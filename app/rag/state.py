"""
LangGraph State Definition.
Defines strongly typed workflow state passed between graph nodes.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict):
    """
    State schema for the LangGraph Grounded RAG workflow.

    Attributes:
        user_query: The incoming user question.
        retrieved_documents: List of retrieved chunks with metadata (text, page, chunk_id, score, source).
        retrieval_scores: List of float similarity scores for retrieved chunks.
        context: Synthesized textual context formatted for the generation prompt.
        confidence: Normalized relevance/confidence heuristic signal [0.0 - 1.0].
        should_answer: Boolean flag evaluated by the relevance gate.
        answer: The final synthesized or fallback answer.
        error: Optional error message if an exception occurred during node execution.
    """
    user_query: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_scores: List[float]
    context: str
    confidence: float
    should_answer: bool
    answer: str
    error: Optional[str]
