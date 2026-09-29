"""
LangGraph Grounded RAG Orchestration Workflow.
Implements an explicit state machine with retrieval, relevance gating,
grounded generation, and deterministic fallback paths.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from langgraph.graph import StateGraph, START, END

from app.config import Settings, get_settings
from app.rag.state import RAGState
from app.rag.prompts import (
    format_context_chunks,
    FALLBACK_ANSWER,
)
from app.rag.generator import BaseLLMGenerator, get_generator
from app.ingestion.embedder import BaseEmbedder, get_embedder
from app.vectorstore.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)


def calculate_confidence_score(scores: List[float]) -> float:
    """
    Transparent retrieval relevance confidence heuristic.
    
    Calculation:
    - If no chunks were retrieved: confidence = 0.0
    - Otherwise, bounded normalized score of the top chunk:
      confidence = round(max(0.0, min(1.0, max(scores))), 4)
      
    NOTE: This is a retrieval similarity signal, NOT a mathematically calibrated
    probability of correctness.
    """
    if not scores:
        return 0.0
    top_score = max(scores)
    return round(max(0.0, min(1.0, float(top_score))), 4)


class RAGWorkflow:
    """Encapsulates the compiled LangGraph state machine and its dependencies."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[BaseEmbedder] = None,
        generator: Optional[BaseLLMGenerator] = None,
        retriever_func: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.generator = generator or get_generator(self.settings)
        self._custom_retriever = retriever_func
        self._graph = self._build_graph()

    def _default_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Default retrieval using Pinecone store."""
        try:
            store = PineconeStore(self.settings)
            query_vector = self.embedder.embed_query(query)
            return store.query(query_embedding=query_vector, top_k=top_k)
        except Exception as e:
            logger.warning("Pinecone query failed or not configured (%s). Returning empty list.", e)
            return []

    # ==================== Graph Node Functions ====================

    def retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 1: Retrieve relevant chunks and calculate relevance metrics."""
        query = state.get("user_query", "").strip()
        top_k = self.settings.RETRIEVAL_TOP_K

        logger.info("LangGraph [retrieve] node executing for query: '%s'", query)

        try:
            if self._custom_retriever is not None:
                docs = self._custom_retriever(query, top_k)
            else:
                docs = self._default_retrieve(query, top_k)
        except Exception as e:
            logger.error("Error in retrieval node: %s", e)
            return {
                "retrieved_documents": [],
                "retrieval_scores": [],
                "context": "",
                "confidence": 0.0,
                "should_answer": False,
                "error": str(e),
            }

        scores = [float(doc.get("score", 0.0)) for doc in docs]
        confidence = calculate_confidence_score(scores)
        context = format_context_chunks(docs)

        # Relevance Gate Check:
        # Requires non-empty documents and top retrieval score >= configured threshold
        is_sufficient = bool(docs and scores and max(scores) >= self.settings.RELEVANCE_THRESHOLD)

        logger.info(
            "LangGraph [retrieve] completed. Chunks: %d, Top Score: %.4f, Threshold: %.2f, Sufficient: %s",
            len(docs),
            max(scores) if scores else 0.0,
            self.settings.RELEVANCE_THRESHOLD,
            is_sufficient,
        )

        return {
            "retrieved_documents": docs,
            "retrieval_scores": scores,
            "context": context,
            "confidence": confidence,
            "should_answer": is_sufficient,
            "error": None,
        }

    def check_relevance(self, state: RAGState) -> str:
        """
        Conditional Router: Directs workflow based on relevance evaluation.
        Routes to 'generate_answer' if context satisfies relevance threshold,
        otherwise routes to 'fallback' without calling the LLM.
        """
        if state.get("should_answer", False):
            return "generate_answer"
        return "fallback"

    def generate_answer_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 2A: Synthesize grounded answer via LLM."""
        query = state.get("user_query", "")
        context = state.get("context", "")

        logger.info("LangGraph [generate_answer] executing grounded synthesis.")
        try:
            answer = self.generator.generate(query=query, context=context)
        except Exception as e:
            logger.error("LLM Generation failed: %s", e)
            return {
                "answer": FALLBACK_ANSWER,
                "error": f"LLM Generation failure: {e}",
            }

        return {
            "answer": answer,
            "error": None,
        }

    def fallback_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Node 2B: Deterministic fallback node executed when retrieved context
        is insufficient or irrelevant. Ensures zero hallucination and bypasses LLM.
        """
        logger.info("LangGraph [fallback] node executed. Bypassing LLM generation.")
        return {
            "answer": FALLBACK_ANSWER,
            "error": None,
        }

    # ==================== Graph Construction ====================

    def _build_graph(self):
        """Construct and compile the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Register nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("fallback", self.fallback_node)

        # Wire edges
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.check_relevance,
            {
                "generate_answer": "generate_answer",
                "fallback": "fallback",
            },
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("fallback", END)

        return workflow.compile()

    def run(self, query: str) -> RAGState:
        """Execute the Grounded RAG workflow for a user query."""
        initial_state: RAGState = {
            "user_query": query,
            "retrieved_documents": [],
            "retrieval_scores": [],
            "context": "",
            "confidence": 0.0,
            "should_answer": False,
            "answer": "",
            "error": None,
        }
        return self._graph.invoke(initial_state)


def create_rag_workflow(
    settings: Optional[Settings] = None,
    embedder: Optional[BaseEmbedder] = None,
    generator: Optional[BaseLLMGenerator] = None,
    retriever_func: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
) -> RAGWorkflow:
    """Factory creating an instance of RAGWorkflow."""
    return RAGWorkflow(
        settings=settings,
        embedder=embedder,
        generator=generator,
        retriever_func=retriever_func,
    )
