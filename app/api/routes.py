"""
API Route Handlers.
Exposes REST endpoints for health checks, application metadata, and RAG chat queries.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.schemas import (
    HealthResponse,
    ChatRequest,
    ChatResponse,
    ContextChunk,
    ErrorResponse,
)
from app.config import Settings, get_settings
from app.rag.graph import RAGWorkflow, create_rag_workflow

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton workflow instance for application lifecycle
_cached_workflow: Optional[RAGWorkflow] = None


def get_rag_workflow(settings: Settings = Depends(get_settings)) -> RAGWorkflow:
    """Dependency provider returning singleton RAGWorkflow instance."""
    global _cached_workflow
    if _cached_workflow is None:
        logger.info("Initializing application RAGWorkflow instance...")
        _cached_workflow = create_rag_workflow(settings=settings)
    return _cached_workflow


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the system status, configuration readiness, and provider settings.",
)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Returns application health and configuration readiness."""
    return HealthResponse(
        status="ok",
        environment=settings.APP_ENV,
        version="0.1.0",
        pinecone_configured=bool(settings.PINECONE_API_KEY),
        llm_provider=settings.LLM_PROVIDER,
        embedding_provider=settings.EMBEDDING_PROVIDER,
    )


@router.get(
    "/info",
    summary="Application Information",
    description="Provides metadata about the knowledge base and configured models.",
)
async def app_info(settings: Settings = Depends(get_settings)):
    """Returns metadata about the active configuration and knowledge base."""
    return {
        "app_name": "Agentic AI eBook - Grounded RAG Chatbot",
        "knowledge_base": {
            "source_pdf": settings.PDF_URL,
            "vector_store": "Pinecone",
            "index_name": settings.PINECONE_INDEX_NAME,
            "namespace": settings.PINECONE_NAMESPACE,
        },
        "models": {
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
        },
        "retrieval_parameters": {
            "top_k": settings.RETRIEVAL_TOP_K,
            "relevance_threshold": settings.RELEVANCE_THRESHOLD,
            "max_context_chunks": settings.MAX_CONTEXT_CHUNKS,
        },
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Knowledge Base",
    description=(
        "Ask a question grounded strictly in the Agentic AI eBook. "
        "Retrieves relevant semantic passages from Pinecone and synthesizes a verified answer via LangGraph."
    ),
    responses={
        200: {"description": "Grounded answer synthesized or deterministic fallback", "model": ChatResponse},
        422: {"description": "Pydantic validation failure (empty, invalid, or oversized query)"},
        500: {"description": "Internal server processing failure", "model": ErrorResponse},
    },
)
async def chat_endpoint(
    request: ChatRequest,
    workflow: RAGWorkflow = Depends(get_rag_workflow),
) -> ChatResponse:
    """Execute LangGraph RAG workflow on incoming user question."""
    query = request.query.strip()
    logger.info("Handling /chat request: '%s' (chars=%d)", query[:80], len(query))

    try:
        result_state = workflow.run(query)
    except Exception as e:
        logger.error("Internal failure during RAG execution: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while executing the RAG workflow.",
        )

    # Check for reported node errors
    if result_state.get("error"):
        logger.warning("RAG state recorded node error: %s", result_state["error"])

    # Map retrieved raw documents to structured ContextChunk models
    raw_docs = result_state.get("retrieved_documents", [])
    context_chunks = [
        ContextChunk(
            text=doc.get("text", ""),
            source=doc.get("source", "Ebook-Agentic-AI.pdf"),
            page=int(doc.get("page", 0)),
            score=round(float(doc.get("score", 0.0)), 4),
            chunk_id=doc.get("chunk_id"),
        )
        for doc in raw_docs
    ]

    is_grounded = bool(result_state.get("should_answer", False))
    confidence = float(result_state.get("confidence", 0.0))
    answer = result_state.get("answer", "")

    logger.info(
        "Finished /chat request: chunks=%d, confidence=%.4f, grounded=%s",
        len(context_chunks),
        confidence,
        is_grounded,
    )

    return ChatResponse(
        answer=answer,
        context=context_chunks,
        confidence=confidence,
        is_grounded=is_grounded,
    )
