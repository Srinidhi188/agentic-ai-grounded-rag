"""
API Route Handlers.
Exposes REST endpoints for health checks, application metadata, and RAG chat queries.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.schemas import (
    HealthResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


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
