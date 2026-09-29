"""
FastAPI Application Entry Point.
Initializes logging, middleware, exception handlers, and API routers.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.config import get_settings

# Configure structured logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agentic_ai_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Starting Grounded RAG Chatbot service [env=%s]", settings.APP_ENV)
    logger.info("Knowledge Base Source: %s", settings.PDF_URL)
    yield
    logger.info("Shutting down Grounded RAG Chatbot service.")


app = FastAPI(
    title="Agentic AI eBook — Grounded RAG Chatbot",
    description=(
        "Production-quality RAG system answering queries strictly from the "
        "Agentic AI eBook with zero hallucination guarantee, orchestrated by LangGraph."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to capture unhandled errors gracefully
    without leaking internal traceback or credentials to the client.
    """
    logger.error("Unhandled server exception at %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please try again later.",
            "error_type": "InternalServerError",
        },
    )


# Register API routes
app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """Root redirect / welcome endpoint."""
    return {
        "message": "Welcome to Agentic AI eBook Grounded RAG API",
        "documentation": "/docs",
        "health_check": "/health",
    }
