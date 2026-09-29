"""
API Data Contracts and Pydantic Schemas.
Enforces request validation, serialization, and OpenAPI documentation schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(default="ok", examples=["ok"])
    environment: str = Field(..., examples=["development"])
    version: str = Field(..., examples=["0.1.0"])
    pinecone_configured: bool = Field(..., examples=[True])
    llm_provider: str = Field(..., examples=["openai"])
    embedding_provider: str = Field(..., examples=["openai"])


class ContextChunk(BaseModel):
    """Schema for a single retrieved knowledge context chunk."""
    text: str = Field(..., description="Cleaned chunk text extracted from the document")
    source: str = Field(..., description="Source document filename or identifier", examples=["Ebook-Agentic-AI.pdf"])
    page: int = Field(..., description="Page number within source document", examples=[5])
    score: float = Field(..., description="Vector similarity / relevance score [0.0 - 1.0]", examples=[0.87])
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk identifier", examples=["agentic-ai-p05-c02"])


class ChatRequest(BaseModel):
    """User query request schema."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is Agentic AI according to the eBook?"
            }
        }
    )

    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="The natural language question to ask the knowledge base.",
    )


class ChatResponse(BaseModel):
    """Grounded RAG response schema."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "According to the provided document, Agentic AI refers to autonomous systems capable of reasoning, planning, and executing multi-step tasks.",
                "context": [
                    {
                        "text": "Agentic AI refers to AI systems designed to operate with autonomy...",
                        "source": "Ebook-Agentic-AI.pdf",
                        "page": 4,
                        "score": 0.89,
                        "chunk_id": "agentic-ai-p04-c01"
                    }
                ],
                "confidence": 0.89,
                "is_grounded": True
            }
        }
    )

    answer: str = Field(..., description="Grounded answer synthesized strictly from retrieved context.")
    context: List[ContextChunk] = Field(default_factory=list, description="Retrieved supporting chunks.")
    confidence: float = Field(..., description="Aggregate retrieval relevance confidence heuristic [0.0 - 1.0].")
    is_grounded: bool = Field(..., description="True if answer is supported by sufficient retrieved context; False if fallback.")


class ErrorResponse(BaseModel):
    """Standardized API error schema."""
    detail: str = Field(..., description="Human-readable error description")
    error_type: Optional[str] = Field(default="InternalError", description="Error classification tag")
