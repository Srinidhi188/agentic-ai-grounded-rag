"""
Tests for API foundation endpoints, Pydantic validation, and /chat RAG integration.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import Settings
from app.api.routes import get_rag_workflow
from app.rag.prompts import FALLBACK_ANSWER


@pytest.fixture
def client():
    """Create a TestClient fixture."""
    return TestClient(app)


# ==================== Phase 2 Baseline Tests ====================

def test_root_endpoint(client):
    """Verify that root endpoint responds with 200 and docs link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert data["documentation"] == "/docs"


def test_health_endpoint(client):
    """Verify health endpoint returns valid schema, status, and no secrets."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "version" in data
    assert "pinecone_configured" in data
    assert "llm_provider" in data
    assert "embedding_provider" in data

    # Verify no credentials leaked
    text = response.text
    assert "key" not in text.lower() or "configured" in text.lower()


def test_info_endpoint(client):
    """Verify app metadata and parameters are exposed properly."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "knowledge_base" in data
    assert "models" in data
    assert "retrieval_parameters" in data
    assert data["knowledge_base"]["vector_store"] == "Pinecone"


def test_settings_validation():
    """Verify Pydantic settings validators catch invalid parameters."""
    with pytest.raises(ValueError):
        Settings(RETRIEVAL_TOP_K=0)

    with pytest.raises(ValueError):
        Settings(RELEVANCE_THRESHOLD=1.5)

    with pytest.raises(ValueError):
        Settings(RELEVANCE_THRESHOLD=-0.1)

    valid_settings = Settings(RETRIEVAL_TOP_K=5, RELEVANCE_THRESHOLD=0.7)
    assert valid_settings.RETRIEVAL_TOP_K == 5
    assert valid_settings.RELEVANCE_THRESHOLD == 0.7


# ==================== Phase 5 /chat Endpoint Tests ====================

def test_chat_valid_supported_query(client):
    """
    Test 1: Valid supported query returns HTTP 200, grounded answer,
    context chunks with citations, and confidence signal.
    """
    mock_workflow = MagicMock()
    mock_workflow.run.return_value = {
        "user_query": "What is Agentic AI?",
        "retrieved_documents": [
            {
                "chunk_id": "agentic-ai-p04-c01",
                "text": "Agentic AI refers to systems with proactive decision-making.",
                "source": "Ebook-Agentic-AI.pdf",
                "page": 4,
                "score": 0.8921,
                "document_name": "Agentic AI eBook",
            }
        ],
        "retrieval_scores": [0.8921],
        "context": "Excerpt from Page 4...",
        "confidence": 0.8921,
        "should_answer": True,
        "answer": "According to the provided document, Agentic AI refers to systems with proactive decision-making [Page 4].",
        "error": None,
    }

    app.dependency_overrides[get_rag_workflow] = lambda: mock_workflow
    try:
        response = client.post("/chat", json={"query": "What is Agentic AI?"})
        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "According to the provided document" in data["answer"]
        assert data["confidence"] == 0.8921
        assert data["is_grounded"] is True
        assert len(data["context"]) == 1
        assert data["context"][0]["page"] == 4
        assert data["context"][0]["chunk_id"] == "agentic-ai-p04-c01"
        assert data["context"][0]["source"] == "Ebook-Agentic-AI.pdf"
        assert data["context"][0]["score"] == 0.8921
    finally:
        app.dependency_overrides.pop(get_rag_workflow, None)


def test_chat_unsupported_query_fallback(client):
    """
    Test 2: Unsupported query returns HTTP 200 with deterministic fallback
    and is_grounded=False, preserving contract without failing.
    """
    mock_workflow = MagicMock()
    mock_workflow.run.return_value = {
        "user_query": "What is the recipe for brownies?",
        "retrieved_documents": [],
        "retrieval_scores": [],
        "context": "",
        "confidence": 0.0,
        "should_answer": False,
        "answer": FALLBACK_ANSWER,
        "error": None,
    }

    app.dependency_overrides[get_rag_workflow] = lambda: mock_workflow
    try:
        response = client.post("/chat", json={"query": "What is the recipe for brownies?"})
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == FALLBACK_ANSWER
        assert data["confidence"] == 0.0
        assert data["is_grounded"] is False
        assert data["context"] == []
    finally:
        app.dependency_overrides.pop(get_rag_workflow, None)


def test_chat_empty_query_validation(client):
    """Test 3: Empty string query rejected with HTTP 422."""
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 422


def test_chat_whitespace_only_query_validation(client):
    """Test 4: Whitespace-only query rejected with HTTP 422."""
    response = client.post("/chat", json={"query": "     "})
    assert response.status_code == 422


def test_chat_oversized_query_validation(client):
    """Test 5: Very long query exceeding 1000 characters rejected with HTTP 422."""
    oversized = "a" * 1001
    response = client.post("/chat", json={"query": oversized})
    assert response.status_code == 422


def test_chat_missing_query_field_validation(client):
    """Test 6: Missing query field rejected with HTTP 422."""
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_service_failure_handling(client):
    """
    Test 7: Unexpected exception in RAG workflow returns HTTP 500
    without leaking internal traceback details.
    """
    mock_workflow = MagicMock()
    mock_workflow.run.side_effect = RuntimeError("Internal connection failure")

    app.dependency_overrides[get_rag_workflow] = lambda: mock_workflow
    try:
        response = client.post("/chat", json={"query": "Valid query triggers failure"})
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Internal connection failure" not in data["detail"]
    finally:
        app.dependency_overrides.pop(get_rag_workflow, None)


def test_chat_response_schema_exact_match(client):
    """
    Test 8: Response conforms exactly to the documented schema keys and types.
    """
    mock_workflow = MagicMock()
    mock_workflow.run.return_value = {
        "user_query": "Explain multi-agent orchestration",
        "retrieved_documents": [
            {
                "chunk_id": "agentic-ai-p28-c02",
                "text": "Multi-agent systems coordinate autonomous agents.",
                "source": "Ebook-Agentic-AI.pdf",
                "page": 28,
                "score": 0.81,
                "document_name": "Agentic AI eBook",
            }
        ],
        "retrieval_scores": [0.81],
        "context": "Context text",
        "confidence": 0.81,
        "should_answer": True,
        "answer": "Multi-agent systems coordinate specialized agents [Page 28].",
        "error": None,
    }

    app.dependency_overrides[get_rag_workflow] = lambda: mock_workflow
    try:
        response = client.post("/chat", json={"query": "Explain multi-agent orchestration"})
        assert response.status_code == 200
        data = response.json()

        # Check required schema keys
        assert set(data.keys()) == {"answer", "context", "confidence", "is_grounded"}
        assert isinstance(data["answer"], str)
        assert isinstance(data["context"], list)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["is_grounded"], bool)

        chunk = data["context"][0]
        assert set(chunk.keys()) == {"text", "source", "page", "score", "chunk_id"}
        assert isinstance(chunk["text"], str)
        assert isinstance(chunk["source"], str)
        assert isinstance(chunk["page"], int)
        assert isinstance(chunk["score"], float)
        assert isinstance(chunk["chunk_id"], str)
    finally:
        app.dependency_overrides.pop(get_rag_workflow, None)
