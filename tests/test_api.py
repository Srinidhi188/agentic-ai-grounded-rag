"""
Tests for API foundation endpoints and configuration validation.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import Settings


@pytest.fixture
def client():
    """Create a TestClient fixture."""
    return TestClient(app)


def test_root_endpoint(client):
    """Verify that root endpoint responds with 200 and docs link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert data["documentation"] == "/docs"


def test_health_endpoint(client):
    """Verify health endpoint returns valid schema and status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "version" in data
    assert "pinecone_configured" in data
    assert "llm_provider" in data
    assert "embedding_provider" in data


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
        # Invalid top_k <= 0
        Settings(RETRIEVAL_TOP_K=0)

    with pytest.raises(ValueError):
        # Invalid relevance_threshold > 1.0
        Settings(RELEVANCE_THRESHOLD=1.5)

    with pytest.raises(ValueError):
        # Invalid relevance_threshold < 0.0
        Settings(RELEVANCE_THRESHOLD=-0.1)

    # Valid settings
    valid_settings = Settings(RETRIEVAL_TOP_K=5, RELEVANCE_THRESHOLD=0.7)
    assert valid_settings.RETRIEVAL_TOP_K == 5
    assert valid_settings.RELEVANCE_THRESHOLD == 0.7
