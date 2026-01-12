"""
Tests for the Aquinas RAG API.

Run with: pytest tests/ -v
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Set test environment variables before importing
os.environ["QDRANT_URL"] = "http://test.qdrant.io"
os.environ["QDRANT_API_KEY"] = "test-api-key"
os.environ["OPENAI_API_KEY"] = "test-openai-key"

from api.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client for testing."""
    with patch("api.main.get_qdrant_client") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing."""
    with patch("api.main.embed_query") as mock_embed:
        # Return a fake embedding vector
        mock_embed.return_value = [0.1] * 1536
        yield mock_embed


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_check_success(self, client, mock_qdrant):
        """Test health check returns status when Qdrant is available."""
        # Mock collection info
        mock_info = MagicMock()
        mock_info.points_count = 11192
        mock_qdrant.get_collection.return_value = mock_info
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["points_count"] == 11192
    
    def test_health_check_degraded(self, client, mock_qdrant):
        """Test health check returns degraded when Qdrant fails."""
        mock_qdrant.get_collection.side_effect = Exception("Connection failed")
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "error" in data


class TestWorksEndpoint:
    """Tests for the /works endpoint."""
    
    def test_list_works(self, client):
        """Test listing available works."""
        response = client.get("/works")
        
        assert response.status_code == 200
        data = response.json()
        assert "works" in data
        assert len(data["works"]) == 7
        
        # Check for expected works
        abbrevs = [w["abbrev"] for w in data["works"]]
        assert "ST" in abbrevs
        assert "SCG" in abbrevs
        assert "DV" in abbrevs


class TestSearchEndpoint:
    """Tests for the /search endpoint."""
    
    def test_search_basic(self, client, mock_qdrant, mock_openai):
        """Test basic search functionality."""
        # Mock search results
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.payload = {
            "chunk_id": "ST_FP_Q1_A1",
            "work": "Summa Theologica",
            "work_abbrev": "ST",
            "location": "Prima Pars, Q1, A1",
            "title": "Whether sacred doctrine is necessary",
            "text": "It was necessary for man's salvation...",
            "source_url": None
        }
        mock_qdrant.search.return_value = [mock_result]
        
        response = client.post("/search", json={
            "query": "What is sacred doctrine?",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is sacred doctrine?"
        assert len(data["results"]) == 1
        assert data["results"][0]["work_abbrev"] == "ST"
    
    def test_search_with_work_filter(self, client, mock_qdrant, mock_openai):
        """Test search with work filter."""
        mock_qdrant.search.return_value = []
        
        response = client.post("/search", json={
            "query": "truth",
            "top_k": 5,
            "work_filter": "DV"
        })
        
        assert response.status_code == 200
        
        # Verify filter was passed to Qdrant
        call_args = mock_qdrant.search.call_args
        assert call_args.kwargs.get("query_filter") is not None
    
    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.post("/search", json={
            "query": "",
            "top_k": 5
        })
        
        # Should still work but may return empty results
        assert response.status_code in [200, 422]


class TestChatEndpoint:
    """Tests for the /chat endpoint."""
    
    def test_chat_basic(self, client, mock_qdrant, mock_openai):
        """Test basic chat functionality."""
        # Mock search results
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.payload = {
            "chunk_id": "ST_FP_Q2_A3",
            "work": "Summa Theologica",
            "work_abbrev": "ST",
            "location": "Prima Pars, Q2, A3",
            "title": "Whether God exists",
            "text": "The existence of God can be proved in five ways...",
            "source_url": None
        }
        mock_qdrant.search.return_value = [mock_result]
        
        # Mock chat completion
        with patch("api.main.generate_chat_response") as mock_chat:
            mock_chat.return_value = "According to Aquinas, God's existence can be demonstrated through five ways [1]."
            
            response = client.post("/chat", json={
                "message": "Does God exist?",
                "top_k": 5
            })
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert len(data["sources"]) == 1
    
    def test_chat_no_results(self, client, mock_qdrant, mock_openai):
        """Test chat when no sources are found."""
        mock_qdrant.search.return_value = []
        
        response = client.post("/chat", json={
            "message": "What is quantum physics?",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "couldn't find any relevant" in data["response"].lower()
        assert data["sources"] == []


class TestRequestValidation:
    """Tests for request validation."""
    
    def test_search_invalid_top_k(self, client):
        """Test search with invalid top_k type."""
        response = client.post("/search", json={
            "query": "test",
            "top_k": "invalid"
        })
        
        assert response.status_code == 422
    
    def test_chat_missing_message(self, client):
        """Test chat without message field."""
        response = client.post("/chat", json={
            "top_k": 5
        })
        
        assert response.status_code == 422
