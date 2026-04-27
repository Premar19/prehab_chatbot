"""
Backend Test Suite for the CKD Chatbot FastAPI Server
-------------------------------------------------------
Tests the /health and /chat endpoints of server.py.

All external dependencies (FAISS, sentence-transformers, Ollama,
and the metadata JSON file) are mocked so tests run quickly and
without requiring a real backend stack.

Run from the project root:
    pytest tests/ -v
"""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

# Make server.py importable when tests are run from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_mock_chunks():
    """Build a small mock chunks list matching the faiss_metadata format."""
    return [
        {
            "index": 0,
            "chunk_id": "nhs-symptoms-001",
            "page_title": "Chronic kidney disease - Symptoms",
            "section_title": "Symptoms of CKD",
            "content": "There are usually no symptoms of kidney disease in the early stages.",
            "source_url": "https://www.nhs.uk/conditions/kidney-disease/symptoms/",
            "word_count": 12,
        },
        {
            "index": 1,
            "chunk_id": "nhs-treatment-002",
            "page_title": "Chronic kidney disease - Treatment",
            "section_title": "Lifestyle changes",
            "content": "Lifestyle changes recommended include stopping smoking and eating a healthy diet.",
            "source_url": "https://www.nhs.uk/conditions/kidney-disease/treatment/",
            "word_count": 12,
        },
    ]


# Metadata JSON that the lifespan event will attempt to load
_MOCK_METADATA_JSON = json.dumps({
    "created_at": "2026-01-01T00:00:00",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "total_chunks": 2,
    "chunks": _make_mock_chunks(),
})


# ── Fixture: TestClient with all dependencies mocked ───────────
@pytest.fixture
def client():
    """
    Provides a FastAPI TestClient with FAISS, sentence-transformers,
    and the metadata file all mocked out BEFORE the lifespan event
    runs. The lifespan is allowed to complete normally against these
    mocks, so server.faiss_index, server.chunks, and server.embed_model
    are properly populated.
    """
    mock_index = MagicMock()
    mock_index.ntotal = 2

    mock_model_instance = MagicMock()
    mock_model_instance.encode = MagicMock(
        return_value=np.array([[0.1] * 384], dtype=np.float32)
    )

    with patch("faiss.read_index", return_value=mock_index), \
         patch("sentence_transformers.SentenceTransformer", return_value=mock_model_instance), \
         patch("builtins.open", mock_open(read_data=_MOCK_METADATA_JSON)):

        # Import server fresh under the patches
        import importlib
        import server as server_module
        importlib.reload(server_module)

        from fastapi.testclient import TestClient
        with TestClient(server_module.app) as test_client:
            # Expose the module so individual tests can swap in new mocks
            test_client.server_module = server_module
            yield test_client


def _mock_faiss_search_relevant():
    """FAISS search result with scores above MIN_SCORE (0.45)."""
    scores = np.array([[0.75, 0.68]], dtype=np.float32)
    indices = np.array([[0, 1]], dtype=np.int64)
    return scores, indices


def _mock_faiss_search_irrelevant():
    """FAISS search result with scores below MIN_SCORE (0.45)."""
    scores = np.array([[0.20, 0.15]], dtype=np.float32)
    indices = np.array([[0, 1]], dtype=np.int64)
    return scores, indices


# ═══════════════════════════════════════════════════════════════
# /health endpoint
# ═══════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_expected_json_structure(self, client):
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "index_loaded" in data
        assert "chunks_loaded" in data
        assert "model_loaded" in data
        assert "total_chunks" in data

    def test_health_reports_ok_status(self, client):
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "ok"
        assert data["index_loaded"] is True
        assert data["chunks_loaded"] is True
        assert data["model_loaded"] is True


# ═══════════════════════════════════════════════════════════════
# /chat endpoint — request validation
# ═══════════════════════════════════════════════════════════════

class TestChatValidation:
    """Tests that /chat properly validates incoming requests."""

    def test_chat_rejects_missing_message_field(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_rejects_empty_string_message(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 400

    def test_chat_rejects_whitespace_only_message(self, client):
        response = client.post("/chat", json={"message": "   "})
        assert response.status_code == 400


# ═══════════════════════════════════════════════════════════════
# /chat endpoint — greeting shortcut
# ═══════════════════════════════════════════════════════════════

class TestGreetingShortcut:
    """Tests that greetings are handled without calling FAISS or Ollama."""

    def test_hello_returns_greeting_response(self, client):
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 200
        data = response.json()
        assert "CKD assistant" in data["answer"]

    def test_hi_returns_greeting_response(self, client):
        response = client.post("/chat", json={"message": "hi"})
        assert response.status_code == 200
        assert "CKD assistant" in response.json()["answer"]

    def test_greeting_returns_empty_sources(self, client):
        response = client.post("/chat", json={"message": "hey"})
        data = response.json()
        assert data["sources"] == []
        assert data["retrieved_chunks"] == []

    def test_greeting_is_case_insensitive(self, client):
        response = client.post("/chat", json={"message": "HELLO"})
        assert response.status_code == 200
        assert "CKD assistant" in response.json()["answer"]

    def test_greeting_ignores_trailing_punctuation(self, client):
        response = client.post("/chat", json={"message": "hi!"})
        assert response.status_code == 200
        assert "CKD assistant" in response.json()["answer"]


# ═══════════════════════════════════════════════════════════════
# /chat endpoint — off-topic fallback
# ═══════════════════════════════════════════════════════════════

class TestOffTopicFallback:
    """Tests that low-relevance queries return the fallback message."""

    def test_off_topic_question_returns_fallback(self, client):
        mock_index = client.server_module.faiss_index
        mock_index.search.return_value = _mock_faiss_search_irrelevant()

        response = client.post(
            "/chat", json={"message": "What is the best football team in England?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "doesn't appear to be related" in data["answer"]
        assert "CKD" in data["answer"]
        assert data["sources"] == []


# ═══════════════════════════════════════════════════════════════
# /chat endpoint — normal question flow
# ═══════════════════════════════════════════════════════════════

class TestChatNormalFlow:
    """Tests the full retrieval → generation pipeline for valid CKD queries."""

    @patch("server.requests.post")
    def test_valid_question_returns_llm_answer(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "CKD is a long-term condition affecting the kidneys."
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        mock_index = client.server_module.faiss_index
        mock_index.search.return_value = _mock_faiss_search_relevant()

        response = client.post(
            "/chat", json={"message": "What are the symptoms of CKD?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "long-term condition" in data["answer"]
        assert len(data["sources"]) > 0
        assert len(data["retrieved_chunks"]) > 0

    @patch("server.requests.post")
    def test_response_schema_contains_required_fields(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "An answer."}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        mock_index = client.server_module.faiss_index
        mock_index.search.return_value = _mock_faiss_search_relevant()

        response = client.post("/chat", json={"message": "What is CKD?"})
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "retrieved_chunks" in data

    @patch("server.requests.post")
    def test_sources_are_deduplicated(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client.server_module.chunks = [
            {
                "chunk_id": "a",
                "page_title": "Page 1",
                "section_title": "Section 1",
                "content": "Content 1",
                "source_url": "https://www.nhs.uk/conditions/kidney-disease/",
                "word_count": 5,
            },
            {
                "chunk_id": "b",
                "page_title": "Page 1",
                "section_title": "Section 2",
                "content": "Content 2",
                "source_url": "https://www.nhs.uk/conditions/kidney-disease/",
                "word_count": 5,
            },
        ]

        mock_index = client.server_module.faiss_index
        mock_index.search.return_value = _mock_faiss_search_relevant()

        response = client.post("/chat", json={"message": "Tell me about CKD"})
        data = response.json()

        assert len(data["sources"]) == 1


# ═══════════════════════════════════════════════════════════════
# /chat endpoint — conversation history
# ═══════════════════════════════════════════════════════════════

class TestConversationHistory:
    """Tests that conversation history is accepted and used in FAISS search."""

    @patch("server.requests.post")
    def test_history_field_is_accepted(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        mock_index = client.server_module.faiss_index
        mock_index.search.return_value = _mock_faiss_search_relevant()

        response = client.post(
            "/chat",
            json={
                "message": "Tell me more",
                "history": [
                    {"role": "user", "text": "What are CKD treatments?"},
                    {"role": "bot", "text": "CKD has several treatments."},
                ],
            },
        )

        assert response.status_code == 200

    @patch("server.requests.post")
    def test_history_expands_search_query(self, mock_post, client):
        """The previous user message should be prepended to the current query
        before FAISS search so that follow-ups get better retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        mock_model = client.server_module.embed_model
        mock_model.encode.reset_mock()

        mock_index = client.server_module.faiss_index
        mock_index.search.return_value = _mock_faiss_search_relevant()

        client.post(
            "/chat",
            json={
                "message": "Tell me more",
                "history": [
                    {"role": "user", "text": "What are CKD treatments?"},
                ],
            },
        )

        # The embed model should have been called with the combined query text
        encode_calls = mock_model.encode.call_args_list
        first_call_text = encode_calls[0][0][0][0]
        assert "What are CKD treatments?" in first_call_text
        assert "Tell me more" in first_call_text

    @patch("server.requests.post")
    def test_empty_history_is_accepted(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        mock_index = client.server_module.faiss_index
        mock_index.search.return_value = _mock_faiss_search_relevant()

        response = client.post(
            "/chat", json={"message": "What is CKD?", "history": []}
        )

        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════
# /chat endpoint — Ollama error handling
# ═══════════════════════════════════════════════════════════════

class TestOllamaErrors:
    """Tests that failures in the Ollama call are properly handled."""

    @patch("server.requests.post")
    def test_ollama_connection_error_returns_503(self, mock_post, client):
        import requests as real_requests
        mock_post.side_effect = real_requests.ConnectionError("Connection refused")

        mock_index = client.server_module.faiss_index
        mock_index.search.return_value = _mock_faiss_search_relevant()

        response = client.post("/chat", json={"message": "What is CKD?"})
        assert response.status_code == 503
        assert "Ollama" in response.json()["detail"]


# ═══════════════════════════════════════════════════════════════
# CORS headers
# ═══════════════════════════════════════════════════════════════

class TestCORS:
    """Tests that CORS headers allow the React frontend origin."""

    def test_cors_allows_frontend_origin(self, client):
        response = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code in (200, 204)
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:5173"
        )