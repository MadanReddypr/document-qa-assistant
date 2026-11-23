
from unittest.mock import patch, Mock

import pytest

from backend.services.qa_service import answer_question


@patch("backend.services.qa_service.requests.post")
@patch("backend.services.qa_service.search_similar")
@patch("backend.services.qa_service.generate_embeddings")
def test_answer_question(
    mock_embeddings,
    mock_search,
    mock_post
):

    mock_embeddings.return_value = [
        [0.1, 0.2, 0.3]
    ]

    mock_search.return_value = [
        {
            "document_id": "document-1",
            "chunk_index": 0,
            "page": 2,
            "text": "FastAPI is used to build Python APIs.",
            "score": 0.95
        }
    ]

    mock_response = Mock()

    mock_response.json.return_value = {
        "response": "FastAPI is used to build Python APIs."
    }

    mock_post.return_value = mock_response

    result = answer_question(
        "Which framework is used to build Python APIs?"
    )

    assert result["status"] == "completed"
    assert "FastAPI" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page"] == 2

    mock_embeddings.assert_called_once()
    mock_search.assert_called_once()
    mock_post.assert_called_once()


@patch("backend.services.qa_service.search_similar")
@patch("backend.services.qa_service.generate_embeddings")
def test_no_relevant_documents(
    mock_embeddings,
    mock_search
):

    mock_embeddings.return_value = [
        [0.1, 0.2, 0.3]
    ]

    mock_search.return_value = []

    result = answer_question("What is machine learning?")

    assert result["status"] == "no_relevant_context"
    assert result["sources"] == []
    assert "could not find" in result["answer"].lower()


def test_empty_question():

    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question("   ")


def test_invalid_top_k():

    with pytest.raises(ValueError, match="top_k"):
        answer_question("What is Python?", top_k=0)


@patch("backend.services.qa_service.requests.post")
@patch("backend.services.qa_service.search_similar")
@patch("backend.services.qa_service.generate_embeddings")
def test_ollama_connection_error(
    mock_embeddings,
    mock_search,
    mock_post
):

    import requests

    mock_embeddings.return_value = [
        [0.1, 0.2, 0.3]
    ]

    mock_search.return_value = [
        {
            "document_id": "document-1",
            "chunk_index": 0,
            "page": 1,
            "text": "Python is a programming language.",
            "score": 0.9
        }
    ]

    mock_post.side_effect = requests.exceptions.ConnectionError()

    with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
        answer_question("What is Python?")


@patch("backend.services.qa_service.requests.post")
@patch("backend.services.qa_service.search_similar")
@patch("backend.services.qa_service.generate_embeddings")
def test_document_id_is_passed_to_search(
    mock_embeddings,
    mock_search,
    mock_post
):

    mock_embeddings.return_value = [
        [0.1, 0.2, 0.3]
    ]

    mock_search.return_value = []

    result = answer_question(
        "What is Python?",
        document_id="document-123"
    )

    assert result["status"] == "no_relevant_context"

    mock_search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        top_k=4,
        document_id="document-123"
    )
