
from unittest.mock import patch, Mock

import pytest

from backend.services.embedding_service import generate_embeddings


@patch("backend.services.embedding_service.requests.post")
def test_generate_single_embedding(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "embeddings": [
            [0.1, 0.2, 0.3]
        ]
    }

    mock_post.return_value = mock_response

    result = generate_embeddings(["Hello world"])

    assert len(result) == 1
    assert len(result[0]) == 3
    assert result[0] == [0.1, 0.2, 0.3]


@patch("backend.services.embedding_service.requests.post")
def test_generate_multiple_embeddings(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    }

    mock_post.return_value = mock_response

    result = generate_embeddings([
        "First document chunk",
        "Second document chunk"
    ])

    assert len(result) == 2
    assert all(len(embedding) == 3 for embedding in result)


def test_empty_input():

    result = generate_embeddings([])

    assert result == []


def test_reject_empty_text():

    with pytest.raises(ValueError):
        generate_embeddings(["Hello", "   "])


@patch("backend.services.embedding_service.requests.post")
def test_invalid_embedding_count(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "embeddings": [
            [0.1, 0.2, 0.3]
        ]
    }

    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError):
        generate_embeddings([
            "First text",
            "Second text"
        ])


@patch("backend.services.embedding_service.requests.post")
def test_ollama_connection_error(mock_post):

    import requests

    mock_post.side_effect = requests.exceptions.ConnectionError()

    with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
        generate_embeddings(["Sample text"])
