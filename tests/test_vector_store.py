
import pytest

from backend.services.vector_store import (
    add_chunks,
    search_similar,
    delete_document_chunks
)


@pytest.fixture
def database_path(tmp_path):
    return str(tmp_path / "test_vectors.db")


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_index": 0,
            "page": 1,
            "text": "Python is a programming language."
        },
        {
            "chunk_index": 1,
            "page": 2,
            "text": "Machine learning uses data to train models."
        },
        {
            "chunk_index": 2,
            "page": 3,
            "text": "FastAPI is a Python web framework."
        }
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0]
    ]


def test_add_chunks(database_path, sample_chunks, sample_embeddings):

    count = add_chunks(
        sample_chunks,
        sample_embeddings,
        "document-1",
        database_path
    )

    assert count == 3


def test_search_similar(database_path, sample_chunks, sample_embeddings):

    add_chunks(
        sample_chunks,
        sample_embeddings,
        "document-1",
        database_path
    )

    results = search_similar(
        [1.0, 0.0, 0.0],
        top_k=2,
        database_path=database_path
    )

    assert len(results) == 2
    assert results[0]["text"] == (
        "Python is a programming language."
    )
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_returns_correct_page(database_path, sample_chunks, sample_embeddings):

    add_chunks(
        sample_chunks,
        sample_embeddings,
        "document-1",
        database_path
    )

    results = search_similar(
        [1.0, 0.0, 0.0],
        top_k=1,
        database_path=database_path
    )

    assert results[0]["page"] == 1


def test_search_by_document_id(database_path, sample_chunks, sample_embeddings):

    add_chunks(
        sample_chunks[:2],
        sample_embeddings[:2],
        "document-1",
        database_path
    )

    add_chunks(
        sample_chunks[2:],
        sample_embeddings[2:],
        "document-2",
        database_path
    )

    results = search_similar(
        [1.0, 0.0, 0.0],
        top_k=5,
        database_path=database_path,
        document_id="document-2"
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "document-2"


def test_delete_document_chunks(database_path, sample_chunks, sample_embeddings):

    add_chunks(
        sample_chunks,
        sample_embeddings,
        "document-1",
        database_path
    )

    deleted = delete_document_chunks(
        "document-1",
        database_path
    )

    assert deleted == 3

    results = search_similar(
        [1.0, 0.0, 0.0],
        database_path=database_path
    )

    assert results == []


def test_reject_mismatched_embeddings(database_path, sample_chunks):

    with pytest.raises(ValueError):
        add_chunks(
            sample_chunks,
            [[1.0, 0.0]],
            "document-1",
            database_path
        )


def test_reject_zero_query_vector(database_path):

    with pytest.raises(ValueError):
        search_similar(
            [0.0, 0.0, 0.0],
            database_path=database_path
        )


def test_reject_invalid_top_k(database_path):

    with pytest.raises(ValueError):
        search_similar(
            [1.0, 0.0, 0.0],
            top_k=0,
            database_path=database_path
        )
