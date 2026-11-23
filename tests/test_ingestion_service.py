
import sqlite3
import pytest

from unittest.mock import patch

from backend.services.ingestion_service import ingest_document


@patch("backend.services.ingestion_service.generate_embeddings")
def test_ingest_text_document(mock_embeddings, tmp_path):

    file_path = tmp_path / "sample.txt"
    database_path = str(tmp_path / "vectors.db")

    file_path.write_text(
        "This is a document about artificial intelligence. " * 40,
        encoding="utf-8"
    )

    # Generate one mock embedding for every input chunk.
    mock_embeddings.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts
    ]

    result = ingest_document(
        str(file_path),
        database_path
    )

    assert result["status"] == "completed"
    assert result["filename"] == "sample.txt"
    assert result["total_pages"] == 1
    assert result["total_chunks"] >= 1

    mock_embeddings.assert_called_once()


@patch("backend.services.ingestion_service.generate_embeddings")
def test_reingestion_does_not_duplicate_chunks(mock_embeddings, tmp_path):

    file_path = tmp_path / "sample.txt"
    database_path = str(tmp_path / "vectors.db")

    file_path.write_text(
        "Artificial intelligence document content. " * 40,
        encoding="utf-8"
    )

    mock_embeddings.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts
    ]

    first_result = ingest_document(
        str(file_path),
        database_path
    )

    second_result = ingest_document(
        str(file_path),
        database_path
    )

    assert first_result["document_id"] == second_result["document_id"]

    connection = sqlite3.connect(database_path)

    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()[0]
    finally:
        connection.close()

    assert count == second_result["total_chunks"]


def test_missing_document(tmp_path):

    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        ingest_document(str(file_path))


def test_unsupported_document(tmp_path):

    file_path = tmp_path / "sample.csv"
    file_path.write_text("name,age\nJohn,25")

    with pytest.raises(ValueError, match="Unsupported"):
        ingest_document(str(file_path))


def test_empty_document(tmp_path):

    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        ingest_document(str(file_path))
