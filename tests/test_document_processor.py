
import pytest

from backend.services.document_processor import (
    extract_document,
    chunk_documents
)


def test_extract_text_file(tmp_path):

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "This is a sample document for testing.",
        encoding="utf-8"
    )

    result = extract_document(str(file_path))

    assert len(result) == 1
    assert result[0]["text"] == (
        "This is a sample document for testing."
    )
    assert result[0]["page"] is None


def test_reject_unsupported_file(tmp_path):

    file_path = tmp_path / "sample.csv"
    file_path.write_text("name,age\nJohn,25")

    with pytest.raises(ValueError):
        extract_document(str(file_path))


def test_reject_empty_document(tmp_path):

    file_path = tmp_path / "empty.txt"
    file_path.write_text("   ")

    with pytest.raises(ValueError):
        extract_document(str(file_path))


def test_chunk_document():

    text = "A" * 2500

    pages = [
        {
            "text": text,
            "page": 1
        }
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=1000,
        overlap=200
    )

    assert len(chunks) == 3
    assert len(chunks[0]["text"]) == 1000
    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 1


def test_chunk_overlap():

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 5

    pages = [
        {
            "text": text,
            "page": 2
        }
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=50,
        overlap=10
    )

    assert len(chunks) > 1

    # The second chunk should begin with text
    # that overlaps the first chunk.
    assert chunks[0]["text"][-10:] == chunks[1]["text"][:10]


def test_invalid_chunk_size():

    pages = [{"text": "Sample text", "page": 1}]

    with pytest.raises(ValueError):
        chunk_documents(pages, chunk_size=0)


def test_invalid_overlap():

    pages = [{"text": "Sample text", "page": 1}]

    with pytest.raises(ValueError):
        chunk_documents(pages, chunk_size=100, overlap=100)
