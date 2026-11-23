
import hashlib
from pathlib import Path

from backend.services.document_processor import (
    extract_document,
    chunk_documents
)
from backend.services.embedding_service import generate_embeddings
from backend.services.vector_store import (
    add_chunks,
    delete_document_chunks
)


def ingest_document(
    file_path: str,
    database_path: str = "data/vector_store/vectors.db"
) -> dict:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    supported_extensions = {".pdf", ".docx", ".txt"}

    if path.suffix.lower() not in supported_extensions:
        raise ValueError("Unsupported document format.")

    document_bytes = path.read_bytes()

    if not document_bytes:
        raise ValueError("Uploaded document is empty.")

    document_id = hashlib.sha256(
        document_bytes
    ).hexdigest()

    pages = extract_document(str(path))

    chunks = chunk_documents(
        pages,
        chunk_size=1000,
        overlap=200
    )

    if not chunks:
        raise ValueError("No text chunks were generated.")

    texts = [
        f"search_document: {chunk['text']}"
        for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    # Replace existing chunks for identical document content.
    delete_document_chunks(
        document_id,
        database_path
    )

    stored_count = add_chunks(
        chunks,
        embeddings,
        document_id,
        database_path
    )

    return {
        "document_id": document_id,
        "filename": path.name,
        "total_pages": len(pages),
        "total_chunks": stored_count,
        "status": "completed"
    }
