
import sqlite3
import uuid

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.schemas import QuestionRequest

from backend.services.ingestion_service import ingest_document
from backend.services.qa_service import answer_question
from backend.services.vector_store import (
    DEFAULT_DB_PATH,
    delete_document_chunks
)


UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

DATABASE_PATH = DEFAULT_DB_PATH

app = FastAPI(
    title="AI Document Q&A Assistant",
    description=(
        "A Retrieval-Augmented Generation application "
        "for asking questions about uploaded documents."
    ),
    version="1.0.0"
)


def initialize_database():
    """Create the document metadata table."""

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                total_pages INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                file_path TEXT NOT NULL
            )
            """
        )

        connection.commit()

    finally:
        connection.close()


initialize_database()


@app.get("/")
def home():
    return {
        "message": "AI Document Q&A Assistant API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "AI Document Q&A Assistant"
    }


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):

    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX and TXT files are supported."
        )

    file_bytes = await file.read(MAX_FILE_SIZE + 1)

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded document is empty."
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 10 MB limit."
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / stored_filename

    try:
        file_path.write_bytes(file_bytes)

        result = await run_in_threadpool(
            ingest_document,
            str(file_path),
            DATABASE_PATH
        )

        connection = sqlite3.connect(DATABASE_PATH)

        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents
                (document_id, filename, total_pages, total_chunks, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result["document_id"],
                    filename,
                    result["total_pages"],
                    result["total_chunks"],
                    str(file_path)
                )
            )

            connection.commit()

        finally:
            connection.close()

        return result

    except (ValueError, FileNotFoundError) as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Document processing failed."
        ) from exc


@app.get("/documents")
def list_documents():

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(
            """
            SELECT document_id, filename, total_pages, total_chunks
            FROM documents
            ORDER BY filename
            """
        ).fetchall()

        return {
            "documents": [dict(row) for row in rows],
            "total_documents": len(rows)
        }

    finally:
        connection.close()


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT file_path
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,)
        ).fetchone()

    finally:
        connection.close()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        delete_document_chunks(
            document_id,
            DATABASE_PATH
        )

        connection = sqlite3.connect(DATABASE_PATH)

        try:
            connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,)
            )
            connection.commit()

        finally:
            connection.close()

        Path(row["file_path"]).unlink(missing_ok=True)

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document deletion failed."
        ) from exc


@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:
        result = answer_question(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc
