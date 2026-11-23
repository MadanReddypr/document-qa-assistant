
import json
import sqlite3
from pathlib import Path

import numpy as np


DEFAULT_DB_PATH = "data/vector_store/vectors.db"


def get_connection(database_path: str = DEFAULT_DB_PATH):
    """Create a SQLite connection and initialise the chunks table."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page INTEGER,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_document_id
        ON document_chunks(document_id)
        """
    )

    connection.commit()
    return connection


def add_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    document_id: str,
    database_path: str = DEFAULT_DB_PATH
) -> int:
    """Store document chunks and their embeddings in SQLite."""

    if not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    if not chunks:
        return 0

    prepared_rows = []
    expected_dimensions = None

    for chunk, embedding in zip(chunks, embeddings):

        if not chunk.get("text", "").strip():
            raise ValueError("Chunk text cannot be empty.")

        vector = np.asarray(embedding, dtype=float)

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Invalid embedding dimensions.")

        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding contains invalid numeric values.")

        if expected_dimensions is None:
            expected_dimensions = len(vector)
        elif len(vector) != expected_dimensions:
            raise ValueError("Embedding dimensions are inconsistent.")

        prepared_rows.append(
            (
                document_id,
                int(chunk["chunk_index"]),
                chunk.get("page"),
                chunk["text"],
                json.dumps(vector.tolist())
            )
        )

    connection = get_connection(database_path)

    try:
        connection.executemany(
            """
            INSERT INTO document_chunks
            (document_id, chunk_index, page, text, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            prepared_rows
        )
        connection.commit()
    finally:
        connection.close()

    return len(prepared_rows)


def delete_document_chunks(
    document_id: str,
    database_path: str = DEFAULT_DB_PATH
) -> int:
    """Delete all stored chunks belonging to a document."""

    connection = get_connection(database_path)

    try:
        cursor = connection.execute(
            "DELETE FROM document_chunks WHERE document_id = ?",
            (document_id,)
        )
        connection.commit()
        return cursor.rowcount
    finally:
        connection.close()


def search_similar(
    query_embedding: list[float],
    top_k: int = 3,
    database_path: str = DEFAULT_DB_PATH,
    document_id: str | None = None
) -> list[dict]:
    """Retrieve the most similar document chunks using cosine similarity."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_vector = np.asarray(query_embedding, dtype=float)

    if query_vector.ndim != 1 or query_vector.size == 0:
        raise ValueError("Invalid query embedding.")

    if not np.all(np.isfinite(query_vector)):
        raise ValueError("Query embedding contains invalid numeric values.")

    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        raise ValueError("Query embedding cannot be a zero vector.")

    connection = get_connection(database_path)

    try:
        if document_id is None:
            rows = connection.execute(
                "SELECT * FROM document_chunks"
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM document_chunks
                WHERE document_id = ?
                """,
                (document_id,)
            ).fetchall()
    finally:
        connection.close()

    results = []

    for row in rows:

        stored_vector = np.asarray(
            json.loads(row["embedding"]),
            dtype=float
        )

        if stored_vector.shape != query_vector.shape:
            continue

        stored_norm = np.linalg.norm(stored_vector)

        if stored_norm == 0:
            continue

        similarity = float(
            np.dot(query_vector, stored_vector)
            / (query_norm * stored_norm)
        )

        results.append({
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "page": row["page"],
            "text": row["text"],
            "score": similarity
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]
