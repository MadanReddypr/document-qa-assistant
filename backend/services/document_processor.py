
from pathlib import Path

from pypdf import PdfReader
from docx import Document


def extract_document(file_path: str) -> list[dict]:
    """
    Extract text from PDF, DOCX, and TXT documents.

    Returns a list of dictionaries containing:
    - text
    - page
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = path.suffix.lower()
    extracted_pages = []

    if extension == ".pdf":

        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                extracted_pages.append({
                    "text": text.strip(),
                    "page": page_number
                })

    elif extension == ".docx":

        document = Document(str(path))

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        if text.strip():
            extracted_pages.append({
                "text": text.strip(),
                "page": None
            })

    elif extension == ".txt":

        text = path.read_text(encoding="utf-8-sig")

        if text.strip():
            extracted_pages.append({
                "text": text.strip(),
                "page": None
            })

    else:
        raise ValueError(
            f"Unsupported document format: {extension}"
        )

    if not extracted_pages:
        raise ValueError(
            "No readable text was found in the document."
        )

    return extracted_pages


def chunk_documents(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[dict]:
    """
    Split extracted document text into overlapping chunks.

    Each chunk retains its original page number.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    chunks = []
    chunk_index = 0
    step = chunk_size - overlap

    for page_data in pages:

        text = page_data.get("text", "").strip()
        page_number = page_data.get("page")

        if not text:
            continue

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_index": chunk_index,
                    "page": page_number,
                    "text": chunk_text
                })

                chunk_index += 1

            if end == len(text):
                break

            start += step

    return chunks
