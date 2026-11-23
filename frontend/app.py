
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}


def get_documents():
    response = requests.get(
        f"{API_BASE_URL}/documents",
        timeout=15
    )
    response.raise_for_status()

    result = response.json()

    if isinstance(result, list):
        return result

    return result.get("documents", [])


def upload_document(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream"
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files=files,
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def ask_question(question, document_id):
    payload = {
        "question": question,
        "document_id": document_id,
        "top_k": 4
    }

    response = requests.post(
        f"{API_BASE_URL}/ask",
        json=payload,
        timeout=240
    )
    response.raise_for_status()
    return response.json()


def delete_document(document_id):
    response = requests.delete(
        f"{API_BASE_URL}/documents/{document_id}",
        timeout=30
    )
    response.raise_for_status()
    return response.json()


# Header
st.title("📚 AI Document Q&A Assistant")
st.caption(
    "Upload documents, ask questions and receive answers "
    "grounded in your own files."
)

st.divider()

# Backend status
try:
    health_response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=5
    )
    health_response.raise_for_status()
    st.success("Backend connected")
except requests.exceptions.RequestException:
    st.error(
        "Backend is unavailable. Start FastAPI using: "
        "./venv/bin/python -m uvicorn backend.main:app --reload"
    )
    st.stop()

# Upload
st.header("1. Upload a document")

uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX or TXT document",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    if st.button("Upload and process document", type="primary"):
        try:
            with st.spinner(
                "Extracting text and generating embeddings..."
            ):
                result = upload_document(uploaded_file)

            st.success("Document processed successfully!")
            st.write(f"**Filename:** {result.get('filename')}")
            st.write(f"**Pages:** {result.get('total_pages')}")
            st.write(f"**Chunks:** {result.get('total_chunks')}")

            st.rerun()

        except requests.exceptions.HTTPError as exc:
            details = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Upload failed: {details}")

        except requests.exceptions.RequestException as exc:
            st.error(f"Upload failed: {exc}")

# Fetch documents
try:
    documents = get_documents()
except requests.exceptions.RequestException as exc:
    st.error(f"Could not retrieve documents: {exc}")
    st.stop()

st.divider()
st.header("2. Select a document")

if not documents:
    st.info("No documents uploaded yet. Upload one to get started.")
    st.stop()

document_options = {
    f"{doc['filename']} ({doc['document_id'][:8]})": doc["document_id"]
    for doc in documents
}

selected_label = st.selectbox(
    "Choose a document",
    options=list(document_options.keys())
)

selected_document_id = document_options[selected_label]

# Delete document
with st.expander("Document management"):
    st.warning("Deleting a document also removes its stored text chunks.")

    if st.button("Delete selected document", type="secondary"):
        try:
            with st.spinner("Deleting document..."):
                delete_document(selected_document_id)

            st.session_state.chat_history.pop(
                selected_document_id,
                None
            )

            st.success("Document deleted successfully.")
            st.rerun()

        except requests.exceptions.HTTPError as exc:
            details = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Deletion failed: {details}")

        except requests.exceptions.RequestException as exc:
            st.error(f"Deletion failed: {exc}")

# Chat interface
st.divider()
st.header("3. Ask questions")

history = st.session_state.chat_history.setdefault(
    selected_document_id,
    []
)

for message in history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message.get("sources"):
            with st.expander("View sources"):
                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):
                    page = source.get("page")
                    page_label = (
                        f"Page {page}"
                        if page is not None
                        else "Page unavailable"
                    )

                    st.markdown(f"**Source {index} — {page_label}**")
                    st.write(source.get("text", ""))
                    st.caption(
                        f"Similarity: "
                        f"{source.get('similarity_score', 0):.3f}"
                    )

question = st.chat_input("Ask something about this document...")

if question:
    history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    try:
        with st.spinner("Searching document and generating answer..."):
            result = ask_question(
                question,
                selected_document_id
            )

        answer = result.get("answer", "No answer returned.")
        sources = result.get("sources", [])

        history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        st.rerun()

    except requests.exceptions.HTTPError as exc:
        details = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Question answering failed: {details}")

    except requests.exceptions.RequestException as exc:
        st.error(f"Could not get an answer: {exc}")

# Uploaded documents
st.divider()
st.header("4. Uploaded documents")

for document in documents:
    with st.expander(document["filename"]):
        st.write(f"**Document ID:** `{document['document_id']}`")
        st.write(f"**Pages:** {document.get('total_pages')}")
        st.write(f"**Chunks:** {document.get('total_chunks')}")
