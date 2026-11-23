
import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "nomic-embed-text"


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate vector embeddings for a list of text chunks
    using the Nomic embedding model through Ollama.
    """

    if not texts:
        return []

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("All input texts must be non-empty strings.")

    payload = {
        "model": MODEL_NAME,
        "input": texts
    }

    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()
        embeddings = result.get("embeddings", [])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "The number of generated embeddings does not "
                "match the number of input texts."
            )

        for embedding in embeddings:
            if (
                not isinstance(embedding, list)
                or not embedding
                or not all(
                    isinstance(value, (int, float))
                    for value in embedding
                )
            ):
                raise RuntimeError(
                    "Ollama returned an invalid embedding."
                )

        dimensions = len(embeddings[0])

        if any(len(embedding) != dimensions for embedding in embeddings):
            raise RuntimeError(
                "Embedding dimensions are inconsistent."
            )

        return embeddings

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Embedding generation timed out."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Ollama embedding API error: {exc}"
        ) from exc
