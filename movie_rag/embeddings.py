from functools import lru_cache

from .config import EMBEDDING_MODEL, QUERY_PREFIX


@lru_cache(maxsize=2)
def get_model(name: str = EMBEDDING_MODEL):
    # Imported lazily so commands that never embed (e.g. generate-data) start fast.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_documents(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Embed catalog entries."""
    vectors = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=show_progress)
    return vectors.tolist()


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Embed search queries (target profiles), with the model's query prefix if it has one.

    Must use the same model as the catalog: vectors from different models aren't comparable.
    """
    vectors = get_model().encode([QUERY_PREFIX + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()
