from functools import lru_cache

from .config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model():
    # Imported lazily so commands that never embed (e.g. generate-data) start fast.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Embed texts with all-MiniLM-L6-v2. Catalog entries and target profiles must use this same function."""
    vectors = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=show_progress)
    return vectors.tolist()
