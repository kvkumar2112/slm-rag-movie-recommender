"""Phase 3: user input -> SLM target profile -> embedding -> ChromaDB -> real movies."""

import re
from dataclasses import dataclass

import chromadb

from .catalog import get_collection
from .embeddings import embed_queries
from .schemas import TargetProfile
from .slm import SLMGenerator, generate_target_profile, mock_slm_generate

LEADING_ARTICLES = {"the", "a", "an"}


@dataclass
class Recommendation:
    title: str
    year: int | None
    genres: str
    similarity: float
    document: str
    rating_mean: float | None = None
    rating_count: int | None = None


def build_where(
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
) -> dict | None:
    """Build a Chroma `where` clause from hard filters.

    Chroma allows only one operator per field expression, so a closed range has to be an
    `$and` of two conditions rather than {"year": {"$gte": a, "$lte": b}}.
    `min_rating` uses the MovieLens mean rating (0.5-5), so it only matches Mongo-built catalogs.
    """
    conditions = []
    if min_year is not None:
        conditions.append({"year": {"$gte": min_year}})
    if max_year is not None:
        conditions.append({"year": {"$lte": max_year}})
    if min_rating is not None:
        conditions.append({"rating_mean": {"$gte": min_rating}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _normalize(text: str) -> str:
    text = text.lower().replace("'s", "")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def is_mentioned(title: str, user_input: str) -> bool:
    """Whether the user named this movie, so we don't recommend it back to them.

    Matches the full title or, for longer titles, its first two significant words, so
    "Ferris Bueller" matches "Ferris Bueller's Day Off".
    """
    haystack = f" {_normalize(user_input)} "
    words = _normalize(title).split()
    if f" {' '.join(words)} " in haystack:
        return True
    if words and words[0] in LEADING_ARTICLES:
        words = words[1:]
    return len(words) > 2 and f" {' '.join(words[:2])} " in haystack


def recommend(
    user_input: str,
    *,
    n_results: int = 3,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    collection: chromadb.Collection | None = None,
    generator: SLMGenerator = mock_slm_generate,
) -> tuple[TargetProfile, list[Recommendation]]:
    collection = collection or get_collection()
    profile = generate_target_profile(user_input, generator)

    # Over-fetch so dropping the movies the user already named still leaves n_results.
    result = collection.query(
        query_embeddings=embed_queries([profile.target_profile]),
        n_results=min(n_results + 10, collection.count()),
        where=build_where(min_year, max_year, min_rating),
    )

    recommendations = []
    for metadata, distance, document in zip(
        result["metadatas"][0], result["distances"][0], result["documents"][0]
    ):
        if is_mentioned(metadata["title"], user_input):
            continue
        recommendations.append(
            Recommendation(
                title=metadata["title"],
                year=metadata.get("year"),
                genres=metadata["genres"],
                similarity=1 - distance,
                document=document,
                rating_mean=metadata.get("rating_mean"),
                rating_count=metadata.get("rating_count"),
            )
        )
        if len(recommendations) == n_results:
            break
    return profile, recommendations
