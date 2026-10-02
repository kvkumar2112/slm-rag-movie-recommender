"""Phase 3: user input -> SLM target profile -> embedding -> ChromaDB -> real movies."""

import re
from dataclasses import dataclass

import chromadb
import numpy as np

from .catalog import get_collection
from .config import DB_PATH
from .embeddings import embed_queries
from .genome import DEFAULT_TAGS_PER_PROFILE, GenomeIndex, load_genome_index
from .schemas import TargetProfile
from .slm import SLMGenerator, generate_target_profile, mock_slm_generate

LEADING_ARTICLES = {"the", "a", "an"}
# Weight of text similarity vs genome score in the blended ranking (1.0 = text only).
DEFAULT_ALPHA = 0.7


@dataclass
class Recommendation:
    title: str
    year: int | None
    genres: str
    similarity: float  # cosine similarity of profile and catalog text
    document: str
    score: float = 0.0  # blended ranking score (z-scored, only comparable within one query)
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


@dataclass
class Ranked:
    movie_id: int
    metadata: dict
    document: str | None
    similarity: float
    score: float


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-6)


def rank(
    collection: chromadb.Collection,
    query_embedding: list[float],
    *,
    where: dict | None = None,
    genome: GenomeIndex | None = None,
    alpha: float = DEFAULT_ALPHA,
    genome_tags: int = DEFAULT_TAGS_PER_PROFILE,
    hidden_tags: tuple[str, ...] = (),
    limit: int = 100,
    with_documents: bool = True,
) -> list[Ranked]:
    """Rank indexed movies for one query embedding: text similarity, blended with the genome
    score when a genome index is given and alpha < 1.

    Every movie passing `where` is scored, because a strong genome match can sit far down the
    text ranking. Fine at tens of thousands of movies; past that, rerank a text shortlist.
    """
    include = ["metadatas", "distances"] + (["documents"] if with_documents else [])
    result = collection.query(query_embeddings=[query_embedding], n_results=collection.count(), where=where, include=include)
    ids = [int(i) for i in result["ids"][0]]
    if not ids:
        return []
    similarity = 1 - np.array(result["distances"][0])
    documents = result["documents"][0] if with_documents else [None] * len(ids)

    if genome is not None and alpha < 1:
        by_movie = genome.score(query_embedding, genome_tags, hidden_tags)
        genome_scores = np.array([by_movie.get(movie_id, 0.0) for movie_id in ids])
        score = alpha * _zscore(similarity) + (1 - alpha) * _zscore(genome_scores)
    else:
        score = similarity

    order = np.argsort(-score)[:limit]
    return [Ranked(ids[i], result["metadatas"][0][i], documents[i], float(similarity[i]), float(score[i])) for i in order]


def recommend(
    user_input: str,
    *,
    n_results: int = 3,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    collection: chromadb.Collection | None = None,
    genome: GenomeIndex | None = None,
    generator: SLMGenerator = mock_slm_generate,
) -> tuple[TargetProfile, list[Recommendation]]:
    """Recommend real movies. Uses the genome index saved by `movie-rag build` unless one is
    passed in; without one (e.g. the JSON mock catalog) ranking is text-only."""
    if collection is None:
        collection = get_collection()
        genome = genome or load_genome_index(DB_PATH)
    profile = generate_target_profile(user_input, generator)

    # Over-fetch so dropping the movies the user already named still leaves n_results.
    ranked = rank(
        collection,
        embed_queries([profile.target_profile])[0],
        where=build_where(min_year, max_year, min_rating),
        genome=genome,
        alpha=alpha,
        limit=n_results + 10,
    )

    recommendations = []
    for r in ranked:
        if is_mentioned(r.metadata["title"], user_input):
            continue
        recommendations.append(
            Recommendation(
                title=r.metadata["title"],
                year=r.metadata.get("year"),
                genres=r.metadata["genres"],
                similarity=r.similarity,
                document=r.document,
                score=r.score,
                rating_mean=r.metadata.get("rating_mean"),
                rating_count=r.metadata.get("rating_count"),
            )
        )
        if len(recommendations) == n_results:
            break
    return profile, recommendations
