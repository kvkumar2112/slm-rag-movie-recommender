"""Retrieval evaluation: does a target profile pull up the right real movies?

Each query in data/eval_queries.json is a target profile (written the way the SLM writes them)
plus 1-2 MovieLens tag-genome "anchor" tags. Ground truth comes from the genome, not from
anyone's taste: a movie is relevant if the genome scores *every* anchor tag >= min_relevance.
Query text never contains its anchor words (see tests), so a high score means the retriever
understood the description rather than matched a keyword.

This tests the retrieval stage only (profile -> embed -> rank). The SLM is bypassed.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import pandas as pd

from .embeddings import embed_queries
from .genome import DEFAULT_TAGS_PER_PROFILE, GenomeIndex
from .pipeline import rank

EVAL_QUERIES_PATH = Path(__file__).parent / "data" / "eval_queries.json"
MIN_RELEVANCE = 0.5


@dataclass
class EvalQuery:
    id: str
    profile: str
    anchor_tags: list[str]


@dataclass
class QueryResult:
    id: str
    n_relevant: int
    precision_at_k: float
    recall_at_n: float
    reciprocal_rank: float
    random_precision: float  # expected precision@k of a random ranking
    top_k: list[tuple[str, bool]] = field(default_factory=list)  # (title, is_relevant)


def load_queries(path: Path = EVAL_QUERIES_PATH) -> list[EvalQuery]:
    with open(path) as f:
        return [EvalQuery(**row) for row in json.load(f)]


def ground_truth(
    queries: list[EvalQuery],
    genome_dir: Path,
    candidate_ids: set[int],
    min_relevance: float = MIN_RELEVANCE,
) -> dict[str, set[int]]:
    """Relevant movie ids per query, restricted to the movies that are in the index."""
    tags = pd.read_csv(genome_dir / "genome-tags.csv")
    tag_ids = dict(zip(tags["tag"], tags["tagId"]))
    missing = sorted({t for q in queries for t in q.anchor_tags if t not in tag_ids})
    if missing:
        raise ValueError(f"Anchor tags not in the tag genome: {missing}")

    needed = {tag_ids[t] for q in queries for t in q.anchor_tags}
    scores = pd.read_csv(
        genome_dir / "genome-scores.csv", dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"}
    )
    scores = scores[scores["tagId"].isin(needed) & scores["movieId"].isin(candidate_ids)]
    by_tag = scores.pivot(index="movieId", columns="tagId", values="relevance")

    truth = {}
    for q in queries:
        weakest = by_tag[[tag_ids[t] for t in q.anchor_tags]].min(axis=1)
        truth[q.id] = {int(movie_id) for movie_id in weakest[weakest >= min_relevance].index}
    return truth


def precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return sum(1 for movie_id in ranked[:k] if movie_id in relevant) / k


def recall_at_n(ranked: list[int], relevant: set[int], n: int) -> float:
    return sum(1 for movie_id in ranked[:n] if movie_id in relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    return next((1 / rank for rank, movie_id in enumerate(ranked, start=1) if movie_id in relevant), 0.0)


def evaluate(
    collection: chromadb.Collection,
    queries: list[EvalQuery],
    truth: dict[str, set[int]],
    k: int = 10,
    n: int = 100,
    genome: GenomeIndex | None = None,
    alpha: float = 1.0,
    genome_tags: int = DEFAULT_TAGS_PER_PROFILE,
    hide_anchors: bool = False,
) -> list[QueryResult]:
    """Score each query with the same ranking the recommender uses.

    hide_anchors=True removes each query's anchor tags from genome scoring. The ground truth is
    defined by those tags, so this is the leak-free estimate for genome-based ranking.
    """
    total = collection.count()
    results = []
    for q, embedding in zip(queries, embed_queries([q.profile for q in queries])):
        ranked_movies = rank(
            collection,
            embedding,
            genome=genome,
            alpha=alpha,
            genome_tags=genome_tags,
            hidden_tags=tuple(q.anchor_tags) if hide_anchors else (),
            limit=n,
            with_documents=False,
        )
        ranked = [r.movie_id for r in ranked_movies]
        relevant = truth[q.id]
        results.append(
            QueryResult(
                id=q.id,
                n_relevant=len(relevant),
                precision_at_k=precision_at_k(ranked, relevant, k),
                recall_at_n=recall_at_n(ranked, relevant, n),
                reciprocal_rank=reciprocal_rank(ranked, relevant),
                random_precision=len(relevant) / total,
                top_k=[(r.metadata["title"], r.movie_id in relevant) for r in ranked_movies[:k]],
            )
        )
    return results


def summarize(results: list[QueryResult]) -> dict[str, float]:
    def mean(values):
        return sum(values) / len(values)

    return {
        "precision_at_k": mean([r.precision_at_k for r in results]),
        "recall_at_n": mean([r.recall_at_n for r in results]),
        "mrr": mean([r.reciprocal_rank for r in results]),
        "random_precision": mean([r.random_precision for r in results]),
    }
