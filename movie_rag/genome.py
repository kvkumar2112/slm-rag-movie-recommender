"""Genome-based scoring: rank movies by the MovieLens tag-genome tags a target profile implies.

1. Every genome tag name ("road trip", "buddy movie", ...) is embedded once.
2. A target profile's embedding is compared to them; its `k` closest non-generic tags, weighted
   by closeness, are the profile's implied genome.
3. A movie's genome score is its weighted relevance on those tags. Relevance is z-scored per
   tag, so tags that score high on every movie don't dominate.

The pipeline blends this with text similarity (see pipeline.rank). Built by `movie-rag build`
next to the Chroma index, because it covers exactly the indexed movies.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EMBEDDING_MODEL
from .embeddings import embed_documents
from .store import is_generic_tag

GENOME_FILE = "genome.npz"
DEFAULT_TAGS_PER_PROFILE = 20


@dataclass
class GenomeIndex:
    movie_ids: np.ndarray  # (n_movies,)
    scores: np.ndarray  # (n_movies, n_tags) per-tag z-scored relevance; 0 for movies without genome data
    tag_names: list[str]
    tag_embeddings: np.ndarray  # (n_tags, dim), normalized
    usable: np.ndarray  # (n_tags,) bool, False for generic tags
    embedding_model: str

    def implied_tags(self, query_embedding, k: int = DEFAULT_TAGS_PER_PROFILE, hidden: tuple[str, ...] = ()):
        """Indices and weights of the k tags closest to the query, closest first."""
        affinity = self.tag_embeddings @ np.asarray(query_embedding, dtype=np.float32)
        affinity[~self.usable] = -np.inf
        for tag in hidden:
            affinity[self.tag_names.index(tag)] = -np.inf
        top = np.argsort(-affinity)[:k]
        # Shift so the k-th tag gets ~0 weight: only relative closeness within the top k matters.
        weights = affinity[top] - affinity[top].min() + 1e-3
        return top, weights

    def score(self, query_embedding, k: int = DEFAULT_TAGS_PER_PROFILE, hidden: tuple[str, ...] = ()) -> dict[int, float]:
        top, weights = self.implied_tags(query_embedding, k, hidden)
        values = self.scores[:, top] @ weights / weights.sum()
        return dict(zip(self.movie_ids.tolist(), values.tolist()))

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            movie_ids=self.movie_ids,
            scores=self.scores.astype(np.float16),
            tag_names=np.array(self.tag_names),
            tag_embeddings=self.tag_embeddings.astype(np.float32),
            usable=self.usable,
            embedding_model=np.array(self.embedding_model),
        )

    @classmethod
    def load(cls, path: Path) -> "GenomeIndex":
        data = np.load(path)
        return cls(
            movie_ids=data["movie_ids"],
            scores=data["scores"].astype(np.float32),
            tag_names=data["tag_names"].tolist(),
            tag_embeddings=data["tag_embeddings"],
            usable=data["usable"],
            embedding_model=str(data["embedding_model"]),
        )


def build_genome_index(movie_ids: list[int], genome_dir: Path) -> GenomeIndex:
    """Genome matrix for `movie_ids` from MovieLens genome-*.csv, plus embedded tag names."""
    tags = pd.read_csv(genome_dir / "genome-tags.csv").sort_values("tagId").reset_index(drop=True)
    tag_col = {tag_id: col for col, tag_id in enumerate(tags["tagId"])}
    row = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    raw = pd.read_csv(
        genome_dir / "genome-scores.csv", dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"}
    )
    raw = raw[raw["movieId"].isin(row)]
    matrix = np.zeros((len(movie_ids), len(tags)), dtype=np.float32)
    matrix[raw["movieId"].map(row).to_numpy(), raw["tagId"].map(tag_col).to_numpy()] = raw["relevance"].to_numpy()

    has_genome = matrix.any(axis=1)
    if has_genome.any():
        mean = matrix[has_genome].mean(axis=0)
        std = matrix[has_genome].std(axis=0) + 1e-6
        matrix = np.where(has_genome[:, None], (matrix - mean) / std, 0.0).astype(np.float32)

    tag_names = tags["tag"].tolist()
    return GenomeIndex(
        movie_ids=np.array(movie_ids, dtype=np.int64),
        scores=matrix,
        tag_names=tag_names,
        tag_embeddings=np.array(embed_documents(tag_names), dtype=np.float32),
        usable=np.array([not is_generic_tag(t) for t in tag_names]),
        embedding_model=EMBEDDING_MODEL,
    )


def load_genome_index(db_path: Path) -> GenomeIndex | None:
    """The genome index saved next to the Chroma index, if there is one for the current model."""
    path = db_path / GENOME_FILE
    if not path.exists():
        return None
    index = GenomeIndex.load(path)
    return index if index.embedding_model == EMBEDDING_MODEL else None
