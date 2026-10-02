from pathlib import Path

import chromadb
import numpy as np
import pytest

from movie_rag.catalog import COLLECTION_NAME
from movie_rag.genome import GenomeIndex, build_genome_index
from movie_rag.pipeline import rank

FIXTURES = Path(__file__).parent / "fixtures" / "movielens"


def toy_index() -> GenomeIndex:
    # Tags live in a 2-d embedding space: "road trip" along x, "zombies" along y.
    return GenomeIndex(
        movie_ids=np.array([1, 2, 3]),
        scores=np.array([[2.0, -1.0, 9.0], [-1.0, 2.0, 9.0], [0.0, 0.0, 9.0]], dtype=np.float32),
        tag_names=["road trip", "zombies", "great movie"],
        tag_embeddings=np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32),
        usable=np.array([True, True, False]),  # "great movie" is generic
        embedding_model="test",
    )


def test_implied_tags_skip_generic_and_hidden():
    index = toy_index()
    top, _ = index.implied_tags([0.71, 0.71], k=2)
    assert {index.tag_names[t] for t in top} == {"road trip", "zombies"}  # not "great movie"
    top, _ = index.implied_tags([1.0, 0.0], k=1, hidden=("road trip",))
    assert index.tag_names[top[0]] == "zombies"


def test_score_follows_closest_tag():
    scores = toy_index().score([1.0, 0.1], k=2)
    assert max(scores, key=scores.get) == 1  # the road-trip movie


def test_rank_blends_text_and_genome():
    client = chromadb.EphemeralClient()
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    col = client.create_collection(COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}})
    # Text alone prefers movie 2; the genome says the query is about road trips (movie 1).
    col.add(ids=["1", "2", "3"], embeddings=[[0.2, 1.0], [1.0, 0.0], [0.5, 0.5]], metadatas=[{"title": t} for t in "ABC"])
    query = [1.0, 0.05]
    assert [r.movie_id for r in rank(col, query, alpha=1.0, with_documents=False)][0] == 2
    assert [r.movie_id for r in rank(col, query, genome=toy_index(), alpha=0.0, genome_tags=1, with_documents=False)][0] == 1


def test_build_genome_index_from_fixtures(tmp_path):
    index = build_genome_index([1, 2, 4], FIXTURES)  # movie 4 has no genome data
    assert index.tag_names == ["virtual reality", "courtroom", "romance"]
    assert index.scores.shape == (3, 3)
    assert np.all(index.scores[2] == 0)
    # Per-tag z-scores: The Matrix is above average on "virtual reality", Vinny below.
    assert index.scores[0, 0] > 0 > index.scores[1, 0]

    index.save(tmp_path / "genome.npz")
    loaded = GenomeIndex.load(tmp_path / "genome.npz")
    assert loaded.tag_names == index.tag_names
    assert np.allclose(loaded.scores, index.scores, atol=1e-2)  # stored as float16
