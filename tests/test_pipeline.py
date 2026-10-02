import json
import random

import chromadb
import pytest

from movie_rag.catalog import build_catalog, load_movies
from movie_rag.pipeline import build_where, is_mentioned, recommend
from movie_rag.prompts import build_user_prompt
from movie_rag.teacher import random_pairs


def test_build_where_shapes():
    assert build_where() is None
    assert build_where(min_year=1990) == {"year": {"$gte": 1990}}
    assert build_where(max_year=1999) == {"year": {"$lte": 1999}}
    assert build_where(1990, 1999) == {"$and": [{"year": {"$gte": 1990}}, {"year": {"$lte": 1999}}]}
    assert build_where(min_rating=4.0) == {"rating_mean": {"$gte": 4.0}}


@pytest.mark.parametrize(
    "title, mentioned",
    [
        ("My Cousin Vinny", True),
        ("Ferris Bueller's Day Off", True),  # partial title
        ("Midnight Run", False),
        ("The Big Lebowski", False),
    ],
)
def test_is_mentioned(title, mentioned):
    assert is_mentioned(title, "I love My Cousin Vinny and Ferris Bueller") is mentioned


def test_catalog_metadata_string():
    movie = next(m for m in load_movies() if m.title == "Midnight Run")
    assert movie.metadata_string.startswith("Midnight Run (1988). Genres: Action, Comedy, Crime. Mechanics: road trip")


def test_random_pairs_are_distinct():
    movies = load_movies()
    pairs = random_pairs(movies, 50, random.Random(0))
    keys = {tuple(sorted((a.movie_id, b.movie_id))) for a, b in pairs}
    assert len(keys) == 50
    assert all(a.movie_id != b.movie_id for a, b in pairs)


def test_user_prompt_names_every_movie():
    prompt = build_user_prompt(["Heat", "Fargo", "Die Hard"], random.Random(0))
    assert "Heat, Fargo and Die Hard" in prompt


@pytest.fixture(scope="module")
def collection():
    return build_catalog(load_movies(), client=chromadb.EphemeralClient())


@pytest.mark.slow
def test_recommend_end_to_end(collection):
    profile, recs = recommend("I love My Cousin Vinny and Ferris Bueller", collection=collection)
    assert profile.target_profile
    assert len(recs) == 3
    titles = {r.title for r in recs}
    assert not titles & {"My Cousin Vinny", "Ferris Bueller's Day Off"}


@pytest.mark.slow
def test_recommend_respects_year_range(collection):
    _, recs = recommend("I love Heat", collection=collection, n_results=5, min_year=1990, max_year=1999)
    assert recs
    assert all(1990 <= r.year <= 1999 for r in recs)


@pytest.mark.slow
def test_custom_generator_is_used(collection):
    sci_fi = json.dumps(
        {
            "rationale": "The viewer loves mind-bending science fiction.",
            "target_profile": "A cyberpunk action film about a hacker who learns reality is a simulation.",
        }
    )
    _, recs = recommend("anything", collection=collection, generator=lambda _: sci_fi, n_results=1)
    assert recs[0].title == "The Matrix"
