from pathlib import Path

import uuid

import httpx
import pytest
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from movie_rag import store
from movie_rag.config import MONGO_URI
from movie_rag.ingest import movielens, tmdb

FIXTURES = Path(__file__).parent / "fixtures" / "movielens"


@pytest.fixture
def db():
    """A throwaway database on the local MongoDB, dropped after the test."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        pytest.skip(f"MongoDB not reachable at {MONGO_URI}")
    name = f"movie_rag_test_{uuid.uuid4().hex[:8]}"
    db = client[name]
    store.ensure_indexes(db)
    yield db
    client.drop_database(name)
    client.close()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Matrix, The (1999)", ("The Matrix", 1999)),
        ("My Cousin Vinny (1992)", ("My Cousin Vinny", 1992)),
        ("Fabuleux destin d'Amélie Poulain, Le (Amelie) (2001)", ("Le Fabuleux destin d'Amélie Poulain (Amelie)", 2001)),
        ("Avventura, L' (1960)", ("L'Avventura", 1960)),
        ("Paris, Texas (1984)", ("Paris, Texas", 1984)),
        ("Babylon 5", ("Babylon 5", None)),
        ("Fawlty Towers (1975-1979)", ("Fawlty Towers", 1975)),
    ],
)
def test_parse_title(raw, expected):
    assert movielens.parse_title(raw) == expected


def test_build_documents():
    docs = {d["_id"]: d for d in movielens.build_documents(FIXTURES)}
    matrix = docs[1]

    assert matrix["title"] == "The Matrix"
    assert matrix["genres"] == ["Action", "Sci-Fi", "Thriller"]
    assert matrix["links"] == {"imdb_id": "tt0133093", "tmdb_id": 603}
    assert matrix["ratings"]["count"] == 3
    assert matrix["ratings"]["mean"] == 4.5
    # Bayesian mean pulls a 3-rating average toward the global mean (4.25).
    assert 4.25 < matrix["ratings"]["bayesian_mean"] < 4.5
    # "Sci-Fi", "sci fi", "scifi", "sci-fi" merge into one tag counted once per user.
    assert matrix["user_tags"] == [{"tag": "sci-fi", "count": 3}, {"tag": "cyberpunk", "count": 2}]
    assert matrix["genome_tags"] == [{"tag": "virtual reality", "relevance": 0.98}]

    assert docs[2]["user_tags"] == [{"tag": "courtroom", "count": 1}]  # 1-char tag "x" dropped
    assert docs[4]["genres"] == []
    assert docs[4]["links"]["tmdb_id"] is None
    assert docs[4]["ratings"]["count"] == 0


def test_reingest_keeps_tmdb_and_vibe(db):
    movielens.ingest(db, FIXTURES)
    db.movies.update_one({"_id": 1}, {"$set": {"tmdb": {"status": "ok"}, "vibe_description": "neon dread"}})

    assert movielens.ingest(db, FIXTURES) == 4
    doc = db.movies.find_one({"_id": 1})
    assert doc["tmdb"] == {"status": "ok"}
    assert doc["vibe_description"] == "neon dread"


TMDB_MATRIX = {
    "id": 603,
    "title": "The Matrix",
    "overview": "A hacker learns the truth about his reality.",
    "tagline": "Welcome to the Real World.",
    "release_date": "1999-03-30",
    "runtime": 136,
    "original_language": "en",
    "vote_average": 8.2,
    "vote_count": 25000,
    "popularity": 80.1,
    "poster_path": "/matrix.jpg",
    "keywords": {"keywords": [{"id": 1, "name": "simulated reality"}, {"id": 2, "name": "hacker"}]},
    "credits": {
        "cast": [
            {"name": "Carrie-Anne Moss", "character": "Trinity", "order": 2},
            {"name": "Keanu Reeves", "character": "Neo", "order": 0},
            {"name": "Laurence Fishburne", "character": "Morpheus", "order": 1},
        ],
        "crew": [
            {"name": "Lana Wachowski", "job": "Director"},
            {"name": "Lilly Wachowski", "job": "Director"},
            {"name": "Lana Wachowski", "job": "Writer"},
            {"name": "Bill Pope", "job": "Director of Photography"},
        ],
    },
}


def test_parse_movie():
    parsed = tmdb.parse_movie(TMDB_MATRIX)
    assert parsed["keywords"] == ["simulated reality", "hacker"]
    assert [c["name"] for c in parsed["cast"]] == ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"]
    assert parsed["directors"] == ["Lana Wachowski", "Lilly Wachowski"]
    assert parsed["writers"] == ["Lana Wachowski"]


def fake_tmdb(status_for_auth: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.url.path == "/3/authentication":
            return httpx.Response(status_for_auth, json={"success": status_for_auth == 200})
        if request.url.path == "/3/movie/603":
            assert request.url.params["append_to_response"] == "credits,keywords"
            return httpx.Response(200, json=TMDB_MATRIX)
        return httpx.Response(404, json={"status_message": "not found"})

    return httpx.MockTransport(handler)


def test_tmdb_ingest_is_resumable(db):
    movielens.ingest(db, FIXTURES)
    client = tmdb.TMDbClient(access_token="test-token", transport=fake_tmdb())

    # Movie 4 has no TMDb id, so only 3 are fetched; only 603 exists in the fake API.
    assert tmdb.ingest(db, client, workers=2) == {"ok": 1, "not_found": 2, "failed": 0}
    assert db.movies.find_one({"_id": 1})["tmdb"]["tagline"] == "Welcome to the Real World."
    assert db.movies.find_one({"_id": 2})["tmdb"]["status"] == "not_found"
    # A second run has nothing left to do.
    assert tmdb.ingest(db, client) == {"ok": 0, "not_found": 0, "failed": 0}


def test_tmdb_bad_credentials(db):
    client = tmdb.TMDbClient(access_token="test-token", transport=fake_tmdb(status_for_auth=401))
    with pytest.raises(tmdb.TMDbAuthError):
        tmdb.ingest(db, client)


def test_tmdb_requires_credentials():
    with pytest.raises(tmdb.TMDbAuthError):
        tmdb.TMDbClient(access_token=None, api_key=None)


def test_movie_from_doc_merges_sources(db):
    movielens.ingest(db, FIXTURES)
    db.movies.update_one({"_id": 1}, {"$set": {"tmdb": tmdb.parse_movie(TMDB_MATRIX)}})

    movie = store.movie_from_doc(db.movies.find_one({"_id": 1}))
    # Genome first, then user tags; "sci-fi" is a genre label, so it's dropped.
    assert movie.tags == ["virtual reality", "cyberpunk"]
    assert movie.cast == ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"]
    assert movie.tmdb_id == 603
    text = movie.metadata_string
    assert text.startswith("The Matrix (1999). Genres: Action, Sci-Fi, Thriller. Tags: virtual reality, cyberpunk.")
    assert "Plot: A hacker learns the truth" in text
    assert text.endswith("Starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss.")


def test_load_movies_filters_and_orders_by_rating_count(db):
    movielens.ingest(db, FIXTURES)
    assert [m.title for m in store.load_movies(db, min_ratings=2)] == ["The Matrix", "My Cousin Vinny"]
    assert len(store.load_movies(db)) == 4


@pytest.mark.parametrize(
    "tag, generic",
    [
        ("great acting", True),
        ("oscar (best picture)", True),
        ("based on a book", True),
        ("scifi", True),  # spelling variant of the "sci-fi" genre label
        ("nudity (topless - brief)", True),
        ("buddy movie", False),
        ("good versus evil", False),
        ("based on a true story", False),
        ("dialogue driven", False),
    ],
)
def test_is_generic_tag(tag, generic):
    assert store.is_generic_tag(tag) is generic


def test_movie_from_doc_merges_tag_spellings():
    doc = {
        "_id": 1, "title": "X", "genres": [],
        "genome_tags": [{"tag": "road trip", "relevance": 0.9}, {"tag": "great movie", "relevance": 0.8}],
        "user_tags": [{"tag": "road-trip", "count": 3}, {"tag": "deadpan", "count": 1}],
    }
    assert store.movie_from_doc(doc).tags == ["road trip", "deadpan"]
