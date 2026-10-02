"""MongoDB: the long-term store for merged movie documents.

One document per movie, keyed by MovieLens movieId:

    {
      "_id": 1,
      "title": "Toy Story", "year": 1995, "genres": ["Adventure", "Animation", ...],
      "links": {"imdb_id": "tt0114709", "tmdb_id": 862},
      "ratings": {"count": 215, "mean": 3.92, "bayesian_mean": 3.89},
      "user_tags": [{"tag": "pixar", "count": 3}, ...],          # MovieLens tags.csv
      "genome_tags": [{"tag": "toys", "relevance": 0.99}, ...],  # MovieLens tag genome (full datasets only)
      "tmdb": {"status": "ok", "overview": ..., "keywords": [...], "cast": [...], "directors": [...], ...},
      "vibe_description": "...",                                 # teacher LLM, via `movie-rag enrich`
      "movielens": {"dataset": "ml-latest-small", "ingested_at": ...}
    }

Each ingest step `$set`s only the fields it owns, so re-running one never wipes another's data.
"""

from collections.abc import Iterator

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import ServerSelectionTimeoutError

from .catalog import Movie
from .config import MONGO_DB, MONGO_URI
from .ingest.movielens import tag_key

# How many tags/keywords/cast members go into the embedded text.
MAX_TAGS = 15
MAX_KEYWORDS = 15
MAX_CAST = 3

# Tags that say how good or what kind of thing a movie is, rather than what it feels like.
# They match every target profile equally ("great movie", "comedy"), so they drown out the
# descriptive tags ("buddy movie", "road trip") that actually separate movies. Mongo keeps
# them; they're only left out of the embedded text.
GENERIC_TAGS = {
    # praise / opinion
    "original", "good", "great", "bad", "excellent", "awesome", "brilliant", "perfect", "genius",
    "masterpiece", "classic", "cool", "interesting", "very interesting", "very good", "overrated",
    "underrated", "boring", "boring!", "disappointing", "lame", "pointless", "stupid",
    "stupid as hell", "dumb", "cheesy", "intelligent", "clever", "suprisingly clever",
    "entertaining", "fun", "fun movie", "funny", "very funny", "funny as hell", "hilarious",
    "great movie", "good acting", "great acting", "bad acting", "exceptional acting",
    "good action", "good dialogue", "great dialogue", "good story", "good music", "great music",
    "good soundtrack", "great soundtrack", "awesome soundtrack", "notable soundtrack",
    "excellent script", "bad script", "bad plot", "no plot", "original plot", "great ending",
    "bad ending", "powerful ending", "good sequel", "bad sequel", "crappy sequel",
    "not as good as the first", "not funny", "sad but good", "so bad it's funny",
    "so bad it's good", "unintentionally funny", "hard to watch", "predictable", "plot holes",
    "bad cgi", "bad science", "good romantic comedies", "amazing", "great performances",
    # craft, with no feel attached
    "cinematography", "amazing cinematography", "great cinematography", "amazing photography",
    "beautiful", "visually appealing", "visually stunning", "visual", "special effects",
    "effects", "big budget", "storytelling", "story", "plot", "script", "dialogue", "music",
    # metadata
    "imdb top 250", "criterion", "best of 2005", "best war films", "potential oscar nom",
    "adaptation", "literary adaptation", "video game adaptation", "sequel", "sequels",
    "franchise", "remake", "pg", "pg-13", "easily confused with other movie(s) (title)",
    "watch the credits", "male nudity", "notable nudity",
    # genre labels (already in `genres`)
    "comedy", "action", "drama", "sci-fi", "science fiction", "romance", "romantic", "horror",
    "adventure", "animation", "animated", "fantasy", "crime", "thriller", "mystery",
    "documentary", "musical", "western", "war", "family", "children", "kids", "kids and family",
    "cartoon",
}
GENERIC_PREFIXES = ("oscar", "afi 100", "saturn award", "nudity", "adapted from:", "based on a book",
                    "based on book", "based on a comic", "based on comic", "based on a play",
                    "based on a tv show", "based on a video game")
GENERIC_KEYS = {tag_key(t) for t in GENERIC_TAGS}  # also catches spellings like "scifi", "sci fi"


def is_generic_tag(tag: str) -> bool:
    return tag_key(tag) in GENERIC_KEYS or tag.startswith(GENERIC_PREFIXES)


def get_db(client: MongoClient | None = None) -> Database:
    if client is None:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        try:
            client.admin.command("ping")
        except ServerSelectionTimeoutError as e:
            raise SystemExit(
                f"Can't reach MongoDB at {MONGO_URI}. Start it (brew services start mongodb-community) "
                "or set MONGO_URI."
            ) from e
    db = client[MONGO_DB]
    ensure_indexes(db)
    return db


def ensure_indexes(db: Database) -> None:
    db.movies.create_index([("ratings.count", DESCENDING)])
    db.movies.create_index([("links.tmdb_id", ASCENDING)])
    db.movies.create_index([("year", ASCENDING)])


def movie_from_doc(doc: dict) -> Movie:
    """Flatten a Mongo document into the Movie shape the vector index embeds."""
    tmdb = doc.get("tmdb") or {}
    ratings = doc.get("ratings") or {}

    # Genome tags are curated and scored, so they go first; user tags fill the rest.
    # Generic tags are dropped and spelling variants ("sci-fi", "scifi") kept once.
    tags: list[str] = []
    seen: set[str] = set()
    for tag in [g["tag"] for g in doc.get("genome_tags", [])] + [t["tag"] for t in doc.get("user_tags", [])]:
        key = tag_key(tag)
        if key not in seen and not is_generic_tag(tag):
            seen.add(key)
            tags.append(tag)

    return Movie(
        movie_id=doc["_id"],
        title=doc["title"],
        year=doc.get("year"),
        genres=doc.get("genres", []),
        tags=tags[:MAX_TAGS],
        keywords=tmdb.get("keywords", [])[:MAX_KEYWORDS],
        tagline=tmdb.get("tagline") or None,
        summary=tmdb.get("overview") or "",
        directors=tmdb.get("directors", []),
        cast=[c["name"] for c in tmdb.get("cast", [])[:MAX_CAST]],
        vibe_description=doc.get("vibe_description"),
        rating_mean=ratings.get("mean"),
        rating_count=ratings.get("count"),
        tmdb_id=(doc.get("links") or {}).get("tmdb_id"),
    )


def iter_movie_docs(db: Database, min_ratings: int = 0, limit: int | None = None) -> Iterator[dict]:
    """Movies with at least `min_ratings` ratings, most-rated first."""
    query = {"ratings.count": {"$gte": min_ratings}} if min_ratings else {}
    cursor = db.movies.find(query).sort("ratings.count", DESCENDING)
    if limit:
        cursor = cursor.limit(limit)
    return iter(cursor)


def load_movies(db: Database, min_ratings: int = 0, limit: int | None = None) -> list[Movie]:
    return [movie_from_doc(doc) for doc in iter_movie_docs(db, min_ratings, limit)]


def set_vibe_description(db: Database, movie_id: int, text: str) -> None:
    db.movies.update_one({"_id": movie_id}, {"$set": {"vibe_description": text}})
