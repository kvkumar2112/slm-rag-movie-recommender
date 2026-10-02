"""Phase 1: build the `movie_catalog` ChromaDB collection.

Movies come either from the bundled JSON mock catalog or from MongoDB (see store.py). Chroma is
a derived index: rebuild it any time the source data or the embedding model changes.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from .config import CATALOG_PATH, COLLECTION_NAME, DB_PATH
from .embeddings import embed

# Chroma rejects very large single inserts, so documents are added in batches.
ADD_BATCH_SIZE = 1000


@dataclass
class Movie:
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    mechanics: list[str] = field(default_factory=list)
    summary: str = ""
    # Filled in by `movie-rag enrich`; richer vocabulary closes "embedding voids".
    vibe_description: str | None = None
    # From MovieLens (tags, ratings) and TMDb (keywords, tagline, credits).
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tagline: str | None = None
    directors: list[str] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)
    rating_mean: float | None = None
    rating_count: int | None = None
    tmdb_id: int | None = None

    @property
    def metadata_string(self) -> str:
        """The text that gets embedded, e.g.
        'Midnight Run (1988). Genres: Comedy, Crime. Mechanics: road trip, ex-cop, ...'

        all-MiniLM-L6-v2 truncates at 256 tokens, so the parts that describe the *feel* of a
        movie come first and credits come last.
        """
        parts = [f"{self.title} ({self.year})." if self.year else f"{self.title}."]
        if self.genres:
            parts.append(f"Genres: {', '.join(self.genres)}.")
        if self.mechanics:
            parts.append(f"Mechanics: {', '.join(self.mechanics)}.")
        if self.vibe_description:
            parts.append(f"Vibe: {self.vibe_description}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}.")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}.")
        if self.tagline:
            parts.append(f"Tagline: {self.tagline}")
        if self.summary:
            parts.append(f"Plot: {self.summary}")
        if self.directors:
            parts.append(f"Directed by {', '.join(self.directors)}.")
        if self.cast:
            parts.append(f"Starring {', '.join(self.cast)}.")
        return " ".join(parts)

    def chroma_metadata(self) -> dict:
        # Chroma metadata values must be non-null scalars, so genres become a string and
        # missing values are left out (a filter on a missing field simply doesn't match).
        metadata = {
            "movie_id": self.movie_id,
            "title": self.title,
            "year": self.year,
            "genres": ", ".join(self.genres),
            "rating_mean": self.rating_mean,
            "rating_count": self.rating_count,
            "tmdb_id": self.tmdb_id,
        }
        return {k: v for k, v in metadata.items() if v is not None}


def load_movies(path: Path = CATALOG_PATH) -> list[Movie]:
    with open(path) as f:
        return [Movie(**row) for row in json.load(f)]


def save_movies(movies: list[Movie], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([m.__dict__ for m in movies], f, indent=2, ensure_ascii=False)


def get_client(db_path: Path = DB_PATH) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(db_path))


def get_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    client = client or get_client()
    return client.get_collection(COLLECTION_NAME)


def build_catalog(movies: list[Movie], client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    """(Re)create the collection and insert every movie with its embedding and filterable metadata."""
    client = client or get_client()
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        # Embeddings are normalized, so cosine distance = 1 - cosine similarity.
        configuration={"hnsw": {"space": "cosine"}},
    )

    documents = [m.metadata_string for m in movies]
    embeddings = embed(documents, show_progress=len(documents) > 200)
    for start in range(0, len(movies), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=[str(m.movie_id) for m in movies[start:end]],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=[m.chroma_metadata() for m in movies[start:end]],
        )
    return collection
