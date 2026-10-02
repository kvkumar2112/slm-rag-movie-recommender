"""Phase 1: build the `movie_catalog` ChromaDB collection from a JSON movie list."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from .config import CATALOG_PATH, COLLECTION_NAME, DB_PATH
from .embeddings import embed


@dataclass
class Movie:
    movie_id: int
    title: str
    year: int
    genres: list[str]
    mechanics: list[str] = field(default_factory=list)
    summary: str = ""
    # Filled in by `movie-rag enrich`; richer vocabulary closes "embedding voids".
    vibe_description: str | None = None

    @property
    def metadata_string(self) -> str:
        """The text that gets embedded, e.g.
        'Midnight Run (1988). Genres: Comedy, Crime. Mechanics: road trip, ex-cop, ...'
        """
        parts = [
            f"{self.title} ({self.year}).",
            f"Genres: {', '.join(self.genres)}.",
        ]
        if self.mechanics:
            parts.append(f"Mechanics: {', '.join(self.mechanics)}.")
        if self.summary:
            parts.append(f"Plot: {self.summary}")
        if self.vibe_description:
            parts.append(f"Vibe: {self.vibe_description}")
        return " ".join(parts)

    def chroma_metadata(self) -> dict:
        # Chroma metadata values must be scalars, so genres are stored as a string.
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "year": self.year,
            "genres": ", ".join(self.genres),
        }


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
    collection.add(
        ids=[str(m.movie_id) for m in movies],
        embeddings=embed(documents),
        documents=documents,
        metadatas=[m.chroma_metadata() for m in movies],
    )
    return collection
