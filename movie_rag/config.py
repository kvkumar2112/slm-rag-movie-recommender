import os
from pathlib import Path

from dotenv import load_dotenv

# Secrets (TMDB_ACCESS_TOKEN, OPENAI_API_KEY, ...) can live in a gitignored .env file.
load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "movie_catalog"

DB_PATH = Path(os.getenv("MOVIE_RAG_DB_PATH", "chroma_db"))
CATALOG_PATH = Path(__file__).parent / "data" / "movies.json"
DATA_DIR = Path(os.getenv("MOVIE_RAG_DATA_DIR", "data"))

# Long-term store: MongoDB holds the merged MovieLens + TMDb documents. Chroma is rebuilt from it.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "movie_rag")

# TMDb accepts either a v4 "API Read Access Token" (preferred, sent as a header) or a v3 API key.
TMDB_ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Teacher LLM used for synthetic training data and catalog enrichment.
TEACHER_MODEL = os.getenv("MOVIE_RAG_TEACHER_MODEL", "gpt-4o")
