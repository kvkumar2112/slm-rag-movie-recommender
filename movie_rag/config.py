import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Secrets (TMDB_ACCESS_TOKEN, OPENAI_API_KEY, ...) live in gitignored env files at the project
# root. Real environment variables win over both; config-local.env wins over .env.
load_dotenv(PROJECT_ROOT / "config-local.env")
load_dotenv(PROJECT_ROOT / ".env")

EMBEDDING_MODEL = os.getenv("MOVIE_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Retrieval models like BGE are trained to embed search queries with an instruction prefix
# (documents get none). MOVIE_RAG_QUERY_PREFIX overrides it; set it to "" to disable.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
QUERY_PREFIX = os.getenv("MOVIE_RAG_QUERY_PREFIX", QUERY_PREFIXES.get(EMBEDDING_MODEL, ""))
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
