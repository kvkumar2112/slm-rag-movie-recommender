import os
from pathlib import Path

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "movie_catalog"

DB_PATH = Path(os.getenv("MOVIE_RAG_DB_PATH", "chroma_db"))
CATALOG_PATH = Path(__file__).parent / "data" / "movies.json"

# Teacher LLM used for synthetic training data and catalog enrichment.
TEACHER_MODEL = os.getenv("MOVIE_RAG_TEACHER_MODEL", "gpt-4o")
