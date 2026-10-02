"""MovieLens -> MongoDB.

Files used (https://grouplens.org/datasets/movielens/):
- movies.csv          movieId, title ("Matrix, The (1999)"), genres ("Action|Sci-Fi")
- links.csv           movieId -> imdbId, tmdbId (the bridge to TMDb)
- ratings.csv         one row per user rating -> aggregated to count / mean / Bayesian mean
- tags.csv            raw free-text user tags -> cleaned, counted per movie
- genome-tags.csv +   ~1,100 curated tags with a 0-1 relevance score per movie.
  genome-scores.csv   Only in the full datasets (e.g. ml-25m), not in ml-latest-small.
"""

import logging
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from pymongo import UpdateOne
from pymongo.database import Database

logger = logging.getLogger(__name__)

DATASETS = {
    # name: approximate download size
    "ml-latest-small": "1 MB, 100k ratings, 9.7k movies, no tag genome",
    "ml-25m": "250 MB, 25M ratings, 62k movies, includes tag genome",
}
DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/{name}.zip"

MAX_USER_TAGS = 15
MAX_GENOME_TAGS = 15
MIN_GENOME_RELEVANCE = 0.5
# Bayesian mean = (PRIOR_WEIGHT * global_mean + sum_of_ratings) / (PRIOR_WEIGHT + count):
# a movie with two 5-star ratings shouldn't outrank one with 2,000 ratings averaging 4.4.
PRIOR_WEIGHT = 10
WRITE_BATCH_SIZE = 1000

YEAR_RE = re.compile(r"\s*\((\d{4})(?:[-–]\d{0,4})?\)\s*$")
TRAILING_ARTICLE_RE = re.compile(r"^(.*), (The|A|An|Les|La|Le|L'|Il|El|Die|Der|Das)( \(.*\))?$")


def download(name: str, data_dir: Path) -> Path:
    """Download and unzip a MovieLens dataset into data_dir/raw/<name>. Skips if already present."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {', '.join(DATASETS)}")
    raw_dir = data_dir / "raw"
    target = raw_dir / name
    if (target / "movies.csv").exists():
        logger.info("%s already downloaded at %s", name, target)
        return target

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{name}.zip"
    with httpx.stream("GET", DOWNLOAD_URL.format(name=name), follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_bytes(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()
    return target


def parse_title(raw: str) -> tuple[str, int | None]:
    """'Matrix, The (1999)' -> ('The Matrix', 1999)."""
    raw = raw.strip()
    match = YEAR_RE.search(raw)
    year = int(match.group(1)) if match else None
    title = YEAR_RE.sub("", raw).strip()
    article = TRAILING_ARTICLE_RE.match(title)
    if article:
        main, art, alt = article.groups()
        sep = "" if art.endswith("'") else " "
        title = f"{art}{sep}{main}{alt or ''}"
    return title, year


def parse_genres(raw: str) -> list[str]:
    return [] if raw == "(no genres listed)" else raw.split("|")


def clean_tag(tag: str) -> str:
    tag = re.sub(r"\s+", " ", str(tag)).strip().lower()
    return tag.strip(" .,!?;:\"'")


def tag_key(tag: str) -> str:
    """Variants that differ only in spacing/hyphens ("sci-fi", "sci fi", "scifi") share a key."""
    return re.sub(r"[\s\-_]+", "", tag)


def aggregate_ratings(ratings: pd.DataFrame) -> dict[int, dict]:
    global_mean = ratings["rating"].mean()
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean", "sum"])
    stats["bayesian_mean"] = (PRIOR_WEIGHT * global_mean + stats["sum"]) / (PRIOR_WEIGHT + stats["count"])
    return {
        int(movie_id): {
            "count": int(row["count"]),
            "mean": round(float(row["mean"]), 3),
            "bayesian_mean": round(float(row["bayesian_mean"]), 3),
        }
        for movie_id, row in stats.iterrows()
    }


def aggregate_user_tags(tags: pd.DataFrame, max_tags: int = MAX_USER_TAGS) -> dict[int, list[dict]]:
    """Clean tags, merge spelling variants, count distinct users per (movie, tag), keep the top ones."""
    tags = tags.assign(tag=tags["tag"].map(clean_tag))
    tags = tags[(tags["tag"].str.len() > 1) & (tags["tag"].str.len() <= 50)]
    tags = tags.assign(key=tags["tag"].map(tag_key))
    # Display each merged tag in its most common spelling across the whole dataset.
    canonical = tags.groupby("key")["tag"].agg(lambda s: s.value_counts().index[0])
    tags = tags.assign(tag=tags["key"].map(canonical))

    counts = (
        tags.drop_duplicates(["userId", "movieId", "tag"])
        .groupby(["movieId", "tag"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["movieId", "count", "tag"], ascending=[True, False, True])
    )
    result: dict[int, list[dict]] = {}
    for row in counts.groupby("movieId").head(max_tags).itertuples(index=False):
        result.setdefault(int(row.movieId), []).append({"tag": row.tag, "count": int(row.count)})
    return result


def aggregate_genome(
    scores: pd.DataFrame,
    genome_tags: pd.DataFrame,
    max_tags: int = MAX_GENOME_TAGS,
    min_relevance: float = MIN_GENOME_RELEVANCE,
) -> dict[int, list[dict]]:
    """Top genome tags per movie by relevance."""
    scores = scores[scores["relevance"] >= min_relevance]
    scores = scores.merge(genome_tags, on="tagId").sort_values(["movieId", "relevance"], ascending=[True, False])
    result: dict[int, list[dict]] = {}
    for row in scores.groupby("movieId").head(max_tags).itertuples(index=False):
        result.setdefault(int(row.movieId), []).append({"tag": row.tag, "relevance": round(float(row.relevance), 3)})
    return result


def read_links(path: Path) -> dict[int, dict]:
    links = pd.read_csv(path, dtype={"movieId": "int64", "imdbId": "string", "tmdbId": "Int64"})
    result = {}
    for row in links.itertuples(index=False):
        result[int(row.movieId)] = {
            "imdb_id": f"tt{row.imdbId}" if pd.notna(row.imdbId) else None,
            "tmdb_id": int(row.tmdbId) if pd.notna(row.tmdbId) else None,
        }
    return result


def build_documents(dataset_dir: Path) -> list[dict]:
    """Read one MovieLens dataset directory and return one Mongo document per movie."""
    movies = pd.read_csv(dataset_dir / "movies.csv", dtype={"movieId": "int64", "title": "string", "genres": "string"})
    links = read_links(dataset_dir / "links.csv")

    logger.info("Aggregating ratings...")
    ratings = aggregate_ratings(
        pd.read_csv(dataset_dir / "ratings.csv", usecols=["movieId", "rating"], dtype={"movieId": "int32", "rating": "float32"})
    )

    logger.info("Cleaning and counting user tags...")
    user_tags = aggregate_user_tags(
        pd.read_csv(dataset_dir / "tags.csv", usecols=["userId", "movieId", "tag"], dtype={"tag": "string"}).dropna()
    )

    genome: dict[int, list[dict]] = {}
    if (dataset_dir / "genome-scores.csv").exists():
        logger.info("Reading tag genome...")
        genome = aggregate_genome(
            pd.read_csv(dataset_dir / "genome-scores.csv", dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"}),
            pd.read_csv(dataset_dir / "genome-tags.csv"),
        )

    ingested_at = datetime.now(timezone.utc)
    docs = []
    for row in movies.itertuples(index=False):
        movie_id = int(row.movieId)
        title, year = parse_title(row.title)
        docs.append(
            {
                "_id": movie_id,
                "title": title,
                "year": year,
                "genres": parse_genres(row.genres),
                "links": links.get(movie_id, {"imdb_id": None, "tmdb_id": None}),
                "ratings": ratings.get(movie_id, {"count": 0, "mean": None, "bayesian_mean": None}),
                "user_tags": user_tags.get(movie_id, []),
                "genome_tags": genome.get(movie_id, []),
                "movielens": {"dataset": dataset_dir.name, "ingested_at": ingested_at},
            }
        )
    return docs


def ingest(db: Database, dataset_dir: Path) -> int:
    """Upsert every MovieLens movie. Only MovieLens-owned fields are $set, so TMDb data and
    vibe descriptions from earlier runs survive a re-ingest."""
    docs = build_documents(dataset_dir)
    for start in range(0, len(docs), WRITE_BATCH_SIZE):
        batch = docs[start : start + WRITE_BATCH_SIZE]
        db.movies.bulk_write(
            [UpdateOne({"_id": d["_id"]}, {"$set": {k: v for k, v in d.items() if k != "_id"}}, upsert=True) for d in batch]
        )
    return len(docs)
