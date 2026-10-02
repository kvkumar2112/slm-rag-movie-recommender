"""TMDb -> MongoDB: overview, tagline, keywords, cast and crew for movies MovieLens links to.

Needs a free TMDb account and API credentials (https://www.themoviedb.org/settings/api).
One request per movie: GET /3/movie/{tmdb_id}?append_to_response=credits,keywords
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import httpx
from pymongo import DESCENDING
from pymongo.database import Database

from ..config import TMDB_ACCESS_TOKEN, TMDB_API_KEY

logger = logging.getLogger(__name__)

BASE_URL = "https://api.themoviedb.org/3"
MAX_CAST = 10
WRITER_JOBS = {"Screenplay", "Writer", "Story", "Novel", "Author"}
MAX_RETRIES = 5


class TMDbAuthError(RuntimeError):
    pass


class TMDbClient:
    def __init__(
        self,
        access_token: str | None = TMDB_ACCESS_TOKEN,
        api_key: str | None = TMDB_API_KEY,
        transport: httpx.BaseTransport | None = None,
    ):
        if not access_token and not api_key:
            raise TMDbAuthError(
                "Set TMDB_ACCESS_TOKEN (preferred) or TMDB_API_KEY. Get one free at https://www.themoviedb.org/settings/api"
            )
        headers = {"accept": "application/json"}
        params = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            params["api_key"] = api_key
        self._http = httpx.Client(base_url=BASE_URL, headers=headers, params=params, timeout=20, transport=transport)

    def _get(self, path: str, **params) -> httpx.Response:
        for attempt in range(MAX_RETRIES):
            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError:
                time.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(float(response.headers.get("Retry-After", 2**attempt)))
                continue
            if response.status_code == 401:
                raise TMDbAuthError("TMDb rejected the credentials (401). Check TMDB_ACCESS_TOKEN / TMDB_API_KEY.")
            return response
        raise RuntimeError(f"TMDb request {path} failed after {MAX_RETRIES} attempts")

    def check_auth(self) -> None:
        self._get("/authentication").raise_for_status()

    def movie(self, tmdb_id: int) -> dict | None:
        """Raw TMDb payload with credits and keywords, or None if TMDb has no such movie."""
        response = self._get(f"/movie/{tmdb_id}", append_to_response="credits,keywords", language="en-US")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def parse_movie(payload: dict) -> dict:
    """Keep the fields useful for retrieval and display."""
    credits = payload.get("credits") or {}
    crew = credits.get("crew", [])
    cast = sorted(credits.get("cast", []), key=lambda c: c.get("order", 999))[:MAX_CAST]
    return {
        "status": "ok",
        "title": payload.get("title"),
        "overview": payload.get("overview") or None,
        "tagline": payload.get("tagline") or None,
        "release_date": payload.get("release_date") or None,
        "runtime": payload.get("runtime"),
        "original_language": payload.get("original_language"),
        "vote_average": payload.get("vote_average"),
        "vote_count": payload.get("vote_count"),
        "popularity": payload.get("popularity"),
        "poster_path": payload.get("poster_path"),
        "keywords": [k["name"] for k in (payload.get("keywords") or {}).get("keywords", [])],
        "cast": [{"name": c["name"], "character": c.get("character"), "order": c.get("order")} for c in cast],
        "directors": [c["name"] for c in crew if c.get("job") == "Director"],
        "writers": sorted({c["name"] for c in crew if c.get("job") in WRITER_JOBS}),
    }


def pending_query(refresh: bool) -> dict:
    query: dict = {"links.tmdb_id": {"$ne": None}}
    if not refresh:
        query["tmdb"] = {"$exists": False}
    return query


def ingest(db: Database, client: TMDbClient, limit: int | None = None, workers: int = 8, refresh: bool = False) -> dict:
    """Fetch TMDb details for movies that don't have them yet, most-rated first.

    Results are written as they arrive, so an interrupted run resumes where it stopped.
    Movies TMDb doesn't know are stored as {"status": "not_found"} and not retried.
    """
    client.check_auth()
    cursor = db.movies.find(pending_query(refresh), {"links.tmdb_id": 1}).sort("ratings.count", DESCENDING)
    if limit:
        cursor = cursor.limit(limit)
    todo = [(doc["_id"], doc["links"]["tmdb_id"]) for doc in cursor]
    logger.info("Fetching TMDb details for %d movies", len(todo))

    counts = {"ok": 0, "not_found": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(client.movie, tmdb_id): movie_id for movie_id, tmdb_id in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            movie_id = futures[future]
            try:
                payload = future.result()
            except TMDbAuthError:
                pool.shutdown(cancel_futures=True)
                raise
            except Exception as e:
                logger.warning("Movie %s: %s", movie_id, e)
                counts["failed"] += 1
                continue
            tmdb = parse_movie(payload) if payload else {"status": "not_found"}
            tmdb["fetched_at"] = datetime.now(timezone.utc)
            db.movies.update_one({"_id": movie_id}, {"$set": {"tmdb": tmdb}})
            counts[tmdb["status"]] += 1
            if i % 100 == 0:
                logger.info("%d/%d fetched", i, len(todo))
    return counts
