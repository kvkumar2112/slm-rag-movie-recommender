# Data pipeline: MovieLens + TMDb → MongoDB → ChromaDB

This guide explains how real movie data gets into the recommender and why it's built this way.
Every step is a `movie-rag` command you can run and inspect.

```
 files.grouplens.org          api.themoviedb.org
 MovieLens CSVs               /movie/{id}?append_to_response=credits,keywords
        │                              │
        │ ingest-movielens             │ ingest-tmdb
        ▼                              ▼
 ┌──────────────────────────────────────────────┐
 │ MongoDB  movie_rag.movies (source of truth)  │  one merged document per movie
 └──────────────────────────────────────────────┘
        │ build --source mongo   (flatten → text → embed)
        ▼
 ┌──────────────────────────────────────────────┐
 │ ChromaDB movie_catalog  (derived index)      │  vector + filterable metadata
 └──────────────────────────────────────────────┘
        │ recommend   (embed target profile → nearest neighbours + where-filters)
        ▼
   top-k real movies
```

## 1. Why two databases?

They do different jobs:

| | MongoDB | ChromaDB |
|---|---|---|
| Role | Long-term store, source of truth | Search index for RAG |
| Holds | Everything: nested tags with counts, cast with characters, raw TMDb fields, LLM descriptions | One embedded text string + flat scalar metadata per movie |
| Rebuilt? | Never; data accumulates and is updated in place | Any time, from Mongo, in ~1-2 min |
| Used by | Ingest, enrichment, training-data generation, a future UI | Retrieval only |

Keeping the raw data in Mongo means you can change *what you embed* (a different text layout, a
better embedding model, a filter on a new field) and just rebuild Chroma, without re-downloading
anything or re-calling TMDb.

## 2. MovieLens: ratings and community tags

[MovieLens](https://grouplens.org/datasets/movielens/) is a research dataset of real user
ratings. `movie-rag ingest-movielens --dataset <name>` downloads and unzips into `data/raw/`.

| Dataset | Size | Ratings | Movies | Tag genome |
|---|---|---|---|---|
| `ml-latest-small` (default) | 1 MB | 100k | 9.7k | no |
| `ml-25m` | 250 MB | 25M | 62k | yes |

What each file becomes in Mongo (`movie_rag/ingest/movielens.py`):

- **movies.csv → `title`, `year`, `genres`.** MovieLens writes titles as `"Matrix, The (1999)"`;
  `parse_title` splits off the year and moves the article back: `"The Matrix"`, `1999`.
- **links.csv → `links.imdb_id`, `links.tmdb_id`.** The bridge to TMDb.
- **ratings.csv → `ratings.{count, mean, bayesian_mean}`.** Individual ratings aren't stored;
  only per-movie aggregates matter for retrieval. The **Bayesian mean** blends a movie's average
  with the global average, weighted by 10 phantom ratings, so a movie with two 5-star ratings
  doesn't outrank one with 300 ratings averaging 4.2.
- **tags.csv → `user_tags: [{tag, count}]`.** Raw tags are messy (`"Sci-Fi"`, `"sci fi"`, `"scifi"`).
  They're lowercased and trimmed, spelling variants that differ only by spaces/hyphens are merged
  into the most common spelling, and each tag is counted once per user. The top 15 per movie are kept.
- **genome-scores.csv + genome-tags.csv → `genome_tags: [{tag, relevance}]`** (ml-25m only).
  The genome scores ~1,100 curated tags from 0 to 1 for every movie, so even movies nobody
  tagged get descriptive vocabulary. Tags with relevance ≥ 0.5 are kept, top 15 per movie.

## 3. TMDb: plot summaries, keywords, cast and crew

MovieLens has no plot text. [TMDb](https://www.themoviedb.org/) does, through a free API.

**Get credentials (you have to do this yourself):**
1. Create an account at themoviedb.org.
2. Settings → API → request an API key (choose "Developer", personal/educational use).
3. Copy the **API Read Access Token** (the long one) into `.env`:
   ```
   TMDB_ACCESS_TOKEN=eyJhbGciOi...
   ```

`movie-rag ingest-tmdb` makes one request per movie with a TMDb id:
`GET /3/movie/{tmdb_id}?append_to_response=credits,keywords`. Appending credits and keywords
avoids two extra calls per movie. From that it stores `tmdb.overview`, `tagline`, `keywords`,
`cast` (top 10, in billing order), `directors`, `writers`, `runtime`, `vote_average`,
`poster_path` and so on (`movie_rag/ingest/tmdb.py`).

How it behaves:
- **Most-rated first.** `--limit 500` fetches the 500 movies people care about most.
- **Resumable.** Each result is written as it arrives and only movies without a `tmdb` field are
  fetched, so Ctrl-C and re-running continues where it stopped. `--refresh` re-fetches everything.
- **Polite.** 8 parallel workers by default; HTTP 429 (rate limited) and 5xx responses are
  retried after `Retry-After`. Movies TMDb doesn't have are saved as `{"status": "not_found"}`
  so they aren't retried every run.

## 4. The MongoDB document

One document per movie, `_id` = MovieLens `movieId`:

```json
{
  "_id": 296,
  "title": "Pulp Fiction", "year": 1994, "genres": ["Comedy", "Crime", "Drama", "Thriller"],
  "links": {"imdb_id": "tt0110912", "tmdb_id": 680},
  "ratings": {"count": 307, "mean": 4.197, "bayesian_mean": 4.175},
  "user_tags": [{"tag": "nonlinear", "count": 2}, {"tag": "hit men", "count": 2}],
  "genome_tags": [],
  "tmdb": {"status": "ok", "overview": "...", "keywords": ["..."], "cast": [{"name": "...", "character": "...", "order": 0}], "directors": ["..."]},
  "vibe_description": "...",
  "movielens": {"dataset": "ml-latest-small", "ingested_at": "..."}
}
```

Design choices worth knowing:
- **Denormalized.** Everything about a movie is in one document, so building the index is a
  single scan, with no joins.
- **Field ownership.** Each step only `$set`s its own fields: MovieLens owns title/genres/
  links/ratings/tags, TMDb owns `tmdb`, `enrich` owns `vibe_description`. Re-ingesting MovieLens
  (say, upgrading to ml-25m) keeps the TMDb data and LLM descriptions you already paid for.
- **Idempotent upserts.** Every ingest can be re-run safely.
- **Indexes** on `ratings.count`, `links.tmdb_id` and `year` keep the "most-rated first" and
  "not fetched yet" queries fast.

Inspect it with `movie-rag stats`, or with MongoDB Compass / `mongosh` (`brew install mongosh`):

```
mongosh movie_rag --eval 'db.movies.findOne({title: "Pulp Fiction"})'
```

## 5. From document to vector

`movie-rag build --source mongo --min-ratings 10` reads the movies with at least 10 ratings
(obscure movies with one rating are mostly noise), flattens each into the `Movie` shape
(`store.movie_from_doc`), and embeds its `metadata_string`:

```
The Matrix (1999). Genres: Action, Sci-Fi, Thriller. Vibe: ... Tags: virtual reality, sci-fi,
cyberpunk. Keywords: simulated reality, hacker. Tagline: Welcome to the Real World. Plot: A hacker
learns the truth about his reality. Directed by Lana Wachowski, Lilly Wachowski. Starring Keanu Reeves, ...
```

The order is deliberate. `all-MiniLM-L6-v2` reads at most **256 tokens** (~180 words) and
silently ignores the rest. Text describing the *feel* of the movie (vibe, tags, keywords) comes
first, because that's the vocabulary a target profile uses; actor names come last, because
profiles never mention them.

Next to the vector, Chroma stores flat metadata for hard filters: `year`, `rating_mean`,
`rating_count`, `genres`, `tmdb_id`.

## 6. Retrieval

```
movie-rag recommend "I love My Cousin Vinny and Ferris Bueller" --min-year 1985 --max-year 1995 --min-rating 3.8 --show-docs
```

1. The SLM writes a target profile (currently a mock).
2. The profile is embedded with the same model as the catalog. This is required: vectors from
   different models aren't comparable.
3. Chroma scores the movies passing the `where` filter by cosine similarity, and that is blended
   with a tag-genome score (see [EVALUATION.md](EVALUATION.md#genome-blended-ranking)). A
   filter looks like `{"$and": [{"year": {"$gte": 1985}}, {"year": {"$lte": 1995}}, {"rating_mean": {"$gte": 3.8}}]}`.
4. Movies the user named are dropped.

`--show-docs` prints the exact text each result was matched on, which is the first thing to
look at when results seem off.

## 7. What you'll see, and how to improve it

With MovieLens alone, 1,345 of the 2,269 indexed movies have no tags, so their embedded text is
just `"Flirting With Disaster (1996). Genres: Comedy."`. Retrieval can only match on the word
"Comedy", which is why results look like a random comedy list. In order of impact:

1. **`ingest-tmdb`.** Adds a plot summary and keywords to nearly every movie. Biggest single gain.
2. **`--dataset ml-25m`.** The tag genome gives every movie curated descriptive tags.
3. **`enrich --source mongo --limit 500`** (needs `OPENAI_API_KEY`). A teacher LLM writes a
   "vibe" paragraph in the same vocabulary target profiles use. Costs money, so start with the
   most-rated movies.

After any of these, run `movie-rag build --source mongo` again: Mongo has changed, so the index has to be rebuilt.
