# SLM-RAG Movie Recommender

A **generative-retrieval** movie recommender. Instead of matching genres, a small language
model (SLM) reasons about *why* you like the movies you named, writes an abstract
**target profile** of your ideal next watch, and that profile is matched against a vector
database of **real** movies. So it can't recommend a film that doesn't exist.

```
"I love My Cousin Vinny            ┌──────────────┐  target_profile   ┌───────────────┐
 and Ferris Bueller"  ───────────▶ │  SLM (brain) │ ────────────────▶ │ all-MiniLM-L6 │
                                   │ JSON, schema │  "brash outsider, │   embedding   │
                                   │  validated   │   mismatched pair │               │
                                   └──────────────┘   late-80s ..."   └───────┬───────┘
                                                                              │
         Top-k real movies  ◀──── year-range filter ◀──── ChromaDB `movie_catalog` (memory)
```

> The earlier Django prototype (genre filter + ratings) is preserved on the `main` branch.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Catalog → `all-MiniLM-L6-v2` embeddings → ChromaDB `movie_catalog` with `year` metadata | ✅ 34-movie mock catalog |
| 2 | Teacher LLM (GPT-4o) generates ChatML training data for QLoRA | ✅ script ready, needs `OPENAI_API_KEY` |
| 3 | Inference pipeline: input → SLM → embed → filtered vector search → top 3 | ✅ SLM is **mocked** (fixed sample output) |
| – | Catalog enrichment to close "embedding voids" | ✅ script ready, needs `OPENAI_API_KEY` |
| – | QLoRA finetuning + serving the real SLM | ⏳ next |
| – | Real MovieLens/TMDb catalog | ⏳ next |

## Setup

Requires Python 3.10+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Intel Macs, `pyproject.toml` automatically pins `torch==2.2.2` / `numpy<2` /
`transformers<4.50` (PyTorch no longer ships newer Intel-Mac builds).

## Usage

```bash
# Phase 1: embed the catalog into ./chroma_db (set MOVIE_RAG_DB_PATH to change)
movie-rag build

# Phase 3: recommend
movie-rag recommend "I love My Cousin Vinny and Ferris Bueller"
movie-rag recommend "I love My Cousin Vinny and Ferris Bueller" --min-year 1990 --max-year 1999 -n 5

# Phase 2: synthetic QLoRA training data (writes data/training/train.jsonl)
export OPENAI_API_KEY=...
movie-rag -v generate-data -n 200

# Enrich catalog entries with rich "vibe" text, then rebuild the index from it
movie-rag -v enrich --out data/movies_enriched.json
movie-rag build --catalog data/movies_enriched.json
```

Set `MOVIE_RAG_TEACHER_MODEL` to use a different teacher model.

Run tests with `pytest` (the end-to-end tests download the ~90 MB embedding model once).

## How it works

### Catalog (`movie_rag/catalog.py`, `movie_rag/data/movies.json`)
Each movie becomes one embedded `metadata_string`:

```
Midnight Run (1988). Genres: Action, Comedy, Crime. Mechanics: road trip, ex-cop bounty hunter,
buddy comedy, mafia accountant, sharp dialogue, mismatched pair bickering. Plot: ...
```

Filterable metadata (`movie_id`, `title`, `year`, `genres`) is stored next to it. The collection
uses cosine distance on normalized embeddings.

### SLM output contract (`movie_rag/schemas.py`, `movie_rag/slm.py`)
The SLM must emit `{"rationale": ..., "target_profile": ...}`. Output is validated with Pydantic;
JSON wrapped in prose or markdown fences is recovered, and invalid output (e.g. truncated
mid-sentence) triggers a retry. To plug in the real model, pass any `str -> str` callable as
`generator=` to `recommend()`.

### Training data (`movie_rag/teacher.py`, `movie_rag/prompts.py`)
Random distinct movie pairs are sent to the teacher with title, year, genres and plot. The
teacher's validated JSON becomes the assistant turn; the user turn is a plain natural-language
request ("I love X and Y. What should I watch next?") using the same templates the SLM will see
at inference. Output is Hugging Face ChatML JSONL:

```json
{"messages": [{"role": "user", "content": "I love Heat and Fargo."},
              {"role": "assistant", "content": "{\"rationale\": \"...\", \"target_profile\": \"...\"}"}]}
```

### Inference (`movie_rag/pipeline.py`)
- **Year filter.** `--min-year/--max-year` become a Chroma `where` clause. Chroma allows only
  one operator per field, so a closed range is `{"$and": [{"year": {"$gte": a}}, {"year": {"$lte": b}}]}`.
  `{"year": {"$gte": a, "$lte": b}}` raises an error.
- **No echoing inputs.** Movies the user named (including partial titles like "Ferris Bueller")
  are dropped from results.

## Known issues to work on
- **Embedding voids.** With the short mock catalog text, the mock profile ("mismatched pair,
  bickering dialogue") ranks *The Big Lebowski* above *Midnight Run*. `movie-rag enrich` rewrites
  entries in the same vocabulary target profiles use; re-run the comparison after enriching.
- The SLM is a mock that returns the same profile for every input.
