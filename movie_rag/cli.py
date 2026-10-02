import argparse
import logging
from pathlib import Path

from .config import CATALOG_PATH, DATA_DIR, TEACHER_MODEL


def load_source_movies(args: argparse.Namespace):
    """Movies from the bundled JSON mock catalog or from MongoDB, per --source."""
    if args.source == "json":
        from .catalog import load_movies

        movies = load_movies(args.catalog)
        return movies[: args.limit] if args.limit else movies

    from . import store

    return store.load_movies(store.get_db(), min_ratings=args.min_ratings, limit=args.limit)


def cmd_ingest_movielens(args: argparse.Namespace) -> None:
    from . import store
    from .ingest import movielens

    db = store.get_db()
    dataset_dir = movielens.download(args.dataset, DATA_DIR)
    count = movielens.ingest(db, dataset_dir)
    print(f"Upserted {count} movies from {args.dataset} into MongoDB '{db.name}.movies'.")


def cmd_ingest_tmdb(args: argparse.Namespace) -> None:
    from . import store
    from .ingest.tmdb import TMDbAuthError, TMDbClient, ingest

    try:
        counts = ingest(store.get_db(), TMDbClient(), limit=args.limit, workers=args.workers, refresh=args.refresh)
    except TMDbAuthError as e:
        raise SystemExit(str(e)) from e
    print(f"TMDb: {counts['ok']} fetched, {counts['not_found']} not found, {counts['failed']} failed.")


def cmd_stats(args: argparse.Namespace) -> None:
    from . import store

    movies = store.get_db().movies
    rows = {
        "movies": {},
        "with ratings": {"ratings.count": {"$gt": 0}},
        "with >= 10 ratings": {"ratings.count": {"$gte": 10}},
        "with user tags": {"user_tags.0": {"$exists": True}},
        "with genome tags": {"genome_tags.0": {"$exists": True}},
        "with a TMDb id": {"links.tmdb_id": {"$ne": None}},
        "TMDb fetched": {"tmdb.status": "ok"},
        "TMDb not found": {"tmdb.status": "not_found"},
        "with vibe description": {"vibe_description": {"$exists": True}},
    }
    for label, query in rows.items():
        print(f"{label:>24}: {movies.count_documents(query):,}")


def cmd_build(args: argparse.Namespace) -> None:
    from .catalog import build_catalog

    movies = load_source_movies(args)
    if not movies:
        raise SystemExit("No movies to index. For --source mongo, run `movie-rag ingest-movielens` first.")
    collection = build_catalog(movies)
    print(f"Indexed {collection.count()} movies into '{collection.name}'.")


def cmd_recommend(args: argparse.Namespace) -> None:
    from .pipeline import recommend

    profile, recs = recommend(
        args.query, n_results=args.n, min_year=args.min_year, max_year=args.max_year, min_rating=args.min_rating
    )
    print(f"\nRationale:      {profile.rationale}")
    print(f"Target profile: {profile.target_profile}\n")
    if not recs:
        print("No movies matched. Try loosening the filters.")
    for i, rec in enumerate(recs, start=1):
        year = f" ({rec.year})" if rec.year else ""
        rating = f"  rating={rec.rating_mean:.2f} ({rec.rating_count})" if rec.rating_mean is not None else ""
        print(f"{i}. {rec.title}{year} [{rec.genres}]  similarity={rec.similarity:.3f}{rating}")
        if args.show_docs:
            print(f"   {rec.document}\n")


def cmd_eval(args: argparse.Namespace) -> None:
    import json
    from dataclasses import asdict

    from .catalog import get_collection
    from .config import EMBEDDING_MODEL
    from .evaluation import evaluate, ground_truth, load_queries, summarize

    collection = get_collection()
    candidate_ids = {int(i) for i in collection.get(include=[])["ids"]}
    queries = load_queries()
    truth = ground_truth(queries, args.genome_dir, candidate_ids, args.min_relevance)
    if not any(truth.values()):
        raise SystemExit("No relevant movies found. Build the index from MongoDB with ml-25m data first.")
    results = evaluate(collection, queries, truth, k=args.k, n=args.n)
    summary = summarize(results)

    print(f"{'query':<22}{'relevant':>9}{f'P@{args.k}':>8}{f'R@{args.n}':>8}{'RR':>7}{'random':>8}")
    for r in results:
        print(f"{r.id:<22}{r.n_relevant:>9}{r.precision_at_k:>8.2f}{r.recall_at_n:>8.2f}{r.reciprocal_rank:>7.2f}{r.random_precision:>8.3f}")
        if args.details:
            for title, hit in r.top_k:
                print(f"{'':<4}{'✓' if hit else '·'} {title}")
    print(
        f"{'MEAN':<22}{'':>9}{summary['precision_at_k']:>8.3f}{summary['recall_at_n']:>8.3f}"
        f"{summary['mrr']:>7.3f}{summary['random_precision']:>8.3f}"
    )
    print(f"\n{len(candidate_ids)} movies indexed. Lift over random P@{args.k}: {summary['precision_at_k'] / summary['random_precision']:.1f}x")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "label": args.label,
            "embedding_model": EMBEDDING_MODEL,
            "indexed_movies": len(candidate_ids),
            "k": args.k, "n": args.n, "min_relevance": args.min_relevance,
            "summary": summary,
            "queries": [asdict(r) for r in results],
        }
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Saved to {args.out}")


def cmd_generate_data(args: argparse.Namespace) -> None:
    from .teacher import generate_training_data

    written = generate_training_data(load_source_movies(args), args.out, args.n, args.model, args.seed)
    print(f"Wrote {written} training examples to {args.out}")


def cmd_enrich(args: argparse.Namespace) -> None:
    from .teacher import enrich_movies

    movies = load_source_movies(args)
    if args.source == "mongo":
        from . import store

        db = store.get_db()
        enriched = enrich_movies(movies, lambda m: store.set_vibe_description(db, m.movie_id, m.vibe_description), args.model, args.force)
        print(f"Enriched {enriched} movies in MongoDB. Rebuild with: movie-rag build --source mongo")
    else:
        from .catalog import save_movies

        enriched = enrich_movies(movies, lambda m: None, args.model, args.force)
        save_movies(movies, args.out)
        print(f"Enriched {enriched} movies -> {args.out}. Rebuild with: movie-rag build --catalog {args.out}")


def add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["json", "mongo"], default="json", help="where movies come from (default: json)")
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="JSON catalog for --source json")
    p.add_argument("--min-ratings", type=int, default=10, help="--source mongo: skip movies with fewer ratings")
    p.add_argument("--limit", type=int, help="use at most this many movies (mongo: most-rated first)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="movie-rag", description="Generative-retrieval movie recommender")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    from .ingest.movielens import DATASETS

    p = sub.add_parser("ingest-movielens", help="Download a MovieLens dataset and upsert it into MongoDB")
    p.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default="ml-latest-small",
        help="; ".join(f"{name}: {desc}" for name, desc in DATASETS.items()),
    )
    p.set_defaults(func=cmd_ingest_movielens)

    p = sub.add_parser("ingest-tmdb", help="Fetch overview, keywords, cast and crew from TMDb into MongoDB")
    p.add_argument("--limit", type=int, help="fetch at most this many movies (most-rated first)")
    p.add_argument("--workers", type=int, default=8, help="parallel requests")
    p.add_argument("--refresh", action="store_true", help="re-fetch movies that already have TMDb data")
    p.set_defaults(func=cmd_ingest_tmdb)

    p = sub.add_parser("stats", help="Show what's in MongoDB")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("build", help="Phase 1: embed the catalog into ChromaDB")
    add_source_args(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("recommend", help="Phase 3: recommend movies for a free-text request")
    p.add_argument("query", help="e.g. \"I love My Cousin Vinny and Ferris Bueller\"")
    p.add_argument("-n", type=int, default=3, help="number of recommendations")
    p.add_argument("--min-year", type=int)
    p.add_argument("--max-year", type=int)
    p.add_argument("--min-rating", type=float, help="minimum MovieLens mean rating, 0.5-5 (mongo-built index only)")
    p.add_argument("--show-docs", action="store_true", help="print the embedded text of each result")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("eval", help="Score retrieval on the genome-labelled query set")
    p.add_argument("--genome-dir", type=Path, default=DATA_DIR / "raw" / "ml-25m", help="dir with genome-*.csv")
    p.add_argument("-k", type=int, default=10, help="precision cutoff")
    p.add_argument("-n", type=int, default=100, help="recall cutoff")
    p.add_argument("--min-relevance", type=float, default=0.5, help="genome score that makes a movie relevant")
    p.add_argument("--details", action="store_true", help="show each query's top-k with hits marked")
    p.add_argument("--label", default="", help="name for this run in the saved report")
    p.add_argument("--out", type=Path, help="save a JSON report, e.g. data/eval/baseline.json")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("generate-data", help="Phase 2: teacher-labelled ChatML data for QLoRA")
    add_source_args(p)
    p.add_argument("--out", type=Path, default=Path("data/training/train.jsonl"))
    p.add_argument("-n", type=int, default=200, help="number of examples (movie pairs)")
    p.add_argument("--model", default=TEACHER_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("enrich", help="Rewrite catalog entries into rich vibe descriptions")
    add_source_args(p)
    p.add_argument("--out", type=Path, default=Path("data/movies_enriched.json"), help="output for --source json")
    p.add_argument("--model", default=TEACHER_MODEL)
    p.add_argument("--force", action="store_true", help="re-enrich movies that already have a description")
    p.set_defaults(func=cmd_enrich)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    # httpx logs every request at INFO; keep -v output readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
