import argparse
import logging
from pathlib import Path

from .config import CATALOG_PATH, TEACHER_MODEL


def cmd_build(args: argparse.Namespace) -> None:
    from .catalog import build_catalog, load_movies

    movies = load_movies(args.catalog)
    collection = build_catalog(movies)
    print(f"Indexed {collection.count()} movies into '{collection.name}'.")


def cmd_recommend(args: argparse.Namespace) -> None:
    from .pipeline import recommend

    profile, recs = recommend(args.query, n_results=args.n, min_year=args.min_year, max_year=args.max_year)
    print(f"\nRationale:      {profile.rationale}")
    print(f"Target profile: {profile.target_profile}\n")
    if not recs:
        print("No movies matched. Try widening the year range.")
    for i, rec in enumerate(recs, start=1):
        print(f"{i}. {rec.title} ({rec.year}) [{rec.genres}]  similarity={rec.similarity:.3f}")


def cmd_generate_data(args: argparse.Namespace) -> None:
    from .teacher import generate_training_data

    written = generate_training_data(args.out, args.n, args.catalog, args.model, args.seed)
    print(f"Wrote {written} training examples to {args.out}")


def cmd_enrich(args: argparse.Namespace) -> None:
    from .teacher import enrich_catalog

    enriched = enrich_catalog(args.out, args.catalog, args.model, args.force)
    print(f"Enriched {enriched} movies -> {args.out}. Rebuild with: movie-rag build --catalog {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="movie-rag", description="Generative-retrieval movie recommender")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="Phase 1: embed the catalog into ChromaDB")
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("recommend", help="Phase 3: recommend movies for a free-text request")
    p.add_argument("query", help="e.g. \"I love My Cousin Vinny and Ferris Bueller\"")
    p.add_argument("-n", type=int, default=3, help="number of recommendations")
    p.add_argument("--min-year", type=int)
    p.add_argument("--max-year", type=int)
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("generate-data", help="Phase 2: teacher-labelled ChatML data for QLoRA")
    p.add_argument("--out", type=Path, default=Path("data/training/train.jsonl"))
    p.add_argument("-n", type=int, default=200, help="number of examples (movie pairs)")
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    p.add_argument("--model", default=TEACHER_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("enrich", help="Rewrite catalog entries into rich vibe descriptions")
    p.add_argument("--out", type=Path, default=Path("data/movies_enriched.json"))
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    p.add_argument("--model", default=TEACHER_MODEL)
    p.add_argument("--force", action="store_true", help="re-enrich movies that already have a description")
    p.set_defaults(func=cmd_enrich)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
