"""Teacher-LLM jobs (OpenAI SDK, needs OPENAI_API_KEY).

- Phase 2: synthetic ChatML training data for QLoRA-finetuning the SLM.
- Enrichment: rewrite catalog entries into rich "vibe" descriptions so target profiles and
  catalog entries share vocabulary (fixes embedding voids).
"""

import json
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .catalog import Movie
from .config import TEACHER_MODEL
from .prompts import ENRICH_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, build_user_prompt
from .schemas import TargetProfile, VibeDescription

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _client():
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. The teacher LLM needs it: export OPENAI_API_KEY=...")
    return OpenAI()


def ask_teacher(client, system: str, user: str, schema: type[T], model: str = TEACHER_MODEL, max_attempts: int = 3) -> T:
    """Ask the teacher for JSON and validate it against `schema`, retrying on bad output."""
    for attempt in range(1, max_attempts + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        content = response.choices[0].message.content or ""
        try:
            return schema.model_validate_json(content)
        except ValidationError:
            logger.warning("Attempt %d/%d: teacher returned invalid %s", attempt, max_attempts, schema.__name__)
    raise ValueError(f"Teacher failed to return a valid {schema.__name__}")


def describe_for_teacher(movie: Movie) -> str:
    year = f" ({movie.year})" if movie.year else ""
    line = f"- {movie.title}{year}. Genres: {', '.join(movie.genres)}."
    if movie.summary:
        line += f" {movie.summary}"
    if movie.tags:
        line += f" Viewer tags: {', '.join(movie.tags[:8])}."
    return line


def random_pairs(movies: list[Movie], n: int, rng: random.Random) -> list[tuple[Movie, Movie]]:
    """Up to `n` distinct, unordered pairs of different movies."""
    max_pairs = len(movies) * (len(movies) - 1) // 2
    seen: set[tuple[int, int]] = set()
    pairs = []
    while len(pairs) < min(n, max_pairs):
        a, b = rng.sample(movies, 2)
        key = tuple(sorted((a.movie_id, b.movie_id)))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))
    return pairs


def to_chatml(user_prompt: str, profile: TargetProfile) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": profile.model_dump_json()},
        ]
    }


def generate_training_data(
    movies: list[Movie],
    out_path: Path,
    n_examples: int,
    model: str = TEACHER_MODEL,
    seed: int = 0,
) -> int:
    """Write teacher-labelled ChatML examples to `out_path` (JSONL). Returns how many were written."""
    rng = random.Random(seed)
    client = _client()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w") as f:
        for i, pair in enumerate(random_pairs(movies, n_examples, rng), start=1):
            teacher_input = "The viewer loves:\n" + "\n".join(describe_for_teacher(m) for m in pair)
            try:
                profile = ask_teacher(client, TEACHER_SYSTEM_PROMPT, teacher_input, TargetProfile, model)
            except ValueError as e:
                logger.error("Skipping %s + %s: %s", pair[0].title, pair[1].title, e)
                continue
            # The SLM only ever sees the plain user request; the teacher saw extra context.
            user_prompt = build_user_prompt([m.title for m in pair], rng)
            f.write(json.dumps(to_chatml(user_prompt, profile), ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            logger.info("[%d/%d] %s + %s", i, n_examples, pair[0].title, pair[1].title)
    return written


def enrich_movies(
    movies: list[Movie],
    on_enriched: Callable[[Movie], None],
    model: str = TEACHER_MODEL,
    force: bool = False,
) -> int:
    """Add a teacher-written `vibe_description` to each movie and hand it to `on_enriched`
    (which saves it). Returns how many were enriched."""
    client = _client()
    enriched = 0
    for movie in movies:
        if movie.vibe_description and not force:
            continue
        hints = [*movie.mechanics, *movie.tags, *movie.keywords]
        known = f" Known mechanics and tags: {', '.join(hints)}." if hints else ""
        try:
            result = ask_teacher(client, ENRICH_SYSTEM_PROMPT, describe_for_teacher(movie) + known, VibeDescription, model)
        except ValueError as e:
            logger.error("Skipping %s: %s", movie.title, e)
            continue
        movie.vibe_description = result.vibe_description
        on_enriched(movie)
        enriched += 1
        logger.info("Enriched %s", movie.title)
    return enriched
