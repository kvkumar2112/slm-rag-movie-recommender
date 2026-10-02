"""Prompts shared by training-data generation, enrichment and inference.

The user-message templates are used both when generating training data and at inference
time, so the finetuned SLM sees the same shape of input it was trained on.
"""

import random

TEACHER_SYSTEM_PROMPT = """\
You are a film analyst who explains *why* someone loves a set of movies and describes the \
ideal next watch for them.

You will be given the movies a viewer loves. Find what genuinely connects them: shared \
story mechanics, character archetypes, tropes, tone, pacing, dialogue style and era \
texture. Look past surface genre labels.

Respond with ONLY a JSON object, no prose and no markdown fences, with exactly these keys:

{
  "rationale": "<one sentence naming the thematic bridge between the input movies>",
  "target_profile": "<3-5 sentences describing the perfect next movie>"
}

Rules for "target_profile":
- NEVER name a real movie, director, actor, character or franchise. Describe, don't cite.
- Write it as if describing a movie that exists: its premise shape, protagonist archetype, \
central mechanics (e.g. "mismatched pair forced on a road trip"), tone (e.g. "warm, \
quick-witted, low-stakes chaos"), and era vibe (e.g. "late-1980s American studio comedy").
- Use concrete, searchable vocabulary: tropes, settings, character types, dialogue style. \
Avoid empty praise such as "a great film" or "a must-watch".
"""

ENRICH_SYSTEM_PROMPT = """\
You write rich, descriptive catalog entries for a movie search engine. The search engine \
matches abstract "target profiles" (descriptions of mechanics, tropes, tone and era vibe) \
against these entries using sentence embeddings, so dry plot synopses match poorly.

Given one movie, respond with ONLY a JSON object:

{"vibe_description": "<4-6 sentences>"}

The description must cover: protagonist archetype and their flaw or edge; core story \
mechanics and tropes; tone and humour style; pacing; setting and era texture (decade, \
place, culture); and the feeling a viewer leaves with. Use concrete vocabulary a person \
might use when describing what they want to watch. Do not name other movies.
"""

USER_PROMPT_TEMPLATES = [
    "I love {movies}. What should I watch next?",
    "I really enjoyed {movies}. Recommend something.",
    "My favourite movies are {movies}.",
    "I just watched {movies} and loved them. What's next?",
    "Something like {movies}, please.",
]


def join_titles(titles: list[str]) -> str:
    if len(titles) <= 1:
        return "".join(titles)
    return ", ".join(titles[:-1]) + " and " + titles[-1]


def build_user_prompt(titles: list[str], rng: random.Random | None = None) -> str:
    template = (rng or random).choice(USER_PROMPT_TEMPLATES)
    return template.format(movies=join_titles(titles))
