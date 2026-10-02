"""The "brain": turns a user's liked movies into a validated TargetProfile.

The finetuned SLM isn't wired in yet, so `mock_slm_generate` stands in for it. Any callable
that takes the user's text and returns the model's raw text output can replace it.
"""

import json
import logging
import re
from collections.abc import Callable

from pydantic import ValidationError

from .schemas import TargetProfile

logger = logging.getLogger(__name__)

SLMGenerator = Callable[[str], str]


class ProfileParseError(ValueError):
    pass


def mock_slm_generate(user_input: str) -> str:
    """Stand-in for the finetuned SLM. Returns the same sample JSON regardless of input."""
    return json.dumps(
        {
            "rationale": "Both films follow a fast-talking, rule-bending outsider who beats a "
            "stuffy authority figure through charm and nerve rather than credentials.",
            "target_profile": "A warm, quick-witted American comedy from the late 1980s or early "
            "1990s built around a mismatched pair. A brash, street-smart protagonist is dropped "
            "into an unfamiliar world and has to talk their way through escalating trouble. "
            "Sharp, bickering dialogue, an exasperated straight man, and an underdog who wins "
            "by outsmarting the system. Breezy pacing, big personalities, a big-hearted finish.",
        }
    )


def parse_target_profile(raw: str) -> TargetProfile:
    """Validate SLM output against the TargetProfile schema.

    Small models often wrap the JSON in prose or markdown fences, so if the raw text doesn't
    validate we retry on the outermost {...} block before giving up.
    """
    try:
        return TargetProfile.model_validate_json(raw)
    except ValidationError as first_error:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return TargetProfile.model_validate_json(match.group(0))
            except ValidationError:
                pass
        raise ProfileParseError(f"SLM output is not a valid TargetProfile: {raw[:200]!r}") from first_error


def generate_target_profile(
    user_input: str,
    generator: SLMGenerator = mock_slm_generate,
    max_attempts: int = 3,
) -> TargetProfile:
    """Call the SLM until it produces schema-valid output, up to `max_attempts` times."""
    for attempt in range(1, max_attempts + 1):
        raw = generator(user_input)
        try:
            return parse_target_profile(raw)
        except ProfileParseError:
            logger.warning("Attempt %d/%d: SLM returned invalid JSON", attempt, max_attempts)
    raise ProfileParseError(f"SLM failed to produce a valid TargetProfile after {max_attempts} attempts")
