import json

import pytest

from movie_rag.slm import ProfileParseError, generate_target_profile, mock_slm_generate, parse_target_profile

VALID = {
    "rationale": "Both follow a charming rule-breaker who outwits authority.",
    "target_profile": "A brash outsider talks their way through a hostile small town.",
}


def test_parses_clean_json():
    assert parse_target_profile(json.dumps(VALID)).rationale == VALID["rationale"]


def test_parses_json_wrapped_in_prose_and_fences():
    raw = f"Sure! Here you go:\n```json\n{json.dumps(VALID)}\n```\nEnjoy."
    assert parse_target_profile(raw).target_profile == VALID["target_profile"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"rationale": "Both follow a charming rule-breaker", "target_prof',  # truncated mid-sentence
        json.dumps({"rationale": VALID["rationale"]}),  # missing key
        json.dumps({**VALID, "target_profile": "short"}),  # too short to be useful
        "no json here",
    ],
)
def test_rejects_invalid_output(raw):
    with pytest.raises(ProfileParseError):
        parse_target_profile(raw)


def test_retries_until_valid():
    outputs = iter(["garbage", '{"broken": ', json.dumps(VALID)])
    profile = generate_target_profile("I love X", generator=lambda _: next(outputs))
    assert profile.rationale == VALID["rationale"]


def test_gives_up_after_max_attempts():
    with pytest.raises(ProfileParseError):
        generate_target_profile("I love X", generator=lambda _: "garbage", max_attempts=2)


def test_mock_slm_output_is_valid():
    parse_target_profile(mock_slm_generate("anything"))
