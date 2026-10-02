from pydantic import BaseModel, Field


class TargetProfile(BaseModel):
    """What the SLM (and the teacher LLM it imitates) must output."""

    rationale: str = Field(min_length=10, description="The thematic bridge between the input movies.")
    target_profile: str = Field(
        min_length=20,
        description="Abstract description of the ideal next watch. Mechanics, tropes, era vibe. No real titles.",
    )


class VibeDescription(BaseModel):
    """Teacher output when enriching a catalog entry."""

    vibe_description: str = Field(min_length=40)
