"""The generate / pre-flight request. Policies are chosen by name only; a request never carries policy text."""

from __future__ import annotations

from argus_cortex.moderation import THINKING_MODES
from argus_cortex.moderation import policies as policy_table
from pydantic import BaseModel, ConfigDict, Field, field_validator

from argus_darkroom.models import SamplingParams


class RequestError(ValueError):
    """The request cannot be served as asked (HTTP 400)."""


class LoRASpec(BaseModel):
    """A LoRA as requested: by filename and weight, before the file is found and hashed."""

    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = 1.0
    trigger_words: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    # Unknown fields are refused, so a client sending ``policy_text`` gets an error, not silence.
    model_config = ConfigDict(extra="forbid")

    user_prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    expand: bool = False
    seed: int | None = None
    sampling: SamplingParams | None = None
    checkpoint: str | None = None
    vae: str | None = None
    loras: list[LoRASpec] = Field(default_factory=list)
    # None means the server default.
    enforcing_policy: str | None = None
    advisory_policies: list[str] = Field(default_factory=list)
    thinking: str | None = None
    nsfw_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("user_prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the prompt is empty")
        return value

    @field_validator("thinking")
    @classmethod
    def _known_thinking(cls, value: str | None) -> str | None:
        if value is not None and value not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {', '.join(THINKING_MODES)}")
        return value

    @property
    def trigger_words(self) -> list[str]:
        return [word for lora in self.loras for word in lora.trigger_words]


def check_policies(enforcing: str, advisory: list[str], allow_nsfw: bool) -> list[str]:
    """The advisory policies, deduplicated, once the selection is known to be valid.

    Raises :class:`RequestError` for an unknown name, or an NSFW policy enforcing without ``--allow-nsfw``.
    NSFW policies can still run as advisory for comparison.
    """
    for name in [enforcing, *advisory]:
        try:
            policy_table.policy_text(name)
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
    if policy_table.mode(enforcing) == "nsfw" and not allow_nsfw:
        raise RequestError(f"{enforcing!r} is an NSFW policy: this server was not started with --allow-nsfw")
    return [name for name in dict.fromkeys(advisory) if name != enforcing]
