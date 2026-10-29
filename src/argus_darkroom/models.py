"""The darkroom wire contract: one :class:`GenerationRecord` per attempt.

A record is kept for every attempt, refused and deleted ones included. It is the
only thing left of a deleted image, so it holds every parameter needed to
regenerate it exactly, and never a pixel, thumbnail or preview of one.

``ModelRef``, ``LoRARef`` and ``SamplingParams`` copy argus-proof's field names
exactly (darkroom does not import proof), so lifting them into cortex later is a
rename.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from argus_cortex.moderation import GuardResult
from argus_cortex.wire import make_versioned_base, render_schema
from pydantic import BaseModel, Field

# 1.0: initial contract.
RECORD_VERSION = "1.0"
SUPPORTED_RECORD_MAJORS: tuple[str, ...] = ("1",)
WIRE_TITLE = "argus-darkroom wire contract"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="lowercase hex SHA256 digest")]

Outcome = Literal["refused", "passed", "review", "deleted", "omitted", "failed"]
Mode = Literal["sfw", "nsfw"]
CellState = Literal["safe", "flagged", "illegal", "unchecked"]

_Versioned = make_versioned_base(
    "record_version", RECORD_VERSION, SUPPORTED_RECORD_MAJORS, label="record_version", class_name="_Versioned"
)


class ModelRef(BaseModel):
    """A model file pinned by content: the name the engine loads and its sha256."""

    name: str
    sha256: Sha256


class LoRARef(ModelRef):
    weight: float = 1.0


class SamplingParams(BaseModel):
    sampler: str
    scheduler: str
    steps: int = Field(gt=0)
    cfg: float
    clip_skip: int = 1
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Expansion(BaseModel):
    """How Ollama expanded the prompt: enough to know which system prompt was used."""

    model: str
    system_prompt_sha256: Sha256


class RequestRecord(BaseModel):
    """What was asked for. ``final_prompt`` is the exact string ComfyUI receives."""

    # Derived from ``guard.enforcing_policy``; stored so records can be filtered by it.
    mode: Mode
    user_prompt: str
    expanded_prompt: str | None = None
    final_prompt: str
    negative_prompt: str = ""
    expansion: Expansion | None = None
    seed: int | None = None
    # The next five are null on a record refused in pre-flight: nothing was resolved or hashed.
    sampling: SamplingParams | None = None
    checkpoint: ModelRef | None = None
    vae: ModelRef | None = None
    loras: list[LoRARef] = Field(default_factory=list)
    workflow_sha256: Sha256 | None = None
    engine: Literal["comfyui"] = "comfyui"
    engine_version: str | None = None


class Cell(BaseModel):
    """One policy's verdict in one context: a state, and the SingGuard answer or the error behind it."""

    state: CellState
    result: GuardResult | None = None
    error: str | None = None


class ScanCells(BaseModel):
    """One policy's two image scans: alone, and alongside the prompt that made the image."""

    isolated: Cell
    with_prompt: Cell


class NudeNetRecord(BaseModel):
    exposed_score: float
    labels: dict[str, float] = Field(default_factory=dict)
    # The user's ``nsfw_threshold`` (blur). The SFW gate is a fixed 0.2 and ignores it.
    threshold: float
    blur: bool


class FaceAge(BaseModel):
    """A MiVOLO age estimate for one detected face. A tool's estimate, never a stated age."""

    age_estimate_years: float
    bbox: tuple[float, float, float, float]


class AgeRecord(BaseModel):
    source: Literal["mivolo"] = "mivolo"
    faces: list[FaceAge] = Field(default_factory=list)
    minor_exposed: bool = False


class GuardRecord(BaseModel):
    enforcing_policy: str
    advisory_policies: list[str] = Field(default_factory=list)
    thinking: str
    # Pre-flight is text-only: one cell per policy, framed prompt and negative prompt together.
    preflight: dict[str, Cell] = Field(default_factory=dict)
    scans: dict[str, ScanCells] = Field(default_factory=dict)
    nudenet: NudeNetRecord | None = None
    # null when the age check is unavailable
    age: AgeRecord | None = None


class ImageRecord(BaseModel):
    """The identity of the output. The hashes are kept even when the pixels are not."""

    sha256: Sha256
    phash: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    # Set only when the image is passed or review (the only outcomes that write it).
    path: str | None = None


class GenerationRecord(_Versioned):  # type: ignore[misc, valid-type]
    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    darkroom_version: str
    outcome: Outcome
    # Rule codes, "minor-exposed", "unchecked", "sfw-exposed", "sfw-unchecked", or the generation error.
    outcome_reason: str | None = None
    # True for illegal and minor-exposed deletions; false for everything else. Lets non-critical
    # records be pruned first when retention is decided.
    critical: bool = False
    request: RequestRecord
    guard: GuardRecord
    image: ImageRecord | None = None
    # Set when the user removed their own output (``DELETE /record/{id}``): the file is gone, the record stays.
    removed_at: datetime | None = None


def render_wire_schema() -> str:
    """The canonical committed-schema string (``schema`` writes it, ``schema --check`` compares)."""
    return render_schema([GenerationRecord], title=WIRE_TITLE)
