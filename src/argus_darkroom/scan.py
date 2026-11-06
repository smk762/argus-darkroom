"""Scan one generated image: SingGuard twice per policy, NudeNet once, and ages when the check is available.

For each selected policy SingGuard judges the image **isolated** (no text) and **with the prompt** (the same
framed text pre-flight checked, as the user turn). The isolated scan is the defence against a prompt written to
talk the guard round: text can sway the with-prompt verdict, not the isolated one. The difference between the two
is the point of the scan, so both are always run and recorded, even once one has already come back illegal.

The age estimate is never sent to SingGuard as context: the isolated scan has to stay text-free, and the
with-prompt scan measures the prompt alone.

Every failure here is a value (an ``unchecked`` cell, a missing NudeNet answer, no age), never an exception, so a
record is always written and the outcome table decides.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import structlog
from argus_cortex.moderation import CheckFailed, NsfwResult, NudeNetClient

from argus_darkroom.age import AgeEstimator
from argus_darkroom.guard import Guard
from argus_darkroom.models import Cell, FaceAge, ScanCells

logger = structlog.get_logger()

OnCell = Callable[[str, str, Cell], None]  # (policy, "isolated" | "with_prompt", cell)


@dataclass(frozen=True)
class Scan:
    scans: dict[str, ScanCells]
    nsfw: NsfwResult | None  # None: NudeNet is not configured, or did not answer
    faces: list[FaceAge] | None  # None: the age check is unavailable, or failed for this image


class Scanner:
    def __init__(self, guard: Guard, nudenet: NudeNetClient | None = None, age: AgeEstimator | None = None) -> None:
        self.guard = guard
        self.nudenet = nudenet
        self.age = age

    def scan(
        self,
        data: bytes,
        framed_prompt: str,
        policies: list[str],
        thinking: str | None = None,
        on_cell: OnCell | None = None,
    ) -> Scan:
        scans: dict[str, ScanCells] = {}
        for policy in policies:
            isolated = self.guard.check_image(policy, data, "", thinking)
            if on_cell:
                on_cell(policy, "isolated", isolated)
            with_prompt = self.guard.check_image(policy, data, framed_prompt, thinking)
            if on_cell:
                on_cell(policy, "with_prompt", with_prompt)
            scans[policy] = ScanCells(isolated=isolated, with_prompt=with_prompt)
        return Scan(scans=scans, nsfw=self._nudenet(data), faces=self._ages(data))

    def _nudenet(self, data: bytes) -> NsfwResult | None:
        if self.nudenet is None:
            return None
        try:
            return self.nudenet.check(data)
        except CheckFailed as exc:
            logger.warning("nudenet.failed", error=str(exc))
        except Exception:
            logger.exception("nudenet.unexpected_error")
        return None

    def _ages(self, data: bytes) -> list[FaceAge] | None:
        if self.age is None:
            return None
        try:
            return self.age.estimate(data)
        except Exception as exc:
            # Includes AgeUnavailable. The minors rule then rests on SingGuard alone, as when the check is off.
            logger.warning("age.failed", error=str(exc))
            return None
