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
from argus_cortex.moderation.nudenet import NudeNetInfo

from argus_darkroom.age import AgeEstimator, read_ages
from argus_darkroom.guard import Guard
from argus_darkroom.models import AgeRecord, Cell, ScanCells

logger = structlog.get_logger()

OnCell = Callable[[str, str, Cell], None]  # (policy, "isolated" | "with_prompt", cell)


@dataclass(frozen=True)
class Scan:
    scans: dict[str, ScanCells]
    nsfw: NsfwResult | None  # None: NudeNet is not configured, or did not answer
    age: AgeRecord | None  # None: the age check is unavailable, or failed for this image
    # Which model the NudeNet server says it runs, asked after it scored this image. None: it was not asked (no score),
    # or could not say. Recorded with the score, never used to decide: 640m and 320n score the same image differently.
    nudenet_info: NudeNetInfo | None = None


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
        nsfw = self._nudenet(data)
        info = self._nudenet_info() if nsfw is not None else None
        return Scan(scans=scans, nsfw=nsfw, age=self._ages(data), nudenet_info=info)

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

    def _nudenet_info(self) -> NudeNetInfo | None:
        """The model the NudeNet server runs. A failure here costs only the record's ``detector``, never the score."""
        if self.nudenet is None:
            return None
        try:
            return self.nudenet.info()
        except CheckFailed as exc:
            logger.warning("nudenet.info_failed", error=str(exc))
        except Exception:
            logger.exception("nudenet.info_unexpected_error")
        return None

    def _ages(self, data: bytes) -> AgeRecord | None:
        if self.age is None:
            return None
        try:
            return read_ages(self.age, data)
        except Exception as exc:
            # Includes AgeUnavailable. The minors rule then rests on SingGuard alone, as when the check is off.
            logger.warning("age.failed", error=str(exc))
            return None
