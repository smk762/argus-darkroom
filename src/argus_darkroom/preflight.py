"""Pre-flight: a text-only check of the final prompt and negative prompt, before anything is generated.

=============================  =====================================  ===================================
Enforcing / advisory result    SFW                                    NSFW
=============================  =====================================  ===================================
illegal rule, any policy       refused                                refused
flagged, enforcing policy      refused, with the reason               warning; generation proceeds
flagged, advisory policy       shown, not acted on                    shown, not acted on
unchecked, enforcing policy    refused (``unchecked``)                refused (``unchecked``)
unchecked, advisory policy     shown as unchecked, not acted on       shown as unchecked, not acted on
=============================  =====================================  ===================================

An unchecked advisory cell cannot refuse: advisory policies never block or release anything except through an
illegal rule, which needs a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from argus_cortex.moderation import policies as policy_table

from argus_darkroom.guard import Guard, illegal_rules_fired, rule_codes
from argus_darkroom.models import Cell
from argus_darkroom.prompt import frame


@dataclass(frozen=True)
class PreflightResult:
    framed: str
    cells: dict[str, Cell]
    verdict: Literal["proceed", "refused"]
    # ``policy:letter`` codes, or ``unchecked``; null when nothing refused.
    reason: str | None = None
    # The cell that refused (or, on a warning, flagged), so the user can be told which rule fired and why.
    refusing_policy: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.verdict == "refused"

    def payload(self) -> dict:
        """The result as sent to a client (``/preflight`` and the ``preflight`` event)."""
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "refusing_policy": self.refusing_policy,
            "warnings": self.warnings,
            "cells": {name: cell.model_dump(mode="json") for name, cell in self.cells.items()},
        }


def run_preflight(
    guard: Guard,
    final_prompt: str,
    negative_prompt: str,
    enforcing_policy: str,
    advisory_policies: list[str] | tuple[str, ...] = (),
    thinking: str | None = None,
) -> PreflightResult:
    """Check the framed prompt under every selected policy and decide whether to proceed."""
    framed = frame(final_prompt, negative_prompt)
    names = [enforcing_policy, *(p for p in advisory_policies if p != enforcing_policy)]
    cells = {name: guard.check_text(name, framed, thinking) for name in names}

    # Any illegal rule under any policy refuses, in both modes.
    illegal = [
        (name, rule_codes(name, illegal_rules_fired(name, cell.result)))
        for name, cell in cells.items()
        if cell.state == "illegal" and cell.result is not None
    ]
    if illegal:
        return PreflightResult(
            framed, cells, "refused", ", ".join(code for _, codes in illegal for code in codes), illegal[0][0]
        )

    enforcing = cells[enforcing_policy]
    if enforcing.state == "unchecked":
        return PreflightResult(framed, cells, "refused", "unchecked", enforcing_policy)

    warnings: list[str] = []
    if enforcing.state == "flagged" and enforcing.result is not None:
        codes = ", ".join(rule_codes(enforcing_policy, enforcing.result.rules)) or "unspecified"
        if policy_table.mode(enforcing_policy) == "sfw":
            return PreflightResult(framed, cells, "refused", codes, enforcing_policy)
        warnings.append(f"{enforcing_policy}: flagged {codes}; the output scan decides")
    for name, cell in cells.items():
        if name != enforcing_policy and cell.state in ("flagged", "unchecked"):
            warnings.append(f"{name} (advisory): {cell.state}")
    return PreflightResult(framed, cells, "proceed", None, None, warnings)
