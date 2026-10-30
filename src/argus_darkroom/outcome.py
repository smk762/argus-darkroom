"""The outcome table: from every scan cell, NudeNet and the age estimate to what happens to an image.

Pure functions over recorded values, so every combination can be tested without a server.

Order of precedence (deletion rules win over everything):

1. **deleted**: any cell ``illegal`` (any policy, either context), or ``minor-exposed``, or any
   *enforcing* cell ``unchecked`` (fails closed; the record regenerates the image once the guard is back).
2. **omitted**: SFW only, NudeNet exposed >= 0.2, or NudeNet did not answer. Dropped like a deleted image,
   but for space and cleanliness, not because it is criminal (``critical: false``).
3. **review**: any cell ``flagged``.
4. **passed**: every cell ``safe``.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus_cortex.moderation import policies as policy_table

from argus_darkroom.guard import illegal_rules_fired, rule_codes
from argus_darkroom.models import AgeRecord, Cell, Mode, Outcome, ScanCells

# NudeNet's reporting floor on the five default exposed classes. The SFW gate and the minor-exposed
# rule use it as is; the user's ``nsfw_threshold`` only sets blur under an NSFW policy.
EXPOSED_FLOOR = 0.2
ADULT_AGE = 18.0
DEFAULT_NSFW_THRESHOLD = 0.2


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: str | None = None
    critical: bool = False
    blur: bool = False

    @property
    def released(self) -> bool:
        """Whether the image is written and served."""
        return self.outcome in ("passed", "review")


def minor_exposed(age: AgeRecord | None, exposed_score: float | None) -> bool:
    """Any face estimated under 18 on an image NudeNet scores as exposed (>= 0.2).

    Curator's rule, and it ignores the user's NudeNet threshold. An age under 18 without exposure does not
    delete. With no age check (``age is None``) or no NudeNet answer the rule cannot fire: the minors rule
    then rests on SingGuard alone.
    """
    if age is None or exposed_score is None or exposed_score < EXPOSED_FLOOR:
        return False
    return any(face.age_estimate_years < ADULT_AGE for face in age.faces)


def _cells(scans: dict[str, ScanCells]) -> list[tuple[str, str, Cell]]:
    return [(policy, context, getattr(cells, context)) for policy, cells in scans.items() for context in _CONTEXTS]


_CONTEXTS = ("isolated", "with_prompt")


def _illegal_codes(scans: dict[str, ScanCells]) -> list[str]:
    codes: list[str] = []
    for policy, _, cell in _cells(scans):
        if cell.state == "illegal" and cell.result is not None:
            codes += [c for c in rule_codes(policy, illegal_rules_fired(policy, cell.result)) if c not in codes]
    return codes


def _flagged_codes(scans: dict[str, ScanCells]) -> list[str]:
    codes: list[str] = []
    for policy, _, cell in _cells(scans):
        if cell.state in ("flagged", "illegal") and cell.result is not None:
            fired = [r for r in cell.result.rules if r not in illegal_rules_fired(policy, cell.result)]
            codes += [c for c in rule_codes(policy, fired) if c not in codes]
    return codes


def decide(
    enforcing_policy: str,
    scans: dict[str, ScanCells],
    exposed_score: float | None,
    age: AgeRecord | None = None,
    nsfw_threshold: float = DEFAULT_NSFW_THRESHOLD,
) -> Decision:
    """The outcome for one scanned image.

    *exposed_score* is NudeNet's exposed score, or ``None`` when it did not answer (or is not configured).
    """
    mode: Mode = policy_table.mode(enforcing_policy)
    enforcing = scans.get(enforcing_policy)
    if enforcing is None:
        raise ValueError(f"no scan cells for the enforcing policy {enforcing_policy!r}")

    # 1. Deletion rules.
    any_illegal = any(cell.state == "illegal" for _, _, cell in _cells(scans))
    minor = minor_exposed(age, exposed_score)
    enforcing_unchecked = any(c.state == "unchecked" for c in (enforcing.isolated, enforcing.with_prompt))
    if any_illegal or minor or enforcing_unchecked:
        reasons = [
            *(_illegal_codes(scans) or (["illegal"] if any_illegal else [])),
            *(["minor-exposed"] if minor else []),
            *(["unchecked"] if enforcing_unchecked else []),
        ]
        return Decision("deleted", ", ".join(reasons), critical=any_illegal or minor)

    # 2. SFW must not return nudity, even behind a shield, nor anything NudeNet never scored.
    if mode == "sfw":
        if exposed_score is None:
            return Decision("omitted", "sfw-unchecked")
        if exposed_score >= EXPOSED_FLOOR:
            return Decision("omitted", "sfw-exposed")
        blur = False
    else:
        # NudeNet cannot see age or consent, so under NSFW it only blurs; no answer blurs too.
        blur = exposed_score is None or exposed_score >= nsfw_threshold

    # 3 and 4. What is left is decided by the cells. Unchecked advisory cells change nothing.
    if any(cell.state == "flagged" for _, _, cell in _cells(scans)):
        return Decision("review", ", ".join(_flagged_codes(scans)) or None, blur=blur)
    return Decision("passed", blur=blur)
