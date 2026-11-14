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

from collections.abc import Sequence
from dataclasses import dataclass

from argus_cortex.moderation import DEFAULT_EXPOSED_LABELS, NsfwDetection
from argus_cortex.moderation import policies as policy_table
from argus_cortex.moderation.nudenet import MIN_REPORTED_SCORE

from argus_darkroom.guard import illegal_rules_fired, rule_codes
from argus_darkroom.models import AgeRecord, Cell, FaceAge, MinorExposedRule, Mode, Outcome, PersonAge, ScanCells

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


def overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two ``[x, y, w, h]`` boxes share any area. Touching edges do not (curator's ``bodies.overlaps``)."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return w > 0 and h > 0


def _young(years: float | None) -> bool:
    return years is not None and years < ADULT_AGE


def _young_person(age: AgeRecord, person: PersonAge) -> bool:
    """A person is young on its own estimate or its matched face's (curator puts a matched age on the face)."""
    if _young(person.age_estimate_years):
        return True
    i = person.face_index
    return i is not None and 0 <= i < len(age.faces) and _young(age.faces[i].age_estimate_years)


def _detected_faces(age: AgeRecord) -> list[FaceAge]:
    """The faces proper: ``faces`` less the faceless people's estimates, which the estimators append last."""
    faceless = sum(1 for p in age.people if p.face_index is None)
    return age.faces[: max(0, len(age.faces) - faceless)]


def _overlap_rule(age: AgeRecord, detections: Sequence[NsfwDetection]) -> bool:
    """Exposure placed by box: on a young person, or on nobody, flags; on adults only, does not.

    With person boxes, else face boxes. Fails closed: a young face with no person box, a young age with no box,
    or exposure with no detection box to place it, flags.
    """
    if not any(_young(face.age_estimate_years) for face in age.faces):
        return False
    young: list[tuple[float, float, float, float]]
    if age.people:
        matched = {p.face_index for p in age.people if p.face_index is not None}
        if any(_young(f.age_estimate_years) and i not in matched for i, f in enumerate(_detected_faces(age))):
            return True  # a young face whose body was not found: any exposure may be theirs
        boxes = [p.bbox for p in age.people]
        young = [p.bbox for p in age.people if _young_person(age, p)]
    else:
        boxes = [f.bbox for f in age.faces]
        young = [f.bbox for f in age.faces if _young(f.age_estimate_years)]
    exposed = [d for d in detections if d.label in DEFAULT_EXPOSED_LABELS and d.score >= MIN_REPORTED_SCORE]
    if not young or not exposed:
        return True  # the young age (or the exposure) has no box to compare
    return any(any(overlaps(d.bbox, b) for b in young) or not any(overlaps(d.bbox, b) for b in boxes) for d in exposed)


def minor_exposed(
    age: AgeRecord | None,
    exposed_score: float | None,
    detections: Sequence[NsfwDetection] = (),
    rule: MinorExposedRule = "frame",
) -> bool:
    """Whether an age estimate under 18 shares the image with exposure NudeNet scores >= 0.2.

    It ignores the user's NudeNet threshold. An age under 18 without exposure does not delete. With no age
    check (``age is None``) or no NudeNet answer the rule cannot fire: the minors rule then rests on SingGuard
    alone.

    ``frame`` (the default): any estimate under 18 anywhere in the frame. ``overlap``: curator's original rule,
    which places each exposed *detection* by box (see :func:`_overlap_rule`), so an adult's exposure beside a
    clothed minor does not fire it.
    """
    if age is None or exposed_score is None or exposed_score < EXPOSED_FLOOR:
        return False
    if rule == "overlap":
        return _overlap_rule(age, detections)
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
    detections: Sequence[NsfwDetection] = (),
    minor_rule: MinorExposedRule = "frame",
) -> Decision:
    """The outcome for one scanned image.

    *exposed_score* is NudeNet's exposed score, or ``None`` when it did not answer (or is not configured).
    *detections* are NudeNet's boxes, which only the ``overlap`` minor rule reads.
    """
    mode: Mode = policy_table.mode(enforcing_policy)
    enforcing = scans.get(enforcing_policy)
    if enforcing is None:
        raise ValueError(f"no scan cells for the enforcing policy {enforcing_policy!r}")

    # 1. Deletion rules.
    any_illegal = any(cell.state == "illegal" for _, _, cell in _cells(scans))
    minor = minor_exposed(age, exposed_score, detections, minor_rule)
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
