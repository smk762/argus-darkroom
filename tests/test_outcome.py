"""The outcome table, over every combination of cell states, and per policy for the illegal table."""

from __future__ import annotations

import itertools

import pytest
from argus_cortex.moderation import GuardResult

from argus_darkroom.guard import classify, unchecked
from argus_darkroom.models import AgeRecord, Cell, FaceAge, ScanCells
from argus_darkroom.outcome import decide, minor_exposed

SAFE = Cell(state="safe", result=GuardResult(safe=True))
FLAGGED = Cell(state="flagged", result=GuardResult(safe=False, rules=["B. Real-World Crimes"]))
ILLEGAL = Cell(state="illegal", result=GuardResult(safe=False, rules=["A. Minors"]))
UNCHECKED = unchecked("timeout")
STATES = {"safe": SAFE, "flagged": FLAGGED, "illegal": ILLEGAL, "unchecked": UNCHECKED}


def scans(enforcing: tuple[Cell, Cell], advisory: tuple[Cell, Cell] | None = None) -> dict[str, ScanCells]:
    out = {"adult-granular": ScanCells(isolated=enforcing[0], with_prompt=enforcing[1])}
    if advisory:
        out["adult"] = ScanCells(isolated=advisory[0], with_prompt=advisory[1])
    return out


def expected(cells: list[tuple[bool, str]]) -> str:
    """The spec's outcome rules, written independently of the implementation.

    *cells* is ``(is_enforcing, state)`` for every cell.
    """
    states = [s for _, s in cells]
    if "illegal" in states or any(e and s == "unchecked" for e, s in cells):
        return "deleted"
    if "flagged" in states:
        return "review"
    return "passed"


@pytest.mark.parametrize("enf", list(itertools.product(STATES, repeat=2)))
@pytest.mark.parametrize("adv", [None, *itertools.product(STATES, repeat=2)])
def test_every_combination_of_cell_states(enf: tuple[str, str], adv: tuple[str, str] | None) -> None:
    cells = scans((STATES[enf[0]], STATES[enf[1]]), (STATES[adv[0]], STATES[adv[1]]) if adv else None)
    layout = [(True, s) for s in enf] + ([(False, s) for s in adv] if adv else [])
    # NSFW enforcing, NudeNet clear: nothing but the cells decides.
    decision = decide("adult-granular", cells, exposed_score=0.0)
    assert decision.outcome == expected(layout)
    assert decision.critical is any(s == "illegal" for _, s in layout)
    assert decision.released is (decision.outcome in ("passed", "review"))


def test_unchecked_advisory_is_grey_and_changes_nothing() -> None:
    cells = scans((SAFE, SAFE), (UNCHECKED, UNCHECKED))
    assert decide("adult-granular", cells, 0.0).outcome == "passed"


def test_unchecked_enforcing_deletes_but_is_not_critical() -> None:
    decision = decide("adult-granular", scans((SAFE, UNCHECKED)), 0.0)
    assert (decision.outcome, decision.reason, decision.critical) == ("deleted", "unchecked", False)


def test_illegal_advisory_deletes_whatever_the_enforcing_policy_says() -> None:
    decision = decide("adult-granular", scans((SAFE, SAFE), (ILLEGAL, SAFE)), 0.0)
    assert decision.outcome == "deleted"
    assert decision.critical
    assert decision.reason == "adult:A"


def test_reason_names_policy_and_letter() -> None:
    cell = Cell(state="illegal", result=GuardResult(safe=False, rules=["H. Non-Consensual Sexual Content"]))
    assert decide("adult-granular", scans((cell, SAFE)), 0.0).reason == "adult-granular:H"


def test_review_reason_lists_flagged_rules() -> None:
    decision = decide("adult-granular", scans((FLAGGED, SAFE)), 0.0)
    assert (decision.outcome, decision.reason) == ("review", "adult-granular:B")


# --- per policy: which rules are illegal ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "rule", "state"),
    [
        ("adult-granular", "A. Minors", "illegal"),
        ("adult-granular", "H. Non-Consensual Sexual Content", "illegal"),
        ("adult-granular", "I. Sexual Violence & Extreme Content", "illegal"),
        ("adult-granular", "B. Real-World Crimes & Public Safety", "flagged"),
        ("adult-granular", "G. Animal Abuse", "flagged"),
        ("adult", "A. Illegal or Extreme Sexual Content", "illegal"),
        ("adult", "H", "flagged"),  # no H under `adult`
        ("adult-cautious", "A. Illegal or Extreme Sexual Content", "illegal"),
        ("adult-cautious", "B. Real-World Crimes & Public Safety", "flagged"),
        # `default`'s A is all sexual content, adult nudity included, so it can only send to review.
        ("default", "A. Sexual Content Risk", "flagged"),
        ("default", "B. Real-World Crimes & Public Safety", "flagged"),
    ],
)
def test_the_illegal_table_is_per_policy(policy: str, rule: str, state: str) -> None:
    assert classify(policy, GuardResult(safe=False, rules=[rule])).state == state


def test_unsafe_with_no_rule_named_is_flagged_not_illegal() -> None:
    assert classify("adult-granular", GuardResult(safe=False)).state == "flagged"


def test_a_safe_verdict_is_safe_under_every_policy() -> None:
    for policy in ("default", "adult", "adult-granular", "adult-cautious"):
        assert classify(policy, GuardResult(safe=True)).state == "safe"


# --- SFW / NudeNet --------------------------------------------------------------------------------------

CLEAN = scans((SAFE, SAFE))


@pytest.mark.parametrize(
    ("score", "outcome", "reason"),
    [(0.0, "passed", None), (0.19, "passed", None), (0.2, "omitted", "sfw-exposed"), (0.97, "omitted", "sfw-exposed")],
)
def test_sfw_exposed_gate(score: float, outcome: str, reason: str | None) -> None:
    sfw = {"default": CLEAN["adult-granular"]}
    decision = decide("default", sfw, score)
    assert (decision.outcome, decision.reason) == (outcome, reason)
    assert decision.critical is False
    assert decision.blur is False


def test_sfw_without_a_nudenet_answer_is_omitted() -> None:
    decision = decide("default", {"default": CLEAN["adult-granular"]}, None)
    assert (decision.outcome, decision.reason, decision.critical) == ("omitted", "sfw-unchecked", False)


def test_sfw_ignores_the_users_nsfw_threshold() -> None:
    decision = decide("default", {"default": CLEAN["adult-granular"]}, 0.5, nsfw_threshold=0.95)
    assert decision.outcome == "omitted"


def test_sfw_flagged_and_unexposed_is_review() -> None:
    assert decide("default", {"default": ScanCells(isolated=FLAGGED, with_prompt=SAFE)}, 0.05).outcome == "review"


def test_sfw_exposed_and_flagged_is_omitted_not_review() -> None:
    assert decide("default", {"default": ScanCells(isolated=FLAGGED, with_prompt=SAFE)}, 0.5).outcome == "omitted"


def test_exposed_and_illegal_is_deleted_not_omitted() -> None:
    decision = decide("default", {"default": ScanCells(isolated=ILLEGAL, with_prompt=SAFE)}, 0.9)
    assert (decision.outcome, decision.critical) == ("deleted", True)


def test_sfw_deleted_beats_unchecked_nudenet() -> None:
    assert decide("default", {"default": ScanCells(isolated=ILLEGAL, with_prompt=SAFE)}, None).outcome == "deleted"


@pytest.mark.parametrize(
    ("score", "threshold", "blur"),
    [(0.1, 0.2, False), (0.2, 0.2, True), (0.5, 0.6, False), (0.7, 0.6, True), (None, 0.2, True)],
)
def test_nsfw_nudenet_only_blurs(score: float | None, threshold: float, blur: bool) -> None:
    decision = decide("adult-granular", CLEAN, score, nsfw_threshold=threshold)
    assert (decision.outcome, decision.blur) == ("passed", blur)


def test_nsfw_never_omits_on_nudenet_alone() -> None:
    assert decide("adult-granular", CLEAN, 0.99).outcome == "passed"
    assert decide("adult-granular", CLEAN, None).outcome == "passed"


# --- age -----------------------------------------------------------------------------------------------


def faces(*ages: float) -> AgeRecord:
    return AgeRecord(faces=[FaceAge(age_estimate_years=a, bbox=(0, 0, 1, 1)) for a in ages])


def test_minor_with_exposure_deletes_in_either_mode() -> None:
    for policy in ("adult-granular", "default"):
        decision = decide(policy, {policy: CLEAN["adult-granular"]}, 0.2, age=faces(35, 17.9))
        assert (decision.outcome, decision.reason, decision.critical) == ("deleted", "minor-exposed", True)


def test_minor_exposed_ignores_the_users_threshold() -> None:
    decision = decide("adult-granular", CLEAN, 0.2, age=faces(12), nsfw_threshold=1.0)
    assert decision.outcome == "deleted"


def test_minor_without_exposure_does_not_delete() -> None:
    decision = decide("adult-granular", CLEAN, 0.19, age=faces(12))
    assert decision.outcome == "passed"


def test_adult_faces_with_exposure_do_not_delete() -> None:
    assert decide("adult-granular", CLEAN, 0.9, age=faces(18, 40)).outcome == "passed"


def test_age_unavailable_leaves_the_minors_rule_to_singguard() -> None:
    assert not minor_exposed(None, 0.9)
    assert decide("adult-granular", CLEAN, 0.9, age=None).outcome == "passed"


def test_no_nudenet_answer_cannot_make_minor_exposed() -> None:
    assert not minor_exposed(faces(10), None)


def test_illegal_and_minor_exposed_report_both() -> None:
    cells = scans((ILLEGAL, SAFE))
    assert decide("adult-granular", cells, 0.5, age=faces(10)).reason == "adult-granular:A, minor-exposed"


def test_enforcing_policy_must_have_cells() -> None:
    with pytest.raises(ValueError, match="enforcing"):
        decide("adult", CLEAN, 0.0)
