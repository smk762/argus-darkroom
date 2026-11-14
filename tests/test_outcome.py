"""The outcome table, over every combination of cell states, and per policy for the illegal table."""

from __future__ import annotations

import itertools

import pytest
from argus_cortex.moderation import GuardResult, NsfwDetection

from argus_darkroom.guard import classify, unchecked
from argus_darkroom.models import AgeRecord, Cell, FaceAge, NudeNetBox, PersonAge, ScanCells
from argus_darkroom.outcome import decide, minor_exposed, overlaps

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
        ("adult-granular", "I. Sexual Violence", "illegal"),
        ("adult-granular", "B. Real-World Crimes & Public Safety", "flagged"),
        ("adult-granular", "G. Animal Abuse", "flagged"),
        ("adult", "A. Illegal Sexual Content", "illegal"),
        ("adult", "H", "flagged"),  # no H under `adult`
        ("adult-cautious", "A. Illegal Sexual Content", "illegal"),
        ("adult-cautious", "B. Real-World Crimes & Public Safety", "flagged"),
        ("adult-cinematic", "A. Illegal Sexual Content", "illegal"),
        ("adult-cinematic", "G. Criminal Instructions", "flagged"),
        ("adult-cinematic", "F. Animal Abuse", "flagged"),
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
    for policy in ("default", "adult", "adult-granular", "adult-cautious", "adult-cinematic"):
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


# --- the overlap rule ------------------------------------------------------------------------------------
#
# Two people side by side: LEFT is x 0-100, RIGHT is x 200-300. A face sits at the top of each body.

LEFT, RIGHT = (0.0, 0.0, 100.0, 300.0), (200.0, 0.0, 100.0, 300.0)
LEFT_FACE, RIGHT_FACE = (30.0, 10.0, 40.0, 40.0), (230.0, 10.0, 40.0, 40.0)
ON_LEFT, ON_RIGHT, ON_NOBODY = (20.0, 150.0, 40.0, 40.0), (220.0, 150.0, 40.0, 40.0), (500.0, 150.0, 40.0, 40.0)


def exposed(*boxes: tuple[float, float, float, float], label: str = "FEMALE_BREAST_EXPOSED") -> list[NsfwDetection]:
    return [NsfwDetection(label=label, score=0.6, bbox=list(b)) for b in boxes]


def two_people(left_years: float, right_years: float) -> AgeRecord:
    """Two faces, each matched to its body; curator puts a matched person's age on the face only."""
    return AgeRecord(
        faces=[
            FaceAge(age_estimate_years=left_years, bbox=LEFT_FACE),
            FaceAge(age_estimate_years=right_years, bbox=RIGHT_FACE),
        ],
        people=[PersonAge(bbox=LEFT, face_index=0), PersonAge(bbox=RIGHT, face_index=1)],
    )


def overlap(age: AgeRecord | None, detections: list[NsfwDetection], score: float = 0.6) -> bool:
    return minor_exposed(age, score, detections, "overlap")


def test_overlaps_is_shared_area_only() -> None:
    assert overlaps((0, 0, 10, 10), (5, 5, 10, 10))
    assert not overlaps((0, 0, 10, 10), (10, 0, 10, 10))  # touching edges share no area
    assert not overlaps((0, 0, 10, 10), (20, 20, 5, 5))


def test_frame_is_the_default_and_ignores_boxes() -> None:
    # The young person is clothed and the exposure is on the adult: frame still fires, as it always has.
    age = two_people(12, 35)
    assert minor_exposed(age, 0.6, exposed(ON_RIGHT))
    assert minor_exposed(age, 0.6, exposed(ON_RIGHT), "frame")
    assert decide("adult-granular", CLEAN, 0.6, age=age, detections=exposed(ON_RIGHT)).outcome == "deleted"


def test_overlap_exposure_on_a_young_person_flags() -> None:
    assert overlap(two_people(12, 35), exposed(ON_LEFT))
    decision = decide(
        "adult-granular", CLEAN, 0.6, age=two_people(12, 35), detections=exposed(ON_LEFT), minor_rule="overlap"
    )
    assert (decision.outcome, decision.reason, decision.critical) == ("deleted", "minor-exposed", True)


def test_overlap_exposure_on_an_adult_only_does_not_flag() -> None:
    assert not overlap(two_people(12, 35), exposed(ON_RIGHT))
    decision = decide(
        "adult-granular", CLEAN, 0.6, age=two_people(12, 35), detections=exposed(ON_RIGHT), minor_rule="overlap"
    )
    assert decision.outcome == "passed"


def test_overlap_exposure_on_no_box_flags() -> None:
    assert overlap(two_people(12, 35), exposed(ON_NOBODY))


def test_overlap_a_young_face_without_a_person_box_flags_any_exposure() -> None:
    age = two_people(12, 35)
    age.people = [age.people[1]]  # the young face's body was not found
    assert overlap(age, exposed(ON_RIGHT))


def test_overlap_a_faceless_young_person_is_placed_by_its_box() -> None:
    # An unmatched person's estimate is in faces too (appended last) and is not a face missing its body.
    age = AgeRecord(
        faces=[FaceAge(age_estimate_years=35, bbox=RIGHT_FACE), FaceAge(age_estimate_years=12, bbox=LEFT)],
        people=[PersonAge(bbox=RIGHT, face_index=0), PersonAge(bbox=LEFT, age_estimate_years=12)],
    )
    assert overlap(age, exposed(ON_LEFT))
    assert not overlap(age, exposed(ON_RIGHT))


def test_overlap_knows_faceless_people_from_people_not_from_the_order_of_faces() -> None:
    # The faceless young person's estimate comes first in faces. Read by position, it would look like a young face
    # whose body was not found, and the adult's exposure would delete; read from people, it is placed by its box.
    age = AgeRecord(
        faces=[FaceAge(age_estimate_years=12, bbox=LEFT), FaceAge(age_estimate_years=35, bbox=RIGHT_FACE)],
        people=[PersonAge(bbox=LEFT, age_estimate_years=12), PersonAge(bbox=RIGHT, face_index=1)],
    )
    assert not overlap(age, exposed(ON_RIGHT))
    assert overlap(age, exposed(ON_LEFT))


def test_overlap_recomputes_from_the_records_boxes() -> None:
    boxes = [NudeNetBox(label="FEMALE_BREAST_EXPOSED", score=0.6, bbox=ON_RIGHT)]
    assert not overlap(two_people(12, 35), boxes)
    assert overlap(two_people(35, 12), boxes)


def test_overlap_without_people_uses_face_boxes() -> None:
    age = AgeRecord(
        faces=[
            FaceAge(age_estimate_years=12, bbox=(0, 0, 100, 100)),
            FaceAge(age_estimate_years=35, bbox=(200, 0, 100, 100)),
        ]
    )
    assert overlap(age, exposed((10, 10, 20, 20)))
    assert not overlap(age, exposed((210, 10, 20, 20)))
    assert overlap(age, exposed((500, 500, 20, 20)))


def test_overlap_exposure_with_no_detection_box_flags() -> None:
    # NudeNet scored exposure but returned no exposed detection to place: fail closed.
    assert overlap(two_people(12, 35), [])
    assert overlap(two_people(12, 35), exposed(ON_RIGHT, label="FACE_FEMALE"))


def test_overlap_needs_a_young_age_and_an_exposed_score() -> None:
    assert not overlap(two_people(30, 35), exposed(ON_LEFT))
    assert not overlap(two_people(12, 35), exposed(ON_LEFT), score=0.19)
    assert not overlap(None, exposed(ON_LEFT))
    assert not minor_exposed(two_people(12, 35), None, exposed(ON_LEFT), "overlap")


def test_overlap_ignores_detections_under_the_floor() -> None:
    weak = [NsfwDetection(label="FEMALE_BREAST_EXPOSED", score=0.1, bbox=list(ON_LEFT))]
    # The only exposed detection on the young person is under 0.2; another on the adult places the rest.
    assert not overlap(two_people(12, 35), [*weak, *exposed(ON_RIGHT)])


def test_enforcing_policy_must_have_cells() -> None:
    with pytest.raises(ValueError, match="enforcing"):
        decide("adult", CLEAN, 0.0)
