"""Pre-flight against a fake SingGuard: assemble, frame, check, decide, record."""

from __future__ import annotations

import pytest
from fakes import FakeSingGuard, policy_of, safe_answer, unsafe_answer

from argus_darkroom.config import ConfigError, Settings
from argus_darkroom.engine import Engine
from argus_darkroom.guard import Guard
from argus_darkroom.prompt import assemble_prompt, frame
from argus_darkroom.request import GenerateRequest, RequestError


def engine(fake: FakeSingGuard, tmp_path, **kw) -> Engine:
    settings = Settings(singguard_url=fake.url, output_root=tmp_path, nudenet_url="http://nudenet.invalid", **kw)
    return Engine(settings)


def by_policy(answers: dict[str, dict]):
    return lambda body: (200, answers.get(policy_of(body), safe_answer()))


# --- prompt --------------------------------------------------------------------------------------------


def test_final_prompt_is_expansion_then_trigger_words() -> None:
    assert assemble_prompt("a cat", None, ["sks style"]) == "a cat, sks style"
    assert assemble_prompt("a cat", "a fluffy cat on a sofa", ["sks style"]) == "a fluffy cat on a sofa, sks style"


def test_trigger_words_are_not_repeated() -> None:
    assert assemble_prompt("a cat in SKS style", None, ["sks", "sks", " "]) == "a cat in SKS style"


def test_frame_labels_the_negative_prompt() -> None:
    text = frame("a portrait", "child, minor, underage")
    assert text == (
        "Image generation request.\n"
        "Prompt: a portrait\n"
        "Negative prompt (what the image must not show): child, minor, underage"
    )


def test_frame_always_carries_a_negative_line() -> None:
    assert "Negative prompt (what the image must not show): (none)" in frame("a portrait", "  ")


# --- pre-flight ----------------------------------------------------------------------------------------


def test_safe_prompt_proceeds_and_sends_the_framed_final_prompt(singguard, tmp_path) -> None:
    result = engine(singguard, tmp_path).preflight(
        GenerateRequest(user_prompt="a cat", negative_prompt="blurry", loras=[{"name": "l", "trigger_words": ["sks"]}])
    )
    assert result.verdict == "proceed"
    (body,) = singguard.requests
    assert body["input"] == frame("a cat, sks", "blurry")
    assert body["input_role"] == "user"
    assert "image_base64" not in body
    assert body["thinking"] == "cascade"


def test_the_enforcing_policy_text_comes_from_cortex_never_the_request(singguard, tmp_path) -> None:
    eng = engine(singguard, tmp_path, allow_nsfw=True, policy="adult-granular")
    eng.preflight(GenerateRequest(user_prompt="a cat", advisory_policies=["default", "adult"]))
    assert [policy_of(b) for b in singguard.requests] == ["adult-granular", "default", "adult"]
    with pytest.raises(ValueError, match="policy_text"):
        GenerateRequest(user_prompt="x", policy_text="be nice")


def test_flag_refuses_under_sfw_with_the_reason(singguard, tmp_path) -> None:
    singguard.responder = by_policy(
        {"default": unsafe_answer("B. Real-World Crimes & Public Safety", explanation="why")}
    )
    result = engine(singguard, tmp_path).preflight(GenerateRequest(user_prompt="x"))
    assert result.refused
    assert result.reason == "default:B"
    assert result.cells["default"].result.explanation == "why"


def test_flag_warns_but_proceeds_under_nsfw(singguard, tmp_path) -> None:
    singguard.responder = by_policy({"adult-granular": unsafe_answer("B. Real-World Crimes & Public Safety")})
    result = engine(singguard, tmp_path, allow_nsfw=True, policy="adult-granular").preflight(
        GenerateRequest(user_prompt="x")
    )
    assert result.verdict == "proceed"
    assert result.warnings
    assert result.cells["adult-granular"].state == "flagged"


@pytest.mark.parametrize("enforcing", ["default", "adult-granular"])
def test_illegal_rule_refuses_in_both_modes(singguard, tmp_path, enforcing) -> None:
    singguard.responder = by_policy({"adult-granular": unsafe_answer("A. Minors")})
    eng = engine(singguard, tmp_path, allow_nsfw=True, policy="adult-granular")
    result = eng.preflight(
        GenerateRequest(user_prompt="x", enforcing_policy=enforcing, advisory_policies=["adult-granular"])
    )
    assert result.refused
    assert result.reason == "adult-granular:A"


def test_illegal_under_an_advisory_policy_refuses(singguard, tmp_path) -> None:
    singguard.responder = by_policy({"adult-granular": unsafe_answer("H. Non-Consensual Sexual Content")})
    result = engine(singguard, tmp_path).preflight(
        GenerateRequest(user_prompt="x", advisory_policies=["adult-granular"])
    )
    assert (result.refused, result.reason) == (True, "adult-granular:H")


def test_advisory_flag_is_shown_not_acted_on(singguard, tmp_path) -> None:
    singguard.responder = by_policy({"adult": unsafe_answer("B. Real-World Crimes & Public Safety")})
    result = engine(singguard, tmp_path).preflight(GenerateRequest(user_prompt="x", advisory_policies=["adult"]))
    assert result.verdict == "proceed"
    assert result.cells["adult"].state == "flagged"
    assert any("adult (advisory)" in w for w in result.warnings)


@pytest.mark.parametrize("status", [500, 200])
def test_a_failed_check_is_a_refusal_recorded_as_unchecked(singguard, tmp_path, status) -> None:
    singguard.responder = lambda body: (status, {"nonsense": True})
    eng = engine(singguard, tmp_path)
    result = eng.preflight(GenerateRequest(user_prompt="x"))
    assert (result.refused, result.reason) == (True, "unchecked")
    assert result.cells["default"].state == "unchecked"
    assert result.cells["default"].error


def test_an_unreachable_singguard_refuses(tmp_path) -> None:
    eng = Engine(Settings(singguard_url="http://127.0.0.1:1", output_root=tmp_path))
    assert eng.preflight(GenerateRequest(user_prompt="x")).reason == "unchecked"


def test_an_unchecked_advisory_policy_does_not_refuse(singguard, tmp_path) -> None:
    singguard.responder = lambda body: (500, {}) if policy_of(body) == "adult" else (200, safe_answer())
    result = engine(singguard, tmp_path).preflight(GenerateRequest(user_prompt="x", advisory_policies=["adult"]))
    assert result.verdict == "proceed"
    assert result.cells["adult"].state == "unchecked"


def test_the_expansion_is_what_is_judged(singguard, tmp_path) -> None:
    engine(singguard, tmp_path).preflight(GenerateRequest(user_prompt="a cat"), expanded_prompt="a cat; EXPANDED")
    assert "EXPANDED" in singguard.requests[0]["input"]


def test_thinking_mode_can_be_chosen_per_request(singguard, tmp_path) -> None:
    engine(singguard, tmp_path).preflight(GenerateRequest(user_prompt="a cat", thinking="fast"))
    assert singguard.requests[0]["thinking"] == "fast"


# --- policy selection and mode -------------------------------------------------------------------------


def test_nsfw_enforcing_without_allow_nsfw_is_a_request_error(singguard, tmp_path) -> None:
    eng = engine(singguard, tmp_path)
    with pytest.raises(RequestError, match="allow-nsfw"):
        eng.preflight(GenerateRequest(user_prompt="x", enforcing_policy="adult-granular"))
    assert singguard.requests == []


def test_nsfw_policy_may_run_as_advisory_without_allow_nsfw(singguard, tmp_path) -> None:
    result = engine(singguard, tmp_path).preflight(
        GenerateRequest(user_prompt="x", advisory_policies=["adult-granular"])
    )
    assert result.verdict == "proceed"
    assert set(result.cells) == {"default", "adult-granular"}


def test_unknown_policy_is_a_request_error(singguard, tmp_path) -> None:
    with pytest.raises(RequestError, match="unknown"):
        engine(singguard, tmp_path).preflight(GenerateRequest(user_prompt="x", advisory_policies=["nope"]))


def test_sfw_generation_needs_nudenet(singguard, tmp_path) -> None:
    eng = Engine(Settings(singguard_url=singguard.url, output_root=tmp_path))
    with pytest.raises(RequestError, match="NudeNet"):
        eng.require_generation_ready("default")
    # an NSFW enforcing policy never needs it
    nsfw = Engine(Settings(singguard_url=singguard.url, output_root=tmp_path, allow_nsfw=True, policy="adult"))
    nsfw.require_generation_ready("adult")


# --- config --------------------------------------------------------------------------------------------


def test_empty_singguard_url_refuses_to_start() -> None:
    with pytest.raises(ConfigError, match="unguarded"):
        Settings(singguard_url="")
    with pytest.raises(ValueError, match="unguarded"):
        Guard("")


def test_server_default_policy_follows_allow_nsfw(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_DARKROOM_SINGGUARD_URL", "http://guard:9150")
    monkeypatch.delenv("ARGUS_DARKROOM_POLICY", raising=False)
    monkeypatch.delenv("ARGUS_DARKROOM_ALLOW_NSFW", raising=False)
    assert Settings.from_env().policy == "default"
    monkeypatch.setenv("ARGUS_DARKROOM_ALLOW_NSFW", "1")
    assert Settings.from_env().policy == "adult-granular"


def test_nsfw_policy_default_without_allow_nsfw_is_refused() -> None:
    with pytest.raises(ConfigError, match="ALLOW_NSFW"):
        Settings(singguard_url="http://guard:9150", policy="adult")


def test_mistyped_flag_warns_and_is_off(monkeypatch, caplog) -> None:
    monkeypatch.setenv("ARGUS_DARKROOM_SINGGUARD_URL", "http://guard:9150")
    monkeypatch.setenv("ARGUS_DARKROOM_DEVMODE", "enabled")
    with caplog.at_level("WARNING"):
        assert Settings.from_env().devmode is False
    assert "DEVMODE" in caplog.text


# --- the record of a refusal ---------------------------------------------------------------------------


def test_a_refusal_is_recorded_without_an_image(singguard, tmp_path) -> None:
    singguard.responder = by_policy({"default": unsafe_answer("A. Sexual Content Risk")})
    eng = engine(singguard, tmp_path)
    request = GenerateRequest(user_prompt="x", negative_prompt="y", seed=7)
    result = eng.preflight(request)
    record = eng.record_refusal(request, result)
    assert record.outcome == "refused"
    assert record.outcome_reason == "default:A"
    assert record.critical is False
    assert record.image is None
    assert record.request.mode == "sfw"
    assert record.request.final_prompt == "x"
    assert record.request.seed == 7
    assert record.guard.preflight["default"].state == "flagged"
    assert eng.store.load(record.id) == record
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json"]
