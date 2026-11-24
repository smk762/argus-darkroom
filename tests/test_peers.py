"""Where darkroom finds its backends: its own variable, then the suite-wide one, and never a guard lost to a typo.

The URLs here are made up and reach nothing: only ``Settings.from_env`` and ``default_estimator`` are under test.
"""

from __future__ import annotations

import pytest
from fakes import FakeFaces

from argus_darkroom import age
from argus_darkroom.age import AgeStatus, CuratorAgeEstimator, HttpAgeEstimator, default_estimator
from argus_darkroom.config import ConfigError, Settings
from argus_darkroom.server import create_app

OWN = "ARGUS_DARKROOM_"
GUARD = "http://guard:9150"
PEERS = [  # (Settings field, darkroom's variable, the suite-wide variable)
    ("singguard_url", "SINGGUARD_URL", "ARGUS_SINGGUARD_URL"),
    ("nudenet_url", "NUDENET_URL", "ARGUS_NUDENET_URL"),
    ("comfyui_url", "COMFYUI_URL", "ARGUS_COMFYUI_URL"),
    ("ollama_url", "OLLAMA_URL", "ARGUS_OLLAMA_URL"),
]


@pytest.fixture(autouse=True)
def only_the_guard(monkeypatch) -> None:
    for _, var, _suite in [*PEERS, ("age_url", "AGE_URL", "")]:
        monkeypatch.delenv(OWN + var, raising=False)
    monkeypatch.setenv(OWN + "SINGGUARD_URL", GUARD)


@pytest.mark.parametrize(("field", "var", "suite"), PEERS)
def test_darkrooms_own_variable_wins_and_the_suite_wide_one_is_the_fallback(monkeypatch, field, var, suite) -> None:
    monkeypatch.setenv(suite, "http://suite:1/")
    monkeypatch.setenv(OWN + var, "http://own:2/")
    assert getattr(Settings.from_env(), field) == "http://own:2"
    monkeypatch.setenv(OWN + var, "")  # compose's ${VAR:-}: blank falls through
    assert getattr(Settings.from_env(), field) == "http://suite:1"


def test_with_nothing_set_only_comfyui_has_a_url(monkeypatch) -> None:
    settings = Settings.from_env()
    assert settings.comfyui_url == "http://localhost:8188"
    assert (settings.nudenet_url, settings.ollama_url, settings.age_url, settings.age_fallback_url) == ("", "", "", "")
    assert settings.sfw_available is False


def test_the_suite_file_supplies_a_url_the_environment_does_not(monkeypatch, tmp_path) -> None:
    suite = tmp_path / "suite.env"
    suite.write_text("ARGUS_NUDENET_URL=http://suite-file:8130\n")
    suite.chmod(0o600)
    monkeypatch.setenv("ARGUS_ENV_FILE", str(suite))
    assert Settings.from_env().nudenet_url == "http://suite-file:8130"


# --- a guard is never lost to a typo, and a required one still refuses ------------------------------------


@pytest.mark.parametrize("var", ["SINGGUARD_URL", "NUDENET_URL", "COMFYUI_URL", "OLLAMA_URL", "AGE_URL"])
@pytest.mark.parametrize("value", ["guard:9150", "/api/guard", "ftp://guard", "http://", "http://guard:port"])
def test_a_malformed_url_is_a_configuration_error_not_an_unconfigured_backend(monkeypatch, var, value) -> None:
    monkeypatch.setenv(OWN + var, value)
    with pytest.raises(ConfigError) as refused:
        Settings.from_env()
    assert OWN + var in str(refused.value)
    assert value not in str(refused.value)  # a URL can carry a credential: the variable is named, never the value


@pytest.mark.parametrize("suite", ["ARGUS_SINGGUARD_URL", "ARGUS_NUDENET_URL", "ARGUS_CURATOR_URL"])
def test_a_malformed_suite_wide_url_is_a_configuration_error_too(monkeypatch, suite) -> None:
    if suite == "ARGUS_SINGGUARD_URL":
        monkeypatch.delenv(OWN + "SINGGUARD_URL")
    monkeypatch.setenv(suite, "/api/somewhere")  # studio's browser path prefix, say
    with pytest.raises(ConfigError, match=suite):
        Settings.from_env()


@pytest.mark.parametrize("value", ["", "  ", "off", "none", "OFF"])
def test_singguard_blank_or_off_builds_no_engine(monkeypatch, tmp_path, value) -> None:
    monkeypatch.setenv(OWN + "SINGGUARD_URL", value)
    monkeypatch.setenv(OWN + "OUTPUT_ROOT", str(tmp_path))
    with pytest.raises(ConfigError, match="unguarded"):
        Settings.from_env()
    assert create_app().state.engine is None  # the server starts closed: tests/test_unguarded.py


def test_singguard_off_is_not_rescued_by_the_suite_wide_url(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_SINGGUARD_URL", "http://suite:9150")
    monkeypatch.setenv(OWN + "SINGGUARD_URL", "off")
    with pytest.raises(ConfigError, match="unguarded"):
        Settings.from_env()


def test_the_suite_wide_singguard_lets_the_server_start_guarded(monkeypatch) -> None:
    monkeypatch.delenv(OWN + "SINGGUARD_URL")
    monkeypatch.setenv("ARGUS_SINGGUARD_URL", "http://suite:9150")
    assert Settings.from_env().singguard_url == "http://suite:9150"


def test_nudenet_off_leaves_sfw_unavailable_whatever_the_suite_says(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_NUDENET_URL", "http://suite:8130")
    assert Settings.from_env().sfw_available is True
    monkeypatch.setenv(OWN + "NUDENET_URL", "off")
    settings = Settings.from_env()
    assert settings.nudenet_url == "" and settings.sfw_available is False  # SFW generation is refused, as when unset


def test_comfyui_off_is_a_configuration_error(monkeypatch) -> None:
    monkeypatch.setenv(OWN + "COMFYUI_URL", "off")
    with pytest.raises(ConfigError, match="COMFYUI_URL"):
        Settings.from_env()


# --- the age check: the suite-wide curator can turn it on, never replace an estimator that works ----------


def test_the_suite_wide_curator_is_only_a_fallback_for_the_age_check(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_URL", "http://suite:8101")
    settings = Settings.from_env()
    assert (settings.age_url, settings.age_fallback_url) == ("", "http://suite:8101")
    monkeypatch.setenv(OWN + "AGE_URL", "http://own:8101")
    settings = Settings.from_env()
    assert (settings.age_url, settings.age_fallback_url) == ("http://own:8101", "")
    monkeypatch.setenv(OWN + "AGE_URL", "off")  # no remote check: the in-process extra, if installed, still runs
    settings = Settings.from_env()
    assert (settings.age_url, settings.age_fallback_url) == ("", "")


def test_a_working_in_process_estimator_is_kept_over_the_suite_wide_curator(monkeypatch) -> None:
    monkeypatch.setattr(age, "age_status", lambda: AgeStatus(True))
    monkeypatch.setattr(age, "CuratorAgeEstimator", lambda: "in-process")
    # A curator that is down: taking it would trade a check that runs for one that cannot.
    estimator, status = default_estimator("", "http://127.0.0.1:1")
    assert estimator == "in-process" and status.available


def test_the_suite_wide_curator_turns_the_age_check_on_when_nothing_else_can_run(monkeypatch) -> None:
    monkeypatch.setattr(age, "age_status", lambda: AgeStatus(False, "not installed"))
    assert default_estimator("", "") == (None, AgeStatus(False, "not installed"))
    with FakeFaces(faces=[{"age_estimate_years": 21, "bbox": [0, 0, 1, 1]}]) as fake:
        estimator, status = default_estimator("", fake.url)
        assert isinstance(estimator, HttpAgeEstimator) and status.available
        assert estimator.estimate(b"x")[0].age_estimate_years == 21.0


def test_darkrooms_own_age_url_still_wins_over_the_in_process_estimator(monkeypatch) -> None:
    monkeypatch.setattr(age, "age_status", lambda: AgeStatus(True))
    estimator, _ = default_estimator("http://127.0.0.1:1", "")
    assert isinstance(estimator, HttpAgeEstimator) and not isinstance(estimator, CuratorAgeEstimator)
