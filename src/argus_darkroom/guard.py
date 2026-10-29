"""SingGuard cells: one policy's verdict in one context, as a state the outcome table reads.

A verdict is a pass or a fail with the rules that fired, not a score. What a rule letter means
depends on the policy, so whether a fired rule is *illegal* is looked up per policy in
``argus_cortex.moderation.policies``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from argus_cortex.moderation import (
    CheckFailed,
    GuardResult,
    SingGuardClient,
    rule_letter,
)
from argus_cortex.moderation import policies as policy_table

from argus_darkroom.models import Cell

DEFAULT_TIMEOUT = 180.0  # "cascade" runs the slow pass on whatever fast flags


def illegal_rules_fired(policy: str, result: GuardResult) -> list[str]:
    """The rules in *result* that are illegal under *policy*, as the policy names them."""
    return [rule for rule in result.rules if policy_table.is_illegal(policy, rule)]


def classify(policy: str, result: GuardResult) -> Cell:
    """*result* as a cell: ``safe``, ``illegal`` (an illegal rule fired) or ``flagged`` (other rules only)."""
    if result.safe:
        return Cell(state="safe", result=result)
    return Cell(state="illegal" if illegal_rules_fired(policy, result) else "flagged", result=result)


def unchecked(error: Exception | str) -> Cell:
    """A check that failed: timeout, HTTP error, no verdict. Never a pass."""
    return Cell(state="unchecked", error=str(error) or type(error).__name__)


def rule_codes(policy: str, rules: list[str]) -> list[str]:
    """``policy:letter`` for each rule, so a code stays unambiguous when several policies ran."""
    return [f"{policy}:{rule_letter(rule).removesuffix('.')}" for rule in rules]


class Guard:
    """SingGuard at one URL, one client per darkroom policy name.

    Requests name policies; the text comes from cortex, so a request can never carry its own.
    """

    def __init__(
        self,
        base_url: str,
        thinking: str = "cascade",
        timeout: float = DEFAULT_TIMEOUT,
        client_factory: Callable[[str, str, float, str | None], SingGuardClient] | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("a SingGuard URL is required: darkroom never runs unguarded")
        self.base_url = base_url
        self.thinking = thinking
        self.timeout = timeout
        self._factory = client_factory or _default_client
        self._clients: dict[tuple[str, str], SingGuardClient] = {}
        self._lock = threading.Lock()

    def client(self, policy: str, thinking: str | None = None) -> SingGuardClient:
        thinking = thinking or self.thinking
        key = (policy, thinking)
        with self._lock:
            if key not in self._clients:
                self._clients[key] = self._factory(
                    self.base_url, thinking, self.timeout, policy_table.policy_text(policy)
                )
            return self._clients[key]

    def check_text(self, policy: str, text: str, thinking: str | None = None) -> Cell:
        """A text-only check of *text* under *policy*; a failure is an ``unchecked`` cell, not an exception."""
        try:
            return classify(policy, self.client(policy, thinking).check_text(text))
        except CheckFailed as exc:
            return unchecked(exc)


def _default_client(base_url: str, thinking: str, timeout: float, policy: str | None) -> SingGuardClient:
    return SingGuardClient(base_url, timeout=timeout, thinking=thinking, policy=policy)
