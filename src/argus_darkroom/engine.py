"""The engine: assemble, pre-flight, and record. No generation yet (ComfyUI is the next stage)."""

from __future__ import annotations

from argus_cortex.moderation import policies as policy_table

from argus_darkroom import __version__
from argus_darkroom.config import Settings
from argus_darkroom.guard import Guard
from argus_darkroom.models import GenerationRecord, GuardRecord, RequestRecord
from argus_darkroom.preflight import PreflightResult, run_preflight
from argus_darkroom.prompt import assemble_prompt
from argus_darkroom.request import GenerateRequest, RequestError, check_policies
from argus_darkroom.store import RecordStore, new_id


class Engine:
    def __init__(self, settings: Settings, guard: Guard | None = None, store: RecordStore | None = None) -> None:
        self.settings = settings
        self.guard = guard or Guard(settings.singguard_url, settings.thinking)
        self.store = store or RecordStore(settings.output_root)

    def policies(self, request: GenerateRequest) -> tuple[str, list[str]]:
        """The enforcing policy and the advisory ones this request selects, validated.

        Raises :class:`RequestError` (HTTP 400) for an unknown policy, or an NSFW policy enforcing without
        ``--allow-nsfw``.
        """
        enforcing = request.enforcing_policy or self.settings.policy
        return enforcing, check_policies(enforcing, request.advisory_policies, self.settings.allow_nsfw)

    def require_generation_ready(self, enforcing_policy: str) -> None:
        """SFW output is gated on NudeNet; without ``NUDENET_URL`` no SFW generation can be served."""
        if policy_table.mode(enforcing_policy) == "sfw" and not self.settings.sfw_available:
            raise RequestError(
                "SFW generation needs NudeNet (ARGUS_DARKROOM_NUDENET_URL is empty), so an SFW output "
                "can never be checked for exposure"
            )

    def final_prompt(self, request: GenerateRequest, expanded_prompt: str | None = None) -> str:
        return assemble_prompt(request.user_prompt, expanded_prompt, request.trigger_words)

    def preflight(self, request: GenerateRequest, expanded_prompt: str | None = None) -> PreflightResult:
        """The text-only check, alone. Writes no record: a form check is not an attempt."""
        enforcing, advisory = self.policies(request)
        return run_preflight(
            self.guard,
            self.final_prompt(request, expanded_prompt),
            request.negative_prompt,
            enforcing,
            advisory,
            request.thinking,
        )

    def record_refusal(
        self, request: GenerateRequest, result: PreflightResult, expanded_prompt: str | None = None
    ) -> GenerationRecord:
        """The record of an attempt that pre-flight refused. Nothing was generated, so there is no image."""
        enforcing, advisory = self.policies(request)
        record = GenerationRecord(
            id=new_id(),
            darkroom_version=__version__,
            outcome="refused",
            outcome_reason=result.reason,
            critical=False,
            request=RequestRecord(
                mode=policy_table.mode(enforcing),
                user_prompt=request.user_prompt,
                expanded_prompt=expanded_prompt,
                final_prompt=self.final_prompt(request, expanded_prompt),
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                sampling=request.sampling,
            ),
            guard=GuardRecord(
                enforcing_policy=enforcing,
                advisory_policies=advisory,
                thinking=request.thinking or self.settings.thinking,
                preflight=result.cells,
            ),
        )
        self.store.save(record)
        return record
