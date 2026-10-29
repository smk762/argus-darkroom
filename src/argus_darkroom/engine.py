"""The engine: assemble, pre-flight, render and record.

Rendering returns the image in memory with the record of how it was made. Nothing here writes the image:
that is the release step, which runs only after the scans.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from argus_cortex.moderation import policies as policy_table

from argus_darkroom import __version__
from argus_darkroom.comfyui import ComfyUIClient, GenerationError, Progress
from argus_darkroom.config import Settings
from argus_darkroom.guard import Guard
from argus_darkroom.modelfiles import ModelLibrary
from argus_darkroom.models import GenerationRecord, GuardRecord, LoRARef, ModelRef, RequestRecord
from argus_darkroom.preflight import PreflightResult, run_preflight
from argus_darkroom.prompt import assemble_prompt
from argus_darkroom.request import GenerateRequest, RequestError, check_policies
from argus_darkroom.store import RecordStore, new_id
from argus_darkroom.workflow import SAVE_NODE, RenderParams, Workflow, load_workflow, render


@dataclass(frozen=True)
class Rendered:
    """A generated image, held in memory, and the request record that reproduces it."""

    image: bytes
    request: RequestRecord


class Engine:
    def __init__(
        self,
        settings: Settings,
        guard: Guard | None = None,
        store: RecordStore | None = None,
        comfyui: ComfyUIClient | None = None,
        workflow: Workflow | None = None,
        models: ModelLibrary | None = None,
    ) -> None:
        self.settings = settings
        self.guard = guard or Guard(settings.singguard_url, settings.thinking)
        self.store = store or RecordStore(settings.output_root)
        self.comfyui = comfyui or ComfyUIClient(settings.comfyui_url)
        self.models = models or ModelLibrary(settings.models_root)
        # A template that could write before the verdict is refused here, at load, not when it runs.
        self.workflow = workflow or (load_workflow(settings.workflow) if settings.workflow else None)
        self._outputs_verified: set[str] = set()

    # -- request ----------------------------------------------------------------------------------------

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

    # -- pre-flight -------------------------------------------------------------------------------------

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

    def _request_record(
        self,
        request: GenerateRequest,
        enforcing: str,
        expanded_prompt: str | None,
        *,
        seed: int | None = None,
        checkpoint: ModelRef | None = None,
        vae: ModelRef | None = None,
        loras: tuple[LoRARef, ...] = (),
        workflow_sha256: str | None = None,
        engine_version: str | None = None,
    ) -> RequestRecord:
        return RequestRecord(
            mode=policy_table.mode(enforcing),
            user_prompt=request.user_prompt,
            expanded_prompt=expanded_prompt,
            final_prompt=self.final_prompt(request, expanded_prompt),
            negative_prompt=request.negative_prompt,
            seed=seed if seed is not None else request.seed,
            sampling=request.sampling,
            checkpoint=checkpoint,
            vae=vae,
            loras=list(loras),
            workflow_sha256=workflow_sha256,
            engine_version=engine_version,
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
            request=self._request_record(request, enforcing, expanded_prompt),
            guard=GuardRecord(
                enforcing_policy=enforcing,
                advisory_policies=advisory,
                thinking=request.thinking or self.settings.thinking,
                preflight=result.cells,
            ),
        )
        self.store.save(record)
        return record

    # -- render -----------------------------------------------------------------------------------------

    def render(
        self,
        request: GenerateRequest,
        expanded_prompt: str | None = None,
        on_progress: Progress | None = None,
    ) -> Rendered:
        """Generate the image for *request*: resolve and hash its models, render, and hold the PNG in memory.

        Call it only after pre-flight let the request through. The final prompt is the one that was checked:
        it is built by :meth:`final_prompt`, the same function pre-flight used.

        Raises :class:`RequestError` (nothing was sent to ComfyUI) or :class:`GenerationError`.
        """
        enforcing, _ = self.policies(request)
        self.require_generation_ready(enforcing)
        workflow = self.workflow
        if workflow is None:
            raise RequestError("no workflow is configured (ARGUS_DARKROOM_WORKFLOW)")
        if request.sampling is None or not request.checkpoint:
            raise RequestError("a generation needs sampling parameters and a checkpoint")

        # Resolve and hash every model first: a missing file fails before any GPU time is spent.
        checkpoint = self.models.ref("checkpoint", request.checkpoint)
        vae = self.models.ref("vae", request.vae) if request.vae else None
        loras = tuple(self.models.lora_ref(lora.name, lora.weight) for lora in request.loras)
        seed = request.seed if request.seed is not None else secrets.randbits(32)

        self._verify_outputs(workflow)
        final_prompt = self.final_prompt(request, expanded_prompt)
        graph = render(
            workflow,
            RenderParams(
                positive=final_prompt,
                negative=request.negative_prompt,
                seed=seed,
                sampling=request.sampling,
                checkpoint=checkpoint.name,
                vae=vae.name if vae else None,
                loras=loras,
            ),
        )
        result = self.comfyui.render(graph, workflow.save_node_id, on_progress)
        record = self._request_record(
            request,
            enforcing,
            expanded_prompt,
            seed=seed,
            checkpoint=checkpoint,
            vae=vae,
            loras=loras,
            workflow_sha256=workflow.sha256,
            engine_version=self.comfyui.engine_version(),
        )
        return Rendered(image=result.image, request=record)

    def _verify_outputs(self, workflow: Workflow) -> None:
        """Ask ComfyUI which of the workflow's nodes are output nodes, once per workflow.

        The load-time check goes by node names. This one catches a node that writes but is not named like it.
        """
        if workflow.sha256 in self._outputs_verified:
            return
        others = sorted(c for c, is_output in self.comfyui.output_classes(workflow.class_types).items() if is_output)
        others = [c for c in others if c != SAVE_NODE]
        if others:
            raise GenerationError(
                f"the workflow has output node(s) other than {SAVE_NODE}: {', '.join(others)}; "
                "nothing may be written before the verdict"
            )
        self._outputs_verified.add(workflow.sha256)
