"""The engine: one guarded generation, from prompt to released or destroyed image, and its record.

:meth:`Engine.generate` runs the whole flow: assemble, pre-flight, render, scan, decide, then release or
destroy. The image is held in memory from the moment ComfyUI returns it until the verdict. Only a passed or
review image is ever written (the last step, :meth:`Engine._release`); a deleted or omitted one is simply not,
and its buffer goes out of scope. A record is kept for every attempt either way.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from argus_cortex.moderation import NudeNetClient
from argus_cortex.moderation import policies as policy_table

from argus_darkroom import __version__
from argus_darkroom.age import AgeEstimator, AgeStatus, default_estimator
from argus_darkroom.comfyui import ComfyUIClient, GenerationError, Progress
from argus_darkroom.config import Settings
from argus_darkroom.guard import Guard
from argus_darkroom.images import UnreadableImage, describe
from argus_darkroom.modelfiles import ModelLibrary
from argus_darkroom.models import (
    AgeRecord,
    GenerationRecord,
    GuardRecord,
    LoRARef,
    ModelRef,
    NudeNetRecord,
    RequestRecord,
)
from argus_darkroom.outcome import decide, minor_exposed
from argus_darkroom.preflight import PreflightResult, run_preflight
from argus_darkroom.prompt import assemble_prompt
from argus_darkroom.request import GenerateRequest, RequestError, check_policies
from argus_darkroom.scan import Scanner
from argus_darkroom.store import RecordStore, new_id
from argus_darkroom.workflow import SAVE_NODE, RenderParams, Workflow, load_workflow, render

logger = structlog.get_logger()

# (event kind, JSON-able payload): "preflight", "queued", "progress", "scan", "complete"
Emit = Callable[[str, dict], None]
_REASON_MAX = 500


@dataclass(frozen=True)
class Prepared:
    """A request with its models found and hashed and its seed fixed, before any GPU time is spent."""

    seed: int
    checkpoint: ModelRef
    vae: ModelRef | None
    loras: tuple[LoRARef, ...]


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
        nudenet: NudeNetClient | None = None,
        age: AgeEstimator | None = None,
    ) -> None:
        self.settings = settings
        self.guard = guard or Guard(settings.singguard_url, settings.thinking)
        self.store = store or RecordStore(settings.output_root)
        self.comfyui = comfyui or ComfyUIClient(settings.comfyui_url)
        self.models = models or ModelLibrary(settings.models_root)
        # A template that could write before the verdict is refused here, at load, not when it runs.
        self.workflow = workflow or (load_workflow(settings.workflow) if settings.workflow else None)
        self._outputs_verified: set[str] = set()
        if nudenet is None and settings.nudenet_url:
            nudenet = NudeNetClient(settings.nudenet_url)
        if age is None:
            age, self.age_status = default_estimator()
        else:
            self.age_status = AgeStatus(True)
        self.scanner = Scanner(self.guard, nudenet, age)

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

    def prepare(self, request: GenerateRequest) -> Prepared:
        """Check a request can be rendered, and resolve and hash its models: a missing file fails before any
        GPU time (or pre-flight time) is spent.

        Raises :class:`RequestError`.
        """
        enforcing, _ = self.policies(request)
        self.require_generation_ready(enforcing)
        if self.workflow is None:
            raise RequestError("no workflow is configured (ARGUS_DARKROOM_WORKFLOW)")
        if request.sampling is None or not request.checkpoint:
            raise RequestError("a generation needs sampling parameters and a checkpoint")
        return Prepared(
            seed=request.seed if request.seed is not None else secrets.randbits(32),
            checkpoint=self.models.ref("checkpoint", request.checkpoint),
            vae=self.models.ref("vae", request.vae) if request.vae else None,
            loras=tuple(self.models.lora_ref(lora.name, lora.weight) for lora in request.loras),
        )

    def render(
        self,
        request: GenerateRequest,
        expanded_prompt: str | None = None,
        on_progress: Progress | None = None,
        prepared: Prepared | None = None,
    ) -> Rendered:
        """Generate the image for *request* and hold the PNG in memory.

        Call it only after pre-flight let the request through. The final prompt is the one that was checked:
        it is built by :meth:`final_prompt`, the same function pre-flight used.

        Raises :class:`RequestError` (nothing was sent to ComfyUI) or :class:`GenerationError`.
        """
        prepared = prepared or self.prepare(request)
        workflow = self.workflow
        assert workflow is not None and request.sampling is not None  # prepare() checked
        enforcing, _ = self.policies(request)

        self._verify_outputs(workflow)
        graph = render(
            workflow,
            RenderParams(
                positive=self.final_prompt(request, expanded_prompt),
                negative=request.negative_prompt,
                seed=prepared.seed,
                sampling=request.sampling,
                checkpoint=prepared.checkpoint.name,
                vae=prepared.vae.name if prepared.vae else None,
                loras=prepared.loras,
            ),
        )
        result = self.comfyui.render(graph, workflow.save_node_id, on_progress)
        return Rendered(
            image=result.image,
            request=self._prepared_record(
                request, enforcing, expanded_prompt, prepared, engine_version=self.comfyui.engine_version()
            ),
        )

    def _prepared_record(
        self,
        request: GenerateRequest,
        enforcing: str,
        expanded_prompt: str | None,
        prepared: Prepared,
        engine_version: str | None = None,
    ) -> RequestRecord:
        return self._request_record(
            request,
            enforcing,
            expanded_prompt,
            seed=prepared.seed,
            checkpoint=prepared.checkpoint,
            vae=prepared.vae,
            loras=prepared.loras,
            workflow_sha256=self.workflow.sha256 if self.workflow else None,
            engine_version=engine_version,
        )

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

    # -- the whole flow ---------------------------------------------------------------------------------

    def generate(self, request: GenerateRequest, emit: Emit | None = None) -> GenerationRecord:
        """One guarded generation. Returns the record, whatever the outcome; never an image.

        1. pre-flight (text only): a refusal is recorded and nothing is generated;
        2. render: the image is held in memory only (a ComfyUI failure is a ``failed`` record);
        3. scan, decide;
        4. release (write, for passed and review) or destroy (write nothing).

        Raises :class:`RequestError` (HTTP 400) before anything happens when the request cannot be served.
        """

        def send(kind: str, data: dict) -> None:
            if emit:
                emit(kind, data)

        enforcing, advisory = self.policies(request)
        prepared = self.prepare(request)
        thinking = request.thinking or self.settings.thinking

        pre = self.preflight(request)
        send("preflight", _preflight_payload(pre))
        if pre.refused:
            record = self.record_refusal(request, pre)
            send("complete", record.model_dump(mode="json"))
            return record

        guard_record = GuardRecord(
            enforcing_policy=enforcing, advisory_policies=advisory, thinking=thinking, preflight=pre.cells
        )
        record_id = new_id()
        try:
            send("queued", {})
            rendered = self.render(
                request, prepared=prepared, on_progress=lambda step, of: send("progress", {"step": step, "of": of})
            )
            image = describe(rendered.image)
        except (GenerationError, UnreadableImage) as exc:
            record = self._save(
                GenerationRecord(
                    id=record_id,
                    darkroom_version=__version__,
                    outcome="failed",
                    outcome_reason=str(exc)[:_REASON_MAX],
                    request=self._prepared_record(request, enforcing, None, prepared),
                    guard=guard_record,
                )
            )
            send("complete", record.model_dump(mode="json"))
            return record

        # From here the bytes exist only in `rendered.image`. Nothing is written until the verdict.
        scan = self.scanner.scan(
            rendered.image,
            pre.framed,
            [enforcing, *advisory],
            thinking,
            on_cell=lambda policy, context, cell: send(
                "scan", {"policy": policy, "context": context, "cell": cell.model_dump(mode="json")}
            ),
        )
        exposed = scan.nsfw.exposed_score if scan.nsfw is not None else None
        age = AgeRecord(faces=scan.faces) if scan.faces is not None else None
        if age is not None:
            age.minor_exposed = minor_exposed(age, exposed)
        decision = decide(enforcing, scan.scans, exposed, age, request.nsfw_threshold)

        guard_record.scans = scan.scans
        guard_record.age = age
        if scan.nsfw is not None:
            guard_record.nudenet = NudeNetRecord(
                exposed_score=scan.nsfw.exposed_score,
                labels=scan.nsfw.labels,
                threshold=request.nsfw_threshold,
                blur=decision.blur,
            )
        image.blur = decision.blur
        record = GenerationRecord(
            id=record_id,
            darkroom_version=__version__,
            outcome=decision.outcome,
            outcome_reason=decision.reason,
            critical=decision.critical,
            request=rendered.request,
            guard=guard_record,
            image=image,
        )
        if decision.released:
            record = self._release(record, rendered.image)
        else:
            self.store.save(record)  # a deleted or omitted image is never written: this is all that is kept
        send("complete", record.model_dump(mode="json"))
        return record

    def _save(self, record: GenerationRecord) -> GenerationRecord:
        self.store.save(record)
        return record

    def _release(self, record: GenerationRecord, data: bytes) -> GenerationRecord:
        """Write a passed or review image and its record. If the record cannot be saved the image goes too."""
        assert record.image is not None
        record.image.path = self.store.write_image(record.id, data)
        try:
            self.store.save(record)
        except Exception:
            self.store.remove_image(record.id)
            raise
        return record


def _preflight_payload(pre: PreflightResult) -> dict:
    return {
        "verdict": pre.verdict,
        "reason": pre.reason,
        "refusing_policy": pre.refusing_policy,
        "warnings": pre.warnings,
        "cells": {name: cell.model_dump(mode="json") for name, cell in pre.cells.items()},
    }
