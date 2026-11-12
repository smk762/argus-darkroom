"""The engine: one guarded generation, from prompt to released or destroyed image, and its record.

:meth:`Engine.generate` runs the whole flow: assemble, pre-flight, scan the input image (image-to-image only),
render, scan, decide, then release or destroy. The image is held in memory from the moment ComfyUI returns it until
the verdict. Only a passed or review image is ever written (the last step, :meth:`Engine._release`); a deleted or
omitted one is simply not, and its buffer goes out of scope. A record is kept for every attempt either way.

An input image gets the same checks an output does, before anything reaches ComfyUI, and the run goes on only if its
outcome is ``passed``. It is never written: it reaches ComfyUI inside the graph, which is deleted from ComfyUI's
history after the run.

An inpainting mask travels the same way and is never written either, but it is not scanned: it only says which
pixels of the checked input are repainted, and carries no imagery. The output is scanned and decided as usual.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from argus_cortex.moderation import NudeNetClient, rule_letter
from argus_cortex.moderation import policies as policy_table

from argus_darkroom import __version__
from argus_darkroom.age import AgeEstimator, AgeStatus, default_estimator
from argus_darkroom.comfyui import ComfyUIClient, GenerationError, Progress
from argus_darkroom.config import Settings
from argus_darkroom.expansion import ExpansionError, OllamaExpander
from argus_darkroom.guard import Guard
from argus_darkroom.images import InputImage, InputMask, UnreadableImage, describe, load_input, load_mask
from argus_darkroom.modelfiles import ModelLibrary
from argus_darkroom.models import (
    AgeRecord,
    GenerationRecord,
    GuardRecord,
    InitImageRef,
    InputScanRecord,
    LoRARef,
    MaskRef,
    ModelRef,
    NudeNetRecord,
    RequestRecord,
)
from argus_darkroom.outcome import Decision, decide, minor_exposed
from argus_darkroom.preflight import PreflightResult, run_preflight
from argus_darkroom.prompt import assemble_prompt
from argus_darkroom.request import GenerateRequest, RequestError, check_policies
from argus_darkroom.scan import Scan, Scanner
from argus_darkroom.store import RecordStore, new_id
from argus_darkroom.workflow import (
    SAVE_NODE,
    Mode,
    RenderParams,
    Workflow,
    WorkflowError,
    WorkflowSet,
    load_workflows,
    render,
)

logger = structlog.get_logger()

# (event kind, JSON-able payload): "preflight", "input_scan", "queued", "progress", "scan", "complete"
Emit = Callable[[str, dict], None]
_REASON_MAX = 500
INPUT_REFUSAL = "input image: "
# A run of base64 this long in an error message is a piece of the input image (ComfyUI can echo a node's inputs).
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")


def _scrub(reason: str) -> str:
    """An error message fit for a record: no piece of an input image."""
    return _BASE64_RUN.sub("<base64 removed>", reason)


@dataclass(frozen=True)
class Prepared:
    """A request with its workflow chosen, its models found and hashed and its seed fixed, before any GPU time is
    spent."""

    workflow: Workflow
    seed: int
    checkpoint: ModelRef
    vae: ModelRef | None
    loras: tuple[LoRARef, ...]
    # Image-to-image: the input, decoded and held in memory only, and its denoise. None for text-to-image.
    init: InputImage | None = None
    denoise: float | None = None
    # Inpainting: the mask, decoded and held in memory only. Never scanned: it carries no imagery.
    mask: InputMask | None = None

    @property
    def init_ref(self) -> InitImageRef | None:
        if self.init is None or self.denoise is None:
            return None
        image = self.init.record
        return InitImageRef(
            sha256=image.sha256,
            phash=image.phash,
            width=image.width,
            height=image.height,
            source_sha256=self.init.source_sha256,
            denoise=self.denoise,
            mask=None
            if self.mask is None
            else MaskRef(
                sha256=self.mask.sha256,
                source_sha256=self.mask.source_sha256,
                masked_fraction=self.mask.masked_fraction,
            ),
        )


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
        workflows: WorkflowSet | None = None,
        models: ModelLibrary | None = None,
        nudenet: NudeNetClient | None = None,
        age: AgeEstimator | None = None,
        expander: OllamaExpander | None = None,
    ) -> None:
        self.settings = settings
        self.guard = guard or Guard(settings.singguard_url, settings.thinking)
        self.store = store or RecordStore(settings.output_root)
        self.comfyui = comfyui or ComfyUIClient(settings.comfyui_url)
        self.models = models or ModelLibrary(settings.models_root)
        # A template that could write before the verdict is refused here, at load, not when it runs.
        self.workflows = workflows or (load_workflows(settings.workflow) if settings.workflow else None)
        self._outputs_verified: set[str] = set()
        if nudenet is None and settings.nudenet_url:
            nudenet = NudeNetClient(settings.nudenet_url)
        if age is None:
            age, self.age_status = default_estimator(settings.age_url)
        else:
            self.age_status = AgeStatus(True)
        self.scanner = Scanner(self.guard, nudenet, age)
        if expander is None and settings.expansion_enabled:
            expander = OllamaExpander(settings.ollama_url, settings.ollama_model)
        self.expander = expander

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

    def preflight_view(
        self, request: GenerateRequest, pre: PreflightResult, expanded_prompt: str | None = None
    ) -> dict:
        """*pre* as a client shows it (``/preflight`` and the ``preflight`` event): the prompts that were judged, an
        ``outcome`` of ``ok``, ``warning`` or ``refused``, and the cell for every policy under ``results``.

        A ``warning`` is the enforcing policy flagging under an NSFW policy: generation proceeds and the output
        scan decides. An advisory flag is only a cell in ``results``.
        """
        enforcing, advisory = self.policies(request)
        refusal, warnings = None, []
        if pre.refused and pre.refusing_policy is not None:
            cell = pre.cells[pre.refusing_policy]
            rules = cell.result.rules if cell.result is not None else []
            refusal = {
                "policy": pre.refusing_policy,
                "reason": rule_letter(rules[0]).removesuffix(".") if rules else "unchecked",
                "rule": rules[0] if rules else None,
                "explanation": (cell.result.explanation if cell.result is not None else cell.error),
            }
        elif pre.cells[enforcing].state == "flagged" and pre.cells[enforcing].result is not None:
            result = pre.cells[enforcing].result
            rule = result.rules[0] if result.rules else None
            warnings.append({"policy": enforcing, "rule": rule, "explanation": result.explanation})
        return {
            "mode": policy_table.mode(enforcing),
            "enforcing_policy": enforcing,
            "advisory_policies": advisory,
            "thinking": request.thinking or self.settings.thinking,
            "user_prompt": request.user_prompt,
            "expanded_prompt": expanded_prompt,
            "final_prompt": self.final_prompt(request, expanded_prompt),
            "negative_prompt": request.negative_prompt,
            "framed_text": pre.framed,
            "outcome": "refused" if refusal else "warning" if warnings else "ok",
            "refusal": refusal,
            "warnings": warnings,
            "results": {name: cell.model_dump(mode="json") for name, cell in pre.cells.items()},
        }

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
        workflow: Workflow | None = None,
        engine_version: str | None = None,
        init_image: InitImageRef | None = None,
    ) -> RequestRecord:
        return RequestRecord(
            mode=policy_table.mode(enforcing),
            user_prompt=request.user_prompt,
            expanded_prompt=expanded_prompt,
            expansion=self.expander.info if expanded_prompt is not None and self.expander else None,
            final_prompt=self.final_prompt(request, expanded_prompt),
            negative_prompt=request.negative_prompt,
            seed=seed if seed is not None else request.seed,
            sampling=request.sampling,
            checkpoint=checkpoint,
            vae=vae,
            loras=list(loras),
            workflow_sha256=workflow.sha256 if workflow else None,
            workflow=workflow.name if workflow else None,
            engine_version=engine_version,
            init_image=init_image,
        )

    def record_refusal(
        self,
        request: GenerateRequest,
        result: PreflightResult,
        expanded_prompt: str | None = None,
        init_image: InitImageRef | None = None,
    ) -> GenerationRecord:
        """The record of an attempt that pre-flight refused. Nothing was generated, so there is no image. An input
        image was not scanned either: the record names it by its hashes only."""
        enforcing, advisory = self.policies(request)
        record = GenerationRecord(
            id=new_id(),
            darkroom_version=__version__,
            outcome="refused",
            outcome_reason=result.reason,
            critical=False,
            request=self._request_record(request, enforcing, expanded_prompt, init_image=init_image),
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
        """Check a request can be rendered, choose its workflow, and resolve and hash its models: a missing file
        or a family with no workflow fails before any GPU time (or pre-flight time) is spent.

        Raises :class:`RequestError`.
        """
        enforcing, _ = self.policies(request)
        self.require_generation_ready(enforcing)
        if self.workflows is None:
            raise RequestError("no workflow is configured (ARGUS_DARKROOM_WORKFLOW)")
        if request.sampling is None or not request.checkpoint:
            raise RequestError("a generation needs sampling parameters and a checkpoint")
        init: InputImage | None = None
        mask: InputMask | None = None
        if request.init_image is not None:
            try:
                init = load_input(request.init_image.data_base64)
                if request.init_image.mask_base64 is not None:
                    mask = load_mask(request.init_image.mask_base64, (init.record.width, init.record.height))
            except UnreadableImage as exc:
                raise RequestError(str(exc)) from exc
        mode: Mode = "txt2img" if init is None else "img2img" if mask is None else "inpaint"
        try:
            workflow = self.workflows.select(request.checkpoint, len(request.loras), mode)
        except WorkflowError as exc:
            raise RequestError(str(exc)) from exc
        prepared = Prepared(
            workflow=workflow,
            seed=request.seed if request.seed is not None else secrets.randbits(32),
            checkpoint=self.models.ref("checkpoint", request.checkpoint),
            vae=self.models.ref("vae", request.vae) if request.vae else None,
            loras=tuple(self.models.lora_ref(lora.name, lora.weight, lora.trigger_words) for lora in request.loras),
            init=init,
            denoise=request.init_image.denoise if request.init_image is not None else None,
            mask=mask,
        )
        try:  # the template's slots must fit this request: better a 400 now than a failure later
            render(workflow, self._params(request, prepared, ""))
        except WorkflowError as exc:
            raise RequestError(str(exc)) from exc
        return prepared

    @staticmethod
    def _params(request: GenerateRequest, prepared: Prepared, positive: str) -> RenderParams:
        assert request.sampling is not None  # prepare() checked
        return RenderParams(
            positive=positive,
            negative=request.negative_prompt,
            seed=prepared.seed,
            sampling=request.sampling,
            checkpoint=prepared.checkpoint.name,
            vae=prepared.vae.name if prepared.vae else None,
            loras=prepared.loras,
            init_image=prepared.init.base64 if prepared.init is not None else None,
            denoise=prepared.denoise,
            init_mask=prepared.mask.base64 if prepared.mask is not None else None,
        )

    def render(
        self,
        request: GenerateRequest,
        expanded_prompt: str | None = None,
        on_progress: Progress | None = None,
        prepared: Prepared | None = None,
        input_scan: InputScanRecord | None = None,
    ) -> Rendered:
        """Generate the image for *request* and hold the PNG in memory.

        Call it only after pre-flight let the request through, and after an input image passed its scan
        (:meth:`check_input`). The final prompt is the one that was checked: it is built by :meth:`final_prompt`, the
        same function pre-flight used.

        Raises :class:`RequestError` (nothing was sent to ComfyUI) or :class:`GenerationError`.
        """
        prepared = prepared or self.prepare(request)
        workflow = prepared.workflow
        enforcing, _ = self.policies(request)
        if prepared.init is not None and (input_scan is None or input_scan.outcome != "passed"):
            # Not reachable through generate(), which never renders an input that did not pass. A direct caller gets
            # the same rule: an input image reaches ComfyUI only after its own scan passed.
            raise RequestError("an input image is rendered only after its scan passed (Engine.check_input)")

        self._verify_outputs(workflow)
        graph = render(workflow, self._params(request, prepared, self.final_prompt(request, expanded_prompt)))
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
            workflow=prepared.workflow,
            engine_version=engine_version,
            init_image=prepared.init_ref,
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

    # -- the input image -------------------------------------------------------------------------------

    def _judge(
        self, request: GenerateRequest, enforcing: str, scan: Scan
    ) -> tuple[Decision, AgeRecord | None, NudeNetRecord]:
        """The outcome table over one image's scan: the same for an input image as for an output."""
        exposed = scan.nsfw.exposed_score if scan.nsfw is not None else None
        age = AgeRecord(faces=scan.faces) if scan.faces is not None else None
        if age is not None:
            age.minor_exposed = minor_exposed(age, exposed)
        decision = decide(enforcing, scan.scans, exposed, age, request.nsfw_threshold)
        nudenet = NudeNetRecord(
            exposed_score=exposed,
            labels=scan.nsfw.labels if scan.nsfw is not None else {},
            threshold=request.nsfw_threshold,
            blur=decision.blur,
        )
        return decision, age, nudenet

    def check_input(
        self,
        request: GenerateRequest,
        init: InputImage,
        framed_prompt: str,
        on_cell: Callable[[str, str, dict], None] | None = None,
    ) -> tuple[InputScanRecord, Decision]:
        """Scan an image-to-image input as if it were an output: SingGuard isolated and with the prompt under every
        selected policy, NudeNet, and the age estimate, then the outcome table.

        The input may go on only when the outcome is ``passed``. Anything else (deleted, omitted, review, an
        enforcing cell that could not be checked) refuses the attempt: an input is not served behind a shield.

        An inpainting mask is not part of this scan: it is a region selector, not imagery.
        """
        enforcing, advisory = self.policies(request)
        scan = self.scanner.scan(
            init.png,
            framed_prompt,
            [enforcing, *advisory],
            request.thinking or self.settings.thinking,
            on_cell=(lambda policy, context, cell: on_cell(policy, context, cell.model_dump(mode="json")))
            if on_cell
            else None,
        )
        decision, age, nudenet = self._judge(request, enforcing, scan)
        return (
            InputScanRecord(
                scans=scan.scans, nudenet=nudenet, age=age, outcome=decision.outcome, reason=decision.reason
            ),
            decision,
        )

    # -- the whole flow ---------------------------------------------------------------------------------

    def generate(self, request: GenerateRequest, emit: Emit | None = None) -> GenerationRecord:
        """One guarded generation. Returns the record, whatever the outcome; never an image.

        1. pre-flight (text only): a refusal is recorded and nothing is generated;
        2. image-to-image only: the input image is scanned like an output, and anything but ``passed`` refuses;
        3. render: the image is held in memory only (a ComfyUI failure is a ``failed`` record);
        4. scan, decide;
        5. release (write, for passed and review) or destroy (write nothing).

        Raises :class:`RequestError` (HTTP 400) before anything happens when the request cannot be served.
        """

        return self.begin(request)(emit)

    def begin(self, request: GenerateRequest) -> Callable[[Emit | None], GenerationRecord]:
        """Validate *request* and fix everything that can fail on its own account, then return the run.

        Splitting it from the run lets a server answer HTTP 400 before it opens a stream. Raises
        :class:`RequestError`: an unknown or NSFW-without-flag policy, SFW without NudeNet, no workflow for the
        checkpoint's family, LoRA count and mode, a missing model, sampling or checkpoint, a value the workflow has no
        slot for, or an input image that cannot be decoded or is too large.
        """
        self.policies(request)
        if request.expand and self.expander is None:
            raise RequestError("prompt expansion is not configured (ARGUS_DARKROOM_OLLAMA_URL and _OLLAMA_MODEL)")
        prepared = self.prepare(request)
        return lambda emit=None: self._run(request, prepared, emit)

    def _run(self, request: GenerateRequest, prepared: Prepared, emit: Emit | None) -> GenerationRecord:
        def send(kind: str, data: dict) -> None:
            # A listener that fails (a client that hung up) must not cost the record: the run goes on.
            if emit:
                try:
                    emit(kind, data)
                except Exception:
                    logger.exception("engine.emit_failed", kind=kind)

        enforcing, advisory = self.policies(request)
        thinking = request.thinking or self.settings.thinking
        record_id = new_id()

        # Expansion comes first, so that pre-flight judges the final prompt: the expansion, not only what the
        # user typed. A failed expansion is a failed attempt, never a silent fall back to the plain prompt.
        expanded: str | None = None
        if request.expand:
            assert self.expander is not None  # begin() checked
            try:
                expanded = self.expander.expand(request.user_prompt)
            except ExpansionError as exc:
                record = self._save(
                    GenerationRecord(
                        id=record_id,
                        darkroom_version=__version__,
                        outcome="failed",
                        outcome_reason=f"expansion failed: {exc}"[:_REASON_MAX],
                        request=self._prepared_record(request, enforcing, None, prepared),
                        guard=GuardRecord(enforcing_policy=enforcing, advisory_policies=advisory, thinking=thinking),
                    )
                )
                send("complete", {"record": record.model_dump(mode="json")})
                return record

        pre = self.preflight(request, expanded)
        send("preflight", self.preflight_view(request, pre, expanded))
        if pre.refused:
            record = self.record_refusal(request, pre, expanded, prepared.init_ref)
            send("complete", {"record": record.model_dump(mode="json")})
            return record

        guard_record = GuardRecord(
            enforcing_policy=enforcing, advisory_policies=advisory, thinking=thinking, preflight=pre.cells
        )

        # Image-to-image: the input is judged before anything reaches ComfyUI, by the same checks as an output, and
        # goes on only if it passes. A refused input is not written anywhere: the record is all that is kept. An
        # inpainting mask is not scanned (it carries no imagery), and is not written either.
        if prepared.init is not None:
            guard_record.input, verdict = self.check_input(
                request,
                prepared.init,
                pre.framed,
                on_cell=lambda policy, context, cell: send(
                    "input_scan", {"policy": policy, "context": context, "cell": cell}
                ),
            )
            if verdict.outcome != "passed":
                record = self._save(
                    GenerationRecord(
                        id=record_id,
                        darkroom_version=__version__,
                        outcome="refused",
                        outcome_reason=(INPUT_REFUSAL + (verdict.reason or verdict.outcome))[:_REASON_MAX],
                        critical=verdict.critical,
                        request=self._prepared_record(request, enforcing, expanded, prepared),
                        guard=guard_record,
                    )
                )
                send("complete", {"record": record.model_dump(mode="json")})
                return record
        try:
            send("queued", {})
            rendered = self.render(
                request,
                expanded,
                lambda step, total: send("progress", {"step": step, "total": total}),
                prepared,
                guard_record.input,
            )
            image = describe(rendered.image)
        except (GenerationError, UnreadableImage) as exc:
            record = self._save(
                GenerationRecord(
                    id=record_id,
                    darkroom_version=__version__,
                    outcome="failed",
                    outcome_reason=_scrub(str(exc))[:_REASON_MAX],
                    request=self._prepared_record(request, enforcing, expanded, prepared),
                    guard=guard_record,
                )
            )
            send("complete", {"record": record.model_dump(mode="json")})
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
        decision, age, nudenet = self._judge(request, enforcing, scan)
        guard_record.scans = scan.scans
        guard_record.age = age
        guard_record.nudenet = nudenet
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
        send("complete", {"record": record.model_dump(mode="json")})
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
