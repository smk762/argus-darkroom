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

A pose input is a photo or keypoints. A photo gets the input image's checks too; once it has passed, argus-puppet
reads its keypoints. Keypoints given in the request are numbers, with nothing to scan. Either way argus-puppet draws
the skeleton (:mod:`argus_darkroom.puppet`); the drawing is what ComfyUI gets, and the photo goes no
further. Both calls are made before the render's GPU lease is asked for, so no lease is held across them.

The render and the prompt expansion each hold a GPU lease (:mod:`argus_darkroom.gpu`) for as long as ComfyUI or
Ollama works on them, and go to the URL the lease names, if it names one. The scans hold none: SingGuard, NudeNet and
curator are tenants of their own.
"""

from __future__ import annotations

import contextlib
import re
import secrets
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import structlog
from argus_cortex.gpu import GpuCoordinator
from argus_cortex.gpu_wire import PriorityClass
from argus_cortex.moderation import NudeNetClient, rule_letter
from argus_cortex.moderation import policies as policy_table

from argus_darkroom import __version__, depth, gpu
from argus_darkroom.age import AgeEstimator, AgeStatus, default_estimator
from argus_darkroom.comfyui import ComfyUIClient, GenerationError, Progress
from argus_darkroom.config import Settings
from argus_darkroom.expansion import ExpansionError, OllamaExpander
from argus_darkroom.guard import Guard
from argus_darkroom.images import (
    ControlImage,
    InputImage,
    InputMask,
    UnreadableImage,
    describe,
    load_control,
    load_input,
    load_mask,
)
from argus_darkroom.modelfiles import ModelLibrary
from argus_darkroom.models import (
    AgeRecord,
    ControlImageRef,
    DepthRecord,
    GenerationRecord,
    GuardRecord,
    ImageRecord,
    InitImageRef,
    InputScanRecord,
    LoRARef,
    MaskRef,
    ModelRef,
    NudeNetBox,
    NudeNetRecord,
    Outcome,
    PoseAdherenceRecord,
    PoseEstimatorRef,
    PosePerson,
    PoseRef,
    RequestRecord,
)
from argus_darkroom.outcome import Decision, decide, minor_exposed
from argus_darkroom.preflight import PreflightResult, run_preflight
from argus_darkroom.prompt import assemble_prompt
from argus_darkroom.puppet import PoseUnavailable, PuppetClient
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

# (event kind, JSON-able payload): "preflight", "input_scan", "pose_scan", "pose", "queued", "progress", "scan",
# "pose_adherence", "complete"
Emit = Callable[[str, dict], None]
_REASON_MAX = 500
INPUT_REFUSAL = "input image: "
POSE_REFUSAL = "pose image: "  # the pose photo did not pass its scan
POSE_FAILURE = "pose: "  # the photo passed, and its pose could not be read or drawn
# A run of base64 this long in an error message is a piece of the input image (ComfyUI can echo a node's inputs). The
# client cuts ComfyUI's error bodies short, so a shorter run that the cut ends is a piece too.
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}|[A-Za-z0-9+/]{40,}={0,2}\Z")


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
    # Pose: the ControlNet (set for every pose request), and the photo the pose comes from, decoded and held in memory
    # only (None when the request gave keypoints, which stay in the request).
    pose: InputImage | None = None
    controlnet: ModelRef | None = None

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
class PoseControl:
    """A pose's skeleton: the people that are drawn (read from a checked photo by argus-puppet, or the request's own
    keypoints), what read them (None for given keypoints), and the drawing
    ComfyUI gets. Made only after the photo passed its scan (:meth:`Engine.pose_control`)."""

    people: tuple[PosePerson, ...]
    estimator: PoseEstimatorRef | None
    image: ControlImage


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
        gpu_coordinator: GpuCoordinator | None = None,
        puppet: PuppetClient | None = None,
    ) -> None:
        self.settings = settings
        self.guard = guard or Guard(settings.singguard_url, settings.thinking)
        self.store = store or RecordStore(settings.output_root)
        self.comfyui = comfyui or ComfyUIClient(settings.comfyui_url)
        # ComfyUI on the nodes the GPU broker sent work to, by base URL, so a purge can reach every one of them.
        self._comfyui_at: dict[str, ComfyUIClient] = {}
        self._comfyui_lock = threading.Lock()
        # The host-wide ARGUS_GPU_* environment: the broker when ARGUS_GPU_BROKER_URL is set, otherwise no gating.
        self.gpu = gpu_coordinator or gpu.default_coordinator()
        self.models = models or ModelLibrary(settings.models_root)
        # A template that could write before the verdict is refused here, at load, not when it runs.
        self.workflows = workflows or (load_workflows(settings.workflow) if settings.workflow else None)
        self._outputs_verified: set[tuple[str, str]] = set()  # (ComfyUI base URL, workflow sha256)
        if nudenet is None and settings.nudenet_url:
            nudenet = NudeNetClient(settings.nudenet_url)
        if age is None:
            age, self.age_status = default_estimator(settings.age_url, settings.age_fallback_url)
        else:
            self.age_status = AgeStatus(True)
        self.scanner = Scanner(self.guard, nudenet, age)
        if expander is None and settings.expansion_enabled:
            expander = OllamaExpander(settings.ollama_url, settings.ollama_model)
        self.expander = expander
        # argus-puppet reads a pose photo and draws its skeleton. None: a request with a pose input is refused.
        if puppet is None and settings.pose_available:
            puppet = PuppetClient(settings.puppet_url)
        self.puppet = puppet

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
                "reason": rule_letter(rules[0]).removesuffix(".") if rules else (pre.reason or "unchecked"),
                "rule": rules[0] if rules else None,
                "explanation": (cell.result.explanation if cell.result is not None else cell.error),
            }
        elif pre.cells[enforcing].state == "flagged":
            result = pre.cells[enforcing].result
            rule = result.rules[0] if result is not None and result.rules else None
            explanation = result.explanation if result is not None else None
            warnings.append({"policy": enforcing, "rule": rule, "explanation": explanation})
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
        pose: PoseRef | None = None,
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
            pose=pose,
        )

    def record_refusal(
        self,
        request: GenerateRequest,
        result: PreflightResult,
        expanded_prompt: str | None = None,
        init_image: InitImageRef | None = None,
        pose: PoseRef | None = None,
    ) -> GenerationRecord:
        """The record of an attempt that pre-flight refused. Nothing was generated, so there is no image. An input
        image or pose photo was not scanned either: the record names it by its hashes only."""
        enforcing, advisory = self.policies(request)
        record = GenerationRecord(
            id=new_id(),
            darkroom_version=__version__,
            outcome="refused",
            outcome_reason=result.reason,
            critical=False,
            request=self._request_record(request, enforcing, expanded_prompt, init_image=init_image, pose=pose),
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
        pose: InputImage | None = None
        controlnet: ModelRef | None = None
        if request.pose is not None:
            if init is not None:
                raise RequestError(
                    "a pose input cannot be combined with an init image: pose conditioning is text-to-image"
                )
            if self.puppet is None:
                raise RequestError(
                    "a pose input needs argus-puppet to read the pose (ARGUS_DARKROOM_PUPPET_URL or ARGUS_PUPPET_URL)"
                )
            name = request.pose.controlnet or self.settings.pose_controlnet
            if not name:
                raise RequestError(
                    "a pose input needs a ControlNet: name one in the request (pose.controlnet) or set "
                    "ARGUS_DARKROOM_POSE_CONTROLNET"
                )
            if request.pose.data_base64 is not None:
                try:
                    pose = load_input(request.pose.data_base64, "the pose image")
                except UnreadableImage as exc:
                    raise RequestError(str(exc)) from exc
            controlnet = self.models.ref("controlnet", name)
        mode: Mode = (
            "pose"
            if controlnet is not None
            else "txt2img"
            if init is None
            else "img2img"
            if mask is None
            else "inpaint"
        )
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
            pose=pose,
            controlnet=controlnet,
        )
        try:  # the template's slots must fit this request: better a 400 now than a failure later
            render(workflow, self._params(request, prepared, ""))
        except WorkflowError as exc:
            raise RequestError(str(exc)) from exc
        return prepared

    @staticmethod
    def _params(
        request: GenerateRequest, prepared: Prepared, positive: str, control: PoseControl | None = None
    ) -> RenderParams:
        """The values that fill the template. Without *control* a pose request's ``$pose_image`` is empty: enough to
        check the template's slots when the request is prepared, and never sent (:meth:`render` insists on one)."""
        assert request.sampling is not None  # prepare() checked
        posed = prepared.controlnet is not None and request.pose is not None
        return RenderParams(
            pose_image=(control.image.base64 if control is not None else "") if posed else None,
            controlnet=prepared.controlnet.name if posed and prepared.controlnet else None,
            pose_strength=request.pose.strength if posed and request.pose else None,
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
        priority: PriorityClass = gpu.DEFAULT_PRIORITY,
        pose_scan: InputScanRecord | None = None,
        control: PoseControl | None = None,
    ) -> Rendered:
        """Generate the image for *request* and hold the PNG in memory.

        Call it only after pre-flight let the request through, and after an input image passed its scan
        (:meth:`check_input`). A pose request also needs its photo's passed scan (*pose_scan*), if it has a photo, and
        the skeleton drawn
        from it (*control*, from :meth:`pose_control`). The final prompt is the one that was checked: it is built by :meth:`final_prompt`, the
        same function pre-flight used.

        The render holds a GPU lease on ``comfyui`` at *priority*, waiting for it first, and goes to the ComfyUI the
        lease names. A lease that is refused or not granted in time is a :class:`GenerationError` like any other.

        Raises :class:`RequestError` (nothing was sent to ComfyUI) or :class:`GenerationError`.
        """
        prepared = prepared or self.prepare(request)
        workflow = prepared.workflow
        enforcing, _ = self.policies(request)
        if prepared.init is not None and (input_scan is None or input_scan.outcome != "passed"):
            # Not reachable through generate(), which never renders an input that did not pass. A direct caller gets
            # the same rule: an input image reaches ComfyUI only after its own scan passed.
            raise RequestError("an input image is rendered only after its scan passed (Engine.check_input)")
        photo_unchecked = prepared.pose is not None and (pose_scan is None or pose_scan.outcome != "passed")
        if prepared.controlnet is not None and (photo_unchecked or control is None):
            # The same rule for a pose photo: nothing is drawn from one that did not pass, and nothing is rendered
            # without the drawing.
            raise RequestError(
                "a pose is rendered only after its photo's scan passed and its skeleton was drawn "
                "(Engine.check_input, Engine.pose_control)"
            )

        graph = render(workflow, self._params(request, prepared, self.final_prompt(request, expanded_prompt), control))
        with self._comfyui_lease(gpu.GENERATE, workflow.name, priority) as comfyui:
            self._verify_outputs(workflow, comfyui)
            result = comfyui.render(graph, workflow.save_node_id, on_progress)
            engine_version = comfyui.engine_version()
        return Rendered(
            image=result.image,
            request=self._prepared_record(
                request, enforcing, expanded_prompt, prepared, engine_version=engine_version, control=control
            ),
        )

    # -- GPU leases -------------------------------------------------------------------------------------

    def comfyui_clients(self) -> list[ComfyUIClient]:
        """The configured ComfyUI and every other one the GPU broker has sent work to."""
        with self._comfyui_lock:
            return [self.comfyui, *self._comfyui_at.values()]

    def _comfyui_for(self, base_url: str | None) -> ComfyUIClient:
        """The ComfyUI at the URL a lease named; the configured one when it named none."""
        if not base_url or base_url.strip().rstrip("/") == self.comfyui.base_url:
            return self.comfyui
        key = base_url.strip().rstrip("/")
        with self._comfyui_lock:
            client = self._comfyui_at.get(key)
            if client is None:
                try:
                    client = ComfyUIClient(key, timeout=self.comfyui.timeout, http_timeout=self.comfyui.http_timeout)
                except ValueError as exc:
                    raise GenerationError(f"the GPU broker named an unusable ComfyUI URL: {exc}") from exc
                self._comfyui_at[key] = client
        return client

    @contextlib.contextmanager
    def _comfyui_lease(self, footprint: gpu.Footprint, caller: str, priority: PriorityClass) -> Iterator[ComfyUIClient]:
        """Hold a GPU lease on ``comfyui`` for the body, and give it the ComfyUI to call.

        A lease that is refused or not granted (:data:`gpu.LEASE_ERRORS`) is a :class:`GenerationError`.
        """
        try:
            with self.gpu.lease(
                caller=caller,
                min_vram_mb=footprint.vram_mb,
                model_mb=footprint.model_mb,
                kind=footprint.kind,
                target=gpu.TARGET_COMFYUI,
                priority=priority,
            ) as lease:
                yield self._comfyui_for(lease.base_url)
        except gpu.LEASE_ERRORS as exc:
            raise GenerationError(gpu.lease_failure(exc)) from exc

    def _expand(self, prompt: str, priority: PriorityClass) -> str:
        """Expand *prompt* under a GPU lease on ``ollama:<model>``. Raises :class:`ExpansionError`."""
        assert self.expander is not None
        try:
            with self.gpu.lease(
                caller=self.expander.model,
                min_vram_mb=gpu.EXPAND.vram_mb,
                model_mb=gpu.EXPAND.model_mb,
                kind=gpu.EXPAND.kind,
                target=gpu.ollama_target(self.expander.model),
                priority=priority,
            ) as lease:
                return self.expander.expand(prompt, base_url=lease.base_url)
        except gpu.LEASE_ERRORS as exc:
            raise ExpansionError(gpu.lease_failure(exc)) from exc

    def _prepared_record(
        self,
        request: GenerateRequest,
        enforcing: str,
        expanded_prompt: str | None,
        prepared: Prepared,
        engine_version: str | None = None,
        control: PoseControl | None = None,
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
            pose=self._pose_ref(request, prepared, control),
        )

    @staticmethod
    def _pose_ref(request: GenerateRequest, prepared: Prepared, control: PoseControl | None = None) -> PoseRef | None:
        """The record's pose: a photo by its hashes or the request's keypoints, and the drawing once *control* exists."""
        if prepared.controlnet is None or request.pose is None:
            return None
        asked = request.pose
        drawn = {
            "controlnet": prepared.controlnet,
            "strength": asked.strength,
            "body": asked.body,
            "hands": asked.hands,
            "face": asked.face,
            "primary_only": asked.primary_only,
            "fit": asked.fit,
            "control": None
            if control is None
            else ControlImageRef(sha256=control.image.sha256, width=control.image.width, height=control.image.height),
        }
        if prepared.pose is None:  # keypoints: they are the request's own, known before anything is drawn
            assert asked.width is not None and asked.height is not None  # PoseInput checked
            people = asked.people()
            return PoseRef(
                source="keypoints",
                width=asked.width,
                height=asked.height,
                source_sha256=asked.source_sha256,
                people=list(people[:1] if asked.primary_only else people),
                **drawn,
            )
        photo = prepared.pose.record
        return PoseRef(
            source="photo",
            sha256=photo.sha256,
            phash=photo.phash,
            width=photo.width,
            height=photo.height,
            source_sha256=prepared.pose.source_sha256,
            people=list(control.people) if control is not None else [],
            estimator=control.estimator if control is not None else None,
            **drawn,
        )

    def _verify_outputs(self, workflow: Workflow, comfyui: ComfyUIClient) -> None:
        """Ask *comfyui* which of the workflow's nodes are output nodes, once per workflow and ComfyUI.

        The load-time check goes by node names. This one catches a node that writes but is not named like it. Each
        ComfyUI is asked on its own: another node may have other custom nodes installed under the same names.
        """
        key = (comfyui.base_url, workflow.sha256)
        if key in self._outputs_verified:
            return
        others = sorted(c for c, is_output in comfyui.output_classes(workflow.class_types).items() if is_output)
        others = [c for c in others if c != SAVE_NODE]
        if others:
            raise GenerationError(
                f"the workflow has output node(s) other than {SAVE_NODE}: {', '.join(others)}; "
                "nothing may be written before the verdict"
            )
        self._outputs_verified.add(key)

    # -- depth maps ------------------------------------------------------------------------------------

    def depth(
        self, record: GenerationRecord, image: bytes, priority: PriorityClass = gpu.DEFAULT_PRIORITY
    ) -> tuple[depth.DepthMap, DepthRecord]:
        """A depth map of *record*'s stored image, whose bytes are *image*. Writes nothing: the caller stores it.

        No pre-flight and no scan: the image was already judged and released, and no prompt is involved (see
        :mod:`argus_darkroom.depth`). The render holds a GPU lease like a generation's. Raises
        :class:`GenerationError`.
        """
        if record.image is None:
            raise GenerationError("the record has no image")
        try:
            with self._comfyui_lease(gpu.DEPTH, depth.WORKFLOW_NAME, priority) as comfyui:
                self._verify_outputs(depth.WORKFLOW, comfyui)
                result = comfyui.render(depth.graph(image), depth.WORKFLOW.save_node_id)
                engine_version = comfyui.engine_version()
            found = depth.to_depth_png(result.image, record.image.width, record.image.height)
        except GenerationError as exc:
            raise GenerationError(_scrub(str(exc))[:_REASON_MAX]) from exc
        except UnreadableImage as exc:
            raise GenerationError(str(exc)) from exc
        return found, DepthRecord(
            model=depth.MODEL,
            model_license=depth.MODEL_LICENSE,
            sha256=found.sha256,
            width=found.width,
            height=found.height,
            workflow=depth.WORKFLOW_NAME,
            engine_version=engine_version,
        )

    # -- the input image -------------------------------------------------------------------------------

    def _judge(
        self, request: GenerateRequest, enforcing: str, scan: Scan
    ) -> tuple[Decision, AgeRecord | None, NudeNetRecord]:
        """The outcome table over one image's scan: the same for an input image as for an output."""
        exposed = scan.nsfw.exposed_score if scan.nsfw is not None else None
        detections = scan.nsfw.detections if scan.nsfw is not None else []
        rule = self.settings.minor_exposed
        if rule == "overlap" and (scan.nudenet_info is None or scan.nudenet_info.detector is None):
            # Boxes from a detector that did not name itself cannot be placed: the upstream 320n image (no /health)
            # offsets y by its letterbox padding, which could move exposure off a young person's box. Unsure: frame.
            rule = "frame"
        age = scan.age.model_copy(deep=True) if scan.age is not None else None
        if age is not None:
            age.minor_exposed_rule = rule
            age.minor_exposed = minor_exposed(age, exposed, detections, rule)
        decision = decide(enforcing, scan.scans, exposed, age, request.nsfw_threshold, detections, rule)
        nudenet = NudeNetRecord(
            exposed_score=exposed,
            labels=scan.nsfw.labels if scan.nsfw is not None else {},
            detections=[NudeNetBox(label=d.label, score=d.score, bbox=tuple(d.bbox)) for d in detections],
            threshold=request.nsfw_threshold,
            blur=decision.blur,
            detector=scan.nudenet_info.detector if scan.nudenet_info is not None else None,
            detector_model=scan.nudenet_info.model if scan.nudenet_info is not None else None,
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

    # -- the pose ---------------------------------------------------------------------------------------

    def pose_control(
        self, request: GenerateRequest, prepared: Prepared, priority: PriorityClass = gpu.DEFAULT_PRIORITY
    ) -> PoseControl:
        """The skeleton a pose request renders from, drawn by argus-puppet at the generation's size.

        For a photo, call it only after the photo passed its scan (:meth:`check_input`): puppet first reads the
        people's keypoints from it, and this is the first time the photo leaves darkroom for anything but a guard.
        Given keypoints are drawn as they are. Only keypoints go into the drawing, and what comes back is checked to
        be a drawing (``images.load_control``), so the ControlNet never sees a picture.

        Raises :class:`PoseUnavailable` (puppet did not answer usefully, or found nobody) or :class:`UnreadableImage`
        (what it drew is not a skeleton of the generation's size).
        """
        assert prepared.controlnet is not None and request.pose is not None and request.sampling is not None
        if self.puppet is None:
            raise PoseUnavailable("argus-puppet is not configured")
        asked = request.pose
        estimator: PoseEstimatorRef | None = None
        if prepared.pose is not None:
            photo = prepared.pose.record
            frame = (photo.width, photo.height)
            read = self.puppet.read(prepared.pose.png, frame, priority)
            people, estimator = read.drawn(asked.primary_only), read.estimator
            if not people:
                raise PoseUnavailable("argus-puppet found nobody in the pose image")
        else:
            assert asked.width is not None and asked.height is not None  # PoseInput checked
            frame = (asked.width, asked.height)
            people = asked.people()[:1] if asked.primary_only else asked.people()
        canvas = (request.sampling.width, request.sampling.height)
        drawing = self.puppet.draw(
            people, frame, canvas, body=asked.body, hands=asked.hands, face=asked.face, fit=asked.fit
        )
        return PoseControl(people=people, estimator=estimator, image=load_control(drawing, canvas))

    def pose_adherence(
        self,
        request: GenerateRequest,
        prepared: Prepared,
        control: PoseControl,
        image: bytes,
        described: ImageRecord,
        priority: PriorityClass = gpu.DEFAULT_PRIORITY,
    ) -> PoseAdherenceRecord:
        """Whether a released *image* holds the pose it was rendered from, as argus-puppet judges it.

        Call it only for an image that was released: this sends the image to argus-puppet. It never raises: a check
        that cannot run is a record with ``adheres`` null and the reason in ``error``.
        """
        assert request.pose is not None
        if self.puppet is None:
            return PoseAdherenceRecord(error="argus-puppet is not configured")
        asked = request.pose
        frame = (
            (prepared.pose.record.width, prepared.pose.record.height)
            if prepared.pose is not None
            else (asked.width or 1, asked.height or 1)
        )
        try:
            return self.puppet.adherence(
                control.people,
                frame,
                image,
                (described.width, described.height),
                threshold=asked.verify_threshold,
                priority=priority,
            )
        except PoseUnavailable as exc:
            return PoseAdherenceRecord(error=_scrub(str(exc))[:_REASON_MAX], target_people=len(control.people))

    # -- the whole flow ---------------------------------------------------------------------------------

    def generate(
        self, request: GenerateRequest, emit: Emit | None = None, priority: PriorityClass = gpu.DEFAULT_PRIORITY
    ) -> GenerationRecord:
        """One guarded generation. Returns the record, whatever the outcome; never an image.

        1. pre-flight (text only): a refusal is recorded and nothing is generated;
        2. image-to-image only: the input image is scanned like an output, and anything but ``passed`` refuses;
           a pose photo likewise, and then argus-puppet reads and draws its skeleton (a failure there is a
           ``failed`` record);
        3. render: the image is held in memory only (a ComfyUI failure is a ``failed`` record);
        4. scan, decide;
        5. release (write, for passed and review) or destroy (write nothing).

        *priority* is the GPU lease class for the expansion and the render (the caller's ``X-Argus-Priority``).

        Raises :class:`RequestError` (HTTP 400) before anything happens when the request cannot be served.
        """

        return self.begin(request, priority)(emit)

    def begin(
        self, request: GenerateRequest, priority: PriorityClass = gpu.DEFAULT_PRIORITY
    ) -> Callable[[Emit | None], GenerationRecord]:
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
        return lambda emit=None: self._run(request, prepared, emit, priority)

    def _run(
        self, request: GenerateRequest, prepared: Prepared, emit: Emit | None, priority: PriorityClass
    ) -> GenerationRecord:
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
                expanded = self._expand(request.user_prompt, priority)
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
            record = self.record_refusal(request, pre, expanded, prepared.init_ref, self._pose_ref(request, prepared))
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

        # A pose photo is an input image too: the same checks, before it reaches argus-puppet or anything else, and
        # only `passed` goes on; then puppet reads the people's keypoints. A pose given as keypoints has no photo and
        # nothing to scan. Either way puppet draws the keypoints, and the drawing is all ComfyUI gets. Nothing is
        # written.
        control: PoseControl | None = None
        if prepared.controlnet is not None:
            failure: str | None = None
            outcome: Outcome = "failed"
            critical = False
            if prepared.pose is not None:
                guard_record.pose_input, verdict = self.check_input(
                    request,
                    prepared.pose,
                    pre.framed,
                    on_cell=lambda policy, context, cell: send(
                        "pose_scan", {"policy": policy, "context": context, "cell": cell}
                    ),
                )
                if verdict.outcome != "passed":
                    outcome, critical = "refused", verdict.critical
                    failure = POSE_REFUSAL + (verdict.reason or verdict.outcome)
            if failure is None:
                try:
                    control = self.pose_control(request, prepared, priority)
                except (PoseUnavailable, UnreadableImage) as exc:
                    failure = POSE_FAILURE + str(exc)
            if failure is not None or control is None:
                record = self._save(
                    GenerationRecord(
                        id=record_id,
                        darkroom_version=__version__,
                        outcome=outcome,
                        outcome_reason=_scrub(failure or POSE_FAILURE + "no skeleton")[:_REASON_MAX],
                        critical=critical,
                        request=self._prepared_record(request, enforcing, expanded, prepared),
                        guard=guard_record,
                    )
                )
                send("complete", {"record": record.model_dump(mode="json")})
                return record
            send(
                "pose",
                {
                    "people": len(control.people),
                    "control": {
                        "sha256": control.image.sha256,
                        "width": control.image.width,
                        "height": control.image.height,
                    },
                },
            )
        try:
            send("queued", {})
            rendered = self.render(
                request,
                expanded,
                lambda step, total: send("progress", {"step": step, "total": total}),
                prepared,
                guard_record.input,
                priority,
                guard_record.pose_input,
                control,
            )
            image = describe(rendered.image)
        except (GenerationError, UnreadableImage) as exc:
            record = self._save(
                GenerationRecord(
                    id=record_id,
                    darkroom_version=__version__,
                    outcome="failed",
                    outcome_reason=_scrub(str(exc))[:_REASON_MAX],
                    request=self._prepared_record(request, enforcing, expanded, prepared, control=control),
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
            if control is not None and request.pose is not None and request.pose.verify:
                # Only now, and only for a released image: a deleted or omitted one is sent nowhere. A measurement,
                # after the verdict, that cannot change it; a check that fails is noted and costs nothing else.
                record.pose_adherence = self.pose_adherence(request, prepared, control, rendered.image, image, priority)
                send("pose_adherence", record.pose_adherence.model_dump(mode="json"))
                self.store.save(record)
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
