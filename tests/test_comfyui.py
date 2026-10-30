"""The ComfyUI client against a fake: image over the websocket, history cleared, failures reported."""

from __future__ import annotations

import pytest
from fake_comfyui import IMAGE_HEADER, PREVIEW_HEADER, FakeComfyUI, event, grey_png

from argus_darkroom.comfyui import ComfyUIClient, GenerationError

GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {}},
    "9": {"class_type": "SaveImageWebsocket", "inputs": {"images": ["3", 0]}},
}


def client(fake: FakeComfyUI, **kw) -> ComfyUIClient:
    return ComfyUIClient(fake.url, ws_url=fake.ws_url, **kw)


def scripted(*messages):
    return lambda fake, pid, graph: [m(pid) if callable(m) else m for m in messages]


def test_the_image_comes_back_over_the_websocket_and_history_is_cleared(comfyui) -> None:
    result = client(comfyui).render(GRAPH, "9")
    assert result.image == comfyui.png
    assert comfyui.graphs == [GRAPH]
    assert comfyui.history == {}


def test_latent_previews_are_not_the_image(comfyui) -> None:
    assert client(comfyui).render(GRAPH, "9").image == comfyui.png != PREVIEW_HEADER + b"latent preview"


def test_progress_is_reported(comfyui) -> None:
    steps: list[tuple[int, int]] = []
    client(comfyui).render(GRAPH, "9", lambda step, of: steps.append((step, of)))
    assert steps == [(1, 2), (2, 2)]


def test_history_is_cleared_when_the_run_fails() -> None:
    script = scripted(
        lambda pid: event("execution_error", prompt_id=pid, node_type="KSampler", exception_message="out of memory\n")
    )
    with FakeComfyUI(script) as fake, pytest.raises(GenerationError, match="KSampler: out of memory"):
        client(fake).render(GRAPH, "9")
    assert fake.history == {}


def test_a_rejected_workflow_is_a_generation_error(comfyui) -> None:
    comfyui.reject_with = {
        "error": {"type": "prompt_outputs_failed_validation"},
        "node_errors": {"3": {"errors": ["bad"]}},
    }
    with pytest.raises(GenerationError, match="HTTP 400.*node_errors"):
        client(comfyui).render(GRAPH, "9")
    assert comfyui.history == {}


def test_a_run_that_never_finishes_times_out_interrupts_and_clears_history() -> None:
    with FakeComfyUI(scripted(lambda pid: event("executing", node="3", prompt_id=pid))) as fake:
        with pytest.raises(GenerationError, match="did not finish"):
            client(fake, timeout=0.5).render(GRAPH, "9")
        assert len(fake.interrupted) == 1
        assert fake.history == {}


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([lambda pid: event("executing", node=None, prompt_id=pid)], "without sending an image"),
        (
            [lambda pid: event("executing", node="9", prompt_id=pid), IMAGE_HEADER + grey_png(), IMAGE_HEADER + grey_png(),
             lambda pid: event("executing", node=None, prompt_id=pid)],
            "2 images",
        ),
        (
            [lambda pid: event("executing", node="9", prompt_id=pid), IMAGE_HEADER + b"JFIF not a png",
             lambda pid: event("executing", node=None, prompt_id=pid)],
            "not a PNG",
        ),
        (
            # a frame while some other node runs is not the save node's image
            [lambda pid: event("executing", node="3", prompt_id=pid), IMAGE_HEADER + grey_png(),
             lambda pid: event("executing", node=None, prompt_id=pid)],
            "without sending an image",
        ),
        ([lambda pid: event("execution_interrupted", prompt_id=pid)], "interrupted"),
    ],
)  # fmt: skip
def test_bad_runs_are_errors_and_still_clear_history(messages, match) -> None:
    with FakeComfyUI(scripted(*messages)) as fake:
        with pytest.raises(GenerationError, match=match):
            client(fake).render(GRAPH, "9")
        assert fake.history == {}


def test_another_clients_messages_are_ignored() -> None:
    script = scripted(
        event("executing", node=None, prompt_id="someone-else"),
        lambda pid: event("executing", node="9", prompt_id=pid),
        IMAGE_HEADER + grey_png(),
        lambda pid: event("executing", node=None, prompt_id=pid),
    )
    with FakeComfyUI(script) as fake:
        assert client(fake).render(GRAPH, "9").image == fake.png


def test_unreachable_comfyui_is_a_generation_error() -> None:
    with pytest.raises(GenerationError):
        ComfyUIClient("http://127.0.0.1:1", http_timeout=1).render(GRAPH, "9")


def test_engine_version(comfyui) -> None:
    assert client(comfyui).engine_version() == "0.3.test"
    assert ComfyUIClient("http://127.0.0.1:1", http_timeout=1).engine_version() == "unknown"


def test_output_classes(comfyui) -> None:
    assert client(comfyui).output_classes({"KSampler", "SaveImageWebsocket"}) == {
        "KSampler": False,
        "SaveImageWebsocket": True,
    }


def test_the_websocket_url_is_derived_from_the_base_url() -> None:
    assert ComfyUIClient("http://host:8188").ws_url == "ws://host:8188/ws"
    assert ComfyUIClient("https://host/comfy/").ws_url == "wss://host/comfy/ws"
    with pytest.raises(ValueError):
        ComfyUIClient("host:8188")
