"""Prompt expansion by Ollama: a short prompt in, a fuller prompt out.

**Ollama is never a guard.** Its output is untrusted text that becomes part of the final prompt, and the final
prompt is what every check judges, so expansion cannot route around the gate. The system prompt below asks the
model to stay on the user's subject; that is a request, not a control. The record keeps the model and the sha256
of the system prompt, so an expansion can be traced to what produced it.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from http.client import HTTPException

from argus_darkroom.models import Expansion

SYSTEM_PROMPT = (
    "You expand a short idea into one detailed prompt for an image generator. "
    "Write a single line of comma-separated phrases covering the subject, setting, lighting, composition and style. "
    "Keep the user's subject and intent. Do not add people, nudity, or sexual, violent or illegal content that the "
    "user did not ask for. Do not explain, apologise, refuse or comment. Reply with the prompt only."
)
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()

MAX_CHARS = 1500  # a runaway model must not make the final prompt unboundedly long
DEFAULT_TIMEOUT = 120.0
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_QUOTES = "\"'`“”‘’"


class ExpansionError(RuntimeError):
    """Ollama could not expand the prompt."""


def clean(text: str) -> str:
    """One line, trimmed, without wrapping quotes, at most :data:`MAX_CHARS`.

    Newlines are folded to spaces: the final prompt is judged inside a labelled frame, and a line break must not
    let text pose as the frame's own labels.
    """
    one_line = " ".join(text.split()).strip(_QUOTES).strip()
    return one_line[:MAX_CHARS].rstrip(" ,")


class OllamaExpander:
    def __init__(self, base_url: str, model: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("expansion needs an Ollama URL and a model")
        self.url = base_url.strip().rstrip("/") + "/api/chat"
        self.model = model.strip()
        self.timeout = timeout

    @property
    def info(self) -> Expansion:
        return Expansion(model=self.model, system_prompt_sha256=SYSTEM_PROMPT_SHA256)

    def expand(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "options": {"num_predict": 400},
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with _OPENER.open(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            exc.close()
            raise ExpansionError(f"Ollama returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, HTTPException, TimeoutError, OSError) as exc:
            raise ExpansionError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise ExpansionError("Ollama returned a body that is not JSON") from exc
        message = payload.get("message") if isinstance(payload, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        text = clean(content) if isinstance(content, str) else ""
        if not text:
            raise ExpansionError("Ollama returned no prompt")
        return text
