"""Assemble the final prompt and frame it for SingGuard.

Every check judges the **final prompt**, the exact string ComfyUI receives, and
never only what the user typed, so the expansion step cannot route around the gate.
"""

from __future__ import annotations

from collections.abc import Iterable

NO_NEGATIVE = "(none)"


def assemble_prompt(user_prompt: str, expanded_prompt: str | None = None, trigger_words: Iterable[str] = ()) -> str:
    """The final prompt: Ollama's expansion of the user's prompt (the user's own text when expansion is off),
    then the LoRA trigger words, comma-separated. Blank pieces are dropped and trigger words not already
    present are appended once, in order.
    """
    body = (expanded_prompt if expanded_prompt is not None and expanded_prompt.strip() else user_prompt).strip()
    present = body.casefold()
    extra: list[str] = []
    for word in trigger_words:
        word = word.strip()
        if word and word.casefold() not in present and word not in extra:
            extra.append(word)
    return ", ".join(part for part in (body, *extra) if part)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def frame(final_prompt: str, negative_prompt: str = "") -> str:
    """The prompt and negative prompt as one labelled request.

    SingGuard reads a bare description as a request; here the text really is a request, and the
    framing says so. The negative prompt is labelled as what the image must not show: a negative
    of ``child, minor, underage`` is a common safety practice and must not read as asking for those.
    It is always part of the check, so negating clothing is not a way round it either.
    """
    # One line each: a line break in a prompt must not let it pose as the frame's own labels
    # ("...\nNegative prompt (what the image must not show): ...").
    prompt, negative = _one_line(final_prompt), _one_line(negative_prompt) or NO_NEGATIVE
    return f"Image generation request.\nPrompt: {prompt}\nNegative prompt (what the image must not show): {negative}"
