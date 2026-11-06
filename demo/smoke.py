"""Drive a running demo through every marker and check the outcome darkroom reaches.

    uv run --no-sync python demo/run.py --fresh &
    uv run --no-sync python demo/smoke.py            # exits 1 if any outcome is not the expected one

The expected outcomes are the point: they are the outcome table, exercised through the real server. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

SAMPLING = {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 4,
    "cfg": 5.0,
    "width": 64,
    "height": 64,
    "clip_skip": 1,
}
SFW, NSFW = "default", "adult-granular"

# (prompt, enforcing policy, expected outcome, expected outcome_reason or None, blur or None)
CASES = [
    ("a quiet street", SFW, "passed", None, None),
    ("a quiet street [flag]", SFW, "review", None, None),
    ("a quiet street [flag-both]", SFW, "review", None, None),
    ("a quiet street [unchecked]", SFW, "deleted", "unchecked", None),
    ("a quiet street [exposed]", SFW, "omitted", "sfw-exposed", None),
    ("a quiet street [nonude]", SFW, "omitted", "sfw-unchecked", None),
    ("a quiet street [noage]", SFW, "passed", None, None),
    ("a quiet street [refuse]", SFW, "refused", None, None),
    ("a quiet street [preflight-unchecked]", SFW, "refused", None, None),
    ("a quiet street [fail]", SFW, "failed", None, None),
    ("a quiet street [illegal]", NSFW, "deleted", "adult-granular:A", None),
    ("a quiet street [refuse-illegal]", NSFW, "refused", "adult-granular:A", None),
    ("a quiet street [minor]", NSFW, "deleted", "minor-exposed", None),
    ("a quiet street [exposed]", NSFW, "passed", None, True),
    ("a quiet street [nonude]", NSFW, "passed", None, True),
    ("a quiet street [refuse]", NSFW, "passed", None, False),  # a warning, not a refusal
]


def post(base: str, path: str, body: dict) -> tuple[int, str]:
    request = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json", "Origin": base}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def generate(base: str, prompt: str, policy: str) -> tuple[dict, list[str]]:
    body = {
        "user_prompt": prompt,
        "negative_prompt": "",
        "checkpoint": "demo-checkpoint.safetensors",
        "sampling": SAMPLING,
        "enforcing_policy": policy,
        "seed": 7,
    }
    status, text = post(base, "/generate/stream", body)
    if status != 200:
        raise SystemExit(f"{status}: {text}")
    events, record = [], None
    for frame in text.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
        if "event" in lines:
            events.append(lines["event"])
            if lines["event"] == "complete":
                record = json.loads(lines["data"])["record"]
    if record is None:
        raise SystemExit(f"no complete event: {events}")
    return record, events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8106")
    base = parser.parse_args().url.rstrip("/")
    failures = 0
    for prompt, policy, outcome, reason, blur in CASES:
        record, events = generate(base, prompt, policy)
        nudenet = record["guard"].get("nudenet") or {}
        got = (record["outcome"], record["outcome_reason"], nudenet.get("blur"))
        ok = got[0] == outcome and (reason is None or got[1] == reason) and (blur is None or got[2] == blur)
        failures += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {policy:15} {prompt:42} -> {got[0]:8} {got[1] or '':14} {len(events)} events"
        )
        if not ok:
            print(f"     expected {outcome} {reason or ''} blur={blur}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
