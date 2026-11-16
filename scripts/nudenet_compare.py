"""NudeNet detector comparison: what cutoff on a new detector reproduces the decisions made at 0.2 on the old one.

Run by hand, not in CI. 640m and 320n score the same image differently, so the 0.2 that darkroom applies to the
exposed score (``outcome.EXPOSED_FLOOR``: SFW omits at or above it; ``DEFAULT_NSFW_THRESHOLD``: NSFW blurs at or
above it; and the minor-exposed rule's exposure test) does not carry over from one detector to the other. This
scores a sample with both and sweeps the candidate's cutoff.

Two kinds of sample, which may be combined:

* ``--records ROOT``: darkroom's output root. Every stored image (``<id>.png`` whose record has ``image.path``). The
  baseline is the score its record kept, the score its decision was made on, unless ``--baseline`` re-scores it.
  Only released images are stored, so deleted and omitted ones are not in this sample.
* ``--labelled DIR``: ``DIR/blur/*`` and ``DIR/clear/*``, sorted by hand. ``--baseline`` scores them on the old
  detector. The report then also scores both detectors against the labels.

    python scripts/nudenet_compare.py --records /data --candidate http://retina:8080
    python scripts/nudenet_compare.py --labelled ~/nudenet-sample --baseline http://old:8010 --candidate http://new:8080

Nothing is copied or written but the report: file names (record ids) and scores, never image bytes. Images that a
server cannot score (after retries) are left out and counted. The sample stays wherever it is; it never belongs in
the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from argus_cortex.moderation import CheckFailed, NudeNetClient

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
BASELINE_CUTOFF = 0.2  # outcome.EXPOSED_FLOOR and DEFAULT_NSFW_THRESHOLD


@dataclass
class Item:
    name: str
    path: Path
    label: bool | None = None  # True: should be blurred (labelled sample); None: records
    baseline: float | None = None
    candidate: float | None = None


def from_records(root: Path) -> list[Item]:
    items = []
    for record_path in sorted(root.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        image = record.get("image") or {}
        png = root / f"{record_path.stem}.png"
        if not image.get("path") or not png.is_file():
            continue
        nudenet = (record.get("guard") or {}).get("nudenet") or {}
        items.append(Item(record_path.stem, png, baseline=nudenet.get("exposed_score")))
    return items


def from_labelled(root: Path) -> list[Item]:
    items = []
    for label, folder in ((True, "blur"), (False, "clear")):
        for path in sorted((root / folder).glob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                items.append(Item(f"{folder}/{path.name}", path, label=label))
    return items


def score(client: NudeNetClient, path: Path, retries: int, wait: float) -> float | None:
    data = path.read_bytes()
    for attempt in range(retries + 1):
        try:
            return client.check(data, filename=path.name).exposed_score
        except CheckFailed:
            if attempt < retries:
                time.sleep(wait)  # a server still loading answers 503
    return None


def score_all(items: list[Item], url: str, field: str, workers: int, retries: int, wait: float) -> None:
    client = NudeNetClient(url)
    info = getattr(client, "info", None)  # cortex >= 0.7
    if info is not None:
        try:
            print(f"{field}: {url} runs {info().model or 'an unnamed model'}", file=sys.stderr)
        except CheckFailed as exc:
            print(f"{field}: {url} did not say which model it runs ({exc})", file=sys.stderr)
    done = 0

    def one(item: Item) -> None:
        nonlocal done
        setattr(item, field, score(client, item.path, retries, wait))
        done += 1
        if done % 50 == 0:
            print(f"{field}: {done}/{len(items)}", file=sys.stderr)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, items))


def agreement(pairs: list[tuple[bool, float]], cutoff: float) -> tuple[int, int, int]:
    """(agree, new_clear, new_blur): the candidate at *cutoff* against a reference decision per image."""
    agree = new_clear = new_blur = 0
    for reference, candidate in pairs:
        flagged = candidate >= cutoff
        if flagged == reference:
            agree += 1
        elif reference:
            new_clear += 1  # the reference blurred it; the candidate would let it through
        else:
            new_blur += 1
    return agree, new_clear, new_blur


def sweep(pairs: list[tuple[bool, float]]) -> list[dict]:
    rows = []
    for step in range(1, 100):
        cutoff = step / 100
        agree, new_clear, new_blur = agreement(pairs, cutoff)
        rows.append({"cutoff": cutoff, "agree": agree, "new_clear": new_clear, "new_blur": new_blur})
    return rows


def best(rows: list[dict]) -> tuple[dict, dict | None]:
    """The cutoff that agrees most (ties go to the lower, which blurs more), and the highest that clears nothing the
    reference blurred (the conservative choice), or None when even 0.01 clears something."""
    most = max(rows, key=lambda r: (r["agree"], -r["cutoff"]))
    safe = [r for r in rows if r["new_clear"] == 0]
    return most, (max(safe, key=lambda r: r["cutoff"]) if safe else None)


def report(title: str, pairs: list[tuple[bool, float]], names: list[str], candidate_cutoff: float) -> dict:
    rows = sweep(pairs)
    most, safe = best(rows)
    agree, new_clear, new_blur = agreement(pairs, candidate_cutoff)
    flips = [
        {"name": n, "reference": ref, "candidate": round(c, 4)}
        for n, (ref, c) in zip(names, pairs, strict=True)
        if (c >= candidate_cutoff) != ref
    ]
    print(f"\n== {title}: {len(pairs)} images, {sum(r for r, _ in pairs)} flagged by the reference")
    print(
        f"   candidate at {candidate_cutoff:g}: agrees on {agree}, clears {new_clear} it flagged, flags {new_blur} new"
    )
    print(
        f"   best agreement: {most['cutoff']:g} (agrees on {most['agree']}, clears {most['new_clear']}, "
        f"flags {most['new_blur']} new)"
    )
    if safe is None:
        print("   no cutoff clears nothing the reference flagged")
    else:
        print(
            f"   highest cutoff that clears nothing the reference flagged: {safe['cutoff']:g} "
            f"(flags {safe['new_blur']} new)"
        )
    return {"title": title, "best": most, "conservative": safe, "at_cutoff": candidate_cutoff, "flips": flips,
            "sweep": rows}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, help="darkroom's output root: every stored image")
    parser.add_argument("--labelled", type=Path, help="a folder with blur/ and clear/ subfolders")
    parser.add_argument("--baseline", help="the old NudeNet server; required with --labelled")
    parser.add_argument("--candidate", required=True, help="the new NudeNet server, e.g. a 640m argus-retina")
    parser.add_argument("--cutoff", type=float, default=BASELINE_CUTOFF, help="the baseline's cutoff (0.2)")
    parser.add_argument(
        "--candidate-cutoff", type=float, help="report the candidate at this cutoff (default: --cutoff)"
    )
    parser.add_argument("--limit", type=int, help="at most this many images per sample")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--wait", type=float, default=3.0, help="seconds between retries")
    parser.add_argument("--json", type=Path, help="write the report (names and scores, no images) here")
    args = parser.parse_args()
    if not args.records and not args.labelled:
        parser.error("give --records, --labelled, or both")
    if args.labelled and not args.baseline:
        parser.error("--labelled needs --baseline: the images have no recorded score")

    items = (from_records(args.records) if args.records else [])[: args.limit] + (
        from_labelled(args.labelled) if args.labelled else []
    )[: args.limit]
    if not items:
        sys.exit("no images found")
    if args.baseline:
        score_all(items, args.baseline, "baseline", args.workers, args.retries, args.wait)
    score_all(items, args.candidate, "candidate", args.workers, args.retries, args.wait)

    scored = [i for i in items if i.baseline is not None and i.candidate is not None]
    print(f"\n{len(scored)} of {len(items)} images scored by both ({len(items) - len(scored)} left out)")
    cutoff = args.candidate_cutoff if args.candidate_cutoff is not None else args.cutoff
    reports = [
        report(
            f"candidate vs baseline at {args.cutoff:g}",
            [(i.baseline >= args.cutoff, i.candidate) for i in scored],
            [i.name for i in scored],
            cutoff,
        )
    ]
    labelled = [i for i in scored if i.label is not None]
    if labelled:
        names = [i.name for i in labelled]
        reports.append(report("candidate vs labels", [(i.label, i.candidate) for i in labelled], names, cutoff))
        reports.append(report("baseline vs labels", [(i.label, i.baseline) for i in labelled], names, args.cutoff))
    if args.json:
        args.json.write_text(
            json.dumps(
                {"items": [asdict(i) | {"path": None} for i in items], "reports": reports}, indent=2, default=str
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
