"""Pre-flight evaluation: labelled prompts through the real framing and SingGuard server.

Run by hand against the real server, not in CI. It uses the same ``Guard`` and ``run_preflight`` as the
service, so it measures the framing as shipped, including both directions of the negative prompt:

* a negative of ``child, minor, underage`` must not read as asking for those (group ``negative-safety``);
* negating clothing must not be a way to nudity (group ``negative-undress``).

If the framing fails either, the framing changes; the negative prompt stays in the check.

    python scripts/preflight_eval.py --url http://localhost:9150
    python scripts/preflight_eval.py --url http://localhost:9150 --policy default --policy adult-granular --repeat 3

SingGuard is not deterministic, so ``--repeat`` runs each case several times and the table counts every run.
A case counts as flagged when the check was unsafe under the policy (illegal or not); an unchecked run is
reported separately and counts as neither.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from argus_cortex.moderation import THINKING_MODES
from argus_cortex.moderation import policies as policy_table

from argus_darkroom.guard import Guard
from argus_darkroom.prompt import frame

DEFAULT_PROBES = Path(__file__).resolve().parent.parent / "eval" / "preflight_probes.jsonl"


def load(path: Path) -> list[dict]:
    cases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        row = json.loads(line)
        if row.get("expect") not in ("pass", "flag"):
            raise SystemExit(f"{path}:{n}: expect must be 'pass' or 'flag'")
        cases.append(row)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="SingGuard server origin, e.g. http://localhost:9150")
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--policy", action="append", choices=policy_table.POLICIES, help="repeatable; default: all")
    parser.add_argument("--thinking", choices=THINKING_MODES, default="cascade")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path, help="write every run as JSONL")
    args = parser.parse_args()

    guard = Guard(args.url, args.thinking)
    cases = load(args.probes)
    policies = args.policy or list(policy_table.POLICIES)
    rows = []
    # (policy, group) -> [correct, wrong, unchecked]
    tally: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for policy in policies:
        for case in cases:
            text = frame(case["prompt"], case.get("negative", ""))
            for run in range(args.repeat):
                cell = guard.check_text(policy, text)
                got = {"safe": "pass", "flagged": "flag", "illegal": "flag"}.get(cell.state)
                slot = tally[(policy, case["group"])]
                slot[2 if got is None else 0 if got == case["expect"] else 1] += 1
                rows.append(
                    {
                        "policy": policy,
                        "id": case["id"],
                        "group": case["group"],
                        "run": run,
                        "expect": case["expect"],
                        "state": cell.state,
                        "rules": cell.result.rules if cell.result else [],
                        "error": cell.error,
                    }
                )
                print(f"{policy:15} {case['id']:26} expect={case['expect']:4} got={cell.state}", file=sys.stderr)

    print(f"\n{'policy':16}{'group':18}{'correct':>8}{'wrong':>7}{'unchecked':>11}")
    for (policy, group), (ok, bad, unk) in sorted(tally.items()):
        print(f"{policy:16}{group:18}{ok:>8}{bad:>7}{unk:>11}")
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return 1 if any(bad or unk for _, bad, unk in tally.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
