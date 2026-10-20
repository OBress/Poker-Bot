"""Phase 4 exit measurement: each baseline's win rate against every style, with 95% CIs.

For every style, duplicate sessions are played with all opponents drawn from that style (spread
1: real player-to-player variation), plus "mixed" sessions whose opponents are drawn from the
population with the real style shares. All listed policies play the same sessions (same cards,
seats and opponents), so their differences are duplicate comparisons. bb/100 with a bootstrap CI
over sessions; all-in runouts replaced by their expectation.
"""

from __future__ import annotations

import json
from pathlib import Path

from pokerbot.eval.duplicate import run_match
from pokerbot.track import Run


def run_matchups(
    policies: list[str],
    fmt: str,
    bots_path: Path,
    sessions: int,
    hands: int,
    seed: int = 0,
    workers: int = 8,
) -> dict:
    doc = json.loads(Path(bots_path).read_text(encoding="utf-8"))
    names = [s["name"] for s in doc["formats"][fmt]["styles"]]
    config = dict(policies=policies, format=fmt, bots=str(bots_path), version=doc["version"],
                  sessions_per_style=sessions, hands_per_session=hands,
                  allin_adjusted=True)  # fmt: skip
    data = {"bots": Path(bots_path)}
    for p in policies:
        if p.startswith("blueprint:"):
            data["blueprint"] = Path(p.split(":", 1)[1])
    with Run.start(f"matchups-{fmt}", kind="baseline", config=config, seeds={"sessions": seed},
                   data=data) as run:  # fmt: skip
        report: dict = {"format": fmt, "bots_version": doc["version"], "styles": {}}
        targets = [(k, n) for k, n in enumerate(names)] + [(None, "mixed")]
        for k, name in targets:
            r = run_match(policies, bots_path, fmt, sessions, hands, seed * 7919 + (k or 0) + 1,
                          style=k, workers=workers)  # fmt: skip
            report["styles"][name] = r
            for p, v in r["heroes"].items():
                half = (v["ci95"][1] - v["ci95"][0]) / 2
                run.log_metrics({f"val/{p}/bb100": v["bb100"], f"val/{p}/bb100_ci_half": half},
                                style=name)  # fmt: skip
            print(name.ljust(26) + "  ".join(
                f"{p.split(':')[0]} {v['bb100']:+7.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                for p, v in r["heroes"].items()), flush=True)  # fmt: skip
        report["hands_per_policy"] = sum(r["hands"] for r in report["styles"].values())
        run.save_artifact("matchups.json", report)
        run.finish(hands_per_policy=report["hands_per_policy"])
        report["run_id"] = run.dir.name
    return report
