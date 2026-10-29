"""Tracked duplicate benchmark: a hero vs a reference player against a pool of opponents."""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from pokerbot.eval.tables import TableFormat, run_duplicate, summarize, summarize_by_opponent
from pokerbot.opponents import fullhouse
from pokerbot.track import Run


def load_config(path: Path, **overrides) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def run_benchmark(cfg: dict, name: str) -> dict[str, dict]:
    """Run every table format in ``cfg["formats"]``; returns the summary per format."""
    bots_path = Path(cfg["bots"])
    data = {"bots": bots_path} if bots_path.exists() else {}
    config = cfg | {"fullhouse_version": fullhouse.VERSION}
    summaries: dict[str, dict] = {}
    with Run.start(name, kind="eval", config=config, seeds={"sessions": cfg["seed"]},
                   data=data) as run:  # fmt: skip
        for key, f in cfg["formats"].items():
            f = dict(f)
            sessions = cfg.get("sessions") or f.pop("sessions")
            f.pop("sessions", None)
            fmt = TableFormat(**f)
            t0 = time.perf_counter()

            def progress(done: int, total: int, key=key) -> None:
                if done % max(1, total // 10) == 0 or done == total:
                    print(f"[{key}] {done}/{total} sessions", flush=True)

            results = run_duplicate(
                cfg["hero"], cfg["reference"], cfg["opponents"], fmt, sessions,
                seed=cfg["seed"], seats=cfg["seats"], workers=cfg["workers"],
                bots_path=bots_path, rotate_hero=cfg.get("rotate_hero", True),
                progress=progress,
            )  # fmt: skip
            seconds = time.perf_counter() - t0
            s = summarize(results, fmt, seed=cfg["seed"])
            if cfg["seats"] == 2:
                s["by_opponent"] = summarize_by_opponent(results, fmt, seed=cfg["seed"])
            summaries[key] = s
            p = f"bench/{key}"
            run.log_metrics(
                {
                    f"{p}/delta_bb100": s["delta"].bb100,
                    f"{p}/delta_bb100_ci": s["delta"].ci,
                    f"{p}/hero_bb100": s["hero"].bb100,
                    f"{p}/hero_bb100_ci": s["hero"].ci,
                    f"{p}/ref_bb100": s["reference"].bb100,
                    f"{p}/ref_bb100_ci": s["reference"].ci,
                    f"{p}/sd_d_per100": s["sd_d_per100"],
                    f"{p}/hands": s["hands"],
                    f"{p}/sessions": s["sessions"],
                    f"{p}/hero_errors": s["hero_errors"],
                    f"{p}/ref_errors": s["ref_errors"],
                    # both copies of every hand are played
                    f"sys/{key}/hands_per_sec": 2 * s["hands"] / seconds,
                },
                mode=f["mode"],
            )
            for opp, o in s.get("by_opponent", {}).items():
                run.log_metrics(
                    {
                        f"{p}/{k}": v
                        for k, v in (
                            ("delta_bb100", o["delta"].bb100), ("delta_bb100_ci", o["delta"].ci),
                            ("hero_bb100", o["hero"].bb100), ("ref_bb100", o["reference"].bb100),
                            ("sessions", o["sessions"]),
                        )
                    },
                    mode=f["mode"], opponent=opp,
                )  # fmt: skip
            if fmt.mode == "match":
                run.log_metrics(
                    {f"{p}/{k}": s[k] for k in ("hero_chip_delta_mean", "ref_chip_delta_mean",
                                                 "hero_bust_rate", "ref_bust_rate")},
                    mode=f["mode"],
                )  # fmt: skip
            run.save_artifact(
                f"sessions-{key}.json",
                {"sessions": [
                    {
                        "session": r.seating.session,
                        "hero_seat": r.seating.hero_seat,
                        "opponents": list(r.seating.opponents),
                        "hands": r.hands,
                        "hero_chips": int(r.hero.sum()),
                        "ref_chips": int(r.ref.sum()),
                    }
                    for r in results
                ]},
            )  # fmt: skip
        run.finish(**_plain(summaries))
        print(f"run: {run.dir.name}")
    return summaries


def _plain(x):
    """Summaries with their Estimates as strings, for run.json."""
    if hasattr(x, "ci"):
        return str(x)
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return x
