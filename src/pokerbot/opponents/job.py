"""Tracked calibration job: profiles/styles-*.json -> profiles/bots-*.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from pokerbot.archetypes.membership import depth_shares
from pokerbot.archetypes.profiles import load_profiles
from pokerbot.opponents.calibrate import (
    CALIBRATED,
    calibrate_format,
    logit,
    make_targets,
    simulate_rates,
    spread,
    to_params,
    write_bots,
)
from pokerbot.track import Run


def run_calibration(
    profiles_path: Path,
    stats_path: Path,
    out_path: Path,
    version: str,
    formats: tuple[str, ...] = ("sh",),
    rounds: int = 2,
    iters: int = 30,
    popsize: int = 14,
    hands: int = 1000,
    workers: int = 8,
    seed: int = 0,
    warm_start: Path | None = None,
) -> Path:
    config = dict(
        profiles=str(profiles_path), formats=list(formats), rounds=rounds, iters=iters,
        popsize=popsize, hands=hands, workers=workers, calibrated=list(CALIBRATED),
        warm_start=str(warm_start) if warm_start else None,
    )  # fmt: skip
    with Run.start(
        f"bots-{version}", kind="calibrate", config=config, seeds={"cma": seed},
        data={"profiles": profiles_path, "player_stats": stats_path},
    ) as run:  # fmt: skip
        profiles = load_profiles(profiles_path)
        stats = pl.read_parquet(stats_path, columns=["pid", "stat", "n", "x"])
        out: dict[str, dict] = {}
        for fmt in formats:
            prof = profiles["formats"][fmt]
            depth = depth_shares(prof, stats, fmt)
            targets = make_targets(prof, depth)

            def log(r, results, fmt=fmt, targets=targets):
                for res in results:
                    name = targets[res.index].name
                    for h in res.history:
                        step = r * iters + h["gen"]
                        key = f"{fmt}/{name}"
                        run.log_metrics(
                            {f"{key}/loss_best": h["best"], f"{key}/loss_mean": h["mean"]},
                            step=step, style=name, round=r,
                        )  # fmt: skip
                    run.log_metric(f"{fmt}/{name}/loss_final", res.loss, step=r, style=name)
                mean = float(np.mean([res.loss for res in results]))
                run.log_metric(f"{fmt}/loss_final_mean", mean, step=r)
                print(f"[{fmt}] round {r}: " + ", ".join(
                    f"{targets[res.index].name}={res.loss:.2f}" for res in results
                ), flush=True)  # fmt: skip

            warm = None
            if warm_start:
                warm_doc = json.loads(Path(warm_start).read_text(encoding="utf-8"))
                warm = warm_doc["formats"][fmt]["styles"]
                if [w["name"] for w in warm] != [t.name for t in targets]:
                    raise ValueError("warm-start styles don't match the profiles")
            results, pop = calibrate_format(
                targets, rounds, iters, popsize, hands, workers, seed, log, warm=warm
            )
            styles = []
            for res, t in zip(results, targets, strict=True):
                final = simulate_rates(
                    to_params(res.x), t, pop, seed + 12345, hands * 4, pop.noise[t.index]
                )
                z = (logit(final) - logit(t.rate)) / t.sigma
                styles.append(
                    {
                        "index": t.index,
                        "name": t.name,
                        "share": t.share,
                        "params": to_params(res.x),
                        "x_mean": res.x.tolist(),
                        "x_cov": spread(res, t).tolist(),
                        "x_scale": pop.scales[t.index],  # re-measured by `bots tune-spread`
                        "depth": t.depth.tolist(),
                        "loss": res.loss,
                        "check": {
                            f: {
                                "target": float(t.rate[j]),
                                "sim": float(final[j]),
                                "z": float(z[j]),
                            }
                            for j, f in enumerate(t.features)
                        },
                    }
                )
                run.log_metric(f"{fmt}/{t.name}/check_max_abs_z", float(np.abs(z).max()))
            out[fmt] = {"calibrated": list(CALIBRATED), "styles": styles}
        write_bots(out_path, version, run.dir.name, out)
        run.save_artifact(out_path.name, out_path)
    return out_path
