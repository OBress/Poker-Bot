"""Tracked calibration job: profiles/styles-*.json -> profiles/bots-*.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pokerbot.archetypes.membership import depth_shares, format_stats, hand_quantiles
from pokerbot.archetypes.profiles import load_profiles
from pokerbot.opponents.bot import STRUCTURE, STRUCTURE_BOUNDS, STRUCTURE_PARAMS
from pokerbot.opponents.calibrate import (
    CALIBRATED,
    GROUPS,
    LAYOUT,
    LINE_GROUPS,
    SHOWDOWN_TOL,
    calibrate_format,
    feature_z,
    final_features,
    group_equity,
    make_targets,
    showdown_z,
    split,
    spread,
    to_params,
    write_bots,
)
from pokerbot.opponents.showdown import load_real
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
    hands: int = 4000,
    workers: int = 8,
    seed: int = 0,
    warm_start: Path | None = None,
    real_showdowns: Path | None = None,
) -> Path:
    """``real_showdowns``: real shown-hand tables (``bots showdown-real``); when given, the loss
    also matches each style's mean shown equity per line group (``calibrate.LINE_GROUPS``)."""
    config = dict(
        profiles=str(profiles_path), formats=list(formats), rounds=rounds, iters=iters,
        popsize=popsize, hands=hands, workers=workers, calibrated=list(CALIBRATED),
        structure=STRUCTURE, layouts={f: vars(LAYOUT[f]) for f in formats},
        warm_start=str(warm_start) if warm_start else None,
        real_showdowns=str(real_showdowns) if real_showdowns else None,
        line_groups=LINE_GROUPS, showdown_tol=SHOWDOWN_TOL,
    )  # fmt: skip
    data = {"profiles": profiles_path, "player_stats": stats_path}
    if real_showdowns:
        data["real_showdowns"] = real_showdowns
    with Run.start(
        f"bots-{version}", kind="calibrate", config=config, seeds={"cma": seed},
        data=data,
    ) as run:  # fmt: skip
        profiles = load_profiles(profiles_path)
        out: dict[str, dict] = {}
        for fmt in formats:
            prof = profiles["formats"][fmt]
            stats = format_stats(stats_path, fmt, prof["transform"].features)
            depth = depth_shares(prof, stats, fmt)
            hands_q = hand_quantiles(stats, fmt)
            del stats
            real = load_real(real_showdowns, fmt) if real_showdowns else None
            targets = make_targets(prof, depth, group_equity(real[0]) if real else None)

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

            warm, warm_names = None, CALIBRATED
            if warm_start:
                warm_doc = json.loads(Path(warm_start).read_text(encoding="utf-8"))
                if fmt in warm_doc["formats"]:
                    warm = warm_doc["formats"][fmt]["styles"]
                    default = STRUCTURE_PARAMS.get(warm_doc.get("structure", 2), CALIBRATED)
                    warm_names = tuple(warm_doc["formats"][fmt].get("calibrated", default))
                    if [w["name"] for w in warm] != [t.name for t in targets]:
                        raise ValueError("warm-start styles don't match the profiles")
            layout = LAYOUT[fmt]
            results, pop = calibrate_format(
                targets, rounds, iters, popsize, hands, workers, seed, log, warm=warm,
                warm_names=warm_names, layout=layout, hands_q=hands_q,
            )  # fmt: skip
            styles = []
            for res, t in zip(results, targets, strict=True):
                final = final_features(
                    res.x, t, pop, seed + 12345, hands, pop.noise[t.index], layout, hands_q, 6
                )
                z = feature_z(final, t)
                sdz = showdown_z(final, t)
                sums = split(final, t)[1]
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
                                "target_rate": float(t.rate[j]),
                                "target_feature": float(t.mean[j]),
                                "sim_feature": float(final[j]),
                                "z": float(z[j]),
                            }
                            for j, f in enumerate(t.features)
                        },
                        "showdown_check": {
                            g: {
                                "real_equity": float(t.showdown[i]),
                                "sim_equity": float(sums[0, i] / max(sums[2, i], 1)),
                                "sim_shown": float(sums[2, i]),
                                "z": float(sdz[i]),
                            }
                            for i, g in enumerate(GROUPS)
                        }
                        if t.showdown is not None
                        else None,
                    }
                )
                run.log_metric(f"{fmt}/{t.name}/check_max_abs_z", float(np.abs(z).max()))
            out[fmt] = {
                "calibrated": list(CALIBRATED),
                "bounds": {k: list(STRUCTURE_BOUNDS[STRUCTURE][k]) for k in CALIBRATED},
                "layout": vars(layout),
                "hands_cap": hands,
                "member_hands_percentiles": hands_q.tolist(),
                "styles": styles,
            }
        write_bots(out_path, version, run.dir.name, out)
        run.save_artifact(out_path.name, out_path)
    return out_path
