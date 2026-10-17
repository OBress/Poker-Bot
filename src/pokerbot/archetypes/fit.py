"""Style discovery job: per table format, choose K and write versioned profiles. Tracked run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from pokerbot.archetypes.cluster import choose, select_k
from pokerbot.archetypes.features import FEATURES, count_matrix, fit_transform, is_holdout
from pokerbot.archetypes.profiles import build_styles, write_profiles
from pokerbot.track import Run

FORMATS = ("sh", "hu", "fr")


def fit_styles(
    stats_path: Path,
    out_path: Path,
    version: str,
    min_hands: int = 500,
    ks: range = range(3, 15),
    boots: int = 5,
    seed: int = 0,
) -> Path:
    config = {
        "min_hands": min_hands,
        "ks": [ks.start, ks.stop],
        "boots": boots,
        "features": list(FEATURES),
        "holdout_fraction": 0.2,
        "version": version,
    }
    with Run.start(
        f"styles-{version}", kind="cluster", config=config, seeds={"gmm": seed},
        data={"player_stats": stats_path},
    ) as run:  # fmt: skip
        stats = pl.read_parquet(stats_path)
        formats = {}
        for fmt in FORMATS:
            sub = stats.filter(pl.col("pid").str.ends_with(f":{fmt}"))
            pids, hands, n, x = count_matrix(sub, FEATURES, min_hands)
            # Drop features that barely exist in this format (e.g. no early position heads-up).
            usable = np.median(n, axis=0) >= 10 if len(pids) else np.ones(len(FEATURES), bool)
            features = tuple(f for f, u in zip(FEATURES, usable, strict=True) if u)
            n, x = n[:, usable], x[:, usable]
            run.log_params(**{f"{fmt}_features": list(features)})
            fit_mask = np.array([not is_holdout(p) for p in pids], dtype=bool)
            run.log_metrics(
                {
                    f"{fmt}/players": len(pids),
                    f"{fmt}/players_fit": int(fit_mask.sum()),
                    f"{fmt}/players_holdout": int((~fit_mask).sum()),
                    f"{fmt}/median_hands": float(np.median(hands)) if len(hands) else 0,
                }
            )
            if fit_mask.sum() < 500:
                run.event("format_skipped", format=fmt, players=int(fit_mask.sum()))
                continue
            transform = fit_transform(n[fit_mask], x[fit_mask], features)
            z, _ = transform.transform(n[fit_mask], x[fit_mask])

            def log(k, r, fmt=fmt):
                run.log_metrics(
                    {
                        f"{fmt}/loglik": r.loglik,
                        f"{fmt}/bic": r.bic,
                        f"{fmt}/ari": r.ari,
                        f"{fmt}/min_share": r.min_share,
                    },
                    step=k,
                )
                print(
                    f"[{fmt}] K={k}: BIC={r.bic:,.0f} ARI={r.ari:.3f} min share={r.min_share:.3f}"
                )

            results = select_k(z, ks, seed, boots, log)
            best = choose(results)
            styles = build_styles(best.model, z, n[fit_mask], x[fit_mask], features)
            run.event("k_chosen", format=fmt, k=best.k, bic=best.bic, ari=best.ari)
            run.log_metric(f"{fmt}/chosen_k", best.k)
            formats[fmt] = {
                "players": int(fit_mask.sum()),
                "transform": transform,
                "styles": styles,
                "selection": [
                    {
                        "k": r.k,
                        "bic": r.bic,
                        "loglik": r.loglik,
                        "ari": r.ari,
                        "min_share": r.min_share,
                    }
                    for r in results
                ],
            }
            for s in styles:
                print(f"  [{fmt}] {s.index}: {s.name:16s} {s.share:5.1%}  {s.description}")
        write_profiles(out_path, version, run.dir.name, formats)
        run.save_artifact(out_path.name, out_path)
        run.finish(formats={f: len(v["styles"]) for f, v in formats.items()})
    return out_path
