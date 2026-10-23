"""Track A baseline job: solve Kuhn/Leduc with CFR+, freeze the strategy, measure it per style.

For each opponent style the values are exact (no sampling of cards): chips per hand, averaged
over both seats.
  nash        the frozen equilibrium baseline vs the opponent
  best        a best response to that exact opponent: the most any exploiter could win
  headroom    best − nash: what perfect identification + exploitation could add
Each style is a distribution of players (``styles.sample_params``); the reported numbers are the
mean over sampled players with a 95% CI (normal approximation over players), plus the style
center. The CI is over opponent draws only: card luck is integrated out exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pokerbot.baselines.cfr import (
    CFRPlus,
    Strategy,
    best_response,
    expected_value,
    exploitability,
    matchup,
)
from pokerbot.games.limit import GAMES, LimitGame
from pokerbot.games.styles import STYLES, center, sample_params, style_strategy
from pokerbot.track import Run


def strategy_hash(strategy: Strategy) -> str:
    blob = json.dumps({k: np.round(v, 12).tolist() for k, v in sorted(strategy.items())})
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def save_strategy(path: Path, game: LimitGame, strategy: Strategy, meta: dict) -> str:
    digest = strategy_hash(strategy)
    doc = {"game": game.name, "hash": digest, **meta,
           "strategy": {k: v.tolist() for k, v in sorted(strategy.items())}}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return digest


def load_strategy(path: Path) -> tuple[LimitGame, Strategy, dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    strategy = {k: np.array(v) for k, v in doc.pop("strategy").items()}
    if strategy_hash(strategy) != doc["hash"]:
        raise ValueError(f"{path}: strategy does not match its recorded hash")
    return GAMES[doc["game"]], strategy, doc


def evaluate(game: LimitGame, baseline: Strategy, params: np.ndarray) -> dict[str, float]:
    opp = style_strategy(game, params)
    nash = matchup(game, baseline, opp)
    best = matchup(game, best_response(game, opp), opp)
    return {"nash": nash, "best": best, "headroom": best - nash}


def run_track_a(
    game_name: str,
    iters: int,
    out_path: Path,
    players_per_style: int = 200,
    seed: int = 0,
    log_every: int = 100,
) -> dict:
    game = GAMES[game_name]
    config = dict(game=game_name, iters=iters, players_per_style=players_per_style,
                  styles={k: np.array(v).tolist() for k, v in STYLES.items()})  # fmt: skip
    with Run.start(f"baseline-{game_name}", kind="baseline", config=config,
                   seeds={"players": seed}) as run:  # fmt: skip
        solver = CFRPlus(game)
        while solver.t < iters:
            solver.iterate(min(log_every, iters - solver.t))
            avg = solver.average()
            run.log_metrics(
                {"train/exploitability": exploitability(game, avg),
                 "train/value_seat0": expected_value(game, avg, avg)}, step=solver.t,
            )  # fmt: skip
        nash = solver.average()
        expl = exploitability(game, nash)
        digest = save_strategy(
            out_path, game, nash,
            {"iterations": iters, "exploitability": expl, "run_id": run.dir.name},
        )  # fmt: skip
        run.save_artifact(out_path.name, out_path)

        rng = np.random.default_rng(seed)
        table: dict[str, dict] = {}
        for style in STYLES:
            rows = [
                evaluate(game, nash, sample_params(style, rng)) for _ in range(players_per_style)
            ]
            entry = {"center": evaluate(game, nash, center(style))}
            for key in ("nash", "best", "headroom"):
                v = np.array([r[key] for r in rows])
                half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
                mean = float(v.mean())
                entry[key] = {"mean": mean, "ci95": [mean - half, mean + half]}
                run.log_metric(f"val/{key}_chips_per_hand", float(v.mean()), style=style)
                run.log_metric(f"val/{key}_ci_half", float(half), style=style)
            table[style] = entry
        report = {"game": game_name, "baseline_hash": digest, "exploitability": expl,
                  "players_per_style": players_per_style, "styles": table}  # fmt: skip
        run.save_artifact("matchups.json", report)
        run.finish(exploitability=expl, baseline_hash=digest)
        report["run_id"] = run.dir.name
    return report
