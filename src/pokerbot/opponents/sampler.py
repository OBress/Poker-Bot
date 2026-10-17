"""Draw semi-randomized opponents from calibrated styles.

Each seat gets its own parameters: x ~ N(x_mean, x_cov) for its style (x is the unbounded CMA
space, so sampled parameters always stay inside their bounds), a buy-in drawn from the style's real
stack-depth distribution, and its own RNG. Each style stores ``x_scale``, measured so that sampled
bots vary as much as real players of that style (``tune_scales``); ``spread`` multiplies it:
0 gives identical clones of the style center, 1 reproduces the real player-to-player spread.

The true style of every seat is returned alongside the bot. It is ground truth for the evaluator
only and must never be passed to a classifier or exploiter (AGENTS.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pokerbot.opponents.bot import StyleBot
from pokerbot.opponents.calibrate import sample_buyin, to_params


@dataclass(frozen=True)
class SeatDraw:
    bot: StyleBot
    style: int  # ground truth, evaluator only
    style_name: str


class StyleSampler:
    def __init__(self, bots_path: Path, fmt: str = "sh", spread: float = 1.0) -> None:
        doc = json.loads(Path(bots_path).read_text(encoding="utf-8"))
        self.version = doc["version"]
        self.styles = doc["formats"][fmt]["styles"]
        self.share = np.array([s["share"] for s in self.styles])
        self.share = self.share / self.share.sum()
        self.spread = spread
        self._chol = [
            np.linalg.cholesky(np.array(s["x_cov"]) + 1e-9 * np.eye(len(s["x_mean"])))
            for s in self.styles
        ]

    @property
    def names(self) -> list[str]:
        return [s["name"] for s in self.styles]

    def draw(self, rng: np.random.Generator, style: int | None = None) -> SeatDraw:
        k = int(rng.choice(len(self.styles), p=self.share)) if style is None else style
        s = self.styles[k]
        scale = self.spread * s.get("x_scale", 1.0)
        x = np.array(s["x_mean"]) + scale * self._chol[k] @ rng.standard_normal(len(s["x_mean"]))
        params = {**to_params(x), "buyin": sample_buyin(np.array(s["depth"]), rng)}
        return SeatDraw(StyleBot(params, int(rng.integers(2**63)), s["name"]), k, s["name"])

    def table(self, rng: np.random.Generator, seats: int) -> list[SeatDraw]:
        return [self.draw(rng) for _ in range(seats)]
