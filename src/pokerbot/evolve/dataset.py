"""Precomputed classifier datasets: seeded sessions of sampled bots with their true styles.

Scoring a candidate replays stored public events instead of re-simulating hands, so evolution
can afford thousands of evaluations. A session is one table of ``SEATS[fmt]`` bots drawn from the
population (``StyleSampler``, spread 1) playing ``max(CHECKPOINTS)`` hands; every seat is scored.
Stored per session: the redacted events of every hand, the harness's feature snapshots at each
checkpoint, and the true style per seat (evaluator only).

Splits use disjoint seed ranges (ARCHITECTURE §5). ``test`` is built and read only by final
evaluations: ``load(..., allow_test=False)`` refuses it, and evolution never passes True.
"""

from __future__ import annotations

import lzma
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pokerbot.engine.table import play_hand
from pokerbot.eval.duplicate import make_policy
from pokerbot.evolve.harness import CHECKPOINTS, FeatureTracker, redact
from pokerbot.opponents.sampler import StyleSampler
from pokerbot.opponents.simulate import BIG_BLIND, SMALL_BLIND, stack_of
from pokerbot.opponents.validate import SEATS

SPLITS = {"train": 1, "val_a": 2, "val_b": 3, "test": 4}
EQ_SEAT_SHARE = 0.15  # with an equilibrium policy, share of seats it takes (its style label: K)
ROOT = Path("data/processed/clf")


def _session(args) -> dict:
    bots_path, fmt, seed, equilibrium = args
    rng = np.random.default_rng(seed)
    sampler = StyleSampler(Path(bots_path), fmt)
    draws = sampler.table(rng, SEATS[fmt])
    bots = [d.bot for d in draws]
    labels = [d.style for d in draws]
    if equilibrium:  # each seat is the baseline itself with probability EQ_SEAT_SHARE
        for s in range(len(bots)):
            if rng.random() < EQ_SEAT_SHARE:
                bots[s] = make_policy(equilibrium, int(rng.integers(2**31)))
                labels[s] = len(sampler.styles)
    stacks = [stack_of(b) for b in bots]
    tracker = FeatureTracker()
    n = len(bots)
    hands, feats = [], {}
    deck_seeds = rng.integers(0, 2**63 - 1, size=max(CHECKPOINTS))
    for h in range(max(CHECKPOINTS)):
        result = play_hand(bots, stacks, button=h % n, deck_seed=int(deck_seeds[h]),
                           small_blind=SMALL_BLIND, big_blind=BIG_BLIND, hand_id=h)  # fmt: skip
        events = redact(result.events)
        tracker.add_hand(events)
        hands.append(events)
        if h + 1 in CHECKPOINTS:
            feats[h + 1] = [tracker.snapshot(s) for s in range(n)]
    return {"seed": seed, "labels": labels, "seats": list(range(n)),
            "hands": hands, "feats": feats}  # fmt: skip


def path_for(fmt: str, split: str, root: Path = ROOT) -> Path:
    return Path(root) / f"{fmt}-{split}.pkl.xz"


def build(bots_path: Path, fmt: str, split: str, sessions: int, workers: int = 8,
          root: Path = ROOT, equilibrium: str | None = None) -> Path:  # fmt: skip
    """``equilibrium``: a policy spec (e.g. ``blueprint:<path>``) that takes seats as an extra
    style (label K); pair with ``StyleProfiles.load(..., equilibrium_path=...)``."""
    base = SPLITS[split] * 10_000_000
    jobs = [(str(bots_path), fmt, base + i, equilibrium) for i in range(sessions)]
    with ProcessPoolExecutor(workers) as pool:
        data = list(pool.map(_session, jobs, chunksize=4))
    out = path_for(fmt, split, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(out, "wb", preset=6) as f:
        pickle.dump({"fmt": fmt, "split": split, "bots": str(bots_path), "sessions": data}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)  # fmt: skip
    return out


def load(fmt: str, split: str, allow_test: bool = False, root: Path = ROOT) -> list[dict]:
    if split == "test" and not allow_test:
        raise PermissionError("test sessions are for final evaluation only (AGENTS.md rule 2)")
    with lzma.open(path_for(fmt, split, root), "rb") as f:
        return pickle.load(f)["sessions"]  # noqa: S301 - our own dataset
