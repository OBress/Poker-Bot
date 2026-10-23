"""External-sampling MCCFR for the abstract heads-up game (``hunl``), trained in parallel.

Each iteration deals both hands and the full board, then traverses the tree once per player: at
the traverser's decisions every action is explored and regrets updated; elsewhere one action is
sampled from the current strategy, whose probabilities are added to the average strategy there.
Regrets are floored at zero (regret matching+).

Parallel training runs in rounds. Every worker gets the current regrets, runs its own iterations
on its own seeds, and returns only the changes it made; the master adds them up (a batched,
slightly stale update) and weights each round's strategy contribution by the round number
(linear averaging).
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pokerbot.baselines import hunl
from pokerbot.engine.cards import hand_rank

MAX_ACTIONS = 6


class Tables:
    """Regret and strategy sums per information set, in growable arrays."""

    def __init__(self, capacity: int = 1 << 16) -> None:
        self.index: dict[str, int] = {}
        self.regret = np.zeros((capacity, MAX_ACTIONS), dtype=np.float32)
        self.strat = np.zeros((capacity, MAX_ACTIONS), dtype=np.float32)
        self.nact = np.zeros(capacity, dtype=np.int8)

    def __len__(self) -> int:
        return len(self.index)

    def row(self, key: str, n: int) -> int:
        i = self.index.get(key)
        if i is None:
            i = len(self.index)
            if i >= len(self.nact):
                self._grow()
            self.index[key] = i
            self.nact[i] = n
        return i

    def _grow(self) -> None:
        cap = 2 * len(self.nact)
        for name in ("regret", "strat"):
            old = getattr(self, name)
            new = np.zeros((cap, MAX_ACTIONS), dtype=np.float32)
            new[: len(old)] = old
            setattr(self, name, new)
        nact = np.zeros(cap, dtype=np.int8)
        nact[: len(self.nact)] = self.nact
        self.nact = nact

    def current(self, i: int) -> np.ndarray:
        n = self.nact[i]
        r = np.maximum(self.regret[i, :n], 0.0)
        s = r.sum()
        return r / s if s > 0 else np.full(n, 1.0 / n)

    def average(self) -> dict[str, np.ndarray]:
        out = {}
        for key, i in self.index.items():
            n = self.nact[i]
            s = self.strat[i, :n].astype(np.float64)
            out[key] = s / s.sum() if s.sum() > 0 else np.full(n, 1.0 / n)
        return out


class Trainer:
    def __init__(self, tables: Tables, edges, seed: int) -> None:
        self.t = tables
        self.edges = edges
        self.rng = np.random.default_rng(seed)
        self.touched: set[int] = set()

    def iterate(self) -> None:
        cards = self.rng.choice(52, size=9, replace=False).tolist()
        holes = ((cards[0], cards[1]), (cards[2], cards[3]))
        board = tuple(cards[4:])
        cache: dict[tuple[int, int], int] = {}

        def bucket(p: int, street: int) -> int:
            k = (p, street)
            if k not in cache:
                cache[k] = hunl.bucket(holes[p], board, street, self.edges)
            return cache[k]

        for p in (0, 1):
            self._walk(hunl.State(), p, holes, board, bucket)

    def _walk(self, s: hunl.State, p: int, holes, board, bucket) -> float:
        if s.terminal():
            u = hunl.showdown_payoff(s, holes, board, hand_rank)
            return u if p == 0 else -u
        q = s.to_act
        acts = hunl.legal(s)
        i = self.t.row(hunl.infoset(s, bucket(q, s.street)), len(acts))
        sigma = self.t.current(i)
        if q != p:
            self.t.strat[i, : len(acts)] += sigma
            self.touched.add(i)
            a = int(self.rng.choice(len(acts), p=sigma))
            return self._walk(hunl.apply(s, acts[a]), p, holes, board, bucket)
        vals = np.array([self._walk(hunl.apply(s, a), p, holes, board, bucket) for a in acts])
        v = float(sigma @ vals)
        self.t.regret[i, : len(acts)] += vals - v
        self.touched.add(i)
        return v


def _worker(args):
    snapshot, edges, seed, iters = args
    keys, regret, nact = pickle.loads(snapshot)
    t = Tables(max(1 << 16, 2 * len(keys)))
    for k, (key, n) in enumerate(zip(keys, nact, strict=True)):
        t.index[key] = k
        t.nact[k] = n
    t.regret[: len(keys)] = regret
    base = t.regret[: len(keys)].copy()
    tr = Trainer(t, edges, seed)
    for _ in range(iters):
        tr.iterate()
    rows = sorted(tr.touched)
    inv = {i: k for k, i in t.index.items()}
    delta = t.regret[rows].copy()
    old = np.array([r < len(keys) for r in rows])
    delta[old] -= base[np.array(rows)[old]]
    return [inv[r] for r in rows], t.nact[rows].copy(), delta, t.strat[rows].copy()


def train(
    rounds: int,
    iters_per_worker: int,
    workers: int,
    seed: int,
    edges,
    log=None,
    tables: Tables | None = None,
    start_round: int = 0,
) -> Tables:
    tables = tables if tables is not None else Tables()
    with ProcessPoolExecutor(workers) as pool:
        for r in range(start_round, start_round + rounds):
            t0 = time.perf_counter()
            n = len(tables)
            keys = list(tables.index)
            snap = pickle.dumps((keys, tables.regret[:n].copy(), tables.nact[:n].copy()))
            jobs = [(snap, edges, seed * 1_000_003 + r * 1000 + w, iters_per_worker)
                    for w in range(workers)]  # fmt: skip
            for keys_w, nact_w, d_regret, d_strat in pool.map(_worker, jobs):
                rows = np.array(
                    [tables.row(k, int(a)) for k, a in zip(keys_w, nact_w, strict=True)]
                )
                if len(rows):
                    tables.regret[rows] += d_regret
                    tables.strat[rows] += (r + 1) * d_strat  # linear averaging by round
            np.maximum(tables.regret, 0.0, out=tables.regret)  # regret matching+
            if log:
                log(r, len(tables), workers * iters_per_worker, time.perf_counter() - t0)
    return tables


def save(path: Path, tables: Tables, edges, meta: dict) -> str:
    """Average strategy as compressed arrays; returns a content hash."""
    import hashlib

    avg = tables.average()
    keys = np.array(list(avg), dtype=object)
    probs = np.zeros((len(avg), MAX_ACTIONS), dtype=np.float16)
    for k, v in enumerate(avg.values()):
        probs[k, : len(v)] = v
    digest = hashlib.sha256(
        "\n".join(keys.tolist()).encode() + probs.tobytes() + repr(edges).encode()
    ).hexdigest()[:16]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keys=keys.astype(str), probs=probs, edges=np.array(edges),
                        meta=np.array(repr({**meta, "hash": digest})))  # fmt: skip
    return digest
