"""Real-player sessions for the classifier transfer metric (ARCHITECTURE §4.7, MATH.md §6.2).

A real session is one player's first ``max(CHECKPOINTS)`` hands of a table format in time order,
each replayed from HandHQ and rotated so the player sits in seat 0 (other seats are whoever sat
there in that hand). The label is the player's soft style membership q computed from *all* their
hands; predictions use only the first t. Players come from two groups:
  real_val  fit-set players (they helped define the styles; usable as an aggregate check)
  real_test held-out players (``is_holdout``): final evaluation only (AGENTS.md rule 2)
Pass 1 finds every hand of the chosen players (keys only); pass 2 replays the first hands.
"""

from __future__ import annotations

import lzma
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from pokerbot.archetypes.features import is_holdout
from pokerbot.archetypes.membership import assign, format_stats
from pokerbot.archetypes.profiles import load_profiles
from pokerbot.data.replay import replay
from pokerbot.engine.events import BlindPosted, HandEnd, HandStart, Muck, PlayerAction, Showdown
from pokerbot.evolve.harness import CHECKPOINTS, FeatureTracker, redact
from pokerbot.stats.aggregate import standard_blinds

CHUNK = 200_000
COLS = ["site", "players", "starting_stacks", "blinds", "antes", "big_blind", "min_bet",
        "actions", "date", "time"]  # fmt: skip


def rotate(events, target: int, n: int):
    """Relabel seats so ``target`` becomes seat 0 (relative positions unchanged)."""
    out = []
    for e in events:
        if isinstance(e, HandStart):
            st = e.stacks[target:] + e.stacks[:target]
            e = replace(e, button=(e.button - target) % n, stacks=st)
        elif isinstance(e, HandEnd):
            e = replace(e, payoffs=e.payoffs[target:] + e.payoffs[:target])
        elif isinstance(e, BlindPosted | PlayerAction | Showdown | Muck):
            e = replace(e, seat=(e.seat - target) % n)
        out.append(e)
    return tuple(out)


def choose_players(profiles_path: Path, stats_path: Path, fmt: str, per_group: int, seed: int):
    """{group: {pid: q}} for ``per_group`` random players with enough hands in each group."""
    prof = load_profiles(profiles_path)["formats"][fmt]
    stats = format_stats(stats_path, fmt, prof["transform"].features)
    pids, q = assign(prof, stats, fmt)
    hands = dict(stats.filter(pl.col("stat") == "hands").select("pid", "n").iter_rows())
    rng = np.random.default_rng(seed)
    out = {}
    for group, holdout in (("real_val", False), ("real_test", True)):
        idx = [i for i, p in enumerate(pids)
               if is_holdout(p) == holdout and hands.get(p, 0) >= 2 * max(CHECKPOINTS)]  # fmt: skip
        pick = rng.choice(idx, size=min(per_group, len(idx)), replace=False)
        out[group] = {pids[i]: q[i].tolist() for i in pick}
    return out


def _keys(args):
    """Pass 1: (pid, date, time, file, row) of every hand involving a target player."""
    src, offset, length, fmt, targets = args
    lo, hi = {"hu": (2, 2), "sh": (3, 6), "fr": (7, 10)}[fmt]
    df = (
        pl.scan_parquet(src)
        .slice(offset, length)
        .with_row_index("row", offset)
        .select("row", "site", "players", "date", "time")
        .filter(pl.col("players").list.len().is_between(lo, hi))
        .explode("players")
        .with_columns(pid=pl.concat_str(["site", "players"], separator=":") + f":{fmt}")
        .filter(pl.col("pid").is_in(list(targets)))
        .select("pid", "date", "time", "row")
        .collect()
    )
    return [(pid, d, t, str(src), row) for pid, d, t, row in df.iter_rows()]


def _replay(args):
    """Pass 2: replay the chosen rows of one file into rotated, redacted hands."""
    src, rows = args  # rows: list of (row, pid)
    wanted = {r for r, _ in rows}
    df = (
        pl.scan_parquet(src)
        .with_row_index("row")
        .filter(pl.col("row").is_in(list(wanted)))
        .select("row", *COLS)
        .collect()
    )
    by_row = {r["row"]: r for r in df.iter_rows(named=True)}
    out = []
    for r, pid in rows:
        row = by_row[r]
        if not standard_blinds(row):
            continue
        try:
            hand = replay(row)
        except Exception:  # noqa: BLE001 - malformed source hands are skipped, as in stats
            continue
        name = pid.split(":")[1]
        seat = hand.players.index(name)
        out.append((pid, row["date"], row["time"], rotate(redact(hand.events), seat,
                                                            len(hand.players))))  # fmt: skip
    return out


def build_real(
    profiles_path: Path,
    stats_path: Path,
    data_dir: Path,
    fmt: str,
    out_path: Path,
    per_group: int = 300,
    seed: int = 0,
    workers: int = 6,
    log=print,
) -> dict:
    groups = choose_players(profiles_path, stats_path, fmt, per_group, seed)
    targets = {p for g in groups.values() for p in g}
    files = sorted(Path(data_dir).glob("*.parquet"))
    jobs = []
    for f in files:
        n = pq.ParquetFile(f).metadata.num_rows
        jobs += [(f, o, min(CHUNK, n - o), fmt, targets) for o in range(0, n, CHUNK)]
    found: dict[str, list] = {p: [] for p in targets}
    with ProcessPoolExecutor(workers) as pool:
        for keys in pool.map(_keys, jobs):
            for pid, date, time, src, row in keys:
                found[pid].append((date or "", time or "", src, row))
    need = max(CHECKPOINTS) + 20  # spare hands for rows that fail to replay
    per_file: dict[str, list] = {}
    for pid, hs in found.items():
        for _, _, src, row in sorted(hs)[:need]:
            per_file.setdefault(src, []).append((row, pid))
    log(f"{len(targets)} players, {sum(len(v) for v in found.values()):,} hands found")
    hands: dict[str, list] = {p: [] for p in targets}
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_replay, list(per_file.items())):
            for pid, date, time, events in part:
                hands[pid].append((date or "", time or "", events))
    sessions = {}
    for group, members in groups.items():
        sessions[group] = []
        for pid, q in members.items():
            seq = [e for _, _, e in sorted(hands[pid], key=lambda h: h[:2])][: max(CHECKPOINTS)]
            if len(seq) < max(CHECKPOINTS):
                continue
            tracker = FeatureTracker()
            feats = {}
            for h, events in enumerate(seq, 1):
                tracker.add_hand(events)
                if h in CHECKPOINTS:
                    feats[h] = [tracker.snapshot(0)]
            sessions[group].append({"pid": pid, "q": q, "seats": [0], "hands": seq,
                                    "feats": feats})  # fmt: skip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(out_path, "wb", preset=6) as f:
        pickle.dump({"fmt": fmt, "sessions": sessions}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {g: len(v) for g, v in sessions.items()}


def load_real(path: Path, group: str, allow_test: bool = False) -> list[dict]:
    if group == "real_test" and not allow_test:
        raise PermissionError("held-out real players are for final evaluation only")
    with lzma.open(path, "rb") as f:
        return pickle.load(f)["sessions"][group]  # noqa: S301 - our own dataset
