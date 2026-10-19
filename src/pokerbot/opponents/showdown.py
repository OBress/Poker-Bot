"""Showdown-range validation: do bots show the hands real players of their style show?

Builds, per style, the table P(strength bucket | line class) of hands shown at showdown
(``stats.showdown``; MATH.md §4.5), from real hands and from simulated bots, and compares them.

Real side: every HandHQ hand that reached a showdown on a full board with cards shown is replayed;
each shown hand counts toward every style with the player's soft membership q (MATH.md §3.4) as
its weight. Only fit-set players with enough hands have a membership; held-out players
(``is_holdout``) are excluded so these tables can later feed the classifier (AGENTS.md rule 2).
Simulated side: every shown hand of a sampled bot counts toward its true style with weight 1.

Both sides are biased the same way: losers may muck (the engine follows real showdown order),
so tables describe *shown* hands, which is also all a classifier will ever see.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from pokerbot.archetypes.features import is_holdout
from pokerbot.archetypes.membership import assign
from pokerbot.archetypes.profiles import load_profiles
from pokerbot.data.replay import replay
from pokerbot.engine.events import Event
from pokerbot.stats.aggregate import standard_blinds, table_format
from pokerbot.stats.showdown import BUCKETS, LINES, STRENGTH_EDGES, shown_hands

CHUNK_ROWS = 50_000
MIN_SHOWN = 30  # effective shown hands per side before a line is compared


@dataclass
class RangeTable:
    """Weighted counts per (style, line, bucket) plus per (style, line) moments of equity."""

    k: int
    counts: np.ndarray = field(init=False)  # (K, L, H) sum of weights
    w2: np.ndarray = field(init=False)  # (K, L) sum of squared weights (effective n)
    eq_sum: np.ndarray = field(init=False)  # (K, L) sum of w * equity
    eq_sq: np.ndarray = field(init=False)  # (K, L) sum of w * equity^2

    def __post_init__(self) -> None:
        self.counts = np.zeros((self.k, len(LINES), BUCKETS))
        self.w2 = np.zeros((self.k, len(LINES)))
        self.eq_sum = np.zeros((self.k, len(LINES)))
        self.eq_sq = np.zeros((self.k, len(LINES)))

    def add(self, weights: np.ndarray, line: str, bucket: int, equity: float) -> None:
        j = LINES.index(line)
        self.counts[:, j, bucket] += weights
        self.w2[:, j] += weights**2
        self.eq_sum[:, j] += weights * equity
        self.eq_sq[:, j] += weights * equity**2

    def add_events(self, events: list[Event], style_of_seat: list[int]) -> None:
        """Simulated hand: each shown hand counts once for its seat's true style."""
        for sh in shown_hands(events):
            w = np.zeros(self.k)
            w[style_of_seat[sh.seat]] = 1.0
            self.add(w, sh.line, sh.bucket, sh.equity)

    def merge(self, other: RangeTable) -> None:
        self.counts += other.counts
        self.w2 += other.w2
        self.eq_sum += other.eq_sum
        self.eq_sq += other.eq_sq

    def to_dict(self) -> dict:
        return {
            "counts": self.counts.tolist(),
            "w2": self.w2.tolist(),
            "eq_sum": self.eq_sum.tolist(),
            "eq_sq": self.eq_sq.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> RangeTable:
        counts = np.array(d["counts"])
        t = RangeTable(counts.shape[0])
        t.counts, t.w2 = counts, np.array(d["w2"])
        t.eq_sum, t.eq_sq = np.array(d["eq_sum"]), np.array(d["eq_sq"])
        return t


# ---------------------------------------------------------------------------------- real data

_MEMBERS: dict[str, dict[str, np.ndarray]] = {}  # per worker process, set by the initializer


def _init_worker(members: dict[str, dict[str, np.ndarray]]) -> None:
    _MEMBERS.update(members)


def _shown_rows(src: Path, offset: int, length: int) -> pl.DataFrame:
    """Rows of a slice that show cards at a showdown with a complete board (3 board deals)."""
    acts = pl.col("actions")
    return (
        pl.scan_parquet(src)
        .slice(offset, length)
        .select(
            "site",
            "players",
            "starting_stacks",
            "blinds",
            "antes",
            "big_blind",
            "min_bet",
            "actions",
        )  # fmt: skip
        .filter(
            acts.list.eval(pl.element().str.contains(r"^p\d+ sm [2-9TJQKA]")).list.any()
            & (acts.list.eval(pl.element().str.starts_with("d db").cast(pl.Int32)).list.sum() == 3)
        )
        .collect()
    )


def _process_chunk(src: Path, offset: int, length: int) -> tuple[dict[str, dict], dict[str, int]]:
    tables = {fmt: RangeTable(len(next(iter(m.values())))) for fmt, m in _MEMBERS.items() if m}
    info = {"rows": 0, "replayed": 0, "rejected": 0, "shown": 0, "shown_members": 0}
    for row in _shown_rows(src, offset, length).iter_rows(named=True):
        info["rows"] += 1
        fmt = table_format(len(row["players"]))
        members = _MEMBERS.get(fmt)
        if not members or not standard_blinds(row):
            continue
        pids = [f"{row['site']}:{p}:{fmt}" for p in row["players"]]
        if not any(p in members for p in pids):
            continue
        try:
            hand = replay(row)
        except Exception:  # noqa: BLE001 - malformed source hands are skipped, as in stats
            info["rejected"] += 1
            continue
        info["replayed"] += 1
        for sh in shown_hands(hand.events):
            info["shown"] += 1
            q = members.get(pids[sh.seat])
            if q is not None:
                info["shown_members"] += 1
                tables[fmt].add(q.astype(float), sh.line, sh.bucket, sh.equity)
    return {fmt: t.to_dict() for fmt, t in tables.items()}, info


def memberships(
    profiles_path: Path, stats_path: Path, fmt: str
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Style names and pid -> soft membership q for the format's fit-set players."""
    prof = load_profiles(profiles_path)["formats"][fmt]
    keep = [*prof["transform"].features, "hands"]
    stats = (
        pl.scan_parquet(stats_path)
        .filter(pl.col("pid").str.ends_with(f":{fmt}") & pl.col("stat").is_in(keep))
        .select("pid", "stat", "n", "x")
        .collect()
    )
    pids, q = assign(prof, stats, fmt)
    members = {p: q[i].astype(np.float32) for i, p in enumerate(pids) if not is_holdout(p)}
    return [s["name"] for s in prof["styles"]], members


def real_tables(
    data_dir: Path,
    profiles_path: Path,
    stats_path: Path,
    formats: tuple[str, ...] = ("sh", "hu", "fr"),
    workers: int = 8,
    limit_files: int | None = None,
    log=None,
    parts_dir: Path | None = None,
) -> dict:
    """Real tables for every format. With ``parts_dir`` each chunk's result is saved there as it
    finishes and chunks already saved are skipped, so an interrupted job resumes (delete the
    folder after changing profiles or code)."""
    names, members = {}, {}
    with ProcessPoolExecutor(1) as one:  # the stats table is large: keep it out of this process
        for fmt in formats:
            names[fmt], members[fmt] = one.submit(
                memberships, profiles_path, stats_path, fmt
            ).result()
    files = sorted(Path(data_dir).glob("*.parquet"))[:limit_files]
    jobs, done = [], []
    for f in files:
        rows = pq.ParquetFile(f).metadata.num_rows
        for o in range(0, rows, CHUNK_ROWS):
            part = parts_dir / f"{f.stem}-{o:09d}.json" if parts_dir else None
            if part is not None and part.exists():
                done.append(part)
            else:
                jobs.append((f, o, min(CHUNK_ROWS, rows - o), part))
    tables = {fmt: RangeTable(len(names[fmt])) for fmt in formats}
    info = {"rows": 0, "replayed": 0, "rejected": 0, "shown": 0, "shown_members": 0}

    def add(part: dict, part_info: dict) -> None:
        for fmt, d in part.items():
            tables[fmt].merge(RangeTable.from_dict(d))
        for key, v in part_info.items():
            info[key] += v

    for path in done:
        saved = json.loads(path.read_text(encoding="utf-8"))
        add(saved["tables"], saved["info"])
    if parts_dir:
        parts_dir.mkdir(parents=True, exist_ok=True)
    # ~0.5 GB per worker. (max_tasks_per_child would cap it, but hangs the pool on Windows.)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(members,)) as pool:
        futures = {pool.submit(_process_chunk, *j[:3]): j[3] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            part, part_info = fut.result()
            add(part, part_info)
            if (path := futures[fut]) is not None:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"tables": part, "info": part_info}), encoding="utf-8")
                tmp.replace(path)
            if log and (i % 20 == 0 or i == len(futures)):
                log(len(done) + i, len(done) + len(futures), info)
    return {
        "lines": list(LINES),
        "strength_edges": list(STRENGTH_EDGES),
        "info": info,
        "formats": {
            fmt: {"styles": names[fmt], "members": len(members[fmt]), **tables[fmt].to_dict()}
            for fmt in formats
        },
    }


# ---------------------------------------------------------------------------------- compare


def _moments(t: RangeTable, k: int, j: int) -> tuple[float, float, float, np.ndarray]:
    """(effective n, mean equity, standard error, bucket distribution) of one cell."""
    w = t.counts[k, j].sum()
    if w <= 0:
        return 0.0, float("nan"), float("nan"), np.full(BUCKETS, np.nan)
    n_eff = w**2 / max(t.w2[k, j], 1e-12)
    mean = t.eq_sum[k, j] / w
    var = max(t.eq_sq[k, j] / w - mean**2, 0.0)
    return float(n_eff), float(mean), math.sqrt(var / n_eff), t.counts[k, j] / w


def _chi2_p(a: np.ndarray, b: np.ndarray) -> float:
    """Homogeneity test of two count vectors (buckets with no counts on either side dropped)."""
    from scipy.stats import chi2_contingency

    keep = (a + b) > 0
    if keep.sum() < 2:
        return 1.0
    return float(chi2_contingency(np.vstack([a[keep], b[keep]]))[1])


def compare(real: RangeTable, sim: RangeTable, names: list[str]) -> dict:
    """Per style and line: shown-strength distributions, mean equity z, total variation (TV)
    and a chi-square p-value (real counts scaled to their effective size). Also the share of
    each line among a style's shown hands, real vs simulated."""
    out = {}
    for k, name in enumerate(names):
        lines, tested, ok = {}, 0, 0
        real_share = real.counts[k].sum(axis=1)
        sim_share = sim.counts[k].sum(axis=1)
        real_share = real_share / max(real_share.sum(), 1e-12)
        sim_share = sim_share / max(sim_share.sum(), 1e-12)
        for j, line in enumerate(LINES):
            nr, mr, ser, dr = _moments(real, k, j)
            ns, ms, ses, ds = _moments(sim, k, j)
            row = {
                "real_n": round(nr, 1), "sim_n": round(ns, 1),
                "real_share": float(real_share[j]), "sim_share": float(sim_share[j]),
                "real_dist": dr.tolist(), "sim_dist": ds.tolist(),
                "real_equity": mr, "sim_equity": ms,
            }  # fmt: skip
            if nr >= MIN_SHOWN and ns >= MIN_SHOWN:
                z = (ms - mr) / math.sqrt(ser**2 + ses**2 + 1e-12)
                tv = 0.5 * float(np.abs(dr - ds).sum())
                p = _chi2_p(dr * nr, ds * ns)
                row |= {"equity_z": z, "tv": tv, "chi2_p": p}
                tested += 1
                ok += int(abs(z) < 2)
            lines[line] = row
        out[name] = {
            "lines": lines,
            "lines_tested": tested,
            "lines_equity_within_2se": ok,
            "line_share_tv": 0.5 * float(np.abs(real_share - sim_share).sum()),
        }
    return out


def load_real(path: Path, fmt: str) -> tuple[RangeTable, list[str]] | None:
    if not Path(path).exists():
        return None
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if fmt not in doc["formats"]:
        return None
    d = doc["formats"][fmt]
    return RangeTable.from_dict(d), d["styles"]
