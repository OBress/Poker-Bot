"""Compute per-player stat counts over the whole real dataset, in parallel.

Work is split into chunks of rows; each worker replays its hands in pokerkit, counts stats per hand,
and sums them per player. Chunk results are written as long-format Parquet
(``pid, stat, n, x``) and then summed into one table.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from pokerbot.data.replay import replay_rows
from pokerbot.stats.hand_stats import hand_counts

CHUNK_ROWS = 100_000  # rows per task (also the resume unit)
BATCH_ROWS = 5_000
FLUSH_HANDS = 20_000


@dataclass
class ChunkResult:
    path: Path
    hands: int
    failed: int
    skipped: int


def standard_blinds(row: dict) -> bool:
    """Seat 0 posted the small blind and seat 1 a larger big blind. Otherwise (missing small
    blind, reversed posts; ~0.3% of HandHQ) positions are ambiguous and the hand is skipped."""
    b = row["blinds"]
    return len(b) >= 2 and 0 < b[0] < b[1]


def _process_chunk(src: Path, offset: int, length: int, out: Path) -> ChunkResult:
    """Count one chunk. Memory stays small: rows are materialized BATCH_ROWS at a time and
    running totals are flushed to the output every FLUSH_HANDS hands (duplicate (pid, stat) rows
    across flushes are summed in the final merge). The output appears atomically when complete,
    so an existing output file means the chunk is done."""
    df = pl.scan_parquet(src).slice(offset, length).collect()
    tmp = out.with_suffix(".tmp")
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    hands = failed = skipped = pending = 0
    with pq.ParquetWriter(tmp, _PART_SCHEMA, compression="zstd") as writer:
        for b in range(0, df.height, BATCH_ROWS):
            for row in df.slice(b, BATCH_ROWS).to_dicts():
                if not standard_blinds(row):
                    skipped += 1
                    continue
                hand = next(replay_rows([row]))
                if hand is None:
                    failed += 1
                    continue
                hands += 1
                pending += 1
                site = row["site"]
                for seat, counts in hand_counts(hand.events).items():
                    pid = f"{site}:{hand.players[seat]}"
                    for stat, (n, x) in counts.items():
                        t = totals[(pid, stat)]
                        t[0] += n
                        t[1] += x
                if pending >= FLUSH_HANDS:
                    _flush(writer, totals)
                    pending = 0
        _flush(writer, totals)
    tmp.replace(out)
    return ChunkResult(out, hands, failed, skipped)


_PART_SCHEMA = pa.schema(
    [("pid", pa.string()), ("stat", pa.string()), ("n", pa.int64()), ("x", pa.int64())]
)


def _flush(writer: pq.ParquetWriter, totals: dict[tuple[str, str], list[int]]) -> None:
    if not totals:
        return
    writer.write_table(
        pa.table(
            {
                "pid": [k[0] for k in totals],
                "stat": [k[1] for k in totals],
                "n": [v[0] for v in totals.values()],
                "x": [v[1] for v in totals.values()],
            },
            schema=_PART_SCHEMA,
        )
    )
    totals.clear()


def compute_player_stats(
    data_dir: Path, out_path: Path, workers: int = 6, limit_files: int | None = None
) -> dict[str, int]:
    """Write ``out_path`` (pid, stat, n, x) summed over all hands. Resumable: chunks whose part
    file already exists are skipped. ~1.5 GB RAM per worker at most."""
    parts_dir = out_path.parent / (out_path.stem + "_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    for stale in parts_dir.glob("*.tmp"):
        stale.unlink()
    files = sorted(data_dir.glob("*.parquet"))[:limit_files]
    jobs = []
    for f in files:
        rows = pq.ParquetFile(f).metadata.num_rows
        for offset in range(0, rows, CHUNK_ROWS):
            part = parts_dir / f"{f.stem}-{offset:09d}.parquet"
            if not part.exists():
                jobs.append((f, offset, min(CHUNK_ROWS, rows - offset), part))
    done_before = len(list(parts_dir.glob("*.parquet")))
    print(f"{done_before} chunks already done, {len(jobs)} to go", flush=True)

    hands = failed = skipped = 0
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_process_chunk, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            hands += r.hands
            failed += r.failed
            skipped += r.skipped
            if i % 5 == 0 or i == len(futures):
                print(
                    f"{i}/{len(futures)} chunks, {hands:,} hands, "
                    f"{failed:,} rejected, {skipped:,} skipped",
                    flush=True,
                )

    (
        pl.scan_parquet(parts_dir / "*.parquet")
        .group_by("pid", "stat")
        .agg(pl.col("n").sum(), pl.col("x").sum())
        .sink_parquet(out_path, engine="streaming")
    )
    summary = pl.scan_parquet(out_path).select(
        pl.len().alias("rows"), pl.col("pid").n_unique().alias("players")
    )
    return summary.collect().row(0, named=True)
