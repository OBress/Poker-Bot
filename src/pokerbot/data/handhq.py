"""Convert the HandHQ subset of the PHH dataset into Parquet.

Source layout: ``<root>/<SITE>-2009-07-01_2009-07-23_<STAKE>NLH_OBFU/<k>/*.phhs``, each file a
TOML document of ~1,000 hands keyed ``[1]``, ``[2]``, ... Player names are obfuscated but
consistent, which is what per-player profiling needs.

Output: one Parquet file per source folder, one row per hand, with the raw PHH action strings
kept intact (parsing them into engine events happens downstream, so this step is lossless).
Amounts stay in the original currency units; ``big_blind`` is stored for normalization.
"""

from __future__ import annotations

import re
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FOLDER = re.compile(r"(?P<site>[A-Z]+)-\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_(?P<stake>\d+)NLH")

SCHEMA = pa.schema(
    [
        ("site", pa.string()),
        ("stake", pa.int32()),  # nominal stake, e.g. 200 for 200NL
        ("file", pa.string()),
        ("index", pa.int32()),  # hand's key inside the file
        ("hand", pa.string()),  # site hand id (numeric on most sites, text on Ongame)
        ("table", pa.string()),
        ("venue", pa.string()),
        ("date", pa.string()),
        ("time", pa.string()),
        ("seat_count", pa.int8()),
        ("seats", pa.list_(pa.int8())),
        ("players", pa.list_(pa.string())),
        ("starting_stacks", pa.list_(pa.float64())),
        ("blinds", pa.list_(pa.float64())),
        ("antes", pa.list_(pa.float64())),
        ("big_blind", pa.float64()),
        ("min_bet", pa.float64()),
        ("actions", pa.list_(pa.string())),
        ("winnings", pa.list_(pa.float64())),
    ]
)


@dataclass
class FolderResult:
    folder: str
    files: int
    hands: int
    bad_files: list[str]


def _floats(xs: list) -> list[float]:
    return [float(x) for x in xs]


def _row(site: str, stake: int, file: str, key: str, h: dict) -> dict:
    blinds = _floats(h.get("blinds_or_straddles", []))
    return {
        "site": site,
        "stake": stake,
        "file": file,
        "index": int(key),
        "hand": str(h["hand"]) if "hand" in h else None,
        "table": str(h.get("table", "")),
        "venue": str(h.get("venue", "")),
        "date": f"{h.get('year', 0):04d}-{h.get('month', 0):02d}-{h.get('day', 0):02d}",
        "time": str(h.get("time", "")),
        "seat_count": int(h.get("seat_count", len(h["players"]))),
        "seats": [int(s) for s in h.get("seats", [])],
        "players": [str(p) for p in h["players"]],
        "starting_stacks": _floats(h["starting_stacks"]),
        "blinds": blinds,
        "antes": _floats(h.get("antes", [])),
        "big_blind": blinds[1] if len(blinds) > 1 else float(h.get("min_bet", 0)),
        "min_bet": float(h.get("min_bet", 0)),
        "actions": [str(a) for a in h["actions"]],
        "winnings": _floats(h.get("winnings", [])),
    }


def convert_folder(folder: Path, out_dir: Path) -> FolderResult:
    m = FOLDER.match(folder.name)
    if not m:
        raise ValueError(f"unexpected folder name {folder.name}")
    site, stake = m["site"], int(m["stake"])
    files = sorted(folder.rglob("*.phhs"))
    bad: list[str] = []
    total = 0
    rows: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(out_dir / f"{site}-{stake}NL.parquet", SCHEMA, compression="zstd") as w:
        for n, f in enumerate(files, 1):
            rel = f.relative_to(folder).as_posix()
            try:
                doc = tomllib.loads(f.read_text(encoding="utf-8"))
                rows.extend(_row(site, stake, rel, k, h) for k, h in doc.items())
            except Exception:  # noqa: BLE001 - one corrupt file shouldn't stop the folder
                bad.append(rel)
            if len(rows) >= 100_000 or (n == len(files) and rows):
                w.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                total += len(rows)
                rows = []
    return FolderResult(folder.name, len(files), total, bad)


def convert_all(src: Path, out_dir: Path, workers: int = 8) -> list[FolderResult]:
    folders = sorted(p for p in src.iterdir() if p.is_dir() and FOLDER.match(p.name))
    results = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(convert_folder, f, out_dir): f for f in folders}
        for fut in as_completed(futures):
            r = fut.result()
            print(f"{r.folder}: {r.files} files, {r.hands:,} hands, {len(r.bad_files)} bad")
            results.append(r)
    return results
