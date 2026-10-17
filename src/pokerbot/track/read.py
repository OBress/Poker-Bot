"""Reading tracked runs back: listing, metrics as tables, plots."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pokerbot.track.run import DEFAULT_ROOT


def run_dirs(root: Path = DEFAULT_ROOT) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/run.json"))


def find_run(run_id: str, root: Path = DEFAULT_ROOT) -> Path:
    """Exact id, unique prefix/substring, or "latest"."""
    dirs = run_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"no runs under {root}")
    if run_id == "latest":
        return dirs[-1]
    matches = [d for d in dirs if d.name == run_id] or [d for d in dirs if run_id in d.name]
    if len(matches) != 1:
        raise LookupError(f"{len(matches)} runs match {run_id!r}")
    return matches[0]


def load_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _jsonl(path: Path) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()
    return pl.read_ndjson(path, infer_schema_length=None)


def load_metrics(run_dir: Path) -> pl.DataFrame:
    return _jsonl(run_dir / "metrics.jsonl")


def load_events(run_dir: Path) -> pl.DataFrame:
    return _jsonl(run_dir / "events.jsonl")


def list_runs(root: Path = DEFAULT_ROOT) -> pl.DataFrame:
    rows = []
    for d in run_dirs(root):
        m = load_meta(d)
        rows.append(
            {
                "id": m["id"],
                "kind": m["kind"],
                "status": m["status"],
                "seconds": m.get("seconds"),
                "commit": (m["git"]["commit"] or "")[:8] + ("+dirty" if m["git"]["dirty"] else ""),
            }
        )
    return pl.DataFrame(rows)


def latest_metrics(run_dir: Path) -> pl.DataFrame:
    """Last logged value, best value and count for every metric name."""
    m = load_metrics(run_dir)
    if m.is_empty():
        return m
    return (
        m.group_by("name", maintain_order=True)
        .agg(
            pl.col("value").last().alias("last"),
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
            pl.len().alias("points"),
            pl.col("step").max().alias("last_step"),
        )
        .sort("name")
    )


def plot_run(run_dir: Path, names: list[str] | None = None, max_panels: int = 16) -> Path:
    """One panel per metric (value vs step); saved to artifacts/metrics.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = load_metrics(run_dir)
    if m.is_empty():
        raise ValueError("run has no metrics")
    if "step" not in m.columns or m["step"].null_count() == m.height:
        m = m.with_row_index("step")
    all_names = sorted(m["name"].unique().to_list())
    chosen = [n for n in all_names if names is None or any(k in n for k in names)][:max_panels]
    cols = min(4, len(chosen))
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, name in zip(axes.flat, chosen, strict=False):
        d = m.filter(pl.col("name") == name).drop_nulls("step").sort("step")
        ax.plot(d["step"].to_list(), d["value"].to_list(), marker="." if d.height < 50 else None)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(chosen) :]:
        ax.axis("off")
    fig.suptitle(run_dir.name, fontsize=10)
    fig.tight_layout()
    out = run_dir / "artifacts" / "metrics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
