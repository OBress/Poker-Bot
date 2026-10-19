"""Experiment tracking: every run leaves a complete, append-only record on disk.

Layout of ``experiments/<run_id>/``:

  run.json          name, kind, status, start/end time, config, params, seeds, git commit,
                    dirty flag, Python/package versions, data fingerprints, summary
  git.diff          uncommitted changes at start (so a dirty run is still reproducible)
  metrics.jsonl     one line per measurement: {"t", "step", "name", "value", ...tags}
  events.jsonl      one line per event: candidate created/gated/scored/promoted, errors, notes
  llm.jsonl         every LLM call: prompt, response, tokens, latency, cache hit, purpose
  candidates/<id>/  program source + meta.json (parents, diff, params, scores, gate results)
  artifacts/        anything else (configs, plots, profiles, reports)

Files are JSON Lines, flushed per line, so a crashed run keeps everything up to the crash.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("experiments")
_PACKAGES = ("numpy", "polars", "pyarrow", "pokerkit", "phevaluator", "openai")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _jsonable(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return x.as_posix()
    if hasattr(x, "item") and callable(x.item):  # numpy scalar
        return x.item()
    if hasattr(x, "tolist"):  # numpy array
        return x.tolist()
    return x


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",  # diffs contain non-ASCII; the Windows default codec fails on them
            errors="replace",
            check=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def fingerprint(path: Path) -> dict[str, Any]:
    """Cheap identity of a data file or directory: total size, file count, newest mtime, and a
    hash of (name, size, mtime) so a changed dataset is detectable without reading it."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    h = hashlib.sha256()
    size = 0
    newest = 0.0
    for f in files:
        st = f.stat()
        size += st.st_size
        newest = max(newest, st.st_mtime)
        h.update(
            f"{f.relative_to(path) if f != path else f.name}|{st.st_size}|{st.st_mtime_ns}".encode()
        )
    return {
        "path": path.as_posix(),
        "files": len(files),
        "bytes": size,
        "newest_mtime": newest,
        "sha256": h.hexdigest()[:16],
    }


class Run:
    """One tracked run. Use as a context manager so failures are recorded:

    with Run.start("classifier-evo", kind="evolve", config=cfg, data={"profiles": path}) as run:
        run.log_metric("val/logloss@25", 1.21, step=gen)
    """

    def __init__(self, directory: Path, meta: dict[str, Any]) -> None:
        self.dir = directory
        self.meta = meta
        self._files: dict[str, Any] = {}
        self._t0 = time.perf_counter()

    # ----------------------------------------------------------- lifecycle

    @classmethod
    def start(
        cls,
        name: str,
        kind: str,
        config: Mapping[str, Any] | None = None,
        seeds: Mapping[str, Any] | None = None,
        data: Mapping[str, Path] | None = None,
        root: Path = DEFAULT_ROOT,
        notes: str = "",
    ) -> Run:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{name}"
        directory = Path(root) / run_id
        suffix = 1
        while directory.exists():
            suffix += 1
            directory = Path(root) / f"{run_id}-{suffix}"
        for sub in ("candidates", "artifacts"):
            (directory / sub).mkdir(parents=True, exist_ok=True)

        diff = _git("diff", "HEAD")
        if diff:
            (directory / "git.diff").write_text(diff, encoding="utf-8")
        versions = {}
        for pkg in _PACKAGES:
            with contextlib.suppress(metadata.PackageNotFoundError):
                versions[pkg] = metadata.version(pkg)
        meta = {
            "id": directory.name,
            "name": name,
            "kind": kind,
            "status": "running",
            "started": _now(),
            "ended": None,
            "seconds": None,
            "notes": notes,
            "config": _jsonable(dict(config or {})),
            "params": {},
            "seeds": _jsonable(dict(seeds or {})),
            "data": {k: fingerprint(Path(v)) for k, v in (data or {}).items()},
            "git": {
                "commit": _git("rev-parse", "HEAD").strip(),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
                "dirty": bool(diff),
            },
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "cpus": os.cpu_count(),
                "argv": sys.argv,
                "packages": versions,
            },
            "summary": {},
        }
        run = cls(directory, meta)
        run._write_meta()
        run.event("run_started")
        return run

    def finish(self, status: str = "completed", **summary: Any) -> None:
        self.meta["summary"].update(_jsonable(summary))
        self.meta["status"] = status
        self.meta["ended"] = _now()
        self.meta["seconds"] = round(time.perf_counter() - self._t0, 3)
        self.event("run_finished", status=status)
        self._write_meta()
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            if self.meta["status"] == "running":
                self.finish("completed")
        else:
            self.event(
                "error",
                error=f"{exc_type.__name__}: {exc}",
                traceback="".join(traceback.format_exception(exc_type, exc, tb)),
            )
            self.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")

    # ----------------------------------------------------------- logging

    def log_params(self, **params: Any) -> None:
        """Hyperparameters and settings decided after start (merged into run.json)."""
        self.meta["params"].update(_jsonable(params))
        self._write_meta()

    def log_metric(self, name: str, value: float, step: int | None = None, **tags: Any) -> None:
        self._append(
            "metrics.jsonl",
            {"t": _now(), "step": step, "name": name, "value": _jsonable(value), **_jsonable(tags)},
        )

    def log_metrics(
        self, values: Mapping[str, float], step: int | None = None, **tags: Any
    ) -> None:
        for name, value in values.items():
            self.log_metric(name, value, step, **tags)

    def event(self, kind: str, **fields: Any) -> None:
        self._append("events.jsonl", {"t": _now(), "kind": kind, **_jsonable(fields)})

    def log_llm_call(
        self, purpose: str, messages: list[dict], completion: Any, **tags: Any
    ) -> None:
        """Full prompt and response for auditability (who wrote what code, from what prompt)."""
        self._append(
            "llm.jsonl",
            {
                "t": _now(),
                "purpose": purpose,
                "model": getattr(completion, "model", None),
                "messages": messages,
                "response": getattr(completion, "text", None),
                "prompt_tokens": getattr(completion, "prompt_tokens", None),
                "completion_tokens": getattr(completion, "completion_tokens", None),
                "seconds": getattr(completion, "seconds", None),
                "cached": getattr(completion, "cached", None),
                **_jsonable(tags),
            },
        )

    def save_candidate(self, candidate_id: str, source: str, **meta: Any) -> Path:
        """Store a program candidate's code and metadata (parents, diff, params, scores...)."""
        d = self.dir / "candidates" / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "program.py").write_text(source, encoding="utf-8")
        self.update_candidate(candidate_id, **meta)
        self.event("candidate_saved", candidate=candidate_id)
        return d

    def update_candidate(self, candidate_id: str, **meta: Any) -> None:
        path = self.dir / "candidates" / candidate_id / "meta.json"
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        current.update(_jsonable(meta))
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def save_artifact(self, name: str, content: str | bytes | Mapping | Path) -> Path:
        path = self.dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            path.write_bytes(content.read_bytes())
        elif isinstance(content, bytes):
            path.write_bytes(content)
        elif isinstance(content, Mapping):
            path.write_text(json.dumps(_jsonable(content), indent=2), encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")
        self.event("artifact_saved", name=name)
        return path

    # ----------------------------------------------------------- internals

    def _append(self, filename: str, record: dict) -> None:
        f = self._files.get(filename)
        if f is None:
            f = open(self.dir / filename, "a", encoding="utf-8")  # noqa: SIM115 - kept open
            self._files[filename] = f
        f.write(json.dumps(record) + "\n")
        f.flush()

    def _write_meta(self) -> None:
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        tmp.replace(self.dir / "run.json")
