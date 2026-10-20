"""Track B baseline job: train the heads-up MCCFR blueprint as a tracked, resumable run.

Logs per round: information sets, iterations, iterations per second, and a convergence proxy,
the mean over information sets of the largest positive regret divided by the iterations so far
(it shrinks toward 0 as the average strategy approaches equilibrium in the abstraction).
The tables are checkpointed after every round; ``resume`` continues from the checkpoint.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from pokerbot.baselines import hunl, mccfr
from pokerbot.track import Run


def _checkpoint(path: Path, tables: mccfr.Tables, edges, done_rounds: int, iterations: int) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"tables": tables, "edges": edges, "rounds": done_rounds,
                     "iterations": iterations}, f, protocol=pickle.HIGHEST_PROTOCOL)  # fmt: skip
    tmp.replace(path)


def run_blueprint(
    rounds: int,
    iters_per_worker: int,
    workers: int,
    seed: int,
    out_path: Path,
    ckpt_path: Path,
    resume: bool = True,
) -> dict:
    state = None
    if resume and ckpt_path.exists():
        with ckpt_path.open("rb") as f:
            state = pickle.load(f)  # noqa: S301 - our own checkpoint
    config = dict(rounds=rounds, iters_per_worker=iters_per_worker, workers=workers,
                  sizes=hunl.SIZES, raise_sizes=hunl.RAISE_SIZES, raise_cap=hunl.RAISE_CAP,
                  buckets=hunl.BUCKETS, stack=hunl.STACK,
                  resumed_from_round=state["rounds"] if state else 0)  # fmt: skip
    with Run.start("blueprint-hu", kind="baseline", config=config, seeds={"mccfr": seed}) as run:
        if state:
            tables, edges = state["tables"], state["edges"]
            done, iterations = state["rounds"], state["iterations"]
        else:
            tables, edges, done, iterations = mccfr.Tables(), hunl.fit_edges(20_000, seed), 0, 0
        counter = {"it": iterations}

        def log(r, n_info, iters, seconds):
            counter["it"] += iters
            n = len(tables)
            proxy = float(np.maximum(tables.regret[:n], 0).max(axis=1).mean() / counter["it"])
            run.log_metrics(
                {"train/infosets": n_info, "train/iterations": counter["it"],
                 "sys/iters_per_sec": iters / seconds, "train/regret_proxy": proxy}, step=r,
            )  # fmt: skip
            _checkpoint(ckpt_path, tables, edges, r + 1, counter["it"])
            print(f"round {r}: {n_info:,} infosets, {counter['it']:,} iterations, "
                  f"{iters / seconds:,.0f} it/s, regret proxy {proxy:.4f}", flush=True)  # fmt: skip

        tables = mccfr.train(rounds - done, iters_per_worker, workers, seed, edges, log,
                             tables=tables, start_round=done)  # fmt: skip
        digest = mccfr.save(out_path, tables, edges,
                            {"iterations": counter["it"], "run_id": run.dir.name,
                             **{k: str(v) for k, v in config.items()}})  # fmt: skip
        info = {"path": str(out_path), "hash": digest, "iterations": counter["it"],
                "infosets": len(tables)}  # fmt: skip
        run.save_artifact("blueprint.json", info)
        run.finish(hash=digest, iterations=counter["it"], infosets=len(tables))
        return {"hash": digest, "iterations": counter["it"], "infosets": len(tables),
                "run_id": run.dir.name}  # fmt: skip
