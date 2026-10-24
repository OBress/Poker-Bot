"""Phase 7 final experiment: every track from one pre-registered config (``configs/final-v1.yaml``).

``run(config)`` plays fresh test-range sessions (indices from ``first_index``, never used before)
for the frozen programs, applies the pre-registered one-sided tests with Holm-Bonferroni over the
primary hypotheses, and returns one results dict; ``write_report`` turns it into
``docs/RESULTS.md`` and plots. Exploiter sessions run in spawned worker processes; programs are
loaded through the sandbox as in evolution.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import ttest_1samp, ttest_rel

from pokerbot.evolve import program

GAMES = ("leduc", "hu")
UNIT = {"leduc": "chips/hand", "hu": "bb/100"}
SCALE = {"leduc": 1.0, "hu": 100.0}  # per-hand d -> reported unit
HYPOTHESES = {
    "A1": "Leduc: evolved exploiter beats the Nash baseline against style players",
    "A2": "Leduc: evolved exploiter loses less than the margin to the Nash baseline itself",
    "A3": "Leduc: evolved exploiter scores higher than the seed (paired fitness)",
    "B1": "Heads-up: evolved exploiter beats the blueprint against style bots",
    "B2": "Heads-up: evolved exploiter loses less than the margin to the blueprint itself",
    "B3": "Heads-up: evolved exploiter scores higher than the seed (paired fitness)",
    "C1": "6-max: evolved classifier has lower weighted log loss than seed A (paired)",
}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["path"] = str(path)
    return cfg


def _program(spec: dict) -> tuple[str, dict | None]:
    source = Path(spec["path"]).read_text(encoding="utf-8")
    params = None
    if spec.get("params"):
        raw = json.loads(Path(spec["params"]).read_text(encoding="utf-8"))
        params = raw.get("params", raw)
    return source, params


def _ci(v: np.ndarray, resamples: int = 4000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(resamples, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5]).tolist()


def _summary(v: np.ndarray, boot: dict) -> dict:
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "ci95": _ci(v, **boot), "sessions": int(len(v))}


# ------------------------------------------------------------------------- exploiter sessions


def _sessions(
    game: str, source: str, params: dict | None, idx: list[int], mode: str, split: str = "test"
) -> dict:
    """Worker: play sessions ``idx`` of ``split``; compact per-session results."""
    if game == "leduc":
        from pokerbot.evolve import leduc as g
        from pokerbot.evolve.leduc_task import TEMPLATE

        ns = program.load(source, TEMPLATE)
        p = params if params is not None else program.defaults(ns)
        table = g.style_table()
        flags = {
            "oracle": mode == "oracle",
            "prior_only": mode == "prior",
            "argmax": mode == "argmax",
        }

        def play(spec):
            return g.run_session(lambda: ns["Exploiter"](table, dict(p)), spec, **flags)
    else:
        from pokerbot.evolve import hu as g
        from pokerbot.evolve.hu_task import TEMPLATE

        ns = program.load(source, TEMPLATE)
        p = params if params is not None else program.defaults(ns)
        prof = g.profiles()

        def play(spec):
            return g.run_session(lambda: ns["Exploiter"](prof, dict(p)), spec, beliefs_mode=mode)

    out = []
    for i in idx:
        r = play(g.SessionSpec(split, i))
        lab = r["labels"][0]
        out.append(
            {
                "index": i,
                "kind": r["kind"],
                "label": -1 if lab is None else int(lab),
                "d": np.asarray(r["d"], float),
                "kl": r["kl"],
                "score": g.session_score(r),
                "invalid": r["invalid"],
            }
        )
    return {"results": out}


def _play(pool, game, source, params, indices, mode, workers, split) -> list[dict]:
    parts = [indices[i::workers] for i in range(workers) if indices[i::workers]]
    jobs = [pool.apply_async(_sessions, (game, source, params, p, mode, split)) for p in parts]
    res = [r for j in jobs for r in j.get(7200)["results"]]
    return sorted(res, key=lambda r: r["index"])


def _nash_indices(game: str, first: int, n: int, split: str) -> list[int]:
    from pokerbot.evolve import hu, leduc

    g = leduc if game == "leduc" else hu
    out, i = [], first
    while len(out) < n:
        if g.SessionSpec(split, i).kind == "nash":
            out.append(i)
        i += 1
    return out


def exploiter_track(game: str, cfg: dict, pool, workers: int, boot: dict, log=print) -> dict:
    from pokerbot.evolve import hu, leduc

    t = cfg[game]
    scale = SCALE[game]
    names = list(leduc.STYLES) if game == "leduc" else list(hu.profiles().names[:-1])
    first, split = int(cfg["first_index"]), cfg.get("split", "test")
    main_idx = list(range(first, first + int(t["sessions"])))
    # safety sessions come after the main block, so the two never overlap
    safety_idx = _nash_indices(game, first + int(t["sessions"]), int(t["safety_sessions"]), split)
    out: dict = {
        "unit": UNIT[game],
        "hands_per_session": (leduc if game == "leduc" else hu).HANDS,
        "sessions": len(main_idx),
        "programs": {},
        "curves": {},
    }
    scores: dict[str, np.ndarray] = {}
    style_d: dict[tuple[str, str], np.ndarray] = {}
    for pname, spec in t["programs"].items():
        source, params = _program(spec)
        for mode in t["beliefs"]:
            if pname != "evolved" and mode != "classifier":
                continue  # ablations are of the evolved program
            log(f"{game}: {pname} / {mode} on {len(main_idx)} sessions")
            res = _play(pool, game, source, params, main_idx, mode, workers, split)
            e: dict = {
                "invalid": int(sum(r["invalid"] for r in res)),
                "kl": float(np.mean([r["kl"] for r in res])),
                "fitness": float(np.mean([r["score"] for r in res])),
            }
            for kind in ("style", "nash", "switch"):
                v = np.array([scale * r["d"].mean() for r in res if r["kind"] == kind])
                e[kind] = _summary(v, boot)
            for k, sname in enumerate(names):
                v = np.array(
                    [scale * r["d"].mean() for r in res if r["kind"] == "style" and r["label"] == k]
                )
                if len(v):
                    e[f"vs {sname}"] = _summary(v, boot)
            curve = np.mean([r["d"] for r in res if r["kind"] == "style"], axis=0) * scale
            out["curves"][f"{pname}/{mode}"] = curve.tolist()
            out["programs"].setdefault(pname, {})[mode] = e
            if mode == "classifier":
                scores[pname] = np.array([r["score"] for r in res])
            style_d[(pname, mode)] = np.array(
                [scale * r["d"].mean() for r in res if r["kind"] == "style"]
            )
    # primary tests
    src, params = _program(t["programs"]["evolved"])
    log(f"{game}: safety on {len(safety_idx)} baseline-only sessions")
    safe = np.array(
        [
            scale * r["d"].mean()
            for r in _play(pool, game, src, params, safety_idx, "classifier", workers, split)
        ]
    )
    margin = float(t["margin"])
    edge = style_d[("evolved", "classifier")]
    paired = scores["evolved"] - scores["seed"]
    out["safety"] = {
        **_summary(safe, boot),
        "margin": margin,
        "first_index": safety_idx[0],
        "last_index": safety_idx[-1],
    }
    out["paired_vs_seed"] = _summary(paired, boot)
    p = "A" if game == "leduc" else "B"
    out["tests"] = {
        f"{p}1": {
            "estimate": float(edge.mean()),
            "ci95": _ci(edge, **boot),
            "p": float(ttest_1samp(edge, 0.0, alternative="greater").pvalue),
        },
        f"{p}2": {
            "estimate": float(safe.mean()),
            "ci95": _ci(safe, **boot),
            "margin": margin,
            "p": float(ttest_1samp(safe, -margin, alternative="greater").pvalue),
        },
        f"{p}3": {
            "estimate": float(paired.mean()),
            "ci95": _ci(paired, **boot),
            "p": float(ttest_rel(scores["evolved"], scores["seed"], alternative="greater").pvalue),
        },
    }
    # secondary: does each belief source matter? (paired on the same sessions, style only)
    out["ablations"] = {}
    for mode in t["beliefs"]:
        if mode != "classifier" and ("evolved", mode) in style_d:
            d = style_d[("evolved", "classifier")] - style_d[("evolved", mode)]
            out["ablations"][f"classifier - {mode}"] = {
                **_summary(d, boot),
                "p_two_sided": float(
                    ttest_rel(style_d[("evolved", "classifier")], style_d[("evolved", mode)]).pvalue
                ),
            }
    return out


# ------------------------------------------------------------------------- classifier track


def classifier_track(cfg: dict, workers: int, boot: dict, log=print) -> dict:
    import lzma
    import pickle

    from pokerbot.evolve import dataset, harness
    from pokerbot.evolve.final import evaluate_programs

    t = cfg["classifier"]
    data = Path(t["data"])
    if not data.exists():
        log(f"classifier: building {t['sessions']} fresh {t['format']} sessions -> {data}")
        dataset.build(
            Path(t["bots"]),
            t["format"],
            cfg.get("split", "test"),
            int(t["sessions"]),
            workers=workers,
            first_index=int(cfg["first_index"]),
            out=data,
        )
    with lzma.open(data, "rb") as f:
        blob = pickle.load(f)  # noqa: S301 - our own dataset
    sessions = blob["sessions"]
    want = dataset.SPLITS[cfg.get("split", "test")] * 10_000_000 + int(cfg["first_index"])
    if sessions[0]["seed"] != want or len(sessions) != int(t["sessions"]):
        raise ValueError(f"{data} does not hold the registered sessions; delete it to rebuild")
    programs = []
    for name, spec in t["programs"].items():
        source, params = _program(spec)
        if params is None:
            params = program.defaults(program.load(source))
        programs.append((name, source, params))
    log(f"classifier: {len(programs)} programs on {len(sessions)} sessions + real players")
    r = evaluate_programs(
        programs,
        t["format"],
        Path(t["profiles"]),
        Path(t["showdowns"]),
        Path(t["real"]),
        test=sessions,
    )
    sc = {k: np.array(v) for k, v in r.pop("session_scores").items()}
    d = sc["evolved"] - sc["seed_a"]
    r["seats"] = int(sum(len(s["labels"]) for s in sessions))
    r["checkpoints"] = list(harness.CHECKPOINTS)
    r["paired"] = {
        f"evolved - {k}": {
            **_summary(sc["evolved"] - sc[k], boot),
            "p": float(ttest_rel(sc["evolved"], sc[k], alternative="greater").pvalue),
        }
        for k in sc
        if k != "evolved"
    }
    r["tests"] = {
        "C1": {
            "estimate": float(d.mean()),
            "ci95": _ci(d, **boot),
            "p": r["paired"]["evolved - seed_a"]["p"],
        }
    }
    return r


# ------------------------------------------------------------------------- whole experiment


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, dict]:
    """Holm-Bonferroni step-down: reject in order of p while p <= alpha / (m - rank)."""
    order = sorted(pvalues, key=pvalues.get)
    m, out, stop = len(order), {}, False
    for rank, h in enumerate(order):
        thr = alpha / (m - rank)
        stop = stop or pvalues[h] > thr
        out[h] = {"p": pvalues[h], "threshold": thr, "supported": not stop}
    return out


def run(cfg: dict, log=print) -> dict:
    workers = int(cfg.get("workers", 8))
    boot = {"resamples": int(cfg["bootstrap"]["resamples"]), "seed": int(cfg["bootstrap"]["seed"])}
    out: dict = {"version": cfg["version"], "first_index": cfg["first_index"], "tracks": {}}
    pool = mp.get_context("spawn").Pool(workers)
    try:
        for game in GAMES:
            if game in cfg:
                out["tracks"][game] = exploiter_track(game, cfg, pool, workers, boot, log)
    finally:
        pool.terminate()
        pool.join()
    if "classifier" in cfg:
        out["tracks"]["classifier"] = classifier_track(cfg, workers, boot, log)
    tests = {h: v for tr in out["tracks"].values() for h, v in tr.get("tests", {}).items()}
    decisions = holm({h: v["p"] for h, v in tests.items()}, float(cfg["alpha"]))
    out["hypotheses"] = {
        h: {"claim": HYPOTHESES[h], **tests[h], **decisions[h]} for h in sorted(tests)
    }
    return out


# ------------------------------------------------------------------------- lineage


def lineage(champion_json: Path, experiments: Path = Path("experiments")) -> list[dict]:
    """Ancestors of a champion, newest first, across runs: each step's run, candidate,
    generation and idea, back to a hand-written seed."""
    meta = json.loads(Path(champion_json).read_text(encoding="utf-8"))
    run_id, cand = meta["run_id"], meta["candidate"]
    chain: list[dict] = []
    for _ in range(200):
        rdir = experiments / run_id
        mpath = rdir / "candidates" / cand / "meta.json"
        m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
        parent = m.get("parent")
        if parent:
            chain.append(
                {
                    "run": run_id,
                    "candidate": cand,
                    "generation": m.get("generation"),
                    "idea": m.get("idea", ""),
                    "parent": parent,
                }
            )
            cand = parent
            continue
        # a seed of this run: a hand-written program or an earlier run's champion
        seeds = (
            json.loads((rdir / "run.json").read_text(encoding="utf-8"))["config"].get(
                "seed_program"
            )
            or []
        )
        src = next(
            (
                x.split(",")
                for x in seeds
                if Path(x.split(",")[0]).stem == cand
                or Path(x.split(",")[0]).stem.startswith(cand + "-")
            ),
            None,
        )
        if src and len(src) > 1 and Path(src[1]).exists():
            prev = json.loads(Path(src[1]).read_text(encoding="utf-8"))
            if prev.get("run_id") and not str(prev.get("candidate", "seed")).startswith("seed"):
                chain.append(
                    {"run": run_id, "candidate": cand, "seed_of_run": True, "file": src[0]}
                )
                run_id, cand = prev["run_id"], prev["candidate"]
                continue
        chain.append({"run": None, "candidate": Path(src[0]).name if src else cand, "seed": True})
        return chain
    raise RuntimeError("lineage too deep")
