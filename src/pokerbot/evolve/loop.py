"""The classifier evolution loop (EVOLUTION.md §5), recorded as an ``evolve-classifier`` run.

Generation g:
 1. pick parents (the champion half the time, otherwise one of the archive's best);
 2. ask the LLM for ``proposals`` edits, each with its own instruction (DeepSeek via ``llm``);
 3. gate each edit (template, AST, valid and order-independent output); one repair round with
    the error, then discard;
 4. tune PARAMS with CMA-ES on train sessions;
 5. score on val_a; keep the archive;
 6. promote the generation's best candidate only if a one-sided paired t-test over val_a
    sessions gives p < 0.01 against the champion *and* the same holds on the fresh val_b batch
    (MATH.md §7.5).
Test sessions and held-out real players are never used here.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from pokerbot.evolve import program, prompts
from pokerbot.evolve.api import StyleProfiles
from pokerbot.evolve.evaluator import Evaluator
from pokerbot.evolve.program import ProgramError
from pokerbot.evolve.report import classifier_report
from pokerbot.evolve.tuner import tune

PROMOTE_P = 0.01
ARCHIVE = 8
# DeepSeek V4.1 Flash reasons before answering (~20k tokens on these prompts); a smaller budget
# returns empty answers.
MAX_TOKENS = 32_000


def _candidate(cid, source, parent, idea, instruction, gen, params, train_f, val) -> dict:
    return {
        "id": cid,
        "source": source,
        "parent": parent,
        "idea": idea,
        "instruction": instruction,
        "gen": gen,
        "params": params,
        "train": train_f,
        "val": val.metrics,
        "val_scores": val.session_scores,
        "val_result": val,
    }


def _log_candidate(run, c: dict, prompt_hash: str | None) -> None:
    m = {f"val/{k}": v for k, v in c["val"].items() if isinstance(v, (int, float))}
    run.log_metrics({**m, "train/fitness": c["train"]}, step=c["gen"], candidate=c["id"])
    run.save_candidate(
        c["id"],
        c["source"],
        parent=c["parent"],
        idea=c["idea"],
        instruction=c["instruction"],
        generation=c["gen"],
        params=c["params"],
        train_fitness=c["train"],
        val=c["val"],
        prompt=prompt_hash,
    )


def evolve_classifier(
    run,
    llm,
    fmt: str,
    profiles_path: Path,
    showdown_path: Path,
    seeds: dict[str, str | tuple[str, dict]],
    generations: int,
    proposals: int,
    workers: int = 8,
    seed: int = 0,
    tune_iters: int = 8,
    max_calls: int = 400,
    data_root: Path | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    prof = StyleProfiles.load(profiles_path, fmt, showdown_path)
    kw = {"root": data_root} if data_root else {}
    ev = Evaluator(fmt, str(profiles_path), str(showdown_path), workers=workers, **kw)
    archive: list[dict] = []
    history: list[str] = []
    try:
        for name, entry in seeds.items():
            src, given = entry if isinstance(entry, tuple) else (entry, {})
            ns = program.load(src)
            start = {k: min(max(given.get(k, v.default), v.low), v.high)
                     for k, v in ns["PARAMS"].items()}  # fmt: skip
            ev.gate(src, start)
            params, train_f = tune(ev, src, ns["PARAMS"], start, iters=tune_iters, seed=seed)
            c = _candidate(
                name,
                src,
                None,
                "hand-written seed",
                "",
                0,
                params,
                train_f,
                ev.evaluate(src, params, "val_a"),
            )
            _log_candidate(run, c, None)
            archive.append(c)
        archive.sort(key=lambda c: -c["val"]["fitness"])
        champion = archive[0]
        run.event("champion", candidate=champion["id"], fitness=champion["val"]["fitness"])
        for g in range(1, generations + 1):
            if llm.usage.calls >= max_calls:
                run.event("budget_stop", calls=llm.usage.calls)
                break
            t0 = time.perf_counter()
            parents = [
                champion if rng.random() < 0.5 else archive[int(rng.integers(len(archive)))]
                for _ in range(proposals)
            ]
            instr = [
                prompts.INSTRUCTIONS[(g * proposals + i) % len(prompts.INSTRUCTIONS)]
                for i in range(proposals)
            ]
            msgs = [
                prompts.build(
                    p,
                    classifier_report(p["val_result"].posteriors, p["val_result"].labels, prof),
                    history,
                    ins,
                )
                for p, ins in zip(parents, instr, strict=True)
            ]
            calls0 = llm.usage.calls
            answers = llm.complete_many(
                msgs, temperature=0.9, max_tokens=MAX_TOKENS, purpose=f"classifier gen {g}"
            )
            failed: dict[str, int] = {}
            new: list[dict] = []
            for i, (parent, m, a) in enumerate(zip(parents, msgs, answers, strict=True)):
                cid = f"g{g:02d}-{i}"
                phash = hashlib.sha256(json.dumps(m).encode()).hexdigest()[:12]
                text = a.text
                src = idea = None
                for attempt in range(2):
                    try:
                        src, idea = prompts.apply_answer(parent["source"], text)
                        ns = program.load(src)
                        start = {
                            k: parent["params"].get(k, v.default) for k, v in ns["PARAMS"].items()
                        }
                        start = {
                            k: min(max(v, ns["PARAMS"][k].low), ns["PARAMS"][k].high)
                            for k, v in start.items()
                        }
                        ev.gate(src, start)
                        break
                    except ProgramError as e:
                        if attempt == 1:
                            failed[e.reason] = failed.get(e.reason, 0) + 1
                            run.event(
                                "candidate_failed",
                                candidate=cid,
                                reason=e.reason,
                                detail=e.detail[-500:],
                            )
                            history.append(f"- {idea or '(unparsed)'} -> rejected ({e.reason})")
                            src = None
                            break
                        text = llm.complete(
                            prompts.repair(m, text, str(e)[-1500:]),
                            temperature=0.5,
                            max_tokens=MAX_TOKENS,
                            purpose=f"repair {cid}",
                        ).text
                if src is None:
                    continue
                try:
                    params, train_f = tune(
                        ev, src, ns["PARAMS"], start, iters=tune_iters, seed=seed + g
                    )
                    val = ev.evaluate(src, params, "val_a")
                except ProgramError as e:
                    failed[e.reason] = failed.get(e.reason, 0) + 1
                    run.event(
                        "candidate_failed", candidate=cid, reason=e.reason, detail=e.detail[-500:]
                    )
                    continue
                c = _candidate(cid, src, parent["id"], idea, instr[i], g, params, train_f, val)
                _log_candidate(run, c, phash)
                delta = c["val"]["fitness"] - parent["val"]["fitness"]
                history.append(f"- {idea} -> {delta:+.4f} (parent {parent['id']})")
                new.append(c)
            # archive and promotion
            fits = [c["val"]["fitness"] for c in new]
            promoted = None
            if new:
                best = max(new, key=lambda c: c["val"]["fitness"])
                if best["val"]["fitness"] > champion["val"]["fitness"]:
                    p_a = float(
                        ttest_rel(
                            best["val_scores"], champion["val_scores"], alternative="greater"
                        ).pvalue
                    )
                    p_b = None
                    if p_a < PROMOTE_P:
                        vb_new = ev.evaluate(best["source"], best["params"], "val_b")
                        vb_old = ev.evaluate(champion["source"], champion["params"], "val_b")
                        p_b = float(
                            ttest_rel(
                                vb_new.session_scores, vb_old.session_scores, alternative="greater"
                            ).pvalue
                        )
                    run.event(
                        "promotion_test",
                        candidate=best["id"],
                        champion=champion["id"],
                        p_val_a=p_a,
                        p_val_b=p_b,
                    )
                    if p_a < PROMOTE_P and p_b is not None and p_b < PROMOTE_P:
                        promoted = best
                        champion = best
                        run.event(
                            "promoted",
                            candidate=best["id"],
                            p_val_a=p_a,
                            p_val_b=p_b,
                            fitness=best["val"]["fitness"],
                        )
            archive = sorted(archive + new, key=lambda c: -c["val"]["fitness"])[:ARCHIVE]
            if champion not in archive:
                archive.append(champion)
            usage = llm.usage
            run.log_metrics(
                {
                    "gen/best_fitness": max(fits) if fits else float("nan"),
                    "gen/median_fitness": float(np.median(fits)) if fits else float("nan"),
                    "gen/worst_fitness": min(fits) if fits else float("nan"),
                    "gen/champion_fitness": champion["val"]["fitness"],
                    "gen/train_val_gap": champion["train"] - champion["val"]["fitness"],
                    "gen/proposed": proposals,
                    "gen/scored": len(new),
                    "gen/failed": sum(failed.values()),
                    "gen/promoted": int(promoted is not None),
                    "gen/archive_size": len(archive),
                    "llm/calls": usage.calls - calls0,
                    "llm/prompt_tokens": usage.prompt_tokens,
                    "llm/completion_tokens": usage.completion_tokens,
                    "llm/cache_hit_rate": usage.cached_calls / max(usage.calls, 1),
                    "sys/gen_seconds": time.perf_counter() - t0,
                },
                step=g,
                **{f"failed_{k}": v for k, v in failed.items()},
            )
            print(
                f"gen {g}: scored {len(new)}/{proposals}, best "
                f"{max(fits) if fits else float('nan'):.4f}, champion {champion['id']} "
                f"{champion['val']['fitness']:.4f}{' (promoted)' if promoted else ''}; "
                f"failed {failed}",
                flush=True,
            )
    finally:
        ev.close()
    return {"champion": champion, "archive": archive}
