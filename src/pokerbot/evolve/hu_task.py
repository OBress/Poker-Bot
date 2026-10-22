"""The heads-up hold'em exploiter as an evolution task (Track B): evaluator, prompts, report.

Sessions per split: train 120, val_a 300, val_b 300, test 400 (``test`` only in final
evaluation). Hold'em hands can't be valued exactly, so splits are large: once the exploiter
deviates the duplicate hands diverge and session results vary by tens of bb/100.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np

from pokerbot.evolve import hu, program
from pokerbot.evolve.evaluator import Result
from pokerbot.evolve.program import ProgramError

TEMPLATE = "exploiter_hu_v1"
SIZES = {"train": 120, "val_a": 300, "val_b": 300, "test": 400}
INDENT = {"params": "", "helpers": "", "state": " " * 8, "observe": " " * 8, "policy": " " * 8}


def _run(source: str, params: dict, split: str, idx: list[int], mode: str):
    try:
        ns = program.load(source, TEMPLATE)
        prof = hu.profiles()
        out = []
        for i in idx:
            out.append(
                hu.run_session(
                    lambda: ns["Exploiter"](prof, dict(params)),
                    hu.SessionSpec(split, i),
                    beliefs_mode=mode,
                )
            )
        return {"ok": True, "results": out}
    except ProgramError as e:
        return {"ok": False, "reason": e.reason, "detail": e.detail}
    except Exception:  # noqa: BLE001 - a candidate crash is a result
        tb = traceback.format_exc().splitlines()
        return {"ok": False, "reason": "crash", "detail": "\n".join(tb[-6:])}


class HUEvaluator:
    def __init__(
        self,
        workers: int = 8,
        timeout_s: float = 900.0,
        allow_test: bool = False,
        mode: str = "classifier",
    ) -> None:
        self.workers, self.timeout_s, self.allow_test, self.mode = (
            workers,
            timeout_s,
            allow_test,
            mode,
        )
        self.sizes = SIZES
        self._start()

    def _start(self) -> None:
        self.pool = mp.get_context("spawn").Pool(self.workers)

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()

    def evaluate(
        self, source: str, params: dict, split: str, n: int | None = None, reverse: bool = False
    ) -> Result:
        if split == "test" and not self.allow_test:
            raise PermissionError("test sessions are for final evaluation only")
        idx = list(range(min(n or SIZES[split], SIZES[split])))
        if reverse:
            idx = idx[::-1]
        parts = [idx[i :: self.workers] for i in range(self.workers) if idx[i :: self.workers]]
        jobs = [self.pool.apply_async(_run, (source, params, split, p, self.mode)) for p in parts]
        outs = []
        try:
            for j in jobs:
                outs.append(j.get(self.timeout_s))
        except mp.TimeoutError:
            self.close()
            self._start()
            raise ProgramError("timeout", f"evaluation exceeded {self.timeout_s:.0f}s") from None
        for o in outs:
            if not o["ok"]:
                raise ProgramError(o["reason"], o["detail"])
        flat_idx = [i for p in parts for i in p]
        flat = [r for o in outs for r in o["results"]]
        results = [r for _, r in sorted(zip(flat_idx, flat, strict=True), key=lambda t: t[0])]
        metrics = hu.summarize(results)
        scores = np.array([hu.session_score(r) for r in results])
        return Result(metrics, scores, results, None, metrics["invalid"])

    def gate(self, source: str, params: dict) -> None:
        program.check(source, TEMPLATE)
        a = self.evaluate(source, params, "train", n=8)
        if a.invalid:
            raise ProgramError("invalid_output", f"{a.invalid} invalid distributions")
        b = self.evaluate(source, params, "train", n=8, reverse=True)
        if not all(
            np.allclose(x["d"], y["d"]) for x, y in zip(a.posteriors, b.posteriors, strict=True)
        ):
            raise ProgramError("state_leak", "results depend on session order")


SYSTEM = """You improve a Python program that exploits opponents in heads-up no-limit hold'em
(100 big-blind stacks). You edit only the EVOLVE blocks of a frozen template.

Rules (violations are rejected automatically):
- Edit only these blocks: params, helpers, state, observe, policy. Keep code outside them.
- No imports of your own. Available: math, np (numpy), MENU_NAMES, HUHandRecord, HUState,
  SeatFeatures, StyleProfiles, param.
- No file/network access, no eval/exec/getattr/globals/type, no attributes or strings with "__",
  no global or nonlocal statements, no module-level mutable state (keep state on self).
- Deterministic (no randomness): return probabilities; the harness samples the action.
- policy(...) must return 6 finite non-negative numbers, one per MENU_NAMES = (fold, call,
  bet33, bet75, bet125, allin): bets/raises of 33%, 75%, 125% of the pot, or all-in. Illegal
  actions (state.legal False) are masked by the harness.
- PARAMS: name -> param(default, low, high); a tuner optimizes them, so expose numeric knobs.
- Keep it fast: about 2000 microseconds per decision at most.

Inputs:
- state: HUState(street 0-3, button, hole, board, strength (preflop 1 - percentile, 1 = AA;
  postflop equity vs a random hand), pot, to_call, my_stack, opp_stack (all in big blinds),
  legal (6 bools), actions (this hand: (street, "me"/"opp", "f"/"c"/"r", street total bb))).
- beliefs: probabilities over self.profiles.names: four heads-up styles and, last, the
  equilibrium baseline itself ("Equilibrium (blueprint)"). From a frozen classifier.
- self.profiles: StyleProfiles. pooled_rate[k, j] are style k's real rates of the stats in
  features (vpip, pfr, limp, rfi_BTN, 3bet, fold_to_3bet, fold_to_steal, cbet_flop, cbet_turn,
  fold_to_cbet_flop, fold_to_cbet_turn, xr_flop, donk_flop, agg_flop, agg_turn, agg_river,
  wtsd); showdown[k, line, bucket] shown-hand strength by line; shares; means/covs.
- opp: SeatFeatures of this opponent: counts[stat] = (opportunities, occurrences), e.g.
  opp.n("fold_to_cbet_flop"), opp.x("fold_to_cbet_flop"); opp.hands.
- baseline: the blueprint (an approximate equilibrium strategy) probabilities over the menu.
- observe_hand(record): HUHandRecord(button, hole, board, opp_cards (None unless shown),
  actions, payoff in bb) after every hand.

Scoring (duplicate: the same cards and opponents are played by your program and by the
blueprint): d = your bb per hand - the blueprint's, reported in bb/100. fitness = mean over
sessions of: d vs style opponents (70% of sessions); -max(0, -d - 2) vs the blueprint itself
(15%); -max(0, -d) vs players who switch style every 50 hands (15%); each minus 2 x mean
KL(your distribution || baseline); minus 0.01 per microsecond per decision above 2000.
Results are noisy (hands diverge once you deviate): prefer robust, well-founded deviations
(fold equity vs fold rates, value betting vs callers, bluff-catching vs aggressive players)
over many small tweaks.

Answer format: one line "IDEA: <one sentence>", then each changed block as
<<<BLOCK name>>>
<code (params/helpers at module level; state/observe/policy without the method's indentation)>
<<<END BLOCK>>>"""

INSTRUCTIONS = [
    "Compute each action's expected value against each style from its fold and call rates "
    "(MATH: a bluff of size B into pot P needs fold rate B/(P+B)) and tilt toward the best.",
    "Use this opponent's own observed counts (opp.n / opp.x) together with the beliefs, so "
    "players who differ from their style are exploited correctly.",
    "Exploit only when confident and stay close to the baseline otherwise, including against "
    "the equilibrium baseline (last belief) and when beliefs are spread.",
    "Improve postflop play: continuation bets, bluff-catching and value bets by street, using "
    "the styles' fold-to-c-bet and aggression rates.",
    "Improve preflop play: steals, 3-bets and defending against each style's opening, limping "
    "and folding tendencies.",
    "Target the style where the gain is smallest in the failure report.",
    "Simplify: remove anything that doesn't help, keeping the gains.",
]


def table(m: dict) -> str:
    keys = ["fitness", "bb100_style", "bb100_nash", "bb100_switch", "kl", "us_per_call"]
    keys += sorted(k for k in m if k.startswith("bb100_vs_"))
    return "\n".join(f"{k}: {m[k]:.3f}" for k in keys if k in m)


def report(r: Result) -> str:
    res = r.posteriors
    names = hu.profiles().names
    lines = []
    for k in range(len(names) - 1):
        v = [100 * x["d"].mean() for x in res if x["kind"] == "style" and x["labels"][0] == k]
        if v:
            lines.append(
                f"vs {names[k]}: {np.mean(v):+.1f} bb/100 over {len(v)} sessions "
                f"(session sd {np.std(v):.0f})"
            )
    for kind in ("nash", "switch"):
        v = [100 * x["d"].mean() for x in res if x["kind"] == kind]
        if v:
            lines.append(f"vs {kind}: {np.mean(v):+.1f} bb/100 over {len(v)} sessions")
    return "\n".join(lines)
