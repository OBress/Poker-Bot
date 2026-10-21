"""The Leduc exploiter as an evolution task: sandboxed evaluator, prompts, failure report.

Session counts per split: train 60, val_a 120, val_b 120, test 200 (``test`` only in final
evaluation). Sessions are generated from their seeds on the fly (Leduc is cheap), in worker
processes, never in the evolution process.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np

from pokerbot.evolve import leduc, program
from pokerbot.evolve.evaluator import Result
from pokerbot.evolve.program import ProgramError

TEMPLATE = "exploiter_leduc_v1"
SIZES = {"train": 60, "val_a": 120, "val_b": 120, "test": 200}
INDENT = {"params": "", "helpers": "", "state": " " * 8, "observe": " " * 8, "policy": " " * 8}


def _run(
    source: str, params: dict, split: str, idx: list[int], oracle: bool, prior_only: bool = False
):
    try:
        ns = program.load(source, TEMPLATE)
        table = leduc.style_table()
        out = []
        for i in idx:
            r = leduc.run_session(
                lambda: ns["Exploiter"](table, dict(params)),
                leduc.SessionSpec(split, i),
                oracle=oracle,
            )
            out.append(r)
        return {"ok": True, "results": out}
    except ProgramError as e:
        return {"ok": False, "reason": e.reason, "detail": e.detail}
    except Exception:  # noqa: BLE001 - a candidate crash is a result
        tb = traceback.format_exc().splitlines()
        return {"ok": False, "reason": "crash", "detail": "\n".join(tb[-6:])}


class LeducEvaluator:
    def __init__(
        self,
        workers: int = 8,
        timeout_s: float = 300.0,
        allow_test: bool = False,
        oracle: bool = False,
        prior_only: bool = False,
    ) -> None:
        self.workers, self.timeout_s = workers, timeout_s
        self.allow_test, self.oracle, self.prior_only = allow_test, oracle, prior_only
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
        jobs = [
            self.pool.apply_async(_run, (source, params, split, p, self.oracle, self.prior_only))
            for p in parts
        ]
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
        metrics = leduc.summarize(results)
        scores = np.array([leduc.session_score(r) for r in results])
        return Result(metrics, scores, results, None, metrics["invalid"])

    def gate(self, source: str, params: dict) -> None:
        program.check(source, TEMPLATE)
        a = self.evaluate(source, params, "train", n=10)
        if a.invalid:
            raise ProgramError("invalid_output", f"{a.invalid} invalid distributions")
        b = self.evaluate(source, params, "train", n=10, reverse=True)
        if not all(
            np.allclose(x["d"], y["d"]) for x, y in zip(a.posteriors, b.posteriors, strict=True)
        ):
            raise ProgramError("state_leak", "results depend on session order")


SYSTEM = """You improve a Python program that exploits opponents in Leduc poker. You edit only
the EVOLVE blocks of a frozen template.

The game (two players, Leduc hold'em): deck J, Q, K twice each (ranks 0, 1, 2). Each player antes
1 chip and gets one private card. Round 1: bets of 2 chips; round 2 (after one public card is
dealt): bets of 4 chips; at most 2 bets/raises per round. Seat 0 acts first each round; seats
alternate every hand. Showdown: a card pairing the public card wins, else the higher card;
equal ranks split. Actions: c = check/call, r = bet/raise, f = fold.

Rules (violations are rejected automatically):
- Edit only these blocks: params, helpers, state, observe, policy. Keep code outside them.
- No imports of your own. Available: math, np (numpy), HandRecord, LeducState,
  OpponentFeatures, StyleTable, param.
- No file/network access, no eval/exec/getattr/globals/type, no attributes or strings with "__",
  no global or nonlocal statements, no module-level mutable state (keep state on self).
- Deterministic (no randomness): return probabilities; the harness samples the action.
- policy(state, beliefs, opp, baseline) must return len(state.legal) finite non-negative numbers.
- PARAMS: name -> param(default, low, high); a tuner optimizes them, so expose numeric knobs.

Inputs:
- state: LeducState(seat, card, public (None in round 1), history (tuple of round strings),
  legal (e.g. "cr" or "fcr"), put (chips each seat has put in), to_call, bet_size).
- beliefs: K probabilities over the opponent's style from a frozen Bayesian classifier (styles:
  self.styles.names = nit, calling station, maniac, fit-or-fold, bluffer).
- self.styles.centers[k] = [bet, call, raise] x [weak, medium, strong]: the style's typical
  player's probabilities of betting when not facing a bet, continuing (call or raise) when facing
  a bet, and raising given it continues. Strength: round 1 J/Q/K = weak/medium/strong; round 2
  pair with the public card = strong, card above the public card = medium, otherwise weak.
  Individual opponents vary around their style's center.
- opp: OpponentFeatures(hands, counts[(round, "open"|"facing", action)], shown[(final round
  line, card)]), opp.rate(round, situation, action): this opponent's own observed frequencies.
- baseline: the exact Nash equilibrium's probabilities at this decision (unexploitable).
- observe_hand(record): HandRecord(seat, card, public, history, opp_card (None unless shown at
  showdown), payoff) after every hand.

Scoring (exact, duplicate: the same cards are played by your program and by the baseline):
d = your expected chips per hand - the baseline's against the same opponent.
fitness = mean over sessions of: d vs style opponents (70% of sessions); -max(0, -d - 0.01)
vs the Nash baseline itself (15%); -max(0, -d) vs players who switch style every 25 hands (15%);
each minus 0.02 x mean KL(your distribution || baseline), minus 0.0001 per microsecond per
decision above 2000 (keep policy fast: it runs ~15 times per hand). A best response to each
style would gain about 0.5-1.2 chips per hand over the baseline, so there is a lot of room; the
risk is
losing to the baseline or to switchers by trusting wrong beliefs.

Answer format: one line "IDEA: <one sentence>", then each changed block as
<<<BLOCK name>>>
<code (params/helpers at module level; state/observe/policy without the method's indentation)>
<<<END BLOCK>>>"""

INSTRUCTIONS = [
    "Compute action values against each style more precisely: use the style's center "
    "probabilities and simple one-step lookahead (pot odds, fold equity, value when called).",
    "Use this opponent's own observed frequencies (opp.rate) together with the beliefs, so "
    "players who differ from their style's center are exploited correctly.",
    "Exploit more when the beliefs are confident and stay close to the baseline otherwise; make "
    "the deviation size depend on how much evidence there is.",
    "Improve the second round (after the public card): pairs, overcards and bluff-catching "
    "decisions are where most chips are won or lost.",
    "Reduce losses against the Nash baseline and against style switchers without giving up the "
    "gains against exploitable styles (detect changes in the opponent's behaviour).",
    "Target the style where the gain is smallest in the failure report.",
    "Simplify: remove anything that doesn't help, keeping the gains.",
    "Stop losing to the Nash baseline: when the opponent's observed behaviour fits no style well "
    "(it may be playing close to equilibrium), shrink the deviation toward the baseline.",
    "Make the program faster without changing its decisions: cache per-hand computations in "
    "self, avoid repeated loops over styles and cards.",
    "Protect against style switchers: weight recent hands more when estimating the opponent's "
    "frequencies and reduce deviation when recent behaviour contradicts the beliefs.",
]


def table(m: dict) -> str:
    keys = [
        "fitness",
        "delta_style",
        "delta_nash",
        "delta_switch",
        "delta_style_first20",
        "delta_style_last40",
        "kl",
        "us_per_call",
    ]
    keys += sorted(k for k in m if k.startswith("delta_vs_"))
    return "\n".join(f"{k}: {m[k]:.4f}" for k in keys if k in m)


def report(r: Result) -> str:
    results = r.posteriors
    worst = sorted((x for x in results if x["kind"] == "style"), key=lambda x: x["d"].mean())[:5]
    names = list(leduc.STYLES)
    lines = ["Lowest-gain style sessions (style, mean d, d over hands 1-20 / 21-60 / 61-100):"]
    for x in worst:
        d = x["d"]
        lines.append(
            f"- {names[x['labels'][0]]}: {d.mean():+.3f} "
            f"({d[:20].mean():+.3f} / {d[20:60].mean():+.3f} / {d[60:].mean():+.3f})"
        )
    nash = [x["d"].mean() for x in results if x["kind"] == "nash"]
    sw = [x["d"].mean() for x in results if x["kind"] == "switch"]
    lines.append(
        f"Vs the Nash baseline: mean d {np.mean(nash):+.4f}, worst session {np.min(nash):+.4f}"
        if nash
        else ""
    )
    lines.append(
        f"Vs switchers: mean d {np.mean(sw):+.4f}, worst session {np.min(sw):+.4f}" if sw else ""
    )
    return "\n".join(x for x in lines if x)
