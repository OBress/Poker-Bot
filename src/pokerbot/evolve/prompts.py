"""Prompts for the classifier evolution loop and parsing of the LLM's answers.

The model sees the frozen contract, the API, the parent program with its validation scores and a
failure report, short notes on recent attempts, and one focused instruction. It answers with an
``IDEA:`` line and whole replacements for the EVOLVE blocks it changes:

    <<<BLOCK posterior>>>
    ...code...
    <<<END BLOCK>>>

(Whole-block replacement instead of search/replace diffs: blocks are small, and it never fails
on a mismatched search string.) Test seeds and held-out real players never appear in a prompt.
"""

from __future__ import annotations

import re
import textwrap

from pokerbot.evolve import program

SYSTEM = """You improve a Python program that identifies the playing style of poker opponents from
what they do at the table. You edit only the EVOLVE blocks of a frozen template.

Rules (violations are rejected automatically):
- Edit only these blocks: params, helpers, state, observe, posterior. Keep the code outside them.
- Imports: none of your own. Available names: math, np (numpy), and from the API: BUCKETS, LINES,
  Event, HandEnd, HandStart, PlayerAction, SeatFeatures, Showdown, StyleProfiles, param.
- No file/network access, no eval/exec/getattr/globals/type, no attributes or strings with "__",
  no global or nonlocal statements, no module-level mutable state (keep state on self).
- Deterministic: no randomness. posterior(seat, feats) must return K finite non-negative numbers
  (the harness normalizes). It is called for every seat after 10, 25, 50, 100 and 200 hands.
- Keep it fast: the time budget is about 2000 microseconds per posterior call including observe.
- PARAMS: dict of name -> param(default, low, high); a tuner optimizes these values afterwards,
  so expose numeric knobs instead of hard-coding guesses.

What the program gets:
- StyleProfiles (self.profiles), K styles, D features: names, shares (population mix), features
  (stat names), prior_a/prior_b (population Beta prior per feature), f_mean/f_std, means (K, D)
  and covs (K, D, D) of each style in standardized logit space, pooled_rate (K, D), beta_a/beta_b
  (K, D) per-style Beta fits of member rates, showdown (K, len(LINES), BUCKETS) = P(strength
  bucket | betting line, style) from real shown hands. transform(n, x) -> (z, s2): counts of the
  D features to standardized logit features and their noise variances.
- SeatFeatures: hands; counts[stat] = (opportunities n, occurrences x); n(stat), x(stat);
  shown[line, bucket] counts of the seat's hands shown at showdown. Stats: hands, vpip, pfr,
  vpip_<POS>, pfr_<POS> (POS in EP, MP, CO, BTN, SB, BB), vpip_d<D>, pfr_d<D> (stack depth
  bucket 0..3: <25, 25-60, 60-150, >150 bb), rfi, rfi_<POS>, limp, steal, fold_to_steal, 3bet,
  fold_to_3bet, 4bet, cbet_<st>, cbet_<st>_hu|mw, cbet_<st>_ip|oop, fold_to_cbet_<st>, donk_<st>,
  xr_<st>, agg_<st> (st in flop, turn, river), wtsd, wsd, betsize_b0..b7 (first bet / pot bins
  at 0.25, 0.4, 0.6, 0.85, 1.1, 1.6, 2.5).
- observe(event): every public event in order (HandStart(hand_id, button, small_blind,
  big_blind, stacks), BlindPosted, PlayerAction(seat, street, type, amount, to, pot_before,
  to_call, is_all_in), BoardDealt, Showdown(seat, cards, rank), Muck, HandEnd(hand_id, payoffs,
  showdown, payoffs_known)). Use it only if you need features the counts don't have.

Score (higher is better): fitness = -(0.30 LL@10 + 0.25 LL@25 + 0.20 LL@50 + 0.15 LL@100
+ 0.10 LL@200) - ECE@100, where LL@t is the mean log loss of the true style after t hands.
Early identification matters most. Probabilities must be honest: overconfidence is punished.

Answer format: one line "IDEA: <one sentence>", then each changed block as
<<<BLOCK name>>>
<code for the block body, indented as at module level for params/helpers, or written without
the method's indentation for state/observe/posterior (it is re-indented automatically)>
<<<END BLOCK>>>
Change one or more blocks; unchanged blocks can be omitted. No other text is needed."""

INSTRUCTIONS = [
    "Improve identification after 10-25 hands: rely on the stats that have opportunities early "
    "(VPIP, PFR, limp, RFI by position, steal) and handle sparse counts well.",
    "Reduce the confusion between the two most confused styles in the failure report: find stats "
    "or splits (position, stack depth, pot type, street) that separate them and use them.",
    "Use information beyond the 18 profile stats: position splits, stack-depth splits, bet-size "
    "histograms or shown hands, while keeping probabilities calibrated.",
    "Improve calibration: posteriors are over- or under-confident at some checkpoints. Make the "
    "confidence depend on how much evidence there is.",
    "Combine two different likelihood models (for example a Beta-Binomial and a Gaussian with "
    "correlations) into one better-calibrated posterior.",
    "Correct for correlated stats (VPIP and PFR say similar things, so do agg_flop and cbet_flop) "
    "instead of counting the same evidence several times.",
    "Model how a player's stats depend on stack depth (short stacks play differently) so that "
    "short-stacked hands don't mislead the posterior.",
    "Simplify: remove anything that doesn't help and make the posterior faster, without losing "
    "accuracy.",
    "After only 10 hands most stats have one or two opportunities: make the 10-hand posterior "
    "better than the population mix by using what is observed (VPIP, PFR, limps, blind play) "
    "without overreacting to it.",
    "Make the posterior's sharpness depend on the number of hands and opportunities seen, so it "
    "is cautious at 10-25 hands and decisive at 100-200.",
]


def classifier_table(metrics: dict) -> str:
    rows = ["checkpoint | log loss | accuracy | Brier | ECE", "---|---|---|---|---"]
    for t in (10, 25, 50, 100, 200):
        rows.append(
            f"{t} | {metrics[f'logloss@{t}']:.3f} | {metrics[f'accuracy@{t}']:.3f} | "
            f"{metrics[f'brier@{t}']:.3f} | {metrics[f'ece@{t}']:.3f}"
        )
    rows.append(f"fitness {metrics['fitness']:.4f}; {metrics['us_per_call']:.0f} us per call")
    return "\n".join(rows)


def build(
    parent: dict,
    report: str,
    history: list[str],
    instruction: str,
    system: str = SYSTEM,
    table=None,
) -> list[dict]:
    table = table or classifier_table
    user = f"""Parent program:
```python
{parent["source"]}```

Tuned PARAMS of the parent: {parent["params"]}

Validation scores of the parent:
{table(parent["val"])}

Failure report:
{report}

Recent attempts (idea -> change in validation fitness):
{chr(10).join(history[-12:]) or "(none yet)"}

Instruction for this attempt: {instruction}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair(messages: list[dict], answer: str, error: str) -> list[dict]:
    return [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": f"The program was rejected:\n{error}\nFix it. Answer in the same format.",
        },
    ]


_IDEA = re.compile(r"IDEA:\s*(.+)")
_BLOCKS = re.compile(r"<<<BLOCK (\w+)>>>\n(.*?)<<<END BLOCK>>>", re.S)
CLASSIFIER_INDENT = {
    "params": "",
    "helpers": "",
    "state": " " * 8,
    "observe": " " * 8,
    "posterior": " " * 8,
}


def apply_answer(parent_source: str, answer: str, indent: dict | None = None) -> tuple[str, str]:
    """(new source, idea). Raises ``ProgramError('format')`` if no usable block is found."""
    indent = indent or CLASSIFIER_INDENT
    blocks = _BLOCKS.findall(answer)
    if not blocks:
        raise program.ProgramError("format", "no <<<BLOCK name>>> ... <<<END BLOCK>>> found")
    source = parent_source
    for name, body in blocks:
        if name not in indent:
            raise program.ProgramError("format", f"unknown block {name}")
        body = body.strip("\n")
        if body.lstrip().startswith("```"):
            body = re.sub(r"^\s*```\w*\n|\n?```\s*$", "", body)
        body = textwrap.dedent(body)
        ind = indent[name]
        body = "\n".join(ind + line if line.strip() else "" for line in body.split("\n"))
        source = program.replace_block(source, name, body)
    m = _IDEA.search(answer)
    return source, (m.group(1).strip() if m else "")[:300]
