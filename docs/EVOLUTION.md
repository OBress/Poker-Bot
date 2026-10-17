# Evolvable programs: how the LLM edits the algorithms

This is the detailed design for the classifier and exploiter programs that DeepSeek rewrites between
rounds. It expands ARCHITECTURE §4.7–4.9. The formulas for the seed programs and the scoring rules are in
[MATH.md](MATH.md).

## 1. The problem

We want two things that pull against each other:

- **Freedom:** the LLM must be able to invent new features, formulas and control flow, not just
  turn knobs.
- **A fixed format:** every candidate must take the same inputs and return the same outputs, so the
  harness can score any candidate, compare candidates fairly, and run them millions of times.

The answer is a **frozen harness around an editable core**. The harness owns everything that touches
the game: what information a program may see, how outputs are validated, how actions are sampled,
how scoring works. The program only owns the math in between.

```
 engine (hidden info)                frozen harness                       editable program
 ─────────────────────   ┌──────────────────────────────────────┐   ┌──────────────────────────┐
 full HandState  ──────► │ redact: public events + own cards    │──►│ observe(event)           │
 true style labels ─┐    │ shared feature tracker (HUD counts)  │──►│ posterior(seat, feats)   │
                    │    │ validate output (finite, sums to 1)  │◄──│   -> np.ndarray[K]       │
                    │    └──────────────────────────────────────┘   └──────────────────────────┘
                    └──► evaluator only (never passed to programs)
```

## 2. One module per candidate, with marked editable blocks

Each candidate is a single Python file. Parts outside `EVOLVE` markers are frozen: the harness checks
that they are byte-identical to the template before running the candidate. The LLM edits only inside
the markers, using search/replace diffs (small, targeted edits work much better than full rewrites
and keep a readable lineage).

```python
"""Classifier candidate. Frozen template v1."""
import math
import numpy as np
from pokerbot.evolve.api import Event, SeatFeatures, StyleProfiles, param

# ── EVOLVE-BLOCK params ──────────────────────────────────────────────
# Numeric knobs the tuner may change. param(default, low, high).
PARAMS = {
    "prior_strength": param(20.0, 1.0, 200.0),
    "aggression_weight": param(1.0, 0.0, 5.0),
}
# ── END-EVOLVE-BLOCK ─────────────────────────────────────────────────


class Classifier:
    def __init__(self, profiles: StyleProfiles, params: dict[str, float]) -> None:
        self.profiles = profiles
        self.p = params
        self.reset()

    def reset(self) -> None:  # called at the start of every session
        # ── EVOLVE-BLOCK state ───────────────────────────────────────
        self.extra: dict[int, dict] = {}
        # ── END-EVOLVE-BLOCK ─────────────────────────────────────────

    def observe(self, event: Event) -> None:
        # ── EVOLVE-BLOCK observe ─────────────────────────────────────
        pass  # optional: build custom features from raw public events
        # ── END-EVOLVE-BLOCK ─────────────────────────────────────────

    def posterior(self, seat: int, feats: SeatFeatures) -> np.ndarray:
        # ── EVOLVE-BLOCK posterior ───────────────────────────────────
        ...  # must return shape (K,), non-negative, finite
        # ── END-EVOLVE-BLOCK ─────────────────────────────────────────
```

The LLM may add helper functions inside an `EVOLVE-BLOCK helpers` section and add or remove entries
in `PARAMS`. It may not change signatures, imports, or the frozen code.

## 3. Standard inputs and outputs

### Classifier

| | Type | Notes |
|---|---|---|
| in | `Event` stream | The engine's public events only: blinds, actions (with pot and to-call), board cards, **showdown reveals and mucks**. No hole cards of other players unless they were actually shown. |
| in | `SeatFeatures` | Maintained by the harness for every seat: opportunity/occurrence **counts** for each stat (VPIP, PFR, 3-bet, c-bet, fold-to-c-bet, WTSD, …), split by position, stack depth, pot type (heads-up/multiway), preflop line and in/out of position, plus bet-size histograms and shown-hand strengths by line. Counts, not ratios, so programs can do proper small-sample statistics. Definitions: [MATH.md](MATH.md) §1–§2. |
| in | `StyleProfiles` | The K style definitions from the real-data clustering (means, covariances, population shares). Read-only. |
| out | `np.ndarray` shape (K,) | Harness clips negatives, renormalizes, and penalizes invalid output. |

### Exploiter

| | Type | Notes |
|---|---|---|
| in | `DecisionState` | The same object any player gets: own hole cards, board, pot, stacks, bets, legal actions, public action history. The harness adds derived context: own position, effective stack in bb, players in the pot, in/out of position. |
| in | `beliefs` | Current classifier posterior for each opponent seat. |
| in | `SeatFeatures` | Same as above, for every opponent. |
| in | `baseline` | The equilibrium baseline's **probability distribution** over the abstract action menu at this decision. |
| out | `ActionDist` | Probabilities over the same menu: fold, check/call, bet 33% / 75% / 125% pot, all-in. |

Why the exploiter outputs a distribution and not an action:

- The harness samples with its own seeded RNG, so programs contain no randomness and are pure,
  deterministic functions. Same inputs, same output, always.
- The harness can measure **how far** the program deviates from the baseline at every decision
  (KL divergence) and enforce or penalize it. This is the safety lever: with uncertain beliefs the
  program should stay close to the baseline.
- Mapping abstract sizes to legal concrete amounts is done once, correctly, by the harness.

## 4. What stops cheating

| Leak | Guard |
|---|---|
| Seeing other players' cards | Programs get only the redacted event stream; cards appear only on real showdown reveals. The engine implements real show/muck order (last river aggressor first, losers may muck, all hands face up when someone is all-in), so a program learns exactly what a real player could. |
| Seeing the true style labels | Labels live only in the evaluator. Programs get posteriors scored against them, never the labels. |
| Reading the deck or future cards | Programs run in a separate process and get immutable copies, never engine objects. |
| Getting information from the harness | Import allow-list (`numpy`, `math`, `pokerbot.evolve.api` types only), no file or network access, no `globals`/`eval`/introspection tricks (AST check). |
| Remembering test results | Test seeds never run during evolution; the LLM never sees them or their scores. |
| State leaking across sessions | The harness calls `reset()` and also checks that running sessions in a different order gives identical scores. |

## 5. The loop (the "RL environment")

Strictly, this is **evolutionary program search with an LLM as the mutation operator**, plus
black-box numeric optimization. It is the right tool here: the output must be a plain numeric
algorithm, and the reward (win rate, classification accuracy) is noisy and not differentiable.

```
environment   = simulated tables of style-sampled opponents (seeded)
episode       = evaluate one program on a batch of seeded sessions
reward        = fitness (see ARCHITECTURE §4.7, §4.8)
policy update = LLM edits code  +  CMA-ES tunes PARAMS
```

Each generation:

1. **Select parents** from an archive kept diverse along a few descriptors (for example: runtime,
   early versus late accuracy, deviation size), so the search doesn't collapse onto one idea.
2. **Build the prompt:** frozen template and contract, parent code, its scores, a structured
   **failure report**, short summaries of related past attempts ("tried X, it didn't help because Y"),
   and one concrete instruction (fix confusion A/B, cut runtime, use a feature, simplify).
3. **LLM proposes** k search/replace diffs (DeepSeek V4.1 Flash; cheap, so k can be large).
4. **Gate:** frozen regions untouched → AST allow-list → contract property tests (valid output on
   random inputs, deterministic, within time budget). Failures get one repair round with the
   traceback, then are discarded.
5. **Tune:** a short CMA-ES run on `PARAMS` using **train** seeds.
6. **Score** on **validation** seeds: cheap screen first (~10k hands); only promising candidates get
   the long evaluation.
7. **Promote** a new champion only if it beats the current one with statistical significance.

Failure reports are what make the LLM useful:

- Classifier: confusion matrix at 10/25/50/100/200 hands, calibration, the styles most confused
  early, a few summarized hand sequences of the worst-classified seats, runtime per call.
- Exploiter: Δ bb/100 vs baseline per opponent style, per street and spot type (where it lost
  money relative to the baseline), average deviation size versus belief confidence.

## 6. Classifier and exploiter together

- **Stage 1:** evolve the classifier alone (its reward needs no exploiter).
- **Stage 2:** freeze the champion classifier; evolve the exploiter using its beliefs. Ablation
  runs give it oracle (true) labels to measure how much classification error costs.
- **Stage 3:** alternate rounds. A classifier change is kept only if it doesn't lower the exploiter's
  win rate. Eventually the classifier is scored on what matters: money won.

## 7. Where it lives

```
src/pokerbot/evolve/
  api.py          frozen types given to programs (Event, SeatFeatures, ActionDist, param, …)
  templates/      frozen templates: classifier_v1.py, exploiter_v1.py
  harness.py      redaction, feature tracking, output validation, action sampling
  sandbox.py      subprocess runner, limits, AST allow-list
  gates.py        frozen-region check, contract property tests
  tuner.py        CMA-ES on PARAMS
  prompts/        prompt templates
  report.py       failure reports
  population.py   archive, lineage, promotion test
  run.py          the generation loop
programs/
  classifier/     seed.py, then <gen>-<id>.py + .json (parents, diff, prompt hash, params, scores, cost)
  exploiter/
```

## 8. Honest expectations

- Classifying from 20–50 hands is noisy; posteriors will be soft early and the exploiter must handle
  that. That's why deviation is scaled by confidence.
- Most of the edge will come from the most exploitable styles (stations, maniacs, nits). Against
  solid players the exploiter should simply play close to the baseline.
- The first full run of this loop is on Leduc poker, where everything is fast and the baseline is
  exact. That's where we debug the loop before spending compute on hold'em.
