# Architecture

This document describes how the system is put together and why. The phased build order lives in
[ROADMAP.md](ROADMAP.md).

## 1. The claim we are testing

> Against a population of exploitable (non-equilibrium) players, an agent that **identifies** each
> opponent's style and **adapts** to it earns significantly more than an equilibrium (GTO-style)
> engine playing the same cards against the same opponents.

An equilibrium strategy is built to be *unexploitable*, not to *maximally exploit*. It wins against
weak players, but it leaves money on the table because it never changes what it does. We want to
measure how much money, and show that a classifier-driven exploiter can collect it without becoming
exploitable itself.

What counts as proof:

| Requirement | How we meet it |
|---|---|
| Same opponents, same luck | **Duplicate poker**: the exploiter and the baseline each sit in the same seat with the same deck seeds against the same sampled opponents. |
| Real effect, not noise | Report Δ bb/100 (exploiter − baseline) with 95% confidence intervals over enough hands to make them tight; pre-register the test seed set. |
| Not overfit to our own bots | Held-out test seeds the LLM never sees, held-out opponent variations, and a check against classifier labels learned from **real** hand histories. |
| Not just "playing worse but luckier" | Exploiter is also evaluated against the baseline itself and against adversarial "style-switching" bots; it must lose little or nothing there. |
| Baseline is actually strong | In small games (Kuhn/Leduc) the baseline is an exact Nash equilibrium. In hold'em it is a CFR blueprint whose strength we sanity-check (see §6). |

## 2. System overview

```
                 ┌────────────────────────────── offline ──────────────────────────────┐
 real hand       │  data/            stats/              archetypes/                    │
 histories  ───► │  download+parse ► per-player HUD  ►  cluster into K styles  ──┐      │
 (PHH, IRC)      │  normalize         stats (VPIP…)      (GMM, K by BIC)          │      │
                 └────────────────────────────────────────────────────────────────┼──────┘
                                                                                  │ style profiles
                                                                                  ▼ (means + covariances)
 ┌──────────────────────────────── simulation ───────────────────────────────────────────────┐
 │  opponents/  parametric bots, params sampled per seat from a style's distribution + noise │
 │  engine/     fast NLHE table runner (deterministic, seeded)                               │
 │  baselines/  Nash (Leduc), CFR blueprint (HU NLHE), simple heuristics                     │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
                 ▲                                   ▲
                 │ evaluate                          │ evaluate
 ┌───────────────┴─────────┐           ┌─────────────┴─────────────┐
 │ classifier/ program     │  beliefs  │ exploit/ program          │
 │ actions ► P(style|seat) │ ────────► │ state + beliefs ► action  │
 └───────────────▲─────────┘           └─────────────▲─────────────┘
                 │ rewrite                           │ rewrite
 ┌───────────────┴───────────────────────────────────┴─────────────┐
 │ evolve/  LLM-driven program search (DeepSeek via Luminal)        │
 │ population ► LLM proposes code edit ► sandbox eval ► keep best   │
 │            ► numeric tuner (CMA-ES/Optuna) on exposed PARAMS     │
 └──────────────────────────────────────────────────────────────────┘
```

**Hard rule:** LLMs never run at decision time. The classifier and exploiter are plain Python +
NumPy programs with numeric parameters. The LLM is only an *offline code author* between evaluation
generations, the same idea as FunSearch / AlphaEvolve / Eureka.

## 3. Game format

| Track | Game | Why |
|---|---|---|
| A — proof of concept | Kuhn and Leduc poker | Exact Nash is computable (CFR, minutes). Clean, undeniable result. Fast to iterate the whole pipeline. |
| B — main result | Heads-up NLHE, 100bb, fixed bet abstraction | Equilibrium is well-defined and approximable with MCCFR. One opponent at a time keeps the duplicate comparison clean. |
| C — realistic demo | 6-max NLHE, 100bb | Where real data and real "table classification" live. Multiplayer Nash is not unexploitable in theory, so this track is a demonstration, not the core proof. |

Bet abstraction (B, C): fold, check/call, bet {33%, 75%, 125% pot}, all-in. Opponent bots may use
any size; the engine maps to the nearest abstract size only for the baseline's lookup.

## 4. Components

### 4.1 `engine/` — game rules and table runner

- Own lightweight NLHE engine optimized for simulation throughput (measured: ~3.5k hands/sec/core
  6-max, ~7k heads-up with random bots; scale with `multiprocessing`).
- Real-table showdown rules: the last river bettor shows first (else first live player left of the
  button), later players muck hands that can't win, and every hand is tabled when someone is
  all-in. Mucked cards never enter the event stream, so classifiers only learn from what a real
  opponent would reveal.
- Hand evaluation via a fast C-backed evaluator (`phevaluator` or equivalent).
- [`pokerkit`](https://github.com/uoftcprg/pokerkit) used as a **reference oracle** in tests (our
  engine must agree with it on pot/side-pot/showdown results) and for parsing PHH files.
- Everything takes an explicit `seed`. A hand is fully reproducible from `(deck_seed, policies, seat)`.
- Emits a uniform event stream (`HandStart`, `Action`, `StreetDealt`, `Showdown`, `HandEnd`) that
  both the stats module and classifiers consume. **The same event schema is produced from real
  hand histories**, so a classifier can run on real data unchanged.

### 4.2 `data/` — real hand histories

Candidate sources (verify license/terms before use; record provenance in `data/SOURCES.md`):

- **PHH dataset** (uoftcprg/phh-dataset): hand histories in the open PHH format, including a large
  anonymized online NLHE corpus (HandHQ) plus pro/AI hands.
- **IRC Poker Database** (University of Alberta): older, millions of hands, limit and no-limit.

Pipeline: download → parse into events → normalize stakes to big blinds → write Parquet partitions.
Raw data goes in `data/raw/` (git-ignored); processed in `data/processed/`.

**Open question to resolve early:** whether player identifiers are consistent across hands in each
source. Per-player profiling needs that. If a source hashes names per-hand it is unusable for
clustering (still usable for engine tests).

### 4.3 `stats/` — player features

Computes HUD-style statistics per player, with counts (not only ratios) so we can do Bayesian
shrinkage for small samples:

- Preflop: VPIP, PFR, 3-bet, fold-to-3-bet, 4-bet, steal attempt, fold-to-steal, limp, cold-call.
- Postflop: aggression factor and frequency, c-bet (flop/turn), fold-to-c-bet, check-raise,
  donk-bet, WTSD, W$SD, river bet frequency, bet-size distribution (as fractions of pot).
- Positional splits where sample allows.

Same code runs on real data (offline) and on simulated events (online, inside evaluations).

### 4.4 `archetypes/` — discovering playing styles

- Filter to players with ≥ N hands (start N=500, tune).
- Transform stats (logit for frequencies, log for ratios), standardize.
- Fit Gaussian Mixture Models for K = 3…14; pick K by BIC + stability (bootstrap agreement) +
  interpretability. We expect roughly 6–10 styles; the data decides, not us.
- Output per style: name, centroid, covariance, population share, example players. Saved as
  `archetypes/profiles/<version>.json` and treated as a versioned artifact.

Expected styles (hypotheses to confirm or reject, not hard-coded):

| Style | Signature |
|---|---|
| Nit / Rock | Very low VPIP/PFR, folds to aggression |
| TAG (tight-aggressive) | Solid reg, low VPIP, PFR close to VPIP |
| LAG (loose-aggressive) | High VPIP and PFR, lots of 3-bets and barrels |
| Maniac | Extreme aggression, bluffs far too often |
| Calling station | High VPIP, low PFR, rarely folds post-flop, high WTSD |
| Loose-passive fish | Limps a lot, calls, rarely raises without the nuts |
| Weak-tight | Plays OK hands, gives up to any pressure, low WTSD |
| Fit-or-fold | Normal preflop, c-bets/folds mechanically post-flop |

### 4.5 `opponents/` — synthetic players

A **parametric bot**: a single rule-based policy whose behavior is controlled by a parameter vector θ
(preflop range width and raise share, 3-bet frequency, c-bet frequency, bluff frequency per street,
call-down threshold vs bet size, sizing preferences, trap frequency, etc.).

- **Calibration:** for each style, find θ such that the bot's simulated stats match that style's
  centroid (CMA-ES on stat-matching loss). Also fit how θ covaries so seats can be sampled from
  the style's distribution.
- **Sampling ("semi-randomized"):** each seat draws θ ~ N(θ_style, Σ_style) in transformed space,
  plus per-decision ε-noise. Optional *drift* (θ changes slowly across a session) and *tilt*
  (aggression rises after big losses) to test adaptivity.
- **Adversarial bots:** "style switchers" that change style mid-session, and a mix-of-styles bot,
  to test that the exploiter doesn't overcommit to a wrong read.
- *Stretch:* behavior-cloned bots per cluster from the real data (small networks), as a more
  realistic second opponent family. Evaluating against both guards against "it only beats our
  own rule-based bots".

### 4.6 `baselines/` — the engines we must beat

- **Track A:** exact Nash via CFR+ (OpenSpiel or our own; Leduc is tiny).
- **Track B:** MCCFR blueprint for abstracted HU NLHE (card abstraction via equity buckets, bet
  abstraction from §3). Sanity checks: low local best-response value, positive result vs simple
  heuristics, and optionally a match vs a public HU bot (e.g. Slumbot's API) to place its strength.
- **Heuristic baselines:** "ABC TAG" and a hand-strength bot. Cheap, useful for debugging.

The baseline is **frozen** once chosen; experiments record its version hash.

### 4.7 `classifier/` — evolved program #1

Full design of the editable-program format, inputs/outputs and anti-cheating guards:
[EVOLUTION.md](EVOLUTION.md).

Contract (simplified; every candidate, hand-written or LLM-written, implements this):

```python
PARAMS: dict[str, float]          # numeric knobs exposed to the tuner

class Classifier:
    def __init__(self, n_styles: int, profiles: StyleProfiles, params: dict[str, float]): ...
    def observe(self, event: Event) -> None: ...          # streaming, public events only
    def posterior(self, seat: int, feats: SeatFeatures) -> np.ndarray: ...  # (n_styles,), sums to 1
```

Seed implementation: Bayesian update of per-seat stat counts with Beta priors, Gaussian likelihood
against each style profile → posterior. The LLM's job is to find better feature use, likelihood
models, handling of sparse data, drift detection, etc.

**Fitness** (higher is better), measured on validation seeds:
- Mean log-loss of the posterior at checkpoints {10, 25, 50, 100, 200} hands (weighted toward early).
- Accuracy and calibration (ECE) at each checkpoint.
- **Real-data transfer:** on held-out real players, label = cluster from *all* their hands;
  predict from their *first* N hands. This is the check that keeps the simulator honest.
- Runtime penalty (must stay cheap enough for millions of hands).

### 4.8 `exploit/` — evolved program #2

Designed as **baseline + deviations**, which is how safe exploitation works:

```python
PARAMS: dict[str, float]

class Exploiter:
    def __init__(self, baseline: Policy, profiles: StyleProfiles, params: dict[str, float]): ...
    def policy(self, state, beliefs, feats, baseline: ActionDist) -> ActionDist: ...  # harness samples
```

- Start from the baseline's action distribution at this decision.
- Apply style-conditioned adjustments (e.g. vs calling station: value-bet thinner, cut bluffs; vs
  nit: steal more, fold to its raises; vs maniac: call down lighter, trap more).
- Scale the deviation by **confidence** in the belief: with a flat posterior the agent plays the
  baseline. This bounds how much it can lose to a misread or a switcher.

**Fitness:** Δ bb/100 vs the baseline in duplicate matches against sampled tables, minus penalties
for losses vs (a) the baseline itself, (b) style-switching bots, (c) unseen held-out style variations.
Uses the classifier champion of the current generation (frozen while the exploiter evolves; the
two are co-evolved in alternating rounds once both are stable).

### 4.9 `evolve/` — LLM-driven program search

```
for generation in range(G):
    parents   = select(population)                  # islands + MAP-Elites-style archive
    prompt    = build_prompt(interface, parents' code, scores, failure report, constraints)
    children  = llm.propose(prompt, n=k)            # DeepSeek-V4.1-Flash via Luminal
    for child in children:
        static_check(child)                         # AST: allowed imports only (numpy, math)
        params = tune(child.PARAMS, budget=small)   # CMA-ES/Optuna on numeric knobs
        score  = sandbox_eval(child, params, val_seeds, timeout)
        population.add(child, params, score)
    champion = promote_if_significantly_better(population, champion)
```

- **Division of labor:** the LLM changes *structure* (features, formulas, control flow); the numeric
  tuner sets *numbers*. LLMs are bad at tuning constants, good at ideas.
- **Failure report** given to the LLM: confusion matrix, which styles are confused early, worst
  spots by EV loss, runtime. Concrete feedback makes mutations much better than "improve this".
- **Sandbox:** child runs in a subprocess with CPU/memory/time limits, no network, no file writes,
  import allow-list. Crashing or slow code scores −∞ and the traceback is fed back once for repair.
- **Promotion:** a child replaces the champion only if it beats it on validation seeds with a
  significance test. The **test seeds are never used during evolution** and never appear in prompts.
- **Lineage:** every program is stored in `programs/<kind>/<id>.py` with a JSON sidecar (parent ids,
  prompt hash, model, params, scores, cost). Fully reproducible and auditable.
- **Budget:** token and dollar spend tracked per run; hard cap in config.

### 4.10 `llm/` — Luminal client

OpenAI-compatible API, so we use the `openai` Python SDK with a custom base URL.

```python
from openai import OpenAI
client = OpenAI(base_url=os.environ["LUMINAL_BASE_URL"], api_key=os.environ["LUMINAL_API_KEY"])
client.chat.completions.create(model=os.environ["LUMINAL_MODEL"], messages=[...])
```

| Env var | Value |
|---|---|
| `LUMINAL_BASE_URL` | `https://flash.luminal.cloud/v1` |
| `LUMINAL_MODEL` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| `LUMINAL_API_KEY` | your key (never committed) |

Wrapper adds: retries with backoff, concurrency limit, response caching keyed by prompt hash (so
re-runs are cheap and deterministic), token/cost logging, and code extraction from responses.

### 4.11 `eval/` — measuring results

- Duplicate match runner: same deck seeds, seats rotated, agent vs baseline in the same seat.
- Variance reduction: duplicate format plus all-in EV adjustment (replace all-in runouts with equity).
- Metrics: bb/100 with 95% CI (bootstrap over hand blocks), Δ vs baseline, per-style breakdown,
  classifier accuracy curve vs hands observed.
- Two-stage eval: cheap screen (~10k hands) during evolution, full eval (≥ 1M hands) for champions
  and final results.
- Reports written to `experiments/<run_id>/` (config, seeds, metrics, plots).

## 5. Data splits and seeds

| Split | Used for | Visible to LLM |
|---|---|---|
| train seeds | numeric tuning inside evolution | scores only |
| validation seeds | selection and promotion | scores + failure report |
| test seeds | final result only, once per champion | never |
| held-out real players | classifier transfer metric, final report | never (aggregate score only in val) |

## 6. Known risks

| Risk | Mitigation |
|---|---|
| Simulator doesn't resemble real players → result doesn't transfer | Calibrate bots to real stats; real-data classifier transfer metric; second opponent family (behavior cloning). |
| Baseline too weak, making exploitation look easy | Exact Nash in Track A; blueprint quality checks in Track B; report baseline's own win rate vs the same opponents. |
| Evolution overfits validation seeds | Large seed sets, re-sampled periodically; significance-based promotion; untouched test set. |
| Variance swamps the signal | Duplicate format, all-in EV, large hand counts; fast engine is a priority, not a nice-to-have. |
| LLM-written code is slow, unsafe or broken | Sandbox, timeouts, import allow-list, runtime penalty in fitness. |
| Player IDs not consistent in data | Check in Phase 1 before building on it; fall back to sources where they are. |

## 7. Technology

- Python 3.11, `uv` for env/deps, `ruff` + `pytest`.
- NumPy, pandas/Polars, PyArrow (Parquet), scikit-learn (GMM), `cma` / Optuna (tuning).
- `pokerkit` (parsing + reference oracle), `phevaluator` (hand strength), OpenSpiel optional (Track A).
- `openai` SDK pointed at Luminal.
- Optional later: PyTorch for behavior-cloned opponents.
