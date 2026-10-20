# Roadmap

Each phase ends with a concrete, checkable deliverable. Don't start a phase until the previous one's
exit criteria pass. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design behind each piece.

## Phase 0 — Foundations ✅

Done: engine matches pokerkit on 10,000 random 2–6 player hands (payouts, side pots, odd chips);
rule tests for showdown order, mucks, short all-ins, min-raises and information hiding;
~3.5k hands/sec/core 6-max; `pokerbot llm ping` answers via Luminal.

- Repo scaffolding: `pyproject.toml` (uv), `src/pokerbot/` package, ruff, pytest, CI-able `make`/scripts.
- `llm/` client for Luminal with retries, caching, cost logging; smoke test that DeepSeek answers.
- `engine/`: NLHE rules, side pots, showdown, event stream, seeded decks.
- Engine cross-checked against `pokerkit` on randomized hands.

**Exit:** 10k random hands agree with pokerkit on every payout; throughput benchmark recorded;
`pokerbot llm ping` returns a completion.

## Phase 1 — Real data ✅

Done: all 21.6M HandHQ hands downloaded and converted to Parquet; player IDs verified consistent;
34,301 players with ≥ 500 hands (see `data/SOURCES.md`). Real hands replay through pokerkit into
engine events; per-(player, table format) stats computed with position, stack-depth and pot-type
splits for 420,914 units. Deferred: shown-hand strength stats (river bluff rate, showdown evidence).

- Download PHH dataset (and IRC DB if useful); document provenance and terms in `data/SOURCES.md`.
- Parser → unified event schema → Parquet.
- Verify player identifiers are consistent across hands.
- `stats/`: per-player HUD stats with raw counts.

**Exit:** a table of ≥ 5k players with ≥ 500 hands each and their stats; spot-checked by hand on a
few players.

## Phase 2 — Archetypes ✅

Done (`profiles/styles-v1.json`, report in [STYLES.md](STYLES.md)): 8 short-handed, 6 full-ring
and 4 heads-up styles, chosen by BIC with a bootstrap-stability floor; 20% of players held out.
Finding: styles differ mostly preflop; postflop tendencies vary much less across the 2009 pool.

- GMM clustering over transformed stats, K from 3 to 14; choose K (BIC, stability, readability).
- Name and describe each style; write versioned `profiles/<version>.json`.
- Hold out 20% of players for the classifier transfer metric.

**Exit:** a short report: chosen K, style table (centroids, shares), stability score, plots.

## Phase 3 — Synthetic opponents (usable; ≥ 90% of style-stat pairs within 1σ in sh and fr)

Current: `profiles/bots-v3.json` (bot structure 3, 27 parameters) for all three formats.
Validation of ~72 randomized bots per style (`pokerbot bots validate`):

| Format | Stat pairs within 1σ | within 2σ | Shown equity gap per (style, line) | Run |
|---|---|---|---|---|
| sh | 132/144 (92%) | 143/144 | 0.045 (39/61 cells ≤ 0.05) | 20261020-014703-bots-validate-bots-v3-sh |
| hu | 59/68 (87%) | 68/68 | 0.064 (15/29) | 20261020-014741-bots-validate-bots-v3-hu |
| fr | 105/108 (97%) | 108/108 | 0.064 (23/42) | 20261020-014811-bots-validate-bots-v3-fr |
| sh, bots-v2 | 123/144 (85%) | 141/144 | 0.087 (21/63) | 20261019-010202-bots-validate-bots-v2-sh |

Only miss beyond 2σ: sh passive limper WTSD (z = −2.02). hu spread scales sit at the top of the
tuning grid (median spread ratio 0.6–1.0), so hu bots vary a bit less than real players.
Calibration runs: sh 20261019-235931-bots-v3-sh-r4 (damped refinement after three earlier
v3 passes), hu/fr 20261019-021540-bots-v3; spread runs are in the file (`spread_run_id`).

What changed from v2 and why:
- **Yardstick.** v2 compared per-bot rates smoothed toward 50% with pooled real rates, which
  overstated misses on rare stats (the v2 short-stacker turn-fold z ≈ +4.8 was mostly this).
  Bots are now measured like real players: counts through the profile's feature transform, hand
  counts drawn from real profiled players, compared with the style mixture's mean and spread.
  The old number is still reported as `legacy`. Calibration uses the same measurement.
- **Showdown ranges** (MATH.md §4.5). Real per-style tables of shown strength bucket by line
  class (8 classes, e.g. bet-bet-bet, call-down) from every HandHQ showdown with shown cards:
  2.59M hands replayed, 2.07M shown hands of fit-set players weighted by soft membership,
  held-out players excluded (run 20261019-001623-showdown-ranges, `bots showdown-real`).
  v2 bots bet and got all-in far too light and checked down hands real players bet. Calibration
  now also matches mean shown equity per line group (0.04 equity ≈ 1σ).
- **Bot structure 3** (v1/v2 files keep structure 2 behaviour and their own parameter bounds):
  noise can no longer limp (caused the TAG and short-stacker limp misses) and is capped at 0.05
  (it sat at 0.2, one decision in five random); `bluff_eq` (only weak hands bluff, medium hands
  check and never bet the river); `respect` (folds more to c-bets than to stabs); `commit`
  (short stacks call on pot odds).
- **Fixes:** tune-spread overwrote `x_scale` with a relative factor (spreads 1.3–1.6× too wide);
  warm starts left parameters pinned at their bounds (three_bet stuck at 0); mean-field rounds
  oscillated, so `--damping` moves style centers part way per round.

Still imperfect: the mix of lines bots take to showdown differs from real players (line-mix
total variation 0.24–0.33); shown-equity per line is closer but not within sampling error.
Not built yet: drift, tilt and style-switching bots (the last bullet below).

- Parametric bot policy with parameter vector θ.
- Calibrate θ per style (CMA-ES on stat-matching loss) and fit per-style θ covariance.
- Seat sampler with noise, drift, tilt; style-switcher adversaries.

**Exit:** for every style, the simulated stats of sampled bots fall within the real style's spread
on all key stats.

## Phase 4 — Baselines

- Track A: CFR+ for Kuhn and Leduc (verified exploitability ≈ 0).
- Track B: MCCFR blueprint for abstracted HU NLHE; quality checks.
- Heuristic baselines (ABC TAG, hand-strength).
- Duplicate match runner and metrics in `eval/`.

**Exit:** baseline win rates vs every style recorded with CIs. This is the number to beat.

## Phase 5 — Classifier evolution

- Hand-written seed classifier (Bayesian stat counts vs style Gaussians).
- `evolve/` loop: prompt builder, sandbox, numeric tuner, population store, promotion test.
- Run evolution on Track C tables (and Track B heads-up).

**Exit:** champion classifier's accuracy-vs-hands curve beats the seed with significance on test
seeds, and real-data transfer accuracy is reported.

## Phase 6 — Exploit evolution

- Hand-written seed exploiter (baseline + simple per-style adjustments, confidence-scaled).
- Same evolution loop with exploit fitness (Δ bb/100 minus robustness penalties).
- Alternate classifier/exploiter rounds once both are stable.
- Track A first (fast, exact baseline), then Track B, then Track C.

**Exit:** Δ bb/100 > 0 with 95% CI excluding zero on test seeds in Track A and B; loss vs the
baseline itself and vs switchers within a pre-set tolerance.

## Phase 7 — Final experiment and write-up

- Pre-register: seeds, hand counts, metrics, tolerance.
- Run full evaluations; per-style breakdown; ablations (no classifier / oracle labels / no
  confidence scaling / seed vs evolved programs).
- Write `docs/RESULTS.md` with plots and the full lineage of champion programs.

**Exit:** a result someone else can re-run from a single config and seed.

## Later / stretch

- Behavior-cloned opponents from real data as a second opponent family.
- Run the classifier on real hand histories as a live "HUD that labels players".
- Online adaptation (exploiter updates mid-session beyond posterior updates).
