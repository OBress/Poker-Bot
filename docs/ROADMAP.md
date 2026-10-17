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

## Phase 3 — Synthetic opponents (working version; criterion not fully met)

Done: parametric `StyleBot` (24 parameters), mean-field CMA-ES calibration per style, measured
per-style spread, seat sampler, validation (`profiles/bots-v2.json`). Validation of ~72 randomized
bots per short-handed style: 110/144 style-stat pairs (76%) within one real-player σ; most of the
rest within 2σ. Known misses: short-stacker turn folding (z ≈ +4.8), passive limper's flop
c-bet defense (z ≈ −3.0), TAG limping (z ≈ +2.0). Heads-up and full-ring styles not calibrated yet.

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
