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

## Phase 4 — Baselines ✅

Done. Every number below is a duplicate measurement (same cards, seats and opponents for each
policy), all-in runouts replaced by their expectation, 95% CI from a bootstrap over sessions.

**Track A** (`profiles/baseline-{kuhn,leduc}-v1.json`, CFR+). Leduc: exploitability 4.0e-5
chips/hand after 5,000 iterations, seat-0 value −0.0856 (matches the published value), hash
2c9e232f2e166454, run 20261020-191123-baseline-leduc. Kuhn: 3.9e-5, value −1/18, run
20261020-191111-baseline-kuhn. Values vs five hand-defined styles are exact (card luck integrated
out; CI over 200 sampled players per style), chips/hand:

| Leduc style | Nash baseline | Best response (exploitation ceiling) |
|---|---|---|
| nit | +0.107 ± 0.002 | +0.968 ± 0.017 |
| calling station | +0.486 ± 0.006 | +1.188 ± 0.009 |
| maniac | +0.348 ± 0.004 | +1.509 ± 0.014 |
| fit-or-fold | +0.235 ± 0.004 | +0.728 ± 0.009 |
| bluffer | +0.213 ± 0.003 | +1.346 ± 0.027 |

**Track B** (`profiles/blueprint-hu-v1.npz`): external-sampling MCCFR on an abstract heads-up game
(bets 33/75/125% pot or all-in, raises 75% or all-in; 169 preflop classes, 12 equity buckets per
later street; 259k information sets), 4.0M iterations, hash 6bb020139df832d9, runs
20261020-192609-blueprint-hu (training, stopped at round 100 when the regret proxy had been flat
since round 60) and 20261020-215732-blueprint-hu (final export). Sanity checks, head-to-head over
100k hands (run 20261020-220236-headsup): vs hand-strength +14.2 bb/100 [+7.2, +21.0]; vs ABC TAG
+2.0 [−4.6, +8.8] (not significant). No local best-response check yet.

**Win rates vs every style** (bb/100 [95% CI]; Track B heads-up 250k hands per policy, run
20261020-215752-matchups-hu; Track C short-handed 405k, run 20261020-192844-matchups-sh;
full ring 210k, run 20261020-200122-matchups-fr; opponents `profiles/bots-v3.json`):

| Format / style | Blueprint (hu) | ABC TAG | Hand-strength |
|---|---|---|---|
| hu HU hyper-aggressor | +31.8 [+20.8, +42.0] | −2.3 [−8.9, +4.1] | +9.5 [+0.7, +18.4] |
| hu HU limper | +36.9 [+27.8, +46.0] | +29.1 [+21.8, +36.1] | +45.8 [+34.7, +57.2] |
| hu HU semi-passive | +25.2 [+16.1, +34.1] | +13.1 [+7.4, +18.5] | +9.2 [+1.3, +17.4] |
| hu HU regular | +29.2 [+19.6, +39.3] | −1.2 [−7.2, +4.9] | +0.4 [−6.3, +6.7] |
| hu mixed | **+21.2 [+10.8, +32.6]** | +3.8 [−2.4, +10.2] | +8.2 [+0.4, +16.0] |
| sh Loose-aggressive fish | | +21.8 [+13.0, +30.7] | +31.7 [+20.2, +43.0] |
| sh Passive limper | | +23.2 [+15.5, +30.7] | −14.2 [−26.2, −2.9] |
| sh Loose-passive | | +30.5 [+22.3, +39.5] | +26.1 [+15.7, +36.4] |
| sh Position-blind aggressor | | +21.3 [+14.1, +28.7] | +23.9 [+14.4, +33.5] |
| sh Semi-loose passive | | +13.6 [+6.6, +20.6] | −2.5 [−11.7, +6.9] |
| sh TAG | | +16.4 [+9.5, +23.8] | −5.0 [−16.6, +6.7] |
| sh Tight-solid (ABC) | | +6.9 [+0.0, +13.9] | −17.8 [−28.0, −7.5] |
| sh Short-stacker | | +4.8 [+0.8, +8.8] | +18.3 [+11.9, +24.5] |
| sh mixed | | **+18.0 [+9.9, +26.2]** | +6.8 [−5.6, +18.3] |
| fr Loose-passive | | +27.1 [+18.2, +36.8] | +30.6 [+17.0, +45.1] |
| fr Passive limper | | +19.1 [+10.0, +27.3] | +17.8 [+4.1, +31.4] |
| fr Weak-tight | | +26.1 [+17.5, +34.1] | +24.7 [+9.5, +38.4] |
| fr TAG | | +18.3 [+9.2, +27.4] | +16.5 [+5.8, +26.7] |
| fr Nit | | +4.6 [−1.5, +11.1] | −14.9 [−22.9, −7.2] |
| fr Short-stacker | | +1.7 [−0.4, +3.8] | −2.8 [−6.8, +1.4] |
| fr mixed | | **+13.9 [+5.3, +21.6]** | +20.5 [+8.1, +32.2] |

The numbers to beat: the blueprint heads-up (Track B) and the Nash strategy in Leduc (Track A).
For 6-max there is no equilibrium baseline; ABC TAG is the reference (Track C is a demonstration).

- Track A: CFR+ for Kuhn and Leduc (verified exploitability ≈ 0).
- Track B: MCCFR blueprint for abstracted HU NLHE; quality checks.
- Heuristic baselines (ABC TAG, hand-strength).
- Duplicate match runner and metrics in `eval/`.

**Exit:** baseline win rates vs every style recorded with CIs. This is the number to beat.

## Phase 5 — Classifier evolution ✅ on Track C (Track B heads-up not run yet)

Done: frozen harness (`evolve/`: API, redaction, feature tracker, AST allow-list sandbox with
timeouts, CMA-ES tuner, prompts with failure reports, two-batch promotion test), two seeds
(`programs/classifier/seed_a.py` Beta-Binomial naive Bayes, `seed_b.py` correlated Gaussian, both
with the showdown term), datasets of 240 seeded short-handed sessions per split (1,440 seats each;
runs 20261020-235922-clf-data-sh for train/val and 20261021-003649-clf-data-sh for test), and 300 fit-set + 300
held-out real players' first 200 hands (run 20261021-011630-clf-real-sh).

First evolution (run 20261021-002216-evolve-classifier-sh; DeepSeek V4.1 Flash, 10 generations
× 8 proposals, 96 calls, 0.50M prompt + 1.72M completion tokens): one promotion, g03-1 (p 1.5e-6
on val_a, 0.002 on val_b), which weights each stat's evidence by how well it separates the styles
and mixes the Beta-Binomial and Gaussian likelihoods; no later candidate passed. Champion:
`programs/classifier/champion-20261021-002216-evolve-classifier-sh.py`.

Final evaluation (run 20261021-011945-eval-classifier-sh; test seeds and held-out players were
used here only):

| Hands | Test accuracy, champion | Test accuracy, seed A (tuned) | Held-out real, champion acc / H(q,p) | Held-out real, seed A acc / H(q,p) |
|---|---|---|---|---|
| 10 | 0.358 [0.334, 0.384] | 0.369 [0.344, 0.394] | 0.350 / 1.660 | 0.367 / 1.634 |
| 25 | 0.447 [0.422, 0.473] | 0.448 [0.423, 0.474] | 0.413 / 1.433 | 0.423 / 1.418 |
| 50 | 0.523 [0.498, 0.549] | 0.510 [0.486, 0.537] | 0.500 / 1.282 | 0.503 / 1.288 |
| 100 | 0.613 [0.588, 0.638] | 0.599 [0.576, 0.626] | 0.580 / 1.081 | 0.563 / 1.114 |
| 200 | 0.662 [0.638, 0.688] | 0.642 [0.618, 0.667] | 0.620 / 0.963 | 0.640 / 1.010 |

The champion's weighted log loss (the fitness, MATH §6.3) beats the seed on test seeds by 0.0147
[0.0023, 0.0266] per session, one-sided paired p = 0.008: significant, but small. The gain is at
50+ hands; at 10–25 hands it is slightly worse. Real-player transfer is about as good as on
simulated tables (62–64% at 200 hands vs 64–66%), so the bots are a fair stand-in; on held-out
players the champion has lower cross-entropy from 100 hands on but not higher accuracy.

Second run (run 20261021-064929-evolve-classifier-sh; started from the first champion and both
seeds, 25 generations × 10 proposals, 291 calls, 2.83M prompt + 6.06M completion tokens, two
instructions added for the first 10–25 hands): four promotions, each passing both validation
batches (g02-2, g07-0, g17-4, g22-5; val_a fitness −1.345 → −1.248). The champion
`programs/classifier/champion-20261021-064929-evolve-classifier-sh.py` discounts every group of
stats that move together (VPIP by position and depth, VPIP and PFR, flop aggression and c-bets)
by their correlation instead of counting the same evidence several times.

Final evaluation of the second champion (test seeds and held-out players, used here only):

| Hands | Test accuracy | vs tuned seed A | vs first champion | Held-out real acc / H(q,p) |
|---|---|---|---|---|
| 10 | 0.394 [0.370, 0.420] | +0.026 | +0.036 | 0.383 / 1.642 |
| 25 | 0.476 [0.449, 0.502] | +0.028 | +0.029 | 0.413 / 1.431 |
| 50 | 0.542 [0.515, 0.568] | +0.031 | +0.019 | 0.497 / 1.275 |
| 100 | 0.613 [0.589, 0.640] | +0.014 | +0.001 | 0.557 / 1.057 |
| 200 | 0.678 [0.654, 0.703] | +0.035 | +0.015 | 0.640 / 0.926 |

Weighted log loss per session, paired over the 240 test sessions: vs the tuned seed +0.085
[+0.063, +0.106], p = 9e-14 (run 20261021-194600-eval-classifier-sh); vs the first champion
+0.070 [+0.052, +0.089], p = 3e-13 (run 20261021-194435-eval-classifier-sh). Exit met: the
accuracy-vs-hands curve is above the seed's at every checkpoint and the log loss gain is highly
significant. On held-out real players the cross-entropy is the best of the three programs from
50 hands on (0.926 vs 1.010 for the seed at 200 hands); accuracy is about the same (64% at 200).
Next: Track B heads-up classifier; the classifier is ready to feed the Phase 6 exploiter.

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
