# Experiment tracking

Every job that trains, tunes, evaluates or builds data is a **tracked run** (`pokerbot.track.Run`).
No untracked training. If it isn't in `experiments/`, it didn't happen.

## What a run records automatically

| File | Contents |
|---|---|
| `run.json` | name, kind, status (running / completed / failed / interrupted), start, end, duration; config; params; seeds; git commit, branch, dirty flag; Python, OS, CPU count, command line, package versions; fingerprints of every input data file; final summary |
| `git.diff` | uncommitted changes at start, so even a dirty run can be reproduced |
| `metrics.jsonl` | every measurement: time, step, name, value, tags (split, style, candidate, …) |
| `events.jsonl` | run start/finish, errors with tracebacks, candidates created/gated/scored/promoted, notes |
| `llm.jsonl` | every LLM call: purpose, full prompt, full response, tokens, latency, cache hit |
| `candidates/<id>/` | each evolved program's source and metadata: parents, diff, prompt reference, tuned params, gate results, scores per split |
| `artifacts/` | configs, profiles, reports, plots |

Files are append-only JSON Lines, flushed per line: a crash loses nothing logged before it, and the
crash itself is recorded as an `error` event with the run marked `failed`.

## Commands

```bash
uv run pokerbot runs list
```

```bash
uv run pokerbot runs show latest
```

```bash
uv run pokerbot runs plot latest --metric logloss
```

## Metric names

`<split>/<metric>[@<checkpoint>]`, with optional tags. Splits: `train`, `val`, `test`, `real`
(held-out real players), `gen` (per-generation aggregates), `llm`, `sys`.

Examples: `val/logloss@25`, `val/brier@100`, `val/ece@100`, `real/xent@50`, `val/delta_bb100`,
`val/delta_bb100_ci`, `gen/best_fitness`, `gen/train_val_gap`, `llm/prompt_tokens`, `sys/hands_per_sec`.

`test/*` metrics are written only by final evaluations, never during evolution (AGENTS.md rule 2).

## Required metrics by run kind

| Kind | Must log |
|---|---|
| `data` | hands processed, rejected, skipped; players; duration |
| `cluster` (Phase 2) | per K: log-likelihood, BIC, bootstrap ARI, smallest share; chosen K; per-style centroid and share (artifact) |
| `calibrate` (Phase 3) | per style, per optimizer iteration: stat-matching loss; final simulated vs real stats per style |
| `baseline` (Phase 4) | per CFR iteration: exploitability (Leduc) or proxy; win rate vs each style with CI |
| `evolve-classifier` | per candidate: `train/` and `val/` logloss, Brier, accuracy at 10/25/50/100/200 hands, ECE, `real/xent@t`, µs per call, fitness. Per generation: best/median/worst fitness, champion fitness, **train−val gap** (overfitting watch), candidates proposed / failed per gate reason / scored / promoted, archive size, LLM tokens, cost, latency, cache hit rate. Promotion events with p-values. |
| `evolve-exploiter` | per candidate: Δ bb/100 and CI overall and per style, vs baseline, vs switchers, mean KL from baseline, fitness. Same per-generation aggregates as above. |
| `eval` | the final pre-registered metrics on `test/`, with hand counts and CIs |

Loss curves to watch during evolution: `val/logloss@*` (classifier) and `val/delta_bb100`
(exploiter) per generation for the champion; `gen/train_val_gap`, which should stay near zero.
A growing gap means the loop is overfitting its seeds.

## Where records live

`experiments/` is git-ignored: raw runs can be large (full prompts, every candidate). Results that
matter are summarized into `docs/RESULTS.md` (committed) with the run ids they came from, so every
reported number traces back to a complete record.
