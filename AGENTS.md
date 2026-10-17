# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, etc.) and humans working in this repo.

## What this project is

A research codebase testing whether a style-classifying, exploiting poker agent beats an equilibrium
(GTO-style) baseline against exploitable opponents. Read [README.md](README.md), then
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md) and
[docs/EVOLUTION.md](docs/EVOLUTION.md) before changing anything structural, and
[docs/MATH.md](docs/MATH.md) before touching stats, classifiers, exploiters or scoring. The scoring
rules in MATH.md §6–§7 are fixed; change them only deliberately and record why. Work phase by phase;
check the current phase's exit criteria.

## Non-negotiable rules

1. **No LLM calls at decision time.** Classifier and exploiter programs are pure Python + NumPy.
   LLMs are used only inside `evolve/` to write or edit code between evaluations.
2. **Never leak test data.** Test seeds and held-out real players are never used during evolution,
   tuning or prompt construction, and never shown to the LLM. Only `eval/` final runs touch them.
3. **Everything is seeded and reproducible.** Every function that uses randomness takes a `seed` or a
   `np.random.Generator`. No global `random`/`np.random` state. Experiments save their full config.
4. **Freeze what you compare against.** Baselines and style profiles are versioned artifacts. Don't
   edit a frozen version; create a new one and record the version in experiment metadata.
5. **Secrets stay out of git.** The Luminal key lives in `.env` (`LUMINAL_API_KEY`). Never print,
   log, or commit it. `.env.example` holds placeholders only.
6. **Data stays out of git.** `data/raw/` and `data/processed/` are git-ignored. Document every
   source, its URL and terms in `data/SOURCES.md`.
7. **Track every run.** Any job that trains, tunes, evaluates or builds data runs inside
   `pokerbot.track.Run` and logs the metrics required for its kind in
   [docs/TRACKING.md](docs/TRACKING.md). LLM clients used by a run get `run=` so every prompt and
   response is recorded. Reported numbers must cite the run id they came from.
8. **Statistical honesty.** Every reported win rate has a confidence interval and a hand count. Do
   not claim an improvement that isn't significant on the test set.

## Program contracts (evolved code)

Evolved classifiers and exploiters must implement the interfaces in
`src/pokerbot/classifier/base.py` and `src/pokerbot/exploit/base.py` (see ARCHITECTURE §4.7–4.8):

- Expose numeric knobs in a module-level `PARAMS: dict[str, float]`.
- Allowed imports: `numpy`, `math`, and the project's read-only types module. Nothing else.
- No I/O, no network, no global mutable state, no randomness without the provided generator.
- Must run within the per-call time budget set in the evolve config.

When changing an interface, update the seed programs, the sandbox allow-list, the prompt templates in
`evolve/prompts/`, and this file in the same change.

## Conventions

- Python 3.11, `uv` for dependencies (`uv add <pkg>`), source under `src/pokerbot/`.
- Type hints everywhere; `ruff` for lint/format; `pytest` for tests.
- Units: chip amounts are integers in the engine; stats and results are reported in big blinds.
- Seats are 0-indexed integers; the button position is explicit in every hand state.
- Hot paths (engine, stats updates, classifier `observe`) must stay fast. Benchmark before and after
  changes there.
- Configs in `configs/*.yaml`; no magic constants buried in code for anything experiment-related.

## Commands

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

```bash
uv run pokerbot bench
```

```bash
uv run pokerbot llm ping
```

```bash
uv run pokerbot data stats
```

## Testing expectations

- Engine changes: run the pokerkit cross-check tests (payouts, side pots, showdowns) and
  `tests/test_engine_rules.py`. Engine rules are documented at the top of `engine/state.py`;
  pokerkit asks for some no-op checks our engine skips (see the test harness), nothing else differs.
- Information hiding is a correctness property: anything a policy or observer receives must be
  public or its own. Add a test whenever you add a new event or field.
- Stats changes: unit tests on hand-built histories with known answers, plus the round-trip test
  in `tests/test_replay_stats.py` (engine hand -> PHH -> replay must give identical stats).
- Evolve changes: test the sandbox rejects disallowed imports, infinite loops and crashes.
- Don't hit the real Luminal API in unit tests; use the cached/mocked client.

## LLM usage (Luminal)

- OpenAI-compatible: `LUMINAL_BASE_URL=https://flash.luminal.cloud/v1`,
  `LUMINAL_MODEL=deepseek-ai/DeepSeek-V4.1-Flash`, key as bearer token.
- Go through `src/pokerbot/llm/` only; it handles retries, caching, and cost logging.
- Respect the run budget cap in config. Log tokens and cost for every evolution run.

## Git

- One coherent change per commit, imperative subject ("Add side-pot handling to engine").
- Don't commit generated data, large experiment outputs, or `.env`.
