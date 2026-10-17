# Poker-Bot: exploiting exploitable players

**Hypothesis:** game-theory-optimal (GTO) poker engines are not the best way to win money against
real, exploitable players. An agent that figures out *how* each opponent plays and adapts to it
should win more, playing the same cards against the same opponents.

This project builds that agent and measures the gap.

## How it works

1. **Learn real playing styles.** Download large datasets of real poker hand histories, compute
   per-player statistics (VPIP, PFR, aggression, c-bet, fold-to-c-bet, showdown rates, …), and
   cluster players into a set of styles (expected ~6–10: nit, TAG, LAG, maniac, calling station, …).
   The data decides how many.
2. **Simulate tables of those players.** Each style becomes a family of bots. Every seat gets its own
   slightly randomized parameters drawn from that style, so no two opponents play exactly alike.
3. **Evolve a classifier.** A numeric program watches opponents' actions and outputs a probability
   for each style. Between evaluation rounds, an LLM (DeepSeek V4.1 Flash via Luminal) rewrites the
   program's code based on how it scored and where it failed; an optimizer tunes its numbers.
4. **Evolve an exploiter.** Same loop, new program: start from an equilibrium (GTO-style) strategy and
   deviate from it based on the classifier's beliefs, more boldly when confident.
5. **Prove it.** Duplicate matches (same cards, same opponents, same seats): exploiter vs the
   equilibrium baseline. Report the difference in big blinds per 100 hands with confidence intervals
   on a held-out test set.

LLMs never make poker decisions. They only write and edit the code between rounds; the programs
that play and classify are plain Python + NumPy.

## Tracks

| Track | Game | Role |
|---|---|---|
| A | Kuhn / Leduc poker | Exact Nash baseline, fastest clean proof |
| B | Heads-up No-Limit Hold'em | Main result vs a CFR blueprint |
| C | 6-max No-Limit Hold'em | Realistic demo matching the real data |

## Status

Phase 0 done (engine, LLM client). Next: Phase 1, real hand-history data.
See [docs/ROADMAP.md](docs/ROADMAP.md) for phases and exit criteria,
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design, and
[docs/EVOLUTION.md](docs/EVOLUTION.md) for how the LLM edits the algorithms.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
cp .env.example .env
```

Then put your Luminal key in `.env` and check it works:

```bash
uv run pokerbot llm ping
```

The LLM endpoint is OpenAI-compatible:

| Setting | Value |
|---|---|
| Base URL | `https://flash.luminal.cloud/v1` |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Auth | your key as the bearer token (`LUMINAL_API_KEY`) |

## Repository layout (planned)

```
src/pokerbot/
  engine/       NLHE rules, table runner, event stream
  data/         download + parse real hand histories
  stats/        per-player statistics
  archetypes/   clustering into playing styles
  opponents/    parametric style bots + randomized seat sampler
  baselines/    CFR / Nash baselines, heuristic bots
  classifier/   classifier interface + seed program
  exploit/      exploiter interface + seed program
  evolve/       LLM-driven program search (prompts, sandbox, population)
  llm/          Luminal client
  eval/         duplicate matches, metrics, significance tests
programs/       every evolved program with lineage metadata
experiments/    run outputs (configs, seeds, metrics, plots)
configs/        experiment configs
data/           raw + processed datasets (git-ignored)
docs/           architecture, roadmap, results
tests/
```

## For AI coding agents

Read [AGENTS.md](AGENTS.md) first.
