# External check: the heads-up blueprint vs Slumbot

Run `20261029-173029-slumbot-blueprint`: 10,000 hands against Slumbot (2018 ACPC winner) over its
public API, blinds 50/100, 200 bb stacks reset every hand, positions alternating. Hero: the frozen
100 bb blueprint `profiles/blueprint-hu-v1.npz`, used as is (no retraining, no depth translation).
Post-hoc external check, not pre-registered. Slumbot deals its own cards, so there is no duplicate
pairing; all-in runouts are replaced by their expectation when Slumbot's cards are shown.

| | bb/100 [95% CI, bootstrap over blocks of 100 hands] |
|---|---|
| all-in adjusted | **-66.8 +/- 36.3** |
| raw | -63.4 +/- 47.3 |
| as big blind (5,000 hands) | -96.9 |
| as small blind (5,000 hands) | -36.7 |

The blueprint answered with its heuristic fallback in 7 of 21,436 decisions, and our replay of
every hand matched Slumbot's reported winnings.

Reading it: the blueprint loses to Slumbot by a wide margin, and the interval excludes zero. The
blueprint is a 100 bb strategy playing 200 bb, so part of the loss is the depth mismatch, and part
is the coarse abstraction (12 equity buckets, a handful of bet sizes). This run cannot separate
the two. The evolved exploiter has not been run against Slumbot yet.

```bash
uv run pokerbot eval slumbot --hero exploiter-hu:champion --hands 10000
```
