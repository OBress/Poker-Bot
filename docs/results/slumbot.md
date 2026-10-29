# External check: the heads-up blueprint and exploiter vs Slumbot

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

The evolved exploiter (frozen Phase 6/7 champion), run `20261029-203943-slumbot-exploiter-hu`,
10,000 hands under the same rules:

| | bb/100 [95% CI] |
|---|---|
| all-in adjusted | **-85.5 +/- 30.8** |
| raw | -75.0 +/- 50.6 |
| as big blind (5,000 hands) | -56.9 |
| as small blind (5,000 hands) | -114.1 |

The blueprint answered with its heuristic fallback in 7 of 21,436 decisions, and our replay of
every hand matched Slumbot's reported winnings.

Reading it: the blueprint loses to Slumbot by a wide margin, and the interval excludes zero. The
blueprint is a 100 bb strategy playing 200 bb, so part of the loss is the depth mismatch, and part
is the coarse abstraction (12 equity buckets, a handful of bet sizes). This run cannot separate
the two. The exploiter loses by a similar amount: -85.5 against -66.8, a difference of about -19 bb/100
with an interval of roughly +/- 47 (the two samples are independent), so the two cannot be told
apart at 10,000 hands each. Neither beats Slumbot. That is what one would expect: the exploiter
earns its edge from opponents with exploitable habits and falls back to the blueprint otherwise,
and Slumbot has no such habits for a classifier built on 2009 human styles to find. The
exploiter's inputs also assume 100 bb, which adds to the mismatch here.

```bash
uv run pokerbot eval slumbot --hero exploiter-hu:champion --hands 10000
```
