# Playing styles (v1)

Generated from `profiles/styles-v1.json` (run `20261017-041248-styles-v1`; source: HandHQ 2009 (data/SOURCES.md)).
Rates are pooled over each style's members (weighted by membership probability).
Regenerate with `uv run pokerbot styles report`.

## Short-handed (3-6 players): 8 styles

16,218 player units fitted. K = 8 chosen: BIC 583,266, bootstrap ARI 0.56.

| # | Style | Share | VPIP | PFR | Limp | Open EP | Open BTN | 3-bet | Fold to 3-bet | Fold to steal | C-bet | Fold to c-bet | Flop agg | WTSD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **Loose-aggressive fish** | 6.5% | 48% | 18% | 28% | 18% | 32% | 5% | 32% | 50% | 65% | 45% | 35% | 30% |
| 1 | **Passive limper** | 5.4% | 37% | 4% | 39% | 2% | 6% | 2% | 28% | 63% | 61% | 50% | 23% | 31% |
| 2 | **Loose-passive** | 14.4% | 34% | 9% | 28% | 8% | 16% | 3% | 31% | 66% | 64% | 52% | 28% | 30% |
| 3 | **Position-blind aggressor** | 10.2% | 30% | 19% | 5% | 21% | 36% | 5% | 43% | 67% | 59% | 48% | 31% | 31% |
| 4 | **Semi-loose passive** | 20.3% | 28% | 13% | 13% | 12% | 28% | 4% | 44% | 71% | 61% | 52% | 29% | 29% |
| 5 | **TAG** | 20.6% | 22% | 18% | 1% | 16% | 40% | 7% | 59% | 80% | 67% | 51% | 39% | 27% |
| 6 | **Tight-solid (ABC)** | 18.9% | 21% | 15% | 3% | 14% | 33% | 5% | 54% | 81% | 66% | 54% | 34% | 27% |
| 7 | **Short-stacker** | 3.8% | 20% | 19% | 1% | 12% | 36% | 10% | 63% | 81% | 61% | 52% | 41% | 43% |

- **Loose-aggressive fish**: Plays half its hands (VPIP 48%), limps 28% yet still raises 18%, and opens almost as wide from early position (18%) as the button (32%).
- **Passive limper**: Limps into ~4 in 10 pots and almost never raises (PFR 4%); a classic recreational caller.
- **Loose-passive**: Plays a third of hands, mostly by limping or calling; raises rarely.
- **Position-blind aggressor**: Raises a lot but opens nearly as wide early as late (21% vs 36%): aggressive without positional discipline.
- **Semi-loose passive**: A bit too loose and still limps 13% of the time; middle of the pool.
- **TAG**: Solid regular: tight, raises first in, widens strongly on the button (16% early vs 40% button), 3-bets 7%.
- **Tight-solid (ABC)**: Tight and straightforward; slightly less aggressive than the TAG and folds to c-bets the most.
- **Short-stacker**: Push/fold short-stack strategy: 69% of hands under 25bb, raises nearly every hand it plays, 3-bets 10%, goes to showdown 43%.

<details><summary>Model selection</summary>

| K | BIC | ARI | Smallest share |
|---|---|---|---|
| 3 | 596,908 | 0.906 | 21.3% |
| 4 | 591,038 | 0.758 | 10.6% |
| 5 | 586,593 | 0.732 | 7.6% |
| 6 | 584,625 | 0.782 | 6.0% |
| 7 | 583,959 | 0.663 | 4.6% |
| 8 | 583,266 | 0.561 | 3.8% |
| 9 | 583,504 | 0.447 | 3.7% |
| 10 | 583,335 | 0.512 | 2.8% |
| 11 | 584,174 | 0.441 | 2.5% |
| 12 | 584,827 | 0.495 | 2.6% |
| 13 | 585,611 | 0.411 | 2.3% |
| 14 | 586,025 | 0.386 | 2.3% |

</details>

## Full ring (7-10): 6 styles

9,338 player units fitted. K = 6 chosen: BIC 355,063, bootstrap ARI 0.73.

| # | Style | Share | VPIP | PFR | Limp | Open EP | Open BTN | 3-bet | Fold to 3-bet | Fold to steal | C-bet | Fold to c-bet | Flop agg | WTSD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **Loose-passive** | 8.2% | 31% | 12% | 19% | 14% | 27% | 3% | 33% | 68% | 66% | 53% | 34% | 30% |
| 1 | **Passive limper** | 10.8% | 24% | 3% | 25% | 2% | 8% | 2% | 25% | 78% | 69% | 55% | 23% | 29% |
| 2 | **Weak-tight** | 28.6% | 20% | 6% | 16% | 6% | 20% | 2% | 39% | 80% | 67% | 56% | 27% | 27% |
| 3 | **TAG** | 18.1% | 14% | 11% | 2% | 10% | 35% | 4% | 58% | 84% | 68% | 57% | 35% | 27% |
| 4 | **Nit** | 24.0% | 14% | 8% | 6% | 7% | 25% | 3% | 46% | 84% | 67% | 57% | 30% | 28% |
| 5 | **Short-stacker** | 10.4% | 9% | 9% | 1% | 6% | 26% | 5% | 50% | 87% | 70% | 51% | 33% | 41% |

- **Loose-passive**: Loosest full-ring group; limps often and raises little.
- **Passive limper**: Almost never raises (PFR 3%); limps a quarter of hands.
- **Weak-tight**: Plays few hands and mostly passively; folds to pressure.
- **TAG**: Tight-aggressive regular with real positional awareness (10% early vs 35% button).
- **Nit**: Very tight and fairly passive preflop.
- **Short-stacker**: Push/fold short stack: 87% of hands under 25bb, VPIP equals PFR.

<details><summary>Model selection</summary>

| K | BIC | ARI | Smallest share |
|---|---|---|---|
| 3 | 359,680 | 0.923 | 19.6% |
| 4 | 356,514 | 0.886 | 14.6% |
| 5 | 355,069 | 0.554 | 10.3% |
| 6 | 355,063 | 0.728 | 8.2% |
| 7 | 355,154 | 0.604 | 6.3% |
| 8 | 355,427 | 0.490 | 5.3% |
| 9 | 356,021 | 0.421 | 3.7% |
| 10 | 356,202 | 0.336 | 3.8% |
| 11 | 357,414 | 0.356 | 3.5% |
| 12 | 358,069 | 0.313 | 2.8% |
| 13 | 359,439 | 0.319 | 2.6% |
| 14 | 359,980 | 0.313 | 1.9% |

</details>

## Heads-up: 4 styles

1,952 player units fitted. K = 4 chosen: BIC 68,661, bootstrap ARI 0.58.

| # | Style | Share | VPIP | PFR | Limp | Open BTN | 3-bet | Fold to 3-bet | Fold to steal | C-bet | Fold to c-bet | Flop agg | WTSD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **HU hyper-aggressor** | 6.1% | 69% | 59% | 1% | 86% | 19% | 55% | 55% | 72% | 41% | 43% | 30% |
| 1 | **HU limper** | 4.0% | 67% | 12% | 68% | 14% | 8% | 39% | 34% | 57% | 41% | 33% | 26% |
| 2 | **HU semi-passive** | 31.2% | 66% | 34% | 28% | 47% | 13% | 36% | 40% | 58% | 46% | 31% | 32% |
| 3 | **HU regular** | 58.7% | 56% | 46% | 1% | 66% | 16% | 45% | 56% | 64% | 46% | 37% | 31% |

- **HU hyper-aggressor**: Opens 86% of buttons and 3-bets 19%: maximum pressure.
- **HU limper**: Limps the button 68% of the time instead of raising.
- **HU semi-passive**: Mixes limps (28%) and raises; plays many hands but without much pressure.
- **HU regular**: Standard heads-up regular: opens 66% of buttons, rarely limps, 3-bets 16%.

<details><summary>Model selection</summary>

| K | BIC | ARI | Smallest share |
|---|---|---|---|
| 3 | 69,345 | 0.504 | 23.0% |
| 4 | 68,661 | 0.582 | 4.0% |
| 5 | 69,029 | 0.639 | 5.9% |
| 6 | 68,852 | 0.579 | 4.0% |
| 7 | 69,501 | 0.599 | 0.2% |
| 8 | 70,403 | 0.439 | 1.6% |
| 9 | 71,057 | 0.328 | 0.2% |
| 10 | 71,902 | 0.362 | 0.2% |
| 11 | 72,880 | 0.338 | 0.2% |
| 12 | 73,681 | 0.309 | 0.7% |
| 13 | 74,960 | 0.282 | 0.2% |
| 14 | 75,924 | 0.232 | 0.2% |

</details>

