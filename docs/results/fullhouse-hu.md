# External check: the heads-up exploiter vs the fullhouse reference bots

A post-hoc check, not part of the pre-registered Phase 7 experiment. Question: does the evolved
heads-up exploiter still beat the equilibrium blueprint against opponents it was never built or
tuned for? The opponents are the five reference bots of the e-x.club Showdown engine
(`src/pokerbot/opponents/fullhouse`, frozen at `fullhouse-engine@a80b269`).

Setup (`configs/bench-fullhouse-hu.yaml`): 400 sessions of 300 hands, 80 per opponent, 100 bb
stacks, 50/100 blinds, button alternating. Each session is played twice with the same cards and the
same opponent (same seed): once by the exploiter, once by the blueprint. Delta = exploiter minus
blueprint, all-in runouts replaced by their expectation, 95% CI from a bootstrap over sessions.
120,000 hands per copy. The sessions use their own seed (20261030), not the test-split indices of
Phases 5-7. The blueprint fell back to its heuristic in at most 0.4% of decisions against these
bots.

| Opponent | Champion − blueprint | Seed − blueprint | Blueprint alone |
|---|---|---|---|
| aggressor | +494.8 ± 48.5 | +114.3 ± 39.1 | +348.2 |
| mathematician | +5.3 ± 1.8 | +0.1 ± 1.8 | +61.3 |
| ref_bot_2 | +4.9 ± 1.7 | −2.3 ± 1.9 | +62.0 |
| template | +4.1 ± 4.4 | −0.4 ± 3.6 | +40.6 |
| shark | −4.1 ± 5.1 | −8.7 ± 4.6 | +39.9 |
| **all five** | **+101.0 ± 21.6** | **+20.6 ± 9.2** | +110.4 ± 14.5 |

Units are bb/100. Runs: champion `20261029-164956-bench-fullhouse-hu-champion`, seed
`20261029-165057-bench-fullhouse-hu-seed`.

## Reading it

- The champion beats the blueprint overall, and the interval excludes zero. But the pooled number
  is almost entirely one opponent: the aggressor raises 70% of the time whatever it holds, and the
  champion wins about 2.4x the blueprint's rate against it (+843 vs +348). Against the other four the gain is
  about +2.6 bb/100 on average, small next to the sampling error: significant against two of them
  (mathematician, ref_bot_2), not against template, and a non-significant loss to shark
  (−4.1 [−9.2, +1.0]).
- That matches Phase 6: the exploiter earns its edge from opponents that are exploitable, and
  stays close to the blueprint otherwise. The tight, passive bots offer little to exploit and the
  classifier cannot say "this is a pot-odds caller" because none of its styles looks like one.
- The seed program does not generalize: +20.6 overall, all of it against the aggressor, and a
  significant loss to shark (−8.7 [−13.3, −4.0]). The evolved champion is better on every opponent.
- Absolute win rates (e.g. +110 for the blueprint) say how weak these bots are, not how strong our
  player is. The reference bots are crude rule-based players.
- Not tested here: a 6-max exploiter (none exists), stacks other than 100 bb, or the 400-hand
  carry-over format (the blueprint is a cash-game strategy at 100 bb).

Re-run:

```bash
uv run pokerbot eval duplicate --config configs/bench-fullhouse-hu.yaml --hero exploiter-hu:champion
```
