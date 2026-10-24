Written after the run. It interprets the results; it doesn't change any pre-registered decision.

**The project's question: does the exploiting agent beat the equilibrium baseline against
exploitable opponents?** Yes, in both games where an exploiter exists. In Leduc the evolved
program wins +0.51 chips/hand more than the exact Nash strategy, which itself already wins
0.11–0.49 against these styles. In heads-up no-limit it wins +107 bb/100 more than the MCCFR
blueprint, which itself wins +21 to +37. The gain holds against every style. It also holds
against players who switch style every 25/50 hands (+0.33 and +96).

**Safety against the baseline itself.** In Leduc the loss is 0.002 chips/hand, well inside the
0.01 margin (A2). In heads-up, B2 is **not supported**: −3.0 bb/100 [−7.2, +1.5] on 2,000
blueprint-only sessions, so a loss of up to about 7 bb/100 against an equilibrium opponent
can't be ruled out at the pre-registered 5 bb/100 margin. Phase 6's separate check gave
+2.5 [−5.9, +10.6]. The honest summary is a small loss of 0–7 bb/100 against the blueprint,
against a gain of about 107 against exploitable players. Even so, the exploiter comes out
ahead unless more than about 94% of opponents play like the blueprint.

**What the classifier is for.** The belief ablations agree across both games: calibrated style
probabilities matter mostly for *not* exploiting a player who can't be exploited.

- Leduc: the program exploits only when its beliefs are confident, so with the prior alone it
  plays Nash and gains nothing. With the classifier's top style as a certainty (argmax) it keeps
  the edge (+0.519) but loses 0.135 chips/hand to Nash, against 0.003 with calibrated beliefs.
- Heads-up: the edge doesn't need the classifier (prior only: +106 vs +107); the program reads
  the opponent's own counts. The classifier's contribution is the loss against the blueprint:
  −22.6 [−39.1, −5.7] with the prior only, −11.6 with argmax, −4.9 with the classifier.
- True labels (oracle) add little against fixed styles (+0.02 Leduc, +4 heads-up). They add
  more against switchers (+0.14 and +12): the classifier lags after each switch.

**Evolution vs hand-written code.** The seeds capture a fraction of the edge: +0.15 in Leduc,
and about zero (+1.5 bb/100) in heads-up, where the seed also loses to switchers. Every primary
comparison of evolved vs seed is significant at p < 1e-36. The 6-max classifier gains 2.6–3.9
accuracy points at every checkpoint (66.4% vs 63.1% at 200 hands) and 13% lower log loss at
200 hands. On held-out real players it repeats the Phase 5 numbers: the same players and
deterministic programs give the same values. Its cross-entropy is lower from 50 hands on, and
its accuracy is the same (64% at 200 hands).

**Speed of exploitation.** In Leduc the edge reaches its plateau within about 20 hands. In
heads-up it is there from the first hands, because the program starts from population priors
and the opponent's counts rather than waiting for a label.

**Limits.** All exploiter opponents are calibrated bots. Before claiming anything about people,
the next check is the exploiter against held-out real players' behavior, e.g. against
behavior-cloned opponents.
