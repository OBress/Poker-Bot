# The math

The formulas behind stats, style discovery, classification, exploitation and evaluation. These
define the **seed** implementations and the **fixed scoring rules**. The evolution loop may replace
the seed formulas with better ones, but never the scoring rules in §6–§7.

Notation: K styles indexed by k; D stats indexed by j; a player's observed data is called 𝒟.
σ(x) = 1/(1+e^(−x)) is the logistic function and logit(r) = log(r/(1−r)) its inverse.

---

## 1. Position and context

### 1.1 Position

With n seats and button b, a seat s has relative position r = (s − b) mod n: r = 0 is the button,
r = 1 the small blind, r = 2 the big blind (heads-up: the button is the small blind). Preflop action
order runs r = 3, 4, …, n−1, 0, 1, 2.

| 6-max r | 3 | 4 | 5 | 0 | 1 | 2 |
|---|---|---|---|---|---|---|
| Name | UTG (EP) | HJ (MP) | CO | BTN | SB | BB |

For tables with fewer players, positions are removed starting from the earliest (UTG first), so BTN,
SB, BB and CO always exist when there are enough seats. Every stat in §2 can be split by position
class P ∈ {EP, MP, CO, BTN, SB, BB}.

Postflop, **in position (IP)** means acting last among the players still in the hand on that street.

### 1.2 Other context

| Variable | Definition | Buckets |
|---|---|---|
| Effective stack | min(own stack, largest opposing stack) / big blind, at the start of the hand | < 25, 25–60, 60–150, > 150 bb |
| Pot type | number of players who saw the flop | heads-up (2), multiway (3+) |
| Preflop line | single-raised, 3-bet, 4-bet+, limped | 4 classes |
| Bet size | b = bet / pot_before (pot before the bet, including current-street bets) | bins edges 0.25, 0.4, 0.6, 0.85, 1.1, 1.6, 2.5 |

## 2. Stats as counts

Every stat j is a pair of counts: **opportunities** n_j (times the player *could* do it) and
**occurrences** x_j (times they did). We never store just the ratio, because 3/5 and 300/500 mean very
different things.

| Stat | Opportunity (counts in n) | Occurrence (counts in x) |
|---|---|---|
| VPIP | every hand dealt where the player isn't the BB in an unopened walk | voluntarily calls or raises preflop (posting blinds and checking don't count) |
| PFR | same as VPIP | raises preflop |
| RFI(P) — open raise by position | action folds to the player in position P | raises (limping is not RFI) |
| Limp | action folds or limps to the player, not in the BB | just calls the big blind |
| 3-bet | faces exactly one raise preflop with the option to re-raise | re-raises |
| Fold to 3-bet | opened, then faces a 3-bet | folds |
| 4-bet | faces a 3-bet with the option to raise | raises |
| Steal | folds to the player in CO, BTN or SB | raises |
| Fold to steal | in SB/BB, facing a steal-position open with no callers | folds |
| C-bet (street, pot type) | was the last preflop aggressor (or last aggressor on the previous street) and is checked to or acts first | bets |
| Fold to c-bet | faces a c-bet | folds |
| Check-raise | checks, then faces a bet on the same street | raises |
| Donk bet | out of position vs the previous street's aggressor, acts first | bets |
| Aggression frequency (street) | any postflop decision facing a bet or with the option to bet | bets or raises |
| WTSD | saw the flop | reached showdown |
| W$SD | reached showdown | finished the hand a net winner (payoff > 0) |
| River bluff rate *(not yet computed)* | bet or raised the river and showed down | the shown hand was in the bottom strength bucket (§4.5) |

**Derived feature — positional awareness:**
  PA = logit(r̂_RFI,BTN) − logit(r̂_RFI,EP)
Solid players open roughly 12% under the gun and 45% on the button (PA ≈ 1.8). Weak players open
nearly the same range everywhere (PA ≈ 0).

The classic aggression factor AF = (bets + raises) / calls is a ratio of two counts with no fixed
denominator, which makes it unstable on small samples. We use aggression *frequency* instead, which
is a proper count/opportunity pair.

## 3. Discovering the styles (offline, on real players)

### 3.1 From counts to features

A rate estimated from a few hands is noisy. Shrink it toward the population with a Beta(a_j, b_j)
prior fitted to all players (method of moments on players with ≥ 500 hands):

  r̂_j = (x_j + a_j) / (n_j + a_j + b_j)

Transform to an unbounded scale and record how uncertain it is (delta method):

  z_j = logit(r̂_j),  s_j² ≈ 1 / ((n_j + a_j + b_j) · r̂_j (1 − r̂_j))

Then standardize each z_j across players (subtract the mean, divide by the standard deviation).
Each player i becomes a feature vector z_i ∈ ℝ^D with noise covariance S_i = diag(s_i1², …, s_iD²).

### 3.2 Gaussian mixture

Styles are mixture components:

  p(z) = Σ_k π_k · 𝒩(z | μ_k, Σ_k)

with population shares π_k (sum to 1), centers μ_k and full covariances Σ_k, fitted by EM.

**Upgrade for noisy players (extreme deconvolution).** A player with 600 hands has a noisier z than one
with 20,000. Rather than discard information, fit the mixture to the *true* stats, accounting for each
player's measurement noise:

  p(z_i) = Σ_k π_k · 𝒩(z_i | μ_k, Σ_k + S_i)

so Σ_k describes how real players in a style differ, not how noisy our measurements are.

### 3.3 Choosing K

  BIC(K) = −2 log L̂ + d_K log N,  d_K = (K − 1) + K·D + K·D(D + 1)/2

where N is the number of players. Pick the K with the lowest BIC, provided that

- **stability:** fitting on bootstrap resamples of players gives a mean adjusted Rand index ≥ 0.8
  against the full fit, and
- **readability:** every component has a clear description and a share π_k ≥ 2%.

### 3.4 Soft membership of a real player

  q_ik = π_k 𝒩(z_i | μ_k, Σ_k + S_i) / Σ_l π_l 𝒩(z_i | μ_l, Σ_l + S_i)

The vector q_i is the "true" label of real player i when we test the classifier on real data (§6.2).

## 4. The classifier (online, during play)

### 4.1 What it outputs

For each opponent, a probability vector

  p = (p_1, …, p_K),  p_k = P(style = k | 𝒟),  p_k ≥ 0,  Σ_k p_k = 1

and a continuous estimate of the player's stats, θ̂ (§4.4). Everything below follows from Bayes' rule:

  p_k = π_k L_k(𝒟) / Σ_l π_l L_l(𝒟),  L_k(𝒟) = P(𝒟 | style k)

With no data L_k = 1, so p = π (the population mix). It is computed in log space as a softmax:

  ℓ_k = log π_k + log L_k(𝒟),  p_k = exp(ℓ_k − max_l ℓ_l) / Σ_l exp(ℓ_l − max_l ℓ_l)

### 4.2 Seed A — Beta-Binomial naive Bayes

For each style k and stat j, fit Beta(α_kj, β_kj) to the rates of real players in that style (weighted by
q_ik). Given the true rate is unknown, the probability of seeing x successes in n opportunities is
the Beta-Binomial:

  P(x | n, α, β) = C(n, x) · B(x + α, n − x + β) / B(α, β)

where B is the Beta function. Then

  ℓ_k = log π_k + Σ_j w_j [ log B(x_j + α_kj, n_j − x_j + β_kj) − log B(α_kj, β_kj) ]

(the C(n, x) term is the same for every k and cancels). Weights w_j ∈ [0, 1] correct for double counting
of correlated stats (VPIP and PFR say similar things); they start at 1 and are tuned. A temperature τ
(p = softmax(ℓ/τ)) is tuned for calibration.

Properties: costs O(K·D) per update, handles n = 0 naturally, and sharpens automatically as hands accumulate.

### 4.3 Seed B — correlated Gaussian in logit space

Seed A treats stats as independent. Seed B keeps the correlations learned in §3.2. Let O be the set of
stats with n_j > 0, and (z_O, S_O) the player's current features and noise from §3.1. Because both the
style spread and the measurement noise are Gaussian, the marginal likelihood is

  L_k = 𝒩(z_O | μ_k,O, Σ_k,OO + S_O)

Stats never observed are simply dropped (the marginal of a Gaussian is the sub-vector). As n grows,
S_O → 0 and the likelihood reflects only how far the player sits from each style's center.

### 4.4 Continuous stat estimate

The player's true features θ also have a closed-form posterior within each style:

  Σ'_k = (Σ_k⁻¹ + S_O⁻¹)⁻¹,  μ'_k = Σ'_k (Σ_k⁻¹ μ_k + S_O⁻¹ z_O)

and overall

  θ̂ = E[θ | 𝒟] = Σ_k p_k μ'_k

Converted back with σ(·), θ̂ gives shrunk estimates such as "fold to c-bet ≈ 0.71". This is what lets the
exploiter treat a player who sits between two styles correctly, instead of forcing them into one.

### 4.5 Showdown evidence

When a player shows a hand, we observe their strength bucket h ∈ {1, …, 5} given the line they took
(e.g. "bet-bet-bet", "check-call-check-call"). The table P(h | line class, k) is estimated from simulated
or real data per style, and multiplied into the likelihood:

  ℓ_k += Σ_showdowns log P(h | line, k)

Strength buckets use hand equity versus a uniform random hand on the final board: < 0.35, 0.35–0.55,
0.55–0.75, 0.75–0.9, ≥ 0.9.

### 4.6 Drift

To follow players who change gears, counts decay geometrically after every hand:

  x_j ← λ x_j,  n_j ← λ n_j,  λ ∈ (0.98, 1]

This gives an effective memory of about 1/(1 − λ) hands. λ = 1 means no forgetting. It is tuned
against the drifting and style-switching bots.

## 5. The exploiter

### 5.1 Break-even formulas

These are the standard results every exploit reduces to. Pot P (before our action), our bet B, the amount
we must call C.

| Decision | Profitable if | Break-even |
|---|---|---|
| Pure bluff | opponent fold probability f > B / (P + B) | f* = B / (P + B) |
| Call | our equity e > C / (P' + C), where P' is the pot including the opponent's bet | e* = C / (P' + C) (pot odds) |
| Value bet (vs checking, ignoring raises) | our equity against the opponent's *calling* range > 0.5 | e* = 0.5 vs calling range |

An equilibrium defender calls often enough that a bluff of size B gains nothing: the **minimum defense
frequency** MDF = P / (P + B). If our estimate f̂ of an opponent's fold rate (from θ̂, e.g. fold to c-bet)
is above 1 − MDF, bluffs show a profit with any two cards; if it's below, bluffs lose and we should bluff
less than equilibrium. The classifier supplies f̂ and how sure we are of it.

### 5.2 The seed policy: tilt the baseline

Action menu 𝒜 = {fold, check/call, bet 33%, bet 75%, bet 125%, all-in}, baseline policy σ_b(a | s) at
information state s.

Each style k has an adjustment score Δ_k(a | s) (seed: hand-written from §5.1, e.g. vs a station:
+value-bet, −bluff). Combine with the beliefs:

  Δ̄(a | s) = Σ_k p_k Δ_k(a | s)

Confidence from the entropy of the beliefs:

  c = 1 − H(p) / log K,  H(p) = −Σ_k p_k log p_k,  c ∈ [0, 1]

Exploiter policy:

  σ(a | s) = σ_b(a | s) · exp(β c Δ̄(a | s)) / Z(s)

where Z normalizes the probabilities. When we know nothing (c = 0), σ = σ_b exactly.

Why this form: it is the exact solution of

  max_σ  E_{a∼σ}[Δ̄(a | s)] − (1 / βc) · KL(σ ‖ σ_b)

that is, "gain as much as the adjustment scores promise, while paying a price for straying from the
baseline". Less confidence → higher price → closer to the baseline. This is what bounds the loss to
misreads and switchers.

### 5.3 What the evolution loop changes

The LLM rewrites how Δ_k is computed (for example, from θ̂ and the formulas in §5.1 rather than per-style
constants), how c is computed, and the policy form itself. The tuner sets β and any constants. The
output contract (a distribution over 𝒜) never changes.

## 6. Scoring the classifier

### 6.1 Proper scoring rules

For M test seats with true styles y_m and predictions p_m(t) after t hands:

  Log loss  LL(t) = −(1/M) Σ_m log max(p_m,y_m(t), 10⁻⁶)

  Brier  BS(t) = (1/M) Σ_m Σ_k (p_m,k(t) − 1[k = y_m])²

Both are *proper*: the expected score is best when the classifier reports its honest probabilities, so it
can't be gamed by over- or under-confidence. The 10⁻⁶ floor bounds the cost of a single confident miss.

**Calibration** (ECE): group predictions into bins B_1…B_10 by confidence max_k p_k:

  ECE = Σ_b (|B_b| / M) · | accuracy(B_b) − mean confidence(B_b) |

### 6.2 Real players (soft labels)

The label is the soft membership q_m from §3.4, so log loss becomes cross-entropy:

  H(q, p) = −Σ_k q_k log p_k

The prediction p uses only the player's first t hands; q uses all of them.

### 6.3 The fitness the loop optimizes

  F_clf = − Σ_t w_t · LL(t) − λ_ece ECE(100) − λ_time · max(0, μs per call − budget)

with checkpoints t ∈ {10, 25, 50, 100, 200} and weights w = (0.30, 0.25, 0.20, 0.15, 0.10): fast
identification is worth more than eventual identification.

### 6.4 Cost-weighted error (once the exploiter exists)

Some mistakes are free (two styles that need the same exploit), others are expensive. Define

  C[i, j] = EV_i(exploit tuned for i) − EV_i(exploit tuned for j)  ≥ 0  (bb/100)

the money lost by playing the style-j exploit against a true style i. The decision the classifier's
beliefs lead to is ĵ = argmin_j Σ_i p_i C[i, j], and its cost is

  Cost = (1/M) Σ_m C[y_m, ĵ_m]

In Stage 3 of the evolution loop this replaces log loss as the primary target.

## 7. Scoring the exploiter

### 7.1 Win rate in duplicate

Hand h is played twice with the same cards, seats and opponents: once by the exploiter (payoff u_h^E)
and once by the baseline (u_h^B), in big blinds. Per-hand difference d_h = u_h^E − u_h^B.

  Δ = 100 · d̄ bb/100,  d̄ = (1/H) Σ_h d_h

Hands within a session aren't independent (opponent state carries over), so the standard error comes
from a **block bootstrap** over whole sessions, not from individual hands. 95% CI = Δ ± 1.96 · SE.

### 7.2 All-in luck removal

If players are all-in before the river, the realized result is replaced by its expectation over the
remaining cards. For a pot V and our equity e (enumerated exactly over the remaining board):

  u_adjusted = e · V − (our contribution)

This removes the largest single source of variance without bias.

### 7.3 Fitness

  F_exp = Δ_mixed − λ_1 · max(0, −Δ_vs-baseline − ε) − λ_2 · max(0, −Δ_switchers) − λ_3 · KL̄

- Δ_mixed: duplicate win-rate edge against sampled tables (validation seeds).
- Δ_vs-baseline: exploiter vs the baseline itself; should be ≈ 0, any loss beyond tolerance ε is penalized.
- Δ_switchers: edge against style-switching bots; negative means the exploiter got burned.
- KL̄: average KL(σ ‖ σ_b), a mild preference for not deviating without reason.

### 7.4 How many hands we need

To detect an edge δ (bb/100) with significance level α and power 1 − β, when the per-100-hand standard
deviation of d is s:

  H ≈ 100 · ((z_{1−α} + z_{1−β}) · s / δ)²

Example (estimate; we'll measure s in Phase 4): α = 0.01, power 0.9 → z-sum ≈ 2.33 + 1.28 ≈ 3.6.
Raw NLHE results have a standard deviation of roughly 80–120 bb per 100 hands; duplicate play plus the
all-in adjustment plausibly brings the *difference* d down to s ≈ 30. To detect δ = 2 bb/100:
H ≈ 100 · (3.6 · 30 / 2)² ≈ 290,000 hands (both copies of each hand are played, so ~580k simulated hands).
The raw engine does that in well under a minute on 8 cores; real bots and baseline lookups will be
slower, but a long evaluation per promising candidate is affordable. Screens use ~10k hands.

### 7.5 Promotion test

A candidate replaces the champion only if a one-sided paired test on the same validation seeds gives
p < 0.01, and the result repeats on a fresh batch of validation seeds (this guards against one lucky
candidate out of hundreds being promoted).
