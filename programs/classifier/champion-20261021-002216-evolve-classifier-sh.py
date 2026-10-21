"""Classifier candidate. Frozen template classifier_v1 (EVOLUTION.md §2).

Only the code between "# EVOLVE-BLOCK <name>" and "# END-EVOLVE-BLOCK" may change. The harness
creates one Classifier per session, feeds it every public event in order (observe) and asks for
a seat's style probabilities with posterior(seat, feats), which must return K non-negative finite
numbers (the harness normalizes them). PARAMS values are tuned within [low, high].
"""

import math

import numpy as np

from pokerbot.evolve.api import (
    BUCKETS,
    LINES,
    Event,
    HandEnd,
    HandStart,
    PlayerAction,
    SeatFeatures,
    Showdown,
    StyleProfiles,
    param,
)

# EVOLVE-BLOCK params
PARAMS = {
    "w_bb": param(0.508, 0.0, 2.0),       # weight on exact marginal Beta-Binomial evidence
    "w_gauss": param(0.251, 0.0, 2.0),    # weight on joint Gaussian evidence in logit space
    "w_sd": param(0.543, 0.0, 2.0),       # weight on shown-hand evidence
    "tau": param(0.944, 0.3, 4.0),        # softmax temperature
    "prior_power": param(0.850, 0.0, 3.0),# sharpness of the population style prior
    "shrink": param(0.278, 0.0, 0.95),    # covariance shrinkage toward identity
    "ridge": param(0.041, 1e-4, 1.0),     # diagonal jitter for numerical stability
    "s2_floor": param(0.0018, 1e-6, 0.5), # floor on measurement variance
    "imp_power": param(0.5, 0.0, 2.0),    # exponent on the per-stat discriminative weight
    "boost_pos": param(1.15, 0.5, 2.5),   # extra weight for position/stack-depth splits
    "boost_street": param(1.15, 0.5, 2.5),# extra weight for street/bet-size/aggression stats
    "boost_action": param(1.3, 0.5, 3.0), # extra weight for limp/steal/3-bet/4-bet
}
# END-EVOLVE-BLOCK

# EVOLVE-BLOCK helpers
def _lgamma(x):
    x = np.asarray(x, dtype=float)
    small = x < 0.5
    xr = np.where(small, 1.0 - x, x) - 1.0
    a = (0.99999999999980993
         + 676.5203681218851 / (xr + 1.0)
         - 1259.1392167224028 / (xr + 2.0)
         + 771.32342877765313 / (xr + 3.0)
         - 176.61502916214059 / (xr + 4.0)
         + 12.507343278686905 / (xr + 5.0)
         - 0.13857109526572012 / (xr + 6.0)
         + 9.9843695780195716e-6 / (xr + 7.0)
         + 1.5056327351493116e-7 / (xr + 8.0))
    t = xr + 7.5
    res = 0.5 * np.log(2.0 * np.pi) + (xr + 0.5) * np.log(t) - t + np.log(a)
    x_safe = np.where(small, x, 0.5)
    refl = np.log(np.pi) - np.log(np.sin(np.pi * x_safe)) - res
    return np.where(small, refl, res)


def _log_beta(a, b):
    return _lgamma(a) + _lgamma(b) - _lgamma(a + b)


def _softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()
# END-EVOLVE-BLOCK


class Classifier:
    def __init__(self, profiles: StyleProfiles, params: dict[str, float]) -> None:
        self.profiles = profiles
        self.p = params
        self.reset()

    def reset(self) -> None:
        # EVOLVE-BLOCK state
        self.beta_a = self.profiles.beta_a
        self.beta_b = self.profiles.beta_b
        self.log_beta_ab = _log_beta(self.beta_a, self.beta_b)
        self.means = self.profiles.means
        self.covs = self.profiles.covs
        self.log_shares = np.log(self.profiles.shares)
        self.K = self.means.shape[0]
        self.D = self.means.shape[1]
        self.I = np.eye(self.D)
        self.hands_idx = [i for i, f in enumerate(self.profiles.features) if f == "hands"]

        # Per-stat discriminative weight: how far apart the style means are relative to
        # the within-style spread (a Fisher ratio computed from the profile itself).
        mu = np.asarray(self.means, dtype=float)
        var_between = mu.var(axis=0)
        var_within = np.zeros(self.D)
        for k in range(self.K):
            var_within += np.diag(np.asarray(self.covs[k], dtype=float))
        var_within = var_within / max(self.K, 1)
        ratio = var_between / (var_within + 1e-6)
        ratio = np.where(np.isfinite(ratio) & (ratio > 0.0), ratio, 0.0)
        nz = ratio > 0.0
        if nz.any():
            ratio = ratio / ratio[nz].mean()
        ratio = np.minimum(ratio, 10.0)

        power = float(self.p["imp_power"])
        w = np.power(np.maximum(ratio, 1e-6), power)

        # Boost stats that actually separate the confused passive/aggressive styles:
        # preflop action verbs, position & stack-depth splits, and street aggression.
        pos_kw = ("vpip_", "pfr_", "rfi_")
        street_kw = ("agg_", "cbet", "donk", "xr", "betsize")
        action_kw = ("limp", "steal", "3bet", "4bet")
        bp = float(self.p["boost_pos"])
        bs = float(self.p["boost_street"])
        ba = float(self.p["boost_action"])
        names = list(self.profiles.features)
        for i, name in enumerate(names):
            mult = 1.0
            for kw in pos_kw:
                if kw in name:
                    mult *= bp
                    break
            for kw in street_kw:
                if kw in name:
                    mult *= bs
                    break
            for kw in action_kw:
                if kw in name:
                    mult *= ba
                    break
            w[i] = w[i] * mult

        for i in self.hands_idx:
            w[i] = 0.0
        self.feat_w = w
        # END-EVOLVE-BLOCK

    def observe(self, event: Event) -> None:
        # EVOLVE-BLOCK observe
        pass
        # END-EVOLVE-BLOCK

    def posterior(self, seat: int, feats: SeatFeatures) -> np.ndarray:
        # EVOLVE-BLOCK posterior
        prof = self.profiles
        p = self.p
        n = np.array([feats.n(f) for f in prof.features], dtype=float)
        x = np.array([feats.x(f) for f in prof.features], dtype=float)

        # The raw hand-count feature is not a behavioural rate; exclude it from both terms.
        for idx in self.hands_idx:
            n[idx] = 0.0
            x[idx] = 0.0

        # Exact marginal Beta-Binomial evidence for every stat (kept per feature).
        a = self.beta_a
        b = self.beta_b
        N = n[None, :]
        X = x[None, :]
        bb_terms = (_lgamma(X + a) + _lgamma(N - X + b) - _lgamma(N + a + b)
                    - self.log_beta_ab)

        # Reweight each stat by how much it actually discriminates styles, renormalised
        # over the currently active stats so early-hand confidence is unchanged.
        active = n > 0.0
        wa = np.where(active, self.feat_w, 0.0)
        if (wa > 0.0).any():
            den = wa[active].mean()
            if (not np.isfinite(den)) or den <= 0.0:
                den = 1.0
            wa = wa / den
        else:
            wa = np.where(active, 1.0, 0.0)
        ll_bb = (bb_terms * wa[None, :]).sum(axis=1)

        # Joint Gaussian evidence: the transform gives a noisy observation z of the true
        # standardized logit with variance s2.  Sparse stats have large s2 and therefore
        # contribute little automatically.
        z, s2 = prof.transform(n, x)
        z = np.asarray(z, dtype=float)
        s2 = np.asarray(s2, dtype=float)
        z = np.where(np.isfinite(z), z, 0.0)
        s2 = np.where(np.isfinite(s2) & (s2 >= 0.0), s2, 1e6)
        mask = n > 0.0
        z = np.where(mask, z, 0.0)
        s2 = np.where(mask, s2, 1e6)
        s2 = np.maximum(s2, p["s2_floor"])

        eye = self.I
        shrink = p["shrink"]
        ridge = p["ridge"]
        ll_g = np.empty(self.K)
        for k in range(self.K):
            cov = (1.0 - shrink) * self.covs[k] + shrink * eye
            cov = cov + np.diag(s2 + ridge)
            cov = 0.5 * (cov + cov.T)
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(cov + 1e-3 * eye)
            diff = z - self.means[k]
            sol = np.linalg.solve(L, diff)
            mahal = sol @ sol
            logdet = 2.0 * np.log(np.diag(L)).sum()
            ll_g[k] = -0.5 * (mahal + logdet)

        # Shown-hand evidence.
        sd = (np.log(prof.showdown + 1e-12) * feats.shown).sum(axis=(1, 2))

        logits = (p["prior_power"] * self.log_shares
                  + p["w_bb"] * ll_bb
                  + p["w_gauss"] * ll_g
                  + p["w_sd"] * sd)
        return _softmax(logits / p["tau"])
        # END-EVOLVE-BLOCK


_UNUSED = (math, BUCKETS, LINES, HandEnd, HandStart, PlayerAction, Showdown, param)
