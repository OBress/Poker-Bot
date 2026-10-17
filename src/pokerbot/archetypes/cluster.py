"""Gaussian-mixture style discovery and model selection (docs/MATH.md §3.2–3.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


@dataclass
class KResult:
    k: int
    model: GaussianMixture
    loglik: float  # total log-likelihood on the fitting data
    bic: float
    ari: float  # mean bootstrap stability (adjusted Rand index vs the full fit)
    min_share: float


def fit_gmm(z: np.ndarray, k: int, seed: int, n_init: int = 4) -> GaussianMixture:
    return GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=n_init,
        max_iter=500,
        reg_covar=1e-4,
        random_state=seed,
    ).fit(z)


def stability(z: np.ndarray, model: GaussianMixture, k: int, seed: int, boots: int) -> float:
    """Refit on bootstrap resamples; agreement of their labels with the full model's labels."""
    rng = np.random.default_rng(seed)
    full = model.predict(z)
    scores = []
    for b in range(boots):
        idx = rng.integers(0, len(z), len(z))
        m = fit_gmm(z[idx], k, seed=seed + 1000 + b, n_init=1)
        scores.append(adjusted_rand_score(full, m.predict(z)))
    return float(np.mean(scores))


def select_k(z: np.ndarray, ks: range, seed: int, boots: int = 5, log=None) -> list[KResult]:
    """Fit each K; ``log(k, result)`` is called after each fit (for run tracking)."""
    results = []
    for k in ks:
        m = fit_gmm(z, k, seed)
        r = KResult(
            k=k,
            model=m,
            loglik=float(m.score(z) * len(z)),
            bic=float(m.bic(z)),
            ari=stability(z, m, k, seed, boots),
            min_share=float(m.weights_.min()),
        )
        results.append(r)
        if log:
            log(k, r)
    return results


def choose(results: list[KResult], min_ari: float = 0.8, min_share: float = 0.02) -> KResult:
    """Lowest BIC among Ks that are stable and have no tiny components (MATH.md §3.3). Falls
    back to the most stable K if none qualifies."""
    ok = [r for r in results if r.ari >= min_ari and r.min_share >= min_share]
    if ok:
        return min(ok, key=lambda r: r.bic)
    return max(results, key=lambda r: r.ari)
