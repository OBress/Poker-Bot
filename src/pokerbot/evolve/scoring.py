"""Classifier scoring (MATH.md §6, fixed rules).

For M scored seats with true styles y and posteriors p(t) after t hands:
  LL(t)  = −mean log max(p_y(t), 1e-6)           log loss
  BS(t)  = mean Σ_k (p_k(t) − 1[k = y])²          Brier
  ECE(t) over 10 equal-width confidence bins      calibration
  F_clf  = −Σ_t w_t LL(t) − λ_ece ECE(100) − λ_time max(0, µs per call − budget)
with w = (0.30, 0.25, 0.20, 0.15, 0.10) at t = (10, 25, 50, 100, 200).
Real players use the soft label q: cross-entropy H(q, p) = −Σ_k q_k log p_k (§6.2).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pokerbot.evolve.harness import CHECKPOINTS

WEIGHTS = (0.30, 0.25, 0.20, 0.15, 0.10)
FLOOR = 1e-6
LAMBDA_ECE = 1.0
LAMBDA_TIME = 1e-4  # fitness per µs over budget
TIME_BUDGET_US = 2000.0


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = p.max(axis=1)
    hit = (p.argmax(axis=1) == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def per_seat_ll(posteriors: Sequence[np.ndarray], labels: Sequence[Sequence[int]]) -> np.ndarray:
    """(seats, checkpoints) log loss for every scored seat, sessions concatenated."""
    rows = []
    for post, y in zip(posteriors, labels, strict=True):
        y = np.asarray(y)
        p = post[np.arange(len(y)), :, y]  # (seats, checkpoints)
        rows.append(-np.log(np.maximum(p, FLOOR)))
    return np.concatenate(rows)


def session_scores(posteriors, labels) -> np.ndarray:
    """Per session −Σ_t w_t LL(t) (higher is better): the unit of paired promotion tests."""
    out = []
    for post, y in zip(posteriors, labels, strict=True):
        ll = per_seat_ll([post], [y]).mean(axis=0)
        out.append(-float(np.dot(WEIGHTS, ll)))
    return np.array(out)


def score(posteriors, labels, us_per_call: float, checkpoints=CHECKPOINTS) -> dict:
    ll = per_seat_ll(posteriors, labels)
    p_all = np.concatenate(posteriors)  # (seats, checkpoints, K)
    y = np.concatenate([np.asarray(v) for v in labels])
    out: dict = {"seats": int(len(y)), "us_per_call": us_per_call}
    for i, t in enumerate(checkpoints):
        p = p_all[:, i]
        onehot = np.eye(p.shape[1])[y]
        out[f"logloss@{t}"] = float(ll[:, i].mean())
        out[f"brier@{t}"] = float(((p - onehot) ** 2).sum(axis=1).mean())
        out[f"accuracy@{t}"] = float((p.argmax(axis=1) == y).mean())
        out[f"ece@{t}"] = ece(p, y)
    weighted = float(np.dot(WEIGHTS, [out[f"logloss@{t}"] for t in checkpoints]))
    over = max(0.0, us_per_call - TIME_BUDGET_US)
    out["fitness"] = -weighted - LAMBDA_ECE * out["ece@100"] - LAMBDA_TIME * over
    return out


def confusion(posteriors, labels, checkpoint_index: int, k: int) -> np.ndarray:
    """Rows true style, columns mean posterior mass (soft confusion matrix)."""
    m = np.zeros((k, k))
    n = np.zeros(k)
    for post, y in zip(posteriors, labels, strict=True):
        for j, label in enumerate(y):
            m[label] += post[j, checkpoint_index]
            n[label] += 1
    return m / np.maximum(n[:, None], 1)


def cross_entropy(q: np.ndarray, p: np.ndarray) -> float:
    """Mean H(q, p) over players (soft real labels, MATH.md §6.2)."""
    return float(-(q * np.log(np.maximum(p, FLOOR))).sum(axis=1).mean())
