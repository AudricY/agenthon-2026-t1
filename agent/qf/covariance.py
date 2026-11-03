"""Covariance: EWMA (RiskMetrics), Ledoit-Wolf shrinkage, correlation repair (Higham's nearest correlation).

A panel is T x N: a row per period, a column per asset, oldest first. A function with several results returns a
dict with named keys.
"""
from __future__ import annotations

import numpy as np

from ._common import Result, choice, integer
from .conventions import SEEDS, _finite, ewma

__all__ = ["ewma_covariance", "ledoit_wolf", "nearest_correlation"]

TARGETS = ("scaled_identity", "constant_correlation")
NEAREST = ("clip", "higham")
EWMA_DEMEANS = ("none", "sample_mean")

def ewma_covariance(returns, lam, seed, demean, window):
    """EWMA (RiskMetrics) covariance matrix of a return panel, its seed, demeaning and trailing window named.

    S_t = lam S_{t-1} + (1 - lam) x_t x_t' over the rows x_t used, oldest first: qf.ewma on each product x_i x_j, so
    the diagonal is qf.ewma_volatility's variance before annualising. seed is the state before the first row: 'first'
    that row's own outer product (S_0 = x_0 x_0'), 'zero' 0, 'mean' the rows' mean outer product, a matrix that matrix
    (a sample covariance: numpy.cov(rows, rowvar=False, ddof=1)); 'adjusted' normalises the weights instead, S_t = sum_i
    lam^i x_{t-i} x_{t-i}' / sum_i lam^i (pandas' adjust=True). The last S, after the last row, is the forecast for the
    next period: a portfolio's variance w' S w, its correlations S_ij / sqrt(S_ii S_jj).

    conventions:
    - lam: the decay, inside (0, 1) (RiskMetrics 0.94 daily; pandas alpha = 1 - lam)
    - seed: 'first', 'zero', 'mean', 'adjusted' or an N x N matrix (the state before the first row used)
    - demean: 'none' (raw returns: RiskMetrics' zero mean) or 'sample_mean' (each column less its mean over the rows
      used)
    - window: None (every row) or W, the last W rows only, the recursion started at the window's first row
    - returns: T x N, rows the dates oldest first, finite (drop or fill incomplete rows first)
    returns: a dict: covariance (N x N, after the last row used); path (an N x N matrix per row used); mean (the
      column means taken out, zeros under 'none')
    source: J.P. Morgan/Reuters, RiskMetrics Technical Document (1996), 5.3 (EWMA covariances); Hull, Risk Management
      and Financial Institutions, "Using EWMA for covariances"
    aliases: EWMA covariance, EWMA covariance matrix, RiskMetrics covariance, exponentially weighted covariance, EWMA correlation
    example:
        # Returns covariance [[1.2442e-4, 1.925e-4], [1.925e-4, 3.6058e-4]] (seed: the first row's outer product).
        qf.ewma_covariance(returns=[[0.01, 0.02], [-0.02, -0.01], [0.015, 0.005]], lam=0.94, seed="first", demean="none", window=None)
    """
    choice("demean", demean, EWMA_DEMEANS)
    x = _finite("returns", returns)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("returns must be T x N, rows the dates and columns the assets")
    if window is not None:
        x = x[-integer("window", window, 1):]
    names = x.shape[1]
    centre = x.mean(axis=0) if demean == "sample_mean" else np.zeros(names)
    rows = x - centre
    if isinstance(seed, str):
        choice("seed", seed, SEEDS)
        start = seed
    else:
        start = np.asarray(seed, dtype=float)
        if start.shape != (names, names) or not np.all(np.isfinite(start)):
            raise ValueError(f"seed must be 'first', 'zero', 'mean', 'adjusted' or a finite {names} x {names} matrix")
        start = start.ravel()
    products = (rows[:, :, None] * rows[:, None, :]).reshape(rows.shape[0], names * names)
    path = ewma(products, lam=lam, seed=start, nan="raise").reshape(rows.shape[0], names, names)
    return Result({"covariance": path[-1], "path": path, "mean": centre})


def ledoit_wolf(returns, *, target, demean, ddof):
    """Ledoit-Wolf shrinkage covariance: toward the scaled identity or constant correlation, at the optimal intensity.

    Sigma = delta F + (1 - delta) S, S = Y'Y / n (Y the panel, demeaned where asked; n = T - ddof). 'scaled_identity'
    F = mu I, mu the mean of S's diagonal (Ledoit and Wolf 2004, scikit-learn's LedoitWolf); 'constant_correlation'
    F_ij = rbar sqrt(s_ii s_jj) off the diagonal and s_ii on it, rbar the mean sample correlation (Ledoit and Wolf's
    "Honey, I shrunk"). delta = max(0, min(1, (pi - rho) / (gamma n))): pi = sum_ij of the mean over t of
    (y_ti y_tj - s_ij)^2, gamma = ||S - F||^2 (Frobenius), rho = 0 for the scaled identity and for the constant
    correlation sum_i pi_ii + rbar sum_{i != j} sqrt(s_jj / s_ii) theta_ij, theta_ij = mean of y_ti^3 y_tj - s_ii s_ij.
    conventions:
        returns: the T x N panel of returns (or any series), oldest first
        target: 'scaled_identity' or 'constant_correlation'
        demean: True (subtract each column's mean) or False (the returns taken as mean zero)
        ddof: 0 (S over T: scikit-learn's and the papers') or 1 (over T - 1: Ledoit and Wolf's 2020 code)
    returns: a dict: covariance (the shrunk N x N); shrinkage (delta, in [0, 1]); target (F); sample (S)
    source: Ledoit and Wolf (2004), "A well-conditioned estimator for large-dimensional covariance matrices", Journal
        of Multivariate Analysis 88(2); Ledoit and Wolf (2004), "Honey, I shrunk the sample covariance matrix",
        Journal of Portfolio Management 30(4); scikit-learn's sklearn.covariance.LedoitWolf
    aliases: Ledoit-Wolf, Ledoit Wolf shrinkage, shrinkage covariance, covariance shrinkage, shrunk covariance
    example:
        import qf
        r = [[0.010, 0.020, -0.010], [-0.020, 0.010, 0.000], [0.015, -0.005, 0.010], [0.000, 0.012, -0.004], [-0.010, -0.020, 0.020]]
        qf.ledoit_wolf(returns=r, target="scaled_identity", demean=True, ddof=0)
        # shrinkage = 0.64030408; covariance[0, 0] = 0.0001612680
    """
    choice("target", target, TARGETS)
    if not isinstance(demean, (bool, np.bool_)):
        raise ValueError("demean must be True or False")
    y = _finite("returns", returns)
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError("returns must be a T x N panel with at least two columns")
    spread = integer("ddof", ddof, 0)
    if spread > 1:
        raise ValueError("ddof must be 0 or 1")
    if demean:
        y = y - y.mean(axis=0)
    t, p = y.shape
    n = t - spread
    if n < 2:
        raise ValueError("the panel needs more rows than ddof + 1")
    sample = y.T @ y / n
    variances = np.diag(sample)
    if np.any(variances <= 0):
        raise ValueError("a column has zero variance")
    pi_matrix = (y * y).T @ (y * y) / n - sample * sample
    pi = float(pi_matrix.sum())
    if target == "scaled_identity":
        prior = np.eye(p) * variances.mean()
        rho = 0.0
    else:
        sd = np.sqrt(variances)
        rbar = float((np.sum(sample / np.outer(sd, sd)) - p) / (p * (p - 1)))
        prior = rbar * np.outer(sd, sd)
        np.fill_diagonal(prior, variances)
        theta = (y ** 3).T @ y / n - variances[:, None] * sample
        np.fill_diagonal(theta, 0.0)
        rho = float(np.trace(pi_matrix) + rbar * np.sum(np.outer(1.0 / sd, sd) * theta))
    gamma = float(np.sum((sample - prior) ** 2))
    delta = 0.0 if gamma == 0 else max(0.0, min(1.0, (pi - rho) / gamma / n))
    return Result({"covariance": delta * prior + (1.0 - delta) * sample, "shrinkage": delta, "target": prior,
                   "sample": sample})


def _psd(matrix, floor, w_half):
    """The nearest matrix with eigenvalues at least `floor` in the weighted Frobenius norm ||W^1/2 A W^1/2||."""
    shift = floor * np.eye(matrix.shape[0])
    scaled = w_half[:, None] * (matrix - shift) * w_half[None, :]
    values, vectors = np.linalg.eigh((scaled + scaled.T) / 2.0)
    kept = (vectors * np.maximum(values, 0.0)) @ vectors.T
    return kept / w_half[:, None] / w_half[None, :] + shift


def nearest_correlation(matrix, *, method, floor, weights, max_iterations, tol):
    """Nearest correlation (Higham's alternating projections) or eigenvalue clipping: a non-PSD estimate repaired.

    'clip' sets the eigenvalues below `floor` to it and rescales to a unit diagonal, C = D^-1/2 A+ D^-1/2 (quick, not
    the nearest). 'higham' finds the nearest correlation matrix in the (weighted) Frobenius norm by Higham's
    alternating projections with Dykstra's correction: from Y = A and dS = 0, repeat R = Y - dS, X = the projection of
    R on matrices with eigenvalues at least `floor`, dS = X - R, Y = X with its diagonal set to 1, until Y moves less
    than tol (relative). Higham's result has eigenvalues at least the floor within the stopping tolerance; clipping's
    rescaling can reduce its eigenvalue floor. Higham raises if the iteration limit is reached before convergence.
    conventions:
        matrix: a symmetric N x N matrix, an estimated correlation matrix that is not positive semidefinite
        method: 'clip' or 'higham'
        floor: the least eigenvalue kept, 0 for positive semidefinite (a small positive number for definite)
        weights: one positive weight per row for Higham's weighted norm (W = diag(weights)), or None for equal
        max_iterations, tol: Higham's stopping rule (100 and 1e-10 are usual); None for 'clip'
    returns: a dict: correlation (N x N); iterations (0 for 'clip'); distance (the Frobenius norm of correlation -
        matrix); min_eigenvalue
    source: Higham (2002), "Computing the nearest correlation matrix - a problem from finance", IMA Journal of
        Numerical Analysis 22(3), algorithm 3.3; Rebonato and Jaeckel (2000) (eigenvalue clipping)
    aliases: nearest correlation, Higham, Higham's algorithm, alternating projections, eigenvalue clipping
    example:
        import qf
        a = [[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]]
        qf.nearest_correlation(matrix=a, method="higham", floor=0.0, weights=None, max_iterations=100, tol=1e-10)
        # correlation = [[1, 0.69421847, 0.52586443], [0.69421847, 1, -0.24714401], [0.52586443, -0.24714401, 1]]
    """
    choice("method", method, NEAREST)
    a = _finite("matrix", matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2 or not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("matrix must be a symmetric N x N array, N at least 2")
    least = float(floor)
    if not 0.0 <= least < 1.0:
        raise ValueError("floor must lie in [0, 1)")
    w_half = np.ones(a.shape[0]) if weights is None else np.sqrt(_finite("weights", weights).ravel())
    if w_half.shape != (a.shape[0],) or np.any(~(w_half > 0)):
        raise ValueError("weights must be one positive number per row, or None")
    iterations = 0
    if method == "clip":
        if max_iterations is not None or tol is not None or weights is not None:
            raise ValueError("weights, max_iterations and tol are for 'higham': pass None")
        values, vectors = np.linalg.eigh(a)
        kept = (vectors * np.maximum(values, least)) @ vectors.T
        d = np.sqrt(np.diag(kept))
        found = kept / np.outer(d, d)
    else:
        most = integer("max_iterations", max_iterations, 1)
        stop = float(tol)
        if not np.isfinite(stop) or not stop > 0:
            raise ValueError("tol must be finite and positive")
        y, correction = a.copy(), np.zeros_like(a)
        for iterations in range(1, most + 1):
            r = y - correction
            x = _psd(r, least, w_half)
            correction = x - r
            moved = y
            y = x.copy()
            np.fill_diagonal(y, 1.0)
            if np.linalg.norm(y - moved) <= stop * np.linalg.norm(y):
                break
        else:
            raise ValueError("Higham's iteration has not converged: increase max_iterations")
        found = y
    found = (found + found.T) / 2.0
    np.fill_diagonal(found, 1.0)
    return Result({"correlation": found, "iterations": iterations, "distance": float(np.linalg.norm(found - a)),
                   "min_eigenvalue": float(np.linalg.eigvalsh(found).min())})
