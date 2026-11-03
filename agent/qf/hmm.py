"""Hidden Markov regimes: filtered and smoothed probabilities, Baum-Welch fit.

Univariate Gaussian emissions; the forward-backward smoother and the Baum-Welch EM fit are both scaled so that a long
series never underflows.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from ._common import Result, choice

__all__ = ["hmm_baum_welch", "hmm_filter", "hmm_smooth"]

STOPS = ("means", "loglik")
FILTERED = ("filtered", "predicted")


def _model(x, means, stds, transition, initial):
    x = np.asarray(x, dtype=float).ravel()
    means, stds = np.asarray(means, dtype=float).ravel(), np.asarray(stds, dtype=float).ravel()
    a, p = np.asarray(transition, dtype=float), np.asarray(initial, dtype=float).ravel()
    k = means.size
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("x must hold at least two finite observations")
    if stds.size != k or a.shape != (k, k) or p.size != k:
        raise ValueError("means, stds, initial (K) and transition (K x K) must agree on the number of states")
    if np.any(~(stds > 0)) or np.any(a < 0) or np.any(p < 0):
        raise ValueError("stds must be positive, transition and initial non-negative")
    if not (np.allclose(a.sum(axis=1), 1.0) and np.isclose(p.sum(), 1.0)):
        raise ValueError("each transition row and the initial distribution must sum to 1")
    return x, means, stds, a, p


def _log_forward(log_b, a, p):
    with np.errstate(divide="ignore"):
        log_a, predicted = np.log(a), np.log(p)
    log_alpha, normalizers = np.empty_like(log_b), np.empty(log_b.shape[0])
    for t in range(log_b.shape[0]):
        if t:
            predicted = logsumexp(log_alpha[t - 1, :, None] + log_a, axis=0)
        row = predicted + log_b[t]
        normalizers[t] = logsumexp(row)
        log_alpha[t] = row - normalizers[t]
    return log_alpha, normalizers, log_b, log_a, True


def _forward(x, means, stds, a, p):
    """The scaled forward pass: each step's emission densities shifted by their largest log-density (b, shift), each
    forward vector normalised (alpha, the filtered probabilities, and scale, the normalisers)."""
    n, k = x.size, means.size
    log_b = -0.5 * np.log(2.0 * np.pi * stds ** 2) - (x[:, None] - means) ** 2 / (2.0 * stds ** 2)
    shift = log_b.max(axis=1)
    b = np.exp(log_b - shift[:, None])
    if np.any(b < np.finfo(float).tiny):
        return _log_forward(log_b, a, p)
    alpha, scale = np.empty((n, k)), np.empty(n)
    for t in range(n):
        predicted = alpha[t - 1] @ a if t else p
        alpha[t] = predicted * b[t]
        if np.any((predicted > 0) & (alpha[t] < np.finfo(float).tiny)):
            return _log_forward(log_b, a, p)
        scale[t] = alpha[t].sum()
        if not scale[t] > 0:
            raise ValueError(f"observation {t} has probability 0 under every state the model allows there")
        alpha[t] /= scale[t]
    return alpha, scale, b, shift, False


def _smooth(x, means, stds, a, p):
    """Log-likelihood, posteriors gamma (T x K) and expected transition counts (K x K, summed over t), by the scaled
    recursions: the log-likelihood the sum of the logs of the forward pass's shifts and normalisers."""
    alpha, scale, b, shift, logarithmic = _forward(x, means, stds, a, p)
    n, k = alpha.shape
    if logarithmic:
        log_alpha, normalizers, log_b, log_a = alpha, scale, b, shift
        log_beta = np.zeros_like(log_alpha)
        for t in range(n - 2, -1, -1):
            log_beta[t] = logsumexp(log_a + log_b[t + 1] + log_beta[t + 1], axis=1) - normalizers[t + 1]
        posterior = log_alpha + log_beta
        gamma = np.exp(posterior - logsumexp(posterior, axis=1, keepdims=True))
        pair = np.zeros((k, k))
        for t in range(n - 1):
            pair += np.exp(log_alpha[t, :, None] + log_a + log_b[t + 1] + log_beta[t + 1] - normalizers[t + 1])
        return float(normalizers.sum()), gamma, pair
    beta = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        beta[t] = a @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
    loglik = float(np.sum(np.log(scale)) + np.sum(shift))
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    pair = np.einsum("ti,tj->ij", alpha[:-1], b[1:] * beta[1:] / scale[1:, None]) * a
    return loglik, gamma, pair


def hmm_smooth(x, means, stds, transition, initial):
    """Smoothed state probabilities and log-likelihood of a Gaussian HMM, by the scaled forward-backward pass.
    The forward-backward pass of a K-state HMM with emissions N(means[k], stds[k]^2), transition[i, j] = P(s_{t+1}
    = j | s_t = i) and `initial` the distribution of s_1: returns (loglik, posteriors, transitions), where loglik = ln
    P(x), posteriors[t, k] = P(s_t = k | x) (smoothed, every row sums to 1) and transitions[i, j] = sum_t P(s_t = i,
    s_{t+1} = j | x). Scaled as Rabiner's section V-A scales it (each forward vector normalised, each step's emission
    densities shifted by their largest in log space), so a long series does not underflow; the result is the log-space
    recursion's to rounding. Source: Rabiner (1989), "A tutorial on hidden Markov models", Proceedings of the IEEE
    77(2), sections III-A, III-C and V-A; Hamilton, Time Series Analysis (1994), chapter 22.

    The Markov chain itself: its stationary distribution pi solves pi = pi A with sum pi = 1 (the left eigenvector of
    the transition matrix A for eigenvalue 1); a state's expected duration is 1 / (1 - A[i, i]) periods.
    aliases: HMM forward-backward, hidden Markov model smoothing, [regime probabilities], Gaussian HMM
    returns: (loglik, posteriors, transitions): a float; a T x K array, each row summing to 1; a K x K
        array of expected transition counts
    example:
        qf.hmm_smooth(x=[0.1, -0.2, 0.3, 1.1, 1.3, 0.9], means=[0.0, 1.0], stds=[0.3, 0.3],
                      transition=[[0.9, 0.1], [0.1, 0.9]], initial=[0.5, 0.5])
    """
    return _smooth(*_model(x, means, stds, transition, initial))


def hmm_baum_welch(x, means, stds, transition, initial, max_iter, tol, stop, update_initial):
    """Baum-Welch (EM) fit of a K-state HMM with univariate Gaussian emissions, from the starting values given.

    Each iteration runs hmm_smooth at the current parameters (the E-step) and re-estimates (the M-step), with gamma
    the posteriors and xi the expected transitions: transition[i, j] = xi[i, j] / sum_j xi[i, j]; means[k] =
    sum_t gamma[t, k] x_t / sum_t gamma[t, k]; stds[k]^2 = sum_t gamma[t, k] (x_t - new means[k])^2 /
    sum_t gamma[t, k]; initial = gamma[0] where `update_initial`, else kept as given. `stop` 'means' ends after the
    iteration whose update moves every mean by less than `tol` (max |new - old| < tol); 'loglik' after the one whose
    E-step log-likelihood rose by less than `tol` over the previous E-step's. At most `max_iter` iterations; the log-
    likelihood never falls from one to the next.
    conventions:
    - stop: 'means' when the task stops once every mean moves by less than tol; 'loglik' when it stops once the
      log-likelihood rises by less than tol (hmmlearn's rule)
    - update_initial: True re-estimates the initial distribution from the first posterior; False keeps it as given
    returns: a dict: means, stds, transition, initial (the fitted values; means and stds in x's units),
        iterations (the M-steps made), converged (the stop rule met within max_iter; False is not a fit), loglik
        (ln P(x)) and posteriors (T x K, rows summing to 1) from one more hmm_smooth at the fitted values.
        The labels keep the starting values' order: name a regime (bull: the higher mean) from the fitted means.
    Source: Baum, Petrie, Soules and Weiss (1970), Annals of Mathematical Statistics 41(1); Rabiner (1989),
    Proceedings of the IEEE 77(2), sections III-C and IV-A (Gaussian observation densities); Hamilton, Time Series
    Analysis (1994), chapter 22.
    aliases: Baum-Welch, HMM EM, hidden Markov model fit, regime-switching fit
    example:
        qf.hmm_baum_welch(x=[0.1, -0.2, 0.3, 1.1, 1.3, 0.9], means=[0.0, 1.0], stds=[0.3, 0.3],
                          transition=[[0.9, 0.1], [0.1, 0.9]], initial=[0.5, 0.5], max_iter=50,
                          tol=1e-8, stop="means", update_initial=True)
    """
    choice("stop", stop, STOPS)
    x, means, stds, a, p = _model(x, means, stds, transition, initial)
    if int(max_iter) < 1 or not tol > 0:
        raise ValueError("max_iter must be at least 1 and tol positive")
    iterations, converged, previous = 0, False, None
    for _ in range(int(max_iter)):
        loglik, gamma, xi = _smooth(x, means, stds, a, p)
        weight = gamma.sum(axis=0)
        new_means = gamma.T @ x / weight
        stds = np.sqrt(np.sum(gamma * (x[:, None] - new_means) ** 2, axis=0) / weight)
        a = xi / xi.sum(axis=1, keepdims=True)
        if update_initial:
            p = gamma[0].copy()
        moved = float(np.max(np.abs(new_means - means)))
        means = new_means
        iterations += 1
        if (moved < tol) if stop == "means" else (previous is not None and loglik - previous < tol):
            converged = True
            break
        previous = loglik
    loglik, gamma, _ = _smooth(x, means, stds, a, p)
    return Result({"means": means, "stds": stds, "transition": a, "initial": p, "iterations": iterations,
                   "converged": converged, "loglik": loglik, "posteriors": gamma})


def hmm_filter(x, means, stds, transition, initial, kind):
    """Filtered regime probabilities of a Gaussian HMM (no look-ahead), or the one-step-ahead predicted ones.

    The forward recursion alone: alpha_1 proportional to initial * b(x_1), alpha_t proportional to (alpha_{t-1} A) *
    b(x_t), normalised each step, so alpha_t[k] = P(s_t = k | x_1 .. x_t), the filtered probability; b(x) the normal
    densities N(means[k], stds[k]^2), A = transition. 'predicted' gives P(s_t = k | x_1 .. x_{t-1}) = alpha_{t-1} A
    (initial at t = 1), the probability before x_t is seen, which a strategy trading at t may use. The last filtered
    row is hmm_smooth's last smoothed one; earlier rows differ, the smoothed ones using later data (look-ahead).

    conventions:
    - x: the observations, oldest first (returns, say), finite
    - means, stds: one per state, the emissions' means and standard deviations, in x's units
    - transition: K x K, transition[i, j] = P(s_{t+1} = j | s_t = i), rows summing to 1; initial: P(s_1), K values
    - kind: 'filtered' (given x_1 .. x_t) or 'predicted' (given x_1 .. x_{t-1}, one step ahead)
    returns: a dict: probabilities (T x K, rows summing to 1, as kind says); next (the state probabilities for the
      period after the last observation, alpha_T A); loglik (ln P(x))
    source: Rabiner (1989), Proceedings of the IEEE 77(2), section III-A; Hamilton, Time Series Analysis (1994), 22.4
      (the filter's inference)
    aliases: HMM filter, filtered probabilities, Hamilton filter, regime filter, filtered regime probabilities, [regime probability]
    example:
        qf.hmm_filter(x=[0.1, -0.2, 0.3, 1.1, 1.3, 0.9], means=[0.0, 1.0], stds=[0.3, 0.3],
                      transition=[[0.9, 0.1], [0.1, 0.9]], initial=[0.5, 0.5],
                      kind="filtered")  # final probabilities approximately [0.001303, 0.998697]
    """
    choice("kind", kind, FILTERED)
    x, means, stds, a, p = _model(x, means, stds, transition, initial)
    alpha, scale, _, shift, logarithmic = _forward(x, means, stds, a, p)
    if logarithmic:
        alpha = np.exp(alpha)
    probabilities = alpha if kind == "filtered" else np.vstack([p, alpha[:-1] @ a])
    return Result({"probabilities": probabilities, "next": alpha[-1] @ a,
                   "loglik": float(np.sum(scale) if logarithmic else np.sum(np.log(scale)) + np.sum(shift))})
