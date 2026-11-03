"""First passage of Brownian motion with drift: probability, density, mean, running maximum.

X_t = drift * t + vol * W_t, X_0 = 0. For a stock S_t = S_0 e^{X_t} under the risk-neutral measure the drift is
r - q - vol^2/2 and a price level H is the level ln(H / S_0); under the real-world measure it is mu - vol^2/2.
Continuous monitoring throughout.
"""
from __future__ import annotations

import numpy as np

from ._common import log_ndtr, ndtr, out, positive

__all__ = ["first_passage_mean", "first_passage_pdf", "hit_probability", "terminal_and_max_cdf"]


def _oriented(level, drift):
    """|level| and the drift toward it: a level below 0 is the running minimum's, read as the maximum of -X."""
    level, drift = np.asarray(level, dtype=float), np.asarray(drift, dtype=float)
    side = np.where(level < 0, -1.0, 1.0)
    return np.abs(level), side * drift


def hit_probability(level, horizon, drift, vol):
    """P(X reaches `level` by `horizon`):
    for level b > 0, P(max_{s<=T} X_s >= b); for level a < 0, P(min_{s<=T} X_s <= a); 1 at level 0. Equally the
    first-passage time's CDF, P(tau_level <= T).

    P(M_T >= b) = N((-b + drift T)/(vol sqrt T)) + e^{2 drift b / vol^2} N((-b - drift T)/(vol sqrt T)),
    P(m_T <= a) = N((a - drift T)/(vol sqrt T)) + e^{2 drift a / vol^2} N((a + drift T)/(vol sqrt T)).
    `horizon` in the units drift and vol are annualised in (years for annual rates). The exponential term is taken in
    log space, so a large drift b / vol^2 does not overflow.
    conventions:
        level: ln(H / S_0) for a price level H; drift: X's, r - q - vol^2/2 risk-neutral (the -vol^2/2 is in already)
    returns: for b > 0, P(max X >= b) = N((-b + drift T)/(vol sqrt T)) + e^{2 drift b / vol^2} N((-b - drift T)/(vol
        sqrt T)): the reflected term added, its argument -b - drift T; a float, or an array
    Source: the reflection principle; Shreve, Stochastic Calculus for Finance II (2004), section 7.2 (the maximum of
    Brownian motion with drift); Karatzas and Shreve, Brownian Motion and Stochastic Calculus (1991), 3.5.C.
    aliases: first passage probability, hitting probability, barrier hitting probability, running maximum probability, running minimum probability, running minimum CDF, first passage time CDF
    example:
        qf.hit_probability(level=0.1, horizon=1.0, drift=0.02, vol=0.2)
    """
    b, mu = _oriented(level, drift)
    t, v = positive("horizon", horizon), positive("vol", vol)
    sd = v * np.sqrt(t)
    hit = ndtr((-b + mu * t) / sd) + np.exp(2.0 * mu * b / (v * v) + log_ndtr((-b - mu * t) / sd))
    return out(np.where(b == 0, 1.0, np.minimum(hit, 1.0)))


def first_passage_pdf(level, t, drift, vol):
    """Density at time t of the first time X reaches `level` (either sign), the inverse Gaussian:

    f(t) = |level| / (vol sqrt(2 pi t^3)) exp(-(|level| - d t)^2 / (2 vol^2 t)),  d = drift toward the level
    (drift for level > 0, -drift for level < 0); 0 for t <= 0. With d < 0 it integrates to e^{2 d |level| / vol^2} < 1
    (the level may never be reached); its integral to T is hit_probability(level, T, drift, vol).
    returns: the density per unit of time at t (0 for t <= 0); level, drift and vol as hit_probability's
    Source: Shreve, Stochastic Calculus for Finance II (2004), section 7.2; Karlin and Taylor, A First Course in
    Stochastic Processes (1975), the inverse Gaussian first-passage law.
    aliases: first passage density, first-passage time density, inverse Gaussian density, hitting time density
    example:
        qf.first_passage_pdf(level=0.1, t=[0.5, 1.0], drift=0.02, vol=0.2)
    """
    b, mu = _oriented(level, drift)
    t, v = np.asarray(t, dtype=float), positive("vol", vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = b / (v * np.sqrt(2.0 * np.pi * t ** 3)) * np.exp(-((b - mu * t) ** 2) / (2.0 * v * v * t))
    return out(np.where(t > 0, dens, 0.0))


def first_passage_mean(level, drift, vol):
    """E[first time X reaches `level`] = |level| / d
    where the drift toward the level d (drift for level > 0, -drift for level < 0) is positive; infinite where d <= 0
    (with d = 0 the level is reached surely but in infinite mean time, with d < 0 possibly never). `vol` does not
    enter the mean and is checked positive only.
    returns: the expected time in drift's unit; numpy.inf where the drift points away from the level
    Source: the inverse Gaussian's mean |level| / d; Karlin and Taylor, A First Course in Stochastic Processes (1975).
    aliases: expected first passage time, mean hitting time, expected hitting time
    example:
        qf.first_passage_mean(level=0.1, drift=0.05, vol=0.2)
    """
    b, mu = _oriented(level, drift)
    positive("vol", vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = b / mu
    return out(np.where(b == 0, 0.0, np.where(mu > 0, mean, np.inf)))


def terminal_and_max_cdf(x, m, horizon, drift, vol):
    """The joint law of the terminal value and the running maximum: P(X_T <= x and max_{s<=T} X_s <= m).

    For m >= 0 and x <= m: N((x - drift T)/(vol sqrt T)) - e^{2 drift m / vol^2} N((x - 2m - drift T)/(vol sqrt T));
    x > m is read as x = m (then it is P(M_T <= m)); for m < 0 the expression is <= 0 (it decreases in x from 0) and
    the result is 0. The minimum's twin by symmetry:
    P(X_T >= x and min X >= a) = terminal_and_max_cdf(-x, -a, T, -drift, vol).
    conventions:
        x, m: bounds in X's units (ln(K / S_0), ln(H / S_0)); drift: r - q - vol^2/2 risk-neutral, not subtracted twice
    returns: P(X_T <= x and max X <= m) = N((x - drift T)/(vol sqrt T)) - e^{2 drift m / vol^2} N((x - 2m - drift
        T)/(vol sqrt T)) for m >= 0, x <= m: the reflected term subtracted, its argument x - 2m - drift T (not 2m - x
        - drift T); x >= m gives P(max X <= m); 0 for m < 0. P(X_T <= x, max X > m) = N((x - drift T)/(vol sqrt T))
        less this.
    Source: the reflection principle; Shreve, Stochastic Calculus for Finance II (2004), section 7.2 (the joint density
    of the maximum and the terminal value, with drift by Girsanov).
    aliases: joint distribution of terminal value and maximum, running maximum joint distribution, terminal and running max, joint probability of the terminal value and the maximum, probability the maximum stays below, joint distribution of the terminal value and the running maximum, running maximum CDF
    example:
        qf.terminal_and_max_cdf(x=0.0, m=0.1, horizon=1.0, drift=0.02, vol=0.2)
    """
    x, m = np.asarray(x, dtype=float), np.asarray(m, dtype=float)
    mu, t, v = np.asarray(drift, dtype=float), positive("horizon", horizon), positive("vol", vol)
    sd = v * np.sqrt(t)
    top = np.minimum(x, m)
    joint = ndtr((top - mu * t) / sd) - np.exp(2.0 * mu * m / (v * v) + log_ndtr((top - 2.0 * m - mu * t) / sd))
    return out(np.maximum(joint, 0.0))
