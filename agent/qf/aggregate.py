"""Aggregate losses: compound Poisson, negative binomial or binomial by FFT or Panjer; severity laws (Pareto, Lomax).

`import qf`. The aggregate loss S = X_1 + ... + X_N, the claim count N of the (a, b, 0) class (Poisson, negative
binomial, binomial) and the claim sizes X_i i.i.d. and independent of N, on a grid of step h; VaR and ES read off its
distribution at confidence levels alpha (0.99), losses positive. Portfolio credit's CreditRisk+ (a gamma-mixed
Poisson by Panjer) is qf.creditrisk_plus on `credit_portfolio`.
"""
from __future__ import annotations

import numpy as np

from ._common import Result, choice, integer
from .primitives import _guard, _counts, discretize_distribution, panjer_recursion
from .risk import ES_RULES

__all__ = ["aggregate_loss"]

DISCRETISATIONS = ("rounding", "lower", "upper")
AGGREGATE_METHODS = ("fft", "panjer")
FREQUENCIES = ("poisson", "negative_binomial", "binomial")
COVER_POINTS = 2 ** 13          # the coverage search's grid: the cover's quantile read to about 1/2,000 of itself
COVER_PASSES = 64               # its end doubled or narrowed at most this often
COVER_TILT = 10.0               # the search's exponential tilt over its grid (Grubel and Hermesmeier 1999): the mass
#                                 past the grid's end, which an FFT wraps round onto the smallest losses, damped e^-10
COVER_MARGIN = 2.0              # a grid set here ends at this many times the cover's quantile: the FFT's own
#                                 padding to twice the support, so the mass it wraps is a fraction of 1 - cover
SET_POINTS = {"fft": 2 ** 15, "panjer": 2 ** 12}    # a grid's points where the function sets them
MOST_POINTS = {"fft": 2 ** 24, "panjer": 2 ** 16}   # the most a given step may need (Panjer's work grows as n^2)


def _var_es(grid, pmf, alphas, es_rule):
    """VaR (the smallest grid loss whose CDF reaches alpha) and ES of a distribution on a grid, per level."""
    cdf = np.cumsum(pmf)
    var, es = [], []
    for alpha in alphas:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alphas must lie strictly inside (0, 1) (confidence levels, e.g. 0.99)")
        at = int(np.searchsorted(cdf, alpha - 1e-13 * max(alpha, 1e-300)))
        at = min(at, grid.size - 1)
        level = grid[at]
        above = pmf[at + 1:] @ grid[at + 1:]
        if es_rule == "beyond":
            mass = pmf[at + 1:].sum()
            es.append(float(above / mass) if mass > 0 else float(level))
        elif es_rule == "at_or_beyond":
            es.append(float((above + pmf[at] * level) / pmf[at:].sum()))
        else:
            es.append(float((above + level * (cdf[at] - alpha)) / (1.0 - alpha)))
        var.append(float(level))
    return np.array(var), np.array(es)


@_guard(level=('alphas', 'cover'))
def aggregate_loss(severity, frequency, frequency_parameters, step, points, cover, discretisation, method, alphas,
                   es_rule):
    """Aggregate loss distribution of a compound Poisson, negative binomial or binomial sum, by FFT or Panjer; cover required.

    The claim size's law F on the grid x_j = j h, j < n, by discretize_distribution ('rounding' f_j = F((j + 1/2) h) -
    F((j - 1/2) h); 'lower', 'upper' a span's mass at its end); the count of the (a, b, 0) class. 'panjer':
    panjer_recursion; 'fft': g = ifft(P_N(fft(f))), P_N the count's generating function. E[S] = E[N] E[X], Var[S] =
    E[N] Var[X] + Var[N] E[X]^2. VaR_alpha: the smallest x with P(S <= x) >= alpha; ES 'beyond' E[S |
    S > VaR], 'at_or_beyond' E[S | S >= VaR], 'exact' the mean of VaR_u, u > alpha. numpy's pareto(a) draws the Lomax.

    n h must pass the aggregate's quantile at `cover`, else the mass past it wraps round ('fft') or is lost ('panjer'):
    a short grid is refused; a grid set here reaches twice it or beyond.

    conventions:
    - severity: a scipy.stats frozen law, e.g. gamma(a=, scale=), lognorm(s=sigma, scale=exp(mu)), pareto(b=alpha,
      scale=x_m) (from x_m), lomax(c=alpha, scale=x_m) (from 0)
    - frequency: 'poisson', 'negative_binomial' or 'binomial'; frequency_parameters: a dict, {'lambda'}, {'r',
      'beta'} (mean r beta) or {'m', 'q'}
    - step: h; points: n (a power of 2 suits 'fft'); either or both None to set the grid from cover
    - cover: required, a probability level above every alpha whose quantile n h must pass
    - discretisation: 'rounding', 'lower' or 'upper'; method: 'fft' or 'panjer'; alphas: confidence levels; es_rule:
      'beyond', 'at_or_beyond' or 'exact'
    returns: a dict: grid; pmf; cdf; var and es (one per alpha); mean and variance (the grid's); mean_exact and
      variance_exact (the law's); tail_mass (severity mass past the grid); cover_quantile; step and points
    source: Klugman, Panjer and Willmot, Loss Models (2019), 9.6 and 9.8; Panjer (1981), ASTIN Bulletin 12(1)
    aliases: aggregate loss, aggregate loss distribution, compound Poisson distribution, compound negative binomial, Panjer recursion, Panjer, (a, b, 0) class, FFT aggregate loss, loss distribution approach, [compound Poisson]
    example:
        import qf, scipy.stats as st
        qf.aggregate_loss(severity=st.lognorm(s=0.8, scale=400.0), frequency="poisson", frequency_parameters={"lambda": 12.0}, step=None, points=None, cover=0.999, discretisation="rounding", method="fft", alphas=[0.99], es_rule="beyond")
    """
    choice("frequency", frequency, FREQUENCIES)
    choice("discretisation", discretisation, DISCRETISATIONS)
    choice("method", method, AGGREGATE_METHODS)
    choice("es_rule", es_rule, ES_RULES)
    if not hasattr(severity, "cdf"):
        raise ValueError("severity must be a scipy.stats frozen distribution (with .cdf)")
    _, _, generating, count_mean, count_variance = _counts(frequency, frequency_parameters, "frequency_parameters")
    levels = [float(x) for x in np.atleast_1d(alphas)]
    level = float(cover)
    if not (0.0 < level < 1.0 and all(level > alpha for alpha in levels)):
        raise ValueError("cover must lie inside (0, 1) and above every alpha (the grid's end past the highest "
                         "level's quantile)")
    with np.errstate(all="ignore"):
        first, second = float(severity.mean()), float(severity.moment(2))
    reach = _cover_quantile(severity, generating, level, discretisation, count_mean * first)
    h, n = _grid(step, points, method, COVER_MARGIN * reach if reach > 0 else float(severity.ppf(level)))
    if n * h < reach:
        raise ValueError(f"the grid's end (points x step) reaches {n * h / reach:.2f} of the aggregate's {level:g} "
                         "quantile (cover), past which the mass wraps round ('fft') or is lost ('panjer'): widen the "
                         "grid (a larger step or more points), or pass step or points None to set it from cover")
    law = discretize_distribution(law=severity, step=h, points=n, method=discretisation)
    grid, f, tail = law["grid"], law["pmf"], law["tail_mass"]
    if method == "fft":
        g = np.maximum(np.fft.ifft(generating(np.fft.fft(f))).real, 0.0)
    else:
        g = panjer_recursion(counting=frequency, parameters=frequency_parameters, severity=f, points=n)
    var, es = _var_es(grid, g, levels, es_rule)
    mean = float(g @ grid)
    return Result({"grid": grid, "pmf": g, "cdf": np.cumsum(g), "var": var, "es": es, "mean": mean,
                   "variance": float(g @ grid ** 2 - mean * mean), "mean_exact": count_mean * first,
                   "variance_exact": count_mean * (second - first * first) + count_variance * first * first,
                   "tail_mass": tail, "cover_quantile": reach, "step": h, "points": n})


def _grid(step, points, method, end):
    """The grid's step and points: both given as they are; else set so that n h is `end`, points `SET_POINTS` where
    not given, or the least power of two that reaches `end` at the given step."""
    if step is not None and points is not None:
        return float(step), integer("points", points, 2)
    if not (np.isfinite(end) and end > 0):
        raise ValueError("cover's quantile is not a finite positive loss: give step and points")
    if points is not None:
        n = integer("points", points, 2)
        return end / n, n
    if step is None:
        return end / SET_POINTS[method], SET_POINTS[method]
    h = float(step)
    if not h > 0:
        raise ValueError("step must be positive")
    n = 2 ** max(int(np.ceil(np.log2(end / h))), 1)
    if n > MOST_POINTS[method]:
        raise ValueError("a step this small needs a grid longer than the method takes to reach twice cover's "
                         "quantile: pass a larger step, or step None")
    return h, n


def _cover_quantile(severity, generating, cover, discretisation, start):
    """The aggregate's quantile at `cover`, to about a step of a grid of `COVER_POINTS` from 0 to an end: the severity
    discretised there and tilted by e^{-theta j} (theta = COVER_TILT over the grid), the aggregate g_j e^{-theta j} =
    ifft(P_N(fft(f_j e^{-theta j}))) untilted, which leaves of the mass past the end, wrapped round, e^-COVER_TILT;
    the end doubled from `start` (E[S], or the severity's median where that is not finite) until the CDF at it
    reaches cover, then narrowed to twice the quantile while that is under a quarter of it."""
    if not (np.isfinite(start) and start > 0):
        start = float(severity.ppf(0.5))
    end = start if np.isfinite(start) and start > 0 else 1.0
    theta = COVER_TILT / COVER_POINTS
    tilt = np.exp(-theta * np.arange(COVER_POINTS))
    for _ in range(COVER_PASSES):
        h = end / COVER_POINTS
        f = discretize_distribution(law=severity, step=h, points=COVER_POINTS, method=discretisation)["pmf"]
        cdf = np.cumsum(np.maximum(np.fft.ifft(generating(np.fft.fft(f * tilt))).real / tilt, 0.0))
        if cdf[-1] < cover:
            end *= 2.0
            continue
        found = h * int(np.searchsorted(cdf, cover))
        if found == 0.0 or found > end / 4.0:
            return found
        end = 2.0 * found
    raise ValueError("cover's quantile was not reached: the severity's law may have no finite quantile there")
