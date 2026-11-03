"""VaR and ES backtests: Kupiec, Christoffersen, Basel traffic light, Acerbi-Szekely.

A hit is a period whose loss exceeded that period's VaR forecast. `returns` are gains-positive (returns or P&L),
oldest first, one per forecast. A function with several results returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.special import xlogy

from ._common import Result, choice
from .conventions import SIGNS, _finite, _tail
from .primitives import _guard

__all__ = ["es_backtest", "var_backtest"]

ES_STATISTICS = ("z1", "z2")


@_guard(level=('alpha',))
def var_backtest(hits, *, alpha):
    """VaR backtest: Kupiec POF, Christoffersen independence and conditional coverage.

    hits[t] is 1 where the loss exceeded that period's VaR. With n periods, x hits, p = 1 - alpha: LR_pof = -2 ln[(1
    - p)^(n - x) p^x / ((1 - x/n)^(n - x) (x/n)^x)] ~ chi2(1). From the transition counts n_ij (state i then j over
    the n - 1 pairs), pi01 = n01/(n00 + n01), pi11 = n11/(n10 + n11), pi = (n01 + n11)/(n - 1): LR_ind = -2 ln[(1 -
    pi)^(n00 + n10) pi^(n01 + n11) / ((1 - pi01)^n00 pi01^n01 (1 - pi11)^n10 pi11^n11)] ~ chi2(1); LR_cc = LR_pof +
    LR_ind ~ chi2(2). 0 ln 0 = 0. The Basel zone from the binomial P(X <= x): green below 0.95, yellow below 0.9999,
    red from there (250 days at 99%: green 0-4, yellow 5-9, red 10+). An ES forecast's backtest is qf.es_backtest.
    conventions:
        alpha: the VaR's confidence level, e.g. 0.99 (the expected hit rate is 1 - alpha)
    returns: a dict: observations, exceptions, expected, rate, lr_pof, p_pof, lr_ind, p_ind, lr_cc, p_cc, n00,
        n01, n10, n11, basel_zone
    source: Kupiec (1995), "Techniques for verifying the accuracy of risk measurement models", Journal of
        Derivatives 3(2); Christoffersen (1998), "Evaluating interval forecasts", International Economic Review
        39(4); Basel Committee (1996), Supervisory framework for the use of backtesting
    aliases: Kupiec, Kupiec test, proportion of failures, POF test, Christoffersen test, Christoffersen independence, conditional coverage, VaR backtest, traffic light
    example:
        import numpy as np, qf
        hits = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0] * 25)
        qf.var_backtest(hits, alpha=0.99)
    """
    h = np.asarray(hits)
    if h.ndim != 1 or h.size < 2 or not np.all((h == 0) | (h == 1)):
        raise ValueError("hits must be a 1-D sequence of 0 and 1 (or booleans), at least two long")
    h = h.astype(bool)
    p = _tail(alpha)
    n, x = h.size, int(h.sum())
    lr_pof = -2.0 * (xlogy(n - x, 1.0 - p) + xlogy(x, p) - xlogy(n - x, 1.0 - x / n) - xlogy(x, x / n))
    a, b = h[:-1], h[1:]
    n00, n01, n10, n11 = (int(np.sum(~a & ~b)), int(np.sum(~a & b)), int(np.sum(a & ~b)), int(np.sum(a & b)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n - 1)
    null = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    alternative = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    lr_ind = -2.0 * (null - alternative)
    lr_cc = lr_pof + lr_ind
    level = float(stats.binom.cdf(x, n, p))
    zone = "green" if level < 0.95 else "yellow" if level < 0.9999 else "red"
    return Result({"observations": n, "exceptions": x, "expected": n * p, "rate": x / n, "lr_pof": float(lr_pof),
                   "p_pof": float(stats.chi2.sf(lr_pof, 1)), "lr_ind": float(lr_ind),
                   "p_ind": float(stats.chi2.sf(lr_ind, 1)), "lr_cc": float(lr_cc),
                   "p_cc": float(stats.chi2.sf(lr_cc, 2)), "n00": n00, "n01": n01, "n10": n10, "n11": n11,
                   "basel_zone": zone})


def _z(x, var, es, tail, statistic):
    """Acerbi and Szekely's Z of each history (a row of x) against positive VaR and ES forecasts; NaN for Z1 where a
    history has no hit. Returns (Z, hits) per row."""
    hit = x < -var
    weighted = np.sum(np.where(hit, x / es, 0.0), axis=-1)
    count = hit.sum(axis=-1)
    if statistic == "z1":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, weighted / np.maximum(count, 1) + 1.0, np.nan), count
    return weighted / (x.shape[-1] * tail) + 1.0, count


@_guard(level=('alpha',))
def es_backtest(returns, *, var, es, alpha, statistic, sign, null_draws):
    """ES backtest (Acerbi-Szekely): Z1 over the VaR exceptions or Z2 unconditional, its null simulated from the model.

    With returns X_t, the forecasts VaR_t and ES_t as positive losses, p = 1 - alpha and the hits I_t = 1 where X_t <
    -VaR_t (N of them over T periods): Z1 = sum_t X_t I_t / ES_t / N + 1, the mean of the hits' losses against their
    ES, and Z2 = sum_t X_t I_t / (T p ES_t) + 1. Both are 0 in expectation when the model is right (Z1 given N > 0)
    and negative when it understates the risk. The p-value is the share of null_draws' statistics at or below the
    observed one: each row a history simulated from the model's own forecast laws (Z1's rows with no hit dropped).
    conventions:
        returns: the realised returns or P&L, gains positive, oldest first
        var, es: each period's VaR and ES forecast (one per period, or one number for all)
        alpha: the forecasts' confidence level, e.g. 0.975 (the tail is 1 - alpha)
        statistic: 'z1' (conditional on the VaR hits) or 'z2' (unconditional)
        sign: how var and es are given: 'loss_positive' (a loss of 0.02 as 0.02) or 'return_negative' (as -0.02)
        null_draws: M x T returns simulated under the model (row m one history, column t drawn from period t's
            law), or None for the statistic alone
    returns: a dict: statistic; exceptions (N); observations (T); p_value and null_mean (NaN without null_draws)
    source: Acerbi and Szekely (2014), "Backtesting expected shortfall", Risk 27(11), tests 1 and 2
    aliases: ES backtest, Acerbi-Szekely, Acerbi Szekely test, Z2 test, backtesting ES
    example:
        import numpy as np, qf
        r, null = np.random.default_rng(3).standard_normal(250) * 0.01, np.random.default_rng(4).standard_normal((2000, 250)) * 0.01
        # Statistic -0.54784522, p-value 0.0855, exceptions 10.
        qf.es_backtest(returns=r, var=0.0196, es=0.0234, alpha=0.975, statistic="z2", sign="loss_positive", null_draws=null)
    """
    choice("statistic", statistic, ES_STATISTICS)
    choice("sign", sign, SIGNS)
    x = _finite("returns", returns).ravel()
    tail = _tail(alpha)
    t = x.size
    flip = 1.0 if sign == "loss_positive" else -1.0
    v = flip * np.broadcast_to(_finite("var", var), x.shape).astype(float)
    e = flip * np.broadcast_to(_finite("es", es), x.shape).astype(float)
    if np.any(e <= 0) or np.any(e < v):
        raise ValueError("es must be a positive loss at least the VaR in each period (check sign)")
    found, count = _z(x, v, e, tail, statistic)
    p_value = null_mean = math.nan
    if null_draws is not None:
        draws = _finite("null_draws", null_draws)
        if draws.ndim != 2 or draws.shape[1] != t:
            raise ValueError(f"null_draws must be M x T, T = {t} the returns' length")
        null, _ = _z(draws, v, e, tail, statistic)
        null = null[np.isfinite(null)]
        if null.size == 0:
            raise ValueError("no simulated history has a VaR hit: Z1's null is empty")
        null_mean = float(null.mean())
        p_value = float(np.mean(null <= found)) if np.isfinite(found) else math.nan
    return Result({"statistic": float(found), "exceptions": int(count), "observations": t, "p_value": p_value,
                   "null_mean": null_mean})
