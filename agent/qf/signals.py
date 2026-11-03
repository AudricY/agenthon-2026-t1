"""Signals: winsorizing, z-scoring and ranking (Barra's standardisation); idiosyncratic momentum.

`import qf`. A panel is T x N: rows the dates in order, columns the stocks, NaN missing (left out of that date's
statistics and NaN in the result). A scalar result is a float; a function with several results returns a dict.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._common import choice, integer
from .conventions import QUANTILE_METHODS

__all__ = ["cross_sectional_rank", "cross_sectional_zscore", "plain_zscore", "residual_momentum", "winsorize"]

CLIPS = ("quantile", "count", "sd")
ZSCORE_CLIPS = ("none", "quantile", "count", "sd")
SCALES = ("equal", "weighted")
TIES = ("average", "min", "max", "dense", "ordinal")
RANK_FORMS = ("rank", "percentile", "zero_one", "normal_score")
INTERCEPTS = ("dropped", "kept", "none")
AGGREGATES = ("compound", "sum", "sum_over_sd")
ZERO_VARIANCE = ("zero", "nan", "raise")


def _rows(name, values):
    """`values` as a float array of one row (N) or rows (T x N), and whether it was one row."""
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2) or array.shape[-1] < 1:
        raise ValueError(f"{name} must be N values (one date) or T x N (dates by stocks)")
    return np.atleast_2d(array), array.ndim == 1


def _ddof(ddof, needed):
    if ddof not in (0, 1) and not (ddof is None and not needed):
        raise ValueError(f"ddof must be 0 or 1{' here' if needed else ' (or None where unused)'}, not {ddof!r}")
    return ddof


def _limits(clip, limits):
    """The clip's (lower, upper) pair, checked: quantile levels, tail shares, or standard deviations."""
    if clip == "none":
        if limits is not None:
            raise ValueError("limits must be None with clip 'none'")
        return None
    try:
        lower, upper = (float(x) for x in limits)
    except (TypeError, ValueError):
        raise ValueError("limits must be a (lower, upper) pair") from None
    if clip == "quantile" and not 0.0 <= lower < upper <= 1.0:
        raise ValueError("limits for 'quantile' are two quantile levels, 0 <= lower < upper <= 1, e.g. (0.01, 0.99)")
    if clip == "count" and not (0.0 <= lower < 1.0 and 0.0 <= upper < 1.0 and lower + upper < 1.0):
        raise ValueError("limits for 'count' are the shares cut from each tail, e.g. (0.05, 0.05)")
    if clip == "sd" and not (lower > 0 and upper > 0):
        raise ValueError("limits for 'sd' are the standard deviations below and above the mean, e.g. (3.0, 3.0)")
    return lower, upper


def _clip_row(x, clip, bounds, centre, spread, quantile_method):
    """One cross-section's finite values clipped: at their quantiles, by scipy's counts, or at centre -/+ k spread."""
    lower, upper = bounds
    if clip == "quantile":
        low, high = np.quantile(x, [lower, upper], method=quantile_method)
    elif clip == "count":
        ordered, n = np.sort(x), x.size
        cut_low, cut_high = int(np.floor(lower * n)), int(np.floor(upper * n))
        low, high = ordered[cut_low], ordered[n - cut_high - 1]
    else:
        low, high = centre - lower * spread, centre + upper * spread
    return np.clip(x, low, high)


def _quantile_method(clip, quantile_method):
    if clip == "quantile":
        choice("quantile_method", quantile_method, QUANTILE_METHODS)
    elif quantile_method is not None:
        raise ValueError("quantile_method must be None unless clip is 'quantile'")


def winsorize(values, clip, limits, ddof, quantile_method):
    """Winsorize: each series or cross-section clipped at percentile cutoffs (numpy's or scipy's rule) or at k sd.

    Each row (a date's stocks) is clipped on its own, a 1-d input as one series: 'quantile' sets values below the
    lower quantile to it and above the upper to it (the named numpy rule, 'linear' matches pandas); 'count'
    replaces the floor(share n) smallest and largest values by the next one inward (scipy.stats.mstats.winsorize);
    'sd' clips at mean - lower sd and mean + upper sd. Values keep their units (no standardising: cross_sectional_zscore
    clips and standardises).

    conventions:
    - values: N values or T x N (rows dates, columns stocks); NaN left out and kept NaN
    - clip: 'quantile' (limits are quantile levels: (0.01, 0.99) for the 1st and 99th percentiles), 'count' (limits are
      the shares cut from each tail, scipy's: (0.05, 0.05)) or 'sd' (limits are standard deviations: (3.0, 3.0))
    - limits: the (lower, upper) pair in the clip's terms
    - ddof: the 'sd' clip's standard deviation, 0 (numpy) or 1 (pandas); None with 'quantile' or 'count'
    - quantile_method: 'linear' interpolates; 'lower', 'higher', 'nearest' choose an observation; 'midpoint' averages
      adjacent ones; 'hazen' or 'weibull' use their plotting positions; 'inverted_cdf' takes the empirical quantile;
      None unless clip is 'quantile'
    returns: an array shaped like values
    source: Tukey (1962) and Dixon (1960) on winsorizing; scipy.stats.mstats.winsorize; Grinold and Kahn (2000) ch. 14
    aliases: Winsorize, Winsorized, Winsorizing, Winsorization, winsorized values, clip outliers, [trim outliers], [outlier clipping]
    example:
        # Returns [[0.5, -1.0, 0.3, 7.025, -0.4, 1.1]].
        qf.winsorize(values=[[0.5, -1.2, 0.3, 9.0, -0.4, 1.1]], clip="quantile", limits=(0.05, 0.95), ddof=None,
                     quantile_method="linear")
    """
    choice("clip", clip, CLIPS)
    _quantile_method(clip, quantile_method)
    bounds = _limits(clip, limits)
    _ddof(ddof, clip == "sd")
    rows, one = _rows("values", values)
    found = rows.copy()
    for row in range(rows.shape[0]):
        kept = np.isfinite(rows[row])
        x = rows[row, kept]
        if x.size < 2:
            continue
        spread = np.std(x, ddof=ddof) if clip == "sd" else 0.0
        found[row, kept] = _clip_row(x, clip, bounds, x.mean(), spread, quantile_method)
    return found[0] if one else found


def _standardise(x, w, scale, ddof):
    if not np.sum(w) > 0:
        raise ValueError("a date's weights must have a positive sum")
    centre = np.sum(w * x) / np.sum(w)
    if scale == "equal":
        spread = np.std(x, ddof=ddof)
    else:
        spread = np.sqrt(np.sum(w * (x - centre) ** 2) / np.sum(w))
    if not spread > 0:
        raise ValueError("a date's values do not vary: no z-score (qf.plain_zscore sets such a date to 0 or NaN)")
    return centre, spread


def cross_sectional_zscore(signal, weights, scale, ddof, clip, limits, rounds, quantile_method):
    """Cross-sectional z-scores after clipping outliers: Barra's descriptor standardisation, two passes.

    Each row (a date's stocks): m = sum w x / sum w (equal weights, or cap or root-cap), s the plain std of the values
    ('equal') or sqrt(sum w (x - m)^2 / sum w) ('weighted'), z = (x - m)/s, so the weighted mean of z is 0. A round
    clips the values (at their quantiles, by counts, or at m -/+ k s) and standardises them.

    Composite: z-score each signal, add them by the weights, z-score the sum. Value signals are ratios to price:
    book-to-price is book equity over market cap, earnings yield earnings over price.

    conventions:
    - signal: N values or T x N (rows dates, columns stocks); NaN left out and kept NaN
    - weights: 'equal', or T x N (or N) cap or root-cap weights for the mean (Barra's cap-weighted mean)
    - scale: 'equal' (Barra's equal-weighted sd) or 'weighted'; ddof: 0 or 1 with 'equal', 0 with 'weighted'
    - clip: 'none', 'quantile', 'count' or 'sd', limits as winsorize's (None with 'none')
    - rounds: 1 clips then standardises ('sd': z-score, truncate at -/+k, z-score again, Barra's two passes); 2 again
    - quantile_method: 'linear' interpolates; 'lower', 'higher', 'nearest' choose an observation; 'midpoint' averages
      adjacent ones; 'hazen' or 'weibull' use their plotting positions; 'inverted_cdf' takes the empirical quantile;
      None unless clip is 'quantile'
    returns: an array shaped like signal
    source: Grinold and Kahn (2000) ch. 14; MSCI Barra USE4 and CNE6 handbooks (descriptor standardisation)
    aliases: cross-sectional z-score, [cross-sectional standardization], standardize exposures, Barra standardization, z-scoring, [z-score], [zscore]
    example:
        # Returns [[0.104197, -1.414104, -0.074427, 1.443874, -0.699609, 0.640068]].
        qf.cross_sectional_zscore(signal=[[0.5, -1.2, 0.3, 2.0, -0.4, 1.1]], weights="equal", scale="equal",
                                  ddof=1, clip="sd", limits=(3.0, 3.0), rounds=1, quantile_method=None)
    """
    choice("scale", scale, SCALES)
    choice("clip", clip, ZSCORE_CLIPS)
    _quantile_method(clip, quantile_method)
    bounds = _limits(clip, limits)
    _ddof(ddof, True)
    if scale == "weighted" and ddof != 0:
        raise ValueError("ddof must be 0 with scale 'weighted' (the weighted variance divides by the weights' sum)")
    times = integer("rounds", rounds, 1)
    rows, one = _rows("signal", signal)
    if isinstance(weights, str):
        choice("weights", weights, ("equal",))
        w = np.ones_like(rows)
    else:
        w = np.broadcast_to(np.atleast_2d(np.asarray(weights, dtype=float)), rows.shape)
        if np.any(w[np.isfinite(w)] < 0):
            raise ValueError("weights must not be negative")
    found = np.full(rows.shape, np.nan)
    for row in range(rows.shape[0]):
        kept = np.isfinite(rows[row]) & np.isfinite(w[row])
        x, weight = rows[row, kept], w[row, kept]
        if x.size < 2:
            continue
        for _ in range(times):
            if clip != "none":
                centre, spread = _standardise(x, weight, scale, ddof) if clip == "sd" else (0.0, 0.0)
                x = _clip_row(x, clip, bounds, centre, spread, quantile_method)
            centre, spread = _standardise(x, weight, scale, ddof)
            x = (x - centre) / spread
        found[row, kept] = x
    return found[0] if one else found

def plain_zscore(signal, ddof, zero_variance):
    """Cross-sectional z-scores each date, ddof named; a zero-variance date gives 0, NaN or a refusal, as named.

    Each row (a date's stocks) on its own over its finite values: m their mean, s their standard deviation with ddof,
    z = (x - m) / s; NaN stays NaN. A date whose values are all equal (s = 0 in exact arithmetic, whatever rounding
    leaves in numpy's std) or that holds no more values than ddof gets zero_variance's rule. No clipping and no
    weights: Barra's two-pass standardisation is qf.cross_sectional_zscore, which refuses such a date.

    conventions:
    - zero_variance: 'zero' (every z of that date 0.0), 'nan' (every z NaN) or 'raise' (a ValueError naming the date)
    - ddof: 0 (the population sd, numpy's and a task's "ddof=0") or 1 (the sample sd, pandas' std)
    - signal: N values (one date) or T x N (rows dates, columns stocks); NaN left out and kept NaN
    returns: an array shaped like signal
    source: Grinold and Kahn, Active Portfolio Management (2000), ch. 14 (standardised exposures)
    aliases: zero-variance z-score, z-score with zero variance, constant cross-section z-score, plain z-score, [standardize signal]
    example:
        # Returns [[-1.224745, 0.0, 1.224745], [0.0, 0.0, 0.0]]: the second date does not vary.
        qf.plain_zscore(signal=[[1.0, 2.0, 3.0], [0.4, 0.4, 0.4]], ddof=0, zero_variance="zero")
    """
    choice("zero_variance", zero_variance, ZERO_VARIANCE)
    _ddof(ddof, True)
    rows, one = _rows("signal", signal)
    found = np.full(rows.shape, np.nan)
    for row in range(rows.shape[0]):
        kept = np.isfinite(rows[row])
        x = rows[row, kept]
        if x.size == 0:
            continue
        if x.size <= ddof or np.max(x) == np.min(x):
            if zero_variance == "raise":
                raise ValueError(f"row {row}'s values do not vary: no z-score (zero_variance 'zero' or 'nan' sets it)")
            found[row, kept] = 0.0 if zero_variance == "zero" else np.nan
            continue
        found[row, kept] = (x - x.mean()) / np.std(x, ddof=ddof)
    return found[0] if one else found


def cross_sectional_rank(signal, ties, form):
    """Cross-sectional ranks: 1 to N, percentile ranks or normal scores (scipy's rankdata).

    Each row (a date's stocks) on its own, the smallest value ranked 1: ties by scipy.stats.rankdata's method. With R
    the number of names (the largest rank under 'dense'): 'percentile' rank / R (pandas' rank(pct=True)), 'zero_one'
    (rank - 1) / (R - 1), 'normal_score' the standard normal quantile of (rank - 0.5) / R (a rank-based z-score).

    The cross-sectional correlation of two signals each date: Spearman's is the Pearson correlation of their ranks
    (the rank IC; information_coefficient for a signal against forward returns).

    conventions:
    - signal: N values or T x N (rows dates, columns stocks); NaN left out and kept NaN
    - ties: 'average' (pandas' and scipy's default), 'min', 'max', 'dense' or 'ordinal' (pandas' 'first', by position)
    - form: 'rank', 'percentile', 'zero_one' or 'normal_score'
    returns: an array shaped like signal
    source: Grinold and Kahn (2000) ch. 14 (scores from ranks); scipy.stats.rankdata; pandas DataFrame.rank
    aliases: cross-sectional rank, rank transform, percentile rank, rank scores, normal scores, [rank], [ranks], [percentile]
    example:
        # Returns [[2/3, 1/6, 1/2, 1, 1/3, 5/6]].
        qf.cross_sectional_rank(signal=[[0.5, -1.2, 0.3, 2.0, -0.4, 1.1]], ties="average", form="percentile")
    """
    choice("ties", ties, TIES)
    choice("form", form, RANK_FORMS)
    rows, one = _rows("signal", signal)
    found = np.full(rows.shape, np.nan)
    for row in range(rows.shape[0]):
        kept = np.isfinite(rows[row])
        if not kept.any():
            continue
        rank = stats.rankdata(rows[row, kept], method=ties)
        top = rank.max() if ties == "dense" else float(rank.size)
        if form == "percentile":
            rank = rank / top
        elif form == "zero_one":
            rank = (rank - 1.0) / (top - 1.0) if top > 1 else np.full_like(rank, np.nan)
        elif form == "normal_score":
            rank = stats.norm.ppf((rank - 0.5) / top)
        found[row, kept] = rank
    return found[0] if one else found


def residual_momentum(returns, factors, risk_free, regression_window, formation_window, skip, intercept, aggregate,
                      ddof):
    """Residual momentum: factor-model residuals summed or compounded over a formation period, the latest skipped.

    Row t uses data through row t: each stock's excess return is regressed on the factors over the regression_window
    rows ending at t; its residuals e over the formation_window rows ending skip rows before t give 'compound'
    prod(1 + e) - 1, 'sum' sum e or 'sum_over_sd' sum e / std(e) (Blitz, Huij and Martens). Intercept 'dropped': e =
    r - a - b'f (Gutierrez and Pirinsky eq. 2); 'kept': e = r - b'f; 'none': no intercept fitted.

    With no factors, 'none' and 'compound' it is price momentum: 12-1 momentum is formation 11, skip 1 on months.

    conventions:
    - returns: T x N simple returns, rows dates, columns stocks (a NaN in a window gives NaN); risk_free: subtracted
      first, a scalar or T rates (0.0 for excess returns)
    - factors: T x K factor returns (T for one, None for none)
    - regression_window (36 months in Blitz, Huij and Martens), formation_window (11), skip (1): rows
    - intercept: 'dropped', 'kept' or 'none'; aggregate: 'compound', 'sum' or 'sum_over_sd'
    - ddof: the 'sum_over_sd' std's, 0 or 1; None otherwise
    returns: an array of scores, T x N, NaN where a window is short
    source: Blitz, Huij and Martens (2011), J. Empirical Finance 18(3); Gutierrez and Pirinsky (2007), J. Financial Markets 10(1)
    aliases: residual momentum signal, idiosyncratic momentum, factor-neutral momentum, [residual momentum], [momentum], [12-1 momentum], [residual returns]
    example:
        # Last row [-0.0001, 0.009925]; first three rows NaN.
        qf.residual_momentum(returns=[[0.02, -0.01], [0.01, 0.03], [-0.02, 0.01], [0.03, 0.02], [0.0, -0.02]],
                             factors=[0.01, 0.02, -0.01, 0.02, -0.01], risk_free=0.0, regression_window=4,
                             formation_window=2, skip=1, intercept="dropped", aggregate="compound", ddof=None)
    """
    choice("intercept", intercept, INTERCEPTS)
    choice("aggregate", aggregate, AGGREGATES)
    _ddof(ddof, aggregate == "sum_over_sd")
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be T x N (rows dates, columns stocks)")
    bars = r.shape[0]
    f = np.zeros((bars, 0)) if factors is None else np.asarray(factors, dtype=float)
    f = f[:, None] if f.ndim == 1 else f
    if f.ndim != 2 or f.shape[0] != bars or not np.all(np.isfinite(f)):
        raise ValueError("factors must be T x K finite factor returns over the returns' rows (None for none)")
    rf = np.asarray(risk_free, dtype=float)
    if rf.ndim and rf.shape != (bars,):
        raise ValueError("risk_free must be a scalar or one rate per row")
    excess = r - (rf[:, None] if rf.ndim else rf)
    window = integer("regression_window", regression_window, 2)
    formation = integer("formation_window", formation_window, 1)
    gap = integer("skip", skip, 0)
    if formation + gap > window:
        raise ValueError("the formation window and the skip must lie inside the regression window")
    design = np.hstack([np.ones((bars, 1))] * (intercept != "none") + [f])
    if design.shape[1] >= window:
        raise ValueError("the regression window must hold more rows than the regression's coefficients")
    found = np.full(r.shape, np.nan)
    for t in range(window - 1, bars):
        rows = slice(t - window + 1, t + 1)
        y, x = excess[rows], design[rows]
        complete = np.all(np.isfinite(y), axis=0)
        if not complete.any():
            continue
        if x.shape[1]:
            coef = np.linalg.lstsq(x, y[:, complete], rcond=None)[0]
            resid = y[:, complete] - x @ coef
            if intercept == "kept":
                resid = resid + coef[0]
        else:
            resid = y[:, complete]
        e = resid[window - gap - formation:window - gap]
        if aggregate == "compound":
            score = np.prod(1.0 + e, axis=0) - 1.0
        elif aggregate == "sum":
            score = e.sum(axis=0)
        else:
            spread = e.std(axis=0, ddof=ddof)
            with np.errstate(divide="ignore", invalid="ignore"):
                score = np.where(spread > 0, e.sum(axis=0) / spread, np.nan)
        found[t, complete] = score
    return found
