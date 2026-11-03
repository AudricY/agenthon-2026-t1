"""Factor research: Fama-MacBeth (Shanken), GRS and Pesaran-Yamagata tests, double sorts, Barra WLS, IC, neutralising.

`import qf`. A panel is T x N: rows the dates in order, columns the assets, NaN missing. A scalar result is a
float; a function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import f as f_dist
from scipy.stats import norm

from ._common import Result, choice, integer
from .conventions import QUANTILE_METHODS
from .econometrics import _hac, _lags
from .metrics import _ddof

__all__ = ["cross_sectional_wls", "double_sort", "fama_macbeth", "grs_test", "hda_test", "information_coefficient",
           "neutralize", "quantile_portfolios", "rank_bins", "split_sample_jackknife"]

GRS_COVARIANCES = ("mle", "unbiased")
BREAKPOINTS = ("quantile", "rank")
IC_METHODS = ("spearman", "pearson")
RETURN_KINDS = ("simple", "log")
T_STATS = ("iid", "newey_west")
EIV_CORRECTIONS = ("none", "shanken", "shanken_multiplier")
SORTS = ("independent", "sequential")
HDA_VERSIONS = ("normal", "student_t", "pesaran_yamagata", "pesaran_yamagata_2017")
JACKKNIFE_METHODS = ("jackknife", "instrumented", "cross")
SPLITS = ("halves", "odd_even")
BIN_ALLOCATIONS = ("floor", "ceil")


def fama_macbeth(y, X, intercept, lags, denominator, correction, factor_covariance):
    """Factor risk premia by Fama-MacBeth: cross-sectional OLS each period, the slopes' mean, error and t.

    Each period t regresses y_t (N assets) on X_t, giving lambda_t; coef = mean_t lambda_t; its variance is
    Newey-West's on the series lambda_t about its mean (the regression on a constant, k = 1): V = S / T^2 with
    S = sum_t u_t^2 + 2 sum_{l=1}^{L} (1 - l/(L+1)) sum_{t>l} u_t u_{t-l}, u_t = lambda_t - coef, times T / (T - 1) for
    'T-k'. Each period drops the assets with a NaN; a period with no more assets than columns is left out.

    Shanken's errors-in-variables correction, where X holds betas on factors of covariance S_f (per period): c =
    lambda' S_f^{-1} lambda over the factor premia, V_shanken = (1 + c)(V - S_f / T) + S_f / T (Shanken 1992; Cochrane
    12.2), the intercept's S_f entries 0; 'shanken_multiplier' is (1 + c) V alone.

    conventions:
    - y: (T, N) returns, period by asset; X: (T, N, k) characteristics or betas, or (T, N) for one
    - a long panel (a row per date and asset) pivots each column to dates x assets (DataFrame.pivot), X's k
      columns stacked last; a period under a task's own minimum count is set to NaN first
    - lags: the Bartlett lags of the slopes' series (0 for the plain error)
    - denominator: 'T' (the slopes' variance over T, Cochrane's 1/T) or 'T-k' (over T - 1: std(ddof=1) / sqrt(T))
    - intercept: True puts a ones column first in every cross-section
    - correction: 'none', 'shanken' or 'shanken_multiplier'; factor_covariance: None with 'none', else S_f (k x k,
      the factors' covariance per period, in X's column order)
    returns: a dict: coef, se and tstat (one per column, se and tstat corrected); slopes (the (T, p) per-period
      estimates, NaN where a period is left out); n_periods (the periods used); multiplier (1 + c; 1.0 for 'none')
    source: Fama and MacBeth (1973), J. Political Economy 81(3); Shanken (1992), RFS 5(1); Cochrane, Asset Pricing (2005), 12.2-12.3
    aliases: Fama-MacBeth, Fama MacBeth, [cross-sectional regression], FM regression, two-pass regression, Shanken correction, [risk premia]
    example:
        qf.fama_macbeth(y=[[0.01, 0.02, 0.03], [0.0, 0.01, -0.01], [0.02, 0.03, 0.05]],
                        X=[[0.5, 1.0, 1.5], [0.6, 1.0, 1.4], [0.5, 0.9, 1.6]],
                        intercept=True, lags=0, denominator="T", correction="none", factor_covariance=None)
    """
    choice("correction", correction, EIV_CORRECTIONS)
    ys = np.asarray(y, dtype=float)
    xs = np.asarray(X, dtype=float)
    xs = xs[..., None] if xs.ndim == 2 else xs
    if ys.ndim != 2 or xs.ndim != 3 or xs.shape[:2] != ys.shape:
        raise ValueError("y must be (T, N) and X (T, N, k) or (T, N) over the same periods and assets")
    if not isinstance(intercept, (bool, np.bool_)):
        raise ValueError(f"intercept must be True or False, not {intercept!r}")
    lags = _lags(lags)
    k = xs.shape[2]
    if correction == "none":
        if factor_covariance is not None:
            raise ValueError("factor_covariance must be None with correction 'none'")
    else:
        cov_f = np.atleast_2d(np.asarray(factor_covariance, dtype=float))
        if cov_f.shape != (k, k) or not np.all(np.isfinite(cov_f)):
            raise ValueError(f"factor_covariance must be the factors' {k} x {k} covariance, in X's column order")
    p = k + intercept
    slopes = np.full((ys.shape[0], p), np.nan)
    for t in range(ys.shape[0]):
        keep = np.isfinite(ys[t]) & np.all(np.isfinite(xs[t]), axis=1)
        if keep.sum() > p:
            design = np.column_stack([np.ones(keep.sum()), xs[t][keep]]) if intercept else xs[t][keep]
            if np.linalg.matrix_rank(design) == p:
                slopes[t] = np.linalg.lstsq(design, ys[t][keep], rcond=None)[0]
    used = slopes[np.all(np.isfinite(slopes), axis=1)]
    if used.shape[0] < 2:
        raise ValueError("fewer than two periods have a cross-section to fit")
    coef = used.mean(axis=0)
    ones = np.ones((used.shape[0], 1))
    variance = np.array([_hac(ones, used[:, j] - coef[j], np.array([[1.0 / used.shape[0]]]), lags,
                              denominator)[0, 0] for j in range(p)])
    multiplier = 1.0
    if correction != "none":
        premia = coef[int(intercept):]
        multiplier = 1.0 + float(premia @ np.linalg.solve(cov_f, premia))
        own = np.concatenate([np.zeros(int(intercept)), np.diag(cov_f)]) / used.shape[0]
        variance = multiplier * (variance - own) + own if correction == "shanken" else multiplier * variance
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.where(variance >= 0, variance, np.nan))
    return Result({"coef": coef, "se": se, "tstat": coef / se, "slopes": slopes, "n_periods": int(used.shape[0]),
                   "multiplier": multiplier})


def grs_test(asset_excess_returns, factor_excess_returns, covariance):
    """Gibbons-Ross-Shanken F-test that N assets' alphas on L factors are jointly zero.

    Each asset's time-series OLS on the factors gives alpha (N,) and residuals e (T, N); mu = the factors' mean.
    'mle': W = (T - N - L)/N * alpha' Sigma^{-1} alpha / (1 + mu' Omega^{-1} mu), Sigma = e'e / T, Omega the factors'
    covariance over T (GRS 1989; Cochrane 12.6), exactly F(N, T - N - L) under normal errors.
    'unbiased': Sigma over T - L - 1, Omega over T - 1, W = (T/N)(T - N - L)/(T - L - 1) * alpha' Sigma^{-1} alpha /
    (1 + mu' Omega^{-1} mu). p_value = P(F > W).

    conventions:
    - covariance: 'mle' (the exact-F form) or 'unbiased' (the form with the degrees-of-freedom estimators)
    - asset_excess_returns: (T, N), factor_excess_returns: (T, L) or (T,), excess of the risk-free rate, same periods
    - needs T > N + L; NaN refused
    returns: a dict: statistic (W); p_value; alpha (one per asset, per period); df (N, T - N - L)
    source: Gibbons, Ross and Shanken (1989), Econometrica 57(5); Cochrane, Asset Pricing (2005), 12.1
    aliases: GRS test, GRS statistic, Gibbons Ross Shanken, joint alpha test, GRS F-test, alphas jointly zero
    example:
        qf.grs_test(asset_excess_returns=[[0.01, 0.02], [0.0, -0.01], [0.03, 0.02], [-0.01, 0.0],
                                          [0.02, 0.01], [0.01, 0.03]],
                    factor_excess_returns=[0.01, 0.0, 0.02, -0.01, 0.01, 0.01], covariance="mle")
    """
    choice("covariance", covariance, GRS_COVARIANCES)
    r = np.asarray(asset_excess_returns, dtype=float)
    f = np.asarray(factor_excess_returns, dtype=float)
    r = r[:, None] if r.ndim == 1 else r
    f = f[:, None] if f.ndim == 1 else f
    if r.ndim != 2 or f.ndim != 2 or r.shape[0] != f.shape[0]:
        raise ValueError("asset_excess_returns must be (T, N) and factor_excess_returns (T, L) over the same T")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(f))):
        raise ValueError("asset_excess_returns and factor_excess_returns must be finite")
    t, n = r.shape
    lf = f.shape[1]
    if t <= n + lf:
        raise ValueError(f"GRS needs more periods than assets plus factors (T = {t}, N = {n}, L = {lf})")
    design = np.column_stack([np.ones(t), f])
    coef = np.linalg.lstsq(design, r, rcond=None)[0]
    alpha = coef[0]
    resid = r - design @ coef
    mu = f.mean(axis=0)
    centred = f - mu
    if covariance == "mle":
        sigma, omega, scale = resid.T @ resid / t, centred.T @ centred / t, (t - n - lf) / n
    else:
        sigma, omega = resid.T @ resid / (t - lf - 1), centred.T @ centred / (t - 1)
        scale = (t / n) * (t - n - lf) / (t - lf - 1)
    stat = scale * (alpha @ np.linalg.solve(sigma, alpha)) / (1.0 + mu @ np.linalg.solve(omega, mu))
    return Result({"statistic": float(stat), "p_value": float(f_dist.sf(stat, n, t - n - lf)), "alpha": alpha,
                   "df": (n, t - n - lf)})


def _groups(x, n_groups, breakpoints):
    """Each value's group, 0 the lowest: right-closed bins at the values' (or their ordinal ranks') n-quantiles."""
    keys = x if breakpoints == "quantile" else np.argsort(np.argsort(x, kind="stable"), kind="stable") + 1.0
    edges = np.quantile(keys, np.linspace(0.0, 1.0, n_groups + 1))
    if np.any(np.diff(edges) <= 0):
        raise ValueError("the signal's quantile edges repeat (ties): breakpoints 'rank' splits tied values by order")
    return np.searchsorted(edges[1:-1], keys, side="left")


def quantile_portfolios(signal, returns, n_groups, breakpoints, weighting):
    """Quantile (quintile, decile) sort on a signal: group returns and the long-short spread.

    Each period (row) the assets with a signal and a return are split into n groups at the signal's n-quantiles,
    right-closed bins, group 0 the lowest ('quantile': pandas.qcut(signal, n); 'rank': qcut of the ordinal ranks, ties
    taken in order, so the counts differ by one at most); a group's return is its assets' mean return, equal- or
    value-weighted; the spread is the top group's less the bottom's. Its weights hold the top group long (summing to
    +1) and the bottom short (-1). The signal is known before the returns it sorts (the caller aligns them).

    conventions:
    - signal, returns: arrays shaped T x N (or N for one period), assets on the last axis; NaN leaves an asset out
    - breakpoints: 'quantile' (the signal's own quantiles; tied edges refused) or 'rank' (equal counts)
    - weighting: 'equal', or an array shaped like signal (market values at formation) for value weights
    returns: a dict: group_returns (T x n, or n), spread (T, or a float), groups (each asset's group, -1 where left
      out) and weights (the long-short spread's)
    source: Fama and French (1993), Journal of Financial Economics 33(1); Jegadeesh and Titman (1993), J. Finance 48(1)
    aliases: quantile portfolios, quintile portfolios, decile portfolios, portfolio sort, long-short portfolio
    example:
        qf.quantile_portfolios(signal=[[0.5, -1.2, 0.3, 2.0, -0.4, 1.1]],
                               returns=[[0.02, -0.01, 0.00, 0.04, 0.01, 0.03]],
                               n_groups=3, breakpoints="rank", weighting="equal")
    """
    choice("breakpoints", breakpoints, BREAKPOINTS)
    x, r = np.asarray(signal, dtype=float), np.asarray(returns, dtype=float)
    if x.shape != r.shape or x.ndim not in (1, 2):
        raise ValueError("signal and returns must have one shape, T x N or N")
    if isinstance(n_groups, (bool, np.bool_)) or not isinstance(n_groups, (int, np.integer)) or n_groups < 2:
        raise ValueError("n_groups must be an int of at least 2")
    if isinstance(weighting, str):
        choice("weighting", weighting, ("equal",))
        w = np.ones_like(x)
    else:
        w = np.asarray(weighting, dtype=float)
        if w.shape != x.shape or np.any(w < 0):
            raise ValueError("weighting must be 'equal' or non-negative values shaped like signal")
    rows_x, rows_r, rows_w = (np.atleast_2d(a) for a in (x, r, w))
    group_returns = np.full((rows_x.shape[0], n_groups), np.nan)
    groups = np.full(rows_x.shape, -1)
    weights = np.zeros(rows_x.shape)
    for row, (sx, sr, sw) in enumerate(zip(rows_x, rows_r, rows_w)):
        kept = np.isfinite(sx) & np.isfinite(sr) & np.isfinite(sw)
        if kept.sum() < n_groups:
            continue
        found = _groups(sx[kept], int(n_groups), breakpoints)
        groups[row, kept] = found
        for g in range(n_groups):
            members = np.flatnonzero(kept)[found == g]
            total = sw[members].sum()
            if members.size and total > 0:
                share = sw[members] / total
                group_returns[row, g] = share @ sr[members]
                if g in (0, n_groups - 1):
                    weights[row, members] = share if g == n_groups - 1 else -share
    spread = group_returns[:, -1] - group_returns[:, 0]
    if x.ndim == 1:
        return Result({"group_returns": group_returns[0], "spread": float(spread[0]), "groups": groups[0],
                       "weights": weights[0]})
    return Result({"group_returns": group_returns, "spread": spread, "groups": groups, "weights": weights})

def rank_bins(values, ids, n_bins, groups, allocation):
    """Rank bins with identifier ties: bin floor(n r / N) of each name's position r, within groups for dependent sorts.

    Each group's N names with a finite value (all of them where groups is None) are sorted by value, a tie broken by
    the identifier ascending, r = 0, 1, ..., N - 1 their positions: 'floor' gives bin floor(n_bins r / N), the
    remainder to the lower bins (7 names in 3 bins hold 3, 2, 2); 'ceil' gives ceil(n_bins (r + 1) / N) - 1, the
    remainder to the upper bins (2, 2, 3). Bin 0 holds the lowest values. Fewer names than bins leaves bins empty: no
    name is moved into one. A dependent (sequential) double sort ranks the second signal within the first's bins
    (groups = the first call's bins); an independent one ranks each over every name. Not quantile_portfolios'
    'rank' breakpoints (pandas.qcut of the ranks), whose counts differ.

    conventions:
    - allocation: 'floor' (floor(n r / N): the remainder to the lower bins) or 'ceil' (the remainder to the upper)
    - ids: one identifier per name (all numbers or all strings, unique), breaking ties in ascending order; positional ties:
      numpy.arange(N); a descending sort: pass the negated values (ties still by identifier ascending)
    - groups: None (independent: one ranking per row) or labels shaped like values (a dependent sort's first bins),
      ranked within each label; a negative or NaN label leaves the name out
    - values: N (one date) or T x N (rows dates, columns names, each row ranked alone); NaN leaves a name out
    returns: an int array shaped like values, each name's bin from 0, -1 where left out
    source: Fama and French (1993), J. Financial Economics 33(1) (sorted portfolios); Bali, Engle and Murray,
      Empirical Asset Pricing: The Cross Section of Stock Returns (2016), ch. 7 (dependent and independent sorts)
    aliases: rank bins, floor rank bins, rank-based quintiles, rank quintiles, identifier ties, dependent sort bins, [equal-count bins]
    example:
        # Returns [0, 2, 0, 1, 1]: of the two at 1.0, "AAA" ranks first and stays in bin 0.
        qf.rank_bins(values=[0.5, 2.0, 1.0, 1.0, 1.5], ids=["CCC", "EEE", "AAA", "BBB", "DDD"], n_bins=3, groups=None, allocation="floor")
    """
    choice("allocation", allocation, BIN_ALLOCATIONS)
    bins = integer("n_bins", n_bins, 1)
    x = np.asarray(values, dtype=float)
    if x.ndim not in (1, 2) or x.shape[-1] < 1:
        raise ValueError("values must be N values (one date) or T x N (dates by names)")
    raw_names = np.asarray(ids, dtype=object)
    kinds = {"text" if isinstance(item, (str, np.str_)) else
             "number" if isinstance(item, (int, float, np.integer, np.floating)) else "other"
             for item in raw_names.ravel()}
    if len(kinds) != 1 or "other" in kinds:
        raise ValueError("ids must be all numbers or all strings: mixed identifier types have no common order")
    names = np.asarray(ids)
    if names.shape != (x.shape[-1],) or np.unique(names).size != names.size:
        raise ValueError("ids must hold one unique identifier per name (column)")
    order = np.empty(names.size, dtype=np.int64)
    order[np.argsort(names, kind="stable")] = np.arange(names.size)
    rows = np.atleast_2d(x)
    if groups is None:
        labels = np.zeros(rows.shape)
    else:
        labels = np.atleast_2d(np.asarray(groups, dtype=float))
        if labels.shape != rows.shape:
            raise ValueError("groups must be None or labels shaped like values")
    found = np.full(rows.shape, -1, dtype=np.int64)
    for row in range(rows.shape[0]):
        kept = np.isfinite(rows[row]) & np.isfinite(labels[row]) & (labels[row] >= 0)
        for label in np.unique(labels[row, kept]):
            members = np.flatnonzero(kept & (labels[row] == label))
            ranked = members[np.lexsort((order[members], rows[row, members]))]
            count, position = ranked.size, np.arange(ranked.size)
            found[row, ranked] = (bins * position // count if allocation == "floor"
                                  else -(-bins * (position + 1) // count) - 1)
    return found[0] if x.ndim == 1 else found


def neutralize(weights, exposures):
    """Weights made neutral to given exposures by projection, w - X (X'X)^+ X'w.

    The weights nearest w (least squares) with X' w = 0: the residual of w regressed on the exposures' columns. A
    column of ones is dollar neutrality (it takes the mean weight out), the betas beta neutrality, a dummy column per
    sector sector neutrality; (X'X)^+ is the pseudo-inverse, so a redundant column is harmless. The gross exposure
    changes: rescale afterwards where a book size is set.

    conventions:
    - weights: one per asset (N)
    - exposures: N x k, a column per exposure to remove (an N vector for one), e.g. column_stack([ones(N), beta])
    returns: an array of N weights, each exposure's sum X' w zero
    source: Grinold and Kahn, Active Portfolio Management (2000), ch. 14 (neutralization); least-squares projection
    aliases: dollar neutral, beta neutral, sector neutral, market neutral weights, neutralize weights, neutralise
    example:
        qf.neutralize(weights=[0.4, 0.3, -0.1, 0.2],
                      exposures=[[1.0, 1.2], [1.0, 0.8],
                                 [1.0, 1.0], [1.0, 0.5]])
    """
    w = np.asarray(weights, dtype=float)
    x = np.asarray(exposures, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if w.ndim != 1 or x.ndim != 2 or x.shape[0] != w.size or not (np.all(np.isfinite(w)) and np.all(np.isfinite(x))):
        raise ValueError("weights must be N finite numbers and exposures N x k (or N) finite numbers")
    return w - x @ (np.linalg.pinv(x.T @ x) @ (x.T @ w))


def _forward(returns, lag, horizon, kind):
    """Row t: the return over periods t + lag to t + lag + horizon - 1, compounded (simple) or summed (log); NaN where
    the window runs past the data."""
    bars = returns.shape[0]
    found = np.full(returns.shape, np.nan)
    for t in range(bars - lag - horizon + 1):
        window = returns[t + lag:t + lag + horizon]
        found[t] = np.prod(1.0 + window, axis=0) - 1.0 if kind == "simple" else window.sum(axis=0)
    return found


def information_coefficient(signal, returns, method, lag, horizon, return_kind, min_names, ddof, t_stat, lags):
    """Information coefficient and its significance: signal correlation with forward returns (Pearson or Spearman).

    Per date t, the correlation across assets of the signal with the forward return over periods t + lag to t + lag
    + horizon - 1: 'spearman' the correlation of their ranks (ties averaged), 'pearson' of the values. Over the dates
    with at least min_names pairs: the mean IC, its standard deviation, the IC information ratio mean / sd and the
    t-statistic mean / sqrt(V / n), V the IC's variance ('iid', with ddof) or Newey-West's long-run variance
    gamma_0 + 2 sum_l (1 - l / (lags + 1)) gamma_l (gamma_l over n). Overlapping horizons make the ICs autocorrelated:
    lags of horizon - 1 at least. The fundamental law: IR = IC sqrt(breadth); alpha = vol x IC x score (Grinold).

    conventions:
    - signal, returns: T x N, rows dates in order, columns assets, NaN missing; returns[t] the return over period t
    - method: 'spearman' or 'pearson'; lag: periods from the signal's date to its first return (1: the next period's)
    - horizon: periods in the forward return; return_kind: 'simple' (compounded) or 'log' (summed)
    - min_names: fewer pairs on a date leave it out; ddof: of the IC's standard deviation (0 or 1)
    - t_stat: 'iid' (lags None) or 'newey_west' (lags, the Bartlett kernel's, an int)
    returns: a dict: ic (one per date, NaN where left out); mean, std, icir (mean / std), t_stat; n_dates (the
      dates used)
    source: Grinold and Kahn, Active Portfolio Management (2000), ch. 6 and 12; Newey and West (1987), Econometrica
    aliases: information coefficient, rank IC, Spearman IC, IC information ratio, ICIR, IC t-statistic, fundamental law of active management, signal IC, [IC]
    example:
        qf.information_coefficient(signal=[[0.1, 0.5, -0.2, 0.3], [0.4, -0.1, 0.0, 0.2], [0.2, 0.1, 0.3, -0.3]],
            returns=[[0.01, 0.02, -0.01, 0.00], [0.02, 0.03, -0.02, 0.01], [0.03, -0.01, 0.00, 0.02]], method="spearman", lag=1,
            horizon=1, return_kind="simple", min_names=3, ddof=1, t_stat="iid", lags=None)
    """
    choice("method", method, IC_METHODS)
    choice("return_kind", return_kind, RETURN_KINDS)
    choice("t_stat", t_stat, T_STATS)
    x, r = np.asarray(signal, dtype=float), np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape != r.shape:
        raise ValueError("signal and returns must both be T x N")
    shift, span = integer("lag", lag, 0), integer("horizon", horizon, 1)
    least, spread = integer("min_names", min_names, 3), _ddof(ddof)
    if (t_stat == "iid") != (lags is None):
        raise ValueError("lags: None with t_stat 'iid', an int of at least 0 with 'newey_west'")
    forward = _forward(r, shift, span, return_kind)
    ic = np.full(x.shape[0], np.nan)
    for t in range(x.shape[0]):
        kept = np.isfinite(x[t]) & np.isfinite(forward[t])
        if kept.sum() < least:
            continue
        a, b = x[t, kept], forward[t, kept]
        if method == "spearman":
            a, b = stats.rankdata(a), stats.rankdata(b)
        if np.ptp(a) > 0 and np.ptp(b) > 0:
            ic[t] = np.corrcoef(a, b)[0, 1]
    used = ic[np.isfinite(ic)]
    n = used.size
    if n <= spread:
        raise ValueError("too few dates with an IC for a standard deviation: lower min_names or add dates")
    mean, sd = float(used.mean()), float(used.std(ddof=spread))
    if t_stat == "iid":
        variance = sd * sd
    else:
        width = integer("lags", lags, 0)
        centred = used - mean
        variance = centred @ centred / n
        for k in range(1, min(width, n - 1) + 1):
            variance += 2.0 * (1.0 - k / (width + 1.0)) * (centred[k:] @ centred[:-k]) / n
    return Result({"ic": ic, "mean": mean, "std": sd, "icir": mean / sd if sd > 0 else np.nan,
                   "t_stat": float(mean / np.sqrt(variance / n)) if variance > 0 else np.nan, "n_dates": n})


def _bins(x, edges):
    """Each value's group, 0 the lowest: right-closed bins at the edges (a value equal to an edge goes below it)."""
    return np.searchsorted(edges, x, side="left")


def double_sort(first, second, returns, first_breakpoints, second_breakpoints, sort, weighting, universe, rebalance,
                quantile_method):
    """Fama-French double sort (2 x 3): SMB and HML from size and value breakpoints, independent or sequential.

    Stocks split at the first characteristic's percentiles, then the second's, over the breakpoint universe
    ('independent') or within each first group ('sequential'). Bins are right-closed: a value on an edge goes below.
    Groups hold to the next rebalance. Cells earn equal- or value-weighted returns. smb averages small less big
    across the second's groups; hml averages high less low across the first's (size first, book-to-market second).

    conventions:
    - first, second, returns: T x N, dates by stocks; lag characteristics to formation first; NaN leaves stocks out
    - first_breakpoints, second_breakpoints: percentile fractions, e.g. [0.5] and [0.3, 0.7]
    - sort: 'independent' (both over the universe) or 'sequential' (second within each first group)
    - weighting: 'equal' or T x N market caps at each row's start
    - universe: None (all stocks) or T x N booleans marking breakpoint stocks (e.g. NYSE)
    - rebalance: None (every row) or T booleans marking formation rows (e.g. June)
    - quantile_method: 'linear' interpolates; 'lower', 'higher', 'nearest' choose an observation; 'midpoint' averages
      adjacent ones; 'hazen' or 'weibull' use their plotting positions; 'inverted_cdf' takes the empirical quantile
    returns: a dict: portfolio_returns (T x groups x groups, NaN for empty cells); counts (stocks per cell);
      first_group, second_group (T x N, -1 for excluded stocks); smb, hml (T each)
    source: Fama and French (1993), J. Financial Economics 33(1); Carhart (1997), J. Finance 52(1)
    aliases: double sort, 2x3 sort, Fama-French 2x3, independent sort, dependent sort, sequential sort, [SMB], [HML], [small minus big], [high minus low], [UMD]
    example:
        # smb = [-0.01], hml = [0.0].
        qf.double_sort(first=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], second=[[0.9, 0.2, 0.5, 0.1, 0.7, 0.4]],
                       returns=[[0.02, -0.01, 0.01, 0.03, 0.00, 0.02]], first_breakpoints=[0.5],
                       second_breakpoints=[0.3, 0.7], sort="independent", weighting="equal", universe=None,
                       rebalance=None, quantile_method="linear")
    """
    choice("sort", sort, SORTS)
    choice("quantile_method", quantile_method, QUANTILE_METHODS)
    a, b, r = (np.asarray(v, dtype=float) for v in (first, second, returns))
    if a.ndim != 2 or a.shape != b.shape or a.shape != r.shape:
        raise ValueError("first, second and returns must each be T x N over the same dates and stocks")
    bars, names = a.shape
    edges = []
    for name, points in (("first_breakpoints", first_breakpoints), ("second_breakpoints", second_breakpoints)):
        levels = np.atleast_1d(np.asarray(points, dtype=float))
        if levels.ndim != 1 or levels.size < 1 or np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
            raise ValueError(f"{name} must be increasing fractions inside (0, 1), e.g. [0.3, 0.7]")
        edges.append(levels)
    n1, n2 = edges[0].size + 1, edges[1].size + 1
    if isinstance(weighting, str):
        choice("weighting", weighting, ("equal",))
        w = np.ones_like(r)
    else:
        w = np.asarray(weighting, dtype=float)
        if w.shape != r.shape or np.any(w[np.isfinite(w)] < 0):
            raise ValueError("weighting must be 'equal' or T x N non-negative market caps")
    setter = np.ones(a.shape, bool) if universe is None else np.asarray(universe, dtype=bool)
    formed = np.ones(bars, bool) if rebalance is None else np.asarray(rebalance, dtype=bool).ravel()
    if setter.shape != a.shape or formed.shape != (bars,):
        raise ValueError("universe must be T x N booleans (or None) and rebalance T booleans (or None)")
    g1, g2 = np.full(a.shape, -1), np.full(a.shape, -1)
    held1, held2 = np.full(names, -1), np.full(names, -1)
    cells = np.full((bars, n1, n2), np.nan)
    counts = np.zeros((bars, n1, n2), dtype=int)
    for t in range(bars):
        if formed[t]:
            valid = np.isfinite(a[t]) & np.isfinite(b[t])
            base = valid & setter[t]
            held1, held2 = np.full(names, -1), np.full(names, -1)
            if base.sum() >= 2:
                held1[valid] = _bins(a[t, valid], np.quantile(a[t, base], edges[0], method=quantile_method))
                for group in (range(n1) if sort == "sequential" else [None]):
                    inside = base if group is None else base & (held1 == group)
                    members = valid if group is None else valid & (held1 == group)
                    if inside.sum() >= 1 and members.any():
                        held2[members] = _bins(b[t, members], np.quantile(b[t, inside], edges[1], method=quantile_method))
                held1[held2 < 0] = -1
        g1[t], g2[t] = held1, held2
        live = (held1 >= 0) & np.isfinite(r[t]) & np.isfinite(w[t])
        for i in range(n1):
            for j in range(n2):
                members = live & (held1 == i) & (held2 == j)
                total = w[t, members].sum()
                counts[t, i, j] = int(members.sum())
                if members.any() and total > 0:
                    cells[t, i, j] = w[t, members] @ r[t, members] / total
    smb = np.mean(cells[:, 0, :] - cells[:, -1, :], axis=1)
    hml = np.mean(cells[:, :, -1] - cells[:, :, 0], axis=1)
    return Result({"portfolio_returns": cells, "counts": counts, "first_group": g1, "second_group": g2, "smb": smb,
                   "hml": hml})


def _wls(y, x, w):
    """Weighted least squares: b minimising sum w (y - x b)^2, refused where x lacks full column rank."""
    root = np.sqrt(w)[:, None]
    if np.linalg.matrix_rank(x * root) < x.shape[1]:
        raise ValueError("the exposures are collinear on a date (a country column with every industry dummy needs "
                         "the industry constraint)")
    return np.linalg.lstsq(x * root, y * root[:, 0], rcond=None)[0]


def cross_sectional_wls(returns, exposures, weights, industries, industry_weights):
    """Barra factor returns: cross-sectional WLS on style and industry exposures, the industries summing to zero.

    Each date: b = (X'WX)^{-1} X'W y, W the diagonal regression weights (the root of market cap in Barra's models),
    residuals e = y - X b (the specific returns), X'W e = 0. With a country column of ones beside a full set of industry
    dummies the design is collinear: the industries' factor returns are held to sum s_j b_j = 0, s_j the industry's
    share of industry_weights (cap weights), by substitution (Menchero); then with cap regression weights and styles of
    zero cap-weighted mean the country return is the cap-weighted market return. W = I gives OLS.

    conventions:
    - returns: T x N (or N) stock returns; exposures: T x N x k (or N x k), styles standardised first
      (cross_sectional_zscore) and industry dummies; NaN leaves a stock out of that date
    - weights: 'equal' (OLS) or T x N (or N) regression weights, e.g. numpy.sqrt(market caps)
    - industries: None (no constraint), or the exposures' columns that are industry dummies
    - industry_weights: None with industries None, else T x N (or N) market caps, the constraint's weights
    returns: a dict: factor_returns (T x k, or k), residuals (shaped like returns, NaN where left out)
    source: Menchero, Orr and Wang, The Barra US Equity Model USE4 (2011), MSCI; Grinold and Kahn (2000), ch. 3
    aliases: Barra factor returns, cross-sectional WLS, fundamental factor model, industry constraint, Barra regression, [factor returns], [specific returns]
    example:
        # factor_returns = [0.007061924, -0.007279315, 0.011198946, -0.001304348].
        qf.cross_sectional_wls(returns=[0.02, -0.01, 0.01, 0.03], exposures=[[1.0, 1.0, 0.0, 0.5], [1.0, 1.0, 0.0, -0.5],
                               [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, -1.0]], weights=[1.0, 2.0, 1.5, 1.0],
                               industries=[1, 2], industry_weights=[1.0, 4.0, 2.25, 1.0])
    """
    y = np.asarray(returns, dtype=float)
    x = np.asarray(exposures, dtype=float)
    one = y.ndim == 1
    ys, xs = np.atleast_2d(y), (x[None] if one else x)
    if xs.ndim != 3 or xs.shape[:2] != ys.shape:
        raise ValueError("returns must be T x N (or N) and exposures T x N x k (or N x k) over the same stocks")
    k = xs.shape[2]
    if isinstance(weights, str):
        choice("weights", weights, ("equal",))
        ws = np.ones_like(ys)
    else:
        ws = np.broadcast_to(np.atleast_2d(np.asarray(weights, dtype=float)), ys.shape)
    columns = None
    if industries is None:
        if industry_weights is not None:
            raise ValueError("industry_weights must be None where industries is None")
        caps = np.ones_like(ys)
    else:
        columns = np.atleast_1d(np.asarray(industries, dtype=int))
        if columns.size < 2 or np.any((columns < 0) | (columns >= k)) or np.unique(columns).size != columns.size:
            raise ValueError("industries must name two or more distinct columns of the exposures")
        if industry_weights is None:
            raise ValueError("industry_weights (market caps) are needed with the industry constraint")
        caps = np.broadcast_to(np.atleast_2d(np.asarray(industry_weights, dtype=float)), ys.shape)
    factors = np.full((ys.shape[0], k), np.nan)
    resid = np.full(ys.shape, np.nan)
    for t in range(ys.shape[0]):
        keep = (np.isfinite(ys[t]) & np.all(np.isfinite(xs[t]), axis=1) & np.isfinite(ws[t]) & (ws[t] > 0)
                & np.isfinite(caps[t]))
        if keep.sum() < k - (columns is not None):
            continue
        yt, xt, wt = ys[t, keep], xs[t, keep], ws[t, keep]
        if columns is None:
            b = _wls(yt, xt, wt)
        else:
            share = caps[t, keep] @ xt[:, columns]
            last = columns[-1]
            if not share[-1] > 0:
                raise ValueError("the last industry column holds no cap weight on a date")
            restrict = np.delete(np.eye(k), last, axis=1)
            restrict[last] = 0.0
            for column, weight in zip(columns[:-1], share[:-1]):
                restrict[last, list(np.delete(np.arange(k), last)).index(column)] = -weight / share[-1]
            b = restrict @ _wls(yt, xt @ restrict, wt)
        factors[t] = b
        resid[t, keep] = yt - xt @ b
    if one:
        return Result({"factor_returns": factors[0], "residuals": resid[0]})
    return Result({"factor_returns": factors, "residuals": resid})


def _time_series(r, f):
    """Each asset's OLS on an intercept and the factors: (coefficients (1 + m) x N, residuals T x N)."""
    design = np.column_stack([np.ones(r.shape[0]), f])
    coef = np.linalg.lstsq(design, r, rcond=None)[0]
    return coef, r - design @ coef, design


def hda_test(asset_excess_returns, factor_excess_returns, version, threshold_p, threshold_power):
    """Pesaran-Yamagata high-dimensional alpha test: the alphas jointly zero when assets outnumber periods.

    Each asset's time-series OLS on an intercept and the m factors gives its alpha's t-ratio t_i (error s^2 = e'e / v
    times the design's (X'X)^{-1} first entry, v = T - m - 1). 'normal' (sum t_i^2 - N) / sqrt(2N), the t-ratios taken
    as normal; 'student_t' J = N^{-1/2} sum (t_i^2 - v/(v - 2)) / [v/(v - 2) sqrt(2(v - 1)/(v - 4))], the exact
    t moments. The Pesaran-Yamagata versions divide J by sqrt(1 + (N - 1) rho^2), rho^2 the mean squared residual
    correlation over pairs with sqrt(v)|rho_ij| above the normal quantile at 1 - p/(2 f(N)). Published f(N)=N^delta;
    the 2017 working paper uses N-1. Large values reject zero alphas (normal upper tail). GRS needs T > N.

    conventions:
    - asset_excess_returns: (T, N); factor_excess_returns: (T, m) or (T,), excess of the risk-free rate; NaN refused
    - version: 'normal', 'student_t', 'pesaran_yamagata' (2024) or 'pesaran_yamagata_2017'; all but 'normal' need v > 4
    - threshold_p: p in (0, 1) with either Pesaran-Yamagata version; None otherwise
    - threshold_power: positive delta for 'pesaran_yamagata' (1 or 2 in the paper); None otherwise
    returns: a dict: statistic; p_value (upper tail); tstats and alpha (per asset); rho2 (0.0 without thresholding)
    source: Pesaran and Yamagata (2024), J. Financial Econometrics 22(2), eqs. (43), (46); York working paper 17/04 (2017), eq. (44)
    aliases: Pesaran-Yamagata test, high-dimensional alpha test, HDA test, J alpha test, [joint alpha test]
    example:
        # statistic = -0.588248, p_value = 0.721817.
        qf.hda_test(asset_excess_returns=[[0.01, 0.02, 0.0], [0.0, -0.01, 0.01], [0.03, 0.02, 0.02], [-0.01, 0.0, -0.02],
                                          [0.02, 0.01, 0.0], [0.01, 0.03, 0.01], [0.0, 0.01, 0.02], [0.02, -0.01, 0.01]],
                    factor_excess_returns=[0.01, 0.0, 0.02, -0.01, 0.01, 0.01, 0.0, 0.01], version="student_t",
                    threshold_p=None, threshold_power=None)
    """
    choice("version", version, HDA_VERSIONS)
    r = np.asarray(asset_excess_returns, dtype=float)
    f = np.asarray(factor_excess_returns, dtype=float)
    r = r[:, None] if r.ndim == 1 else r
    f = f[:, None] if f.ndim == 1 else f
    if r.ndim != 2 or f.ndim != 2 or r.shape[0] != f.shape[0]:
        raise ValueError("asset_excess_returns must be (T, N) and factor_excess_returns (T, m) over the same T")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(f))):
        raise ValueError("asset_excess_returns and factor_excess_returns must be finite")
    t, n = r.shape
    v = t - f.shape[1] - 1
    thresholded = version in ("pesaran_yamagata", "pesaran_yamagata_2017")
    if thresholded != (threshold_p is not None):
        raise ValueError("threshold_p: a p in (0, 1) with either Pesaran-Yamagata version, None otherwise")
    if version == "pesaran_yamagata":
        if threshold_power is None or not np.isfinite(threshold_power) or float(threshold_power) <= 0:
            raise ValueError("threshold_power must be positive for 'pesaran_yamagata'")
    elif threshold_power is not None:
        raise ValueError("threshold_power must be None outside 'pesaran_yamagata'")
    if version != "normal" and v <= 4:
        raise ValueError(f"the Student-t versions need T - m - 1 > 4 (it is {v})")
    if v < 1:
        raise ValueError("more periods than factors plus one are needed")
    coef, resid, design = _time_series(r, f)
    s2 = (resid ** 2).sum(axis=0) / v
    tstats = coef[0] / np.sqrt(s2 * np.linalg.inv(design.T @ design)[0, 0])
    rho2 = 0.0
    if version == "normal":
        stat = (np.sum(tstats ** 2) - n) / np.sqrt(2.0 * n)
    else:
        if thresholded:
            level = float(threshold_p)
            if not 0.0 < level < 1.0 or n < 2:
                raise ValueError("threshold_p must lie in (0, 1), and N must be at least 2")
            corr = np.corrcoef(resid, rowvar=False)
            pairs = corr[np.triu_indices(n, 1)]
            threshold_size = n ** float(threshold_power) if version == "pesaran_yamagata" else n - 1
            cut = norm.ppf(1.0 - level / (2.0 * threshold_size))
            rho2 = float(np.sum(np.where(np.sqrt(v) * np.abs(pairs) > cut, pairs ** 2, 0.0)) * 2.0 / (n * (n - 1)))
        centre = v / (v - 2.0)
        stat = (np.sum(tstats ** 2 - centre) / np.sqrt(n)) / (
            centre * np.sqrt(2.0 * (v - 1.0) / (v - 4.0) * (1.0 + (n - 1) * rho2)))
    return Result({"statistic": float(stat), "p_value": float(norm.sf(stat)), "tstats": tstats, "alpha": coef[0],
                   "rho2": rho2})


def _betas(returns, factors):
    """Each asset's slopes on the factors in its time-series OLS with an intercept, N x m."""
    return _time_series(returns, factors)[0][1:].T


def _priced(betas, mean_returns, intercept, instruments=None):
    """The cross-section's premia: OLS of mean returns on the betas, or IV with other betas as the instruments."""
    design = np.column_stack([np.ones(betas.shape[0]), betas]) if intercept else betas
    if instruments is None:
        return np.linalg.lstsq(design, mean_returns, rcond=None)[0]
    z = np.column_stack([np.ones(instruments.shape[0]), instruments]) if intercept else instruments
    return np.linalg.solve(z.T @ design, z.T @ mean_returns)


def split_sample_jackknife(returns, factors, intercept, method, split):
    """Two-pass premia bias-corrected by a split sample: split-panel jackknife, instrumented betas or crossed halves.

    The two-pass estimate: each asset's time-series OLS on an intercept and the factors gives its betas; the assets'
    mean returns are regressed on the betas (with an intercept, the zero-beta rate, or without). The betas' errors bias
    it toward zero by about 1/T, so 'jackknife' takes 2 lambda_full minus the length-weighted half estimates.
    With odd T, 'halves' averages both adjacent split points (Dhaene-Jochmans eq. 2.4). 'instrumented' prices on one
    half's betas with the other half's as instruments, both ways averaged; 'cross' prices one half's betas against the
    other half's mean returns, both ways averaged, which leaves the shrinkage toward zero in place.

    conventions:
    - returns: T x N asset returns (excess where the factors are); factors: T x m (or T); NaN refused
    - intercept: True (a zero-beta rate in the cross-section) or False
    - method: 'jackknife', 'instrumented' or 'cross'
    - split: 'halves' (the first floor(T/2) periods and the rest) or 'odd_even' (alternate periods)
    returns: a dict: premia (the corrected estimates, the intercept first where there is one); full (the
      full-sample two-pass estimate); halves (2 x p: each half's own estimate, or the method's two; for odd-T
      jackknife, rows are length-adjusted to average to the split estimate)
    source: Dhaene and Jochmans (2015), Rev. Econ. Studies 82(3); Jegadeesh, Noh, Pukthuanthong, Roll and Wang (2019), J. Financial Economics 133(2); Shanken (1992)
    aliases: split-sample jackknife, split-panel jackknife, jackknife bias correction, half-panel jackknife, instrumented betas, [bias-corrected premia], [two-pass]
    example:
        # premia = [0.004461270, 0.006743297].
        qf.split_sample_jackknife(returns=[[0.02, 0.01, 0.03], [-0.01, 0.0, -0.02], [0.03, 0.02, 0.04],
                                           [0.0, 0.01, 0.0], [0.01, -0.01, 0.02], [0.02, 0.02, 0.01]],
                                  factors=[0.02, -0.01, 0.03, 0.0, 0.01, 0.02], intercept=True, method="jackknife",
                                  split="halves")
    """
    choice("method", method, JACKKNIFE_METHODS)
    choice("split", split, SPLITS)
    if not isinstance(intercept, (bool, np.bool_)):
        raise ValueError(f"intercept must be True or False, not {intercept!r}")
    r = np.asarray(returns, dtype=float)
    f = np.asarray(factors, dtype=float)
    f = f[:, None] if f.ndim == 1 else f
    if r.ndim != 2 or f.ndim != 2 or r.shape[0] != f.shape[0]:
        raise ValueError("returns must be T x N and factors T x m over the same periods")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(f))):
        raise ValueError("returns and factors must be finite")
    bars, m = f.shape
    order = np.arange(bars)
    one, two = (order[:bars // 2], order[bars // 2:]) if split == "halves" else (order[0::2], order[1::2])
    if min(one.size, two.size) <= m + 1:
        raise ValueError("each half needs more periods than factors plus one")
    if r.shape[1] <= m + intercept:
        raise ValueError("the cross-section needs more assets than premia")
    full = _priced(_betas(r, f), r.mean(axis=0), intercept)
    first, second = _betas(r[one], f[one]), _betas(r[two], f[two])
    if method == "jackknife":
        halves = np.array([_priced(first, r[one].mean(axis=0), intercept),
                           _priced(second, r[two].mean(axis=0), intercept)])
        halves *= 2 * np.array([one.size, two.size])[:, None] / bars
        if bars % 2 and split == "halves":
            later = order[:bars // 2 + 1], order[bars // 2 + 1:]
            alternative = np.array([_priced(_betas(r[rows], f[rows]), r[rows].mean(axis=0), intercept)
                                    for rows in later])
            alternative *= 2 * np.array([rows.size for rows in later])[:, None] / bars
            halves = (halves + alternative) / 2
        return Result({"premia": 2.0 * full - halves.mean(axis=0), "full": full, "halves": halves})
    if method == "instrumented":
        halves = np.array([_priced(second, r.mean(axis=0), intercept, instruments=first),
                           _priced(first, r.mean(axis=0), intercept, instruments=second)])
    else:
        halves = np.array([_priced(first, r[two].mean(axis=0), intercept),
                           _priced(second, r[one].mean(axis=0), intercept)])
    return Result({"premia": halves.mean(axis=0), "full": full, "halves": halves})
