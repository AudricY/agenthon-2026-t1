"""Regression and tests: OLS, Newey-West, influence and VIF, PCA, ADF, cointegration, Lo-MacKinlay, FDR, bootstrap.

`import qf`. Regressors come as an (n, k) array, or (n,) for one; `intercept` names whether a column of
ones goes first, so the coefficients read [const, x1, ..., xk] with it and [x1, ..., xk] without. A function with
several results returns a dict with named keys; a scalar result is a float.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.stats import chi2, norm

from . import _mackinnon
from ._common import Result, choice, integer
from .primitives import _guard, _stream

__all__ = ["adf_test", "block_bootstrap", "cross_sectional_r2", "engle_granger", "johansen", "multiple_testing",
           "newey_west_cov", "newey_west_ols", "ols", "ols_influence", "pca", "variance_ratio"]

DENOMINATORS = ("T", "T-k")
MATRICES = ("covariance", "correlation")
PCA_INPUTS = ("panel", "matrix")
SIGNS = ("max_abs_positive", "sum_positive")
SIGN_ANCHORS = ("loading", "weights", "series")
TRENDS = ("none", "constant", "constant_trend")
LAG_RULES = ("aic", "bic", "t_stat", "fixed")
T_STAT_STOP = 1.6448536269514722    # N^-1(0.95): 't_stat' keeps the longest lag whose own |t| reaches it


def _design(y, X, intercept):
    """y as (n,), X as (n, p) with the ones column first where `intercept`; both finite, X of full column rank."""
    if not isinstance(intercept, (bool, np.bool_)):
        raise ValueError(f"intercept must be True or False, not {intercept!r}")
    y = np.asarray(y, dtype=float)
    x = np.asarray(X, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("y must be (n,) and X (n, k) or (n,) with the same n")
    if intercept:
        x = np.column_stack([np.ones(y.size), x])
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError("y and X must be finite (drop the rows with NaN first)")
    if x.shape[1] == 0 or y.size <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("X must have full column rank and fewer columns than rows (a singular X: qf.ols_lstsq, "
                         "singular='min_norm', the pseudo-inverse's solution)")
    return y, x


def _fit(y, x):
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return coef, y - x @ coef, np.linalg.inv(x.T @ x)


def _lags(lags):
    if not isinstance(lags, (int, np.integer)) or isinstance(lags, bool) or lags < 0:
        raise ValueError(f"lags must be a whole number, 0 or more, not {lags!r}")
    return int(lags)


def _hac(x, resid, bread, lags, denominator):
    """Newey-West's sandwich with Bartlett weights 1 - l / (L + 1), scaled by n / (n - p) for 'T-k'."""
    choice("denominator", denominator, DENOMINATORS)
    scores = x * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (lags + 1.0)) * (cross + cross.T)
    cov = bread @ meat @ bread
    n, p = x.shape
    return cov * (n / (n - p)) if denominator == "T-k" else cov


def ols(y, X, intercept):
    """OLS coefficients, classical standard errors, t-statistics, R^2; rolling betas by rolling_apply.

    b = (X'X)^{-1} X'y, e = y - X b, s^2 = e'e / (n - p) with p the columns (the ones column counted),
    se = sqrt(diag(s^2 (X'X)^{-1})), t = b / se; R^2 = 1 - e'e / sum (y - mean y)^2 with an intercept, and the
    uncentered 1 - e'e / sum y^2 without one.

    The minimum-variance hedge ratio h* = rho sigma_S / sigma_F = Cov(dS, dF) / Var(dF) is the slope of spot changes
    on futures changes with an intercept (its R^2 the hedge's effectiveness); contracts N* = h* Q_S / Q_F.

    Rolling-window coefficients: rolling_apply(lambda y, X: ols(y, X, intercept=True)['coef'], [y, X], window=w, kind='rolling'), NaN before the first full window.

    The Durbin-Watson statistic of the residuals in time order: d = sum_{t>1} (e_t - e_{t-1})^2 / e'e, near 2 when
    they are not autocorrelated (about 2 (1 - rho), rho their first-order autocorrelation).

    R^2 split over the regressors (with an intercept): R^2 = sum_j b_j Cov(x_j, y) / Var(y), regressor j's share.

    A default probability by logistic regression (a logit): p = 1 / (1 + exp(-X b)), b maximising sum [d ln p + (1 - d)
    ln(1 - p)] (scipy.optimize.minimize); its AUC is scipy.stats.mannwhitneyu(p of defaulters, p of the rest).statistic
    / (n1 n0), its KS statistic scipy.stats.ks_2samp(the two groups' p).statistic.

    conventions:
    - intercept: True puts a ones column first (coef[0] is alpha), False fits through the origin
    - s^2 divides by n - p, the unbiased residual variance; the standard errors assume homoskedastic errors
    - rows with NaN are refused, not dropped: align and drop them first
    - a rank-deficient X is refused: qf.ols_lstsq(singular='min_norm') gives the pseudo-inverse's (lstsq) solution
    returns: a dict: coef, se and tstat (one per column, the constant first with an intercept); resid and fitted (one
      per row); r2; sigma2 (s^2); dof (n - p); cov (p x p, s^2 (X'X)^{-1})
    source: Greene, Econometric Analysis, ch. 4 (least squares); Hayashi, Econometrics (2000), 1.2-1.3
    aliases: [OLS], [ordinary least squares], [linear regression], regression, [least squares], [market model], beta, [rolling OLS], [rolling regression], [rolling beta]
    example:
        qf.ols(y=[0.01, 0.03, -0.01, 0.02, 0.04],
               X=[0.005, 0.02, -0.01, 0.01, 0.03],
               intercept=True)
    """
    y, x = _design(y, X, intercept)
    coef, resid, bread = _fit(y, x)
    dof = y.size - x.shape[1]
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * bread
    se = np.sqrt(np.diag(cov))
    total = np.sum((y - y.mean()) ** 2) if intercept else np.sum(y * y)
    return Result({"coef": coef, "se": se, "tstat": coef / se, "resid": resid, "fitted": y - resid,
                   "r2": float(1.0 - resid @ resid / total), "sigma2": sigma2, "dof": dof, "cov": cov})


def newey_west_ols(y, X, intercept, lags, denominator):
    """OLS with Newey-West (HAC) standard errors: Bartlett kernel, lags and denominator named.

    b as ols; V = (X'X)^{-1} S (X'X)^{-1},  S = sum_t e_t^2 x_t x_t' + sum_{l=1}^{L} w_l sum_{t>l} e_t e_{t-l}
    (x_t x_{t-l}' + x_{t-l} x_t'),  w_l = 1 - l / (L + 1); 'T-k' multiplies V by n / (n - p), p the columns.
    L = 0 is White's heteroskedasticity-robust covariance (HC0 with 'T', HC1 with 'T-k').

    conventions:
    - lags: L, the last autocovariance kept (no default rule: floor(4 (n/100)^{2/9}) is one, if the task names it)
    - denominator: 'T' keeps Newey and West's 1/T (statsmodels' HAC uncorrected); 'T-k' scales V by n/(n-p)
    - intercept: True puts a ones column first; rows in time order, NaN refused
    returns: a dict: coef, se and tstat (one per column, the constant first with an intercept); cov (the HAC
      covariance, p x p); resid (one per row)
    source: Newey and West (1987), Econometrica 55(3); Greene, Econometric Analysis, the Newey-West estimator
    aliases: Newey-West, HAC standard errors, Newey West, heteroskedasticity and autocorrelation consistent, NW t-stat
    example:
        qf.newey_west_ols(y=[0.02, 0.01, -0.01, 0.03, 0.0, 0.02, -0.02, 0.01],
                          X=[0.01, 0.02, -0.02, 0.02, 0.01, 0.01, -0.01, 0.0],
                          intercept=True, lags=2, denominator="T")
    """
    y, x = _design(y, X, intercept)
    coef, resid, bread = _fit(y, x)
    cov = _hac(x, resid, bread, _lags(lags), denominator)
    se = np.sqrt(np.diag(cov))
    return Result({"coef": coef, "se": se, "tstat": coef / se, "cov": cov, "resid": resid})


def rolling_ols(y, X, intercept, window):
    """Rolling-window OLS coefficients: row t fits the `window` rows ending at row t, inclusive.

    b_t = (X_W' X_W)^{-1} X_W' y_W over rows t - window + 1, ..., t; rows before the first full window are numpy.nan.

    conventions:
    - window: the rows in each fit, more than the columns; the estimate at t uses data through t (lag it to trade on it)
    - intercept: True puts a ones column first; rows with NaN are refused (drop them first)
    returns: an (n, p) array of coefficients, columns in the design's order ([const, x1, ...] with an intercept),
      numpy.nan in the rows before the first full window
    source: Zivot and Wang, Modeling Financial Time Series with S-PLUS (2006), rolling regression; Greene, ch. 4
    aliases: [rolling OLS], [rolling regression], [rolling beta], [moving window regression], [rolling window OLS]
    example:
        qf.rolling_ols(y=[0.01, 0.03, -0.01, 0.02, 0.04, 0.0],
                       X=[0.005, 0.02, -0.01, 0.01, 0.03, -0.002],
                       intercept=True, window=4)
    """
    y, x = _design(y, X, intercept)
    n, p = x.shape
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or not p < window <= n:
        raise ValueError(f"window must be a whole number above the {p} columns and at most {n}, not {window!r}")
    coef = np.full((n, p), np.nan)
    for t in range(window - 1, n):
        rows = slice(t - window + 1, t + 1)
        coef[t] = np.linalg.lstsq(x[rows], y[rows], rcond=None)[0]
    return coef


def pca(data, data_kind, matrix, ddof, sign):
    """Principal components of a panel: loadings, explained variances, scores, on the covariance or correlation.

    X_c = X - mean (and divided by each column's std with `ddof` for 'correlation'); C = X_c' X_c / (n - ddof);
    C = V diag(lambda) V', lambda in decreasing order; loadings V (column j the j-th component), variances lambda,
    explained lambda / sum(lambda), scores X_c V (their variances with the same ddof are lambda). Given a covariance
    or correlation matrix instead of a panel, C is that matrix.

    The Marchenko-Pastur edge: the eigenvalues of a correlation matrix of N series over T dates with no common factor
    stay below sigma^2 (1 + sqrt(N/T))^2, sigma^2 = 1; a component above the edge carries more than noise.

    conventions:
    - data_kind: 'panel' (data n x p, rows the dates) or 'matrix' (a p x p covariance or correlation: no scores)
    - matrix: 'covariance' (scale-dependent) or 'correlation' (each column standardized first)
    - ddof: 1 (the sample covariance, scikit-learn's) or 0; None for a matrix
    - sign: 'sum_positive' (loadings positive on average), 'max_abs_positive' (the largest positive), or an anchor
      made positive: ('loading', j) the j-th loading, ('weights', v) v'loadings, ('series', s) corr(scores, s)
    - data: rows the observations (dates), columns the variables (tenors, assets), NaN refused; or the matrix
    returns: a dict: loadings (p x p, column j the j-th component); variances (decreasing); explained (shares, summing
      to 1); scores (n x p; None for a matrix); mean and scale (the centring and the scaling applied, one per column;
      mean None for a matrix)
    source: Jolliffe, Principal Component Analysis (2002), ch. 1-2; Litterman and Scheinkman (1991), yield-curve factors
    aliases: PCA, principal component analysis, [principal components], [eigen decomposition], [factor loadings], [PC1], Marchenko-Pastur
    example:
        qf.pca(data=[[0.01, 0.012, 0.011], [-0.02, -0.018, -0.015], [0.005, 0.004, 0.006],
                     [0.0, 0.002, 0.003]],
               data_kind="panel", matrix="covariance", ddof=1, sign="sum_positive")
    """
    choice("data_kind", data_kind, PCA_INPUTS)
    choice("matrix", matrix, MATRICES)
    anchor, sign = _anchor(sign)
    choice("sign", sign, SIGNS)
    if data_kind == "matrix":
        return _pca_matrix(data, matrix, ddof, sign, anchor)
    if ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, not {ddof!r}")
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or not np.all(np.isfinite(x)):
        raise ValueError("data must be a finite (n, p) array with at least two rows")
    mean = x.mean(axis=0)
    centred = x - mean
    scale = np.ones(x.shape[1])
    if matrix == "correlation":
        scale = centred.std(axis=0, ddof=ddof)
        if np.any(scale == 0):
            raise ValueError("data has a constant column: its correlation is undefined")
        centred = centred / scale
    values, vectors = _signed_eigen(centred.T @ centred / (x.shape[0] - ddof), sign)
    vectors = _anchored(vectors, anchor, centred)
    return Result({"loadings": vectors, "variances": values, "explained": values / values.sum(),
                   "scores": centred @ vectors, "mean": mean, "scale": scale})


def _signed_eigen(c, sign):
    """The eigenvalues of the symmetric c in decreasing order and their vectors, each column's sign fixed by `sign`."""
    values, vectors = np.linalg.eigh(c)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if sign == "max_abs_positive":
        flip = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
    else:
        flip = np.sign(vectors.sum(axis=0))
    return values, vectors * np.where(flip == 0, 1.0, flip)


def _anchor(sign):
    """(the anchor, the sign rule): an anchor tuple, ('loading', j), ('weights', v) or ('series', s), with
    'max_abs_positive' for a component its anchor leaves at zero; a rule's name with no anchor."""
    if not isinstance(sign, tuple):
        return None, sign
    if len(sign) != 2 or sign[0] not in SIGN_ANCHORS:
        raise ValueError(f"sign must be one of {', '.join(map(repr, SIGNS))} or an anchor ('loading', j), "
                         f"('weights', v) or ('series', s), not {sign!r}")
    return sign, "max_abs_positive"


def _anchored(vectors, anchor, centred):
    """`vectors` with each column's sign set by the anchor (as it was where the anchor gives zero)."""
    if anchor is None:
        return vectors
    kind, target = anchor
    p = vectors.shape[0]
    if kind == "loading":
        if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)) or not 0 <= target < p:
            raise ValueError(f"sign ('loading', j) needs j a variable's position, 0 to {p - 1}, not {target!r}")
        weight = vectors[int(target)]
    elif kind == "weights":
        v = np.asarray(target, dtype=float)
        if v.shape != (p,) or not np.all(np.isfinite(v)):
            raise ValueError(f"sign ('weights', v) needs v finite, one weight per variable ({p})")
        weight = v @ vectors
    else:
        if centred is None:
            raise ValueError("sign ('series', s) needs the panel (data_kind 'panel'): a matrix has no scores")
        s = np.asarray(target, dtype=float)
        if s.shape != (centred.shape[0],) or not np.all(np.isfinite(s)):
            raise ValueError(f"sign ('series', s) needs s finite, one value per row ({centred.shape[0]})")
        weight = (s - s.mean()) @ (centred @ vectors)
    return vectors * np.where(weight < 0, -1.0, 1.0)


def _pca_matrix(data, matrix, ddof, sign, anchor):
    """pca of a covariance or correlation matrix given: no centring, no scores."""
    if ddof not in (None, 0, 1):
        raise ValueError(f"ddof must be None, 0 or 1, not {ddof!r}")
    c = np.asarray(data, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.all(np.isfinite(c)) or not np.allclose(c, c.T):
        raise ValueError("data must be a finite symmetric p x p matrix with data_kind='matrix'")
    scale = np.sqrt(np.diag(c))
    if np.any(~(scale > 0)):
        raise ValueError("the matrix needs a positive diagonal (a variance for each variable)")
    if matrix == "correlation":
        c = c / np.outer(scale, scale)
    else:
        scale = np.ones(c.shape[0])
    values, vectors = _signed_eigen(c, sign)
    vectors = _anchored(vectors, anchor, None)
    if values[-1] < -1e-12 * np.max(np.abs(values)):
        raise ValueError("data must be a positive semidefinite covariance or correlation matrix")
    values = np.maximum(values, 0.0)
    return Result({"loadings": vectors, "variances": values, "explained": values / values.sum(), "scores": None,
                   "mean": None, "scale": scale})


def _ols_fit(y, x):
    """Coefficients, their classical standard errors (s^2 = SSR / (n - k)) and the SSR."""
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("the regression's columns are collinear")
    resid = y - x @ coef
    ssr = float(resid @ resid)
    cov = ssr / (y.size - x.shape[1]) * np.linalg.inv(x.T @ x)
    return coef, np.sqrt(np.diag(cov)), ssr


def _adf_design(y, lags, first, trend):
    """dy_t on [deterministic terms, y_{t-1}, dy_{t-1}, ..., dy_{t-lags}] for t = first, ..., the last change."""
    dy = np.diff(y)
    rows = np.arange(first, dy.size)
    columns = [y[rows]] + [dy[rows - j] for j in range(1, lags + 1)]
    terms = {"none": [], "constant": [np.ones(rows.size)],
             "constant_trend": [np.ones(rows.size), np.arange(1.0, rows.size + 1)]}[trend]
    return dy[rows], np.column_stack(terms + columns), len(terms)


def _adf(y, trend, max_lag, lag_rule):
    if lag_rule == "fixed":
        lag = max_lag
    else:
        fits = {}
        for p in range(max_lag + 1):
            target, x, k = _adf_design(y, p, max_lag, trend)
            coef, se, ssr = _ols_fit(target, x)
            n = target.size
            loglik = -n / 2 * (np.log(2 * np.pi) + np.log(ssr / n) + 1)
            penalty = 2 * x.shape[1] if lag_rule == "aic" else np.log(n) * x.shape[1]
            fits[p] = (-2 * loglik + penalty, abs(coef[-1] / se[-1]) if p else 0.0)
        if lag_rule == "t_stat":
            lag = next((p for p in range(max_lag, 0, -1) if fits[p][1] >= T_STAT_STOP), 0)
        else:
            lag = min((value[0], p) for p, value in fits.items())[1]
    target, x, k = _adf_design(y, lag, lag, trend)
    coef, se, _ = _ols_fit(target, x)
    return float(coef[k] / se[k]), float(coef[k]), lag, target.size


def _unit_root_inputs(y, trend, max_lag, lag_rule):
    choice("trend", trend, TRENDS)
    choice("lag_rule", lag_rule, LAG_RULES)
    series = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(series)):
        raise ValueError("the series must be finite (drop the NaN first)")
    most = integer("max_lag", max_lag, 0)
    if most > series.size // 2 - TRENDS.index(trend) - 1:
        raise ValueError(f"max_lag must be at most len(series) // 2 - {TRENDS.index(trend) + 1} (here "
                         f"{series.size // 2 - TRENDS.index(trend) - 1})")
    return series, most


def adf_test(series, trend, max_lag, lag_rule):
    """Augmented Dickey-Fuller unit-root test, MacKinnon's p-value and critical values.

    Regress dy_t on y_{t-1}, dy_{t-1}, ..., dy_{t-p} and the deterministic terms; the statistic is the t-ratio of
    y_{t-1}'s coefficient gamma = rho - 1, its null a unit root (gamma = 0) against stationarity (gamma < 0), so a
    statistic below the critical value rejects. The lag p: 'fixed' max_lag; 'aic' or 'bic' the criterion's minimum
    over p = 0..max_lag, every p fitted on the same sample (the last n - 1 - max_lag changes), the chosen p then
    refitted on all the changes it allows; 't_stat' the longest p whose last lag's |t| reaches 1.645. The p-value is
    MacKinnon's (1994) asymptotic one, the critical values his (2010) at the regression's size: statsmodels'
    adfuller to rounding. Schwert's rule for max_lag is 12 (n/100)^{1/4}.

    conventions:
    - series: the levels y, oldest first (prices or log prices, not returns, to test for a unit root)
    - trend: 'none', 'constant' or 'constant_trend' (statsmodels' 'n', 'c', 'ct')
    - max_lag: the largest p (an int); lag_rule: 'aic', 'bic', 't_stat' or 'fixed'
    returns: a dict: statistic (the ADF t); p_value; critical_values ({'1%', '5%', '10%'}); lag (p used); nobs (the
      final regression's observations); gamma (y_{t-1}'s coefficient)
    source: Dickey and Fuller (1979); Said and Dickey (1984); MacKinnon (1994, 2010); Hamilton (1994), ch. 17
    aliases: ADF test, augmented Dickey-Fuller, Dickey-Fuller test, unit root test, stationarity test, adfuller, MacKinnon p-value
    example:
        qf.adf_test(series=[100.0, 101.2, 100.7, 102.3, 101.9, 103.4, 102.8, 104.1, 103.5, 105.2, 104.4, 106.0],
                    trend="constant", max_lag=1,
                    lag_rule="aic")
    """
    y, most = _unit_root_inputs(series, trend, max_lag, lag_rule)
    statistic, gamma, lag, nobs = _adf(y, trend, most, lag_rule)
    return Result({"statistic": statistic, "p_value": _mackinnon.p_value(statistic, trend, 1),
                   "critical_values": _mackinnon.critical_values(trend, 1, nobs), "lag": lag, "nobs": nobs,
                   "gamma": gamma})


def engle_granger(y, x, trend, max_lag, lag_rule):
    """Engle-Granger cointegration test: the ADF on the cointegrating regression's residuals.

    Regress y on x and the deterministic terms by OLS, u = its residuals; the statistic is adf_test's t on u with no
    deterministic terms (the regression took them) and the lag rule named, its null no cointegration. Its
    distribution is not Dickey-Fuller's: the p-value and critical values are MacKinnon's for N = 1 + the columns
    of x series and the regression's trend, the critical values at n - 1 observations (statsmodels' coint and
    Stata's egranger); none exists for trend 'none' (numpy.nan). The residual u is the spread, y less the hedge.

    conventions:
    - y: (n,); x: (n,) or (n, k), the levels (prices or log prices), oldest first, aligned
    - trend: 'none', 'constant' or 'constant_trend', in the cointegrating regression
    - max_lag, lag_rule: of the residuals' ADF, as adf_test ('aic', 'bic', 't_stat' or 'fixed')
    returns: a dict: statistic; p_value (numpy.nan past N = 6); critical_values ({'1%', '5%', '10%'}); lag;
      coefficients ([const, trend, x1, ..., xk], the terms the trend names first); residuals (u, the spread)
    source: Engle and Granger (1987), Econometrica 55(2); MacKinnon (1994, 2010); Hamilton (1994), ch. 19
    aliases: Engle-Granger, cointegration test, two-step cointegration, coint, cointegrating regression, pairs trading cointegration, hedge ratio cointegration
    example:
        qf.engle_granger(y=[10.0, 10.4, 10.1, 10.9, 11.3, 11.0, 11.8, 12.1, 11.7, 12.6, 12.9, 12.4],
                         x=[5.0, 5.3, 5.0, 5.5, 5.6, 5.5, 5.9, 6.1, 5.8, 6.3, 6.4, 6.2],
                         trend="constant", max_lag=1, lag_rule="aic")
    """
    choice("trend", trend, TRENDS)
    target = np.asarray(y, dtype=float).ravel()
    regressors = np.asarray(x, dtype=float)
    regressors = regressors[:, None] if regressors.ndim == 1 else regressors
    if regressors.ndim != 2 or regressors.shape[0] != target.size:
        raise ValueError("y must be (n,) and x (n,) or (n, k) with the same n")
    n = target.size
    terms = {"none": [], "constant": [np.ones(n)], "constant_trend": [np.ones(n), np.arange(1.0, n + 1)]}[trend]
    design = np.column_stack(terms + [regressors])
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(design))):
        raise ValueError("y and x must be finite (drop the rows with NaN first)")
    coef, _, _ = _ols_fit(target, design)
    resid = target - design @ coef
    if float(resid @ resid) <= 1e-12 * float(np.sum((target - target.mean()) ** 2)):
        raise ValueError("y is (almost) a linear function of x: the test is not defined")
    u, most = _unit_root_inputs(resid, "none", max_lag, lag_rule)
    statistic, _, lag, _ = _adf(u, "none", most, lag_rule)
    series_count = regressors.shape[1] + 1
    found = {"statistic": statistic,
             "p_value": (_mackinnon.p_value(statistic, trend, series_count) if series_count <= 6 else np.nan),
             "critical_values": (_mackinnon.critical_values(trend, series_count, n - 1) if trend != "none"
                                 else Result({level: np.nan for level in _mackinnon.LEVELS})),
             "lag": lag, "coefficients": coef, "residuals": resid}
    return Result(found)


BOOTSTRAPS = ("iid", "moving", "circular", "stationary")


def block_bootstrap(n_obs, kind, block_length, n_draws, rng, statistic):
    """Bootstrap resampling of a series, iid or in blocks (moving, circular, Politis-Romano stationary).

    Each draw resamples the positions 0 .. n - 1. 'iid': n uniform draws with replacement (Efron). Blocks keep
    neighbours together, so dependence within a block is kept: 'moving' (Kunsch) blocks of length l starting
    uniformly in 0 .. n - l, joined and cut to n; 'circular' starts in 0 .. n - 1, the series wrapped; 'stationary'
    the first position uniform, then each next a fresh start with probability p = 1/l, else the previous + 1
    (wrapped): geometric blocks of mean l. Under all but 'moving' every position is uniform. Data rows: x[indices];
    filtered historical simulation scales resampled standardised residuals by today's vol.

    An interval for any statistic (a Sharpe ratio, an alpha, a mean) from resampled returns: pass the statistic, then
    np.percentile(draws, [2.5, 97.5]) is the 95% percentile interval.

    conventions:
    - n_obs: n, the series' length; kind: 'iid', 'moving', 'circular' or 'stationary'
    - block_length: l, the block's length ('moving', 'circular'), or its mean 1/p ('stationary'; p = 0.1 is l = 10);
      1 or None for 'iid'
    - n_draws: the resamples; rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream)
    - statistic: None returns the positions; else a function of one draw's positions (lambda ix: r[ix].mean() for the
      mean of returns r), evaluated on each draw
    - draws: 'stationary' a uniform (n_draws, n) for the restarts, then the starts (n_draws, n) by integers; 'iid' the
      positions by integers; the blocks' starts (n_draws, blocks) otherwise
    returns: an n_draws x n array of positions, int; with a statistic, its value on each draw (n_draws values)
    source: Politis and Romano (1994), JASA 89(428); Kunsch (1989), Annals of Statistics 17(3); Politis and Romano
      (1992); Efron and Tibshirani, An Introduction to the Bootstrap (1993), ch. 13
    aliases: stationary bootstrap, block bootstrap, moving block bootstrap, circular block bootstrap, Politis-Romano, Politis Romano bootstrap, geometric block bootstrap, filtered historical simulation, iid bootstrap, bootstrap confidence interval, [bootstrap]
    example:
        qf.block_bootstrap(n_obs=500, kind="stationary", block_length=10.0, n_draws=200,
                           rng=7, statistic=None)
    """
    choice("kind", kind, BOOTSTRAPS)
    if statistic is not None and not callable(statistic):
        raise ValueError("statistic must be None (the positions) or a function of one draw's positions")
    n, draws = integer("n_obs", n_obs, 2), integer("n_draws", n_draws, 1)
    rng = _stream(rng)
    if kind == "iid":
        if block_length not in (None, 1):
            raise ValueError("block_length must be 1 or None for kind='iid' (each position its own block)")
        return _statistic_draws(rng.integers(0, n, (draws, n)), statistic)
    length = float(block_length)
    if not 1.0 <= length <= n:
        raise ValueError("block_length must lie inside [1, n_obs]")
    if kind == "stationary":
        restart = rng.random((draws, n)) < 1.0 / length
        starts = rng.integers(0, n, (draws, n))
        steps = np.arange(n)
        last = np.maximum.accumulate(np.where(restart, steps, 0), axis=1)     # position 0 starts the first block
        return _statistic_draws((np.take_along_axis(starts, last, axis=1) + steps - last) % n, statistic)
    size = int(round(length))
    blocks = -(-n // size)
    starts = rng.integers(0, n - size + 1 if kind == "moving" else n, (draws, blocks))
    positions = (starts[:, :, None] + np.arange(size)).reshape(draws, blocks * size)[:, :n]
    return _statistic_draws(positions % n, statistic)


def _statistic_draws(positions, statistic):
    """The positions where `statistic` is None, else its value on each draw's positions, as one array."""
    if statistic is None:
        return positions
    return np.asarray([statistic(row) for row in positions], dtype=float)


def ols_influence(y, X, intercept):
    """Influence diagnostics: leverage (hat values), Cook's distance, studentized residuals, DFBETAS, DFFITS, VIF.

    Leverage h_i = H_ii, H = X (X'X)^{-1} X' the hat matrix (the h_i sum to p, the columns, the ones column counted);
    e = y - X b, s^2 = e'e / (n - p); the deleted variance s_(i)^2 = ((n - p) s^2 - e_i^2 / (1 - h_i)) / (n - p - 1),
    the regression's without row i; internally studentized r_i = e_i / (s sqrt(1 - h_i)), externally
    t_i = e_i / (s_(i) sqrt(1 - h_i)); Cook's D_i = r_i^2 h_i / (p (1 - h_i)); DFBETA_i = b - b_(i) =
    (X'X)^{-1} x_i e_i / (1 - h_i), DFBETAS that over s_(i) sqrt(diag (X'X)^{-1}); DFFITS_i = t_i sqrt(h_i / (1 - h_i)).
    The variance inflation factor VIF_j = 1 / (1 - R_j^2), R_j^2 of regressor j on a constant and the other regressors
    (statsmodels' variance_inflation_factor, standardised as by its default), with or without an intercept. Flags:
    h_i above 2p/n, D_i above 4/n, |DFFITS_i| above 2 sqrt(p/n), a VIF above 5 or 10 (multicollinearity).

    conventions:
    - y: (n,); X: (n, k) regressors, or (n,) for one; rows with NaN refused (drop them first)
    - intercept: True puts a ones column first (p = k + 1, the constant counted); False fits through the origin (p = k)
    returns: a dict: leverage, cooks_distance, student_internal (r_i), student_external (t_i), dffits and
      sigma_deleted (s_(i)), one per row; dfbeta and dfbetas (n x p, the constant's column first); vif (one per
      regressor, the constant left out); p; sigma2 (s^2)
    source: Belsley, Kuh and Welsch, Regression Diagnostics (1980), ch. 2; Cook (1977), Technometrics 19(1); statsmodels'
      OLSInfluence and variance_inflation_factor
    aliases: Cook's distance, studentized residuals, DFBETAS, DFFITS, variance inflation, VIF, influence diagnostics, multicollinearity
    example:
        # The largest Cook's distance is about 1.532205.
        qf.ols_influence(y=[1.0, 2.1, 2.9, 4.2, 4.8, 6.5],
                         X=[[1.0, 0.5], [2.0, 0.4], [3.0, 1.1], [4.0, 0.9], [5.0, 1.6], [6.0, 1.2]],
                         intercept=True)
    """
    y, x = _design(y, X, intercept)
    n, p = x.shape
    if n < p + 2:
        raise ValueError("the deleted variance needs at least p + 2 rows")
    coef, resid, bread = _fit(y, x)
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    if np.any(leverage >= 1.0 - 1e-12):
        raise ValueError("a row has leverage 1 (it fits itself exactly): its deletion diagnostics are undefined")
    sigma2 = float(resid @ resid / (n - p))
    deleted = ((n - p) * sigma2 - resid ** 2 / (1.0 - leverage)) / (n - p - 1)
    internal = resid / np.sqrt(sigma2 * (1.0 - leverage))
    external = resid / np.sqrt(deleted * (1.0 - leverage))
    dfbeta = (x @ bread) * (resid / (1.0 - leverage))[:, None]
    regressors = x[:, 1:] if intercept else x
    vif = []
    for j in range(regressors.shape[1]):
        target = regressors[:, j]
        others = np.column_stack([np.ones(n), np.delete(regressors, j, axis=1)])
        rest = target - others @ np.linalg.lstsq(others, target, rcond=None)[0]
        vif.append(np.sum((target - target.mean()) ** 2) / (rest @ rest))
    return Result({"leverage": leverage, "cooks_distance": internal ** 2 * leverage / (p * (1.0 - leverage)),
                   "student_internal": internal, "student_external": external,
                   "dffits": external * np.sqrt(leverage / (1.0 - leverage)), "sigma_deleted": np.sqrt(deleted),
                   "dfbeta": dfbeta, "dfbetas": dfbeta / np.sqrt(deleted)[:, None] / np.sqrt(np.diag(bread)),
                   "vif": np.array(vif), "p": p, "sigma2": sigma2})


NW_DENOMINATORS = ("T", "T-1", "T-lag")
NW_SCALES = ("long_run", "mean")


def newey_west_cov(data, lags, denominator, demean, scale):
    """Newey-West covariance matrix of a multivariate series, and the Wald test of its means.

    The long-run (HAC) covariance with Bartlett weights: u_t = x_t - the mean (or x_t as given); Gamma_l = sum_{t>l}
    u_t u_{t-l}' / D, D the denominator; S = Gamma_0 + sum_{l=1}^{L} (1 - l/(L + 1)) (Gamma_l + Gamma_l'), positive
    semidefinite under 'T'; the covariance of the sample mean is S / T. Wald = mean' (S/T)^{-1} mean, chi-square with
    k degrees of freedom under a zero mean; t_j = mean_j / sqrt((S/T)_jj). L = 0 is the plain covariance (ddof 0
    under 'T', 1 under 'T-1').
    Used for the covariance of factor returns or of risk premia that are autocorrelated (annualise S or S / T by the
    periods a year where the task does), and to test that a vector of mean returns or premia is zero.

    conventions:
    - data: (T, k), rows the dates in time order, columns the series; (T,) for one series
    - lags: L, the last autocovariance kept, a whole number (no default rule: floor(4 (T/100)^{2/9}) is one, if the
      task names it)
    - denominator: D = T for every lag ('T', Newey and West's; statsmodels' S_hac_simple over T), T - 1 ('T-1', the
      sample covariance's at L = 0; newey_west_ols's 'T-k' on a constant), or T - l for lag l ('T-lag')
    - demean: True subtracts each series' mean first; False uses the data as given (a series whose mean is zero
      under the test, or a regression's scores)
    - scale: 'long_run' returns S; 'mean' returns S / T, the covariance of the sample mean
    - a singular covariance still returns cov; its joint Wald test and p_value are NaN
    returns: a dict: cov (k x k, S or S / T as scale says); mean (one per series); se (sqrt of the diagonal of S / T);
      tstat (mean / se); wald and p_value (the chi-square test that every mean is zero); lags
    source: Newey and West (1987), Econometrica 55(3); Hamilton, Time Series Analysis (1994), 10.5; Cochrane, Asset
      Pricing (2005), 11.7
    aliases: Newey-West covariance, Newey-West covariance matrix
    example:
        # cov is about [[0.0001015432, 0.0000132716], [0.0000132716, 0.0000608025]].
        qf.newey_west_cov(data=[[0.01, 0.02], [-0.01, 0.0], [0.02, 0.01], [0.0, -0.01], [0.01, 0.03], [-0.02, 0.0]],
                          lags=2, denominator="T", demean=True,
                          scale="long_run")
    """
    choice("denominator", denominator, NW_DENOMINATORS)
    choice("scale", scale, NW_SCALES)
    if not isinstance(demean, (bool, np.bool_)):
        raise ValueError(f"demean must be True or False, not {demean!r}")
    x = np.asarray(data, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    if x.ndim != 2 or x.shape[0] < 2 or not np.all(np.isfinite(x)):
        raise ValueError("data must be a finite (T, k) array with at least two rows, in time order")
    t, k = x.shape
    most = _lags(lags)
    if most >= t - 1:
        raise ValueError("lags must be below T - 1")
    mean = x.mean(axis=0)
    u = x - mean if demean else x

    def gamma(lag):
        return u[lag:].T @ u[:t - lag] / {"T": t, "T-1": t - 1, "T-lag": t - lag}[denominator]

    s = gamma(0)
    for lag in range(1, most + 1):
        g = gamma(lag)
        s = s + (1.0 - lag / (most + 1.0)) * (g + g.T)
    of_mean = s / t
    se = np.sqrt(np.diag(of_mean))
    wald = float(mean @ np.linalg.solve(of_mean, mean)) if np.linalg.matrix_rank(of_mean) == k else float("nan")
    return Result({"cov": s if scale == "long_run" else of_mean, "mean": mean, "se": se, "tstat": mean / se,
                   "wald": wald, "p_value": float(chi2.sf(wald, k)), "lags": most})


CSR_METHODS = ("ols", "gls")


def cross_sectional_r2(mean_returns, betas, method, cov, intercept):
    """Cross-sectional OLS or GLS R^2 of average returns on betas, and the premia.

    R the assets' mean returns, B = [1, betas] with an intercept (betas alone without); W = I ('ols') or cov^{-1}
    ('gls'); the premia gamma = (B'WB)^{-1} B'W R, the pricing errors e = R - B gamma; R^2 = 1 - e'We / (R - c)'W(R - c),
    c the W-weighted mean 1'WR / 1'W1 with an intercept (the plain mean under 'ols'), 0 without (uncentred). 'ols' is
    qf.ols's R^2 on the same regression; 'gls' weighs the assets by the inverse covariance, so a model cannot score by
    pricing the noisy assets alone (Lewellen, Nagel and Shanken). The second pass of a two-pass (Fama-MacBeth) test;
    qf.fama_macbeth gives the premia period by period.

    conventions:
    - mean_returns: (N,), each asset's average return over the task's period; betas: (N, K), or (N,) for one factor,
      from the time-series regressions (or their average over rolling windows)
    - method: 'ols' or 'gls'
    - cov: None for 'ols'; for 'gls' the positive definite N x N covariance whose inverse weighs: the time-series residual covariance
      (Cochrane's GLS) or the returns' covariance (Lewellen, Nagel and Shanken), as the task says
    - intercept: True puts a constant in the cross-sectional regression (the zero-beta rate); False forces it
      through the origin
    returns: a dict: r2; gamma (the intercept first, then one premium per factor); errors (the pricing errors, one per
      asset); fitted (B gamma)
    source: Cochrane, Asset Pricing (2005), 12.2; Lewellen, Nagel and Shanken (2010), J. Financial Economics 96(2);
      Kan, Robotti and Shanken (2013), J. Finance 68(6)
    aliases: cross-sectional R2, cross-sectional R-squared, GLS R2, GLS R-squared
    example:
        # r2 is about 0.988785; gamma is [-0.00025, 0.0115].
        qf.cross_sectional_r2(mean_returns=[0.010, 0.012, 0.008, 0.015], betas=[0.9, 1.1, 0.7, 1.3], method="ols",
                              cov=None,
                              intercept=True)
    """
    choice("method", method, CSR_METHODS)
    if not isinstance(intercept, (bool, np.bool_)):
        raise ValueError(f"intercept must be True or False, not {intercept!r}")
    r = np.asarray(mean_returns, dtype=float).ravel()
    b = np.asarray(betas, dtype=float)
    b = b[:, None] if b.ndim == 1 else b
    if b.ndim != 2 or b.shape[0] != r.size or not (np.all(np.isfinite(r)) and np.all(np.isfinite(b))):
        raise ValueError("mean_returns must be (N,) and betas (N, K) or (N,), finite")
    design = np.column_stack([np.ones(r.size), b]) if intercept else b
    if r.size <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("the cross-section needs more assets than coefficients and betas of full rank")
    if method == "ols":
        if cov is not None:
            raise ValueError("cov is for method='gls'; pass None with 'ols'")
        w = np.eye(r.size)
    else:
        c = np.asarray(cov, dtype=float) if cov is not None else np.zeros(0)
        if c.shape != (r.size, r.size) or not np.all(np.isfinite(c)) or not np.allclose(c, c.T):
            raise ValueError("cov must be a finite symmetric N x N matrix with method='gls'")
        try:
            np.linalg.cholesky(c)
        except np.linalg.LinAlgError as error:
            raise ValueError("cov must be positive definite with method='gls'") from error
        w = np.linalg.inv(c)
    gamma = np.linalg.solve(design.T @ w @ design, design.T @ w @ r)
    fitted = design @ gamma
    errors = r - fitted
    ones = np.ones(r.size)
    centred = r - (ones @ w @ r) / (ones @ w @ ones) if intercept else r
    return Result({"r2": float(1.0 - (errors @ w @ errors) / (centred @ w @ centred)), "gamma": gamma,
                   "errors": errors, "fitted": fitted})


VR_STATISTICS = ("homoskedastic", "robust")


def variance_ratio(log_prices, q, overlapping, bias_correction, z, demean):
    """Variance ratio test (Lo-MacKinlay): VR(q), the q-period variance over q one-period ones, and its z.

    From log prices p_0 .. p_{nq}: one-period returns r_t, mu their mean (the drift, or 0). sigma_a^2 = sum (r_t -
    mu)^2 / nq; the q-period sigma_c^2 = sum_t (p_t - p_{t-q} - q mu)^2 / (nq q) over every overlapping q-period
    change (over the n non-overlapping ones, / nq); with the bias correction nq - 1 and m = q (nq - q + 1)(1 - q/nq)
    replace nq and nq q. VR(q) = sigma_c^2 / sigma_a^2, 1 for a random walk, above 1 with positive autocorrelation
    (momentum), below with negative (mean reversion). z = sqrt(nq) (VR - 1) / sqrt(V): V = 2(2q - 1)(q - 1)/(3q)
    homoskedastic, 2(q - 1) non-overlapping, sum_{j<q} [2(q - j)/q]^2 delta_j robust to heteroskedasticity, delta_j =
    nq sum (r_t - mu)^2 (r_{t-j} - mu)^2 / [sum (r_t - mu)^2]^2. The p-value two-sided, standard normal.

    conventions:
    - log_prices: the log price levels, oldest first (from returns r: np.r_[0.0, np.cumsum(r)])
    - q: the holding period in periods, an int of 2 or more
    - overlapping: True uses every q-period change; False the non-overlapping ones (the extra prices at the end dropped)
    - bias_correction: True takes Lo and MacKinlay's unbiased variances (overlapping only); False the plain ones
    - z: 'homoskedastic' (Lo and MacKinlay's z(q)) or 'robust' (their z*(q), overlapping only)
    - demean: True estimates the drift mu by the mean return; False takes mu = 0 (a driftless walk)
    returns: a dict: vr; z; p_value; variance_one (sigma_a^2, per period); variance_q (q sigma_c^2, the q-period
      variance); nobs (nq, the returns used); q
    source: Lo and MacKinlay (1988), Review of Financial Studies 1(1); Campbell, Lo and MacKinlay, The Econometrics of
      Financial Markets (1997), 2.4; the arch package's VarianceRatio
    aliases: variance ratio test, Lo-MacKinlay, Lo MacKinlay test, VR test, [variance ratio]
    example:
        # vr is about 0.080983; z is -2.662377; p_value is 0.007759.
        qf.variance_ratio(log_prices=[4.60, 4.62, 4.61, 4.64, 4.63, 4.66, 4.65, 4.69, 4.68, 4.70],
                          q=2, overlapping=True, bias_correction=True, z="robust",
                          demean=True)
    """
    choice("z", z, VR_STATISTICS)
    for name, flag in (("overlapping", overlapping), ("bias_correction", bias_correction), ("demean", demean)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be True or False, not {flag!r}")
    if not overlapping and (bias_correction or z == "robust"):
        raise ValueError("non-overlapping changes take bias_correction=False and z='homoskedastic' (Lo and "
                         "MacKinlay give the bias correction and the robust z for overlapping ones)")
    y = np.asarray(log_prices, dtype=float).ravel()
    span = integer("q", q, 2)
    if not np.all(np.isfinite(y)):
        raise ValueError("log_prices must be finite (drop the NaN first)")
    if not overlapping:
        y = y[:y.size - (y.size - 1) % span]
    if y.size < (2 * span if not overlapping else span + 1 + int(bias_correction)) or y.size < 3:
        raise ValueError("too few prices for q")
    mu = (y[-1] - y[0]) / (y.size - 1) if demean else 0.0
    d = np.diff(y)
    nq = d.size
    one = float(np.sum((d - mu) ** 2) / nq)
    if overlapping:
        long = float(np.sum((y[span:] - y[:-span] - span * mu) ** 2) / (nq * span))
    else:
        long = float(np.sum((y[span::span] - y[:-span:span] - span * mu) ** 2) / nq)
    if bias_correction:
        one *= nq / (nq - 1.0)
        long *= nq * span / (span * (nq - span + 1) * (1.0 - span / nq))
    if not overlapping:
        variance = 2.0 * (span - 1)
    elif z == "homoskedastic":
        variance = 2.0 * (2 * span - 1) * (span - 1) / (3.0 * span)
    else:
        squares = (d - mu) ** 2
        variance = sum(4.0 * (1.0 - j / span) ** 2 * nq * (squares[j:] @ squares[:-j]) / np.sum(squares) ** 2
                       for j in range(1, span))
    ratio = long / one
    statistic = float(np.sqrt(nq) * (ratio - 1.0) / np.sqrt(variance))
    return Result({"vr": ratio, "z": statistic, "p_value": float(2.0 * norm.sf(abs(statistic))), "variance_one": one,
                   "variance_q": span * long, "nobs": nq, "q": span})


JOHANSEN_CASES = ("none", "constant", "linear_trend")
JOHANSEN_LEVELS = ("90%", "95%", "99%")
# MacKinnon, Haug and Michelis (1999)'s critical values at 90, 95 and 99%, as LeSage's johdist and statsmodels'
# coint_tables (0.15.0) carry them: a row for each of 1 to 12 non-cointegrated directions, a table a case
_JOHANSEN_TRACE = {
    "none": """2.9762 4.1296 6.9406 10.4741 12.3212 16.3640 21.7781 24.2761 29.5147 37.0339 40.1749 46.5716
        56.2839 60.0627 67.6367 79.5329 83.9383 92.7136 106.7351 111.7797 121.7375 137.9954 143.6691 154.7977
        173.2292 179.5199 191.8122 212.4721 219.4051 232.8291 255.6732 263.2603 277.9962 302.9054 311.1288 326.9716""",
    "constant": """2.7055 3.8415 6.6349 13.4294 15.4943 19.9349 27.0669 29.7961 35.4628 44.4929 47.8545 54.6815
        65.8202 69.8189 77.8202 91.1090 95.7542 104.9637 120.3673 125.6185 135.9825 153.6341 159.5290 171.0905
        190.8714 197.3772 210.0366 232.1030 239.2468 253.2526 277.3740 285.1402 300.2821 326.5354 334.9795 351.2150""",
    "linear_trend": """2.7055 3.8415 6.6349 16.1619 18.3985 23.1485 32.0645 35.0116 41.0815 51.6492 55.2459 62.5202
        75.1027 79.3422 87.7748 102.4674 107.3429 116.9829 133.7852 139.2780 150.0778 169.0618 175.1584 187.1891
        208.3582 215.1268 228.2226 251.6293 259.0267 273.3838 298.8836 306.8988 322.4264 350.1125 358.7190 375.3203"""}
_JOHANSEN_MAX = {
    "none": """2.9762 4.1296 6.9406 9.4748 11.2246 15.0923 15.7175 17.7961 22.2519 21.8370 24.1592 29.0609
        27.9160 30.4428 35.7359 33.9271 36.6301 42.2333 39.9085 42.7679 48.6606 45.8930 48.8795 55.0335
        51.8528 54.9629 61.3449 57.7954 61.0404 67.6415 63.7248 67.0756 73.8856 69.6513 73.0946 80.0937""",
    "constant": """2.7055 3.8415 6.6349 12.2971 14.2639 18.5200 18.8928 21.1314 25.8650 25.1236 27.5858 32.7172
        31.2379 33.8777 39.3693 37.2786 40.0763 45.8662 43.2947 46.2299 52.3069 49.2855 52.3622 58.6634
        55.2412 58.4332 64.9960 61.2041 64.5040 71.2525 67.1307 70.5392 77.4877 73.0563 76.5734 83.7105""",
    "linear_trend": """2.7055 3.8415 6.6349 15.0006 17.1481 21.7465 21.8731 24.2522 29.2631 28.2398 30.8151 36.1930
        34.4202 37.1646 42.8612 40.5244 43.4183 49.4095 46.5583 49.5875 55.8171 52.5858 55.7302 62.1741
        58.5316 61.8051 68.5030 64.5292 67.9040 74.7434 70.4630 73.9355 81.0678 76.4081 79.9878 87.2395"""}


def _johansen_critical(table, case, directions):
    """The (90%, 95%, 99%) critical values for `directions` non-cointegrated directions, nan past twelve."""
    values = np.array(table[case].split(), dtype=float).reshape(-1, 3)
    return values[directions - 1] if 1 <= directions <= values.shape[0] else np.full(3, np.nan)


def _detrended(a, order):
    """a less its least-squares polynomial in time of `order` (-1: a itself; 0: demeaned; 1: a linear trend out)."""
    if order < 0 or a.shape[1] == 0:
        return a
    design = np.vander(np.linspace(-1.0, 1.0, a.shape[0]), order + 1)
    return a - design @ np.linalg.lstsq(design, a, rcond=None)[0]


def johansen(levels, deterministic, lagged_differences):
    """Johansen cointegration test (trace and eigenvalue statistics): rank, critical values, cointegrating vectors.

    The VECM dx_t = Pi x_{t-1} + sum_{i=1}^{k} G_i dx_{t-i} + e_t: R0 and R1 the residuals of dx_t and of x_{t-1} on
    the lagged differences (after the deterministic case's terms), S_ij = R_i'R_j / n; the eigenvalues lambda_1 >= ...
    of S11^{-1} S10 S00^{-1} S01 and their vectors v (v'S11 v = 1, the cointegrating vectors). For H0 rank <= r: trace
    = -n sum_{i>r} ln(1 - lambda_i), maximum eigenvalue = -n ln(1 - lambda_{r+1}), n the regression's rows; rejected
    above the critical value for k - r directions. The rank is the first r not rejected at 95%. statsmodels'
    coint_johansen to rounding (its det_order -1, 0, 1) with one lagged difference or more.

    conventions:
    - levels: (T, k), the levels (prices or log prices), oldest first, k series (twelve at most for critical values)
    - deterministic: 'none' (no deterministic terms; det_order -1), 'constant' (the differences and the lagged levels
      demeaned; det_order 0), 'linear_trend' (the levels first detrended by a linear trend; det_order 1)
    - lagged_differences: k, the lagged differences in the VECM (k_ar_diff; a VAR in levels of order p has p - 1)
    returns: a dict: trace and max_eigen (one per r = 0 .. k - 1); trace_critical and max_eigen_critical (k x 3: 90%,
      95%, 99%); eigenvalues (decreasing); vectors (k x k, column i the i-th cointegrating vector, its first nonzero
      entry positive); rank_trace and rank_max_eigen (the rank each test gives at 95%); nobs (n)
    source: Johansen (1991), Econometrica 59(6); MacKinnon, Haug and Michelis (1999), J. Applied Econometrics 14(5),
      the critical values; Hamilton, Time Series Analysis (1994), 20.2
    aliases: Johansen test, Johansen cointegration test, trace test, maximum eigenvalue test, coint_johansen, cointegration rank, [VECM]
    example:
        # trace is about [11.371470, 2.279423]; rank_trace is 0 at 95%.
        qf.johansen(levels=[[10.0, 5.0], [10.4, 5.3], [10.1, 5.0], [10.9, 5.5], [11.3, 5.6], [11.0, 5.5],
                            [11.8, 5.9], [12.1, 6.1], [11.7, 5.8], [12.6, 6.3], [12.9, 6.4], [12.4, 6.2]],
                    deterministic="constant", lagged_differences=1)
    """
    choice("deterministic", deterministic, JOHANSEN_CASES)
    x = np.asarray(levels, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("levels must be a finite (T, k) array, oldest first")
    if x.shape[1] > 12:
        raise ValueError("critical values support at most twelve series; the rank cannot be determined above twelve")
    lags = _lags(lagged_differences)
    order = JOHANSEN_CASES.index(deterministic) - 1
    inner = min(order, 0)
    xl = _detrended(x, order)
    dx = np.diff(xl, axis=0)
    rows = np.arange(lags, dx.shape[0])
    if rows.size <= x.shape[1] * (lags + 1) + 1:
        raise ValueError("too few rows for the series and lagged_differences")
    z = np.column_stack([dx[rows - j] for j in range(1, lags + 1)]) if lags else np.empty((rows.size, 0))
    z = _detrended(z, inner)

    def residual(a):
        a = _detrended(a, inner)
        return a - z @ np.linalg.lstsq(z, a, rcond=None)[0] if z.shape[1] else a

    r0, r1 = residual(dx[rows]), residual(xl[rows])
    n = rows.size
    s00, s11, s01 = r0.T @ r0 / n, r1.T @ r1 / n, r0.T @ r1 / n
    values, vectors = linalg.eigh(s01.T @ np.linalg.solve(s00, s01), s11)
    values, vectors = np.clip(values[::-1], 0.0, 1.0 - 1e-15), vectors[:, ::-1]
    first = vectors[np.argmax(np.abs(vectors) > 1e-12 * np.abs(vectors).max(axis=0), axis=0), np.arange(x.shape[1])]
    vectors = vectors * np.where(first < 0, -1.0, 1.0)
    logs = np.log1p(-values)
    trace = -n * np.cumsum(logs[::-1])[::-1]
    largest = -n * logs
    k = x.shape[1]
    trace_cv = np.array([_johansen_critical(_JOHANSEN_TRACE, deterministic, k - r) for r in range(k)])
    max_cv = np.array([_johansen_critical(_JOHANSEN_MAX, deterministic, k - r) for r in range(k)])

    def rank(statistic, critical):
        below = np.nonzero(~(statistic > critical[:, 1]))[0]
        return int(below[0]) if below.size else k

    return Result({"trace": trace, "max_eigen": largest, "trace_critical": trace_cv, "max_eigen_critical": max_cv,
                   "eigenvalues": values, "vectors": vectors, "rank_trace": rank(trace, trace_cv),
                   "rank_max_eigen": rank(largest, max_cv), "nobs": n})


TEST_CORRECTIONS = ("bonferroni", "sidak", "holm", "hochberg", "benjamini_hochberg", "benjamini_yekutieli")


@_guard(level=('level',))
def multiple_testing(p_values, method, level):
    """Multiple testing corrections: adjusted p-values and rejections by Bonferroni, Holm, Benjamini-Hochberg (FDR).

    m p-values, p_(1) <= ... <= p_(m) sorted. 'bonferroni' m p; 'sidak' 1 - (1 - p)^m; 'holm' step-down, the running
    maximum of (m - i + 1) p_(i); 'hochberg' step-up, the running minimum from the largest of (m - i + 1) p_(i);
    'benjamini_hochberg' the false discovery rate's step-up, the running minimum from the largest of m p_(i) / i;
    'benjamini_yekutieli' that times sum_{j<=m} 1/j (any dependence). Each capped at 1 and returned in the input's
    order; a test is rejected where its adjusted p-value is at most the level. The first four bound the family-wise
    error rate, the last two the false discovery rate. One test leaves p unchanged.

    conventions:
    - p_values: the tests' p-values, any order
    - method: 'bonferroni', 'sidak', 'holm', 'hochberg', 'benjamini_hochberg' (BH, FDR) or 'benjamini_yekutieli'
    - level: the family-wise error rate or the false discovery rate to hold (0.05, say)
    returns: a dict: adjusted (one per test, in the input's order); reject (adjusted <= level, one per test);
      n_rejected
    source: Holm (1979), Scandinavian J. Statistics 6(2); Hochberg (1988), Biometrika 75(4); Benjamini and Hochberg
      (1995), JRSS B 57(1); Benjamini and Yekutieli (2001), Annals of Statistics 29(4); statsmodels' multipletests
    aliases: multiple testing, Bonferroni correction, Holm-Bonferroni, Holm correction, Benjamini-Hochberg, false discovery rate, FDR, Sidak correction, Benjamini-Yekutieli, Hochberg, [adjusted p-values]
    example:
        # adjusted is about [0.04, 0.053333, 0.053333, 0.2]; only the first is rejected.
        qf.multiple_testing(p_values=[0.01, 0.04, 0.03, 0.20], method="benjamini_hochberg",
                            level=0.05)
    """
    choice("method", method, TEST_CORRECTIONS)
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0 or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("p_values must be one or more numbers inside [0, 1]")
    if not 0 < float(level) < 1:
        raise ValueError("level must lie inside (0, 1)")
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    i = np.arange(1, m + 1)
    if method == "bonferroni":
        adjusted = ranked * m
    elif method == "sidak":
        adjusted = -np.expm1(m * np.log1p(-ranked))
    elif method == "holm":
        adjusted = np.maximum.accumulate((m - i + 1) * ranked)
    elif method == "hochberg":
        adjusted = np.minimum.accumulate(((m - i + 1) * ranked)[::-1])[::-1]
    else:
        factor = np.sum(1.0 / i) if method == "benjamini_yekutieli" else 1.0
        adjusted = np.minimum.accumulate((ranked * m * factor / i)[::-1])[::-1]
    found = np.empty(m)
    found[order] = np.minimum(adjusted, 1.0)
    reject = found <= float(level)
    return Result({"adjusted": found, "reject": reject, "n_rejected": int(reject.sum())})
