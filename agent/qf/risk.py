"""VaR and ES: historical, parametric, Cornish-Fisher, EWMA, kernel, delta-gamma, normal mixture, spectral, copula, mapped.

`returns` are gains-positive (returns or P&L), oldest first; an argument named `losses` is losses-positive. A VaR or
ES function takes `alpha`, the confidence level (0.99), and `sign`: 'loss_positive' reports a 2% loss as 0.02,
'return_negative' as -0.02. Where a sample quantile is taken its rule is named (`method`, numpy's names) and so is the
ES tail (`es_rule`); 1 - alpha is rounded to 12 places, so 1 - 0.95 is 0.05 exactly. The backtests are on the
`backtests` shelf, Hill and the GPD on `tails`, Basel's capital on `capital`; their old paths here still work.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, stats
from scipy.special import ndtr, ndtri, stdtrit

from ._common import Result, choice, out, positive
from .backtests import var_backtest  # noqa: F401 - moved to backtests; qf.risk.var_backtest still resolves
from .capital import ima_capital  # noqa: F401 - moved to capital
from .conventions import QUANTILE_METHODS, SIGNS, _finite, _tail, ewma
from .copulas import _correlation, _draws, gaussian_copula_sample, t_copula_sample
from .tails import gpd_fit, hill_estimator  # noqa: F401 - moved to tails
from .primitives import _guard

__all__ = ["age_weighted_var_es", "cash_flow_map", "copula_loss_simulation", "cornish_fisher_var_es",
           "delta_gamma_var", "ewma_var_es", "historical_var_es", "kde_var_es", "mixture_normal_var_es",
           "normal_var_es", "spectral_risk", "t_var_es"]

ES_RULES = ("at_or_beyond", "beyond", "exact")
AGE_METHODS = ("inverted_cdf", "hazen")
SIGMA_KINDS = ("scale", "std")
BANDWIDTHS = ("scott", "silverman")
DELTA_GAMMA_METHODS = ("cornish_fisher", "simulation")


def _series(name: str, value) -> np.ndarray:
    series = _finite(name, value)
    if series.ndim != 1 or series.size < 2:
        raise ValueError(f"{name} must be a 1-D series of at least two values")
    return series


def _signed(sign: str, var_return, es_return):
    """(VaR, ES) from the return-space quantile and tail mean (both negative for a loss), in the sign asked."""
    if sign == "loss_positive":
        return out(-np.asarray(var_return)), out(-np.asarray(es_return))
    return out(var_return), out(es_return)


def _tail_mean(r_sorted, weights, cum, tail, q, es_rule):
    """The ES's tail mean in return space over sorted returns with weights summing to 1 and cumulative weights cum."""
    if es_rule == "exact":
        k = min(int(np.searchsorted(cum, tail, side="left")), r_sorted.size - 1)
        below = cum[k - 1] if k else 0.0
        return float((np.sum(weights[:k] * r_sorted[:k]) + (tail - below) * r_sorted[k]) / tail)
    mask = r_sorted <= q if es_rule == "at_or_beyond" else r_sorted < q
    if not np.any(mask):
        raise ValueError("no return lies beyond the VaR quantile; es_rule 'at_or_beyond' or 'exact' has one")
    return float(np.sum(weights[mask] * r_sorted[mask]) / np.sum(weights[mask]))


# ------------------------------------------------------------------------------------------------ VaR and ES


@_guard(level=('alpha',))
def historical_var_es(returns, *, alpha, method, es_rule, sign):
    """VaR and ES by historical simulation of a return or P&L sample, the quantile, tail and sign rules named.

    q = numpy.quantile(returns, 1 - alpha, method=method); 'lower' selects the more extreme gain. ES averages
    r <= q ('at_or_beyond') or r < q ('beyond'). 'exact' is the empirical tail integral, independent of method:
    [sum(m worst) + (n (1 - alpha) - m) next] / [n (1 - alpha)], m = floor(n (1 - alpha)).

    ES is subadditive, ES(X + Y) <= ES(X) + ES(Y), and VaR need not be: the diversification benefit is the sum of
    the positions' stand-alone figures less the portfolio's.
    conventions:
        alpha: the confidence level, e.g. 0.99 (the tail is 1 - alpha)
        method: 'linear', 'lower', 'higher', 'nearest', 'midpoint', 'hazen', 'weibull' or 'inverted_cdf'
        es_rule: 'at_or_beyond', 'beyond' or 'exact'
        sign: 'loss_positive' (VaR and ES of a loss as positive numbers) or 'return_negative'; returns are gains:
            losses go in as -losses ('lower' then their 'higher')
    returns: (var, es)
    source: Jorion, Value at Risk (2007), historical simulation; Acerbi and Tasche (2002), "On the coherence of
        expected shortfall", Journal of Banking and Finance 26(7) (the exact empirical ES)
    aliases: historical VaR, historical simulation, HS VaR, historical expected shortfall, historical CVaR, [CVaR from hist]
    example:
        import qf
        r = [-0.021, 0.004, -0.013, 0.008, -0.030, 0.011, -0.002, 0.006, -0.017, 0.012]
        qf.historical_var_es(r, alpha=0.9, method="linear", es_rule="at_or_beyond", sign="loss_positive")
    """
    choice("method", method, QUANTILE_METHODS)
    choice("es_rule", es_rule, ES_RULES)
    choice("sign", sign, SIGNS)
    r = np.sort(_series("returns", returns))
    tail = _tail(alpha)
    q = float(np.quantile(r, tail, method=method))
    n = r.size
    es = _tail_mean(r, np.full(n, 1.0 / n), np.arange(1, n + 1) / n, tail, q, es_rule)
    return _signed(sign, q, es)


@_guard(level=('alpha',))
def normal_var_es(*, mu, sigma, alpha, sign):
    """VaR and ES under a normal distribution (parametric) from a mean and std of returns, sign named.

    With z = Phi^-1(1 - alpha) (negative): the VaR quantile mu + sigma z, the ES tail mean mu - sigma phi(z) /
    (1 - alpha). An h-period horizon of i.i.d. returns: see mu; a portfolio's sigma is sqrt(w' Sigma w).

    Liquidity-adjusted VaR adds the cost of closing out at half the bid-offer spread: VaR + sum_i s_i a_i / 2, s_i the
    proportional spread (offer - bid over mid) and a_i the position's value; with a random spread, s_i = mu_s + k
    sigma_s at a chosen k. ES is coherent (monotone, subadditive, positively homogeneous, translation invariant);
    VaR need not be subadditive.
    conventions:
        mu: the mean gain per period (a loss sample's mean negated; 0 for zero mean); h periods: mu h and sigma
            sqrt(h) drift-aware, or mu sqrt(h) and sigma sqrt(h), the whole 1-period VaR times sqrt(h)
        sigma: the return standard deviation per period, above 0 (its ddof is the caller's)
        alpha: the confidence level, e.g. 0.99
        sign: 'loss_positive' or 'return_negative'
    returns: (var, es)
    source: Jorion, Value at Risk (2007), parametric VaR; McNeil, Frey and Embrechts, Quantitative Risk Management
        (2005), example 2.14 (the normal ES); Hull, Risk Management and Financial Institutions, "Liquidity-adjusted
        VaR"; Artzner, Delbaen, Eber and Heath (1999), Mathematical Finance 9(3) (coherence)
    aliases: parametric VaR, normal VaR, Gaussian VaR, variance-covariance VaR, delta-normal VaR, [parametric VaR ES]
    example:
        import qf
        mu, sigma = 0.0004, 0.012
        qf.normal_var_es(mu=mu, sigma=sigma, alpha=0.99, sign="loss_positive")
    """
    choice("sign", sign, SIGNS)
    sigma = positive("sigma", sigma)
    tail = _tail(alpha)
    z = ndtri(tail)
    mu = np.asarray(mu, dtype=float)
    return _signed(sign, mu + sigma * z, mu - sigma * stats.norm.pdf(z) / tail)


@_guard(level=('alpha',))
def t_var_es(*, mu, sigma, nu, alpha, sigma_kind, sign):
    """Student-t VaR and ES, the t's dispersion given as its scale or its standard deviation.

    With s the scale (sigma itself for 'scale', sigma sqrt((nu - 2)/nu) for 'std') and t_q = t_nu^-1(1 - alpha): the
    VaR quantile mu + s t_q, the ES tail mean mu - s g_nu(t_q) (nu + t_q^2) / ((nu - 1)(1 - alpha)), g_nu the t
    density. Needs nu > 1 (nu > 2 for 'std').
    conventions:
        mu: the location (mean) per period, of gains (a loss sample's mean negated)
        sigma: the dispersion per period, above 0: the t's scale or the returns' standard deviation, as sigma_kind
            says
        nu: the degrees of freedom
        alpha: the confidence level, e.g. 0.99
        sigma_kind: 'scale' where sigma is the t's scale parameter, as a fit gives it:
            scipy.stats.t.fit gives (df, loc, scale): nu=df, mu=loc, sigma=scale, 'scale'; or 'std' where sigma is
            the returns' standard deviation (the standardized t, scaled by sigma sqrt((nu - 2)/nu), the variance
            correction)
        sign: 'loss_positive' or 'return_negative'
    returns: (var, es): with s the scale and t_q = t_nu^-1(1 - alpha), VaR = -(mu + s t_q) and ES = s g_nu(t_q)
        (nu + t_q^2) / ((nu - 1)(1 - alpha)) - mu as losses ('loss_positive'); g_nu the t density
    source: McNeil, Frey and Embrechts, Quantitative Risk Management (2005), example 2.18 (the t ES)
    aliases: Student-t VaR, [t VaR], Student t expected shortfall, t-distribution VaR, fat-tailed VaR
    example:
        import qf
        nu, mu, scale = 4.0, 0.0, 0.009
        qf.t_var_es(mu=mu, sigma=scale, nu=nu, alpha=0.99, sigma_kind="scale", sign="loss_positive")  # VaR .03372253; ES .04698526
    """
    choice("sigma_kind", sigma_kind, SIGMA_KINDS)
    choice("sign", sign, SIGNS)
    sigma = positive("sigma", sigma)
    nu = np.asarray(nu, dtype=float)
    floor = 2.0 if sigma_kind == "std" else 1.0
    if np.any(~(nu > floor)):
        raise ValueError(f"nu must exceed {floor:g} for sigma_kind={sigma_kind!r}")
    tail = _tail(alpha)
    scale = sigma if sigma_kind == "scale" else sigma * np.sqrt((nu - 2.0) / nu)
    tq = stdtrit(nu, tail)
    mu = np.asarray(mu, dtype=float)
    shortfall = stats.t.pdf(tq, nu) * (nu + tq * tq) / ((nu - 1.0) * tail)
    return _signed(sign, mu + scale * tq, mu - scale * shortfall)


@_guard(level=('alpha',))
def cornish_fisher_var_es(*, mu, sigma, skew, excess_kurtosis, alpha, sign):
    """Cornish-Fisher (modified) VaR, and the ES of the Cornish-Fisher quantile function.

    With z = Phi^-1(1 - alpha), S the skewness and K the excess kurtosis: z_cf = z + (z^2 - 1) S/6 + (z^3 - 3z) K/24
    - (2 z^3 - 5z) S^2/36 and the VaR quantile mu + sigma z_cf. The ES is the mean of that quantile function over
    the tail, (1/(1 - alpha)) int_0^{1 - alpha} q_cf(u) du, in closed form: mu + sigma phi(z) / (1 - alpha) [-1 - z S/6
    + (1 - z^2) K/24 - (1 - 2 z^2) S^2/36] (not Boudt, Peterson and Croux's Edgeworth-density ES). Valid where q_cf is
    increasing (moderate S and K). S = K = 0 is the normal.

    The Jarque-Bera test of normality on the same moments: JB = n/6 (S^2 + K^2/4) ~ chi2(2) under normality, S and K
    the sample skewness and excess kurtosis from population (biased, ddof 0) moments, as scipy.stats.skew(x) and
    scipy.stats.kurtosis(x) give them: scipy.stats.jarque_bera(x) returns (statistic, pvalue) on these.
    conventions:
        mu: the mean return per period
        sigma: the return standard deviation per period, above 0
        skew: the skewness S (its bias correction is the caller's: scipy.stats.skew(bias=...))
        excess_kurtosis: K = kurtosis - 3 (scipy.stats.kurtosis(fisher=True))
        alpha: the confidence level, e.g. 0.99
        sign: 'loss_positive' or 'return_negative'
    returns: (var, es)
    source: Cornish and Fisher (1938), Revue de l'Institut International de Statistique 5; Zangari (1996),
        RiskMetrics Monitor Q4 (modified VaR); Maillard (2012), "A user's guide to the Cornish Fisher expansion";
        Jarque and Bera (1987), International Statistical Review 55(2)
    aliases: Cornish-Fisher, Cornish Fisher VaR, modified VaR, mVaR, CF expansion, Jarque-Bera, Jarque-Bera test
    example:
        import qf
        mu, sigma = 0.0003, 0.011
        qf.cornish_fisher_var_es(mu=mu, sigma=sigma, skew=-0.4, excess_kurtosis=2.1, alpha=0.99, sign="loss_positive")
    """
    choice("sign", sign, SIGNS)
    sigma = positive("sigma", sigma)
    tail = _tail(alpha)
    s, k = np.asarray(skew, dtype=float), np.asarray(excess_kurtosis, dtype=float)
    z = ndtri(tail)
    z_cf = z + (z * z - 1.0) * s / 6.0 + (z ** 3 - 3.0 * z) * k / 24.0 - (2.0 * z ** 3 - 5.0 * z) * s * s / 36.0
    bracket = -1.0 - z * s / 6.0 + (1.0 - z * z) * k / 24.0 - (1.0 - 2.0 * z * z) * s * s / 36.0
    mu = np.asarray(mu, dtype=float)
    return _signed(sign, mu + sigma * z_cf, mu + sigma * stats.norm.pdf(z) / tail * bracket)


@_guard(level=('alpha',))
def delta_gamma_var(*, delta, gamma, covariance, horizon, alpha, method, n, rng, sign):
    """Delta-gamma VaR and ES: the quadratic P&L by Cornish-Fisher or by simulation.

    P&L ~ delta' X + X' Gamma X / 2 over the horizon, X ~ N(0, Sigma h) the risk factors' changes (a vega is one more
    factor, the vol's change; cross gammas are Gamma's off-diagonal). With A = Sigma h: mean tr(Gamma A)/2; variance
    delta' A delta + tr((Gamma A)^2)/2; third cumulant 3 delta' A Gamma A delta + tr((Gamma A)^3); fourth 12 delta' A
    (Gamma A)^2 delta + 3 tr((Gamma A)^4). One factor of sd s: s^2 (delta^2 + gamma^2 s^2 / 2) and 3 delta^2 gamma s^4
    + gamma^3 s^6. 'cornish_fisher' passes the four to cornish_fisher_var_es; 'simulation' draws X = Z L' (Sigma h
    = L L') and takes historical_var_es of the P&L ('inverted_cdf', 'exact'). Gamma = 0 is delta-normal VaR.
    conventions:
        delta: dP/dX per factor (a number, or one per factor), in P&L per unit of X
        gamma: d2P/dX2 (a number, or the factors' matrix, cross terms off the diagonal)
        covariance: of one period's factor changes, in X's units (a number, the variance, for one factor)
        horizon: periods; the square-root-of-time rule scales the covariance by it
        alpha: the confidence level, e.g. 0.99
        method: 'cornish_fisher' (n and rng None) or 'simulation'
        n: the draws ('simulation')
        rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); its standard normals, n x
            factors ('simulation')
        sign: 'loss_positive' or 'return_negative'
    returns: a dict: var, es; mean, std, skew, excess_kurtosis (the approximate P&L's)
    source: Britten-Jones and Schaefer (1999), Review of Finance 2(2); Jorion (2007), Value at Risk, ch. 10; Zangari (1996)
    aliases: delta-gamma VaR, delta gamma VaR, delta-gamma-vega VaR, quadratic VaR, Cornish-Fisher delta-gamma, delta-gamma approximation, gamma VaR, option portfolio VaR
    example:
        import qf
        s = 100.0 * 0.02
        qf.delta_gamma_var(delta=50.0, gamma=-8.0, covariance=s * s, horizon=10, alpha=0.99, method="cornish_fisher", n=None, rng=None, sign="loss_positive")
    """
    choice("method", method, DELTA_GAMMA_METHODS)
    choice("sign", sign, SIGNS)
    b = np.atleast_1d(np.asarray(delta, dtype=float)).ravel()
    g = np.atleast_2d(np.asarray(gamma, dtype=float))
    cov = np.atleast_2d(np.asarray(covariance, dtype=float))
    k = b.size
    if g.shape != (k, k) or cov.shape != (k, k) or not (np.all(np.isfinite(g)) and np.all(np.isfinite(cov))):
        raise ValueError("gamma and covariance must be k x k for k deltas (numbers for one factor)")
    if not (np.allclose(g, g.T) and np.allclose(cov, cov.T)) or np.linalg.eigvalsh(cov).min() < -1e-12 * abs(cov).max():
        raise ValueError("gamma must be symmetric and covariance symmetric positive semidefinite")
    a = cov * float(positive("horizon", horizon))
    ga = g @ a
    powers = [None, ga, ga @ ga]
    cumulants = (0.5 * np.trace(ga), b @ a @ b + 0.5 * np.trace(powers[2]),
                 3.0 * b @ a @ ga @ b + np.trace(powers[2] @ ga),
                 12.0 * b @ a @ powers[2] @ b + 3.0 * np.trace(powers[2] @ powers[2]))
    mean, variance = float(cumulants[0]), float(cumulants[1])
    if not variance > 0:
        raise ValueError("the P&L's variance is zero: no factor moves it")
    skew, kurt = float(cumulants[2] / variance ** 1.5), float(cumulants[3] / variance ** 2)
    if method == "cornish_fisher":
        if n is not None or rng is not None:
            raise ValueError("n and rng are for 'simulation': pass None")
        var, es = cornish_fisher_var_es(mu=mean, sigma=math.sqrt(variance), skew=skew, excess_kurtosis=kurt,
                                        alpha=alpha, sign=sign)
    else:
        draws, rng = _draws(n, rng)
        values, vectors = np.linalg.eigh(a)
        x = rng.standard_normal((draws, k)) @ (vectors * np.sqrt(np.clip(values, 0.0, None))).T
        pnl = x @ b + 0.5 * np.einsum("ij,jk,ik->i", x, g, x)
        var, es = historical_var_es(pnl, alpha=alpha, method="inverted_cdf", es_rule="exact", sign=sign)
    return Result({"var": float(var), "es": float(es), "mean": mean, "std": math.sqrt(variance), "skew": skew,
                   "excess_kurtosis": kurt})


@_guard(level=('alpha',))
def age_weighted_var_es(returns, *, lam, alpha, method, es_rule, sign):
    """Age-weighted historical VaR and ES (Boudoukh-Richardson-Whitelaw): weights by age.

    The return i periods before the last (i = 0 the newest) weighs lam^i (1 - lam) / (1 - lam^n); lam = 1 is equal
    weights. Sort the returns, carrying each one's weight, and accumulate: 'inverted_cdf' takes the first return at
    which the cumulative weight reaches 1 - alpha (Hull's rule; QuantLib's weighted percentile), 'hazen'
    interpolates linearly between the returns placed at their weight's midpoint (cumulative weight - own weight /
    2), clamped at the ends. With equal weights the two are numpy's 'inverted_cdf' and 'hazen'. ES: 'at_or_beyond'
    and 'beyond' the weighted mean of r <= q or r < q; 'exact' the weighted distribution's own ES (the boundary
    return weighted by the part of its weight inside the tail).
    conventions:
        lam: the decay, inside (0, 1] (1 for equal weights; 0.98 and 0.97 are BRW's)
        alpha: the confidence level, e.g. 0.99
        method: 'inverted_cdf' or 'hazen'
        es_rule: 'at_or_beyond', 'beyond' or 'exact'
        sign: 'loss_positive' or 'return_negative'; returns are gains: losses go in as -losses
    returns: (var, es)
    source: Boudoukh, Richardson and Whitelaw (1998), "The best of both worlds", Risk 11(5); Hull, Risk Management
        and Financial Institutions, "Weighting of observations"
    aliases: age-weighted VaR, age weighted historical simulation, BRW, hybrid historical simulation
    example:
        import qf
        r = [-0.021, 0.004, -0.013, 0.008, -0.030, 0.011, -0.002, 0.006, -0.017, 0.012]
        qf.age_weighted_var_es(r, lam=0.98, alpha=0.9, method="inverted_cdf", es_rule="exact", sign="loss_positive")
    """
    choice("method", method, AGE_METHODS)
    choice("es_rule", es_rule, ES_RULES)
    choice("sign", sign, SIGNS)
    r = _series("returns", returns)
    lam = float(lam)
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"lam must lie inside (0, 1], not {lam}")
    tail = _tail(alpha)
    raw = lam ** np.arange(r.size - 1, -1, -1, dtype=float)
    order = np.argsort(r, kind="stable")
    r_sorted, raw_sorted = r[order], raw[order]
    total = raw.sum()
    weights, cum = raw_sorted / total, np.cumsum(raw_sorted) / total
    if method == "inverted_cdf":
        q = float(r_sorted[min(int(np.searchsorted(cum, tail, side="left")), r.size - 1)])
    else:
        q = float(np.interp(tail, cum - weights / 2.0, r_sorted))
    return _signed(sign, q, _tail_mean(r_sorted, weights, cum, tail, q, es_rule))


@_guard(level=('alpha',))
def ewma_var_es(returns, *, lam, seed, alpha, sign):
    """RiskMetrics EWMA VaR and ES: normal, zero mean, the variance an EWMA of squared returns.

    sigma^2_{n} = lam sigma^2_{n-1} + (1 - lam) r^2_{n-1} run through the last return (qf.ewma on r^2
    with the seed named), so sigma is the forecast for the next period; then the normal VaR and ES with mu = 0: -z
    sigma and sigma phi(z) / (1 - alpha), z = Phi^-1(1 - alpha). Squared returns, not squared deviations: the mean is
    taken as 0 throughout. The vol path itself is qf.ewma_volatility.
    conventions:
        lam: the decay, inside (0, 1) (RiskMetrics 0.94 daily, 0.97 monthly)
        seed: the variance before the first return: 'first' r_0^2, 'zero', 'mean' the mean of r^2, 'adjusted'
            (pandas adjust=True) or a number (e.g. a sample variance)
        alpha: the confidence level, e.g. 0.99
        sign: 'loss_positive' or 'return_negative'
    returns: (var, es)
    source: J.P. Morgan/Reuters, RiskMetrics Technical Document (1996), 5.2.1 and 6.2; Hull, Options, Futures, and
        Other Derivatives, "The exponentially weighted moving average model"
    aliases: EWMA VaR, RiskMetrics VaR, exponentially weighted VaR, EWMA expected shortfall
    example:
        import qf
        r = [-0.021, 0.004, -0.013, 0.008, -0.030, 0.011, -0.002, 0.006, -0.017, 0.012]
        qf.ewma_var_es(r, lam=0.94, seed="first", alpha=0.99, sign="loss_positive")
    """
    choice("sign", sign, SIGNS)
    r = _series("returns", returns)
    variance = float(ewma(r * r, lam=lam, seed=seed, nan="raise")[-1])
    return normal_var_es(mu=0.0, sigma=math.sqrt(variance), alpha=alpha, sign=sign)


@_guard(level=('alpha',))
def kde_var_es(returns, *, bandwidth, alpha, sign):
    """Gaussian-kernel density VaR and ES of a return sample, the bandwidth rule named.

    The density is (1/n) sum_i phi((x - r_i)/h)/h. The VaR quantile q solves (1/n) sum_i Phi((q - r_i)/h) = 1 -
    alpha; the ES tail mean is exact under the kernel mixture, (1/(n (1 - alpha))) sum_i [r_i Phi(d_i) - h phi(d_i)],
    d_i = (q - r_i)/h. 'scott' h = n^(-1/5) s and 'silverman' h = (3n/4)^(-1/5) s, s the sample standard deviation
    (ddof 1): scipy.stats.gaussian_kde's factors, not Silverman's 0.9 min(s, IQR/1.34) rule.
    conventions:
        bandwidth: 'scott', 'silverman' (scipy's) or a number h, the kernel's standard deviation in return units
        alpha: the confidence level, e.g. 0.99
        sign: 'loss_positive' or 'return_negative'; returns are gains: losses go in as -losses
    returns: (var, es)
    source: Silverman, Density Estimation for Statistics and Data Analysis (1986); Scott, Multivariate Density
        Estimation (1992); scipy.stats.gaussian_kde
    aliases: KDE VaR, kernel density VaR, kernel density estimation, Gaussian KDE, gaussian_kde
    example:
        import qf
        r = [-0.021, 0.004, -0.013, 0.008, -0.030, 0.011, -0.002, 0.006, -0.017, 0.012]
        qf.kde_var_es(r, bandwidth="scott", alpha=0.95, sign="loss_positive")
    """
    choice("sign", sign, SIGNS)
    r = _series("returns", returns)
    if isinstance(bandwidth, str):
        choice("bandwidth", bandwidth, BANDWIDTHS)
        n = r.size
        factor = n ** -0.2 if bandwidth == "scott" else (0.75 * n) ** -0.2
        h = factor * float(np.std(r, ddof=1))
        if np.ptp(r) == 0 or not h > 0:          # a constant sample's std is rounding (1e-18), not 0
            raise ValueError("returns are constant: the bandwidth rule gives 0")
    else:
        h = float(positive("bandwidth", bandwidth))
    tail = _tail(alpha)
    q = optimize.brentq(lambda x: float(np.mean(ndtr((x - r) / h))) - tail, r.min() - 12.0 * h, r.max() + 12.0 * h,
                        xtol=1e-15 * max(1.0, float(np.max(np.abs(r)))), rtol=4 * np.finfo(float).eps, maxiter=500)
    d = (q - r) / h
    return _signed(sign, q, float(np.mean(r * ndtr(d) - h * stats.norm.pdf(d))) / tail)


# ------------------------------------------------------------------------------------------------ mixtures and spectra


@_guard(level=('alpha',))
def mixture_normal_var_es(*, weights, means, stds, alpha, sign):
    """VaR and ES of a normal mixture of returns (a mixture of Gaussians), the quantile solved, the tail exact.

    The returns' law is sum_k w_k N(mu_k, sigma_k^2). The VaR quantile q solves sum_k w_k Phi((q - mu_k) / sigma_k) =
    1 - alpha (a root, bracketed by the components' own quantiles); the ES tail mean is exact,
    (1 / (1 - alpha)) sum_k w_k [mu_k Phi(z_k) - sigma_k phi(z_k)], z_k = (q - mu_k) / sigma_k. One component is
    qf.normal_var_es. Not the mixture's mean and std put into the normal formula: that is a different (normal) law.
    conventions:
        weights: the components' probabilities w_k, not negative, summing to 1
        means: mu_k, each component's mean return per period
        stds: sigma_k, each component's standard deviation per period (not its variance), above 0
        alpha: the confidence level, e.g. 0.99 (the tail is 1 - alpha)
        sign: 'loss_positive' (VaR and ES of a loss as positive numbers) or 'return_negative'
    returns: (var, es)
    source: McNeil, Frey and Embrechts, Quantitative Risk Management (2005), 3.2 (normal mixtures) and 2.2.4 (ES);
        Alexander, Market Risk Analysis IV (2008), IV.2.7 (normal mixture VaR)
    aliases: normal mixture VaR, Gaussian mixture VaR, mixture of normals, normal mixture ES
    example:
        import qf
        # VaR 0.05134571 and ES 0.06388140.
        qf.mixture_normal_var_es(weights=[0.8, 0.2], means=[0.001, -0.002], stds=[0.01, 0.03], alpha=0.99, sign="loss_positive")
    """
    choice("sign", sign, SIGNS)
    w = _finite("weights", weights).ravel()
    mu = np.broadcast_to(_finite("means", means), w.shape).astype(float)
    sd = np.broadcast_to(positive("stds", stds), w.shape).astype(float)
    if np.any(w < 0) or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("weights must be probabilities summing to 1")
    tail = _tail(alpha)
    own = mu + sd * ndtri(tail)                     # each component's quantile: the root lies between them
    low, high = float(own.min()), float(own.max())
    if high - low <= 1e-15 * max(1.0, abs(high)):
        q = high
    else:
        q = optimize.brentq(lambda x: float(w @ ndtr((x - mu) / sd)) - tail, low, high, xtol=1e-16,
                            rtol=4 * np.finfo(float).eps, maxiter=500)
    z = (q - mu) / sd
    return _signed(sign, q, float(w @ (mu * ndtr(z) - sd * stats.norm.pdf(z))) / tail)


SPECTRA = ("exponential", "expected_shortfall")
DISCRETISATIONS = ("exact", "midpoint")


def _spectrum(weight, risk_aversion, alpha):
    """(phi, an antiderivative of it) of a named weight function, or the callable and None."""
    if callable(weight):
        if risk_aversion is not None or alpha is not None:
            raise ValueError("risk_aversion and alpha name a built-in weight: pass None with a callable weight")
        return weight, None
    choice("weight", weight, SPECTRA)
    if weight == "exponential":
        if alpha is not None:
            raise ValueError("alpha is for 'expected_shortfall': pass None")
        k = float(positive("risk_aversion", risk_aversion))
        norm = -math.expm1(-k)
        return (lambda u: k * np.exp(-k * (1.0 - u)) / norm,
                lambda u: np.exp(-k * (1.0 - u)) * -np.expm1(-k * u) / norm)
    if risk_aversion is not None:
        raise ValueError("risk_aversion is for 'exponential': pass None")
    level = 1.0 - _tail(alpha)
    return (lambda u: np.where(u > level, 1.0 / (1.0 - level), 0.0),
            lambda u: np.maximum(u - level, 0.0) / (1.0 - level))


def spectral_risk(returns, *, weight, risk_aversion, alpha, discretisation, sign):
    """Spectral risk measure (Acerbi): the loss quantiles averaged under an exponential or ES aversion spectrum.

    M = int_0^1 phi(u) VaR_u du, VaR_u the losses' u-quantile and phi a weight that is not negative, rises in u and
    integrates to 1. 'exponential' phi(u) = k e^(-k (1 - u)) / (1 - e^(-k)), k the risk aversion (the larger, the
    more the worst losses weigh); 'expected_shortfall' phi = 1 / (1 - alpha) on (alpha, 1), which gives the ES; or a
    function phi(u) of numpy arrays. On the sorted losses L_(1) <= ... <= L_(n): 'exact' integrates phi over each
    order statistic's span, w_i = int_{(i-1)/n}^{i/n} phi, so M = sum_i w_i L_(i); 'midpoint' w_i = phi((i - 1/2)/n),
    normalised to sum to 1. Coherent, and comonotonic additive.
    conventions:
        returns: the returns or P&L, gains positive (the losses are their negatives)
        weight: 'exponential', 'expected_shortfall' or a function phi(u) on (0, 1)
        risk_aversion: k for 'exponential', above 0; None otherwise
        alpha: the confidence level for 'expected_shortfall', e.g. 0.975; None otherwise
        discretisation: 'exact' or 'midpoint'
        sign: 'loss_positive' (the measure of a loss as a positive number) or 'return_negative'
    returns: a float, the spectral risk measure
    source: Acerbi (2002), "Spectral measures of risk: a coherent representation of subjective risk aversion",
        Journal of Banking and Finance 26(7); Dowd, Cotter and Sorwar (2008), "Spectral risk measures: properties and
        limitations", Journal of Financial Services Research 34
    aliases: spectral risk measure, exponential spectral measure, Acerbi spectral measure, SRM
    example:
        import qf
        r = [-0.021, 0.004, -0.013, 0.008, -0.030, 0.011, -0.002, 0.006, -0.017, 0.012]
        # Spectral risk 0.02570050.
        qf.spectral_risk(returns=r, weight="exponential", risk_aversion=10.0, alpha=None, discretisation="exact", sign="loss_positive")
    """
    choice("discretisation", discretisation, DISCRETISATIONS)
    choice("sign", sign, SIGNS)
    losses = np.sort(-_series("returns", returns))
    n = losses.size
    phi, cumulative = _spectrum(weight, risk_aversion, alpha)
    edges = np.arange(n + 1) / n
    if discretisation == "midpoint":
        w = np.asarray(phi((np.arange(n) + 0.5) / n), dtype=float)
        if np.any(~np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
            raise ValueError("the weight function must not be negative and must not vanish")
        w = w / w.sum()
    elif cumulative is not None:
        w = np.diff(cumulative(edges))
    else:
        w = np.array([integrate.quad(phi, a, b, limit=200)[0] for a, b in zip(edges[:-1], edges[1:])])
        if np.any(w < -1e-12) or abs(w.sum() - 1.0) > 1e-6:
            raise ValueError("the weight function must not be negative and must integrate to 1 over (0, 1)")
    if np.any(~np.isfinite(w)) or np.any(np.diff(w) < -1e-12):
        raise ValueError("the spectrum's weights must be finite and must not decrease toward worse losses")
    measure = float(w @ losses)
    return measure if sign == "loss_positive" else -measure


# ------------------------------------------------------------------------------------------------ mapping and copulas


COMPOUNDINGS = ("annual", "continuous")


def _mapping_share(sl, sr, rho, st, linear):
    """The share alpha on the earlier vertex keeping the variance: the root in [0, 1] of the mapping's quadratic."""
    a = sl * sl + sr * sr - 2.0 * rho * sl * sr
    b = 2.0 * rho * sl * sr - 2.0 * sr * sr
    c = sr * sr - st * st
    if abs(a) < 1e-300 or (b * b - 4.0 * a * c) < -1e-24 * max(b * b, 1e-300):
        return linear
    root = math.sqrt(max(b * b - 4.0 * a * c, 0.0))
    inside = [x for x in ((-b + root) / (2.0 * a), (-b - root) / (2.0 * a)) if -1e-12 <= x <= 1.0 + 1e-12]
    if not inside:
        raise ValueError("no share in [0, 1] keeps the cash flow's variance: check vols and corr")
    return min(max(min(inside, key=lambda x: abs(x - linear)), 0.0), 1.0)


@_guard(rate=('yields',), time=('times', 'vertices',), level=('alpha',), curve=(('vertices', 'yields'),))
def cash_flow_map(*, times, amounts, vertices, yields, vols, corr, compounding, alpha, sign):
    """Cash-flow mapping onto RiskMetrics vertices: each flow's PV split to two maturities keeping its vol; the VaR.

    A flow at t between vertices t_L < t < t_R takes the yield and the price volatility interpolated linearly in time,
    y_t and sigma_t; its present value at y_t is split, alpha to t_L and 1 - alpha to t_R, alpha the root in [0, 1] of
    sigma_t^2 = alpha^2 sigma_L^2 + (1 - alpha)^2 sigma_R^2 + 2 rho alpha (1 - alpha) sigma_L sigma_R. A flow on a
    vertex, or before the first or after the last, goes whole to the nearest (at its yield). The mapped positions x
    give the P&L's std sqrt(x' C x), C_ij = corr_ij vol_i vol_j, and the normal VaR and ES at zero mean.
    conventions:
        times, amounts: each cash flow's time in years and its amount
        vertices: the standard maturities in years, increasing; yields: their zero rates (decimals): annual rates,
            -ln(D)/t, not rates times their times
        vols: each vertex zero bond's price-return standard deviation per period (0.001 for 0.1% a day; RiskMetrics'
            published 1.65 sigma divided by 1.65), the VaR's horizon
        corr: the vertices' correlation matrix
        compounding: 'annual' (PV = amount / (1 + y)^t) or 'continuous' (amount e^(-y t))
        alpha: the VaR's confidence level, e.g. 0.99; sign: 'loss_positive' or 'return_negative'
    returns: a dict: mapped (the present value on each vertex); present_values and shares (each flow's PV and its
        alpha on the earlier vertex); std; var; es
    source: J.P. Morgan/Reuters, RiskMetrics Technical Document (1996), 6.2; Hull, Options, Futures, and Other
        Derivatives, "Cash-flow mapping" (Technical Note 25)
    aliases: cash-flow mapping, RiskMetrics cash-flow map, RiskMetrics vertices, vertex mapping
    example:
        import qf
        # Mapped [37.39662, 331.38145, 678.07349]; VaR 3.77164 and ES 4.32103.
        qf.cash_flow_map(times=[0.3, 0.8], amounts=[50.0, 1050.0], vertices=[0.25, 0.5, 1.0], yields=[0.055, 0.06, 0.07], vols=[0.0006, 0.001, 0.002], corr=[[1.0, 0.9, 0.6], [0.9, 1.0, 0.7], [0.6, 0.7, 1.0]], compounding="annual", alpha=0.99, sign="loss_positive")
    """
    choice("compounding", compounding, COMPOUNDINGS)
    t = _finite("times", times).ravel()
    cf = np.broadcast_to(_finite("amounts", amounts), t.shape).astype(float)
    v = _finite("vertices", vertices).ravel()
    y = _finite("yields", yields).ravel()
    s = positive("vols", vols).ravel()
    c = _finite("corr", corr)
    k = v.size
    if np.any(np.diff(v) <= 0) or y.shape != v.shape or s.shape != v.shape or np.any(t < 0):
        raise ValueError("vertices must increase, with one yield and one vol each; times must not be negative")
    if c.shape != (k, k) or not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0) or np.any(np.abs(c) > 1):
        raise ValueError("corr must be the vertices' symmetric correlation matrix with a unit diagonal")
    mapped, pv, share = np.zeros(k), np.empty(t.size), np.ones(t.size)
    for i, (when, amount) in enumerate(zip(t, cf)):
        right = int(np.searchsorted(v, when))
        on = right < k and abs(v[right] - when) <= 1e-12 * max(1.0, when)
        whole = right if on else 0 if right == 0 else k - 1 if right == k else None
        if whole is None:                                   # strictly between v[right - 1] and v[right]
            left, frac = right - 1, (when - v[right - 1]) / (v[right] - v[right - 1])
            rate = y[left] + frac * (y[right] - y[left])
            vol = s[left] + frac * (s[right] - s[left])
            share[i] = _mapping_share(s[left], s[right], c[left, right], vol, 1.0 - frac)
        else:
            rate = y[whole]
        pv[i] = amount / (1.0 + rate) ** when if compounding == "annual" else amount * math.exp(-rate * when)
        if whole is None:
            mapped[left] += share[i] * pv[i]
            mapped[right] += (1.0 - share[i]) * pv[i]
        else:
            mapped[whole] += pv[i]
    std = float(np.sqrt(max(mapped @ (c * np.outer(s, s)) @ mapped, 0.0)))
    var, es = normal_var_es(mu=0.0, sigma=std, alpha=alpha, sign=sign) if std > 0 else (0.0, 0.0)
    return Result({"mapped": mapped, "present_values": pv, "shares": share, "std": std, "var": var, "es": es})


MARGIN_LAWS = ("normal", "t")
COPULAS = ("gaussian", "t")


def _margin(spec, sigma_kind):
    """A marginal law's inverse CDF: a frozen scipy law, or ('normal', mean, std), or ('t', mean, dispersion, nu)."""
    if hasattr(spec, "ppf"):
        return spec.ppf
    spec = tuple(spec)
    choice("a marginal's law", spec[0], MARGIN_LAWS)
    if spec[0] == "normal":
        mean, std = float(spec[1]), float(positive("a normal marginal's std", spec[2]))
        return lambda u: mean + std * ndtri(u)
    if sigma_kind is None:
        raise ValueError("a ('t', mean, dispersion, nu) marginal needs sigma_kind 'scale' or 'std'")
    mean, dispersion, nu = float(spec[1]), float(positive("a t marginal's dispersion", spec[2])), float(spec[3])
    if not nu > (2.0 if sigma_kind == "std" else 0.0):
        raise ValueError("a t marginal's nu must exceed 2 for sigma_kind 'std' (0 for 'scale')")
    scale = dispersion if sigma_kind == "scale" else dispersion * math.sqrt((nu - 2.0) / nu)
    return lambda u: mean + scale * stdtrit(nu, u)


@_guard(level=('alpha',))
def copula_loss_simulation(*, copula, corr, nu, marginals, sigma_kind, weights, n, rng, alpha, sign):
    """Copula VaR and ES, simulated: Gaussian or t copula draws mapped through each marginal law and summed.

    Draw U from the copula (qf.gaussian_copula_sample, or qf.t_copula_sample with nu), map each column through its
    asset's inverse CDF, X_ij = F_j^-1(U_ij), and sum the positions, P&L_i = sum_j w_j X_ij; then the VaR and ES of
    the simulated P&L by qf.historical_var_es ('inverted_cdf', 'exact'; another rule: pass portfolio_returns there).
    A Gaussian copula with normal margins is the multivariate normal: the delta-normal VaR within simulation error.
    conventions:
        copula: 'gaussian' or 't'; corr: the copula's correlation matrix; nu: the t copula's degrees of freedom, or
            None for 'gaussian'
        marginals: one per asset: a frozen scipy law (scipy.stats.t(df, loc, scale), anything with .ppf), or
            ('normal', mean, std), or ('t', mean, dispersion, nu)
        sigma_kind: a ('t', ...) marginal's dispersion: 'scale' (scipy's scale, as scipy.stats.t.fit gives it) or
            'std' (its standard deviation, scale = std sqrt((nu - 2)/nu)); None where no such tuple is given
        weights: each asset's position (portfolio weights for a portfolio return, or currency amounts for P&L)
        n: the number of draws; rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream)
        alpha: the confidence level, e.g. 0.99; sign: 'loss_positive' or 'return_negative'
    returns: a dict: var; es; portfolio_returns (n); asset_returns (n x assets)
    source: McNeil, Frey and Embrechts, Quantitative Risk Management (2005), 5.1 and 5.5 (copula models and their
        simulation); Cherubini, Luciano and Vecchiato, Copula Methods in Finance (2004), ch. 8
    aliases: copula VaR, t copula VaR, Gaussian copula VaR, copula ES
    example:
        import numpy as np, qf
        # VaR 0.02995700 and ES 0.03976239 for these draws.
        qf.copula_loss_simulation(copula="t", corr=[[1.0, 0.5], [0.5, 1.0]], nu=5.0, marginals=[("normal", 0.0, 0.01), ("t", 0.0, 0.02, 4.0)], sigma_kind="std", weights=[0.6, 0.4], n=20000, rng=np.random.default_rng(1), alpha=0.99, sign="loss_positive")
    """
    choice("copula", copula, COPULAS)
    if sigma_kind is not None:
        choice("sigma_kind", sigma_kind, SIGMA_KINDS)
    lower = _correlation(corr)
    d = lower.shape[0]
    laws = list(marginals)
    w = np.broadcast_to(_finite("weights", weights), (d,)).astype(float)
    if len(laws) != d:
        raise ValueError(f"marginals must hold one law per asset ({d})")
    inverse = [_margin(spec, sigma_kind) for spec in laws]
    n, rng = _draws(n, rng)
    if copula == "gaussian":
        if nu is not None:
            raise ValueError("nu is for the t copula: pass None with 'gaussian'")
        u = gaussian_copula_sample(corr, n=n, rng=rng)
    else:
        u = t_copula_sample(corr, nu=nu, n=n, rng=rng)
    u = np.clip(u, 1e-16, 1.0 - 1e-16)             # a t copula's tail can round to 0 or 1
    x = np.column_stack([np.asarray(f(u[:, j]), dtype=float) for j, f in enumerate(inverse)])
    pnl = x @ w
    var, es = historical_var_es(pnl, alpha=alpha, method="inverted_cdf", es_rule="exact", sign=sign)
    return Result({"var": float(var), "es": float(es), "portfolio_returns": pnl, "asset_returns": x})
