"""Volatility from prices: close-to-close, high-low ranges, EWMA, GARCH(1,1), GJR and EGARCH, DCC dynamic correlation.

Returns are log returns, oldest first, unless a function says otherwise. A volatility is annualised by the required
`periods_per_year` (1 keeps it per period); a variance or a parameter stays per period. The shelf `realized` has
the realized variance and the intraday measures, `surfaces` SVI and the local volatilities, `ou` the AR(1) and OU
fits.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, signal
from scipy.special import gammaln

from ._common import Result, choice, positive
from .conventions import _finite, ewma
from .primitives import _guard

__all__ = ["close_to_close_vol", "dcc_fit", "ewma_volatility", "garch11_fit", "garch11_forecast", "garch11_variance",
           "garman_klass_vol", "gjr_garch_fit", "parkinson_vol", "rogers_satchell_vol", "yang_zhang_vol"]

MEANS = ("zero", "constant")
DISTS = ("normal", "t")
STARTS = ("sigma2_0", "backcast")
LN2 = math.log(2.0)
ASYMMETRIC = ("gjr", "egarch")
ABS_NORMAL = math.sqrt(2.0 / math.pi)     # E|z| for a standard normal z: EGARCH's centring (arch's, whatever the law)
LOG_VARIANCE_BOUND = 700.0       # EGARCH's log variance held inside +-700 while the optimiser roams (exp stays finite)


def _periods(periods_per_year) -> float:
    return float(positive("periods_per_year", periods_per_year))


def _series(name: str, value, least: int = 2) -> np.ndarray:
    series = _finite(name, value)
    if series.ndim != 1 or series.size < least:
        raise ValueError(f"{name} must be a 1-D series of at least {least} values")
    return series


def _int(name: str, value, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value < high:
        raise ValueError(f"{name} must be an int with {low} <= {name} < {high}")
    return int(value)


def _vol(variance: float, periods: float) -> float:
    return math.sqrt(variance * periods) if variance >= 0 else math.nan


# ------------------------------------------------------------------------------------------------ EWMA and GARCH


def ewma_volatility(returns, *, lam, seed, nan, periods_per_year):
    """EWMA (RiskMetrics) vol path sigma^2_t = lam sigma^2_{t-1} + (1 - lam) r_t^2, annualised.

    qf.ewma on the squared returns with the seed and NaN rule named, then sqrt(sigma^2 P). Zero mean:
    squared returns, not squared deviations. Entry t uses returns up to t, so it is the forecast for t + 1: lag it
    one period to size a position (no look-ahead).
    conventions:
        lam: the decay, inside (0, 1) (RiskMetrics 0.94 daily); a span N or alpha = 2/(N + 1) gives lam = 1 -
            2/(N + 1) = 1 - alpha (pandas ewm(alpha=1 - lam))
        seed: the variance before the first return: 'first' r_0^2, 'zero', 'mean' the mean of r^2, 'adjusted'
            (pandas adjust=True) or a number (a sample variance)
        nan: 'raise' or 'skip' (a NaN return leaves the variance as it was)
        periods_per_year: P, e.g. 252 (1 keeps per-period vol)
    returns: an array of vols shaped like returns
    source: J.P. Morgan/Reuters, RiskMetrics Technical Document (1996), 5.2.1; Hull, Options, Futures, and Other
        Derivatives, "The exponentially weighted moving average model"
    aliases: EWMA volatility, RiskMetrics volatility, exponentially weighted volatility, EWMA vol
    example:
        import qf
        r = [0.010, -0.020, 0.015, 0.003, -0.007]
        qf.ewma_volatility(r, lam=0.94, seed="first", nan="raise", periods_per_year=252)
    """
    periods = _periods(periods_per_year)
    r = np.asarray(returns, dtype=float)
    return np.sqrt(ewma(r * r, lam=lam, seed=seed, nan=nan) * periods)


def garch11_variance(returns, *, mu, omega, alpha, beta, start_variance, start):
    """The GARCH(1,1) conditional variance path for given parameters, its start rule named.

    e_t = r_t - mu and sigma^2_t = omega + alpha e^2_{t-1} + beta sigma^2_{t-1}. 'sigma2_0' sets sigma^2_0 =
    start_variance; 'backcast' sets e^2_{-1} = sigma^2_{-1} = start_variance, so sigma^2_0 = omega + (alpha + beta)
    start_variance (the arch package's rule; qf.garch11_fit returns the backcast it used). The next
    period's variance omega + alpha e^2_{n-1} + beta sigma^2_{n-1} is returned beside the path.
    conventions:
        mu: the mean return per period (0: zero mean)
        omega: above 0, per period, in the returns' units squared
        alpha: the ARCH weight, at least 0
        beta: the GARCH weight, at least 0
        start_variance: a variance per period
        start: 'sigma2_0' or 'backcast'
    returns: (variances, next_variance): per-period variances (not vols), one per return, then the next one
    source: Bollerslev (1986), "Generalized autoregressive conditional heteroskedasticity", Journal of
        Econometrics 31(3); Hull, Options, Futures, and Other Derivatives, "The GARCH(1,1) model"
    aliases: GARCH, GARCH filter, [conditional variance], GARCH recursion
    example:
        import qf
        r = [0.010, -0.020, 0.015, 0.003, -0.007]
        qf.garch11_variance(r, mu=0.0, omega=2e-6, alpha=0.1, beta=0.88, start_variance=1e-4, start="sigma2_0")
    """
    choice("start", start, STARTS)
    r = _series("returns", returns, least=1)
    omega = float(positive("omega", omega))
    alpha, beta, v0 = float(alpha), float(beta), float(positive("start_variance", start_variance))
    if not (alpha >= 0 and beta >= 0):
        raise ValueError("alpha and beta must be at least 0")
    e2 = (r - float(mu)) ** 2
    variance = np.empty(r.size)
    variance[0] = v0 if start == "sigma2_0" else omega + (alpha + beta) * v0
    for t in range(1, r.size):
        variance[t] = omega + alpha * e2[t - 1] + beta * variance[t - 1]
    return variance, float(omega + alpha * e2[-1] + beta * variance[-1])


def garch11_forecast(next_variance, *, omega, alpha, beta, horizon):
    """GARCH(1,1) variance forecasts 1..H periods ahead: v_{h+1} = omega + (alpha + beta) v_h.

    v_1 is next_variance (the one-step forecast); with persistence alpha + beta < 1 this is Hull's E[sigma^2_{n+h}] =
    V_L + (alpha + beta)^{h-1} (v_1 - V_L), V_L = omega / (1 - alpha - beta). An h-period VaR's variance is the sum
    of the first h (not h v_1).
    conventions:
        omega: above 0, per period, in next_variance's units
        alpha: the ARCH weight, at least 0
        beta: the GARCH weight, at least 0
        horizon: H, the number of periods ahead, a positive int
    returns: an array of H variances (not vols) per period
    source: Hull, Options, Futures, and Other Derivatives, "Using GARCH(1,1) to forecast future volatility";
        Bollerslev (1986), Journal of Econometrics 31(3)
    aliases: GARCH forecast, [variance forecast], [volatility term structure], multi-step GARCH
    example:
        import qf
        v1 = 1.8e-4
        qf.garch11_forecast(v1, omega=2e-6, alpha=0.1, beta=0.88, horizon=10)
    """
    steps = _int("horizon", horizon, 1, 10 ** 7)
    v = float(positive("next_variance", next_variance))
    omega, persistence = float(positive("omega", omega)), float(alpha) + float(beta)
    if not (float(alpha) >= 0 and float(beta) >= 0):
        raise ValueError("alpha and beta must be at least 0")
    path = np.empty(steps)
    for h in range(steps):
        path[h] = v
        v = omega + persistence * v
    return path


def garch11_fit(returns, *, mean, dist, scale, start_variance, periods_per_year):
    """Fit GARCH(1,1) by maximum likelihood through the arch package, every option explicit.

    arch_model(returns * scale, mean, vol='GARCH', p=1, q=1, dist, rescale=False).fit(backcast=...), the parameters
    converted back to the returns' units (mu / scale, omega / scale^2) and the log-likelihood to the returns' own
    density (arch's + n ln scale). start_variance sets the backcast, the value of e^2 and sigma^2 before the
    first return: 'arch' arch's own (sum_i 0.94^i e_i^2 / sum_i 0.94^i over the first min(75, n) residuals at the
    starting mean), 'sample' the mean squared residual, or a number. The conditional vol is per period and
    annualised by sqrt(P); next_vol is the one-step forecast after the last return. Near scale 1 on decimal returns
    arch's optimiser can stop at its start values (round numbers, arch_model(...).volatility.starting_values(r))
    and still report convergence.
    conventions:
        mean: 'zero' or 'constant'
        dist: 'normal' or 't' (the standardised Student t; nu is fitted)
        scale: 100 for daily decimal returns (0.01 = 1%), 1 for returns in percent; results come back unscaled
        start_variance: 'arch', 'sample' or a variance per period
        periods_per_year: P for the annualised vols, e.g. 252
    returns: a dict, every value unscaled (the returns' own units): mu, omega (a variance per period), alpha, beta,
        nu (nan for 'normal'), persistence (alpha + beta), long_run_variance (per period), loglik, converged (False,
        or alpha and beta left at start values: no fit, refit at scale 100), backcast (a variance per period),
        conditional_vol (per-period sds, not variances), conditional_vol_annual (times sqrt(P)), next_vol (the
        one-step sd), next_vol_annual (next_vol sqrt(P), no more sqrt)
    source: Bollerslev (1986), Journal of Econometrics 31(3); Sheppard, the arch package (arch.univariate.arch_model)
    aliases: GARCH, GARCH fit, arch_model, fit GARCH, [conditional volatility], GARCH volatility
    example:
        import numpy as np, qf
        r = np.random.default_rng(5).standard_normal(1000) * 0.01
        qf.garch11_fit(r, mean="zero", dist="normal", scale=100.0, start_variance="arch", periods_per_year=252)
    """
    choice("mean", mean, MEANS)
    choice("dist", dist, DISTS)
    periods = _periods(periods_per_year)
    factor = float(positive("scale", scale))
    r = _series("returns", returns, least=10)
    from arch import arch_model

    y = r * factor
    residual = y if mean == "zero" else y - y.mean()
    if isinstance(start_variance, str):
        choice("start_variance", start_variance, ("arch", "sample"))
        if start_variance == "arch":
            weights = 0.94 ** np.arange(min(75, y.size))
            backcast = float(np.sum(weights * residual[:weights.size] ** 2) / weights.sum())
        else:
            backcast = float(np.mean(residual ** 2))
    else:
        backcast = float(positive("start_variance", start_variance)) * factor ** 2
    model = arch_model(y, mean="Zero" if mean == "zero" else "Constant", vol="GARCH", p=1, q=1,
                       dist="normal" if dist == "normal" else "studentst", rescale=False)
    fitted = model.fit(disp="off", backcast=backcast, show_warning=False)
    params = fitted.params
    mu = float(params.get("mu", 0.0)) / factor
    omega = float(params["omega"]) / factor ** 2
    alpha, beta = float(params["alpha[1]"]), float(params["beta[1]"])
    conditional = np.asarray(fitted.conditional_volatility, dtype=float) / factor
    next_variance = omega + alpha * (r[-1] - mu) ** 2 + beta * conditional[-1] ** 2
    persistence = alpha + beta
    return Result({"mu": mu, "omega": omega, "alpha": alpha, "beta": beta, "nu": float(params.get("nu", math.nan)),
                   "persistence": persistence,
                   "long_run_variance": omega / (1.0 - persistence) if persistence < 1 else math.inf,
                   "loglik": float(fitted.loglikelihood) + r.size * math.log(factor),
                   "converged": int(fitted.convergence_flag) == 0, "backcast": backcast / factor ** 2,
                   "conditional_vol": conditional, "conditional_vol_annual": conditional * math.sqrt(periods),
                   "next_vol": math.sqrt(next_variance), "next_vol_annual": math.sqrt(next_variance * periods)})


# ------------------------------------------------------------------------------------------------ range estimators


DCC_LIKELIHOODS = ("engle", "quadratic", "gaussian")


def _dcc_path(z, a, b, qbar):
    """The DCC(1,1) correlations R_t and the terms ln|R_t| and z_t' R_t^-1 z_t, Q_1 = Qbar."""
    t, n = z.shape
    q = qbar.copy()
    corr, logdet, quad = np.empty((t, n, n)), np.empty(t), np.empty(t)
    for k in range(t):
        if k:
            q = (1.0 - a - b) * qbar + a * np.outer(z[k - 1], z[k - 1]) + b * q
        scale = 1.0 / np.sqrt(np.diag(q))
        r = q * np.outer(scale, scale)
        corr[k] = r
        sign, logdet[k] = np.linalg.slogdet(r)
        if sign <= 0:
            return corr, np.full(t, np.inf), quad
        quad[k] = z[k] @ np.linalg.solve(r, z[k])
    return corr, logdet, quad


def dcc_fit(residuals, *, start, likelihood):
    """DCC(1,1) dynamic conditional correlation by quasi-maximum likelihood on GARCH-standardised residuals (Engle).

    Second step of Engle's two: z_t the univariate GARCH(1,1) standardised residuals (qf.garch11_fit's, one column an
    asset), Qbar = z'z / T, Q_1 = Qbar, Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}, R_t = diag(Q_t)^-1/2
    Q_t diag(Q_t)^-1/2; a and b maximise the correlation likelihood by L-BFGS-B from `start` within a, b >= 0, a + b
    < 1. The likelihood reported: 'engle' -1/2 sum [ln|R_t| + z_t' R_t^-1 z_t - z_t' z_t] (Engle's correlation part),
    'quadratic' without the - z_t' z_t, 'gaussian' -1/2 sum [N ln(2 pi) + ln|R_t| + z_t' R_t^-1 z_t]; a and b are the
    same under each. a = b = 0 is the constant correlation (CCC) R_t = Qbar scaled. The conditional covariance is H_t
    = D_t R_t D_t, D_t the diagonal of the GARCH vols, and a portfolio's variance w' H_t w.
    conventions:
        residuals: z, a T x N array of standardised residuals, oldest first (N assets)
        start: (a0, b0), the optimiser's first point
        likelihood: 'engle', 'quadratic' or 'gaussian'
    returns: a dict: a; b; persistence (a + b); log_likelihood; correlations (T x N x N, R_t for each t, from the
        information to t - 1); qbar; converged
    source: Engle (2002), "Dynamic conditional correlation", J. Business and Economic Statistics 20(3); Engle and
        Sheppard (2001), NBER working paper 8554
    aliases: DCC, DCC-GARCH, DCC GARCH, dynamic conditional correlation, Engle DCC, multivariate GARCH, conditional correlation, time-varying correlation
    example:
        import numpy as np, qf
        z = np.random.default_rng(2).standard_normal((300, 3))
        qf.dcc_fit(residuals=z, start=(0.02, 0.95), likelihood="engle")
    """
    choice("likelihood", likelihood, DCC_LIKELIHOODS)
    z = np.asarray(residuals, dtype=float)
    if z.ndim != 2 or z.shape[1] < 2 or z.shape[0] < 3 or not np.all(np.isfinite(z)):
        raise ValueError("residuals must be a T x N array of finite standardised residuals, N >= 2")
    a0, b0 = (float(x) for x in start)
    if not (a0 >= 0 and b0 >= 0 and a0 + b0 < 1):
        raise ValueError("start must be (a0, b0) with a0, b0 >= 0 and a0 + b0 < 1")
    t, n = z.shape
    qbar = z.T @ z / t
    squares = float(np.sum(z * z))

    def loss(theta):
        a, b = theta
        if a + b >= 1.0 - 1e-8:
            return 1e10 * (1.0 + a + b)
        _, logdet, quad = _dcc_path(z, a, b, qbar)
        return 0.5 * float(np.sum(logdet + quad))

    found = optimize.minimize(loss, np.array([a0, b0]), method="L-BFGS-B", bounds=[(0.0, 1.0), (0.0, 1.0)])
    a, b = (float(x) for x in found.x)
    corr, logdet, quad = _dcc_path(z, a, b, qbar)
    core = -0.5 * float(np.sum(logdet + quad))
    value = {"engle": core + 0.5 * squares, "quadratic": core, "gaussian": core - 0.5 * t * n * math.log(2 * math.pi)}
    return Result({"a": a, "b": b, "persistence": a + b, "log_likelihood": value[likelihood], "correlations": corr,
                   "qbar": qbar, "converged": bool(found.success)})


def _bars(open, high, low, close):
    o, h, l, c = (_series(name, value) for name, value in
                  (("open", open), ("high", high), ("low", low), ("close", close)))
    if not o.size == h.size == l.size == c.size:
        raise ValueError("open, high, low and close must have equal lengths")
    if np.any(o <= 0) or np.any(l <= 0):
        raise ValueError("prices must be positive")
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)) or np.any(h < l):
        raise ValueError("each bar needs low <= open, close <= high")
    return o, h, l, c


def _rs_points(o, h, l, c):
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


@_guard(price=('close',))
def close_to_close_vol(close, *, ddof, periods_per_year):
    """Annualised (historical) volatility of prices: std of log returns, ddof named, times sqrt(P).

    sqrt(P sum (r - rbar)^2 / (n - ddof)), r_t = ln(C_t / C_{t-1}): the benchmark the range estimators' efficiencies
    are quoted against.
    conventions:
        ddof: 0 or 1 (or any int below the number of returns)
        periods_per_year: P, e.g. 252 (1 keeps per-period vol)
    returns: a float
    source: Hull, Options, Futures, and Other Derivatives, "Estimating volatility from historical data"; Yang and
        Zhang (2000), Journal of Business 73(3) (the classical estimator)
    aliases: close-to-close volatility, historical volatility, realized volatility from closes, close to close, [annualized volatility], [annualised volatility]
    example:
        import qf
        closes = [100.0, 101.2, 100.5, 102.0, 101.1, 103.4]
        qf.close_to_close_vol(closes, ddof=1, periods_per_year=252)
    """
    c = _series("close", close, least=3)
    if np.any(c <= 0):
        raise ValueError("close must be positive")
    r = np.diff(np.log(c))
    step = _int("ddof", ddof, 0, r.size)
    return _vol(float(np.var(r, ddof=step)), _periods(periods_per_year))


@_guard(price=('high', 'low',))
def parkinson_vol(high, low, *, periods_per_year):
    """Parkinson's high-low range volatility, sqrt(P mean[(ln H/L)^2] / (4 ln 2)).

    Unbiased for a driftless Brownian motion monitored continuously (E[(ln H/L)^2] = 4 ln 2 sigma^2 per bar);
    Garman and Klass quote its efficiency against close-to-close as about 5.2. A discretely sampled high and low
    bias it down.
    conventions:
        periods_per_year: P, e.g. 252 for daily bars (1 keeps per-bar vol)
    returns: a float
    source: Parkinson (1980), "The extreme value method for estimating the variance of the rate of return",
        Journal of Business 53(1); QuantLib's ParkinsonSigma
    aliases: Parkinson, Parkinson volatility, Parkinson estimator, high-low volatility, range volatility
    example:
        import qf
        high, low = [101.0, 102.5, 101.8], [99.0, 100.1, 99.7]
        qf.parkinson_vol(high, low, periods_per_year=252)
    """
    h, l = _series("high", high), _series("low", low)
    if h.size != l.size or np.any(l <= 0) or np.any(h < l):
        raise ValueError("high and low must be equal-length positive prices with low <= high")
    return _vol(float(np.mean(np.log(h / l) ** 2)) / (4.0 * LN2), _periods(periods_per_year))


@_guard(price=('open', 'high', 'low', 'close',))
def garman_klass_vol(open, high, low, close, *, periods_per_year):
    """Garman-Klass volatility, sqrt(P mean[0.5 (ln H/L)^2 - (2 ln 2 - 1)(ln C/O)^2]).

    Their practical estimator (sigma-hat-5), unbiased for a driftless Brownian motion within the bar and blind to
    overnight jumps; efficiency about 7.4 against close-to-close.
    conventions:
        periods_per_year: P, e.g. 252 for daily bars (1 keeps per-bar vol)
    returns: a float
    source: Garman and Klass (1980), "On the estimation of security price volatilities from historical data",
        Journal of Business 53(1); QuantLib's GarmanKlassSigma5
    aliases: Garman-Klass, Garman Klass volatility, GK volatility, OHLC volatility
    example:
        import qf
        o, h, l, c = [100.0, 100.8, 101.9], [101.0, 102.5, 102.3], [99.0, 100.1, 99.7], [100.6, 101.7, 100.2]
        qf.garman_klass_vol(o, h, l, c, periods_per_year=252)
    """
    o, h, l, c = _bars(open, high, low, close)
    points = 0.5 * np.log(h / l) ** 2 - (2.0 * LN2 - 1.0) * np.log(c / o) ** 2
    return _vol(float(np.mean(points)), _periods(periods_per_year))


@_guard(price=('open', 'high', 'low', 'close',))
def rogers_satchell_vol(open, high, low, close, *, periods_per_year):
    """Rogers-Satchell volatility, sqrt(P mean[ln(H/C) ln(H/O) + ln(L/C) ln(L/O)]), drift-free.

    Unbiased for a Brownian motion with any drift within the bar; blind to overnight jumps.
    conventions:
        periods_per_year: P, e.g. 252 for daily bars (1 keeps per-bar vol)
    returns: a float
    source: Rogers and Satchell (1991), "Estimating variance from high, low and closing prices", Annals of Applied
        Probability 1(4); Rogers, Satchell and Yoon (1994), Applied Financial Economics 4
    aliases: Rogers-Satchell, Rogers Satchell volatility, RS volatility, drift-independent volatility
    example:
        import qf
        o, h, l, c = [100.0, 100.8, 101.9], [101.0, 102.5, 102.3], [99.0, 100.1, 99.7], [100.6, 101.7, 100.2]
        qf.rogers_satchell_vol(o, h, l, c, periods_per_year=252)
    """
    o, h, l, c = _bars(open, high, low, close)
    return _vol(float(np.mean(_rs_points(o, h, l, c))), _periods(periods_per_year))


@_guard(price=('open', 'high', 'low', 'close',))
def yang_zhang_vol(open, high, low, close, *, periods_per_year):
    """Yang-Zhang volatility: overnight, open-to-close and Rogers-Satchell variances with k.

    Over the n = N - 1 bars that have a previous close: sigma^2 = V_o + k V_c + (1 - k) V_rs, V_o the variance (ddof
    1) of the overnight returns ln(O_t / C_{t-1}), V_c that of ln(C_t / O_t), V_rs the mean Rogers-Satchell term, and
    k = 0.34 / (1.34 + (n + 1)/(n - 1)) (their alpha = 1.34), the minimum-variance weight; then sqrt(P sigma^2).
    Unbiased under drift and opening jumps.
    conventions:
        periods_per_year: P, e.g. 252 for daily bars (1 keeps per-bar vol); k = 0.34 / (1.34 + (n + 1)/(n - 1)),
            n the bars with a previous close (N - 1 of the N given), fixed by n and not an argument
    returns: a float
    source: Yang and Zhang (2000), "Drift-independent volatility estimation based on high, low, open, and close
        prices", Journal of Business 73(3)
    aliases: Yang-Zhang, Yang Zhang volatility, YZ estimator, YZ volatility
    example:
        import numpy as np, qf
        bars = np.array([[100.0, 101.0, 99.0, 100.6], [100.8, 102.5, 100.1, 101.7], [101.9, 102.3, 99.7, 100.2]])
        qf.yang_zhang_vol(*bars.T, periods_per_year=252)
    """
    o, h, l, c = _bars(open, high, low, close)
    if o.size < 3:
        raise ValueError("Yang-Zhang needs at least 3 bars (2 with a previous close)")
    n = o.size - 1
    overnight = np.log(o[1:] / c[:-1])
    intraday = np.log(c[1:] / o[1:])
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    variance = (np.var(overnight, ddof=1) + k * np.var(intraday, ddof=1)
                + (1.0 - k) * np.mean(_rs_points(o[1:], h[1:], l[1:], c[1:])))
    return _vol(float(variance), _periods(periods_per_year))


# ------------------------------------------------------------------------------------------------ asymmetric GARCH


def _asymmetric_variance(e, model, omega, alpha, gamma, beta, backcast):
    """The GJR or EGARCH(1,1) conditional variances of the residuals e, started from the backcast as arch starts them:
    GJR's first variance omega + (alpha + gamma / 2 + beta) backcast, EGARCH's first log variance omega + beta ln
    backcast."""
    if model == "gjr":
        e2 = e * e
        shocks = np.r_[(alpha + 0.5 * gamma + beta) * backcast,
                       alpha * e2[:-1] + gamma * e2[:-1] * (e[:-1] < 0)] + omega
        return signal.lfilter([1.0], [1.0, -beta], shocks)
    log_variance = np.empty(e.size)
    log_variance[0] = min(max(omega + beta * math.log(backcast), -LOG_VARIANCE_BOUND), LOG_VARIANCE_BOUND)
    for t in range(1, e.size):
        z = e[t - 1] * math.exp(-0.5 * log_variance[t - 1])
        log_variance[t] = min(max(omega + alpha * (abs(z) - ABS_NORMAL) + gamma * z + beta * log_variance[t - 1],
                                  -LOG_VARIANCE_BOUND), LOG_VARIANCE_BOUND)
    return np.exp(log_variance)


def _asymmetric_loglik(e, variance, nu):
    """The normal (nu None) or standardised Student t log-likelihood of residuals e with these variances."""
    if nu is None:
        return -0.5 * float(np.sum(np.log(2.0 * np.pi * variance) + e * e / variance))
    return float(np.sum(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
                        - 0.5 * np.log(variance) - (nu + 1.0) / 2.0 * np.log1p(e * e / (variance * (nu - 2.0)))))


def gjr_garch_fit(returns, model, dist, mean, start_variance):
    """GJR-GARCH and EGARCH(1,1) estimates: asymmetric GARCH with the leverage effect's gamma.

    GJR (Glosten, Jagannathan and Runkle): sigma^2_t = omega + (alpha + gamma 1{e_{t-1} < 0}) e^2_{t-1} + beta
    sigma^2_{t-1}, e_t = r_t - mu; gamma > 0 when bad news raises the variance more; persistence alpha + gamma/2 + beta
    (< 1), long-run variance omega / (1 - persistence). EGARCH (Nelson): ln sigma^2_t = omega + alpha (|z_{t-1}| -
    sqrt(2/pi)) + gamma z_{t-1} + beta ln sigma^2_{t-1}, z = e / sigma; gamma < 0 for the leverage effect; persistence
    beta. The arch package's GJR (vol='GARCH', p=1, o=1, q=1) and EGARCH (vol='EGARCH', p=1, o=1, q=1) likelihood and
    backcast; fitted here by scipy (SLSQP), on the returns rescaled to unit variance and the results scaled back.
    gamma = 0 in GJR is GARCH(1,1) (qf.garch11_fit).

    conventions:
    - returns: per-period returns (decimals), oldest first
    - model: 'gjr' or 'egarch'
    - dist: 'normal' or 't' (the standardised Student t, nu fitted)
    - mean: 'zero' or 'constant' (mu fitted)
    - start_variance: the backcast, the variance before the first return: 'arch' arch's own (sum_i 0.94^i e_i^2 / sum_i
      0.94^i over the first min(75, n) residuals at the sample mean), 'sample' the mean squared residual, or a variance
    returns: a dict, in the returns' own units: mu; omega; alpha; gamma; beta; nu (nan for 'normal'); persistence;
      long_run_variance (GJR's arithmetic mean variance; nan for EGARCH); geometric_variance (EGARCH's
      exp(mean log variance), nan for GJR); loglik; converged; backcast; conditional_vol (per-period
      standard deviations, one per return); next_vol (the one-step-ahead standard deviation); annualise by sqrt(P)
    source: Glosten, Jagannathan and Runkle (1993), J. Finance 48(5); Nelson (1991), Econometrica 59(2); Tsay, Analysis
      of Financial Time Series (2010), 3.8 and 3.9; Sheppard, the arch package (arch.univariate)
    aliases: GJR-GARCH, GJR GARCH, TGARCH, EGARCH, asymmetric GARCH, leverage effect
    example:
        qf.gjr_garch_fit(returns=[0.004, -0.009, 0.003, 0.011, -0.006, 0.002, -0.013, 0.008,
                                 0.005, -0.004, 0.007, -0.008, 0.001, 0.009, -0.005, 0.006,
                                 -0.012, 0.003, 0.010, -0.007, 0.002, 0.005, -0.009, 0.004],
                         model="gjr", dist="normal", mean="zero", start_variance="arch")
        # next_vol approximately 0.006894
    """
    choice("model", model, ASYMMETRIC)
    choice("dist", dist, DISTS)
    choice("mean", mean, MEANS)
    y = _series("returns", returns, least=20)
    factor = 1.0 / float(np.std(y))
    if not np.isfinite(factor):
        raise ValueError("returns must vary")
    x = y * factor
    centre = x.mean() if mean == "constant" else 0.0
    residual = x - centre
    if isinstance(start_variance, str):
        choice("start_variance", start_variance, ("arch", "sample"))
        if start_variance == "arch":
            weights = 0.94 ** np.arange(min(75, x.size))
            backcast = float(np.sum(weights * residual[:weights.size] ** 2) / weights.sum())
        else:
            backcast = float(np.mean(residual ** 2))
    else:
        backcast = float(positive("start_variance", start_variance)) * factor ** 2
    has_mu, has_nu = mean == "constant", dist == "t"

    def unpack(theta):
        mu = theta[0] if has_mu else 0.0
        omega, alpha, gamma, beta = theta[int(has_mu):int(has_mu) + 4]
        return mu, omega, alpha, gamma, beta, (theta[-1] if has_nu else None)

    def loss(theta):
        mu, omega, alpha, gamma, beta, nu = unpack(theta)
        e = x - mu
        variance = _asymmetric_variance(e, model, omega, alpha, gamma, beta, backcast)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            return 1e10
        return -_asymmetric_loglik(e, variance, nu)

    var = float(np.mean(residual ** 2))
    if model == "gjr":
        grid = [(0.03, 0.05, 0.90), (0.05, 0.10, 0.85), (0.10, 0.05, 0.80), (0.02, 0.02, 0.95), (0.10, 0.20, 0.60)]
        starts = [(var * (1.0 - a - g / 2.0 - b), a, g, b) for a, g, b in grid]
        bounds = [(1e-8 * var, 10.0 * var), (0.0, 1.0), (-1.0, 2.0), (0.0, 1.0)]
        constraints = [{"type": "ineq", "fun": lambda t: 1.0 - 1e-6 - t[int(has_mu) + 1] - 0.5 * t[int(has_mu) + 2]
                        - t[int(has_mu) + 3]},
                       {"type": "ineq", "fun": lambda t: t[int(has_mu) + 1] + t[int(has_mu) + 2]}]
    else:
        grid = [(0.10, -0.05, 0.95), (0.20, -0.10, 0.90), (0.05, 0.0, 0.98), (0.30, -0.15, 0.80)]
        starts = [(math.log(var) * (1.0 - b), a, g, b) for a, g, b in grid]
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-1.0 + 1e-6, 1.0 - 1e-6)]
        constraints = []
    head = [centre] if has_mu else []
    bounds = ([(-10.0 * abs(centre) - 10.0, 10.0 * abs(centre) + 10.0)] if has_mu else []) + bounds + (
        [(2.05, 500.0)] if has_nu else [])
    best = None
    for start in starts:
        theta = np.array(head + list(start) + ([8.0] if has_nu else []))
        found = optimize.minimize(loss, theta, method="SLSQP", bounds=bounds, constraints=constraints,
                                  options={"maxiter": 500, "ftol": 1e-12})
        if best is None or found.fun < best.fun:
            best = found
    mu, omega, alpha, gamma, beta, nu = unpack(best.x)
    e = x - mu
    variance = _asymmetric_variance(e, model, omega, alpha, gamma, beta, backcast)
    if model == "gjr":
        following = omega + (alpha + gamma * (e[-1] < 0)) * e[-1] ** 2 + beta * variance[-1]
        persistence = alpha + 0.5 * gamma + beta
        long_run = omega / (1.0 - persistence) / factor ** 2 if persistence < 1 else math.inf
        geometric = math.nan
        omega_out = omega / factor ** 2
    else:
        z = e[-1] / math.sqrt(variance[-1])
        following = math.exp(omega + alpha * (abs(z) - ABS_NORMAL) + gamma * z + beta * math.log(variance[-1]))
        persistence = beta
        omega_out = omega - 2.0 * (1.0 - beta) * math.log(factor)
        mean_abs = ABS_NORMAL if nu is None else math.sqrt((nu - 2.0) / math.pi) * math.exp(
            gammaln((nu - 1.0) / 2.0) - gammaln(nu / 2.0))
        geometric = math.exp((omega_out + alpha * (mean_abs - ABS_NORMAL)) / (1.0 - beta))
        long_run = math.nan
    return Result({"mu": mu / factor, "omega": omega_out, "alpha": alpha, "gamma": gamma, "beta": beta,
                   "nu": nu if nu is not None else math.nan, "persistence": persistence,
                   "long_run_variance": long_run, "geometric_variance": geometric,
                   "loglik": -best.fun + y.size * math.log(factor),
                   "converged": bool(best.success), "backcast": backcast / factor ** 2,
                   "conditional_vol": np.sqrt(variance) / factor, "next_vol": math.sqrt(following) / factor})
