"""Performance and backtests: Sharpe and its inference, Sortino, Calmar, CAGR, drawdowns, Brinson, Carino, rebalancing.

`import qf`. Returns are simple per-period returns in time order, as decimals (0.01 is 1%); `periods_per_year`
scales a per-period figure to a year (252 trading days, 12 months, 52 weeks) and is always the caller's to name. A
scalar result is a float; a function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._common import Result, choice, positive
from .conventions import SIGNS, _finite
from .primitives import _guard

__all__ = ["backtest_signals", "brinson_fachler", "cagr", "calmar_ratio", "carino_linking", "max_drawdown",
           "max_drawdown_dates", "rebalance_path", "sharpe_inference", "sharpe_ratio", "sortino_ratio",
           "vol_target_leverage"]

RISK_FREE_BASES = ("per_period", "annual_simple", "annual_compound", "annual_continuous")
EXECUTIONS = ("close", "next_open")
ANNUAL_RETURNS = ("cagr", "arithmetic")
TIMINGS = ("start", "end")
FEE_ORDERS = ("before", "after")
TURNOVERS = ("one_way", "two_way")
TURNOVER_BASES = ("drifted", "target")
STARTS = ("cash", "invested")
SORTINO_NUMERATORS = ("arithmetic", "compounded")
BETWEEN = ("drift", "fixed")
ACTIVE_RETURNS = ("arithmetic", "geometric", "log")
MOMENTS = ("normal", "sample", "sample_unbiased")
DEPENDENCE = ("iid", "autocorrelated")
EULER_GAMMA = 0.5772156649015329


def max_drawdown(values, *, sign):
    """Maximum drawdown of a price or wealth path (levels, not returns), its sign named.

    `values` are levels (prices, an equity curve), not returns: from simple returns r use np.cumprod(np.r_[1.0, 1 +
    r]), whose leading 1.0 counts the starting capital as a peak. A rise-only path has drawdown 0.
    conventions:
        sign: 'loss_positive' (a fall of 25% is 0.25) or 'return_negative' (it is -0.25)
    returns: a float
    source: Magdon-Ismail and Atiya (2004), "Maximum drawdown", Risk 17(10); Bacon, Practical Portfolio
        Performance Measurement and Attribution (2008), drawdown
    aliases: [max drawdown], [maximum drawdown], [MDD], [peak to trough], [drawdown]
    example:
        import qf
        equity = [1.0, 1.08, 0.95, 1.02, 0.90, 1.10]
        qf.max_drawdown(equity, sign="loss_positive")
    """
    choice("sign", sign, SIGNS)
    v = _finite("values", values).ravel()
    if np.any(v <= 0):
        raise ValueError("values must be positive levels (prices or wealth), not returns")
    worst = float(np.max(1.0 - v / np.maximum.accumulate(v)))
    return worst if sign == "loss_positive" else -worst


def brinson_fachler(portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
    """Brinson-Fachler attribution of active return: allocation, selection and interaction.

    R_b = sum_i wb_i rb_i, R_p = sum_i wp_i rp_i; per segment: allocation_i = (wp_i - wb_i)(rb_i - R_b),
    selection_i = wb_i (rp_i - rb_i), interaction_i = (wp_i - wb_i)(rp_i - rb_i); their sum over segments is
    R_p - R_b.

    conventions:
    - weights: each set sums to 1 (include cash as a segment); returns: the period's segment returns, as decimals
    - allocation is against the benchmark's total return (Brinson-Fachler), not against 0 (Brinson-Hood-Beebower)
    - interaction stays separate (selection plus interaction is selection on portfolio weights); one period, no linking
    returns: a dict: allocation, selection and interaction (one per segment); total_allocation,
      total_selection and total_interaction; active_return (R_p - R_b, their sum); in return units
    source: Brinson and Fachler (1985), J. Portfolio Management 11(3); Bacon, Practical Portfolio Performance (2008)
    aliases: Brinson-Fachler, Brinson attribution, [performance attribution], allocation effect, selection effect, BHB
    example:
        qf.brinson_fachler(portfolio_weights=[0.4, 0.3, 0.3], benchmark_weights=[0.4, 0.2, 0.4],
                           portfolio_returns=[0.20, -0.05, 0.06],
                           benchmark_returns=[0.10, -0.04, 0.08])
    """
    wp, wb, rp, rb = (np.asarray(a, dtype=float).ravel() for a in (portfolio_weights, benchmark_weights,
                                                                   portfolio_returns, benchmark_returns))
    if not wp.shape == wb.shape == rp.shape == rb.shape:
        raise ValueError("portfolio_weights, benchmark_weights and both returns must have one entry per segment")
    for name, weights in (("portfolio_weights", wp), ("benchmark_weights", wb)):
        if abs(weights.sum() - 1.0) > 1e-8:
            raise ValueError(f"{name} must sum to 1 (include cash as a segment), not {weights.sum()!r}")
    total_b = float(wb @ rb)
    allocation, selection, interaction = (wp - wb) * (rb - total_b), wb * (rp - rb), (wp - wb) * (rp - rb)
    return Result({"allocation": allocation, "selection": selection, "interaction": interaction,
                   "total_allocation": float(allocation.sum()), "total_selection": float(selection.sum()),
                   "total_interaction": float(interaction.sum()), "active_return": float(wp @ rp - total_b)})


def _log_ratio(a, b):
    """(ln(1 + a) - ln(1 + b)) / (a - b), elementwise, and its limit 1 / (1 + a) where a = b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    same = np.isclose(a, b, rtol=0.0, atol=1e-14)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(same, 1.0 / (1.0 + a), (np.log1p(a) - np.log1p(b)) / np.where(same, 1.0, a - b))


def carino_linking(portfolio_returns, benchmark_returns, effects, active):
    """Carino linking of attribution effects across periods: log-scaled, summing to the active return.

    k_t = (ln(1 + R_p,t) - ln(1 + R_b,t)) / (R_p,t - R_b,t) per period (1 / (1 + R_t) where they are equal) and k
    the same of the cumulative returns R_p = prod(1 + R_p,t) - 1 and R_b; an effect's linked value is sum_t c_t e_t with
    c_t = k_t / k ('arithmetic': the effects add to R_p - R_b), c_t = k_t ('log': they add to ln((1 + R_p)/(1 + R_b)))
    or c_t = k_t G / ln(1 + G) ('geometric': they add to G = (1 + R_p)/(1 + R_b) - 1). The sums hold where each
    period's effects add to its own active return R_p,t - R_b,t (Brinson's allocation, selection and interaction do).

    conventions:
    - portfolio_returns, benchmark_returns: T simple returns per period, as decimals
    - effects: T x m (or T x m x segments) per-period effects, e.g. brinson_fachler's totals stacked by period
    - active: 'arithmetic' (Carino's), 'log' or 'geometric', the active return the effects add to
    returns: a dict: linked (the effects summed over periods, shaped effects[0]); coefficients (c_t, T);
      active_return (in the active's form); portfolio_return and benchmark_return (cumulative)
    source: Carino (1999), "Combining attribution effects over time", J. Performance Measurement 3(4); Bacon (2008), ch. 8
    aliases: Carino linking, Carino smoothing, multi-period attribution, linking attribution effects, logarithmic linking, attribution smoothing, [multi-period Brinson], [Carno linking]
    example:
        # linked = [0.00829770, 0.00414148, 0.00306481]; active_return = 0.015504.
        qf.carino_linking(portfolio_returns=[0.03, -0.01, 0.02], benchmark_returns=[0.02, -0.02, 0.025],
                          effects=[[0.004, 0.005, 0.001], [0.006, 0.003, 0.001], [-0.002, -0.004, 0.001]],
                          active="arithmetic")
    """
    choice("active", active, ACTIVE_RETURNS)
    rp = _series("portfolio_returns", portfolio_returns, 1)
    rb = _series("benchmark_returns", benchmark_returns, rp.size)
    e = np.asarray(effects, dtype=float)
    if rb.size != rp.size or e.ndim < 2 or e.shape[0] != rp.size or not np.all(np.isfinite(e)):
        raise ValueError("benchmark_returns must match portfolio_returns and effects be T x m finite effects")
    if np.any(rp <= -1.0) or np.any(rb <= -1.0):
        raise ValueError("returns must be above -1")
    total_p, total_b = float(np.prod(1.0 + rp) - 1.0), float(np.prod(1.0 + rb) - 1.0)
    k_t = _log_ratio(rp, rb)
    log_active = float(np.log1p(total_p) - np.log1p(total_b))
    if active == "arithmetic":
        coefficients, figure = k_t / float(_log_ratio(total_p, total_b)), total_p - total_b
    elif active == "log":
        coefficients, figure = k_t, log_active
    else:
        figure = (1.0 + total_p) / (1.0 + total_b) - 1.0
        coefficients = k_t * (figure / log_active if abs(log_active) > 1e-14 else 1.0)
    linked = np.tensordot(coefficients, e, axes=(0, 0))
    return Result({"linked": linked, "coefficients": coefficients, "active_return": figure,
                   "portfolio_return": total_p, "benchmark_return": total_b})


def sharpe_inference(returns, periods_per_year, ddof, moments, dependence, lags, benchmark_sharpe, trials,
                     trials_variance):
    """Sharpe ratio inference: Lo's error, the probabilistic (PSR) and deflated (DSR) Sharpe ratios.

    SR = mean / std(ddof) of the excess returns per period, annualised by sqrt(P) ('iid') or Lo's eta(P) = P / sqrt(P
    + 2 sum_{k<P} (P - k) rho_k) ('autocorrelated', rho_k their autocorrelations). Its error: sqrt((1 - g3 SR + (g4 -
    1)/4 SR^2) / T), (1 + SR^2/2)/T for normal returns (Lo 2002; Mertens 2002), or Lo's GMM error with Newey-West lags.
    PSR = N((SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)), the probability the true SR exceeds SR*, per
    period. DSR is PSR at SR* = sqrt(V) ((1 - gamma) N^{-1}(1 - 1/n) + gamma N^{-1}(1 - 1/(n e))), the expected
    largest SR of n trials of variance V (gamma Euler's 0.5772): Bailey and Lopez de Prado. With 'autocorrelated',
    se_annual treats rho as known; PSR and DSR are available only for 'iid'.

    conventions:
    - returns: excess returns per period (subtract the risk-free rate first); periods_per_year: P (an integer for
      'autocorrelated'); ddof: the std's, 0 or 1
    - moments: 'normal' (g3 = 0, g4 = 3), 'sample' (scipy's skew and kurtosis, biased), 'sample_unbiased', or a
      task's own (skewness, kurtosis) pair, the kurtosis not in excess (3 for a normal)
    - dependence: 'iid' (lags None) or 'autocorrelated' (lags, an int: the error's Newey-West lags)
    - benchmark_sharpe: SR* per year (0.0: the probability of skill)
    - trials: None, or the independent trials n behind the strategy; trials_variance: V, the variance of the trials'
      annual Sharpe ratios (None with trials None)
    returns: a dict: sharpe and se (per period); sharpe_annual and se_annual; psr (NaN for 'autocorrelated');
      dsr (NaN for 'autocorrelated' or trials None); sr0 (annual threshold, NaN with trials None); skew, kurtosis
    source: Lo (2002), FAJ 58(4); Mertens (2002); Bailey and Lopez de Prado (2012), J. Risk 15(2); (2014), JPM 40(5)
    aliases: probabilistic Sharpe ratio, deflated Sharpe ratio, PSR, DSR, Sharpe ratio standard error, Sharpe ratio significance, Lo Sharpe ratio
    example:
        # sharpe_annual = 2.49444; psr = 0.955222; dsr = 0.826378.
        qf.sharpe_inference(returns=[0.01, 0.02, -0.01, 0.03, 0.0, 0.015, -0.005, 0.02], periods_per_year=12, ddof=1,
                            moments="normal", dependence="iid", lags=None, benchmark_sharpe=0.0, trials=10,
                            trials_variance=0.5)
    """
    choice("dependence", dependence, DEPENDENCE)
    r = _series("returns", returns, 3)
    periods = _periods(periods_per_year)
    spread = float(np.std(r, ddof=_ddof(ddof)))
    if not spread > 0:
        raise ValueError("returns have no dispersion: the Sharpe ratio is undefined")
    n = r.size
    sr = float(np.mean(r) / spread)
    if not isinstance(moments, str):
        try:
            g3, g4 = (float(x) for x in moments)
        except (TypeError, ValueError):
            raise ValueError("moments must be 'normal', 'sample', 'sample_unbiased' or a (skewness, kurtosis) pair") \
                from None
    elif choice("moments", moments, MOMENTS) == "normal":
        g3, g4 = 0.0, 3.0
    else:
        g3 = float(stats.skew(r, bias=moments == "sample"))
        g4 = float(stats.kurtosis(r, fisher=False, bias=moments == "sample"))
    shape = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if (dependence == "iid") != (lags is None):
        raise ValueError("lags: None with 'iid', an int of at least 0 with 'autocorrelated'")
    if dependence == "iid":
        scale, se = np.sqrt(periods), float(np.sqrt(shape / n))
    else:
        q = int(round(periods))
        width = int(lags) if isinstance(lags, (int, np.integer)) and not isinstance(lags, bool) else -1
        if abs(periods - q) > 1e-12 or q < 1 or q >= n or width < 0:
            raise ValueError("'autocorrelated' needs an integer periods_per_year below the sample size and lags >= 0")
        centred = r - r.mean()
        gamma0 = centred @ centred / n
        rho = np.array([centred[k:] @ centred[:-k] / n / gamma0 for k in range(1, q)])
        scale = q / np.sqrt(q + 2.0 * np.sum((q - np.arange(1, q)) * rho))
        mu, var = float(np.mean(r)), float(centred @ centred / n)
        moments_t = np.column_stack([centred, centred ** 2 - var])
        long_run = moments_t.T @ moments_t / n
        for lag in range(1, width + 1):
            cross = moments_t[lag:].T @ moments_t[:-lag] / n
            long_run += (1.0 - lag / (width + 1.0)) * (cross + cross.T)
        gradient = np.array([1.0 / np.sqrt(var), -mu / (2.0 * var ** 1.5)])
        se = float(np.sqrt(gradient @ long_run @ gradient / n))
    target = float(benchmark_sharpe) / np.sqrt(periods)
    psr = float(stats.norm.cdf((sr - target) * np.sqrt(n - 1) / np.sqrt(shape))) if dependence == "iid" else float("nan")
    dsr = sr0 = float("nan")
    if (trials is None) != (trials_variance is None):
        raise ValueError("trials and trials_variance: both given, or both None")
    if trials is not None:
        count = float(trials)
        if not count >= 2:
            raise ValueError("trials must be at least 2")
        sr0 = float(np.sqrt(float(trials_variance)) * ((1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / count)
                                                       + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (count * np.e))))
        if dependence == "iid":
            dsr = float(stats.norm.cdf((sr - sr0 / np.sqrt(periods)) * np.sqrt(n - 1) / np.sqrt(shape)))
    return Result({"sharpe": sr, "se": se, "sharpe_annual": float(sr * scale), "se_annual": float(se * scale),
                   "psr": psr, "dsr": dsr, "sr0": sr0, "skew": g3, "kurtosis": g4})


def _series(name, values, least):
    """A 1-d float array of finite numbers with at least `least` entries, else a ValueError naming it."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < least or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a 1-d series of at least {least} finite numbers")
    return array


def _periods(periods_per_year):
    return float(positive("periods_per_year", periods_per_year))


def _ddof(ddof):
    if ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, not {ddof!r}")
    return int(ddof)


def _per_period(risk_free, basis, periods):
    """The risk-free rate per period from its quoted basis."""
    choice("risk_free_basis", basis, RISK_FREE_BASES)
    rf = np.asarray(risk_free, dtype=float)
    if basis == "per_period":
        return rf
    if basis == "annual_simple":
        return rf / periods
    if basis == "annual_compound":
        return (1.0 + rf) ** (1.0 / periods) - 1.0
    return np.expm1(rf / periods)


def _excess(returns, risk_free, risk_free_basis, periods_per_year):
    r = _series("returns", returns, 2)
    periods = _periods(periods_per_year)
    rf = _per_period(risk_free, risk_free_basis, periods)
    if rf.ndim and rf.shape != r.shape:
        raise ValueError("risk_free must be a scalar or one rate per return")
    return r - rf, periods


def _wealth(returns):
    """Wealth from 1 before the first return: [1, (1 + r_1), (1 + r_1)(1 + r_2), ...]."""
    r = _series("returns", returns, 1)
    if np.any(r <= -1.0):
        raise ValueError("returns must be above -1 (a loss of everything or more)")
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def sharpe_ratio(returns, risk_free, risk_free_basis, periods_per_year, ddof):
    """Annualised Sharpe ratio: mean excess return over its std times sqrt(P), ddof and risk-free basis named.

    SR = sqrt(P) mean(r_t - rf_t) / std(r_t - rf_t, ddof), rf_t the risk-free rate per period from its basis:
    'per_period' rf as given; 'annual_simple' rf / P; 'annual_compound' (1 + rf)^{1/P} - 1; 'annual_continuous'
    e^{rf/P} - 1. A series with no dispersion is refused.

    Kelly's fraction of wealth for a normal excess return is mu / sigma^2 (the mean excess over its variance).

    Skewness and kurtosis: scipy.stats.skew(r) and scipy.stats.kurtosis(r, fisher=False) (3 for a normal; fisher=True
    gives the excess) are the biased sample moments; bias=False gives the adjusted ones, as pandas' skew() and kurt().

    Treynor's ratio is the mean excess return over beta, (mean(r) - rf) / beta, annualised by P, not sqrt(P).

    Jensen's alpha is the intercept of the excess return on the market's excess return (ols with an intercept).

    M^2 (Modigliani) is rf + SR sigma_m: the Sharpe ratio scaled to the market's vol, annual figures throughout.

    conventions:
    - risk_free: a scalar or one rate per return, in risk_free_basis; 0.0 for a ratio on raw returns
    - periods_per_year: P, the periods in a year (252 daily, 52 weekly, 12 monthly); the sqrt(P) scaling
    - ddof: 1 for the sample standard deviation (pandas' default), 0 for the population one (numpy's)
    returns: a float, annualised by sqrt(P)
    source: Sharpe (1994), "The Sharpe ratio", J. Portfolio Management 21(1); Lo (2002), Financial Analysts Journal
    aliases: [Sharpe ratio], [annualized Sharpe], [Sharpe], [risk-adjusted return], [reward-to-variability ratio]
    example:
        qf.sharpe_ratio(returns=[0.01, 0.02, -0.01, 0.03], risk_free=0.02,
                        risk_free_basis="annual_compound",
                        periods_per_year=12, ddof=1)
    """
    excess, periods = _excess(returns, risk_free, risk_free_basis, periods_per_year)
    spread = np.std(excess, ddof=_ddof(ddof))
    if not spread > 0:
        raise ValueError("returns have no dispersion: the Sharpe ratio is undefined")
    return float(np.sqrt(periods) * np.mean(excess) / spread)


def sortino_ratio(returns, risk_free, risk_free_basis, periods_per_year, ddof, numerator):
    """Annualized Sortino ratio: excess return over the downside deviation, the mean's or the compounded's.

    'arithmetic': Sortino = sqrt(P) mean(r_t - rf_t) / DD,  DD = sqrt( sum_t min(r_t - rf_t, 0)^2 / (n - ddof) ),
    the target (the minimum acceptable return) being rf_t, from its basis as in sharpe_ratio. 'compounded': (CAGR -
    rf_annual) / (DD sqrt(P)), CAGR the returns' compound annual growth and rf_annual the risk-free rate as an effective
    annual rate. The sum runs over all n periods, a period above the target adding 0, not over the losing periods
    alone. No period below the target is refused (the ratio is unbounded).

    conventions:
    - risk_free: the target, in risk_free_basis (the four of sharpe_ratio); 0.0 for a zero target
    - periods_per_year: P, the sqrt(P) scaling; ddof: 0 divides by n (Sortino and Price), 1 by n - 1
    - numerator: 'arithmetic' (the mean excess return times P) or 'compounded' (the compound annual return less rf)
    returns: a float, annualised
    source: Sortino and Price (1994), "Performance measurement in a downside risk framework", J. Investing 3(3)
    aliases: [Sortino ratio], [Sortino], [downside deviation], [downside risk ratio], [target downside deviation]
    example:
        qf.sortino_ratio(returns=[0.01, 0.02, -0.01, 0.03, -0.02], risk_free=0.0,
                         risk_free_basis="per_period",
                         periods_per_year=252, ddof=0, numerator="arithmetic")
    """
    choice("numerator", numerator, SORTINO_NUMERATORS)
    excess, periods = _excess(returns, risk_free, risk_free_basis, periods_per_year)
    below = np.minimum(excess, 0.0)
    if not np.any(below < 0):
        raise ValueError("no return falls below the target: the Sortino ratio is unbounded")
    downside = np.sqrt(np.sum(below * below) / (excess.size - _ddof(ddof)))
    if numerator == "arithmetic":
        return float(np.sqrt(periods) * np.mean(excess) / downside)
    r = _series("returns", returns, 2)
    rf = np.broadcast_to(_per_period(risk_free, risk_free_basis, periods), r.shape)
    growth = cagr(r, periods) - (np.prod(1.0 + rf) ** (periods / r.size) - 1.0)
    return float(growth / (downside * np.sqrt(periods)))


def max_drawdown_dates(values, dates):
    """Maximum drawdown as a positive fraction, with the dates of its peak, trough and recovery.

    peak_t = max_{s <= t} V_s, drawdown_t = 1 - V_t / peak_t (0 at a new high), max_drawdown = max_t drawdown_t;
    trough the first date of the maximum, peak the last date at or before it where V reached its high, recovery the
    first date after the trough where V is back at or above that high (None if it never is). With no decline the
    maximum is 0.0 and the three dates None.

    conventions:
    - values: positive levels (wealth, NAV, prices), not returns; from returns use [1, cumprod(1 + r)]
    - max_drawdown: a positive fraction of the peak (0.25 is a 25% decline); negate it for a signed figure
    - dates: one label per value (ISO strings, datetimes or positions), returned as given
    returns: a dict: max_drawdown (a positive fraction of the peak); peak, trough and recovery (dates as
      given; recovery None if never); drawdowns (the path 1 - V_t / peak_t, one per value)
    source: Magdon-Ismail and Atiya (2004), "Maximum drawdown", Risk 17(10); Bacon, Practical Portfolio Performance
    aliases: [max drawdown], [maximum drawdown], [MDD], [peak-to-trough decline], [drawdown dates], [drawdown recovery], [maximum drawdown dates]
    example:
        qf.max_drawdown_dates(values=[100.0, 120.0, 90.0, 130.0, 80.0, 135.0],
                              dates=["2024-01-31", "2024-02-29", "2024-03-31",
                                     "2024-04-30", "2024-05-31", "2024-06-30"])
    """
    v = _series("values", values, 1)
    if np.any(v <= 0):
        raise ValueError("values must be positive levels, not returns")
    if len(dates) != v.size:
        raise ValueError("dates must hold one label per value")
    high = np.maximum.accumulate(v)
    drawdowns = 1.0 - v / high
    worst = float(np.max(drawdowns))
    if worst == 0.0:
        return Result({"max_drawdown": 0.0, "peak": None, "trough": None, "recovery": None, "drawdowns": drawdowns})
    trough = int(np.argmax(drawdowns))
    peak = int(np.flatnonzero(v[:trough + 1] == high[trough])[-1])
    back = np.flatnonzero(v[trough + 1:] >= high[trough])
    recovery = int(trough + 1 + back[0]) if back.size else None

    def label(i):
        if i is None:
            return None
        return dates[i].item() if isinstance(dates[i], np.generic) else dates[i]

    return Result({"max_drawdown": worst, "peak": label(peak), "trough": label(trough), "recovery": label(recovery),
                   "drawdowns": drawdowns})


def cagr(returns, periods_per_year):
    """Compound annual growth rate of a simple return series (the geometric annual return).

    CAGR = [prod_t (1 + r_t)]^{P / n} - 1 over the n returns: the constant annual rate that compounds to the same
    final wealth. Not the arithmetic mean times P (annualized_mean_return), which does not compound.

    conventions:
    - returns: simple per-period returns, above -1; n of them span n / P years
    - periods_per_year: P (252 daily, 12 monthly); from dates instead of periods, P = n / years elapsed
    returns: a float, a decimal (0.08 is 8% a year)
    source: Bacon, Practical Portfolio Performance Measurement and Attribution (2008), annualised returns
    aliases: [CAGR], [compound annual growth rate], [geometric mean return], [annualized return], [annualised return]
    example:
        qf.cagr(
            returns=[0.02, -0.01, 0.015, 0.03, -0.005, 0.01], periods_per_year=12
        )
    """
    wealth = _wealth(returns)
    return float(wealth[-1] ** (_periods(periods_per_year) / (wealth.size - 1)) - 1.0)


def annualized_mean_return(returns, periods_per_year):
    """Arithmetic annual mean return: the mean per-period return times periods a year (not CAGR).

    mu_annual = P mean_t(r_t), the drift a Sharpe ratio's numerator annualizes. It does not compound: a constant
    return g gives P g here and (1 + g)^P - 1 from cagr, and varying returns lower cagr further (the variance drag).

    conventions:
    - returns: simple per-period returns; log returns give the continuously compounded mean instead
    - periods_per_year: P (252 daily, 12 monthly)
    returns: a float: P times the mean per-period return
    source: Bacon, Practical Portfolio Performance Measurement and Attribution (2008), annualised returns
    aliases: [annualized mean return], [arithmetic annual return], [mean times 252], [annualised average return], [annual drift]
    example:
        qf.annualized_mean_return(
            returns=[0.002, -0.001, 0.0015, 0.003], periods_per_year=252
        )
    """
    return float(_periods(periods_per_year) * np.mean(_series("returns", returns, 1)))


def calmar_ratio(returns, periods_per_year, annual_return):
    """Calmar ratio: the annual return over the maximum drawdown of the compounded wealth.

    Calmar = A / MDD, A the annual return named ('cagr': prod(1 + r)^{P/n} - 1; 'arithmetic': P mean(r)) and MDD
    the maximum drawdown (a positive fraction) of the wealth 1, (1 + r_1), (1 + r_1)(1 + r_2), ..., which starts at 1
    before the first return, so a first-period loss is a drawdown. A series that never declines is refused.

    conventions:
    - annual_return: 'cagr' (Young's original, the compound rate) or 'arithmetic' (the mean times P)
    - periods_per_year: P; the drawdown over the whole series given (Young used the last 36 months)
    returns: a float
    source: Young (1991), "Calmar ratio: a smoother tool", Futures 20(1); Bacon (2008), drawdown measures
    aliases: [Calmar ratio], [Calmar], [return over max drawdown], [MAR ratio], [drawdown ratio]
    example:
        qf.calmar_ratio(returns=[0.02, -0.05, 0.03, 0.01, -0.02, 0.04],
                        periods_per_year=12,
                        annual_return="cagr")
    """
    choice("annual_return", annual_return, ANNUAL_RETURNS)
    wealth = _wealth(returns)
    worst = float(np.max(1.0 - wealth / np.maximum.accumulate(wealth)))
    if worst == 0.0:
        raise ValueError("returns never draw down: the Calmar ratio is unbounded")
    rate = cagr(returns, periods_per_year) if annual_return == "cagr" else annualized_mean_return(returns,
                                                                                               periods_per_year)
    return rate / worst


@_guard(price=('close', 'open',))
def backtest_signals(signals, close, open, lag, execution, cost_bps):
    """Strategy returns of trading a signal: positions lagged, filled at the close or next open, less costs.

    A signal s_t, known at bar t's close, is a target position as a fraction of wealth (1 long, -1 short, 0 flat).
    positions p_t = s_{t-lag} (0 before the first fill), filled at bar t-1's close ('close') or t's open ('next_open');
    'close':     gross_t = p_t (C_t / C_{t-1} - 1), so lag 1 trades at the signal's own close;
    'next_open': gross_t = (1 + p_{t-1} (O_t / C_{t-1} - 1)) (1 + p_t (C_t / O_t - 1)) - 1,
                 the overnight move on the old position, the session on the new, so lag 1 is the next day's open;
    turnover_t = |p_t - p_{t-1}|, costs_t = turnover_t cost_bps / 10,000, returns_t = gross_t - costs_t; bar 0 is 0.

    conventions:
    - lag: positions trail signals by lag bars, at least 1 (0 would fill before the signal is known)
    - execution: 'close' or 'next_open'; open is read only by 'next_open' (None is fine for 'close')
    - cost_bps: charged on each change of position, both ways; no closing trade is charged at the end
    - signals: finite (fill a warm-up's NaN first); positions rebalance to their fraction of wealth every bar
    returns: a dict: returns (net of costs), gross, positions, turnover and costs, each an array of one
      entry per bar, returns as decimals; wealth is cumprod(1 + returns)
    source: Chan, Quantitative Trading (2008), backtesting; Bacon (2008), returns from positions and costs
    aliases: backtest, [signal backtest], [vectorized backtest], [positions from signals], [transaction costs], [next open]
    example:
        qf.backtest_signals(signals=[0, 1, 1, 0, -1, 0], close=[10.0, 10.2, 10.1, 10.4, 10.3, 10.0],
                            open=[10.0, 10.1, 10.25, 10.2, 10.35, 10.2], lag=1,
                            execution="next_open", cost_bps=5.0)
    """
    choice("execution", execution, EXECUTIONS)
    s = _series("signals", signals, 1)
    c = _series("close", close, s.size)
    if c.size != s.size or np.any(c <= 0):
        raise ValueError("close must hold one positive price per signal")
    if not isinstance(lag, (int, np.integer)) or isinstance(lag, bool) or lag < 1:
        raise ValueError(f"lag must be a whole number of bars, at least 1, not {lag!r}")
    if not cost_bps >= 0:
        raise ValueError("cost_bps must not be negative")
    p = np.zeros_like(s)
    if lag < s.size:
        p[lag:] = s[:-lag]
    held = np.concatenate([[0.0], p[:-1]])
    gross = np.zeros_like(s)
    if execution == "close":
        gross[1:] = p[1:] * (c[1:] / c[:-1] - 1.0)
    else:
        o = _series("open", open, s.size)
        if o.size != s.size or np.any(o <= 0):
            raise ValueError("open must hold one positive price per signal")
        overnight, session = o[1:] / c[:-1] - 1.0, c[1:] / o[1:] - 1.0
        gross[1:] = (1.0 + held[1:] * overnight) * (1.0 + p[1:] * session) - 1.0
    turnover = np.abs(p - held)
    costs = turnover * cost_bps / 1e4
    return Result({"returns": gross - costs, "gross": gross, "positions": p, "turnover": turnover, "costs": costs})


def vol_target_leverage(returns, target_vol, lookback, periods_per_year, ddof, max_leverage, lag):
    """Vol-targeting leverage: target over trailing annualised vol (ddof), capped, lagged: lag=1 times the same bar's return.

    sigma_t = std(r_{t-L+1}, ..., r_t; ddof) sqrt(P), raw_t = min(target_vol / sigma_t, max_leverage), and
    leverage_t = raw_{t-lag}. With lag 1 each bar's leverage comes from the returns through the bar before, so
    leverage * returns is the strategy's return bar by bar, with no further shift; lag 0 gives raw_t, from returns
    through bar t itself, which a backtest must shift before use (leverage[:-1] * returns[1:]).

    conventions:
    - lag: the bars the leverage trails its window, at least 0: 1 to size each bar on the data before it, 0 for the
      unlagged series
    - target_vol: annualized, as sigma_t is (0.10 is 10% a year); lookback: L, the bars in the trailing window
    - periods_per_year: P for the sqrt(P) annualization; ddof: 0 or 1 in the window's standard deviation
    - max_leverage: the cap (numpy.inf for none); a window with zero vol takes the cap
    returns: an array of leverage, one per bar (numpy.nan for the first lookback - 1 + lag bars)
    source: Moreira and Muir (2017), "Volatility-managed portfolios", J. Finance 72(4); Harvey et al. (2018)
    aliases: volatility targeting, vol targeting, vol target, volatility scaling, risk targeting, inverse vol scaling
    example:
        qf.vol_target_leverage(returns=[0.01, -0.02, 0.015, -0.005, 0.02, -0.01],
                               target_vol=0.10, lookback=3, periods_per_year=252,
                               ddof=1, max_leverage=2.0, lag=1)
    """
    r = _series("returns", returns, 1)
    periods = _periods(periods_per_year)
    if not isinstance(lookback, (int, np.integer)) or isinstance(lookback, bool) or not 2 <= lookback <= r.size:
        raise ValueError(f"lookback must be a whole number of bars from 2 to len(returns), not {lookback!r}")
    if not isinstance(lag, (int, np.integer)) or isinstance(lag, bool) or lag < 0:
        raise ValueError(f"lag must be a whole number of bars, 0 or more (1 sizes each bar on the bars before it), "
                         f"not {lag!r}")
    if lookback + lag > r.size:
        raise ValueError("lookback + lag must not exceed len(returns): no bar would have a leverage")
    target = float(positive("target_vol", target_vol))
    cap = float(positive("max_leverage", max_leverage))
    windows = np.lib.stride_tricks.sliding_window_view(r, lookback)
    sigma = np.std(windows, axis=1, ddof=_ddof(ddof)) * np.sqrt(periods)
    with np.errstate(divide="ignore"):
        lever = np.minimum(target / sigma, cap)
    return np.concatenate([np.full(lookback - 1 + lag, np.nan), lever[:lever.size - lag]])


def rebalance_path(returns, weights, rebalance, initial_value, cash_return, timing, fee_order, turnover,
                   turnover_base, cost_rate, start, between_rebalances):
    """Self-financing rebalancing with drifted weights, turnover and costs.

    Bar t earns r_p = sum_i w_i r_ti + (1 - sum_i w_i) r_cash on the weights held, which drift between trades,
    w_i (1 + r_ti) / (1 + r_p), or stay at the target ('fixed'). A trade's turnover is sum_i |w_target - w_base|
    ('two_way') or half that ('one_way'); its cost, cost_rate times the turnover, comes off before the bar's return
    (V_t = V (1 - c)(1 + r_p)) or after it (V_t = V (1 + r_p) - V c). With no cost V_T / V_0 = prod(1 + r_p).
    Concentration: the Herfindahl index sum_i w_i^2, the effective number of names its inverse.

    conventions:
    - returns: T x N simple returns per bar; weights: the targets, N (fixed) or T x N (row t bar t's); cash the rest
    - rebalance: T booleans, True where bar t trades to its target; cash_return: per bar, a number or T (0.0 for none)
    - timing: 'start' (row t traded at bar t's start, earning bar t) or 'end' (traded at its end: one bar later)
    - fee_order: 'before' or 'after' the bar's return; cost_rate: per unit of turnover (0.001 is 10 bps)
    - turnover: 'one_way' or 'two_way', over the assets (cash left out); turnover_base: 'drifted' or 'target'
    - start: 'cash' (the first trade buys the book, turnover and cost counted) or 'invested' (the first target held)
    - between_rebalances: 'drift' (weights move with prices: buy and hold) or 'fixed' (held at the target every bar)
    returns: a dict: values (V after each bar); returns (net, per bar); gross_returns (r_p); turnover, costs (in value)
      per bar; weights (T x N held through each bar); drifted (T x N at each bar's end)
    source: Bacon, Practical Portfolio Performance Measurement and Attribution (2008); Grinold and Kahn (2000), ch. 16
    aliases: rebalancing, portfolio rebalancing, drifted weights, weight drift, self-financing portfolio, buy and hold, constant mix, [turnover], [transaction costs], [Herfindahl], [backtest]
    example:
        qf.rebalance_path(returns=[[0.02, -0.01], [0.01, 0.03]], weights=[0.6, 0.4], rebalance=[True, True],
            initial_value=100.0, cash_return=0.0, timing="start", fee_order="before", turnover="two_way",
            turnover_base="drifted", cost_rate=0.001, start="cash", between_rebalances="drift")
    """
    choice("timing", timing, TIMINGS)
    choice("fee_order", fee_order, FEE_ORDERS)
    choice("turnover", turnover, TURNOVERS)
    choice("turnover_base", turnover_base, TURNOVER_BASES)
    choice("start", start, STARTS)
    choice("between_rebalances", between_rebalances, BETWEEN)
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 1 or not np.all(np.isfinite(r)) or np.any(r <= -1):
        raise ValueError("returns must be T x N finite simple returns above -1")
    bars, names = r.shape
    target = np.asarray(weights, dtype=float)
    if target.shape not in ((names,), (bars, names)) or not np.all(np.isfinite(target)):
        raise ValueError("weights must be N or T x N finite weights")
    target = np.broadcast_to(target, (bars, names)).copy()
    trade = np.asarray(rebalance).ravel()
    if trade.shape != (bars,) or trade.dtype != bool:
        raise ValueError("rebalance must be T booleans")
    cash = np.broadcast_to(np.asarray(cash_return, dtype=float), (bars,))
    rate, value = float(cost_rate), float(positive("initial_value", initial_value))
    if not rate >= 0 or not np.all(np.isfinite(cash)):
        raise ValueError("cost_rate must not be negative and cash_return must be finite")
    if timing == "end":            # row t traded at bar t's end is bar t + 1's start; before it, all cash
        target, trade = np.vstack([np.zeros((1, names)), target[:-1]]), np.r_[False, trade[:-1]]
    held = target[0].copy() if start == "invested" else np.zeros(names)
    last = held.copy()
    found = {key: np.zeros(bars) for key in ("values", "returns", "gross_returns", "turnover", "costs")}
    found["weights"], found["drifted"] = np.zeros((bars, names)), np.zeros((bars, names))
    for t in range(bars):
        traded = 0.0
        if trade[t]:
            traded = float(np.abs(target[t] - (held if turnover_base == "drifted" else last)).sum())
            traded *= 0.5 if turnover == "one_way" else 1.0
            held, last = target[t].copy(), target[t].copy()
        gross = float(held @ r[t] + (1.0 - held.sum()) * cash[t])
        charge = rate * traded
        after = value * (1.0 - charge) * (1.0 + gross) if fee_order == "before" else value * (1.0 + gross - charge)
        found["weights"][t] = held
        if between_rebalances == "drift":
            held = held * (1.0 + r[t]) / (1.0 + gross)
        found["drifted"][t] = held
        for key, number in (("returns", after / value - 1.0), ("gross_returns", gross), ("turnover", traded),
                            ("costs", value * charge), ("values", after)):
            found[key][t] = number
        value = after
    return Result(found)
