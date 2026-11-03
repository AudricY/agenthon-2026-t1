"""Inflation: Ref CPI of TIPS and linkers, breakeven CPI from inflation swaps (ZCIS), YoY convexity (Jarrow-Yildirim).

`import qf`. Times are year fractions from today under the caller's day count; index levels are the CPI's own units;
rates are decimals. A scalar result is a float; a function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np

from ._common import Result, choice, integer, out, positive
from .conventions import _dates
from .curves import _increasing
from .primitives import _guard

__all__ = ["implied_cpi_curve", "reference_cpi", "yoy_convexity_adjustment"]

CPI_INTERPOLATIONS = ("daily", "monthly")
CPI_ROUNDINGS = ("treasury", "none")
INDEX_INTERPOLATIONS = ("log_linear", "linear_rate")


def _treasury(values):
    """31 CFR 356 Appendix B: truncated to six decimals, then rounded (half up) to five."""
    micro = np.trunc(np.round(np.asarray(values, dtype=float) * 1e6, 3))
    return np.floor((micro + 5.0) / 10.0) / 1e5


def reference_cpi(dates, cpi, lag_months, interpolation, base_cpi, rounding):
    """Reference CPI of an inflation-indexed bond (TIPS, linkers) by one model, lagged and interpolated, and its index ratio.

    Ref CPI on day d of month M = CPI_{M-L} + (d - 1) / D_M (CPI_{M-L+1} - CPI_{M-L}), L the lag in months (three for
    TIPS) and D_M the days in month M ('daily'); 'monthly' holds CPI_{M-L} all month. Index ratio = Ref CPI / base,
    the base being the Ref CPI of the bond's dated date; the indexed principal is face times the ratio, each coupon
    the real coupon rate over its frequency times that. 'treasury' truncates each to six decimals and rounds it to
    five, as 31 CFR 356 Appendix B does.

    The deflation floor: TIPS repay max(face, face times the ratio) at maturity, the coupons get no floor. The Fisher
    relation: (1 + nominal yield) = (1 + real yield)(1 + inflation), so the breakeven is about nominal less real.

    conventions:
    - the model: this lagged, interpolated Ref CPI alone; a task that states its own index ratio or steps (another lag
      or none, its own ratio period by period) is followed by hand, not by this function
    - dates: ISO 'YYYY-MM-DD' (one or a list); cpi: a dict {'YYYY-MM': index level} holding months M - L and M - L + 1
    - lag_months: L, 3 for TIPS, UK linkers since 2005 and Canada's real return bonds (8 for older gilts)
    - interpolation: 'daily' (within the month, by day) or 'monthly' (the lagged month's level all month)
    - base_cpi: the base Ref CPI for the index ratio, or None for the Ref CPI alone
    - rounding: 'treasury' (truncate to six decimals, round to five) or 'none'
    returns: a dict: reference_cpi (a float, an array for several dates); index_ratio (with base_cpi)
    source: 31 CFR 356 Appendix B (Treasury inflation-protected securities); Fabozzi, Bond Markets, Analysis and
      Strategies, ch. 7
    aliases: reference CPI, Ref CPI, index ratio, CPI interpolation, TIPS index ratio, inflation-adjusted principal, [TIPS], [CPI-U]
    example:
        qf.reference_cpi(dates="2024-04-15", cpi={"2024-01": 300.0, "2024-02": 301.5}, lag_months=3,
                         interpolation="daily", base_cpi=None, rounding="none")  # reference_cpi 300.7
    """
    choice("interpolation", interpolation, CPI_INTERPOLATIONS)
    choice("rounding", rounding, CPI_ROUNDINGS)
    days = _dates("dates", dates)
    lag = np.timedelta64(integer("lag_months", lag_months, 0), "M")
    if not isinstance(cpi, dict) or not cpi:
        raise ValueError("cpi must be a dict {'YYYY-MM': level}")
    table = {np.datetime64(str(month)[:7], "M"): float(level) for month, level in cpi.items()}
    months = days.astype("datetime64[M]")
    day = (days - months.astype("datetime64[D]")).astype(int) + 1
    one = np.timedelta64(1, "M")
    length = ((months + one).astype("datetime64[D]") - months.astype("datetime64[D]")).astype(int)

    def level(month):
        if month not in table:
            raise ValueError(f"cpi holds no level for {month}")
        return table[month]

    flat = months.ravel()
    low = np.array([level(month - lag) for month in flat]).reshape(days.shape)
    if interpolation == "daily":
        high = np.array([level(month - lag + one) for month in flat]).reshape(days.shape)
        found = low + (day - 1) / length * (high - low)
    else:
        found = low
    if rounding == "treasury":
        found = _treasury(found)
    result = Result({"reference_cpi": out(found)})
    if base_cpi is not None:
        ratio = found / float(positive("base_cpi", base_cpi))
        result["index_ratio"] = out(_treasury(ratio) if rounding == "treasury" else ratio)
    return result


@_guard(rate=('swap_rates',), time=('maturities', 'times',), price=('base_index',))
def implied_cpi_curve(base_index, maturities, swap_rates, times, interpolation):
    """Implied CPI curve (forward index) from zero-coupon inflation swap rates, read at any times.

    A zero-coupon inflation swap of maturity T fixes (1 + K_T)^T - 1 against I(T) / I(0) - 1, so the index the market
    implies is I(0, T) = I(0) (1 + K_T)^T at each quote, which the curve reprices. Between quotes, 'log_linear' holds ln
    I linear in time from (0, ln I(0)) (a constant forward inflation rate), 'linear_rate' the swap rate K linear in
    maturity (flat below the first); times past the last quote are refused. A zero breakeven gives a flat index.

    The naive YoY forward is I(0, T_i) / I(0, T_{i-1}) - 1, before qf.yoy_convexity_adjustment's correction. The
    breakeven K is expected inflation plus an inflation risk premium (less a liquidity premium), not the forecast
    alone; by Fisher it is about the nominal less the real zero yield.

    conventions:
    - base_index: I(0), the index level the swaps reference (the lagged fixing, three months back for most markets)
    - maturities: the swaps' tenors in years, increasing; swap_rates: their fixed rates K_T, decimals, compounded
      annually
    - times: the tenors to read the curve at, from 0 to the last maturity
    - interpolation: 'log_linear' (in the log index) or 'linear_rate' (in the swap rate)
    returns: a dict: index (I(0, t) at the times); zero_rates (the breakeven K(t) = (I(0, t) / I(0))^{1/t} - 1)
    source: Brigo and Mercurio, Interest Rate Models (2006), ch. 15 (zero-coupon inflation-indexed swaps); Kerkhof
      (2005), "Inflation derivatives explained", Lehman Brothers
    aliases: zero-coupon inflation swap, ZCIS, inflation swap curve, implied CPI, forward CPI, breakeven inflation curve, inflation curve bootstrap, [breakeven inflation]
    example:
        qf.implied_cpi_curve(base_index=300.0, maturities=[1.0, 2.0, 5.0, 10.0],
                             swap_rates=[0.025, 0.024, 0.023, 0.0235], times=[0.5, 1.0, 3.0, 7.0],
                             interpolation="log_linear")
        # index [303.72685, 307.5, 321.59843, 352.45165]; zero_rates [0.025, 0.025, 0.02344432, 0.02328568]
    """
    choice("interpolation", interpolation, INDEX_INTERPOLATIONS)
    start = float(positive("base_index", base_index))
    tenors = _increasing("maturities", maturities)
    rates = np.asarray(swap_rates, dtype=float).ravel()
    if rates.shape != tenors.shape or np.any(~(rates > -1.0)):
        raise ValueError("swap_rates must hold one rate above -1 per maturity")
    t = np.asarray(times, dtype=float)
    if np.any(t < 0) or np.any(t > tenors[-1] * (1 + 1e-12)):
        raise ValueError("times must lie between 0 and the last maturity (no extrapolation)")
    if interpolation == "log_linear":
        logs = np.log(start) + tenors * np.log1p(rates)
        index = np.exp(np.interp(t, np.r_[0.0, tenors], np.r_[np.log(start), logs]))
    else:
        index = start * np.exp(t * np.log1p(np.interp(t, tenors, rates)))
    with np.errstate(divide="ignore", invalid="ignore"):
        zero = np.where(t > 0, np.expm1(np.log(index / start) / t), rates[0])
    return Result({"index": out(index), "zero_rates": out(zero)})


@_guard(vol=('nominal_vol', 'real_vol', 'index_vol',), time=('start_times', 'end_times',))
def yoy_convexity_adjustment(start_times, end_times, index_start, index_end, nominal_reversion, nominal_vol,
                             real_reversion, real_vol, index_vol, rho_nominal_real, rho_real_index):
    """Year-on-year (YoY) inflation forward with its Jarrow-Yildirim convexity adjustment, period by period.

    A YoY swaplet pays I(T_i) / I(T_{i-1}) - 1 at T_i. Under Jarrow-Yildirim (Hull-White nominal and real rates,
    a lognormal index) its forward is (I(0, T_i) / I(0, T_{i-1})) e^C - 1, the naive ratio of the implied index
    corrected by C = s_r B_r(T_{i-1}, T_i) [B_r(0, T_{i-1}) (rho_rI s_I - s_r B_r(0, T_{i-1}) / 2 + rho_nr s_n (1 +
    a_r B_n(0, T_{i-1})) / (a_n + a_r)) - rho_nr s_n B_n(0, T_{i-1}) / (a_n + a_r)], B_x(s, u) = (1 - e^{-a_x (u -
    s)}) / a_x. Zero real-rate vol gives no adjustment; so does a period starting today. A YoY swap's floating leg is
    sum_i psi_i P_n(0, T_i) times each forward.

    conventions:
    - start_times, end_times: each period's T_{i-1} (0 for one starting today) and T_i, years (arrays per period)
    - index_start, index_end: the implied index I(0, T_{i-1}) and I(0, T_i) (qf.implied_cpi_curve), one each
    - nominal_reversion, real_reversion: a_n, a_r (positive); nominal_vol, real_vol: s_n, s_r, the rates' absolute
      vols; index_vol: s_I, the index's lognormal vol
    - rho_nominal_real: rho_nr between the nominal and real rates; rho_real_index: rho_rI between the real rate and
      the index
    returns: a dict: yoy_rates (the adjusted forwards); naive_rates (the index ratios less 1); convexity_factors (e^C)
    source: Jarrow and Yildirim (2003), J. Financial and Quantitative Analysis 38(2); Brigo and Mercurio, Interest
      Rate Models (2006), 15.4 (year-on-year inflation-indexed swaps)
    aliases: YoY convexity adjustment, year-on-year inflation swap, YoY inflation swap, YoY swaplet, Jarrow-Yildirim, JY model, inflation convexity adjustment, YYIIS
    example:
        qf.yoy_convexity_adjustment(start_times=[1.0, 2.0], end_times=[2.0, 3.0], index_start=[307.5, 315.0],
                                    index_end=[315.0, 322.5], nominal_reversion=0.1, nominal_vol=0.01,
                                    real_reversion=0.1, real_vol=0.008, index_vol=0.02, rho_nominal_real=0.5,
                                    rho_real_index=0.3)  # yoy_rates [0.02442418, 0.02385588]
    """
    t0, t1 = (np.asarray(x, dtype=float) for x in (start_times, end_times))
    i0, i1 = positive("index_start", index_start), positive("index_end", index_end)
    if np.any(t0 < 0) or np.any(~(t1 > t0)):
        raise ValueError("start_times must be at least 0 and each end time after its start")
    a_n = float(positive("nominal_reversion", nominal_reversion))
    a_r = float(positive("real_reversion", real_reversion))
    s_n, s_r, s_i = float(nominal_vol), float(real_vol), float(index_vol)
    rho_nr, rho_ri = float(rho_nominal_real), float(rho_real_index)
    if min(s_n, s_r, s_i) < 0 or max(abs(rho_nr), abs(rho_ri)) > 1:
        raise ValueError("the vols must not be negative and the correlations must lie in [-1, 1]")

    def big_b(a, s, u):
        return -np.expm1(-a * (u - s)) / a

    b_r, b_n = big_b(a_r, 0.0, t0), big_b(a_n, 0.0, t0)
    pull = rho_nr * s_n / (a_n + a_r)
    c = s_r * big_b(a_r, t0, t1) * (b_r * (rho_ri * s_i - 0.5 * s_r * b_r + pull * (1 + a_r * b_n)) - pull * b_n)
    ratio = i1 / i0
    return Result({"yoy_rates": out(ratio * np.exp(c) - 1.0), "naive_rates": out(ratio - 1.0),
                   "convexity_factors": out(np.exp(c))})
