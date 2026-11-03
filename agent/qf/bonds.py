"""Bonds: cash flows, price and yield, duration, convexity, key rates, z-spread, immunization, floaters, mortgages, CTD.

`import qf`. Times are year fractions from today under the caller's day count; a yield's compounding and a curve's
interpolation are named where a function takes them. A scalar result is a float; a function with several results
returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ._common import Result, choice, integer, positive
from .conventions import BASES, _dates, year_fraction
from .curves import (INTERPOLATIONS, _curve, _flows, _increasing, _whole_frequency, _zero_compounding,
                     _zero_discounts)
from .primitives import _guard

__all__ = ["bond_cash_flows", "bond_futures_ctd", "bond_price", "bond_risk", "bond_yield", "curve_risk", "frn_price",
           "immunization_weights", "immunize_lp", "key_rate_durations", "mortgage_cash_flows", "z_spread"]

ZERO_INTERPOLATIONS = ("linear_zero", "log_linear_discount")
ICMA = "ACT/ACT ICMA"
BOND_DAY_COUNTS = (*BASES, ICMA)
ROLLS = ("same_day", "end_of_month")
STUBS = ("short_front", "long_front", "short_back", "long_back")


def _yield_discounts(t, ytm, frequency):
    """Discount factors at times t from a yield compounded `frequency` times a year, or continuously."""
    if frequency == "continuous":
        return np.exp(-ytm * t)
    f = _whole_frequency(frequency)
    base = 1.0 + ytm / f
    if not base > 0:
        raise ValueError(f"ytm must exceed -{f} for frequency {f}")
    return base ** (-f * t)


@_guard(rate=('ytm',), time=('times',))
def bond_price(cash_flows, times, ytm, frequency):
    """Price of a stream of cash flows at a yield to maturity compounded f times a year.

    P = sum_i CF_i (1 + y/f)^{-f t_i}, or sum_i CF_i e^{-y t_i} for 'continuous'. The price at a settlement date
    between coupons is the dirty (full) price; the clean price is it less the accrued interest.

    conventions:
    - times: years from settlement to each cash flow under the bond's day count (k / f on coupon dates)
    - frequency: the yield's compounding, a whole number a year (2 for a semi-annual bond's yield) or 'continuous'
    - cash_flows: coupons and principal per the face the price is quoted on (per 100 for a percent price)
    returns: a float, the dirty price in the cash flows' units
    source: Fabozzi, Bond Markets, Analysis and Strategies, ch. 2 and 3; Hull, Options, Futures, and Other Derivatives
    aliases: [bond price], [price from yield], [present value of cash flows], [dirty price], [full price], [PV at yield]
    example:
        qf.bond_price(cash_flows=[2.5, 2.5, 2.5, 102.5],
                      times=[0.5, 1.0, 1.5, 2.0], ytm=0.045,
                      frequency=2)
    """
    cf, t = _flows(cash_flows, times)
    return float(cf @ _yield_discounts(t, float(ytm), frequency))


@_guard(time=('times',), price=('price',))
def bond_yield(price, cash_flows, times, frequency):
    """Yield to maturity: the yield compounded f times a year that prices the cash flows.

    Solves sum_i CF_i (1 + y/f)^{-f t_i} = price (or with e^{-y t_i} for 'continuous') by Brent's method on y in
    (-0.99 f, 100) (-1 to 100 for 'continuous'), to 1e-14; for positive cash flows the root is unique. A price
    outside the range those yields give is refused.

    conventions:
    - price: the dirty (full) price, on the face the cash flows use; add the accrued interest to a clean price
    - times: years from settlement to each cash flow under the bond's day count
    - frequency: the yield's compounding (2 for the semi-annual bond-equivalent yield) or 'continuous'
    returns: a float, the yield as a decimal compounded `frequency` times a year
    source: Fabozzi, Bond Markets, Analysis and Strategies, ch. 3; Hull, ch. 4 ("Bond yield"); Brent (1973)
    aliases: [yield to maturity], [YTM], [bond yield], [internal rate of return], [IRR], [redemption yield], [yield from price]
    example:
        qf.bond_yield(price=101.2, cash_flows=[2.5, 2.5, 2.5, 102.5],
                      times=[0.5, 1.0, 1.5, 2.0],
                      frequency=2)
    """
    cf, t = _flows(cash_flows, times)
    target = float(price)
    low, high = (-1.0, 100.0) if frequency == "continuous" else (-0.99 * _whole_frequency(frequency), 100.0)

    def gap(rate):
        return cf @ _yield_discounts(t, rate, frequency) - target

    if gap(low) * gap(high) > 0:
        raise ValueError("price: no yield in the searched range discounts the cash flows to this price")
    return float(brentq(gap, low, high, xtol=1e-14, rtol=4.5e-15, maxiter=500))


@_guard(rate=('ytm',), time=('times',))
def bond_risk(cash_flows, times, ytm, frequency):
    """Macaulay and modified duration, convexity and DV01 of a bond at its yield to maturity.

    P = sum_i CF_i d_i, d_i = (1 + y/f)^{-f t_i}; Macaulay D = sum_i t_i CF_i d_i / P; modified D* = D / (1 + y/f);
    convexity C = (1/P) d^2P/dy^2 = sum_i CF_i t_i (t_i + 1/f) (1 + y/f)^{-f t_i - 2} / P; DV01 = D* P / 10,000.
    For 'continuous', d_i = e^{-y t_i}, D* = D and C = sum_i t_i^2 CF_i d_i / P.

    A portfolio's duration is the value-weighted mean of its bonds' durations, sum_k V_k D_k / sum_k V_k (exact for
    one parallel shift of a common curve, the usual approximation with each bond at its own yield); convexity alike,
    and DV01s add.

    conventions:
    - durations in years; convexity in years squared, as (1/P) d^2P/dy^2 (some texts halve it or quote it per 100)
    - dv01: the price change, positive, for a one basis point fall in the yield, per the cash flows' face
    - frequency: the yield's compounding, a whole number a year or 'continuous'; times as bond_price
    returns: a dict: price; macaulay_duration and modified_duration (years); convexity (years squared); dv01
      (the price change for 1 bp of yield, in the price's units)
    source: Macaulay (1938); Fabozzi, Bond Markets, Analysis and Strategies, ch. 4; Hull, "Duration" and "Convexity"
    aliases: duration, [Macaulay duration], [modified duration], convexity, [DV01], [PV01], [dollar duration], [price value of a bp]
    example:
        qf.bond_risk(cash_flows=[2.5, 2.5, 2.5, 102.5],
                     times=[0.5, 1.0, 1.5, 2.0], ytm=0.045,
                     frequency=2)
    """
    cf, t = _flows(cash_flows, times)
    y = float(ytm)
    d = _yield_discounts(t, y, frequency)
    price = float(cf @ d)
    if not price > 0:
        raise ValueError("cash_flows must have a positive value at this yield")
    macaulay = float(t @ (cf * d)) / price
    if frequency == "continuous":
        modified, convexity = macaulay, float((t * t) @ (cf * d)) / price
    else:
        f = _whole_frequency(frequency)
        modified = macaulay / (1.0 + y / f)
        convexity = float((t * (t + 1.0 / f)) @ (cf * d)) / (1.0 + y / f) ** 2 / price
    return Result({"price": price, "macaulay_duration": macaulay, "modified_duration": modified, "convexity": convexity,
                   "dv01": modified * price * 1e-4})


@_guard(rate=('curve_zero_rates',), time=('times', 'curve_times', 'key_times',),
        curve=(('curve_times', 'curve_zero_rates'),))
def key_rate_durations(cash_flows, times, curve_times, curve_zero_rates, key_times, shift, bump, difference):
    """Key-rate durations: the price's sensitivity to a bump of the zero curve at each key tenor.

    PV = sum_i CF_i e^{-z(t_i) t_i}, z linear in t through the curve's nodes, flat beyond the ends. Key k's bump adds
    bump * s_k(node) to each node's zero rate: 'node' bumps the key's node alone (s_k = 1 there, else 0);
    'tent' is Ho's (1992) shift, 1 at key k falling linearly to 0 at the neighbouring keys and flat beyond the end
    keys, so the tents sum to a parallel shift. KRD_k = -(PV_up - PV) / (bump PV) ('forward') or -(PV_up - PV_down) /
    (2 bump PV) ('central').

    conventions:
    - curve_zero_rates: continuously compounded, at curve_times (annual rates, -ln(D)/t, not rates times their
      times); key_times: a subset of curve_times
    - shift: 'node' or 'tent' (Ho's), as above; bump: the shift in rate units (0.0001 is 1 bp); difference: 'forward'
      (bump up only) or 'central' (up and down)
    returns: an array of durations, one per key; times PV the dollar KRD, times PV / 10,000 the KR01
    source: Ho (1992), "Key rate durations: measures of interest rate risks", J. Fixed Income 2(2); Tuckman, ch. 5
    aliases: key rate duration, KRD, key rate durations, partial durations, bucketed duration, key-rate risk, KR01
    example:
        qf.key_rate_durations(cash_flows=[5.0, 5.0, 105.0], times=[1.0, 2.0, 3.0], curve_times=[1.0, 2.0, 3.0],
                              curve_zero_rates=[0.04, 0.042, 0.045], key_times=[2.0, 3.0], shift="node", bump=0.0001,
                              difference="central")
    """
    choice("shift", shift, ("node", "tent"))
    choice("difference", difference, ("forward", "central"))
    cf, t = _flows(cash_flows, times)
    nodes = _increasing("curve_times", curve_times)
    zeros = np.asarray(curve_zero_rates, dtype=float).ravel()
    if zeros.shape != nodes.shape or not np.all(np.isfinite(zeros)):
        raise ValueError("curve_zero_rates must hold one finite rate per curve time")
    keys = _increasing("key_times", key_times)
    if not np.all(np.isin(keys, nodes)):
        raise ValueError("key_times must be among curve_times")
    size = float(positive("bump", bump))

    def value(curve):
        return float(cf @ np.exp(-np.interp(t, nodes, curve) * t))

    base = value(zeros)
    if not base > 0:
        raise ValueError("cash_flows must have a positive value on the curve")
    durations = np.empty(keys.size)
    for k, key in enumerate(keys):
        s = (nodes == key).astype(float) if shift == "node" else np.interp(nodes, keys, np.eye(keys.size)[k])
        up = value(zeros + size * s)
        down = base if difference == "forward" else value(zeros - size * s)
        durations[k] = -(up - down) / ((1.0 if difference == "forward" else 2.0) * size * base)
    return durations


def _add_months(anchor, months, roll):
    """`anchor` moved by whole months: its day of the month kept, cut to the month's length ('same_day'), or each
    date its month's last day where the anchor is one ('end_of_month')."""
    one_month, one_day = np.timedelta64(1, "M"), np.timedelta64(1, "D")
    month = anchor.astype("datetime64[M]") + np.asarray(months).astype("timedelta64[M]")
    last = (month + one_month).astype("datetime64[D]") - one_day
    day = (anchor - anchor.astype("datetime64[M]").astype("datetime64[D]")).astype(int)
    on_last = anchor == (anchor.astype("datetime64[M]") + one_month).astype("datetime64[D]") - one_day
    if roll == "end_of_month" and on_last:
        return last
    return np.minimum(month.astype("datetime64[D]") + np.timedelta64(day, "D"), last)


def _schedule(issue, maturity, months, roll, stub):
    """The period ends from issue to maturity, every `months` months from maturity back (front stubs) or from issue
    on (back stubs), with each period's notional regular start and end (ICMA's reference periods)."""
    back = stub.endswith("front")
    anchor = maturity if back else issue
    k = np.arange(0, 12 * 2000 // months)
    grid = _add_months(anchor, -k * months if back else k * months, roll)
    inside = grid[(grid > issue) & (grid < maturity)]
    dates = np.unique(np.r_[inside, maturity] if back else np.r_[issue, inside])
    if back:
        regular = np.r_[_add_months(anchor, -(len(dates)) * months, roll), dates]
        if stub == "long_front" and regular[0] < issue < regular[1] and len(dates) > 1:
            dates = dates[1:]
        starts = np.r_[issue, dates[:-1]]
        return starts, dates, regular
    regular = np.r_[dates, _add_months(anchor, len(dates) * months, roll)]
    if stub == "long_back" and regular[-2] < maturity < regular[-1] and len(dates) > 1:
        dates = dates[:-1]
    return dates, np.r_[dates[1:], maturity], regular


def _icma(start, end, regular, frequency):
    """ICMA's accrual of [start, end]: in each regular period [a, b] it touches, days in both over (f days(a, b))."""
    total = 0.0
    for a, b in zip(regular[:-1], regular[1:]):
        lo, hi = max(start, a), min(end, b)
        if hi > lo:
            total += (hi - lo).astype(int) / (frequency * (b - a).astype(int))
    return total


@_guard(rate=('coupon_rate',))
def bond_cash_flows(issue, settlement, maturity, coupon_rate, frequency, face, day_count, roll, stub):
    """A fixed-coupon bond's cash-flow schedule: dates, year fractions, amounts, accrued.

    Coupon dates step 12/f months from maturity back to the issue (front stubs) or from the issue on (back stubs);
    where the steps do not meet the other end, the odd period is a short stub, or merged with its neighbour into a
    long one. A period's coupon is face c times its accrual: under 'ACT/ACT ICMA' 1/f for a regular period and, for a
    stub, its days over f times its notional regular period's (QuantLib's ActualActual ISMA); under any other day
    count the year fraction of its two dates (qf.year_fraction). Accrued interest at settlement is face
    c times the accrual from the current period's start; the flows are those paid after settlement, with their year
    fractions from settlement under the same day count (ICMA: the accrual).

    The standard CDS schedule (premiums on the 20th of March, June, September and December, ACT/360 accruals) is this
    function with frequency=4, day_count='ACT/360', roll='same_day', stub='short_front', the issue the last IMM 20th
    on or before the trade (the accrual start) and the maturity an IMM 20th; the dates are not moved off weekends.

    conventions:
    - issue, settlement, maturity: dates (ISO 'YYYY-MM-DD', datetime.date or datetime64), issue <= settlement < maturity
    - coupon_rate: annual, as a decimal; frequency: coupons a year (1, 2, 4, 12); face: the principal repaid
    - day_count: 'ACT/ACT ICMA', 'ACT/360', 'ACT/365F', 'ACT/ACT ISDA', '30/360' or '30E/360'
    - roll: 'same_day' (the anchor's day of the month, cut to short months) or 'end_of_month' (month ends stay)
    - stub: 'short_front', 'long_front', 'short_back' or 'long_back'; no business-day adjustment
    returns: a dict: dates (ISO strings), times (years from settlement), amounts (coupon, plus face at maturity),
      coupons, accrued (at settlement) and period_start (the current period's start)
    source: ICMA Rule 251 (ACT/ACT ICMA); Fabozzi, Bond Markets, Analysis and Strategies, ch. 2 and 3
    aliases: bond cash flows, coupon schedule, bond schedule, [cash flow schedule], accrued interest, stub period
    example:
        qf.bond_cash_flows(issue="2024-03-15", settlement="2025-01-10", maturity="2027-03-15", coupon_rate=0.06,
                           frequency=2, face=100.0, day_count="ACT/ACT ICMA", roll="same_day",
                           stub="short_front")
    """
    choice("day_count", day_count, BOND_DAY_COUNTS)
    choice("roll", roll, ROLLS)
    choice("stub", stub, STUBS)
    f = _whole_frequency(frequency)
    if 12 % f:
        raise ValueError("frequency must divide 12 (1, 2, 3, 4, 6 or 12 coupons a year)")
    first, settle, last = (_dates(name, value).reshape(()) for name, value in
                           (("issue", issue), ("settlement", settlement), ("maturity", maturity)))
    if not first <= settle < last:
        raise ValueError("the dates must satisfy issue <= settlement < maturity")
    rate, principal = float(coupon_rate), float(positive("face", face))
    starts, ends, regular = _schedule(first, last, 12 // f, roll, stub)

    def accrual(start, end):
        if day_count == ICMA:
            return _icma(start, end, regular, f)
        return float(year_fraction(start, end, basis=day_count))

    coupons = np.array([principal * rate * accrual(a, b) for a, b in zip(starts, ends)])
    current = int(np.searchsorted(ends, settle, side="right"))
    paid = slice(current, None)
    times = np.array([accrual(settle, end) for end in ends[paid]])
    amounts = coupons[paid].copy()
    amounts[-1] += principal
    return Result({"dates": [str(date) for date in ends[paid]], "times": times, "amounts": amounts,
                   "coupons": coupons[paid], "accrued": float(principal * rate * accrual(starts[current], settle)),
                   "period_start": str(starts[current])})


def _zeros_at(t, curve_times, curve_zero_rates, interpolation, compounding):
    """The zero rates at times t, in the curve's compounding: linear in the rate between the nodes and flat beyond
    them ('linear_zero'), or from -ln D linear in time between (0, 0) and the nodes ('log_linear_discount', none past
    the last node)."""
    choice("interpolation", interpolation, ZERO_INTERPOLATIONS)
    nodes = _increasing("curve_times", curve_times)
    zeros = np.asarray(curve_zero_rates, dtype=float).ravel()
    if zeros.shape != nodes.shape or not np.all(np.isfinite(zeros)):
        raise ValueError("curve_zero_rates must hold one finite rate per curve time")
    if interpolation == "linear_zero":
        return np.interp(t, nodes, zeros)
    if np.any(t > nodes[-1] * (1 + 1e-12)):
        raise ValueError("times must not pass the last curve time under 'log_linear_discount' (no extrapolation)")
    log_d = np.interp(t, np.r_[0.0, nodes], np.r_[0.0, np.log(_zero_discounts(zeros, nodes, compounding)[0])])
    with np.errstate(divide="ignore", invalid="ignore"):
        if compounding == "continuous":
            found = -log_d / t
        elif compounding == "simple":
            found = np.expm1(-log_d) / t
        else:
            found = compounding * np.expm1(-log_d / (compounding * t))
    return np.where(t > 0, found, np.interp(0.0, nodes, zeros))


@_guard(rate=('curve_zero_rates',), time=('times', 'curve_times',), curve=(('curve_times', 'curve_zero_rates'),))
def curve_risk(cash_flows, times, curve_times, curve_zero_rates, interpolation, compounding):
    """Price, Fisher-Weil duration, convexity and DV01 of cash flows on a zero curve, compounding named.

    PV = sum_i CF_i D(t_i), D from the zero rate z(t_i) the curve gives at t_i under the named compounding (e^{-z t},
    (1 + z/k)^{-k t} or 1 / (1 + z t)). Under a parallel shift s of the zero rates: modified duration -PV'(s) / PV,
    convexity PV''(s) / PV, DV01 -PV'(s) / 10,000; the Fisher-Weil duration sum_i t_i CF_i D(t_i) / PV, which is the
    modified duration when the rates are continuous. On a flat curve these are qf.bond_risk's at that yield.

    conventions:
    - curve_times, curve_zero_rates: the curve's nodes, years and zero rates in the named compounding: annual rates,
      -ln(D)/t, not rates times their times
    - interpolation: 'linear_zero' (linear in the zero rate, flat beyond the end nodes) or 'log_linear_discount'
      (-ln D linear in time from D(0) = 1, piecewise-flat forwards; none past the last node)
    - compounding: 'continuous', 'simple' or periods a year (1 annual, 2 semi-annual) of the zero rates and the shift
    - times: years from today to each cash flow under the caller's day count
    returns: a dict: price; fisher_weil_duration and modified_duration (years); convexity (years squared); dv01 (the
      PV's fall for 1 bp up, in the cash flows' units); discount_factors and zero_rates at the times
    source: Fisher and Weil (1971), Journal of Business 44(4); Fabozzi, Bond Markets, Analysis and Strategies, ch. 4
    aliases: Fisher-Weil duration, Fisher-Weil, zero curve duration, curve duration, PV off a zero curve, [DV01]
    example:
        qf.curve_risk(cash_flows=[5.0, 5.0, 105.0], times=[1.0, 2.0, 3.0], curve_times=[0.5, 2.0, 5.0],
                      curve_zero_rates=[0.03, 0.035, 0.04], interpolation="linear_zero",
                      compounding="continuous")
    """
    rule = _zero_compounding(compounding)
    cf, t = _flows(cash_flows, times)
    z = _zeros_at(t, curve_times, curve_zero_rates, interpolation, rule)
    d, first, second = _zero_discounts(z, t, rule)
    price = float(cf @ d)
    if not price > 0:
        raise ValueError("cash_flows must have a positive value on the curve")
    slope = float(cf @ first)
    return Result({"price": price, "fisher_weil_duration": float((t * cf) @ d) / price,
                   "modified_duration": -slope / price, "convexity": float(cf @ second) / price,
                   "dv01": -slope * 1e-4, "discount_factors": d, "zero_rates": z})


@_guard(rate=('curve_zero_rates',), time=('times', 'curve_times',), price=('price',),
        curve=(('curve_times', 'curve_zero_rates'),))
def z_spread(price, cash_flows, times, curve_times, curve_zero_rates, interpolation, compounding):
    """Z-spread: the constant spread over a zero curve that reprices a bond's cash flows.

    The s with sum_i CF_i D(z(t_i) + s, t_i) = price: each zero rate the curve gives at a cash-flow time, in the named
    compounding, plus s, discounted as qf.curve_risk discounts (Brent's method, to 1e-14). On a flat curve with
    continuous rates it is the yield less the curve's rate; a bond priced off the curve itself has s = 0.

    conventions:
    - price: the dirty (full) price, in the cash flows' units (clean price plus accrued interest)
    - curve_times, curve_zero_rates, interpolation, compounding: the curve as qf.curve_risk takes it (annual rates,
      -ln(D)/t, not rates times their times); the spread is in the same compounding
    - times: years from settlement to each cash flow under the caller's day count
    returns: the spread as a decimal (0.0125 is 125 bp), a float
    source: Fabozzi, Bond Markets, Analysis and Strategies, ch. 18 (static spread); Tuckman and Serrat (2011), ch. 3
    aliases: z-spread, zero-volatility spread, static spread, spread over the zero curve, Z spread
    example:
        qf.z_spread(price=98.5, cash_flows=[5.0, 5.0, 105.0], times=[1.0, 2.0, 3.0], curve_times=[0.5, 2.0, 5.0],
                    curve_zero_rates=[0.03, 0.035, 0.04], interpolation="linear_zero",
                    compounding="continuous")
    """
    rule = _zero_compounding(compounding)
    cf, t = _flows(cash_flows, times)
    z = _zeros_at(t, curve_times, curve_zero_rates, interpolation, rule)
    target = float(positive("price", price))
    if np.any(cf < 0):
        raise ValueError("cash_flows must not be negative (a bond's coupons and principal)")
    floor = -np.min(z) - (rule if isinstance(rule, int) else 1.0 / np.max(t) if rule == "simple" else 1.0)

    def gap(spread):
        return float(cf @ _zero_discounts(z + spread, t, rule)[0]) - target

    low, high = max(-0.05, floor * 0.999), 0.05
    while gap(low) < 0 and low > floor * 0.999 + 1e-12:
        low = max(2 * low, floor * 0.999)
    while gap(high) > 0 and high < 100.0:
        high *= 2
    if gap(low) * gap(high) > 0:
        raise ValueError("price: no spread in the searched range reprices the cash flows")
    return float(brentq(gap, low, high, xtol=1e-14, rtol=1e-15, maxiter=500))


def immunization_weights(exposures, liability_value, liability_exposures):
    """Immunization: amounts in n instruments matching a liability's value and n - 1 exposures.

    Solves sum_j a_j = L and sum_j a_j e_kj = L e_kL for each exposure k: a square linear system, with exposures per
    unit of value (durations, convexities or key-rate durations of each instrument and of the liability). Two bonds
    on duration match value and duration; a third on convexity adds Redington's second condition. A negative amount
    is a short position; a singular system (two instruments alike) is refused.

    conventions:
    - exposures: (k, n) per unit of value, row k one measure across the n = k + 1 instruments (a 1-d row for k = 1)
    - liability_value: L, the present value to match; liability_exposures: (k,) the same measures for the liability
    - returns amounts of value (dollars), each divided by its price for units; any duration, if used throughout
    returns: an array of amounts of value, one per instrument (negative is short)
    source: Redington (1952), J. Institute of Actuaries 78; Luenberger, Investment Science (1998), ch. 3
    aliases: immunization, immunisation, duration matching, Redington immunization, [liability hedge], [cash-flow hedge]
    example:
        qf.immunization_weights(exposures=[[2.0, 10.0]],
                                liability_value=1000000.0,
                                liability_exposures=[6.0])
    """
    e = np.atleast_2d(np.asarray(exposures, dtype=float))
    targets = np.atleast_1d(np.asarray(liability_exposures, dtype=float)).ravel()
    k, n = e.shape
    if n != k + 1 or targets.size != k:
        raise ValueError(f"exposures must be (k, k + 1) with k liability_exposures; got {e.shape} and {targets.size}")
    system = np.vstack([np.ones(n), e])
    if not np.all(np.isfinite(system)) or np.linalg.cond(system) > 1e12:
        raise ValueError("exposures are singular: no unique immunizing amounts")
    return np.linalg.solve(system, float(liability_value) * np.r_[1.0, targets])


# ------------------------------------------------------------------------------------------------ floaters, LP, MBS


@_guard(rate=('margin', 'last_fixing',), time=('payment_times', 'call_time',), price=('market_price',))
def frn_price(payment_times, accruals, margin, last_fixing, discount_curve, projection_curve, interpolation, face,
              call_time, market_price):
    """Floating-rate note (floater) price off projected forwards plus a quoted margin, and its discount margin.

    Coupon i pays face (L_i + m) tau_i at T_i: L_i the forward (P(T_{i-1}) / P(T_i) - 1) / tau_i projected off the
    projection curve (last_fixing for the current coupon if known), m the simple quoted margin; face paid at the last
    payment or call_time. Price = sum of flows times D(T_i) off the
    discount curve (dirty). The discount margin dm solves the market price = sum of flows times prod_j 1 / (1 + (L_j +
    dm) s_j), s_1 = T_1, s_j = tau_j after. At a reset on one curve with zero margin it prices at par; at par the
    discount margin is the quoted margin; between resets it is the known coupon and par at the next reset.

    conventions:
    - payment_times: the coupon dates after today in years, increasing; accruals: each period's tau_i (the index's
      day count, ACT/360 for SOFR)
    - margin: the quoted margin over the index, a decimal (0.005 is 50 bp); face: the principal
    - last_fixing: the current period's index rate where already fixed (a decimal), or None where the first period
      starts today and is projected
    - discount_curve: (node_times, node_discounts); projection_curve: the same for the index, or None (one curve);
      interpolation: 'linear_zero', 'log_linear_discount' or 'cubic_log_discount', as qf.curve_discount reads them
    - call_time: None, or a payment time at which the note is redeemed at par (the price to that call)
    - market_price: None, or the dirty price whose discount margin is solved
    returns: a dict: price (dirty, in the face's units); coupons (the projected amounts); forwards (L_i); with a
      market_price discount_margin
    source: Fabozzi, Bond Markets, Analysis and Strategies, ch. 2 and 3 (floaters, discount margin); Tuckman and
      Serrat, Fixed Income Securities (2012), ch. 15
    aliases: floating-rate note, FRN, floater, discount margin, quoted margin, FRN price, floating rate bond, callable floater
    example:
        # price = 100.9499468025 in the face's units.
        qf.frn_price(payment_times=[0.5, 1.0, 1.5, 2.0], accruals=[0.5, 0.5, 0.5, 0.5], margin=0.005,
                     last_fixing=None, discount_curve=([0.5, 1.0, 2.0], [0.98, 0.96, 0.92]), projection_curve=None,
                     interpolation="log_linear_discount", face=100.0, call_time=None, market_price=None)
    """
    choice("interpolation", interpolation, INTERPOLATIONS)
    times = _increasing("payment_times", payment_times)
    tau = positive("accruals", accruals).ravel()
    if tau.shape != times.shape:
        raise ValueError("accruals must hold one period per payment time")
    principal, m = float(positive("face", face)), float(margin)
    discount = _curve("discount_curve", discount_curve, interpolation)
    project = discount if projection_curve is None else _curve("projection_curve", projection_curve, interpolation)
    if call_time is not None:
        on = np.flatnonzero(np.isclose(times, float(call_time), rtol=0, atol=1e-9))
        if on.size != 1:
            raise ValueError("call_time must be one of the payment times")
        times, tau = times[:on[0] + 1], tau[:on[0] + 1]
    levels = project(np.r_[0.0, times])
    forwards = (levels[:-1] / levels[1:] - 1.0) / tau
    if last_fixing is not None:
        forwards[0] = float(last_fixing)
    coupons = principal * (forwards + m) * tau
    flows = coupons.copy()
    flows[-1] += principal
    result = Result({"price": float(flows @ discount(times)), "coupons": coupons, "forwards": forwards})
    if market_price is not None:
        target, steps = float(positive("market_price", market_price)), np.r_[times[0], tau[1:]]

        def gap(dm):
            return float(flows @ np.cumprod(1.0 / (1.0 + (forwards + dm) * steps))) - target

        low, high = -float(np.min(forwards + 1.0 / steps)) * 0.999 + 1e-9, 1.0
        low = max(low, -1.0)
        while gap(high) > 0 and high < 64.0:
            high *= 2.0
        if gap(low) * gap(high) > 0:
            raise ValueError("market_price: no discount margin in the searched range reprices the note")
        result["discount_margin"] = float(brentq(gap, low, high, xtol=1e-15, rtol=4.5e-15, maxiter=500))
    return result


MATCHINGS = ("exposures", "cash_flows")
LP_OBJECTIVES = ("cost", "mismatch")


def _boxes(bounds, n):
    """bounds as linprog takes them: one (low, high) for every instrument, or one pair each; None is unbounded."""
    pairs = [tuple(bounds)] * n if len(bounds) == 2 and all(b is None or np.ndim(b) == 0 for b in bounds) else \
        [tuple(pair) for pair in bounds]
    if len(pairs) != n or any(len(pair) != 2 for pair in pairs):
        raise ValueError("bounds must be one (low, high) pair, or one pair per instrument (None unbounded)")
    return [(None if low is None else float(low), None if high is None else float(high)) for low, high in pairs]


@_guard(rate=('reinvestment_rate',), time=('times',), price=('prices',))
def immunize_lp(matching, objective, prices, bounds, exposures, liability_exposures, cash_flows, liabilities, times,
                reinvestment_rate):
    """Cash-flow matching (dedication) or long-only immunization by a linear program: the cheapest bonds.

    Holdings x_j (units of each bond, priced p_j) by scipy.optimize.linprog ('highs'). 'cash_flows' (dedication): the
    cheapest sum p_j x_j whose flows cover each liability date by date, any surplus carried to the next date at the
    reinvestment rate: s_k = s_{k-1} (1 + r)^{t_k - t_{k-1}} + sum_j c_jk x_j - L_k >= 0. 'exposures': each row e_k of
    the bonds' exposures per unit (price for the value, price times duration, dollar convexity, key-rate dollar
    durations) matched to the liability's, sum_j e_kj x_j = l_k; 'cost' the cheapest x (a price row fixes cost), 'mismatch' the first row
    (the value) matched and the other rows' absolute misses summed least (rows in one unit). Bounds (0, None): long
    only (immunization_weights solves the square system, shorts allowed). Value and duration matched by bonds either
    side of a single-date liability (a barbell of zeros) meet Redington's convexity condition too.

    conventions:
    - matching: 'cash_flows' or 'exposures'; objective: 'cost' or 'mismatch' ('exposures' only)
    - prices: p_j per unit (dirty); bounds: (low, high) units for all or one pair per bond, (0, None) long-only
    - exposures: (rows, bonds) per unit, liability_exposures: one per row, both None for 'cash_flows'
    - cash_flows: (bonds, dates) per unit; liabilities: the amount due at each date; times: the dates in years;
      reinvestment_rate: r a year, annual compounding (0.0 for none); all None for 'exposures'
    returns: a dict: units (x); amounts (p_j x_j); cost; with 'exposures' matched (the portfolio's rows); with
      'cash_flows' surplus (after each date)
    source: Luenberger, Investment Science (2013), ch. 3 and 4 (cash matching, immunization); Redington (1952)
    aliases: cash flow matching, dedication, long-only immunization, immunization linear program, key rate duration matching, [liability matching]
    example:
        # cost = 1000.0; matched value and dollar duration = [1000.0, 5000.0].
        qf.immunize_lp(matching="exposures", objective="cost", prices=[98.0, 95.0, 90.0], bounds=(0.0, None),
                       exposures=[[98.0, 95.0, 90.0], [196.0, 475.0, 900.0]], liability_exposures=[1000.0, 5000.0],
                       cash_flows=None, liabilities=None, times=None, reinvestment_rate=None)
    """
    from scipy.optimize import linprog
    choice("matching", matching, MATCHINGS)
    choice("objective", objective, LP_OBJECTIVES)
    cost = positive("prices", prices).ravel()
    n = cost.size
    box = _boxes(bounds, n)
    if matching == "exposures":
        if any(v is not None for v in (cash_flows, liabilities, times, reinvestment_rate)):
            raise ValueError("cash_flows, liabilities, times and reinvestment_rate are None for 'exposures'")
        rows = np.atleast_2d(np.asarray(exposures, dtype=float))
        target = np.atleast_1d(np.asarray(liability_exposures, dtype=float)).ravel()
        if rows.shape != (target.size, n) or not (np.all(np.isfinite(rows)) and np.all(np.isfinite(target))):
            raise ValueError("exposures must be (rows, bonds) with one liability exposure per row")
        if objective == "cost":
            found = linprog(cost, A_eq=rows, b_eq=target, bounds=box, method="highs")
        else:
            k = target.size - 1
            if k < 1:
                raise ValueError("'mismatch' needs a value row and at least one more")
            gaps = np.zeros((k + 1, 2 * k))
            gaps[1:, :k], gaps[1:, k:] = -np.eye(k), np.eye(k)
            found = linprog(np.r_[np.zeros(n), np.ones(2 * k)], A_eq=np.hstack([rows, gaps]), b_eq=target,
                            bounds=box + [(0, None)] * (2 * k), method="highs")
        if found.status != 0:
            raise ValueError(f"no portfolio within the bounds matches the exposures ({found.message})")
        units = found.x[:n]
        return Result({"units": units, "amounts": units * cost, "cost": float(cost @ units), "matched": rows @ units})
    if objective != "cost":
        raise ValueError("'cash_flows' takes objective 'cost'")
    if exposures is not None or liability_exposures is not None:
        raise ValueError("exposures and liability_exposures are None for 'cash_flows'")
    flows = np.atleast_2d(np.asarray(cash_flows, dtype=float))
    owed = np.asarray(liabilities, dtype=float).ravel()
    dates = np.asarray(times, dtype=float).ravel()
    if flows.shape != (n, owed.size) or dates.shape != owed.shape or np.any(dates < 0) or np.any(np.diff(dates) <= 0):
        raise ValueError("cash_flows must be (bonds, dates), with one liability and one increasing time per date")
    rate = float(reinvestment_rate)
    if not rate > -1:
        raise ValueError("reinvestment_rate must exceed -1")
    m = owed.size
    carry = np.zeros((m, m))
    carry[np.arange(m), np.arange(m)] = 1.0
    carry[np.arange(1, m), np.arange(m - 1)] = -(1.0 + rate) ** np.diff(dates)
    found = linprog(np.r_[cost, np.zeros(m)], A_eq=np.hstack([-flows.T, carry]), b_eq=-owed,
                    bounds=box + [(0, None)] * m, method="highs")
    if found.status != 0:
        raise ValueError(f"no portfolio within the bounds covers the liabilities ({found.message})")
    units = found.x[:n]
    return Result({"units": units, "amounts": units * cost, "cost": float(cost @ units), "surplus": found.x[n:]})


PREPAYMENTS = ("cpr", "psa")


@_guard(rate=('mortgage_rate', 'pass_through_rate',))
def mortgage_cash_flows(balance, mortgage_rate, pass_through_rate, months, prepayment, speed, age):
    """Mortgage pass-through (MBS) cash flows with prepayment at a CPR or a PSA speed: interest and principal.

    Each month on the balance B: the level payment B c / (1 - (1 + c)^{-N}) over the N months left, c the mortgage
    rate (WAC) / 12; scheduled principal = payment - B c; prepayment = SMM (B - scheduled principal), SMM = 1 - (1 -
    CPR)^{1/12}; the investor gets B times the pass-through rate / 12 (the servicing fee is the gap), the scheduled
    principal and the prepayment. PSA: CPR = 6% min(1, age / 30) times the speed, age the loan's month (age + t). With
    no prepayment the balance amortises to zero; the average life is sum_t t P_t / sum_t P_t / 12, P_t the principal.

    conventions:
    - balance: the pool's balance today; mortgage_rate: the WAC, annual decimal; pass_through_rate: the net coupon
    - months: the months left (WAM); age: the months already seasoned (WAM 357 of 360 is age 3)
    - prepayment: 'cpr' (speed: the annual CPR, a decimal, 0.06 is 6%) or 'psa' (speed: the PSA multiple, 1.0 is 100%
      PSA, 1.5 is 150%)
    returns: a dict: month; balance (at the month's start); smm; payment (the scheduled mortgage payment); interest
      (to investors); scheduled_principal; prepayment; principal; cash_flow (to investors); average_life (years)
    source: Fabozzi, Bond Markets, Analysis and Strategies, ch. 10 (pass-through cash flows at 100% PSA); Public
      Securities Association (1985), the PSA standard prepayment model
    aliases: mortgage pass-through, MBS cash flows, PSA, PSA prepayment, CPR, SMM, conditional prepayment rate, prepayment model
    example:
        # first cash flow = 5745.49215498; average life = 11.3634768746 years.
        qf.mortgage_cash_flows(balance=1000000.0, mortgage_rate=0.06, pass_through_rate=0.055, months=360,
                               prepayment="psa", speed=1.0, age=0)
    """
    choice("prepayment", prepayment, PREPAYMENTS)
    left, seasoned = integer("months", months, 1), integer("age", age, 0)
    owed, c, net = float(positive("balance", balance)), float(mortgage_rate) / 12.0, float(pass_through_rate) / 12.0
    pace = float(speed)
    if not (c >= 0 and net >= 0 and pace >= 0):
        raise ValueError("mortgage_rate, pass_through_rate and speed must not be negative")
    rows = {key: np.zeros(left) for key in ("balance", "smm", "payment", "interest", "scheduled_principal",
                                            "prepayment")}
    for t in range(left):
        cpr = pace if prepayment == "cpr" else pace * 0.06 * min(1.0, (seasoned + t + 1) / 30.0)
        if not 0 <= cpr < 1:
            raise ValueError("the CPR must lie in [0, 1): lower the speed")
        smm = -np.expm1(np.log1p(-cpr) / 12.0)
        rest = left - t
        payment = owed * c / -np.expm1(-rest * np.log1p(c)) if c > 0 else owed / rest
        scheduled = min(payment - owed * c, owed)
        prepaid = smm * (owed - scheduled)
        for key, value in (("balance", owed), ("smm", smm), ("payment", payment), ("interest", owed * net),
                           ("scheduled_principal", scheduled), ("prepayment", prepaid)):
            rows[key][t] = value
        owed -= scheduled + prepaid
    principal = rows["scheduled_principal"] + rows["prepayment"]
    month = np.arange(1, left + 1)
    return Result({"month": month, **rows, "principal": principal, "cash_flow": rows["interest"] + principal,
                   "average_life": float(month @ principal / principal.sum() / 12.0)})


CF_ROUNDINGS = ("quarter", "month")


@_guard(rate=('coupon_rates',), price=('futures_price', 'clean_prices',))
def bond_futures_ctd(coupon_rates, maturities, delivery_month, rounding, factor_decimals, futures_price,
                     clean_prices):
    """Treasury bond futures conversion factors (the 6% rule) and the cheapest-to-deliver bond.

    The conversion factor is the price per unit face at a 6% yield (3% a half-year) on the delivery month's first day,
    the time to maturity cut to whole months: n whole years, z further months (cut to whole quarters for the bond and
    10-year contracts). CME's formula: CF = a (c/2 + C + d) - b, a = 1/1.03^{v/6}, b = c/2 (6 - v)/6, C = 1/1.03^{2n}
    (z < 7) or 1/1.03^{2n+1}, d = (c/0.06)(1 - C), v = z (z < 7), else 3 ('quarter') or z - 6 ('month'). The cost of
    delivering bond i is its quoted price less the futures price times its factor; the cheapest to deliver has the
    least. The invoice is futures price times factor plus accrued interest.

    conventions:
    - coupon_rates: annual decimals, rounded to the nearest 0.00125 (one-eighth percent), ties up; semi-annual coupons;
      maturities: each one's maturity (or first call) date, ISO 'YYYY-MM-DD'
    - delivery_month: the contract month, 'YYYY-MM'
    - rounding: 'quarter' (T-bond, Ultra, 10-year note futures) or 'month' (2-, 3- and 5-year note futures)
    - factor_decimals: 4 (the exchange's published factors) or None (unrounded)
    - futures_price, clean_prices: the futures quote and each bond's clean price per 100, or both None (factors only)
    returns: a dict: conversion_factors; with prices delivery_costs (quote - futures x factor) and cheapest (the
      index of the cheapest to deliver, from 0)
    source: CME Group (2024), "Calculating U.S. Treasury Futures Conversion Factors" (cmegroup.com/articles/2024/
      calculating-us-treasury-futures-conversion-factors.html); Hull, Options, Futures, and Other Derivatives
    aliases: conversion factors, cheapest to deliver, cheapest-to-deliver, CTD, Treasury futures, bond futures, T-bond futures, invoice price, delivery cost
    example:
        # conversion_factors = [0.8060, 0.8926, 1.0000]; delivery_costs = [6.340, 5.814, 5.000]; cheapest = 2.
        qf.bond_futures_ctd(coupon_rates=[0.04, 0.05, 0.06], maturities=["2040-02-15", "2042-11-15", "2045-05-15"],
                            delivery_month="2025-03", rounding="quarter", factor_decimals=4, futures_price=110.0,
                            clean_prices=[95.0, 104.0, 115.0])
    """
    choice("rounding", rounding, CF_ROUNDINGS)
    c = np.asarray(coupon_rates, dtype=float).ravel()
    ends = _dates("maturities", maturities).ravel()
    try:
        month = np.datetime64(str(delivery_month)[:7], "M")
    except ValueError:
        raise ValueError("delivery_month must be 'YYYY-MM'") from None
    if c.shape != ends.shape or np.any(~(c >= 0)) or np.any(ends <= month.astype("datetime64[D]")):
        raise ValueError("coupon_rates and maturities must pair up, coupons not negative, maturities after delivery")
    from decimal import Decimal, ROUND_HALF_UP
    step = Decimal("0.00125")
    c = np.asarray([float((Decimal(str(value)) / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step)
                    for value in c])
    whole = (ends.astype("datetime64[M]") - month).astype(int)
    n, z = whole // 12, whole % 12
    if rounding == "quarter":
        z = 3 * (z // 3)
    v = np.where(z < 7, z, 3 if rounding == "quarter" else z - 6)
    a, b = 1.03 ** (-v / 6.0), c / 2.0 * (6.0 - v) / 6.0
    big_c = 1.03 ** -np.where(z < 7, 2.0 * n, 2.0 * n + 1.0)
    factors = a * (c / 2.0 + big_c + c / 0.06 * (1.0 - big_c)) - b
    if factor_decimals is not None:
        places = integer("factor_decimals", factor_decimals, 0)
        factors = np.floor(factors * 10.0 ** places + 0.5) / 10.0 ** places
    result = Result({"conversion_factors": factors})
    if (futures_price is None) != (clean_prices is None):
        raise ValueError("give futures_price and clean_prices together, or neither")
    if futures_price is not None:
        quotes = np.asarray(clean_prices, dtype=float).ravel()
        if quotes.shape != factors.shape:
            raise ValueError("clean_prices must hold one price per bond")
        costs = quotes - float(futures_price) * factors
        result["delivery_costs"] = costs
        result["cheapest"] = int(np.argmin(costs))
    return result
