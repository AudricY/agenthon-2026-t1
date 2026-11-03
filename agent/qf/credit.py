"""Single-name credit: Merton distance to default, CDS legs and hazard bootstrap, CVA, defaultable bonds, convertibles.

`import qf`. Rates are continuously compounded and annual, times in years, vols annual. A hazard curve is
piecewise constant: rate j on (hazard_times[j-1], hazard_times[j]], the last beyond its end. A function with several
results returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from ._common import Result, choice, integer, ndtr, positive
from .curves import INTERPOLATIONS, curve_discount
from .primitives import _guard

__all__ = ["cds_bootstrap", "cds_price", "convertible_tree", "cva", "merton_calibrate", "risky_bond_price"]

BARRIERS = ("total", "kmv")
METHODS = ("simultaneous", "kmv_iteration")
SETTLED = 1e-12                 # the KMV iteration stops when the asset vol moves less than this
INTEGRATIONS = ("midpoint", "exact")
SIDES = ("buyer", "seller")
QUOTE_KINDS = ("par_spread", "upfront")
TIMINGS = ("end", "midpoint", "given_at_midpoints")
CVA_KINDS = ("unilateral", "bilateral", "first_to_default")
COLLATERAL_KEYS = ("threshold", "minimum_transfer", "lag")
RECOVERIES = ("face", "treasury", "market_value")
RECOVERY_TIMINGS = ("at_default", "midpoint", "period_end")
PRICE_BASES = ("clean", "dirty")
HAZARD_HIGH = 50.0              # the bootstrap's bracket for one node's hazard rate, per year: [0, this]
ON_GRID = 1e-9                  # a maturity within this of a payment time (relative) is that payment
SERIES = 0.5                    # below this, (1 - e^{-x}(1 + x)) / x^2 by its series (the closed form cancels)
SERIES_TERMS = tuple((-1) ** n * (n - 1) / math.factorial(n) for n in range(2, 22))   # x^0, x^1, ... of that series


def _call(v, sigma, debt, t, r):
    """Black-Scholes call on the assets struck at the debt, and its d1."""
    sd = sigma * np.sqrt(t)
    d1 = (np.log(v / debt) + (r + 0.5 * sigma * sigma) * t) / sd
    return v * ndtr(d1) - debt * np.exp(-r * t) * ndtr(d1 - sd), d1


def _assets(equity, sigma, debt, t, r):
    """The asset value whose call on the debt is worth the equity: between E and E + D e^{-rT}."""
    low, high = equity, equity + debt * np.exp(-r * t)
    return brentq(lambda v: _call(v, sigma, debt, t, r)[0] - equity, low * (1 - 1e-12), high * (1 + 1e-12),
                  xtol=1e-14, rtol=1e-15)


@_guard(rate=('rate',), vol=('equity_vol',), time=('horizon',), price=('equity',))
def merton_calibrate(equity, equity_vol, short_debt, long_debt, barrier, horizon, rate, drift, method,
                     periods_per_year, ddof):
    """Merton structural credit model: asset value and volatility from equity, distance to default and PD.

    Equity is a call on the assets V struck at the debt D: E = V N(d1) - D e^{-rT} N(d2), and Ito gives sigma_E E =
    N(d1) sigma_V V. 'simultaneous' solves the two for V and sigma_V from today's E and sigma_E; 'kmv_iteration'
    inverts every E_t of a series for V_t at a trial sigma_V, takes sigma_V from the V_t's log returns, and repeats
    until it settles (Crosbie and Bohn; Vassalou and Xing), the asset drift the V_t's mean log return plus sigma_V^2/2.
    Then DD = (ln(V/D) + (mu - sigma_V^2/2) T) / (sigma_V sqrt(T)) and PD = N(-DD). D is the barrier in both.

    conventions:
    - equity: E today ('simultaneous') or E_t oldest first, today last ('kmv_iteration'); equity_vol: sigma_E, or None
      for 'kmv_iteration'
    - short_debt, long_debt: book values; barrier: 'total' (D = short + long, Merton's face) or 'kmv' (D = short +
      long / 2, the KMV default point)
    - horizon: T; rate: r; drift: mu of the physical PD (pass rate for the risk-neutral N(-d2))
    - method: 'simultaneous' or 'kmv_iteration'; periods_per_year, ddof: the series' (None for 'simultaneous')
    returns: a dict: asset_value (V today); asset_vol (sigma_V); default_point (D); distance_to_default;
      default_probability; and from 'kmv_iteration' asset_values (the V_t), asset_drift, iterations
    source: Merton (1974), J. Finance 29(2); Crosbie and Bohn (2003), Moody's KMV; Hull, "Credit risk", Merton's model
    aliases: Merton structural model, Merton credit model, structural credit model, structural model, distance to default, KMV, Moody's KMV, Merton default probability, asset volatility from equity, Vassalou-Xing, [default probability]
    example:
        qf.merton_calibrate(equity=3.0, equity_vol=0.8, short_debt=10.0, long_debt=0.0, barrier="total",
                            horizon=1.0, rate=0.05, drift=0.05, method="simultaneous",
                            periods_per_year=None, ddof=None)
    """
    choice("barrier", barrier, BARRIERS)
    choice("method", method, METHODS)
    short, long = float(short_debt), float(long_debt)
    if not (short >= 0 and long >= 0 and short + long > 0):
        raise ValueError("short_debt and long_debt must not be negative, nor both zero")
    debt = short + (long if barrier == "total" else 0.5 * long)
    t, r, mu = float(positive("horizon", horizon)), float(rate), float(drift)
    if method == "simultaneous":
        e = float(positive("equity", equity))
        sigma_e = float(positive("equity_vol", equity_vol))
        if periods_per_year is not None or ddof is not None:
            raise ValueError("periods_per_year and ddof are for 'kmv_iteration': pass None")

        def gap(sigma):
            v = _assets(e, sigma, debt, t, r)
            return ndtr(_call(v, sigma, debt, t, r)[1]) * sigma * v - sigma_e * e

        sigma_v = brentq(gap, 1e-10, sigma_e, xtol=1e-15, rtol=1e-15)
        found = {"asset_value": _assets(e, sigma_v, debt, t, r), "asset_vol": float(sigma_v)}
    else:
        path = positive("equity", equity).ravel()
        if path.size < 3:
            raise ValueError("equity must be a series of at least 3 values for 'kmv_iteration'")
        if equity_vol is not None:
            raise ValueError("equity_vol is estimated from the series with 'kmv_iteration': pass None")
        scale = np.sqrt(float(positive("periods_per_year", periods_per_year)))
        spread = integer("ddof", ddof, 0)
        sigma_v, values = float(np.std(np.diff(np.log(path)), ddof=spread) * scale), path
        for iterations in range(1, 1001):
            values = np.array([_assets(e, sigma_v, debt, t, r) for e in path])
            moved, sigma_v = sigma_v, float(np.std(np.diff(np.log(values)), ddof=spread) * scale)
            if abs(sigma_v - moved) < SETTLED:
                break
        else:
            raise ValueError("kmv_iteration: the asset vol did not settle")
        steps = np.diff(np.log(values))
        found = {"asset_value": float(values[-1]), "asset_vol": sigma_v, "asset_values": values,
                 "asset_drift": float(steps.mean() * scale * scale + 0.5 * sigma_v * sigma_v),
                 "iterations": iterations}
    v, sigma_v = found["asset_value"], found["asset_vol"]
    dd = (np.log(v / debt) + (mu - 0.5 * sigma_v * sigma_v) * t) / (sigma_v * np.sqrt(t))
    found.update(default_point=debt, distance_to_default=float(dd), default_probability=float(ndtr(-dd)))
    return Result(found)


# ------------------------------------------------------------------------------------------------ CDS


def _hazard(hazard_times, hazard_rates):
    nodes = np.asarray(hazard_times, dtype=float).ravel()
    rates = np.asarray(hazard_rates, dtype=float).ravel()
    if nodes.size == 0 or rates.shape != nodes.shape or np.any(np.diff(nodes) <= 0) or nodes[0] <= 0:
        raise ValueError("hazard_times must be increasing positive times with one hazard rate each")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("hazard_rates must be finite and not negative")
    return nodes, rates


def _survival(t, nodes, rates):
    """S(t) = exp(-int_0^t lambda) on the piecewise-constant curve."""
    t = np.asarray(t, dtype=float)
    starts, ends = np.r_[0.0, nodes[:-1]], np.r_[nodes[:-1], np.inf]
    return np.exp(-(np.clip(t[..., None] - starts, 0.0, ends - starts) @ rates))


def _schedule(payment_times, accruals, accrued):
    times = np.asarray(payment_times, dtype=float).ravel()
    tau = positive("accruals", accruals).ravel()
    if times.size == 0 or tau.shape != times.shape or times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("payment_times must be increasing positive times with one accrual each")
    done = float(accrued)
    if not 0.0 <= done < tau[0]:
        raise ValueError("accrued must lie in [0, accruals[0]): the first period's part before today")
    return times, tau, done


def _flat_integrals(w, d):
    """(1 - e^{-wd}) / w and (1 - e^{-wd}(1 + wd)) / w^2, their limits at w = 0."""
    x = w * d
    first = np.where(x == 0, d, -np.expm1(-x) / np.where(x == 0, 1.0, w))
    small = np.abs(x) < SERIES
    safe = np.where(small, 1.0, x)
    series = np.zeros_like(x, dtype=float)
    for term in reversed(SERIES_TERMS):
        series = series * x + term
    second = d * d * np.where(small, series, (1 - np.exp(-safe) * (1 + safe)) / safe ** 2)
    return first, second


def _cds_legs(nodes, rates, recovery, times, tau, accrued, curve, integration, accrual_on_default):
    """(protection, the dirty RPV01, S at the payment times) per unit notional."""
    discount, knots = curve
    starts = np.r_[0.0, times[:-1]]
    begun = np.r_[accrued, np.zeros(times.size - 1)]          # each period's accrual at its start (today's part)
    pace = (tau - begun) / (times - starts)                     # accrual per unit of time within a period
    s_pay = _survival(times, nodes, rates)
    scheduled = float(np.sum(tau * discount(times) * s_pay))
    if integration == "midpoint":
        mid = 0.5 * (starts + times)
        lost = (_survival(starts, nodes, rates) - s_pay) * discount(mid)
        protection = (1.0 - recovery) * float(lost.sum())
        on_default = float(np.sum((begun + pace * (mid - starts)) * lost))
    else:
        grid = np.unique(np.r_[0.0, times, nodes[nodes < times[-1]], knots[knots < times[-1]]])
        a, b = grid[:-1], grid[1:]
        lam = rates[np.searchsorted(nodes[:-1], a, side="right")]
        d_a, d_b = discount(a), discount(b)
        fwd = np.log(d_a / d_b) / (b - a)
        period = np.searchsorted(times, b - 1e-12 * times[-1])
        first, second = _flat_integrals(lam + fwd, b - a)
        weight = lam * d_a * _survival(a, nodes, rates)
        protection = (1.0 - recovery) * float(np.sum(weight * first))
        start_accrual = begun[period] + pace[period] * (a - starts[period])
        on_default = float(np.sum(weight * (start_accrual * first + pace[period] * second)))
    return protection, scheduled + (on_default if accrual_on_default else 0.0), s_pay


def _discounting(discount_times, discount_factors, interpolation):
    choice("interpolation", interpolation, INTERPOLATIONS)
    nodes = np.asarray(discount_times, dtype=float).ravel()
    values = positive("discount_factors", discount_factors).ravel()
    return (lambda t: np.asarray(curve_discount(np.asarray(t, dtype=float), nodes, values, interpolation))), nodes


def _recovery(recovery, whole=False):
    """R as a fraction: below 1 for a CDS (a quote needs a loss), up to 1 where `whole` (a CVA, nothing lost at 1)."""
    r = float(recovery)
    if not (0.0 <= r <= 1.0 if whole else 0.0 <= r < 1.0):
        raise ValueError(f"recovery must lie in [0, 1{']' if whole else ')'}, a fraction (0.4, not 40)")
    return r


@_guard(time=('hazard_times', 'payment_times', 'discount_times',))
def cds_price(hazard_times, hazard_rates, recovery, payment_times, accruals, accrued, discount_times,
              discount_factors, interpolation, integration, accrual_on_default, coupon, side):
    """CDS: premium and protection legs, par spread and upfront from a hazard curve.

    Per unit notional, S(t) = exp(-int_0^t lambda), D(t) the curve. Protection (1 - R) int_0^T D dQ, Q = 1 - S;
    RPV01 = sum_i tau_i D(t_i) S(t_i), plus with accrual on default int A(u) D(u) dQ(u), A the accrued premium.
    'midpoint': defaults at period midpoints (Hull's); 'exact': closed form on the payment, hazard and curve
    grid (the ISDA standard model). Par spread = protection / RPV01; upfront = protection - c RPV01; the credit
    triangle s ~ lambda (1 - R).

    Standard contracts pay quarterly on the IMM 20ths (Mar, Jun, Sep, Dec), ACT/360: the taus are qf.bond_cash_flows'
    coupons at frequency=4, day_count="ACT/360", coupon_rate=1.0, face=1.0 from IMM dates. A bond's par-adjusted
    spread: the curve's par spread plus (model price - market price) / (face x RPV01), the same schedule.

    conventions:
    - hazard_times, hazard_rates: the piecewise-constant hazard (one node: flat); recovery: R, a fraction
    - payment_times: premium dates, years from today; accruals: each period's tau, the first whole; accrued: its
      part before today, in tau's units
    - discount_times, discount_factors, interpolation: the curve as qf.curve_discount reads it, out to T
    - integration: 'midpoint' or 'exact'; accrual_on_default: True or False
    - coupon: the running spread c (0.01 for 100 bp); side: 'buyer' or 'seller' of protection
    returns: a dict: survival (S at payment_times); protection_leg; rpv01 (clean: less accrued); rpv01_dirty;
      par_spread; premium_leg (c rpv01); upfront (clean, to side: protection - c rpv01 for the buyer);
      upfront_dirty (the cash paid: upfront less c accrued)
    source: O'Kane (2008), Modelling Single-name and Multi-name Credit Derivatives, ch. 6; ISDA; Martin (2022), 1.6
    aliases: CDS, credit default swap, CDS pricing, CDS par spread, CDS upfront, risky annuity, RPV01, protection leg, premium leg, accrual on default, credit triangle, [survival probability], [hazard rate]
    example:
        qf.cds_price(hazard_times=[3.0], hazard_rates=[0.02], recovery=0.4, payment_times=[1.0, 2.0, 3.0],
            accruals=[1.0] * 3, accrued=0.0, discount_times=[3.0], discount_factors=[0.86], interpolation="log_linear_discount",
            integration="midpoint", accrual_on_default=True, coupon=0.01, side="buyer")
    """
    choice("integration", integration, INTEGRATIONS)
    choice("side", side, SIDES)
    if not isinstance(accrual_on_default, (bool, np.bool_)):
        raise ValueError("accrual_on_default must be True or False")
    nodes, rates = _hazard(hazard_times, hazard_rates)
    times, tau, done = _schedule(payment_times, accruals, accrued)
    curve = _discounting(discount_times, discount_factors, interpolation)
    c = float(coupon)
    protection, dirty, s_pay = _cds_legs(nodes, rates, _recovery(recovery), times, tau, done, curve, integration,
                                         bool(accrual_on_default))
    clean = dirty - done
    sign = 1.0 if side == "buyer" else -1.0
    upfront = sign * (protection - c * clean)
    return Result({"survival": s_pay, "protection_leg": protection, "rpv01": clean, "rpv01_dirty": dirty,
                   "par_spread": protection / clean, "premium_leg": c * clean, "upfront": upfront,
                   "upfront_dirty": upfront - sign * c * done})


@_guard(time=('maturities', 'payment_times', 'discount_times',))
def cds_bootstrap(maturities, quotes, quote_kind, coupon, recovery, payment_times, accruals, discount_times,
                  discount_factors, interpolation, integration, accrual_on_default):
    """CDS hazard bootstrap: the piecewise-constant hazard repricing each quote, node by node.

    The contracts share one premium schedule, contract k paying on it up to its maturity T_k. In maturity order, the
    hazard on (T_{k-1}, T_k] is the root that sets contract k's value to its quote: protection - s_k RPV01 = 0 for a
    par spread s_k, protection - c RPV01 = U_k for an upfront U_k at the standard coupon c, the earlier nodes held
    (qf.cds_price's legs, accrued 0). S(t) = exp(-int_0^t lambda) then falls through the nodes; a quote that
    needs a negative hazard is refused. At r = 0, with accrual on default and each tau its period's length, a flat
    spread s gives the flat hazard s / (1 - R).

    conventions:
    - maturities: increasing, each one of payment_times; quotes: one per maturity, decimals (0.012 for 120 bp, an
      upfront 0.03 for 3 points paid by the buyer)
    - quote_kind: 'par_spread' (coupon None) or 'upfront' (clean, at the standard coupon c: 0.01 or 0.05)
    - payment_times, accruals: the longest contract's schedule, years from today and each period's tau (ACT/360
      for standard contracts)
    - recovery, discount_times, discount_factors, interpolation, integration, accrual_on_default: as cds_price
    returns: a dict: hazard_times (the maturities); hazard_rates (one per node); survival (S at the maturities);
      par_spreads (each contract's par spread on the curve); repriced (each quote recomputed from the curve)
    source: O'Kane (2008), Modelling Single-name and Multi-name Credit Derivatives, ch. 7; the ISDA Standard Model
    aliases: CDS bootstrap, hazard rate bootstrap, credit curve bootstrap, bootstrap hazard rates, survival curve from CDS spreads, CDS curve, implied hazard rates, default probability term structure
    example:
        qf.cds_bootstrap(maturities=[1.0, 3.0, 5.0], quotes=[0.006, 0.009, 0.012], quote_kind="par_spread", coupon=None,
            recovery=0.4, payment_times=[0.5 * k for k in range(1, 11)], accruals=[0.5] * 10, discount_times=[5.0], discount_factors=[0.8],
            interpolation="log_linear_discount", integration="exact", accrual_on_default=True)
    """
    choice("quote_kind", quote_kind, QUOTE_KINDS)
    choice("integration", integration, INTEGRATIONS)
    if not isinstance(accrual_on_default, (bool, np.bool_)):
        raise ValueError("accrual_on_default must be True or False")
    times, tau, _ = _schedule(payment_times, accruals, 0.0)
    nodes = np.asarray(maturities, dtype=float).ravel()
    quoted = np.asarray(quotes, dtype=float).ravel()
    if nodes.size == 0 or quoted.shape != nodes.shape or np.any(np.diff(nodes) <= 0) or not np.all(np.isfinite(quoted)):
        raise ValueError("maturities must be increasing with one finite quote each")
    ends = np.searchsorted(times, nodes * (1 - ON_GRID))
    if np.any(ends >= times.size) or np.any(np.abs(times[np.minimum(ends, times.size - 1)] - nodes) > ON_GRID * nodes):
        raise ValueError("each maturity must be one of payment_times")
    if (coupon is None) != (quote_kind == "par_spread"):
        raise ValueError("coupon: None for 'par_spread' quotes, the standard coupon for 'upfront' quotes")
    c = None if coupon is None else float(coupon)
    r = _recovery(recovery)
    curve = _discounting(discount_times, discount_factors, interpolation)
    rates = np.zeros(nodes.size)

    def legs(k, lam):
        trial = np.r_[rates[:k], lam]
        return _cds_legs(nodes[:k + 1], trial, r, times[:ends[k] + 1], tau[:ends[k] + 1], 0.0, curve, integration,
                         bool(accrual_on_default))

    def gap(lam, k):
        protection, rpv01, _ = legs(k, lam)
        return protection - quoted[k] * rpv01 if c is None else protection - c * rpv01 - quoted[k]

    for k in range(nodes.size):
        low, high = gap(0.0, k), gap(HAZARD_HIGH, k)
        if not low <= 0.0 <= high:
            raise ValueError(f"quotes[{k}]: no hazard in [0, {HAZARD_HIGH:g}] reprices maturity {nodes[k]:g}")
        rates[k] = 0.0 if low == 0.0 else brentq(gap, 0.0, HAZARD_HIGH, args=(k,), xtol=1e-16, rtol=1e-15)
    spreads, repriced = np.empty(nodes.size), np.empty(nodes.size)
    for k in range(nodes.size):
        protection, rpv01, _ = legs(k, rates[k])
        spreads[k] = protection / rpv01
        repriced[k] = spreads[k] if c is None else protection - c * rpv01
    return Result({"hazard_times": nodes, "hazard_rates": rates, "survival": _survival(nodes, nodes, rates),
                   "par_spreads": spreads, "repriced": repriced})


def _curve_on(name, values, size, capped):
    """One positive value per time, 1 at times[0]; at most 1 where `capped` (a survival; a discount factor passes 1
    under negative rates)."""
    found = np.asarray(values, dtype=float).ravel()
    if (found.size != size or not np.all(np.isfinite(found)) or np.any(found <= 0)
            or capped and np.any(found > 1 + 1e-12)):
        raise ValueError(f"{name} must hold one value {'in (0, 1]' if capped else 'above 0'} per time, times[0]'s 1")
    if abs(found[0] - 1.0) > 1e-12:
        raise ValueError(f"{name}[0] must be 1, at times[0] = 0")
    return found


def _collateralised(v, collateral):
    """V less the collateral a symmetric CSA holds: C_i = sign(V_{i-l}) (|V_{i-l}| - H)^+, nothing below the minimum
    transfer M, none before the lag l (in grid steps) has passed."""
    if collateral is None:
        return v
    if not isinstance(collateral, dict) or set(collateral) != set(COLLATERAL_KEYS):
        raise ValueError(f"collateral must be None or a dict of exactly {', '.join(COLLATERAL_KEYS)}")
    h, m = float(collateral["threshold"]), float(collateral["minimum_transfer"])
    lag = integer("collateral['lag']", collateral["lag"], 0)
    if h < 0 or m < 0:
        raise ValueError("collateral's threshold and minimum_transfer must not be negative")
    called = np.zeros_like(v)
    seen = v[:, :v.shape[1] - lag] if lag else v
    amount = np.maximum(np.abs(seen) - h, 0.0)
    called[:, lag:] = np.sign(seen) * np.where(amount >= m, amount, 0.0)
    return v - called


@_guard(time=('times',))
def cva(values, times, discount_factors, counterparty_survival, counterparty_recovery, own_survival, own_recovery,
        timing, kind, collateral):
    """CVA and DVA from simulated values of a netting set: the exposure profile and the charge.

    Values are netted over the set's trades before the positive part: E = (V - C)^+, EE its mean over paths, NEE
    that of (V - C)^-. CVA = (1 - R_c) sum_i D EE [S_c(t_{i-1}) - S_c(t_i)] over the intervals (t_{i-1}, t_i]; DVA
    the same with NEE, the own survival and recovery. 'bilateral': each as if the other party could not default
    (Hull's); 'first_to_default': each interval also weighted by the other's survival to its start (Gregory's). R = 1
    or no default gives 0. PFE is a quantile of the returned exposure (numpy.quantile(exposure, 0.95, axis=0)).

    conventions:
    - values: n_paths x n_times values of the netting set to us (x n_trades, summed); times: increasing years,
      times[0] = 0; for 'given_at_midpoints' one column per interval (n_times - 1), at its midpoint
    - discount_factors: D at each time (1 at times[0]), or at each interval's midpoint for 'given_at_midpoints'
    - counterparty_survival, own_survival: S_c and S_o at every time, 1 at times[0]; recoveries are fractions;
      own_survival and own_recovery None for 'unilateral'
    - timing: how D EE is read in an interval: 'end' at its end; 'midpoint' the mean of D EE at its two ends;
      'given_at_midpoints' as passed at its midpoint (Hull's: the exposure simulated there)
    - kind: 'unilateral', 'bilateral' or 'first_to_default'
    - collateral: None, or {'threshold': H, 'minimum_transfer': M, 'lag': l steps}, a symmetric CSA holding
      sign(V) (|V| - H)^+ as of l steps before, none below M
    returns: a dict: cva; dva (0 when unilateral); bcva (cva - dva); ee and nee (one per value column); exposure
      (paths x columns, (V - C)^+)
    source: Gregory (2015), The xVA Challenge, ch. 14; Hull, "Credit value adjustment"; Basel III (2011), CVA
    aliases: CVA, DVA, credit valuation adjustment, debit valuation adjustment, bilateral CVA, BCVA, expected exposure, expected positive exposure, potential future exposure, PFE, netting set, counterparty credit risk
    example:
        qf.cva(values=[[0.0, 2.0, -1.0], [0.0, -0.5, 3.0], [0.0, 1.0, 0.5]], times=[0.0, 0.5, 1.0], discount_factors=[1.0, 0.98, 0.96],
            counterparty_survival=[1.0, 0.99, 0.975], counterparty_recovery=0.4, own_survival=None, own_recovery=None,
            timing="end", kind="unilateral", collateral=None)
    """
    choice("timing", timing, TIMINGS)
    choice("kind", kind, CVA_KINDS)
    v = np.asarray(values, dtype=float)
    v = v.sum(axis=2) if v.ndim == 3 else v
    t = np.asarray(times, dtype=float).ravel()
    given = timing == "given_at_midpoints"
    columns = t.size - 1 if given else t.size
    if v.ndim != 2 or v.shape[1] != columns or t.size < 2 or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("values must be n_paths x n_times (x n_trades) on increasing times from times[0] = 0, one "
                         "column per interval's midpoint (n_times - 1) for 'given_at_midpoints'")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")
    if given:
        d = np.asarray(discount_factors, dtype=float).ravel()
        if d.size != columns or not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError("discount_factors must hold one positive value per interval's midpoint")
    else:
        d = _curve_on("discount_factors", discount_factors, t.size, False)
    s_c = _curve_on("counterparty_survival", counterparty_survival, t.size, True)
    r_c = _recovery(counterparty_recovery, whole=True)
    if kind == "unilateral":
        if own_survival is not None or own_recovery is not None:
            raise ValueError("own_survival and own_recovery are for 'bilateral' and 'first_to_default': pass None")
    else:
        s_o, r_o = _curve_on("own_survival", own_survival, t.size, True), _recovery(own_recovery, whole=True)
    net = _collateralised(v, collateral)
    exposure = np.maximum(net, 0.0)
    ee, nee = exposure.mean(axis=0), np.maximum(-net, 0.0).mean(axis=0)

    def charge(profile, survival, recovery, other):
        held = d * profile
        weight = held if given else held[1:] if timing == "end" else 0.5 * (held[:-1] + held[1:])
        alive = other[:-1] if kind == "first_to_default" else 1.0
        return float((1.0 - recovery) * np.sum(weight * (survival[:-1] - survival[1:]) * alive))

    found_cva = charge(ee, s_c, r_c, None if kind != "first_to_default" else s_o)
    found_dva = 0.0 if kind == "unilateral" else charge(nee, s_o, r_o, s_c)
    return Result({"cva": found_cva, "dva": found_dva, "bcva": found_cva - found_dva, "ee": ee, "nee": nee,
                   "exposure": exposure})


# ------------------------------------------------------------------------------------------------ bonds


@_guard(time=('payment_times', 'hazard_times', 'discount_times',))
def risky_bond_price(*, payment_times, coupons, face, hazard_times, hazard_rates, recovery_rate, recovery,
                     recovery_timing, discount_times, discount_factors, interpolation):
    """Defaultable bond price under a hazard rate, recovery of face, of treasury or of market value (Duffie-Singleton).

    With S(t) = exp(-int_0^t lambda) and D(t) the riskless curve, the promised flows (coupons, the face at the end)
    are worth sum_j cf_j D(t_j) S(t_j), plus default's payment: 'face', R times the face at default, R F int_0^T D dQ
    (Q = 1 - S); 'treasury', R times the riskless value of the flows still due, R sum_j cf_j D(t_j) (1 - S(t_j));
    'market_value' (Duffie-Singleton), each flow discounted at r + (1 - R) lambda, sum_j cf_j D(t_j) S(t_j)^(1 - R).
    A zero hazard gives the riskless price; a zero-coupon bond's credit spread is -ln(price / riskless_price) / T.
    conventions:
        payment_times: years from today, increasing; coupons: each payment's coupon (0s for a zero)
        face: the principal repaid at the last payment time
        hazard_times, hazard_rates: the piecewise-constant intensity, rate j up to hazard_times[j] (one: flat)
        recovery_rate: R, a fraction of the face, of the riskless value or of the pre-default value (by recovery)
        recovery: 'face' (of par), 'treasury' or 'market_value'
        recovery_timing: when 'face' recovery is paid: 'at_default' (the exact integral), 'midpoint' (defaults at
            each period's midpoint, Hull's) or 'period_end' (at the period's payment date); unused by the others
        discount_times, discount_factors, interpolation: the riskless curve as qf.curve_discount reads it
    returns: a dict: price; riskless_price; expected_loss (riskless_price - price); survival (S at payment_times)
    source: Lando, Credit Risk Modeling (2004), 5.7 (the three recoveries); Duffie and Singleton (1999), Review of
        Financial Studies 12(4); Hull, Options, Futures, and Other Derivatives, "Credit risk"
    aliases: defaultable bond, risky bond, risky bond price, defaultable zero-coupon bond, recovery of face, recovery of par, recovery of treasury, recovery of market value, Duffie-Singleton, intensity model, [credit spread]
    example:
        import qf
        qf.risky_bond_price(payment_times=[1.0, 2.0, 3.0], coupons=[5.0, 5.0, 5.0], face=100.0, hazard_times=[3.0], hazard_rates=[0.02], recovery_rate=0.4, recovery="face", recovery_timing="at_default", discount_times=[3.0], discount_factors=[0.88], interpolation="log_linear_discount")
        # price = 98.315164
    """
    choice("recovery", recovery, RECOVERIES)
    choice("recovery_timing", recovery_timing, RECOVERY_TIMINGS)
    times = np.asarray(payment_times, dtype=float).ravel()
    if times.size == 0 or times[0] <= 0 or np.any(np.diff(times) <= 0) or not np.all(np.isfinite(times)):
        raise ValueError("payment_times must be increasing positive times")
    flows = np.broadcast_to(np.asarray(coupons, dtype=float), times.shape).copy()
    principal = float(face)
    if not np.all(np.isfinite(flows)) or not math.isfinite(principal):
        raise ValueError("coupons and face must be finite")
    flows[-1] += principal
    nodes, rates = _hazard(hazard_times, hazard_rates)
    r = _recovery(recovery_rate, whole=True)
    discount, knots = _discounting(discount_times, discount_factors, interpolation)
    d, s = discount(times), _survival(times, nodes, rates)
    riskless = float(flows @ d)
    if recovery == "market_value":
        price = float(flows @ (d * s ** (1.0 - r)))
    elif recovery == "treasury":
        price = float(flows @ (d * (s + r * (1.0 - s))))
    else:
        starts = np.r_[0.0, times[:-1]]
        if recovery_timing == "at_default":
            grid = np.unique(np.r_[0.0, times, nodes[nodes < times[-1]], knots[knots < times[-1]]])
            a, b = grid[:-1], grid[1:]
            lam = rates[np.searchsorted(nodes[:-1], a, side="right")]
            if interpolation == "log_linear_discount":
                fwd = np.log(discount(a) / discount(b)) / (b - a)
                lost = float(np.sum(lam * discount(a) * _survival(a, nodes, rates) *
                                    _flat_integrals(lam + fwd, b - a)[0]))
            else:
                lost = sum(quad(lambda u: intensity * float(discount(u)) *
                                float(_survival(u, nodes, rates)), start, stop,
                                epsabs=1e-12, epsrel=1e-12)[0]
                           for start, stop, intensity in zip(a, b, lam))
        else:
            paid = discount(0.5 * (starts + times)) if recovery_timing == "midpoint" else d
            lost = float(np.sum((_survival(starts, nodes, rates) - s) * paid))
        price = float(flows @ (d * s)) + r * principal * lost
    return Result({"price": price, "riskless_price": riskless, "expected_loss": riskless - price, "survival": s})


def _covered(windows, name, steps, dt):
    """{step: price} of the steps a list of (start, end, price) windows covers: steps round(start/dt) to round(end/dt)."""
    found = {}
    for window in windows or ():
        start, end, price = (float(x) for x in window)
        if not (0.0 <= start <= end and math.isfinite(price) and price > 0):
            raise ValueError(f"{name} must be (start, end, price) windows, 0 <= start <= end, the price positive")
        for k in range(int(round(start / dt)), min(int(round(end / dt)), steps) + 1):
            found[k] = price
    return found


@_guard(rate=('rate', 'dividend_yield', 'credit_spread',), vol=('vol',), time=('maturity',), price=('spot',))
def convertible_tree(*, spot, vol, rate, dividend_yield, credit_spread, maturity, face, conversion_ratio,
                     coupon_times, coupons, last_coupon_time, calls, puts, price_basis, steps):
    """Convertible bond on a binomial tree, Tsiveriotis-Fernandes model only: cash part discounted at the credit spread.

    On a CRR share tree V splits into U, the holder's cash, at r + s, and V - U, the shares, at r. At maturity
    V = max(ratio S, face + coupon); back each step a call above K_c makes V = max(K_c, ratio S), a put below K_p
    makes V = K_p, and conversion (U = 0) wins where ratio S beats holding; a step's coupon is paid unless converted,
    before the call and put. Without calls the price is at least max(parity, bond floor).

    Parity = ratio S; conversion premium = price / parity - 1; investment premium = price / floor - 1 (floor: the
    flows at r + s); current yield = coupon / price; payback = (price - parity) / (coupon - ratio x dividend), in
    years; delta = the shares per bond that hedge it.
    conventions:
        the model: this split alone; a task's own stated tree (one rate on the whole value, its steps) is done by
          hand or with qf.lattice_backward
        spot, vol, rate, dividend_yield: S, its vol, r and q, continuous and annual
        credit_spread: s, the issuer's spread over r on the cash part
        maturity: T in years; face: paid at T unless converted
        conversion_ratio: shares per bond (face / conversion price)
        coupon_times, coupons: each coupon's time (on the nearest step) and amount; [] and [] for none
        last_coupon_time: the last coupon date before today (0 or negative), the accrual's start
        calls, puts: (start, end, price) windows over the steps round(start/dt) to round(end/dt); None for none
        price_basis: 'clean' (accrued paid on top when called or put) or 'dirty'
        steps: the tree's steps
    returns: a dict: price; cash_part (U); equity_part (V - U); parity; bond_floor; delta; conversion_premium;
        investment_premium
    source: Tsiveriotis and Fernandes (1998), Journal of Fixed Income 8(2); Hull, "Convertible bonds"
    aliases: convertible bond, convertible bond price, Tsiveriotis-Fernandes, TF model, convertible bond tree, conversion premium
    example:
        qf.convertible_tree(spot=50.0, vol=0.3, rate=0.05, dividend_yield=0.0, credit_spread=0.02, maturity=2.0, face=100.0, conversion_ratio=2.0, coupon_times=[1.0, 2.0], coupons=[4.0, 4.0], last_coupon_time=0.0, calls=None, puts=None, price_basis="clean", steps=200)
        # price about 115.35
    """
    choice("price_basis", price_basis, PRICE_BASES)
    s0, sigma, t = (float(positive(name, value)) for name, value in
                    (("spot", spot), ("vol", vol), ("maturity", maturity)))
    r, q, spread = float(rate), float(dividend_yield), float(credit_spread)
    principal, ratio = float(positive("face", face)), float(positive("conversion_ratio", conversion_ratio))
    n = integer("steps", steps, 1)
    dt = t / n
    up = math.exp(sigma * math.sqrt(dt))
    p = (math.exp((r - q) * dt) - 1.0 / up) / (up - 1.0 / up)
    if not 0.0 < p < 1.0:
        raise ValueError("the tree's probability leaves (0, 1): take more steps")
    when = np.asarray(coupon_times, dtype=float).ravel()
    paid = np.broadcast_to(np.asarray(coupons, dtype=float), when.shape)
    if np.any(when <= 0) or np.any(when > t * (1 + 1e-12)) or np.any(np.diff(when) <= 0) or np.any(paid < 0):
        raise ValueError("coupon_times must increase within (0, maturity], with one coupon amount each")
    begun = float(last_coupon_time)
    if begun > 0:
        raise ValueError("last_coupon_time must be 0 or negative (the last coupon date before today)")
    at = np.minimum(np.rint(when / dt).astype(int), n)
    coupon = np.zeros(n + 1)
    np.add.at(coupon, at, paid)
    edges = np.r_[begun, at * dt]                   # each coupon period's start and end, on the steps

    def accrued(k):
        if price_basis == "dirty" or when.size == 0:
            return 0.0
        if np.any(at == k):
            return float(paid[at == k].sum())
        j = int(np.searchsorted(at, k, side="right"))
        if j >= at.size:
            return 0.0
        return float(paid[j] * (k * dt - edges[j]) / (edges[j + 1] - edges[j]))

    called, put = _covered(calls, "calls", n, dt), _covered(puts, "puts", n, dt)
    v = np.full(n + 1, principal)
    u = v.copy()
    grow, risky = math.exp(-r * dt), math.exp(-(r + spread) * dt)
    first = None
    for k in range(n, -1, -1):
        prices = s0 * up ** (2.0 * np.arange(k + 1) - k)
        if k < n:
            equity = grow * (p * (v[1:k + 2] - u[1:k + 2]) + (1.0 - p) * (v[:k + 1] - u[:k + 1]))
            u = risky * (p * u[1:k + 2] + (1.0 - p) * u[:k + 1])
            v = equity + u
        parity = ratio * prices
        v, u = v + coupon[k], u + coupon[k]
        if k in called:
            price = called[k] + accrued(k)
            take = v > price
            shares = take & (parity > price)
            v = np.where(take, np.maximum(price, parity), v)
            u = np.where(take & ~shares, price, np.where(shares, 0.0, u))
        if k in put:
            price = put[k] + accrued(k)
            low = v < price
            v, u = np.where(low, price, v), np.where(low, price, u)
        converted = parity > v                         # converting forgoes a coupon paid on this step
        v, u = np.where(converted, parity, v), np.where(converted, 0.0, u)
        if k == 1:
            first = (v.copy(), prices.copy())
    floor = principal * math.exp(-(r + spread) * t) + float(coupon @ np.exp(-(r + spread) * dt * np.arange(n + 1)))
    price, cash = float(v[0]), float(u[0])
    delta = float((first[0][1] - first[0][0]) / (first[1][1] - first[1][0])) if first is not None else math.nan
    parity0 = ratio * s0
    return Result({"price": price, "cash_part": cash, "equity_part": price - cash, "parity": parity0,
                   "bond_floor": floor, "delta": delta, "conversion_premium": price / parity0 - 1.0,
                   "investment_premium": price / floor - 1.0})
