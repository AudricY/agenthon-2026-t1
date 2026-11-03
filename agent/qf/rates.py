"""Rate derivatives: swaps, caps, caplet stripping, CMS and Eurodollar convexity, SOFR in arrears, LMM (Rebonato).

`import qf`. Times are year fractions from today under the caller's day count (cash-flow times, maturities,
fixings); zero rates are continuously compounded unless a function names another basis; discount factors are
P(0, t). A scalar result is a float; a function with several results returns a dict with named keys. The short-rate
models' functions are on the shortrate shelf since lane lib-rates1, their old paths kept.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.optimize import brentq

from ._common import Result, choice, integer, ndtr, positive
from .conventions import _dates
from .curves import INTERPOLATIONS, _curve, _increasing, _whole_frequency
from .options import black76_price
from .shortrate import (SIDES, cir_fit, hull_white_calibrate, hull_white_caplet,  # noqa: F401 - moved to shortrate
                        hull_white_zcb, hull_white_zcb_option, jamshidian_swaption, short_rate_bond_price,
                        short_rate_tree)
from .primitives import _guard

__all__ = ["black76_cap_floor", "caplet_vol_strip", "cms_convexity_adjustment", "compounded_overnight_rate",
           "futures_convexity_adjustment", "lmm_vols", "swap_rate"]

OVERNIGHT_DAY_COUNTS = ("ACT/360", "ACT/365F")
OBSERVATIONS = ("lag", "shift")
CONVEXITY_MODELS = ("ho_lee", "hull_white")
QUOTE_BASES = ("continuous", "simple_act360")
CMS_METHODS = ("replication", "approximation")
CMS_MAPPINGS = ("linear", "standard")
VOL_KINDS = ("black", "normal")
FIRST_CAPLETS = ("excluded", "included")
STRIP_SHAPES = ("constant", "linear")
STRIP_HIGH = 5.0                # a stripped Black vol is sought in (0, this]; a normal vol in (0, this / 50]
LEGENDRE = 64                   # Gauss-Legendre nodes for the abcd integrals (smooth integrands: exact to rounding)
CMS_SPAN = 12.0                 # a normal smile's replication runs over the forward plus or minus this many deviations
CMS_STEP = 1e-4                 # the step of the standard mapping's finite differences, in rate units


@_guard(rate=('strike',), time=('start', 'fixed_times', 'float_times',))
def swap_rate(start, fixed_times, fixed_accruals, float_times, discount_curve, projection_curve, interpolation,
              strike, side, notional):
    """Par swap rate, annuity and swap value.

    Annuity A = sum_i tau_i D(T_i) over the fixed leg; floating leg sum_j (P(t_{j-1}) / P(t_j) - 1) D(t_j) from
    t_0 = start, the forwards projected off P and discounted off D, which on one curve (P = D) is D(T_0) - D(T_n); the
    par rate S = floating leg / A; a payer swap (pay fixed K) is worth N A (S - K), a receiver N A (K - S). One
    period gives the simple forward (D(T_0) / D(T_1) - 1) / tau. A is the PV01 per unit of rate (times 0.0001 a basis
    point); a European swaption is qf.black76_price on the forward S with A as its discount factor.

    conventions:
    - start: T_0, years from today (0.0 spot-starting); fixed_times, fixed_accruals: the fixed leg's payments and tau_i
      under its day count; float_times: the floating periods' ends, after T_0
    - discount_curve: (node_times, node_discounts) of D; projection_curve: the same for P, or None to project off D
    - interpolation: 'linear_zero', 'log_linear_discount' or 'cubic_log_discount', as curve_discount reads a curve
    - strike: K, the fixed rate; side: 'payer' (pay fixed) or 'receiver'; notional: N
    returns: a dict: annuity (A per unit notional); par_rate (S); float_leg (per unit notional); value (N times the
      side's sign times (float_leg - K A))
    source: Hull, Options, Futures, and Other Derivatives, "Swaps" and "Swaptions"; Andersen and Piterbarg (2010), 5.5
    aliases: swap rate, par swap rate, forward swap rate, swap annuity, annuity factor, PV01, swap valuation, interest rate swap, swap value, fixed leg, floating leg, [swap]
    example:
        qf.swap_rate(start=0.0, fixed_times=[1.0, 2.0, 3.0], fixed_accruals=[1.0, 1.0, 1.0],
                     float_times=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], discount_curve=([1.0, 3.0], [0.97, 0.91]),
                     projection_curve=None, interpolation="log_linear_discount", strike=0.03, side="payer", notional=1e6)
    """
    choice("interpolation", interpolation, INTERPOLATIONS)
    choice("side", side, SIDES)
    t0 = float(start)
    fixed, floating = _increasing("fixed_times", fixed_times), _increasing("float_times", float_times)
    tau = positive("fixed_accruals", fixed_accruals).ravel()
    if tau.shape != fixed.shape or not (0 <= t0 < min(fixed[0], floating[0])):
        raise ValueError("fixed_accruals must hold one accrual per fixed time, and start must be before every payment")
    d = _curve("discount_curve", discount_curve, interpolation)
    p = d if projection_curve is None else _curve("projection_curve", projection_curve, interpolation)
    annuity = float(tau @ d(fixed))
    projected = p(np.r_[t0, floating])
    float_leg = float((projected[:-1] / projected[1:] - 1.0) @ d(floating))
    sign = 1.0 if side == "payer" else -1.0
    return Result({"annuity": annuity, "par_rate": float_leg / annuity, "float_leg": float_leg,
                   "value": sign * float(notional) * (float_leg - float(strike) * annuity)})


# ------------------------------------------------------------------------------------------------ caps


@_guard(rate=('forwards', 'strike',), vol=('vol',), time=('fixing_times',))
def black76_cap_floor(option, forwards, strike, fixing_times, accruals, discount_factors, vol, notional):
    """Cap or floor price from Black-76 caplets or floorlets on a schedule of forward rates.

    caplet_i = N tau_i P(0, T_i + tau_i) [F_i N(d1) - K N(d2)], floorlet_i = N tau_i P(0, ...) [K N(-d2) - F_i N(-d1)],
    d1 = [ln(F_i/K) + vol_i^2 t_i / 2] / (vol_i sqrt(t_i)), t_i the fixing time (bs.black76_price); a period fixing
    now (t_i = 0) is worth its intrinsic value. Cap less floor is the payer swap, N sum_i tau_i P_i (F_i - K).

    conventions:
    - forwards: the simply compounded forward rate of each period; fixing_times: each period's reset, not its payment
    - accruals: tau_i, each period's year fraction; discount_factors: to each period's payment date
    - vol: Black's lognormal vol, one for all or one per period; notional: N; strike: K, simply compounded
    returns: a dict: price (in the notional's units); caplets (the per-period values; floorlets for a floor)
    source: Black (1976); Hull, Options, Futures, and Other Derivatives, "Caps and floors"; Brigo and Mercurio, 1.6
    aliases: cap, floor, caplet, floorlet, [interest rate cap], Black caplet, Black-76 cap, cap-floor parity
    example:
        qf.black76_cap_floor(option="cap", forwards=[0.040, 0.043, 0.045], strike=0.042,
                             fixing_times=[0.0, 0.25, 0.5], accruals=[0.25, 0.25, 0.25],
                             discount_factors=[0.990, 0.979, 0.968], vol=0.2, notional=1000000.0)
    """
    choice("option", option, ("cap", "floor"))
    f = positive("forwards", forwards).ravel()
    t = np.asarray(fixing_times, dtype=float).ravel()
    tau, p = positive("accruals", accruals).ravel(), positive("discount_factors", discount_factors).ravel()
    if not (t.shape == tau.shape == p.shape == f.shape) or np.any(t < 0):
        raise ValueError("forwards, fixing_times (not negative), accruals and discount_factors must have one entry "
                         "per period")
    v = np.broadcast_to(np.asarray(vol, dtype=float), f.shape)
    if np.any(v < 0):
        raise ValueError("vol must not be negative")
    lets = float(notional) * tau * np.asarray(black76_price("call" if option == "cap" else "put", f, strike, t, v, p))
    return Result({"price": float(lets.sum()), "caplets": lets})


def _normal_call(forward, strike, expiry, vol):
    """Bachelier's undiscounted call on a normal forward: (F - K) N(d) + v sqrt(T) phi(d), d = (F - K) / (v sqrt(T))."""
    spread = np.asarray(forward, dtype=float) - np.asarray(strike, dtype=float)
    width = np.asarray(vol, dtype=float) * np.sqrt(np.asarray(expiry, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = spread / width
        found = spread * ndtr(d) + width * np.exp(-0.5 * d * d) / np.sqrt(2.0 * np.pi)
    return np.where(width > 0, found, np.maximum(spread, 0.0))


def _undiscounted_call(kind, forward, strike, expiry, vol):
    if kind == "normal":
        return _normal_call(forward, strike, expiry, vol)
    return black76_price("call", forward, strike, expiry, vol, 1.0)


@_guard(rate=('strikes', 'forwards',), time=('cap_maturities', 'fixing_times',))
def caplet_vol_strip(cap_maturities, cap_vols, strikes, fixing_times, accruals, forwards, discount_factors,
                     first_caplet, interpolation, vol_kind):
    """Caplet volatility stripping: caplet volatilities from flat cap volatilities, solved cap by cap.

    Each cap is priced at its flat vol, every caplet at that one vol (Black-76, or Bachelier for normal vols): cap_k =
    sum_i tau_i P_i Black(F_i, K_k, T_i, flat_k). New caplets take vols solved so that
    the caplets at their own vols sum to cap_k, the earlier caplets keeping theirs; 'constant' gives the new caplets one
    vol, 'linear' a vol linear in the fixing time from the last stripped caplet's to the new segment's end (the first
    cap's caplets share one). Caplet vols that are all one value give that value as every flat vol: the round trip.

    conventions:
    - cap_maturities: each cap's end in years, increasing; cap_vols: its flat vol (Black's lognormal, or the normal
      vol in rate units for 'normal'); strikes: one per cap (an at-the-money cap's own swap rate) or one for all
    - fixing_times, accruals, forwards, discount_factors: the caplets' schedule, one entry each: the fixing T_i, the
      period tau_i, the forward F_i (simply compounded) and the discount to its payment T_i + tau_i
    - first_caplet: 'excluded' (the first period, fixed today, belongs to no cap: the market's caps) or 'included'
    - interpolation: 'constant' or 'linear', the caplet vols between cap maturities; vol_kind: 'black' or 'normal'
    returns: a dict: caplet_vols (one per caplet in a cap); fixing_times (theirs); cap_prices (each cap at its flat
      vol, per unit notional)
    source: Hull, Options, Futures, and Other Derivatives, "Caps and floors" (spot and flat volatilities); Brigo and
      Mercurio, Interest Rate Models (2006), 6.4 (stripping caplet volatilities from cap quotes)
    aliases: caplet stripping, caplet vol stripping, caplet volatility stripping, strip caplet volatilities, optionlet stripping, flat cap volatility, flat vol to spot vol, caplet spot volatility, cap volatility bootstrap
    example:
        # caplet_vols: [0.2, 0.22603408, 0.22603408]
        qf.caplet_vol_strip(cap_maturities=[1.0, 2.0], cap_vols=[0.2, 0.22], strikes=0.04,
                            fixing_times=[0.0, 0.5, 1.0, 1.5], accruals=[0.5, 0.5, 0.5, 0.5],
                            forwards=[0.035, 0.038, 0.041, 0.043], discount_factors=[0.983, 0.964, 0.945, 0.925],
                            first_caplet="excluded", interpolation="constant", vol_kind="black")
    """
    choice("first_caplet", first_caplet, FIRST_CAPLETS)
    choice("interpolation", interpolation, STRIP_SHAPES)
    choice("vol_kind", vol_kind, VOL_KINDS)
    t = np.asarray(fixing_times, dtype=float).ravel()
    tau, p = positive("accruals", accruals).ravel(), positive("discount_factors", discount_factors).ravel()
    f = np.asarray(forwards, dtype=float).ravel()
    if not (t.shape == tau.shape == p.shape == f.shape) or t.size == 0 or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError("fixing_times (increasing, not negative), accruals, forwards and discount_factors must hold "
                         "one entry per caplet")
    if vol_kind == "black" and np.any(f <= 0):
        raise ValueError("forwards must be positive for Black vols")
    caps = _increasing("cap_maturities", cap_maturities)
    flat = positive("cap_vols", cap_vols).ravel()
    k = np.broadcast_to(np.asarray(strikes, dtype=float), caps.shape).astype(float)
    if flat.shape != caps.shape or (vol_kind == "black" and np.any(k <= 0)):
        raise ValueError("cap_vols must hold one vol per cap maturity, and strikes one per cap (positive for Black)")
    used = np.arange(t.size)[1:] if first_caplet == "excluded" else np.arange(t.size)
    ends = t + tau

    def caplet(i, strike, vol):
        return tau[i] * p[i] * _undiscounted_call(vol_kind, f[i], strike, t[i], vol)

    stripped, done, prices = np.full(t.size, np.nan), np.array([], dtype=int), []
    high = STRIP_HIGH if vol_kind == "black" else STRIP_HIGH / 50.0
    for cap, vol, strike in zip(caps, flat, k):
        members = used[ends[used] <= cap * (1 + 1e-12)]
        new = np.setdiff1d(members, done)
        if new.size == 0:
            raise ValueError(f"the cap ending at {cap:g} adds no caplet to the shorter caps")
        target = float(sum(caplet(i, strike, vol) for i in members))
        known = float(sum(caplet(i, strike, stripped[i]) for i in done))
        prices.append(target)
        if interpolation == "linear" and done.size:
            last = done[-1]
            weights = (t[new] - t[last]) / (t[new[-1]] - t[last])

            def vols(s, last=last, weights=weights):
                return stripped[last] + (s - stripped[last]) * weights
        else:
            def vols(s, new=new):
                return np.full(new.size, s)

        def gap(s, new=new, strike=strike, known=known, target=target, vols=vols):
            return known + float(sum(caplet(i, strike, v) for i, v in zip(new, vols(s)))) - target

        if gap(1e-8) > 0 or gap(high) < 0:
            raise ValueError(f"no caplet vol in (0, {high:g}] reprices the cap ending at {cap:g}")
        stripped[new] = vols(brentq(gap, 1e-8, high, xtol=1e-15, rtol=4.5e-15))
        done = np.union1d(done, new)
    return Result({"caplet_vols": stripped[done], "fixing_times": t[done], "cap_prices": np.array(prices)})


# ------------------------------------------------------------------------------------------------ convexity


@_guard(time=('start', 'end',), price=('futures_price',))
def futures_convexity_adjustment(model, futures_price, start, end, sigma, mean_reversion, quote_basis):
    """Futures-forward convexity adjustment of rate futures (Eurodollar, SOFR): the forward rate from the futures price.

    The futures rate exceeds the forward rate for [t1, t2], both continuously compounded, by Ho-Lee's 1/2 sigma^2 t1
    t2, or Hull-White's B(t1, t2) / (t2 - t1) [B(t1, t2)(1 - e^{-2a t1}) + 2a B(0, t1)^2] sigma^2 / (4a), B(s, u) = (1
    - e^{-a (u - s)}) / a, which tends to Ho-Lee's as a -> 0: forward = futures rate - adjustment. The futures rate is
    1 - price / 100; 'simple_act360' first makes it continuous on ACT/365 over the period's days d = 360 (t2 - t1),
    (365 / d) ln(1 + r d / 360), as Hull's example does, and gives the forward back on the quote's basis.

    conventions:
    - model: 'ho_lee' (mean_reversion 0.0) or 'hull_white' (mean_reversion a, positive)
    - futures_price: the quote, 100 less the rate in percent (94.0 is 6%)
    - start: t1, the future's expiry and the rate's start; end: t2, the rate's end, in years
    - sigma: the short rate's absolute (normal) vol a year, a decimal (0.012 is 1.2%)
    - quote_basis: 'continuous' (the quoted rate taken as continuously compounded, the formula as stated) or
      'simple_act360' (the quoted rate a simple ACT/360 rate over the period, converted first)
    returns: a dict: futures_rate (1 - price / 100); futures_continuous (the rate adjusted); adjustment;
      forward_continuous (futures_continuous - adjustment); forward_rate (that on the quote's basis)
    source: Hull, Options, Futures, and Other Derivatives, "Convexity adjustments" (Example 6.4); Kirikos and Novak
      (1997), "Convexity conundrums", Risk 10(3)
    aliases: convexity adjustment, futures convexity adjustment, Eurodollar convexity adjustment, futures-forward bias, futures rate to forward rate, Eurodollar futures, SOFR futures convexity
    example:
        # forward_rate: 0.055248; adjustment: 0.004752
        qf.futures_convexity_adjustment(model="ho_lee", futures_price=94.0, start=8.0, end=8.25, sigma=0.012,
                                        mean_reversion=0.0, quote_basis="continuous")
    """
    choice("model", model, CONVEXITY_MODELS)
    choice("quote_basis", quote_basis, QUOTE_BASES)
    t1, t2, vol, a = float(start), float(end), float(sigma), float(mean_reversion)
    if not 0 <= t1 < t2 or not vol >= 0:
        raise ValueError("start and end must satisfy 0 <= start < end, and sigma must not be negative")
    if (model == "ho_lee" and a != 0) or (model == "hull_white" and not a > 0):
        raise ValueError("mean_reversion must be 0.0 for 'ho_lee' and positive for 'hull_white'")
    quoted = (100.0 - float(futures_price)) / 100.0
    days = 360.0 * (t2 - t1)
    if quote_basis == "continuous":
        rate = quoted
    else:
        if not 1.0 + quoted * days / 360.0 > 0:
            raise ValueError("futures_price gives a growth factor that is not positive over the period")
        rate = 365.0 / days * np.log1p(quoted * days / 360.0)
    if model == "ho_lee":
        adjustment = 0.5 * vol * vol * t1 * t2
    else:
        def big_b(s, u):
            return -np.expm1(-a * (u - s)) / a
        span = big_b(t1, t2)
        adjustment = (span / (t2 - t1) * (span * -np.expm1(-2.0 * a * t1) + 2.0 * a * big_b(0.0, t1) ** 2)
                      * vol * vol / (4.0 * a))
    forward = rate - adjustment
    back = forward if quote_basis == "continuous" else np.expm1(forward * days / 365.0) * 360.0 / days
    return Result({"futures_rate": quoted, "futures_continuous": float(rate), "adjustment": float(adjustment),
                   "forward_continuous": float(forward), "forward_rate": float(back)})


def _smile(vol):
    """A vol for any strike: one number (a flat smile), or (strikes, vols) read linearly and flat beyond the ends."""
    if np.ndim(vol) == 0:
        level = float(vol)
        if not np.isfinite(level) or level < 0:
            raise ValueError("vol must be finite and not negative")
        return lambda strike: level
    try:
        strikes, vols = (np.asarray(part, dtype=float).ravel() for part in vol)
    except (TypeError, ValueError):
        raise ValueError("vol must be a number or a pair (strikes, vols)") from None
    if (strikes.shape != vols.shape or strikes.size < 2 or not np.all(np.isfinite(strikes))
            or np.any(np.diff(strikes) <= 0) or not np.all(np.isfinite(vols)) or np.any(vols < 0)):
        raise ValueError("vol's finite strikes must increase, with one finite, nonnegative vol each")
    return lambda strike: float(np.interp(strike, strikes, vols))


@_guard(rate=('forward_swap_rate',), time=('expiry', 'payment_time',))
def cms_convexity_adjustment(method, mapping, forward_swap_rate, annuity, discount_payment, expiry, payment_time,
                             fixed_periods, frequency, vol, vol_kind):
    """CMS convexity adjustment: the constant-maturity swap rate paid at a date, by swaption replication or Hagan's formula.

    A CMS coupon pays the swap rate S(T) fixed at T, at t_p. Its rate is (A(0) / P(0, t_p)) E^A[S a(S)], E^A under
    the annuity measure, a(S) the annuity mapping (the payment bond over the annuity as a function of S), scaled to
    a(S0) = P(0, t_p) / A(0). 'replication': CMS = S0 + (A(0) / P(0, t_p)) [int_{S0}^inf f''(K) Payer(K) dK +
    int^{S0} f''(K) Receiver(K) dK], f(K) = K a(K), the swaptions undiscounted per unit annuity at the smile's vol.
    'approximation' (Hagan): CMS = S0 + a'(S0) / a(S0) Var^A(S), Var = S0^2 (e^{v^2 T} - 1) (Black) or v^2 T (normal).
    Mappings: 'linear' a(S) = a1 S + a2, a2 = q / n (the annuity's year fractions summed); 'standard' Hagan's flat
    yield, G(S) = S (1 + S/q)^{-q (t_p - T)} / (1 - (1 + S/q)^{-n}). A constant mapping or a zero vol adds nothing.

    conventions:
    - method: 'replication' or 'approximation'; mapping: 'linear' (linear swap-rate model) or 'standard' (Hagan's)
    - forward_swap_rate: S0, the forward swap rate fixing at expiry T; annuity: A(0), its PV01 per unit of rate
    - discount_payment: P(0, t_p) to the CMS payment; payment_time: t_p, at or after expiry (the payment lag)
    - fixed_periods: n, the swap's fixed payments; frequency: q of them a year
    - vol: the swaption smile, one vol (flat) or (strikes, vols) read linearly; vol_kind: 'black' or 'normal'
    returns: a dict: cms_rate (the convexity-adjusted forward); adjustment (cms_rate - forward_swap_rate); a CMS
      swaplet is notional tau P(0, t_p) cms_rate
    source: Hagan (2003), "Convexity conundrums: pricing CMS swaps, caps, and floors", Wilmott; Andersen and
      Piterbarg, Interest Rate Modeling (2010), 16.6 (linear swap-rate model)
    aliases: CMS convexity adjustment, CMS rate, constant maturity swap, CMS swaplet, CMS replication, Hagan convexity, annuity mapping, linear swap rate model, linear TSR
    example:
        # cms_rate: 0.04029809; adjustment: 0.00029809
        qf.cms_convexity_adjustment(method="replication", mapping="standard", forward_swap_rate=0.04, annuity=4.3,
                                    discount_payment=0.95, expiry=1.0, payment_time=1.0, fixed_periods=5,
                                    frequency=1, vol=0.25, vol_kind="black")
    """
    choice("method", method, CMS_METHODS)
    choice("mapping", mapping, CMS_MAPPINGS)
    choice("vol_kind", vol_kind, VOL_KINDS)
    s0, big_t = float(forward_swap_rate), float(positive("expiry", expiry))
    a0, p_pay = float(positive("annuity", annuity)), float(positive("discount_payment", discount_payment))
    t_pay, n, q = float(payment_time), integer("fixed_periods", fixed_periods, 1), _whole_frequency(frequency)
    if not t_pay >= big_t:
        raise ValueError("payment_time must be at or after expiry")
    if vol_kind == "black" and not s0 > 0:
        raise ValueError("forward_swap_rate must be positive for Black vols")
    smile, target = _smile(vol), p_pay / a0
    if mapping == "linear":
        slope = (target - q / n) / s0

        def bend(strike):
            return 2.0 * slope
        ratio = slope / target
    else:
        lag = q * (t_pay - big_t)

        def g(strike):
            strike = np.asarray(strike, dtype=float)
            if np.any(~(1.0 + strike / q > 0)):
                raise ValueError("the standard mapping needs every strike above -frequency")
            log_u = np.log1p(strike / q)
            with np.errstate(divide="ignore", invalid="ignore"):
                level = strike / -np.expm1(-n * log_u)
            return np.where(np.abs(strike) < 1e-12, q / n, level) * np.exp(-lag * log_u)

        scale = target / float(g(s0))

        def bend(strike):
            h = CMS_STEP
            values = np.array([strike - h, strike, strike + h])
            f = values * scale * g(values)
            return float((f[0] - 2.0 * f[1] + f[2]) / (h * h))
        ratio = float((np.log(g(s0 + CMS_STEP)) - np.log(g(s0 - CMS_STEP))) / (2.0 * CMS_STEP))
    if method == "approximation":
        level = smile(s0)
        variance = s0 * s0 * np.expm1(level * level * big_t) if vol_kind == "black" else level * level * big_t
        cms = s0 + ratio * variance
        return Result({"cms_rate": float(cms), "adjustment": float(cms - s0)})

    def payer(strike):
        return float(_undiscounted_call(vol_kind, s0, strike, big_t, smile(strike)))

    def receiver(strike):
        if vol_kind == "black" and strike <= 0:
            return 0.0
        return payer(strike) - (s0 - strike)

    if vol_kind == "black":
        low, high = 0.0, np.inf
    else:
        width = CMS_SPAN * smile(s0) * np.sqrt(big_t)
        low, high = s0 - width, s0 + width
        if mapping == "standard":
            low = max(low, -q * (1.0 - 1e-6) + CMS_STEP)
    options = {"epsabs": 1e-13, "epsrel": 1e-11, "limit": 400}
    upper = integrate.quad(lambda k: bend(k) * payer(k), s0, high, **options)[0]
    lower = integrate.quad(lambda k: bend(k) * receiver(k), low, s0, **options)[0]
    cms = s0 + (upper + lower) / target
    return Result({"cms_rate": float(cms), "adjustment": float(cms - s0)})


# ------------------------------------------------------------------------------------------------ SOFR, LMM


@_guard(rate=('fixings',))
def compounded_overnight_rate(fixing_dates, fixings, start, end, day_count, lookback, lockout, observation):
    """Compounded overnight rate in arrears (SOFR, SONIA, ESTR) over an interest period: lookback, lockout, shift.

    R = [prod_i (1 + r_i n_i / B) - 1] B / D: r_i business day i's rate, n_i the calendar days it applies (three over
    a weekend; a day before the first business day takes the preceding one's), B the day basis, D the period's days.
    'lag': each day takes the rate `lookback` business days earlier, n_i and D the interest period's. 'shift'
    (observation shift): the period moved back `lookback` business days, its days weighting each rate and counting D.
    A lockout of k: the period's last k business days take the rate of the first of them (the rate cut-off date). The
    SOFR Index gives R as (I_end / I_start - 1) B / D.

    conventions:
    - fixing_dates: every business day (ISO 'YYYY-MM-DD') from `lookback` business days before start through the
      period's last business day, increasing; fixings: each day's published rate, decimals (percent / 100)
    - start, end: the interest period, start included, end (the payment or next reset date) not
    - day_count: 'ACT/360' (SOFR, ESTR) or 'ACT/365F' (SONIA)
    - lookback: business days back, 0 for none; lockout: business days locked at the end, 0 for none
    - observation: 'lag' (the rate looked back, weighted by the interest period's days) or 'shift' (the observation
      period shifted; ARRC's and ISDA's observation shift); the same where lookback is 0
    returns: a dict: rate (the annualised compounded rate, a decimal); growth (prod (1 + r_i n_i / B)); days (D)
    source: ARRC, "SOFR Floating Rate Notes Conventions Matrix" (2019); FRBNY, SOFR Averages and Index
    aliases: compounded SOFR, SOFR in arrears, compounded in arrears, SONIA compounded, ESTR compounded, OIS compounding, SOFR lookback, observation shift, lockout, compounded overnight rate
    example:
        # rate: 0.05312014; growth: 1.00103289; days: 7
        qf.compounded_overnight_rate(fixing_dates=["2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06",
                                                   "2024-03-07", "2024-03-08"],
                                     fixings=[0.0531, 0.0532, 0.0531, 0.0530, 0.0531, 0.0531], start="2024-03-04",
                                     end="2024-03-11", day_count="ACT/360", lookback=1, lockout=0, observation="lag")
    """
    choice("day_count", day_count, OVERNIGHT_DAY_COUNTS)
    choice("observation", observation, OBSERVATIONS)
    days = _dates("fixing_dates", fixing_dates).ravel()
    rates = np.asarray(fixings, dtype=float).ravel()
    if days.size == 0 or rates.shape != days.shape or np.any(np.diff(days).astype(int) <= 0) or \
            not np.all(np.isfinite(rates)):
        raise ValueError("fixing_dates must increase, with one finite rate in fixings each")
    first, last = (_dates(name, value).reshape(()) for name, value in (("start", start), ("end", end)))
    back, locked = integer("lookback", lookback, 0), integer("lockout", lockout, 0)
    basis = 360.0 if day_count == "ACT/360" else 365.0
    if not first < last:
        raise ValueError("start must come before end")
    if observation == "shift" and back:
        opens, closes = np.searchsorted(days, first), np.searchsorted(days, last)
        if opens - back < 0 or closes - back < 0:
            raise ValueError("fixing_dates must reach lookback business days before start")
        first, last, back = days[opens - back], days[closes - back], 0
    calendar = first + np.arange((last - first).astype(int))
    held = np.searchsorted(days, calendar, side="right") - 1
    if held[0] < 0:
        raise ValueError("fixing_dates must hold a business day on or before the period's first day")
    business, weights = np.unique(held, return_counts=True)
    source = business - back
    if source[0] < 0:
        raise ValueError("fixing_dates must reach lookback business days before start")
    if locked > 1:
        if business.size < locked:
            raise ValueError("lockout must not exceed the period's business days")
        source = np.where(business >= business[-locked], source[-locked], source)
    growth = float(np.prod(1.0 + rates[source] * weights / basis))
    span = int(weights.sum())
    return Result({"rate": (growth - 1.0) * basis / span, "growth": growth, "days": span})


@_guard(rate=('forwards',), time=('fixing_times',))
def lmm_vols(fixing_times, accruals, forwards, discount_factors, abcd, correlation, swaption):
    """LIBOR market model (LMM) volatility: caplet volatilities from the abcd instantaneous vol, Rebonato's swaption vol.

    Each forward F_i (fixing T_i) has the instantaneous vol s_i(t) = (a + b (T_i - t)) e^{-c (T_i - t)} + d. Its
    caplet (Black) vol is v_i with v_i^2 T_i = int_0^{T_i} s_i(t)^2 dt. Rebonato's swaption vol, the swap rate S =
    sum_i w_i F_i over the forwards from expiry T_a to maturity, w_i = tau_i P(0, T_i + tau_i) / A frozen at today:
    v^2 T_a = sum_{i,j} w_i w_j F_i F_j rho_ij int_0^{T_a} s_i s_j dt / S^2. A swaption on one forward is its caplet.
    Integrals by Gauss-Legendre quadrature, exact to rounding on these smooth integrands.

    conventions:
    - fixing_times: each forward's fixing T_i (positive, increasing); accruals: tau_i; forwards: F_i, simply
      compounded; discount_factors: P(0, T_i + tau_i) to each period's end
    - abcd: {'a': ..., 'b': ..., 'c': ..., 'd': ...} (Rebonato's parameters, c positive)
    - correlation: the forwards' positive semidefinite correlation matrix, or nonnegative beta for
      rho_ij = exp(-beta |T_i - T_j|)
    - swaption: (expiry, maturity) in years on the forwards' dates (expiry a fixing time, maturity a period's end),
      or None for the caplet vols alone
    returns: a dict: caplet_vols (one per forward); with a swaption swaption_vol (Black) and swap_rate (S)
    source: Rebonato (2002), Modern Pricing of Interest-Rate Derivatives, ch. 6 and 9; Brigo and Mercurio, Interest
      Rate Models (2006), 6.3.1 and (6.67)
    aliases: LIBOR market model, LMM, BGM, abcd volatility, Rebonato formula, Rebonato swaption volatility
    example:
        # swaption_vol: 0.18916871; swap_rate: 0.04362598
        qf.lmm_vols(fixing_times=[1.0, 2.0, 3.0, 4.0], accruals=[1.0, 1.0, 1.0, 1.0],
                    forwards=[0.04, 0.042, 0.044, 0.045], discount_factors=[0.92, 0.88, 0.85, 0.81],
                    abcd={"a": 0.05, "b": 0.09, "c": 0.44, "d": 0.11}, correlation=0.1, swaption=(2.0, 5.0))
    """
    t = _increasing("fixing_times", fixing_times)
    tau, p = positive("accruals", accruals).ravel(), positive("discount_factors", discount_factors).ravel()
    f = positive("forwards", forwards).ravel()
    if not (t.shape == tau.shape == p.shape == f.shape):
        raise ValueError("fixing_times, accruals, forwards and discount_factors must hold one entry per forward")
    if not isinstance(abcd, dict) or set(abcd) != {"a", "b", "c", "d"}:
        raise ValueError("abcd must be a dict of exactly 'a', 'b', 'c', 'd'")
    a, b, c, d = (float(abcd[key]) for key in "abcd")
    if not c > 0:
        raise ValueError("abcd's c must be positive")
    if np.ndim(correlation) == 0:
        beta = float(correlation)
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("correlation's decay beta must be finite and not negative")
        rho = np.exp(-beta * np.abs(t[:, None] - t[None, :]))
    else:
        rho = np.asarray(correlation, dtype=float)
        if (rho.shape != (t.size, t.size) or not np.all(np.isfinite(rho))
                or not np.allclose(rho, rho.T, rtol=1e-10, atol=1e-12)
                or not np.allclose(np.diag(rho), 1.0, rtol=1e-10, atol=1e-12)
                or np.any(np.abs(rho) > 1.0 + 1e-12)
                or np.linalg.eigvalsh(rho).min() < -1e-10):
            raise ValueError("correlation must be a nonnegative decay beta, or a finite symmetric positive "
                             "semidefinite matrix with a unit diagonal and entries in [-1, 1], one row per forward")
    nodes, weights = np.polynomial.legendre.leggauss(LEGENDRE)

    def vol(i, s):
        x = t[i] - s
        return (a + b * x) * np.exp(-c * x) + d

    def integral(i, j, horizon):
        s = 0.5 * horizon * (nodes + 1.0)
        return float(0.5 * horizon * weights @ (vol(i, s) * vol(j, s)))

    caplets = np.array([np.sqrt(integral(i, i, t[i]) / t[i]) for i in range(t.size)])
    if swaption is None:
        return Result({"caplet_vols": caplets})
    try:
        expiry, maturity = (float(x) for x in swaption)
    except (TypeError, ValueError):
        raise ValueError("swaption must be (expiry, maturity) or None") from None
    ends = t + tau
    first, last = np.flatnonzero(np.isclose(t, expiry)), np.flatnonzero(np.isclose(ends, maturity))
    if first.size != 1 or last.size != 1 or last[0] < first[0]:
        raise ValueError("swaption's expiry must be a fixing time and its maturity a later period's end")
    span = np.arange(first[0], last[0] + 1)
    w = tau[span] * p[span]
    w = w / w.sum()
    rate = float(w @ f[span])
    moments = np.array([[integral(i, j, expiry) for j in span] for i in span])
    weighted = w * f[span]
    variance = float(weighted @ (rho[np.ix_(span, span)] * moments) @ weighted) / rate ** 2
    return Result({"caplet_vols": caplets, "swaption_vol": float(np.sqrt(max(variance, 0.0) / expiry)), "swap_rate": rate})
