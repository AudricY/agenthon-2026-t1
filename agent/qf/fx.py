"""FX: value dates, CIP forwards and points, swaps, cross rates, xccy basis, Garman-Kohlhagen deltas, smile, vanna-volga.

`import qf`. A pair FOR/DOM is quoted as DOM per unit of FOR (EUR/USD 1.10: USD per EUR); the domestic rate r_d
discounts DOM, the foreign rate r_f FOR, both continuously compounded and annual unless a function names another
basis; the forward F = S e^{(r_d - r_f)T}; an option's price is qf.bs_price with dividend_yield = r_f, in DOM per unit
of FOR. A function with several results returns a dict with named keys.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtri

from ._common import Result, choice, ndtr, out, positive
from .curves import INTERPOLATIONS, _curve, _increasing
from .impvol import implied_vol
from .options import bs_price
from .primitives import _guard

__all__ = ["cip_basis", "cip_forward", "cross_rate", "fx_delta", "fx_swap_outrights", "fx_value_dates",
           "smile_from_quotes", "strike_from_delta", "vanna_volga", "xccy_basis_swap"]

OPTIONS = ("call", "put")
DELTA_TYPES = ("spot", "forward", "spot_premium_adjusted", "forward_premium_adjusted")
ATM_RULES = ("forward", "spot", "delta_neutral")
BUTTERFLIES = ("smile_strangle", "market_strangle")
WIDTH = 8.0                     # a strike search starts within this many vol sqrt(T) of ln F in log strike
WIDEST = 64.0                   # and widens, doubling, to this many before it refuses
DAY_COUNTS = {"ACT/360": 360.0, "ACT/365": 365.0}
COMPOUNDINGS = ("simple", "continuous")
MID_RULES = ("legs", "cross")
BASIS_LEGS = ("base", "quote")
SPREAD_LEGS = ("domestic", "foreign")
NOTIONAL_TYPES = ("constant", "mtm_reset")
XCCY_SIDES = ("receive_foreign", "pay_foreign")


def _d1_d2(forward, strike, maturity, vol):
    sd = vol * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * sd * sd) / sd
    return d1, d1 - sd


def _delta(phi, forward, strike, maturity, foreign_rate, vol, delta_type):
    d1, d2 = _d1_d2(forward, strike, maturity, vol)
    adjusted = delta_type.endswith("premium_adjusted")
    found = phi * (strike / forward * ndtr(phi * d2) if adjusted else ndtr(phi * d1))
    return found * np.exp(-foreign_rate * maturity) if delta_type.startswith("spot") else found


@_guard(rate=('domestic_rate', 'foreign_rate',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def fx_delta(option, spot, strike, maturity, domestic_rate, foreign_rate, vol, delta_type):
    """FX option delta under the four quoting conventions (Garman-Kohlhagen).

    phi = +1 call, -1 put; d1,2 = (ln(F/K) +- vol^2 T/2) / (vol sqrt(T)). Forward delta phi N(phi d1); spot delta
    phi e^{-r_f T} N(phi d1), dV/dS; premium-adjusted forward delta phi (K/F) N(phi d2); premium-adjusted spot delta
    phi e^{-r_f T} (K/F) N(phi d2) = spot delta - V/S, the premium V paid in FOR taken off (it equals phi e^{-r_d T}
    (K/S) N(phi d2), not e^{-r_f T} (K/S) N(phi d2)). Call less put: 1 (forward), e^{-r_f T} (spot), K/F (forward,
    premium-adjusted). Delta-neutral ATM strikes, where call and put deltas cancel: F e^{vol^2 T/2}, and F e^{-vol^2
    T/2} premium-adjusted; ATM forward K = F.

    conventions:
    - option: 'call' or 'put' on FOR; spot: S, DOM per unit of FOR; strike: K in the same quote
    - domestic_rate, foreign_rate: r_d and r_f, continuously compounded; maturity: T in years; vol: at the strike
    - delta_type: 'spot', 'forward', 'spot_premium_adjusted' or 'forward_premium_adjusted' (premium paid in FOR, the
      convention of USD/JPY, say)
    returns: the delta in FOR per unit of FOR notional (0.25 is 25 delta; a put's is negative), a float or an array
    source: Reiswich and Wystup (2010), J. Derivatives 18(2); Wystup (2006), FX Options and Structured Products; Clark (2011)
    aliases: FX option delta, Garman-Kohlhagen delta, premium-adjusted delta, delta convention, delta quoting convention, delta-neutral straddle, ATM DNS, [FX delta]
    example:
        qf.fx_delta(option="call", spot=1.10, strike=1.15, maturity=0.5, domestic_rate=0.045,
                    foreign_rate=0.03, vol=0.08,
                    delta_type="spot_premium_adjusted")
    """
    choice("option", option, OPTIONS)
    choice("delta_type", delta_type, DELTA_TYPES)
    s, k, t = positive("spot", spot), positive("strike", strike), positive("maturity", maturity)
    sigma = positive("vol", vol)
    forward = s * np.exp((np.asarray(domestic_rate, dtype=float) - np.asarray(foreign_rate, dtype=float)) * t)
    return out(_delta(1.0 if option == "call" else -1.0, forward, k, t, np.asarray(foreign_rate, dtype=float), sigma,
                      delta_type))


def _vol_at(vol):
    """A callable of the strike, from a number (flat) or a callable (a smile)."""
    if callable(vol):
        return lambda k: float(positive("vol(strike)", vol(k)))
    flat = float(positive("vol", vol))
    return lambda k: flat


def _strike(phi, delta, forward, t, r_f, vol, delta_type):
    sigma = vol(forward)
    sd = sigma * np.sqrt(t)
    target = delta * np.exp(r_f * t) if delta_type.startswith("spot") else delta
    adjusted = delta_type.endswith("premium_adjusted")
    if not adjusted and not 0 < phi * target < 1:
        raise ValueError(f"delta {delta!r} is not a {'call' if phi > 0 else 'put'}'s {delta_type} delta")
    if adjusted and not phi * target > 0:
        raise ValueError(f"delta {delta!r} is not a {'call' if phi > 0 else 'put'}'s {delta_type} delta")

    def gap(x):
        k = forward * np.exp(x)
        return float(_delta(phi, forward, k, t, r_f, vol(k), delta_type)) - delta

    width = WIDTH
    while True:
        low, high = -width * sd, width * sd
        if adjusted and phi > 0:
            peak = minimize_scalar(lambda x: -gap(x), bounds=(low, high), method="bounded",
                                   options={"xatol": 1e-12}).x
            if gap(peak) < 0:
                raise ValueError(f"delta {delta!r} lies above the premium-adjusted call delta's maximum, "
                                 f"{gap(peak) + delta:.6g}: no strike gives it")
            low = peak
        if gap(low) >= 0 >= gap(high):
            return float(forward * np.exp(brentq(gap, low, high, xtol=1e-15, rtol=1e-15)))
        width *= 2
        if width > WIDEST:
            raise ValueError(f"no strike gives delta {delta!r} within {WIDEST:g} vol sqrt(T) of the forward")


@_guard(rate=('domestic_rate', 'foreign_rate',), vol=('vol',), time=('maturity',), price=('spot',))
def strike_from_delta(option, delta, spot, maturity, domestic_rate, foreign_rate, vol, delta_type):
    """FX strike from a delta under its quoting convention, the vol flat or a smile in K.

    Inverts qf.fx_delta. Forward delta in closed form: K = F exp(-phi N^-1(phi Delta) vol sqrt(T) + vol^2 T/2),
    a spot delta first divided by e^{-r_f T}. A premium-adjusted delta phi (K/F) N(phi d2) has no closed form: its
    root in the strike. For a call it rises then falls in K, so a delta may have two strikes: the larger, on the
    falling side, is the market's (Reiswich and Wystup's bracket between the maximum and the unadjusted strike); a
    delta above the maximum is refused. With a smile, vol(K) is read at each trial strike, so the strike and its vol
    agree (25-delta pillars). Delta-neutral ATM: F e^{vol^2 T/2}, premium-adjusted F e^{-vol^2 T/2}.

    conventions:
    - option: 'call' or 'put' on FOR; delta: in FOR per unit of FOR, a call's positive and a put's negative (0.25 for
      25 delta, -0.1 for a 10-delta put)
    - spot, maturity, domestic_rate, foreign_rate, delta_type: as fx_delta
    - vol: a number (flat), or a callable vol(strike) giving the smile's vol at a strike
    returns: the strike K (DOM per unit of FOR), a float
    source: Reiswich and Wystup (2010), J. Derivatives 18(2), section 3; Clark (2011), Foreign Exchange Option Pricing
    aliases: strike from delta, delta to strike, FX strike from delta, 25-delta strike, 10-delta strike, premium-adjusted strike, delta strike conversion
    example:
        qf.strike_from_delta(option="put", delta=-0.25, spot=1.10, maturity=0.5, domestic_rate=0.045,
                             foreign_rate=0.03, vol=0.085,
                             delta_type="forward_premium_adjusted")
    """
    choice("option", option, OPTIONS)
    choice("delta_type", delta_type, DELTA_TYPES)
    s, t = float(positive("spot", spot)), float(positive("maturity", maturity))
    r_d, r_f = float(domestic_rate), float(foreign_rate)
    forward = s * np.exp((r_d - r_f) * t)
    phi = 1.0 if option == "call" else -1.0
    level = _vol_at(vol)
    target = float(delta)
    if not callable(vol) and not delta_type.endswith("premium_adjusted"):
        sd = level(forward) * np.sqrt(t)
        forward_delta = target * np.exp(r_f * t) if delta_type == "spot" else target
        if not 0 < phi * forward_delta < 1:
            raise ValueError(f"delta {delta!r} is not a {option}'s {delta_type} delta")
        return float(forward * np.exp(-phi * ndtri(phi * forward_delta) * sd + 0.5 * sd * sd))
    return _strike(phi, target, forward, t, r_f, level, delta_type)


def _bracketed_root(gap, centre, step):
    """The root of gap nearest centre: steps out both ways, growing, past points where gap is undefined (nan)."""
    here = gap(centre)
    if here == 0:
        return centre
    for _ in range(60):
        for side in (centre - step, centre + step):
            there = gap(side)
            if np.isfinite(there) and np.sign(there) != np.sign(here):
                low, high = sorted((centre, side))
                return brentq(gap, low, high, xtol=1e-15, rtol=1e-15)
        step *= 1.5
    raise ValueError("no smile butterfly reprices the market strangle")


def _parabola(points):
    """The vol as a parabola in x = ln(K/F) through three (x, vol) points."""
    x, v = zip(*points)
    return np.polyfit(np.array(x), np.array(v), 2)


@_guard(rate=('domestic_rate', 'foreign_rate',), vol=('atm_vol',), time=('maturity',), price=('spot',))
def smile_from_quotes(spot, maturity, domestic_rate, foreign_rate, atm_vol, risk_reversal, butterfly, delta,
                      delta_type, atm, butterfly_type, strikes):
    """FX smile from ATM, risk-reversal and butterfly quotes: the pillar strikes and vols.

    'smile_strangle': vol_call = vol_ATM + BF + RR/2 and vol_put = vol_ATM + BF - RR/2 at the delta's call and put
    strikes, each strike solved at its own vol (qf.strike_from_delta). 'market_strangle' (Clark, ch. 3; Reiswich
    and Wystup 2012): the quoted strangle, a call and a put at +-delta both at vol_ATM + BF, priced at that one vol,
    must equal the same two strikes priced on the smile: the smile's butterfly is solved so they match, RR kept. The
    smile between the pillars is a parabola in ln(K/F) through the three of them, the ATM strike's vol vol_ATM: the
    market strangle's answer depends on that choice. RR is call minus put.

    conventions:
    - spot, maturity, domestic_rate, foreign_rate: as fx_delta; atm_vol, risk_reversal, butterfly: decimals (an RR
      of -1.2 vols is -0.012)
    - delta: the pillars' delta, 0.25 or 0.1; delta_type: as fx_delta, for the pillars and the ATM rule
    - atm: 'forward' (K = F), 'spot' (K = S) or 'delta_neutral' (call and put deltas cancel: F e^{vol^2 T/2},
      F e^{-vol^2 T/2} premium-adjusted)
    - butterfly_type: 'smile_strangle' or 'market_strangle'; strikes: where to read the smile, or None
    returns: a dict: atm_strike, call_strike, put_strike and atm_vol, call_vol, put_vol (the pillars); smile_butterfly
      ((call_vol + put_vol)/2 - atm_vol); vols (the smile at strikes, None without them)
    source: Clark (2011), Foreign Exchange Option Pricing, ch. 3; Reiswich and Wystup (2012), Wilmott; Castagna (2010)
    aliases: FX smile, risk reversal, butterfly, market strangle, smile strangle, 25-delta risk reversal, 25-delta butterfly, RR and BF
    example:
        qf.smile_from_quotes(spot=1.10, maturity=0.5, domestic_rate=0.045, foreign_rate=0.03, atm_vol=0.08,
            risk_reversal=-0.012, butterfly=0.003, delta=0.25, delta_type="spot", atm="delta_neutral",
            butterfly_type="market_strangle", strikes=[1.0, 1.1, 1.2])
    """
    choice("delta_type", delta_type, DELTA_TYPES)
    choice("atm", atm, ATM_RULES)
    choice("butterfly_type", butterfly_type, BUTTERFLIES)
    s, t = float(positive("spot", spot)), float(positive("maturity", maturity))
    r_d, r_f = float(domestic_rate), float(foreign_rate)
    sigma, rr, bf, level = float(positive("atm_vol", atm_vol)), float(risk_reversal), float(butterfly), float(delta)
    if not 0 < level < 0.5:
        raise ValueError("delta must be the pillars' delta, inside (0, 0.5): 0.25 or 0.1")
    forward = s * np.exp((r_d - r_f) * t)
    adjusted = delta_type.endswith("premium_adjusted")
    k_atm = {"forward": forward, "spot": s,
             "delta_neutral": forward * np.exp((-0.5 if adjusted else 0.5) * sigma * sigma * t)}[atm]

    def pillars(fly):
        vol_c, vol_p = sigma + fly + 0.5 * rr, sigma + fly - 0.5 * rr
        if not (vol_c > 0 and vol_p > 0):
            raise ValueError("the quotes give a pillar a vol that is not positive")
        k_c = strike_from_delta("call", level, s, t, r_d, r_f, vol_c, delta_type)
        k_p = strike_from_delta("put", -level, s, t, r_d, r_f, vol_p, delta_type)
        return (k_p, vol_p), (k_c, vol_c)

    def smile(fly):
        (k_p, vol_p), (k_c, vol_c) = pillars(fly)
        return _parabola([(np.log(k_p / forward), vol_p), (np.log(k_atm / forward), sigma),
                          (np.log(k_c / forward), vol_c)])

    fly = bf
    if butterfly_type == "market_strangle":
        wide = sigma + bf
        k_c = strike_from_delta("call", level, s, t, r_d, r_f, wide, delta_type)
        k_p = strike_from_delta("put", -level, s, t, r_d, r_f, wide, delta_type)
        quoted = bs_price("call", s, k_c, t, r_d, r_f, wide) + bs_price("put", s, k_p, t, r_d, r_f, wide)

        def gap(trial):
            try:
                curve = smile(trial)
            except ValueError:
                return np.nan
            vol_c, vol_p = np.polyval(curve, np.log(k_c / forward)), np.polyval(curve, np.log(k_p / forward))
            if not (vol_c > 0 and vol_p > 0):
                return np.nan
            return bs_price("call", s, k_c, t, r_d, r_f, vol_c) + bs_price("put", s, k_p, t, r_d, r_f, vol_p) - quoted

        fly = _bracketed_root(gap, bf, 1e-4 + 0.1 * (abs(bf) + abs(rr)))
    (k_p, vol_p), (k_c, vol_c) = pillars(fly)
    curve = smile(fly)
    vols = None if strikes is None else out(np.polyval(curve, np.log(positive("strikes", strikes) / forward)))
    return Result({"atm_strike": float(k_atm), "call_strike": k_c, "put_strike": k_p, "atm_vol": sigma,
                   "call_vol": vol_c, "put_vol": vol_p, "smile_butterfly": fly, "vols": vols})


VV_VARIANTS = ("price", "first_order", "second_order")


@_guard(rate=('domestic_rate', 'foreign_rate',), vol=('pillar_vols',), time=('maturity',),
        price=('spot', 'strike', 'pillar_strikes',))
def vanna_volga(option, strike, spot, maturity, domestic_rate, foreign_rate, pillar_strikes, pillar_vols, variant):
    """Vanna-volga (Castagna-Mercurio): the smile from three pillars.

    K1 < K2 < K3 the pillars' strikes with vols v1, v2, v3 (v2 the ATM vol, the flat vol of reference); y1 =
    ln(K2/K) ln(K3/K) / [ln(K2/K1) ln(K3/K1)], y2 = ln(K/K1) ln(K3/K) / [ln(K2/K1) ln(K3/K2)], y3 = ln(K/K1) ln(K/K2)
    / [ln(K3/K1) ln(K3/K2)]. 'price': C(K) = C_BS(K, v2) + sum_i x_i [C_BS(K_i, v_i) - C_BS(K_i, v2)], x_i = y_i
    vega(K) / vega(K_i) at v2, the weights whose three options hedge the option's vega, vanna and volga; its vol by
    inverting. 'first_order': vol = y1 v1 + y2 v2 + y3 v3; 'second_order': vol = v2 + [-v2 + sqrt(v2^2 + d1 d2 (2 v2
    D1 + D2))] / (d1 d2), D1 = y1 v1 + y2 v2 + y3 v3 - v2, D2 = y1 d1(K1) d2(K1) (v1 - v2)^2 + y3 d1(K3) d2(K3) (v3 -
    v2)^2, d1, d2 Black-Scholes's at v2 (Castagna and Mercurio's first- and second-order expansions). Each reprices the pillars; a flat
    smile gives Black-Scholes. The pillars from quotes: qf.smile_from_quotes.

    conventions:
    - option: 'call' or 'put' on the foreign currency; strike: K (a number or an array), domestic per foreign
    - spot: S, domestic per foreign; domestic_rate r_d, foreign_rate r_f continuously compounded; maturity T in years
    - pillar_strikes: [K1, K2, K3] increasing (25-delta put, ATM, 25-delta call); pillar_vols: their vols [v1, v2, v3]
    - variant: 'price' (the hedge's price, its implied vol), 'first_order' or 'second_order' (the vol approximations,
      priced at that vol)
    returns: a dict: price (domestic per unit of foreign); vol (its Black-Scholes vol); each a float or an array
    source: Castagna and Mercurio (2007), "The vanna-volga method for implied volatilities", Risk 20(1); Wystup
      (2006), FX Options and Structured Products, "Vanna-volga pricing"
    aliases: vanna-volga, vanna volga, Castagna-Mercurio
    example:
        import qf
        qf.vanna_volga(option="call", strike=1.25, spot=1.2, maturity=0.25, domestic_rate=0.04,
                       foreign_rate=0.02, pillar_strikes=[1.16, 1.21, 1.26],
                       pillar_vols=[0.095, 0.087, 0.089], variant="price")  # result: price 0.00624395, vol 0.08785093
    """
    choice("option", option, OPTIONS)
    choice("variant", variant, VV_VARIANTS)
    k = positive("strike", strike)
    s, t = float(positive("spot", spot)), float(positive("maturity", maturity))
    r_d, r_f = float(domestic_rate), float(foreign_rate)
    pk, pv = positive("pillar_strikes", pillar_strikes).ravel(), positive("pillar_vols", pillar_vols).ravel()
    if pk.size != 3 or pv.size != 3 or np.any(np.diff(pk) <= 0):
        raise ValueError("pillar_strikes must be three increasing strikes and pillar_vols their three vols")
    forward, atm = s * np.exp((r_d - r_f) * t), pv[1]
    l1, l2, l3 = (np.log(x) for x in pk)
    lk = np.log(k)
    y = np.array([(l2 - lk) * (l3 - lk) / ((l2 - l1) * (l3 - l1)), (lk - l1) * (l3 - lk) / ((l2 - l1) * (l3 - l2)),
                  (lk - l1) * (lk - l2) / ((l3 - l1) * (l3 - l2))])
    first = np.tensordot(pv, y, axes=1)
    if variant == "price":
        d1 = _d1_d2(forward, k, t, atm)[0]
        vega = np.exp(-0.5 * d1 * d1)
        legs = 0.0
        for i in range(3):
            d1_i = _d1_d2(forward, pk[i], t, atm)[0]
            cost = bs_price("call", s, pk[i], t, r_d, r_f, pv[i]) - bs_price("call", s, pk[i], t, r_d, r_f, atm)
            legs = legs + y[i] * vega / np.exp(-0.5 * d1_i * d1_i) * cost
        price = np.asarray(bs_price(option, s, k, t, r_d, r_f, atm)) + legs        # the hedge costs a put as a call
        vol = implied_vol(option, price, s, k, t, r_d, r_f, "bisection")
        return Result({"price": out(price), "vol": out(vol)})
    if variant == "first_order":
        vol = first
    else:
        d1, d2 = _d1_d2(forward, k, t, atm)
        p1, p3 = _d1_d2(forward, pk[0], t, atm), _d1_d2(forward, pk[2], t, atm)
        lead = first - atm
        second = y[0] * p1[0] * p1[1] * (pv[0] - atm) ** 2 + y[2] * p3[0] * p3[1] * (pv[2] - atm) ** 2
        product = d1 * d2
        with np.errstate(divide="ignore", invalid="ignore"):
            vol = np.where(np.abs(product) > 1e-12,
                           atm + (-atm + np.sqrt(atm * atm + product * (2.0 * atm * lead + second))) / product,
                           atm + lead + second / (2.0 * atm))
    return Result({"price": out(bs_price(option, s, k, t, r_d, r_f, vol)), "vol": out(vol)})


@_guard(rate=('base_rate', 'quote_rate',), price=('spot',))
def cip_forward(spot, base_rate, quote_rate, days, base_day_count, quote_day_count, compounding, pip_multiplier):
    """Covered-interest-parity FX forward and its forward points (pips) from the two deposit rates.

    For a pair BASE/QUOTE (QUOTE units per BASE unit), tau = days / 360 or days / 365 per currency's day count:
    'simple' F = S (1 + r_quote tau_quote) / (1 + r_base tau_base) (money-market deposits), 'continuous' F = S
    exp(r_quote tau_quote - r_base tau_base); points = (F - S) pip_multiplier.

    A forward bought at F0 on N BASE is worth N (F - F0) DF today, in QUOTE, DF the QUOTE discount factor to its
    value date. A non-deliverable forward settles N (fixing - F0) in QUOTE, or N (fixing - F0) / fixing in BASE.

    conventions:
    - spot: QUOTE per BASE (EUR/USD 1.08 is 1.08 USD per EUR); a bid or ask spot gives that side's outright
    - base_rate, quote_rate: decimals (convert percent first); days: calendar days to the forward's value date
    - base_day_count, quote_day_count: 'ACT/360' or 'ACT/365' each; pip_multiplier: 10,000 for most pairs, 100 for JPY
    returns: a dict: forward (QUOTE units per BASE unit); points ((F - S) pip_multiplier)
    source: Keynes (1923), A Tract on Monetary Reform; Hull, Options, Futures, and Other Derivatives, ch. 5 (5.10)
    aliases: covered interest parity, covered interest rate parity, FX forward points, [CIP], [FX forward], [forward points], [swap points], [outright forward], [interest parity]
    example:
        qf.cip_forward(spot=1.1200, base_rate=0.0325, quote_rate=0.0450, days=91, base_day_count="ACT/360",
                       quote_day_count="ACT/360", compounding="simple",
                       pip_multiplier=10000.0)
    """
    choice("base_day_count", base_day_count, tuple(DAY_COUNTS))
    choice("quote_day_count", quote_day_count, tuple(DAY_COUNTS))
    choice("compounding", compounding, COMPOUNDINGS)
    s = positive("spot", spot)
    n = np.asarray(days, dtype=float)
    if np.any(n < 0):
        raise ValueError("days must not be negative")
    tau_b, tau_q = n / DAY_COUNTS[base_day_count], n / DAY_COUNTS[quote_day_count]
    r_b, r_q = np.asarray(base_rate, dtype=float), np.asarray(quote_rate, dtype=float)
    if compounding == "simple":
        forward = s * (1.0 + r_q * tau_q) / (1.0 + r_b * tau_b)
    else:
        forward = s * np.exp(r_q * tau_q - r_b * tau_b)
    return Result({"forward": out(forward),
                   "points": out((forward - s) * float(positive("pip_multiplier", pip_multiplier)))})


@_guard(rate=('base_rate', 'quote_rate',), price=('spot', 'forward',))
def cip_basis(spot, forward, base_rate, quote_rate, days, base_day_count, quote_day_count, compounding, basis_leg):
    """Cross-currency basis: the CIP deviation from spot, forward and the two rates.

    For a pair BASE/QUOTE (QUOTE units per BASE unit), tau = days / 360 or / 365 by each currency's day count. The
    basis x is what the named leg's rate needs for covered interest parity to hold at the quoted forward: 'simple'
    F / S = (1 + r_q tau_q) / (1 + (r_b + x) tau_b) ('base') or (1 + (r_q + x) tau_q) / (1 + r_b tau_b) ('quote');
    'continuous' F / S = e^{r_q tau_q - (r_b + x) tau_b} or e^{(r_q + x) tau_q - r_b tau_b}. Parity holding, x = 0.
    Du, Tepper and Verdelhan's basis is x on the non-dollar currency's leg (negative: dollars dearer synthetically).

    conventions:
    - spot, forward: QUOTE per BASE (EUR/USD 1.08 is 1.08 USD per EUR), the outright forward to the same date
    - base_rate, quote_rate: decimals; days: calendar days to the value date; day counts 'ACT/360' or 'ACT/365' each
    - compounding: 'simple' (money-market deposits) or 'continuous'
    - basis_leg: 'base' or 'quote', the currency whose rate carries x (added to it)
    returns: a dict: basis (x, a decimal); implied_rate (that leg's rate plus x); parity_forward (the forward at x = 0,
      cip_forward's)
    source: Du, Tepper and Verdelhan (2018), J. Finance 73(3); Hull, Options, Futures, and Other Derivatives, ch. 5
    aliases: cross-currency basis, xccy basis, CIP deviation, covered interest parity deviation, CIP violation, FX swap basis, synthetic dollar rate
    example:
        qf.cip_basis(spot=1.0800, forward=1.0890, base_rate=0.035, quote_rate=0.050, days=182,
                     base_day_count="ACT/360", quote_day_count="ACT/360", compounding="simple",
                     basis_leg="base")
    """
    choice("base_day_count", base_day_count, tuple(DAY_COUNTS))
    choice("quote_day_count", quote_day_count, tuple(DAY_COUNTS))
    choice("compounding", compounding, COMPOUNDINGS)
    choice("basis_leg", basis_leg, BASIS_LEGS)
    s, f = positive("spot", spot), positive("forward", forward)
    n = positive("days", days)
    tau_b, tau_q = n / DAY_COUNTS[base_day_count], n / DAY_COUNTS[quote_day_count]
    r_b, r_q = np.asarray(base_rate, dtype=float), np.asarray(quote_rate, dtype=float)
    if compounding == "simple":
        x = ((1.0 + r_q * tau_q) * s / f - 1.0) / tau_b - r_b if basis_leg == "base" else \
            ((f / s) * (1.0 + r_b * tau_b) - 1.0) / tau_q - r_q
        parity = s * (1.0 + r_q * tau_q) / (1.0 + r_b * tau_b)
    else:
        x = (r_q * tau_q - np.log(f / s)) / tau_b - r_b if basis_leg == "base" else \
            (np.log(f / s) + r_b * tau_b) / tau_q - r_q
        parity = s * np.exp(r_q * tau_q - r_b * tau_b)
    return Result({"basis": out(x), "implied_rate": out((r_b if basis_leg == "base" else r_q) + x),
                   "parity_forward": out(parity)})


@_guard(time=('start', 'payment_times',), price=('spot',))
def xccy_basis_swap(spot, foreign_notional, start, payment_times, domestic_accruals, foreign_accruals,
                    domestic_discount, domestic_projection, foreign_discount, foreign_projection, interpolation, spread,
                    spread_leg, notional_type, side):
    """Cross-currency basis swap: two floating legs, notionals exchanged, the basis on one leg.

    Each leg is projected and discounted on its own currency's curves: PV = N [-D(T_0) + sum_j (P(t_{j-1}) / P(t_j) -
    1 + s tau_j) D(t_j) + D(T_n)], the notional paid at T_0 and returned at T_n, s the spread on the named leg. The
    domestic notional is S N_f; 'mtm_reset' resets it each period to N_f X(t_{j-1}), X(t) = S D_f(t) / D_d(t) the FX
    forward, its difference exchanged, so period j is worth N_f S D_f(t_{j-1}) [-1 + (P_d(t_{j-1}) / P_d(t_j) + s
    tau_j) D_d(t_j) / D_d(t_{j-1})]. Value (domestic) = S PV_foreign - PV_domestic receiving the foreign leg; the par
    spread zeroes it. With one curve a currency and no spread, every leg is worth 0.

    conventions:
    - spot: domestic units per foreign unit; foreign_notional: N_f; start: T_0; payment_times: both legs' period ends
    - domestic_accruals, foreign_accruals: each leg's tau_j by its day count
    - each curve (node_times, node_discounts), read by interpolation (as curve_discount); a projection None: its
      currency's discount curve projects too
    - spread: s; spread_leg: 'domestic' or 'foreign', the leg that carries it
    - notional_type: 'constant' or 'mtm_reset' (the domestic leg resets); side: 'receive_foreign' or 'pay_foreign'
    returns: a dict: value (domestic units); par_spread (on spread_leg); foreign_leg (its PV in foreign units);
      domestic_leg (its PV in domestic units)
    source: Andersen and Piterbarg, Interest Rate Modeling (2010), 15.2; Fujii, Shimada and Takahashi (2010)
    aliases: cross-currency basis swap, xccy basis swap, cross currency swap, currency basis swap, mark-to-market cross-currency swap, MtM resetting swap, [CCS]
    example:
        qf.xccy_basis_swap(spot=1.10, foreign_notional=1e6, start=0.0, payment_times=[0.5, 1.0], domestic_accruals=[0.5, 0.5],
            foreign_accruals=[0.5, 0.5], domestic_discount=([1.0], [0.96]), domestic_projection=([1.0], [0.955]), foreign_discount=([1.0], [0.975]),
            foreign_projection=([1.0], [0.97]), interpolation="log_linear_discount", spread=-0.002, spread_leg="foreign", notional_type="constant", side="receive_foreign")
    """
    choice("interpolation", interpolation, INTERPOLATIONS)
    choice("spread_leg", spread_leg, SPREAD_LEGS)
    choice("notional_type", notional_type, NOTIONAL_TYPES)
    choice("side", side, XCCY_SIDES)
    s, n_f, t0 = float(positive("spot", spot)), float(positive("foreign_notional", foreign_notional)), float(start)
    times = _increasing("payment_times", payment_times)
    tau_d, tau_f = positive("domestic_accruals", domestic_accruals).ravel(), positive("foreign_accruals",
                                                                                     foreign_accruals).ravel()
    if tau_d.shape != times.shape or tau_f.shape != times.shape or not 0 <= t0 < times[0]:
        raise ValueError("one accrual per payment time on each leg, and start before the first payment")
    d_d = _curve("domestic_discount", domestic_discount, interpolation)
    d_f = _curve("foreign_discount", foreign_discount, interpolation)
    p_d = d_d if domestic_projection is None else _curve("domestic_projection", domestic_projection, interpolation)
    p_f = d_f if foreign_projection is None else _curve("foreign_projection", foreign_projection, interpolation)
    bounds = np.r_[t0, times]
    ends = times

    def periods(d, p, tau):
        """Each period's floating coupon (P(t_{j-1}) / P(t_j) - 1) and the annuity term tau_j D(t_j), discounted."""
        projected = p(bounds)
        return (projected[:-1] / projected[1:] - 1.0) * d(ends), tau * d(ends)

    float_f, annuity_f = periods(d_f, p_f, tau_f)
    float_d, annuity_d = periods(d_d, p_d, tau_d)
    foreign_zero = n_f * (float_f.sum() + d_f(ends[-1:])[0] - d_f(np.array([t0]))[0])
    if notional_type == "constant":
        notional = np.full(times.size, s * n_f)
        domestic_zero = s * n_f * (float_d.sum() + d_d(ends[-1:])[0] - d_d(np.array([t0]))[0])
    else:
        notional = n_f * s * d_f(bounds[:-1]) / d_d(bounds[:-1])           # N_f X(t_{j-1})
        domestic_zero = float(notional @ (float_d + d_d(ends) - d_d(bounds[:-1])))
    slope_f, slope_d = n_f * annuity_f.sum(), float(notional @ annuity_d)
    spread_value = float(spread)
    foreign_leg = foreign_zero + (spread_value * slope_f if spread_leg == "foreign" else 0.0)
    domestic_leg = domestic_zero + (spread_value * slope_d if spread_leg == "domestic" else 0.0)
    sign = 1.0 if side == "receive_foreign" else -1.0
    value_zero = s * foreign_zero - domestic_zero
    slope = s * slope_f if spread_leg == "foreign" else -slope_d
    return Result({"value": float(sign * (s * foreign_leg - domestic_leg)), "par_spread": float(-value_zero / slope),
                   "foreign_leg": float(foreign_leg), "domestic_leg": float(domestic_leg)})


def _pair(name, pair):
    parts = pair.split("/") if isinstance(pair, str) else []
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"{name} must read 'BASE/QUOTE', such as 'EUR/USD', not {pair!r}")
    return parts[0].strip().upper(), parts[1].strip().upper()


@_guard(price=('first_bid', 'first_ask', 'second_bid', 'second_ask',))
def cross_rate(first_bid, first_ask, first_pair, second_bid, second_ask, second_pair, cross, mid_rule):
    """Cross rate bid, ask and mid from two pairs that share a currency, sides kept executable.

    For the cross X/Y through the shared currency C, the leg with X is turned to X/C and the leg with Y to C/Y,
    inverting a leg quoted the other way with its sides swapped (bid 1/ask, ask 1/bid); then bid = (X/C bid)(C/Y bid)
    and ask = (X/C ask)(C/Y ask): selling X for Y goes through both legs at their bids. mid: 'legs' multiplies the
    legs' mids, an inverted leg's mid being 1 / its quoted mid; 'cross' is (bid + ask) / 2.

    conventions:
    - first_pair, second_pair, cross: 'BASE/QUOTE' codes, such as 'GBP/USD', 'USD/JPY', 'GBP/JPY'; one currency shared
    - bids and asks as quoted for their own pairs (QUOTE per BASE), bid below ask
    - mid_rule: 'legs' (the product of the legs' mids) or 'cross' (the average of the cross bid and ask)
    returns: a dict: bid, ask and mid of the cross
    source: Shapiro, Multinational Financial Management, ch. 7 (cross rates); Hull, ch. 5 (foreign exchange quotes)
    aliases: cross rate, cross rates, triangulation, synthetic cross, currency cross, bid ask cross, FX cross
    example:
        qf.cross_rate(first_bid=1.3110, first_ask=1.3114, first_pair="GBP/USD", second_bid=143.80,
                      second_ask=143.84, second_pair="USD/JPY", cross="GBP/JPY",
                      mid_rule="legs")
    """
    choice("mid_rule", mid_rule, MID_RULES)
    legs = [(_pair("first_pair", first_pair), positive("first_bid", first_bid), positive("first_ask", first_ask)),
            (_pair("second_pair", second_pair), positive("second_bid", second_bid),
             positive("second_ask", second_ask))]
    for (pair, bid, ask), name in zip(legs, ("first", "second")):
        if np.any(bid > ask):
            raise ValueError(f"{name}_bid must not exceed {name}_ask")
    x, y = _pair("cross", cross)
    shared = set(legs[0][0]) & set(legs[1][0])
    if len(shared) != 1 or {x, y} != (set(legs[0][0]) | set(legs[1][0])) - shared:
        raise ValueError(f"cross {cross!r} is not the pair the two legs make through one shared currency")
    common = shared.pop()
    side = {}
    for (pair, bid, ask) in legs:
        outer = pair[0] if pair[1] == common else pair[1]
        wanted = (x, common) if outer == x else (common, y)
        if pair == wanted:
            side[outer] = (bid, ask, 0.5 * (bid + ask))
        else:
            side[outer] = (1.0 / ask, 1.0 / bid, 1.0 / (0.5 * (bid + ask)))
    (xb, xa, xm), (yb, ya, ym) = side[x], side[y]
    bid, ask = xb * yb, xa * ya
    return Result({"bid": out(bid), "ask": out(ask), "mid": out(xm * ym if mid_rule == "legs" else 0.5 * (bid + ask))})


ADJUSTMENTS = ("following", "modified_following")
LAG_RULES = ("joint", "usd_rule")
FAR_LEGS = ("buy_base", "sell_base")
NEAR_SIDES = ("opposite", "same")
TENOR = re.compile(r"^(\d+)([DWMY])$")
WEEKMASK = "1111100"            # Saturday and Sunday closed


def _day(name, value):
    try:
        return np.datetime64(value, "D")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a date (an ISO string 'YYYY-MM-DD', datetime.date or datetime64)") from error


def _calendar(days):
    return np.busdaycalendar(weekmask=WEEKMASK, holidays=np.asarray(sorted(days), dtype="datetime64[D]"))


def _adjusted(day, calendar, adjustment):
    rolled = np.busday_offset(day, 0, roll="forward", busdaycal=calendar)
    if adjustment == "modified_following" and rolled.astype("datetime64[M]") != day.astype("datetime64[M]"):
        rolled = np.busday_offset(day, 0, roll="backward", busdaycal=calendar)
    return rolled


def fx_value_dates(trade_date, spot_lag, tenor, holidays, adjustment, end_of_month, lag_rule, fixing_lag):
    """FX value date: the spot lag and tenor over both currencies' holidays, broken date, NDF fixing.

    Spot is the trade date plus spot_lag business days (T + 2 for most pairs, T + 1 for USD/CAD and some others):
    'joint' counts days open in every calendar given; 'usd_rule' counts days open in the non-USD calendars and then
    moves the spot date on to the first day open in all (an intermediate USD holiday does not delay spot). A tenor
    runs from spot: 'nD' or 'nW' calendar days, 'nM' or 'nY' months (the same day of the month, clipped to its end;
    with end_of_month, a spot on its month's last business day gives the target month's last business day), then
    the date is rolled on the joint calendar, 'following' to the next business day or 'modified_following' unless
    that leaves the month, then back. A non-deliverable forward fixes fixing_lag business days before its settlement
    (the delivery date). Weekends are Saturday and Sunday.

    conventions:
    - trade_date: a date (ISO 'YYYY-MM-DD'); spot_lag: business days, an int (2, or 1 for USD/CAD)
    - tenor: '1W', '6W', '1M', '3M', '1Y', '10D' and the like, or None for spot alone
    - holidays: each currency's holidays, a dict such as {'EUR': [...], 'USD': [...]}, dates
    - adjustment: 'following' or 'modified_following'; end_of_month: True or False
    - lag_rule: 'joint' or 'usd_rule'; fixing_lag: business days before settlement (2), or None
    returns: a dict: spot_date; delivery_date; days_from_spot (calendar days); fixing_date (or None); ISO strings
    source: ISDA 2006 Definitions, 4.12 (Business Day Conventions); 1998 FX and Currency Option Definitions (Spot
      Date); Steiner, Mastering Financial Calculations (2012), ch. 6 (value dates and forward dates)
    aliases: FX spot date, FX value date, FX settlement date, value date, settlement date, spot lag, NDF fixing date, [broken date]
    example:
        qf.fx_value_dates(trade_date="2024-03-27", spot_lag=2, tenor="1M", holidays={"EUR": ["2024-03-29", "2024-04-01"], "USD": ["2024-03-29"]},
                          adjustment="modified_following", end_of_month=True, lag_rule="joint",
                          fixing_lag=None)
    """
    choice("adjustment", adjustment, ADJUSTMENTS)
    choice("lag_rule", lag_rule, LAG_RULES)
    trade, lag = _day("trade_date", trade_date), int(spot_lag)
    if isinstance(spot_lag, (bool, np.bool_)) or lag != spot_lag or lag < 0:
        raise ValueError("spot_lag must be a non-negative int of business days")
    if not isinstance(holidays, dict) or not holidays:
        raise ValueError("holidays must be a dict of each currency's holiday dates, e.g. {'EUR': [...], 'USD': [...]}")
    if not isinstance(end_of_month, (bool, np.bool_)):
        raise ValueError("end_of_month must be True or False")
    days = {code: {_day(f"holidays[{code!r}]", d) for d in dates} for code, dates in holidays.items()}
    joint = _calendar(set().union(*days.values()))
    if lag_rule == "usd_rule":
        others = [dates for code, dates in days.items() if code.upper() != "USD"]
        counting = _calendar(set().union(*others)) if others else joint
        spot = np.busday_offset(np.busday_offset(trade, lag, roll="forward", busdaycal=counting), 0, roll="forward",
                                busdaycal=joint)
    else:
        spot = np.busday_offset(trade, lag, roll="forward", busdaycal=joint)
    delivery = spot
    if tenor is not None:
        found = TENOR.match(str(tenor).upper())
        if not found:
            raise ValueError("tenor must read like '3D', '6W', '3M' or '1Y', or be None")
        count, unit = int(found.group(1)), found.group(2)
        if unit in "DW":
            delivery = _adjusted(spot + np.timedelta64(count * (7 if unit == "W" else 1), "D"), joint, adjustment)
        else:
            months = count * (12 if unit == "Y" else 1)
            month = spot.astype("datetime64[M]") + np.timedelta64(months, "M")
            last_day = (month + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
            last_business = np.busday_offset(spot.astype("datetime64[M]") + np.timedelta64(1, "M"), -1,
                                             roll="forward", busdaycal=joint)
            if end_of_month and spot == last_business:
                delivery = np.busday_offset(last_day, 0, roll="backward", busdaycal=joint)
            else:
                day_of_month = (spot - spot.astype("datetime64[M]").astype("datetime64[D]")).astype(int)
                target = min(month.astype("datetime64[D]") + np.timedelta64(day_of_month, "D"), last_day)
                delivery = _adjusted(target, joint, adjustment)
    fixing = None
    if fixing_lag is not None:
        if isinstance(fixing_lag, (bool, np.bool_)) or int(fixing_lag) != fixing_lag or fixing_lag < 0:
            raise ValueError("fixing_lag must be a non-negative int of business days, or None")
        fixing = str(np.busday_offset(delivery, -int(fixing_lag), roll="backward", busdaycal=joint))
    return Result({"spot_date": str(spot), "delivery_date": str(delivery),
                   "days_from_spot": int((delivery - spot).astype(int)), "fixing_date": fixing})


@_guard(price=('spot',))
def fx_swap_outrights(spot, near_points, far_points, point_multiplier, far_leg, near_side):
    """FX swap and forward rates from forward points, bid or offer by the far leg, the near leg's side named.

    An outright is spot + points / point_multiplier (points quoted bid / offer; negative points a discount). The far
    leg is dealt on its direction's side: the client selling the base currency forward ('sell_base') on the bid
    points, buying it ('buy_base') on the offer points. near_side names the near leg's: 'opposite' the other side, as
    a dealer quotes a forward-forward swap (bid swap points = far bid - near offer, offer = far offer - near bid:
    FIX's BidSwapPoints and OfferSwapPoints); 'same' the far leg's side on both legs. near outright = spot + near
    points / multiplier and far outright = spot + far points / multiplier, each on its leg's side; the swap points
    far minus near. A spot-start swap (or an outright) has near points (0, 0), where the two agree. An NDF settles in
    the deliverable currency (USD) the difference (F_fixing - F_contract) N / F_fixing per N USD bought forward at
    F_contract, against the fixing rate F_fixing (USD per one: N (F_fixing - F_contract) / F_fixing; its dates
    qf.fx_value_dates); CIP forwards and points from deposit rates: qf.cip_forward.

    conventions:
    - spot: the spot rate both legs build on, quote currency per base unit (mid, or the side dealt)
    - near_points, far_points: each a (bid, offer) pair of forward points, in pips
    - point_multiplier: pips per unit of the rate (10,000 for most pairs, 100 for JPY)
    - far_leg: 'sell_base' (bid points) or 'buy_base' (offer points), the client's side at the far date
    - near_side: 'opposite' (a dealer's forward-forward) or 'same' (both legs on the far leg's side)
    returns: a dict: side (the far leg's, 'bid' or 'offer'); near_quote (the near leg's); near_outright; far_outright;
      swap_points (far minus near, pips)
    source: Steiner, Mastering Financial Calculations (2012), ch. 6 (forward outrights, FX swaps, forward-forwards);
      FIX Protocol, fields BidSwapPoints (1065) and OfferSwapPoints (1066); ACI, The Model Code
    aliases: FX swap points, FX swap roll, swap roll, FX roll, NDF settlement, [FX swap], [swap points], [forward points], [roll cost]
    example:
        qf.fx_swap_outrights(spot=1.0850, near_points=(1.2, 1.5), far_points=(45.0, 47.0),
                             point_multiplier=10000.0, far_leg="sell_base",
                             near_side="opposite")
    """
    choice("far_leg", far_leg, FAR_LEGS)
    choice("near_side", near_side, NEAR_SIDES)
    s, scale = float(positive("spot", spot)), float(positive("point_multiplier", point_multiplier))
    try:
        (near_bid, near_offer), (far_bid, far_offer) = ((float(a), float(b)) for a, b in (near_points, far_points))
    except (TypeError, ValueError) as error:
        raise ValueError("near_points and far_points must each be a (bid, offer) pair") from error
    if near_bid > near_offer or far_bid > far_offer:
        raise ValueError("a points pair is (bid, offer), its bid at or below its offer")
    selling = far_leg == "sell_base"
    side, far = ("bid", far_bid) if selling else ("offer", far_offer)
    near_bid_side = selling if near_side == "same" else not selling
    near_quote, near = ("bid", near_bid) if near_bid_side else ("offer", near_offer)
    return Result({"side": side, "near_quote": near_quote, "near_outright": s + near / scale,
                   "far_outright": s + far / scale, "swap_points": far - near})
