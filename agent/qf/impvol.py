"""Implied vol: exact or approximated (Brenner-Subrahmanyam, Li, Corrado-Miller), Bachelier; parity forward; variance swap.

The closed forms are Brenner and Subrahmanyam's, Li's at the money and away from it, and Corrado and Miller's (with
Hallerbach's refinement); the forward and the rate are read from put-call parity on quotes.

The approximations take the discounted underlying S~ = S e^{-qT} and the discounted strike K~ = K e^{-rT} (rates
continuously compounded) and a European call's price; they return the annual volatility, T in years.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special

from ._common import Result, choice, out, positive
from .options import bachelier_price, bs_price
from .primitives import _guard

__all__ = ["brenner_subrahmanyam_vol", "corrado_miller_vol", "implied_vol", "li_atm_vol", "li_nonatm_vol",
           "normal_implied_vol", "parity_forward", "parity_rate_and_forward", "variance_swap_strike"]

VARIANTS = ("corrado_miller", "hallerbach", "hallerbach_unadjusted")
VOL_HIGH = 20.0     # the bisection's upper bracket: an annual volatility of 2,000%
METHODS = ("bisection", "newton")
LI_SWITCH = 1.4      # Li's rho above which the quadratic branch is taken


@_guard(rate=('rate',), time=('maturity',), price=('strike', 'spot',))
def parity_forward(call_bid, call_ask, put_bid, put_ask, strike, spot, rate, maturity):
    """Synthetic forward bid, ask and mid from put-call parity on bid/ask quotes.

    A synthetic long forward buys the call at its ask and sells the put at its bid; a short one does the reverse:
    forward_bid = K + (C_bid - P_ask) / DF,  forward_ask = K + (C_ask - P_bid) / DF,  DF = e^{-rT},
    mid = (forward_bid + forward_ask) / 2,  implied_yield = r - ln(mid / S) / T,
    the continuous yield (dividends plus borrow cost) that makes S e^{(r - q)T} the mid forward.

    European put-call parity, C - P = S e^{-qT} - K e^{-rT}, turns a put into a call (before a smile fit on calls);
    quotes off it by more than their spreads are an arbitrage.

    Across expiries the total implied variance vol^2 T must not fall at a fixed moneyness: the forward variance
    (w2 - w1) / (T2 - T1) >= 0 is the calendar check (qf.arbitrage_checks).

    conventions:
    - quotes: European call and put at the same strike and expiry; a bid above its ask is refused (clean them first)
    - rate: continuously compounded, so DF = e^{-r T}; maturity in years (ACT/365 from dates is the caller's)
    - implied_yield: continuously compounded; an implied borrow rate is implied_yield less the known dividend yield
    returns: a dict: bid, ask and mid (forward prices, in the spot's units); implied_yield (continuously
      compounded, annual)
    source: Stoll (1969), "The relationship between put and call option prices", J. Finance 24; Hull, "Put-call parity"
    aliases: [put-call parity], synthetic forward, [implied forward], parity forward, implied borrow, implied dividend
    example:
        qf.parity_forward(call_bid=6.1, call_ask=6.3, put_bid=4.0, put_ask=4.2,
                          strike=100.0, spot=100.0, rate=0.04,
                          maturity=0.5)
    """
    cb, ca, pb, pa = (np.asarray(x, dtype=float) for x in (call_bid, call_ask, put_bid, put_ask))
    if np.any(cb > ca) or np.any(pb > pa):
        raise ValueError("call_bid and put_bid must not exceed call_ask and put_ask (crossed quotes)")
    k, s, t = positive("strike", strike), positive("spot", spot), positive("maturity", maturity)
    r = np.asarray(rate, dtype=float)
    df = np.exp(-r * t)
    bid, ask = k + (cb - pa) / df, k + (ca - pb) / df
    mid = 0.5 * (bid + ask)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = r - np.log(mid / s) / t
    return Result({"bid": out(bid), "ask": out(ask), "mid": out(mid),
                   "implied_yield": out(np.where(mid > 0, implied, np.nan))})


@_guard(time=('maturity',), price=('strikes',))
def parity_rate_and_forward(strikes, call_prices, put_prices, maturity):
    """Discount factor, rate and forward implied by put-call parity across strikes.

    C(K) - P(K) = DF F - DF K for every strike, so a least-squares line of C - P on K has slope -DF and intercept
    DF F: DF = -slope, forward = intercept / DF, rate = -ln(DF) / T. With two strikes the line is exact, the box
    spread: DF = [(C1 - P1) - (C2 - P2)] / (K2 - K1).

    conventions:
    - prices: one call and one put price per strike, the same expiry (mids, or the side the task names)
    - rate: continuously compounded, from DF = e^{-r T}; maturity in years
    - at least two distinct strikes; a non-positive implied discount factor is refused
    returns: a dict: discount_factor; rate (continuously compounded, annual); forward; residuals (C - P
      less the line, one per strike)
    source: Hull, "Put-call parity" and box spreads; Brenner and Galai (1986), implied interest rates
    aliases: implied rate from options, box spread rate, [implied forward], parity regression, implied discount factor
    example:
        qf.parity_rate_and_forward(strikes=[90.0, 100.0, 110.0],
                                   call_prices=[14.0, 7.1, 2.6],
                                   put_prices=[2.3, 5.1, 10.3], maturity=0.5)
    """
    k = positive("strikes", strikes).ravel()
    c, p = np.asarray(call_prices, dtype=float).ravel(), np.asarray(put_prices, dtype=float).ravel()
    if c.shape != k.shape or p.shape != k.shape:
        raise ValueError("strikes, call_prices and put_prices must have the same length")
    if np.unique(k).size < 2:
        raise ValueError("strikes must hold at least two distinct strikes")
    t = float(positive("maturity", maturity))
    gap = c - p
    (intercept, slope), *_ = np.linalg.lstsq(np.column_stack([np.ones_like(k), k]), gap, rcond=None)
    df = -slope
    if df <= 0:
        raise ValueError("call_prices and put_prices imply a non-positive discount factor")
    return Result({"discount_factor": float(df), "rate": float(-np.log(df) / t), "forward": float(intercept / df),
                   "residuals": gap - (intercept + slope * k)})


@_guard(rate=('rate', 'dividend_yield',), time=('maturity',), price=('spot', 'strike',))
def implied_vol(option, price, spot, strike, maturity, rate, dividend_yield, method):
    """Black-Scholes-Merton implied vol of a price, by bisection or by Newton-Raphson on vega.

    The vol at which bs_price(option, spot, strike, maturity, rate, dividend_yield, vol) equals `price`. 'bisection'
    halves [0, 20] a hundred times (to about 1e-15; the price increases in vol); 'newton' iterates vol - (bs_price -
    price) / vega, vega = S e^{-qT} phi(d1) sqrt(T), from Manaster and Koehler's start sqrt(2 |ln(S~/K~)| / T) (the
    price's inflection in vol; Brenner and Subrahmanyam's sqrt(2 pi / T) price / S~ at the money forward) until a
    step is below 1e-12, numpy.nan where 100 steps do not get there (vega near 0, far from the money). numpy.nan
    too where the price lies outside the no-arbitrage bounds (a call within (max(S~ - K~, 0), S~), a put within
    (max(K~ - S~, 0), K~)), S~ = S e^{-qT}, K~ = K e^{-rT}. Arrays broadcast.
    conventions:
    - method: 'bisection' or 'newton' (a task naming Newton-Raphson wants it)
    Source: the Black-Scholes-Merton formula (bs_price) inverted; Hull, Options, Futures, and Other Derivatives,
    "Implied volatilities"; Manaster and Koehler (1982), Journal of Finance 37(1).
    aliases: [implied volatility], [implied vol], bisection implied vol, Newton-Raphson implied vol, invert Black-Scholes
    returns: the annual vol, a float or an array; numpy.nan outside the no-arbitrage bounds or where Newton stalls
    example:
        price = qf.bs_price("call", 100.0, 95.0, 0.5, 0.03, 0.01, 0.25)
        qf.implied_vol(option="call", price=price, spot=100.0, strike=95.0, maturity=0.5, rate=0.03,
                       dividend_yield=0.01, method="newton")
    """
    choice("option", option, ("call", "put"))
    choice("method", method, METHODS)
    price = np.asarray(price, dtype=float)
    s, k, t = positive("spot", spot), positive("strike", strike), positive("maturity", maturity)
    r, q = np.asarray(rate, dtype=float), np.asarray(dividend_yield, dtype=float)
    fs, fk = s * np.exp(-q * t), k * np.exp(-r * t)
    low, high = (np.maximum(fs - fk, 0.0), fs) if option == "call" else (np.maximum(fk - fs, 0.0), fk)
    shape = np.broadcast(price, s, k, t, r, q).shape
    inside = (price > low) & (price < high)
    if method == "bisection":
        lo, hi = np.zeros(shape), np.full(shape, VOL_HIGH)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            above = np.asarray(bs_price(option, s, k, t, r, q, mid)) > price
            hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
        return out(np.where(inside, 0.5 * (lo + hi), np.nan))
    start = np.sqrt(2.0 * np.abs(np.log(fs / fk)) / t)
    vol = np.broadcast_to(np.where(start > 0, start, np.sqrt(2.0 * np.pi / t) * price / fs), shape).copy()
    done = ~np.broadcast_to(inside, shape).copy()
    for _ in range(100):
        sd = vol * np.sqrt(t)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d1 = (np.log(fs / fk) + 0.5 * sd * sd) / sd
            vega = fs * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi) * np.sqrt(t)
            step = (np.asarray(bs_price(option, s, k, t, r, q, np.maximum(vol, 0.0))) - price) / vega
        step = np.where(done, 0.0, step)
        vol = vol - step
        done = done | (np.abs(step) < 1e-12)
        if np.all(done):
            break
    good = np.broadcast_to(inside, shape) & done & np.isfinite(vol) & (vol > 0)
    return out(np.where(good, vol, np.nan))


NORMAL_METHODS = ("jaeckel", "bisection")
_LAPLACE_TERMS = 200   # the Mills ratio's continued fraction, depth: converged to rounding for u >= 2


def _normal_time_value(u):
    """phi(u) - u Phi(-u) for u >= 0: an out-of-the-money normal-model price over its standard deviation, with no
    cancellation where u is large (Laplace's continued fraction for the Mills ratio Phi(-u)/phi(u) beyond 2)."""
    density = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    if u < 2.0:
        return density - u * 0.5 * special.erfc(u / np.sqrt(2.0))
    tail = u
    for j in range(_LAPLACE_TERMS, 1, -1):
        tail = u + j / tail                  # u + 2/(u + 3/(u + ...))
    return density / (tail * (u + 1.0 / tail))


def _normal_solve(ratio):
    """The u > 0 with (phi(u) - u Phi(-u)) / u = ratio > 0 (decreasing in u), to rounding."""
    def gap(u):
        return np.log(_normal_time_value(u) / u) - np.log(ratio)
    low, high = 1e-300, 1.0
    while gap(high) > 0:
        high *= 2.0
    while gap(low) < 0:
        low *= 1e-3
    return optimize.brentq(gap, low, high, xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=500)


@_guard(time=('expiry',))
def normal_implied_vol(option, price, forward, strike, expiry, discount_factor, method):
    """Normal implied volatility (Bachelier): Jaeckel's inversion or bisection.

    The vol at which qf.bachelier_price(option, forward, strike, expiry, vol, discount_factor) is `price`. 'jaeckel':
    the time value v = price/DF - max(+-(F - K), 0) of the out-of-the-money option, h = vol sqrt T and u = |F - K|/h
    satisfy v / |F - K| = (n(u) - u N(-u)) / u, one equation in u (Jaeckel's reduction), solved to rounding; at the
    money vol = v sqrt(2 pi / T). 'bisection': halves [0, (|F - K| + 16 v) / sqrt T] on the price a hundred and fifty
    times. numpy.nan where the price is at or below the intrinsic value (the normal model has no upper bound).

    conventions:
    - option: 'call' or 'put'; price: the option's price, discounted by discount_factor
    - forward F, strike K in the same units (rates in decimals: the vol is then in decimals a year, 0.008 for 80 bp)
    - expiry: T in years; discount_factor: DF (an annuity for a swaption); method: 'jaeckel' or 'bisection'
    returns: the normal vol a year in F's units, a float or an array; numpy.nan where no vol gives the price
    source: Jaeckel (2017), "Implied normal volatility", Wilmott; Bachelier (1900); Le Floc'h (2016), "Fast and
      accurate analytic basis point volatility"
    aliases: normal implied volatility, Bachelier implied volatility
    example:
        import qf
        qf.normal_implied_vol(option="call", price=0.0061, forward=0.03, strike=0.025, expiry=1.0,
                              discount_factor=0.97, method="jaeckel")  # result: 0.00797957
    """
    choice("option", option, ("call", "put"))
    choice("method", method, NORMAL_METHODS)
    p, f, k = (np.asarray(x, dtype=float) for x in (price, forward, strike))
    t, df = positive("expiry", expiry), positive("discount_factor", discount_factor)
    sign = 1.0 if option == "call" else -1.0
    shape = np.broadcast(p, f, k, t, df).shape
    p, f, k, t, df = (np.broadcast_to(x, shape).ravel() for x in (p, f, k, t, df))
    found = np.full(p.size, np.nan)
    for i in range(p.size):
        value = p[i] / df[i] - max(sign * (f[i] - k[i]), 0.0)
        gap = abs(f[i] - k[i])
        if not value > 0 or not np.isfinite(value):
            continue
        if method == "bisection":
            lo, hi = 0.0, (gap + 16.0 * value) / np.sqrt(t[i])
            for _ in range(150):
                mid = 0.5 * (lo + hi)
                if bachelier_price(option, f[i], k[i], t[i], mid, df[i]) > p[i]:
                    hi = mid
                else:
                    lo = mid
            found[i] = 0.5 * (lo + hi)
        elif gap == 0:
            found[i] = value * np.sqrt(2.0 * np.pi / t[i])
        else:
            found[i] = gap / _normal_solve(value / gap) / np.sqrt(t[i])
    return out(found.reshape(shape))


@_guard(time=('maturity',), price=('underlying_pv',))
def brenner_subrahmanyam_vol(call_price, underlying_pv, maturity):
    """vol = sqrt(2 pi / T) C / S~: the first-order inversion of an at-the-money-forward call (S~ = K~),
    where C = S~ (2 N(vol sqrt(T)/2) - 1) ~ S~ vol sqrt(T) / sqrt(2 pi). Its error there is about -vol^3 T / 24
    (it undershoots). `underlying_pv` S~ = S e^{-qT}.
    Source: Brenner and Subrahmanyam (1988), "A simple formula to compute the implied standard deviation", Financial
    Analysts Journal 44(5).
    aliases: Brenner-Subrahmanyam, Brenner Subrahmanyam approximation
    returns: the annual implied vol, a float or an array
    example:
        qf.brenner_subrahmanyam_vol(call_price=8.0, underlying_pv=100.0, maturity=1.0)
    """
    c = np.asarray(call_price, dtype=float)
    s, t = positive("underlying_pv", underlying_pv), positive("maturity", maturity)
    return out(np.sqrt(2.0 * np.pi / t) * c / s)


@_guard(time=('maturity',), price=('underlying_pv',))
def li_atm_vol(call_price, underlying_pv, maturity):
    """Li's at-the-money implied vol, the root of the cubic expansion of C = S~ (2 N(vol sqrt(T)/2) - 1):
    alpha = sqrt(2 pi) C / S~, z = cos(arccos(3 alpha / sqrt(32)) / 3),
    vol = [2 sqrt(2) z - sqrt(8 z^2 - 6 alpha / (sqrt(2) z))] / sqrt(T),
    which solves alpha = v - v^3/24 for v = vol sqrt(T) (v errs by about +v^5/640 at the money forward). numpy.nan where
    3 alpha / sqrt(32) > 1 (the arccos's domain). `underlying_pv` S~ = S e^{-qT}.
    Source: Li (2005), "A new formula for computing implied volatility", Applied Mathematics and Computation 170(1).
    aliases: Li approximation, Li (2005), Li ATM implied vol, Li ATM
    returns: the annual implied vol, a float or an array; numpy.nan outside the arccos's domain
    example:
        qf.li_atm_vol(call_price=8.0, underlying_pv=100.0, maturity=1.0)
    """
    c = np.asarray(call_price, dtype=float)
    s, t = positive("underlying_pv", underlying_pv), positive("maturity", maturity)
    alpha = np.sqrt(2.0 * np.pi) * c / s
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.cos(np.arccos(3.0 * alpha / np.sqrt(32.0)) / 3.0)
        v = 2.0 * np.sqrt(2.0) * z - np.sqrt(8.0 * z * z - 6.0 * alpha / (np.sqrt(2.0) * z))
    return out(v / np.sqrt(t))


@_guard(time=('maturity',), price=('underlying_pv', 'strike_pv',))
def corrado_miller_vol(call_price, underlying_pv, strike_pv, maturity, variant):
    """Implied vol of a call price at any moneyness by a quadratic approximation: Corrado-Miller or Hallerbach.
    With S~ = S e^{-qT} (`underlying_pv`), K~ = K e^{-rT} (`strike_pv`) and C the call price:
    'corrado_miller' (1996): vol sqrt(T) = sqrt(2 pi)/(S~ + K~) [C - (S~ - K~)/2 + sqrt((C - (S~ - K~)/2)^2 -
      (S~ - K~)^2 / pi)];
    'hallerbach' (2004): vol sqrt(T) = sqrt(2 pi)/(2 (S~ + K~)) [2C + K~ - S~ + sqrt((2C + K~ - S~)^2 -
      1.85 (S~ + K~)(K~ - S~)^2 / (pi sqrt(K~ S~)))].
    'hallerbach_unadjusted': his eq. 20, replacing the discriminant's correction by
      2 (S~ + K~)(K~ - S~)^2 / (pi S~).
    A negative discriminant is clipped to 0. At S~ = K~ both are brenner_subrahmanyam_vol.
    conventions:
    - variant: the formula the task names: 'hallerbach' for Hallerbach's (2004) improved estimator (his eq. 24, the
      1.85 factor), 'hallerbach_unadjusted' for his unadjusted estimator (eq. 20, the factor 2),
      'corrado_miller' for Corrado and Miller's (1996) quadratic formula
    - call_price: a European call's; underlying_pv: S~ = S e^{-qT}; strike_pv: K~ = K e^{-rT}; maturity: T in years
    Source: Corrado and Miller (1996), "A note on a simple, accurate formula to compute implied standard deviations",
    Journal of Banking and Finance 20(3); Hallerbach (2004), "An improved estimator for Black-Scholes-Merton implied
    volatility", ERIM Report Series Research in Management, Erasmus University Rotterdam.
    aliases: Corrado-Miller, Corrado Miller approximation, Hallerbach, Hallerbach approximation, Hallerbach's improved estimator
    returns: the annual implied vol, a float or an array
    example:
        qf.corrado_miller_vol(call_price=10.0, underlying_pv=100.0, strike_pv=95.0, maturity=1.0,
                              variant="hallerbach")
    """
    choice("variant", variant, VARIANTS)
    c = np.asarray(call_price, dtype=float)
    s, k, t = positive("underlying_pv", underlying_pv), positive("strike_pv", strike_pv), positive("maturity", maturity)
    if variant == "corrado_miller":
        half = c - (s - k) / 2.0
        root = np.sqrt(np.maximum(half * half - (s - k) ** 2 / np.pi, 0.0))
        v = np.sqrt(2.0 * np.pi) / (s + k) * (half + root)
    else:
        lead = 2.0 * c + k - s
        correction = (2.0 / s if variant == "hallerbach_unadjusted" else 1.85 / np.sqrt(k * s))
        root = np.sqrt(np.maximum(lead * lead - correction * (s + k) * (k - s) ** 2 / np.pi, 0.0))
        v = np.sqrt(2.0 * np.pi) / (2.0 * (s + k)) * (lead + root)
    return out(v / np.sqrt(t))


@_guard(time=('maturity',), price=('underlying_pv', 'strike_pv',))
def li_nonatm_vol(call_price, underlying_pv, strike_pv, maturity):
    """Li (2005) implied vol at any moneyness: a cubic near the money, a quadratic away from it.

    With S~ = S e^{-qT}, K~ = K e^{-rT}, eta = K~ / S~, c = C / S~ and v = vol sqrt(T):
      alpha = sqrt(2 pi) / (1 + eta) (2 c + eta - 1),   rho = |eta - 1| / c^2,
      rho <= 1.4: z = cos(arccos(3 alpha / sqrt(32)) / 3),  v = 2 sqrt(2) z - sqrt(8 z^2 - 6 alpha / (sqrt(2) z)),
                  the root of alpha = v - v^3 / 24 (li_atm_vol's cubic on Corrado and Miller's moneyness-adjusted
                  price);
      rho > 1.4:  v = [alpha + sqrt(alpha^2 - 4 (eta - 1)^2 / (1 + eta))] / 2,
                  the larger root of v^2 - alpha v + (eta - 1)^2 / (1 + eta) = 0;
    vol = v / sqrt(T). At eta = 1 it is li_atm_vol. Undefined (numpy.nan) where its branch has no real value (3 alpha
    / sqrt(32) > 1, or a negative square root): far from the money with little time value, where implied_vol (exact)
    is the answer.

    conventions:
    - call_price: a European call's (a put's by parity, C = P + S~ - K~); maturity T in years
    - underlying_pv: S~ = S e^{-qT}; strike_pv: K~ = K e^{-rT} (q and r continuously compounded)
    - S. Li's (2005) closed form, not M. Li's (2008) rational approximation (not in this library)
    returns: the annual implied vol: a float, or an array; numpy.nan where the formula is undefined
    source: Li (2005), "A new formula for computing implied volatility", Applied Mathematics and Computation 170(1), 611-625; restated in Orlando and Taglialatela (2017), "A review on implied volatility calculation", J. Comput. Appl. Math. 320
    aliases: Li approximation, Li non-ATM, Li (2005) non-ATM, non-ATM implied vol, non-ATM implied volatility, Li formula, Li implied volatility
    example:
        qf.li_nonatm_vol(call_price=7.2,
                         underlying_pv=100.0, strike_pv=104.0,
                         maturity=0.5)
    """
    c = np.asarray(call_price, dtype=float)
    s, k, t = positive("underlying_pv", underlying_pv), positive("strike_pv", strike_pv), positive("maturity", maturity)
    eta, norm = k / s, c / s
    alpha = np.sqrt(2.0 * np.pi) / (1.0 + eta) * (2.0 * norm + eta - 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.abs(eta - 1.0) / (norm * norm)
        z = np.cos(np.arccos(3.0 * alpha / np.sqrt(32.0)) / 3.0)
        cubic = 2.0 * np.sqrt(2.0) * z - np.sqrt(8.0 * z * z - 6.0 * alpha / (np.sqrt(2.0) * z))
        quadratic = 0.5 * (alpha + np.sqrt(alpha * alpha - 4.0 * (eta - 1.0) ** 2 / (1.0 + eta)))
    return out(np.where(rho <= LI_SWITCH, cubic, quadratic) / np.sqrt(t))


VARIANCE_FORMS = ("replication", "vix", "forward")


@_guard(rate=('rate',), time=('maturity',), price=('strikes', 'forward',))
def variance_swap_strike(strikes, call_prices, put_prices, forward, rate, maturity, form):
    """Variance-swap fair strike by static replication from an option strip: the log contract, or the VIX formula.

    K_var = (2 e^{rT} / T) sum_i (dK_i / K_i^2) Q(K_i) + boundary term, Q the out-of-the-money price at each strike,
    dK_i = (K_{i+1} - K_{i-1}) / 2 inside the strip and one-sided at its ends. 'replication' (Demeterfi, Derman,
    Kamal and Zou's model-free integral, Hull's strip): the boundary S* = K0, the highest strike at or below F, puts
    below it, calls above, the mean of the two at K0, the term (2/T)[ln(F/K0) - (F/K0 - 1)] exact; 'vix' (Cboe's
    VIX formula): the same strip, the term -(1/T)(F/K0 - 1)^2, its second-order expansion (the two agree as the
    strip's grid fines); 'forward': S* = F itself, puts at K < F, calls at K >= F, no term. Under Black-Scholes a
    dense, wide strip gives vol^2 (the log contract -(2/T) ln(S_T/F) is replicated by these options). The fair vol is
    sqrt(K_var); a long variance swap of variance notional N pays N (realised variance - K_var) at T, its realised
    variance annualised as the contract says (qf.realized_variance gives one period's), vega notional 2 sqrt(K_var) N.

    conventions:
    - strikes: increasing; call_prices, put_prices: today's prices at them (a side not used may be NaN)
    - forward: F for the maturity; rate: r continuously compounded; maturity: T in years
    - form: 'replication', 'vix' or 'forward' ('forward' where a task picks each strike's out-of-the-money side
      against its given forward, puts below F and calls at or above, and adds no correction term)
    returns: a dict: variance (K_var, annual); vol (sqrt); boundary_strike (K0, or F); weights (2 e^{rT} dK_i /
      (T K_i^2)); otm_prices (Q)
    source: Demeterfi, Derman, Kamal and Zou (1999), "More than you ever wanted to know about volatility swaps",
      Goldman Sachs; Cboe Volatility Index Mathematics Methodology (2023), its worked example; Hull, "Variance swaps"
    aliases: variance swap, variance swap strike, fair variance, fair variance strike, variance swap replication, log contract, VIX formula, static replication
    example:
        qf.variance_swap_strike(strikes=[80.0, 90.0, 100.0, 110.0, 120.0], call_prices=[21.0, 12.4, 5.6, 1.9, 0.5],
                                put_prices=[0.4, 1.6, 4.7, 10.9, 19.4], forward=100.5, rate=0.02, maturity=0.25,
                                form="replication")
    """
    choice("form", form, VARIANCE_FORMS)
    k = positive("strikes", strikes).ravel()
    calls, puts = np.asarray(call_prices, dtype=float).ravel(), np.asarray(put_prices, dtype=float).ravel()
    if k.size < 2 or np.any(np.diff(k) <= 0) or calls.shape != k.shape or puts.shape != k.shape:
        raise ValueError("strikes must be two or more, increasing, with a call and a put price at each")
    f, t, r = float(positive("forward", forward)), float(positive("maturity", maturity)), float(rate)
    if form == "forward":
        edge, q = f, np.where(k < f, puts, calls)
        term = 0.0
    else:
        below = np.flatnonzero(k <= f)
        if below.size == 0:
            raise ValueError(f"{form!r} needs a strike at or below the forward (K0)")
        at = below[-1]
        edge = k[at]
        q = np.where(k < edge, puts, calls)
        q[at] = 0.5 * (calls[at] + puts[at])
        x = f / edge - 1.0
        term = 2.0 / t * (np.log1p(x) - x) if form == "replication" else -x * x / t
    if not np.all(np.isfinite(q)):
        raise ValueError("an out-of-the-money price the strip uses is missing")
    dk = np.empty_like(k)
    dk[1:-1] = 0.5 * (k[2:] - k[:-2])
    dk[0], dk[-1] = k[1] - k[0], k[-1] - k[-2]
    weights = 2.0 * np.exp(r * t) / t * dk / (k * k)
    variance = float(weights @ q + term)
    return Result({"variance": variance, "vol": float(np.sqrt(variance)) if variance >= 0 else float("nan"),
                   "boundary_strike": float(edge), "weights": weights, "otm_prices": q})
