"""Exotic options: Asian, digital, lookback, forward-start, compound (Geske), chooser (Rubinstein), power.

Every function takes continuously compounded annual rates (`rate` r, `dividend_yield` q), times in years and an annual
`vol`; arrays broadcast, and a scalar result is a float, a function with several results a dict with named keys. The
European price and its Greeks are on `options`, barrier options on `barriers`, two-asset options on `multiasset`.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from ._common import Result, choice, log_ndtr, ndtr, out, positive
from ._common import bivariate_normal_cdf as _bivariate_normal_cdf
from .barriers import PAYOFFS
from .barriers import barrier_digital_price, barrier_price  # noqa: F401 - moved to `barriers`, still qf.exotics.<name>
from .multiasset import spread_price  # noqa: F401 - moved to `multiasset`, still qf.exotics.spread_price
from .options import OPTIONS, _density, _inputs, bs_price
from .primitives import _guard

__all__ = ["asian_price", "chooser_price", "complex_chooser_price", "compound_price", "digital_price",
           "forward_start_price", "lookback_price", "power_price"]

METHODS = ("geometric", "levy", "curran_taylor", "curran_root")


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity', 'times',), price=('spot', 'strike',))
def asian_price(method, option, spot, strike, times, maturity, rate, dividend_yield, vol):
    """Price of a European average-price option on the average of S at the monitoring `times`
    (years from today, a 1-D array; the caller decides whether t = 0 is among them), paid at `maturity` T >= max(times):
    payoff max(+-(Avg - K), 0) for `option` 'call' / 'put'.

    With b = r - q, mu_i = ln S + (b - vol^2/2) t_i, s_i^2 = vol^2 t_i, n monitoring dates, and ln G the mean of the
    ln S(t_i): mu_G = mean(mu_i), s_G^2 = vol^2/n^2 sum_ij min(t_i, t_j), s_xi = vol^2/n sum_j min(t_i, t_j)
    (= Cov(ln S(t_i), ln G)), M1 = E[A] = mean(S e^{b t_i}):
    - 'geometric' (exact, on the geometric average G): e^{-rT} [e^{mu_G + s_G^2/2} N(d1) - K N(d1 - s_G)],
      d1 = (mu_G - ln K + s_G^2)/s_G. Kemna and Vorst (1990), discretely monitored.
    - 'levy' (the arithmetic average A matched to a lognormal by its first two moments): M2 = E[A^2] =
      S^2/n^2 sum_ij e^{b(t_i + t_j) + vol^2 min(t_i, t_j)}, w^2 = ln M2 - 2 ln M1, price e^{-rT} [M1 N(d1) -
      K N(d1 - w)], d1 = (ln(M1/K) + w^2/2)/w. Levy (1992), "Pricing European average rate currency options", J. Int.
      Money and Finance 11.
    - 'curran_taylor' and 'curran_root' (A conditioned on G): e^{-rT} [mean_i(S e^{b t_i} N((mu_G - ln Kh)/s_G +
      s_xi/s_G)) - K N((mu_G - ln Kh)/s_G)], where Kh solves E[A | G = Kh] = K, that is mean_i exp(mu_i + s_xi (ln Kh -
      mu_G)/s_G^2 + (s_i^2 - s_xi^2/s_G^2)/2) = K. 'curran_root' solves it (by bisection; the price is then
      E[(E[A|G] - K)^+] discounted, a lower bound of the true price); 'curran_taylor' takes Curran's first-order
      expansion about ln K, Kh = 2K - mean_i exp(mu_i + s_xi (ln K - mu_G)/s_G^2 + (s_i^2 - s_xi^2/s_G^2)/2), as in
      Haug (2007). A Kh <= 0 leaves the call at e^{-rT}(M1 - K). Curran (1994), "Valuing Asian and portfolio options
      by conditioning on the geometric mean price", Management Science 40(12).
    Puts come from the average's parity, call - put = e^{-rT} (E[average] - K), with E[G] for 'geometric' and M1 for
    the others. Rates continuously compounded and annual, vol annual; with one monitoring date at T every method is
    bs_price. Scalar or broadcasting spot, strike, maturity, rate, dividend_yield and vol.
    Source: Haug, The Complete Guide to Option Pricing Formulas (2007), "Asian options" (geometric, Levy, Curran).
    aliases: Asian option, average price option, Levy approximation, Curran approximation, Curran, arithmetic average option, geometric Asian
    """
    choice("method", method, METHODS)
    choice("option", option, ("call", "put"))
    t = np.asarray(times, dtype=float).ravel()
    if t.size == 0 or np.any(t < 0) or not np.any(t > 0):
        raise ValueError("times must be a non-empty array of non-negative times, one of them positive")
    positive("spot", spot)
    positive("strike", strike)
    positive("vol", vol)
    if np.any(np.asarray(maturity, dtype=float) < t.max()):
        raise ValueError("maturity must not precede the last monitoring time")

    def one(s, k, big_t, r, q, v):
        return _price(method, option, s, k, t, big_t, r, q, v)

    return out(np.vectorize(one, otypes=[float])(spot, strike, maturity, rate, dividend_yield, vol))


def _price(method, option, s, k, t, big_t, r, q, v):
    n, b = t.size, r - q
    mu = np.log(s) + (b - 0.5 * v * v) * t
    var = v * v * t
    nearest = np.minimum.outer(t, t)
    cov = v * v * nearest.sum(axis=1) / n
    mu_g, var_g = mu.mean(), v * v * nearest.sum() / (n * n)
    sd_g = np.sqrt(var_g)
    disc = np.exp(-r * big_t)
    m1 = np.mean(s * np.exp(b * t))
    if method == "geometric":
        forward = np.exp(mu_g + 0.5 * var_g)
        d1 = (mu_g - np.log(k) + var_g) / sd_g
        call = disc * (forward * ndtr(d1) - k * ndtr(d1 - sd_g))
        return call if option == "call" else call - disc * (forward - k)
    if method == "levy":
        m2 = s * s * np.mean(np.exp(b * np.add.outer(t, t) + v * v * nearest))
        w = np.sqrt(np.log(m2) - 2.0 * np.log(m1))
        d1 = (np.log(m1 / k) + 0.5 * w * w) / w
        call = disc * (m1 * ndtr(d1) - k * ndtr(d1 - w))
        return call if option == "call" else call - disc * (m1 - k)

    def given(y):
        """E[A | ln G = y]."""
        return np.mean(np.exp(mu + cov * (y - mu_g) / var_g + 0.5 * (var - cov * cov / var_g)))

    if method == "curran_taylor":
        k_hat = 2.0 * k - given(np.log(k))
    else:
        k_hat = np.exp(_root(given, k))
    if k_hat <= 0:
        call = disc * (m1 - k)
    else:
        z = (mu_g - np.log(k_hat)) / sd_g
        call = disc * (np.mean(s * np.exp(b * t) * ndtr(z + cov / sd_g)) - k * ndtr(z))
    return call if option == "call" else call - disc * (m1 - k)


def _root(given, k):
    """The y with given(y) = k, given increasing: E[A | G = K] >= K (the arithmetic mean is never below the
    geometric), so the root lies at or below ln K; bisection after a bracket stepped down."""
    hi = np.log(k)
    if given(hi) <= k:
        return hi
    step, lo = 1.0, hi - 1.0
    while given(lo) > k:
        step *= 2.0
        lo = hi - step
        if step > 1e6:
            raise ValueError("no root of E[A | G] = K below ln K")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if given(mid) > k:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-15 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('start', 'maturity',), price=('spot',))
def forward_start_price(option, spot, alpha, start, maturity, rate, dividend_yield, vol):
    """Price today of a forward-start European option: its strike is set at time `start` to `alpha` times the spot then,
    and it expires at `maturity` (both measured from today, start <= maturity).

    V = S e^{-q t1} * BS(option; spot 1, strike alpha, life T - t1, r, q, vol),   t1 = start, T = maturity,
    i.e. for a call S e^{-q t1} [e^{-q(T-t1)} N(d1) - alpha e^{-r(T-t1)} N(d2)], d1 = [-ln(alpha) + (r - q +
    vol^2/2)(T - t1)] / (vol sqrt(T - t1)). An at-the-money forward start has alpha = 1; a cliquet (ratchet) is a sum
    of them over its reset periods. Conventions as bs_price.
    Source: Rubinstein (1990), "Exotic options"; Haug, The Complete Guide to Option Pricing Formulas (2007),
    "Forward start options"; Hull, Options, Futures, and Other Derivatives, "Forward start options".
    aliases: forward-start option, forward start, cliquet, ratchet option, Rubinstein forward start
    returns: the price today, a float or an array
    example:
        qf.forward_start_price(option="call", spot=100.0, alpha=1.0, start=0.25, maturity=1.0,
                               rate=0.03, dividend_yield=0.0, vol=0.2)
    """
    choice("option", option, OPTIONS)
    s, a = positive("spot", spot), positive("alpha", alpha)
    t1, t = np.asarray(start, dtype=float), np.asarray(maturity, dtype=float)
    if np.any(t1 < 0) or np.any(t < t1):
        raise ValueError("start must be in [0, maturity]")
    q = np.asarray(dividend_yield, dtype=float)
    return out(s * np.exp(-q * t1) * bs_price(option, 1.0, a, t - t1, rate, q, vol))


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def digital_price(payoff, option, spot, strike, maturity, rate, dividend_yield, vol):
    """Price of a European digital (binary): `payoff` 'cash' pays 1 at maturity, 'asset' pays S_T,
    where S_T > K (`option` 'call') or S_T < K ('put').

    cash call e^{-rT} N(d2), cash put e^{-rT} N(-d2), asset call S e^{-qT} N(d1), asset put S e^{-qT} N(-d1), with
    bs_price's d1 and d2; another cash amount scales the cash price. The pieces add up: cash call + cash put =
    e^{-rT}, asset call + asset put = S e^{-qT}, and the vanilla call is asset call - K * cash call; a gap call
    (strike K1, trigger K2, paying S_T - K1 where S_T > K2) is the asset call at K2 less K1 cash calls at K2, the
    gap put K1 cash puts less the asset put, at K2. Conventions as bs_price; at vol sqrt(T) = 0, S_T = S e^{(r-q)T}.

    A cash call is -dC/dK, a call spread's limit (C(K - h) - C(K + h)) / 2h.

    Delta: cash call e^{-rT} n(d2)/(S vol sqrt T), the put's its negative; asset call e^{-qT} [N(d1) + n(d1)/(vol
    sqrt T)], put e^{-qT} [N(-d1) - n(d1)/(vol sqrt T)].
    Source: Reiner and Rubinstein (1991), "Unscrambling the binary code", Risk 4(9); Hull, Options, Futures, and Other
    Derivatives, "Binary options"; Haug (2007), "Cash-or-nothing" and "Asset-or-nothing options".
    aliases: digital option, cash-or-nothing, asset-or-nothing, binary option, gap option
    returns: the price today, a float or an array (a cash payoff per unit paid)
    example:
        qf.digital_price(payoff="cash", option="call", spot=100.0, strike=105.0, maturity=0.5, rate=0.03,
                         dividend_yield=0.0, vol=0.2)
    """
    choice("payoff", payoff, PAYOFFS)
    choice("option", option, OPTIONS)
    s, k, t, r, q, v = _inputs(spot, strike, maturity, rate, dividend_yield, vol)
    sign = 1.0 if option == "call" else -1.0
    fs, fk = s * np.exp(-q * t), k * np.exp(-r * t)
    sd = v * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(fs / fk) + 0.5 * sd * sd) / sd
        prob = ndtr(sign * (d1 if payoff == "asset" else d1 - sd))
    prob = np.where(sd > 0, prob, (sign * (fs - fk) > 0).astype(float))
    return out(prob * (fs if payoff == "asset" else np.exp(-r * t)))


def _critical_spot(inner, inner_strike, life, rate, dividend_yield, vol, outer_strike):
    """The spot I at which the inner option, with `life` left, is worth the outer strike."""
    def gap(spot):
        return float(bs_price(inner, spot, inner_strike, life, rate, dividend_yield, vol)) - outer_strike
    if inner == "call":
        high = (outer_strike + inner_strike * np.exp(-rate * life)) * np.exp(dividend_yield * life)
        return optimize.brentq(gap, 1e-12 * inner_strike, high * 1.01 + 1e-12, xtol=1e-14, rtol=1e-15, maxiter=500)
    ceiling = inner_strike * np.exp(-rate * life)
    if not outer_strike < ceiling:
        raise ValueError("outer_strike must lie below the inner put's largest value K2 e^{-r (T2 - t1)}")
    low = max((ceiling - outer_strike) * np.exp(dividend_yield * life), 1e-12 * inner_strike)
    high = 2.0 * inner_strike
    while gap(high) > 0:
        high *= 2.0
    return optimize.brentq(gap, low, high, xtol=1e-14, rtol=1e-15, maxiter=500)


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('outer_maturity', 'inner_maturity',),
        price=('spot', 'outer_strike', 'inner_strike',))
def compound_price(outer, inner, spot, outer_strike, inner_strike, outer_maturity, inner_maturity, rate,
                   dividend_yield, vol):
    """Geske's compound option (a call or put on a call or put), with its critical spot.

    The outer option buys ('call') or sells ('put') the inner European option at t1 for X1; the inner expires at
    T2 > t1 with strike X2. I is the spot at which the inner is worth X1 at t1 (bs_price, life T2 - t1). y1 = [ln(S/I)
    + (r - q + vol^2/2) t1] / (vol sqrt(t1)), y2 = y1 - vol sqrt(t1), z1 and z2 the same at X2 and T2, rho =
    sqrt(t1 / T2), M the bivariate normal CDF (not a product of two normal CDFs):
    call on call S e^{-qT2} M(z1, y1; rho) - X2 e^{-rT2} M(z2, y2; rho) - X1 e^{-rt1} N(y2);
    put on call X2 e^{-rT2} M(z2, -y2; -rho) - S e^{-qT2} M(z1, -y1; -rho) + X1 e^{-rt1} N(-y2);
    call on put X2 e^{-rT2} M(-z2, -y2; rho) - S e^{-qT2} M(-z1, -y1; rho) - X1 e^{-rt1} N(-y2);
    put on put S e^{-qT2} M(-z1, y1; -rho) - X2 e^{-rT2} M(-z2, y2; -rho) + X1 e^{-rt1} N(y2).
    Parity: call on X less put on X, same inner, is the inner's price today less X1 e^{-rt1}.

    conventions:
    - outer: the compound's own right, 'call' (to buy the inner) or 'put' (to sell it); inner: the inner's, 'call'
      or 'put'
    - outer_strike X1 paid at outer_maturity t1; inner_strike X2 at inner_maturity T2, both years from today
    - rate r and dividend_yield q continuously compounded and annual; vol annual; every argument a number, one
      contract a call (its critical spot is solved for it): a grid is a loop
    returns: a dict: price; critical_spot, the spot I at t1 at which the inner is worth X1
    source: Geske (1979), Journal of Financial Economics 7(1); Haug, The Complete Guide to Option Pricing Formulas
      (2007), "Options on options"; Hull, Options, Futures, and Other Derivatives, "Compound options"
    aliases: Geske, compound option, option on an option, call on a call, call on a put, put on a call, put on a put
    example:
        qf.compound_price(outer="call", inner="call", spot=100.0, outer_strike=4.0, inner_strike=95.0,
                          outer_maturity=0.4, inner_maturity=1.0, rate=0.04, dividend_yield=0.015,
                          vol=0.3)
    """
    choice("outer", outer, OPTIONS)
    choice("inner", inner, OPTIONS)
    s, x1, x2 = (float(positive(name, value)) for name, value in
                 (("spot", spot), ("outer_strike", outer_strike), ("inner_strike", inner_strike)))
    t1, t2, v = (float(positive(name, value)) for name, value in
                 (("outer_maturity", outer_maturity), ("inner_maturity", inner_maturity), ("vol", vol)))
    if not t1 < t2:
        raise ValueError("outer_maturity must come before inner_maturity")
    r, q = float(rate), float(dividend_yield)
    critical = _critical_spot(inner, x2, t2 - t1, r, q, v, x1)
    y1 = (np.log(s / critical) + (r - q + 0.5 * v * v) * t1) / (v * np.sqrt(t1))
    z1 = (np.log(s / x2) + (r - q + 0.5 * v * v) * t2) / (v * np.sqrt(t2))
    y2, z2, rho = y1 - v * np.sqrt(t1), z1 - v * np.sqrt(t2), np.sqrt(t1 / t2)
    share, cash, paid = s * np.exp(-q * t2), x2 * np.exp(-r * t2), x1 * np.exp(-r * t1)
    m = _bivariate_normal_cdf
    if (outer, inner) == ("call", "call"):
        price = share * m(z1, y1, rho) - cash * m(z2, y2, rho) - paid * ndtr(y2)
    elif (outer, inner) == ("put", "call"):
        price = cash * m(z2, -y2, -rho) - share * m(z1, -y1, -rho) + paid * ndtr(-y2)
    elif (outer, inner) == ("call", "put"):
        price = cash * m(-z2, -y2, rho) - share * m(-z1, -y1, rho) - paid * ndtr(-y2)
    else:
        price = share * m(-z1, y1, -rho) - cash * m(-z2, y2, -rho) + paid * ndtr(y2)
    return Result({"price": float(price), "critical_spot": float(critical)})


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity', 'choose_time',), price=('spot', 'strike',))
def chooser_price(spot, strike, choose_time, maturity, rate, dividend_yield, vol):
    """Rubinstein's simple chooser: a call or a put chosen at t1, both expiring at T, one strike.

    At t1 the holder takes the dearer of the call and the put (strike K, expiry T), so by parity at t1 its value is
    the call plus e^{-q (T - t1)} times a put on S expiring at t1, strike K e^{-(r - q)(T - t1)}; in closed form
    V = S e^{-qT} N(d) - K e^{-rT} N(d - vol sqrt(T)) - S e^{-qT} N(-y) + K e^{-rT} N(-y + vol sqrt(t1)),
    d = [ln(S/K) + (r - q + vol^2/2) T] / (vol sqrt(T)), y = [ln(S/K) + (r - q) T + vol^2 t1 / 2] / (vol sqrt(t1)).
    t1 = T is a straddle; t1 -> 0 the dearer of the two today.

    conventions:
    - choose_time t1 and maturity T: years from today, 0 < t1 <= T
    - rate r and dividend_yield q continuously compounded and annual; vol annual; arrays broadcast
    returns: the price today, a float or an array
    source: Rubinstein (1991), "Options for the undecided", Risk 4(4); Hull, "Chooser options"; Haug (2007), 4.6.1
    aliases: chooser option, simple chooser, Rubinstein chooser, as-you-like-it option, [chooser]
    example:
        qf.chooser_price(spot=100.0, strike=100.0, choose_time=0.25, maturity=1.0, rate=0.03,
                         dividend_yield=0.01,
                         vol=0.25)
    """
    s, k, t, r, q, v = _inputs(spot, strike, maturity, rate, dividend_yield, vol)
    t1 = np.asarray(choose_time, dtype=float)
    positive("vol", v)
    if np.any(~(t1 > 0)) or np.any(t1 > t):
        raise ValueError("choose_time must lie in (0, maturity]")
    share, cash = s * np.exp(-q * t), k * np.exp(-r * t)
    d = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * np.sqrt(t))
    y = (np.log(s / k) + (r - q) * t + 0.5 * v * v * t1) / (v * np.sqrt(t1))
    return out(share * ndtr(d) - cash * ndtr(d - v * np.sqrt(t)) - share * ndtr(-y) + cash * ndtr(-y + v * np.sqrt(t1)))


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('choose_time', 'call_maturity', 'put_maturity',),
        price=('spot', 'call_strike', 'put_strike',))
def complex_chooser_price(spot, call_strike, put_strike, choose_time, call_maturity, put_maturity, rate,
                          dividend_yield, vol):
    """Complex chooser (Rubinstein): a chooser of two unequal legs.

    At the choice time t the holder takes the dearer of a call (strike Kc, expiry Tc) and a put (Kp, Tp); I is the
    spot at which the two are worth the same then, c(I, Kc, Tc - t) = p(I, Kp, Tp - t). With b = r - q,
    d1 = [ln(S/I) + (b + vol^2/2) t] / (vol sqrt t), d2 = d1 - vol sqrt t, y1 = [ln(S/Kc) + (b + vol^2/2) Tc] /
    (vol sqrt Tc), y2 = [ln(S/Kp) + (b + vol^2/2) Tp] / (vol sqrt Tp), rho1 = sqrt(t/Tc), rho2 = sqrt(t/Tp), M the
    bivariate normal CDF:
    V = S e^{-q Tc} M(d1, y1; rho1) - Kc e^{-r Tc} M(d2, y1 - vol sqrt Tc; rho1) - S e^{-q Tp} M(-d1, -y2; rho2)
        + Kp e^{-r Tp} M(-d2, -y2 + vol sqrt Tp; rho2).
    One strike and one expiry for both legs is the simple chooser, qf.chooser_price.

    conventions:
    - spot S; call_strike Kc, put_strike Kp; choose_time t, call_maturity Tc, put_maturity Tp: years from today, t < Tc, Tp
    - rate r and dividend_yield q continuously compounded; vol annual; every argument a number
    returns: a dict: price; critical_spot (I)
    source: Rubinstein (1991), "Options for the undecided", Risk 4(4); Haug (2007), "Complex chooser options"
    aliases: complex chooser, Rubinstein complex chooser
    example:
        import qf
        qf.complex_chooser_price(spot=50.0, call_strike=55.0, put_strike=48.0, choose_time=0.25,
                                 call_maturity=0.5, put_maturity=0.6, rate=0.1,
                                 dividend_yield=0.05, vol=0.35)  # result: price 6.07664469
    """
    s, kc, kp = (float(positive(name, value)) for name, value in
                 (("spot", spot), ("call_strike", call_strike), ("put_strike", put_strike)))
    t, tc, tp, v = (float(positive(name, value)) for name, value in
                    (("choose_time", choose_time), ("call_maturity", call_maturity), ("put_maturity", put_maturity),
                     ("vol", vol)))
    if not (t < tc and t < tp):
        raise ValueError("choose_time must come before call_maturity and put_maturity")
    r, q = float(rate), float(dividend_yield)

    def gap(x):
        return float(bs_price("call", x, kc, tc - t, r, q, v)) - float(bs_price("put", x, kp, tp - t, r, q, v))
    low, high = 1e-8 * min(kc, kp), 2.0 * max(kc, kp)
    while gap(high) < 0:
        high *= 2.0
    while gap(low) > 0:
        low *= 0.5
    critical = optimize.brentq(gap, low, high, xtol=1e-14 * high, rtol=1e-15, maxiter=500)
    b = r - q
    d1 = (np.log(s / critical) + (b + 0.5 * v * v) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    y1 = (np.log(s / kc) + (b + 0.5 * v * v) * tc) / (v * np.sqrt(tc))
    y2 = (np.log(s / kp) + (b + 0.5 * v * v) * tp) / (v * np.sqrt(tp))
    rho1, rho2 = np.sqrt(t / tc), np.sqrt(t / tp)
    m = _bivariate_normal_cdf
    price = (s * np.exp(-q * tc) * m(d1, y1, rho1) - kc * np.exp(-r * tc) * m(d2, y1 - v * np.sqrt(tc), rho1)
             - s * np.exp(-q * tp) * m(-d1, -y2, rho2) + kp * np.exp(-r * tp) * m(-d2, -y2 + v * np.sqrt(tp), rho2))
    return Result({"price": float(price), "critical_spot": float(critical)})


POWER_PAYOFFS = ("standard", "capped", "powered")


def _power_moment(s, j, t, b, v):
    """E[S_T^j] under the pricing measure, ln S_T ~ N(ln S + (b - vol^2/2) T, vol^2 T)."""
    return s ** j * np.exp(j * (b - 0.5 * v * v) * t + 0.5 * (j * v) ** 2 * t)


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot',))
def power_price(option, payoff, spot, strike, maturity, rate, dividend_yield, vol, power, cap):
    """Power option price (Haug): on S^i, capped, or powered.

    S_T^i is lognormal with mean M = S^i e^{i (b - vol^2/2) T + i^2 vol^2 T / 2} (b = r - q) and vol |i| vol, so
    'standard' (payoff max(+-(S_T^i - K), 0)) is Black-76 on M: e^{-rT} [M N(d1) - K N(d2)] for a call, d1 = [ln(M/K)
    + i^2 vol^2 T / 2] / (|i| vol sqrt T), d2 = d1 - |i| vol sqrt T; 'capped' pays at most `cap`: the call
    is the standard call at K less the one at K + cap, the put the put at K less the one at K - cap; 'powered' (payoff
    max(+-(S_T - K), 0)^i, i a whole number) is sum_j C(i, j) (-K)^(i - j) e^{-rT} E[S_T^j; S_T > K] for a call,
    E[S_T^j; S_T > K] = E[S_T^j] N(d + j vol sqrt T), d = [ln(S/K) + (b - vol^2/2) T] / (vol sqrt T) (the put alike
    on S_T < K). The powered payoff is integrated directly against the lognormal law, avoiding subtraction of large
    moments for small payoffs. Power 1 is Black-Scholes in all three (a capped call a call spread).

    conventions:
    - option: 'call' or 'put'; payoff: 'standard', 'capped' or 'powered' (above)
    - spot: S; strike: K, on the scale of S^i for 'standard' and 'capped', of S for 'powered'
    - power: i, any nonzero number ('powered': a whole number, at least 1); cap: the most paid ('capped'), else None
    - rate r and dividend_yield q continuously compounded; maturity T in years; vol: the annual vol of S
    returns: the price, a float
    source: Haug (2007), "Power contracts and power options"; Zhang (1998), Exotic Options, "Power options"
    aliases: power option, capped power option, powered option, symmetric power option
    example:
        import qf
        qf.power_price(option="call", payoff="standard", spot=100.0, strike=10000.0, maturity=0.5,
                       rate=0.05, dividend_yield=0.0, vol=0.2, power=2.0,
                       cap=None)  # result: 1524.60226422
    """
    choice("option", option, OPTIONS)
    choice("payoff", payoff, POWER_PAYOFFS)
    s, t, v = float(positive("spot", spot)), float(positive("maturity", maturity)), float(positive("vol", vol))
    k, r, q, i = float(strike), float(rate), float(dividend_yield), float(power)
    if i == 0:
        raise ValueError("power must not be 0")
    if (payoff == "capped") != (cap is not None):
        raise ValueError("cap is the most a 'capped' option pays, and None for the other payoffs")
    b, disc, sign = r - q, np.exp(-r * t), (1.0 if option == "call" else -1.0)
    if payoff == "powered":
        if i != int(i) or i < 1 or not k > 0:
            raise ValueError("'powered' takes a whole power of at least 1 and a positive strike")
        n = int(i)
        if n == 1:
            return float(bs_price(option, s, k, t, r, q, v))
        sd = v * np.sqrt(t)
        boundary = (np.log(k / s) - (b - 0.5 * v * v) * t) / sd

        def log_integrand(x):
            u = sd * x
            log_payoff = np.log(k) + np.log(-np.expm1(-u)) + (u if sign > 0 else 0.0)
            return n * log_payoff - 0.5 * (boundary + sign * x) ** 2 - 0.5 * np.log(2.0 * np.pi)

        def slope(x):
            if x == 0:
                return np.inf
            u = sd * x
            payoff_slope = n * sd / -np.expm1(-u)
            return payoff_slope * (1.0 if sign > 0 else np.exp(-u)) - sign * boundary - x

        upper = abs(boundary) + n * sd + np.sqrt(n) + 1.0
        mode = optimize.brentq(slope, 0.0, upper, xtol=1e-14, rtol=1e-14)
        peak = log_integrand(mode)

        def scaled(y):
            x = mode + y
            return np.exp(log_integrand(x) - peak) if x > 0 else 0.0

        total = (integrate.quad(scaled, -np.inf, 0.0, epsabs=1e-12, epsrel=1e-12)[0]
                 + integrate.quad(scaled, 0.0, np.inf, epsabs=1e-12, epsrel=1e-12)[0])
        return float(np.exp(peak - r * t) * total)
    mean, sd = _power_moment(s, i, t, b, v), abs(i) * v * np.sqrt(t)

    def standard(level):
        if level <= 0:
            return disc * (mean - level) if option == "call" else 0.0
        d1 = (np.log(mean / level) + 0.5 * sd * sd) / sd
        return disc * sign * (mean * ndtr(sign * d1) - level * ndtr(sign * (d1 - sd)))
    if payoff == "standard":
        return float(standard(k))
    c = float(positive("cap", cap))
    return float(standard(k) - standard(k + sign * c))


STRIKE_TYPES = ("floating", "fixed")
MONITORINGS = ("continuous",)
BGK_BETA = 0.5825971579390106   # -zeta(1/2)/sqrt(2 pi): Broadie, Glasserman and Kou's shift


def _bracket(s, level, t, b, v, side):
    """Haug's lookback term sigma^2/(2b) [...] at the extremum `level`, side +1 (a maximum) or -1 (a minimum), and its
    b = 0 limit sigma sqrt(T) [phi(d) +- d N(+-d)]."""
    sd = v * np.sqrt(t)
    d = (np.log(s / level) + (b + 0.5 * v * v) * t) / sd
    if abs(b) < 1e-12:
        return sd * (_density(d) + side * d * ndtr(side * d))
    power = -2.0 * b / (v * v) * np.log(s / level)          # in logs: a low vol's power would overflow
    shift = 2.0 * b / v * np.sqrt(t)
    return v * v / (2.0 * b) * side * (-np.exp(power + log_ndtr(side * (d - shift)))
                                       + np.exp(b * t + log_ndtr(side * d)))


def _lookback(option, strike_type, s, k, extreme, t, r, q, v):
    b, sd = r - q, v * np.sqrt(t)
    if strike_type == "floating":
        level, side, intrinsic = extreme, (-1.0 if option == "call" else 1.0), 0.0
    else:
        level = max(k, extreme) if option == "call" else min(k, extreme)
        side, intrinsic = (1.0, max(extreme - k, 0.0)) if option == "call" else (-1.0, max(k - extreme, 0.0))
    d = (np.log(s / level) + (b + 0.5 * v * v) * t) / sd
    phi = 1.0 if option == "call" else -1.0
    plain = phi * (s * np.exp(-q * t) * ndtr(phi * d) - level * np.exp(-r * t) * ndtr(phi * (d - sd)))
    return np.exp(-r * t) * intrinsic + plain + s * np.exp(-r * t) * _bracket(s, level, t, b, v, side)


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike', 'extreme',))
def lookback_price(option, strike_type, spot, strike, extreme, maturity, rate, dividend_yield, vol, monitoring):
    """Lookback option price: floating strike (Goldman-Sosin-Gatto), fixed strike (Conze-Viswanathan), discrete shift.

    The running maximum M or minimum m of S over the life. Floating: call S_T - m_T, put M_T - S_T. Fixed: call
    max(M_T - K, 0), put max(K - m_T, 0). With b = r - q, d(L) = (ln(S/L) + (b + vol^2/2) T)/(vol sqrt T) and Haug's
    terms: floating call S e^{-qT} N(d) - m e^{-rT} N(d - vol sqrt T) + S e^{-rT} vol^2/(2b) [(S/m)^{-2b/vol^2}
    N(-d + 2b sqrt(T)/vol) - e^{bT} N(-d)] at L = m; fixed call e^{-rT} max(M - K, 0) + S e^{-qT} N(d) - L e^{-rT}
    N(d - vol sqrt T) + S e^{-rT} vol^2/(2b) [-(S/L)^{-2b/vol^2} N(d - 2b sqrt(T)/vol) + e^{bT} N(d)] at L = max(K,
    M); the puts mirror them; b = 0 takes the limit. Discrete monitoring every dt (today a monitoring date), Broadie,
    Glasserman and Kou's shift, a = e^{0.5826 vol sqrt(dt)}: fixed call ~ C(K a, M a)/a, fixed put ~ a P(K/a, m/a),
    floating put ~ P(M a)/a - (1 - 1/a) S e^{-qT}, floating call ~ a C(m/a) - (a - 1) S e^{-qT}.

    conventions:
    - option: 'call' or 'put'; strike_type: 'floating' (strike None) or 'fixed' (strike K)
    - extreme: the extremum observed so far, M for a fixed call or a floating put, m for a fixed put or a floating
      call (spot itself for a new option)
    - rate r, dividend_yield q continuously compounded and annual; maturity: T in years; vol: annual
    - monitoring: 'continuous' or the monitoring interval dt in years (1/252 for daily closes)
    returns: the price today, a float
    source: Goldman, Sosin and Gatto (1979), J. Finance 34(5); Conze and Viswanathan (1991), J. Finance 46(5);
      Broadie, Glasserman and Kou (1999), Finance and Stochastics 3; Haug (2007), 4.15
    aliases: lookback option, lookback call, lookback put, floating strike lookback, fixed strike lookback, Goldman-Sosin-Gatto, Conze-Viswanathan, running maximum option, discrete lookback
    example:
        qf.lookback_price(option="call", strike_type="floating", spot=120.0, strike=None, extreme=100.0,
                          maturity=0.5, rate=0.10, dividend_yield=0.04, vol=0.30,
                          monitoring="continuous")
    """
    choice("option", option, OPTIONS)
    choice("strike_type", strike_type, STRIKE_TYPES)
    s, t, v = float(positive("spot", spot)), float(positive("maturity", maturity)), float(positive("vol", vol))
    r, q, m = float(rate), float(dividend_yield), float(positive("extreme", extreme))
    if strike_type == "fixed":
        k = float(positive("strike", strike))
    elif strike is not None:
        raise ValueError("a floating-strike lookback takes strike None (its strike is the extremum)")
    else:
        k = 0.0
    is_max = (option == "call") == (strike_type == "fixed")
    if (is_max and m < s) or (not is_max and m > s):
        raise ValueError("extreme must be the running maximum (>= spot) for a fixed call or floating put, the "
                         "running minimum (<= spot) for a fixed put or floating call")
    if isinstance(monitoring, str):
        choice("monitoring", monitoring, MONITORINGS)
        return float(_lookback(option, strike_type, s, k, m, t, r, q, v))
    a = float(np.exp(BGK_BETA * v * np.sqrt(float(positive("monitoring", monitoring)))))
    if strike_type == "fixed":
        if option == "call":
            return float(_lookback(option, "fixed", s, k * a, m * a, t, r, q, v) / a)
        return float(a * _lookback(option, "fixed", s, k / a, m / a, t, r, q, v))
    forward_value = s * np.exp(-q * t)
    if option == "put":
        return float(_lookback(option, "floating", s, 0.0, m * a, t, r, q, v) / a - (1.0 - 1.0 / a) * forward_value)
    return float(a * _lookback(option, "floating", s, 0.0, m / a, t, r, q, v) - (a - 1.0) * forward_value)
