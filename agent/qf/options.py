"""Vanilla options: Black-Scholes-Merton, Black-76, Bachelier, CEV; Greeks; American (BAW, trees, Roll-Geske-Whaley).

Every function takes continuously compounded annual rates (`rate` r, `dividend_yield` q), times in years and an annual
`vol`; arrays broadcast, and a scalar result is a float, a function with several results a dict with named keys. The
exotic closed forms are on `exotics`, the implied vols and the parity forward on `impvol`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import gammaincc

from ._common import Result, bivariate_normal_cdf, choice, log_ndtr, ndtr, out, positive
from .primitives import _guard

__all__ = ["american_price", "bachelier_price", "binomial_price", "black76_price", "bs_greeks", "bs_price", "cev_price",
           "escrowed_dividend_option", "roll_geske_whaley"]

PER_UNIT = (1.0, 0.01)
THETA_DAYS = {"year": 1.0, "calendar_day": 365.0, "trading_day": 252.0}
OPTIONS = ("call", "put")


def _density(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def bs_greeks(option, spot, strike, maturity, rate, dividend_yield, vol, vega_per, theta_per, rho_per):
    """Black-Scholes-Merton price, delta, gamma, vega, theta and rho, their units named.

    d1 = [ln(S/K) + (r - q + vol^2/2) T] / (vol sqrt T), d2 = d1 - vol sqrt T, n() the standard normal density;
    call: delta = e^{-qT} N(d1), rho = K T e^{-rT} N(d2),
          theta = -S e^{-qT} n(d1) vol / (2 sqrt T) + q S e^{-qT} N(d1) - r K e^{-rT} N(d2);
    put:  delta = -e^{-qT} N(-d1), rho = -K T e^{-rT} N(-d2),
          theta = -S e^{-qT} n(d1) vol / (2 sqrt T) - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2);
    both: gamma = e^{-qT} n(d1) / (S vol sqrt T), vega = S e^{-qT} n(d1) sqrt T.

    Black-76 on a futures or forward price F: these with spot = F and dividend_yield = rate (a call's delta
    e^{-rT} N(d1) per unit of F), but rho: Black-76's, F held, is -T times the price.

    FX (Garman-Kohlhagen, dividend_yield = r_f): rho is dV/dr_d; the foreign-rate rho dV/dr_f is -T S e^{-r_f T}
    N(d1) for a call, T S e^{-r_f T} N(-d1) for a put. A premium in the foreign currency is V/S. The skew-adjusted
    delta is delta + vega dvol/dS: -(K/S) times the smile's slope in strike where the smile moves with the spot.

    conventions:
    - vega_per: 1.0 for the plain derivative dV/dvol (an analytic vega, the PDE's); 0.01 only where the task asks
      the change for a one-point (1%) move in the vol
    - theta_per: 'year' for the plain dV/dt per year (the Black-Scholes PDE's theta); 'calendar_day' where the task
      asks the decay over one day of 365; 'trading_day' over one of 252
    - rho_per: 1.0 for the plain derivative dV/dr; 0.01 only where the task asks the change for a one-point move
    - rate and dividend_yield continuously compounded; maturity and vol must be positive
    returns: a dict: price; delta (per unit of spot); gamma (per unit of spot, squared); vega (per vega_per of vol);
      theta (dV/dt, the value's change as time passes, per theta_per); rho (per rho_per of the rate)
    source: Hull, Options, Futures, and Other Derivatives, "The Greek letters" (with a dividend yield)
    aliases: Black-Scholes Greeks, option Greeks, BS Greeks, analytic Greeks, delta, gamma, vega, theta, rho
    example:
        qf.bs_greeks(option="call", spot=49.0, strike=50.0, maturity=0.3846,
                     rate=0.05, dividend_yield=0.0, vol=0.2, vega_per=1.0,
                     theta_per="year", rho_per=1.0)
    """
    choice("option", option, OPTIONS)
    choice("vega_per", vega_per, PER_UNIT)
    choice("theta_per", theta_per, tuple(THETA_DAYS))
    choice("rho_per", rho_per, PER_UNIT)
    s, k = positive("spot", spot), positive("strike", strike)
    t, v = positive("maturity", maturity), positive("vol", vol)
    r, q = np.asarray(rate, dtype=float), np.asarray(dividend_yield, dtype=float)
    sign = 1.0 if option == "call" else -1.0
    sd = v * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / sd
    d2 = d1 - sd
    fs, fk = s * np.exp(-q * t), k * np.exp(-r * t)
    n1 = _density(d1)
    theta = -fs * n1 * v / (2.0 * np.sqrt(t)) + sign * (q * fs * ndtr(sign * d1) - r * fk * ndtr(sign * d2))
    return Result({"price": bs_price(option, s, k, t, r, q, v),
                   "delta": out(sign * np.exp(-q * t) * ndtr(sign * d1)),
                   "gamma": out(np.exp(-q * t) * n1 / (s * sd)),
                   "vega": out(fs * n1 * np.sqrt(t) * vega_per),
                   "theta": out(theta / THETA_DAYS[theta_per]),
                   "rho": out(sign * fk * t * ndtr(sign * d2) * rho_per)})


@_guard(vol=('vol',), time=('expiry',), price=('forward', 'strike',))
def black76_price(option, forward, strike, expiry, vol, discount_factor):
    """Black-76 price of a European option on a forward or futures price.

    c = DF [F N(d1) - K N(d2)],  p = DF [K N(-d2) - F N(-d1)],
    d1 = [ln(F/K) + vol^2 T / 2] / (vol sqrt T),  d2 = d1 - vol sqrt T;
    where vol sqrt T is 0 the price is DF max(+-(F - K), 0). Parity: c - p = DF (F - K).

    conventions:
    - forward: the forward or futures price for delivery at the option's expiry (or the forward rate of a caplet)
    - expiry: years to the option's expiry (a caplet's fixing time), the time the vol applies over
    - discount_factor: the discount from the payment date to today, e^{-rT} for a payment at expiry
    returns: a float, or an array for arrays: the price, discounted by discount_factor
    source: Black (1976), "The pricing of commodity contracts", J. Financial Economics 3; Hull, "Black's model"
    aliases: Black-76, Black 76, Black's model, Black model, futures option, option on a forward, Black formula
    example:
        qf.black76_price(option="put", forward=20.0, strike=20.0,
                         expiry=0.3333, vol=0.25,
                         discount_factor=0.9704)
    """
    choice("option", option, OPTIONS)
    f, k = positive("forward", forward), positive("strike", strike)
    df = positive("discount_factor", discount_factor)
    return bs_price(option, f * df, k * df, expiry, 0.0, 0.0, vol)


@_guard(rate=('rate',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def escrowed_dividend_option(option, spot, strike, maturity, rate, vol, dividend_amounts, dividend_times):
    """European option price and delta with known cash dividends, by the escrowed-dividend model.

    S* = S - sum_i D_i e^{-r t_i} over the dividends with 0 < t_i <= T (the remaining ones before expiry), then the
    Black-Scholes price on S* with no yield; delta = dV/dS = N(d1(S*)) for a call and N(d1(S*)) - 1 for a put, since
    the dividends' PV does not move with S.

    conventions:
    - dividend_times: years from today to each ex-dividend date; times <= 0 (paid) and > maturity are left out
    - rate: continuously compounded, discounting the dividends and the strike alike
    - spot: the observed price, dividends not yet removed; maturity and vol must be positive
    returns: a dict: price; delta (dV/dS per unit of the observed spot); escrowed_spot (S*, the spot
      less the dividends' present value)
    source: Hull, Options, Futures, and Other Derivatives, "Dividends" (European options on dividend-paying stocks)
    aliases: discrete dividends, escrowed dividend, [cash dividends], dividend-adjusted Black-Scholes, [known dividends]
    example:
        qf.escrowed_dividend_option(option="call", spot=40.0, strike=40.0, maturity=0.5,
                                    rate=0.09, vol=0.3, dividend_amounts=[0.5, 0.5],
                                    dividend_times=[0.1667, 0.4167])
    """
    choice("option", option, OPTIONS)
    amounts = np.asarray(dividend_amounts, dtype=float).ravel()
    times = np.asarray(dividend_times, dtype=float).ravel()
    if amounts.shape != times.shape:
        raise ValueError("dividend_amounts and dividend_times must have the same length")
    if np.any(amounts < 0):
        raise ValueError("dividend_amounts must not be negative")
    s, k = positive("spot", spot), positive("strike", strike)
    t, v = positive("maturity", maturity), positive("vol", vol)
    r = np.asarray(rate, dtype=float)
    live = (times > 0) & (times <= t[..., None])
    pv = np.sum(np.where(live, amounts * np.exp(-r[..., None] * times), 0.0), axis=-1)
    s_star = s - pv
    if np.any(s_star <= 0):
        raise ValueError("spot must exceed the present value of the dividends")
    sd = v * np.sqrt(t)
    d1 = (np.log(s_star / k) + (r + 0.5 * v * v) * t) / sd
    delta = ndtr(d1) if option == "call" else ndtr(d1) - 1.0
    return Result({"price": bs_price(option, s_star, k, t, r, 0.0, v), "delta": out(delta),
                   "escrowed_spot": out(s_star)})


def _inputs(spot, strike, maturity, rate, dividend_yield, vol):
    s, k = positive("spot", spot), positive("strike", strike)
    t, r, q, v = (np.asarray(x, dtype=float) for x in (maturity, rate, dividend_yield, vol))
    if np.any(t < 0) or np.any(v < 0):
        raise ValueError("maturity and vol must not be negative")
    return s, k, t, r, q, v


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def bs_price(option, spot, strike, maturity, rate, dividend_yield, vol):
    """Black-Scholes-Merton price of European calls or puts with a dividend yield; arrays broadcast.

    c = S e^{-qT} N(d1) - K e^{-rT} N(d2),   p = K e^{-rT} N(-d2) - S e^{-qT} N(-d1),
    d1 = [ln(S/K) + (r - q + vol^2/2) T] / (vol sqrt(T)),   d2 = d1 - vol sqrt(T).

    `option` 'call' or 'put'; r and q continuously compounded and annual, T in years, vol annual. Where vol sqrt(T) is 0
    the price is the discounted forward's intrinsic value, max(+-(S e^{-qT} - K e^{-rT}), 0).

    No-arbitrage bounds: max(S e^{-qT} - K e^{-rT}, 0) <= c <= S e^{-qT}; max(K e^{-rT} - S e^{-qT}, 0) <= p <=
    K e^{-rT}.
    Source: Black and Scholes (1973) and Merton (1973); Hull, Options, Futures, and Other Derivatives, the
    Black-Scholes-Merton pricing formulas, with the dividend yield of its chapter on options on indices and currencies.
    aliases: [Black-Scholes price], [Black-Scholes-Merton], [European option price], [BSM price]
    returns: the price today, a float or an array
    example:
        qf.bs_price(option="call", spot=100.0, strike=95.0, maturity=0.5, rate=0.03, dividend_yield=0.01,
                    vol=0.25)
    """
    choice("option", option, OPTIONS)
    s, k, t, r, q, v = _inputs(spot, strike, maturity, rate, dividend_yield, vol)
    sign = 1.0 if option == "call" else -1.0
    fs, fk = s * np.exp(-q * t), k * np.exp(-r * t)
    sd = v * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(fs / fk) + 0.5 * sd * sd) / sd
        price = sign * (fs * ndtr(sign * d1) - fk * ndtr(sign * (d1 - sd)))
    return out(np.where(sd > 0, price, np.maximum(sign * (fs - fk), 0.0)))


# ------------------------------------------------------------------------------------------------ other diffusions

VOL_FORMS = ("alpha", "at_spot")


@_guard(time=('expiry',))
def bachelier_price(option, forward, strike, expiry, vol, discount_factor):
    """Bachelier price (the normal model) on a forward, negative rates allowed.

    c = DF [(F - K) N(d) + vol sqrt(T) n(d)],  p = DF [(K - F) N(-d) + vol sqrt(T) n(d)],  d = (F - K) / (vol sqrt(T)),
    n the standard normal density; at the money c = p = DF vol sqrt(T / (2 pi)); where vol sqrt(T) is 0 the price is
    DF max(+-(F - K), 0). Parity: c - p = DF (F - K). The forward moves by vol dW in price units (an arithmetic
    Brownian motion), so F and K may be zero or negative (rates, spreads).

    Not Black-76 on a shifted forward: that is lognormal in F + shift (qf.black76_price on F + s and K + s), another
    model with other prices. The normal implied vol of a price is qf.normal_implied_vol.

    conventions:
    - option: 'call' or 'put'; forward: F for the expiry; strike: K, in F's units
    - expiry: T in years; vol: the normal (absolute, basis-point) vol in F's units a year, 0.0080 for 80 bp a year on
      a rate quoted in decimals; not a lognormal (percentage) vol
    - discount_factor: DF from the payment date, e^{-rT}; a swaption passes its annuity here
    returns: the price, a float or an array
    source: Bachelier (1900), Theorie de la speculation; Hull, Options, Futures, and Other Derivatives, the normal
      model of rates; Tuckman and Serrat, Fixed Income Securities, normal volatility
    aliases: Bachelier, Bachelier model, Bachelier formula, normal model
    example:
        import qf
        qf.bachelier_price(option="call", forward=0.03, strike=0.025, expiry=1.0,
                           vol=0.008, discount_factor=0.97)  # result: 0.00610650
    """
    choice("option", option, OPTIONS)
    f, k = np.asarray(forward, dtype=float), np.asarray(strike, dtype=float)
    t, v = np.asarray(expiry, dtype=float), np.asarray(vol, dtype=float)
    df = positive("discount_factor", discount_factor)
    if np.any(t < 0) or np.any(v < 0):
        raise ValueError("expiry and vol must not be negative")
    sign = 1.0 if option == "call" else -1.0
    sd = v * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (f - k) / sd
        price = sign * (f - k) * ndtr(sign * d) + sd * _density(d)
    return out(df * np.where(sd > 0, price, np.maximum(sign * (f - k), 0.0)))


@_guard(rate=('rate',), vol=('vol',), time=('maturity', 'dividend_time',), price=('spot', 'strike',))
def roll_geske_whaley(spot, strike, maturity, rate, vol, dividend_amount, dividend_time):
    """Roll-Geske-Whaley (RGW): early exercise ahead of one dividend, exact.

    S' = S - D e^{-r t1} (the spot net of the dividend's present value, t1 the ex-dividend time); the call is exercised,
    if at all, just before t1, where the spot I solves c(I, K, T - t1) = I + D - K (c the Black-Scholes call):
    C = S' N(b1) + S' M(a1, -b1; -sqrt(t1/T)) - K e^{-rT} M(a2, -b2; -sqrt(t1/T)) - (K - D) e^{-r t1} N(b2),
    a1 = [ln(S'/K) + (r + vol^2/2) T] / (vol sqrt T), a2 = a1 - vol sqrt T, b1 = [ln(S'/I) + (r + vol^2/2) t1] /
    (vol sqrt t1), b2 = b1 - vol sqrt t1, M the bivariate normal CDF. Where D <= K (1 - e^{-r (T - t1)}) early exercise
    never pays: the price is the European on S', qf.escrowed_dividend_option's. No dividend: Black-Scholes.
    Where D >= K, exercise always pays at t1 and the price is S - K e^{-r t1}.

    conventions:
    - spot: the observed price, the dividend not yet paid (its present value is taken off inside); strike: K
    - maturity: T in years; dividend_time: t1, years to the ex-dividend date, 0 < t1 < T; dividend_amount: D, cash
    - rate: r continuously compounded, at least zero (early exercise at other times can matter at negative rates);
      vol: the annual vol of S' (the escrowed model), not of S
    returns: a dict: price (the American call); european (on S'); critical_spot (I, the ex-dividend spot above which
      the call is exercised just before t1, observed spot I + D; inf where never, 0 where always); net_spot (S')
    source: Roll (1977), Geske (1979), Whaley (1981), J. Financial Economics 4, 7, 9; Haug (2007), "Roll-Geske-Whaley";
      Hull, technical note 4; Lu (2026), arXiv:2607.24973, section 3.2 (dividend at least the strike)
    aliases: Roll-Geske-Whaley, RGW, Roll Geske Whaley
    example:
        import qf
        qf.roll_geske_whaley(spot=80.0, strike=82.0, maturity=0.5, rate=0.06, vol=0.3,
                             dividend_amount=4.0, dividend_time=0.25)  # result: price 5.25601706
    """
    s, k = float(positive("spot", spot)), float(positive("strike", strike))
    t, v = float(positive("maturity", maturity)), float(positive("vol", vol))
    t1, d, r = float(positive("dividend_time", dividend_time)), float(dividend_amount), float(rate)
    if not r >= 0:
        raise ValueError("rate must not be negative: early exercise need not be confined to the dividend date")
    if not t1 < t or d < 0:
        raise ValueError("dividend_time must come before maturity and dividend_amount must not be negative")
    net = s - d * np.exp(-r * t1)
    if net <= 0:
        raise ValueError("spot must exceed the dividend's present value")
    european = float(bs_price("call", net, k, t, r, 0.0, v))
    if d >= k:
        return Result({"price": float(s - k * np.exp(-r * t1)), "european": european,
                       "critical_spot": 0.0, "net_spot": float(net)})
    if d <= k * -np.expm1(-r * (t - t1)):
        return Result({"price": european, "european": european, "critical_spot": np.inf, "net_spot": float(net)})

    def gap(x):
        return float(bs_price("call", x, k, t - t1, r, 0.0, v)) - x - d + k
    high = 2.0 * k
    while gap(high) > 0:
        high *= 2.0
    critical = brentq(gap, 1e-12 * k, high, xtol=1e-14 * k, rtol=1e-15, maxiter=500)
    rho = -np.sqrt(t1 / t)
    a1 = (np.log(net / k) + (r + 0.5 * v * v) * t) / (v * np.sqrt(t))
    b1 = (np.log(net / critical) + (r + 0.5 * v * v) * t1) / (v * np.sqrt(t1))
    a2, b2 = a1 - v * np.sqrt(t), b1 - v * np.sqrt(t1)
    price = (net * ndtr(b1) + net * bivariate_normal_cdf(a1, -b1, rho)
             - k * np.exp(-r * t) * bivariate_normal_cdf(a2, -b2, rho) - (k - d) * np.exp(-r * t1) * ndtr(b2))
    return Result({"price": float(price), "european": european, "critical_spot": float(critical), "net_spot": float(net)})


@_guard(rate=('rate', 'dividend_yield',), time=('maturity',), price=('spot', 'strike',))
def cev_price(option, spot, strike, maturity, rate, dividend_yield, vol, beta, vol_form):
    """CEV (constant elasticity of variance) price, by Schroder's formula.

    dS = (r - q) S dt + alpha S^beta dW, the local vol alpha S^(beta - 1). With v = alpha^2 [e^{2 (r - q)(beta - 1) T}
    - 1] / [2 (r - q)(beta - 1)] (alpha^2 T where r = q), a = [K e^{-(r - q) T}]^{2 (1 - beta)} / [(1 - beta)^2 v],
    b = 1 / (1 - beta), c = S^{2 (1 - beta)} / [(1 - beta)^2 v], X2(z; k, l) the noncentral chi-squared CDF at z
    with k degrees of freedom and noncentrality l (scipy.stats.ncx2.cdf(z, k, l)):
    beta < 1: call S e^{-qT} [1 - X2(a; b + 2, c)] - K e^{-rT} X2(c; b, a), put K e^{-rT} [1 - X2(c; b, a)] -
      S e^{-qT} X2(a; b + 2, c);
    beta > 1: call S e^{-qT} [G(-b/2, c/2) - X2(c; -b, a)] - K e^{-rT} X2(a; 2 - b, c), put K e^{-rT} [1 - X2(a; 2 - b, c)] -
      S e^{-qT} X2(c; -b, a);
    G is the regularized lower incomplete gamma. Above 1, the discounted, dividend-adjusted stock is a strict local martingale: its expected
    value is S e^{-qT} G(-b/2, c/2), so call-put parity uses this expectation rather than S e^{-qT}.
    beta = 1 is Black-Scholes with vol alpha (qf.bs_price). Below 1 the price can reach 0 and stays there (absorbed),
    so the vol rises as the price falls (the leverage effect, a skew).

    conventions:
    - option: 'call' or 'put'; spot S, strike K; rate r and dividend_yield q continuously compounded; maturity T
    - beta: the elasticity, 0 <= beta (beta = 1 is lognormal, 0.5 the square-root process, 0 the absolute diffusion)
    - vol_form: 'alpha' (vol is alpha itself, the coefficient of S^beta) or 'at_spot' (vol is the volatility at today's
      spot, an at-the-money vol read as the local vol there: alpha = vol S^(1 - beta))
    returns: a dict: price; alpha (the coefficient used)
    source: Cox (1975), Stanford working paper; Schroder (1989), J. Finance 44(1); Hull, Options, Futures, and Other
      Derivatives, "The constant elasticity of variance model"; QuantLib 1.43, CEVCalculator (beta > 1 correction)
    aliases: CEV, CEV model, constant elasticity of variance, Schroder, Cox CEV
    example:
        import qf
        qf.cev_price(option="call", spot=100.0, strike=100.0, maturity=1.0, rate=0.05,
                     dividend_yield=0.0, vol=0.2, beta=0.5,
                     vol_form="at_spot")  # result: price 10.45388533, alpha 2.0
    """
    choice("option", option, OPTIONS)
    choice("vol_form", vol_form, VOL_FORMS)
    s, k = float(positive("spot", spot)), float(positive("strike", strike))
    t, v = float(positive("maturity", maturity)), float(positive("vol", vol))
    r, q, b_el = float(rate), float(dividend_yield), float(beta)
    if not b_el >= 0:
        raise ValueError("beta must be at least 0")
    alpha = v if vol_form == "alpha" else v * s ** (1.0 - b_el)
    if b_el == 1.0:
        return Result({"price": float(bs_price(option, s, k, t, r, q, alpha)), "alpha": alpha})
    carry = (r - q) * (b_el - 1.0)
    var = alpha * alpha * (np.expm1(2.0 * carry * t) / (2.0 * carry) if carry != 0 else t)
    one = 1.0 - b_el
    a = (k * np.exp(-(r - q) * t)) ** (2.0 * one) / (one * one * var)
    c = s ** (2.0 * one) / (one * one * var)
    b = 1.0 / one
    share, cash = s * np.exp(-q * t), k * np.exp(-r * t)
    if b_el < 1.0:
        share_in, cash_in = (a, b + 2.0, c), (c, b, a)
    else:
        share_in, cash_in = (c, -b, a), (a, 2.0 - b, c)
    share_tail = stats.ncx2.sf(*share_in)
    if b_el > 1.0:
        share_tail -= gammaincc(-0.5 * b, 0.5 * c)
    call = share * share_tail - cash * stats.ncx2.cdf(*cash_in)
    put = cash * stats.ncx2.sf(*cash_in) - share * stats.ncx2.cdf(*share_in)
    return Result({"price": float(call if option == "call" else put), "alpha": alpha})


# ------------------------------------------------------------------------------------------------ American exercise

AMERICAN_METHODS = ("barone_adesi_whaley", "bjerksund_stensland_1993", "bjerksund_stensland_2002")
EXERCISES = ("european", "american")
TREES = ("crr", "jarrow_rudd", "leisen_reimer", "trinomial")


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def american_price(method, option, spot, strike, maturity, rate, dividend_yield, vol):
    """American option price by approximation: Barone-Adesi-Whaley or Bjerksund-Stensland, early-exercise premium.

    b = r - q. 'barone_adesi_whaley' (quadratic): with N = 2b/vol^2, M = 2r/vol^2, K = 1 - e^{-rT}, q2 = (-(N - 1)
    + sqrt((N - 1)^2 + 4M/K))/2 and q1 its other root, the call is c(S) + A2 (S/S*)^q2 below the critical price S*
    and S - X above it, A2 = (S*/q2)(1 - e^{(b-r)T} N(d1(S*))), S* the root of S* - X = c(S*) + (1 - e^{(b-r)T}
    N(d1(S*))) S*/q2; the put p(S) + A1 (S/S**)^q1 above S** and X - S below, A1 = -(S**/q1)(1 - e^{(b-r)T}
    N(-d1(S**))). 'bjerksund_stensland_1993' exercises at one flat trigger I (the call is alpha S^beta less five
    phi terms below I); 'bjerksund_stensland_2002' at two, I2 before t1 = (sqrt(5) - 1) T/2 and I1 after (Psi terms
    of the bivariate normal); a put is the call transformed, P(S, X, T, r, b) = C(X, S, T, r - b, -b). A call with
    q <= 0, and a put with r <= 0, is never exercised early: the European (qf.bs_price). A fine
    qf.binomial_price tree is the benchmark.

    conventions:
    - method: 'barone_adesi_whaley', 'bjerksund_stensland_1993' or 'bjerksund_stensland_2002'
    - option: 'call' or 'put'; spot S and strike X; rate r and dividend_yield q continuously compounded and annual
      (carry b = r - q)
    - maturity: T in years, above 0; vol: annual, above 0; arrays broadcast
    returns: a dict: price (American); european (qf.bs_price); premium (price - european); critical (today's
      exercise boundary: S* for a call, S** for a put; inf for a call, 0 for a put never exercised early)
    source: Barone-Adesi and Whaley (1987), J. Finance 42(2); Bjerksund and Stensland (1993), Scand. J. Management
      9; Bjerksund and Stensland (2002), NHH discussion paper; Haug (2007), 3.3 and 3.4
    aliases: Barone-Adesi-Whaley, Barone-Adesi Whaley, BAW, Bjerksund-Stensland, quadratic approximation, American option approximation, early exercise premium, [American option]
    example:
        qf.american_price(method="barone_adesi_whaley", option="put", spot=100.0, strike=100.0,
                          maturity=0.5, rate=0.06, dividend_yield=0.02,
                          vol=0.25)
    """
    choice("method", method, AMERICAN_METHODS)
    choice("option", option, OPTIONS)
    s, k, t, r, q, v = _inputs(spot, strike, maturity, rate, dividend_yield, vol)
    if np.any(~(t > 0)) or np.any(~(v > 0)):
        raise ValueError("maturity and vol must be above 0")
    call = option == "call"

    def one(s1, k1, t1, r1, q1, v1):
        if method == "barone_adesi_whaley":
            return _baw(call, s1, k1, t1, r1, q1, v1)
        version = int(method[-4:])
        if call:
            return _bjerksund_call(s1, k1, t1, r1, r1 - q1, v1, version)
        price, trigger = _bjerksund_call(k1, s1, t1, q1, q1 - r1, v1, version)     # P(S, X, r, b) = C(X, S, r - b, -b)
        return price, (k1 * s1 / trigger if np.isfinite(trigger) else 0.0)

    price, critical = np.vectorize(one, otypes=[float, float])(s, k, t, r, q, v)
    european = np.asarray(bs_price(option, s, k, t, r, q, v), dtype=float)
    return Result({"price": out(price), "european": out(european), "premium": out(price - european),
                   "critical": out(critical)})


def _over_k(r, t):
    """r / (1 - e^{-rT}), 1/T at r = 0."""
    return r / -np.expm1(-r * t) if r != 0 else 1.0 / t


def _baw(call, s, k, t, r, q, v):
    b = r - q
    if (call and q <= 0) or (not call and r <= 0):
        return float(bs_price("call" if call else "put", s, k, t, r, q, v)), (np.inf if call else 0.0)
    sd, carry = v * np.sqrt(t), np.exp((b - r) * t)
    n, m_over_k = 2.0 * b / (v * v), 2.0 * _over_k(r, t) / (v * v)
    root = np.sqrt((n - 1.0) ** 2 + 4.0 * m_over_k)
    option = "call" if call else "put"

    def d1(x):
        return (np.log(x / k) + (b + 0.5 * v * v) * t) / sd

    if call:
        q2 = 0.5 * (-(n - 1.0) + root)

        def gap(x):
            return x - k - float(bs_price("call", x, k, t, r, q, v)) - (1.0 - carry * ndtr(d1(x))) * x / q2
        high = 2.0 * k
        while gap(high) < 0:
            high *= 2.0
        star = brentq(gap, k, high, xtol=1e-12 * k, rtol=1e-15, maxiter=500)
        if s >= star:
            return s - k, star
        a2 = star / q2 * (1.0 - carry * ndtr(d1(star)))
        return float(bs_price("call", s, k, t, r, q, v)) + a2 * (s / star) ** q2, star
    q1 = 0.5 * (-(n - 1.0) - root)

    def gap(x):
        return k - x - float(bs_price("put", x, k, t, r, q, v)) + (1.0 - carry * ndtr(-d1(x))) * x / q1
    star = brentq(gap, 1e-12 * k, k, xtol=1e-12 * k, rtol=1e-15, maxiter=500)
    if s <= star:
        return k - s, star
    a1 = -star / q1 * (1.0 - carry * ndtr(-d1(star)))
    return float(bs_price(option, s, k, t, r, q, v)) + a1 * (s / star) ** q1, star


def _phi(s, t, gamma, h, i, r, b, v, unit=1.0):
    """Bjerksund and Stensland's phi, e^lambda S^gamma [N(d) - (I/S)^kappa N(d - 2 ln(I/S)/(v sqrt t))], over
    unit^gamma, its powers taken in logs so a large gamma or kappa (a low vol) neither overflows nor cancels."""
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v * v) * t
    sd = v * np.sqrt(t)
    d = -(np.log(s / h) + (b + (gamma - 0.5) * v * v) * t) / sd
    kappa = 2.0 * b / (v * v) + 2.0 * gamma - 1.0
    lead = lam + gamma * np.log(s / unit)
    return (np.exp(lead + log_ndtr(d))
            - np.exp(lead + kappa * np.log(i / s) + log_ndtr(d - 2.0 * np.log(i / s) / sd)))


def _psi(s, t2, gamma, h, i2, i1, t1, r, b, v, unit=1.0):
    """Bjerksund and Stensland's (2002) Psi, over the bivariate normal M of correlation sqrt(t1/t2), over
    unit^gamma, its powers taken in logs as _phi's."""
    drift = b + (gamma - 0.5) * v * v
    e1 = (np.log(s / i1) + drift * t1) / (v * np.sqrt(t1))
    e2 = (np.log(i2 * i2 / (s * i1)) + drift * t1) / (v * np.sqrt(t1))
    e3 = (np.log(s / i1) - drift * t1) / (v * np.sqrt(t1))
    e4 = (np.log(i2 * i2 / (s * i1)) - drift * t1) / (v * np.sqrt(t1))
    f1 = (np.log(s / h) + drift * t2) / (v * np.sqrt(t2))
    f2 = (np.log(i2 * i2 / (s * h)) + drift * t2) / (v * np.sqrt(t2))
    f3 = (np.log(i1 * i1 / (s * h)) + drift * t2) / (v * np.sqrt(t2))
    f4 = (np.log(s * i1 * i1 / (h * i2 * i2)) + drift * t2) / (v * np.sqrt(t2))
    rho = np.sqrt(t1 / t2)
    lead = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v * v) * t2 + gamma * np.log(s / unit)
    kappa = 2.0 * b / (v * v) + 2.0 * gamma - 1.0

    def term(power, x, y, correlation):
        found = bivariate_normal_cdf(x, y, correlation)
        return np.exp(lead + power + np.log(found)) if found > 0 else 0.0

    with np.errstate(divide="ignore"):
        return (term(0.0, -e1, -f1, rho) - term(kappa * np.log(i2 / s), -e2, -f2, rho)
                - term(kappa * np.log(i1 / s), -e3, -f3, -rho) + term(kappa * np.log(i1 / i2), -e4, -f4, -rho))


def _bjerksund_call(s, k, t, r, b, v, version):
    """The call and its trigger today (Bjerksund and Stensland 1993 or 2002); b >= r, or a flat boundary at or below
    the strike (b T + 2 v sqrt(T) < 0 can put it there: no exercise region), the European and no trigger."""
    european = float(bs_price("call", s, k, t, r, r - b, v))
    if b >= r:
        return european, np.inf
    vv = v * v
    beta = (0.5 - b / vv) + np.sqrt((b / vv - 0.5) ** 2 + 2.0 * r / vv)
    b_inf, b0 = beta / (beta - 1.0) * k, max(k, r / (r - b) * k)
    if version == 1993:
        h = -(b * t + 2.0 * v * np.sqrt(t)) * b0 / (b_inf - b0)
        trigger = b0 + (b_inf - b0) * (1.0 - np.exp(h))
        if trigger <= k:
            return european, np.inf
        if s >= trigger:
            return s - k, trigger
        phi = lambda gamma, h_, unit=1.0: _phi(s, t, gamma, h_, trigger, r, b, v, unit)    # noqa: E731
        return ((trigger - k) * (np.exp(beta * np.log(s / trigger)) - phi(beta, trigger, trigger))
                + phi(1.0, trigger) - phi(1.0, k) - k * phi(0.0, trigger) + k * phi(0.0, k)), trigger
    t1 = 0.5 * (np.sqrt(5.0) - 1.0) * t
    scale = k * k / ((b_inf - b0) * b0)
    i1 = b0 + (b_inf - b0) * (1.0 - np.exp(-(b * t1 + 2.0 * v * np.sqrt(t1)) * scale))
    i2 = b0 + (b_inf - b0) * (1.0 - np.exp(-(b * t + 2.0 * v * np.sqrt(t)) * scale))
    if min(i1, i2) <= k:
        return european, np.inf
    if s >= i2:
        return s - k, i2
    phi = lambda gamma, h_, i, unit=1.0: _phi(s, t1, gamma, h_, i, r, b, v, unit)          # noqa: E731
    psi = lambda gamma, h_, unit=1.0: _psi(s, t, gamma, h_, i2, i1, t1, r, b, v, unit)    # noqa: E731
    return ((i2 - k) * (np.exp(beta * np.log(s / i2)) - phi(beta, i2, i2, i2)) + phi(1.0, i2, i2)
            - phi(1.0, i1, i2) - k * phi(0.0, i2, i2) + k * phi(0.0, i1, i2) + (i1 - k) * phi(beta, i1, i2, i1)
            - (i1 - k) * psi(beta, i1, i1) + psi(1.0, i1) - psi(1.0, k) - k * psi(0.0, i1) + k * psi(0.0, k)), i2


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def binomial_price(option, exercise, tree, spot, strike, maturity, rate, dividend_yield, vol, steps):
    """Binomial or trinomial tree (CRR, Jarrow-Rudd, Leisen-Reimer, Hull's trinomial) price and Greeks, European or American.

    dt = T/n; S at node (i, j) is S u^j d^(i - j); V = e^{-r dt} [p V_up + (1 - p) V_down] backward from the payoff,
    and under 'american' max(that, intrinsic) at every node. 'crr': u = e^{vol sqrt(dt)}, d = 1/u, p = (e^{(r-q)dt}
    - d)/(u - d); 'jarrow_rudd': u, d = e^{(r - q - vol^2/2) dt +- vol sqrt(dt)}, p = 1/2; 'leisen_reimer' (n odd):
    p = h(d2), p' = h(d1) by Peizer-Pratt inversion h(z) = 1/2 + sign(z) sqrt(1/4 - 1/4 exp(-(z/(n + 1/3 + 0.1/(n +
    1)))^2 (n + 1/6))), u = e^{(r-q)dt} p'/p, d = (e^{(r-q)dt} - p u)/(1 - p). From the first nodes (Hull): delta =
    (V_u - V_d)/(S_u - S_d); gamma the change of the step-2 deltas over (S_uu - S_dd)/2; theta = (V_ud - V)/(2 dt).
    European prices converge to qf.bs_price (CRR's error oscillates as 1/n; Leisen-Reimer's falls as 1/n^2).
    'trinomial' (Hull's): u = e^{vol sqrt(3 dt)}, d = 1/u, the middle node unmoved, p_m = 2/3, p_u and p_d = 1/6 +- (r - q
    - vol^2/2) sqrt(dt / (12 vol^2)); its Greeks from the first step's three nodes, theta (V_m - V)/dt.

    conventions:
    - option: 'call' or 'put'; exercise: 'european' or 'american' (at every node)
    - tree: 'crr', 'jarrow_rudd', 'leisen_reimer' (steps odd) or 'trinomial'
    - spot S, strike K; rate r and dividend_yield q continuously compounded and annual; maturity T and vol annual,
      each above 0; steps n, an int of at least 2; scalars
    returns: a dict: price; delta; gamma; theta (per year, Hull's from the nodes two steps in)
    source: Cox, Ross and Rubinstein (1979), J. Financial Economics 7; Jarrow and Rudd (1983), Option Pricing; Leisen
      and Reimer (1996), Applied Mathematical Finance 3; Hull, "Binomial trees in practice"
    aliases: binomial tree, binomial model, binomial lattice, CRR tree, Cox-Ross-Rubinstein, Jarrow-Rudd, Leisen-Reimer, American binomial tree, [trinomial tree], [trinomial lattice], [lattice]
    example:
        qf.binomial_price(option="put", exercise="american", tree="crr", spot=50.0, strike=52.0,
                          maturity=2.0, rate=0.05, dividend_yield=0.0, vol=0.3,
                          steps=500)
    """
    choice("option", option, OPTIONS)
    choice("exercise", exercise, EXERCISES)
    choice("tree", tree, TREES)
    s, k = float(positive("spot", spot)), float(positive("strike", strike))
    t, v = float(positive("maturity", maturity)), float(positive("vol", vol))
    r, q = float(rate), float(dividend_yield)
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 2:
        raise ValueError("steps must be an int of at least 2")
    n = int(steps)
    if tree == "leisen_reimer" and n % 2 == 0:
        raise ValueError("steps must be odd for 'leisen_reimer' (its tree centres on the strike at odd n)")
    dt = t / n
    growth, disc = np.exp((r - q) * dt), np.exp(-r * dt)
    if tree == "trinomial":
        return _trinomial(option == "call", exercise == "american", s, k, t, r, q, v, n)
    if tree == "crr":
        u = np.exp(v * np.sqrt(dt))
        d = 1.0 / u
        p = (growth - d) / (u - d)
    elif tree == "jarrow_rudd":
        mid = (r - q - 0.5 * v * v) * dt
        u, d, p = np.exp(mid + v * np.sqrt(dt)), np.exp(mid - v * np.sqrt(dt)), 0.5
    else:
        d1 = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * np.sqrt(t))

        def h(z):
            return 0.5 + np.sign(z) * np.sqrt(0.25 - 0.25 * np.exp(-(z / (n + 1.0 / 3.0 + 0.1 / (n + 1))) ** 2
                                                                       * (n + 1.0 / 6.0)))
        p, p_bar = h(d1 - v * np.sqrt(t)), h(d1)
        u = growth * p_bar / p
        d = (growth - p * u) / (1.0 - p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"the tree's probability {p} lies outside (0, 1): take more steps")
    sign = 1.0 if option == "call" else -1.0

    def prices(i):
        return s * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))

    values = np.maximum(sign * (prices(n) - k), 0.0)
    kept = {}
    for i in range(n - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if exercise == "american":
            values = np.maximum(values, sign * (prices(i) - k))
        if i <= 2:
            kept[i] = values.copy()
    s1, s2 = prices(1), prices(2)
    up_delta = (kept[2][2] - kept[2][1]) / (s2[2] - s2[1])
    down_delta = (kept[2][1] - kept[2][0]) / (s2[1] - s2[0])
    return Result({"price": float(kept[0][0]), "delta": float((kept[1][1] - kept[1][0]) / (s1[1] - s1[0])),
                   "gamma": float((up_delta - down_delta) / (0.5 * (s2[2] - s2[0]))),
                   "theta": float((kept[2][1] - kept[0][0]) / (2.0 * dt))})


def _trinomial(call, american, s, k, t, r, q, v, n):
    """Hull's trinomial tree: price and the Greeks of its first step's nodes."""
    dt = t / n
    u, disc = np.exp(v * np.sqrt(3.0 * dt)), np.exp(-r * dt)
    tilt = (r - q - 0.5 * v * v) * np.sqrt(dt / (12.0 * v * v))
    p_up, p_mid, p_down = 1.0 / 6.0 + tilt, 2.0 / 3.0, 1.0 / 6.0 - tilt
    if not 0.0 < p_down < 1.0 or not 0.0 < p_up < 1.0:
        raise ValueError(f"the trinomial tree's probabilities {p_up}, {p_down} lie outside (0, 1): take more steps")
    sign = 1.0 if call else -1.0

    def prices(i):
        return s * u ** np.arange(-i, i + 1)

    values = np.maximum(sign * (prices(n) - k), 0.0)
    for i in range(n - 1, -1, -1):
        values = disc * (p_up * values[2:] + p_mid * values[1:-1] + p_down * values[:-2])
        if american:
            values = np.maximum(values, sign * (prices(i) - k))
        if i == 1:
            first = values.copy()
    s1 = prices(1)
    up_delta, down_delta = (first[2] - first[1]) / (s1[2] - s1[1]), (first[1] - first[0]) / (s1[1] - s1[0])
    return Result({"price": float(values[0]), "delta": float((first[2] - first[0]) / (s1[2] - s1[0])),
                   "gamma": float((up_delta - down_delta) / (0.5 * (s1[2] - s1[0]))),
                   "theta": float((first[1] - values[0]) / dt)})
