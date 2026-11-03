"""Two assets: exchange (Margrabe), spread (Kirk), rainbow best-of and worst-of (Stulz); quanto and compo.

Every function takes continuously compounded annual rates, times in years and annual vols; a correlation is that of
the log returns. A scalar result is a float, a function with several results a dict with named keys. Correlated
paths for a basket: qf.correlated_gbm_paths.
"""
from __future__ import annotations

import numpy as np

from ._common import Result, bivariate_normal_cdf, choice, ndtr, out, positive
from .options import OPTIONS, bs_price
from .primitives import _guard

__all__ = ["quanto_price", "rainbow_price", "spread_price"]

SPREAD_METHODS = ("kirk", "margrabe")


@_guard(rate=('rate', 'dividend_yield1', 'dividend_yield2',), vol=('vol1', 'vol2',), time=('maturity',),
        price=('spot1', 'spot2',))
def spread_price(method, option, spot1, spot2, strike, maturity, rate, dividend_yield1, dividend_yield2, vol1, vol2,
                 correlation):
    """Spread and exchange options on two assets: Kirk's approximation, Margrabe's exact exchange option (K = 0).

    The payoff max(+-(S1_T - S2_T - K), 0) for 'call' / 'put', the forwards F_i = S_i e^{(r - q_i)T}. 'margrabe'
    (K = 0 only; the right to receive S1 for S2): c = S1 e^{-q1 T} N(d1) - S2 e^{-q2 T} N(d2), d1 = (ln(S1/S2) + (q2 -
    q1 + vol^2/2) T)/(vol sqrt T), d2 = d1 - vol sqrt T, vol^2 = vol1^2 + vol2^2 - 2 rho vol1 vol2. 'kirk' treats F2
    + K as one lognormal: c = e^{-rT} [F1 N(d1) - (F2 + K) N(d2)], p = e^{-rT} [(F2 + K) N(-d2) - F1 N(-d1)], d1 =
    (ln(F1/(F2 + K)) + vol_K^2 T/2)/(vol_K sqrt T), vol_K^2 = vol1^2 - 2 rho vol1 vol2 w + vol2^2 w^2, w = F2/(F2 +
    K); at K = 0 it is Margrabe's exactly. Parity: c - p = e^{-rT} (F1 - F2 - K). On futures prices, pass them as the
    spots with dividend_yield1 = dividend_yield2 = rate. Monte Carlo paths of the pair: qf.correlated_gbm_paths.

    A tolling deal (the right to turn fuel into power each period) is a strip of spread calls, one per period, each
    on the power price less the heat rate times the fuel's (spot2 = heat rate x fuel price).

    conventions:
    - method: 'kirk' (any K with F2 + K > 0) or 'margrabe' (strike 0); option: 'call' or 'put'
    - spot1, spot2: the two assets' prices (the long and the short leg); strike: K, in price units (0 for an exchange)
    - rate, dividend_yield1, dividend_yield2: continuously compounded and annual; maturity: T in years
    - vol1, vol2: annual; correlation: rho of the two log returns, in [-1, 1]
    returns: a dict: price; vol (Margrabe's vol, or Kirk's vol_K)
    source: Margrabe (1978), J. Finance 33(1); Kirk (1995), "Correlation in the energy markets", in Managing Energy
      Price Risk; Haug (2007), 5.1 and 5.2; Hull, "Options to exchange one asset for another"
    aliases: spread option, Kirk approximation, Kirk's approximation, Kirk formula, Margrabe, Margrabe formula, exchange option, option to exchange, crack spread option, spark spread option, [Margrabe price], [Kirk price]
    example:
        qf.spread_price(method="kirk", option="call", spot1=110.0, spot2=100.0, strike=5.0, maturity=0.75,
                        rate=0.04, dividend_yield1=0.01, dividend_yield2=0.0, vol1=0.3, vol2=0.25,
                        correlation=0.6)
    """
    choice("method", method, SPREAD_METHODS)
    choice("option", option, OPTIONS)
    s1, s2 = positive("spot1", spot1), positive("spot2", spot2)
    k, t, r = (np.asarray(x, dtype=float) for x in (strike, maturity, rate))
    q1, q2 = np.asarray(dividend_yield1, dtype=float), np.asarray(dividend_yield2, dtype=float)
    v1, v2, rho = positive("vol1", vol1), positive("vol2", vol2), np.asarray(correlation, dtype=float)
    if np.any(~(t > 0)) or np.any(np.abs(rho) > 1):
        raise ValueError("maturity must be above 0 and correlation inside [-1, 1]")
    f1, f2 = s1 * np.exp((r - q1) * t), s2 * np.exp((r - q2) * t)
    if method == "margrabe":
        if np.any(k != 0):
            raise ValueError("'margrabe' prices the exchange option, strike 0; a spread with K != 0 is 'kirk'")
        weight = 1.0
    else:
        if np.any(~(f2 + k > 0)):
            raise ValueError("Kirk's approximation needs F2 + K > 0")
        weight = f2 / (f2 + k)
    vol = np.sqrt(np.maximum(v1 * v1 - 2.0 * rho * v1 * v2 * weight + (v2 * weight) ** 2, 0.0))
    sd = vol * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(f1 / (f2 + k)) + 0.5 * sd * sd) / sd
    sign = 1.0 if option == "call" else -1.0
    price = np.exp(-r * t) * sign * (f1 * ndtr(sign * d1) - (f2 + k) * ndtr(sign * (d1 - sd)))
    price = np.where(sd > 0, price, np.exp(-r * t) * np.maximum(sign * (f1 - f2 - k), 0.0))
    return Result({"price": out(price), "vol": out(vol)})


EXTREMES = ("maximum", "minimum")


def _exchange(s1, s2, t, r, q1, q2, sd):
    """Present values of S1 and S2 at T and of max(S1_T - S2_T, 0) (Margrabe), sd the ratio's vol sqrt T, and d."""
    pv1, pv2 = s1 * np.exp(-q1 * t), s2 * np.exp(-q2 * t)
    d = (np.log(pv1 / pv2) + 0.5 * sd * sd) / sd
    return pv1, pv2, pv1 * ndtr(d) - pv2 * ndtr(d - sd), d


@_guard(rate=('rate', 'dividend_yield1', 'dividend_yield2',), vol=('vol1', 'vol2',), time=('maturity',),
        price=('spot1', 'spot2',))
def rainbow_price(option, payoff_on, spot1, spot2, strike, maturity, rate, dividend_yield1, dividend_yield2, vol1, vol2,
                  correlation):
    """Rainbow option (Stulz): on the best-of or worst-of two, struck at K.

    With b_i = r - q_i, vol^2 = vol1^2 + vol2^2 - 2 rho vol1 vol2, rho1 = (vol1 - rho vol2)/vol, rho2 = (vol2 - rho
    vol1)/vol, d = [ln(S1/S2) + (b1 - b2 + vol^2/2) T] / (vol sqrt T), y_i = [ln(S_i/K) + (b_i + vol_i^2/2) T] / (vol_i
    sqrt T), M the bivariate normal CDF (qf.bivariate_normal_cdf):
    call on the minimum: S1 e^{-q1 T} M(y1, -d; -rho1) + S2 e^{-q2 T} M(y2, d - vol sqrt T; -rho2) - K e^{-rT} M(y1 -
      vol1 sqrt T, y2 - vol2 sqrt T; rho);
    call on the maximum: S1 e^{-q1 T} M(y1, d; rho1) + S2 e^{-q2 T} M(y2, -d + vol sqrt T; rho2) - K e^{-rT} [1 -
      M(-y1 + vol1 sqrt T, -y2 + vol2 sqrt T; rho)];
    a put by parity: K e^{-rT} - (the present value of the maximum or minimum, from Margrabe's exchange option) +
    the call. The call on the maximum plus the call on the minimum is the two vanilla calls; strike 0 gives the
    best-of (or worst-of) asset's present value.

    conventions:
    - option: 'call' or 'put'; payoff_on: 'maximum' (the better of the two at expiry, best-of) or 'minimum' (the worse,
      worst-of): call max(extreme - K, 0), put max(K - extreme, 0)
    - strike: K >= 0, in the assets' common currency; spot1, spot2 in it too
    - rate r, dividend_yield1 and dividend_yield2 continuously compounded; maturity T in years; vol1, vol2 annual
    - correlation: rho of the two assets' log returns, inside (-1, 1)
    returns: the price, a float
    source: Stulz (1982), J. Financial Economics 10(2); Johnson (1987); Haug (2007), "Options on the maximum or the
      minimum of two risky assets"; Hull, "Rainbow options"
    aliases: rainbow option, best-of option, worst-of option, Stulz, Stulz rainbow
    example:
        import qf
        qf.rainbow_price(option="call", payoff_on="maximum", spot1=100.0, spot2=105.0, strike=100.0,
                         maturity=0.5, rate=0.05, dividend_yield1=0.0, dividend_yield2=0.0,
                         vol1=0.2, vol2=0.3, correlation=0.5)  # result: 15.10965937
    """
    choice("option", option, OPTIONS)
    choice("payoff_on", payoff_on, EXTREMES)
    s1, s2, t = (float(positive(name, value)) for name, value in (("spot1", spot1), ("spot2", spot2),
                                                                   ("maturity", maturity)))
    v1, v2 = float(positive("vol1", vol1)), float(positive("vol2", vol2))
    k, r, q1, q2, rho = float(strike), float(rate), float(dividend_yield1), float(dividend_yield2), float(correlation)
    if k < 0 or not -1.0 < rho < 1.0:
        raise ValueError("strike must not be negative and correlation must lie inside (-1, 1)")
    vol = np.sqrt(v1 * v1 + v2 * v2 - 2.0 * rho * v1 * v2)
    sd = vol * np.sqrt(t)
    pv1, pv2, exchange, d = _exchange(s1, s2, t, r, q1, q2, sd)
    extreme = pv2 + exchange if payoff_on == "maximum" else pv1 - exchange      # the PV of the max or the min
    disc = np.exp(-r * t)
    if k == 0:
        call = extreme
    else:
        rho1, rho2 = (v1 - rho * v2) / vol, (v2 - rho * v1) / vol
        y1 = (np.log(s1 / k) + (r - q1 + 0.5 * v1 * v1) * t) / (v1 * np.sqrt(t))
        y2 = (np.log(s2 / k) + (r - q2 + 0.5 * v2 * v2) * t) / (v2 * np.sqrt(t))
        a1, a2 = y1 - v1 * np.sqrt(t), y2 - v2 * np.sqrt(t)
        m = bivariate_normal_cdf
        if payoff_on == "maximum":
            call = pv1 * m(y1, d, rho1) + pv2 * m(y2, -d + sd, rho2) - k * disc * (1.0 - m(-a1, -a2, rho))
        else:
            call = pv1 * m(y1, -d, -rho1) + pv2 * m(y2, d - sd, -rho2) - k * disc * m(a1, a2, rho)
    return float(call if option == "call" else k * disc - extreme + call)


QUANTO_PAYOFFS = ("quanto", "compo")
FX_QUOTES = ("domestic_per_foreign", "foreign_per_domestic")


@_guard(rate=('domestic_rate', 'foreign_rate', 'dividend_yield',), vol=('vol', 'fx_vol',), time=('maturity',),
        price=('spot',))
def quanto_price(option, payoff, spot, strike, maturity, domestic_rate, foreign_rate, dividend_yield, vol, fx_vol,
                 correlation, fx_quote, fx_rate, fixed_rate):
    """Quanto option price (quanto drift adjustment), or a compo struck in domestic currency.

    S a foreign asset in foreign currency, X the exchange rate (domestic per foreign), rho their log returns'
    correlation. 'quanto' pays fixed_rate x max(+-(S_T - K), 0) in domestic currency, K in foreign units: under the
    domestic measure S drifts at r_f - q - rho vol vol_X, F = S e^{(r_f - q - rho vol vol_X) T}, and a call is
    fixed_rate e^{-r_d T} [F N(d1) - K N(d2)], d1 = [ln(F/K) + vol^2 T/2] / (vol sqrt T), d2 = d1 - vol sqrt T.
    'compo' pays max(+-(S_T X_T - K), 0) in domestic currency, K in domestic units: Black-Scholes on S X with r_d,
    q and the vol sqrt(vol^2 + 2 rho vol vol_X + vol_X^2). rho = 0: Black-Scholes with the foreign carry.

    conventions:
    - option: 'call' or 'put'; payoff: 'quanto' (fixed conversion, fixed_rate given, fx_rate None) or 'compo'
      (converted at expiry's rate, fx_rate given, fixed_rate None)
    - fx_quote: the quote of fx_rate, fixed_rate and the rate in the correlation: 'domestic_per_foreign' (USD per EUR
      for a EUR asset paid in USD) or 'foreign_per_domestic' (inverted inside, the correlation's sign flipped)
    - correlation: rho of the asset's and the exchange rate's log returns; spot S and strike K as above
    - domestic_rate r_d (the payoff's currency), foreign_rate r_f (the asset's), dividend_yield q: continuously
      compounded; maturity T in years; vol, fx_vol: annual
    returns: a dict: price (in domestic currency); forward (F for 'quanto', S X e^{(r_d - q) T} for 'compo'); vol
      (the vol priced with)
    source: Reiner (1992), "Quanto mechanics", Risk 5; Haug (2007), "Currency-translated options"; Hull, "Quantos"
    aliases: quanto, quanto option, quanto adjustment, quanto drift adjustment, compo option, currency-translated option, [compo price]
    example:
        import qf
        qf.quanto_price(option="call", payoff="quanto", spot=100.0, strike=105.0, maturity=0.5,
                        domestic_rate=0.08, foreign_rate=0.05, dividend_yield=0.04, vol=0.2,
                        fx_vol=0.1, correlation=0.3, fx_quote="domestic_per_foreign",
                        fx_rate=None, fixed_rate=1.5)  # result: price 5.32803392
    """
    choice("option", option, OPTIONS)
    choice("payoff", payoff, QUANTO_PAYOFFS)
    choice("fx_quote", fx_quote, FX_QUOTES)
    s, k, t = (float(positive(name, value)) for name, value in (("spot", spot), ("strike", strike),
                                                                 ("maturity", maturity)))
    v, v_x = float(positive("vol", vol)), float(positive("fx_vol", fx_vol))
    r_d, r_f, q, rho = float(domestic_rate), float(foreign_rate), float(dividend_yield), float(correlation)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    used, unused = ("fixed_rate", "fx_rate") if payoff == "quanto" else ("fx_rate", "fixed_rate")
    given = {"fixed_rate": fixed_rate, "fx_rate": fx_rate}
    if given[used] is None or given[unused] is not None:
        raise ValueError(f"payoff {payoff!r} takes {used} and {unused}=None")
    rate_x = float(positive(used, given[used]))
    if fx_quote == "foreign_per_domestic":
        rate_x, rho = 1.0 / rate_x, -rho
    if payoff == "quanto":
        forward = s * np.exp((r_f - q - rho * v * v_x) * t)
        price = rate_x * float(bs_price(option, forward * np.exp(-r_d * t), k * np.exp(-r_d * t), t, 0.0, 0.0, v))
        return Result({"price": price, "forward": float(forward), "vol": v})
    combined = np.sqrt(v * v + 2.0 * rho * v * v_x + v_x * v_x)
    price = float(bs_price(option, s * rate_x, k, t, r_d, q, combined))
    return Result({"price": price, "forward": float(s * rate_x * np.exp((r_d - q) * t)), "vol": float(combined)})
