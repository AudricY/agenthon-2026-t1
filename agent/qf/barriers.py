"""Barriers: Reiner-Rubinstein, double (Ikeda-Kunitomo), partial-time (Heynen-Kat), barrier digitals, one-touch, no-touch.

Every function takes continuously compounded annual rates (`rate` r, `dividend_yield` q), times in years and an annual
`vol`; a barrier is watched continuously unless the card says otherwise. A scalar result is a float, a function with
several results a dict with named keys. Monte Carlo of a discretely watched barrier: qf.barrier_mc_price.
"""
from __future__ import annotations

import numpy as np

from ._common import Result, choice, integer, log_ndtr, ndtr, out, positive
from .options import OPTIONS, _density, _inputs, bs_price
from .primitives import _guard

__all__ = ["barrier_digital_price", "barrier_price", "double_barrier_price", "one_touch_price",
           "partial_barrier_price"]

REBATE_TIMINGS = ("hit", "expiry")
HAUG = {   # Reiner and Rubinstein's options as sums of Haug's A to F: (K >= H, K < H); Haug (2007), 4.17.1
    ("down-and-in", "call"): ("C + E", "A - B + D + E"),
    ("up-and-in", "call"): ("A + E", "B - C + D + E"),
    ("down-and-in", "put"): ("B - C + D + E", "A + E"),
    ("up-and-in", "put"): ("A - B + D + E", "C + E"),
    ("down-and-out", "call"): ("A - C + F", "B - D + F"),
    ("up-and-out", "call"): ("F", "A - B + C - D + F"),
    ("down-and-out", "put"): ("A - B + C - D + F", "F"),
    ("up-and-out", "put"): ("B - D + F", "A - C + F"),
}
PAYOFFS = ("cash", "asset")
BARRIERS = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def _terms(spot, terms):
    """Value and delta of a sum of terms c S^m (H/S)^k N(a), each `(c, m, k, power, a, da)` with power = (H/S)^k and
    da = d a / d ln S: dV/dS = c S^(m-1) (H/S)^k [(m - k) N(a) + da n(a)]."""
    value = delta = 0.0
    for c, m, k, power, a, da in terms:
        cdf = ndtr(a)
        value = value + c * spot ** m * power * cdf
        delta = delta + c * spot ** (m - 1) * power * ((m - k) * cdf + da * _density(a))
    return value, delta


def _haug_sum(rule, parts):
    """`rule` ('A - B + D + E') over `parts`, each a (value, delta) pair: the signed sum of both."""
    value = delta = 0.0
    sign = 1.0
    for token in rule.split():
        if token in "+-":
            sign = 1.0 if token == "+" else -1.0
        else:
            value, delta = value + sign * parts[token][0], delta + sign * parts[token][1]
    return value, delta


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike', 'barrier',))
def barrier_price(option, barrier_type, spot, strike, barrier, rebate, rebate_timing, maturity, rate, dividend_yield,
                  vol):
    """Single-barrier option price and delta (Reiner-Rubinstein): continuous barrier, rebate.

    Haug's terms, b = r - q, mu = (b - vol^2/2)/vol^2, lambda = sqrt(mu^2 + 2r/vol^2), s = vol sqrt(T), phi = +1 call
    and -1 put, eta = +1 down and -1 up, R the rebate:
      A = phi S e^{-qT} N(phi x1) - phi K e^{-rT} N(phi x1 - phi s), x1 = ln(S/K)/s + (1 + mu) s; B: x2 = ln(S/H)/s
      + (1 + mu) s for x1
      C = phi S e^{-qT} (H/S)^{2(mu+1)} N(eta y1) - phi K e^{-rT} (H/S)^{2mu} N(eta y1 - eta s), y1 = ln(H^2/(S K))/s
      + (1 + mu) s; D: y2 = ln(H/S)/s + (1 + mu) s for y1
      E = R e^{-rT} [N(eta x2 - eta s) - (H/S)^{2mu} N(eta y2 - eta s)] (paid at expiry if H is never hit)
      F = R [(H/S)^{mu+lambda} N(eta z) + (H/S)^{mu-lambda} N(eta z - 2 eta lambda s)], z = ln(H/S)/s + lambda s (paid
      at the hit)
    For K >= H | K < H: down-and-in call C + E | A - B + D + E; up-and-in call A + E | B - C + D + E; down-and-in put
    B - C + D + E | A + E; up-and-in put A - B + D + E | C + E; down-and-out call A - C + F | B - D + F; up-and-out
    call F | A - B + C - D + F; down-and-out put A - B + C - D + F | F; up-and-out put B - D + F | A - C + F. An
    out's rebate at expiry is R e^{-rT} - E for F. With R = 0, in + out = the vanilla (qf.bs_price). delta is the
    closed-form dV/dS of the same sum.

    conventions:
    - barrier_type: 'down-and-out', 'down-and-in', 'up-and-out' or 'up-and-in'; H monitored continuously (every dt
      instead: move H away from the spot by e^{0.5826 vol sqrt(dt)}, Broadie-Glasserman-Kou)
    - rebate: cash >= 0; rebate_timing: 'hit' (an out's, at the knock-out) or 'expiry' (an in's; an out's at expiry)
    - a barrier crossed already: an in is the vanilla, an out its rebate
    returns: a dict: price; delta (dV/dS)
    source: Reiner and Rubinstein (1991), "Breaking down the barriers", Risk 4(8); Haug (2007), 4.17.1; Hull, "Barrier options"
    aliases: barrier option, knock-out, knock-in, knock-out option, knock-in option, down-and-out call, down-and-in call, up-and-out call, up-and-in call, down-and-out put, down-and-in put, up-and-out put, up-and-in put, Reiner-Rubinstein, [rebate]
    example:
        qf.barrier_price(option="call", barrier_type="down-and-out", spot=100.0, strike=100.0, barrier=90.0,
                         rebate=0.0, rebate_timing="hit", maturity=1.0, rate=0.03, dividend_yield=0.01,
                         vol=0.25)
    """
    choice("option", option, OPTIONS)
    choice("barrier_type", barrier_type, BARRIERS)
    choice("rebate_timing", rebate_timing, REBATE_TIMINGS)
    s, k, h = positive("spot", spot), positive("strike", strike), positive("barrier", barrier)
    t, v = positive("maturity", maturity), positive("vol", vol)
    r, q = np.asarray(rate, dtype=float), np.asarray(dividend_yield, dtype=float)
    cash = np.asarray(rebate, dtype=float)
    if np.any(~(cash >= 0)):
        raise ValueError("rebate must be at least 0")
    knock_in, up = barrier_type.endswith("in"), barrier_type.startswith("up")
    if knock_in and rebate_timing == "hit" and np.any(cash > 0):
        raise ValueError("an in's rebate is paid at expiry, owed only if the barrier is never hit: rebate_timing='expiry'")
    phi, eta = (1.0 if option == "call" else -1.0), (-1.0 if up else 1.0)
    alive = s < h if up else s > h
    sd, disc, carry = v * np.sqrt(t), np.exp(-r * t), np.exp(-q * t)
    mu = (r - q - 0.5 * v * v) / (v * v)
    lam_sq = mu * mu + 2.0 * r / (v * v)
    if not knock_in and rebate_timing == "hit" and np.any((cash > 0) & alive & (lam_sq < 0)):
        raise ValueError("a rebate at the hit needs mu^2 + 2 r / vol^2 >= 0, which a rate this negative breaks")
    lam = np.sqrt(np.maximum(lam_sq, 0.0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        hh = np.where(alive, h, s * (2.0 if up else 0.5))    # a stand-in barrier where knocked; its value is unused
        ratio, one = hh / s, np.ones(np.broadcast(s, hh).shape)
        x1 = np.log(s / k) / sd + (1.0 + mu) * sd
        x2 = np.log(s / hh) / sd + (1.0 + mu) * sd
        y1 = np.log(hh * hh / (s * k)) / sd + (1.0 + mu) * sd
        y2 = np.log(hh / s) / sd + (1.0 + mu) * sd
        z = np.log(hh / s) / sd + lam * sd
        p2m, p2m2 = ratio ** (2.0 * mu), ratio ** (2.0 * mu + 2.0)

        def direct(x):       # A at x1, B at x2
            return _terms(s, [(phi * carry, 1, 0.0, one, phi * x, phi / sd),
                              (-phi * k * disc, 0, 0.0, one, phi * (x - sd), phi / sd)])

        def image(y):        # C at y1, D at y2
            return _terms(s, [(phi * carry, 1, 2.0 * mu + 2.0, p2m2, eta * y, -eta / sd),
                              (-phi * k * disc, 0, 2.0 * mu, p2m, eta * (y - sd), -eta / sd)])

        parts = {"A": direct(x1), "B": direct(x2), "C": image(y1), "D": image(y2)}
        parts["E"] = _terms(s, [(cash * disc, 0, 0.0, one, eta * (x2 - sd), eta / sd),
                                (-cash * disc, 0, 2.0 * mu, p2m, eta * (y2 - sd), -eta / sd)])
        if rebate_timing == "hit":
            parts["F"] = _terms(s, [(cash, 0, mu + lam, ratio ** (mu + lam), eta * z, -eta / sd),
                                    (cash, 0, mu - lam, ratio ** (mu - lam), eta * (z - 2.0 * lam * sd), -eta / sd)])
        else:
            parts["F"] = (cash * disc - parts["E"][0], -parts["E"][1])
        high, low = (_haug_sum(rule, parts) for rule in HAUG[(barrier_type, option)])
        above = k >= hh
        price, delta = np.where(above, high[0], low[0]), np.where(above, high[1], low[1])
    knocked = parts["A"] if knock_in else (cash * (1.0 if rebate_timing == "hit" else disc), 0.0 * cash)
    return Result({"price": out(np.where(alive, price, knocked[0])), "delta": out(np.where(alive, delta, knocked[1]))})


def _vanilla_digital(payoff, option, s, k, t, r, q, v):
    """The digital's price with no barrier (qf.digital_price's, vol and maturity positive)."""
    sign = 1.0 if option == "call" else -1.0
    sd = v * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / sd
    if payoff == "asset":
        return s * np.exp(-q * t) * ndtr(sign * d1)
    return np.exp(-r * t) * ndtr(sign * (d1 - sd))


def _kept(s, k, h, t, nu, v, up):
    """P(S_T > k and the minimum stays above h) for a down barrier h < s, k >= h (`up` False); P(S_T < k and the
    maximum stays below h) for an up barrier h > s, k <= h (`up` True); ln S drifts at nu with volatility v. The
    reflection principle: N(e d(S/k)) - (h/S)^{2 nu / v^2} N(e d(h^2/(S k))), d(y) = (ln y + nu T)/(v sqrt T), e = +1
    down and -1 up."""
    sd = v * np.sqrt(t)
    sign = -1.0 if up else 1.0
    direct = ndtr(sign * (np.log(s / k) + nu * t) / sd)
    image = np.exp(2.0 * nu / (v * v) * np.log(h / s) + log_ndtr(sign * (np.log(h * h / (s * k)) + nu * t) / sd))
    return direct - image


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike', 'barrier',))
def barrier_digital_price(payoff, option, barrier_type, spot, strike, barrier, maturity, rate, dividend_yield, vol):
    """Price of a single-barrier digital, continuously monitored, paid at maturity:
    'cash' pays 1, 'asset' S_T, a call where S_T > K, a put where S_T < K; H the barrier.

    By the reflection principle, with ln S drifting at nu (nu = r - q - vol^2/2 for cash, priced e^{-rT} P; nu = r - q
    + vol^2/2 for asset, the share measure, priced S e^{-qT} P) and d(y) = (ln y + nu T)/(vol sqrt T):
      down, H < S:  P(S_T > k, min S > H) = N(d(S/k)) - (H/S)^{2 nu/vol^2} N(d(H^2/(S k))), k = max(K, H);
                    the put's P(S_T < K, min S > H) is that at k = H less that at k = max(K, H);
      up, H > S:    P(S_T < k, max S < H) = N(-d(S/k)) - (H/S)^{2 nu/vol^2} N(-d(H^2/(S k))), k = min(K, H);
                    the call's is that at k = H less that at k = min(K, H).
    A knock-in is the vanilla digital (digital_price) less its knock-out. A barrier already crossed today (S <= H down,
    S >= H up) has knocked: the out is worth 0 and the in the vanilla. No rebate; conventions as bs_price. One-touch
    and no-touch (no strike): qf.one_touch_price.
    Source: Reiner and Rubinstein (1991), "Unscrambling the binary code", Risk 4(9) (the binary barrier formulas);
    Rich (1994), "The mathematical foundations of barrier option-pricing theory"; Haug (2007), "Binary barrier options".
    aliases: barrier digital, digital barrier, knock-out digital, knock-in digital
    returns: the price today, a float or an array (a cash payoff per unit paid)
    example:
        qf.barrier_digital_price(payoff="cash", option="call", barrier_type="down-and-out",
                                 spot=100.0, strike=100.0, barrier=90.0, maturity=0.5, rate=0.03,
                                 dividend_yield=0.0, vol=0.2)
    """
    choice("payoff", payoff, PAYOFFS)
    choice("option", option, OPTIONS)
    choice("barrier_type", barrier_type, BARRIERS)
    s, k, t, r, q, v = _inputs(spot, strike, maturity, rate, dividend_yield, vol)
    h = positive("barrier", barrier)
    positive("vol", v)
    positive("maturity", t)
    up = barrier_type.startswith("up")
    nu = r - q + (0.5 if payoff == "asset" else -0.5) * v * v
    alive = s < h if up else s > h
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe_h = np.where(alive, h, s * (2.0 if up else 0.5))
        edge = np.minimum(k, safe_h) if up else np.maximum(k, safe_h)
        kept = _kept(s, edge, safe_h, t, nu, v, up)
        if option == ("call" if up else "put"):
            kept = _kept(s, safe_h, safe_h, t, nu, v, up) - kept
    knocked_out = np.where(alive, kept, 0.0)
    scale = s * np.exp(-q * t) if payoff == "asset" else np.exp(-r * t)
    vanilla = _vanilla_digital(payoff, option, s, k, t, r, q, v)
    price = scale * knocked_out
    return out(price if barrier_type.endswith("out") else vanilla - price)


KINDS = ("out", "in")


def _double(s, lower, upper, t, b, v, curve_low, curve_up, n_terms, low_end, high_end):
    """Ikeda and Kunitomo's sums: (share, cash), the probabilities that S_T lies in (low_end, high_end) and no barrier
    was touched, under the share measure and the pricing measure (b the carry r - q)."""
    sd = v * np.sqrt(t)
    share = cash = 0.0
    if not low_end < high_end:
        return share, cash
    for n in range(-n_terms, n_terms + 1):
        mu1 = 2.0 * (b - curve_low - n * (curve_up - curve_low)) / (v * v) + 1.0
        mu2 = 2.0 * n * (curve_up - curve_low) / (v * v)
        mu3 = 2.0 * (b - curve_low + n * (curve_up - curve_low)) / (v * v) + 1.0
        up_down, low_spot = n * np.log(upper / lower), np.log(lower / s)
        image = (n + 1) * np.log(lower) - n * np.log(upper) - np.log(s)      # ln(L^(n+1) / (U^n S))

        def d(x):
            return (np.log(s / x) + 2.0 * up_down + (b + 0.5 * v * v) * t) / sd

        def e(x):
            return (np.log(lower * lower / (x * s)) - 2.0 * up_down + (b + 0.5 * v * v) * t) / sd

        def band(f, shift, x0, x1):
            return ndtr(f(x0) - shift) - ndtr(f(x1) - shift)

        share += (np.exp(mu1 * up_down + mu2 * low_spot) * band(d, 0.0, low_end, high_end)
                  - np.exp(mu3 * image) * band(e, 0.0, low_end, high_end))
        cash += (np.exp((mu1 - 2.0) * up_down + mu2 * low_spot) * band(d, sd, low_end, high_end)
                 - np.exp((mu3 - 2.0) * image) * band(e, sd, low_end, high_end))
    return share, cash


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike', 'lower', 'upper',))
def double_barrier_price(option, kind, spot, strike, lower, upper, maturity, rate, dividend_yield, vol,
                         lower_curvature, upper_curvature, terms):
    """Double-barrier knock-out or knock-in (Ikeda-Kunitomo), and the survival odds.

    The barriers L e^{delta2 t} and U e^{delta1 t} are watched all the time; a knock-out dies when the price touches
    either. With b = r - q, sums over n = -N..N of image terms (Ikeda and Kunitomo; Haug):
    call = S e^{-qT} sum_n [(U^n/L^n)^{mu1} (L/S)^{mu2} (N(d1) - N(d2)) - (L^{n+1}/(U^n S))^{mu3} (N(d3) - N(d4))]
         - K e^{-rT} sum_n [the same with powers mu1 - 2, mu3 - 2 and each d less vol sqrt T],
    d1, d2 = [ln(S U^{2n} / (x L^{2n})) + (b + vol^2/2) T] / (vol sqrt T) at x = K and x = F = U e^{delta1 T},
    d3, d4 = [ln(L^{2n+2} / (x S U^{2n})) + (b + vol^2/2) T] / (vol sqrt T) there, mu1 = 2 [b - delta2 - n (delta1 -
    delta2)] / vol^2 + 1, mu2 = 2 n (delta1 - delta2) / vol^2, mu3 = 2 [b - delta2 + n (delta1 - delta2)] / vol^2 + 1;
    the put the same between E = L e^{delta2 T} and K, signs swapped (a strike outside the corridor is cut to it).
    A knock-in is the vanilla (qf.bs_price) less the knock-out. One barrier only: qf.barrier_price.

    conventions:
    - option: 'call' or 'put'; kind: 'out' (dies at a touch of either barrier) or 'in' (lives from a touch)
    - spot S, strike K; lower, upper: L < S < U today; lower_curvature delta2, upper_curvature delta1: 0 for flat barriers
    - terms: N, the series runs n = -N..N (an int of at least 0); increase until the price stabilizes
    - rate r and dividend_yield q continuously compounded; maturity T in years; vol annual
    returns: a dict: price; survival (the probability, under the pricing measure, that the price stays between the
      barriers to expiry: a double no-touch paying 1 is e^{-rT} survival)
    source: Ikeda and Kunitomo (1992), Mathematical Finance 2(4); Haug (2007), "Double-barrier options"
    aliases: double barrier, double knock-out, double knock-in, Ikeda-Kunitomo, double no-touch
    example:
        import qf
        qf.double_barrier_price(option="call", kind="out", spot=100.0, strike=100.0, lower=80.0, upper=130.0,
                                maturity=0.5, rate=0.05, dividend_yield=0.0, vol=0.2,
                                lower_curvature=0.0, upper_curvature=0.0, terms=5)  # result: price 4.56216501, survival 0.82635256
    """
    choice("option", option, OPTIONS)
    choice("kind", kind, KINDS)
    s, k, t, v = (float(positive(name, value)) for name, value in
                  (("spot", spot), ("strike", strike), ("maturity", maturity), ("vol", vol)))
    low, high = float(positive("lower", lower)), float(positive("upper", upper))
    r, q, curve_low, curve_up = float(rate), float(dividend_yield), float(lower_curvature), float(upper_curvature)
    n_terms = integer("terms", terms, 0)
    if not low < high:
        raise ValueError("lower must lie below upper")
    vanilla = float(bs_price(option, s, k, t, r, q, v))
    if not low < s < high:
        out_price = survival = 0.0
    else:
        floor, ceiling = low * np.exp(curve_low * t), high * np.exp(curve_up * t)
        if option == "call":
            share, cash = _double(s, low, high, t, r - q, v, curve_low, curve_up, n_terms, max(k, floor), ceiling)
            out_price = s * np.exp(-q * t) * share - k * np.exp(-r * t) * cash
        else:
            share, cash = _double(s, low, high, t, r - q, v, curve_low, curve_up, n_terms, floor, min(k, ceiling))
            out_price = k * np.exp(-r * t) * cash - s * np.exp(-q * t) * share
        survival = _double(s, low, high, t, r - q, v, curve_low, curve_up, n_terms, floor, ceiling)[1]
    out_price = max(float(out_price), 0.0)
    return Result({"price": out_price if kind == "out" else vanilla - out_price,
                   "survival": float(min(max(survival, 0.0), 1.0))})


TOUCHES = ("one", "no")
PAYS = ("hit", "expiry")
DIRECTIONS = ("down", "up")


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'barrier',))
def one_touch_price(touch, pay, payout, direction, spot, barrier, maturity, rate, dividend_yield, vol):
    """One-touch or no-touch (Reiner-Rubinstein): paid at the hit or at expiry.

    eta = +1 for a barrier below the spot ('down'), -1 above ('up'); b = r - q, mu = (b - vol^2/2)/vol^2, lambda =
    sqrt(mu^2 + 2r/vol^2), s = vol sqrt T, x = ln(S/H)/s + (1 + mu) s, y = ln(H/S)/s + (1 + mu) s, z = ln(H/S)/s +
    lambda s. No-touch at expiry: cash e^{-rT} [N(eta (x - s)) - (H/S)^{2 mu} N(eta (y - s))], asset S e^{-qT}
    [N(eta x) - (H/S)^{2 mu + 2} N(eta y)]; one-touch at expiry: e^{-rT} (or S e^{-qT}) less it; one-touch at the
    hit: (H/S)^{mu + lambda} N(eta z) + (H/S)^{mu - lambda} N(eta z - 2 eta lambda s) for cash, H times it for the
    asset (worth H at the hit). A barrier touched already: a one-touch pays now (at the hit) or at expiry, a no-touch
    nothing. A cash amount other than 1 scales the cash prices. One-touch + no-touch = e^{-rT} paid at expiry.

    conventions:
    - touch: 'one' (pays if the barrier is touched before expiry) or 'no' (pays if it never is)
    - pay: 'hit' (the moment of the touch; a one-touch only) or 'expiry'
    - payout: 'cash' (1 unit of cash, the domestic currency in FX) or 'asset' (1 unit of the asset, the foreign
      currency in FX)
    - direction: 'down' (barrier H below the spot) or 'up' (above); monitored continuously
    - rate r and dividend_yield q continuously compounded (FX: the domestic and the foreign rate); maturity T in
      years; vol annual
    returns: the price, a float
    source: Reiner and Rubinstein (1991), "Unscrambling the binary code", Risk 4(9); Haug (2007), "Binary barrier
      options"; Wystup (2006), FX Options and Structured Products, "One-touch"
    aliases: one-touch, no-touch, American digital, American binary
    example:
        import qf
        qf.one_touch_price(touch="one", pay="hit", payout="cash", direction="down", spot=105.0,
                           barrier=100.0, maturity=0.5, rate=0.05, dividend_yield=0.0,
                           vol=0.2)  # result: 0.69880601
    """
    choice("touch", touch, TOUCHES)
    choice("pay", pay, PAYS)
    choice("payout", payout, PAYOFFS)
    choice("direction", direction, DIRECTIONS)
    if touch == "no" and pay == "hit":
        raise ValueError("a no-touch pays at expiry: pay='expiry'")
    s, h, t, v = (float(positive(name, value)) for name, value in
                  (("spot", spot), ("barrier", barrier), ("maturity", maturity), ("vol", vol)))
    r, q = float(rate), float(dividend_yield)
    eta = 1.0 if direction == "down" else -1.0
    unit = s * np.exp(-q * t) if payout == "asset" else np.exp(-r * t)
    if (s <= h) if direction == "down" else (s >= h):
        if touch == "no":
            return 0.0
        return float((s if payout == "asset" else 1.0) if pay == "hit" else unit)
    sd = v * np.sqrt(t)
    mu = (r - q - 0.5 * v * v) / (v * v)
    ratio = np.log(h / s)
    if pay == "hit":
        lam_sq = mu * mu + 2.0 * r / (v * v)
        if lam_sq < 0:
            raise ValueError("paid at the hit needs mu^2 + 2 r / vol^2 >= 0, which a rate this negative breaks")
        lam = np.sqrt(lam_sq)
        z = ratio / sd + lam * sd
        cash = (np.exp((mu + lam) * ratio) * ndtr(eta * z)
                + np.exp((mu - lam) * ratio) * ndtr(eta * z - 2.0 * eta * lam * sd))
        return float(cash * (h if payout == "asset" else 1.0))
    x = -ratio / sd + (1.0 + mu) * sd
    y = ratio / sd + (1.0 + mu) * sd
    if payout == "asset":
        never = s * np.exp(-q * t) * (ndtr(eta * x) - np.exp((2.0 * mu + 2.0) * ratio) * ndtr(eta * y))
    else:
        never = np.exp(-r * t) * (ndtr(eta * (x - sd)) - np.exp(2.0 * mu * ratio) * ndtr(eta * (y - sd)))
    never = min(max(float(never), 0.0), float(unit))
    return never if touch == "no" else float(unit) - never


PARTIAL_CASES = ("A", "B1", "B2")
_NODES = np.polynomial.legendre.leggauss(16)
_SPAN = 13.0        # the standard normal integrated over [-13, 13]: its tails beyond are below 1e-38


def _panels(low, high, kinks):
    """Gauss-Legendre nodes and weights on [low, high], panels of at most one unit, split at the kinks."""
    if low >= high:
        return np.empty(0), np.empty(0)
    edges = sorted({low, high, *(x for x in kinks if low < x < high)})
    nodes, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        cuts = np.linspace(a, b, int(np.ceil(b - a)) + 1)
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            nodes.append(0.5 * (c1 - c0) * _NODES[0] + 0.5 * (c1 + c0))
            weights.append(0.5 * (c1 - c0) * _NODES[1])
    if not nodes:
        return np.empty(0), np.empty(0)
    return np.concatenate(nodes), np.concatenate(weights)


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity', 'monitor_time',),
        price=('spot', 'strike', 'barrier',))
def partial_barrier_price(option, barrier_type, case, spot, strike, barrier, monitor_time, maturity, rate,
                          dividend_yield, vol):
    """Partial-time barrier (Heynen-Kat): watched early, to t1, or late, from t1 (Haug's A, B1, B2).

    'A': the barrier is watched from today to t1 (t1 <= T); 'B1': from t1 to expiry, the option out where the price
    crosses it either way (barrier_type's up or down unused); 'B2': from t1 to expiry, out where the price is
    already beyond the barrier at t1 or touches it after. Heynen and Kat price these with bivariate normals; this
    takes the same integral over the price at t1 by Gauss-Legendre quadrature (to about 1e-12 of the price):
    'A' out = e^{-r t1} int over paths kept to t1 of the Black-Scholes price at t1 (life T - t1), the kept paths'
    density by reflection, n(x) - (H/S)^{2 nu / vol^2} n(x - 2 ln(H/S)), nu = r - q - vol^2/2; 'B1', 'B2' out =
    e^{-r t1} E[the single-barrier out price at t1 (qf.barrier_price, life T - t1)]. An in is the vanilla less the
    out. t1 = T in 'A' is the standard barrier option.

    conventions:
    - option: 'call' or 'put'; barrier_type: 'down-and-out', 'down-and-in', 'up-and-out' or 'up-and-in'
    - case: 'A' (watched early, from today to monitor_time), 'B1' or 'B2' (watched late, from monitor_time to
      expiry), as above; monitor_time: t1 in years from today, 0 < t1 < T (t1 <= T for 'A')
    - spot S, strike K; barrier: H, watched continuously in its window; no rebate
    - rate r and dividend_yield q continuously compounded; maturity T in years; vol annual
    returns: the price, a float
    source: Heynen and Kat (1994), "Partial barrier options", J. Financial Engineering 3; Haug (2007),
      "Partial-time single-asset barrier options"
    aliases: partial-time barrier, partial barrier, Heynen-Kat, Heynen and Kat
    example:
        import qf
        qf.partial_barrier_price(option="call", barrier_type="down-and-out", case="A", spot=100.0,
                                 strike=100.0, barrier=90.0, monitor_time=0.25, maturity=1.0,
                                 rate=0.05, dividend_yield=0.0, vol=0.25)  # result: 9.98070270
    """
    choice("option", option, OPTIONS)
    choice("barrier_type", barrier_type, BARRIERS)
    choice("case", case, PARTIAL_CASES)
    s, k, h, t1, t, v = (float(positive(name, value)) for name, value in
                         (("spot", spot), ("strike", strike), ("barrier", barrier), ("monitor_time", monitor_time),
                          ("maturity", maturity), ("vol", vol)))
    r, q = float(rate), float(dividend_yield)
    if not (t1 <= t if case == "A" else t1 < t):
        raise ValueError("monitor_time must lie in (0, maturity] for 'A' and in (0, maturity) for 'B1' and 'B2'")
    up, knock_in = barrier_type.startswith("up"), barrier_type.endswith("in")
    vanilla = float(bs_price(option, s, k, t, r, q, v))
    nu, sd = r - q - 0.5 * v * v, v * np.sqrt(t1)
    level = (np.log(h / s) - nu * t1) / sd                  # the barrier at t1, in standard normal units
    life = t - t1
    strike_level = (np.log(k / s) - nu * t1) / sd
    kinks = [strike_level]
    if life > 0:
        width = np.sqrt(life / t1)
        # Resolve the short remaining life's strike and barrier transition layers.
        kinks.extend(center + offset * width for center in (level, strike_level)
                     for offset in (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0))
    if case == "A":
        if (s >= h) if up else (s <= h):
            out_price = 0.0
        else:
            low, high = (-_SPAN, min(level, _SPAN)) if up else (max(level, -_SPAN), _SPAN)
            z, w = _panels(low, high, kinks)
            x = nu * t1 + sd * z
            kept = np.exp(-0.5 * z * z) - np.exp(2.0 * nu / (v * v) * np.log(h / s)
                                                  - 0.5 * (z - 2.0 * np.log(h / s) / sd) ** 2)
            later = np.asarray(bs_price(option, s * np.exp(x), k, life, r, q, v), dtype=float)
            out_price = np.exp(-r * t1) * float(np.sum(w * kept * later)) / np.sqrt(2.0 * np.pi)
    else:
        def inner(z, kind):
            return np.asarray(barrier_price(option, kind, s * np.exp(nu * t1 + sd * z), k, h, 0.0, "expiry", life,
                                            r, q, v)["price"], dtype=float)
        total = 0.0
        sides = [("down-and-out", max(level, -_SPAN), _SPAN), ("up-and-out", -_SPAN, min(level, _SPAN))]
        if case == "B2":
            sides = [sides[1]] if up else [sides[0]]
        for kind, low, high in sides:
            z, w = _panels(low, high, kinks)
            if z.size:
                total += float(np.sum(w * np.exp(-0.5 * z * z) * inner(z, kind)))
        out_price = np.exp(-r * t1) * total / np.sqrt(2.0 * np.pi)
    out_price = max(float(out_price), 0.0)
    return out_price if not knock_in else vanilla - out_price
