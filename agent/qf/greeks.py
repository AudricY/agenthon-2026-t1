"""Greeks by bump and revalue or on simulated paths; hedging P&L.

`import qf`. Each estimator names the unit of every Greek it returns: the plain derivative per unit of the input
unless a scale is passed. A Monte Carlo Greek comes with its standard error; several results come back as a dict with
named keys.
"""
from __future__ import annotations

import numpy as np

from ._common import Result, choice, ndtr, positive
from .primitives import _guard

__all__ = ["bump_greeks", "delta_hedge_pnl", "likelihood_ratio_greeks", "pathwise_greeks"]

SCHEMES = ("central", "forward", "backward")
BUMP_FORMS = ("absolute", "relative")
OPTIONS = ("call", "put")
LR_OPTIONS = ("call", "put", "digital_call", "digital_put")
PER_UNIT = (1.0, 0.01)
POSITIONS = ("short", "long")
COST_RULES = ("proportional", "per_share")
COMPOUNDING = ("continuous", "simple")
SETTLEMENTS = ("cash", "physical")


def _terminal(spot, strike, maturity, rate, dividend_yield, vol, normals):
    """S_T = S e^{(r - q - vol^2/2) T + vol sqrt(T) Z} for each normal, and the checked inputs."""
    s, k = float(positive("spot", spot)), float(positive("strike", strike))
    t, v = float(positive("maturity", maturity)), float(positive("vol", vol))
    z = np.asarray(normals, dtype=float).ravel()
    if z.size < 2 or not np.all(np.isfinite(z)):
        raise ValueError("normals must hold at least two finite standard normal draws, one per path")
    r, q = float(rate), float(dividend_yield)
    return s, k, t, r, v, z, s * np.exp((r - q - 0.5 * v * v) * t + v * np.sqrt(t) * z)


def _estimates(samples):
    found = {}
    for name, x in samples.items():
        found[name], found[f"{name}_se"] = float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))
    return Result(found)



def bump_greeks(price, params, bumps, scheme):
    """Greeks of any pricer by bump and revalue: central or one-sided bumps, common random numbers.

    Each named x moves by its step h ('absolute') or h x ('relative'): 'central' gives (V(x + h) - V(x - h)) / 2h and
    (V(x + h) - 2 V(x) + V(x - h)) / h^2; 'forward' (V(x + h) - V(x)) / h and (V(x + 2h) - 2 V(x + h) + V(x)) / h^2;
    'backward' their mirror. Errors O(h^2) central, O(h) one-sided, plus the pricer's noise over h (h^2 for the second).
    A Monte Carlo pricer must reseed its generator inside every call (numpy.random.default_rng(seed)), so that every
    bump reuses the same draws: with fresh draws the differences are noise. Theta is minus the derivative in maturity.

    conventions:
    - price: a callable taking params' names as keywords and returning a number
    - params: a dict of every argument the pricer takes, at the base point
    - bumps: a dict of the names to bump, each (h, 'absolute' or 'relative'); a 'relative' x must not be 0
    - scheme: 'central', 'forward' or 'backward'
    returns: a dict: price, V at params; first and second, each a dict of the bumped names to dV/dx and d2V/dx2 per
      unit of x (a vega per 1.00 of vol: times 0.01 for a vol point)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 7.1; Hull, "The Greek letters"
    aliases: bump and revalue, bump-and-reprice, finite-difference Greeks, common random numbers, [Greeks]
    example:
        qf.bump_greeks(price=lambda spot, vol: qf.bs_price("call", spot, 100.0, 1.0, 0.03, 0.0, vol),
                       params={"spot": 100.0, "vol": 0.2},
                       bumps={"spot": (0.01, "relative"), "vol": (0.0001, "absolute")}, scheme="central")
    """
    choice("scheme", scheme, SCHEMES)
    if not callable(price):
        raise ValueError("price must be a callable of params' names as keywords")
    base = dict(params)
    if not bumps:
        raise ValueError("bumps must name at least one argument")
    value = float(price(**base))
    first, second = {}, {}
    for name, bump in dict(bumps).items():
        if name not in base:
            raise ValueError(f"bumps names {name!r}, which params does not hold")
        try:
            size, form = bump
        except (TypeError, ValueError) as error:
            raise ValueError(f"bumps[{name!r}] must be a pair (h, 'absolute' or 'relative')") from error
        choice(f"bumps[{name!r}]'s form", form, BUMP_FORMS)
        x = float(base[name])
        step = float(size) * (x if form == "relative" else 1.0)
        if not (np.isfinite(step) and step != 0.0):
            raise ValueError(f"bumps[{name!r}] gives a zero or infinite step")

        def at(shift, name=name, x=x):
            return float(price(**{**base, name: x + shift}))

        if scheme == "central":
            up, down = at(step), at(-step)
            first[name], second[name] = (up - down) / (2.0 * step), (up - 2.0 * value + down) / step ** 2
        elif scheme == "forward":
            up, twice = at(step), at(2.0 * step)
            first[name], second[name] = (up - value) / step, (twice - 2.0 * up + value) / step ** 2
        else:
            down, twice = at(-step), at(-2.0 * step)
            first[name], second[name] = (value - down) / step, (value - 2.0 * down + twice) / step ** 2
    return Result({"price": value, "first": first, "second": second})


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def pathwise_greeks(option, spot, strike, maturity, rate, dividend_yield, vol, normals, vega_per, rho_per):
    """Pathwise Monte Carlo Greeks of a European call or put under Black-Scholes.

    S_T = S e^{(r - q - vol^2/2) T + vol sqrt(T) Z}, one path per normal Z; each Greek the mean of the discounted
    payoff's derivative along the path (Glasserman 7.2). Call: delta e^{-rT} 1{S_T > K} S_T / S; vega e^{-rT}
    1{S_T > K} S_T (sqrt(T) Z - vol T); rho e^{-rT} 1{S_T > K} K T; the put's with 1{S_T < K} and the signs
    flipped. The payoff's kink leaves gamma no pathwise estimator: gamma here is the mixed pathwise-likelihood
    ratio e^{-rT} 1{S_T > K} (S_T / S^2) (Z / (vol sqrt(T)) - 1) (put negated). A digital's pathwise Greeks are
    zero almost surely: qf.likelihood_ratio_greeks.

    conventions:
    - option: 'call' or 'put'; spot, strike, maturity, rate, dividend_yield, vol as qf.bs_price
    - normals: the paths' standard normals, one per path (numpy.random.default_rng(seed).standard_normal(n))
    - vega_per: 1.0 for the plain derivative dV/dvol, per unit of vol; 0.01 only where the task asks the change for
      a one-point move in the vol; rho_per: 1.0 for dV/dr, 0.01 for a one-point (1%) move in the rate
    returns: a dict: price, delta, gamma, vega, rho (the sample means) and each one's standard error, price_se,
      delta_se, gamma_se, vega_se, rho_se (ddof 1)
    source: Glasserman (2004), Monte Carlo Methods in Financial Engineering, 7.2; Broadie and Glasserman (1996)
    aliases: pathwise Greeks, pathwise derivative, pathwise delta, pathwise vega, infinitesimal perturbation analysis, Monte Carlo Greeks, Monte Carlo delta
    example:
        qf.pathwise_greeks(option="call", spot=100.0, strike=105.0, maturity=0.5, rate=0.03,
            dividend_yield=0.01, vol=0.25, normals=[0.31, -1.2, 1.83, 0.4, -0.72, 1.1, -0.05, 0.66],
            vega_per=1.0, rho_per=1.0)
    """
    choice("option", option, OPTIONS)
    choice("vega_per", vega_per, PER_UNIT)
    choice("rho_per", rho_per, PER_UNIT)
    s, k, t, r, v, z, s_t = _terminal(spot, strike, maturity, rate, dividend_yield, vol, normals)
    sign = 1.0 if option == "call" else -1.0
    hit = np.exp(-r * t) * (sign * (s_t - k) > 0)
    return _estimates({"price": np.exp(-r * t) * np.maximum(sign * (s_t - k), 0.0),
                       "delta": sign * hit * s_t / s,
                       "gamma": sign * hit * s_t / (s * s) * (z / (v * np.sqrt(t)) - 1.0),
                       "vega": sign * hit * s_t * (np.sqrt(t) * z - v * t) * vega_per,
                       "rho": sign * hit * k * t * rho_per})


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def likelihood_ratio_greeks(option, spot, strike, maturity, rate, dividend_yield, vol, normals, vega_per, rho_per):
    """Likelihood-ratio Monte Carlo Greeks of a European or digital option under Black-Scholes.

    S_T = S e^{(r - q - vol^2/2) T + vol sqrt(T) Z}, one path per normal Z; each Greek the mean of the discounted
    payoff e^{-rT} f(S_T) times the score of S_T's lognormal density in that parameter (Glasserman 7.3): delta
    Z / (S vol sqrt(T)); gamma (Z^2 - 1 - Z vol sqrt(T)) / (S^2 vol^2 T); vega (Z^2 - 1) / vol - Z sqrt(T); rho
    Z sqrt(T) / vol - T (the discounting's -T in). No derivative of the payoff is taken, so a digital's Greeks
    come out where the pathwise ones are zero; the price is that of the variance: larger errors than pathwise on a
    call, growing as T or vol shrinks.

    conventions:
    - option: 'call', 'put', 'digital_call' or 'digital_put' (a digital pays 1 at T where S_T is above, or below,
      K); spot, strike, maturity, rate, dividend_yield, vol as qf.bs_price
    - normals: the paths' standard normals, one per path (numpy.random.default_rng(seed).standard_normal(n))
    - vega_per: 1.0 for the plain derivative dV/dvol, per unit of vol; 0.01 only where the task asks the change for
      a one-point move in the vol; rho_per: 1.0 for dV/dr, 0.01 for a one-point (1%) move in the rate
    returns: a dict: price, delta, gamma, vega, rho (the sample means) and each one's standard error, price_se,
      delta_se, gamma_se, vega_se, rho_se (ddof 1)
    source: Glasserman (2004), Monte Carlo Methods in Financial Engineering, 7.3; Broadie and Glasserman (1996)
    aliases: likelihood ratio method, likelihood ratio Greeks, score function method, LR Greeks, LRM, digital option Greeks Monte Carlo, Monte Carlo gamma
    example:
        qf.likelihood_ratio_greeks(option="digital_call", spot=100.0, strike=105.0, maturity=0.5, rate=0.03,
            dividend_yield=0.01, vol=0.25, normals=[0.31, -1.2, 1.83, 0.4, -0.72, 1.1, -0.05, 0.66],
            vega_per=1.0, rho_per=1.0)
    """
    choice("option", option, LR_OPTIONS)
    choice("vega_per", vega_per, PER_UNIT)
    choice("rho_per", rho_per, PER_UNIT)
    s, k, t, r, v, z, s_t = _terminal(spot, strike, maturity, rate, dividend_yield, vol, normals)
    above = s_t > k
    payoff = {"call": np.maximum(s_t - k, 0.0), "put": np.maximum(k - s_t, 0.0), "digital_call": above * 1.0,
              "digital_put": (s_t < k) * 1.0}[option]
    value, root = np.exp(-r * t) * payoff, np.sqrt(t)
    return _estimates({"price": value, "delta": value * z / (s * v * root),
                       "gamma": value * (z * z - 1.0 - z * v * root) / (s * s * v * v * t),
                       "vega": value * ((z * z - 1.0) / v - z * root) * vega_per,
                       "rho": value * (z * root / v - t) * rho_per})


@_guard(rate=('rate', 'financing_rate', 'dividend_yield',), vol=('hedge_vol',), time=('times',),
        price=('paths', 'strike',))
def delta_hedge_pnl(paths, times, option, strike, rate, financing_rate, dividend_yield, cash_dividends, hedge_vol,
                    premium, position, rebalance_every, cost, cost_rule, compounding, settlement):
    """Delta-hedging P&L, rebalanced discretely along each path, with transaction costs and financing.

    Hedge a short option with Black-Scholes delta shares (a long with -delta), on spot less PV of remaining cash
    dividends at rate. Cash earns financing_rate; dividends arrive before rebalancing and
    each trade pays its cost. At expiry settle the option and liquidate shares: pnl is final cash. For a short with
    no costs, drift r - q and financing_rate = rate, E[present_value] = premium - Black-Scholes; the hedge's spread
    falls as 1/sqrt(N) for N rebalancings.

    conventions:
    - paths: n_paths x len(times) prices including dividend drops; times: years, from the trade to expiry
    - option: 'call' or 'put'; strike: K; position: 'short' or 'long'
    - rate r (the delta's), financing_rate (the cash's and pnl's discount rate), dividend_yield q,
      hedge_vol (one, or one a date): annual
    - cash_dividends: None, or (time, amount) pairs; shares held receive each payment, financed from its payment date
    - premium: the option price at times[0]
    - rebalance_every: trade at every k-th date (1: each); interest and dividends at each date
    - cost: a rate or (buy, sell); cost_rule: 'proportional' (cost x |dn| S) or 'per_share' (cost x |dn|)
    - compounding: 'continuous' (cash e^{financing_rate h}, dividend n S (e^{q h} - 1)) or 'simple'
      (cash (1 + financing_rate h), n S q h)
    - settlement: 'cash' (all shares sold) or 'physical' (exercise delivers the share against K)
    returns: a dict: pnl (a path's final cash); present_value (pnl discounted to times[0]); mean and std (of
      present_value, ddof 1); costs and dividends (a path's raw payments, summed); rebalances (interior trades)
    source: Hull, "Delta hedging"; Derman and Kamal (1999), Risk
    aliases: discrete delta hedging, delta hedging P&L, cost of hedging
    example:
        qf.delta_hedge_pnl(paths=[[100.0, 104.0, 103.0], [100.0, 96.0, 92.0]], times=[0.0, 0.5, 1.0], option="call", strike=100.0,
                           rate=0.03, financing_rate=0.03, dividend_yield=0.0, cash_dividends=None, hedge_vol=0.2, premium=9.4, position="short",
                           rebalance_every=1, cost=0.0, cost_rule="proportional", compounding="continuous", settlement="cash")
        # result: {'pnl': [6.462529, 3.854105], 'mean': 5.005866}
    """
    choice("option", option, OPTIONS)
    choice("position", position, POSITIONS)
    choice("cost_rule", cost_rule, COST_RULES)
    choice("compounding", compounding, COMPOUNDING)
    choice("settlement", settlement, SETTLEMENTS)
    s = np.asarray(paths, dtype=float)
    t = np.asarray(times, dtype=float).ravel()
    if s.ndim != 2 or s.shape[1] != t.size or t.size < 2 or np.any(np.diff(t) <= 0) or np.any(~(s > 0)):
        raise ValueError("paths must be n_paths x len(times) positive prices, times increasing, at least two")
    k, r, financing, q, paid = (float(positive("strike", strike)), float(rate), float(financing_rate),
                               float(dividend_yield), float(premium))
    vol = np.broadcast_to(positive("hedge_vol", hedge_vol), t.shape) if np.size(hedge_vol) in (1, t.size) else None
    if vol is None:
        raise ValueError("hedge_vol must be one vol, or one a date")
    buy, sell = (float(cost), float(cost)) if np.size(cost) == 1 else tuple(float(c) for c in np.ravel(cost))
    if not (buy >= 0 and sell >= 0):
        raise ValueError("cost must be a non-negative rate, or a (buy, sell) pair of them")
    every = int(rebalance_every) if isinstance(rebalance_every, (int, np.integer)) else 0
    if isinstance(rebalance_every, (bool, np.bool_)) or every < 1:
        raise ValueError("rebalance_every must be an int of at least 1")
    paid_out = np.zeros((0, 2)) if cash_dividends is None else np.asarray(cash_dividends, dtype=float).reshape(-1, 2)
    call, sign, expiry = option == "call", 1.0 if position == "short" else -1.0, t[-1]
    continuous = compounding == "continuous"

    def delta(j):
        tau = expiry - t[j]
        ahead = paid_out[(paid_out[:, 0] > t[j]) & (paid_out[:, 0] <= expiry)]
        spot = s[:, j] - np.sum(ahead[:, 1] * np.exp(-r * (ahead[:, 0] - t[j])))
        if np.any(spot <= 0):
            raise ValueError("the spot less the dividends to come must stay positive")
        d1 = (np.log(spot / k) + (r - q + 0.5 * vol[j] ** 2) * tau) / (vol[j] * np.sqrt(tau))
        return np.exp(-q * tau) * (ndtr(d1) - (0.0 if call else 1.0))

    def fee(shares, price):
        rate_paid = np.where(shares > 0, buy, sell)
        return rate_paid * np.abs(shares) * (price if cost_rule == "proportional" else 1.0)

    held = sign * delta(0)
    costs = fee(held, s[:, 0])
    cash, income, trades = sign * paid - held * s[:, 0] - costs, np.zeros(s.shape[0]), 0
    for j in range(1, t.size):
        h = t[j] - t[j - 1]
        cash = cash * (np.exp(financing * h) if continuous else 1.0 + financing * h)
        flow = held * (s[:, j] * np.expm1(q * h) if continuous else s[:, j - 1] * q * h)
        due = paid_out[(paid_out[:, 0] > t[j - 1]) & (paid_out[:, 0] <= t[j])]
        remaining = t[j] - due[:, 0]
        growth = np.exp(financing * remaining) if continuous else 1.0 + financing * remaining
        cash = cash + flow + held * np.sum(due[:, 1] * growth)
        income = income + flow + held * due[:, 1].sum()
        if j < t.size - 1 and j % every == 0:
            trade = sign * delta(j) - held
            charge = fee(trade, s[:, j])
            cash, costs, held, trades = cash - trade * s[:, j] - charge, costs + charge, held + trade, trades + 1
    end = s[:, -1]
    payoff = np.maximum((end - k) if call else (k - end), 0.0)
    if settlement == "cash":
        charge = fee(-held, end)
        cash = cash - sign * payoff + held * end - charge
    else:
        exercised = payoff > 0
        moved = np.where(exercised, sign * (-1.0 if call else 1.0), 0.0)      # the share the option moves to us
        left = held + moved
        charge = fee(-left, end)
        cash = cash + np.where(exercised, sign * (k if call else -k), 0.0) + left * end - charge
    costs = costs + charge
    grown = np.exp(financing * (expiry - t[0])) if continuous else np.prod(1.0 + financing * np.diff(t))
    value = cash / grown
    return Result({"pnl": cash, "present_value": value, "mean": float(value.mean()),
                   "std": float(value.std(ddof=1)) if value.size > 1 else float("nan"), "costs": costs,
                   "dividends": income, "rebalances": trades})
