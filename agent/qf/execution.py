"""Execution: TWAP and VWAP schedules, Almgren-Chriss trajectories, shortfall and slippage, impact cost.

`import qf`. Quantities are shares (or contracts), lots their trading unit; an interval is one slice of the
schedule, in time order. A function with several results returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np

from ._common import Result, choice, integer, positive
from .primitives import _guard

__all__ = ["almgren_chriss", "impact_cost", "implementation_shortfall", "schedule"]

KINDS = ("twap", "vwap")
ROUNDINGS = ("largest_remainder", "floor_last", "none")
TIES = ("earliest", "latest")
TIMES = ("discrete", "continuous")
SIDES = ("buy", "sell")
DELAY_ON = ("order", "filled")
BPS = 1e4
SMALL_KAPPA_T = 1e-2            # below this kappa T, the continuous cost and variance integrals by their series
WHOLE = 1e-9                    # a quota within this of a whole number of lots is that number (float noise)


def schedule(kind, quantity, intervals, profile, lot, rounding, tie):
    """Execution schedules: TWAP, VWAP, integer lots.

    TWAP slices X / N in each of N intervals; VWAP slices X v_i / sum v, in proportion to each interval's share of
    the volume profile. In lots, each slice's quota is (X / lot) w_i: 'largest_remainder' (Hamilton's method) floors
    every quota and gives the lots left over, one each, to the intervals with the largest remainders; 'floor_last'
    floors every quota and adds the lots left over to the last interval; 'none' keeps the fractional slices. The slices
    sum to X, and under largest remainders each lies within one lot of its quota.

    conventions:
    - kind: 'twap' (profile None) or 'vwap' (profile: one volume per interval, any scale, normalised here)
    - quantity: X, the order in shares, a whole number of lots unless rounding is 'none'; intervals: N
    - lot: the trading unit in shares (100 for round lots, 1 for single shares)
    - rounding: 'largest_remainder', 'floor_last' or 'none'
    - tie: 'earliest' or 'latest', which of equal remainders take a leftover lot first
    returns: a dict: slices (shares per interval); quotas (X w_i, before rounding); weights (w_i); cumulative (the
      shares done by each interval's end)
    source: Kissell (2013), The Science of Algorithmic Trading, ch. 2; Balinski and Young (2001), Hamilton's method
    aliases: TWAP, VWAP, time-weighted average price, volume-weighted average price, execution schedule, slicing schedule, child orders, largest remainder, Hamilton's method, [round lots]
    example:
        qf.schedule(kind="vwap", quantity=10000, intervals=4, profile=[3.0, 2.0, 2.0, 2.0], lot=100,
                    rounding="largest_remainder",
                    tie="earliest")
    """
    choice("kind", kind, KINDS)
    choice("rounding", rounding, ROUNDINGS)
    choice("tie", tie, TIES)
    total, unit = float(positive("quantity", quantity)), float(positive("lot", lot))
    n = integer("intervals", intervals, 1)
    if kind == "twap":
        if profile is not None:
            raise ValueError("profile is for 'vwap'; pass None for 'twap'")
        weights = np.full(n, 1.0 / n)
    else:
        volume = np.asarray(profile, dtype=float).ravel() if profile is not None else np.array([])
        if volume.size != n or not np.all(np.isfinite(volume)) or np.any(volume < 0) or not volume.sum() > 0:
            raise ValueError("profile must hold one volume per interval, none negative, not all zero")
        weights = volume / volume.sum()
    quotas = total * weights
    if rounding == "none":
        slices = quotas.copy()
    else:
        lots = total / unit
        if abs(lots - round(lots)) > WHOLE * max(lots, 1.0):
            raise ValueError("quantity must be a whole number of lots when rounding")
        lots = int(round(lots))
        exact = lots * weights
        whole = np.floor(exact + WHOLE)
        left = lots - int(whole.sum())
        if rounding == "floor_last":
            whole[-1] += left
        else:
            remainders = np.round(exact - whole, 9)
            order = np.arange(n) if tie == "earliest" else -np.arange(n)
            whole[np.lexsort((order, -remainders))[:left]] += 1
        slices = whole * unit
    return Result({"slices": slices, "quotas": quotas, "weights": weights, "cumulative": np.cumsum(slices)})


def almgren_chriss(quantity, horizon, intervals, vol, risk_aversion, temporary_impact, permanent_impact, fixed_cost,
                   time):
    """Almgren-Chriss optimal liquidation: the trajectory, its expected cost and variance.

    Selling X over T in N intervals of tau = T/N, minimising E[C] + lambda Var[C] with the price an arithmetic random
    walk, permanent impact gamma v and temporary impact epsilon + eta v at the trading rate v = n/tau. 'discrete':
    eta~ = eta - gamma tau/2, cosh(kappa tau) = 1 + kappa~^2 tau^2/2, kappa~^2 = lambda sigma^2/eta~; holdings x_j =
    X sinh(kappa (T - t_j)) / sinh(kappa T); E[C] = gamma X^2/2 + epsilon sum |n_k| + (eta~/tau) sum n_k^2,
    Var[C] = sigma^2 tau sum_{k>=1} x_k^2. 'continuous': kappa^2 = lambda sigma^2/eta, x(t) as above read on the grid,
    E[C] = gamma X^2/2 + epsilon X + eta int xdot^2 dt, Var[C] = sigma^2 int x^2 dt. lambda = 0 is TWAP (x linear,
    Var = sigma^2 X^2 T/3 in continuous time); a larger lambda trades sooner.

    conventions:
    - quantity: X, shares to sell (a buy's schedule mirrors it); horizon: T; intervals: N; times in one unit
      (days, say), vol in that unit too: sigma, the price's standard deviation per sqrt(unit), in price units
    - risk_aversion: lambda, per unit of cost (1/$); temporary_impact: eta, price per (share per unit time);
      permanent_impact: gamma, price per share; fixed_cost: epsilon, price per share (half the spread plus fees)
    - time: 'discrete' (Almgren and Chriss's intervals) or 'continuous' (the limit, read on the grid)
    returns: a dict: times (N + 1 points from 0 to T); holdings (x at them, X to 0); trades (n_k, the N sales);
      kappa; expected_cost; variance (both in price units times shares, of the shortfall against X S_0); objective
      (expected_cost + lambda variance)
    source: Almgren and Chriss (2000), J. Risk 3(2); Cartea, Jaimungal and Penalva (2015), ch. 6
    aliases: Almgren-Chriss, Almgren and Chriss, optimal execution, optimal liquidation, efficient frontier of trading, implementation shortfall optimal trajectory, trading trajectory, risk-averse liquidation
    example:
        qf.almgren_chriss(quantity=1000000.0, horizon=5.0, intervals=5, vol=0.95, risk_aversion=1e-6,
                          temporary_impact=2.5e-6, permanent_impact=2.5e-7, fixed_cost=0.0625,
                          time="discrete")
    """
    choice("time", time, TIMES)
    x, t = float(positive("quantity", quantity)), float(positive("horizon", horizon))
    n = integer("intervals", intervals, 1)
    sigma, lam, eta, gamma, eps = (float(v) for v in (vol, risk_aversion, temporary_impact, permanent_impact,
                                                      fixed_cost))
    if sigma < 0 or lam < 0 or eta <= 0 or gamma < 0 or eps < 0:
        raise ValueError("vol, risk_aversion, permanent_impact and fixed_cost must not be negative; "
                         "temporary_impact must be positive")
    tau = t / n
    grid = np.linspace(0.0, t, n + 1)
    if time == "discrete":
        eta = eta - gamma * tau / 2
        if eta <= 0:
            raise ValueError("temporary_impact must exceed permanent_impact * tau / 2 (eta~ > 0)")
        z = lam * sigma * sigma * tau * tau / (2.0 * eta)
        kappa = float(np.log1p(z + np.sqrt(z * (z + 2.0))) / tau)       # arccosh(1 + z), stable for small z
    else:
        kappa = float(np.sqrt(lam * sigma * sigma / eta))
    a = kappa * t
    if a > 0:
        holdings = x * np.exp(-kappa * grid) * np.expm1(-2.0 * kappa * (t - grid)) / np.expm1(-2.0 * a)
    else:
        holdings = x * (1.0 - grid / t)
    holdings[-1] = 0.0
    trades = -np.diff(holdings)
    if time == "discrete":
        cost = 0.5 * gamma * x * x + eps * np.abs(trades).sum() + eta / tau * (trades @ trades)
        variance = sigma * sigma * tau * (holdings[1:] @ holdings[1:])
    else:
        if a < SMALL_KAPPA_T:
            speed = 1.0 + a ** 4 / 45.0                                  # (a^2 / sinh^2 a + a coth a) / 2
            level = 1.0 / 3.0 - 2.0 * a * a / 45.0 + 2.0 * a ** 4 / 315.0  # coth a / (2a) - 1 / (2 sinh^2 a)
        else:
            coth = 1.0 / np.tanh(a)
            speed, level = 0.5 * a * a * (coth * coth - 1.0) + 0.5 * a * coth, coth / (2.0 * a) - 0.5 * (coth * coth - 1.0)
        cost = 0.5 * gamma * x * x + eps * x + eta * x * x / t * speed
        variance = sigma * sigma * x * x * t * level
    return Result({"times": grid, "holdings": holdings, "trades": trades, "kappa": kappa,
                   "expected_cost": float(cost), "variance": float(variance),
                   "objective": float(cost + lam * variance)})


@_guard(price=('decision_price', 'arrival_price', 'fill_prices', 'close_price',))
def implementation_shortfall(side, order_quantity, decision_price, arrival_price, fill_prices, fill_quantities,
                             close_price, fees, delay_on, benchmarks):
    """Implementation shortfall (Perold): delay, execution and opportunity costs and fees; slippage.

    The paper portfolio trades Q at P_d; fills are q_j at p_j (F = sum q_j), unfilled shares valued at P_c.
    With s = +1 for a buy, -1 for a sell, the shortfall (a cost when positive) is s [sum q_j (p_j -
    P_d) + (Q - F)(P_c - P_d)] + fees = delay + execution + opportunity + fees: execution s sum q_j (p_j - P_a);
    'order': delay s Q (P_a - P_d), opportunity s (Q - F)(P_c - P_a) (Kissell); 'filled': delay s F (P_a - P_d),
    opportunity s (Q - F)(P_c - P_d) (Perold). Also in bps of Q P_d. Slippage to a benchmark B: s (avg fill - B) / B.

    A TCA summary: the parts in bps, the slippage to each benchmark (arrival, the interval's VWAP np.average(prices,
    weights=volumes), TWAP), the fill rate, the participation F / the market's volume over the order's life.

    conventions:
    - side: 'buy' or 'sell'; order_quantity: Q; fill_prices, fill_quantities: one price and size per fill
    - decision_price: P_d (the paper portfolio's); arrival_price: P_a, the mid when the order reached the market
      (P_d again where the task has no delay); close_price: the unfilled shares' price (the close, or at the cancel)
    - fees: total commissions and fees in currency (0 for none)
    - delay_on: 'order' or 'filled' (above): same total
    - benchmarks: a dict of name to price ({'arrival': ..., 'vwap': ...}), {} for none
    returns: a dict: total, delay, execution, opportunity, fees (currency, a cost positive); total_bps, delay_bps,
      execution_bps, opportunity_bps, fees_bps; paper_value (Q P_d); filled (F); average_price; fill_rate (F / Q);
      slippage_bps (a dict, one per benchmark)
    source: Perold (1988), J. Portfolio Management 14(3); Kissell, The Science of Algorithmic Trading (2013), ch. 3
    aliases: implementation shortfall, Perold, arrival price slippage, slippage, transaction cost analysis, TCA, [execution cost], opportunity cost, delay cost, VWAP slippage
    example:
        # total 330; total_bps 66.
        qf.implementation_shortfall(side="buy", order_quantity=1000, decision_price=50.0, arrival_price=50.1,
            fill_prices=[50.2, 50.3], fill_quantities=[400, 400], close_price=50.6, fees=10.0, delay_on="order",
            benchmarks={"arrival": 50.1, "vwap": 50.25})
    """
    choice("side", side, SIDES)
    choice("delay_on", delay_on, DELAY_ON)
    sign = 1.0 if side == "buy" else -1.0
    q, decision, arrival, close = (float(positive(name, value)) for name, value in (
        ("order_quantity", order_quantity), ("decision_price", decision_price), ("arrival_price", arrival_price),
        ("close_price", close_price)))
    prices, sizes = np.asarray(fill_prices, dtype=float).ravel(), np.asarray(fill_quantities, dtype=float).ravel()
    if prices.shape != sizes.shape or not (np.all(np.isfinite(prices)) and np.all(np.isfinite(sizes))):
        raise ValueError("fill_prices and fill_quantities must be finite, one price per fill")
    if np.any(prices <= 0) or np.any(sizes < 0) or sizes.sum() > q * (1.0 + WHOLE):
        raise ValueError("fills need positive prices, sizes not negative, and no more than order_quantity in all")
    cost = float(fees)
    if not (np.isfinite(cost) and cost >= 0):
        raise ValueError("fees must be a total in currency, 0 or more")
    if not isinstance(benchmarks, dict):
        raise ValueError("benchmarks must be a dict of name to price ({} for none)")
    filled = float(sizes.sum())
    average = float(sizes @ prices / filled) if filled > 0 else math.nan
    left = q - filled
    execution = sign * float(sizes @ (prices - arrival))
    if delay_on == "order":
        delay, opportunity = sign * q * (arrival - decision), sign * left * (close - arrival)
    else:
        delay, opportunity = sign * filled * (arrival - decision), sign * left * (close - decision)
    paper = q * decision
    parts = {"total": delay + execution + opportunity + cost, "delay": delay, "execution": execution,
             "opportunity": opportunity, "fees": cost}
    return Result({**parts, **{f"{name}_bps": value / paper * BPS for name, value in parts.items()},
                   "paper_value": paper, "filled": filled, "average_price": average, "fill_rate": filled / q,
                   "slippage_bps": {name: sign * (average - float(positive(name, price))) / float(price) * BPS
                                    for name, price in benchmarks.items()}})


def impact_cost(quantity, adv, daily_vol, spread, coefficient, exponent, price, participation):
    """Square-root impact cost Y sigma sqrt(Q / ADV) and half the spread; the days to liquidate.

    As a fraction of the price: cost = spread / 2 + Y sigma (Q / V)^delta, Q the order, V the average daily volume,
    sigma the daily volatility, Y the impact coefficient (of order 1) and delta = 1/2 the square-root law; the
    impact at Q = V is Y sigma and doubling Q multiplies it by sqrt 2. In currency: cost times Q times the price.
    Liquidating Q at a participation rate rho of the daily volume takes Q / (rho V) days.

    conventions:
    - quantity: Q, the order; adv: V, the average daily volume, in the same unit (shares, or notional with price 1)
    - daily_vol: sigma, the daily volatility of returns, a decimal (0.02 for 2%)
    - spread: the full quoted bid-ask spread as a fraction of the price (0.001 for 10 bps); half of it is charged
    - coefficient: Y (1.0 is the plain square-root law; the task's own value where it gives one)
    - exponent: delta, 0.5 the square-root law (Toth et al.; Grinold and Kahn); another power where the task names
      one (0.6, the 3/5 of Almgren et al.'s temporary impact)
    - price: the price, for the cost in currency (1.0 where quantity is notional)
    - participation: rho, the share of the daily volume traded, for the liquidation days, or None
    returns: a dict: spread_cost and impact (fractions of the price); total (their sum); total_bps; cost (currency,
      total times Q times the price); liquidation_days (Q / (rho V), nan without rho)
    source: Toth, Lemperiere, Deremble, de Lataillade, Kockelkoren and Bouchaud (2011), Physical Review X 1; Almgren,
      Thum, Hauptmann and Li (2005), "Direct estimation of equity market impact", Risk 18(7); Grinold and Kahn,
      Active Portfolio Management (2000), ch. 16
    aliases: square-root impact, square root law, half spread cost, days to liquidate, [transaction cost]
    example:
        # total_bps about 49.72136; cost about 9944.27191; liquidation_days 0.5.
        qf.impact_cost(quantity=50000, adv=1000000, daily_vol=0.02, spread=0.001, coefficient=1.0,
                       exponent=0.5, price=40.0,
                       participation=0.1)
    """
    q, volume, sigma = (float(positive(name, value)) for name, value in (("quantity", quantity), ("adv", adv),
                                                                         ("daily_vol", daily_vol)))
    full, y, delta, p = (float(value) for value in (spread, coefficient, exponent, price))
    if not (full >= 0 and y >= 0 and delta > 0 and p > 0):
        raise ValueError("spread and coefficient must not be negative, exponent and price must be positive")
    impact = y * sigma * (q / volume) ** delta
    total = full / 2.0 + impact
    days = math.nan if participation is None else q / (float(positive("participation", participation)) * volume)
    return Result({"spread_cost": full / 2.0, "impact": impact, "total": total, "total_bps": total * BPS,
                   "cost": total * q * p, "liquidation_days": days})
