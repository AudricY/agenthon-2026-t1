"""Microstructure: Lee-Ready, effective and Roll spreads, price-time priority, book imbalance and OFI, Hasbrouck, Hawkes.

`import qf`. Times are numbers in one unit throughout (seconds since midnight, say), each series in time
order; prices in the instrument's units. A function with several results returns a dict with named keys.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import optimize

from ._common import Result, choice, integer, positive
from .primitives import _guard, _stream

__all__ = ["book_imbalance", "hawkes_fit", "hawkes_simulate", "information_share", "match_orders",
           "order_flow_imbalance", "roll_spread", "spread_measures"]

SIGN_RULES = ("lee_ready", "tick", "quote", "given")
QUOTE_MATCHES = ("at_or_before", "before")
SCALES = ("price", "relative", "bps")
WEIGHTINGS = ("equal", "size", "dollar")
SELF_TRADES = ("allow", "cancel_resting", "cancel_incoming")
ROLL_CHANGES = ("price", "log")
ROLL_POSITIVE = ("zero", "nan")
MOST_VENUES = 6                 # the information share's bounds try every ordering: 720 of them at six venues
EVENT_FIELDS = {"limit": {"type", "id", "side", "price", "quantity"}, "market": {"type", "id", "side", "quantity"},
                "cancel": {"type", "id"}}
ON_TICK = 1e-9                  # a price within this many ticks of a multiple of the tick is on it
SAME_PRICE = 1e-9               # relative: two prices this close are one (a price at the midpoint, a zero tick)


def _series(name, values, size=None):
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0 or not np.all(np.isfinite(array)) or (size is not None and array.size != size):
        raise ValueError(f"{name} must hold {'one finite number per trade' if size else 'finite numbers'}")
    return array


def _tick_signs(prices):
    """+1 on an uptick, -1 on a downtick, a zero tick carrying the last nonzero one; 0 before the first change."""
    moves = np.diff(prices, prepend=prices[0])
    change = np.where(np.abs(moves) <= SAME_PRICE * prices, 0.0, np.sign(moves))
    filled = np.zeros_like(change)
    last = 0.0
    for i, move in enumerate(change):
        last = move if move != 0 else last
        filled[i] = last
    return filled


def _prevailing(quote_times, at, match):
    """The index of the quote standing at each time `at` (the last at or before it, or strictly before), -1 if none."""
    return np.searchsorted(quote_times, at, side="right" if match == "at_or_before" else "left") - 1


@_guard(price=('trade_prices', 'bids', 'asks',))
def spread_measures(trade_times, trade_prices, trade_sizes, quote_times, bids, asks, sign_rule, signs, quote_lag,
                    quote_match, horizon, scale, weighting):
    """Trade signing (Lee-Ready, tick) and the quoted, effective and realized spreads.

    Each trade at t meets the quote standing at t - quote_lag, midpoint M = (bid + ask) / 2. Its sign D: the quote
    rule +1 above M, -1 below; the tick test +1 on an uptick, -1 on a downtick, a zero tick taking the last change's;
    Lee-Ready the quote rule, the tick test at M. Quoted = ask - bid; effective = 2 D (P - M); realized = 2 D (P - M')
    with M' the midpoint standing at t + horizon; impact = effective - realized = 2 D (M' - M). The means are over
    trades with a quote and a sign, weighted. Amihud's illiquidity is the mean of |r| / dollar volume by day; Kyle's
    lambda the slope of price changes on signed volume (qf.ols).

    Bars: time bars group trades by clock interval, volume bars by a fixed volume (np.cumsum(sizes) // bucket).

    conventions:
    - sign_rule: 'lee_ready', 'tick', 'quote' (at M unsigned) or 'given' (signs: +1 buy, -1 sell, one per trade;
      None for the other rules)
    - quote_lag: taken from a trade's time before the lookup (Lee and Ready's 5 seconds; 0 for none)
    - quote_match: 'at_or_before' (a quote stamped at the trade's time stands) or 'before' (strictly earlier)
    - horizon: the realized spread's delay, in the times' unit; the last quote stands after the data
    - scale: 'price', 'relative' (over the trade's M) or 'bps' (relative times 10,000)
    - weighting: 'equal', 'size' or 'dollar' (size times price) in the means
    returns: a dict: sign, midpoint, quoted, effective, realized, impact (one per trade, nan without a quote);
      quoted_mean, effective_mean, realized_mean, impact_mean; quoted_time_weighted (each quote by how long it stood)
    source: Lee and Ready (1991), J. Finance 46(2); Huang and Stoll (1996), JFE 41; Hasbrouck (2007), ch. 3
    aliases: Lee-Ready, Lee and Ready, tick test, tick rule, quote rule, trade classification, trade signing, effective spread, realized spread, quoted spread, price impact, Amihud illiquidity, Kyle's lambda, [bid-ask spread]
    example:
        qf.spread_measures(trade_times=[1.0, 4.0, 7.0], trade_prices=[100.02, 100.00, 99.98],
            trade_sizes=[100, 200, 100], quote_times=[0.0, 5.0], bids=[99.98, 99.96], asks=[100.02, 100.00], sign_rule="lee_ready",
            signs=None, quote_lag=0.0, quote_match="at_or_before", horizon=5.0, scale="price", weighting="size")
    """
    choice("sign_rule", sign_rule, SIGN_RULES)
    choice("quote_match", quote_match, QUOTE_MATCHES)
    choice("scale", scale, SCALES)
    choice("weighting", weighting, WEIGHTINGS)
    times = _series("trade_times", trade_times)
    prices, sizes = _series("trade_prices", trade_prices, times.size), _series("trade_sizes", trade_sizes, times.size)
    stamps = _series("quote_times", quote_times)
    bid, ask = _series("bids", bids, stamps.size), _series("asks", asks, stamps.size)
    if np.any(np.diff(times) < 0) or np.any(np.diff(stamps) < 0):
        raise ValueError("trade_times and quote_times must each be in time order")
    if np.any(bid > ask) or np.any(prices <= 0) or np.any(sizes < 0):
        raise ValueError("every bid must not exceed its ask, prices must be positive and sizes not negative")
    lag, delay = float(quote_lag), float(horizon)
    if not (lag >= 0 and delay >= 0):
        raise ValueError("quote_lag and horizon must not be negative")
    if (signs is None) != (sign_rule != "given"):
        raise ValueError("signs: one per trade with sign_rule='given', None with the other rules")
    mid = (bid + ask) / 2.0
    now, later = _prevailing(stamps, times - lag, quote_match), _prevailing(stamps, times + delay, quote_match)
    mid_now = np.where(now >= 0, mid[now], np.nan)
    if sign_rule == "given":
        sign = _series("signs", signs, times.size)
        if not np.all(np.isin(sign, (-1.0, 0.0, 1.0))):
            raise ValueError("signs must be +1 (buy), -1 (sell) or 0 (unsigned)")
    elif sign_rule == "tick":
        sign = _tick_signs(prices)
    else:
        gap = np.nan_to_num(prices - mid_now)
        sign = np.where(np.isnan(mid_now) | (np.abs(gap) <= SAME_PRICE * prices), 0.0, np.sign(gap))
        if sign_rule == "lee_ready":
            sign = np.where((sign == 0) & ~np.isnan(mid_now), _tick_signs(prices), sign)
    base = {"price": np.ones_like(prices), "relative": mid_now, "bps": mid_now / 1e4}[scale]
    midpoint_later = np.where((now >= 0) & (later >= 0), mid[np.maximum(later, 0)], np.nan)
    quoted = np.where(now >= 0, (ask - bid)[np.maximum(now, 0)], np.nan) / base
    effective = 2.0 * sign * (prices - mid_now) / base
    realized = 2.0 * sign * (prices - midpoint_later) / base
    weight = {"equal": np.ones_like(prices), "size": sizes, "dollar": sizes * prices}[weighting]

    def mean(values, signed):
        keep = ~np.isnan(values) & ((sign != 0) if signed else True)
        return float(np.sum(weight[keep] * values[keep]) / np.sum(weight[keep])) if np.sum(weight[keep]) > 0 else np.nan

    lives = np.diff(stamps)
    spread = (ask - bid) / {"price": np.ones_like(mid), "relative": mid, "bps": mid / 1e4}[scale]
    return Result({"sign": sign, "midpoint": mid_now, "quoted": quoted, "effective": effective, "realized": realized,
                   "impact": effective - realized, "quoted_mean": mean(quoted, False),
                   "effective_mean": mean(effective, True), "realized_mean": mean(realized, True),
                   "impact_mean": mean(effective - realized, True),
                   "quoted_time_weighted": float(spread[:-1] @ lives / lives.sum()) if lives.sum() > 0 else np.nan})


def _event(index, event, tick):
    kind = event.get("type") if isinstance(event, dict) else None
    if kind not in EVENT_FIELDS or not EVENT_FIELDS[kind] <= set(event) <= EVENT_FIELDS[kind] | {"owner"}:
        raise ValueError(f"events[{index}] must be a dict of " + "; ".join(
            f"'{name}': {', '.join(sorted(fields))}" for name, fields in EVENT_FIELDS.items()) + " (and 'owner')")
    if kind != "cancel":
        choice(f"events[{index}]['side']", event["side"], ("buy", "sell"))
        if not float(event["quantity"]) > 0:
            raise ValueError(f"events[{index}]'s quantity must be positive")
    if kind == "limit" and tick is not None:
        units = float(event["price"]) / tick
        if abs(units - round(units)) > ON_TICK:
            raise ValueError(f"events[{index}]'s price {event['price']} is not a multiple of the tick {tick}")
    return kind


def match_orders(events, tick_size, self_trade):
    """Price-time priority order matching: a limit order book replayed event by event.

    An incoming order meets the opposite side's best price first and, within a price, the earliest order first
    (FIFO); each trade is at the resting order's price for the smaller of the two quantities, so one order may sweep
    several levels and fill partly. A limit order trades while the best opposite price is at or better than its
    limit, then rests for what is left; a market order trades until filled or the book's side is empty, the rest
    cancelled; a cancel removes what is left of a resting order (an unknown or filled id is listed as rejected). With
    owners named, a trade between one owner's orders is 'allow'ed, or 'cancel_resting' cancels the resting order and
    goes on, or 'cancel_incoming' cancels what is left of the incoming one. After every event the best bid is below
    the best ask, and each trade's quantity is the buyer's and the seller's both.

    conventions:
    - events: in time order, dicts: {'type': 'limit', 'id', 'side': 'buy' or 'sell', 'price', 'quantity'},
      {'type': 'market', 'id', 'side', 'quantity'} or {'type': 'cancel', 'id'}, each with 'owner' or without
    - tick_size: limit prices must be its multiples, or None for any price
    - self_trade: 'allow', 'cancel_resting' or 'cancel_incoming'
    returns: a dict: trades (dicts of buy_id, sell_id, price, quantity, aggressor 'buy' or 'sell', event, the
      index); bids and asks (the resting orders in priority, (id, price, quantity) each); best_bid and best_ask
      (after each event, numpy.nan when a side is empty); rejected (the cancels' ids that found nothing)
    source: Gould, Porter, Williams, McDonald, Fenn and Howison (2013), "Limit order books", Quantitative Finance 13(11)
    aliases: order matching, matching engine, limit order book, price-time priority, FIFO matching, order book simulation, continuous double auction
    example:
        qf.match_orders(events=[{"type": "limit", "id": 1, "side": "sell", "price": 100.1, "quantity": 200},
            {"type": "limit", "id": 2, "side": "sell", "price": 100.2, "quantity": 300}, {"type": "market", "id": 3, "side": "buy", "quantity": 350}],
            tick_size=0.1, self_trade="allow")
    """
    choice("self_trade", self_trade, SELF_TRADES)
    tick = None if tick_size is None else float(positive("tick_size", tick_size))
    book = {"buy": [], "sell": []}          # resting [price, sequence, id, quantity, owner], in priority order
    trades, rejected, best_bid, best_ask = [], [], [], []
    for index, event in enumerate(events):
        kind = _event(index, event, tick)
        if kind == "cancel":
            for side in ("buy", "sell"):
                found = [order for order in book[side] if order[2] == event["id"]]
                if found:
                    book[side].remove(found[0])
                    break
            else:
                rejected.append(event["id"])
        else:
            side, left, owner = event["side"], float(event["quantity"]), event.get("owner")
            other = book["sell" if side == "buy" else "buy"]
            limit = float(event["price"]) if kind == "limit" else None
            while left > 0 and other:
                resting = other[0]
                if limit is not None and (resting[0] > limit if side == "buy" else resting[0] < limit):
                    break
                if owner is not None and resting[4] == owner and self_trade != "allow":
                    if self_trade == "cancel_resting":
                        other.pop(0)
                        continue
                    left = 0.0
                    break
                size = min(left, resting[3])
                buyer, seller = (event["id"], resting[2]) if side == "buy" else (resting[2], event["id"])
                trades.append({"buy_id": buyer, "sell_id": seller, "price": resting[0], "quantity": size,
                               "aggressor": side, "event": index})
                left -= size
                resting[3] -= size
                if resting[3] <= 0:
                    other.pop(0)
            if kind == "limit" and left > 0:
                own = book[side]
                own.append([limit, index, event["id"], left, owner])
                own.sort(key=lambda order: ((-order[0] if side == "buy" else order[0]), order[1]))
        best_bid.append(book["buy"][0][0] if book["buy"] else np.nan)
        best_ask.append(book["sell"][0][0] if book["sell"] else np.nan)
    return Result({"trades": trades, "bids": [(o[2], o[0], o[3]) for o in book["buy"]],
                   "asks": [(o[2], o[0], o[3]) for o in book["sell"]], "best_bid": np.array(best_bid),
                   "best_ask": np.array(best_ask), "rejected": rejected})


# ------------------------------------------------------------------------------------------------ spread and book


@_guard(price=('prices',))
def roll_spread(prices, changes, ddof, positive_cov):
    """Roll's bid-ask spread estimate from serial covariance: s = 2 sqrt(-Cov(dP_t, dP_{t-1})).

    Trades bounce between the bid and the ask: P_t = m_t + (s/2) q_t, q_t = +1 at the ask and -1 at the bid with
    equal odds, independent of the efficient price m_t's random-walk steps; then Cov(dP_t, dP_{t-1}) = -s^2/4, so
    s = 2 sqrt(-c), c the sample covariance of the pairs (dP_t, dP_{t-1}) (np.cov of the two shifted series). A
    covariance at or above zero has no Roll spread. On log prices s is a fraction of the price (times 10,000 in bps).
    A rolling estimate: rolling_apply(lambda p: roll_spread(p, ...)['spread'], [prices], window=w, kind='rolling').

    Glosten-Milgrom's quotes, the spread from adverse selection: with the value V_H or V_L (probability theta of V_H)
    and a share mu of informed traders, ask = E[V | a buy], bid = E[V | a sell] by Bayes' rule; with uninformed
    traders buying or selling with equal odds, P(V_H | buy) = theta (1 + mu) / (theta (1 + mu) + (1 - theta)(1 - mu)).

    conventions:
    - prices: trade or closing prices, oldest first
    - changes: 'price' (dP in price units, the spread in price units) or 'log' (d ln P, the spread a fraction of the
      price)
    - ddof: 1 (the sample covariance: np.cov's and pandas' default) or 0
    - positive_cov: what a covariance at or above zero gives, 'zero' (spread 0) or 'nan'
    returns: a dict: spread (s); covariance (c); n_pairs
    source: Roll (1984), "A simple implicit measure of the effective bid-ask spread", J. Finance 39(4); Hasbrouck,
      Empirical Market Microstructure (2007), 3.2 and 5.1; Glosten and Milgrom (1985), J. Financial Economics 14(1)
    aliases: Roll spread, Roll's spread, Roll measure, Roll estimator, Roll (1984), bid-ask bounce, Glosten-Milgrom
    example:
        # spread about 0.151186; covariance about -0.00571429.
        qf.roll_spread(prices=[100.0, 100.1, 100.0, 100.1, 100.2, 100.1, 100.2, 100.1, 100.0, 100.1],
                       changes="price", ddof=1,
                       positive_cov="zero")
    """
    choice("changes", changes, ROLL_CHANGES)
    choice("positive_cov", positive_cov, ROLL_POSITIVE)
    p = _series("prices", prices)
    if ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, not {ddof!r}")
    if changes == "log":
        p = np.log(positive("prices", p))
    d = np.diff(p)
    if d.size < 2 + ddof:
        raise ValueError("too few prices for the covariance of successive changes")
    c = float(np.cov(d[1:], d[:-1], ddof=ddof)[0, 1])
    spread = 2.0 * math.sqrt(-c) if c < 0 else (0.0 if positive_cov == "zero" else math.nan)
    return Result({"spread": spread, "covariance": c, "n_pairs": d.size - 1})


def _book(bid_prices, bid_sizes, ask_prices, ask_sizes, weights):
    """The four sides as (T, L) arrays, the first L levels the weights name, level 1 the best; a (T,) side one level."""
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
        raise ValueError("weights must hold one non-negative weight per level used, not all zero")
    sides = []
    for name, side in (("bid_prices", bid_prices), ("bid_sizes", bid_sizes), ("ask_prices", ask_prices),
                       ("ask_sizes", ask_sizes)):
        a = np.asarray(side, dtype=float)
        a = a[:, None] if a.ndim == 1 else a
        if a.ndim != 2 or a.shape[1] < w.size or not np.all(np.isfinite(a[:, :w.size])):
            raise ValueError(f"{name} must be (T, L) with at least the {w.size} levels the weights name, finite")
        sides.append(a[:, :w.size])
    if len({a.shape for a in sides}) != 1:
        raise ValueError("the four sides must have the same snapshots")
    if np.any(sides[1] < 0) or np.any(sides[3] < 0):
        raise ValueError("sizes must not be negative")
    return (*sides, w)


@_guard(price=('bid_prices', 'ask_prices',))
def book_imbalance(bid_prices, bid_sizes, ask_prices, ask_sizes, weights):
    """Depth imbalance of the limit order book, (bids - asks) / (bids + asks) over the levels; the microprice.

    Snapshot t: I_t = (sum_l w_l b_l - sum_l w_l a_l) / (sum_l w_l b_l + sum_l w_l a_l), b_l and a_l the bid and ask
    sizes at level l (1 the best), w_l the weights: inside [-1, 1], positive when the bids are deeper (buying
    pressure), 0 for a symmetric book. The weighted mid at the best level, (P_ask q_bid + P_bid q_ask) / (q_bid +
    q_ask), leans toward the side with less size; it is often called the microprice (Stoikov's (2018) micro-price
    adjusts it further and is not this). The order flow between snapshots is order_flow_imbalance.

    conventions:
    - bid_prices, bid_sizes, ask_prices, ask_sizes: (T, L), a row a snapshot, column l the l-th best level; (T,) for
      the best level alone
    - weights: one weight per level used, the first levels taken: [1.0] the best level alone, [1, 1, 1] three levels
      equally, a decaying form such as np.exp(-0.5 * np.arange(5)) as the task names it
    returns: a dict: imbalance (one per snapshot); bid_depth and ask_depth (the weighted sizes); mid and weighted_mid
      (one per snapshot)
    source: Cont, Kukanov and Stoikov (2014), J. Financial Econometrics 12(1); Cartea, Jaimungal and Penalva,
      Algorithmic and High-Frequency Trading (2015), 4.3; Gould and Bonart (2016), Market Microstructure and Liquidity
    aliases: order book imbalance, book imbalance, depth imbalance, queue imbalance, weighted mid price, microprice, [LOB imbalance]
    example:
        # imbalance about [0.066667, -0.230769]; weighted_mid about [99.9625, 99.933333].
        qf.book_imbalance(bid_prices=[[99.9, 99.8], [99.9, 99.8]], bid_sizes=[[500, 300], [200, 300]],
                          ask_prices=[[100.0, 100.1], [100.0, 100.1]], ask_sizes=[[300, 400], [400, 400]],
                          weights=[1.0, 1.0])
    """
    bid_p, bid_q, ask_p, ask_q, w = _book(bid_prices, bid_sizes, ask_prices, ask_sizes, weights)
    bids, asks = bid_q @ w, ask_q @ w
    total, best = bids + asks, bid_q[:, 0] + ask_q[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        imbalance = np.where(total > 0, (bids - asks) / total, np.nan)
        weighted = np.where(best > 0, (ask_p[:, 0] * bid_q[:, 0] + bid_p[:, 0] * ask_q[:, 0]) / best, np.nan)
    return Result({"imbalance": imbalance, "bid_depth": bids, "ask_depth": asks,
                   "mid": (bid_p[:, 0] + ask_p[:, 0]) / 2.0, "weighted_mid": weighted})


@_guard(price=('bid_prices', 'ask_prices',))
def order_flow_imbalance(bid_prices, bid_sizes, ask_prices, ask_sizes, weights):
    """Order flow imbalance (OFI) of Cont, Kukanov and Stoikov from book snapshots, by level and summed.

    Between snapshots n - 1 and n, at level l: e_l = b_n 1{B_n >= B_{n-1}} - b_{n-1} 1{B_n <= B_{n-1}} - a_n 1{A_n <=
    A_{n-1}} + a_{n-1} 1{A_n >= A_{n-1}}, B and A the bid and ask prices, b and a their sizes: a bid that rises adds
    its new size, one that falls takes off the old, one that stays adds the change in size; the ask the other way.
    OFI_n = sum_l w_l e_l (the multi-level OFI; level 1 alone is Cont, Kukanov and Stoikov's). Summed over an
    interval's snapshots it is that interval's OFI, whose regression explains the mid-price change (CKS's price
    impact slope); dividing by the average depth makes it comparable across stocks.

    conventions:
    - bid_prices, bid_sizes, ask_prices, ask_sizes: (T, L), a row a snapshot in time order, column l the l-th best
      level; (T,) for the best level alone
    - weights: one weight per level used, the first levels taken: [1.0] the best level alone (CKS), [1, 1, 1] three
      levels equally, a decaying form such as np.exp(-0.5 * np.arange(5)) as the task names it
    returns: a dict: ofi (one per change, T - 1: between snapshots n - 1 and n); by_level (T - 1 x L, e_l unweighted);
      cumulative (the running sum of ofi)
    source: Cont, Kukanov and Stoikov (2014), "The price impact of order book events", J. Financial Econometrics 12(1);
      Xu, Gould and Howison (2018), "Multi-level order-flow imbalance in a limit order book"
    aliases: order flow imbalance, OFI, multi-level OFI, Cont-Kukanov-Stoikov, [order flow]
    example:
        # ofi [100, 400]; cumulative [100, 500].
        qf.order_flow_imbalance(bid_prices=[99.9, 99.9, 100.0], bid_sizes=[500, 600, 200],
                                ask_prices=[100.1, 100.1, 100.1], ask_sizes=[300, 300, 100],
                                weights=[1.0])
    """
    bid_p, bid_q, ask_p, ask_q, w = _book(bid_prices, bid_sizes, ask_prices, ask_sizes, weights)
    if bid_p.shape[0] < 2:
        raise ValueError("order flow needs at least two snapshots")
    up_b, down_b = bid_p[1:] >= bid_p[:-1], bid_p[1:] <= bid_p[:-1]
    down_a, up_a = ask_p[1:] <= ask_p[:-1], ask_p[1:] >= ask_p[:-1]
    levels = bid_q[1:] * up_b - bid_q[:-1] * down_b - ask_q[1:] * down_a + ask_q[:-1] * up_a
    flow = levels @ w
    return Result({"ofi": flow, "by_level": levels, "cumulative": np.cumsum(flow)})


# ------------------------------------------------------------------------------------------------ price discovery


def information_share(prices, lags, constant):
    """Hasbrouck information shares of venues quoting one security (VECM; Cholesky bounds), Gonzalo-Granger shares.

    The n venues' prices p_t (log prices) share one efficient price: p_1 - p_j is stationary. The VECM dp_t = c +
    alpha z_{t-1} + sum_{i=1}^{k} G_i dp_{t-i} + e_t, z = (p_1 - p_2, ..., p_1 - p_n), is fitted by OLS, Omega =
    cov(e). The efficient price's innovation psi e_t: psi = alpha_perp' / (alpha_perp' (I - sum G_i) 1), alpha_perp
    spanning the null space of alpha'. With F the lower Cholesky factor of Omega in the venues' order, venue j's share
    IS_j = (psi F)_j^2 / (psi Omega psi'); the shares sum to 1 at each ordering. Each venue's upper and lower bounds
    are its largest and smallest share over every ordering of the venues (their midpoint is often reported). The
    component share (Gonzalo-Granger) is alpha_perp_j / sum alpha_perp.

    conventions:
    - prices: (T, n), the n venues' (log) prices of one asset at common times, oldest first; n from 2 to six
    - lags: k, the lagged price changes in the VECM (0 or more)
    - constant: True puts a constant in each equation; False leaves it out
    returns: a dict: shares (n, at the order given, the first venue first in the Cholesky factor); upper and lower (n,
      over every ordering); midpoint ((upper + lower) / 2); component_share (n); psi (n); omega (n x n); alpha (n x n -
      1, the loadings on p_1 - p_j)
    source: Hasbrouck (1995), "One security, many markets", J. Finance 50(4); Baillie, Booth, Tse and Zabotina (2002),
      J. Financial Markets 5(3); Gonzalo and Granger (1995), J. Business and Economic Statistics 13(1)
    aliases: information share, Hasbrouck information share, price discovery, Gonzalo-Granger, Hasbrouck (1995), [VECM]
    example:
        # shares about [0.997836, 0.002164] in the given ordering.
        qf.information_share(prices=[[4.600, 4.601], [4.602, 4.600], [4.601, 4.602], [4.604, 4.601], [4.603, 4.604],
                                     [4.606, 4.604], [4.605, 4.606], [4.603, 4.605], [4.607, 4.603], [4.606, 4.608]],
                             lags=1, constant=True)
    """
    if not isinstance(constant, (bool, np.bool_)):
        raise ValueError(f"constant must be True or False, not {constant!r}")
    p = np.asarray(prices, dtype=float)
    if p.ndim != 2 or not 2 <= p.shape[1] <= MOST_VENUES or not np.all(np.isfinite(p)):
        raise ValueError(f"prices must be a finite (T, n) array of 2 to {MOST_VENUES} venues")
    k, n = integer("lags", lags, 0), p.shape[1]
    dp = np.diff(p, axis=0)
    rows = np.arange(k, dp.shape[0])
    if rows.size <= n * (k + 1) + 2:
        raise ValueError("too few prices for the venues and lags")
    gap = p[rows][:, :1] - p[rows][:, 1:]                       # z_{t-1}: the level before the change dp[t]
    design = [np.ones((rows.size, 1))] if constant else []
    design = np.column_stack(design + [gap] + [dp[rows - i] for i in range(1, k + 1)])
    coef = np.linalg.lstsq(design, dp[rows], rcond=None)[0]
    errors = dp[rows] - design @ coef
    omega = errors.T @ errors / rows.size
    first = int(constant)
    alpha = coef[first:first + n - 1].T
    gammas = sum((coef[first + n - 1 + (i - 1) * n:first + n - 1 + i * n].T for i in range(1, k + 1)),
                 np.zeros((n, n)))
    perp = np.linalg.svd(alpha.T)[2][-1]
    psi = perp / (perp @ (np.eye(n) - gammas) @ np.ones(n))
    total = float(psi @ omega @ psi)

    def shares(order):
        factor = np.linalg.cholesky(omega[np.ix_(order, order)])
        found = np.empty(n)
        found[list(order)] = (psi[list(order)] @ factor) ** 2 / total
        return found

    every = np.array([shares(order) for order in itertools.permutations(range(n))])
    upper, lower = every.max(axis=0), every.min(axis=0)
    return Result({"shares": shares(tuple(range(n))), "upper": upper, "lower": lower, "midpoint": (upper + lower) / 2,
                   "component_share": perp / perp.sum(), "psi": psi, "omega": omega, "alpha": alpha})


# ------------------------------------------------------------------------------------------------ Hawkes


def _hawkes_terms(times, mu, alpha, beta):
    """The intensities just before each event, mu + alpha sum_{t_j < t_i} exp(-beta (t_i - t_j)), and the
    compensator's increments between events, by the recursion A_i = exp(-beta (t_i - t_{i-1})) (1 + A_{i-1})."""
    excite = np.zeros(times.size)
    for i in range(1, times.size):
        excite[i] = math.exp(-beta * (times[i] - times[i - 1])) * (1.0 + excite[i - 1])
    return mu + alpha * excite, excite


def hawkes_fit(times, start, end, initial):
    """Hawkes self-exciting process by MLE, exponential kernel: baseline intensity, jump size and decay rate.

    The intensity lambda(t) = mu + sum_{t_i < t} alpha exp(-beta (t - t_i)): each event raises it by alpha, which
    decays at rate beta. On the window [start, end] the log-likelihood is sum_i ln lambda(t_i) - mu (end - start) -
    (alpha / beta) sum_i (1 - exp(-beta (end - t_i))), maximised over mu, alpha, beta > 0 with the branching ratio
    n = alpha / beta below 1 (stationary): n is the share of events triggered by earlier ones, mu / (1 - n) the mean
    intensity. Some write the kernel alpha' beta exp(-beta t), alpha' = n the branching ratio (the tick library's
    adjacency): both are returned. Under the fit the compensator's increments between events are exponential with mean
    1 (the time-rescaling check).

    conventions:
    - times: the event times in one unit (seconds, say), increasing, inside [start, end]
    - start, end: the observation window (start the first event's time, or the window's open, as the task says)
    - initial: (mu, alpha, beta), the optimiser's first point, alpha below beta
    returns: a dict: mu; alpha (the intensity's jump at an event); beta (the decay rate, per time unit);
      branching_ratio (alpha / beta); mean_intensity (mu / (1 - alpha / beta)); loglik; converged; intensity (just
      before each event); residuals (the compensator's increments, one per gap between events)
    source: Hawkes (1971), Biometrika 58(1); Ozaki (1979), Annals of the Institute of Statistical Mathematics 31; Bacry,
      Mastromatteo and Muzy (2015), "Hawkes processes in finance", Market Microstructure and Liquidity 1(1)
    aliases: Hawkes process, Hawkes, self-exciting process, branching ratio, Hawkes MLE
    example:
        # loglik about -8.881809; branching_ratio about 0.228437.
        qf.hawkes_fit(times=[0.5, 0.9, 1.0, 2.6, 2.7, 2.75, 4.1, 5.8, 5.9, 7.4, 8.8, 8.85],
                      start=0.0, end=10.0,
                      initial=(0.5, 0.8, 2.0))
    """
    t = np.asarray(times, dtype=float).ravel()
    t0, t1 = float(start), float(end)
    if (t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite([t0, t1]))
            or np.any(np.diff(t) <= 0) or t[0] < t0 or t[-1] > t1 or t1 <= t0):
        raise ValueError("times must be two or more strictly increasing event times inside finite [start, end]")
    mu0, alpha0, beta0 = (float(v) for v in initial)
    if not (np.all(np.isfinite([mu0, alpha0, beta0])) and mu0 > 0 and alpha0 > 0 and beta0 > alpha0):
        raise ValueError("initial must be finite (mu, alpha, beta) with mu, alpha > 0 and alpha < beta")

    def unpack(theta):
        mu, beta = math.exp(theta[0]), math.exp(theta[2])
        return mu, beta / (1.0 + math.exp(-theta[1])), beta

    def loss(theta):
        mu, alpha, beta = unpack(theta)
        intensity, _ = _hawkes_terms(t, mu, alpha, beta)
        return -(float(np.sum(np.log(intensity))) - mu * (t1 - t0)
                 - alpha / beta * float(np.sum(-np.expm1(-beta * (t1 - t)))))

    ratio = alpha0 / beta0
    found = optimize.minimize(loss, np.array([math.log(mu0), math.log(ratio / (1.0 - ratio)), math.log(beta0)]),
                              method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    found = optimize.minimize(loss, found.x, method="BFGS", options={"gtol": 1e-9})
    mu, alpha, beta = unpack(found.x)
    intensity, excite = _hawkes_terms(t, mu, alpha, beta)
    gaps = np.diff(t)
    residuals = mu * gaps + alpha / beta * (1.0 + excite[:-1]) * (-np.expm1(-beta * gaps))
    return Result({"mu": mu, "alpha": alpha, "beta": beta, "branching_ratio": alpha / beta,
                   "mean_intensity": mu / (1.0 - alpha / beta), "loglik": -float(found.fun),
                   "converged": bool(found.success), "intensity": intensity, "residuals": residuals})


def hawkes_simulate(mu, alpha, beta, start, end, rng):
    """Simulated Hawkes process (Ogata's thinning), exponential kernel: the event times.

    From an empty history at start: at time s the intensity mu + alpha sum exp(-beta (s - t_i)) can only fall until
    the next event, so its value now bounds it; a candidate s + Exp(that bound) is kept with probability lambda(s) /
    bound, the bound renewed at each candidate. The expected count from an empty start is mu T / (1 - n) - mu n (1 -
    exp(-(beta - alpha) T)) / ((beta - alpha)(1 - n)), n = alpha / beta < 1, T = end - start.

    conventions:
    - mu: the baseline intensity, events per time unit; alpha: the intensity's jump at an event; beta: its decay
      rate (alpha below beta)
    - start, end: the window, in the times' unit
    - rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); each candidate draws an
      exponential gap, then a uniform
    returns: an array of the event times, increasing
    source: Ogata (1981), "On Lewis' simulation method for point processes", IEEE Transactions on Information Theory
      27(1); Hawkes (1971), Biometrika 58(1)
    aliases: simulate Hawkes process, Ogata thinning
    example:
        # Event times begin [1.415059, 1.852404, 1.985898].
        qf.hawkes_simulate(mu=0.5, alpha=0.8, beta=2.0, start=0.0, end=100.0,
                           rng=7)
    """
    mu, alpha, beta = (float(v) for v in (mu, alpha, beta))
    if not (np.all(np.isfinite([mu, alpha, beta])) and mu > 0 and alpha >= 0 and beta > alpha):
        raise ValueError("mu must be positive and finite; alpha and beta must be finite with 0 <= alpha < beta")
    rng = _stream(rng)
    now, finish = float(start), float(end)
    if not (np.isfinite(now) and np.isfinite(finish) and finish > now):
        raise ValueError("start and end must be finite, with end after start")
    found, excite = [], 0.0                     # excite: sum exp(-beta (now - t_i)) over the events so far
    while True:
        bound = mu + alpha * excite
        gap = rng.exponential(1.0 / bound)
        now += gap
        if now > finish:
            break
        excite *= math.exp(-beta * gap)
        if rng.random() * bound <= mu + alpha * excite:
            found.append(now)
            excite += 1.0
    return np.array(found)
