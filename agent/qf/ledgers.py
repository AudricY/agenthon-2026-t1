"""Strategy ledgers: whole shares and cash, next-open fills, flat and percent fees; drawdowns from returns.

`import qf`. A series is one value per bar, oldest first. Where qf.backtest_signals holds fractional positions
rebalanced to their weight every bar, the ledger here holds whole shares bought with the cash there is; a function
with several results returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np

from ._common import Result, choice
from .conventions import SIGNS

__all__ = ["drawdown_from_returns", "share_backtest"]

FILLS = ("next_open", "same_close")
SIZINGS = ("cash", "cash_after_fees")
DRAWDOWN_STARTS = ("initial_capital", "first_return")
WEALTH_FORMS = ("compound", "log", "sum")
DRAWDOWN_MEASURES = ("ratio", "difference")


def _bars(name, values, size):
    """`values` as `size` positive finite prices, else a ValueError naming them."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size != size or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must hold one positive finite price per signal")
    return array


def _amount(name, value):
    amount = float(value)
    if not (math.isfinite(amount) and amount >= 0):
        raise ValueError(f"{name} must be a finite amount, 0 or more")
    return amount


def share_backtest(signals, open, close, initial_cash, execution, sizing, fee_flat, fee_rate, final_close):
    """Whole-share long-or-flat backtest with cash: fills at the next open or the close, flat and percent fees.

    signals[t] is the target, 1 long or 0 flat, known at bar t's close. Where it differs from the book (long while
    shares are held) the trade fills at bar t + 1's open ('next_open'; the last bar's signal never fills) or at bar
    t's close ('same_close'). A buy takes floor(cash / price) shares ('cash', the fees then taken from the cash left,
    which they may push below zero) or the most whole shares whose cost and fees the cash covers ('cash_after_fees');
    a sell sells every share. Each trade pays fee_flat + fee_rate x its value. Where final_close is True a position
    still open is sold at the last close, a trade with its fees. equity[t] = cash + shares x close[t], after the
    bar's trades; returns[0] = 0 and returns[t] = equity[t] / equity[t - 1] - 1; wealth is cumprod(1 + returns).

    conventions:
    - execution: 'next_open' (signal at t's close, fill at t + 1's open) or 'same_close' (fill at t's close)
    - sizing: 'cash' (floor(cash / price), fees after) or 'cash_after_fees' (fees reserved before the floor)
    - fee_flat: a currency amount each trade (0 for none); fee_rate: a decimal of each trade's value (0.001 is
      10 bp; 0 for none)
    - final_close: True sells an open position at the last close; False leaves it marked at that close
    - signals: 1 or 0 each bar (a warm-up's NaN filled with 0 first); a crossover entry is the state after the cross
    - open: the opens, read by 'next_open' only (None with 'same_close'); close: the closes; initial_cash: above 0
    returns: a dict: equity, returns, cash, shares (one per bar, after its trades); trades (a list of dicts: bar,
      side 'BUY' or 'SELL', shares, price, fee, cash after)
    source: Chan, Quantitative Trading (2008), backtesting
    aliases: whole-share backtest, whole shares backtest, cash accounting backtest, SMA crossover backtest, moving average crossover backtest, [forced final close]
    example:
        # Buys 9 shares at 10.25 at the third open, sells them at 10.4 at the fifth: equity ends 101.35.
        qf.share_backtest(signals=[0, 1, 1, 0, 0], open=[10.0, 10.1, 10.25, 10.6, 10.4], close=[10.0, 10.2, 10.5, 10.4, 10.3], initial_cash=100.0, execution="next_open", sizing="cash", fee_flat=0.0, fee_rate=0.0, final_close=True)
    """
    choice("execution", execution, FILLS)
    choice("sizing", sizing, SIZINGS)
    s = np.asarray(signals, dtype=float)
    if s.ndim != 1 or s.size < 1 or not np.all(np.isin(s, (0.0, 1.0))):
        raise ValueError("signals must be a 1-d series of 1 (long) or 0 (flat): fill a warm-up's NaN with 0 first")
    c = _bars("close", close, s.size)
    o = _bars("open", open, s.size) if execution == "next_open" else None
    cash = float(initial_cash)
    if not (math.isfinite(cash) and cash > 0):
        raise ValueError("initial_cash must be a positive amount")
    flat, rate = _amount("fee_flat", fee_flat), _amount("fee_rate", fee_rate)
    if not isinstance(final_close, (bool, np.bool_)):
        raise ValueError(f"final_close must be True or False, not {final_close!r}")
    shares, trades = 0, []
    equity, cash_path, shares_path = np.empty(s.size), np.empty(s.size), np.zeros(s.size, dtype=np.int64)

    def trade(bar, target, price):
        nonlocal cash, shares
        price = float(price)
        if target == 1.0 and shares == 0:
            room = cash if sizing == "cash" else (cash - flat) / (1.0 + rate)
            count = int(math.floor(room / price)) if room > 0 else 0
            if count >= 1:
                fee = flat + rate * count * price
                cash -= count * price + fee
                shares = count
                trades.append({"bar": bar, "side": "BUY", "shares": count, "price": price, "fee": fee,
                               "cash": cash})
        elif target == 0.0 and shares > 0:
            fee = flat + rate * shares * price
            cash += shares * price - fee
            trades.append({"bar": bar, "side": "SELL", "shares": shares, "price": price, "fee": fee,
                           "cash": cash})
            shares = 0

    for bar in range(s.size):
        if execution == "next_open" and bar > 0:
            trade(bar, s[bar - 1], o[bar])
        elif execution == "same_close":
            trade(bar, s[bar], c[bar])
        if final_close and bar == s.size - 1:
            trade(bar, 0.0, c[bar])
        equity[bar], cash_path[bar], shares_path[bar] = cash + shares * c[bar], cash, shares
    returns = np.zeros(s.size)
    returns[1:] = equity[1:] / equity[:-1] - 1.0
    return Result({"equity": equity, "returns": returns, "cash": cash_path, "shares": shares_path, "trades": trades})


def drawdown_from_returns(returns, start, wealth, measure, sign):
    """Drawdown path and maximum from a return series, the start point, wealth form and sign named.

    The wealth path V: 'compound' V_k = prod_{j<=k} (1 + r_j) (simple returns), 'log' exp(sum_{j<=k} r_j) (log
    returns), 'sum' 1 + sum_{j<=k} r_j (returns added, not compounded). 'initial_capital' puts V = 1 before the first
    return, a peak a loss from the start falls from; 'first_return' starts the path at V_1, so a first-period loss is
    no drawdown. With M_k = max_{j<=k} V_j: 'ratio' D_k = 1 - V_k / M_k, 'difference' D_k = M_k - V_k (return
    points on a 'sum' path). From levels (prices, equity) instead: qf.max_drawdown.

    conventions:
    - start: 'initial_capital' (a 1.0 before the first return) or 'first_return' (the path's first point is 1 + r_1)
    - wealth: 'compound' (simple returns), 'log' (log returns) or 'sum' (added returns)
    - measure: 'ratio' (a fraction of the peak) or 'difference' (the peak less the value)
    - sign: 'loss_positive' (a 25% fall is 0.25) or 'return_negative' (it is -0.25)
    returns: a dict: max_drawdown (signed); drawdown (signed, one per wealth point); wealth (the path); peak and
      trough (the worst fall's positions in wealth, 0 the initial capital's under 'initial_capital')
    source: Magdon-Ismail and Atiya (2004), "Maximum drawdown", Risk 17(10); Bacon, Practical Portfolio Performance
      Measurement and Attribution (2008), drawdown
    aliases: drawdown from returns, maximum drawdown from returns, drawdown with initial capital, [underwater curve], [drawdown series]
    example:
        # Returns max_drawdown 0.1 (the first loss falls from the initial 1.0).
        qf.drawdown_from_returns(returns=[-0.1, 0.05, 0.02], start="initial_capital", wealth="compound", measure="ratio", sign="loss_positive")
    """
    choice("start", start, DRAWDOWN_STARTS)
    choice("wealth", wealth, WEALTH_FORMS)
    choice("measure", measure, DRAWDOWN_MEASURES)
    choice("sign", sign, SIGNS)
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 1 or not np.all(np.isfinite(r)):
        raise ValueError("returns must be a 1-d series of finite returns")
    if wealth == "compound":
        if np.any(r <= -1.0):
            raise ValueError("compound returns must be above -1 (a loss of everything or more)")
        path = np.cumprod(1.0 + r)
    elif wealth == "log":
        path = np.exp(np.cumsum(r))
    else:
        path = 1.0 + np.cumsum(r)
    if start == "initial_capital":
        path = np.concatenate([[1.0], path])
    peak = np.maximum.accumulate(path)
    if measure == "ratio":
        if np.any(peak <= 0):
            raise ValueError("measure 'ratio' needs a positive peak: an added-return path at or below 0 has none")
        fall = 1.0 - path / peak
    else:
        fall = peak - path
    trough = int(np.argmax(fall))
    top = int(np.argmax(path[:trough + 1]))
    drawdown = fall if sign == "loss_positive" else 0.0 - fall
    return Result({"max_drawdown": float(drawdown[trough]), "drawdown": drawdown, "wealth": path, "peak": top,
                   "trough": trough})
