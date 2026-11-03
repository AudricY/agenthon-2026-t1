"""Realized variance from intraday prices, every k ticks or k minutes; microstructure noise, bipower variation.

Returns are log returns of one day (or one period) in order; every result is on that period's scale, not
annualised: the realized variance, Bandi and Russell's microstructure-noise variance, the noise-corrected realized
variance and Barndorff-Nielsen and Shephard's bipower variation.
"""
from __future__ import annotations

import re

import numpy as np

from ._common import Result, choice, out
from .clocks import _instants
from .volatility import _int, _series

__all__ = ["bandi_russell_noise_var", "bipower_variation", "clock_realized_variance", "noise_corrected_rv",
           "realized_variance"]

SCALES = ("none", "pi/2", "pi/2*n/(n-1)")
SUBSAMPLES = ("first", "average")
GRID_SAMPLES = ("on_grid", "previous_tick", "last_in_interval")
EMPTY_RULES = ("skip", "drop")
TIME_OF_DAY = re.compile(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$")


def _returns(log_returns):
    r = np.asarray(log_returns, dtype=float).ravel()
    if r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("log_returns must be a non-empty array of finite numbers")
    return r


def bandi_russell_noise_var(log_returns):
    """The noise variance E[eta^2] of observed log prices p_j = p*_j + eta_j (i.i.d. noise)
    from the highest-frequency returns available: RV / (2 n) = sum r_j^2 / (2 n), n the number of returns. Under the
    model E[RV_n] = IV + 2 n E[eta^2], so as n grows the sum of squared returns is dominated by the noise, and the
    estimator is consistent.
    Source: Bandi and Russell (2006), "Separating microstructure noise from volatility", Journal of Financial Economics
    79(3), section 3.1; Bandi and Russell (2008), Review of Economic Studies 75(2).
    aliases: Bandi-Russell, microstructure noise variance, Bandi Russell
    returns: the noise variance E[eta^2], a float in squared log-price units
    example:
        qf.bandi_russell_noise_var(log_returns=[0.0004, -0.0003, 0.0002, -0.0005, 0.0001])
    """
    r = _returns(log_returns)
    return float(np.sum(r * r) / (2.0 * r.size))


def noise_corrected_rv(rv, n_returns, noise_var):
    """The additive bias correction of a realized variance computed from `n_returns` returns: RV - 2 n E[eta^2],
    from E[RV_n] = IV + 2 n E[eta^2] (i.i.d. noise of variance `noise_var`). It can be negative on a quiet day;
    clipping is the caller's choice. Source: Bandi and Russell (2006), J. Financial Economics 79(3); Zhang, Mykland
    and Ait-Sahalia (2005), "A tale of two time scales", JASA 100, the bias of the sparse realized variance.
    aliases: noise-corrected realized variance, bias-corrected realized variance, noise-corrected RV
    returns: the corrected realized variance, a float (it can be negative)
    example:
        qf.noise_corrected_rv(rv=0.0001, n_returns=390, noise_var=1e-8)
    """
    rv, n, w = (np.asarray(a, dtype=float) for a in (rv, n_returns, noise_var))
    return out(rv - 2.0 * n * w)


def bipower_variation(log_returns, scale):
    """Realized bipower variation, sum_{j=2}^{n} |r_j| |r_{j-1}| over the n returns in order, times `scale`:
    'none' the sum itself (Barndorff-Nielsen and Shephard's {Y}^[1,1]); 'pi/2' mu_1^{-2} = pi/2 (mu_1 = E|Z| =
    sqrt(2/pi)), the estimator of integrated variance robust to jumps; 'pi/2*n/(n-1)' that with the finite-sample
    factor n/(n-1). Returns adjacent in time: a gap the caller dropped joins two returns that were not adjacent.
    Source: Barndorff-Nielsen and Shephard (2004), "Power and bipower variation with stochastic volatility and jumps",
    Journal of Financial Econometrics 2(1), definition 1 and theorem 2; Huang and Tauchen (2005), J. Financial
    Econometrics 3(4) (the n/(n-1) factor).
    aliases: bipower variation, realized bipower, jump-robust variance
    returns: the bipower variation, a float in the returns' squared units
    example:
        qf.bipower_variation(log_returns=[0.001, -0.002, 0.0015, -0.0005, 0.001], scale="pi/2")
    """
    choice("scale", scale, SCALES)
    r = np.abs(_returns(log_returns))
    total = float(np.sum(r[1:] * r[:-1]))
    if scale == "none":
        return total
    factor = np.pi / 2.0
    if scale == "pi/2*n/(n-1)":
        factor *= r.size / (r.size - 1.0)
    return factor * total


# ------------------------------------------------------------------------------------------------ realized variance


def realized_variance(log_prices, *, step, subsample):
    """Realized variance of one period's intraday log prices sampled every `step` observations.

    'first': the grid 0, step, 2 step, ...: sum of its squared log returns. 'average': the mean of that sum over the
    step grids that start at offsets 0, 1, ..., step - 1 (Zhang, Mykland and Ait-Sahalia's RV-avg, the subsampled
    estimator). step = 1 is the all-observations RV. From returns r: log prices np.r_[0.0, np.cumsum(r)]. Not
    annualised. Noise corrections: qf.bandi_russell_noise_var and qf.noise_corrected_rv; jumps: qf.bipower_variation.
    conventions:
        step: the rows between the prices sampled, a positive int below the sample size: 1 uses every price given
            (prices already at the task's sampling frequency, as 5-minute bars for a 5-minute RV); k > 1 samples
            every k-th price (5 for a 5-minute RV from 1-minute prices)
        subsample: 'first' (the one grid from the first price) or 'average' (the mean over the step grids)
    returns: a float
    source: Andersen, Bollerslev, Diebold and Labys (2001), JASA 96 (realized variance); Zhang, Mykland and
        Ait-Sahalia (2005), "A tale of two time scales", JASA 100 (the average over subgrids)
    aliases: realized variance, realised variance, [RV], subsampled realized variance, sparse sampling, [realized variation]
    example:
        import qf
        p = [4.6052, 4.6060, 4.6049, 4.6071, 4.6066, 4.6080]
        qf.realized_variance(p, step=1, subsample="first")  # 0.0000089
    """
    choice("subsample", subsample, SUBSAMPLES)
    p = _series("log_prices", log_prices)
    k = _int("step", step, 1, p.size)
    if subsample == "first":
        return float(np.sum(np.diff(p[::k]) ** 2))
    return float(np.mean([np.sum(np.diff(p[offset::k]) ** 2) for offset in range(k)]))


def clock_realized_variance(timestamps, log_prices, minutes, origin, sample, empty):
    """Realized variance on a clock grid every k minutes (not every k observations), empty intervals named.

    The grid g_j = origin + j k minutes. 'on_grid' keeps the prices stamped exactly at a grid time (minute bars whose
    minute from the origin divides by k); 'previous_tick' takes at each grid time from the first price's to the last's
    the last price at or before it (a gap carries the price before it); 'last_in_interval' each interval [g_j, g_j +
    k)'s last price (pandas' resample(k).last(), labelled g_j). A grid time left with no price (not under
    'previous_tick'): 'skip' joins the sampled prices either side into one return (no fill), 'drop' leaves out the two
    returns that touch it (pandas' diff() of the resampled prices, NaN skipped). RV = the sum of the sampled log
    prices' squared returns, one period's (not annualised); sampled by observation count: qf.realized_variance.

    conventions:
    - minutes: k, the grid's spacing (5 for a 5-minute RV); origin: 'first' (the first timestamp), a time of day
      'HH:MM' on the first timestamp's date (the session's open, '09:30'), or a full time
    - sample: 'on_grid', 'previous_tick' or 'last_in_interval'; empty: 'skip' (no fill) or 'drop'
    - timestamps: one period's, increasing, one per price (collapse duplicates first by the task's rule); ISO strings
      or numpy.datetime64 (a zone made UTC: then a full origin, not 'HH:MM'); log_prices: finite, one each
    returns: a dict: rv (a float); n_returns; times and log_prices (the sampled grid's)
    source: Andersen, Bollerslev, Diebold and Labys (2001), JASA 96 (calendar-time sampling); Hansen and Lunde (2006),
      J. Business and Economic Statistics 24(2) (previous-tick sampling)
    aliases: clock-time realized variance, calendar time sampling, previous-tick sampling, 5-minute realized variance, [previous tick], [every k minutes], [5-minute returns]
    example:
        # Returns rv 0.0002 from 0.01 and -0.01 on the 5-minute grid; the 09:33 price is off it.
        qf.clock_realized_variance(timestamps=["2024-03-04T09:30", "2024-03-04T09:33", "2024-03-04T09:35", "2024-03-04T09:40"], log_prices=[4.60, 4.70, 4.61, 4.60], minutes=5, origin="09:30", sample="on_grid", empty="skip")
    """
    choice("sample", sample, GRID_SAMPLES)
    choice("empty", empty, EMPTY_RULES)
    stamps, zoned = _instants("timestamps", timestamps)
    if zoned.any() and not zoned.all():
        raise ValueError("timestamps mix times with a zone and times without one: give every time a zone, or none")
    if stamps.ndim != 1 or stamps.size < 2 or np.any(np.isnat(stamps)):
        raise ValueError("timestamps must be a 1-d series of at least two times")
    ns = stamps.astype(np.int64)
    if np.any(np.diff(ns) <= 0):
        raise ValueError("timestamps must be increasing: collapse duplicated times first, by the task's rule")
    p = np.asarray(log_prices, dtype=float)
    if p.shape != stamps.shape or not np.all(np.isfinite(p)):
        raise ValueError("log_prices must hold one finite log price per timestamp")
    spacing = float(minutes) * 60e9
    step = int(round(spacing)) if np.isfinite(spacing) else 0
    if step <= 0:
        raise ValueError("minutes must be a positive spacing")
    start = _origin(origin, stamps, ns, zoned)
    offsets = ns - start
    if sample == "on_grid":
        kept = offsets % step == 0
        index, values = offsets[kept] // step, p[kept]
    elif sample == "previous_tick":
        index = np.arange(-(-offsets[0] // step), offsets[-1] // step + 1)
        values = p[np.searchsorted(ns, start + index * step, side="right") - 1]
    else:
        bins = offsets // step
        last = np.r_[bins[1:] != bins[:-1], True]
        index, values = bins[last], p[last]
    moves = np.diff(values)
    if empty == "drop":
        moves = moves[np.diff(index) == 1]
    if moves.size == 0:
        raise ValueError("the grid holds no return: fewer than two sampled prices (or none adjacent under 'drop')")
    return Result({"rv": float(np.sum(moves * moves)), "n_returns": int(moves.size),
                   "times": (start + index * step).astype("datetime64[ns]"), "log_prices": values})


def _origin(origin, stamps, ns, zoned):
    """The grid's origin in nanoseconds: the first timestamp, a time of day on its date, or a time given."""
    if isinstance(origin, str) and origin == "first":
        return int(ns[0])
    if isinstance(origin, str) and TIME_OF_DAY.match(origin.strip()):
        if zoned.any():
            raise ValueError("origin 'HH:MM' reads the timestamps' own clock: zoned timestamps take a full origin time")
        parts = [float(part) for part in origin.strip().split(":")]
        seconds = parts[0] * 3600 + parts[1] * 60 + (parts[2] if len(parts) > 2 else 0.0)
        day = stamps[0].astype("datetime64[D]").astype("datetime64[ns]").astype(np.int64)
        return int(day) + int(round(seconds * 1e9))
    found, zone = _instants("origin", origin)
    if found.ndim != 0 or np.isnat(found):
        raise ValueError("origin must be 'first', a time of day 'HH:MM' or one full time")
    if bool(zone) != bool(zoned.any()):
        raise ValueError("origin and timestamps mix a time with a zone and times without one")
    return int(found.astype(np.int64))
