"""Clocks: year fractions between timestamps (ACT/365, ACT/365.25), US 30/360, a calendar-span growth rate.

`import qf`. The day counts of qf.year_fraction count whole days; these count what their names say: elapsed time to
the nanosecond, the SIA's US 30/360 with its end-of-February rule, a return over the calendar days it spans. A scalar
result is a float.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np

from ._common import choice, out, positive
from .conventions import _dates, _month_and_day

__all__ = ["calendar_annualized_return", "thirty_360_us", "timestamp_year_fraction"]

TIMESTAMP_BASES = ("ACT/365", "ACT/365F", "ACT/365.25", "ACT/360")
DAYS_A_YEAR = {"ACT/365": 365.0, "ACT/365F": 365.0, "ACT/365.25": 365.25, "ACT/360": 360.0}
GROWTH_KINDS = ("log", "simple")
NS_A_DAY = 86_400 * 10 ** 9
ZONED = re.compile(r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}(?::?\d{2})?)$", re.I)


def _instant(name, item):
    """One time as numpy.datetime64[ns] (a zoned time in UTC) and whether it carried a zone."""
    if isinstance(item, np.datetime64):
        return item.astype("datetime64[ns]"), False
    if hasattr(item, "to_datetime64"):                      # a pandas Timestamp: its own nanoseconds, UTC if zoned
        return np.datetime64(item.to_datetime64(), "ns"), getattr(item, "tzinfo", None) is not None
    if isinstance(item, str):
        text = item.strip()
        if not ZONED.search(text):
            try:
                return np.datetime64(text, "ns"), False
            except ValueError as error:
                raise ValueError(f"{name}: {item!r} is not an ISO date or time") from error
        try:
            item = dt.datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError(f"{name}: {item!r} is not an ISO date or time") from error
    if isinstance(item, dt.datetime):
        if item.tzinfo is not None and item.utcoffset() is not None:
            return np.datetime64(item.astimezone(dt.timezone.utc).replace(tzinfo=None), "ns"), True
        return np.datetime64(item, "ns"), False
    if isinstance(item, dt.date):
        return np.datetime64(item, "ns"), False
    raise ValueError(f"{name} must be dates or times (ISO strings, datetime, numpy.datetime64), not {item!r}")


def _instants(name, value):
    """`value` as a numpy.datetime64[ns] array and an array of whether each entry carried a zone (made UTC). Numbers
    are refused: epoch milliseconds are numpy.asarray(ms, dtype='datetime64[ms]')."""
    array = np.asarray(value)
    if array.dtype.kind == "M":
        return array.astype("datetime64[ns]"), np.zeros(array.shape, bool)
    if array.dtype.kind in "biuf":
        raise ValueError(f"{name} must be dates or times, not numbers: epoch milliseconds are "
                         f"numpy.asarray(ms, dtype='datetime64[ms]'), seconds dtype='datetime64[s]'")
    flat = array.ravel()
    found = np.empty(flat.size, dtype="datetime64[ns]")
    zoned = np.zeros(flat.size, bool)
    for i, item in enumerate(flat):
        found[i], zoned[i] = _instant(name, item)
    return found.reshape(array.shape), zoned.reshape(array.shape)


def timestamp_year_fraction(start, end, basis):
    """Year fraction between timestamps, hours and seconds counted: ACT/365, ACT/365.25 or ACT/360.

    (end - start) in elapsed time to the nanosecond over days_per_year x 86,400 seconds: 365 for 'ACT/365' (=
    'ACT/365F'), 365.25 for 'ACT/365.25', 360 for 'ACT/360'. qf.year_fraction counts whole days (a time of day is
    dropped there); this counts the time between the two instants. A start after the end gives a negative fraction;
    arrays broadcast.

    conventions:
    - basis: 'ACT/365' (= 'ACT/365F'), 'ACT/365.25' or 'ACT/360', the days in a year
    - start, end: ISO strings ('2024-03-01T16:00:00', a 'Z' or '+01:00' zone made UTC), datetime or numpy.datetime64;
      a zoned and a zone-free time together are refused; epoch milliseconds: numpy.asarray(ms, dtype='datetime64[ms]')
    returns: a float, or an array for arrays of times
    source: ISDA 2006 Definitions, 4.16 (Actual/365 (Fixed), Actual/360); the Julian year of 365.25 days (IAU)
    aliases: ACT/365.25, timestamp year fraction, exact year fraction, year fraction from timestamps, time to expiry from timestamps
    example:
        # Returns 0.25 (91 days and 7.5 hours over 365.25 days).
        qf.timestamp_year_fraction(start="2024-01-01T08:30:00", end="2024-04-01T16:00:00", basis="ACT/365.25")
    """
    choice("basis", basis, TIMESTAMP_BASES)
    (s, s_zoned), (e, e_zoned) = _instants("start", start), _instants("end", end)
    if np.any(np.isnat(s)) or np.any(np.isnat(e)):
        raise ValueError("start and end must be valid times, not NaT")
    if (s_zoned.any() or e_zoned.any()) and not (s_zoned.all() and e_zoned.all()):
        raise ValueError("start and end mix times with a zone and times without one: give every time a zone, or none")
    s, e = np.broadcast_arrays(s, e)
    elapsed = (e - s).astype("timedelta64[ns]").astype(np.int64)
    return out(elapsed / (DAYS_A_YEAR[basis] * NS_A_DAY))


def thirty_360_us(start, end, eom):
    """US 30/360 year fraction (SIA), its end-of-February rule named; Bond Basis is year_fraction's.

    [360 (Y2 - Y1) + 30 (M2 - M1) + (D2 - D1)] / 360 after the SIA's rules in order: (1) where eom, D1 and D2 both a
    February's last day: D2 = 30; (2) where eom, D1 a February's last day: D1 = 30; (3) D2 = 31 and D1 30 or 31: D2 =
    30; (4) D1 = 31: D1 = 30. Without the February rules (eom False) it is ISDA's 30/360 Bond Basis, qf.year_fraction's
    '30/360'. A start after the end gives a negative fraction; arrays broadcast.

    conventions:
    - eom: True applies the end-of-February rules (1) and (2), the SIA's US 30/360 and QuantLib's Thirty360 USA;
      False leaves February's last day as it is (Bond Basis)
    - start, end: dates (ISO strings, datetime.date or numpy.datetime64; a time of day is dropped)
    returns: a float, or an array for arrays of dates
    source: SIA, Standard Securities Calculation Methods (1993), 30/360 day count; ISDA 2006 Definitions, 4.16(f)
    aliases: 30/360 US, US 30/360, 30U/360, 30/360 SIA, 30/360 NASD, [bond basis]
    example:
        # Returns 0.5: 2023-02-28 is February's last day, read as the 30th.
        qf.thirty_360_us(start="2023-02-28", end="2023-08-31", eom=True)
    """
    if not isinstance(eom, (bool, np.bool_)):
        raise ValueError(f"eom must be True or False, not {eom!r}")
    s, e = np.broadcast_arrays(_dates("start", start), _dates("end", end))
    m1, d1 = _month_and_day(s)
    m2, d2 = _month_and_day(e)
    if eom:
        def february_end(days, months):
            return (months % 12 == 1) & (_month_and_day(days + np.timedelta64(1, "D"))[1] == 1)

        end_1, end_2 = february_end(s, m1), february_end(e, m2)
        d2 = np.where(end_1 & end_2, 30, d2)
        d1 = np.where(end_1, 30, d1)
    d2 = np.where((d2 == 31) & (d1 >= 30), 30, d2)
    d1 = np.minimum(d1, 30)
    return out((30 * (m2 - m1) + (d2 - d1)) / 360.0)


def calendar_annualized_return(returns, dates, kind, days_per_year):
    """Annualised return over the calendar span, first to last date over 365.25 or 365 days.

    years = (last date - first date) in days / days_per_year; 'log': sum r / years, the continuously compounded
    annual rate of log returns (exp of it less 1 is the simple rate); 'simple': [prod (1 + r)]^(1 / years) - 1, the
    compound annual growth of simple returns. Not the trading-period form (mean x 252 is qf.annualize, the count of
    returns qf.cagr): the clock is the calendar. The dates are the returns' own, so the span runs from the first
    return's date to the last's, one period short of the holding span; to span from the first price's date, put a
    zero return first, dated there.

    conventions:
    - kind: 'log' (log returns: sum over years) or 'simple' (simple returns: compounded, then the years' root)
    - days_per_year: 365.25 (the Julian year: a span's leap days averaged) or 365, a positive number
    - returns: one finite return per date, in order; dates: increasing dates of the same length (whole days)
    returns: a float, the annual rate as a decimal
    source: Bacon, Practical Portfolio Performance Measurement and Attribution (2008), annualised returns; CFA
      Institute, GIPS (2020), annualising by the actual days of the period
    aliases: calendar-span annualisation, calendar span, elapsed years, annualised over calendar days, [days / 365.25]
    example:
        # Returns 0.015: log returns summing to 0.06 over 1,461 days, four years of 365.25.
        qf.calendar_annualized_return(returns=[0.01, 0.02, 0.03], dates=["2020-01-01", "2022-01-01", "2024-01-01"], kind="log", days_per_year=365.25)
    """
    choice("kind", kind, GROWTH_KINDS)
    per_year = float(positive("days_per_year", days_per_year))
    r = np.asarray(returns, dtype=float)
    days = _dates("dates", dates)
    if r.ndim != 1 or r.size < 2 or not np.all(np.isfinite(r)):
        raise ValueError("returns must be a 1-d series of at least two finite returns")
    if days.shape != r.shape or np.any(np.diff(days).astype(int) <= 0):
        raise ValueError("dates must be increasing, one per return")
    years = float((days[-1] - days[0]).astype(int)) / per_year
    if kind == "log":
        return float(r.sum() / years)
    if np.any(r <= -1.0):
        raise ValueError("simple returns must be above -1 (a loss of everything or more)")
    return float(np.exp(np.log1p(r).sum() / years) - 1.0)
