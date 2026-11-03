"""Scaffolding: files, returns, annualising, day counts, compounding, windows, std, quantile, EWMA.

Every convention is a required keyword argument: what units disagree on (ddof, periods a year, simple or log
returns, the quantile rule, the seed of an EWMA, the day count) is the caller's to state. The writers' `decimals` alone
defaults, to None (full precision, which loses nothing): the caller states it where the task states a precision. Arrays in time order, oldest
first; a 2-D array is one series per column. A scalar result is a float, ready for JSON.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import math
from pathlib import Path

import numpy as np

from ._common import choice, out, positive
from .primitives import _guard

__all__ = ["annualize", "compound_factor", "convert_rate", "ewma", "quantile", "read_table", "returns_from_prices",
           "rolling_apply", "std", "write_csv", "write_json", "year_fraction"]

DELIMITERS = (",", "\t", ";", "|")
QUANTITY = ("mean", "variance", "vol", "compound")
RETURN_KINDS = ("simple", "log")
BASES = ("ACT/360", "ACT/365", "ACT/365F", "ACT/ACT ISDA", "30/360", "30E/360")
SEEDS = ("first", "zero", "mean", "adjusted")
NAN_RULES = ("raise", "skip")
QUANTILE_METHODS = ("linear", "lower", "higher", "nearest", "midpoint", "hazen", "weibull", "inverted_cdf")
SIGNS = ("loss_positive", "return_negative")
WINDOW_KINDS = ("rolling", "expanding")
ONE_YEAR = np.timedelta64(1, "Y")


def _finite(name: str, value) -> np.ndarray:
    """`value` as a float array, refused where it is empty or holds a NaN or an infinity (the message counts them)."""
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    bad = int(np.sum(~np.isfinite(array)))
    if bad:
        raise ValueError(f"{name} holds {bad} NaN or infinite values; drop or fill them first")
    return array


def _tail(alpha) -> float:
    """1 - alpha for a confidence level in (0, 1), rounded to 12 places so that 1 - 0.95 is 0.05 exactly."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly inside (0, 1) (the confidence level, e.g. 0.99), not {alpha}")
    return round(1.0 - alpha, 12)


# ------------------------------------------------------------------------------------------------ files


def _plain(value, where: str):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{where} is {number}: JSON has no NaN or infinity; write None (null) or a number")
        return number
    if isinstance(value, dict):
        plain = {}
        for key, item in value.items():
            if isinstance(key, (bool, np.bool_)) or not isinstance(key, (str, int, np.integer)):
                raise ValueError(f"{where} has the key {key!r}; a JSON key is a string")
            text = key if isinstance(key, str) else str(int(key))
            if text in plain:
                raise ValueError(f"{where} has the key {text!r} twice once keys are strings")
            plain[text] = _plain(item, f"{where}[{key!r}]")
        return plain
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):     # tolist() turns datetime64[ns] into integers
            value = np.datetime_as_string(value, unit="auto")
        return _plain(value.tolist(), where)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="auto"))
    raise ValueError(f"{where} is a {type(value).__name__}, which JSON cannot hold; convert it to a number, string, "
                     "list or dict")


def write_json(path, payload, *, decimals=None):
    """Write a payload as JSON of plain numbers; a NaN or an infinity is refused by its key.

    numpy scalars become float, int or bool, arrays and tuples lists, dates ISO strings, int keys strings; a NaN or
    an infinity anywhere raises a ValueError naming its place (payload['risk']['var'] is nan), since json.dumps would
    write the non-JSON token NaN. The parent directory is made; the file is UTF-8 with an indent of 2.

    Where the task states a precision ("rounded to 6 decimals", "fixed-point with exactly 4 decimals"), write exactly
    it: decimals=6 writes every float with six decimals (0.5 as 0.500000), decimals={"price": 2, "vol": 6} the floats
    under those keys at any depth (a list's items under its key), the rest at full precision; integers, booleans and
    strings are written as they are.
    conventions:
        decimals: None (full precision, the default), an int for every float, or a dict key -> int, as the task states
    returns: the plain payload written (dicts, lists, floats, ints, strings, None), a fixed float as the value it reads
        back as
    source: RFC 8259, The JavaScript Object Notation (JSON) Data Interchange Format, section 6 (no NaN or Infinity)
    aliases: results.json, write json, json output, save json
    example:
        import numpy as np, qf
        result = {"var_99": np.float64(0.0231), "weights": np.array([0.4, 0.6]), "n": np.int64(250)}
        qf.write_json("output/results.json", result, decimals=None)
    """
    plain = _plain(payload, "payload")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if decimals is None:
        target.write_text(json.dumps(plain, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return plain
    fixed = _fixed(plain, _places(decimals, "decimals"), None)
    target.write_text(_dumps(fixed) + "\n", encoding="utf-8")
    return _read_back(fixed)


class _Fixed:
    """A float as it is written at its stated decimals."""

    def __init__(self, value: float, places: int):
        self.text = f"{value:.{places}f}"


def _places(decimals, where: str):
    """`decimals` checked: an int of 0 or more, or a dict name -> such an int."""
    def one(value, at: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{at} must be a whole number of decimals, 0 or more, not {value!r}")
        return int(value)
    if isinstance(decimals, dict):
        if not decimals:
            raise ValueError(f"{where} is an empty dict; pass None for full precision")
        named = {str(key): one(value, f"{where}[{key!r}]") for key, value in decimals.items()}
        return named
    return one(decimals, where)


def _fixed(value, places, key):
    """The plain payload with each float `places` names (all where an int; by its key, a list's items by their list's)
    as `_Fixed`."""
    if isinstance(value, dict):
        fixed = {name: _fixed(item, places, name) for name, item in value.items()}
        return fixed
    if isinstance(value, list):
        return [_fixed(item, places, key) for item in value]
    if isinstance(value, float):
        chosen = places if isinstance(places, int) else places.get(key) if key is not None else None
        return _Fixed(value, chosen) if chosen is not None else value
    return value


def _dumps(value, depth: int = 0) -> str:
    """JSON as json.dumps(indent=2) writes it, each `_Fixed` as its text."""
    pad = "  " * depth
    if isinstance(value, _Fixed):
        return value.text
    if isinstance(value, dict) and value:
        return "{\n" + ",\n".join(f"{pad}  {json.dumps(name)}: {_dumps(item, depth + 1)}"
                                   for name, item in value.items()) + f"\n{pad}}}"
    if isinstance(value, list) and value:
        return "[\n" + ",\n".join(f"{pad}  {_dumps(item, depth + 1)}" for item in value) + f"\n{pad}]"
    return json.dumps(value, allow_nan=False)


def _read_back(value):
    if isinstance(value, _Fixed):
        return float(value.text)
    if isinstance(value, dict):
        read = {name: _read_back(item) for name, item in value.items()}
        return read
    if isinstance(value, list):
        return [_read_back(item) for item in value]
    return value


def read_table(path, *, delimiter, missing):
    """Load a CSV or TSV into one array per column, with a dtype and missing-value report.

    A column whose every cell is a number or a missing token becomes float64 with NaN where missing; any other column
    stays text (a numpy array of str). The report says, per column: dtype ('float64' or 'str'), rows, missing (cells
    equal to a token in `missing` after stripping spaces) and unparsed (cells neither missing nor a number: why a
    column stayed text). Header names are stripped of spaces; a repeated name or a ragged row is refused.
    conventions:
        delimiter: ',' (CSV), '\\t' (TSV), ';' or '|'
        missing: the tokens read as missing, e.g. ("", "NA", "NaN", "null"); "" alone for empty cells only
    returns: (columns, report): dicts keyed by column name, in the file's order
    source: RFC 4180, Common Format and MIME Type for Comma-Separated Values (CSV) Files; Python's csv module
    aliases: read csv, load csv, tsv file, csv loader, read tsv
    example:
        import qf
        qf.write_csv("data/p.csv", {"close": [101.5, float("nan")]}, delimiter=",", missing="")
        columns, report = qf.read_table("data/p.csv", delimiter=",", missing=("", "NA"))
    """
    choice("delimiter", delimiter, DELIMITERS)
    tokens = {str(token).strip() for token in ([missing] if isinstance(missing, str) else missing)}
    rows = [row for row in csv.reader(io.StringIO(Path(path).read_text(encoding="utf-8-sig")), delimiter=delimiter)
            if row]
    if not rows:
        raise ValueError(f"{path} has no header row")
    names = [name.strip() for name in rows[0]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} repeats a column name: {names}")
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(names):
            raise ValueError(f"{path} row {number} has {len(row)} cells against {len(names)} names")
    columns, report = {}, {}
    for j, name in enumerate(names):
        cells = [row[j] for row in rows[1:]]
        gone = [cell.strip() in tokens for cell in cells]
        values, unparsed = [], 0
        for cell, absent in zip(cells, gone):
            if absent:
                values.append(math.nan)
                continue
            try:
                values.append(float(cell))
            except ValueError:
                unparsed += 1
        numeric = unparsed == 0
        columns[name] = np.array(values, dtype=float) if numeric else np.array(cells, dtype=str)
        report[name] = {"dtype": "float64" if numeric else "str", "rows": len(cells), "missing": int(sum(gone)),
                        "unparsed": unparsed}
    return columns, report


def _cell(value, where: str, missing, places: int | None = None):
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)):
        if missing is None:
            raise ValueError(f"{where} is missing (NaN or None) and missing=None refuses it; pass missing='' or 'NaN'")
        return missing
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isinf(value):
            raise ValueError(f"{where} is {float(value)}; an infinity is refused")
        return repr(float(value)) if places is None else f"{float(value):.{places}f}"
    if isinstance(value, str):
        return value
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="auto"))
    raise ValueError(f"{where} is a {type(value).__name__}; write numbers, strings or dates")


def write_csv(path, columns, *, delimiter, missing, decimals=None):
    """Write equal-length columns as a CSV or TSV with a header, floats at full precision.

    `columns` maps each header name to a 1-D sequence, in order. A float is written as Python's shortest repr that
    reads back exactly (0.1, 1e-05), an int as an int, a date as ISO; an infinity is refused. The parent directory
    is made; lines end in \\n.

    Where the task states a precision ("rounded to 6 decimal places", "fixed-point with exactly 6 decimals"), write
    exactly it: decimals=6 writes every float cell with six decimals (0.5 as 0.500000, 1e-05 as 0.000010),
    decimals={"price": 2} the named columns' floats, the rest at full precision; integers, booleans, text, dates and
    missing cells are written as they are.
    conventions:
        delimiter: ',' (CSV), '\\t' (TSV), ';' or '|'
        missing: the text written for a NaN or None cell ('' or 'NaN'), or None to refuse one by its column and row
        decimals: None (full precision, the default), an int for every float column, or a dict column -> int, as the
            task states
    returns: the number of data rows written
    source: RFC 4180, Common Format and MIME Type for Comma-Separated Values (CSV) Files; Python's csv module
    aliases: write csv, save csv, csv output, to_csv, tsv output, [write table]
    example:
        import numpy as np, qf
        table = {"date": ["2024-01-02", "2024-01-03"], "signal": np.array([1.0, -1.0])}
        qf.write_csv("output/signals.csv", table, delimiter=",", missing="", decimals=None)
    """
    choice("delimiter", delimiter, DELIMITERS)
    if missing is not None and not isinstance(missing, str):
        raise ValueError("missing must be a string ('' or 'NaN') or None")
    if not isinstance(columns, dict) or not columns:
        raise ValueError("columns must be a non-empty dict of name -> sequence")
    names = list(columns)
    places = None if decimals is None else _places(decimals, "decimals")
    if isinstance(places, dict) and set(places) - {str(name) for name in names}:
        raise ValueError(f"decimals names columns the table does not hold: {sorted(set(places) - set(map(str, names)))}")
    data = []
    for name in names:
        values = np.asarray(columns[name])
        if np.issubdtype(values.dtype, np.datetime64):     # astype(object) turns datetime64[ns] into integers
            values = np.datetime_as_string(values, unit="auto")
        data.append(np.asarray(values, dtype=object).ravel())
    lengths = {len(values) for values in data}
    if len(lengths) != 1:
        raise ValueError(f"columns must have equal lengths, not {dict(zip(names, map(len, data)))}")
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow([str(name) for name in names])
    n = lengths.pop()
    for i in range(n):
        writer.writerow([_cell(values[i], f"columns[{name!r}][{i}]", missing,
                               places.get(str(name)) if isinstance(places, dict) else places)
                         for name, values in zip(names, data)])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(buffer.getvalue(), encoding="utf-8")
    return n


# ------------------------------------------------------------------------------------------------ returns and scale


def annualize(value, *, quantity, periods_per_year):
    """Annualise a per-period mean, variance, vol or compounded return; Sharpe is sharpe_ratio, CAGR cagr.

    With P = periods_per_year: 'mean' value * P (an arithmetic mean of simple or log returns); 'variance' value * P;
    'vol' value * sqrt(P) (the square-root-of-time rule, for i.i.d. returns); 'compound' (1 + value)^P - 1 (a
    per-period simple return compounded over a year). Applies elementwise.

    The arithmetic annual mean return (not a CAGR): annualize(mean(r), quantity='mean', periods_per_year=P).

    Autocorrelated returns: a q-period variance sums the autocovariances, q g_0 + 2 sum_{k<q} (q - k) g_k, g_k at lag k.
    conventions:
        quantity: 'mean', 'variance', 'vol' or 'compound'
        periods_per_year: P, e.g. 252 trading days, 365 calendar days, 52 weeks, 12 months
    returns: a float, or an array for an array
    source: Hull, Options, Futures, and Other Derivatives, "Estimating volatility from historical data"; Bacon,
        Practical Portfolio Performance Measurement and Attribution (2008), annualised return and risk
    aliases: annualise, annualization factor, sqrt(252), [annualized mean return], [arithmetic annual return], [mean times 252]
    example:
        import qf
        daily_vol = 0.0125
        qf.annualize(daily_vol, quantity="vol", periods_per_year=252)
    """
    choice("quantity", quantity, QUANTITY)
    p = float(positive("periods_per_year", periods_per_year))
    value = _finite("value", value)
    if quantity in ("mean", "variance"):
        return out(value * p)
    if quantity == "vol":
        return out(value * math.sqrt(p))
    return out(np.power(1.0 + value, p) - 1.0)


@_guard(price=('prices',))
def returns_from_prices(prices, *, kind):
    """Period returns from a price series: simple p_t / p_{t-1} - 1 or log ln(p_t / p_{t-1}).

    One fewer return than prices; a 2-D array gives one column of returns per column of prices. The two are related
    by log = ln(1 + simple). A NaN price makes its neighbouring returns NaN; a price at or below 0 is refused.
    conventions:
        kind: 'simple' (arithmetic, discrete) or 'log' (continuously compounded)
    returns: an array of length n - 1 (n - 1 rows)
    source: Campbell, Lo and MacKinlay, The Econometrics of Financial Markets (1997), 1.4.1 (simple and
        continuously compounded returns)
    aliases: log returns, simple returns, arithmetic returns, pct_change, continuously compounded returns
    example:
        import qf
        closes = [100.0, 101.5, 99.8, 102.3]
        qf.returns_from_prices(closes, kind="log")
    """
    choice("kind", kind, RETURN_KINDS)
    p = np.asarray(prices, dtype=float)
    if p.ndim not in (1, 2) or p.shape[0] < 2:
        raise ValueError("prices must be 1-D or 2-D with at least two rows")
    if np.any(p[np.isfinite(p)] <= 0) or np.any(np.isinf(p)):
        raise ValueError("prices must be positive and finite (NaN marks a missing price)")
    ratio = p[1:] / p[:-1]
    return np.log(ratio) if kind == "log" else ratio - 1.0


# ------------------------------------------------------------------------------------------------ day counts and rates


def _dates(name: str, value) -> np.ndarray:
    try:
        return np.asarray(value, dtype="datetime64[D]")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be dates (ISO strings 'YYYY-MM-DD', datetime.date or datetime64)") from error


def _month_and_day(days: np.ndarray):
    """Months since 1970-01 and the day of the month: 360 (Y2 - Y1) + 30 (M2 - M1) is 30 times the months between."""
    months = days.astype("datetime64[M]")
    return months.astype(int), (days - months).astype(int) + 1


def _year_length(years: np.ndarray) -> np.ndarray:
    first = years.astype("datetime64[D]")
    return ((years + ONE_YEAR).astype("datetime64[D]") - first).astype(int)


def year_fraction(start, end, *, basis):
    """Year fraction between two dates by a named day count (ACT/360, ACT/365, 30/360, ACT/ACT).

    ACT is the actual number of days. 'ACT/360' days/360; 'ACT/365' and 'ACT/365F' days/365 (Fixed); 'ACT/ACT ISDA'
    the days in each calendar year over that year's length (365 or 366), summed. '30/360' (ISDA 2006 4.16(f),
    Bond Basis) and '30E/360' (4.16(g), Eurobond Basis) are [360 (Y2 - Y1) + 30 (M2 - M1) + (D2 - D1)] / 360 with
    D1 = 30 where it is 31, and D2 = 30 where it is 31 and (Bond Basis) D1 is then 30, or (Eurobond) always. A start
    after the end gives a negative fraction. Arrays of dates broadcast.
    conventions:
        basis: 'ACT/360', 'ACT/365' (= 'ACT/365F'), 'ACT/ACT ISDA', '30/360' (Bond Basis) or '30E/360' (Eurobond)
    returns: a float, or an array for arrays of dates
    source: ISDA 2006 Definitions, section 4.16 (Day Count Fraction); QuantLib's Actual360, Actual365Fixed,
        ActualActual(ISDA) and Thirty360(BondBasis, European)
    aliases: day count, day count convention, ACT/360, ACT/365, 30/360, Actual/360, year fraction
    example:
        import qf
        start, end = "2024-01-15", "2024-07-15"
        qf.year_fraction(start, end, basis="ACT/360")
    """
    choice("basis", basis, BASES)
    s, e = np.broadcast_arrays(_dates("start", start), _dates("end", end))
    if basis.startswith("30"):
        m1, d1 = _month_and_day(s)
        m2, d2 = _month_and_day(e)
        d1 = np.minimum(d1, 30)
        d2 = np.where((d2 == 31) & ((d1 == 30) | (basis == "30E/360")), 30, d2)
        return out((30 * (m2 - m1) + (d2 - d1)) / 360.0)
    days = (e - s).astype(int)
    if basis == "ACT/360":
        return out(days / 360.0)
    if basis in ("ACT/365", "ACT/365F"):
        return out(days / 365.0)
    lo, hi = np.minimum(s, e), np.maximum(s, e)
    ylo, yhi = lo.astype("datetime64[Y]"), hi.astype("datetime64[Y]")
    head = ((ylo + ONE_YEAR).astype("datetime64[D]") - lo).astype(int) / _year_length(ylo)
    tail = (hi - yhi.astype("datetime64[D]")).astype(int) / _year_length(yhi)
    whole = (yhi - ylo).astype(int) - 1
    span = np.where(ylo == yhi, (hi - lo).astype(int) / _year_length(ylo), head + whole + tail)
    return out(np.where(e < s, -span, span))


def _compounding(name: str, value):
    if value in ("simple", "continuous"):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 1:
        return int(value)
    raise ValueError(f"{name} must be 'simple', 'continuous' or a positive int of periods a year (1 annual, 2 "
                     f"semiannual, 4, 12), not {value!r}")


@_guard(rate=('rate',), time=('maturity',))
def compound_factor(rate, *, maturity, compounding):
    """The growth of 1 at `rate` over `maturity` years: 1 + r t, (1 + r/m)^(m t) or e^(r t).

    The discount factor is its reciprocal. `rate` annual; `maturity` t in years (a year fraction from
    year_fraction). Arrays broadcast.

    A forward on an asset worth S grows at the cost of carry: F = S e^{(r + u - y - q) t} with storage u, convenience
    yield y and income yield q (continuous); a long forward struck at K is worth (F - K) e^{-r t} today.
    conventions:
        maturity: t in years, at least 0
        compounding: 'simple', 'continuous', or an int m of periods a year (1 annual, 2 semiannual, 12 monthly)
    returns: a float, or an array for arrays
    source: Hull, Options, Futures, and Other Derivatives, "Measuring interest rates" (compounding frequency);
        QuantLib's InterestRate.compoundFactor
    aliases: compounding, compound factor, discount factor, continuously compounded, semiannual compounding
    example:
        import qf
        r = 0.05
        qf.compound_factor(r, maturity=2.0, compounding=2)
    """
    rule = _compounding("compounding", compounding)
    r, t = np.asarray(rate, dtype=float), np.asarray(maturity, dtype=float)
    if np.any(~(t >= 0)):
        raise ValueError("maturity must be at least 0 (years)")
    if rule == "simple":
        return out(1.0 + r * t)
    if rule == "continuous":
        return out(np.exp(r * t))
    return out(np.power(1.0 + r / rule, rule * t))


@_guard(rate=('rate',), time=('maturity',))
def convert_rate(rate, *, maturity, from_compounding, to_compounding):
    """The equivalent rate under another compounding rule (simple, m-periodic, continuous).

    Equivalent means the same growth over t: compound_factor(r1, t, rule1) = compound_factor(r2, t, rule2), solved
    for r2: simple (F - 1)/t, continuous ln(F)/t, m-periodic m (F^(1/(m t)) - 1). Between continuous and periodic
    rules t cancels; it matters wherever 'simple' is on either side.

    Money-market yields of a bill quoted at a bank discount d with t days left: price 1 - d t / 360; its CD-equivalent
    (money-market) yield d / (1 - d t / 360) on 360 days; its bond-equivalent yield 365 d / (360 - d t).
    conventions:
        maturity: t in years, above 0
        from_compounding: 'simple', 'continuous', or an int m of periods a year (1 annual, 2 semiannual, 12 monthly)
        to_compounding: the same choices
    returns: a float, or an array for arrays
    source: Hull, Options, Futures, and Other Derivatives, "Measuring interest rates", conversion formulas;
        QuantLib's InterestRate.equivalentRate
    aliases: convert compounding, equivalent rate, continuous to annual, annual to continuous, rate conversion
    example:
        import qf
        semiannual = 0.10
        qf.convert_rate(semiannual, maturity=1.0, from_compounding=2, to_compounding="continuous")
    """
    rule = _compounding("to_compounding", to_compounding)
    t = np.asarray(maturity, dtype=float)
    if np.any(~(t > 0)):
        raise ValueError("maturity must be above 0 (years)")
    factor = np.asarray(compound_factor(rate, maturity=t, compounding=from_compounding))
    if np.any(~(factor > 0)):
        raise ValueError("the growth factor must be positive (a simple rate below -1/t has none)")
    if rule == "simple":
        return out((factor - 1.0) / t)
    if rule == "continuous":
        return out(np.log(factor) / t)
    return out(rule * (np.power(factor, 1.0 / (rule * t)) - 1.0))


# ------------------------------------------------------------------------------------------------ statistics


def ewma(x, *, lam, seed, nan):
    """EWMA or EMA s_t = lam s_{t-1} + (1 - lam) x_t, seed (an SMA seed: a number) and NaN rule named; vol: ewma_volatility.

    seed is the state before the first observation: 'first' x_0 itself (so s_0 = x_0: pandas ewm(alpha=1 - lam,
    adjust=False)); 'zero' 0; 'mean' the mean of every finite x; a number that number. 'adjusted' is pandas'
    adjust=True instead, s_t = sum_i lam^i x_{t-i} / sum_i lam^i over the history so far. pandas alpha = 1 - lam;
    span N gives lam = 1 - 2/(N + 1); an SMA-seeded EMA(N) (TA-Lib's) is that lam with seed=mean(x[:N]) run on
    x[N:], its value at index N - 1 the seed and NaN before it. s_t uses x up to t: as a forecast it is for t + 1
    (RiskMetrics: x the squared returns, s_t the variance for t + 1).
    conventions:
        lam: the decay lambda, strictly inside (0, 1) (RiskMetrics 0.94 daily)
        seed: 'first', 'zero', 'mean', 'adjusted' or a number
        nan: 'raise' (refuse any NaN) or 'skip' (a NaN leaves the state as it was, and the output there is that
            state; pandas ignore_na=True)
    returns: an array shaped like x (a 2-D x one column per series)
    source: J.P. Morgan/Reuters, RiskMetrics Technical Document (1996), 5.2.1; Hunter (1986), "The exponentially
        weighted moving average", Journal of Quality Technology 18(4)
    aliases: EWMA, exponentially weighted moving average, exponential moving average, EMA, SMA-seeded EMA, RiskMetrics, ewm
    example:
        import qf
        x = [0.010, -0.020, 0.015, 0.003, -0.007]
        qf.ewma(x, lam=0.94, seed="first", nan="raise")
    """
    choice("nan", nan, NAN_RULES)
    lam = float(lam)
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must lie strictly inside (0, 1), not {lam}")
    data = np.asarray(x, dtype=float)
    if data.ndim not in (1, 2) or data.shape[0] == 0:
        raise ValueError("x must be a non-empty 1-D or 2-D array")
    if np.any(np.isinf(data)) or (nan == "raise" and np.any(np.isnan(data))):
        raise ValueError("x holds NaN or infinite values (nan='skip' steps over a NaN; an infinity is refused)")
    cols = data.reshape(data.shape[0], -1)
    result = np.empty_like(cols)
    if isinstance(seed, str):
        choice("seed", seed, SEEDS)
        if seed == "adjusted":
            num = np.zeros(cols.shape[1])
            den = np.zeros(cols.shape[1])
            for t, row in enumerate(cols):
                seen = ~np.isnan(row)
                num = np.where(seen, lam * num + np.where(seen, row, 0.0), num)
                den = np.where(seen, lam * den + 1.0, den)
                with np.errstate(invalid="ignore", divide="ignore"):
                    result[t] = np.where(den > 0, num / den, np.nan)
            return result.reshape(data.shape)
        if seed == "zero":
            state = np.zeros(cols.shape[1])
        elif seed == "mean":
            with np.errstate(invalid="ignore"):
                state = np.array([np.mean(c[~np.isnan(c)]) if np.any(~np.isnan(c)) else np.nan for c in cols.T])
        else:
            state = np.full(cols.shape[1], np.nan)
    else:
        state = np.broadcast_to(_finite("seed", seed), (cols.shape[1],)).astype(float)
    for t, row in enumerate(cols):
        seen = ~np.isnan(row)
        start = seen & np.isnan(state)
        state = np.where(start, row, np.where(seen, lam * state + (1.0 - lam) * row, state))
        result[t] = state
    return result.reshape(data.shape)


def std(x, *, ddof):
    """A standard deviation, ddof named; a volatility of prices is close_to_close_vol.

    ddof 1 is the unbiased variance's root (pandas .std(), statistics.stdev); ddof 0 the population form (numpy's
    np.std default, statistics.pstdev). A 2-D x is taken by column. NaN is refused, never skipped.
    conventions:
        ddof: 0 (divide by n) or 1 (divide by n - 1), or any int below n
    returns: a float for 1-D x, else an array per column
    source: Bessel's correction; Casella and Berger, Statistical Inference (2002), 5.2 (the sample variance)
    aliases: ddof, standard deviation, sample standard deviation, population standard deviation, Bessel correction
    example:
        import qf
        r = [0.010, -0.020, 0.015, 0.003, -0.007]
        qf.std(r, ddof=1)
    """
    data = _finite("x", x)
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative int (0 or 1)")
    if data.shape[0] <= ddof:
        raise ValueError(f"ddof={ddof} needs more than {ddof} observations")
    return out(np.std(data, axis=0, ddof=int(ddof)))


def quantile(x, q, *, method):
    """The q-quantile of a sample by a named rule (numpy's types); VaR and ES are on the risk shelf.

    With x_(1) <= ... <= x_(n) and the 1-based position h: 'linear' (type 7; numpy's default, Excel PERCENTILE.INC)
    h = (n - 1) q + 1 interpolated; 'lower', 'higher' and 'nearest' the order statistic at floor(h), ceil(h) or the
    nearest (a tie to the even index); 'midpoint' the mean of those two; 'hazen' (type 5) h = n q + 1/2
    interpolated; 'weibull' (type 6, Excel PERCENTILE.EXC) h = (n + 1) q interpolated; 'inverted_cdf' (type 1) the
    smallest x_(k) with k/n >= q, the empirical distribution's inverse. Positions beyond the sample clamp to its
    ends. A 2-D x is taken by column; NaN is refused. Note 1 - 0.95 is 0.050000000000000044 in floating point.
    conventions:
        method: 'linear', 'lower', 'higher', 'nearest', 'midpoint', 'hazen', 'weibull' or 'inverted_cdf'
    returns: a float for 1-D x and a scalar q, else an array
    source: Hyndman and Fan (1996), "Sample quantiles in statistical packages", The American Statistician 50(4);
        numpy.quantile's method argument
    aliases: quantile method, percentile method, sample quantile, empirical quantile, np.percentile
    example:
        import qf
        r = [0.010, -0.020, 0.015, 0.003, -0.007, -0.031, 0.022]
        qf.quantile(r, 0.05, method="lower")
    """
    choice("method", method, QUANTILE_METHODS)
    data = _finite("x", x)
    level = np.asarray(q, dtype=float)
    if np.any(~((level >= 0) & (level <= 1))):
        raise ValueError("q must lie inside [0, 1]")
    return out(np.quantile(data, level, axis=0, method=method))


def rolling_apply(fn, arrays, *, window, kind):
    """Any estimator over rolling or expanding windows (rolling vol, beta, VaR): row t uses the window ending at t.

    Row t is fn(a[t - window + 1 : t + 1], b[...], ...) for 'rolling', the last `window` rows of each array, or fn(a[:
    t + 1], ...) for 'expanding', every row so far; rows t < window - 1, before the first full window, are NaN
    (pandas' rolling(window) and expanding(min_periods=window) with apply(fn, raw=True)). A window holding a NaN is
    passed as it is: fn decides. A function of k numbers gives n x k. Row t uses data to t: as a forecast it is for
    t + 1, lagged one row before it is used (no look-ahead).
    conventions:
        arrays: a list of series, one per argument of fn in its order: [r] for fn(x), [y, x] for fn(y, x); never
            one stacked array for a function of two
        window: the rows of a window ('rolling') or of the first full one ('expanding'), an int of at least 1
        kind: 'rolling' or 'expanding'
    returns: an array of n rows (n x k where fn returns k numbers), NaN before the first full window
    source: pandas, DataFrame.rolling and DataFrame.expanding; Jorion, Value at Risk (2007), the estimation window
    aliases: rolling window, expanding window, moving window, rolling estimate, walk-forward estimate, rolling apply
    example:
        import qf
        y, x = [0.01, -0.02, 0.015, 0.003, -0.007, 0.012, -0.004], [0.008, -0.015, 0.01, 0.0, -0.004, 0.009, -0.002]
        qf.rolling_apply(lambda y, x: qf.ols(y=y, X=x, intercept=True)["coef"][1], [y, x], window=4, kind="rolling")
    """
    choice("kind", kind, WINDOW_KINDS)
    if not callable(fn):
        raise ValueError("fn must be a callable of one window per array")
    if not isinstance(arrays, (list, tuple)) or not arrays:
        raise ValueError("arrays must be a list of equal-length series, each a positional argument of fn ([r] for one)")
    series = [np.asarray(a, dtype=float) for a in arrays]
    if any(a.ndim not in (1, 2) for a in series):
        raise ValueError("arrays must be a list of 1-D or 2-D series, time on the first axis ([r] for one)")
    n = series[0].shape[0]
    if any(a.shape[0] != n for a in series):
        raise ValueError("every array must have the same number of rows")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or not 1 <= window <= n:
        raise ValueError(f"window must be an int from 1 to the series' {n} rows")
    rows = []
    for t in range(window - 1, n):
        start = t - window + 1 if kind == "rolling" else 0
        rows.append(np.asarray(fn(*(a[start:t + 1] for a in series)), dtype=float))
    width = rows[0].shape
    if len(width) > 1 or any(row.shape != width for row in rows):
        raise ValueError("fn must return a number, or one 1-D array of the same length at every window")
    found = np.full((n, *width), np.nan)
    found[window - 1:] = np.stack(rows)
    return found
