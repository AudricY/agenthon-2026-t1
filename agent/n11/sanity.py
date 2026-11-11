"""Summarises the output files for the model (the output digest) and flags values no task allows, such as a negative
probability or an expected shortfall below its VaR.

Step: 4, the loop's feedback after a run that exits 0.
Called by: agent/n11/commands.py (`Workspace.sanity`).
Start at: `read`, which returns the digest and the findings in one pass.

The rules in detail
-------------------
The output digest and the sanity checks: what a run that exits 0 left in the output directory, read back for the
model, and the findings that need nothing from the task. Outputs that contradict the instruction or plain arithmetic
(a negative Hill index, a negative gamma, ES below VaR, a negative probability) pass every structural check, so the
values are read too.

- `read(out, named, rules=...)` reads each output file once and returns both:
  - **the digest**: per output file (the named ones first, then by path), its name and size; a CSV's or TSV's rows and
    columns, each numeric column's min and max and its NaN, zero and negative counts, a header-only flag and its text
    columns named; a JSON's scalars at the top level and one level down (`parent.key`), at most `JSON_SCALARS`, the
    rest counted, a list said by its length and, where numeric, its range; any other file its size. Within
    `DIGEST_BYTES` in all, the files past it counted.
  - **the findings**, one line each, at most `FINDINGS` shown and the rest counted, over every CSV, TSV and JSON file:
    a negative value under a name holding a `NONNEGATIVE` word (a price, premium, volatility, VaR, ES, probability,
    gamma, standard error or variance); a value outside [0, 1] under a name holding a `UNIT_RANGE` word (a
    probability, p-value or fraction); ES (es, cvar, expected shortfall) below VaR in magnitude where both stand at the
    same level with the same remaining words (a JSON object's keys, sibling objects' same keys, a table's row); a
    numeric column (a table's, or a key across a JSON list of records) all zero or constant over `MIN_ROWS` values or
    more; and a table holding a header and no row.
- Names are read by their words (`words`): runs of letters and of digits, a run split before a capital that starts a
  word (`impliedVol` is `implied vol`) unless the run is one word whole (`VaR`, `CVaR`, `pValue`); a word matches its
  plural (`prices`, `probabilities`). A name holding an `EXEMPT` word (a change, a difference, a return, an error, a
  log, a contribution, a weight, a ratio, a percentage, a statistic, a count) is never read for its sign or range or
  paired; a column holding an `ECHO` word (an input echoed: a window, a confidence level, a strike) is never read for
  constancy. What a value is near (a search bound the instruction names) is never a check: it would need the task.

The rules (`Ruled`, read in the same pass): three rules that need the output alone, each finding a broken rule with the
name and the value, never a question or a count: `vol_scale` (a value far past any market's, `VOL_MAX`, under a name
holding a `VOL_WORDS` word and no `EXEMPT` one: a percent or an amount where a volatility is a decimal a year),
`siblings` (a JSON object's numeric keys, `SIBLINGS` or more, all at one value that is neither zero nor whole, where
neither the object's name nor every key's holds an `EXEMPT` word: a scalar written where each key holds its own) and
`pct_level` (a name whose level suffix holds a percent word, `es_1pct`, read with the percent word as the level's
(`levelled`): ES below VaR at one level, and the sign and range rules, which `EXEMPT`'s `pct` keeps from such a name
otherwise). Their thresholds are never said.

Pure: the standard library alone, nothing written, no task named; every rule holds for any task of its kind.
"""
from __future__ import annotations

import csv
import functools
import io
import itertools
import json
import math
import re
from pathlib import Path

DIGEST_BYTES = 2_500        # the digest in all
JSON_SCALARS = 40           # a JSON file's scalars shown, at most
FINDINGS = 8                # findings shown, at most; the rest counted
FINDING_CHARS = 200         # a finding's line, at most
MIN_ROWS = 3                # a column's values before it is read for constancy
WHOLE_TOLERANCE = 1e-6      # a constant this close to a whole number, relative, is a whole number
MAX_FILES = 64              # output files read
MAX_BYTES = 8 * 1024 * 1024  # a larger file is listed by its size alone
MAX_ROWS = 200_000          # a table's rows read
MAX_ITEMS = 2_000           # a JSON list's items walked
NAME_CHARS = 48             # a name shown at most this long
UNSHOWN = re.compile(r"[^\x20-\x7e]")    # a name's character shown as `?`
TEXT_CHARS = 40             # a string value shown at most this long
TABLES = {".csv": ",", ".tsv": "\t"}
JSONS = (".json",)
PARQUETS = (".parquet", ".pqt")
NONFINITE = {"nan", "-nan", "+nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"}

NONNEGATIVE = frozenset({"price", "premium", "vol", "volatility", "var", "es", "cvar", "prob", "probability",
                         "gamma", "se", "stderr", "std", "variance"})
NONNEGATIVE_PAIRS = frozenset({("expected", "shortfall")})
UNIT_RANGE = frozenset({"prob", "probability", "pvalue", "fraction"})
UNIT_PAIRS = frozenset({("p", "value")})
VAR_WORDS = frozenset({"var"})
ES_WORDS = frozenset({"es", "cvar"})
ES_PAIRS = frozenset({("expected", "shortfall")})
EXEMPT = frozenset({        # a name holding one is never read for its sign or range, nor paired (ES against VaR)
    "change", "chg", "diff", "difference", "delta", "range", "return", "ret", "pnl", "profit", "error", "err",
    "relerr", "abserr", "rmse", "mse", "mae", "mape", "tol", "tolerance", "residual", "resid", "bias", "skew",
    "skewness", "slope", "coef", "coefficient", "beta", "corr", "correlation", "cov", "covariance", "weight", "loading",
    "exposure", "contribution", "contrib", "component", "marginal", "incremental", "log", "ln", "spread", "basis",
    "drift", "jump", "shock", "innovation", "surprise", "z", "zscore", "score", "tstat", "stat", "statistic", "ratio",
    "excess", "pct", "percent", "percentage", "bps", "odds", "logit", "density", "pdf", "count", "n", "num", "number",
    "breaches", "exceptions", "violations", "hits", "exceedances", "days", "rank", "risk", "carry", "gap", "trend",
    "signal", "momentum", "alpha", "mean", "mu"})
QUIET = frozenset({         # a column holding one may well be all zero or constant: an error, a difference, a count
    "error", "err", "relerr", "abserr", "rmse", "mse", "mae", "mape", "tol", "tolerance", "diff", "difference",
    "residual", "resid", "change", "chg", "count", "n", "num", "number", "rows", "defaults", "breaches", "exceptions",
    "violations", "hits", "exceedances", "trades", "fills", "events", "obs", "flag", "flags"})
ECHO = frozenset({          # an input a column echoes: never read for constancy
    "window", "alpha", "confidence", "level", "horizon", "strike", "maturity", "tenor", "rate", "lam", "lambda", "seed",
    "year", "month", "day", "days", "date", "k", "n", "id", "index", "period", "periods", "freq", "frequency",
    "notional", "spot", "rf", "free", "dividend", "q", "t", "dt", "steps", "paths", "sims", "simulations", "trials",
    "obs", "observations", "size"})
WHOLE = NONNEGATIVE | UNIT_RANGE | EXEMPT | QUIET | ECHO
WORD = re.compile(r"[A-Za-z]+|[0-9]+")
CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")


# ---------------------------------------------------------------------------------------------------- names


@functools.lru_cache(maxsize=4096)
def words(name: str) -> tuple[str, ...]:
    """A name's words, lower-cased (above); read once a name (a table's rows ask it again for each column)."""
    found: list[str] = []
    for run in WORD.findall(str(name)):
        if run.isdigit() or run.lower() in WHOLE:
            found.append(run.lower())
        else:
            found += [part.lower() for part in CAMEL.findall(run)]
    return tuple(found)


def same(word: str, base: str) -> bool:
    """Whether `word` is `base` or its plural (`prices`, `probabilities`)."""
    return word == base or word == base + "s" or (base.endswith("y") and word == base[:-1] + "ies")


def holds(found: tuple[str, ...], vocabulary: frozenset, pairs: frozenset = frozenset()) -> bool:
    """Whether the words hold one of the vocabulary's (its plural too) or one of its pairs in a row."""
    return any(same(word, base) for word in found for base in vocabulary) or \
        any(pair in pairs for pair in zip(found, found[1:]))


def signed(name: str) -> bool:
    """Whether the name says a value that is never negative."""
    found = words(name)
    return holds(found, NONNEGATIVE, NONNEGATIVE_PAIRS) and not holds(found, EXEMPT)


def unit_ranged(name: str) -> bool:
    """Whether the name says a value inside [0, 1]."""
    found = words(name)
    return holds(found, UNIT_RANGE, UNIT_PAIRS) and not holds(found, EXEMPT)


@functools.lru_cache(maxsize=4096)
def risk_base(name: str) -> tuple[str, tuple[str, ...]] | None:
    """("var" or "es", the name's other words) where the name is a VaR's or an ES's, else None."""
    return risk_of(words(name))


def risk_of(found: tuple[str, ...]) -> tuple[str, tuple[str, ...]] | None:
    """`risk_base` over a name's words as read (`words`, or `levelled`)."""
    if holds(found, EXEMPT):
        return None
    for k, word in enumerate(found):
        if word in ES_WORDS:
            return "es", tuple(found[:k] + found[k + 1:])
        if (word, found[k + 1] if k + 1 < len(found) else "") in ES_PAIRS:
            return "es", tuple(found[:k] + found[k + 2:])
    for k, word in enumerate(found):
        if word in VAR_WORDS:
            return "var", tuple(found[:k] + found[k + 1:])
    return None


def steady(name: str) -> bool:
    """Whether a column of this name may be constant: an input it echoes, an error, a difference or a count."""
    return holds(words(name), ECHO | QUIET)


def shown(name: object) -> str:
    text = UNSHOWN.sub("?", str(name))
    return text[:NAME_CHARS] + ("..." if len(text) > NAME_CHARS else "")


def number(value: float) -> str:
    return f"{value:.6g}"


# ---------------------------------------------------------------------------------------------------- values


class Column:
    """A numeric column's (or a JSON key's) values: their count, range, NaN, zero and negative counts and the values
    outside [0, 1]."""

    def __init__(self) -> None:
        self.count = self.nan = self.zero = self.negative = self.outside = 0
        self.low = self.high = None
        self.lowest_outside = None
        self.distinct: set[float] = set()

    def add(self, value: float) -> None:
        self.count += 1
        if not math.isfinite(value):
            self.nan += 1
            return
        self.low = value if self.low is None else min(self.low, value)
        self.high = value if self.high is None else max(self.high, value)
        self.zero += value == 0
        self.negative += value < 0
        if value < 0 or value > 1:
            self.outside += 1
            if self.lowest_outside is None or abs(value - 0.5) > abs(self.lowest_outside - 0.5):
                self.lowest_outside = value
        if len(self.distinct) < 2:
            self.distinct.add(value)

    @property
    def finite(self) -> int:
        return self.count - self.nan

    def said(self) -> str:
        """`[min, max]` and the counts that are not zero."""
        bounds = f"[{number(self.low)}, {number(self.high)}]" if self.low is not None else "[no finite value]"
        counts = [f"{n} {what}" for n, what in ((self.nan, "NaN"), (self.zero, "zero"), (self.negative, "neg")) if n]
        return bounds + (f" {', '.join(counts)}" if counts else "")


def cell(text: str) -> float | None:
    """A table cell as a number (NaN and infinity included), None where it is not one or is empty."""
    text = text.strip()
    if not text:
        return None
    if text.lower() in NONFINITE:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return None


def scalar_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        return math.inf


class Findings:
    """The findings, keyed so a column or key is said once."""

    def __init__(self) -> None:
        self.lines: dict[tuple, str] = {}

    def add(self, key: tuple, line: str) -> None:
        if key not in self.lines:
            self.lines[key] = line[:FINDING_CHARS]

    def bounds(self, file: str, name: str, values: Column, table: bool) -> None:
        """A negative value under a never-negative name; a value outside [0, 1] under a probability's."""
        where = f"{file}: {'column' if table else 'key'} `{shown(name)}`"
        if values.outside and unit_ranged(name):
            self.add((file, name, "range"), f"{where} holds {values.outside} value(s) outside [0, 1], e.g. "
                                             f"{number(values.lowest_outside)} ({article(kind(name))} lies in [0, 1])")
        elif values.negative and signed(name):
            self.add((file, name, "sign"), f"{where} holds {values.negative} negative value(s), min {number(values.low)}"
                                            f" ({article(kind(name))} is never negative)")

    def constant(self, file: str, name: str, values: Column, table: bool) -> None:
        """A column of `MIN_ROWS` finite values or more all zero, or every one the same number that is not whole
        within `WHOLE_TOLERANCE` (a whole one is a count's, an index's or a bound's: an optimiser's value a hair
        inside a whole-number bound), its name no input's echo, error, difference or count."""
        if values.finite < MIN_ROWS or len(values.distinct) != 1 or values.nan or steady(name):
            return
        value = next(iter(values.distinct))
        if value != 0 and abs(value - round(value)) <= WHOLE_TOLERANCE * max(1.0, abs(value)):
            return
        self.add((file, name, "constant"), f"{file}: {'column' if table else 'key'} `{shown(name)}` is "
                                            + ("all zero" if value == 0 else f"constant at {number(value)}")
                                            + f" over {values.finite} values")

    def pairs(self, file: str, where: str, entries: dict[str, float]) -> None:
        """ES below VaR among one level's numbers."""
        found: dict[tuple, dict[str, tuple[str, float]]] = {}
        for name, value in entries.items():
            based = risk_base(name)
            if based is not None and math.isfinite(value):
                found.setdefault(based[1], {})[based[0]] = (name, value)
        for both in found.values():
            if "var" in both and "es" in both:
                (var_name, var), (es_name, es) = both["var"], both["es"]
                if var * es > 0 and abs(es) < abs(var) * (1 - 1e-9):
                    self.add((file, where, var_name, es_name), f"{file}: {where}`{shown(es_name)}` = {number(es)} is "
                                                               f"below `{shown(var_name)}` = {number(var)} (ES is at "
                                                               f"least VaR)")


def article(said: str) -> str:
    """`said` after its article: "an ES", "an expected shortfall", "a price"."""
    return ("an " if said[:1].lower() in "aeiou" or said.startswith("ES") else "a ") + said


LABELS = {"es": "ES", "cvar": "CVaR", "var": "VaR", "se": "standard error", "std": "standard deviation",
          "stderr": "standard error", "vol": "volatility", "prob": "probability", "pvalue": "p-value"}


def kind(name: str) -> str:
    """What the name says, for a finding's line: its first word of `UNIT_RANGE` or `NONNEGATIVE`, else its pair."""
    found = words(name)
    for word in found:
        for base in (*sorted(UNIT_RANGE), *sorted(NONNEGATIVE)):
            if same(word, base):
                return LABELS.get(base, base)
    if holds(found, frozenset(), UNIT_PAIRS):
        return "p-value"
    return "expected shortfall" if holds(found, frozenset(), ES_PAIRS) else "value"


# ----------------------------------------------------------------------------------------------------- files


def table(path: Path, name: str, delimiter: str, found: Findings) -> str:
    """A CSV's or TSV's digest line; its findings into `found`."""
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        header = [column.strip() for column in next(reader, [])]
        rows = [row for row in itertools.islice(reader, MAX_ROWS) if any(part.strip() for part in row)]
    except csv.Error:
        return "not a readable table"
    return frame(name, header, rows, cell, found)


def parquet(path: Path, name: str, found: Findings) -> str:
    """A parquet file's digest line (pyarrow, where it is installed); its findings into `found`."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return "parquet, not read here"
    try:
        read = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001 - a file pyarrow cannot read is said, never raised
        return f"not a readable parquet file ({type(exc).__name__})"
    rows = [list(row.values()) for row in read.slice(0, MAX_ROWS).to_pylist()]
    return frame(name, [str(column) for column in read.column_names], rows, value_cell, found)


def value_cell(value: object) -> float | None:
    """A parquet cell as a number, None where it is empty or not one."""
    return scalar_number(value)


def frame(name: str, header: list[str], rows: list[list], read_cell, found: Findings) -> str:
    """A table's digest line: its rows and columns, each numeric column's range and counts, its text columns named;
    each numeric column's findings, and each row's ES against its VaR."""
    if not header:
        return "no header row"
    if not rows:
        found.add((name, "header-only"), f"{name}: a header and no row ({len(header)} column(s))")
        return f"header only, no row ({len(header)} columns: {', '.join(shown(c) for c in header[:12])})"
    columns = {index: Column() for index in range(len(header))}
    numeric = dict.fromkeys(range(len(header)), True)
    for row in rows:
        level: dict[str, float] = {}
        for index, part in enumerate(row[:len(header)]):
            value = read_cell(part)
            if value is None:
                numeric[index] = numeric[index] and (part is None or not str(part).strip())
                continue
            columns[index].add(value)
            level[header[index]] = value
        found.pairs(name, "", level)
    said, text_columns = [], []
    for index, column in enumerate(header):
        if numeric[index] and columns[index].count:
            found.bounds(name, column, columns[index], True)
            found.constant(name, column, columns[index], True)
            said.append(f"{shown(column)} {columns[index].said()}")
        else:
            text_columns.append(shown(column))
    return (f"{len(rows):,} rows x {len(header)} columns" + (f"; {'; '.join(said)}" if said else "")
            + (f"; text: {', '.join(text_columns)}" if text_columns else ""))


def document(doc: object, name: str, found: Findings) -> None:
    """A JSON document's findings: every number read under its innermost key (a list's items under the list's) for
    its sign and range; each object's numbers paired, ES against VaR, and sibling objects named for a VaR and an ES
    paired key by key; a list of `MIN_ROWS` numbers or more, and each key across a list of records, read for
    constancy."""
    columns: dict[str, Column] = {}

    def visit(value: object, key: str, where: str) -> None:
        if isinstance(value, dict):
            level = {str(inner): item for inner, item in value.items()}
            found.pairs(name, where, {inner: v for inner, item in level.items()
                                      if (v := scalar_number(item)) is not None})
            objects = {inner: item for inner, item in level.items() if isinstance(item, dict)}
            for (one, first), (other, second) in itertools.combinations(objects.items(), 2):
                a, b = risk_base(one), risk_base(other)
                if a and b and a[1] == b[1] and {a[0], b[0]} == {"var", "es"}:
                    for inner in [inner for inner in first if inner in second]:
                        x, y = scalar_number(first[inner]), scalar_number(second[inner])
                        if x is not None and y is not None:
                            found.pairs(name, f"{where}{shown(inner)}: ", {one: x, other: y})
            for inner, item in level.items():
                visit(item, inner, f"{where}{shown(inner)}.")
        elif isinstance(value, list):
            kept = value[:MAX_ITEMS]
            numbers = Column()
            for item in kept:
                if (v := scalar_number(item)) is not None:
                    numbers.add(v)
            if numbers.count == len(kept):
                found.constant(name, key, numbers, False)
            records = [item for item in kept if isinstance(item, dict)]
            if len(records) == len(kept) >= MIN_ROWS:
                for field in dict.fromkeys(inner for record in records for inner in record):
                    across = Column()
                    for record in records:
                        if (v := scalar_number(record.get(field))) is not None:
                            across.add(v)
                    if across.count == len(records):
                        found.constant(name, f"{key}[].{field}" if key else f"[].{field}", across, False)
            for item in kept:
                visit(item, key, where)
        elif (v := scalar_number(value)) is not None:
            columns.setdefault(key, Column()).add(v)

    visit(doc, "", "")
    for key, values in columns.items():
        found.bounds(name, key, values, False)


def scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        v = scalar_number(value)
        return number(v) if v is not None else str(value)
    if isinstance(value, str):
        return json.dumps(value[:TEXT_CHARS] + ("..." if len(value) > TEXT_CHARS else ""))
    if isinstance(value, list):
        column = Column()
        for item in value[:MAX_ITEMS]:
            if (v := scalar_number(item)) is not None:
                column.add(v)
        records = sum(isinstance(item, dict) for item in value[:MAX_ITEMS])
        if column.count and column.count == len(value[:MAX_ITEMS]):
            return f"{len(value)} numbers {column.said()}"
        if records:
            return f"{len(value)} records"
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    return type(value).__name__


def json_line(doc: object) -> str:
    """A JSON document's scalars at the top level and one level down, at most `JSON_SCALARS`."""
    if isinstance(doc, list):
        return f"a list: {scalar(doc)}"
    if not isinstance(doc, dict):
        return f"a {type(doc).__name__}: {scalar(doc)}"
    parts: list[str] = []
    for key, value in doc.items():
        if isinstance(value, dict):
            parts += [f"{shown(key)}.{shown(inner)}={scalar(item)}" for inner, item in value.items()]
        else:
            parts.append(f"{shown(key)}={scalar(value)}")
    more = len(parts) - JSON_SCALARS
    return ", ".join(parts[:JSON_SCALARS]) + (f", and {more} more" if more > 0 else "")


def read(out: Path, named: tuple[str, ...] = (), *, rules: Ruled) -> tuple[str, list[str]]:
    """The output directory's digest and the sanity findings over it (above), the rules' kept in `rules`; ("", [])
    where it holds no file."""
    out = Path(out)
    try:
        files = sorted(path.relative_to(out).as_posix() for path in out.rglob("*")
                       if path.is_file() and not path.is_symlink())
    except OSError:
        files = []
    order = [name for name in named if name in files] + [name for name in files if name not in named]
    found, lines, used, left = BesideRules(rules), [], 0, 0
    for number_, name in enumerate(order):
        path = out / name
        try:
            size = path.stat().st_size
            suffix = path.suffix.lower()
            rules.json = suffix in JSONS
            if number_ >= MAX_FILES or size > MAX_BYTES or size == 0:
                said = "empty" if size == 0 else "not read"
            elif suffix in TABLES:
                said = table(path, name, TABLES[suffix], found)
            elif suffix in PARQUETS:
                said = parquet(path, name, found)
            elif suffix in JSONS:
                try:
                    doc = json.loads(path.read_bytes())
                except (ValueError, RecursionError):
                    said = "not valid JSON"
                else:
                    document(doc, name, found)
                    said = json_line(doc)
            else:
                said = ""
        except Exception as exc:  # noqa: BLE001 - a file that cannot be read is said, never raised into the run
            said, size = f"could not be read ({type(exc).__name__})", locals().get("size", 0)
        line = f"- {shown(name)} ({size:,} B)" + (f": {said}" if said else "")
        room = DIGEST_BYTES - used - 60
        if room < 80:
            left += 1
            continue
        line = clip(line, room)
        lines.append(line)
        used += len(line.encode("utf-8")) + 1
    if left:
        lines.append(f"- and {left} more file(s) not shown")
    return "\n".join(lines), [line for line in found.lines.values()]


def clip(line: str, room: int) -> str:
    data = line.encode("utf-8")
    if len(data) <= room:
        return line
    return data[:room - 4].decode("utf-8", errors="ignore") + " ..."


# ------------------------------------------------------------------------------------------------- the rules

VOL_WORDS = frozenset({"vol", "volatility", "iv", "sigma"})     # `vol_scale`'s names
VOL_MAX = 2.0               # a volatility past this (200% a year) is no market's: a percent or an amount; never said
SIBLINGS = 3                # `siblings`: a JSON object's numeric keys at one value, at the least; never said
LEVEL_WORDS = frozenset({"pct", "percent", "p"})    # `pct_level`: a percent word right after a level's digits


@functools.lru_cache(maxsize=4096)
def levelled(name: str) -> tuple[str, ...]:
    """A name's words with a percent word that follows a level's digits left out (`es_1pct` is `es 1`): the suffix read
    as the level it is, not as a percentage `EXEMPT` keeps from the sign, range and ES rules."""
    found = words(name)
    return tuple(word for k, word in enumerate(found) if not (word in LEVEL_WORDS and k and found[k - 1].isdigit()))


@functools.lru_cache(maxsize=4096)
def level_base(name: str) -> tuple[str, tuple[str, ...]] | None:
    return risk_of(levelled(name))


@functools.lru_cache(maxsize=4096)
def suffixed(name: str) -> bool:
    """Whether the name's words change under `levelled`: a level suffix holding a percent word."""
    return levelled(name) != words(name)


PER_RULE = 2                # a rule's lines a file, at most (as props.PER_PROP a property's)


def listed_names(names: list[str], most: int = 4) -> str:
    """Names in backticks, `a`, `b` and `c`, the first `most` and "and more" after them: never a count."""
    shown_ = [f"`{shown(name)}`" for name in names[:most]]
    if len(names) > most:
        return ", ".join(shown_) + " and more"
    return shown_[0] if len(shown_) == 1 else ", ".join(shown_[:-1]) + " and " + shown_[-1]


class Ruled(Findings):
    """The rules, fed each file's values as `read` reads them (`BesideRules`): their findings, each with its rule and
    what it is known by across runs (its rule, file and name: `rules`)."""

    def __init__(self) -> None:
        super().__init__()
        self.rules: dict[tuple, tuple[str, str]] = {}
        self.json = False           # the file at hand is a JSON document: `siblings` reads its objects

    def rule(self, key: tuple, rule: str, ident: str, line: str) -> None:
        """A rule's line, once a key and at most `PER_RULE` a rule and file."""
        said_ = sum(each == rule and known[0] == key[0] for known, (each, _) in self.rules.items())
        if key not in self.lines and said_ < PER_RULE:
            self.lines[key] = line[:FINDING_CHARS]
            self.rules[key] = (rule, ident)

    def bounds(self, file: str, name: str, values: Column, table: bool) -> None:
        """`vol_scale` over a name's values; `pct_level`'s sign and range under a suffixed name (`levelled`)."""
        where = f"{file}: {'column' if table else 'key'} `{shown(name)}`"
        found = words(name)
        if values.low is not None and holds(found, VOL_WORDS) and not holds(found, EXEMPT):
            peak = values.high if abs(values.high) >= abs(values.low) else values.low
            if abs(peak) > VOL_MAX:
                self.rule((file, name, "vol"), "vol_scale", f"vol_scale|{file}|{name}",
                          f"{where} holds {number(peak)}, far past any market's volatility; a volatility is a "
                          f"decimal a year (0.2 for 20%)")
        level = levelled(name)
        if level == found or holds(level, EXEMPT):
            return
        if values.outside and holds(level, UNIT_RANGE, UNIT_PAIRS):
            self.rule((file, name, "range"), "pct_level", f"pct_level|{file}|{name}",
                      f"{where} holds {number(values.lowest_outside)} ({article(kind(name))} lies in [0, 1])")
        elif values.negative and holds(level, NONNEGATIVE, NONNEGATIVE_PAIRS):
            self.rule((file, name, "sign"), "pct_level", f"pct_level|{file}|{name}",
                      f"{where} holds {number(values.low)} ({article(kind(name))} is never negative)")

    def pairs(self, file: str, where: str, entries: dict[str, float]) -> None:
        """`siblings` over a JSON object's numbers; `pct_level`'s ES below VaR where a suffixed name is paired."""
        if self.json and len(entries) >= SIBLINGS:
            self.siblings(file, where, entries)
        found: dict[tuple, dict[str, tuple[str, float]]] = {}
        for name, value in entries.items():
            based = level_base(name)
            if based is not None and math.isfinite(value):
                found.setdefault(based[1], {})[based[0]] = (name, value)
        for both in found.values():
            if "var" in both and "es" in both:
                (var_name, var), (es_name, es) = both["var"], both["es"]
                if (suffixed(var_name) or suffixed(es_name)) and var * es > 0 and abs(es) < abs(var) * (1 - 1e-9):
                    self.rule((file, where, var_name, es_name), "pct_level", f"pct_level|{file}|{where}{es_name}",
                              f"{file}: {where}`{shown(es_name)}` = {number(es)} is below `{shown(var_name)}` = "
                              f"{number(var)} (ES is at least VaR)")

    def siblings(self, file: str, where: str, entries: dict[str, float]) -> None:
        values = set(entries.values())
        if len(values) != 1:
            return
        value = next(iter(values))
        if not math.isfinite(value) or value == 0:
            return
        owner = where.rstrip(". ").rsplit(".", 1)[-1]
        whole = round(value) != 0 and abs(value - round(value)) <= WHOLE_TOLERANCE * abs(value)  # a count's, a bound's
        if whole or holds(words(owner), EXEMPT) or \
                all(holds(words(name), EXEMPT) for name in entries):
            return
        self.rule((file, where, "siblings"), "siblings", f"siblings|{file}|{where}",
                  f"{file}: " + (f"`{shown(owner)}`'s keys" if owner else "the top-level keys")
                  + f" {listed_names(list(entries))} all hold {number(value)}, one figure where each key holds its own")


class BesideRules(Findings):
    """The plain findings, each file's parsed values handed to the rules (`Ruled`) too."""

    def __init__(self, rules: Ruled) -> None:
        super().__init__()
        self.rules = rules

    def bounds(self, file: str, name: str, values: Column, table: bool) -> None:
        super().bounds(file, name, values, table)
        try:
            self.rules.bounds(file, name, values, table)
        except Exception:  # noqa: BLE001 - a rule reader never changes the ordinary observation
            pass

    def pairs(self, file: str, where: str, entries: dict[str, float]) -> None:
        super().pairs(file, where, entries)
        try:
            self.rules.pairs(file, where, entries)
        except Exception:  # noqa: BLE001 - a rule reader never changes the ordinary observation
            pass


def said(findings: list[str]) -> str:
    """The findings as an observation's lines under `sanity:`, at most `FINDINGS` and the rest counted."""
    if not findings:
        return ""
    more = len(findings) - FINDINGS
    return "sanity:\n" + "\n".join(f"- {line}" for line in findings[:FINDINGS]) + \
        (f"\n- and {more} more" if more > 0 else "")
