"""The harness's own checks of a run's output, from the task's instruction and input files alone (lane derive1; the
wrong-answer trace read runs/lanes/codex-traces-wrong.md, the reasoning read runs/lanes/codex-redesign-reasoning.md
item 2, docs/research/redesign-1016.md). The strongest recurring cause of a wrong clean answer was the identity of a
date, a period or a portfolio's state lost while assembling a runnable calculation; these checks read what the task's
own text and data fix about those identities, never a check's or an oracle's, and say what an output breaks as plain
findings.

Under `AGENTHON_N11_DERIVE=on` (agent/n11/derive.py) each run that exits 0 is read (agent/n11/commands.py
`Workspace.observe`) and the findings of the checks `KEPT` names are one paragraph of the run's observation, after the
conventions card's span and scale lines and before the checks' verdict: advisory, never a refusal, never a count put
as an instruction. Each check is one function with its rule in its docstring; a check that cannot apply to the task
says nothing.

- `dates`: an output's daily series covers the input's calendar exactly inside its own span.
- `cases`: an output table holds each case the instruction or a parameters file lists for its columns once, filled.
- `counts`: a count of observations an output reports is one the input's dates give.
- `schedule`: a schedule's dates sit on the input's calendar where the instruction fixes them by it.
- `coupons`: each period's coupon, principal and discount factor are paired by the same period.
- `nav`: a reported NAV is the one its own weights, the input's returns and its turnover costs give.
- `information`: a row's information date is not later than its use date.

`scripts/derive_replay.py` runs every check over the recorded outputs of the documented wrong answers and of solved
readings; a check is kept (`KEPT`) only where it fires on failures and stays silent on solved runs.

Pure: the standard library alone (pyarrow where a parquet input is read, through agent/n11/conventions.py), nothing
written, nothing named of any unit (rule 8).
"""
from __future__ import annotations

import bisect
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from n11 import conventions

CHECKS = ("dates", "cases", "counts", "schedule", "coupons", "nav", "information")    # every check, in its said order
KEPT = ("cases", "counts", "schedule", "coupons", "nav")    # the checks a run's observation says: those the
#   replay kept, each firing on documented failures and on no solved reading (runs/lanes/derive1-report.md, section 1)
HEAD = "The harness's own checks against the task's calendar and inputs (advisory):"
SAID = 6                    # findings said a run, at most: the harness's bound, never said
NAMED = 3                   # dates, cases or rows one finding names, at most
LINE_CHARS = 260            # a finding's line, at most
MAX_JSON_BYTES = 4 * 1024 * 1024
DAILY_SHARE = 0.9           # an output's dates on the input calendar, at least, for a daily series
MIN_SERIES = 20             # distinct dates a daily series needs before it is read
NUMERIC_SHARE = 0.5         # a column's filled cells that parse as numbers, at least, for a numeric column
RELATIVE = 1e-3             # a coupon's and a price's relative slack
RETURN_SLACK = 1e-6         # a day's return's absolute slack past what the NAV's printed digits allow
MISFITS = 3                 # days a NAV must miss its reconstruction on before it is said
FULL_SHARE = 0.75           # a schedule's period is full at this share of the median period's dates, or more
PERIODS = {"month": lambda d: (d.year, d.month), "week": lambda d: tuple(d.isocalendar()[:2]),
           "quarter": lambda d: (d.year, (d.month - 1) // 3), "year": lambda d: (d.year,)}
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "Δ": "delta", "ε": "epsilon", "η": "eta",
         "θ": "theta", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
         "σ": "sigma", "τ": "tau", "φ": "phi", "ω": "omega"}
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
SET = re.compile(r"(?<![\w])([A-Za-z_]\w{0,30}|[Ͱ-Ͽ])[`*]*\s*(?:∈|=|\bin\b)\s*\{\s*(" + NUMBER
                 + r"(?:\s*,\s*" + NUMBER + r")+)\s*\}")
TUPLES = re.compile(r"\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*\)[\s:*_`]{0,8}((?:\(\s*" + NUMBER
                    + r"(?:\s*,\s*" + NUMBER + r")+\s*\)[\s,;]*(?:and\s+)?)+)")
TUPLE = re.compile(r"\(\s*(" + NUMBER + r"(?:\s*,\s*" + NUMBER + r")+)\s*\)")
SCHEDULED = re.compile(r"\b(last|final|first)\s+(?:trading|business)\s+day\s+of\s+(?:each|every|the|a)\s+"
                       r"(?:calendar\s+)?(month|week|quarter|year)", re.I)
PERIOD_END = re.compile(r"\b(month|quarter|week|year)[- ]end\b", re.I)
NEXT_DAY = re.compile(r"\bnext\s+(?:trading\s+|business\s+)?(?:day|bar|session)\b|\bfollowing\s+(?:trading\s+)?day\b"
                      r"|\bt\s*\+\s*1\b", re.I)
BPS = re.compile(r"(" + NUMBER + r")\s*(?:bps|bp|basis[- ]points?)\b", re.I)
COST_WORDS = re.compile(r"transaction[- ]costs?|trading[- ]costs?|\bturnover\b|\bcosts?\b", re.I)
WEIGHT_WORDS = re.compile(r"\bweights?\b|\bholdings?\b|\bpositions?\b|\bbook\b", re.I)
RETURN_WORDS = re.compile(r"\breturns?\b", re.I)
COUPON_WORDS = re.compile(r"\bcoupons?\b|\bpayments?\b", re.I)
NAV_NAME = re.compile(r"nav|net_?asset|wealth|equity|portfolio_?value|cum(?:ulative)?_?(?:value|wealth)", re.I)
EQUAL_NAME = re.compile(r"equal|(?:^|_)ew(?:_|$)|1_?over_?n", re.I)
COUNT_WORD = re.compile(r"(?:^|_)(?:n|num|number|count|cnt)(?:_|$)", re.I)
COUNT_UNIT = re.compile(r"obs|observation|days?(?:_|$)|rows?(?:_|$)|returns?(?:_|$)|periods?(?:_|$)|months?(?:_|$)"
                        r"|weeks?(?:_|$)|dates?(?:_|$)|quarters?(?:_|$)|years?(?:_|$)", re.I)
EXCEPTION_WORDS = re.compile(r"missing|dropped|removed|duplicat|invalid|skipped|excluded|filled|outlier|nan(?:_|$)", re.I)
CALENDAR_UNITS = ((r"days?(?:_|$)|dates?(?:_|$)|periods?(?:_|$)", "day"), (r"months?(?:_|$)", "month"),
                  (r"weeks?(?:_|$)", "week"), (r"quarters?(?:_|$)", "quarter"), (r"years?(?:_|$)", "year"))
INFO_NAME = re.compile(r"signal|formation|estimat|decision|rebalanc|observ|fixing|as_?of|information|info(?:_|$)|"
                       r"lookback", re.I)
USE_NAME = re.compile(r"activat|active|effective|trade|execut|holding|appl(?:y|ied)|payment|pay_|settle|value_?date|"
                      r"entry|start", re.I)
COUPON_NAME = re.compile(r"coupon|cash_?flow|(?:^|_)cf(?:s|_|$)|payment", re.I)
PRINCIPAL_NAME = re.compile(r"principal|notional|(?:^|_)face|(?:^|_)par(?:_|$)", re.I)
DISCOUNT_NAME = re.compile(r"discount|(?:^|_)df(?:s|_|$)", re.I)
PRICE_NAME = re.compile(r"(?:^|_)(?:dirty_?|clean_?|full_?)?price(?:_|$)|(?:^|_)pv(?:_|$)|present_?value", re.I)
RATE_NAME = re.compile(r"coupon", re.I)
FREQ_NAME = re.compile(r"freq|per_?year|payments_?per", re.I)


# ------------------------------------------------------------------------------------------------ reading


def moment(cell) -> datetime | None:
    """A cell as a date: an ISO date or datetime read directly, else conventions.moment's reading."""
    text = str(cell).strip() if cell is not None else ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.fromisoformat(text[:10])
        except ValueError:
            pass
    return conventions.moment(text) if text else None


def number(cell) -> float | None:
    """A cell as a number: None where it is empty or no number, NaN or infinite where it says so."""
    if isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    text = str(cell).strip() if cell is not None else ""
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def blank(cell) -> bool:
    """A cell that holds no value: empty, NaN or infinite, or a word pandas writes for none."""
    if cell is None:
        return True
    if isinstance(cell, float):
        return not math.isfinite(cell)
    text = str(cell).strip().lower()
    if text in ("", "nan", "none", "null", "na", "n/a", "inf", "-inf", "+inf", "nat"):
        return True
    value = number(text)
    return value is not None and not math.isfinite(value)


def plain(name: str) -> str:
    """A name for matching: Greek letters spelled, lower case, letters and digits alone."""
    return re.sub(r"[^a-z0-9]", "", "".join(GREEK.get(char, char) for char in str(name)).lower())


def singular(name: str) -> str:
    return name[:-3] + "y" if name.endswith("ies") else name[:-2] if name.endswith("ses") else \
        name[:-1] if name.endswith("s") and not name.endswith("ss") else name


@dataclass
class Table:
    """A table: its name, header and cells by column, and its dates by row where a column reads as dates."""
    name: str
    header: list[str]
    cells: list[list[str]]          # by column
    date_column: int | None = None
    row_dates: list[datetime | None] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return max((len(column) for column in self.cells), default=0)

    def cell(self, column: int, row: int) -> str:
        return self.cells[column][row] if row < len(self.cells[column]) else ""

    def numeric(self, column: int) -> bool:
        filled = [cell for cell in self.cells[column] if str(cell).strip()]
        return bool(filled) and sum(number(cell) is not None for cell in filled) >= NUMERIC_SHARE * len(filled)

    def dates(self) -> list[datetime]:
        return sorted({each for each in self.row_dates if each is not None})


def table(path: Path, name: str) -> Table | None:
    """A CSV, TSV or parquet file as a `Table` (conventions.columns), its first column reading as dates by name first
    (conventions.dates_in); None where it is no table."""
    header, column = conventions.columns(path)
    if not header or column is None:
        return None
    cells = [[("" if cell is None else str(cell)) for cell in column(index)] for index in range(len(header))]
    found = Table(name, header, cells)
    order = sorted(range(len(header)), key=lambda k: (not conventions.DATE_NAME.search(header[k]), k))
    for index in order:
        if conventions.dates_in(cells[index], bool(conventions.DATE_NAME.search(header[index]))):
            found.date_column = index
            found.row_dates = [moment(cell) for cell in cells[index]]
            break
    return found


def flat(value, path: str = "") -> list[tuple[str, object]]:
    """A JSON value as (key path, value) pairs, a list of numbers kept whole as one value."""
    if isinstance(value, dict):
        return [pair for key, inner in value.items() for pair in flat(inner, f"{path}.{key}" if path else str(key))]
    if isinstance(value, list) and value and not all(isinstance(each, (int, float)) and not isinstance(each, bool)
                                                     for each in value):
        if all(isinstance(each, dict) for each in value):
            return [(path, value)]
        return [pair for at, inner in enumerate(value) for pair in flat(inner, f"{path}[{at}]")]
    return [(path, value)]


def leaf(path: str) -> str:
    """A key path's last key, its list indexes and the `value` wrapper left out."""
    parts = [part for part in re.sub(r"\[\d+\]", "", path).split(".") if part and part != "value"]
    return parts[-1] if parts else path


@dataclass
class Files:
    """A directory's tables and JSON files, read once."""
    tables: list[Table] = field(default_factory=list)
    jsons: list[tuple[str, object]] = field(default_factory=list)     # (file, parsed)

    def pairs(self) -> list[tuple[str, str, object]]:
        return [(name, path, value) for name, parsed in self.jsons for path, value in flat(parsed)]


def files(where: Path) -> Files:
    """Every table and JSON file under `where` (conventions' table suffixes, `.json`); one that cannot be read is
    left out."""
    where, found = Path(where), Files()
    try:
        paths = sorted(path for path in where.rglob("*") if path.is_file() and not path.is_symlink())
    except OSError:
        return found
    for path in paths:
        name, suffix = path.relative_to(where).as_posix(), path.suffix.lower()
        try:
            if suffix in (*conventions.TABLES, *conventions.PARQUETS) and path.stat().st_size <= MAX_JSON_BYTES * 4:
                read = table(path, name)
                if read is not None:
                    found.tables.append(read)
            elif suffix == ".json" and path.stat().st_size <= MAX_JSON_BYTES:
                found.jsons.append((name, json.loads(path.read_text(encoding="utf-8", errors="replace"))))
        except Exception:  # noqa: BLE001 - a file read wrong is no finding
            continue
    return found


@dataclass
class Task:
    """What the checks read of the task: its instruction and input files (read once a unit), and the calendar."""
    instruction: str
    inputs: Files
    calendar: list[datetime] = field(default_factory=list)        # every input table's dates, sorted
    complete: set = field(default_factory=set)                    # the dates of input rows with every cell filled

    def step_days(self) -> float:
        steps = [(later - earlier).days for earlier, later in zip(self.calendar, self.calendar[1:])]
        return statistics.median(steps) if steps else 0.0


def task(instruction: str, data_dir: Path) -> Task:
    """The task as the checks read it: the instruction's text and the input files under `data_dir`."""
    inputs = files(Path(data_dir))
    calendar, complete = set(), set()
    for each in inputs.tables:
        if each.date_column is None:
            continue
        for row, day in enumerate(each.row_dates):
            if day is None:
                continue
            calendar.add(day)
            if all(not blank(each.cell(column, row)) for column in range(len(each.header))):
                complete.add(day)
    return Task(instruction, inputs, sorted(calendar), complete)


def said_dates(days, limit: int = NAMED) -> str:
    days = sorted(days)
    text = ", ".join(day.strftime("%Y-%m-%d") for day in days[:limit])
    return text + (" among others" if len(days) > limit else "")


def daily(task_: Task, read: Table) -> bool:
    """Whether a table's dates run as a series on the input calendar: `MIN_SERIES` distinct dates or more,
    `DAILY_SHARE` of them on it, at the calendar's step."""
    days = read.dates()
    if len(days) < MIN_SERIES or not task_.calendar:
        return False
    on = set(task_.calendar)
    if sum(day in on for day in days) < DAILY_SHARE * len(days):
        return False
    steps = [(later - earlier).days for earlier, later in zip(days, days[1:])]
    return statistics.median(steps) <= 1.5 * max(task_.step_days(), 1)


# ------------------------------------------------------------------------------------------------ the checks


def dates(task_: Task, out: Files) -> list[str]:
    """An output table whose dates run as a daily series on the input's calendar (`daily`) covers that calendar exactly
    inside its own span: no date off the calendar, no date twice where the table holds one row a date, and no calendar
    date between its first and last date missing where an input table holding that date has its row there complete
    (a row with an empty cell is one the task's cleaning may drop, never asked for). The reference is the input table,
    or the dates every input table shares, that leaves the fewest dates missing."""
    found = []
    tables = [each for each in task_.inputs.tables if each.date_column is not None]
    complete_by = []
    for each in tables:
        rows = {}
        for row, day in enumerate(each.row_dates):
            if day is not None:
                rows[day] = rows.get(day, True) and all(not blank(each.cell(column, row))
                                                        for column in range(len(each.header)))
        complete_by.append({day for day, whole in rows.items() if whole})
    if len(complete_by) > 1:
        complete_by.append(set.intersection(*complete_by))
    for read in out.tables:
        if not daily(task_, read):
            continue
        days = read.dates()
        first, last = days[0], days[-1]
        held = set(days)
        missing = min(({day for day in reference if first <= day <= last} - held for reference in complete_by),
                      key=len, default=set())
        foreign = held - set(task_.calendar)
        counted = Counter(day for day in read.row_dates if day is not None)
        twice = {day for day, times in counted.items() if times > 1} if len(held) >= 0.95 * sum(counted.values()) \
            else set()
        parts = [f"skips input dates inside its span ({said_dates(missing)})" if missing else "",
                 f"holds dates the inputs never hold ({said_dates(foreign)})" if foreign else "",
                 f"holds a date twice ({said_dates(twice)})" if twice else ""]
        parts = [part for part in parts if part]
        if parts:
            found.append(f"`{read.name}`'s daily dates against the input's calendar: " + "; ".join(parts) + ".")
    return found


def stated_cases(task_: Task) -> list[tuple[tuple[str, ...], set[tuple[float, ...]]]]:
    """The cases the instruction and the parameters files list: `NAME ∈ {a, b}` (or `in`, `=`), `(A, B): (a, b),
    (c, d)`, and a parameters file's list of numbers under a key; each as its names (`plain`) and its set of value
    tuples."""
    found = []
    for match in SET.finditer(task_.instruction):
        values = [float(each) for each in re.findall(NUMBER, match.group(2))]
        found.append(((plain(match.group(1)),), {(value,) for value in values}))
    for match in TUPLES.finditer(task_.instruction):
        names = tuple(plain(each) for each in match.group(1).split(","))
        cases = {tuple(float(each) for each in re.findall(NUMBER, inner.group(1)))
                 for inner in TUPLE.finditer(match.group(2))}
        if cases and all(len(case) == len(names) for case in cases) and len(cases) >= 2:
            found.append((names, cases))
    for _, path, value in task_.inputs.pairs():
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(each, (int, float)) for each in value):
            found.append(((plain(leaf(path)),), {(float(each),) for each in value}))
    return found


def matches(column: str, name: str) -> bool:
    mine = plain(column)
    return bool(mine) and mine in (name, singular(name))


def same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def member(value: tuple[float, ...], cases: set[tuple[float, ...]]) -> tuple[float, ...] | None:
    for case in cases:
        if len(case) == len(value) and all(same(a, b) for a, b in zip(case, value)):
            return case
    return None


def said_case(names: tuple[str, ...], case: tuple) -> str:
    return ", ".join(f"{name}={value:g}" for name, value in zip(names, case))


def cases(task_: Task, out: Files) -> list[str]:
    """An output table whose columns the instruction or a parameters file lists cases for (`stated_cases`; of several
    lists under one name, the one holding most of the column's values, and none where the table holds fewer than two
    of its values, or its one value outside it: another grid) holds each case alike: no row at a value outside the
    list, no case held more often than the others (rows keyed by the listed
    columns with the table's text and date columns; a grid over an axis the task does not list repeats every case
    alike), every listed value present where the table holds two or more of a list's values (one value is a slice the
    task asks for), each combination of the lists it holds present, and every numeric cell of a listed case's row
    filled and finite."""
    stated = stated_cases(task_)
    grouped: dict[tuple[str, ...], list[set]] = {}
    for names, listed in stated:
        grouped.setdefault(names, []).append(listed)
    found = []
    for read in out.tables:
        factors, used = [], set()
        for names, lists in grouped.items():
            columns = []
            for name in names:
                at = next((k for k, column in enumerate(read.header) if k not in used and matches(column, name)), None)
                columns.append(at)
            if None in columns or not read.rows or not all(read.numeric(k) for k in columns):
                continue
            held = {value for row in range(read.rows)
                    if None not in (value := tuple(number(read.cell(k, row)) for k in columns))}
            overlap = {id(listed): sum(member(value, listed) is not None for value in held) for listed in lists}
            listed = max(lists, key=lambda each: (overlap[id(each)], -len(each)))
            if overlap[id(listed)] < min(2, len(held)):
                continue        # the column holds another grid than any list of its name: not these cases
            used.update(columns)
            factors.append((tuple(read.header[k] for k in columns), tuple(columns), listed))
        if not factors:
            continue
        keys = [k for k in range(len(read.header)) if k not in used and (not read.numeric(k) or k == read.date_column)]
        values = [k for k in range(len(read.header)) if k not in used and k not in keys and read.numeric(k)]
        outside, seen, rows_of, gaps = [], Counter(), [], []
        for row in range(read.rows):
            case = []
            for names, columns, listed in factors:
                value = tuple(number(read.cell(k, row)) for k in columns)
                hit = member(value, listed) if None not in value else None
                if hit is None:
                    outside.append(said_case(names, tuple(v if v is not None else math.nan for v in value)))
                case.append(hit)
            if None in case:
                continue
            key = (tuple(case), tuple(read.cell(k, row) for k in keys))
            seen[key] += 1
            rows_of.append(tuple(case))
            empty = [read.header[k] for k in values if blank(read.cell(k, row))]
            if empty:
                said = "; ".join(said_case(names, value) for (names, _, _), value in zip(factors, case))
                gaps.append(f"{said}: {', '.join(empty[:NAMED])}")
        fewest = min(seen.values(), default=1)
        twice = [key for key, times in seen.items() if times > fewest]     # a grid over an unlisted axis repeats
        #                                                                    every case alike: only an uneven one is said
        held = [{case[at] for case in rows_of} for at in range(len(factors))]
        expected = [listed if len(held[at]) >= 2 else held[at] for at, (_, _, listed) in enumerate(factors)]
        grid = [()]
        for each in expected:
            grid = [(*prefix, value) for prefix in grid for value in sorted(each)]
        missing = [case for case in grid if case not in set(rows_of)]
        parts = []
        if outside:
            parts.append(f"rows at values the task does not list ({'; '.join(dict.fromkeys(outside[:NAMED]))})")
        if twice:
            parts.append("uneven case repetition (" + "; ".join(
                "; ".join(said_case(names, value) for (names, _, _), value in zip(factors, key[0]))
                for key in twice[:NAMED]) + ")")
        if missing:
            parts.append("listed cases missing (" + " | ".join(
                "; ".join(said_case(names, value) for (names, _, _), value in zip(factors, case))
                for case in missing[:NAMED]) + ")")
        if gaps:
            parts.append(f"a case's cells empty or not finite ({' | '.join(gaps[:NAMED])})")
        if parts:
            found.append(f"`{read.name}` against the cases the task lists for its columns: " + "; ".join(parts) + ".")
    return found


def calendar_most(days: list[datetime], unit: str) -> int:
    """The most a calendar gives at a unit: its dates, or its distinct months, weeks, quarters or years."""
    return len(days) if unit == "day" else len({PERIODS[unit](day) for day in days})


def counts(task_: Task, out: Files) -> list[str]:
    """A whole number an output reports under a key naming a count (`n`, `num`, `number`, `count`) of observations,
    rows, returns, days, dates, periods, months, weeks, quarters or years, and no word of an exception (missing,
    dropped, removed, duplicate, invalid, skipped, excluded, filled, outlier, nan), where it names a calendar unit
    (days, dates, periods, months, weeks, quarters, years) is not more
    than the input's calendar holds at that unit (its dates, or its distinct months, weeks, quarters or years). A
    count under the input's is never said, including an empty sample: filtering, a warm-up, or a subset may take it
    there. Findings never state the input's cardinality."""
    if not task_.calendar:
        return []
    found = []
    for name, path, value in out.pairs():
        key = leaf(path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
            continue
        if not (COUNT_WORD.search(key) and COUNT_UNIT.search(key)) or EXCEPTION_WORDS.search(key):
            continue
        unit = next((each for word, each in CALENDAR_UNITS if re.search(word, key, re.I)), None)
        most = calendar_most(task_.calendar, unit) if unit else None
        if most is not None and value > most:
            found.append(f"`{name}`'s `{key}` exceeds the input's calendar at the reported time scale.")
    return found


def scheduled(instruction: str) -> tuple[str, str] | None:
    """The schedule the instruction fixes by the calendar: (`last` or `first`, the period), or None."""
    match = SCHEDULED.search(instruction)
    if match:
        return ("first" if match.group(1).lower() == "first" else "last"), match.group(2).lower()
    match = PERIOD_END.search(instruction)
    return ("end", match.group(1).lower()) if match else None


def calendars_of(task_: Task) -> list[list[datetime]]:
    """The calendars a task's dates may run on: every input date, the dates of complete input rows, each input
    table's own dates and the dates they all share; each sorted, an empty one left out."""
    tables = [set(each.dates()) for each in task_.inputs.tables if each.date_column is not None]
    found = [task_.calendar, sorted(task_.complete), *(sorted(each) for each in tables)]
    if len(tables) > 1:
        found.append(sorted(set.intersection(*tables)))
    return [each for each in found if each]


def schedule(task_: Task, out: Files) -> list[str]:
    """Where the instruction fixes a schedule by the calendar ("the last (first) trading day of each month, week,
    quarter or year", or a period's end), an output table whose dates fall at most one a period (a rebalance, a
    fixing) has each date at its period's last (first) date on one of the input's calendars (`calendars_of`), or at
    the calendar's own period end where a period's end is said, a date one trading day past or short of it on the
    calendar that holds most of the table's dates named as the off-by-one at the boundary; and no period between
    its first and last date that this calendar holds a full period's dates in (`FULL_SHARE` of the median a period
    holds, or more: a holiday-cut period may be one the task leaves out) left without a date."""
    fixed = scheduled(task_.instruction)
    if fixed is None or not task_.calendar:
        return []
    which, unit = fixed
    period = PERIODS[unit]
    calendars = calendars_of(task_)
    ends = []
    for each in calendars:
        by = {}
        for day in each:
            by.setdefault(period(day), []).append(day)
        ends.append({key: (days[0] if which == "first" else days[-1]) for key, days in by.items()})

    def due(day: datetime | None) -> bool:
        return day is not None and any(day == each.get(period(day)) for each in ends)

    found = []
    for read in out.tables:
        days = read.dates()
        if len(days) < 3 or daily(task_, read):
            continue
        keys = Counter(period(day) for day in days)
        if sum(times > 1 for times in keys.values()) > 0.1 * len(keys):
            continue
        if not any(task_.calendar[0] <= day <= task_.calendar[-1] for day in days):
            continue
        best = max(calendars, key=lambda each: (len(set(days) & set(each)), -len(each)))
        on = set(best)
        off, shifted = [], []
        for day in days:
            if not best[0] <= day <= best[-1]:
                continue
            if due(day) or (which == "end" and day == period_last(day, unit)):
                continue
            at = bisect.bisect_left(best, day)
            before = best[at - 1] if at > 0 else None
            after = best[at + 1 if day in on else at] if at + (day in on) < len(best) else None
            if day in on and due(before):
                shifted.append(f"{day:%Y-%m-%d} is one trading day after a {unit}'s {which} ({before:%Y-%m-%d})")
            elif day in on and due(after):
                shifted.append(f"{day:%Y-%m-%d} is one trading day before its {unit}'s {which} ({after:%Y-%m-%d})")
            else:
                off.append(day)
        sizes = Counter(period(day) for day in best)
        full = FULL_SHARE * statistics.median(sizes.values()) if sizes else 0
        skipped = [key for key in sorted(sizes) if min(keys) <= key <= max(keys) and key not in keys
                   and sizes[key] >= full]
        parts = []
        if shifted:
            parts.append("; ".join(shifted[:NAMED]))
        if off:
            parts.append(f"dates not at their {unit}'s {which} on the input's calendar ({said_dates(off)})")
        if skipped:
            parts.append(f"{unit}s between its first and last date with no date ("
                         + ", ".join("-".join(f"{part:02d}" for part in key) for key in skipped[:NAMED])
                         + (" among others" if len(skipped) > NAMED else "") + ")")
        if parts:
            found.append(f"`{read.name}`'s dates against the task's {which}-of-{unit} schedule: "
                         + "; ".join(parts) + ".")
    return found


def period_last(day: datetime, unit: str) -> datetime:
    """The calendar's own last day of a date's period (a month's, a quarter's, a year's; a week's Sunday)."""
    from datetime import timedelta
    if unit == "week":
        return day + timedelta(days=6 - day.weekday())
    if unit == "year":
        return datetime(day.year, 12, 31)
    month = day.month if unit == "month" else 3 * ((day.month - 1) // 3) + 3
    first_next = datetime(day.year + (month == 12), month % 12 + 1, 1)
    return first_next - timedelta(days=1)


def series_of(out: Files, pattern: re.Pattern) -> list[tuple[str, str, list[float]]]:
    """Every list of numbers the outputs hold under a name `pattern` meets: a JSON list's, or a table's numeric
    column's (file, name, values)."""
    found = []
    for name, path, value in out.pairs():
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(each, (int, float)) for each in value) \
                and pattern.search(leaf(path)):
            found.append((name, leaf(path), [float(each) for each in value]))
    for read in out.tables:
        for k, column in enumerate(read.header):
            if pattern.search(column) and read.numeric(k):
                values = [number(cell) for cell in read.cells[k]]
                if None not in values and len(values) >= 2:
                    found.append((read.name, column, values))
    return found


def rate_of(task_: Task) -> float | None:
    """The coupon rate a period, where the inputs give it: a parameters number under a key naming a coupon and a
    rate, below one, over the payments a year a key naming a frequency gives (one where none)."""
    rate = frequency = None
    for _, path, value in task_.inputs.pairs():
        key = leaf(path).lower()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if RATE_NAME.search(key) and "rate" in key and 0 < value < 1:
            rate = float(value)
        elif FREQ_NAME.search(key) and value >= 1 and value == int(value):
            frequency = int(value)
    return rate / (frequency or 1) if rate is not None else None


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=RELATIVE, abs_tol=1e-6)


def coupons(task_: Task, out: Files) -> list[str]:
    """Where the instruction names coupons or payments and an output holds a coupon (or cash-flow) series beside a
    principal series of the same length, each period's coupon is paired with its own period's principal: the coupon
    over the principal one rate across the periods before the last (the inputs' coupon rate a period where they give
    one), a cash flow the last period carrying its principal too; a coupon that pairs with the next or the previous
    period's principal instead named as that shift, and a zero coupon on a positive principal named. Where a
    discount-factor series of the same length and a price stand beside them, the price is the coupons' sum each at
    its own period's factor."""
    if not COUPON_WORDS.search(task_.instruction):
        return []
    found = []
    flows, principals = series_of(out, COUPON_NAME), series_of(out, PRINCIPAL_NAME)
    factors = series_of(out, DISCOUNT_NAME)
    rate = rate_of(task_)
    for file, name, flow in flows:
        for _, principal_name, principal in principals:
            if len(principal) != len(flow) or principal_name == name or len(flow) < 3:
                continue
            coupons_, held = flow[:-1], principal[:-1]      # the last period may carry its principal: not read

            def fits(pairs) -> list[bool]:      # the task's rate, else the pairing's own median ratio
                pairs = list(pairs)
                ratios = [c / p for c, p in pairs if p]
                ratio = rate if rate is not None else statistics.median(ratios) if ratios else 0.0
                return [bool(ratio) and close(c, ratio * p) for c, p in pairs]
            own = fits(zip(coupons_, held))
            if all(own):
                continue
            shifts = {"the next period's": fits(zip(coupons_, principal[1:])),
                      "the previous period's": fits(zip(coupons_[1:], held))}
            zero = [at for at, (c, p) in enumerate(zip(coupons_, held)) if c == 0 and p > 0]
            said = next((f"each coupon in `{name}` pairs with {which} principal in `{principal_name}`, not its own"
                         for which, fitted in shifts.items() if fitted and all(fitted)), None)
            if said is None and zero:
                said = (f"`{name}` holds a zero coupon in period {zero[0] + 1} on a positive principal in "
                        f"`{principal_name}`")
            if said is None:
                bad = [at + 1 for at, fitted in enumerate(own) if not fitted]
                said = (f"`{name}` against `{principal_name}`: periods {', '.join(map(str, bad[:NAMED]))} hold a coupon "
                        f"off its own period's principal at the task's rate")
            found.append(f"`{file}`: {said}.")
        for factor_file, factor_name, factor in factors:
            if len(factor) != len(flow) or factor_file != file or re.search("coupon", name, re.I):
                continue
            total = sum(c * d for c, d in zip(flow, factor))
            for price_file, path, value in out.pairs():
                if price_file == file and isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and PRICE_NAME.search(leaf(path)) and value \
                        and not math.isclose(total, value, rel_tol=5 * RELATIVE):
                    found.append(f"`{file}`: `{leaf(path)}` is not the sum of `{name}` each at its own period's "
                                 f"`{factor_name}`.")
    return list(dict.fromkeys(found))


def assets(task_: Task, read: Table) -> tuple[Table, list[int], list[int]] | None:
    """The input table whose columns an output table's columns name (two or more), and the matching column places in
    each: (input, the output's, the input's)."""
    best = None
    for each in task_.inputs.tables:
        if each.date_column is None:
            continue
        theirs = {plain(name): k for k, name in enumerate(each.header) if k != each.date_column}
        pairs = [(k, theirs[plain(name)]) for k, name in enumerate(read.header)
                 if k != read.date_column and plain(name) in theirs and read.numeric(k)]
        if len(pairs) >= 2 and (best is None or len(pairs) > len(best[1])):
            best = (each, [mine for mine, _ in pairs], [its for _, its in pairs])
    return best


def returns_of(source: Table, columns: list[int], days: list[datetime]) -> dict[datetime, list[float | None]]:
    """Each date of `days` but the first, its simple returns of `columns` of an input table from the previous date of
    `days`: a level table's ratios less one, its levels carried forward over an empty cell and its last row taken
    where a date holds two; a table of returns as given where every value is under one in magnitude."""
    rows = {}
    for row, day in enumerate(source.row_dates):
        if day is not None:
            rows[day] = row
    levels, carried = {}, [None] * len(columns)
    for day in sorted(rows):
        here = [number(source.cell(k, rows[day])) for k in columns]
        carried = [value if value is not None and math.isfinite(value) else last for value, last in zip(here, carried)]
        levels[day] = (here, list(carried))
    values = [v for here, _ in levels.values() for v in here if v is not None and math.isfinite(v)]
    is_price = bool(re.search(r"price|close|level", source.name, re.I))
    given = not is_price and bool(values) and max(abs(v) for v in values) < 1 \
        and statistics.fmean(abs(v) for v in values) < 0.2
    found = {}
    for earlier, later in zip(days, days[1:]):
        if later not in levels or earlier not in levels:
            continue
        if given:
            found[later] = levels[later][0]
        else:
            found[later] = [(b / a - 1) if a not in (None, 0) and b is not None else None
                            for a, b in zip(levels[earlier][1], levels[later][1])]
    return found


def cost_rate(task_: Task) -> float | None:
    """The cost a unit of turnover pays, where the inputs or the instruction give it: a parameters number under a
    key naming basis points over ten thousand, else the instruction's `N bps`."""
    for _, path, value in task_.inputs.pairs():
        key = leaf(path).lower()
        if ("bps" in key or "basis" in key) and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) / 10_000
    match = BPS.search(task_.instruction)
    return float(match.group(1)) / 10_000 if match else None


def nav(task_: Task, out: Files) -> list[str]:
    """Where the instruction names weights (or holdings), returns and costs, and an output holds a table of target
    weights by rebalance date over columns an input price (or return) table names and a daily table of NAVs: each NAV
    column is reconstructed from its book (the weights table's for a strategy's NAV; equal weights over the same
    columns on the same dates for one whose name says equal weight), the input's returns and, on the day a book
    activates, its turnover (the weights table's turnover column for the strategy, the book's own change for an
    equal-weight one) at the task's cost rate. A book activates on the date after its rebalance date in the daily
    table where the instruction says new weights are active the next day, else on that date or its own, whichever
    fits; before the first activation the NAV stays flat; between activations a book's weights drift with their
    assets' returns, or are held fixed, whichever fits. A NAV whose daily returns miss the reconstruction on
    `MISFITS` days or more is said with its first such dates and the boundary most sit at (a rebalance date, an
    activation date, the day after one, inside a holding period, before the first book). The task must specify simple
    returns and starting in cash; explicit log or percent return units are outside this reconstruction. Source file
    names must identify prices, levels, closes or returns; price sources are levels even below unity."""
    if not (WEIGHT_WORDS.search(task_.instruction) and RETURN_WORDS.search(task_.instruction)
            and COST_WORDS.search(task_.instruction)):
        return []
    # The accounting cannot identify return units or an initial book from column names.
    if not re.search(r"simple\s+(?:daily\s+)?returns", task_.instruction, re.I) \
            or re.search(r"\b(?:log(?:arithmic)?|percent(?:age)?)[- ](?:daily[- ])?returns\b|"
                         r"\breturns?\s+(?:(?:are|given|expressed)\s+)?in\s+(?:percent|%)", task_.instruction, re.I) \
            or not re.search(r"(?:stay|remain|start|begin)[^.\n]{0,30}(?:in\s+)?cash", task_.instruction, re.I):
        return []
    lags = (1,) if NEXT_DAY.search(task_.instruction) else (1, 0)
    rate = cost_rate(task_)
    found = []
    weights = [(read, matched) for read in out.tables if read.date_column is not None and not daily(task_, read)
               and (matched := assets(task_, read)) is not None]
    navs = [read for read in out.tables if daily(task_, read)
            and any(NAV_NAME.search(column) for column in read.header)]
    for book_table, (source, mine, theirs) in weights:
        if not re.search(r"price|close|level|return", source.name, re.I):
            continue
        turnover_at = next((k for k, column in enumerate(book_table.header) if "turnover" in column.lower()), None)
        books = []
        for row, day in enumerate(book_table.row_dates):
            if day is None:
                continue
            target = [number(book_table.cell(k, row)) or 0.0 for k in mine]
            turnover = number(book_table.cell(turnover_at, row)) if turnover_at is not None else None
            books.append((day, target, turnover))
        books.sort(key=lambda each: each[0])
        if not books:
            continue
        for read in navs:
            days = [day for day in read.row_dates if day is not None]
            if len(days) != len(set(days)) or len(days) != read.rows:
                continue
            moves = returns_of(source, theirs, days)
            for k, column in enumerate(read.header):
                if not NAV_NAME.search(column) or not read.numeric(k):
                    continue
                values = [number(read.cell(k, row)) for row in range(read.rows)]
                if None in values:
                    continue
                own = books
                if EQUAL_NAME.search(column):
                    own, previous = [], [0.0] * len(mine)
                    for day, _, _ in books:
                        target = [1 / len(mine)] * len(mine)
                        own.append((day, target, sum(abs(a - b) for a, b in zip(target, previous))))
                        previous = target
                slack = RETURN_SLACK + 2 * 10 ** -digits(read.cells[k]) / max(min(abs(v) for v in values), 1e-9)
                misses = min((misfit(days, values, own, moves, lag, rate, slack) for lag in lags), key=len)
                if len(misses) >= MISFITS:
                    kinds = Counter(where for _, where in misses)
                    found.append(f"`{read.name}`'s `{column}` is not the NAV its book, the input's returns and its "
                                 f"costs give on some days (first {said_dates([day for day, _ in misses])}; most "
                                 f"{kinds.most_common(1)[0][0]}"
                                 + (", the task's books active the next trading day" if lags == (1,) else "") + ").")
    return found


def digits(cells: list[str]) -> int:
    """The most digits after the point a column's cells print (12 for an exponent, at most 12)."""
    most = 0
    for cell in cells:
        text = str(cell).strip().lower()
        if "e" in text:
            return 12
        if "." in text:
            most = max(most, len(text.split(".", 1)[1]))
    return min(most, 12)


def misfit(days: list[datetime], values: list[float], books, moves, lag: int, rate: float | None,
           slack: float) -> list[tuple[datetime, str]]:
    """`nav`'s reading of one NAV column at one activation lag: the days whose return misses the reconstruction, each
    with the boundary it sits at, under the holding rule (drifting or fixed weights) that misses fewer."""
    place = {day: at for at, day in enumerate(days)}
    activations = {}
    for day, target, turnover in books:
        at = (place[day] if day in place else bisect.bisect_right(days, day) - 1) + lag
        if 0 <= at < len(days):
            activations[days[at]] = (target, turnover)
    if not activations:
        return []
    rebalances = {day for day, _, _ in books}
    first_active = min(activations)
    misses = {"drift": [], "fixed": []}
    drift = fixed = None
    for at in range(1, len(days)):
        day, before = days[at], days[at - 1]
        reported = values[at] / values[at - 1] - 1 if values[at - 1] else 0.0
        move = moves.get(day)
        known = move is not None and None not in move
        if day < first_active:
            if abs(reported) > slack:
                for each in misses.values():
                    each.append((day, "before the first book activates"))
            continue
        if day in activations:
            target, turnover = activations[day]
            fixed = target
            drift = None
            if not known:
                continue
            gross = sum(w * r for w, r in zip(target, move))
            drift = [w * (1 + r) / (1 + gross) for w, r in zip(target, move)] if gross != -1 else None
            if rate is not None and turnover is not None:
                cost = turnover * rate
                if abs(reported - (gross - cost)) > slack and abs(reported - ((1 + gross) * (1 - cost) - 1)) > slack:
                    for each in misses.values():
                        each.append((day, "at an activation date"))
            continue
        if fixed is None:
            continue
        if not known:
            drift = None
            continue
        where = ("at a rebalance date, the day before its book activates" if lag and day in rebalances else
                 "the day after an activation date" if before in activations else "inside a holding period")
        if drift is not None:
            expected = sum(w * r for w, r in zip(drift, move))
            if abs(reported - expected) > slack:
                misses["drift"].append((day, where))
            drift = [w * (1 + r) / (1 + expected) for w, r in zip(drift, move)] if expected != -1 else None
        if abs(reported - sum(w * r for w, r in zip(fixed, move))) > slack:
            misses["fixed"].append((day, where))
    return min(misses.values(), key=len)


def information(task_: Task, out: Files) -> list[str]:
    """Where an output table, or a JSON list of records, holds two date columns, one naming the date a quantity's
    information is taken (a signal, formation, estimation, decision, rebalance, observation or fixing date) and one
    naming the date it is used (activation, effective, trade, execution, holding, payment or settlement), every row's
    information date is not later than its use date, and earlier where the instruction says the use is the next
    day."""
    strict = bool(NEXT_DAY.search(task_.instruction))
    found = []
    frames = [(read.name, read.header, [[read.cell(k, row) for k in range(len(read.header))]
                                        for row in range(read.rows)]) for read in out.tables]
    for name, path, value in out.pairs():
        if isinstance(value, list) and value and all(isinstance(each, dict) for each in value):
            header = list(dict.fromkeys(key for each in value for key in each))
            frames.append((f"{name}:{path}", header, [[each.get(key, "") for key in header] for each in value]))
    for name, header, rows in frames:
        info = [k for k, column in enumerate(header) if INFO_NAME.search(column) and conventions.DATE_NAME.search(column)]
        use = [k for k, column in enumerate(header) if USE_NAME.search(column) and k not in info
               and (conventions.DATE_NAME.search(column) or "activ" in column.lower())]
        for i in info:
            for u in use:
                bad = []
                for row in rows:
                    a, b = moment(row[i]), moment(row[u])
                    if a is not None and b is not None and (a > b or (strict and a == b)):
                        bad.append(a)
                if bad:
                    found.append(f"`{name}`: `{header[i]}` is not before `{header[u]}` on some rows "
                                 f"({said_dates(bad)}).")
    return found


FUNCTIONS = {"dates": dates, "cases": cases, "counts": counts, "schedule": schedule, "coupons": coupons, "nav": nav,
             "information": information}


@dataclass(frozen=True)
class Found:
    """One check's finding: the check's name and its line."""
    check: str
    line: str


def run(task_: Task, out: Path, checks=KEPT) -> list[Found]:
    """The findings of `checks` over the output directory `out` (every check's, in `CHECKS` order); a check that
    raises says nothing."""
    read = files(Path(out))
    found = []
    for name in CHECKS:
        if name not in checks:
            continue
        try:
            lines = FUNCTIONS[name](task_, read)
        except Exception:  # noqa: BLE001 - a check that cannot read the output says nothing
            lines = []
        found.extend(Found(name, line if len(line) <= LINE_CHARS else line[:LINE_CHARS - 1] + "…") for line in lines)
    return found


def said(found: list[Found]) -> str:
    """The run's paragraph: `HEAD`, then one line a finding, at most `SAID`; "" where none."""
    if not found:
        return ""
    return "\n".join([HEAD, *(f"- {each.line}" for each in found[:SAID])])
