"""Two more findings after a run that exits 0: an output table whose dates stop short of the input's, and output
values in the wrong scale (a percent where the instruction says a decimal).

Step: 4, the loop's feedback. Neither finding blocks a finish.
Called by: agent/n11/commands.py (`Workspace.against`).
Calls: agent/n11/props.py for reading the instruction's sentences, agent/n11/commands.py for reading tables.
Start at: `span` and `scale`.

The rules in detail
-------------------
Two findings that read a run's output against the instruction. Many clean-but-wrong outputs come from a
convention or shape the instruction fixed in words the model read once and did not carry into its code: a scale, a
window, a count, a row order.

- **The span finding** (`span`): an output table whose dates run as a series (`Series.regular`: one spacing, no wide
  gap), against the dates of the task's input tables (`Dated`): where it starts after the span the middle input
  table holds (`Dated.held`) by more than the longest window the instruction names (`windows`), two steps and a
  weekend, or ends before it by more than two steps and a weekend (and the longest horizon, a length beside a word
  that looks ahead of a row: `LEAD`), the table still covering at least `COVERS` of that span (a few dates a condition
  picks are no series cut short); or where, overlapping them, it runs past every date the input tables hold (in any
  column: `Dated.joint`) by more than a step. One line, its file named, no date said.
- **The scale finding** (`scale`): an output name the instruction says is in percent, in basis points or a decimal
  (`bound_scales`: the scale word bound to the name nearest before it in its sentence, or the list it ends), whose
  values read in another scale: all under one in magnitude where percent or basis points are said, all at least one
  where a decimal is said. One line naming the names, never a value.
Both findings are said after a run that exits 0 (agent/n11/commands.py `Workspace.against`), beside the instruction's
findings and apart from them: neither blocks a finish nor makes a run unclean.

Pure: the standard library alone (pyarrow where a parquet file is read), nothing written, nothing named of any task.
It imports agent/n11/commands.py, so commands.py imports this module inside the method that uses it.
"""
from __future__ import annotations

import csv
import io
import itertools
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from n11 import commands, props

MAX_ROWS = 200_000          # a table's rows read for its dates
SNIFF_ROWS = 20             # a column's first cells read to tell a date column
DATED_SHARE = 0.9           # the share of those cells that must read as dates
MIN_DATES = 6               # distinct dates a table needs before its span is read
GAPS, GAP_DAYS = 5, timedelta(days=3)    # an output series' widest gap, at most: its median steps and days
COVERS = 0.5                # the share of the inputs' common span an output table's must cover to be read as short
TABLES = (".csv", ".tsv", ".txt", ".dat")
PARQUETS = (".parquet", ".pqt")
DATE_NAME = re.compile(r"date|time|month|period|day|week|year|asof|as_of|stamp|^dt$", re.I)
YEAR_MONTH = re.compile(r"(\d{4})[-/](\d{1,2})")
COMPACT_DAY = re.compile(r"(19|20)\d{6}")                # 20200131, read where the column's name says a date
SAID_NAMES = 3              # the names the scale finding says, at most; never said
REGULAR = 0.8               # the share of steps its spacing's class must hold, else the spacing is irregular
IRREGULAR_DATES = 6         # a series' steps said irregular only over at least this many dates
DAY = 86_400                # seconds
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# (label, the steps in days it holds, the least and the most: a calendar's, give or take a business day's roll)
CLASSES = (("weekly", 6, 8), ("monthly", 27, 34), ("quarterly", 85, 96), ("half-yearly", 175, 190),
           ("yearly", 358, 372))
MONTH_FORMS = ("%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d-%b-%y", "%d.%m.%Y")   # dates beside ISO's and slashed ones


# ----------------------------------------------------------------------------------------------- the dates


def slashed(cell: str) -> tuple[int, int, int] | None:
    """A slashed date's fields (first, second, year), a two-digit year read in the 2000s; None where it is none."""
    parts = cell.split("/")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    first, second, year = (int(part) for part in parts)
    return first, second, year + (2000 if year < 100 else 0)


def when(cell: str) -> tuple | None:
    """A date cell's place on the calendar, comparable with another's: an ISO date or time, a slashed date (day
    first where its first field passes 12, else month first), or one with its month named (`MONTH_FORMS`); None
    where it reads as none, a bare number among them."""
    if commands.number(cell) is not None:
        return None
    try:
        return tuple(datetime.fromisoformat(cell.replace("Z", "+00:00")).timetuple()[:6])
    except ValueError:
        pass
    fields = slashed(cell)
    if fields is not None:
        first, second, year = fields
        return (year, second, first) if first > 12 else (year, first, second)
    for form in MONTH_FORMS:
        try:
            moment = datetime.strptime(cell, form)
        except ValueError:
            continue
        return tuple(moment.timetuple()[:6])
    return None


def moment_of(fields: tuple) -> datetime:
    return datetime(*fields[:3], *(fields[3:6] or (0, 0, 0)))


def said_step(seconds: int) -> str:
    """A step between two dates in the largest unit that says it as a whole number or nearly: days, hours, minutes or
    seconds."""
    for unit, scale in (("day", 86_400), ("hour", 3_600), ("minute", 60), ("second", 1)):
        if seconds >= scale or unit == "second":
            value = round(seconds / scale, 2)
            value = int(value) if value == int(value) else value
            return f"{value} {unit}{'s' if value != 1 else ''}"


def spacing(moments: list[datetime]) -> str:
    """Sorted distinct dates' spacing, read over the steps between them in whole seconds: `business days` (weekdays
    alone, a weekend skipped, the weekdays absent counted), `daily` (the days absent counted), a calendar class
    (`CLASSES`), `intraday` (its most common step), each where at least `REGULAR` of the steps hold it, else
    `irregular` with the steps' range and the most common where there are `IRREGULAR_DATES` dates or more; "" with
    fewer than three dates."""
    if len(moments) < 3:
        return ""
    steps = [round((later - earlier).total_seconds()) for earlier, later in zip(moments, moments[1:])]
    common = Counter(steps).most_common(1)[0][0]
    days = sorted({moment.date() for moment in moments})
    label, regular = "", 0
    if common < DAY:
        label = f"intraday, most steps {said_step(common)}"
        regular = steps.count(common)
    elif common == DAY:
        weekend = [step == 3 * DAY and earlier.weekday() == 4 for earlier, step in zip(moments, steps)]
        if any(weekend) and all(day.weekday() < 5 for day in days):
            span = (days[-1] - days[0]).days + 1
            expected = sum(date.fromordinal(days[0].toordinal() + k).weekday() < 5 for k in range(span))
            absent = expected - len(days)
            label = "business days: weekdays alone" + (f", {absent:,} weekday{'s' if absent != 1 else ''} absent"
                                                       if absent else ", every weekday")
            regular = steps.count(DAY) + sum(weekend)
        else:
            absent = (days[-1] - days[0]).days + 1 - len(days)
            label = "daily" + (f", {absent:,} day{'s' if absent != 1 else ''} absent" if absent else
                               ", every calendar day")
            regular = steps.count(DAY)
    else:
        for name, least, most in CLASSES:
            if least * DAY <= common <= most * DAY:
                label = name + calendar_note(name, moments)
                regular = sum(least * DAY <= step <= most * DAY for step in steps)
                break
    if label and regular >= REGULAR * len(steps):
        return label
    if len(moments) < IRREGULAR_DATES:
        return ""
    return (f"irregular: steps from {said_step(min(steps))} to {said_step(max(steps))}, most often "
            f"{said_step(common)}")


def calendar_note(name: str, moments: list[datetime]) -> str:
    """A weekly series' one weekday; a monthly, quarterly or yearly one's month ends or month starts."""
    if name == "weekly":
        weekdays = {moment.weekday() for moment in moments}
        return f", on {WEEKDAYS[weekdays.pop()]}s" if len(weekdays) == 1 else ""
    if all((moment.date().toordinal() + 1) == date(moment.year + moment.month // 12, moment.month % 12 + 1,
                                                    1).toordinal() for moment in moments):
        return ", at month ends"
    if all(moment.day == 1 for moment in moments):
        return ", on the first of the month"
    return ""


def moment(cell: object, named: bool = False) -> datetime | None:
    """A cell as a date: a date or datetime value; an ISO date or time, a slashed date, a month by name
    (`when`); a year and month (`2020-01`); an eight-digit day (`20200131`) where `named` (the column's name
    says a date); None where it reads as none."""
    if isinstance(cell, datetime):
        return cell.replace(tzinfo=None)
    if isinstance(cell, date):
        return datetime(cell.year, cell.month, cell.day)
    text = str(cell).strip() if cell is not None else ""
    if not text:
        return None
    try:
        found = YEAR_MONTH.fullmatch(text)
        if found:
            return datetime(int(found.group(1)), int(found.group(2)), 1)
        if named and COMPACT_DAY.fullmatch(text):
            return datetime(int(text[:4]), int(text[4:6]), int(text[6:]))
        fields = when(text)
        return moment_of(fields) if fields is not None else None
    except (ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class Series:
    """A table's dates: its file, the date column, its distinct dates sorted, and their median step."""
    file: str
    column: str
    dates: tuple[datetime, ...]

    @property
    def step(self) -> timedelta:
        steps = [later - earlier for earlier, later in zip(self.dates, self.dates[1:])]
        return statistics.median(steps) if steps else timedelta(0)

    @property
    def regular(self) -> bool:
        """Its dates at one spacing (`spacing`: business days, daily, a calendar class, intraday) with no gap
        past `GAPS` steps and `GAP_DAYS` (a weekend and a holiday): a series, not a list of dates a condition picks."""
        spaced = spacing(list(self.dates))
        gap = max(later - earlier for earlier, later in zip(self.dates, self.dates[1:]))
        return bool(spaced) and not spaced.startswith("irregular") and gap <= GAPS * self.step + GAP_DAYS


def series(path: Path, name: str, read: tuple | None = None) -> Series | None:
    """A table's first date column (by its name first, then by place) read through `MAX_ROWS` rows, as its distinct
    dates; None where it has none, or fewer than `MIN_DATES`. `read`: the table's `columns`, where already read."""
    header, column = read if read is not None else columns(path)
    order = sorted(range(len(header)), key=lambda k: (not DATE_NAME.search(header[k]), k))
    for index in order:
        dates = dates_in(column(index), bool(DATE_NAME.search(header[index])))
        if dates is None:
            continue
        return Series(name, header[index], tuple(dates)) if len(dates) >= MIN_DATES else None
    return None


def dates_in(cells: list, named: bool) -> list[datetime] | None:
    """A column's distinct dates, sorted, where `DATED_SHARE` of its first `SNIFF_ROWS` filled cells read as dates;
    None where they do not."""
    sniffed = [cell for cell in cells[:SNIFF_ROWS * 5] if cell is not None and str(cell).strip()][:SNIFF_ROWS]
    if not sniffed or sum(moment(cell, named) is not None for cell in sniffed) < DATED_SHARE * len(sniffed):
        return None
    return sorted({found for cell in set(cells) if (found := moment(cell, named)) is not None})


def reach(read: tuple) -> tuple[datetime, datetime] | None:
    """The earliest and the latest date in any column of a table (its `columns`) that reads as dates: an output may
    run to a date the inputs hold beside their series' (a resolution, an expiry); None where none."""
    header, column = read
    found = [dates for index, name in enumerate(header)
             if (dates := dates_in(column(index), bool(DATE_NAME.search(name))))]
    return (min(each[0] for each in found), max(each[-1] for each in found)) if found else None


def columns(path: Path) -> tuple[list[str], object]:
    """A table's header and a reader of one column's cells by its place (`MAX_ROWS` rows at most): a CSV, TSV or
    text table's (commands.table_like), a parquet file's (pyarrow, where installed); ([], None) for any other."""
    suffix = path.suffix.lower()
    if suffix in PARQUETS:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return [], None
        read = pq.read_table(path)
        return [str(name) for name in read.column_names], lambda k: read.column(k).to_pylist()[:MAX_ROWS]
    if suffix not in TABLES:
        return [], None
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    if not commands.table_like(text[:65_536]):
        return [], None
    delimiter = "\t" if suffix == ".tsv" else commands.sniff(text[:65_536])
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    header = [column.strip() for column in next(reader, [])]
    rows = [row for row in itertools.islice(reader, MAX_ROWS) if any(part.strip() for part in row)]
    return header, lambda k: [row[k] if k < len(row) else "" for row in rows]


@dataclass
class Dated:
    """The task's input tables' dates (`dated`), read once a unit: each table's `Series`, and each table's `reach`."""
    tables: list[Series] = field(default_factory=list)
    reaches: list[tuple[datetime, datetime]] = field(default_factory=list)

    def held(self) -> tuple[datetime, datetime] | None:
        """The span the middle input table holds at each end: the median of the first dates (the later of the two
        middle ones) to the median of the last (the earlier); one table's own span, two tables' shared one; None
        where it is empty."""
        if not self.tables:
            return None
        first = statistics.median_high([each.dates[0] for each in self.tables])
        last = statistics.median_low([each.dates[-1] for each in self.tables])
        return (first, last) if first < last else None

    def joint(self) -> tuple[datetime, datetime] | None:
        """The span any input table reaches, in its series or any column whose name says a date: the earliest
        first date to the latest last."""
        ends = [(each.dates[0], each.dates[-1]) for each in self.tables] + self.reaches
        if not ends:
            return None
        return min(first for first, _ in ends), max(last for _, last in ends)

    def step(self) -> timedelta:
        return max((each.step for each in self.tables), default=timedelta(0))


def dated(data_dir: Path) -> Dated:
    """Every input table's dates under the unit's data directory (a table that cannot be read holds none)."""
    data_dir = Path(data_dir)
    found = Dated()
    for path in sorted(data_dir.rglob("*")) if data_dir.is_dir() else []:
        if path.is_file() and path.suffix.lower() in (*TABLES, *PARQUETS):
            try:
                table = columns(path)
                read, reached = series(path, path.relative_to(data_dir).as_posix(), table), reach(table)
            except Exception:  # noqa: BLE001 - an input read wrong holds no dates here
                read, reached = None, None
            if read is not None:
                found.tables.append(read)
            if reached is not None:
                found.reaches.append(reached)
    return found


# ----------------------------------------------------------------------------------------- the span finding


WINDOW = re.compile(r"(?<![\w.])(\d{1,4})(?:st|nd|rd|th)?\s*[- ]?\s*(trading[- ]days?|business[- ]days?|calendar[- ]"
                    r"(?:days?|weeks?|months?|quarters?|years?)|days?|weeks?|months?|quarters?|years?|observations?|"
                    r"obs|periods?|bars?|rows?)\b", re.I)    # an ordinal too: "after the 120th calendar month"
WINDOW_WORD = re.compile(r"window|rolling|trailing|look-?back|formation|estimation|warm-?up|burn-?in|lag|skip|moving|"
                         r"span|history|momentum|horizon|holding|ewm|ema\b|sma\b|z-?score|min_periods|last|first|"
                         r"recent|previous|prior", re.I)
WINDOW_NEAR = 40            # characters around a length where a window's word must stand
ANNUAL = re.compile(r"annuali[sz]\w*\W+(?:\w+\W+){0,2}$|sqrt\W*$|(?:√|×)\s*$", re.I)    # a length that annualises,
YEARLY = re.compile(r"^\W*(?:per|a|each|in a)\s+(?:year|annum)\b", re.I)                 # no window
DAYS = {"trading": 365 / 252, "business": 365 / 252, "calendar": 1, "day": 365 / 252, "week": 7, "month": 31,
        "quarter": 92, "year": 366}     # a window's unit in calendar days, at its longest reading


def windows(instruction: str, step: timedelta, words: re.Pattern = WINDOW_WORD) -> timedelta:
    """The longest window the instruction names: a length and its unit with a window's word (`words`) near it
    (`WINDOW_NEAR`), in calendar days at the unit's longest reading; a count of observations, periods, bars or rows at
    `step`, the data's; a length that annualises (`ANNUAL` before it, `YEARLY` after) none; zero where it names
    none. With `LEAD` for `words`: the longest horizon, a length beside a word that looks ahead."""
    longest = timedelta(0)
    for found in WINDOW.finditer(instruction):
        before = instruction[max(0, found.start() - WINDOW_NEAR):found.start()]
        after = instruction[found.end():found.end() + WINDOW_NEAR]
        if not words.search(before + found.group(0) + after) or ANNUAL.search(before) or YEARLY.match(after):
            continue
        count, unit = int(found.group(1)), re.sub(r"^calendar[- ](?!day)", "", found.group(2).lower())
        scale = next((days for word, days in DAYS.items() if unit.startswith(word)), None)
        length = timedelta(days=count * scale) if scale is not None else count * step
        longest = max(longest, length)
    return longest


SPAN_INSIDE = ("the dates in {file} span less than the data's (the {ends} rows are missing): is a window, a warm-up or "
               "a merge dropping them, and does the instruction want them?")
LEAD = re.compile(r"\bt ?\+ ?\d|\bnext[- ](?:day|week|month|period|bar|quarter|year)|\blead|\bforward[- ]"
                  r"(?:looking|returns?)|\bhorizon|\bahead\b", re.I)    # outputs that look past a row: its last rows
SPAN_PAST = ("the dates in {file} run past the data's (rows {where}): is a date shifted, and does the instruction "
             "want them?")


def span(instruction: str, out: Path, data: Dated) -> str:
    """The span finding over the output's tables, the first that meets it, in name order; "" where none does."""
    held, joint = data.held(), data.joint()
    if joint is None:
        return ""
    window = windows(instruction, data.step())
    lead = windows(instruction, data.step(), LEAD)        # the last rows a horizon drops
    for path in output_tables(out):
        try:
            read = series(path, path.relative_to(out).as_posix())
        except Exception:  # noqa: BLE001 - an output read wrong is no finding
            read = None
        if read is None or not read.regular:
            continue
        first, last = read.dates[0], read.dates[-1]
        slack = data.step() + timedelta(days=1)
        past = [where for where, beyond in (("before its first date", joint[0] - first > slack and last >= joint[0]),
                                            ("after its last date", last - joint[1] > slack and first <= joint[1]))
                if beyond]      # a series wholly past the data is a schedule or a forecast it asks for, not a shift
        if past:
            return SPAN_PAST.format(file=f"`{read.file}`", where=" and ".join(past))
        if held is None or min(last, held[1]) - max(first, held[0]) < COVERS * (held[1] - held[0]):
            continue
        steps = 2 * max(data.step(), read.step) + GAP_DAYS     # a returns series' first row across a weekend
        ends = [end for end, short in (("first", first - held[0] > window + steps),
                                       ("last", held[1] - last > lead + steps)) if short]
        if ends:
            return SPAN_INSIDE.format(file=f"`{read.file}`", ends=" and the ".join(ends))
    return ""


def output_tables(out: Path) -> list[Path]:
    out = Path(out)
    try:
        return sorted(path for path in out.rglob("*") if path.is_file() and not path.is_symlink()
                      and path.suffix.lower() in (*TABLES, *PARQUETS) and path.stat().st_size <= props.MAX_BYTES)
    except OSError:
        return []


# ---------------------------------------------------------------------------------------- the scale finding


PERCENT_SAID = re.compile(r"\bin (?:\*\*)?(?:per ?cent(?:ages?)?|percentage points?|%)|\bas (?:an? )?(?:\*\*)?"
                          r"percentages?\b|\b(?:are|is) (?:\*\*)?(?:per ?cent(?:ages?)?|percentage points?)\b|"
                          r"\(\s*(?:in\s+)?%\s*\)", re.I)
BPS_SAID = re.compile(r"\bin (?:\*\*)?(?:basis[- ]points?|bps|bp)\b|\b(?:are|is|as) (?:\*\*)?basis[- ]points?\b|"
                      r"\(\s*(?:in\s+)?bps?\s*\)", re.I)
DECIMAL_SAID = re.compile(r"\bin (?:\*\*)?decimals?\b(?!\s+places?)|\b(?:as|are|is) (?:an? )?(?:\*\*)?(?:raw )?"
                          r"decimals?\b(?!\s+places?)|\bdecimal (?:form|fractions?|units?|ratios?|terms)\b|"
                          r"\bnot (?:as )?(?:a )?percentages?\b|\bnever percentages?\b|\bas (?:a )?fractions?\b",
                          re.I)
NEGATED = re.compile(r"\b(?:not|never|rather than|instead of)\s+(?:as\s+|in\s+)?(?:an?\s+)?(?:\*\*)?$", re.I)
SUFFIXED = re.compile(r"\b(?:ending|ends|end)\s+(?:in|with)\s+`([A-Za-z0-9_]+)`|\bwith\s+`(_[A-Za-z0-9_]+)`", re.I)
SCALES = (("percent", PERCENT_SAID), ("basis points", BPS_SAID), ("decimal", DECIMAL_SAID))
LISTED = re.compile(r"^[\s,]*(?:and|or|,)?[\s,]*$")


def bound_scales(instruction: str) -> dict[str, set[str]]:
    """Each name the instruction binds to a scale it says (`SCALES`), {name: {scale, ...}}: in each sentence of each
    statement (props.segment), a scale word not negated (`NEGATED`) goes to the name nearest before it and to the
    names listed with that one (`a`, `b` and `c`), else to the nearest after it; a name a suffix rule names
    (fields ending in `_bps`) is that suffix as a pattern (`*_bps`)."""
    statements, _ = props.segment(instruction, ())
    bound: dict[str, set[str]] = {}
    for statement in statements:
        for sentence in props.sentences(props.normal(statement.text)):
            names = [(start, end, name) for start, end, name in props.identifiers(sentence)]
            for scale, pattern in SCALES:
                for found in pattern.finditer(sentence):
                    if NEGATED.search(sentence[:found.start()]):
                        continue
                    for name in nearest(sentence, names, found.start()):
                        rule = SUFFIXED.search(sentence)
                        key = f"*{name}" if rule and name in rule.groups() else name
                        bound.setdefault(key, set()).add(scale)
    return bound


def nearest(sentence: str, names: list[tuple[int, int, str]], at: int) -> list[str]:
    """The name nearest before `at` with the names listed before it (joined by `and`, `or` or commas alone), else
    the name nearest after it; [] where the sentence names none."""
    before = [each for each in names if each[1] <= at]
    if before:
        chain = [before[-1]]
        for each in reversed(before[:-1]):
            if not LISTED.match(sentence[each[1]:chain[-1][0]].replace("`", "").replace("*", "")):
                break
            chain.append(each)
        return [name for _, _, name in reversed(chain)]
    after = [each for each in names if each[0] >= at]
    return [after[0][2]] if after else []


def listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


SCALE_SAID = {("percent", "low"): "is in percent, and the values read as decimals",
              ("basis points", "low"): "is in basis points, and the values read as decimals",
              ("decimal", "high"): "is a decimal, and the values read as percent"}


def scale(instruction: str, out: Path) -> str:
    """The scale finding over the output's names: each the instruction binds to one scale alone (`bound_scales`)
    whose numbers, one at least not zero, all read in another (`SCALE_SAID`), said in one line naming at most
    `SAID_NAMES`; "" where none."""
    bound = {name: scales.pop() for name, scales in bound_scales(instruction).items() if len(scales) == 1}
    if not bound:
        return ""
    outputs = props.Outputs(out)
    found: dict[str, list[str]] = {}
    for held in outputs.values:
        numbers = [abs(n) for n in held.numbers if n is not None and n == n and abs(n) != float("inf")]
        if not numbers or not any(numbers):
            continue
        low, high = all(n < 1 for n in numbers), all(n >= 1 for n in numbers)
        for name, said in bound.items():
            if not props.meets(name, held.name, False):
                continue
            read = "low" if low and said != "decimal" else "high" if high and said == "decimal" else ""
            shown = held.name.rsplit(".", 1)[-1]
            if read and shown not in found.setdefault(f"{said}/{read}", []):
                found[f"{said}/{read}"].append(shown)
    parts, named = [], 0
    for key, names in found.items():
        said, read = key.split("/")
        kept = names[:max(0, SAID_NAMES - named)]
        if not kept:
            break
        named += len(kept)
        verb = SCALE_SAID[(said, read)]
        if len(kept) > 1:
            verb = verb.replace("is in", "are in").replace("is a decimal", "are decimals")
        parts.append(f"{listed([f'`{name}`' for name in kept])} {verb}")
    return ("the instruction says " + "; it says ".join(parts)) if parts else ""
