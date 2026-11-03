"""The conventions the instruction fixes, put to the model as questions, and two findings that read a run's output
against the instruction (lane convcard1; the owner's in616 and in618, agenthon-t2). The loop's and the holdout's
failure reads (docs/research/loop-wrong-outputs-1011.md, docs/research/holdout-failures-1012.md) traced most
clean-but-wrong outputs to a convention or shape the instruction fixed in words the model read once and did not carry
into its code: a scale, a window, a count, a row order, a parameter.

- **The card** (`card`; agent/n11/solve.py's `CONVCARD` switch, `AGENTHON_N11_CONVENTIONS=on`): the instruction
  matched against a fixed table of convention words and phrases (`TERMS`: scale, time, windows, returns, estimators,
  rounding and ordering), each word found quoted as the instruction writes it and put as a question its code must
  answer, never an answer; then each parameters file's keys (`parameters`), named as given (their values never
  said). Within `CARD_TOKENS` of the model's tokens (`CARD_BYTES` where the tokenizer does not load), never said: the
  lines past it are left out whole. "" where the instruction uses none and no parameters file is there.
- **The span finding** (`span`): an output table whose dates run as a series (`Series.regular`: one spacing, no wide
  gap), against the dates of the task's input tables (`Dated`): where it starts after the span the middle input
  table holds (`Dated.held`) by more than the longest window the instruction names (`windows`), two steps and a
  weekend, or ends before it by more than two steps and a weekend (and the longest horizon, a length beside a word
  that looks ahead of a row: `LEAD`), the table still covering at least `COVERS` of that span (a few dates a condition picks
  are no series cut short); or where, overlapping them, it runs past every date the input tables hold (in any
  column: `Dated.joint`) by more than a step. One line, its file named, no date said.
- **The scale finding** (`scale`): an output name the instruction says is in percent, in basis points or a decimal
  (`bound_scales`: the scale word bound to the name nearest before it in its sentence, or the list it ends), whose
  values read in another scale: all under one in magnitude where percent or basis points are said, all at least one
  where a decimal is said. One line naming the names, never a value.
Both findings are said after a run that exits 0 (agent/n11/commands.py `Workspace.against`), beside the instruction's
and the contract's findings and apart from them: neither blocks a finish nor makes a run unclean.

Pure: the standard library alone (pyarrow where a parquet file is read), nothing written, nothing named of any unit
(rule 8). It imports agent/n11/commands.py (and state.py and inputs.py, which import it too): commands.py imports
this module inside the method that uses it.
"""
from __future__ import annotations

import csv
import io
import itertools
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from n11 import commands, inputs, measure, props, state

CARD_TOKENS = 600           # the card's section, at most, in the model's tokens: the harness's, never said
CARD_BYTES = 2_400          # and in bytes where the tokenizer does not load (the program's 4 bytes a token)
CARD_HEAD = "# The task's conventions, as questions"
CARD_LEAD = ("Each line quotes the task's own words and asks what they fix. Settle each from the task's text before "
             "computing, and write the code to that answer:")
VALUE_KEY = re.compile(r"[-+]?\d[\d_.,eE%+-]*")           # a key that is a number: a value, never named
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
SAID_NAMES = 3              # the names the scale finding says, at most: the harness's bound, never said


# ------------------------------------------------------------------------------------------------ the card


@dataclass(frozen=True)
class Term:
    kind: str                   # scale, time, window, returns, estimator, order
    pattern: re.Pattern
    question: str               # asked of the words as the instruction writes them (`{said}`)


def term(kind: str, pattern: str, question: str) -> Term:
    return Term(kind, re.compile(pattern, re.I), question)


TERMS = (   # the table, in the card's order: each a question, never an answer (no row names a unit's words)
    term("scale", r"\bper ?cent(?:age)?s?\b|\bpercentage points?\b|\(\s*%\s*\)|\bin %",
         "which outputs are in percent, and which inputs?"),
    term("scale", r"\bbasis[- ]points?\b|\bbps\b|\bbp\b",
         "which values are in basis points, and where does the code convert them on the way in and out?"),
    term("scale", r"(?<!\d )\bdecimals?\b(?!\s+places?)|\braw floats?\b|\bas (?:a )?fractions?\b",
         "which values are decimals, and which inputs arrive in another scale?"),
    term("scale", r"\bper unit\b|\bper 1(?:\.0)? ?%|\bper (?:one |1 )?(?:vol(?:atility)? point|percentage point|"
                  r"basis point|bp)\b|\bpartial derivatives?\b|(?<!days )(?<!steps )(?<!times )(?<!bars )(?<!payments )"
                  r"\bper[- ](?:year|day|trading day|calendar day)\b",
         "is each output the plain quantity, or scaled per unit, per point, per percent, per day or per year?"),
    term("time", r"\bannuali[sz](?:e|ed|es|ing|ation)\b|\bper annum\b",
         "which quantities are annualised, and with how many periods a year?"),
    term("time", r"\btrading[- ]days?\b",
         "which windows, counts and annualisations run in trading days, and where does a calendar day count instead?"),
    term("time", r"\bcalendar[- ]days?\b",
         "which spans run in calendar days, and where do trading days count instead?"),
    term("time", r"\bbusiness[- ]days?\b",
         "which dates must fall on business days, and how is a date that is not one moved?"),
    term("time", r"\b(?:act|actual|a)/(?:360|365(?:f|l)?|act|actual)\b|\b30(?:e|u)?/360\b|\bday[- ]count\w*",
         "which year fractions use it, and between which dates?"),
    term("time", r"\byear[- ]fractions?\b|\bin years\b|\btime to (?:expiry|maturity)\b",
         "which inputs take times in years, and computed from which dates under which day count?"),
    term("time", r"\bmonth[- ]ends?\b|\bmonthly\b",
         "which series are monthly, and is a month's value its last day's or its first's?"),
    term("time", r"\bcontinuously[- ]compounded\b|\bcompound(?:ed|ing)?\b(?!\s+(?:poisson|options?)\b)|\bsimple (?:interest|rates?|division)\b",
         "which rates are compounded, how often, and which are simple?"),
    term("window", r"\bwindows?\b",
         "how long is each window, which rows does it hold, and is its first full window the first row kept?"),
    term("window", r"\brolling\b",
         "which quantities roll, over which rows, and what stands before the first full window?"),
    term("window", r"\btrailing\b(?!\s+(?:`?z\b|spaces?|whitespace|comments?|commas?|zeros?|newlines?|stop))",
         "does each trailing window end on the current row or on the one before it?"),
    term("window", r"\blook-?backs?\b(?!\s+options?\b)",
         "how far back does each lookback reach, and does it hold the current row?"),
    term("window", r"\bexpanding\b",
         "from which row does each expanding window start?"),
    term("window", r"\binclusive(?:ly)?\b|\bexclusive\b|\bhalf-open\b",
         "which ends of each range are held, and does the code's slicing hold the same ones?"),
    term("window", r"\bformation\b|\bskip(?:s|ping|ped)? the (?:most recent|last|latest|first)\b|\bholding period\b",
         "which rows form each signal, which are skipped, and in which period does it trade?"),
    term("window", r"\blag(?:s|ged)?\b|\bt ?- ?1\b|\bt ?\+ ?1\b|\bprevious (?:day|bar|row|period|month|close)\b|"
                   r"\b(?:next|following) (?:day|bar|row|period|month|open|session)\b|"
                   r"(?<!parallel )(?<!rate )(?<!drift )(?<!curve )\bshift(?:s|ed)?\b",
         "which values are lagged, by how far, and does a signal act on its own row or a later one?"),
    term("window", r"\bwarm-?up\b|\bburn-?in\b|\bmin_periods\b|\bfirst full\b|\bfirst valid\b",
         "which first rows lack a full history, and does the task keep, fill or drop them?"),
    term("window", r"\bvalid (?:observations?|rows?|returns?|months?|days?|data points?)\b|"
                   r"\bnumber of observations\b|\bnon-?missing\b",
         "which count does each output report: the rows left after dropping the missing, or a window's length?"),
    term("returns", r"\blog(?:arithmic)?[- ]returns?\b|\blog[- ](?:price )?(?:changes?|differences?)\b",
         "which returns are log returns, and which simple?"),
    term("returns", r"\bsimple returns?\b|\barithmetic returns?\b|\bpct_change\b|\bpercentage changes?\b",
         "which returns are simple, and which log?"),
    term("returns", r"\bexcess (?:returns?|of)\b",
         "which returns are in excess of which rate, taken at which period?"),
    term("returns", r"\btotal returns?\b|\badj(?:usted)?[ _]?close\b|\bdividends? reinvested\b",
         "which prices hold dividends and splits, and which returns are total?"),
    term("returns", r"\bcumulative (?:returns?|wealth|pnl|p&l)\b|\bcumprod\b|\bcompounded returns?\b",
         "is each cumulative series compounded or summed, and from which starting value?"),
    term("returns", r"(?<!stop-)(?<!stop )\blosse?s?\b|\bsigned\b|\b(?:as|is) (?:a )?positive (?:number|value|loss|magnitude)\b",
         "which values are losses and which returns, and which sign does each function the code calls take?"),
    term("estimator", r"\bsample (?:standard deviation|std|variance|covariance|mean|statistics|estimates?)\b|"
                      r"\bpopulation (?:standard deviation|std|variance|covariance)\b|\b(?:un)?biased\b",
         "which estimators are the sample's and which the population's, and with which ddof?"),
    term("estimator", r"\bddof\b|\bbessel\b|\b(?:by|over) \(?n ?- ?1\b",
         "which estimators divide by n and which by n - 1?"),
    term("estimator", r"\bewma\b|\bexponentially[- ]weighted\b|\briskmetrics\b|\bema\b|\bewm\b|\bhalf-?life\b|"
                      r"\bdecay (?:factor|rate)\b",
         "what seeds the recursion, what is its decay, and is it adjusted?"),
    term("estimator", r"\bseed(?:s|ed)?\b|\binitiali[sz](?:e|ed|es|ation)\b|\binitial (?:value|variance|guess)\b",
         "which random draws or recursions take a seed or a start, and from where?"),
    term("order", r"\bround(?:ed|ing)?\b|\bdecimal places?\b|\bsignificant (?:figures|digits)\b",
         "which outputs are rounded, at which step, and to which precision?"),
    term("order", r"\bsort(?:ed|ing)?\b|\bordered by\b|\bchronological(?:ly)?\b",
         "which outputs are sorted, by which key, and before which step?"),
    term("order", r"\bascending\b|\bdescending\b|\bmost[- ]recent[- ]first\b|\b(?:oldest|earliest|newest|latest)"
                  r"[- ]first\b|\breverse[- ]chronological\b",
         "in which direction does each input arrive, and in which must each output run?"),
    term("order", r"\blatest\b|\bmost recent\b|\bearliest\b|\blast (?:available|observation|date|row|value)\b",
         "latest or earliest by which date, in a frame sorted which way?"),
    term("order", r"\b(?:date|day|month)s? (?:of|with|at) (?:the )?(?:largest|widest|highest|lowest|maximum|"
                  r"minimum|smallest|narrowest|biggest|peak|max|min)\b|\bwidest\b|\bnarrowest\b",
         "over which rows is the extreme taken, and which date wins a tie?"),
    term("order", r"\bties?\b|\btie-?break\w*",
         "how is each tie broken, and does the code's sort keep that order?"),
)
KINDS = ("scale", "time", "window", "returns", "estimator", "order")


@dataclass(frozen=True)
class Card:
    """The card's section (`text`, "" where nothing fires), the words of each line kept (a parameters line: its
    file), and the lines the bound left out."""
    text: str = ""
    terms: tuple[str, ...] = ()
    left: int = 0


def lines_of(instruction: str, given: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
    """The card's lines in `TERMS`' order and then the parameters', each (its words, its line): a row's first match
    in the instruction's prose (a whole word, never inside a name such as `annualized_return`), quoted as written,
    its spaces and line breaks as one space; a parameters file's keys named."""
    text = " ".join(instruction.split())        # a phrase broken across lines reads as one
    found: list[tuple[str, str]] = []
    for row in TERMS:
        match = row.pattern.search(text)
        if match:
            said = match.group(0)
            found.append((said, f"- the task says '{said}': {row.question}"))
    for name, keys in given:
        named = listed([f"`{key}`" for key in keys])
        found.append((name, f"- the parameter{'s' if len(keys) > 1 else ''} {named} "
                            f"{'are' if len(keys) > 1 else 'is'} given in {name}: use "
                            f"{'each' if len(keys) > 1 else 'it'}, do not assume it."))
    return found


def card(instruction: str, given: list[tuple[str, list[str]]] = (), bound: int | None = None) -> Card:
    """The card for `instruction` and the parameters files `given` (`parameters`), within `bound` (`CARD_TOKENS` in
    the process's unit, measure.bound): its head, its lead and the lines that fit whole, in order."""
    found = lines_of(instruction, list(given))
    if not found:
        return Card()
    bound = measure.bound(CARD_TOKENS, CARD_BYTES) if bound is None else bound
    kept: list[tuple[str, str]] = []
    text = ""
    for said, line in found:
        trial = "\n".join([CARD_HEAD, "", CARD_LEAD, *[each for _, each in kept], line])
        if measure.size(trial) <= bound:
            kept.append((said, line))
            text = trial
    return Card(text, tuple(said for said, _ in kept), len(found) - len(kept)) if kept else Card(left=len(found))


def listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def parameters(data_dir: Path) -> list[tuple[str, list[str]]]:
    """Each parameters file among the inputs and its top-level keys, as lane tools1's reader names them
    (commands.parameter_files: a key opening with `_` or holding prose left out), less a key that reads as a number
    or a date (a value, not a name: `VALUE_KEY`, `moment`); [] where none reads."""
    found = []
    for name, keys in commands.parameter_files(Path(data_dir)):
        kept = [key for key in keys if not VALUE_KEY.fullmatch(key) and moment(key) is None]
        if kept:
            found.append((name, kept))
    return found


# ----------------------------------------------------------------------------------------------- the dates


def moment(cell: object, named: bool = False) -> datetime | None:
    """A cell as a date: a date or datetime value; an ISO date or time, a slashed date, a month by name
    (state.when); a year and month (`2020-01`); an eight-digit day (`20200131`) where `named` (the column's name
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
        fields = state.when(text)
        return inputs.moment_of(fields) if fields is not None else None
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
        """Its dates at one spacing (inputs.spacing: business days, daily, a calendar class, intraday) with no gap
        past `GAPS` steps and `GAP_DAYS` (a weekend and a holiday): a series, not a list of dates a condition picks."""
        spaced = inputs.spacing(list(self.dates))
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
