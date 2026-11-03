"""The input files described `rich` (lane inputs1; the owner's in346, agenthon-t2): what the loop's first message
(`# Input files, described`) and form `state`'s INPUT FILES section carry under `AGENTHON_N11_INPUTS=rich`
(agent/n11/solve.py `INPUTS`), the same text in every form. Unset, they carry the standard descriptions
(commands.Workspace.described, state.inputs), byte for byte.

`described(data_dir, instruction, canaries)` describes every file under the unit's data directory, the files the task
names first (in the order it first names them), within `DESCRIBED_RICH_BYTES`, a constant of the harness's that no
text states: each whole where all fit, else a small file whole and the rest sharing what is left, a file the task names
with twice the room (`allotted`); a file past the bound is named at the end. Each file within its room:
- a table (a CSV, TSV or text table, read through its first `TABLE_ROWS` rows; a parquet file): its row and column
  counts, each column's type, its share missing, its values with their counts where few (`DISTINCT`, or
  `NUMERIC_DISTINCT` for numbers) and repeated, else its range (numbers) or its distinct count with its first and last
  value (text); a date-like column (an ISO date or time, a slashed date read day first where one's first field passes
  12, a month by name: state.when) its span as written, its distinct dates, its spacing (`spacing`: business days,
  daily, weekly, monthly, quarterly, half-yearly, yearly, intraday, or irregular with its steps), its order and its
  rows a date where a date repeats; then its first and last rows (`ROWS`, `NAMED_ROWS`; fewer where they would pass
  the room, `FEWER`, the columns kept first);
- JSON: its top's kind and its keys to `KEY_LEVELS` levels, each value's kind (common.digest.tree), and the file
  whole where it is small (`JSON_WHOLE`); JSON lines, the records and the first's keys;
- text: its lines and its first lines, the tags of XML or HTML counted;
- any other file as `inspect` describes it (commands.describe).
Pure: no request, no run, nothing named of any unit (rule 8).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path

from common import canary, digest
from n11 import commands, state
from n11.commands import size

DESCRIBED_RICH_BYTES = 20_000   # the whole description, at most: the harness's, never said to the model
MIN_SHARE = 1_500               # a file's room at the least (twice for a named one), else the last are left out
NOTE_BYTES = 400                # kept for the line naming the files past the bound
TABLE_ROWS = 200_000            # a delimited table's rows read for its statistics
COLUMNS = 40                    # a table's columns described, at most
ROWS = (10, 5)                  # a table's first and last rows shown (the state's standard: 5 and 2)
NAMED_ROWS = (15, 5)            # and a table's the task names (the state's standard: 10 and 3)
FEWER = ((5, 2), (2, 1), (0, 0))   # the rows a table falls back to where its description passes its room
DISTINCT = 12                   # a column's values listed with their counts where at most this many and one repeats
NUMERIC_DISTINCT = 5            # and a numeric column's
COUNTED = 10_000                # a column's distinct values counted up to this many
SPACED = 50_000                 # a date-like column's distinct dates read for its spacing, at most
REGULAR = 0.8                   # the share of steps its spacing's class must hold, else the spacing is irregular
TEXT_LINES = 40                 # a text file's first lines shown, at most
KEY_LEVELS = 2                  # a JSON file's keys shown to this depth
JSON_WHOLE = 2_000              # and the file shown whole too where its indented text is at most this many bytes, as
#                                 the state's standard shows a small one
CELL = 24                       # a value shown, at most
LINE = 300                      # a row shown, at most
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SEPARATED = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}
IRREGULAR_DATES = 6             # a date column's steps said irregular only over at least this many dates
DAY = 86_400                    # seconds
# (label, the steps in days it holds, the least and the most: a calendar's, give or take a business day's roll)
CLASSES = (("weekly", 6, 8), ("monthly", 27, 34), ("quarterly", 85, 96), ("half-yearly", 175, 190),
           ("yearly", 358, 372))


def described(data_dir: Path, instruction: str, canaries=(), limit: int = DESCRIBED_RICH_BYTES) -> str:
    """Every input file described within `limit` bytes (`allotted`), the files the task names first, those past the
    bound named last, the canaries marked `<canary>`."""
    data_dir = Path(data_dir)
    found = sorted(path for path in data_dir.rglob("*") if path.is_file()) if data_dir.is_dir() else []
    if not found:
        return f"No input files: `{data_dir}` holds none."
    first = {path: named_at(path, instruction) for path in found}
    named = sorted((path for path in found if first[path] is not None), key=lambda path: first[path])
    portraits = [Portrait(path, data_dir, path in named) for path in named + [path for path in found
                                                                              if first[path] is None]]
    rooms = allotted([size(each.said(limit)) + 2 for each in portraits], [2 if each.named else 1 for each in portraits],
                     limit - NOTE_BYTES)
    text = "\n\n".join(each.said(room - 2) for each, room in zip(portraits, rooms) if room is not None)
    left = [each.name for each, room in zip(portraits, rooms) if room is None]
    if left:
        text += "\n\n" + commands.fit("not described here: " + ", ".join(left), limit - size(text) - 2)
    return canary.redact(text, canaries)


def allotted(need: list[int], weight: list[int], budget: int) -> list[int | None]:
    """Each file's room within `budget`, in order: every file whole where all fit; else the last left out (None)
    while the rest could not each have `MIN_SHARE` (twice for a file the task names) or its whole, then each file whose
    whole fits its weighted share of what is left given its whole, and the rest that share."""
    rooms: list[int | None] = [None] * len(need)
    kept = list(range(len(need)))
    while kept and sum(min(need[k], MIN_SHARE * weight[k]) for k in kept) > budget:
        kept.pop()
    left, open_ = budget, kept
    while open_:
        pending = sum(weight[k] for k in open_)
        fits = [k for k in open_ if need[k] * pending <= left * weight[k]]
        if not fits:
            for k in open_:
                rooms[k] = left * weight[k] // pending
            break
        for k in fits:
            rooms[k] = need[k]
            left -= need[k]
        open_ = [k for k in open_ if k not in fits]
    return rooms


def named_at(path: Path, instruction: str) -> int | None:
    """Where the instruction first names the file by its name, never inside a longer name; None where it does not."""
    found = re.search(rf"(?<![\w.-]){re.escape(path.name)}(?![\w-])", instruction)
    return found.start() if found else None


class Portrait:
    """One file read once and said within any room: its name, its size and whether the task names it, then its
    description by its kind (a table's with the most rows that fit, a text's with the most lines), cut at the room."""

    def __init__(self, path: Path, data_dir: Path, named: bool) -> None:
        self.path, self.named, self.name = path, named, path.relative_to(data_dir).as_posix()
        self.head = f"{self.name} ({path.stat().st_size:,} B" + ("; the task names it" if named else "") + "): "
        self.table: Table | None = None
        self.text = ""                  # a description that is the same in any room; "" for a table and a text file
        suffix = path.suffix.lower()
        try:
            if suffix in (".parquet", ".pqt"):
                self.table = Table.parquet(path)
            elif suffix in (".json", ".jsonl", ".ndjson") or (suffix == "" and path.read_bytes()[:1] in (b"{", b"[")):
                self.text = json_keys(path)
            elif suffix in (".xlsx", ".zip") or digest.binary(path):
                self.text = commands.describe(path, 0)
            elif suffix in commands.TABLES and commands.table_like(sample(path)):
                self.table = Table.delimited(path)
        except Exception as exc:  # noqa: BLE001 - a file that cannot be read is said, never raised
            self.text = f"could not be read here ({type(exc).__name__})"

    def said(self, room: int) -> str:
        inner = room - size(self.head)
        try:
            body = self.table.fitted(inner, self.named) if self.table is not None else self.text or \
                text_lines(self.path, inner)
        except Exception as exc:  # noqa: BLE001
            body = f"could not be read here ({type(exc).__name__})"
        return digest.clip(self.head + body.rstrip("\n"), room).rstrip("\n")


def sample(path: Path) -> str:
    return path.read_bytes()[:65_536].decode("utf-8-sig", errors="replace")


# ------------------------------------------------------------------------------------------------- dates


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


@dataclass
class Dates:
    """A date-like column's distinct cells with their rows, and its order and its span as read, each slashed cell both
    ways."""
    cells: Counter = field(default_factory=Counter)
    whole: bool = True                  # every distinct cell counted (`SPACED`)
    spans: list = field(default_factory=lambda: [None, None])      # (month first, day first): the earliest and latest
    #                                                                 cell over every row read, counted or not
    day_first: bool = False             # a slashed cell's first field past 12
    month_first: bool = False           # its second field past 12
    rising: list = field(default_factory=lambda: [True, True])     # (month first, day first) never falling
    falling: list = field(default_factory=lambda: [True, True])    # never rising
    last: list | None = None

    def add(self, cell: str, moment: tuple) -> None:
        fields = state.slashed(cell)
        if fields is not None:
            self.day_first |= fields[0] > 12
            self.month_first |= fields[1] > 12
            both = [(fields[2], fields[0], fields[1]), (fields[2], fields[1], fields[0])]
        else:
            both = [moment, moment]
        if self.last is not None:
            for k in (0, 1):
                self.rising[k] &= both[k] >= self.last[k]
                self.falling[k] &= both[k] <= self.last[k]
        self.last = both
        for k in (0, 1):
            self.spans[k] = state.widened(self.spans[k], both[k], cell)
        if cell in self.cells or len(self.cells) < SPACED:
            self.cells[cell] += 1
        else:
            self.whole = False

    def said(self) -> str:
        """Its span as written (over every row read, where its distinct dates pass `SPACED`), its distinct dates, its
        spacing, its order and its rows a date where one repeats."""
        day = int(self.day_first and not self.month_first)
        read = {}
        for cell in self.cells:
            fields = state.slashed(cell)
            moment = ((fields[2], fields[1], fields[0]) if day else (fields[2], fields[0], fields[1])) \
                if fields is not None else state.when(cell, list(state.DATED))
            if moment is not None:
                try:
                    read.setdefault(moment_of(moment), cell)
                except ValueError:       # a slashed cell read the other way round names no day
                    continue
        if not read:
            return ""
        moments = sorted(read)
        early, late = (read[moments[0]], read[moments[-1]]) if self.whole else \
            (self.spans[day][0][1], self.spans[day][1][1])
        parts = [f"from {early[:CELL]} to {late[:CELL]}",
                 f"{len(moments):,}{'' if self.whole else ' or more'} distinct"]
        spaced = spacing(moments)
        if spaced:
            parts.append(spaced)
        if len(moments) > 1:
            parts.append("ascending" if self.rising[day] else "descending" if self.falling[day] else "unsorted")
        counts = set(self.cells.values())
        if max(counts) > 1:
            parts.append(f"each date on {max(counts):,} rows" if len(counts) == 1 else
                         f"a date on up to {max(counts):,} rows")
        return ", ".join(parts)


# ------------------------------------------------------------------------------------------------- tables


class Column:
    """One column read cell by cell: its missing cells, its numeric range, its values counted (`COUNTED`), its first
    and last value, and its dates where every cell reads as one (`Dates`)."""

    def __init__(self, name: str, kind: str | None = None) -> None:
        self.name, self.kind = name, kind
        self.missing = self.filled = 0
        self.sample: list[str] = []
        self.numeric, self.low, self.high = True, None, None
        self.values: Counter | None = Counter()
        self.first = self.last = ""
        self.dates: Dates | None = Dates()
        self.forms = list(state.DATED)

    def add(self, cell) -> None:
        cell = "" if cell is None else str(cell).strip()
        if cell.lower() in commands.EMPTY:
            self.missing += 1
            return
        self.filled += 1
        self.first, self.last = self.first or cell, cell
        if len(self.sample) < 2_000:
            self.sample.append(cell)
        number = commands.number(cell)
        if number is None:
            self.numeric = False
        elif self.numeric:
            self.low = number if self.low is None else min(self.low, number)
            self.high = number if self.high is None else max(self.high, number)
        if self.values is not None:
            self.values[cell] += 1
            if len(self.values) > COUNTED:
                self.values = None
        if self.dates is not None:
            moment = state.when(cell, self.forms)
            if moment is None:
                self.dates = None
            else:
                self.dates.add(cell, moment)

    def arrow(self, values) -> None:
        """A numeric or boolean parquet column read whole by pyarrow's compute, as `add` reads it cell by cell: a
        float NaN missing, its range, its values counted, its first and last value; no date."""
        import pyarrow as pa
        import pyarrow.compute as pc

        valid = pc.drop_null(values)
        if pa.types.is_floating(values.type):
            valid = valid.filter(pc.invert(pc.is_nan(valid)))
        self.filled, self.dates = len(valid), None
        self.missing = len(values) - self.filled
        self.numeric = not pa.types.is_boolean(values.type)
        if not self.filled:
            return
        if self.numeric:
            bounds = pc.min_max(valid).as_py()
            self.low, self.high = bounds["min"], bounds["max"]
        distinct = pc.count_distinct(valid).as_py()
        self.values = Counter({said_cell(item["values"]): item["counts"]
                               for item in pc.value_counts(valid).to_pylist()}) if distinct <= COUNTED else None
        self.first, self.last = said_cell(valid[0].as_py()), said_cell(valid[len(valid) - 1].as_py())

    def line(self) -> str:
        dated = self.dates.said() if self.dates is not None and self.filled else ""
        kind = self.kind or ("date" if dated else digest.kind(self.sample) if self.sample else "empty")
        total = self.missing + self.filled
        name = self.name[:40] if self.name == self.name.strip() and self.name else f"`{self.name[:40]}`"
        parts = [f"- {name}: {kind}", missing(self.missing, total)]
        few = DISTINCT if not self.numeric else NUMERIC_DISTINCT
        if dated:
            parts.append(dated)
        elif self.values is not None and 0 < len(self.values) <= few and len(self.values) < self.filled and (
                not self.numeric or min(self.values.values()) > 1):       # a number listed where each repeats
            ordered = sorted(self.values.items(), key=lambda item: (commands.number(item[0]) is None,
                                                                    commands.number(item[0]) or 0.0, item[0]))
            parts.append(f"{len(ordered)} value{'s' if len(ordered) != 1 else ''}: " + ", ".join(
                f"{digest.short(value, 2 * CELL)} ({count:,})" for value, count in ordered))
        elif self.numeric and self.low is not None:
            parts.append(f"min {said_number(self.low)}, max {said_number(self.high)}")
            if self.values is not None and 2 * len(self.values) <= self.filled:     # a key's or a code's
                parts.append(f"{len(self.values):,} distinct")
        elif self.filled:
            distinct = f"{len(self.values):,} distinct" if self.values is not None else \
                f"over {COUNTED:,} distinct"
            parts.append(f"{distinct}, first {self.first[:CELL]}, last {self.last[:CELL]}")
        return ", ".join(parts)


def said_number(value) -> str:
    """A whole number whole, any other to six significant digits."""
    return str(int(value)) if float(value).is_integer() and abs(value) < 1e15 else f"{value:.6g}"


def missing(count: int, total: int) -> str:
    """A column's missing cells and their share of its rows."""
    if not count:
        return "none missing"
    share = 100 * count / total if total else 100.0
    return f"{count:,} missing ({'<0.1' if share < 0.1 else f'{share:.1f}'}%)"


@dataclass
class Table:
    """A table read once (its head line, its columns' statistics, its first rows and its last, which may overlap them
    in a short table) and said within a room, with fewer rows where the room asks it."""
    title: str
    columns: list[str]
    more: str
    header: str
    first: list[str]
    last: list[str]
    count: int

    @classmethod
    def delimited(cls, path: Path) -> Table:
        """A CSV, TSV or text table, its statistics over its first `TABLE_ROWS` rows."""
        delimiter = "\t" if path.suffix.lower() == ".tsv" else commands.sniff(sample(path))
        head, tail = max(ROWS[0], NAMED_ROWS[0]), max(ROWS[1], NAMED_ROWS[1])
        with open(path, encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, [])
            columns = [Column(name) for name in header[:COLUMNS]]
            first: list[list[str]] = []
            last: deque = deque(maxlen=tail)
            count = 0
            for row in reader:
                if not row:
                    continue
                count += 1
                if len(first) < head:
                    first.append(row)
                last.append(row)
                if count <= TABLE_ROWS:
                    for column, cell in zip(columns, row):
                        column.add(cell)
        title = (f"a table of {count:,} rows x {len(header)} columns, "
                 f"{SEPARATED.get(delimiter, repr(delimiter))}-separated"
                 + (f"; its statistics over its first {TABLE_ROWS:,} rows" if count > TABLE_ROWS else ""))
        more = (f"- and {len(header) - COLUMNS} more columns: " + ", ".join(name[:30] for name in header[COLUMNS:80])
                if len(header) > COLUMNS else "")
        return cls(title, [column.line() for column in columns], more,
                   digest.short(delimiter.join(header), LINE),
                   [digest.short(delimiter.join(row), LINE) for row in first],
                   [digest.short(delimiter.join(row), LINE) for row in last], count)

    @classmethod
    def parquet(cls, path: Path) -> Table:
        """A parquet file, its types the file's, its statistics over its first `TABLE_ROWS` rows: a numeric or boolean
        column's by pyarrow's compute (`Column.arrow`), any other's cell by cell as a delimited table's."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        handle = pq.ParquetFile(path)
        schema = handle.schema_arrow
        names = schema.names[:COLUMNS]
        count = handle.metadata.num_rows
        batches, read = [], 0
        for batch in handle.iter_batches(batch_size=65_536, columns=names):
            batches.append(batch)
            read += batch.num_rows
            if read >= TABLE_ROWS:
                break
        data = pa.Table.from_batches(batches, schema=pa.schema([schema.field(name) for name in names])) \
            .slice(0, TABLE_ROWS)
        columns = []
        for name in names:
            column, values = Column(name, str(schema.field(name).type)), data.column(name)
            kind = values.type
            if pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_boolean(kind):
                column.arrow(values)
            else:
                for value in values.to_pylist():
                    column.add(said_cell(value))
            columns.append(column)
        head, tail = max(ROWS[0], NAMED_ROWS[0]), max(ROWS[1], NAMED_ROWS[1])
        first = data.slice(0, head).to_pylist()
        last: list[dict] = []
        if handle.num_row_groups:
            group = handle.read_row_group(handle.num_row_groups - 1, columns=names)
            last = group.slice(max(0, group.num_rows - tail)).to_pylist()
        title = f"parquet, {count:,} rows x {len(schema)} columns" + \
            (f"; its statistics over its first {data.num_rows:,} rows" if data.num_rows < count else "")
        more = f"- and {len(schema) - COLUMNS} more columns" if len(schema) > COLUMNS else ""
        return cls(title, [column.line() for column in columns], more, "",
                   [digest.short(json.dumps(row, default=str), LINE) for row in first],
                   [digest.short(json.dumps(row, default=str), LINE) for row in last], count)

    def said(self, head: int, tail: int) -> str:
        lines = [self.title, *self.columns] + ([self.more] if self.more else [])
        first = self.first[:head]
        if first:
            lines.append(f"its first {len(first)} rows" + (", under its header:" if self.header else ":"))
            lines += ([self.header] if self.header else []) + first
        shown = min(tail, len(self.last), self.count - len(first))     # the last rows, none among the first
        last = self.last[len(self.last) - shown:] if shown > 0 else []
        if last:
            lines.append(f"its last {len(last)} rows:")
            lines += last
        return "\n".join(lines)

    def fitted(self, room: int, named: bool) -> str:
        """Said with the most rows that fit `room` (`NAMED_ROWS` or `ROWS`, then `FEWER`); the columns kept first."""
        text = ""
        for head, tail in ((NAMED_ROWS if named else ROWS), *FEWER):
            text = self.said(head, tail)
            if size(text) <= room:
                return text
        return text


def said_cell(value) -> str:
    """A parquet cell as a CSV holds it: a time in ISO form, a float NaN missing."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    return str(value)


# ------------------------------------------------------------------------------------------- JSON and text


def json_keys(path: Path) -> str:
    """A JSON file's top kind and keys to `KEY_LEVELS` levels, each value's kind, and the file whole where its indented
    text is at most `JSON_WHOLE`; JSON lines' records and the first's keys."""
    if path.stat().st_size > commands.JSON_BYTES:
        return digest.head(path, commands.TEXT_HEAD, "JSON (too large to parse here)")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        doc = json.loads(text)
    except ValueError:
        try:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        except ValueError:
            return "not valid JSON; its first lines:\n" + "\n".join(text.splitlines()[:TEXT_LINES])
        return (f"JSON lines, {len(records):,} records; the first's keys to two levels, each with its kind:\n"
                + "\n".join(digest.tree(records[0] if records else {}, limit=KEY_LEVELS)))
    whole = digest.pretty(doc)
    return (f"JSON ({digest.label(doc)}); its keys to two levels, each with its kind:\n"
            + "\n".join(digest.tree(doc, limit=KEY_LEVELS)) + (f"\nshown whole:\n{whole}" if size(whole) <= JSON_WHOLE
                                                              else ""))


def text_lines(path: Path, room: int) -> str:
    """A text file's lines and its first lines within `room`; an XML or HTML file's tags counted."""
    text = path.read_bytes()[:2_000_000].decode("utf-8", errors="replace")
    tags = Counter(match.lower() for match in commands.TAG.findall(text))
    counted = ("\ntag counts: " + ", ".join(f"{tag} {n}" for tag, n in tags.most_common(20))
               if tags and path.suffix.lower() in (".xml", ".html", ".htm", ".xbrl") else "")
    said = f"text, {text.count(chr(10)):,} lines; its first lines:"
    room -= size(said) + size(counted)
    shown = []
    for line in text.split("\n")[:TEXT_LINES]:
        line = digest.short(line, LINE)
        if size(line) + 1 > room:
            break
        shown.append(line)
        room -= size(line) + 1
    return "\n".join([said, *shown]) + counted
