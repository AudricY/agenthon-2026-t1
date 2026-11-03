"""The output contract checked after a clean run (lane contract1; docs/research/conventions-1017.md change A, the checks
reads runs/lanes/codex-checks-{a,b,c}.md change 1): each named output file against the terms its instruction states
(agent/n11/props.py `terms`), one plain line a departure. The organisers' scorer is binary per task and its checks
compare exact key sets and orders where stated, row orders, literal cells and native types, so a departure from the
stated contract fails a unit whatever its values.

The switch: `AGENTHON_N11_CONTRACT` or agent/n11/CONTRACT, `off` (today's), `on` or `refuse`, armed in forms `loop`,
`review` and `sketch` (agent/n11/solve.py `contract_armed`). Under `on` and `refuse` the loop's first message says
`ASK` after its output files, and each run that exits 0 is read (agent/n11/commands.py `Workspace.observe`): the
findings of the kinds `KEPT` names are one paragraph of the run's observation, after the span and scale lines (and lane
derive1's paragraph) and before the checks' verdict, advisory. Under `refuse` a finish the checks pass while one stands
is turned away once a unit (`REFUSED`, the findings quoted); the next finish is today's. No text states a count; under
`off` every body is today's. Keys `KEYS`. Each kind is one function with its rule in its docstring; a
kind the text waives ("harmless differences in ... row order") says nothing, and what today's `instruction:` and
`contract:` lines already say (a stated rounding broken, a named key or column absent) is left to them.

- `exact`: where the text lists the keys or columns exactly, the output holds no other and lacks none.
- `order`: where the text fixes their order, the output's keys or columns run in it.
- `decimals`: a JSON number under a fixed-point rounding is written with exactly its decimals (a table's are today's).
- `na`: a missing cell is written with the word the text asks (empty, `NaN`), a JSON file's as `null`.
- `type`: a JSON value has its template's native type.
- `labels`: a column or key whose values the text lists ("one of", "the only valid labels", `true` or `false`) holds
  no other.
- `nested`: every path of a JSON template is present, its `{"value": ...}` wrappers among them.
- `rows`: a table's rows run in the stated sort, or in the order the text lists their values.

`scripts/contract_replay.py` runs every kind over the recorded outputs of failing and solved readings; a kind is kept
(`KEPT`) only where it fires on failing readings and on no solved one.

Pure: the standard library alone (pyarrow where a parquet file is read), nothing written, nothing named of any unit
(rule 8): a finding names the file, the key or column and at most a count of decimals, never a value of the task's.
"""
from __future__ import annotations

import csv
import io
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from n11 import props

ENV = "AGENTHON_N11_CONTRACT"
FILE = "CONTRACT"
SETTINGS = ("off", "on", "refuse")
FORMS = ("loop", "review", "sketch")    # the loop's forms (form `state` builds its own message, run text and finish)
KEYS = ("contract_mode", "contract_terms", "contract_runs", "contract_said", "contract_kinds", "contract_refused")
MAX = 1                     # the finishes the contract turns away a unit; the next is today's
ASK = ("Write each output exactly as the task states it: its keys or columns (in the stated order where it states "
       "one), the stated decimals, the stated word for a missing cell, native types (an integer as an integer, a "
       "boolean in the task's own words), only the labels it lists, every nested field, and the rows in the stated "
       "order.")                # the first message's sentence, after its output files
REFUSED = ("finish not taken: the output departs from what the task states of it, so the task goes on. Write each "
           "output as the task states it, run the script, then finish:\n{lines}")
KINDS = ("exact", "order", "decimals", "na", "type", "labels", "nested", "rows")     # every kind, in its said order
KEPT = ("exact", "order", "type", "labels", "nested", "rows")   # the kinds a run's observation says and a refusal
#   answers: those the replay kept, each firing on failing readings and on no solved one over every record the ledgers
#   name; `na` fired on solved readings, `decimals` on none (runs/lanes/contract1-report.md, the replay)
HEAD = "The output against the contract the task states (advisory):"
SAID = 6                    # findings said a run, at most: the harness's bound, never said
NAMED = 3                   # names one finding lists, at most
LINE_CHARS = 260            # a finding's line, at most
MAX_BYTES = 8 * 1024 * 1024     # an output file larger than this is not read
MAX_ROWS = 200_000          # a table's rows read
NA_WORDS = frozenset({"nan", "-nan", "none", "null", "na", "n/a", "<na>", "nat"})    # what a writer puts for none
TYPE_SAID = {"int": "an integer", "number": "a number", "bool": "a boolean", "str": "a string", "list": "a list",
             "object": "an object", "float": "a float"}


class Written(float):
    """A JSON number as written: its value, and its text (`text`)."""

    def __new__(cls, text: str):
        made = super().__new__(cls, text)
        made.text = text
        return made


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]               # each row's cells as written ("" for a parquet null)


@dataclass
class Read:
    """The named output files as written: each JSON document (numbers as `Written`), each table."""
    json: dict[str, object] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)


def read(out: Path, files: dict[str, str]) -> Read:
    """The named output files under `out` by their format; one absent, too large or unreadable is left out."""
    found = Read()
    for name, kind in files.items():
        path = Path(out) / name
        try:
            if not path.is_file() or path.stat().st_size > MAX_BYTES:
                continue
            if kind == "json":
                found.json[name] = json.loads(path.read_bytes().decode("utf-8-sig", errors="replace"),
                                              parse_float=Written, parse_constant=Written)
            elif kind in ("csv", "tsv"):
                text = path.read_bytes().decode("utf-8-sig", errors="replace")
                reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t" if kind == "tsv" else ",")
                header = [cell.strip() for cell in next(reader, [])]
                found.tables[name] = Table(header, [row for row in itertools.islice(reader, MAX_ROWS)
                                                    if any(cell.strip() for cell in row)])
            elif kind == "parquet":
                import pyarrow.parquet as pq
                frame = pq.read_table(path)
                columns = [frame.column(k).to_pylist()[:MAX_ROWS] for k in range(frame.num_columns)]
                found.tables[name] = Table([str(name_) for name_ in frame.column_names],
                                           [["" if cell is None else str(cell) for cell in row]
                                            for row in zip(*columns)])
        except Exception:  # noqa: BLE001 - a file read wrong is no finding
            continue
    return found


# ------------------------------------------------------------------------------------------------ reading

def names(items, limit: int = NAMED) -> str:
    items = list(dict.fromkeys(items))
    text = ", ".join(f"`{item}`" for item in items[:limit])
    return text + (" among others" if len(items) > limit else "")


def shown(path: tuple[str, ...]) -> str:
    return ".".join(path)


def object_at(doc, path: tuple[str, ...]) -> list[dict]:
    """The JSON objects at `path` (`[]` each record of a list); [] where none."""
    found = [doc]
    for part in path:
        if part == "[]":
            found = [item for each in found if isinstance(each, list) for item in each]
        else:
            found = [each[part] for each in found if isinstance(each, dict) and part in each]
    return [each for each in found if isinstance(each, dict)]


def values_at(doc, path: tuple[str, ...]) -> list:
    """The values at `path` (`[]` each item of a list) that are present."""
    found = [doc]
    for part in path:
        if part == "[]":
            found = [item for each in found if isinstance(each, list) for item in each]
        else:
            found = [each[part] for each in found if isinstance(each, dict) and part in each]
    return found


def first_missing(doc, path: tuple[str, ...]) -> tuple[str, ...] | None:
    """The shortest prefix of `path` some value lacks (an object without the key, or no object where one belongs);
    None where every value holds it whole or a list on the way is empty."""
    found = [doc]
    for depth, part in enumerate(path):
        if part == "[]":
            if any(not isinstance(each, list) for each in found):
                return path[:depth + 1]
            found = [item for each in found for item in each]
            continue
        if any(not isinstance(each, dict) or part not in each for each in found):
            return path[:depth + 1]
        found = [each[part] for each in found]
    return None


def kind_of(value) -> str | None:
    """A JSON value's native type as written; None for null."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, Written):
        text = value.text.lower()
        return "float" if any(mark in text for mark in (".", "e", "nan", "inf")) else "int"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return {str: "str", list: "list", dict: "object"}.get(type(value))


def fits(kind: str, wanted: str) -> bool:
    return kind == wanted or wanted == "number" and kind in ("int", "float")


def decimals_of(text: str) -> int | None:
    """The decimals a number is written with (`1.50` two, `1e-05` five, `12` none); None for no finite number."""
    from decimal import Decimal, InvalidOperation
    try:
        exponent = Decimal(text.strip()).as_tuple().exponent
    except (InvalidOperation, ValueError):
        return None
    return None if not isinstance(exponent, int) else max(-exponent, 0)


def number(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ------------------------------------------------------------------------------------------------ the kinds

def exact(terms: props.Terms, out: Read) -> list[str]:
    """Where the text lists a JSON object's keys or a table's columns exactly ("exactly these keys", "only these
    columns"), the output holds none the list leaves out and lacks none of it (a name today's `contract:` lines read
    left to them)."""
    found = []
    for term in terms.terms:
        if term.kind not in ("keys", "columns") or not term.exact:
            continue
        today = terms.today.get(term.file, frozenset())
        for held in held_names(term, out):
            extra = [name for name in held if name not in term.names]
            gone = [name for name in term.names if name not in held
                    and shown((*term.path, name)) not in today and name not in today]
            what = "keys" if term.kind == "keys" else "columns"
            parts = ([f"holds {names(extra)} beyond the {what} the task lists exactly"] if extra else []) + \
                    ([f"lacks {names(gone)} of them"] if gone else [])
            if parts:
                found.append(f"`{where(term)}` " + ", and ".join(parts) + f": write exactly the {what} it lists.")
                break
    return found


def held_names(term: props.Term, out: Read) -> list[list[str]]:
    """The keys of each object at the term's path, or the table's header; [] where the file is not read."""
    if term.kind == "columns":
        table = out.tables.get(term.file)
        return [table.header] if table is not None else []
    if term.file not in out.json:
        return []
    return [list(each) for each in object_at(out.json[term.file], term.path)]


def where(term: props.Term) -> str:
    return term.file + (f"`'s `{shown(term.path)}" if term.path and term.path != ("[]",) else "")


def order(terms: props.Terms, out: Read) -> list[str]:
    """Where the text fixes the order of a JSON object's keys or a table's columns ("in this exact order", "write keys
    in the listed order"), the output's run in it: the names both hold, compared position by position."""
    found = []
    for term in terms.terms:
        if term.kind not in ("keys", "columns") or not term.ordered:
            continue
        for held in held_names(term, out):
            mine = [name for name in held if name in term.names]
            theirs = [name for name in term.names if name in held]
            off = next((k for k, (a, b) in enumerate(zip(mine, theirs)) if a != b), None)
            if off is not None:
                what = "keys" if term.kind == "keys" else "columns"
                found.append(f"`{where(term)}`'s {what} are not in the order the task states: it puts "
                             f"`{theirs[off]}` where the output has `{mine[off]}`.")
                break
    return found


def decimals(terms: props.Terms, out: Read) -> list[str]:
    """A JSON number under a fixed-point rounding ("exactly 6 decimals", "fixed-point"), by key or over the file's
    numbers, is written with exactly those decimals, read from its text; an integer as written passes. A table's
    roundings, and a JSON rounding to at most N decimals, are today's `instruction:` lines'."""
    found = []
    for term in terms.terms:
        if term.kind != "decimals" or not term.strict or term.scope == "table":
            continue
        for file, doc in out.json.items():
            if term.file and file != term.file:
                continue
            bad = [(key, places) for key, value in leaves(doc) if isinstance(value, Written)
                   and (not term.names or key in term.names)
                   and (places := decimals_of(value.text)) is not None and kind_of(value) == "float"
                   and places != term.value]
            if bad:
                key, places = bad[0]
                found.append(f"`{file}`'s `{key}` is written with {places} decimals; the task states exactly "
                             f"{term.value}.")
                break
    return found


def leaves(doc, key: str = ""):
    """A JSON document's (last key, value) pairs, a list's items under their list's key."""
    if isinstance(doc, dict):
        for name, value in doc.items():
            yield from leaves(value, str(name))
    elif isinstance(doc, list):
        for value in doc:
            yield from leaves(value, key)
    else:
        yield key, doc


def na(terms: props.Terms, out: Read) -> list[str]:
    """A table column whose missing cell the text asks written empty holds no word for none (`nan`, `None`, `NA`);
    one asked written `NaN` leaves none empty and writes it so; a JSON file whose missing values the text asks
    `null` holds no NaN or infinity and no `"NaN"` string."""
    found = []
    for term in terms.terms:
        if term.kind != "na":
            continue
        if term.value == "null":
            doc = out.json.get(term.file)
            if doc is None:
                continue
            bad = [key for key, value in leaves(doc) if isinstance(value, Written) and kind_of(value) == "float"
                   and value.text.lower().lstrip("-+") in ("nan", "infinity")
                   or isinstance(value, str) and value.strip().lower() in NA_WORDS - {"null"}]
            if bad:
                found.append(f"`{term.file}` writes NaN for `{bad[0]}` where the task asks JSON `null`.")
            continue
        table = out.tables.get(term.file)
        if table is None:
            continue
        for column in term.names:
            if column not in table.header:
                continue
            at = table.header.index(column)
            cells = [row[at].strip() if at < len(row) else "" for row in table.rows]
            if term.value == "":
                word = next((cell for cell in cells if cell.lower() in NA_WORDS), None)
                if word is not None:
                    found.append(f"`{term.file}`'s `{column}` writes `{word}` for a missing cell; the task asks it "
                                 f"left empty.")
                    break
            else:
                word = next((cell for cell in cells if (not cell or cell.lower() in NA_WORDS) and cell != "NaN"), None)
                if word is not None:
                    shown_ = f"writes `{word}` for a missing cell" if word else "leaves a missing cell empty"
                    found.append(f"`{term.file}`'s `{column}` {shown_}; the task writes `NaN`.")
                    break
    return found


def types(terms: props.Terms, out: Read) -> list[str]:
    """A JSON value under a template's placeholder has its native type (`<int>` an integer as written, never `3.0`;
    `<bool>` true or false; `<string>` a string; `<float>` a number; a list, an object), null aside."""
    found = []
    for term in terms.terms:
        if term.kind != "type":
            continue
        doc = out.json.get(term.file)
        if doc is None:
            continue
        wrong = [kind for kind in map(kind_of, values_at(doc, term.path))
                 if kind is not None and not fits(kind, term.value)]
        if wrong:
            found.append(f"`{term.file}`'s `{shown(term.path)}` is written as {TYPE_SAID[wrong[0]]}; the task's "
                         f"template gives {TYPE_SAID[term.value]}.")
    return found[:NAMED]


def labels(terms: props.Terms, out: Read) -> list[str]:
    """A table column, or a JSON key at any depth, whose values the text lists holds no value outside the list as
    written (case and spelling kept); an empty cell, a word for none and a non-text JSON value aside."""
    found = []
    for term in terms.terms:
        if term.kind != "labels" or not term.names:
            continue
        name, allowed = term.names[0], set(term.value)
        if term.file in out.tables:
            table = out.tables[term.file]
            if name not in table.header:
                continue
            at = table.header.index(name)
            held = [row[at].strip() for row in table.rows if at < len(row)]
        elif term.file in out.json:
            held = [value.strip() for key, value in leaves(out.json[term.file]) if key == name and isinstance(value, str)]
        else:
            continue
        word = next((cell for cell in held if cell and cell.lower() not in NA_WORDS and cell not in allowed), None)
        if word is not None:
            written = f"`{word}`" if number(word) is None and len(word) <= 60 else "a value"
            found.append(f"`{term.file}`'s `{name}` writes {written}, which the task's labels for it do not hold "
                         f"({names(term.value)}).")
    return found


def nested(terms: props.Terms, out: Read) -> list[str]:
    """Every path a JSON template gives two keys deep or more is present in its file: the shortest missing prefix
    named (a `{"value": ...}` wrapper left out shows as its `value`), once, and a prefix today's `contract:` lines read
    left to them."""
    missing: dict[str, list[str]] = {}
    for term in terms.terms:
        if term.kind != "nested":
            continue
        doc = out.json.get(term.file)
        if doc is None:
            continue
        gap = first_missing(doc, term.path)
        if gap is None or shown(gap) in terms.today.get(term.file, frozenset()):
            continue
        missing.setdefault(term.file, [])
        if shown(gap) not in missing[term.file]:
            missing[term.file].append(shown(gap))
    return [f"`{file}` lacks {names(paths)}: write each nested field the task's template gives, as it nests it."
            for file, paths in missing.items()]


def rows(terms: props.Terms, out: Read) -> list[str]:
    """A table's rows run in the sort the text states (`sorted by \\`a\\` ascending, then \\`b\\``: numbers compared as
    numbers, text as text, a pair with an empty key cell skipped), or the values it lists in order (`Rows must appear
    in this order:`) first appear in that order in the column that holds them."""
    found = []
    for term in terms.terms:
        if term.kind != "rows":
            continue
        table = out.tables.get(term.file)
        if table is None:
            continue
        if term.value == "listed":
            listed = set(term.names)
            for at, column in enumerate(table.header):
                seen = list(dict.fromkeys(row[at].strip() for row in table.rows if at < len(row)
                                          and row[at].strip() in listed))
                if len(seen) >= 2:
                    if seen != [value for value in term.names if value in seen]:
                        found.append(f"`{term.file}`'s rows are not in the order the task lists for `{column}`.")
                    break
            continue
        if not all(column in table.header for column in term.names):
            continue
        at = [table.header.index(column) for column in term.names]
        keys = [[row[k].strip() if k < len(row) else "" for k in at] for row in table.rows]
        if not ordered_rows(keys, term.value):
            said = ", then ".join(f"`{column}` {way}" for column, way in zip(term.names, term.value))
            found.append(f"`{term.file}`'s rows are not sorted by {said} as the task states.")
    return found


def ordered_rows(keys: list[list[str]], ways: tuple[str, ...]) -> bool:
    """Whether consecutive rows' keys run in the sort: the first key that differs decides, as a number where both
    cells read as one, else as text; a pair with an empty cell before the deciding key is skipped."""
    for before, after in zip(keys, keys[1:]):
        for a, b, way in zip(before, after, ways):
            if not a or not b:
                break
            x, y = number(a), number(b)
            left, right = (x, y) if x is not None and y is not None else (a, b)
            if left == right:
                continue
            if (left > right) if way == "ascending" else (left < right):
                return False
            break
    return True


FUNCTIONS = {"exact": exact, "order": order, "decimals": decimals, "na": na, "type": types, "labels": labels,
             "nested": nested, "rows": rows}


@dataclass(frozen=True)
class Found:
    """One kind's finding: the kind and its line."""
    kind: str
    line: str


def run(terms: props.Terms, out: Path, kinds=KEPT) -> list[Found]:
    """The findings of `kinds` over the output directory `out` against `terms`, in `KINDS` order; a kind the text
    waives, or one that raises, says nothing."""
    if not terms.terms:
        return []
    found_out = read(out, terms.files)
    found = []
    for kind in KINDS:
        if kind not in kinds or kind in terms.waived:
            continue
        try:
            lines = FUNCTIONS[kind](terms, found_out)
        except Exception:  # noqa: BLE001 - a kind that cannot read the output says nothing
            lines = []
        found.extend(Found(kind, line if len(line) <= LINE_CHARS else line[:LINE_CHARS - 1] + "…") for line in lines)
    return found


def said(found: list[Found]) -> str:
    """The run's paragraph: `HEAD`, then one line a finding, at most `SAID`; "" where none."""
    if not found:
        return ""
    return "\n".join([HEAD, *(f"- {each.line}" for each in found[:SAID])])
