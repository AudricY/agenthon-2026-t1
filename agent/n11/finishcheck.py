"""The finish contract (lane finishcheck1; the cold reader's form 5, runs/lanes/cold-velocity-proposals.toml): a check
over the output at a finish that makes no model request. `AGENTHON_N11_FINISHCHECK` or agent/n11/FINISHCHECK, `off`
(today's) or `on`, armed in forms `loop`, `review` and `sketch`; agent/n11/solve.py `finishcheck_turn` acts on it.

- `stated(instruction, named, inputs, tie)` reads the instruction once: the names it puts in backticks or quotes that
  stand for a column or a key of an output file (`Named`), each with the output files its context names. A name is
  counted only where its context ties it to an output: under `sentence` (the switch's rule, the brief's conservative
  one) the name's own sentence names an output file of `named` or a governing "column", "key", "field" or "header";
  explicitly introduced values, formats and placeholders are left out, as are names held by the inputs; under
  `block` the lead line of the list, table or code block it sits in and the heading above it count as its context
  too; under `any` every name counts (the replay's comparison, never the switch's). A context naming an input file
  (one of `inputs`, or a path under `/input`) and no output file ties the name to the input: it is not counted.
- `check(out, named, names, kinds)` reads the output directory and returns the findings (`Finding`), one plain line
  each, never a count: a named file missing or empty (no bytes, an empty JSON value), or a table with a header and
  no row (`norows`); a file that does not parse by its extension (`.csv`/`.tsv` by the csv module with a header row,
  `.json` by json, `.jsonl` a line at a time, `.txt`/`.md` as UTF-8 text, `.parquet` by pandas or pyarrow where
  either imports, else not checked); a counted name held by no output file that could hold it (a table's header or
  a cell, a JSON key at any depth or a string value, a text's words); a counted name's values holding NaN or
  infinity (`nan`) or, in a table's column of numbers, an empty cell, or under a JSON key, null (`blank`).

Pure: the standard library alone (pandas or pyarrow for parquet where installed), nothing written. No unit's names.
"""
from __future__ import annotations

import csv
import io
import itertools
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from common import contract, paths

ENV = "AGENTHON_N11_FINISHCHECK"
FILE = "FINISHCHECK"
SETTINGS = ("off", "on")
FORMS = ("loop", "review", "sketch")    # the loop's forms (form `state` keeps its own finish)
KEYS = ("finishcheck_mode", "finishcheck_turned", "finishcheck_flags")
MAX = 1                                 # the finishes the contract turns away a unit; the next is counted as today's
KINDS = ("missing", "empty", "parse", "column", "nan", "blank", "norows")
ARMED = ("missing", "empty", "parse", "column", "nan")  # the kinds the switch answers a finish on: `blank` and
#                                         `norows` flagged passing finishes as often as failing ones in the replay
TIES = ("sentence", "block", "any")
TIE = "sentence"                        # the switch's tie (the brief's conservative rule)
SHOWN = 10                              # findings listed in the contract's answer
MAX_ROWS = 200_000                      # a table's rows read
TABLES = ("csv", "tsv")
HOLDERS = ("csv", "tsv", "json", "jsonl", "parquet", "txt", "md")   # the formats a name is looked for in
TIE_WORDS = re.compile(r"\b(?:columns?|keys?|fields?|headers?)\b", re.I)
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_ .%\-]*[A-Za-z0-9%]")
MODULES = frozenset({*sys.stdlib_module_names, "np", "pd", "qf", "df", "numpy", "pandas", "scipy", "sklearn",
                     "statsmodels", "arch", "dt", "plt", "work", "self"})
LITERALS = frozenset({"nan", "na", "n/a", "null", "none", "true", "false", "inf", "infinity", "yes", "no", "str",
                      "string", "float", "float32", "float64", "int", "int32", "int64", "bool", "boolean", "object",
                      "datetime64", "dict", "list"})     # values and type names, never a column's
EXPRESSION = re.compile(r"\s[-+*/%]\s")    # `a - b`: an expression, never a name
COMPOSED = re.compile(r"[A-Z0-9]+(?:[\s_\-/]+[A-Z0-9]+)+")    # `AAA-BBB`, `YYYY-MM-DD`: values or a placeholder
NEGATION = re.compile(r"\b(?:not|never|without|no longer)\b|n't\b", re.I)
ASIDE = re.compile(r"(?:\b(?:not|instead of|rather than|e\.g\.|eg|i\.e\.|for example|for instance|such as|one of|"
                   r"values?(?:\s+(?:is|are|of))?|placeholders?|patterns?|formats?|templates?|set to)"
                   r"[\s,:(]*(?:[`\"'“‘][^`\"'”’]*[`\"'”’][\s,]*(?:or|and)?[\s,]*)*)$", re.I)
FILLER = frozenset({"and", "or", "then", "the", "a", "an", "named", "called", "with", "plus", "also", "its", "their",
                    "following", "these", "this", "new", "extra", "additional", "output", "same"})
INPUT_HEADING = re.compile(r"\b(?:inputs?|data)\b", re.I)
OUTPUT_WORD = re.compile(r"\boutputs?\b", re.I)
BACKTICK = re.compile(r"`([^`\n]{1,80})`")
QUOTED = re.compile(r"(?<![\w])(?:\"([^\"\n]{1,60})\"|“([^”\n]{1,60})”|'([^'\n]{1,60})'|‘([^’\n]{1,60})’)(?![\w])")
PAREN = re.compile(r"\([^()]*\)")
INPUT_BYTES = 4_000_000                 # an input JSON file read for its keys at most this large
INPUT_FILES = 64                        # input files read for their names
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(*\[])")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
TABLE_ROW = re.compile(r"^\s*\|")
HEADING = re.compile(r"^\s*#{1,6}\s")
FENCE = re.compile(r"^\s*(```|~~~)")
INPUT_PATH = re.compile(r"(?<![\w.~$-])/(?:input|app/data|data)(?:/|\b)")
NONFINITE = contract.NONFINITE


@dataclass(frozen=True)
class Named:
    name: str                   # as the instruction writes it
    files: tuple[str, ...]      # the output files its context names; () for any output file


@dataclass(frozen=True)
class Finding:
    kind: str                   # one of KINDS
    file: str                   # the output file it is about; "" for a name no file holds
    line: str                   # one plain line for the model


# ------------------------------------------------------------------------------------------------- the instruction


@dataclass(frozen=True)
class Context:
    text: str                   # the sentence
    lead: str                   # the line its list, table or code block hangs from ("" where none)
    heading: str                # the nearest heading above it ("" where none)


def blocks(instruction: str) -> list[tuple[str, str]]:
    """The instruction's logical lines, each with its kind: a heading, a table row and a code line (inside a fence)
    as written; a paragraph or a list item with the wrapped lines that follow it joined by a space."""
    found: list[tuple[str, str]] = []
    fenced = False
    for line in instruction.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            found.append(("fence", ""))
            continue
        if fenced:
            if line.strip():
                found.append(("code", line.strip()))
            continue
        if not line.strip():
            found.append(("blank", ""))
        elif HEADING.match(line):
            found.append(("heading", line.strip()))
        elif TABLE_ROW.match(line):
            found.append(("row", line.strip()))
        elif LIST_ITEM.match(line):
            found.append(("item", line.strip()))
        elif found and found[-1][0] in ("paragraph", "item"):
            found[-1] = (found[-1][0], f"{found[-1][1]} {line.strip()}")
        else:
            found.append(("paragraph", line.strip()))
    return found


def contexts(instruction: str) -> list[Context]:
    """The instruction's sentences (a paragraph's or a list item's split at their ends; a heading, a table row or a
    code line whole), each with its lead and its heading: a list item, a table row or a code line leads from the last
    sentence of the paragraph above its list, table or block; a heading leads nothing but itself."""
    found: list[Context] = []
    heading = lead = ""
    for kind, text in blocks(instruction):
        if kind in ("blank", "fence"):
            continue
        if kind == "heading":
            heading, lead = text, ""
            found.append(Context(text, "", ""))
            continue
        parts = SENTENCE_END.split(text) if kind in ("paragraph", "item") else [text]
        for part in parts:
            if part.strip():
                found.append(Context(part.strip(), lead if kind != "paragraph" else "", heading))
        if kind == "paragraph":
            lead = parts[-1].strip()
    return found


@dataclass(frozen=True)
class Span:
    text: str                   # as written between the marks
    start: int                  # where the marks open and close in the sentence
    end: int


def spans(text: str) -> list[Span]:
    """The text's backticked spans and its quoted ones outside them, in order."""
    found = [Span(match.group(1), match.start(), match.end()) for match in BACKTICK.finditer(text)]
    rest = BACKTICK.sub(lambda match: " " * len(match.group(0)), text)
    for match in QUOTED.finditer(rest):
        found.append(Span(next(group for group in match.groups() if group is not None), match.start(), match.end()))
    return sorted(found, key=lambda span: span.start)


def a_name(span: str) -> str | None:
    """A span that can be a column or a key: an identifier-like name (a letter first, a letter, digit or `%` last,
    words, digits and `_ . % -` between, at most four words), never a file name, a literal (`NaN`, `null`, a type's
    name), a composition of capitals (`AAA-BBB`, `YYYY-MM-DD`: values or a placeholder), an expression (`a - b`) or a
    module's attribute."""
    name = span.strip()
    if len(name) < 2 or len(name) > 48 or not NAME.fullmatch(name) or len(name.split()) > 4:
        return None
    if name.lower() in LITERALS or COMPOSED.fullmatch(name) or EXPRESSION.search(name):
        return None
    if paths.BARE.fullmatch(name) or name.lower().endswith(paths.SCRIPTS):
        return None
    if "." in name and name.split(".", 1)[0] in MODULES:
        return None
    return name


def aside(text: str, span: Span) -> bool:
    """Whether the span is introduced as an example, value, format or placeholder (other marked spans may follow
    in the same list), or is under a negation anywhere before it in its sentence."""
    return ASIDE.search(text[:span.start]) is not None or NEGATION.search(text[:span.start]) is not None


def governed(text: str, span: Span, others: Sequence[Span]) -> bool:
    """Whether a tie word ("column", "key", "field", "header") governs the span: next to it, or opening the list it
    sits in, with only other marked spans, parenthesised asides, punctuation and filler words (`FILLER`, at most
    three) between."""
    masked = list(text)
    for other in others:
        masked[other.start:other.end] = " " * (other.end - other.start)
    masked = "".join(masked)
    for match in TIE_WORDS.finditer(masked):
        if match.end() <= span.start:
            between = masked[match.end():span.start]
        elif match.start() >= span.end:
            between = masked[span.end:match.start()]
        else:
            continue
        words = re.findall(r"[A-Za-z]+", PAREN.sub(" ", between))
        if len(words) <= 3 and all(word.lower() in FILLER for word in words):
            return True
    return False


def files_in(text: str, names: Iterable[str]) -> list[str]:
    """The files of `names` the text names: the whole name or its last part, never inside a longer name."""
    found = []
    for name in names:
        base = name.rsplit("/", 1)[-1]
        if any(re.search(r"(?<![\w.-])" + re.escape(each) + r"(?![\w-])", text) for each in {name, base}):
            found.append(name)
    return found


def tie_of(context: Context, span: Span, others: Sequence[Span], named: Sequence[str], inputs: Sequence[str],
           tie: str) -> tuple[str, tuple[str, ...]]:
    """What the context ties a name to: `output` (with the output files its sentence or its block names, () for
    any), `input` or `none`. The sentence naming an output file ties it there; else a sentence, lead or heading naming
    an input file or path, or a heading on the inputs or the data and not the output, ties it to the input; else a
    tie word governing it in its sentence (`governed`), or under `block` its lead or its heading naming an output
    file or a tie word, ties it to the output."""
    outputs = files_in(context.text, named)
    if outputs:
        return "output", tuple(outputs)
    around = [text for text in (context.text, context.lead, context.heading) if text]
    if any(files_in(text, inputs) or INPUT_PATH.search(text) for text in around) or \
            (INPUT_HEADING.search(context.heading) and not OUTPUT_WORD.search(context.heading)):
        return "input", ()
    if governed(context.text, span, others):
        return "output", ()
    if tie == "block":
        for text in (context.lead, context.heading):
            outputs = files_in(text, named) if text else []
            if outputs:
                return "output", tuple(outputs)
            if text and TIE_WORDS.search(text):
                return "output", ()
    return "none", ()


def stated(instruction: str, named: Sequence[str], inputs: Sequence[str] = (), tie: str = TIE,
           held: Iterable[str] = ()) -> tuple[Named, ...]:
    """The names the instruction ties to an output (`tie`: `sentence`, `block` or `any`), once each, in order, each
    with the files of its first tie: never a name said as an example or a negation (`aside`), nor one the input files
    hold (`held`, `input_names`), nor an output file's own name. Under `any` every name counts, those rules aside."""
    if tie not in TIES:
        raise ValueError(f"tie {tie!r}: one of {', '.join(TIES)}")
    named = [name for name in named if name]
    inputs = [name for name in inputs if name and name not in named]
    held = {name.lower() for name in held}
    bases = {each.rsplit("/", 1)[-1] for each in named} | set(named)
    found: dict[str, tuple[str, ...]] = {}
    for context in contexts(instruction or ""):
        marked = spans(context.text)
        for span in marked:
            name = a_name(span.text)
            if name is None or name in found or name in bases:
                continue
            if tie == "any":
                found[name] = ()
                continue
            if name.lower() in held or aside(context.text, span):
                continue
            kind, files = tie_of(context, span, [each for each in marked if each is not span], named, inputs, tie)
            if kind == "output":
                found[name] = files
    return tuple(Named(name, files) for name, files in found.items())


def input_names(data_dir: Path) -> frozenset[str]:
    """The names the input files hold: each table's header, each JSON file's keys at any depth (a file up to
    `INPUT_BYTES`), each JSONL file's first lines' keys; `INPUT_FILES` files at most, an unreadable one skipped."""
    data_dir, found = Path(data_dir), set()
    files = sorted(path for path in data_dir.rglob("*") if path.is_file())[:INPUT_FILES] if data_dir.is_dir() else []
    for path in files:
        fmt = contract.fmt_of(path.name)
        try:
            if fmt in TABLES:
                with open(path, encoding="utf-8-sig", errors="replace", newline="") as handle:
                    header = next(csv.reader(handle, delimiter="\t" if fmt == "tsv" else ","), [])
                found.update(cell.strip() for cell in header if cell.strip())
            elif fmt == "json" and path.stat().st_size <= INPUT_BYTES:
                found.update(keys(json.loads(path.read_text(encoding="utf-8", errors="replace"))))
            elif fmt == "jsonl":
                with open(path, encoding="utf-8", errors="replace") as handle:
                    for line in itertools.islice(handle, 200):
                        if line.strip():
                            found.update(keys(json.loads(line)))
        except (OSError, ValueError, csv.Error):
            continue
    return frozenset(found)


# ------------------------------------------------------------------------------------------------------ the output


@dataclass
class Loaded:
    fmt: str
    header: list[str] | None = None         # a table's
    rows: list[list[str]] | None = None
    docs: list[object] | None = None        # JSON's (one, or a JSONL's lines)
    text: str | None = None                 # a text's
    unchecked: bool = False                 # parquet with no reader installed


def load(path: Path, name: str) -> tuple[Loaded | None, Finding | None]:
    """A named output file read by its extension, or the finding that stops it (missing, empty, parse)."""
    fmt = contract.fmt_of(name)
    if not path.is_file():
        return None, Finding("missing", name, f"{name}: the task names this output file and the output directory "
                                              f"does not hold it")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, Finding("parse", name, f"{name}: could not be read ({type(exc).__name__})")
    if not raw.strip():
        return None, Finding("empty", name, f"{name}: the file is empty")
    if fmt == "parquet":
        return parquet(path, name)
    if fmt not in HOLDERS:
        return Loaded(fmt, unchecked=True), None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, Finding("parse", name, f"{name}: not UTF-8 text")
    if fmt in TABLES:
        return table(text, name, fmt)
    if fmt == "json":
        try:
            doc = json.loads(text)
        except ValueError:
            return None, Finding("parse", name, f"{name}: does not parse as JSON")
        if doc in ({}, [], "", None):
            return None, Finding("empty", name, f"{name}: holds an empty JSON value")
        return Loaded(fmt, docs=[doc]), None
    if fmt == "jsonl":
        docs = []
        for line in text.splitlines():
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except ValueError:
                    return None, Finding("parse", name, f"{name}: does not parse as JSONL")
        return Loaded(fmt, docs=docs), None
    return Loaded(fmt, text=text), None


def table(text: str, name: str, fmt: str) -> tuple[Loaded | None, Finding | None]:
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t" if fmt == "tsv" else ",", strict=True)
    try:
        header = next(reader, None)
        rows = [row for row in itertools.islice(reader, MAX_ROWS) if any(cell.strip() for cell in row)]
    except csv.Error:
        return None, Finding("parse", name, f"{name}: does not parse as {fmt.upper()}")
    if not header or not any(cell.strip() for cell in header):
        return None, Finding("parse", name, f"{name}: does not parse as {fmt.upper()} with a header row")
    if not rows:
        return None, Finding("norows", name, f"{name}: a header and no rows")
    return Loaded(fmt, header=[cell.strip() for cell in header], rows=rows), None


def parquet(path: Path, name: str) -> tuple[Loaded | None, Finding | None]:
    """A parquet file's columns and rows as text, by pandas or else pyarrow; unchecked where neither imports."""
    try:
        import pandas
    except ImportError:
        pandas = None
    try:
        if pandas is not None:
            frame = pandas.read_parquet(path)
            header = [str(column) for column in frame.columns]
            rows = [["" if value is None else str(value) for value in row]
                    for row in frame.head(MAX_ROWS).itertuples(index=False)]
        else:
            import pyarrow.parquet
            got = pyarrow.parquet.read_table(path).slice(0, MAX_ROWS).to_pydict()
            header = [str(column) for column in got]
            rows = [["" if value is None else str(value) for value in row] for row in zip(*got.values())]
    except ImportError:
        return Loaded("parquet", unchecked=True), None
    except Exception as exc:    # noqa: BLE001 - a reader's error is the file's, said as one line
        return None, Finding("parse", name, f"{name}: does not read as parquet ({type(exc).__name__})")
    if not rows:
        return None, Finding("norows", name, f"{name}: no rows")
    return Loaded("parquet", header=header, rows=rows), None


def keys(doc: object) -> Iterable[str]:
    """Every key of a JSON value at any depth."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            yield str(key)
            yield from keys(value)
    elif isinstance(doc, list):
        for value in doc:
            yield from keys(value)


def strings(doc: object) -> Iterable[str]:
    """Every string value of a JSON value at any depth."""
    if isinstance(doc, str):
        yield doc.strip()
    elif isinstance(doc, dict):
        for value in doc.values():
            yield from strings(value)
    elif isinstance(doc, list):
        for value in doc:
            yield from strings(value)


def values(doc: object, key: str) -> Iterable[object]:
    """Every value under `key` at any depth (a dotted key's last part)."""
    if isinstance(doc, dict):
        for name, value in doc.items():
            if name == key:
                yield value
            yield from values(value, key)
    elif isinstance(doc, list):
        for value in doc:
            yield from values(value, key)


def flat(value: object) -> Iterable[object]:
    """A value's scalars: itself, or each item of a list (nested lists flattened)."""
    if isinstance(value, list):
        for item in value:
            yield from flat(item)
    else:
        yield value


def number(cell: str) -> bool:
    try:
        float(cell)
    except ValueError:
        return False
    return True


def fold(name: str) -> str:
    """A name up to case and separators."""
    return re.sub(r"[\s_\-.]+", "", name).lower()


def holds(loaded: Loaded, name: str) -> str | None:
    """The name as a read file holds it, where it does: a table's header (equal up to case and separators) or one of
    its cells, a JSON key at any depth (a dotted name's whole or last part, the same way) or a string value, a text's
    words; None where it does not."""
    wanted = {fold(name), fold(name.rsplit(".", 1)[-1])}
    if loaded.header is not None:
        column = next((each for each in loaded.header if fold(each) in wanted), None)
        if column is not None:
            return column
        return name if any(name == cell.strip() for row in loaded.rows for cell in row) else None
    if loaded.docs is not None:
        key = next((each for doc in loaded.docs for each in keys(doc) if fold(each) in wanted), None)
        if key is not None:
            return key
        return name if any(name == text for doc in loaded.docs for text in strings(doc)) else None
    if loaded.text is not None:
        return name if re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", loaded.text) else None
    return None


def bad_values(loaded: Loaded, name: str) -> tuple[bool, bool]:
    """Whether the values under a name as the file holds it (a table's column, a JSON key's) hold NaN or infinity,
    and whether a table's column of numbers holds an empty cell or a JSON key holds null."""
    if loaded.header is not None:
        if name not in loaded.header:
            return False, False
        index = loaded.header.index(name)
        cells = [row[index].strip() if index < len(row) else "" for row in loaded.rows]
        nonfinite = any(cell.lower() in NONFINITE for cell in cells)
        numeric = any(number(cell) and cell.lower() not in NONFINITE for cell in cells)
        return nonfinite, numeric and any(not cell for cell in cells)
    if loaded.docs is not None:
        found = [item for doc in loaded.docs for value in values(doc, name) for item in flat(value)]
        nonfinite = any(isinstance(item, float) and not math.isfinite(item) for item in found)
        return nonfinite, any(item is None for item in found)
    return False, False


def check(out: Path, named: Sequence[str], names: Sequence[Named], kinds: Iterable[str] = ARMED) -> list[Finding]:
    """The output directory against the contract: each named file read (missing, empty, parse), then each counted
    name looked for in every named file of a format that holds names (`holds`; none is looked for while one of them
    is missing, unreadable or unchecked) and its values read (`nan`, `blank`); the findings of `kinds` alone, in that
    order. A name's finding names the files its context names, else every such file."""
    out, kinds = Path(out), set(kinds)
    findings: list[Finding] = []
    read: dict[str, Loaded] = {}
    for name in named:
        loaded, finding = load(out / name, name)
        if finding is not None:
            findings.append(finding)
        elif loaded is not None:
            read[name] = loaded
    holders = [name for name in named if contract.fmt_of(name) in HOLDERS]
    whole = holders and all(name in read and not read[name].unchecked for name in holders)
    for one in names if whole else ():
        holding = [(name, found) for name in holders if (found := holds(read[name], one.name)) is not None]
        if not holding:
            could = [name for name in one.files if name in holders] or holders
            findings.append(Finding("column", "", f"`{one.name}`: the task names it as a column or key of "
                                                  f"{', '.join(could)} and no output file holds it"))
            continue
        for name, found in holding:
            nonfinite, blank = bad_values(read[name], found)
            if nonfinite:
                findings.append(Finding("nan", name, f"{name}: `{found}` holds NaN or infinite values"))
            elif blank:
                findings.append(Finding("blank", name, f"{name}: `{found}` holds empty values among its numbers"))
    return [finding for finding in findings if finding.kind in kinds]


def flags(findings: Iterable[Finding]) -> str:
    """The kinds found, in `KINDS` order, joined; `-` where none."""
    found = {finding.kind for finding in findings}
    return ",".join(kind for kind in KINDS if kind in found) or "-"
