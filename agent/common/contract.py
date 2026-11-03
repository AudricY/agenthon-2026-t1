"""The contract: the output an agent says its script writes, read from its reply, and the output directory checked
against it.

A reply states its contract in a fenced JSON block (agent/l1 asks for it before the script):

    {"files": [{"name": "results.json", "format": "json", "keys": ["price", "greeks.delta", "legs.*.pv"],
                "dtypes": {"price": "float"}, "rounding": {"price": 4}},
               {"name": "trades.csv", "format": "csv", "columns": ["date", "qty"], "dtypes": {"qty": "int"}}],
     "checks": ["files['results.json']['price'] > 0"]}

A key is a dotted path into the JSON (`*` for every item of a list or an object; a key holding a dot is matched
whole); a column is a table's header name; `rounding` gives decimal places, `"*"` for every number in the file.
`parse` reads the block leniently (trailing commas, Python literals) and says what it could not read. `check` reads
the output directory against the contract and against the output files the instruction itself names by path
(`named`): each file present, parsed in its format, its keys or columns present (columns in the contract's order),
its numbers finite, its dtypes and rounding kept, a file at the output's root, no file holding one of the unit's
canaries (the close would remove it). Each failure is one line of text a repair can quote. The model's own `checks`
are code: `program` writes the script that evaluates them over the output in a process of the runner's, and
`verdicts` reads what it wrote. No finance check is ours: the invariants are the model's to state.

Pure functions over the output directory and the reply's text, standard library only; pyarrow for parquet where
the image has it.
"""
from __future__ import annotations

import ast
import csv
import io
import itertools
import json
import math
import posixpath
import re
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from . import canary, runner
from .digest import DATE
from .guard import SCORER_FILES

FORMATS = ("json", "jsonl", "csv", "tsv", "parquet", "txt", "md", "html", "xlsx")
ALIASES = {"htm": "html", "pqt": "parquet", "text": "txt", "markdown": "md", "ndjson": "jsonl", "xls": "xlsx"}
DTYPES = {"float": "float", "double": "float", "number": "float", "numeric": "float", "decimal": "float",
          "float64": "float", "int": "int", "integer": "int", "int64": "int", "str": "str", "string": "str",
          "text": "str", "bool": "bool", "boolean": "bool", "date": "date", "datetime": "date", "timestamp": "date",
          "list": "list", "array": "list", "object": "object", "dict": "object"}
JSON_TAGS = ("json", "jsonc", "json5")
MAX_CHECKS = 5
CHECK_CHARS = 400           # a check longer than this is not a short expression
MAX_ROWS = 200_000          # a table's rows read for the value checks
SHOWN = 8                   # names listed where a key or column is missing
NOISE = 1e-12               # a rounded number's float error, relative to it: a sum's noise, never a decimal
NONFINITE = {"nan", "-nan", "+nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"}
EXTENSION = re.compile(r"\.([A-Za-z0-9]+)$")
INTEGER = re.compile(r"[+-]?\d+")
COMMENT = re.compile(r'^\s*//[^\n]*$|[ \t]+//[^"\n]*$', re.M)   # a `//` line, or a `//` tail holding no quote
LOADS = ("missing", "empty", "parse")   # a file that cannot be read: the model's checks over it would only echo this


@dataclass(frozen=True)
class FileSpec:
    name: str                               # relative to the output directory
    format: str
    keys: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    dtypes: tuple[tuple[str, str], ...] = ()
    rounding: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Contract:
    files: tuple[FileSpec, ...]
    checks: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.files)


@dataclass(frozen=True)
class Parsed:
    contract: Contract | None
    problems: tuple[Failure, ...] = ()      # what in the block could not be read
    text: str = ""                          # the block as the reply wrote it

    def summary(self) -> str:
        if self.contract is None:
            return "no contract" if not self.text else f"a contract that does not parse ({self.problems[0].text})"
        tail = f", {len(self.problems)} problem(s)" if self.problems else ""
        return f"a contract of {len(self.contract.files)} file(s) and {len(self.contract.checks)} check(s){tail}"


@dataclass(frozen=True)
class Failure:
    kind: str       # contract, named, missing, empty, parse, key, column, order, dtype, nonfinite, rounding, canary,
    text: str       # nofile, check; an agent adds its run's own (exit, timeout)


# ------------------------------------------------------------------------------------------------ the reply's block


def fmt_of(name: str, declared: object = None) -> str:
    """The file's format: its extension's where that is one we read, else the declared one, else the extension."""
    found = EXTENSION.search(name)
    ext = ALIASES.get(found.group(1).lower(), found.group(1).lower()) if found else ""
    said = str(declared).strip().lower().lstrip(".") if isinstance(declared, str) else ""
    said = ALIASES.get(said, said)
    return ext if ext in FORMATS else said if said in FORMATS else ext or said or "txt"


def file_name(raw: str) -> str | None:
    """A contract's file name as a path relative to the output directory; None where it points elsewhere."""
    name = raw.strip().replace("\\", "/")
    for root in ("/app/output/", "/output/"):
        if name.startswith(root):
            name = name[len(root):]
    name = posixpath.normpath(name) if name else ""
    if not name or name in (".", "..") or name.startswith(("/", "../")):
        return None
    return name


def load_block(text: str) -> object:
    """The block's JSON, read leniently: as written, then without `//` comments and trailing commas, then as a Python
    literal; the first error where none reads."""
    try:
        return json.loads(text)
    except ValueError as first:
        cleaned = re.sub(r",\s*([}\]])", r"\1", COMMENT.sub("", text))
        try:
            return json.loads(cleaned)
        except ValueError:
            pass
        try:
            return ast.literal_eval(cleaned.strip())
        except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
            raise first from None


def array(text: str, key: str) -> str | None:
    """The bracketed array after `"key":` in `text`, its brackets matched outside strings; None where it never
    closes."""
    found = re.search(rf'"{key}"\s*:\s*\[', text)
    if not found:
        return None
    depth, quoted, escaped = 0, False, False
    for index in range(found.end() - 1, len(text)):
        char = text[index]
        if quoted:
            escaped, quoted = (False, True) if escaped else (char == "\\", char != '"')
        elif char == '"':
            quoted = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[found.end() - 1:index + 1]
    return None


def check_lines(text: str) -> list[str]:
    """The checks one a line where their array is no JSON (a check quoting a name in double quotes): each line's text
    between its first and last double quote, up to the line that closes the array."""
    found = re.search(r'"checks"\s*:\s*\[', text)
    checks = []
    for line in text[found.end():].splitlines() if found else ():
        raw = line.strip()
        body = re.sub(r"[\s,\]}]+$", "", raw)
        if len(body) >= 2 and body[0] == body[-1] == '"':
            checks.append(body[1:-1].replace('\\"', '"'))
        if "]" in raw[len(body):]:
            break
    return checks


def salvage(text: str) -> dict | None:
    """A block that is no JSON as a whole, its `files` and `checks` arrays read apart; None where the files cannot
    be."""
    try:
        files = load_block(array(text, "files") or "")
    except ValueError:
        return None
    if not isinstance(files, list):
        return None
    try:
        checks = load_block(array(text, "checks") or "")
    except ValueError:
        checks = None
    return {"files": files, "checks": checks if isinstance(checks, list) else check_lines(text)}


def strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if isinstance(item, (str, int, float))] if isinstance(value, list) else []


def spec(entry: object, problems: list[Failure]) -> FileSpec | None:
    if isinstance(entry, str):
        entry = {"name": entry}
    alias = next((entry[key] for key in ("file", "filename", "path") if isinstance(entry.get(key), str)), None) \
        if isinstance(entry, dict) and not isinstance(entry.get("name"), str) else None
    if alias is not None:
        entry = {**entry, "name": alias}
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
        problems.append(Failure("contract", f"the contract's file entry {short(entry)} has no \"name\""))
        return None
    name = file_name(entry["name"])
    if name is None:
        problems.append(Failure("contract", f"the contract's file `{entry['name']}` is not inside the output "
                                            f"directory"))
        return None
    dtypes = []
    for field, kind in (entry.get("dtypes") or {}).items() if isinstance(entry.get("dtypes"), dict) else ():
        known = DTYPES.get(str(kind).strip().lower())
        if known is None:
            problems.append(Failure("contract", f"{name}: the dtype `{kind}` of `{field}` is none of float, int, str, "
                                                f"bool, date, list or object"))
        else:
            dtypes.append((str(field), known))
    rounding = []
    for field, places in (entry.get("rounding") or {}).items() if isinstance(entry.get("rounding"), dict) else ():
        if isinstance(places, bool) or not isinstance(places, (int, float)) or not 0 <= places <= 15 \
                or int(places) != places:
            problems.append(Failure("contract", f"{name}: the rounding of `{field}` is not a number of decimal places "
                                                f"from 0 to 15"))
        else:
            rounding.append((str(field), int(places)))
    return FileSpec(name, fmt_of(name, entry.get("format")), tuple(strings(entry.get("keys"))),
                    tuple(strings(entry.get("columns"))), tuple(dtypes), tuple(rounding))


def expressions(value: object, problems: list[Failure]) -> tuple[str, ...]:
    checks = []
    for number, text in enumerate(strings(value)[:MAX_CHECKS], 1):
        text = text.strip()
        if len(text) > CHECK_CHARS:
            problems.append(Failure("contract", f"check {number} is over {CHECK_CHARS} characters: a check is one "
                                                f"short expression"))
            continue
        try:
            compile(text, "<check>", "eval")
        except (SyntaxError, ValueError) as exc:
            problems.append(Failure("contract", f"check {number}, `{text}`, is not a Python expression "
                                                f"({type(exc).__name__}: {getattr(exc, 'msg', exc)})"))
            continue
        checks.append(text)
    return tuple(checks)


def contract_block(reply: str) -> str | None:
    """The reply's contract block: its first JSON-tagged block, else the first untagged one holding `"files"`."""
    found = [block for block in runner.blocks(runner.THINK.sub("", reply)) if block.text.strip()]
    for block in found:
        if block.tag in JSON_TAGS:
            return block.text
    for block in found:
        if not block.tag and '"files"' in block.text:
            return block.text
    return None


def parse(reply: str) -> Parsed:
    """The contract a reply states, with what could not be read; `Parsed.contract` is None where nothing could."""
    text = contract_block(reply)
    if text is None:
        return Parsed(None, (Failure("contract", "the reply held no ```json contract block"),))
    try:
        doc = load_block(text)
    except ValueError as exc:
        doc = salvage(text)
        if doc is None:
            where = f" at line {exc.lineno} column {exc.colno}" if isinstance(exc, json.JSONDecodeError) else ""
            reason = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            return Parsed(None, (Failure("contract", f"the contract block is not valid JSON ({reason}{where})"),),
                          text)
    if isinstance(doc, list):
        doc = {"files": doc}
    if not isinstance(doc, dict) or not isinstance(doc.get("files"), list):
        return Parsed(None, (Failure("contract", "the contract is not an object with a \"files\" list"),), text)
    problems: list[Failure] = []
    files: dict[str, FileSpec] = {}
    for entry in doc["files"]:
        made = spec(entry, problems)
        if made is not None and made.name not in files:
            files[made.name] = made
    if not files:
        problems.append(Failure("contract", "the contract lists no output file"))
    checks = expressions(doc.get("checks", []), problems)
    if not checks and not any(f.text.startswith("check ") for f in problems) and re.search(r'"checks"\s*:\s*\[\s*"',
                                                                                          text):
        problems.append(Failure("contract", "the checks could not be read: write each as a JSON string with single "
                                            "quotes inside"))
    return Parsed(Contract(tuple(files.values()), checks) if files else None, tuple(problems), text)


# --------------------------------------------------------------------------------------------- the output directory


def named(pathmap) -> tuple[str, ...]:
    """The output files the instruction names by path (`/app/output/<name>`, `/output/<name>`, as the path map reads
    them), relative to the output directory: names with no extension (a directory, or a pattern the path's reading
    cut short) and the scorer's own files left out."""
    out, found = Path(pathmap.out), []
    for entry in pathmap.entries:
        if entry.kind != "output" or entry.path is None or Path(entry.path) == out:
            continue
        try:
            name = Path(entry.path).relative_to(out).as_posix()
        except ValueError:
            continue
        if name not in SCORER_FILES and EXTENSION.search(name) and name not in found:
            found.append(name)
    return tuple(found)


def named_all(pathmap, instruction: str, inputs=()) -> tuple[str, ...]:
    """`named` plus the output files the instruction names without a path (common.paths.outputs: "### File 2:
    prices.csv", "`summary.json`"), the data files' names (`inputs`) left out, once each, the path-named first. Four of
    four opening replies carried no contract in L1's probe (its review, 10-07), so the instruction's own names are the
    floor of the check."""
    from . import paths as _paths

    found = list(named(pathmap))
    for name in _paths.outputs(instruction, inputs):
        if name not in found and name not in SCORER_FILES and EXTENSION.search(name):
            found.append(name)
    return tuple(found)


def snapshot(out: Path) -> dict[str, tuple[int, int]]:
    """Every regular file under `out`: its path relative to `out`, its size and its modification time."""
    found = {}
    try:
        for path in Path(out).rglob("*"):
            if path.is_file() and not path.is_symlink():
                st = path.stat()
                found[path.relative_to(out).as_posix()] = (st.st_size, st.st_mtime_ns)
    except OSError:
        pass
    return found


def listing(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]], shown: int = 40) -> str:
    """The files a run wrote (new or changed between the two snapshots), and those left from earlier runs."""
    wrote = sorted(name for name in after if before.get(name) != after[name])
    left = sorted(name for name in after if before.get(name) == after[name])

    def names(which: list[str]) -> str:
        more = f", and {len(which) - shown} more" if len(which) > shown else ""
        return ", ".join(f"{name} ({after[name][0]:,} bytes)" for name in which[:shown]) + more

    text = f"Files the script wrote: {names(wrote)}." if wrote else "The script wrote no file."
    return text + (f" Files left from earlier scripts: {names(left)}." if left else "")


def deliverables(out: Path) -> list[str]:
    """The regular files at the output's root besides the scorer's own: what g1 counts."""
    try:
        return sorted(path.name for path in Path(out).iterdir()
                      if path.is_file() and not path.is_symlink() and path.name not in SCORER_FILES)
    except OSError:
        return []


def check(found: Contract | None, out: Path, named_files: Iterable[str] = (),
          canaries: Iterable[str] = ()) -> list[Failure]:
    """The output directory against the contract and the instruction's named files; each failure one line."""
    out, failures = Path(out), []
    listed = set(found.names) if found else set()
    for name in named_files:
        if name not in listed:
            failures.append(Failure("named", f"{name}: the task names this output file and the contract does not "
                                             f"list it"))
            failures += check_file(FileSpec(name, fmt_of(name)), out)
    for one in found.files if found else ():
        failures += check_file(one, out)
    if not deliverables(out):
        failures.append(Failure("nofile", "the output directory holds no file at its root"))
    marked = [c for c in canaries if c]
    for name in sorted(snapshot(out)) if marked else ():
        try:
            held = canary.file_holds(out / name, marked)
        except OSError:
            held = False
        if held:
            failures.append(Failure("canary", f"{name}: holds a watermark string (`<canary>` in the task or the data),"
                                              f" so it would be deleted; never write it"))
    return failures


def check_file(one: FileSpec, out: Path) -> list[Failure]:
    path = out / one.name
    if not path.is_file():
        elsewhere = sorted(name for name in snapshot(out) if posixpath.basename(name) == posixpath.basename(one.name))
        hint = f" (found at {', '.join(elsewhere[:3])})" if elsewhere else ""
        return [Failure("missing", f"{one.name}: missing from the output directory{hint}")]
    try:
        if path.stat().st_size == 0:
            return [Failure("empty", f"{one.name}: empty")]
        if one.format in ("json", "jsonl"):
            return json_file(one, path)
        if one.format in ("csv", "tsv"):
            return table_file(one, path)
        if one.format == "parquet":
            return parquet_file(one, path)
        if one.format in ("txt", "md", "html"):
            text = path.read_bytes().decode("utf-8")
            if one.format == "html" and not re.search(r"<[A-Za-z!]", text):
                return [Failure("parse", f"{one.name}: holds no HTML tag")]
            return []
        if one.format == "xlsx" and not zipfile.is_zipfile(path):
            return [Failure("parse", f"{one.name}: not an xlsx workbook")]
    except UnicodeDecodeError:
        return [Failure("parse", f"{one.name}: not UTF-8 text")]
    except OSError as exc:
        return [Failure("parse", f"{one.name}: could not be read ({type(exc).__name__})")]
    return []


# ------------------------------------------------------------------------------------------------------- json files


def lookup(doc: object, key: str) -> tuple[list[tuple[str, object]], str | None]:
    """The values at a key path, each with where it is; or, where the path breaks, why."""
    found: list[tuple[str, object]] = []

    def walk(value: object, parts: list[str], where: str) -> str | None:
        if not parts:
            found.append((where, value))
            return None
        if parts[0] == "*":
            items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else None
            if items is None:
                return f"`{where or '(the top)'}` is not a list or an object"
            for name, item in items:
                broke = walk(item, parts[1:], f"{where}.{name}" if where else str(name))
                if broke:
                    return broke
            return None
        if isinstance(value, dict):
            for size in range(len(parts), 0, -1):       # a key holding a dot ("0.5") is matched whole first
                name = ".".join(parts[:size])
                if name in value:
                    return walk(value[name], parts[size:], f"{where}.{name}" if where else name)
            has = ", ".join(str(name) for name in list(value)[:SHOWN]) + (", ..." if len(value) > SHOWN else "")
            return f"no key `{parts[0]}`" + (f" under `{where}`" if where else "") + f" (it has: {has or 'nothing'})"
        if isinstance(value, list) and parts[0].isdigit() and int(parts[0]) < len(value):
            return walk(value[int(parts[0])], parts[1:], f"{where}.{parts[0]}" if where else parts[0])
        return f"no key `{parts[0]}`: `{where or '(the top)'}` is {kind_of(value)}"

    broke = walk(doc, [part for part in key.split(".") if part != ""] if key.strip() else [], "")
    return found, broke


def kind_of(value: object) -> str:
    return {dict: "an object", list: "a list", str: "a string", bool: "a boolean", type(None): "null"}.get(
        type(value), "a number")


def numbers(value: object, where: str = "") -> Iterator[tuple[str, float]]:
    """Every number inside `value`, with where it is."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield where, value
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from numbers(item, f"{where}.{name}" if where else str(name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from numbers(item, f"{where}.{index}" if where else str(index))


def scalars(value: object, where: str) -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for name, item in value.items():
            yield from scalars(item, f"{where}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from scalars(item, f"{where}.{index}")
    else:
        yield where, value


def is_type(kind: str, value: object) -> bool:
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "int":
        return (isinstance(value, int) and not isinstance(value, bool)) or \
            (isinstance(value, float) and value.is_integer())
    if kind == "str":
        return isinstance(value, str)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "date":
        return isinstance(value, str) and bool(DATE.match(value.strip()))
    if kind == "list":
        return isinstance(value, list)
    return isinstance(value, dict)


def rounded(value: float, places: int) -> bool:
    return not math.isfinite(value) or abs(value - round(value, places)) <= NOISE * max(1.0, abs(value))


def json_file(one: FileSpec, path: Path) -> list[Failure]:
    text = path.read_bytes().decode("utf-8")
    try:
        doc = [json.loads(line) for line in text.splitlines() if line.strip()] if one.format == "jsonl" \
            else json.loads(text)
    except ValueError as exc:
        where = f" at line {exc.lineno} column {exc.colno}" if isinstance(exc, json.JSONDecodeError) else ""
        return [Failure("parse", f"{one.name}: not valid JSON ({getattr(exc, 'msg', exc)}{where})")]
    failures, broken = [], set()
    for key in dict.fromkeys([*one.keys, *(field for field, _ in one.dtypes),
                              *(field for field, _ in one.rounding if field != "*")]):
        _, broke = lookup(doc, key)
        if broke:
            broken.add(key)
            failures.append(Failure("key", f"{one.name}: {broke}" if broke.startswith("no key") else
                                    f"{one.name}: the key `{key}` cannot be reached: {broke}"))
    bad = [where for where, value in numbers(doc) if isinstance(value, float) and not math.isfinite(value)]
    if bad:
        failures.append(Failure("nonfinite", f"{one.name}: {len(bad)} non-finite number(s) (NaN or infinity), first "
                                             f"at `{bad[0] or '(the top)'}`"))
    for field, kind in one.dtypes:
        if field in broken:
            continue
        values, _ = lookup(doc, field)
        leaves = [(where, item) for where, value in values
                  for where, item in (scalars(value, where) if kind not in ("list", "object") else [(where, value)])]
        wrong = [(where, item) for where, item in leaves if not is_type(kind, item)]
        if wrong:
            where, item = wrong[0]
            failures.append(Failure("dtype", f"{one.name}: `{where or field}` should be {kind}, found {short(item)}"
                                    + (f" ({len(wrong)} such values)" if len(wrong) > 1 else "")))
    for field, places in one.rounding:
        if field in broken:
            continue
        values = [("", doc)] if field == "*" else lookup(doc, field)[0]
        loose = [(where, number) for base, value in values for where, number in numbers(value, base)
                 if not rounded(number, places)]
        if loose:
            where, number = loose[0]
            failures.append(Failure("rounding", f"{one.name}: `{where or field}` should have at most {places} "
                                                f"decimal places, found {number!r}"
                                    + (f" ({len(loose)} such values)" if len(loose) > 1 else "")))
    return failures


# ------------------------------------------------------------------------------------------------------ table files


def cell_type(kind: str, cell: str) -> bool:
    if kind in ("float", "int"):
        try:
            value = float(cell)
        except ValueError:
            return False
        return kind == "float" or bool(INTEGER.fullmatch(cell)) or (math.isfinite(value) and value.is_integer()
                                                                     and "." not in cell)
    if kind == "bool":
        return cell.lower() in ("true", "false", "0", "1")
    if kind == "date":
        return bool(DATE.match(cell))
    return kind == "str"


def as_number(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def columns_failures(one: FileSpec, header: list[str]) -> list[Failure]:
    failures, present = [], [column for column in one.columns if column in header]
    has = ", ".join(header[:SHOWN]) + (", ..." if len(header) > SHOWN else "")
    for column in dict.fromkeys([*one.columns, *(field for field, _ in one.dtypes),
                                 *(field for field, _ in one.rounding if field != "*")]):
        if column not in header:
            failures.append(Failure("column", f"{one.name}: no column `{column}` (it has: {has or 'nothing'})"))
    order = [column for column in header if column in present]
    if order != present:
        failures.append(Failure("order", f"{one.name}: the columns are not in the contract's order "
                                         f"({', '.join(present)})"))
    return failures


def table_file(one: FileSpec, path: Path) -> list[Failure]:
    text = path.read_bytes().decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t" if one.format == "tsv" else ",")
    try:
        header = next(reader, None)
        rows = [row for row in itertools.islice(reader, MAX_ROWS) if row]
    except csv.Error as exc:
        return [Failure("parse", f"{one.name}: not a readable {one.format} ({exc})")]
    if not header:
        return [Failure("parse", f"{one.name}: no header row")]
    header = [name.strip() for name in header]
    failures = columns_failures(one, header)
    ragged = [number for number, row in enumerate(rows, 2) if len(row) != len(header)]
    if ragged:
        failures.append(Failure("parse", f"{one.name}: {len(ragged)} row(s) with a field count other than the "
                                         f"header's {len(header)}, first line {ragged[0]}"))
    cells = {name: [row[index].strip() for row in rows if index < len(row)] for index, name in enumerate(header)}
    for name, values in cells.items():
        bad = sum(value.lower() in NONFINITE for value in values)
        if bad:
            failures.append(Failure("nonfinite", f"{one.name}: column `{name}` holds {bad} non-finite value(s) "
                                                 f"(NaN or infinity)"))
    for field, kind in one.dtypes:
        values = cells.get(field)
        if values is None:
            continue
        filled = [value for value in values if value and value.lower() not in NONFINITE]
        if values and not filled and kind in ("float", "int"):
            failures.append(Failure("dtype", f"{one.name}: column `{field}` should be {kind} and every value is "
                                             f"empty or NaN"))
            continue
        wrong = [value for value in filled if not cell_type(kind, value)]
        if wrong:
            failures.append(Failure("dtype", f"{one.name}: column `{field}` should be {kind}, found {short(wrong[0])}"
                                    + (f" ({len(wrong)} such values)" if len(wrong) > 1 else "")))
    for field, places in one.rounding:
        for name in (list(cells) if field == "*" else [field] if field in cells else []):
            loose = [value for value in cells[name]
                     if (number := as_number(value)) is not None and not rounded(number, places)]
            if loose:
                failures.append(Failure("rounding", f"{one.name}: column `{name}` should have at most {places} "
                                                    f"decimal places, found {loose[0]}"
                                        + (f" ({len(loose)} such values)" if len(loose) > 1 else "")))
    return failures


def parquet_file(one: FileSpec, path: Path) -> list[Failure]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError:                     # no reader in this image: present is all that is known
        return []
    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001 - pyarrow raises its own kinds for a file that is no parquet
        return [Failure("parse", f"{one.name}: not a readable parquet file ({type(exc).__name__})")]
    header = [str(name) for name in table.column_names]
    failures = columns_failures(one, header)
    for name, column in zip(header, table.columns):
        if pa.types.is_floating(column.type) and len(column):
            bad = len(column) - column.null_count - pc.sum(pc.is_finite(column)).as_py()
            if bad:
                failures.append(Failure("nonfinite", f"{one.name}: column `{name}` holds {bad} non-finite value(s) "
                                                     f"(NaN or infinity)"))
    want = {"float": lambda t: pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_decimal(t),
            "int": pa.types.is_integer, "str": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
            "bool": pa.types.is_boolean, "date": lambda t: pa.types.is_temporal(t) or pa.types.is_string(t),
            "list": pa.types.is_list, "object": pa.types.is_struct}
    for field, kind in one.dtypes:
        if field in header:
            kind_type = table.schema.field(field).type
            if not want[kind](kind_type):
                failures.append(Failure("dtype", f"{one.name}: column `{field}` should be {kind}, found {kind_type}"))
    for field, places in one.rounding:
        for name in (header if field == "*" else [field] if field in header else []):
            column = table.column(name)
            if not pa.types.is_floating(column.type):
                continue
            loose = [value for value in column.to_pylist() if value is not None and not rounded(value, places)]
            if loose:
                failures.append(Failure("rounding", f"{one.name}: column `{name}` should have at most {places} "
                                                    f"decimal places, found {loose[0]!r}"
                                        + (f" ({len(loose)} such values)" if len(loose) > 1 else "")))
    return failures


# --------------------------------------------------------------------------------------------- the model's checks


PROGRAM = '''import json, math, os, re
SPEC = json.loads({spec!r})
ns = {{"math": math, "json": json}}
if any(re.search(r"\\bnp\\b", text) for text in SPEC["checks"]):
    import numpy as np
    ns["np"] = np
if any(re.search(r"\\bpd\\b", text) for text in SPEC["checks"]):
    import pandas as pd
    ns["pd"] = pd


class Files(dict):
    def __missing__(self, name):
        path = os.path.join(SPEC["out"], name)
        kind = SPEC["formats"].get(name) or name.rpartition(".")[2].lower()
        if kind == "json":
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
        elif kind == "jsonl":
            with open(path, encoding="utf-8") as handle:
                value = [json.loads(line) for line in handle if line.strip()]
        elif kind in ("csv", "tsv", "parquet", "pqt"):
            import pandas as pd
            value = pd.read_parquet(path) if kind in ("parquet", "pqt") else \\
                pd.read_csv(path, sep="\\t" if kind == "tsv" else ",")
        else:
            with open(path, encoding="utf-8", errors="replace") as handle:
                value = handle.read()
        self[name] = value
        return value


ns["files"] = Files()
said = []
for text in SPEC["checks"]:
    try:
        said.append(["pass" if bool(eval(text, dict(ns))) else "fail", ""])
    except BaseException as exc:
        said.append(["error", (type(exc).__name__ + ": " + str(exc))[:300]])
with open(SPEC["result"], "w", encoding="utf-8") as handle:
    json.dump(said, handle)
'''


def program(found: Contract, out: Path, result: Path) -> str:
    """The script that evaluates the contract's checks over `out` and writes each verdict to `result`: `files[name]`
    is the file's content (parsed JSON, a pandas DataFrame for a table, else text), `np`, `pd` and `math` beside it.
    It runs as the model's script does, in a process of its own under a time cap."""
    spec = json.dumps({"out": str(out), "result": str(result), "checks": list(found.checks),
                       "formats": {one.name: one.format for one in found.files}})
    return PROGRAM.format(spec=spec)


def verdicts(found: Contract, result: Path, why: str = "") -> list[Failure]:
    """The checks that did not hold, each with its text; one failure where the program wrote no verdicts (`why`)."""
    try:
        said = json.loads(Path(result).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [Failure("check", f"your checks could not be evaluated ({why or 'no verdict was written'})")]
    failures = []
    for text, (verdict, detail) in zip(found.checks, said):
        if verdict == "fail":
            failures.append(Failure("check", f"your check `{text}` is False"))
        elif verdict == "error":
            failures.append(Failure("check", f"your check `{text}` raised {detail}"))
    return failures


# ------------------------------------------------------------------------------------------------------ the report


def short(value: object, width: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= width else text[:width - 3] + "..."


def report(failures: Iterable[Failure], limit: int = 3_000) -> str:
    """The failures as lines, cut at `limit` bytes with the rest counted."""
    failures, lines, size = list(failures), [], 0
    for number, failure in enumerate(failures):
        line = f"- {failure.text}"
        if size + len(line.encode()) + 1 > limit:
            lines.append(f"- and {len(failures) - number} more")
            break
        lines.append(line)
        size += len(line.encode()) + 1
    return "\n".join(lines)


def kinds(failures: Iterable[Failure]) -> str:
    """The failures counted by kind, for a log line that quotes no text: `2 key, 1 exit`."""
    counted: dict[str, int] = {}
    for failure in failures:
        counted[failure.kind] = counted.get(failure.kind, 0) + 1
    return ", ".join(f"{count} {kind}" for kind, count in counted.items())
