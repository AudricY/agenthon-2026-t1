"""Works out which output files the instruction names, and checks that each exists, parses and holds finite numbers.

Step: 1 (`named_all`) and 4 (`check`, after every run and at a finish).
Called by: agent/n11/solve.py (`named_all`, `snapshot`) and agent/n11/commands.py (`check` in `Workspace.checks`,
    `listing`).
Start at: `named_all` and `check`.

The rules in detail
-------------------
The output checks: the output directory read against the output files the instruction names.

`named_all` reads which output files the instruction names. `check` reads each of them in the output directory:
present, parsed in its format (JSON and JSON Lines, CSV and TSV, parquet where pyarrow is installed, text, HTML,
xlsx), its numbers finite; then that a file sits at the output's root, and that no file holds one of the unit's
canaries (the close would remove it). Each failure is one line of text the model is shown. `snapshot` and `listing`
say which files a run wrote.

Pure functions over the output directory, standard library only; pyarrow for parquet where the image has it.
"""
from __future__ import annotations

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

from . import canary, paths
from .guard import SCORER_FILES

ALIASES = {"htm": "html", "pqt": "parquet", "text": "txt", "markdown": "md", "ndjson": "jsonl", "xls": "xlsx"}
MAX_ROWS = 200_000          # a table's rows read for the value checks
NONFINITE = {"nan", "-nan", "+nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"}
EXTENSION = re.compile(r"\.([A-Za-z0-9]+)$")


@dataclass(frozen=True)
class FileSpec:
    name: str                               # relative to the output directory
    format: str


@dataclass(frozen=True)
class Contract:
    files: tuple[FileSpec, ...]


@dataclass(frozen=True)
class Failure:
    kind: str       # missing, empty, parse, nonfinite, canary or nofile
    text: str


def fmt_of(name: str) -> str:
    """The file's format: its extension, an alias read as the format it names; "txt" where it has none."""
    found = EXTENSION.search(name)
    ext = ALIASES.get(found.group(1).lower(), found.group(1).lower()) if found else ""
    return ext or "txt"


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
    prices.csv", "`summary.json`"), the data files' names (`inputs`) left out, once each, the path-named first."""
    found = list(named(pathmap))
    for name in paths.outputs(instruction, inputs):
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
    """The regular files at the output's root besides the scorer's own."""
    try:
        return sorted(path.name for path in Path(out).iterdir()
                      if path.is_file() and not path.is_symlink() and path.name not in SCORER_FILES)
    except OSError:
        return []


def check(found: Contract, out: Path, canaries: Iterable[str] = ()) -> list[Failure]:
    """The output directory against the files `found` lists; each failure one line."""
    out, failures = Path(out), []
    for one in found.files:
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


def json_file(one: FileSpec, path: Path) -> list[Failure]:
    text = path.read_bytes().decode("utf-8")
    try:
        doc = [json.loads(line) for line in text.splitlines() if line.strip()] if one.format == "jsonl" \
            else json.loads(text)
    except ValueError as exc:
        where = f" at line {exc.lineno} column {exc.colno}" if isinstance(exc, json.JSONDecodeError) else ""
        return [Failure("parse", f"{one.name}: not valid JSON ({getattr(exc, 'msg', exc)}{where})")]
    bad = [where for where, value in numbers(doc) if isinstance(value, float) and not math.isfinite(value)]
    if bad:
        return [Failure("nonfinite", f"{one.name}: {len(bad)} non-finite number(s) (NaN or infinity), first "
                                     f"at `{bad[0] or '(the top)'}`")]
    return []


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
    failures = []
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
    failures = []
    for name, column in zip(header, table.columns):
        if pa.types.is_floating(column.type) and len(column):
            bad = len(column) - column.null_count - pc.sum(pc.is_finite(column)).as_py()
            if bad:
                failures.append(Failure("nonfinite", f"{one.name}: column `{name}` holds {bad} non-finite value(s) "
                                                     f"(NaN or infinity)"))
    return failures
