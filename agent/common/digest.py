"""The data digest: what the unit's data files hold, built by code and bounded in bytes.

Every file under the data directory with its size; then, file by file while the cap lasts, what its kind shows best:
a CSV's or TSV's header, first and last rows and each column's type; a JSON file whole where it is small, else its
key tree; a JSON Lines file's first records; a parquet file's schema, row count and first rows (pyarrow, where the
image has it); a zip's listing; an xlsx's sheets and their first rows (read from the zip by the standard library);
a text file's head. Each file gets an even share of what the cap leaves after the listing, at least `MIN_SHARE`.
A canary of the unit never reaches the digest: it reads `<canary>` (canary.redact).
"""
from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from pathlib import Path

from . import canary

DIGEST_BYTES = 24_000       # the whole digest; measured in runs/lanes/agent-report.md ("The digest's cap")
MIN_SHARE = 1_200           # a file's section never shrinks below this while the cap lasts
LISTED = 120                # files named in the listing; the rest are counted
HEAD_ROWS = 5
CELL = 60                   # a cell or a scalar shown at most this long
LINE = 400                  # a row shown at most this long
SAMPLE = 64 * 1024          # bytes read from a table's head to infer its columns
WHOLE_JSON = 2_000_000      # a JSON file larger than this is not parsed
TEXT = (".txt", ".md", ".py", ".xml", ".html", ".htm", ".yaml", ".yml", ".toml", ".sql", ".r", ".ini", ".cfg")


def digest(data_dir: Path, cap: int = DIGEST_BYTES, canaries: Iterable[str] = ()) -> str:
    data_dir = Path(data_dir)
    files = sorted(path for path in data_dir.rglob("*") if path.is_file()) if data_dir.is_dir() else []
    if not files:
        return f"No input files: `{data_dir}` holds none."
    total = sum(path.stat().st_size for path in files)
    lines = [f"{len(files)} input file{'s' if len(files) != 1 else ''} under `{data_dir}`, {total:,} bytes:"]
    lines += [f"- {path.relative_to(data_dir)} ({path.stat().st_size:,} bytes)" for path in files[:LISTED]]
    if len(files) > LISTED:
        lines.append(f"- ... and {len(files) - LISTED} more")
    text = "\n".join(lines) + "\n"
    share = max(MIN_SHARE, (cap - len(text)) // len(files))
    described = 0
    for path in files:
        room = min(share, cap - len(text.encode()))
        if room < MIN_SHARE // 2:
            break
        section = clip(f"\n### {path.relative_to(data_dir)}\n" + describe(path, room), room)
        text += section
        described += 1
    if described < len(files):
        text += f"\n[{len(files) - described} more file{'s' if len(files) - described != 1 else ''} not described: " \
                f"the digest's cap is {cap:,} bytes]\n"
    return canary.redact(text, canaries)


def clip(text: str, room: int) -> str:
    data = text.encode()
    if len(data) <= room:
        return text
    marker = "\n[... cut]\n"
    return data[:max(0, room - len(marker))].decode(errors="ignore").rsplit("\n", 1)[0] + marker


def describe(path: Path, room: int) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in (".csv", ".tsv"):
            return table(path, "\t" if suffix == ".tsv" else None)
        if suffix == ".json":
            return json_file(path, room)
        if suffix == ".jsonl":
            return head(path, room, "JSON Lines")
        if suffix in (".parquet", ".pqt"):
            return parquet(path)
        if suffix == ".zip":
            return zip_listing(path)
        if suffix in (".xlsx", ".xlsm"):
            return xlsx(path)
        if suffix in TEXT or not binary(path):
            return head(path, room, "text")
        return f"binary, {path.stat().st_size:,} bytes\n"
    except Exception as exc:  # noqa: BLE001 - a file the digest cannot read is named, never fatal
        return f"unreadable here ({type(exc).__name__})\n"


def binary(path: Path) -> bool:
    with open(path, "rb") as handle:
        return b"\0" in handle.read(4096)


def lines_in(path: Path) -> int:
    count, last = 0, b"\n"
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            count += block.count(b"\n")
            last = block[-1:]
    return count + (last != b"\n")


def tail_line(path: Path) -> str:
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 8192))
        lines = [line for line in handle.read().decode("utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def short(value, width: int = CELL) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= width else text[:width - 3] + "..."


DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?([+-]\d{2}:?\d{2}|Z)?)?$|^\d{1,2}/\d{1,2}/\d{2,4}$")


def kind(values: list[str]) -> str:
    """A column's type from its sampled cells: int, float, bool, date, str, or empty."""
    cells = [value.strip() for value in values if value.strip() != ""]
    if not cells:
        return "empty"
    for name, test in (("int", int), ("float", float)):
        try:
            for cell in cells:
                test(cell)
            return name
        except ValueError:
            pass
    if all(cell.lower() in ("true", "false") for cell in cells):
        return "bool"
    if all(DATE.match(cell) for cell in cells):
        return "date"
    return "str"


def table(path: Path, delimiter: str | None) -> str:
    with open(path, "rb") as handle:
        raw = handle.read(SAMPLE)
    sample = raw.decode("utf-8-sig", errors="replace")
    if len(raw) == SAMPLE:
        sample = sample.rsplit("\n", 1)[0]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    rows = [row for row in csv.reader(io.StringIO(sample), delimiter=delimiter) if row]
    if not rows:
        return f"empty table, {path.stat().st_size:,} bytes\n"
    header, body = rows[0], rows[1:]
    count = lines_in(path)
    columns = [f"{name} [{kind([row[i] for row in body if i < len(row)])}]" for i, name in enumerate(header)]
    name = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}.get(delimiter, repr(delimiter))
    out = [f"table, {name}-separated, {count:,} lines with the header; {len(header)} columns (types from the first "
           f"{len(body)} rows): " + ", ".join(columns),
           f"first {min(HEAD_ROWS, len(body))} rows:"]
    out += [short(delimiter.join(row), LINE) for row in rows[:HEAD_ROWS + 1]]
    if count > HEAD_ROWS + 1:
        out.append("last row: " + short(tail_line(path), LINE))
    return "\n".join(out) + "\n"


def tree(value, depth: int = 0, limit: int = 4, keys: int = 30) -> list[str]:
    pad = "  " * depth
    if isinstance(value, dict):
        out = []
        for i, (key, inner) in enumerate(value.items()):
            if i == keys:
                out.append(f"{pad}... {len(value) - keys} more keys")
                break
            if isinstance(inner, (dict, list)) and depth + 1 < limit and inner:
                out.append(f"{pad}{short(str(key))}: {label(inner)}")
                out += tree(inner, depth + 1, limit, keys)
            else:
                out.append(f"{pad}{short(str(key))}: {label(inner)}")
        return out
    if isinstance(value, list) and value and depth < limit:
        first = value[0]
        if isinstance(first, (dict, list)):
            return [f"{pad}[0]: {label(first)}"] + tree(first, depth + 1, limit, keys)
    return []


def label(value) -> str:
    if isinstance(value, dict):
        return f"object, {len(value)} keys"
    if isinstance(value, list):
        if value and not isinstance(value[0], (dict, list)):
            return f"list of {len(value)}, e.g. {short(json.dumps(value[:6]), CELL * 2)}"
        return f"list of {len(value)}"
    return f"{type(value).__name__} {short(json.dumps(value))}"


def json_file(path: Path, room: int) -> str:
    size = path.stat().st_size
    if size > WHOLE_JSON:
        return head(path, room, "JSON (too large to parse here)")
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    whole = pretty(doc)
    if len(whole.encode()) + 40 <= room:
        return f"JSON, shown whole:\n{whole}\n"
    return f"JSON, {size:,} bytes; its key tree ({label(doc)}):\n" + "\n".join(tree(doc)) + "\n"


def pretty(value, depth: int = 0) -> str:
    """JSON indented by its objects and nested lists, a list of scalars kept on one line."""
    pad, inner = "  " * depth, "  " * (depth + 1)
    if isinstance(value, dict) and value:
        return "{\n" + ",\n".join(f"{inner}{json.dumps(key)}: {pretty(item, depth + 1)}"
                                   for key, item in value.items()) + f"\n{pad}}}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        return "[\n" + ",\n".join(inner + pretty(item, depth + 1) for item in value) + f"\n{pad}]"
    return json.dumps(value)


def head(path: Path, room: int, what: str) -> str:
    with open(path, "rb") as handle:
        data = handle.read(room)
    text = data.decode("utf-8", errors="replace")
    size = path.stat().st_size
    if size > len(data):
        text = text.rsplit("\n", 1)[0] + f"\n[... the first {len(data):,} of {size:,} bytes]"
    return f"{what}, {size:,} bytes:\n{text}\n"


def parquet(path: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return f"parquet, {path.stat().st_size:,} bytes (no reader in this image)\n"
    handle = pq.ParquetFile(path)
    schema = handle.schema_arrow
    out = [f"parquet, {handle.metadata.num_rows:,} rows, {len(schema)} columns: "
           + ", ".join(f"{field.name} [{field.type}]" for field in schema)]
    columns = schema.names[:40]
    batch = next(handle.iter_batches(batch_size=HEAD_ROWS, columns=columns), None)
    if batch is not None and batch.num_rows:
        out.append(f"first {batch.num_rows} rows ({', '.join(columns)}):")
        for row in batch.to_pylist():
            out.append(short(", ".join(short(str(row[name]), 24) for name in columns), LINE))
    return "\n".join(out) + "\n"


def zip_listing(path: Path, shown: int = 40) -> str:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    out = [f"zip, {path.stat().st_size:,} bytes, {len(infos)} members:"]
    out += [f"- {info.filename} ({info.file_size:,} bytes)" for info in infos[:shown]]
    if len(infos) > shown:
        out.append(f"- ... and {len(infos) - shown} more")
    return "\n".join(out) + "\n"


MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
DOC_RELS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def xlsx(path: Path, sheets_shown: int = 6) -> str:
    """Sheet names and each sheet's first rows, read from the workbook's XML (no spreadsheet library needed)."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        targets = {}
        if "xl/_rels/workbook.xml.rels" in names:
            for rel in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels")).iter(f"{RELS}Relationship"):
                target = rel.get("Target", "").lstrip("/")
                targets[rel.get("Id")] = target if target.startswith("xl/") else "xl/" + target
        shared = []
        if "xl/sharedStrings.xml" in names:
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(f"{MAIN}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{MAIN}t")))
        sheets = [(sheet.get("name"), targets.get(sheet.get(f"{DOC_RELS}id"), f"xl/worksheets/sheet{i}.xml"))
                  for i, sheet in enumerate(workbook.iter(f"{MAIN}sheet"), 1)]
        out = [f"xlsx, {path.stat().st_size:,} bytes, {len(sheets)} sheets: " + ", ".join(name for name, _ in sheets)]
        for name, member in sheets[:sheets_shown]:
            if member not in names:
                continue
            rows = []
            for row in ET.fromstring(archive.read(member)).iter(f"{MAIN}row"):
                cells = []
                for cell in row.iter(f"{MAIN}c"):
                    value = cell.find(f"{MAIN}v")
                    text = value.text if value is not None and value.text is not None else \
                        "".join(node.text or "" for node in cell.iter(f"{MAIN}t"))
                    if cell.get("t") == "s" and text.isdigit() and int(text) < len(shared):
                        text = shared[int(text)]
                    cells.append(short(text, 24))
                rows.append(short(" | ".join(cells), LINE))
                if len(rows) == HEAD_ROWS + 1:
                    break
            out.append(f"sheet {name!r}, first rows:")
            out += rows
    return "\n".join(out) + "\n"
