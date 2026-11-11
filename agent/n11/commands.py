"""Turns a model reply into calls, and each call into the text sent back to the model.

Step: 4, the loop.
Called by: agent/n11/solve.py (`N11.calls`, `N11.execute`, `N11.after_write`, `N11.counted`).
Calls: common/runner.py to run a script; common/contract.py, common/selfreport.py, agent/n11/sanity.py,
    agent/n11/props.py and agent/n11/conventions.py for what is said after a run.
Start at: `from_fenced` (a reply's text to its `Call`s), then `Workspace.write`, `Workspace.run`, `Workspace.checks`
    and `Workspace.sanity`. `Ran` is the record of one run. The second half of the file is the readers behind
    `inspect` and the helpers the above use.

The rules in detail
-------------------
The agent's five commands, the checks, and the fenced reply form that calls them.

Each command returns an observation built by code, within its bound in bytes (`BOUNDS`), at no request:
- `write(path, text)`: a file under `work/` alone (its absolute forms, `/work/<name>`, `/app/work/<name>` and the
  instruction's `/app/<name>.py`, taken as `work/<name>`: `work_form`), of any length where it arrived whole (the
  reply's output limit is the real limit); a name that shadows a standard-library module (`inspect.py`, `json.py`:
  `sys.stdlib_module_names`) refused; a `.py` file compiled, its SyntaxError said with the three lines around it;
  every refusal names the form accepted;
- `run(path)`: a `.py` file under `work/` run as `python3 <file>` from `work/`, in its own process group, its streams
  through the bounded writer (common.runner.run_bounded), the agent's own variables withheld and `OUTPUT_DIR` naming
  the output (common.runner.child_env); the exit and seconds, the ends of stdout, the end of stderr with the
  traceback's frames inside `work/` (file, line, function, source line), the output's files written or left
  (common.contract.listing); after an exit 0 the output's digest and the findings (`Workspace.sanity`: the
  instruction's own properties broken and its keys and columns missing, agent/n11/props.py; the sanity checks',
  agent/n11/sanity.py; the rules'), each kept whole, and the keys of the input's parameters files no script under
  `work/` names (`unread`), and apart from them the span and the scale findings (`Workspace.against`,
  agent/n11/conventions.py); then the checks;
- `inspect(path, rows)`: a data file of the task's: a table's shape, columns with type, empty count, min, max and
  first value, its first and last rows; JSON's key tree; text, XML and HTML's size, head and tag counts; a file in the
  output directory or under `work/` alike, and either directory listed (`made`); an input file the loop's first
  message describes is answered from that description (`Workspace.answered`, `carried`);
- `lib(name=...)` or `lib(shelf=...)`: parsed, and answered by agent/n11/solve.py in one line;
- `finish(reason)`: the checks and the sanity findings once more (the agent decides what follows:
  agent/n11/solve.py; `reason`, "accept: <why>", answers a finish refused for its findings).
The rules (`Workspace.ruling`) are said among the findings after a run's exit 0 and at the finish alike, each known
by its rule (`Workspace.ruled`, `Ran.rules`): the four that read the instruction as properties (agent/n11/props.py,
said as `instruction:` lines), the three that read the output alone after the sanity checks' (agent/n11/sanity.py
`ruled`), `literal` (`literals`: a script under work/ writing the instruction's example-output figures as literals),
then the six wide rules (agent/n11/props.py `wide_found`).
The checks (`Workspace.checks`): every output file the instruction names present at its path, parsing as its kind,
its numbers finite, a file at the output's root and none holding a canary (common.contract.check over the named
files), and the self-report scan (common.selfreport.flags) without its bound on `*_error` keys. Every observation has
the task's canaries marked `<canary>`.

A reply's calls (`Call`) come from its text (`from_fenced`): one block per file headed `python file=work/<name>.py`
(a block naming another file of any kind is that file's, never work/main.py's: `labelled`), and `inspect`, `lib`,
`run` and `finish` as one-line directives (`finish reason=...` or `finish accept: ...` with its reason). A block the
output cap cut is marked `cut` and never run (a cut write's text is salvaged by the agent: `cut_arguments`); a call
whose name or arguments cannot be read is marked with its `problem`.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import os
import posixpath
import re
import sys
import threading
import tomllib
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from common import canary, contract, digest, paths, runner, selfreport
from common.guard import WRITING
from n11 import measure, props, sanity

WORK = "work"                   # the directory the model writes under, as it names it
BOUNDS = {"write": 600, "run": 7_000, "inspect": 3_000, "finish": 1_500}   # run's raised for the
#                                 digest, which the observation keeps whole with the checks
SHADOWED = frozenset(sys.stdlib_module_names)   # a file or directory under work/ so named shadows the module
WRITE_FORM = f'write(path="{WORK}/<name>.py", text=<the whole file>), the path relative, under {WORK}/'
WORK_ROOTS = ("/work/", "/app/work/")   # the model's absolute forms of work/, each taken as work/ (and the
APP = "/app/"                   #         instruction's `/app/<name>.py`, a script it asks saved there)
PARAM_WORDS = frozenset({"param", "params", "parameter", "parameters", "config", "configuration", "settings", "cfg",
                         "conf"})   # an input file whose name holds one is a parameters file
PARAM_KINDS = (".json", ".toml", ".yaml", ".yml", ".txt", ".ini", ".cfg", ".conf", ".env", ".properties", "")
PARAM_BYTES = 65_536            # a larger parameters file is not read for its keys
PARAM_NAMES = 3                 # the keys the unread-parameter finding names, at most: the harness's bound, never said
RULE_OF = {"fraction": "fraction", "header": "inline_header", "rowsum": "row_sum", "formula": "formula"}   # the
#                                 rule each of the instruction's rule kinds (agent/n11/props.py `rule_props`) is
LITERALS = 3                    # `literal`: the example's figures a script writes as literals, at the least; never said
LITERAL_NAMES = 4               # and those its finding names, at most
KEY_LINE = re.compile(r"^([A-Za-z_][\w.-]*)\s*[:=]\s*(.*)$")    # a YAML or key-value text's top-level line
MAIN_BLOCK = re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""", re.M)
MALFORMED_BYTES = 600           # a malformed or cut call's observation
STDOUT_ENDS = 1_000             # stdout's first and last characters shown
STDOUT_CUT = "\n[... cut ...]\n"  # between them where stdout is longer: said as cut, never as a count
SPLIT = (f"into as few files as fit, each a module {WORK}/main.py imports, and {WORK}/main.py last; write one file "
         "a reply")             # the split a cut reply asks for: a shape, never a count
STDERR_TAIL = 2_000             # stderr's last characters shown
FRAMES = 8                      # the traceback's frames inside work/ shown, the innermost kept
SOURCE_CHARS = 160              # a frame's source line, at most
MAX_ROWS = 10                   # inspect's head and tail rows, each, at most
TABLE_ROWS = 200_000            # a table's rows read for inspect's statistics
COLUMNS = 40                    # columns described, at most
TEXT_HEAD = 1_500               # a text file's head shown
JSON_BYTES = 20_000_000         # a JSON file larger than this is not parsed
DESCRIBED_TOKENS = 2_850        # the input files' descriptions the loop's first message carries, at most, in the
#                                 model's tokens (agent/n11/measure.py)
FILE_TOKENS = 1_450             # and one file's
DESCRIBED_BYTES = 6_000         # the byte fallbacks where the tokenizer does not load: DESCRIBED_TOKENS at the
FILE_BYTES = 1_500              # densest description's 2.11 bytes a token, FILE_TOKENS at a table of digits' 1.05
NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
FRAME = re.compile(r'^\s*File "(?P<file>[^"\n]+)", line (?P<line>\d+), in (?P<name>[^\n]+)$', re.M)
ERROR = re.compile(r"^(?:[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)|[A-Z]\w*Error)\b.*$", re.M)
ADDRESS = re.compile(r"0x[0-9a-fA-F]+")      # an object's address in a message, which differs run to run
TABLES = (".csv", ".tsv", ".txt", ".dat")
EMPTY = {"", "nan", "na", "n/a", "null", "none", "-nan", "inf", "-inf"}
NAMES = ("write", "run", "inspect", "lib", "finish")    # the commands, in the order a refusal names them
REQUIRED = {"write": ("path", "text"), "run": ("path",), "inspect": ("path",), "lib": (), "finish": ()}   # and the
#                                                                                       arguments each requires


def size(text: str) -> int:
    return len(text.encode("utf-8"))


def fit(body: str, bound: int, tail: str = "", unit: measure.Unit = measure.BYTES) -> str:
    """`body` and then `tail` within `bound` (bytes, or `unit`'s): the tail kept whole (the checks and the budget
    line), the body cut at a line inside it with a marker where they do not fit."""
    joined = body + (("\n" + tail) if tail else "")
    if unit.size(joined) <= bound:
        return joined
    marker = "\n[... cut to the bound]"
    room = bound - unit.size(tail) - unit.size(marker) - (1 if tail else 0)
    if room <= 0:
        return unit.prefix(tail, bound)
    head = unit.prefix(body, room)
    if "\n" in head[len(head) // 2:]:
        head = head.rsplit("\n", 1)[0]
    return head + marker + (("\n" + tail) if tail else "")


# ------------------------------------------------------------------------------------------------- the calls


@dataclass
class Call:
    name: str
    arguments: dict = field(default_factory=dict)
    problem: str | None = None      # why it cannot be run, where it is malformed
    cut: bool = False               # the output cap cut inside it: never run

    def describe(self) -> str:
        """The call in a log line: its name and its path, never a text it carries."""
        path = self.arguments.get("path") if isinstance(self.arguments, dict) else None
        what = self.arguments.get("name") or self.arguments.get("shelf") if self.name == "lib" else path
        return self.name + (f" {str(what)[:80]}" if what else "") + (" (cut)" if self.cut else "") + \
            (" (malformed)" if self.problem else "")


def checked(name: str, arguments) -> str | None:
    """Why a call cannot be run as written; None where it can."""
    if name not in NAMES:
        return f"there is no tool `{name[:40]}`; the tools are {', '.join(NAMES)}"
    if not isinstance(arguments, dict):
        return "its arguments are not a JSON object"
    missing = [key for key in REQUIRED[name] if not isinstance(arguments.get(key), str) or
               (key == "path" and not arguments[key].strip())]
    if missing:
        return f"it lacks the argument{'s' if len(missing) > 1 else ''} {', '.join(missing)} (a string)"
    if name == "lib" and not any(isinstance(arguments.get(key), str) and arguments[key].strip()
                                 for key in ("name", "shelf")):
        return "it names neither `name` (a function) nor `shelf` (a module)"
    if name == "finish" and arguments.get("reason") is not None and not isinstance(arguments["reason"], str):
        return "its `reason` is not a string"
    return None


# ---------------------------------------------------------------------------------------------- the fenced blocks

FENCE = re.compile(r"^\s*(`{3,})(.*)$")
TAG = re.compile(r"^[A-Za-z][\w+-]*$")
PYTHON = ("python", "python3", "py")
KINDS = ("py", "txt", "json", "csv", "tsv", "md", "yaml", "yml", "toml", "ini", "cfg", "sh", "sql")
PATH_LINE = re.compile(r"""^\s*(?P<mark>\#+|//)?\s*(?:(?P<label>file(?:name)?|path)\s*[=:]\s*)?[`'"*]*"""
                       r"""(?P<path>[\w./-]*\w\.\w+)[`'"*]*\s*$""", re.I)
NAMED_LINE = re.compile(r"""^\s*(?P<mark>\#+|//)?\s*(?:file(?:name)?|path)\s*[=:]\s*(?P<quote>[`'"*]*)"""
                        r"""(?P<path>[\w./-]*\w)[`'"*]*\s*$""", re.I)     # a labelled line, any path
FINISH_LINE = re.compile(r"^finish\b\s*(?:$|[.:]\s*$|accept\s*:|reason\s*[=:])", re.I)
VERB_LINE = re.compile(r"""^(?:(?:run|inspect)\s+[`'"]?[\w./-]*\w\.\w+[`'"]?(?:\s+rows\s*=\s*\d+)?|lib\s+(?:(?:name|"""
                       r"""shelf)\s*=\s*)?`?[A-Za-z_][\w.]*`?)\s*$""", re.I)


def opened(rest: str) -> tuple[str, str | None]:
    """The language and the path a fence's opening line names after its backticks (`python file=work/a.py`, `python
    work/a.py`, `text file=work/b.txt`); the path None where it names none."""
    tag, path = "", None
    for k, word in enumerate(re.sub(r"file\s*=\s*", "file=", rest).split()):
        if word.lower().startswith("file="):
            path = word[5:].strip("`'\"") or path
        elif k == 0 and TAG.match(word):
            tag = word.lower()
        elif path is None and ("/" in word or "." in word):
            path = word.lstrip("#").strip("`'\"") or None
    return tag, path


def headed(line: str, tag: str = "") -> str | None:
    """The path a block's first line names as the block's header, alone on it: `file=work/a.py`, `# file: work/a.py`,
    `work/a.py`, `# a.py`; None where the line is none of these. The path is under work/ (a bare name is); in a python
    block it is a `.py` file's, and elsewhere, unlabelled (no `file=` or `file:`), it holds a directory or a known
    file kind's suffix (`KINDS`): so a line of code (`path = "data.csv"`) or an example's is never read as one."""
    found = PATH_LINE.match(line)
    if found is None or not plain(found.group("path")).startswith(f"{WORK}/"):
        return None
    path = found.group("path")
    suffix = path.rsplit(".", 1)[1].lower()
    if tag in PYTHON:
        return path if suffix == "py" else None
    return path if found.group("label") or "/" in path or suffix in KINDS else None


def labelled(line: str, tag: str = "") -> str | None:
    """The path a labelled line names (`file=work/requirements.txt`, `# file: notes.md`, `path: work/.env`), a
    directory's or a known kind's (`KINDS`), so that a block naming another file (a python block headed
    `file=work/requirements.txt`) is never work/main.py's. In a python or untagged block, which would otherwise be
    main.py's, wherever it points (the write refuses one outside work/ by name); in another language's, under work/
    alone (an example's output path is no header). None in a python block where the line is code: its value quoted,
    uncommented (`file = "results.json"`)."""
    found = NAMED_LINE.match(line)
    if found is None:
        return None
    path = found.group("path")
    if "/" not in path and posixpath.splitext(path)[1][1:].lower() not in KINDS:
        return None
    if tag in PYTHON and not found.group("mark") and set(found.group("quote")) & {"'", '"'}:
        return None
    return path if tag in PYTHON or not tag or plain(path).startswith(f"{WORK}/") else None


def plain(path: str) -> str:
    """A path as the model may write it, as `work/<name>` (`./work/a.py`, `a.py`, an absolute form `work_form` takes):
    what two blocks naming one file share."""
    named = posixpath.normpath(work_form(path.strip().strip("`'\"")).removeprefix("./")) if path.strip() else ""
    return named if "/" in named else f"{WORK}/{named}"


def work_form(path: str) -> str:
    """A path in one of the model's absolute forms of work/ as `work/<name>`: `/work/<name>`, `/app/work/<name>`, and
    `/app/<name>.py` (a script the instruction asks saved there); any other as it is."""
    for root in WORK_ROOTS:
        if path.startswith(root):
            return WORK + "/" + path[len(root):]
    rest = path[len(APP):] if path.startswith(APP) else ""
    return f"{WORK}/{rest}" if rest.endswith(".py") and "/" not in rest else path


def directive_line(line: str) -> bool:
    """A line that is one of the forms' directives, alone (`finish`, `finish accept: <why>`, `run work/a.py`,
    `inspect <path>`, `lib <name>`), markdown's emphasis and quoting aside; never a line of Python."""
    text = line.strip().strip("`*").strip()
    return bool(FINISH_LINE.match(text) or VERB_LINE.match(text))


@dataclass
class Fenced:
    """One fenced block of a reply: its language, the path it names (on the fence line, else on the line after it,
    which is then no line of the file), its lines, whether a closing fence ended it, and why it saves nothing (`void`:
    "empty" where it holds no code, "path" where it held its path line alone, "directive" where it holds directive
    lines alone; "" where it holds a file's text)."""
    tag: str
    path: str | None
    lines: list[str]
    closed: bool
    void: str = ""
    words: tuple[str, ...] = ()     # its fence line's words after the backticks (`fence_finish`)

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def fenced_block(rest: str, lines: list[str], closed: bool) -> Fenced:
    """A block from its fence line's words after the backticks (`rest`) and its lines: the path the fence line names,
    else the path its first non-blank line names alone (`headed`, else `labelled`), that line dropped; a first line
    naming the fence's own path dropped too. Then why it saves nothing, if it does not (`Fenced.void`)."""
    tag, path = opened(rest)
    dropped = False
    first = next((k for k, line in enumerate(lines) if line.strip()), None)
    said = headed(lines[first], tag) if first is not None else None
    other = labelled(lines[first], tag) if first is not None and said is None else None
    said = said or other
    if said is not None and (path is None or plain(said) == plain(path)):
        lines, dropped = lines[:first] + lines[first + 1:], True
        path = path or said
    body = [line for line in lines if line.strip()]
    python = path.endswith(".py") if path is not None else tag in PYTHON
    code = [line for line in body if not (python and line.lstrip().startswith("#"))]
    void = ("path" if dropped and not body else "empty") if not code else \
        "directive" if all(directive_line(line) for line in body) else ""
    return Fenced(tag, path, lines, closed, void, tuple(rest.split()))


def fenced(text: str) -> list[str | Fenced]:
    """A reply's lines outside any block and its fenced blocks, in order: a line opening with three backticks or more
    opens a block, a line of backticks alone closes it, and a block the reply ends inside is left open
    (`closed` False). The thinking a route writes into the content is the caller's to drop."""
    found: list[str | Fenced] = []
    rest, lines = "", None
    for line in text.split("\n"):
        if lines is None:
            fence = FENCE.match(line)
            if fence is None:
                found.append(line)
            else:
                rest, lines = fence.group(2), []
        elif line.strip() and set(line.strip()) == {"`"} and len(line.strip()) >= 3:
            found.append(fenced_block(rest, lines, True))
            lines = None
        else:
            lines.append(line)
    if lines is not None:
        found.append(fenced_block(rest, lines, False))
    return found


DIRECTIVE = re.compile(r"^\s*(?P<verb>inspect|lib|run|finish)\b(?P<rest>[^\n]*)$")
ARGUMENT = re.compile(r"(\w+)\s*=\s*(\S+)")


def directive(line: str) -> tuple[str, dict] | None:
    """A line outside any block read as its command: `inspect <path> [rows=N]`, `lib name=<f>` or `lib shelf=<m>`,
    `run <path>` and `finish` (or `finish reason=<why>`, `finish accept: <why>`); None where it is none."""
    found = DIRECTIVE.match(line)
    if found is None:
        return None
    verb, rest = found.group("verb"), found.group("rest").strip()
    named = dict(ARGUMENT.findall(rest))
    words = [word for word in rest.split() if "=" not in word]
    if verb == "finish" and not rest:
        return "finish", {}
    if verb == "finish" and rest.lower().startswith(("reason=", "accept:")):
        return "finish", {"reason": rest.split("=", 1)[1].strip().strip("`'\"") if rest.lower().startswith(
            "reason=") else rest}
    if verb == "run" and words and words[0].strip("`").endswith(".py"):      # not a sentence's "run it"
        return "run", {"path": words[0].strip("`")}
    if verb == "inspect" and (words or "path" in named):
        arguments = {"path": named.get("path", words[0] if words else "").strip("`")}
        if named.get("rows", "").isdigit():
            arguments["rows"] = int(named["rows"])
        return "inspect", arguments
    if verb == "lib" and ("name" in named or "shelf" in named):
        return "lib", {key: named[key].strip("`") for key in ("name", "shelf") if key in named}
    return None


def block_directive(item: Fenced) -> tuple[str, dict] | None:
    """The `run` or `finish` a closed block holds alone, its one line (`directive`); `finish` on an empty closed block's
    fence line (`fence_finish`); None where it holds anything else."""
    lines = [line for line in item.lines if line.strip()]
    if item.closed and not lines and fence_finish(item):
        return "finish", {}
    said = directive(lines[0].strip().strip("`*").strip()) if len(lines) == 1 and item.closed and \
        item.void == "directive" else None
    return said if said is not None and said[0] in ("run", "finish") else None


def fence_finish(item: Fenced) -> bool:
    """`finish` alone on a block's fence line after its language, if any (```` ```python finish ````). A block holding
    the line `finish` alone is `block_directive`'s reading already, whatever its fence line says."""
    words = item.words[1:] if item.words and item.words[0].lower() in PYTHON else item.words
    return words == ("finish",)


def from_fenced(text: str) -> list[Call]:
    """The fenced form's calls, in order: a block naming its file (`fenced`: on its fence line, `python
    file=work/<name>.py`, or alone on the line after it, which is no line of the file; a labelled line naming any kind,
    `labelled`, whose quoted-code and other-language example exceptions still apply) is a `write` of it, subject
    to the workspace path check (which also refuses dotfiles under work/), and a python or untagged block naming none
    `work/main.py`'s, a block left open at the reply's end a cut one; a block holding no
    code, its path line alone or directive lines alone is no call, nor a block naming no file in another language; a
    later block for a file replaces an earlier, as two writes do. `inspect <path> [rows=N]`, `lib name=<f>` or `lib
    shelf=<m>`, `run <path>` and `finish` (or `finish reason=<why>`, `finish accept: <why>`) outside any block are their
    commands. A block holding one `run` or `finish` alone is that directive (`block_directive`, and `finish` on an
    empty closed block's fence line), and a block left open with nothing in it after a closed one is no call, never a
    cut write. What runs after a reply's last write is the agent's rule (agent/n11/solve.py `after_write`)."""
    calls: list[Call] = []

    def add(name: str, arguments: dict, cut: bool = False) -> None:
        calls.append(Call(name, arguments, checked(name, arguments), cut))

    closed = False
    for item in fenced(runner.THINK.sub("", text or "")):
        if not isinstance(item, Fenced):
            said = directive(item)
        elif (said := block_directive(item)) is not None:
            closed = True
        else:
            path = item.path or (f"{WORK}/main.py" if item.tag in PYTHON or not item.tag else None)
            stray = closed and not item.closed and item.void == "empty"
            if path is not None and (not item.closed or not item.void) and not stray:
                add("write", {"path": path, "text": item.text}, cut=not item.closed)
            closed = closed or item.closed
            continue
        if said is not None:
            add(*said)
    return calls


def cut_arguments(raw: str) -> dict[str, str]:
    """The string arguments a call's JSON holds where the output cap cut it: each key with its value, the last value
    as far as it got (an escape the cut split dropped); a value that is no string ends the read."""
    found: dict[str, str] = {}
    at = raw.find("{")
    if at < 0:
        return found
    at += 1
    while at < len(raw):
        while at < len(raw) and raw[at] in " \t\r\n,":
            at += 1
        if at >= len(raw) or raw[at] != '"':
            break
        key, at, closed = json_string(raw, at)
        if not closed:
            break
        while at < len(raw) and raw[at] in " \t\r\n:":
            at += 1
        if at >= len(raw) or raw[at] != '"':
            break
        value, at, closed = json_string(raw, at)
        found[key] = value
        if not closed:
            break
    return found


def json_string(raw: str, at: int) -> tuple[str, int, bool]:
    """The JSON string opening at `raw[at]`: its value, where reading stopped and whether it closed."""
    decoder = json.JSONDecoder(strict=False)
    end = at + 1
    while end < len(raw):
        if raw[end] == "\\":
            end += 2
            continue
        if raw[end] == '"':
            try:
                return decoder.decode(raw[at:end + 1]), end + 1, True
            except ValueError:
                return "", end + 1, False
        end += 1
    body = raw[at + 1:]
    for trim in range(7):                       # a cut inside an escape: \u00 or a lone backslash
        try:
            return decoder.decode('"' + body[:len(body) - trim] + '"'), len(raw), False
        except ValueError:
            continue
    return "", len(raw), False


# --------------------------------------------------------------------------------------------- the workspace


@dataclass
class Ran:
    """One `run`: its exit (None: stopped at its limit), seconds, the frames inside work/, the error line; after an
    exit 0 the output's digest and the sanity findings over it."""
    path: str
    exit: int | None
    seconds: float
    error: str = ""
    frames: tuple[tuple[str, int, str], ...] = ()       # (work/<file>, line, function), outermost first
    failures: tuple[str, ...] = ()                      # the checks' lines after it
    findings: tuple[str, ...] = ()                      # the sanity findings after an exit 0 (agent/n11/sanity.py)
    digest: str = ""                                    # the output's digest after an exit 0
    line: str = ""                                      # the innermost frame's source line in work/
    wrote: tuple[str, ...] = ()                         # the output's files it wrote, new or changed
    stdout: str = ""                                    # its stdout, the canaries marked
    unread: tuple[tuple[str, str], ...] = ()            # after an exit 0, the parameters no script reads, (file, key)
    span: str = ""                                      # after an exit 0, the span finding's line and the scale
    scale: str = ""                                     # finding's (`Workspace.against`), "" where none
    rules: tuple["Finding", ...] = ()                   # after an exit 0, the rules' findings among `findings`

    @property
    def clean(self) -> bool:
        return self.exit == 0 and not self.error and not self.failures

    @property
    def identity(self) -> tuple | None:
        """What a repeat of this failure is known by: the exception's type and its message's first line (the type
        alone, or with its line, would match two different errors); None where it raised nothing."""
        if not self.error:
            return None
        kind, _, message = self.error.partition(":")
        return kind.strip(), ADDRESS.sub("0x?", message.strip().split("\n", 1)[0][:300])


@dataclass(frozen=True)
class Finding:
    """One rule's finding as `Workspace.sanity` said it: the rule, what it is known by across runs (the rule, the
    file and the name), its line as said, and the instruction's property where a rule reads one."""
    rule: str
    ident: str
    line: str
    prop: props.Prop | None = None


class Workspace:
    """The model's files (`root/work/`), the task's output and inputs, and the commands over them."""

    def __init__(self, root: Path, out: Path, task, pathmap, named: tuple[str, ...],
                 stated: props.Stated | None = None) -> None:
        self.root, self.out, self.task, self.pathmap = Path(root), Path(out), task, pathmap
        self.named = tuple(named)
        self.dir = self.root / WORK
        self.dir.mkdir(parents=True, exist_ok=True)
        self.canaries = tuple(task.canaries) if task is not None else ()
        self.stated = stated            # what the instruction states of its outputs (agent/n11/props.py)
        self.current = None
        self.lock = threading.Lock()
        self.params: list[tuple[str, list[str]]] | None = None     # the parameters files' keys, read once
        self.dated = None               # the input tables' dates (agent/n11/conventions.py `dated`), read once
        self.ruled: tuple[Finding, ...] = ()            # the rules' findings as the last `sanity` said them

    def redact(self, text: str) -> str:
        return canary.redact(text, self.canaries)

    # ---------------------------------------------------------------------------------------------- paths

    def local(self, path: str) -> tuple[Path | None, str]:
        """A path the model names under work/ (relative, bare, by this machine's path, or in an absolute form
        `work_form` takes) as (the file here, `work/<name>`), or (None, why not)."""
        named = str(path).strip().strip("`'\"")
        if named.startswith(str(self.dir) + "/"):
            named = WORK + "/" + named[len(str(self.dir)) + 1:]
        named = posixpath.normpath(work_form(named).removeprefix("./")) if named else ""
        if "/" not in named:
            named = f"{WORK}/{named}"
        parts = named.split("/")
        outside = (f"`{str(path)[:120]}` is not under {WORK}/: files are written and run under {WORK}/ alone, and "
                   f"only a run's script writes the output files; name it relative, {WORK}/<name>.py (e.g. "
                   f"{WORK}/main.py)")
        if parts[0] != WORK or len(parts) < 2 or len(parts) > 4 or not all(NAME.match(part) for part in parts[1:]):
            return None, outside
        here = self.root.joinpath(*parts)
        try:                                    # a link a run left under work/ leads nowhere else
            inside = here.resolve().is_relative_to(self.dir.resolve())
        except (OSError, RuntimeError):
            inside = False
        return (here, named) if inside else (None, outside)

    def data(self, path: str) -> Path | None:
        """An input file the model names: as the instruction names it (the path map), under the task's data directory,
        or by its name there."""
        named = str(path).strip().strip("`'\"")
        data_dir = Path(self.pathmap.data_dir)
        for entry in self.pathmap.entries:
            if entry.kind == "read" and entry.path is not None and posixpath.normpath(named) == entry.named:
                return Path(entry.path)
        here = Path(named)
        try:
            if here.is_absolute() and here.resolve().is_relative_to(data_dir.resolve()) and here.exists():
                return here
        except (OSError, ValueError):
            return None
        inside = (data_dir / named.lstrip("/")).resolve() if named else None
        if inside is not None and inside.is_relative_to(data_dir.resolve()) and inside.exists():
            return inside
        same = [file for file in sorted(data_dir.rglob("*")) if file.name == posixpath.basename(named)] \
            if data_dir.is_dir() else []
        return same[0] if len(same) == 1 else None

    # ---------------------------------------------------------------------------------------------- write

    def write(self, path: str, text: str, tail: str = "") -> str:
        """The `write` observation, `tail` (the budget line) after it: the file saved whole at any length (the reply's
        cut is the limit), a path outside work/ or a name shadowing a standard-library module refused, each refusal
        naming the form accepted; a `.py` file compiled."""
        bound = BOUNDS["write"]
        here, named = self.local(path)
        if here is None:
            return fit(f"write refused: {named}; nothing was written. The form: {WRITE_FORM}", bound, tail)
        clash = shadows(named)
        if clash:
            return fit(f"write refused: {named} would shadow the standard library's `{clash}` (a script importing "
                       f"{clash} would get this file); name it otherwise, e.g. {WORK}/my_{clash}.py; nothing was "
                       f"written", bound, tail)
        text = unfenced(text)
        try:
            here.parent.mkdir(parents=True, exist_ok=True)
            here.write_text(text, encoding="utf-8")
        except OSError as exc:                  # a directory where a file is named, or a file where a directory is
            return fit(f"write refused: {named} could not be written ({type(exc).__name__}): a directory or a file "
                       f"already stands on its path; nothing was written. The form: {WRITE_FORM}", bound, tail)
        said = f"wrote {named}: {size(text):,} B"
        if not named.endswith(".py"):
            return fit(said, bound, tail)
        failed = compiled(text, named)
        if failed:
            return fit(f"{said}; {failed}", bound, tail)
        return fit(said + "; compiles", bound, tail)

    # ------------------------------------------------------------------------------------------------ run

    def hold(self, proc) -> None:
        with self.lock:
            self.current = proc

    def kill(self) -> None:
        """The running script's process group, killed; nothing where none runs (the watchdog's)."""
        with self.lock:
            proc = self.current
        if proc is not None and proc.poll() is None:
            runner.kill_group(proc)

    def execute(self, script: Path, timeout: float) -> tuple[int | None, float, str, str]:
        """`script` run from work/ under `timeout` seconds: its exit, seconds, stdout and stderr (each its last MiB
        at most)."""
        exit_, seconds, out, err = runner.run_bounded([sys.executable, str(script)], timeout, cwd=self.dir,
                                                      env=runner.child_env(self.out), hold=self.hold)
        stdout = out.value().decode("utf-8", errors="replace")
        stderr = err.value().decode("utf-8", errors="replace")
        with WRITING:       # a watchdog close under way finishes (and exits) before the run is read
            pass
        return exit_, seconds, stdout, stderr

    def run(self, path: str, timeout: float, seen: set | None = None, tail=lambda ran: "") -> tuple[str, Ran | None]:
        """The `run` observation (its checks, and `tail(run)` after them, `tail(None)` for a run refused) and the run,
        None where refused. An exception `seen` before (its `identity`) is said once: the next time, by name."""
        here, named = self.local(path)
        if here is None or not named.endswith(".py"):
            why = named if here is None else f"{named} is not a .py file"
            return fit(f'run refused: {why}. The form: run(path="{WORK}/<name>.py"), a .py file under {WORK}/',
                       BOUNDS["run"], tail(None)), None
        if not here.is_file():
            return fit(f"run refused: {named} does not exist; write it first ({WRITE_FORM})", BOUNDS["run"],
                       tail(None)), None
        before = contract.snapshot(self.out)
        exit_, seconds, stdout, stderr = self.execute(here, timeout)
        return self.observe(named, timeout, exit_, seconds, stdout, stderr, before, seen, tail)

    def observe(self, named: str, timeout: float, exit_: int | None, seconds: float, stdout: str, stderr: str,
                before: dict, seen: set | None = None, tail=lambda ran: "") -> tuple[str, Ran]:
        """A run's observation from what it did: `before` the output's snapshot before it, the output as it stands
        after."""
        after = contract.snapshot(self.out)
        stdout, stderr = self.redact(stdout), self.redact(stderr)
        frames = self.frames(stderr)
        error = last_error(stderr) if runner.TRACEBACK in stderr or exit_ not in (0, None) else ""
        failures = tuple(self.checks())
        digest, findings = self.sanity() if exit_ == 0 else ("", [])
        span, scale = self.against() if exit_ == 0 else ("", "")
        ran = Ran(named, exit_, seconds, error, tuple((file, line, name) for file, line, name, _ in frames), failures,
                  tuple(findings), digest, frames[-1][3] if frames and error else "",
                  tuple(sorted(name for name in after if before.get(name) != after[name])), stdout,
                  unread=self.unread() if exit_ == 0 else (), span=span, scale=scale,
                  rules=self.ruled if exit_ == 0 else ())
        head = (f"run {named}: stopped at its limit of {timeout:.0f} s" if exit_ is None else
                f"run {named}: exit {exit_} after {seconds:.1f} s")
        parts = [head]
        if stdout.strip():
            parts.append(f"stdout ({size(stdout):,} B):\n"
                         + runner.bounded(stdout.strip("\n"), 2 * STDOUT_ENDS, marker=STDOUT_CUT))
        repeat = ran.identity is not None and seen is not None and ran.identity in seen
        if repeat:
            parts.append(f"the same error as an earlier run: {ran.error[:300]}" + (
                f" at {frames[-1][0]}:{frames[-1][1]}" if frames else "") + "; that fix did not work, try another")
        else:
            if frames:
                parts.append("traceback in work/, outermost first:\n" + "\n".join(
                    f"  {file}:{line} in {name}: {source}" for file, line, name, source in frames[-FRAMES:]))
            if stderr.strip():
                parts.append("stderr (its end):\n" + stderr.strip("\n")[-STDERR_TAIL:])
        parts.append(contract.listing(before, after))
        kept = [part for part in (f"outputs:\n{digest}" if digest else "", sanity.said(findings),
                                  self.redact(unread_said(ran.unread)), span, scale, verdict(failures), tail(ran))
                if part]
        return fit("\n".join(parts), BOUNDS["run"], "\n".join(kept)), ran

    def frames(self, stderr: str) -> list[tuple[str, int, str, str]]:
        """The traceback's frames inside work/: (work/<file>, line, function, source line), outermost first, the last
        traceback's where the error output holds several."""
        found = []
        text = stderr[stderr.rfind(runner.TRACEBACK):] if runner.TRACEBACK in stderr else stderr
        base = str(self.dir.resolve())
        for match in FRAME.finditer(text):
            file = match.group("file")
            try:
                resolved = str(Path(file).resolve()) if os.path.isabs(file) else str((self.dir / file).resolve())
            except OSError:
                continue
            if not resolved.startswith(base + "/"):
                continue
            relative = WORK + "/" + resolved[len(base) + 1:]
            line = int(match.group("line"))
            found.append((relative, line, match.group("name").strip(), source_line(Path(resolved), line)))
        return found

    # ------------------------------------------------------------------------------------------------ checks

    def checks(self) -> list[str]:
        """The checks' failing lines over the output as it stands; [] where every one passes."""
        spec = contract.Contract(tuple(contract.FileSpec(name, contract.fmt_of(name)) for name in self.named))
        lines = [failure.text for failure in contract.check(spec, self.out, self.canaries)]
        lines += [f"the output reports a failure: {flag}" for flag in selfreport.flags(self.out)]
        return [self.redact(line) for line in lines]

    def sanity(self) -> tuple[str, list[str]]:
        """The output's digest and the findings over it as it stands (`ruling`). A reader's fault is never raised into
        the run."""
        ruled = sanity.Ruled()
        digest_, findings = sanity.read(self.out, self.named, rules=ruled)
        return self.ruling(digest_, findings, ruled)

    def ruling(self, digest_: str, values: list[str], ruled: sanity.Ruled) -> tuple[str, list[str]]:
        """The findings: the instruction's own properties the output breaks (`instruction:`) and the keys and columns
        it names that the files lack (`contract:`), agent/n11/props.py; the sanity checks' value lines; then the
        rules' findings, the six wide rules' after the eight's. Each rule is kept in `ruled`."""
        found: list[Finding] = []
        said, missing = [], []
        outputs = None
        if self.stated is not None:
            try:
                outputs = props.Outputs(self.out)
                lines, missing = props.checked(self.stated, self.out, outputs)
                for prop, line in lines:
                    if prop.kind in RULE_OF:
                        found.append(Finding(RULE_OF[prop.kind], f"{RULE_OF[prop.kind]}|{prop.file}|"
                                                                 f"{'/'.join(prop.names)}|{prop.expr}", f"instruction: {line}", prop))
                    else:
                        said.append(f"instruction: {line}")
            except Exception:  # noqa: BLE001 - the instruction's checks never cost the run's observation
                said, missing, found = [], [], []
        alone = [Finding(*ruled.rules[key], line) for key, line in ruled.lines.items()]
        found += alone + [Finding("literal", f"literal|{file}", line) for file, line in self.literals()]
        try:
            outputs = outputs if outputs is not None else props.Outputs(self.out)
            found += [Finding(rule, ident, line) for rule, ident, line in props.wide_found(
                self.stated.wide if self.stated is not None else None, outputs)]
        except Exception:  # noqa: BLE001 - the six rules never cost the run's observation
            pass
        lines = [*said, *(f"contract: {line}" for line in missing), *values, *(each.line for each in found)]
        self.ruled = tuple(Finding(each.rule, each.ident, self.redact(each.line), each.prop) for each in found)
        return self.redact(digest_), [self.redact(line) for line in lines]

    def literals(self) -> list[tuple[str, str]]:
        """The `literal` rule: each script under work/ whose literal numbers hold `LITERALS` or more of the
        instruction's example-output figures (props.Stated.examples, by value), as (its name, its finding's line); []
        where the instruction gives too few, and where the scripts cannot be read (never raised into the run)."""
        examples = self.stated.examples if self.stated is not None else ()
        if len(examples) < LITERALS:
            return []
        found = []
        try:
            paths = sorted(path for path in self.dir.rglob("*.py") if "__pycache__" not in path.parts)
            for path in paths:
                written = literal_numbers(path.read_text(encoding="utf-8", errors="replace"))
                met = [figure for figure in examples if any(abs(value - float(figure)) <= 1e-9 * max(1.0, abs(
                    float(figure))) for value in written)]
                if len(met) >= LITERALS:
                    named = f"{WORK}/{path.relative_to(self.dir).as_posix()}"
                    shown = props.named_list(met, LITERAL_NAMES)
                    line = (f"{named} writes the task's example output's figures {shown} as literals; "
                            "the task's figures are computed from its inputs, and its example shows their form")
                    found.append((named, line[:sanity.FINDING_CHARS]))
        except (OSError, ValueError):
            return []
        return found

    def against(self) -> tuple[str, str]:
        """The span finding's line and the scale finding's over the output as it stands, against the instruction and the
        input tables' dates (agent/n11/conventions.py), each "" where it does not meet; ("", "") where they cannot be
        read, never raised into the run."""
        from n11 import conventions     # it imports this module

        instruction = self.task.instruction if self.task is not None else ""
        try:
            if self.dated is None:
                self.dated = conventions.dated(Path(self.pathmap.data_dir))
            return (self.redact(conventions.span(instruction, self.out, self.dated)),
                    self.redact(conventions.scale(instruction, self.out)))
        except Exception:  # noqa: BLE001 - the findings never cost the run its observation
            return "", ""

    def parameters(self) -> list[tuple[str, list[str]]]:
        """The task's parameters files and their keys (`parameter_files`); read once."""
        if self.params is None:
            self.params = parameter_files(Path(self.pathmap.data_dir))
        return self.params

    def unread(self) -> tuple[tuple[str, str], ...]:
        """The parameters files' keys (`parameters`) whose names appear nowhere in the model's scripts under work/, a
        whole word of any `.py` file's text, as (file, key) in the files' order, at most `PARAM_NAMES`; () where none,
        and where the files cannot be read (never raised into the run)."""
        try:
            files = self.parameters()
            if not files:
                return ()
            texts = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                              for path in sorted(self.dir.rglob("*.py")) if "__pycache__" not in path.parts)
        except (OSError, ValueError):
            return ()
        found = [(name, key) for name, keys in files for key in keys
                 if not re.search(rf"(?<!\w){re.escape(key)}(?!\w)", texts)]
        return tuple(found[:PARAM_NAMES])

    def reached(self, start: str = "main") -> tuple[set[str], list[str]]:
        """The modules under work/ a file there (`start`, a module's name) imports, directly or through others there
        (their names), and the modules those files import that are nowhere: neither under work/, nor the standard
        library's, nor installed, nor among the task's input files (the files still to write)."""
        here = {path.stem for path in self.dir.glob("*.py")} | {path.name for path in self.dir.iterdir()
                                                                   if path.is_dir()}
        provided = self.provided()
        seen, queue, gone = set(), [start], {}
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            path = self.dir / f"{name}.py"
            for wanted in imported(path) if path.is_file() else ():
                if wanted in here:
                    queue.append(wanted)
                elif wanted not in SHADOWED and wanted not in provided and not installed(wanted):
                    gone[wanted] = None
        return seen - {start}, list(gone)

    def missing_modules(self) -> list[str]:
        """The modules the files under work/ import that are neither under work/, nor the standard library's, nor
        installed, nor among the task's input files: the files still to write."""
        here = {path.stem for path in self.dir.glob("*.py")} | {path.name for path in self.dir.iterdir()
                                                                   if path.is_dir()}
        wanted: dict[str, None] = {}
        for path in sorted(self.dir.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    wanted.update(dict.fromkeys(alias.name.split(".")[0] for alias in node.names))
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        wanted.update(dict.fromkeys(alias.name for alias in node.names if not node.module))
                        if node.module:
                            wanted[node.module.split(".")[0]] = None
                    elif node.module:
                        wanted[node.module.split(".")[0]] = None
        provided = self.provided()
        return [name for name in wanted if name not in here and name not in SHADOWED and name not in provided
                and not installed(name)]

    def unwritten(self, missing: list[str], first: str = "") -> tuple[list[str], str]:
        """`missing` (the modules still to write, `reached`'s or `missing_modules`') less work/'s own name where a file
        under work/ imports its modules package-style, and the note saying so (`packaged_said`, `first`'s import said
        first where it has one); `missing` as it is and "" where none does."""
        found = packaged(self.dir) if WORK in missing else []
        if not found:
            return missing, ""
        return [name for name in missing if name != WORK], packaged_said(found, first)

    def provided(self) -> set[str]:
        """The modules the task's input files hold (a helper `.py` the instruction names beside its data, imported
        from the data directory): never a file still to write."""
        data = Path(self.pathmap.data_dir)
        return {path.stem for path in data.rglob("*.py")} if data.is_dir() else set()

    def missing(self) -> list[str]:
        """The named output files not yet in the output directory."""
        return [name for name in self.named if not (self.out / name).is_file()]

    # -------------------------------------------------------------------------------------------- inspect

    def inspect(self, path: str, rows=None, tail: str = "") -> str:
        """The `inspect` observation, `tail` (the budget line) after it: a file the output directory holds named by its
        path (`output`), else an input file (`data`), else a file the runs or the model made (`made`), described alike
        within the same bounds; a directory listed."""
        bound = BOUNDS["inspect"]
        try:
            rows = max(0, min(MAX_ROWS, int(rows))) if rows is not None else 3
        except (TypeError, ValueError):
            rows = 3
        here, shown = self.output(path)
        if here is None:
            here, shown = self.data(path), path
        if here is None:
            here, shown = self.made(path)
        if here is None:
            return fit(f"inspect: no input file `{str(path)[:160]}`; the inputs are under "
                       f"{self.pathmap.data_dir} (the paths and the digest list them)", bound, tail)
        try:
            if here.is_dir():
                files = sorted(p for p in here.rglob("*") if p.is_file()
                               and "__pycache__" not in p.relative_to(here).parts)
                return fit(self.redact(f"inspect {shown}: a directory of {len(files)} file(s):\n" + "\n".join(
                    f"- {p.relative_to(here)} ({p.stat().st_size:,} B)" for p in files[:60])), bound, tail)
            text = describe(here, rows)
        except Exception as exc:  # noqa: BLE001 - a file that cannot be read is said, never raised
            text = f"could not be read ({type(exc).__name__})"
        return fit(self.redact(f"inspect {shown} ({here.stat().st_size:,} B): {text}"), bound, tail)

    def output(self, path: str) -> tuple[Path | None, str]:
        """A file or directory under the output directory named by its path (the instruction's `/app/output/<name>` or
        `/output/<name>`, this machine's, or `output/<name>`) or that directory itself, as (it, its name said back);
        (None, "") where none is. `inspect` reads it before an input, whose lookup by a name alone would answer an
        output's path with the input of that name."""
        named = str(path).strip().strip("`'\"")
        clean = posixpath.normpath(named.removeprefix("./")) if named else ""
        for root in (str(self.out), *paths.OUTPUTS, "output"):
            rest = paths.under(clean, root)
            here = (self.out / rest if rest else self.out) if rest is not None else None
            if here is not None and inside(here, self.out) and here.exists():
                return here, f"{self.out}/{rest}" if rest else str(self.out)
        return None, ""

    def made(self, path: str) -> tuple[Path | None, str]:
        """A file or directory the runs or the model made, as (it, its name said back): under the output directory by
        its path (`output`); under work/ by `local`'s forms, or work/ itself; else a file of the output directory by its
        name alone, where one file alone has it. (None, "") where none is."""
        here, said = self.output(path)
        if here is not None:
            return here, said
        named = str(path).strip().strip("`'\"")
        clean = posixpath.normpath(named.removeprefix("./")) if named else ""
        if clean in (WORK, f"/{WORK}", f"{APP}{WORK}", str(self.dir)):
            return self.dir, WORK
        here, said = self.local(named) if clean else (None, "")
        if here is not None and here.exists():
            return here, said
        name = posixpath.basename(clean)
        same = [file for file in sorted(self.out.rglob("*")) if file.name == name and file.is_file()
                and inside(file, self.out)] if name and self.out.is_dir() else []
        return (same[0], f"{self.out}/{same[0].relative_to(self.out).as_posix()}") if len(same) == 1 else (None, "")

    def described(self, limit: int | None = None) -> str:
        """Every input file's `inspect` with no rows (its shape and each column's type, empties, range and first
        value; the digest already shows rows), each within `FILE_TOKENS`, all within `limit` (`DESCRIBED_TOKENS` by
        default; the byte fallbacks where the tokenizer does not load, agent/n11/measure.py); the files past it named.
        The loop's first message carries it, so the model need not spend a request on `inspect` before it writes."""
        unit = measure.UNIT
        limit = measure.bound(DESCRIBED_TOKENS, DESCRIBED_BYTES) if limit is None else limit
        each = measure.bound(FILE_TOKENS, FILE_BYTES)
        data_dir = Path(self.pathmap.data_dir)
        files = sorted(path for path in data_dir.rglob("*") if path.is_file()) if data_dir.is_dir() else []
        parts, left = [], []
        for path in files:
            name = path.relative_to(data_dir).as_posix()
            try:
                text = describe(path, 0)
            except Exception as exc:  # noqa: BLE001 - a file that cannot be read is said, never raised
                text = f"could not be read ({type(exc).__name__})"
            part = fit(f"- {name} ({path.stat().st_size:,} B): {text}", each, unit=unit)
            if unit.size("\n".join([*parts, part])) + 1 > limit:
                left.append(name)
                continue
            parts.append(part)
        if left:
            parts.append(f"- not described here (`inspect` shows them): {', '.join(left[:40])}"
                         + (f" and {len(left) - 40} more" if len(left) > 40 else ""))
        return self.redact("\n".join(parts))

    def answered(self, path: str, parts: dict[str, str], rows=None, tail: str = "") -> str | None:
        """`inspect`'s observation of an input file the loop's first message describes (`parts`, `carried`), from that
        description: `inspect`'s head, the description as the first message carried it, then the rows `inspect` shows
        past it (`describe`'s first and last), within the same bound; None where `path` names no such file (an output's,
        a file under work/, an input past the description's bound, none), which `inspect` answers."""
        if self.output(path)[0] is not None:
            return None
        here = self.data(path)
        try:
            name = here.resolve().relative_to(Path(self.pathmap.data_dir).resolve()).as_posix() if here else ""
        except (OSError, ValueError):
            return None
        if name not in parts or not here.is_file():
            return None
        try:
            rows = max(0, min(MAX_ROWS, int(rows))) if rows is not None else 3
        except (TypeError, ValueError):
            rows = 3
        try:
            bare, shown = describe(here, 0), describe(here, rows)
            added = shown[len(bare):] if rows and shown.startswith(bare) else ""
        except Exception:  # noqa: BLE001 - a file that cannot be read again: the description alone
            added = ""
        return fit(self.redact(f"inspect {path} ({here.stat().st_size:,} B): {parts[name]}{added}"), BOUNDS["inspect"],
                   tail)


NOT_DESCRIBED = ("\n- not described here (", "\n\nnot described here: ")    # the descriptions' notes of the files
#                                                                     past their bound (`Workspace.described`)


def carried(text: str, data_dir: Path) -> dict[str, str]:
    """Each input file the loop's first message describes (`text`, its `# Input files, described`), by its name under
    `data_dir`, with what the text says of it after its head (`portraits`); a file named only in the note of the files
    past the bound, or whose portrait was cut to its bound, is not fully described."""
    found = {}
    for name, _, start, end, _ in portraits(text, data_dir):
        part = text[start:end]
        if not any(marker in part for marker in ("[... cut to the bound]", "[... cut]")):
            found[name] = part
    return found


def portraits(text: str, data_dir: Path) -> list[tuple[str, int, int, int, str]]:
    """Each input file under `data_dir` that `text` describes (`Workspace.described`'s `- <name> (<size> B): ` entries,
    and the `<name> (<size> B...): ` form), in the text's order: its name, where its head starts, where what the text
    says of it starts and ends (the next file's head, the separator between them left out, or the note naming the files
    past the bound), and that separator."""
    data_dir = Path(data_dir)
    files = sorted(path for path in data_dir.rglob("*") if path.is_file()) if data_dir.is_dir() else []
    heads = []
    for path in files:
        name = path.relative_to(data_dir).as_posix()
        said = f"{name} ({path.stat().st_size:,} B"
        for head in (f"- {said}): ", f"{said}): ", f"{said}; the task names it): "):
            if found := re.search("^" + re.escape(head), text, re.M):
                heads.append((found.start(), found.end(), name, "\n" if head.startswith("- ") else "\n\n"))
                break
    heads.sort()
    found = []
    for k, (at, start, name, separator) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(text)
        notes = [stop for stop in (text.find(note, start, end) for note in NOT_DESCRIBED) if stop >= 0]
        end = min(notes, default=end)
        if not notes and k + 1 < len(heads) and text[start:end].endswith(separator):
            end -= len(separator)
        found.append((name, at, start, end, separator))
    return found


def inside(here: Path, root: Path) -> bool:
    """Whether `here`, its links followed, lies under `root` or is it."""
    try:
        return here.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def parameter_files(data_dir: Path) -> list[tuple[str, list[str]]]:
    """Each parameters file among the inputs (its name holding a `PARAM_WORDS` word, of a kind in `PARAM_KINDS`) with
    the top-level keys a reader names (`top_keys`), in the data directory's order; the unread-parameter finding's."""
    data = Path(data_dir)
    found = []
    for file in sorted(data.rglob("*")) if data.is_dir() else []:
        words = set(re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", file.stem).lower()))
        if file.is_file() and file.suffix.lower() in PARAM_KINDS and words & PARAM_WORDS:
            keys = top_keys(file)
            if keys:
                found.append((file.relative_to(data).as_posix(), keys))
    return found


def top_keys(path: Path) -> list[str]:
    """A parameters file's top-level keys in its order (a JSON or TOML object's; a YAML or key-value text's lines at the
    first column, `KEY_LINE`), but a key opening with `_` (a comment's, a watermark's) and one whose value is prose
    (`prose`); [] where it reads as none."""
    try:
        if path.stat().st_size > PARAM_BYTES:
            return []
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        suffix = path.suffix.lower()
        if suffix == ".json" or (not suffix and text.lstrip()[:1] == "{"):
            doc = json.loads(text)
            pairs = list(doc.items()) if isinstance(doc, dict) else []
        elif suffix == ".toml":
            pairs = list(tomllib.loads(text).items())
        else:
            pairs = [(found.group(1), found.group(2).strip().strip("'\"")) for found in map(KEY_LINE.match,
                                                                                         text.splitlines()) if found]
    except (OSError, ValueError):               # tomllib's and json's errors are ValueErrors
        return []
    return list(dict.fromkeys(key for key, value in pairs if isinstance(key, str) and key and not key.startswith("_")
                              and not prose(value)))


def prose(value) -> bool:
    """A value that describes rather than sets: a string opening with a letter and holding a space."""
    said = value.strip() if isinstance(value, str) else ""
    return said[:1].isalpha() and any(char.isspace() for char in said)


def comparison_nodes(tree: ast.AST) -> set[ast.AST]:
    """Expressions used only to compare outputs, including local expected values."""
    pure_calls = {"abs", "all", "any", "round", "float", "int", "isclose", "allclose"}

    def pure(expr: ast.AST) -> bool:
        return all(not isinstance(node, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)) and
                   (not isinstance(node, ast.Call) or
                    (isinstance(node.func, ast.Name) and node.func.id in pure_calls) or
                    (isinstance(node.func, ast.Attribute) and node.func.attr in pure_calls))
                   for node in ast.walk(expr))

    ignored = set()
    for node in ast.walk(tree):
        expr = node.test if isinstance(node, ast.Assert) else node if isinstance(node, ast.Compare) else None
        if expr is not None and pure(expr):
            ignored.update(ast.walk(expr))
    assignments = {}
    loads = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments.setdefault(node.targets[0].id, []).append(node.value)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.setdefault(node.id, set()).add(node)
    changed = True
    while changed:
        changed = False
        for name, values in assignments.items():
            if len(values) == 1 and loads.get(name) and loads[name] <= ignored and pure(values[0]):
                fresh = set(ast.walk(values[0])) - ignored
                if fresh:
                    ignored.update(fresh)
                    changed = True
    return ignored


def literal_numbers(text: str) -> list[float]:
    """A script's numeric literals, excluding expressions used only to compare output values."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return [float(found) for found in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])", text)]
    ignored, found = comparison_nodes(tree), []
    for node in ast.walk(tree):
        if node in ignored:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            found.append(float(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(
                node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)) and not isinstance(
                node.operand.value, bool):
            found.append(-float(node.operand.value))
    return found


def unread_finding(found) -> str:
    """The unread parameters (`Workspace.unread`) as the finish's finding, a broken rule naming each key and its file,
    never a count or a value; "" where none."""
    if not found:
        return ""
    files = list(dict.fromkeys(name for name, _ in found))
    names = [f"`{key}`" + (f" in {name}" if len(files) > 1 else "") for name, key in found]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"the task's figures depend on {listed}" + (f" in {files[0]}" if len(files) == 1 else "") + \
        ", which no script reads"


def unread_said(found) -> str:
    """The unread-parameter finding, one line naming each key and its file, never a count or a value; "" where none."""
    if not found:
        return ""
    files = list(dict.fromkeys(name for name, _ in found))
    names = [f"`{key}`" + (f" in {name}" if len(files) > 1 else "") for name, key in found]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"the parameter{'s' if len(names) > 1 else ''} {listed}" + (f" in {files[0]}" if len(files) == 1 else "")
            + f" {'are' if len(names) > 1 else 'is'} not read by any script")


def verdict(failures) -> str:
    """The checks' lines after a run: `checks: all pass`, or each failing line, at most 12 and the rest counted."""
    return ("checks: all pass" if not failures else
            "checks failing:\n" + "\n".join(f"- {line}" for line in failures[:12]) +
            (f"\n- and {len(failures) - 12} more" if len(failures) > 12 else ""))


def unfenced(text: str) -> str:
    """A file's text with a fence the model wrapped it in removed: a first line opening a block and a last closing
    it."""
    lines = (text or "").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return "\n".join(lines) + "\n" if lines else ""


def shadows(named: str) -> str:
    """The standard-library module a path under work/ would shadow (its file's stem or a directory's name), "" where
    none."""
    parts = named.split("/")[1:]
    names = [*parts[:-1], parts[-1].removesuffix(".py")] if parts and parts[-1].endswith(".py") else parts[:-1]
    return next((name for name in names if name in SHADOWED), "")


def imported(path: Path) -> list[str]:
    """The top-level modules a Python file imports (`import a.b`, `from a import b`, `from . import b`), in order; []
    where it does not parse."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return []
    found: dict[str, None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(dict.fromkeys(alias.name.split(".")[0] for alias in node.names))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found[node.module.split(".")[0]] = None
            elif node.level:
                found.update(dict.fromkeys(alias.name for alias in node.names))
    return list(found)


PACKAGED = ("{file} imports `{form}`: the files under {work}/ import each other by their bare names (`{bare}`), "
            "since the scripts run from {work}/, where no package is named {work}; write the imports that way{more}")


def packaged(directory: Path) -> list[tuple[str, str, str]]:
    """The imports in the files under work/ (`directory`) that name its modules as a package (`from work.utils import
    f`, `import work.utils`, `from work import utils`), each (the file, the import as written, the import by the bare
    name), in the files' order; [] where none does."""
    found = []
    for path in sorted(directory.rglob("*.py")) if directory.is_dir() else []:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        named = f"{WORK}/{path.relative_to(directory).as_posix()}"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level and node.module and \
                    (node.module == WORK or node.module.startswith(WORK + ".")):
                names = ", ".join(alias.name for alias in node.names[:3]) + (", ..." if len(node.names) > 3 else "")
                rest = node.module[len(WORK) + 1:]
                found.append((named, f"from {node.module} import {names}",
                              f"from {rest} import {names}" if rest else f"import {names}"))
            elif isinstance(node, ast.Import):
                found += [(named, f"import {alias.name}", f"import {alias.name[len(WORK) + 1:]}")
                          for alias in node.names if alias.name.startswith(WORK + ".")]
    return found


def packaged_said(found: list[tuple[str, str, str]], first: str = "") -> str:
    """The note on package-style imports (`packaged`): one, as written and by its bare name (the file `first`'s where it
    has one, else the first found); "" where none."""
    if not found:
        return ""
    file, form, bare = next((each for each in found if each[0] == first), found[0])
    more = "; the other files' such imports likewise" if len({each[0] for each in found}) > 1 else \
        "; its other such imports likewise" if len(found) > 1 else ""
    return PACKAGED.format(file=file, form=form, bare=bare, work=WORK, more=more)


def has_main_block(text: str) -> bool:
    """Whether a script holds an `if __name__ == "__main__":` block at its top level."""
    return bool(MAIN_BLOCK.search(text or ""))


def compiled(text: str, named: str) -> str:
    """"" where `text` compiles as the file `named`, else what a write says of it: the error and the three lines around
    it, or the error's kind alone where it names no line."""
    try:
        with warnings.catch_warnings():            # the model's invalid escapes warn on stderr, which is the log's
            warnings.simplefilter("ignore")
            compile(text, named, "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"does not compile: {type(exc).__name__} at line {exc.lineno}: {exc.msg}\n" + around(text, exc.lineno)
    except (ValueError, MemoryError, RecursionError) as exc:
        return f"does not compile ({type(exc).__name__})"
    return ""


def installed(name: str) -> bool:
    """Whether a top-level module is importable here (the agent's interpreter is the runs')."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def around(text: str, line: int | None) -> str:
    """The three lines around `line` (from 1), numbered, the line itself marked."""
    if not line:
        return ""
    lines = text.split("\n")
    shown = []
    for number in range(max(1, line - 1), min(len(lines), line + 1) + 1):
        shown.append(f"{'>' if number == line else ' '} {number:4d} | {lines[number - 1][:SOURCE_CHARS]}")
    return "\n".join(shown)


def source_line(path: Path, line: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for number, text in enumerate(handle, 1):
                if number == line:
                    text = text.strip()
                    return text[:SOURCE_CHARS] + (" ..." if len(text) > SOURCE_CHARS else "")
    except OSError:
        pass
    return ""


def last_error(stderr: str) -> str:
    """The exception's own line: the last line of the error output that names an error, else its last line."""
    found = ERROR.findall(stderr)
    if found:
        return found[-1].strip()[:400]
    lines = [line for line in stderr.strip().split("\n") if line.strip()]
    return lines[-1].strip()[:400] if lines else ""


# ------------------------------------------------------------------------------------------------- inspect


def describe(path: Path, rows: int) -> str:
    """A data file described by its kind."""
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pqt"):
        return parquet(path, rows)
    if suffix in (".json", ".jsonl", ".ndjson") or (suffix == "" and path.read_bytes()[:1] in (b"{", b"[")):
        return json_tree(path)
    if suffix == ".xlsx":
        return digest.xlsx(path)
    if suffix in (".zip",):
        return digest.zip_listing(path)
    if digest.binary(path):
        return "a binary file"
    head = path.read_bytes()[:65_536].decode("utf-8", errors="replace")
    if suffix in TABLES and table_like(head):
        return table(path, rows)
    return text_file(path)


def table_like(head: str) -> bool:
    lines = [line for line in head.split("\n")[:20] if line.strip()]
    if len(lines) < 2:
        return False
    for delimiter in (",", "\t", ";", "|"):
        counts = {line.count(delimiter) for line in lines[:-1]}
        if len(counts) == 1 and counts.pop() >= 1:
            return True
    return False


def sniff(head: str) -> str:
    lines = [line for line in head.split("\n")[:20] if line.strip()]
    best = max((",", "\t", ";", "|"), key=lambda d: (len({line.count(d) for line in lines[:-1]}) == 1,
                                                       lines[0].count(d) if lines else 0))
    return best


def number(cell: str) -> float | None:
    try:
        value = float(cell)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def table(path: Path, rows: int) -> str:
    raw = path.read_bytes()[:65_536].decode("utf-8-sig", errors="replace")
    delimiter = sniff(raw)
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        body: list[list[str]] = []
        last: list[list[str]] = []
        count = 0
        for row in reader:
            if not row:
                continue
            count += 1
            if count <= TABLE_ROWS:
                body.append(row)
            last.append(row)
            if len(last) > rows:
                last.pop(0)
    width = len(header)
    lines = [f"a table, {count:,} rows x {width} columns (delimiter {delimiter!r}"
             + (f"; statistics over the first {TABLE_ROWS:,} rows" if count > TABLE_ROWS else "") + ")"]
    for index, name in enumerate(header[:COLUMNS]):
        cells = [row[index].strip() for row in body if index < len(row)]
        empty = sum(cell.lower() in EMPTY for cell in cells)
        filled = [cell for cell in cells if cell.lower() not in EMPTY]
        kind = digest.kind(filled[:2000]) if filled else "empty"
        numbers = [value for value in (number(cell) for cell in filled) if value is not None]
        if numbers and len(numbers) == len(filled):
            low, high = f"{min(numbers):.6g}", f"{max(numbers):.6g}"
        else:
            low, high = (min(filled)[:24], max(filled)[:24]) if filled else ("-", "-")
        first = (filled[0] if filled else "")[:24]
        lines.append(f"- {name[:40]}: {kind}, {empty} empty, min {low}, max {high}, first {first}")
    if width > COLUMNS:
        lines.append(f"- and {width - COLUMNS} more columns: " + ", ".join(name[:30] for name in header[COLUMNS:80]))
    if rows:
        lines.append(f"first {min(rows, len(body))} rows:")
        lines += [digest.short(delimiter.join([*header]), 300)] + [digest.short(delimiter.join(row), 300)
                                                                    for row in body[:rows]]
        if count > rows:
            lines.append(f"last {len(last)} rows:")
            lines += [digest.short(delimiter.join(row), 300) for row in last]
    return "\n".join(lines)


def parquet(path: Path, rows: int) -> str:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError:
        return digest.parquet(path)
    table_ = pq.read_table(path)
    lines = [f"parquet, {table_.num_rows:,} rows x {table_.num_columns} columns"]
    for name, column in list(zip(table_.column_names, table_.columns))[:COLUMNS]:
        try:
            bounds = pc.min_max(column).as_py()
            low, high = str(bounds.get("min"))[:24], str(bounds.get("max"))[:24]
        except Exception:  # noqa: BLE001 - a type min_max does not take
            low = high = "-"
        first = str(column[0].as_py())[:24] if len(column) else ""
        lines.append(f"- {name}: {column.type}, {column.null_count} null, min {low}, max {high}, first {first}")
    if table_.num_columns > COLUMNS:
        lines.append(f"- and {table_.num_columns - COLUMNS} more columns")
    if rows and table_.num_rows:
        head, tail = table_.slice(0, rows).to_pylist(), table_.slice(max(0, table_.num_rows - rows)).to_pylist()
        lines.append(f"first {len(head)} rows:")
        lines += [digest.short(json.dumps(row, default=str), 300) for row in head]
        if table_.num_rows > rows:
            lines.append(f"last {len(tail)} rows:")
            lines += [digest.short(json.dumps(row, default=str), 300) for row in tail]
    return "\n".join(lines)


def json_tree(path: Path) -> str:
    if path.stat().st_size > JSON_BYTES:
        return digest.head(path, TEXT_HEAD, "JSON (too large to parse here)")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        doc = json.loads(text)
    except ValueError:
        try:
            doc = [json.loads(line) for line in text.splitlines() if line.strip()]
            return f"JSON lines, {len(doc)} records; the first's key tree:\n" + "\n".join(
                digest.tree(doc[0] if doc else {}, limit=4))
        except ValueError:
            return "not valid JSON; its head:\n" + text[:TEXT_HEAD]
    return f"JSON ({digest.label(doc)}); its key tree to depth 4:\n" + "\n".join(digest.tree(doc, limit=4))


def text_file(path: Path) -> str:
    text = path.read_bytes()[:2_000_000].decode("utf-8", errors="replace")
    tags = Counter(match.lower() for match in TAG.findall(text))
    said = f"text, {text.count(chr(10)):,} lines; its first {min(TEXT_HEAD, len(text)):,} characters:\n" \
           f"{text[:TEXT_HEAD]}"
    if tags and path.suffix.lower() in (".xml", ".html", ".htm", ".xbrl", ".txt", ""):
        said += "\ntag counts: " + ", ".join(f"{tag} {count}" for tag, count in tags.most_common(20))
    return said
