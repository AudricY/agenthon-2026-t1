"""N11's five commands, the checks, and the two reply forms that call them.

Each command returns an observation built by code, within its bound in bytes (`BOUNDS`), at no request:
- `write(path, text)`: a file under `work/` alone (its absolute forms, `/work/<name>`, `/app/work/<name>` and the
  instruction's `/app/<name>.py`, taken as `work/<name>`: `work_form`, lane tools1), of any length where it arrived
  whole (lane n11d: the reply's cap is the real limit, and the House is told to write the solution as one
  `work/main.py` where one reply holds it and to split it into modules main.py imports only where a reply would be
  cut, in shapes, never a count: lanes state3 and files1); a name that shadows a standard-library module
  (`inspect.py`, `json.py`: `sys.stdlib_module_names`) refused; a `.py` file compiled, its SyntaxError said with the
  three lines around it; every refusal names the form accepted. Lane fixdef1's `defs` (agent/n11/solve.py
  `AGENTHON_N11_WRITE`, the House's own writes alone): a text holding only top-level definitions for a `.py` file that
  exists replaces those definitions by name and keeps the rest (`splice`, `Workspace.write`), the tool's words saying
  so (`TOOLS_DEFS`, `DEFS_SAID`);
- `run(path)`: a `.py` file under `work/` run as `python3 <file>` from `work/`, in its own process group, its streams
  through the bounded writer (common.runner.run_bounded), the House's variables withheld and `OUTPUT_DIR` naming the
  output (common.runner.child_env, lane outguard1); the exit and seconds, the
  ends of stdout, the end of stderr with the traceback's frames inside `work/` (file, line, function, source line),
  the output's files written or left (common.contract.listing); after an exit 0 the output's digest and the findings
  (`Workspace.sanity`: the instruction's own properties broken and its keys and columns missing, agent/n11/props.py,
  lane n11e; the sanity checks', agent/n11/sanity.py, lane n11d), each kept whole, and the keys of the input's
  parameters files no script under `work/` names (`unread`, lane tools1), and apart from them the span and the
  scale findings (`Workspace.against`, agent/n11/conventions.py, lane convcard1), lane derive1's paragraph where
  armed and lane contract1's (`Workspace.contracted`, agent/n11/contract.py: the output against the contract its
  instruction states) where armed; then the checks, where lane recon1's reconciliation is armed (`Workspace.reconciled`,
  agent/n11/recon.py) after a clean run with a finding standing its verdict line and its findings (`held_said`) in
  place of the checks' passing line; under
  `AGENTHON_N11_CALLED=on` (lane called1; agent/n11/solve.py), whatever the exit, beside the findings after the
  instruction's and the contract's, a line for each function whose card reached the House that a file under `work/`
  computes by hand (`Workspace.called`, `called_said`: `library.rewritten`'s rule, the def or routine named);
- `inspect(path, rows)`: a data file of the unit's: a table's shape, columns with type, empty count, min, max and
  first value, its first and last rows; JSON's key tree; text, XML and HTML's size, head and tag counts; a file in the
  output directory or under `work/` alike, and either directory listed (`made`, lane tools1);
  under `AGENTHON_N11_INSPECT=described` (lane first1; agent/n11/solve.py) an input file the loop's first message
  describes is answered from that description (`Workspace.answered`, `carried`), its rows as `inspect` shows them;
- `lib(name=...)` or `lib(shelf=...)`: the library's documentation (agent/n11/library.py);
- `finish(reason)`: the checks and the sanity findings once more (the agent decides what follows:
  agent/n11/solve.py; `reason`, "accept: <why>", answers a finish refused for its findings).
Under `AGENTHON_N11_RULES=on` (lane rules1; agent/n11/solve.py `RULES`) the workspace reads its rules (`Workspace(...,
rules=True)`): `sanity` says the eight rules' findings (`RULES`; docs/research/wrong-1013.md, section 5) among the
findings, after a run's exit 0 and at the finish alike, each known by its rule (`Workspace.ruled`, `Ran.rules`): the
four that read the instruction as properties (agent/n11/props.py `RULE_KINDS`, said as `instruction:` lines), the three
that read the output alone after the sanity checks' (agent/n11/sanity.py `ruled`), and `literal` last (`literals`: a
script under work/ writing the instruction's example-output figures as literals, where `unread` reads the scripts);
and `unread_finding` says the unread parameters as a broken rule, the finish's (agent/n11/solve.py `finish`).
Under `AGENTHON_N11_RULES=wide` (lane rules2; `Workspace(..., rules=True, wide=True)`) the same step says six more
(`WIDE_RULES`; docs/research/wrong-1014.md, section 3) after the eight's, over the output the instruction's check reads
(agent/n11/props.py `wide_found`), each kept as the eight's are.
The checks (`Workspace.checks`): every output file the instruction names present at its path, parsing as its kind,
its numbers finite, a file at the output's root and none holding a canary (common.contract.check over the named
files), and the self-report scan (common.selfreport.flags) without its bound on `*_error` keys (lane n11d: E039's
false alarms on it cost 11 and 19 turns). Every observation has the unit's canaries marked `<canary>`.

A reply's calls (`Call`) come from its native `tool_calls` (`from_tools`) or, in the fenced form, from its text
(`from_fenced`): one block per file headed `python file=work/<name>.py` (a block naming another file of any kind is
that file's, never work/main.py's: `labelled`), and `inspect`, `lib`, `run` and `finish` as
one-line directives (`finish reason=...` or `finish accept: ...` with its reason). A call the output cap cut is
marked `cut` and never run (a cut write's text is salvaged by the agent: `cut_arguments`); one whose name or
arguments cannot be read is marked with its `problem`.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import itertools
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

from common import canary, contract, digest, paths, runner, selfreport, trace
from common.guard import WRITING
from n11 import contract as stated_contract
from n11 import library, lint, measure, props, sanity, shim, snapshot

WORK = "work"                   # the directory the House writes under, as it names it
BOUNDS = {"write": 600, "run": 7_000, "inspect": 3_000, "lib": 2_500, "finish": 1_500}   # run's 6,000 raised by
#                                 the digest (lane n11d), which the observation keeps whole with the checks
SHADOWED = frozenset(sys.stdlib_module_names)   # a file or directory under work/ so named shadows the module
WRITE_FORM = f'write(path="{WORK}/<name>.py", text=<the whole file>), the path relative, under {WORK}/'
WORK_ROOTS = ("/work/", "/app/work/")   # lane tools1: the House's absolute forms of work/, each taken as work/ (and the
APP = "/app/"                   #         instruction's `/app/<name>.py`, a script it asks saved there)
RUN_DATA = ("/app/environment/data", "/app/data")    # lane clock1 of 10-13: the unit image's data directories, and
RUN_TASK = "/input"             # its task's, as a run's script opens them (`run_table`)
RUN_RELATIVE = ("data", "environment/data", "output", WORK)     # and the same relative to /app (a script run there)
QF_PACKAGE = Path(__file__).resolve().parent.parent / "qf"      # the library's package, for the write's check
PARAM_WORDS = frozenset({"param", "params", "parameter", "parameters", "config", "configuration", "settings", "cfg",
                         "conf"})   # lane tools1: an input file whose name holds one is a parameters file
PARAM_KINDS = (".json", ".toml", ".yaml", ".yml", ".txt", ".ini", ".cfg", ".conf", ".env", ".properties", "")
PARAM_BYTES = 65_536            # a larger parameters file is not read for its keys
PARAM_NAMES = 3                 # the keys the unread-parameter finding names, at most: the harness's bound, never said
RULES = ("fraction", "pct_level", "vol_scale", "siblings", "inline_header", "literal", "row_sum", "formula")   # lane
#                                 rules1's eight, as docs/research/wrong-1013.md names them (`RULE_OF` the props' kinds)
RULE_OF = {"fraction": "fraction", "header": "inline_header", "rowsum": "row_sum", "formula": "formula"}
WIDE_RULES = props.WIDE_RULES   # lane rules2's six, under `RULES=wide` after the eight
LITERALS = 3                    # `literal`: the example's figures a script writes as literals, at the least; never said
LITERAL_NAMES = 4               # and those its finding names, at most
KEY_LINE = re.compile(r"^([A-Za-z_][\w.-]*)\s*[:=]\s*(.*)$")    # a YAML or key-value text's top-level line
CALLED_HEAD = "library: "       # lane called1: the called finding's head, beside props' `instruction:` and `contract:`
CALLED_SAID = {"def": CALLED_HEAD + "`{name}` in {file} computes by hand what `{call}` computes: call it under the "
                                    "task's conventions, or accept at finish where the task's formula differs",
               "routine": CALLED_HEAD + "{file} calls `{name}` to compute by hand what `{call}` computes: call it "
                                        "under the task's conventions, or accept at finish where the task's formula "
                                        "differs"}     # a def that computes a carded function, or its formula's routine
MAIN_BLOCK = re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""", re.M)
MALFORMED_BYTES = 600           # a malformed or cut call's observation
STDOUT_ENDS = 1_000             # stdout's first and last characters shown
STDOUT_CUT = "\n[... cut ...]\n"  # between them where stdout is longer: said as cut, never as a count (lane state3)
SPLIT = (f"into as few files as fit, each a module {WORK}/main.py imports, and {WORK}/main.py last; write one file "
         "a reply")             # the split a cut reply asks for, a shape (lane files1; the owner's in304, in308)
STDERR_TAIL = 2_000             # stderr's last characters shown
FRAMES = 8                      # the traceback's frames inside work/ shown, the innermost kept
SOURCE_CHARS = 160              # a frame's source line, at most
MAX_ROWS = 10                   # inspect's head and tail rows, each, at most
TABLE_ROWS = 200_000            # a table's rows read for inspect's statistics
COLUMNS = 40                    # columns described, at most
TEXT_HEAD = 1_500               # a text file's head shown
JSON_BYTES = 20_000_000         # a JSON file larger than this is not parsed
DESCRIBED_TOKENS = 2_850        # the input files' descriptions the loop's first turn carries, at most, in the model's
#                                 tokens (lane tokens1; agent/n11/measure.py): the most the byte bound admitted on the
#                                 58 dev units (2,838), so none is described less in tokens than in bytes
FILE_TOKENS = 1_450             # and one file's: the most its byte bound admitted (1,429, a table of digits, a digit a
#                                 token; a filing's prose runs 3.5 bytes a token, one file's description 2.04 at the median)
DESCRIBED_BYTES_PER_TOKEN = 2.11    # the densest description at its bound, and one file's (the fallbacks' ratios)
FILE_BYTES_PER_TOKEN = 1.05
DESCRIBED_BYTES = 6_000         # the byte fallbacks where the tokenizer does not load: DESCRIBED_TOKENS at its ratio
FILE_BYTES = 1_500              # (6,014) and FILE_TOKENS at its (1,523), the values kept
NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
FRAME = re.compile(r'^\s*File "(?P<file>[^"\n]+)", line (?P<line>\d+), in (?P<name>[^\n]+)$', re.M)
ERROR = re.compile(r"^(?:[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)|[A-Z]\w*Error)\b.*$", re.M)
QF_NAMED = re.compile(r"(?<![\w.])qf\.(?:[A-Za-z_]\w*\.)?([A-Za-z_]\w*)")    # lane cardfail1 (`failed_in`): a library
#                                                 function an error's line names, as the guard's refusals do
CALL_REFUSED = re.compile(r"^TypeError: ([A-Za-z_]\w*)\(\) (?:got|missing|takes)\b")    # Python's words refusing a
#                                                 call's arguments, the function named first
TAG = re.compile(r"<([A-Za-z][\w:.-]*)")
ADDRESS = re.compile(r"0x[0-9a-fA-F]+")      # an object's address in a message, which differs run to run
TABLES = (".csv", ".tsv", ".txt", ".dat")
EMPTY = {"", "nan", "na", "n/a", "null", "none", "-nan", "inf", "-inf"}

TOOLS = [
    {"type": "function", "function": {
        "name": "write", "description": f"Save a file under {WORK}/, replacing it whole. A file that arrives whole "
                                        "is saved whole; a write the reply's output limit cuts is not saved, only its "
                                        f"complete definitions as a draft, so write the solution as one {WORK}/main.py "
                                        f"where one reply holds it, and split it into modules {WORK}/main.py imports "
                                        "only where it would not. A .py file is compiled and its syntax error, if any, "
                                        f"reported. Then {WORK}/main.py runs where it was written, or where "
                                        "it imports the module written and its last run was not clean (or the file "
                                        "written runs, where it has an `if __name__ == \"__main__\":` block and "
                                        f"{WORK}/main.py does not import it).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": f"the file's path, under {WORK}/, e.g. {WORK}/main.py"},
            "text": {"type": "string", "description": "the file's whole text"}},
            "required": ["path", "text"]}}},
    {"type": "function", "function": {
        "name": "run", "description": f"Run a .py file under {WORK}/ with python3. It reports the exit, the output's "
                                      "ends, the traceback, the files in the output directory, the checks and the "
                                      "budget.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": f"the .py file to run, e.g. {WORK}/main.py"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "inspect", "description": "Describe one input data file: a table's shape, columns, types, ranges "
                                          "and first and last rows; a JSON file's key tree; a text file's head.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "the data file, as the task or the paths name it"},
            "rows": {"type": "integer", "description": "head and tail rows to show (the harness caps them)"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "lib", "description": "The library's documentation: a function's whole docstring (name), or a "
                                      "module's signatures and one-line cards (shelf).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "a function of qf, e.g. asian_price"},
            "shelf": {"type": "string", "description": "a module of qf, e.g. options"}}}}},
    {"type": "function", "function": {
        "name": "finish", "description": "End the task: the checks and the sanity findings run once more and, "
                                         "where they pass, the files in the output directory are the answer. Only "
                                         "finish ends the task.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "after a finish refused for sanity findings: \"accept: "
                                                        "<why the values stand>\""}}}}},
]
NAMES = tuple(tool["function"]["name"] for tool in TOOLS)
REQUIRED = {tool["function"]["name"]: tuple(tool["function"]["parameters"].get("required", ())) for tool in TOOLS}
# lane fixdef1 (`AGENTHON_N11_WRITE=defs`): what a write saves, said where `write` and the system prompt said "whole"
DEFS_SAID = ("a file that already exists is replaced whole by a whole text, and a text holding only top-level "
             "definitions (functions, classes, imports and plain assignments) for a .py file that exists replaces "
             "those definitions by name and keeps the rest, so a fix of one function is that function alone")
WHOLE_SAID = (", replacing it whole.", "the file's whole text")     # the write tool's words `DEFS_WORDS` replace
DEFS_WORDS = (f"; {DEFS_SAID}.",
              "the file's whole text, or, for a .py file that exists, only the definitions it replaces")
TOOLS_DEFS = json.loads(json.dumps(TOOLS))
TOOLS_DEFS[0]["function"]["description"] = TOOLS[0]["function"]["description"].replace(WHOLE_SAID[0], DEFS_WORDS[0], 1)
TOOLS_DEFS[0]["function"]["parameters"]["properties"]["text"]["description"] = \
    TOOLS[0]["function"]["parameters"]["properties"]["text"]["description"].replace(WHOLE_SAID[1], DEFS_WORDS[1], 1)


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
    id: str = ""
    raw: str = "{}"                 # the arguments as the reply wrote them, sent back as written where they read
    problem: str | None = None      # why it cannot be run, where it is malformed
    cut: bool = False               # the output cap cut inside it: never run
    wrote: str = ""                 # the arguments as the reply wrote them, kept where they do not read (a cut call's)

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


def from_tools(reply, turn: int, cap: int | None = None) -> list[Call]:
    """A reply's native calls, in order; where the output cap cut the reply, its last call marked `cut`: its finish
    `length`, or, `cap` given (lane redirect1's `AGENTHON_N11_REDIRECT=write`), its completion at `cap` tokens whatever
    finish the route said (a pooled route's `tool_calls` at the cap, the last call's arguments cut). A reply with a
    `<tool_call>` the route did not parse into a call has it as one malformed call."""
    calls = []
    for number, made in enumerate(reply.tool_calls, 1):
        arguments = made.arguments if made.arguments is not None else None
        calls.append(Call(made.name, arguments if isinstance(arguments, dict) else {}, f"call_{turn}_{number}",
                          made.raw if isinstance(arguments, dict) else "{}", checked(made.name, arguments),
                          wrote=made.raw or ""))
    if not calls and "<tool_call>" in (reply.text or ""):
        calls.append(Call("unparsed", {}, f"call_{turn}_1", "{}", "the reply wrote a <tool_call> the route could not "
                                                                   "read as a call"))
    if reply.finish_reason == "length" or (cap is not None and int(reply.completion_tokens or 0) >= cap):
        if calls:
            calls[-1].cut, calls[-1].problem = True, None      # cut, not malformed: its arguments end at the cap
        else:
            calls.append(Call("cut", {}, f"call_{turn}_1", "{}", None, True))
    return calls


# ---------------------------------------------------------------------------------------------- the fenced blocks
# Lane fence1: the block grammar both fenced parsers read (`from_fenced` here, agent/n11/state.py `read`).

FENCE = re.compile(r"^\s*(`{3,})(.*)$")
TAG = re.compile(r"^[A-Za-z][\w+-]*$")
PYTHON = ("python", "python3", "py")
KINDS = ("py", "txt", "json", "csv", "tsv", "md", "yaml", "yml", "toml", "ini", "cfg", "sh", "sql")
PATH_LINE = re.compile(r"""^\s*(?P<mark>\#+|//)?\s*(?:(?P<label>file(?:name)?|path)\s*[=:]\s*)?[`'"*]*"""
                       r"""(?P<path>[\w./-]*\w\.\w+)[`'"*]*\s*$""", re.I)
NAMED_LINE = re.compile(r"""^\s*(?P<mark>\#+|//)?\s*(?:file(?:name)?|path)\s*[=:]\s*(?P<quote>[`'"*]*)"""
                        r"""(?P<path>[\w./-]*\w)[`'"*]*\s*$""", re.I)     # lane fencedpath1: a labelled line, any path
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
    """Lane fencedpath1: the path a labelled line names (`file=work/requirements.txt`, `# file: notes.md`, `path:
    work/.env`), a directory's or a known kind's (`KINDS`), so that a block naming another file is never work/main.py's
    (E182 wrote a python block headed `file=work/requirements.txt` over it). In a python or untagged block, which
    would otherwise be main.py's, wherever it points (the write refuses one outside work/ by name); in another
    language's, under work/ alone (an example's output path is no header). None in a python block where the line is
    code: its value quoted, uncommented (`file = "results.json"`)."""
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
    """A path as the House may write it, as `work/<name>` (`./work/a.py`, `a.py`, an absolute form `work_form` takes):
    what two blocks naming one file share."""
    named = posixpath.normpath(work_form(path.strip().strip("`'\"")).removeprefix("./")) if path.strip() else ""
    return named if "/" in named else f"{WORK}/{named}"


def work_form(path: str) -> str:
    """Lane tools1: a path in one of the House's absolute forms of work/ as `work/<name>`: `/work/<name>`,
    `/app/work/<name>`, and `/app/<name>.py` (a script the instruction asks saved there); any other as it is."""
    for root in WORK_ROOTS:
        if path.startswith(root):
            return WORK + "/" + path[len(root):]
    rest = path[len(APP):] if path.startswith(APP) else ""
    return f"{WORK}/{rest}" if rest.endswith(".py") and "/" not in rest else path


def run_table(pathmap, work: Path, out: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Lane clock1 of 10-13 (`AGENTHON_N11_SHIM=on`, agent/n11/shim.py): the paths a run's script may open in the unit
    image's forms, each with this machine's, one table with the write tool's (`work_form`): the path map's (each file
    the instruction names to read; its outputs are the output directory's), the House's forms of work/ (`WORK_ROOTS`)
    and of the output (`paths.OUTPUTS`), the image's data directories (`RUN_DATA`) and task (`RUN_TASK`), a data
    file's bare `/app/<file>`, and `RUN_RELATIVE` from /app; then, read after the instruction's `/app/<name>.py` (`APP`,
    as `work_form` takes it), /app itself as work/ (a script run from there). A row maps only where its own path is
    missing (agent/n11/preload.py); a row naming its own path (this machine is the image's there) is none."""
    data, work, out = Path(pathmap.data_dir), Path(work), Path(out)
    table: dict[str, str] = {}
    for entry in pathmap.entries:
        if entry.kind == "read" and entry.path is not None:
            table[entry.named] = str(entry.path)
    table.update({root.rstrip("/"): str(work) for root in WORK_ROOTS})
    table.update({root: str(out) for root in paths.OUTPUTS})
    table.update({root: str(data) for root in RUN_DATA})
    table[RUN_TASK] = str(data.parent.parent)
    for file in sorted(data.iterdir()) if data.is_dir() else []:
        table.setdefault(f"{APP}{file.name}", str(file))
    table.update({root: str({WORK: work, "output": out}.get(root, data)) for root in RUN_RELATIVE})
    after = [(APP.rstrip("/"), str(work))]
    return sorted((row for row in table.items() if row[0] != row[1]), key=lambda row: row[0]), \
        [row for row in after if row[0] != row[1]]


def directive_line(line: str) -> bool:
    """A line that is one of the forms' directives, alone (`finish`, `finish accept: <why>`, `run work/a.py`,
    `inspect <path>`, `lib <name>`), markdown's emphasis and quoting aside; never a line of Python."""
    text = line.strip().strip("`*").strip()
    return bool(FINISH_LINE.match(text) or VERB_LINE.match(text))


@dataclass
class Fenced:
    """One fenced block of a reply: its language, the path it names (on the fence line, else on the line after it,
    which is then no line of the file: `header` "line"), its lines, whether a closing fence ended it, and why it saves
    nothing (`void`: "empty" where it holds no code, "path" where it held its path line alone, "directive" where it
    holds directive lines alone; "" where it holds a file's text)."""
    tag: str
    path: str | None
    lines: list[str]
    closed: bool
    header: str = ""                # where its path stood: "fence", "line"; "" where it names none
    void: str = ""
    raw: list[str] = field(default_factory=list)    # its lines as the reply wrote them, a path line among them
    words: tuple[str, ...] = ()     # its fence line's words after the backticks (lane resend1: `fence_finish`)
    named: bool = False             # its path a labelled line's that `headed` does not read (`labelled`, fencedpath1)

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    @property
    def directive(self) -> str:
        """Its first line, where it holds directive lines alone."""
        return next((line.strip() for line in self.lines if line.strip()), "") if self.void == "directive" else ""


def fenced_block(rest: str, lines: list[str], closed: bool) -> Fenced:
    """A block from its fence line's words after the backticks (`rest`) and its lines: the path the fence line names,
    else the path its first non-blank line names alone (`headed`, else `labelled`), that line dropped; a first line
    naming the fence's own path dropped too. Then why it saves nothing, if it does not (`Fenced.void`)."""
    tag, path = opened(rest)
    header, dropped, raw = ("fence" if path is not None else ""), False, lines
    first = next((k for k, line in enumerate(lines) if line.strip()), None)
    said = headed(lines[first], tag) if first is not None else None
    other = labelled(lines[first], tag) if first is not None and said is None else None
    said = said or other
    if said is not None and (path is None or plain(said) == plain(path)):
        lines, dropped = lines[:first] + lines[first + 1:], True
        path, header = path or said, header or "line"
    body = [line for line in lines if line.strip()]
    python = path.endswith(".py") if path is not None else tag in PYTHON
    code = [line for line in body if not (python and line.lstrip().startswith("#"))]
    void = ("path" if dropped and not body else "empty") if not code else \
        "directive" if all(directive_line(line) for line in body) else ""
    return Fenced(tag, path, lines, closed, header, void, raw, tuple(rest.split()), header == "line" and bool(other))


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
    """Lane join2: the `run` or `finish` a closed block holds alone, its one line (`directive`); lane resend1: `finish`
    on an empty closed block's fence line (`fence_finish`); None where it holds anything else."""
    lines = [line for line in item.lines if line.strip()]
    if item.closed and not lines and fence_finish(item):
        return "finish", {}
    said = directive(lines[0].strip().strip("`*").strip()) if len(lines) == 1 and item.closed and \
        item.void == "directive" else None
    return said if said is not None and said[0] in ("run", "finish") else None


def fence_finish(item: Fenced) -> bool:
    """Lane resend1: `finish` alone on a block's fence line after its language, if any (```` ```python finish ````).
    A block holding the line `finish` alone is join2's reading already, whatever its fence line says."""
    words = item.words[1:] if item.words and item.words[0].lower() in PYTHON else item.words
    return words == ("finish",)


def from_fenced(text: str, turn: int, lenient: bool = False) -> list[Call]:
    """The fenced form's calls, in order: a block naming its file (`fenced`: on its fence line, `python
    file=work/<name>.py`, or alone on the line after it, which is no line of the file; a labelled line naming any kind,
    `labelled`, whose quoted-code and other-language example exceptions still apply) is a `write` of it, subject
    to the workspace path check (which also refuses dotfiles under work/), and a python or untagged block naming none
    `work/main.py`'s, a block left open at the reply's end a cut one; a block holding no
    code, its path line alone or directive lines alone is no call (lane fence1), nor a block naming no file in another
    language; a later block for a file replaces an earlier, as two writes do. `inspect <path> [rows=N]`, `lib
    name=<f>` or `lib shelf=<m>`, `run <path>` and `finish` (or `finish reason=<why>`, `finish accept: <why>`) outside
    any block are their commands. Under `lenient` (lane join2, N11's `CUT=continue`) a block holding one `run` or
    `finish` alone is that directive (`block_directive`; lane resend1: and `finish` on an empty closed block's fence
    line), and a block left open with nothing in it after a closed one is no call, never a cut write. What runs after a reply's last write is the agent's rule for both forms
    (agent/n11/solve.py `after_write`)."""
    calls: list[Call] = []

    def add(name: str, arguments: dict, cut: bool = False) -> None:
        calls.append(Call(name, arguments, f"call_{turn}_{len(calls) + 1}", json.dumps(arguments),
                          checked(name, arguments), cut))

    closed = False
    for item in fenced(runner.THINK.sub("", text or "")):
        if not isinstance(item, Fenced):
            said = directive(item)
        elif lenient and (said := block_directive(item)) is not None:
            closed = True
        else:
            path = item.path or (f"{WORK}/main.py" if item.tag in PYTHON or not item.tag else None)
            stray = lenient and closed and not item.closed and item.void == "empty"
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
    line: str = ""                                      # the innermost frame's source line in work/ (lane n11e)
    wrote: tuple[str, ...] = ()                         # the output's files it wrote, new or changed (lane loop1)
    stdout: str = ""                                    # its stdout, the canaries marked (lane cap1: the check's verdict)
    unread: tuple[tuple[str, str], ...] = ()            # after an exit 0, the parameters no script reads, (file, key)
    span: str = ""                                      # after an exit 0, the span finding's line and the scale
    scale: str = ""                                     # finding's (lane convcard1: `Workspace.against`), "" where none
    shimmed: tuple[str, ...] = ()                       # the shim's kinds that fired in it (lane clock1 of 10-13)
    called: tuple[library.Rewrite, ...] = ()            # lane called1: the carded functions its files compute by hand
    rules: tuple["Finding", ...] = ()                   # after an exit 0, the rules' findings among `findings` (lane
    #                                                     rules1: `Workspace.ruled`)
    failed_in: library.Function | None = None          # lane cardfail1: the library function it failed in, read
    #                                                     where `Workspace.cardfail` (`failed_in`)
    snapshot: snapshot.Shown = snapshot.Shown()         # lane snapshot1: the state at the failure as its answer said
    #                                                     it, or why not where one was due (`Workspace.snapshot`)
    witness: tuple = ()                                 # lane derive1: the harness's own checks' findings its
    #                                                     observation said (agent/n11/witness.py `Found`s)
    contract: tuple = ()                                # lane contract1: the contract's findings its observation said
    #                                                     (agent/n11/contract.py `Found`s)
    recon: tuple = ()                                   # lane recon1: the reconciliation's findings its observation
    reconciled: bool = False                            # said (agent/n11/recon.py `Found`s), and whether it was read

    @property
    def clean(self) -> bool:
        return self.exit == 0 and not self.error and not self.failures

    @property
    def identity(self) -> tuple | None:
        """What a repeat of this failure is known by: the exception's type and its message's first line (lane n11d:
        the type alone, or with its line, matched two different IndexErrors); None where it raised nothing."""
        if not self.error:
            return None
        kind, _, message = self.error.partition(":")
        return kind.strip(), ADDRESS.sub("0x?", message.strip().split("\n", 1)[0][:300])


@dataclass(frozen=True)
class Finding:
    """Lane rules1: one rule's finding as `Workspace.sanity` said it: the rule (`RULES`), what it is known by across
    runs (the rule, the file and the name), its line as said, and the instruction's property where a rule reads one."""
    rule: str
    ident: str
    line: str
    prop: props.Prop | None = None


class Workspace:
    """The House's files (`root/work/`), the unit's output and inputs, and the commands over them."""
    kind = "script"                 # its runs' kind on the trace (lane repairpair1's rewrite runs under its own)

    def __init__(self, root: Path, out: Path, task, pathmap, library, named: tuple[str, ...],
                 python: str = sys.executable, stated: props.Stated | None = None, shimmed: bool = False,
                 linted: bool = False, rules: bool = False, wide: bool = False, cardfail: bool = False,
                 lint2: bool = False, snapshotted: bool = False, witnessed: bool = False,
                 contracted: props.Terms | None = None, reconciled: bool = False) -> None:
        self.root, self.out, self.task, self.pathmap, self.library = Path(root), Path(out), task, pathmap, library
        self.named = tuple(named)
        self.dir = self.root / WORK
        self.dir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.canaries = tuple(task.canaries) if task is not None else ()
        self.stated = stated            # what the instruction states of its outputs (agent/n11/props.py)
        self.current = None
        self.lock = threading.Lock()
        self.runs = 0
        self.params: list[tuple[str, list[str]]] | None = None     # the parameters files' keys, read once
        self.dated = None               # the input tables' dates (agent/n11/conventions.py `dated`), read once
        self.shimmed, self.linted = shimmed, linted     # lane clock1 of 10-13: the run's shim (agent/n11/shim.py)
        self.free: dict[str, list[str]] = {}            # and the write's check (agent/n11/lint.py); the standard
        self.lint_found: Counter = Counter()            # names the last run's files use unbound; the check's findings
        self.lint_writes = 0                            # by kind, and the writes it flagged
        self.known: lint.Known | None = None            # the library's names and the inputs' columns, read once
        self.columns: dict[str, list[str]] | None = None
        self.spliced: Counter = Counter()               # lane fixdef1: the `defs` writes by what each did
        self.carded: dict[str, library.Function] | None = None     # lane called1: the functions whose card reached
        self.methods: frozenset[str] | None = None                  # the House, by name (None: the switch off), and
        #                                                             the library's methods, read once (`called`)
        self.rules = rules                              # lane rules1: the rules read (`sanity`), and their findings
        self.ruled: tuple[Finding, ...] = ()            # as the last `sanity` said them
        self.wide = wide                                # lane rules2: the six rules after the eight (`ruling`'s)
        self.cardfail = cardfail                        # lane cardfail1: a failed run's library function read
        self.lint2 = lint2                              # lane lint2: the write's two narrow findings (lint.check2)
        self.lint2_found: Counter = Counter()           # and those said, by kind
        self.snapshotted = snapshotted                  # lane snapshot1: the state at a failure, read in the run
        self.witnessed = witnessed                      # lane derive1: the harness's own checks after a clean run,
        self.witness_task = None                        # the task they read built once (`witnessed_now`)
        self.contracted = contracted                    # lane contract1: the instruction's output contract, read
        #                                                 against each clean run's output where armed (None: not)
        self.reconciled = reconciled                    # lane recon1: the reconciliation after each clean run, on
        #                                                 the task witness's checks read (`witness_task`)
        self.output_env = 0                             # lane outguard1: the last run's OUTPUT_DIR named the output

    def redact(self, text: str) -> str:
        return canary.redact(text, self.canaries)

    # ---------------------------------------------------------------------------------------------- paths

    def local(self, path: str) -> tuple[Path | None, str]:
        """A path the House names under work/ (relative, bare, by this machine's path, or in an absolute form
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
        """An input file the House names: as the instruction names it (the path map), under the unit's data
        directory, or by its name there."""
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

    def write(self, path: str, text: str, tail: str = "", defs: bool = False, adding: bool = False) -> str:
        """The `write` observation, `tail` (the budget line) after it: the file saved whole at any length (the reply's
        cut is the limit), a path outside work/ or a name shadowing a standard-library module refused, each refusal
        naming the form accepted. Lane fixdef1, `defs` (the House's own writes under `AGENTHON_N11_WRITE=defs`): a
        `.py` file that exists takes a definitions text by name (`splice`), said by name and compiled before it is
        saved, a splice that does not compile refused and the file left as it was; a text that looked partial and
        is saved whole said so. Each `defs` write of a `.py` file counted (`spliced`) and on the trace. Lane
        joinask1, `adding`: the splice's (`splice`)."""
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
        made = splice(here, text, adding) if defs and named.endswith(".py") else None
        if made is not None and made.kind == "spliced":
            failed = compiled(made.text, named)
            if failed:
                self.splice_said(named, made, "refused")
                return fit(SPLICE_REFUSED.format(named=named, what=made.taken(), failed=failed), bound, tail)
            text = made.text
        try:
            here.parent.mkdir(parents=True, exist_ok=True)
            here.write_text(text, encoding="utf-8")
        except OSError as exc:                  # a directory where a file is named, or a file where a directory is
            return fit(f"write refused: {named} could not be written ({type(exc).__name__}): a directory or a file "
                       f"already stands on its path; nothing was written. The form: {WRITE_FORM}", bound, tail)
        spliced = made is not None and made.kind == "spliced"
        if made is not None:
            self.splice_said(named, made, made.kind)
        said = f"wrote {named}: " + (f"{made.said()}; kept the rest" if spliced else f"{size(text):,} B") + (
            f"; saved whole: {made.why}" if made is not None and made.why else "")
        if not named.endswith(".py"):
            return fit(said, bound, tail)
        failed = "" if spliced else compiled(text, named)
        if failed:
            return fit(f"{said}; {failed}", bound, tail)
        found = self.lint_lines(named, text) if self.linted else []
        found2 = self.lint2_lines(named, text) if self.lint2 else []
        if found or found2:
            return fit("\n".join([said + "; compiles", *found, *found2]),
                       bound + lint.FINDINGS * lint.LINE_CHARS * (int(self.linted) + int(self.lint2)), tail)
        return fit(said + "; compiles", bound, tail)

    def splice_said(self, named: str, made: Splice, saved: str) -> None:
        """Lane fixdef1: a `defs` write of a `.py` file counted by what it did (`SPLICE_KEYS`: spliced, spliced and
        adding a definition, refused for a splice that does not compile, or a text that looked partial saved whole) and
        its event on the trace: how it was saved (`spliced`, `refused`, `partial` or `whole`), the names it replaced
        (`spliced`) and those it added."""
        self.spliced["write_defs"] += saved == "spliced"
        self.spliced["write_defs_added"] += saved == "spliced" and bool(made.added)
        self.spliced["write_defs_failed"] += saved == "refused"
        self.spliced["write_partial_whole"] += saved == "partial"
        trace.event("write", path=named, saved=saved, spliced=list(made.replaced), added=list(made.added))

    def lint_lines(self, named: str, text: str) -> list[str]:
        """Lane clock1 of 10-13 (`AGENTHON_N11_LINT=on`, agent/n11/lint.py): a saved file's findings, at most
        `lint.FINDINGS`, each a line of the write's answer; the shim's standard names left out under the shim, which
        imports them. Counted by kind (`lint_found`) and the write flagged (`lint_writes`)."""
        if self.known is None:
            self.known = lint.known(self.library, QF_PACKAGE)
        if self.columns is None:
            self.columns = lint.inputs_columns(Path(self.pathmap.data_dir)) if self.pathmap is not None else {}
        try:
            found = lint.check(text, named, self.known, self.columns,
                               frozenset(shim.STANDARD) if self.shimmed else frozenset(),
                               self.beside(named))[:lint.FINDINGS]
        except Exception:  # noqa: BLE001 - the check never costs the write
            return []
        self.lint_found.update(finding.kind for finding in found)
        self.lint_writes += bool(found)
        return [self.redact(finding.line) for finding in found]

    def lint2_lines(self, named: str, text: str) -> list[str]:
        """Lane lint2 (`AGENTHON_N11_LINT2=on`, agent/n11/lint.py `check2`): a saved file's two narrow findings, an
        imported name shadowed and a configuration value never read, at most `lint.FINDINGS`, each a line of the write's
        answer after the lint's. Counted by kind (`lint2_found`)."""
        try:
            found = lint.check2(text, self.beside(named))[:lint.FINDINGS]
        except Exception:  # noqa: BLE001 - the check never costs the write
            return []
        self.lint2_found.update(finding.kind for finding in found)
        return [self.redact(finding.line) for finding in found]

    def beside(self, named: str) -> dict[str, str]:
        """The `.py` files under work/ beside `named`, each text by its module name (a `*` import's source, and the
        files that may import its names)."""
        here = self.local(named)[0]
        return {path.stem: path.read_text(encoding="utf-8", errors="replace")
                for path in self.dir.glob("*.py") if path != here}

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
        at most). Its own method, so a test's clock stands in for it."""
        self.runs += 1
        text = script.read_text(encoding="utf-8", errors="replace")
        trace.script(self.kind, self.runs, text, timeout)
        env = self.child_env()
        self.output_env = int(env[runner.OUTPUT_ENV] == str(self.out.absolute()))
        if self.shimmed:                        # lane clock1 of 10-13: the run's shim (agent/n11/shim.py)
            reached = [self.dir / f"{name}.py" for name in sorted(self.reached(script.stem)[0])]
            self.free = shim.free([script, *(path for path in reached if path.is_file())], self.root)
            env = shim.environ(self.root, env, run_table(self.pathmap, self.dir, self.out), APP, self.dir, self.free)
        if self.snapshotted:                    # lane snapshot1: the hook in the run's own process
            env = snapshot.environ(self.root, env, self.dir)
        exit_, seconds, out, err = runner.run_bounded([self.python, str(script)], timeout, cwd=self.dir,
                                                      env=env, hold=self.hold)
        stdout = out.value().decode("utf-8", errors="replace")
        stderr = err.value().decode("utf-8", errors="replace")
        with WRITING:
            if trace.active() and not getattr(self, "closed", False):
                streams = self.root / "streams"
                streams.mkdir(exist_ok=True)
                (streams / f"run_{self.runs}.out").write_text(stdout, encoding="utf-8")
                (streams / f"run_{self.runs}.err").write_text(stderr, encoding="utf-8")
                trace.ran(self.kind, self.runs, exit_, seconds, streams / f"run_{self.runs}.out",
                          streams / f"run_{self.runs}.err")
        return exit_, seconds, stdout, stderr

    def child_env(self) -> dict[str, str]:
        """A run's environment: `OUTPUT_DIR` naming the output unless the environment names one (common.runner)."""
        return runner.child_env(self.out)

    def run(self, path: str, timeout: float, seen: set | None = None, tail=lambda ran: "") -> tuple[str, Ran | None]:
        """The `run` observation (its checks, and `tail(run)` after them: the gate's line and the budget's, `tail(None)`
        for a run refused) and the run, None where refused. An exception `seen` before (its `identity`) is said
        once: the next time, by name."""
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
        fired = shim.fired(self.root) if self.shimmed else []
        snapped = snapshot.read(self.root) if self.snapshotted else None
        return self.observe(named, timeout, exit_, seconds, stdout, stderr, before, seen, tail,
                            fired=fired, snapped=snapped)

    def observe(self, named: str, timeout: float, exit_: int | None, seconds: float, stdout: str, stderr: str,
                before: dict, seen: set | None = None, tail=lambda ran: "",
                fired: list[tuple[str, str]] | None = None, snapped: dict | None = None) -> tuple[str, Ran]:
        """A run's observation from what it did (`run`'s, and the offline replay's over a recorded run,
        scripts/n11_offline.py): `before` the output's snapshot before it, the output as it stands after; `fired`,
        what the run's shim did (lane clock1 of 10-13, agent/n11/shim.py), said one line a kind after its head; lane
        snapshot1, where `snapshotted`, `snapped` the state its hook wrote at the failure (agent/n11/snapshot.py),
        said after the traceback's tail and the output's listing and before the checks, whole, even where the error is
        said in one line as an earlier run's, the bound raised by its bytes so that today's body is cut as today; lane
        derive1, where `witnessed`, after an exit 0 the harness's own checks' findings (agent/n11/witness.py) said
        after the span and scale lines and before the checks' verdict, the bound raised by their bytes likewise; lane
        contract1, where `contracted`, after an exit 0 the contract's findings (agent/n11/contract.py) said after
        derive1's paragraph, the bound raised by their bytes likewise; lane recon1, where `reconciled`, after a clean
        run (exit 0, no error, the checks passing) derive1's checks and the reconciliation's (agent/n11/recon.py) read,
        one finding an object, and where any stands the verdict line `recon.VERDICT` and the findings after it
        (`held_said`) in place of the checks' passing line, the observation's last block before its tail, the bound
        raised by their bytes likewise."""
        after = contract.snapshot(self.out)
        stdout, stderr = self.redact(stdout), self.redact(stderr)
        frames = self.frames(stderr)
        error = last_error(stderr) if runner.TRACEBACK in stderr or exit_ not in (0, None) else ""
        failures = tuple(self.checks())
        digest, findings = self.sanity() if exit_ == 0 else ("", [])
        span, scale = self.against() if exit_ == 0 else ("", "")
        called = self.called()          # lane called1: whatever the exit
        clean = exit_ == 0 and not error and not failures
        reading = self.reconciled and clean     # lane recon1: derive1's checks read beside the reconciliation's
        witnessed = self.witnessed_now() if (self.witnessed and exit_ == 0) or reading else ()
        owed = self.contracted_now() if self.contracted is not None and exit_ == 0 else ()
        reconciled = self.reconciled_now() if reading else ()
        if reading:
            from n11 import recon
            witnessed = recon.deduped(witnessed, reconciled)
        holding = bool(reading and (witnessed or reconciled))
        failed = failed_in(stderr, error, self.library, self.dir) if self.cardfail and error else None
        ran = Ran(named, exit_, seconds, error, tuple((file, line, name) for file, line, name, _ in frames), failures,
                  tuple(findings), digest, frames[-1][3] if frames and error else "",
                  tuple(sorted(name for name in after if before.get(name) != after[name])), stdout,
                  unread=self.unread() if exit_ == 0 else (), span=span, scale=scale,
                  shimmed=tuple(dict.fromkeys(kind for kind, _ in fired or ())), rules=self.ruled if exit_ == 0 else (), called=called,
                  failed_in=failed,
                  snapshot=snapshot.shown(snapped, exit_, error, stderr) if self.snapshotted else snapshot.Shown(),
                  witness=witnessed, contract=owed, recon=reconciled, reconciled=reading)
        head = (f"run {named}: stopped at its limit of {timeout:.0f} s" if exit_ is None else
                f"run {named}: exit {exit_} after {seconds:.1f} s")
        parts = [head, *(self.redact(line) for line in shim.said(fired or [], self.free))]
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
        said = self.redact(ran.snapshot.text) if ran.snapshot.text else ""
        checked = self.redact(witness_said(witnessed)) if witnessed and not holding else ""
        held = self.redact(held_said(witnessed, reconciled)) if holding else ""
        contracted = self.redact(stated_contract.said(list(owed))) if owed else ""
        kept = [part for part in (said, f"outputs:\n{digest}" if digest else "", sanity.said(beside(findings, called, self.ruled if exit_ == 0 else ())),
                                  self.redact(unread_said(ran.unread)), span, scale, checked, contracted,
                                  held or verdict(failures), tail(ran)) if part]
        return fit("\n".join(parts), BOUNDS["run"] + (size(said) + 1 if said else 0)
                   + (size(checked) + 1 if checked else 0) + (size(contracted) + 1 if contracted else 0)
                   + (size(held) - size(verdict(failures)) if held else 0), "\n".join(kept)), ran

    def witnessed_now(self) -> tuple:
        """Lane derive1: the harness's own checks (agent/n11/witness.py, `witness.KEPT`) over the output as it stands,
        against the instruction and the input files (read once a unit); () where none finds anything, and where they
        cannot be read (never raised into the run)."""
        from n11 import witness     # it imports conventions.py, which imports this module
        try:
            if self.witness_task is None:
                instruction = self.task.instruction if self.task is not None else ""
                self.witness_task = witness.task(instruction, Path(self.pathmap.data_dir))
            return tuple(witness.run(self.witness_task, self.out)[:witness.SAID])
        except Exception:  # noqa: BLE001 - the checks never cost the run its observation
            return ()

    def reconciliation(self) -> tuple[tuple, tuple]:
        """Lane recon1: what stands over the output as it stands, as a clean run's observation would say it: derive1's
        checks' findings (`witnessed_now`, less those a reconciliation check names the object of) and the
        reconciliation's (`reconciled_now`)."""
        from n11 import recon
        reconciled = self.reconciled_now()
        return recon.deduped(self.witnessed_now(), reconciled), reconciled

    def reconciled_now(self) -> tuple:
        """Lane recon1: the reconciliation's findings (agent/n11/recon.py, `recon.KEPT`) over the output as it stands,
        against the instruction's words and the input files (the task derive1's checks read, built once a unit); ()
        where none finds anything, and where they cannot be read (never raised into the run)."""
        from n11 import recon, witness     # witness imports conventions.py, which imports this module
        try:
            if self.witness_task is None:
                instruction = self.task.instruction if self.task is not None else ""
                self.witness_task = witness.task(instruction, Path(self.pathmap.data_dir))
            return tuple(recon.run(self.witness_task, self.out)[:recon.SAID])
        except Exception:  # noqa: BLE001 - the reconciliation never costs the run its observation
            return ()

    def contracted_now(self) -> tuple:
        """Lane contract1: the contract's findings (agent/n11/contract.py, `contract.KEPT`) over the output as it
        stands, against the instruction's terms; () where none finds anything, and where they cannot be read (never
        raised into the run)."""
        try:
            return tuple(stated_contract.run(self.contracted, self.out)[:stated_contract.SAID])
        except Exception:  # noqa: BLE001 - the contract never costs the run its observation
            return ()

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
        lines = [failure.text for failure in contract.check(spec, self.out, (), self.canaries)]
        lines += [f"the output reports a failure: {flag}" for flag in selfreport.flags(self.out, errors=False)]
        return [self.redact(line) for line in lines]

    def sanity(self) -> tuple[str, list[str]]:
        """The output's digest and the findings over it as it stands: the instruction's own properties it breaks
        (`instruction:`) and the keys and columns it names that the files lack (`contract:`), agent/n11/props.py;
        then the sanity checks' (agent/n11/sanity.py). A reader's fault is never raised into the run."""
        ruled = sanity.Ruled() if self.rules else None
        digest_, findings = sanity.read(self.out, self.named, rules=ruled) if self.rules else sanity.read(self.out, self.named)
        if self.rules:
            return self.ruling(digest_, findings, ruled)
        if self.stated is not None:
            try:
                said, missing = props.check(self.stated, self.out)
            except Exception:  # noqa: BLE001 - the instruction's checks never cost the run's observation
                said, missing = [], []
            findings = [f"instruction: {line}" for line in said] + [f"contract: {line}" for line in missing] + findings
        return self.redact(digest_), [self.redact(line) for line in findings]

    def ruling(self, digest_: str, values: list[str], ruled: sanity.Ruled) -> tuple[str, list[str]]:
        """The instruction's, contract's and value lines, followed by the rules' findings (lane rules2's six after the
        eight's, under `wide`); the called lines join them before the values in `beside`. Each rule is kept in
        `ruled`."""
        found: list[Finding] = []
        said, missing = [], []
        outputs = None
        if self.stated is not None:
            try:
                outputs = props.Outputs(self.out, row_counts=self.wide)
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
        if self.wide:
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
        """Lane rules1's `literal`: each script under work/ whose literal numbers hold `LITERALS` or more of the
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
        """Lane convcard1: the span finding's line and the scale finding's over the output as it stands, against the
        instruction and the input tables' dates (agent/n11/conventions.py), each "" where it does not meet; ("", "")
        where they cannot be read, never raised into the run."""
        from n11 import conventions     # it imports this module, through state.py and inputs.py among others

        instruction = self.task.instruction if self.task is not None else ""
        try:
            if self.dated is None:
                self.dated = conventions.dated(Path(self.pathmap.data_dir))
            return (self.redact(conventions.span(instruction, self.out, self.dated)),
                    self.redact(conventions.scale(instruction, self.out)))
        except Exception:  # noqa: BLE001 - the findings never cost the run its observation
            return "", ""

    def parameters(self) -> list[tuple[str, list[str]]]:
        """Lane tools1: the unit's parameters files and their keys (`parameter_files`); read once."""
        if self.params is None:
            self.params = parameter_files(Path(self.pathmap.data_dir))
        return self.params

    def unread(self) -> tuple[tuple[str, str], ...]:
        """Lane tools1: the parameters files' keys (`parameters`) whose names appear nowhere in the House's scripts
        under work/, a whole word of any `.py` file's text, as (file, key) in the files' order, at most `PARAM_NAMES`;
        () where none, and where the files cannot be read (never raised into the run)."""
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

    def called(self) -> tuple[library.Rewrite, ...]:
        """Lane called1 (`AGENTHON_N11_CALLED=on`): each function whose card reached the House (`carded`) that a
        `.py` file under work/ computes by hand, by `library.rewritten`'s rule (`library.rewrites`, the library's
        methods but its conventions shelf `known`, as agent/n11/solve.py's `lib_rewritten` reads them), the first
        match for each def or routine name; () where the switch is off, nothing is carded or found, and where the files
        cannot be read (never raised into the run)."""
        if not self.carded:
            return ()
        try:
            if self.methods is None:
                self.methods = frozenset(function.name for function in self.library.functions
                                         if function.shelf not in library.UNPOINTED)
            files = [(f"{WORK}/{path.relative_to(self.dir).as_posix()}",
                      path.read_text(encoding="utf-8", errors="replace"))
                     for path in sorted(self.dir.rglob("*.py")) if "__pycache__" not in path.parts]
            found = library.rewrites(list(self.carded.values()), files, self.methods)
        except Exception:  # noqa: BLE001 - the finding never costs the run its observation
            return ()
        first: dict[tuple[str, str], library.Rewrite] = {}
        for rewrite in found:
            first.setdefault((rewrite.by, rewrite.name), rewrite)
        return tuple(first.values())

    def reached(self, start: str = "main") -> tuple[set[str], list[str]]:
        """Lane n11e: the modules under work/ a file there (`start`, a module's name) imports, directly or through
        others there (their names), and the modules those files import that are nowhere: neither under work/, nor the
        standard library's, nor installed, nor among the unit's input files (the files still to write)."""
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
        installed, nor among the unit's input files: the files still to write."""
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
        """Lane clock1: `missing` (the modules still to write, `reached`'s or `missing_modules`') less work/'s own name
        where a file under work/ imports its modules package-style, and the note saying so (`packaged_said`, `first`'s
        import said first where it has one); `missing` as it is and "" where none does."""
        found = packaged(self.dir) if WORK in missing else []
        if not found:
            return missing, ""
        return [name for name in missing if name != WORK], packaged_said(found, first)

    def provided(self) -> set[str]:
        """The modules the unit's input files hold (a helper `.py` the instruction names beside its data, imported
        from the data directory): never a file still to write."""
        data = Path(self.pathmap.data_dir)
        return {path.stem for path in data.rglob("*.py")} if data.is_dir() else set()

    def missing(self) -> list[str]:
        """The named output files not yet in the output directory."""
        return [name for name in self.named if not (self.out / name).is_file()]

    # -------------------------------------------------------------------------------------------- inspect

    def inspect(self, path: str, rows=None, tail: str = "") -> str:
        """The `inspect` observation, `tail` (the budget line) after it: a file the output directory holds named by its
        path (`output`), else an input file (`data`), else a file the runs or the House made (`made`, lane tools1),
        described alike within the same bounds; a directory listed."""
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
        """Lane tools1: a file or directory under the output directory named by its path (the instruction's
        `/app/output/<name>` or `/output/<name>`, this machine's, or `output/<name>`) or that directory itself, as (it,
        its name said back); (None, "") where none is. `inspect` reads it before an input, whose lookup by a name alone
        would answer an output's path with the input of that name (the review's)."""
        named = str(path).strip().strip("`'\"")
        clean = posixpath.normpath(named.removeprefix("./")) if named else ""
        for root in (str(self.out), *paths.OUTPUTS, "output"):
            rest = paths.under(clean, root)
            here = (self.out / rest if rest else self.out) if rest is not None else None
            if here is not None and inside(here, self.out) and here.exists():
                return here, f"{self.out}/{rest}" if rest else str(self.out)
        return None, ""

    def made(self, path: str) -> tuple[Path | None, str]:
        """Lane tools1: a file or directory the runs or the House made, as (it, its name said back): under the output
        directory by its path (`output`); under work/ by `local`'s forms, or work/ itself; else a file of the output
        directory by its name alone, where one file alone has it. (None, "") where none is."""
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
        The loop's first turn carries it, so the House need not spend a request on `inspect` before it writes."""
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
        """Lane first1 (`AGENTHON_N11_INSPECT=described`): `inspect`'s observation of an input file the loop's first
        message describes (`parts`, `carried`), from that description: `inspect`'s head, the description as the first
        message carried it, then the rows `inspect` shows past it (`describe`'s first and last), within the same bound;
        None where `path` names no such file (an output's, a file under work/, an input past the description's bound,
        none), which `inspect` answers as before."""
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

    # ------------------------------------------------------------------------------------------------- lib

    def lib(self, name: str | None = None, shelf: str | None = None, tail: str = "") -> str:
        """The `lib` observation, `tail` (the budget line) after it."""
        bound = BOUNDS["lib"]
        room = bound - (size(tail) + 1 if tail else 0)
        if name and str(name).strip():
            return fit(self.library.by_name(str(name), room), bound, tail)
        if shelf and str(shelf).strip():
            return fit(self.library.by_shelf(str(shelf), room), bound, tail)
        return fit("lib: name a function (`name`) or a module (`shelf`)", bound, tail)


NOT_DESCRIBED = ("\n- not described here (", "\n\nnot described here: ")    # the two descriptions' notes of the files
#                                                                     past their bound (`described`, inputs.described)


def carried(text: str, data_dir: Path) -> dict[str, str]:
    """Lane first1: each input file the loop's first message describes (`text`, its `# Input files, described`), by
    its name under `data_dir`, with what the text says of it after its head (`portraits`); a file named only in the
    note of the files past the bound, or whose portrait was cut to its bound, is not fully described."""
    found = {}
    for name, _, start, end, _ in portraits(text, data_dir):
        part = text[start:end]
        if not any(marker in part for marker in ("[... cut to the bound]", "[... cut]")):
            found[name] = part
    return found


def portraits(text: str, data_dir: Path) -> list[tuple[str, int, int, int, str]]:
    """Each input file under `data_dir` that `text` describes (`Workspace.described`'s `- <name> (<size> B): `
    entries, or agent/n11/inputs.py's `<name> (<size> B...): ` portraits under `rich`), in the text's order: its name,
    where its head starts, where what the text says of it starts and ends (the next file's head, the separator between
    them left out, or the note naming the files past the bound), and that separator."""
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
    """Lane tools1: each parameters file among the inputs (its name holding a `PARAM_WORDS` word, of a kind in
    `PARAM_KINDS`) with the top-level keys a reader names (`top_keys`), in the data directory's order; the
    unread-parameter finding's and the conventions card's (agent/n11/conventions.py `parameters`)."""
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
    """Lane tools1: a parameters file's top-level keys in its order (a JSON or TOML object's; a YAML or key-value
    text's lines at the first column, `KEY_LINE`), but a key opening with `_` (a comment's, a watermark's) and one
    whose value is prose (`prose`); [] where it reads as none."""
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
    """Lane rules1: the unread parameters (`Workspace.unread`) as the finish's finding, a broken rule naming each key
    and its file, never a count or a value; "" where none."""
    if not found:
        return ""
    files = list(dict.fromkeys(name for name, _ in found))
    names = [f"`{key}`" + (f" in {name}" if len(files) > 1 else "") for name, key in found]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"the task's figures depend on {listed}" + (f" in {files[0]}" if len(files) == 1 else "") + \
        ", which no script reads"


def unread_said(found) -> str:
    """Lane tools1: the unread-parameter finding, one line naming each key and its file, never a count or a value; ""
    where none."""
    if not found:
        return ""
    files = list(dict.fromkeys(name for name, _ in found))
    names = [f"`{key}`" + (f" in {name}" if len(files) > 1 else "") for name, key in found]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"the parameter{'s' if len(names) > 1 else ''} {listed}" + (f" in {files[0]}" if len(files) == 1 else "")
            + f" {'are' if len(names) > 1 else 'is'} not read by any script")


def called_said(found) -> list[str]:
    """Lane called1: the called finding, a line for each function a file computes by hand (`Workspace.called`): the def
    and its file, or the file and the routine, and the function's call, as a broken rule and its fix
    (`CALLED_SAID`), never a count; a long name cut so the line keeps its fix within `sanity.FINDING_CHARS`."""
    lines = []
    for rewrite in found:
        said = CALLED_SAID[rewrite.by]
        room = sanity.FINDING_CHARS - len(said.format(name="", file=rewrite.file, call=rewrite.function.call))
        name = rewrite.name if len(rewrite.name) <= room else rewrite.name[:max(room - 3, 1)] + "..."
        lines.append(said.format(name=name, file=rewrite.file, call=rewrite.function.call)[:sanity.FINDING_CHARS])
    return lines


def beside(findings, found, rules=()) -> list[str]:
    """Lane called1: a run's or a finish's findings with the called finding's lines (`called_said` of `found`) after
    the instruction's and the contract's and before the sanity checks' own; `findings` as they are where none."""
    if not found:
        return list(findings)
    ruled = {each.line for each in rules}
    stated = [line.startswith(("instruction:", "contract:")) and line not in ruled for line in findings]
    return [line for line, first in zip(findings, stated) if first] + called_said(found) + \
        [line for line, first in zip(findings, stated) if not first]


def verdict(failures) -> str:
    """The checks' lines after a run: `checks: all pass`, or each failing line, at most 12 and the rest counted."""
    return ("checks: all pass" if not failures else
            "checks failing:\n" + "\n".join(f"- {line}" for line in failures[:12]) +
            (f"\n- and {len(failures) - 12} more" if len(failures) > 12 else ""))


def unfenced(text: str) -> str:
    """A file's text with a fence the House wrapped it in removed: a first line opening a block and a last closing
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
    """Lane clock1: the imports in the files under work/ (`directory`) that name its modules as a package (`from
    work.utils import f`, `import work.utils`, `from work import utils`), each (the file, the import as written, the
    import by the bare name), in the files' order; [] where none does."""
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
        with warnings.catch_warnings():            # the House's invalid escapes warn on stderr, which is the log's
            warnings.simplefilter("ignore")
            compile(text, named, "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"does not compile: {type(exc).__name__} at line {exc.lineno}: {exc.msg}\n" + around(text, exc.lineno)
    except (ValueError, MemoryError, RecursionError) as exc:
        return f"does not compile ({type(exc).__name__})"
    return ""


# ------------------------------------------------------------------------------- lane fixdef1: a fix by definition

DEFINED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)    # a definition taken by its name (decorated or not)
IMPORTS = (ast.Import, ast.ImportFrom)
SPLICE_KEYS = ("write_defs", "write_defs_added", "write_defs_failed", "write_partial_whole")
SPLICE_REFUSED = "write refused: {named} with {what} {failed}\nnothing was written: {named} stands as it was"
PARTIAL = {     # why a text holding only definitions, for a .py file that exists, was saved whole
    "absent": "none of its definitions is in the file",
    "nameless": "it holds no function or class",
    "unparsed": "the file as it stood does not compile, so none of its definitions could be found in it"}


@dataclass(frozen=True)
class Splice:
    """What a `defs` write of a `.py` file does (`splice`): `spliced` (`text` the file with the text's definitions in
    it, the names `replaced` and the definitions `added`), `partial` (a text holding only definitions the file could
    not take by name, saved whole and said, `why`) or `whole` (any other text, saved whole as under `whole`)."""
    kind: str
    text: str = ""
    replaced: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    why: str = ""

    def said(self) -> str:
        """`replaced `a` and `b`, added `c``: the write's words for it."""
        return ", ".join(part for part in (f"replaced {listed(self.replaced)}" if self.replaced else "",
                                           f"added {listed(self.added)}" if self.added else "") if part)

    def taken(self) -> str:
        """`a` and `b` replaced and `c` added: a refused splice's words for it."""
        return f"{listed(self.replaced)} replaced" + (f" and {listed(self.added)} added" if self.added else "")


def listed(names) -> str:
    """Names in backquotes: `a`, `a` and `b`, `a`, `b` and `c`."""
    quoted = [f"`{name}`" for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def assigned(node: ast.stmt) -> tuple[str, ...]:
    """The names a plain assignment binds (`x = ...`, `x = y = ...`, `x: float = ...`); () for any other statement."""
    if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
        return tuple(target.id for target in node.targets)
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        return (node.target.id,)
    return ()


def span(node: ast.stmt) -> tuple[int, int]:
    """A top-level statement's lines, from 1, its decorators included."""
    return min([node.lineno, *(each.lineno for each in getattr(node, "decorator_list", ()))]), node.end_lineno


def main_guard(node: ast.stmt) -> bool:
    """Whether a top-level statement is an `if __name__ == "__main__":` block."""
    test = node.test if isinstance(node, ast.If) else None
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


GUARD_NAME = 'if __name__ == "__main__"'      # how a write names the main block it replaced or added


def splice(here: Path, text: str, adding: bool = False) -> Splice:
    """Lane fixdef1 (`AGENTHON_N11_WRITE=defs`): the file `here` (a `.py` file under work/) as a write of `text` leaves
    it. `text` is a definitions block where the file exists and parses, every top-level statement of `text` is an
    import, a function or class definition or a plain assignment to names (so no `if __name__` block and no other
    statement), and one of its functions or classes is a top-level function or class of the file. Then each such
    definition of the file is replaced in place by the text's, by the nodes' line spans, decorators included (a name
    the text defines twice taken at its last); a plain assignment replaces the file's top-level one to the same names
    where one stands on lines of its own; a function, class or assignment the file lacks goes after the last function
    or class replaced, and an import the file's top-level imports lack after the last of them (a `__future__` import
    and, where the file has none, every import, at its head, after its docstring). A text holding only definitions
    that the file cannot take so is `partial`; any other, and any text for a file that does not exist, `whole`.
    Lane joinask1, `adding` (the write that answers agent/n11/solve.py's `JOIN_RAN` or `JOIN_DEFS`): the text may
    hold no definition the file has, and an `if __name__ == "__main__":` block, which replaces
    the file's last one or goes at its end; where it replaces no definition, what it adds goes before the file's main
    block, else at its end."""
    if not here.is_file():
        return Splice("whole")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return Splice("whole")
    if not tree.body or not all(isinstance(node, (*DEFINED, *IMPORTS)) or assigned(node)
                                or (adding and main_guard(node)) for node in tree.body):
        return Splice("whole")
    try:
        old = here.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        base = ast.parse(old)
    except (OSError, SyntaxError, ValueError, MemoryError, RecursionError):
        return Splice("partial", why=PARTIAL["unparsed"])
    held = {node.name: node for node in base.body if isinstance(node, DEFINED)}
    named = [node.name for node in tree.body if isinstance(node, DEFINED)]
    if not adding and (not named or not held.keys() & set(named)):
        return Splice("partial", why=PARTIAL["absent" if named else "nameless"])
    guard = next((node for node in reversed(tree.body) if main_guard(node)), None)
    taken: dict[tuple, ast.stmt] = {}               # the text's definitions by name, each at its last
    for node in tree.body:
        if not isinstance(node, IMPORTS) and not main_guard(node):
            key = ("def", node.name) if isinstance(node, DEFINED) else ("set", assigned(node))
            taken.pop(key, None)
            taken[key] = node
    lines = text.split("\n")
    spans = {id(node): range(span(node)[0], span(node)[1] + 1) for node in base.body}
    owners = Counter(line for node in base.body for line in spans[id(node)])     # `a = 1; b = 2` shares its line
    sets = {assigned(node): node for node in base.body
            if assigned(node) and all(owners[line] == 1 for line in spans[id(node)])}

    def segment(node: ast.stmt) -> str:
        start, end = span(node)
        return "\n".join(lines[start - 1:end]) if isinstance(node, DEFINED) else ast.get_source_segment(text, node)

    swap: dict[int, tuple[int, str]] = {}           # the file's lines replaced: from its first, (its last, the text)
    after: dict[int, list[str]] = {}                # the lines put after a line of the file (0: before its first)
    replaced, added, new, last = [], [], [], 0
    for (kind, key), node in taken.items():
        found = held.get(key) if kind == "def" else sets.get(key)
        if found is None:
            new += ["", "", *segment(node).split("\n")]
            added += [key] if kind == "def" else list(key)
            continue
        start, end = span(found)
        swap[start] = (end, segment(node))
        replaced += [key] if kind == "def" else list(key)
        last = max(last, end) if kind == "def" else last
    old_lines = old.split("\n")
    stands = next((node for node in reversed(base.body) if main_guard(node)), None)
    if adding and not last:         # nothing replaced: what is added goes before the file's main block, or at its end
        last = span(stands)[0] - 1 if stands is not None else len(old_lines)
    after[last] = new
    if guard is not None and stands is not None:
        start, end = span(stands)
        swap[start] = (end, segment(guard))
        replaced.append(GUARD_NAME)
    elif guard is not None:
        after.setdefault(len(old_lines), []).extend(["", "", *segment(guard).split("\n")])
        added.append(GUARD_NAME)
    head = base.body[0].end_lineno if isinstance(base.body[0], ast.Expr) and isinstance(
        base.body[0].value, ast.Constant) and isinstance(base.body[0].value.value, str) else 0
    imports = [node for node in base.body if isinstance(node, IMPORTS)]
    stand = {ast.get_source_segment(old, node).strip() for node in imports}
    for node in (node for node in tree.body if isinstance(node, IMPORTS)):
        said = ast.get_source_segment(text, node)
        if said.strip() in stand:
            continue
        stand.add(said.strip())
        future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        at = imports[-1].end_lineno if imports and not future else head
        after.setdefault(at, []).extend(said.split("\n"))
        added.append(said)
    out, k = list(after.get(0, [])), 1
    while k <= len(old_lines):
        end, kept = swap.get(k, (k, old_lines[k - 1]))
        out += [*kept.split("\n"), *after.get(end, [])]
        k = end + 1
    return Splice("spliced", "\n".join(out), tuple(replaced), tuple(added))


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


def failed_in(stderr: str, error: str, lib: library.Library, work: Path) -> library.Function | None:
    """Lane cardfail1: the library function a failed run's last traceback ends in, else None. Its frames from the most
    recent outward: a frame of the package's module of a shelf (`library.PACKAGE`, `<shelf>.py`) in a function of that
    shelf is it; a frame of the package in none (a helper, the guard's wrapper), or outside both the package and `work`
    (numpy, scipy, the standard library), is passed; a frame under `work` ends the walk, the error the script's own.
    Else a library function `error` names as `qf.<name>` (the guard's refusals); else one a TypeError refusing a
    call's arguments names first (`<name>() got ...`), where the last frame is the package's or its source calls that
    name under `qf.`, or bare with an unshadowed import from `qf`: `np.std()` and a model-defined `std()` never
    meet a library's `std`."""
    if runner.TRACEBACK not in stderr:
        return None
    text = stderr[stderr.rfind(runner.TRACEBACK):]
    frames = list(FRAME.finditer(text))
    for match in reversed(frames):
        file = Path(match.group("file"))
        if file.parent.name == library.PACKAGE and file.suffix == ".py":
            shelf = lib.shelf(file.stem) if not file.stem.startswith("_") else None
            found = next((function for function in shelf.functions if function.name == match.group("name").strip()),
                         None) if shelf is not None else None
            if found is not None:
                return found
        elif within(str(file), work):
            break
    for name in QF_NAMED.findall(error):
        if (found := lib.named(name)) is not None:
            return found
    refused = CALL_REFUSED.match(error)
    found = lib.named(refused.group(1)) if refused else None
    if found is None or not frames:
        return None
    last = frames[-1]
    if Path(last.group("file")).parent.name == library.PACKAGE:
        return found
    source = "\n".join(itertools.takewhile(lambda line: line[:1].isspace(), text[last.end():].split("\n")[1:]))
    called = re.compile(rf"(?<![\w.])qf\.(?:[A-Za-z_]\w*\.)?{re.escape(found.name)}\s*\(")
    if called.search(source):
        return found
    bare = re.compile(rf"(?<![\w.]){re.escape(found.name)}\s*\(")
    file = Path(last.group("file"))
    file = file if file.is_absolute() else work / file
    return found if bare.search(source) and within(str(file), work) and qf_imported(file, found.name) else None


def qf_imported(file: Path, name: str) -> bool:
    """A bare refused call needs an explicit qf import and no other binding of its name in the script.
    Ambiguous bindings are left uncarded rather than attributing the model's own TypeError to the library."""
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return False
    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return False
                if (alias.asname or alias.name) == name:
                    if node.level or not (node.module == "qf" or (node.module or "").startswith("qf.")) or alias.name != name:
                        return False
                    imported = True
        elif isinstance(node, ast.Import):
            if any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names):
                return False
        elif ((isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == name)
              or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name)
              or (isinstance(node, ast.arg) and node.arg == name)
              or (isinstance(node, ast.ExceptHandler) and node.name == name)):
            return False
    return imported


def within(file: str, work: Path) -> bool:
    """Whether a traceback's `file` lies under `work`, resolved as `Workspace.frames` resolves it."""
    try:
        resolved = Path(file).resolve() if os.path.isabs(file) else (work / file).resolve()
        return str(resolved).startswith(str(Path(work).resolve()) + "/")
    except OSError:
        return False


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


def witness_said(found) -> str:
    """Lane derive1: the harness's own checks' paragraph (agent/n11/witness.py `said`)."""
    from n11 import witness
    return witness.said(list(found))


def held_said(witnessed, reconciled) -> str:
    """Lane recon1: a clean run's verdict while a finding stands, as the observation's last block: `recon.VERDICT` in
    place of the checks' passing line, then one paragraph under `recon.HEAD`, derive1's lines first and then the
    reconciliation's."""
    from n11 import recon, witness
    return f"{recon.VERDICT}\n" + recon.said([*list(witnessed)[:witness.SAID], *reconciled],
                                             limit=witness.SAID + recon.SAID)


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
