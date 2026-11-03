"""W1: width with an execution filter and an agreement vote. `python3 -m w1.solve solve --task-dir /input --out /app/output`.

The cold read's form 2 (runs/lanes/cold-day2.md; runs/lanes/w1-prompt.md). N0's one request is drawn `DRAFTS` times
at once, on threads, each under the next of the fixed system prompts (`SYSTEMS`: N0's own, N1's compact clause,
explicit intermediate checks; then helpers first and vectorised, for a form of five; the user prompt N0's, the same for
all) at `TEMPERATURE`, the seed pinned.
Each draft's script then runs in turn, in a work directory of its own, under a cap sized from the time left and the
drafts still to run (a script the output cap cut runs as its longest prefix that compiles, N1's salvage); it writes
into `--out` as the prompt says, and its files are moved after its run to a store of its own, so the next run starts
from an empty output. A draft is clean when its run exits 0 with no traceback and
leaves a file at the output's root, and named-complete when every output file the instruction names is among its
files (common.contract.named_all). The vote (`vote`): among the clean drafts, the named-complete ones where there are
any; two agree when they wrote the same of the files the instruction names (every file both wrote where they wrote
none of those), hold at least one number at the same place in them, and every number the two hold at the same place
agrees within a relative 1e-6 (JSON's numeric leaves by key path, a table's numeric cells by row and column once the
rows are sorted by their first column; a file where either holds a number and the two share none disagrees); the
largest group of drafts that all agree wins, a tie to the group holding the earliest preamble, and the winner (its
earliest preamble) has its files copied into `--out`: its own script wrote them at run time, in this run. Where no
draft is clean, the draft whose run went longest (else the first with a script, else the first that replied) is laid
into `--out` and repaired once with N0's repair prompt, and the repaired script runs in the time left over those
files. At most `DRAFTS` drafts and one repair, two rounds of latency; at most `REQUESTS` requests
counted, so a draft's retries have room. A 407 from the proxy is the client's to retry, uncounted (model.py). Any
error on a draft loses that draft, never the unit. The unit's deadline (model.unit_deadline) over everything; the
watchdog kills the running script, delivers the best clean draft so far where none is delivered yet, closes the
output and exits 0. Whatever happens, the output is closed (common.guard.close) and the exit is 0. Each draft's outcome
(its request, its run, clean or not, its files, its group in the vote) is one line of the step log after the vote, and
under the desk's trace one `draft` event and a copy of its files in the trace's `drafts/<i>/`, so a read can score
every draft offline (mean, oracle and selected success); none of it reaches `--out`.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from common import canary, contract, digest, guard, house, paths, runner, trace, unit
from common.house import model

AGENT = "w1"
DRAFTS = 3                  # drafts, all asked at once: the registered form's three (SYSTEMS holds five)
REQUESTS = 2 * (DRAFTS + 1) # the drafts and a repair, twice over: room for the client's retries (12 at five drafts)
TEMPERATURE = 0.6           # the drafts' sampling; the seed stays pinned
ROUND_SHARE = 0.5           # the round waits for its drafts at most this share of the time left
RUN_SHARE = 0.5             # the drafts' runs' share of the time left at their start while a repair can follow
RUN_RESERVE_S = 20.0        # kept from the deadline after the last run: the close, before the watchdog's 10 s
MIN_RUN_S = 5.0             # a run is not started with less time than this
REL_TOL = 1e-6              # two numbers agree within this share of the larger,
ABS_TOL = 1e-12             # or within this of each other near zero

LIBRARIES = ("numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, openpyxl, matplotlib (the Agg "
             "backend), plotly, seaborn, TA-Lib (imported as talib) and backtrader")

HEAD = """You are an expert quantitative developer. You solve a task by writing one complete Python 3 script, \
which is run once, unattended, with no network; hidden tests then check the files it wrote against the task.

Reply with the whole script in a single ```python fenced block."""

RULES = f"""The script must:
- read its inputs from the true paths given under "Paths": the paths in the task's text are those of another machine;
- write every output file the task asks for, with exactly the names, formats, keys and columns it gives, directly \
into the output directory given under "Paths";
- use only the Python standard library and these installed packages: {LIBRARIES}; no network, no pip, no subprocess;
- be deterministic: seed any random generator the task does not seed itself;
- write JSON with plain Python numbers (convert numpy values with float() or int());
- never copy the task's text into an output file, and never write a file named reward.json; `<canary>` in the task \
or the data stands for a watermark string: never write it, or an input field that holds it, into any output file;
- finish well within the time it is given;
- print nothing but one short final line saying it is done."""

# The working styles, one a draft and the drafts' only difference, in the order the drafts take them: N0's own (none),
# N1's compact clause (whose work at the top level is the top-level flow), explicit intermediate checks (the only one
# that changes what the script verifies, not how it is laid out), then helpers first and vectorised, for a form of
# five. Each is general: no unit, task, file or library beyond the installed list is named (tests/test_w1.py reads
# them for each).
COMPACT = ("Keep the script compact, since a reply is cut off at about 4,000 tokens: no comments, no docstrings, no "
           "type hints, no argparse, no logging beyond the one final line, no plot unless the task asks for one as an "
           "output file, no re-validation of inputs the task guarantees, and helper functions only where they shorten "
           "the script; the work at the top level, not inside a main() function, each output file written as soon as "
           "it is computed; the whole script under about 200 lines.")
HELPERS = ("Organise the script as small helper functions defined first, one for each step the task describes (reading "
           "the inputs, each calculation, writing each output), then a short sequence at the end that calls them in "
           "the task's order and writes each output file as soon as its values are ready. Keep it short: no comments "
           "or docstrings, since a reply is cut off at about 4,000 tokens.")
VECTORISED = ("Work on whole arrays and tables at once: vectorised numpy and pandas operations in place of Python loops "
              "over rows wherever the calculation allows, and a routine of the installed packages in place of a "
              "hand-written numerical method only where it computes exactly what the task defines. Keep it short: no "
              "comments or docstrings, since a reply is cut off at about 4,000 tokens.")
CHECKED = ("Check the intermediate results as the script goes: after reading the inputs, the shapes, columns and "
           "ranges the task implies; after each calculation, the identities and bounds the task states and that its "
           "values must satisfy (finite numbers, sums, signs, orderings); where a check would fail, correct the "
           "computation rather than the check. Then write each output file. Keep it short: no comments or "
           "docstrings, since a reply is cut off at about 4,000 tokens.")
STYLES = ("", COMPACT, CHECKED, HELPERS, VECTORISED)
PREAMBLES = ("n0", "compact", "checked", "helpers", "vectorised")     # each style's name, for the outcome lines
SYSTEMS = tuple(f"{HEAD}\n\n{style}\n\n{RULES}" if style else f"{HEAD}\n\n{RULES}" for style in STYLES)


def run_cap(left: float, remaining: int, kept: float) -> float:
    """A draft's run cap: the time left less what is kept for a repair, shared among the drafts still to run, never
    under `MIN_RUN_S` nor over the time left."""
    return min(left, max(MIN_RUN_S, (left - kept) / remaining))


def prompt(task: unit.Unit, pathmap: paths.PathMap, data: str, seconds: float) -> str:
    return (f"# Task\n\n{task.instruction.strip()}\n\n# Paths\n\n{pathmap.text()}\n\n# Input data\n\n{data.strip()}"
            f"\n\n# Time\n\nThe script is stopped after {seconds:.0f} seconds.\n\nWrite the complete script now.\n")


def listing(out: Path) -> str:
    try:
        files = sorted(path for path in out.rglob("*") if path.is_file() and not path.is_symlink())
    except OSError:
        files = []
    if not files:
        return "The output directory holds no file."
    return "Files now in the output directory: " + ", ".join(
        f"{path.relative_to(out)} ({path.stat().st_size:,} bytes)" for path in files[:40])


def lines(text: str) -> int:
    """A script's lines, as `runner.parsing_prefix` counts them."""
    return len(text.removesuffix("\n").split("\n")) if text else 0


def logged(text: str) -> str:
    """A message for the log: any line carrying a GUID-shaped token dropped."""
    return canary.strip(text)[0]


def files_in(root: Path) -> dict[str, Path]:
    """Every regular file under `root` by its path relative to it, links never followed, the scorer's own left out."""
    found = {}
    try:
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() and not path.is_symlink():
                name = path.relative_to(root).as_posix()
                if name not in guard.SCORER_FILES:
                    found[name] = path
    except OSError:
        pass
    return found


# ------------------------------------------------------------------------------------------------------------ the vote


def number(value: str) -> float | None:
    try:
        return float(value) if value.strip() else None
    except ValueError:
        return None


def json_leaves(value: object, where: tuple = (), every: bool = False) -> dict[tuple, object]:
    """Every number in a JSON value by its key path (an object's keys, a list's indices); booleans are no number.
    `every` (agent/n11/drafts.py): every leaf, the numbers as floats and any other (a text, a boolean, null, an empty
    object or list) as it is."""
    if isinstance(value, bool):
        return {where: value} if every else {}
    if isinstance(value, (int, float)):
        return {where: float(value)}
    if every and (not isinstance(value, (dict, list)) or not value):
        return {where: value}
    found: dict[tuple, object] = {}
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, item in items:
        found.update(json_leaves(item, (*where, key), every))
    return found


def table_leaves(text: str, delimiter: str, every: bool = False) -> dict[tuple, object]:
    """Every numeric cell of a table with a header row by (row, column name, the name's occurrence), the rows sorted
    by their first column (numbers by value, before text; a tie by the whole row), so two tables in another row order
    line up. `every` (agent/n11/drafts.py): every cell, a number as a float and any other as its text, and the header
    in its order under `("columns",)`."""
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        return {}
    header, body = rows[0], rows[1:]
    seen: dict[str, int] = {}
    columns = []
    for name in header:
        seen[name] = seen.get(name, 0) + 1
        columns.append((name, seen[name]))

    def order(row: list[str]) -> tuple:
        first = number(row[0]) if row else None
        if first is not None and not math.isnan(first):
            return 0, first, "", row
        return 1, 0.0, row[0] if row else "", row
    found: dict[tuple, object] = {("columns",): tuple(header)} if every else {}
    for index, row in enumerate(sorted(body, key=order)):
        for column, cell in enumerate(row):
            value = number(cell)
            if value is not None or every:
                found[(index, *(columns[column] if column < len(columns) else (column, 1)))] = \
                    cell if value is None else value
    return found


def leaves(path: Path, every: bool = False) -> dict[tuple, object] | None:
    """A file's numbers by where they are: a JSON (or JSON lines) file's numeric leaves, a CSV or TSV file's numeric
    cells; {} for any other kind, which holds none to compare; None where a JSON or table file does not parse.
    `every`: every leaf and cell (`json_leaves`, `table_leaves`)."""
    kind = contract.fmt_of(path.name)
    if kind not in ("json", "jsonl", "csv", "tsv"):
        return {}
    try:
        text = path.read_bytes().decode("utf-8")
        if kind == "json":
            return json_leaves(json.loads(text), every=every)
        if kind == "jsonl":
            return json_leaves([json.loads(line) for line in text.splitlines() if line.strip()], every=every)
        return table_leaves(text, "\t" if kind == "tsv" else ",", every)
    except (OSError, ValueError, csv.Error, RecursionError):
        return None


def near(a: float, b: float, rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL) -> bool:
    """Two numbers agree within the tolerances (the vote's by default); NaN agrees with NaN alone."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


class Ballot:
    """The drafts voted on, each its files by name, and the output files the instruction names; `agree` compares
    two, each file read and each pair compared once."""

    def __init__(self, outputs: dict[int, dict[str, Path]], named=()) -> None:
        self.outputs = outputs
        self.named = frozenset(named)
        self.read: dict[Path, dict[tuple, float] | None] = {}
        self.pairs: dict[tuple[int, int], bool] = {}

    def numbers(self, path: Path) -> dict[tuple, float] | None:
        if path not in self.read:
            self.read[path] = leaves(path)
        return self.read[path]

    def same(self, a: Path, b: Path) -> int | None:
        """One file of two drafts: how many numbers the two hold at the same place, every one agreeing; None where
        they disagree (a shared number apart, or numbers on either side and none shared). A file that does not parse
        agrees with the same bytes alone, and shares no number."""
        left, right = self.numbers(a), self.numbers(b)
        if left is None or right is None:
            try:
                return 0 if left is None and right is None and a.read_bytes() == b.read_bytes() else None
            except OSError:
                return None
        shared = left.keys() & right.keys()
        if (left or right) and not shared or not all(near(left[key], right[key]) for key in shared):
            return None
        return len(shared)

    def compared(self, i: int, j: int) -> list[str] | None:
        """The files two drafts are compared on: the named ones where either wrote one, which both must have written
        alike (a debug file beside them splits no group), else every file both wrote; None where they wrote different
        named files."""
        a, b = self.outputs[i].keys(), self.outputs[j].keys()
        mine, theirs = a & self.named, b & self.named
        if mine or theirs:
            return sorted(mine) if mine == theirs else None
        return sorted(a & b)

    def agree(self, i: int, j: int) -> bool:
        """Two drafts agree on every file compared and share at least one number: two that share none (other key
        names, or no number at all) never agree by default."""
        if (i, j) not in self.pairs:
            a, b, names = self.outputs[i], self.outputs[j], self.compared(i, j)
            shared = [self.same(a[name], b[name]) for name in names] if names is not None else [None]
            self.pairs[(i, j)] = None not in shared and sum(shared) > 0
        return self.pairs[(i, j)]


@dataclass(frozen=True)
class Verdict:
    winner: int | None                          # the preamble whose files ship; None where no draft is clean
    cluster: tuple[int, ...] = ()               # the drafts agreeing with it, itself included
    pool: tuple[int, ...] = ()                  # the drafts voted on
    clusters: tuple[tuple[int, ...], ...] = ()  # the pool in groups, the winner's first: the log's

    def sizes(self) -> str:
        return "+".join(str(len(group)) for group in self.clusters) or "none"


def groups(pool: list[int], agree) -> list[tuple[int, ...]]:
    """The pool in groups of drafts that all agree, largest first: each the largest such group left (every subset is
    tried: five drafts at most), a tie to the group holding the earliest preamble, then the earliest group in
    order."""
    left, found = sorted(pool), []
    while left:
        best = ()
        for size in range(len(left), 0, -1):
            for group in itertools.combinations(left, size):
                if all(agree(i, j) for i, j in itertools.combinations(group, 2)):
                    best = group
                    break
            if best:
                break
        found.append(best)
        left = [index for index in left if index not in best]
    return found


def vote(outputs: dict[int, dict[str, Path]], complete: set[int] | frozenset[int] = frozenset(), named=()) -> Verdict:
    """The clean drafts' files by preamble, those named-complete and the output files the instruction names: the
    winner (see the module's text)."""
    pool = sorted(index for index in outputs if index in complete) or sorted(outputs)
    if not pool:
        return Verdict(None)
    ballot = Ballot(outputs, named)
    found = groups(pool, ballot.agree)
    return Verdict(found[0][0], found[0], tuple(pool), tuple(found))


# ----------------------------------------------------------------------------------------------------------- the agent


@dataclass
class Draft:
    index: int                              # its preamble
    reply: model.Reply | None = None
    error: str = ""                         # why it has no reply
    code: runner.Code | None = None
    ran: runner.Run | None = None
    store: Path | None = None               # its files, moved there from --out after its run
    clean: bool = False
    complete: bool = False                  # clean, and every output file the instruction names is among its files
    kept: int | None = None                 # a cut script's lines its salvage kept; None where it was not cut
    done: bool = False                      # its request answered or failed
    abandoned: bool = False                 # still unanswered when the round ended: whatever comes later is dropped

    @property
    def cut(self) -> bool:
        return self.reply is not None and (self.reply.finish_reason == "length" or
                                           (self.code is not None and not self.code.closed))


class W1:
    def __init__(self, task_dir: Path, out: Path, client, steps: guard.Steps, work: Path, deadline: float) -> None:
        self.task_dir, self.out, self.client, self.log = Path(task_dir), Path(out), client, steps
        self.deadline, self.work = deadline, Path(work)
        self.unit: unit.Unit | None = None
        self.named: tuple[str, ...] = ()
        self.drafts: list[Draft] = []
        self.runner: runner.Runner | None = None      # the one running, for the watchdog's kill
        self.before: set[str] = set()                 # what --out held before any run: never moved
        self.delivered = False                        # --out holds what the unit ships (or the repair's base)
        self.ended = threading.Event()                # the watchdog has closed the output: nothing more is written
        self.gate = threading.Lock()
        self.reported = False                         # each draft's outcome written (once, by the vote or the watchdog)
        self.count = DRAFTS                           # the drafts a round sends (agent/n11/drafts.py's round sets its own)

    def left(self) -> float:
        return self.deadline - RUN_RESERVE_S - time.time()

    def redact(self, text: str) -> str:
        return canary.redact(text, self.unit.canaries if self.unit else ())

    def requests_left(self) -> int:
        return self.client.cap - self.client.usage.requests

    def runner_in(self, name: str) -> runner.Runner:
        """A runner whose work directory is `name` under the unit's, its own."""
        (self.work / name).mkdir(parents=True, exist_ok=True)
        return runner.Runner(self.work / name)

    def kill(self) -> None:
        if self.runner is not None:
            self.runner.kill()

    # ---------------------------------------------------------------------------------------------------- the round

    def ask(self, draft: Draft, user: str) -> None:
        """One draft's request, on its own thread; its outcome kept unless the round has already ended."""
        reply, error = None, ""
        try:
            client, keywords = self.sender(draft)
            reply = client.chat(self.system_of(draft), user, **{"temperature": TEMPERATURE, **keywords})
        except BaseException as exc:  # noqa: BLE001 - any error loses this draft alone
            error = type(exc).__name__ + (f": {logged(str(exc))[:200]}" if isinstance(exc, model.ModelError) else "")
        with self.gate:
            if draft.abandoned:
                return
            draft.reply, draft.error, draft.done = reply, error, True
            draft.code = runner.extract(reply.text) if reply is not None else None

    def system_of(self, draft: Draft) -> str:
        """A draft's system prompt: its preamble's (agent/n11/drafts.py's round says its own)."""
        return SYSTEMS[draft.index]

    def sender(self, draft: Draft) -> tuple[object, dict]:
        """The client a draft's request goes through and its keywords: the unit's own, none (agent/n11/drafts.py's
        round gives each draft a client of its own, bounded to the round, and its one draft the client's temperature)."""
        return self.client, {}

    def round(self, user: str) -> None:
        began, turned = time.time(), self.client.usage.turned_away
        self.drafts = [Draft(index) for index in range(self.count)]
        threads = [threading.Thread(target=self.ask, args=(draft, user), name=f"draft-{draft.index}", daemon=True)
                   for draft in self.drafts]
        for thread in threads:
            thread.start()
        until = began + max(0.0, self.left()) * ROUND_SHARE
        for thread in threads:
            thread.join(max(0.0, until - time.time()))
        with self.gate:
            for draft in self.drafts:
                if not draft.done:
                    draft.abandoned = True
                    draft.error = draft.error or "no reply in the round's time"
        for draft in self.drafts:
            if draft.reply is None:
                self.log("draft", f"{draft.index}: no reply ({draft.error})")
                continue
            reply, code = draft.reply, draft.code
            self.log("draft", f"{draft.index}: {reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} "
                              f"completion tokens, finish {reply.finish_reason}; "
                              + (f"a script of {len(code.text):,} bytes{', cut' if draft.cut else ''}" if code else
                                 "no script"))
        answered = [draft for draft in self.drafts if draft.reply is not None]
        scripts = [draft for draft in answered if draft.code is not None]
        self.log("round", f"{len(answered)} of {self.count} drafts answered in {time.time() - began:.0f} s; {len(scripts)} "
                          f"script(s), {sum(draft.cut for draft in scripts)} cut; "
                          f"{self.client.usage.turned_away - turned} turned away (407)")

    # --------------------------------------------------------------------------------------------------- the filter

    def keep(self, draft: Draft) -> bool:
        """What the draft's run left in --out, moved to its store; False where an entry could not be moved (it is
        removed, so the next run starts empty)."""
        draft.store = self.work / f"draft_{draft.index}" / "out"
        draft.store.mkdir(parents=True, exist_ok=True)
        whole = True
        for entry in sorted(self.out.iterdir()):
            if entry.name in self.before:
                continue
            try:
                shutil.move(str(entry), str(draft.store / entry.name))
            except OSError:
                whole = False
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
        return whole

    def run(self, draft: Draft, remaining: int, kept: float) -> None:
        """The draft's script in its own work directory, under its cap (`run_cap`): `kept` is held back for a repair
        while one can still follow (no draft clean yet, a request left)."""
        left = self.left()
        repair = not any(other.clean for other in self.drafts) and self.requests_left() >= 1
        seconds = self.cap(left, remaining, kept if repair else 0.0)
        text = draft.code.text
        if not draft.code.closed:       # cut at the output cap: its longest prefix that compiles (N1's salvage)
            text = runner.parsing_prefix(runner.complete_lines(text))
            draft.kept = lines(text)
            self.log("salvage", f"draft {draft.index}: {draft.kept:,} of {lines(draft.code.text):,} lines kept, the "
                                 f"rest cut at the output limit")
        if seconds < MIN_RUN_S or not text:
            self.log("run", f"draft {draft.index}: not started: " + (f"{max(left, 0):.0f} s left" if text else
                                                                       "no part of it compiles"))
            return
        self.runner = self.runner_in(f"draft_{draft.index}")
        ran = self.runner.run(text, seconds)
        with guard.WRITING:
            if self.ended.is_set():
                return
            whole = self.keep(draft)
            found = files_in(draft.store)
            draft.ran = ran
            draft.clean = not ran.failed and whole and any("/" not in name for name in found)
            draft.complete = draft.clean and all(name in found for name in self.named)
        self.log("run", f"draft {draft.index}: exit {ran.exit} after {ran.seconds:.1f} s of {seconds:.0f}; "
                        f"{'a traceback; ' if runner.TRACEBACK in ran.stderr else ''}{len(found)} file(s)"
                        f"{'' if whole else ', one could not be kept'}; "
                        + ("clean, named-complete" if draft.complete else "clean" if draft.clean else "not clean"))

    def cap(self, left: float, remaining: int, kept: float) -> float:
        """A draft's run cap (`run_cap`; agent/n11/drafts.py's round bounds it further)."""
        return run_cap(left, remaining, kept)

    def deliver(self, draft: Draft) -> int:
        """The draft's files copied into --out (the caller holds WRITING); the files copied."""
        copied = 0
        for name, path in files_in(draft.store).items():
            target = self.out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied += 1
        self.delivered = True
        return copied

    def verdict(self) -> Verdict:
        clean = [draft for draft in self.drafts if draft.clean]
        return vote({draft.index: files_in(draft.store) for draft in clean},
                    {draft.index for draft in clean if draft.complete}, self.named)

    def finish(self) -> None:
        """The watchdog's, holding WRITING: the best clean draft so far delivered where nothing is yet (the running
        draft's own files cleared first), each draft's outcome so far, then no more writes."""
        self.ended.set()
        if self.delivered or not any(draft.clean for draft in self.drafts):
            self.outcomes(self.verdict())
            return
        chosen = self.verdict()
        for entry in self.out.iterdir():
            if entry.name not in self.before:
                shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
        files = self.deliver(self.drafts[chosen.winner])
        self.log("vote", f"at the watchdog: winner preamble {chosen.winner}, cluster size {len(chosen.cluster)}, "
                         f"files {files}")
        self.outcomes(chosen)

    def outcomes(self, chosen: Verdict) -> None:
        """Each draft's outcome, once: a line of the step log (the record's, the trace on or off), and under the
        desk's trace a `draft` event and its files copied to the trace's `drafts/<i>/`. The line quotes no text of the
        unit's or the script's; nothing is written into --out."""
        with self.gate:
            if self.reported:
                return
            self.reported = True
        group_of = {index: group for group in chosen.clusters for index in group}
        where = trace.directory()
        for draft in self.drafts:
            reply, ran, group = draft.reply, draft.ran, group_of.get(draft.index)
            found = sorted(files_in(draft.store)) if draft.store is not None else []
            request = ({"error": draft.error} if reply is None else
                       {"finish": reply.finish_reason, "prompt_tokens": reply.prompt_tokens,
                        "completion_tokens": reply.completion_tokens, "script": draft.code is not None,
                        "cut": draft.cut, "salvaged": draft.kept is not None, "kept_lines": draft.kept})
            run = None if ran is None else {"exit": ran.exit, "seconds": round(ran.seconds, 3),
                                            "stderr_tail": self.redact(ran.stderr)}
            trace.event("draft", index=draft.index, preamble=PREAMBLES[draft.index], request=request, run=run,
                        clean=draft.clean, complete=draft.complete, files=found,
                        cluster=list(group) if group else None, winner=draft.index == chosen.winner)
            if where is not None and found:
                try:
                    for name in found:
                        target = where / "drafts" / str(draft.index) / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(draft.store / name, target)
                except OSError:
                    pass                        # the desk's copy alone is lost: the unit goes on as before
            said = (f"no reply ({draft.error})" if reply is None else
                    f"finish {reply.finish_reason}, {reply.completion_tokens:,} tokens"
                    + (", no script" if draft.code is None else ", cut" if draft.cut else "")
                    + (f", salvaged {draft.kept:,} lines" if draft.kept is not None else ""))
            self.log(f"draft {draft.index}", f"{PREAMBLES[draft.index]}; {said}; "
                     + ("not run" if ran is None else f"run exit {ran.exit} in {ran.seconds:.1f} s") + "; "
                     + ("clean, named-complete" if draft.complete else "clean" if draft.clean else "not clean")
                     + f"; {len(found)} file(s); "
                     + ("not voted" if group is None else f"cluster {'+'.join(map(str, group))}"
                        + (", the winner" if draft.index == chosen.winner else "")))

    # --------------------------------------------------------------------------------------------------- the repair

    def repair(self, user: str) -> None:
        """No draft is clean: the one that ran longest (else the first with a script, else the first that replied)
        laid into --out, N0's repair asked once under its preamble, and the repaired script run over its files."""
        ran = [draft for draft in self.drafts if draft.ran is not None]
        scripts = [draft for draft in self.drafts if draft.code is not None]
        replied = [draft for draft in self.drafts if draft.reply is not None]
        pick = (max(ran, key=lambda draft: (draft.ran.seconds, -draft.index)) if ran else
                scripts[0] if scripts else replied[0] if replied else None)
        if pick is None:
            self.log("repair", "not asked: no draft replied")
            trace.event("stop", reason="no draft replied")
            return
        if self.requests_left() < 1 or self.left() < MIN_RUN_S:
            self.log("repair", "not asked: " + ("no request left" if self.requests_left() < 1 else "no time left"))
            trace.event("stop", reason="no clean draft and no repair: " + ("no request left" if
                                                                           self.requests_left() < 1 else
                                                                           "no time left"))
            return
        with guard.WRITING:
            if self.ended.is_set():
                return
            laid = self.deliver(pick) if pick.store is not None else 0
        if pick.code is None:
            what = "Your reply held no script in a ```python block."
            why = "no script in the reply"
        else:
            run = pick.ran
            why = "it could not be started in the time left" if run is None else run.why()
            if run is not None and not run.failed:
                why = "it exited with status 0 but wrote no file at the output directory's root"
            if pick.kept is not None:
                cut = "it was too long for one reply and was cut off at the output limit"
                why = (f"{cut}, and no part of it compiles, so nothing was run" if not pick.kept else
                       f"{cut}, so only its first {pick.kept:,} of {lines(pick.code.text):,} lines, the part that "
                       f"compiles, were run: {why}")    # a salvage not run for time leaves none for a repair
            what = (f"# Your previous script\n\n```python\n{pick.code.text}```\n\n# What happened\n\n{why}.\n\n"
                    + (f"The end of its error output:\n```\n{run.stderr.strip()[-runner.TAIL_BYTES:]}\n```\n\n"
                       if run is not None and run.stderr.strip() else "")
                    + listing(self.out))
        self.log("repair", f"draft {pick.index}: {why}; {laid} of its file(s) laid in the output")
        trace.event("repair", reason=why, draft=pick.index)
        try:
            reply = self.client.chat(SYSTEMS[pick.index],
                                     self.redact(f"{user}\n{what}\n\nReply with the complete corrected script.\n"))
        except model.ModelError as exc:
            self.log("repair", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
            trace.event("stop", reason=f"the House did not answer the repair ({type(exc).__name__})")
            return
        code = runner.extract(reply.text)
        self.log("repair", f"{reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} completion tokens, finish "
                           f"{reply.finish_reason}; " + (f"a script of {len(code.text):,} bytes" if code else
                                                         "no script"))
        if code is None:
            trace.event("stop", reason="the repair held no script")
            return
        left = self.left()
        if left < MIN_RUN_S:
            self.log("run", f"repair: not started: {max(left, 0):.0f} s left")
            trace.event("stop", reason="no time left for the repaired script")
            return
        self.runner = self.runner_in("repair")
        done = self.runner.run(code.text, left)
        self.log("run", f"repair: exit {done.exit} after {done.seconds:.1f} s of {left:.0f}; "
                        f"{'a traceback; ' if runner.TRACEBACK in done.stderr else ''}"
                        f"{len(contract.deliverables(self.out))} file(s) at the output's root")
        trace.event("stop", reason="the repaired script ran")

    # ---------------------------------------------------------------------------------------------------- the unit

    def solve(self) -> None:
        task = self.unit = unit.read(self.task_dir)
        self.log("unit", f"{len(task.instruction):,} bytes of instruction, {task.dropped} canary line(s) removed, "
                         f"{self.deadline - time.time():.0f} s to the deadline")
        pathmap = paths.build(task.task_dir, task.instruction, task.dockerfile, self.out)
        data = digest.digest(task.data_dir, canaries=task.canaries)
        inputs = [path.name for path in task.data_dir.rglob("*") if path.is_file()] if task.data_dir.is_dir() else []
        self.named = contract.named_all(pathmap, task.instruction, inputs)
        self.before = {entry.name for entry in self.out.iterdir()}
        kinds = sorted({entry.kind for entry in pathmap.entries})
        self.log("digest", f"{len(data.encode()):,} bytes; {len(pathmap.entries)} path(s) named ({', '.join(kinds)}); "
                           f"{len(self.named)} output file(s) named")
        user = self.redact(prompt(task, pathmap, data, max(0.0, self.left()) * RUN_SHARE / DRAFTS))
        self.round(user)
        scripts = [draft for draft in self.drafts if draft.code is not None]
        kept = max(0.0, self.left()) * (1 - RUN_SHARE)       # the repair's: the runs share the rest
        for number, draft in enumerate(scripts):
            if self.ended.is_set():
                return
            self.run(draft, len(scripts) - number, kept)
        chosen = self.verdict()
        clean = sum(draft.clean for draft in self.drafts)
        if chosen.winner is None:
            self.log("vote", f"0 clean of {DRAFTS}")
            trace.event("vote", clean=0, winner=None)
            self.outcomes(chosen)
            self.repair(user)
            return
        with guard.WRITING:
            if self.ended.is_set():
                return
            files = self.deliver(self.drafts[chosen.winner])
        complete = sum(draft.complete for draft in self.drafts)
        self.log("vote", f"{clean} clean of {DRAFTS}, winner preamble {chosen.winner}, cluster size "
                         f"{len(chosen.cluster)}, files {files} ({complete} named-complete; clusters {chosen.sizes()})")
        trace.event("vote", clean=clean, complete=complete, winner=chosen.winner, cluster=list(chosen.cluster),
                    clusters=[list(group) for group in chosen.clusters], files=files)
        self.outcomes(chosen)
        trace.event("stop", reason="the vote's winner delivered")


def solve(task_dir: Path, out: Path, *, client=None, work: Path | None = None, requests: int = REQUESTS,
          exit=os._exit, watchdog_s: float = guard.WATCHDOG_S, stream=None) -> int:
    """One unit, start to close; 0 whatever happens. `requests` lowers the cap (a desk probe's budget), never raises
    it; `client`, `exit` and `watchdog_s` are a test's."""
    out = Path(out)
    steps = guard.Steps(AGENT, lambda: client.usage.requests if client is not None else 0, stream)
    keep = work is not None
    canaries: frozenset[str] = frozenset()
    watchdog = agent = None
    try:
        out.mkdir(parents=True, exist_ok=True)
        trace.begin(AGENT, out)
        if client is None:
            client = house.client(Path(task_dir) / "card.toml", max(0, min(requests, REQUESTS)), out)
        deadline = client.deadline if client.deadline is not None else time.time() + model.UNIT_SECONDS
        work = Path(work) if keep else Path(tempfile.mkdtemp(prefix=f"{AGENT}-"))
        work.mkdir(parents=True, exist_ok=True)
        agent = W1(task_dir, out, client, steps, work, deadline)

        def finish() -> None:
            try:
                agent.finish()
            finally:
                closed = guard.close(out, agent.unit.canaries if agent.unit else ())
                steps("close", closed.summary())

        watchdog = guard.Watchdog(deadline, agent.kill, finish, steps, reserve=watchdog_s, exit=exit).arm()
        agent.solve()
    except BaseException as exc:  # noqa: BLE001 - the exit is 0 whatever failed
        steps("error", failure(exc))
    finally:
        try:
            if watchdog is not None:
                watchdog.cancel()
            if agent is not None and agent.unit is not None:
                canaries = agent.unit.canaries
            with guard.WRITING:
                closed = guard.close(out, canaries)
            steps("close", closed.summary())
            if work is not None and not keep:
                shutil.rmtree(work, ignore_errors=True)
        except BaseException as exc:  # noqa: BLE001
            steps("error", failure(exc))
    return 0


def failure(exc: BaseException) -> str:
    """The exception's type and where it was raised; a model error's own message (model.py's, which never carries a
    credential), with any canary line dropped. No other message is logged: it could quote the instruction."""
    frame = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
    where = f" at {Path(frame.filename).name}:{frame.lineno}" if frame else ""
    said = f": {logged(str(exc))[:200]}" if isinstance(exc, model.ModelError) else ""
    return f"{type(exc).__name__}{where}{said}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="solve", description="W1: five concurrent drafts, the clean ones voted on.")
    parser.add_argument("verb", choices=["solve"])
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default=None, help="keep each draft's script, logs and files here (the desk's; the "
                                                      "image's go to a temporary directory, removed at the end)")
    parser.add_argument("--requests", type=int, default=REQUESTS, help=f"at most this many requests, below "
                                                                        f"{REQUESTS} (a desk probe's budget)")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 0
    return solve(Path(args.task_dir), Path(args.out), work=Path(args.work) if args.work else None,
                 requests=args.requests)


if __name__ == "__main__":
    try:
        main()
    finally:
        # The status is 0 whatever the interpreter's own shutdown meets: a stderr nobody reads fails its last flush,
        # which Python reports as status 120.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        os._exit(0)
