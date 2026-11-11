"""Step 3: before the loop, one request for a complete script, run once, its output files held aside as a fallback
(the "floor") and written out at the close only if the loop leaves nothing to grade.

Called by: agent/n11/solve.py (`N11.drafts_first` starts it, `N11.drafts_close` and `N11.release` write the floor).
Calls: common/runner.py to take the script from the reply and run it; the model through a second client
    (common/house.py `Client.sibling`) that shares the unit's request count.
Start at: `Round.go` (the request and the run) and `Round.close` (the floor written out).

The rules in detail
-------------------
The first draft before the loop: one request for a complete script, the script run once, and its files held aside
as the unit's floor.

The round (`Round`, on the loop's clock) runs before the loop's first request, after the selection call, and only
where the loop keeps `FLOOR_REQUESTS` beyond the round's `ALLOWANCE` (requests, retries counted). The draft's request
is a one-shot baseline's (`SYSTEM` and `prompt`: the instruction, the path map, the data's digest, the run's seconds),
at the client's own temperature, through a client of its own (common.house `Client.sibling`: the unit's one count and
cap), sent on a thread and waited for at most `ROUND_SHARE` of the time left, its request bounded to that time.
Its script runs in a directory of its own under the loop's work (`WORK`), at most `RUN_S` and its share of the time
left (`run_cap`); a script the output cap cut runs as its longest prefix that compiles. Its files are then moved to a
store of its own, so the loop starts from the output as it was (`keep`). The draft is clean when its run exits 0 and
leaves a file at the output's root.

The floor is never written before the loop, so the loop is never told of files it did not write: a clean draft's files
stay in its store, and at the unit's close (`close`; the watchdog's too, `finish`) they are written to the output, what
it holds first removed, only where it holds nothing the harness grades (`graded`: no file at its root, or none of the
files the task names). A draft lost (no reply, a refusal, a failing run, any error in the round) costs the draft,
never the unit.
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from common import canary, guard, paths, runner, unit
from common.house import model

ALLOWANCE = 2                   # the round's requests at most, retries counted (room for a retry)
FLOOR_REQUESTS = 2              # the round runs only where the loop keeps this many requests beyond the allowance
RUN_S = 60.0                    # the draft's run at most
WORK = "drafts"                 # the round's directory under the loop's work
ROUND_SHARE = 0.5               # the round waits for its draft at most this share of the time left
RUN_SHARE = 0.5                 # the draft's run's share of the time left at its start
MIN_RUN_S = 5.0                 # a run is not started with less time than this

LIBRARIES = ("numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, openpyxl, matplotlib (the Agg "
             "backend), plotly, seaborn, TA-Lib (imported as talib) and backtrader")

SYSTEM = f"""You are an expert quantitative developer. You solve a task by writing one complete Python 3 script, \
which is run once, unattended, with no network; hidden tests then check the files it wrote against the task.

Reply with the whole script in a single ```python fenced block.

The script must:
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


def prompt(task: unit.Unit, pathmap: paths.PathMap, data: str, seconds: float) -> str:
    """The draft's user turn."""
    return (f"# Task\n\n{task.instruction.strip()}\n\n# Paths\n\n{pathmap.text()}\n\n# Input data\n\n{data.strip()}"
            f"\n\n# Time\n\nThe script is stopped after {seconds:.0f} seconds.\n\nWrite the complete script now.\n")


def delivered(out: Path) -> bool:
    """A regular file at the output's root besides the scorer's own."""
    try:
        return any(path.is_file() and not path.is_symlink() and path.name not in guard.SCORER_FILES
                   for path in out.iterdir())
    except OSError:
        return False


def graded(out: Path, named) -> bool:
    """Whether the output holds a file the harness grades: a regular file at its root besides the scorer's own
    (`delivered`) and, where the task names output files, one of them."""
    out = Path(out)
    return delivered(out) and (not named or any((out / name).is_file() for name in named))


def run_cap(left: float, remaining: int, kept: float) -> float:
    """A draft's run cap: the time left less what is `kept`, shared among the drafts still to run, never under
    `MIN_RUN_S` nor over the time left."""
    return min(left, max(MIN_RUN_S, (left - kept) / remaining))


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


def cleared(out: Path, keep=frozenset()) -> None:
    """Every entry of the output removed but those named in `keep`, a link as itself, never followed."""
    for entry in sorted(Path(out).iterdir()):
        if entry.name in keep:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


@dataclass
class Draft:
    index: int
    reply: model.Reply | None = None
    error: str = ""                         # why it has no reply
    code: runner.Code | None = None
    ran: runner.Run | None = None
    store: Path | None = None               # its files, moved there from the output after its run
    clean: bool = False
    kept: int | None = None                 # a cut script's lines its salvage kept; None where it was not cut
    done: bool = False                      # its request answered or failed
    abandoned: bool = False                 # still unanswered when the round ended: whatever comes later is dropped

    @property
    def cut(self) -> bool:
        return self.reply is not None and (self.reply.finish_reason == "length" or
                                           (self.code is not None and not self.code.closed))


class Round:
    """The round over one unit (above), for the loop `loop` (agent/n11/solve.py's N11)."""

    def __init__(self, loop, task: unit.Unit, pathmap: paths.PathMap, data: str) -> None:
        self.loop, self.unit, self.pathmap, self.data = loop, task, pathmap, data
        self.out, self.client, self.work = Path(loop.out), loop.client, Path(loop.work) / WORK
        self.named = tuple(loop.named_files)
        self.drafts: list[Draft] = []
        self.draft_client = None        # the draft's own client (`Client.sibling`)
        self.runner: runner.Runner | None = None      # the one running, for the watchdog's kill
        self.before: set[str] = set()                 # what the output held before any run: never moved
        self.delivered = False                        # the output holds the floor
        self.ended = threading.Event()                # the watchdog has closed the output: nothing more is written
        self.gate = threading.Lock()
        self.reported = False                         # the draft's outcome logged (once)
        self.bound = 0.0                # the seconds the draft's request may take: the round's
        self.reserve = 0.0              # the time left kept from the run: the loop's
        self.held: int | None = None    # the draft whose files are the floor, held aside until the close
        self.closed = False             # the close read (`close`), once
        self.done = False               # the floor held (or the round given up): the watchdog leaves the output alone

    def log(self, step: str, detail: str = "") -> None:
        self.loop.log(f"drafts {step}", detail)

    def left(self) -> float:
        return self.loop.left()

    def redact(self, text: str) -> str:
        return canary.redact(text, self.unit.canaries if self.unit else ())

    def kill(self) -> None:
        if self.runner is not None:
            self.runner.kill()

    def ask(self, draft: Draft, user: str) -> None:
        """The draft's request, on its own thread, bounded to the round, at the client's own temperature; its outcome
        kept unless the round has already ended."""
        reply, error = None, ""
        try:
            client = self.draft_client
            reply = client.chat(SYSTEM, user, bound=max(model.MIN_CALL_S, self.bound))
        except BaseException as exc:  # noqa: BLE001 - any error loses this draft alone
            error = type(exc).__name__ + (f": {logged(str(exc))[:200]}" if isinstance(exc, model.ModelError) else "")
        with self.gate:
            if draft.abandoned:
                return
            draft.reply, draft.error, draft.done = reply, error, True
            draft.code = runner.extract(reply.text) if reply is not None else None

    def round(self, user: str) -> None:
        """The draft's request, waited for at most `ROUND_SHARE` of the time left."""
        began = time.time()
        draft = Draft(0)
        self.drafts = [draft]
        thread = threading.Thread(target=self.ask, args=(draft, user), name=f"draft-{draft.index}", daemon=True)
        thread.start()
        until = began + max(0.0, self.left()) * ROUND_SHARE
        thread.join(max(0.0, until - time.time()))
        with self.gate:
            if not draft.done:
                draft.abandoned = True
                draft.error = draft.error or "no reply in the round's time"
        if draft.reply is None:
            self.log("draft", f"{draft.index}: no reply ({draft.error})")
            return
        reply, code = draft.reply, draft.code
        self.log("draft", f"{draft.index}: {reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} completion "
                          f"tokens, finish {reply.finish_reason}; "
                          + (f"a script of {len(code.text):,} bytes{', cut' if draft.cut else ''}" if code else
                             "no script"))

    def keep(self, draft: Draft) -> bool:
        """What the draft's run left in the output, moved to its store (`stored`); False where an entry could not be
        moved, and then the store is removed too, releasing space for the loop."""
        try:
            whole = self.stored(draft)
        except OSError:
            cleared(self.out, self.before)
            whole = False
        if not whole and draft.store is not None:
            shutil.rmtree(draft.store, ignore_errors=True)
        return whole

    def stored(self, draft: Draft) -> bool:
        """What the draft's run left in the output, moved to its store; False where an entry could not be moved (it is
        removed, so the loop starts from the output as it was)."""
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

    def run(self, draft: Draft, remaining: int) -> None:
        """The draft's script in its own work directory: `RUN_S` at most, and its share of the time left less the
        loop's reserve (`run_cap`)."""
        left = self.left()
        seconds = min(RUN_S, run_cap(left, remaining, self.reserve))
        text = draft.code.text
        if not draft.code.closed:       # cut at the output cap: its longest prefix that compiles
            text = runner.parsing_prefix(runner.complete_lines(text))
            draft.kept = lines(text)
            self.log("salvage", f"draft {draft.index}: {draft.kept:,} of {lines(draft.code.text):,} lines kept, the "
                                 f"rest cut at the output limit")
        if seconds < MIN_RUN_S or not text:
            self.log("run", f"draft {draft.index}: not started: " + (f"{max(left, 0):.0f} s left" if text else
                                                                       "no part of it compiles"))
            return
        work = self.work / f"draft_{draft.index}"
        work.mkdir(parents=True, exist_ok=True)
        self.runner = runner.Runner(work, out=self.out)
        ran = self.runner.run(text, seconds)
        with guard.WRITING:
            if self.ended.is_set():
                return
            whole = self.keep(draft)
            found = files_in(draft.store)
            draft.ran = ran
            draft.clean = not ran.failed and whole and any("/" not in name for name in found)
        self.log("run", f"draft {draft.index}: exit {ran.exit} after {ran.seconds:.1f} s of {seconds:.0f}; "
                        f"{'a traceback; ' if runner.TRACEBACK in ran.stderr else ''}{len(found)} file(s)"
                        f"{'' if whole else ', one could not be kept'}; " + ("clean" if draft.clean else "not clean"))

    def winner(self) -> int | None:
        """The floor: the draft where it is clean, else none."""
        return next((draft.index for draft in self.drafts if draft.clean), None)

    def deliver(self, draft: Draft) -> int:
        """The draft's files copied into the output (the caller holds WRITING); the files copied."""
        copied = 0
        for name, path in files_in(draft.store).items():
            target = self.out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied += 1
        self.delivered = True
        return copied

    def finish(self) -> None:
        """The watchdog's, holding WRITING: while the round is open, the running script's files cleared and a clean
        draft's laid where nothing is yet; after it, the held floor as at the close (`close`). No more writes."""
        if self.done:
            self.close()
            return
        cleared(self.out, self.before)
        self.ended.set()
        winner = self.winner()
        if self.delivered or winner is None:
            self.outcome(winner)
            return
        for entry in self.out.iterdir():
            if entry.name not in self.before:
                shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
        files = self.deliver(self.drafts[winner])
        self.log("floor", f"at the watchdog: draft {winner}'s {files} file(s)")
        self.outcome(winner)

    def close(self) -> None:
        """At the unit's close, the caller holding WRITING, once: the held floor written to the output where it holds
        nothing the harness grades (`graded`), what it holds first removed; nothing where no floor is held."""
        if self.held is None or self.closed:
            return
        self.closed = True
        if graded(self.out, self.named):
            self.log("floor", "not written at the close: the output holds a file to grade")
            return
        cleared(self.out, self.before)
        laid = self.deliver(self.drafts[self.held])
        self.log("floor", f"written at the close: draft {self.held}'s {laid} file(s), the output holding nothing to grade")

    def go(self) -> None:
        """The round and the run, the floor held aside (`held`)."""
        began = self.left()
        self.out.mkdir(parents=True, exist_ok=True)
        self.before = {entry.name for entry in self.out.iterdir()}
        since, cap = self.client.deadline, self.client.usage.requests + ALLOWANCE
        self.draft_client = self.client.sibling(cap)
        self.bound = max(0.0, began) * ROUND_SHARE
        seconds = min(RUN_S, max(MIN_RUN_S, max(0.0, began) * RUN_SHARE))
        user = self.redact(prompt(self.unit, self.pathmap, self.data, seconds))
        self.round(user)
        if moved := self.client.carry([self.draft_client], since):
            self.log("clock", f"{moved:.0f} s of the draft's untimed waits carried onto the unit's clock")
        scripts = [draft for draft in self.drafts if draft.code is not None]
        self.reserve = max(0.0, self.left()) * (1 - RUN_SHARE)
        for number, draft in enumerate(scripts):
            if self.ended.is_set():
                return
            self.run(draft, len(scripts) - number)
        winner = self.winner()
        with guard.WRITING:
            if self.ended.is_set():
                return
            self.done, self.held = True, winner
        self.outcome(winner)
        self.log("floor", "none: the draft is not clean" if winner is None else
                 f"held aside: draft {winner}'s {len(files_in(self.drafts[winner].store))} file(s)")

    def given_up(self, why: str) -> None:
        """The round failed in its own code: what it left in the output removed and no floor held, so the loop starts
        and ends as without it."""
        with guard.WRITING:
            if not self.ended.is_set():
                cleared(self.out, self.before)
            self.done, self.held = True, None
        self.log("given up", why)

    def outcome(self, winner: int | None) -> None:
        """The draft's outcome, once, a line of the step log; the line quotes no text of the unit's or the script's."""
        with self.gate:
            if self.reported:
                return
            self.reported = True
        for draft in self.drafts:
            reply, ran = draft.reply, draft.ran
            found = sorted(files_in(draft.store)) if draft.store is not None else []
            said = (f"no reply ({draft.error})" if reply is None else
                    f"finish {reply.finish_reason}, {reply.completion_tokens:,} tokens"
                    + (", no script" if draft.code is None else ", cut" if draft.cut else "")
                    + (f", salvaged {draft.kept:,} lines" if draft.kept is not None else ""))
            self.log(f"draft {draft.index}", f"{said}; "
                     + ("not run" if ran is None else f"run exit {ran.exit} in {ran.seconds:.1f} s") + "; "
                     + ("clean" if draft.clean else "not clean") + f"; {len(found)} file(s)"
                     + ("; the floor" if draft.index == winner else ""))
