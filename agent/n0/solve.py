"""N0: the naive one-call agent, hardened. `python3 -m n0.solve solve --task-dir /input --out /app/output`.

The floor every later agent is compared against, and the proof of the whole path (agenthon-t4's track1-handoff.md,
"The agents"). One request asks the House for a complete script, shown the instruction with its canary lines
removed, the path map (common.paths) and a digest of the data (common.digest); the script is run under a timeout
(common.runner). On a failure (a non-zero exit, a traceback, a timeout, no file written, or no script in the reply)
one repair request carries the traceback's tail and the files now in `--out`, and the repaired script runs in the
time left. A reply the output cap cut (`finish_reason` "length", or a block left open) earns one continuation,
appended. At most `REQUESTS` requests in all, attempts and retries counted (common.house), and the unit's deadline
(model.unit_deadline: the card's clock under the roster cap) over everything; the watchdog ends the process with
status 0 before it (common.guard). Whatever happens, the output is closed (common.guard.close) and the exit is 0.
A unit whose model call fails leaves `--out` empty, which the scorer reads as `no_output`: a placeholder file would
earn nothing either (a unit scores only when its checks pass whole, docs/challenge.md "The score"), and rule 9
gives no credit to a file the House did not shape.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

from common import canary, digest, guard, house, paths, runner, trace, unit
from common.house import model

AGENT = "n0"
REQUESTS = 3                # one script, one repair, one continuation of a reply the output cap cut
RUN_SHARE = 0.5             # the first run's share of the time left while a repair can still follow it
RUN_RESERVE_S = 20.0        # kept from the deadline after the last run: the close, before the watchdog's 10 s
MIN_RUN_S = 5.0             # a run is not started with less time than this

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


def delivered(out: Path) -> bool:
    """A regular file at the output's root besides the scorer's own: what g1 asks."""
    try:
        return any(path.is_file() and not path.is_symlink() and path.name not in guard.SCORER_FILES
                   for path in out.iterdir())
    except OSError:
        return False


def logged(text: str) -> str:
    """A message for the log: any line carrying a GUID-shaped token dropped."""
    return canary.strip(text)[0]


class N0:
    def __init__(self, task_dir: Path, out: Path, client, steps: guard.Steps, work: Path, deadline: float) -> None:
        self.task_dir, self.out, self.client, self.log = Path(task_dir), Path(out), client, steps
        self.deadline = deadline
        self.runner = runner.Runner(work, out=self.out)
        self.unit: unit.Unit | None = None

    def left(self) -> float:
        return self.deadline - RUN_RESERVE_S - time.time()

    def redact(self, text: str) -> str:
        """Every prompt as sent: the unit's canaries read `<canary>`, the script and traceback a repair quotes
        included."""
        return canary.redact(text, self.unit.canaries if self.unit else ())

    def requests_left(self) -> int:
        return self.client.cap - self.client.usage.requests

    def ask(self, user: str, step: str) -> runner.Code | None:
        """One request, and its continuation where the output cap cut the script and a request remains; None where
        the reply held no script. A request the House never answered raises `model.ModelError`: no later request
        would fare better (the client has spent its attempts, or the cap, the clock or the credential refused it)."""
        reply = self.client.chat(SYSTEM, self.redact(user))
        code = runner.extract(reply.text)
        cut = reply.finish_reason == "length" or (code is not None and not code.closed)
        self.log(step, f"{reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} completion tokens, finish "
                       f"{reply.finish_reason}; " + (f"a script of {len(code.text):,} bytes" if code else "no script"))
        if code is None or not cut:
            return code
        if self.requests_left() < 1:
            return code
        partial = runner.complete_lines(code.text)
        last = partial.rstrip("\n").rpartition("\n")[2]
        rest = (f"{user}\n# Your script so far, cut off at the output limit\n\n```python\n{partial}```\n\nContinue "
               f"the script from the line after its last line above (`{last.strip()[:120]}`). Reply with only the "
               f"remaining lines in one ```python block, repeating no line already written.\n")
        try:
            more = self.client.chat(SYSTEM, self.redact(rest))
        except model.ModelError as exc:
            self.log("continue", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
            return code
        joined = runner.append(partial, more.text)
        self.log("continue", f"{more.completion_tokens:,} completion tokens, finish {more.finish_reason}; the script "
                             f"now {len(joined):,} bytes")
        return runner.Code(joined, more.finish_reason != "length")

    def execute(self, code: runner.Code, last: bool) -> runner.Run | None:
        left = self.left()
        seconds = left if last or self.requests_left() < 1 else max(MIN_RUN_S, left * RUN_SHARE)
        if seconds < MIN_RUN_S:
            self.log("run", f"not started: {max(left, 0):.0f} s left")
            return None
        ran = self.runner.run(code.text, seconds)
        self.log("run", f"exit {ran.exit} after {ran.seconds:.1f} s of {seconds:.0f}; "
                        f"{'a traceback; ' if runner.TRACEBACK in ran.stderr else ''}"
                        f"{'a file' if delivered(self.out) else 'no file'} at the output's root")
        return ran

    def solve(self) -> None:
        task = self.unit = unit.read(self.task_dir)
        self.log("unit", f"{len(task.instruction):,} bytes of instruction, {task.dropped} canary line(s) removed, "
                         f"{self.deadline - time.time():.0f} s to the deadline; roster cap "
                         f"{(roster := model.roster_cap())[0]:g} s ({roster[1]})")
        pathmap = paths.build(task.task_dir, task.instruction, task.dockerfile, self.out)
        data = digest.digest(task.data_dir, canaries=task.canaries)
        kinds = sorted({entry.kind for entry in pathmap.entries})
        self.log("digest", f"{len(data.encode()):,} bytes; {len(pathmap.entries)} path(s) named ({', '.join(kinds)})")
        user = prompt(task, pathmap, data, max(0.0, self.left()) * RUN_SHARE)
        try:
            code = self.ask(user, "request")
        except model.ModelError as exc:
            self.log("request", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
            trace.event("stop", reason=f"the House did not answer the request ({type(exc).__name__})")
            return
        ran = self.execute(code, last=False) if code else None
        if ran is not None and not ran.failed and delivered(self.out):
            trace.event("stop", reason="the script ran and wrote a file at the output's root")
            return
        if self.requests_left() < 1 or self.left() < MIN_RUN_S:
            self.log("repair", "not asked: " + ("no request left" if self.requests_left() < 1 else "no time left"))
            trace.event("stop", reason="no repair: " + ("no request left" if self.requests_left() < 1 else
                                                         "no time left"))
            return
        if code is None:
            what = "Your reply held no script in a ```python block."
        else:
            why = "it could not be started in the time left" if ran is None else ran.why()
            if ran is not None and not ran.failed:
                why = "it exited with status 0 but wrote no file at the output directory's root"
            what = (f"# Your previous script\n\n```python\n{code.text}```\n\n# What happened\n\n{why}.\n\n"
                    + (f"The end of its error output:\n```\n{ran.stderr.strip()[-runner.TAIL_BYTES:]}\n```\n\n"
                       if ran is not None and ran.stderr.strip() else "")
                    + listing(self.out))
        trace.event("repair", reason="no script in the reply" if code is None else why)
        try:
            fixed = self.ask(f"{user}\n{what}\n\nReply with the complete corrected script.\n", "repair")
        except model.ModelError as exc:
            self.log("repair", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
            trace.event("stop", reason=f"the House did not answer the repair ({type(exc).__name__})")
            return
        if fixed is not None:
            self.execute(fixed, last=True)
        trace.event("stop", reason="the repaired script ran" if fixed is not None else "the repair held no script")


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
        agent = N0(task_dir, out, client, steps, work, deadline)

        def finish() -> None:
            closed = guard.close(out, agent.unit.canaries if agent.unit else ())
            steps("close", closed.summary())

        watchdog = guard.Watchdog(deadline, agent.runner.kill, finish, steps, reserve=watchdog_s, exit=exit).arm()
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
    parser = argparse.ArgumentParser(prog="solve", description="N0: one House request for a complete script.")
    parser.add_argument("verb", choices=["solve"])
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default=None, help="keep the scripts and their logs here (the desk's; the image's "
                                                      "go to a temporary directory, removed at the end)")
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
