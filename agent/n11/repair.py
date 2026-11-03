"""N11's concurrent repair pair, kept by execution (lane repairpair1; runs/lanes/repairpair1-prompt.md: the cold reader's
second concurrency form, runs/lanes/cold-sched1019.md sections (a) and (c), COLD ROW 669).

`AGENTHON_N11_REPAIR` (or agent/n11/REPAIR, one word, an image's; agent/n11/solve.py `repair_of`): `off`, today's, or
`pair`, armed in the loop's forms `loop` and `sketch` (`FORMS`). Armed, the loop's request after a turn whose run failed
its own observation (a traceback, an exit other than 0, a named output file missing: the checks failing) or whose
finish was not taken on a failing output (agent/n11/solve.py `repair_due`) is sent beside a second: A, the loop's own
request, on the unit's client; B, a rewrite (`rewrite_turn`: the loop's system prompt as A sends it, the task as the
loop's first message frames it, the failed run's scripts, its observation and `REWRITE`'s ask for work/main.py whole,
plain enough to fit one reply, `FIT`),
on a client of its own (common.house `Client.sibling`: the unit's one count under `LOCK`, its own pacing floor), at
once; a rewrite the output cap cut (a whole script too long for one reply, the live smoke's first pair) is continued
with B's second request as the loop's request 1 is (`rest`: its head kept to its last complete statement, the rest
asked under `REST_HEAD`, the two joined), and kept as one write where the join compiles. A's reply goes through the loop as today, its run into the output; B's script runs on B's thread in a directory
of its own under the loop's work (`WORK`; `Beside`: the loop workspace's settings, its work/ holding the loop's files
but work/main.py, `seeded`, `OUTPUT_DIR` forced to its own output, every literal naming the output's root in the
script naming it there, `rooted`), each under the loop's run cap and killed at it. The loop waits on B at most a
request's median after A's turn (B `abandoned` past it: its run killed, no further request or run). Before the loop's
next request (or at its stop) the pair is settled by the keep rule (`keep`): the
candidate that exited 0 with no traceback and every named output file present; then the one with more named files
present; then the one with fewer findings from the loop's own checks (the run's checks and sanity findings, lane
finishcheck1's contract where armed, lint's rows over the script); then A. A kept: B's directory removed and nothing
else changes. B kept: the loop's state as A's turn found it put back (agent/n11/solve.py `TURN_STATE`, the files under
work/ as they stood, `held`), the output's root emptied and B's files moved there, B's script written as work/main.py
and its run the turn's observation, the conversation carrying B's reply as one write of work/main.py and nothing of A's
(`solve.N11.repair_settle`), as if one request had been sent.

B is held to its own room so the loop keeps its own (`room`): at most `ROOM` requests a unit and never past the House's
cap less the loop's, `PER_PAIR` a pair (a retry counted); the loop's client's cap is raised by B's attempts as they are
counted, so the loop's requests left are today's (`Repair.unspent`). A repair turn is skipped where the time left is
under the run cap and a request's median (`skipped_clock`) or where B's room is spent (`skipped_requests`). The model is
told nothing of the pair, and no text states a count (the owner's in281). A failure in B's request, reply or run loses
B, never the unit. Under `off` every body is today's.

The trace: B's request and its runs under `KIND` (`rewrite`), a `repair` event at each send and each settle; the step
log's `repair` lines; the summary line carries `KEYS` (agent/n11/solve.py `repair_keys`).
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from common import guard, paths, runner, trace
from common.house import LOCK, model
from n0 import solve as n0
from n11 import commands, lint

ENV = "AGENTHON_N11_REPAIR"     # the desk's switch (src/agenthon_t1/harness.py hands it to an --agent run's container)
FILE = "REPAIR"                 # or agent/n11/REPAIR, one word, written by an image's builder
SETTINGS = ("off", "pair")      # the default first
FORMS = ("loop", "sketch")      # the forms it is armed in: those whose loop runs from its own first message
PER_PAIR = 2                    # B's requests a pair at most, a retry counted
ROOM = 5                        # B's requests a unit at most, never past the House's cap less the loop's (`room`)
KIND = "rewrite"                # B's requests' and runs' kind on the trace
WORK = "pair"                   # B's directories, under the loop's work
WORK_BYTES = 4 * 1024 * 1024    # the files under work/ kept to be put back where B is kept, at most
MAIN = f"{commands.WORK}/main.py"
WORKSPACE_STATE = ("runs", "params", "dated", "free", "lint_found", "lint_writes", "known", "columns", "spliced",
                   "carded", "methods", "ruled", "lint2_found", "witness_task", "output_env")
KEYS = ("repair_mode", "repair_pairs", "repair_kept_rewrite", "repair_kept_patch", "repair_skipped_clock",
        "repair_skipped_requests", "repair_a_clean", "repair_b_clean", "repair_b_lost", "repair_b_joined",
        "repair_wait_s")
REST_HEAD = "# Your script so far, cut off at the output limit"     # a cut rewrite's continuation (`rest`)
REWRITE_HEAD = "# The script so far, and its last run"
REWRITE = ("{head}\n\nThe script below was written for this task and run, and its run failed.\n\n{scripts}\n\n"
           "Its last run:\n\n{run}\n\n# Now\n\nWrite work/main.py anew from the task: a whole script that reads the "
           "input files and writes every output file the task names, complete in this reply. Keep what the script "
           "above did right and change what made its run fail, by another method where that one cannot work. {how}"
           "\n{budget}\n")
FIT = ("plain enough to fit this one reply: no comments, docstrings or prints beyond what the outputs need")  # the
#   live smoke's three pairs: each rewrite cut at the output limit
HOW = {"tools": 'Call the write tool with the whole script (write(path="work/main.py", text=<the whole script>)), '
                f"{FIT}; the harness runs it.",
       "fenced": f"Reply with the whole script as a block opened by ```python file=work/main.py, {FIT}; the harness "
                 "runs it."}


# ------------------------------------------------------------------------------------------------- the keep rule


@dataclass(frozen=True)
class Candidate:
    """One side of the pair as the keep rule reads it: its run's exit and traceback, the named output files present
    (and those the task names), a file at the output's root where it names none, and its findings."""
    exit: int | None
    error: bool
    present: int
    named: int
    delivered: bool
    findings: int

    @property
    def clean(self) -> bool:
        """Exit 0, no traceback, every named output file present (a file at the root where the task names none)."""
        return self.exit == 0 and not self.error and (self.present == self.named if self.named else self.delivered)

    def said(self) -> dict:
        """The candidate as the trace's `repair` event says it."""
        return {**dataclasses.asdict(self), "clean": self.clean}


def keep(a: Candidate, b: Candidate | None) -> tuple[str, str]:
    """The keep rule: (`patch`, A's, or `rewrite`, B's; the rule that decided): the clean one where one alone is
    clean; then more named files present; then fewer findings; then A. B with no candidate: A (`lost`)."""
    if b is None:
        return "patch", "lost"
    if a.clean != b.clean:
        return ("rewrite" if b.clean else "patch"), "clean"
    if a.present != b.present:
        return ("rewrite" if b.present > a.present else "patch"), "present"
    if a.findings != b.findings:
        return ("rewrite" if b.findings < a.findings else "patch"), "findings"
    return "patch", "tie"


def candidate(ran: commands.Ran | None, out: Path, named, findings: int) -> Candidate:
    """A side's candidate from its run (None: no run, read as neither exit 0 nor clean) and its output as it stands."""
    out = Path(out)
    return Candidate(None if ran is None else ran.exit, ran is None or bool(ran.error),
                     sum((out / name).is_file() for name in named), len(named), n0.delivered(out), findings)


def lint_rows(ws: commands.Workspace, text: str) -> int:
    """Lint's rows over a script as the loop's write would say them (agent/n11/lint.py `check`), counted alone: the
    keep rule's, which changes no count of the loop's."""
    try:
        if ws.known is None:
            ws.known = lint.known(ws.library, commands.QF_PACKAGE)
        if ws.columns is None:
            ws.columns = lint.inputs_columns(Path(ws.pathmap.data_dir)) if ws.pathmap is not None else {}
        return len(lint.check(text, MAIN, ws.known, ws.columns, frozenset(), {})[:lint.FINDINGS])
    except Exception:  # noqa: BLE001 - a reader's fault counts no finding
        return 0


# ------------------------------------------------------------------------------------------- B's run, apart


def rooted(text: str, out: Path, to: Path) -> str:
    """A script as B's run takes it: every literal naming the output's root (`out` as the path map states it, and the
    instructions' `/app/output` and `/output`), alone or before a `/`, naming `to` instead."""
    roots = sorted({str(out), *paths.OUTPUTS}, key=len, reverse=True)
    pattern = re.compile(r"(?<![\w.~/-])(?:" + "|".join(map(re.escape, roots)) + r")(?=/|[\"'\s),]|$)", re.M)
    return pattern.sub(lambda _: str(to), text)


def mapped(text: str, pairs) -> str:
    """`text` with each of B's paths said as the loop's (`pairs`: (B's, the loop's), the longest first)."""
    for here, there in pairs:
        text = text.replace(here, there)
    return text


def mapped_ran(ran: commands.Ran, pairs) -> commands.Ran:
    """B's run as the loop keeps it: its texts naming the loop's paths."""
    return dataclasses.replace(ran, error=mapped(ran.error, pairs), line=mapped(ran.line, pairs),
                               stdout=mapped(ran.stdout, pairs), digest=mapped(ran.digest, pairs),
                               failures=tuple(mapped(line, pairs) for line in ran.failures),
                               findings=tuple(mapped(line, pairs) for line in ran.findings))


class Beside(commands.Workspace):
    """B's workspace: the loop workspace's settings over a directory of its own, each run's `OUTPUT_DIR` its own output
    whatever the environment names, and each run on the trace under `KIND`."""
    kind = KIND

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = False

    def hold(self, proc) -> None:
        super().hold(proc)
        if self.closed:
            runner.kill_group(proc)

    def kill(self) -> None:
        self.closed = True
        super().kill()

    def write(self, *args, **kwargs):
        with guard.WRITING:
            if self.closed:
                return "write refused: the rewrite is closed"
            return super().write(*args, **kwargs)

    def child_env(self) -> dict[str, str]:
        env = super().child_env()
        env[runner.OUTPUT_ENV] = str(self.out.absolute())
        return env


def beside(ws: commands.Workspace, root: Path) -> Beside:
    """B's workspace under `root` (its work/ and out/), the loop workspace's settings and caches."""
    made = Beside(root, root / "out", ws.task, ws.pathmap, ws.library, ws.named, python=ws.python, stated=ws.stated,
                  shimmed=ws.shimmed, linted=ws.linted, rules=ws.rules, wide=ws.wide, cardfail=ws.cardfail,
                  lint2=ws.lint2, snapshotted=ws.snapshotted, witnessed=ws.witnessed, contracted=ws.contracted,
                  reconciled=ws.reconciled)
    made.out.mkdir(parents=True, exist_ok=True)
    made.carded = dict(ws.carded) if ws.carded is not None else None
    made.known, made.columns, made.params, made.dated = ws.known, ws.columns, ws.params, ws.dated
    return made


def written(reply, turn: int) -> list[commands.Call]:
    """A reply's writes of a `.py` file, read as its native calls where it made any, else as the fenced form."""
    calls = commands.from_tools(reply, turn) if reply.tool_calls else []
    if not any(call.name == "write" for call in calls):
        calls = commands.from_fenced(reply.text or "", turn)
    return [call for call in calls if call.name == "write" and (str(call.arguments.get("path", "")).endswith(".py")
                                                                or call.cut)]


def cut_text(reply, turn: int) -> str:
    """What arrived of a write the output cap cut in the reply's last call: its text, read from the arguments as they
    stand where a native call's do not parse (`commands.cut_arguments`); "" where the reply holds no cut write."""
    cut = [call for call in written(reply, turn) if call.cut]
    if not cut:
        return ""
    return str(cut[-1].arguments.get("text") or commands.cut_arguments(cut[-1].wrote).get("text", ""))


def as_main(text: str, turn: int) -> commands.Call:
    """A write of work/main.py holding `text`, as a native call writes it."""
    arguments = {"path": MAIN, "text": text}
    return commands.Call("write", arguments, f"call_{turn}_1", json.dumps(arguments))


def script_of(reply, mode: str, turn: int) -> commands.Call | None:
    """B's whole script: the reply's last write of a `.py` file neither cut nor malformed, read as its native calls
    where it made any, else as the fenced form; said as a write of work/main.py whatever file it named. None where it
    holds none."""
    writes = [call for call in written(reply, turn) if not call.cut and not call.problem
              and str(call.arguments.get("path", "")).endswith(".py")]
    if not writes:
        return None
    call = writes[-1]
    if call.arguments["path"] != MAIN or mode != "tools":
        arguments = {"path": MAIN, "text": call.arguments["text"]}
        return dataclasses.replace(call, arguments=arguments, raw=json.dumps(arguments), wrote="")
    return call


# ------------------------------------------------------------------------------------------------------ the pair


@dataclass
class Pair:
    """One repair turn's pair: what the loop held before A's turn, and B as it went."""
    turn: int                       # the turn A's reply takes
    mark: int                       # the conversation's length before A's turn
    held: dict                      # the loop's state before it (agent/n11/solve.py `TURN_STATE`)
    files: dict[str, bytes] | None  # the files under work/ before it, None where over `WORK_BYTES`
    root: Path                      # B's directory
    ws: Beside
    client: object
    messages: list[dict]
    keywords: dict                  # B's request's tools, tool choice, thinking and bound
    allowed: int                    # B's requests at most
    seen: set                       # the errors the loop had seen, for B's run's observation
    mode: str
    there: tuple[str, str]          # the loop's work root and its work/, which B's paths are said as
    min_run: float                  # a run is not started with less time than this (the loop's)
    last: commands.Ran | None       # the loop's last run before A's turn: A's where A's turn runs nothing
    rest: object = None             # the loop's reading of a cut script: (its kept head, the ask for the rest), and
    code_of: object = None          # of a continuation's reply: its code (agent/n11/solve.py `rest_asked`, `continuation_code`)
    began: float = field(default_factory=time.time)
    admitted: int = 0               # B's requests counted so far
    reply: object = None
    call: commands.Call | None = None
    ran: commands.Ran | None = None
    said: str = ""                  # B's run's observation, its paths the loop's
    cap: float = 0.0                # B's run's limit
    lost: str = ""                  # why B has no candidate, "" where it has one
    joined: bool = False            # B's script the join of a cut reply and its continuation's rest
    abandoned: bool = False         # the loop stopped waiting: no further request or run of B's
    text: str = ""                  # B's reply as the conversation carries it where B is kept, and whether as a native
    native: bool = False            # call
    thread: threading.Thread | None = None
    workspace: dict = field(default_factory=dict)
    fallback: dict = field(default_factory=dict)
    request_s: float = 0.0

    def paths(self) -> tuple[tuple[str, str], ...]:
        """B's paths and the loop's they are said as: its output, its work/ and its root."""
        root, work = self.there
        return ((str(self.ws.out), str(self.ws.pathmap.out)), (str(self.ws.dir), str(work)),
                (str(self.root), str(root)))


def files_of(directory: Path) -> dict[str, bytes] | None:
    """The regular files under `directory` by their relative paths, None where over `WORK_BYTES` in all."""
    found, size = {}, 0
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            size += len(data)
            if size > WORK_BYTES:
                return None
            found[path.relative_to(directory).as_posix()] = data
    return found


def seeded(directory: Path, files: dict[str, bytes] | None) -> None:
    """B's work/ given the loop's files but work/main.py, so a rewrite importing a module the failed run's program
    holds runs as it would in the loop's work/ (`files`: `files_of`'s, nothing where None)."""
    for name, data in (files or {}).items():
        if name != "main.py":
            here = Path(directory) / name
            here.parent.mkdir(parents=True, exist_ok=True)
            here.write_bytes(data)


def put_back(directory: Path, files: dict[str, bytes]) -> None:
    """The files under `directory` as `files` held them: those added since removed, those changed rewritten."""
    directory = Path(directory)
    for path in sorted(directory.rglob("*"), reverse=True):
        if (path.is_file() or path.is_symlink()) and path.relative_to(directory).as_posix() not in files:
            path.unlink(missing_ok=True)
    for name, data in files.items():
        here = directory / name
        if not here.is_file() or here.read_bytes() != data:
            here.parent.mkdir(parents=True, exist_ok=True)
            here.write_bytes(data)


def moved(source: Path, out: Path) -> int:
    """`source`'s entries moved into `out`, what `out` held first removed (agent/n11/drafts.py `cleared`): the entries
    moved."""
    from n11 import drafts
    out.mkdir(parents=True, exist_ok=True)
    drafts.cleared(out)
    count = 0
    for entry in sorted(Path(source).iterdir()):
        shutil.move(str(entry), str(out / entry.name))
        count += 1
    return count


class Repair:
    """The unit's pairs: the setting, B's room and the counts the keys carry."""

    def __init__(self, setting: str, source: str, form: str, base: int) -> None:
        self.setting, self.source, self.form = setting, source, form
        self.base = base                # the loop client's cap at the unit's start
        self.spent = 0                  # B's requests counted over the unit
        self.pending: Pair | None = None
        self.killed = False
        self.number = 0                 # the pairs sent, naming B's directories
        self.counts = dict.fromkeys(KEYS[1:-1], 0)
        self.wait_s = 0.0

    @property
    def armed(self) -> bool:
        return self.setting == "pair" and self.form in FORMS

    def room(self) -> int:
        """B's requests left: `ROOM` a unit, never past the House's cap less the loop's."""
        return max(0, min(ROOM, model.REQUESTS_PER_UNIT - self.base) - self.spent)

    def raised(self) -> int:
        """The requests the loop's client's cap holds above its base for B: those B spent and a pair's room out."""
        return self.spent + (self.pending.allowed if self.pending is not None else 0)

    def unspent(self) -> int:
        """The requests the loop's cap holds for B that B has not had counted: off the loop's requests left."""
        pair = self.pending
        return max(0, pair.allowed - pair.admitted) if pair is not None and pair.allowed else 0

    def kill(self) -> None:
        """B's running script's process group, killed (the watchdog's); nothing moves after it."""
        self.killed = True
        if self.pending is not None:
            self.pending.abandoned = True
            self.pending.ws.kill()
            shutil.rmtree(self.pending.root, ignore_errors=True)

    def discard(self, primary) -> None:
        """Close an outstanding B before the unit closes; the caller holds guard.WRITING."""
        pair = self.pending
        if pair is not None:
            pair.abandoned = True
            pair.ws.kill()
            self.close(pair, primary)
            shutil.rmtree(pair.root, ignore_errors=True)

    def keys(self) -> dict:
        """The summary line's keys (`KEYS`): the setting, the counts, the seconds waited on B (`-` where no pair was
        sent); every key `-` outside `FORMS`."""
        if self.form not in FORMS:
            return {key: "-" for key in KEYS}
        return {"repair_mode": self.setting, **self.counts,
                "repair_wait_s": round(self.wait_s, 1) if self.counts["repair_pairs"] else "-"}

    # ----------------------------------------------------------------------------------------------- the send

    def counted(self, pair: Pair, client, primary) -> None:
        """B's client's admissions counted on the pair (under `LOCK`, as the unit's count is), a turned-away attempt
        uncounted as the unit's count uncounts it."""
        admit, turned, uncharged = client._admit, client._turned_away, client._rate_uncharged

        def admitted():
            with LOCK:
                if pair.abandoned or self.killed or pair.admitted >= pair.allowed:
                    raise model.RequestCapReached("the rewrite's requests are spent or closed")
                timeout = admit()
                pair.admitted += 1
            return timeout

        def uncounted(undo):
            def undone():
                undo()
                with LOCK:
                    pair.admitted -= 1
                    if self.pending is not pair:
                        self.spent -= 1
                        primary.cap -= 1
            return undone
        client._admit, client._turned_away, client._rate_uncharged = admitted, uncounted(turned), uncounted(uncharged)

    def send(self, pair: Pair, primary, run_cap) -> None:
        """B sent on its thread: its room raised on the loop's client, its client the loop's sibling capped at the
        unit's count now, A's admission and B's room."""
        self.pending = pair
        with LOCK:
            primary.cap += pair.allowed
            pair.client = primary.sibling(primary.usage.requests + 1 + pair.allowed)
        pair.client.kind = KIND
        self.counted(pair, pair.client, primary)
        self.counts["repair_pairs"] += 1
        pair.thread = threading.Thread(target=drawn, args=(pair, run_cap), name="repair-b", daemon=True)
        pair.thread.start()

    def close(self, pair: Pair, primary) -> None:
        """B's room returned to what B spent: the loop's cap its base and B's requests counted."""
        with LOCK:
            self.spent += pair.admitted
            primary.cap = self.base + self.spent
        self.pending = None


def rewrite_turn(framing: str, scripts: list[tuple[str, str]], run: str, mode: str, budget: str) -> str:
    """B's user turn: the task as the loop's first message frames it, then the failed run's scripts each under its
    name, its observation and the ask (`REWRITE`, `HOW`)."""
    shown = "\n\n".join(f"```python file={name}\n{text.rstrip()}\n```" for name, text in scripts)
    return f"{framing.rstrip()}\n\n" + REWRITE.format(head=REWRITE_HEAD, scripts=shown, run=run.rstrip(),
                                                      how=HOW[mode], budget=budget)


def rest(pair: Pair, until: float) -> commands.Call | None:
    """B's reply the output cap cut (a whole script too long for one reply): where B's room holds a request, its head
    kept to its last complete top-level statement and the rest asked as the loop's request 1 asks it (`Pair.rest`:
    B's own request with the head under `REST_HEAD` and solve.py's `CONTINUE_ASK`, no tools, by `until`), the rest's
    code joined under the head (`runner.stitch`); the join a write of work/main.py where the rest was not cut again and
    the join compiles, else None."""
    text = cut_text(pair.reply, pair.turn)
    if not text or pair.rest is None or pair.abandoned or pair.admitted >= pair.allowed or \
            until - time.time() < model.MIN_CALL_S:
        return None
    kept, ask = pair.rest(text)
    if not kept.strip():
        return None
    system, user = pair.messages
    keywords = {key: value for key, value in pair.keywords.items() if key not in ("tools", "tool_choice", "bound")}
    more = pair.client.converse([system, {"role": "user", "content": f"{user['content'].rstrip()}\n\n{ask}"}],
                                bound=until - time.time(), **keywords)
    code = pair.code_of(more.text)
    if not code.strip() or more.finish_reason == "length":
        return None
    joined, _ = runner.stitch(kept, f"```python\n{code}\n```\n")
    return as_main(joined, pair.turn) if runner.compiles(joined) else None


def drawn(pair: Pair, run_cap) -> None:
    """B on its own thread: the request, its whole script (a cut one continued, `rest`) and the script's run in B's
    directory under `run_cap()`'s limit; any failure loses B (`lost`), never the unit."""
    try:
        until = time.time() + pair.keywords.get("bound", 0.0)
        try:
            began = time.time()
            pair.reply = pair.client.converse(pair.messages, **pair.keywords)
            pair.call = script_of(pair.reply, pair.mode, pair.turn)
            if pair.call is None:
                pair.call = rest(pair, until)
                pair.joined = pair.call is not None
            pair.request_s = time.time() - began
        except model.ModelError as exc:
            pair.lost = f"no reply ({type(exc).__name__})"
            return
        if pair.call is None or pair.abandoned:
            pair.lost = pair.lost or "no whole script in the reply" + (
                ", nor its rest" if cut_text(pair.reply, pair.turn) else "")
            return
        pair.native = pair.mode == "tools" and (pair.joined or bool(pair.reply.tool_calls))
        pair.text = "" if pair.joined and pair.native else \
            f"```python file={MAIN}\n{pair.call.arguments['text'].rstrip()}\n```\n" if pair.joined else \
            pair.reply.text or ""
        ws = pair.ws
        ws.write(MAIN, rooted(pair.call.arguments["text"], Path(ws.pathmap.out), ws.out))
        pair.cap = run_cap()
        if pair.cap < pair.min_run:
            pair.lost = f"no time for its run ({pair.cap:.0f} s)"
            return
        said, ran = ws.run(MAIN, pair.cap, set(pair.seen), lambda done: "")
        if ran is None:
            pair.lost = "its run was refused"
            return
        pair.said, pair.ran = mapped(said, pair.paths()), mapped_ran(ran, pair.paths())
    except Exception as exc:  # noqa: BLE001 - B's failure costs B, never the unit
        pair.lost = f"failed ({type(exc).__name__})"
    finally:
        if pair.abandoned or pair.ws.closed:
            shutil.rmtree(pair.root, ignore_errors=True)
