"""The failure snapshot in the run's answer (lane snapshot1; the mid-loop trace read runs/lanes/codex-traces-midloop.md,
"Failure snapshot"; the owner's in49): `AGENTHON_N11_SNAPSHOT` or agent/n11/SNAPSHOT, `off` (today's) or `on`, armed
in forms `loop`, `review` and `sketch` (agent/n11/solve.py `snapshot_armed`).

Under `on` each run's script starts with agent/n11/preload.py as its `sitecustomize` (agent/n11/shim.py `preloaded`,
whether the shim is on or not), its spec here naming the House's directory and a side file (`environ`): at an
uncaught exception the preload's hook writes the state at the failure there (preload.py `snapshot`), never to the
run's stdout, which the House reads as the run's output. After a run that raised (`due`: a non-zero exit, a traceback,
an error that is no SyntaxError's kind), `shown` reads the side file (`read`) into one paragraph (`said`): the error,
the innermost frame in a file under work/ with the source around its line, each local the failing statement reads and
the frame's others assigned nearest before it, by its type and the parts read of it (`words`), and a library call's
actual arguments; within `TOKENS` in the model's tokens (agent/n11/measure.py; `BYTES` where the tokenizer does not
load), the longest fields cut first (`fitted`), the whole dropped where even at `FLOOR` a field it does not fit. A run
whose hook ran and left only its mark keeps today's answer and counts as dropped if it exited non-zero; a stopped
run is never due and counts neither. A run with no side file had no uncaught exception reach the hook (a script that caught its
own and printed the traceback) and is neither. The words name each field; no count is stated as an instruction (the owner's in281).
The run's answer carries the paragraph after the traceback's tail and the output's listing and before the checks,
today's body cut as today (agent/n11/commands.py `Workspace.observe`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from common import runner
from n11 import measure, shim

ENV, FILE = "AGENTHON_N11_SNAPSHOT", "SNAPSHOT"     # the desk's switch, or the image's one-word file
SETTINGS = ("off", "on")
FORMS = ("loop", "review", "sketch")    # armed in the loop's forms (form `state` keeps its own run text)
KEYS = ("snapshot_mode", "snapshot_shown", "snapshot_dropped", "snapshot_tokens")
SPEC_ENV = "N11_SNAPSHOT"               # the spec's path, as preload.py reads it (`SNAPSHOT_ENV`)
SPEC, SIDE = "snapshot-spec.json", "snapshot.json"  # beside the preload, under the workspace's shim/
TOKENS = 600                            # the paragraph's budget in the model's tokens: the harness's bound, never said
BYTES = TOKENS                         # the byte-level tokenizer cannot emit more tokens than UTF-8 bytes;
#                                         a sampled bytes/token ratio cannot bound arbitrary names or values
FLOOR, FLOOR_BYTES = 16, 40             # a field is never cut below this; past it, the snapshot is dropped whole
PASSES = 6                              # the cuts measured again, at most
CUT = " ..."                            # a cut field's end
HEAD = "the state when it raised, read from the run's own objects:"
SYNTAX = ("SyntaxError", "IndentationError", "TabError")    # left to the write's check: no snapshot
READ = {"statement": "read by the failing statement", "frame": "assigned before it"}


@dataclass(frozen=True)
class Shown:
    """A failed run's snapshot as said: the paragraph and its measure in `measure.UNIT`, or why none was where one was
    due (`hook`: only the hook's mark written; `budget`: no fit), all empty where none was due."""
    text: str = ""
    size: int = 0
    dropped: str = ""


def environ(root: Path, env: dict[str, str], work: Path) -> dict[str, str]:
    """The run's environment with the snapshot's hook (`env`, the child's, otherwise as it is): preload.py loaded first
    (shim.preloaded), the spec beside it naming `work` and the side file, the side file of the last run removed."""
    here, env = shim.preloaded(root, env)
    for stale in (here / SIDE, here / f"{SIDE}.part"):
        stale.unlink(missing_ok=True)
    (here / SPEC).write_text(json.dumps({"work": str(work), "side": str(here / SIDE)}), encoding="utf-8")
    env[SPEC_ENV] = str(here / SPEC)
    return env


def read(root: Path) -> dict | None:
    """What the run's hook wrote, None where it wrote nothing readable."""
    try:
        found = json.loads((Path(root) / shim.DIR / SIDE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def due(exit_: int | None, error: str, stderr: str) -> bool:
    """Whether a run raised as the hook takes it: a non-zero exit, a traceback, an error of no SyntaxError's kind."""
    kind = error.split(":", 1)[0].strip().rsplit(".", 1)[-1]
    return exit_ not in (0, None) and bool(error) and runner.TRACEBACK in stderr and kind not in SYNTAX


def listed(items, more: bool = False) -> str:
    return ", ".join(str(item) for item in items) + (", ..." if more else "")


def index_words(index: dict) -> str:
    words = f"index {index.get('type', '?')}" + (f" ({index['dtype']})" if index.get("dtype") else "")
    if "first" in index:
        words += f", first {listed(index['first'])}"
    if "last" in index:
        words += f", last {listed(index['last'])}"
    return words


def words(found: dict) -> str:
    """One value as the snapshot read it, its parts named: its type, then its shape or length, dtype, name, columns
    or keys, index, first row or labels, missing and infinite cells, and a few values."""
    kind = str(found.get("type", "?"))
    parts = [kind]
    shape = found.get("shape")
    if isinstance(shape, list):
        parts.append(f"{shape[0]} rows x {shape[1]} columns" if kind == "DataFrame" and len(shape) == 2 else
                     f"shape ({', '.join(map(str, shape))}{',' if len(shape) == 1 else ''})")
    if "length" in found:
        parts.append(f"length {found['length']}")
    if found.get("dtype"):
        parts.append(f"dtype {found['dtype']}")
    if "named" in found:
        parts.append(f"named {found['named']}")
    if "columns" in found:
        more = isinstance(shape, list) and len(shape) == 2 and shape[1] > len(found["columns"])
        parts.append(f"columns {listed(found['columns'], more)}")
    if "keys" in found:
        parts.append(f"keys {listed(found['keys'], found.get('length', 0) > len(found['keys']))}")
    if isinstance(found.get("index"), dict):
        parts.append(index_words(found["index"]))
    if "first" in found:
        parts.append(f"{'first row' if kind == 'DataFrame' else 'first'} {listed(found['first'])}")
    if "last" in found:
        parts.append(f"last {listed(found['last'])}")
    if "missing" in found:
        parts.append(f"missing {found['missing']}")
    if "infinite" in found:
        parts.append(f"infinite {found['infinite']}")
    if found.get("values"):
        parts.append(f"values {listed(found['values'], found.get('length', 0) > len(found['values']))}")
    if "value" in found:
        parts.append(f"value {found['value']}")
    return "; ".join(parts)


def fields(found: dict) -> list[str]:
    """The paragraph's lines after its head, each a field the budget may cut: the error, where, the source lines, one
    line a local, the library call and one line an argument."""
    error = found.get("error") or {}
    message = " ".join(str(error.get("message", "")).split())
    lines = [f"- error: {error.get('type', '?')}" + (f": {message}" if message else "")]
    frame = found.get("frame") or {}
    lines.append(f"- where: {frame.get('file', '?')}, line {frame.get('line', '?')}, in {frame.get('function', '?')}")
    for number, text in frame.get("source") or ():
        lines.append(f"  {'>' if number == frame.get('line') else ' '} {number} | {text}")
    for name in found.get("names") or ():
        lines.append(f"- {name.get('name', '?')}, {READ.get(name.get('read'), READ['frame'])}: {words(name)}")
    call = found.get("call")
    if isinstance(call, dict):
        lines.append(f"- the library call {call.get('function', '?')}, its arguments as passed:")
        for argument in call.get("arguments") or ():
            lines.append(f"  - {argument.get('name', '?')}: " + (f"the same object as {argument['same']}" if
                                                                  argument.get("same") else words(argument)))
    return lines


def fitted(lines: list[str], room: int) -> list[str] | None:
    """`lines` within `room` joined (in `measure.UNIT`): the longest cut first, every line longer than one level cut to
    it, the level the highest that removes what is over and never below `FLOOR`, each cut ending `CUT`, measured
    again (a cut's tokens need not add up) a few times; None where even at `FLOOR` they do not fit."""
    lines, floor = list(lines), measure.bound(FLOOR, FLOOR_BYTES)
    for _ in range(PASSES):
        over = measure.size("\n".join(lines)) - room
        if over <= 0:
            return lines
        sizes = [measure.size(line) for line in lines]
        if sum(max(0, size - floor) for size in sizes) < over:
            return None
        level = max(sizes)
        while sum(max(0, size - level) for size in sizes) < over:
            level -= 1
        lines = [measure.prefix(line, level - measure.size(CUT)).rstrip() + CUT if size > level else line
                 for line, size in zip(lines, sizes)]
    return lines if measure.size("\n".join(lines)) <= room else None


def said(found: dict) -> str | None:
    """The paragraph: `HEAD`, then the fields within `TOKENS` (`BYTES` where the tokenizer does not load); None where
    they do not fit."""
    room = measure.bound(TOKENS, BYTES) - measure.size(HEAD) - 1
    lines = fitted(fields(found), room)
    return None if lines is None else "\n".join([HEAD, *lines])


def shown(found: dict | None, exit_: int | None, error: str, stderr: str) -> Shown:
    """A run's snapshot as its answer carries it (`due`'s runs alone): the paragraph where the hook wrote a frame and
    it fits; dropped where the hook ran and left its mark alone (`hook`: it failed after the mark) or
    the paragraph does not fit (`budget`); none where no uncaught exception reached the hook (no side file: the script
    printed its own traceback), it found no frame under work/ or left a SyntaxError."""
    if not due(exit_, error, stderr) or found is None:
        return Shown()
    if not isinstance(found.get("frame"), dict):
        return Shown(dropped="hook") if "raised" in found else Shown()
    try:
        text = said(found)
    except Exception:  # noqa: BLE001 - a side file of another shape is dropped; the run's answer is today's
        return Shown(dropped="hook")
    return Shown(dropped="budget") if text is None else Shown(text, measure.size(text))
