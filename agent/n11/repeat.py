"""Lane repeat1 (the Codex read runs/lanes/codex-errors-time.md rank 2; docs/research/levers-1015.md; the owner's in27):
the same assembled program failing the same way again, recognised and answered with a changed-repair ask.

In 7 of the 30 no-output and failing-run traces the Codex read went through turn by turn, the same program came back
with the same failure: re-sent unchanged (the write refused, today's `UNCHANGED`) or rerun, or written back after a
variant failed otherwise (E180's CTA script alternated two versions; E181's spread pricer alternated unchanged writes
with reruns). The re-send rule (`RESEND`) and the repeat refusal see one write at a time; this module sees the program.

The state, kept under either setting by `Ledger`: after each run the House is shown, the assembled program as run
(`program`: every `.py` file under work/, by content) and the failure's signature (`signature`: the exit, then the
exception's line with its numbers and paths folded, or, where the run exited 0 with no traceback, the first failing
check's line folded the same way; none for a clean run). A write refused as unchanged re-sends the program as it
stands: where that program's last run failed, the re-send carries its signature. A recurrence is a program met before
in the unit with the same signature, run or re-sent; `Ledger` counts them in the unit, every state together, so a
cycle between two failing versions counts as surely as one version sent again.

This module is the bookkeeping, shared by the agent (agent/n11/solve.py `repeat_*`) and lane repeat1's replay over the
records. The words the model reads are solve.py's (`REPEAT_ASK`), and none states a count.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

ENV, FILE = "AGENTHON_N11_REPEAT", "REPEAT"     # the desk's variable, an image's one-word file under agent/n11/
SETTINGS = ("off", "change")                     # `off`, today's: the state kept and counted, nothing said
FORMS = ("loop", "review", "sketch")             # the loop's forms (form `state` answers its own re-sends)
KEYS = ("repeat_mode", "repeat_states", "repeat_asked", "repeat_fallback", "repeat_floor")
ASK_AT = 2                  # the unit's recurrence from which each turn holding one closes on the changed-repair ask
FALLBACK_AT = 3             # and the one at which the one-shot takes the unit where lane fallback1 arms it
FOLDED_CHARS = 300          # a signature's folded line, at most

ADDRESS = re.compile(r"0x[0-9a-fA-F]+")                             # an object's address, which differs run to run
PATH = re.compile(r"(?:[A-Za-z]:)?(?:[\w.~-]*[\\/])+[\w.-]*")       # anything holding a slash, read as a path
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")     # a number not inside a name


def fold(text: str) -> str:
    """A failure's line with what varies between runs of one program folded away: addresses, paths and numbers, and
    its whitespace; at most `FOLDED_CHARS`."""
    text = ADDRESS.sub("#", text or "")
    text = PATH.sub("<path>", text)
    text = NUMBER.sub("#", text)
    return " ".join(text.split())[:FOLDED_CHARS]


def signature(exit_: int | None, error: str, failures=()) -> str | None:
    """A run's failure as a recurrence knows it: the exit (`limit` for one stopped at its limit), and the exception's
    line folded where it raised, or the first failing check's line folded where it exited 0 with no traceback; None for
    a clean run (exit 0, no traceback, no check failing), which no recurrence counts."""
    if exit_ == 0 and not error and not failures:
        return None
    head = "limit" if exit_ is None else f"exit {exit_}"
    if error:
        return f"{head}: {fold(error)}"
    if exit_ == 0 and failures:
        return f"{head}: check {fold(failures[0])}"
    return head


def composed(files: dict[str, str]) -> str:
    """The program's digest from each file's path under work/ and its content's sha256."""
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(f"{name}\0{files[name]}\n".encode("utf-8"))
    return digest.hexdigest()


def program(directory: Path) -> str:
    """The assembled program under `directory` (work/): every `.py` file there, its path and its bytes; "" where the
    directory cannot be read, which no recurrence matches."""
    try:
        files = {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in Path(directory).rglob("*.py") if path.is_file() and "__pycache__" not in path.parts}
    except OSError:
        return ""
    return composed(files)


class Ledger:
    """The unit's program states: each (program, signature) by the times it was met (`met`), each program's last run's
    signature (`last`, what a re-send carries), and the unit's recurrences (`states`)."""

    def __init__(self) -> None:
        self.met: Counter = Counter()
        self.last: dict[str, str | None] = {}
        self.states = 0

    def ran(self, program_: str, signature_: str | None) -> int:
        """A run of `program_` that failed as `signature_` (None: clean): the unit's recurrence it is, counted from 1,
        or 0 where it is none (a clean run, a program unread, a state not met before)."""
        if not program_:
            return 0
        self.last[program_] = signature_
        return self.met_again(program_, signature_)

    def resent(self, program_: str) -> int:
        """A write refused as unchanged, the program standing as `program_`: a recurrence where its last run failed
        (that run's signature carried), else 0 (never run, or its last run clean)."""
        return self.met_again(program_, self.last.get(program_)) if program_ else 0

    def met_again(self, program_: str, signature_: str | None) -> int:
        """The state met once more: the unit's recurrence where it was met before with a failure, else 0."""
        if signature_ is None:
            return 0
        self.met[(program_, signature_)] += 1
        if self.met[(program_, signature_)] < 2:
            return 0
        self.states += 1
        return self.states
