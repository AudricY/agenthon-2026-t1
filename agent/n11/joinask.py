"""Lane joinask1 (E168 ROW 421, SPINE 414; the owner's in13): the ask after a failed continuation changes.

Under `CUT=continue` a long script arrives in parts: its head kept, the rest asked and stitched under it
(agent/n11/solve.py `hold`, `resumed`, `joined`). At temperature 0 with the seed pinned the same ask draws the same
reply, so a rest that did not join, a rest cut again or the same text sent again, asked for in the same words, comes
back as it was: E168's nine long units spent most of their requests so. Under `JOINASK=change` the ask after such a
failure is never the one before it, a junction gets at most `ASKS` asks (the cut's own and two changed ones), and the
head is never dropped for a rest's fault; the third failure runs what stands instead (`solve.N11.head_run`).

This module is the bookkeeping, shared by the agent and the offline replay (scripts/replay.py `--joinask`): the
switch's words, the failures' kinds, each file's open junction (`Ledger`) and the shape of the next ask, and the
names an ask gives a head (`last_statement`, `cut_line`). The asks' words are solve.py's (`JOIN_STATEMENT`,
`JOIN_MODULE`, `JOIN_DEFS`, `JOIN_RAN`). No text here is sent to the model, and none states a count.
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter

from common import runner

ENV, FILE = "AGENTHON_N11_JOINASK", "JOINASK"     # the desk's variable, an image's one-word file under agent/n11/
SETTINGS = ("same", "change")                    # `same`, today's: every ask as before
FORMS = ("loop", "review", "sketch")             # armed in the loop's forms, under `CUT=continue` alone
KINDS = ("junction", "cut_again", "refused")     # a rest that does not compile with the head; a rest cut again; the
#                                                  same text again (a join refused as saved, or a cut head held before)
SHAPES = ("statement", "module")                 # the rest from a named statement; the head saved as a module and
#                                                  the file written anew importing it
PREFERRED = {"junction": "statement", "cut_again": "module", "refused": "module"}
ASKS = 3                    # asks one junction gets at most: the cut's own and two changed; the third failure runs
MODULE_SHARE = 0.5          # the module shape where the head's complete top-level part holds this share of the head
KEYS = ("joinask_mode", "joinask_changed", "joinask_module_asked", "joinask_head_run", "joinask_defs_asked")
SHOWN = 120                 # characters of a statement's first line an ask quotes


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class Ledger:
    """Each file's open junction: the failures since its head was first held (`failures`), the last changed ask's
    shape (`last`), and every head held in the unit by digest (`heads`), so a head cut at the same place again reads
    as the same text. A junction closes when the file is saved whole and compiling (`close`). The counts the summary
    line carries: the changed asks by kind, the module asks, the heads run and the definitions asked."""

    def __init__(self) -> None:
        self.failures: Counter = Counter()
        self.last: dict[str, str] = {}
        self.heads: dict[str, set[str]] = {}
        self.changed: Counter = Counter()
        self.modules = 0
        self.head_runs = 0
        self.defs_asked = 0

    def open(self, named: str) -> bool:
        """Whether `named` has a junction a failure was counted on and no whole compiling save since."""
        return self.failures[named] > 0

    def close(self, named: str) -> None:
        self.failures.pop(named, None)
        self.last.pop(named, None)

    def seen(self, named: str, head: str) -> bool:
        """Whether `head` was held for `named` before in the unit; it is recorded either way."""
        held = self.heads.setdefault(named, set())
        sha = digest(head)
        found = sha in held
        held.add(sha)
        return found

    def failed(self, named: str, kind: str, module_fits: bool) -> str | None:
        """A failure of `kind` on `named`'s junction counted, and the next ask's shape: `PREFERRED`'s for the kind,
        the statement where the module does not fit, the other shape where that one was the last ask's and the other
        fits; None where the junction has had its asks (the head runs instead)."""
        if kind not in KINDS:
            raise ValueError(kind)
        self.failures[named] += 1
        if self.failures[named] >= ASKS:
            return None
        shape = PREFERRED[kind] if module_fits else "statement"
        if shape == self.last.get(named):
            other = "statement" if shape == "module" else "module"
            if other == "statement" or module_fits:
                shape = other
        self.last[named] = shape
        self.changed[kind] += 1
        self.modules += shape == "module"
        return shape

    def changed_said(self) -> str:
        """`joinask_changed`'s value: each kind with a changed ask, `kind:n` in `KINDS`' order, or `-`."""
        return ",".join(f"{kind}:{self.changed[kind]}" for kind in KINDS if self.changed[kind]) or "-"


DEFINES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
OPENER = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)")


def first_line(text: str) -> str:
    """`text`'s first line that is not blank or a comment, stripped, at most `SHOWN` characters."""
    for line in text.split("\n"):
        if line.strip() and not line.lstrip().startswith("#"):
            line = line.strip()
            return line if len(line) <= SHOWN else line[:SHOWN].rstrip() + " ..."
    return ""


def enclosing(head: str) -> str:
    """The top-level definition `head` ends inside (its last top-level line opening a `def` or `class` whose body
    the head's last lines are), or ""."""
    lines = [line for line in head.rstrip("\n").split("\n") if line.strip() and not line.lstrip().startswith("#")]
    if not lines or not lines[-1][0].isspace():
        return ""
    for line in reversed(lines):
        if not line[0].isspace():
            found = OPENER.match(line)
            return found.group(1) if found else ""
    return ""


def last_statement(head: str) -> str:
    """The head's last statement as an ask names it: the first line of the statement that starts last (the innermost
    where the head ends inside a block), else the head's last line that is not blank or a comment."""
    try:
        tree = ast.parse(head)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        tree = None
    lines = head.split("\n")
    starts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt)] if tree is not None else []
    if starts:
        return first_line(lines[max(starts) - 1])
    for line in reversed(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            return first_line(line)
    return ""


def last_top(text: str) -> str:
    """The first line of `text`'s last top-level statement (a definition's own line), or "" where none parses."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return ""
    return first_line(text.split("\n")[tree.body[-1].lineno - 1]) if tree.body else ""


def cut_line(text: str, head: str) -> str:
    """The first line of what a cut write held past its kept `head` (the statement the cut fell in, dropped where the
    head was held at a statement boundary), or "" where the head is the whole of it."""
    rest = text[len(head):] if text.startswith(head) else ""
    return first_line(rest)


def module_part(head: str) -> str:
    """What of a kept head a module of its own can hold: its longest compiling prefix (`runner.parsing_prefix`), less
    the top-level statement it ends inside (its last line indented: the cut fell in a function or a block), or ""."""
    prefix = runner.parsing_prefix(head)
    lines = prefix.rstrip("\n").split("\n") if prefix else []
    last = next((line for line in reversed(lines) if line.strip() and not line.lstrip().startswith("#")), "")
    if not last or not last[0].isspace():
        return prefix
    node = ast.parse(prefix).body[-1]
    start = min([node.lineno, *(each.lineno for each in getattr(node, "decorator_list", ()))])
    kept = "\n".join(lines[:start - 1]).rstrip("\n")
    return kept + "\n" if kept.strip() else ""


def module_fits(module: str, head: str) -> bool:
    """Whether the head's complete top-level part (`module`) is worth a module of its own: a definition or a statement
    beyond imports, and `MODULE_SHARE` of the head at least."""
    try:
        tree = ast.parse(module) if module.strip() else None
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    if tree is None or not any(not isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body):
        return False
    return len(module.encode()) >= MODULE_SHARE * len(head.encode())


def module_name(named: str, taken) -> str:
    """The module a head of `named` is saved as: `<stem>_kept.py` beside it, the first of `<stem>_kept2.py`, ...
    where `taken(path)` says a path is used."""
    folder, _, base = named.rpartition("/")
    stem = base[:-3] if base.endswith(".py") else base
    for number in range(1, 100):
        path = f"{folder}/{stem}_kept{number if number > 1 else ''}.py" if folder else \
            f"{stem}_kept{number if number > 1 else ''}.py"
        if not taken(path):
            return path
    return f"{folder}/{stem}_kept_last.py" if folder else f"{stem}_kept_last.py"
