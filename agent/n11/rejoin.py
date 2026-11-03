"""The continuation's join repaired on the harness side (lane join2 of 10-16; the cut trace read's `join` form,
runs/lanes/codex-traces-cut.md; docs/research/course-1016.md; the owner's in49): `AGENTHON_N11_JOIN=today|repair`,
`today` by default, armed in forms `loop`, `review` and `sketch`. Text alone: nothing here runs a script, reads a file
or changes a word the model reads. agent/n11/solve.py calls it where a rest is stitched under a held head (`resumed`)
and where a write would replace a saved file (`written`); scripts/join_replay.py calls it over the records.

Under `repair`, the repairs the replay vindicated (runs/lanes/join2-report.md):
- (a) the prefix kept (`repair`, kind `prefix`). Today's head of a cut write ends at the last statement it finished
  (solve.py `boundary_head`), so where the cut fell inside a multiline statement (a dict, a list, a call) the
  statement's received lines are dropped, and a rest that completes the statement does not compile. The repair keeps
  those lines (`tail`) where the rest goes on from the cut (`resumes`: it re-sends the tail from one of its lines on,
  or none of it, then opens on the line the cut fell in, whose received part is `partial`; or the cut fell at a
  line's start), with exactly one alignment, and appends the rest verbatim under the lines it does not re-send
  (`PREFIX`). Overlap is resolved once by `resumes`; a second overlap reading could delete repeated entries or
  string lines. A rest that begins elsewhere left the cut line out (the model's own copy of its reply holds it, so
  it never learns it is gone): kept, the statement could compile without that line, so no reading is made.
- (b) validated: the prefix's reading is made only where today's join fails, compiling (`sound` under a rest cut
  again: no error above its end), and taken only where it compiles (is sound); else today's join stands and is
  called failed as today.
- (c) a restart (kind `restart`): a rest whose first statements repeat the head's opening (`restarted`: at least two
  of its first `OPENING`, docstrings aside), which today read as a piece (today's join is not the rest alone: a rest
  repeating the head line for line is read past it already), replaces the head where it compiles (or is sound, cut
  again).
- (d) staged (`staged`): a write over a saved `.py` file that compiles, shorter than it, whose text cannot run in its
  place (`alone`: it does not compile; or its top level reads a name only the saved file binds; or it holds nothing
  to run while the saved file does) is saved beside it as `<name>` + `STAGED`, never over it.
"""
from __future__ import annotations

import ast
import builtins
import codeop
import warnings
from dataclasses import dataclass

from common import runner

ENV = "AGENTHON_N11_JOIN"
FILE = "JOIN"
SETTINGS = ("today", "repair")
FORMS = ("loop", "review", "sketch")
KEYS = ("join_mode", "join_prefix_kept", "join_validated", "join_restarts", "join_staged")
STAGED = ".staged"          # a fragment's file beside the saved one it would have replaced, work/<name>.py.staged
OPENING = 3                 # the head's first statements a restart repeats, at most (two at the least)
PREFIX = "prefix kept"      # how a repaired join is named: the rest under the cut statement's received lines
RESTART = "read as a restart"           # the rest in the head's place
TODAY_RESTARTS = ("restarted", "written again whole")   # today's readings of a rest that starts again
MODULE_NAMES = frozenset({"__name__", "__file__", "__doc__", "__builtins__", "__spec__", "__loader__", "__package__"})
PLAIN = frozenset(dir(builtins)) | MODULE_NAMES     # names a script reads that nothing in it need bind
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


@dataclass(frozen=True)
class Repair:
    """A join the repair read: its text (under a rest cut again, before the rest's cut line), how it was read and
    its kind (`prefix` or `restart`)."""
    joined: str
    how: str
    kind: str


def sound(text: str) -> bool:
    """`text` compiles, or fails only for its end, a statement left open there that more lines could finish."""
    if runner.compiles(text):
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return codeop.compile_command(text, "<join>", "exec") is None
    except (SyntaxError, ValueError, OverflowError, MemoryError, RecursionError):
        return False


def tail_of(complete: str, head: str) -> str:
    """The received lines of the statement a cut fell in: what `complete` (the cut write's complete lines) holds past
    `head` (today's head, ended at a statement); "" where the head is them all."""
    return complete[len(head):] if complete.startswith(head) else ""


def resumes(tail: str, partial: str, rest: str) -> int | None:
    """How many of the tail's lines (from its first, blank ones among them) stay before a rest that goes on from the
    cut: the rest re-sends the tail's non-blank lines from one of them to its end (or none of them), whitespace aside,
    then opens on the line the cut fell in (its next line starts with the received part, `partial`), or the cut fell
    at a line's start (`partial` blank). None where the rest begins elsewhere or more than one alignment fits: a
    repeated entry may be an overlap or the next entry, and compiling cannot distinguish them."""
    lines = tail.splitlines(keepends=True)
    kept = [k for k, line in enumerate(lines) if line.strip()]
    said = [line.strip() for line in rest.splitlines() if line.strip()]
    cut = partial.strip()
    matches = []
    for start in range(len(kept) + 1):
        again = [lines[k].strip() for k in kept[start:]]
        after = said[len(again)] if len(said) > len(again) else ""
        if said[:len(again)] == again and (not cut or after.startswith(cut)):
            matches.append(kept[start] if start < len(kept) else len(lines))
    return matches[0] if len(matches) == 1 else None


def same(one: str, other: str) -> bool:
    """Two texts hold the same lines but for blank ones and trailing whitespace (a join today reads past a head the
    rest repeats line for line drops the blank lines between them)."""
    def lines(text: str) -> list[str]:
        return [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines(one) == lines(other)


def opening(text: str, count: int) -> list[str]:
    """The first `count` top-level statements of `text`'s longest compiling prefix, docstrings and bare constants
    aside, each as its syntax tree prints."""
    tree = runner.parse(runner.parsing_prefix(text))
    if tree is None:
        return []
    body = [node for node in tree.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    return [ast.dump(node) for node in body[:count]]


def restarted(head: str, code: str) -> bool:
    """A rest that begins like the program's start: its first statements repeat the head's first ones, at least two
    and all of the head's first `OPENING`."""
    want = opening(head, OPENING)
    return len(want) >= 2 and opening(code, len(want)) == want


def repair(head: str, tail: str, partial: str, code: str, rest: str, cut: bool, joined: str,
           how: str) -> tuple[Repair | None, bool]:
    """The repaired join of a rest under `head` (today's head; `tail` the received lines of the statement the cut
    fell in and `partial` the received part of the line it fell in; `code` the rest as the reply gave it and `rest`
    as today stitched it; `cut` where the cap cut the rest again), given today's (`joined`, read `how`), and whether
    the prefix's reading was tried (`join_validated`): (None, ...) where today's join stands. A restart first (c);
    then, where today's join fails and the cut fell inside a statement, the prefix's reading (a) where the rest goes
    on from the cut (`resumes`), taken where it compiles (b)."""
    good = sound if cut else runner.compiles
    if how not in TODAY_RESTARTS and not same(joined, rest) and restarted(head, code) and good(rest):
        return Repair(rest, RESTART, "restart"), False
    if not tail.strip() or good(joined):
        return None, False
    keep = resumes(tail, partial, rest)
    if keep is None:
        return None, True
    under = "".join(tail.splitlines(keepends=True)[:keep])
    text = head + under + rest
    return (Repair(text, PREFIX, "prefix") if good(text) else None), True


# ------------------------------------------------------------------------------------- (d): a fragment staged


def bound(tree: ast.AST) -> set[str]:
    """Every name `tree` binds anywhere: assigned, deleted, defined, imported, an argument, an exception's, declared
    global or nonlocal, or captured by a pattern."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
    return names


def starred(tree: ast.AST) -> bool:
    """`tree` imports `*` from somewhere, so the names it binds are not all its own to read."""
    return any(isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
               for node in ast.walk(tree))


def top_reads(tree: ast.Module) -> set[str]:
    """The names a module's top level reads when it runs: every statement's but within a function, lambda or class
    body, which run later or not at all."""
    found, stack = set(), [node for node in tree.body if not isinstance(node, SCOPES)]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            found.add(node.id)
        stack.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, SCOPES))
    return found


def guarded(node: ast.stmt) -> bool:
    """`node` is an `if __name__ == "__main__":` block."""
    return isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) and "__main__" in ast.unparse(node.test)


def inert(tree: ast.Module) -> bool:
    """A module that holds nothing to run: imports, `pass`, docstrings and main blocks holding the same, alone."""
    def quiet(body: list[ast.stmt]) -> bool:
        return all(isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass))
                   or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
                   or (guarded(node) and quiet(node.body) and quiet(node.orelse)) for node in body)
    return quiet(tree.body)


def alone(text: str, whole: str) -> str:
    """Why `text` cannot run in the place of `whole` (a file that compiles): `compile` (it does not), `names` (its top
    level reads a name it binds nowhere and `whole` binds, a NameError there), `inert` (it holds nothing to run and
    `whole` does); "" where it can."""
    if not runner.compiles(text):
        return "compile"
    tree, old = runner.parse(text), runner.parse(whole)
    if tree is None or old is None:
        return ""
    if inert(tree) and not inert(old):
        return "inert"
    if not starred(tree) and (top_reads(tree) - bound(tree) - PLAIN) & bound(old):
        return "names"
    return ""


def staged(text: str, whole: str) -> str:
    """Why a write of `text` over the saved `whole` is staged beside it: `whole` compiles, `text` is shorter, and it
    cannot run in its place (`alone`); "" where the write is saved as today."""
    if not runner.compiles(whole) or len(text.encode()) >= len(whole.encode()):
        return ""
    return alone(text, whole)
