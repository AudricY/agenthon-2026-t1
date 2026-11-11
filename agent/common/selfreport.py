"""Reads a script's own verdict back from its JSON output: a check it reported as failed, or a NaN it wrote.

Step: 4, among the checks after a run.
Called by: agent/n11/commands.py (`Workspace.checks`).
Start at: `flags`.

The rules in detail
-------------------
The script's own verdict, read back from its output: answer-free flags over the JSON a run wrote.

A script that checks its own numbers often writes the check's outcome beside them (`mc_validates: false`) and still
exits 0. `flags(out_dir)` reads that verdict back, knowing nothing of the task or its answer. Two rules, each over
every value of every `*.json` file under `out_dir`, a value named by its innermost key (an item of a list by the
list's key):
- `false: KEY`, a boolean false under a key that names a check: one of its words is `ok`, `passed`, `passes`,
  `holds`, `confirmed`, `consistent`, `converged`, `match`, `matches` or `matched`, or begins `validat`
  (`mc_validates`, `calibration_rmse_ok`, `parity_holds`, `allChecksPassed`). A key holding a negating word (`not`,
  `fail`, `fails`, `failed`, `failure`; `mismatch` and `inconsistent` are words of their own) is never a check, nor is
  a property (`is_stationary`, `yz_most_efficient`), whose false is a finding a correct output may hold.
- `nonfinite: KEY`, a number that is NaN or infinite (JSON's `NaN`, `Infinity` and `-Infinity`, which Python writes).
A key's words, lower-cased, are its runs of letters and of digits, a run split before an upper-case letter that
follows a lower-case one or starts a capitalised word (`MCValidates` is `mc validates`).

Each flag is one short line, a rule's word and the key; the list is sorted, without repeats, and at most `MAX_FLAGS`
long. Bounded: at most `MAX_FILES` JSON files, each at most `MAX_BYTES`, found in at most `MAX_ENTRIES` entries of
the tree (links never followed); a list's first `MAX_ITEMS` items. A file that does not parse as JSON is skipped.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

MAX_FILES = 64                # JSON files read
MAX_BYTES = 8 * 1024 * 1024   # a larger file is skipped
MAX_ENTRIES = 4096            # the tree's entries looked at (guard.MAX_ENTRIES, the organisers' bound)
MAX_ITEMS = 2000              # a list's items walked
MAX_FLAGS = 32
KEY_CHARS = 80                # a key shown at most this long

CHECK_WORDS = frozenset({"ok", "passed", "passes", "holds", "confirmed", "consistent", "converged", "match",
                         "matches", "matched"})
CHECK_PREFIX = "validat"
NEGATING = frozenset({"not", "fail", "fails", "failed", "failure"})
WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
UNSHOWN = re.compile(r"[^\x20-\x7e]")


def words(key: str) -> list[str]:
    """The key's words, lower-cased: split at every non-alphanumeric character and before an upper-case letter."""
    return [part.lower() for part in WORD.findall(key)]


def names_check(key: str) -> bool:
    found = words(key)
    if any(word in NEGATING for word in found):
        return False
    return any(word in CHECK_WORDS or word.startswith(CHECK_PREFIX) for word in found)


def shown(key: str) -> str:
    return UNSHOWN.sub("?", key)[:KEY_CHARS] or "?"


def values(doc, top: str):
    """(key, value) for every scalar in `doc`, the innermost key naming it (`top` for one outside any object); a
    list's first `MAX_ITEMS` items; iterative, so a deep document never exhausts the stack."""
    stack = [(top, doc)]
    while stack:
        key, value = stack.pop()
        if isinstance(value, dict):
            stack.extend((str(name), item) for name, item in value.items())
        elif isinstance(value, list):
            stack.extend((key, item) for item in value[:MAX_ITEMS])
        else:
            yield key, value


def judge(doc, top: str, found: set[str]) -> None:
    for key, value in values(doc, top):
        if value is False:
            if names_check(key):
                found.add(f"false: {shown(key)}")
        elif isinstance(value, float) and not math.isfinite(value):
            found.add(f"nonfinite: {shown(key)}")


def documents(out_dir: Path) -> list[Path]:
    """The JSON files under `out_dir`, by path, within the bounds; links never followed."""
    found, seen = [], 0
    for where, dirs, files in os.walk(out_dir):
        dirs.sort()
        for name in sorted(files):
            seen += 1
            if seen > MAX_ENTRIES or len(found) >= MAX_FILES:
                return found
            path = Path(where) / name
            if name.lower().endswith(".json") and path.is_file() and not path.is_symlink():
                found.append(path)
        seen += len(dirs)
    return found


def flags(out_dir: Path) -> list[str]:
    """The output's self-reported failures, one short line each (the rules above); [] where none, or no such
    directory."""
    found: set[str] = set()
    for path in documents(Path(out_dir)):
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            doc = json.loads(path.read_bytes())
        except (OSError, ValueError, RecursionError):
            continue
        judge(doc, path.name, found)
    return sorted(found)[:MAX_FLAGS]
