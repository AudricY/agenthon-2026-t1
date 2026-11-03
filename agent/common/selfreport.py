"""The script's own verdict, read back from its output: answer-free flags over the JSON a run wrote (lane sr;
docs/research/wrong-numerics.md, section 3).

A script that checks its own numbers often writes the check's outcome beside them (`mc_validates: false`, a relative
error of 0.41) and still exits 0, which ends N0's unit. `flags(out_dir)` reads that verdict back, knowing nothing of
the task or its answer. Three rules, each over every value of every `*.json` file under `out_dir`, a value named by
its innermost key (an item of a list by the list's key):
- `false: KEY`, a boolean false under a key that names a check: one of its words is `ok`, `passed`, `passes`,
  `holds`, `confirmed`, `consistent`, `converged`, `match`, `matches` or `matched`, or begins `validat`
  (`mc_validates`, `calibration_rmse_ok`, `parity_holds`, `allChecksPassed`). A key holding a negating word (`not`,
  `fail`, `fails`, `failed`, `failure`; `mismatch` and `inconsistent` are words of their own) is never a check, nor is
  a property (`is_stationary`, `yz_most_efficient`), whose false is a finding a correct output may hold.
- `nonfinite: KEY`, a number that is NaN or infinite (JSON's `NaN`, `Infinity` and `-Infinity`, which Python writes).
- `error: KEY=VALUE`, a finite number above `BOUND` (0.1) in absolute value under a key named as an error: its last
  word `error` or `err` (`max_bs_decomp_error`), or a word `relerr` or `abserr` (`max_mc_var_relerr`), or the words
  `rel err`, `abs err` or `error pct` in a row (`max_abs_error_cmh`, `max_mc_var_error_pct`); never a statistic that
  is an error only by name, a standard or tracking error (`mc_std_error`, `tracking_error`), which a task asks for.
  The value shown is the largest in absolute value under that key, to three significant figures.
A key's words, lower-cased, are its runs of letters and of digits, a run split before an upper-case letter that
follows a lower-case one or starts a capitalised word (`MCValidates` is `mc validates`).

Each flag is one short line, a rule's word and the key; the list is sorted, without repeats, and at most `MAX_FLAGS`
long. `quoted(found)` is what a repair request carries: `LINE`, then the flags, one a line. Bounded: at most `MAX_FILES` JSON files, each at most `MAX_BYTES`, found in at most `MAX_ENTRIES` entries of the
tree (links never followed); a list's first `MAX_ITEMS` items. A file that does not parse as JSON is skipped. Pure:
the standard library alone, nothing written, no unit or task named; the rules are general (rules 8 and 9).
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

BOUND = 0.1                   # an error-named number above this in absolute value is a failure the script reports
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
ERROR_LAST = frozenset({"error", "err"})
ERROR_WORDS = frozenset({"relerr", "abserr"})
ERROR_PAIRS = frozenset({("rel", "err"), ("rel", "error"), ("abs", "err"), ("abs", "error"), ("error", "pct")})
STATISTICS = frozenset({"std", "standard", "stderr", "tracking"})
LINE = "The script's own output reports these failures; fix the computation behind each one, not the check:"
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


def names_error(key: str) -> bool:
    found = words(key)
    if not found or any(word in STATISTICS for word in found):
        return False
    return (found[-1] in ERROR_LAST or any(word in ERROR_WORDS for word in found)
            or any(pair in ERROR_PAIRS for pair in zip(found, found[1:])))


def shown(key: str) -> str:
    return UNSHOWN.sub("?", key)[:KEY_CHARS] or "?"


def magnitude(value: int | float) -> str:
    try:
        return f"{float(value):.3g}"
    except OverflowError:
        return "inf" if value > 0 else "-inf"


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


def judge(doc, top: str, found: set[str], errors: dict[str, float], bounded: bool = True) -> None:
    for key, value in values(doc, top):
        if value is False:
            if names_check(key):
                found.add(f"false: {shown(key)}")
        elif isinstance(value, float) and not math.isfinite(value):
            found.add(f"nonfinite: {shown(key)}")
        elif bounded and isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) > BOUND \
                and names_error(key):
            if abs(value) > abs(errors.get(shown(key), 0)):
                errors[shown(key)] = value


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


def flags(out_dir: Path, *, errors: bool = True) -> list[str]:
    """The output's self-reported failures, one short line each (the rules above); [] where none, or no such
    directory. `errors=False` leaves the `error:` rule out (agent/n11, lane n11d: an error-named key's bound is a
    guess at the task's tolerance, and its false alarms cost E039's units turns)."""
    found: set[str] = set()
    largest: dict[str, float] = {}
    for path in documents(Path(out_dir)):
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            doc = json.loads(path.read_bytes())
        except (OSError, ValueError, RecursionError):
            continue
        judge(doc, path.name, found, largest, bounded=errors)
    found |= {f"error: {key}={magnitude(value)}" for key, value in largest.items()}
    return sorted(found)[:MAX_FLAGS]


def quoted(found: list[str]) -> str:
    """`LINE` and the flags under it, one a line: the self-report a repair request quotes."""
    return LINE + "\n" + "".join(f"- {flag}\n" for flag in found)
