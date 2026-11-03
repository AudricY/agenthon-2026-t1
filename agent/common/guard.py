"""The guard: what the agent leaves in `--out` is the script's deliverables and nothing the scorer refuses, and the
process ends with status 0 on time whatever stalls.

`close` is the last thing written to the output: it removes every regular file whose bytes hold one of the unit's
canaries (canary.collect; g2's `CONTAMINATION_CANARY`, docs/challenge.md "The canary"), `reward.json` and
`pytest_report.json` (g1 and g3 read them), and every entry the organisers' output checker refuses (the kit's
README.md, "Output folder rules": links and special files, set-id and sticky bits, names, collisions, depth, count,
sizes, sparse files), so a refused tree never costs a unit a script's work did not. It never writes a file. Every
close holds `WRITING`, which the `Watchdog` takes before the process ends, so no exit cuts a close in half.

The `Watchdog` ends the process with status 0 `WATCHDOG_S` before the unit's deadline: it kills the running script,
takes the lock, closes the output and exits, whatever the main thread is waiting on; a watchdog given `moved` (the
seconds the deadline has moved since it was armed: agent/n11's, the client's deadline under the desk's
`AGENTHON_MODEL_RATE_UNTIMED`, lane clock1) waits that much longer. `Steps` is the stderr log, one
line a step with the seconds and the House requests so far; it is handed no instruction text and no credential.
Under the desk's trace (`common.trace`) each log line and each close is written there too.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from . import canary, trace

WRITING = threading.Lock()
WATCHDOG_S = 10.0             # the watchdog fires this long before the deadline
LOCK_WAIT_S = 5.0             # it waits this long for a close in flight
MAX_FILES = 256
MAX_ENTRIES = 4096
MAX_DEPTH = 8                 # a folder this deep (out/1/2/.../8) is refused, even an empty one
MAX_FILE = 64 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
SPARSE = 64                   # apparent size over allocated, at most
SCORER_FILES = ("reward.json", "pytest_report.json")
UNSAFE_BITS = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX
DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass
class Closed:
    removed: list[tuple[str, str]] = field(default_factory=list)   # (path relative to the output, why)
    files: list[str] = field(default_factory=list)                 # the regular files left at the root

    def summary(self) -> str:
        reasons: dict[str, int] = {}
        for _, why in self.removed:
            reasons[why] = reasons.get(why, 0) + 1
        gone = ", ".join(f"{count} {why}" for why, count in sorted(reasons.items())) or "nothing"
        return f"{len(self.files)} file(s) at the root; removed {gone}"


def bad_name(name: str, root: bool) -> str | None:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return "a name not UTF-8"
    if unicodedata.normalize("NFC", name) != name:
        return "a name not NFC"
    if "\\" in name or any(ord(char) < 32 or ord(char) == 127 for char in name):
        return "a name with a backslash or a control character"
    if root and DRIVE.match(name):
        return "a drive-letter name"
    return None


def close(out: Path, canaries: Iterable[str] = ()) -> Closed:
    """Scrub `out` as described above; the caller holds `WRITING`. Removal is by the entry's own path, links never
    followed."""
    out, canaries, done = Path(out), [c for c in canaries if c], Closed()
    if not out.is_dir():
        return done

    def remove(path: Path, why: str) -> None:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            why += " (could not be removed)"
        done.removed.append((path.relative_to(out).as_posix(), why))

    kept: list[tuple[Path, int, int]] = []          # (file, depth, size), in walk order
    entries: list[tuple[Path, int]] = []            # every entry kept, (path, depth)

    def walk(directory: Path, depth: int) -> None:
        folded: set[str] = set()
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return
        for name in names:
            path = directory / name
            why = bad_name(name, depth == 1)
            key = unicodedata.normalize("NFC", name).casefold()
            if why is None and key in folded:
                why = "a name that differs from another only in case or Unicode form"
            if why:
                remove(path, why)
                continue
            folded.add(key)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode):
                if depth >= MAX_DEPTH:
                    remove(path, "a folder nested 8 deep")
                    continue
                if st.st_mode & UNSAFE_BITS:
                    os.chmod(path, stat.S_IMODE(st.st_mode) & ~UNSAFE_BITS)
                entries.append((path, depth))
                walk(path, depth + 1)
            elif not stat.S_ISREG(st.st_mode):
                remove(path, "a link or special file")
            elif st.st_nlink > 1:
                remove(path, "a hard link")
            elif depth == 1 and name in SCORER_FILES:
                remove(path, "a scorer's file")
            elif st.st_size > MAX_FILE:
                remove(path, "a file over 64 MiB")
            elif st.st_blocks and st.st_size > SPARSE * st.st_blocks * 512:
                remove(path, "a sparse file")
            else:
                if st.st_mode & UNSAFE_BITS or not st.st_mode & stat.S_IRUSR:
                    os.chmod(path, (stat.S_IMODE(st.st_mode) & ~UNSAFE_BITS) | stat.S_IRUSR)
                try:
                    tainted = canary.file_holds(path, canaries)
                except OSError:
                    tainted = True
                if tainted:
                    remove(path, "a canary")
                    continue
                kept.append((path, depth, st.st_size))
                entries.append((path, depth))

    walk(out, 1)
    # Over the bounds, the deepest and then the last-named entries go first: the root's files are the deliverables.
    order = sorted(kept, key=lambda item: (item[1], item[0].as_posix()))
    while len(order) > MAX_FILES or sum(size for _, _, size in order) > MAX_TOTAL:
        path, _, _ = order.pop() if len(order) > MAX_FILES else max(order, key=lambda item: (item[2], item[1]))
        order = [item for item in order if item[0] != path]
        entries[:] = [item for item in entries if item[0] != path]
        remove(path, "a file over the tree's count or size")
    if len(entries) > MAX_ENTRIES:
        for path, _ in sorted(entries, key=lambda item: (item[1], item[0].as_posix()))[MAX_ENTRIES:][::-1]:
            remove(path, "an entry over the tree's count")
    done.files = sorted(path.name for path in out.iterdir() if path.is_file() and not path.is_symlink())
    trace.close(done.removed, done.files)
    return done


class Steps:
    """stderr, one line a step: the agent, the seconds since its start, the House requests so far, the step."""

    def __init__(self, agent: str, requests: Callable[[], int] = lambda: 0, stream=None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.agent, self.requests, self.stream, self.clock = agent, requests, stream, clock
        self.started = clock()
        self.lines: list[str] = []

    def __call__(self, step: str, detail: str = "") -> None:
        line = f"{self.agent} {self.clock() - self.started:7.1f}s req={self.requests()} {step}" + \
            (f": {detail}" if detail else "")
        self.lines.append(line)
        trace.event("log", name=step, detail=detail)
        try:
            print(line, file=self.stream or sys.stderr, flush=True)
        except (OSError, ValueError):       # a stderr nobody reads (EPIPE) or closed costs the line, not the run
            pass


class Watchdog:
    """At `deadline - reserve`: `stop()` (the running script killed), `WRITING` taken, `finish()` (the close), then
    `exit(0)`. Not armed where the deadline leaves no room. `moved`, where given, says how far the deadline has moved
    since (seconds, never less than 0), and the watch waits on to the moved time; None, the time is fixed."""

    def __init__(self, deadline: float, stop: Callable[[], None], finish: Callable[[], None],
                 log: Callable[[str, str], None], *, reserve: float = WATCHDOG_S,
                 exit: Callable[[int], None] = os._exit, moved: Callable[[], float] | None = None) -> None:
        self.at = deadline - reserve
        self.stop, self.finish, self.log, self.exit = stop, finish, log, exit
        self.moved = moved
        self.cancelled = threading.Event()
        self.fired = threading.Event()
        self.thread: threading.Thread | None = None

    def arm(self) -> Watchdog:
        if self.at > time.time():
            self.thread = threading.Thread(target=self._watch, name="watchdog", daemon=True)
            self.thread.start()
        return self

    def cancel(self) -> None:
        self.cancelled.set()

    def _watch(self) -> None:
        if self.cancelled.wait(max(0.0, self.at - time.time())):
            return
        while self.moved is not None and time.time() < self.at + max(0.0, self.moved()):
            if self.cancelled.wait(max(0.0, self.at + max(0.0, self.moved()) - time.time())):
                return
        self.fired.set()
        try:
            self.stop()
        except Exception:  # noqa: BLE001 - nothing stops the exit
            pass
        held = WRITING.acquire(timeout=LOCK_WAIT_S)
        try:
            self.log("watchdog", f"the deadline is near; the lock {'taken' if held else 'not free'}; exit 0")
            self.finish()
        except Exception:  # noqa: BLE001
            pass
        self.exit(0)
        if held:                # reached only where `exit` returns (a test's)
            WRITING.release()
