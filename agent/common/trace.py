"""The desk's transcript of one unit (docs/trace.md): what the agent sent and got, ran and saw, and decided, one JSON
line an event in `<AGENTHON_TRACE_DIR>/trace.jsonl`.

Off unless that variable names a directory. The harness sets it under `--trace` (src/agenthon_t1/harness.py); the
image never does (agent/Dockerfile names no such variable, and tests/test_trace.py pins that), so on the platform
every call here returns at once and computes nothing. On, `begin` opens the unit's trace, refused where the directory
is the output directory or inside it (nothing is written into `--out`), and the hooks write where the events happen:
`common.house` each request, `common.runner` each script and its run, `common.guard` the log's lines and the close,
the agents' loops their decisions (`repair`, `reask`, `sample`, `stop`, each with its reason), L1 its contract's
check and W1 each draft's outcome, with a copy of each draft's files beside the trace (`directory`). Every line holds
`utc`, `seconds` since `begin`, `step`, `requests` (the House requests so far) and the step's payload.

Bounded: a text (a prompt, a reply, a script) whole up to `TEXT_BYTES`, a run's stdout and stderr each whole up to
`STREAM_BYTES`, past which their head and tail are kept and the bytes cut are said (the whole's size is its own
field); a line over `EVENT_BYTES` has every text cut to `CLIP_BYTES`; past `UNIT_BYTES` one `cap` line and nothing
more. No credential enters a line: the values of the House's secret variables (the token, the proxies' logins) are
replaced wherever they appear. A failed write turns the trace off and is never raised: the agent's behaviour never
depends on its trace. Standard library only, as the rest of `common`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

ENV = "AGENTHON_TRACE_DIR"
FILE = "trace.jsonl"
TEXT_BYTES = 64_000           # a prompt (the digest's 24,000 and a repair's script and errors), a reply, a script
STREAM_BYTES = 16_000         # a run's stdout and stderr, each: its head's quarter and its tail's three quarters
EVENT_BYTES = 256_000         # one line; over it, every text cut to CLIP_BYTES
CLIP_BYTES = 4_000
UNIT_BYTES = 4_000_000        # the unit's file: 12 requests of L1 with their scripts and runs come to about 1.5 MB
FILES_SHOWN = 64              # the files at the output's root a run or the close lists
LOCK_WAIT_S = 0.5             # an event waits this long for another thread's write, then is dropped: the watchdog's
                              # close keeps its 10 s
SECRET_ENV = ("MODEL_TOKEN", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
REDACTED = "<redacted>"
BOUNDS = {"text_bytes": TEXT_BYTES, "stream_bytes": STREAM_BYTES, "event_bytes": EVENT_BYTES, "clip_bytes": CLIP_BYTES,
          "unit_bytes": UNIT_BYTES, "files_shown": FILES_SHOWN}


def size(text: str) -> int:
    """Its bytes as written: a lone surrogate (a JSON reply may hold one) as its escape, never an error."""
    return len(text.encode("utf-8", errors="backslashreplace"))


def clip(text: str, limit: int, head: float = 0.5) -> str:
    """`text` whole up to `limit` bytes, else its first `head` share and its end, the bytes cut said between."""
    data = text.encode("utf-8", errors="backslashreplace")
    if len(data) <= limit:
        return text
    first = int(limit * head)
    return (data[:first].decode("utf-8", errors="ignore") + f"\n... [{len(data) - limit:,} bytes cut] ...\n"
            + data[len(data) - (limit - first):].decode("utf-8", errors="ignore"))


def secrets(environ=None) -> tuple[str, ...]:
    """The values no line may hold: each secret variable's, and a proxy URL's login and password apart."""
    environ = os.environ if environ is None else environ
    found: set[str] = set()
    for name in SECRET_ENV:
        value = environ.get(name) or ""
        if len(value) < 6:
            continue
        found.add(value)
        try:
            parts = urlsplit(value)
            for part in (parts.password, f"{parts.username}:{parts.password}" if parts.password else None):
                if part and len(part) >= 6:
                    found.add(part)
        except ValueError:
            pass
    return tuple(sorted(found, key=len, reverse=True))


def inside(directory: Path, out: Path) -> bool:
    """The directory is `out` or below it, by the files themselves (/app/output and /output are one mount)."""
    try:
        target = os.stat(out)
    except OSError:
        return False
    for path in (directory, *directory.parents):
        try:
            found = os.stat(path)
        except OSError:
            continue
        if (found.st_dev, found.st_ino) == (target.st_dev, target.st_ino):
            return True
    return False


def root_files(out: Path) -> tuple[list[list], int]:
    """The regular files at the output's root, (name, bytes), sorted, at most `FILES_SHOWN`; and how many in all."""
    try:
        found = sorted([entry.name, entry.stat(follow_symlinks=False).st_size] for entry in os.scandir(out)
                       if entry.is_file(follow_symlinks=False))
    except OSError:
        return [], 0
    return found[:FILES_SHOWN], len(found)


def read(path: Path) -> tuple[str, int]:
    """A run's stream file within `STREAM_BYTES` (head and tail) and its whole size; ("", 0) where it cannot be read."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size <= STREAM_BYTES:
                return handle.read().decode("utf-8", errors="replace"), size
            first = STREAM_BYTES // 4
            head = handle.read(first)
            handle.seek(size - (STREAM_BYTES - first))
            tail = handle.read()
    except OSError:
        return "", 0
    return (head.decode("utf-8", errors="ignore") + f"\n... [{size - STREAM_BYTES:,} bytes cut] ...\n"
            + tail.decode("utf-8", errors="ignore")), size


class Trace:
    """One unit's trace file; `write` never raises."""

    def __init__(self, path: Path, out: Path, hidden: tuple[str, ...] = (), clock=time.monotonic) -> None:
        self.path, self.out, self.hidden, self.clock = path, out, hidden, clock
        self.started = clock()
        self.size = self.requests = self.dropped = 0
        self.full = False
        self.systems: set[str] = set()      # the system prompts already written whole, by their sha256
        self.lock = threading.Lock()

    def scrub(self, value, limit: int):
        """A payload made JSON's: every text with the secrets replaced and cut to `limit`, a dataclass as its
        fields, a path as its text."""
        if isinstance(value, str):
            for secret in self.hidden:
                value = value.replace(secret, REDACTED)
            return clip(value, limit)
        if isinstance(value, dict):
            return {str(key): self.scrub(item, limit) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self.scrub(item, limit) for item in value]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self.scrub(dataclasses.asdict(value), limit)
        if isinstance(value, Path):
            return self.scrub(str(value), limit)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self.scrub(repr(value), limit)

    def line(self, step: str, payload: dict, limit: int) -> str:
        head = {"utc": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "seconds": round(self.clock() - self.started, 3), "step": step, "requests": self.requests}
        body = self.scrub(payload, limit)
        return json.dumps({**head, **{key: value for key, value in body.items() if key not in head}},
                          ensure_ascii=False)

    def write(self, step: str, payload: dict) -> None:
        global _active
        if not self.lock.acquire(timeout=LOCK_WAIT_S):
            self.dropped += 1
            return
        try:
            if self.full:
                return
            text = self.line(step, payload, TEXT_BYTES)
            if size(text) > EVENT_BYTES:
                text = self.line(step, {**payload, "clipped": True}, CLIP_BYTES)
            if size(text) > EVENT_BYTES:
                text = self.line(step, {"dropped": "the event was over the line's bound with every text cut"}, 0)
            written = size(text) + 1
            if self.size + written > UNIT_BYTES:
                self.full = True
                text = self.line("cap", {"reason": f"the unit's trace reached {UNIT_BYTES:,} bytes: nothing more is "
                                                   f"written"}, CLIP_BYTES)
                written = size(text) + 1
            with open(self.path, "a", encoding="utf-8", errors="backslashreplace") as handle:
                handle.write(text + "\n")
            self.size += written
        except Exception:  # noqa: BLE001 - a trace that cannot be written is off; the agent goes on as before
            self.full = True
            if _active is self:
                _active = None
        finally:
            self.lock.release()


_active: Trace | None = None


def begin(agent: str, out: Path, *, environ=None) -> Trace | None:
    """The unit's trace where `ENV` names a directory that is not the output's nor inside it; None, and every later
    call a no-op, otherwise."""
    global _active
    _active = None
    environ = os.environ if environ is None else environ
    where = environ.get(ENV)
    if not where:
        return None
    try:
        directory, out = Path(where).absolute(), Path(out).absolute()
        if inside(directory, out):
            return None
        directory.mkdir(parents=True, exist_ok=True)
        _active = Trace(directory / FILE, out, secrets(environ))
    except Exception:  # noqa: BLE001
        _active = None
        return None
    event("begin", agent=agent, pid=os.getpid(), bounds=BOUNDS)
    return _active


def active() -> bool:
    return _active is not None


def directory() -> Path | None:
    """The unit's trace directory, never the output's nor inside it; None where the trace is off."""
    trace = _active
    return trace.path.parent if trace is not None else None


def event(step: str, **payload) -> None:
    """One event; nothing where the trace is off."""
    trace = _active
    if trace is not None:
        trace.write(step, payload)


def request(system: str, user: str, reply, error: BaseException | None, latency: float, ordinal: int,
            attempts: int, kind: str | None = None) -> None:
    """A House request as `common.house` saw it: the prompt, the reply (None where the call raised `error`), the
    tokens and finish, its wall seconds with the client's retries, the unit's request count after it, and how many
    requests it took; `kind`, the request's kind where the agent names one (agent/n11's, lane cards1: `request1`,
    `plan`, `loop`, `review`, `libpass`, `continuation`), else no such field; `provider`, the serving that answered,
    after the reply's fields where the reply names one (`model.Reply.provider`, lane paidpin1), else no such field. The
    system prompt is written whole the first time, then by its digest."""
    trace = _active
    if trace is None:
        return
    try:
        digest = hashlib.sha256(system.encode("utf-8", errors="backslashreplace")).hexdigest()
        trace.requests = max(trace.requests, int(ordinal))
        payload: dict = {"ordinal": ordinal, "attempts": attempts, "latency_s": round(latency, 3),
                         "system_sha256": digest, "system_bytes": size(system)}
        if kind is not None:
            payload["kind"] = kind
        if digest not in trace.systems:
            trace.systems.add(digest)
            payload["system"] = system
        payload.update(prompt=user, prompt_bytes=size(user))
        if reply is not None:
            payload.update(reply=reply.text, reply_bytes=size(reply.text),
                           finish_reason=reply.finish_reason, prompt_tokens=reply.prompt_tokens,
                           completion_tokens=reply.completion_tokens,
                           reasoning_tokens=getattr(reply, "reasoning_tokens", 0))
            if serving := getattr(reply, "provider", ""):
                payload["provider"] = serving
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"
    except Exception:  # noqa: BLE001
        return
    trace.write("request", payload)


def script(kind: str, ordinal: int, text: str, timeout: float) -> None:
    """A script about to run (`kind`: the agent's `script`, or L1's `checks` program): its text, bytes and digest."""
    if _active is None:
        return
    try:
        data = text.encode("utf-8", errors="backslashreplace")
        event("script", kind=kind, ordinal=ordinal, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
              timeout_s=round(timeout, 1), text=text)
    except Exception:  # noqa: BLE001
        return


def ran(kind: str, ordinal: int, exit: int | None, seconds: float, stdout: Path, stderr: Path) -> None:
    """A run's end: its exit (None: stopped at its limit), seconds, stdout and stderr, and the files then at the
    output's root."""
    trace = _active
    if trace is None:
        return
    try:
        out, out_bytes = read(stdout)
        err, err_bytes = read(stderr)
        files, total = root_files(trace.out)
        event("run", kind=kind, ordinal=ordinal, exit=exit, timed_out=exit is None, run_s=round(seconds, 3),
              stdout=out, stdout_bytes=out_bytes, stderr=err, stderr_bytes=err_bytes, files=files, files_total=total)
    except Exception:  # noqa: BLE001
        return


def close(removed, files) -> None:
    """The close: each entry it removed with why, and the files it left at the root."""
    if _active is not None:
        event("close", removed=[list(item) for item in removed], files=list(files)[:FILES_SHOWN],
              files_total=len(files))
