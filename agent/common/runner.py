"""Runs one of the model's scripts in its own process group under a timeout, and bounds what it printed. It also
holds the text helpers for taking code out of a reply and for joining a script that arrived in two replies.

Step: 3 and 4.
Called by: agent/n11/commands.py (`run_bounded`, `child_env`, `bounded`), agent/n11/drafts.py (`Runner`, `extract`)
    and agent/n11/solve.py (`stitch` and the parsing helpers).
Start at: `run_bounded` for running, `stitch` for joining.

The rules in detail
-------------------
The model's script: taken from its reply, saved in its work directory, run as `python3 <file>` under a timeout.

/tmp is the platform's only writable place besides the output directory, a `noexec` tmpfs, so a script runs through
the interpreter, never by its own path, in its own process group, which a timeout or the watchdog kills whole. It runs
without the platform's model variables, so only the agent spends the unit's requests, and with `OUTPUT_DIR` naming
the output directory (`child_env`). What a run printed comes back to the model bounded in characters (`bounded`): its
head and its tail, the middle cut with a marker saying how much. A script cut at the output limit and continued in a
later reply is joined at a statement boundary (`stitch`). `run_bounded` pipes a run's stdout and stderr through a
bounded writer (`Tail`), each kept to its last `STREAM_CAP_BYTES`, so a script that prints without end fills neither
memory nor /tmp.
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tokenize
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PYTHON_TAGS = ("python", "py", "python3")
THINK = re.compile(r"<think>.*?(</think>|\Z)", re.S)
HOUSE_ENV = ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
             "NO_PROXY", "no_proxy")
OUTPUT_ENV = "OUTPUT_DIR"   # the output directory's variable, as public tasks' scripts read it (`child_env`)
TAIL_BYTES = 3_000          # a failed run's stderr shown to the model (canaries redacted there): its traceback's end
TRACEBACK = "Traceback (most recent call last)"
CUT = "\n[... {:,} characters cut ...]\n"


@dataclass
class Block:
    tag: str
    lines: list[str] = field(default_factory=list)
    closed: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class Code:
    text: str
    closed: bool            # False: the reply ended inside the block, so the script is cut


def blocks(reply: str) -> list[Block]:
    """The reply's fenced blocks in order, a last one left open where the reply ends inside it."""
    found, current = [], None
    for line in reply.splitlines():
        stripped = line.strip()
        if current is None:
            if stripped.startswith("```"):
                current = Block(stripped[3:].strip().lower())
        elif stripped.startswith("```") and not stripped.strip("`"):
            current.closed = True
            found.append(current)
            current = None
        else:
            current.lines.append(line)
    if current is not None:
        found.append(current)
    return found


def code_like(text: str) -> bool:
    """Unfenced text taken as a script: it compiles and imports something."""
    try:
        compile(text, "<reply>", "exec")
    except (SyntaxError, ValueError):
        return False
    return bool(re.search(r"^(import|from)\s+\w", text, re.M))


def extract(reply: str) -> Code | None:
    """The script in a reply: its longest python block (else its longest untagged one), or the whole reply where it
    is unfenced code; None where it holds none. Thinking the route wrote into the content is dropped first."""
    reply = THINK.sub("", reply)
    found = [block for block in blocks(reply) if block.text.strip()]
    tagged = [block for block in found if block.tag in PYTHON_TAGS] or [block for block in found if not block.tag]
    if tagged:
        best = max(tagged, key=lambda block: len(block.text))
        return Code(best.text.strip("\n") + "\n", best.closed)
    text = reply.strip()
    return Code(text + "\n", True) if text and code_like(text) else None


def complete_lines(code: str) -> str:
    """A cut script up to its last complete line (the one the cut fell in is dropped)."""
    head, _, _ = code.rstrip("\n").rpartition("\n")
    return head + "\n" if head else ""


def append(partial: str, reply: str) -> str:
    """A cut script with its continuation: the reply's code (fenced or not), less any lines that repeat the cut
    script's end."""
    found = [block for block in blocks(THINK.sub("", reply)) if block.text.strip()]
    rest = max(found, key=lambda block: len(block.text)).text if found else THINK.sub("", reply).strip("\n")
    old, new = partial.rstrip("\n").split("\n"), rest.split("\n")
    for k in range(min(len(old), len(new), 40), 0, -1):
        if old[-k:] == new[:k] and any(line.strip() for line in new[:k]):
            new = new[k:]
            break
    return partial.rstrip("\n") + "\n" + "\n".join(new).strip("\n") + "\n"


def parsing_prefix(code: str) -> str:
    """A cut script's longest prefix of whole lines that compiles, "" where none does: tried from the end, a line at a
    time, a blank or comment line never the last. `compile`, not `ast.parse`: a prefix the parser
    takes and the compiler refuses (a `return` outside a function) would not run either; a line it refuses outright
    (a lone surrogate, an expression nested past the parser's stack or chained past the compiler's recursion limit)
    ends a prefix as a syntax error does."""
    lines = code.split("\n")
    for end in range(len(lines), 0, -1):
        last = lines[end - 1].strip()
        if not last or last.startswith("#"):
            continue
        text = "\n".join(lines[:end]) + "\n"
        try:
            compile(text, "<salvage>", "exec", dont_inherit=True)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
        return text
    return ""


def bounded(text: str, limit: int, unread: int = 0, marker: str = CUT) -> str:
    """`text` whole where it is at most `limit` characters and nothing was left `unread`; else its head and its tail,
    about half each, with `marker` between them (`CUT`, saying how many characters went, the `unread` ones counted
    in; a marker with no `{}` says none), in at most `limit` characters all told. Each cut moves to a line's end
    where one falls in the outer quarter of its half."""
    if len(text) <= limit and not unread:
        return text
    room = max(0, limit - len(marker.format(len(text) + unread)))
    if len(text) <= room:
        cut_from = cut_to = len(text) // 2
    else:
        cut_from, cut_to = room // 2, len(text) - (room - room // 2)
    if 0 < cut_from < cut_to and text[cut_from - 1] != "\n":
        end = text.rfind("\n", 0, cut_from)
        if end >= 0 and cut_from - (end + 1) <= cut_from // 4:
            cut_from = end + 1
    if cut_from < cut_to < len(text) and text[cut_to - 1] != "\n":
        start = text.find("\n", cut_to)
        if start >= 0 and start + 1 - cut_to <= (len(text) - cut_to) // 4:
            cut_to = start + 1
    return text[:cut_from].removesuffix("\n") + marker.format(cut_to - cut_from + unread) + text[cut_to:]


@dataclass(frozen=True)
class Run:
    exit: int | None        # None: killed at its timeout (or by the watchdog)
    seconds: float
    stderr: str             # its tail

    @property
    def failed(self) -> bool:
        return self.exit != 0 or TRACEBACK in self.stderr


def child_env(out: Path) -> dict[str, str]:
    """A script's environment: the agent's without the platform's model variables and the agent's own settings, the
    runner's own, and `OUTPUT_ENV` naming `out` (absolute: a script runs from its work directory)."""
    env = {name: value for name, value in os.environ.items()
           if name not in HOUSE_ENV and not name.startswith("AGENTHON_")}
    env.update(MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    env[OUTPUT_ENV] = str(Path(out).absolute())
    return env


def tail(path: Path, size: int = TAIL_BYTES) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - size))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return text


class Runner:
    """Runs one script at a time in `work`, `out` its environment's output directory (`child_env`); `kill` ends the one
    running, from any thread."""

    def __init__(self, work: Path, *, out: Path) -> None:
        self.work, self.out = Path(work), out
        self.current: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.runs = 0

    def run(self, code: str, timeout: float) -> Run:
        self.runs += 1
        script = self.work / f"solve_{self.runs}.py"
        script.write_text(code, encoding="utf-8")
        out, err = self.work / f"solve_{self.runs}.out", self.work / f"solve_{self.runs}.err"
        began = time.monotonic()
        with open(out, "wb") as stdout, open(err, "wb") as stderr:
            with self.lock:
                self.current = subprocess.Popen([sys.executable, str(script)], cwd=self.work, stdin=subprocess.DEVNULL,
                                                stdout=stdout, stderr=stderr, env=child_env(self.out),
                                                start_new_session=True)
            try:
                code_ = self.current.wait(timeout=max(0.0, timeout))
            except subprocess.TimeoutExpired:
                self.kill()
                self.current.wait()
                code_ = None
            finally:
                with self.lock:
                    self.current = None
        return Run(None if code_ == -signal.SIGKILL else code_, time.monotonic() - began, tail(err))

    def kill(self) -> None:
        """The running script's whole process group, killed; nothing where none runs."""
        proc = self.current
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# ------------------------------------------------------------------------------------- a script over several replies

DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
COMPOUND = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try, *DEFINITIONS,
            *((ast.TryStar,) if hasattr(ast, "TryStar") else ()), *((ast.Match,) if hasattr(ast, "Match") else ()))


def compiles(code: str) -> bool:
    """`code` compiles whole, as `parsing_prefix` tests each prefix."""
    try:
        compile(code, "<join>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def parse(code: str) -> ast.Module | None:
    """`code`'s syntax tree; None where it does not parse."""
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def continuation(reply: str) -> str:
    """The code a reply continuing a script holds, as `append` reads it (its longest fenced block, else the reply
    unfenced), but "" where its only blocks are empty: a reply saying there is nothing more adds nothing."""
    reply = THINK.sub("", reply)
    found = blocks(reply)
    if found:
        held = [block for block in found if block.text.strip()]
        return max(held, key=lambda block: len(block.text)).text if held else ""
    return reply.strip("\n")


def _bodies(node: ast.AST) -> list[list[ast.stmt]]:
    """A compound statement's statement lists: its body, its handlers' and cases' bodies, its else and finally."""
    found = [getattr(node, name) for name in ("body", "orelse", "finalbody") if getattr(node, name, None)]
    found += [handler.body for handler in getattr(node, "handlers", ()) if handler.body]
    found += [case.body for case in getattr(node, "cases", ()) if case.body]
    return [body for body in found if isinstance(body, list) and isinstance(body[0], ast.stmt)]


def _last_chain(tree: ast.Module) -> list[ast.stmt]:
    """The statements that hold a script's last line, outermost first: its last top-level statement, the last
    statement of that one's last body, and so on down to a simple statement."""
    chain: list[ast.stmt] = []
    body = tree.body
    while body:
        node = body[-1]
        chain.append(node)
        inner = _bodies(node) if isinstance(node, COMPOUND) else []
        body = max(inner, key=lambda stmts: (stmts[-1].end_lineno or 0, stmts[-1].end_col_offset or 0)) \
            if inner else []
    return chain


def depth(code: str) -> str:
    """The indentation of a script's last statement (its first line's); "" at the top level or where it does not
    parse."""
    tree = parse(code)
    chain = _last_chain(tree) if tree is not None else []
    if not chain:
        return ""
    line = code.split("\n")[chain[-1].lineno - 1]
    return line[:len(line) - len(line.lstrip())]


def _is_main(node: ast.stmt) -> bool:
    """`if __name__ == "__main__":`."""
    return isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) and "__main__" in ast.unparse(node.test)


def _readings(kept: str, rest: str) -> list[tuple[str, str]]:
    """`rest` as written, then, where its first line's indentation is not that of `kept`'s last statement, with its
    first line given that indentation (a reply's leading whitespace dropped), and shifted whole to it."""
    found = [(rest, "as written")]
    lines = rest.split("\n")
    first = next((k for k, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return found
    want, lead = depth(kept), lines[first][:len(lines[first]) - len(lines[first].lstrip())]
    if want == lead:
        return found
    found.append(("\n".join(lines[:first] + [want + lines[first].lstrip()] + lines[first + 1:]),
                  "its first line re-indented"))
    found.append(("\n".join(want + line[len(lead):] if line.startswith(lead) else want + line.lstrip()
                            if line.strip() else line for line in lines), "shifted whole"))
    return found


OVERLAP = "overlap of the kept end"     # how `stitch` names a join that took the repeated lines once


def _normal(text: str) -> list[str]:
    """`text`'s non-blank lines, whitespace collapsed."""
    return [" ".join(line.split()) for line in text.split("\n") if line.strip()]


def repeated(kept: str, rest: str) -> int:
    """How many of `kept`'s last non-blank lines `rest` opens with, whitespace normalised, the most that match; 0
    where none does."""
    old, new = _normal(kept), _normal(rest)
    for k in range(min(len(old), len(new)), 0, -1):
        if old[-k:] == new[:k]:
            return k
    return 0


def unrepeated(kept: str, lines: int) -> str:
    """`kept` less its last `lines` non-blank lines and the blank ones among and after them, each line it keeps
    ending in a newline; `kept` where `lines` is 0."""
    if lines <= 0:
        return kept
    found = [k for k, line in enumerate(kept.split("\n")) if line.strip()]
    return "".join(line + "\n" for line in kept.split("\n")[:found[-lines]])


def _read(kept: str, readings: list[tuple[str, str]]) -> tuple[str, str]:
    """The first of the joins that compiles, else the one whose longest compiling prefix holds the most lines, the
    first among equals; `kept`, "nothing added", where there is none."""
    best, how, size = kept, "nothing added", -1
    for joined, said in readings:
        if compiles(joined):
            return joined, said
        if (found := parsing_prefix(joined).count("\n")) > size:
            best, how, size = joined, said, found
    return best, how


UNIFORM = "shifted by one amount"       # how `stitch` names its two readings past `_readings`
OPENING = "its opening run shifted"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def frame(kept: str) -> tuple[int, bool]:
    """The indentation `kept`'s open frame wants its next line at, and whether that frame is a bracket:
    its last statement's where it parses; else its last line of code's, one level deeper where that line opens a
    block. Tokens distinguish an open bracket from a header, string or clause still due."""
    if parse(kept) is not None:
        return len(depth(kept)), False
    code = [line for line in kept.split("\n") if line.strip() and not line.lstrip().startswith("#")]
    if not code:
        return 0, False
    brackets, last = 0, None
    try:
        for token in tokenize.generate_tokens(io.StringIO(kept).readline):
            if token.type == tokenize.OP:
                brackets += token.string in "([{"
                brackets -= token.string in ")]}"
            if token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT,
                                  tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                last = token
    except (tokenize.TokenError, SyntaxError):
        pass
    header = last is not None and last.string == ":" and brackets == 0
    return _indent(code[-1]) + 4 * header, brackets > 0


def shifted(rest: str, by: int, opening: bool = False, closing: bool = False) -> str:
    """`rest` with `by` spaces of indentation taken from each line that has them (the others as written),
    or given to each where `by` is negative; under `opening` only its opening run: its lines up to the first shallower
    than its first, that one too where it is `closing` a bracket, the rest as written."""
    lines = rest.split("\n")
    lead = next((_indent(line) for line in lines if line.strip()), 0)
    out, running = [], True
    for line in lines:
        moved = running or not opening
        if opening and running and line.strip() and _indent(line) < lead:
            running, moved = False, closing
        if not line.strip() or not moved:
            out.append(line)
        else:
            out.append(" " * -by + line if by < 0 else line[by:] if _indent(line) >= by else line)
    return "\n".join(out)


def share(code: str) -> float:
    """The share of `code`'s lines its longest compiling prefix holds (`parsing_prefix`), 1 where it
    compiles whole."""
    return 1.0 if compiles(code) else parsing_prefix(code).count("\n") / max(code.count("\n"), 1)


def _reread(kept: str, rest: str, lines: int, found: tuple[str, str], again: bool) -> tuple[str, str]:
    """`found`, the join so far, unless it fails and a reading past `_readings` compiles: the rest shifted by one
    amount so its first line sits where the head's open frame wants it (`UNIFORM`), then its opening run alone
    (`OPENING`), the first that compiles taken; for a rest cut `again`, which no reading may compile whole, the one
    whose longest compiling prefix holds the greatest share of its lines (`share`), ties to `found`. Under the head
    less a repeated end (`lines`), as the overlap reads it."""
    if not again and compiles(found[0]):
        return found
    head = unrepeated(kept, lines) if lines else kept
    want, bracket = frame(head)
    lead = next((_indent(line) for line in rest.split("\n") if line.strip()), want)
    if want == lead:
        return found
    named = f"{OVERLAP}, " if lines else ""
    joins = [(head + text.strip("\n") + "\n" if lines else append(kept, text), named + said) for text, said in (
        (shifted(rest, lead - want), UNIFORM), (shifted(rest, lead - want, True, bracket), OPENING))]
    if not again:
        return next(((joined, said) for joined, said in joins if compiles(joined)), found)
    best, most = found, share(found[0])
    for joined, said in joins:
        if (part := share(joined)) > most:
            best, most = (joined, said), part
    return best


def stitch(kept: str, reply: str, *, overlap: bool = False, again: bool = False) -> tuple[str, str]:
    """`kept`, a script ending at a whole statement, joined with the code `reply` continues it by, and how the reply
    was read: as written, or with its first line re-indented or shifted whole to the indentation of `kept`'s last
    statement (`_readings`), each joined by `append` (which drops lines repeating `kept`'s end) and the reading whose
    longest compiling prefix holds the most lines taken, the first among equals. A reply that starts again from `kept`'s first
    line is taken past `kept` where it repeats `kept` line for line, else as the script itself (restarted); a reply
    with no code adds nothing. Under `overlap`, where the reply opens with `kept`'s last lines, whitespace aside
    (`repeated`, the most that match), those lines go in once as the reply has them: `kept` less them (`unrepeated`)
    and the reply's readings under it, the first that compiles taken (`OVERLAP` naming the reading); else the plain
    joins where one compiles; else the overlap's reading whose longest compiling prefix holds the most lines. Under
    `overlap` too, where that join does not compile, or the reply was cut `again`, two readings more (`_reread`)."""
    rest = continuation(reply)
    if not rest.strip():
        return kept, "nothing added"
    old = [line.rstrip() for line in kept.rstrip("\n").split("\n")]
    new = rest.split("\n")
    first = next((line for line in old if line.strip()), None)
    if first is not None and len(old) > 1 and next((line.rstrip() for line in new if line.strip()), None) == first:
        if [line.rstrip() for line in new[:len(old)]] != old:
            return rest.strip("\n") + "\n", "restarted"
        rest = "\n".join(new[len(old):])
        if not rest.strip():
            return kept, "repeated, nothing added"
    plain = [(append(kept, reading), said) for reading, said in _readings(kept, rest)]
    lines = repeated(kept, rest) if overlap else 0
    if not lines:
        found = _read(kept, plain)
    else:
        head = unrepeated(kept, lines)
        taken = [(head + reading.strip("\n") + "\n", f"{OVERLAP}, {said}") for reading, said in _readings(head, rest)]
        found = next(((joined, said) for joined, said in taken + plain if compiles(joined)), None) or \
            _read(kept, taken)
    return _reread(kept, rest, lines, found, again) if overlap else found


# ------------------------------------------------------------------------------------------------ what runs a script

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _calls(node: ast.AST, names: set[str]) -> set[str]:
    """The names in `names` that `node` calls by name, the bodies of the functions, classes and lambdas inside it
    left out (defining code runs none of it)."""
    found: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Call) and isinstance(current.func, ast.Name) and current.func.id in names:
            found.add(current.func.id)
        stack.extend(child for child in ast.iter_child_nodes(current)
                     if not isinstance(child, (*DEFINITIONS, ast.Lambda)))
    return found


def has_driver(code: str) -> bool:
    """`code` runs what it defines when it runs: it has a main guard (`if __name__ == "__main__":`), or a top-level
    statement (a bare call, an assignment's value, inside a top-level `try`, `with`, `if` or loop) calls a function or
    class it defines, and where it defines `main`, something in it calls `main`. A call of an imported name alone
    (`np.random.seed(0)`) runs nothing of the script's. True where it defines no function (its top level is its
    program) or does not parse (no driver can be read or added)."""
    tree = parse(code)
    if tree is None or not any(isinstance(node, FUNCTIONS) for node in tree.body):
        return True
    if any(_is_main(node) for node in tree.body):
        return True
    defined = {node.name for node in tree.body if isinstance(node, DEFINITIONS)}
    called = set().union(*(_calls(node, defined) for node in tree.body if not isinstance(node, DEFINITIONS)))
    if not called:
        return False
    return "main" not in defined or any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                        and node.func.id == "main" for node in ast.walk(tree))


def _no_arguments(node: ast.FunctionDef) -> bool:
    """A function called with no arguments: each positional and keyword-only one it takes has a default."""
    args = node.args
    return len(args.defaults) >= len(args.posonlyargs + args.args) and None not in args.kw_defaults


def entry(code: str) -> str | None:
    """The function a driver calls: `main` where the script defines it at its top level and it can be called with no
    arguments; else, where it defines no `main`, its last top-level function where that is the only one so callable.
    None otherwise, or where it does not parse."""
    tree = parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)] if tree is not None else []
    mains = [node for node in functions if node.name == "main"]
    if mains:
        return "main" if _no_arguments(mains[-1]) else None
    bare = [node for node in functions if _no_arguments(node)]
    return bare[0].name if bare and bare[0] is functions[-1] else None   # the first so callable, last: the only one


# ---------------------------------------------------------------------------- a run's output through a bounded writer

STREAM_CAP_BYTES = 1 << 20      # a run's stdout and stderr, each, kept to its last MiB: the work directory's /tmp is a
#                                 64 MiB tmpfs
CHUNK_BYTES = 1 << 16           # one read of a stream's pipe
DRAIN_S = 5.0                   # a stream's reader is waited on this long once the run has ended


class Tail:
    """The bounded writer: the last `cap` bytes written to it."""

    def __init__(self, cap: int = STREAM_CAP_BYTES) -> None:
        self.cap, self.held = max(0, cap), 0
        self.chunks: deque[bytes] = deque()
        self.lock = threading.Lock()

    def write(self, data: bytes) -> None:
        with self.lock:
            self.chunks.append(data)
            self.held += len(data)
            while self.chunks and self.held - len(self.chunks[0]) >= self.cap:
                self.held -= len(self.chunks.popleft())

    def value(self) -> bytes:
        """The last `cap` bytes written."""
        with self.lock:
            data = b"".join(self.chunks)
        return data[len(data) - self.cap:] if len(data) > self.cap else data


def drain(fd: int, sink: Tail) -> threading.Thread:
    """A started daemon thread reading `fd` into `sink` to its end, so the writer never waits on a full pipe."""
    def read() -> None:
        try:
            while chunk := os.read(fd, CHUNK_BYTES):
                sink.write(chunk)
        except OSError:
            pass

    thread = threading.Thread(target=read, name="drain", daemon=True)
    thread.start()
    return thread


def kill_group(proc: subprocess.Popen) -> None:
    """`proc`'s process group, killed; nothing where it is gone."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


def run_bounded(argv: list[str], timeout: float, *, cwd: Path | None = None, env: dict[str, str] | None = None,
                hold: Callable[[subprocess.Popen | None], None] | None = None) -> tuple[int | None, float, Tail, Tail]:
    """`argv` in a session of its own, its stdin empty, its stdout and stderr each through the bounded writer on a
    pipe its own thread drains; its process group killed at `timeout`. `hold` is given the process as it starts and
    None as it ends (a runner's handle for a kill from another thread). Once it has ended, whatever its group left
    running is killed and the readers are joined, `DRAIN_S` each at most. Its exit (None: stopped at `timeout`, or
    killed by SIGKILL), its seconds, and its stdout's and stderr's `Tail`."""
    out, err = Tail(), Tail()
    began = time.monotonic()
    proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, start_new_session=True)
    if hold is not None:
        hold(proc)
    readers = [drain(proc.stdout.fileno(), out), drain(proc.stderr.fileno(), err)]
    try:
        code = proc.wait(timeout=max(0.0, timeout))
    except subprocess.TimeoutExpired:
        kill_group(proc)
        proc.wait()
        code = None
    finally:
        if hold is not None:
            hold(None)
    seconds = time.monotonic() - began
    kill_group(proc)
    for reader, pipe in zip(readers, (proc.stdout, proc.stderr)):
        reader.join(DRAIN_S)
        if not reader.is_alive():
            pipe.close()
    return (None if code == -signal.SIGKILL else code), seconds, out, err
