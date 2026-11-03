"""The unit N11's bounds on the texts the House reads are measured in (lane tokens1; the owner's in508, in514, in516):
the House's own tokens, counted by the model's public tokenizer the image carries (`TOKENIZER` beside this file, read
by the `tokenizers` library the image installs), each bound a token target; and only where the tokenizer fails to
load, the target's calibrated byte bound (the target at its text kind's measured bytes a token: agent/n11/library.py
and commands.py). Which unit, and why bytes where it is bytes, is said on the unit's log and summary line
(`measure=tokens` or `measure=bytes`), never to the House: no count is stated to the model (the owner's in281).

The switch (`SWITCH_ENV`, or agent/n11/TOKENS, one word, written by an image's builder): `on` (the default) or `off`,
the byte bounds by choice. The desk keeps the tokenizer in its scratch, never under agent/: a desk process names it
by `TOKENIZER_ENV`, and the harness lays a copy beside this file in the snapshot a desk run mounts.

- `UNIT`: the unit read once at import (`resolved`); `using(unit)` runs a block under another (the desk's reads).
- `size(text)` and `prefix(text, room)`: a text's measure, and its prefix within `room`, in the unit.
- `bound(tokens, size)`: a bound in the unit, its token target, else its bytes.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = "tokenizer.json"            # Nemotron 3 Super's public BPE, beside this file in the image (agent/Dockerfile)
TOKENIZER_ENV = "AGENTHON_N11_TOKENIZER"    # the desk's path to it, where no copy lies beside this file
SWITCH_ENV, SWITCH_FILE = "AGENTHON_N11_TOKENS", "TOKENS"
SWITCHES = ("on", "off")


@dataclass(frozen=True)
class Unit:
    name: str                       # "tokens" or "bytes"
    why: str = ""                   # where bytes: the switch, or why the tokenizer did not load
    tokenizer: object | None = None

    def size(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.encode("utf-8"))
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids) if text else 0

    def prefix(self, text: str, room: int) -> str:
        """`text`'s prefix within `room`: its longest in whole characters within `room` bytes, or the text before its
        own token at `room` (stepped back while a merge at the cut counts more), never cut inside that token."""
        if room <= 0:
            return ""
        if self.tokenizer is None:
            return text.encode("utf-8")[:room].decode("utf-8", errors="ignore")
        offsets = self.tokenizer.encode(text, add_special_tokens=False).offsets
        if len(offsets) <= room:
            return text
        head = text[:offsets[room][0]]
        while head and self.size(head) > room:
            head = head[:-1]
        return head

    def said(self) -> str:
        """The unit for the log: `tokens`, or `bytes (<why>)`."""
        return self.name if self.tokenizer is not None else f"bytes ({self.why})"


def loaded(path: Path) -> Unit:
    """The unit `path`'s tokenizer gives: tokens where the library and the file load, else bytes saying why."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return Unit("bytes", "the tokenizers library is not installed")
    if not Path(path).is_file():
        return Unit("bytes", f"no tokenizer at {path}")
    try:
        return Unit("tokens", tokenizer=Tokenizer.from_file(str(path)))
    except Exception as exc:  # noqa: BLE001 - a file that does not load: the byte bounds, and the log says why
        return Unit("bytes", f"the tokenizer did not load ({type(exc).__name__})")


def resolved(environ=None, here: Path = HERE) -> tuple[Unit, str]:
    """The unit this process measures in and where the switch was read: the switch's word (`SWITCH_ENV` where it is
    set, else the file `SWITCH_FILE` under `here`, else `on`); under `on`, the tokenizer of `TOKENIZER_ENV` where it is
    set, else the one beside this file (`loaded`). A word that is no switch runs `on`, and the source says so."""
    environ = os.environ if environ is None else environ
    said, source = (environ.get(SWITCH_ENV) or "").strip(), SWITCH_ENV
    if not said:
        try:
            said, source = (here / SWITCH_FILE).read_text(encoding="utf-8").strip(), SWITCH_FILE
        except (OSError, ValueError):
            said, source = "on", "no switch named"
    if said not in SWITCHES:
        said, source = "on", f"{source} names {said[:40]!r}, no switch"
    if said == "off":
        return Unit("bytes", f"{source}: off"), source
    return loaded(Path((environ.get(TOKENIZER_ENV) or "").strip() or here / TOKENIZER)), source


UNIT, UNIT_SOURCE = resolved()          # read once at import
BYTES = Unit("bytes", "named by the caller")


@contextlib.contextmanager
def using(unit: Unit):
    """`unit` as this process's for the block, the import's put back after."""
    global UNIT
    before, UNIT = UNIT, unit
    try:
        yield unit
    finally:
        UNIT = before


def size(text: str) -> int:
    return UNIT.size(text)


def prefix(text: str, room: int) -> str:
    return UNIT.prefix(text, room)


def bound(tokens: int, size_: int) -> int:
    """A bound in this process's unit: `tokens`, its token target, else `size_`, its calibrated bytes."""
    return tokens if UNIT.tokenizer is not None else size_
