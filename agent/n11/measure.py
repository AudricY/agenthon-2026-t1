"""Counts text in the model's own tokens, so that every size bound in the prompt is a token count.

Called by: agent/n11/library.py, agent/n11/commands.py and agent/n11/selection.py.
Start at: `size` and `bound`.

The rules in detail
-------------------
The unit the agent's text bounds are measured in: the model's own tokens, counted by its public tokenizer
(`TOKENIZER`, beside this file in the image, read by the `tokenizers` library), each bound a token target. Only where
the tokenizer fails to load does a bound fall back to its calibrated byte size.

- `UNIT`: the unit, read once at import (`resolved`).
- `size(text)`: a text's measure in the unit (`Unit.prefix`: its prefix within a room).
- `bound(tokens, size)`: a bound in the unit: its token target, else its bytes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = "tokenizer.json"            # Nemotron 3 Super's public BPE
TOKENIZER_ENV = "AGENTHON_N11_TOKENIZER"    # the build's path to it (Dockerfile's index stage), where none lies here


@dataclass(frozen=True)
class Unit:
    name: str                       # "tokens" or "bytes"
    why: str = ""                   # where bytes: why the tokenizer did not load
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


def resolved() -> Unit:
    """The unit this process measures in: the tokenizer `TOKENIZER_ENV` names where it is set, else the one beside
    this file (`loaded`)."""
    return loaded(Path((os.environ.get(TOKENIZER_ENV) or "").strip() or HERE / TOKENIZER))


UNIT = resolved()                       # read once at import
BYTES = Unit("bytes", "named by the caller")


def size(text: str) -> int:
    return UNIT.size(text)


def bound(tokens: int, size_: int) -> int:
    """A bound in this process's unit: `tokens`, its token target, else `size_`, its calibrated bytes."""
    return tokens if UNIT.tokenizer is not None else size_
