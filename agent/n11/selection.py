"""Step 2's request: the model is shown the task and the library's index, and names the functions a correct solution
would call. Their documentation then replaces the full index in the loop's system prompt.

Called by: agent/n11/solve.py (`N11.selected`, `N11.made_whole`).
Calls: agent/n11/library.py for the index and the cards; the model through the loop's client.
Start at: `Selection.call` (the request) and `shaped` (the system prompt's library section built from the picks).

The rules in detail
-------------------
The selection call: one bounded request before the loop names the library functions a correct solution would
call, and their cards and their shelves' names take the index's place in the loop's system prompt.

The call (`Selection.call`) is the unit's first request, sent where `FLOOR` requests and as many of a request's seconds
are left: `SYSTEM`, and the user turn `render` writes (the task's instruction, its input files described as the loop's
first message describes them, the library's scored index, `Library.index(task=)`, and `ASK`, which states no count).
It asks low-effort thinking within `MAX_TOKENS_LOW`. The reply is read by `parse` (solve.py's `functions_named`, passed
in: every `qf.` name and every bare index name, looked up in the whole library, the first `MOST` kept). A reply cut at
its bound has its picks completed by the pointer's functions it did not name (`completed`), so a function the pointer
named is never lost to a cut reply; nothing is said to the model of the cut or of the completion.

The picks shape the system prompt (`shaped`): the cards of the pointer's functions, then the picks' (`CARDS_HEAD`),
within the call's own cards' bound (`card_bound`, trimmed by `Library.carded`), a scaffolding pick (`scaffolding`) never
carded; and in the index's place (`SHELVES_HEAD`) the line of each shelf one of those functions or picks lies on with
every function it holds by name, then the other shelves' names (`OTHERS_LINE`). A call that draws no reply, or names no
function the library holds but the scaffolding (and was not completed), leaves the system prompt with the index and the
pointer's cards.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from common import runner
from common.house import model
from n11 import library, measure

MOST = 8                        # the picks kept, at most; never said to the model
MAX_TOKENS_LOW = 2_000  # the call's output with low-effort thinking, the reasoning counted in it. A reply cut while it
#                         still reasons names nothing, which costs the request and leaves the prompt as without the call
FLOOR = 2           # the call is sent only with this many requests left and this many of a request's seconds
CARD_TOKENS = 3_300  # the cards' bound under the call, its head included: the index's room is freed for them
CARD_BYTES = 8_800   # the byte fallback where the tokenizer does not load: library.CARD_BYTES twice, as the tokens are

SYSTEM = ("You are an expert quantitative developer. You read a task and the index of qf, a Python library of "
          "quantitative methods, and name the library's functions a correct solution of the task would call. You "
          "write no code.")
ASK = ("Name the few functions that matter for a correct solution of this task: the library's functions it would "
       "call where the task names their method or needs the quantity one computes. Where one function computes a "
       "quantity the task asks as a whole, name that function rather than the building blocks a solution could "
       "compute it from. Answer with the list alone, the one that matters most first, one function a line, written "
       "as below, and nothing else:\n\nqf.<name>\n\nWhere the library holds nothing the task needs, answer none.")
TASK_HEAD, DESCRIBED_HEAD, LIBRARY_HEAD, ASK_HEAD = "# Task", "# Input files, described", "# The library", "# The ask"
QUALIFIED = re.compile(r"(?<![\w.])qf(?:\.[A-Za-z_]\w*)+")     # a `qf.` name
NONE = re.compile(r"^[\W_]*none[\W_]*$", re.I)                  # a reply of `none` alone
CARDS_HEAD = ("Cards of the library's functions this task needs, as a reading of the task named them; where one "
              "computes a quantity the task asks, call it rather than writing its formula:")
SHELVES_HEAD = ("qf, the method library (import qf; numpy and scipy; every convention a required argument). The "
                "shelves (modules) of the functions carded below, each with every function it holds by name; `lib "
                "name=<function>` shows a function's card and `lib shelf=<module>` a shelf's, these shelves or the "
                "others:")
OTHERS_LINE = "the other shelves: {names}"


def render(instruction: str, described: str, index: str) -> tuple[str, str]:
    """The call's system and user turns: the task, its input files described (where it has any), the index, then
    the ask."""
    user = f"{TASK_HEAD}\n\n{instruction}\n\n" + (f"{DESCRIBED_HEAD}\n\n{described}\n\n" if described else "")
    user += f"{LIBRARY_HEAD}\n\n{index}\n\n"
    return SYSTEM, user + f"{ASK_HEAD}\n\n{ASK}\n"


def parse(text: str, lib: library.Library, read) -> dict:
    """A reply's picks: `read` (solve.py `functions_named`: each `qf.` name and each bare index name, first named first,
    once each, looked up in the whole library) on the text with any `<think>` block left out; the `qf.` names the
    library lacks (`unknown`); the picks a bare name alone gave (`bare`); every pick (`named`) and the first `MOST`
    (`picks`), the rest counted (`over`); and whether the reply is `none` alone (`none`)."""
    text = runner.THINK.sub("", text or "")
    named, unknown = read(text, lib)
    qualified = {found.name for found in map(lib.find, QUALIFIED.findall(text)) if found is not None}
    return {"picks": [f.call for f in named[:MOST]], "named": [f.call for f in named], "unknown": unknown,
            "bare": sum(f.name not in qualified for f in named), "over": max(0, len(named) - MOST),
            "none": NONE.match(text.strip()) is not None}


def completed(picks: list[library.Function], pointed: list[tuple[library.Function, str]]) -> list[library.Function]:
    """The pointer's functions (`pointed`, the loop's `fired`, in the pointer's order) a cut reply's `picks` lack."""
    named = {function.name for function in picks}
    return [function for function, _ in pointed if function.name not in named]


def card_bound() -> int:
    """The cards' bound under the call in this process's unit (agent/n11/measure.py): `CARD_TOKENS`, else
    `CARD_BYTES`."""
    return measure.bound(CARD_TOKENS, CARD_BYTES)


def scaffolding(function: library.Function) -> bool:
    """A function of the library's scaffolding shelf (`library.UNPOINTED`: files, returns, annualising, day counts,
    compounding, windows and the statistics whose convention a task names). A selection names these often, beside
    the function a task wants, at the cost of the cards' room: such a pick stays named on its shelf's line, never
    carded."""
    return function.shelf in library.UNPOINTED


@dataclass(frozen=True)
class Shaped:
    """The system prompt's library under the call: the section in the index's place, the cards and what their trim
    cut (a line a card, for the log), the functions whose cards are shown, the functions the section names, and the
    functions carded in their order before the trim (the pointer's, then the picks' but the scaffolding's)."""
    index: str
    cards: str
    cuts: tuple[str, ...]
    carded: tuple[library.Function, ...]
    named: tuple[library.Function, ...]
    order: tuple[library.Function, ...] = ()


def shaped(lib: library.Library, picks: list[library.Function],
           pointed: list[tuple[library.Function, str]]) -> Shaped | None:
    """The library as the system prompt shows it under the call (above), None where the picks hold nothing but the
    scaffolding: the pointer's functions first, then each pick the pointer lacks but the scaffolding's, carded within
    the call's cards' bound (`Library.carded` under `CARDS_HEAD`, the last carded cut first); the shelves those
    functions and the picks lie on, in the order first named, each line with every function it holds; the other
    shelves by name."""
    first = [function for function, _ in pointed]
    chosen = [function for function in picks if not scaffolding(function)]
    if not chosen:
        return None
    order = first + [function for function in chosen if function not in first]
    ranked = [(function, function.name) for function in order]
    cards, cuts = library.Library.carded(ranked, card_bound(), head=CARDS_HEAD)
    kept, _ = library.Library.trimmed(ranked, card_bound(), head=CARDS_HEAD)
    shelves = list(dict.fromkeys(lib.shelf(function.shelf) for function in [*order, *picks]))
    shelves = [shelf for shelf in shelves if shelf is not None]
    lines = [SHELVES_HEAD]
    lines += [f"{shelf.name}" + (f" ({shelf.line})" if shelf.line else "") + ": "
              + ", ".join(function.name for function in shelf.functions) for shelf in shelves]
    others = [shelf.name for shelf in lib.shelves if shelf not in shelves]
    if others:
        lines.append(OTHERS_LINE.format(names=", ".join(others)))
    named = tuple(dict.fromkeys([*(f for shelf in shelves for f in shelf.functions), *order, *picks]))
    return Shaped("\n".join(lines), cards, tuple(cuts), tuple(card.function for card in kept), named, tuple(order))


def trims(section: str) -> list[str]:
    """The call's section (`shaped`'s index) with one more line given up at each step to make room for the cards
    whole, the last first: the other shelves' line, then the shelves' lines; the head kept."""
    lines = section.split("\n")
    return ["\n".join(lines[:kept]) for kept in range(len(lines) - 1, 0, -1)]


class Selection:
    """The call over one unit: what it picked, and the library as the picks shape it (`result`)."""

    def __init__(self) -> None:
        self.parsed: dict = {}
        self.picks: list[library.Function] = []
        self.completed: list[library.Function] = []     # the pointer's functions a cut reply's picks were completed by
        self.cut = False
        self.seconds: float | None = None
        self.reasoning = 0
        self.result: Shaped | None = None

    def call(self, loop, task, lib: library.Library, read, least: float, shown=None) -> Shaped | None:
        """The selection call for the loop `loop` (agent/n11/solve.py's N11) on `task`, read by `read`, where `FLOOR`
        requests and `FLOOR` times `least` seconds are left; the library as the picks shape it (`shaped`), a cut
        reply's completed by the pointer's (`completed`), or None where it was not sent, drew no reply, or named no
        function the library holds but the scaffolding and was not completed. `shown`, where given, rewrites the index
        the call shows."""
        requests, left = loop.requests_left(), loop.left()
        if requests < FLOOR or left < FLOOR * least:
            loop.log("select", "not sent: " + (f"{requests} request(s) left" if requests < FLOOR else
                                               f"{max(left, 0):.0f} s left"))
            return None
        index = lib.index(task=task.instruction)
        system, user = render(task.instruction.strip(), loop.described(), shown(index) if shown else index)
        began = time.monotonic()
        try:
            reply = loop.client.chat(system, loop.redact(user), max_tokens=MAX_TOKENS_LOW, thinking="low")
        except model.ModelError as exc:
            self.seconds = time.monotonic() - began
            loop.log("select", f"no reply ({type(exc).__name__}) after {self.seconds:.1f} s: the system prompt "
                               f"without the call")
            return None
        self.seconds = time.monotonic() - began
        self.cut, self.reasoning = reply.finish_reason == "length", int(reply.reasoning_tokens or 0)
        self.parsed = parse(reply.text, lib, read)
        self.picks = [found for found in map(lib.find, self.parsed["picks"]) if found is not None]
        self.completed = completed(self.picks, loop.fired) if self.cut else []
        chosen = [*self.picks, *self.completed]
        self.result = shaped(lib, chosen, loop.fired) if chosen else None
        unknown = self.parsed["unknown"]
        loop.log("select", f"low thinking, {MAX_TOKENS_LOW:,} tokens asked: "
                           f"{self.seconds:.1f} s, {int(reply.completion_tokens or 0):,} completion tokens"
                           + (f" ({self.reasoning:,} reasoning)" if self.reasoning else "")
                           + (", cut at its bound" if self.cut else "")
                           + ("; completed by the pointer's: " + ", ".join(function.name for function in self.completed)
                              if self.completed else "") + "; "
                           + ("it picks " + ", ".join(function.name for function in self.picks) if self.picks else
                              "it answered none" if self.parsed["none"] else "it names no function of the library")
                           + ("" if self.result or not self.picks else ", the scaffolding alone")
                           + (f" ({self.parsed['over']} more past the bound)" if self.parsed["over"] else "")
                           + ("; not in the library: " + ", ".join(unknown) if unknown else "")
                           + self.uncarded()
                           + (f"; {len(self.result.carded)} card(s) shown, {len(self.result.named)} function(s) named"
                              if self.result else "; the system prompt without the call"))
        return self.result

    @property
    def scaffolded(self) -> list[library.Function]:
        """The picks on the scaffolding's shelf (`scaffolding`): named, never carded."""
        return [function for function in self.picks if scaffolding(function)]

    def uncarded(self) -> str:
        """The log's words for the scaffolding picks left uncarded, where the picks hold any."""
        return ("; the scaffolding named, not carded: " + ", ".join(function.name for function in self.scaffolded)
                if self.scaffolded and self.result else "")
