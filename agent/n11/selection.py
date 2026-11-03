"""N11's selection call (lane select2; the owner's in494, in496 and in532, agenthon-t2): one bounded request before the
loop names the library functions a correct solution would call, and their cards and their shelves' names take the
index's place in the loop's system prompt.

`AGENTHON_N11_SELECT` (or agent/n11/SELECT, one word, an image's; agent/n11/solve.py `select_of`): `off`, today's, or
`call`, armed in the loop's forms `loop` and `sketch`, in form `state` (lane state4) and in form `review` (`FORMS`;
lane review2: there every unit's request 1 is then the loop's, and the review opens at the loop's first clean finish).
Armed, the unit's first request, where `FLOOR` requests
and as many of a request's seconds are left, is the selection call (`Selection.call`): `SYSTEM`, and the user turn
`render` writes, the task's instruction, its input files described as the loop's first message describes them, the
library's scored index (`Library.index(task=)`) and the ask (`ASK`), which states no count. Its `max_tokens` is the
harness's, `MAX_TOKENS` (room for the list and a short reasoning, never the House's cap), or `MAX_TOKENS_LOW` where
`AGENTHON_N11_SELECT_THINKING` (or agent/n11/SELECT_THINKING) says `low`, which asks the House's low-effort thinking as
the sketch's switch does. The reply is read by `parse` (solve.py's `sketch_names`, passed in: every `qf.` name and every
bare index name, looked up in the whole library, the `qf.` names it lacks counted and dropped, the first `MOST` kept).
A reply cut at its bound has its picks completed by the pointer's functions it did not name (`completed`, lane errors1;
the owner's in673 and in675), so a function the pointer named is never lost to a cut reply; nothing is said to the
model of the cut or of the completion.

The picks shape the system prompt (`shaped`): the cards of the pointer's functions, then the picks' (`CARDS_HEAD`), in
that order, within the call's own cards' bound (`card_bound`) under lane cards2's trim (`Library.carded`), a
scaffolding pick (`scaffolding`, lane libshelves1: a function of the library's scaffolding shelf, the I/O helpers
among them) never carded and counted (`select_scaffolding`);
and in the index's place (`SHELVES_HEAD`) the line of each shelf one of those functions or picks lies on with every
function it holds by name, then the other shelves' names (`OTHERS_LINE`), so a shelf-mate is one `lib` away (the
owner's in137). No index. A call that draws no reply, names no function the library holds but the scaffolding or
answers `none` leaves the system prompt as today's (the index and the pointer's cards), the request alone spent, but
where its reply was cut and the pointer named a function: the completion's picks shape it then. The completion
changes the system prompt only there: `shaped` cards the pointer's functions first whatever the picks.
Nothing of the call is said to the model in the loop. Form `state` (lane state4) composes the same: the section in its
system prompt's index place, and the cards, in the order `Shaped.order` keeps, in its CARDS section under the same
head and bound.

`AGENTHON_N11_SELECT_BOUND` (or agent/n11/SELECT_BOUND, an image's; lane first1, the owner's in693): `route`, today's,
or `short`: where the clock at the call buys fewer than the short-clock mode's requests at the measured pace, else the
platform's (agent/n11/solve.py `short_clock`, lane pace1's rule), the call asks `MAX_TOKENS` with thinking off whatever
`AGENTHON_N11_SELECT_THINKING` says, so it costs a short request (E134: a call at low effort ran 55 s to its bound);
otherwise as today. A cut reply is completed by the pointer's as above. Its keys, `BOUND_KEYS`.

scripts/select_read.py renders and parses its requests by this module, so the desk's grader reads the agent's own ask.

`given` (lane given1; the owner's in578, agenthon-t2), the desk's alone (`DESK_ONLY`): no request; the unit's list in
the picks file the harness mounts and names (`PICKS_ENV`; its `--picks`, a JSON of `{unit: [qf.<name>, ...]}` that
scripts/picks_file.py writes) is read as the call's reply is (`parse`) and shapes the system prompt as the call's
picks do (`Selection.given`), so the loop is read with retrieval fixed. A file not named or not read, no list for the
unit, or a list naming nothing the library holds but the scaffolding leaves today's prompt. An image's agent/n11/SELECT
saying `given` is refused at the start (agent/n11/solve.py `select_of`: `off` runs) and at the build
(agent/Dockerfile).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from common import runner, trace
from common.house import model
from n11 import library, measure

ENV = "AGENTHON_N11_SELECT"     # the desk's switch (src/agenthon_t1/harness.py hands it to an --agent run's container)
FILE = "SELECT"                 # or agent/n11/SELECT, one word, written by an image's builder
SETTINGS = ("off", "call", "given")     # the default first
DESK_ONLY = ("given",)          # never an image's: agent/n11/SELECT saying one is refused at the start and the build
REFUSED = "refused, the desk's alone"   # the source's words where it is (agent/n11/solve.py `select_of`)
PICKS_ENV = "AGENTHON_N11_SELECT_PICKS"     # under `given`, the picks file's path, named by the harness alone
HEADER = "#"                    # a picks file's entry naming its source and commit, no unit's
THINKING_ENV = "AGENTHON_N11_SELECT_THINKING"   # the call's thinking: `off`, or `low` (the House's low effort)
THINKING_FILE = "SELECT_THINKING"               # or agent/n11/SELECT_THINKING, an image's
THINKINGS = ("off", "low")
FORMS = ("loop", "sketch", "state", "review")     # the forms it is armed in (lane state4's `state`, review2's `review`)
KIND = "select"                 # its request's kind on the trace (common.house.Client.kind)
MOST = 8                        # the picks kept, at most: the harness's bound, never said; the model answers' gold is 4
#                                 a unit at the median and 11 at most, and the scored index shows 8 whole (lane select1)
MAX_TOKENS = 512    # the call's output with thinking off: E098's free-variant read answered in 84 tokens at the median
#                     (full index, thinking off), and the replies that ran on into prose or code (13 of 58, cut at the
#                     House's 4,000) had the list first: the bound keeps the list and drops the rest in seconds
MAX_TOKENS_LOW = 2_000  # with low-effort thinking, the reasoning counted in it: at the route's own (uncontrolled)
#                         effort E098 reasoned 1,220 and 1,123 tokens at the median (3,077 and 2,454 at the 90th
#                         percentile); low effort asks less. A reply cut while it still reasons names nothing, which
#                         costs the request and leaves today's prompt
FLOOR = 2           # the call is sent only with this many requests left and this many of a request's seconds
CARD_TOKENS = 3_300  # the cards' bound under the call in the model's tokens, its head included (lane select2's review):
#                      the index's room is freed, and the pointer's cards and the picks' run 2,137 to 2,386 tokens a
#                      unit whole at the median, 3,708 to 4,245 at the 90th percentile (E098's four records' picks).
#                      At the pointer's own bound (library.CARD_TOKENS, 1,650) a pointer card is cut on 23 to 32 units
#                      of 49 to 57 (12 of 49 today); at this one on 2 to 5, the system prompt 4,100 to 4,304 tokens a
#                      unit at the median (3,480 to 3,510 at 1,650; 8,259 today), the cards' recall unchanged
CARD_BYTES = 8_800   # the byte fallback where the tokenizer does not load: library.CARD_BYTES twice, as the tokens are
KEYS = ("select_mode", "select_thinking", "select_sent", "select_picks", "select_unknown", "select_scaffolding",
        "select_none", "select_cut", "select_completed", "select_seconds", "select_reasoning_tokens",
        "select_system_tokens", "select_system_today", "select_fallback", "select_source")
BOUND_ENV = "AGENTHON_N11_SELECT_BOUND"     # lane first1: the call's bound, `route` (today's: `MAX_TOKENS`, or
BOUND_FILE = "SELECT_BOUND"                 # `MAX_TOKENS_LOW` with thinking low) or `short` (`MAX_TOKENS` and thinking
BOUNDS = ("route", "short")                 # off where the clock at the call is short: the loop's `short_clock`); the
#                                             desk's switch, or agent/n11/SELECT_BOUND, one word, an image's
BOUND_KEYS = ("select_bound", "select_short")   # its setting (`-` outside `FORMS`) and whether the call was bounded so

SYSTEM = ("You are an expert quantitative developer. You read a task and the index of qf, a Python library of "
          "quantitative methods, and name the library's functions a correct solution of the task would call. You "
          "write no code.")
ASK = ("Name the few functions that matter for a correct solution of this task: the library's functions it would "
       "call where the task names their method or needs the quantity one computes. Where one function computes a "
       "quantity the task asks as a whole, name that function rather than the building blocks a solution could "
       "compute it from. Answer with the list alone, the one that matters most first, one function a line, written "
       "as below, and nothing else:\n\nqf.<name>\n\nWhere the library holds nothing the task needs, answer none.")
TASK_HEAD, DESCRIBED_HEAD, LIBRARY_HEAD, ASK_HEAD = "# Task", "# Input files, described", "# The library", "# The ask"
IO_HELPER = re.compile(r"^(?:read|write)_")     # an I/O helper's name (`helper`)
QUALIFIED = re.compile(r"(?<![\w.])qf(?:\.[A-Za-z_]\w*)+")     # a `qf.` name: agent/n11/solve.py QF_NAMED's
NONE = re.compile(r"^[\W_]*none[\W_]*$", re.I)                  # a reply of `none` alone
CARDS_HEAD = ("Cards of the library's functions this task needs, as a reading of the task named them; where one "
              "computes a quantity the task asks, call it rather than writing its formula:")
SHELVES_HEAD = ("qf, the method library (import qf; numpy and scipy; every convention a required argument). The "
                "shelves (modules) of the functions carded below, each with every function it holds by name; `lib "
                "name=<function>` shows a function's card and `lib shelf=<module>` a shelf's, these shelves or the "
                "others:")
OTHERS_LINE = "the other shelves: {names}"


def render(instruction: str, described: str, index: str, extra: str = "", ask: str = ASK) -> tuple[str, str]:
    """The call's system and user turns: the task, its input files described (where it has any), the index, `extra`
    (a desk variant's section, scripts/select_read.py's candidates), then the ask."""
    user = f"{TASK_HEAD}\n\n{instruction}\n\n" + (f"{DESCRIBED_HEAD}\n\n{described}\n\n" if described else "")
    user += f"{LIBRARY_HEAD}\n\n{index}\n\n" + extra
    return SYSTEM, user + f"{ASK_HEAD}\n\n{ask}\n"


def parse(text: str, lib: library.Library, read) -> dict:
    """A reply's picks: `read` (solve.py `sketch_names`: each `qf.` name and each bare index name, first named first,
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


def helper(function: library.Function) -> bool:
    """An I/O helper (ROW 227): a function of a shelf the pointer never points to (`library.UNPOINTED`, conventions and
    I/O) whose name reads or writes a file (`IO_HELPER`). Every task reads its inputs and writes its outputs, so a
    selection names these often and first, harmlessly for what it finds (no gold set holds them, E098's four reads):
    the grade's `--no-io` sets them aside (scripts/library_reach.py `io_helpers`); `scaffolding` holds them too."""
    return function.shelf in library.UNPOINTED and IO_HELPER.match(function.name) is not None


def scaffolding(function: library.Function) -> bool:
    """A scaffolding function (lane libshelves1; docs/research/library-design-1011.md, "The library's identity in the
    index"): one on the library's scaffolding shelf, read from its own shelf (`library.UNPOINTED`: files, returns,
    annualising, day counts, compounding, windows and the statistics whose convention a task names), never from a list
    of names. A selection names these in about four picks of ten, beside the composite a task wants, at the cost of the
    cards' room (lane libframe1's review: 13 to 24% of a system prompt): a pick of one stays named on its shelf's line
    where that shelf is listed, never carded, and counted (`select_scaffolding`), as the I/O helpers among them were."""
    return function.shelf in library.UNPOINTED


def picks_for(path: str, unit: str) -> tuple[list[str] | None, str, bool]:
    """Under `given`: the unit `unit`'s list in the picks file at `path` (`PICKS_ENV`'s), empty or not, with "" and
    False; else None, why, and whether it is the desk's fault (no file named, or one that is no `{unit: [name, ...]}`)
    rather than a file holding no list for the unit."""
    if not path:
        return None, f"no picks file named ({PICKS_ENV} unset)", True
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"the picks file does not read ({type(exc).__name__})", True
    if not isinstance(doc, dict):
        return None, "the picks file is no object of lists", True
    found = doc.get(unit) if unit != HEADER else None
    if found is None:
        return None, "the picks file holds no list for the unit", False
    if not (isinstance(found, list) and all(isinstance(name, str) for name in found)):
        return None, "the unit's entry in the picks file is no list of names", True
    return found, "", False


@dataclass(frozen=True)
class Shaped:
    """The system prompt's library under the call: the section in the index's place, the cards and what their trim
    cut (a line a card, for the log), the functions whose cards are shown, the functions the section names, and the
    functions carded in their order before the trim (the pointer's, then the picks' but the scaffolding's: form
    `state`'s CARDS base, lane state4)."""
    index: str
    cards: str
    cuts: tuple[str, ...]
    carded: tuple[library.Function, ...]
    named: tuple[library.Function, ...]
    order: tuple[library.Function, ...] = ()


def shaped(lib: library.Library, picks: list[library.Function],
           pointed: list[tuple[library.Function, str]]) -> Shaped | None:
    """The library as the system prompt shows it under the call (above), None where the picks hold nothing but the
    scaffolding (`scaffolding`): the pointer's functions first, then each pick the pointer lacks but the scaffolding's,
    carded within the call's cards' bound (`card_bound`; `Library.carded` under `CARDS_HEAD`: lane cards2's trim by
    rank, the last carded cut first); the shelves those functions and the picks lie on, in the order first named, each
    line with every function it holds, a scaffolding pick's shelf among them; the other shelves by name."""
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
    """Lane first1 (agent/n11/solve.py `made_whole`): the call's section (`shaped`'s index) with one more line given up
    at each step to make room for the cards whole, the last first: the other shelves' line, those no carded function or
    pick lies on (`shaped` writes it last), then the shelves' lines; the head kept."""
    lines = section.split("\n")
    return ["\n".join(lines[:kept]) for kept in range(len(lines) - 1, 0, -1)]


class Selection:
    """The switch over one unit: armed (`call`, in one of `FORMS`), what the call did, and its keys."""

    def __init__(self, setting: str, source: str, thinking: str, thinking_source: str, form: str,
                 bound: str = BOUNDS[0], bound_source: str = "no setting named") -> None:
        self.setting, self.source = setting, source
        self.thinking, self.thinking_source = thinking, thinking_source
        self.bound, self.bound_source = bound, bound_source     # lane first1's switch (`BOUNDS`) and its source
        self.short = False                                      # the call sent at `MAX_TOKENS`, thinking off, under it
        self.form = form
        self.armed = setting in ("call", "given") and form in FORMS
        self.sent = False
        self.listed = False     # under `given`: the picks file consulted for the unit, no request sent
        self.parsed: dict = {}
        self.picks: list[library.Function] = []
        self.completed: list[library.Function] = []     # the pointer's functions a cut reply's picks were completed by
        self.cut = False
        self.seconds: float | None = None
        self.reasoning = 0
        self.failed = False
        self.result: Shaped | None = None
        self.system_tokens: int | None = None   # the system prompt as sent, and today's, where the call was sent
        self.system_today: int | None = None

    @property
    def carded(self) -> set[str]:
        """The functions whose cards the system prompt shows by the call (none where it picked nothing)."""
        return {function.name for function in self.result.carded} if self.result is not None else set()

    @property
    def consulted(self) -> bool:
        """The call sent, or the given list read: the system prompt's size as sent and today's are counted."""
        return self.sent or self.listed

    @property
    def thinks(self) -> bool:
        """The call asks the House's low effort: `low` named, and the call not bounded by a short clock (`short`)."""
        return self.thinking == "low" and not self.short

    def max_tokens(self) -> int:
        return MAX_TOKENS_LOW if self.thinks else MAX_TOKENS

    def call(self, loop, task, lib: library.Library, read, least: float, bounded: dict | None = None,
             shown=None) -> Shaped | None:
        """The selection call for the loop `loop` (agent/n11/solve.py's N11) on `task`, read by `read`, where armed and
        `FLOOR` requests and `FLOOR` times `least` seconds are left, under the client's keyword `bounded` (lane
        fallback2's bound while the last resort's reserve is held, so the call never runs into it); the library as the
        picks shape it (`shaped`), a cut reply's completed by the pointer's (`completed`), or None where it was not
        sent, drew no reply, or named no function the library holds but the scaffolding and was not completed. Under
        `given` the unit's given list in its place (`given`), no request sent. `shown`, where given, says the index the
        call shows (lane clean1: without the lookup the call never offers)."""
        if not self.armed:
            if self.setting != "off":
                loop.log("select", f"{self.setting} ({self.source}), not armed in form {self.form}")
            elif REFUSED in self.source:
                loop.log("select", f"off ({self.source})")
            return None
        if self.setting == "given":
            return self.given(loop, task, lib, read)
        requests, left = loop.requests_left(), loop.left()
        if requests < FLOOR or left < FLOOR * least:
            loop.log("select", "not sent: " + (f"{requests} request(s) left" if requests < FLOOR else
                                               f"{max(left, 0):.0f} s left"))
            return None
        if self.bound == "short":               # lane first1: the short clock's bound, read as the call is sent
            self.short, said = loop.short_clock()
            loop.log("select", f"bound short ({self.bound_source}): {said}")
        index = lib.index(task=task.instruction)
        system, user = render(task.instruction.strip(), loop.described(), shown(index) if shown else index)
        client = loop.client
        client.kind = KIND
        self.sent = True
        began = time.monotonic()
        try:
            reply = client.chat(system, loop.redact(user), max_tokens=self.max_tokens(),
                                thinking="low" if self.thinks else None, **(bounded or {}))
        except model.ModelError as exc:
            self.seconds, self.failed = time.monotonic() - began, True
            loop.log("select", f"no reply ({type(exc).__name__}) after {self.seconds:.1f} s: the system prompt as "
                               f"today's")
            trace.event("select", sent=True, replied=False)
            return None
        self.seconds = time.monotonic() - began
        self.cut, self.reasoning = reply.finish_reason == "length", int(reply.reasoning_tokens or 0)
        self.parsed = parse(reply.text, lib, read)
        self.picks = [found for found in map(lib.find, self.parsed["picks"]) if found is not None]
        self.completed = completed(self.picks, loop.fired) if self.cut else []
        chosen = [*self.picks, *self.completed]
        self.result = shaped(lib, chosen, loop.fired) if chosen else None
        unknown = self.parsed["unknown"]
        loop.log("select", ("thinking off (the short clock), " if self.short else
                            f"thinking {self.thinking} ({self.thinking_source}), ")
                           + f"{self.max_tokens():,} tokens asked: "
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
                              if self.result else "; the system prompt as today's"))
        trace.event("select", sent=True, replied=True, picks=[function.name for function in self.picks],
                    unknown=unknown, none=self.parsed["none"], cut=self.cut,
                    carded=sorted(self.carded), named=len(self.result.named) if self.result else 0,
                    **({"completed": [function.name for function in self.completed]} if self.cut else {}))
        return self.result

    def given(self, loop, task, lib: library.Library, read) -> Shaped | None:
        """Under `given` (the desk's alone, lane given1): the unit's list (`picks_for`, the unit named by its card's
        `[task] id`) read as the call's reply is (`parse`, by `read`) and the library as its picks shape it
        (`shaped`), or None where the file gives no list, the list is empty, or it names no function the library
        holds but the scaffolding; a file not named or not read is counted as the call's no reply is
        (`select_fallback`). No request is sent."""
        unit = str((task.card.get("task") or {}).get("id") or "")
        names, why, self.failed = picks_for(os.environ.get(PICKS_ENV, ""), unit)
        self.listed = True
        self.parsed = parse("\n".join(names or []), lib, read)
        self.picks = [found for found in map(lib.find, self.parsed["picks"]) if found is not None]
        self.result = shaped(lib, self.picks, loop.fired) if self.picks else None
        unknown = self.parsed["unknown"]
        loop.log("select", f"given ({self.source}): "
                           + (why if names is None else
                              "the unit's list is empty" if not names else
                              "it picks " + ", ".join(function.name for function in self.picks) if self.picks else
                              "it names no function of the library")
                           + ("" if self.result or not self.picks else ", the scaffolding alone")
                           + (f" ({self.parsed['over']} more past the bound)" if self.parsed["over"] else "")
                           + ("; not in the library: " + ", ".join(unknown) if unknown else "")
                           + self.uncarded()
                           + (f"; {len(self.result.carded)} card(s) shown, {len(self.result.named)} function(s) named"
                              if self.result else "; the system prompt as today's"))
        trace.event("select", sent=False, given=True, listed=names is not None,
                    picks=[function.name for function in self.picks], unknown=unknown, carded=sorted(self.carded),
                    named=len(self.result.named) if self.result else 0)
        return self.result

    @property
    def scaffolded(self) -> list[library.Function]:
        """The picks on the scaffolding's shelf (`scaffolding`): named, never carded."""
        return [function for function in self.picks if scaffolding(function)]

    def uncarded(self) -> str:
        """The log's words for the scaffolding picks left uncarded, where the picks hold any."""
        return ("; the scaffolding named, not carded: " + ", ".join(function.name for function in self.scaffolded)
                if self.scaffolded and self.result else "")

    def measured(self, sent: int, today: int) -> None:
        """The system prompt's size as sent and today's, in the bounds' unit (agent/n11/measure.py)."""
        self.system_tokens, self.system_today = sent, today

    def keys(self) -> dict:
        """The summary line's keys (`KEYS`): `-` where the form is not one of `FORMS` or the call was not sent;
        `select_source`, what gave the picks: `call`, `given` (the picks file) or `-`; `select_scaffolding`, the picks
        on the scaffolding's shelf, never carded (lane libshelves1); `select_completed`, the pointer's functions a cut
        reply's picks were completed by (lane errors1), never counted in `select_picks`, the reply's own; and lane
        first1's `BOUND_KEYS`: the bound's setting, and whether the call was sent at `MAX_TOKENS` with thinking off
        for a short clock (`select_thinking` stays the setting named)."""
        sent = self.sent
        return {"select_mode": self.setting if self.form in FORMS else "-",
                "select_thinking": self.thinking if sent else "-", "select_sent": int(sent),
                "select_picks": len(self.picks), "select_unknown": len(self.parsed.get("unknown", [])),
                "select_scaffolding": len(self.scaffolded),
                "select_none": int(bool(self.parsed.get("none"))), "select_cut": int(self.cut),
                "select_completed": len(self.completed),
                "select_seconds": f"{self.seconds:.1f}" if self.seconds is not None else "-",
                "select_reasoning_tokens": self.reasoning,
                "select_system_tokens": self.system_tokens if self.system_tokens is not None else "-",
                "select_system_today": self.system_today if self.system_today is not None else "-",
                "select_fallback": int(self.failed),
                "select_source": "given" if self.listed else "call" if sent else "-",
                "select_bound": self.bound if self.form in FORMS else "-", "select_short": int(self.short)}
