"""Edits the loop's system prompt and first message before they are sent.

The prompt's base text is in agent/n11/solve.py (`SYSTEM`, `FENCED_PART`). This module splits it into paragraphs and
applies a fixed list of text edits, each anchored on a phrase that must occur exactly once.

Called by: agent/n11/solve.py (`N11.worded`, `N11.first_turn`, `N11.cleaned`).
Start at: `clean_parts`.

The rules in detail
-------------------
The loop's system prompt as sent: the base prompt (agent/n11/solve.py `N11.system`) read in its paragraphs
(`Control`) and edited (`clean_parts`, `clean_text`).

The edits remove phrases the rest of the prompt already says or contradicts: the cut paragraph says what a cut reply
leads to, the library paragraph is scoped to the quantities a card computes, `finish` requests completion rather than
ending the task, the defaults' timing lines move into step 1, and the must list comes before the runtime notes. Each
edit is made only where its anchor text stands exactly once; an edit that cannot be made leaves its paragraph as it
was. The first message's own edits (`CLEAN_PATHS`, `CLEAN_POINTER`, `repeated_portraits`) are made by
agent/n11/solve.py.
"""
from __future__ import annotations

from dataclasses import dataclass


OPENINGS = ("You are an expert", "", "How to work:", "A reply that runs past", "The runtime", "Unless the task says "
            "otherwise:", "Every script must:", "", "The library:")    # `Control`'s paragraphs, each as it opens


@dataclass(frozen=True)
class Control:
    """The base system prompt in its paragraphs: the head, the reply form's paragraph, How to work, the cut paragraph,
    the runtime, the defaults, the must list, the example, the library paragraph, the index and the cards. `read`
    gives None where the text is not in these paragraphs."""
    head: str
    tools: str
    how: str
    cut: str
    api: str
    defaults: str
    must: str
    example: str
    library: str
    index: str
    cards: str

    @classmethod
    def read(cls, system: str, cards: str) -> Control | None:
        tail = f"\n\n{cards}" if cards else ""
        if not system.endswith(tail):
            return None
        parts = system[:len(system) - len(tail)].split("\n\n", len(OPENINGS))
        if len(parts) != len(OPENINGS) + 1 or any(not part.startswith(opening) for part, opening in
                                                  zip(parts, OPENINGS)):
            return None
        return cls(*parts, cards)


CLEAN_PATHS = (", and nothing anywhere else but /tmp", "")  # the Paths section's (agent/common/paths.py): the must
#                                                              list allows work/ beside /tmp, and work/main.py is there
CLEAN_STEP_CALL = (" (a quantity the library computes is a call of its function)", "")
CLEAN_LIBRARY_LAST = "Write a formula yourself only where the task states one that differs from the card's."
CLEAN_SCOPE = ("These rules hold for a quantity a card computes by the method the task names; the rest of the script "
               "(reading and preparing the inputs, every quantity no card computes) they leave to you.")
CLEAN_CARDS = ("; where one computes a quantity the task asks, call it rather than writing its formula:", ":")
CLEAN_POINTER = (": call each for its quantity rather than writing its formula", "")
CLEAN_CUT = (   # the cut as agent/n11/solve.py handles it: a cut file continued (`hold`), else split (`salvage`)
    ("A reply that runs past the output limit is cut, and a write the cut falls in is not saved, only its complete "
     "definitions as a draft.",
     "A reply that runs past the output limit is cut: a file the cut falls in is kept up to its last complete line, "
     "and the next message asks for the rest of it; where nothing of it is complete, or replies keep being cut, the "
     "next message asks you to split the script instead."),
    ("; after a cut, write the rest one file a reply.", "."))
CLEAN_STEP_TWO = (", that reads the inputs, computes each quantity and writes every output file the task asks for.",
                  ".")
CLEAN_STEP_THREE = ("3. A write of work/main.py runs it: read each run.", "3. Read each run.")
CLEAN_LAGS = "Settle the lags the task states the same way; where it states none:"
CLEAN_START = " Start by writing code"      # step 1's last sentence: the lags go before it
CLEAN_OPENER = ("You act through the form of your reply. ", "")
CLEAN_NETWORK = (", with no network", "")   # the head's: the must list's "no network, no pip, no subprocess" stays
CLEAN_COMPLETION = ("`finish` ends the task, and nothing else does.",
                    "`finish` requests completion; follow any refusal.")
CLEAN_ORDER = ("head", "tools", "how", "cut", "must", "example", "api", "defaults", "library", "index")   # then the
#                                                                     cards; the example empty where the edit is made


def made(parts: dict[str, str | None], edits: tuple[tuple[str, str, str], ...]) -> dict[str, str | None] | None:
    """`parts` with each `(paragraph, old, new)` of `edits` made in turn, or None where an `old` does not stand exactly
    once in its paragraph: an edit is made whole or not at all."""
    parts = dict(parts)
    for name, old, new in edits:
        if parts[name] is None or parts[name].count(old) != 1:
            return None
        parts[name] = parts[name].replace(old, new, 1)
    return parts


def lagged(timing: tuple[str, ...]) -> str:
    """Step 1's lags before its last sentence (`CLEAN_START`): `CLEAN_LAGS`, then `timing`, the defaults' timing
    lines, each as it stands but the last's `;` a `.`, indented as the step's own lines."""
    lines = [f"   {line}" for line in timing]
    lines[-1] = lines[-1].removesuffix(";") + "."
    return "\n".join([f" {CLEAN_LAGS}", *lines, f"  {CLEAN_START}"])


def clean_tools(part: str) -> str:
    """The reply form's paragraph without its empty opener (`CLEAN_OPENER`) and with `finish` said to request
    completion (`CLEAN_COMPLETION`), each where it stands once."""
    for old, new in (CLEAN_OPENER, CLEAN_COMPLETION):
        part = part.replace(old, new, 1) if part.count(old) == 1 else part
    return part


def clean_parts(control: Control, timing: tuple[str, ...], example: str) -> tuple[dict[str, str | None], list[str]]:
    """The system prompt's paragraphs (`control`) with the edits made where their anchors stand (`made`), and the
    names of those not made. `timing` is the defaults' timing lines (moved to step 1), `example` the reply form's
    example (left out where the paragraph is it)."""
    parts: dict[str, str | None] = dict(control.__dict__)
    table = (
        ("library", (("how", *CLEAN_STEP_CALL),
                     ("library", CLEAN_LIBRARY_LAST, f"{CLEAN_LIBRARY_LAST} {CLEAN_SCOPE}"))),
        ("cards", (("cards", *CLEAN_CARDS),) if control.cards else ()),
        ("cut", tuple(("cut", *edit) for edit in CLEAN_CUT)),
        ("writing", (("how", *CLEAN_STEP_TWO), ("how", *CLEAN_STEP_THREE), ("example", example, ""))),
        ("lags", (("how", CLEAN_START, lagged(timing)), ("defaults", "\n" + "\n".join(timing), ""))),
        ("opener", (("tools", *CLEAN_OPENER),)),
        ("network", (("head", *CLEAN_NETWORK),)),
        ("finish", (("tools", *CLEAN_COMPLETION),)))
    missed = []
    for name, edits in table:
        found = made(parts, edits)
        if found is None:
            missed.append(name)
        else:
            parts = found
    return parts, missed


def clean_text(parts: dict[str, str | None]) -> str:
    """The system prompt: its paragraphs (`clean_parts`) in `CLEAN_ORDER`, an empty one (the example the writing edit
    took out) left out, then the cards."""
    text = "\n\n".join(parts[name] for name in CLEAN_ORDER if parts[name])
    return text + (f"\n\n{parts['cards']}" if parts["cards"] else "")


def repeated_portraits(described: str, data: str, portraits: list[tuple[str, int, int, int, str]]
                       ) -> tuple[str, list[str]]:
    """`described` (the first message's input files described) without each file's portrait that repeats what
    `data` (its input data) shows: every line after its first stands as a line there (a JSON file's key tree said
    twice); a portrait holding a line the data lacks (a table's statistics, a deeper tree) is kept. With the files
    left out, by name. `portraits` is agent/n11/commands.py `portraits`' reading of `described`."""
    shown = {line.strip() for line in data.splitlines()} - {""}
    spans, dropped = [], []
    for name, at, start, end, separator in portraits:
        body = [line.strip() for line in described[start:end].split("\n")[1:] if line.strip()]
        if body and all(line in shown for line in body):
            after = at >= len(separator) and described[at - len(separator):at] == separator
            spans.append((at - len(separator) if after else at, end))
            dropped.append(name)
    for left, right in reversed(spans):
        described = described[:left] + described[right:]
    return (described.strip("\n") if dropped else described), dropped
