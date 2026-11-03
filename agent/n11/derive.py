"""The dated derivation as a switch (lane derive1; the reasoning read runs/lanes/codex-redesign-reasoning.md item 2,
the wrong-answer trace read runs/lanes/codex-traces-wrong.md, docs/research/redesign-1016.md): `AGENTHON_N11_DERIVE`
or agent/n11/DERIVE, `off` (today's) or `on`, armed in forms `loop`, `review` and `sketch` (agent/n11/solve.py
`derive_armed`).

Under `on` the loop's first message closes on `ASK`, a paragraph of its own after any clause: the first reply opens
with a short derivation naming, for each quantity the task asks for, its information date, its window's endpoints,
its payment or activation date and the state it is computed from, and in the same reply implements those steps as
small functions in one script that keeps labelled indexes between stages and computes every reported figure from the
objects those steps build, ending on its own checks of its stated steps as assertions that run. No request is added:
the derivation and the code come in one reply, no module is prescribed, and no clean finish is answered with a search
for mistakes (the closed plan step, scaffold and verify turn). After each run that exits 0 the harness's own checks
(agent/n11/witness.py, `witness.KEPT`) are said in the run's observation (agent/n11/commands.py `Workspace.observe`).
No text states a count; under `off` every body is today's.

The readings, under either setting (they change no text): `opened`, whether a reply's text before its first code
opens with a derivation; `checks`, the assertions a script holds; `functions`, its top-level functions. Keys `KEYS`.
"""
from __future__ import annotations

import re

ENV = "AGENTHON_N11_DERIVE"
FILE = "DERIVE"
SETTINGS = ("off", "on")
FORMS = ("loop", "review", "sketch")
KEYS = ("derive_mode", "derive_said", "derive_checks", "derive_functions", "witness_runs", "witness_findings",
        "witness_kinds")
ASK = ("Open your first reply with a short derivation, before any code: for each quantity the task asks for, name "
       "its information date (the last date whose data it may use), the first and last dates of its window, the date "
       "it is paid or becomes active, and the state it is computed from (which holdings, which book, which cash-flow "
       "period). Then, in the same reply, write the script as small functions, one for each step of that "
       "derivation: keep the dates as labelled indexes from stage to stage, and compute every reported figure from "
       "the objects those steps build (the signal, the holdings path, the cash-flow table), never from a second "
       "calculation of its own. End the script with your own checks of the steps you stated, as assertions that run: "
       "a window's endpoints, an activation date, a period's pairing, a total rebuilt from its parts.")
ASPECTS = (     # the derivation's four parts, by their words: a reply naming two or more opens with one
    re.compile(r"information date|as[- ]of|observ\w* date|known (?:at|on|by)|signal date|formation|decision date|"
               r"data (?:up )?(?:to|through)|last (?:usable |available )?date", re.I),
    re.compile(r"window|look-?back|endpoints?|first and last|from \S+ (?:to|through) \S+|start\w* and end\w*|"
               r"trailing", re.I),
    re.compile(r"activ\w+|effective|payment|paid|pays|settle\w*|next (?:trading |business )?day|rebalanc\w*", re.I),
    re.compile(r"holdings?|\bbook\b|portfolio|\bstate\b|positions?|cash[- ]?flows?|principal|\bweights?\b", re.I),
)
MIN_ASPECTS = 2             # the parts a derivation names, at least: the harness's reading, never said
MIN_CHARS = 40              # and its prose's characters
FENCE = "```"
ASSERT = re.compile(r"^\s*assert\b", re.M)
FUNCTION = re.compile(r"^(?:async\s+)?def\s+\w+", re.M)
LEADING = re.compile(r'\A\s*(?:(?:#[^\n]*\n\s*)+|("""|\'\'\')(.*?)\1)', re.S)


def prose(text: str) -> str:
    """A reply's text before its first fenced block (the whole text where it holds none)."""
    return (text or "").split(FENCE, 1)[0]


def opened(text: str, script: str = "") -> bool:
    """Whether a reply opens with a derivation: its text before its first code (`prose`), or where that is empty the
    leading comments or docstring of the script it writes, holds `MIN_CHARS` or more and names `MIN_ASPECTS` or more
    of the four parts (`ASPECTS`: information dates, windows, payment or activation, state)."""
    said = prose(text).strip()
    if not said and script:
        found = LEADING.match(script)
        said = (found.group(0) if found else "").strip()
    return len(said) >= MIN_CHARS and sum(bool(aspect.search(said)) for aspect in ASPECTS) >= MIN_ASPECTS


def checks(source: str) -> int:
    """The assertions a script holds: its `assert` statements, read by line (a cut script's too)."""
    return len(ASSERT.findall(source or ""))


def functions(source: str) -> int:
    """The top-level functions a script defines, read by line."""
    return len(FUNCTION.findall(source or ""))
