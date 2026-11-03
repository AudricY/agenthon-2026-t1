"""The audit on the first clean run as a switch (lane audit1; the owner's in90 and in91; RULING ROW 605; the cold
reader's design D, runs/lanes/cold-prompt1017.md "Design D"; the failure reads' cause 1, docs/research/failures-1017.md:
a clean run taken as validation).

A run's checks test the output files' form (each named file present at its path, parsing as its kind, its numbers
finite, none holding a canary, no failure the output reports of itself: agent/n11/commands.py `Workspace.checks`) and
its sanity findings a few signs and ranges (agent/n11/sanity.py); the hidden tests test the values. The audit asks the
House itself, once a unit, at its first clean run, the task's own questions, each answered by a quote of the task's
words beside the line of code that applies them:

- the section (`section`): after the observation of the unit's first clean run in the loop, under `HEAD`, the answer
  form (`INTRO`), the two fixed items (`FIXED`: the outputs named identically beside the digest's names; the method's
  sentence quoted and the lines computing it named) and at most `TRIGGERED` of `ITEMS`, each switched on by its words
  in the task's text (case-insensitive but `VaR`), in `ITEMS`' order;
- the hold: a finish in the reply whose run carried the section turned away once (`HELD`), through the finish guards'
  one shared refusal (agent/n11/solve.py `guards_turn`, `audit_turn`);
- the step (`STEP`): the system prompt's fifth step (`TODAY`) said the audit's way (agent/n11/solve.py `system`), the
  template's text unchanged;
- the answers (`answers`): the reply after the section read for its lines in the answer form, those whose first field
  quotes the task verbatim (whitespace and case normalised), and those answered FIX.

The switch: `AGENTHON_N11_AUDIT` or agent/n11/AUDIT, `off` (today's) or `on`, armed in forms `loop`, `review` and
`sketch` (agent/n11/solve.py `audit_armed`), on the loop's runs alone (form `review`'s request 1 and form `state`
build their own). Skipped where under `FLOOR_S` seconds of the loop's clock or no request remain, the skip logged,
traced and counted, never said. No text states a count (the owner's in281): `TRIGGERED` and `FLOOR_S` are the
harness's. Under `off` every body is today's. Keys `KEYS`.

Pure: the standard library alone, nothing written, no unit named (rule 8).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ENV = "AGENTHON_N11_AUDIT"
FILE = "AUDIT"
SETTINGS = ("off", "on")
FORMS = ("loop", "review", "sketch")    # the loop's forms (form `state` builds its own run text and finish)
KEYS = ("audit_mode", "audit_shown", "audit_items", "audit_answered", "audit_quoted", "audit_fix", "audit_rewrote",
        "audit_held", "audit_skipped")
MAX = 1                 # the finishes the audit holds a unit; the next is today's
FLOOR_S = 120.0         # the loop's seconds left under which the first clean run is not audited (the design's), never said
TRIGGERED = 4           # the triggered items said, at most (the design's), never said

TODAY = ("5. Read each run's digest of the output files against what the task says of each value (its sign, its range, "
         "its order: ES at least VaR, a probability in [0, 1]); finish when the checks pass and the values hold.")
FINISHED = {"tools": "call `finish`", "fenced": "write `finish`"}     # each reply form's finish
FORM_LINE = "the task's words, quoted | the line of your code that applies them | OK or FIX"
STEP = {mode: ("5. A clean run is not a right answer: the checks in the next message test the files' form, the sanity "
               "findings a few signs and ranges, and the hidden tests the values. After your first clean run the next "
               "message asks a few questions about this task's conventions. Answer each on one line: "
               f"{FORM_LINE}. Then rewrite each file a FIX needs, whole, or {finish}.")
        for mode, finish in FINISHED.items()}
HEAD = "# Audit"
INTRO = {mode: ("Your run is clean, and a clean run does not test the values. Answer each question on one line: "
                f"{FORM_LINE}. Then rewrite each file a FIX needs, whole, or {finish}.")
         for mode, finish in FINISHED.items()}
HELD = "Your `finish` is held until you answer."
FIXED = ("Outputs: each file and each field or column the task names, beside the name your digest shows. Identical, "
         "character for character?",
         "Method: quote the sentence that names the method. Which function or lines compute it: that method, or a near "
         "relative (another estimator, another variant, a card that does something close)?")


STEMS = frozenset(("rebalanc", "annuali", "accru", "correlat", "regress", "covarian"))
SUFFIXES = {"loss": "(?:es)?", "trade": "[sd]?", "backtest": "(?:s|ed|ing)?",
            "forecast": "(?:s|ed|ing)?", "predict": "(?:s|ed|ing|ion|ions|ive)?",
            "train": "(?:s|ed|ing)?", "fit": "(?:s|ted|ting)?",
            "compound": "(?:s|ed|ing)?", "discount": "(?:s|ed|ing)?", "yield": "(?:s|ed|ing)?",
            "round": "(?:s|ed|ing)?", "percent": "(?:s|age|ages)?"}


def words(*terms: str) -> str:
    """Bound words and their inflections, retaining the design's explicit stems and snake_case components.
    Spaces and hyphens interchange; numbers are whole, and `%` matches anywhere."""
    said = []
    for term in terms:
        if term == "%":
            said.append("%")
        elif term[0].isdigit():
            said.append(rf"(?<![\d.]){re.escape(term)}(?!\.?\d)")
        else:
            pattern = r"[\s-]+".join(re.escape(part) for part in re.split(r"[\s-]+", term))
            suffix = "[a-z]*" if term in STEMS else SUFFIXES.get(term, "s?")
            if term in ("strategy", "maturity", "volatility"):
                pattern, suffix = pattern[:-1], "(?:y|ies)"
            if term == "sample":
                pattern, suffix = r"(?<!out[-\s]of[-\s])sampl", r"(?:e[sd]?|ing)"
            if term == "act/":
                suffix = ""  # the day-count denominator follows the slash
            pattern = r"(?<![a-z])" + pattern + suffix + r"(?![a-z])"
            if term == "return":
                pattern += (r"(?!(?:\s+(?:the|a|an))?\s+"
                            r"(?:result|answer|output|value|object|file|json|csv)\b)")
            said.append(pattern)
    return "|".join(said)


VAR = r"(?-i:(?<![A-Za-z])VaR(?![A-Za-z]))"     # the one case-sensitive word: never `var`, `variance` or `CVaR`
ITEMS = tuple((name, re.compile(pattern, re.I), text) for name, pattern, text in (     # the design's priority order
    ("timing", words("signal", "position", "trade", "backtest", "strategy", "rebalanc"),
     "Timing: the position held over bar t uses data up to which bar? Quote the line that shifts."),
    ("training", words("forecast", "predict", "train", "fit", "out-of-sample", "walk-forward", "expanding"),
     "Training rows: does the fit for row t end at row t-1? Quote the line."),
    ("window", words("rolling", "window", "lookback", "trailing", "moving"),
     "Window: its length in rows, whether it holds the current row, and min_periods, as the task says?"),
    ("sign", f"{VAR}|" + words("value at risk", "shortfall", "drawdown", "loss"),
     "Sign: is each loss reported with the sign the task asks?"),
    ("annual", words("annuali", "per annum", "252", "365", "360"),
     "Annualisation: the factor and the form (xN, x sqrt N, (1+r)^N - 1)?"),
    ("ddof", words("standard deviation", "std", "volatility", "variance", "sample", "ddof"),
     "ddof at each std or var: which one, and what does the task say?"),
    ("returns", words("return"),
     "Returns: log or simple? Is the first row's NaN dropped as the task says?"),
    ("rates", words("compound", "continuous", "discount", "yield", "accru", "day count", "act/", "30/360", "coupon",
                    "maturity"),
     "Rates and dates: compounding frequency, day count, and are input rates in percent or fraction?"),
    ("intercept", words("correlat", "regress", "beta", "ols", "covarian"),
     "Are NaN rows dropped pairwise, and is the intercept as the task says?"),
    ("calendar", words("trading day", "business day", "calendar", "holiday"),
     "Calendar: dates from the data, or the calendar the task names?"),
    ("units", words("percent", "%", "bps", "basis point", "round", "decimal"),
     "Units and rounding: percent, fraction or bps, and rounding only at output, to the places asked?"),
))
NAMES = tuple(name for name, _, _ in ITEMS)

FENCE = re.compile(r"^\s*(?:```|~~~)")          # a fenced block's line: the lines inside are code, never an answer
VERDICT = re.compile(r"[\s*_`'\"\[(]*(OK|FIX)(?![\w/])", re.I)   # an answer's last field opens on its verdict
BOTH = re.compile(r"\s*(?:/|or\b)", re.I)       # and goes on into the form's own words (`OK or FIX`): no answer
LEAD = re.compile(r"^\s*(?:[-*•>]+\s*)?(?:\(?\d+[.):]\s*)?")    # an answer's bullet or number
LABEL = re.compile(r"^(?:Outputs?|Method|Timing|Training rows|Window|Sign|Annualisation|Annualization|ddof|Returns?|"
                   r"Rates and dates|Calendar|Units and rounding|NaN and intercept):\s+", re.I)
EDGES = " \t'\"“”‘’`*_.,;:"


def triggered(task: str) -> tuple[str, ...]:
    """The triggered items whose words the task's text holds, by name, in `ITEMS`' order, at most `TRIGGERED`."""
    return tuple(name for name, found, _ in ITEMS if found.search(task))[:TRIGGERED]


def section(task: str, mode: str) -> tuple[str, tuple[str, ...]]:
    """The audit as the first clean run's observation says it in reply form `mode`, and its triggered items by name:
    `HEAD`, `INTRO`, then the fixed items and the triggered ones as a numbered list."""
    names = triggered(task)
    texts = dict((name, text) for name, _, text in ITEMS)
    asked = (*FIXED, *(texts[name] for name in names))
    return "\n".join([HEAD, INTRO[mode], *(f"{number}. {item}" for number, item in enumerate(asked, 1))]), names


@dataclass(frozen=True)
class Answers:
    """The reply after the audit, read: its lines in the answer form (three fields or more between `|`, the last
    opening on OK or FIX), those whose first field quotes the task's text, and those answered FIX."""
    answered: int = 0
    quoted: int = 0
    fix: int = 0


def folded(text: str) -> str:
    """`text` with its whitespace runs made one space and its case folded."""
    return " ".join(text.split()).casefold()


def quotes(field: str, told: str) -> bool:
    """Whether the whole first field quotes the folded task, after removing answer labels and outer marks."""
    field = LEAD.sub("", field.replace("**", ""), count=1)
    bare = LABEL.sub("", field, count=1).strip(EDGES)
    return bool(bare) and folded(bare) in told


def answers(reply: str, task: str) -> Answers:
    """The reply's lines in the answer form, outside its fenced blocks (`Answers`), against the task's text `task`."""
    told = folded(task)
    answered = quoted = fixed = 0
    inside = False
    for line in reply.splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        if inside or "|" not in line:
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) < 3:
            continue
        verdict = VERDICT.match(fields[-1])
        if verdict is None or BOTH.match(fields[-1][verdict.end():]):
            continue
        answered += 1
        fixed += verdict.group(1).upper() == "FIX"
        quoted += quotes(fields[0], told)
    return Answers(answered, quoted, fixed)
