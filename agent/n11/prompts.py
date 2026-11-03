"""The loop's prompt as a switch (lane prompt1; the owner's in90 and in91; RULING ROW 605; the cold reader's designs
A, B and C, runs/lanes/cold-prompt1017.md, forms 1 to 3 of cold-prompt1017-proposals.toml): `AGENTHON_N11_PROMPT` or
agent/n11/PROMPT, one word, armed in form `loop` where the loop starts in the fenced form (the candidate tree's), since
each word's text shows fenced replies; anywhere else, and under `full`, today's text (agent/n11/solve.py `SYSTEM`).

- `lean` (design A): the system prompt is `LEAN` (the paths the first message's Paths section names, the runtime's
  versions, the run's limit filled in), then, where any function is carded, `LEAN_LIBRARY` and the cards in design
  A's form (`lean_card`: the signature without defaults, `computes:`, `conventions:` with their allowed values,
  `returns:`; no formula, no example; at most `LEAN_CARD_BYTES` each). No index. The first message is today's.
  Its cards' form is a setting of its own (lane lean2; the lean reading's synthesis, READ ROW 632, which retired the
  design at its card dose): `AGENTHON_N11_LEAN_CARDS` or agent/n11/LEAN_CARDS, `cut` (the above, the default), `whole`
  (each function's card as today's full prompt gives it, `library.Card` uncut, formula and example kept, no bound) or
  a whole number of bytes within `LEAN_CARD_BOUNDS` (design A's form at that bound). Under `cut` alone the cards are
  bounded in all as before (`lean_cards`); `whole` and a number card every function the first message cards.
- `taskfirst` (design B): today's text re-placed (`Control`, read from the system prompt `N11.system` renders, every
  other switch's replacement in it): the system prompt keeps the head with `TASKFIRST_SAID`, the reply form's
  paragraph, the cut paragraph, the must list and the example; the first message is ordered `TASKFIRST_ORDER`, then
  How to work (steps 1 to 5, `TASKFIRST_STEP` the bridge, the runtime paragraph and the defaults), then Library (the
  library paragraph, the index, the cards), then today's clock, budget and closing lines, then `TASKFIRST_CLOSE`.
- `constants` (design C): today's text with steps 1 and 2 replaced (`CONSTANTS_STEPS`), step 5's insertion
  (`CONSTANTS_STEP_FIVE`), the silent library defaults after the defaults list (`CONSTANTS_SILENT`), the literal
  example in the prose one's place (`CONSTANTS_EXAMPLE`), and each card's conventions as constant stubs with its
  example calling them (`constants_cards`).
- `leanconst` (lane leanconst1; the lean line's next form, the untried "lean whole combined with the constants block"):
  `lean`'s system prompt with design C's parts where `constants` puts them in today's text (`LEANCONST`): steps 1 and
  2 `CONSTANTS_STEPS`, step 4 reading the digest against the CONVENTIONS block (`LEANCONST_STEP_FOUR`, step 5's
  insertion: the finish rule said once), `CONSTANTS_SILENT` after the defaults, `CONSTANTS_EXAMPLE` in the example's
  place; its library paragraph `LEANCONST_LIBRARY` (`LEANCONST_LIBRARY_EDITS`: the clause step 2 says in the block's
  words left out, the card's argument and `returns:` contract of today's library paragraph said); the cards in lane
  lean2's form, `constants_cards` writing a whole card's conventions as stubs (a cut card of design A's form has no
  typed signature to be known by, and goes as `lean` sends it). The first message is today's.
- `norules` and `rulesafter` (the rules probe's arms ii and iii, form 5; the coordinator's addition): today's text
  with the runtime paragraph and the defaults list (`MOVED`) left out of the system prompt; under `rulesafter` they
  open a section of the first message (`RULES_AFTER`, `rules_after`) after the task's sections and before the clock.
- `minimal` (lane prompt2; the refuter's accepted form, runs/lanes/r-prompt1017-proposals.toml:2, ACCEPT ROW 614): the
  high end of the dose. The system prompt is `minimal`'s whole: the one-shot's head (`MINIMAL_HEAD`), today's fenced
  reply form as the tree sends it with the module run line (`MINIMAL_REPLY`), the cut as the unit's `CUT` runs it
  (`MINIMAL_CUT`), four working sentences (`MINIMAL_WORK`) and the one-shot's must list (`MINIMAL_MUST`); no step,
  default, runtime note, library paragraph, index or card, at most `MINIMAL_BYTES`. The first message is today's
  without the pointer's line, and no selection call is sent (agent/n11/solve.py `selected`).
- `clean` (lane clean1; the owner's in90 and in134; the static audit runs/lanes/codex-audit-full-prompt-1019.md, its
  section 5): today's text, every guidance sentence kept, with the audit's edits (`CLEAN_EDITS`), each made where its
  anchors stand and the rest logged. The system prompt (`clean_parts`, `clean_text`, read from today's render through
  `Control`): the head without its network phrase (the must list says it); the reply form without its empty opener
  and with `finish` requesting completion (`clean_tools`, which agent/n11/solve.py `told` reads too); step 1 with the
  defaults' timing lines after sentence A, the task's lags first (`lagged`); step 2 without its library clause and
  its restated contract, step 3 without its automatic-run clause, the prose example left out; the cut said as the
  unit's `CUT=continue` runs it (`CLEAN_CUT`); the library paragraph scoped to the quantities a card computes
  (`CLEAN_SCOPE`), the cards' head without its call imperative; in the order `CLEAN_ORDER` (the must list before the
  runtime and the defaults, the library paragraph beside the defaults' library line and its cards). The first
  message (agent/n11/solve.py `first_turn`): the Paths section without its "nothing anywhere else but /tmp" tail, a
  file's described portrait left out where the input data shows its tree already (`repeated_portraits`), no Time
  section (the must list says the run's limit, Budget the unit's and the requests'), no closing convention reminder
  (step 1 says it), the pointer without its call imperative; the selection call's index without the lookup it
  never offers (agent/n11/solve.py `selected`).

No text states a count (in281): the cut is said at the output limit, and the run's limit is today's own. The reached
checks read a unit's first `work/main.py` and its first message (`constants_read`, `layout`). Keys `KEYS`."""
from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter
from dataclasses import dataclass

from n11 import library

ENV = "AGENTHON_N11_PROMPT"
FILE = "PROMPT"
SETTINGS = ("full", "lean", "taskfirst", "constants", "norules", "rulesafter", "minimal", "leanconst",
            "clean")
LEAN_WORDS = ("lean", "leanconst")  # the words whose system prompt is design A's prose and whose cards are lane lean2's
FORMS = ("loop",)               # armed in form `loop` alone (the others keep today's text, and the log says so)
MODE = "fenced"                 # and where the loop starts in this reply form: each word's text shows fenced blocks
KEYS = ("prompt_mode", "prompt_bytes_system", "lib_uncarded_calls", "prompt_layout", "constants_block",
        "constants_quoted", "outputs_block", "constants_first", "lean_cards_mode", "lean_cards_bytes",
        "prompt_bytes_first")     # lane clean1's last: the first loop request's messages, their bytes as sent

# ---------------------------------------------------------------------------------------- design A, `lean`

LEAN = """You are an expert quantitative developer. Solve the task in the next message by writing one Python 3 \
script, work/main.py, that reads the task's input files and writes its output files. It runs unattended, with no \
network. Hidden tests then check those files: the exact file names, field names and shapes, and every value within a \
tight tolerance under the conventions the task states. A script that runs cleanly can still be wrong.

How to reply
- Each file is one fenced block whose opening line is ```python file=work/<name>.py. The block is saved as that whole \
file.
- A saved work/main.py runs if changed, its imports present and time left. The next message shows its exit, output, \
traceback, output-file values, sanity findings, checks and time left.
- Write `finish` alone outside a block to request completion. Follow any refusal.
- A reply is cut at the output limit. Follow the next message to continue or split it. Write work/main.py whole in \
one reply. If it cannot fit, write functions in work/helpers.py first, then work/main.py importing it.
- If you change only a module, add the line `run work/main.py` after it.

How to work
1. Read the task and its input files. Find each convention the task states: day count, annualisation, compounding, \
log or simple returns, ddof, the sign of a loss, window lengths, rounding, units. Use exactly those.
2. In your first reply, write work/main.py. It reads the inputs, computes each quantity and writes every output file \
with exactly the names, keys and columns the task gives.
3. If the run or a check fails, rewrite the failing file whole.
4. Before you finish, read each value in the output digest against the task: its sign, its range, its units, its \
order (ES at least VaR; a probability in [0, 1]).

If the task does not say otherwise
- A signal from bar t's close trades at bar t+1's open at the earliest.
- A model that predicts row t is trained on rows up to t-1 only.
- Align predictions with targets by date or key, never by position after a shift.
- Before a correlation, drop the rows where either side is NaN. "Valid observations" are the rows left after dropping \
NaN, never filled values.

Every script must
- read inputs from, and write outputs to, the directories under "# Paths". Paths in the task text belong to another \
machine. Write nothing outside the output directory, work/ and /tmp.
- use only the standard library, qf, numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, \
openpyxl, matplotlib (Agg backend), plotly, seaborn, talib and backtrader. No network, pip or subprocess.
- seed every random generator the task does not seed.
- write JSON with plain Python numbers: float(x), int(x).
- never write the task's text, a file named reward.json, or the watermark <canary> (or an input field holding it) \
into an output file.
- end within its run's limit of {run} seconds, printing a short line a step at most.

Versions: {versions}. These names are gone: np.trapz (use np.trapezoid), np.product (np.prod), np.NaN (np.nan), \
np.in1d (np.isin), scipy simps and cumtrapz (simpson, cumulative_trapezoid), linalg.pinv2 (pinv), fillna(method=...) \
(.ffill(), .bfill()), DataFrame.append (pd.concat), frequencies "M" and "H" ("ME", "h"). In pandas 3, df["a"][i] = x \
changes nothing: write df.loc[i, "a"] = x. On a labelled index s[k] is a label: write s.iloc[k]. An integrate.quad \
integrand returns float(...). RandomState draws Student-t with .standard_t(df). The hess_inv of minimize(..., \
method="L-BFGS-B") needs .todense() before indexing.

An example reply (the shape, not the method):
```python file=work/main.py
import json
from pathlib import Path
import pandas as pd

IN = Path("{input_dir}")
OUT = Path("{output_dir}")

def main():
    prices = pd.read_csv(IN / "prices.csv", parse_dates=["date"]).sort_values("date")
    result = {"n_obs": int(len(prices))}
    (OUT / "result.json").write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```"""
LEAN_LIBRARY = """The library qf (import qf). Pass every convention explicitly. The cards below are the \
functions a first reading of this task picked, and they may not fit. If a card computes what the task asks, call it \
and pass each convention as the task states it. If the task asks for another method or states another formula, write \
the task's. Call no qf function that has no card below.

{cards}"""      # said only where a function is carded: with none, the library is the must list's name alone
LEAN_BYTES = 4_200              # design A's dose: the system prompt before its library paragraph and cards (`LEAN`
#                                 filled in), at most (a test's bound); with that paragraph the verbatim text is
#                                 about 4.4 KB, the cold reader's "about 3.9 KB" its text before the paragraph
LEAN_CARD_BYTES = 450           # and a card in its form, at most
LEAN_CARDS_ENV = "AGENTHON_N11_LEAN_CARDS"  # lane lean2: the lean cards' form, `cut` (at `LEAN_CARD_BYTES`, the
LEAN_CARDS_FILE = "LEAN_CARDS"              # default), `whole` (`library.Card`'s, uncut) or design A's form at a
LEAN_CARDS = ("cut", "whole")               # whole number of bytes within `LEAN_CARD_BOUNDS`; the desk's switch, or
#                                             agent/n11/LEAN_CARDS, one word, an image's
LEAN_CARD_BOUNDS = (250, 2_000)     # a bound outside them refused: under 250 the longest signature (235 B) is clipped;
#                                     over 2,000 no card cuts (the longest uncut is 1,277 B)
CARD_FLOOR = 40                 # a card's part `lean_card` cuts no shorter than this (bytes), the signature uncut
CUT = " ..."                    # a part cut to fit ends so

# ---------------------------------------------------------------------------------------- design B, `taskfirst`

TASKFIRST_SAID = ("The first message holds the task, its outputs, its inputs and paths, then how to work, then the "
                  "library's cards for it.")
TASKFIRST_ORDER = ("Task", "Output files", "Input data", "Input files, described", "Paths")   # then How to work and
#                                                                                               Library, then the clock
TASKFIRST_HOW, TASKFIRST_LIBRARY = "How to work", "Library"
TASKFIRST_STEP = ("which the first message describes in full", "above")    # step 1's bridge, now that the task is above
TASKFIRST_CLOSE = "Write work/main.py now, whole, in one block."
TASKFIRST_BYTES = 4_500         # design B's dose: the system prompt, at most (a test's bound)

# ---------------------------------------------------------------------------------- the probe's arms ii and iii

CONTROLLED = ("taskfirst", "constants", "norules", "rulesafter", "clean")    # the words that read today's
#                                                                             render (`Control`)
MOVED = ("api", "defaults")     # `Control`'s paragraphs `norules` leaves out and `rulesafter` moves: the runtime
#                                 paragraph and the "Unless the task says otherwise" list
RULES_AFTER = "How to work, continued"      # `rulesafter`'s section of the first message, after the task's

# ---------------------------------------------------------------------------------------- `minimal` (lane prompt2)

MINIMAL_HEAD = ("You are an expert quantitative developer. You solve a task by writing one complete Python 3 script, "
                "which is run, unattended, with no network; hidden tests then check the files it wrote against the "
                "task.")       # the one-shot's head (agent/n0/solve.py `SYSTEM`), its "run once" said "run": the loop
#                                runs the script again after each write
MINIMAL_REPLY = """You act through the form of your reply. Each file is one fenced block whose opening line names it, \
```python file=work/<name>.py, and is saved whole. After the blocks, work/main.py runs where the reply wrote it (or \
the last file written, where it has an `if __name__ == "__main__":` block and work/main.py does not import it), or the \
file a line `run work/<name>.py` names. These lines, each alone on its line outside any block, are commands: `finish` \
ends the task, and nothing else does. What each did comes back in the next message: the run's exit, its output's \
ends, its traceback, the files in the output directory with their values (a digest), the sanity findings, the checks \
and your budget. After a clean run, where you change only a module, add the line `run work/main.py` after it."""
#   today's fenced form (agent/n11/solve.py `FENCED_PART`) as the tree sends it, `inspect` and `lib` withheld, then
#   the module run line (How to work's step 3)
MINIMAL_CUT = {     # the cut as the unit's `CUT` runs it: `continue` (the tree's) keeps the head, `split` a draft
    "continue": ("A reply that runs past the output limit is cut: a file the cut falls in is kept up to its last "
                 "complete line, and the next message asks for the rest."),
    "split": ("A reply that runs past the output limit is cut, and a file the cut falls in is not saved, only its "
              "complete definitions as a draft; the next message asks you to split it.")}
MINIMAL_WORK = ("Write the whole solution as one script, work/main.py, in one reply. Read each run. Where a run or a "
                "check fails, write the failing file again, whole. Finish when the checks pass and the values hold.")
MINIMAL_MUST = """The script must:
- read its inputs from the true paths given under "Paths": the paths in the task's text are those of another machine;
- write every output file the task asks for, with exactly the names, formats, keys and columns it gives, directly \
into the output directory given under "Paths", and nothing anywhere else but work/ and /tmp;
- use only the Python standard library and these installed packages: {libraries}; no network, no pip, no subprocess;
- be deterministic: seed any random generator the task does not seed itself;
- write JSON with plain Python numbers (convert numpy values with float() or int());
- never copy the task's text into an output file, and never write a file named reward.json; `<canary>` in the task \
or the data stands for a watermark string: never write it, or an input field that holds it, into any output file;
- finish well within the time it is given;
- print nothing but one short final line saying it is done."""
#   the one-shot's must list verbatim, its output-file item as today's loop says it (work/ and /tmp beside the output)
MINIMAL_BYTES = 2_500           # the dose: the system prompt whole, at most (a test's bound; the one-shot's 1,340 B)

# ---------------------------------------------------------------------------------------- design C, `constants`

CONSTANTS_STEPS = """1. Read the task, its paths and its input files, which the first message describes in full. \
Start with work/main.py, and start work/main.py with two blocks of constants, before any function or computation:
   CONVENTIONS: one constant per convention the numbers depend on. Its comment quotes the task's words for it, or \
reads "task silent:" and names the default you chose. Cover at least: log or simple returns; the annualisation factor \
and its form; ddof; each window's length in rows and whether it holds the current row; compounding and day count; when \
a signal trades and which rows a fit may see; the sign a loss is reported with; units (percent, fraction, basis \
points); rounding.
   OUTPUTS: each output file's name and its field names or columns, copied character for character from the task.
2. Then the computation, in the same file: every function takes its conventions from these constants, and no window \
length, factor, ddof or sign is typed anywhere else. If a qf card computes a quantity, pass the constants as its \
convention arguments."""
CONSTANTS_STEP_FIVE = ("Read each run's digest of the output files", " against your CONVENTIONS block and")
CONSTANTS_SILENT = """Library defaults that change a number without an error. Set each from CONVENTIONS, every time:
- np.std and np.var default to ddof=0; pandas .std() and .var() default to ddof=1.
- s.rolling(n) at row t holds rows t-n+1 through t, the current row included; .shift(1) it where row t must not be \
seen.
- s.ewm(...) defaults to adjust=True and takes exactly one of span, com, halflife, alpha.
- np.quantile and np.percentile default to method="linear".
- statsmodels OLS fits no intercept unless X goes through sm.add_constant.
- arch_model defaults to a constant mean, GARCH(1,1) and normal errors; set mean, vol, p, q, dist and the returns' \
scale (percent or fraction) as the task says.
- resample("W") ends weeks on Sunday; pd.bdate_range knows no holidays: count trading days from the dates in the data.
- round() and np.round round halves to even.
- scipy.stats.norm.ppf(0.05) is negative."""
CONSTANTS_EXAMPLE = """An example reply (the shape: your constants and their quotes come from your task):
```python file=work/main.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

IN = Path("{input_dir}")
OUT = Path("{output_dir}")

# CONVENTIONS
RETURNS = "simple"      # "daily simple returns"
PERIODS_PER_YEAR = 252  # "annualise with 252 trading days"
DDOF = 1                # "sample standard deviation"
VOL_WINDOW = 21         # "21-day rolling volatility"; holds the current row
LOSS_SIGN = 1           # task silent: VaR reported as a positive loss
DECIMALS = None         # task silent: no rounding

# OUTPUTS
OUT_FILE = "risk.json"
FIELDS = ["annual_vol", "var_95"]

def simple_returns(prices):
    return prices.pct_change().dropna()

def main():
    ...

if __name__ == "__main__":
    main()
```"""      # its constants' values are the example's (the cold reader's), no count stated to the model
CONSTANTS_HEAD = "conventions, as CONVENTIONS constants (fill each from the task; ? is not Python, on purpose):"
CONSTANTS_ADDED = 2_500         # design C's dose, about 1.8 KB added before the library: at most this (a test's
#                                 bound)

# ---------------------------------------------------------------------------------------- the reached checks

CONVENTIONS_MARK = re.compile(r"^#\s*CONVENTIONS\b")       # the block's comment, a comment token at column 0
OUTPUTS_MARK = re.compile(r"^#\s*OUTPUTS\b")
UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")                    # a module-level UPPER_CASE constant's name
STUBBED = re.compile(r"^(\s*[A-Za-z_]\w*\s*(?::[^=]*)?=\s*)\?(\s*(?:#.*)?)$", re.M)   # design C's `NAME = ?` stub
FIRST = ("saved", "rejected", "none")   # `constants_first`: the unit's first attempted work/main.py saved as sent,
#                                         not saved as sent (refused, or another text saved first), or never attempted
QUOTED = re.compile(r"\"([^\"]+)\"|'([^']+)'|“([^”]+)”")    # a comment's quoted words
SILENT = "task silent:"
QUOTE_WORDS = 4                 # a comment quotes the task where a quote of at least this many words is in its text
CONSTANTS_LEAST = 3             # `constants_block`: the block holds at least this many constants


def fill(template: str, **values) -> str:
    """`template` with each `{name}` of `values` written in, any other brace kept (the texts hold code)."""
    for name, value in values.items():
        template = template.replace("{" + name + "}", str(value))
    return template


def flat(text: str) -> str:
    """`text` on one line, its whitespace runs one space."""
    return " ".join(text.split())


def cut_words(text: str, limit: int) -> str:
    """`text` at most `limit` bytes, cut at a word and ended `CUT` where it was longer; never longer than before."""
    if size(text) <= limit:
        return text
    room = max(limit - size(CUT), 0)
    head = text.encode("utf-8")[:room].decode("utf-8", errors="ignore")
    kept = head.rsplit(" ", 1)[0] if " " in head.strip() else head
    shorter = kept.rstrip(" ,;:") + CUT
    return shorter if size(shorter) < size(text) else text[:-1]


def size(text: str) -> int:
    return len(text.encode("utf-8"))


# ------------------------------------------------------------------------------------------ the cards' forms


def arguments_of(function: library.Function) -> list[str]:
    """The names of `function`'s arguments, in order (none where its `def` line does not parse)."""
    try:
        args = ast.parse(f"def f({function.arguments}): pass").body[0].args
    except SyntaxError:
        return []
    return [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]


def plain_signature(function: library.Function) -> str:
    """`qf.<name>(<arguments>)` with no default and no annotation: nothing pre-filled."""
    try:
        args = ast.parse(f"def f({function.arguments}): pass").body[0].args
    except SyntaxError:
        return function.signature
    args.defaults, args.kw_defaults = [], [None] * len(args.kwonlyargs)
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if arg is not None:
            arg.annotation = None
    return f"{function.call}({ast.unparse(args)})"


@dataclass(frozen=True)
class Entry:
    """One entry of a card's `conventions:` section: the arguments it opens with (none: a sentence) and its words."""
    names: tuple[str, ...]
    text: str


def convention_entries(section: str, arguments: list[str]) -> list[Entry]:
    """A `conventions:` section's entries (`library.entries`), each read for the arguments its words open with
    (`name:` or `a, b:`, every name an argument of the function), its text on one line; the label line's own words,
    where it has any, an entry of no argument first."""
    label, found = library.entries(section)
    said = label.split(":", 1)[1].strip() if ":" in label else ""
    entries = [Entry((), flat(said))] if said else []
    for entry in found:
        text = flat(entry)
        text = text[2:] if text.startswith("- ") else text
        lead = re.match(r"([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:\s*(.*)$", text)
        names = tuple(name.strip() for name in lead.group(1).split(",")) if lead else ()
        if names and all(name in arguments for name in names):
            entries.append(Entry(names, lead.group(2)))
        else:
            entries.append(Entry((), text))
    return entries


def allowed(values: tuple[str, ...]) -> str:
    """An argument's allowed names as the code checks them: `'a' | 'b'`, a number where one is allowed."""
    return " | ".join("a number" if value == library.NUMBER else repr(value) for value in values)


def lean_card(function: library.Function, bound: int = LEAN_CARD_BYTES) -> str:
    """Design A's card: the signature without defaults; `computes:`, the docstring's first line; `conventions:`, each
    convention entry as `<arg> = <its words>`, an argument the code checks against names as `<arg> = 'a' | 'b'` (its
    allowed names, never cut), a sentence of no argument as it is; `returns:`; no formula, no example. Within `bound`
    bytes: the parts cut at a word, the longest first in this order, each to `CARD_FLOOR` at the least: the sentences,
    the arguments' words, `returns:`, `computes:`; then the last conventions left out, every other one before an
    argument's allowed names (each checked argument has its own, one in a shared entry too); then the card
    clipped."""
    arguments, checked = arguments_of(function), dict(function.enums)
    entries = convention_entries(function.sections.get("conventions", ""), arguments)
    listed = {entry.names[0] for entry in entries if len(entry.names) == 1 and entry.names[0] in checked}
    items = [[f"{', '.join(entry.names)} = ", allowed(checked[entry.names[0]]), None]
             if len(entry.names) == 1 and entry.names[0] in checked else
             [f"{', '.join(entry.names)} = ", entry.text, 1] if entry.names else ["", entry.text, 0]
             for entry in entries]
    items += [[f"{name} = ", allowed(values), None] for name, values in function.enums if name not in listed]
    returns = function.sections.get("returns", "")
    computes = ["", flat(function.card), 3]
    said = ["", flat(returns.split(":", 1)[1] if ":" in returns else returns), 2]

    def render() -> str:
        lines = [plain_signature(function), f"  computes: {computes[1]}"]
        lines += [f"  conventions: {'; '.join(lead + text for lead, text, _ in items)}"] if items else []
        lines += [f"  returns: {said[1]}"] if said[1] else []
        return "\n".join(lines)

    while size(render()) > bound:
        parts = [part for part in (*items, said, computes) if part[2] is not None and size(part[1]) > CARD_FLOOR]
        if not parts:
            break
        part = max(parts, key=lambda each: (-each[2], size(each[1])))
        part[1] = cut_words(part[1], max(CARD_FLOOR, size(part[1]) - (size(render()) - bound)))
    while size(render()) > bound and items:
        words = [k for k, item in enumerate(items) if item[2] is not None]
        items.pop(words[-1] if words else -1)       # an argument's allowed names last
    card = render()
    return card if size(card) <= bound else card.encode("utf-8")[:bound].decode("utf-8", errors="ignore")


def lean_cards_words() -> tuple[str, ...]:
    """The lean cards' words, as the desk's row says them: every whole number within `LEAN_CARD_BOUNDS`, then
    `LEAN_CARDS`."""
    return (*(str(n) for n in range(LEAN_CARD_BOUNDS[0], LEAN_CARD_BOUNDS[1] + 1)), *LEAN_CARDS)


def lean_card_as(function: library.Function, form: str) -> str:
    """`function`'s card in the lean cards' form `form` (a word of `lean_cards_words`): `whole`, its card as today's
    full prompt gives it (`library.Function.whole_card`); `cut`, design A's at `LEAN_CARD_BYTES`; a number, design A's
    at that many bytes."""
    return function.whole_card() if form == "whole" else \
        lean_card(function, LEAN_CARD_BYTES if form == "cut" else int(form))


def lean_cards(functions: list[library.Function], limit: int, form: str = LEAN_CARDS[0],
               word: str = LEAN_WORDS[0]) -> tuple[str, list[library.Function]]:
    """The lean cards of `functions` in the form `form` (`lean_card_as`), each once, in order, joined by blank lines;
    under `cut` within `limit` (`library.measured`'s unit), the last left out first, and under `whole` or a number
    every one; under `leanconst` (`word`) in design C's form where it reads them (`constants_cards`). With the
    functions carded."""
    kept = list(dict.fromkeys(functions))
    cards = [lean_card_as(function, form) for function in kept]
    while form == "cut" and kept and library.measured("\n\n".join(cards)) > limit:
        kept.pop()
        cards.pop()
    joined = "\n\n".join(cards)
    return (constants_cards(joined, kept) if word == "leanconst" else joined), kept


def lean(input_dir: str, output_dir: str, versions: str, run: float, cards: str, word: str = LEAN_WORDS[0]) -> str:
    """Design A's system prompt (`word` a word of `LEAN_WORDS`: under `leanconst` with design C's parts), its library
    paragraph and cards where any function is carded."""
    prose, library_ = (LEANCONST, LEANCONST_LIBRARY) if word == "leanconst" else (LEAN, LEAN_LIBRARY)
    text = fill(prose, input_dir=input_dir, output_dir=output_dir, versions=versions, run=f"{run:.0f}")
    return text + (f"\n\n{fill(library_, cards=cards)}" if cards else "")


def minimal(cut: str, libraries: str) -> str:
    """`minimal`'s system prompt: the head, the reply form, the cut as `cut` (a `CUT` word) runs it, the working
    sentences and the must list naming `libraries` (the one-shot's)."""
    return "\n\n".join([MINIMAL_HEAD, MINIMAL_REPLY, MINIMAL_CUT[cut], MINIMAL_WORK,
                        fill(MINIMAL_MUST, libraries=libraries)])


def versions_said(python: str, installed: dict[str, str]) -> str:
    """`Python 3.13, numpy 2.5, ...`: each version as its major and minor numbers."""
    def short(version: str) -> str:
        found = re.match(r"(\d+)(?:\.(\d+))?", version or "")
        return f"{found.group(1)}.{found.group(2) or 0}" if found else version
    return ", ".join([f"Python {short(python)}", *(f"{name} {short(version)}" for name, version in installed.items())])


def stub_name(name: str, function: library.Function | None = None) -> str:
    """A convention's constant: its argument in capitals, qualified by its function where `function` is given (a name
    two cards share: `STD_DDOF`)."""
    return f"{function.name.upper()}_{name.upper()}" if function is not None else name.upper()


DATA_WORDS = re.compile(r"\b(?:a|an|one|the)\s+(?:\w+\s+)?(?:array|series|list|vector|matrix|frame|table|panel)"
                        r"\b(?!')|"
                        r"\bof the same length\b|\boldest first\b|\bper[- ](?:date|row|period|asset|day)\b|"
                        r"\bone\b[^;:]{0,40}?\bper\b|\b[A-Za-z] x [A-Za-z]\b",
                        re.I)   # words that say a sequence or a shape (`T x N`), not a choice
ASIDE = re.compile(r"\([^()]*\)")     # a parenthesis: an aside, never what the argument is


def data_arguments(function: library.Function, entries: list[Entry]) -> set[str]:
    """The arguments a card's conventions name that are data, passed as given, not conventions: each the code does not
    check against names whose entry's first clause (before its first `;` or `:`, its asides left out) says a
    sequence or a shape (`DATA_WORDS`: `returns: one finite return per date`, `T x N, rows the dates`)."""
    checked = dict(function.enums)
    return {name for entry in entries for name in entry.names
            if name not in checked and DATA_WORDS.search(ASIDE.sub("", re.split(r"[;:]", entry.text, maxsplit=1)[0]))}


def stubs_of(function: library.Function, entries: list[Entry], shared: frozenset[str] = frozenset()) -> dict[str, str]:
    """Each convention argument's constant, in the card's order: the entries' arguments that are not data
    (`data_arguments`), then each argument the code checks against names that no entry names; a constant in `shared`
    qualified by the function (`stub_name`)."""
    data = data_arguments(function, entries)
    names = [name for entry in entries for name in entry.names if name not in data]
    names += [name for name, _ in function.enums if name not in names and name not in data]
    return {name: stub_name(name, function if stub_name(name) in shared else None) for name in dict.fromkeys(names)}


def constants_block(function: library.Function, entries: list[Entry], stubs: dict[str, str]) -> list[str]:
    """Design C's conventions section of one card: `CONSTANTS_HEAD`, then a stub `CONSTANT = ?` for each convention
    argument an entry opens with (the entry's words its comment, on its first stub's line), an entry of data
    arguments alone a comment line saying so, an entry of no argument a comment line, and a stub for each argument the
    code checks against names that no entry names (its names the comment)."""
    listed = {name for entry in entries for name in entry.names}
    rows: list[tuple[str, str]] = []
    for entry in entries:
        named = [name for name in entry.names if name in stubs]
        if not entry.names:
            rows.append(("", entry.text))
        elif not named:
            rows.append(("", f"data, not a convention: {', '.join(entry.names)}: {entry.text}"))
        for k, name in enumerate(named):
            rows.append((f"{stubs[name]} = ?", f"{', '.join(entry.names)}: {entry.text}" if k == 0 else ""))
    rows += [(f"{stubs[name]} = ?", f"{name}: {allowed(values)}") for name, values in function.enums
             if name not in listed and name in stubs]
    width = max((len(stub) for stub, _ in rows), default=0)
    lines = [CONSTANTS_HEAD]
    for stub, comment in rows:
        if not stub:
            lines.append(f"  # {comment}")
        else:
            lines.append(f"  {stub:<{width}}" + (f"  # {comment}" if comment else "").rstrip())
    return lines


def stubbed_example(lines: list[str], stubs: dict[str, str]) -> list[str]:
    """An example's lines with each keyword argument of `stubs` passed a literal (a string, a number, None, True or
    False) passing its constant instead: `ddof=1` written `ddof=DDOF`."""
    if not stubs:
        return lines
    keyword = re.compile(r"([(,]\s*)(" + "|".join(re.escape(name) for name in stubs)
                         + r")(\s*=\s*)(?:\"[^\"\n]*\"|'[^'\n]*'|-?\d[\w.+-]*|None|True|False)(?=\s*[,)\n])")
    text = keyword.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{stubs[m.group(2)]}", "\n".join(lines))
    return text.split("\n")


SECTION_LINE = re.compile(r"^(?:conventions?|returns|formula|source|aliases|examples?)\s*:", re.I)
CONVENTIONS_LINE = re.compile(r"^conventions?\s*:", re.I)
EXAMPLE_LINE = re.compile(r"^examples?\s*:", re.I)


def section_span(lines: list[str], start: int) -> int:
    """The end of the section whose label opens `lines[start]`: the next label, a cut card's closing line (`[`) or a
    blank line, else the card's end."""
    end = start + 1
    while end < len(lines) and lines[end].strip() and not SECTION_LINE.match(lines[end]) \
            and not lines[end].startswith("["):
        end += 1
    return end


def card_entries(lines: list[str], function: library.Function) -> tuple[int | None, int | None, list[Entry]]:
    """A card's `conventions:` section (its first line and end, None where it has none) and its entries."""
    at = next((k for k, line in enumerate(lines) if CONVENTIONS_LINE.match(line)), None)
    if at is None:
        return None, None, []
    end = section_span(lines, at)
    return at, end, convention_entries("\n".join(lines[at:end]), arguments_of(function))


def constants_card(lines: list[str], function: library.Function, shared: frozenset[str] = frozenset()) -> list[str]:
    """One card (its lines, as the cards carry it: whole or cut) in design C's form: its `conventions:` section as
    `constants_block`, where it has none and the code checks an argument against names the block put before its
    `returns:` (else its example, else its end); its example calling the stubs' constants (`stubs_of`, those in
    `shared` qualified); the rest as it came."""
    at, end, entries = card_entries(lines, function)
    stubs = stubs_of(function, entries, shared)
    if at is not None:
        lines = [*lines[:at], *constants_block(function, entries, stubs), *lines[end:]]
    elif function.enums:
        place = next((k for k, line in enumerate(lines) if re.match(r"^(?:returns|examples?)\s*:", line, re.I)
                      or line.startswith("[")), len(lines))
        lines = [*lines[:place], *constants_block(function, entries, stubs), *lines[place:]]
    at = next((k for k, line in enumerate(lines) if EXAMPLE_LINE.match(line)), None)
    if at is not None:
        end = section_span(lines, at)
        lines = [*lines[:at], *stubbed_example(lines[at:end], stubs), *lines[end:]]
    return lines


def constants_cards(cards: str, functions: list[library.Function]) -> str:
    """The cards as they came (`N11.cards`: their head, each card whole or cut) with each card of `functions` in
    design C's form (`constants_card`), a card known by its typed signature opening a line; a constant two of these
    functions' cards would both use qualified by its function in each; the head and any line of no card's as they
    came."""
    if not cards:
        return cards
    heads = {function.typed: function for function in functions}
    lines = cards.split("\n")
    starts = [k for k, line in enumerate(lines) if line in heads]
    spans = []
    for n, start in enumerate(starts):
        body = lines[start:starts[n + 1] if n + 1 < len(starts) else len(lines)]
        kept = len(body)
        while kept and not body[kept - 1].strip():
            kept -= 1           # the blank lines between two cards stay as they were
        spans.append((heads[lines[start]], body[:kept], body[kept:]))
    once = {function: set(stubs_of(function, card_entries(body, function)[2]).values()) for function, body, _ in spans}
    used = Counter(stub for stubs in once.values() for stub in stubs)     # by function: a card twice shares nothing
    shared = frozenset(stub for stub, count in used.items() if count > 1)
    out = lines[:starts[0]] if starts else lines
    for function, body, blanks in spans:
        out += constants_card(body, function, shared) + blanks
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ the control


OPENINGS = ("You are an expert", "", "How to work:", "A reply that runs past", "The runtime", "Unless the task says "
            "otherwise:", "Every script must:", "", "The library:")    # `Control`'s paragraphs, each as it opens
RULES = ("head", "tools", "how", "cut", "api", "defaults", "must", "example")   # those before the library's


@dataclass(frozen=True)
class Control:
    """Today's system prompt as `N11.system` renders it (agent/n11/solve.py `SYSTEM`, every switch's replacement in
    it) in its paragraphs: the head, the reply form's paragraph, How to work, the cut paragraph, the runtime, the
    defaults, the must list, the example, the library paragraph, the index (or the selection call's section) and the
    cards apart. None where the text is not in these paragraphs (`read`)."""
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

    def rules(self, **parts: str | None) -> str:
        """The prompt before its library (`RULES`: the library paragraph, the index and the cards after), any of
        `parts` in its paragraph's place, a paragraph given None left out: each word's dose (`prompt_bytes_system`)."""
        said = {**self.__dict__, **parts}
        return "\n\n".join(said[name] for name in RULES if said[name] is not None)

    def text(self, **parts: str | None) -> str:
        """The prompt again, any of `parts` in its paragraph's place, a paragraph given None left out."""
        said = {**self.__dict__, **parts}
        return "\n\n".join([self.rules(**parts), said["library"], said["index"]]) \
            + (f"\n\n{said['cards']}" if said["cards"] else "")


def taskfirst_system(control: Control) -> str:
    """Design B's system prompt: the head with `TASKFIRST_SAID`, the reply form, the cut, the must list, the
    example."""
    return "\n\n".join([f"{control.head} {TASKFIRST_SAID}", control.tools, control.cut, control.must, control.example])


def taskfirst_sections(control: Control, step_one: tuple[str, str] | None = None) -> str:
    """Design B's two sections of the first message after the paths: How to work (the steps without their label, step
    1's bridge `TASKFIRST_STEP` and `step_one`, the request's own wording where a tool is withheld; the runtime; the
    defaults) and Library (the library paragraph, the index, the cards)."""
    steps = control.how.split("\n", 1)[1] if "\n" in control.how else control.how
    steps = steps.replace(*TASKFIRST_STEP, 1)
    if step_one is not None:
        steps = steps.replace(*step_one, 1)
    library_ = "\n\n".join(part for part in (control.library, control.index, control.cards) if part)
    return (f"# {TASKFIRST_HOW}\n\n{steps}\n\n{control.api}\n\n{control.defaults}\n\n"
            f"# {TASKFIRST_LIBRARY}\n\n{library_}\n\n")


def moved_out() -> dict[str, None]:
    """`norules`' and `rulesafter`'s parts (`Control.text`'s): the runtime paragraph and the defaults left out."""
    return dict.fromkeys(MOVED)


def rules_after(control: Control) -> str:
    """`rulesafter`'s section of the first message: `RULES_AFTER`, then the runtime paragraph and the defaults list as
    today's system prompt says them."""
    return f"# {RULES_AFTER}\n\n" + "".join(f"{getattr(control, name)}\n\n" for name in MOVED)


def constants_steps(how: str) -> str | None:
    """A How to work paragraph with its steps 1 and 2, two lines in a row, written `CONSTANTS_STEPS`; None where they
    are not."""
    lines = how.split("\n")
    one = next((k for k, line in enumerate(lines) if line.startswith("1. ")), None)
    if one is None or one + 1 >= len(lines) or not lines[one + 1].startswith("2. "):
        return None
    lines[one:one + 2] = CONSTANTS_STEPS.split("\n")
    return "\n".join(lines)


def constants_parts(control: Control, input_dir: str, output_dir: str, functions: list[library.Function],
                    fenced_example: str) -> tuple[dict[str, str], list[str]]:
    """Design C's edits to today's paragraphs (`Control.text`'s `parts`), each where its anchor stands, and the edits
    whose anchor did not (by name: `steps`, `step 5`, `example`)."""
    missed = []
    how = constants_steps(control.how)
    if how is None:
        how = control.how
        missed.append("steps")
    if CONSTANTS_STEP_FIVE[0] in how:
        how = how.replace(CONSTANTS_STEP_FIVE[0], "".join(CONSTANTS_STEP_FIVE), 1)
    else:
        missed.append("step 5")
    example = control.example
    if example == fenced_example:
        example = fill(CONSTANTS_EXAMPLE, input_dir=input_dir, output_dir=output_dir)
    else:
        missed.append("example")
    return {"how": how, "defaults": f"{control.defaults}\n\n{CONSTANTS_SILENT}", "example": example,
            "cards": constants_cards(control.cards, functions)}, missed


# ------------------------------------------------------------------------- `leanconst` (lane leanconst1): A with C

LEAN_HOW = "How to work\n"                              # lean's paragraphs design C's parts go in or after, as each
LEAN_DEFAULTS = "If the task does not say otherwise\n"  # opens; the example runs from its opening to the text's end
LEAN_EXAMPLE = "An example reply (the shape, not the method):\n"
LEANCONST_STEP_FOUR = ("read each value in the output digest", CONSTANTS_STEP_FIVE[1])   # step 5's insertion, in
#                                                                                          lean's step 4
LEANCONST_LIBRARY_EDITS = (
    (" and pass each convention as the task states it", ""),
    (" they may not fit.", " they may not fit. Read a card first: its signature names each convention as an argument, "
                           "and its `returns:` line says what comes back."),
)   # the first: step 2 says it in the block's words, and theirs win (a convention the task is silent on); the second:
#     today's library paragraph's sentence on reading a card (`library.USE`), its passing clause left out as the
#     first's, the diagnosis's sentence B (runs/lanes/codex-diag-lean-prose-1019.md), whose argument half the stubs
#     carry and whose `returns:` half nothing in design C does
LEANCONST_BYTES = 7_500         # the dose: the system prompt before its library paragraph and cards (`LEANCONST` filled
#                                 in), at most (a test's bound; the registration's `prompt_bytes_system` letter)


def edited(text: str, edits: tuple[tuple[str, str], ...]) -> str:
    """`text` with each `(old, new)` of `edits` made in turn; an `old` not standing in it exactly once raises (these
    are this module's own texts, edited at import)."""
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"{old!r} stands {text.count(old)} times in the text, not once")
        text = text.replace(old, new, 1)
    return text


def leanconst_text() -> str:
    """`LEAN` with design C's parts where `constants_parts` puts them in today's text: steps 1 and 2 `CONSTANTS_STEPS`
    (`constants_steps`), step 4 reading the digest against the CONVENTIONS block (`LEANCONST_STEP_FOUR`),
    `CONSTANTS_SILENT` after the defaults, `CONSTANTS_EXAMPLE` in the example's place; a paragraph not where it stood
    raises."""
    head, said, _ = LEAN.partition(f"\n\n{LEAN_EXAMPLE}")
    paragraphs = head.split("\n\n")
    how = [k for k, paragraph in enumerate(paragraphs) if paragraph.startswith(LEAN_HOW)]
    defaults = [k for k, paragraph in enumerate(paragraphs) if paragraph.startswith(LEAN_DEFAULTS)]
    steps = constants_steps(paragraphs[how[0]]) if len(how) == 1 else None
    if not said or steps is None or len(defaults) != 1:
        raise ValueError("the lean text's steps 1 and 2, its defaults or its example are not where they stood")
    paragraphs[how[0]] = edited(steps, ((LEANCONST_STEP_FOUR[0], "".join(LEANCONST_STEP_FOUR)),))
    paragraphs.insert(defaults[0] + 1, CONSTANTS_SILENT)
    return "\n\n".join([*paragraphs, CONSTANTS_EXAMPLE])


LEANCONST = leanconst_text()
LEANCONST_LIBRARY = edited(LEAN_LIBRARY, LEANCONST_LIBRARY_EDITS)


# ------------------------------------------------------------------------- `clean` (lane clean1): today's, edited

CLEAN_EDITS = ("paths", "library", "cards", "pointer", "cut", "writing", "lags", "opener", "network", "time",
               "portrait", "selection", "finish", "order")
#   the audit's section 5 in its order of priority (runs/lanes/codex-audit-full-prompt-1019.md): 1 `paths`; 2
#   `library`, `cards` and `pointer`; 3 `cut`; 4 `writing`; 5 `lags`; 6 `opener` and `network`; 7 `time`; 8
#   `portrait`; 9 `selection`; 10 `finish`; 11 `order`
CLEAN_PATHS = (", and nothing anywhere else but /tmp", "")  # the Paths section's (agent/common/paths.py): the must
#                                                              list allows work/ beside /tmp, and work/main.py is there
CLEAN_STEP_CALL = (" (a quantity the library computes is a call of its function)", "")
CLEAN_LIBRARY_LAST = "Write a formula yourself only where the task states one that differs from the card's."
CLEAN_SCOPE = ("These rules hold for a quantity a card computes by the method the task names; the rest of the script "
               "(reading and preparing the inputs, every quantity no card computes) they leave to you.")
CLEAN_CARDS = ("; where one computes a quantity the task asks, call it rather than writing its formula:", ":")
CLEAN_POINTER = (": call each for its quantity rather than writing its formula", "")
CLEAN_CUT = (   # under `CUT=continue` (agent/n11/solve.py `hold`, its fallback the split: `salvage`); `split`'s is true
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
    """The reply form's paragraph as `clean` says it: its empty opener left out (`CLEAN_OPENER`) and `finish` said to
    request completion (`CLEAN_COMPLETION`), each where it stands once (agent/n11/solve.py `told` reads it so)."""
    for old, new in (CLEAN_OPENER, CLEAN_COMPLETION):
        part = part.replace(old, new, 1) if part.count(old) == 1 else part
    return part


def clean_parts(control: Control, cut: str, timing: tuple[str, ...], example: str
                ) -> tuple[dict[str, str | None], list[str]]:
    """`clean`'s paragraphs: today's (`control`) with the audit's edits to the system prompt, each made where its
    anchors stand (`made`), and those not made, by name (`CLEAN_EDITS`). `cut` is the unit's `CUT` word (the cut
    paragraph said as `continue` runs it, `split`'s today's), `timing` the defaults' timing lines (moved to step 1),
    `example` the reply form's example (left out where the paragraph is it)."""
    parts: dict[str, str | None] = dict(control.__dict__)
    table = (
        ("library", (("how", *CLEAN_STEP_CALL),
                     ("library", CLEAN_LIBRARY_LAST, f"{CLEAN_LIBRARY_LAST} {CLEAN_SCOPE}"))),
        ("cards", (("cards", *CLEAN_CARDS),) if control.cards else ()),
        ("cut", tuple(("cut", *edit) for edit in CLEAN_CUT) if cut == "continue" else ()),
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


def clean_text(parts: dict[str, str | None], library_: bool = True) -> str:
    """`clean`'s system prompt: its paragraphs (`clean_parts`) in `CLEAN_ORDER`, an empty one (the example the
    writing edit took out) left out, then the cards; without `library_`, the text before its library paragraph (its
    dose, `prompt_bytes_system`)."""
    names = CLEAN_ORDER if library_ else CLEAN_ORDER[:CLEAN_ORDER.index("library")]
    text = "\n\n".join(parts[name] for name in names if parts[name])
    return text + (f"\n\n{parts['cards']}" if library_ and parts["cards"] else "")


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


# ------------------------------------------------------------------------------------------ the reached checks


def normal(text: str) -> str:
    """`text` in lower case, its whitespace runs one space: how a quote is read against the task."""
    return " ".join(text.lower().split())


def quotes_task(comment: str, task: str) -> bool:
    """Whether a constant's comment begins `SILENT`, or quotes at least `QUOTE_WORDS` words that stand in the task's
    text (each quote, else the whole comment; whitespace and case normalised)."""
    if normal(comment).startswith(SILENT):
        return True
    quotes = [next(group for group in found.groups() if group) for found in QUOTED.finditer(comment)] or [comment]
    said = normal(task)
    return any(len(normal(quote).split()) >= QUOTE_WORDS and normal(quote) in said for quote in quotes)


@dataclass(frozen=True)
class Constants:
    """The reached read of a first script (design C): the constants under its `# CONVENTIONS` comment before its first
    definition, by name and comment; the names its `# OUTPUTS` block's literals give."""
    conventions: tuple[tuple[str, str], ...]
    outputs: tuple[str, ...] | None         # None where the script has no OUTPUTS block

    def block(self) -> int:
        return int(len(self.conventions) >= CONSTANTS_LEAST)

    def quoted(self, task: str) -> int | None:
        """The percent of the constants whose comment quotes the task or says it silent (`quotes_task`), whole; None
        where none stands."""
        if not self.conventions:
            return None
        return round(100 * sum(quotes_task(comment, task) for _, comment in self.conventions) / len(self.conventions))

    def outputs_named(self, task: str) -> int:
        return int(bool(self.outputs) and all(name in task for name in self.outputs))


def code_of(script: str) -> tuple[ast.Module, list[tokenize.TokenInfo]]:
    """A script's tree and tokens, design C's `NAME = ?` stubs read as `None` (a stub left in does not hide the
    rest): the tree of its longest leading run of lines that parses, the tokens up to where the tokenizer stops."""
    text = STUBBED.sub(r"\1None\2", script)
    tokens: list[tokenize.TokenInfo] = []
    try:
        tokens.extend(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    lines = text.splitlines()
    for end in range(len(lines), -1, -1):
        try:
            return ast.parse("\n".join(lines[:end])), tokens
        except (SyntaxError, ValueError):
            continue
    return ast.Module(body=[], type_ignores=[]), tokens


def constants_read(script: str) -> Constants:
    """`script`'s blocks (`Constants`), read as code: a block opens at a comment token `# CONVENTIONS` or `# OUTPUTS`
    at column 0 (never text inside a string or docstring) and ends at the next such comment of capitals or the
    first definition; its constants are the module-level assignments to UPPER_CASE names within it, each name once
    (its first assignment, with the comment closing that line); OUTPUTS' names are the string literals of its
    assignments' values, read from the tree."""
    tree, tokens = code_of(script)
    marks = [(token.start[0], "conventions" if CONVENTIONS_MARK.match(token.string) else
              "outputs" if OUTPUTS_MARK.match(token.string) else "other")
             for token in tokens if token.type == tokenize.COMMENT and token.start[1] == 0
             and (CONVENTIONS_MARK.match(token.string) or OUTPUTS_MARK.match(token.string)
                  or token.string[1:].strip().isupper())]
    said = {token.start[0]: token.string[1:].strip() for token in tokens
            if token.type == tokenize.COMMENT and token.start[1] > 0}
    defined = min((node.decorator_list[0].lineno if node.decorator_list else node.lineno for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), default=None)
    edges = [line for line, _ in marks] + ([defined] if defined is not None else [])

    def within(kind: str) -> list[ast.stmt]:
        found = []
        for line, which in marks:
            if which != kind:
                continue
            end = min((edge for edge in edges if edge > line), default=None)
            found += [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                      and node.lineno > line and (end is None or node.lineno < end)]
        return found

    conventions: dict[str, str] = {}
    for node in within("conventions"):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for name in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                if isinstance(name, ast.Name) and UPPER.match(name.id) and name.id not in conventions:
                    conventions[name.id] = said.get(node.end_lineno, said.get(node.lineno, ""))
    outputs = [part.value for node in within("outputs") if node.value is not None for part in ast.walk(node.value)
               if isinstance(part, ast.Constant) and isinstance(part.value, str)]
    has_outputs = any(which == "outputs" for _, which in marks)
    return Constants(tuple(conventions.items()), tuple(outputs) if has_outputs else None)


def layout(first: str) -> int:
    """1 where a first message holds design B's order: its sections `TASKFIRST_ORDER` (the described inputs where it
    has them), How to work, Library, Time and Budget, each a `# ` line in that order, and `TASKFIRST_CLOSE` its last
    line; else 0."""
    at = 0
    for head in (*TASKFIRST_ORDER, TASKFIRST_HOW, TASKFIRST_LIBRARY, "Time", "Budget"):
        found = re.compile(rf"^# {re.escape(head)}\n\n", re.M).search(first, at)
        if found is None:
            if head == TASKFIRST_ORDER[3]:      # the described inputs, where the unit has any
                continue
            return 0
        at = found.end()
    return int(first.rstrip("\n").endswith(f"\n{TASKFIRST_CLOSE}"))
