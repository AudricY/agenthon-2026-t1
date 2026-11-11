"""N11, the agent: drives one unit from the command line to the close.

    python3 -m n11.solve solve --task-dir /input --out /app/output

Start at: `main` and `solve` at the bottom of the file (the process: the client, the watchdog, the close), then
`N11.solve` (everything before the loop), `N11.loop` (one iteration is one request) and `N11.execute` (one call of
a reply). docs/architecture.md maps the other modules and defines the words used below.

The steps
---------
Every step runs under the clock the model client holds (`client.deadline`), and the unit's `REQUESTS` requests count
the selection call and the draft as well as the loop.

1. Read the unit (agent/common): the instruction with its canary lines out, the path map, the data digest, the output
   files the instruction names and the properties it states of them (agent/n11/props.py).
2. Pick library functions, twice. Without a request (agent/n11/library.py): the instruction is read for the aliases
   the library's docstrings name, and the functions it names this way are pointed at, their cards carried by the
   system prompt and named on the first message's last line (`pointer_line`). Then the selection call
   (agent/n11/selection.py): one bounded request naming the library functions a correct solution would call. Where
   it picks, its shelves and its picks' cards take the index's and the pointer's cards' place. The cards are then
   made whole, the index trimmed to fit them (`made_whole`).
3. Draft once (agent/n11/drafts.py): one request for a complete script, its script run and its output files held
   aside; they are written to the output at the close only where the loop leaves nothing there to grade.
4. Loop (`N11.loop`). The model replies in fenced blocks (`FENCED_PART`): each block a file saved under work/, after
   which the harness runs work/main.py (`after_write`), or the commands `run` and `finish` on lines of their own.
   Every observation is built by code under a byte bound: the run's exit, the ends of its output, its traceback, the
   output's digest, the findings (agent/n11/props.py, agent/n11/sanity.py), the checks and the budget line.
   `inspect` and `lib` are never offered: an `inspect` still asked is answered from the first message's description
   where it holds the file, a `lib` in one line.
5. Close (`solve`, the function): an earlier clean output put back where the last one broke a rule (`put_back`), the
   draft's files written where nothing else is there (`drafts_close`), then common.guard's close. The watchdog does
   the same if the clock runs out first, and the process exits 0 whatever happens.

The loop's rules
----------------
- A write the output cap cut keeps its head, to its last whole statement, and asks the rest (`hold`), which the next
  reply's code completes (`resumed`).
- A reply with no call is answered by an ask for one (`ask_call`).
- A text a file already holds is refused (`written`), but read as the finish where it follows a clean run with every
  named output present.
- A `finish` after a write runs that file first (`finish_writes`).
- A finish on a failing run is turned away up to `FINISHFAIL_MAX` times (`finish`).
- Where no named output exists and the clock buys one more reply, the last call asks the whole script and runs it
  with the time left (`call_last`).
- `write` reaches work/ alone, so every output file comes from a run of the model's scripts; the canaries are
  stripped from the instruction and redacted in what the model is shown.

The stops
---------
Each is a `return self.stop(...)` in `N11.loop`: a finish that ends the unit (`counted`), the request cap, the clock
(`ASK_S`), a run past its reserve, the last call, and replies with nothing that could be run. Where the unit ends
with no request left and its output breaks a rule a kept clean output holds, that output is put back (`put_back`)."""
from __future__ import annotations

import argparse
import ast
import codeop
import functools
import hashlib
import importlib.metadata
import io
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import tokenize
import traceback
import warnings
from collections import Counter
from pathlib import Path

from common import canary, contract, digest, guard, house, paths, runner, unit
from common.house import model
from common.qfuse import qf_use
from n11 import commands, drafts, library, props, prompts, sanity, selection
from n11.commands import Call, fit

AGENT = "n11"
REQUESTS = 20               # the loop's request cap, counted at admission
RUN_RESERVE_S = 20.0        # kept from the deadline after the last run: the close, before the watchdog's 10 s
MIN_RUN_S = 5.0             # a run is not started with less time than this
RUN_S = 90.0                # a loop run's limit
ASK_S = 45.0                # no request with less time than this left; and a run leaves it where a request may follow
KEEP_TURNS = 3              # observations older than the last three turns are cut to their first line
NO_CALL_CAP = 3             # replies in a row with no call before the unit ends as it stands
LAST_CALL_S = 2 * ASK_S     # under this many seconds left the time buys one more reply and not two
FINISHFAIL_MAX = 3          # the finishes on a failing run turned away a unit; the next one counts
PART = ".part"              # a cut write's kept head, beside its file (work/<name>.py.part), never run
HEAD_LINES = 4              # the kept head's last lines the continuation ask quotes, at most
HEAD_BYTES = 600            # and their bytes (a longer last line quoted by its end)
HELD_BYTES = 2_000          # the continuation ask's bound, its budget line kept whole
PUTBACK_RULES = ("formula", "row_sum", "fraction")  # the properties an earlier clean output is put back for
KEPT = "clean-output"               # the last clean run's output, copied beside work/ (`keep_clean`)
KEPT_BYTES = 8 * 1024 * 1024        # an output larger than this in all is not copied, and none is kept then
HERE = Path(__file__).resolve().parent

LIBRARIES = ("numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, openpyxl, matplotlib (the Agg "
             "backend), plotly, seaborn, TA-Lib (imported as talib) and backtrader")


def release(version: str) -> tuple[int, int]:
    """A version's major and minor numbers, (0, 0) where it names none."""
    found = re.match(r"(\d+)(?:\.(\d+))?", version or "")
    return (int(found.group(1)), int(found.group(2) or 0)) if found else (0, 0)


def installed() -> dict[str, str]:
    """The versions of `API_DISTRIBUTIONS` this process runs under, read from the installed distributions; one not
    installed is left out."""
    versions: dict[str, str] = {}
    for name in API_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


@functools.lru_cache(maxsize=1)
def api_card() -> str:
    """The runtime paragraph of the system prompt: the Python and library versions this process runs under
    (`installed`), and for each library the calls its version removed or changed (`API_NOTES`, the newest note its
    version reaches)."""
    versions = installed()
    notes, said = [], set()
    for name, least, note in API_NOTES:
        if name in versions and name not in said and release(versions[name]) >= least:
            notes.append(note)
            said.add(name)
    return (f"The runtime: Python {platform.python_version()}"
            + "".join(f", {name} {version}" for name, version in versions.items()) + ". " + " ".join(notes)).strip()


def turns_away(exit_: int | None, requests: int, left: float, turned: int) -> bool:
    """Whether a finish whose run exited `exit_` (None: stopped at its limit) is turned away: the run did not exit 0,
    a request and `ASK_S` are left, and fewer than `FINISHFAIL_MAX` finishes were turned away before it in the unit.
    `finish` also requires the current checks to fail."""
    return exit_ != 0 and requests >= 1 and left >= ASK_S and turned < FINISHFAIL_MAX


# The system prompt: `SYSTEM` filled by `N11.system`, edited by `prompts.clean_parts` (`N11.worded`), and each request
# leaving out the two commands no request offers (`told`).
FENCED_PART ="""You act through the form of your reply. Each file is one fenced block whose opening line names it, \
```python file=work/<name>.py, and is saved whole. After the blocks, work/main.py runs where the reply wrote it (or \
the last file written, where it has an `if __name__ == "__main__":` block and work/main.py does not import it), or the \
file a line `run work/<name>.py` names. These lines, each alone on its line outside any block, are commands: `inspect <path> rows=<n>` describes an \
input data file; `lib name=<function>` or `lib shelf=<module>` shows the library's documentation; `finish` ends the \
task, and nothing else does. What each did comes back in the next message: the run's exit, its output's ends, its \
traceback, the files in the output directory with their values (a digest), the sanity findings, the checks and your \
budget."""

FENCED_EXAMPLE = """An example reply: one block opened by ```python file=work/main.py holding the whole script, which \
reads the inputs, computes each value (a function log_returns(prices), and the like) and writes every output file, \
and nothing after it: main.py is run."""

SYSTEM = """You are an expert quantitative developer. You solve a task by writing a Python 3 script and running \
it, unattended, with no network; hidden tests then check the files your scripts wrote into the output directory \
against the task.

{tools}

How to work:
1. Read the task, its paths and its input files, which the first message describes in full, and {settle} \
Start by writing code: `inspect` a file only where its description leaves you short.
2. Write the whole solution as one script, work/main.py, that reads the inputs, computes each quantity (a quantity \
the library computes is a call of its function) and writes every output file the task asks for.
3. A write of work/main.py runs it: read each run. Where you split the script into modules, write them first and \
main.py last; a module's write runs work/main.py only where main.py imports it and its last run was not clean: after \
a clean run, where you change a module, run work/main.py after it in the same reply.
4. Where a run, a check or a sanity finding fails, fix the failing piece: write that one file again, whole.
5. Read each run's digest of the output files against what the task says of each value (its sign, its range, its \
order: ES at least VaR, a probability in [0, 1]); finish when the checks pass and the values hold.

A reply that runs past the output limit is cut, and a write the cut falls in is not saved, only its complete \
definitions as a draft. So write work/main.py whole, in one reply, where one reply holds it; where a reply would be \
cut, split the script into as few files as fit, each a module work/main.py imports, and for no other reason; after a \
cut, write the rest one file a reply.

{api}

Unless the task says otherwise:
{conventions}

Every script must:
- read its inputs from the true paths given under "Paths": the paths in the task's text are those of another machine;
- write every output file the task asks for, with exactly the names, formats, keys and columns it gives, directly \
into the output directory given under "Paths", and nothing anywhere else but work/ and /tmp;
- use only the Python standard library, qf and these installed packages: {libraries}; no network, no pip, no \
subprocess;
- be deterministic: seed any random generator the task does not seed itself;
- write JSON with plain Python numbers (convert numpy values with float() or int());
- never copy the task's text into an output file, and never write a file named reward.json; `<canary>` in the task \
or the data stands for a watermark string: never write it, or an input field that holds it, into any output file;
- end within its run's limit of {run:.0f} seconds, printing a short line a step at most.

{example}

{library}

{index}"""

WITHHELD_LINES = (  # the reply form's lines of `inspect` and `lib`, which no request offers
    " `inspect <path> rows=<n>` describes an input data file;",
    " `lib name=<function>` or `lib shelf=<module>` shows the library's documentation;")
STEP_ONE = ("Start by writing code: `inspect` a file only where its description leaves you short.",
            "Start by writing code.")   # How to work's first step, and as a request without `inspect` says it


def told(system: str) -> str:
    """The loop's system prompt as a request sends it: the reply form's paragraph (as `prompts.clean_tools` words it)
    without the lines of `inspect` and `lib` (`WITHHELD_LINES`), and How to work's first step without its `inspect`
    clause (`STEP_ONE`)."""
    part = prompts.clean_tools(FENCED_PART)
    said = part
    for line in WITHHELD_LINES:
        said = said.replace(line, "", 1)
    system = system.replace(part, said, 1)
    return system.replace(STEP_ONE[0], STEP_ONE[1], 1)


LIBRARY_USE = library.USE.format(cards="the cards below")   # when to call the library, how to read a card
LOOKUP_CLAUSE = ", or `lib` on a name in the index"         # its clause naming the lookup, which no request
LIBRARY_USE_NEVER = LIBRARY_USE.replace(LOOKUP_CLAUSE, "", 1)   # offers
CONVENTIONS = ("Before computing, apply every convention the task states: its day count, annualisation, ddof, "
               "compounding, sign, lag and rounding rules.")
SETTLE = ("settle every convention the task states (day counts, annualisation, ddof, compounding, signs, rounding) "
          "before computing anything.")     # How to work's first step, its sentence on conventions
TIMING_LINES = (    # textbook conventions, one line each
    "- a signal computed from bar t's close trades no earlier than bar t+1's open;",
    "- a model's training window for predicting row t ends at row t-1: it never holds row t;",
    "- align predictions with their targets by index (date or key), never by position after a shift;")
MISSING_LINES = (   # rows with a value missing
    "- before a correlation, drop the rows where either side is NaN;",
    "- \"valid observations\" are the rows left after dropping NaN, never values filled in;")
CODE_LINES = (
    "- define each formula once, as a function, and call it wherever it is needed;",
    "- prefer an installed library's function (numpy, scipy, statsmodels, arch, qf) to a hand-written algorithm.")
CONVENTION_LINES = (*TIMING_LINES, *MISSING_LINES, *CODE_LINES)
API_DISTRIBUTIONS = ("numpy", "pandas", "scipy", "statsmodels", "arch", "scikit-learn")
API_NOTES = (   # the calls the runtime's versions removed or changed, by major version
    ("numpy", (2, 0), "numpy 2: `np.trapezoid` (there is no `np.trapz`), `np.prod` (no `np.product`), `np.nan` (no "
                 "`np.NaN`), `np.isin` (no `np.in1d`); a `np.random.RandomState` draws Student-t by `.standard_t(df)` "
                 "(it has no `.t`)."),
    ("scipy", (1, 14), "scipy: an `integrate.quad` integrand returns a Python float (`float(...)`, never a one-element "
                 "array); `integrate.simpson` and `integrate.cumulative_trapezoid` (no `simps`, `cumtrapz`); "
                 "`linalg.pinv` (no `pinv2`); `optimize.minimize(..., method=\"L-BFGS-B\").hess_inv` is an operator: "
                 "`.todense()` before indexing it."),
    ("pandas", (3, 0), "pandas 3: copy-on-write, so a chained assignment (`df[\"a\"][i] = x`) changes nothing: "
                  "`df.loc[i, \"a\"] = x`; `.iloc[k]` for a position (`s[k]` on a labelled index is a label); "
                  "`.ffill()` and `.bfill()` (no `fillna(method=...)`); `pd.concat` (no `DataFrame.append`); "
                  "frequencies `\"ME\"` and `\"h\"` (not `\"M\"`, `\"H\"`); text columns are dtype `str`."),
    ("pandas", (2, 0), "pandas 2: `.iloc[k]` for a position; `pd.concat` (no `DataFrame.append`)."))

# The observations' words.
SAME_FRAME =("same failing line as the previous run, {where}: `{source}`; the fix left the line that fails as it was. "
              "Change that line or the call it makes (a different fix), and check the call against the runtime's "
              "versions (the system prompt's runtime paragraph).")
ENTRY = ("entry: the stitched work/main.py defines `{name}` and nothing at its top level calls it, so it ran and "
         "wrote nothing: add `if __name__ == \"__main__\":` and the call that runs it.")
CUT_CALL = ("not run: your reply was cut at the output limit inside this call, so what it held was lost and no file "
            f"was written: it does not fit one reply, so split the script {commands.SPLIT}, the first now.")
CUT_FENCED = ("not written: your reply was cut at the output limit before any file block arrived, so no file was "
              f"written. Split the script {commands.SPLIT}; reply with the first file as a block opened by "
              "```python file=work/<name>.py.")
CUT_DRAFT = ("not written: your reply was cut at the output limit inside this write of {named}, so the file was not "
             f"saved; {{kept}}. It does not fit one reply: split it {commands.SPLIT}, the first now.")
CUT_KEPT = "its complete definitions are kept in {draft}{defines}, the cut falling {where}"
CUT_NONE = "nothing in it was complete enough to keep"
CUT_SHORT = (" {n} replies in a row were cut: write one small file now, a module work/main.py imports holding one or "
             "two functions, and the next after it.")
SHORT_STREAK = 3                # cut replies in a row before one small file is asked
CUT_HELD = ("not written: your reply was cut at the output limit inside this write of {named}, so the file was not "
            "saved whole; what arrived is kept up to its last complete line, which ends:\n\n```python\n{ends}\n```\n\n"
            "Reply with the rest of {named} only: every line after that last one, {through}. One ```python block of "
            "code, no prose before or after it and no line already written above; a reply past the limit is cut "
            "again.")         # the continuation ask
CUT_THROUGH = {True: "through the code that runs it and writes every output file", False: "through the file's end"}
STITCHED = "the rest joined under what was kept of {named}, the file whole: "     # the stitched write's observation
JUNCTION_SHOWN = "where what was kept meets the rest you sent, as joined:\n{lines}\n{opened}"
JUNCTION = (JUNCTION_SHOWN + "Reply with the rest of {named} again: every line after what was kept, {through}, in one "
            "```python block of code, joined under what was kept as before, at the depth of what was kept; or write "
            "{named} again whole.")
JUNCTION_MARK = "       | --- the rest you sent ---"             # the line between the two sides
NEVER_CLOSED = "The {what} opened at line {line}, in what was kept, is never closed by the rest you sent. "
JOIN_REFUSED = ("write refused: {named} already holds this join of what was kept and the rest you sent, which does not "
                "compile: {error}; nothing was written and nothing ran.")    # a failed join sent again
KEPT_BROKEN = ("The error stands in what was kept, above where the rest you sent joins it, so no rest can mend it: "
               "write {named} again whole, from its first line; what was kept is dropped.")   # said once a file
UNCLOSED = re.compile(r"'([(\[{])' was never closed")             # Python's words for a bracket left open
UNTERMINATED = "unterminated triple-quoted string literal"         # and for a string
FINISH_REFUSED = ("finish refused: {n} sanity finding(s) stand; write a fix or write the line `finish "
                  "accept: <why>`\n{lines}")
UNCHANGED = ("write refused: {named} unchanged from turn {turn}, which gave: {gave}; nothing was written and nothing "
             "ran. Change the file to fix it, or run it as it is: {run}")
HELD = ("write refused: {named} already holds this text; nothing was written and nothing ran. Change the file to fix "
        "it, or run it as it is: {run}")
CYCLE = ("{named}: a cycle among {k} versions: this text is turn {turn}'s again, which gave: {gave}. Before the next "
         "edit, print the failing line's operands (shapes, dtypes, keys) and read them.")
NO_RUN = "(no run: the file was not written whole and compiling)"     # `after_write`'s line after a write not saved
# the line heading an unchanged write's observation where it is read as the finish; and the run line of one refused,
#   the file unchanged since the turn it matched and, after a clean run, the outputs missing
RESEND_READ = ("{named} is unchanged from turn {turn}, whose run exited clean with every named output present: read as "
               "the finish.")
RESEND_SINCE = "(no run: {named} is unchanged since turn {turn})"
RESEND_MISSING = ("(no run: {named} is unchanged since turn {turn}, whose run exited clean; these named outputs are now "
                  "missing: {missing}; write the file that writes them, or {run})")
CYCLE_RECURRENCES = 3           # returns to an earlier version of a file before the cycle is named
CYCLE_WINDOW = 6                # the file's last versions a cycle is read among
MALFORMED = ("not run: this call is malformed: {why}. Call it again in the form its tool gives: write(path, text), "
             "run(path), inspect(path, rows), lib(name) or lib(shelf), finish().")
FINISH_FAILED = ("finish: {n} check(s) fail:\n{lines}\nFix them and finish again; a second finish ends the task "
                 "whatever the checks say.")
FINISH_FINDINGS = "\nsanity findings, which stand too:\n{lines}"
FINISHFAIL_SAID = ("finish not taken: the last run, {named}, failed, so the task goes on. Fix it so "
                   "that it runs clean, then finish. That run:")   # the failed run's observation follows it
LAST_CALL = ("last call: no output file the task names exists yet, and the budget buys this one reply alone. Write "
             "work/main.py, whole, as one block opened by ```python file=work/main.py, so that it writes every "
             "output file the task names from what the files under work/ already compute: each output file in "
             "its own try/except, so that one failure does not stop the others; where nothing computes a value, "
             "leave it out rather than invent one, and still write the file with what is there. Nothing else in "
             "this reply: the harness runs work/main.py with all the time left, and the task ends on its output.")
LAST_CALL_BYTES = 3_000         # the last call's evidence (the files under work/ and their runs), at most
NOCALL_ASK = ("no file: your reply held no file block and no line `finish`, so nothing was written or run; prose is "
              "not read.")      # the answer to a reply with no call, before the call asked (`NOCALL_NOW`)
NOCALL_NOW = "Write it now: {call}."
NOCALL_CUT = ("your reply was cut at the output limit inside its call, so what it held was lost and no file was "
              f"written: it does not fit one reply, so split it {commands.SPLIT}.")     # before the ask, after a cut
NOCALL_CALL = {     # what the ask asks: the file the reply's prose named, after a cut the first part, else the next
    "named": "{named}, whole, as one block opened by ```python file={named}",
    "cut": "the first of those files, whole, as one block opened by ```python file=work/<name>.py",
    "next": ("work/main.py, whole, as one block opened by ```python file=work/main.py, or the next module "
             "where you split it; or the line `finish` where the output is done")}
PROSE_FILE = re.compile(r"(?<![\w/.-])(?:\./)?(?:" + re.escape(commands.WORK)
                        + r"/)?([A-Za-z_]\w*\.py)\b(?![\w/])")     # a file a reply's prose names (`prose_files`)
RESERVE_END = ("a run went past the request's reserve, so no request could follow it: it had the time left, and the "
               "unit ends on its output")
NOT_IMPORTED = "; it does not import {named}, so nothing ran {named}: import it in {main} where it is needed"
SELF_RUN = (" ({named}, by its own `if __name__ == \"__main__\":` block: {main} does not import it, so {main} did not "
            "run)")
NO_WORK = re.compile(r"No module named '" + re.escape(commands.WORK) + r"[.']")   # a package-style import's failure
LIB_UNSAID = (  # each text of the index and of the selection call's section naming the lookup, and as said without it
    #             (`library_unsaid`): the scored index's head, the section's head, the tail of an index cut at its
    #             bound; a cut card's last line is `CARD_CUT_SAID`'s
    (library.SCORED_HEAD, library.SCORED_HEAD.replace(" after `others:` (`lib name=<function>` shows any one's card):",
                                                      " after `others`:", 1)),
    (selection.SHELVES_HEAD, selection.SHELVES_HEAD.replace(
        "; `lib name=<function>` shows a function's card and `lib shelf=<module>` a shelf's, these shelves or the "
        "others:", ", then the other shelves:", 1)),
    ("\n[... the rest by `lib shelf=<module>`]", "\n[...]"))
LIB_ASKED = {   # the one-line answer to a lookup still asked, by whether the system prompt carries cards
    True: ("not run: the library lookup is not offered; the cards of the functions the task needs are above, in the "
           "system prompt."),
    False: "not run: the library lookup is not offered; the system prompt's index above names the library's functions."}
MALFORMED_FORMS = {"inspect": " inspect(path, rows),", "lib": " lib(name) or lib(shelf),"}  # the forms of the two
#                                                     commands no request offers, left out of `MALFORMED`'s list


MAIN_FILE = f"{commands.WORK}/main.py"


QF_NAMED = re.compile(r"(?<![\w.])qf(?:\.[A-Za-z_]\w*)+")      # a `qf.` name
IDENTIFIER = re.compile(r"(?<![\w.])[A-Za-z_]\w*")              # a bare name: never another module's (`np.std`)


def functions_named(text: str, lib: library.Library | None) -> tuple[list[library.Function], list[str]]:
    """The library's functions a text names, by an exact lookup (`Library.find`), nothing fuzzy: each `qf.<name>` and
    `qf.<shelf>.<name>`, and each bare index name, written as code or as a word, in the order first named, once each;
    and the `qf.` names the library lacks (a shelf's name alone is none), the only names a text holds that claim the
    library. The selection call's reply is read by it."""
    if lib is None or not text:
        return [], []
    names = {function.name: function for function in lib.functions}
    said = sorted([(match.start(), match.group(0)) for match in QF_NAMED.finditer(text)]
                  + [(match.start(), match.group(0)) for match in IDENTIFIER.finditer(text) if match.group(0) in names])
    found: dict[str, library.Function] = {}
    unknown: dict[str, None] = {}
    for _, name in said:
        function = lib.find(name) if name.startswith(f"{library.PACKAGE}.") else names[name]
        if function is not None:
            found.setdefault(function.name, function)
        elif lib.shelf(name) is None:
            unknown.setdefault(name, None)
    return list(found.values()), list(unknown)


def prose_files(text: str) -> list[str]:
    """The `.py` files a reply's prose names, as work/<name>.py, in the order first named, once each; a path under
    another directory, or a standard library module's name (which `write` refuses), is none."""
    found: list[str] = []
    for match in PROSE_FILE.finditer(runner.THINK.sub("", text or "")):
        named = f"{commands.WORK}/{match.group(1)}"
        if named not in found and not commands.shadows(named):
            found.append(named)
    return found


def pointer_line(found: list[tuple[library.Function, str]]) -> str:
    """The pointer: the library's functions the instruction names by an alias, each to be called for its quantity."""
    if not found:
        return ""
    named = ", ".join(f"`{function.call}` for {alias}" for function, alias in found)
    return f"the library has {named}: call each for its quantity rather than writing its formula"


def library_unsaid(index: str) -> str:
    """The system prompt's index, or the selection call's section in its place, each of its texts naming the lookup
    said as `LIB_UNSAID` says it; the rest as it came."""
    for naming, said in LIB_UNSAID:
        index = index.replace(naming, said, 1)
    return index


CARD_CUT = re.compile(r"\[cut to fit: [^\]\n]*\]")   # a trimmed card's last line (library.CARD_CUT), which names `lib`
CARD_CUT_SAID = "[cut to fit]"


def cards_unsaid(cards: str) -> str:
    """Cards whose cut ones end on `[cut to fit]`, the lookup they name left out (`CARD_CUT_SAID`)."""
    return CARD_CUT.sub(CARD_CUT_SAID, cards)


def first_line(text: str) -> str:
    return (text or "").strip().split("\n", 1)[0]


def text_write(call: Call) -> bool:
    """A `write` of a file that is no `.py` file (a `.txt`, `.json` or `.csv` a block named), which never runs: the
    run after a reply's writes, and the run before a finish, pass over it."""
    return call.name == "write" and not str(call.arguments.get("path", "")).endswith(".py")


def logged(text: str) -> str:
    """A message for the log: any line carrying a GUID-shaped token dropped."""
    return canary.strip(text)[0]


def asked(said: str, short: str, bound: int, tail: str) -> str:
    """A cut call's observation within `bound` bytes before `tail`: `said` and the `SHORT_STREAK` ask `short` on one
    line where they fit; else the ask kept whole after `said` cut, with the tail (a long draft's names and path cut
    the ask away before)."""
    if short and commands.size(said + short + ("\n" + tail if tail else "")) > bound:
        return fit(said, bound, short.strip() + ("\n" + tail if tail else ""))
    return fit(said + short, bound, tail)


OPENS = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)")      # a top-level line opening a definition


def draft(code: str) -> tuple[str, list[str], str]:
    """A cut script's complete top-level definitions (imports, functions, classes, plain assignments), its last
    definition left out where the cut may have fallen in it (no top-level line follows it); the names it defines and
    where the cap cut (the definition the last top-level line after them opens, where one does)."""
    complete = runner.complete_lines(code)
    kept = runner.parsing_prefix(complete)
    tree = runner.parse(kept) if kept else None
    if tree is None:
        return "", [], "before any complete statement"
    lines = kept.split("\n")
    nodes = list(tree.body)
    later = [line for line in complete[len(kept):].split("\n") if line.strip() and not line[0].isspace()
             and not line.startswith("#")]
    opened = OPENS.match(later[-1]) if later else None
    where = f"inside `{opened.group(1)}`" if opened else "after the last complete statement" if later else \
        "at the script's end"
    if not later and nodes and isinstance(nodes[-1], runner.DEFINITIONS):
        last = nodes.pop()
        where = f"inside `{last.name}` (left out of the draft)"
    keep = (ast.Import, ast.ImportFrom, *runner.DEFINITIONS, ast.Assign, ast.AnnAssign)
    chunks, names = [], []
    for node in nodes:
        if not isinstance(node, keep):
            continue
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        chunks.append("\n".join(lines[start - 1:node.end_lineno]))
        if isinstance(node, runner.DEFINITIONS):
            names.append(node.name)
    text = "\n\n".join(chunks).strip("\n")
    return (text + "\n" if text else ""), names, where


def opened_fence(text: str, closed: bool = False) -> str:
    """A write's text with the fence the model opened it with dropped (its first line opening a block) and, where
    `closed`, the one closing it (its last such line); a cut text keeps its last line, the cut one."""
    lines = (text or "").split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if closed:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip() and set(lines[-1].strip()) == {"`"}:
            lines.pop()
    return "\n".join(lines)


def complete_head(code: str) -> str:
    """Keep newline-terminated physical lines, including those inside unfinished Python constructs."""
    head, mark, _ = code.rpartition("\n")
    return head + mark


def unfinished(head: str) -> bool:
    """`head` does not compile only for its end, a statement left open there (a bracket, a string, a block's header
    or a clause still due), which more lines could finish (`codeop`); False where it compiles or a line above its end
    is broken."""
    if runner.compiles(head):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return codeop.compile_command(head, "<head>", "exec") is None
    except (SyntaxError, ValueError, OverflowError, MemoryError, RecursionError):
        return False


def boundary_head(code: str) -> str:
    """A cut write's head held at a statement boundary: its complete lines (`complete_head`) to the end of the last
    statement they finish, never inside a bracket or a string nor on a block's header or a decorator with no body,
    and the comment and blank lines straight after it; its complete lines where they compile, where a line above
    their end is broken (`unfinished`), or where no statement ends in them."""
    head = complete_head(code)
    if not unfinished(head):
        return head
    end, line = 0, []
    try:
        for token in tokenize.generate_tokens(io.StringIO(head).readline):
            if token.type == tokenize.NEWLINE:
                if line and not (line[-1].string == ":" or line[0].string == "@"):
                    end = token.start[0]
                line = []
            elif token.type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT,
                                    tokenize.ENCODING, tokenize.ENDMARKER):
                line.append(token)
    except (tokenize.TokenError, SyntaxError):
        pass
    lines = head.split("\n")
    while end and end < len(lines) and (not lines[end].strip() or lines[end].lstrip().startswith("#")):
        end += 1
    return "".join(text + "\n" for text in lines[:end]) if 0 < end < len(lines) else head


def rewritten(head: str, code: str) -> bool:
    """A reply asked for a head's rest that wrote the file again whole instead: it compiles by itself and defines and
    imports at its top level every name the head's compiling prefix does, a definition among them (a rest never
    repeats a line already written)."""
    kept, tree = runner.parse(runner.parsing_prefix(head)), runner.parse(code)
    if kept is None or tree is None or not runner.compiles(code):
        return False

    def names(module: ast.Module) -> tuple[set[str], set[str]]:
        return ({node.name for node in module.body if isinstance(node, runner.DEFINITIONS)},
                {alias.asname or alias.name for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 for alias in node.names})

    (defined_, imported), (redefined, reimported) = names(kept), names(tree)
    return bool(defined_) and defined_ <= redefined and imported <= reimported


def stitched(head: str, code: str, cut: bool) -> tuple[str, str, str]:
    """A continuation's join as `N11.resumed` makes it: the join, how it was read, and the rest as stitched (its
    complete lines where the cap cut it again, `cut`, else the code newline-ended): the head alone where a cut rest
    holds no complete line, the code alone where it wrote the file again whole (`rewritten`), else `runner.stitch`
    under the head, the head's end the rest repeats taken once."""
    rest = complete_head(code) if cut else code.rstrip() + "\n"
    if cut and not rest.strip():
        return head, "nothing added", rest
    if not cut and rewritten(head, code):
        return code, "written again whole", rest
    joined, how = runner.stitch(head, f"```python\n{rest}```\n", overlap=True, again=cut)
    return joined, how, rest


def ends(head: str) -> str:
    """A kept head's last lines as `CUT_HELD` quotes them, as written: at most `HEAD_LINES` that are not blank (the
    blank ones between them kept) within `HEAD_BYTES`, a last line longer than that by its end."""
    lines = head.rstrip("\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    picked, shown = [], 0
    for line in reversed(lines):
        if line.strip():
            if shown == HEAD_LINES or (picked and commands.size("\n".join([line, *picked])) > HEAD_BYTES):
                break
            shown += 1
        picked.insert(0, line)
    while picked and not picked[0].strip():
        picked.pop(0)
    text = "\n".join(picked)
    if commands.size(text) > HEAD_BYTES:
        text = text.encode("utf-8")[-HEAD_BYTES:].decode("utf-8", errors="ignore")
    return text


def junction(joined: str, start: int, mark: int | None) -> str:
    """Where a join meets, as `JUNCTION` shows it: `joined`'s lines before line `start` (from 1, the rest's first)
    and from it, on each side at most `HEAD_LINES` that are not blank (the blank ones between them kept) within
    `HEAD_BYTES`, numbered as a write's syntax error shows its lines (`commands.around`), line `mark` marked, and
    `JUNCTION_MARK` between the sides."""
    lines = joined.split("\n")

    def side(numbers: range) -> list[str]:
        picked, shown, size = [], 0, 0
        for number in numbers:
            line = lines[number - 1]
            if line.strip():
                if shown == HEAD_LINES:
                    break
                shown += 1
            row = f"{'>' if number == mark else ' '} {number:4d} | {line[:commands.SOURCE_CHARS]}"
            if picked and size + commands.size(row) > HEAD_BYTES:
                break
            picked.append(row)
            size += commands.size(row) + 1
        while picked and not picked[-1].split("|", 1)[1].strip():
            picked.pop()
        return picked

    before = side(range(min(start, len(lines) + 1) - 1, 0, -1))[::-1]
    return "\n".join([*before, JUNCTION_MARK, *side(range(start, len(lines)))])


def never_closed(joined: str, start: int) -> str:
    """`NEVER_CLOSED` where `joined` does not compile for a bracket or a triple-quoted string opened before line
    `start` (what was kept) and never closed; "" otherwise."""
    try:
        compile(joined, "<join>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        found = UNCLOSED.fullmatch(exc.msg or "")
        what = f"`{found.group(1)}`" if found else "string" if (exc.msg or "").startswith(UNTERMINATED) else ""
        if what and exc.lineno and exc.lineno < start:
            return NEVER_CLOSED.format(what=what, line=exc.lineno)
    except (ValueError, MemoryError, RecursionError):
        pass
    return ""


def compile_error(text: str) -> tuple[str, int | None]:
    """`text`'s compile error as a write's observation names it (`commands.Workspace.write`) and its line; ("", None)
    where it compiles."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(text, "<join>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"{type(exc).__name__} at line {exc.lineno}: {exc.msg}", exc.lineno
    except (ValueError, MemoryError, RecursionError) as exc:
        return type(exc).__name__, None
    return "", None


def defined(text: str) -> list[str]:
    """A file's top-level functions, each with its arguments' names, and its classes, in order (the last call's
    evidence); [] where it does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            names = [arg.arg for arg in (*args.posonlyargs, *args.args)]
            names += [f"*{args.vararg.arg}"] if args.vararg else ["*"] if args.kwonlyargs else []
            names += [arg.arg for arg in args.kwonlyargs] + ([f"**{args.kwarg.arg}"] if args.kwarg else [])
            found.append(f"{node.name}({', '.join(names)})")
        elif isinstance(node, ast.ClassDef):
            found.append(f"class {node.name}")
    return found


CODE_START = re.compile(r"^(?:\s|def |class |import |from |if |elif |else:|for |while |try:|except|finally:|with |"
                        r"return |raise |@|#|[A-Za-z_][\w.]*\s*(?:=|\(|\[|\.|,|\+=|-=))")


def continuation_code(reply: str) -> str:
    """The code a continuation's reply holds: its blocks' lines in order with every fence line dropped (a fence the
    reply restarts inside a block, ```` ```python ```` again, opens nothing new: the code runs on), prose outside the
    blocks dropped; where the reply opens with code and no fence, that code is a block its first bare fence closes;
    the reply whole where it holds no fence and opens with code, else nothing."""
    lines = runner.THINK.sub("", reply or "").split("\n")
    fences = [k for k, line in enumerate(lines) if line.strip().startswith("```")]
    head = [line for line in (lines[:fences[0]] if fences else lines) if line.strip()]
    if not fences:
        return "\n".join(lines).strip("\n") if head and CODE_START.match(head[0]) else ""
    inside, kept = bool(head) and bool(CODE_START.match(head[0])), []
    for line in lines:
        fence = line.strip()
        if fence.startswith("```"):
            inside = True if fence.strip("`").strip() else not inside
            continue
        if inside:
            kept.append(line)
    return "\n".join(kept).strip("\n")


def task_text(task, pathmap, data: str, described: str, named: tuple[str, ...], edit) -> str:
    """The loop's first message up to its budget: the task, the paths, the input data and files described and the
    output files named, each a `# ` section, in that order, as `edit` says them (`N11.cleaned`)."""
    files = ("The task names these output files; the checks look for each at its path in the output directory: "
             + ", ".join(named) + "." if named else
             "The task names no output file by name; write each one it describes into the output directory.")
    sections = {"Task": task.instruction.strip(), "Paths": pathmap.text(), "Input data": data.strip(),
                "Input files, described": described, "Output files": files}
    sections = edit(sections)
    return "".join(f"# {head}\n\n{sections[head]}\n\n" for head in sections
                   if sections[head] or head != "Input files, described")     # that one only where any


class N11:
    """One unit's state and its steps.

    The spine is `solve` (the setup, the selection call, the draft), `loop` (a request an iteration), `execute` (one
    call), `after_write` (the run the harness makes after a write) and `finish` with `counted` (whether a finish ends
    the unit). The other methods serve those, grouped under the rulers below: the clock (`left`, `run_limit`), the
    last call (`last_call_due`, `call_last`, `last_turn`), a cut reply's write kept and completed (`salvage`, `hold`,
    `resumed`, `joined`), a repeated write (`written`), the kept clean output (`keep_clean`, `put_back`), the
    messages (`worded`, `first_turn`, `outgoing`) and a reply with no call (`ask_call`)."""

    def __init__(self, task_dir: Path, out: Path, client, steps: guard.Steps, work: Path, deadline: float) -> None:
        self.task_dir, self.out, self.client, self.log = Path(task_dir), Path(out), client, steps
        self.deadline = deadline
        self.began = getattr(client, "deadline", None)  # the client's deadline at the start (`moved`)
        self.work = Path(work)
        self.unit: unit.Unit | None = None
        self.ws: commands.Workspace | None = None
        self.selection = selection.Selection()
        self.drafting: drafts.Round | None = None       # the draft's round, where it ran
        self.named_files: tuple[str, ...] = ()          # the output files the task names (the draft's round reads them)
        self.messages: list[dict] = []
        self.counts: Counter = Counter()
        self.summarised = False
        self.cards = ""                 # the system prompt's cards
        self.fired: list[tuple[library.Function, str]] = []     # the pointer's functions, each with its alias
        self.reminded = False                           # the pointer's reminder said (`reminder`)
        self.inputs_text: str | None = None             # the input files described, as first built (`described`)
        self.parts: dict[str, str] | None = None        # and that description by file
        self.seen: set = set()                          # the runs' errors, by identity
        self.versions: dict[str, list[tuple[int, str]]] = {}    # each file's writes: (turn, sha256), in order
        self.recurrences: Counter = Counter()                   # each file's returns to an earlier version
        self.gave: dict[int, str] = {}                          # each turn's last run, said in a line
        self.runs_at: dict[int, commands.Ran | None] = {}       # each turn's last run, `gave`'s record (None: refused)
        self.outcome: dict[str, tuple[int, str]] = {}           # each file's last run: its turn, said in a line
        self.last_ran: dict[str, tuple[int, str, bool]] = {}    # each file's last run: (turn, its sha256 then, whether
        #                                                         it ran clean: exit 0, the checks passing, no finding)
        self.shown_run: tuple[int, str, int | None, str] | None = None  # the last run the model was shown (its turn,
        self.tail_budget = ""                           # file, exit and observation), and the budget line `tail` ended
        #                                                 that observation with
        self.last_frame = ""                            # the previous run's innermost failing line, "" where none
        self.entry = ""                                 # the stitched script's uncalled entry, said at its run
        self.held: tuple[str, str] | None = None        # a cut write's file and its kept head, the rest asked
        self.stitching: Call | None = None              # the write the stitch made, said as stitched
        self.joining: tuple[str, str, str, bool, str, int | None] | None = None  # a join that does not compile: its
        #                       file, its head, its junction's words, whether the head is broken above its end (so the
        #                       error is the head's), the error and its line, said with its write
        self.said_broken: set[str] = set()              # the files whose broken head was said and dropped
        self.resent: tuple[str, str, int, tuple[str, ...]] | None = None    # the last write's unchanged refusal:
        #                                                 ("ended", "read" or "refused", the file, the turn it matched,
        #                                                 the named outputs missing after its run)
        self.finishes = 0
        self.finish_said = ""
        self.sanity_refused = False
        self.no_call_streak = 0                         # replies in a row with no call
        self.cut_streak = 0                             # replies in a row the output cap cut
        self.reply_text = ""
        self.kept: Path | None = None                   # the last clean run's output kept, and its snapshot
        self.kept_snapshot: dict | None = None
        self.ending = ""                                # why a run ended the unit on its output (`RESERVE_END`)
        self.end_kind = ""                              # what stopped the loop (`put_back` reads it)
        self.last_kind = ""                             # what made the last call due: `requests` or `clock`

    def moved(self) -> float:
        """The seconds the client's deadline has moved since the unit began: 0 but under
        `AGENTHON_MODEL_RATE_UNTIMED`, where each 429's wait moves it."""
        now = getattr(self.client, "deadline", None)
        return now - self.began if now is not None and self.began is not None else 0.0

    def left(self) -> float:
        """The seconds left to the loop."""
        return self.deadline + self.moved() - RUN_RESERVE_S - time.time()

    def requests_left(self) -> int:
        return self.client.cap - self.client.usage.requests

    def redact(self, text: str) -> str:
        return canary.redact(text, self.unit.canaries if self.unit else ())

    def kill(self) -> None:
        if self.ws is not None:
            self.ws.kill()
        if self.drafting is not None:   # the draft's script running
            self.drafting.kill()

    # --------------------------------------------------------------------------------------------- the clock

    def run_cap(self) -> float:
        """A loop run's seconds: at most `RUN_S`, and the time left less `ASK_S` where a request could follow it."""
        left = self.left()
        if self.requests_left() >= 1 and left - ASK_S >= MIN_RUN_S:
            return min(RUN_S, left - ASK_S)
        return min(RUN_S, left)

    def run_limit(self) -> tuple[float, float]:
        """A loop run's limit and its reserve's line, `run_cap` both, but where that cap keeps a request's
        reserve below `RUN_S`: there the run's limit is the whole of the time left and the line the cap. A run that
        ends before the line leaves the request its reserve; one that runs past it leaves under `ASK_S`, so no request
        could follow it either way, and it runs on (`past_reserve`)."""
        cap, left = self.run_cap(), self.left()
        if self.requests_left() >= 1 and MIN_RUN_S <= left - ASK_S < RUN_S:
            return left, cap
        return cap, cap

    def past_reserve(self, ran: commands.Ran | None, limit: float, line: float) -> bool:
        """Whether a run given the rest of the clock (`run_limit`) ran past its reserve's line: then no request can
        follow it, and the unit ends on its output (`RESERVE_END`, `run_uncapped`)."""
        if ran is None or limit <= line or ran.seconds <= line:
            return False
        self.counts["run_uncapped"] = 1
        self.ending = RESERVE_END
        self.log("run", f"past the request's reserve at {line:.0f} s: it had the time left, {limit:.0f} s; the unit "
                        "ends on its output")
        return True

    # ---------------------------------------------------------------------------------------- the last call

    def unwritten(self) -> bool:
        """No output file the task names exists (no output file at all where it names none)."""
        if self.ws is None:
            return False
        if self.ws.named:
            return all(not (self.out / name).is_file() for name in self.ws.named)
        return not any(path.is_file() for path in self.out.rglob("*")) if self.out.is_dir() else True

    def last_call_due(self) -> bool:
        """The last call, once a unit, where no output file the task names exists and the budget buys one more reply
        and not two (one request left, or under `LAST_CALL_S` left); the loop's own stops come first."""
        if self.counts["last_call_asked"] or (self.requests_left() > 1 and self.left() >= LAST_CALL_S):
            return False
        return self.unwritten()

    def evidence(self) -> str:
        """The last call's evidence: each .py file under work/ as it stands (work/main.py last), the functions and
        classes it defines, and its last run's line where it ran (changed since, where it was), within
        `LAST_CALL_BYTES`."""
        lines = []
        for path in sorted(self.ws.dir.glob("*.py"), key=lambda each: (each.name == "main.py", each.name)):
            named = f"{commands.WORK}/{path.name}"
            text = path.read_text(encoding="utf-8", errors="replace")
            ran = self.outcome.get(named)
            last = self.last_ran.get(named)
            said = f"; its last run, turn {ran[0]}: {ran[1]}" if ran else "; not run"
            if ran and last is not None and last[1] != sha256(text):
                said += " (changed since)"
            lines.append(f"- {named}: " + (", ".join(defined(text)) or "no function") + said)
        return self.redact(fit("the files under work/ as they stand:\n" + "\n".join(lines) if lines else
                               "no file under work/ yet", LAST_CALL_BYTES))

    def call_last(self) -> None:
        """The last call's ask (`LAST_CALL`, the evidence and the budget) after the conversation's last turn: into its
        last user turn where it ends on one, else a user turn of its own."""
        self.counts["last_call_asked"] = 1
        self.last_kind = "requests" if self.requests_left() <= 1 else "clock"
        ask = f"{LAST_CALL}\n{self.evidence()}\n{self.budget()}"
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1] = {**self.messages[-1], "content": f"{self.messages[-1]['content']}\n\n{ask}"}
        else:
            self.messages.append({"role": "user", "content": ask})
        self.log("last call", f"asked: {self.requests_left()} request(s) and {max(self.left(), 0):.0f} s left; no "
                              "output file the task names")

    def last_turn(self, found: list[Call], turn: int) -> str:
        """The last call's reply: each whole write saved as `write` saves one (a cut one's head or definitions kept),
        nothing else in it run; then work/main.py run with all the time left (`run_last`)."""
        for call in found:
            if call.cut and call.name in ("write", "cut", "unparsed"):
                self.salvage(call, "", final=True)
            elif call.name == "write" and not call.problem and not call.cut:
                said = self.written(call.arguments["path"], call.arguments["text"], "")
                self.log("write", first_line(said)[:200])
        return self.stop(self.run_last(), self.last_kind)

    def run_last(self) -> str:
        """The last call's run: work/main.py as it stands, run with all the time left (no reserve: no request follows),
        and the unit's end on its output (`last_call_output`: a file the task names written by it; `last_call_clean`:
        it exited 0 with the checks passing). The stop's reason."""
        mainpath = f"{commands.WORK}/main.py"
        if not (self.ws.dir / "main.py").is_file():
            self.log("last call", f"no {mainpath} to run")
            return f"the last call: no {mainpath} to run; the unit ends with the output as it stands"
        seconds = self.left()
        if seconds < MIN_RUN_S:
            self.log("last call", f"not run: {max(seconds, 0):.0f} s left")
            return f"the last call: {max(seconds, 0):.0f} s left, under a run's {MIN_RUN_S:.0f} s"
        self.counts["implicit_runs"] += 1
        _, ran = self.ws.run(mainpath, seconds, self.seen, lambda done: "")
        self.ran(ran, seconds)
        self.counts["last_call_output"] = int(ran is not None and bool(ran.wrote) and not self.unwritten())
        self.counts["last_call_clean"] = int(ran is not None and ran.clean)
        self.log("last call", f"{mainpath} ran with the time left: " + ("a named file written" if
                                                                         self.counts["last_call_output"] else
                                                                         "no named file written")
                 + ("; clean" if self.counts["last_call_clean"] else ""))
        return f"the last call: {mainpath} ran with the time left; the unit ends on its output"

    def budget(self) -> str:
        missing = self.ws.missing() if self.ws else []
        return (f"budget: {self.requests_left()} request(s) and {max(self.left(), 0):.0f} s left; "
                + (f"missing: {', '.join(missing[:8])}" + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
                   if missing else "every named output file is present"))

    # -------------------------------------------------------------------------------- the pointer's reminder

    def used(self) -> tuple[bool, set[str]]:
        """Whether any file under work/ imports qf, and the qf functions they call."""
        imported, called = False, set()
        for path in sorted(self.ws.dir.rglob("*.py")) if self.ws else []:
            try:
                found, names = qf_use(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            imported, called = imported or found, called | names
        return imported, called

    def reminder(self, ran: commands.Ran | None) -> str:
        """After a run that exits 0 with a failing check or a finding, where no file under work/ calls a pointed
        function: once a unit, the line saying the task names the method and the function computes it; "" otherwise."""
        if ran is None or not self.fired or self.reminded or ran.exit != 0 or (ran.clean and not ran.findings):
            return ""
        imported, called = self.used()
        if imported and called & {function.name for function, _ in self.fired}:
            return ""
        function, alias = next(((f, a) for f, a in self.fired if f.name not in called), self.fired[0])
        self.reminded = True
        self.log("reminder", f"said: {function.call}")
        said = function.conventions
        return (f"the task names {alias}; `{function.typed}` computes it" + (f" with {said[:300]}" if said else "")
                + "; call it")

    def tail(self, ran: commands.Ran | None, last: bool) -> str:
        """What a run's observation ends with, kept whole: the stitched script's uncalled entry, the same failing
        line, a package-style import's failure, the pointer's reminder and, where the run is the reply's last call,
        the budget's."""
        entry, self.entry = self.entry, ""
        parts = (entry, self.same_frame(ran), self.packaged(ran), self.reminder(ran))
        self.tail_budget = self.budget() if last else ""     # kept for `saw` to drop
        return "\n".join(part for part in (*parts, self.tail_budget) if part)

    def packaged(self, ran: commands.Ran | None) -> str:
        """A run that failed on importing work/ as a package, said with the import's form and the bare name modules
        under work/ import each other by (commands.packaged); "" otherwise."""
        if ran is None or not ran.error or not NO_WORK.search(ran.error):
            return ""
        return commands.packaged_said(commands.packaged(self.ws.dir), ran.frames[-1][0] if ran.frames else ran.path)

    def same_frame(self, ran: commands.Ran | None) -> str:
        """A run that fails on the same source line, in the same file, as the run before it failed on is answered by
        quoting that line and asking for a different fix (`SAME_FRAME`); "" otherwise."""
        if ran is None:
            return ""
        here = f"{ran.frames[-1][0]}\n{ran.line}" if ran.error and ran.frames and ran.line else ""
        before, self.last_frame = self.last_frame, here
        if not here or here != before:
            return ""
        self.counts["same_frame"] += 1
        self.log("run", "the same failing line as the previous run")
        return SAME_FRAME.format(where=f"{ran.frames[-1][0]}:{ran.frames[-1][1]}", source=ran.line)

    # ------------------------------------------------------------------------------------------- the commands

    def execute(self, call: Call, last: bool = False, final: bool = False) -> tuple[str, bool]:
        """One call's observation, the budget line after it where it is the reply's `last` call, and whether it ends
        the unit; `final`, the reply's last write, after which the harness runs (`implicit`)."""
        tail = self.budget() if last else ""
        if call.cut:
            return self.salvage(call, tail), bool(self.ending)
        if call.name == "lib":                      # never offered: answered in one line
            self.counts["lib_asked"] += 1
            self.log("lib", f"{call.describe()}: not offered, answered in one line")
            return fit(LIB_ASKED[bool(self.cards)], commands.MALFORMED_BYTES, tail), False
        if call.problem:
            return fit(self.malformed(call.problem), commands.MALFORMED_BYTES, tail), False
        args, ws = call.arguments, self.ws
        if call.name == "write":
            said = self.joined(args, tail) if call is self.stitching and self.joining else \
                self.written(args["path"], args["text"], tail, final=final)
            self.log("write", first_line(said)[:200])
            return said, (self.resent is not None and self.resent[0] == "ended") or bool(self.ending)
        if call.name == "run":
            limit, line = self.run_limit()
            if limit < MIN_RUN_S:
                return fit(f"not run: {max(self.left(), 0):.0f} s left, under a run's {MIN_RUN_S:.0f} s",
                           commands.BOUNDS["run"], tail), False
            said, ran = ws.run(args["path"], limit, self.seen, lambda done: self.tail(done, last))
            self.ran(ran, limit)
            self.saw(said, ran)
            return said, self.past_reserve(ran, limit, line)
        if call.name == "inspect":
            said = self.inspected(args["path"], args.get("rows"), tail)
            self.log("inspect", first_line(said)[:120])
            return said, False
        return self.finish(last, str(args.get("reason") or ""))

    def malformed(self, why: str) -> str:
        """A malformed call's answer (`MALFORMED`, `why` its problem), the commands it lists those a request offers:
        `inspect` and `lib` left out (`MALFORMED_FORMS`)."""
        said = MALFORMED.format(why=why)
        gone = ("inspect", "lib")
        said = said.replace(f"the tools are {', '.join(commands.NAMES)}",
                            "the tools are " + ", ".join(name for name in commands.NAMES if name not in gone), 1)
        for name in gone:
            said = said.replace(MALFORMED_FORMS[name], "", 1)
        return said

    def written(self, path: str, text: str, tail: str, final: bool = False) -> str:
        """A `write` hashed: a text the file already holds refused, naming the turn that wrote it and what its run
        gave; a return to an earlier version counted, and on the file's `CYCLE_RECURRENCES`-th, where its last
        `CYCLE_WINDOW` versions are at most three, the cycle named. The reply's last write (`final`) refused as
        unchanged from a turn whose run exited clean with no check failing, every named output present, is read as
        the finish (`resend_finish`); any other unchanged refusal kept for `after_write`'s line (`resent`)."""
        self.resent = None
        here, named = self.ws.local(path)
        if here is None or commands.shadows(named):
            return self.ws.write(path, text, tail)
        sha = sha256(commands.unfenced(text))
        history = self.versions.setdefault(named, [])
        if here.is_file() and sha256(here.read_text(encoding="utf-8", errors="replace")) == sha:
            self.counts["write_repeats"] += 1
            turn = history[-1][0] if history and history[-1][1] == sha else None
            if turn is not None:
                ran = self.resend_ran(turn)
                missing = tuple(self.ws.missing()) if ran is not None else ()
                if ran is not None and not ran.failures and not missing and final:
                    return self.resend_finish(named, turn)
                self.resent = ("refused", named, turn, missing)
            said = UNCHANGED.format(named=named, turn=turn, gave=self.gave.get(turn, "no run"),
                                    run=self.run_form(named)) if turn is not None else \
                HELD.format(named=named, run=self.run_form(named))
            return fit(said, commands.BOUNDS["write"], tail)
        earlier = next((turn for turn, seen in history if seen == sha), None)
        note = ""
        if earlier is not None:
            self.recurrences[named] += 1
            versions = {seen for _, seen in history[-(CYCLE_WINDOW - 1):]} | {sha}
            if self.recurrences[named] == CYCLE_RECURRENCES and len(versions) <= 3:
                note = CYCLE.format(named=named, k=len(versions), turn=earlier, gave=self.gave.get(earlier, "no run"))
                self.log("write", f"{named}: a cycle among {len(versions)} versions")
        said = self.ws.write(path, text, "\n".join(part for part in (note, tail) if part))
        if said.startswith("wrote "):
            history.append((self.counts["turns"], sha))
        return said

    def resend_ran(self, turn: int) -> commands.Ran | None:
        """Turn `turn`'s run, the record its `gave` line was built from, where it exited clean: exit 0 with no
        traceback (so not stopped at its limit, nor refused); else None."""
        ran = self.runs_at.get(turn)
        return ran if ran is not None and ran.exit == 0 and not ran.error else None

    def resend_finish(self, named: str, turn: int) -> str:
        """An unchanged write read as the finish (`written`): `finish` run as a written one is, its observation under
        `RESEND_READ`, the budget line after it; `resent` says whether it ended the unit."""
        self.counts["resend_finished"] += 1
        said, ended = self.finish(True, resent=True)
        self.resent = ("ended" if ended else "read", named, turn, ())
        return f"{RESEND_READ.format(named=named, turn=turn)}\n{said}"

    def run_form(self, named: str) -> str:
        """How the model asks a run of `named`."""
        return f"a line `run {named}`"

    def salvage(self, call: Call, tail: str, final: bool = False) -> str:
        """A call the output cap cut: a .py write's head kept and the rest asked (`hold`; `final`, the last call's
        reply, keeps it and asks nothing), but at the `SHORT_STREAK`-th cut reply in a row, which asks for one small
        file; there, and for any other file, its complete definitions kept as `work/<name>.draft` and named, else the
        call said lost."""
        arguments = self.recovered(call)
        text, path = arguments.get("text"), arguments.get("path") or f"{commands.WORK}/main.py"
        short = CUT_SHORT.format(n=self.cut_streak) if self.cut_streak >= SHORT_STREAK else ""
        here, named = self.ws.local(path)
        if call.name not in ("write", "cut", "unparsed") or not text or here is None:
            empty = CUT_FENCED if call.name == "cut" else CUT_CALL
            return asked(empty, short, commands.MALFORMED_BYTES, tail)
        if named.endswith(".py"):
            if not short:
                held = self.hold(named, text, tail, final)
                if held is not None:
                    return held
            else:
                self.drop_part(named)
                self.log("continue", f"{named}: {self.cut_streak} replies cut in a row: the split asked")
        if named.endswith(".py"):
            kept, names, where = draft(text)
        else:
            kept, names, where = "".join(text.splitlines(keepends=True)[:-1]), [], "at its last line"
        draft_path = f"{commands.WORK}/{Path(named).stem}.draft"
        if kept:
            self.ws.write(draft_path, kept)
            self.counts["drafts"] += 1
            self.log("draft", f"{named}: {len(kept.encode()):,} B kept as {draft_path}, {len(names)} definition(s); "
                              f"the cut {where}")
        said = CUT_DRAFT.format(named=named, kept=CUT_KEPT.format(
            draft=draft_path, defines=f" (it defines {', '.join(names[:12])})" if names else "", where=where)
            if kept else CUT_NONE)
        return asked(said, short, commands.BOUNDS["write"], tail)

    def hold(self, named: str, text: str, tail: str, final: bool = False) -> str | None:
        """A .py write the cap cut kept as the head of `named`, to the end of its last whole statement
        (`boundary_head`, so the quote ends one and the rest starts the next), saved as `named` + `PART` (never run),
        and the rest asked (`CUT_HELD`: the head's last lines quoted as written, no count; one block of code, no line
        already written), the next reply read as that rest (`resumed`). No draft is written. None where no line of it
        is complete (the split's words then); "" under `final`, which asks nothing."""
        body = opened_fence(text)
        head = boundary_head(body)
        if not head.strip():
            return None
        self.ws.write(f"{named}{PART}", head)
        self.log("continue", f"{named}: {len(head.encode()):,} B kept as {named}{PART} to its last complete line"
                             + ("" if final else "; the rest asked"))
        if final:
            return ""
        self.held = (named, head)
        self.counts["cut_continued"] += 1
        said = CUT_HELD.format(named=named, ends=self.redact(ends(head)), through=self.through(named))
        return fit(said, HELD_BYTES, tail)

    def through(self, named: str) -> str:
        """What the rest of `named`'s held head is asked through (`CUT_THROUGH`): work/main.py's run and its writes,
        another file's end."""
        return CUT_THROUGH[named == MAIN_FILE]

    def drop_part(self, named: str) -> None:
        """The head's file of `named` removed, where there is one: the stitched file, or the split's draft, stands in
        its place."""
        here, _ = self.ws.local(f"{named}{PART}")
        if here is not None:
            here.unlink(missing_ok=True)

    def resumed(self, reply, found: list[Call]) -> list[Call]:
        """The reply after a head was kept (`held`) read as the rest of its file. Its code (`rest_of`) is stitched
        under the head (`runner.stitch`; a reply that wrote the file again whole, `rewritten`, taken as it is) and
        given back as one write of the file, whole, in the reply's place before its other calls (a run, a finish), so
        it is saved, compiled and run as a write is; a stitched work/main.py whose top level calls nothing said at its
        run (`ENTRY`). A rest opening with the head's last lines, whitespace aside, has them once, as it has them
        (`runner.stitch`'s `overlap`); a join that does not compile, a restart aside, is said with its junction and
        the head held again (`joined`). A reply the cap cut again gives back a cut write of the head and its complete
        lines, which `salvage` holds again. A reply holding no code is read as it came, the head dropped."""
        named, head = self.held
        self.held, self.joining = None, None
        code, taken, untagged = self.rest_of(reply, found, named)
        if code is None:
            self.drop_part(named)
            self.log("continue", f"{named}: the reply held no code; the head dropped")
            return found
        read = (named, MAIN_FILE) if untagged else (named,)     # the blocks read as the rest, as the parser named them
        others = [call for call in found if call is not taken and call.name not in ("cut", "unparsed")
                  and not (taken is None and call.name == "write" and self.ws.local(str(call.arguments.get(
                      "path", "")))[1] in read)]
        cut = reply.finish_reason == "length"   # a block the reply left open below the cap is the rest, whole
        #                                         where the join compiles
        joined, how, rest = stitched(head, code, cut)
        text, whole = (joined + code[len(rest):], False) if cut else (joined, runner.compiles(joined))
        overlap = runner.repeated(head, rest) if how.startswith(runner.OVERLAP) else 0
        write = Call("write", {"path": named, "text": text}, None, cut)
        self.log("continue", f"{named}: {len(head.encode()):,} B kept, the rest {len(code.encode()):,} B"
                             + (", cut again" if cut else "") + f"; the join {len(joined.encode()):,} B ({how})"
                             + ("" if cut else ", compiles" if whole else ", does not compile"))
        if not cut:
            self.drop_part(named)
            self.stitching = write
            if not whole and how != "restarted":        # said with its junction, the head held again
                start = runner.unrepeated(head, overlap).count("\n") + 1 if overlap else \
                    head.rstrip("\n").count("\n") + 2         # the rest's first line in the join
                error, line = compile_error(joined)
                shown = {"lines": junction(joined, start, line), "opened": never_closed(joined, start)}
                self.joining = (named, head, JUNCTION.format(**shown, named=named, through=self.through(named)),
                                not runner.compiles(head) and not unfinished(head),     # its error the head's own,
                                error, line)                                            # above it
            if whole and named == MAIN_FILE and not runner.has_driver(joined):
                name = runner.entry(joined) or "main"
                self.entry = ENTRY.format(name=name)
                self.log("entry", f"the stitched script calls none of what it defines (`{name}`)")
        return [write, *others]

    def joined(self, args: dict, tail: str) -> str:
        """The write of a join that does not compile (`joining`), said as a write is with its junction after the
        error's lines (`JUNCTION`), the head held again as `named` + `PART` so the next reply is read as its rest
        (`resumed`). The join refused as the file already saved is said so with its error, never a run offered, and
        its junction, the head held again (`JOIN_REFUSED`); where the error stands in a head broken above its end, the
        file asked whole and the head dropped (`KEPT_BROKEN`), once a file. A write refused otherwise, as it came."""
        named, head, said_after, broken, error, line = self.joining
        self.joining = None
        repeats = self.counts["write_repeats"]
        said = self.written(args["path"], args["text"], "")
        refused = self.counts["write_repeats"] > repeats
        if not refused and not (said.startswith("wrote ") and "does not compile" in first_line(said)):
            return f"{said}\n{tail}" if tail else said
        self.resent = None              # a refused join's run line is the plain one
        if refused:
            said = fit(JOIN_REFUSED.format(named=named, error=error) + (
                f"\n{commands.around(commands.unfenced(args['text']), line)}" if line else ""), commands.BOUNDS["write"])
        if broken and named not in self.said_broken:
            self.said_broken.add(named)
            self.log("continue", f"{named}: the join does not compile, its error in what was kept; the file asked "
                                 "whole, the head dropped")
            said = f"{said}\n{KEPT_BROKEN.format(named=named)}"
        else:
            self.ws.write(f"{named}{PART}", head)
            self.held = (named, head)
            self.log("continue", f"{named}: the join does not compile" + (", refused as the file saved" if refused
                                                                          else "")
                     + "; its junction said, the head held again")
            said = f"{said}\n{fit(self.redact(said_after), HELD_BYTES)}"
        return f"{said}\n{tail}" if tail else said

    def rest_of(self, reply, found: list[Call], named: str) -> tuple[str | None, Call | None, bool]:
        """The code a reply gives as `named`'s rest, the call that carried it, and whether a block naming no file was
        read: the text of its first write (or cut call) of `named`, or naming no path; else its ```python blocks' code
        that name no other file, in order; else, where it holds no block, the code it opens with
        (`continuation_code`). (None, None, False) where it holds none."""
        for call in found:
            if call.name not in ("write", "cut", "unparsed") or (call.problem and not call.cut):
                continue
            arguments = self.recovered(call)
            path, text = arguments.get("path"), arguments.get("text")
            if text and (not path or self.ws.local(path)[1] == named):
                return opened_fence(text, closed=not call.cut), call, False
        items = commands.fenced(runner.THINK.sub("", reply.text or ""))
        blocks = [item for item in items if isinstance(item, commands.Fenced) and not item.void
                  and (item.path is None and (item.tag in commands.PYTHON or not item.tag)
                       or item.path is not None and self.ws.local(item.path)[1] == named)]
        if blocks:
            return ("\n".join("\n".join(block.lines) for block in blocks), None,
                    any(block.path is None for block in blocks))
        if not any(isinstance(item, commands.Fenced) for item in items):
            code = continuation_code(reply.text or "")
            if code.strip():
                return code, None, False
        return None, None, False

    def recovered(self, call: Call) -> dict:
        """A cut call's arguments as far as they got: its own where they hold a text, else the reply's text's
        (`cut_from_text`)."""
        return call.arguments if isinstance(call.arguments, dict) and call.arguments.get("text") else \
            cut_from_text(self.reply_text)

    def implicit(self, found: list[Call]) -> int | None:
        """The index of the reply's last call after which the harness may run a file (`after_write`): a `write` of a
        `.py` file under work/ that no `run` or `finish` follows, a text file's write after it, which never runs, set
        aside (`text_write`); None where no run follows the reply's calls."""
        last = next((k for k in range(len(found) - 1, -1, -1) if found[k].name in ("write", "run", "finish")
                     and not found[k].problem and not found[k].cut and not text_write(found[k])), None)
        if last is None or found[last].name != "write" or not str(found[last].arguments.get("path", "")).endswith(".py"):
            return None
        return last

    def finish_writes(self, found: list[Call]) -> dict[int, int]:
        """Each `finish` of the reply that follows a `write` of a `.py` file with no `run` or earlier `finish` between
        (a text file's write between set aside, as `implicit` sets it), by its index, mapped to that write's: the file
        the harness runs before the finish (`after_write`'s `finishing`)."""
        pairs, last = {}, None
        for number, call in enumerate(found):
            if call.problem or call.cut or call.name not in ("write", "run", "finish") or text_write(call):
                continue
            if call.name == "finish" and last is not None and found[last].name == "write" \
                    and str(found[last].arguments.get("path", "")).endswith(".py"):
                pairs[number] = last
            last = number
        return pairs

    def after_write(self, said: str, path: str, finishing: bool = False) -> str:
        """A reply's last write's observation with the run the harness makes after it, the budget line at the end:
        no run where the write was refused or does not compile, nor while work/main.py is missing (the imports still
        missing named); work/main.py where it changed since its last run (else said unchanged, not run); a file
        written other than main.py with an `if __name__ == "__main__":` block that main.py does not import, itself,
        said so (one main.py imports is a module, and main.py runs). After a module's write, nothing runs while
        work/main.py imports a module that is nowhere (named; work/'s own modules imported package-style said as
        such); and work/main.py runs where it imports the module written (directly or through a module under work/)
        and its last run was not clean, the observation naming the module, and says so where main.py does not import
        it. The run's limit is `run_limit`'s. `finishing` (a write a `finish` follows, `finish_writes`): the run's
        observation without the budget line, which the reply's last call carries; where no run is made, the write's
        observation as it stood."""
        main, mainpath = self.ws.dir / "main.py", f"{commands.WORK}/main.py"
        here, named = self.ws.local(path)

        def unrun(line: str) -> str:
            return said if finishing else f"{said}\n{line}\n{self.budget()}"
        if self.resent is not None and self.resent[0] == "read":     # the finish's observation, its budget line
            return said                                              # after it
        if said.startswith(("write refused", "not written")) or "does not compile" in first_line(said):
            return unrun(self.unrun_line())
        if not main.is_file():
            missing, note = self.ws.unwritten(self.ws.missing_modules())
            return unrun(f"(no run: {mainpath} does not exist yet" + (
                f"; the files import these modules not yet under {commands.WORK}/: "
                + ", ".join(f"{name}.py" for name in missing[:12]) + "; write them, then work/main.py, which runs"
                if missing else "; write it next, and it runs") + (f"; {note}" if note else "") + ")")
        script = here.read_text(encoding="utf-8", errors="replace") if here is not None and here.is_file() else ""
        top = named != mainpath and named.count("/") == 1          # a module main.py can import by its name
        local, missing = self.ws.reached() if named != mainpath else (set(), [])
        imports = top and Path(named).stem in local
        alone = named != mainpath and commands.has_main_block(script) and not imports
        target = named if alone else mainpath
        module = target == mainpath and named != mainpath
        why = SELF_RUN.format(named=named, main=mainpath) if alone and top else ""
        if module:
            missing, note = self.ws.unwritten(missing, mainpath)
            if missing or note:
                return unrun("(no run: " + "; ".join(part for part in (
                    f"{mainpath} imports these modules not yet under {commands.WORK}/: "
                    + ", ".join(f"{name}.py" for name in missing[:12]) + f"; write them, and {mainpath} runs"
                    if missing else "", note) if part) + ")")
            last = self.last_ran.get(mainpath)
            if imports and last is not None and not last[2]:
                why = f" ({mainpath} imports {named}, and its last run, turn {last[0]}, was not clean)"
        if target == mainpath and not why:
            last = self.last_ran.get(mainpath)
            if last is not None and last[1] == sha256(main.read_text(encoding="utf-8", errors="replace")):
                self.counts["unchanged_main"] += not finishing
                hint = NOT_IMPORTED.format(named=named, main=mainpath) if top and not imports else \
                    f"; {self.run_form(mainpath)} runs it with {named}" if named != mainpath else ""
                return unrun(f"({mainpath} unchanged since turn {last[0]}: not run{hint})")
            if module:
                why = f" ({mainpath} changed since its last run; " + (
                    f"it does not import {named})" if top and not imports else f"{named} written)")
        limit, line = self.run_limit()
        if limit < MIN_RUN_S:
            return unrun(f"(no run of {target}: {max(self.left(), 0):.0f} s left)")
        self.counts["implicit_runs"] += 1
        self.counts["module_runs"] += module
        if finishing:
            self.log("finishrun", f"{target} run before the finish that follows {named}")
        shown, ran = self.ws.run(target, limit, self.seen, lambda done: self.tail(done, not finishing))
        self.ran(ran, limit)
        self.saw(shown, ran)
        self.past_reserve(ran, limit, line)
        return f"{said}\nthen, run for you{why}: {shown}"

    def unrun_line(self) -> str:
        """The run line after a write not saved whole and compiling (`after_write`); an unchanged one's said true:
        the file unchanged since the turn it matched (`RESEND_SINCE`), and after a clean run the named outputs still
        missing (`RESEND_MISSING`)."""
        if self.resent is None or self.resent[0] != "refused":
            return NO_RUN
        _, named, turn, missing = self.resent
        if missing:
            return RESEND_MISSING.format(named=named, turn=turn, missing=", ".join(missing[:12]),
                                         run=self.run_form(named))
        return RESEND_SINCE.format(named=named, turn=turn)

    def ran(self, ran: commands.Ran | None, cap: float) -> None:
        """A run counted and logged: its turn's line (`gave`), its file's sha at the run (`last_ran`), and the output
        of a clean one with no finding kept (`keep_clean`)."""
        turn = self.counts["turns"]
        if ran is None:
            self.log("run", "refused")
            self.gave[turn] = "the run was refused"
            self.runs_at[turn] = None
            return
        self.counts["runs"] += 1
        seen_before = ran.identity is not None and ran.identity in self.seen
        if ran.identity is not None:
            self.seen.add(ran.identity)
        here = self.ws.local(ran.path)[0]
        if here is not None and here.is_file():
            self.last_ran[ran.path] = (turn, sha256(here.read_text(encoding="utf-8", errors="replace")),
                                       ran.clean and not ran.findings)
        if ran.exit == 0 and ran.clean and not ran.findings:
            self.keep_clean()
        self.gave[turn] = ("stopped at its limit" if ran.exit is None else ran.error.split("\n", 1)[0][:160]
                           if ran.error else f"exit {ran.exit}, {len(ran.failures)} check(s) failing" if ran.failures
                           else f"exit 0, {len(ran.findings)} sanity finding(s)" if ran.findings
                           else f"exit {ran.exit}, the checks pass")
        self.runs_at[turn] = ran
        self.outcome[ran.path] = (turn, self.gave[turn])
        self.log("run", ("stopped at its limit" if ran.exit is None else f"exit {ran.exit}")
                 + f" after {ran.seconds:.1f} s of {cap:.0f}; {len(ran.failures)} check(s) failing"
                 + (f"; {len(ran.findings)} sanity finding(s)" if ran.findings else "")
                 + (f"; {ran.error.split(':', 1)[0][:60]}" if ran.error else "")
                 + ("; the same error as an earlier run" if seen_before else ""))

    def finish(self, last: bool, reason: str = "", resent: bool = False) -> tuple[str, bool]:
        """`finish`, by `counted`'s rule, but a finish whose last shown run did not exit 0 (`shown_run`, the run of a
        write before the finish among them) is turned away where the current output fails the checks (`turns_away`):
        answered with that run's observation under `FINISHFAIL_SAID` and not counted among the finishes, while a
        request and `ASK_S` remain, at most `FINISHFAIL_MAX` times a unit; the next finish on a failing run counts. The
        finish an unchanged re-send is read as (`resent`) is never turned away."""
        shown = self.shown_run if not resent else None
        if shown is None or not self.ws.checks():
            return self.counted(last, reason)
        turn, named, exit_, said = shown
        if turns_away(exit_, self.requests_left(), self.left(), self.counts["finishfail_turned"]):
            self.counts["finishfail_turned"] += 1
            self.log("finish", f"not taken: the last run, {named} at turn {turn}, failed; turned away "
                               f"{self.counts['finishfail_turned']} of {FINISHFAIL_MAX}")
            return fit(f"{FINISHFAIL_SAID.format(named=named)}\n{said}", commands.BOUNDS["run"],
                       self.budget() if last else ""), False
        return self.counted(last, reason)

    def saw(self, said: str, ran: commands.Ran | None) -> None:
        """A run the model is shown kept with its observation, the budget line `tail` ended it with dropped, for a
        finish on it (`finish`); a run refused keeps the one before."""
        if ran is not None:
            budget = f"\n{self.tail_budget}" if self.tail_budget else ""
            self.shown_run = (self.counts["turns"], ran.path, ran.exit, said.removesuffix(budget) if budget else said)

    def counted(self, last: bool, reason: str = "") -> tuple[str, bool]:
        """A `finish` judged by the checks and the findings over the output as it stands. The checks failing, the first
        finish with a request left shows them (and any finding) and the loop goes on, a second ends the unit. The
        checks passing with a finding standing, one finish is refused (fix it, or finish again with `reason`,
        "accept: <why>"), once a unit; the next ends it, its findings accepted; with no finding the first ends it. A
        refusal is made only with a request and `ASK_S` left."""
        self.finishes += 1
        failures = self.ws.checks()
        _, findings = self.ws.sanity()
        tail = self.budget() if last else ""
        unread = commands.unread_finding(self.ws.unread())     # the parameters the scripts never read
        findings = [*findings, self.redact(unread)] if unread else findings
        able = self.requests_left() >= 1 and self.left() >= ASK_S
        listed = "\n".join(f"- {line}" for line in findings[:sanity.FINDINGS])
        if not failures:
            if findings and not self.sanity_refused and able:
                self.sanity_refused, self.counts["finish_refused"] = True, 1
                self.log("finish", f"refused: {len(findings)} sanity finding(s) stand")
                return fit(FINISH_REFUSED.format(n=len(findings), lines=listed), commands.BOUNDS["finish"],
                           tail), False
            return self.ended(findings, reason)
        final = self.finishes > 1 or not able
        self.finish_said = "finish: the checks fail"
        self.log("finish", f"{len(failures)} check(s) failing" + ("; the task ends" if final else ""))
        if final:
            return fit(f"finish: {len(failures)} check(s) fail; the task ends with the output as it stands.",
                       commands.BOUNDS["finish"]), True
        if findings:                            # shown with the failures: the one review turn they get
            self.sanity_refused, self.counts["finish_refused"] = True, 1
        return fit(FINISH_FAILED.format(n=len(failures), lines="\n".join(f"- {line}" for line in failures[:10]))
                   + (FINISH_FINDINGS.format(lines=listed) if findings else ""), commands.BOUNDS["finish"],
                   tail), False

    def ended(self, findings: list[str], reason: str) -> tuple[str, bool]:
        """A finish with the checks passing that ends the unit: clean, or its findings accepted."""
        self.counts["finish_clean"] = 1
        if not findings:
            self.log("finish", "the checks pass")
            self.finish_said = "finish: the checks pass"
            return fit("finish: the checks pass; the task ends.", commands.BOUNDS["finish"]), True
        self.finish_said = f"finish: the checks pass; {len(findings)} sanity finding(s) accepted"
        self.log("finish", f"the checks pass; {len(findings)} sanity finding(s) stand, accepted"
                           + (f" with a reason of {len(reason):,} characters" if reason.strip() else ""))
        return fit(f"finish: the checks pass; {len(findings)} sanity finding(s) accepted; the task ends.",
                   commands.BOUNDS["finish"]), True

    # ------------------------------------------------------------------------------------- the clean output

    def keep_clean(self) -> None:
        """The output as it stands copied beside work/ (`KEPT`), where it changed since the copy kept and holds at
        most `KEPT_BYTES`; where it holds more, no copy is kept (an older output is never put back over a later
        clean one). Never raises: a copy that fails keeps none."""
        try:
            now = contract.snapshot(self.out)
            if self.kept is not None and now == self.kept_snapshot:
                return
            kept, self.kept, self.kept_snapshot = self.work / KEPT, None, None
            shutil.rmtree(kept, ignore_errors=True)
            if sum(path.stat().st_size for path in self.out.rglob("*") if path.is_file()) > KEPT_BYTES:
                self.log("rules", "the clean output is past the copy's bound: none kept")
                return
            shutil.copytree(self.out, kept, symlinks=True)
            self.kept, self.kept_snapshot = kept, now
        except Exception as exc:  # noqa: BLE001 - the copy never costs the run
            shutil.rmtree(self.work / KEPT, ignore_errors=True)
            self.log("rules", f"the clean output not kept ({type(exc).__name__})")

    def put_back(self, ended: str = "") -> None:
        """The caller holding guard.WRITING: where the unit ended with no request left (the cap, the clock, the last
        call; `ended` the watchdog's close) and the output as it stands breaks a `PUTBACK_RULES` property that the kept
        clean output holds (every one it breaks), that output is put back whole. Never raises, so the close that
        follows always runs."""
        if self.kept is None or (self.end_kind not in ("requests", "clock") and not ended):
            return
        try:
            self.ws.sanity()
            broken = [each for each in self.ws.ruled if each.rule in PUTBACK_RULES and each.prop is not None]
            if not broken:
                return
            if not all(props.holds(each.prop, self.kept) for each in broken):
                self.log("rules", "not put back: the kept clean output does not hold "
                         + ", ".join(dict.fromkeys(each.rule for each in broken)))
                return
            clear(self.out)
            for entry in sorted(self.kept.iterdir()):
                (shutil.copytree if entry.is_dir() and not entry.is_symlink() else shutil.copy2)(
                    str(entry), str(self.out / entry.name))
            self.counts["putback"] = 1
            self.log("rules", "the last clean output put back: the output at the end broke "
                     + ", ".join(dict.fromkeys(each.rule for each in broken))
                     + f", which it held ({ended or self.end_kind})")
        except Exception as exc:  # noqa: BLE001 - the close follows whatever failed here
            self.log("rules", f"not put back ({type(exc).__name__})")

    # ------------------------------------------------------------------------------------------------ the draft

    def drafts_first(self, task, pathmap, data: str) -> None:
        """The draft's round before the loop's first request (agent/n11/drafts.py), where the loop keeps
        `drafts.FLOOR_REQUESTS` beyond the round's allowance. A failure in the round's own code is logged, what it left
        in the output removed, and the loop starts without it."""
        allowed = drafts.ALLOWANCE
        if self.requests_left() < allowed + drafts.FLOOR_REQUESTS:
            self.log("drafts not sent", f"{self.requests_left()} request(s) left, under the round's {allowed} and the "
                                        f"loop's {drafts.FLOOR_REQUESTS}")
            return
        self.drafting = drafts.Round(self, task, pathmap, data)
        try:
            self.drafting.go()
        except Exception as exc:  # noqa: BLE001 - the round's failure costs its draft, never the unit
            self.drafting.given_up(f"{type(exc).__name__}; the loop without it")

    def drafts_close(self) -> None:
        """At the unit's close, the caller holding guard.WRITING: the round's held floor written where the output holds
        nothing to grade (`drafts.Round.close`); nothing where no round ran."""
        if self.drafting is not None:
            self.drafting.close()

    # ------------------------------------------------------------------------------------------ the messages

    def system(self, index: str) -> str:
        """The loop's system prompt before `worded`'s edits: `SYSTEM` in the fenced form, the index in it, the cards
        after it, the library rule without its lookup clause (`LIBRARY_USE_NEVER`)."""
        system = SYSTEM.format(tools=FENCED_PART, libraries=LIBRARIES, run=RUN_S, example=FENCED_EXAMPLE, index=index,
                               api=api_card(), conventions="\n".join(CONVENTION_LINES), settle=SETTLE,
                               library=LIBRARY_USE_NEVER)
        return system + (f"\n\n{self.cards}" if self.cards else "")

    def control(self, index: str) -> prompts.Control | None:
        """The system prompt (`system`) in its paragraphs (`prompts.Control`), which `prompts.clean_parts` edits."""
        return prompts.Control.read(self.system(index), self.cards)

    def worded(self, index: str) -> str:
        """The loop's system prompt as sent: `system`'s with the edits of `prompts.clean_parts`, in
        `prompts.CLEAN_ORDER`, an edit whose anchor is not met logged."""
        said, missed = prompts.clean_parts(self.control(index), TIMING_LINES, FENCED_EXAMPLE)
        if missed:
            self.log("prompt", f"clean: {', '.join(missed)} left as written (the anchor not met)")
        return prompts.clean_text(said)

    def selected(self, task, lib: library.Library, index: str) -> str:
        """The selection call before the loop's first message (agent/n11/selection.py), its index said without the
        lookup (`library_unsaid`): where it picked, its shelves' section, which takes the index's place; else
        `index`. The cards it shapes are made whole after it (`made_whole`)."""
        shaped = self.selection.call(self, task, lib, functions_named, ASK_S, shown=library_unsaid)
        return shaped.index if shaped is not None else index

    def made_whole(self, lib: library.Library, index: str) -> str:
        """The system prompt's cards: the selection call's (its picks' and the pointer's, `selection.Shaped.order`) or,
        where it shaped nothing, the pointer's, each whole after the index within `library.WHOLE_TOKENS` in all
        (`library.made_whole`): where they do not fit, the index trimmed for them (the call's section by
        `selection.trims`, else `Library.trims`, the shelves no carded function lies on reduced first), and only where
        that leaves too little, the cards cut; then the lookup taken out of the index's head and a cut card's last
        line (`library_unsaid`, `cards_unsaid`). The index returned."""
        shaped = self.selection.result
        carded = list(shaped.order) if shaped is not None else [function for function, _ in self.fired]
        head = selection.CARDS_HEAD if shaped is not None else library.CARDS_HEAD
        trims = selection.trims(index) if shaped is not None else \
            lib.trims(index, {function.shelf for function in carded})
        made = library.made_whole(carded, index, trims, head=head)
        self.cards = cards_unsaid(made.cards)
        self.log("cards", f"{len(carded)} card(s), {len(made.cuts)} cut past the bound")
        for cut in made.cuts:
            self.log("cards", f"past the bound: {cut}")
        return library_unsaid(made.index)

    def described_parts(self) -> dict[str, str]:
        """The input files the loop's first message describes, each with its description as carried there
        (`commands.carried`), read once the description is built; {} before."""
        if self.parts is None and self.inputs_text is not None and self.unit is not None:
            self.parts = commands.carried(self.inputs_text, self.unit.data_dir)
        return self.parts or {}

    def inspected(self, path: str, rows, tail: str) -> str:
        """`inspect`'s observation: an input file the loop's first message describes answered from that description
        (`commands.Workspace.answered`), any other as `inspect` shows it."""
        local = self.ws.answered(path, self.described_parts(), rows, tail)
        return local if local is not None else self.ws.inspect(path, rows, tail)

    def described(self) -> str:
        """The input files described (`commands.Workspace.described`), built once, a file past the description's bound
        said so without naming `inspect`."""
        if self.inputs_text is None:
            self.inputs_text = self.ws.described().replace("- not described here (`inspect` shows them): ",
                                                           "- not described here (past the description's bound): ", 1)
        return self.inputs_text

    def first_turn(self, task, pathmap, data: str, named: tuple[str, ...], pointer: str, index: str) -> str:
        """The loop's first message: the task's sections (`task_text`, as `cleaned` says them), the budget, the
        conventions' sentence where step 1 does not say the lags (`prompts.clean_parts`), and the pointer's line
        without its call imperative (`prompts.CLEAN_POINTER`)."""
        control = self.control(index)
        said = task_text(task, pathmap, data, self.described(), named, self.cleaned)
        lags = "lags" not in prompts.clean_parts(control, TIMING_LINES, FENCED_EXAMPLE)[1]
        if pointer and pointer.count(prompts.CLEAN_POINTER[0]) == 1:
            pointer = pointer.replace(*prompts.CLEAN_POINTER, 1)
        elif pointer:
            self.log("prompt", "clean: pointer left as written (the anchor not met)")
        return (said + f"# Budget\n\n{self.budget()}\n" + ("" if lags else f"\n{CONVENTIONS}\n")
                + (f"\n{pointer}\n" if pointer else ""))

    def cleaned(self, sections: dict[str, str]) -> dict[str, str]:
        """The first message's sections as sent: Paths without its "nothing anywhere else but /tmp" tail
        (`prompts.CLEAN_PATHS`; the system prompt's must list allows work/ beside it), and the input files described
        without a file's portrait that repeats the tree its input data shows (`prompts.repeated_portraits`); an edit
        not made logged."""
        said = dict(sections)
        if said["Paths"].count(prompts.CLEAN_PATHS[0]) == 1:
            said["Paths"] = said["Paths"].replace(*prompts.CLEAN_PATHS, 1)
        else:
            self.log("prompt", "clean: paths left as written (the anchor not met)")
        described = said["Input files, described"]
        if described and self.unit is not None:
            said["Input files, described"], dropped = prompts.repeated_portraits(
                described, said["Input data"], commands.portraits(described, self.unit.data_dir))
            if dropped:
                self.log("prompt", f"clean: the described portrait of {', '.join(dropped)} left out (the input data "
                                   "shows its tree)")
        return said

    def outgoing(self) -> list[dict]:
        """The messages as sent: observations older than the last `KEEP_TURNS` turns cut to their first line (the
        first user turn whole)."""
        turns = [k for k, message in enumerate(self.messages) if message["role"] == "assistant"]
        if len(turns) <= KEEP_TURNS:
            return list(self.messages)
        edge = turns[-KEEP_TURNS]
        return [{**message, "content": first_line(message["content"])} if 1 < k < edge and message["role"] == "user"
                else message for k, message in enumerate(self.messages)]

    def record(self, text: str, calls: list[Call], results: list[str]) -> None:
        """A turn's reply and its observations into the conversation: the reply, then one user turn holding each
        call's observation under the call."""
        self.messages.append({"role": "assistant", "content": text or "(no text)"})
        self.messages.append({"role": "user", "content": "\n\n".join(
            f"[{call.describe()}]\n{result}" for call, result in zip(calls, results)) or "(nothing ran)"})

    # ------------------------------------------------------------------------------------------- the request

    def ask(self) -> object:
        """The loop's request: the conversation as `said` sends it."""
        return self.client.converse(self.said())

    def said(self) -> list[dict]:
        """The messages a loop request sends (`outgoing`), its system prompt as `told` says it."""
        messages = self.outgoing()
        system = told(messages[0]["content"])
        if system != messages[0]["content"]:
            messages[0] = {**messages[0], "content": system}
        return messages

    def calls(self, reply) -> list[Call]:
        """A reply's calls, read from its fenced blocks and command lines leniently (`commands.from_fenced`): a write
        the cap cut takes its text from the block left open, and in a reply the cap cut with every call whole the
        last is taken as cut."""
        found = commands.from_fenced(reply.text)
        cut = reply.finish_reason == "length"
        blocks = [item for item in commands.fenced(runner.THINK.sub("", reply.text or ""))
                  if isinstance(item, commands.Fenced) and not item.closed]
        for call in found:
            if call.name == "write" and call.cut:
                block = next((item for item in blocks if self.ws.local(item.path or MAIN_FILE)[1] ==
                              self.ws.local(call.arguments["path"])[1]), None)
                if block is not None:
                    call.arguments["text"] = "\n".join(block.lines)
        if found and cut and not any(call.cut for call in found):
            found[-1].cut, found[-1].problem = True, None
        if not found and cut:
            found = [Call("cut", {}, None, True)]
        return found

    def loop(self) -> str:
        """The loop's turns until a stop: no request or `ASK_S` left; a finish that ends the unit; a run past its
        reserve (`past_reserve`); two replies in a row with nothing runnable; the `NO_CALL_CAP`-th ask for a call in a
        row met by prose (`ask_call`); and the last call (`LAST_CALL`), whose reply's work/main.py runs with all the
        time left and ends the unit."""
        streak = 0
        while True:
            if self.requests_left() < 1:
                return self.stop("no request left", "requests")
            if self.left() < ASK_S:
                return self.stop(f"{max(self.left(), 0):.0f} s left, under a request's {ASK_S:.0f} s", "clock")
            last = self.last_call_due()
            if last:
                self.call_last()
            try:
                reply = self.ask()
            except model.ModelError as exc:
                self.log("request", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
                return self.stop(f"the model did not answer ({type(exc).__name__})")
            self.counts["turns"] += 1
            turn = self.counts["turns"]
            self.reply_text = reply.text or ""
            found = self.calls(reply)
            if self.held is not None:   # the rest of the file whose head was kept
                found = self.resumed(reply, found)
            self.log("request", f"turn {turn}: {reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} "
                                f"completion tokens, finish {reply.finish_reason}; "
                                + (", ".join(call.describe() for call in found) or "no call"))
            if last:
                return self.last_turn(found, turn)
            if not found or self.lost(found):   # every reply with no call answered by an ask for one
                ended = self.ask_call(reply, found, turn)
                if ended:
                    return ended
                streak = 0
                continue
            self.no_call_streak = 0
            cut = any(call.cut for call in found)
            self.counts["cut_replies"] += cut
            self.cut_streak = self.cut_streak + 1 if cut else 0
            bad = [call for call in found if call.problem and not call.cut]
            self.counts["calls"] += len(found)
            then = self.implicit(found)
            before = self.finish_writes(found)  # the write each finish follows, run before it
            results, ended = [], False
            for number, call in enumerate(found):
                if number in before and not ended:
                    results[before[number]] = self.after_write(
                        results[before[number]], str(found[before[number]].arguments.get("path", "")), finishing=True)
                    ended = bool(self.ending)
                if ended:
                    results.append("not run: the task ended at finish" if not self.ending else
                                   "not run: the task ended on the last run's output")
                    continue
                said, ended = self.execute(call, number == len(found) - 1 and then is None, number == then)
                results.append(said)
            if then is not None and not ended:
                results[then] = self.after_write(results[then], str(found[then].arguments.get("path", "")))
                ended = bool(self.ending)
            if self.stitching is not None:     # the stitched write said as such
                results = [STITCHED.format(named=call.arguments["path"]) + said if call is self.stitching else said
                           for call, said in zip(found, results)]
                self.stitching = None
            if ended:
                return self.stop(self.ending or self.finish_said, "clock" if self.ending else "finish")
            self.record(reply.text, found, results)
            runnable = [call for call in found if not call.problem and not call.cut]
            streak = streak + 1 if bad and not runnable else 0
            if streak > 1:
                return self.stop("two replies in a row with no call that could be run", "no_call")

    # ---------------------------------------------------------------------------------------- a reply with no call

    def lost(self, found: list[Call]) -> bool:
        """A reply whose one call the cap cut with nothing of it left to keep (`salvage` would say `CUT_CALL`)."""
        if len(found) != 1 or not found[0].cut:
            return False
        return found[0].name not in ("write", "cut", "unparsed") or not self.recovered(found[0]).get("text")

    def prose_file(self, text: str) -> str:
        """The file a reply's prose names (`prose_files`) for the ask: the first not yet under work/, else the last
        named; "" where it names none."""
        named = prose_files(text)
        fresh = [each for each in named if not (self.ws.dir / each.split("/", 1)[1]).is_file()]
        return fresh[0] if fresh else named[-1] if named else ""

    def ask_call(self, reply, found: list[Call], turn: int) -> str:
        """A reply with no call, or one whose cut call left nothing (`lost`), answered every time by one short ask for
        the call (`NOCALL_ASK`): the file its prose named, else after a cut the cut said (`NOCALL_CUT`, the
        `SHORT_STREAK`-th in a row asking one small file) and the first file of the split, else the next file or
        finish. The backstop: the `NO_CALL_CAP`-th ask in a row met by prose ends the unit (a cut call is a call
        tried: it neither counts nor ends the streak). The stop's reason, "" where the loop goes on."""
        cut = bool(found)
        if cut:
            self.counts["cut_replies"] += 1
            self.cut_streak += 1
            self.no_call_streak = 0
        else:
            self.counts["no_call"] += 1
            self.no_call_streak += 1
        self.gave[turn] = "no call"
        if self.no_call_streak > NO_CALL_CAP:
            return self.stop(f"{self.no_call_streak} replies in a row with no call, {self.no_call_streak - 1} of "
                             "them after an ask for one: the unit ends with the output as it stands", "no_call")
        named = "" if cut else self.prose_file(reply.text)
        call = NOCALL_CALL["cut" if cut else "named" if named else "next"].format(named=named)
        short = CUT_SHORT.format(n=self.cut_streak) if self.cut_streak >= SHORT_STREAK else ""
        said = (NOCALL_CUT + short if cut else NOCALL_ASK) + " " + NOCALL_NOW.format(call=call)
        self.log("request", "the call asked" + (f": {named}" if named else ", the cut said" if cut else ""))
        self.messages.append({"role": "assistant", "content": reply.text or "(no text)"})
        self.messages.append({"role": "user", "content": f"{said}\n{self.budget()}"})
        return ""

    def stop(self, reason: str, kind: str = "") -> str:
        """The unit's end logged; `kind`, what stopped the loop (`put_back` reads `requests` and `clock`)."""
        self.end_kind = kind
        self.log("stop", reason)
        return reason

    def release(self) -> None:
        """At the watchdog's close, the caller holding guard.WRITING: the draft's round (still open, the best clean
        draft so far delivered; done, its held floor as at the close), then the clean output's put-back."""
        if self.drafting is not None:
            self.drafting.finish()
        self.put_back("the watchdog's close")

    # ------------------------------------------------------------------------------------------------- solve

    def solve(self) -> None:
        task = self.unit = unit.read(self.task_dir)
        self.log("unit", f"{task.dropped} canary line(s) removed; {self.deadline - time.time():.0f} s to the deadline")
        pathmap = paths.build(task.task_dir, task.instruction, task.dockerfile, self.out)
        data = digest.digest(task.data_dir, canaries=task.canaries)
        files = [path.name for path in task.data_dir.rglob("*") if path.is_file()] if task.data_dir.is_dir() else []
        named = contract.named_all(pathmap, task.instruction, files)
        self.log("digest", f"{len(data.encode()):,} bytes; {len(pathmap.entries)} path(s) named; {len(named)} output "
                           f"file(s) named")
        lib, index = library.load(HERE)
        self.fired = lib.pointers(task.instruction)
        pointer = pointer_line(self.fired)
        self.log("library", f"{len(lib.functions)} function(s) on {len(lib.shelves)} shelves, an index of "
                            f"{len(index.encode()):,} B; pointer: "
                            + (", ".join(function.call for function, _ in self.fired) or "none"))
        stated = props.stated(task.instruction, named)
        self.log("stated", f"{len(stated.props)} propert(ies) of the outputs; {len(stated.keys)} JSON file(s) with "
                           f"{sum(map(len, stated.keys.values()))} key(s), {len(stated.columns)} table(s) with "
                           f"{sum(map(len, stated.columns.values()))} column(s) named")
        self.named_files = tuple(named)
        self.ws = commands.Workspace(self.work, self.out, task, pathmap, named, stated=stated)
        index = self.made_whole(lib, self.selected(task, lib, index))
        self.drafts_first(task, pathmap, data)
        first = self.first_turn(task, pathmap, data, named, pointer, index)
        self.messages = [{"role": "system", "content": self.worded(index)},
                         {"role": "user", "content": self.redact(first)}]
        self.loop()
        with guard.WRITING:     # an earlier clean output put back at an end with no request left
            self.put_back()

    def summary(self) -> str:
        """The unit's counts, one line."""
        return " ".join(f"{key}={value}" for key, value in sorted(self.counts.items()))

    def summarise(self) -> None:
        if not self.summarised:
            self.summarised = True
            try:
                self.log("summary", self.summary())
            except Exception as exc:  # noqa: BLE001 - the summary never costs the close
                self.log("summary", f"unavailable ({type(exc).__name__})")


def solve(task_dir: Path, out: Path) -> int:
    """One unit, start to close; 0 whatever happens."""
    out = Path(out)
    client = None
    steps = guard.Steps(AGENT, lambda: client.usage.requests if client is not None else 0)
    canaries: frozenset[str] = frozenset()
    watchdog = agent = work = None
    try:
        out.mkdir(parents=True, exist_ok=True)
        client = house.client(Path(task_dir) / "card.toml", REQUESTS)
        deadline = client.deadline if client.deadline is not None else time.time() + model.UNIT_SECONDS
        work = Path(tempfile.mkdtemp(prefix=f"{AGENT}-"))
        agent = N11(task_dir, out, client, steps, work, deadline)

        def finish() -> None:
            agent.release()
            agent.summarise()
            closed = guard.close(out, agent.unit.canaries if agent.unit else ())
            steps("close", closed.summary())

        watchdog = guard.Watchdog(deadline, agent.kill, finish, steps, moved=agent.moved).arm()
        agent.solve()
    except BaseException as exc:  # noqa: BLE001 - the exit is 0 whatever failed
        steps("error", failure(exc))
    finally:
        try:
            if watchdog is not None:
                watchdog.cancel()
            if agent is not None:
                with guard.WRITING:         # the draft's held floor, where the output holds nothing to grade
                    agent.drafts_close()
                agent.summarise()
                if agent.unit is not None:
                    canaries = agent.unit.canaries
            with guard.WRITING:
                closed = guard.close(out, canaries)
            steps("close", closed.summary())
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
        except BaseException as exc:  # noqa: BLE001
            steps("error", failure(exc))
    return 0


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


PARAMETER = re.compile(r"<parameter=(\w+)>\n?(.*?)(?:\n?</parameter>|\Z)", re.S)


def cut_from_text(text: str) -> dict[str, str]:
    """A cut call's arguments from the reply's text where the route parsed no call: a `<tool_call>`'s JSON
    `arguments` as far as they got, or its `<parameter=...>` blocks."""
    text = text or ""
    at = text.rfind('"arguments"')
    if at >= 0:
        found = commands.cut_arguments(text[at + len('"arguments"'):])
        if found:
            return found
    return {name: value for name, value in PARAMETER.findall(text) if name in ("path", "text")}


def clear(out: Path) -> None:
    """Every entry of the output directory removed, a link as itself, never followed."""
    for entry in sorted(Path(out).iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def failure(exc: BaseException) -> str:
    """The exception's type and where it was raised; a model error's own message (model.py's, which never carries a
    credential), with any canary line dropped. No other message is logged: it could quote the instruction."""
    frame = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
    where = f" at {Path(frame.filename).name}:{frame.lineno}" if frame else ""
    said = f": {logged(str(exc))[:200]}" if isinstance(exc, model.ModelError) else ""
    return f"{type(exc).__name__}{where}{said}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="solve", description="N11: a short loop of fenced replies over small files "
                                                               "and the method library.")
    parser.add_argument("verb", choices=["solve"])
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--out", required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 0
    return solve(Path(args.task_dir), Path(args.out))


if __name__ == "__main__":
    try:
        main()
    finally:
        # The status is 0 whatever the interpreter's own shutdown meets: a stderr nobody reads fails its last flush,
        # which Python reports as status 120.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        os._exit(0)
