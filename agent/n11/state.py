"""Form `state` (lane state1; the owner's in249): the loop over a state the harness rebuilds every turn, never over a
transcript.

The House runs no command whose output it must read: the only feedback it needs is whether the code it wrote ran and
what failed. So each request carries one system prompt (`system`: the rules for scripts, how to work, the reply form,
the runtime card, the convention lines and the library's index) and one user turn rebuilt from the harness's own state
(`render`), each section under a plain heading line, in `SECTIONS`' order: the task (its canary lines removed), the
paths, the input files described (`inputs`), the cards, the House's files under work/ as they stand (`files`), the
last run's result with the harness's answers to the last reply, the ledger (one line a turn, `ledger_line`, a run said
by `outcome`) and the budget. No tool-call JSON, no earlier observation,
no earlier reply. Where form `sketch`'s switch is named (lane plan4's addendum 3), the first state closes on
`SKETCH_ASK` and every later one carries the House's sketch as SKETCH after the input files (`sketch`: `SKETCH_HEAD`
and the paragraph verbatim, cut at its end within `SKETCH_TOKENS`), the one reply the state carries back.

- `read(reply, finish_reason)` is the reply form: fenced blocks headed `python file=work/<name>.py` (another text file
  under work/ by its path: a block whose fence line or first line names `file=work/notes.txt` is that file's, never
  work/main.py's, subject to the workspace path check; a labelled first line outside work/ is read only in a python
  or untagged block, and otherwise stays an example), and the directives `lib <function or
  shelf>` and `finish` (`finish accept: <why>`), each alone on its line outside any block; nothing else is read. A block left open where the output cap cut the reply is
  marked `cut` (left open in a reply the cap did not cut, it is whole).
- `inputs(data_dir, instruction, canaries)` describes the input files within `INPUTS_TOKENS` (SPINE 126): a table's
  row count and columns, each column's type, empties and range (a date column's first and last date by the calendar,
  in any of the forms `when` reads), the values of a column holding at most `DISTINCT` of them and repeating one, in
  place of its range, its first and last rows (`NAMED_ROWS`
  for a file the task names by its name, `ROWS` for any other, the named described first with twice the room); any
  other file as the data digest describes it (common.digest).
- `files(root, shown)` is the House's files under work/, each a fenced block headed by its path, whole, within
  `FILES_TOKENS` (a file past it named with its size); the files the runs wrote there are named with their sizes.
- `last_run(runs, notes, setting, cards)` is the LAST RUN section in one of `RESULTS` (lane state2; the owner's
  in275): the last turn's runs (`Report`) and the harness's answers to the last reply (`Note`). `full` is lane
  state1's, each run as `run` observes it and each answer whole; `lean` is each run's command and exit, its one failing
  line (`failing`), its findings and checks one line each, the output digest where it ran clean (where a check or
  finding fails, the digest's lines for the files they name alone, `files_named`) and the observation's closing lines,
  then each answer in one line (a `lib` answer's documentation whole); `minimal` is each run's command and exit, its
  failing line, findings and failing checks, then the refusals alone (a save that does not compile among them) and
  each `lib` answer: a function's name where the request's CARDS section (`cards`) holds its card whole, else the
  documentation whole, as a shelf's (`card`).

Every bound on a text the House reads is in the model's tokens (lane state4, as lane tokens1 made the loop's;
agent/n11/measure.py): each `_TOKENS` target is set so that no text of form state's own (a stored state record's and
the dev units' input files) is shown less than under its byte bound, and each `_BYTES` (or `_CHARS`) value is the
fallback where the tokenizer does not load, under which every text is as before lane state4. No bound is said to the
House.

Pure: no request and no run. agent/n11/solve.py drives the loop (`N11.state_loop`), and scripts/n11_offline.py renders
the state a stored turn would carry. Nothing here names a unit or a quantity (rule 8).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from common import canary, digest, runner
from n11 import commands, library, measure, sanity

FILES = f"FILES UNDER {commands.WORK}/"   # the House's files section's heading
SECTIONS = ("TASK", "PATHS", "INPUT FILES", "CARDS", FILES, "LAST RUN", "LEDGER", "BUDGET")
# The bounds in the model's tokens (lane state4) and their fallbacks. Each target is the most tokens its byte (or
#   character) bound admitted on form state's texts: E083's 479 requests over the 58 dev units (the rebuilt state with
#   the sketch, no withheld unit) and, for the input files, the 58 dev units' own (runs/lanes/state4-report.md)
LEDGER_TOKENS = 35              # a ledger line, at most: the most a line E083 cut at its 100 characters held (34)
LEDGER_CHARS = 100              # the fallback, in characters
FILES_TOKENS = 14_600           # the files section, at most: a file past it is named, never cut. E083's sections run
#                                 3.40 bytes a token at the median and 2.74 at the densest of 1 KB or more (39,330 B the
#                                 largest): the byte bound at that ratio
FILES_BYTES = 40_000            # the fallback
INPUTS_TOKENS = 19_000          # the input files described, at most, and `MIN_SHARE_TOKENS` a file's least share, the
MIN_SHARE_TOKENS = 950          #   byte cap's proportion: the least cap (to 500) at which no file of a dev unit is
#                                 described less than under the bytes (at 18,500 a table of digits, 1.14 bytes a token,
#                                 is cut shorter); 52 of the 58 read as under the bytes, the largest 11,735 tokens
INPUTS_BYTES = digest.DIGEST_BYTES     # the fallback: the data digest's own cap
MIN_SHARE = 1_200               # a file's description never shrinks below this while the cap lasts (the fallback's)
ROWS = (5, 2)                   # a table's first and last rows shown
NAMED_ROWS = (10, 3)            # and a table's the task names
DISTINCT = 12                   # a column with at most this many values has them listed
TABLE_ROWS = 200_000            # a table's rows read for its statistics
COLUMNS = 40                    # a table's columns described, at most
CELL = 24                       # a value shown, at most
LINE = 300                      # a row shown, at most
NOTES_TOKENS = 3_400            # the harness's answers to one reply, at most: the byte bound at the densest answer,
#                                 a `lib` answer's 2.37 bytes a token (lane tokens1's; E083's answers 2.38, 955 B at most)
NOTES_BYTES = 8_000             # the fallback
LIB_SHOWN = 3                   # the `lib` directives one reply has answered
NOTHING_CAP = 2                 # replies in a row with no file and no directive before the unit ends
STUCK_ENDS = 3                  # an identical file resent this many times ends the unit
CLEAN_OUTPUT = "clean-output"   # the last clean run's output, beside work/ (the stuck end puts it back)
RESULTS = ("full", "lean", "minimal")   # the LAST RUN section's settings (lane state2), `full` lane state1's
MESSAGE_TOKENS = 75             # a failing run's error message in `lean` and `minimal`, at most: 200 characters at
#                                 the densest of E083's messages of 60 characters or more (2.74 bytes a token)
MESSAGE_CHARS = 200             # the fallback, in characters
SKETCH = "SKETCH"               # lane plan4's addendum 3: form `sketch`'s section, after INPUT FILES, where switched on
SKETCH_HEAD = "Your sketch, written before any code; the run's result overrides it:"
SKETCH_TOKENS = 600             # the House's paragraph, at most: cut at its end, said in the ledger. The most a
#                                 sketch E083 cut at the byte bound held, 561 (3.55 bytes a token; 4.59 at the median)
SKETCH_BYTES = 2_000            # the fallback

EXAMPLE = """An example reply, in its shape (its path on the fence line, the file's own first line right under it):
```python file=work/main.py
import json
<the rest of the script, whole: it reads the inputs, computes each quantity and writes every output file>
```"""     # lane fence1: the reply form's one example, in shapes, after its directives in `SYSTEM`

SYSTEM = """You are an expert quantitative developer. You solve a task by writing a Python 3 script under work/, \
which the harness saves and runs, unattended, with no network; hidden tests then check the files your scripts wrote \
into the output directory against the task.

Each message is the task's whole state, rebuilt by the harness every turn under plain headings: the task, its paths, \
its input files described in full, the library's cards, your files under work/ as they stand, the last run's result, \
a ledger of the turns so far (one line each) and your budget. Nothing else carries over from one message to the \
next: what is not in your files is gone.

How to work:
1. Settle every convention the task states (day counts, annualisation, ddof, compounding, signs, rounding) before \
computing anything.
2. Write the whole solution as one script, work/main.py, that reads the inputs, computes each quantity (a quantity \
the library computes is a call of its function) and writes every output file the task asks for.
3. Where the last run failed, or a check or a finding stands, fix the failing file: write it again, whole.
4. Read the last run's result against the task: each value's sign, range and order (ES at least VaR, a probability \
in [0, 1]).
5. Finish when the checks pass and the values hold.

Your reply is read for two things, and nothing else in it is read:
- files: each file one fenced block whose opening line names it, ```python file=work/<name>.py (another text file \
under work/ by its path), saved whole. A saved work/main.py runs; a saved module runs work/main.py where main.py \
imports it; any other saved file with an `if __name__ == "__main__":` block runs itself, after main.py. A reply that \
saves nothing runs nothing.
- directives, each alone on its line outside any block: `lib <function or shelf>` shows the library's documentation \
and adds the function's card to the cards; `finish` ends the task (`finish accept: <why>` where a finding stands), \
and nothing else does.
""" + EXAMPLE + """
A reply that runs past the output limit is cut, and a block the cut falls in is not saved, only its complete \
definitions, as a draft. So write work/main.py whole, in one block, where one reply holds it; where a reply would be \
cut, split the script into as few files as fit, each a module work/main.py imports, and for no other reason; after a \
cut, write the rest one file a reply.

{api}

Unless the task says otherwise:
{conventions}

Every script must:
- read its inputs from the true paths given under PATHS: the paths in the task's text are those of another machine;
- write every output file the task asks for, with exactly the names, formats, keys and columns it gives, directly \
into the output directory given under PATHS, and nothing anywhere else but work/ and /tmp;
- use only the Python standard library, qf and these installed packages: {libraries}; no network, no pip, no \
subprocess;
- be deterministic: seed any random generator the task does not seed itself;
- write JSON with plain Python numbers (convert numpy values with float() or int());
- never copy the task's text into an output file, and never write a file named reward.json; `<canary>` in the task \
or the data stands for a watermark string: never write it, or an input field that holds it, into any output file;
- end within its run's limit of {run:.0f} seconds, printing a short line a step at most.

{library}

{index}"""

NOTHING = ("nothing read: your reply held no file block and no directive, and prose is not read. Write the next file "
           f"as a block opened by ```python file={commands.WORK}/<name>.py, or the line `finish` where the output is "
           "done; a second reply in a row with nothing is asked once more for a file where the budget allows it, "
           "else it ends the task.")   # lane prune1: the code's forced ask (clock1), said as it is
UNREAD = (" `run` and `inspect` are no directives here: a saved file runs, and the input files are described above.")
IDENTICAL = ("write refused: {named} already holds this text{since}; nothing was written and nothing ran. Change the "
             "file to fix it.")
STUCK = ("write refused: {named} resent identical a second time (turns {first} and {second}): the file as it stands "
         "is not what fails, and resending it changes nothing. Change the method (another formula, another library "
         "call, another way to read the inputs), not the file as it is; a third identical resend ends the task.")
DONE = ("the checks pass and every output file the task names is present: write the line `finish` now, alone on its "
        "line, unless a value breaks a convention the task states (its day count, annualisation, ddof, compounding, "
        "signs, rounding) or its range (ES at least VaR, a probability in [0, 1]); only then change the file, for that "
        "value alone.")   # lane fence1: a clean run's next state asks for `finish` (`CLEAN`, and the resends' below)
CLEAN = "{named} ran clean, " + DONE
IDENTICAL_CLEAN = ("write refused: {named} already holds this text{since}; nothing was written and nothing ran. The "
                   "last run was clean, " + DONE)
STUCK_CLEAN = ("write refused: {named} resent identical a second time (turns {first} and {second}); nothing was written "
               "and nothing ran. The last run was clean, " + DONE + " A third identical resend ends the task on that "
               "output.")
UNSAVED = {    # lane fence1: a block never saved (`read`, commands.Fenced.void), said as a refusal
    "empty": "not written: the block for {named} held no code; nothing was saved from it.",
    "path": ("not written: a block held the path {named} alone, and no text; a file's block holds the file whole, its "
             "path on the fence line itself (```python file=work/<name>.py), and nothing was saved from it."),
    "directive": ("not read: `{what}` stood inside a block; a directive stands alone on its line outside any block, and "
                  "nothing was saved from that block.")}
UNSAVED_PART = {"empty": "{name}: an empty block, not saved", "path": "{name}: its path line alone, not saved",
                "directive": "a directive in a block, not read"}     # the ledger's words for each
OUTSIDE = "write refused: {why}; nothing was written"
CUT_SHORT = (" {n} replies in a row were cut: write one small file now, a module work/main.py imports holding one or "
             "two functions, as its own block, and the next after it.")
FORCED = ("nothing read again: your reply held no file block. Write the next file now, whole, as one block opened by "
          f"```python file={commands.WORK}/<name>.py: {commands.WORK}/main.py, or the next module where you split it; "
          "a reply with no file block now ends the task.")     # lane clock1: the forced ask, the state's words
LAST_CALL = ("last call: no output file the task names exists yet, and the budget buys this one reply alone. Write "
             f"{commands.WORK}/main.py, whole, as one block opened by ```python file={commands.WORK}/main.py, so that "
             f"it writes every output file the task names from what the files under {commands.WORK}/ above already "
             "compute: each output file in its own try/except, so that one failure does not stop the others; where "
             "nothing computes a value, leave it out rather than invent one, and still write the file with what is "
             f"there. Nothing else in this reply: the harness saves it, runs {commands.WORK}/main.py with all the time "
             "left, and the task ends on its output.")                  # lane clock1: the last call, the state's words
SELF_RUN = ("({named} ran itself, by its own `if __name__ == \"__main__\":` block: {main} does not import it, so {main} "
            "did not run)")                                             # lane clock1
SKETCH_ASK = {  # lane plan4's addendum 3: the first state's closing ask, in the state's reply form's words
    "turn": ("Before any code, describe in a short paragraph how you will solve the task, in what order, and which "
             "library functions from {source} you will call where the task names their methods. Answer in prose "
             "alone: no file block, no directive and no list of files in this reply; where you are unsure, say so."),
    "inline": ("Begin your reply with a short paragraph in prose, before any block: how you will solve the task, in "
               "what order, and which library functions from {source} you will call where the task names their "
               "methods; no list of files; where you are unsure, say so. Then, in the same reply, write the code, each "
               f"file a block opened by ```python file={commands.WORK}/<name>.py.")}
SKETCH_NEXT = (f"your sketch is kept above under {SKETCH}; now write the code, each file a block opened by ```python "
               f"file={commands.WORK}/<name>.py; where the task or a run's result differs from your sketch, follow "
               "them.")                                                 # `turn`: the note after the sketch


def system(index: str, api: str, conventions: str, libraries: str, run: float) -> str:
    """The state form's system prompt: `SYSTEM` with the runtime card, the convention lines, the installed packages,
    a run's limit and the library's index."""
    return SYSTEM.format(api=api, conventions=conventions, libraries=libraries, run=run, index=index,
                         library=library.USE.format(cards="the CARDS section"))


def sketch(text: str, bound: int | None = None) -> tuple[str, bool]:
    """The SKETCH section's body (lane plan4's addendum 3): `SKETCH_HEAD` and the House's paragraph verbatim, cut at
    its end where it runs past `bound` (`SKETCH_TOKENS`, else `SKETCH_BYTES`; at the last space before the bound,
    `library.CUT` after it); and whether it was cut. No size is said."""
    whole = text.strip()
    bound = measure.bound(SKETCH_TOKENS, SKETCH_BYTES) if bound is None else bound
    if measure.size(whole) <= bound:
        return f"{SKETCH_HEAD}\n{whole}", False
    room = measure.prefix(whole, bound - measure.size(library.CUT))
    kept = room.rsplit(None, 1)[0] if len(room.split()) > 1 else room
    return f"{SKETCH_HEAD}\n{kept.rstrip()}{library.CUT}", True


def render(sections: list[tuple[str, str]]) -> str:
    """The user turn: each section under its plain heading line, in the order given (`SECTIONS`')."""
    return "\n\n".join(f"{head}\n{body.strip(chr(10))}" for head, body in sections) + "\n"


def sized(sections: list[tuple[str, str]]) -> dict[str, int]:
    """Each section's bytes as `render` writes it, its heading line included."""
    return {head: len(f"{head}\n{body.strip(chr(10))}".encode("utf-8")) for head, body in sections}


# ------------------------------------------------------------------------------------------------- the reply


@dataclass
class Block:
    path: str
    text: str
    cut: bool = False               # left open where the output cap cut the reply
    line: bool = False              # its path named on the line after the fence (lane fence1), not on the fence
    named: bool = False             # that path a labelled line's `headed` does not read (commands.labelled)


@dataclass
class Reply:
    blocks: list[Block] = field(default_factory=list)
    libs: list[str] = field(default_factory=list)
    finish: str | None = None       # None where no `finish` line; else its reason, "" where none
    unread: list[str] = field(default_factory=list)     # `run` and `inspect` lines, which are no directives here
    cut: bool = False               # the output cap cut the reply
    unsaved: list[tuple[str, str, str]] = field(default_factory=list)   # (path, why, its directive): blocks never
    #                                 saved (lane fence1): `void`'s "empty", "path" or "directive"
    twice: list[tuple[str, bool]] = field(default_factory=list)         # (path, the same text): a later block for a
    #                                 file the reply had already given one, which replaced it

    @property
    def nothing(self) -> bool:
        return not self.blocks and not self.libs and self.finish is None

    def describe(self) -> str:
        """The reply in a log line: its files' paths and its directives, never a text it carries."""
        parts = [f"write {block.path[:80]}" + (" (cut)" if block.cut else "") for block in self.blocks]
        return ", ".join(parts + [f"lib {name[:60]}" for name in self.libs]
                         + (["finish"] if self.finish is not None else [])
                         + [f"unsaved {path[:80]} ({why})" for path, why, _ in self.unsaved])


LIB = re.compile(r"^lib\s+(?:(?:name|shelf)\s*=\s*)?`?([A-Za-z_][\w.]*)`?\s*$")
FINISH = re.compile(r"^finish\b\s*(.*)$", re.I)
REASON = re.compile(r"^(?:accept\s*:|reason\s*[=:])", re.I)
UNREAD_VERB = re.compile(r"^(run|inspect)\s+\S", re.I)


def directive(line: str) -> tuple[str, str] | None:
    """A line outside any block read as a directive: ("finish", its reason), ("lib", a name), ("unread", `run` or
    `inspect`); None where it is none (prose)."""
    text = line.strip().strip("`*").strip()
    found = FINISH.match(text)
    if found:
        rest = found.group(1).strip().strip("`*").strip()
        if rest in ("", ".", ":") or REASON.match(rest):
            reason = re.sub(r"(?i)^reason\s*[=:]\s*", "", rest).strip().strip("`'\"")
            return "finish", "" if reason in (".", ":") else reason
        return None
    found = LIB.match(text)
    if found:
        return "lib", found.group(1)
    found = UNREAD_VERB.match(text)
    return ("unread", found.group(1).lower()) if found else None


def read(text: str, finish_reason: str | None = "stop") -> Reply:
    """A reply read in the state form (above): its blocks in order (commands.fenced: a block's path on its fence line,
    or alone on the line after it, which is then no line of the file), a block opened by ```python and naming no file
    `work/main.py`'s, a block naming neither a file nor python skipped; a block holding no code, its path line alone
    or directive lines alone never saved (`unsaved`); a later whole block for a file replacing the reply's earlier
    one (`twice`); its `lib` and `finish` lines (the first `finish`); the `run` and `inspect` lines it holds, which
    are not read.
    The rule (lane fencedpath1, after E182): a block whose fence line or first line names `file=<path>` is that
    file's, of any kind (a `.txt`, `.json` or `.csv` file as well as a module), its header line never part of the
    text; the write saves it where the workspace path check accepts it and refuses it by name otherwise (dotfiles
    under work/ included). `labelled` requires a directory or known suffix; another language's outside first-line
    path stays an example. A block is never work/main.py's unless it names work/main.py or is a python block naming
    no file (commands.labelled; a python block's quoted assignment, `file = "out.json"`, is code)."""
    reply = Reply(cut=finish_reason == "length")
    for item in commands.fenced(runner.THINK.sub("", text or "")):
        if isinstance(item, commands.Fenced):
            path = item.path or (f"{commands.WORK}/main.py" if item.tag in commands.PYTHON else None)
            cut = not item.closed and reply.cut
            if path is None:
                continue
            if item.void and not cut:
                reply.unsaved.append((path, item.void, item.directive))
                continue
            if not cut:
                earlier = next((k for k, block in enumerate(reply.blocks)
                                if not block.cut and commands.plain(block.path) == commands.plain(path)), None)
                if earlier is not None:
                    reply.twice.append((path, reply.blocks.pop(earlier).text == item.text))
            reply.blocks.append(Block(path, item.text, cut, item.header == "line", item.named))
            continue
        said = directive(item)
        if said is None:
            continue
        kind, value = said
        if kind == "finish" and reply.finish is None:
            reply.finish = value
        elif kind == "lib":
            reply.libs.append(value)
        elif kind == "unread":
            reply.unread.append(value)
    return reply


# -------------------------------------------------------------------------------------------- the ledger


def ledger_line(turn: int, parts) -> str:
    """One ledger line: the turn and what came of its reply, its saves said as one (`wrote a.py, main.py`), within
    `LEDGER_TOKENS` (`within`)."""
    said: list[str] = []
    for part in (part for part in parts if part):
        if part.startswith("wrote ") and said and said[-1].startswith("wrote ") and "; " not in said[-1]:
            said[-1] += ", " + part.removeprefix("wrote ")
        else:
            said.append(part)
    return within(f"turn {turn}: " + ("; ".join(said) or "nothing"), LEDGER_TOKENS, LEDGER_CHARS)


def within(text: str, tokens: int, chars: int, end: str = "...") -> str:
    """`text` within a line's bound, `end` closing it where it was cut: `tokens` of the model's where the bounds are in
    tokens (agent/n11/measure.py), else `chars` characters, as before lane state4."""
    if measure.UNIT.tokenizer is None:
        if len(text) <= chars:
            return text
        head = text[:chars - len(end)]
        return head.rstrip() + end if end else head
    if measure.size(text) <= tokens:
        return text
    head = measure.prefix(text, tokens - measure.size(end))
    return (head.rstrip() if end else head) + end


def bare(named: str) -> str:
    return named.removeprefix(f"{commands.WORK}/")


def where(ran: commands.Ran) -> str:
    """A failing run's innermost frame inside work/, `file:line`; "" where it has none."""
    return f"{bare(ran.frames[-1][0])}:{ran.frames[-1][1]}" if ran.frames else ""


def outcome(ran: commands.Ran, same: int | None = None) -> str:
    """A run in a ledger line: the file, its exit and where it failed (the same traceback as turn `same`'s, where an
    earlier turn's run failed there on the same error), or what its output met."""
    name = bare(ran.path)
    if ran.exit is None:
        return f"{name} stopped at its time limit"
    if ran.exit != 0 or ran.error:
        at = where(ran)
        if same is not None and at:
            return f"{name} exit {ran.exit}, the same traceback at {at} as turn {same}"
        kind = ran.error.split(":", 1)[0].strip().rsplit(".", 1)[-1][:40] if ran.error else ""
        return f"{name} exit {ran.exit}" + (f" at {at}" if at else "") + (f" ({kind})" if kind else "")
    if ran.failures:
        return f"{name} exit 0, {len(ran.failures)} check(s) failing"
    if ran.findings:
        return f"{name} exit 0, {len(ran.findings)} finding(s)"
    return f"{name} exit 0, the checks pass"


# ------------------------------------------------------------------------------------------ the last run


CUT = ("not written: the output limit cut your reply inside {named}, so it was not saved; {kept}. It does not fit one "
       f"reply: split it {commands.SPLIT}.")
CUT_STREAK = (" {n} replies in a row were cut: write one small file now (a module work/main.py imports, one or two "
              "functions), and the next after it.")
SIZED = re.compile(r"^(wrote \S+): [\d,]+ B")                  # commands.Workspace.write's first line
DIGEST_FILE = re.compile(r"^- (.+?) \([\d,]+ B\)")             # sanity.read's line for one output file


def first(text: str) -> str:
    return (text or "").strip().split("\n", 1)[0]


@dataclass(frozen=True)
class Note:
    """One of the harness's answers to a reply as each setting says it: `full` whole (lane state1's words), `lean` in
    one line (a `lib` answer's documentation whole), `minimal` in one line where it is a refusal, a save that does not
    compile or a card's name, else nothing (""); but where the answer is `lib` on a function (`card`, its card whole)
    and the request's CARDS section (`cards`) does not hold that card whole, `minimal` says it whole too."""
    full: str
    lean: str
    minimal: str = ""
    card: str = ""              # a `lib` answer on a function: its card whole (library.Card), as CARDS holds it

    def said(self, setting: str, cards: str = "") -> str:
        if setting == "minimal" and self.card and self.card not in cards:
            return self.full
        return self.lean if setting == "lean" else self.minimal if setting == "minimal" else self.full


def note(text: str, line: str | None = None, refusal: bool = False) -> Note:
    """An answer whose one line is `line` (its first line where None), `minimal`'s too where it is a refusal."""
    line = first(text) if line is None else line
    return Note(text, line, line if refusal else "")


def saved(said: str, cycle: str = "") -> Note:
    """A save's answer (commands.Workspace.write's), a cycle's line after it: `lean` its first line without the file's
    size, then the cycle's; `minimal` that first line where the file does not compile, else nothing."""
    line = SIZED.sub(r"\1", first(said))
    return Note("\n".join(part for part in (said, cycle) if part), "\n".join(part for part in (line, cycle) if part),
                line if "does not compile" in line else "")


def cut(full: str, named: str, kept: str, streak: int = 0) -> Note:
    """A block the output cap cut, so not saved (a refusal): `full` in the loop's words, `lean` and `minimal` in
    `CUT`'s, which state no size, the replies cut in a row said where `streak`."""
    line = CUT.format(named=named, kept=kept) + (CUT_STREAK.format(n=streak) if streak else "")
    return Note(full, line, line)


def card(name: str, said: str, function: library.Function | None = None) -> Note:
    """A `lib` answer (`function` the function it names, else None): the documentation whole, but under `minimal` a
    function's name and where its card went where the request's CARDS section holds that card whole (`Note.said`). A
    card CARDS trimmed asks for this very `lib`, one it left out is not there, and a shelf's cards stand in no section,
    so those answers stay whole: a setting under which the documentation could never be read would confound the A/B
    on the section's size (lane state2's review). A name that is none is refused in one line."""
    if function is None:
        return Note(said, said, said)
    return Note(said, said, f"lib {name}: its card is under CARDS", library.Card(function).text)


@dataclass(frozen=True)
class Report:
    """One run of the last turn: as `run` observes it (commands.Workspace.observe: `full`'s), the run, and the lines
    its observation closes with (`tail`: the same failing line, the gate's)."""
    said: str
    ran: commands.Ran
    tail: str = ""

    def text(self, setting: str) -> str:
        """The run in `setting` (`last_run`), within the run's bound."""
        if setting not in ("lean", "minimal"):
            return self.said
        ran = self.ran
        failures, findings = [first(line) for line in ran.failures], [first(line) for line in ran.findings]
        parts = [head(ran), failing(ran)]
        if setting == "lean":
            shown = files_named(ran.digest, failures + findings) if failures or findings else ran.digest
            parts += [f"outputs:\n{shown}" if shown else "", sanity.said(findings), commands.verdict(failures),
                      self.tail]
        else:
            parts += [sanity.said(findings), commands.verdict(failures) if failures else ""]
        return commands.fit("\n".join(part for part in parts if part), commands.BOUNDS["run"])


def head(ran: commands.Ran) -> str:
    """A run's command and its exit."""
    return f"run {ran.path}: " + ("stopped at its time limit" if ran.exit is None else f"exit {ran.exit}")


def failing(ran: commands.Ran) -> str:
    """A failing run's one line: its innermost frame inside work/ as `file:line`, the error's kind and its message's
    first line, each cut at `MESSAGE_TOKENS` (`within`); "" where it raised nothing."""
    if not ran.error:
        return ""
    kind, _, message = ran.error.partition(":")
    message = within(first(message), MESSAGE_TOKENS, MESSAGE_CHARS)
    at = f" at {ran.frames[-1][0]}:{ran.frames[-1][1]}" if ran.frames else ""
    return f"error{at}: " + ": ".join(part for part in (within(kind.strip(), MESSAGE_TOKENS, MESSAGE_CHARS, ""),
                                                        message) if part)


def files_named(digest: str, texts) -> str:
    """The output digest's lines (sanity.read's, one an output file) for the files `texts` (the failing checks and the
    findings) name, by the file's path or its name as the digest shows it (`sanity.shown`: a character outside
    printable ASCII as `?`); "" where they name none."""
    said, kept = "\n".join(sanity.UNSHOWN.sub("?", text) for text in texts), []
    for line in digest.split("\n"):
        found = DIGEST_FILE.match(line)
        if found is None:
            continue
        name = found.group(1)
        clipped = name.endswith("...")          # sanity.shown cuts a long name: read as its start
        name = name[:-3] if clipped else name
        end = "" if clipped else r"(?![\w-])"
        if any(re.search(r"(?<![\w.-])" + re.escape(each) + end, said)
               for each in {name, name.rsplit("/", 1)[-1]} if each):
            kept.append(line)
    return "\n".join(kept)


def last_run(runs: tuple[int, list[Report]] | None, notes: tuple[int, list[Note]], setting: str = "full",
             cards: str = "") -> str:
    """The LAST RUN section in `setting` (`RESULTS`): the last turn's runs, labelled by the turn they ran at, each as
    `Report.text` says it ("no run yet" before any); then the harness's answers to the last reply as each `Note` says
    it there beside `cards` (the CARDS section the same request carries), within `NOTES_TOKENS`."""
    said = "\n\n".join(f"turn {runs[0]}: {report.text(setting)}" for report in runs[1]) if runs is not None else \
        "no run yet"
    turn, answers = notes
    lines = [line for line in (answer.said(setting, cards) for answer in answers) if line]
    if lines:
        said += f"\n\nthe harness, on turn {turn}'s reply:\n" + \
            commands.fit("\n".join(lines), measure.bound(NOTES_TOKENS, NOTES_BYTES), unit=measure.UNIT)
    return said


# --------------------------------------------------------------------------------------------- the files


def files(root: Path, shown, bound: int | None = None) -> str:
    """The House's files (`shown`, `work/<name>` paths in order) as they stand under `root`, each a fenced block
    headed by its path, whole, a file gone from disk left out and one past `bound` (`FILES_TOKENS`, else
    `FILES_BYTES`) named with its size; then the other files under work/ (the runs' own) named with their sizes."""
    root = Path(root)
    bound = measure.bound(FILES_TOKENS, FILES_BYTES) if bound is None else bound
    parts, size, left, listed = [], 0, [], set(shown)
    for named in shown:
        here = root / named
        try:
            text = here.read_text(encoding="utf-8", errors="replace") if here.is_file() else None
        except OSError:
            text = None
        if text is None:
            continue
        block = f"```{'python' if named.endswith('.py') else 'text'} file={named}\n{text.rstrip(chr(10))}\n```"
        if size + measure.size(block) > bound:
            left.append(f"{named} ({here.stat().st_size:,} B)")
            continue
        parts.append(block)
        size += measure.size(block) + measure.size("\n\n")
    work, others = root / commands.WORK, []
    for path in sorted(work.rglob("*")) if work.is_dir() else []:
        named = f"{commands.WORK}/{path.relative_to(work).as_posix()}"
        if path.is_file() and "__pycache__" not in path.parts and named not in listed:
            others.append(f"{named} ({path.stat().st_size:,} B)")
    if left:
        parts.append("not shown, the section" + ("'s" + f" {bound:,} B" if measure.UNIT.tokenizer is None else "")
                     + " being full: " + ", ".join(left))       # in tokens, no size said
    if others:
        parts.append(f"also under {commands.WORK}/, written by the runs (not shown): " + ", ".join(others[:20])
                     + (f" and {len(others) - 20} more" if len(others) > 20 else ""))
    return "\n\n".join(parts) if parts else "none yet"


# ---------------------------------------------------------------------------------------- the input files


def inputs(data_dir: Path, instruction: str, canaries=(), cap: int | None = None) -> str:
    """The input files described (above) within `cap` (`INPUTS_TOKENS`, else `INPUTS_BYTES`), a file's share never
    under `MIN_SHARE_TOKENS` (else `MIN_SHARE`) while the cap lasts, the canaries marked `<canary>`. In tokens the
    data digest's room for a file (a JSON file shown whole, a text's head) is its share at the cap's own bytes a
    token, and no size is said."""
    data_dir = Path(data_dir)
    found = sorted(path for path in data_dir.rglob("*") if path.is_file()) if data_dir.is_dir() else []
    if not found:
        return f"No input files: `{data_dir}` holds none."
    tokens = measure.UNIT.tokenizer is not None
    cap = measure.bound(INPUTS_TOKENS, INPUTS_BYTES) if cap is None else cap
    least = measure.bound(MIN_SHARE_TOKENS, MIN_SHARE)
    named = [path for path in found if path.name in instruction]
    total = sum(path.stat().st_size for path in found)
    lines = [f"{len(found)} input file{'s' if len(found) != 1 else ''} under `{data_dir}`, {total:,} bytes"
             + (f"; the {len(named)} the task names come first, with more rows" if named else "") + ":"]
    lines += [f"- {path.relative_to(data_dir)} ({path.stat().st_size:,} bytes)" for path in found[:digest.LISTED]]
    if len(found) > digest.LISTED:
        lines.append(f"- ... and {len(found) - digest.LISTED} more")
    text = "\n".join(lines) + "\n"
    share = max(least, (cap - measure.size(text)) // (len(found) + len(named)))
    described = 0
    for path in named + [path for path in found if path not in named]:
        room = min(share * (2 if path in named else 1), cap - measure.size(text))
        if room < least // 2:
            break
        head = f"\n{path.relative_to(data_dir)}" + (" (the task names it)" if path in named else "") + ":\n"
        text += clip(head + portrait(path, room * INPUTS_BYTES // INPUTS_TOKENS if tokens else room, path in named),
                     room)
        described += 1
    if described < len(found):
        rest = len(found) - described
        text += f"\n[{rest} more file{'s' if rest != 1 else ''} not described: " + \
            ("the section is full]\n" if tokens else f"the cap is {cap:,} bytes]\n")
    return canary.redact(text, canaries)


def clip(text: str, room: int) -> str:
    """`text` within `room` in the bounds' unit, cut at a line with the data digest's marker after it: the digest's
    own cut (common.digest.clip) where the unit is bytes."""
    if measure.UNIT.tokenizer is None:
        return digest.clip(text, room)
    if measure.size(text) <= room:
        return text
    marker = "\n[... cut]\n"
    return measure.prefix(text, max(0, room - measure.size(marker))).rsplit("\n", 1)[0] + marker


def portrait(path: Path, room: int, named: bool = False) -> str:
    """One input file described: a table (a CSV, TSV or text table, a parquet file) by `table` or `parquet`, any
    other file, or a table neither can read, as the data digest describes it."""
    head, tail = NAMED_ROWS if named else ROWS
    suffix = path.suffix.lower()
    try:
        if suffix in (".parquet", ".pqt"):
            return parquet(path, head, tail)
        if suffix in commands.TABLES and not digest.binary(path):
            sample = path.read_bytes()[:65_536].decode("utf-8-sig", errors="replace")
            if commands.table_like(sample):
                return table(path, head, tail, "\t" if suffix == ".tsv" else commands.sniff(sample))
    except Exception:  # noqa: BLE001 - a file read wrong here is described as the digest describes it
        pass
    return digest.describe(path, room)


DATED = ("%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d-%b-%y", "%d.%m.%Y")   # dates beside ISO's and slashed ones


def slashed(cell: str) -> tuple[int, int, int] | None:
    """A slashed date's fields (first, second, year), a two-digit year read in the 2000s; None where it is none."""
    parts = cell.split("/")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    first, second, year = (int(part) for part in parts)
    return first, second, year + (2000 if year < 100 else 0)


def when(cell: str, hint: list[str] | None = None) -> tuple | None:
    """A date cell's place on the calendar, comparable with another's: an ISO date or time, a slashed date (day
    first where its first field passes 12, else month first; `Column` reads a column's in one order), or one with its
    month named (`DATED`; `hint`, a column's, keeps the form that read its last cell first); None where it reads as
    none, a bare number among them."""
    if commands.number(cell) is not None:
        return None
    try:
        return tuple(datetime.fromisoformat(cell.replace("Z", "+00:00")).timetuple()[:6])
    except ValueError:
        pass
    fields = slashed(cell)
    if fields is not None:
        first, second, year = fields
        return (year, second, first) if first > 12 else (year, first, second)
    forms = hint if hint is not None else list(DATED)
    for form in list(forms):
        try:
            moment = datetime.strptime(cell, form)
        except ValueError:
            continue
        if hint is not None:
            hint.remove(form)
            hint.insert(0, form)
        return tuple(moment.timetuple()[:6])
    return None


def widened(span: tuple | None, moment: tuple, cell: str) -> tuple:
    """A column's (earliest, latest) dates, each (its place, its cell), widened by one cell."""
    if span is None:
        return (moment, cell), (moment, cell)
    early, late = span
    return (early if early[0] <= moment else (moment, cell)), (late if late[0] >= moment else (moment, cell))


class Column:
    """One table column's statistics, read cell by cell: its empties, its first and last values, its numeric range
    where every value is a number, its first and last date where every value is a date (`when`; its slashed dates
    read day first throughout where one's first field passes 12 and none's second does), and its values where it
    holds at most `DISTINCT` of them."""

    def __init__(self, name: str) -> None:
        self.name, self.empty, self.filled = name, 0, 0
        self.sample: list[str] = []
        self.numeric, self.low, self.high = True, None, None
        self.values: Counter | None = Counter()
        self.first = self.last = ""
        self.dated, self.forms = True, list(DATED)
        self.span: tuple | None = None          # the earliest and latest date, each slashed cell read by `when`
        self.span_day: tuple | None = None      # and with every slashed cell read day first
        self.day_first = self.month_first = False   # a slashed cell's first field past 12; its second

    def add(self, cell: str) -> None:
        cell = cell.strip()
        if cell.lower() in commands.EMPTY:
            self.empty += 1
            return
        self.filled += 1
        self.first, self.last = self.first or cell, cell
        if len(self.sample) < 2_000:
            self.sample.append(cell)
        number = commands.number(cell)
        if number is None:
            self.numeric = False
        elif self.numeric:
            self.low = number if self.low is None else min(self.low, number)
            self.high = number if self.high is None else max(self.high, number)
        if self.values is not None:
            self.values[cell] += 1
            if len(self.values) > DISTINCT:
                self.values = None
        if self.dated:
            moment = when(cell, self.forms)
            if moment is None:
                self.dated = False
            else:
                fields = slashed(cell)
                if fields is not None:
                    self.day_first |= fields[0] > 12
                    self.month_first |= fields[1] > 12
                self.span = widened(self.span, moment, cell)
                self.span_day = widened(self.span_day, (fields[2], fields[1], fields[0]) if fields else moment, cell)

    def line(self) -> str:
        dated = self.dated and self.span is not None
        kind = "date" if dated else digest.kind(self.sample) if self.sample else "empty"
        parts = [f"- {self.name[:40]}: {kind}"]
        if self.empty:
            parts.append(f"{self.empty:,} empty")
        listed = self.values is not None and not dated and 0 < len(self.values) < self.filled
        if dated:
            early, late = self.span_day if self.day_first and not self.month_first else self.span
            parts.append(f"from {early[1][:CELL]} to {late[1][:CELL]}")
        elif listed:
            ordered = sorted(self.values.items(), key=lambda item: (commands.number(item[0]) is None,
                                                                    commands.number(item[0]) or 0.0, item[0]))
            parts.append(f"{len(ordered)} value{'s' if len(ordered) != 1 else ''}: " + ", ".join(
                f"{digest.short(value, 2 * CELL)} ({count:,})" for value, count in ordered))
        elif self.numeric and self.low is not None:
            parts.append(f"min {self.low:.6g}, max {self.high:.6g}")
        elif self.filled:
            parts.append(f"first {self.first[:CELL]}, last {self.last[:CELL]}")
        return ", ".join(parts)


def table(path: Path, head: int, tail: int, delimiter: str) -> str:
    """A delimited table: its row count and columns, each column's statistics (`Column`), its first `head` and last
    `tail` rows."""
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        columns = [Column(name) for name in header[:COLUMNS]]
        first: list[list[str]] = []
        last: deque = deque(maxlen=tail)
        count = 0
        for row in reader:
            if not row:
                continue
            count += 1
            if len(first) < head:
                first.append(row)
            else:
                last.append(row)
            if count <= TABLE_ROWS:
                for column, cell in zip(columns, row):
                    column.add(cell)
    said = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}.get(delimiter, repr(delimiter))
    lines = [f"a table, {count:,} rows x {len(header)} columns, {said}-separated"
             + (f"; statistics over its first {TABLE_ROWS:,} rows" if count > TABLE_ROWS else "")]
    lines += [column.line() for column in columns]
    if len(header) > COLUMNS:
        lines.append(f"- and {len(header) - COLUMNS} more columns: " + ", ".join(name[:30] for name in
                                                                                header[COLUMNS:80]))
    if first:
        lines.append(f"its first {len(first)} rows, under its header:")
        lines += [digest.short(delimiter.join(row), LINE) for row in [header, *first]]
    if last:
        lines.append(f"its last {len(last)} rows:")
        lines += [digest.short(delimiter.join(row), LINE) for row in last]
    return "\n".join(lines) + "\n"


def parquet(path: Path, head: int, tail: int) -> str:
    """A parquet file: its row count and columns, each column's type, nulls and range (a temporal column's first and
    last), the values of one holding at most `DISTINCT` of them and repeating one in place of its range, its first
    `head` and last `tail` rows."""
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError:
        return digest.parquet(path)
    data = pq.read_table(path)
    lines = [f"parquet, {data.num_rows:,} rows x {data.num_columns} columns"]
    for name, column in list(zip(data.column_names, data.columns))[:COLUMNS]:
        parts = [f"- {name[:40]}: {column.type}"]
        if column.null_count:
            parts.append(f"{column.null_count:,} null")
        temporal = str(column.type).startswith(("timestamp", "date", "time"))
        try:
            counts = [item for item in pc.value_counts(column).to_pylist() if item["values"] is not None] \
                if not temporal else []
        except Exception:  # noqa: BLE001 - a type value_counts does not take
            counts = []
        if counts and len(counts) <= DISTINCT and len(counts) < len(column) - column.null_count:
            parts.append(f"{len(counts)} value{'s' if len(counts) != 1 else ''}: " + ", ".join(
                f"{digest.short(str(item['values']), 2 * CELL)} ({item['counts']:,})" for item in counts))
        else:
            try:
                bounds = pc.min_max(column).as_py()
                if bounds.get("min") is not None:
                    low, high = str(bounds["min"])[:CELL], str(bounds["max"])[:CELL]
                    parts.append(f"from {low} to {high}" if temporal else f"min {low}, max {high}")
            except Exception:  # noqa: BLE001 - a type min_max does not take
                pass
        lines.append(", ".join(parts))
    if data.num_columns > COLUMNS:
        lines.append(f"- and {data.num_columns - COLUMNS} more columns")
    if head and data.num_rows:
        rows = data.slice(0, head).to_pylist()
        lines.append(f"its first {len(rows)} rows:")
        lines += [digest.short(json.dumps(row, default=str), LINE) for row in rows]
        if tail and data.num_rows > head:
            rows = data.slice(max(head, data.num_rows - tail)).to_pylist()
            lines.append(f"its last {len(rows)} rows:")
            lines += [digest.short(json.dumps(row, default=str), LINE) for row in rows]
    return "\n".join(lines) + "\n"
