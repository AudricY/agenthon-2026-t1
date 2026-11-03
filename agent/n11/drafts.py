"""N11's first drafts: the one-shot-first floor, and three drafts with their disagreement table (lane drafts1;
runs/lanes/drafts1-prompt.md: the refuter's accepted form, runs/lanes/r-cold1017-proposals.toml, ACCEPT ROW 600; what a
live vote needs, runs/lanes/votereplay1-report.md section 7; the coordinator's scope addition of 10-18, the one-shot
floor after the pace replay's REJECT ROW 603).

`AGENTHON_N11_DRAFTS` (or agent/n11/DRAFTS, one word, an image's; agent/n11/solve.py `drafts_of`): `off`, today's,
`one` or `three`, the first drafts sent (`COUNTS`), armed in the loop's forms `loop` and `sketch` (`FORMS`). Armed, the
round (`Round`, agent/w1's `W1` on the loop's clock) runs before the loop's first request, after the selection call
where that is armed, and only where the loop keeps `FLOOR_REQUESTS` beyond the round's allowance (`allowance`: `PER_DRAFT`
requests a draft, retries counted, so a route that refuses them costs the loop no more). N0's request (agent/n0
`prompt`: the instruction, the path map, the data's digest, the run's seconds) is drawn, each draft under the next of
agent/w1's preambles as this lane says them (`SYSTEMS`: N0's own, the compact clause, the explicit checks, each count
w1's say reworded to none, `UNCOUNTED`, the owner's in281). Under `one` the one draft is N0's request as N0 sends it,
its system prompt, the client's temperature (0) and the seed; under `three` the three go at once at w1's
`TEMPERATURE`, the seed pinned (the organisers' rule: temperature and seed pinned where supported; one request
thrice at temperature 0 and one seed would return one draft). Each goes through a client of its own (common.house
`Client.sibling`: the unit's one count under `LOCK` and the cap, the desk's pacing floor its own), bounded to the
round's time (`w1.ROUND_SHARE` of the time left). Each script then runs in turn in a directory of its own under the
loop's work (`WORK`), at most `RUN_S` and its share of half the time left (`w1.run_cap`); its files are moved to a
store of its own, so the next starts from an empty output (`w1.W1.keep`). A draft is clean when its run exits 0 with no
traceback and leaves a file at the output's root.

The comparison (`compare`): among the clean drafts, two match where they wrote the same of the files compared (those
the instruction names that any of them wrote, else every file), with the same leaves at the same places (`fields`:
agent/w1's `leaves` over every leaf, a parquet file's frame read as a table), every number within `RTOL` and `ATOL`
(w1's `near`) and every other leaf equal; a file of a kind with no leaves by its presence alone, one that does not
parse or is over `MAX_BYTES` by its bytes. The groups are w1's (`w1.groups`).

The floor is held aside, never written before the loop (the coordinator's decision of 10-18: the read isolates an
empty output replaced by a draft's files from a loop told its files are present): the agreed group's first draft
(`agreed`, two or more matching), else the first clean draft, keeps its files in its store, and at the unit's close
(`close`; the watchdog's too) they are written to the output, what it holds first removed, only where it holds nothing
the harness grades (agent/n11/fallback.py `graded`: no file at its root, or none of the files the task names). The loop
runs as today, its every body today's but for the requests and seconds the round spent. Two or more clean and none
matching: the loop's first message closes on `said`'s block, the clean drafts' scripts and the field table (`table`: every field where two of them
differ, each one's value, `ROWS` rows at most, every field's first before any field's second), asking each field
settled by the task's sentence that decides it and then one script. Fewer than two clean (and always under `one`):
nothing added to the first request, the one clean draft's files the floor where there is one. A
draft lost (no reply, a refusal, a failing run, any error in the round) costs that draft, never the unit. No text
states a count (the owner's in281): the drafts are lettered, never numbered; the table's values are the drafts'
outputs, said live alone (rule 8).

The trace: each draft's `draft` event and its files copied under the trace's `drafts/<i>/` (`w1.W1.outcomes`), and the
round's `drafts` event; the step log's `drafts` line and the summary line carry `KEYS` (agent/n11/solve.py
`drafts_keys`; the `drafts` line the round's, `floor_written` 0 there, and the summary's `floor_written` 1 where the held
floor was written at the close, with a `drafts_floor` event); the round's own lines are its steps under `drafts `
(`drafts round`, `drafts run`, `drafts draft 0`, `drafts floor`).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from common import contract, guard, runner, trace
from common.house import model
from n0 import solve as n0
from n11 import fallback
from w1 import solve as w1

ENV = "AGENTHON_N11_DRAFTS"     # the desk's switch (src/agenthon_t1/harness.py hands it to an --agent run's container)
FILE = "DRAFTS"                 # or agent/n11/DRAFTS, one word, written by an image's builder
SETTINGS = ("off", "one", "three")  # the default first; the others the first drafts sent (`COUNTS`)
COUNTS = {"one": 1, "three": w1.DRAFTS}     # w1's first preambles: N0's own, the compact clause, the explicit checks
FORMS = ("loop", "sketch")      # the forms it is armed in: those whose loop opens on its own first message
PER_DRAFT = 2                   # the round's requests a draft at most, retries counted (w1's room for a retry each)
FLOOR_REQUESTS = 2              # the round runs only where the loop keeps this many requests beyond the allowance
RUN_S = 60.0                    # a draft's run at most
RTOL, ATOL = 1e-4, 1e-9         # two numbers match within these (w1's `near`)
MAX_BYTES = 4 * 1024 * 1024     # a file over this is compared by its bytes alone (lane votereplay1 left such out)
ROWS = 40                       # the table's rows said at most; past them, a line says more differ
VALUE_CHARS = 48                # a value as the table says it, at most
LETTERS = "ABCDE"               # the clean drafts as said, in their order
KIND = "draft"                  # the round's requests' kind on the trace, and its runs'
WORK = "drafts"                 # the round's directory under the loop's work
KEYS = ("drafts_mode", "drafts_sent", "drafts_clean", "floor_written", "drafts_agreed", "table_said", "table_fields",
        "drafts_s")
UNCOUNTED = (("a reply is cut off at about 4,000 tokens", "a reply is cut off at the output limit"),     # w1's counts
             ("the whole script under about 200 lines", "the whole script kept short"))   # and this lane's words
HEAD = "# The first drafts disagree"
SAID = (HEAD + "\n\nBefore this conversation the task was written as a whole script more than once, independently, "
        "and each script was run. The scripts below ran cleanly, but their outputs differ, so they cannot all compute "
        "what the task asks.\n\n{scripts}\n\n"
        "## Where their outputs differ\n\n{table}\n\n"
        "Settle each field in the table before writing code: quote the sentence of the task that decides how it is "
        "computed, and say which value, if any, follows it. Then write one script, work/main.py, that computes every "
        "output that way, and run it.\n")
MORE = "More fields differ than this table lists."


def uncounted(text: str) -> str:
    """A preamble of w1's with each count it states said with none (`UNCOUNTED`)."""
    for counted, said in UNCOUNTED:
        text = text.replace(counted, said)
    return text


SYSTEMS = tuple(uncounted(system) for system in w1.SYSTEMS[:COUNTS["three"]])     # the drafts' system prompts


@dataclass(frozen=True)
class Said:
    """A cell that is no leaf's value, said as its text: `WRITTEN` and `ABSENT` on a file's own row (and `ABSENT` for a
    leaf a draft lacks), `NO_FILE` for a leaf of a file the draft lacks (that file's own row says so, and its leaves
    are not compared), a file's bytes' digest where it is compared by them."""
    text: str


ABSENT, NO_FILE, WRITTEN = Said("absent"), Said("-"), Said("written")
ITSELF, WHOLE = (Said("file"),), (Said("whole"),)   # a file's own place, and its bytes' where it is compared by
#                                                     them: no key path or table cell is a `Said`


# ------------------------------------------------------------------------------------------------- the comparison


def fields(path: Path) -> dict[tuple, object] | None:
    """A file's leaves by their places: agent/w1's `leaves` over every leaf (JSON's by key path, a table's by row and
    column, its header under `("columns",)`), a parquet file's frame as a table (pandas; {} where it does not import);
    {} for a kind with none; None where the file does not parse or is over `MAX_BYTES`."""
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
    except OSError:
        return None
    if contract.fmt_of(path.name) != "parquet":
        return w1.leaves(path, every=True)
    try:
        import pandas
    except ImportError:
        return {}
    try:
        frame = pandas.read_parquet(path)
        return w1.table_leaves(frame.to_csv(index=not isinstance(frame.index, pandas.RangeIndex)), ",", every=True)
    except Exception:  # noqa: BLE001 - a reader's failure is the file's: compared by its bytes
        return None


def digest(path: Path) -> Said:
    try:
        return Said("sha256 " + hashlib.sha256(path.read_bytes()).hexdigest()[:12])
    except OSError:
        return Said("unreadable")


def same(a: object, b: object) -> bool:
    """Two leaves match: two numbers within `RTOL` and `ATOL`, anything else equal and of one type (a flag is no
    number, a text no number)."""
    if isinstance(a, float) and isinstance(b, float):
        return w1.near(a, b, RTOL, ATOL)
    return type(a) is type(b) and a == b


@dataclass(frozen=True)
class Row:
    file: str
    where: tuple            # the leaf's place; `ITSELF` the file itself; `WHOLE` its bytes
    values: tuple           # each pool draft's value, in the pool's order
    row: str = ""           # a table's row as its first cell gives it, where the place is a table's cell

    def differs(self, a: int, b: int) -> bool:
        """Whether the pool's `a`-th and `b`-th drafts differ here (a leaf of a file one lacks is the file's row's)."""
        left, right = self.values[a], self.values[b]
        return NO_FILE not in (left, right) and not same(left, right)

    def split(self) -> bool:
        return any(self.differs(a, b) for a, b in itertools.combinations(range(len(self.values)), 2))


@dataclass(frozen=True)
class Compared:
    pool: tuple[int, ...]                       # the clean drafts compared, by preamble
    groups: tuple[tuple[int, ...], ...] = ()    # w1's groups of drafts that all match, the largest first
    rows: tuple[Row, ...] = ()                  # every field where two of them differ

    @property
    def agreed(self) -> bool:
        return bool(self.groups) and len(self.groups[0]) >= 2

    @property
    def floor(self) -> int | None:
        """The draft whose files are laid in the output: the agreed group's first, else the first clean one."""
        return self.groups[0][0] if self.groups else None

    @property
    def disagreed(self) -> bool:
        """Two or more clean drafts and no two matching: the table is said."""
        return len(self.pool) >= 2 and not self.agreed

    def verdict(self) -> w1.Verdict:
        return w1.Verdict(self.floor, self.groups[0] if self.groups else (), self.pool, self.groups)


def compared_files(outputs: dict[int, dict[str, Path]], named=()) -> list[str]:
    """The files compared: those the instruction names that any draft wrote, in its order, else every file any wrote."""
    wrote = list(dict.fromkeys(name for index in sorted(outputs) for name in outputs[index]))
    return [name for name in named if name in wrote] or wrote


def places(outputs: dict[int, dict[str, Path]], named=()) -> list[Row]:
    """Every place of the files compared, each pool draft's value there: the file's own row, then its bytes where a
    draft's copy is not read field by field, else each leaf in the order the drafts first hold it."""
    pool = sorted(outputs)
    rows = []
    for name in compared_files(outputs, named):
        has = [index for index in pool if name in outputs[index]]
        rows.append(Row(name, ITSELF, tuple(WRITTEN if index in has else ABSENT for index in pool)))
        read = {index: fields(outputs[index][name]) for index in has}
        if any(found is None for found in read.values()):
            rows.append(Row(name, WHOLE, tuple(digest(outputs[index][name]) if index in has else NO_FILE
                                               for index in pool)))
            continue
        heads = {index: found.get(("columns",)) for index, found in read.items()}
        for where in dict.fromkeys(where for index in has for where in read[index]):
            values = tuple(read[index].get(where, ABSENT) if index in has else NO_FILE for index in pool)
            rows.append(Row(name, where, values, first_cell(where, read, heads)))
    return rows


def first_cell(where: tuple, read: dict, heads: dict) -> str:
    """A table cell's row as its first cell gives it, from the first draft holding that row ("" for any other place)."""
    if len(where) != 3 or not isinstance(where[0], int):
        return ""
    for index, found in read.items():
        head = heads.get(index)
        if head and (cell := found.get((where[0], head[0], 1))) is not None:
            return shown(cell) if not isinstance(cell, str) else shown(Said(cell))
    return ""


def compare(outputs: dict[int, dict[str, Path]], named=()) -> Compared:
    """The clean drafts' files by preamble and the files the instruction names: the groups (w1's, over `places`) and
    the rows where two differ."""
    pool = tuple(sorted(outputs))
    if not pool:
        return Compared(())
    rows = places(outputs, named)
    at = {index: k for k, index in enumerate(pool)}
    groups = w1.groups(list(pool), lambda i, j: not any(row.differs(at[i], at[j]) for row in rows))
    return Compared(pool, tuple(groups), tuple(row for row in rows if row.split()))


# ------------------------------------------------------------------------------------------------------ the block


def shown(value: object) -> str:
    """A value as the table says it: a number to ten significant digits, a text quoted, a mark as itself, cut to
    `VALUE_CHARS`, kept on its line and out of the table's bars."""
    if isinstance(value, Said):
        text = value.text
    elif isinstance(value, float):
        text = format(value, ".10g")
    elif isinstance(value, tuple):
        text = ", ".join(str(item) for item in value)
    else:
        text = json.dumps(value)
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= VALUE_CHARS else text[:VALUE_CHARS - 3] + "..."


def label(row: Row) -> str:
    """A field as the table names it: the file, and the key path, the row and column, the header or the whole file."""
    name = f"`{row.file}`"
    if row.where == ITSELF:
        return f"{name}, the file"
    if row.where == WHOLE:
        return f"{name}, the whole file (not read field by field)"
    tabled = contract.fmt_of(row.file) in ("csv", "tsv", "parquet")
    if tabled and row.where == ("columns",):
        return f"{name}, its columns in order"
    if tabled and len(row.where) == 3:
        index, column, occurrence = row.where
        said = f"{name}, row `{row.row}`" if row.row else f"{name}, row {index + 1} by its first column"
        return said + f", column `{column}`" + (f" (its occurrence {occurrence})" if occurrence > 1 else "")
    path = "".join(f"[{key}]" if isinstance(key, int) else (f".{key}" if k else str(key))
                   for k, key in enumerate(row.where))
    return f"{name}, `{path}`" if path else f"{name}, its value"


def family(row: Row) -> tuple:
    """The field a row is an instance of: a table's column, a JSON key path with its list positions dropped, or the
    file's own row."""
    if contract.fmt_of(row.file) in ("csv", "tsv", "parquet") and len(row.where) == 3:
        return row.file, row.where[1], row.where[2]
    return row.file, tuple(key for key in row.where if not isinstance(key, int))


def chosen(rows: tuple[Row, ...], limit: int = ROWS) -> list[Row]:
    """The rows the table says, `limit` at most: every field's first differing row before any field's second, and so
    on (so one column's many rows never hide another field), kept in the table's order."""
    families: dict[tuple, list[int]] = {}
    for at, row in enumerate(rows):
        families.setdefault(family(row), []).append(at)
    picked: list[int] = []
    for depth in itertools.count():
        level = [ats[depth] for ats in families.values() if depth < len(ats)]
        if not level or len(picked) >= limit:
            break
        picked += level[:limit - len(picked)]
    return [rows[at] for at in sorted(picked)]


def table(compared: Compared) -> str:
    """The field table: a row a field where two clean drafts differ, each one's value under its letter; `ROWS` rows at
    most (`chosen`), `MORE` after them where more differ."""
    letters = LETTERS[:len(compared.pool)]
    said = chosen(compared.rows)
    lines = [f"| field | {' | '.join(letters)} |", "|---|" + "---|" * len(letters)]
    lines += [f"| {label(row)} | {' | '.join(shown(value) for value in row.values)} |" for row in said]
    return "\n".join(lines) + (f"\n\n{MORE}" if len(compared.rows) > len(said) else "")


def ran_text(draft: w1.Draft) -> str:
    """The script a draft ran: a cut one's longest prefix that compiles, as w1's run took it."""
    if draft.code is None:
        return ""
    if draft.code.closed:
        return draft.code.text
    return runner.parsing_prefix(runner.complete_lines(draft.code.text))


def said(compared: Compared, drafts: list[w1.Draft]) -> str:
    """The first message's block where the clean drafts disagree: each one's script under its letter, the table and
    the ask."""
    by_index = {draft.index: draft for draft in drafts}
    scripts = "\n\n".join(f"## Draft {letter}\n\n```python\n{ran_text(by_index[index]).rstrip()}\n```"
                          for letter, index in zip(LETTERS, compared.pool))
    return SAID.format(scripts=scripts, table=table(compared))


# ------------------------------------------------------------------------------------------------------ the round


def allowance(setting: str) -> int:
    """The round's requests at most under `setting`: `PER_DRAFT` a draft."""
    return PER_DRAFT * COUNTS[setting]


def steps(log):
    """The loop's step log with the round's own steps under `drafts `."""
    return lambda step, detail="": log(f"drafts {step}", detail)


class Round(w1.W1):
    """The round over one unit (above): agent/w1's `W1` with the loop's clock, each draft a client of its own bounded
    to the round, its runs `RUN_S` at most within half the time left, and the comparison for its vote."""

    def __init__(self, loop, task, pathmap, data: str, setting: str = "three") -> None:
        super().__init__(loop.task_dir, loop.out, loop.client, steps(loop.log), Path(loop.work) / WORK, loop.deadline)
        self.loop, self.unit, self.pathmap, self.data = loop, task, pathmap, data
        self.setting, self.count = setting, COUNTS[setting]
        self.named = tuple(loop.named_files)
        self.clients: dict[int, object] = {}
        self.bound = 0.0                # the seconds each draft's request may take: the round's
        self.reserve = 0.0              # the time left kept from the runs: the loop's
        self.compared = Compared(())
        self.text = ""                  # the first message's block, "" where none is said
        self.sent = 0
        self.held: int | None = None    # the draft whose files are the floor, held aside until the close
        self.closed = False             # the close read (`close`), once
        self.floor_written = False      # the held floor written at the close (or the watchdog's floor mid-round)
        self.seconds = 0.0
        self.done = False               # the floor laid (or the round given up): the watchdog leaves the output alone

    def left(self) -> float:
        return self.loop.left()

    def sender(self, draft: w1.Draft) -> tuple[object, dict]:
        """The draft's own client and its keywords: the round's bound, and under `one` the client's own temperature, so
        the one draft is N0's request as N0 sends it."""
        keywords = {"bound": max(model.MIN_CALL_S, self.bound)}
        return self.clients[draft.index], keywords if self.count > 1 else {**keywords, "temperature": None}

    def runner_in(self, name: str) -> runner.Runner:
        (self.work / name).mkdir(parents=True, exist_ok=True)
        return runner.Runner(self.work / name, kind=KIND, out=self.out)

    def cap(self, left: float, remaining: int, kept: float) -> float:
        """A draft's run: `RUN_S` at most, and its share of the time left less the loop's reserve (`w1.run_cap`)."""
        return min(RUN_S, w1.run_cap(left, remaining, self.reserve))

    def keep(self, draft: w1.Draft) -> bool:
        """A failed move loses its store too, releasing space for the following loop."""
        try:
            whole = super().keep(draft)
        except OSError:
            cleared(self.out, self.before)
            whole = False
        if not whole and draft.store is not None:
            shutil.rmtree(draft.store, ignore_errors=True)
        return whole

    def verdict(self) -> w1.Verdict:
        clean = [draft for draft in self.drafts if draft.clean]
        self.compared = compare({draft.index: w1.files_in(draft.store) for draft in clean}, self.named)
        return self.compared.verdict()

    def system_of(self, draft: w1.Draft) -> str:
        return SYSTEMS[draft.index]

    def finish(self) -> None:
        """The watchdog's, holding WRITING: while the round is open, w1's (the running script's files cleared, the floor
        so far laid, each draft's outcome); after it, the held floor as at the close (`close`)."""
        if self.done:
            self.close()
            return
        cleared(self.out, self.before)
        super().finish()
        self.floor_written = self.delivered

    def close(self) -> None:
        """At the unit's close, the caller holding WRITING, once: the held floor written to the output where it holds
        nothing the harness grades (`fallback.graded`), what it holds first removed; nothing where no floor is held."""
        if self.held is None or self.closed:
            return
        self.closed = True
        if fallback.graded(self.out, self.named):
            self.log("floor", "not written at the close: the output holds a file to grade")
            trace.event("drafts_floor", written=False, draft=self.held)
            return
        cleared(self.out, self.before)
        laid = self.deliver(self.drafts[self.held])
        self.floor_written = laid > 0
        self.log("floor", f"written at the close: draft {self.held}'s {laid} file(s), the output holding nothing to grade")
        trace.event("drafts_floor", written=self.floor_written, draft=self.held, files=laid)

    def go(self) -> str:
        """The round, the runs and the comparison, the floor held aside (`held`); the block the loop's first message
        closes on where the clean drafts disagree, else ""."""
        began = self.left()
        self.out.mkdir(parents=True, exist_ok=True)
        self.before = {entry.name for entry in self.out.iterdir()}
        since, cap = self.client.deadline, self.client.usage.requests + allowance(self.setting)
        self.clients = {index: self.client.sibling(cap) for index in range(self.count)}
        for made in self.clients.values():
            made.kind = KIND
        self.bound = max(0.0, began) * w1.ROUND_SHARE
        seconds = min(RUN_S, max(w1.MIN_RUN_S, max(0.0, began) * w1.RUN_SHARE / self.count))
        user = self.redact(n0.prompt(self.unit, self.pathmap, self.data, seconds))
        self.sent = self.count
        self.round(user)
        if moved := self.client.carry(self.clients.values(), since):
            self.log("clock", f"{moved:.0f} s of the drafts' untimed waits carried onto the unit's clock")
        scripts = [draft for draft in self.drafts if draft.code is not None]
        self.reserve = max(0.0, self.left()) * (1 - w1.RUN_SHARE)
        for number, draft in enumerate(scripts):
            if self.ended.is_set():
                return ""
            self.run(draft, len(scripts) - number, self.reserve)
        verdict = self.verdict()
        with guard.WRITING:
            if self.ended.is_set():
                return ""
            self.done, self.held = True, verdict.winner
        self.outcomes(verdict)
        self.seconds = max(0.0, began - self.left())
        self.text = said(self.compared, self.drafts) if self.compared.disagreed else ""
        self.log("floor", "none: no draft is clean" if verdict.winner is None else
                 f"held aside: draft {verdict.winner}'s {len(w1.files_in(self.drafts[verdict.winner].store))} file(s), "
                 + ("agreed" if self.compared.agreed else "the first clean" if len(self.compared.pool) > 1 else
                    "the one clean"))
        trace.event("drafts", **self.keys(), groups=[list(group) for group in self.compared.groups],
                    fields=[label(row) for row in chosen(self.compared.rows)])
        return self.text

    def given_up(self, why: str) -> None:
        """The round failed in its own code: what it left in the output removed and no floor held, so the loop starts
        and ends as today."""
        with guard.WRITING:
            if not self.ended.is_set():
                cleared(self.out, self.before)
            self.done, self.held = True, None
        self.text = ""
        self.log("given up", why)

    def keys(self) -> dict:
        """The summary line's keys but the setting (`KEYS[1:]`)."""
        return {"drafts_sent": self.sent, "drafts_clean": sum(draft.clean for draft in self.drafts),
                "floor_written": int(self.floor_written), "drafts_agreed": int(self.compared.agreed),
                "table_said": int(bool(self.text)),
                "table_fields": len(chosen(self.compared.rows)) if self.text else 0, "drafts_s": round(self.seconds, 1)}


def cleared(out: Path, keep=frozenset()) -> None:
    """Every entry of the output removed but those named in `keep`, a link as itself, never followed."""
    for entry in sorted(Path(out).iterdir()):
        if entry.name in keep:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
