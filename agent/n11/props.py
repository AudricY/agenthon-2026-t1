"""The instruction's own properties of its outputs, read at run time (lane n11e), and the output checked against them.

E039's anatomy and the intermittent read (docs/research/e039-anatomy.md, docs/research/intermittent.md): most failing
runs exit 0 with values that break what the instruction itself states in words ("should be > 1", "a positive
magnitude", "starts at 1.0", "rounded to 2 decimals") or miss keys and columns it lists; the sanity checks
(agent/n11/sanity.py) read only what any unit of a kind agrees on. This module reads the unit's own instruction with
one rule set for every unit (rule 8: patterns over quantity names and comparatives, no unit's vocabulary).

- `stated(instruction, named)` is what the instruction states (`Stated`):
  - **properties** (`Prop`): a statement (a bullet, a table row, a sentence) binding a named quantity (a key or column
    name in backticks or bold, a bare snake_case name, a definition's head, the name a parenthesis follows, or a
    phrase whose words an output name's words meet) to a bound (`> 1`, `at least 0`, `positive`, `non-negative`,
    `negative`), an interval (`between 0 and 1`, `in [0, 1]`), an equality (`must be 18`), a sum (`sums to 1`), a
    first value (`starts at 1.0`), a direction (`decreasing`) or a rounding (`rounded to 6 decimals`; every numeric
    value of a file or of the JSON outputs). Never one hedged (`typically`, `~`, `may`, `e.g.`), conditional or
    filtering (`if`, `when`, `with`, `where`, `fraction of`, `counts`), in an imperative step (`Use`, `Set`, `Clip`),
    restricted to a subset (`for calls`, `except`), in a unit (`%`, `years`), against an expression (`> 2 * p / n`, a
    code span), or said in a definition of another name: what is extracted holds of every value the name holds.
  - **the contract**: each output file's keys (a JSON block under the file's heading or after a line naming it) and
    columns (a header line in a block, a prose list right after `columns:` or `keys:`, a table of columns under the
    file's heading).
  The file a statement is about is the output file (`named`) its heading, its lead or its own text names.
- `check(stated, out)` reads the output directory and returns the findings, one line each: `instruction:` a value the
  name holds that contradicts a property (the name found as a key, a key path's last part or a column, in the
  statement's file first), quoting the property's words; `contract:` a named key or column absent from its file (a
  file that is absent is the checks'). A name the output does not hold is never a finding.
- `audit(stated, out, named)` is the review's reading (lane loop1): every property `held` with what the output holds
  of it, `broken` with its finding's line, or `unchecked` where no output name meets it; then each named output file,
  key and column, present or absent.
- Under `stated(..., rules=True)` (lane rules1, `AGENTHON_N11_RULES=on`; docs/research/wrong-1013.md, section 5) four
  more kinds (`RULE_KINDS`), each read where today's parse takes none: `fraction` (a name the task defines as a
  fraction, share or proportion of something, `x = fraction of ...`, holding a value outside [0, 1]), `header` (a
  backticked comma-separated header the task gives under an output table's heading, a column of it absent), `rowsum`
  (the columns stated to sum to a figure for every row, a row off past its written rounding) and `formula` (a formula
  `a = expr` among an output table's own columns, a row off past the written rounding); and the example output's
  figures (`Stated.examples`: a JSON example block's numbers the task writes nowhere else), which agent/n11/commands.py
  reads against the scripts' literals. Each finding is a broken rule with the value and the task's words, never a
  question or a count.
- Under `stated(..., rules=True, wide=True)` (lane rules2, `AGENTHON_N11_RULES=wide`; docs/research/wrong-1014.md,
  section 3) the instruction is read for six more rules too (`Stated.wide`, `wide_stated`), checked by `wide_found` over
  the `Outputs` the check reads: `stated_rows` (a row count the instruction states for an output table, "= N rows",
  "(N rows", "exactly N rows", "has N rows", and another number of data rows), `abs_bound` (`|x| < b` or `|x| <= b`
  stated, a value under `x` breaking it), `key_spelling` (a name the instruction writes absent, and a name equal to it
  up to case and separators that it never writes present), and `group_count` (an explicitly stated unfiltered row
  count for a table's group), and two over the output alone: `twin_equal` (a JSON object's with/without twins every
  one at one value), `unnamed_index` (a table's column with no name or pandas'
  `Unnamed: N`). Each finding names the file and the name, never a number of the output's or the task's.

Pure: the standard library alone (pyarrow where a parquet file is read), nothing written, nothing named.
"""
from __future__ import annotations

import ast
import csv
import fnmatch
import io
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024     # an output file larger than this is not read
MAX_FILES = 64                  # output files read
MAX_ROWS = 200_000              # a table's rows read
MAX_ITEMS = 5_000               # a JSON list's items read
FINDING_CHARS = 220             # a finding's line, at most
QUOTE_CHARS = 70                # the property's words a finding quotes, at most
PER_PROP = 2                    # findings one property makes, at most
TOLERANCE = 1e-9                # a non-strict bound's and an equality's relative slack
SUM_TOLERANCE = 1e-6            # a sum's
PHRASE_WORDS = 4                # a phrase subject's words, at most
TABLES = {".csv": ",", ".tsv": "\t"}
PARQUETS = (".parquet", ".pqt")
OUTPUT_FILE = re.compile(r"([A-Za-z0-9_][\w.\-]*\.(?:json|csv|tsv|parquet|pqt|txt|html|xlsx))\b", re.I)

NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
WORDNUM = NUM + r"|\b(?:" + "|".join(NUMBER_WORDS) + r")\b"
GOES_ON = re.compile(r"(?:[A-Za-z_(\[]|\s*(?:[*/×·^%]|[-+]\s*[A-Za-z(])|\s+(?!(?:for|and|or|in|on|at|by|when|if|where|"
                     r"unless|except|per|with|as|across|over|under|to|from|after|before|during|within|is|are|which|"
                     r"that|so|then|but|while|because|since|inclusive|exclusive|everywhere|always)\b)[A-Za-z])", re.I)
#                             a number an operand, a unit or a counted noun goes on from is no bound
IDENT = r"[A-Za-z_][A-Za-z0-9_]*\*?"
TYPES = frozenset({"true", "false", "null", "none", "int", "float", "str", "bool", "list", "dict", "float64",
                   "int64", "object", "string", "number", "integer", "boolean", "array", "nan", "inf"})
TICKED = re.compile(r"`(" + IDENT + r")`")
BOLD = re.compile(r"\*\*`?(" + IDENT + r")`?\*\*")
BARE = re.compile(r"(?<![\w`.$\\/])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\*?)(?![\w`(/])")

HEDGES = re.compile(r"\b(?:typically|usually|generally|often|approximately|approx|roughly|about|around|nearly|"
                    r"almost|ideally|likely|may|might|can|could|e\.g|for example|for instance|close to|tends? to|"
                    r"suggests?|expected to be close)\b|[~≈]", re.I)
CONDITIONS = re.compile(r"\b(?:if|iff|when|whenever|where|unless|whether|otherwise|else|with|without|whose|which|"
                        r"having|only|count|counts|counted|counting|number of|fraction|share|proportion|percentage|"
                        r"percent|means|indicates|implies|so that|such that|then|after|before|while|until|unlike|"
                        r"instead|rather|vs|versus|relative to|compared|true|false|same|than the)\b", re.I)
EXCEPT = re.compile(r"\b(?:except|excluding|other than|apart from|but not|unless)\b", re.I)
IMPERATIVE = frozenset({
    "use", "set", "compute", "calculate", "apply", "fit", "run", "call", "initialize", "initialise", "choose",
    "select", "draw", "sample", "simulate", "estimate", "solve", "minimize", "minimise", "maximize", "maximise", "find",
    "build", "construct", "load", "read", "drop", "filter", "clip", "clamp", "flag", "classify", "assign", "normalize",
    "normalise", "renormalize", "renormalise", "scale", "multiply", "divide", "replace", "keep", "exclude", "remove",
    "skip", "sort", "convert", "take", "add", "subtract", "define", "fix", "flip", "mark", "label", "treat", "record",
    "generate", "create", "plot", "identify", "detect", "discard", "impose", "constrain", "bound", "cap", "floor",
    "shift", "adjust", "rescale", "repeat", "iterate", "loop", "start", "begin", "stop", "compare", "match", "map",
    "store", "save", "write", "report", "return", "print", "output", "aggregate", "group", "merge", "join", "align",
    "resample", "interpolate", "extrapolate", "discount", "annualize", "annualise", "winsorize", "winsorise", "trim",
    "split", "partition", "rank", "pick", "accept", "reject", "enforce", "impute", "fill", "ffill", "bfill", "parse",
    "extract", "infer", "derive", "invert", "transform", "rotate", "project", "regress", "smooth", "bootstrap"})
TRAILING = re.compile(r"^\s*(?:for|when|if|where|unless|except|per|on|at|with|as|across|between|relative|compared|"
                      r"than|in|by|of|from|after|before|during|within|whenever|once|provided|given|times|x|"
                      r"multiplied|divided|plus|minus|and|or|to|under|over|beyond|since|until)\b", re.I)
EVERY = re.compile(r"^\s*(?:for\s+(?:all|both|every|each|any)\b|everywhere\b|in\s+(?:all|every|each)\b|"
                   r"in\s+absolute\s+(?:value|terms)\b|in\s+magnitude\b|by\s+construction\b|at\s+all\s+times\b)", re.I)
ABSOLUTE = re.compile(r"^\s*(?:in\s+absolute\s+(?:value|terms)|in\s+magnitude)\b", re.I)
LINK = re.compile(r"\b(?:is|are|be|been|remain|remains|stay|stays|lie|lies|equal|equals|reported\s+as|report\s+as|"
                  r"reported|expressed\s+as|given\s+as|kept)\b", re.I)
MODAL = re.compile(r"\b(?:should|must|shall|will|always|never|cannot)\b", re.I)
GLUE = re.compile(r"(?:\s|\b(?:is|are|be|been|should|must|shall|will|always|never|all|each|every|both|strictly|"
                  r"weakly|monotonically|a|an|the|reported|report|as|expressed|given|kept|stated|remain|remains|"
                  r"stay|stays|lie|lies|verify|ensure|check|that|note|also|here|so)\b|[:=])*", re.I)
SIGN_NOUN = r"(?:\s+(?:number|numbers|value|values|magnitude|magnitudes|quantity|quantities|float|floats|integer|" \
            r"integers|real|scalar|amount|amounts|definite))?"

KINDS_SAID = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "="}


@dataclass(frozen=True)
class Prop:
    """One property: the quantity's `names` (identifiers as written; a phrase as its words joined by spaces, `phrase`
    set), its `kind` and bounds, the output `file` the statement is about ("" where none), the property's own words
    (`said`) and the statement (`sentence`)."""
    names: tuple[str, ...]
    kind: str               # gt ge lt le eq between sum start increasing decreasing decimals
    value: float | None = None
    upper: float | None = None
    strict: bool = False    # an open interval's upper end, a strict direction; for `decimals`, fixed-point (exactly N)
    lower_open: bool = False
    phrase: bool = False
    absolute: bool = False  # the bound is on the value's magnitude
    percent: bool = False   # a sum stated in percent: c or c/100
    scope: str = ""         # `decimals` with no name: "json", "table" or "all" numeric values (of `file` where set)
    file: str = ""
    said: str = ""
    sentence: str = ""
    expr: str = ""          # `formula`: its right side as written, read against a table's columns at the check

    def describe(self) -> str:
        """The property in a short line (the offline read's and the tests')."""
        who = (f"[{self.scope}{' of ' + self.file if self.file else ''}]" if self.scope else
               ("~" if self.phrase else "") + "/".join(self.names) + (f" ({self.file})" if self.file else ""))
        if self.kind == "header":
            return f"{self.file}: the columns {', '.join(self.names)}"
        if self.kind == "rowsum":
            columns = "columns " + ", ".join(self.names) if self.names else "columns"
            figure = f"{self.value:g}{'%' if self.percent else ''}"
            return f"{self.file or '[a table]'}: each row's {columns} sum to {figure}"
        if self.kind == "formula":
            return f"{who}: = {self.expr}"
        if self.kind == "fraction":
            return f"{who}: a fraction, in [0, 1]"
        if self.kind == "between":
            what = (f"in {'(' if self.lower_open else '['}{self.value:g}, {self.upper:g}"
                    f"{')' if self.strict else ']'}")
        elif self.kind == "sum":
            what = f"sums to {self.value:g}{'%' if self.percent else ''}"
        elif self.kind == "start":
            what = f"starts at {self.value:g}"
        elif self.kind == "decimals":
            what = f"{'exactly' if self.strict else 'at most'} {self.value:g} decimals"
        elif self.kind in ("increasing", "decreasing"):
            what = f"{'strictly ' if self.strict else ''}{self.kind}"
        else:
            what = f"{'|x| ' if self.absolute else ''}{KINDS_SAID[self.kind]} {self.value:g}"
        return f"{who}: {what}"


@dataclass
class Stated:
    """What the instruction states of its outputs: the properties, each JSON file's keys (a nested key as
    `parent.key`; a list of records' keys as `[].key`) and each table's columns."""
    props: list[Prop] = field(default_factory=list)
    keys: dict[str, list[str]] = field(default_factory=dict)
    columns: dict[str, list[str]] = field(default_factory=dict)
    numbers: frozenset = frozenset()    # every number the instruction writes (`violated`'s exemption)
    examples: tuple[str, ...] = ()      # lane rules1: the example output's figures, as written (`examples`)
    wide: "Wide | None" = None          # lane rules2: what the six rules read of the instruction (`wide_stated`)

    @property
    def contract(self) -> int:
        return sum(len(v) for v in self.keys.values()) + sum(len(v) for v in self.columns.values())


# ------------------------------------------------------------------------------------------------- the text


def normal(text: str) -> str:
    """A statement's text as the rules read it: bold markers, math markers and LaTeX or Unicode comparators to plain
    ASCII. A code span keeps its backticks: a name in it is a name, and an expression in it (`qty_x > 0`) holds
    none (`BARE` takes no name a backtick opens), so its operators find no subject."""
    text = (text.replace("≥", ">=").replace("≤", "<=").replace("−", "-").replace("∈", " in ").replace("⩾", ">=")
            .replace("⩽", "<=").replace("\u00a0", " ").replace("’", "'").replace("**", ""))
    text = re.sub(r"\\(?:geq?|geqslant)\b", ">=", text)
    text = re.sub(r"\\(?:leq?|leqslant)\b", "<=", text)
    text = re.sub(r"\\gt\b", ">", text)
    text = re.sub(r"\\lt\b", "<", text)
    text = re.sub(r"\\in\b", " in ", text)
    return re.sub(r"[ \t]+", " ", text.replace("$", "")).strip()


def quoted(text: str, end: int) -> str:
    """The words of `text` that end at `end`, whole, within `QUOTE_CHARS`: the property's own words for a finding."""
    words = text[:end].split()
    kept: list[str] = []
    while words and len(" ".join([words[-1], *kept])) <= QUOTE_CHARS:
        kept.insert(0, words.pop())
    return " ".join(kept)


def number(text: str) -> float | None:
    text = text.strip().lower()
    if text in NUMBER_WORDS:
        return float(NUMBER_WORDS[text])
    try:
        return float(text)
    except ValueError:
        return None


def identifiers(text: str) -> list[tuple[int, int, str]]:
    """The names a text writes: in backticks, in bold, or bare with an underscore; (start, end, name) in order, a
    type's name (`float`, `true`) left out."""
    found: dict[int, tuple[int, int, str]] = {}
    for pattern in (TICKED, BOLD, BARE):
        for match in pattern.finditer(text):
            name = match.group(1)
            if name.lower().rstrip("*") in TYPES:
                continue
            if any(start <= match.start() < end for start, end, _ in found.values()):
                continue
            found[match.start()] = (match.start(), match.end(), name)
    return [found[k] for k in sorted(found)]


def files_in(text: str, named: tuple[str, ...]) -> list[str]:
    """The output files a text names (each as `named` writes it, matched by its base name; any file where `named` is
    empty)."""
    by_base = {name.rsplit("/", 1)[-1].lower(): name for name in named}
    found: list[str] = []
    for match in OUTPUT_FILE.finditer(text):
        name = by_base.get(match.group(1).lower()) if named else match.group(1)
        if name and name not in found:
            found.append(name)
    return found


@dataclass
class Statement:
    text: str
    file: str                       # the output file of its section, "" where none
    heads: tuple[str, ...] = ()     # a definition's head names (a bullet's or a table row's leading names)
    table: bool = False
    cells: tuple[str, ...] = ()


@dataclass
class Block:
    lines: list[str]
    file: str
    lead: str                       # the text just before it


HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
BULLET = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")
HEAD = re.compile(r"^(?:\*\*)?`?(?P<name>" + IDENT + r"(?:`?\s*,\s*`?" + IDENT + r")*)`?(?:\*\*)?"
                  r"(?:\s*\([^()]*\))?\s*(?::|—|–|-{1,2}(?=\s))\s*(?P<body>.*)$")
LABELLED = re.compile(r"^(?:\*\*)?[^:()`]{1,48}?(?:\*\*)?\s*\(`?(?P<name>" + IDENT + r")`?\)\s*(?::|—|–)\s*"
                      r"(?P<body>.*)$")


def heads(text: str) -> tuple[tuple[str, ...], str]:
    """A definition's head names (`name:`, `**name**:`, `` `a`, `b`: ``, `name — ...`, `Label (name):`) and its body;
    ((), the text) where it opens with none."""
    stripped = text.strip()
    for pattern in (LABELLED, HEAD):
        found = pattern.match(stripped)
        if found:
            names = tuple(name.strip(" `*") for name in found.group("name").split(","))
            if all(names) and not any(name.lower() in TYPES for name in names):
                return names, found.group("body")
    return (), stripped


def segment(instruction: str, named: tuple[str, ...]) -> tuple[list[Statement], list[Block]]:
    """The instruction as statements (a bullet with its continuation lines, a table row, a paragraph, a paragraph's
    line that opens with a definition's head) and code blocks (fenced, or indented after a blank line), each with the
    output file its section is about: a heading's, else a line's that names one file, until the next heading."""
    lines = instruction.replace("\r\n", "\n").split("\n")
    statements: list[Statement] = []
    blocks: list[Block] = []
    levels: list[tuple[int, str]] = []
    current: list[str] = []
    kind, lead, local = "", "", ""

    def section() -> str:
        return local or next((file for _, file in reversed(levels) if file), "")

    def flush() -> None:
        nonlocal current, kind, lead, local
        text = " ".join(part.strip() for part in current if part.strip())
        if text:
            files = files_in(text, named)
            statements.append(Statement(text, files[0] if len(files) == 1 else section()))
            lead = text
            if kind != "bullet" and len(files) == 1:
                local = files[0]
        current, kind = [], ""

    k = 0
    while k < len(lines):
        line = lines[k]
        if FENCE.match(line):
            flush()
            fence, body = FENCE.match(line).group(1), []
            k += 1
            while k < len(lines) and not lines[k].strip().startswith(fence):
                body.append(lines[k])
                k += 1
            blocks.append(Block(body, section(), lead))
            k += 1
            continue
        heading = HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            levels = [(lv, file) for lv, file in levels if lv < level]
            files = files_in(heading.group(2), named)
            levels.append((level, files[0] if files else ""))
            local, lead = "", heading.group(2)
            k += 1
            continue
        if not line.strip():
            flush()
            k += 1
            if k < len(lines) and re.match(r"^(?: {4}|\t)\S", lines[k]) and not BULLET.match(lines[k]):
                body = []
                while k < len(lines) and (re.match(r"^(?: {4}|\t)", lines[k]) or not lines[k].strip()):
                    if lines[k].strip():
                        body.append(lines[k].strip())
                    k += 1
                blocks.append(Block(body, section(), lead))
            continue
        if line.lstrip().startswith("|"):
            flush()
            cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
            if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
                statements.append(Statement(line.strip(), section(), table=True, cells=cells))
            k += 1
            continue
        bullet = BULLET.match(line)
        if bullet:
            flush()
            current, kind = [bullet.group(2)], "bullet"
        elif kind == "bullet" or (current and not heads(line)[0]):
            current.append(line)
        else:
            flush()
            current, kind = [line], "paragraph"
        k += 1
    flush()
    return statements, blocks


SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*(\[\"<])")
PROTECT = re.compile(r"\b(?:e\.g|i\.e|vs|etc|approx|cf|eq|fig|no)\.", re.I)


def sentences(text: str) -> list[str]:
    guarded = PROTECT.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    guarded = re.sub(r"(\d)\.(\d)", "\\1\x00\\2", guarded)
    return [part.replace("\x00", ".") for part in SENTENCE.split(guarded) if part.strip()]


INTERVAL = re.compile(r"(?P<open>[\[(])\s*(?P<a>" + NUM + r")\s*,\s*(?P<b>" + NUM + r")\s*(?P<close>[\])])"
                      r"(?!\s*\^)")


def owner(outer: str) -> str:
    """The name a parenthesis that follows `outer` is about: the name it follows directly, else the name its clause
    opens with; "" where neither."""
    clause = re.split(r"[;,]", outer)[-1]
    label, colon, rest = clause.partition(":")
    if colon and not identifiers(label):
        clause = rest
    named = identifiers(clause)
    if named and not clause[named[-1][1]:].strip(" `*:"):
        return named[-1][2]
    if named and not GLUE.sub("", clause[:named[0][0]]).strip(" `*"):
        return named[0][2]
    return ""


LISTED_NAME = re.compile(r"\s*(?:`" + IDENT + r"`|\*\*`?" + IDENT + r"`?\*\*)\s*(?:,\s*(?:`" + IDENT + r"`|\*\*`?"
                         + IDENT + r"`?\*\*)\s*)*")


def clauses(text: str) -> list[tuple[str, str]]:
    """A sentence's clauses, each with the name a parenthesis it came from follows (`owner`, "" for the others):
    split at semicolons, commas and dashes outside brackets and code spans; a parenthesis opened after a space is
    its own clause, its text dropped from the clause around it (a call's `f(x)` kept); an interval `[a, b]` or
    `(a, b)` written after `in`, `between`, `within` or `on` kept whole."""
    intervals: list[str] = []

    def hold(match: re.Match) -> str:
        if not text[:match.start()].rstrip().lower().endswith(("in", "within", "between", "on", "inside")):
            return match.group(0)
        intervals.append(match.group(0))
        return f"\x01{len(intervals) - 1}\x01"

    held = INTERVAL.sub(hold, text)
    outer, inner, depth, ticked, k = [], [], 0, False, 0
    while k < len(held):
        char = held[k]
        if char == "`":
            ticked = not ticked
        elif not ticked and char == "[":
            depth += 1
        elif not ticked and char == "]":
            depth = max(0, depth - 1)
        elif not ticked and depth == 0 and char == "(" and (k == 0 or held[k - 1] in " \t"):
            level, close = 0, -1
            for at in range(k, len(held)):
                level += {"(": 1, ")": -1}.get(held[at], 0)
                if level == 0:
                    close = at
                    break
            if close > k:
                inner.append((held[k + 1:close], owner("".join(outer))))
                outer.append(" ")
                k = close + 1
                continue
        outer.append(char)
        k += 1

    def split(part: str) -> list[str]:
        pieces, start, depth_, ticked_ = [], 0, 0, False
        for at, char in enumerate(part):
            if char == "`":
                ticked_ = not ticked_
            elif not ticked_ and char in "[(":
                depth_ += 1
            elif not ticked_ and char in "])":
                depth_ = max(0, depth_ - 1)
            elif not ticked_ and depth_ == 0 and (char in ";," or part.startswith((" — ", " – ", " -- "), at)):
                if char == "," and LISTED_NAME.fullmatch(part[start:at]):
                    continue                    # `a`, `b` and `c` ...: a list of names goes on
                pieces.append(part[start:at])
                start = at + 1
        return [*pieces, part[start:]]

    found = [(piece, "") for piece in split("".join(outer))]
    found += [(piece, name) for content, name in inner for piece in split(content)]
    restored = []
    for piece, name in found:
        piece = re.sub(r"\x01(\d+)\x01", lambda m: intervals[int(m.group(1))], piece).strip(" .:—–-")
        if piece:
            restored.append((piece, name))
    return restored


# -------------------------------------------------------------------------------------------- the properties


@dataclass(frozen=True)
class Match:
    kind: str
    value: float | None
    upper: float | None
    strict: bool
    lower_open: bool
    start: int
    end: int
    percent: bool = False
    verbal: bool = False        # the phrase carries its own predicate (sums to, starts at): no link needed


PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?P<strict>strictly\s+)?\bbetween\s+(?P<a>" + NUM + r")\s+and\s+(?P<b>" + NUM + r")", re.I),
     "between"),
    (re.compile(r"\b(?:in|within|inside|on)\s+" + INTERVAL.pattern, re.I), "interval"),
    (re.compile(r"\b(?:sums?|summing|add(?:s)?\s+up)\s+to\s+(?:exactly\s+)?(?P<num>" + WORDNUM +
                r")(?P<pct>\s*%|\s+percent)?", re.I), "sum"),
    (re.compile(r"\b(?:starts?|begins?)\s+(?:at|with|from)\s+(?P<num>" + NUM + r")", re.I), "start"),
    (re.compile(r"(?P<mode>\bmonotonically\s+|\bweakly\s+|\bstrictly\s+)?\b(?P<dir>non-?decreasing|non-?increasing|"
                r"increasing|decreasing)\b", re.I), "direction"),
    (re.compile(r"(?P<strict>\bstrictly\s+)?\b(?P<w>greater|larger|higher|bigger|more|less|smaller|lower|fewer)\s+"
                r"than(?P<eq>\s+or\s+equal\s+to)?\s+(?P<num>" + NUM + r")", re.I), "than"),
    (re.compile(r"\b(?P<w>at\s+least|at\s+most|no\s+(?:more|greater|larger|higher)\s+than|no\s+(?:less|fewer|"
                r"smaller|lower)\s+than|exceeds?|above|below)\s+(?P<num>" + NUM + r")", re.I), "word"),
    (re.compile(r"(?P<op>>=|<=|=>|=<|>|<)\s*(?P<num>" + NUM + r")"), "op"),
    (re.compile(r"(?P<neg>\b(?:not|never)\s+)?(?P<strict>\bstrictly\s+)?\b(?P<sign>non-?negative|non-?positive|"
                r"positive|negative)\b(?=" + SIGN_NOUN + r"\s*(?:$|[.;:,]|\s(?:for\s+(?:all|both|every|each)\b|"
                r"everywhere\b|in\s+all\b|at\s+all\s+times\b|by\s+construction\b)))", re.I), "sign"),
    (re.compile(r"\b(?:should|must|shall|will|always)\s+(?:always\s+)?(?:be|equal|equals)\s+(?:exactly\s+)?"
                r"(?P<num>" + NUM + r")(?=\s*(?:$|[.;,]))", re.I), "eq"),
]


def matches(clause: str) -> list[Match]:
    """The property phrases in a clause, the first pattern's of overlapping ones kept."""
    found: list[Match] = []
    for pattern, family in PATTERNS:
        for m in pattern.finditer(clause):
            if any(not (m.end() <= f.start or m.start() >= f.end) for f in found):
                continue
            made = build(family, m, clause)
            if made is not None:
                found.append(made)
    return sorted(found, key=lambda f: f.start)


def build(family: str, m: re.Match, clause: str) -> Match | None:
    """A pattern's match as a property; None where its number goes on into an expression or a unit, or its bounds
    make no interval."""
    groups = m.groupdict()
    after = clause[m.end():]
    if family == "sum":
        value = number(groups["num"])
        if value is None or (not groups.get("pct") and GOES_ON.match(after)):
            return None
        return Match("sum", value, None, False, False, m.start(), m.end(), bool(groups.get("pct")), True)
    if groups.get("num") is not None and GOES_ON.match(after):
        return None
    if family in ("between", "interval"):
        a, b = number(groups["a"]), number(groups["b"])
        if a is None or b is None or a >= b or GOES_ON.match(after):
            return None
        if family == "between":
            return Match("between", a, b, bool(groups.get("strict")), bool(groups.get("strict")), m.start(), m.end())
        return Match("between", a, b, groups["close"] == ")", groups["open"] == "(", m.start(), m.end())
    if family == "start":
        return Match("start", number(groups["num"]), None, False, False, m.start(), m.end(), verbal=True)
    if family == "direction":
        word = groups["dir"].lower().replace("-", "")
        if word.startswith("non"):
            return Match("increasing" if word == "nondecreasing" else "decreasing", None, None, False, False,
                         m.start(), m.end())
        return Match(word, None, None, (groups.get("mode") or "").strip().lower() == "strictly", False, m.start(),
                     m.end())
    if family == "than":
        bigger = groups["w"].lower() in ("greater", "larger", "higher", "bigger", "more")
        kind = ("ge" if bigger else "le") if groups.get("eq") else ("gt" if bigger else "lt")
        return Match(kind, number(groups["num"]), None, False, False, m.start(), m.end())
    if family == "word":
        word = re.sub(r"\s+", " ", groups["w"].lower())
        kind = {"at least": "ge", "at most": "le", "above": "gt", "below": "lt", "exceed": "gt", "exceeds": "gt"}.get(
            word, "le" if any(w in word for w in ("more", "greater", "larger", "higher")) else "ge")
        return Match(kind, number(groups["num"]), None, False, False, m.start(), m.end())
    if family == "op":
        kind = {">": "gt", ">=": "ge", "=>": "ge", "<": "lt", "<=": "le", "=<": "le"}[groups["op"]]
        return Match(kind, number(groups["num"]), None, False, False, m.start(), m.end())
    if family == "sign":
        sign = groups["sign"].lower().replace("-", "")
        strict = bool(groups.get("strict"))
        if re.search(r"\b(?:most|least|more|less|no|non\s*zero|any|some)\s*$", clause[:m.start()], re.I):
            return None
        if groups.get("neg"):
            sign = {"negative": "nonnegative", "positive": "nonpositive"}.get(sign, "")
            strict = False
        kind = {"positive": "gt" if strict else "ge", "nonnegative": "ge", "negative": "lt" if strict else "le",
                "nonpositive": "le"}.get(sign)
        return Match(kind, 0.0, None, False, False, m.start(), m.end()) if kind else None
    if family == "eq":
        return Match("eq", number(groups["num"]), None, False, False, m.start(), m.end())
    return None


PRONOUNS = frozenset({"it", "its", "they", "them", "these", "those", "this", "which", "we", "you", "he", "she",
                      "one", "has", "have", "had", "holds", "contains", "there", "here"})
STOP = frozenset({"the", "a", "an", "all", "each", "every", "its", "their", "this", "that", "these", "those", "both",
                  "series", "column", "columns", "value", "values", "field", "fields", "entry", "entries", "final",
                  "s", "verify", "ensure", "check", "note", "also"})


def subject_phrase(text: str) -> list[str]:
    """The content words a clause names its subject by, before its predicate, in lower case; [] where it holds a
    symbol, a number or a word of a condition, or more than `PHRASE_WORDS` words."""
    text = re.sub(r"\b(?:should|must|shall|will|always|is|are|be|remain|remains|lie|lies|stays?|reported|"
                  r"expressed)\b.*$", "", text, flags=re.I)
    text = re.sub(r"'s\b", "", text).replace("-", " ")
    tokens = re.findall(r"\S+", text)
    if not tokens or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", token) for token in tokens):
        return []
    if any(token.lower() in PRONOUNS for token in tokens):
        return []
    words = [token.lower() for token in tokens if token.lower() not in STOP]
    if not words or len(words) > PHRASE_WORDS or CONDITIONS.search(" ".join(words)):
        return []
    return words


def subject(before: str, m: Match, owner_: str, heads_: tuple[str, ...]) -> tuple[list[str], bool]:
    """The names a property is said of: the clause's names before it where only a link stands between them and it
    (`a`, `b` and `c` must ...); else, where the clause is all property, the name its parenthesis follows or the
    statement's head; else a phrase (two words or more, or one with a link verb); ([], False) where none holds."""
    named = identifiers(before)
    linked = bool(LINK.search(before)) or bool(MODAL.search(before))
    if named:
        last = named[-1]
        gap = before[last[1]:].strip(" `*")
        if gap and (not (linked or m.verbal) or len(gap.split()) > 5 or GLUE.sub("", gap).strip(" `*:")):
            return [], False
        k = len(named) - 1
        while k > 0 and re.fullmatch(r"[`*\s]*(?:,|and|&|or|,\s*and)?[`*\s]*", before[named[k - 1][1]:named[k][0]]):
            k -= 1
        lead = before[:named[k][0]]
        if GLUE.sub("", lead).strip(" `*"):
            return [], False
        return [name for _, _, name in named[k:]], False
    stripped = before.strip(" `*")
    if not GLUE.sub("", stripped).strip(" `*:"):
        if owner_:
            return [owner_], False
        return list(heads_), False
    words = subject_phrase(stripped)
    if not words or (len(words) < 2 and not LINK.search(before)):
        return [], False
    if not (linked or m.verbal or m.kind in ("gt", "ge", "lt", "le", "between")):
        return [], False
    return [" ".join(words)], True


def properties(clause: str, owner_: str, heads_: tuple[str, ...], file: str, sentence: str) -> list[Prop]:
    """A clause's properties, each bound to its subject (above, which takes none past a word that is no link: a
    condition, a filter); none where a hedge or a qualifier after it makes it a property of some values only."""
    found: list[Prop] = []
    if HEDGES.search(clause):
        return found
    for m in matches(clause):
        before, after = clause[:m.start], clause[m.end:]
        if TRAILING.match(after) and not EVERY.match(after) and m.kind != "start" and not (
                m.kind in ("increasing", "decreasing") and re.match(r"^\s*over\s+time\b", after, re.I)):
            continue
        names, phrase = subject(before, m, owner_, heads_)
        if not names:
            continue
        said = quoted(clause, m.end)
        found.append(Prop(tuple(names), m.kind, m.value, m.upper, m.strict, m.lower_open, phrase,
                          bool(ABSOLUTE.match(after)) and m.kind in ("gt", "ge", "lt", "le"), m.percent, "", file,
                          said, sentence))
    return found


DECIMALS = re.compile(r"(?P<lead>\b(?:at\s+least|minimum(?:\s+of)?|no\s+fewer\s+than|up\s+to)\s+)?"
                      r"(?P<exact>\b(?:exactly|fixed(?:[- ]point)?(?:\s+with)?(?:\s+exactly)?)\s+)?"
                      r"\b(?P<n>\d+|" + "|".join(NUMBER_WORDS) + r")\s+(?:decimal(?:s|\s+places?|\s+digits?)?\b|"
                      r"d\.p\.)(?P<precision>\s+of\s+precision)?", re.I)
SCOPE_NOUNS = r"(?P<noun>values?|numbers?|floats?|fields?|columns?|outputs?|entries|cells?)\b"
QUANTIFIED = re.compile(r"\b(?:(?P<all>all|every|each|any)|for)\s+(?P<words>(?:[\w-]+\s+){0,3}?)" + SCOPE_NOUNS, re.I)
LEADING = re.compile(r"^\s*(?:the\s+)?(?P<words>(?:[\w-]+\s+){0,2}?)" + SCOPE_NOUNS, re.I)
SCOPE_WORDS = frozenset({"numeric", "numerical", "float", "floating", "floating-point", "json", "csv", "the",
                         "other", "output", "reported", "decimal", "number", "real", "all", "every", "each", "any"})
ROUNDING = re.compile(r"\b(?:round(?:ed)?|use|uses|report(?:ed)?|written|write|given|formatted|kept|stored|with|to|"
                      r"are|is|be|must|should|in|values?)\b", re.I)


def decimals(sentence: str, file: str, heads_: tuple[str, ...]) -> list[Prop]:
    """A sentence's roundings: the names it rounds (in its clause before the decimals, its head, or a phrase), or the
    scope of every numeric value it rounds (all, a file's, the JSON outputs', a table's); none where the sentence
    excepts some values, asks a least precision, or rounds values a word it does not explain qualifies (`monetary`)."""
    text = normal(sentence)
    found = []
    for m in DECIMALS.finditer(text):
        n = number(m.group("n"))
        rest = text[m.end():]
        if n is None or m.group("lead") or m.group("precision") or re.search(
                r"\b\d+\s+for\b|\bexcept\b|\bunless\b|\bif\b|\bwhen\b", rest, re.I):
            continue
        clause = re.split(r"[;]|\.\s", text[:m.start()])[-1]
        if HEDGES.search(clause) or re.search(r"\b(?:if|when|where|unless)\b", clause, re.I):
            continue
        exact = bool(m.group("exact"))
        said = quoted(text, m.end())
        names = [name for _, _, name in identifiers(clause) if not files_in(name, ())]
        if not names and heads_:
            names = list(heads_)
        scope = QUANTIFIED.search(clause + " " + rest[:60]) or (LEADING.match(clause) if not names else None)
        if scope is not None:
            qualifiers = [w.lower() for w in scope.group("words").split()]
            if not all(word in SCOPE_WORDS for word in qualifiers):
                continue
            words_ = " ".join(qualifiers) + " " + scope.group("noun").lower() + " " + clause.lower()
            kind = "json" if "json" in words_ else "table" if re.search(r"\b(?:csv|columns?|cells?)\b", words_) \
                else "all"
            found.append(Prop((), "decimals", n, strict=exact, scope=kind, file=file, said=said, sentence=sentence))
            continue
        phrase = False
        if not names:
            words = subject_phrase(ROUNDING.sub(" ", clause).strip())
            if len(words) < 2:
                continue
            names, phrase = [" ".join(words)], True
        found.append(Prop(tuple(dict.fromkeys(names)), "decimals", n, strict=exact, phrase=phrase, file=file,
                          said=said, sentence=sentence))
    return found


def statement_props(statement: Statement) -> list[Prop]:
    """A statement's properties: a table row's cells about its first cell's name; a bullet's or a paragraph's
    sentences, the first about its head where it has one; a later sentence about another name than its head left out
    (a definition's case, as `case_cut: ... The x will be negative.`); an imperative step's left out but for its
    rounding."""
    found: list[Prop] = []
    if statement.table:
        first = statement.cells[0].strip(" `*") if statement.cells else ""
        if not re.fullmatch(IDENT, first) or first.lower() in TYPES:
            return found
        for cell in statement.cells[1:]:
            for piece, name in clauses(normal(cell)):
                found += properties(piece, name, (first,), statement.file, statement.text)
            found += decimals(cell, statement.file, (first,))
        return found
    head, body = heads(statement.text)
    for number_, sentence in enumerate(sentences(body)):
        verb = re.match(r"^\s*(?:\d+[.)]\s*)?(?:then\s+)?([A-Za-z]+)", sentence)
        imperative = verb is not None and verb.group(1).lower() in IMPERATIVE and not re.match(
            r"^\s*(?:\d+[.)]\s*)?\w+(?:\s+\w+){0,2}\s+(?:is|are|should|must|will|stays?|remains?)\b", sentence)
        heads_ = head if number_ == 0 else ()
        found += decimals(sentence, statement.file, heads_)
        if imperative or EXCEPT.search(sentence):
            continue
        for piece, name in clauses(normal(sentence)):
            for prop in properties(piece, name, heads_, statement.file, sentence):
                if head and number_ > 0 and not prop.phrase and set(prop.names) - set(head):
                    continue
                found.append(prop)
    return found


TEMPLATE = re.compile(r'"(?P<key>[^"\n]+)"\s*:\s*\[?\s*<(?P<spec>[^>\n]*)>')


def block_props(block: Block) -> list[Prop]:
    """A JSON block's template lines: `"key": <float, rounded to 2 decimals>` and their like."""
    found = []
    for line in block.lines:
        for m in TEMPLATE.finditer(line):
            key = m.group("key")
            if not re.fullmatch(IDENT, key):
                continue
            found += decimals(m.group("spec"), block.file, (key,))
            for piece, name in clauses(normal(m.group("spec"))):
                found += [prop for prop in properties(piece, name, (key,), block.file, line) if prop.kind != "decimals"]
    return found


# --------------------------------------------------------------------------------------------- the contract

PLACEHOLDERS = frozenset({"name", "key", "field", "item"})     # a template's stand-in for a key it does not name


def template_keys(lines: list[str]) -> list[str]:
    """A JSON block's keys, read from its text where its values are placeholders (`<float>`): the top level's and one
    level down's (`parent.key`), a top-level list of records' as `[].key`; a key that is no name (`<ticker>`, `0.95`,
    a date, a stand-in such as `"name": {...}`) is left out with everything under it."""
    tokens = [m.group(0) for m in re.finditer(r'"(?:[^"\\\n]|\\.)*"|[{}\[\]:,]', "\n".join(lines))]
    stack: list[dict] = []
    keys: list[str] = []
    for k, token in enumerate(tokens):
        if token.startswith('"'):
            if k + 1 < len(tokens) and tokens[k + 1] == ":" and stack and stack[-1]["kind"] == "{":
                key = token[1:-1]
                if key.lower() in PLACEHOLDERS and k + 2 < len(tokens) and tokens[k + 2] == "{":
                    key = "<placeholder>"
                stack[-1]["key"] = key
                path = [*stack[-1]["path"], key]
                if len(path) <= 2 and all(part == "[]" or re.fullmatch(r"[A-Za-z_][\w\-]*", part) for part in path):
                    keys.append(".".join(path))
            continue
        if token in "{[":
            parent = stack[-1] if stack else None
            path = [] if parent is None else [*parent["path"], parent["key"] or "?"] if parent["kind"] == "{" else \
                [*parent["path"], "[]"]
            stack.append({"kind": token, "path": path, "key": None})
        elif token in "}]":
            if stack:
                stack.pop()
        elif token == "," and stack and stack[-1]["kind"] == "{":
            stack[-1]["key"] = None
    return list(dict.fromkeys(keys))


HEADER_LINE = re.compile(r"^\s*`?(" + IDENT + r")`?(?:\s*,\s*`?" + IDENT + r"`?)+\s*$")
LISTED = re.compile(r"\b(?P<what>keys?|columns?|fields?|header)\b\s*(?:\((?:in order|in this order)\)|exactly|"
                    r"in order|in this order)?\s*:?\s*(?=`)", re.I)


def contract(statements: list[Statement], blocks: list[Block], named: tuple[str, ...]) -> tuple[dict, dict]:
    """Each output file's keys and columns the instruction names (above)."""
    keys: dict[str, list[str]] = {}
    columns: dict[str, list[str]] = {}

    def add(table: dict, file: str, names: list[str]) -> None:
        held = table.setdefault(file, [])
        held += [name for name in names if name not in held]

    for block in blocks:
        suffix = Path(block.file).suffix.lower() if block.file else ""
        if suffix == ".json":
            add(keys, block.file, template_keys(block.lines))
        elif suffix in (*TABLES, *PARQUETS) and len(block.lines) == 1 and HEADER_LINE.match(block.lines[0]) and (
                re.search(r"\b(?:columns?|header)\b", block.lead, re.I) or files_in(block.lead, named)):
            add(columns, block.file, [name.strip(" `") for name in block.lines[0].split(",")])
    listing = False
    for statement in statements:
        if statement.table:
            first = statement.cells[0].strip(" `*").lower() if statement.cells else ""
            if first in ("column", "columns", "field", "key", "name", "variable", "col"):
                listing = True
                continue
            if listing and statement.file and re.fullmatch(r"`" + IDENT + r"`", statement.cells[0].strip()):
                suffix = Path(statement.file).suffix.lower()
                add(keys if suffix == ".json" else columns, statement.file, [statement.cells[0].strip(" `")])
            continue
        listing = False
        listed = LISTED.search(statement.text)
        if listed is None:
            continue
        files = files_in(statement.text[:listed.start()], named)
        target = files[-1] if files else statement.file
        if not target:
            continue
        tail = re.sub(r"\([^()]*\)", " ", statement.text[listed.end():])
        stop = re.search(r"\.\s+\S|;", tail)
        tail = tail[:stop.start()] if stop else tail
        names = [m.group(1) for m in TICKED.finditer(tail) if m.group(1).lower() not in TYPES and "*" not in m.group(1)]
        if len(names) < 2:
            continue
        suffix, what = Path(target).suffix.lower(), listed.group("what").lower()
        if suffix == ".json" and what.startswith(("key", "field")):
            add(keys, target, names)
        elif suffix in (*TABLES, *PARQUETS) and what.startswith(("column", "field", "header")):
            add(columns, target, names)
    return {f: v for f, v in keys.items() if v}, {f: v for f, v in columns.items() if v}


# ------------------------------------------------------------------------------------- the rules (lane rules1)

RULE_KINDS = ("fraction", "header", "rowsum", "formula")    # the kinds `stated` reads under `rules` alone
NAMED = r"`[A-Za-z_][A-Za-z0-9_]*`|[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+"     # a name as a key's: ticked or snake
FRACTION_DEF = re.compile(r"(?<![\w.`])(?P<name>" + NAMED + r")\s*(?:=|:|\bis\b|—|–|-{1,2})\s*(?:the\s+|a\s+)?"
                          r"(?:fraction|share|proportion)\s+of\b|(?<![\w.`])(?P<as>" + NAMED + r")\s+(?:as|in)\s+"
                          r"(?:a\s+)?(?:decimal\s+)?fraction\b", re.I)
INLINE_HEADER = re.compile(r"`([A-Za-z_][\w\-]*(?:\s*,\s*[A-Za-z_][\w\-]*){2,})`")
ROW_SUM = re.compile(r"\b(?:columns?|percentages|shares|weights|parts)\b[^.\n]{0,80}?\bmust\s+sum\s+to\s+"
                     r"(?:exactly\s+)?(?P<num>\d+(?:\.\d+)?)\s*(?P<pct>%|percent\b)?[^.\n]{0,40}?"
                     r"(?:for\s+|on\s+|in\s+)?(?:every|each)\s+(?:row|window|date|period)\b", re.I)
FORMULA = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*=\s*([^=\n`;]+)")     # `a = expr`, anywhere in a line
DEFINED = re.compile(r"^\s*[-*]?\s*`?([A-Za-z_]\w*)`?\s*:\s*([^:\n]+)$")    # `- a: expr`, an operator in it
TABLE_KINDS = (*TABLES, *PARQUETS)
EXAMPLE_FIGURE_DIGITS = 3       # an example's figure read with this many significant digits at the least


def opening(text: str) -> str:
    """The words a text opens with, whole, within `QUOTE_CHARS`: a definition's own words for a finding; a first word
    longer than that (a header written without spaces) cut at its last comma within it."""
    kept: list[str] = []
    for word in text.split():
        if len(" ".join([*kept, word])) > QUOTE_CHARS:
            cut = " ".join(kept) if kept else word[:word.rfind(",", 0, QUOTE_CHARS) + 1 or QUOTE_CHARS]
            return cut + " ..."
        kept.append(word)
    return " ".join(kept)


def figures(text: str) -> list[str]:
    """The numbers a text writes outside its strings, each as written."""
    unquoted = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', text)
    return re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])", unquoted)


def significant(figure: str) -> int:
    """A figure's significant digits as written, its exponent's left out."""
    return len(re.sub(r"[^0-9]", "", re.split(r"[eE]", figure)[0]).strip("0"))


def examples(blocks: list[Block], instruction: str) -> tuple[str, ...]:
    """The example output's figures: each number of a JSON object block that is no template (no `<...>`), its strings
    and `//` comments left out, with `EXAMPLE_FIGURE_DIGITS` significant digits or more, and written nowhere in the
    instruction outside such blocks (an input the task states is no example's figure)."""
    found: list[str] = []
    inside: list[str] = []
    for block in blocks:
        text = "\n".join(re.sub(r"//.*$", "", line) for line in block.lines).strip()
        if not (text.startswith("{") and text.endswith("}")) or "<" in text:
            continue
        inside.append("\n".join(block.lines))
        found += [figure for figure in figures(text) if significant(figure) >= EXAMPLE_FIGURE_DIGITS]
    if not found:
        return ()
    outside = instruction or ""
    for text in inside:
        outside = outside.replace(text, " ")
    stated_ = {number(figure) for figure in figures(outside)}
    return tuple(dict.fromkeys(figure for figure in found if number(figure) not in stated_))


def named_table(text: str, file: str, named: tuple[str, ...]) -> str:
    """The output table a statement is about: the one output file its text names where it names one, else its
    section's; "" where that is no table."""
    files = files_in(text, named)
    chosen = files[0] if len(files) == 1 else file
    return chosen if Path(chosen).suffix.lower() in TABLE_KINDS else ""


def rule_props(instruction: str, statements: list[Statement], blocks: list[Block], named: tuple[str, ...],
               columns: dict[str, list[str]]) -> list[Prop]:
    """The rules' properties (`RULE_KINDS`, above): a definition's fraction and a formula from every line, code
    blocks' among them; an inline header and a row sum from each statement, about its output table. A header's columns
    the contract already names (`columns`) are left to the contract; a formula's left side must be a column the
    instruction names (the contract's or an inline header's)."""
    found: list[Prop] = []
    lines = (instruction or "").replace("\r\n", "\n").split("\n")
    for line in lines:
        for m in FRACTION_DEF.finditer(normal(line)):
            name = (m.group("name") or m.group("as")).strip("`")
            if name.lower() not in TYPES:
                found.append(Prop((name,), "fraction", 0.0, 1.0, said=opening(normal(line)[m.start():]),
                                  sentence=line.strip()))
    headed: dict[str, list[str]] = {}
    for statement in statements:
        text = normal(statement.text)
        table = named_table(statement.text, statement.file, named)
        for m in INLINE_HEADER.finditer(text) if table else ():
            wanted = [column.strip() for column in m.group(1).split(",")]
            fresh = [column for column in wanted if column not in columns.get(table, [])]
            headed.setdefault(table, []).extend(wanted)
            if fresh:
                found.append(Prop(tuple(fresh), "header", file=table, said=opening(m.group(0)),
                                  sentence=statement.text))
        for m in ROW_SUM.finditer(text):
            sentence = re.split(r"(?<=[.!?:])\s+", text[:m.start("num")])[-1]      # its own sentence's names
            names = tuple(name for _, _, name in identifiers(sentence) if not files_in(name, ()))
            start = m.start("num") - len(sentence)
            found.append(Prop(names if len(names) > 1 else (), "rowsum", float(m.group("num")),
                              percent=bool(m.group("pct")), file=table,
                              said=quoted(text[start:m.end()], m.end() - start), sentence=statement.text))
    known = {column for file in (*columns, *headed) for column in (*columns.get(file, ()), *headed.get(file, ()))}
    for line in lines:
        plain = line.replace("×", "*").replace("−", "-").replace("·", "*")
        candidates = [(m.group(1), m.group(2)) for m in FORMULA.finditer(plain)]
        defined = DEFINED.match(plain)
        if defined and re.search(r"[-+*/]", defined.group(2)):
            candidates.append((defined.group(1), defined.group(2)))
        for lhs, rhs in candidates:
            rhs = rhs.strip().rstrip(".,")
            if lhs in known and rhs and re.search(r"[-+*/(]", rhs):
                found.append(Prop((lhs,), "formula", said=opening(f"{lhs} = {rhs}"), sentence=line.strip(),
                                  expr=rhs))
    return found


def stated(instruction: str, named: tuple[str, ...] = (), rules: bool = False, wide: bool = False) -> Stated:
    """What the instruction states of its outputs (above), each property once; under `rules` (lane rules1) the
    rules' kinds after today's (`rule_props`) and the example output's figures (`examples`); under `wide` too (lane
    rules2) what the six rules read (`wide_stated`)."""
    statements, blocks = segment(instruction or "", tuple(named))
    props: list[Prop] = []
    for statement in statements:
        props += statement_props(statement)
    for block in blocks:
        props += block_props(block)
    keys, columns = contract(statements, blocks, tuple(named))
    if rules:
        summed = {frozenset(prop.names) for prop in props if prop.kind == "sum" and len(prop.names) > 1}
        props += [prop for prop in rule_props(instruction, statements, blocks, tuple(named), columns)
                  if not (prop.kind == "rowsum" and frozenset(prop.names) in summed)]    # today's `sum` reads it
    unique, seen = [], set()
    for prop in props:
        key = (prop.names, prop.kind, prop.value, prop.upper, prop.scope, prop.file, prop.expr)
        if key not in seen:
            seen.add(key)
            unique.append(prop)
    written = frozenset(value for value in (number(token) for token in re.findall(NUM, instruction or ""))
                        if value is not None)
    return Stated(unique, keys, columns, written, examples(blocks, instruction) if rules else (),
                  wide_stated(instruction, tuple(named)) if rules and wide else None)


# ----------------------------------------------------------------------------------------------- the outputs


@dataclass
class Values:
    """One name's values in one file, in order: numbers (None for a cell that is not one), the cells' text where the
    file is a table, and whether it is a table's column, a JSON list or scalars."""
    file: str
    name: str
    numbers: list[float | None] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    table: bool = False
    ordered: bool = False


def words(name: str) -> list[str]:
    out = []
    for run in re.findall(r"[A-Za-z]+|[0-9]+", name):
        out += [part.lower() for part in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", run)]
    return [singular(word) for word in out]


def singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def scalar(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        return math.inf


class Outputs:
    """The output directory's names and their values: each JSON key (by its last part and its full path; a list's
    numbers in order; a list of records' fields as columns), each table's columns; and the files' raw shapes for
    the contract."""

    def __init__(self, out: Path, row_counts: bool = True) -> None:
        self.values: list[Values] = []
        self.index: dict[tuple[str, str, bool], Values] = {}
        self.json: dict[str, object] = {}
        self.headers: dict[str, list[str]] = {}
        self.rows: dict[str, int] = {}                  # each table's data rows read (lane rules2's `stated_rows`)
        self.row_counts = row_counts
        self.incomplete: set[str] = set()
        out = Path(out)
        try:
            files = sorted(p for p in out.rglob("*") if p.is_file() and not p.is_symlink())
        except OSError:
            files = []
        for path in files[:MAX_FILES]:
            name = path.relative_to(out).as_posix()
            try:
                if path.stat().st_size > MAX_BYTES or path.stat().st_size == 0:
                    continue
                suffix = path.suffix.lower()
                if suffix == ".json":
                    doc = json.loads(path.read_bytes())
                    self.json[name] = doc
                    self.document(name, doc)
                elif suffix in TABLES:
                    self.table(name, path.read_bytes().decode("utf-8-sig", errors="replace"), TABLES[suffix])
                elif suffix in PARQUETS:
                    self.parquet(name, path)
            except Exception:  # noqa: BLE001 - a file that cannot be read holds nothing here
                continue

    def add(self, file: str, name: str, value: object, text: str | None = None, table: bool = False,
            ordered: bool = False) -> Values:
        held = self.index.get((file, name, table))
        if held is None:
            held = self.index[(file, name, table)] = Values(file, name, table=table, ordered=ordered)
            self.values.append(held)
        held.numbers.append(scalar(value) if text is None else value)
        held.texts.append(text if text is not None else "")
        return held

    def document(self, file: str, doc: object) -> None:
        """A JSON document's numbers under each key's path (`a.b`; `meets` reads its last part too): a scalar; a list
        of numbers in order; a list of records' fields as columns; an object of numbers' values under the object's
        own path too (unordered: a sum's)."""
        def visit(value: object, path: list[str]) -> None:
            if isinstance(value, dict):
                items = list(value.items())[:MAX_ITEMS]
                numbers = [scalar(item) for _, item in items]
                if path and items and all(n is not None for n in numbers):
                    self.values.append(Values(file, ".".join(path), list(numbers), [""] * len(numbers)))
                for key, item in items:
                    visit(item, [*path, str(key)])
            elif isinstance(value, list):
                kept = value[:MAX_ITEMS]
                if path and kept and all(scalar(item) is not None for item in kept):
                    for item in kept:
                        self.add(file, ".".join(path), item, ordered=True)
                elif kept and all(isinstance(item, dict) for item in kept):
                    for record in kept:
                        for key, item in record.items():
                            if scalar(item) is not None:
                                self.add(file, ".".join([*path, str(key)]), item, ordered=True)
                            elif isinstance(item, (dict, list)):
                                visit(item, [*path, str(key)])
                else:
                    for item in kept:
                        visit(item, path)
            elif path and (found := scalar(value)) is not None:
                self.add(file, ".".join(path), found)

        visit(doc, [])

    def table(self, file: str, text: str, delimiter: str) -> None:
        self.incomplete.add(file)
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
        header = [column.strip() for column in next(reader, [])]
        self.headers[file] = header
        self.rows[file] = 0
        for row in itertools.islice(reader, MAX_ROWS):
            if not any(part.strip() for part in row):
                continue
            if self.row_counts:
                self.rows[file] += 1
            for index, column in enumerate(header):
                cell = row[index].strip() if index < len(row) else ""
                self.add(file, column, cell_number(cell), cell, table=True, ordered=True)
        if next(reader, None) is None:
            self.incomplete.discard(file)

    def parquet(self, file: str, path: Path) -> None:
        import pyarrow.parquet as pq
        read = pq.read_table(path)
        self.headers[file] = [str(column) for column in read.column_names]
        self.rows[file] = min(read.num_rows, MAX_ROWS)
        for column, values in zip(read.column_names, read.columns):
            for item in values.to_pylist()[:MAX_ROWS]:
                self.add(file, str(column), scalar(item), "", table=True, ordered=True)

    def find(self, prop: Prop) -> list[Values]:
        """The values `prop` names: in its file where it holds them there, else in any file."""
        found = [held for held in self.values if any(meets(name, held.name, prop.phrase) for name in prop.names)]
        if prop.file:
            local = [held for held in found if held.file == prop.file]
            if local:
                return local
        return found


def cell_number(text: str) -> float | None:
    if not text:
        return None
    if text.lower() in ("nan", "-nan", "inf", "-inf", "infinity", "-infinity"):
        return math.nan
    try:
        return float(text)
    except ValueError:
        return None


def meets(said: str, name: str, phrase: bool) -> bool:
    """Whether a name the instruction writes meets an output's name: the same, case aside (a `*` matching any
    run); a phrase where the output name's words, two or more (or one, the phrase itself), stand in it in a row."""
    if not phrase:
        last = name.rsplit(".", 1)[-1] if "." in name and "." not in said else name
        if "*" in said:
            return fnmatch.fnmatchcase(last.lower(), said.lower())
        return last.lower() == said.lower() or name.lower() == said.lower()
    have = words(name.rsplit(".", 1)[-1])
    want = [singular(word) for word in said.split()]
    if not have:
        return False
    if len(have) == 1:
        return want == have
    return any(want[k:k + len(have)] == have for k in range(len(want) - len(have) + 1))


# ------------------------------------------------------------------------------------------------- the check


def violated(prop: Prop, values: Values, exempt: frozenset = frozenset()) -> tuple[int, float | str] | None:
    """How many of the values break `prop` and the first that does (a table's cell as written); None where none does
    or none can be read. A rounding passes over the values in `exempt` (the instruction's own numbers: an input it
    echoes, as a confidence level of 0.999 under "values rounded to 2 decimals") and, where it asks exactly N
    decimals, over a table's integer cells."""
    if prop.kind == "decimals":
        bad: list[float | str] = []
        for n, text in zip(values.numbers, values.texts):
            if n is None or not math.isfinite(n) or n in exempt:
                continue
            if text and "e" not in text.lower():
                if "." not in text and prop.strict:
                    continue
                places = len(text.split(".", 1)[1]) if "." in text else 0
                if places > prop.value or (prop.strict and places != prop.value):
                    bad.append(text)
            elif abs(round(n, int(prop.value)) - n) > 1e-9 * max(1.0, abs(n)):
                bad.append(repr(n))
        return (len(bad), bad[0]) if bad else None
    numbers = [n for n in values.numbers if n is not None and math.isfinite(n)]
    if not numbers:
        return None
    if prop.kind in ("gt", "ge", "lt", "le", "eq", "between", "fraction"):
        bad = [n for n in numbers if breaks(prop, abs(n) if prop.absolute else n)]
        return (len(bad), bad[0]) if bad else None
    if prop.kind == "start":
        if not values.ordered or values.numbers[0] is None:
            return None
        first = numbers[0]
        return None if abs(first - prop.value) <= TOLERANCE * max(1.0, abs(prop.value)) else (1, first)
    if prop.kind in ("increasing", "decreasing"):
        if not values.ordered:
            return None
        sign = 1 if prop.kind == "increasing" else -1
        for a, b in zip(numbers, numbers[1:]):
            if sign * (b - a) < 0 or (prop.strict and b == a):
                return 1, b
        return None
    if prop.kind == "sum" and not values.table and len(prop.names) == 1:
        total = math.fsum(numbers)
        wanted = [prop.value] + ([prop.value / 100.0] if prop.percent else [])
        return None if any(abs(total - w) <= SUM_TOLERANCE * max(1.0, abs(w)) for w in wanted) else (1, total)
    return None


def breaks(prop: Prop, n: float) -> bool:
    """Whether a value breaks a bound, an equality or an interval: a strict one exactly, a non-strict one past a
    relative `TOLERANCE`; a strict bound at zero only by the value's sign (a zero there is an underflow as often as
    an error: E020 to E030's solved greeks hold gamma and vega of 0 on their shortest maturities)."""
    slack = TOLERANCE * max(1.0, abs(prop.value or 0.0))
    if prop.kind == "gt":
        return n < 0 if prop.value == 0 else not n > prop.value
    if prop.kind == "ge":
        return n < prop.value - slack
    if prop.kind == "lt":
        return n > 0 if prop.value == 0 else not n < prop.value
    if prop.kind == "le":
        return n > prop.value + slack
    if prop.kind == "eq":
        return abs(n - prop.value) > slack
    low_ok = n > prop.value if prop.lower_open else n >= prop.value - slack
    high_ok = n < prop.upper if prop.strict else n <= prop.upper + TOLERANCE * max(1.0, abs(prop.upper))
    return not (low_ok and high_ok)


def shown(value: float | str) -> str:
    return value if isinstance(value, str) else f"{value:.6g}"


def rows_summing(prop: Prop, outputs: Outputs) -> list[str]:
    """`a`, `b` and `c` sum to a value: per table row (the columns named), per JSON object (the keys named)."""
    found = []
    by_file: dict[str, dict[str, Values]] = {}
    for held in outputs.values:
        for name in prop.names:
            if meets(name, held.name, False) and held.name == held.name.rsplit(".", 1)[-1]:
                by_file.setdefault(held.file, {})[name] = held
    for file, named in by_file.items():
        if len(named) != len(prop.names):
            continue
        columns = list(named.values())
        rows = min(len(c.numbers) for c in columns)
        wanted = [prop.value] + ([prop.value / 100.0] if prop.percent else [])
        for k in range(rows):
            parts = [c.numbers[k] for c in columns]
            if any(p is None or not math.isfinite(p) for p in parts):
                continue
            total = math.fsum(parts)
            if not any(abs(total - w) <= SUM_TOLERANCE * max(1.0, abs(w)) for w in wanted):
                found.append(f"{file}: {', '.join(f'`{n}`' for n in prop.names)} sum to {total:.6g}"
                             + (f" on row {k + 1}" if columns[0].table else "") + f", which the task states: "
                             f"\"{prop.said}\"")
                break
    return found


# lane rules1: the rules over an output table's columns (`header`, `rowsum`, `formula`), each the finding's lines (at
# most one a table) and the tables it reads (`TABLE_READS`, the audit's `held`)

DATED = frozenset({"date", "time", "timestamp", "datetime", "index", "id", "year", "month", "day", "period", "row"})
FORMULA_TOLERANCE = 1e-3        # a formula's row off by more than this, relative, past its written rounding
NEAR = 0.05                     # a table whose row sums sit this near the stated figure on some row is the statement's
FUNCTIONS = {"sqrt": math.sqrt, "log": math.log, "ln": math.log, "exp": math.exp, "abs": abs}
OPERATORS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}


def table_columns(outputs: Outputs, file: str) -> dict[str, Values]:
    """A table's columns by name, each its values in row order."""
    return {held.name: held for held in outputs.values if held.file == file and held.table}


def tables_of(prop: Prop, outputs: Outputs) -> list[str]:
    """The output tables a rule reads: its own where the output holds it, else every one."""
    return [prop.file] if prop.file in outputs.headers else list(outputs.headers)


def half_unit(text: str) -> float:
    """Half a unit in a cell's last written place: its rounding, as written ("0.125" 0.0005); 0 where none shows."""
    text = text.strip()
    if "." not in text or "e" in text.lower():
        return 0.0
    return 0.5 * 10.0 ** -len(text.split(".", 1)[1])


def header_off(prop: Prop, outputs: Outputs) -> list[str]:
    """`header`: the columns of the header the task gives that its table lacks."""
    have = outputs.headers.get(prop.file)
    gone = [column for column in prop.names if have is not None and column not in have]
    if not gone:
        return []
    return [f"{prop.file} has no column {named_list(gone)}, which the header the task gives holds: \"{prop.said}\""]


def named_list(names: list[str], most: int = 4) -> str:
    """Names in backticks, `a`, `b` and `c`, the first `most` and "and more" after them: never a count."""
    shown_ = [f"`{name}`" for name in names[:most]]
    if len(names) > most:
        return ", ".join(shown_) + " and more"
    return shown_[0] if len(shown_) == 1 else ", ".join(shown_[:-1]) + " and " + shown_[-1]


def header_read(prop: Prop, outputs: Outputs) -> list[str]:
    return [prop.file] if prop.file in outputs.headers else []


def summed_columns(prop: Prop, outputs: Outputs, file: str) -> list[Values]:
    """`rowsum`'s columns in a table: those it names where it names two or more the table holds, else every numeric
    column but a date's, an index's or an id's."""
    columns = table_columns(outputs, file)
    if prop.names:
        found = [columns[name] for name in prop.names if name in columns]
        return found if len(found) > 1 else []
    numeric = [held for name, held in columns.items() if not set(words(name)) & DATED and any(
        n is not None for n in held.numbers[:50]) and all(n is not None or not text.strip() for n, text in
                                                           zip(held.numbers[:50], held.texts[:50]))]
    return numeric if len(numeric) > 1 else []


def rows_off(prop: Prop, outputs: Outputs) -> list[str]:
    """`rowsum`: a row of the statement's table whose columns sum off the stated figure (or its hundredth, said in
    percent) past their written rounding; the statement's table, where the statement names none, the one whose row
    sums sit within `NEAR` of the figure on some row."""
    found = []
    for file in tables_of(prop, outputs):
        columns = summed_columns(prop, outputs, file)
        if not columns:
            continue
        rows = []
        for k in range(min(len(held.numbers) for held in columns)):
            parts = [held.numbers[k] for held in columns]
            if all(part is not None and math.isfinite(part) for part in parts):
                rows.append((k, math.fsum(parts), sum(half_unit(held.texts[k]) for held in columns)))
        wanted = [w for w in (prop.value, *([prop.value / 100.0] if prop.percent else []))
                  if any(abs(total - w) <= NEAR * abs(w) for _, total, _ in rows)]
        if not wanted:
            continue
        for k, total, slack in rows:
            if abs(total - wanted[0]) > max(slack, SUM_TOLERANCE * max(1.0, abs(wanted[0]))):
                found.append(f"{file}: row {k + 1}'s " + (", ".join(f"`{held.name}`" for held in columns[:6]) + " "
                                                          if prop.names else "columns ")
                             + f"sum to {total:.6g}; the task states: \"{prop.said}\"")
                break
    return found


def rows_read(prop: Prop, outputs: Outputs) -> list[str]:
    return [file for file in tables_of(prop, outputs) if summed_columns(prop, outputs, file)]


def resolved(prop: Prop, header: list[str]) -> tuple[ast.Expression, str] | None:
    """A formula's right side against a table's header: the longest prefix of its words that reads as an expression
    over the table's columns, numbers and `FUNCTIONS` alone, naming a column and not its left side, where what follows
    it is a remark in brackets (a prefix of a longer formula is another formula); None where none does."""
    lhs, tokens = prop.names[0], prop.expr.split()
    for cut in range(len(tokens), 0, -1):
        text = " ".join(tokens[:cut]).rstrip(".,")
        rest = tokens[cut:]
        if re.fullmatch(r"\s*[A-Za-z_]\w*\s*", text) or not re.search(r"[-+*/(]", text):
            continue
        if rest and not (rest[0].startswith("(") and re.match(r"[A-Za-z]{2,}", rest[0].lstrip("("))
                         and rest[0].lstrip("(").rstrip(",.)") not in header):
            continue
        names = set(re.findall(r"[A-Za-z_]\w*", text))
        if any(name not in header and name not in FUNCTIONS for name in names) or lhs in names or \
                not any(name in header for name in names):
            continue
        try:
            tree = ast.parse(text.replace("^", "**"), mode="eval")
        except SyntaxError:
            continue
        if all(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load,
                                 ast.USub, ast.UAdd, *OPERATORS)) for node in ast.walk(tree)) and all(
                isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and len(node.args) == 1 and
                not node.keywords for node in ast.walk(tree) if isinstance(node, ast.Call)) and all(
                isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
                for node in ast.walk(tree) if isinstance(node, ast.Constant)):
            return tree, text
    return None


def evaluated(node: ast.AST, row: dict[str, float]) -> float:
    """A resolved formula's value on one row (`resolved` admits nothing else); NaN where it has none there."""
    try:
        if isinstance(node, ast.Expression):
            return evaluated(node.body, row)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return row[node.id]
        if isinstance(node, ast.UnaryOp):
            value = evaluated(node.operand, row)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            return float(OPERATORS[type(node.op)](evaluated(node.left, row), evaluated(node.right, row)))
        return float(FUNCTIONS[node.func.id](evaluated(node.args[0], row)))
    except (ArithmeticError, ValueError, TypeError, KeyError):
        return math.nan


def formula_off(prop: Prop, outputs: Outputs) -> list[str]:
    """`formula`: a row of a table holding the formula's left side and every column its right side names, where the
    left side's value is off the formula's past `FORMULA_TOLERANCE` and its written rounding."""
    found = []
    for file, header in outputs.headers.items():
        made = resolved(prop, header) if prop.names[0] in header else None
        if made is None:
            continue
        tree, text = made
        columns = table_columns(outputs, file)
        left = columns[prop.names[0]]
        for k, value in enumerate(left.numbers):
            row = {name: held.numbers[k] for name, held in columns.items()
                   if k < len(held.numbers) and held.numbers[k] is not None}
            got = evaluated(tree, row)
            if value is None or not math.isfinite(value) or not math.isfinite(got):
                continue
            if abs(value - got) > FORMULA_TOLERANCE * abs(got) + 1e-6 + half_unit(left.texts[k]):
                found.append(f"{file}: `{prop.names[0]}` is {shown(value)} on row {k + 1} where the task's formula "
                             f"gives {shown(got)}: \"{prop.names[0]} = {text}\"")
                break
    return found


def formula_read(prop: Prop, outputs: Outputs) -> list[str]:
    return [file for file, header in outputs.headers.items()
            if prop.names[0] in header and resolved(prop, header) is not None]


TABLE_RULES = {"header": header_off, "rowsum": rows_off, "formula": formula_off}
TABLE_READS = {"header": header_read, "rowsum": rows_read, "formula": formula_read}


def holds(prop: Prop, out: Path) -> bool:
    """Lane rules1's put-back (agent/n11/solve.py): whether the output directory `out` holds `prop`, read and kept."""
    try:
        return audited(prop, Outputs(out), frozenset()).startswith("held:")
    except Exception:  # noqa: BLE001 - an output that cannot be read holds nothing here
        return False


def said_of(prop: Prop, held: Values, broke: tuple[int, float | str]) -> str:
    """A finding's line: where, what the value is, and the task's own words."""
    where = f"{held.file}: {'column' if held.table else 'key'} `{held.name}`"
    if prop.kind == "decimals":
        how = f"not written with exactly {prop.value:g} decimals" if prop.strict else \
            f"with more than {prop.value:g} decimals"
        return f"{where} holds {broke[0]} value(s) {how}, e.g. {broke[1]}, which the task states: \"{prop.said}\""
    value = shown(broke[1])
    if prop.kind == "fraction":         # lane rules1: the value and the definition, no count
        return f"{where} {'holds' if held.table or held.ordered else 'is'} {value}; the task defines it as a " \
               f"fraction: \"{prop.said}\""
    head = f"{where} holds {broke[0]} value(s) that break it, e.g. {value}" if broke[0] > 1 or held.table or \
        held.ordered else f"{where} = {value}"
    return f"{head}, which the task states: \"{prop.said}\" ({prop.describe().split(': ', 1)[1]})"


def check(stated_: Stated, out: Path) -> tuple[list[str], list[str]]:
    """The findings over the output directory: (`instruction:` lines, `contract:` lines), without their prefix."""
    found, missing = checked(stated_, out)
    return [line for _, line in found], missing


def checked(stated_: Stated, out: Path, outputs: Outputs | None = None) -> tuple[list[tuple[Prop, str]], list[str]]:
    """`check`'s findings, each `instruction:` line with the property it breaks (lane rules1 knows its rules' by it);
    over `outputs` where the caller read the output already (lane rules2's six read the same)."""
    outputs = outputs if outputs is not None else Outputs(out)
    said: list[tuple[Prop, str]] = []
    for prop in stated_.props:
        lines: list[str] = []
        if prop.kind in TABLE_RULES:
            lines = TABLE_RULES[prop.kind](prop, outputs)
        elif prop.kind == "sum" and len(prop.names) > 1:
            lines = rows_summing(prop, outputs)
        elif prop.kind == "decimals" and prop.scope:
            for held in outputs.values:
                if prop.file and held.file != prop.file or \
                        prop.scope == "json" and held.table or prop.scope == "table" and not held.table:
                    continue
                broke = violated(prop, held, stated_.numbers)
                if broke:
                    lines.append(said_of(prop, held, broke))
                    break
        else:
            for held in outputs.find(prop):
                broke = violated(prop, held)
                if broke:
                    lines.append(said_of(prop, held, broke))
                if len(lines) >= PER_PROP:
                    break
        said += [(prop, line[:FINDING_CHARS]) for line in lines[:PER_PROP]]
    missing: list[str] = []
    for file, wanted in stated_.keys.items():
        if file in outputs.json:
            gone = [key for key in wanted if not has_key(outputs.json[file], key)]
            if gone:
                missing.append(f"{file} lacks {len(gone)} key(s) the task names: " + ", ".join(
                    f"`{key}`" for key in gone[:8]) + (" and more" if len(gone) > 8 else ""))
    for file, wanted in stated_.columns.items():
        if file in outputs.headers:
            gone = [column for column in wanted if column not in outputs.headers[file]]
            if gone:
                missing.append(f"{file} lacks {len(gone)} column(s) the task names: " + ", ".join(
                    f"`{column}`" for column in gone[:8]) + (" and more" if len(gone) > 8 else ""))
    lines = {line: prop for prop, line in reversed(said)}                 # each line once, with its first property
    return [(lines[line], line) for line in dict.fromkeys(line for _, line in said)], \
        [line[:FINDING_CHARS] for line in missing]


def has_key(doc: object, key: str) -> bool:
    """Whether a JSON document holds a key path (`a`, `a.b`; `[].a` in each record of a top-level list)."""
    parts = key.split(".")
    if parts[0] == "[]":
        return isinstance(doc, list) and all(isinstance(item, dict) and parts[1] in item for item in doc[:MAX_ITEMS]) \
            if len(parts) > 1 else isinstance(doc, list)
    if not isinstance(doc, dict):
        return False
    if parts[0] not in doc:
        return False
    if len(parts) == 1:
        return True
    inner = doc[parts[0]]
    if isinstance(inner, dict):
        return parts[1] in inner
    if isinstance(inner, list):
        return all(isinstance(item, dict) and parts[1] in item for item in inner[:MAX_ITEMS]) if inner else True
    return True


# ------------------------------------------------------------------------------------ the six rules (lane rules2)

WIDE_RULES = ("stated_rows", "abs_bound", "twin_equal", "key_spelling", "unnamed_index", "group_count")
STATED_ROWS = re.compile(r"(?:(?:=|\()\s*|\bexactly\s+|\bhas\s+(?:exactly\s+)?)(\d{1,6})\s+(?:data\s+)?rows?\b", re.I)
NOT_COUNT = re.compile(r"(?:at least|at most|minimum|maximum|more than|fewer than|less than|up to|first|last|top|"
                       r"bottom|over|>=|<=|>|<)\s*$", re.I)    # a bound or a part of the rows before it: no count
NOT_COUNT_CHARS = 14            # the characters before a stated count read for one
LEAST_ROWS = 2                  # a stated count under this is no table's count (a one-row summary's)
ABS_BOUND = re.compile(r"\|\s*`?([A-Za-z_]\w*)`?\s*\|\s*(<=|<)\s*`?(" + NUM + r")`?")
SELECTING = frozenset({"keep", "drop", "filter", "exclude", "remove", "discard", "skip", "select", "clip", "clamp",
                       "trim", "winsorize", "winsorise", "flag"})  # a line with one selects by the bound: none stated
NAMED_TICKED = re.compile(r"`([A-Za-z_][\w.\-:]{1,60})`")      # a name the instruction writes: in backticks, or a
NAMED_KEY = re.compile(r"\"([^\"\n<>{}]{2,60})\"\s*:")          # JSON key it quotes
FILE_LIKE = re.compile(r"\.(?:csv|tsv|json|py|txt|parquet|png|md)$", re.I)
UNNAMED = re.compile(r"^(?:|Unnamed: \d+)$")      # a header cell with no name, or pandas' for an index without one
WITHOUT = frozenset({"no", "without"})          # `twin_equal`: the word that makes a key its twin's without
GROUP_ROWS = re.compile(
    r"^\s*[-*]\s*`((?:n|num)_\w+)`\s*:\s*(?:how many|the (?:number|count) of)\s+(?:all\s+)?rows\s+"
    r"(?:of\s+|in\s+)?`?([\w./-]+\.(?:csv|tsv))`?\s+"
    r"(?:hold\s+(?:the\s+)?`?([A-Za-z_]\w*)`?\s+`?([A-Za-z_]\w*)`?|"
    r"where\s+`?([A-Za-z_]\w*)`?\s+(?:is|=|==)\s+`?([A-Za-z_]\w*)`?)\s*\.?\s*$", re.I)
OBJECT_ITEMS = 500              # a JSON list's items read for their objects (`twin_equal`, `key_spelling`)
TWIN_TOLERANCE = 1e-12          # twins this close, relative, hold one value
NAME_SHOWN = 48                 # a name shown at most this long
PER_RULE = 2                    # a rule's lines a file, at most (as agent/n11/sanity.py's)


@dataclass(frozen=True)
class Wide:
    """What the six rules read of the instruction (`wide_stated`): each output table's stated row counts, each
    absolute bound (name, `<` or `<=`, the bound, its words), every name it writes, and bound group-row counts."""
    rows: dict = field(default_factory=dict)
    bounds: tuple = ()
    named: frozenset = frozenset()
    groups: tuple = ()
    group_json: str | None = None


def wide_stated(instruction: str, named: tuple[str, ...]) -> Wide:
    """The six rules' reading of the instruction: a row count bound to the output table its line names, else its
    heading's, else the last line's that named one table alone; the absolute bounds; the names in backticks and the
    quoted JSON keys, and explicit unfiltered group-row counts."""
    tables = {Path(name).name.lower(): name for name in named if Path(name).suffix.lower() in TABLE_KINDS}
    rows: dict[str, set[int]] = {}
    current, fence = None, False
    for line in (instruction or "").replace("\r\n", "\n").split("\n"):
        if FENCE.match(line):
            fence = not fence
        files = [tables[found.group(1).lower()] for found in OUTPUT_FILE.finditer(line)
                 if found.group(1).lower() in tables]
        heading = not fence and HEADING.match(line) is not None
        if heading:
            current = files[-1] if files else None
        for m in STATED_ROWS.finditer(line):
            count, target = int(m.group(1)), files[-1] if files else current
            if target is not None and count >= LEAST_ROWS and not NOT_COUNT.search(
                    line[:m.start()][-NOT_COUNT_CHARS:]):
                rows.setdefault(target, set()).add(count)
        if files and not heading and len(set(files)) == 1:
            current = files[-1]
    bounds = []
    for line in (instruction or "").split("\n"):
        text = normal(line)
        if SELECTING & {word.lower() for word in re.findall(r"[A-Za-z]+", text)}:
            continue
        for m in ABS_BOUND.finditer(text):
            if GOES_ON.match(text, m.end()) or number(m.group(3)) is None:
                continue
            bounds.append((m.group(1), m.group(2), number(m.group(3)), f"|{m.group(1)}| {m.group(2)} {m.group(3)}"))
    asked = set(NAMED_TICKED.findall(instruction or "")) | set(NAMED_KEY.findall(instruction or ""))
    groups = []
    for line in (instruction or "").splitlines():
        match = GROUP_ROWS.fullmatch(line)
        if match is None:
            continue
        key, table, column, label, where_column, where_label = match.groups()
        column, label = column or where_column, label or where_label
        path = table.removeprefix("/app/output/").removeprefix("/output/").lower()
        targets = [name for name in named if Path(name).suffix.lower() in TABLE_KINDS and
                   (name.lower() == path or "/" not in table and Path(name).name.lower() == path)]
        if len(targets) == 1 and key.lower().split("_")[-1] == label.lower():
            groups.append((key, targets[0], column, label))
    json_files = [name for name in named if Path(name).suffix.lower() == ".json"]
    return Wide({table: frozenset(counts) for table, counts in rows.items()}, tuple(dict.fromkeys(bounds)),
                frozenset(asked), tuple(dict.fromkeys(groups)), json_files[0] if len(json_files) == 1 else None)


def name_shown(name: object) -> str:
    text = re.sub(r"[^\x20-\x7e]|`", "?", str(name))
    return text[:NAME_SHOWN] + ("..." if len(text) > NAME_SHOWN else "")


def leaves(doc: object, path: tuple[str, ...] = ()):
    """A JSON document's scalars under their key paths (a list's items under the list's), `MAX_ITEMS` a list."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            yield from leaves(value, (*path, str(key)))
    elif isinstance(doc, list):
        for item in doc[:MAX_ITEMS]:
            yield from leaves(item, path)
    else:
        yield path, doc


def objects(doc: object):
    """A JSON document's objects, `OBJECT_ITEMS` of a list's items each."""
    if isinstance(doc, dict):
        yield doc
        for value in doc.values():
            yield from objects(value)
    elif isinstance(doc, list):
        for item in doc[:OBJECT_ITEMS]:
            yield from objects(item)


def output_table(outputs: Outputs, named: str) -> str | None:
    """The output table a name the instruction writes is: by its path, else by its base name where one alone has it."""
    if named in outputs.headers:
        return named
    same = [file for file in outputs.headers if Path(file).name.lower() == Path(named).name.lower()]
    return same[0] if len(same) == 1 else None


def rows_unstated(wide: Wide, outputs: Outputs) -> list[tuple[str, str, str]]:
    """`stated_rows`: an output table holding a number of data rows the instruction states for it nowhere."""
    found = []
    for named, counts in wide.rows.items():
        file = output_table(outputs, named)
        if file is not None and file not in outputs.incomplete and outputs.rows.get(file, 0) < MAX_ROWS and \
                outputs.rows.get(file, 0) not in counts:
            found.append(("stated_rows", file, f"instruction: `{name_shown(file)}` holds another number of rows than "
                                               "the task states for it"))
    return found


def bounds_broken(wide: Wide, outputs: Outputs) -> list[tuple[str, str, str]]:
    """`abs_bound`: a value under a name the instruction bounds in absolute value (a JSON key's, a table's column's)
    past the bound: `<` at it or past it, `<=` past it beyond `TOLERANCE`. A line selecting by a bound (`SELECTING`:
    keep, drop, clip ...) states none."""
    found = []
    for name, op, bound, _ in wide.bounds:
        slack = 0.0 if op == "<" else TOLERANCE * max(1.0, abs(bound))
        for file in [*outputs.json, *outputs.headers]:
            if file in outputs.json:
                values = [scalar(value) for path, value in leaves(outputs.json[file]) if path and path[-1] == name]
            else:
                held = table_columns(outputs, file).get(name)
                values = list(held.numbers) if held is not None else []
            if any(value is not None and math.isfinite(value) and (abs(value) >= bound if op == "<" else
                                                                  abs(value) > bound + slack) for value in values):
                found.append(("abs_bound", f"{file}|{name}", f"instruction: `{name_shown(name)}` in "
                                                             f"`{name_shown(file)}` breaks the bound the task states "
                                                             "on its absolute value"))
    return found


def twins(keys: list[str]) -> list[tuple[str, str]]:
    """An object's with/without twins: `x` and `x_no_y` (or `x_without_y`), `x_with_y` and `x_no_y`, by their words
    between underscores, case aside."""
    found = []
    split = {key: key.lower().split("_") for key in keys}
    for a in keys:
        for b in keys:
            wa, wb = split[a], split[b]
            if a == b:
                continue
            if len(wb) == len(wa) + 2 and wb[:len(wa)] == wa and wb[len(wa)] in WITHOUT:
                found.append((a, b))
            elif len(wa) == len(wb) and "with" in wa:
                k = wa.index("with")
                if wb[:k] == wa[:k] and wb[k] in WITHOUT and wb[k + 1:] == wa[k + 1:]:
                    found.append((a, b))
    return found


def twins_equal(outputs: Outputs) -> list[tuple[str, str, str]]:
    """`twin_equal`: a JSON object whose with/without twins, two values or more compared (a pair of numbers, or two
    objects' or lists' numbers in order), all hold one value."""
    found = []
    for file, doc in outputs.json.items():
        for obj in objects(doc):
            pairs = twins(list(obj))
            same = total = 0
            for a, b in pairs:
                va, vb = obj[a], obj[b]
                compared = twin_numbers(va, vb)
                if not compared:
                    total = same = 0
                    break
                total += len(compared)
                same += sum(alike(x, y) for x, y in compared)
            if total >= 2 and same == total:
                a, b = pairs[0]
                found.append(("twin_equal", f"{file}|{a}", f"`{name_shown(file)}`: each `{name_shown(a)}` equals its "
                                                           f"`{name_shown(b)}`, which the task computes under other "
                                                           f"inputs"))
    return found


def twin_numbers(a: object, b: object) -> list[tuple[float, float]] | None:
    """Comparable numeric leaves at the same fields and list positions; differing structure or text is no twin."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return None
        pairs = [twin_numbers(a[key], b[key]) for key in a]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b) or len(a) > MAX_ITEMS:
            return None
        pairs = [twin_numbers(x, y) for x, y in zip(a, b)]
    elif isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return None
    else:
        x, y = scalar(a), scalar(b)
        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            return [(x, y)]
        return [] if a == b else None
    return None if any(pair is None for pair in pairs) else [p for pair in pairs for p in pair]


def alike(x: float, y: float) -> bool:
    return abs(x - y) <= TWIN_TOLERANCE * max(1.0, abs(x))


def compact(name: str) -> str:
    """A name up to case and separators."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def respelled(wide: Wide, outputs: Outputs) -> list[tuple[str, str, str]]:
    """`key_spelling`: a name the instruction writes (not a file's) that no output holds as a JSON key or a column,
    where an output holds a name equal to it up to case and separators that the instruction never writes."""
    held: dict[str, str] = {}               # each name the output holds, the first file holding it
    for file, doc in outputs.json.items():
        for obj in objects(doc):
            for key in obj:
                held.setdefault(str(key), file)
    for file, header in outputs.headers.items():
        for column in header:
            if column:
                held.setdefault(column, file)
    by_compact: dict[str, list[str]] = {}
    for name in held:
        if compact(name):
            by_compact.setdefault(compact(name), []).append(name)
    found = []
    for wanted in sorted(wide.named):
        if wanted in held or FILE_LIKE.search(wanted):
            continue
        theirs = sorted(name for name in by_compact.get(compact(wanted), ()) if name != wanted and
                        name not in wide.named)
        if theirs:
            file = held[theirs[0]]
            found.append(("key_spelling", f"{file}|{wanted}", f"instruction: `{name_shown(file)}` writes "
                                                              f"`{name_shown(theirs[0])}` where the task names "
                                                              f"`{name_shown(wanted)}`"))
    return found


def unnamed(outputs: Outputs) -> list[tuple[str, str, str]]:
    """`unnamed_index`: a table holding rows whose header holds a column with no name, or pandas' `Unnamed: N`."""
    return [("unnamed_index", file, f"`{name_shown(file)}` has a column with no name, an index written without its "
                                    "label; write it under the name the task gives")
            for file, header in outputs.headers.items()
            if outputs.rows.get(file) and any(UNNAMED.match(column) for column in header)]


def group_counts(wide: Wide, outputs: Outputs) -> list[tuple[str, str, str]]:
    """`group_count`: only an explicit unfiltered count of a named table's rows with a named column's label.
    A shared label alone cannot bind a filtered or distinct count to the table's rows."""
    if wide.group_json not in outputs.json:
        return []
    found = []
    for key, named, column, label in wide.groups:
        table = output_table(outputs, named)
        if table is None or table in outputs.incomplete or outputs.rows.get(table, MAX_ROWS) >= MAX_ROWS:
            continue
        held = table_columns(outputs, table).get(column)
        if held is None or not held.texts or outputs.headers[table].count(column) != 1:
            continue
        count = sum(text == label for text in held.texts)
        for file, doc in ((wide.group_json, outputs.json[wide.group_json]),):
            value = doc.get(key) if isinstance(doc, dict) else None
            if not isinstance(value, int) or isinstance(value, bool) or value == count:
                continue
            found.append(("group_count", f"{file}|{key}", f"instruction: `{name_shown(key)}` in `{name_shown(file)}` "
                                                          "differs from the table's rows for its group"))
    return found


def wide_found(wide: Wide | None, outputs: Outputs) -> list[tuple[str, str, str]]:
    """The six rules' findings over the output (`WIDE_RULES`' order), each (rule, what it is known by across runs: the
    rule, the file and the name, its line as said); the four that read the instruction said as `instruction:` lines,
    as lane rules1's are; at most `PER_RULE` a rule and file. A rule that cannot read the output holds none."""
    wide = wide if wide is not None else Wide()
    found = []
    for rule in (lambda: rows_unstated(wide, outputs), lambda: bounds_broken(wide, outputs),
                 lambda: twins_equal(outputs), lambda: respelled(wide, outputs), lambda: unnamed(outputs),
                 lambda: group_counts(wide, outputs)):
        try:
            found += rule()
        except Exception:  # noqa: BLE001 - a rule's fault never costs the run its observation
            continue
    said, each, lines = [], Counter(), set()
    for name, known, line in found:
        file = known.split("|", 1)[0]
        line = line[:FINDING_CHARS + (len("instruction: ") if line.startswith("instruction: ") else 0)]
        if each[(name, file)] < PER_RULE and line not in lines:
            each[(name, file)] += 1
            lines.add(line)
            said.append((name, f"{name}|{known}", line))
    return said


# ------------------------------------------------------------------------------------------------- the audit


AUDIT_CHARS = 300               # an audit's line, at most
AUDIT_NAMES = 24                # the keys or columns an audit's line names, each list, at most
AUDIT_LIST = 180                # and their characters, each list, at most: the rest counted, never cut mid-name


def audit(stated_: Stated, out: Path, named: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """The review's reading of the output against the instruction (lane loop1), one line each: every property,
    `held` with what the output holds of it, `broken` with its finding's line, or `unchecked` where no output name
    meets it or its values cannot be read for its kind; then the output files the instruction names (`named`), each
    present or absent, and each file's keys and columns it names, each present or absent (a file absent or unread
    said so)."""
    outputs = Outputs(out)
    lines = [audited(prop, outputs, stated_.numbers)[:AUDIT_CHARS] for prop in stated_.props]
    return lines, [line[:AUDIT_CHARS] for line in contract_audit(stated_, outputs, Path(out), tuple(named))]


def audited(prop: Prop, outputs: Outputs, exempt: frozenset) -> str:
    """One property's line (`audit`)."""
    said = f"\"{prop.said}\" ({prop.describe().split(': ', 1)[1]})"
    if prop.kind in TABLE_RULES:            # lane rules1's kinds over a table: the tables they read
        broke = TABLE_RULES[prop.kind](prop, outputs)
        if broke:
            return f"broken: {broke[0]}"
        read = TABLE_READS[prop.kind](prop, outputs)
        return f"held: {said}; read in {', '.join(read)}" if read else \
            f"unchecked: {said}; no output table holds its columns"
    if prop.kind == "sum" and len(prop.names) > 1:
        broke = rows_summing(prop, outputs)
        if broke:
            return f"broken: {broke[0]}"
        by_file: dict[str, set[str]] = {}           # as `rows_summing` reads them: top-level names alone
        for held in outputs.values:
            for name in prop.names:
                if meets(name, held.name, False) and held.name == held.name.rsplit(".", 1)[-1]:
                    by_file.setdefault(held.file, set()).add(name)
        whole = sorted(file for file, found in by_file.items() if len(found) == len(prop.names))
        return f"held: {said}; read in {', '.join(whole)}" if whole else \
            f"unchecked: {said}; no output file holds every one of {names_of(prop)}"
    if prop.kind == "decimals" and prop.scope:
        scoped = [held for held in outputs.values if not (prop.file and held.file != prop.file) and not (
            prop.scope == "json" and held.table) and not (prop.scope == "table" and not held.table)]
        for held in scoped:
            broke = violated(prop, held, exempt)
            if broke:
                return f"broken: {said_of(prop, held, broke)}"
        read = [held for held in scoped if readable(prop, held, exempt)]
        if not read:
            return f"unchecked: {said}; no value of the output it covers reads as a number"
        count = sum(len([n for n in held.numbers if n is not None and math.isfinite(n) and n not in exempt])
                    for held in read)
        return f"held: {said}; {count} value(s) read in {', '.join(sorted({held.file for held in read}))}"
    found = outputs.find(prop)
    if not found:
        return f"unchecked: {said}; no output key or column meets {names_of(prop)}"
    for held in found:
        broke = violated(prop, held)
        if broke:
            return f"broken: {said_of(prop, held, broke)}"
    read = [held for held in found if readable(prop, held)]
    if not read:
        return f"unchecked: {said}; {found[0].file}: `{found[0].name}` holds no value this reads"
    return f"held: {said}; " + "; ".join(read_of(prop, held) for held in read[:2]) + (
        f"; and {len(read) - 2} more" if len(read) > 2 else "")


def names_of(prop: Prop) -> str:
    return ", ".join(f"`{name}`" for name in prop.names) or "its name"


def readable(prop: Prop, values: Values, exempt: frozenset = frozenset()) -> bool:
    """Whether `violated` can read `values` for `prop` at all: its None otherwise means nothing was read."""
    numbers = [n for n in values.numbers if n is not None and math.isfinite(n)]
    if prop.kind == "decimals":
        return any(n not in exempt for n in numbers)
    if not numbers:
        return False
    if prop.kind in ("gt", "ge", "lt", "le", "eq", "between", "fraction"):
        return True
    if prop.kind == "start":
        return values.ordered and values.numbers[0] is not None
    if prop.kind in ("increasing", "decreasing"):
        return values.ordered and len(numbers) > 1
    return prop.kind == "sum" and not values.table and len(prop.names) == 1


def read_of(prop: Prop, values: Values) -> str:
    """What the output holds under a name, for a property it holds: one value, or the count and the range."""
    numbers = [n for n in values.numbers if n is not None and math.isfinite(n)]
    where = f"{values.file}: {'column' if values.table else 'key'} `{values.name}`"
    if prop.kind == "start":
        return f"{where} starts at {shown(numbers[0])}"
    if prop.kind == "sum":
        return f"{where} sums to {shown(math.fsum(numbers))}"
    if len(numbers) == 1:
        return f"{where} = {shown(numbers[0])}"
    return f"{where}: {len(numbers)} values in [{shown(min(numbers))}, {shown(max(numbers))}]"


def listed(names: list[str]) -> str:
    """The names, each whole, within `AUDIT_NAMES` and `AUDIT_LIST` characters, the rest counted."""
    kept: list[str] = []
    for name in names[:AUDIT_NAMES]:
        if kept and len(", ".join([*kept, f"`{name}`"])) > AUDIT_LIST:
            break
        kept.append(f"`{name}`")
    return ", ".join(kept) + (f" and {len(names) - len(kept)} more" if len(names) > len(kept) else "")


def present(what: str, wanted: list[str], have: set[str] | list[str]) -> str:
    """`<what> (N): present ...; absent ...`, the names in the instruction's order."""
    here = [name for name in wanted if name in have]
    gone = [name for name in wanted if name not in have]
    return f"{what} ({len(wanted)}): " + "; ".join(part for part in (
        f"present {listed(here)}" if here else "", f"absent {listed(gone)}" if gone else "") if part)


def contract_audit(stated_: Stated, outputs: Outputs, out: Path, named: tuple[str, ...]) -> list[str]:
    """The named output files and the keys and columns the instruction names, each present or absent (`audit`)."""
    lines = [present("the output files the task names", list(named), {name for name in named
                                                                     if (out / name).is_file()})] if named else []
    for file, wanted in stated_.keys.items():
        if file in outputs.json:
            lines.append(present(f"{file} keys", wanted, {key for key in wanted if has_key(outputs.json[file], key)}))
        else:
            lines.append(f"{file} keys ({len(wanted)}): unread, the file " + (
                "is absent" if not (out / file).is_file() else "does not read as JSON"))
    for file, wanted in stated_.columns.items():
        if file in outputs.headers:
            lines.append(present(f"{file} columns", wanted, set(outputs.headers[file])))
        else:
            lines.append(f"{file} columns ({len(wanted)}): unread, the file " + (
                "is absent" if not (out / file).is_file() else "does not read as a table"))
    return lines


# ------------------------------------------------------------------------------- the output contract (lane contract1)

TERM_KINDS = ("keys", "columns", "decimals", "na", "type", "labels", "nested", "rows")
FORMATS = {".json": "json", ".jsonl": "jsonl", ".csv": "csv", ".tsv": "tsv", ".parquet": "parquet",
           ".pqt": "parquet", ".txt": "txt", ".md": "md", ".html": "html", ".xlsx": "xlsx"}
TABLE_FORMATS = ("csv", "tsv", "parquet")
LIST_WHAT = re.compile(r"\b(?:keys?|columns?|fields?|headers?|structure)\b", re.I)
EXACT_SET = re.compile(r"\b(?:exactly|only)\s+(?:these|the\s+following|this|the|\d+|" + "|".join(NUMBER_WORDS)
                       + r")\s+(?:top-level\s+)?(?:keys|columns|fields|structure)\b"
                       r"|\bno\s+(?:other|extra|additional)\s+(?:keys|columns|fields)\b"
                       r"|\b(?:top-level\s+)?(?:keys|columns)\s+must\s+be\b(?=\s*:?\s*`)", re.I)
ORDER_SET = re.compile(r"\b(?:in\s+)?(?:this|the\s+listed|the\s+following|the\s+exact|exactly\s+(?:this|the\s+listed|"
                       r"the))\s+(?:exact\s+)?order\b|\bexact\s+order\b|\bin\s+order\s*:|\breordered\b", re.I)
ABOVE = re.compile(r"\b(?:above|shown\s+before)\b", re.I)
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\*\*)?`(?P<name>[^`\n]{1,80})`(?:\*\*)?")
INLINE_LIST = re.compile(r"^`(?P<names>[^`\n]+,[^`\n]+)`[.:]?$")
COLUMN_NAME = re.compile(r"^[^\s`,]{1,80}$")
KEY_NAME = re.compile(r"^[A-Za-z_][\w\-.]{0,79}$")
TEMPLATE_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|<[^<>\n]*>|[{}\[\]:,]|\.\.\.|…|[^\s{}\[\]:,"<>]+')
SPEC_TYPES = (("bool", re.compile(r"\bbool(?:ean)?\b|\btrue\s*[|/]\s*false\b", re.I)),
              ("int", re.compile(r"\bint(?:eger)?(?:64|32)?\b", re.I)),
              ("number", re.compile(r"\b(?:float(?:64)?|number|double|numeric|real)\b", re.I)),
              ("str", re.compile(r"\bstr(?:ing)?\b|\bdate(?:time)?\b|YYYY|\bticker\b|\bname\b|\bidentifier\b", re.I)),
              ("list", re.compile(r"\blist\b|\barray\b", re.I)),
              ("object", re.compile(r"\bdict\b|\bobject\b|\bmapping\b", re.I)))
EMPTY_WORD = re.compile(r"\b(?:as|leave|leaves|left|write|written\s+as|use|uses|set\s+to|be|remain|remains|stay)\s+"
                        r"(?:it\s+|them\s+|the\s+cells?\s+)?(?:an?\s+)?(?:empty|blank)\b(?!-)|\bempty\s+strings?\b"
                        r"|\bblank\s+cells?\b", re.I)
NAN_WORD = re.compile(r"\b(?i:to|as|be|report|write|written\s+as|use|uses|have|has)\s+`?NaN`?(?![\w-])"
                      r"|`?NaN`?\s+(?i:if|when|where)\s+(?i:it\s+is\s+)?(?i:not\s+available|unavailable|missing|"
                      r"not\s+applicable|absent|undefined)\b")      # the word `NaN` as written, its verbs in any case
NULL_WORD = re.compile(r"\b(?:use|uses|report|reported\s+as|write|written\s+as|be|as|set\s+to|return|returns)\s+"
                       r"(?:a\s+)?(?:JSON\s+)?`?null`?(?![\w-])", re.I)
BOOLEAN_WORDS = re.compile(r"`(true|false|True|False|TRUE|FALSE)`")     # a table column's two literal words
LABEL_SET = re.compile(r"\b(?:is|are|must\s+be|should\s+be|takes?|holds?)\s+(?:exactly\s+)?one\s+of\b|^\W*one\s+of\b"
                       r"|\b(?:labels?|values?)\s+(?:are|is|must\s+be)\b|\bvalid\s+(?:labels?|values?)\b", re.I)
LABEL_PAIR = re.compile(r"`(?P<subject>[^`]+)`\s+(?:is|must\s+be|should\s+be)\s+(?:either\s+)?`(?P<a>[^`]+)`\s+or\s+"
                        r"`(?P<b>[^`]+)`")
IN_ORDER = re.compile(r"\bin\s+this\s+order\b", re.I)
SORT_BY = re.compile(r"\b(?:sort(?:ed)?|order(?:ed)?)\s+(?:the\s+)?(?:rows\s+)?(?P<way>alphabetically\s+|chronologically\s+|"
                     r"(?:in\s+)?(?:ascending|descending)\s+(?:order\s+)?)?(?:by|on)\s+(?P<keys>[^.;:()\n]+)"
                     r"|\bin\s+order\s+\((?P<inner>[^)]+)\)"
                     r"|\bsort(?:ed)?\s+(?:the\s+)?rows\s+in\s+(?P<dir>ascending|descending)\s+`(?P<col>[^`]+)`"
                     r"|\bin\s+(?P<dir2>ascending|descending)\s+`(?P<col2>[^`]+)`\s+order\b", re.I)
SORT_KEY = re.compile(r"^\s*(?:`(?P<col>[^`]{1,80})`|(?P<bare>[A-Za-z_][\w\-]{0,79}))\s*\(?(?P<dir>ascending|descending|"
                      r"asc|desc)?\)?\s*$", re.I)
SORT_SPLIT = re.compile(r",?\s+then\s+(?:by\s+)?|,\s*and\s+|,\s*|\s+and\s+", re.I)
LISTED_ROWS = re.compile(r"\brows?\s+(?:must\s+|should\s+)?(?:appear|are|be|come)\s+in\s+(?:this|the\s+following)\s+"
                         r"(?:exact\s+)?(?:\w+\s+)?order\b|\bsort\s+(?:the\s+)?rows\s+in\s+this\s+(?:exact\s+)?order\b",
                         re.I)
WAIVER = re.compile(r"\b(?:harmless|tolerat\w*|semantic\s+rather\s+than\s+byte|not\s+byte-exact|any\s+reasonable\s+"
                    r"precision|need\s+not|do(?:es)?\s+not\s+need|(?:is|are)\s+ignored|do(?:es)?\s+not\s+matter)\b", re.I)
WAIVED = (("rows", re.compile(r"\brow\s+order\b", re.I)), ("order", re.compile(r"\b(?:key|column)\s+order\b", re.I)),
          ("decimals", re.compile(r"\bprecision\b|\bdecimals?\b|\btrailing\s+zeros?\b", re.I)),
          ("na", re.compile(r"\bblank\b|\bnull\b|\bempty\b", re.I)))


@dataclass(frozen=True)
class Term:
    """One term of an output file's contract as the instruction states it (`terms`): its kind (`TERM_KINDS`), the
    output file, and what the text fixes. `keys` and `columns`: the names in the text's order, under `path` (a JSON
    object's, () the top level), `exact` where it says exactly or only, `ordered` where it fixes their order.
    `decimals`: the names or the `scope` ("json", "table", "all") rounded to `value` decimals, `strict` where exactly
    (today's parse, `decimals`). `na`: the word for a missing cell of `names` ("" for empty, "NaN"), or a JSON file's
    "null". `type`: the native type of the value at `path` (`int`, `number`, `bool`, `str`, `list`, `object`).
    `labels`: the values the one column or key of `names` may hold (`value`), two boolean words among such sets.
    `nested`: a JSON value's whole path. `rows`: the sort
    columns (`names`, `value` each one's direction) or the listed values' order (`value` "listed")."""
    kind: str
    file: str
    names: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    exact: bool = False
    ordered: bool = False
    value: object = None
    strict: bool = False
    scope: str = ""
    said: str = ""

    def describe(self) -> str:
        """The term in a short line (the offline read's, the replay's and the tests')."""
        where = f"{self.file}" + (f":{'.'.join(self.path)}" if self.path else "")
        if self.kind in ("keys", "columns"):
            flags = "/".join(flag for flag, on in (("exact", self.exact), ("ordered", self.ordered)) if on)
            return f"{where}: {self.kind} {', '.join(self.names)} [{flags}]"
        if self.kind == "decimals":
            who = ", ".join(self.names) or f"[{self.scope}]"
            return f"{where}: {who} {'at exactly' if self.strict else 'rounded to'} {self.value} decimals"
        if self.kind == "na":
            return f"{where}: {', '.join(self.names) or '[every value]'} missing as {self.value!r}"
        if self.kind == "type":
            return f"{where}: {self.value}"
        if self.kind == "labels":
            return f"{where}: {', '.join(self.names)} one of {', '.join(map(str, self.value))}"
        if self.kind == "rows":
            return f"{where}: rows " + ("in the listed order of " + ", ".join(self.names) if self.value == "listed"
                                        else "sorted by " + ", ".join(f"{name} {way}" for name, way in
                                                                      zip(self.names, self.value)))
        return f"{where}: present"


@dataclass
class Terms:
    """An instruction's output contract: each named output file's format, the terms, the keys and columns today's
    lines already read by file (`Stated.keys` and `Stated.columns`, and lane rules1's inline headers where its rules
    are read), which the check leaves to them, and the check's kinds the text itself waives ("harmless differences in
    ... row order")."""
    files: dict[str, str] = field(default_factory=dict)
    terms: list[Term] = field(default_factory=list)
    today: dict[str, frozenset] = field(default_factory=dict)
    waived: frozenset = frozenset()


@dataclass
class Piece:
    kind: str           # heading, text (a paragraph), item (a bullet with its continuation), code, row
    text: str
    file: str           # the output file of its section, or the one file it names itself
    lines: tuple[str, ...] = ()
    lang: str = ""      # code: the fence's language


def layout(instruction: str, named: tuple[str, ...]) -> list[Piece]:
    """The instruction as pieces in their order, each with its output file as `segment` assigns it: a heading's, else
    a paragraph's that names one file (until the next heading), else the one file a piece names itself."""
    lines = (instruction or "").replace("\r\n", "\n").split("\n")
    pieces: list[Piece] = []
    levels: list[tuple[int, str]] = []
    local = ""
    current: list[str] = []
    kind = ""

    def section() -> str:
        return local or next((file for _, file in reversed(levels) if file), "")

    def flush() -> None:
        nonlocal current, kind, local
        text = " ".join(part.strip() for part in current if part.strip())
        if text:
            files = files_in(text, named)
            pieces.append(Piece(kind, text, files[0] if len(files) == 1 else section()))
            if kind == "text" and len(files) == 1:
                local = files[0]
        current, kind = [], ""

    k = 0
    while k < len(lines):
        line = lines[k]
        fence = FENCE.match(line)
        if fence:
            flush()
            lang = line.strip()[3:].strip().split(" ")[0].lower()
            body = []
            k += 1
            while k < len(lines) and not lines[k].strip().startswith(fence.group(1)):
                body.append(lines[k])
                k += 1
            pieces.append(Piece("code", "\n".join(body), section(), tuple(body), lang))
            k += 1
            continue
        heading = HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            levels = [(lv, file) for lv, file in levels if lv < level]
            files = files_in(heading.group(2), named)
            levels.append((level, files[0] if files else ""))
            local = ""
            pieces.append(Piece("heading", heading.group(2).strip(), section()))
        elif not line.strip():
            flush()
        elif line.lstrip().startswith("|"):
            flush()
            cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
            if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
                pieces.append(Piece("row", line.strip(), section(), cells))
        elif BULLET.match(line):
            flush()
            current, kind = [line.strip()], "item"
        elif current:
            current.append(line)
        else:
            current, kind = [line], "text"
        k += 1
    flush()
    return pieces


def format_of(file: str) -> str:
    return FORMATS.get(Path(file).suffix.lower(), "")


def header_of(lines: tuple[str, ...]) -> list[str]:
    """A code block's header: its first line, joined with the next while a line ends on a comma; [] where a name is
    no column's (a space, a backtick) or fewer than two."""
    rows = [line.strip() for line in lines if line.strip()]
    if not rows:
        return []
    text = rows[0]
    for row in rows[1:]:
        if not text.endswith(","):
            break
        text += row
    names = [name.strip().strip("`") for name in text.split(",")]
    return names if len(names) >= 2 and all(COLUMN_NAME.match(name) for name in names) else []


def template(text: str):
    """A JSON template block (`{"a": <float>, "b": {"value": <int>}}`, `...` for more) as a tree: ("object", [(key,
    node)], open), ("list", item or None, open) or ("leaf", spec, quoted), `spec` a placeholder's words (None for a
    literal); None where the block is no template."""
    text = re.sub(r"(?m)(^|\s)//[^\n]*$", r"\1", text)
    tokens = TEMPLATE_TOKEN.findall(text)
    if not tokens or tokens[0] not in ("{", "["):
        return None

    def value(k: int):
        token = tokens[k]
        if token == "{":
            items, opened, k = [], False, k + 1
            while k < len(tokens) and tokens[k] != "}":
                if tokens[k] in ("...", "…", ","):
                    opened |= tokens[k] != ","
                    k += 1
                    continue
                if not tokens[k].startswith('"') or k + 1 >= len(tokens) or tokens[k + 1] != ":":
                    raise ValueError("no key")
                key = json.loads(tokens[k])
                node, k = value(k + 2)
                items.append((key, node))
            if k >= len(tokens):
                raise ValueError("unclosed")
            return ("object", items, opened), k + 1
        if token == "[":
            item, opened, k = None, False, k + 1
            while k < len(tokens) and tokens[k] != "]":
                if tokens[k] in ("...", "…", ","):
                    opened |= tokens[k] != ","
                    k += 1
                    continue
                node, k = value(k)
                item = item or node
            if k >= len(tokens):
                raise ValueError("unclosed")
            return ("list", item, opened), k + 1
        if token.startswith("<"):
            return ("leaf", token[1:-1], False), k + 1
        if token.startswith('"'):
            inner = json.loads(token)
            quoted = re.fullmatch(r"<([^<>]*)>", inner)
            return ("leaf", quoted.group(1) if quoted else None, True), k + 1
        if token in ("}", "]", ":"):
            raise ValueError("misplaced")
        return ("leaf", None, False), k + 1

    try:
        node, _ = value(0)
    except (ValueError, IndexError, json.JSONDecodeError, RecursionError):
        return None
    return node


def spec_type(spec: str | None, quoted: bool) -> str | None:
    """A placeholder's type (`SPEC_TYPES`): the one its words name (null aside); a quoted placeholder naming none a
    string; None where it names none or two."""
    if spec is None:
        return None
    named = [kind for kind, pattern in SPEC_TYPES if pattern.search(spec)]
    if "number" in named and "int" in named:
        named.remove("int")         # `<float, an int count of decimals>` reads as its first word's
    if len(named) == 1:
        return named[0]
    return "str" if quoted and not named else None


def key_name(key: str) -> bool:
    return bool(KEY_NAME.match(key)) and key.lower() not in PLACEHOLDERS


def walk(node, path: tuple[str, ...]):
    """A template's (path, node) pairs, depth first; no paths under placeholder keys, or numeric example keys in
    an open map. Closed objects keep their literal numeric keys."""
    yield path, node
    if node[0] == "object":
        if not all(key_name(key) for key, _ in node[1]):
            return
        for key, inner in node[1]:
            if node[2] and re.search(r"\d", key):
                continue
            yield from walk(inner, (*path, key))
    elif node[0] == "list" and node[1] is not None:
        yield from walk(node[1], (*path, "[]"))


def template_terms(node, file: str, said: str) -> list[Term]:
    """A JSON template's terms: each placeholder's native type (`spec_type`) and each value's whole path two keys deep
    or more (`nested`)."""
    found = []
    for path, inner in walk(node, ()):
        keys = [part for part in path if part != "[]"]
        if inner[0] == "leaf":
            kind = spec_type(inner[1], inner[2])
            if kind is not None and keys:
                found.append(Term("type", file, path=path, value=kind, said=said))
        elif inner[0] == "list" and keys and inner[1] is not None and inner[1][0] == "leaf":
            found.append(Term("type", file, path=path, value="list", said=said))
        if len(keys) >= 2 and (inner[0] == "leaf" or inner[0] == "list" and (inner[1] is None or inner[1][0] == "leaf")):
            found.append(Term("nested", file, path=path, said=said))
    return found


def object_at(node, path: tuple[str, ...]):
    """The template's object at `path` (its top level at ()), or None."""
    for key in path:
        if node is None or node[0] != "object":
            return None
        node = dict(node[1]).get(key)
    return node if node is not None and node[0] == "object" else None


def lead_list(pieces: list[Piece], k: int, lead_end: int, file: str, last_code: dict[str, Piece]):
    """The names a lead (`pieces[k]`, its matched words ending at `lead_end`) lists for `file`: its own backticked
    names after a colon past the words; else the block above (where it says so) or the next piece's: a code block (a
    JSON template's keys, a table's header), an inline header, consecutive list items each opening on a backticked name,
    or a table's first cells. (names, the template node or None); ([], None) where none reads whole."""
    piece = pieces[k]
    rest = piece.text[lead_end:]
    stop = re.search(r"\.(?:\s|$)", rest)
    inline = [m.group(1).strip('"') for m in re.finditer(r"`([^`\n]{1,80})`", rest[:stop.start() if stop else None])]
    inline = [name for name in inline if name.lower() not in TYPES]
    if len(inline) >= 2:
        return inline, None
    if ABOVE.search(piece.text):
        block = last_code.get(file)
        return (block_names(block, file) if block is not None else ([], None))
    nxt = k + 1
    if nxt >= len(pieces):
        return [], None
    after = pieces[nxt]
    if after.kind == "code":
        return block_names(after, file)
    if after.kind == "text":
        found = INLINE_LIST.match(after.text.strip())
        if found:
            names = [name.strip() for name in found.group("names").split(",")]
            return (names, None) if all(COLUMN_NAME.match(name) for name in names) else ([], None)
        return [], None
    if after.kind == "item":
        names = []
        for each in pieces[nxt:]:
            if each.kind != "item":
                break
            found = LIST_ITEM.match(each.text)
            if not found:
                return [], None
            names.append(found.group("name").strip())
        return names, None
    if after.kind == "row":
        names, header = [], True
        for each in pieces[nxt:]:
            if each.kind != "row":
                break
            first = each.lines[0].strip(" *") if each.lines else ""
            if header and first.strip("`").lower() in ("column", "columns", "field", "key", "name", "col"):
                header = False
                continue
            header = False
            ticked = re.fullmatch(r"`([^`]+)`", first)
            if not ticked:
                return [], None
            names.append(ticked.group(1))
        return names, None
    return [], None


def block_names(block: Piece, file: str):
    """A code block's names for `file`: a JSON template's top-level keys (a list of records' keys under `[]`), the
    template with them; a table's header."""
    if format_of(file) == "json":
        node = template(block.text)
        if node is None:
            return [], None
        top = node if node[0] == "object" else node[1] if node[0] == "list" else None
        if top is None or top[0] != "object" or not all(key_name(key) for key, _ in top[1]):
            return [], None
        return [key for key, _ in top[1]], node
    if format_of(file) in TABLE_FORMATS:
        return header_of(block.lines), None
    return [], None


def sort_keys(text: str, known=()) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """The sort a sentence states over columns (`sorted by \\`a\\` ascending, then \\`b\\` descending`; `sorted
    alphabetically by ticker` where `ticker` is one of the file's `known` columns): (columns, directions), ascending
    where unsaid (descending where the words before "by" say so); None where a key is neither a backticked name nor a
    known column."""
    for found in SORT_BY.finditer(text):
        if found.group("col") or found.group("col2"):
            way = (found.group("dir") or found.group("dir2")).lower()
            return (found.group("col") or found.group("col2"),), ("descending" if way.startswith("desc") else
                                                                  "ascending",)
        keys = found.group("keys") or found.group("inner") or ""
        parts = [part for part in SORT_SPLIT.split(keys.strip()) if part.strip()]
        read = [SORT_KEY.match(part) for part in parts]
        if not parts or not all(read) or any(each.group("bare") and each.group("bare") not in known for each in read):
            continue
        first = "descending" if "descending" in (found.group("way") or "").lower() else "ascending"
        return (tuple(each.group("col") or each.group("bare") for each in read),
                tuple(("descending" if each.group("dir").lower().startswith("desc") else "ascending")
                      if each.group("dir") else first for each in read))
    return None


def na_word(text: str) -> str | None:
    """The one word a sentence asks for a missing value: "" (empty), "NaN" or "null"; None where it asks none or more
    than one (`blank or null`)."""
    words = [word for word, pattern in (("", EMPTY_WORD), ("NaN", NAN_WORD), ("null", NULL_WORD))
             if pattern.search(text)]
    if len(words) != 1 or re.search(r"\b(?:blank|empty)\s+or\s+(?:null|NaN)\b|\b(?:null|NaN)\s+or\s+(?:blank|empty|"
                                    r"null|NaN)\b", text, re.I):
        return None
    return words[0]


def subjects(piece: Piece) -> list[str]:
    """The names a piece is about: its head's (a bullet's `name:`), else its backticked names that are no type or
    literal word."""
    bullet = BULLET.match(piece.text)
    head, _ = heads(bullet.group(2) if bullet else piece.text)
    if head:
        return list(head)
    return [name for name in (m.group(1) for m in TICKED.finditer(piece.text))
            if name.lower() not in TYPES and not files_in(name, ())]


def label_terms(pieces: list[Piece], k: int, file: str) -> list[Term]:
    """A piece's stated value sets ("x is one of a, b", "the only valid x labels are:" and the items after it, "- x:
    one of ...", "x is A or B", each name and value in backticks): a `labels` term for x, and where the text adds "in
    this order" a `rows` term for the values listed. None where a value reads as a number or the subject is one of
    them."""
    piece = pieces[k]
    found = []
    bullet = BULLET.match(piece.text)
    head, body = heads(bullet.group(2) if bullet else piece.text)
    for sentence in sentences(body if head else piece.text):
        values: list[str] = []
        subject = ""
        pair = LABEL_PAIR.search(sentence)
        match = LABEL_SET.search(sentence)
        if pair:
            subject, values = pair.group("subject"), [pair.group("a"), pair.group("b")]
        elif match:
            before = TICKED.findall(sentence[:match.start()])
            subject = before[-1] if before else head[0] if len(head) == 1 else ""
            values = [m.group(1).strip('"') for m in re.finditer(r"`([^`\n]{1,80})`", sentence[match.end():])]
            if not values and sentence.rstrip().endswith(":"):
                for each in pieces[k + 1:]:
                    item = LIST_ITEM.match(each.text) if each.kind == "item" else None
                    if not item or each.text[item.end():].strip(" .;,"):
                        break
                    values.append(item.group("name").strip('"'))
        values = list(dict.fromkeys(values))
        if not subject or len(values) < 2 or subject in values or any(number(value) is not None for value in values):
            continue
        said = quoted(normal(sentence), len(normal(sentence)))
        found.append(Term("labels", file, (subject,), value=tuple(values), said=said))
        if IN_ORDER.search(sentence[match.end() if match else 0:]) and format_of(file) in TABLE_FORMATS:
            found.append(Term("rows", file, tuple(values), value="listed", said=said))
    return found


def terms(instruction: str, named: tuple[str, ...] = (), stated_: Stated | None = None) -> Terms:
    """The output contract the instruction states (lane contract1; `Term`, above), from its text alone and only where
    the text states it: key and column lists with the words that make them exact or ordered; each JSON template's
    native types and nested paths; the word for a missing cell; a table's sort or its listed rows; the roundings
    today's parse reads (`decimals`); and the kinds the text waives. Rule 8: patterns over the text's own words, no
    unit's vocabulary."""
    named = tuple(named)
    stated_ = stated_ if stated_ is not None else stated(instruction, named)
    today: dict[str, set] = {}
    for file, names in (*stated_.keys.items(), *stated_.columns.items()):
        today.setdefault(file, set()).update(names)
    for prop in stated_.props:      # lane rules1's inline header, said where today's parse reads it (`rules`)
        if prop.kind == "header" and prop.file:
            today.setdefault(prop.file, set()).update(prop.names)
    found = Terms(files={file: format_of(file) for file in named},
                  today={file: frozenset(names) for file, names in today.items()})
    pieces = layout(instruction, named)
    last_code: dict[str, Piece] = {}
    waived: set[str] = set()
    out: list[Term] = []
    for k, piece in enumerate(pieces):
        file, kind = piece.file, format_of(piece.file)
        if piece.kind == "code":
            if kind == "json":
                node = template(piece.text)
                if node is not None:
                    out += template_terms(node, file, quoted(piece.text.replace("\n", " "), len(piece.text)))
            if file:
                last_code[file] = piece
            continue
        if piece.kind not in ("text", "item", "heading"):
            continue
        for sentence in sentences(piece.text):
            if WAIVER.search(sentence):
                waived |= {name for name, pattern in WAIVED if pattern.search(sentence)}
        if piece.kind == "heading":
            continue
        text = normal(piece.text)
        exact, ordered = EXACT_SET.search(piece.text), ORDER_SET.search(piece.text)
        if (exact or ordered) and LIST_WHAT.search(piece.text) and kind in ("json", *TABLE_FORMATS):
            end = max(match.end() for match in (exact, ordered) if match)
            names, node = lead_list(pieces, k, end, file, last_code)
            path: tuple[str, ...] = ()
            owner_ = re.match(r"^\s*(?:[-*+]\s+)?`([^`]+)`\s+(?:must|should|is|has)\b", piece.text)
            if owner_ and kind == "json" and key_name(owner_.group(1)):
                path = (owner_.group(1),)
            elif node is not None and node[0] == "list":
                path = ("[]",)
            if len(names) >= 2 and len(set(names)) == len(names):
                out.append(Term("keys" if kind == "json" else "columns", file, tuple(names), path,
                                exact=bool(exact), ordered=bool(ordered), said=quoted(text, len(text))))
        if kind in TABLE_FORMATS:
            sort = sort_keys(piece.text, stated_.columns.get(file, ()))
            if sort is not None:
                out.append(Term("rows", file, sort[0], value=sort[1], said=quoted(text, len(text))))
            if LISTED_ROWS.search(piece.text):
                values = []
                for each in pieces[k + 1:]:
                    if each.kind != "item":
                        break
                    item = LIST_ITEM.match(each.text)
                    if not item or each.text[item.end():].strip(" .;,"):
                        values = []
                        break
                    values.append(item.group("name"))
                if len(values) >= 2:
                    out.append(Term("rows", file, tuple(values), value="listed", said=quoted(text, len(text))))
            for sentence in sentences(piece.text):
                words = BOOLEAN_WORDS.findall(sentence)
                pair = tuple(dict.fromkeys(words))
                first = BOOLEAN_WORDS.search(sentence)
                before = [name for name in TICKED.findall(sentence[:first.start()]) if name not in pair] if first else []
                if len(pair) == 2 and before:
                    out.append(Term("labels", file, (before[-1],), value=pair,
                                    said=quoted(normal(sentence), len(normal(sentence)))))
        if kind in ("json", *TABLE_FORMATS):
            for term in label_terms(pieces, k, file):
                out.append(term)
        for sentence in sentences(piece.text):
            rounding = re.sub(r"\b(\d+)\s+dp\b", r"\1 decimal places", sentence, flags=re.I)
            rounding = re.sub(r"\bto the nearest basis point\b", "to 4 decimal places", rounding, flags=re.I)
            if rounding != sentence:
                for prop in decimals(rounding, file, tuple(subjects(piece))):
                    if not prop.phrase and prop.value is not None:
                        out.append(Term("decimals", file, prop.names, value=int(prop.value), strict=prop.strict,
                                        scope=prop.scope, said=sentence))
            word = na_word(normal(sentence))
            if word is None or word == "null" and kind != "json" or word != "null" and kind not in TABLE_FORMATS:
                continue
            names = () if word == "null" else tuple(name for name in subjects(Piece(piece.kind, sentence, file))
                                                    if COLUMN_NAME.match(name))
            if word == "null" or names:
                out.append(Term("na", file, names, value=word, said=quoted(normal(sentence), len(normal(sentence)))))
    for prop in stated_.props:
        if prop.kind == "decimals" and not prop.phrase and prop.value is not None:
            out.append(Term("decimals", prop.file, prop.names, value=int(prop.value), strict=prop.strict,
                            scope=prop.scope, said=prop.said))
    found.terms = list(dict.fromkeys(out))
    found.waived = frozenset(waived)
    return found
