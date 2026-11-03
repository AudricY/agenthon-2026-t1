"""The method library `qf` as N11 shows it to the House: an index once, in the system prompt, and `lib` for the rest.

The library is read from its source by `ast` (`scan`), never imported: the agent's process loads no numpy, and a
module the two library lanes add appears with no edit here. A shelf is a module of the package not named with a
leading underscore, its line the module docstring's first line; its functions are the module's public top-level
`def`s (its `__all__` where it has one), each with its signature, its docstring and its card, the docstring's first
line. A function the package's `__init__` imports, by its name or with its whole shelf (`from .<shelf> import *`), is
called `qf.<name>`, any other `qf.<shelf>.<name>`; the package imports every one (lane names1), so every call the index
and the cards write is `qf.<name>`, and the shelf a module's name alone.

- `Library.index` is the system prompt's index (lane index1; the owner's in330): a line per shelf, its line whole,
  and under it a line per function, `qf.<name>: <card>` (`qf.<shelf>.<name>` where that is the call), the card
  whole, in the shelf's order, so the House can choose beyond the pointer's matches; within `INDEX_TOKENS` of the
  model's tokens (agent/n11/measure.py), or `INDEX_BYTES` where the tokenizer does not load (lane tokens1: each bound a
  token target, its byte fallback the target at its text kind's bytes a token). Where a library outgrows it, every
  shelf line and card is cut
  to the same length at a word, the longest that fits; where that leaves a card under `MIN_CARD` characters, each
  shelf's functions are named on its own line after its shelf line, the shelf lines cut to the longest that fits,
  and `lib shelf=<module>` gives the cards. `Library.whole` counts the functions an index shows with their card
  whole (the summary's and the usage line's `index_functions`). `index.txt` beside this file is the index the
  image's build wrote (scripts/qf_index.py); without it the agent builds the same text at start.
- `Library.by_name` and `Library.by_shelf` are `lib name=<function>` and `lib shelf=<module>`: the full docstring
  under the signature and the import, or the shelf's signatures each with its card; within `LIB_BYTES`.
- `Library.find` is how a name the model writes meets a function (`resolve`, lane libalias1): the selection call's
  and the sketch's `qf.` names, `lib name=`, the grader's gold. A function's name as written, at any shelf or none;
  a shelf's name alone none; else, folded as the pointer folds an alias, a function's name, one of its aliases (every
  one on its card) or either's plural or singular; a name two functions answer to is none, as is one no rule meets.
  `Library.named` is the name alone. An alias added for the resolver alone is bracketed, so the pointer never matches
  it.
- `Library.pointers` is the one exact-name pointer: a function whose docstring's `aliases:` line names a term the
  instruction holds, matched case-insensitively with hyphens, underscores and spaces alike and never inside a longer
  word, its plural too, accents, typographic apostrophes, a possessive's 's and the British -ise spellings folded
  (`normal`, lane lib4); at most `POINTERS`, the longest alias first, and never one on an `UNPOINTED` shelf
  (conventions and I/O, whose aliases are every task's words). Only a `specific` alias is matched (lane n11c): a
  method's name or a phrase of two or more words (`Yang-Zhang`, `Curran`, `GPD`, `Hill estimator`), never one word
  written in lower case (`delta`, `cap`, `backtest`), nor an alias the line writes in brackets, a common word or
  phrase most tasks compute in their own convention (`[Sharpe ratio]`, `[OLS]`, `[implied volatility]`).
  `Function.aliases` keeps every alias, the brackets dropped: the offline read's coverage counts them all. A docstring
  with no `aliases:` line fires nothing.
- `Library.carded` is the pointed functions' cards (lanes n11b and n11d) and what was cut from them: each `Card`
  rendered from the full docstring (`Function.whole_card`): the call's signature with every argument the code checks
  against a set of names annotated with them (`Function.typed`, the code's `choice` calls read by `ast`), the
  docstring's text before its first labelled section (its description and its formula), its `formula:`,
  `conventions:`, `returns:` and `example:` sections; source and aliases left out. Within `CARD_TOKENS` (else
  `CARD_BYTES`) in all with
  `CARDS_HEAD` above them, in the pointer's order, and cut by its rank (lane cards2): while they do not fit, the last
  pointed card loses its next part (`Card.trim`: its example first, then its text's paragraphs after the formula,
  then its convention entries from the last, each entry whole, never cut mid-list, then its formula's lines from the
  last, the first kept) until it is at its smallest, and only then the card before it, so the first pointed keep
  their conventions and examples whole as long as the bound allows; a trimmed card's last line names the `lib` call
  that shows it whole. Where every card at its smallest still does not fit, the last pointed is left out and the rest
  fitted again from whole. `Library.cards` is the text alone. The loop's system prompt and the library pass after a
  clean first run carry them (lane n11c; agent/n11/solve.py), which logs what was cut and by how much.

- `made_whole` (lane first1; `AGENTHON_N11_CARDS=whole`, agent/n11/solve.py) is the first message's cards whole:
  every carded function's `Card` untrimmed after the index, within `WHOLE_TOKENS` in all (else `WHOLE_BYTES`); where
  they do not fit, the index trimmed for them (`Library.trims`, or the selection call's own section's trims), a shelf
  no carded function lies on reduced first, to its line and its functions by name; and only where the index at its
  most trimmed leaves too little, the cards cut as `carded` cuts them.

- The frame (lane libframe1: a header line and the scaffolding whole atop the index) left with lane prune2 on E102's
  reject (ROW 245, docs/research/library-design-1011.md): it lost recall.

- `rewritten` (lane lib4) is the pointed functions a House's files compute by hand: a `def` whose name holds every
  word of the function's name or of one of its specific aliases (`marks`), whose body calls no function of that name,
  or a call of a routine the function's docstring writes as a call and lists among its aliases (`logm`), in files
  that never call the function. The summary's `lib_rewritten` counts them where a pointed card reached the House.
  `rewrites` (lane called1) is the same rule naming what it finds, each def or routine with its file; `rewritten`
  reads it, and so does the called finding under `AGENTHON_N11_CALLED=on` (agent/n11/commands.py `called_said`).

Nothing here names a unit, and the instruction is only searched for the library's own aliases (rule 8).
"""
from __future__ import annotations

import ast
import difflib
import functools
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from n11 import measure

PACKAGE = "qf"
INDEX_TOKENS = 12_000           # the index's bound in the model's tokens (lane tokens1; the owner's in508): raised
#     by lane libprep1 and its review for the library the seven library lanes grow to, about 296 functions on 46
#     shelves, so that no merge re-measures it (docs/research/library-partition-1012.md, "Before the lanes open"): at
#     lane libgaps1's 39.4 tokens a function at the margin (its line 32.4, a new shelf's line 29, the lines it rewrote),
#     10,500 to 11,050 tokens whole, about 1,000 to spare. The full index is on the candidate's path wherever the
#     selection call is not sent, fails or names nothing (agent/n11/selection.py: the system prompt as without it).
#     6,800 held lane libgaps1's 185 functions on 30 shelves whole, 6,546 tokens (23,548 B); 6,000 lane libshelves1's
#     165 on 28 shelves, 5,757 tokens (20,490 B), at the design's bound (docs/research/library-design-1011.md).
#     Before it 5,750 held lane lib7's 167 functions on 22 shelves, 5,616 tokens (19,961 B), with lane names1's spare:
#     lane tokens1 set each bound at the most tokens its byte bound admitted on the 58 dev units, so no dev unit was
#     shown less in tokens than in bytes.
#     Lane names1 wrote 5,100 at the program's four bytes a token (at 5,100 tokens 41 of the 167 would stay whole):
#     every call `qf.<name>`, no shelf in it, took the index from 21.1 KB to 20.0 KB, about 0.4 KB to spare; lane
#     lib6's 4,800 held 146 functions on 21 shelves, lane index1's 4,500 116, and lane n11's 600 held names alone past
#     about 70. The House charges no input tokens: the cost is the prefill of every request that carries a system prompt
INDEX_BYTES_PER_TOKEN = 3.55    # the index's lines by the model's tokenizer: the full index 3.55 (3.60 at lane
#                                 libgaps1), the scored 3.44 to 3.58 (3.50 to 3.61) over the 58 dev units
#                                 (scripts/n11_offline.py --ratios)
INDEX_BYTES = 42_600            # the byte fallback where the tokenizer does not load: INDEX_TOKENS at
#                                 INDEX_BYTES_PER_TOKEN (42,600; 24,100 at 6,800 before lane libprep1, 21,300 at 6,000
#                                 before lane libgaps1); the grown index runs at 3.60, so this one binds first, at
#                                 about 310 functions (the token bound at about 320)
BYTES_PER_TOKEN = 4             # the program's estimate of a reply's prose (model._estimate_tokens: solve.py's sketch
#                                 where the route reports no count); no bound is measured by it
LIB_BYTES = 2_500               # a `lib` answer's bound (docs/research/weak-model-agents.md, 3.2)
INDEX_FILE = "index.txt"        # the image's index, written beside this file at build time
HEAD = ("qf, the method library (import qf; numpy and scipy; every convention a required argument). Its shelves "
        "(modules) and functions:")
ALIASES = re.compile(r"^\s*aliases\s*:\s*(.+?)\s*$", re.I | re.M)
ALIAS_SPLIT = ",;"              # an `aliases:` line splits at these outside parentheses: `A(t,T) B(t,T)` is one
COMMON = re.compile(r"^\[(.*)\]$")  # an alias in brackets: a common word or phrase, never matched (`specific`)
CONVENTIONS = re.compile(r"^\s*conventions?\s*:\s*(.+?)\s*$", re.I | re.M)
SECTION = re.compile(r"^(conventions?|returns|formula|source|aliases|examples?)\s*:", re.I)   # a section's label
#                                 line (`returns` alone: a wrapped line of garch11_fit's text begins `return: 'arch'`)
SECTION_KEYS = {"convention": "conventions", "examples": "example"}   # a label's other forms
CARD_SECTIONS = ("formula", "conventions", "returns", "example")     # the sections a whole card carries, in order
ENUM_CHECK = "choice"           # qf._common.choice(name, value, allowed): the code's check of a set of names
NUMBER = "number"               # an enum's last value where its argument also takes a number
CONDITIONS = (ast.If, ast.IfExp, ast.For, ast.While, ast.Try)   # a `choice` under one of these is conditional
CARD_TOKENS = 1_650             # the cards' bound in the model's tokens, their head included (lane tokens1): the
#     most the byte bound admitted on the 58 dev units, 1,622 (the 16 units whose cards it cuts run 2.71 to 3.35 bytes a
#     token, so a token bound shows a sparse unit's cards up to a quarter more). The byte bound's history: lane n11b's
#     3,000, 3,600 at lane lib4's review (examples and `returns:` lines on the smaller cards pushed a larger card's
#     conventions past the bound), 4,400 at lane cards3 (lane cards2's review): on the densest tasks five rates cards
#     were all cut to their smallest even by rank, the first pointed keeping no convention entry. Over the 58 dev units'
#     gold cards (scripts/library_reach.py) every convention is kept on 76 of 89 at 4,400 B (70 at 3,600, 78 at
#     4,800). The House charges no input tokens: the cost is the prefill and attention
CARD_BYTES_PER_TOKEN = 2.71     # the densest cards at the bound, by the model's tokenizer (3.00 the cards' median)
CARD_BYTES = 4_400              # the byte fallback: CARD_TOKENS at CARD_BYTES_PER_TOKEN (4,472), the value kept
WHOLE_TOKENS = INDEX_TOKENS + CARD_TOKENS   # lane first1 (`AGENTHON_N11_CARDS=whole`): the library's part of the
#     first message at most, the index (or the selection call's shelves) and the cards with their head, in the model's
#     tokens: the room the index and the pointer's cards take between them today, so whole cards never make the system
#     prompt longer than today's longest; the cards whole first, the index trimmed for them (`made_whole`)
WHOLE_BYTES = INDEX_BYTES + CARD_BYTES      # its byte fallback, the two fallbacks' sum
CARDS_HEAD = ("Cards of the library's functions the task's words name; where one computes a quantity the task asks, "
              "call it rather than writing its formula:")   # a name can misfire (a citation's): the House judges each
USE = ("The library: where a qf function computes a quantity the task asks, call it under the conventions the task "
       "states rather than writing its formula again. Read its card first ({cards}, or `lib` on a name in the index): "
       "its signature names each convention as an argument, to pass as the task states it, and its `returns:` line "
       "says what comes back. Write a formula yourself only where the task states one that differs from the card's.")
#     lane lib4: when to call a library function and how to read its card, said once before the index in every
#     loop's system prompt (agent/n11/solve.py `SYSTEM`, agent/n11/state.py `SYSTEM`); `{cards}` where its cards are
CARD_CUT = "[cut to fit: `lib name={name}` shows it whole]"     # a trimmed card's last line
SEPARATORS = re.compile(r"[\s\-_‐-―−]+")
APOSTROPHES = re.compile(r"[’‘]")           # a typographic apostrophe: the straight one
POSSESSIVE = re.compile(r"'s\b")              # Black's model, Kendall's tau: the name alone
SPELLINGS = ((re.compile(r"(?<=[a-z])is(?=(?:e|ed|es|ing|er|ation|ations)\b)"), "iz"),   # British -ise: -ize
             (re.compile(r"sced"), "sked"))   # heteroscedastic: heteroskedastic (both sides fold alike)
PLURAL = "(?:e?s)?"             # an alias's plural matches it too (gap options, ES contributions)
CUT = " [...]"
MIN_CARD = 30                   # a card cut shorter than this says too little: the shelf lines and names instead
POINTERS = 5                    # the functions one pointer names, at most: the recall bar reads 0.87 at 5 and 0.66 at 3
#                                 on the library lanes' 98 functions (scripts/n11_offline.py)
UNPOINTED = ("conventions",)    # shelves of conventions and I/O helpers, whose aliases are the words of every task
#                                 (JSON, standard deviation, drawdown): never pointed to; the index lists them
INDEX_MODES = ("full", "scored")    # lane index2 (the owner's in340): every function's line, or the task's own
SCORED_LINES = 8                # the functions the scored index shows whole, at most: the harness's bound, never said;
#     scripts/library_reach.py over the 58 dev units: the gold functions shown whole 0.83 at 4, 0.92 at 8, 0.96 at 12
#     (the pointer's cards 0.67, the full index 1.0), the index 4.8 KB at 8 (4.0 with none, 15.0 the full one)
SCORED_HIT = 2                  # a function whose score reaches this is shown whole (the pointer's always are)
PHRASE_BONUS = 0                # added for each word of an alias the task holds whole as a phrase, above its words':
#                                 at 1 the grader read 0.80 at 4 and 0.91 at 8, at 2 0.77 and 0.91
SHELF_TERMS = 1                 # a shelf line sharing this many terms with the task offers its every function
FILLER = frozenset({            # no term of a function or a task: the field's filler (the brief's), the language's,
    #                             a task's own form (its files and their shapes) and a formula's tokens; the grader read
    #                             0.83 at 4 with the last two, 0.78 without
    "price", "return", "value", "rate", "model", "series", "data", "compute", "estimate", "daily", "annual",
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "be",
    "its", "it", "this", "that", "each", "every", "any", "all", "per", "into", "under", "over", "via", "using", "use",
    "given", "one", "two", "no", "not", "their", "which", "where", "when", "than", "then", "both", "also",
    "output", "input", "file", "json", "csv", "column", "row", "result", "report", "summary", "write", "save", "read",
    "path", "app", "task", "python", "numpy", "scipy", "float", "int", "number", "string", "list", "key", "field",
    "format", "decimal", "table", "you", "must", "should", "if", "only", "same", "do", "has", "these", "exactly",
    "across", "before", "below", "after", "above",
    "named", "times", "sqrt", "ln", "exp", "log", "sum", "total", "step", "time", "level", "zero", "positive",
    "negative", "formula", "parameter", "day", "year", "period", "rule"})
SCORED_HEAD = HEAD.removesuffix(":") + (", those nearest the task's words with their first line and each shelf's "
                                        "others by name after `others:` (`lib name=<function>` shows any one's card):")
OTHERS = "others"               # the scored index's line of a shelf's functions named alone
RULES = ("name", "shelf", "folded", "alias", "plural", "ambiguous", "none")    # `Library.resolve`'s, in its order


@dataclass(frozen=True)
class Function:
    name: str
    shelf: str
    arguments: str                  # as written in its `def`, defaults and markers kept
    doc: str                        # the docstring, cleaned (inspect.cleandoc's rule, ast.get_docstring)
    exported: bool                  # the package's __init__ imports it: `qf.<name>`
    enums: tuple[tuple[str, tuple[str, ...]], ...] = ()   # (argument, the names its code allows; "number" last
    #                                                       where a number is allowed too), from `choices`

    @property
    def card(self) -> str:
        return self.doc.strip().split("\n", 1)[0].strip() if self.doc.strip() else ""

    @property
    def call(self) -> str:
        return f"{PACKAGE}.{self.name}" if self.exported else f"{PACKAGE}.{self.shelf}.{self.name}"

    @property
    def signature(self) -> str:
        return f"{self.call}({self.arguments})"

    @property
    def typed(self) -> str:
        """The call's signature, each argument the code checks against a set of names annotated with every one:
        `qf.asian_price(method: 'geometric'|'levy'|..., option: 'call'|'put', spot, ...)`."""
        if not self.enums:
            return self.signature
        try:
            tree = ast.parse(f"def f({self.arguments}): pass").body[0]
        except SyntaxError:
            return self.signature
        allowed = dict(self.enums)
        for argument in [*tree.args.posonlyargs, *tree.args.args, *tree.args.kwonlyargs]:
            if argument.arg in allowed:
                argument.annotation = ast.Name("|".join(value if value == NUMBER else repr(value)
                                                        for value in allowed[argument.arg]))
        return f"{self.call}({ast.unparse(tree.args)})"

    @property
    def written(self) -> tuple[str, ...]:
        """Every alias as the `aliases:` lines write it, a common one's brackets kept."""
        found = []
        for line in ALIASES.findall(self.doc):
            found += [alias.strip().strip("`'\".") for alias in split_aliases(line)]
        return tuple(dict.fromkeys(alias for alias in found if alias))

    @property
    def aliases(self) -> tuple[str, ...]:
        """Every alias, a common one's brackets dropped: the words a task may use for the function."""
        found = (COMMON.sub(r"\1", alias).strip() for alias in self.written)
        return tuple(dict.fromkeys(alias for alias in found if alias))

    @property
    def pointed(self) -> tuple[str, ...]:
        """The aliases the pointer matches (`specific`)."""
        return tuple(alias for alias in self.written if specific(alias))

    @property
    def conventions(self) -> str:
        return "; ".join(CONVENTIONS.findall(self.doc))

    @property
    def sections(self) -> dict[str, str]:
        """The docstring's labelled sections (`SECTION`, at a line's start), each its label line and every line after it
        up to the next label, keyed by the label in lower case (`SECTION_KEYS` its other forms)."""
        found: dict[str, list[str]] = {}
        current = None
        for line in self.doc.strip().split("\n"):
            label = SECTION.match(line)
            if label is not None:
                current = SECTION_KEYS.get(label.group(1).lower(), label.group(1).lower())
                found[current] = [line]
            elif current is not None:
                found[current].append(line)
        return {key: "\n".join(lines).rstrip() for key, lines in found.items()}

    @property
    def paragraphs(self) -> list[str]:
        """The docstring's text before its first labelled section, by paragraph: its description first, then its
        formula, then any other."""
        lines = []
        for line in self.doc.strip().split("\n"):
            if SECTION.match(line):
                break
            lines.append(line.rstrip())
        found, current = [], []
        for line in lines:
            if line.strip():
                current.append(line)
            elif current:
                found.append("\n".join(current))
                current = []
        if current:
            found.append("\n".join(current))
        return found

    def whole_card(self) -> str:
        """The function's card, nothing cut (`Card`)."""
        return Card(self).text


class Card:
    """A function's card: its typed signature, its paragraphs (the description, the formula and any other), its
    `CARD_SECTIONS`; `trim` drops the next part a card can lose (its example; then its paragraphs after the formula,
    the last first; then its convention entries, the last first, each whole; then its formula's lines, the last first,
    its first line kept) and says so in `cut`. A trimmed card ends with `CARD_CUT`, naming the `lib` call that shows it
    whole."""

    def __init__(self, function: Function) -> None:
        self.function = function
        paragraphs = function.paragraphs
        self.head, self.others = paragraphs[:1], paragraphs[2:]
        self.formula = paragraphs[1].split("\n") if len(paragraphs) > 1 else []
        sections = function.sections
        self.labelled, self.returns, self.example = (sections.get(key, "") for key in ("formula", "returns", "example"))
        self.label, self.entries = entries(sections.get("conventions", ""))
        self.kept_others, self.kept_entries, self.kept_formula = len(self.others), len(self.entries), len(self.formula)
        self.has_example = bool(self.example)
        self.dropped = {"example": 0, "others": 0, "entries": 0, "formula": 0}

    @property
    def text(self) -> str:
        conventions = "\n".join([self.label, *self.entries[:self.kept_entries]]) if self.kept_entries else ""
        formula = "\n".join(self.formula[:self.kept_formula]) + (" [...]" if self.kept_formula < len(self.formula)
                                                                 else "")
        parts = (self.function.typed, *self.head, formula, *self.others[:self.kept_others], self.labelled,
                 conventions, self.returns, self.example if self.has_example else "",
                 CARD_CUT.format(name=self.function.name) if any(self.dropped.values()) else "")
        return "\n".join(part for part in parts if part)

    @property
    def shrinks(self) -> bool:
        return self.has_example or self.kept_others > 0 or self.kept_entries > 0 or self.kept_formula > 1

    @property
    def smallest(self) -> str:
        """The card with every part `trim` drops dropped."""
        card = Card(self.function)
        while card.trim():
            pass
        return card.text

    def trim(self) -> bool:
        """The next part dropped; False where nothing is left to drop."""
        before = size(self.text)
        if self.has_example:
            self.has_example, part = False, "example"
        elif self.kept_others:
            self.kept_others, part = self.kept_others - 1, "others"
        elif self.kept_entries:
            self.kept_entries, part = self.kept_entries - 1, "entries"
        elif self.kept_formula > 1:
            self.kept_formula, part = self.kept_formula - 1, "formula"
        else:
            return False
        self.dropped[part] += before - size(self.text)
        return True

    @property
    def cut(self) -> str:
        """What `trim` dropped and the bytes of what it dropped (the closing line it adds not counted), "" where
        nothing."""
        parts = []
        if self.dropped["example"]:
            parts.append(f"the example ({self.dropped['example']:,} B)")
        if self.dropped["others"]:
            parts.append(f"{len(self.others) - self.kept_others} paragraph(s) after the formula "
                         f"({self.dropped['others']:,} B)")
        if self.dropped["entries"]:
            parts.append(f"{len(self.entries) - self.kept_entries} of {len(self.entries)} convention entries "
                         f"({self.dropped['entries']:,} B)")
        if self.dropped["formula"]:
            parts.append(f"{len(self.formula) - self.kept_formula} of {len(self.formula)} lines of the formula "
                         f"({self.dropped['formula']:,} B)")
        return ", ".join(parts)


def entries(section: str) -> tuple[str, list[str]]:
    """A `conventions:` section as its label line and its entries, each an entry's line with the lines indented
    under it (a list's continuation never parted from its entry)."""
    if not section:
        return "", []
    lines = section.split("\n")
    label, body = lines[0], [line for line in lines[1:] if line.strip()]
    if not body:
        return label, []
    margin = min(len(line) - len(line.lstrip()) for line in body)
    found: list[str] = []
    for line in body:
        if found and len(line) - len(line.lstrip()) > margin:
            found[-1] += "\n" + line
        else:
            found.append(line)
    return label, found


@dataclass(frozen=True)
class Scored:
    """A function's standing on a task (`Library.scored`): its score, its rank among the pointer's matches (None where
    none), and whether its shelf's line touches the task."""
    function: Function
    score: int
    pointed: int | None
    touched: bool

    @property
    def hit(self) -> bool:
        """Shown whole whatever its shelf: the pointer matched it, or it scores `SCORED_HIT` or more."""
        return self.pointed is not None or self.score >= SCORED_HIT


@dataclass(frozen=True)
class Shelf:
    name: str
    line: str
    functions: tuple[Function, ...]


def normal(text: str) -> str:
    """How an alias and the instruction meet: lower case, accents dropped (Lévy, Pynnönen), a typographic apostrophe
    straight and a possessive's 's dropped, the British -is(e) and -isation as -iz(e) and -ization, -scedastic as
    -skedastic, and every run of hyphens, dashes, underscores and spaces one space."""
    text = "".join(char for char in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(char))
    text = POSSESSIVE.sub("", APOSTROPHES.sub("'", text))
    for pattern, said in SPELLINGS:
        text = pattern.sub(said, text)
    return SEPARATORS.sub(" ", text).strip()


@functools.lru_cache(maxsize=8_192)
def matcher(key: str) -> re.Pattern[str]:
    """How the pointer and the scored index find a folded alias or phrase (`normal`) in a task: whole words, its plural
    too; compiled once, since the library's aliases and phrases outnumber re's own cache, which then compiled each
    again on every call (lane lib6's review)."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(key)}{PLURAL}(?![a-z0-9])")


@dataclass(frozen=True)
class Library:
    shelves: tuple[Shelf, ...]

    @property
    def functions(self) -> tuple[Function, ...]:
        return tuple(function for shelf in self.shelves for function in shelf.functions)

    def find(self, name: str) -> Function | None:
        """The function `name` names (`resolve`), None where it names none or could be two."""
        return self.resolve(name)[0]

    def resolve(self, name: str) -> tuple[Function | None, str]:
        """How a name the model says meets a function (lane libalias1), with the rule that met it (`RULES`): a trailing
        `()` or `.py` and the package's `qf.` dropped, then the last dotted part (`qf.<shelf>.<name>` and
        `<shelf>.<name>` alike, whatever the shelf) is, in order, a function's name as written (`name`); else a shelf's
        name alone names no function (`shelf`); else, folded as the pointer folds an alias (`folding`: case, accents,
        hyphens, underscores and spaces), a function's name (`folded`), one of its aliases, every one on its card, the
        common ones' brackets dropped (`alias`), or either with its last word's plural or singular (`plural`, -s and
        -es: `day_counts` meets `day count`). A name two functions answer to at the first rule that meets any is
        resolved to none (`ambiguous`), as is one no rule meets (`none`)."""
        said = unwrapped(name)
        bare = said.rpartition(".")[2]
        names, folded, aliased = self._keys
        if bare in names:
            return names[bare], "name" if names[bare] is not None else "ambiguous"
        if "." not in said and self.shelf(said) is not None:
            return None, "shelf"
        key = folding(bare)
        if not key:
            return None, "none"
        for rule, found in (("folded", folded.get(key, ())), ("alias", aliased.get(key, ())),
                            ("plural", tuple(dict.fromkeys(function for other in variants(key)
                                                           for table in (folded, aliased)
                                                           for function in table.get(other, ()))))):
            if len(found) == 1:
                return found[0], rule
            if found:
                return None, "ambiguous"
        return None, "none"

    def named(self, name: str) -> Function | None:
        """The function `name` names by its own name alone (`resolve`'s first rule, no alias read): whether the library
        holds a function so named, as a reader of the design map's folds asks (scripts/picks_file.py `folding`)."""
        return self._keys[0].get(unwrapped(name).rpartition(".")[2])

    @functools.cached_property
    def _keys(self) -> tuple[dict[str, Function | None], dict[str, tuple[Function, ...]], dict[str, tuple[Function, ...]]]:
        """`resolve`'s tables, built once a library: each function by its name (None where two share one), by its
        name folded (`normal`), and by each of its aliases folded."""
        names: dict[str, Function | None] = {}
        folded: dict[str, list[Function]] = {}
        aliased: dict[str, list[Function]] = {}
        for function in self.functions:
            names[function.name] = None if function.name in names else function
            folded.setdefault(folding(function.name), []).append(function)
            for key in dict.fromkeys(folding(alias) for alias in function.aliases):
                if key:
                    aliased.setdefault(key, []).append(function)
        return (names, {key: tuple(found) for key, found in folded.items()},
                {key: tuple(found) for key, found in aliased.items()})

    def shelf(self, name: str) -> Shelf | None:
        name = name.strip().removeprefix(f"{PACKAGE}.").removesuffix(".py")
        return next((shelf for shelf in self.shelves if shelf.name == name), None)

    # ------------------------------------------------------------------------------------------------ the index

    def index(self, limit: int | None = None, task: str | None = None) -> str:
        """The index within `limit` (above; `index_bound()` by default, in the unit `measured` counts): every shelf
        line and card cut to one width, the widest that fits, where that leaves a card at least `MIN_CARD` characters;
        else each shelf's functions named on its line, the shelf lines cut to the widest that fits. With the `task`'s
        text, the scored index (`_scored`)."""
        limit = index_bound() if limit is None else limit
        if task is not None:
            return self._scored(task, limit)
        widest = max((len(text) for text in (*(shelf.line for shelf in self.shelves),
                                             *(function.card for function in self.functions))), default=0)
        if measured(self._lines(widest)) <= limit:
            return self._lines(widest)
        width = widest_fitting(self._lines, widest, limit)
        if width is not None and width >= MIN_CARD:
            return self._lines(width)
        width = widest_fitting(self._named, max((len(shelf.line) for shelf in self.shelves), default=0), limit)
        if width is not None:
            return self._named(width)
        return clip(self._named(0), limit, "\n[... the rest by `lib shelf=<module>`]", measure.UNIT)

    def _lines(self, width: int) -> str:
        lines = [HEAD]
        for shelf in self.shelves:
            line = cut_at(shelf.line, width)
            lines.append(f"{shelf.name}: {line}" if line else shelf.name)
            for function in shelf.functions:
                card = cut_at(function.card, width)
                lines.append(f"  {function.call}" + (f": {card}" if card else ""))
        return "\n".join(lines)

    def _named(self, width: int) -> str:
        lines = [HEAD]
        for shelf in self.shelves:
            line = cut_at(shelf.line, width)
            lines.append(f"{shelf.name}" + (f" ({line})" if line else "") + ": "
                         + ", ".join(function.name for function in shelf.functions))
        return "\n".join(lines)

    def whole(self, index: str) -> int:
        """The functions `index` shows with their card whole (`shown`), counted."""
        return len(self.shown(index))

    def shown(self, index: str) -> list[Function]:
        """The functions `index` shows with their card whole: each whose line `_lines` writes uncut is a line of it."""
        lines = set(index.split("\n"))
        return [function for function in self.functions if line_of(function) in lines]

    # ------------------------------------------------------------------------------------------ the scored index

    def scored(self, task: str) -> list[Scored]:
        """Every function's score on the task's text (lane index2; `Scored`), in the library's order: the count of its
        `terms` the task holds, plus `PHRASE_BONUS` a word for each of its aliases the task holds whole; whether the
        pointer matches it (`pointers`, every match, not only the first `POINTERS`); whether its shelf's line shares
        `SHELF_TERMS` terms with the task."""
        held = stems_of(terms(task))
        text = normal(task)
        matched = {function: rank for rank, (function, _) in
                   enumerate(self.pointers(task, most=len(self.functions)))}
        found = []
        for shelf in self.shelves:
            touched = len([term for term in terms(shelf.line) if stems_of({term}) & held]) >= SHELF_TERMS
            for function in shelf.functions:
                score = len([term for term in function_terms(function) if stems_of({term}) & held])
                if PHRASE_BONUS:
                    score += PHRASE_BONUS * sum(len(phrase.split()) for phrase in phrases(function)
                                                if matcher(phrase).search(text))
                found.append(Scored(function, score, matched.get(function), touched))
        return found

    def chosen(self, task: str, most: int | None = None) -> list[Function]:
        """The functions the scored index shows whole, at most `most`, best first: the pointer's matches in its order,
        then by score, the library's order breaking ties; every function scoring `SCORED_HIT` or more, and every
        function of a touched shelf, until the bound, the lowest scores left out first."""
        offered = [(position, each) for position, each in enumerate(self.scored(task)) if each.hit or each.touched]
        offered.sort(key=lambda item: (item[1].pointed is None, item[1].pointed or 0, -item[1].score, item[0]))
        return [each.function for _, each in offered[:SCORED_LINES if most is None else most]]

    def _scored(self, task: str, limit: int) -> str:
        """`index` under `scored` (above): the head, each shelf's line whole, under it the line of each function
        `chosen` shows whole and the rest by name on one line, in the shelf's order; past `limit` the lowest of the
        chosen named alone, one by one, and the full index's fallback where even none fits."""
        chosen = self.chosen(task)
        while True:
            text = self._chosen(set(chosen))
            if measured(text) <= limit:
                return text
            if not chosen:
                return self.index(limit)
            chosen.pop()

    def _chosen(self, shown: set[Function]) -> str:
        lines = [SCORED_HEAD]
        for shelf in self.shelves:
            lines.append(f"{shelf.name}: {shelf.line}" if shelf.line else shelf.name)
            lines += [line_of(function) for function in shelf.functions if function in shown]
            rest = [function.name for function in shelf.functions if function not in shown]
            if rest:
                lines.append(f"  {OTHERS}: " + ", ".join(rest))
        return "\n".join(lines)

    def trims(self, index: str, touched: set[str]) -> list[str]:
        """Lane first1 (`made_whole`): `index` as `index` wrote it, full (`HEAD`) or scored (`SCORED_HEAD`), with one
        more shelf reduced at each step to make room for the cards whole: the shelves no carded function lies on
        (`touched`, by name) first, the last first, then the others, the last first. A reduced shelf is said as the
        index says one when it holds no room for more: in the full index its line and its functions by name (`_named`),
        in the scored its line and every function after `others:`; a shelf its reduction would not shorten keeps its
        lines and takes no step. [] for a text of another form or of another library's shelves."""
        lines = index.split("\n")
        if not lines or lines[0] not in (HEAD, SCORED_HEAD):
            return []
        blocks: list[list[str]] = []
        for line in lines[1:]:
            if line.startswith("  ") and blocks:
                blocks[-1].append(line)
            else:
                blocks.append([line])
        if len(blocks) != len(self.shelves) or any(not block[0].startswith(shelf.name)
                                                   for block, shelf in zip(blocks, self.shelves)):
            return []
        scored = lines[0] == SCORED_HEAD
        reduced = {}
        for k, (block, shelf) in enumerate(zip(blocks, self.shelves)):
            names = ", ".join(function.name for function in shelf.functions)
            said = [block[0], f"  {OTHERS}: {names}"] if scored else \
                [f"{shelf.name}" + (f" ({shelf.line})" if shelf.line else "") + f": {names}"]
            if size("\n".join(said)) < size("\n".join(block)):
                reduced[k] = said
        order = [k for k in reversed(range(len(blocks))) if self.shelves[k].name not in touched]
        order += [k for k in reversed(range(len(blocks))) if self.shelves[k].name in touched]
        found, done = [], set()
        for k in (k for k in order if k in reduced):
            done.add(k)
            found.append("\n".join([lines[0], *(line for j, block in enumerate(blocks)
                                                  for line in (reduced[j] if j in done else block))]))
        return found

    # ---------------------------------------------------------------------------------------------------- `lib`

    def by_name(self, name: str, limit: int = LIB_BYTES) -> str:
        found = self.find(name)
        if found is None:
            return clip(self.unknown(name), limit)
        return answer(found, limit)

    def by_shelf(self, name: str, limit: int = LIB_BYTES) -> str:
        found = self.shelf(name)
        if found is None:
            close = difflib.get_close_matches(name, [shelf.name for shelf in self.shelves], n=3, cutoff=0.5)
            return clip(f"lib: no shelf `{name}`" + (f" (close: {', '.join(close)})" if close else "")
                        + f"; the shelves are {', '.join(shelf.name for shelf in self.shelves)}", limit)
        lines = [f"{found.name}: {found.line}"]
        for function in found.functions:
            lines += [function.signature, f"    {function.card}"]
        return clip("\n".join(lines), limit, "\n[... cut; `lib name=<function>` shows one]")

    def unknown(self, name: str) -> str:
        names = [function.name for function in self.functions]
        close = difflib.get_close_matches(name.rpartition(".")[2], names, n=4, cutoff=0.5)
        return (f"lib: no function `{name}`" + (f" (close: {', '.join(close)})" if close else "")
                + f"; the shelves are {', '.join(shelf.name for shelf in self.shelves)}")

    # ------------------------------------------------------------------------------------------- the pointer

    def pointers(self, instruction: str, most: int = POINTERS) -> list[tuple[Function, str]]:
        """The functions an alias of which the instruction holds (`normal` on both sides, its plural too), each with
        the longest such alias, the most specific first (the longest alias, then the earliest named), at most `most`;
        never one on an `UNPOINTED` shelf."""
        text, found = normal(instruction), []
        for function in self.functions:
            if function.shelf in UNPOINTED:
                continue
            best = None
            for alias in function.pointed:
                key = normal(alias)
                match = matcher(key).search(text) if key else None
                if match and (best is None or (len(key), -match.start()) > (best[0], -best[1])):
                    best = (len(key), match.start(), alias)
            if best is not None:
                found.append((best, function))
        ranked = sorted(found, key=lambda item: (-item[0][0], item[0][1], item[1].name))
        return [(function, best[2]) for best, function in ranked[:most]]

    @staticmethod
    def carded(pointed: list[tuple[Function, str]], limit: int | None = None,
               head: str = CARDS_HEAD) -> tuple[str, list[str]]:
        """The cards of the functions `pointers` named (above): `head` (`CARDS_HEAD`; a sketch's picks', lane plan4)
        and each `Card`, in order, within `limit` in all (`card_bound()` by default, in the unit `measured` counts),
        fitted by `trimmed`. With what was cut, one line a card (`qf.<fn>: ...`), for the log; ("", the cuts) where
        none fits."""
        limit = card_bound() if limit is None else limit
        cards, left_out = Library.trimmed(pointed, limit, head)
        unit = "tokens" if measure.UNIT.tokenizer is not None else "B"
        cuts = [f"{card.function.call}: {card.cut} cut" for card in cards if card.cut]
        cuts += [f"{function.call}: left out (its smallest card {measured(Card(function).smallest):,} {unit}; the "
                 f"cards' {limit:,} {unit} full)" for function in reversed(left_out)]
        text = "\n\n".join([head, *(card.text for card in cards)]) if cards else ""
        return text, cuts

    @staticmethod
    def trimmed(pointed: list[tuple[Function, str]], limit: int | None = None,
                head: str = CARDS_HEAD) -> tuple[list[Card], list[Function]]:
        """`carded`'s fit, measured with `head`: the `Card`s kept, each trimmed as it is shown, and the functions left
        out, the last pointed first. Past `limit` the cards are cut by the pointer's rank (lane cards2): the last
        pointed loses its parts (`Card.trim`) one by one until they fit or it is at its smallest, then the one before
        it, so a card loses nothing while a card pointed after it can still lose a part; where every card at its
        smallest is still past `limit`, the last pointed is left out and the rest fitted again from whole."""
        limit = card_bound() if limit is None else limit
        shown = [function for function, _ in pointed]
        left_out: list[Function] = []
        while shown:
            cards = [Card(function) for function in shown]
            for card in reversed(cards):
                while fitted(cards, head) > limit and card.trim():
                    pass
            if fitted(cards, head) <= limit:
                return cards, left_out
            left_out.append(shown.pop())
        return [], left_out

    @staticmethod
    def cards(pointed: list[tuple[Function, str]], limit: int | None = None) -> str:
        """The cards' text alone (`carded`)."""
        return Library.carded(pointed, limit)[0]


@dataclass(frozen=True)
class Whole:
    """Lane first1 (`made_whole`): the first message's index and cards under `AGENTHON_N11_CARDS=whole`, what the cards'
    fit cut (a line a card, as `carded` writes them; none where every card is whole), the index's bytes given up for
    the cards and the trim's steps taken."""
    index: str
    cards: str
    cuts: tuple[str, ...]
    given_up: int
    steps: int


def made_whole(functions: list[Function], index: str, trims: list[str], limit: int | None = None,
               head: str = CARDS_HEAD) -> Whole:
    """Lane first1: `functions`' cards whole (`Card`, nothing trimmed), under `head` and after `index`, within `limit`
    in all (`whole_bound()` by default, in the unit `measured` counts) and never past it: where they do not fit, the
    first of the index's `trims` (in their order) that leaves them room; where even the last leaves too little, the
    cards fitted in the room it leaves as today (`Library.carded`: lane cards2's trim by rank, the last carded cut
    first), and an index past the bound alone cut to it."""
    limit = whole_bound() if limit is None else limit

    def joined(text: str, cards: str) -> str:
        return f"{text}\n\n{cards}" if cards else text

    cards = "\n\n".join([head, *(Card(function).text for function in functions)]) if functions else ""
    for step, text in enumerate([index, *trims]):
        if measured(joined(text, cards)) <= limit:
            return Whole(text, cards, (), size(index) - size(text), step)
    text = trims[-1] if trims else index
    if measured(text) > limit:
        text = clip(text, limit, "", measure.UNIT)
    ranked = [(function, function.name) for function in functions]
    room = limit - measured(f"{text}\n\n")
    cards, cuts = Library.carded(ranked, max(room, 0), head)
    while cards and measured(joined(text, cards)) > limit:     # in tokens the join is not quite the parts' sum
        room -= measured(joined(text, cards)) - limit
        cards, cuts = Library.carded(ranked, max(room, 0), head)
    return Whole(text, cards, tuple(cuts), size(index) - size(text), len(trims))


def fitted(cards: list[Card], head: str = CARDS_HEAD) -> int:
    """The cards with their head and the blank lines between them, as `carded` joins them, `measured`."""
    return measured("\n\n".join([head, *(card.text for card in cards)]))


def widest_fitting(render, widest: int, limit: int) -> int | None:
    """The largest width at most `widest` whose `render(width)` fits in `limit` (`measured`); None where none does."""
    if measured(render(0)) > limit:
        return None
    low, high = 0, widest
    while low < high:
        middle = (low + high + 1) // 2
        low, high = (middle, high) if measured(render(middle)) <= limit else (low, middle - 1)
    return low


def size(text: str) -> int:
    """`text`'s bytes (a `lib` answer's bound, and what the log says in bytes)."""
    return len(text.encode("utf-8"))


def measured(text: str) -> int:
    """`text` in the unit the index's and the cards' bounds are in (agent/n11/measure.py): the model's tokens, else
    bytes where the tokenizer does not load."""
    return measure.size(text)


def index_bound() -> int:
    """The index's bound in that unit: `INDEX_TOKENS`, else `INDEX_BYTES`."""
    return measure.bound(INDEX_TOKENS, INDEX_BYTES)


def card_bound() -> int:
    """The cards' bound in that unit: `CARD_TOKENS`, else `CARD_BYTES`."""
    return measure.bound(CARD_TOKENS, CARD_BYTES)


def whole_bound() -> int:
    """Lane first1: the library's part of the first message under `whole`, in that unit: `WHOLE_TOKENS`, else
    `WHOLE_BYTES`."""
    return measure.bound(WHOLE_TOKENS, WHOLE_BYTES)


def specific(alias: str) -> bool:
    """Whether the pointer matches `alias`, as its `aliases:` line writes it: two or more words (hyphens, underscores
    and spaces alike), or one word written with a capital (a name or an acronym); never one in brackets."""
    if COMMON.match(alias):
        return False
    words = normal(alias).split()
    return len(words) >= 2 or (len(words) == 1 and alias[:1].isupper())


GENERIC_WORDS = frozenset({"compute", "calculate", "calc", "get", "make", "build", "run", "do", "own", "my", "the", "a",
                           "an", "of", "and", "to", "from", "for", "with", "by", "on", "in"})   # no method's word
SHORT_WORD = 2                  # a mark's words this short are dropped where two longer ones remain (euler_es_...)
CODE_ALIAS = re.compile(r"[a-z_][a-z0-9_.]*")   # an alias written as a routine's name (`logm`, `arch_model`)
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z][a-z])")   # a name's next word: computeVaR, BSCallPrice (VaR stays whole)
DEF = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M)   # a `def` line of a text that does not parse


def words(text: str, camel: bool = False) -> list[str]:
    """A phrase's words as the pointer folds them (`normal`), split at anything not a letter or a digit; with
    `camel`, an identifier's camel-case words apart too."""
    return [word for word in re.split(r"[^a-z0-9]+", normal(CAMEL.sub(" ", text) if camel else text)) if word]


def marks(function: Function) -> tuple[frozenset[str], ...]:
    """The word sets a hand-written `def`'s name must hold to compute `function` (`rewritten`): its name's and each
    specific alias's words but `GENERIC_WORDS`; a short word written in lower case (`es`, `t`) dropped where two
    longer remain, a short name or acronym (`Li`, `ES`) kept."""
    found: list[frozenset[str]] = []
    for said in (function.name, *function.pointed):
        kept = [(raw, word) for raw in re.split(r"[^A-Za-z0-9]+", said) for word in words(raw)
                if word not in GENERIC_WORDS]
        longer = [word for raw, word in kept if len(word) > SHORT_WORD or raw != raw.lower()]
        mark = frozenset(longer if len(longer) >= 2 else (word for _, word in kept))
        if mark and mark not in found:
            found.append(mark)
    return tuple(found)


def formula_calls(function: Function) -> tuple[str, ...]:
    """The routines that compute `function`'s formula: an alias written as a routine's name that its docstring also
    writes as a call (`logm`: Q = logm(P) / dt)."""
    return tuple(alias.rpartition(".")[2] for alias in function.aliases if CODE_ALIAS.fullmatch(alias)
                 and re.search(rf"(?<![\w.]){re.escape(alias)}\(", function.doc))


def _callee(node: ast.AST) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


@dataclass(frozen=True)
class Rewrite:
    """One hand computation of a library function `rewrites` finds (lane called1): the `def` that computes it, or the
    routine of its formula called (`by`, `def` or `routine`), and the file that holds it."""
    function: Function
    name: str                       # the def's name, or the routine's
    file: str                       # the file's name as `rewrites` was given it
    by: str                         # "def" or "routine"


def rewrites(functions, files, known=frozenset()) -> list[Rewrite]:
    """The hand computations of `functions` in the source `files`, each (name, text), by `rewritten`'s rule (lane lib4),
    named (lane called1): every `def` whose name holds every word of one of the function's `marks` and whose body
    calls neither the function nor any name in `known`, with its file; and, where no file calls the function, each
    file's call of one of its `formula_calls`. In the functions' order, a function's defs before its routines, each in
    the files' order. A text that does not parse is read by its lines, its every call standing for each of its
    `def`s' bodies."""
    defs: list[tuple[str, set[str], str]] = []
    called: dict[str, set[str]] = {}
    for file, text in files:
        here = called.setdefault(file, set())
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            bodies = DEF.sub("", text)                  # a `def name(` line is no call of `name`
            flat = {name for pair in re.findall(r"(?<![\w.])([A-Za-z_]\w*)\s*\(|\.([A-Za-z_]\w*)\s*\(", bodies)
                    for name in pair if name}
            defs += [(name, flat, file) for name in DEF.findall(text)]
            here |= flat
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                here.add(_callee(node.func))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append((node.name, {_callee(inner.func) for inner in ast.walk(node)
                                         if isinstance(inner, ast.Call)}, file))
    anywhere = set().union(*called.values())
    found = []
    for function in functions:
        held = marks(function)
        found += [Rewrite(function, name, file, "def") for name, inner, file in defs
                  if any(mark <= set(words(name, camel=True)) for mark in held)
                  and function.name not in inner and not inner & known]
        if function.name not in anywhere:
            found += [Rewrite(function, routine, file, "routine") for file, names in called.items()
                      for routine in formula_calls(function) if routine in names]
    return found


def rewritten(functions, texts, known=frozenset()) -> list[Function]:
    """The `functions` the source `texts` compute by hand (lane lib4): a `def` whose name holds every word of one of
    the function's `marks` and whose body calls neither the function nor any name in `known` (the library's methods,
    solve.py's outside the conventions shelf: a wrapper or a neighbour's caller computes nothing by hand, a JSON
    writer's caller may), or a call of one of its `formula_calls` in files that never call the function
    (`rewrites`, which names each)."""
    found = {rewrite.function for rewrite in rewrites(functions, [(str(k), text) for k, text in enumerate(texts)],
                                                      known)}
    return [function for function in functions if function in found]


def line_of(function: Function) -> str:
    """A function's line in the index, its card whole."""
    return f"  {function.call}" + (f": {function.card}" if function.card else "")


def terms(text: str) -> set[str]:
    """A text's terms (lane index2): its words as the pointer folds them (`words`), but a lone letter, a number and
    `FILLER`, its plurals too (`stems_of`)."""
    return {word for word in words(text) if len(word) > 1 and not word.isdigit() and not stems_of({word}) & FILLER}


def unwrapped(name: str) -> str:
    """A name as `resolve` reads it: a trailing `()` or `.py` and the package's `qf.` dropped."""
    return name.strip().removesuffix("()").removesuffix(".py").removeprefix(f"{PACKAGE}.")


def folding(name: str) -> str:
    """A name or an alias as `resolve` folds it: `normal`, its underscores spaces first, so the British spellings fold
    inside a code name too (`annualised_return` meets `annualized return`)."""
    return normal(name.replace("_", " "))


def variants(key: str) -> set[str]:
    """A folded name (`normal`) with its last word's plural (-s, -es) or singular (`stems_of`) in its place, `resolve`'s
    last rule: `day counts` and `day count` meet."""
    head, _, last = key.rpartition(" ")
    return {f"{head} {word}".strip() for word in {last + "s", last + "es", *stems_of({last})}} - {key}


def stems_of(found: set[str]) -> set[str]:
    """Each word with the singular its plural may stand for (`PLURAL`'s -s and -es), where three letters or more
    remain: `options` meets `option`, `processes` `process`, on either side."""
    out = set(found)
    for word in found:
        out |= {word[:-len(end)] for end in ("s", "es") if word.endswith(end) and len(word) - len(end) >= 3}
    return out


def function_terms(function: Function) -> set[str]:
    """A function's terms: those of every alias, a common one's brackets dropped, and of its card."""
    return set().union(terms(function.card), *(terms(alias) for alias in function.aliases))


def phrases(function: Function) -> set[str]:
    """A function's aliases of two words or more, as `normal` writes them."""
    return {key for key in (normal(alias) for alias in function.aliases) if len(key.split()) >= 2}


def split_aliases(line: str) -> list[str]:
    """An `aliases:` line cut at each `ALIAS_SPLIT` character at parenthesis depth 0, nested pairs included
    (`f(x, g(y))` is one alias); a stray `)` never takes the depth below 0."""
    parts, depth, start = [], 0, 0
    for at, char in enumerate(line):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char in ALIAS_SPLIT and depth == 0:
            parts.append(line[start:at])
            start = at + 1
    return [*parts, line[start:]]


def cut_at(text: str, width: int) -> str:
    """`text` whole where it fits in `width` characters, else cut at the last word that fits, `...` after it."""
    if len(text) <= width:
        return text
    if width < 8:
        return ""
    head = text[:width - 3].rsplit(" ", 1)[0].rstrip(" ,;:(")
    return head + "..." if head else ""


def answer(function: Function, limit: int = LIB_BYTES) -> str:
    """`lib name=`'s answer for `function` (`Library.by_name`): its signature, its import and shelf, and its whole
    docstring, within `limit`; lane cardfail1's card after a failing call is this text."""
    how = f"import {PACKAGE}" if function.exported else f"import {PACKAGE}.{function.shelf}"
    return clip(f"{function.signature}   ({how}; shelf {function.shelf})\n{function.doc.strip()}", limit)


def clip(text: str, limit: int, tail: str = CUT, unit: measure.Unit = measure.BYTES) -> str:
    """`text` whole within `limit` (bytes, or `unit`'s), else cut at a line inside it with `tail` after."""
    if unit.size(text) <= limit:
        return text
    room = limit - unit.size(tail)
    data = unit.prefix(text, max(0, room))
    if "\n" in data:
        data = data.rsplit("\n", 1)[0]
    return data + tail


# ------------------------------------------------------------------------------------------------- the source


def exported(init: Path) -> tuple[set[str], set[str]]:
    """The names the package's `__init__.py` imports from its own modules, and the modules it imports whole (`from
    .<module> import *`: the shelf's every function, its `__all__` where it has one, as `scan` reads a shelf)."""
    try:
        tree = ast.parse(init.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return set(), set()
    found = [(node.module, alias) for node in tree.body if isinstance(node, ast.ImportFrom) and node.level == 1
             for alias in node.names]
    return ({alias.asname or alias.name for _, alias in found if alias.name != "*"},
            {module for module, alias in found if alias.name == "*" and module})


def listed(tree: ast.Module) -> set[str] | None:
    """A module's `__all__` where it assigns a list or tuple of strings; None where it has none."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return None
            return {str(item) for item in value} if isinstance(value, (list, tuple)) else None
    return None


def assigned(tree: ast.Module) -> dict[str, tuple]:
    """A module's top-level names bound to a tuple or list of literals (`METHODS = ("levy", ...)`)."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                continue
            if isinstance(value, (tuple, list)):
                found[node.targets[0].id] = tuple(value)
    return found


def imported(tree: ast.Module, modules: dict[str, dict[str, tuple]]) -> dict[str, tuple]:
    """The constants a module imports from a sibling (`from .conventions import SIGNS`), as that sibling binds them."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in modules:
            for alias in node.names:
                if alias.name in modules[node.module]:
                    found[alias.asname or alias.name] = modules[node.module][alias.name]
    return found


def choices(node: ast.FunctionDef, constants: dict[str, tuple]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Each argument the function's code checks with `choice(name, argument, allowed)` and the names it allows, in
    the arguments' order; `NUMBER` after them where the check sits under a condition (a name or a number: seed=,
    bandwidth=, start_variance=). One walk of the body, each node carrying whether a condition holds it."""
    arguments = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
    found: dict[str, tuple[str, ...]] = {}
    stack = [(child, False) for child in node.body]
    while stack:
        call, conditional = stack.pop()
        stack.extend((child, conditional or isinstance(call, CONDITIONS)) for child in ast.iter_child_nodes(call))
        if not (isinstance(call, ast.Call) and len(call.args) == 3 and isinstance(call.args[1], ast.Name)
                and (getattr(call.func, "id", None) == ENUM_CHECK or getattr(call.func, "attr", None) == ENUM_CHECK)):
            continue
        argument, allowed = call.args[1].id, call.args[2]
        if argument not in arguments or argument in found:
            continue
        if isinstance(allowed, ast.Name):
            values = constants.get(allowed.id)
        else:
            try:
                values = tuple(ast.literal_eval(allowed))
            except (ValueError, SyntaxError, TypeError):
                values = None
        if values and all(isinstance(value, str) for value in values):
            found[argument] = tuple(values) + ((NUMBER,) if conditional else ())
    return tuple((argument, found[argument]) for argument in arguments if argument in found)


def scan(package: Path) -> Library:
    """The library under `package` (agent/qf, or /opt/agent/qf), read by `ast`; an empty one where it is absent. A
    module that does not parse is left out. Read once a process while its files stand as they were (`scanned`, keyed
    by each file's name, size and modification time): the image's agent scans once a unit either way."""
    package = Path(package)
    stood = []
    for path in sorted(package.glob("*.py")) if package.is_dir() else []:
        try:
            st = path.stat()
        except OSError:
            continue
        stood.append((path.name, st.st_size, st.st_mtime_ns))
    return scanned(str(package), tuple(stood))


@functools.lru_cache(maxsize=8)
def scanned(package: str, stood: tuple[tuple[str, int, int], ...]) -> Library:
    """`scan`'s reading of the package's files as `stood` names them."""
    package = Path(package)
    names, whole = exported(package / "__init__.py")
    trees: dict[str, ast.Module] = {}
    for name, _, _ in stood:
        try:
            trees[Path(name).stem] = ast.parse((package / name).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
    modules = {stem: assigned(tree) for stem, tree in trees.items()}
    shelves = []
    for stem, tree in trees.items():
        if stem.startswith("_"):
            continue
        public = listed(tree)
        constants = {**imported(tree, modules), **modules[stem]}
        functions = tuple(
            Function(node.name, stem, ast.unparse(node.args), ast.get_docstring(node) or "",
                     node.name in names or stem in whole, choices(node, constants))
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
            and (public is None or node.name in public))
        if functions:
            line = (ast.get_docstring(tree) or "").strip().split("\n", 1)[0].strip()
            shelves.append(Shelf(stem, line, functions))
    return Library(tuple(shelves))


def load(here: Path) -> tuple[Library, str]:
    """The library beside the agent (`<here>/../qf`) and its index: the build's `index.txt` where the image wrote
    one, else the index built now."""
    library = scan(Path(here).parent / PACKAGE)
    built = Path(here) / INDEX_FILE
    try:
        return library, built.read_text(encoding="utf-8").rstrip("\n")
    except OSError:
        return library, library.index()
