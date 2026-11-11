"""Reads the method library (agent/qf) from its source and renders it for the prompt.

Step: 2, picking library functions.
Called by: agent/n11/solve.py (`load`, `Library.pointers`, `made_whole`), agent/n11/selection.py, and
    scripts/qf_index.py at the image's build.
Calls: agent/n11/measure.py for sizes. It never imports agent/qf.
Start at: `load` and `scan` (source to `Library`), `Library.index`, `Library.pointers`, `Library.carded`, `made_whole`.

Words: a shelf is one module of agent/qf. A function's card is its documentation as the model is shown it (nothing to
do with the task card, card.toml). The index is the whole library, a line per function. A pointer is a function
whose docstring's `aliases:` line matches a term in the instruction.

The rules in detail
-------------------
The method library `qf` as the agent shows it to the model: an index in the system prompt, and cards of the
functions the task names.

The library is read from its source by `ast` (`scan`), never imported: the agent's process loads no numpy. A shelf is
a module of the package not named with a leading underscore, its line the module docstring's first line; its
functions are the module's public top-level `def`s (its `__all__` where it has one), each with its signature, its
docstring and its card, the docstring's first line. A function the package's `__init__` imports, by its name or with
its whole shelf (`from .<shelf> import *`), is called `qf.<name>`, any other `qf.<shelf>.<name>`.

- `Library.index` is the system prompt's index: a line per shelf, and under it a line per function,
  `qf.<name>: <card>`, in the shelf's order; within `INDEX_TOKENS` of the model's tokens (agent/n11/measure.py), or
  `INDEX_BYTES` where the tokenizer does not load. Where a library outgrows it, every shelf line and card is cut to
  the same length at a word, the longest that fits; where that leaves a card under `MIN_CARD` characters, each
  shelf's functions are named on its own line after its shelf line. With a task's text it is the scored index
  (`_scored`): the functions nearest the task's words with their card, the rest by name. `index.txt` beside this
  file is the index the image's build wrote (scripts/qf_index.py); without it the agent builds the same text at
  start.
- `Library.find` is how a name the model writes meets a function (`resolve`): a function's name as written, at any
  shelf or none; a shelf's name alone none; else, folded as the pointer folds an alias, a function's name, one of
  its aliases or either's plural or singular; a name two functions answer to is none, as is one no rule meets.
- `Library.pointers` is the exact-name pointer: a function whose docstring's `aliases:` line names a term the
  instruction holds, matched case-insensitively with hyphens, underscores and spaces alike and never inside a longer
  word, its plural too, accents, typographic apostrophes, a possessive's 's and the British -ise spellings folded
  (`normal`); at most `POINTERS`, the longest alias first, and never one on an `UNPOINTED` shelf. Only a `specific`
  alias is matched: a method's name or a phrase of two or more words, never one word written in lower case, nor an
  alias the line writes in brackets (a common word most tasks compute in their own convention).
- `Library.carded` is the pointed functions' cards and what was cut from them: each `Card` rendered from the full
  docstring: the call's signature with every argument the code checks against a set of names annotated with them
  (`Function.typed`), the docstring's text before its first labelled section, its `formula:`, `conventions:`,
  `returns:` and `example:` sections. Within `CARD_TOKENS` (else `CARD_BYTES`) in all with `CARDS_HEAD` above them,
  in the pointer's order, and cut by rank: while they do not fit, the last pointed card loses its next part
  (`Card.trim`) until it is at its smallest, and only then the card before it; a trimmed card's last line names the
  `lib` call that shows it whole. Where every card at its smallest still does not fit, the last pointed is left out
  and the rest fitted again from whole.
- `made_whole` is the first message's cards whole: every carded function's `Card` untrimmed after the index, within
  `WHOLE_TOKENS` in all (else `WHOLE_BYTES`); where they do not fit, the index trimmed for them (`Library.trims`), a
  shelf no carded function lies on reduced first; and only where the index at its most trimmed leaves too little,
  the cards cut as `carded` cuts them.

Nothing here names a task, and the instruction is only searched for the library's own aliases.
"""
from __future__ import annotations

import ast
import functools
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from n11 import measure

PACKAGE = "qf"
INDEX_TOKENS = 12_000           # the index's bound in the model's tokens: room for about 300 functions whole
INDEX_BYTES = 42_600            # the byte fallback where the tokenizer does not load: INDEX_TOKENS at the full index's
#                                 3.55 bytes a token
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
ENUM_CHECK = "choice"           # qf._common.choice(name, value, allowed): the code's check of a set of names
NUMBER = "number"               # an enum's last value where its argument also takes a number
CONDITIONS = (ast.If, ast.IfExp, ast.For, ast.While, ast.Try)   # a `choice` under one of these is conditional
CARD_TOKENS = 1_650             # the cards' bound in the model's tokens, their head included
CARD_BYTES = 4_400              # the byte fallback: CARD_TOKENS at the densest cards' 2.71 bytes a token
WHOLE_TOKENS = INDEX_TOKENS + CARD_TOKENS   # the index and the cards whole together, at most: whole cards never make
#                                             the system prompt longer than the index and the cut cards would
WHOLE_BYTES = INDEX_BYTES + CARD_BYTES      # its byte fallback, the two fallbacks' sum
CARDS_HEAD = ("Cards of the library's functions the task's words name; where one computes a quantity the task asks, "
              "call it rather than writing its formula:")   # a name can misfire (a citation's): the model judges each
USE = ("The library: where a qf function computes a quantity the task asks, call it under the conventions the task "
       "states rather than writing its formula again. Read its card first ({cards}, or `lib` on a name in the index): "
       "its signature names each convention as an argument, to pass as the task states it, and its `returns:` line "
       "says what comes back. Write a formula yourself only where the task states one that differs from the card's.")
#     when to call a library function and how to read its card, said once before the index (agent/n11/solve.py
#     `SYSTEM`); `{cards}` where its cards are
CARD_CUT = "[cut to fit: `lib name={name}` shows it whole]"     # a trimmed card's last line
SEPARATORS = re.compile(r"[\s\-_‐-―−]+")
APOSTROPHES = re.compile(r"[’‘]")           # a typographic apostrophe: the straight one
POSSESSIVE = re.compile(r"'s\b")              # Black's model, Kendall's tau: the name alone
SPELLINGS = ((re.compile(r"(?<=[a-z])is(?=(?:e|ed|es|ing|er|ation|ations)\b)"), "iz"),   # British -ise: -ize
             (re.compile(r"sced"), "sked"))   # heteroscedastic: heteroskedastic (both sides fold alike)
PLURAL = "(?:e?s)?"             # an alias's plural matches it too (gap options, ES contributions)
CUT = " [...]"
MIN_CARD = 30                   # a card cut shorter than this says too little: the shelf lines and names instead
POINTERS = 5                    # the functions one pointer names, at most
UNPOINTED = ("conventions",)    # shelves of conventions and I/O helpers, whose aliases are the words of every task
#                                 (JSON, standard deviation, drawdown): never pointed to; the index lists them
SCORED_LINES = 8                # the functions the scored index shows whole, at most
SCORED_HIT = 2                  # a function whose score reaches this is shown whole (the pointer's always are)
SHELF_TERMS = 1                 # a shelf line sharing this many terms with the task offers its every function
FILLER = frozenset({            # no term of a function or a task: the field's filler, the language's, a task's own
    #                             form (its files and their shapes) and a formula's tokens
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


class Card:
    """A function's card: its typed signature, its paragraphs (the description, the formula and any other), its
    `formula:`, `conventions:`, `returns:` and `example:` sections; `trim` drops the next part a card can lose (its
    example; then its paragraphs after the formula, the last first; then its convention entries, the last first, each
    whole; then its formula's lines, the last first, its first line kept) and says so in `cut`. A trimmed card ends with
    `CARD_CUT`, naming the `lib` call that shows it whole."""

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
    again on every call."""
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
        """How a name the model says meets a function, with the rule that met it: a trailing `()` or `.py` and the
        package's `qf.` dropped, then the last dotted part (`qf.<shelf>.<name>` and `<shelf>.<name>` alike, whatever the
        shelf) is, in order, a function's name as written (`name`); else a shelf's name alone names no function
        (`shelf`); else, folded as the pointer folds an alias (`folding`: case, accents, hyphens, underscores and
        spaces), a function's name (`folded`), one of its aliases, every one on its card, the common ones' brackets
        dropped (`alias`), or either with its last word's plural or singular (`plural`, -s and -es: `day_counts` meets
        `day count`). A name two functions answer to at the first rule that meets any is resolved to none (`ambiguous`),
        as is one no rule meets (`none`)."""
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
        """Every function's score on the task's text (`Scored`), in the library's order: the count of its `terms` the
        task holds; whether the pointer matches it (`pointers`, every match, not only the first `POINTERS`); whether its
        shelf's line shares `SHELF_TERMS` terms with the task."""
        held = stems_of(terms(task))
        matched = {function: rank for rank, (function, _) in
                   enumerate(self.pointers(task, most=len(self.functions)))}
        found = []
        for shelf in self.shelves:
            touched = len([term for term in terms(shelf.line) if stems_of({term}) & held]) >= SHELF_TERMS
            for function in shelf.functions:
                score = len([term for term in function_terms(function) if stems_of({term}) & held])
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
        """`index` as `index` wrote it, full (`HEAD`) or scored (`SCORED_HEAD`), with one more shelf reduced at each
        step to make room for the cards whole: the shelves no carded function lies on (`touched`, by name) first, the
        last first, then the others, the last first. A reduced shelf is said as the index says one when it holds no room
        for more: in the full index its line and its functions by name (`_named`), in the scored its line and every
        function after `others:`; a shelf its reduction would not shorten keeps its lines and takes no step. [] for a
        text of another form or of another library's shelves."""
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
        """The cards of the functions `pointers` named (above): `head` (`CARDS_HEAD`) and each `Card`, in order, within
        `limit` in all (`card_bound()` by default, in the unit `measured` counts), fitted by `trimmed`. With what was
        cut, one line a card (`qf.<fn>: ...`), for the log; ("", the cuts) where none fits."""
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
        out, the last pointed first. Past `limit` the cards are cut by the pointer's rank: the last pointed loses its
        parts (`Card.trim`) one by one until they fit or it is at its smallest, then the one before it, so a card loses
        nothing while a card pointed after it can still lose a part; where every card at its smallest is still past
        `limit`, the last pointed is left out and the rest fitted again from whole."""
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
    """`made_whole`'s result: the first message's index and cards, and what the cards' fit cut (a line a card, as
    `carded` writes them; none where every card is whole)."""
    index: str
    cards: str
    cuts: tuple[str, ...]


def made_whole(functions: list[Function], index: str, trims: list[str], limit: int | None = None,
               head: str = CARDS_HEAD) -> Whole:
    """`functions`' cards whole (`Card`, nothing trimmed), under `head` and after `index`, within `limit` in all
    (`whole_bound()` by default, in the unit `measured` counts) and never past it: where they do not fit, the first of
    the index's `trims` (in their order) that leaves them room; where even the last leaves too little, the cards fitted
    in the room it leaves (`Library.carded`: the trim by rank, the last carded cut first), and an index past the bound
    alone cut to it."""
    limit = whole_bound() if limit is None else limit

    def joined(text: str, cards: str) -> str:
        return f"{text}\n\n{cards}" if cards else text

    cards = "\n\n".join([head, *(Card(function).text for function in functions)]) if functions else ""
    for text in [index, *trims]:
        if measured(joined(text, cards)) <= limit:
            return Whole(text, cards, ())
    text = trims[-1] if trims else index
    if measured(text) > limit:
        text = clip(text, limit, "", measure.UNIT)
    ranked = [(function, function.name) for function in functions]
    room = limit - measured(f"{text}\n\n")
    cards, cuts = Library.carded(ranked, max(room, 0), head)
    while cards and measured(joined(text, cards)) > limit:     # in tokens the join is not quite the parts' sum
        room -= measured(joined(text, cards)) - limit
        cards, cuts = Library.carded(ranked, max(room, 0), head)
    return Whole(text, cards, tuple(cuts))


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
    """The index and the whole cards' bound in that unit: `WHOLE_TOKENS`, else `WHOLE_BYTES`."""
    return measure.bound(WHOLE_TOKENS, WHOLE_BYTES)


def specific(alias: str) -> bool:
    """Whether the pointer matches `alias`, as its `aliases:` line writes it: two or more words (hyphens, underscores
    and spaces alike), or one word written with a capital (a name or an acronym); never one in brackets."""
    if COMMON.match(alias):
        return False
    words = normal(alias).split()
    return len(words) >= 2 or (len(words) == 1 and alias[:1].isupper())


def words(text: str) -> list[str]:
    """A phrase's words as the pointer folds them (`normal`), split at anything not a letter or a digit."""
    return [word for word in re.split(r"[^a-z0-9]+", normal(text)) if word]


def line_of(function: Function) -> str:
    """A function's line in the index, its card whole."""
    return f"  {function.call}" + (f": {function.card}" if function.card else "")


def terms(text: str) -> set[str]:
    """A text's terms: its words as the pointer folds them (`words`), but a lone letter, a number and `FILLER`, its
    plurals too (`stems_of`)."""
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
