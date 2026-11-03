"""N11's worked example (lane example1; the cold reader's form 4, runs/lanes/cold-velocity-proposals.toml; the error
reads' largest class, docs/research/levers-1015.md; the owner's in25 and in767): under `EXAMPLE=nearest` the loop's
first message shows one public task the candidate solved on every recorded reading, the nearest to this task by the
words of the two instructions: its instruction, its input files' first lines and the script that passed its checks.

- **The corpus** (`CORPUS` beside this file, one JSON line an `Entry`): built on the desk by
  scripts/examples_corpus.py, whose docstring is its rebuild recipe; read once a path where the switch arms it
  (`load`), a malformed line skipped and counted.
- **Retrieval** (`nearest`): the Jaccard overlap of two word sets (`words`: lower-cased runs of three or more letters,
  `STOP` left out) between this task's instruction and each entry's; the highest first, a tie to an entry of the task's
  family (`family`: the card's category through `FAMILIES`, which a test holds equal to agenthon_t1.units'), then to
  the name first in order. Never this task, its stress twin or its source (`kin`), nor an entry whose unit or split
  the exclusion names (`EXCLUDE_ENV`, `excluded`), nor one whose overlap reaches `SAME`: the image recognises a unit
  by its content too, or whose nonempty set of named output files equals this task's. The challenge's rule 8 lets
  our solutions to public units travel as examples for other units only (docs/challenge.md). An overlap below
  `MIN_OVERLAP` shows nothing; ranking remains Jaccard.
- **The text** (`shown`): `HEAD`; the instruction quoted within `INSTRUCTION_TOKENS`; the input files' headers within
  `HEADERS_TOKENS`; the script in a fenced block within `SCRIPT_TOKENS`, a cut said in one line (`CUT`); `CLOSE`. The
  instruction cut keeps its head and an output/deliverables section where headed; an oversized output section is
  itself cut. The whole within `WHOLE_TOKENS`, a budget of its own beside the input files' description's, the script
  cut further where the rest leaves it less. Each bound in the model's tokens, else its calibrated bytes (agent/n11/measure.py);
  no count is said to the model (the owner's in281).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from common import paths
from n11 import measure

HERE = Path(__file__).resolve().parent
ENV, FILE = "AGENTHON_N11_EXAMPLE", "EXAMPLE"   # the desk's switch, or agent/n11/EXAMPLE, one word, an image's
SETTINGS = ("off", "nearest")                   # `off` (today's: no example) or the nearest solved task's
FORMS = ("loop", "review", "sketch")            # armed in the loop's forms (form `state` builds its own message)
KEYS = ("example_mode", "example_unit", "example_overlap", "example_tokens", "example_cut")
EXCLUDE_ENV = "AGENTHON_N11_EXAMPLE_EXCLUDE"    # the entries retrieval skips, unit and split names joined by commas;
#                                                 the desk's alone (a read's registration names it), an image none
SPLITS = ("dev", "holdout", "fresh-dev", "fresh-drafts")   # the open splits an exclusion may name
CORPUS = "examples.jsonl"
KIT_PREFIX, STRESS_PREFIX = "t1-", "t1s-"       # agenthon_t1.units' `twin` and `source`, which the image lacks

# the bounds, in the model's tokens, and in bytes where the tokenizer does not load: each token target at its text
# kind's 10th-percentile bytes a token over the corpus of 10-16 (scripts/examples_corpus.py prints them: instructions
# 2.75, scripts 3.12, headers 2.62; the whole at the densest part's), so the byte bound rarely passes the target
WHOLE_TOKENS, WHOLE_BYTES = 3_000, 8_250
INSTRUCTION_TOKENS, INSTRUCTION_BYTES = 600, 1_650
HEADERS_TOKENS, HEADERS_BYTES = 200, 524
SCRIPT_TOKENS, SCRIPT_BYTES = 2_000, 6_240
HEADER_CHARS = 300                              # a header's line, at most
SAME = 0.95     # an overlap taken for this task itself (rule 8: never its own solution, by id or by content): a kit
#                 unit and its reworded stress twin overlap at 0.970 at the median, two distinct units of one template
#                 at most 0.928 in the corpus of 10-16, and no unit's nearest of the listed reads passed 0.661
MIN_OVERLAP = 0.12  # below this the example shares too little vocabulary; the paired read must judge this floor
OUTPUT_HEADING = re.compile(r"^(#{1,6})[ \t]+[^\n]*\b(?:outputs?|deliverables?)\b[^\n]*$", re.I | re.M)
HEADING = re.compile(r"^(#{1,6})[ \t]+", re.M)

WORD = re.compile(r"[a-z]{3,}")
STOP = frozenset("""
the and for with from that this these those each its into onto are was were been being has have had not but all any
can may must should will shall would could than then them they their there here which where when what who whom how why
also only both such per via use used using one two three first last other same some more most less over under after
before between within without while about above below upon your you our out off own very just does did done ever
every given give gives file files data input output app json csv write writes written read reads save saved
""".split())

HEAD = "# A solved example: another task near this one, not this task"
SAID_INSTRUCTION = "Its instruction:"
SAID_INPUTS = "Its input files, each with its first line or its keys:"
SAID_SCRIPT = "The script that passed its checks:"
CUT_INSTRUCTION = "(parts of its instruction are not shown)"
CUT = "(the script is cut here; its rest is not shown)"
MORE = "- more files, not shown"
CLOSE = ("It shows file names, columns, rounding and sign conventions of the kind the checks read; none of its numbers "
         "belong to this task.")

# the families of agenthon_t1.units (`FAMILIES`): a card's category string to its family, `other` where none names it
FAMILIES: dict[str, tuple[str, ...]] = {
    "cross-domain": ("cross-domain",),
    "derivatives-pricing": ("derivatives-pricing", "derivatives_pricing", "derivatives", "pricing",
                            "interest-rate-derivatives", "fx-pricing", "options-pricing", "credit-derivatives",
                            "fixed-income-derivatives", "fx-derivatives", "volatility-derivatives"),
    "risk-management": ("risk-management", "risk-modeling", "credit-risk", "credit-analysis", "extreme-value-theory",
                        "dependence-modeling", "volatility-modeling", "risk"),
    "fixed-income": ("fixed-income", "fixed-income-nlp", "cross-currency-rates", "interest-rate-models"),
    "strategies": ("backtesting", "factor-research", "factor-models", "execution", "strategy", "fx-strategy",
                   "cross-sectional-strategies", "predictive-alpha-modeling", "trading-strategy",
                   "quantitative-strategy", "quantitative-trading", "systematic-trading", "statistical-arbitrage",
                   "technical-analysis"),
    "other": ("tool-using", "portfolio-analysis", "performance-attribution", "statistical-analysis",
              "event-driven-analysis, data-processing", "crypto", "cross-asset-analysis", "stochastic-processes"),
    "portfolio": ("portfolio-management", "portfolio-optimization", "portfolio-theory"),
}
_FAMILY = {category: name for name, categories in FAMILIES.items() for category in categories}


@dataclass(frozen=True)
class Entry:
    """One solved task of the corpus: its name, split and family, its instruction, its input files' headers (each
    `[name, header]`) and its passing script."""
    unit: str
    split: str
    family: str
    instruction: str
    inputs: tuple[tuple[str, str], ...]
    script: str
    words: frozenset[str] = field(default=frozenset(), compare=False)


@dataclass(frozen=True)
class Shown:
    """The example as the first message carries it (`text`, "" where none), and its keys' values."""
    text: str = ""
    unit: str = ""
    overlap: float = 0.0
    tokens: int = 0
    cut: tuple[str, ...] = ()       # the parts cut: `instruction`, `headers`, `script`

    def keys(self, setting: str) -> dict:
        """`KEYS` as the summary line says them: the setting, and `-` (or 0) where no example was shown."""
        return {"example_mode": setting, "example_unit": self.unit or "-",
                "example_overlap": f"{self.overlap:.3f}" if self.unit else "-", "example_tokens": self.tokens,
                "example_cut": int(bool(self.cut))}


def words(text: str) -> frozenset[str]:
    return frozenset(WORD.findall(text.lower())) - STOP


def overlap(one: frozenset[str], other: frozenset[str]) -> float:
    """The Jaccard overlap of two word sets: the shared words over all; 0 where both are empty."""
    union = one | other
    return len(one & other) / len(union) if union else 0.0


def family(category: str) -> str:
    return _FAMILY.get(category, "other")


def source(name: str) -> str:
    """The kit unit a stress twin varies; any other name is its own (agenthon_t1.units.source)."""
    return KIT_PREFIX + name.removeprefix(STRESS_PREFIX) if name.startswith(STRESS_PREFIX) else name


def twin(name: str) -> str:
    """A kit unit's stress twin's name (agenthon_t1.units.twin)."""
    return STRESS_PREFIX + name.removeprefix(KIT_PREFIX)


def kin(name: str) -> frozenset[str]:
    """This task, its source and its source's twin: the names retrieval never shows; none where it has no name."""
    return frozenset({name, source(name), twin(source(name))}) if name else frozenset()


def exclusion(said: str) -> frozenset[str]:
    """The exclusion's comma list as names: each stripped, blank ones dropped."""
    return frozenset(part.strip() for part in (said or "").split(",") if part.strip())


def excluded(entry: Entry, names: frozenset[str]) -> bool:
    return entry.unit in names or entry.split in names


def parsed(line: str) -> Entry | None:
    """One corpus line as an `Entry`; None where it is no entry."""
    try:
        row = json.loads(line)
        entry = Entry(str(row["unit"]), str(row["split"]), str(row["family"]), str(row["instruction"]),
                      tuple((str(name), str(header)) for name, header in row["inputs"]), str(row["script"]))
    except (ValueError, KeyError, TypeError):
        return None
    return Entry(entry.unit, entry.split, entry.family, entry.instruction, entry.inputs, entry.script,
                 words(entry.instruction))


@cache
def load(path: Path) -> tuple[tuple[Entry, ...], int]:
    """The corpus at `path`: its entries, and the lines that were none (`parsed`); nothing where the file is not
    there."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return (), 0
    entries = [parsed(line) for line in lines if line.strip()]
    return tuple(entry for entry in entries if entry is not None), sum(entry is None for entry in entries)


def nearest(name: str, instruction: str, category: str, entries, exclude: frozenset[str] = frozenset(), *,
            outputs: tuple[str, ...] | None = None
            ) -> tuple[Entry | None, float]:
    """The entry nearest this task (`overlap` of `words`), never its kin, an excluded one nor one at `SAME` or over
    (this task by its content), nor the same nonempty set of named output files; a tie to its family's, then to the
    name first in order. (None, 0.0) where none reaches `MIN_OVERLAP`. Callers with the task's input inventory supply
    `outputs` from the existing output-name reader, so bare input filenames cannot disguise a match."""
    own, mine, never = words(instruction), family(category), kin(name)
    named = frozenset(paths.outputs(instruction) if outputs is None else outputs)
    scored = [(score, entry) for entry in entries if entry.unit not in never and not excluded(entry, exclude)
              if not named or named != frozenset(paths.outputs(entry.instruction, (n for n, _ in entry.inputs)))
              for score in [overlap(own, entry.words)] if score < SAME]
    best = min(scored, key=lambda pair: (-pair[0], pair[1].family != mine, pair[1].unit), default=None)
    if best is None or best[0] < MIN_OVERLAP:
        return None, 0.0
    return best[1], best[0]


def lines_within(text: str, room: int) -> tuple[str, bool]:
    """`text` within `room` in the measure's unit, cut after its last whole line inside it; and whether it was cut."""
    if measure.size(text) <= room:
        return text, False
    head = measure.prefix(text, room)
    return (head.rsplit("\n", 1)[0] if "\n" in head else ""), True


def instruction_within(text: str, room: int) -> tuple[str, bool]:
    """An instruction within its own room: keep its head and a headed output section when cut, reserving up to
    half the room for that section. Single long paragraphs may end mid-line; a script still cuts at whole lines."""
    if measure.size(text) <= room:
        return text, False
    section = OUTPUT_HEADING.search(text)
    if section is None or section.start() == 0:
        return measure.prefix(text, room), True
    end = next((m.start() for m in HEADING.finditer(text, section.end()) if len(m[1]) <= len(section[1])), len(text))
    output = measure.prefix(text[section.start():end].strip(), room // 2)
    gap = "\n\n...\n\n"
    head = measure.prefix(text[:section.start()].rstrip(), max(0, room - measure.size(gap + output)))
    said = head + gap + output
    while measure.size(said) > room and head:
        head = measure.prefix(head, max(0, measure.size(head) - 1))
        said = head + gap + output
    return said, True


def fence(script: str) -> str:
    """A fence longer than any run of backticks the script holds, so the block closes where it should."""
    longest = max((len(run) for run in re.findall(r"`+", script)), default=0)
    return "`" * max(3, longest + 1)


def headers_said(inputs: tuple[tuple[str, str], ...], room: int) -> tuple[str, bool]:
    """The input files' headers within `room`: one item a file, its name and its header's first line, then the
    header's other lines indented, each within `HEADER_CHARS`, line by line while they fit; what does not fit said in
    one line (`MORE`)."""
    lines = [line for name, header in inputs for line in [
        f"- {name}: {(header.splitlines() or ['(empty)'])[0][:HEADER_CHARS]}",
        *(f"  {line[:HEADER_CHARS]}" for line in header.splitlines()[1:])]]
    if measure.size("\n".join(lines)) <= room:
        return "\n".join(lines), False
    said: list[str] = []
    for line in lines:
        if measure.size("\n".join([*said, line, MORE])) > room:
            break
        said.append(line)
    return "\n".join([*said, MORE]), True


def shown(entry: Entry | None, score: float = 0.0) -> Shown:
    """The example's text for `entry` (`Shown()` where none): the parts within their bounds and the whole within
    `WHOLE_TOKENS`, the script cut to what the rest leaves it."""
    if entry is None:
        return Shown()
    instruction, instruction_cut = instruction_within(entry.instruction.strip("\n"),
                                                      measure.bound(INSTRUCTION_TOKENS, INSTRUCTION_BYTES))
    quoted = "\n".join(f"> {line}".rstrip() for line in instruction.splitlines())
    headers, headers_cut = headers_said(entry.inputs, measure.bound(HEADERS_TOKENS, HEADERS_BYTES))
    script = entry.script.strip("\n")
    marks = fence(script)

    def text(body: str, cut: bool) -> str:
        parts = [HEAD, SAID_INSTRUCTION, quoted + (f"\n{CUT_INSTRUCTION}" if instruction_cut else "")]
        if headers:
            parts += [SAID_INPUTS + "\n" + headers]
        block = f"{marks}python\n{body}\n{marks}" + (f"\n{CUT}" if cut else "")
        return "\n\n".join([*parts, SAID_SCRIPT, block, CLOSE])

    whole = measure.bound(WHOLE_TOKENS, WHOLE_BYTES)
    room = min(measure.bound(SCRIPT_TOKENS, SCRIPT_BYTES), whole - measure.size(text("", True)))
    body, script_cut = lines_within(script, max(room, 0))
    said = text(body, script_cut)
    while measure.size(said) > whole and body:     # tokens merging across the block's edges: the script cut again
        room -= measure.size(said) - whole
        body, script_cut = lines_within(script, max(room, 0))[0], True
        said = text(body, script_cut)
    cut = tuple(part for part, was in (("instruction", instruction_cut), ("headers", headers_cut),
                                       ("script", script_cut)) if was)
    return Shown(said, entry.unit, score, measure.size(said), cut)
