"""A bounded thinking budget on the loop's write replies (lane thinkwrite1; the owner's in53 and in55, the wrong clean
answer finished early, docs/research/errors.md): `AGENTHON_N11_WRITE_THINKING=off|first|all`, `off` by default, armed
in forms `loop`, `review` and `sketch`, and `AGENTHON_N11_WRITE_THINKING_BUDGET`, the reasoning tokens a request asks
(a whole number within `BOUNDS`, `BUDGET` unset) or the word `low`, the House's low effort. Text alone: nothing here
sends a request; agent/n11/solve.py asks each loop request's thinking here (`Ledger.asks`) and hands the client the
template's argument (`thinking=<budget>`, src/agenthon_t1/model.py `thinking_template`), and counts each reply that
asked (`Ledger.observe`).

The request a reply answers is read by the harness's state when it is sent (`KINDS`): `continuation` while a cut
write's head is held for its rest; `resend` after a reply whose write was refused as unchanged; `fix` after a finish
not taken (the unit went on); then the short clock's rule (lane pace1): `finish` where the last run was clean with
every named output present and no finding, `write` where work/main.py is not written, else `fix`. Under `first` every
loop request before the unit's first reply that wrote or was cut asks (in practice request 1 of the loop); under
`all` every `write` and `fix` request asks. A continuation, the finish and the request after a re-send never ask under
either word, nor the requests outside the loop's own (the selection call, form `review`'s library pass and its
continuations of request 1); form `sketch`'s turn, form `review`'s review request and a request carrying the sketch's
ask under `SKETCH_THINKING=low` keep their own switches. Under `off` no request asks and every body is today's.

The House counts reasoning inside the reply's `max_tokens` (docs/challenge.md, "Thinking"; t1#28 c2): a budget B
leaves the code about 4,000 - B tokens where the serving honours it, which the House's docs leave untested
(HOUSE-MODEL.md, "Development thinking controls"). So each reply that asked is read: its reasoning tokens summed
(`write_thinking_reasoning`), those over the budget counted (`write_thinking_over`, the serving not honouring it), and
a reply the cap cut whose reasoning took more than half of it stands the switch down for the rest of the unit
(`write_thinking_dropped`): thinking ate the reply, and asking again would spend the next one the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

ENV = "AGENTHON_N11_WRITE_THINKING"
FILE = "WRITE_THINKING"
SETTINGS = ("off", "first", "all")
FORMS = ("loop", "review", "sketch")        # form `state` builds its own requests
BUDGET_ENV = "AGENTHON_N11_WRITE_THINKING_BUDGET"
BUDGET_FILE = "WRITE_THINKING_BUDGET"
BUDGET = 800                    # reasoning tokens a request asks where no budget is named
BOUNDS = (100, 3000)            # a budget outside them refused: under 100 the House barely starts; over 3,000 leaves
#                                 the code under a quarter of the 4,000-token reply
LOW = "low"                     # the budget word asking the House's `low_effort` instead of a number
KINDS = ("write", "fix", "continuation", "resend", "finish")
WRITES = ("write", "fix")       # the kinds that ask: under `all` each, under `first` those before the first write
KEYS = ("write_thinking_mode", "write_thinking_budget", "write_thinking_asked", "write_thinking_reasoning",
        "write_thinking_over", "write_thinking_dropped")


def words() -> tuple[str, ...]:
    """The budget's words, as the desk's row says them: every whole number within `BOUNDS`, then `LOW`."""
    return (*(str(n) for n in range(BOUNDS[0], BOUNDS[1] + 1)), LOW)


def budget_of(said: str, source: str) -> tuple[int | str, str]:
    """The budget a word names and where it was read: `LOW`, or a whole number within `BOUNDS`; nothing named,
    `BUDGET`; any other word refused, `BUDGET` run and the source saying why."""
    if not said:
        return BUDGET, "no budget named"
    if said == LOW:
        return LOW, source
    if not (said.isascii() and said.isdigit()):     # `²` is a digit to isdigit and no number to int
        return BUDGET, f"{source} names {said[:40]!r}, no budget: refused"
    low, high = BOUNDS
    if not low <= int(said) <= high:
        return BUDGET, f"{source} names {int(said)}, {'above' if int(said) > high else 'below'} " \
                       f"{high if int(said) > high else low}: refused"
    return int(said), source


def kind_of(held: bool, resent: bool, finish_not_taken: bool, state: str) -> str:
    """The request's kind by the harness's state at its send (`KINDS`): a held head's rest, the answer to an
    unchanged re-send, the answer to a finish not taken, else `state`, the short clock's reading of the last run
    (solve.py `state_kind`: `finish`, `write` or `fix`)."""
    if held:
        return "continuation"
    if resent:
        return "resend"
    return "fix" if finish_not_taken else state


@dataclass
class Ledger:
    """One unit's asks: the setting (`off` outside `FORMS`), the budget, and the replies that asked."""
    setting: str
    budget: int | str
    asked: int = 0
    reasoning: int = 0
    over: int = 0
    dropped: int = 0

    def asks(self, kind: str, first_open: bool) -> dict:
        """The client's `thinking` argument for a request of `kind` (`{}`: today's, thinking off): under `first` while
        the unit's first write is still to come (`first_open`), under `all` on a write or a fix; never once stood
        down."""
        if self.dropped or kind not in WRITES:
            return {}
        return {"thinking": self.budget} if self.setting == "all" or self.setting == "first" and first_open else {}

    def observe(self, reasoning: int, completion: int, cut: bool) -> str:
        """A reply that asked: its reasoning counted, over the budget where a number was asked, and the switch stood
        down where the cap cut it with more than half of it reasoning. What it said, for the log."""
        self.asked += 1
        self.reasoning += reasoning
        over = isinstance(self.budget, int) and reasoning > self.budget
        self.over += over
        said = f"{reasoning:,} reasoning of {completion:,} completion tokens" + ("; over the budget" if over else "")
        if cut and 2 * reasoning > completion and not self.dropped:
            self.dropped = 1
            said += "; cut at the cap with most of it reasoning: no later request asks"
        return said

    def keys(self, armed: bool) -> dict:
        """`KEYS`: the setting and budget (`-` outside `FORMS`, the budget `-` under `off` too) and the counts."""
        return {"write_thinking_mode": self.setting if armed else "-",
                "write_thinking_budget": self.budget if armed and self.setting != "off" else "-",
                "write_thinking_asked": self.asked, "write_thinking_reasoning": self.reasoning,
                "write_thinking_over": self.over, "write_thinking_dropped": self.dropped}
