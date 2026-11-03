"""N11's last resort (lane fallback1; the owner's in520, agenthon-t2): where the loop would end with nothing the harness
could grade, N0's one-shot request, its script run once.

`AGENTHON_N11_FALLBACK` (or agent/n11/FALLBACK, one word, an image's; agent/n11/solve.py `fallback_of`): `off`, today's,
or `oneshot`, armed in the loop's forms `loop` and `sketch` (`FORMS`; form `review`'s request 1 is N0's own already, and
form `state` keeps its own ends). Armed, the loop holds a reserve back from the start, seconds off its clock and one
request off its cap, and releases it for good at the first run that leaves an output to grade (`graded`): a unit with an
output never pays for the fallback. Where the loop then stops with nothing to grade (`REASONS`: its requests spent, its
clock under a request's, replies with no call, or a finish), `Resort.fire` spends the reserve on agent/n0's one-shot
request, by import: N0's system prompt, its user turn (the instruction, the paths, the data's digest) carrying the input
files as the loop's first message describes them and, where the pointer named functions, their cards (`user_turn`); its
reply parsed by N0's own `ask` and its script written and run once by N0's own `execute`, with all the time left. N0's
continuation of a reply the cap cut is sent only where the loop left a request beyond the reserved one and the clock a
request beyond the reserve's floor (`CONTINUE_S`). The unit ends on whatever the script wrote. Nothing of this is said
to the model in the loop: its bodies differ only by the budget it is told while the reserve is held.

Lane repeat1 sends it mid-loop too (agent/n11/solve.py `repeat_handoff`, reason `REPEAT`): at the unit's third
recurrence of a failing program state under `AGENTHON_N11_REPEAT=change`, once a unit like the rest, its run bounded to
a loop run's (`fire`'s `run_s`), the loop going on after it.

The reserve's time (lane fallback2; lane fallback1's live smoke, where a loop request ran 80 s into the reserve and a
reply at the cap took two to three times it). While it is held, each loop request is bounded by the loop's own clock,
the time left before the reserve begins (`bounded`; src/agenthon_t1/model.py's `bound`), so none runs into it: a
request cut there ends the loop on its clock (`cuts`), and the one-shot follows. The reserve is sized from the route's
observed pace (`sized`): a reply at the output cap (`CAP_TOKENS`) at the unit's own rate, the median of its loop
replies' completion tokens a second so far (`observe`), its script's run (`RUN_S`) and a margin (`MARGIN_S`), within
`FLOOR_S` (lane fallback1's 90 s, and the reserve before the first reply) and `CEILING_S`. The one-shot is bounded too,
to the clock left less its script's run, and is not sent where that is under a request's least (`skipped` `time`);
where its first attempt brings no reply and the clock still holds a request's least and a request is left, it is sent
once more, else not.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from pathlib import Path

from common import trace
from common.house import model
from n0 import solve as n0

ENV = "AGENTHON_N11_FALLBACK"   # the desk's switch (src/agenthon_t1/harness.py hands it to an --agent run's container)
FILE = "FALLBACK"               # or agent/n11/FALLBACK, one word, written by an image's builder
SETTINGS = ("off", "oneshot")   # the default first
FORMS = ("loop", "sketch")      # the forms it is armed in
REQUEST_S = 70.0    # the one-shot's request at the platform's pace: its reply is a whole script near the output cap
#                     (the baseline's median reply on the dev records, 3,745 tokens), so the 90th percentile of the
#                     platform's requests (63 to 73 s on 10-07, docs/challenge.md), not lane pace1's median (44 s)
RUN_S = 20.0        # and its script's one run: N0's first runs on the dev records take 0.8 s at the median and 2.6 s
#                     at the 90th percentile (E072, lane fallback2), the loop's 0.9 s and 2.1 s (the 10-11 record)
FLOOR_S = REQUEST_S + RUN_S     # the reserve held off the loop's clock before its first reply, and its least
CEILING_S = 120.0   # and its most: past it the cost begins (lane fallback2's offline read of seven records: at 150 s
#                     one solved loop of E093's stopped before its first output, at 135 s it came 0.1 s inside the
#                     line, at 120 s every solved one keeps 15 s); the one-shot's bound then reaches 100 s, past the
#                     platform's slowest request measured, 92 s (docs/challenge.md)
MARGIN_S = 10.0                 # beyond a reply at the cap at the unit's rate and the script's run
CAP_TOKENS = model.MAX_OUTPUT_TOKENS    # the one-shot's reply is a whole script near the House's output cap
CONTINUE_S = FLOOR_S + REQUEST_S    # N0's continuation of a cut reply asked only with this much left at the start
REPEAT = "repeat"     # lane repeat1's: the third recurrence of a failing program state, mid-loop
REASONS = ("requests", "clock", "no_call", "finish", REPEAT)    # why it is sent: the loop's stops it answers, and
#                                                                 lane repeat1's hand-off, which no stop is named
SKIPS = ("time", "requests")    # why a one-shot due is not sent: its bound under a request's least, or no request left
AGAIN = (model.ModelCallError, model.RequestCapReached)     # a one-shot attempt with no reply sent once more: a route's
#                                 failure or a cut, or the client's own retry refused at the one-shot's cap; never the
#                                 House's refusal, the unit's clock or the credential
KIND = "fallback"               # its requests' kind on the trace (common.house.Client.kind)
WORK = "fallback"               # its script's directory beside work/
KEYS = ("fallback_armed", "fallback_used", "fallback_reason", "fallback_wrote", "fallback_admissible",
        "fallback_reserve_s", "fallback_bound_cuts", "fallback_skipped")
LIBRARY = "The script may also import qf, a method library on numpy and scipy installed beside those packages."
CUT = re.compile(r"\[cut to fit: [^\]\n]*\]")   # a trimmed card's last line names `lib`, which the one-shot lacks
CUT_SAID = "[cut to fit]"


def graded(out: Path, named) -> bool:
    """Whether the output holds a file the harness grades: a regular file at its root besides the scorer's own (N0's
    `delivered`, the platform's g1) and, where the task names output files, one of them."""
    out = Path(out)
    return n0.delivered(out) and (not named or any((out / name).is_file() for name in named))


def user_turn(task, pathmap, data: str, described: str, cards: str, seconds: float) -> str:
    """N0's user turn (agent/n0 `prompt`) with the input files described, as the loop's first message describes them,
    after the data's digest and, where the pointer named functions, the library's section after them (`LIBRARY`, the
    cards, a trimmed card's `lib` line said as cut): both in the slot of N0's `# Input data`, under headings of their
    own, so N0's words stand whole around them."""
    shown = data.strip()
    if described:
        shown += f"\n\n# Input files, described\n\n{described}"
    if cards:
        shown += f"\n\n# The library\n\n{LIBRARY}\n\n{CUT.sub(CUT_SAID, cards)}"
    return n0.prompt(task, pathmap, shown, seconds)


def sized(rate: float) -> float:
    """The reserve at `rate` completion tokens a second: a reply at the cap, the script's run and the margin, within
    the floor and the ceiling."""
    return max(FLOOR_S, min(CEILING_S, CAP_TOKENS / rate + RUN_S + MARGIN_S))


class Bounded:
    """The loop's client as N0's one-shot uses it: each request bounded to end `RUN_S` before the loop's clock
    (`left`, the seconds it reads), so the script's run keeps its time; every other attribute the client's own."""

    def __init__(self, client, left) -> None:
        self._client, self._left = client, left

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    def chat(self, system: str, user: str, **kwargs):
        return self._client.chat(system, user, bound=self._left() - RUN_S, **kwargs)


def hashes(out: Path) -> dict[str, str]:
    """Each regular file under the output directory and its bytes' sha256."""
    found = {}
    try:
        for path in sorted(Path(out).rglob("*")):
            if path.is_file() and not path.is_symlink():
                found[path.relative_to(out).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        pass
    return found


class Resort:
    """The switch over one unit: armed (`oneshot`, in one of `FORMS`), the reserve held, and what the fallback did."""

    def __init__(self, setting: str, source: str, form: str) -> None:
        self.setting, self.source = setting, source
        self.armed = setting == "oneshot" and form in FORMS
        self.holding = self.armed
        self.used = False
        self.reason = ""
        self.wrote = 0
        self.admissible: bool | None = None
        self.agent: n0.N0 | None = None     # N0's, while the one-shot runs: the watchdog's kill reaches its script
        self.reserve = FLOOR_S              # lane fallback2: the reserve's seconds, sized while it is held
        self.rates: list[float] = []        # the loop's replies' completion tokens a second while it is held
        self.cuts = 0                       # loop requests its bound cut
        self.skipped = ""                   # why a one-shot due was not sent (`SKIPS`), "" where none was skipped

    def held_s(self) -> float:
        """The seconds held off the loop's clock."""
        return self.reserve if self.holding else 0.0

    def held_requests(self) -> int:
        """The requests held off the loop's cap."""
        return 1 if self.holding else 0

    def bounded(self, left: float) -> dict:
        """A loop request's keyword for the client while the reserve is held: its bound, the `left` seconds before the
        reserve begins; none where it is not held, the request as before."""
        return {"bound": left} if self.holding else {}

    def observe(self, tokens: int, seconds: float) -> str:
        """A loop reply's completion tokens and its seconds on the unit's clock, while the reserve is held: the unit's
        rate, the median of its replies' tokens a second, and the reserve sized at it (`sized`); the change said for
        the log, "" where the whole seconds stand (or a reply took no time, under a fixed clock)."""
        if not self.holding or tokens <= 0 or seconds <= 0:
            return ""
        self.rates.append(tokens / seconds)
        rate = statistics.median(self.rates)
        was, self.reserve = self.reserve, sized(rate)
        if round(was) == round(self.reserve):
            return ""
        return f"the reserve {self.reserve:.0f} s: a reply at the cap at {rate:.1f} tokens a second, and the run"

    def kill(self) -> None:
        if self.agent is not None:
            self.agent.runner.kill()

    def keys(self) -> dict:
        """The summary line's keys (`KEYS`): `fallback_reason` and `fallback_admissible` `-` where it was not sent."""
        return {"fallback_armed": int(self.armed), "fallback_used": int(self.used),
                "fallback_reason": self.reason or "-", "fallback_wrote": self.wrote,
                "fallback_admissible": "-" if self.admissible is None else int(self.admissible),
                "fallback_reserve_s": round(self.reserve) if self.armed else "-", "fallback_bound_cuts": self.cuts,
                "fallback_skipped": self.skipped or "-"}

    def fire(self, loop, task, pathmap, data: str, reason: str, least: float, run_s: float | None = None) -> None:
        """The one-shot (above), once a unit, for the loop `loop` (agent/n11/solve.py's N11) that stopped for `reason`
        with nothing to grade: the reserve released; sent where a request is left and its bound, the clock left less
        the script's run, is `least` seconds or more, else skipped (`SKIPS`); sent once more where its first attempt
        brought no reply (`AGAIN`) and a request and that bound are still left; the files the run wrote or changed
        (`wrote`) and whether the output then holds a file to grade (`admissible`). Lane repeat1's `run_s`, the loop
        going on after it: the script's run and the seconds its user turn states at most that."""
        self.holding = False
        client, left = loop.client, loop.left()
        spare = client.cap - client.usage.requests
        if spare < 1 or left - RUN_S < least:
            self.skipped = "requests" if spare < 1 else "time"
            loop.log("fallback", f"{reason}: not sent: " + ("no request left" if spare < 1 else
                                                            f"{max(left, 0):.0f} s left, under {least + RUN_S:.0f} s "
                                                            "for a request and the run"))
            trace.event("fallback", reason=reason, sent=False, skipped=self.skipped)
            return
        self.used, self.reason = True, reason
        allowed = 2 if spare >= 2 and left >= CONTINUE_S else 1
        work = Path(loop.work) / WORK
        work.mkdir(parents=True, exist_ok=True)

        def said(step: str, detail: str = "") -> None:
            loop.log("fallback", f"{step}: {detail}" if detail else step)

        self.agent = one = n0.N0(task.task_dir, loop.out, Bounded(client, loop.left), said, work,
                                 loop.deadline + loop.moved())
        one.unit = task
        stated = left - REQUEST_S if run_s is None else min(left - REQUEST_S, run_s)
        user = user_turn(task, pathmap, data, loop.described(), loop.cards, max(n0.MIN_RUN_S, stated))
        said("sent", f"{reason}: N0's one-shot request with {left:.0f} s and {spare} request(s) left"
                     + (", its continuation allowed" if allowed > 1 else ""))
        before = hashes(loop.out)
        cap, client.kind, code, attempts = client.cap, KIND, None, 0
        try:
            while True:
                attempts += 1
                client.cap = client.usage.requests + (allowed if attempts == 1 else 1)
                try:
                    code = one.ask(user, "request")
                    break
                except model.ModelError as exc:
                    said("request", f"no reply ({type(exc).__name__})")
                    if attempts > 1 or not isinstance(exc, AGAIN) or cap - client.usage.requests < 1 or \
                            loop.left() - RUN_S < least:
                        break
                    said("again", f"one more attempt with {loop.left():.0f} s and {cap - client.usage.requests} "
                                  "request(s) left")
        finally:
            client.cap = cap
        if code is not None:
            one.deadline = loop.deadline + loop.moved()
            if run_s is not None:   # N0's `left` reads the loop's clock less the same reserve: its run `run_s` at most
                one.deadline = min(one.deadline, one.deadline - loop.left() + run_s)
            one.execute(code, last=True)
        after = hashes(loop.out)
        self.wrote = sum(after[name] != before.get(name) for name in after)
        self.admissible = graded(loop.out, loop.ws.named)
        said("end", f"{self.wrote} file(s) written; " + ("an output to grade" if self.admissible else
                                                         "nothing to grade"))
        trace.event("fallback", reason=reason, sent=True, attempts=attempts, script=code is not None,
                    wrote=self.wrote, admissible=self.admissible)
