"""The House through the one client, model.py, under the agent's own request cap.

In the image model.py sits beside this package as /opt/agent/model.py; on the desk it is the program's
`agenthon_t1.model`, which finds the desk's model file from lab.toml by itself. `Client` is model.Client with one
more refusal: a request past the agent's `cap` (below the House's 25) is refused unsent, as the House's own cap is,
attempts and retries counted alike. The unit's counts are safe under threads (W's concurrent drafts, agent/w1): `LOCK`
holds every change to them and every write of the usage file, so the cap refuses the request past it whichever
thread asks. Under the desk's trace (`common.trace`) each request is written there with its prompt, reply, tokens,
latency and the attempts its own thread made; with the trace off, `chat` is model.Client's own. An agent that runs in
forms names its form on the usage line (`form`, agent/n11's since lane loop1); unset, the line is model.Client's. An
agent that names its requests' kinds (`kind`, agent/n11's since lane cards1) has each written on its trace's request
line; unset, the line is as before. An agent's own keys (`marks`, agent/n11's form `state` since lane state2) follow
the form on the usage line; unset, the line is as before. `next_send` and `last_sent` read the desk's pacing of the sends (agent/n11's last call).
`sibling` makes a client of its own for a request sent beside this one's (agent/n11's drafts, lane drafts1): the unit's
one count and cap, its own pacing floor; `carry` moves this client's deadline past the untimed waits its siblings made.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

from . import trace

try:
    import model
except ImportError:                     # the desk: agent/ on the path, the program's package installed
    from agenthon_t1 import model


# One process runs one unit, so one lock for the unit's counts, whichever clients and threads the agent has
LOCK = threading.RLock()


class Client(model.Client):
    cap = model.REQUESTS_PER_UNIT
    form: str | None = None             # the agent's form, named on the usage line where set
    kind: str | None = None             # the next request's kind, named on its trace line where set
    marks: dict | None = None           # the agent's own keys, after the form on the usage line where set

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sent = threading.local()  # each thread's requests, so a traced call counts its own attempts alone

    def _admit(self) -> float:
        with LOCK:
            if self.usage.requests >= self.cap:
                self.usage.refused += 1
                self._write_usage()
                raise model.RequestCapReached(f"request {self.usage.requests + 1} of the unit refused: the agent's "
                                              f"cap is {self.cap}")
            timeout = super()._admit()
        self._sent.count = self.sent() + 1
        return timeout

    def _turned_away(self) -> None:
        with LOCK:
            super()._turned_away()
        self._sent.count = self.sent() - 1

    def _rate_uncharged(self) -> None:
        with LOCK:
            super()._rate_uncharged()
        self._sent.count = self.sent() - 1

    def _count(self, prompt: int, completion: int, cost: float | None) -> None:
        with LOCK:
            super()._count(prompt, completion, cost)

    def _reply(self, doc: dict, thinking_off: bool = False):
        with LOCK:
            return super()._reply(doc, thinking_off)

    def _write_usage(self) -> None:
        with LOCK:
            super()._write_usage()

    def usage_line(self) -> dict:
        """model.Client's usage line, the agent's `form` after it where one is set, and its `marks` after that where
        any are."""
        line = super().usage_line()
        line = {**line, "form": self.form} if self.form else line
        return {**line, **self.marks} if self.marks else line

    def sibling(self, cap: int | None = None) -> Client:
        """A client of its own for a request sent beside this one's (agent/n11/drafts.py, lane drafts1): the same route,
        transport, clocks, deadline and usage file, and the unit's one count (`usage`, every change under `LOCK`), so no
        request of either passes the cap; its own pacing floor (`model.Client._last_sent`), so the desk's floor spaces
        its sends alone; `cap` where given, never above this one's; the form, marks and kind this one's."""
        made = type(self)(self.config, responder=self._responder, usage=self.usage, usage_path=self.usage_path,
                          deadline=self.deadline, transport=self._transport, sleep=self._sleep, clock=self._clock,
                          max_output=self.max_output)
        made.cap = self.cap if cap is None else min(cap, self.cap)
        made.form, made.marks, made.kind = self.form, self.marks, self.kind
        return made

    def carry(self, siblings, since: float) -> float:
        """The siblings' deadlines moved past `since` (their deadline when made: the desk's untimed waits after a 429,
        `model.RATE_UNTIMED`) carried onto this one's by the largest move, as one wait, their waits having run at once;
        the seconds moved, 0 where none moved (the image: its deadline never moves)."""
        moved = max((sibling.deadline - since for sibling in siblings if sibling.deadline is not None), default=0.0)
        if self.deadline is None or moved <= 0:
            return 0.0
        with LOCK, model.LOCK:
            self.deadline += moved
            self._origin += moved
        return moved

    def sent(self) -> int:
        """The requests this thread has had counted."""
        return getattr(self._sent, "count", 0)

    def next_send(self) -> float:
        """When a request asked now would be sent: the desk's pacing's moment (`model.MIN_INTERVAL_S` after the last
        send, never before the first) where it is set, else now (agent/n11's last call by its send, lane clock1 of
        10-13)."""
        now = self._clock()
        return max(now, self._last_sent + model.MIN_INTERVAL_S) if model.MIN_INTERVAL_S > 0 and self._last_sent else now

    @property
    def last_sent(self) -> float | None:
        """The clock at the last attempt's send, None before any."""
        return self._last_sent or None

    def chat(self, system: str, user: str, **kwargs):
        if not trace.active():
            return super().chat(system, user, **kwargs)
        before, began = self.sent(), time.monotonic()
        try:
            reply = super().chat(system, user, **kwargs)
        except BaseException as exc:
            trace.request(system, user, None, exc, time.monotonic() - began, self.usage.requests,
                          self.sent() - before, kind=self.kind)
            raise
        trace.request(system, user, reply, None, time.monotonic() - began, self.usage.requests,
                      self.sent() - before, kind=self.kind)
        return reply

    def converse(self, messages: list, **kwargs):
        """model.Client's `converse`; under the trace, written there as `chat` writes a request: the system turn, and
        as its prompt the turns since the last assistant turn (each earlier one is in an earlier request's line), the
        reply with each call it made after its text."""
        if not trace.active():
            return super().converse(messages, **kwargs)
        system = next((str(m.get("content") or "") for m in messages if m.get("role") == "system"), "")
        last = max((k for k, m in enumerate(messages) if m.get("role") == "assistant"), default=0)
        user = "\n\n".join(f"[{m.get('role')}{' ' + m['tool_call_id'] if m.get('tool_call_id') else ''}]\n"
                           f"{m.get('content') or ''}" for m in messages[last + 1 if last else 1:])
        before, began = self.sent(), time.monotonic()
        try:
            reply = super().converse(messages, **kwargs)
        except BaseException as exc:
            trace.request(system, user, None, exc, time.monotonic() - began, self.usage.requests,
                          self.sent() - before, kind=self.kind)
            raise
        shown = reply.text + "".join(f"\n[call {call.name}] {call.raw}" for call in reply.tool_calls)
        trace.request(system, user, dataclasses.replace(reply, text=shown), None, time.monotonic() - began,
                      self.usage.requests, self.sent() - before, kind=self.kind)
        return reply


def client(card: Path, cap: int, out: Path) -> Client:
    """The unit's client from the environment (the mock under `model.MOCK_ENV`), capped at `cap` requests; a usage
    file named inside `out` is not written (the usage line never enters the output)."""
    made = Client.from_env(card=card)
    made.cap = cap
    if made.usage_path is not None and Path(out).resolve() in (made.usage_path.resolve(), *made.usage_path.resolve().parents):
        made.usage_path = None
    return made
