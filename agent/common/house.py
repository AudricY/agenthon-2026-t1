"""The model client (model.py) with one addition: the agent's own request cap, below the platform's.

Called by: agent/n11/solve.py (`client`, which every step's request goes through) and agent/n11/drafts.py
    (`Client.sibling`).
Start at: `client`.

The rules in detail
-------------------
The platform's model through the one client, model.py, under the agent's own request cap.

model.py sits beside this package as /opt/agent/model.py. `Client` is model.Client with one more refusal: a request
past the agent's `cap` (below the platform's 25) is refused unsent, attempts and retries counted alike. `LOCK` holds
the check against the cap and the count as one step, whichever thread asks. `sibling` makes a client of its own for a
request sent beside this one's (agent/n11's drafts): the unit's one count and cap, its own pacing floor; `carry` moves
this client's deadline past the untimed waits its siblings made.
"""
from __future__ import annotations

import threading
from pathlib import Path

import model

# One process runs one unit, so one lock for the unit's counts, whichever clients and threads the agent has
LOCK = threading.RLock()


class Client(model.Client):
    cap = model.REQUESTS_PER_UNIT

    def _admit(self) -> float:
        with LOCK:
            if self.usage.requests >= self.cap:
                raise model.RequestCapReached(f"request {self.usage.requests + 1} of the unit refused: the agent's "
                                              f"cap is {self.cap}")
            return super()._admit()

    def sibling(self, cap: int | None = None) -> Client:
        """A client of its own for a request sent beside this one's: the same route, deadline and count, so no
        request of either passes the cap; its own pacing floor (`model.Client._last_sent`); `cap` where given, never
        above this one's."""
        made = type(self)(self.config, deadline=self.deadline)
        made.cap = self.cap if cap is None else min(cap, self.cap)
        return made

    def carry(self, siblings, since: float) -> float:
        """The siblings' deadlines moved past `since` (their deadline when made: the untimed waits after a 429,
        `model.RATE_UNTIMED`) carried onto this one's by the largest move, as one wait, their waits having run at once;
        the seconds moved, 0 where none moved."""
        moved = max((sibling.deadline - since for sibling in siblings), default=0.0)
        if moved <= 0:
            return 0.0
        with LOCK, model.LOCK:
            self.deadline += moved
        return moved


def client(card: Path, cap: int) -> Client:
    """The unit's client from the environment, capped at `cap` requests."""
    made = Client.from_env(card=card)
    made.cap = cap
    return made
