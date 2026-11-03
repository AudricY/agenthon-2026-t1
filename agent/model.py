"""The one model client: every model request the agent makes goes through `Client`.

In the sandbox it reads `MODEL_ENDPOINT`, `MODEL_NAME` and `MODEL_TOKEN` (docs/challenge.md, "The model"). On the
desk, where `MODEL_ENDPOINT` is unset, it reads the model file (docs/keys.md): its path is the entry before `KEY_NAME`
in lab.toml's `[custody] token_names`, or `KEY_FILE_ENV` for a test; an omitted endpoint or model line takes the
OpenRouter defaults. The file's key goes only to the file's endpoint, never to one the environment names. A second
file, `model-<provider>.env`, is read where `PROVIDER_ENV` names its provider, and the record's model line
(`Config.label`) then ends with that word, so two keys to one host and model are counted apart.

The pool (lane pool1; the owner's in655, in656): under `PROVIDER_ENV=openrouter-pool` (`POOL`) the desk's client holds
both OpenRouter keys' model files (`POOLS`: the default file's key, called `FIRST` there, and the second file's), one
host and the free variant alone (`POOL-FREE`), and sends each attempt through one of them (`Client._pick`): of the keys
not spent, the one with the most of its day left, the desk's day count as the unit started (`PoolKey.left`) less the
attempts admitted through it since, or where the desk counted none the one with the fewest admitted; ties go to the
pool's order. A key that answers with OpenRouter's day's cap (`DAY_CAP_SAID`) is spent for the process and the attempt
goes at once through the next. With both free keys spent the request goes through the paid fallback (`PAID`, the
owner's in667: the first key's file and the model without `:free`, as the paid route sends it), and with that off too
(`POOL_PAID_ENV=0`) it is refused unsent (`PoolSpent`). The usage line counts each key's attempts (and dollars) under
`pool`, by its word, so the day and the dollars are counted per route, with the platform's own figure of the key's
day as its last answer's headers gave it (`day_remaining`, `day_reset`, `day_at`: OpenRouter counts refused attempts
toward a key's day, which the desk's count of admitted requests misses), which the rotation prefers; the record's line
names the pool and its keys
(`openrouter.ai <model>:free openrouter-pool first+openrouter2+paid`). The harness hands a unit's container the keys by
name (`Config.variables`, `POOL_KEYS_ENV`) and each free key's day left as the unit starts (`POOL_LEFT_ENV`); the image
sets neither, so there the client is as before.

The house limits hold here, in code: the seed pinned and the temperature pinned unless a sampling arm names its own,
at most `MAX_OUTPUT_TOKENS` asked of a call, and the process's request past `REQUESTS_PER_UNIT` refused before it is
sent (one process runs one unit). Every request sent counts, failures and retries included, as the house counts them.
The counts are atomic: one lock, the process's (`LOCK`), holds the check against the cap and the count as one step,
and every other change to the counts and every write of the usage file, so the threads and clients of a unit sending
at once (a fan-out) never send past the cap; a request asked while the last places are in flight is refused, not
queued.
The unit's clock holds here too: a client from `from_env` knows the unit's deadline (`unit_deadline`), gives each
attempt the smaller of `TIMEOUT_S` and the time left less `RESERVE_S`, abandons a reply not complete by then, and refuses a request with no time left as it refuses one past
the cap (`DeadlineReached`). A caller may bound one request's wall time as well (`bound=`, the seconds it allows
the request, its attempts and waits included; agent/n11's loop while its last resort's reserve is held, lane
fallback2): no attempt or wait runs past it, the attempt in flight there is abandoned (counted, as the House counts a
lost reply), and none is sent after it (`BoundReached`); unset, every timeout and wait is as before. Every body asks
thinking off in the house's own field, `chat_template_kwargs`
(`THINKING_OFF`; HOUSE-MODEL.md: omitting it keeps thinking on) unless the caller names low effort, which asks
thinking on with the house's `low_effort` (`THINKING_LOW`), or a thinking budget, which asks it on with the house's
`reasoning_budget`; the usage line names the mode asked (`thinking`) and counts the replies showing thinking the
body did not ask off (`thinking_shown`: the route honoured the mode). A route whose 400 or 422 names that field is asked
again without it, and never sent it again in this process. A caller may also ask a reasoning setting (`reasoning=`,
OpenRouter's `reasoning` object); a route that answers it with a 400 or 422 is asked again without it, and is never
sent it again in this process if that answer succeeds. A caller may offer tools (`tools=`, `tool_choice=`, the
OpenAI fields); the reply's `tool_calls` come back beside its text, each its name and its arguments read as JSON, and
a route whose 400 or 422 names a tool is asked again without them, and never sent them again in this process
(`tool_refused`); without tools the body is as before, byte for byte. A 410 is a route that no longer serves the
model: no retry, the usage line counts it (`gone`), and every later request of the process is refused unsent and
counted `refused` (`RouteGone`, lane ship1). A 407 is the proxy turning a request away before the House
admits it, which a burst of concurrent requests drew on the platform and the House did not charge (docs/challenge.md,
open fact 6): it is uncounted, sent once more after `PROXY_BACKOFF_S`, and a second 407 in the call ends the call; the
usage line counts each (`turned_away`). On the desk alone, `AGENTHON_MODEL_RATE_UNCHARGED=1` (`RATE_UNCHARGED`, beside
the pacing's `RATE_WAIT_S`) uncounts an attempt the route answered 429 as a 407 is uncounted, since the House has no
such limiter: the cap is then spent on admitted requests alone, and the usage line counts each (`rate_uncharged`, a
key it carries only where the flag is in force). The image never sets it, so there a 429 counts as every request.
The seconds the pacing waited after a 429 are on the usage line wherever the pacing is set, the switch below or not
(`rate_wait_s`, lane clock1: a key the image's line never carries, which agent/pi's proxy mirrors whole), and on the
desk alone `AGENTHON_MODEL_RATE_UNTIMED=1` (`RATE_UNTIMED`) takes them off the unit's clock, since the House has no
limiter to wait on: each wait is waited whole and the client's `deadline` moves past it (an agent reading the
client's deadline, agent/n11's, moves with it), and the usage line says so (`rate_untimed`, a key it carries only
where the flag is in force). Unset, every wait, clock reading and body is as before.
The floor (lane floor1; docs/research/clock-1014.md): on the desk `AGENTHON_MODEL_MIN_INTERVAL_S` (`MIN_INTERVAL_S`,
every desk route's 44 s since then, `agenthon_t1.routes.PACE`) stands in for the House's latency as the least gap
between two sends: an attempt waits until that long after the floor's clock (`_last_sent`), the send of the last
attempt the host did not answer 429. An attempt answered 429 never moves the floor's clock, so its retry goes as soon
as the 429's own wait (`RATE_WAIT_S` times the attempt, from the answer) has passed, with no floor wait after it; any
other attempt moves it at its send, as before; unset (the image), no send waits. Where the floor or rate wait is
positive, or RATE_UNTIMED is on (`paced`), each request's time is kept apart (`Timing`): `sent_s`, its last attempt's send in seconds on the unit's
clock as the client keeps it (from the client's making, the untimed waits off it as they are off the deadline);
`floor_wait_s`, the seconds it waited for the floor; `served_s`, the serving's own seconds from that send to the
reply's end (or the attempt's failure); and `rate_wait_s`, its 429s' waits. The usage line sums the unit's
`floor_wait_s` and `served_s` (keys the image's line never carries), and where the agent's trace is open
(agent/common/trace.py, `TRACE_MODULE`) the client writes each request's four as a `send` event (`SEND_STEP`) just
before the agent's `request` row, with that row's `ordinal`: the row's `latency_s` holds the floor's wait, the 429s'
and the serving's together. A route whose 400 or 422 names
`tool_choice` is asked again with the tools and without it, the tools kept (`tool_choice_refused`), and so is a 404
to a request carrying it whose text contains both "No endpoints found" and "Tool Compatibility" (`ROUTED_OUT`:
on the pinned `paid-bf16` its tool-compatibility filter removes the one serving, E155; lane toolchoice1). That 404
is remembered for this client's lifetime: later requests keep the tools and omit the choice, counted once;
a new client asks again. Any other 404 is a refused route, as before; the 400/422 retry stays per request.
The desk's model file may name a provider pin
(`FILE_PROVIDER_PIN`, a JSON object), sent as the body's `provider` (OpenRouter's provider routing); without it the
body has no such field. A pin that holds the request to one serving (`pin_word`: its `quantizations`, else the providers
its `only` or `order` names) ends the record's model line with `PINNED` and that word (`openrouter.ai <model> pinned
DeepInfra` against `... pinned fp8`; lane paidpin1, the owner's in745 and in747), so two pinned routes to one host and
model read apart; an unpinned line is as before. The serving that answered, the response's top-level `provider` where
the route names one (OpenRouter does), is the reply's (`Reply.provider`), and the usage line counts the replies by it
(`servings`, a key it carries only where a reply named one).
On the desk alone, `DESK_MAX_OUTPUT_ENV` naming an integer above `MAX_OUTPUT_TOKENS` raises the clamp and a call's
default ask to it (`desk_max_output`: the uncapped oracle, E009); the image never sets it, the mock ignores it, and
`describe()` and the usage line name it where it is in force.
With `MOCK_ENV=1` the client answers from a deterministic responder and opens no connection. Where the harness names a usage file (`USAGE_ENV`), the unit's
totals are rewritten there after every request, so a unit killed at its deadline still leaves its count.

The image carries this file as /app/model.py, so it imports the standard library alone and needs no lab.toml there.
The key is never printed, logged, written to the usage file, or carried in an exception's text.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

REQUESTS_PER_UNIT = 25
MAX_OUTPUT_TOKENS = 4000
TEMPERATURE = 0.0
SEED = 20260731                 # carried from agenthon-t4: its organisers' strong baseline's default (T4_SEED)
ATTEMPTS = 3                    # a transient failure is retried; each attempt is a request
# Desk-only pacing for a hosted route's limiter (the free route's 429s on the loop's short, fast requests, E034): unset in the
# image, so the platform's behaviour is the line above; the harness forwards them into the container when set (harness.PACING).
RATE_WAIT_S = float(os.environ.get("AGENTHON_MODEL_RATE_WAIT_S", "0") or 0)        # the wait after a 429, times the attempt
RATE_ATTEMPTS = int(os.environ.get("AGENTHON_MODEL_RATE_ATTEMPTS", "0") or 0)     # attempts while every failure is a 429
MIN_INTERVAL_S = float(os.environ.get("AGENTHON_MODEL_MIN_INTERVAL_S", "0") or 0)  # the least gap between two sends
RATE_UNCHARGED = os.environ.get("AGENTHON_MODEL_RATE_UNCHARGED", "") == "1"       # a 429 attempt uncounted, as a 407
RATE_UNTIMED = os.environ.get("AGENTHON_MODEL_RATE_UNTIMED", "") == "1"           # a 429's wait off the unit's clock
SEND_STEP = "send"              # the trace's event for a request's time on the desk (`Client._timed`)
TRACE_MODULE = "common.trace"   # the agent's trace (agent/common/trace.py), as agent/common/house.py loads it
TIMEOUT_S = 300.0               # one attempt's own bound: 4,000 tokens at the slowest rate measured, 16/s, is 250 s
UNIT_SECONDS = 1800.0           # the unit's clock where its card names no [agent] timeout_sec: 62 of 86 Track 1 cards say 1,800 s (docs/challenge.md)
# The platform's clock this process cannot see, kept from the card's: the container's creation, the image's pull and
# the provisioning, which the card's clock counts and g2 checks the elapsed time against (docs/challenge.md, "The
# runtime"). 200 s (lane ship1, on its cold review) holds a cold first pull to its measured top, 187 s, on the
# watchdog's path (187 + 390 = 577 s inside a 600 s card), and a warm one (about 15 s) with room for creation and
# provisioning. It binds only on a card shorter than the roster cap plus itself: under 800 s at the image's 600.
PLATFORM_ALLOWANCE_S = 200.0
# The roster cap: the Development stage clock is 43,200 s for 86 units in sequence, 502 s a unit, and it binds before
# the cards' 1,200 to 5,400 s (docs/challenge.md, "The runtime"). 86 x 400 s = 34,400 s leaves 8,800 s, 102 s a unit,
# for what runs on the stage's clock outside this process (the setup and provisioning after the House activates the
# unit, the container's creation and its cleanup grace, a warm pull of about 15 s and the first unit's cold one of 90 to
# 187 s); the checks run on the scoring stage's own clock. The overhead a unit actually costs is open until a run page
# shows it (docs/challenge.md, open fact 6); agenthon-t4's track1-handoff.md starts at the same 400 s. The Final
# stage's clock is 86,400 s (Codabench phase 29641; docs/challenge.md fact 8) over a number of units it does not
# publish. The organisers' per-unit limits are 600 to 2,400 s, 1,800 on most (t1#3, 10-10; SPINE 405), distinct from
# the kit cards' 1,200 to 5,400 s above: 1,005 s a unit at 86, so the same
# 102 s of overhead leaves about 900 s. The cap is a setting (lane finalclock1, `roster_cap`): the desk's
# `ROSTER_CAP_ENV`, else an image's word in `ROSTER_CAP_FILE` beside this file (agent/ROSTER_CAP, 600 s since lane
# ship1, ASK-10's default), else `ROSTER_CAP_S`; the card's clock less the allowance bounds it whatever it says
# (`unit_deadline`).
ROSTER_CAP_S = 400.0
ROSTER_CAP_ENV = "AGENTHON_MODEL_ROSTER_CAP_S"   # desk only: the harness forwards it into a container; never an image's
ROSTER_CAP_FILE = "ROSTER_CAP"     # one number beside this file, /opt/agent/ROSTER_CAP (agent/ROSTER_CAP): an image's
RESERVE_S = 30.0                # kept from the deadline: the answer's last write
MIN_CALL_S = 1.0                # less time left than this is no time left
PROXY_BACKOFF_S = 2.0           # a request the proxy turned away (HTTP 407) is sent once more after this
REASONING_FIELD_HOSTS = ("openrouter.ai",)   # desk routes that take thinking off only through their own `reasoning` field
REASONING_FIELD = os.environ.get("AGENTHON_MODEL_REASONING_FIELD", "") == "1"   # desk only: send that field there; unset in the image
TUNNEL_407 = "Tunnel connection failed: 407"   # http.client's error where the proxy refuses an https CONNECT
THINKING_OFF = {"enable_thinking": False}   # the house's switch, top level of the body (the toolkit's docs/HOUSE-MODEL.md)
THINKING_LOW = {"enable_thinking": True, "low_effort": True}   # HOUSE-MODEL.md: passed through, its effect untested

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

KEY_NAME = "AGENTHON_MODEL_API_KEY"
FILE_ENDPOINT = "AGENTHON_MODEL_ENDPOINT"
FILE_MODEL = "AGENTHON_MODEL_NAME"
FILE_PROVIDER_PIN = "AGENTHON_MODEL_PROVIDER_PIN"   # optional: `{"quantizations": ["fp8"]}`, the house's snapshot
KEY_FILE_ENV = "AGENTHON_MODEL_FILE"
PROVIDER_ENV = "AGENTHON_MODEL_PROVIDER"   # names a second model file, model-<provider>.env in custody, in place of model.env
POOL = "openrouter-pool"                   # the provider word of the pool of the owner's two OpenRouter keys (lane pool1)
FIRST = "first"                            # a pool's word for the default model file's key, whose line carries no word
POOLS = {POOL: (FIRST, "openrouter2")}     # each pool's keys in order: FIRST, else the provider of model-<word>.env
PAID = "paid"                              # a pool's word for its paid fallback: the first key, the model unfree
POOL_PAID_ENV = "AGENTHON_MODEL_POOL_PAID"   # desk only: "0" keeps a pool on its free keys (no paid fallback)
POOL_KEYS_ENV = "AGENTHON_MODEL_POOL_KEYS"   # desk only: a pool's keys, handed to a unit's container by name alone
POOL_LEFT_ENV = "AGENTHON_MODEL_POOL_LEFT"   # desk only: each pool key's free requests left as the unit starts
DAY_CAP_SAID = "free-models-per-day"         # OpenRouter's 429 once a key's free requests of the day are spent
# the headers a key's day reads from (10-12 16:13Z, the day's-cap 429: limit 1000, remaining 0, reset at 00:00Z in epoch
# ms); a limit under DAY_LIMIT_FLOOR is a per-minute window's, not the day's
DAY_HEADERS = ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset")
DAY_LIMIT_FLOOR = 1000
PIN_ENV = "AGENTHON_MODEL_PROVIDER_PIN"   # desk only: the body's `provider` object as JSON, over the model file's line; the harness forwards it
PINNED = "pinned"                          # a model line's word before the pin's (`pin_word`), where the pin holds
MOCK_ENV = "AGENTHON_MODEL_MOCK"
DESK_MAX_OUTPUT_ENV = "AGENTHON_MODEL_DESK_MAX_OUTPUT"   # the desk's output clamp above the House's: never the image's
USAGE_ENV = "AGENTHON_MODEL_USAGE"
DEADLINE_ENV = "AGENTHON_UNIT_DEADLINE"    # the unit's end in epoch seconds, where the harness names it
ALLOWANCE_ENV = "AGENTHON_PLATFORM_ALLOWANCE"   # a test's allowance in seconds, in place of PLATFORM_ALLOWANCE_S
MOCK = "mock"
MOCK_REPLY = '{"mock": true}'

# The house's 403 once a unit's allowance is used up (the organisers' strong baseline, client.py).
_BUDGET_CODE, _BUDGET_MESSAGE = "grant_denied", "request admission refused"
# OpenRouter's 404 when its routing filters out every endpoint, as for a forced `tool_choice` on a route pinned to a
# serving that does not take it (E155: "No endpoints found for <model>. ... Filter by Tool Compatibility removed ...")
ROUTED_OUT = ("No endpoints found", "Tool Compatibility")


class ModelError(RuntimeError):
    """A model call that produced no reply."""


class ModelConfigError(ModelError):
    """No usable endpoint, model or key, or the endpoint refused the request as misconfigured."""


class ModelCallError(ModelError):
    """The call failed after its attempts (network, timeout, server error, malformed reply)."""


class RouteGone(ModelConfigError):
    """The client refused a request unsent: the route answered 410 (`gone`) earlier in this process."""


class BudgetExhausted(ModelError):
    """The unit's request allowance is used up; no further request is sent."""


class RequestCapReached(BudgetExhausted):
    """The client refused a request past `REQUESTS_PER_UNIT` before sending it."""


class DeadlineReached(BudgetExhausted):
    """The client refused a request the unit's clock leaves no time for, before sending it."""


class PoolSpent(BudgetExhausted):
    """Every key of the pool has met its day (it answered with the day's cap, or the desk's count leaves it none); the
    request is refused unsent."""


class BoundReached(ModelCallError):
    """The caller's bound on one request (`bound=`) ran out before a reply: the attempt in flight abandoned at it, and
    none sent after it."""


class _Secret:
    """A credential that shows as redacted wherever it is printed."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__


@dataclass(frozen=True)
class PoolKey:
    """One key of a pool: its word (`FIRST`, or its model file's provider; `PAID` the fallback), its credential, its
    free requests left of the day as the desk counted them when the unit started (None: not counted), the model its
    requests name where not the config's, and whether it is asked only once every other key is spent."""
    word: str
    token: _Secret
    left: int | None = None
    model: str | None = None
    fallback: bool = False


@dataclass(frozen=True)
class Config:
    endpoint: str               # the base URL: the house route's origin, or a provider's URL ending /v1
    model: str
    token: _Secret | None       # a pool's first key's
    source: str                 # "sandbox" or "desk"
    provider_pin: dict | None = field(default=None, hash=False)   # the body's `provider`, from the model file
    provider: str = ""          # the model file's provider where `PROVIDER_ENV` chose it; "" for the default file
    pool: tuple[PoolKey, ...] = ()   # a pool's keys (`POOLS`), each attempt sent through one (`Client._pick`)

    @property
    def host(self) -> str:
        return urlsplit(self.endpoint).hostname or ""

    @property
    def url(self) -> str:
        base = self.endpoint.rstrip("/")
        # a base that ends /v1 (OpenRouter, NVIDIA) or carries /v1/ inside it (DeepInfra's /v1/openai) is complete;
        # the house route's bare origin gets /v1
        if not (base.endswith("/v1") or "/v1/" in base):
            base += "/v1"
        return base + "/chat/completions"

    @property
    def label(self) -> str:
        """The gate record's `model`: the endpoint's host and the model's name, then the model file's provider where
        one was chosen, so two keys to one host and model read apart (`openrouter.ai <model> openrouter2`); a pool's
        word is followed by its keys' (`openrouter.ai <model> openrouter-pool first+openrouter2`); last, where the pin
        holds the request to a serving, `PINNED` and its word (`pin_words`: `openrouter.ai <model> pinned DeepInfra`)."""
        keys = "+".join(key.word for key in self.pool) if self.provider in POOLS else ""
        return " ".join(word for word in (self.host, self.model, self.provider, keys,
                                          *pin_words(pin_word(self.provider_pin))) if word)

    def variables(self) -> dict[str, str]:
        """What a unit's container is handed for this route, each by name (the harness's `--live`): the endpoint, the
        model and the key, and a pool's keys beside its first's (`POOL_KEYS_ENV`). The values are credentials: never
        printed, logged or put in an argument."""
        out = {"MODEL_ENDPOINT": self.endpoint, "MODEL_NAME": self.model,
               "MODEL_TOKEN": self.token.reveal() if self.token is not None else ""}
        if self.pool:
            out[POOL_KEYS_ENV] = json.dumps({key.word: key.token.reveal() for key in self.pool})
        return out


@dataclass
class Usage:
    """One unit's counts: requests sent (every attempt), requests refused at the cap, tokens the host reported; each
    changed under `LOCK`."""

    requests: int = 0
    refused: int = 0
    late: int = 0               # requests refused for want of time on the unit's clock
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0   # of the completion tokens, those the host reports as reasoning
    truncated: int = 0          # replies the output cap cut short (finish_reason "length")
    rate_limited: int = 0       # attempts the host answered 429: the hosted route's limit, never the house's
    gone: int = 0               # attempts the host answered 410: the route no longer serves the model
    turned_away: int = 0        # attempts the proxy answered 407 before admission: never charged, never in `requests`
    rate_uncharged: int = 0     # attempts answered 429 and uncounted under the desk's RATE_UNCHARGED: not in `requests`
    rate_wait_s: float = 0.0    # the seconds the pacing waited after a 429 (RATE_WAIT_S); off the clock under RATE_UNTIMED
    floor_wait_s: float = 0.0   # the seconds the sends waited for the floor (MIN_INTERVAL_S), on the line where `paced`
    served_s: float = 0.0       # the serving's own seconds, each request's last attempt's, on the line where `paced`
    tool_choice_refused: int = 0    # requests a route refused for `tool_choice`, asked again without it, the tools kept
    servings: dict[str, int] = field(default_factory=dict)   # the replies by the serving that answered (`Reply.provider`)
    cost_usd: float | None = None
    exhausted: bool = False     # the house said the allowance is used up: send nothing more
    reasoning_refused: bool = False   # the route refused the reasoning setting: it is not sent again
    template_refused: bool = False    # the route refused `chat_template_kwargs`: it is not sent again
    thinking_leaked: int = 0    # replies showing thinking although the body asked it off: reasoning tokens, `_thought`
    finish_reasons: dict[str, int] = field(default_factory=dict)   # each reply's finish_reason, counted
    tool_calls: int = 0         # the tool calls the replies carried
    tool_refused: bool = False  # the route refused the tools: they are not sent again
    thinking: str | None = None  # the mode the bodies asked: "off", "low" or "budget"; "mixed" where they differed
    thinking_shown: int = 0     # replies showing thinking the body did not ask off: reasoning tokens, `_thought`
    pool: dict[str, dict] = field(default_factory=dict)   # a pool's attempts by key word (`Client._tally`)
    pool_spent: int = 0         # requests refused unsent with every key of the pool spent (`PoolSpent`)
    day: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """One call a reply made: the tool's name, its arguments read as JSON (None where they are not a JSON object),
    and the arguments as the route wrote them."""
    name: str
    arguments: dict | None
    raw: str


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    reasoning_tokens: int = 0
    tool_calls: tuple[ToolCall, ...] = ()
    message: Mapping | None = field(default=None, compare=False, repr=False)   # the assistant message as the route
    #                                                                            returned it (None under the mock)
    provider: str = ""          # the serving that answered: the response's top-level `provider`; "" where none is named


@dataclass
class Timing:
    """One request's time on the desk (lane floor1, the module docstring): its last attempt's send on the unit's clock
    (None while none is sent), the seconds it waited for the floor and after its 429s, and that attempt's serving."""
    sent_s: float | None = None
    floor_wait_s: float = 0.0
    served_s: float = 0.0
    rate_wait_s: float = 0.0


def paced() -> bool:
    """A positive floor or rate wait, or untimed waits, enables timing; none is set in the image."""
    return MIN_INTERVAL_S > 0 or RATE_WAIT_S > 0 or RATE_UNTIMED


def _traced(step: str, **payload) -> None:
    """An event in the agent's trace (`TRACE_MODULE`'s `event`, which writes nothing while the trace is off, as on the
    platform), where the agent has loaded it; never an error: the client's behaviour never depends on the trace."""
    found = sys.modules.get(TRACE_MODULE)
    if found is None:
        return
    try:
        found.event(step, **payload)
    except Exception:  # noqa: BLE001 - the trace is the agent's; a failure there is never the request's
        pass


# One process runs one unit, so the unit's count is the process's, whichever clients the agent builds, and so is the
# lock over the counts: every change to a usage and every write of a usage file holds it (`Client._admit`).
_UNIT = Usage()
LOCK = threading.RLock()


def key_file_path(lab_toml: Path | None = None, provider: str | None = None) -> Path:
    """The model file: `KEY_FILE_ENV`, else the `[custody] token_names` entry just before `KEY_NAME`; with `PROVIDER_ENV`
    set, the entry named model-<provider>.env among those (a second route to the same house model, the owner's NVIDIA key
    beside the OpenRouter one), so no shell command need name a key file to choose the route. `provider`, a pool's
    key's (`pooled`), stands in for `PROVIDER_ENV`'s ("" the default file); under `KEY_FILE_ENV`, a test's stand-in, a
    pool's key with a provider reads the stand-in's sibling model-<provider>.env."""
    member = provider is not None
    provider = os.environ.get(PROVIDER_ENV, "").strip() if provider is None else provider
    if override := os.environ.get(KEY_FILE_ENV):
        path = Path(override).expanduser()
        return path.with_name(f"model-{provider}.env") if member and provider else path
    here = Path(__file__).resolve()
    lab_toml = lab_toml or (here.parents[2] / "lab.toml" if len(here.parents) > 2 else None)   # /app/model.py: none
    if lab_toml is None or not lab_toml.is_file():
        raise ModelConfigError("no model configured: MODEL_ENDPOINT is unset and there is no lab.toml to name "
                               "the desk's model file")
    names = tomllib.loads(lab_toml.read_text(encoding="utf-8")).get("custody", {}).get("token_names", [])
    paths = [Path(str(names[i - 1])).expanduser() for i, name in enumerate(names)
             if name == KEY_NAME and i >= 1 and str(names[i - 1]).startswith(("/", "~"))]
    if not paths:
        raise ModelConfigError("lab.toml's [custody] token_names does not pair the model key with its file")
    if not provider:
        return paths[0]
    wanted = f"model-{provider}.env"
    found = [path for path in paths if path.name == wanted]
    if not found:
        raise ModelConfigError(f"{PROVIDER_ENV}={provider!r} names no model file in custody: none is called {wanted}")
    return found[0]


def read_key_file(path: Path) -> dict[str, str]:
    """`KEY=VALUE` lines (docs/keys.md); a malformed line is named by its number, never by its text."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelConfigError(f"the model file cannot be read ({type(exc).__name__})") from None
    out: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if not sep or not name.strip():
            raise ModelConfigError(f"line {number} of the model file is not KEY=VALUE")
        out[name.strip()] = value.strip()
    return out


def resolve(environ: Mapping[str, str] | None = None) -> Config:
    """The sandbox's variables where `MODEL_ENDPOINT` is set (the file unread, so its key never reaches an endpoint
    the environment names), else the desk's model file, `MODEL_NAME` and `MODEL_TOKEN` winning over its lines."""
    env = os.environ if environ is None else environ
    if env.get("MODEL_ENDPOINT"):
        token = env.get("MODEL_TOKEN")
        return Config(env["MODEL_ENDPOINT"], env.get("MODEL_NAME", ""), _Secret(token) if token else None, "sandbox",
                      provider_pin(env.get(PIN_ENV)), pool=handed_pool(env))
    if (provider := os.environ.get(PROVIDER_ENV, "").strip()) in POOLS:
        return pooled(provider, env)
    path = key_file_path()
    lines = read_key_file(path)
    key = env.get("MODEL_TOKEN") or lines.get(KEY_NAME)
    if not key:
        raise ModelConfigError("the model file has no API key line")
    return Config(lines.get(FILE_ENDPOINT) or DEFAULT_ENDPOINT,
                  env.get("MODEL_NAME") or lines.get(FILE_MODEL) or DEFAULT_MODEL, _Secret(key), "desk",
                  provider_pin(env.get(PIN_ENV) or lines.get(FILE_PROVIDER_PIN)), file_provider(path))


def pool_provider(word: str) -> str:
    """The provider whose model file a pool's key word names: "" (the default file) for `FIRST`, else the word."""
    return "" if word == FIRST else word


def pool_paid(environ: Mapping[str, str]) -> bool:
    """Whether a pool falls back to the paid route once its free keys are spent: unless `POOL_PAID_ENV` says "0"."""
    return environ.get(POOL_PAID_ENV, "").strip() != "0"


def paid_key(token: _Secret, free_model: str) -> PoolKey:
    """A pool's paid fallback (`PAID`, the owner's in667): the first key, the free variant's model without `:free`, as
    the paid route sends it, asked only once the free keys are spent."""
    return PoolKey(PAID, token, None, free_model.removesuffix(":free"), True)


def pooled(provider: str, env: Mapping[str, str]) -> Config:
    """The pool `provider` names (`POOLS`): each key's model file read as `resolve` reads one (`key_file_path` by its
    provider), the endpoint they share, the model `MODEL_NAME`'s else the first file's, which must be the free
    variant (`:free`; POOL-FREE: the pool's keys are the free tier's, the paid route its fallback alone), the pin the
    environment's else the first file's, each key's day left where the desk counted it (`POOL_LEFT_ENV`), and the paid
    fallback after them unless `POOL_PAID_ENV` turns it off (`pool_paid`)."""
    left = pool_left(env.get(POOL_LEFT_ENV))
    keys, endpoints, first = [], set(), None
    for word in POOLS[provider]:
        lines = read_key_file(key_file_path(provider=pool_provider(word)))
        if not lines.get(KEY_NAME):
            raise ModelConfigError(f"the pool's key {word}: its model file has no API key line")
        first = lines if first is None else first
        endpoints.add(lines.get(FILE_ENDPOINT) or DEFAULT_ENDPOINT)
        keys.append(PoolKey(word, _Secret(lines[KEY_NAME]), left.get(word)))
    if len(endpoints) > 1:
        raise ModelConfigError(f"the pool {provider}'s model files name {len(endpoints)} endpoints: a pool is one host's")
    name = env.get("MODEL_NAME") or first.get(FILE_MODEL) or DEFAULT_MODEL
    if not name.endswith(":free"):
        raise ModelConfigError(f"POOL-FREE: the pool {provider} runs the free variant alone, and the model is {name}; "
                               f"remedy: MODEL_NAME=<model>:free (the route free-variant-pool sets it), or a single "
                               f"key's route for a paid model")
    if pool_paid(env):
        keys.append(paid_key(keys[0].token, name))
    return Config(endpoints.pop(), name, keys[0].token, "desk",
                  provider_pin(env.get(PIN_ENV) or first.get(FILE_PROVIDER_PIN)), provider, tuple(keys))


def handed_pool(env: Mapping[str, str]) -> tuple[PoolKey, ...]:
    """A pool's keys as the desk handed them to a unit's container (`Config.variables`), each with its day left
    (`POOL_LEFT_ENV`); () where none was handed, as in the image. A malformed variable is named, never its text."""
    said = env.get(POOL_KEYS_ENV)
    if not said:
        return ()
    try:
        keys = json.loads(said)
    except ValueError:
        keys = None
    if not isinstance(keys, dict) or not keys or not all(
            isinstance(word, str) and word and isinstance(token, str) and token for word, token in keys.items()):
        raise ModelConfigError(f"{POOL_KEYS_ENV} is not a JSON object of a pool's keys by word")
    left = pool_left(env.get(POOL_LEFT_ENV))
    return tuple(paid_key(_Secret(token), env.get("MODEL_NAME", "")) if word == PAID else
                 PoolKey(word, _Secret(token), left.get(word)) for word, token in keys.items())


def pool_left(text: str | None) -> dict[str, int]:
    """`POOL_LEFT_ENV` read: each pool key's free requests left by its word, the keys the desk counted none for absent
    or null; {} where it is unset."""
    if not text:
        return {}
    try:
        said = json.loads(text)
    except ValueError:
        said = None
    if not isinstance(said, dict) or not all(
            isinstance(word, str) and (left is None or (isinstance(left, int) and not isinstance(left, bool)))
            for word, left in said.items()):
        raise ModelConfigError(f"{POOL_LEFT_ENV} is not a JSON object of whole requests left by a pool key's word")
    return {word: left for word, left in said.items() if left is not None}


def left_said(left: Mapping[str, int | None]) -> str:
    """Each pool key's requests left as `POOL_LEFT_ENV` carries them."""
    return json.dumps(dict(left), sort_keys=False)


def file_provider(path: Path) -> str:
    """The provider a model file is named for, `model-<provider>.env`, where `PROVIDER_ENV` chose a file; "" where it
    is unset (the default file) or the file is named otherwise (a test's stand-in)."""
    name = path.name
    if not os.environ.get(PROVIDER_ENV, "").strip() or not (name.startswith("model-") and name.endswith(".env")):
        return ""
    return name[len("model-"):-len(".env")]


def provider_pin(text: str | None) -> dict | None:
    """The model file's pin line as the body's `provider` object; None where the line is absent or empty."""
    if not text:
        return None
    try:
        pin = json.loads(text)
    except ValueError:
        pin = None
    if not isinstance(pin, dict) or not pin:
        raise ModelConfigError(f"the model file's {FILE_PROVIDER_PIN} line is not a non-empty JSON object")
    return pin


def pin_word(pin: Mapping | None) -> str | None:
    """What a provider pin holds the request to, as a model line names it after `PINNED`: its quantizations where it
    names any (`fp8`), else the providers its `only` or `order` names (`DeepInfra`), a list's names joined by "+" and
    a name's own spaces by "-", so the word is one; "" for a pin that turns fallbacks off alone; None for no pin or one
    that holds nothing (`{"allow_fallbacks": true}`)."""
    if not isinstance(pin, Mapping):
        return None
    for name in ("quantizations", "only", "order"):
        said = pin.get(name)
        names = [said] if isinstance(said, str) else said if isinstance(said, (list, tuple)) else []
        if words := ["-".join(str(word).split()) for word in names if str(word).strip()]:
            return "+".join(words)
    return "" if pin.get("allow_fallbacks") is False else None


def pin_words(word: str | None) -> tuple[str, ...]:
    """The words a model line carries for a pin's `pin_word`: `PINNED` and the word, `PINNED` alone for "", none for
    None."""
    return () if word is None else (PINNED, word) if word else (PINNED,)


def describe() -> str:
    """What a run's gate record names as its model: `MOCK` under the mock, else the host and the model, and
    ` desk-max-output=N` where the desk's clamp is in force."""
    if os.environ.get(MOCK_ENV) == "1":
        return MOCK
    desk = desk_max_output()
    return resolve().label + (f" desk-max-output={desk}" if desk else "")


def desk_max_output(environ: Mapping[str, str] | None = None) -> int | None:
    """The desk's output clamp: `DESK_MAX_OUTPUT_ENV` where it names an integer above `MAX_OUTPUT_TOKENS`, else None
    (the House's cap holds)."""
    env = os.environ if environ is None else environ
    try:
        asked = int(env.get(DESK_MAX_OUTPUT_ENV, ""))
    except ValueError:
        return None
    return asked if asked > MAX_OUTPUT_TOKENS else None


def unit_deadline(card: Path | None = None, environ: Mapping[str, str] | None = None, *,
                  allowance: bool = True) -> float:
    """The unit's end in epoch seconds: `DEADLINE_ENV` where the harness names it (its own clock, started just before
    the agent), else this process's start plus the smaller of the card's `[agent] timeout_sec` (`UNIT_SECONDS`
    without one) less `PLATFORM_ALLOWANCE_S`, and `roster_cap`'s setting. The platform's clock starts before the process
    does: it counts the container's creation and the image's pull (DEVELOPMENT-RUNTIME.md, "Execution clocks"), and
    under the House timing release the provisioning after the unit's activation, none of which the process can see.
    `ALLOWANCE_ENV` sets the allowance for a test; `allowance=False` gives the card's whole clock from the process's
    start, the end the allowance is taken from, which the roster cap does not shorten. The roster cap is `roster_cap`'s
    under `environ`."""
    env = os.environ if environ is None else environ
    if env.get(DEADLINE_ENV):
        return float(env[DEADLINE_ENV])
    seconds = UNIT_SECONDS
    try:
        seconds = float(tomllib.loads(card.read_text(encoding="utf-8"))["agent"]["timeout_sec"])
    except (AttributeError, OSError, KeyError, TypeError, ValueError):
        pass
    taken = PLATFORM_ALLOWANCE_S
    try:
        asked = float(env[ALLOWANCE_ENV])
        taken = asked if math.isfinite(asked) and asked >= 0 else taken
    except (KeyError, TypeError, ValueError):
        pass
    if not allowance:
        return time.time() - process_age() + seconds
    return time.time() - process_age() + min(seconds - taken, roster_cap(env)[0])


def roster_cap(environ: Mapping[str, str] | None = None, here: Path | None = None) -> tuple[float, str]:
    """The roster cap in seconds and where it was read: `ROSTER_CAP_ENV` where it is set (the desk's), else the one
    number of `ROSTER_CAP_FILE` under `here` (beside this file: an image's), else `ROSTER_CAP_S`. A value that is no
    positive finite number of seconds runs `ROSTER_CAP_S`, the source saying why; the desk refuses it before any
    container starts (harness `clocks`), and the image's build refuses its word (agent/Dockerfile)."""
    env = os.environ if environ is None else environ
    said, source = (env.get(ROSTER_CAP_ENV) or "").strip(), ROSTER_CAP_ENV
    if not said:
        try:
            said = ((Path(__file__).resolve().parent if here is None else here) / ROSTER_CAP_FILE).read_text(
                encoding="utf-8").strip()
            source = ROSTER_CAP_FILE
        except (OSError, ValueError):
            return ROSTER_CAP_S, "no roster cap named"
        if not said:
            return ROSTER_CAP_S, "no roster cap named"
    seconds = roster_seconds(said)
    if seconds is None:
        return ROSTER_CAP_S, f"{source} names {said[:40]!r}, no seconds: refused"
    return seconds, source


def roster_seconds(said: str) -> float | None:
    """A roster cap as said, in seconds: a positive finite number, else None."""
    try:
        seconds = float(said)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def process_age() -> float:
    """Seconds since this process started (Linux's /proc), or 0 where that cannot be read."""
    try:
        ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return max(0.0, uptime - ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return 0.0


Responder = Callable[[str, str], str]
# the status and the body, and the reply's headers where the transport gives them (`day_of`)
Transport = Callable[[str, dict[str, str], bytes, float], tuple]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes, dict]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        heard = dict(exc.headers.items()) if exc.headers is not None else {}
        try:
            return exc.code, exc.read(), heard
        except (OSError, http.client.HTTPException):
            return exc.code, b"", heard


def day_of(heard: Mapping[str, str]) -> dict[str, int] | None:
    """A key's day as an answer's headers give it (`DAY_HEADERS`): `day_remaining`, `day_limit` and `day_reset` (epoch
    ms), None where they say none or a window shorter than the day's (`DAY_LIMIT_FLOOR`)."""
    said = {name.lower(): value for name, value in heard.items()}
    try:
        limit, remaining = int(said[DAY_HEADERS[0]]), int(said[DAY_HEADERS[1]])
    except (KeyError, TypeError, ValueError):
        return None
    if limit < DAY_LIMIT_FLOOR:
        return None
    try:
        reset = int(said.get(DAY_HEADERS[2], ""))
    except (TypeError, ValueError):
        reset = None
    return {"day_remaining": max(0, remaining), "day_limit": limit, **({"day_reset": reset} if reset else {})}


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def thinking_template(thinking: int | str | None) -> tuple[str, dict]:
    """The mode the usage line names and the body's `chat_template_kwargs`: None, "off" (`THINKING_OFF`); "low",
    thinking on at low effort (`THINKING_LOW`); a budget B of at least 1, "budget", thinking on with `reasoning_budget`
    B."""
    if thinking is None:
        return "off", dict(THINKING_OFF)
    if thinking == "low":
        return "low", dict(THINKING_LOW)
    if isinstance(thinking, (bool, str)) or int(thinking) < 1:
        raise ValueError('thinking is None, "low" or a budget of at least 1')
    return "budget", {"enable_thinking": True, "reasoning_budget": int(thinking)}


class Client:
    """`chat` and `complete`, under the house limits; `Client.from_env()` is what an agent builds."""

    def __init__(self, config: Config | None = None, *, responder: Responder | None = None,
                 usage: Usage | None = None, usage_path: Path | None = None, deadline: float | None = None,
                 transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time, max_output: int = MAX_OUTPUT_TOKENS) -> None:
        """`deadline`: the unit's end in epoch seconds; None for a caller that runs no unit (a probe). `max_output`:
        the clamp on a call's output tokens and its default ask, above `MAX_OUTPUT_TOKENS` on the desk alone."""
        if config is None and responder is None:
            raise ModelConfigError("a client needs a config or, under the mock, a responder")
        self.config = config
        self._responder = responder
        self.usage = _UNIT if usage is None else usage
        self.usage_path = usage_path
        self.deadline = deadline
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self._clock = clock
        self.max_output = max_output
        self._last_sent = 0.0        # the floor's clock (MIN_INTERVAL_S): the last send the host did not answer 429
        self._origin = clock()       # the unit's clock's start as the client keeps it, moved past each untimed wait
        self._choice_routed_out = False

    @classmethod
    def from_env(cls, responder: Responder | None = None, card: Path | None = None) -> Client:
        """The mock where `MOCK_ENV=1` (answering through `responder`), else the resolved endpoint; either way under
        the unit's deadline (`unit_deadline`, from the unit's `card` where the agent names it)."""
        usage_path = Path(os.environ[USAGE_ENV]) if os.environ.get(USAGE_ENV) else None
        deadline = unit_deadline(card)
        if os.environ.get(MOCK_ENV) == "1":
            return cls(responder=responder or (lambda system, user: MOCK_REPLY), usage_path=usage_path,
                       deadline=deadline)
        return cls(resolve(), usage_path=usage_path, deadline=deadline,
                   max_output=desk_max_output() or MAX_OUTPUT_TOKENS)

    @property
    def mock(self) -> bool:
        return self.config is None

    @property
    def label(self) -> str:
        return MOCK if self.config is None else self.config.label

    def __repr__(self) -> str:
        return f"Client({self.label}, requests={self.usage.requests})"

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        return self.chat(system, user, max_tokens=max_tokens).text

    def chat(self, system: str, user: str, *, max_tokens: int | None = None,
             reasoning: Mapping | None = None, thinking: int | str | None = None, temperature: float | None = None,
             top_p: float | None = None, tools: list | None = None, tool_choice=None, prior: Mapping | None = None,
             follow: str = "", bound: float | None = None) -> Reply:
        """`reasoning`: the request's `reasoning` object (`{"enabled": False}`, `{"effort": "low"}`); None sends
        none, the route's default. `thinking` (`thinking_template`): None asks thinking off (`THINKING_OFF`); "low"
        asks it on at the house's low effort (`THINKING_LOW`); a budget B asks it on with the house's server-side
        `reasoning_budget` B. `temperature` and `top_p`: a sampling arm's own (the model
        card's 1.0 and 0.95); None keeps `TEMPERATURE` and sends no top_p. `tools` and `tool_choice`: the OpenAI
        fields, sent as given (None sends neither); the reply's `tool_calls` carry what the model called. The mock
        answers in text alone. `max_tokens`: None asks the clamp, `max_output`; more is clamped to it. `prior` and
        `follow` (a multi-turn continuation, lane plan2): the system and user turns, then `prior`, an assistant message
        replayed exactly as a route returned it (`Reply.message`), then the user turn `follow`; a route whose 400 or
        422 names one of `prior`'s fields other than its role and content is asked once more with its text alone
        (`replayed`). Unset, the body is as before, byte for byte; the mock answers `follow`. `bound`: the seconds the
        caller allows this request (above; the mock, which takes none, ignores it); None, the unit's clock alone."""
        max_tokens = self.max_output if max_tokens is None else min(int(max_tokens), self.max_output)
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        mode, template = thinking_template(thinking)
        if self.config is None:
            self._admit()
            text = self._responder(system, user if prior is None else follow)
            reply = Reply(text, _estimate_tokens(system + user), _estimate_tokens(text), "stop")
            with LOCK:
                self.usage.finish_reasons["stop"] = self.usage.finish_reasons.get("stop", 0) + 1
            self._count(reply.prompt_tokens, reply.completion_tokens, None)
            return reply
        self._asked(mode)
        sampling = {"temperature": TEMPERATURE if temperature is None else float(temperature),
                    **({} if top_p is None else {"top_p": float(top_p)})}
        if prior is None:
            return self._post(system, user, max_tokens, reasoning, template, sampling, tools, tool_choice, bound=bound)
        turns = [{"role": "system", "content": system}, {"role": "user", "content": user}, dict(prior),
                 {"role": "user", "content": follow}]
        return self._post(system, user, max_tokens, reasoning, template, sampling, tools, tool_choice,
                          messages=turns, replayed=2, bound=bound)

    def converse(self, messages: list, *, max_tokens: int | None = None, thinking: int | str | None = None,
                 temperature: float | None = None, tools: list | None = None, tool_choice=None,
                 bound: float | None = None) -> Reply:
        """A whole conversation, OpenAI's `messages` list (a system turn, then user, assistant and `tool` turns, an
        assistant turn's `tool_calls` and a tool turn's `tool_call_id` as OpenAI writes them), sent where `chat` sends
        its one system and one user turn, under the same limits: counted at admission, the seed, `max_tokens`
        clamped, thinking off unless asked, a 407 retried uncharged and a 429 counted, as every request. The reply's
        `tool_calls` come back beside its text, and `bound` is `chat`'s. The mock answers the system turn and the last
        user or tool turn's text through its responder."""
        max_tokens = self.max_output if max_tokens is None else min(int(max_tokens), self.max_output)
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        mode, template = thinking_template(thinking)
        if self.config is None:
            system = next((str(m.get("content") or "") for m in messages if m.get("role") == "system"), "")
            user = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") in ("user", "tool")),
                        "")
            self._admit()
            text = self._responder(system, user)
            reply = Reply(text, _estimate_tokens(system + user), _estimate_tokens(text), "stop")
            with LOCK:
                self.usage.finish_reasons["stop"] = self.usage.finish_reasons.get("stop", 0) + 1
            self._count(reply.prompt_tokens, reply.completion_tokens, None)
            return reply
        self._asked(mode)
        sampling = {"temperature": TEMPERATURE if temperature is None else float(temperature)}
        return self._post("", "", max_tokens, None, template, sampling, tools, tool_choice, messages=messages,
                          bound=bound)

    def usage_line(self) -> dict:
        """The unit's line for the usage file: counts and the endpoint's host, never a credential."""
        u = self.usage
        with LOCK:     # a snapshot: no count moves while it is read
            return {"requests": u.requests, "refused": u.refused, "late": u.late, "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens, "reasoning_tokens": u.reasoning_tokens,
                    "truncated": u.truncated, "rate_limited": u.rate_limited, "gone": u.gone, "cost_usd": u.cost_usd,
                    "reasoning_refused": u.reasoning_refused, "template_refused": u.template_refused,
                    "thinking": u.thinking, "thinking_shown": u.thinking_shown,
                    "thinking_leaked": u.thinking_leaked, "finish_reasons": dict(u.finish_reasons),
                    "tool_calls": u.tool_calls, "tool_refused": u.tool_refused, "turned_away": u.turned_away,
                    **({"desk_max_output": self.max_output} if self.max_output > MAX_OUTPUT_TOKENS else {}),
                    **({"rate_uncharged": u.rate_uncharged} if RATE_UNCHARGED else {}),
                    **({"rate_wait_s": round(u.rate_wait_s, 3)} if RATE_WAIT_S > 0 or RATE_UNTIMED else {}),
                    **({"rate_untimed": True} if RATE_UNTIMED else {}),
                    **({"floor_wait_s": round(u.floor_wait_s, 3), "served_s": round(u.served_s, 3)} if paced() else {}),
                    **({"tool_choice_refused": u.tool_choice_refused} if u.tool_choice_refused else {}),
                    **({"pool": {word: dict(said) for word, said in u.pool.items()}} if u.pool else {}),
                    **({"pool_spent": u.pool_spent} if u.pool_spent else {}),
                    **({"servings": dict(u.servings)} if u.servings else {}),
                    **u.day,
                    "host": MOCK if self.config is None else self.config.host,
                    "model": MOCK if self.config is None else self.config.model}

    def time_left(self) -> float:
        """The seconds the next attempt may take: its own bound, less where the unit's clock ends sooner."""
        if self.deadline is None:
            return TIMEOUT_S
        return min(TIMEOUT_S, self.deadline - RESERVE_S - self._clock())

    @contextmanager
    def _counting(self) -> Iterator[Usage]:
        """`LOCK` held while the caller changes the counts, and the usage file rewritten once it is let go. No method
        agent/common/house.py overrides to take its own lock is called under `LOCK`, here or anywhere: those take the
        agent's lock and then this one, so holding this one while asking for that one could deadlock. Its `usage_line`,
        which takes no lock of its own, is the one override read under `LOCK` (by `_write_usage`)."""
        with LOCK:
            yield self.usage
        self._write_usage()

    def _admit(self) -> float:
        """Count the request about to be sent and return its timeout, or refuse it unsent. The check against the cap
        and the count are one step under `LOCK`: of requests asked at once, by threads or by clients sharing the
        usage, none passes the cap, and one asked while the last places are in flight is refused, not queued,
        though a 407 may yet give its place back."""
        u = self.usage
        refusal: ModelError | None = None
        with LOCK:
            if u.exhausted:
                u.refused += 1
                refusal = BudgetExhausted("the house refused this unit's requests (403 grant_denied); none sent")
            elif u.gone:
                u.refused += 1
                refusal = RouteGone(f"request {u.requests + 1} of the unit refused: the route answered 410, it no "
                                    f"longer serves the model; none sent")
            elif u.requests >= REQUESTS_PER_UNIT:
                u.refused += 1
                refusal = RequestCapReached(f"request {u.requests + 1} of the unit refused: the house allows "
                                            f"{REQUESTS_PER_UNIT}")
            elif (timeout := self.time_left()) < MIN_CALL_S:
                u.late += 1
                refusal = DeadlineReached(f"request {u.requests + 1} of the unit refused: {max(timeout, 0.0):.1f} s "
                                          f"left before the unit's clock less its {RESERVE_S:.0f} s reserve")
            else:
                u.requests += 1
        self._write_usage()
        if refusal is not None:
            raise refusal
        return timeout

    def _turned_away(self) -> None:
        """The attempt `_admit` counted, uncounted: the proxy turned it away (407) before the House admitted it."""
        with self._counting() as u:
            u.requests -= 1
            u.turned_away += 1

    def _rate_uncharged(self) -> None:
        """The attempt `_admit` counted, uncounted: the route answered 429 and the desk's `RATE_UNCHARGED` says the
        House, which has no such limiter, would not have charged it."""
        with self._counting() as u:
            u.requests -= 1
            u.rate_uncharged += 1

    def _asked(self, mode: str) -> None:
        """The thinking mode a body asks, on the usage line: "mixed" once the unit's bodies differ."""
        with LOCK:
            self.usage.thinking = mode if self.usage.thinking in (None, mode) else "mixed"

    def _rate_wait(self, seconds: float, bound: float | None = None) -> None:
        """The pacing's wait after a 429 (lane clock1), its seconds counted (`rate_wait_s`): within the time left, as
        before; under the desk's `RATE_UNTIMED` waited whole and the unit's deadline moved past it, so the wait costs
        the unit's clock nothing (the House, which has no such limiter, would never have made it). The move and the
        count come before the wait and the usage file says them (lane clock2): through the wait, the agent's watchdog
        reading the moved deadline and the harness's kill reading the file stand past it already. `bound`: the seconds a
        caller's bound on the request leaves (`_post`), which a timed wait never passes either. The seconds waited."""
        if RATE_UNTIMED and self.deadline is not None:
            waited = max(0.0, seconds)
            with self._counting() as u:
                self.deadline += waited
                self._origin += waited
                u.rate_wait_s += waited
            self._sleep(waited)
            return waited
        waited = max(0.0, min(seconds, self.time_left(), math.inf if bound is None else bound))
        self._sleep(waited)
        with LOCK:
            self.usage.rate_wait_s += waited
        return waited

    def _timed(self, timing: Timing) -> None:
        """A request's time kept where the desk's pacing is set (`paced`) and an attempt was sent: summed on the usage
        line, and its `send` event in the agent's trace (`_traced`) with the request's ordinal, the unit's count."""
        if timing.sent_s is None or not paced():
            return
        with self._counting() as u:
            u.floor_wait_s += timing.floor_wait_s
            u.served_s += timing.served_s
            ordinal = u.requests
        _traced(SEND_STEP, ordinal=ordinal, sent_s=round(timing.sent_s, 3), floor_wait_s=round(timing.floor_wait_s, 3),
                served_s=round(timing.served_s, 3), rate_wait_s=round(timing.rate_wait_s, 3))

    def _count(self, prompt: int, completion: int, cost: float | None) -> None:
        with self._counting() as u:
            u.prompt_tokens += prompt
            u.completion_tokens += completion
            if cost is not None:
                u.cost_usd = (u.cost_usd or 0.0) + cost

    def _write_usage(self) -> None:
        """The usage line to the usage file, read and written under `LOCK`, so of two threads' writes the later holds
        the later counts and neither replaces the other's part file."""
        if self.usage_path is None:
            return
        with LOCK:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            part = self.usage_path.with_name(self.usage_path.name + ".part")
            part.write_text(json.dumps(self.usage_line()) + "\n", encoding="utf-8")
            part.replace(self.usage_path)

    def _scrub(self, text: str) -> str:
        tokens = [self.config.token] if self.config and self.config.token else []
        for token in [*tokens, *(key.token for key in (self.config.pool if self.config else ()))]:
            if token.reveal():
                text = text.replace(token.reveal(), "<redacted>")
        return text

    def _pick(self) -> PoolKey | None:
        """The pool's key the next attempt goes through: of the keys not spent, the one with the most of its day left
        (its `left` as the unit started less the attempts admitted through it since, `harness.attempts_of`'s
        reading), or where the desk counted no key's day the one with the fewest admitted; ties go to the pool's
        order; where an answer through the key gave the platform's own figure of its day (`day_of`), that figure is its
        day left. A key is spent once it answered with its day's cap or its count is at none. With every free key spent,
        the paid fallback (`PoolKey.fallback`) where the pool holds it. None off a pool; where every key is spent,
        `PoolSpent`, nothing sent or counted against the unit."""
        pool = self.config.pool if self.config is not None else ()
        if not pool:
            return None
        ranked, spent, fallback = [], [], []
        with LOCK:
            for at, key in enumerate(pool):
                said = self.usage.pool.get(key.word, {})
                if key.fallback:
                    fallback += [] if said.get("capped") else [key]
                    continue
                admitted = said.get("requests", 0) + said.get("rate_uncharged", 0) - said.get("rate_limited", 0)
                left = None if key.left is None else key.left - admitted
                whose = "the desk's count"
                if "day_remaining" in said:                  # the platform's own, as its last answer said it
                    since = said.get("requests", 0) + said.get("rate_uncharged", 0) - said.get("day_attempts", 0)
                    left, whose = said["day_remaining"] - max(0, since), "the platform's count"
                if said.get("capped") or (left is not None and left <= 0):
                    spent.append(f"{key.word}: " + ("answered with its day's cap" if said.get("capped") else
                                                    f"{max(left, 0)} left by {whose}"))
                    continue
                ranked.append(((left if left is not None else -admitted), -at, key))
        if ranked:
            return max(ranked, key=lambda rank: rank[:2])[2]
        if fallback:
            return fallback[0]
        with self._counting() as u:
            u.pool_spent += 1
        paid = "" if any(key.fallback for key in pool) else f", or with its paid fallback ({POOL_PAID_ENV} unset)"
        raise PoolSpent(f"POOL-SPENT: every key of the pool {self.config.provider or 'handed'} has met its day "
                        f"({'; '.join(spent)}); none sent; remedy: run after 00:00Z (UTC), on another route{paid}")

    def _note(self, key: PoolKey, **values: int) -> None:
        """A pool key's day as the platform last said it (`day_of`), in place of the last."""
        with self._counting() as u:
            said = u.pool.setdefault(key.word, {"requests": 0, "rate_limited": 0, "rate_uncharged": 0})
            said.update(values, day_attempts=said["requests"] + said["rate_uncharged"])

    def _tally(self, key: PoolKey | None, **changes: float) -> None:
        """A pool key's counts moved (`Usage.pool`): `requests`, `rate_limited`, `rate_uncharged` and `cost_usd` as the
        line's own, `capped` the day's cap answers; nothing off a pool."""
        if key is None:
            return
        with self._counting() as u:
            said = u.pool.setdefault(key.word, {"requests": 0, "rate_limited": 0, "rate_uncharged": 0})
            for name, by in changes.items():
                said[name] = said.get(name, 0) + by

    def _post(self, system: str, user: str, max_tokens: int, reasoning: Mapping | None = None,
              template: Mapping | None = None, sampling: Mapping | None = None, tools: list | None = None,
              tool_choice=None, messages: list | None = None, replayed: int | None = None,
              bound: float | None = None) -> Reply:
        """The request's attempts (`_attempts`), its time kept (`Timing`, `_timed`) however it ends."""
        timing = Timing()
        try:
            return self._attempts(timing, system, user, max_tokens, reasoning, template, sampling, tools, tool_choice,
                                  messages, replayed, bound)
        finally:
            self._timed(timing)

    def _attempts(self, timing: Timing, system: str, user: str, max_tokens: int, reasoning: Mapping | None = None,
                  template: Mapping | None = None, sampling: Mapping | None = None, tools: list | None = None,
                  tool_choice=None, messages: list | None = None, replayed: int | None = None,
                  bound: float | None = None) -> Reply:
        """`timing`: the request's (`Timing`), filled as its attempts go. `messages`: the whole conversation
        (`converse`) in place of the one system and one user turn; `replayed`: the index among them of an assistant
        message replayed as a route returned it (`chat`'s `prior`); `bound`: `chat`'s, its end moved with the unit's
        deadline, as an untimed wait moves that."""
        config = self.config
        until, began = (None, None) if bound is None else (self._clock() + bound, self.deadline)

        def bound_left() -> float:
            moved = self.deadline - began if self.deadline is not None and began is not None else 0.0
            return until + moved - self._clock()

        def allowed() -> float:
            """The seconds the next attempt or wait may take: the unit's clock's, and the caller's bound's where set."""
            return self.time_left() if until is None else min(self.time_left(), bound_left())

        payload = {"model": config.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
                   if messages is None else [dict(message) for message in messages],
                   "temperature": TEMPERATURE, "seed": SEED, "max_tokens": max_tokens, **(sampling or {})}
        if config.host == "openrouter.ai":
            payload["usage"] = {"include": True}      # OpenRouter reports the call's cost only when asked
        if config.provider_pin is not None:
            payload["provider"] = dict(config.provider_pin)
        if not self.usage.template_refused:
            payload["chat_template_kwargs"] = dict(template or THINKING_OFF)
        asked = reasoning is not None and not self.usage.reasoning_refused
        if asked:
            payload["reasoning"] = dict(reasoning)
        elif REASONING_FIELD and config.host in REASONING_FIELD_HOSTS and payload.get("chat_template_kwargs", {}).get("enable_thinking") is False:
            # OpenRouter drops the house's chat_template_kwargs and reasons anyway (10-09 02:1xZ: 154 reasoning tokens and
            # 14 s where its own field gave 0 and 2 s); its own field says the same thing. Never sent to the house.
            payload["reasoning"] = {"enabled": False}
        if tools is not None and not self.usage.tool_refused:
            payload["tools"] = list(tools)
            if tool_choice is not None and not self._choice_routed_out:
                payload["tool_choice"] = tool_choice
        body = json.dumps(payload).encode("utf-8")
        failure, turned_away = "no attempt made", False
        attempts, only_limited = ATTEMPTS, True   # a run of 429s alone may take RATE_ATTEMPTS (desk only)
        attempt = -1
        retry_key = None
        while attempt + 1 < attempts:
            attempt += 1
            while True:
                if MIN_INTERVAL_S > 0:
                    gap = self._last_sent + MIN_INTERVAL_S - self._clock()
                    if gap > 0:
                        waited = max(0.0, min(gap, allowed()))
                        self._sleep(waited)
                        timing.floor_wait_s += waited
                if until is not None and bound_left() < MIN_CALL_S <= self.time_left():
                    raise BoundReached(f"the request's bound of {bound:.0f} s reached; none sent after it: {failure}")
                key = retry_key or self._pick()
                retry_key = None
                timeout = self._admit()
                self._tally(key, requests=1)
                if until is not None:
                    timeout = min(timeout, bound_left())
                headers = {"Content-Type": "application/json"}
                token = key.token if key is not None else config.token
                if token is not None:
                    headers["Authorization"] = "Bearer " + token.reveal()
                floor, sent = self._last_sent, self._clock()
                self._last_sent = sent
                wire = body if key is None or key.model is None else json.dumps(
                    {**payload, "model": key.model,
                     **({"provider": {"allow_fallbacks": True}} if key.fallback else {})}).encode("utf-8")
                try:
                    status, raw, *heard = self._send(config.url, headers, wire, timeout)
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    status, raw, heard = 407 if TUNNEL_407 in str(exc) else 0, b"", []
                    failure = f"{type(exc).__name__} reaching {config.host}"
                timing.sent_s, timing.served_s = sent - self._origin, self._clock() - sent
                if key is not None and not key.fallback and heard and (day := day_of(heard[0])) is not None:
                    self._note(key, **day, day_at=int(self._clock()))
                elif key is None and config.host == "openrouter.ai" and config.model.endswith(":free") and heard \
                        and (day := day_of(heard[0])) is not None:
                    with self._counting() as u:
                        u.day.update(day, day_at=int(self._clock()))
                if status != 407:
                    break
                self._turned_away()
                self._tally(key, requests=-1)
                if turned_away:
                    raise ModelCallError(f"the proxy turned the request away twice before admission (HTTP 407 "
                                         f"reaching {config.host}); neither was charged")
                turned_away = True
                self._sleep(max(0.0, min(PROXY_BACKOFF_S, allowed())))
            if status == 200:
                try:
                    off = payload.get("chat_template_kwargs", {}).get("enable_thinking") is False
                    doc = json.loads(raw.decode("utf-8"))
                    reply = self._reply(doc, off)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    failure = f"a malformed reply from {config.host} ({type(exc).__name__})"
                else:
                    if key is not None and isinstance(cost := (doc.get("usage") or {}).get("cost"), (int, float)):
                        self._tally(key, cost_usd=float(cost))
                    if "reasoning" not in payload and reasoning is not None and asked:
                        with self._counting() as u:
                            u.reasoning_refused = True          # answered without it: never sent again
                    return reply
            elif status:
                said = self._scrub(raw.decode("utf-8", errors="replace"))[:500]
                if status == 403 and _budget_refusal(said):
                    with LOCK:
                        self.usage.exhausted = True
                    raise BudgetExhausted("the house refused the request: the unit's allowance is used up")
                # the routing's 404 for a request that names its tool (lane toolchoice1): not a refused route but a
                # refused `tool_choice`, answered as the 400 and 422 that name it are, below
                routed_out = status == 404 and "tool_choice" in payload and all(words in said for words in ROUTED_OUT)
                if status in (401, 403, 404, 410) and not routed_out:
                    # 410: the route retired the model (NVIDIA's, 10-03 09:00Z), so no attempt can succeed
                    with self._counting() as u:
                        u.gone += status == 410
                    raise ModelConfigError(f"{config.host} refused the credential or the route: HTTP {status}: {said}")
                if status in (400, 422) and "chat_template_kwargs" in payload and "chat_template_kwargs" in said:
                    # a route that names the house's field as the fault: without it, and never again; tested
                    # first, so a refusal naming the field never costs the reasoning setting
                    del payload["chat_template_kwargs"]
                    with self._counting() as u:
                        u.template_refused = True
                    body = json.dumps(payload).encode("utf-8")
                    failure = f"HTTP {status} from {config.host} with chat_template_kwargs: {said}"
                    continue
                fields = [] if replayed is None else [name for name in payload["messages"][replayed]
                                                      if name not in ("role", "content")]
                if status in (400, 422) and any(name in said for name in fields):
                    # a route that refuses a replayed message's own field (its reasoning, say): the message's text
                    # alone, once; tested before the tools and the reasoning setting, which it never costs
                    message = payload["messages"][replayed]
                    payload["messages"][replayed] = {"role": message.get("role", "assistant"),
                                                     "content": message.get("content") or ""}
                    replayed = None
                    body = json.dumps(payload).encode("utf-8")
                    failure = f"HTTP {status} from {config.host} with the replayed message's fields: {said}"
                    continue
                if "tool_choice" in payload and (routed_out or status in (400, 422) and "tool_choice" in said):
                    # a route that refuses the choice of tool alone (lane clock1's forced write), or a routing that
                    # found no endpoint for it: the same tools without it, once; tested before the tools, which it
                    # never costs
                    del payload["tool_choice"]
                    with self._counting() as u:
                        u.tool_choice_refused += 1
                        if routed_out:
                            self._choice_routed_out = True
                    body = json.dumps(payload).encode("utf-8")
                    failure = f"HTTP {status} from {config.host} with tool_choice: {said}"
                    continue
                if status in (400, 422) and "tools" in payload and "tool" in said.lower():
                    # a route that names the tools as the fault (DeepInfra's 422 names `tools.0...` in its `param`):
                    # without them, and never again; tested before the reasoning setting, which it never costs
                    del payload["tools"]
                    payload.pop("tool_choice", None)
                    with self._counting() as u:
                        u.tool_refused = True
                    body = json.dumps(payload).encode("utf-8")
                    failure = f"HTTP {status} from {config.host} with tools: {said}"
                    continue
                if status in (400, 422) and "reasoning" in payload:
                    # a route that does not take the setting: the same request without it, once
                    del payload["reasoning"]
                    body = json.dumps(payload).encode("utf-8")
                    failure = f"HTTP {status} from {config.host} with the reasoning setting: {said}"
                    continue
                if status in (400, 413, 422):
                    # this request's bytes (a prompt past the context, say): no retry, and it costs this call alone
                    raise ModelCallError(f"{config.host} refused the request: HTTP {status}: {said}")
                if status == 429:
                    if self._last_sent == sent:     # a refused attempt never moves the floor's clock (lane floor1)
                        self._last_sent = floor
                    day_cap = key is not None and DAY_CAP_SAID in said
                    with self._counting() as u:     # the last attempt's 429 too: no later request rewrites the file
                        u.rate_limited += 1
                        if key is not None:
                            through = u.pool[key.word]
                            through["rate_limited"] += 1
                            if day_cap:
                                through["capped"] = through.get("capped", 0) + 1
                    if RATE_UNCHARGED:
                        self._rate_uncharged()
                        self._tally(key, requests=-1, rate_uncharged=1)
                    if day_cap:
                        # a pool's key whose day is spent: never asked again, and the attempt given back to go at
                        # once through the next key (or be refused, `_pick`, where none is left)
                        attempts += 1
                        failure = f"HTTP 429 from {config.host} through the pool's key {key.word}: {said}"
                        continue
                    if only_limited and RATE_ATTEMPTS > attempts:
                        attempts = RATE_ATTEMPTS
                    retry_key = key
                else:
                    only_limited = False
                failure = f"HTTP {status} from {config.host}: {said}"
            else:
                only_limited = False
            if attempt + 1 < attempts:
                if status == 429 and RATE_WAIT_S > 0:
                    timing.rate_wait_s += self._rate_wait(RATE_WAIT_S * (attempt + 1),
                                                          None if until is None else bound_left())
                else:
                    self._sleep(max(0.0, min(2 ** attempt, 8, allowed())))
        if until is not None and bound_left() < MIN_CALL_S <= self.time_left():
            raise BoundReached(f"the request's bound of {bound:.0f} s reached after {attempts} attempts: {failure}")
        raise ModelCallError(f"the model call failed after {attempts} attempts: {failure}")

    def _send(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple:
        """The transport, its reply abandoned at `timeout` seconds even where the host keeps the socket busy past
        the per-read timeout urllib applies; the abandoned thread ends with that socket."""
        done: list = []

        def call() -> None:
            try:
                done.append(self._transport(url, headers, body, timeout))
            except BaseException as exc:  # noqa: BLE001 - handed to the caller's thread, which sorts it
                done.append(exc)

        worker = threading.Thread(target=call, name="model-call", daemon=True)
        worker.start()
        worker.join(timeout)
        if not done:
            raise TimeoutError(f"no complete reply in {timeout:.0f} s")
        if isinstance(done[0], BaseException):
            raise done[0]
        return done[0]

    def _reply(self, doc: dict, thinking_off: bool = False) -> Reply:
        """The reply, counted; `thinking_off`: the body asked thinking off, so thinking the reply shows is a leak
        (`thinking_leaked`); else it is the thinking a route asked on gave (`thinking_shown`)."""
        choice = doc["choices"][0]
        usage = doc.get("usage") or {}
        cost = usage.get("cost")
        details = usage.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        serving = doc.get("provider")
        reply = Reply(choice["message"].get("content") or "", int(usage.get("prompt_tokens") or 0),
                      int(usage.get("completion_tokens") or 0), choice.get("finish_reason"),
                      int(reasoning) if isinstance(reasoning, int) and not isinstance(reasoning, bool) else 0,
                      _tool_calls(choice["message"].get("tool_calls")), dict(choice["message"]),
                      serving.strip() if isinstance(serving, str) else "")
        shown = reply.reasoning_tokens > 0 or _thought(choice["message"], reply.text)
        finish = str(reply.finish_reason)
        u = self.usage
        with LOCK:
            u.tool_calls += len(reply.tool_calls)
            u.truncated += reply.finish_reason == "length"
            u.reasoning_tokens += reply.reasoning_tokens
            u.thinking_leaked += thinking_off and shown
            u.thinking_shown += not thinking_off and shown
            u.finish_reasons[finish] = u.finish_reasons.get(finish, 0) + 1
            if reply.provider:
                u.servings[reply.provider] = u.servings.get(reply.provider, 0) + 1
        self._count(reply.prompt_tokens, reply.completion_tokens,
                    float(cost) if isinstance(cost, (int, float)) else None)
        return reply


def _thought(message: dict, text: str) -> bool:
    """Thinking a reply shows whatever tokens it reports: a non-blank `reasoning_content` (a route whose reasoning
    parser is on) or a non-blank `<think>` block in its content (one whose parser is off). The empty `<think></think>`
    the House's template writes with thinking off is none."""
    reasoning = message.get("reasoning_content")
    _, opened, rest = text.partition("<think>")
    return (isinstance(reasoning, str) and bool(reasoning.strip())) or (
        bool(opened) and bool(rest.partition("</think>")[0].strip()))


def _tool_calls(calls) -> tuple[ToolCall, ...]:
    """A reply's `tool_calls` (OpenAI's shape: each a `function` with a `name` and `arguments`, a JSON string or an
    object): a call without a name is skipped, arguments that do not read as a JSON object are kept as written."""
    out = []
    for call in calls if isinstance(calls, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            continue
        raw = function.get("arguments")
        raw = raw if isinstance(raw, str) else json.dumps(raw)
        try:
            arguments = json.loads(raw)
        except ValueError:
            arguments = None
        out.append(ToolCall(name, arguments if isinstance(arguments, dict) else None, raw))
    return tuple(out)


def _budget_refusal(body: str) -> bool:
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return False
    return isinstance(error, dict) and error.get("code") == _BUDGET_CODE and error.get("message") == _BUDGET_MESSAGE
