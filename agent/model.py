"""The HTTP client for the competition's model. Every request the agent makes goes through `Client`.

Called by: common/house.py, which subclasses `Client` to add the agent's own request cap; agent/n11 uses that.
Start at: `Client.converse` (the loop's requests), `Client.chat` (the selection's and the draft's) and
    `unit_deadline` (the clock). The constants below this docstring are the limits.

The rules in detail
-------------------
The model client: every request the agent makes goes through `Client`.

It reads the platform's `MODEL_ENDPOINT`, `MODEL_NAME` and `MODEL_TOKEN`, and holds the platform's limits in code:
the seed and the temperature pinned, at most `MAX_OUTPUT_TOKENS` asked of a call, a request past `REQUESTS_PER_UNIT`
or past the unit's clock (`unit_deadline`) refused before it is sent. Every attempt sent counts, retries included,
except one the proxy turned away with a 407 before admission. The counts are shared by every client and thread of the
process (one process runs one unit) and changed under one lock (`LOCK`), so concurrent requests never pass the cap.

Every body asks thinking off through the platform's `chat_template_kwargs` unless the caller asks low effort; a route
that refuses that field by name is asked again without it, and never sent it again. A 410 means the route no longer
serves the model: every later request is refused unsent. A caller may bound one request's wall time (`bound=`): no
attempt or wait runs past it (`BoundReached`).

Seven variables tune the client for a hosted route such as OpenRouter's (the public reproduction sets them; the
platform sets none): `AGENTHON_MODEL_REASONING_FIELD`, `AGENTHON_MODEL_PROVIDER_PIN`, `AGENTHON_MODEL_MIN_INTERVAL_S`,
`AGENTHON_MODEL_RATE_WAIT_S`, `AGENTHON_MODEL_RATE_ATTEMPTS`, `AGENTHON_MODEL_RATE_UNCHARGED` and
`AGENTHON_MODEL_RATE_UNTIMED`. Unset, the client behaves as on the platform.

Standard library only. The key is never printed, logged or carried in an exception's text.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import threading
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

REQUESTS_PER_UNIT = 25
MAX_OUTPUT_TOKENS = 4000
TEMPERATURE = 0.0
SEED = 20260731
ATTEMPTS = 3                    # a transient failure is retried; each attempt is a request
# A hosted route's rate limiter: the wait after a 429 (times the attempt), the attempts while every failure is a 429,
# the least gap between two sends, a 429 attempt left uncounted, and a 429's wait taken off the unit's clock.
RATE_WAIT_S = float(os.environ.get("AGENTHON_MODEL_RATE_WAIT_S", "0") or 0)
RATE_ATTEMPTS = int(os.environ.get("AGENTHON_MODEL_RATE_ATTEMPTS", "0") or 0)
MIN_INTERVAL_S = float(os.environ.get("AGENTHON_MODEL_MIN_INTERVAL_S", "0") or 0)
RATE_UNCHARGED = os.environ.get("AGENTHON_MODEL_RATE_UNCHARGED", "") == "1"
RATE_UNTIMED = os.environ.get("AGENTHON_MODEL_RATE_UNTIMED", "") == "1"
TIMEOUT_S = 300.0               # one attempt's own bound: 4,000 tokens at the slowest rate measured, 16/s, is 250 s
UNIT_SECONDS = 1800.0           # the unit's clock where its card names no [agent] timeout_sec
# The platform's clock starts before this process does (the container's creation, the image's pull, the
# provisioning), so the card's clock less this allowance is what the process may use.
PLATFORM_ALLOWANCE_S = 200.0
ROSTER_CAP_S = 600.0            # a unit's clock at most, whatever its card says
RESERVE_S = 30.0                # kept from the deadline: the answer's last write
MIN_CALL_S = 1.0                # less time left than this is no time left
PROXY_BACKOFF_S = 2.0           # a request the proxy turned away (HTTP 407) is sent once more after this
REASONING_FIELD_HOSTS = ("openrouter.ai",)   # routes that take thinking off only through their own `reasoning` field
REASONING_FIELD = os.environ.get("AGENTHON_MODEL_REASONING_FIELD", "") == "1"
PIN_ENV = "AGENTHON_MODEL_PROVIDER_PIN"      # the body's `provider` object (OpenRouter's provider routing), as JSON
TUNNEL_407 = "Tunnel connection failed: 407"   # http.client's error where the proxy refuses an https CONNECT
THINKING_OFF = {"enable_thinking": False}   # the platform's switch; omitting it leaves thinking on
THINKING_LOW = {"enable_thinking": True, "low_effort": True}

# The platform's 403 once a unit's allowance is used up.
_BUDGET_CODE, _BUDGET_MESSAGE = "grant_denied", "request admission refused"


class ModelError(RuntimeError):
    """A model call that produced no reply."""


class ModelConfigError(ModelError):
    """No usable endpoint, model or key, or the endpoint refused the request as misconfigured."""


class ModelCallError(ModelError):
    """The call failed after its attempts (network, timeout, server error, malformed reply)."""


class RouteGone(ModelConfigError):
    """The client refused a request unsent: the route answered 410 earlier in this process."""


class BudgetExhausted(ModelError):
    """The unit's request allowance is used up; no further request is sent."""


class RequestCapReached(BudgetExhausted):
    """The client refused a request past the cap before sending it."""


class DeadlineReached(BudgetExhausted):
    """The client refused a request the unit's clock leaves no time for, before sending it."""


class BoundReached(ModelCallError):
    """The caller's bound on one request (`bound=`) ran out before a reply."""


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
class Config:
    endpoint: str               # the base URL: the platform's origin, or a provider's URL ending /v1
    model: str
    token: _Secret | None
    provider_pin: dict | None = field(default=None, hash=False)   # the body's `provider`

    @property
    def host(self) -> str:
        return urlsplit(self.endpoint).hostname or ""

    @property
    def url(self) -> str:
        base = self.endpoint.rstrip("/")
        # a base that ends /v1 (OpenRouter) or carries /v1/ inside it is complete; the platform's bare origin gets /v1
        if not (base.endswith("/v1") or "/v1/" in base):
            base += "/v1"
        return base + "/chat/completions"


@dataclass
class Usage:
    """The unit's state shared by its clients, each change under `LOCK`."""

    requests: int = 0           # requests sent (every attempt)
    gone: int = 0               # attempts the host answered 410: refuse every later request
    exhausted: bool = False     # the platform said the allowance is used up: send nothing more
    template_refused: bool = False    # the route refused `chat_template_kwargs`: it is not sent again


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    reasoning_tokens: int = 0


_UNIT = Usage()
LOCK = threading.RLock()


def resolve() -> Config:
    """The endpoint, model and key the platform names."""
    env = os.environ
    if not env.get("MODEL_ENDPOINT"):
        raise ModelConfigError("no model configured: MODEL_ENDPOINT is unset")
    token = env.get("MODEL_TOKEN")
    return Config(env["MODEL_ENDPOINT"], env.get("MODEL_NAME", ""), _Secret(token) if token else None,
                  provider_pin(env.get(PIN_ENV)))


def provider_pin(text: str | None) -> dict | None:
    """`PIN_ENV` as the body's `provider` object; None where it is unset or empty."""
    if not text:
        return None
    try:
        pin = json.loads(text)
    except ValueError:
        pin = None
    if not isinstance(pin, dict) or not pin:
        raise ModelConfigError(f"{PIN_ENV} is not a non-empty JSON object")
    return pin


def unit_deadline(card: Path | None = None) -> float:
    """The unit's end in epoch seconds: this process's start plus the smaller of the card's `[agent] timeout_sec`
    (`UNIT_SECONDS` without one) less `PLATFORM_ALLOWANCE_S`, and `ROSTER_CAP_S`."""
    seconds = UNIT_SECONDS
    try:
        seconds = float(tomllib.loads(card.read_text(encoding="utf-8"))["agent"]["timeout_sec"])
    except (AttributeError, OSError, KeyError, TypeError, ValueError):
        pass
    return time.time() - process_age() + min(seconds - PLATFORM_ALLOWANCE_S, ROSTER_CAP_S)


def process_age() -> float:
    """Seconds since this process started (Linux's /proc), or 0 where that cannot be read."""
    try:
        ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return max(0.0, uptime - ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return 0.0


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read()
        except (OSError, http.client.HTTPException):
            return exc.code, b""


def thinking_template(thinking: str | None) -> dict:
    """The body's `chat_template_kwargs`: None asks thinking off (`THINKING_OFF`), "low" asks it on at low effort
    (`THINKING_LOW`)."""
    if thinking is None:
        return dict(THINKING_OFF)
    if thinking == "low":
        return dict(THINKING_LOW)
    raise ValueError('thinking is None or "low"')


class Client:
    """`chat` and `converse` under the platform's limits; `Client.from_env()` is what an agent builds."""

    def __init__(self, config: Config, *, deadline: float) -> None:
        """`deadline`: the unit's end in epoch seconds."""
        self.config = config
        self.usage = _UNIT
        self.deadline = deadline
        self._last_sent = 0.0        # the floor's clock (MIN_INTERVAL_S): the last send the host did not answer 429

    @classmethod
    def from_env(cls, card: Path | None = None) -> Client:
        """The platform's endpoint under the unit's deadline (`unit_deadline`, from the unit's `card`)."""
        deadline = unit_deadline(card)
        return cls(resolve(), deadline=deadline)

    def chat(self, system: str, user: str, *, max_tokens: int | None = None, thinking: str | None = None,
             bound: float | None = None) -> Reply:
        """One system and one user turn. `max_tokens`: None asks `MAX_OUTPUT_TOKENS`; more is clamped to it.
        `thinking`: see `thinking_template`. `bound`: the seconds the caller allows this request, its attempts and
        waits included; None, the unit's clock alone."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._post(messages, max_tokens, thinking, bound)

    def converse(self, messages: list) -> Reply:
        """A whole conversation, OpenAI's `messages` list, under `chat`'s limits."""
        return self._post(messages, None, None, None)

    def time_left(self) -> float:
        """The seconds the next attempt may take: its own bound, less where the unit's clock ends sooner."""
        return min(TIMEOUT_S, self.deadline - RESERVE_S - time.time())

    def _admit(self) -> float:
        """Count the request about to be sent and return its timeout, or refuse it unsent. The check against the cap
        and the count are one step under `LOCK`, so of requests asked at once none passes the cap."""
        u = self.usage
        refusal: ModelError | None = None
        with LOCK:
            if u.exhausted:
                refusal = BudgetExhausted("the platform refused this unit's requests (403 grant_denied); none sent")
            elif u.gone:
                refusal = RouteGone(f"request {u.requests + 1} of the unit refused: the route answered 410, it no "
                                    f"longer serves the model; none sent")
            elif u.requests >= REQUESTS_PER_UNIT:
                refusal = RequestCapReached(f"request {u.requests + 1} of the unit refused: the platform allows "
                                            f"{REQUESTS_PER_UNIT}")
            elif (timeout := self.time_left()) < MIN_CALL_S:
                refusal = DeadlineReached(f"request {u.requests + 1} of the unit refused: {max(timeout, 0.0):.1f} s "
                                          f"left before the unit's clock less its {RESERVE_S:.0f} s reserve")
            else:
                u.requests += 1
        if refusal is not None:
            raise refusal
        return timeout

    def _uncount(self) -> None:
        """The attempt `_admit` counted, uncounted: the proxy turned it away (407) before admission, or the route
        answered 429 under `RATE_UNCHARGED`."""
        with LOCK:
            self.usage.requests -= 1

    def _rate_wait(self, seconds: float, bound: float | None = None) -> None:
        """The wait after a 429, within the time left and the caller's `bound`; under `RATE_UNTIMED` waited whole,
        the deadline moved past it first, so the wait costs the unit's clock nothing."""
        if RATE_UNTIMED:
            waited = max(0.0, seconds)
            with LOCK:
                self.deadline += waited
            time.sleep(waited)
            return
        time.sleep(max(0.0, min(seconds, self.time_left(), math.inf if bound is None else bound)))

    def _scrub(self, text: str) -> str:
        token = self.config.token
        if token is not None and token.reveal():
            text = text.replace(token.reveal(), "<redacted>")
        return text

    def _post(self, messages: list, max_tokens: int | None, thinking: str | None, bound: float | None) -> Reply:
        """The request's attempts. `bound`'s end moves with the unit's deadline, as an untimed wait moves that."""
        max_tokens = MAX_OUTPUT_TOKENS if max_tokens is None else min(int(max_tokens), MAX_OUTPUT_TOKENS)
        template = thinking_template(thinking)
        config = self.config
        until, began = (None, None) if bound is None else (time.time() + bound, self.deadline)

        def bound_left() -> float:
            return until + (self.deadline - began) - time.time()

        def allowed() -> float:
            """The seconds the next attempt or wait may take: the unit's clock's, and the caller's bound's where set."""
            return self.time_left() if until is None else min(self.time_left(), bound_left())

        payload = {"model": config.model, "messages": [dict(message) for message in messages],
                   "temperature": TEMPERATURE, "seed": SEED,
                   "max_tokens": max_tokens}
        if config.host == "openrouter.ai":
            payload["usage"] = {"include": True}      # OpenRouter reports the call's cost only when asked
        if config.provider_pin is not None:
            payload["provider"] = dict(config.provider_pin)
        if not self.usage.template_refused:
            payload["chat_template_kwargs"] = template
        if REASONING_FIELD and config.host in REASONING_FIELD_HOSTS and payload.get("chat_template_kwargs", {}).get("enable_thinking") is False:
            # OpenRouter drops the platform's chat_template_kwargs and reasons anyway; its own field says the same.
            payload["reasoning"] = {"enabled": False}
        body = json.dumps(payload).encode("utf-8")
        failure, turned_away = "no attempt made", False
        attempts, only_limited = ATTEMPTS, True   # a run of 429s alone may take RATE_ATTEMPTS
        attempt = -1
        while attempt + 1 < attempts:
            attempt += 1
            while True:
                if MIN_INTERVAL_S > 0:
                    gap = self._last_sent + MIN_INTERVAL_S - time.time()
                    if gap > 0:
                        time.sleep(max(0.0, min(gap, allowed())))
                if until is not None and bound_left() < MIN_CALL_S <= self.time_left():
                    raise BoundReached(f"the request's bound of {bound:.0f} s reached; none sent after it: {failure}")
                timeout = self._admit()
                if until is not None:
                    timeout = min(timeout, bound_left())
                headers = {"Content-Type": "application/json"}
                if config.token is not None:
                    headers["Authorization"] = "Bearer " + config.token.reveal()
                floor, sent = self._last_sent, time.time()
                self._last_sent = sent
                try:
                    status, raw = self._send(config.url, headers, body, timeout)
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    status, raw = 407 if TUNNEL_407 in str(exc) else 0, b""
                    failure = f"{type(exc).__name__} reaching {config.host}"
                if status != 407:
                    break
                self._uncount()
                if turned_away:
                    raise ModelCallError(f"the proxy turned the request away twice before admission (HTTP 407 "
                                         f"reaching {config.host}); neither was charged")
                turned_away = True
                time.sleep(max(0.0, min(PROXY_BACKOFF_S, allowed())))
            if status == 200:
                try:
                    return _reply(json.loads(raw.decode("utf-8")))
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    failure = f"a malformed reply from {config.host} ({type(exc).__name__})"
            elif status:
                said = self._scrub(raw.decode("utf-8", errors="replace"))[:500]
                if status == 403 and _budget_refusal(said):
                    with LOCK:
                        self.usage.exhausted = True
                    raise BudgetExhausted("the platform refused the request: the unit's allowance is used up")
                if status in (401, 403, 404, 410):
                    with LOCK:
                        self.usage.gone += status == 410
                    raise ModelConfigError(f"{config.host} refused the credential or the route: HTTP {status}: {said}")
                if status in (400, 422) and "chat_template_kwargs" in payload and "chat_template_kwargs" in said:
                    # tested first, so a refusal naming this field never costs the reasoning setting
                    del payload["chat_template_kwargs"]
                    with LOCK:
                        self.usage.template_refused = True
                    body = json.dumps(payload).encode("utf-8")
                    failure = f"HTTP {status} from {config.host} with chat_template_kwargs: {said}"
                    continue
                if status in (400, 422) and "reasoning" in payload:
                    del payload["reasoning"]
                    body = json.dumps(payload).encode("utf-8")
                    failure = f"HTTP {status} from {config.host} with the reasoning setting: {said}"
                    continue
                if status in (400, 413, 422):
                    # this request's bytes (a prompt past the context, say): no retry
                    raise ModelCallError(f"{config.host} refused the request: HTTP {status}: {said}")
                if status == 429:
                    if self._last_sent == sent:     # a refused attempt never moves the floor's clock
                        self._last_sent = floor
                    if RATE_UNCHARGED:
                        self._uncount()
                    if only_limited and RATE_ATTEMPTS > attempts:
                        attempts = RATE_ATTEMPTS
                else:
                    only_limited = False
                failure = f"HTTP {status} from {config.host}: {said}"
            else:
                only_limited = False
            if attempt + 1 < attempts:
                if status == 429 and RATE_WAIT_S > 0:
                    self._rate_wait(RATE_WAIT_S * (attempt + 1), None if until is None else bound_left())
                else:
                    time.sleep(max(0.0, min(2 ** attempt, 8, allowed())))
        if until is not None and bound_left() < MIN_CALL_S <= self.time_left():
            raise BoundReached(f"the request's bound of {bound:.0f} s reached after {attempts} attempts: {failure}")
        raise ModelCallError(f"the model call failed after {attempts} attempts: {failure}")

    def _send(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
        """The transport, its reply abandoned at `timeout` seconds even where the host keeps the socket busy past
        the per-read timeout urllib applies; the abandoned thread ends with that socket."""
        done: list = []

        def call() -> None:
            try:
                done.append(_urllib_transport(url, headers, body, timeout))
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


def _reply(doc: dict) -> Reply:
    choice = doc["choices"][0]
    usage = doc.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    message = choice["message"]
    return Reply(message.get("content") or "", int(usage.get("prompt_tokens") or 0),
                 int(usage.get("completion_tokens") or 0), choice.get("finish_reason"),
                 int(reasoning) if isinstance(reasoning, int) and not isinstance(reasoning, bool) else 0)


def _budget_refusal(body: str) -> bool:
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return False
    return isinstance(error, dict) and error.get("code") == _BUDGET_CODE and error.get("message") == _BUDGET_MESSAGE
