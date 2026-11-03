"""The reconciliation after a clean run (lane recon1; the failure reads' cause 1, docs/research/failures-1017.md: a clean
run taken as validation while the task's quantity is wrong; the relitigation's lesson, docs/research/
relitigation-1017.md: a finding must name the object and the comparison, while a repair turn remains).

Each check reads an identity the harness can compute from the run's output files, the unit's input files and the
instruction's words alone, at no request and with no value from a check or an oracle, and says where the output's
numbers fail it: the file, the quantity, the identity and the side that disagrees, with the task's own words for the
convention where they pin one and "the task does not say which" where they do not (both sides tried then). Three
families (`FAMILIES`):

- the accounting replay: a path chains its returns (`chain`); a reported total return is the path's end over its start
  less one (`total`); a reported maximum drawdown is the one the path gives, its sign as the task's words pin it
  (`drawdown`); a reported annualised volatility or return agrees with the series at the stated periods and ddof
  (`annual`); a weights table's rows sum to what the task states (`weights`).
- the repricing identity: discount factors and zero rates agree at their times (`discount`); a model quantity beside
  the market quote it was bootstrapped to reprices it (`quotes`); survival probabilities are the hazards' (`survival`);
  an implied volatility reprices its option through the library's Black-Scholes (`pricer`).
- the date and window comparison: a series' dates lie in the inputs' range (`range`), on observed input dates where
  the task says trading days (`trading`), inside the window the task's words state (`bounds`), with no period the
  inputs hold skipped (`gaps`); a reported count of rebalances or periods is its table's (`count`); a path starts at
  the initial capital the task states (`start`).

The switch: `AGENTHON_N11_RECON` or agent/n11/RECON, `off` (today's), `on` or `refuse`, armed in forms `loop`, `review`
and `sketch` (agent/n11/solve.py `recon_armed`). Under `on` and `refuse` each clean run (exit 0, the checks passing) is
read (agent/n11/commands.py `Workspace.observe`) with lane derive1's checks `witness.KEPT` beside the checks `KEPT`
names (one finding an object: `DEDUPE`), and while a finding stands the run's verdict line is `VERDICT` in place of the
checks' passing line, the findings after it under `HEAD` as the observation's last block; a finding standing unchanged
after the scripts changed is said again in the same-frame form (`SAME`, agent/n11/solve.py `recon_same`); under
`refuse` a finish while one stands is turned away once a unit with the findings quoted (`REFUSED`), sharing one refusal
with the contract's and finishcheck's (agent/n11/solve.py `guards_turn`). No text states a count; under `off` every
body is today's. Keys `KEYS`.

`scripts/recon_replay.py` runs every check over the recorded final outputs of failing and solved readings; a check is
kept (`KEPT`) only where it fires on failing readings and on no solved one.

Pure but for `pricer`, which imports the library (agent/qf) when it has a row to price: nothing written, nothing named
of any unit (rule 8): a finding names files, keys, columns and the output's own dates or row labels, never a value of
the task's data.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from n11 import witness
from n11.witness import Files, Table, Task, leaf, number

ENV = "AGENTHON_N11_RECON"
FILE = "RECON"
SETTINGS = ("off", "on", "refuse")
FORMS = ("loop", "review", "sketch")    # the loop's forms (form `state` builds its own run text and finish)
KEYS = ("recon_mode", "recon_runs", "recon_said", "recon_kinds", "recon_refused", "recon_acted")
MAX = 1                     # the finishes the reconciliation turns away a unit; the next is today's
FAMILIES = {"replay": ("chain", "total", "drawdown", "annual", "weights"),
            "reprice": ("discount", "quotes", "survival", "pricer", "flag"),
            "window": ("range", "trading", "bounds", "gaps", "count", "start", "event", "empty")}
CHECKS = tuple(check for checks in FAMILIES.values() for check in checks)     # every check, in its said order
FAMILY = {check: family for family, checks in FAMILIES.items() for check in checks}
KEPT = ("chain", "total", "drawdown", "annual", "discount", "quotes", "survival", "pricer", "range", "trading", "gaps",
        "event", "empty")   # the checks a run's observation says and a refusal answers: those the replay kept, each
#   firing on failing readings and on no solved one over every record the ledgers name; `flag` fired on solved
#   readings, `weights`, `bounds`, `count` and `start` on none (runs/lanes/recon1-report.md, the replay)
HEAD = "The harness's reconciliation of the output's numbers: each line names a quantity and the identity it fails."
VERDICT = "checks: the files pass; the numbers do not reconcile yet:"   # the verdict line while a finding stands
SAME = ("same reconciliation failing as at the previous run, for {names}; the change since did not reach the code that "
        "computes the named quantity. Change that computation (a different fix), not the line that writes it.")
DEDUPE = {"schedule": ("gaps", "trading"), "nav": ("chain",)}   # derive1's checks said once an object beside these
REFUSED = ("finish not taken: the output's numbers fail an identity the harness reconciles from the output, the inputs "
           "and the task's words, so the task goes on. Correct the quantity each line names (or its convention, where "
           "the task states one), run the script, then finish:\n{lines}")
UNPINNED = "the task does not say which"
SAID = 6                    # findings said a run, at most: the harness's bound, never said
NAMED = 3                   # rows, dates or keys one finding names, at most
LINE_CHARS = 300            # a finding's line, at most
RELATIVE = 1e-4             # an identity's relative slack, past what the printed digits allow
MISSES = 2                  # rows a path must miss its chain on before it is said
SERIES = 6                  # rows a series needs before a window check reads it
HAZARD_SLACK = 0.02         # a cumulative hazard's relative slack (a tenor's label against its day count)
PRICE_SLACK = 1e-3          # a repriced option's relative slack, and in volatility (the price's miss over its vega)
VOL_SLACK = 0.005
DAILY_STEP, MONTHLY_STEP = 5, 35    # a series' median step in days: daily at or under the first, monthly the second
PERIODS_A_YEAR = (252, 365, 12, 52, 4, 1, 250, 260, 360)
PERIOD_OF = {"day": 252, "week": 52, "month": 12, "quarter": 4, "year": 1}

# the quantities' names
PATH_NAME = re.compile(r"nav|net_?asset|wealth|equity|portfolio_?value|value_?index|cum(?:ulative)?_?(?:value|wealth|"
                       r"ret(?:urn)?s?|growth)|growth|(?:^|_)capital(?:_|$)|balance|(?:^|_)index(?:_|$)", re.I)
CUM_NAME = re.compile(r"cum(?:ulative)?_?ret", re.I)
RETURN_NAME = re.compile(r"(?:^|_)(?:ret|rets|return|returns|r)(?:_|$)|ret$|_ret_", re.I)
NOT_RETURN = re.compile(r"cum|total|ann|excess_?over|mean|avg|average|std|vol|signal(?!_ret)|lookback|forward|lead|"
                        r"expected|exp_|predicted|forecast|bench(?!.*strat)|rank|z_?score|t_?stat|weight|turnover", re.I)
TOTAL_KEY = re.compile(r"^(?:total|cum(?:ulative)?)_?(?:return|ret)$|^(?:strategy|portfolio)_?total_?return$", re.I)
DRAWDOWN_KEY = re.compile(r"^(?:max(?:imum)?_?)drawdown$|^mdd$|^max_?dd$", re.I)
VOL_KEY = re.compile(r"^ann(?:ual(?:i[sz]ed)?)?_?(?:vol(?:atility)?|std|sigma)$", re.I)
ANNRET_KEY = re.compile(r"^ann(?:ual(?:i[sz]ed)?)?_?(?:ret(?:urn)?|mean_?return)$|^cagr$", re.I)
META_COLUMN = re.compile(r"turnover|cost|fee|leverage|gross|(?:^|_)net(?:_|$)|total|sum|count|(?:^|_)n(?:_|$)|num|"
                         r"return|(?:^|_)ret|nav|value|score|signal|rank|date|period|beta|vol|sharpe|exposure|"
                         r"(?:^|_)id(?:_|$)|month|year|day", re.I)
WEIGHT_TABLE = re.compile(r"weight|allocation|holding|position|portfolio_?composition", re.I)
TIME_COLUMN = re.compile(r"^(?:t|tau|time|times|time_?years|years?|year_?frac(?:tion)?|maturity|maturity_?years|"
                         r"tenor|tenor_?years|ttm|time_?to_?maturity|t_?years|T)$", re.I)
DF_COLUMN = re.compile(r"discount|(?:^|_)df(?:s|_|$)|zcb|zero_?coupon_?(?:price|bond)|(?:^|_)p(?:_|$)|^price$|"
                       r"bond_?price|risk_?free_?price", re.I)
RATE_COLUMN = re.compile(r"zero|yield|(?:^|_)rate(?:_|$)|spot_?rate|(?:^|_)r(?:_|$)", re.I)
NOT_RATE = re.compile(r"forward|fwd|(?:^|_)par|swap|spread|hazard|coupon|libor|discount|diff|error|resid|market|input|"
                      r"quoted?|observed|target|actual|(?:^|_)mkt", re.I)
MARKET = re.compile(r"market|input|quoted?|observed|target|actual|(?:^|_)mkt(?:_|$)", re.I)
MODEL = re.compile(r"model|fitted|(?:^|_)fit(?:_|$)|repriced|reprice|calibrated|bootstrapped|computed|"
                   r"(?:^|_)mdl(?:_|$)", re.I)
HAZARD_NAME = re.compile(r"hazard|intensit|lambda|(?:^|_)h(?:_|$)", re.I)
SURVIVAL_NAME = re.compile(r"surviv|(?:^|_)q(?:_|$)|(?:^|_)s(?:_|$)", re.I)
INTERVAL_KEY = re.compile(r"^h_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$", re.I)
TENOR = re.compile(r"(\d+(?:\.\d+)?)\s*([ymwd])\b|^(\d+(?:\.\d+)?)$", re.I)
STRIKE_COLUMN = re.compile(r"^(?:k|strike|strike_?price)$", re.I)
EXPIRY_COLUMN = re.compile(r"^(?:t|tau|maturity|expiry|time_?to_?(?:expiry|maturity)|ttm|years?)$", re.I)
IV_COLUMN = re.compile(r"implied_?vol(?:atility)?|(?:^|_)iv(?:_|$)|impl_?vol", re.I)
PRICE_COLUMN = re.compile(r"price|premium|(?:^|_)value(?:_|$)", re.I)
TYPE_COLUMN = re.compile(r"^(?:option_?type|type|cp|call_?put|right|kind)$", re.I)
SPOT_KEY = re.compile(r"^(?:s0|s_0|spot|spot_?price|underlying_?price|stock_?price|s)$", re.I)
RATE_KEY = re.compile(r"^(?:r|rate|risk_?free(?:_?rate)?|rf|interest_?rate|r_?f)$", re.I)
DIVIDEND_KEY = re.compile(r"^(?:q|dividend_?yield|div_?yield|dividend)$", re.I)
COUNT_KEY = re.compile(r"^(?:n|num|number|count|cnt)_?(?:of_?)?(rebalance|rebalances|rebalancing|rebalance_?dates|"
                       r"months|signal_?months|weeks|holding_?periods)$", re.I)
FLAG_KEY = re.compile(r"validat|reprices?$|repriced|converged|consistent|within_?tol", re.I)
EVENT_COLUMN = re.compile(r"event_?date|announce\w*_?date|earnings_?date", re.I)
RELATIVE_COLUMN = re.compile(r"^(?:relative_?day|rel_?day|event_?day|day_?rel(?:ative)?|event_?time|tau|offset)$", re.I)
SOURCE_COLUMN = re.compile(r"meeting|release|statement|scheduled|decision|report", re.I)
STATED_NUMBER = re.compile(r"`?\b(r|q|D|S0|S_0|rf|risk_free_rate|dividend_yield)\b`?\s*=\s*`?(-?\d+(?:\.\d+)?)")
SERIES_COLUMN = re.compile(r"return|ret(?:_|$)|nav|wealth|equity|value|weight|pnl|exposure", re.I)
SERIES_DATE = re.compile(r"date|month|period|(?:^|_)day|week|time", re.I)
NOT_SERIES_DATE = re.compile(r"maturity|pillar|expir|payment|settle|entry|exit|announce|event|fixing|start|end|"
                             r"reset|accrual|coupon|call|first|last", re.I)

# the task's words
SIMPLE_WORDS = re.compile(r"\bsimple\*{0,2}\s+(?:\(arithmetic\)\s+)?\*{0,2}(?:daily\s+|monthly\s+)?returns?\b|"
                          r"\barithmetic\s+returns?\b", re.I)
LOG_WORDS = re.compile(r"\blog(?:arithmic)?[- ](?:daily[- ]|monthly[- ])?returns?\b|continuously[- ]compounded returns?",
                       re.I)
POSITIVE_DD = re.compile(r"drawdown[^.\n]{0,90}?\b(positive(?: magnitude| number| value| fraction)?|"
                         r"(?:as an? )?(?:absolute )?magnitude|non-?negative)\b|\b(positive)\b[^.\n]{0,40}drawdown",
                         re.I)
NEGATIVE_DD = re.compile(r"drawdown[^.\n]{0,90}?\b(negative(?: number| value| fraction)?|non-?positive|"
                         r"(?:≤|<=)\s*0)\b|\b(negative)\b[^.\n]{0,40}drawdown", re.I)
DDOF_WORDS = re.compile(r"ddof\s*=\s*([01])")
SAMPLE_WORDS = re.compile(r"\bsample\s+(?:std|standard deviation|variance)", re.I)
POPULATION_WORDS = re.compile(r"\bpopulation\s+(?:std|standard deviation|variance)", re.I)
ANNUAL_LINE = re.compile(r"annuali[sz]|ann_|annual", re.I)
PERIOD_NUMBER = re.compile(r"sqrt\(\s*(\d+)\s*\)|√\s*\(?\s*(\d+)|[×x*]\s*`?(\d+)`?(?![\d.])|\^\s*\(?\s*(\d+)\s*/|"
                           r"factor (?:is|of) `?(\d+)|multiply(?:ing)? by `?(\d+)|(\d+) trading days (?:per|a|in a) year",
                           re.I)
KEY_IN_LINE = re.compile(r"`([A-Za-z_][\w]*)`|^\s*[-*|]?\s*([A-Za-z_][\w]*)\s*(?:[:=|(])")
CONTINUOUS_WORDS = re.compile(r"continuous(?:ly)?[- ]compound(?:ed|ing)?|continuous compounding", re.I)
ANNUAL_WORDS = re.compile(r"annual(?:ly)?[- ]compound(?:ed|ing)|compounded annually", re.I)
SEMI_WORDS = re.compile(r"semi-?annual(?:ly)?[- ]compound(?:ed|ing)|compounded semi-?annually", re.I)
SIMPLE_RATE_WORDS = re.compile(r"simpl[ey](?:[- ]compound(?:ed|ing)|[- ]interest)", re.I)
EXACT_WORDS = re.compile(r"\bbootstrap\w*|\bstrip(?:ping|ped|s)?\b|exact(?:ly)? (?:reprice|fit|match|recover)\w*|"
                         r"\breprice[sd]? (?:each|every|all|the input|the market|the quoted)|"
                         r"makes? this equal to the input|correct bootstrap|reprices? (?:each|every) ", re.I)
FIT_WORDS = re.compile(r"least[- ]squares|minimi[sz]|best[- ]fit|nelson|svensson|regression|maximum likelihood|"
                       r"log-?likelihood|\bsmooth", re.I)
TOLERANCE_WORDS = re.compile(r"(?:residual|error|reprice|repricing|differ)\w*[^.\n]{0,60}?(?:<|≤|<=|within|below|less "
                             r"than|under)\s*\$?(\d+(?:\.\d+)?(?:e-?\d+)?)\s*(bps?\b|basis points?)?", re.I)
TRADING_WORDS = re.compile(r"last trading day|trading[- ]day month[- ]end|month[- ]end trading day|"
                           r"last (?:available|observed) (?:trading )?(?:day|date|observation|price)|"
                           r"observed (?:trading )?dates", re.I)
CALENDAR_WORDS = re.compile(r"calendar month[- ]end(?:s| dates?| labels?)?|last calendar day|label(?:led|ed)? (?:by|with|"
                            r"as) (?:the )?calendar", re.I)
ISO = r"\**`?(\d{4}-\d{2}-\d{2})`?\**"
FROM_WORDS = re.compile(r"\b(?:from|starting(?: on| at| from)?|beginning(?: on| at)?|on or after|since)\s+" + ISO, re.I)
TO_WORDS = re.compile(r"\b(?:to|through|thru|until|till|ending(?: on| at)?|on or before|up to and including)\s+" + ISO,
                      re.I)
WINDOW_CONTEXT = re.compile(r"sample|period|window|backtest|data|observations?|returns|rows|trading|evaluation|"
                            r"history|range", re.I)
SUM_ONE = re.compile(r"(?:weights?|allocations?)[^.\n]{0,60}?sum(?:s|ming)? to (?:one|1(?:\.0)?\b|100\s?%)|"
                     r"fully[- ]invested", re.I)
SUM_ZERO = re.compile(r"dollar[- ]neutral|(?:weights?)[^.\n]{0,60}?sum(?:s|ming)? to (?:zero|0)\b|zero[- ]net "
                      r"(?:exposure|investment)", re.I)
GROSS_ONE = re.compile(r"gross (?:exposure|leverage)[^.\n]{0,30}?(?:of|=|equals?(?: to)?|is) (?:one|1(?:\.0)?\b|100\s?%)|"
                       r"sum of (?:the )?absolute (?:values of the )?weights[^.\n]{0,20}?(?:is|=|equals?) (?:one|1)\b",
                       re.I)
AFTER_WORDS = re.compile(r"on or after|next (?:trading|valid|available|business) (?:day|date)|first (?:valid |trading "
                         r"|available |business )?(?:\w+ )?(?:day|date) (?:on or )?after", re.I)
BEFORE_WORDS = re.compile(r"on or before|previous (?:trading|valid|available|business) (?:day|date)|last (?:valid |trading "
                          r"|available |business )?(?:\w+ )?(?:day|date) (?:on or )?before", re.I)
CAPITAL_WORDS = re.compile(r"(?:initial|starting) (?:capital|nav|wealth|value|portfolio value|equity)\s*(?:of|=|is|:)\s*"
                           r"\$?`?(\d[\d,]*(?:\.\d+)?)|start(?:s|ing)? (?:at|with|from) (?:an? )?(?:nav|wealth|value|"
                           r"capital) of \$?`?(\d[\d,]*(?:\.\d+)?)", re.I)


# ------------------------------------------------------------------------------------------------ the task's words


@dataclass(frozen=True)
class Words:
    """What the instruction's words pin for the checks; each pin keeps the words that pin it, None where unpinned."""
    returns: tuple[str, str] | None             # ("simple" or "log", the words)
    drawdown: tuple[int, str] | None            # (+1 or -1, the words)
    periods: dict                               # a key's stated periods a year, from its formula's line
    ddof: dict                                  # a key's stated ddof, from its formula's line
    periods_any: int | None                     # periods a year the text states for every key
    ddof_any: int | None                        # the ddof the text states, where it states one alone
    compounding: tuple[str, str] | None         # ("continuous", "annual", "semi", "simple"; the words)
    exact: str | None                           # the words asking a curve bootstrapped to its quotes
    tolerance: tuple[float, bool] | None        # the stated repricing tolerance, and whether in basis points
    trading: str | None                         # the words saying trading days
    calendar: str | None                        # the words saying calendar labels
    window: tuple[datetime | None, datetime | None, str]    # the earliest stated start, the latest stated end, words
    sums: tuple[tuple[str, float, str], ...]    # ("net" or "gross", the target, the words)
    capital: tuple[float, str] | None           # the stated initial capital, and the words


def pinned(found: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The one convention a text pins: its words where every match names the same one, else None."""
    kinds = {kind for kind, _ in found}
    return found[0] if len(kinds) == 1 else None


@lru_cache(maxsize=64)
def words(text: str) -> Words:
    """The instruction read once for the conventions the checks compare against."""
    returns = pinned([("simple", match.group(0)) for match in SIMPLE_WORDS.finditer(text)]
                     + [("log", match.group(0)) for match in LOG_WORDS.finditer(text)])
    drawdown = pinned([("1", match.group(0)) for match in POSITIVE_DD.finditer(text)]
                      + [("-1", match.group(0)) for match in NEGATIVE_DD.finditer(text)])
    periods, ddof = {}, {}
    for line in text.splitlines():
        if not ANNUAL_LINE.search(line):
            continue
        keys = [a or b for a, b in KEY_IN_LINE.findall(line)]
        found = [int(next(group for group in match.groups() if group)) for match in PERIOD_NUMBER.finditer(line)]
        found = [each for each in found if each in PERIODS_A_YEAR]
        stated = DDOF_WORDS.search(line)
        for key in keys:
            if found:
                periods.setdefault(key.lower(), found[0])
            if stated:
                ddof.setdefault(key.lower(), int(stated.group(1)))
    every = [int(next(group for group in match.groups() if group)) for match in PERIOD_NUMBER.finditer(text)
             if re.search(r"factor|trading days|multiply", match.group(0), re.I)]
    every = [each for each in every if each in PERIODS_A_YEAR]
    stated = {int(each) for each in DDOF_WORDS.findall(text)} | ({1} if SAMPLE_WORDS.search(text) else set()) \
        | ({0} if POPULATION_WORDS.search(text) else set())
    compounding = pinned([(kind, match.group(0)) for kind, pattern in (
        ("continuous", CONTINUOUS_WORDS), ("annual", ANNUAL_WORDS), ("semi", SEMI_WORDS),
        ("simple", SIMPLE_RATE_WORDS)) for match in pattern.finditer(text)])
    exact = None if FIT_WORDS.search(text) else next((match.group(0) for match in EXACT_WORDS.finditer(text)), None)
    tolerance = next(((float(match.group(1)), bool(match.group(2))) for match in TOLERANCE_WORDS.finditer(text)), None)
    starts, ends, said = [], [], []
    for sentence in re.split(r"(?<=[.;\n])\s+", text):
        if not WINDOW_CONTEXT.search(sentence):
            continue
        for pattern, into in ((FROM_WORDS, starts), (TO_WORDS, ends)):
            for match in pattern.finditer(sentence):
                day = witness.moment(match.group(1))
                if day is not None:
                    into.append(day)
                    said.append(match.group(0).replace("*", "").replace("`", ""))
    sums = tuple([("net", 1.0, match.group(0)) for match in SUM_ONE.finditer(text)][:1]
                 + [("net", 0.0, match.group(0)) for match in SUM_ZERO.finditer(text)][:1]
                 + [("gross", 1.0, match.group(0)) for match in GROSS_ONE.finditer(text)][:1])
    if sum(kind == "net" for kind, _, _ in sums) > 1:
        sums = tuple(each for each in sums if each[0] != "net")     # both net targets named: neither pinned
    capital = next(((float((match.group(1) or match.group(2)).replace(",", "")), match.group(0))
                    for match in CAPITAL_WORDS.finditer(text)), None)
    return Words(returns=returns, drawdown=(int(drawdown[0]), drawdown[1]) if drawdown else None,
                 periods=periods, ddof=ddof, periods_any=every[0] if len(set(every)) == 1 else None,
                 ddof_any=next(iter(stated)) if len(stated) == 1 else None, compounding=compounding, exact=exact,
                 tolerance=tolerance, trading=(TRADING_WORDS.search(text) or [None])[0],
                 calendar=(CALENDAR_WORDS.search(text) or [None])[0],
                 window=(min(starts) if starts else None, max(ends) if ends else None, "; ".join(said[:2])),
                 sums=sums, capital=capital)


def quoted(words_: str | None) -> str:
    """The task's words for a convention, as a finding says them; `UNPINNED` where none."""
    if not words_:
        return UNPINNED
    text = re.sub(r"\s+", " ", words_).strip()
    return f'the task says "{text[:80]}"'


# ------------------------------------------------------------------------------------------------ reading the output


def frames(out: Files) -> list[Table]:
    """The output's tables, and its JSON lists of records and maps of records (two or more records sharing a numeric
    key), each as a `Table` named `file:key`, its first column the record's key for a map."""
    found = list(out.tables)
    for name, parsed in out.jsons:
        for path, value in json_records(parsed):
            if isinstance(value, list):
                header = list(dict.fromkeys(key for each in value for key in each))
                rows = [[each.get(key, "") for key in header] for each in value]
            else:
                inner = {key: each for key, each in value.items() if isinstance(each, dict)}
                header = ["key", *dict.fromkeys(key for each in inner.values() for key in each)]
                rows = [[key, *(each.get(column, "") for column in header[1:])] for key, each in inner.items()]
            cells = [["" if row[at] is None else str(row[at]) for row in rows] for at in range(len(header))]
            read = Table(f"{name}:{path}" if path else name, header, cells)
            for at in sorted(range(len(header)), key=lambda k: not witness.conventions.DATE_NAME.search(header[k])):
                if witness.conventions.dates_in(cells[at], bool(witness.conventions.DATE_NAME.search(header[at]))):
                    read.date_column, read.row_dates = at, [witness.moment(cell) for cell in cells[at]]
                    break
            found.append(read)
    return found


def json_records(value, path: str = "") -> list[tuple[str, object]]:
    """A JSON value's lists of records (two or more dicts) and maps of records (two or more dict values sharing a
    numeric key), with their key paths."""
    found = []
    if isinstance(value, list) and len(value) >= 2 and all(isinstance(each, dict) for each in value):
        found.append((path, value))
    elif isinstance(value, dict):
        inner = [each for each in value.values() if isinstance(each, dict)]
        shared = set.intersection(*(set(key for key, v in each.items() if isinstance(v, (int, float))
                                        and not isinstance(v, bool)) for each in inner)) if len(inner) >= 2 else set()
        if shared:
            found.append((path, value))
        for key, each in value.items():
            found.extend(json_records(each, f"{path}.{key}" if path else str(key)))
    return found


def values(read: Table, column: int) -> list[float | None]:
    """A column's cells as numbers, None where a cell holds none or a non-finite one."""
    found = []
    for cell in read.cells[column]:
        value = number(cell)
        found.append(value if value is not None and math.isfinite(value) else None)
    return found


def numeric(read: Table, column: int) -> bool:
    return read.numeric(column) and column != read.date_column


@dataclass(frozen=True)
class Scalar:
    """A reported number: its file, its key path, its key, its value, and the names it is reported under."""
    file: str
    path: str
    key: str
    value: float
    owner: frozenset


def tokens(name: str) -> frozenset:
    return frozenset(part for part in re.split(r"[^a-z0-9]+", str(name).lower()) if part)


OWNER_NOISE = frozenset({"results", "result", "metrics", "summary", "stats", "statistics", "performance", "json",
                         "csv", "output", "outputs", "value", "values", "portfolio", "daily", "monthly"})


def scalars(out: Files) -> list[Scalar]:
    """Every number the output reports alone: a JSON number (its parents' names its owner) and a one-row table's cell
    (the table's name its owner)."""
    found = []
    for name, path, value in out.pairs():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        parts = [part for part in re.sub(r"\[\d+\]", "", path).split(".") if part]
        key = leaf(path)
        owner = frozenset().union(*(tokens(part) for part in parts[:-1] if part != "value")) - OWNER_NOISE
        found.append(Scalar(name, path, key, float(value), owner))
    for read in out.tables:
        if read.rows == 1:
            for k, column in enumerate(read.header):
                value = number(read.cell(k, 0))
                if value is not None and math.isfinite(value):
                    found.append(Scalar(read.name, column, column, value, frozenset()))
    return found


@dataclass(frozen=True)
class Path_:
    """A wealth path the output holds or its returns give: its file, the name it is said by, its values, its dates,
    the column's tokens it is tied by, and how it was read (`as given`, `one plus`, `chained simple`, `chained log`)."""
    file: str
    name: str
    values: tuple[float, ...]
    dates: tuple
    owner: frozenset
    how: str


def dated_series(read: Table) -> bool:
    """Whether a table's rows run as a dated series (its date column names a date, month, period or day, not a trade's
    entry or exit, an event or a maturity): its return columns are a portfolio's returns, not a list of trades'."""
    return read.date_column is not None and bool(SERIES_DATE.search(read.header[read.date_column])) \
        and not NOT_SERIES_DATE.search(read.header[read.date_column])


def scaled(name: str, series: list) -> list:
    """A return column's values as fractions: one named in percent (`pct`, `percent`) over a hundred."""
    if re.search(r"pct|percent", name, re.I):
        return [None if value is None else value / 100 for value in series]
    return series


def paths(out: Files, frames_: list[Table]) -> list[Path_]:
    """Every path a table holds (a NAV, wealth, equity or cumulative column, positive, two rows or more; a cumulative
    return column read as given and as one plus it) and every return column of a dated series chained (simple and
    log)."""
    found = []
    for read in frames_:
        for k, column in enumerate(read.header):
            if not numeric(read, k):
                continue
            series = scaled(column, values(read, k))
            rows = [at for at, value in enumerate(series) if value is not None]
            if len(rows) < 2:
                continue
            kept = tuple(series[at] for at in rows)
            dates = tuple(read.row_dates[at] if read.row_dates else None for at in rows)
            owner = tokens(column) | tokens(Path(read.name.split(":")[0]).stem)
            if PATH_NAME.search(column):
                if all(value > 0 for value in kept):
                    found.append(Path_(read.name, column, kept, dates, owner, "as given"))
                if CUM_NAME.search(column) and all(value > -1 for value in kept):
                    found.append(Path_(read.name, column, tuple(1 + value for value in kept), dates, owner,
                                       "one plus"))
            elif returns_column(column) and dated_series(read) and all(value > -1 for value in kept):
                for how in ("simple", "log"):
                    found.append(Path_(read.name, column, chained(kept, how), dates, owner, f"chained {how}"))
    return found


def returns_column(name: str) -> bool:
    return bool(RETURN_NAME.search(name)) and not NOT_RETURN.search(name) and not PATH_NAME.search(name)


def chained(series, how: str) -> tuple[float, ...]:
    """A return series as the wealth path it gives from one before its first row: simple returns compounded, or log
    returns summed and exponentiated."""
    level, path = 1.0, []
    for value in series:
        level = level * (1 + value) if how == "simple" else level * math.exp(value)
        path.append(level)
    return tuple(path)


def digits_slack(cells) -> float:
    """What a column's printed digits allow: twice its last printed place."""
    return 2 * 10.0 ** -witness.digits([str(cell) for cell in cells])


def near(left: float, right: float, slack: float = 0.0) -> bool:
    return math.isclose(left, right, rel_tol=RELATIVE, abs_tol=1e-9 + slack)


def said_rows(read: Table, rows: list[int]) -> str:
    """Rows as a finding names them: their dates where the table holds them, else their key cell, else their numbers."""
    shown = []
    labels = {str(read.cell(read.date_column, row)) if read.date_column is not None else "" for row in rows[:NAMED]}
    plain_rows = len(rows) > 1 and len(labels) == 1 and read.date_column is not None
    for row in rows[:NAMED]:
        if plain_rows:
            shown.append(f"row {row + 1}")
            continue
        if read.row_dates and row < len(read.row_dates) and read.row_dates[row] is not None:
            shown.append(read.row_dates[row].strftime("%Y-%m-%d"))
        elif (labels := [str(read.cell(k, row)).strip() for k in range(min(2, len(read.header)))
                         if not read.numeric(k) and str(read.cell(k, row)).strip()]):
            shown.append(f"`{' '.join(labels)[:40]}`")
        else:
            shown.append(f"row {row + 1}")
    return ", ".join(shown) + (" among others" if len(rows) > NAMED else "")


def where(file: str) -> str:
    """A frame's name as a finding says it: `file`, or `file`'s `key`."""
    name, _, key = file.partition(":")
    return f"`{name}`'s `{key}`" if key else f"`{name}`"


# ------------------------------------------------------------------------------------------------ the replay


def chain(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """Where a table holds a path (a NAV, wealth, equity or cumulative-return column) beside a return column, the path
    chains the returns: each row's step over the row before is its return, simple or log (both tried; the task's
    words pin one where they name only one). A path whose step matches no return column of its table, under either
    convention, on `MISSES` rows or more is said with its first rows, and as the row before's return where the step
    matches that; one that chains only under the convention the task does not name is said with the task's words."""
    found = []
    pin = words(task_.instruction).returns
    for read in frames_:
        columns = [k for k in range(len(read.header)) if numeric(read, k)]
        returns = [k for k in columns if returns_column(read.header[k])]
        if not returns:
            continue
        for k in columns:
            name = read.header[k]
            if not PATH_NAME.search(name):
                continue
            path = values(read, k)
            if sum(value is not None for value in path) < 3:
                continue
            forms = [("as given", path)]
            if CUM_NAME.search(name):
                forms.append(("one plus", [None if value is None else 1 + value for value in path]))
            best = None
            for form, levels in forms:
                if any(value is not None and value <= 0 for value in levels):
                    continue
                for r in returns:
                    series = scaled(read.header[r], values(read, r))
                    slack = digits_slack(read.cells[k]) / max(min(abs(v) for v in levels if v is not None), 1e-9) \
                        + digits_slack(read.cells[r])
                    for how in ("simple", "log"):
                        for lag in (0, 1):
                            misses = misfits(levels, series, how, lag, slack)
                            if misses is None:
                                continue
                            score = (len(misses), lag, how != (pin[0] if pin else how))
                            if best is None or score < best[0]:
                                best = (score, misses, how, lag, read.header[r])
            if best is None:
                continue
            (count, lag, off_pin), misses, how, _, column = best
            if count >= MISSES:
                found.append((f"{name}", f"{where(read.name)}: `{name}` does not chain `{column}`, simple or log "
                              f"({quoted(pin[1] if pin else None)}): its step over the row before differs from the "
                              f"return on {said_rows(read, misses)}."))
            elif lag:
                found.append((f"{name}", f"{where(read.name)}: `{name}` chains `{column}` one row late: each step "
                              f"matches the row before's return, not its own row's."))
            elif off_pin and pin:
                found.append((f"{name}", f"{where(read.name)}: `{name}` chains `{column}` as {how} returns where "
                              f"{quoted(pin[1])}."))
    return found


def misfits(levels, series, how: str, lag: int, slack: float) -> list[int] | None:
    """The rows whose step over the row before misses its return (`lag` 1: the row before's); None where fewer than
    three rows can be compared."""
    misses, compared = [], 0
    for at in range(1 + lag, len(levels)):
        before, now, move = levels[at - 1], levels[at], series[at - lag] if at - lag < len(series) else None
        if before is None or now is None or move is None or before <= 0 or now <= 0:
            continue
        compared += 1
        step = now / before - 1 if how == "simple" else math.log(now / before)
        if abs(step - move) > slack + 1e-9 + RELATIVE * abs(move):
            misses.append(at)
    return misses if compared >= 3 else None


BENCHMARK = frozenset({"benchmark", "bench", "market", "mkt", "equal", "ew", "index", "passive", "buy", "hold", "bh",
                       "spy", "spx"})


def tied(owner: frozenset, candidates: list, of=lambda each: (each.file, each.owner)) -> list:
    """The candidates (paths, or return series) a reported number belongs to: those sharing a name with its owner
    where any does; else, where its owner names no benchmark, every candidate where they all come from one table (a
    path and its own returns among them); [] where it cannot be told (a benchmark's number beside a strategy's path,
    or paths of several tables)."""
    if owner:
        shared = [each for each in candidates if of(each)[1] & owner]
        if shared:
            return shared
        if owner & BENCHMARK:
            return []
    tables = {of(each)[0] for each in candidates}
    return candidates if len(tables) == 1 else []


def total(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A reported total (or cumulative) return is its path's end over its start less one, the start the path's first
    value or the one before it (a path that opens after its first return), as a fraction or in percent. Said where it
    matches no path the output holds or its returns give, and the path it belongs to can be told (`tied`)."""
    found = []
    candidates = paths(out, frames_)
    for scalar in scalars(out):
        if not TOTAL_KEY.search(scalar.key):
            continue
        mine = tied(scalar.owner, candidates)
        if not mine:
            continue
        options = []
        for each in mine:
            first, last = each.values[0], each.values[-1]
            options += [last / first - 1, last - 1 if each.how != "as given" or abs(first - 1) < 0.5 else last / first - 1]
        if any(near(scalar.value, option, 1e-6) or near(scalar.value, 100 * option, 1e-4) for option in options):
            continue
        names = ", ".join(dict.fromkeys(f"`{each.name}`" for each in mine))
        found.append((scalar.path, f"`{scalar.file}`'s `{scalar.path}` is not the growth its path gives ({names}: the "
                      f"end over the start less one)."))
    return found


def drawdowns(levels) -> list[float]:
    """A path's maximum drawdown, three ways: as a fraction of its running peak, with the start before its first row
    as a peak, and in log terms."""
    found = []
    for opening in (None, 1.0):
        peak, worst, worst_log = opening or levels[0], 0.0, 0.0
        for value in levels:
            peak = max(peak, value)
            worst = max(worst, 1 - value / peak)
            worst_log = max(worst_log, math.log(peak / value))
        found += [worst, worst_log]
    return found


def drawdown(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A reported maximum drawdown has the sign the task's words pin (a positive magnitude, or a negative number), and
    its magnitude is the drawdown its path gives (`drawdowns`: of the running peak, from the start, or in log terms;
    as a fraction or in percent). The magnitude is said where it matches no path the output holds or its returns
    give, and the path it belongs to can be told (`tied`)."""
    found = []
    pin = words(task_.instruction).drawdown
    candidates = paths(out, frames_)
    for scalar in scalars(out):
        if not DRAWDOWN_KEY.search(scalar.key):
            continue
        if pin and scalar.value and (scalar.value > 0) != (pin[0] > 0):
            found.append((scalar.path + ":sign", f"`{scalar.file}`'s `{scalar.path}` is "
                          f"{'negative' if scalar.value < 0 else 'positive'} where {quoted(pin[1])}."))
        mine = tied(scalar.owner, candidates)
        if not mine:
            continue
        size = abs(scalar.value)
        options = [each_dd for each in mine for each_dd in drawdowns(each.values)]
        if any(near(size, option, 1e-6) or near(size, 100 * option, 1e-4) for option in options):
            continue
        names = ", ".join(dict.fromkeys(f"`{each.name}`" for each in mine))
        found.append((scalar.path, f"`{scalar.file}`'s `{scalar.path}` is not the drawdown {names} gives (the "
                      f"largest fall from a running peak, as a fraction of the peak)."))
    return found


def returns_series(frames_: list[Table]) -> list[tuple[str, str, list[float], frozenset, float | None]]:
    """Every return series the output holds or its paths give: (file, name, returns, tokens, the median step of its
    dates in days, None where undated); each also without its first row and without its leading zeros."""
    found = []
    for read in frames_:
        steps = None
        days = [day for day in read.row_dates if day is not None] if read.row_dates else []
        if len(days) >= 3:
            steps = statistics.median((later - earlier).days for earlier, later in zip(days, days[1:]))
        for k, column in enumerate(read.header):
            if not numeric(read, k):
                continue
            series = [value for value in scaled(column, values(read, k)) if value is not None]
            if len(series) < 3:
                continue
            owner = tokens(column) | tokens(Path(read.name.split(":")[0]).stem)
            if returns_column(column) and dated_series(read):
                found.append((read.name, column, series, owner, steps))
            elif PATH_NAME.search(column) and all(value > 0 for value in series):
                for how in ("simple", "log"):
                    moves = [now / before - 1 if how == "simple" else math.log(now / before)
                             for before, now in zip(series, series[1:])]
                    found.append((read.name, column, moves, owner, steps))
                    found.append((read.name, column, [series[0] - 1, *moves] if how == "simple"
                                  else [math.log(series[0]), *moves], owner, steps))
    expanded = []
    for file, name, series, owner, steps in found:
        expanded.append((file, name, series, owner, steps))
        expanded.append((file, name, series[1:], owner, steps))
        lead = next((at for at, value in enumerate(series) if value != 0), 0)
        if lead:
            expanded.append((file, name, series[lead:], owner, steps))
    return expanded


def annual(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A reported annualised volatility is its series' standard deviation times the root of the periods a year, and a
    reported annualised return its mean times the periods (or the series compounded to a year), at the periods and
    ddof the task's words state for that key (else for every key; else the periods its dates give, 252 or 365 a day,
    52 a week, 12 a month, and both ddof tried). Said where no return series the output holds or its paths give
    agrees, and the series it belongs to can be told (`tied`)."""
    found = []
    read_words = words(task_.instruction)
    series = returns_series(frames_)
    for scalar in scalars(out):
        vol, ret = bool(VOL_KEY.search(scalar.key)), bool(ANNRET_KEY.search(scalar.key))
        if not (vol or ret):
            continue
        mine = tied(scalar.owner, series, of=lambda each: (each[0], each[3]))
        if not mine:
            continue
        key = scalar.key.lower()
        stated_periods = read_words.periods.get(key) or read_words.periods_any
        stated_ddof = read_words.ddof.get(key, read_words.ddof_any)
        options = []
        for _, _, values_, _, steps in mine:
            if len(values_) < 3:
                continue
            periods = (stated_periods,) if stated_periods else periods_of(steps)
            for p in periods:
                if vol:
                    for d in ((stated_ddof,) if stated_ddof is not None else (0, 1)):
                        if len(values_) > d:
                            sd = statistics.pstdev(values_) if d == 0 else statistics.stdev(values_)
                            options.append(sd * math.sqrt(p))
                else:
                    mean = statistics.fmean(values_)
                    options.append(mean * p)
                    growth = math.prod(1 + value for value in values_) if all(v > -1 for v in values_) else None
                    if growth and growth > 0:
                        options.append(growth ** (p / len(values_)) - 1)
                        options.append(math.exp(statistics.fmean(math.log1p(v) for v in values_) * p) - 1)
                    options.append(math.exp(mean * p) - 1)
        if not options or any(near(scalar.value, option, 1e-6) or near(scalar.value, 100 * option, 1e-4)
                              for option in options):
            continue
        names = ", ".join(dict.fromkeys(f"`{name}`" for _, name, *_ in mine))
        what = "volatility" if vol else "return"
        convention = []
        if stated_periods:
            convention.append(f"{stated_periods} periods a year")
        if vol:
            convention.append(f"ddof {stated_ddof}" if stated_ddof is not None else f"ddof 0 or 1, {UNPINNED}")
        found.append((scalar.path, f"`{scalar.file}`'s `{scalar.path}` is not the annualised {what} {names} gives"
                      + (f" at the task's {' and '.join(convention)}" if convention else "") + "."))
    return found


def periods_of(steps: float | None) -> tuple[int, ...]:
    """The periods a year a series' median step in days gives: a day 252 or 365, a week 52, a month 12, a quarter 4."""
    if steps is None:
        return (252, 12)
    if steps <= DAILY_STEP:
        return (252, 365)
    if steps <= 10:
        return (52,)
    if steps <= MONTHLY_STEP:
        return (12,)
    return (4,) if steps <= 100 else (1,)


def weights(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """Where the task's words state what a portfolio's weights sum to (fully invested, summing to one; dollar neutral,
    summing to zero; a gross exposure of one), each row of an output table of weights (a dated table named for weights,
    allocations, holdings or positions, or one whose columns name the input's assets) sums to it, a row of all zeros
    (no book yet) aside."""
    stated = words(task_.instruction).sums
    if not stated:
        return []
    found = []
    for read in out.tables:
        if read.date_column is None:
            continue
        matched = witness.assets(task_, read)
        if matched is not None:
            columns = matched[1]
        elif WEIGHT_TABLE.search(read.name):
            columns = [k for k in range(len(read.header)) if numeric(read, k) and not META_COLUMN.search(read.header[k])]
        else:
            continue
        if len(columns) < 2:
            continue
        slack = 1e-4 + digits_slack([cell for k in columns for cell in read.cells[k]]) * len(columns)
        for kind, target, said in stated:
            bad = []
            for row in range(read.rows):
                cells = [number(read.cell(k, row)) for k in columns]
                if any(cell is None or not math.isfinite(cell) for cell in cells) or not any(cells):
                    continue
                held = sum(cells) if kind == "net" else sum(abs(cell) for cell in cells)
                if abs(held - target) > slack:
                    bad.append(row)
            if bad:
                what = "sum" if kind == "net" else "absolute sum"
                found.append((f"{read.name}:{kind}", f"`{read.name}`: the weights' {what} on {said_rows(read, bad)} "
                              f"is not what {quoted(said)} asks."))
    return found


# ------------------------------------------------------------------------------------------------ the repricing


RATE_FORMS = {
    "continuous": lambda p, t: -math.log(p) / t,
    "annual": lambda p, t: p ** (-1 / t) - 1,
    "semi": lambda p, t: 2 * (p ** (-1 / (2 * t)) - 1),
    "quarterly": lambda p, t: 4 * (p ** (-1 / (4 * t)) - 1),
    "monthly": lambda p, t: 12 * (p ** (-1 / (12 * t)) - 1),
    "simple": lambda p, t: (1 / p - 1) / t,
    "unscaled": lambda p, t: -math.log(p),      # -ln P, which a task may name a zero rate too
}


def rate_formula(text: str, column: str, factor: str, time: str) -> tuple[str, str] | None:
    """A column explicitly defined as -log(factor), with or without division by its time."""
    match = re.search(r"(?<![\w])" + re.escape(column) + r"`?\s*[:=]\s*`?-(?:np\.|math\.)?(?:log|ln)\(\s*"
                      + re.escape(factor) + r"\s*\)(?:\s*/\s*([A-Za-z_]\w*))?", text)
    if match is None or (match.group(1) is not None and match.group(1) != time):
        return None
    return ("continuous" if match.group(1) else "unscaled", match.group(0))


def discount(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """Where a table holds times in years beside a discount factor (or zero-coupon price) and a zero rate or yield,
    the rate is the one the factor gives at its time: continuously, annually, semi-annually, quarterly or monthly
    compounded, simple, or -ln P (all tried), in decimals, percent or basis points. The compounding the column's name
    or the task's words pin is the one compared; one that matches only another compounding is said with it, one that
    matches none said as such. A market quote's column, a forward, par, swap or spread is not read."""
    found = []
    pin = words(task_.instruction).compounding
    for read in frames_:
        time_at = next((k for k, column in enumerate(read.header) if TIME_COLUMN.match(column) and numeric(read, k)),
                       None)
        if time_at is None:
            continue
        times = values(read, time_at)
        factors = [k for k, column in enumerate(read.header) if DF_COLUMN.search(column) and numeric(read, k)
                   and k != time_at and all(v is None or 0 < v <= 1.5 for v in values(read, k))]
        rates = [k for k, column in enumerate(read.header) if RATE_COLUMN.search(column) and not NOT_RATE.search(column)
                 and numeric(read, k) and k != time_at and k not in factors]
        if len(factors) != 1 or not rates:
            continue
        factor = values(read, factors[0])
        rows = [row for row in range(read.rows) if times[row] and times[row] > 0 and factor[row]]
        if len(rows) < 2:
            continue
        for k in rates:
            rate = values(read, k)
            column = read.header[k]
            formula = rate_formula(task_.instruction, column, read.header[factors[0]], read.header[time_at])
            named = next((form for form, word in (("continuous", "cont"), ("annual", "annual"), ("semi", "semi"),
                                                  ("simple", "simple")) if word in column.lower()), None)
            slack = digits_slack(read.cells[k])
            fits = [form for form, rule in RATE_FORMS.items()
                    if any(all(rate[row] is not None and near(rate[row], scale * rule(factor[row], times[row]), slack)
                               for row in rows) for scale in (1, 100, 10_000))]
            wanted = formula[0] if formula else named or (pin[0] if pin else None)
            if wanted in fits or (wanted is None and fits):
                continue
            words_ = quoted(formula[1]) if formula else f'its name says "{column}"' if named else quoted(pin[1] if pin else None)
            if fits:
                found.append((column, f"{where(read.name)}: `{column}` is the {fits[0]} rate `{read.header[factors[0]]}` "
                              f"gives at `{read.header[time_at]}`, not the {wanted} one ({words_})."))
            else:
                found.append((column, f"{where(read.name)}: `{column}` is not the rate `{read.header[factors[0]]}` "
                              f"gives at `{read.header[time_at]}` under any compounding tried (continuous, annual, "
                              f"semi-annual, quarterly, monthly, simple, -ln P; {words_})."))
    return found


def stem(name: str, pattern: re.Pattern) -> str:
    return re.sub(r"_+", "_", pattern.sub("_", name.lower())).strip("_")


def quotes(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """Where the task's words ask a curve bootstrapped (or stripped, or repricing its quotes exactly) and fit no model
    by least squares or likelihood, each pair of an output's market and model columns of one quantity (`market_X`
    beside `model_X`: market, input, quoted, observed or target against model, fitted, repriced, calibrated,
    bootstrapped or computed) agree on every row, to the tolerance the task states where it states one."""
    read_words = words(task_.instruction)
    if not read_words.exact:
        return []
    found = []
    for read in frames_:
        markets = {stem(column, MARKET): k for k, column in enumerate(read.header) if MARKET.search(column)
                   and numeric(read, k)}
        models = {stem(column, MODEL): k for k, column in enumerate(read.header) if MODEL.search(column)
                  and not MARKET.search(column) and numeric(read, k)}
        for name, m in markets.items():
            if name not in models:
                continue
            k = models[name]
            market, model = values(read, m), values(read, k)
            slack = max(digits_slack(read.cells[m]), digits_slack(read.cells[k]))
            tolerance = read_words.tolerance
            limit = None
            if tolerance is not None:
                limit = tolerance[0] / 10_000 if tolerance[1] and "bp" not in read.header[k].lower() else tolerance[0]
            bad = [row for row in range(read.rows) if market[row] is not None and model[row] is not None
                   and (abs(model[row] - market[row]) > max(limit, slack) if limit is not None
                        else not near(model[row], market[row], slack))]
            if bad:
                found.append((f"{read.name}:{read.header[k]}", f"{where(read.name)}: `{read.header[k]}` does not "
                              f"reprice `{read.header[m]}` on {said_rows(read, bad)}, where {quoted(read_words.exact)}"
                              f" (the curve is to reprice its quotes)."))
    return found


def tenor(label: str) -> float | None:
    """A tenor's label in years: `6M`, `1Y`, `10y`, `2W`, `30D`, or a number."""
    match = TENOR.search(str(label).strip())
    if not match:
        return None
    if match.group(3):
        return float(match.group(3))
    amount, unit = float(match.group(1)), match.group(2).lower()
    return amount * {"y": 1.0, "m": 1 / 12, "w": 7 / 365.25, "d": 1 / 365.25}[unit]


def survival_rows(out: Files, frames_: list[Table]) -> list[tuple[str, list[tuple[float, float]], str]]:
    """The output's survival curves as (where, [(time, probability)], name): a table's time and survival columns, and
    a JSON map of tenors to survival probabilities."""
    found = []
    for read in frames_:
        time_at = next((k for k, column in enumerate(read.header) if (TIME_COLUMN.match(column) or column.lower() in
                        ("tenor", "t_years")) and numeric(read, k)), None)
        if time_at is None:
            continue
        for k, column in enumerate(read.header):
            if k != time_at and SURVIVAL_NAME.search(column) and numeric(read, k):
                times, probabilities = values(read, time_at), values(read, k)
                pairs = [(t, q) for t, q in zip(times, probabilities) if t is not None and q is not None and 0 < q <= 1]
                if len(pairs) >= 2:
                    found.append((read.name, pairs, column))
    for name, parsed in out.jsons:
        for path, value in maps(parsed):
            if SURVIVAL_NAME.search(leaf(path)) or "surviv" in path.lower():
                pairs = [(tenor(key), float(each)) for key, each in value.items() if tenor(key) is not None
                         and isinstance(each, (int, float)) and not isinstance(each, bool) and 0 < each <= 1]
                if len(pairs) >= 2:
                    found.append((name, pairs, leaf(path)))
    return found


def maps(value, path: str = "") -> list[tuple[str, dict]]:
    """A JSON value's maps of two or more numbers, with their key paths."""
    found = []
    if isinstance(value, dict):
        numbers = {key: each for key, each in value.items() if isinstance(each, (int, float))
                   and not isinstance(each, bool)}
        if len(numbers) >= 2:
            found.append((path, numbers))
        for key, each in value.items():
            found.extend(maps(each, f"{path}.{key}" if path else str(key)))
    return found


def hazard_curves(out: Files, frames_: list[Table]) -> list[tuple[str, list[tuple[float, float, float]], str]]:
    """The output's piecewise-constant hazards as (where, [(start, end, rate)], name): a JSON map of interval keys
    (`h_0_1`) or of tenors (each hazard on the interval from the tenor before), and a table's time and hazard
    columns."""
    found = []
    for name, parsed in out.jsons:
        for path, value in maps(parsed):
            intervals = [(float(match.group(1)), float(match.group(2)), float(each)) for key, each in value.items()
                         if (match := INTERVAL_KEY.match(key))]
            if len(intervals) >= 2:
                found.append((name, sorted(intervals), leaf(path) or name))
            elif HAZARD_NAME.search(leaf(path)) or "hazard" in path.lower():
                ends = sorted((tenor(key), float(each)) for key, each in value.items() if tenor(key) is not None)
                if len(ends) >= 2:
                    starts = [0.0, *(end for end, _ in ends[:-1])]
                    found.append((name, [(start, end, rate) for start, (end, rate) in zip(starts, ends)], leaf(path)))
    for read in frames_:
        time_at = next((k for k, column in enumerate(read.header) if (TIME_COLUMN.match(column) or column.lower() in
                        ("tenor", "t_years")) and numeric(read, k)), None)
        if time_at is None:
            continue
        for k, column in enumerate(read.header):
            if k != time_at and HAZARD_NAME.search(column) and not SURVIVAL_NAME.search(column) and numeric(read, k):
                pairs = sorted((t, h) for t, h in zip(values(read, time_at), values(read, k))
                               if t is not None and h is not None)
                if len(pairs) >= 2:
                    starts = [0.0, *(t for t, _ in pairs[:-1])]
                    found.append((read.name, [(start, end, rate) for start, (end, rate) in zip(starts, pairs)], column))
    return found


def cumulative(intervals, t: float) -> float | None:
    """The hazard integrated from zero to `t` over piecewise-constant intervals; None past the last interval's end."""
    if t > intervals[-1][1] + 1e-9:
        return None
    total_ = 0.0
    for start, end, rate in intervals:
        if t <= start:
            break
        total_ += rate * (min(t, end) - start)
    return total_


def survival(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """Where the output holds piecewise-constant hazard rates and survival probabilities, each probability is the
    exponential of minus the hazard integrated to its time (`HAZARD_SLACK` on the cumulative hazard, a tenor's label
    against its day count): said with the times that miss, where a hazard curve and a survival curve of one file, or
    of the output's only pair, disagree."""
    found = []
    hazards, curves = hazard_curves(out, frames_), survival_rows(out, frames_)
    for h_where, intervals, h_name in hazards:
        for s_where, pairs, s_name in curves:
            if len(hazards) > 1 and len(curves) > 1 and h_where.split(":")[0] != s_where.split(":")[0]:
                continue
            bad = []
            for t, q in pairs:
                expected = cumulative(intervals, t)
                if expected is None or t <= 0:
                    continue
                if abs(-math.log(q) - expected) > HAZARD_SLACK * expected + 1e-5:
                    bad.append(t)
            if bad:
                hazards_said = where(h_where) + ("" if h_name in (h_where, h_where.split(":")[-1]) else f"'s `{h_name}`")
                found.append((f"{s_where}:{s_name}", f"{where(s_where)}: `{s_name}` is not the survival the hazards "
                              f"in {hazards_said} give, exp of minus the hazard integrated to each time, "
                              f"at times " + ", ".join(f"{t:g}" for t in bad[:NAMED])
                              + (" among others" if len(bad) > NAMED else "") + "."))
    return found


def option_inputs(task_: Task, out: Files) -> tuple[list[float], list[float], list[float]]:
    """The spot, rate and dividend yield candidates: those the instruction states (`r = 0.05`, `D = 0.01`) alone where
    it states them, else those the output and the inputs' parameters name (a dividend yield of zero among them)."""
    stated = {"spot": [], "rate": [], "dividend": []}
    for name, value in STATED_NUMBER.findall(task_.instruction):
        kind = "spot" if name.lower().startswith("s") else "rate" if name.lower() in ("r", "rf", "risk_free_rate") \
            else "dividend"
        stated[kind].append(float(value))
    spot, rate, dividend = [], [], [0.0]
    for _, path, value in [*out.pairs(), *task_.inputs.pairs()]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        key = leaf(path)
        if SPOT_KEY.match(key) and value > 0:
            spot.append(float(value))
        elif RATE_KEY.match(key) and -0.2 < value < 1:
            rate.append(float(value))
        elif DIVIDEND_KEY.match(key) and 0 <= value < 1:
            dividend.append(float(value))
    return (list(dict.fromkeys(stated["spot"] or spot)), list(dict.fromkeys(stated["rate"] or rate)),
            list(dict.fromkeys(stated["dividend"] or dividend)))


def pricer(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """Where an output table holds a strike, a time to expiry, an implied volatility and the option's price on a row,
    the library's Black-Scholes price (qf.bs_price) at that volatility reproduces the price (`repriced`: within
    `PRICE_SLACK` of it or `VOL_SLACK` in volatility) for some spot, rate and dividend yield: those the instruction
    states where it states them, else those the output and the inputs name (a dividend yield of zero among them); the
    call or put a row's type column names, else its price column's name, both tried where neither does. Said with the
    rows that miss where no candidate reprices most rows."""
    found = []
    for read in frames_:
        strike = next((k for k, column in enumerate(read.header) if STRIKE_COLUMN.match(column) and numeric(read, k)),
                      None)
        expiry = next((k for k, column in enumerate(read.header) if EXPIRY_COLUMN.match(column) and numeric(read, k)),
                      None)
        vols = [k for k, column in enumerate(read.header) if IV_COLUMN.search(column) and numeric(read, k)]
        prices = [k for k, column in enumerate(read.header) if PRICE_COLUMN.search(column) and numeric(read, k)
                  and not IV_COLUMN.search(column)]
        if strike is None or expiry is None or not vols or not prices:
            continue
        spots, rates, dividends = option_inputs(task_, out)
        if not spots or not rates:
            continue
        try:
            import numpy as np
            import qf
        except Exception:  # noqa: BLE001 - no library, no pricer
            return found
        for v in vols:
            mine = [k for k in prices if tokens(read.header[k]) & (tokens(read.header[v]) - {"implied", "vol", "iv"})] \
                or (prices if len(prices) == 1 else [])
            for p in mine:
                rows = [row for row in range(read.rows) if None not in (number(read.cell(strike, row)),
                        number(read.cell(expiry, row)), number(read.cell(v, row)), number(read.cell(p, row)))]
                rows = [row for row in rows if number(read.cell(expiry, row)) > 0 and number(read.cell(v, row)) > 0]
                if len(rows) < 2:
                    continue
                k_ = np.array([number(read.cell(strike, row)) for row in rows])
                t_ = np.array([number(read.cell(expiry, row)) for row in rows])
                v_ = np.array([number(read.cell(v, row)) for row in rows])
                p_ = np.array([number(read.cell(p, row)) for row in rows])
                typed = next((k for k, column in enumerate(read.header) if TYPE_COLUMN.match(column)), None)
                named = [kind for kind in ("call", "put") if kind in read.header[p].lower()]
                if typed is not None:
                    row_kinds = [option_kind(read.cell(typed, row)) for row in rows]
                    choices = [row_kinds] if None not in row_kinds else []
                else:
                    choices = [[kind] * len(rows) for kind in (named or ["call", "put"])]
                best = None
                for s in spots:
                    for r in rates:
                        for q in dividends:
                            for kinds in choices:
                                misses = repriced(qf, np, kinds, s, k_, t_, r, q, v_, p_)
                                if misses is None:
                                    continue
                                misses = [rows[at] for at in misses]
                                if best is None or len(misses) < len(best):
                                    best = misses
                if best and len(best) * 2 > len(rows):
                    found.append((f"{read.name}:{read.header[v]}", f"{where(read.name)}: `{read.header[v]}` does not "
                                  f"reprice `{read.header[p]}` through Black-Scholes (qf.bs_price) on "
                                  f"{said_rows(read, best)} for any spot, rate and dividend yield the output and the "
                                  f"inputs name."))
    return found


def flag(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A boolean the output reports for its own reconciliation (a key naming a validation, a repricing, a convergence
    or a consistency) is true: one written false says the output's numbers fail the comparison it was asked to make."""
    found = []
    for name, path, value in out.pairs():
        if value is False and FLAG_KEY.search(leaf(path)):
            found.append((path, f"`{name}`'s `{path}` is false: the output's own comparison fails, so a number it "
                          f"compares is not yet the one the task asks."))
    return found


def option_kind(cell) -> str | None:
    text = str(cell).strip().lower()
    return "call" if text in ("call", "c", "calls") else "put" if text in ("put", "p", "puts") else None


def repriced(qf, np, kinds, s, k_, t_, r, q, v_, p_) -> list[int] | None:
    """The rows a candidate's Black-Scholes prices miss: off the price by more than `PRICE_SLACK` of it (and a ten
    thousandth) and by more than `VOL_SLACK` in volatility (the miss over the price's vega); None where the library
    refuses the candidate."""
    misses = []
    for at, kind in enumerate(kinds):
        try:
            model, up, down = (float(qf.bs_price(option=kind, spot=s, strike=k_[at], maturity=t_[at], rate=r,
                                                 dividend_yield=q, vol=vol))
                               for vol in (v_[at], v_[at] + 1e-4, max(v_[at] - 1e-4, 1e-8)))
        except Exception:  # noqa: BLE001 - a candidate the library refuses fits nothing
            return None
        miss = abs(model - p_[at])
        vega = (up - down) / (v_[at] + 1e-4 - max(v_[at] - 1e-4, 1e-8))
        if miss > max(PRICE_SLACK * abs(p_[at]), 1e-4) and (vega <= 0 or miss / vega > VOL_SLACK):
            misses.append(at)
    return misses


# ------------------------------------------------------------------------------------------------ the dates


def series_tables(frames_: list[Table]) -> list[tuple[Table, list[datetime], float]]:
    """The output's dated series: a table whose date column names a date, month, period or day (not a maturity,
    payment, event or trade's), `SERIES` rows or more, a return, NAV, value, weight or P&L column among its own, with
    its sorted distinct dates and median step in days."""
    found = []
    for read in frames_:
        if read.date_column is None or NOT_SERIES_DATE.search(read.header[read.date_column]) \
                or not SERIES_DATE.search(read.header[read.date_column]):
            continue
        if not any(SERIES_COLUMN.search(column) for k, column in enumerate(read.header) if k != read.date_column) \
                and not WEIGHT_TABLE.search(read.name):
            continue
        days = read.dates()
        if len(days) < SERIES:
            continue
        step = statistics.median((later - earlier).days for earlier, later in zip(days, days[1:]))
        found.append((read, days, step))
    return found


def unit_of(step: float) -> str | None:
    if step <= DAILY_STEP:
        return "day"
    if step <= 10:
        return "week"
    if step <= MONTHLY_STEP:
        return "month"
    return "quarter" if step <= 100 else None


def range_(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A dated series' dates lie inside the inputs' dates: none before the first input date, none after the last (a
    period-end label of the inputs' last period aside)."""
    if not task_.calendar:
        return []
    found = []
    first, last = task_.calendar[0], task_.calendar[-1]
    for read, days, step in series_tables(frames_):
        unit = unit_of(step)
        end = witness.period_last(last, unit) if unit in ("week", "month", "quarter") else last
        before = [day for day in days if day < first]
        after = [day for day in days if day > end]
        if before or after:
            parts = ([f"before the inputs' first date ({witness.said_dates(before)})"] if before else []) + \
                    ([f"after the inputs' last date ({witness.said_dates(after)})"] if after else [])
            found.append((read.name, f"`{read.name}` holds dates " + " and ".join(parts) + ": its rows are dated "
                          f"outside the data the task gives."))
    return found


def trading(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A weekly, monthly or quarterly series' dates are observed input dates where the task's words say trading days
    (the last trading day of a month, observed dates), and calendar period ends where they say calendar labels: said
    with the dates that are the other kind. Where the words say neither, nothing is said."""
    read_words = words(task_.instruction)
    if not task_.calendar or bool(read_words.trading) == bool(read_words.calendar):
        return []
    found = []
    observed = set(task_.calendar)
    for read, days, step in series_tables(frames_):
        unit = unit_of(step)
        if unit not in ("week", "month", "quarter"):
            continue
        inside = [day for day in days if task_.calendar[0] <= day <= task_.calendar[-1]]
        if read_words.trading:
            off = [day for day in inside if day not in observed]
            if off:
                labels = sum(day == witness.period_last(day, unit) for day in off)
                found.append((read.name, f"`{read.name}`'s dates are not observed input dates on "
                              f"{witness.said_dates(off)}" + (f" (calendar {unit}-end labels)" if labels else "")
                              + f", where {quoted(read_words.trading)}."))
        else:
            off = [day for day in inside if day != witness.period_last(day, unit)]
            if off:
                found.append((read.name, f"`{read.name}`'s dates are not calendar {unit} ends on "
                              f"{witness.said_dates(off)}, where {quoted(read_words.calendar)}."))
    return found


def bounds(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A dated series lies inside the window the task's words state for its sample (the earliest date a sentence on
    the sample, period, window or data starts from, the latest it runs to): said with the dates outside it."""
    start, end, said = words(task_.instruction).window
    if start is None and end is None:
        return []
    found = []
    for read, days, step in series_tables(frames_):
        unit = unit_of(step)
        last = witness.period_last(end, unit) if end is not None and unit in ("week", "month", "quarter") else end
        before = [day for day in days if start is not None and day < start]
        after = [day for day in days if last is not None and day > last]
        if before or after:
            parts = ([f"before the window's start ({witness.said_dates(before)})"] if before else []) + \
                    ([f"after its end ({witness.said_dates(after)})"] if after else [])
            found.append((read.name, f"`{read.name}` holds dates " + " and ".join(parts) + f", where the task says "
                          f'"{said[:100]}".'))
    return found


def gaps(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A weekly, monthly or quarterly series (one row a period) skips no period between its first and last date that
    the inputs hold a full period's dates in (`witness.FULL_SHARE` of the median a period holds): said with the
    periods missing."""
    if not task_.calendar:
        return []
    found = []
    for read, days, step in series_tables(frames_):
        unit = unit_of(step)
        if unit not in ("week", "month", "quarter"):
            continue
        period = witness.PERIODS[unit]
        held = [period(day) for day in days]
        if len(set(held)) < 0.9 * len(held):
            continue
        sizes = {}
        for day in task_.calendar:
            sizes[period(day)] = sizes.get(period(day), 0) + 1
        full = witness.FULL_SHARE * statistics.median(sizes.values()) if sizes else 0
        missing = [key for key in sorted(sizes) if min(held) < key < max(held) and key not in set(held)
                   and sizes[key] >= full]
        if missing:
            found.append((read.name, f"`{read.name}` skips {unit}s the inputs hold between its first and last date ("
                          + ", ".join("-".join(f"{part:02d}" for part in key) for key in missing[:NAMED])
                          + (" among others" if len(missing) > NAMED else "") + ")."))
    return found


def count(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """A reported count of rebalances (or of months, weeks or holding periods) is the rows of the output's one table
    of that kind: a dated table with one row a date that is not daily (a rebalance schedule, a weights table) for
    rebalances, a monthly or weekly series for months or weeks. Said where it is neither the table's rows nor one
    fewer, and the table is the output's only one of its kind."""
    found = []
    dated = [(read, days, step) for read, days, step in series_tables(frames_)] + \
        [(read, read.dates(), 0.0) for read in frames_ if read.date_column is not None and WEIGHT_TABLE.search(read.name)
         and len(read.dates()) >= 2 and not any(read is each for each, _, _ in series_tables(frames_))]
    for scalar in scalars(out):
        match = COUNT_KEY.match(scalar.key)
        if not match or scalar.value != int(scalar.value):
            continue
        kind = match.group(1).lower()
        if kind.startswith("rebalanc") or "holding" in kind:
            mine = [(read, days) for read, days, step in dated if len(days) == read.rows and
                    (unit_of(statistics.median((b - a).days for a, b in zip(days, days[1:]))) or "year") != "day"]
        else:
            unit = "week" if "week" in kind else "month"
            mine = [(read, days) for read, days, step in dated if unit_of(step) == unit]
        if len(mine) != 1:
            continue
        read, days = mine[0]
        if int(scalar.value) not in (len(days), len(days) - 1):
            found.append((scalar.path, f"`{scalar.file}`'s `{scalar.path}` is not the count of dated rows in "
                          f"`{read.name}`, the output's table of them."))
    return found


def start(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """Where the task's words state an initial capital (or a path's starting value), a NAV, wealth or equity path of
    the output that opens at a value of that scale (within a factor of two of it, or of one) opens at it exactly, the
    capital as one where the path reads in units of it."""
    stated = words(task_.instruction).capital
    if stated is None:
        return []
    capital, said = stated
    found = []
    for read in frames_:
        for k, column in enumerate(read.header):
            if not (numeric(read, k) and PATH_NAME.search(column) and not CUM_NAME.search(column)):
                continue
            series = [value for value in values(read, k) if value is not None]
            if len(series) < 2:
                continue
            first = series[0]
            scale = capital if 0.5 * capital <= first <= 2 * capital else 1.0 if 0.5 <= first <= 2 else None
            if scale is not None and not near(first, scale, digits_slack(read.cells[k])):
                found.append((f"{read.name}:{column}", f"{where(read.name)}: `{column}` opens at a value other than "
                              f"the initial capital, where {quoted(said)} (its first row is the ledger's start)."))
    return found


def on_calendar(calendar: list[datetime], day: datetime, after: bool) -> int | None:
    """The place on `calendar` of `day`, or of the first date after it (`after`) or the last before it; None past an
    end."""
    import bisect
    at = bisect.bisect_left(calendar, day)
    if at < len(calendar) and calendar[at] == day:
        return at
    at = at if after else at - 1
    return at if 0 <= at < len(calendar) else None


def date_cells(read: Table, column: int) -> list[datetime | None]:
    return [witness.moment(cell) for cell in read.cells[column]]


def event(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """An event table's dates sit on the inputs' trading calendar as its own columns say: a row's date is its relative
    day's trading days from its event date (day 0 the event date, or the trading day after it or before it where it
    is none, both tried), and an event date mapped from a meeting, release or report date is the first input date on
    or after it, or the last on or before it, as the task's words say (both tried where they say neither). Each of the
    inputs' calendars is tried (every input date, complete rows', each table's, the dates all share); said where none
    places most rows."""
    calendars = witness.calendars_of(task_)
    if not calendars:
        return []
    read_words = task_.instruction
    sides = (True,) if AFTER_WORDS.search(read_words) and not BEFORE_WORDS.search(read_words) else \
        (False,) if BEFORE_WORDS.search(read_words) and not AFTER_WORDS.search(read_words) else (True, False)
    found = []
    for read in frames_:
        events = [k for k, column in enumerate(read.header) if EVENT_COLUMN.search(column)]
        if not events:
            continue
        e = events[0]
        event_days = date_cells(read, e)
        if sum(day is not None for day in event_days) < 2:
            continue
        relative = next((k for k, column in enumerate(read.header) if RELATIVE_COLUMN.match(column)
                         and read.numeric(k)), None)
        dated = next((k for k, column in enumerate(read.header) if k != e and witness.conventions.DATE_NAME.search(column)
                      and not SOURCE_COLUMN.search(column) and sum(day is not None for day in date_cells(read, k)) >= 2),
                     None)
        if relative is not None and dated is not None:
            offsets, days = values(read, relative), date_cells(read, dated)
            rows = [row for row in range(read.rows) if event_days[row] and days[row] and offsets[row] is not None
                    and offsets[row] == int(offsets[row])]
            best = None
            for calendar in calendars:
                for after in (True, False):
                    misses = []
                    for row in rows:
                        at = on_calendar(calendar, event_days[row], after)
                        target = at + int(offsets[row]) if at is not None else None
                        if target is None or not 0 <= target < len(calendar) or calendar[target] != days[row]:
                            misses.append(row)
                    if best is None or len(misses) < len(best):
                        best = misses
            if rows and best and 2 * len(best) > len(rows):
                found.append((f"{read.name}:{read.header[dated]}", f"{where(read.name)}: `{read.header[dated]}` is not "
                              f"`{read.header[relative]}` trading days from `{read.header[e]}` on the inputs' calendar "
                              f"on {said_rows(read, best)} (day 0 the event's own trading day)."))
            continue
        source = next((k for k, column in enumerate(read.header) if k != e and SOURCE_COLUMN.search(column)
                       and sum(day is not None for day in date_cells(read, k)) >= 2), None)
        if source is None:
            continue
        sources = date_cells(read, source)
        rows = [row for row in range(read.rows) if sources[row] and event_days[row]]
        best = None
        for calendar in calendars:
            for after in sides:
                misses = [row for row in rows if (at := on_calendar(calendar, sources[row], after)) is None
                          or calendar[at] != event_days[row]]
                if best is None or len(misses) < len(best):
                    best = misses
        if best and len(best) >= MISSES:
            side = ("on or after" if sides == (True,) else "on or before" if sides == (False,) else
                    f"on or after it or on or before it, {UNPINNED}")
            found.append((f"{read.name}:{read.header[e]}", f"{where(read.name)}: `{read.header[e]}` is not the input "
                          f"date {side} `{read.header[source]}` on {said_rows(read, best)}."))
    return found


EMPTY_TABLE = re.compile(r"return|result|metric|summary|series|portfolio|nav|alpha|coef|weight", re.I)
EMPTY_KEY = re.compile(r"^(?:n|num|number|count)_(?:(?:valid|clean|signal|selected|ic|oos|is|total|used|regime|"
                       r"sample|return)_)*(?:obs|observations|returns|periods|months|weeks|days|dates|rebalances|"
                       r"rebalance_dates|trading_days|rows)$", re.I)


def empty(task_: Task, out: Files, frames_: list[Table]) -> list[tuple[str, str]]:
    """The output reports a sample: where the inputs hold rows (dated or not), a count of the sample's observations,
    returns, periods, months, weeks, days, dates, rebalances or rows (`EMPTY_KEY`: the count's word, the sample's
    qualifiers alone, then the unit) is not zero; and a table of returns, results, metrics, a series or a portfolio
    holds rows with values (a header alone, or every value cell empty, is none)."""
    found = []
    held = bool(task_.calendar) or any(each.rows for each in task_.inputs.tables)
    for scalar in scalars(out):
        if held and scalar.value == 0 and EMPTY_KEY.match(scalar.key):
            found.append((scalar.path, f"`{scalar.file}`'s `{scalar.path}` is zero where the inputs hold rows: the "
                          f"sample the task asks for came out empty."))
    for read in out.tables:
        if not EMPTY_TABLE.search(read.name) or not read.header:
            continue
        filled = [k for k in range(len(read.header)) if k != read.date_column
                  and any(str(cell).strip() and not witness.blank(cell) for cell in read.cells[k])]
        labels = [k for k in filled if not read.numeric(k)]
        if not [k for k in filled if k not in labels and read.numeric(k)
                                  and not witness.COUNT_WORD.search(read.header[k])]:
            found.append((read.name, f"`{read.name}` holds no values"
                          + (" (its header alone)" if read.rows == 0 else " (its value cells are empty)")
                          + ": the sample the task asks for came out empty."))
    return found


FUNCTIONS = {"chain": chain, "total": total, "drawdown": drawdown, "annual": annual, "weights": weights,
             "discount": discount, "quotes": quotes, "survival": survival, "pricer": pricer, "flag": flag,
             "range": range_, "trading": trading, "bounds": bounds, "gaps": gaps, "count": count, "start": start,
             "event": event, "empty": empty}


@dataclass(frozen=True)
class Found:
    """One check's finding: the check, its line, and what it is known by across runs (`recon_acted`)."""
    check: str
    line: str
    ident: str

    @property
    def family(self) -> str:
        return FAMILY[self.check]


def bare(where: Path) -> list[Table]:
    """The output's CSV and TSV files written with a header and no row, as tables of no rows (witness's reader takes
    none of them)."""
    found = []
    try:
        named = sorted(path for path in Path(where).rglob("*") if path.is_file() and not path.is_symlink()
                       and path.suffix.lower() in (".csv", ".tsv"))
    except OSError:
        return found
    for path in named:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines = [line for line in (handle.readline(), handle.readline()) if line.strip()]
        except OSError:
            continue
        if len(lines) == 1:
            header = [cell.strip() for cell in lines[0].split("\t" if path.suffix.lower() == ".tsv" else ",")]
            found.append(Table(path.relative_to(where).as_posix(), header, [[] for _ in header]))
    return found


def run(task_: Task, out: Path | Files, checks=KEPT) -> list[Found]:
    """The findings of `checks` over the output directory `out` (its tables, JSON files and header-only tables:
    `bare`; or its files read), in `CHECKS` order; a check that raises says nothing."""
    if isinstance(out, Files):
        read = out
    else:
        read = witness.files(Path(out))
        read = Files([*read.tables, *bare(Path(out))], read.jsons)
    frames_ = frames(read)
    found = []
    for name in CHECKS:
        if name not in checks:
            continue
        try:
            pairs = FUNCTIONS[name](task_, read, frames_)
        except Exception:  # noqa: BLE001 - a check that cannot read the output says nothing
            pairs = []
        for ident, line in dict(pairs).items():
            found.append(Found(name, line if len(line) <= LINE_CHARS else line[:LINE_CHARS - 1] + "…",
                               f"{name}:{ident}"))
    return found


def subject(line: str) -> str:
    """A finding's object: the first name it quotes (its file, or its file's key)."""
    found = re.match(r"`([^`]+)`", line)
    return found.group(1).split(":", 1)[0] if found else ""


def quantity(line: str) -> str:
    """The quantity a finding names, as it quotes it: the file and its key or column where it names both."""
    found = re.match(r"(`[^`]+`)(?:'s |: )(`[^`]+`)", line)
    if found:
        return f"{found.group(1)}'s {found.group(2)}"
    found = re.match(r"`[^`]+`", line)
    return found.group(0) if found else ""


def ident_of(each) -> str:
    """What a finding is known by across runs: a reconciliation finding's `ident`; one of derive1's checks, its check
    and object."""
    return getattr(each, "ident", None) or f"witness.{each.check}:{subject(each.line)}"


def deduped(witnessed, reconciled) -> tuple:
    """derive1's findings less those whose object a reconciliation check of the same kind already names (`DEDUPE`):
    one finding an object."""
    named = {(each.check, subject(each.line)) for each in reconciled}
    return tuple(each for each in witnessed
                 if not any((check, subject(each.line)) in named for check in DEDUPE.get(each.check, ())))


def said(found: list, limit: int = SAID) -> str:
    """The run's paragraph: `HEAD`, then one line a finding (recon's, or derive1's before them where both are armed:
    commands.findings_said), at most `limit`; "" where none."""
    if not found:
        return ""
    return "\n".join([HEAD, *(f"- {each.line}" for each in found[:limit])])
