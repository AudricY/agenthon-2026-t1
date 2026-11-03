"""agent/qf/conventions.py (runs/lanes/lib1-prompt.md): every function against a published definition or worked
example (RFC 8259 and 4180, ISDA 2006 4.16, Hull's compounding examples, Hyndman and Fan's quantile types, the
RiskMetrics recursion), an identity, or an independent implementation as the oracle (QuantLib 1.43 where the lane's
environment has it, pandas' ewm, Python's statistics module). The card checks shared by the three new shelves live
here too (`card_problems`). No unit's instruction values, checks, expected values or data appear: every parameter is
a textbook example or chosen freely."""
import datetime
import inspect
import itertools
import json
import math
import os
import statistics
import textwrap

import numpy as np
import pytest

import qf
import qf.conventions as cv
from conftest import FIRST_LINE, folded, shelved_before

pytestmark = pytest.mark.fast

SECTIONS = ("conventions:", "returns:", "source:", "aliases:", "example:")


# ------------------------------------------------------------------------------------------------ the cards


OPTIONAL = {"write_csv": ("decimals",), "write_json": ("decimals",)}   # lane contract1: the writers' stated
#   precision, None (full precision) by default: no convention units disagree on, and nothing lost unstated


def card_problems(module) -> list[str]:
    """What breaks the card format of runs/lanes/lib1-prompt.md in a shelf module: its line 1 within `FIRST_LINE`
    characters (the index shows it) and the public functions exactly `__all__` but the folded (lane libshelves1); and
    for every function the shelf held before the re-shelving (`shelved_before`, wherever it lives now): line 1 within
    `FIRST_LINE`, then the formula, then `conventions:` with one line per keyword argument, `returns:`, `source:`,
    `aliases:` on one line and a 3-line `example:` naming every keyword; no default anywhere."""
    problems = []
    head = module.__doc__.splitlines()[0]
    if len(head) > FIRST_LINE or ":" not in head:
        problems.append(f"module line 1 {head!r}")
    public = {name for name, f in inspect.getmembers(module, inspect.isfunction)
              if f.__module__ == module.__name__ and not name.startswith("_")}
    if public - folded() != set(module.__all__):
        problems.append(f"public {sorted((public - folded()) ^ set(module.__all__))}")
    for name in shelved_before(module.__name__.rpartition(".")[2]):
        f = getattr(qf, name)
        parameters = inspect.signature(f).parameters.values()
        if any(p.default is not inspect.Parameter.empty and not (p.default is None and p.name in OPTIONAL.get(name, ()))
               for p in parameters):
            problems.append(f"{name}: a default")
        keywords = [p.name for p in parameters if p.kind is inspect.Parameter.KEYWORD_ONLY]
        doc = inspect.getdoc(f)
        lines = doc.splitlines()
        if len(lines[0]) > FIRST_LINE or not lines[0].endswith("."):
            problems.append(f"{name}: line 1 {len(lines[0])} characters")
        at = [next((i for i, line in enumerate(lines) if line.startswith(section)), -1) for section in SECTIONS]
        if -1 in at or at != sorted(at) or at[0] < 2:
            problems.append(f"{name}: sections {dict(zip(SECTIONS, at))}")
            continue
        conventions = lines[at[0] + 1:at[1]]
        missing = [k for k in keywords if not any(line.startswith(f"    {k}: ") for line in conventions)]
        if missing:
            problems.append(f"{name}: conventions lack {missing}")
        aliases = [a.strip() for a in lines[at[3]][len("aliases:"):].split(",")]
        if at[4] != at[3] + 1 or not all(aliases):
            problems.append(f"{name}: aliases")
        example = [line for line in lines[at[4] + 1:] if line.strip()]
        if len(example) != 3 or f"{name}(" not in example[-1] or any(f"{k}=" not in example[-1] for k in keywords):
            problems.append(f"{name}: example")
    return problems


def run_examples(module, tmp_path, skip=()):
    """Execute in a scratch directory the card's example of each function the shelf held before the re-shelving: the
    call a model copies must run as written."""
    os.chdir(tmp_path)
    for name in shelved_before(module.__name__.rpartition(".")[2]):
        if name in skip:
            continue
        doc = inspect.getdoc(getattr(qf, name))
        exec(textwrap.dedent(doc.split("example:\n", 1)[1]), {})


def test_the_cards_follow_the_format_and_every_example_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert card_problems(cv) == []
    run_examples(cv, tmp_path)
    assert (tmp_path / "output/results.json").is_file()


# ------------------------------------------------------------------------------------------------ files


def test_write_json_writes_plain_numbers_and_names_the_key_of_a_nan(tmp_path):
    """RFC 8259 section 6: JSON numbers have no NaN or Infinity (json.dumps writes the token NaN, which a strict
    reader refuses). numpy scalars and arrays come out as Python numbers and lists, dates as ISO strings."""
    payload = {"a": np.float64(0.1), "b": np.int64(3), "c": np.array([[1.5, 2.0], [3.0, 4.25]]), "d": (np.bool_(True),),
               "e": {7: datetime.date(2024, 2, 29), "f": np.datetime64("2024-03-01")}, "g": None}
    plain = cv.write_json(tmp_path / "out/x.json", payload)
    assert json.loads((tmp_path / "out/x.json").read_text()) == plain
    assert plain == {"a": 0.1, "b": 3, "c": [[1.5, 2.0], [3.0, 4.25]], "d": [True],
                     "e": {"7": "2024-02-29", "f": "2024-03-01"}, "g": None}
    assert [type(plain[k]) for k in "abd"] == [float, int, list] and type(plain["d"][0]) is bool
    stamps = np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[ns]")
    assert cv.write_json(tmp_path / "d.json", {"dates": stamps, "first": stamps[0]}) == {
        "dates": ["2024-01-02", "2024-01-03"], "first": "2024-01-02"}
    with pytest.raises(ValueError, match=r"payload\['risk'\]\['var'\] is nan"):
        cv.write_json(tmp_path / "y.json", {"risk": {"var": np.float64("nan")}})
    with pytest.raises(ValueError, match=r"payload\['w'\]\[1\] is inf"):
        cv.write_json(tmp_path / "y.json", {"w": np.array([0.5, np.inf])})
    for bad in ({"z": 1 + 2j}, {"s": {1, 2}}, {1: "a", "1": "b"}, {(1, 2): 3}):
        with pytest.raises(ValueError):
            cv.write_json(tmp_path / "y.json", bad)
    assert not (tmp_path / "y.json").exists()


def test_a_written_table_reads_back_exactly_with_its_dtype_and_missing_report(tmp_path):
    """RFC 4180 tables: floats at full precision read back bit for bit (Python's shortest round-trip repr); a column
    of numbers and missing tokens is float64 with NaN, anything else stays text, and the report says why."""
    values = np.array([0.1, 1e-5, 1.0 / 3.0, math.nan, -2.5e300])
    for delimiter in (",", "\t"):
        path = tmp_path / ("t.csv" if delimiter == "," else "t.tsv")
        written = cv.write_csv(path, {"date": ["2024-01-02", "2024-01-03", "x, y", "", "z"], "v": values,
                                      "n": np.arange(5)}, delimiter=delimiter, missing="")
        assert written == 5
        columns, report = cv.read_table(path, delimiter=delimiter, missing=("", "NA"))
        assert list(columns) == ["date", "v", "n"]
        np.testing.assert_array_equal(columns["v"], values)
        assert columns["v"].dtype == np.float64 and columns["n"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert columns["date"].tolist() == ["2024-01-02", "2024-01-03", "x, y", "", "z"]
        assert report["v"] == {"dtype": "float64", "rows": 5, "missing": 1, "unparsed": 0}
        assert report["date"] == {"dtype": "str", "rows": 5, "missing": 1, "unparsed": 4}
    (tmp_path / "m.csv").write_text("﻿ a ,b\n1,NA\n2,n/a\n NaN ,3\n", encoding="utf-8")
    columns, report = cv.read_table(tmp_path / "m.csv", delimiter=",", missing=("NA", "NaN"))
    assert np.isnan(columns["a"][2]) and columns["a"][:2].tolist() == [1.0, 2.0]
    assert report["a"] == {"dtype": "float64", "rows": 3, "missing": 1, "unparsed": 0}
    assert report["b"] == {"dtype": "str", "rows": 3, "missing": 1, "unparsed": 1}
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n3\n")
    with pytest.raises(ValueError, match="row 3"):
        cv.read_table(tmp_path / "bad.csv", delimiter=",", missing="")
    (tmp_path / "dup.csv").write_text("a,a\n1,2\n")
    with pytest.raises(ValueError, match="repeats"):
        cv.read_table(tmp_path / "dup.csv", delimiter=",", missing="")
    with pytest.raises(ValueError, match=r"columns\['v'\]\[1\] is missing"):
        cv.write_csv(tmp_path / "n.csv", {"v": [1.0, math.nan]}, delimiter=",", missing=None)
    with pytest.raises(ValueError, match="infinity"):
        cv.write_csv(tmp_path / "n.csv", {"v": [1.0, math.inf]}, delimiter=",", missing="")
    with pytest.raises(ValueError, match="equal lengths"):
        cv.write_csv(tmp_path / "n.csv", {"v": [1.0], "w": [1.0, 2.0]}, delimiter=",", missing="")
    cv.write_csv(tmp_path / "ns.csv", {"date": np.array(["2024-01-02"], dtype="datetime64[ns]"), "v": [1.5]},
                 delimiter=",", missing="")
    assert (tmp_path / "ns.csv").read_text() == "date,v\n2024-01-02,1.5\n"
    cv.write_csv(tmp_path / "nan.csv", {"v": [math.nan, 2.0]}, delimiter=";", missing="NaN")
    assert (tmp_path / "nan.csv").read_bytes() == b"v\nNaN\n2.0\n"


# ------------------------------------------------------------------------------------------------ returns and scale


def test_annualize_scales_a_mean_by_p_a_vol_by_its_root_and_compounds_a_return():
    """Hull, "Estimating volatility": a daily vol of 1% is 0.01 sqrt(252) = 15.87% a year; a 1% monthly return
    compounds to 1.01^12 - 1 = 12.68%. Annualising a variance then taking its root is annualising the vol."""
    assert cv.annualize(0.01, quantity="vol", periods_per_year=252) == pytest.approx(0.158745, abs=1e-6)
    assert cv.annualize(0.01, quantity="compound", periods_per_year=12) == pytest.approx(0.126825, abs=1e-6)
    assert cv.annualize(0.0004, quantity="mean", periods_per_year=252) == pytest.approx(0.1008, abs=1e-15)
    assert cv.annualize(0.0004, quantity="mean", periods_per_year=365) == pytest.approx(0.146, abs=1e-15)
    vols = np.array([0.008, 0.013])
    assert np.sqrt(cv.annualize(vols ** 2, quantity="variance", periods_per_year=52)) == pytest.approx(
        cv.annualize(vols, quantity="vol", periods_per_year=52), rel=1e-15)
    for bad in (dict(quantity="vol", periods_per_year=0), dict(quantity="sharpe", periods_per_year=252)):
        with pytest.raises(ValueError):
            cv.annualize(0.01, **bad)


def test_log_returns_are_log_of_one_plus_simple_and_both_telescope_to_the_price_ratio():
    """Campbell, Lo and MacKinlay 1.4.1: r = ln(1 + R); log returns add over time, simple returns compound."""
    p = np.array([[100.0, 50.0], [104.0, 49.0], [98.8, 51.5], [101.3, 52.0]])
    simple, log = cv.returns_from_prices(p, kind="simple"), cv.returns_from_prices(p, kind="log")
    assert simple.shape == (3, 2)
    assert simple[0].tolist() == pytest.approx([0.04, -0.02], abs=1e-15)
    assert log == pytest.approx(np.log1p(simple), abs=1e-15)
    assert log.sum(axis=0) == pytest.approx(np.log(p[-1] / p[0]), abs=1e-14)
    assert np.prod(1 + simple, axis=0) == pytest.approx(p[-1] / p[0], rel=1e-14)
    assert np.isnan(cv.returns_from_prices([1.0, math.nan, 2.0], kind="log")).all()
    for bad in ([1.0, 0.0, 2.0], [1.0, -1.0], [3.0]):
        with pytest.raises(ValueError):
            cv.returns_from_prices(bad, kind="simple")


def test_max_drawdown_is_the_largest_fall_from_a_running_peak():
    """Magdon-Ismail and Atiya (2004): MDD = max_t (1 - V_t / max_{s<=t} V_s); against every pair (i <= j) on random
    paths, and a hand path whose second fall (130 to 65) is the larger."""
    path = [100.0, 120.0, 90.0, 130.0, 65.0, 140.0]
    assert qf.max_drawdown(path, sign="loss_positive") == pytest.approx(0.5, abs=1e-15)
    assert qf.max_drawdown(path, sign="return_negative") == pytest.approx(-0.5, abs=1e-15)
    rng = np.random.default_rng(11)
    for _ in range(20):
        v = np.exp(np.cumsum(rng.normal(0, 0.05, 40)))
        brute = max(1 - v[j] / v[i] for i in range(40) for j in range(i, 40))
        assert qf.max_drawdown(v, sign="loss_positive") == pytest.approx(brute, abs=1e-14)
    assert qf.max_drawdown([1.0, 2.0, 3.0], sign="loss_positive") == 0.0
    with pytest.raises(ValueError):
        qf.max_drawdown([0.01, -0.02, 0.03], sign="loss_positive")


# ------------------------------------------------------------------------------------------------ day counts and rates


def test_year_fraction_follows_isdas_definitions_on_hand_cases():
    """ISDA 2006 section 4.16: (d) Actual/365 (Fixed), (e) Actual/360, (b) Actual/Actual (ISDA), (f) 30/360 Bond
    Basis, (g) 30E/360 Eurobond Basis, each case's arithmetic written out."""
    yf = cv.year_fraction
    assert yf("2024-01-01", "2024-07-01", basis="ACT/360") == 182 / 360
    assert yf("2024-01-01", "2024-07-01", basis="ACT/365") == yf("2024-01-01", "2024-07-01", basis="ACT/365F") \
        == 182 / 365
    assert yf("2023-12-15", "2025-03-01", basis="ACT/ACT ISDA") == pytest.approx(17 / 365 + 1 + 59 / 365, abs=1e-15)
    assert yf("2024-02-01", "2024-03-01", basis="ACT/ACT ISDA") == pytest.approx(29 / 366, abs=1e-15)
    assert yf("2022-07-01", "2024-03-01", basis="ACT/ACT ISDA") == pytest.approx(184 / 365 + 1 + 60 / 366, abs=1e-15)
    assert yf("2024-11-01", "2025-02-01", basis="ACT/ACT ISDA") == pytest.approx(61 / 366 + 31 / 365, abs=1e-15)
    # Bond Basis: D1 31 -> 30; D2 31 -> 30 only where D1 is then 30
    assert yf("2024-01-31", "2024-03-31", basis="30/360") == 60 / 360
    assert yf("2024-01-29", "2024-03-31", basis="30/360") == 62 / 360
    assert yf("2024-01-29", "2024-03-31", basis="30E/360") == 61 / 360
    assert yf("2024-02-29", "2024-08-31", basis="30/360") == (180 + 2) / 360
    assert yf("2024-02-29", "2024-08-31", basis="30E/360") == (180 + 1) / 360
    assert yf("2023-08-31", "2024-02-29", basis="30/360") == (360 - 180 + 29 - 30) / 360
    assert yf("2024-07-01", "2024-01-01", basis="ACT/360") == -182 / 360
    assert yf("2025-03-01", "2023-12-15", basis="ACT/ACT ISDA") == pytest.approx(-(17 / 365 + 1 + 59 / 365))
    assert yf(datetime.date(2024, 1, 1), np.datetime64("2024-01-31"), basis="ACT/360") == 30 / 360
    both = yf(["2024-01-01", "2024-01-31"], "2024-03-31", basis="30/360")
    assert both.tolist() == [90 / 360, 60 / 360]
    with pytest.raises(ValueError, match="'ACT/360'"):
        yf("2024-01-01", "2024-07-01", basis="ACT/365L")


def test_year_fraction_agrees_with_quantlib_on_month_ends_and_leap_years():
    """QuantLib 1.43 as the oracle: Actual360, Actual365Fixed, ActualActual(ISDA), Thirty360(BondBasis) and
    Thirty360(European) on every pair of a grid of month ends, the 30th, 29 Feb and mid-months over 2023 to 2026."""
    ql = pytest.importorskip("QuantLib")
    counters = {"ACT/360": ql.Actual360(), "ACT/365F": ql.Actual365Fixed(),
                "ACT/ACT ISDA": ql.ActualActual(ql.ActualActual.ISDA),
                "30/360": ql.Thirty360(ql.Thirty360.BondBasis), "30E/360": ql.Thirty360(ql.Thirty360.European)}
    days = sorted({datetime.date(y, m, d) for y in (2023, 2024, 2026) for m in (1, 2, 3, 8, 12)
                   for d in (1, 15, 28, 29, 30, 31) if _valid(y, m, d)})
    pairs = list(itertools.product(days, days))
    starts, ends = [a for a, _ in pairs], [b for _, b in pairs]
    dated = {d: ql.Date(d.day, d.month, d.year) for d in days}
    for basis, counter in counters.items():
        ours = cv.year_fraction(starts, ends, basis=basis)
        theirs = [counter.yearFraction(dated[a], dated[b]) for a, b in pairs]
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-14, err_msg=basis)


def _valid(y, m, d):
    try:
        datetime.date(y, m, d)
    except ValueError:
        return False
    return True


def test_hulls_compounding_examples_and_the_round_trip():
    """Hull, "Measuring interest rates": 10% with semiannual compounding is 2 ln(1.05) = 9.758% continuous; 8%
    continuous paid quarterly is 4 (e^0.02 - 1) = 8.08%. A conversion and its inverse give the rate back, and a
    simple rate over t is (growth - 1) / t."""
    assert cv.convert_rate(0.10, maturity=1.0, from_compounding=2, to_compounding="continuous") == pytest.approx(
        2 * math.log(1.05), abs=1e-15)
    assert cv.convert_rate(0.08, maturity=1.0, from_compounding="continuous", to_compounding=4) == pytest.approx(
        0.080805, abs=1e-6)
    assert cv.compound_factor(0.05, maturity=2.0, compounding=2) == pytest.approx(1.025 ** 4, rel=1e-15)
    assert cv.compound_factor(0.05, maturity=0.5, compounding="simple") == 1.025
    assert cv.compound_factor(0.05, maturity=3.0, compounding="continuous") == pytest.approx(math.exp(0.15))
    assert cv.convert_rate(0.05, maturity=0.5, from_compounding="continuous", to_compounding="simple") == \
        pytest.approx((math.exp(0.025) - 1) / 0.5, rel=1e-15)
    rules = ("simple", "continuous", 1, 2, 4, 12)
    for a, b in itertools.product(rules, rules):
        there = cv.convert_rate(np.array([0.01, 0.07]), maturity=2.5, from_compounding=a, to_compounding=b)
        back = cv.convert_rate(there, maturity=2.5, from_compounding=b, to_compounding=a)
        assert back == pytest.approx([0.01, 0.07], rel=1e-12), (a, b)
    for bad in ("annual", 0, True, 2.0):
        with pytest.raises(ValueError):
            cv.compound_factor(0.05, maturity=1.0, compounding=bad)
    with pytest.raises(ValueError):
        cv.convert_rate(0.05, maturity=0.0, from_compounding=1, to_compounding=2)


def test_compounding_agrees_with_quantlibs_interest_rate():
    """QuantLib 1.43's InterestRate as the oracle: compoundFactor(t) and equivalentRate(rule, frequency, t) for
    every pair of simple, continuous, annual, semiannual, quarterly and monthly over four maturities."""
    ql = pytest.importorskip("QuantLib")
    dc = ql.Actual365Fixed()
    rules = {"simple": (ql.Simple, ql.Annual), "continuous": (ql.Continuous, ql.Annual), 1: (ql.Compounded, ql.Annual),
             2: (ql.Compounded, ql.Semiannual), 4: (ql.Compounded, ql.Quarterly), 12: (ql.Compounded, ql.Monthly)}
    for (a, (ca, fa)), (b, (cb, fb)), t in itertools.product(rules.items(), rules.items(), (0.25, 1.0, 2.5, 7.0)):
        rate = ql.InterestRate(0.0425, dc, ca, fa)
        assert cv.compound_factor(0.0425, maturity=t, compounding=a) == pytest.approx(rate.compoundFactor(t),
                                                                                      rel=1e-14)
        assert cv.convert_rate(0.0425, maturity=t, from_compounding=a, to_compounding=b) == pytest.approx(
            rate.equivalentRate(cb, fb, t).rate(), rel=1e-12), (a, b, t)


# ------------------------------------------------------------------------------------------------ statistics


def test_ewma_gives_hulls_example_and_each_seed_its_closed_form():
    """Hull, "The EWMA model": lambda 0.90, yesterday's variance 0.01^2, yesterday's move 2%: today's variance
    0.9 x 0.0001 + 0.1 x 0.0004 = 0.00013. Unrolled, s_t = lam^{t+1} s_{-1} + (1 - lam) sum_i lam^{t-i} x_i with
    s_{-1} the seed (x_0, 0, the sample mean or the number given); TA-Lib's EMA(N) is lam = 1 - 2/(N + 1) seeded with
    the mean of the first N on the rest."""
    assert cv.ewma([0.0004], lam=0.9, seed=0.0001, nan="raise") == pytest.approx([0.00013], rel=1e-14)
    x = np.array([0.4, -1.1, 2.3, 0.7, -0.2, 1.5, 0.9])
    lam, t = 0.8, np.arange(x.size)
    for seed, s_minus in (("first", x[0]), ("zero", 0.0), ("mean", x.mean()), (0.37, 0.37)):
        closed = [lam ** (k + 1) * s_minus + (1 - lam) * np.sum(lam ** (k - t[:k + 1]) * x[:k + 1]) for k in t]
        assert cv.ewma(x, lam=lam, seed=seed, nan="raise") == pytest.approx(closed, rel=1e-13), seed
    adjusted = [np.sum(lam ** (k - t[:k + 1]) * x[:k + 1]) / np.sum(lam ** (k - t[:k + 1])) for k in t]
    assert cv.ewma(x, lam=lam, seed="adjusted", nan="raise") == pytest.approx(adjusted, rel=1e-13)
    n = 3
    head = cv.ewma(x[n:], lam=1 - 2 / (n + 1), seed=x[:n].mean(), nan="raise")
    talib = [x[:n].mean()]
    for value in x[n:]:
        talib.append(talib[-1] + 2 / (n + 1) * (value - talib[-1]))
    assert head == pytest.approx(talib[1:], rel=1e-13)
    both = cv.ewma(np.column_stack([x, 2 * x]), lam=lam, seed="first", nan="raise")
    assert both[:, 1] == pytest.approx(2 * cv.ewma(x, lam=lam, seed="first", nan="raise"), rel=1e-14)
    for bad in (dict(lam=1.0, seed="first"), dict(lam=0.9, seed="sma")):
        with pytest.raises(ValueError):
            cv.ewma(x, nan="raise", **bad)
    with pytest.raises(ValueError, match="NaN"):
        cv.ewma([1.0, math.nan], lam=0.5, seed="first", nan="raise")


def test_ewma_skips_a_nan_by_carrying_the_state():
    """The NaN rule 'skip' leaves the state as it was and reports it there (pandas' ignore_na=True), a leading NaN
    staying NaN under 'first'. lam 0.5 on [nan, 1, nan, 3, 4, nan, nan, 2]: 'first' gives 1, 1, 2, 3, 3, 3, 2.5; the
    'adjusted' mean weighs the observations by count, 1, 1, (3 + 1/2)/(3/2), (4 + 3/2 + 1/4)/(7/4), ..., (2 + 4/2 +
    3/4 + 1/8)/(15/8)."""
    x = [math.nan, 1.0, math.nan, 3.0, 4.0, math.nan, math.nan, 2.0]
    first = cv.ewma(x, lam=0.5, seed="first", nan="skip")
    assert np.isnan(first[0]) and first[1:].tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.5]
    adjusted = cv.ewma(x, lam=0.5, seed="adjusted", nan="skip")
    assert np.isnan(adjusted[0]) and adjusted[1:] == pytest.approx(
        [1.0, 1.0, 3.5 / 1.5, 5.75 / 1.75, 5.75 / 1.75, 5.75 / 1.75, 4.875 / 1.875], rel=1e-14)
    assert cv.ewma(x, lam=0.5, seed=0.0, nan="skip")[:3].tolist() == [0.0, 0.5, 0.5]


@pytest.mark.slow
def test_ewma_matches_pandas_ewm_with_adjust_and_ignore_na():
    """pandas' Series.ewm(alpha=1 - lam) as the oracle: seed 'first' is adjust=False, 'adjusted' adjust=True; the NaN
    rule 'skip' is ignore_na=True, leading NaNs included."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    x = rng.normal(size=60)
    x[[0, 1, 7, 8, 9, 30, 59]] = np.nan
    for seed, adjust in (("first", False), ("adjusted", True)):
        theirs = pd.Series(x).ewm(alpha=0.06, adjust=adjust, ignore_na=True).mean().to_numpy()
        np.testing.assert_allclose(cv.ewma(x, lam=0.94, seed=seed, nan="skip"), theirs, rtol=1e-13, atol=0)


def test_std_is_the_statistics_modules_stdev_and_pstdev():
    """Python's statistics module as the oracle: stdev divides by n - 1 (ddof 1), pstdev by n (ddof 0)."""
    x = [0.012, -0.031, 0.004, 0.027, -0.008, 0.019, -0.015]
    assert cv.std(x, ddof=1) == pytest.approx(statistics.stdev(x), rel=1e-14)
    assert cv.std(x, ddof=0) == pytest.approx(statistics.pstdev(x), rel=1e-14)
    cols = np.column_stack([x, np.array(x) * 3])
    assert cv.std(cols, ddof=1) == pytest.approx([statistics.stdev(x), 3 * statistics.stdev(x)], rel=1e-14)
    for bad in (dict(x=x, ddof=-1), dict(x=x, ddof=1.0), dict(x=[1.0], ddof=1), dict(x=[1.0, math.nan], ddof=0)):
        with pytest.raises(ValueError):
            cv.std(**bad)


def hyndman_fan(x, q, kind):
    """Hyndman and Fan (1996): types 1 (inverted cdf), 5 (hazen), 6 (weibull) and 7 (linear) from the order
    statistics, x_(h) interpolated at the 1-based position h and clamped to the sample."""
    s, n = np.sort(x), len(x)
    if kind == 1:
        return s[max(math.ceil(n * q), 1) - 1]
    h = {5: n * q + 0.5, 6: (n + 1) * q, 7: (n - 1) * q + 1}[kind]
    h = min(max(h, 1.0), float(n))
    lo = math.floor(h)
    return s[lo - 1] + (h - lo) * (s[min(lo, n - 1)] - s[lo - 1])


def test_quantile_rules_are_hyndman_and_fans_types_and_the_statistics_modules_cut_points():
    """Hyndman and Fan (1996), each type's position written out; Python's statistics.quantiles as a second oracle
    ('inclusive' is type 7, 'exclusive' type 6); 'lower', 'higher', 'nearest' and 'midpoint' the neighbours of type 7's
    position (nearest's tie goes to the even index)."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=23)
    for q in (0.01, 0.05, 0.1, 0.37, 0.5, 0.9, 0.99):
        for kind, method in ((1, "inverted_cdf"), (5, "hazen"), (6, "weibull"), (7, "linear")):
            assert cv.quantile(x, q, method=method) == pytest.approx(hyndman_fan(x, q, kind), abs=1e-14), (q, method)
        s, h = np.sort(x), (len(x) - 1) * q
        assert cv.quantile(x, q, method="lower") == s[math.floor(h)]
        assert cv.quantile(x, q, method="higher") == s[math.ceil(h)]
        assert cv.quantile(x, q, method="midpoint") == pytest.approx((s[math.floor(h)] + s[math.ceil(h)]) / 2)
    quartiles = np.array([0.25, 0.5, 0.75])
    assert cv.quantile(x, quartiles, method="linear") == pytest.approx(statistics.quantiles(x, n=4,
                                                                                          method="inclusive"))
    assert cv.quantile(x, quartiles, method="weibull") == pytest.approx(statistics.quantiles(x, n=4,
                                                                                           method="exclusive"))
    five = np.arange(5.0)
    assert cv.quantile(five, 0.375, method="nearest") == 2.0 and cv.quantile(five, 0.625, method="nearest") == 2.0
    assert cv.quantile(np.column_stack([x, -x]), 0.5, method="linear").shape == (2,)
    for bad in (dict(q=0.5, method="median_unbiased"), dict(q=1.5, method="linear")):
        with pytest.raises(ValueError):
            cv.quantile(x, **bad)


# ------------------------------------------------------------------------------------------------ windows


def test_rolling_apply_is_pandas_rolling_and_expanding_for_a_mean_a_pair_and_a_vector():
    """pandas' rolling(window).mean() and expanding(min_periods=window).mean(), NaN before the first full window;
    two arrays windowed together (a correlation, pandas' rolling corr); a function of two numbers gives two columns."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(21)
    x, y = rng.standard_normal(30), rng.standard_normal(30)
    for window in (1, 5, 30):
        rolled = cv.rolling_apply(np.mean, [x], window=window, kind="rolling")
        np.testing.assert_allclose(rolled, pd.Series(x).rolling(window).mean().to_numpy(), rtol=1e-12, atol=1e-15)
        grown = cv.rolling_apply(np.mean, [x], window=window, kind="expanding")
        np.testing.assert_allclose(grown, pd.Series(x).expanding(min_periods=window).mean().to_numpy(), rtol=1e-12,
                                   atol=1e-15)
    pair = cv.rolling_apply(lambda a, b: np.corrcoef(a, b)[0, 1], [x, y], window=8, kind="rolling")
    np.testing.assert_allclose(pair, pd.Series(x).rolling(8).corr(pd.Series(y)).to_numpy(), rtol=1e-10, atol=1e-12)
    both = cv.rolling_apply(lambda a: [a.min(), a.max()], [x], window=3, kind="rolling")
    assert both.shape == (30, 2) and np.all(np.isnan(both[:2]))
    assert both[2:, 0] == pytest.approx(pd.Series(x).rolling(3).min().to_numpy()[2:])


def test_rolling_apply_written_out_on_a_short_series():
    """Row t holds the window ending at t: [1, 2, 3, 4, 5] summed over 2 rows is NaN, 3, 5, 7, 9; expanding from 2
    rows, NaN, 3, 6, 10, 15; a 2-D series windows its rows."""
    rolled = cv.rolling_apply(np.sum, [[1.0, 2.0, 3.0, 4.0, 5.0]], window=2, kind="rolling")
    assert np.isnan(rolled[0]) and list(rolled[1:]) == [3.0, 5.0, 7.0, 9.0]
    grown = cv.rolling_apply(np.sum, [[1.0, 2.0, 3.0, 4.0, 5.0]], window=2, kind="expanding")
    assert np.isnan(grown[0]) and list(grown[1:]) == [3.0, 6.0, 10.0, 15.0]
    matrix = np.arange(8.0).reshape(4, 2)
    rows = cv.rolling_apply(lambda m: m.sum(axis=0), [matrix], window=3, kind="rolling")
    assert rows.shape == (4, 2) and list(rows[2]) == [6.0, 9.0] and list(rows[3]) == [12.0, 15.0]
    for bad, match in ((dict(arrays=[1.0, 2.0, 3.0]), "1-D or 2-D"), (dict(arrays=np.ones(3)), "list"),
                       (dict(window=4), "window"), (dict(window=0), "window"), (dict(kind="ewm"), "kind"),
                       (dict(arrays=[[1.0, 2.0, 3.0], [1.0, 2.0]]), "same number")):
        args = {"arrays": [[1.0, 2.0, 3.0]], "window": 2, "kind": "rolling", **bad}
        with pytest.raises(ValueError, match=match):
            cv.rolling_apply(np.sum, args["arrays"], window=args["window"], kind=args["kind"])
