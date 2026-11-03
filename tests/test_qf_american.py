"""qf.american_price and qf.binomial_price (lane libgaps1; docs/research/library-syllabus-1011.md rows 50 and 29):
American options by Barone-Adesi and Whaley's quadratic approximation, by Bjerksund and Stensland's flat boundaries
(1993, 2002), and on binomial trees (CRR, Jarrow-Rudd, Leisen-Reimer). References: QuantLib 1.43's engines (frozen in
`QL`; `__main__` recomputes them where QuantLib is installed), the MathWorks Financial Instruments Toolbox's
documented examples, Hull's worked tree, and the identities every American price keeps. Every input is a textbook's,
a documentation example's or this file's own; no unit's values appear (rule 8)."""
import functools
import math

import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast

QL = {   # QuantLib 1.43, Actual365Fixed, days from 2024-01-15 (`quantlib_values`)
    "barone_adesi_whaley": [
        (("call", 100.0, 100.0, 146, 0.06, 0.04, 0.25), 6.57901494596754),
        (("put", 100.0, 100.0, 146, 0.06, 0.02, 0.25), 5.5911685669853615),
        (("put", 90.0, 100.0, 365, 0.08, 0.0, 0.3), 13.64771716497794),
        (("call", 110.0, 100.0, 365, 0.03, 0.07, 0.35), 17.685416842229987),
        (("put", 100.0, 100.0, 18, 0.2, 0.0, 0.05), 0.19582334700301712),        # a low vol: large powers
        (("put", 200.0, 100.0, 3650, 0.2, 0.05, 0.2), 0.019056868122101694)],
    "bjerksund_stensland_1993": [    # QuantLib's BjerksundStenslandApproximationEngine is the 1993 form
        (("call", 100.0, 100.0, 146, 0.06, 0.04, 0.25), 6.576513628576592),
        (("put", 100.0, 100.0, 146, 0.06, 0.02, 0.25), 5.537455878057997),
        (("put", 90.0, 100.0, 365, 0.08, 0.0, 0.3), 13.568257529490658),
        (("call", 110.0, 100.0, 365, 0.03, 0.07, 0.35), 17.572359046419407),
        (("put", 100.0, 100.0, 18, 0.2, 0.0, 0.05), 0.19860989555684938),        # beta near 160: S^beta overflows
        (("put", 200.0, 100.0, 3650, 0.2, 0.05, 0.2), 0.0016294076593966749)],   # the boundary below the strike
    "trees": [   # BinomialVanillaEngine, 301 steps; its "crr" takes p = 1/2 + (r - q - vol^2/2) sqrt(dt) / (2 vol), the
        #          drift of ln S, not Cox, Ross and Rubinstein's (e^{(r-q)dt} - d)/(u - d): Hull's tree is the CRR check
        (("jarrow_rudd", "american", "put", 100.0, 100.0, 146, 0.06, 0.02, 0.25), 5.60029025024087),
        (("jarrow_rudd", "european", "put", 100.0, 100.0, 146, 0.06, 0.02, 0.25), 5.449383546716083),
        (("leisen_reimer", "american", "put", 100.0, 100.0, 146, 0.06, 0.02, 0.25), 5.596254095999368),
        (("leisen_reimer", "european", "put", 100.0, 100.0, 146, 0.06, 0.02, 0.25), 5.445791038446188),
        (("jarrow_rudd", "american", "put", 90.0, 100.0, 365, 0.08, 0.0, 0.3), 13.69695221819837),
        (("jarrow_rudd", "european", "put", 90.0, 100.0, 365, 0.08, 0.0, 0.3), 12.071641071637655),
        (("leisen_reimer", "american", "put", 90.0, 100.0, 365, 0.08, 0.0, 0.3), 13.687626087597728),
        (("leisen_reimer", "european", "put", 90.0, 100.0, 365, 0.08, 0.0, 0.3), 12.06333028163571)],
}
METHODS = ("barone_adesi_whaley", "bjerksund_stensland_1993", "bjerksund_stensland_2002")


def american(method, option, s, k, t, r, q, v):
    return qf.american_price(method=method, option=option, spot=s, strike=k, maturity=t, rate=r, dividend_yield=q,
                             vol=v)


def tree(tree, exercise, option, s, k, t, r, q, v, steps):
    return qf.binomial_price(option=option, exercise=exercise, tree=tree, spot=s, strike=k, maturity=t, rate=r,
                             dividend_yield=q, vol=v, steps=steps)


@pytest.mark.parametrize("method", ["barone_adesi_whaley", "bjerksund_stensland_1993"])
def test_the_approximations_equal_quantlibs_engines(method):
    """QuantLib's BaroneAdesiWhaleyApproximationEngine solves S* by Newton to about 1e-6, so the BAW prices meet to
    that; QuantLib's Bjerksund-Stensland engine is the 1993 form, met to rounding. Two edges: a low vol (the powers
    S^beta and (I/S)^kappa past a float's range, taken in logs) and a long-dated put far out of the money whose flat
    boundary falls below the strike (b T + 2 vol sqrt(T) < 0), where the approximation has no exercise region and
    gives the European, as QuantLib's does."""
    for (option, s, k, days, r, q, v), expected in QL[method]:
        found = american(method, option, s, k, days / 365, r, q, v)
        assert found["price"] == pytest.approx(expected, abs=2e-6 if method == "barone_adesi_whaley" else 1e-11)
    edge = american("bjerksund_stensland_2002", "put", 200.0, 100.0, 10.0, 0.2, 0.05, 0.2)
    assert edge["premium"] == pytest.approx(0.0, abs=1e-15) and edge["critical"] == 0.0


def test_bjerksund_stensland_2002_gives_the_mathworks_documented_example():
    """MathWorks Financial Instruments Toolbox, optstockbybjs (Bjerksund and Stensland 2002): K 100, r 8%, q 4%,
    vol 20%, six months (T 0.5): calls at S 80 and 90 0.4144 and 2.1804, puts at S 100 and 110 4.7253 and 1.7164.
    The 1993 form gives the puts 4.70 and 1.71, below: the 2002 boundary exercises better."""
    for option, s, expected in (("call", 80.0, 0.4144), ("call", 90.0, 2.1804), ("put", 100.0, 4.7253),
                                ("put", 110.0, 1.7164)):
        found = american("bjerksund_stensland_2002", option, s, 100.0, 0.5, 0.08, 0.04, 0.2)["price"]
        assert found == pytest.approx(expected, abs=5e-5), (option, s)
    older = american("bjerksund_stensland_1993", "put", 100.0, 100.0, 0.5, 0.08, 0.04, 0.2)["price"]
    assert older < 4.7253 - 0.01


def test_bjerksund_stensland_2002_gives_the_two_current_mathworks_pages():
    """The review's check on the two pages as they read on 10-12 (MathWorks Financial Instruments Toolbox), each
    K 100, r 8%, q 4%, the time actual/actual over 2008's 366 days. optstockbybjs: vol 20%, 1 Jan to 1 Apr (91 days),
    a call at S 90 0.8420 and a put at S 120 0.1108. optstocksensbybjs: vol 40%, 1 Jun to 1 Oct (122 days), puts at
    S 90, 100, 110 and 120 13.7594, 8.4830, 4.9451 and 2.7475."""
    for option, s, expected in (("call", 90.0, 0.8420), ("put", 120.0, 0.1108)):
        found = american("bjerksund_stensland_2002", option, s, 100.0, 91 / 366, 0.08, 0.04, 0.2)["price"]
        assert found == pytest.approx(expected, abs=5e-5), (option, s)
    puts = american("bjerksund_stensland_2002", "put", np.array([90.0, 100.0, 110.0, 120.0]), 100.0, 122 / 366, 0.08,
                    0.04, 0.4)["price"]
    assert puts == pytest.approx([13.7594, 8.4830, 4.9451, 2.7475], abs=5e-5)


def test_barone_adesi_whaley_gives_the_mathworks_documented_example():
    """MathWorks Financial Instruments Toolbox, optstockbybaw: a call, S 125, K 120, r 4%, q 2%, vol 14%, two years:
    14.5180."""
    assert american("barone_adesi_whaley", "call", 125.0, 120.0, 2.0, 0.04, 0.02, 0.14)["price"] == \
        pytest.approx(14.5180, abs=5e-5)


@pytest.mark.parametrize("method", METHODS)
def test_each_method_keeps_the_american_identities(method):
    """A call without a dividend yield is never exercised early (Merton 1973): it is the European, premium 0 and no
    boundary; a put with r <= 0 likewise. Elsewhere the American is at least the European and its intrinsic value,
    the premium is the difference, and at or beyond the critical price the option is worth its intrinsic value."""
    plain = american(method, "call", 95.0, 100.0, 1.0, 0.05, 0.0, 0.3)
    assert plain["price"] == pytest.approx(qf.bs_price("call", 95.0, 100.0, 1.0, 0.05, 0.0, 0.3), abs=1e-12)
    assert plain["premium"] == 0.0 and plain["critical"] == math.inf
    no_rate = american(method, "put", 95.0, 100.0, 1.0, 0.0, 0.02, 0.3)
    assert no_rate["premium"] == 0.0 and no_rate["critical"] == 0.0
    for option, q in (("call", 0.06), ("put", 0.01)):
        found = american(method, option, np.array([70.0, 100.0, 130.0]), 100.0, 0.75, 0.05, q, 0.25)
        sign = 1.0 if option == "call" else -1.0
        assert np.all(found["price"] >= found["european"] - 1e-12)
        assert np.all(found["price"] >= np.maximum(sign * (np.array([70.0, 100.0, 130.0]) - 100.0), 0.0) - 1e-12)
        assert np.allclose(found["premium"], found["price"] - found["european"])
        beyond = float(np.atleast_1d(found["critical"])[0]) * (1.01 if option == "call" else 0.99)
        assert american(method, option, beyond, 100.0, 0.75, 0.05, q, 0.25)["price"] == \
            pytest.approx(sign * (beyond - 100.0), abs=1e-9)


def test_baw_value_matching_at_the_critical_price():
    """Barone-Adesi and Whaley's S* solves S* - X = c(S*) + A2: the approximation meets the intrinsic value there
    from below (value matching), for the call and, at S**, for the put."""
    for option, q in (("call", 0.05), ("put", 0.0)):
        found = american("barone_adesi_whaley", option, 100.0, 100.0, 1.0, 0.07, q, 0.3)
        star = found["critical"]
        sign = 1.0 if option == "call" else -1.0
        near = american("barone_adesi_whaley", option, star * (1 - sign * 1e-9), 100.0, 1.0, 0.07, q, 0.3)["price"]
        assert near == pytest.approx(sign * (star - 100.0), abs=1e-6)


@functools.cache
def bench(option, s, q):
    """A 1,001-step CRR tree's American price, K 100, a year, r 6%, vol 30%: taken once for every method."""
    return tree("crr", "american", option, s, 100.0, 1.0, 0.06, q, 0.3, 1001)["price"]


@pytest.mark.parametrize("method", METHODS)
def test_each_approximation_lies_near_a_fine_tree(method):
    """Against a 1,001-step CRR tree (the benchmark of every paper on these approximations): within 1% across
    moneyness for a put and a dividend-paying call, the 1993 single boundary within 1.5%. A flat boundary is one
    exercise strategy, so Bjerksund and Stensland's prices lie below the American's (a lower bound)."""
    for option, s, q in (("put", 90.0, 0.0), ("put", 110.0, 0.03), ("call", 110.0, 0.07), ("call", 95.0, 0.05)):
        found = american(method, option, s, 100.0, 1.0, 0.06, q, 0.3)["price"]
        assert found == pytest.approx(bench(option, s, q), rel=0.015 if method.endswith("1993") else 0.01), (option, s)
        if method.startswith("bjerksund"):
            assert found < bench(option, s, q)


def test_the_trees_equal_quantlibs_binomial_engines():
    for (kind, exercise, option, s, k, days, r, q, v), expected in QL["trees"]:
        found = tree(kind, exercise, option, s, k, days / 365, r, q, v, 301)["price"]
        assert found == pytest.approx(expected, abs=1e-10), (kind, exercise)


def test_the_crr_tree_gives_hulls_american_put_and_its_greeks():
    """Hull, Options, Futures, and Other Derivatives, "Binomial trees in practice": an American put, S 50, K 50, r 10%,
    vol 40%, five months. Five steps give 4.49, delta -0.41, gamma 0.03 and theta -4.30 a year; 30, 50, 100 and 500
    steps give 4.263, 4.272, 4.278 and 4.283."""
    hull = dict(option="put", s=50.0, k=50.0, t=5 / 12, r=0.10, q=0.0, v=0.40)
    five = tree("crr", "american", steps=5, **hull)
    assert five["price"] == pytest.approx(4.49, abs=0.005)
    assert five["delta"] == pytest.approx(-0.41, abs=0.005) and five["gamma"] == pytest.approx(0.03, abs=0.005)
    assert five["theta"] == pytest.approx(-4.30, abs=0.005)
    for steps, expected in ((30, 4.263), (50, 4.272), (100, 4.278), (500, 4.283)):
        assert tree("crr", "american", steps=steps, **hull)["price"] == pytest.approx(expected, abs=5e-4), steps


def test_european_trees_converge_to_black_scholes_and_leisen_reimer_fastest():
    """Cox, Ross and Rubinstein's error falls as 1/n; Leisen and Reimer's as 1/n^2 (their Table): at 201 steps
    within 1e-4 of qf.bs_price, the CRR within 0.02. An American call on a stock with no yield is the European."""
    point = dict(option="call", s=100.0, k=105.0, t=0.8, r=0.04, q=0.01, v=0.22)
    exact = qf.bs_price("call", 100.0, 105.0, 0.8, 0.04, 0.01, 0.22)
    assert tree("leisen_reimer", "european", steps=201, **point)["price"] == pytest.approx(exact, abs=1e-4)
    for kind in ("crr", "jarrow_rudd"):
        assert tree(kind, "european", steps=201, **point)["price"] == pytest.approx(exact, abs=0.02), kind
    no_yield = dict(point, q=0.0)
    assert tree("crr", "american", steps=150, **no_yield)["price"] == \
        pytest.approx(tree("crr", "european", steps=150, **no_yield)["price"], abs=1e-12)
    greeks = qf.bs_greeks("call", 100.0, 105.0, 0.8, 0.04, 0.01, 0.22, 1.0, "year", 1.0)
    lr = tree("leisen_reimer", "european", steps=401, **point)
    assert lr["delta"] == pytest.approx(greeks["delta"], abs=2e-3) and lr["gamma"] == pytest.approx(greeks["gamma"],
                                                                                                   abs=2e-3)


def test_the_conventions_are_refused_when_unnamed_or_unfit():
    with pytest.raises(ValueError, match="method"):
        american("baw", "put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError, match="above 0"):
        american("barone_adesi_whaley", "put", 100.0, 100.0, 0.0, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError, match="odd"):
        tree("leisen_reimer", "american", "put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 100)
    with pytest.raises(ValueError, match="steps"):
        tree("crr", "american", "put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1)
    with pytest.raises(ValueError, match="exercise"):
        tree("crr", "bermudan", "put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 10)


def quantlib_values() -> dict:
    """The frozen `QL` values recomputed by QuantLib (scratch: a Python with QuantLib 1.43)."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    count = ql.Actual365Fixed()

    def process(s, r, q, v):
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(s)), ql.YieldTermStructureHandle(ql.FlatForward(today, q, count)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), v, count)))

    def priced(engine, option, s, k, days, r, q, v, exercise="american"):
        kind = ql.Option.Call if option == "call" else ql.Option.Put
        made = ql.AmericanExercise(today, today + days) if exercise == "american" else ql.EuropeanExercise(today + days)
        instrument = ql.VanillaOption(ql.PlainVanillaPayoff(kind, k), made)
        instrument.setPricingEngine(engine(process(s, r, q, v)))
        return instrument.NPV()

    engines = {"barone_adesi_whaley": ql.BaroneAdesiWhaleyApproximationEngine,
               "bjerksund_stensland_1993": ql.BjerksundStenslandApproximationEngine}
    found = {key: [(args, priced(engines[key], *args)) for args, _ in QL[key]] for key in engines}
    names = {"jarrow_rudd": "jr", "leisen_reimer": "lr"}
    found["trees"] = [((kind, ex, *rest), priced(lambda p, kind=kind: ql.BinomialVanillaEngine(p, names[kind], 301),
                                                 *rest, exercise=ex)) for (kind, ex, *rest), _ in QL["trees"]]
    return found


if __name__ == "__main__":
    fresh = quantlib_values()
    print("QL frozen values match QuantLib:", all(
        np.allclose([v for _, v in fresh[key]], [v for _, v in QL[key]], rtol=1e-14, atol=0) for key in QL))
