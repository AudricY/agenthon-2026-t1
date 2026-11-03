"""qf.bs.barrier_price (runs/lanes/lib3-prompt.md): Reiner and Rubinstein's eight single-barrier options with a rebate,
against Haug's (2007) table of them, QuantLib 1.43's AnalyticBarrierEngine (frozen below; recomputed where QuantLib is
installed), the in-out parity, the rebates as first-passage integrals (qf.hit_probability, qf.first_passage_pdf), and
the delta as the price's derivative. Every parameter is Haug's example or chosen freely; no unit's value appears."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import qf  # noqa: E402
import qf.bs  # noqa: E402

pytestmark = pytest.mark.fast

TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")
OPTIONS = ("call", "put")


def barrier(**given):
    point = dict(option="call", barrier_type="down-and-out", spot=100.0, strike=100.0, barrier=90.0, rebate=0.0,
                 rebate_timing="hit", maturity=0.75, rate=0.04, dividend_yield=0.015, vol=0.3)
    point.update(given)
    if point["barrier_type"].endswith("in") and "rebate_timing" not in given:
        point["rebate_timing"] = "expiry"
    return qf.bs.barrier_price(**point)


# ------------------------------------------------------------------------------------------------ Haug's table

HAUG = {   # Haug (2007), 4.17.1, Table 4-13: S 100, T 0.5, r 8%, b 4% (q 4%), rebate 3, vol 25%; K 90, 100, 110
    ("down-and-out", "call", 95.0): (9.0246, 6.7924, 4.8759),
    ("up-and-out", "call", 105.0): (2.6789, 2.3580, 2.3453),
    ("down-and-in", "call", 95.0): (7.7627, 4.0109, 2.0576),
    ("up-and-in", "call", 105.0): (14.1112, 8.4482, 4.5910),
    ("down-and-in", "put", 95.0): (2.9586, 6.5677, 11.9752),
    ("up-and-in", "put", 105.0): (1.4653, 3.3721, 7.0846),
    ("down-and-out", "put", 95.0): (2.2798, 2.2947, 2.6252),
    ("up-and-out", "put", 105.0): (3.7760, 5.4932, 7.5187),
}


@pytest.mark.parametrize("barrier_type, option, h", list(HAUG))
def test_barrier_price_gives_haugs_table(barrier_type, option, h):
    """Haug's table of the Reiner-Rubinstein formulas (QuantLib's test suite carries it too, and QuantLib 1.43
    reproduces every entry to 4 decimals): the outs pay the rebate at the hit, the ins at expiry."""
    timing = "expiry" if barrier_type.endswith("in") else "hit"
    got = qf.bs.barrier_price(option=option, barrier_type=barrier_type, spot=100.0, strike=np.array([90.0, 100.0, 110.0]),
                              barrier=h, rebate=3.0, rebate_timing=timing, maturity=0.5, rate=0.08,
                              dividend_yield=0.04, vol=0.25)
    assert got["price"] == pytest.approx(HAUG[(barrier_type, option, h)], abs=5e-5)


def test_a_barrier_crossed_already_has_knocked_as_haugs_table_reads_at_h_equal_s():
    """Haug's rows with H = S = 100: an out is its rebate (3, paid now), an in the vanilla; delta 0 and the vanilla's.
    An out paying at expiry is worth the rebate discounted."""
    for option, k in itertools.product(OPTIONS, (90.0, 100.0, 110.0)):
        common = dict(option=option, spot=100.0, strike=k, barrier=100.0, rebate=3.0, maturity=0.5, rate=0.08,
                      dividend_yield=0.04, vol=0.25)
        vanilla = qf.bs.bs_greeks(option=option, spot=100.0, strike=k, maturity=0.5, rate=0.08, dividend_yield=0.04,
                                  vol=0.25, vega_per=1.0, theta_per="year", rho_per=1.0)
        for side in ("down", "up"):
            knocked_in = qf.bs.barrier_price(barrier_type=f"{side}-and-in", rebate_timing="expiry", **common)
            assert knocked_in["price"] == pytest.approx(vanilla["price"], rel=1e-13)
            assert knocked_in["delta"] == pytest.approx(vanilla["delta"], rel=1e-13)
            assert qf.bs.barrier_price(barrier_type=f"{side}-and-out", rebate_timing="hit", **common) == {
                "price": 3.0, "delta": 0.0}
            late = qf.bs.barrier_price(barrier_type=f"{side}-and-out", rebate_timing="expiry", **common)
            assert late["price"] == pytest.approx(3.0 * math.exp(-0.08 * 0.5), rel=1e-15) and late["delta"] == 0.0
    beyond = barrier(barrier_type="up-and-out", barrier=95.0, rebate=1.0)
    assert beyond == {"price": 1.0, "delta": 0.0}


# ------------------------------------------------------------------------------------------------ QuantLib

QL = {   # (barrier_type, option, strike, barrier, rebate, days/360, rate, dividend_yield, vol): (NPV, FD delta)
    ('down-and-out', 'call', 85.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (12.148168058659465, 1.1795066950170963),
    ('down-and-out', 'call', 85.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (13.538872227630444, 1.1242426345017265),
    ('down-and-out', 'call', 100.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (7.876004080286396, 0.7870683146258273),
    ('down-and-out', 'call', 100.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (9.266708249257375, 0.7318042541104575),
    ('down-and-out', 'call', 115.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (4.444573140386193, 0.4644307487255972),
    ('down-and-out', 'call', 115.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (5.835277309357172, 0.4091666882102274),
    ('down-and-out', 'put', 85.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (0.0, 0.0),
    ('down-and-out', 'put', 85.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (1.3907041689709787, -0.055264060515369806),
    ('down-and-out', 'put', 100.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (0.08029508231130222, 0.00656650668418024),
    ('down-and-out', 'put', 100.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (1.470999251282281, -0.048697553831189566),
    ('down-and-out', 'put', 115.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (1.0013232030954597, 0.08293382787005754),
    ('down-and-out', 'put', 115.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (2.3920273720664387, 0.027669767354687735),
    ('down-and-in', 'call', 85.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (7.5948690938888035, -0.3923583120020169),
    ('down-and-in', 'call', 85.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (8.175196968646718, -0.3391576603926083),
    ('down-and-in', 'call', 100.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (3.193779404320953, -0.21348751830174706),
    ('down-and-in', 'call', 100.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (3.774107279078868, -0.1602868666918944),
    ('down-and-in', 'call', 115.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (1.223713028827472, -0.10006902930914663),
    ('down-and-in', 'call', 115.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (1.804040903585387, -0.04686837769929397),
    ('down-and-in', 'put', 85.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (3.3496030430481376, -0.20166466159565744),
    ('down-and-in', 'put', 85.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (3.9299309178060526, -0.14846400998580478),
    ('down-and-in', 'put', 100.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (9.152737296023545, -0.421798754981495),
    ('down-and-in', 'put', 100.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (9.73306517078146, -0.3685981033720864),
    ('down-and-in', 'put', 115.0, 90.0, 0.0, 270, 0.04, 0.015, 0.3): (17.386894863073323, -0.707385153065232),
    ('down-and-in', 'put', 115.0, 90.0, 2.0, 270, 0.04, 0.015, 0.3): (17.96722273783124, -0.6541845014549352),
    ('up-and-out', 'call', 85.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (1.2546707265994934, -0.08897803916418923),
    ('up-and-out', 'call', 85.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (2.535423124334607, -0.031328324660906404),
    ('up-and-out', 'call', 100.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (0.1090390859942616, -0.007582544878204089),
    ('up-and-out', 'call', 100.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (1.389791483729375, 0.05006716962496771),
    ('up-and-out', 'call', 115.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (0.0, 0.0),
    ('up-and-out', 'call', 115.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (1.2807523977351134, 0.0576497145031718),
    ('up-and-out', 'put', 85.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (2.730463303030491, -0.25697721947715024),
    ('up-and-out', 'put', 85.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (4.011215700765604, -0.19932750497408946),
    ('up-and-out', 'put', 100.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (6.739355868491302, -0.592975231580084),
    ('up-and-out', 'put', 100.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (8.020108266226416, -0.5353255170765792),
    ('up-and-out', 'put', 115.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (11.784840988563094, -1.0027861930730353),
    ('up-and-out', 'put', 115.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (13.065593386298207, -0.9451364785695304),
    ('up-and-in', 'call', 85.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (18.488366425948776, 0.8761264221792686),
    ('up-and-in', 'call', 85.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (19.175636320090916, 0.8204739546613382),
    ('up-and-in', 'call', 100.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (10.960744398613087, 0.5811633412022843),
    ('up-and-in', 'call', 100.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (11.648014292755226, 0.5255108736852421),
    ('up-and-in', 'call', 115.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (5.6682861692136655, 0.36436171941645057),
    ('up-and-in', 'call', 115.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (6.355556063355804, 0.30870925189940834),
    ('up-and-in', 'put', 85.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (0.6191397400176468, 0.05531255788149281),
    ('up-and-in', 'put', 85.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (1.3064096341597855, -0.000339909635549418),
    ('up-and-in', 'put', 100.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (2.493676509843546, 0.17774298328276927),
    ('up-and-in', 'put', 100.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (3.1809464039856845, 0.12209051576572705),
    ('up-and-in', 'put', 115.0, 112.0, 0.0, 270, 0.04, 0.015, 0.3): (6.603377077605689, 0.37833486787874904),
    ('up-and-in', 'put', 115.0, 112.0, 2.0, 270, 0.04, 0.015, 0.3): (7.290646971747828, 0.3226824003617068),
    ('down-and-out', 'call', 104.0, 90.0, 1.5, 450, -0.005, 0.01, 0.2): (6.082584417221302, 0.5010826762332776),
    ('down-and-out', 'put', 104.0, 90.0, 1.5, 450, -0.005, 0.01, 0.2): (1.37109362560922, -0.016244528371678335),
    ('down-and-in', 'call', 104.0, 90.0, 1.5, 450, -0.005, 0.01, 0.2): (1.8301937857968782, -0.0648353357395326),
    ('down-and-in', 'put', 104.0, 90.0, 1.5, 450, -0.005, 0.01, 0.2): (12.43594001641207, -0.5350859316353151),
    ('up-and-out', 'call', 104.0, 112.0, 1.5, 450, -0.005, 0.01, 0.2): (0.8747580715779346, 0.04728499509670803),
    ('up-and-out', 'put', 104.0, 112.0, 1.5, 450, -0.005, 0.01, 0.2): (10.422188406632765, -0.7454026610460573),
    ('up-and-in', 'call', 104.0, 112.0, 1.5, 450, -0.005, 0.01, 0.2): (7.039056480089728, 0.3882918413671277),
    ('up-and-in', 'put', 104.0, 112.0, 1.5, 450, -0.005, 0.01, 0.2): (3.385881584038008, 0.1934016970084329),
}


@pytest.mark.parametrize("key", list(QL))
def test_barrier_price_and_delta_equal_quantlibs_analytic_barrier_engine(key):
    """QuantLib 1.43, AnalyticBarrierEngine on a BlackScholesMertonProcess, T = days / 360 on Actual360: its NPV to
    1e-12 and its spot-bumped delta (central, 1e-3) to 1e-6. QuantLib pays an out's rebate at the hit, an in's at
    expiry; a negative rate is among the points."""
    barrier_type, option, k, h, r_cash, days, rate, q, vol = key
    got = qf.bs.barrier_price(option=option, barrier_type=barrier_type, spot=100.0, strike=k, barrier=h,
                              rebate=r_cash, rebate_timing="expiry" if barrier_type.endswith("in") else "hit",
                              maturity=days / 360, rate=rate, dividend_yield=q, vol=vol)
    assert got["price"] == pytest.approx(QL[key][0], abs=1e-12)
    assert got["delta"] == pytest.approx(QL[key][1], abs=1e-6)


def test_the_frozen_values_are_quantlibs():
    """Recomputes a third of the frozen table where QuantLib is installed (the lane's venv; not the image)."""
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual360()
    kinds = {"down-and-out": ql.Barrier.DownOut, "down-and-in": ql.Barrier.DownIn, "up-and-out": ql.Barrier.UpOut,
             "up-and-in": ql.Barrier.UpIn}
    for key in list(QL)[::3]:
        barrier_type, option, k, h, r_cash, days, rate, q, vol = key
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(100.0)), ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, dc)))
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if option == "call" else ql.Option.Put, k)
        instrument = ql.BarrierOption(kinds[barrier_type], h, r_cash, payoff, ql.EuropeanExercise(today + days))
        instrument.setPricingEngine(ql.AnalyticBarrierEngine(process))
        assert instrument.NPV() == pytest.approx(QL[key][0], abs=1e-12)


# ------------------------------------------------------------------------------------------------ identities


@pytest.mark.parametrize("option", OPTIONS)
def test_in_plus_out_is_the_vanilla_for_every_strike_and_barrier(option):
    """In-out parity with no rebate: a knock-in and its knock-out together are the vanilla (qf.bs_price), on either side
    of the barrier and for strikes above and below it."""
    strikes = np.array([70.0, 88.0, 95.0, 100.0, 109.0, 130.0])
    for side, barriers in (("down", (60.0, 88.0, 97.0)), ("up", (103.0, 109.0, 140.0))):
        for h in barriers:
            common = dict(option=option, strike=strikes, barrier=h, rebate=0.0, rebate_timing="expiry")
            knock_in = barrier(barrier_type=f"{side}-and-in", **common)
            knock_out = barrier(barrier_type=f"{side}-and-out", **common)
            vanilla = qf.bs_price(option, 100.0, strikes, 0.75, 0.04, 0.015, 0.3)
            assert knock_in["price"] + knock_out["price"] == pytest.approx(vanilla, abs=1e-11)
            delta = qf.bs.bs_greeks(option=option, spot=100.0, strike=strikes, maturity=0.75, rate=0.04,
                                    dividend_yield=0.015, vol=0.3, vega_per=1.0, theta_per="year", rho_per=1.0)["delta"]
            assert knock_in["delta"] + knock_out["delta"] == pytest.approx(delta, abs=1e-11)
            assert np.all(knock_out["price"] >= -1e-12) and np.all(knock_out["price"] <= vanilla + 1e-12)


@pytest.mark.parametrize("side, h", [("down", 85.0), ("up", 120.0)])
def test_the_rebates_are_the_first_passage_integrals(side, h):
    """An in's rebate R e^{-rT} P(no hit) and an out's at the hit R E[e^{-r tau}; tau <= T] (Haug's E and F), from
    the first-passage law of ln S, drift r - q - vol^2/2 (qf.hit_probability; qf.first_passage_pdf integrated by
    quad); an out's at expiry R e^{-rT} P(hit)."""
    rate, q, vol, t, cash = 0.04, 0.015, 0.3, 0.75, 2.0
    level, drift = math.log(h / 100.0), rate - q - 0.5 * vol * vol
    hit = qf.hit_probability(level, t, drift, vol)
    at_hit = integrate.quad(lambda s: math.exp(-rate * s) * qf.first_passage_pdf(level, s, drift, vol), 0.0, t,
                            epsabs=1e-13, epsrel=1e-12)[0]
    for option, k in itertools.product(OPTIONS, (80.0, 100.0, 125.0)):
        def price(kind, rebate, timing):
            return barrier(option=option, barrier_type=f"{side}-and-{kind}", strike=k, barrier=h, rebate=rebate,
                           rebate_timing=timing)["price"]
        assert price("in", cash, "expiry") - price("in", 0.0, "expiry") == pytest.approx(
            cash * math.exp(-rate * t) * (1.0 - hit), abs=1e-12)
        assert price("out", cash, "hit") - price("out", 0.0, "hit") == pytest.approx(cash * at_hit, abs=1e-10)
        assert price("out", cash, "expiry") - price("out", 0.0, "expiry") == pytest.approx(
            cash * math.exp(-rate * t) * hit, abs=1e-12)


@pytest.mark.parametrize("barrier_type, option", list(itertools.product(TYPES, OPTIONS)))
def test_delta_is_the_derivative_of_the_price(barrier_type, option):
    """The closed-form delta against central differences of the price (step 1e-4), with a rebate at either timing,
    strikes on both sides of the barrier."""
    h = 88.0 if barrier_type.startswith("down") else 115.0
    for k, timing in itertools.product((80.0, 100.0, 120.0), ("hit", "expiry")):
        if barrier_type.endswith("in") and timing == "hit":
            continue
        point = dict(option=option, barrier_type=barrier_type, strike=k, barrier=h, rebate=1.5, rebate_timing=timing)
        up, down = barrier(spot=100.0 + 1e-4, **point)["price"], barrier(spot=100.0 - 1e-4, **point)["price"]
        assert barrier(**point)["delta"] == pytest.approx((up - down) / 2e-4, abs=2e-8)


def test_arrays_broadcast_and_a_bad_argument_is_refused():
    strikes = np.array([90.0, 100.0, 110.0])
    together = barrier(strike=strikes, rebate=1.0)
    alone = [barrier(strike=k, rebate=1.0) for k in strikes]
    assert together["price"] == pytest.approx([a["price"] for a in alone], abs=1e-14)
    assert type(alone[0]["price"]) is float and type(alone[0]["delta"]) is float
    spots = barrier(spot=np.array([85.0, 95.0, 105.0]))          # the first spot is through the barrier already
    assert spots["price"][0] == 0.0 and spots["delta"][0] == 0.0 and spots["price"][1] > 0
    with pytest.raises(ValueError, match="rebate_timing='expiry'"):
        barrier(barrier_type="down-and-in", rebate=1.0, rebate_timing="hit")
    assert barrier(barrier_type="down-and-in", rebate=0.0, rebate_timing="hit")["price"] > 0
    for bad in (dict(rebate=-1.0), dict(barrier_type="double-knock-out"), dict(rebate_timing="never"), dict(vol=0.0),
                dict(maturity=0.0), dict(barrier=0.0), dict(option="straddle")):
        with pytest.raises(ValueError):
            barrier(**bad)
    with pytest.raises(ValueError, match="negative"):
        barrier(rebate=1.0, rate=-0.02, dividend_yield=-0.025, vol=0.1)    # mu = 0, lambda^2 = 2 r / vol^2 < 0
