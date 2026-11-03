"""Lane lib-options1's barrier options (runs/lanes/lib-options1-prompt.md; docs/research/library-syllabus-1011.md rows
92, 93, 96): the Ikeda-Kunitomo double barrier, Heynen and Kat's partial-time barriers, the one-touch and no-touch.
References: Haug (2007)'s tables as QuantLib's test suite cites them (doublebarrieroption.cpp, Haug p. 156;
digitaloption.cpp, Haug (1998) p. 95), Haug's type A closed form with an exact bivariate normal, QuantLib 1.43's
partial-time engine (frozen in `QL`, to its bivariate normal's accuracy; `__main__` recomputes them where QuantLib is
installed), and the identities: in + out = vanilla, one-touch + no-touch = the discount factor, the far barrier gives
the single barrier, equal curvatures shift the carry. Every input is a textbook's or this file's own; no unit's values
appear (rule 8)."""
import math

import numpy as np
import pytest

import qf
from qf._common import bivariate_normal_cdf

pytestmark = pytest.mark.fast

QL = {   # QuantLib 1.43, Actual365Fixed, days from 2024-01-15 (`quantlib_values`): AnalyticPartialTimeBarrierOptionEngine
    "partial": [
        (('call', 'down-and-out', 'A', 100.0, 95.0, 90.0, 73, 365, 0.05, 0.02, 0.25), 11.169265720012639),
        (('call', 'up-and-out', 'A', 100.0, 110.0, 120.0, 219, 365, 0.05, 0.02, 0.25), 1.4824410011406979),
        (('call', 'down-and-out', 'B1', 100.0, 95.0, 90.0, 73, 365, 0.05, 0.02, 0.25), 10.519856286219301),
        (('put', 'up-and-out', 'B1', 100.0, 110.0, 120.0, 219, 365, 0.05, 0.02, 0.25), 13.127503527245114),
        (('put', 'down-and-out', 'B2', 100.0, 110.0, 90.0, 73, 365, 0.05, 0.02, 0.25), 0.6817710797219281),
        (('call', 'up-and-out', 'B2', 100.0, 95.0, 120.0, 219, 365, 0.05, 0.02, 0.25), 1.644959067728884),
    ],
}
HAUG_DOUBLE = [   # Haug (2007) p. 156, flat barriers: S 100, K 100, r 10%, b 10% (q 0); (L, U, option, T, vol, value)
    (50.0, 150.0, "call", 0.25, 0.15, 4.3515), (50.0, 150.0, "call", 0.5, 0.35, 6.5088),
    (60.0, 140.0, "call", 0.25, 0.25, 5.8500), (70.0, 130.0, "call", 0.5, 0.25, 4.0004),
    (80.0, 120.0, "call", 0.25, 0.35, 1.4903), (90.0, 110.0, "call", 0.5, 0.15, 0.5537),
    (90.0, 110.0, "call", 0.25, 0.35, 0.0477), (50.0, 150.0, "put", 0.25, 0.15, 1.8825),
    (50.0, 150.0, "put", 0.5, 0.25, 4.7033)]
HAUG_TOUCH = [   # Haug (1998) p. 95, cases 1 to 12, as QuantLib's digitaloption.cpp cites them: H 100, T 0.5, r 10%,
    #              q 0, vol 20%; cash 15 (the cash cases scaled by it), asset one unit
    ("one", "hit", "cash", "down", 105.0, 15 * 1, 9.7264), ("one", "hit", "cash", "up", 95.0, 15, 11.6553),
    ("one", "hit", "asset", "down", 105.0, 1, 64.8426), ("one", "hit", "asset", "up", 95.0, 1, 77.7017),
    ("one", "expiry", "cash", "down", 105.0, 15, 9.3604), ("one", "expiry", "cash", "up", 95.0, 15, 11.2223),
    ("no", "expiry", "cash", "down", 105.0, 15, 4.9081), ("no", "expiry", "cash", "up", 95.0, 15, 3.0461),
    ("one", "expiry", "asset", "down", 105.0, 1, 64.8426), ("one", "expiry", "asset", "up", 95.0, 1, 77.7017),
    ("no", "expiry", "asset", "down", 105.0, 1, 40.1574), ("no", "expiry", "asset", "up", 95.0, 1, 17.2983)]


# ------------------------------------------------------------------------------------------------ double barrier

@pytest.mark.parametrize("low, high, option, t, vol, expected", HAUG_DOUBLE)
def test_the_double_barrier_gives_haugs_table(low, high, option, t, vol, expected):
    found = qf.double_barrier_price(option, "out", 100.0, 100.0, low, high, t, 0.1, 0.0, vol, 0.0, 0.0, 5)
    assert found["price"] == pytest.approx(expected, abs=5e-5)


def test_the_double_barriers_in_and_out_and_its_far_barriers():
    s, k, t, r, q, v = 100.0, 95.0, 0.75, 0.04, 0.01, 0.3
    for option in ("call", "put"):
        out = qf.double_barrier_price(option, "out", s, k, 70.0, 140.0, t, r, q, v, 0.0, 0.0, 5)["price"]
        inn = qf.double_barrier_price(option, "in", s, k, 70.0, 140.0, t, r, q, v, 0.0, 0.0, 5)["price"]
        assert out + inn == pytest.approx(qf.bs_price(option, s, k, t, r, q, v), rel=1e-12)
        far = qf.double_barrier_price(option, "out", s, k, 80.0, 1e6, t, r, q, v, 0.0, 0.0, 5)["price"]
        single = qf.barrier_price(option, "down-and-out", s, k, 80.0, 0.0, "expiry", t, r, q, v)["price"]
        assert far == pytest.approx(single, rel=1e-10)
    up = qf.double_barrier_price("call", "out", s, k, 1e-6, 130.0, t, r, q, v, 0.0, 0.0, 5)["price"]
    assert up == pytest.approx(qf.barrier_price("call", "up-and-out", s, k, 130.0, 0.0, "expiry", t, r, q, v)["price"],
                               rel=1e-10)



def test_a_strike_below_the_corridor_pays_the_lower_barriers_call_and_the_rest_in_cash():
    """Inside the corridor S_T - K = (S_T - L) + (L - K): the call at K < L is the call at L plus (L - K) e^{-rT}
    times the survival odds; the put at K > U alike."""
    s, t, r, q, v, low, high = 100.0, 0.75, 0.04, 0.01, 0.25, 80.0, 125.0
    at_low = qf.double_barrier_price("call", "out", s, low, low, high, t, r, q, v, 0.0, 0.0, 5)
    below = qf.double_barrier_price("call", "out", s, 60.0, low, high, t, r, q, v, 0.0, 0.0, 5)
    assert below["price"] == pytest.approx(at_low["price"] + 20.0 * math.exp(-r * t) * at_low["survival"], rel=1e-12)
    at_high = qf.double_barrier_price("put", "out", s, high, low, high, t, r, q, v, 0.0, 0.0, 5)
    above = qf.double_barrier_price("put", "out", s, 140.0, low, high, t, r, q, v, 0.0, 0.0, 5)
    assert above["price"] == pytest.approx(at_high["price"] + 15.0 * math.exp(-r * t) * at_high["survival"],
                                           rel=1e-12)

def test_equal_curvatures_are_flat_barriers_under_a_lower_carry():
    """Barriers L e^{dt}, U e^{dt}: S e^{-dt} sees flat ones with carry b - d, so the price is e^{dT} times the flat
    price at strike K e^{-dT} and yield q + d."""
    s, k, t, r, q, v, d = 100.0, 100.0, 0.5, 0.05, 0.01, 0.25, 0.08
    for option in ("call", "put"):
        curved = qf.double_barrier_price(option, "out", s, k, 75.0, 125.0, t, r, q, v, d, d, 5)
        flat = qf.double_barrier_price(option, "out", s, k * math.exp(-d * t), 75.0, 125.0, t, r, q + d, v, 0.0, 0.0,
                                       5)
        assert curved["price"] == pytest.approx(math.exp(d * t) * flat["price"], rel=1e-12)
        assert curved["survival"] == pytest.approx(flat["survival"], rel=1e-12)



def test_unequal_curvatures_lie_within_a_bridge_monte_carlo():
    """Barriers L e^{delta2 t} and U e^{delta1 t} are straight lines in ln S: a Brownian bridge crosses a line between
    two steps with probability exp(-2 a0 a1 / (vol^2 dt)), a0 and a1 the distances at the steps' ends, so a coarse
    simulation weighted by the odds of crossing neither line is close to exact."""
    s, k, low, high, t, r, q, v, up_curve, low_curve = 100.0, 100.0, 80.0, 120.0, 0.5, 0.05, 0.0, 0.25, 0.1, -0.1
    found = qf.double_barrier_price("call", "out", s, k, low, high, t, r, q, v, low_curve, up_curve, 5)
    rng, n, steps = np.random.default_rng(3), 40_000, 25
    dt, drift = t / steps, r - q - v * v / 2
    x, alive = np.zeros(n), np.ones(n)
    for i in range(steps):
        nxt = x + drift * dt + v * math.sqrt(dt) * rng.standard_normal(n)
        up = [math.log(high / s) + up_curve * j * dt - y for j, y in ((i, x), (i + 1, nxt))]
        down = [y - math.log(low / s) - low_curve * j * dt for j, y in ((i, x), (i + 1, nxt))]
        cross_up = np.exp(-2 * np.maximum(up[0], 0) * np.maximum(up[1], 0) / (v * v * dt))
        cross_down = np.exp(-2 * np.maximum(down[0], 0) * np.maximum(down[1], 0) / (v * v * dt))
        alive = alive * (up[1] > 0) * (down[1] > 0) * (1 - cross_up) * (1 - cross_down)
        x = nxt
    paid = math.exp(-r * t) * alive * np.maximum(s * np.exp(x) - k, 0.0)
    assert abs(paid.mean() - found["price"]) < 4 * paid.std() / math.sqrt(n) + 0.01
    assert abs(alive.mean() - found["survival"]) < 4 * alive.std() / math.sqrt(n) + 0.005

def test_the_survival_odds_are_a_double_no_touch_and_the_series_has_converged():
    """With the upper barrier far, the odds of staying inside are the down no-touch's (Reiner and Rubinstein) over the
    discount factor; a narrower corridor keeps fewer paths; five terms each way have converged."""
    s, t, r, q, v = 100.0, 1.0, 0.03, 0.01, 0.2
    found = qf.double_barrier_price("call", "out", s, 100.0, 80.0, 120.0, t, r, q, v, 0.0, 0.0, 5)
    more = qf.double_barrier_price("call", "out", s, 100.0, 80.0, 120.0, t, r, q, v, 0.0, 0.0, 12)
    assert found["price"] == pytest.approx(more["price"], rel=1e-14) and found["survival"] == pytest.approx(
        more["survival"], rel=1e-14)
    far = qf.double_barrier_price("call", "out", s, 100.0, 80.0, 1e6, t, r, q, v, 0.0, 0.0, 5)["survival"]
    no_touch = qf.one_touch_price("no", "expiry", "cash", "down", s, 80.0, t, r, q, v)
    assert far == pytest.approx(no_touch * math.exp(r * t), rel=1e-11)
    narrow = qf.double_barrier_price("call", "out", s, 100.0, 90.0, 110.0, t, r, q, v, 0.0, 0.0, 5)["survival"]
    assert 0 < narrow < found["survival"] < far < 1
    wide = qf.double_barrier_price("call", "out", s, 100.0, 1e-3, 1e6, t, r, q, v, 0.0, 0.0, 5)["survival"]
    assert wide == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------------------------------------ one-touch

@pytest.mark.parametrize("touch, pay, payout, way, spot, scale, expected", HAUG_TOUCH)
def test_the_touch_options_give_haugs_table(touch, pay, payout, way, spot, scale, expected):
    found = qf.one_touch_price(touch, pay, payout, way, spot, 100.0, 0.5, 0.1, 0.0, 0.2)
    assert scale * found == pytest.approx(expected, abs=1e-4)


def test_one_touch_plus_no_touch_is_the_discount_factor_and_the_hit_pays_more():
    t, r, q, v = 0.75, 0.04, 0.01, 0.15
    for way, spot in (("down", 1.10), ("up", 0.95)):
        one = qf.one_touch_price("one", "expiry", "cash", way, spot, 1.0, t, r, q, v)
        no = qf.one_touch_price("no", "expiry", "cash", way, spot, 1.0, t, r, q, v)
        assert one + no == pytest.approx(math.exp(-r * t), rel=1e-14)
        assert qf.one_touch_price("one", "hit", "cash", way, spot, 1.0, t, r, q, v) > one
        asset = qf.one_touch_price("one", "expiry", "asset", way, spot, 1.0, t, r, q, v) + qf.one_touch_price(
            "no", "expiry", "asset", way, spot, 1.0, t, r, q, v)
        assert asset == pytest.approx(spot * math.exp(-q * t), rel=1e-14)
    assert qf.one_touch_price("one", "hit", "cash", "down", 99.0, 100.0, t, r, q, v) == 1.0
    assert qf.one_touch_price("no", "expiry", "cash", "up", 101.0, 100.0, t, r, q, v) == 0.0


def test_the_no_touch_paid_at_expiry_is_the_barrier_digital_with_its_strike_at_the_barrier():
    """The class keyed by qf.barrier_digital_price in docs/research/selection-equivalents-1011.toml: paid at expiry,
    cash, the strike at the barrier."""
    no = qf.one_touch_price("no", "expiry", "cash", "down", 105.0, 100.0, 0.5, 0.05, 0.0, 0.2)
    digital = qf.barrier_digital_price("cash", "call", "down-and-out", 105.0, 100.0, 100.0, 0.5, 0.05, 0.0, 0.2)
    assert no == pytest.approx(digital, rel=1e-13)


# ------------------------------------------------------------------------------------------------ partial-time

def haug_type_a(s, k, h, t1, t, r, b, v, down):
    """Haug (2007)'s type A out call (cdoA, cuoA), with the exact bivariate normal of qf._common."""
    eta, sd, sd1 = (1.0 if down else -1.0), v * math.sqrt(t), v * math.sqrt(t1)
    mu = (b - v * v / 2) / (v * v)
    d1 = (math.log(s / k) + (b + v * v / 2) * t) / sd
    f1 = d1 + 2 * math.log(h / s) / sd
    e1 = (math.log(s / h) + (b + v * v / 2) * t1) / sd1
    e3 = e1 + 2 * math.log(h / s) / sd1
    rho, m = math.sqrt(t1 / t), bivariate_normal_cdf
    return (s * math.exp((b - r) * t) * (m(d1, eta * e1, eta * rho) - (h / s) ** (2 * (mu + 1)) * m(f1, eta * e3, eta * rho))
            - k * math.exp(-r * t) * (m(d1 - sd, eta * (e1 - sd1), eta * rho)
                                      - (h / s) ** (2 * mu) * m(f1 - sd, eta * (e3 - sd1), eta * rho)))


@pytest.mark.parametrize("k, h, t1", [(95.0, 90.0, 0.2), (110.0, 90.0, 0.6), (95.0, 120.0, 0.6), (110.0, 120.0, 0.2)])
def test_type_a_equals_haugs_closed_form(k, h, t1):
    s, t, r, q, v = 100.0, 1.0, 0.05, 0.02, 0.25
    kind = "down-and-out" if h < s else "up-and-out"
    found = qf.partial_barrier_price("call", kind, "A", s, k, h, t1, t, r, q, v)
    assert found == pytest.approx(haug_type_a(s, k, h, t1, t, r, r - q, v, h < s), rel=1e-11)


@pytest.mark.parametrize("args, expected", QL["partial"])
def test_the_partial_barriers_lie_within_quantlibs_engine(args, expected):
    """QuantLib's engine takes Haug's formulas with an approximate bivariate normal: agreement to 5e-5."""
    option, kind, case, s, k, h, d1, d, r, q, v = args
    assert qf.partial_barrier_price(option, kind, case, s, k, h, d1 / 365, d / 365, r, q, v) == pytest.approx(
        expected, abs=5e-5)


def test_the_partial_barriers_limits_and_parities():
    s, k, t, r, q, v = 100.0, 105.0, 1.0, 0.05, 0.02, 0.25
    for kind, h in (("down-and-out", 90.0), ("up-and-out", 120.0), ("down-and-in", 90.0), ("up-and-in", 120.0)):
        for option in ("call", "put"):
            whole = qf.partial_barrier_price(option, kind, "A", s, k, h, t, t, r, q, v)
            assert whole == pytest.approx(qf.barrier_price(option, kind, s, k, h, 0.0, "expiry", t, r, q, v)["price"],
                                          rel=1e-10, abs=1e-12)
    vanilla = qf.bs_price("put", s, k, t, r, q, v)
    for case in ("A", "B1", "B2"):
        out = qf.partial_barrier_price("put", "down-and-out", case, s, k, 90.0, 0.4, t, r, q, v)
        inn = qf.partial_barrier_price("put", "down-and-in", case, s, k, 90.0, 0.4, t, r, q, v)
        assert out + inn == pytest.approx(vanilla, rel=1e-13)
    early = [qf.partial_barrier_price("call", "down-and-out", "A", s, k, 90.0, t1, t, r, q, v) for t1 in (0.1, 0.5, 0.9)]
    assert early[0] > early[1] > early[2]


@pytest.mark.parametrize("life", [1e-4, 1e-8])
def test_short_remaining_life_resolves_the_partial_barriers_strike_transition(life):
    """Reiner-Rubinstein's full barrier and Black-Scholes are the limits here. The barrier and strike are separated
    by over fifty remaining-life log-price standard deviations, so a touch followed by an in-the-money expiry has
    negligible probability. Reviewer-chosen inputs, unrelated to a unit."""
    s, k, t, r, q, v = 100.0, 105.0, 1.0, 0.05, 0.02, 0.25
    for option, kind, h in (("call", "down-and-out", 90.0), ("put", "up-and-out", 120.0)):
        whole = qf.barrier_price(option, kind, s, k, h, 0.0, "expiry", t, r, q, v)["price"]
        early = qf.partial_barrier_price(option, kind, "A", s, k, h, t - life, t, r, q, v)
        assert early == pytest.approx(whole, abs=5e-12, rel=0)
        for case in ("B1", "B2"):
            late = qf.partial_barrier_price(option, kind, case, s, k, h, t - life, t, r, q, v)
            assert late == pytest.approx(qf.bs_price(option, s, k, t, r, q, v), abs=5e-12, rel=0)


@pytest.mark.parametrize("option, kind, h, case, reference", [
    ("call", "up-and-out", 120.0, "B1", 8.940482019718957),
    ("call", "up-and-out", 120.0, "B2", 1.2643724448014726),
    ("put", "down-and-out", 90.0, "B1", 10.799565058451233),
    ("put", "down-and-out", 90.0, "B2", 1.7429056515416665),
])
def test_short_remaining_life_resolves_the_partial_barriers_touch_transition(option, kind, h, case, reference):
    """Rich (1994), reflected Gaussian transition density: independently integrate its cash and share moments over
    the payoff's interval, then adaptively integrate over the monitoring-start price. These reviewer-chosen cases
    leave a nonzero payoff at the barrier, so the narrow touch transition matters. The independent derivation is in
    scratch/lib-options1/review/barriers/late_quadrature.py; it uses no library barrier formula."""
    found = qf.partial_barrier_price(option, kind, case, 100.0, 105.0, h, 1.0 - 1e-8, 1.0, 0.05, 0.02, 0.25)
    assert found == pytest.approx(reference, abs=5e-12, rel=0)


def test_the_barrier_conventions_are_refused_when_unnamed_or_unfit():
    with pytest.raises(ValueError):
        qf.one_touch_price("no", "hit", "cash", "down", 105.0, 100.0, 0.5, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError):
        qf.partial_barrier_price("call", "down-and-out", "C", 100.0, 100.0, 90.0, 0.5, 1.0, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError):
        qf.partial_barrier_price("call", "down-and-out", "B1", 100.0, 100.0, 90.0, 1.0, 1.0, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError):
        qf.double_barrier_price("call", "knock", 100.0, 100.0, 90.0, 110.0, 1.0, 0.05, 0.0, 0.2, 0.0, 0.0, 5)
    with pytest.raises(ValueError):
        qf.double_barrier_price("call", "out", 100.0, 100.0, 110.0, 90.0, 1.0, 0.05, 0.0, 0.2, 0.0, 0.0, 5)


def quantlib_values() -> dict:
    """The frozen `QL` values recomputed by QuantLib (scratch: a Python with QuantLib 1.43)."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    count = ql.Actual365Fixed()
    kinds = {"down-and-out": ql.Barrier.DownOut, "up-and-out": ql.Barrier.UpOut}
    cases = {"A": ql.PartialBarrier.Start, "B1": ql.PartialBarrier.EndB1, "B2": ql.PartialBarrier.EndB2}
    found = []
    for (option, kind, case, s, k, h, d1, d, r, q, v), _ in QL["partial"]:
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(s)), ql.YieldTermStructureHandle(ql.FlatForward(today, q, count)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), v, count)))
        made = ql.PartialTimeBarrierOption(kinds[kind], cases[case], h, 0.0, today + d1, ql.PlainVanillaPayoff(
            ql.Option.Call if option == "call" else ql.Option.Put, k), ql.EuropeanExercise(today + d))
        made.setPricingEngine(ql.AnalyticPartialTimeBarrierOptionEngine(process))
        found.append(((option, kind, case, s, k, h, d1, d, r, q, v), made.NPV()))
    return {"partial": found}


if __name__ == "__main__":
    fresh = quantlib_values()
    print("QL frozen values match QuantLib:", all(
        np.allclose([v for _, v in fresh[key]], [v for _, v in QL[key]], rtol=1e-14, atol=0) for key in QL))
