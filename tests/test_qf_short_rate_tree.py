"""agent/qf/rates.py's short_rate_tree (runs/lanes/lib6-prompt.md, rank 8 of docs/research/library-syllabus-1011.md):
Hull's worked Hull-White tree, the curve's zero-coupon bonds repriced by every model, zero-coupon bond options against
`hull_white_zcb_option`'s closed form and Ho-Lee's, a European swaption against `jamshidian_swaption`, and QuantLib
1.43's tree engines on a curve of this file's own (values frozen in `QL`, recomputed by
`scratch/lib2/venv/bin/python tests/test_qf_short_rate_tree.py`). No unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf
import qf.rates as r

pytestmark = pytest.mark.fast

TIMES = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
ZEROS = np.array([0.030, 0.032, 0.035, 0.037, 0.040, 0.042, 0.044])
CURVE = (TIMES, np.exp(-ZEROS * TIMES))
A, SIGMA, BK_SIGMA = 0.08, 0.012, 0.2
SWAP_PAY, SWAP_K = [3.0, 4.0, 5.0], 0.045
QL = {   # QuantLib 1.43 (quantlib_values below): TreeSwaptionEngine and TreeCallableFixedRateBondEngine, 20 steps a year
    "hw_european": 0.013928425938653417, "hw_bermudan": 0.01704210207152144,
    "bk_european": 0.009840104366478012, "bk_bermudan": 0.012213494555675835, "hw_callable": 101.71327857080952,
    "hw_fast_bermudan": 0.009404266590445557}   # mean reversion 0.5 and steps of half a year: the nearest-node branching at the edges


def discount(t):
    return float(qf.curve_discount(t, CURVE[0], CURVE[1], "linear_zero"))


def tree(model="hull_white", a=A, sigma=SIGMA, dt=0.05, discretization="exact", branching="nearest",
         compounding="continuous", flow_times=(5.0,), flows=(1.0,), exercise="none", exercise_times=None, strikes=None,
         exercise_flows="after", oas=0.0, market_price=None):
    return r.short_rate_tree(model, a, sigma, dt, CURVE, "linear_zero", discretization, branching, compounding,
                             list(flow_times), list(flows), exercise, exercise_times, strikes, exercise_flows, oas,
                             market_price)


def hull_probabilities(j, m, top):
    """Hull's branching (Hull and White 1994, as Hull writes it): (down, middle, up) and the middle node, normal
    inside jmax, down at +jmax, up at -jmax."""
    if j == top:
        return (1 / 6 + (j * j * m * m + j * m) / 2, -1 / 3 - j * j * m * m - 2 * j * m,
                7 / 6 + (j * j * m * m + 3 * j * m) / 2), j - 1
    if j == -top:
        return (7 / 6 + (j * j * m * m - 3 * j * m) / 2, -1 / 3 - j * j * m * m + 2 * j * m,
                1 / 6 + (j * j * m * m - j * m) / 2), j + 1
    return (1 / 6 + (j * j * m * m - j * m) / 2, 2 / 3 - j * j * m * m, 1 / 6 + (j * j * m * m + j * m) / 2), j


def test_hulls_tree_has_his_shifts_and_his_edges():
    """Hull, "Interest rate trees" (Hull and White 1994): a = 0.1, sigma = 0.01, steps of a year, the zero curve at
    3.824%, 4.512% and 5.086% (continuous): jmax 2, M = -a dt, and the central nodes alpha 3.824%, 5.205%, 6.252%.
    Two more years on the curve (5.33%, 5.55%) take the tree past its edges: the shifts are those of Hull's
    probabilities run forward here, node by node."""
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    zeros = np.array([0.03824, 0.04512, 0.05086, 0.0533, 0.0555])
    found = r.short_rate_tree("hull_white", 0.1, 0.01, 1.0, (t, np.exp(-zeros * t)), "linear_zero", "euler",
                              "hull_white", "continuous", [5.0], [1.0], "none", None, None, "after", 0.0, None)
    assert found["shifts"][:3] == pytest.approx([0.03824, 0.05205, 0.06252], abs=5e-6)
    dx, top, q, alphas = 0.01 * math.sqrt(3.0), 2, {0: 1.0}, []
    for i in range(5):
        alpha = math.log(sum(v * math.exp(-j * dx) for j, v in q.items()) / math.exp(-zeros[i] * t[i]))
        alphas.append(alpha)
        following = {}
        for j, v in q.items():
            probabilities, middle = hull_probabilities(j, -0.1, top)
            for shift, p in zip((-1, 0, 1), probabilities):
                following[middle + shift] = following.get(middle + shift, 0.0) + v * math.exp(-(alpha + j * dx)) * p
        q = following
    assert found["shifts"] == pytest.approx(alphas, abs=1e-14) and max(q) == top


@pytest.mark.parametrize("model, a, sigma", [("hull_white", A, SIGMA), ("ho_lee", 0.0, SIGMA),
                                             ("black_karasinski", A, BK_SIGMA), ("bdt", 0.0, BK_SIGMA)])
@pytest.mark.parametrize("compounding", ["continuous", "simple"])
def test_the_arrow_debreu_fit_reprices_every_zero_coupon_bond_of_the_curve(model, a, sigma, compounding):
    """sum_j Q_ij d_ij = P(0, t_{i+1}) at every step: each zero-coupon bond on the tree is the curve's discount
    factor, under either branching and discretization."""
    for steps in (1, 6, 23, 40):
        for discretization, branching in (("exact", "nearest"), ("euler", "hull_white")):
            found = tree(model, a, sigma, 0.25, discretization, branching, compounding, [0.25 * steps], [1.0])
            assert found["price"] == pytest.approx(discount(0.25 * steps), abs=1e-10)
            assert found["straight"] == found["price"]


def test_a_zero_coupon_bond_option_converges_to_the_hull_white_closed_form():
    """Jamshidian (1989): the call on the 5-year zero expiring in 2 years, struck at 0.88, and the put at the money
    forward, against hull_white_zcb_option."""
    for option, strike in (("call", 0.88), ("put", discount(5.0) / discount(2.0))):
        closed = r.hull_white_zcb_option(option, 2.0, 5.0, strike, A, SIGMA, discount(2.0), discount(5.0))
        found = tree(dt=0.0125, exercise=option, exercise_times=[2.0], strikes=[strike])
        assert found["price"] == pytest.approx(closed, rel=2e-3)


def test_ho_lee_is_the_trinomial_without_mean_reversion_and_meets_its_closed_form():
    """Ho and Lee (1986); Hull, "Interest rate derivatives: models of the short rate": the bond option's vol is
    sigma (S - T) sqrt(T), Hull-White's at a -> 0."""
    strike, big_t, s = 0.88, 2.0, 5.0
    vol = SIGMA * (s - big_t) * math.sqrt(big_t)
    h = math.log(discount(s) / (discount(big_t) * strike)) / vol + vol / 2
    closed = discount(s) * norm.cdf(h) - strike * discount(big_t) * norm.cdf(h - vol)
    found = tree("ho_lee", 0.0, dt=0.025, exercise="call", exercise_times=[2.0], strikes=[strike])
    assert found["price"] == pytest.approx(closed, rel=3e-3)
    near = r.hull_white_zcb_option("call", big_t, s, strike, 1e-6, SIGMA, discount(big_t), discount(s))
    assert near == pytest.approx(closed, rel=1e-5)


def test_a_european_swaption_on_the_tree_is_jamshidians_and_the_bermudan_is_worth_more():
    """A payer swaption is a put on the fixed leg's coupon bond struck at 1; exercisable once, Jamshidian's closed
    form; at every reset, at least that."""
    forward = 0.035 + 2.0 * (0.037 - 0.035)            # f(0, 2) on the linear zero curve: z + t z'
    flows = [SWAP_K, SWAP_K, 1.0 + SWAP_K]
    closed = r.jamshidian_swaption("payer", SWAP_K, 2.0, SWAP_PAY, [1.0] * 3, A, SIGMA, discount(2.0),
                                   [discount(t) for t in SWAP_PAY], forward)["price"]
    european = tree(dt=0.025, flow_times=SWAP_PAY, flows=flows, exercise="put", exercise_times=[2.0], strikes=[1.0])
    bermudan = tree(dt=0.025, flow_times=SWAP_PAY, flows=flows, exercise="put", exercise_times=[2.0, 3.0, 4.0],
                    strikes=[1.0] * 3)
    assert european["price"] == pytest.approx(closed, rel=2e-3) and bermudan["price"] > european["price"] + 1e-3


def test_quantlibs_tree_engines_give_the_same_prices():
    """QuantLib's HullWhite and BlackKarasinski trees (TrinomialTree: the exact OU step, each node branching to the
    node nearest its mean, the shift fitted step by step) on the same curve and grid."""
    flows = [SWAP_K, SWAP_K, 1.0 + SWAP_K]
    for name, model, a, sigma in (("hw", "hull_white", A, SIGMA), ("bk", "black_karasinski", A, BK_SIGMA)):
        for kind, dates in (("european", [2.0]), ("bermudan", [2.0, 3.0, 4.0])):
            found = tree(model, a, sigma, flow_times=SWAP_PAY, flows=flows, exercise="put", exercise_times=dates,
                         strikes=[1.0] * len(dates))
            assert found["price"] == pytest.approx(QL[f"{name}_{kind}"], abs=1e-10)
    fast = tree(a=0.5, dt=0.5, flow_times=SWAP_PAY, flows=flows, exercise="put", exercise_times=[2.0, 3.0, 4.0],
                strikes=[1.0] * 3)
    assert fast["price"] == pytest.approx(QL["hw_fast_bermudan"], abs=1e-10)
    callable_ = tree(flow_times=[1.0, 2.0, 3.0, 4.0, 5.0], flows=[5.0] * 4 + [105.0], exercise="callable",
                     exercise_times=[2.0, 3.0, 4.0], strikes=[100.0] * 3)
    assert callable_["price"] == pytest.approx(QL["hw_callable"], abs=1e-9)


def test_a_callable_is_the_bond_less_the_issuers_option_and_a_puttable_the_bond_plus_the_holders():
    """min(H, K) = B - max(B - K, O) by induction: the callable bond is the straight bond less a Bermudan call on its
    flows; the puttable the straight plus a Bermudan put. A coupon taken by the exercise ('from') at K + c is the
    coupon paid either way ('after') at K."""
    bond = dict(flow_times=[1.0, 2.0, 3.0, 4.0, 5.0], flows=[4.0] * 4 + [104.0], exercise_times=[2.0, 3.0, 4.0])
    callable_ = tree(**bond, exercise="callable", strikes=[100.5, 100.0, 100.0])
    call = tree(**bond, exercise="call", strikes=[100.5, 100.0, 100.0])
    assert callable_["price"] == pytest.approx(callable_["straight"] - call["price"], abs=1e-11)
    puttable = tree(**bond, exercise="puttable", strikes=[99.0, 99.0, 99.5])
    put = tree(**bond, exercise="put", strikes=[99.0, 99.0, 99.5])
    assert puttable["price"] == pytest.approx(puttable["straight"] + put["price"], abs=1e-11)
    assert callable_["price"] < callable_["straight"] < puttable["price"]
    taken = tree(**bond, exercise="callable", strikes=[104.5, 104.0, 104.0], exercise_flows="from")
    assert taken["price"] == pytest.approx(callable_["price"], abs=1e-11)


def test_an_option_free_bonds_oas_is_its_z_spread_and_a_price_gives_back_its_oas():
    """On continuously compounded node rates a constant spread s discounts each flow by e^{-st} more: the OAS of a
    bond with no option is its z-spread over the zero curve (Tuckman and Serrat)."""
    flows = dict(flow_times=[1.0, 2.0, 3.0, 4.0, 5.0], flows=[4.0] * 4 + [104.0])
    price = 98.2
    found = tree(**flows, oas=None, market_price=price)
    times = np.array(flows["flow_times"])
    zeros = -np.log([discount(t) for t in times]) / times
    assert found["oas"] == pytest.approx(qf.z_spread(price, flows["flows"], times, times, zeros, "linear_zero",
                                                    "continuous"), abs=1e-10)
    assert found["price"] == pytest.approx(price, abs=1e-9)
    callable_ = dict(**flows, exercise="callable", exercise_times=[2.0, 3.0], strikes=[100.0, 100.0])
    at = tree(**callable_, oas=0.0125)
    assert tree(**callable_, oas=None, market_price=at["price"])["oas"] == pytest.approx(0.0125, abs=1e-10)


def test_black_derman_toys_first_steps_by_hand():
    """Two annual steps on a flat 5% curve compounded per step, sigma 20%: r_0 = 5%; r at step 1 is U e^{+-0.2} with
    1/2 / 1.05 (1 / (1 + U e^0.2) + 1 / (1 + U e^-0.2)) = 1.05^-2, solved here."""
    flat = (np.array([1.0, 2.0]), np.array([1.05 ** -1, 1.05 ** -2]))
    found = r.short_rate_tree("bdt", 0.0, 0.2, 1.0, flat, "log_linear_discount", "exact", "nearest", "simple", [2.0],
                              [1.0], "none", None, None, "after", 0.0, None)
    u = brentq(lambda u: 0.5 / 1.05 * (1 / (1 + u * math.exp(0.2)) + 1 / (1 + u * math.exp(-0.2))) - 1.05 ** -2,
               0.001, 0.5)
    assert math.exp(found["shifts"][0]) == pytest.approx(0.05, abs=1e-14)
    assert math.exp(found["shifts"][1]) == pytest.approx(u, abs=1e-12)


def test_the_conventions_change_the_tree():
    base = tree(exercise="call", exercise_times=[2.0], strikes=[0.88])["price"]
    for change in (dict(discretization="euler"), dict(branching="hull_white"), dict(compounding="simple")):
        assert tree(exercise="call", exercise_times=[2.0], strikes=[0.88], **change)["price"] != base


@pytest.mark.parametrize("bad, match", [
    (dict(model="cir"), "model"), (dict(model="ho_lee"), "mean_reversion"), (dict(a=0.0), "mean_reversion"),
    (dict(flow_times=[5.01]), "steps"), (dict(exercise="call"), "exercise_times"),
    (dict(exercise_times=[2.0], strikes=[1.0]), "None"), (dict(oas=None), "market_price"),
    (dict(exercise="bermudan"), "exercise"), (dict(flow_times=[12.0]), "last node")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises((ValueError, TypeError), match=match):
        tree(**bad)


def quantlib_values():
    """QL recomputed with QuantLib 1.43: `scratch/lib2/venv/bin/python tests/test_qf_short_rate_tree.py`. Whole
    years under SimpleDayCounter; the zero curve linear in the rate, flat before its first node."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc, cal = ql.SimpleDayCounter(), ql.NullCalendar()
    times, zeros = [0.0, *TIMES], [ZEROS[0], *ZEROS]
    curve = ql.ZeroCurve([today + ql.Period(int(round(t * 12)), ql.Months) for t in times], zeros, dc, cal,
                         ql.Linear(), ql.Continuous)
    handle = ql.YieldTermStructureHandle(curve)
    values = {}
    for name, model in (("hw", ql.HullWhite(handle, A, SIGMA)), ("bk", ql.BlackKarasinski(handle, A, BK_SIGMA))):
        fixed = ql.Schedule(today + ql.Period(2, ql.Years), today + ql.Period(5, ql.Years), ql.Period(1, ql.Years), cal,
                            ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
        index = ql.IborIndex("idx", ql.Period(1, ql.Years), 0, ql.USDCurrency(), cal, ql.Unadjusted, False, dc, handle)
        swap = ql.VanillaSwap(ql.Swap.Payer, 1.0, fixed, SWAP_K, dc, fixed, index, 0.0, dc)
        for kind, exercise in (("european", ql.EuropeanExercise(fixed[0])),
                               ("bermudan", ql.BermudanExercise([today + ql.Period(y, ql.Years) for y in (2, 3, 4)]))):
            option = ql.Swaption(swap, exercise)
            option.setPricingEngine(ql.TreeSwaptionEngine(model, 100))
            values[f"{name}_{kind}"] = option.NPV()
        if name == "hw":
            fast = ql.Swaption(swap, ql.BermudanExercise([today + ql.Period(y, ql.Years) for y in (2, 3, 4)]))
            fast.setPricingEngine(ql.TreeSwaptionEngine(ql.HullWhite(handle, 0.5, SIGMA), 10))
            values["hw_fast_bermudan"] = fast.NPV()
            schedule = ql.Schedule(today, today + ql.Period(5, ql.Years), ql.Period(1, ql.Years), cal, ql.Unadjusted,
                                   ql.Unadjusted, ql.DateGeneration.Backward, False)
            calls = ql.CallabilitySchedule()
            for y in (2, 3, 4):
                calls.append(ql.Callability(ql.BondPrice(100.0, ql.BondPrice.Clean), ql.Callability.Call,
                                            today + ql.Period(y, ql.Years)))
            bond = ql.CallableFixedRateBond(0, 100.0, schedule, [0.05], dc, ql.Unadjusted, 100.0, today, calls)
            bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(model, 100))
            values["hw_callable"] = bond.cleanPrice()
    return values


@pytest.mark.slow
def test_the_frozen_values_are_quantlibs():
    pytest.importorskip("QuantLib")
    assert quantlib_values() == pytest.approx(QL, rel=1e-12)


if __name__ == "__main__":
    print(quantlib_values())
