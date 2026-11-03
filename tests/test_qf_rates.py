"""agent/qf/rates.py, the rates and FX shelf (runs/lanes/lib2-prompt.md): every function against QuantLib 1.43 (its
bootstrap, bond functions, Hull-White model, Jamshidian and analytic cap engines; values frozen in `QL`, recomputed and
compared by running this file under a Python with QuantLib, scratch/lib2/venv), a worked example in Hull, or an
identity the method must satisfy (a par curve's closed form, the forward-measure expectation of a Hull-White payoff,
parity), each source named in its test. No unit's instruction values, checks, expected values or data appear here
(rules 8 and 9): the parameters are textbook examples or chosen freely."""
import math

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf
import qf.rates as r

pytestmark = pytest.mark.fast

PAR_T = [0.25, 0.5, 1.0, 1.25, 2.0, 3.0, 5.0, 7.0, 10.0]
PAR_C = [0.0412, 0.0418, 0.0425, 0.0427, 0.0431, 0.0436, 0.0443, 0.0450, 0.0458]
OFF = [0.1, 0.4, 0.8, 1.6, 2.7, 4.2, 9.3]
BOND_DAYS = np.array([183, 365, 548, 730, 913, 1095])
BOND_CF = [2.25] * 5 + [102.25]
A, SIGMA = 0.08, 0.012
SWAP_T0 = 546 / 365          # 2025-01-15 to 2026-07-15 on ACT/365F, the swaption's expiry
SWAP_PAY = [2.0, 2.495890410958904, 3.0, 3.4986301369863013, 4.002739726027397]
SWAP_ACC = [0.5041095890410959, 0.4958904109589041, 0.5041095890410959, 0.4986301369863014, 0.5041095890410959]

QL = {   # QuantLib 1.43 (quantlib_values below)
    # PiecewiseLinearZero / PiecewiseLogLinearDiscount on 30/360 USA: 3M and 6M DepositRateHelpers, then
    # FixedRateBondHelpers at par (semi-annual, backward schedules, a 3-month front stub at 15 months)
    "linear_zero": dict(
        nodes=[0.9898050084133881, 0.9795278675678554, 0.9588103136491316, 0.9484911748359952, 0.9182198054681163,
               0.8785366325414857, 0.8029632806748084, 0.7316262983053453, 0.6342487852056822],
        off=[0.9959094625528937, 0.983648278573039, 0.9671246308653817, 0.9343038688340781, 0.8903527757491433,
             0.8326685384081847, 0.656055848836287]),
    "log_linear_discount": dict(
        nodes=[0.9898050084133425, 0.9795278675678972, 0.9588103136491379, 0.948492929355132, 0.9182209374955353,
               0.8785401305979699, 0.8029824631360456, 0.7316594569162705, 0.6343157042197438],
        off=[0.9959094625528753, 0.98362585657123, 0.9670442270099193, 0.9342437650278229, 0.890260860508952,
             0.832392729147709, 0.6558023268214002]),
    # CashFlows on SimpleCashFlows at BOND_DAYS, ACT/365F, yield 5.15%: npv, Macaulay, modified, convexity; yieldRate
    # at a price of 98.7
    "bond": {2: (98.21390480366993, 2.8380906813125137, 2.7668444370582637, 9.264236900216094, 0.04971652982979401),
             1: (98.38978190704654, 2.8382631131018607, 2.699251652973714, 10.100266139246905, 0.050334463164373074),
             4: (98.12386362338798, 2.8380022064174044, 2.8019273912549965, 8.809296399265241, 0.04941134470657961),
             "continuous": (98.03236921621584, 2.8379121660853626, 2.8379121660853626, 8.327742357109232, None)},
    # HullWhite(a 0.08, sigma 0.012) on a DiscountCurve (log-linear, ACT/365F): (t, T, r, P(0,t), P(0,T), f(0,t)) ->
    # discountBond; (T, S, X, P(0,T), P(0,S)) -> discountBondOption call, put
    "zcb": [((0.0, 3.0, 0.04010717568686616, 1.0, 0.883, 0.04010717568686616), 0.8829999999999999),
            ((1.3, 4.6, 0.031, 0.9488371883997404, 0.8245847067063706, 0.041763493657351165), 0.8959680529769525),
            ((2.5, 9.0, 0.047, 0.9020446219561424, 0.6780974154076139, 0.042677576506489415), 0.7326461661863951)],
    "zbo": [((1.5, 4.0, 0.91, 0.9409448442921614, 0.8460232752476984), 0.006315816299710786, 0.016552349357879304),
            ((3.2, 3.7, 0.975, 0.8754775726260343, 0.8569506823734719), 0.005134626148066146, 0.001774577084977691)],
    # JamshidianSwaptionEngine, 18 months into 30 months semi-annual (fixed leg ACT/365F), notional 1: strike ->
    # payer, receiver
    "swaption": {"P0": 0.9411063532641949, "f0": 0.04176349365513074,
                 "Pi": [0.9215, 0.9022028430020822, 0.883, 0.864364570518016, 0.8459241263669324],
                 0.037: (0.01915042060923967, 0.005904554823745165), 0.042: (0.01245153251379624, 0.010278145425938369),
                 0.05: (0.005237966638212704, 0.02078055054301313)},
    # AnalyticCapFloorEngine(HullWhite): one 6-month IborLeg period fixing at 1y: strike -> caplet, floorlet
    "hw_cap": {"T2": 1.4958904109589042, "P1": 0.9608, "P2": 0.9411063532641949,
               0.03: (0.006092116840379066, 0.00039903859286047664), 0.045: (0.0015604898823229354,
                                                                              0.0028676958789475827)},
}


def test_the_rates_cards_keep_the_format_and_their_examples_run():
    check_cards(r)


# ------------------------------------------------------------------------------------------------ curves


@pytest.mark.parametrize("interpolation", ["linear_zero", "log_linear_discount"])
def test_the_bootstrap_equals_quantlibs_piecewise_curve_at_the_nodes_and_between(interpolation):
    """QuantLib 1.43's PiecewiseLinearZero and PiecewiseLogLinearDiscount on the same quotes (deposits up to 6 months,
    par bonds beyond, a 15-month bond with a 3-month stub), its bootstrap accurate to 1e-12; both root-finders give
    the same nodes; curve_discount reads QuantLib's curve between them."""
    found = {solver: qf.bootstrap_par_curve(maturities=PAR_T, par_rates=PAR_C, frequency=2,
                                           interpolation=interpolation, solver=solver) for solver in ("brentq", "newton")}
    nodes = found["brentq"]["discount_factors"]
    assert nodes == pytest.approx(QL[interpolation]["nodes"], rel=1e-11)
    assert found["newton"]["discount_factors"] == pytest.approx(nodes, rel=1e-13)
    assert found["brentq"]["zero_rates"] == pytest.approx(-np.log(nodes) / PAR_T, rel=1e-13)
    between = qf.curve_discount(times=OFF, node_times=PAR_T, node_discounts=nodes, interpolation=interpolation)
    assert between == pytest.approx(QL[interpolation]["off"], rel=1e-11)


def par_price(maturity, coupon, frequency, curve):
    """A par instrument's price written out: coupons c * accrual at T, T - 1/f, ... > 0 (the earliest a stub)."""
    times = [maturity - j / frequency for j in range(int(math.ceil(maturity * frequency - 1e-9)))][::-1]
    accruals = np.diff([0.0] + times)
    return coupon * accruals @ curve(times) + curve([maturity])[0]


@pytest.mark.parametrize("interpolation", ["linear_zero", "log_linear_discount", "cubic_log_discount"])
def test_every_curve_reprices_its_quotes_and_annual_pars_have_the_closed_form(interpolation):
    """The defining property: on the final curve every par instrument prices at 1 (for the cubic too, which is solved
    globally); deposits are 1 / (1 + c T). With annual coupons on whole-year maturities no interpolation enters and
    the textbook recursion d_n = (1 - c_n sum_{i<n} d_i) / (1 + c_n) holds (Hull, "Determining zero rates")."""
    curve = qf.bootstrap_par_curve(maturities=PAR_T + [20.0], par_rates=PAR_C + [0.0471], frequency=2,
                                  interpolation=interpolation, solver="brentq")["discount_factors"]

    def read(times):
        return np.asarray(qf.curve_discount(times=times, node_times=PAR_T + [20.0], node_discounts=curve,
                                           interpolation=interpolation))

    for maturity, coupon in zip(PAR_T + [20.0], PAR_C + [0.0471]):
        assert par_price(maturity, coupon, 2, read) == pytest.approx(1.0, abs=1e-13), maturity
    assert curve[0] == pytest.approx(1 / (1 + 0.0412 * 0.25), rel=1e-14)
    years, pars = [1.0, 2.0, 3.0, 4.0, 5.0], [0.031, 0.034, 0.036, 0.037, 0.0375]
    expected = []
    for coupon in pars:
        expected.append((1 - coupon * sum(expected)) / (1 + coupon))
    annual = qf.bootstrap_par_curve(maturities=years, par_rates=pars, frequency=1, interpolation=interpolation,
                                   solver="newton")
    assert annual["discount_factors"] == pytest.approx(expected, rel=1e-13)


def test_the_cubic_curve_is_a_not_a_knot_spline_on_minus_log_discount_through_zero():
    """'cubic_log_discount' is what it names: between nodes -ln D is scipy's not-a-knot cubic spline through (0, 0)
    and the nodes (Hagan and West 2006 on interpolating ln D); it differs from the log-linear curve between nodes."""
    nodes = qf.bootstrap_par_curve(maturities=PAR_T, par_rates=PAR_C, frequency=2, interpolation="cubic_log_discount",
                                  solver="brentq")["discount_factors"]
    spline = CubicSpline([0.0] + PAR_T, np.r_[0.0, -np.log(nodes)], bc_type="not-a-knot")
    found = qf.curve_discount(times=OFF, node_times=PAR_T, node_discounts=nodes, interpolation="cubic_log_discount")
    assert found == pytest.approx(np.exp(-spline(OFF)), rel=1e-14)
    assert not np.allclose(found, QL["log_linear_discount"]["off"], rtol=1e-6)
    assert qf.curve_discount(times=0.0, node_times=PAR_T, node_discounts=nodes, interpolation="linear_zero") == 1.0
    with pytest.raises(ValueError, match="no extrapolation"):
        qf.curve_discount(times=[10.25], node_times=PAR_T, node_discounts=nodes, interpolation="linear_zero")


def test_the_bootstrap_refuses_what_it_cannot_build():
    base = dict(maturities=PAR_T, par_rates=PAR_C, frequency=2, interpolation="linear_zero", solver="brentq")
    for bad, name in ((dict(maturities=PAR_T[::-1]), "increasing"), (dict(par_rates=PAR_C[:-1]), "par_rates"),
                      (dict(frequency=0), "frequency"), (dict(frequency=2.0), "frequency"),
                      (dict(interpolation="spline"), "interpolation"), (dict(solver="bisect"), "solver"),
                      (dict(par_rates=[5.0] * 9), "par_rates")):
        with pytest.raises(ValueError, match=name):
            qf.bootstrap_par_curve(**(base | bad))


# ------------------------------------------------------------------------------------------------ Nelson-Siegel

NS = [0.045, -0.012, 0.008, 1.8]
NSS = [0.045, -0.012, 0.008, -0.006, 1.8, 7.5]


def test_nelson_siegel_yields_are_the_average_of_their_instantaneous_forward():
    """Nelson and Siegel (1987) define the curve by its instantaneous forward b0 + b1 e^{-x} + b2 x e^{-x}, x = T/tau
    (Svensson 1994 adds b3 x2 e^{-x2}); the yield is its average from 0 to T, integrated here. The 'rate' form with
    lambda = 1/tau is the same curve; y(0) = b0 + b1, y(inf) = b0; Svensson with b3 = 0 is Nelson-Siegel."""
    s = np.linspace(0.0, 30.0, 300001)
    x1, x2 = s / NSS[4], s / NSS[5]
    forward = NSS[0] + NSS[1] * np.exp(-x1) + NSS[2] * x1 * np.exp(-x1) + NSS[3] * x2 * np.exp(-x2)
    integral = np.concatenate([[0.0], np.cumsum((forward[1:] + forward[:-1]) / 2 * np.diff(s))])
    for t in (0.5, 2.0, 7.0, 30.0):
        average = integral[int(round(t * 10000))] / t
        assert qf.nelson_siegel_yield(variant="svensson", maturities=t, params=NSS, decay_form="time") == \
            pytest.approx(average, abs=1e-10), t
    t = np.array([0.0, 0.25, 3.0, 12.0, 1e6])
    as_time = qf.nelson_siegel_yield(variant="nelson_siegel", maturities=t, params=NS, decay_form="time")
    as_rate = qf.nelson_siegel_yield(variant="nelson_siegel", maturities=t, params=NS[:3] + [1 / NS[3]],
                                    decay_form="rate")
    assert as_time == pytest.approx(as_rate, rel=1e-13)
    assert as_time[0] == pytest.approx(NS[0] + NS[1]) and as_time[-1] == pytest.approx(NS[0], abs=1e-7)
    assert qf.nelson_siegel_yield(variant="svensson", maturities=t, params=NS[:3] + [0.0, NS[3], 4.0],
                                 decay_form="time") == pytest.approx(as_time, rel=1e-14)
    for bad, name in ((dict(params=NS[:3]), "params"), (dict(params=NS[:3] + [0.0]), "decays"),
                      (dict(variant="nss"), "variant"), (dict(decay_form="tau"), "decay_form")):
        with pytest.raises(ValueError, match=name):
            qf.nelson_siegel_yield(**(dict(variant="nelson_siegel", maturities=t, params=NS, decay_form="time") | bad))


NS_T = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)


def test_nelson_siegel_fit_recovers_its_parameters_and_stops_where_the_gradient_vanishes():
    """Exact model yields are fitted back to their parameters from a start nearby; on noisy yields the fitted point
    is a stationary point of the sum of squares (central differences of it vanish), and rmse is sqrt(SSE / n) of the
    fitted yields. L-BFGS-B runs with ftol 0: on decimal yields its default stops early (the card says why)."""
    t = NS_T
    exact = qf.nelson_siegel_yield(variant="nelson_siegel", maturities=t, params=NS, decay_form="time")
    tight = {"ftol": 0.0, "gtol": 1e-14, "maxiter": 20000}
    fit = qf.nelson_siegel_fit(variant="nelson_siegel", maturities=t, yields=exact, initial=[0.04, -0.01, 0.0, 1.5],
                              decay_form="time", method="L-BFGS-B", options=tight, fixed_decays=None)
    assert fit["params"] == pytest.approx(NS, rel=1e-4) and fit["rmse"] < 1e-7
    noisy = qf.nelson_siegel_yield(variant="nelson_siegel", maturities=t, params=NS, decay_form="rate") + \
        0.0004 * np.sin(np.arange(11.0))
    fit = qf.nelson_siegel_fit(variant="nelson_siegel", maturities=t, yields=noisy, initial=[0.04, -0.01, 0.0, 1.0],
                              decay_form="rate", method="L-BFGS-B", options=tight, fixed_decays=None)

    def sse(p):
        return np.sum((qf.nelson_siegel_yield(variant="nelson_siegel", maturities=t, params=p, decay_form="rate")
                       - noisy) ** 2)

    for j in range(4):
        step = np.zeros(4)
        step[j] = 1e-6
        assert abs(sse(fit["params"] + step) - sse(fit["params"] - step)) / 2e-6 < 1e-6, j
    assert fit["rmse"] == pytest.approx(math.sqrt(sse(fit["params"]) / 11), rel=1e-10)
    assert fit["sse"] == pytest.approx(sse(fit["params"]), rel=1e-10) and fit["success"]
    assert not qf.nelson_siegel_fit(variant="nelson_siegel", maturities=t, yields=noisy, initial=[0.04, -0.01, 0.0, 1.0],
                                   decay_form="rate", method="L-BFGS-B", options={"maxiter": 1},
                                   fixed_decays=None)["success"]
    assert fit["fitted"] == pytest.approx(noisy, abs=0.001)
    with pytest.raises(ValueError, match="method"):
        qf.nelson_siegel_fit(variant="nelson_siegel", maturities=t, yields=noisy, initial=NS, decay_form="rate",
                            method="BFGS", options={}, fixed_decays=None)


@pytest.mark.slow
def test_a_svensson_fit_recovers_its_six_parameters():
    """Svensson's six parameters back from exact yields, from a start nearby, by Powell's method."""
    exact = qf.nelson_siegel_yield(variant="svensson", maturities=NS_T, params=NSS, decay_form="time")
    fit = qf.nelson_siegel_fit(variant="svensson", maturities=NS_T, yields=exact,
                              initial=[0.046, -0.011, 0.007, -0.005, 1.6, 8.0], decay_form="time", method="Powell",
                              options={"xtol": 1e-12, "ftol": 1e-22, "maxiter": 40000}, fixed_decays=None)
    assert fit["params"] == pytest.approx(NSS, rel=1e-6) and fit["rmse"] < 1e-11 and fit["success"]


# ------------------------------------------------------------------------------------------------ bonds


@pytest.mark.parametrize("frequency", [2, 1, 4, "continuous"])
def test_bond_price_yield_and_risk_equal_quantlibs_cashflows(frequency):
    """QuantLib 1.43's CashFlows.npv, duration (Macaulay, Modified), convexity and yieldRate on the same cash flows at
    days / 365 (ACT/365F), the yield compounded at the frequency named."""
    price, macaulay, modified, convexity, at_price = QL["bond"][frequency]
    times = BOND_DAYS / 365
    risk = qf.bond_risk(cash_flows=BOND_CF, times=times, ytm=0.0515, frequency=frequency)
    assert qf.bond_price(cash_flows=BOND_CF, times=times, ytm=0.0515, frequency=frequency) == pytest.approx(price,
                                                                                                         rel=1e-13)
    assert (risk["price"], risk["macaulay_duration"], risk["modified_duration"], risk["convexity"]) == pytest.approx(
        (price, macaulay, modified, convexity), rel=1e-12)
    if at_price is not None:
        assert qf.bond_yield(price=98.7, cash_flows=BOND_CF, times=times, frequency=frequency) == pytest.approx(
            at_price, rel=1e-11)


def test_bond_risk_gives_hulls_duration_example_and_dv01_is_the_price_change():
    """Hull, "Duration" and "Convexity": a 3-year 10% semi-annual bond at a 12% continuously compounded yield is worth
    94.213, with a duration of 2.653 years and a convexity of 7.570; at the same yield written semi-annually (12.3673%) the modified duration is 2.653 /
    1.0618 = 2.499. DV01 is the price change of a one basis point fall, (P(y - h) - P(y + h)) / 2 at h = 1 bp, and the
    yield of a priced bond is the yield it was priced at, negative where the price exceeds the cash flows' sum."""
    flows, times = [5.0] * 5 + [105.0], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    cont = qf.bond_risk(cash_flows=flows, times=times, ytm=0.12, frequency="continuous")
    assert cont["price"] == pytest.approx(94.213, abs=5e-4) and cont["macaulay_duration"] == pytest.approx(2.653,
                                                                                                         abs=5e-4)
    assert cont["convexity"] == pytest.approx(7.570, abs=5e-4)
    semi = qf.bond_risk(cash_flows=flows, times=times, ytm=2 * math.expm1(0.06), frequency=2)
    assert semi["price"] == pytest.approx(cont["price"], rel=1e-13)
    assert semi["macaulay_duration"] == pytest.approx(cont["macaulay_duration"], rel=1e-13)
    assert semi["modified_duration"] == pytest.approx(2.499, abs=5e-4)
    for frequency in (2, "continuous"):
        risk = qf.bond_risk(cash_flows=flows, times=times, ytm=0.05, frequency=frequency)
        up, down = (qf.bond_price(cash_flows=flows, times=times, ytm=0.05 + h, frequency=frequency) for h in (1e-4, -1e-4))
        assert risk["dv01"] == pytest.approx((down - up) / 2, rel=1e-7)
        assert risk["convexity"] == pytest.approx((up + down - 2 * risk["price"]) / (1e-8 * risk["price"]), rel=1e-5)
        assert qf.bond_yield(price=risk["price"], cash_flows=flows, times=times, frequency=frequency) == pytest.approx(
            0.05, abs=1e-13)
    above = qf.bond_yield(price=sum(flows) + 2.0, cash_flows=flows, times=times, frequency=2)
    assert above < 0 and qf.bond_price(cash_flows=flows, times=times, ytm=above, frequency=2) == pytest.approx(
        sum(flows) + 2.0, rel=1e-13)
    with pytest.raises(ValueError, match="price"):
        qf.bond_yield(price=0.0, cash_flows=flows, times=times, frequency=2)


def test_key_rate_durations_tents_sum_to_the_parallel_duration_and_nodes_isolate_their_cash_flow():
    """Ho (1992): the tents sum to a parallel shift, so the tent KRDs sum to the Fisher-Weil duration sum t PV / P (to
    the central difference's O(bump^2)); a 'node' bump moves only the cash flow on that node (its t PV / P); a
    one-sided bump differs from the central one by about half the bump times the convexity."""
    curve_t, curve_z = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], [0.040, 0.041, 0.043, 0.044, 0.046, 0.047, 0.048]
    flows, times = [3.0] * 9 + [103.0], [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8]
    zeros = np.interp(times, curve_t, curve_z)
    pv = np.array(flows) * np.exp(-zeros * times)
    fisher_weil = times @ pv / pv.sum()
    tents = qf.key_rate_durations(cash_flows=flows, times=times, curve_times=curve_t, curve_zero_rates=curve_z,
                                 key_times=[1.0, 3.0, 5.0, 10.0], shift="tent", bump=1e-4, difference="central")
    assert tents.sum() == pytest.approx(fisher_weil, rel=1e-6) and np.all(tents > 0)
    on_nodes = qf.key_rate_durations(cash_flows=[4.0, 4.0, 104.0], times=[1.0, 2.0, 3.0], curve_times=[1.0, 2.0, 3.0],
                                    curve_zero_rates=[0.03, 0.035, 0.04], key_times=[2.0, 3.0], shift="node",
                                    bump=1e-4, difference="central")
    values = np.array([4.0, 4.0, 104.0]) * np.exp(-np.array([0.03, 0.035, 0.04]) * [1.0, 2.0, 3.0])
    assert on_nodes == pytest.approx(np.array([2.0, 3.0]) * values[1:] / values.sum(), rel=1e-6)
    forward = qf.key_rate_durations(cash_flows=flows, times=times, curve_times=curve_t, curve_zero_rates=curve_z,
                                   key_times=[1.0, 3.0, 5.0, 10.0], shift="tent", bump=1e-4, difference="forward")
    assert forward == pytest.approx(tents, rel=1e-3) and not np.allclose(forward, tents, rtol=1e-6)
    with pytest.raises(ValueError, match="among curve_times"):
        qf.key_rate_durations(cash_flows=flows, times=times, curve_times=curve_t, curve_zero_rates=curve_z,
                             key_times=[4.0], shift="node", bump=1e-4, difference="central")


def test_immunization_weights_match_value_duration_and_convexity():
    """Redington (1952): two bonds matching value and duration, a = L (D2 - DL) / (D2 - D1) in the first; with a third
    on convexity the hedge's value tracks the liability's to second order under a parallel yield shift (on a flat
    continuously compounded curve): its gap shrinks as the shift cubed, the duration hedge's as the shift squared."""
    two = qf.immunization_weights(exposures=[[3.0, 12.0]], liability_value=500.0, liability_exposures=[7.5])
    assert two == pytest.approx([500 * (12 - 7.5) / 9, 500 * (7.5 - 3) / 9])
    y = 0.045
    bonds = [([4.0] * 3 + [104.0], [1.0, 2.0, 3.0, 4.0]), ([5.0] * 9 + [105.0], list(range(1, 11))),
             ([100.0], [20.0])]
    liability = ([60.0, 80.0, 100.0], [6.0, 8.0, 12.0])
    risks = [qf.bond_risk(cash_flows=f, times=t, ytm=y, frequency="continuous") for f, t in bonds]
    target = qf.bond_risk(cash_flows=liability[0], times=liability[1], ytm=y, frequency="continuous")
    amounts = qf.immunization_weights(exposures=[[k["modified_duration"] for k in risks],
                                                [k["convexity"] for k in risks]],
                                     liability_value=target["price"],
                                     liability_exposures=[target["modified_duration"], target["convexity"]])

    def gap(shift, weights):
        hedge = sum(w / k["price"] * qf.bond_price(cash_flows=f, times=t, ytm=y + shift, frequency="continuous")
                    for w, k, (f, t) in zip(weights, risks, bonds))
        return hedge - qf.bond_price(cash_flows=liability[0], times=liability[1], ytm=y + shift,
                                    frequency="continuous")

    assert amounts.sum() == pytest.approx(target["price"]) and abs(gap(0.0, amounts)) < 1e-9
    duration_only = list(qf.immunization_weights(
        exposures=[[risks[0]["modified_duration"], risks[1]["modified_duration"]]], liability_value=target["price"],
        liability_exposures=[target["modified_duration"]])) + [0.0]
    h = 1e-4
    assert gap(2 * h, amounts) / gap(h, amounts) == pytest.approx(8.0, rel=0.05)       # a third-order gap
    assert gap(2 * h, duration_only) / gap(h, duration_only) == pytest.approx(4.0, rel=0.05)   # a second-order one
    with pytest.raises(ValueError, match="singular"):
        qf.immunization_weights(exposures=[[5.0, 5.0]], liability_value=100.0, liability_exposures=[5.0])


# ------------------------------------------------------------------------------------------------ Hull-White


def forward_measure(payoff, horizon, forward, points=200001):
    """E^{T}[payoff(r(T))] for the Hull-White short rate: under the T-forward measure r(T) is normal with mean
    f(0, T) (the forward is the forward measure's expected short rate) and variance sigma^2 (1 - e^{-2aT}) / (2a)
    (Brigo and Mercurio, 3.3); integrated on normal scores."""
    z = np.linspace(-11.0, 11.0, points)
    rate = forward + SIGMA * math.sqrt(-math.expm1(-2 * A * horizon) / (2 * A)) * z
    return np.trapezoid(payoff(rate) * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi), z)


def test_hull_white_bonds_equal_quantlib_and_are_forward_martingales():
    """QuantLib 1.43's HullWhite.discountBond at three points (at t = 0, r = f(0, 0) it is today's P(0, T)); and
    P(t, T) / P(t, t) is a martingale under the t-forward measure, so E^t[P(t, T)] = P(0, T) / P(0, t)."""
    for (t, big_t, rate, p_t, p_big, f_t), expected in QL["zcb"]:
        assert r.hull_white_zcb(short_rate=rate, t=t, maturity=big_t, a=A, sigma=SIGMA, discount_t=p_t,
                                discount_maturity=p_big, forward_t=f_t) == pytest.approx(expected, rel=1e-13)
        mean = forward_measure(lambda x: np.asarray(r.hull_white_zcb(
            short_rate=x, t=t, maturity=big_t, a=A, sigma=SIGMA, discount_t=p_t, discount_maturity=p_big,
            forward_t=f_t)), t, f_t) if t > 0 else p_big
        assert mean == pytest.approx(p_big / p_t, rel=1e-12)
    with pytest.raises(ValueError, match="a"):
        r.hull_white_zcb(short_rate=0.03, t=1.0, maturity=2.0, a=0.0, sigma=SIGMA, discount_t=0.97,
                         discount_maturity=0.94, forward_t=0.03)


def test_hull_white_bond_options_equal_quantlib_parity_and_the_forward_measure_integral():
    """QuantLib 1.43's discountBondOption (calls and puts); ZBC - ZBP = P(0, S) - X P(0, T); and the put is
    P(0, T) E^T[(X - P(T, S))^+], integrated over the forward measure's short rate (any f(0, T) gives the same)."""
    for (big_t, s, x, p_t, p_s), call, put in QL["zbo"]:
        args = dict(expiry=big_t, maturity=s, strike=x, a=A, sigma=SIGMA, discount_expiry=p_t, discount_maturity=p_s)
        zbc, zbp = r.hull_white_zcb_option(option="call", **args), r.hull_white_zcb_option(option="put", **args)
        assert (zbc, zbp) == pytest.approx((call, put), rel=1e-12)
        assert zbc - zbp == pytest.approx(p_s - x * p_t, abs=1e-15)
        for f_t in (0.02, 0.05):
            bond = lambda rate: np.asarray(r.hull_white_zcb(short_rate=rate, t=big_t, maturity=s, a=A, sigma=SIGMA,  # noqa: E731
                                                            discount_t=p_t, discount_maturity=p_s, forward_t=f_t))
            assert p_t * forward_measure(lambda rate: np.maximum(x - bond(rate), 0.0), big_t, f_t) == pytest.approx(
                zbp, rel=1e-7)


def test_hull_white_caplets_equal_quantlibs_analytic_cap_engine_and_their_payoff_integral():
    """QuantLib 1.43's AnalyticCapFloorEngine on a HullWhite model (one 6-month period fixing at 1 year, accrual on
    ACT/365F); and the caplet is P(0, T1) E^{T1}[(1 + K tau)(1/(1 + K tau) - P(T1, T2))^+], the payoff tau (L - K)^+
    paid at T2 valued at T1."""
    cap = QL["hw_cap"]
    for strike in (0.03, 0.045):
        args = dict(strike=strike, fixing=1.0, payment=cap["T2"], accrual=cap["T2"] - 1.0, a=A, sigma=SIGMA,
                    discount_fixing=cap["P1"], discount_payment=cap["P2"])
        caplet, floorlet = (r.hull_white_caplet(option=o, **args) for o in ("caplet", "floorlet"))
        assert (caplet, floorlet) == pytest.approx(cap[strike], rel=1e-12)
        face = 1 + strike * (cap["T2"] - 1.0)
        bond = lambda rate: np.asarray(r.hull_white_zcb(short_rate=rate, t=1.0, maturity=cap["T2"], a=A, sigma=SIGMA,  # noqa: E731
                                                        discount_t=cap["P1"], discount_maturity=cap["P2"],
                                                        forward_t=0.04))
        assert cap["P1"] * forward_measure(lambda rate: face * np.maximum(1 / face - bond(rate), 0.0), 1.0, 0.04) == \
            pytest.approx(caplet, rel=1e-7)


def test_jamshidian_equals_quantlib_the_payoff_integral_and_swaption_parity():
    """QuantLib 1.43's JamshidianSwaptionEngine, 18 months into 30, semi-annual, to its r* tolerance: it solves r* to
    1e-8, which moves the price by at most P(0, T0) max B(T0, T_i) 1e-8, under 3e-8 here;
    the payer is P(0, T0) E^{T0}[(1 - sum c_i P(T0, T_i))^+] integrated past the kink found here; payer less
    receiver is the forward-starting payer swap, P(0, T0) - sum c_i P(0, T_i) (a model-free parity)."""
    sw = QL["swaption"]
    base = dict(expiry=SWAP_T0, payment_times=SWAP_PAY, accruals=SWAP_ACC, a=A, sigma=SIGMA, discount_expiry=sw["P0"],
                discount_payments=sw["Pi"], forward_expiry=sw["f0"])
    for strike in (0.037, 0.042, 0.05):
        payer = r.jamshidian_swaption(option="payer", strike=strike, **base)
        receiver = r.jamshidian_swaption(option="receiver", strike=strike, **base)
        assert (payer["price"], receiver["price"]) == pytest.approx(sw[strike], abs=3e-8)
        coupons = strike * np.array(SWAP_ACC)
        coupons[-1] += 1.0
        assert payer["price"] - receiver["price"] == pytest.approx(sw["P0"] - coupons @ sw["Pi"], abs=1e-15)
        assert coupons @ payer["strikes"] == pytest.approx(1.0, abs=1e-14)

    coupons = 0.042 * np.array(SWAP_ACC)
    coupons[-1] += 1.0

    def bond(rate):
        return np.asarray(r.hull_white_zcb(short_rate=np.asarray(rate)[..., None], t=SWAP_T0, maturity=np.array(SWAP_PAY),
                                           a=A, sigma=SIGMA, discount_t=sw["P0"], discount_maturity=np.array(sw["Pi"]),
                                           forward_t=sw["f0"])) @ coupons

    kink = brentq(lambda x: float(bond(x)) - 1.0, -0.5, 0.5, xtol=1e-16)
    sd = SIGMA * math.sqrt(-math.expm1(-2 * A * SWAP_T0) / (2 * A))
    z = np.linspace((kink - sw["f0"]) / sd, 11.0, 100001)
    integral = sw["P0"] * np.trapezoid((1 - bond(sw["f0"] + sd * z)) * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi), z)
    found = r.jamshidian_swaption(option="payer", strike=0.042, **base)
    assert found["price"] == pytest.approx(integral, rel=1e-9) and found["r_star"] == pytest.approx(kink, abs=1e-12)


# ------------------------------------------------------------------------------------------------ caps, FX


def test_black76_caps_give_hulls_caplet_and_cap_less_floor_is_the_swap():
    """Hull, "Caps and floors" (Black's model): a caplet on 10,000 at 8% on a 3-month rate fixing in one year, the
    forward at 7%, vol 20%, the curve flat at 6.9394% continuous, is worth 5.162; a period fixing now is its
    intrinsic value; cap less floor is N sum tau P (F - K), the payer swap."""
    hull = r.black76_cap_floor(option="cap", forwards=[0.07], strike=0.08, fixing_times=[1.0], accruals=[0.25],
                               discount_factors=[math.exp(-0.069394 * 1.25)], vol=0.2, notional=10000.0)
    assert hull["price"] == pytest.approx(5.162, abs=0.0015)
    forwards, fixes = [0.031, 0.036, 0.041, 0.044], [0.0, 0.5, 1.0, 1.5]
    accruals, discounts, vols = [0.5, 0.5, 0.5, 0.5], [0.985, 0.968, 0.949, 0.929], [0.25, 0.22, 0.2, 0.19]
    cap = r.black76_cap_floor(option="cap", forwards=forwards, strike=0.037, fixing_times=fixes, accruals=accruals,
                              discount_factors=discounts, vol=vols, notional=1e6)
    floor = r.black76_cap_floor(option="floor", forwards=forwards, strike=0.037, fixing_times=fixes,
                                accruals=accruals, discount_factors=discounts, vol=vols, notional=1e6)
    swap = 1e6 * np.sum(np.array(accruals) * discounts * (np.array(forwards) - 0.037))
    assert cap["price"] - floor["price"] == pytest.approx(swap, rel=1e-12)
    assert cap["caplets"][0] == 0.0 and floor["caplets"][0] == pytest.approx(1e6 * 0.5 * 0.985 * 0.006)
    flat = r.black76_cap_floor(option="cap", forwards=forwards, strike=0.037, fixing_times=fixes, accruals=accruals,
                               discount_factors=discounts, vol=0.2, notional=1e6)
    assert flat["caplets"][2] == pytest.approx(cap["caplets"][2]) and flat["caplets"][1] != cap["caplets"][1]


def test_cip_forward_gives_hulls_example_and_leaves_no_covered_arbitrage():
    """Hull, "Forward contracts on currencies": 2-year rates of 3% (AUD) and 1% (USD), continuous, spot 0.7500 USD
    per AUD give a forward of 0.7206. With money-market rates, borrowing QUOTE, buying BASE spot, depositing it and
    selling it forward gains nothing; ACT/360 accrues 365/360 of ACT/365; points are (F - S) times the pip multiplier."""
    hull = qf.cip_forward(spot=0.75, base_rate=0.03, quote_rate=0.01, days=730, base_day_count="ACT/365",
                         quote_day_count="ACT/365", compounding="continuous", pip_multiplier=10000.0)
    assert hull["forward"] == pytest.approx(0.7206, abs=5e-5) and hull["points"] == pytest.approx(
        (hull["forward"] - 0.75) * 1e4)
    spot, days, r_b, r_q = 132.40, 182, 0.0375, 0.0040
    deal = qf.cip_forward(spot=spot, base_rate=r_b, quote_rate=r_q, days=days, base_day_count="ACT/360",
                         quote_day_count="ACT/365", compounding="simple", pip_multiplier=100.0)
    base_held = (1.0 / spot) * (1 + r_b * days / 360)
    assert base_held * deal["forward"] == pytest.approx(1 + r_q * days / 365, rel=1e-14)
    assert deal["points"] == pytest.approx((deal["forward"] - spot) * 100)
    with pytest.raises(ValueError, match="base_day_count"):
        qf.cip_forward(spot=spot, base_rate=r_b, quote_rate=r_q, days=days, base_day_count="30/360",
                      quote_day_count="ACT/365", compounding="simple", pip_multiplier=100.0)


def through_legs(amount, legs, path):
    """Sell `amount` of path[0] for path[1], then that for path[2], each at the side a dealer would deal:
    a pair A/B sells A at its bid and buys A at its ask (selling B gets 1/ask of A)."""
    for start, end in zip(path, path[1:]):
        for (base, quote), bid, ask in legs:
            if (base, quote) == (start, end):
                amount *= bid
            elif (base, quote) == (end, start):
                amount /= ask
    return amount


def test_cross_rates_are_what_dealing_through_the_legs_gives():
    """Shapiro, cross rates: the cross bid is what selling one unit of X through the two legs brings in Y, each leg
    dealt on the side that trade takes; the cross ask is the Y it costs to buy one X back; this for every
    orientation of the legs, and the inverse cross is the reciprocal with the sides swapped."""
    gbpusd, usdjpy, eurusd = (("GBP", "USD"), 1.3110, 1.3114), (("USD", "JPY"), 143.80, 143.84), \
        (("EUR", "USD"), 1.1150, 1.1153)
    cases = [(gbpusd, usdjpy, "GBP/JPY"), (usdjpy, gbpusd, "GBP/JPY"), (eurusd, gbpusd, "EUR/GBP"),
             (gbpusd, eurusd, "EUR/GBP"), (usdjpy, eurusd, "JPY/EUR")]
    for first, second, cross in cases:
        found = qf.cross_rate(first_bid=first[1], first_ask=first[2], first_pair="/".join(first[0]),
                             second_bid=second[1], second_ask=second[2], second_pair="/".join(second[0]),
                             cross=cross, mid_rule="cross")
        x, y = cross.split("/")
        common = (set(first[0]) & set(second[0])).pop()
        assert found["bid"] == pytest.approx(through_legs(1.0, [first, second], [x, common, y]), rel=1e-14), cross
        assert found["ask"] == pytest.approx(1.0 / through_legs(1.0, [first, second], [y, common, x]), rel=1e-14)
        assert found["mid"] == pytest.approx((found["bid"] + found["ask"]) / 2) and found["bid"] < found["ask"]
    straight = qf.cross_rate(first_bid=1.3110, first_ask=1.3114, first_pair="GBP/USD", second_bid=143.80,
                            second_ask=143.84, second_pair="USD/JPY", cross="GBP/JPY", mid_rule="legs")
    assert straight["mid"] == pytest.approx(1.3112 * 143.82, rel=1e-14)
    assert (straight["bid"], straight["ask"]) == pytest.approx((1.3110 * 143.80, 1.3114 * 143.84), rel=1e-14)
    inverse = qf.cross_rate(first_bid=1.3110, first_ask=1.3114, first_pair="GBP/USD", second_bid=143.80,
                           second_ask=143.84, second_pair="USD/JPY", cross="JPY/GBP", mid_rule="legs")
    assert (inverse["bid"], inverse["ask"]) == pytest.approx((1 / straight["ask"], 1 / straight["bid"]), rel=1e-14)
    assert inverse["mid"] == pytest.approx(1 / straight["mid"], rel=1e-14)
    divided = qf.cross_rate(first_bid=1.1150, first_ask=1.1153, first_pair="EUR/USD", second_bid=1.3110,
                           second_ask=1.3114, second_pair="GBP/USD", cross="EUR/GBP", mid_rule="legs")
    assert divided["mid"] == pytest.approx(1.11515 / 1.3112, rel=1e-14)
    for bad in (dict(cross="GBP/EUR"), dict(second_pair="EUR/CHF"), dict(first_pair="GBPUSD")):
        with pytest.raises(ValueError):
            qf.cross_rate(**(dict(first_bid=1.3110, first_ask=1.3114, first_pair="GBP/USD", second_bid=143.80,
                                 second_ask=143.84, second_pair="USD/JPY", cross="GBP/JPY", mid_rule="legs") | bad))


def quantlib_values():
    """QL recomputed with QuantLib 1.43: `scratch/lib2/venv/bin/python tests/test_qf_rates.py`."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    cal, d360, a365 = ql.NullCalendar(), ql.Thirty360(ql.Thirty360.USA), ql.Actual365Fixed()
    values = {}
    helpers = []
    for maturity, rate in zip(PAR_T, PAR_C):
        months = int(round(maturity * 12))
        if months <= 6:
            helpers.append(ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)), ql.Period(months, ql.Months), 0,
                                                cal, ql.Unadjusted, False, d360))
        else:
            schedule = ql.Schedule(today, today + ql.Period(months, ql.Months), ql.Period(6, ql.Months), cal,
                                   ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
            helpers.append(ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(100.0)), 0, 100.0, schedule, [rate],
                                                  d360, ql.Unadjusted, 100.0, today))
    for name, kind in (("linear_zero", ql.PiecewiseLinearZero), ("log_linear_discount", ql.PiecewiseLogLinearDiscount)):
        curve = kind(today, helpers, d360)
        values[name] = dict(nodes=[curve.discount(t) for t in PAR_T], off=[curve.discount(t) for t in OFF])
    leg = [ql.SimpleCashFlow(amount, today + int(days)) for days, amount in zip(BOND_DAYS, BOND_CF)]
    values["bond"] = {}
    for frequency, code in ((2, ql.Semiannual), (1, ql.Annual), (4, ql.Quarterly), ("continuous", None)):
        rate = (ql.InterestRate(0.0515, a365, ql.Continuous, ql.Annual) if code is None
                else ql.InterestRate(0.0515, a365, ql.Compounded, code))
        macaulay = (ql.CashFlows.duration(leg, rate, ql.Duration.Simple, False, today) if code is None
                    else ql.CashFlows.duration(leg, rate, ql.Duration.Macaulay, False, today))
        at_price = None if code is None else ql.CashFlows.yieldRate(leg, 98.7, a365, ql.Compounded, code, False,
                                                                    today, today, 1e-15, 1000, 0.05)
        values["bond"][frequency] = (ql.CashFlows.npv(leg, rate, False, today), macaulay,
                                     ql.CashFlows.duration(leg, rate, ql.Duration.Modified, False, today),
                                     ql.CashFlows.convexity(leg, rate, False, today), at_price)
    dates = [today] + [today + d for d in (182, 365, 730, 1095, 1826, 2557, 3652)]
    curve = ql.DiscountCurve(dates, [1.0, 0.9802, 0.9608, 0.9215, 0.8830, 0.8105, 0.7420, 0.6480], a365)
    handle = ql.YieldTermStructureHandle(curve)
    model = ql.HullWhite(handle, A, SIGMA)
    values["zcb"] = []
    for t, big_t, rate in ((0.0, 3.0, None), (1.3, 4.6, 0.031), (2.5, 9.0, 0.047)):
        f_t = curve.forwardRate(t, t, ql.Continuous, ql.NoFrequency, True).rate()
        rate = f_t if rate is None else rate
        values["zcb"].append(((t, big_t, rate, curve.discount(t), curve.discount(big_t), f_t),
                              model.discountBond(t, big_t, rate)))
    values["zbo"] = [((big_t, s, x, curve.discount(big_t), curve.discount(s)),
                      model.discountBondOption(ql.Option.Call, x, big_t, s),
                      model.discountBondOption(ql.Option.Put, x, big_t, s))
                     for big_t, s, x in ((1.5, 4.0, 0.91), (3.2, 3.7, 0.975))]
    index = ql.IborIndex("flat", ql.Period(6, ql.Months), 0, ql.USDCurrency(), cal, ql.Unadjusted, False, a365, handle)
    start = today + ql.Period(18, ql.Months)
    fixed = ql.Schedule(start, start + ql.Period(30, ql.Months), ql.Period(6, ql.Months), cal, ql.Unadjusted,
                        ql.Unadjusted, ql.DateGeneration.Forward, False)
    values["swaption"] = {"P0": curve.discount(SWAP_T0), "f0": curve.forwardRate(SWAP_T0, SWAP_T0, ql.Continuous,
                                                                                 ql.NoFrequency, True).rate(),
                          "Pi": [curve.discount(t) for t in SWAP_PAY]}
    for strike in (0.037, 0.042, 0.05):
        prices = []
        for kind in (ql.Swap.Payer, ql.Swap.Receiver):
            swaption = ql.Swaption(ql.VanillaSwap(kind, 1.0, fixed, strike, a365, fixed, index, 0.0, a365),
                                   ql.EuropeanExercise(start))
            swaption.setPricingEngine(ql.JamshidianSwaptionEngine(model, handle))
            prices.append(swaption.NPV())
        values["swaption"][strike] = tuple(prices)
    begin = today + ql.Period(1, ql.Years)
    schedule = ql.Schedule(begin, begin + ql.Period(6, ql.Months), ql.Period(6, ql.Months), cal, ql.Unadjusted,
                           ql.Unadjusted, ql.DateGeneration.Forward, False)
    leg = ql.IborLeg([1.0], schedule, index, a365, ql.Unadjusted, [0], [1.0], [0.0])
    t2 = a365.yearFraction(today, begin + ql.Period(6, ql.Months))
    values["hw_cap"] = {"T2": t2, "P1": curve.discount(1.0), "P2": curve.discount(t2)}
    for strike in (0.03, 0.045):
        prices = []
        for kind in (ql.Cap, ql.Floor):
            instrument = kind(leg, [strike])
            instrument.setPricingEngine(ql.AnalyticCapFloorEngine(model, handle))
            prices.append(instrument.NPV())
        values["hw_cap"][strike] = tuple(prices)
    return values


def leaves(tree):
    """The numbers of a nested dict, list or tuple in a fixed order (dict keys sorted), None skipped."""
    if isinstance(tree, dict):
        for key in sorted(tree, key=str):
            yield from leaves(tree[key])
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            yield from leaves(item)
    elif tree is not None:
        yield float(tree)


if __name__ == "__main__":
    print("QL frozen values match QuantLib:", np.allclose(list(leaves(quantlib_values())), list(leaves(QL)),
                                                          rtol=1e-14, atol=0))


# ------------------------------------------------------------------------------------------------ the curve set

CURVE_T, CURVE_Z = [0.5, 1.0, 2.0, 5.0, 10.0], [0.031, 0.033, 0.036, 0.041, 0.044]
FLOWS, FLOW_T = [2.5] * 7 + [102.5], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


@pytest.mark.parametrize("compounding", ["continuous", 1, 2, 12])
def test_curve_risk_on_a_flat_curve_is_bond_risk_at_that_yield(compounding):
    """Fisher and Weil (1971): on a flat curve the zero rate is the yield, so PV, durations, convexity and DV01 are
    bond_risk's (Macaulay = Fisher-Weil; modified = Macaulay / (1 + y/k), Macaulay itself when continuous)."""
    flat = qf.curve_risk(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T, curve_zero_rates=[0.045] * 5,
                        interpolation="linear_zero", compounding=compounding)
    bond = qf.bond_risk(cash_flows=FLOWS, times=FLOW_T, ytm=0.045, frequency=compounding)
    assert flat["price"] == pytest.approx(bond["price"], rel=1e-13)
    assert flat["fisher_weil_duration"] == pytest.approx(bond["macaulay_duration"], rel=1e-13)
    for key in ("modified_duration", "convexity", "dv01"):
        assert flat[key] == pytest.approx(bond[key], rel=1e-12), key


def test_curve_risk_written_out_on_two_cash_flows_and_by_bumping_the_curve():
    """5 at 2y and 105 at 3y on zeros 2% at 1y and 4% at 3y, continuous and linear: z(2) = 3%, PV = 5 e^-0.06 + 105
    e^-0.12, Fisher-Weil (2 5 e^-0.06 + 3 105 e^-0.12) / PV; then each compounding's duration and convexity are
    -PV'/PV and PV''/PV, by central differences of the node rates (linear, so a node shift is a parallel shift)."""
    found = qf.curve_risk(cash_flows=[5.0, 105.0], times=[2.0, 3.0], curve_times=[1.0, 3.0],
                         curve_zero_rates=[0.02, 0.04], interpolation="linear_zero", compounding="continuous")
    pv = 5 * math.exp(-0.06) + 105 * math.exp(-0.12)
    assert found["price"] == pytest.approx(pv, rel=1e-15) and list(found["zero_rates"]) == pytest.approx([0.03, 0.04])
    assert found["fisher_weil_duration"] == pytest.approx((10 * math.exp(-0.06) + 315 * math.exp(-0.12)) / pv)
    assert found["dv01"] == pytest.approx(found["modified_duration"] * pv * 1e-4, rel=1e-14)
    h = 1e-5
    for compounding, interpolation in (("continuous", "linear_zero"), (2, "linear_zero"), ("simple", "linear_zero"),
                                       ("continuous", "log_linear_discount")):
        def pv_at(shift):
            return qf.curve_risk(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T,
                                curve_zero_rates=np.add(CURVE_Z, shift), interpolation=interpolation,
                                compounding=compounding)["price"]
        found = qf.curve_risk(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                             interpolation=interpolation, compounding=compounding)
        assert found["modified_duration"] == pytest.approx(-(pv_at(h) - pv_at(-h)) / (2 * h * found["price"]),
                                                           rel=1e-8), compounding
        assert found["convexity"] == pytest.approx((pv_at(h) - 2 * found["price"] + pv_at(-h)) / (h * h
                                                   * found["price"]), rel=1e-4), compounding


def test_curve_risk_interpolates_as_named_and_its_duration_is_the_tent_krds_sum():
    """'log_linear_discount' is qf.rates.curve_discount's on the nodes' discount factors (piecewise-flat forwards);
    'linear_zero' with continuous rates is key_rate_durations' curve, whose tent KRDs sum to the Fisher-Weil."""
    early = [0.25, *FLOW_T]                                      # one cash flow before the first node
    log_linear = qf.curve_risk(cash_flows=[1.0, *FLOWS], times=early, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                              interpolation="log_linear_discount", compounding="continuous")
    nodes = np.exp(-np.array(CURVE_Z) * CURVE_T)
    assert log_linear["discount_factors"] == pytest.approx(qf.curve_discount(
        times=early, node_times=CURVE_T, node_discounts=nodes, interpolation="log_linear_discount"), rel=1e-14)
    flat_left = qf.curve_risk(cash_flows=[1.0, *FLOWS], times=early, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                             interpolation="linear_zero", compounding=2)
    assert flat_left["zero_rates"][:2] == pytest.approx([CURVE_Z[0], CURVE_Z[0]], rel=1e-15)
    for compounding in (2, 12, "simple"):          # a node's own rate comes back from its discount factor
        at_nodes = qf.curve_risk(cash_flows=[1.0] * 5, times=CURVE_T, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                                interpolation="log_linear_discount", compounding=compounding)
        assert at_nodes["zero_rates"] == pytest.approx(CURVE_Z, rel=1e-12), compounding
    linear = qf.curve_risk(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                          interpolation="linear_zero", compounding="continuous")
    tents = qf.key_rate_durations(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                                 key_times=CURVE_T, shift="tent", bump=1e-5, difference="central")
    assert tents.sum() == pytest.approx(linear["modified_duration"], rel=1e-8)
    assert linear["modified_duration"] == linear["fisher_weil_duration"]
    annual = qf.curve_risk(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                          interpolation="log_linear_discount", compounding=1)
    assert annual["discount_factors"][-1] == pytest.approx((1 + annual["zero_rates"][-1]) ** -4.0, rel=1e-14)
    with pytest.raises(ValueError, match="last curve time"):
        qf.curve_risk(cash_flows=[100.0], times=[12.0], curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                     interpolation="log_linear_discount", compounding="continuous")
    with pytest.raises(ValueError, match="compounding"):
        qf.curve_risk(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                     interpolation="linear_zero", compounding="annual")


BOND = dict(issue="2024-03-15", settlement="2025-01-10", maturity="2027-03-15", coupon_rate=0.06, frequency=2,
            face=100.0, roll="same_day", stub="short_front")


def test_bond_cash_flows_written_out_under_30_360_and_icma():
    """A 6% semi-annual bond, coupons on 15 March and 15 September, settled 10 January 2025: five flows of 3, the
    last with the 100; accrued from 15 September, 115/360 of 6 on 30/360 Bond Basis ((2025 - 2024) 360 + (1 - 9) 30
    + (10 - 15) days) and 117/181 of 3 on ACT/ACT ICMA (QuantLib's FixedRateBond accruedAmount: 1.9166666666666554
    and 1.939226519337023); times 65/360 + k/2 and 64/362 + k/2."""
    thirty = qf.bond_cash_flows(**BOND, day_count="30/360")
    assert thirty["dates"] == ["2025-03-15", "2025-09-15", "2026-03-15", "2026-09-15", "2027-03-15"]
    assert list(thirty["amounts"]) == [3.0, 3.0, 3.0, 3.0, 103.0] and thirty["period_start"] == "2024-09-15"
    assert thirty["accrued"] == pytest.approx(115 / 360 * 6, rel=1e-14) == pytest.approx(1.9166666666666554)
    assert thirty["times"] == pytest.approx(65 / 360 + np.arange(5) / 2, rel=1e-14)
    icma = qf.bond_cash_flows(**BOND, day_count="ACT/ACT ICMA")
    assert icma["accrued"] == pytest.approx(3 * 117 / 181, rel=1e-14) == pytest.approx(1.939226519337023, rel=1e-13)
    assert icma["times"] == pytest.approx(64 / 362 + np.arange(5) / 2, rel=1e-14)
    act = qf.bond_cash_flows(**BOND, day_count="ACT/365F")
    assert act["coupons"][0] == pytest.approx(100 * 0.06 * 181 / 365, rel=1e-14)        # 2024-09-15 to 2025-03-15


@pytest.mark.parametrize("stub, dates, first", [
    ("short_front", ["2024-09-15", "2025-03-15", "2025-09-15", "2026-03-15"], 2.233695652173906),
    ("long_front", ["2025-03-15", "2025-09-15", "2026-03-15"], 5.233695652173909),
    ("short_back", ["2024-11-01", "2025-05-01", "2025-11-01", "2026-03-15"], 2.220994475138127),
    ("long_back", ["2024-11-01", "2025-05-01", "2026-03-15"], 3.0 + 2.220994475138127)])
def test_bond_stubs_are_quantlibs_schedules_and_isma_coupons(stub, dates, first):
    """Issued 1 May 2024, maturing 15 March 2026, 6% semi-annual on ACT/ACT ICMA: QuantLib's Schedule (Backward for
    the front stubs, Forward for the back ones, a firstDate for the long front) and FixedRateBond coupons; the odd
    period accrues its days over twice its notional regular period's (137 / 368 of 6 at the front)."""
    found = qf.bond_cash_flows(**dict(BOND, issue="2024-05-01", settlement="2024-07-01", maturity="2026-03-15",
                                     stub=stub), day_count="ACT/ACT ICMA")
    assert found["dates"] == dates
    odd = found["coupons"][0] if stub.endswith("front") else found["coupons"][-1]
    assert odd == pytest.approx(first, rel=1e-13)
    assert found["accrued"] == pytest.approx(0.9945652173912967 if stub.endswith("front") else 6 * 61 / 368,
                                             rel=1e-13)


def test_end_of_month_roll_keeps_month_ends_and_a_par_bond_prices_at_par_off_a_flat_curve():
    """QuantLib's Schedule with endOfMonth: a bond maturing 28 February 2026 pays on 31 August and the last day of
    February; 'same_day' keeps the 28th. A bond whose coupon is the flat curve's rate, settled on a coupon date with
    the curve compounded at the coupon frequency, is worth its face (qf.rates.curve_risk; Fabozzi, ch. 2)."""
    eom = qf.bond_cash_flows(issue="2024-02-29", settlement="2024-07-01", maturity="2026-02-28", coupon_rate=0.05,
                            frequency=2, face=100.0, day_count="ACT/ACT ICMA", roll="end_of_month", stub="short_front")
    assert eom["dates"] == ["2024-08-31", "2025-02-28", "2025-08-31", "2026-02-28"]
    assert eom["accrued"] == pytest.approx(1.6711956521739069, rel=1e-13)
    same = qf.bond_cash_flows(issue="2024-02-28", settlement="2024-07-01", maturity="2026-02-28", coupon_rate=0.05,
                             frequency=2, face=100.0, day_count="ACT/ACT ICMA", roll="same_day", stub="short_front")
    assert same["dates"] == ["2024-08-28", "2025-02-28", "2025-08-28", "2026-02-28"]
    clipped = qf.bond_cash_flows(issue="2025-05-31", settlement="2025-06-02", maturity="2026-05-31", coupon_rate=0.05,
                                frequency=4, face=100.0, day_count="30/360", roll="same_day", stub="short_front")
    assert clipped["dates"] == ["2025-08-31", "2025-11-30", "2026-02-28", "2026-05-31"]      # the 31st cut short
    for frequency, day_count in ((1, "30/360"), (2, "ACT/ACT ICMA"), (4, "30/360")):
        par = qf.bond_cash_flows(issue="2023-06-30", settlement="2024-06-30", maturity="2031-06-30", coupon_rate=0.047,
                                frequency=frequency, face=100.0, day_count=day_count, roll="end_of_month",
                                stub="short_front")
        assert par["accrued"] == 0.0
        priced = qf.curve_risk(cash_flows=par["amounts"], times=par["times"], curve_times=[1.0, 10.0],
                              curve_zero_rates=[0.047, 0.047], interpolation="linear_zero", compounding=frequency)
        assert priced["price"] == pytest.approx(100.0, rel=1e-13), frequency
    with pytest.raises(ValueError, match="issue <= settlement < maturity"):
        qf.bond_cash_flows(**dict(BOND, settlement="2027-03-15"), day_count="30/360")
    with pytest.raises(ValueError, match="divide 12"):
        qf.bond_cash_flows(**dict(BOND, frequency=5), day_count="30/360")


def test_forward_rates_give_hulls_table_and_compose_with_the_zeros_in_each_compounding():
    """Hull, Options, Futures, and Other Derivatives, "Forward rates" (Table 4.5): continuous zeros 3.0, 4.0, 4.6,
    5.0 and 5.3% at 1 to 5 years give the forwards 5.0, 5.8, 6.2 and 6.5%. In each compounding the forward composes:
    D(t1) / D(t2) is the forward's growth over t2 - t1, and from today it is the zero rate."""
    zeros = [0.030, 0.040, 0.046, 0.050, 0.053]
    found = qf.forward_rates(start=[1.0, 2.0, 3.0, 4.0], end=[2.0, 3.0, 4.0, 5.0], zero_start=zeros[:-1],
                            zero_end=zeros[1:], compounding="continuous")
    assert found == pytest.approx([0.050, 0.058, 0.062, 0.065], abs=1e-12)
    t1, t2, z1, z2 = 1.5, 4.0, 0.031, 0.043
    annual = qf.forward_rates(start=t1, end=t2, zero_start=z1, zero_end=z2, compounding=1)
    assert (1 + z2) ** t2 == pytest.approx((1 + z1) ** t1 * (1 + annual) ** (t2 - t1), rel=1e-14)
    quarterly = qf.forward_rates(start=t1, end=t2, zero_start=z1, zero_end=z2, compounding=4)
    assert (1 + z2 / 4) ** (4 * t2) == pytest.approx((1 + z1 / 4) ** (4 * t1) * (1 + quarterly / 4) ** (4 * (t2 - t1)))
    simple = qf.forward_rates(start=t1, end=t2, zero_start=z1, zero_end=z2, compounding="simple")
    assert 1 + z2 * t2 == pytest.approx((1 + z1 * t1) * (1 + simple * (t2 - t1)), rel=1e-14)
    assert qf.forward_rates(start=0.0, end=t2, zero_start=0.0, zero_end=z2, compounding=2) == pytest.approx(z2)
    curve = qf.curve_risk(cash_flows=[1.0, 1.0], times=[1.5, 4.0], curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                         interpolation="linear_zero", compounding="continuous")
    implied = -math.log(curve["discount_factors"][1] / curve["discount_factors"][0]) / 2.5
    assert qf.forward_rates(start=1.5, end=4.0, zero_start=curve["zero_rates"][0], zero_end=curve["zero_rates"][1],
                           compounding="continuous") == pytest.approx(implied, rel=1e-13)
    with pytest.raises(ValueError, match="end after"):
        qf.forward_rates(start=2.0, end=2.0, zero_start=0.03, zero_end=0.03, compounding="continuous")


def test_z_spread_written_out_and_against_curve_risk_and_bond_yield():
    """One flow of 100 at 2 years on a flat 3% continuous curve priced 90: s = -ln(0.9)/2 - 0.03. On a flat
    continuous curve the spread is bond_yield('continuous') less the curve's rate; the curve's own price gives 0;
    and in every compounding the curve's zero rates moved by s reprice the bond (qf.rates.curve_risk; Fabozzi)."""
    assert qf.z_spread(90.0, [100.0], [2.0], [1.0, 3.0], [0.03, 0.03], "linear_zero", "continuous") == pytest.approx(
        -math.log(0.9) / 2 - 0.03, rel=1e-13)
    ytm = qf.bond_yield(price=97.0, cash_flows=FLOWS, times=FLOW_T, frequency="continuous")
    assert qf.z_spread(97.0, FLOWS, FLOW_T, CURVE_T, [0.04] * 5, "linear_zero", "continuous") == pytest.approx(
        ytm - 0.04, abs=1e-13)
    for compounding, interpolation in (("continuous", "log_linear_discount"), (2, "linear_zero"),
                                       ("simple", "linear_zero"), (1, "linear_zero"), (2, "log_linear_discount")):
        fair = qf.curve_risk(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T, curve_zero_rates=CURVE_Z,
                            interpolation=interpolation, compounding=compounding)["price"]
        assert qf.z_spread(fair, FLOWS, FLOW_T, CURVE_T, CURVE_Z, interpolation, compounding) == pytest.approx(0.0,
                                                                                                         abs=1e-13)
        for price in (fair * 0.7, fair * 0.9, fair * 1.05, fair * 1.3):     # spreads beyond the first bracket too
            s = qf.z_spread(price, FLOWS, FLOW_T, CURVE_T, CURVE_Z, interpolation, compounding)
            if interpolation == "linear_zero":
                moved = qf.curve_risk(cash_flows=FLOWS, times=FLOW_T, curve_times=CURVE_T,
                                     curve_zero_rates=np.add(CURVE_Z, s), interpolation=interpolation,
                                     compounding=compounding)["price"]
                assert moved == pytest.approx(price, rel=1e-12), (compounding, price)
            assert (s > 0) == (price < fair)
    with pytest.raises(ValueError, match="reprices"):
        qf.z_spread(1e9, FLOWS, FLOW_T, CURVE_T, CURVE_Z, "linear_zero", "continuous")
