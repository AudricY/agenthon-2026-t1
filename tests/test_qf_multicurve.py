"""agent/qf/rates.py, lane lib6's curve functions (runs/lanes/lib6-prompt.md; ranks 2, 6 and 10 of
docs/research/library-syllabus-1011.md): the multi-curve bootstrap against curves built by hand here, whose quotes it
must return node for node, and against the single-curve bootstrap; the instruments' formulas are written out in this
file, apart from the library's. The inputs are chosen freely; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf
import qf.rates as r

pytestmark = pytest.mark.fast

LAG = 2 / 365
OIS_NODES = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
OIS_DF = np.array([0.9855, 0.9702, 0.9395, 0.9080, 0.8452])
PROJ_NODES = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0])
PROJ_DF = np.array([0.9910, 0.9815, 0.9719, 0.9622, 0.9290, 0.8952, 0.8290])


def hand_curve(nodes, dfs):
    """ln D linear in t between (0, 0) and the nodes: flat forwards (Hagan and West's log-linear discount)."""
    def read(t):
        return np.exp(np.interp(np.asarray(t, dtype=float), np.r_[0.0, nodes], np.r_[0.0, np.log(dfs)]))
    return read


def ois_par(d, start, ends, accruals, lag):
    """An OIS's fixed rate: the compounded overnight leg, D(T_{i-1}) / D(T_i) - 1 paid at T_i + lag, over the fixed
    leg's annuity at the same payment dates."""
    bounds = d(np.r_[start, ends])
    paid = d(np.asarray(ends) + lag)
    return float((bounds[:-1] / bounds[1:] - 1.0) @ paid / (np.asarray(accruals) @ paid))


def swap_par(d, p, start, fixed, fixed_acc, floating):
    projected = p(np.r_[start, floating])
    return float((projected[:-1] / projected[1:] - 1.0) @ d(floating) / (np.asarray(fixed_acc) @ d(fixed)))


def quarterly(end):
    return list(np.arange(0.25, end + 1e-9, 0.25))


def hand_quotes(d, p, lag):
    """The par quotes of a full instrument set on the curves d and p: OIS of half a year to five years, a 3-month
    deposit, a 3x6 FRA, a 6x9 future (convexity 0.0004) and 1- to 5-year swaps, annual fixed against quarterly
    floating."""
    ois = [{"kind": "ois", "rate": ois_par(d, 0.0, [0.5], [0.5], lag), "start": 0.0, "times": [0.5],
            "accruals": [0.5]}]
    for end in (1.0, 2.0, 3.0, 5.0):
        times = list(np.arange(1.0, end + 1e-9))
        ois.append({"kind": "ois", "rate": ois_par(d, 0.0, times, [1.0] * len(times), lag), "start": 0.0,
                    "times": times, "accruals": [1.0] * len(times)})
    fra = lambda a, b: (p(a) / p(b) - 1.0) / (b - a)  # noqa: E731
    projection = [{"kind": "deposit", "rate": fra(0.0, 0.25), "start": 0.0, "end": 0.25, "accrual": 0.25},
                  {"kind": "fra", "rate": fra(0.25, 0.5), "start": 0.25, "end": 0.5, "accrual": 0.25},
                  {"kind": "future", "price": 100.0 * (1.0 - fra(0.5, 0.75) - 0.0004), "start": 0.5, "end": 0.75,
                   "accrual": 0.25, "convexity": 0.0004}]
    for end in (1.0, 2.0, 3.0, 5.0):
        fixed = list(np.arange(1.0, end + 1e-9))
        projection.append({"kind": "swap", "rate": swap_par(d, p, 0.0, fixed, [1.0] * len(fixed), quarterly(end)),
                           "start": 0.0, "fixed_times": fixed, "fixed_accruals": [1.0] * len(fixed),
                           "float_times": quarterly(end)})
    return ois, projection


def test_the_rates_cards_keep_the_format_and_their_examples_run():
    check_cards(r)


def test_quotes_made_from_two_curves_built_by_hand_return_them_node_for_node():
    """Ametrano and Bianchetti (2013): the OIS curve from OIS quotes alone, then the projection curve with every
    floating leg projected off it and discounted off the OIS curve. Quotes priced here from hand-built curves (with
    a payment lag and a future's convexity) return those curves' discount factors at every node: an OIS's node is
    its last payment, its last period's end plus the lag."""
    d, p = hand_curve(OIS_NODES + LAG, OIS_DF), hand_curve(PROJ_NODES, PROJ_DF)
    ois, projection = hand_quotes(d, p, LAG)
    found = qf.bootstrap_multicurve(ois, projection[::-1], LAG, "log_linear_discount")    # any order
    assert found["discount_times"] == pytest.approx(OIS_NODES + LAG, abs=1e-15)
    assert found["discount_factors"] == pytest.approx(OIS_DF, abs=1e-12)
    assert found["projection_times"] == pytest.approx(PROJ_NODES, abs=1e-15)
    assert found["projection_factors"] == pytest.approx(PROJ_DF, abs=1e-12)


@pytest.mark.parametrize("interpolation", ["linear_zero", "log_linear_discount", "cubic_log_discount"])
def test_every_instrument_reprices_to_par_on_the_curves_it_builds(interpolation):
    """Whatever the interpolation, each quote is the par rate of the curves returned, read back by curve_discount."""
    ois, projection = hand_quotes(hand_curve(OIS_NODES, OIS_DF), hand_curve(PROJ_NODES, PROJ_DF), 0.0)
    found = qf.bootstrap_multicurve(ois, projection, 0.0, interpolation)

    def reader(times, factors):
        return lambda t: np.asarray(qf.curve_discount(np.atleast_1d(t), times, factors, interpolation))

    d = reader(found["discount_times"], found["discount_factors"])
    p = reader(found["projection_times"], found["projection_factors"])
    for quote in ois:
        assert ois_par(d, 0.0, quote["times"], quote["accruals"], 0.0) == pytest.approx(quote["rate"], abs=1e-12)
    for quote in projection:
        if quote["kind"] == "swap":
            par = swap_par(d, p, 0.0, quote["fixed_times"], quote["fixed_accruals"], quote["float_times"])
            assert par == pytest.approx(quote["rate"], abs=1e-12)
        else:
            rate = quote.get("rate", (100.0 - quote.get("price", 0.0)) / 100.0 - quote.get("convexity", 0.0))
            forward = (p(quote["start"])[0] / p(quote["end"])[0] - 1.0) / quote["accrual"]
            assert forward == pytest.approx(rate, abs=1e-12)


@pytest.mark.parametrize("interpolation", ["linear_zero", "log_linear_discount", "cubic_log_discount"])
def test_one_curve_is_the_par_bootstrap(interpolation):
    """With no OIS quotes the projection curve discounts too, and a par swap K sum tau D(T_i) = 1 - D(T_n) is
    bootstrap_par_curve's par instrument: the same nodes."""
    maturities, rates = [1.0, 2.0, 3.0, 5.0, 7.0], [0.031, 0.033, 0.0345, 0.036, 0.0372]
    swaps = [{"kind": "swap", "rate": rate, "start": 0.0, "fixed_times": list(np.arange(1.0, end + 1e-9)),
              "fixed_accruals": [1.0] * int(end), "float_times": quarterly(end)} for end, rate in zip(maturities, rates)]
    found = qf.bootstrap_multicurve(None, swaps, 0.0, interpolation)
    par = qf.bootstrap_par_curve(maturities, rates, 1, interpolation, "brentq")
    assert found["projection_factors"] == pytest.approx(par["discount_factors"], abs=1e-12)
    assert found["discount_factors"] == pytest.approx(found["projection_factors"], abs=0.0)


def test_a_flat_ois_curve_and_a_flat_spread_give_flat_zero_rates_a_spread_apart():
    """The flat-curve case: OIS continuously compounded at 3% everywhere, the projection curve 40 basis points above
    it: every projection node's zero rate is 3.4% and every OIS node's 3%."""
    d = lambda t: np.exp(-0.03 * np.asarray(t, dtype=float))  # noqa: E731
    p = lambda t: np.exp(-0.034 * np.asarray(t, dtype=float))  # noqa: E731
    ois, projection = hand_quotes(d, p, 0.0)
    found = qf.bootstrap_multicurve(ois, projection, 0.0, "linear_zero")
    assert -np.log(found["discount_factors"]) / found["discount_times"] == pytest.approx(0.03, abs=1e-13)
    assert -np.log(found["projection_factors"]) / found["projection_times"] == pytest.approx(0.034, abs=1e-13)


def test_a_future_is_the_fra_at_its_rate_less_the_convexity_adjustment():
    """Hull, "Eurodollar futures": the forward rate is the futures rate less the convexity adjustment."""
    deposit = {"kind": "deposit", "rate": 0.03, "start": 0.0, "end": 0.25, "accrual": 0.25}
    fra = qf.bootstrap_multicurve(None, [deposit, {"kind": "fra", "rate": 0.031, "start": 0.25, "end": 0.5,
                                                  "accrual": 0.25}], 0.0, "log_linear_discount")
    future = qf.bootstrap_multicurve(None, [deposit, {"kind": "future", "price": 96.85, "start": 0.25, "end": 0.5,
                                                     "accrual": 0.25, "convexity": 0.0005}], 0.0, "log_linear_discount")
    assert future["projection_factors"] == pytest.approx(fra["projection_factors"], abs=1e-15)
    assert fra["projection_factors"][1] == pytest.approx(1 / (1.0075 * (1 + 0.031 * 0.25)), abs=1e-15)
    act360 = qf.bootstrap_multicurve(None, [{**deposit, "accrual": 91 / 360}], 0.0, "log_linear_discount")
    assert act360["projection_factors"][0] == pytest.approx(1 / (1 + 0.03 * 91 / 360), abs=1e-15)   # the day count's


@pytest.mark.parametrize("ois, projection, match", [
    (None, [{"kind": "bond", "rate": 0.03}], "kind"),
    (None, [{"kind": "fra", "rate": 0.03, "start": 0.0, "end": 0.5}], "accrual"),
    (None, [{"kind": "fra", "rate": 0.03, "start": 0.0, "end": 0.5, "accrual": 0.5, "notional": 1.0}], "notional"),
    (None, [{"kind": "deposit", "rate": 0.03, "start": 0.0, "end": 0.5, "accrual": 0.5},
            {"kind": "fra", "rate": 0.03, "start": 0.25, "end": 0.5, "accrual": 0.25}], "same date"),
    ([{"kind": "ois", "rate": 0.03, "start": 0.0, "times": [1.0], "accruals": [1.0]}],
     [{"kind": "swap", "rate": 0.03, "start": 0.0, "fixed_times": [1.0, 2.0], "fixed_accruals": [1.0, 1.0],
       "float_times": [1.0, 2.0]}], "reach"),
    ([{"kind": "swap", "rate": 0.03}], [{"kind": "deposit", "rate": 0.03, "start": 0.0, "end": 0.5,
                                         "accrual": 0.5}], "kind")])
def test_a_malformed_instrument_is_refused_by_what_is_wrong(ois, projection, match):
    with pytest.raises(ValueError, match=match):
        qf.bootstrap_multicurve(ois, projection, 0.0, "log_linear_discount")


# ------------------------------------------------------------------------------------------------ the swap rate

CURVE = ([0.5, 1.0, 2.0, 3.0], [0.985, 0.968, 0.935, 0.900])
FIXED, FIXED_ACC = [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]
FLOAT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def swap(**changes):
    args = dict(start=0.0, fixed_times=FIXED, fixed_accruals=FIXED_ACC, float_times=FLOAT, discount_curve=CURVE,
                projection_curve=None, interpolation="log_linear_discount", strike=0.03, side="payer", notional=1e6)
    return r.swap_rate(**{**args, **changes})


def test_the_annuity_and_the_par_rate_by_hand_on_one_curve():
    """Hull, "Swaps": A = sum tau_i D(T_i) = 0.968 + 0.935 + 0.900; on one curve the floating leg is 1 - D(T_n), so
    S = (1 - 0.900) / A; a payer at K is worth N A (S - K) and a receiver its negative."""
    found = swap()
    annuity = 0.968 + 0.935 + 0.900
    assert found["annuity"] == pytest.approx(annuity, abs=1e-15)
    assert found["float_leg"] == pytest.approx(0.1, abs=1e-15)
    assert found["par_rate"] == pytest.approx(0.1 / annuity, abs=1e-15)
    assert found["value"] == pytest.approx(1e6 * annuity * (0.1 / annuity - 0.03), abs=1e-8)
    assert swap(side="receiver")["value"] == pytest.approx(-found["value"], abs=1e-9)


def test_a_swap_at_its_par_rate_is_worth_nothing_and_one_period_is_the_simple_forward():
    found = swap(start=0.5, fixed_times=[1.5, 2.5], fixed_accruals=[1.02, 0.99], float_times=[1.0, 1.5, 2.0, 2.5],
                 projection_curve=([0.5, 1.0, 3.0], [0.982, 0.962, 0.885]))
    assert swap(start=0.5, fixed_times=[1.5, 2.5], fixed_accruals=[1.02, 0.99], float_times=[1.0, 1.5, 2.0, 2.5],
                projection_curve=([0.5, 1.0, 3.0], [0.982, 0.962, 0.885]),
                strike=found["par_rate"])["value"] == pytest.approx(0.0, abs=1e-9)
    d = hand_curve(np.array(CURVE[0]), np.array(CURVE[1]))
    assert found["annuity"] == pytest.approx(1.02 * d(1.5) + 0.99 * d(2.5), abs=1e-15)    # tau_i by the day count
    one = swap(start=1.0, fixed_times=[2.0], fixed_accruals=[1.0], float_times=[2.0])
    assert one["par_rate"] == pytest.approx((0.968 / 0.935 - 1.0) / 1.0, abs=1e-15)


def test_two_curves_project_off_one_and_discount_off_the_other_and_a_projection_equal_to_the_discount_is_one_curve():
    """Floating leg sum_j (P(t_{j-1}) / P(t_j) - 1) D(t_j), written out here on curves read at their nodes."""
    projection = ([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [0.983, 0.965, 0.947, 0.929, 0.911, 0.893])
    discount = ([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [0.986, 0.971, 0.956, 0.941, 0.926, 0.911])
    found = swap(discount_curve=discount, projection_curve=projection)
    p, d = np.r_[1.0, projection[1]], np.array(discount[1])
    assert found["float_leg"] == pytest.approx((p[:-1] / p[1:] - 1.0) @ d, abs=1e-15)
    assert found["annuity"] == pytest.approx(0.971 + 0.941 + 0.911, abs=1e-15)
    assert swap(projection_curve=CURVE)["par_rate"] == pytest.approx(swap()["par_rate"], abs=1e-15)


def test_a_quoted_swap_reprices_to_zero_on_the_multicurve_bootstrap():
    """Ranks 2 and 6 together: the 3-year swap quoted to bootstrap_multicurve is worth nothing on its curves."""
    ois, projection = hand_quotes(hand_curve(OIS_NODES, OIS_DF), hand_curve(PROJ_NODES, PROJ_DF), 0.0)
    curves = qf.bootstrap_multicurve(ois, projection, 0.0, "linear_zero")
    quote = next(q for q in projection if q["kind"] == "swap" and q["fixed_times"][-1] == 3.0)
    found = r.swap_rate(0.0, quote["fixed_times"], quote["fixed_accruals"], quote["float_times"],
                        (curves["discount_times"], curves["discount_factors"]),
                        (curves["projection_times"], curves["projection_factors"]), "linear_zero", quote["rate"],
                        "receiver", 1.0)
    assert found["value"] == pytest.approx(0.0, abs=1e-12) and found["par_rate"] == pytest.approx(quote["rate"])


@pytest.mark.parametrize("bad, match", [
    (dict(side="pay"), "side"), (dict(fixed_accruals=[1.0, 1.0]), "fixed_accruals"), (dict(start=1.0), "start"),
    (dict(discount_curve=[0.97, 0.91]), "pair"), (dict(fixed_times=[1.0, 2.0, 4.0]), "last node"),
    (dict(interpolation="spline"), "interpolation")])
def test_a_bad_swap_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        swap(**bad)


# ------------------------------------------------------------------------------------------------ the basis

PAIR = dict(spot=1.0800, base_rate=0.035, quote_rate=0.050, days=182, base_day_count="ACT/360",
            quote_day_count="ACT/365")


@pytest.mark.parametrize("compounding", ["simple", "continuous"])
@pytest.mark.parametrize("leg", ["base", "quote"])
def test_the_basis_is_zero_where_parity_holds_and_puts_the_quoted_forward_back(compounding, leg):
    """Du, Tepper and Verdelhan (2018): the basis is the rate gap that covered interest parity leaves; at the parity
    forward it is zero, and cip_forward with the basis leg's rate moved by it gives the quoted forward."""
    parity = qf.cip_forward(**PAIR, compounding=compounding, pip_multiplier=1e4)["forward"]
    assert qf.cip_basis(**PAIR, forward=parity, compounding=compounding, basis_leg=leg)["basis"] == pytest.approx(
        0.0, abs=1e-15)
    found = qf.cip_basis(**PAIR, forward=1.0905, compounding=compounding, basis_leg=leg)
    moved = {**PAIR, f"{leg}_rate": PAIR[f"{leg}_rate"] + found["basis"]}
    assert qf.cip_forward(**moved, compounding=compounding, pip_multiplier=1e4)["forward"] == pytest.approx(1.0905,
                                                                                                         abs=1e-14)
    assert found["implied_rate"] == pytest.approx(PAIR[f"{leg}_rate"] + found["basis"], abs=1e-16)
    assert found["parity_forward"] == pytest.approx(parity, abs=1e-15)


def test_the_basis_by_hand():
    """Simple compounding, the base leg: x = ((1 + r_q tau_q) S / F - 1) / tau_b - r_b."""
    found = qf.cip_basis(**PAIR, forward=1.0905, compounding="simple", basis_leg="base")
    assert found["basis"] == pytest.approx(((1 + 0.05 * 182 / 365) * 1.08 / 1.0905 - 1) / (182 / 360) - 0.035,
                                           abs=1e-15)


DOM_D, DOM_P = ([0.5, 1.0, 2.0], [0.9805, 0.9612, 0.9230]), ([0.5, 1.0, 2.0], [0.9790, 0.9580, 0.9170])
FOR_D, FOR_P = ([0.5, 1.0, 2.0], [0.9880, 0.9762, 0.9520]), ([0.5, 1.0, 2.0], [0.9870, 0.9740, 0.9480])
TIMES4, TAU_D, TAU_F = [0.5, 1.0, 1.5, 2.0], [0.507, 0.503, 0.507, 0.503], [0.5, 0.5, 0.5, 0.5]


def xccy(**changes):
    args = dict(spot=1.25, foreign_notional=1e6, start=0.0, payment_times=TIMES4, domestic_accruals=TAU_D,
                foreign_accruals=TAU_F, domestic_discount=DOM_D, domestic_projection=DOM_P, foreign_discount=FOR_D,
                foreign_projection=FOR_P, interpolation="log_linear_discount", spread=0.0, spread_leg="foreign",
                notional_type="constant", side="receive_foreign")
    return qf.xccy_basis_swap(**{**args, **changes})


@pytest.mark.parametrize("notional_type", ["constant", "mtm_reset"])
def test_one_curve_a_currency_and_no_spread_is_worth_nothing(notional_type):
    """A floating leg with its notional exchanged, projected and discounted on one curve, prices at par."""
    for start, times, tau in ((0.0, TIMES4, TAU_D), (0.5, TIMES4[1:], TAU_D[1:])):      # spot and forward starting
        found = xccy(domestic_projection=None, foreign_projection=None, notional_type=notional_type, start=start,
                     payment_times=times, domestic_accruals=tau, foreign_accruals=tau)
        assert found["value"] == pytest.approx(0.0, abs=1e-9) and found["foreign_leg"] == pytest.approx(0.0, abs=1e-9)
        assert found["domestic_leg"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("notional_type", ["constant", "mtm_reset"])
@pytest.mark.parametrize("leg", ["foreign", "domestic"])
def test_the_swap_at_its_par_basis_is_worth_nothing_and_the_sides_mirror(notional_type, leg):
    found = xccy(spread_leg=leg, notional_type=notional_type)
    assert xccy(spread_leg=leg, notional_type=notional_type, spread=found["par_spread"])["value"] == pytest.approx(
        0.0, abs=1e-8)
    assert xccy(spread_leg=leg, notional_type=notional_type, spread=0.001, side="pay_foreign")["value"] == \
        pytest.approx(-xccy(spread_leg=leg, notional_type=notional_type, spread=0.001)["value"], abs=1e-9)


def test_the_legs_by_hand_with_constant_and_resetting_notionals():
    """Andersen and Piterbarg 15.2: a leg is N [-D(T_0) + sum_j (P(t_{j-1}) / P(t_j) - 1 + s tau_j) D(t_j) +
    D(T_n)]; the resetting leg's period j has the notional N_f S D_f(t_{j-1}) / D_d(t_{j-1})."""
    d_d, p_d = hand_curve(*map(np.array, DOM_D)), hand_curve(*map(np.array, DOM_P))
    d_f, p_f = hand_curve(*map(np.array, FOR_D)), hand_curve(*map(np.array, FOR_P))
    t = np.array([0.0, *TIMES4])
    foreign = 1e6 * (-1.0 + ((p_f(t[:-1]) / p_f(t[1:]) - 1.0 + 0.0015 * np.array(TAU_F)) @ d_f(t[1:])) + d_f(2.0))
    domestic = 1.25e6 * (-1.0 + (p_d(t[:-1]) / p_d(t[1:]) - 1.0) @ d_d(t[1:]) + d_d(2.0))
    found = xccy(spread=0.0015)
    assert found["foreign_leg"] == pytest.approx(foreign, abs=1e-8)
    assert found["domestic_leg"] == pytest.approx(domestic, abs=1e-8)
    assert found["value"] == pytest.approx(1.25 * foreign - domestic, abs=1e-8)
    notionals = 1.25e6 * d_f(t[:-1]) / d_d(t[:-1])
    reset = notionals @ (-d_d(t[:-1]) + (p_d(t[:-1]) / p_d(t[1:]) + 0.0015 * np.array(TAU_D)) * d_d(t[1:]))
    assert xccy(spread=0.0015, spread_leg="domestic", notional_type="mtm_reset")["domestic_leg"] == pytest.approx(
        reset, abs=1e-8)


def test_a_one_period_swaps_par_basis_is_the_cip_basis():
    """One period of 182 days ACT/360 on both legs, the domestic (quote) currency on one curve and the foreign
    (base) discount curve the FX forward's, D_f = F D_d / S: the par spread on the foreign leg over its projection
    rate is cip_basis's x on the base leg."""
    tau, spot, forward, r_d, r_f = 182 / 360, 1.08, 1.0905, 0.050, 0.035
    d_d = 1 / (1 + r_d * tau)
    d_f = forward * d_d / spot
    swap_ = qf.xccy_basis_swap(spot, 1e6, 0.0, [tau], [tau], [tau], ([tau], [d_d]), None, ([tau], [d_f]),
                              ([tau], [1 / (1 + r_f * tau)]), "log_linear_discount", 0.0, "foreign", "constant",
                              "receive_foreign")
    basis = qf.cip_basis(spot, forward, r_f, r_d, 182, "ACT/360", "ACT/360", "simple", "base")["basis"]
    assert swap_["par_spread"] == pytest.approx(basis, abs=1e-14)


@pytest.mark.parametrize("bad, match", [
    (dict(spread_leg="base"), "spread_leg"), (dict(notional_type="resetting"), "notional_type"),
    (dict(side="receive"), "side"), (dict(foreign_accruals=[0.5]), "accrual"), (dict(start=0.5), "start"),
    (dict(payment_times=[0.5, 1.0, 1.5, 2.5]), "last node")])
def test_a_bad_xccy_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        xccy(**bad)
