"""Lane lib-rates1's bonds rows (runs/lanes/lib-rates1-prompt.md; syllabus rows 36, 37, 64, 117): the floating-rate
note and its discount margin, cash-flow matching and long-only immunization by a linear program, mortgage pass-through
cash flows with prepayment, and Treasury futures conversion factors with the cheapest to deliver. References: Fabozzi,
Bond Markets, Analysis and Strategies (the 100% PSA pass-through's first months: $400 million, WAC 8.125%, pass-through
7.5%, WAM 357); Hull, Options, Futures, and Other Derivatives ("Treasury bond futures": the conversion factors 1.4623
and 1.2199) with CME's formula; closed-form identities: a floater at par at a reset and its discount margin equal to
its margin at par (Fabozzi; Tuckman and Serrat), zero-coupon dedication's cost as the liabilities' present value,
and a conversion factor as the bond's price at 6%. Every input is this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast

CURVE = ([0.5, 1.0, 2.0, 3.0], [0.985, 0.968, 0.93, 0.89])


def frn(**changes):
    args = dict(payment_times=[0.5, 1.0, 1.5, 2.0], accruals=[0.5, 0.5, 0.5, 0.5], margin=0.0, last_fixing=None,
                discount_curve=CURVE, projection_curve=None, interpolation="log_linear_discount", face=100.0,
                call_time=None, market_price=None)
    return qf.frn_price(**{**args, **changes})


def test_a_floater_at_a_reset_on_one_curve_with_no_margin_is_at_par_and_at_par_its_discount_margin_is_its_margin():
    """Fabozzi; Tuckman and Serrat: the projected coupons telescope to par; at a price of par the discount margin, its
    periods discounted at the forward plus the margin, is the quoted margin itself."""
    assert frn()["price"] == pytest.approx(100.0, rel=1e-14)
    for margin in (0.0, 0.004, 0.012):
        found = frn(margin=margin, market_price=100.0)
        assert found["discount_margin"] == pytest.approx(margin, abs=1e-13), margin
    called = frn(call_time=1.5)
    assert called["price"] == pytest.approx(100.0, rel=1e-14) and called["coupons"].size == 3


def test_a_margin_adds_its_annuity_and_between_resets_the_known_coupon_and_par_at_the_reset():
    """On one curve the floater less par is the margin's annuity, face m sum tau_i D(T_i); with the current period's
    rate fixed, the note is that coupon and par at the next reset, (1 + L tau) face D(T_1)."""
    d = qf.curve_discount(times=[0.5, 1.0, 1.5, 2.0], node_times=CURVE[0], node_discounts=CURVE[1],
                          interpolation="log_linear_discount")
    assert frn(margin=0.005)["price"] == pytest.approx(100.0 + 100 * 0.005 * 0.5 * d.sum(), rel=1e-14)
    between = frn(payment_times=[0.2, 0.7, 1.2, 1.7], last_fixing=0.031)
    first = qf.curve_discount(times=0.2, node_times=CURVE[0], node_discounts=CURVE[1],
                              interpolation="log_linear_discount")
    assert between["price"] == pytest.approx(100 * (1 + 0.031 * 0.5) * first, rel=1e-13)
    assert between["forwards"][0] == 0.031 and between["coupons"][0] == pytest.approx(100 * 0.031 * 0.5)


def test_a_projection_curve_above_the_discount_curve_lifts_the_price_and_the_discount_margin_reprices_it():
    projection = ([0.5, 1.0, 2.0, 3.0], [0.983, 0.964, 0.922, 0.878])
    found = frn(projection_curve=projection)
    assert found["price"] > 100.0
    margin = frn(projection_curve=projection, market_price=99.0)
    steps, flows = np.full(4, 0.5), margin["coupons"].copy()
    flows[-1] += 100.0
    disc = np.cumprod(1 / (1 + (margin["forwards"] + margin["discount_margin"]) * steps))
    assert flows @ disc == pytest.approx(99.0, rel=1e-13) and margin["discount_margin"] > 0


@pytest.mark.parametrize("bad, match", [(dict(call_time=1.2), "call_time"), (dict(accruals=[0.5]), "accruals"),
                                        (dict(interpolation="linear"), "interpolation")])
def test_frn_refuses_a_bad_argument_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        frn(**bad)


# ------------------------------------------------------------------------------------------------ the LP


def lp(**changes):
    args = dict(matching="cash_flows", objective="cost", prices=None, bounds=(0.0, None), exposures=None,
                liability_exposures=None, cash_flows=None, liabilities=None, times=None, reinvestment_rate=None)
    return qf.immunize_lp(**{**args, **changes})


def test_dedication_with_a_zero_per_date_buys_each_liability_at_its_present_value():
    """With one zero-coupon bond paying 100 at each liability date the cheapest cover is the liabilities' own
    present value, units L_t / 100, nothing left over."""
    times, owed, price = np.array([1.0, 2.0, 3.0]), np.array([500.0, 800.0, 300.0]), np.array([96.0, 92.0, 88.0])
    found = lp(prices=price, cash_flows=100 * np.eye(3), liabilities=owed, times=times, reinvestment_rate=0.0)
    assert found["units"] == pytest.approx(owed / 100, rel=1e-9) and found["cost"] == pytest.approx(owed @ price / 100)
    assert found["surplus"] == pytest.approx(0.0, abs=1e-9)


def test_dedication_with_coupon_bonds_covers_every_date_and_reinvested_surplus_lowers_the_cost():
    """Luenberger ch. 3 (the cash-matching problem): every date's cash with the carried surplus covers its
    liability, net flow never negative; a positive reinvestment rate can only help."""
    flows = np.array([[105.0, 0.0, 0.0], [5.0, 105.0, 0.0], [6.0, 6.0, 106.0], [0.0, 0.0, 100.0]])
    price, owed, times = np.array([101.0, 100.5, 102.0, 85.0]), np.array([300.0, 400.0, 500.0]), [1.0, 2.0, 3.0]
    plain = lp(prices=price, cash_flows=flows, liabilities=owed, times=times, reinvestment_rate=0.0)
    carried, surplus = 0.0, []
    for k in range(3):
        carried = carried + flows[:, k] @ plain["units"] - owed[k]
        surplus.append(carried)
    assert min(surplus) >= -1e-9 and plain["surplus"] == pytest.approx(surplus, abs=1e-8)
    assert np.all(plain["units"] >= -1e-12)
    richer = lp(prices=price, cash_flows=flows, liabilities=owed, times=times, reinvestment_rate=0.05)
    assert richer["cost"] <= plain["cost"] + 1e-9


def test_a_surplus_carried_at_the_reinvestment_rate_sets_the_cost():
    """One bond paying 50 then 100 against 150 due at the second date: x units cover it where 50 x (1 + r)^2 + 100 x
    = 150, the surplus reinvested over the two years between, so the cost is price times 150 / (50 (1 + r)^2 + 100)."""
    for rate in (0.0, 0.04, 0.1):
        found = lp(prices=[120.0], cash_flows=[[50.0, 100.0]], liabilities=[0.0, 150.0], times=[1.0, 3.0],
                   reinvestment_rate=rate)
        assert found["cost"] == pytest.approx(120.0 * 150 / (50 * (1 + rate) ** 2 + 100), rel=1e-9), rate
        assert found["surplus"] == pytest.approx([50 * found["units"][0], 0.0], abs=1e-7)


def test_exposure_matching_long_only_meets_the_square_system_and_redingtons_conditions():
    """Value and dollar duration of a 5-year liability matched by 2- and 10-year zeros: the unique long-only point is
    qf.immunization_weights's square solution; the barbell's convexity exceeds the bullet's (Redington)."""
    prices = np.array([90.0, 60.0])
    durations, value = np.array([2.0, 10.0]), 1000.0
    rows = np.vstack([prices, prices * durations])
    found = lp(matching="exposures", prices=prices, exposures=rows, liability_exposures=[value, value * 5.0])
    square = qf.immunization_weights(exposures=[durations], liability_value=value, liability_exposures=[5.0])
    assert found["amounts"] == pytest.approx(square, rel=1e-9) and found["matched"] == pytest.approx([value, 5000.0])
    convexity = found["amounts"] @ durations ** 2 / value
    assert convexity > 5.0 ** 2


def test_mismatch_finds_the_closest_long_only_portfolio_where_an_exact_match_is_out_of_reach():
    """Two key-rate rows no long-only mix can meet: the value row matched, the rows' summed absolute miss the least
    over a grid of the feasible portfolios; the cost objective refuses the same rows."""
    prices = np.array([100.0, 100.0, 100.0])
    rows = np.array([[100.0, 100.0, 100.0], [10.0, 50.0, 0.0], [0.0, 20.0, 60.0]])
    target = [1000.0, 600.0, 100.0]
    found = lp(matching="exposures", objective="mismatch", prices=prices, exposures=rows, liability_exposures=target)
    assert found["matched"][0] == pytest.approx(1000.0) and np.all(found["units"] >= -1e-12)
    best = min(abs(rows[1] @ x - 600) + abs(rows[2] @ x - 100)
               for a in np.linspace(0, 10, 101) for b in np.linspace(0, 10 - a, 101) for x in [np.array([a, b, 10 - a - b])])
    assert abs(found["matched"][1] - 600) + abs(found["matched"][2] - 100) <= best + 1e-9
    with pytest.raises(ValueError, match="no portfolio"):
        lp(matching="exposures", prices=prices, exposures=rows, liability_exposures=target)


@pytest.mark.parametrize("bad, match", [(dict(objective="mismatch", prices=[1.0], cash_flows=[[1.0]], liabilities=[1.0],
                                              times=[1.0], reinvestment_rate=0.0), "objective"),
                                        (dict(matching="exposures", prices=[1.0], exposures=[[1.0]],
                                              liability_exposures=[1.0], reinvestment_rate=0.0), "None for"),
                                        (dict(matching="duration"), "matching")])
def test_the_lp_refuses_a_bad_argument_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        lp(**bad)


# ------------------------------------------------------------------------------------------------ pass-throughs


def mbs(**changes):
    args = dict(balance=400e6, mortgage_rate=0.08125, pass_through_rate=0.075, months=357, prepayment="psa",
                speed=1.0, age=3)
    return qf.mortgage_cash_flows(**{**args, **changes})


def test_fabozzis_100_psa_pass_through():
    """Fabozzi's monthly cash flow table at 100% PSA, to the dollar: month 1 SMM 0.00067, payment 2,975,868, net
    interest 2,500,000, scheduled principal 267,535, prepayment 267,470, cash flow 3,035,005; month 2 balance
    399,464,995, payment 2,973,877, scheduled principal 269,166, prepayment 334,198."""
    found = mbs()
    assert round(found["smm"][0], 5) == 0.00067
    assert [round(found[k][0]) for k in ("payment", "interest", "scheduled_principal", "prepayment", "principal",
                                         "cash_flow")] == [2975868, 2500000, 267535, 267470, 535005, 3035005]
    assert [round(found[k][1]) for k in ("balance", "payment", "scheduled_principal", "prepayment")] == \
        [399464995, 2973877, 269166, 334198]


def test_no_prepayment_amortises_to_zero_at_a_level_payment_and_psa_ramps_to_six_percent():
    flat = mbs(prepayment="cpr", speed=0.0, age=0, months=360)
    assert flat["payment"] == pytest.approx(flat["payment"][0], rel=1e-12)
    assert flat["principal"].sum() == pytest.approx(400e6, rel=1e-12)
    assert flat["balance"][-1] - flat["principal"][-1] == pytest.approx(0.0, abs=1e-5)
    ramp = mbs(age=0, months=360, speed=1.5)
    cpr = 1 - (1 - ramp["smm"]) ** 12
    assert cpr[:30] == pytest.approx(1.5 * 0.06 * np.arange(1, 31) / 30, rel=1e-12)
    assert cpr[30:] == pytest.approx(0.09, rel=1e-12)
    assert ramp["average_life"] < flat["average_life"]
    fixed = mbs(prepayment="cpr", speed=0.06)
    assert fixed["smm"] == pytest.approx(1 - 0.94 ** (1 / 12), rel=1e-14)


# ------------------------------------------------------------------------------------------------ conversion factors


def ctd(**changes):
    args = dict(coupon_rates=[0.10, 0.08], maturities=["2050-05-15", "2048-07-15"], delivery_month="2030-03",
                rounding="quarter", factor_decimals=4, futures_price=None, clean_prices=None)
    return qf.bond_futures_ctd(**{**args, **changes})


def test_hulls_conversion_factors():
    """Hull: a 10% bond 20 years and 2 months from the delivery month, cut to 20 years, has factor 1.4623; an 8% bond of
    18 years and 4 months, cut to 18 years and 3 months, 1.2199 (CME's formula gives both)."""
    assert ctd()["conversion_factors"] == pytest.approx([1.4623, 1.2199], abs=1e-12)


@pytest.mark.parametrize("coupon, rounded_coupon", [(0.0425, 0.0425), (0.043125, 0.04375),
                                                       (0.04312499, 0.0425), (0.04312501, 0.04375),
                                                       (0.018125, 0.01875)])
def test_cme_coupon_rounding_uses_eighths_of_a_percent_and_rounds_ties_up(coupon, rounded_coupon):
    """CME Group (2024), Calculating U.S. Treasury Futures Conversion Factors:
    https://www.cmegroup.com/articles/2024/calculating-us-treasury-futures-conversion-factors.html
    The quoted coupon rounds to one-eighth percent, ties up, before pricing at 6%. Independently sum the semiannual
    cash flows on a 15-year 3-month cut schedule, then subtract three months' accrued interest. The 1.8125% tie also
    checks that a binary floating-point representation does not round a decimal tie down."""
    flows = np.full(31, rounded_coupon / 2)
    flows[-1] += 1.0
    times = 0.25 + np.arange(31) / 2
    clean = float(flows @ 1.03 ** (-2 * times)) - rounded_coupon / 4
    plain = ctd(coupon_rates=[coupon], maturities=["2045-06-15"], factor_decimals=None)
    published = ctd(coupon_rates=[coupon], maturities=["2045-06-15"])
    assert plain["conversion_factors"][0] == pytest.approx(clean, rel=1e-13)
    assert published["conversion_factors"][0] == pytest.approx(round(clean, 4), abs=1e-12)


def test_a_factor_is_the_bonds_price_at_six_percent_on_the_cut_schedule_less_its_accrued():
    """z = 0: the price per unit at 3% a half-year of the cut schedule (qf.bond_price); z = 3 (a quarter): three months
    before a coupon date, the value there discounted by sqrt(1.03), less the quarter's accrued c/4 (Hull)."""
    for coupon in (0.03, 0.045, 0.06, 0.09):
        found = ctd(coupon_rates=[coupon, coupon], maturities=["2045-03-15", "2045-06-15"], factor_decimals=None)
        flows = np.full(30, 100 * coupon / 2)
        flows[-1] += 100
        whole = qf.bond_price(cash_flows=flows, times=np.arange(1, 31) / 2, ytm=0.06, frequency=2) / 100
        later = (coupon / 2 + qf.bond_price(cash_flows=flows, times=np.arange(1, 31) / 2, ytm=0.06, frequency=2) / 100)
        assert found["conversion_factors"][0] == pytest.approx(whole, rel=1e-13)
        assert found["conversion_factors"][1] == pytest.approx(later / math.sqrt(1.03) - coupon / 4, rel=1e-13)
    assert ctd(coupon_rates=[0.06], maturities=["2049-03-15"])["conversion_factors"] == pytest.approx([1.0])


@pytest.mark.parametrize("maturity, rounding, months", [("2045-12-15", "quarter", 3), ("2045-11-15", "month", 2),
                                                        ("2045-10-15", "month", 1)])
def test_past_seven_months_the_next_coupon_is_months_away_and_the_rest_are_whole_half_years(maturity, rounding, months):
    """Fifteen years and nine, eight or seven months from the delivery month: the next coupon comes in v months (3 for
    the quarter cut; z - 6 for the month cut), then 31 half-years; the factor is that bond at 6% less the accrued
    c/2 (6 - v)/6 (CME's C = 1/1.03^{2n+1} and a = 1/1.03^{v/6})."""
    for coupon in (0.025, 0.07):
        found = ctd(coupon_rates=[coupon], maturities=[maturity], rounding=rounding, factor_decimals=None)
        flows = np.full(31, 100 * coupon / 2)
        flows[-1] += 100
        rest = qf.bond_price(cash_flows=flows, times=np.arange(1, 32) / 2, ytm=0.06, frequency=2) / 100
        expected = (coupon / 2 + rest) / 1.03 ** (months / 6) - coupon / 2 * (6 - months) / 6
        assert found["conversion_factors"][0] == pytest.approx(expected, rel=1e-13), (coupon, maturity)


def test_month_rounding_differs_from_quarter_rounding_and_the_cheapest_has_the_least_delivery_cost():
    """Five-year note futures cut to whole months: 4 months left gives v = 4, a bond future's quarter cut 3."""
    quarter = ctd(coupon_rates=[0.05], maturities=["2034-07-20"], factor_decimals=None)["conversion_factors"][0]
    month = ctd(coupon_rates=[0.05], maturities=["2034-07-20"], rounding="month",
                factor_decimals=None)["conversion_factors"][0]
    assert month != pytest.approx(quarter, rel=1e-9)
    priced = ctd(futures_price=112.0, clean_prices=[164.0, 136.0])
    assert priced["delivery_costs"] == pytest.approx([164.0 - 112.0 * 1.4623, 136.0 - 112.0 * 1.2199])
    assert priced["cheapest"] == int(np.argmin(priced["delivery_costs"]))


@pytest.mark.parametrize("bad, match", [(dict(rounding="year"), "rounding"), (dict(futures_price=110.0), "together"),
                                        (dict(maturities=["2029-01-15", "2048-07-15"]), "after delivery")])
def test_the_ctd_refuses_a_bad_argument_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        ctd(**bad)
