"""agent/qf/credit.py's CDS (runs/lanes/lib7-prompt.md, rank 11 of docs/research/library-syllabus-1011.md): the legs
against Hull's worked example and a flat hazard's closed form, the 'exact' integrals against the definitions integrated
by quadrature here, the credit triangle, the clean and dirty upfronts. Every input is this file's own or Hull's; no
unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import integrate

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf.credit as cr

pytestmark = pytest.mark.fast

ANNUAL = dict(payment_times=[1.0, 2.0, 3.0, 4.0, 5.0], accruals=[1.0] * 5, accrued=0.0)


def flat(rate, horizon):
    """A flat continuously compounded curve as one log-linear node."""
    return dict(discount_times=[horizon], discount_factors=[math.exp(-rate * horizon)],
                interpolation="log_linear_discount")


def price(**changes):
    args = dict(hazard_times=[5.0], hazard_rates=[0.02], recovery=0.4, **ANNUAL, **flat(0.05, 5.0),
                integration="exact", accrual_on_default=True, coupon=0.01, side="buyer")
    return cr.cds_price(**{**args, **changes})


def test_the_credit_cards_keep_the_format_and_their_examples_run():
    check_cards(cr)


def test_hulls_five_year_cds_by_the_midpoint_rule():
    """Hull, "Credit derivatives" (the worked CDS valuation): default 2% a year conditional on no earlier default
    (lambda = -ln 0.98), R = 40%, r = 5% continuous, annual premiums, defaults mid-year. Payments 4.0704 s, accrual
    payments 0.0426 s, the expected payoff 0.0511, the breakeven spread 0.0124 (124 bp)."""
    found = price(hazard_rates=[-math.log(0.98)], integration="midpoint")
    scheduled = price(hazard_rates=[-math.log(0.98)], integration="midpoint", accrual_on_default=False)
    assert found["survival"] == pytest.approx(0.98 ** np.arange(1, 6), rel=1e-12)
    assert scheduled["rpv01"] == pytest.approx(4.0704, abs=1e-4)
    assert found["rpv01"] - scheduled["rpv01"] == pytest.approx(0.0426, abs=1e-4)
    assert found["protection_leg"] == pytest.approx(0.0511, abs=1e-4)
    assert found["par_spread"] == pytest.approx(0.0124, abs=5e-5)
    assert scheduled["protection_leg"] == found["protection_leg"]


def test_a_flat_hazard_gives_the_protection_leg_in_closed_form():
    """(1 - R) lambda / (lambda + r) (1 - e^{-(lambda + r) T}) on a flat hazard and a flat curve, any schedule."""
    for lam, r, t in [(0.02, 0.05, 5.0), (0.15, 0.01, 3.0), (0.0, 0.03, 2.0), (0.04, 0.0, 7.0)]:
        times = np.arange(0.25, t + 1e-9, 0.25)
        found = price(hazard_times=[t], hazard_rates=[lam], payment_times=times, accruals=np.full(times.size, 0.25),
                      **flat(r, t))
        w = lam + r
        assert found["protection_leg"] == pytest.approx(0.6 * lam / w * -math.expm1(-w * t) if w else 0.0,
                                                        rel=1e-13, abs=1e-16)


def reference(hazard_times, hazard_rates, recovery, payment_times, accruals, accrued, curve):
    """The legs' definitions integrated by scipy.integrate.quad, period by period: S(t) = exp(-int lambda), dQ =
    lambda S du, the accrued premium rising linearly within a period from its part before today."""
    nodes, rates = np.asarray(hazard_times), np.asarray(hazard_rates)

    def hazard(u):
        return rates[min(np.searchsorted(nodes, u, side="left"), rates.size - 1)]

    def survival(u):
        edges = np.r_[0.0, nodes]
        spent = sum(rates[j] * max(0.0, min(u, edges[j + 1] if j < rates.size - 1 else np.inf) - edges[j])
                    for j in range(rates.size))
        return math.exp(-spent)

    protection = on_default = scheduled = 0.0
    start = 0.0
    for i, (t, tau) in enumerate(zip(payment_times, accruals)):
        begun = accrued if i == 0 else 0.0
        points = sorted({start, t, *[x for x in [*nodes, *curve.x] if start < x < t]})
        for a, b in zip(points[:-1], points[1:]):
            density = lambda u: hazard(u) * survival(u) * curve(u)
            protection += integrate.quad(density, a, b, epsabs=1e-15, epsrel=1e-13)[0]
            on_default += integrate.quad(lambda u: (begun + (tau - begun) * (u - start) / (t - start)) * density(u),
                                         a, b, epsabs=1e-15, epsrel=1e-13)[0]
        scheduled += tau * curve(t) * survival(t)
        start = t
    return (1 - recovery) * protection, scheduled + on_default


class LogLinear:
    """Discount factors log-linear between nodes, D(0) = 1: the curve the 'exact' rule integrates exactly."""

    def __init__(self, times, factors):
        self.x, self.g = np.asarray(times), -np.log(factors)

    def __call__(self, u):
        return math.exp(-np.interp(u, np.r_[0.0, self.x], np.r_[0.0, self.g]))


def test_the_exact_legs_are_the_definitions_integrated_on_a_stepped_hazard_and_curve():
    curve = LogLinear([0.5, 1.7, 3.0, 4.6], [0.985, 0.95, 0.905, 0.84])
    hazard = dict(hazard_times=[0.8, 2.2, 3.5], hazard_rates=[0.011, 0.034, 0.052])
    schedule = dict(payment_times=[0.3, 0.8, 1.55, 2.3, 3.05, 3.8, 4.55], accruals=[0.27, 0.51, 0.76, 0.76, 0.76, 0.76,
                                                                                    0.76], accrued=0.22)
    found = cr.cds_price(**hazard, recovery=0.35, **schedule, discount_times=curve.x, discount_factors=np.exp(-curve.g),
                         interpolation="log_linear_discount", integration="exact", accrual_on_default=True,
                         coupon=0.01, side="buyer")
    protection, dirty = reference(**hazard, recovery=0.35, **schedule, curve=curve)
    assert found["protection_leg"] == pytest.approx(protection, rel=1e-11)
    assert found["rpv01_dirty"] == pytest.approx(dirty, rel=1e-11)
    assert found["rpv01"] == pytest.approx(dirty - 0.22, rel=1e-12)


def test_the_midpoint_rule_approaches_the_exact_legs_and_the_credit_triangle_holds():
    """Monthly premiums: the two rules within a basis point of RPV01. The credit triangle s = lambda (1 - R) is exact
    at r = 0 with accrual on default, the premium then the expected accrual to default or T, int S du (O'Kane, 6.6);
    at r = 5% the premiums paid in arrears put it off by about r dt / 2."""
    times = np.arange(1, 61) / 12
    monthly = dict(payment_times=times, accruals=np.full(60, 1 / 12))
    exact, mid = price(**monthly), price(**monthly, integration="midpoint")
    assert abs(exact["rpv01"] - mid["rpv01"]) < 1e-4 and abs(exact["protection_leg"] - mid["protection_leg"]) < 1e-5
    assert exact["par_spread"] == pytest.approx(0.02 * 0.6, rel=0.05 / 24 * 1.1)
    for schedule in (monthly, ANNUAL):
        found = price(**schedule, hazard_times=[1.5, 5.0], hazard_rates=[0.01, 0.03], **flat(0.0, 5.0))
        assert found["rpv01"] == pytest.approx((1 - math.exp(-0.015)) / 0.01 + math.exp(-0.015) *
                                               (1 - math.exp(-0.105)) / 0.03, rel=1e-13)
    assert price(**monthly, **flat(0.0, 5.0))["par_spread"] == pytest.approx(0.012, rel=1e-13)
    assert price(**monthly, accrual_on_default=False)["par_spread"] > exact["par_spread"]


def test_the_midpoint_rule_by_hand_with_a_period_already_begun():
    """Two periods, the first begun 0.1 before today: default at each period's midpoint, the accrual there half way
    from the part already accrued to the whole, both discounted at the midpoint."""
    lam, r = 0.03, 0.04
    found = price(hazard_times=[1.0], hazard_rates=[lam], payment_times=[0.4, 0.9], accruals=[0.5, 0.5], accrued=0.1,
                  **flat(r, 1.0), integration="midpoint")
    s0, s1, s2 = 1.0, math.exp(-lam * 0.4), math.exp(-lam * 0.9)
    d = lambda t: math.exp(-r * t)
    on_default = 0.3 * d(0.2) * (s0 - s1) + 0.25 * d(0.65) * (s1 - s2)
    assert found["rpv01_dirty"] == pytest.approx(0.5 * d(0.4) * s1 + 0.5 * d(0.9) * s2 + on_default, rel=1e-14)
    assert found["protection_leg"] == pytest.approx(0.6 * (d(0.2) * (s0 - s1) + d(0.65) * (s1 - s2)), rel=1e-14)


@pytest.mark.parametrize("x", [0.0, 1e-7, -1e-4, 9e-4, 1.1e-3, 0.03, 0.3, 0.49, -0.49, 0.51, 4.0])
def test_the_flat_integrals_match_their_series_near_zero_and_closed_forms_beyond(x):
    """(1 - e^{-x}) / x and (1 - e^{-x}(1 + x)) / x^2 at d = 1, against sum_n (-1)^n x^n / (n + 1)! and
    sum_{n >= 2} (-1)^n (n - 1) x^{n - 2} / n!, taken to 30 terms."""
    first, second = cr._flat_integrals(np.array(x), 1.0)
    assert first == pytest.approx(sum((-x) ** n / math.factorial(n + 1) for n in range(30)), rel=1e-15)
    assert second == pytest.approx(sum((-1) ** n * (n - 1) * x ** (n - 2) / math.factorial(n) for n in range(2, 32)),
                                   rel=1e-14)


def test_the_upfront_is_protection_less_the_coupons_value_clean_and_dirty_by_side():
    found = price(accrued=0.3, coupon=0.05)
    assert found["rpv01"] == pytest.approx(found["rpv01_dirty"] - 0.3, abs=1e-15)
    assert found["upfront"] == pytest.approx(found["protection_leg"] - 0.05 * found["rpv01"], abs=1e-15)
    assert found["upfront_dirty"] == pytest.approx(found["upfront"] - 0.05 * 0.3, abs=1e-15)
    seller = price(accrued=0.3, coupon=0.05, side="seller")
    assert (seller["upfront"], seller["upfront_dirty"]) == (-found["upfront"], -found["upfront_dirty"])
    assert found["premium_leg"] == pytest.approx(0.05 * found["rpv01"])
    at_par = price(coupon=price()["par_spread"])
    assert abs(at_par["upfront"]) < 1e-15
    assert price(recovery=0.0)["protection_leg"] == pytest.approx(price()["protection_leg"] / 0.6, rel=1e-14)
    assert price(hazard_rates=[0.0])["protection_leg"] == 0.0


@pytest.mark.parametrize("bad, match", [
    (dict(integration="trapezoid"), "integration"), (dict(side="long"), "side"), (dict(recovery=40.0), "recovery"),
    (dict(hazard_rates=[-0.01]), "hazard_rates"), (dict(accrued=1.0), "accrued"), (dict(accrual_on_default=1), "True"),
    (dict(payment_times=[2.0, 1.0, 3.0, 4.0, 5.0]), "payment_times"), (dict(interpolation="spline"), "interpolation")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        price(**bad)


# ------------------------------------------------------------------------------------------------ the bootstrap

QUARTERLY = dict(payment_times=np.arange(1, 29) / 4, accruals=np.full(28, 0.25 * 365 / 360))
CURVE = dict(discount_times=[0.5, 2.0, 4.0, 7.0], discount_factors=[0.99, 0.955, 0.91, 0.84],
             interpolation="log_linear_discount")


def boot(**changes):
    args = dict(maturities=[1.0, 3.0, 5.0, 7.0], quotes=[0.006, 0.009, 0.012, 0.0135], quote_kind="par_spread",
                coupon=None, recovery=0.4, **QUARTERLY, **CURVE, integration="exact", accrual_on_default=True)
    return cr.cds_bootstrap(**{**args, **changes})


@pytest.mark.parametrize("integration, accrual_on_default", [("exact", True), ("midpoint", True), ("exact", False)])
def test_the_bootstrap_returns_the_hazards_that_made_the_par_spreads(integration, accrual_on_default):
    """A round trip: par spreads priced off a stepped hazard by cds_price, each on its own maturity's part of the
    quarterly ACT/360 schedule, come back as that hazard; each spread reprices to 1e-12."""
    hazard = dict(hazard_times=[1.0, 3.0, 5.0, 7.0], hazard_rates=[0.008, 0.017, 0.026, 0.031])
    spreads = []
    for t in (1.0, 3.0, 5.0, 7.0):
        n = int(t * 4)
        spreads.append(cr.cds_price(**hazard, recovery=0.4, payment_times=QUARTERLY["payment_times"][:n],
                                    accruals=QUARTERLY["accruals"][:n], accrued=0.0, **CURVE, integration=integration,
                                    accrual_on_default=accrual_on_default, coupon=0.01, side="buyer")["par_spread"])
    found = boot(quotes=spreads, integration=integration, accrual_on_default=accrual_on_default)
    assert found["hazard_rates"] == pytest.approx(hazard["hazard_rates"], rel=1e-11)
    assert found["par_spreads"] == pytest.approx(spreads, rel=1e-12) and found["repriced"] == pytest.approx(spreads)
    assert np.all(np.diff(found["survival"]) < 0)
    assert found["survival"] == pytest.approx(np.exp(-np.cumsum(np.diff(np.r_[0.0, 1.0, 3.0, 5.0, 7.0]) *
                                                                hazard["hazard_rates"])), rel=1e-12)


@pytest.mark.parametrize("coupon", [0.01, 0.05])
def test_upfront_quotes_at_a_standard_coupon_come_back_as_their_hazards(coupon):
    hazard = dict(hazard_times=[1.0, 3.0, 5.0], hazard_rates=[0.012, 0.024, 0.04])
    upfronts = [cr.cds_price(**hazard, recovery=0.4, payment_times=QUARTERLY["payment_times"][:n],
                             accruals=QUARTERLY["accruals"][:n], accrued=0.0, **CURVE, integration="exact",
                             accrual_on_default=True, coupon=coupon, side="buyer")["upfront"] for n in (4, 12, 20)]
    found = boot(maturities=[1.0, 3.0, 5.0], quotes=upfronts, quote_kind="upfront", coupon=coupon)
    assert found["hazard_rates"] == pytest.approx(hazard["hazard_rates"], rel=1e-11)
    assert found["repriced"] == pytest.approx(upfronts, rel=1e-10, abs=1e-15)


def test_a_flat_spread_at_a_zero_rate_gives_the_flat_hazard_of_the_credit_triangle():
    found = boot(quotes=[0.012] * 4, **flat(0.0, 7.0), accruals=np.full(28, 0.25))
    assert found["hazard_rates"] == pytest.approx(np.full(4, 0.012 / 0.6), rel=1e-12)


def test_hulls_spread_bootstraps_to_his_hazard():
    """Hull's example in reverse: the spread his table gives, 0.0511 / (4.0704 + 0.0426), at R = 40%, r = 5%,
    annual premiums and the midpoint rule, returns lambda = -ln 0.98 to his tables' four digits."""
    found = boot(maturities=[5.0], quotes=[0.0511 / (4.0704 + 0.0426)], payment_times=ANNUAL["payment_times"],
                 accruals=ANNUAL["accruals"], **flat(0.05, 5.0), integration="midpoint")
    assert found["hazard_rates"][0] == pytest.approx(-math.log(0.98), rel=1e-3)


@pytest.mark.parametrize("bad, match", [
    (dict(quote_kind="spread"), "quote_kind"), (dict(coupon=0.01), "coupon"),
    (dict(maturities=[1.0, 3.0, 5.0, 7.1]), "payment_times"), (dict(quotes=[0.006, 0.009, 0.012]), "quote"),
    (dict(quotes=[0.03, 0.005, 0.012, 0.0135]), r"quotes\[1\]"), (dict(maturities=[3.0, 1.0, 5.0, 7.0]), "increasing")])
def test_the_bootstrap_refuses_what_it_cannot_solve_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        boot(**bad)
