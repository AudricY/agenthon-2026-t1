"""agent/qf/options.py's compound and chooser options (runs/lanes/lib5-prompt.md): Geske's four compound options
against Haug's (2007) worked example, the expectation of the outer payoff over the spot at the outer expiry (by
quadrature, and by a Monte Carlo of the same contract within four standard errors), the compound put-call parity and
QuantLib 1.43's AnalyticCompoundOptionEngine (frozen in `QL`; its bivariate normal is Drezner's 1978 approximation,
so it agrees to about 2e-5 on a spot of 100). Every other parameter is chosen freely; no unit's values appear."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

import qf.exotics
import qf
import qf.options as O

pytestmark = pytest.mark.fast

QL = {   # QuantLib 1.43, CompoundOption with AnalyticCompoundOptionEngine; days on ACT/365F from 2025-01-02
    "near": dict(point=(100.0, 4.0, 95.0, 146 / 365, 365 / 365, 0.04, 0.015, 0.3),
                 prices={("call", "call"): 11.687000789326484, ("call", "put"): 4.81047505790721,
                         ("put", "call"): 0.3298148007025006, ("put", "put"): 0.68948631011877}),
    "far": dict(point=(100.0, 2.5, 110.0, 73 / 365, 547 / 365, 0.04, 0.015, 0.3),
                prices={("call", "call"): 9.470184840674868, ("call", "put"): 15.28058011641632,
                        ("put", "call"): 0.012333414227597359, ("put", "put"): 8.62690714363431e-05}),
}
PAIRS = [(outer, inner) for outer in ("call", "put") for inner in ("call", "put")]


def outer_payoff_expectation(outer, inner, s, x1, x2, t1, t2, r, q, v, critical):
    """e^{-r t1} E[max(+-(V(S_t1) - X1), 0)], V the inner's bs_price at t1, by quadrature over ln S_t1 (split at the
    critical spot, where the payoff kinks)."""
    sd, mu = v * math.sqrt(t1), math.log(s) + (r - q - v * v / 2) * t1
    def integrand(z):
        value = qf.bs_price(inner, math.exp(mu + sd * z), x2, t2 - t1, r, q, v)
        payoff = max((value - x1) if outer == "call" else (x1 - value), 0.0)
        return payoff * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    kink = (math.log(critical) - mu) / sd
    return math.exp(-r * t1) * sum(integrate.quad(integrand, a, b, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
                                   for a, b in ((-12.0, kink), (kink, 12.0)))


def test_haugs_put_on_call_example():
    """Haug, The Complete Guide to Option Pricing Formulas (2007), "Options on options": a put on a call, S 500, X1
    50, X2 520, t1 0.25, T2 0.5, r 8%, b 5% (q 3%), vol 35%: 21.1965 to his root-finder's precision (the payoff's
    expectation below agrees with this function to 1e-10), its critical spot near 538.32."""
    found = qf.compound_price("put", "call", 500.0, 50.0, 520.0, 0.25, 0.5, 0.08, 0.03, 0.35)
    assert found["price"] == pytest.approx(21.1965, abs=5e-4)
    assert found["critical_spot"] == pytest.approx(538.32, abs=0.01)
    assert qf.bs_price("call", found["critical_spot"], 520.0, 0.25, 0.08, 0.03, 0.35) == pytest.approx(50.0, abs=1e-9)


@pytest.mark.parametrize("case", ["near", "far"])
@pytest.mark.parametrize("outer, inner", PAIRS)
def test_each_compound_option_is_its_outer_payoffs_discounted_expectation_and_quantlibs(case, outer, inner):
    point = QL[case]["point"]
    found = qf.compound_price(outer, inner, *point)
    assert found["price"] == pytest.approx(outer_payoff_expectation(outer, inner, *point, found["critical_spot"]),
                                           rel=1e-10, abs=1e-12)
    assert found["price"] == pytest.approx(QL[case]["prices"][outer, inner], abs=5e-5)      # Drezner (1978)


def test_a_monte_carlo_of_the_contract_agrees_within_its_error_and_the_product_of_normals_does_not():
    """The contract simulated: S_t1 lognormal, the inner priced there by bs_price, the outer's payoff discounted;
    within four standard errors. The bivariate normal written as a product of two univariate ones (a hand-written
    slip) misses it by many errors."""
    s, x1, x2, t1, t2, r, q, v = 100.0, 6.0, 100.0, 0.5, 1.5, 0.05, 0.0, 0.35
    found = qf.compound_price("call", "call", s, x1, x2, t1, t2, r, q, v)
    z = np.random.default_rng(17).standard_normal(400_000)
    st1 = s * np.exp((r - q - v * v / 2) * t1 + v * math.sqrt(t1) * z)
    payoff = math.exp(-r * t1) * np.maximum(qf.bs_price("call", st1, x2, t2 - t1, r, q, v) - x1, 0.0)
    error = payoff.std(ddof=1) / math.sqrt(payoff.size)
    assert abs(found["price"] - payoff.mean()) < 4 * error
    i = found["critical_spot"]
    y1 = (math.log(s / i) + (r - q + v * v / 2) * t1) / (v * math.sqrt(t1))
    z1 = (math.log(s / x2) + (r - q + v * v / 2) * t2) / (v * math.sqrt(t2))
    rho, N = math.sqrt(t1 / t2), stats.norm.cdf
    product = (s * math.exp(-q * t2) * N(z1) * N(y1) - x2 * math.exp(-r * t2) * N(z1 - v * math.sqrt(t2))
               * N(y1 - v * math.sqrt(t1)) - x1 * math.exp(-r * t1) * N(y1 - v * math.sqrt(t1)))
    assert abs(product - payoff.mean()) > 20 * error and rho > 0.5


@pytest.mark.parametrize("inner", ["call", "put"])
def test_compound_put_call_parity(inner):
    """Hull, "Compound options": a call on the inner less a put on it at the same X1 is the inner's price today less
    X1 e^{-r t1}."""
    s, x1, x2, t1, t2, r, q, v = 90.0, 3.0, 92.0, 0.3, 1.1, 0.02, 0.01, 0.25
    call = qf.compound_price("call", inner, s, x1, x2, t1, t2, r, q, v)["price"]
    put = qf.compound_price("put", inner, s, x1, x2, t1, t2, r, q, v)["price"]
    assert call - put == pytest.approx(qf.bs_price(inner, s, x2, t2, r, q, v) - x1 * math.exp(-r * t1), abs=1e-11)


def test_the_bivariate_normal_cdf_is_the_definitions_and_compound_price_refuses_bad_arguments():
    M = qf.exotics._bivariate_normal_cdf
    assert M(0.0, 0.0, 0.7) == pytest.approx(0.25 + math.asin(0.7) / (2 * math.pi), abs=1e-14)      # Sheppard
    assert M(0.4, -1.1, 0.0) == pytest.approx(stats.norm.cdf(0.4) * stats.norm.cdf(-1.1), abs=1e-14)
    assert M(-50.0, 1.0, 0.3) == 0.0 and M(9.0, 9.0, -0.4) == pytest.approx(1.0, abs=1e-15)
    for h, k, rho in ((0.3, -0.2, 0.5), (-1.0, 2.0, -0.95), (1.5, 1.5, 0.999)):
        assert M(h, k, rho) + M(h, -k, -rho) == pytest.approx(stats.norm.cdf(h), abs=1e-13)
    with pytest.raises(ValueError, match="before"):
        qf.compound_price("call", "call", 100.0, 4.0, 95.0, 1.0, 0.5, 0.04, 0.0, 0.3)
    with pytest.raises(ValueError, match="largest value"):
        qf.compound_price("call", "put", 100.0, 120.0, 95.0, 0.5, 1.0, 0.04, 0.0, 0.3)
    with pytest.raises(ValueError, match="outer"):
        qf.compound_price("chooser", "call", 100.0, 4.0, 95.0, 0.5, 1.0, 0.04, 0.0, 0.3)


# ------------------------------------------------------------------------------------------------ the chooser


def test_the_chooser_gives_haugs_example_quantlib_and_rubinsteins_put_call_identity():
    """Haug (2007), 4.6.1: S 50, K 50, t1 0.25, T 0.5, r 8%, b 8% (q 0), vol 25%: 6.1071. QuantLib 1.43's
    AnalyticSimpleChooserEngine on 91 and 182 days of ACT/365F, 6.0984767237743185, and on S 100, K 95, 120 and 400
    days, r 4%, q 2%, vol 30%, 19.136135774932672. Hull, "Chooser options": the call plus e^{-q (T - t1)} puts on S
    to t1 at strike K e^{-(r - q)(T - t1)}; at t1 = T a straddle, and as t1 -> 0 the dearer of call and put today."""
    assert qf.chooser_price(50.0, 50.0, 0.25, 0.5, 0.08, 0.0, 0.25) == pytest.approx(6.1071, abs=5e-5)
    assert qf.chooser_price(50.0, 50.0, 91 / 365, 182 / 365, 0.08, 0.0, 0.25) == pytest.approx(6.0984767237743185,
                                                                                            rel=1e-13)
    assert qf.chooser_price(100.0, 95.0, 120 / 365, 400 / 365, 0.04, 0.02, 0.3) == pytest.approx(19.136135774932672,
                                                                                              rel=1e-13)
    s, k, t1, t, r, q, v = 80.0, np.array([60.0, 80.0, 110.0]), 0.4, 1.3, 0.05, 0.02, 0.35
    identity = qf.bs_price("call", s, k, t, r, q, v) + math.exp(-q * (t - t1)) * qf.bs_price(
        "put", s, k * math.exp(-(r - q) * (t - t1)), t1, r, q, v)
    assert qf.chooser_price(s, k, t1, t, r, q, v) == pytest.approx(identity, rel=1e-13)
    straddle = qf.bs_price("call", s, k, t, r, q, v) + qf.bs_price("put", s, k, t, r, q, v)
    assert qf.chooser_price(s, k, t, t, r, q, v) == pytest.approx(straddle, rel=1e-13)
    dearer = np.maximum(qf.bs_price("call", s, k, t, r, q, v), qf.bs_price("put", s, k, t, r, q, v))
    assert qf.chooser_price(s, k, 1e-10, t, r, q, v) == pytest.approx(dearer, rel=1e-6)
    with pytest.raises(ValueError, match="choose_time"):
        qf.chooser_price(s, k, 1.5, t, r, q, v)
