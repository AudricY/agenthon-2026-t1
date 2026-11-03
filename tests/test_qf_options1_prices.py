"""Lane lib-options1's vanilla, exotic and two-asset pricers (runs/lanes/lib-options1-prompt.md; docs/research/
library-syllabus-1011.md rows 29, 76, 78, 86, 94, 95, 120 and docs/research/fresh-gaps-1012.md's row 4): the Bachelier
price and its normal implied vol, CEV, Roll-Geske-Whaley, the trinomial tree, the complex chooser, power options, the
Stulz rainbow and the quanto. References: Haug (2007)'s worked values as QuantLib's test suite cites them (its
quantooption.cpp, basketoption.cpp and chooseroption.cpp), QuantLib 1.43's engines (frozen in `QL`; `__main__`
recomputes them where QuantLib is installed), and closed-form identities (parity, limits, the payoff integrated
against the lognormal law). Every input is a textbook's or this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

import qf

pytestmark = pytest.mark.fast

QL = {   # QuantLib 1.43, Actual365Fixed, days from 2024-01-15 (`quantlib_values`)
    "bachelier": [   # bachelierBlackFormula(type, strike, forward, vol sqrt T, discount)
        (('call', 0.03, 0.025, 1.0, 0.008, 0.97), 0.006106499300064711),
        (('put', 0.03, 0.025, 1.0, 0.008, 0.97), 0.0012564993000647134),
        (('call', -0.004, 0.002, 2.5, 0.0065, 0.99), 0.001761819025728848),
        (('put', 100.0, 110.0, 0.5, 12.0, 0.98), 10.287462112952946),
    ],
    "cev": [   # AnalyticCEVEngine on the forward S e^{(r-q)T}, its alpha^2 T the spot model's variance there
        (('call', 100.0, 100.0, 365, 0.05, 0.0, 2.0, 0.5), 10.45388532871785),
        (('put', 100.0, 90.0, 365, 0.05, 0.02, 2.0, 0.5), 2.8713488429363507),
        (('call', 100.0, 110.0, 730, 0.03, 0.01, 0.002, 1.5), 0.03148088482113092),
        (('call', 100.0, 95.0, 365, 0.02, 0.0, 20.0, 0.0), 11.812515687270757),
        (('put', 50.0, 55.0, 365, 0.04, 0.01, 0.97, 0.7), 7.832355865671669),
        (('put', 100.0, 120.0, 1095, 0.0, 0.0, 0.0063, 1.8), 31.904980615534825),
        (('call', 1.0, 1.0, 365, 0.0, 0.0, 1.0, 1.5), 0.25041747748980936),
        (('put', 1.0, 1.0, 365, 0.0, 0.0, 1.0, 1.5), 0.3857527607264221),
    ],
}


def lognormal(s, t, r, q, v):
    return stats.lognorm(v * math.sqrt(t), scale=s * math.exp((r - q - 0.5 * v * v) * t))


def expectation(payoff, law, r, t, points=()):
    """e^{-rT} E[payoff(S_T)] by adaptive quadrature over the lognormal law."""
    low, high = law.ppf(1e-15), law.ppf(1.0 - 1e-15)
    return math.exp(-r * t) * integrate.quad(lambda x: payoff(x) * law.pdf(x), low, high, limit=400, epsabs=1e-13,
                                             epsrel=1e-13, points=[p for p in points if low < p < high] or None)[0]


# ------------------------------------------------------------------------------------------------ Bachelier

@pytest.mark.parametrize("args, expected", QL["bachelier"])
def test_bachelier_equals_quantlibs_formula(args, expected):
    assert qf.bachelier_price(*args) == pytest.approx(expected, rel=1e-13, abs=1e-16)


def test_bachelier_at_the_money_parity_and_no_vol():
    f, t, v, df = 0.02, 2.0, 0.007, 0.95
    assert qf.bachelier_price("call", f, f, t, v, df) == pytest.approx(df * v * math.sqrt(t / (2 * math.pi)), rel=1e-14)
    k = np.array([-0.01, 0.0, 0.015, 0.04])
    gap = qf.bachelier_price("call", f, k, t, v, df) - qf.bachelier_price("put", f, k, t, v, df)
    assert np.allclose(gap, df * (f - k), rtol=0, atol=1e-16)
    assert np.allclose(qf.bachelier_price("call", f, k, t, 0.0, df), df * np.maximum(f - k, 0.0), atol=0)
    assert qf.bachelier_price("put", f, 0.04, 0.0, v, df) == pytest.approx(df * 0.02, rel=1e-15)


@pytest.mark.parametrize("method", ["jaeckel", "bisection"])
def test_the_normal_implied_vol_inverts_bachelier(method):
    """A price made from a normal vol returns it, both sides of the money and a negative forward; at the money the
    vol is price sqrt(2 pi / T) / DF (Jaeckel 2017)."""
    f, t, df = np.array([0.03, 0.03, -0.005, 100.0]), np.array([1.0, 0.5, 2.0, 0.25]), 0.97
    k, v = np.array([0.02, 0.045, 0.01, 104.0]), np.array([0.008, 0.006, 0.0045, 9.0])
    for option in ("call", "put"):
        price = qf.bachelier_price(option, f, k, t, v, df)
        assert np.allclose(qf.normal_implied_vol(option, price, f, k, t, df, method), v, rtol=1e-12, atol=0)
    atm = qf.bachelier_price("call", 0.03, 0.03, 1.5, 0.0075, df)
    assert qf.normal_implied_vol("call", atm, 0.03, 0.03, 1.5, df, method) == pytest.approx(
        atm * math.sqrt(2 * math.pi / 1.5) / df, rel=1e-12)


def test_the_normal_implied_vol_is_nan_at_or_below_intrinsic_and_exact_deep_out_of_the_money():
    assert np.isnan(qf.normal_implied_vol("put", 0.009, 0.03, 0.04, 1.0, 0.9, "jaeckel"))
    assert np.isnan(qf.normal_implied_vol("call", 0.0, 0.03, 0.04, 1.0, 0.9, "bisection"))
    deep = qf.bachelier_price("call", 0.01, 0.06, 0.5, 0.005, 1.0)        # ten standard deviations out
    assert 0 < deep < 1e-20 and qf.normal_implied_vol("call", deep, 0.01, 0.06, 0.5, 1.0, "jaeckel") == pytest.approx(
        0.005, rel=1e-10)


# ------------------------------------------------------------------------------------------------ CEV

@pytest.mark.parametrize("args, expected", QL["cev"])
def test_cev_equals_quantlibs_analytic_engine(args, expected):
    option, s, k, days, r, q, alpha, beta = args
    found = qf.cev_price(option, s, k, days / 365, r, q, alpha, beta, "alpha")
    assert found["price"] == pytest.approx(expected, rel=1e-11) and found["alpha"] == alpha


def test_cev_parity_its_beta_one_limit_and_the_at_spot_vol():
    s, t, r, q = 100.0, 0.75, 0.04, 0.01
    for beta in (0.0, 0.5, 0.8, 1.3):
        call = qf.cev_price("call", s, 95.0, t, r, q, 0.25, beta, "at_spot")
        put = qf.cev_price("put", s, 95.0, t, r, q, 0.25, beta, "at_spot")
        assert call["price"] - put["price"] == pytest.approx(s * math.exp(-q * t) - 95.0 * math.exp(-r * t), rel=1e-11)
        assert call["alpha"] == pytest.approx(0.25 * s ** (1 - beta), rel=1e-15)
    bs = qf.bs_price("call", s, 95.0, t, r, q, 0.25)
    assert qf.cev_price("call", s, 95.0, t, r, q, 0.25, 1.0, "at_spot")["price"] == bs
    assert qf.cev_price("call", s, 95.0, t, r, q, 0.25, 0.999, "at_spot")["price"] == pytest.approx(bs, rel=2e-3)


def test_cev_below_one_makes_a_skew():
    """beta < 1: the vol falls as the strike rises (the leverage effect), read through Black-Scholes implied vols."""
    s, t, r = 100.0, 1.0, 0.03
    vols = [qf.implied_vol("call", qf.cev_price("call", s, k, t, r, 0.0, 0.2, 0.5, "at_spot")["price"], s, k, t, r,
                           0.0, "bisection") for k in (80.0, 100.0, 120.0)]
    assert vols[0] > vols[1] > vols[2] and vols[1] == pytest.approx(0.2, abs=5e-3)


def test_cev_above_one_uses_its_expected_stock_in_parity():
    """QuantLib 1.43 CEVCalculator's incomplete-gamma correction: for beta 1.5 and S=alpha=T=1,
    the discounted stock's martingale defect is exp(-2). Its pricing expectation is 1 - exp(-2), not 1.
    Source: ql/pricingengines/vanilla/analyticcevengine.cpp, CEVCalculator::value's delta > 2 branch."""
    call = qf.cev_price("call", 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.5, "alpha")["price"]
    put = qf.cev_price("put", 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.5, "alpha")["price"]
    assert call - put == pytest.approx(-math.exp(-2), rel=1e-13)


@pytest.mark.slow
def test_cev_lies_within_a_monte_carlo_of_its_sde():
    """An Euler scheme of dS = r S dt + alpha S^beta dW, absorbed at zero, fine in time (Cox 1975's process)."""
    rng = np.random.default_rng(7)
    s0, k, t, r, beta, alpha, steps, n = 100.0, 105.0, 1.0, 0.03, 0.5, 2.0, 400, 200_000
    s, dt = np.full(n, s0), t / steps
    for _ in range(steps):
        s = np.maximum(s + r * s * dt + alpha * np.sqrt(np.maximum(s, 0.0)) ** (2 * beta) * math.sqrt(dt)
                       * rng.standard_normal(n), 0.0)
    payoff = math.exp(-r * t) * np.maximum(s - k, 0.0)
    exact = qf.cev_price("call", s0, k, t, r, 0.0, alpha, beta, "alpha")["price"]
    assert abs(payoff.mean() - exact) < 4 * payoff.std() / math.sqrt(n) + 0.01


# ------------------------------------------------------------------------------------------------ Roll-Geske-Whaley

def rgw_by_quadrature(s, k, t, r, v, d, t1):
    """The escrowed model's American call exercised only just before t1: e^{-r t1} E[max(S'_{t1} + D - K,
    c(S'_{t1}, K, T - t1))], S' lognormal from S - D e^{-r t1}."""
    law = lognormal(s - d * math.exp(-r * t1), t1, r, 0.0, v)
    return expectation(lambda x: max(x + d - k, float(qf.bs_price("call", x, k, t - t1, r, 0.0, v))), law, r * 0 + r,
                       t1, points=(k,))


def test_roll_geske_whaley_gives_haugs_worked_value_and_the_exact_integral():
    """Haug (2007), the Roll-Geske-Whaley section: S 80, K 82, dividend 4 in three months, expiry in four, r 6%,
    vol 30%: 4.3860."""
    assert qf.roll_geske_whaley(80.0, 82.0, 4 / 12, 0.06, 0.3, 4.0, 3 / 12)["price"] == pytest.approx(4.3860, abs=5e-5)
    for args in ((80.0, 82.0, 1 / 3, 0.06, 0.3, 4.0, 0.25), (100.0, 95.0, 1.0, 0.05, 0.25, 3.0, 0.5)):
        assert qf.roll_geske_whaley(*args)["price"] == pytest.approx(rgw_by_quadrature(*args), rel=1e-9)


def test_roll_geske_whaley_bounds_and_its_no_exercise_case():
    found = qf.roll_geske_whaley(80.0, 82.0, 0.5, 0.06, 0.3, 4.0, 0.25)
    european = qf.escrowed_dividend_option("call", 80.0, 82.0, 0.5, 0.06, 0.3, [4.0], [0.25])
    assert found["european"] == pytest.approx(european["price"], rel=1e-14) and found["price"] > found["european"]
    assert found["net_spot"] == pytest.approx(80.0 - 4.0 * math.exp(-0.06 * 0.25), rel=1e-15)
    small = qf.roll_geske_whaley(100.0, 100.0, 1.0, 0.05, 0.2, 0.5, 0.5)      # D below K (1 - e^{-r(T - t1)})
    assert small["price"] == small["european"] and math.isinf(small["critical_spot"])
    none = qf.roll_geske_whaley(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 0.5)
    assert none["price"] == pytest.approx(qf.bs_price("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2), rel=1e-14)


@pytest.mark.parametrize("dividend", [10.0, 12.0])
def test_roll_geske_whaley_exercises_every_path_when_dividend_reaches_strike(dividend):
    """Lu (2026), arXiv:2607.24973, section 3.2: D >= K makes the dividend-date exercise payoff always exceed
    continuation, giving the discounted mean S - K exp(-r t1); the exercise boundary in the net process is zero."""
    s, k, t, r, v, t1 = 100.0, 10.0, 1.0, 0.03, 0.25, 0.5
    found = qf.roll_geske_whaley(s, k, t, r, v, dividend, t1)
    assert found["price"] == pytest.approx(s - k * math.exp(-r * t1), rel=1e-14)
    assert found["critical_spot"] == 0.0 and found["price"] >= found["european"]


def test_roll_geske_whaley_refuses_rates_where_other_exercise_times_matter():
    """Lu (2026), Proposition 2.8(vi), restricts dividend-date-only exercise to positive rates. At negative rates
    a no-dividend American call can exercise now; its value cannot be the dividend-date-only expectation."""
    with pytest.raises(ValueError, match="rate must not be negative"):
        qf.roll_geske_whaley(200.0, 100.0, 1.0, -0.05, 0.2, 0.0, 0.5)


# ------------------------------------------------------------------------------------------------ trinomial

def test_the_trinomial_tree_converges_to_black_scholes_and_a_fine_crr_tree():
    s, k, t, r, q, v = 50.0, 52.0, 2.0, 0.05, 0.01, 0.3
    for option in ("call", "put"):
        tri = qf.binomial_price(option, "european", "trinomial", s, k, t, r, q, v, 1000)
        exact = qf.bs_greeks(option, s, k, t, r, q, v, 1.0, "year", 1.0)
        assert tri["price"] == pytest.approx(exact["price"], abs=1.5e-3)
        assert tri["delta"] == pytest.approx(exact["delta"], abs=2e-3)
        assert tri["gamma"] == pytest.approx(exact["gamma"], abs=2e-4)
        assert tri["theta"] == pytest.approx(exact["theta"], abs=2e-2)
    american = qf.binomial_price("put", "american", "trinomial", s, k, t, r, q, v, 1000)["price"]
    crr = qf.binomial_price("put", "american", "crr", s, k, t, r, q, v, 2000)["price"]
    assert american == pytest.approx(crr, abs=3e-3) and american > qf.bs_price("put", s, k, t, r, q, v)


def test_hulls_trinomial_probabilities_are_used():
    """One step: the price is the discounted mean over Hull's three nodes, u = e^{vol sqrt(3 dt)}."""
    s, k, t, r, q, v = 100.0, 100.0, 0.5, 0.04, 0.0, 0.2
    found = qf.binomial_price("call", "european", "trinomial", s, k, t, r, q, v, 2)["price"]
    dt = t / 2
    u, tilt = math.exp(v * math.sqrt(3 * dt)), (r - q - v * v / 2) * math.sqrt(dt / (12 * v * v))
    p = np.array([1 / 6 - tilt, 2 / 3, 1 / 6 + tilt])
    last = np.maximum(s * u ** np.arange(-2, 3) - k, 0.0)
    mid = math.exp(-r * dt) * np.array([p @ last[i:i + 3] for i in range(3)])
    assert found == pytest.approx(math.exp(-r * dt) * p @ mid, rel=1e-14)


# ------------------------------------------------------------------------------------------------ exotics

def test_the_complex_chooser_gives_haugs_worked_value_and_the_simple_chooser():
    """Haug (2007): S 50, Kc 55, Kp 48, t three months, Tc six, Tp seven, r 10%, b 5%, vol 35%: 6.0508."""
    found = qf.complex_chooser_price(50.0, 55.0, 48.0, 3 / 12, 6 / 12, 7 / 12, 0.10, 0.05, 0.35)
    assert found["price"] == pytest.approx(6.0508, abs=5e-5)
    simple = qf.complex_chooser_price(100.0, 100.0, 100.0, 0.25, 1.0, 1.0, 0.03, 0.01, 0.25)["price"]
    assert simple == pytest.approx(qf.chooser_price(100.0, 100.0, 0.25, 1.0, 0.03, 0.01, 0.25), rel=1e-9)


def test_the_complex_choosers_critical_spot_makes_the_legs_equal():
    found = qf.complex_chooser_price(50.0, 55.0, 48.0, 0.25, 0.5, 0.5833, 0.10, 0.05, 0.35)
    i = found["critical_spot"]
    assert qf.bs_price("call", i, 55.0, 0.25, 0.10, 0.05, 0.35) == pytest.approx(
        qf.bs_price("put", i, 48.0, 0.3333, 0.10, 0.05, 0.35), rel=1e-10)


@pytest.mark.parametrize("option", ["call", "put"])
def test_power_options_are_their_payoffs_expectation(option):
    s, t, r, q, v = 100.0, 0.5, 0.05, 0.01, 0.2
    law, sign = lognormal(s, t, r, q, v), 1.0 if option == "call" else -1.0
    assert qf.power_price(option, "standard", s, 10000.0, t, r, q, v, 2.0, None) == pytest.approx(
        expectation(lambda x: max(sign * (x * x - 10000.0), 0.0), law, r, t, (100.0,)), rel=1e-10)
    assert qf.power_price(option, "capped", s, 10000.0, t, r, q, v, 2.0, 1500.0) == pytest.approx(
        expectation(lambda x: min(max(sign * (x * x - 10000.0), 0.0), 1500.0), law, r, t, (100.0, 107.0, 92.0)),
        rel=1e-10)
    assert qf.power_price(option, "powered", s, 100.0, t, r, q, v, 3.0, None) == pytest.approx(
        expectation(lambda x: max(sign * (x - 100.0), 0.0) ** 3, law, r, t, (100.0,)), rel=1e-10)
    assert qf.power_price(option, "standard", s, 0.01, t, r, q, v, -1.0, None) == pytest.approx(
        expectation(lambda x: max(sign * (1.0 / x - 0.01), 0.0), law, r, t, (100.0,)), rel=1e-10)


def test_power_one_is_black_scholes_in_every_form():
    s, k, t, r, q, v = 100.0, 95.0, 0.5, 0.05, 0.01, 0.2
    for option in ("call", "put"):
        bs = qf.bs_price(option, s, k, t, r, q, v)
        assert qf.power_price(option, "standard", s, k, t, r, q, v, 1.0, None) == pytest.approx(bs, rel=1e-14)
        assert qf.power_price(option, "powered", s, k, t, r, q, v, 1.0, None) == pytest.approx(bs, rel=1e-13)
    spread = qf.bs_price("call", s, k, t, r, q, v) - qf.bs_price("call", s, k + 10.0, t, r, q, v)
    assert qf.power_price("call", "capped", s, k, t, r, q, v, 1.0, 10.0) == pytest.approx(spread, rel=1e-14)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("power", [6, 8, 10])
def test_powered_options_keep_small_payoffs_positive_without_moment_cancellation(option, power):
    """Haug's powered payoff max(+-(S_T-K),0)^i, independently integrated in the standard normal variable.
    Own inputs at small volatility make the truncated-moment expansion subtract nearly equal large terms."""
    s, k, t, r, q, v = 100.0, 100.0, 0.5, 0.05, 0.01, 0.01
    sign = 1.0 if option == "call" else -1.0
    mu, sd = math.log(s) + (r - q - v * v / 2) * t, v * math.sqrt(t)
    boundary = (math.log(k) - mu) / sd
    low, high = (boundary, 12.0) if option == "call" else (-12.0, boundary)
    reference = math.exp(-r * t) * integrate.quad(
        lambda z: max(sign * (math.exp(mu + sd * z) - k), 0.0) ** power * math.exp(-z * z / 2) / math.sqrt(2 * math.pi),
        low, high, epsabs=1e-13, epsrel=1e-12)[0]
    found = qf.power_price(option, "powered", s, k, t, r, q, v, power, None)
    assert found > 0 and found == pytest.approx(reference, rel=1e-9)



def test_a_power_call_struck_at_or_below_zero_is_its_discounted_mean():
    s, t, r, q, v = 100.0, 0.5, 0.05, 0.01, 0.2
    mean = s ** 2 * math.exp(2 * (r - q - v * v / 2) * t + 2 * v * v * t)
    assert qf.power_price("call", "standard", s, -50.0, t, r, q, v, 2.0, None) == pytest.approx(
        math.exp(-r * t) * (mean + 50.0), rel=1e-14)
    assert qf.power_price("put", "standard", s, 0.0, t, r, q, v, 2.0, None) == 0.0

# ------------------------------------------------------------------------------------------------ two assets

HAUG_STULZ = [   # Haug (1998), pp. 56-58, as QuantLib's basketoption.cpp cites them: K 98, S1 100, S2 105, r 5%,
    #              T 0.5, vols 11% and 16%, rho 0.63, yields (0, 0) and (6%, 9%)
    ("minimum", "call", 0.0, 0.0, 4.8177), ("maximum", "call", 0.0, 0.0, 11.6323), ("minimum", "put", 0.0, 0.0, 2.0376),
    ("maximum", "put", 0.0, 0.0, 0.5731), ("minimum", "call", 0.06, 0.09, 2.9340), ("minimum", "put", 0.06, 0.09, 3.5224),
    ("maximum", "call", 0.06, 0.09, 8.0701), ("maximum", "put", 0.06, 0.09, 1.2181)]


@pytest.mark.parametrize("on, option, q1, q2, expected", HAUG_STULZ)
def test_the_rainbow_gives_haugs_table(on, option, q1, q2, expected):
    found = qf.rainbow_price(option, on, 100.0, 105.0, 98.0, 0.5, 0.05, q1, q2, 0.11, 0.16, 0.63)
    assert found == pytest.approx(expected, abs=1e-4)


def test_the_rainbows_identities():
    """Call on the max plus call on the min is the two calls (Stulz 1982); strike 0 is the max's or min's present
    value; puts by parity."""
    args = (100.0, 95.0, 100.0, 0.75, 0.04, 0.01, 0.02, 0.25, 0.35, -0.4)
    s1, s2, k, t, r, q1, q2, v1, v2, rho = args
    calls = qf.bs_price("call", s1, k, t, r, q1, v1) + qf.bs_price("call", s2, k, t, r, q2, v2)
    both = qf.rainbow_price("call", "maximum", *args) + qf.rainbow_price("call", "minimum", *args)
    assert both == pytest.approx(calls, rel=1e-10)
    best = qf.rainbow_price("call", "maximum", s1, s2, 0.0, t, r, q1, q2, v1, v2, rho)
    exchange = qf.spread_price("margrabe", "call", s1, s2, 0.0, t, r, q1, q2, v1, v2, rho)["price"]
    assert best == pytest.approx(s2 * math.exp(-q2 * t) + exchange, rel=1e-13)
    for on in ("maximum", "minimum"):
        gap = qf.rainbow_price("call", on, *args) - qf.rainbow_price("put", on, *args)
        worth = qf.rainbow_price("call", on, s1, s2, 0.0, t, r, q1, q2, v1, v2, rho)
        assert gap == pytest.approx(worth - k * math.exp(-r * t), rel=1e-12)


def test_the_quanto_gives_haugs_worked_value_and_its_put():
    """Haug (1998), pp. 105-106, as QuantLib's quantooption.cpp cites it: S 100, K 105, T 0.5, r_d 8%, r_f 5%, q 4%,
    vol 20%, vol_X 10%, rho 0.3, fixed rate 1.5: call 5.3280; the put per unit of the fixed rate 8.1636 (Haug's VBA)."""
    found = qf.quanto_price("call", "quanto", 100.0, 105.0, 0.5, 0.08, 0.05, 0.04, 0.2, 0.1, 0.3,
                            "domestic_per_foreign", None, 1.5)
    assert found["price"] == pytest.approx(5.3280, abs=5e-5)
    put = qf.quanto_price("put", "quanto", 100.0, 105.0, 0.5, 0.08, 0.05, 0.04, 0.2, 0.1, 0.3,
                          "domestic_per_foreign", None, 1.0)
    assert put["price"] == pytest.approx(8.1636, abs=5e-5)


def test_the_quantos_identities_and_its_quote_direction():
    s, k, t, rd, rf, q, v, vx = 100.0, 95.0, 0.75, 0.03, 0.01, 0.02, 0.25, 0.12
    flat = qf.quanto_price("call", "quanto", s, k, t, rd, rf, q, v, vx, 0.0, "domestic_per_foreign", None, 2.0)
    assert flat["price"] == pytest.approx(2.0 * math.exp((rf - rd) * t) * qf.bs_price("call", s, k, t, rf, q, v),
                                          rel=1e-13)
    up = qf.quanto_price("call", "quanto", s, k, t, rd, rf, q, v, vx, 0.4, "domestic_per_foreign", None, 2.0)
    inverted = qf.quanto_price("call", "quanto", s, k, t, rd, rf, q, v, vx, -0.4, "foreign_per_domestic", None, 0.5)
    assert inverted["price"] == pytest.approx(up["price"], rel=1e-14) and up["price"] < flat["price"]
    compo = qf.quanto_price("call", "compo", s, 150.0, t, rd, rf, q, v, vx, 0.4, "domestic_per_foreign", 1.5, None)
    combined = math.sqrt(v * v + 2 * 0.4 * v * vx + vx * vx)
    assert compo["price"] == pytest.approx(qf.bs_price("call", 150.0, 150.0, t, rd, q, combined), rel=1e-14)


def test_the_quanto_and_compo_lie_within_a_simulation_of_their_payoffs():
    """The asset and the rate sampled jointly at T under the domestic measure (exact lognormals)."""
    s, x0, t, rd, rf, q, v, vx, rho = 100.0, 1.5, 0.5, 0.08, 0.05, 0.04, 0.2, 0.12, 0.45
    z = np.random.default_rng(5).standard_normal((2, 200_000))
    w = rho * z[0] + math.sqrt(1 - rho * rho) * z[1]
    s_t = s * np.exp((rf - q - rho * v * vx - v * v / 2) * t + v * math.sqrt(t) * z[0])
    x_t = x0 * np.exp((rd - rf - vx * vx / 2) * t + vx * math.sqrt(t) * w)
    for payoff, strike, fx_rate, fixed, paid in (("quanto", 105.0, None, 1.5, 1.5 * np.maximum(s_t - 105.0, 0.0)),
                                                  ("compo", 160.0, x0, None, np.maximum(s_t * x_t - 160.0, 0.0))):
        found = qf.quanto_price("call", payoff, s, strike, t, rd, rf, q, v, vx, rho, "domestic_per_foreign", fx_rate,
                                fixed)["price"]
        sample = math.exp(-rd * t) * paid
        assert abs(sample.mean() - found) < 4 * sample.std() / math.sqrt(sample.size), payoff


def test_the_conventions_are_refused_when_unnamed_or_unfit():
    with pytest.raises(ValueError):
        qf.cev_price("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 0.5, "atm")
    with pytest.raises(ValueError):
        qf.normal_implied_vol("call", 0.01, 0.03, 0.03, 1.0, 1.0, "newton")
    with pytest.raises(ValueError):
        qf.quanto_price("call", "quanto", 100.0, 100.0, 1.0, 0.05, 0.02, 0.0, 0.2, 0.1, 0.3, "domestic_per_foreign",
                        1.5, 1.5)
    with pytest.raises(ValueError):
        qf.power_price("call", "capped", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 2.0, None)
    with pytest.raises(ValueError):
        qf.rainbow_price("call", "best", 100.0, 100.0, 100.0, 1.0, 0.05, 0.0, 0.0, 0.2, 0.2, 0.5)
    with pytest.raises(ValueError):
        qf.roll_geske_whaley(100.0, 100.0, 1.0, 0.05, 0.2, 2.0, 1.5)


def quantlib_values() -> dict:
    """The frozen `QL` values recomputed by QuantLib (scratch: a Python with QuantLib 1.43)."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    count = ql.Actual365Fixed()
    kind = {"call": ql.Option.Call, "put": ql.Option.Put}
    found = {"bachelier": [((o, f, k, t, v, df), ql.bachelierBlackFormula(kind[o], k, f, v * math.sqrt(t), df))
                           for (o, f, k, t, v, df), _ in QL["bachelier"]], "cev": []}
    for (o, s, k, days, r, q, alpha, beta), _ in QL["cev"]:
        t, carry = days / 365, (r - q) * (beta - 1)
        variance = alpha ** 2 * (math.expm1(2 * carry * t) / (2 * carry) if carry else t)
        forward_variance = variance * math.exp(2 * (r - q) * (1 - beta) * t)
        option = ql.VanillaOption(ql.PlainVanillaPayoff(kind[o], k), ql.EuropeanExercise(today + days))
        option.setPricingEngine(ql.AnalyticCEVEngine(s * math.exp((r - q) * t), math.sqrt(forward_variance / t), beta,
                                                     ql.YieldTermStructureHandle(ql.FlatForward(today, r, count))))
        found["cev"].append(((o, s, k, days, r, q, alpha, beta), option.NPV()))
    return found


if __name__ == "__main__":
    fresh = quantlib_values()
    print("QL frozen values match QuantLib:", all(
        np.allclose([v for _, v in fresh[key]], [v for _, v in QL[key]], rtol=1e-14, atol=0) for key in QL))
