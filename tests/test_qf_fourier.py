"""agent/qf/fourier.py (runs/lanes/lib5-prompt.md): Gil-Pelaez inversion of Black-Scholes's own characteristic function
against qf.bs_price; Heston's in its stable form against QuantLib 1.43's AnalyticHestonEngine (values frozen in `QL`;
running this file under a Python with QuantLib recomputes them), Fang and Oosterlee's (2008) published reference,
Black-Scholes at a vanishing vol of variance, Heston's (1993) original form where its branch does not jump, and the
mean of ln S_T. Every parameter is a textbook example or chosen freely; no unit's values appear."""
import math

import numpy as np
import pytest

import qf
import qf.fourier as F
from test_qf_bs import check_cards

pytestmark = pytest.mark.fast

QL = {   # QuantLib 1.43, AnalyticHestonEngine (relative tolerance 1e-14), ACT/365F, 365 or 3,650 days (see __main__)
    "one_year": dict(call=(23.161630420020003, 8.651774867026562, 1.5227282678140976),
                     put=(1.7922897289838593, 6.691344846960579, 18.97120891871828)),
    "ten_years": dict(call=(55.61060857292433, 30.17476972437284, 4.110125041066052)),
}
ONE_YEAR = dict(spot=100.0, maturity=1.0, rate=0.03, dividend_yield=0.01, v0=0.04, kappa=2.0, theta=0.05, sigma=0.6,
                rho=-0.6)
TEN_YEARS = dict(spot=100.0, maturity=10.0, rate=0.02, dividend_yield=0.0, v0=0.09, kappa=0.5, theta=0.06, sigma=1.0,
                 rho=-0.9)
STRIKES = {"one_year": (80.0, 100.0, 120.0), "ten_years": (60.0, 100.0, 160.0)}


def heston(point, factor_only=False):
    return lambda u: F.heston_charfn(u, **point, factor_only=factor_only)


def priced(charfn, option, point, strike, quadrature="gauss_legendre", upper=200.0, nodes=256):
    return F.charfn_price(charfn, option, point["spot"], strike, point["maturity"], point["rate"],
                          point["dividend_yield"], quadrature, upper, nodes)


def black_scholes(spot, maturity, rate, dividend_yield, vol):
    """phi of ln S_T under Black-Scholes: ln S_T ~ N(ln S + (r - q - vol^2/2) T, vol^2 T)."""
    return lambda u: np.exp(1j * u * (math.log(spot) + (rate - dividend_yield - vol * vol / 2) * maturity)
                            - vol * vol * maturity * u * u / 2)


def test_the_fourier_cards_keep_the_format_and_their_examples_run():
    check_cards(F)


@pytest.mark.parametrize("quadrature, nodes", [("gauss_legendre", 128), ("quad", 200)])
def test_inverting_black_scholes_characteristic_function_gives_bs_price(quadrature, nodes):
    """Gil-Pelaez on phi_BS is Black-Scholes-Merton: calls and puts across moneyness, either quadrature."""
    point = dict(spot=100.0, maturity=0.75, rate=0.03, dividend_yield=0.01)
    strikes = np.array([60.0, 90.0, 100.0, 110.0, 150.0])
    charfn = black_scholes(**point, vol=0.25)
    for option in ("call", "put"):
        found = priced(charfn, option, point, strikes, quadrature, 100.0, nodes)
        assert found["price"] == pytest.approx(qf.bs_price(option, 100.0, strikes, 0.75, 0.03, 0.01, 0.25), abs=1e-10)
    d1 = (np.log(100.0 / strikes) + (0.03 - 0.01 + 0.25 ** 2 / 2) * 0.75) / (0.25 * math.sqrt(0.75))
    probabilities = priced(charfn, "call", point, strikes, quadrature, 100.0, nodes)
    assert probabilities["p1"] == pytest.approx(qf.digital_price("asset", "call", 100.0, strikes, 0.75, 0.03, 0.01,
                                                                 0.25) / (100.0 * math.exp(-0.0075)), abs=1e-10)
    assert probabilities["p2"] == pytest.approx(0.5 * np.vectorize(math.erfc)(-(d1 - 0.25 * math.sqrt(0.75))
                                                                               / math.sqrt(2)), abs=1e-10)


@pytest.mark.parametrize("case, point", [("one_year", ONE_YEAR), ("ten_years", TEN_YEARS)])
def test_heston_prices_equal_quantlibs_analytic_heston_engine(case, point):
    for option, expected in QL[case].items():
        found = priced(heston(point), option, point, np.array(STRIKES[case]), nodes=512)
        assert found["price"] == pytest.approx(expected, rel=1e-9, abs=1e-9), (case, option)


def test_heston_gives_fang_and_oosterlees_reference_value():
    """Fang and Oosterlee (2008), SIAM J. Sci. Comput. 31(2), 5.3: S 100, K 100, T 1, r 0, kappa 1.5768, vol of
    variance 0.5751, theta 0.0398, v0 0.0175, rho -0.5711; reference 5.785155450 (QuantLib: 5.7851554344)."""
    point = dict(spot=100.0, maturity=1.0, rate=0.0, dividend_yield=0.0, v0=0.0175, kappa=1.5768, theta=0.0398,
                 sigma=0.5751, rho=-0.5711)
    for quadrature, nodes in (("gauss_legendre", 256), ("quad", 200)):
        assert priced(heston(point), "call", point, 100.0, quadrature, 200.0, nodes)["price"] == pytest.approx(
            5.785155450, abs=2e-8)


def test_heston_at_a_vanishing_vol_of_variance_is_black_scholes_at_the_integrated_variance():
    """sigma -> 0: v is deterministic, v(t) = theta + (v0 - theta) e^{-kappa t}, and the price is Black-Scholes at
    the mean variance (1/T) int_0^T v dt; at rho 0 the error is of order sigma^2."""
    point = dict(spot=100.0, maturity=0.8, rate=0.04, dividend_yield=0.02, v0=0.09, kappa=1.2, theta=0.04, sigma=1e-4,
                 rho=0.0)
    strikes = np.array([70.0, 95.0, 100.0, 130.0])
    mean = (0.04 * 0.8 + (0.09 - 0.04) * (1 - math.exp(-1.2 * 0.8)) / 1.2) / 0.8
    for option in ("call", "put"):
        found = priced(heston(point), option, point, strikes)["price"]
        assert found == pytest.approx(qf.bs_price(option, 100.0, strikes, 0.8, 0.04, 0.02, math.sqrt(mean)), abs=1e-7)


def test_the_stable_form_is_hestons_original_where_its_branch_does_not_jump_and_ln_s_has_its_mean():
    """Heston (1993): g = (xi + d)/(xi - d) and e^{+dT}, the same function at a short maturity; -i phi'(0) =
    E[ln S_T] = ln S + (r - q) T - (1/2) E int v dt; phi(0) = 1 and phi(-i) = S e^{(r-q)T}."""
    s, t, r, q, v0, kappa, theta, sigma, rho = (ONE_YEAR[key] for key in ONE_YEAR)
    u = np.linspace(0.01, 6.0, 60)
    iu = 1j * u
    xi = kappa - sigma * rho * iu
    d = np.sqrt(xi * xi + sigma ** 2 * (u * u + iu))
    g = (xi + d) / (xi - d)
    c = kappa * theta / sigma ** 2 * ((xi + d) * t - 2 * np.log((1 - g * np.exp(d * t)) / (1 - g)))
    big_d = (xi + d) / sigma ** 2 * (1 - np.exp(d * t)) / (1 - g * np.exp(d * t))
    original = np.exp(iu * (math.log(s) + (r - q) * t) + c + big_d * v0)
    assert np.max(np.abs(F.heston_charfn(u, **ONE_YEAR, factor_only=False) - original)) < 1e-13
    h = 1e-5
    phi = heston(ONE_YEAR)
    slope = (phi(h) - phi(-h)) / (2 * h)
    mean_variance = theta * t + (v0 - theta) * (1 - math.exp(-kappa * t)) / kappa
    assert (-1j * slope).real == pytest.approx(math.log(s) + (r - q) * t - mean_variance / 2, abs=1e-8)
    assert F.heston_charfn(0.0, **ONE_YEAR, factor_only=False) == 1.0
    assert F.heston_charfn(-1j, **ONE_YEAR, factor_only=False) == pytest.approx(s * math.exp((r - q) * t), rel=1e-14)
    steep = dict(ONE_YEAR, kappa=0.5, sigma=1.0, rho=0.9)       # sigma rho > kappa: xi + d is 0 at u = -i
    assert F.heston_charfn(-1j, **steep, factor_only=True) == 1.0


def test_a_two_factor_model_is_one_charfn_times_the_others_factor_and_prices_as_one():
    """Christoffersen, Heston and Jacobs (2009): independent variance factors multiply. The factor alone carries no
    drift (its value at -i is 1), so the product keeps the forward and E[ln S_T] loses half of both mean variances."""
    first = dict(ONE_YEAR)
    second = dict(ONE_YEAR, v0=0.01, kappa=0.3, theta=0.02, sigma=0.2, rho=-0.2)
    def both(u):
        return F.heston_charfn(u, **first, factor_only=False) * F.heston_charfn(u, **second, factor_only=True)
    assert F.heston_charfn(-1j, **second, factor_only=True) == 1.0 and F.heston_charfn(0.0, **second,
                                                                                       factor_only=True) == 1.0
    h, t = 1e-5, 1.0
    mean = sum(p["theta"] * t + (p["v0"] - p["theta"]) * (1 - math.exp(-p["kappa"] * t)) / p["kappa"]
               for p in (first, second))
    assert (-1j * (both(h) - both(-h)) / (2 * h)).real == pytest.approx(math.log(100.0) + 0.02 - mean / 2, abs=1e-8)
    call = priced(both, "call", first, 100.0)["price"]
    put = priced(both, "put", first, 100.0)["price"]
    assert call - put == pytest.approx(100.0 * math.exp(-0.01) - 100.0 * math.exp(-0.03), abs=1e-10)
    assert call > priced(heston(first), "call", first, 100.0)["price"]          # more variance, a dearer option


def test_a_charfn_whose_forward_is_not_the_pricers_is_refused_and_bad_arguments_too():
    point = dict(spot=100.0, maturity=1.0, rate=0.03, dividend_yield=0.0)
    twice = lambda u: black_scholes(**point, vol=0.2)(u) * np.exp(-1j * u * 0.02)     # the Ito term taken twice
    with pytest.raises(ValueError, match="forward"):
        priced(twice, "call", point, 100.0)
    with pytest.raises(ValueError, match="quadrature"):
        priced(black_scholes(**point, vol=0.2), "call", point, 100.0, "simpson")
    with pytest.raises(ValueError, match="nodes"):
        priced(black_scholes(**point, vol=0.2), "call", point, 100.0, nodes=1)
    with pytest.raises(ValueError, match="sigma"):
        F.heston_charfn(1.0, **dict(ONE_YEAR, sigma=0.0), factor_only=False)
    with pytest.raises(ValueError, match="rho"):
        F.heston_charfn(1.0, **dict(ONE_YEAR, rho=-1.5), factor_only=False)


if __name__ == "__main__":
    import QuantLib as ql
    today = ql.Date(2, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    for case, point in (("one_year", ONE_YEAR), ("ten_years", TEN_YEARS)):
        curve = {key: ql.YieldTermStructureHandle(ql.FlatForward(today, point[key], dc))
                 for key in ("rate", "dividend_yield")}
        process = ql.HestonProcess(curve["rate"], curve["dividend_yield"], ql.QuoteHandle(ql.SimpleQuote(100.0)),
                                   *(point[key] for key in ("v0", "kappa", "theta", "sigma", "rho")))
        engine = ql.AnalyticHestonEngine(ql.HestonModel(process), 1e-14, 100000)
        for option, expected in QL[case].items():
            fresh = []
            for strike in STRIKES[case]:
                kind = ql.Option.Call if option == "call" else ql.Option.Put
                deal = ql.VanillaOption(ql.PlainVanillaPayoff(kind, strike),
                                        ql.EuropeanExercise(today + int(round(point["maturity"] * 365))))
                deal.setPricingEngine(engine)
                fresh.append(deal.NPV())
            print(case, option, np.allclose(fresh, expected, rtol=1e-14, atol=0))


def test_the_cards_variance_swap_strike_is_the_charfns_expected_integrated_variance():
    """Lane lib6's line on heston_charfn: E[ln S_T] = ln S + (r - q) T - E[int_0^T v dt] / 2, and E[ln S_T] is
    -i phi'(0); the fair annual variance theta + (v0 - theta)(1 - e^{-kappa T}) / (kappa T) is E[int v] / T."""
    s, t, rate, q, v0, kappa, theta, sigma, rho = 100.0, 1.7, 0.03, 0.01, 0.09, 2.2, 0.04, 0.6, -0.5
    h = 1e-5
    slope = (qf.fourier.heston_charfn(h, s, t, rate, q, v0, kappa, theta, sigma, rho, False)
             - qf.fourier.heston_charfn(-h, s, t, rate, q, v0, kappa, theta, sigma, rho, False)) / (2 * h)
    mean_log = (-1j * slope).real
    strike = 2.0 * (math.log(s) + (rate - q) * t - mean_log) / t
    assert strike == pytest.approx(theta + (v0 - theta) * (1 - math.exp(-kappa * t)) / (kappa * t), rel=1e-7)


def test_the_gauss_legendre_nodes_are_solved_once_per_count_and_kept_read_only():
    """Lane lib7: leggauss's eigenproblem dominated a calibration's pricing passes; the nodes are kept per count."""
    first, again = F._gauss_legendre(48), F._gauss_legendre(48)
    assert first is again and not first[0].flags.writeable and not first[1].flags.writeable
    x, w = np.polynomial.legendre.leggauss(48)
    assert np.array_equal(first[0], x) and np.array_equal(first[1], w)
