"""qf.montecarlo.pathwise_greeks and likelihood_ratio_greeks (runs/lanes/lib7-prompt.md, rank 16 of
docs/research/library-syllabus-1011.md): each estimator's mean against the Black-Scholes Greeks (qf.bs.bs_greeks,
and a digital's closed-form Greeks differentiated here) within four standard errors, the scalings, and each sample's
definition on a few paths written out (Glasserman 2004, 7.2 and 7.3). Every input is this file's own; no unit's
values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf
import qf.bs
import qf.montecarlo as MC

pytestmark = pytest.mark.fast

CASE = dict(spot=100.0, strike=105.0, maturity=0.75, rate=0.04, dividend_yield=0.015, vol=0.3)
Z = np.random.default_rng(17).standard_normal(400_000)


@pytest.mark.parametrize("option", ["call", "put"])
def test_the_pathwise_greeks_are_the_black_scholes_greeks_within_their_errors(option):
    found = MC.pathwise_greeks(option, **CASE, normals=Z, vega_per=1.0, rho_per=1.0)
    exact = qf.bs.bs_greeks(option, **CASE, vega_per=1.0, theta_per="year", rho_per=1.0)
    for greek in ("price", "delta", "gamma", "vega", "rho"):
        assert abs(found[greek] - exact[greek]) < 4 * found[f"{greek}_se"], greek
    assert found["delta_se"] < 1e-3 and found["vega_se"] < 0.2 and found["rho_se"] < 0.1


def test_each_pathwise_sample_is_its_derivative_along_the_path():
    """Three paths written out: S_T, then each payoff's derivative in S, vol and r with Z held, by a central
    difference of the discounted payoff along the path (the call's kink not between the bumps)."""
    z = np.array([0.8, -0.4, 1.9])
    h = 1e-6

    def payoff(spot=CASE["spot"], vol=CASE["vol"], rate=CASE["rate"]):
        t = CASE["maturity"]
        s_t = spot * np.exp((rate - CASE["dividend_yield"] - vol * vol / 2) * t + vol * math.sqrt(t) * z)
        return math.exp(-rate * t) * np.maximum(s_t - CASE["strike"], 0.0)

    for path in range(3):
        one = MC.pathwise_greeks("call", **CASE, normals=np.r_[z[path], z[path]], vega_per=1.0, rho_per=1.0)
        for greek, name in (("delta", "spot"), ("vega", "vol"), ("rho", "rate")):
            up = payoff(**{name: CASE[name] + h})[path]
            down = payoff(**{name: CASE[name] - h})[path]
            assert one[greek] == pytest.approx((up - down) / (2 * h), rel=1e-6, abs=1e-9), (path, greek)


def test_the_scalings_and_call_less_put():
    unit = MC.pathwise_greeks("call", **CASE, normals=Z[:1000], vega_per=1.0, rho_per=1.0)
    point = MC.pathwise_greeks("call", **CASE, normals=Z[:1000], vega_per=0.01, rho_per=0.01)
    for greek in ("vega", "rho", "vega_se", "rho_se"):
        assert point[greek] == pytest.approx(unit[greek] / 100, rel=1e-14)
    put = MC.pathwise_greeks("put", **CASE, normals=Z[:1000], vega_per=1.0, rho_per=1.0)
    t = CASE["maturity"]
    s_t = CASE["spot"] * np.exp((CASE["rate"] - CASE["dividend_yield"] - 0.045) * t + 0.3 * math.sqrt(t) * Z[:1000])
    assert unit["delta"] - put["delta"] == pytest.approx(math.exp(-CASE["rate"] * t) * np.mean(s_t) / 100, rel=1e-13)


@pytest.mark.parametrize("bad, match", [(dict(option="digital"), "option"), (dict(vega_per=100), "vega_per"),
                                        (dict(normals=[0.1]), "normals"), (dict(vol=0.0), "vol")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        MC.pathwise_greeks(**{**dict(option="call", **CASE, normals=Z[:10], vega_per=1.0, rho_per=1.0), **bad})


# ------------------------------------------------------------------------------------------------ likelihood ratio


def bumped(option, greek, h=1e-4):
    """A closed-form Greek by central differences of qf.bs_price or qf.digital_price (cash, paying 1)."""
    def value(**changes):
        args = {**CASE, **changes}
        if option.startswith("digital"):
            return qf.digital_price("cash", option.split("_")[1], args["spot"], args["strike"], args["maturity"],
                                    args["rate"], args["dividend_yield"], args["vol"])
        return qf.bs_price(option, args["spot"], args["strike"], args["maturity"], args["rate"],
                           args["dividend_yield"], args["vol"])

    if greek == "price":
        return value()
    name = {"delta": "spot", "gamma": "spot", "vega": "vol", "rho": "rate"}[greek]
    step = h * (CASE["spot"] if name == "spot" else 1.0)
    up, down = value(**{name: CASE[name] + step}), value(**{name: CASE[name] - step})
    if greek == "gamma":
        return (up - 2 * value() + down) / step ** 2
    return (up - down) / (2 * step)


@pytest.mark.parametrize("option", ["call", "put", "digital_call", "digital_put"])
def test_the_likelihood_ratio_greeks_are_the_closed_forms_within_their_errors(option):
    found = MC.likelihood_ratio_greeks(option, **CASE, normals=Z, vega_per=1.0, rho_per=1.0)
    for greek in ("price", "delta", "gamma", "vega", "rho"):
        assert abs(found[greek] - bumped(option, greek)) < 4 * found[f"{greek}_se"], (option, greek)


def test_each_likelihood_ratio_sample_is_the_payoff_times_its_score_written_out():
    """On three paths: the discounted payoff times the derivative of ln of S_T's lognormal density in S, vol and r
    (a central difference of the log density, x = ln S_T held), and the second derivative's score for gamma."""
    z = np.array([0.8, -0.4, 1.9])
    t, h = CASE["maturity"], 1e-5

    def log_density(x, spot=CASE["spot"], vol=CASE["vol"], rate=CASE["rate"]):
        mean = math.log(spot) + (rate - CASE["dividend_yield"] - vol * vol / 2) * t
        return -0.5 * ((x - mean) / (vol * math.sqrt(t))) ** 2 - math.log(vol * math.sqrt(t))

    for path in range(3):
        x = math.log(CASE["spot"]) + (CASE["rate"] - CASE["dividend_yield"] - 0.045) * t + 0.3 * math.sqrt(t) * z[path]
        one = MC.likelihood_ratio_greeks("digital_call", **CASE, normals=np.r_[z[path], z[path]], vega_per=1.0,
                                         rho_per=1.0)
        paid = math.exp(-CASE["rate"] * t) * (math.exp(x) > CASE["strike"])
        for greek, name in (("delta", "spot"), ("vega", "vol"), ("rho", "rate")):
            score = (log_density(x, **{name: CASE[name] + h}) - log_density(x, **{name: CASE[name] - h})) / (2 * h)
            expected = paid * (score - (t if greek == "rho" else 0.0))
            assert one[greek] == pytest.approx(expected, rel=1e-7, abs=1e-12), (path, greek)
        first = (log_density(x, spot=100 + h) - log_density(x, spot=100 - h)) / (2 * h)
        wide = 1e-2
        second = (log_density(x, spot=100 + wide) - 2 * log_density(x) + log_density(x, spot=100 - wide)) / wide ** 2
        assert one["gamma"] == pytest.approx(paid * (first ** 2 + second), rel=1e-6, abs=1e-12)


def test_likelihood_ratio_refuses_by_name():
    with pytest.raises(ValueError, match="option"):
        MC.likelihood_ratio_greeks("barrier", **CASE, normals=Z[:10], vega_per=1.0, rho_per=1.0)
    with pytest.raises(ValueError, match="rho_per"):
        MC.likelihood_ratio_greeks("call", **CASE, normals=Z[:10], vega_per=1.0, rho_per=0.1)
