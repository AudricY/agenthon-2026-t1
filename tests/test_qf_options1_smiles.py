"""Lane lib-options1's smiles, surfaces and jump models (runs/lanes/lib-options1-prompt.md; docs/research/
library-syllabus-1011.md rows 30, 54, 60, 77, 97, 111): SABR and its fit, the static no-arbitrage checks, the
Breeden-Litzenberger density, total-variance interpolation, the jump characteristic functions and vanna-volga.
References: QuantLib 1.43's SABR formulas and variance-gamma and Bates engines (frozen in `QL`; `__main__` recomputes
them where QuantLib is installed); Gatheral and Jacquier's (2014) arbitrageable SVI slice (Vogt's); Castagna and
Mercurio's (2007) EUR/USD pillars; and the identities (a known smile's parameters return, the lognormal density, the
pillars reprice, no jumps give Black-Scholes and Heston, Merton's series). Every input is a textbook's, a paper's or
this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import stats

import qf

pytestmark = pytest.mark.fast

QL = {   # QuantLib 1.43, Actual365Fixed, days from 2024-01-15 (`quantlib_values`)
    "sabr": [   # shiftedSabrVolatility(K, F, T, alpha, beta, nu, rho, shift, ShiftedLognormal or Normal)
        ((0.02, 0.03, 1.0, 0.035, 0.5, -0.3, 0.4, 0.0, 'lognormal'), 0.261838549165959),
        ((0.02, 0.03, 1.0, 0.035, 0.5, -0.3, 0.4, 0.0, 'normal'), 0.006444392761046336),
        ((0.045, 0.03, 1.0, 0.035, 0.5, -0.3, 0.4, 0.0, 'lognormal'), 0.1830561852748376),
        ((0.045, 0.03, 1.0, 0.035, 0.5, -0.3, 0.4, 0.0, 'normal'), 0.006762759055488562),
        ((0.03, 0.03, 1.0, 0.035, 0.5, -0.3, 0.4, 0.0, 'lognormal'), 0.20387661581963498),
        ((0.03, 0.03, 1.0, 0.035, 0.5, -0.3, 0.4, 0.0, 'normal'), 0.00610598435259259),
        ((0.08, 0.05, 2.0, 0.2, 0.7, 0.2, 0.6, 0.0, 'lognormal'), 0.5465813302539805),
        ((0.08, 0.05, 2.0, 0.2, 0.7, 0.2, 0.6, 0.0, 'normal'), 0.03432213129912882),
        ((-0.004, 0.002, 5.0, 0.01, 0.3, -0.2, 0.3, 0.02, 'lognormal'), 0.18414190450730455),
        ((-0.004, 0.002, 5.0, 0.01, 0.3, -0.2, 0.3, 0.02, 'normal'), 0.003451152776302614),
        ((0.06, 0.04, 1.5, 0.008, 0.0, 0.1, 0.25, 0.0, 'lognormal'), 0.17759699185057662),
        ((0.06, 0.04, 1.5, 0.008, 0.0, 0.1, 0.25, 0.0, 'normal'), 0.008745696548873567),
    ],
    "variance_gamma": [   # VarianceGammaEngine (Madan, Carr and Chang's closed form): S, K, days, r, q, sigma, nu, theta
        (('call', 100.0, 90.0, 365, 0.05, 0.01, 0.2, 0.2, -0.15), 16.266329756937253),
        (('call', 100.0, 100.0, 365, 0.05, 0.01, 0.2, 0.2, -0.15), 10.006973644066273),
        (('put', 100.0, 115.0, 365, 0.05, 0.01, 0.2, 0.2, -0.15), 14.382702569341552),
    ],
    "bates": [   # BatesEngine, order 192: S, K, days, r, q, v0, kappa, theta, sigma, rho, lambda, jump mean, jump vol
        (('call', 100.0, 90.0, 365, 0.05, 0.01, 0.04, 1.5, 0.04, 0.5, -0.7, 0.8, -0.1, 0.15), 17.795546169022906),
        (('call', 100.0, 100.0, 365, 0.05, 0.01, 0.04, 1.5, 0.04, 0.5, -0.7, 0.8, -0.1, 0.15), 11.480580708719728),
        (('put', 100.0, 115.0, 365, 0.05, 0.01, 0.04, 1.5, 0.04, 0.5, -0.7, 0.8, -0.1, 0.15), 15.035269596872041),
    ],
}


# ------------------------------------------------------------------------------------------------ SABR

@pytest.mark.parametrize("args, expected", QL["sabr"])
def test_sabr_equals_quantlibs_hagan_formulas(args, expected):
    k, f, t, alpha, beta, rho, nu, shift, kind = args
    assert qf.sabr_vol(k, f, t, alpha, beta, rho, nu, shift, kind) == pytest.approx(expected, rel=1e-13)


def test_sabrs_backbone_its_at_the_money_series_and_its_flat_cases():
    f, t = 0.03, 2.0
    k = np.array([0.02, 0.03 * (1 + 1e-9), 0.03, 0.04])
    backbone = qf.sabr_vol(k, f, t, 0.03, 0.5, 0.0, 0.0, 0.0, "lognormal")
    fk = f * k
    expected = 0.03 / (fk ** 0.25 * (1 + np.log(f / k) ** 2 / 96 + np.log(f / k) ** 4 / 30720)) * (
        1 + 0.03 ** 2 / (96 * fk ** 0.5) * t)
    assert np.allclose(backbone, expected, rtol=1e-14)
    near = qf.sabr_vol(k[1:3], f, t, 0.03, 0.5, -0.4, 0.5, 0.0, "lognormal")
    assert near[0] == pytest.approx(near[1], rel=1e-8)
    assert np.allclose(qf.sabr_vol(k, f, t, 0.2, 1.0, 0.0, 0.0, 0.0, "lognormal"), 0.2, rtol=1e-15)
    assert np.allclose(qf.sabr_vol(k, f, t, 0.006, 0.0, 0.0, 0.0, 0.0, "normal"), 0.006, rtol=1e-15)


@pytest.mark.parametrize("kind", ["lognormal", "normal"])
def test_sabr_fit_returns_the_parameters_of_a_smile_made_from_them(kind):
    """West (2005)'s round trip: alpha from the ATM cubic with (rho, nu) by least squares, or all three free."""
    k, f, t = np.linspace(0.015, 0.06, 10), 0.03, 1.5
    vols = qf.sabr_vol(k, f, t, 0.04, 0.5, -0.35, 0.45, 0.0, kind)
    atm = qf.sabr_vol(f, f, t, 0.04, 0.5, -0.35, 0.45, 0.0, kind)
    pinned = qf.sabr_fit(k, vols, f, t, 0.5, 0.0, kind, "atm", atm, "vol", "equal", {"rho": 0.0, "nu": 0.2})
    free = qf.sabr_fit(k, vols, f, t, 0.5, 0.0, kind, "free", None, "price", "equal",
                       {"alpha": 0.03, "rho": 0.0, "nu": 0.2})
    for found in (pinned, free):
        assert found["converged"] and found["beta"] == 0.5
        assert (found["alpha"], found["rho"], found["nu"]) == pytest.approx((0.04, -0.35, 0.45), rel=1e-7)
        assert found["rmse"] < 1e-10 and np.allclose(found["model"], vols, rtol=1e-8)


def test_the_atm_cubic_matches_the_atm_vol_exactly():
    k, f, t = np.array([0.02, 0.025, 0.03, 0.035, 0.04]), 0.03, 1.0
    found = qf.sabr_fit(k, [0.31, 0.27, 0.245, 0.235, 0.233], f, t, 0.5, 0.0, "lognormal", "atm", 0.245, "vol",
                        "equal", {"rho": 0.0, "nu": 0.5})
    assert qf.sabr_vol(f, f, t, found["alpha"], 0.5, found["rho"], found["nu"], 0.0, "lognormal") == pytest.approx(
        0.245, rel=1e-12)


# ------------------------------------------------------------------------------------------------ static arbitrage

S, R, Q, VOL = 100.0, 0.03, 0.01, 0.25


def black_scholes_surface(strikes, maturities, vol=VOL):
    return np.array([qf.bs_price("call", S, strikes, t, R, Q, vol) for t in maturities])


def test_a_black_scholes_surface_passes_every_check():
    k, t = np.linspace(60.0, 160.0, 41), np.array([0.25, 0.5, 1.0, 2.0])
    found = qf.arbitrage_checks("calls", k, t, black_scholes_surface(k, t), S * np.exp((R - Q) * t), np.exp(-R * t),
                                1e-10)
    assert found["arbitrage_free"] and np.all(found["density"][:, 1:-1] > 0)
    w = (VOL * VOL * t)[:, None] * np.ones(21)
    assert qf.arbitrage_checks("total_variance", np.linspace(-1, 1, 21), t, w, [1.0] * 4, None, 0.0)["arbitrage_free"]


def test_gatheral_and_jacquiers_arbitrageable_svi_slice_is_flagged():
    """Gatheral and Jacquier (2014), Vogt's raw SVI slice a -0.0410, b 0.1331, rho 0.3060, m 0.3586, sigma 0.4153
    (T 1): its density factor g(k) goes negative to the right of the money; an ordinary slice passes."""
    k = np.linspace(-1.5, 1.5, 301)
    vogt = qf.svi_total_variance(k, a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    found = qf.arbitrage_checks("total_variance", k, [1.0], vogt, [1.0], None, 0.0)
    flagged = k[found["butterfly"][0]]
    assert not found["arbitrage_free"] and flagged.size and 0.5 < flagged.min() < flagged.max() < 1.5
    plain = qf.svi_total_variance(k, a=0.02, b=0.1, rho=-0.4, m=0.0, sigma=0.2)
    assert qf.arbitrage_checks("total_variance", k, [1.0], plain, [1.0], None, 0.0)["arbitrage_free"]


def test_a_calendar_break_a_bump_and_a_price_above_its_bound_are_flagged_where_they_are():
    k, t = np.linspace(60.0, 160.0, 41), np.array([0.25, 0.5, 1.0, 2.0])
    forwards, discount = S * np.exp((R - Q) * t), np.exp(-R * t)
    calls = black_scholes_surface(k, t)
    lower = calls.copy()
    lower[2] = qf.bs_price("call", S, k, 1.0, R, Q, 0.15)       # a lower vol at the later expiry
    found = qf.arbitrage_checks("calls", k, t, lower, forwards, discount, 1e-10)
    assert not found["arbitrage_free"] and found["calendar"][1].any() and not found["calendar"][[0, 2]].any()
    bumped = calls.copy()
    bumped[1, 20] += 0.5
    found = qf.arbitrage_checks("calls", k, t, bumped, forwards, discount, 1e-10)
    assert np.argwhere(found["butterfly"]).tolist() == [[1, 20]]
    high = calls.copy()
    high[0, 0] = forwards[0] * discount[0] + 1.0
    found = qf.arbitrage_checks("calls", k, t, high, forwards, discount, 1e-10)
    assert found["bounds"][0, 0] and found["monotonic"][0, 1]
    forward_form = qf.arbitrage_checks("forward_calls", k, t, calls / discount[:, None], forwards, None, 1e-10)
    assert forward_form["arbitrage_free"]
    steep = calls.copy()
    steep[0, 0] += 1.0                  # deep in the money, a slope of about -1.4 to the next strike, inside the bounds
    found = qf.arbitrage_checks("calls", k, t, steep, forwards, discount, 1e-10)
    assert found["monotonic"][0, 1] and not found["bounds"].any()



def test_the_calendar_is_read_at_a_fixed_moneyness_not_a_fixed_strike():
    """A falling forward (a yield above the rate) with the same call over the forward at each log-moneyness: no
    calendar arbitrage, though the later prices at a fixed strike are lower."""
    k_over_f = np.linspace(0.6, 1.6, 21)
    c = np.array([qf.bs_price("call", 1.0, k_over_f, 1.0, 0.0, 0.0, 0.2)] * 2)       # forward prices over F, T = 1
    forwards = np.array([100.0, 70.0])
    strikes = forwards[:, None] * k_over_f
    found = qf.arbitrage_checks("forward_calls", strikes, [1.0, 1.5], c * forwards[:, None], forwards, None, 1e-12)
    assert found["arbitrage_free"]


def test_deterministic_forwards_have_no_calendar_arbitrage_when_the_forward_changes():
    """A deterministic normalized underlying is always one: C/F = max(1 - K/F, 0) at every expiry.
    Interpolating these calls in log-moneyness used to put later prices below their exact intrinsic value."""
    found = qf.arbitrage_checks("forward_calls", [50.0, 60.0, 70.0], [1.0, 2.0],
                                [[50.0, 40.0, 30.0], [60.0, 50.0, 40.0]], [100.0, 110.0], None, 1e-12)
    assert found["arbitrage_free"] and not found["calendar"].any()


@pytest.mark.parametrize("inputs", ["calls", "forward_calls", "total_variance"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_arbitrage_checks_refuse_missing_or_infinite_quotes(inputs, bad):
    """An incomplete smile cannot establish absence of arbitrage; all-NaN comparisons used to pass every flag."""
    grid = [-0.1, 0.0, 0.1] if inputs == "total_variance" else [90.0, 100.0, 110.0]
    discounts = [0.97] if inputs == "calls" else None
    with pytest.raises(ValueError, match="values"):
        qf.arbitrage_checks(inputs, grid, [1.0], [[bad, bad, bad]], [100.0], discounts, 0.0)


@pytest.mark.parametrize("inputs", ["calls", "forward_calls", "total_variance"])
def test_arbitrage_checks_refuse_a_missing_grid_point(inputs):
    grid = [-0.1, np.nan, 0.1] if inputs == "total_variance" else [90.0, np.nan, 110.0]
    values = [[0.04, 0.04, 0.04]] if inputs == "total_variance" else [[12.0, 6.0, 2.0]]
    discounts = [0.97] if inputs == "calls" else None
    with pytest.raises(ValueError, match="grid"):
        qf.arbitrage_checks(inputs, grid, [1.0], values, [100.0], discounts, 0.0)


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -0.01])
def test_arbitrage_checks_refuse_a_tolerance_that_disables_the_checks(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        qf.arbitrage_checks("forward_calls", [90.0, 100.0, 110.0], [1.0], [[200.0, 201.0, 202.0]],
                            [100.0], None, tolerance)


def test_arbitrage_checks_require_each_expirys_finite_inputs_and_discount_factor():
    given = dict(inputs="calls", grid=[80.0, 100.0, 130.0], maturities=[0.4, 1.2],
                 values=[[14.0, 7.0, 3.0], [16.0, 10.0, 6.0]], forwards=[105.0, 108.0],
                 discount_factors=[0.97, 0.94], tolerance=0.0)
    for key, changed in [("maturities", [0.4, np.inf]), ("forwards", [105.0, np.inf]),
                         ("discount_factors", [0.97, np.inf]), ("discount_factors", [0.97])]:
        with pytest.raises(ValueError, match=key):
            qf.arbitrage_checks(**{**given, key: changed})

# ------------------------------------------------------------------------------------------------ density, time

def test_black_scholes_prices_give_the_lognormal_density_which_integrates_to_one():
    """Breeden and Litzenberger (1978): q(K) = e^{rT} C''(K); the CDF 1 + e^{rT} C'(K); -C'(K) the cash digital."""
    t, k = 0.75, np.linspace(20.0, 400.0, 3801)
    found = qf.risk_neutral_density(k, qf.bs_price("call", S, k, t, R, Q, VOL), t, R, "prices", None, None)
    law = stats.lognorm(VOL * math.sqrt(t), scale=S * math.exp((R - Q - VOL * VOL / 2) * t))
    inner = slice(1, -1)
    assert np.max(np.abs(found["density"][inner] - law.pdf(k[inner]))) < 1e-7
    assert np.trapezoid(found["density"][inner], k[inner]) == pytest.approx(1.0, abs=1e-8)
    assert np.max(np.abs(found["cdf"][inner] - law.cdf(k[inner]))) < 1e-5
    digital = qf.digital_price("cash", "call", S, k[inner], t, R, Q, VOL)
    assert np.max(np.abs(found["digital"][inner] - digital)) < 1e-5
    assert np.isnan(found["density"][[0, -1]]).all()
    coarse = k[::10]
    from_vols = qf.risk_neutral_density(coarse, np.full(coarse.size, VOL), t, R, "vols", S, Q)
    assert np.max(np.abs(from_vols["density"][1:-1] - law.pdf(coarse[1:-1]))) < 5e-6
    assert np.max(np.abs(from_vols["cdf"][1:-1] - law.cdf(coarse[1:-1]))) < 5e-4


def test_total_variance_interpolation_hits_the_expiries_and_keeps_forward_variance():
    t, vols = np.array([0.5, 1.0, 2.0]), np.array([0.2, 0.22, 0.21])
    found = qf.total_variance_interp(t, t, vols, "total_variance", "none")
    assert np.allclose(found["vol"], vols, rtol=1e-15)
    mid = qf.total_variance_interp(0.75, t, vols, "total_variance", "none")
    w = 0.5 * (0.2 ** 2 * 0.5 + 0.22 ** 2 * 1.0)
    assert mid["total_variance"] == pytest.approx(w, rel=1e-14) and mid["vol"] == pytest.approx(math.sqrt(w / 0.75))
    assert mid["forward_vol"] == pytest.approx(math.sqrt((0.22 ** 2 - 0.2 ** 2 * 0.5) / 0.5), rel=1e-14)
    falling = qf.total_variance_interp(1.5, [1.0, 2.0], [0.3, 0.2], "total_variance", "none")
    assert np.isnan(falling["forward_vol"])               # w falls: calendar arbitrage
    beyond = qf.total_variance_interp(3.0, t, vols, "total_variance", "flat_forward")
    forward = (0.21 ** 2 * 2 - 0.22 ** 2) / 1.0
    assert beyond["total_variance"] == pytest.approx(0.21 ** 2 * 2 + forward, rel=1e-14)
    assert qf.total_variance_interp(3.0, t, vols, "total_variance", "flat_vol")["vol"] == pytest.approx(0.21)
    assert qf.total_variance_interp(0.75, t, vols, "vol", "none")["vol"] == pytest.approx(0.21, rel=1e-14)
    assert qf.total_variance_interp(0.75, t, vols, "variance", "none")["vol"] == pytest.approx(
        math.sqrt(0.5 * (0.04 + 0.0484)), rel=1e-14)
    grid = qf.total_variance_interp([0.75, 1.5], t, np.column_stack([vols, vols + 0.05]), "total_variance", "none")
    assert grid["vol"].shape == (2, 2)
    with pytest.raises(ValueError):
        qf.total_variance_interp(3.0, t, vols, "total_variance", "none")


# ------------------------------------------------------------------------------------------------ jump models

def priced(charfn, option, k, t, r, q, upper=200.0, nodes=512):
    return qf.charfn_price(charfn, option, 100.0, k, t, r, q, "gauss_legendre", upper, nodes)["price"]


def test_mertons_characteristic_function_prices_mertons_series():
    """The class keyed by qf.merton_jump_price in docs/research/selection-equivalents-1011.toml."""
    p = {"vol": 0.2, "intensity": 1.0, "jump_mean": -0.1, "jump_vol": 0.15}
    k = np.array([80.0, 100.0, 125.0])
    found = priced(lambda u: qf.jump_charfn(u, "merton", 100.0, 0.75, 0.04, 0.01, p), "call", k, 0.75, 0.04, 0.01)
    series = qf.merton_jump_price("call", 100.0, k, 0.75, 0.04, 0.01, 0.2, 1.0, -0.1, 0.15, 60)
    assert np.allclose(found, series, rtol=1e-11)


def test_no_jumps_give_black_scholes_and_heston():
    u = np.linspace(-5.0, 5.0, 11) + 0.3j
    merton = qf.jump_charfn(u, "merton", 100.0, 1.0, 0.03, 0.01, {"vol": 0.2, "intensity": 0.0, "jump_mean": -0.1,
                                                                    "jump_vol": 0.15})
    lognormal = np.exp(1j * u * (math.log(100.0) + (0.02 - 0.02) * 1.0) - 0.02 * u * u)
    assert np.allclose(merton, lognormal, rtol=1e-14)
    heston = {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma": 0.5, "rho": -0.7}
    bates = qf.jump_charfn(u, "bates", 100.0, 1.0, 0.03, 0.01, {**heston, "intensity": 0.0, "jump_mean": -0.1,
                                                                 "jump_vol": 0.15})
    assert np.allclose(bates, qf.heston_charfn(u, 100.0, 1.0, 0.03, 0.01, *heston.values(), False), rtol=1e-14)


@pytest.mark.parametrize("model", ["merton", "bates", "variance_gamma"])
def test_each_jump_model_is_risk_neutral(model):
    parameters = {"merton": {"vol": 0.2, "intensity": 2.0, "jump_mean": 0.05, "jump_vol": 0.2},
                  "bates": {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma": 0.5, "rho": -0.7, "intensity": 2.0,
                            "jump_mean": 0.05, "jump_vol": 0.2},
                  "variance_gamma": {"sigma": 0.25, "nu": 0.3, "theta": -0.2}}[model]
    forward = qf.jump_charfn(-1j, model, 100.0, 1.5, 0.04, 0.01, parameters)
    assert forward == pytest.approx(100.0 * math.exp(0.03 * 1.5), rel=1e-13)


@pytest.mark.parametrize("args, expected", QL["variance_gamma"])
def test_variance_gamma_equals_quantlibs_closed_form(args, expected):
    option, s, k, days, r, q, sigma, nu, theta = args
    t = days / 365
    p = {"sigma": sigma, "nu": nu, "theta": theta}
    assert priced(lambda u: qf.jump_charfn(u, "variance_gamma", s, t, r, q, p), option, k, t, r, q) == pytest.approx(
        expected, rel=1e-9)


@pytest.mark.parametrize("args, expected", QL["bates"])
def test_bates_equals_quantlibs_engine(args, expected):
    option, s, k, days, r, q, v0, kappa, theta, sigma, rho, lam, mean, spread = args
    t = days / 365
    p = {"v0": v0, "kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "intensity": lam, "jump_mean": mean,
         "jump_vol": spread}
    assert priced(lambda u: qf.jump_charfn(u, "bates", s, t, r, q, p), option, k, t, r, q) == pytest.approx(
        expected, rel=1e-11)


# ------------------------------------------------------------------------------------------------ vanna-volga

T_CM = 94 / 365          # Castagna and Mercurio (2007), Tables 1 and 2: EUR/USD, 94 days
CM = dict(spot=1.205, maturity=T_CM, domestic_rate=-math.log(0.9902752) / T_CM,
          foreign_rate=-math.log(0.9945049) / T_CM, pillar_strikes=[1.1720, 1.2115, 1.2504],
          pillar_vols=[0.0979, 0.09375, 0.0929])


@pytest.mark.parametrize("variant", ["price", "first_order", "second_order"])
def test_vanna_volga_reprices_its_pillars_and_a_flat_smile_is_black_scholes(variant):
    for option in ("call", "put"):
        found = qf.vanna_volga(option=option, strike=CM["pillar_strikes"], variant=variant, **CM)
        assert np.allclose(found["vol"], CM["pillar_vols"], rtol=1e-9)
    flat = qf.vanna_volga(option="call", strike=[1.10, 1.20, 1.30], variant=variant,
                          **{**CM, "pillar_vols": [0.09, 0.09, 0.09]})
    assert np.allclose(flat["price"], qf.bs_price("call", 1.205, np.array([1.10, 1.20, 1.30]), T_CM,
                                                  CM["domestic_rate"], CM["foreign_rate"], 0.09), rtol=1e-12)



def test_the_vanna_volga_weights_hedge_vega_vanna_and_volga_at_the_atm_vol():
    """Castagna and Mercurio's weights solve the three-by-three system matching the option's vega, vanna and volga
    with the pillars', each at the ATM vol (their eqs. (4)-(6)): the price from the solved system is the function's."""
    s, t, rd, rf = CM["spot"], T_CM, CM["domestic_rate"], CM["foreign_rate"]
    pk, pv, atm = np.array(CM["pillar_strikes"]), np.array(CM["pillar_vols"]), CM["pillar_vols"][1]
    forward = s * math.exp((rd - rf) * t)

    def greeks(k):
        d1 = (np.log(forward / k) + atm * atm * t / 2) / (atm * math.sqrt(t))
        d2 = d1 - atm * math.sqrt(t)
        vega = s * math.exp(-rf * t) * stats.norm.pdf(d1) * math.sqrt(t)
        return np.array([vega, -math.exp(-rf * t) * stats.norm.pdf(d1) * d2 / atm, vega * d1 * d2 / atm])

    for k in (1.15, 1.19, 1.23, 1.27):
        weights = np.linalg.solve(np.column_stack([greeks(x) for x in pk]), greeks(k))
        costs = [qf.bs_price("call", s, x, t, rd, rf, v) - qf.bs_price("call", s, x, t, rd, rf, atm) for x, v in zip(pk, pv)]
        expected = qf.bs_price("call", s, k, t, rd, rf, atm) + weights @ np.array(costs)
        assert qf.vanna_volga(option="call", strike=k, variant="price", **CM)["price"] == pytest.approx(expected, rel=1e-10)

def test_the_second_order_vol_tracks_the_vanna_volga_price_and_the_first_order_drifts_in_the_wings():
    """Castagna and Mercurio: the second-order approximation is very close to the price's implied vol across the
    strikes; the first-order one is accurate between the pillars and drifts outside them."""
    k = np.array([1.10, 1.15, 1.20, 1.25, 1.30])
    exact = qf.vanna_volga(option="call", strike=k, variant="price", **CM)["vol"]
    second = qf.vanna_volga(option="call", strike=k, variant="second_order", **CM)["vol"]
    first = qf.vanna_volga(option="call", strike=k, variant="first_order", **CM)["vol"]
    assert np.max(np.abs(second - exact)) < 4e-4 and np.max(np.abs(first - exact)) > 3 * np.max(np.abs(second - exact))
    assert np.all(exact[[0, -1]] > exact[2])           # the smile


def test_the_smile_conventions_are_refused_when_unnamed_or_unfit():
    with pytest.raises(ValueError):
        qf.sabr_vol(0.03, 0.03, 1.0, 0.03, 0.5, -0.3, 0.4, 0.0, "black")
    with pytest.raises(ValueError):
        qf.sabr_fit([0.02, 0.03, 0.04], [0.3, 0.25, 0.24], 0.03, 1.0, 0.5, 0.0, "lognormal", "atm", None, "vol",
                    "equal", {"rho": 0.0, "nu": 0.5})
    with pytest.raises(ValueError):
        qf.jump_charfn(1.0, "kou", 100.0, 1.0, 0.03, 0.0, {})
    with pytest.raises(ValueError):
        qf.jump_charfn(1.0, "merton", 100.0, 1.0, 0.03, 0.0, {"vol": 0.2})
    with pytest.raises(ValueError):
        qf.arbitrage_checks("calls", [90.0, 100.0, 110.0], [1.0], [[12.0, 6.0, 2.0]], [100.0], None, 0.0)
    with pytest.raises(ValueError):
        qf.vanna_volga(option="call", strike=1.2, variant="third_order", **CM)


def quantlib_values() -> dict:
    """The frozen `QL` values recomputed by QuantLib (scratch: a Python with QuantLib 1.43)."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    count = ql.Actual365Fixed()
    kind = {"call": ql.Option.Call, "put": ql.Option.Put}

    def curve(r):
        return ql.YieldTermStructureHandle(ql.FlatForward(today, r, count))

    def npv(engine, option, k, days):
        made = ql.VanillaOption(ql.PlainVanillaPayoff(kind[option], k), ql.EuropeanExercise(today + days))
        made.setPricingEngine(engine)
        return made.NPV()

    types = {"lognormal": ql.ShiftedLognormal, "normal": ql.Normal}
    found = {"sabr": [((k, f, t, a, b, rho, nu, s, name), ql.shiftedSabrVolatility(k, f, t, a, b, nu, rho, s,
                                                                                    types[name]))
                      for (k, f, t, a, b, rho, nu, s, name), _ in QL["sabr"]]}
    found["variance_gamma"] = [(args, npv(ql.VarianceGammaEngine(ql.VarianceGammaProcess(
        ql.QuoteHandle(ql.SimpleQuote(args[1])), curve(args[5]), curve(args[4]), *args[6:])), args[0], args[2],
        args[3])) for args, _ in QL["variance_gamma"]]
    found["bates"] = [(args, npv(ql.BatesEngine(ql.BatesModel(ql.BatesProcess(
        curve(args[4]), curve(args[5]), ql.QuoteHandle(ql.SimpleQuote(args[1])), *args[6:])), 192), args[0], args[2],
        args[3])) for args, _ in QL["bates"]]
    return found


if __name__ == "__main__":
    fresh = quantlib_values()
    print("QL frozen values match QuantLib:", all(
        np.allclose([v for _, v in fresh[key]], [v for _, v in QL[key]], rtol=1e-14, atol=0) for key in QL))
