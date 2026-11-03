"""qf.volatility.svi_local_vol (runs/lanes/lib7-prompt.md, rank 24 of docs/research/library-syllabus-1011.md): a flat
surface returns its vol; the slopes in k and T against finite differences of svi_total_variance; Gatheral's local vol
against qf.volatility.dupire_local_vol on call prices made from the same surface. Every SVI parameter set is this
file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_conventions import card_problems  # noqa: F401 - the shelf's card check is test_qf_volatility's
import qf
import qf.volatility as vol

pytestmark = pytest.mark.fast

S, R, Q = 100.0, 0.03, 0.01
SURFACE = dict(maturities=[0.5, 1.0, 1.5], a=[0.012, 0.025, 0.04], b=[0.12, 0.11, 0.1], rho=[-0.45, -0.4, -0.35],
               m=[0.02, 0.03, 0.04], sigma=[0.18, 0.22, 0.26])


def local(strike, maturity, rule="forward_difference", **changes):
    return qf.svi_local_vol(strike, spot=S, maturity=maturity, rate=R, dividend_yield=Q, **{**SURFACE, **changes},
                             t_derivative=rule)


def total(k, t):
    """w(k, t): the slices' svi_total_variance mixed linearly in t at fixed k."""
    times = SURFACE["maturities"]
    i = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
    theta = (t - times[i]) / (times[i + 1] - times[i])
    parts = [qf.svi_total_variance(k, **{key: SURFACE[key][j] for key in ("a", "b", "rho", "m", "sigma")})
             for j in (i, i + 1)]
    return (1 - theta) * parts[0] + theta * parts[1]


@pytest.mark.parametrize("rule", ["forward_difference", "central_difference"])
def test_a_flat_surface_returns_its_vol(rule):
    times = np.array(SURFACE["maturities"])
    found = local([60.0, 100.0, 150.0], 1.0, rule, a=0.2 ** 2 * times, b=[0.0] * 3)
    assert found["local_vol"] == pytest.approx([0.2] * 3, rel=1e-13)


def test_the_slopes_are_the_surfaces_finite_differences():
    strikes, t, h = np.array([75.0, 95.0, 104.0, 130.0]), 0.8, 1e-4
    found = local(strikes, t)
    k = np.log(strikes / (S * math.exp((R - Q) * t)))
    assert found["k"] == pytest.approx(k, rel=1e-14) and found["w"] == pytest.approx(total(k, t), rel=1e-14)
    assert found["w_k"] == pytest.approx((total(k + h, t) - total(k - h, t)) / (2 * h), rel=1e-7)
    assert found["w_kk"] == pytest.approx((total(k + h, t) - 2 * total(k, t) + total(k - h, t)) / h ** 2, rel=1e-5)
    assert found["w_t"] == pytest.approx((total(k, t + h) - total(k, t - h)) / (2 * h), rel=1e-9)


def test_at_a_slice_the_rules_take_its_neighbours():
    """At the middle slice of 0.5, 1.0, 1.8: forward the slope to the next; central the derivative at 1.0 of the
    parabola through the three slices' w (numpy.polyfit, unequal steps); at the last slice the slope before it."""
    uneven = dict(maturities=[0.5, 1.0, 1.8])
    strikes = np.array([90.0, 110.0])
    k = np.log(strikes / (S * math.exp((R - Q) * 1.0)))
    w = np.array([qf.svi_total_variance(k, **{key: SURFACE[key][j] for key in ("a", "b", "rho", "m", "sigma")})
                  for j in range(3)])
    assert local(strikes, 1.0, **uneven)["w_t"] == pytest.approx((w[2] - w[1]) / 0.8, rel=1e-12)
    central = [np.polyval(np.polyder(np.polyfit([0.5, 1.0, 1.8], w[:, i], 2)), 1.0) for i in range(2)]
    assert local(strikes, 1.0, "central_difference", **uneven)["w_t"] == pytest.approx(central, rel=1e-9)
    k = np.log(strikes / (S * math.exp((R - Q) * 1.5)))
    assert local(strikes, 1.5)["w_t"] == pytest.approx((total(k, 1.5 - 1e-12) - total(k, 1.0)) / 0.5, rel=1e-9)


def test_gatherals_local_vol_is_dupires_on_the_surfaces_call_prices():
    """Calls priced at sqrt(w / T) on a fine grid inside one slice interval; Dupire's finite differences there
    agree with the formula to 2e-4."""
    strikes = np.linspace(70.0, 140.0, 281)
    maturities = np.linspace(1.05, 1.45, 41)
    calls = np.empty((maturities.size, strikes.size))
    for row, t in enumerate(maturities):
        k = np.log(strikes / (S * math.exp((R - Q) * t)))
        calls[row] = qf.bs_price("call", S, strikes, t, R, Q, np.sqrt(total(k, t) / t))
    dupire = qf.dupire_local_vol(calls, strikes=strikes, maturities=maturities, rate=R, dividend_yield=Q)
    for row in (10, 20, 30):
        for col in (40, 120, 200, 260):
            expected = local(strikes[col], maturities[row])["local_vol"]
            assert dupire[row, col] == pytest.approx(expected, rel=2e-4), (row, col)


def test_calendar_arbitrage_gives_nan():
    found = local([100.0], 0.75, a=[0.04, 0.01, 0.04])
    assert np.isnan(found["local_vol"]) and np.isnan(found["local_variance"]) and found["w_t"] < 0


@pytest.mark.parametrize("bad, match", [
    (dict(t_derivative="analytic"), "t_derivative"), (dict(maturity=2.0), "maturity"), (dict(rho=[0.1, 1.2, 0.0]), "rho"),
    (dict(maturities=[1.0]), "maturities"), (dict(a=[0.01, 0.02]), "one value per maturity")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(spot=S, maturity=1.0, rate=R, dividend_yield=Q, **SURFACE, t_derivative="forward_difference")
    if "maturities" in bad:
        args.update(a=[0.01], b=[0.1], rho=[0.0], m=[0.0], sigma=[0.2])
    with pytest.raises(ValueError, match=match):
        qf.svi_local_vol(100.0, **{**args, **bad})
