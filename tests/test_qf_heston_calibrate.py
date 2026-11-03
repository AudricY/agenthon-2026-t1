"""qf.fourier.heston_calibrate (runs/lanes/lib7-prompt.md, rank 19 of docs/research/library-syllabus-1011.md): prices
and implied vols made by qf.fourier.charfn_price from known parameters come back from a start away from them; the
Feller condition enforced on a smile that breaks it; weights move the fit. Every parameter set is this file's own;
no unit's values appear (rule 8)."""
import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf
import qf.fourier as F

pytestmark = pytest.mark.fast

S, R, Q = 100.0, 0.025, 0.01
K = np.tile([80.0, 90.0, 100.0, 110.0, 125.0], 2)
T = np.repeat([0.5, 1.5], 5)
KINDS = np.where(K < 100, "put", "call")
BOUNDS = dict(v0=(1e-4, 1.0), kappa=(1e-3, 10.0), theta=(1e-4, 1.0), sigma=(1e-3, 3.0), rho=(-0.999, 0.999))
START = dict(v0=0.03, kappa=1.0, theta=0.03, sigma=0.3, rho=-0.3)


def prices(p, kinds=KINDS):
    out = []
    for kind, k, t in zip(kinds, K, T):
        def charfn(u, t=t):
            return F.heston_charfn(u, S, t, R, Q, p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"], False)
        out.append(F.charfn_price(charfn, kind, S, k, t, R, Q, "gauss_legendre", 200.0, 128)["price"])
    return np.array(out)


def fit(market, loss="price", weights="equal", feller="free", start=START):
    return F.heston_calibrate(KINDS, K, T, market, S, R, Q, loss, weights, start, BOUNDS, feller, 200.0, 128)


TRUE = dict(v0=0.05, kappa=1.8, theta=0.045, sigma=0.35, rho=-0.65)     # 2 kappa theta = 0.162 >= sigma^2


def test_prices_made_from_known_parameters_return_them():
    found = fit(prices(TRUE))
    assert found["converged"] and found["rmse"] < 1e-10 and found["feller"]
    assert [found[name] for name in F.HESTON] == pytest.approx([TRUE[name] for name in F.HESTON], rel=1e-7)


def vols_of(market):
    return np.array([qf.implied_vol(kind, price, S, k, t, R, Q, "bisection")
                     for kind, price, k, t in zip(KINDS, market, K, T)])


def test_the_implied_vol_loss_compares_vols():
    """Started at the parameters that made them, the fit stays and its model is the market's implied vols."""
    vols = vols_of(prices(TRUE))
    found = fit(vols, loss="implied_vol", start=TRUE)
    assert found["rmse"] < 1e-9 and found["model"] == pytest.approx(vols, abs=1e-9)


@pytest.mark.slow
def test_implied_vols_made_from_known_parameters_return_them():
    vols = vols_of(prices(TRUE))
    found = fit(vols, loss="implied_vol", start=dict(v0=0.045, kappa=1.5, theta=0.05, sigma=0.3, rho=-0.5))
    assert found["rmse"] < 1e-9 and found["model"] == pytest.approx(vols, abs=1e-9)
    assert [found[name] for name in F.HESTON] == pytest.approx([TRUE[name] for name in F.HESTON], rel=1e-6)


def test_the_feller_condition_enforced_on_a_smile_that_breaks_it():
    """sigma 0.7 against 2 kappa theta = 0.162: 'free' returns the parameters, 'enforce' a fit on Feller's side."""
    breaking = {**TRUE, "sigma": 0.7}
    market = prices(breaking)
    free = fit(market)
    assert not free["feller"] and free["sigma"] == pytest.approx(0.7, rel=1e-6)
    held = fit(market, feller="enforce")
    assert held["feller"] and 2 * held["kappa"] * held["theta"] >= held["sigma"] ** 2 * (1 - 1e-9)
    assert held["rmse"] > free["rmse"]


@pytest.mark.slow
def test_a_quote_weighted_up_is_fitted_closer():
    noisy = prices(TRUE) + np.random.default_rng(4).normal(0.0, 0.05, K.size)
    even = fit(noisy, start=TRUE)
    weights = np.ones(K.size)
    weights[6] = 200.0
    heavy = fit(noisy, weights=weights, start=TRUE)
    assert abs(heavy["model"][6] - noisy[6]) < abs(even["model"][6] - noisy[6]) / 5


@pytest.mark.parametrize("bad, match", [
    (dict(loss="vega"), "loss"), (dict(feller="soft"), "feller"), (dict(weights=[1.0]), "weights"),
    (dict(initial={"v0": 0.04}), "initial"), (dict(options="straddle"), "options"),
    (dict(bounds={**BOUNDS, "rho": (-2.0, 0.9)}), "bound"), (dict(market=[1.0, 2.0]), "quote")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(options="call", strikes=K, maturities=T, market=np.ones(K.size), spot=S, rate=R, dividend_yield=Q,
                loss="price", weights="equal", initial=START, bounds=BOUNDS, feller="free", upper=200.0, nodes=128)
    with pytest.raises(ValueError, match=match):
        F.heston_calibrate(**{**args, **bad})
