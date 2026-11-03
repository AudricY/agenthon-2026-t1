"""qf.rates.hull_white_calibrate (runs/lanes/lib7-prompt.md, rank 23 of docs/research/library-syllabus-1011.md): Black
vols implied from Hull-White prices at a known (a, sigma) return it, to caplets and to swaptions, in price and in vol;
each market price is Black-76's on the curve written out. Every input is this file's own; no unit's values appear
(rule 8)."""
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf
import qf.bs
import qf.rates as r

pytestmark = pytest.mark.fast

NODES, DISCOUNTS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0], [0.985, 0.968, 0.932, 0.895, 0.818, 0.742]
A, SIGMA = 0.08, 0.011
BOUNDS = {"a": (1e-3, 1.0), "sigma": (1e-4, 0.1)}


def curve(t):
    return float(qf.curve_discount(t, NODES, DISCOUNTS, "log_linear_discount"))


def caplets():
    """Caplets and a floorlet whose Black vols are those of Hull-White's prices at (A, SIGMA)."""
    out = []
    for option, t1, strike in [("caplet", 1.0, 0.035), ("caplet", 2.0, 0.04), ("floorlet", 3.0, 0.04),
                               ("caplet", 5.0, 0.045)]:
        t2, tau = t1 + 0.5, 0.5
        price = r.hull_white_caplet(option, strike, t1, t2, tau, A, SIGMA, curve(t1), curve(t2))
        forward = (curve(t1) / curve(t2) - 1) / tau
        side = "call" if option == "caplet" else "put"
        vol = brentq(lambda v: tau * qf.bs.black76_price(side, forward, strike, t1, v, curve(t2)) - price, 1e-6, 3.0,
                     xtol=1e-15)
        out.append({"kind": "caplet", "option": option, "strike": strike, "fixing": t1, "payment": t2, "accrual": tau,
                    "vol": vol})
    return out


def swaptions():
    out = []
    for option, t0, length, strike, step in [("payer", 1.0, 3, 0.04, 1.0), ("receiver", 2.0, 6, 0.038, 0.5),
                                             ("payer", 3.0, 4, 0.042, 1.0)]:
        times = t0 + step * np.arange(1, length + 1, dtype=float)
        tau = np.full(length, step)
        price = r.jamshidian_swaption(option, strike, t0, times, tau, A, SIGMA, curve(t0), [curve(t) for t in times],
                                      0.03)["price"]
        annuity = sum(step * curve(t) for t in times)
        rate = (curve(t0) - curve(times[-1])) / annuity
        side = "call" if option == "payer" else "put"
        vol = brentq(lambda v: qf.bs.black76_price(side, rate, strike, t0, v, annuity) - price, 1e-6, 3.0, xtol=1e-15)
        out.append({"kind": "swaption", "option": option, "strike": strike, "expiry": t0, "payment_times": times,
                    "accruals": tau, "vol": vol})
    return out


def fit(instruments, loss="price", weights="equal", start=None):
    return r.hull_white_calibrate(instruments, NODES, DISCOUNTS, "log_linear_discount", loss, weights,
                                  start or {"a": 0.3, "sigma": 0.005}, BOUNDS)


@pytest.mark.parametrize("loss", ["price", "vol"])
@pytest.mark.parametrize("make", [caplets, swaptions])
def test_black_vols_made_from_a_known_hull_white_return_it(make, loss):
    instruments = make()
    found = fit(instruments, loss)
    assert found["converged"] and found["rmse"] < 1e-12
    assert (found["a"], found["sigma"]) == pytest.approx((A, SIGMA), rel=1e-7)
    assert found["model_prices"] == pytest.approx(found["market_prices"], rel=1e-10)


def test_the_market_prices_are_black_76_on_the_curve():
    both = caplets()[:1] + swaptions()[:1]
    found = fit(both)
    cap, swo = both
    forward = (curve(1.0) / curve(1.5) - 1) / 0.5
    assert found["market_prices"][0] == pytest.approx(
        0.5 * qf.bs.black76_price("call", forward, 0.035, 1.0, cap["vol"], curve(1.5)), rel=1e-14)
    annuity = sum(curve(t) for t in (2.0, 3.0, 4.0))
    assert found["market_prices"][1] == pytest.approx(qf.bs.black76_price(
        "call", (curve(1.0) - curve(4.0)) / annuity, 0.04, 1.0, swo["vol"], annuity), rel=1e-14)


def test_the_vol_loss_measures_black_vols_and_finds_its_own_optimum():
    """With the quotes moved off Hull-White, the 'vol' fit's rmse is that of the model prices' Black vols against the
    quotes, solved here, and differs from the 'price' fit."""
    instruments = swaptions()
    for entry, shift in zip(instruments, (0.01, -0.012, 0.008)):
        entry["vol"] += shift
    by_vol, by_price = fit(instruments, "vol"), fit(instruments, "price")
    vols = []
    for entry, price in zip(instruments, by_vol["model_prices"]):
        times, tau = entry["payment_times"], entry["accruals"]
        annuity = float(np.sum(tau * [curve(t) for t in times]))
        rate = (curve(entry["expiry"]) - curve(times[-1])) / annuity
        side = "call" if entry["option"] == "payer" else "put"
        vols.append(brentq(lambda v: qf.bs.black76_price(side, rate, entry["strike"], entry["expiry"], v, annuity)
                           - price, 1e-6, 3.0, xtol=1e-15))
    quoted = np.array([entry["vol"] for entry in instruments])
    assert by_vol["rmse"] == pytest.approx(math.sqrt(np.mean((np.array(vols) - quoted) ** 2)), rel=1e-8)
    assert abs(by_vol["a"] - by_price["a"]) > 1e-4


def test_a_weighted_instrument_is_fitted_closer():
    instruments = caplets()
    instruments[2]["vol"] += 0.02
    even = fit(instruments, start={"a": A, "sigma": SIGMA})
    heavy = fit(instruments, weights=[1.0, 1.0, 100.0, 1.0], start={"a": A, "sigma": SIGMA})
    gap = lambda f: abs(f["model_prices"][2] - f["market_prices"][2])
    assert gap(heavy) < gap(even) / 5


@pytest.mark.parametrize("bad, match", [
    (dict(loss="normal_vol"), "loss"), (dict(instruments=[{"kind": "cap"}]), "kind"),
    (dict(instruments=[{**caplets()[0], "option": "payer"}]), "option"), (dict(weights=[1.0]), "weights"),
    (dict(initial={"a": 0.05}), "initial"), (dict(bounds={"a": (0.0, 1.0), "sigma": (1e-4, 0.1)}), "bounds")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(instruments=caplets(), curve_times=NODES, curve_discounts=DISCOUNTS,
                interpolation="log_linear_discount", loss="price", weights="equal", initial={"a": 0.1, "sigma": 0.01},
                bounds=BOUNDS)
    with pytest.raises(ValueError, match=match):
        r.hull_white_calibrate(**{**args, **bad})
