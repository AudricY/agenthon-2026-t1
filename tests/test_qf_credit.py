"""agent/qf/credit.py (runs/lanes/lib6-prompt.md, rank 9 of docs/research/library-syllabus-1011.md): Merton's model
calibrated to equity, against Hull's worked example and by round trips from asset values and vols chosen here, by the
simultaneous equations and by the KMV iteration (Crosbie and Bohn 2003). No unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf
import qf.credit as cr

pytestmark = pytest.mark.fast


def equity_of(v, sigma, debt, t, r):
    """E = V N(d1) - D e^{-rT} N(d2) (Merton 1974), and sigma_E = N(d1) sigma V / E, written out here."""
    d1 = (math.log(v / debt) + (r + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    e = v * norm.cdf(d1) - debt * math.exp(-r * t) * norm.cdf(d1 - sigma * math.sqrt(t))
    return e, norm.cdf(d1) * sigma * v / e


def test_the_credit_card_keeps_the_format_and_its_example_runs():
    check_cards(cr)


def test_hulls_example_from_equity_to_the_default_probability():
    """Hull, "Credit risk" (Merton's model): equity 3, its vol 80%, debt 10 due in a year, r = 5%: V0 = 12.40,
    sigma_V = 21.23%, and the risk-neutral default probability N(-d2) = 12.7%."""
    found = cr.merton_calibrate(3.0, 0.8, 10.0, 0.0, "total", 1.0, 0.05, 0.05, "simultaneous", None, None)
    assert found["asset_value"] == pytest.approx(12.40, abs=0.005)
    assert found["asset_vol"] == pytest.approx(0.2123, abs=5e-5)
    assert found["default_probability"] == pytest.approx(0.127, abs=5e-4)
    assert found["default_probability"] == pytest.approx(norm.cdf(-found["distance_to_default"]), abs=1e-15)


@pytest.mark.parametrize("v, sigma, debt, t, r", [(120.0, 0.25, 80.0, 1.0, 0.03), (55.0, 0.4, 50.0, 2.0, 0.01),
                                                   (300.0, 0.15, 90.0, 0.5, 0.06)])
def test_the_simultaneous_equations_return_the_asset_value_and_vol_that_made_the_equity(v, sigma, debt, t, r):
    """A round trip: E and sigma_E from a known V and sigma_V give them back; sigma_V < sigma_E with debt; the
    distance to default with the drift mu as written."""
    e, sigma_e = equity_of(v, sigma, debt, t, r)
    found = cr.merton_calibrate(e, sigma_e, debt, 0.0, "total", t, r, 0.07, "simultaneous", None, None)
    assert found["asset_value"] == pytest.approx(v, rel=1e-10) and found["asset_vol"] == pytest.approx(sigma, rel=1e-9)
    assert found["asset_vol"] < sigma_e
    dd = (math.log(v / debt) + (0.07 - sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    assert found["distance_to_default"] == pytest.approx(dd, rel=1e-8)


def test_the_kmv_default_point_is_short_debt_plus_half_the_long():
    e, sigma_e = equity_of(150.0, 0.3, 60.0 + 40.0 / 2, 1.0, 0.04)
    found = cr.merton_calibrate(e, sigma_e, 60.0, 40.0, "kmv", 1.0, 0.04, 0.04, "simultaneous", None, None)
    assert found["default_point"] == 80.0 and found["asset_value"] == pytest.approx(150.0, rel=1e-10)
    assert cr.merton_calibrate(e, sigma_e, 60.0, 40.0, "total", 1.0, 0.04, 0.04, "simultaneous", None,
                               None)["default_point"] == 100.0


def test_the_kmv_iteration_returns_an_asset_path_whose_own_vol_made_its_equity():
    """The iteration's fixed point: equity made from an asset path at that path's own realized vol (ddof 1, 252 a
    year) gives back the path, the vol, and Vassalou and Xing's drift, the mean log return plus sigma^2 / 2."""
    rng = np.random.default_rng(7)
    path = 120.0 * np.exp(np.r_[0.0, np.cumsum(0.0003 + 0.016 * rng.standard_normal(120))])
    sigma = float(np.std(np.diff(np.log(path)), ddof=1) * math.sqrt(252))
    equity = np.array([equity_of(v, sigma, 80.0, 1.0, 0.03)[0] for v in path])
    found = cr.merton_calibrate(equity, None, 80.0, 0.0, "total", 1.0, 0.03, 0.08, "kmv_iteration", 252, 1)
    assert found["asset_vol"] == pytest.approx(sigma, rel=1e-9) and found["iterations"] > 2
    assert found["asset_values"] == pytest.approx(path, rel=1e-9) and found["asset_value"] == pytest.approx(path[-1])
    steps = np.diff(np.log(path))
    assert found["asset_drift"] == pytest.approx(steps.mean() * 252 + sigma * sigma / 2, rel=1e-8)
    assert found["asset_vol"] < np.std(np.diff(np.log(equity)), ddof=1) * math.sqrt(252)


@pytest.mark.parametrize("bad, match", [
    (dict(method="newton"), "method"), (dict(barrier="face"), "barrier"), (dict(short_debt=0.0), "debt"),
    (dict(periods_per_year=252), "periods_per_year"), (dict(equity_vol=0.0), "equity_vol"),
    (dict(method="kmv_iteration", equity=[3.0, 3.1, 3.05]), "equity_vol"),
    (dict(method="kmv_iteration", equity=[3.0, 3.1], equity_vol=None, periods_per_year=252, ddof=1), "series")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(equity=3.0, equity_vol=0.8, short_debt=10.0, long_debt=0.0, barrier="total", horizon=1.0, rate=0.05,
                drift=0.05, method="simultaneous", periods_per_year=None, ddof=None)
    with pytest.raises(ValueError, match=match):
        qf.credit.merton_calibrate(**{**args, **bad})
