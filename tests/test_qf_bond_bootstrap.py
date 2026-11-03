"""qf.rates.bootstrap_bond_prices (runs/lanes/lib7-prompt.md, rank 15 of docs/research/library-syllabus-1011.md):
coupon bonds priced off a curve built here come back as its nodes and reprice, by the sequential bootstrap and by
least squares; zero-coupon bonds give their discount factors; a smooth fit is the weighted least-squares optimum.
Every input is this file's own; no unit's values appear (rule 8)."""
import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf
import qf.rates as r

pytestmark = pytest.mark.fast

NODES = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
ZEROS = np.array([0.031, 0.034, 0.037, 0.039, 0.042])


def bonds(maturities, coupons):
    """Semiannual coupon bonds on 100, the first period a stub where the maturity is not whole half-years."""
    flows, times = [], []
    for maturity, coupon in zip(maturities, coupons):
        t = maturity - np.arange(int(np.ceil(maturity * 2 - 1e-9)))[::-1] / 2
        cf = np.full(t.size, 100 * coupon / 2)
        cf[-1] += 100
        flows.append(cf), times.append(t)
    return flows, times


def priced(flows, times, interpolation):
    return np.array([cf @ qf.curve_discount(t, NODES, np.exp(-ZEROS * NODES), interpolation)
                     for cf, t in zip(flows, times)])


@pytest.mark.parametrize("interpolation", ["linear_zero", "log_linear_discount", "cubic_log_discount"])
def test_the_sequential_bootstrap_returns_the_curve_that_priced_the_bonds(interpolation):
    flows, times = bonds(NODES, [0.03, 0.035, 0.04, 0.02, 0.05])
    found = qf.bootstrap_bond_prices(flows, times, priced(flows, times, interpolation), None, "sequential", None, None,
                                    interpolation, "continuous")
    assert found["node_times"] == pytest.approx(NODES) and found["zero_rates"] == pytest.approx(ZEROS, rel=1e-11)
    assert np.all(np.abs(found["residuals"]) < 1e-11) and np.all(np.diff(found["discount_factors"]) < 0)


def test_clean_prices_with_accrued_are_the_dirty_prices_and_zeros_give_their_discount_factors():
    flows, times = bonds(NODES, [0.03, 0.035, 0.04, 0.02, 0.05])
    dirty = priced(flows, times, "log_linear_discount")
    accrued = np.array([0.4, 1.1, 0.0, 0.7, 2.2])
    one = qf.bootstrap_bond_prices(flows, times, dirty, None, "sequential", None, None, "log_linear_discount",
                                  "continuous")
    two = qf.bootstrap_bond_prices(flows, times, dirty - accrued, accrued, "sequential", None, None,
                                  "log_linear_discount", "continuous")
    assert two["discount_factors"] == pytest.approx(one["discount_factors"], rel=1e-14)
    zeros = qf.bootstrap_bond_prices([[100.0], [100.0]], [[0.75], [2.5]], [97.5, 91.0], None, "sequential", None, None,
                                    "log_linear_discount", "continuous")
    assert zeros["discount_factors"] == pytest.approx([0.975, 0.91], rel=1e-14)


def test_least_squares_on_the_maturities_is_the_sequential_bootstrap():
    flows, times = bonds([4.0, 0.5, 2.0, 3.0, 1.0], [0.04, 0.03, 0.035, 0.03, 0.02])
    prices = np.array([103.0, 100.4, 101.0, 100.2, 99.4])
    sequential = qf.bootstrap_bond_prices(flows, times, prices, None, "sequential", None, None, "linear_zero", "continuous")
    fitted = qf.bootstrap_bond_prices(flows, times, prices, None, "least_squares", sequential["node_times"], "equal",
                                     "linear_zero", "continuous")
    assert fitted["zero_rates"] == pytest.approx(sequential["zero_rates"], rel=1e-9)
    assert np.all(np.abs(fitted["residuals"]) < 1e-9)


def test_a_smooth_fit_is_the_weighted_least_squares_optimum():
    """Nine bonds on three nodes: the weighted residuals' gradient in the node zero rates is zero (a central
    difference here), the fit is no worse than the curve that priced the bonds plus noise, and a bond weighted up
    is fitted closer."""
    rng = np.random.default_rng(5)
    maturities = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    flows, times = bonds(maturities, rng.uniform(0.01, 0.06, 9))
    prices = priced(flows, times, "cubic_log_discount") + rng.normal(0, 0.05, 9)
    nodes = np.array([1.0, 3.0, 5.0])
    weights = rng.uniform(0.5, 2.0, 9)
    found = qf.bootstrap_bond_prices(flows, times, prices, None, "least_squares", nodes, weights, "cubic_log_discount",
                                     "continuous")

    def loss(zeros):
        model = np.array([cf @ qf.curve_discount(t, nodes, np.exp(-zeros * nodes), "cubic_log_discount")
                          for cf, t in zip(flows, times)])
        return weights @ (model - prices) ** 2

    z = found["zero_rates"]
    for k in range(3):
        step = np.eye(3)[k] * 1e-6
        assert abs(loss(z + step) - loss(z - step)) / 2e-6 < 1e-4
    assert loss(z) <= loss(np.interp(nodes, NODES, ZEROS))
    heavier = weights.copy()
    heavier[4] *= 50
    again = qf.bootstrap_bond_prices(flows, times, prices, None, "least_squares", nodes, heavier, "cubic_log_discount",
                                     "continuous")
    assert abs(again["residuals"][4]) < abs(found["residuals"][4])


@pytest.mark.parametrize("bad, match", [
    (dict(method="spline"), "method"), (dict(node_times=[1.0, 2.0]), "node_times"),
    (dict(method="least_squares", node_times=[1.0, 1.5]), "node_times"), (dict(prices=[99.0]), "prices"),
    (dict(method="least_squares", node_times=[1.0, 2.0], weights=[1.0]), "weights"),
    (dict(cash_flows=[[100.0], [100.0]], times=[[1.0], [1.0]], prices=[97.0, 96.0]), "one bond per maturity"),
    (dict(accrued=[0.1]), "accrued"), (dict(times=[[0.0], [2.0]]), "after settlement")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(cash_flows=[[100.0], [100.0]], times=[[1.0], [2.0]], prices=[97.0, 93.0], accrued=None,
                method="sequential", node_times=None, weights=None, interpolation="linear_zero",
                compounding="continuous")
    with pytest.raises(ValueError, match=match):
        qf.bootstrap_bond_prices(**{**args, **bad})
