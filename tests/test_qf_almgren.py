"""qf.execution.almgren_chriss (runs/lanes/lib7-prompt.md, rank 14 of docs/research/library-syllabus-1011.md): the
discrete trajectory against the mean-variance problem solved directly, its cost and variance by their sums, the
continuous integrals by quadrature, TWAP at lambda = 0 and the discrete trajectory's limit. The impact parameters are
Almgren and Chriss's (2000) illustrative ones and this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import integrate

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf.execution as ex

pytestmark = pytest.mark.fast

AC = dict(quantity=1e6, horizon=5.0, vol=0.95, temporary_impact=2.5e-6, permanent_impact=2.5e-7, fixed_cost=0.0625)


def run(intervals, lam, time, **changes):
    return ex.almgren_chriss(**{**AC, **changes}, intervals=intervals, risk_aversion=lam, time=time)


@pytest.mark.parametrize("lam", [0.0, 2e-7, 1e-6, 2e-5])
@pytest.mark.parametrize("intervals", [1, 5, 12])
def test_the_discrete_holdings_solve_the_mean_variance_problem(lam, intervals):
    """E + lambda Var as a quadratic in x_1..x_{N-1} (x_0 = X, x_N = 0, monotone sales): (eta~/tau) sum (x_{k-1} -
    x_k)^2 + lambda sigma^2 tau sum x_k^2; its stationary point from the tridiagonal system written out here."""
    found = run(intervals, lam, "discrete")
    x, t, n = AC["quantity"], AC["horizon"], intervals
    tau = t / n
    eta = AC["temporary_impact"] - AC["permanent_impact"] * tau / 2
    if n > 1:
        a = 4 * eta / tau + 2 * lam * AC["vol"] ** 2 * tau
        system = np.diag(np.full(n - 1, a)) - np.diag(np.full(n - 2, 2 * eta / tau), 1) - \
            np.diag(np.full(n - 2, 2 * eta / tau), -1)
        right = np.zeros(n - 1)
        right[0] = 2 * eta / tau * x
        inner = np.linalg.solve(system, right)
        assert found["holdings"][1:-1] == pytest.approx(inner, rel=1e-10, abs=1e-6)
    sales = -np.diff(found["holdings"])
    assert found["trades"] == pytest.approx(sales) and found["trades"].sum() == pytest.approx(x, rel=1e-14)
    cost = 0.5 * AC["permanent_impact"] * x * x + AC["fixed_cost"] * x + eta / tau * np.sum(sales ** 2)
    variance = AC["vol"] ** 2 * tau * np.sum(found["holdings"][1:] ** 2)
    assert found["expected_cost"] == pytest.approx(cost, rel=1e-13)
    assert found["variance"] == pytest.approx(variance, rel=1e-13)
    assert found["objective"] == pytest.approx(cost + lam * variance, rel=1e-13)


def test_lambda_zero_is_twap_in_both_times_and_a_larger_lambda_trades_sooner():
    for time in ("discrete", "continuous"):
        assert run(5, 0.0, time)["holdings"] == pytest.approx(np.linspace(1e6, 0.0, 6), rel=1e-15, abs=1e-9)
    firsts = [run(5, lam, "discrete")["holdings"][1] for lam in (0.0, 1e-7, 1e-6, 1e-5)]
    assert np.all(np.diff(firsts) < 0)
    twap = run(5, 0.0, "continuous")
    assert twap["variance"] == pytest.approx(AC["vol"] ** 2 * 1e12 * 5.0 / 3, rel=1e-15)
    assert twap["expected_cost"] == pytest.approx(0.5 * 2.5e-7 * 1e12 + 0.0625 * 1e6 + 2.5e-6 * 1e12 / 5.0, rel=1e-15)


@pytest.mark.parametrize("lam", [1e-14, 4e-12, 1e-8, 1e-6, 1e-4])
def test_the_continuous_cost_and_variance_are_the_integrals_of_the_trajectory(lam):
    """x(t) = X sinh(kappa (T - t)) / sinh(kappa T), kappa^2 = lambda sigma^2 / eta: E = gamma X^2/2 + epsilon X +
    eta int xdot^2, Var = sigma^2 int x^2, by quadrature, on both sides of the series' threshold (kappa T = 0.01)."""
    found = run(5, lam, "continuous")
    kappa, t, x = found["kappa"], AC["horizon"], AC["quantity"]
    assert kappa == pytest.approx(math.sqrt(lam * AC["vol"] ** 2 / AC["temporary_impact"]), rel=1e-15)
    path = lambda u: x * math.sinh(kappa * (t - u)) / math.sinh(kappa * t)
    speed = lambda u: x * kappa * math.cosh(kappa * (t - u)) / math.sinh(kappa * t)
    rate = integrate.quad(lambda u: speed(u) ** 2, 0, t, epsrel=1e-13, limit=200)[0]
    level = integrate.quad(lambda u: path(u) ** 2, 0, t, epsrel=1e-13, limit=200)[0]
    assert found["expected_cost"] == pytest.approx(0.5 * 2.5e-7 * x * x + 0.0625 * x + 2.5e-6 * rate, rel=1e-11)
    assert found["variance"] == pytest.approx(AC["vol"] ** 2 * level, rel=1e-11)
    assert found["holdings"] == pytest.approx([path(u) for u in found["times"]], rel=1e-12, abs=1e-6)


def test_the_discrete_trajectory_tends_to_the_continuous_one():
    """As N grows eta~ tends to eta, kappa to its continuous value, and the sums to the integrals, at first order."""
    limit = run(2000, 1e-6, "continuous")
    coarse, fine = run(500, 1e-6, "discrete"), run(2000, 1e-6, "discrete")
    assert fine["kappa"] == pytest.approx(limit["kappa"], rel=1e-4)
    assert fine["holdings"] == pytest.approx(limit["holdings"], rel=1e-3, abs=1.0)
    for key in ("variance", "expected_cost"):
        assert fine[key] == pytest.approx(limit[key], rel=3e-3)
        assert abs(fine[key] - limit[key]) < abs(coarse[key] - limit[key]) / 3


def test_a_steep_risk_aversion_sells_at_once_without_overflow():
    found = run(5, 1e-1, "continuous")
    assert np.all(np.isfinite(found["holdings"])) and found["holdings"][1] < 1e-60
    assert found["variance"] == pytest.approx(AC["vol"] ** 2 * 1e12 / (2 * found["kappa"]), rel=1e-12)


@pytest.mark.parametrize("bad, match", [(dict(time="hourly"), "time"), (dict(temporary_impact=1e-7), "eta~"),
                                        (dict(vol=-1.0), "vol"), (dict(intervals=0), "intervals"),
                                        (dict(temporary_impact=0.0, time="continuous"), "temporary_impact")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(**AC, intervals=5, risk_aversion=1e-6, time="discrete")
    with pytest.raises(ValueError, match=match):
        ex.almgren_chriss(**{**args, **bad})
