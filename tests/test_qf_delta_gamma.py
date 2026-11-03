"""qf.risk.delta_gamma_var (runs/lanes/lib7-prompt.md, rank 21 of docs/research/library-syllabus-1011.md): the
quadratic P&L's cumulants against the sum of independent terms after diagonalising (and the page's one-factor
forms), gamma = 0 against the delta-normal VaR, Cornish-Fisher against a Monte Carlo of the quadratic, and the
simulation against historical_var_es of the same draws. Every input is this file's own; no unit's values appear
(rule 8)."""
import math

import numpy as np
import pytest

from test_qf_conventions import card_problems  # noqa: F401 - the shelf's card check is test_qf_risk's
import qf.risk as risk

pytestmark = pytest.mark.fast

DELTA = np.array([120.0, -40.0, 15.0])
GAMMA = np.array([[-6.0, 1.5, 0.0], [1.5, 2.0, -0.8], [0.0, -0.8, 0.5]])
COV = np.array([[4.0, 1.2, -0.3], [1.2, 2.5, 0.4], [-0.3, 0.4, 1.0]])


def run(method="cornish_fisher", n=None, rng=None, **changes):
    args = dict(delta=DELTA, gamma=GAMMA, covariance=COV, horizon=10, alpha=0.99, method=method, n=n, rng=rng,
                sign="loss_positive")
    return risk.delta_gamma_var(**{**args, **changes})


def test_the_cumulants_are_the_independent_terms_after_diagonalising():
    """A^{1/2} Gamma A^{1/2} = V diag(l) V': P&L = sum_i (c_i y_i + l_i y_i^2 / 2), y iid N(0, 1), c = V' A^{1/2}
    delta; each term's cumulants l/2, c^2 + l^2/2, 3 c^2 l + l^3, 12 c^2 l^2 + 3 l^4."""
    a = COV * 10
    values, vectors = np.linalg.eigh(a)
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    l, v = np.linalg.eigh(root @ GAMMA @ root)
    c = v.T @ root @ DELTA
    k1, k2 = np.sum(l / 2), np.sum(c * c + l * l / 2)
    k3, k4 = np.sum(3 * c * c * l + l ** 3), np.sum(12 * c * c * l * l + 3 * l ** 4)
    found = run()
    assert found["mean"] == pytest.approx(k1, rel=1e-12) and found["std"] == pytest.approx(math.sqrt(k2), rel=1e-12)
    assert found["skew"] == pytest.approx(k3 / k2 ** 1.5, rel=1e-10)
    assert found["excess_kurtosis"] == pytest.approx(k4 / k2 ** 2, rel=1e-10)


def test_one_factor_gives_the_pages_forms_and_gamma_zero_the_delta_normal_var():
    d, g, s2 = 50.0, -8.0, 4.0
    found = run(delta=d, gamma=g, covariance=s2, horizon=1)
    assert found["std"] ** 2 == pytest.approx(d * d * s2 + 0.5 * g * g * s2 * s2, rel=1e-14)
    assert found["skew"] * found["std"] ** 3 == pytest.approx(3 * d * d * g * s2 ** 2 + g ** 3 * s2 ** 3, rel=1e-13)
    flat = run(gamma=np.zeros((3, 3)))
    var, es = risk.normal_var_es(mu=0.0, sigma=math.sqrt(DELTA @ (COV * 10) @ DELTA), alpha=0.99, sign="loss_positive")
    assert (flat["var"], flat["es"]) == pytest.approx((var, es), rel=1e-12)


def test_cornish_fisher_against_a_monte_carlo_of_the_quadratic():
    """Within 2% of the simulated quantile at 99% on 400,000 draws (the expansion's error at this skew), the
    simulation within its own sampling error of the CF's moments."""
    cf = run()
    mc = run("simulation", 400_000, np.random.default_rng(12))
    assert cf["var"] == pytest.approx(mc["var"], rel=0.02)
    assert cf["es"] == pytest.approx(mc["es"], rel=0.03)


def test_the_simulation_is_historical_var_of_the_same_draws():
    rng = np.random.default_rng(5)
    found = run("simulation", 2000, rng)
    z = np.random.default_rng(5).standard_normal((2000, 3))
    values, vectors = np.linalg.eigh(COV * 10)
    x = z @ (vectors * np.sqrt(values)).T
    pnl = x @ DELTA + 0.5 * np.array([row @ GAMMA @ row for row in x])
    expected = risk.historical_var_es(pnl, alpha=0.99, method="inverted_cdf", es_rule="exact", sign="loss_positive")
    assert (found["var"], found["es"]) == pytest.approx(expected, rel=1e-12)
    assert run("simulation", 2000, np.random.default_rng(5), sign="return_negative")["var"] == -found["var"]


@pytest.mark.parametrize("bad, match", [
    (dict(method="normal"), "method"), (dict(n=100), "None"), (dict(gamma=np.ones((2, 2))), "k x k"),
    (dict(covariance=-COV), "semidefinite"), (dict(horizon=0), "horizon"), (dict(method="simulation", n=10, rng="seed"), "rng")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        run(**bad)
