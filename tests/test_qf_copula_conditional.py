"""Lane lib-risk1 (runs/lanes/lib-risk1-prompt.md), conditional copula sampling on the `copulas` shelf: each family's
closed-form inverse h-function against the numerical root of its h-function, the h-function against the copula's
partial derivative by finite differences, and the sample's Kendall's tau against copula_param_from_tau's (Nelsen 2006,
2.9 and 5.1). Synthetic uniforms alone; no unit's."""
import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtri

import qf
from qf import copulas

pytestmark = pytest.mark.fast

U = np.array([0.02, 0.1, 0.5, 0.9, 0.995])
W = np.array([0.3, 0.97, 0.6, 0.01, 0.5])
CASES = [("gaussian", 0.6, None), ("gaussian", -0.4, None), ("t", 0.6, 4.0), ("clayton", 2.0, None),
         ("frank", 5.0, None), ("frank", -3.0, None)]


@pytest.mark.parametrize("family, parameter, nu", CASES)
def test_the_closed_form_inverse_is_the_root_of_the_h_function(family, parameter, nu):
    closed = qf.conditional_copula_sample(u=U, family=family, parameter=parameter, nu=nu, inverse="closed_form", w=W,
                                          rng=None)
    numerical = qf.conditional_copula_sample(u=U, family=family, parameter=parameter, nu=nu, inverse="numerical",
                                             w=W, rng=None)
    assert closed == pytest.approx(numerical, rel=1e-9, abs=1e-14)
    assert copulas._h(U, closed, family, parameter, nu) == pytest.approx(W, abs=1e-12)


def cdf(family, u, v, p, nu):
    if family == "gaussian":
        return stats.multivariate_normal([0, 0], [[1, p], [p, 1]]).cdf([ndtri(u), ndtri(v)])
    if family == "clayton":
        return (u ** -p + v ** -p - 1) ** (-1 / p)
    if family == "gumbel":
        return np.exp(-((-np.log(u)) ** p + (-np.log(v)) ** p) ** (1 / p))
    return -np.log1p(np.expm1(-p * u) * np.expm1(-p * v) / np.expm1(-p)) / p


@pytest.mark.parametrize("family, parameter", [("clayton", 2.0), ("gumbel", 1.8), ("frank", 5.0), ("frank", -3.0)])
def test_the_h_function_is_the_copulas_partial_derivative_in_u(family, parameter):
    """C_{2|1}(v | u) = dC(u, v)/du (Nelsen 2006, theorem 2.2.7), by a central difference of the copula itself."""
    for u, v in ((0.2, 0.7), (0.6, 0.3), (0.9, 0.95)):
        step = 1e-6
        derivative = (cdf(family, u + step, v, parameter, None) - cdf(family, u - step, v, parameter, None)) / (2 * step)
        assert copulas._h(np.array(u), np.array(v), family, parameter, None) == pytest.approx(derivative, rel=1e-6)


@pytest.mark.parametrize("family, tau", [("gaussian", 0.4), ("t", 0.4), ("clayton", 0.4), ("gumbel", 0.4),
                                         ("frank", 0.4), ("frank", -0.3)])
def test_the_pairs_have_the_familys_kendalls_tau(family, tau):
    """Nelsen (2006), 5.1: the conditional method draws the copula itself, so (U, V) has the tau its parameter came
    from (5,000 pairs; Kendall's tau within 0.03)."""
    rng = np.random.default_rng(5)
    u = rng.uniform(size=5000)
    p = qf.copula_param_from_tau(family, tau)
    v = qf.conditional_copula_sample(u=u, family=family, parameter=p, nu=5.0 if family == "t" else None,
                                     inverse="numerical" if family == "gumbel" else "closed_form", w=None, rng=rng)
    assert stats.kendalltau(u, v).statistic == pytest.approx(tau, abs=0.03)
    assert stats.kstest(v, "uniform").pvalue > 0.001


def test_rng_draws_w_uniformly_and_exactly_one_of_w_and_rng_is_given():
    v = qf.conditional_copula_sample(u=0.3, family="clayton", parameter=1.5, nu=None, inverse="closed_form", w=None,
                                     rng=np.random.default_rng(2))
    w = np.random.default_rng(2).uniform(size=())
    assert v == pytest.approx(float(qf.conditional_copula_sample(u=0.3, family="clayton", parameter=1.5, nu=None,
                                                                 inverse="closed_form", w=w, rng=None)), rel=1e-15)
    assert isinstance(v, float)
    for bad in (dict(w=None, rng=None), dict(w=[0.5] * 5, rng=np.random.default_rng(1)),
                dict(family="gumbel", parameter=2.0), dict(family="clayton", parameter=-1.0),
                dict(family="t", parameter=0.5, nu=None), dict(nu=4.0)):
        args = dict(u=U, family="clayton", parameter=2.0, nu=None, inverse="closed_form", w=W, rng=None) | bad
        with pytest.raises(ValueError):
            qf.conditional_copula_sample(**args)
