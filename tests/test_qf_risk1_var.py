"""Lane lib-risk1 (runs/lanes/lib-risk1-prompt.md), the `risk` shelf's new VaR and ES: the normal mixture against the
normal and the integral of its own quantile; the spectral measure's box weight against the empirical ES (Acerbi 2002)
and its exponential weight's limits; Hull's cash-flow mapping example (Technical Note 25: the 0.3-year and 0.8-year
flows of a 0.8-year 10% Treasury); the copula simulation against the multivariate normal. Synthetic values and
textbook examples alone; no unit's."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

import qf

pytestmark = pytest.mark.fast

LOSS, RETURN = "loss_positive", "return_negative"


# ------------------------------------------------------------------------------------------------ normal mixture


def test_one_component_is_the_normal_and_the_sign_flips():
    """McNeil, Frey and Embrechts (2005), 2.2.4: a mixture of one normal is that normal."""
    one = qf.mixture_normal_var_es(weights=[1.0], means=[0.0004], stds=[0.012], alpha=0.99, sign=LOSS)
    assert one == pytest.approx(qf.normal_var_es(mu=0.0004, sigma=0.012, alpha=0.99, sign=LOSS), rel=1e-12)
    same = qf.mixture_normal_var_es(weights=[0.3, 0.7], means=[0.001, 0.001], stds=[0.02, 0.02], alpha=0.975,
                                    sign=LOSS)
    assert same == pytest.approx(qf.normal_var_es(mu=0.001, sigma=0.02, alpha=0.975, sign=LOSS), rel=1e-12)
    flipped = qf.mixture_normal_var_es(weights=[0.3, 0.7], means=[0.001, -0.002], stds=[0.02, 0.01], alpha=0.975,
                                       sign=RETURN)
    loss = qf.mixture_normal_var_es(weights=[0.3, 0.7], means=[0.001, -0.002], stds=[0.02, 0.01], alpha=0.975,
                                    sign=LOSS)
    assert flipped == pytest.approx((-loss[0], -loss[1]), rel=1e-15)


def test_the_mixture_var_is_its_cdfs_root_and_its_es_the_tail_mean_of_its_quantile():
    w, mu, sd = np.array([0.8, 0.2]), np.array([0.001, -0.004]), np.array([0.01, 0.035])
    var, es = qf.mixture_normal_var_es(weights=w, means=mu, stds=sd, alpha=0.99, sign=LOSS)
    cdf = lambda x: float(w @ stats.norm.cdf(x, mu, sd))  # noqa: E731
    assert cdf(-var) == pytest.approx(0.01, abs=1e-13)
    density = lambda x: float(w @ stats.norm.pdf(x, mu, sd))  # noqa: E731
    tail = integrate.quad(lambda x: x * density(x), -1.0, -var, epsabs=1e-14, limit=200)[0] / 0.01
    assert es == pytest.approx(-tail, rel=1e-9) and es > var       # the mean of the returns below the quantile
    stressed = qf.mixture_normal_var_es(weights=w, means=mu, stds=sd * np.array([1.0, 1.5]), alpha=0.99, sign=LOSS)
    assert stressed[0] > var and stressed[1] > es


def test_the_mixture_is_not_the_normal_at_its_moments():
    """The card's warning: the mixture's mean and std in the normal formula give a thinner tail."""
    w, mu, sd = np.array([0.9, 0.1]), np.array([0.0, 0.0]), np.array([0.01, 0.05])
    var, es = qf.mixture_normal_var_es(weights=w, means=mu, stds=sd, alpha=0.995, sign=LOSS)
    moment = qf.normal_var_es(mu=0.0, sigma=math.sqrt(w @ sd ** 2), alpha=0.995, sign=LOSS)
    assert var > moment[0] and es > moment[1]


@pytest.mark.parametrize("bad", [dict(weights=[0.5, 0.4]), dict(weights=[1.2, -0.2]), dict(stds=[0.01, 0.0])])
def test_the_mixture_refuses_weights_that_are_not_probabilities_and_a_zero_std(bad):
    args = dict(weights=[0.5, 0.5], means=[0.0, 0.0], stds=[0.01, 0.02], alpha=0.99, sign=LOSS) | bad
    with pytest.raises(ValueError):
        qf.mixture_normal_var_es(**args)


# ------------------------------------------------------------------------------------------------ spectral


R = np.array([-0.021, 0.004, -0.013, 0.008, -0.030, 0.011, -0.002, 0.006, -0.017, 0.012, -0.007, 0.003])


@pytest.mark.parametrize("alpha", [0.75, 0.9, 0.95])
def test_the_box_weight_gives_the_empirical_expected_shortfall(alpha):
    """Acerbi (2002): phi = 1/(1 - alpha) on (alpha, 1) is the ES; 'exact' integrates the empirical quantile function,
    which is historical_var_es's 'exact' ES at any n (1 - alpha) (2.4 and 1.2 order statistics here, 3 at 0.75)."""
    es = qf.historical_var_es(R, alpha=alpha, method="inverted_cdf", es_rule="exact", sign=LOSS)[1]
    found = qf.spectral_risk(returns=R, weight="expected_shortfall", risk_aversion=None, alpha=alpha,
                             discretisation="exact", sign=LOSS)
    assert found == pytest.approx(es, rel=1e-12)
    assert qf.spectral_risk(returns=R, weight="expected_shortfall", risk_aversion=None, alpha=alpha,
                            discretisation="exact", sign=RETURN) == pytest.approx(-es, rel=1e-12)


def test_the_exact_weights_integrate_the_weight_function_over_each_order_statistic():
    k, losses = 10.0, np.sort(-R)
    n = losses.size
    phi = lambda u: k * math.exp(-k * (1 - u)) / (1 - math.exp(-k))  # noqa: E731
    w = [integrate.quad(phi, i / n, (i + 1) / n)[0] for i in range(n)]
    found = qf.spectral_risk(returns=R, weight="exponential", risk_aversion=k, alpha=None, discretisation="exact",
                             sign=LOSS)
    assert found == pytest.approx(float(np.dot(w, losses)), rel=1e-12)
    given = qf.spectral_risk(returns=R, weight=lambda u: k * np.exp(-k * (1 - u)) / (1 - math.exp(-k)),
                             risk_aversion=None, alpha=None, discretisation="exact", sign=LOSS)
    assert given == pytest.approx(found, rel=1e-9)
    mid = np.array([phi((i + 0.5) / n) for i in range(n)])
    assert qf.spectral_risk(returns=R, weight="exponential", risk_aversion=k, alpha=None, discretisation="midpoint",
                            sign=LOSS) == pytest.approx(float(mid / mid.sum() @ losses), rel=1e-12)
    linear = np.array([2 * (i + 0.5) / n for i in range(n)])       # phi(u) = 2u, at each span's midpoint
    assert qf.spectral_risk(returns=R, weight=lambda u: 2 * u, risk_aversion=None, alpha=None,
                            discretisation="midpoint", sign=LOSS) == pytest.approx(float(linear / linear.sum() @ losses),
                                                                                   rel=1e-12)


def test_the_exponential_measure_rises_with_risk_aversion_from_the_mean_loss_to_the_worst():
    values = [qf.spectral_risk(returns=R, weight="exponential", risk_aversion=k, alpha=None, discretisation="exact",
                               sign=LOSS) for k in (1e-12, 1e-6, 1.0, 5.0, 25.0, 400.0)]
    assert values[0] == pytest.approx(float(np.mean(-R)), abs=1e-8)
    assert all(a < b for a, b in zip(values, values[1:])) and values[-1] == pytest.approx(0.030, abs=1e-6)


def test_the_spectral_measure_of_a_large_normal_sample_is_its_integral_and_is_comonotonic_additive():
    """On N(0, 1) returns, M = int_0^1 phi(u) Phi^-1(u) du; a comonotone pair (one series a positive multiple of the
    other) adds exactly."""
    x = stats.norm.ppf((np.arange(200000) + 0.5) / 200000)
    k = 8.0
    law = integrate.quad(lambda u: k * math.exp(-k * (1 - u)) / -math.expm1(-k) * stats.norm.ppf(u), 0, 1,
                         limit=400)[0]
    assert qf.spectral_risk(returns=x, weight="exponential", risk_aversion=k, alpha=None, discretisation="midpoint",
                            sign=LOSS) == pytest.approx(law, rel=2e-4)
    one = qf.spectral_risk(returns=R, weight="exponential", risk_aversion=k, alpha=None, discretisation="exact",
                           sign=LOSS)
    two = qf.spectral_risk(returns=3 * R, weight="exponential", risk_aversion=k, alpha=None, discretisation="exact",
                           sign=LOSS)
    both = qf.spectral_risk(returns=4 * R, weight="exponential", risk_aversion=k, alpha=None, discretisation="exact",
                            sign=LOSS)
    assert both == pytest.approx(one + two, rel=1e-12)


@pytest.mark.parametrize("bad", [dict(weight="exponential", risk_aversion=None), dict(weight="box"),
                                 dict(weight="exponential", risk_aversion=2.0, alpha=0.9),
                                 dict(weight=lambda u: 2.0 * np.ones_like(u))])
def test_the_spectral_measure_refuses_a_weight_it_cannot_use(bad):
    args = dict(returns=R, weight="exponential", risk_aversion=5.0, alpha=None, discretisation="exact",
                sign=LOSS) | bad
    with pytest.raises(ValueError):
        qf.spectral_risk(**args)


@pytest.mark.parametrize("discretisation", ["exact", "midpoint"])
def test_a_decreasing_spectrum_is_refused_because_it_breaks_subadditivity(discretisation):
    """Acerbi (2002), Theorem 5.3: the empirical spectrum must put nondecreasing weights on increasing losses.
    phi(u) = 2(1-u) would give 0.25 for each of two opposing unit losses and 1 for their sum."""
    with pytest.raises(ValueError, match="must not decrease"):
        qf.spectral_risk(returns=[-1.0, 0.0], weight=lambda u: 2 * (1 - u), risk_aversion=None, alpha=None,
                         discretisation=discretisation, sign=LOSS)


# ------------------------------------------------------------------------------------------------ cash-flow mapping


HULL = dict(vertices=[0.25, 0.5, 1.0], yields=[0.055, 0.06, 0.07], vols=[0.0006, 0.001, 0.002],
            corr=[[1.0, 0.9, 0.6], [0.9, 1.0, 0.7], [0.6, 0.7, 1.0]], compounding="annual", alpha=0.99, sign=LOSS)


def test_hulls_cash_flow_mapping_example():
    """Hull, Options, Futures, and Other Derivatives, Technical Note 25 ("A cash-flow mapping procedure"): a long 0.8-year
    Treasury, principal 1,000,000, 10% coupons semiannually: 50,000 at 0.3 years and 1,050,000 at 0.8 years; zero
    rates 5.50%, 6.00%, 7.00% (annual compounding) and daily bond price vols 0.06%, 0.10%, 0.20% at 3 months, 6 months,
    1 year; correlations 0.9, 0.6, 0.7. The 0.3-year flow: rate 5.60%, vol 0.068%, PV 49,189, alpha 0.760259, so
    37,397 to 3 months and 11,793 to 6 months; the 0.8-year flow: rate 6.60%, vol 0.16%, PV 997,662, alpha 0.320337,
    so 319,589 to 6 months and 678,074 to 1 year. (The note's PDF could not be opened from this lane; its figures are
    recalled, and the recalled inputs reproduce every recalled output to the digits printed.)"""
    found = qf.cash_flow_map(times=[0.3, 0.8], amounts=[50000.0, 1050000.0], **HULL)
    assert found["shares"] == pytest.approx([0.760259, 0.320337], abs=1e-6)   # six decimals as printed
    assert found["present_values"] == pytest.approx([49189, 997662], abs=0.5)
    assert found["mapped"] == pytest.approx([37397, 11793 + 319589, 678074], abs=1.0)
    c = np.array(HULL["corr"]) * np.outer(HULL["vols"], HULL["vols"])
    std = math.sqrt(found["mapped"] @ c @ found["mapped"])
    assert found["std"] == pytest.approx(std, rel=1e-12)
    assert found["var"] == pytest.approx(stats.norm.ppf(0.99) * std, rel=1e-12)


def test_a_mapped_flow_keeps_its_value_and_its_interpolated_variance():
    found = qf.cash_flow_map(times=[0.6], amounts=[100.0], **HULL)
    a = found["shares"][0]
    v = 0.001 + (0.6 - 0.5) / 0.5 * (0.002 - 0.001)
    assert found["mapped"].sum() == pytest.approx(found["present_values"][0], rel=1e-14)
    assert (a * 0.001) ** 2 + ((1 - a) * 0.002) ** 2 + 2 * 0.7 * a * (1 - a) * 0.001 * 0.002 == pytest.approx(v * v,
                                                                                                      rel=1e-12)
    assert found["present_values"][0] == pytest.approx(100.0 / 1.062 ** 0.6, rel=1e-14)


def test_a_flow_on_a_vertex_or_beyond_the_ends_goes_whole_and_continuous_compounding_discounts_exponentially():
    found = qf.cash_flow_map(times=[0.5, 0.1, 2.0], amounts=[10.0, 20.0, 30.0], **(HULL | dict(compounding="continuous")))
    assert list(found["shares"]) == [1.0, 1.0, 1.0]
    assert found["mapped"] == pytest.approx([20 * math.exp(-0.055 * 0.1), 10 * math.exp(-0.06 * 0.5),
                                             30 * math.exp(-0.07 * 2.0)], rel=1e-14)


def test_a_perfectly_correlated_pair_of_vertices_splits_linearly_in_the_vol():
    """With rho = 1 the combined vol is a sigma_L + (1 - a) sigma_R, so a is linear in the vol: here the time share."""
    found = qf.cash_flow_map(times=[0.4], amounts=[1.0], vertices=[0.25, 0.5], yields=[0.05, 0.05],
                             vols=[0.001, 0.002], corr=[[1.0, 1.0], [1.0, 1.0]], compounding="annual", alpha=0.99,
                             sign=LOSS)
    assert found["shares"][0] == pytest.approx(0.4, abs=1e-9)


# ------------------------------------------------------------------------------------------------ copula simulation


def test_a_gaussian_copula_with_normal_margins_is_the_delta_normal_var():
    """McNeil, Frey and Embrechts (2005), 5.1: the Gaussian copula with normal margins is the multivariate normal, so
    the simulated VaR and ES are the normal's of w' X within simulation error."""
    corr = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]])
    sd, mu, w = np.array([0.01, 0.02, 0.015]), np.array([0.001, 0.0, -0.0005]), np.array([0.5, 0.3, 0.2])
    found = qf.copula_loss_simulation(copula="gaussian", corr=corr, nu=None,
                                      marginals=[("normal", m, s) for m, s in zip(mu, sd)], sigma_kind=None,
                                      weights=w, n=200000, rng=np.random.default_rng(3), alpha=0.99, sign=LOSS)
    sigma = math.sqrt(w @ (corr * np.outer(sd, sd)) @ w)
    var, es = qf.normal_var_es(mu=float(w @ mu), sigma=sigma, alpha=0.99, sign=LOSS)
    assert found["var"] == pytest.approx(var, rel=0.02) and found["es"] == pytest.approx(es, rel=0.02)
    assert found["asset_returns"].shape == (200000, 3)


def test_the_simulation_maps_the_samplers_own_draws_through_each_margin():
    corr = [[1.0, 0.6], [0.6, 1.0]]
    u = qf.t_copula_sample(corr, nu=4.0, n=2000, rng=np.random.default_rng(9))
    law = stats.t(5.0, loc=0.001, scale=0.01)
    found = qf.copula_loss_simulation(copula="t", corr=corr, nu=4.0, marginals=[law, ("t", 0.0, 0.02, 6.0)],
                                      sigma_kind="std", weights=[2.0, -1.0], n=2000, rng=np.random.default_rng(9),
                                      alpha=0.95, sign=LOSS)
    x0 = law.ppf(u[:, 0])
    x1 = 0.02 * math.sqrt(4.0 / 6.0) * stats.t.ppf(u[:, 1], 6.0)
    assert found["asset_returns"] == pytest.approx(np.column_stack([x0, x1]), rel=1e-12, abs=1e-15)
    pnl = 2.0 * x0 - x1
    assert found["var"] == pytest.approx(qf.historical_var_es(pnl, alpha=0.95, method="inverted_cdf", es_rule="exact",
                                                              sign=LOSS)[0], rel=1e-12)
    scale = qf.copula_loss_simulation(copula="t", corr=corr, nu=4.0, marginals=[law, ("t", 0.0, 0.02, 6.0)],
                                      sigma_kind="scale", weights=[2.0, -1.0], n=2000, rng=np.random.default_rng(9),
                                      alpha=0.95, sign=LOSS)
    assert scale["asset_returns"][:, 1] == pytest.approx(0.02 * stats.t.ppf(u[:, 1], 6.0), rel=1e-12)


@pytest.mark.parametrize("bad", [dict(nu=4.0), dict(marginals=[("normal", 0.0, 0.01)]),
                                 dict(marginals=[("t", 0.0, 0.01, 5.0), ("normal", 0.0, 0.01)]),
                                 dict(marginals=[("lognormal", 0.0, 0.01), ("normal", 0.0, 0.01)])])
def test_the_simulation_refuses_a_t_margin_without_its_dispersion_kind_and_a_wrong_count(bad):
    args = dict(copula="gaussian", corr=[[1.0, 0.3], [0.3, 1.0]], nu=None,
                marginals=[("normal", 0.0, 0.01), ("normal", 0.0, 0.02)], sigma_kind=None, weights=[0.5, 0.5],
                n=100, rng=np.random.default_rng(1), alpha=0.99, sign=LOSS) | bad
    with pytest.raises(ValueError):
        qf.copula_loss_simulation(**args)
