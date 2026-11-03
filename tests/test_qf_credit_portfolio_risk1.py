"""Lane lib-risk1 (runs/lanes/lib-risk1-prompt.md), the `credit_portfolio` shelf's new functions: Vasicek's ES as the
tail mean of his quantile (Vasicek 2002; Acerbi and Tasche 2002) and its limits; the tranche loss under the one-factor
Gaussian copula against the integral of 1 - vasicek_loss_cdf (O'Kane 2008, ch. 16), across the capital structure, as
the finite pool grows, and against factor_copula_losses' simulation of a homogeneous pool; Basel's IRB correlations at
their stated values and through gordy_capital to Basel II's illustrative risk weights (Annex 5). Synthetic values
alone; no unit's."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

import qf
import qf.credit_portfolio

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("q, pd, rho", [(0.999, 0.02, 0.15), (0.99, 0.005, 0.3), (0.95, 0.1, 0.05)])
def test_vasiceks_es_is_the_mean_of_his_quantile_over_the_tail(q, pd, rho):
    es = qf.vasicek_loss_es(q, pd=pd, rho=rho, lgd=1.0)
    tail = integrate.quad(lambda u: qf.vasicek_loss_quantile(u, pd=pd, rho=rho), q, 1.0, limit=400,
                          epsabs=1e-14)[0] / (1 - q)
    assert es == pytest.approx(tail, rel=1e-7) and es > qf.vasicek_loss_quantile(q, pd=pd, rho=rho)
    assert qf.vasicek_loss_es(q, pd=pd, rho=rho, lgd=0.45) == pytest.approx(0.45 * es, rel=1e-14)


def test_vasiceks_es_falls_to_pd_as_the_correlation_vanishes_and_takes_arrays():
    assert qf.vasicek_loss_es(0.99, pd=0.03, rho=1e-8, lgd=1.0) == pytest.approx(0.03, rel=1e-3)
    found = qf.vasicek_loss_es(np.array([0.99, 0.999]), pd=0.02, rho=0.2, lgd=1.0)
    assert found.shape == (2,) and found[1] > found[0]
    with pytest.raises(ValueError):
        qf.vasicek_loss_es(0.99, pd=0.02, rho=0.2, lgd=45.0)


def tranche(a, d, **kw):
    args = dict(attachment=a, detachment=d, pd=0.05, recovery=0.4, rho=0.3, pool="large", names=None, nodes=None)
    return qf.tranche_loss(**(args | kw))


@pytest.mark.parametrize("a, d", [(0.0, 0.03), (0.03, 0.07), (0.07, 0.1), (0.1, 0.15), (0.15, 0.3), (0.3, 1.0)])
def test_the_large_pool_tranche_loss_is_the_integral_of_the_loss_laws_tail(a, d):
    """E[min(max(L - A, 0), D - A)] = int_A^D P(L > x) dx, L = (1 - R) X and X Vasicek's default rate."""
    found = tranche(a, d)
    tail = integrate.quad(lambda x: 1.0 - qf.vasicek_loss_cdf(min(x / 0.6, 1.0), pd=0.05, rho=0.3), a, d,
                          points=[0.6] if a < 0.6 < d else None, limit=400, epsabs=1e-14)[0]
    assert found["expected_loss_pool"] == pytest.approx(tail, rel=1e-7, abs=1e-14)
    assert found["expected_loss"] == pytest.approx(tail / (d - a), rel=1e-7, abs=1e-14)


def test_tranches_across_the_capital_structure_add_to_the_pools_loss_and_the_equity_tranche_falls_with_rho():
    """Hull, "Synthetic CDOs"; Li (2000): the tranches partition the pool's loss; correlation moves loss from the
    equity tranche to the senior ones."""
    cuts = [0.0, 0.03, 0.07, 0.1, 0.15, 0.3, 1.0]
    for pool, extra in (("large", {}), ("finite", dict(names=125, nodes=64))):
        total = sum(tranche(a, d, pool=pool, **extra)["expected_loss_pool"] for a, d in zip(cuts, cuts[1:]))
        assert total == pytest.approx(0.6 * 0.05, rel=1e-9), pool
    beyond = tranche(0.3, 0.8)["expected_loss_pool"]          # the pool never loses more than 1 - R = 0.6
    assert beyond == pytest.approx(tranche(0.3, 1.0)["expected_loss_pool"], rel=1e-14) and beyond > 0
    equity = [tranche(0.0, 0.03, rho=r)["expected_loss"] for r in (0.05, 0.2, 0.5, 0.8)]
    senior = [tranche(0.15, 0.3, rho=r)["expected_loss"] for r in (0.05, 0.2, 0.5, 0.8)]
    assert all(a > b for a, b in zip(equity, equity[1:])) and all(a < b for a, b in zip(senior, senior[1:]))


def test_the_finite_pool_tends_to_the_large_and_its_default_law_is_the_mixed_binomial():
    """The finite pool's excess over the large pool's falls about as 1/n (100 names to 400); the law of the default
    count is the binomial mixed over the factor, by quadrature."""
    large = tranche(0.03, 0.07)["expected_loss"]
    gaps = [tranche(0.03, 0.07, pool="finite", names=n, nodes=200)["expected_loss"] - large for n in (100, 400)]
    assert 0 < gaps[1] < gaps[0] / 3
    with pytest.raises(ValueError):
        tranche(0.03, 0.07, pool="finite", names=100, nodes=400)
    small = tranche(0.0, 0.1, pool="finite", names=10, nodes=64)
    law = small["default_distribution"]
    p_z = lambda z: stats.norm.cdf((stats.norm.ppf(0.05) - math.sqrt(0.3) * z) / math.sqrt(0.7))  # noqa: E731
    for k in (0, 1, 4):
        direct = integrate.quad(lambda z: stats.binom.pmf(k, 10, p_z(z)) * stats.norm.pdf(z), -12, 12)[0]
        assert law[k] == pytest.approx(direct, rel=1e-6)
    assert law.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.slow
def test_the_finite_pool_matches_factor_copula_losses_on_a_homogeneous_pool():
    """The census's variant: a homogeneous pool's loss law by the conditional binomial against a draw per name."""
    names, rho = 50, 0.25
    sim = qf.factor_copula_losses(pd=[0.05] * names, lgd=[0.6] * names, exposure=[1.0 / names] * names,
                                  loading=[math.sqrt(rho)] * names, factor_corr=[[1.0]], factor_index=[0] * names,
                                  copula="gaussian", nu=None, n=200000, rng=np.random.default_rng(3))
    simulated = np.mean(np.clip(sim["portfolio_losses"] - 0.03, 0.0, 0.04))
    found = tranche(0.03, 0.07, pool="finite", names=names, nodes=64, rho=rho)["expected_loss_pool"]
    assert simulated == pytest.approx(found, rel=0.02)


def test_basels_irb_correlations_at_their_stated_values():
    """Basel II (2006) paragraphs 272, 273, 328, 329 and 330; Basel III (2011) paragraph 102."""
    f50 = lambda pd: (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))  # noqa: E731
    f35 = lambda pd: (1 - math.exp(-35 * pd)) / (1 - math.exp(-35))  # noqa: E731
    corp = lambda pd: 0.12 * f50(pd) + 0.24 * (1 - f50(pd))  # noqa: E731
    for pd in (0.0003, 0.01, 0.05, 0.2):
        assert qf.basel_irb_correlation(pd=pd, asset_class="corporate", sales=None) == pytest.approx(corp(pd), rel=1e-14)
        assert qf.basel_irb_correlation(pd=pd, asset_class="large_financial", sales=None) == pytest.approx(
            1.25 * corp(pd), rel=1e-14)
        assert qf.basel_irb_correlation(pd=pd, asset_class="other_retail", sales=None) == pytest.approx(
            0.03 * f35(pd) + 0.16 * (1 - f35(pd)), rel=1e-14)
        assert qf.basel_irb_correlation(pd=pd, asset_class="mortgage", sales=None) == 0.15
        assert qf.basel_irb_correlation(pd=pd, asset_class="revolving", sales=None) == 0.04
    for sales, cut in ((2.0, 0.04), (5.0, 0.04), (27.5, 0.02), (50.0, 0.0), (80.0, 0.0)):
        assert qf.basel_irb_correlation(pd=0.01, asset_class="sme", sales=sales) == pytest.approx(corp(0.01) - cut,
                                                                                               rel=1e-13)
    assert qf.basel_irb_correlation(pd=1e-9, asset_class="corporate", sales=None) == pytest.approx(0.24, abs=1e-8)
    assert qf.basel_irb_correlation(pd=np.array([0.01, 0.02]), asset_class="corporate", sales=None).shape == (2,)
    with pytest.raises(ValueError):
        qf.basel_irb_correlation(pd=0.01, asset_class="sme", sales=None)


@pytest.mark.parametrize("pd, weight", [(0.0003, 14.44), (0.001, 29.65), (0.01, 92.32), (0.05, 149.86), (0.2, 238.23)])
def test_the_corporate_correlation_through_gordy_gives_basels_illustrative_risk_weights(pd, weight):
    """Basel II (2006), Annex 5: corporate exposures, LGD 45%, M 2.5 years, RW = 12.5 K."""
    rho = qf.basel_irb_correlation(pd=pd, asset_class="corporate", sales=None)
    k = qf.gordy_capital(pd=pd, lgd=0.45, rho=rho, q=0.999, expected_loss="subtract", maturity=2.5)
    assert 100 * 12.5 * k == pytest.approx(weight, abs=0.0151)


def test_migrations_moved_and_their_old_paths_still_resolve():
    for name in ("cohort_transition_matrix", "generator_from_transition", "creditmetrics"):
        assert getattr(qf.credit_portfolio, name) is getattr(qf, name) is getattr(qf.migrations, name)
        assert name not in qf.credit_portfolio.__all__ and name in qf.migrations.__all__
