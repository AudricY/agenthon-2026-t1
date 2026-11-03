"""Lane lib-risk1 (runs/lanes/lib-risk1-prompt.md), the `tails` shelf: the peaks-over-threshold VaR and ES against an
exact GPD (McNeil, Frey and Embrechts 2005, 7.2.3, equations 7.18 and 7.19; scipy 1.18's genpareto) and on
gpd_fit's output; the GEV fit against scipy's genextreme and gumbel_r fits and against a population sample by
Hosking, Wallis and Wood's PWM; the moved Hill and GPD still `qf.risk.`. Synthetic samples alone; no unit's."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

import qf
import qf.risk

pytestmark = pytest.mark.fast

LOSS = "loss_positive"


@pytest.mark.parametrize("xi, beta", [(0.3, 1.5), (-0.2, 0.8), (0.0, 1.1)])
def test_on_an_exact_gpd_tail_the_var_and_es_are_its_quantile_and_tail_mean(xi, beta):
    """With the threshold at 0 and every loss above it (N_u = n), the estimate is the GPD itself: VaR its alpha-quantile
    and ES the mean of the quantile beyond it."""
    var, es = qf.gpd_var_es(threshold=0.0, xi=xi, beta=beta, n=1000, exceedances=1000, alpha=0.99, sign=LOSS)
    law = stats.genpareto(xi, scale=beta)
    assert var == pytest.approx(law.ppf(0.99), rel=1e-12)
    assert es == pytest.approx(integrate.quad(law.ppf, 0.99, 1.0, limit=200)[0] / 0.01, rel=1e-8)


def test_the_estimators_threshold_and_exceedance_scaling():
    """Equation 7.18 by hand at u = 2, N_u / n = 0.05; and continuity of the xi = 0 limit."""
    u, xi, beta, n, nu = 2.0, 0.25, 0.6, 1000, 50
    var, es = qf.gpd_var_es(threshold=u, xi=xi, beta=beta, n=n, exceedances=nu, alpha=0.995, sign=LOSS)
    by_hand = u + beta / xi * ((n / nu * 0.005) ** -xi - 1)
    assert var == pytest.approx(by_hand, rel=1e-14)
    assert es == pytest.approx(by_hand / (1 - xi) + (beta - xi * u) / (1 - xi), rel=1e-14)
    near = qf.gpd_var_es(threshold=u, xi=1e-7, beta=beta, n=n, exceedances=nu, alpha=0.995, sign=LOSS)
    zero = qf.gpd_var_es(threshold=u, xi=0.0, beta=beta, n=n, exceedances=nu, alpha=0.995, sign=LOSS)
    assert near == pytest.approx(zero, rel=1e-6) and zero[1] == pytest.approx(zero[0] + beta, rel=1e-15)
    assert qf.gpd_var_es(threshold=u, xi=xi, beta=beta, n=n, exceedances=nu, alpha=0.995,
                         sign="return_negative") == pytest.approx((-var, -es), rel=1e-15)
    assert qf.gpd_var_es(threshold=u, xi=1.2, beta=beta, n=n, exceedances=nu, alpha=0.995, sign=LOSS)[1] == math.inf


def test_on_gpd_fits_output_the_tail_estimate_tracks_the_true_quantile_of_a_student_t():
    """MFE 7.2.3's use: a t(4) sample, threshold its 95th percentile, gpd_fit, then the 99.5% VaR near the true one."""
    losses = stats.t.rvs(4, size=20000, random_state=5)
    u = float(np.quantile(losses, 0.95))
    fit = qf.gpd_fit(losses, threshold=u)
    var, es = qf.gpd_var_es(threshold=u, xi=fit["xi"], beta=fit["beta"], n=losses.size,
                            exceedances=fit["exceedances"], alpha=0.995, sign=LOSS)
    assert var == pytest.approx(stats.t.ppf(0.995, 4), rel=0.05) and es > var


@pytest.mark.parametrize("bad", [dict(exceedances=0), dict(exceedances=2000), dict(alpha=0.9), dict(beta=0.0)])
def test_the_estimator_refuses_a_level_inside_the_threshold_and_impossible_counts(bad):
    args = dict(threshold=2.0, xi=0.2, beta=0.5, n=1000, exceedances=50, alpha=0.99, sign=LOSS) | bad
    with pytest.raises(ValueError):
        qf.gpd_var_es(**args)


@pytest.mark.parametrize("xi", [0.2, -0.15, 0.0])
def test_the_gev_likelihood_fit_reaches_scipys_and_its_return_level_is_the_quantile(xi):
    """scipy 1.18's genextreme.fit (shape c = -xi) as the second optimiser: the lane's fit's log-likelihood is at least
    scipy's and its parameters agree; the k-block return level is genextreme's (1 - 1/k)-quantile."""
    m = stats.genextreme.rvs(-xi, loc=3.0, scale=0.8, size=300, random_state=4)
    fit = qf.gev_fit(losses=m, block_size=None, method="mle", shape="free", return_period=50)
    c, loc, scale = stats.genextreme.fit(m)
    assert fit["loglik"] >= np.sum(stats.genextreme.logpdf(m, c, loc, scale)) - 1e-9
    assert (fit["xi"], fit["mu"], fit["sigma"]) == pytest.approx((-c, loc, scale), abs=2e-4)
    assert fit["loglik"] == pytest.approx(np.sum(stats.genextreme.logpdf(m, -fit["xi"], fit["mu"], fit["sigma"])),
                                          rel=1e-12)
    assert fit["return_level"] == pytest.approx(stats.genextreme.ppf(1 - 1 / 50, -fit["xi"], fit["mu"], fit["sigma"]),
                                                rel=1e-12)


def test_the_gumbel_fit_is_scipys_gumbel_and_its_return_level_the_limit():
    m = stats.gumbel_r.rvs(loc=2.0, scale=0.5, size=400, random_state=8)
    fit = qf.gev_fit(losses=m, block_size=None, method="mle", shape="gumbel", return_period=10)
    assert (fit["mu"], fit["sigma"]) == pytest.approx(stats.gumbel_r.fit(m), rel=1e-7) and fit["xi"] == 0.0
    assert fit["return_level"] == pytest.approx(fit["mu"] - fit["sigma"] * math.log(-math.log(0.9)), rel=1e-14)
    pwm = qf.gev_fit(losses=m, block_size=None, method="pwm", shape="gumbel", return_period=None)
    s = np.sort(m)
    b0, b1 = s.mean(), np.mean(np.arange(s.size) / (s.size - 1) * s)
    assert pwm["sigma"] == pytest.approx((2 * b1 - b0) / math.log(2), rel=1e-12)
    assert pwm["mu"] == pytest.approx(b0 - np.euler_gamma * pwm["sigma"], rel=1e-12) and pwm["return_level"] is None


def test_pwm_recovers_the_parameters_of_a_population_sample():
    """Hosking, Wallis and Wood (1985): the PWM estimates of the GEV from its own quantiles at the plotting positions
    are the parameters, up to Hosking's approximation of the shape's equation (within 1e-3)."""
    x = stats.genextreme.ppf((np.arange(100000) + 0.5) / 100000, -0.2, loc=1.0, scale=2.0)
    fit = qf.gev_fit(losses=x, block_size=None, method="pwm", shape="free", return_period=None)
    assert (fit["xi"], fit["mu"], fit["sigma"]) == pytest.approx((0.2, 1.0, 2.0), abs=3e-3)


def test_block_maxima_take_whole_blocks_and_drop_the_last_incomplete_one():
    x = np.arange(23.0)
    fit = qf.gev_fit(losses=x, block_size=5, method="pwm", shape="free", return_period=None)
    assert list(fit["maxima"]) == [4.0, 9.0, 14.0, 19.0] and fit["blocks"] == 4
    with pytest.raises(ValueError):
        qf.gev_fit(losses=x[:10], block_size=5, method="mle", shape="free", return_period=None)


def test_hill_and_the_gpd_fit_moved_to_tails_and_still_resolve_on_risk():
    assert qf.risk.hill_estimator is qf.hill_estimator is qf.tails.hill_estimator
    assert qf.risk.gpd_fit is qf.gpd_fit is qf.tails.gpd_fit
    assert "hill_estimator" not in qf.risk.__all__ and "gpd_fit" in qf.tails.__all__
