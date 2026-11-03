"""agent/qf/risk.py (runs/lanes/lib1-prompt.md): every function against a published table or worked example
(Kupiec's nonrejection regions, Basel's traffic-light probabilities and IRB risk weights, the normal ES/VaR constants
of McNeil, Frey and Embrechts), an identity (ES as the integral of the quantile, Euler's theorem, expm of the
generator, the Renyi representation behind Hill), an exact distribution the method approximates (Cornish-Fisher
against the gamma), or an independent implementation as the oracle (QuantLib 1.43's RiskStatistics where the lane's
environment has it, scipy's gaussian_kde and genpareto). Monte Carlo checks are seeded, at four standard errors. No
unit's instruction values, checks, expected values or data appear."""
import math

import numpy as np
import pytest
from scipy import integrate, linalg, stats
from scipy.special import ndtr, ndtri

import qf
import qf.risk as risk
import qf.tails as tails     # Hill and the GPD's shelf since lane lib-risk1
from test_qf_conventions import card_problems, run_examples

pytestmark = pytest.mark.fast

LOSS, RETURN = "loss_positive", "return_negative"


def sample(n, seed, df=None):
    rng = np.random.default_rng(seed)
    return (rng.standard_t(df, n) if df else rng.standard_normal(n)) * 0.01


def test_the_cards_follow_the_format_and_every_example_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert card_problems(risk) == []
    run_examples(risk, tmp_path)


# ------------------------------------------------------------------------------------------------ historical


def test_historical_var_takes_the_named_quantile_rule_on_returns_and_three_tail_rules():
    """Jorion, historical simulation; Acerbi and Tasche (2002), the exact empirical ES. 23 returns at alpha 0.9:
    the tail holds n (1 - alpha) = 2.3 observations, so the rules differ and each is written out on the order
    statistics."""
    r = sample(23, 1)
    s = np.sort(r)
    h = 22 * 0.1                                                   # numpy's 0-based position for 'linear'
    expected = {"linear": s[2] + (h - 2) * (s[3] - s[2]), "lower": s[2], "higher": s[3], "inverted_cdf": s[2]}
    for method, q in expected.items():
        var, es = risk.historical_var_es(r, alpha=0.9, method=method, es_rule="at_or_beyond", sign=LOSS)
        assert var == pytest.approx(-q, abs=1e-15), method
        assert es == pytest.approx(-np.mean(s[s <= q]), abs=1e-15), method
    _, es = risk.historical_var_es(r, alpha=0.9, method="lower", es_rule="beyond", sign=LOSS)
    assert es == pytest.approx(-np.mean(s[:2]), abs=1e-15)
    _, es = risk.historical_var_es(r, alpha=0.9, method="linear", es_rule="exact", sign=LOSS)
    assert es == pytest.approx(-(s[0] + s[1] + 0.3 * s[2]) / 2.3, abs=1e-15)
    var, es = risk.historical_var_es(r, alpha=0.9, method="linear", es_rule="exact", sign=RETURN)
    assert (var, es) == pytest.approx((expected["linear"], (s[0] + s[1] + 0.3 * s[2]) / 2.3), abs=1e-15)
    for bad in (dict(alpha=1.0, method="linear"), dict(alpha=0.99, method="median")):
        with pytest.raises(ValueError):
            risk.historical_var_es(r, es_rule="exact", sign=LOSS, **bad)
    with pytest.raises(ValueError, match="beyond"):
        risk.historical_var_es(r, alpha=0.99, method="lower", es_rule="beyond", sign=LOSS)
    with pytest.raises(ValueError, match="NaN"):
        risk.historical_var_es([0.01, math.nan, -0.02], alpha=0.9, method="linear", es_rule="exact", sign=LOSS)


def test_one_minus_alpha_is_exact_so_a_step_rule_takes_the_right_order_statistic():
    """1 - 0.95 is 0.050000000000000044 in floating point, which moves numpy's 'inverted_cdf' from the 10th to the
    11th worst of 200; F_n(x_(10)) = 10/200 = 0.05, so the definition takes the 10th."""
    r = sample(200, 2)
    s = np.sort(r)
    assert np.quantile(r, 1 - 0.95, method="inverted_cdf") == s[10]
    var, _ = risk.historical_var_es(r, alpha=0.95, method="inverted_cdf", es_rule="exact", sign=LOSS)
    assert var == -s[9]


def test_the_exact_es_is_the_integral_of_the_empirical_quantile_function():
    """ES = (1/(1 - alpha)) int_0^{1-alpha} F_n^-1(u) du, by a midpoint sum of numpy's 'inverted_cdf' quantiles."""
    r = sample(137, 3, df=4)
    for alpha in (0.9, 0.95, 0.99):
        tail = 1 - alpha
        u = (np.arange(20000) + 0.5) / 20000 * tail
        integral = np.mean(np.quantile(r, u, method="inverted_cdf"))
        _, es = risk.historical_var_es(r, alpha=alpha, method="hazen", es_rule="exact", sign=RETURN)
        assert es == pytest.approx(integral, abs=1e-6), alpha


def test_historical_var_and_es_agree_with_quantlibs_risk_statistics():
    """QuantLib 1.43's RiskStatistics as the oracle: valueAtRisk(p) is minus the smallest return whose cumulative
    weight reaches 1 - p ('inverted_cdf') and expectedShortfall(p) minus the mean of the returns strictly below it
    ('beyond'). 301 returns, so no n (1 - p) is a whole number: there QuantLib's unrounded 1 - p takes the next
    order statistic (the case test_one_minus_alpha_is_exact covers)."""
    ql = pytest.importorskip("QuantLib")
    r = np.round(sample(301, 4, df=5), 5)
    stats_ = ql.RiskStatistics()
    for value in r:
        stats_.add(float(value), 1.0)
    for alpha in (0.9, 0.95, 0.975, 0.99):
        var, es = risk.historical_var_es(r, alpha=alpha, method="inverted_cdf", es_rule="beyond", sign=LOSS)
        assert var == pytest.approx(stats_.valueAtRisk(alpha), abs=1e-15)
        assert es == pytest.approx(stats_.expectedShortfall(alpha), abs=1e-14)


# ------------------------------------------------------------------------------------------------ parametric


def test_normal_var_and_es_give_the_textbook_constants_and_es_integrates_var():
    """McNeil, Frey and Embrechts (2005) example 2.14 and the Basel FRTB comparison: for a standard normal loss
    VaR_0.99 = 2.3263 and ES_0.975 = 2.3378; ES_0.99 = 2.6652. ES is (1/(1 - alpha)) int_alpha^1 VaR_u du."""
    assert risk.normal_var_es(mu=0.0, sigma=1.0, alpha=0.99, sign=LOSS) == pytest.approx((2.326348, 2.665214),
                                                                                         abs=1e-6)
    assert risk.normal_var_es(mu=0.0, sigma=1.0, alpha=0.975, sign=LOSS)[1] == pytest.approx(2.337803, abs=1e-6)
    mu, sigma = 0.0005, 0.013
    for alpha in (0.95, 0.99):
        var, es = risk.normal_var_es(mu=mu, sigma=sigma, alpha=alpha, sign=LOSS)
        assert var == pytest.approx(-mu - sigma * ndtri(1 - alpha), rel=1e-14)
        integral = integrate.quad(lambda u: risk.normal_var_es(mu=mu, sigma=sigma, alpha=u, sign=LOSS)[0], alpha, 1)
        assert es == pytest.approx(integral[0] / (1 - alpha), rel=1e-8)
        assert risk.normal_var_es(mu=mu, sigma=sigma, alpha=alpha, sign=RETURN) == pytest.approx((-var, -es))
    assert risk.normal_var_es(mu=0.0, sigma=np.array([1.0, 2.0]), alpha=0.99, sign=LOSS)[0].shape == (2,)
    with pytest.raises(ValueError):
        risk.normal_var_es(mu=0.0, sigma=0.0, alpha=0.99, sign=LOSS)


def test_t_var_and_es_integrate_the_t_quantile_with_scale_or_std():
    """McNeil, Frey and Embrechts (2005) example 2.18: ES of a t with scale s is s g(t_q) (nu + t_q^2) / ((nu - 1)
    (1 - alpha)), checked against the integral of the t quantile; 'std' is the scale sqrt((nu - 2)/nu) sigma; nu large
    is the normal."""
    for nu, alpha in ((3.0, 0.99), (4.0, 0.975), (8.0, 0.95)):
        var, es = risk.t_var_es(mu=0.001, sigma=0.02, nu=nu, alpha=alpha, sigma_kind="scale", sign=LOSS)
        assert var == pytest.approx(-0.001 + 0.02 * stats.t.ppf(alpha, nu), rel=1e-12)
        integral = integrate.quad(lambda u: stats.t.ppf(u, nu), alpha, 1, limit=200)[0] / (1 - alpha)
        assert es == pytest.approx(-0.001 + 0.02 * integral, rel=1e-7)
        by_std = risk.t_var_es(mu=0.001, sigma=0.02, nu=nu, alpha=alpha, sigma_kind="std", sign=LOSS)
        scaled = risk.t_var_es(mu=0.001, sigma=0.02 * math.sqrt((nu - 2) / nu), nu=nu, alpha=alpha,
                               sigma_kind="scale", sign=LOSS)
        assert by_std == pytest.approx(scaled, rel=1e-14)
    assert risk.t_var_es(mu=0.0, sigma=1.0, nu=1e7, alpha=0.99, sigma_kind="std", sign=LOSS) == pytest.approx(
        risk.normal_var_es(mu=0.0, sigma=1.0, alpha=0.99, sign=LOSS), rel=1e-6)
    with pytest.raises(ValueError, match="nu"):
        risk.t_var_es(mu=0.0, sigma=1.0, nu=2.0, alpha=0.99, sigma_kind="std", sign=LOSS)


def test_cornish_fisher_approximates_an_exact_skewed_law_and_its_es_integrates_its_quantile():
    """Cornish and Fisher (1938): the expansion is the quantile of a law with the given cumulants. A standardised
    gamma(n) has skewness 2/sqrt(n) and excess kurtosis 6/n; at n = 400 the expansion meets its exact 1% and 5%
    quantiles within 5e-5 (the normal misses by 0.03 to 0.07, the S^2 and K terms move it by about 0.004), and the
    closed-form ES meets the gamma's exact tail mean, n F_{n+1}(q) / F_n(q). The ES is the integral of the expansion's
    quantile; S = K = 0 is the normal."""
    n = 400
    skew, kurt = 2 / math.sqrt(n), 6 / n
    for alpha in (0.95, 0.99):
        q = stats.gamma.ppf(1 - alpha, n)
        exact_q = (q - n) / math.sqrt(n)
        exact_es = (n * stats.gamma.cdf(q, n + 1) / stats.gamma.cdf(q, n) - n) / math.sqrt(n)
        var, es = risk.cornish_fisher_var_es(mu=0.0, sigma=1.0, skew=skew, excess_kurtosis=kurt, alpha=alpha,
                                             sign=RETURN)
        assert var == pytest.approx(exact_q, abs=5e-5)
        assert es == pytest.approx(exact_es, abs=2e-4)

        def quantile(u):
            z = ndtri(u)
            return z + (z * z - 1) * skew / 6 + (z ** 3 - 3 * z) * kurt / 24 - (2 * z ** 3 - 5 * z) * skew ** 2 / 36
        assert es == pytest.approx(integrate.quad(quantile, 0, 1 - alpha, limit=200)[0] / (1 - alpha), rel=1e-9)
    assert risk.cornish_fisher_var_es(mu=0.01, sigma=0.2, skew=0.0, excess_kurtosis=0.0, alpha=0.99, sign=LOSS) == \
        pytest.approx(risk.normal_var_es(mu=0.01, sigma=0.2, alpha=0.99, sign=LOSS), rel=1e-14)
    var, es = risk.cornish_fisher_var_es(mu=0.002, sigma=0.03, skew=-0.7, excess_kurtosis=3.0, alpha=0.99, sign=LOSS)
    base = risk.cornish_fisher_var_es(mu=0.0, sigma=1.0, skew=-0.7, excess_kurtosis=3.0, alpha=0.99, sign=LOSS)
    assert (var, es) == pytest.approx((-0.002 + 0.03 * base[0], -0.002 + 0.03 * base[1]), rel=1e-13)


# ------------------------------------------------------------------------------------------------ weighted and EWMA


def test_age_weighted_var_weighs_by_age_on_a_hand_case_and_is_historical_at_lam_one():
    """Boudoukh, Richardson and Whitelaw (1998); Hull, "Weighting of observations". Five returns, oldest first, lam
    0.5: the weights are 1, 2, 4, 8, 16 over 31, so the newest -4% carries the tail at 10% where the oldest -5% (1/31)
    does not; equal weights (lam = 1) give the plain historical rules exactly."""
    r = [-0.05, 0.01, -0.03, 0.02, -0.04]
    var, es = risk.age_weighted_var_es(r, lam=0.5, alpha=0.9, method="inverted_cdf", es_rule="exact", sign=LOSS)
    assert var == pytest.approx(0.04, abs=1e-15)
    assert es == pytest.approx((0.05 / 31 + (0.1 - 1 / 31) * 0.04) / 0.1, rel=1e-13)
    var, es = risk.age_weighted_var_es(r, lam=0.5, alpha=0.9, method="hazen", es_rule="at_or_beyond", sign=LOSS)
    lo, hi = 0.5 / 31, 1 / 31 + 8 / 31                             # each return at its weight's midpoint
    assert var == pytest.approx(0.05 - (0.1 - lo) / (hi - lo) * 0.01, rel=1e-13)
    assert es == pytest.approx(0.05, abs=1e-15)
    newest_first = r[::-1]
    assert risk.age_weighted_var_es(newest_first, lam=0.5, alpha=0.9, method="inverted_cdf", es_rule="exact",
                                    sign=LOSS)[0] == pytest.approx(0.05)
    x = sample(61, 5, df=4)
    for (age, plain), rule in zip((("inverted_cdf", "inverted_cdf"), ("hazen", "hazen")), ("beyond", "exact")):
        for alpha in (0.9, 0.95, 0.975):
            ours = risk.age_weighted_var_es(x, lam=1.0, alpha=alpha, method=age, es_rule=rule, sign=LOSS)
            assert ours == pytest.approx(risk.historical_var_es(x, alpha=alpha, method=plain, es_rule=rule,
                                                                sign=LOSS), rel=1e-12), (age, alpha)
    with pytest.raises(ValueError):
        risk.age_weighted_var_es(x, lam=1.2, alpha=0.9, method="hazen", es_rule="exact", sign=LOSS)
    with pytest.raises(ValueError):
        risk.age_weighted_var_es(x, lam=0.9, alpha=0.9, method="linear", es_rule="exact", sign=LOSS)


def test_age_weighted_var_and_es_agree_with_quantlibs_weighted_risk_statistics():
    """QuantLib 1.43's RiskStatistics with the age weights added per return: its weighted percentile is the
    'inverted_cdf' rule and its expected shortfall the 'beyond' rule."""
    ql = pytest.importorskip("QuantLib")
    r = np.round(sample(250, 6, df=4), 5)
    lam = 0.97
    weights = lam ** np.arange(r.size - 1, -1, -1.0)
    stats_ = ql.RiskStatistics()
    for value, weight in zip(r, weights):
        stats_.add(float(value), float(weight))
    for alpha in (0.9, 0.95, 0.99):
        var, es = risk.age_weighted_var_es(r, lam=lam, alpha=alpha, method="inverted_cdf", es_rule="beyond",
                                           sign=LOSS)
        assert var == pytest.approx(stats_.valueAtRisk(alpha), abs=1e-15)
        assert es == pytest.approx(stats_.expectedShortfall(alpha), rel=1e-12)


def test_ewma_var_is_the_normal_var_at_the_riskmetrics_variance_forecast():
    """RiskMetrics Technical Document 5.2.1: sigma^2 after the last return, unrolled, lam^n s + (1 - lam) sum_t
    lam^{n-1-t} r_t^2 with s the seed; then the zero-mean normal VaR and ES."""
    r = sample(40, 7)
    lam, n, t = 0.94, r.size, np.arange(r.size)
    for seed, s in (("first", r[0] ** 2), ("zero", 0.0), (2.5e-4, 2.5e-4)):
        variance = lam ** n * s + (1 - lam) * np.sum(lam ** (n - 1 - t) * r * r)
        assert risk.ewma_var_es(r, lam=lam, seed=seed, alpha=0.99, sign=LOSS) == pytest.approx(
            risk.normal_var_es(mu=0.0, sigma=math.sqrt(variance), alpha=0.99, sign=LOSS), rel=1e-12), seed


def test_kde_var_and_es_are_scipys_gaussian_kde_quantile_and_tail_mean():
    """scipy.stats.gaussian_kde as the oracle: its CDF (integrate_box_1d) at the VaR quantile is 1 - alpha, its
    kernel width is the bandwidth rule's, and the ES is its tail mean by quadrature."""
    r = sample(150, 8, df=5)
    for bandwidth, kde in (("scott", stats.gaussian_kde(r, bw_method="scott")),
                           ("silverman", stats.gaussian_kde(r, bw_method="silverman")),
                           (0.004, stats.gaussian_kde(r, bw_method=0.004 / np.std(r, ddof=1)))):
        for alpha in (0.95, 0.99):
            var, es = risk.kde_var_es(r, bandwidth=bandwidth, alpha=alpha, sign=RETURN)
            assert kde.integrate_box_1d(-np.inf, var) == pytest.approx(1 - alpha, abs=1e-12)
            tail = integrate.quad(lambda x: x * kde.pdf(x)[0], -0.5, var, points=[var - 0.05], limit=400)[0]
            assert es == pytest.approx(tail / (1 - alpha), rel=1e-7), (bandwidth, alpha)
    with pytest.raises(ValueError):
        risk.kde_var_es(r, bandwidth="isj", alpha=0.99, sign=LOSS)
    with pytest.raises(ValueError, match="constant"):
        risk.kde_var_es([0.01] * 10, bandwidth="scott", alpha=0.99, sign=LOSS)


# ------------------------------------------------------------------------------------------------ backtest


@pytest.mark.parametrize("p, n, low, high", [
    (0.01, 255, None, 7), (0.01, 510, 1, 11), (0.01, 1000, 4, 17),
    (0.025, 255, 2, 12), (0.025, 510, 6, 21), (0.025, 1000, 15, 36),
    (0.05, 255, 6, 21), (0.05, 510, 16, 36), (0.05, 1000, 37, 65)])
def test_kupiecs_nonrejection_regions_at_five_percent(p, n, low, high):
    """Kupiec (1995) as tabulated in Jorion, Value at Risk: the failure counts N the POF test accepts at 95%,
    low < N < high (the table leaves the lowest region open at T = 255, p = 1%)."""
    accepted = [x for x in range(high + 5) if risk.var_backtest(np.r_[np.ones(x), np.zeros(n - x)],
                                                                alpha=1 - p)["p_pof"] > 0.05]
    assert max(accepted) == high - 1
    if low is not None:
        assert min(accepted) == low + 1


def test_the_backtest_likelihood_ratios_are_their_log_likelihoods_and_basel_zones_follow_the_binomial():
    """Christoffersen (1998): LR_ind = 2 (Markov log-likelihood - i.i.d. log-likelihood) over the n - 1 transitions,
    summed observation by observation; Kupiec's LR_pof the same for the binomial. The Basel Committee (1996) table for
    250 days at 99%: P(X <= 4) = 89.22% green, P(X <= 5) = 95.88% and P(X <= 9) = 99.97% yellow, 10 red."""
    rng = np.random.default_rng(9)
    hits = (rng.uniform(size=500) < 0.03).astype(int)
    hits[100:104] = 1
    out = risk.var_backtest(hits, alpha=0.99)
    a, b = hits[:-1], hits[1:]
    p01, p11 = np.sum((a == 0) & (b == 1)) / np.sum(a == 0), np.sum((a == 1) & (b == 1)) / np.sum(a == 1)
    pi = np.mean(b)
    markov = np.sum(np.where(a == 0, np.where(b == 1, np.log(p01), np.log(1 - p01)),
                             np.where(b == 1, np.log(p11), np.log(1 - p11))))
    iid = np.sum(np.where(b == 1, np.log(pi), np.log(1 - pi)))
    assert out["lr_ind"] == pytest.approx(2 * (markov - iid), rel=1e-12)
    x, n = hits.sum(), hits.size
    pof = 2 * (stats.binom.logpmf(x, n, x / n) - stats.binom.logpmf(x, n, 0.01))
    assert out["lr_pof"] == pytest.approx(pof, rel=1e-12)
    assert out["lr_cc"] == pytest.approx(out["lr_pof"] + out["lr_ind"], rel=1e-15)
    assert out["p_cc"] == pytest.approx(stats.chi2.sf(out["lr_cc"], 2), rel=1e-12)
    assert (out["n00"] + out["n01"] + out["n10"] + out["n11"], out["exceptions"]) == (499, x)
    zones = [risk.var_backtest(np.r_[np.ones(k), np.zeros(250 - k)], alpha=0.99)["basel_zone"] for k in range(12)]
    assert zones == ["green"] * 5 + ["yellow"] * 5 + ["red"] * 2
    assert [round(100 * stats.binom.cdf(k, 250, 0.01), 2) for k in (4, 5, 9)] == [89.22, 95.88, 99.97]
    clean = risk.var_backtest(np.zeros(250, dtype=bool), alpha=0.99)
    assert clean["lr_ind"] == 0.0 and clean["lr_pof"] == pytest.approx(-2 * 250 * math.log(0.99))
    with pytest.raises(ValueError):
        risk.var_backtest([0, 2, 1], alpha=0.99)


# ------------------------------------------------------------------------------------------------ tails


def test_hill_is_unbiased_on_exact_pareto_samples_by_renyis_representation():
    """For X = U^(-xi) (an exact Pareto tail) the Hill statistic over the k largest with threshold X_(k+1) is xi
    times a Gamma(k, 1) / k: mean xi, standard deviation xi / sqrt(k), whatever n (Embrechts, Kluppelberg and Mikosch
    6.4.2). 2,000 samples of 300 with k = 60, at four standard errors; and the definition on a hand sample."""
    xi, k = 0.4, 60
    draws = np.random.default_rng(10).uniform(size=(2000, 300)) ** -xi
    estimates = np.array([risk.hill_estimator(row, k=k)["xi"] for row in draws])
    assert abs(estimates.mean() - xi) < 4 * xi / math.sqrt(k * len(estimates))
    assert estimates.std() == pytest.approx(xi / math.sqrt(k), rel=0.05)
    fit = risk.hill_estimator([5.0, 1.0, 3.0, 2.0, 8.0, 0.5], k=3)
    assert fit["xi"] == pytest.approx((math.log(8) + math.log(5) + math.log(3)) / 3 - math.log(2), rel=1e-14)
    assert (fit["alpha"], fit["threshold"], fit["se_xi"], fit["se_alpha"]) == pytest.approx(
        (1 / fit["xi"], 2.0, fit["xi"] / 3 ** .5, 1 / (fit["xi"] * 3 ** .5)))
    for bad in (0, 6, 2.0):
        with pytest.raises(ValueError):
            risk.hill_estimator([5.0, 1.0, 3.0, 2.0, 8.0, 0.5], k=bad)
    with pytest.raises(ValueError, match="positive"):
        risk.hill_estimator([5.0, 1.0, -3.0, -2.0], k=2)


def test_gpd_fit_solves_the_score_matches_scipy_and_its_errors_invert_the_information():
    """Smith (1987): the fit makes both score components vanish and agrees with scipy.stats.genpareto.fit (loc
    fixed at 0); the standard errors are the inverse expected information, here computed by quadrature of the score's
    outer product rather than from the closed form. Only losses strictly above the threshold count."""
    for xi, beta in ((0.3, 1.5), (-0.2, 0.8)):
        y = stats.genpareto.rvs(xi, scale=beta, size=1500, random_state=11)
        losses = np.r_[y + 2.0, np.full(5, 2.0), np.linspace(-1.0, 1.9, 40)]
        fit = risk.gpd_fit(losses, threshold=2.0)
        assert fit["exceedances"] == 1500 and fit["threshold"] == 2.0
        assert tails._gpd_score([fit["xi"], fit["beta"]], y) == pytest.approx([0.0, 0.0], abs=1e-8)
        c, _, scale = stats.genpareto.fit(y, floc=0)
        assert (fit["xi"], fit["beta"]) == pytest.approx((c, scale), rel=2e-4)
        assert fit["loglik"] == pytest.approx(np.sum(stats.genpareto.logpdf(y, fit["xi"], scale=fit["beta"])),
                                              rel=1e-12)
        f_xi, f_beta = fit["xi"], fit["beta"]
        top = np.inf if f_xi > 0 else -f_beta / f_xi

        def score(t):
            z = 1 + f_xi * t / f_beta
            return (np.log(z) / f_xi ** 2 - (1 + 1 / f_xi) * t / (f_beta * z),
                    -1 / f_beta + (1 + 1 / f_xi) * f_xi * t / (f_beta ** 2 * z))
        info = np.array([[integrate.quad(lambda t: score(t)[i] * score(t)[j] * stats.genpareto.pdf(
            t, f_xi, scale=f_beta), 0, top, limit=200)[0] for j in range(2)] for i in range(2)])
        cov = np.linalg.inv(info) / 1500
        assert (fit["se_xi"], fit["se_beta"], fit["cov_xi_beta"]) == pytest.approx(
            (math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), cov[0, 1]), rel=1e-6)
    exponential = risk.gpd_fit(stats.expon.rvs(scale=2.0, size=2000, random_state=12), threshold=0.0)
    assert abs(exponential["xi"]) < 4 * exponential["se_xi"] and exponential["beta"] == pytest.approx(2.0, rel=0.1)
    with pytest.raises(ValueError, match="above the threshold"):
        risk.gpd_fit([1.0, 2.0, 3.0], threshold=2.5)


# ------------------------------------------------------------------------------------------------ copulas


def test_copula_samples_follow_their_draw_order_and_have_uniform_margins_and_the_elliptical_tau():
    """McNeil, Frey and Embrechts algorithms 5.9 and 5.10, rebuilt from the same generator stream: normals first,
    then one chi-square per row. Margins uniform (Kolmogorov-Smirnov); Kendall's tau of any elliptical copula
    (2/pi) arcsin(rho) (Lindskog, McNeil and Schmock 2003), within four standard errors of tau-hat."""
    corr = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    lower = np.linalg.cholesky(corr)
    rng, twin = np.random.default_rng(13), np.random.default_rng(13)
    u = qf.gaussian_copula_sample(corr, n=4000, rng=rng)
    assert u == pytest.approx(ndtr(twin.standard_normal((4000, 3)) @ lower.T), abs=0)
    t = qf.t_copula_sample(corr, nu=4.0, n=4000, rng=rng)
    z, w = twin.standard_normal((4000, 3)) @ lower.T, twin.chisquare(4.0, 4000)
    assert t == pytest.approx(stats.t.cdf(z / np.sqrt(w / 4.0)[:, None], 4.0), abs=1e-12)
    for draws in (u, t):
        for j in range(3):
            assert stats.kstest(draws[:, j], "uniform").pvalue > 1e-3
        for i, j in ((0, 1), (0, 2), (1, 2)):
            tau = stats.kendalltau(draws[:, i], draws[:, j]).statistic
            assert abs(tau - 2 / math.pi * math.asin(corr[i, j])) < 4 * math.sqrt(4 / 9 / 4000)
    lower_tail = [np.mean((draws[:, 0] < 0.01) & (draws[:, 1] < 0.01)) / 0.01 for draws in (u, t)]
    assert lower_tail[1] > lower_tail[0]
    for bad in (np.array([[1.0, 1.2], [1.2, 1.0]]), np.array([[1.0, 0.5], [0.4, 1.0]]), np.eye(1)):
        with pytest.raises(ValueError):
            qf.gaussian_copula_sample(bad, n=10, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="Generator"):
        qf.t_copula_sample(corr, nu=4.0, n=10, rng=4.2)


def test_copula_fits_maximise_qfs_densities():
    """Genest, Ghoudi and Rivest (1995), maximum pseudo-likelihood on qf's densities: the Gaussian fit's score is 0
    and it recovers rho within four standard errors ((1 - rho^2)/sqrt(n)); the t fit by 'tau' takes rho =
    sin(pi tau-hat / 2) exactly and maximises over nu; 'mle' maximises over both, so its likelihood is at least the
    'tau' fit's; AIC and BIC by their definitions."""
    uv = qf.gaussian_copula_sample(np.array([[1.0, -0.45], [-0.45, 1.0]]), n=1500, rng=np.random.default_rng(14))
    fit = qf.gaussian_copula_fit(uv[:, 0], uv[:, 1])
    ll = lambda rho: np.sum(qf.gaussian_copula_logpdf(uv[:, 0], uv[:, 1], rho))  # noqa: E731
    assert (ll(fit["rho"] + 1e-5) - ll(fit["rho"] - 1e-5)) / 2e-5 == pytest.approx(0.0, abs=1e-3)
    assert abs(fit["rho"] + 0.45) < 4 * (1 - 0.45 ** 2) / math.sqrt(1500)
    assert (fit["aic"], fit["bic"]) == pytest.approx((2 - 2 * ll(fit["rho"]), math.log(1500) - 2 * ll(fit["rho"])))
    uv = qf.t_copula_sample(np.array([[1.0, 0.5], [0.5, 1.0]]), nu=4.0, n=500, rng=np.random.default_rng(15))
    by_tau = qf.t_copula_fit(uv[:, 0], uv[:, 1], method="tau")
    assert by_tau["rho"] == pytest.approx(math.sin(math.pi * stats.kendalltau(uv[:, 0], uv[:, 1]).statistic / 2))
    joint = qf.t_copula_fit(uv[:, 0], uv[:, 1], method="mle")
    tll = lambda rho, nu: np.sum(qf.t_copula_logpdf(uv[:, 0], uv[:, 1], rho, nu))  # noqa: E731
    assert joint["loglik"] >= by_tau["loglik"] - 1e-9
    for fit_ in (by_tau, joint):
        nu = fit_["nu"]
        assert (tll(fit_["rho"], nu * 1.001) - tll(fit_["rho"], nu / 1.001)) / (2 * 0.001) == pytest.approx(0, abs=0.01)
        assert fit_["aic"] == pytest.approx(4 - 2 * tll(fit_["rho"], nu)) and 1.5 < nu < 12
    d = 1e-5
    assert (tll(joint["rho"] + d, joint["nu"]) - tll(joint["rho"] - d, joint["nu"])) / (2 * d) == pytest.approx(
        0, abs=0.01)
    with pytest.raises(ValueError):
        qf.t_copula_fit(uv[:, 0], uv[:, 1], method="ifm")


# ------------------------------------------------------------------------------------------------ credit


def test_vasicek_cdf_inverts_the_quantile_comes_from_the_factor_and_has_mean_pd():
    """Vasicek (2002): L(Z) = Phi((Phi^-1(pd) - sqrt(rho) Z) / sqrt(1 - rho)) is decreasing in the factor, so P(L <=
    x) = P(Z >= L^-1(x)) = Phi(-L^-1(x)); E[L] = int_0^1 (1 - F(x)) dx = pd; the quantile is the CDF's inverse."""
    pd, rho = 0.02, 0.15
    xs = np.array([0.001, 0.01, 0.02, 0.05, 0.2, 0.6])
    z_star = (ndtri(pd) - math.sqrt(1 - rho) * ndtri(xs)) / math.sqrt(rho)
    assert qf.vasicek_loss_cdf(xs, pd=pd, rho=rho) == pytest.approx(ndtr(-z_star), rel=1e-12)
    qs = np.array([0.5, 0.9, 0.99, 0.999])
    assert qf.vasicek_loss_cdf(qf.vasicek_loss_quantile(qs, pd=pd, rho=rho), pd=pd, rho=rho) == pytest.approx(qs)
    mean = integrate.quad(lambda x: 1 - qf.vasicek_loss_cdf(x, pd=pd, rho=rho), 0, 1, limit=200, points=[pd])[0]
    assert mean == pytest.approx(pd, rel=1e-7)
    assert qf.vasicek_loss_cdf(0.0, pd=pd, rho=rho) == 0.0 and qf.vasicek_loss_cdf(1.0, pd=pd, rho=rho) == 1.0
    for bad in (dict(pd=0.0, rho=0.1), dict(pd=0.02, rho=1.0)):
        with pytest.raises(ValueError):
            qf.vasicek_loss_quantile(0.99, **bad)


@pytest.mark.parametrize("pd, weight", [
    (0.0003, 14.44), (0.0005, 19.65), (0.001, 29.65), (0.0025, 49.47), (0.004, 62.72), (0.005, 69.61),
    (0.0075, 82.78), (0.01, 92.32), (0.013, 100.95), (0.015, 105.59), (0.02, 114.86), (0.025, 122.16),
    (0.03, 128.44), (0.04, 139.58), (0.05, 149.86), (0.06, 159.61), (0.1, 193.09), (0.15, 221.54),
    (0.2, 238.23)])
def test_gordy_capital_gives_basels_illustrative_corporate_risk_weights(pd, weight):
    """Basel II (2006), Annex 5, illustrative IRB risk weights for corporate exposures, LGD 45%, M 2.5 years: RW =
    12.5 K with Basel's corporate correlation 0.12 f + 0.24 (1 - f), f = (1 - e^(-50 PD)) / (1 - e^(-50)), q =
    0.999, the expected loss subtracted (to the table's two decimals, a last-digit rounding allowed)."""
    f = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
    k = qf.gordy_capital(pd=pd, lgd=0.45, rho=0.12 * f + 0.24 * (1 - f), q=0.999, expected_loss="subtract",
                           maturity=2.5)
    assert 100 * 12.5 * k == pytest.approx(weight, abs=0.0151)


def test_gordy_capital_is_lgd_times_the_worst_case_rate_less_pd_and_its_maturity_factor():
    """Gordy (2003) and BCBS (2005): without the maturity adjustment K = LGD (WCDR - PD); 'include' adds LGD PD back;
    the adjustment is (1 + (M - 2.5) b)/(1 - 1.5 b) with b = (0.11852 - 0.05478 ln PD)^2."""
    pd, lgd, rho = 0.008, 0.4, 0.18
    wcdr = qf.vasicek_loss_quantile(0.999, pd=pd, rho=rho)
    bare = qf.gordy_capital(pd=pd, lgd=lgd, rho=rho, q=0.999, expected_loss="subtract", maturity=None)
    assert bare == pytest.approx(lgd * (wcdr - pd), rel=1e-14)
    assert qf.gordy_capital(pd=pd, lgd=lgd, rho=rho, q=0.999, expected_loss="include", maturity=None) == \
        pytest.approx(lgd * wcdr, rel=1e-14)
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    assert qf.gordy_capital(pd=pd, lgd=lgd, rho=rho, q=0.999, expected_loss="subtract", maturity=4.0) == \
        pytest.approx(bare * (1 + 1.5 * b) / (1 - 1.5 * b), rel=1e-14)
    with pytest.raises(ValueError):
        qf.gordy_capital(pd=pd, lgd=lgd, rho=rho, q=0.999, expected_loss="net", maturity=None)
    for bad in (45.0, -0.1):
        with pytest.raises(ValueError, match="lgd"):
            qf.gordy_capital(pd=pd, lgd=bad, rho=rho, q=0.999, expected_loss="subtract", maturity=None)


def test_euler_contributions_add_up_and_are_the_marginal_risks():
    """Euler's theorem for a homogeneous risk measure (Tasche 2000): sum_i w_i d sigma / d w_i = sigma, each term by
    a central difference; scenario ES contributions are the tail means of each position, sum to the ES, scale with
    the losses, and the VaR on losses at alpha is minus historical simulation's on returns at 1 - alpha ('higher' on
    losses being 'lower' on returns)."""
    rng = np.random.default_rng(16)
    a = rng.normal(size=(4, 4))
    cov, w = a @ a.T / 100 + np.eye(4) * 0.001, np.array([0.4, -0.1, 0.5, 0.2])
    rc = qf.euler_vol_contributions(w, cov=cov)
    sigma = lambda v: math.sqrt(v @ cov @ v)  # noqa: E731
    assert rc.sum() == pytest.approx(sigma(w), rel=1e-14)
    h = 1e-6
    marginal = [(sigma(w + h * e) - sigma(w - h * e)) / (2 * h) for e in np.eye(4)]
    assert rc == pytest.approx(w * np.array(marginal), rel=1e-7)
    losses = rng.lognormal(size=(5000, 3)) * [1.0, 2.0, 0.5]
    out = qf.euler_es_contributions(losses, alpha=0.99, method="higher", es_rule="at_or_beyond")
    total = losses.sum(axis=1)
    tail = total >= out["var"]
    assert out["contributions"] == pytest.approx(losses[tail].mean(axis=0), rel=1e-14)
    assert out["contributions"].sum() == pytest.approx(out["es"], rel=1e-13) and out["tail_scenarios"] == tail.sum()
    var, es = risk.historical_var_es(-total, alpha=0.99, method="lower", es_rule="at_or_beyond", sign=LOSS)
    assert (out["var"], out["es"]) == pytest.approx((var, es), rel=1e-14)
    doubled = qf.euler_es_contributions(2 * losses, alpha=0.99, method="higher", es_rule="beyond")
    beyond = qf.euler_es_contributions(losses, alpha=0.99, method="higher", es_rule="beyond")
    assert doubled["contributions"] == pytest.approx(2 * beyond["contributions"], rel=1e-14)
    assert beyond["tail_scenarios"] == out["tail_scenarios"] - 1
    with pytest.raises(ValueError):
        qf.euler_es_contributions(losses, alpha=0.99, method="higher", es_rule="exact")


# ------------------------------------------------------------------------------------------------ migrations


def test_the_cohort_matrix_counts_transitions_and_pools_periods():
    """Jafry and Schuermann (2004), the cohort method: P_ij = N_ij / N_i, counts pooled across periods; the default
    row absorbing; a state no obligor started in is NaN; labels outside the states refused."""
    start = ["A", "A", "A", "B", "B", "B", "B", "C", "A", "B"]
    end = ["A", "A", "B", "B", "A", "C", "D", "D", "A", "B"]
    matrix, counts = qf.cohort_transition_matrix(start, end, states=["A", "B", "C", "D"], absorbing=["D"])
    assert counts.tolist() == [[3, 1, 0, 0], [1, 2, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    assert matrix[:3].tolist() == [[0.75, 0.25, 0, 0], [0.2, 0.4, 0.2, 0.2], [0, 0, 0, 1]]
    assert matrix[3].tolist() == [0, 0, 0, 1]
    _, first = qf.cohort_transition_matrix(start[:5], end[:5], states=["A", "B", "C", "D"], absorbing=["D"])
    _, second = qf.cohort_transition_matrix(start[5:], end[5:], states=["A", "B", "C", "D"], absorbing=["D"])
    assert (first + second).tolist() == counts.tolist()
    unseen, _ = qf.cohort_transition_matrix(["A", "B"], ["B", "A"], states=["A", "B", "C"], absorbing=[])
    assert np.isnan(unseen[2]).all() and unseen[:2].tolist() == [[0, 1, 0], [1, 0, 0]]
    with pytest.raises(ValueError, match="NR"):
        qf.cohort_transition_matrix(["A", "NR"], ["A", "A"], states=["A", "B"], absorbing=[])
    with pytest.raises(ValueError):
        qf.cohort_transition_matrix(["A"], ["A"], states=["A", "B"], absorbing=["D"])


def test_the_generator_is_the_matrix_log_and_its_fixes_follow_israel_rosenthal_and_wei():
    """Israel, Rosenthal and Wei (2001): Q = logm(P) / dt with expm(Q dt) = P; for a valid generator every fix leaves
    it alone. A matrix with a zero one-step default rate from a rating that migrates to one that defaults has a
    negative log entry: 'diagonal' (DA) zeroes it and moves it to the diagonal, 'weighted' (WA) zeroes it and takes its
    mass from the row's non-zero entries by their size; both give rows summing to 0 and non-negative rates."""
    true = np.array([[-0.12, 0.10, 0.02], [0.05, -0.15, 0.10], [0.0, 0.0, 0.0]])
    p = linalg.expm(true)
    for fix in ("none", "diagonal", "weighted"):
        assert qf.generator_from_transition(p, dt=1.0, fix=fix) == pytest.approx(true, abs=1e-12), fix
    assert qf.generator_from_transition(linalg.expm(true * 0.5), dt=0.5, fix="none") == pytest.approx(true,
                                                                                                    abs=1e-12)
    p = np.array([[0.90, 0.10, 0.00], [0.05, 0.85, 0.10], [0.0, 0.0, 1.0]])
    log = qf.generator_from_transition(p, dt=1.0, fix="none")
    assert linalg.expm(log) == pytest.approx(p, abs=1e-12) and log[0, 2] < 0
    da = qf.generator_from_transition(p, dt=1.0, fix="diagonal")
    off = ~np.eye(3, dtype=bool)
    assert da[off] == pytest.approx(np.maximum(log, 0)[off], abs=1e-15)
    assert np.diag(da) == pytest.approx(np.diag(log) + np.sum(np.where(off, np.minimum(log, 0), 0), axis=1))
    wa = qf.generator_from_transition(p, dt=1.0, fix="weighted")
    for q in (da, wa):
        assert q.sum(axis=1) == pytest.approx(np.zeros(3), abs=1e-15) and np.all(q[off] >= 0)
    b, g = -log[0, 2], -log[0, 0] + log[0, 1]
    assert wa[0] == pytest.approx([log[0, 0] * (1 + b / g), log[0, 1] * (1 - b / g), 0.0], rel=1e-13)
    assert wa[1] == pytest.approx(log[1], rel=1e-13) and da[1] == pytest.approx(log[1], rel=1e-13)
    for bad in (np.array([[0.5, 0.6], [0.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])):
        with pytest.raises(ValueError):
            qf.generator_from_transition(bad, dt=1.0, fix="diagonal")


# ------------------------------------------------------------------------------------------------ factor copula

FACTOR = dict(pd=[0.01, 0.03, 0.05, 0.02], lgd=[0.4, 0.6, 0.5, 0.45], exposure=[100.0, 40.0, 70.0, 20.0],
              loading=[0.5, 0.3, 0.6, 0.4], factor_corr=[[1.0, 0.6], [0.6, 1.0]], factor_index=[0, 0, 1, 1])


@pytest.mark.parametrize("copula, nu", [("gaussian", None), ("t", 4.0)])
def test_factor_copula_defaults_keep_each_pd_the_mean_loss_and_the_pairs_joint_default(copula, nu):
    """CreditMetrics (1997) and Li (2000): each obligor defaults with its pd under either copula, so the mean loss is
    sum EAD LGD PD; two obligors default together with the bivariate law's probability at their thresholds,
    correlation a_i a_j corr(k_i, k_j) (normal, or Student t with nu); each within four standard errors."""
    n = 200_000
    found = qf.factor_copula_losses(**FACTOR, copula=copula, nu=nu, n=n, rng=np.random.default_rng(11))
    d = found["defaults"]
    pd = np.array(FACTOR["pd"])
    assert np.all(np.abs(d.mean(axis=0) - pd) < 4 * np.sqrt(pd * (1 - pd) / n))
    expected = float(np.sum(pd * np.array(FACTOR["lgd"]) * np.array(FACTOR["exposure"])))
    portfolio = found["portfolio_losses"]
    assert abs(portfolio.mean() - expected) < 4 * portfolio.std(ddof=1) / math.sqrt(n)
    assert np.array_equal(found["obligor_losses"].sum(axis=1), portfolio)
    law = ((lambda cov: stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)) if copula == "gaussian"
           else lambda cov: stats.multivariate_t(loc=[0.0, 0.0], shape=cov, df=nu))
    quantile = ndtri if copula == "gaussian" else (lambda p: stats.t.ppf(p, nu))
    for i, j in ((0, 1), (1, 2), (2, 3)):
        rho = FACTOR["loading"][i] * FACTOR["loading"][j] * FACTOR["factor_corr"][FACTOR["factor_index"][i]][
            FACTOR["factor_index"][j]]
        joint = float(law([[1.0, rho], [rho, 1.0]]).cdf([quantile(pd[i]), quantile(pd[j])]))
        both = np.mean(d[:, i] & d[:, j])
        assert abs(both - joint) < 4 * math.sqrt(joint * (1 - joint) / n) + 1e-5, (i, j, both, joint)


def test_the_t_copula_clusters_defaults_beyond_the_gaussians_at_one_pd():
    """Same pds, loadings and draws of the factors and the e_i: the t's chi-square divides them all at once, and
    every pair's joint default probability rises (Frey, McNeil and Nyfeler (2001), Risk 14(11))."""
    common = dict(FACTOR, n=100_000)
    normal = qf.factor_copula_losses(**common, copula="gaussian", nu=None, rng=np.random.default_rng(12))["defaults"]
    heavy = qf.factor_copula_losses(**common, copula="t", nu=3.0, rng=np.random.default_rng(12))["defaults"]
    for i, j in ((0, 1), (0, 2), (2, 3)):
        assert np.mean(heavy[:, i] & heavy[:, j]) > 1.5 * np.mean(normal[:, i] & normal[:, j])


def test_a_large_homogeneous_pool_reaches_vasiceks_conditional_default_rate_and_its_quantiles():
    """Vasicek (2002): given the factor Z, the default rate of a large pool is Phi((Phi^-1(pd) - a Z) / sqrt(1 -
    a^2)); the factor is the rng's first draw (the card's order), so each scenario's rate is held to it within five
    binomial errors, and the loss fraction's CDF at Vasicek's quantiles to its level."""
    m, n, pd, rho = 1000, 2000, 0.02, 0.15
    found = qf.factor_copula_losses(pd=np.full(m, pd), lgd=0.5, exposure=1.0 / m, loading=math.sqrt(rho),
                                      factor_corr=[[1.0]], factor_index=np.zeros(m, dtype=int), copula="gaussian",
                                      nu=None, n=n, rng=np.random.default_rng(13))
    z = np.random.default_rng(13).standard_normal((n, 1))[:, 0]
    conditional = ndtr((ndtri(pd) - math.sqrt(rho) * z) / math.sqrt(1 - rho))
    rate = found["defaults"].mean(axis=1)
    assert np.all(np.abs(rate - conditional) < 5 * np.sqrt(conditional * (1 - conditional) / m) + 1.0 / m)
    for q in (0.5, 0.9, 0.99):
        level = 0.5 * qf.vasicek_loss_quantile(q, pd=pd, rho=rho)
        assert abs(np.mean(found["portfolio_losses"] <= level) - q) < 0.035, q


def test_factor_copula_losses_refuse_what_they_cannot_simulate():
    rng = np.random.default_rng(1)
    for bad, match in ((dict(copula="clayton", nu=None), "copula"), (dict(copula="gaussian", nu=4.0), "None"),
                       (dict(copula="t", nu=None), "nu"), (dict(copula="gaussian", nu=None, factor_index=[0, 0, 1, 2]),
                                                           "factor_index"),
                       (dict(copula="gaussian", nu=None, factor_corr=[[1.0, 1.2], [1.2, 1.0]]), "positive definite"),
                       (dict(copula="gaussian", nu=None, lgd=45.0), "lgd")):
        with pytest.raises((ValueError, TypeError), match=match):
            qf.factor_copula_losses(**{**FACTOR, **bad, "n": 10, "rng": rng})


# ------------------------------------------------------------------------------------------------ tail dependence


def test_tail_dependence_near_zero_when_independent_one_when_comonotonic_and_zero_when_counter():
    """Joe (1997), 2.1.10: at a finite q independence gives P(V > q | U > q) = 1 - q (and q in the lower tail),
    within three binomial errors of the share; a comonotonic pair gives 1, a countermonotonic one 0."""
    rng = np.random.default_rng(61)
    u, v = rng.random(200_000), rng.random(200_000)
    for tail, q, expected in (("upper", 0.99, 0.01), ("lower", 0.02, 0.02)):
        found = qf.tail_dependence(u, v, q=q, tail=tail)
        beyond = 200_000 * (1 - q if tail == "upper" else q)
        assert abs(found - expected) < 3 * math.sqrt(expected * (1 - expected) / beyond), tail
    assert qf.tail_dependence(u, u, q=0.95, tail="upper") == 1.0 == qf.tail_dependence(u, u, q=0.05, tail="lower")
    assert qf.tail_dependence(u, 1 - u, q=0.95, tail="upper") == 0.0
    assert math.isnan(qf.tail_dependence([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], q=0.9, tail="upper"))
    u4, v4 = [0.2, 0.4, 0.6, 0.8], [0.4, 0.2, 0.8, 0.6]          # written out: the lower tail holds its threshold
    assert qf.tail_dependence(u4, v4, q=0.4, tail="lower") == 1.0 and qf.tail_dependence(u4, v4, q=0.6,
                                                                                            tail="upper") == 0.0


def test_tail_dependence_estimates_a_copulas_conditional_tail_probability_at_its_threshold():
    """The estimate's target at a finite q is a closed form: Clayton's C(q, q) / q = (2 q^-theta - 1)^(-1/theta) / q
    in the lower tail, sampled by conditional inversion (Nelsen (2006), 4.3), tending to 2^(-1/theta); a Gaussian's
    (1 - 2q + C(q, q)) / (1 - q) in the upper tail, C its bivariate normal; each within four binomial errors."""
    rng = np.random.default_rng(62)
    n, theta = 400_000, 3.0
    u, w = rng.random(n), rng.random(n)
    v = (u ** -theta * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    for q in (0.05, 0.01):
        target = (2 * q ** -theta - 1) ** (-1 / theta) / q
        assert abs(qf.tail_dependence(u, v, q=q, tail="lower") - target) < 4 * math.sqrt(target * (1 - target) /
                                                                                            (n * q)), q
    assert (2 * 1e-6 ** -theta - 1) ** (-1 / theta) / 1e-6 == pytest.approx(2 ** (-1 / theta), rel=1e-6)
    rho = 0.6
    g, h = qf.gaussian_copula_sample(np.array([[1.0, rho], [rho, 1.0]]), n=n, rng=rng).T
    q = 0.95
    joint = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf([ndtri(q), ndtri(q)])
    target = (1 - 2 * q + joint) / (1 - q)
    assert abs(qf.tail_dependence(g, h, q=q, tail="upper") - target) < 4 * math.sqrt(target * (1 - target) /
                                                                                       (n * (1 - q)))
    with pytest.raises(ValueError, match="tail"):
        qf.tail_dependence(g, h, q=q, tail="both")
    with pytest.raises(ValueError, match="q must"):
        qf.tail_dependence(g, h, q=95, tail="upper")


# ------------------------------------------------------------------------------------------------ Archimedean samples

ARCHIMEDEAN_CDF = {   # Nelsen (2006), table 4.1: families (4.2.1), (4.2.4) and (4.2.5)
    "clayton": lambda u, v, th: (u ** -th + v ** -th - 1) ** (-1 / th),
    "gumbel": lambda u, v, th: np.exp(-((-np.log(u)) ** th + (-np.log(v)) ** th) ** (1 / th)),
    "frank": lambda u, v, th: -np.log1p(np.expm1(-th * u) * np.expm1(-th * v) / np.expm1(-th)) / th,
}
ARCHIMEDEAN_TAU = {"clayton": lambda th: th / (th + 2), "gumbel": lambda th: 1 - 1 / th,
                   "frank": lambda th: 1 - 4 / th * (1 - integrate.quad(lambda x: x / math.expm1(x), 0, th)[0] / th)}


@pytest.mark.parametrize("family, thetas", [("clayton", (0.8, 4.0)), ("gumbel", (1.4, 3.0)), ("frank", (2.0, 9.0))])
def test_marshall_olkin_samples_have_their_familys_kendalls_tau_and_copula(family, thetas):
    """McNeil, Frey and Embrechts (2005), algorithm 5.48: at two thetas each, Kendall's tau of the sample against the
    family's closed form (Clayton theta / (theta + 2), Gumbel 1 - 1/theta, Frank's Debye form) and the sample's joint
    CDF at three points against C(u, v), within four standard errors; uniform margins."""
    n = 20_000
    for theta in thetas:
        x = qf.archimedean_copula_sample(family, theta=theta, dim=2, n=n, rng=np.random.default_rng(71))
        assert x.shape == (n, 2) and np.all((x > 0) & (x < 1))
        assert abs(stats.kendalltau(x[:, 0], x[:, 1])[0] - ARCHIMEDEAN_TAU[family](theta)) < 0.02, theta
        for u, v in ((0.2, 0.3), (0.5, 0.5), (0.9, 0.7)):
            c = float(ARCHIMEDEAN_CDF[family](u, v, theta))
            assert abs(np.mean((x[:, 0] <= u) & (x[:, 1] <= v)) - c) < 4 * math.sqrt(c * (1 - c) / n), (theta, u, v)
        assert np.all(np.abs(x.mean(axis=0) - 0.5) < 4 * math.sqrt(1 / 12 / n))


def test_marshall_olkin_samples_are_exchangeable_in_three_dimensions_and_frank_inverts_from_tau():
    x = qf.archimedean_copula_sample("gumbel", theta=2.5, dim=3, n=8_000, rng=np.random.default_rng(72))
    taus = [stats.kendalltau(x[:, i], x[:, j])[0] for i, j in ((0, 1), (0, 2), (1, 2))]
    assert all(abs(tau - 0.6) < 0.03 for tau in taus)
    theta = qf.copula_param_from_tau("frank", 0.4)
    assert ARCHIMEDEAN_TAU["frank"](theta) == pytest.approx(0.4, abs=1e-10)
    flat = qf.archimedean_copula_sample("gumbel", theta=1.0, dim=2, n=20_000, rng=np.random.default_rng(73))
    assert abs(stats.kendalltau(flat[:, 0], flat[:, 1])[0]) < 0.02                 # theta 1 is independence
    for family, theta, match in (("clayton", 0.0, "positive"), ("gumbel", 0.5, "at least 1"), ("frank", -2.0,
                                                                                               "positive"),
                                 ("joe", 2.0, "family")):
        with pytest.raises(ValueError, match=match):
            qf.archimedean_copula_sample(family, theta=theta, dim=2, n=10, rng=np.random.default_rng(1))


def test_euler_es_contributions_never_exceed_the_stand_alone_es_and_es_is_subadditive():
    """Lane lib6's lines on euler_es_contributions and historical_var_es: over the k tail scenarios, a position's
    mean loss is at most the mean of its own k worst; the portfolio's ES is at most the positions' ES summed, the gap
    the diversification benefit (Acerbi and Tasche 2002)."""
    rng = np.random.default_rng(31)
    losses = rng.standard_t(4, size=(4000, 3)) * [1.0, 2.0, 0.5] + rng.standard_normal((4000, 1))
    found = qf.euler_es_contributions(losses, alpha=0.975, method="higher", es_rule="at_or_beyond")
    k = found["tail_scenarios"]
    alone = np.sort(losses, axis=0)[-k:].mean(axis=0)
    assert np.all(found["contributions"] <= alone + 1e-12) and found["es"] < alone.sum()
    single = [risk.historical_var_es(-losses[:, i], alpha=0.975, method="lower", es_rule="exact", sign="loss_positive")[1]
              for i in range(3)]
    whole = risk.historical_var_es(-losses.sum(axis=1), alpha=0.975, method="lower", es_rule="exact",
                                   sign="loss_positive")[1]
    assert whole <= sum(single)
