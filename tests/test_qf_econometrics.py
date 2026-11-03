"""agent/qf/econometrics.py, the econometrics and factors shelf (runs/lanes/lib2-prompt.md): every function against
statsmodels 0.15 (OLS and its HAC covariance, values frozen in `SM`; running this file under a Python with statsmodels
recomputes and compares them), the estimator written out another way here (a double sum, a stacked GLS, the regression
prediction error), an exact distribution by Monte Carlo (slow), or an identity the method must satisfy, each source
named in its test. No unit's instruction values, checks, expected values or data appear here (rules 8 and 9): the
data are drawn from numpy's legacy RandomState, whose stream numpy keeps fixed."""
import math

import numpy as np
import pytest
from scipy import stats

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf
import qf.econometrics as e

pytestmark = pytest.mark.fast


def regression_data():
    """y = 0.3 + 1.2 x1 - 0.7 x2 + u, u an AR(1) with coefficient 0.6: 60 rows, RandomState(31)."""
    rng = np.random.RandomState(31)
    x = rng.normal(size=(60, 2))
    shocks, u = rng.normal(size=60) * 0.5, np.zeros(60)
    for t in range(60):
        u[t] = 0.6 * (u[t - 1] if t else 0.0) + shocks[t]
    return 0.3 + 1.2 * x[:, 0] - 0.7 * x[:, 1] + u, x


SM = {   # statsmodels 0.15 on regression_data(): OLS; fit(cov_type="HAC", maxlags 3, use_correction False/True); HC0, HC1
    "coef": [0.045625731937191355, 1.1782072315135792, -0.7211728153460498],
    "se": [0.05917156960011178, 0.060755857961059125, 0.05245587952487168],
    "r2": 0.923584402761537, "sigma2": 0.20099708585737783,
    "hac3": [0.0777220190074619, 0.03859213030586069, 0.06768064018728308],
    "hac3_corrected": [0.07974110898200677, 0.03959469025466249, 0.06943887168741694],
    "hc0": [0.056235426676107876, 0.05305844799208658, 0.05555185526741694],
    "hc1": [0.05769633038995867, 0.054436819035116844, 0.05699500092253742],
    "origin_r2": 0.9229556694222115, "origin_coef": [1.1722992565787864, -0.7286938331799502],
    "origin_se": [0.06005976200466993, 0.05136058992852826],
}


def test_the_econometrics_cards_keep_the_format_and_their_examples_run():
    check_cards(e)


# ------------------------------------------------------------------------------------------------ regressions


def test_ols_equals_statsmodels_with_and_without_an_intercept():
    """statsmodels' OLS: coefficients, classical standard errors (s^2 over n - p), R^2, and the uncentered R^2 of a fit
    through the origin; the residuals are orthogonal to every column (the normal equations)."""
    y, x = regression_data()
    fit = e.ols(y=y, X=x, intercept=True)
    assert fit["coef"] == pytest.approx(SM["coef"], rel=1e-12) and fit["se"] == pytest.approx(SM["se"], rel=1e-12)
    assert np.sqrt(np.diag(fit["cov"])) == pytest.approx(SM["se"], rel=1e-12)
    assert fit["r2"] == pytest.approx(SM["r2"], rel=1e-13) and fit["sigma2"] == pytest.approx(SM["sigma2"], rel=1e-13)
    assert fit["dof"] == 57 and fit["tstat"] == pytest.approx(fit["coef"] / fit["se"])
    assert fit["fitted"] == pytest.approx(np.column_stack([np.ones(60), x]) @ SM["coef"], abs=1e-12)
    assert np.column_stack([np.ones(60), x]).T @ fit["resid"] == pytest.approx(np.zeros(3), abs=1e-12)
    origin = e.ols(y=y, X=x, intercept=False)
    assert origin["coef"] == pytest.approx(SM["origin_coef"], rel=1e-12)
    assert origin["se"] == pytest.approx(SM["origin_se"], rel=1e-12) and origin["r2"] == pytest.approx(SM["origin_r2"])
    for bad, name in ((dict(X=np.column_stack([x[:, 0], 2 * x[:, 0]])), "full column rank"),
                      (dict(y=np.r_[y[:-1], np.nan]), "finite"), (dict(intercept="yes"), "intercept"),
                      (dict(X=x[:-1]), "same n")):
        with pytest.raises(ValueError, match=name):
            e.ols(**(dict(y=y, X=x, intercept=True) | bad))


def test_newey_west_equals_statsmodels_hac_and_the_double_sum_it_abbreviates():
    """statsmodels' HAC with maxlags 3, uncorrected ('T') and with its n / (n - k) correction ('T-k'); at 0 lags HC0
    and HC1 (White). And Newey and West's (1987) S written as the band double sum sum_{|t - s| <= L} (1 - |t - s| /
    (L + 1)) e_t e_s x_t x_s'."""
    y, x = regression_data()
    for lags, denominator, key in ((3, "T", "hac3"), (3, "T-k", "hac3_corrected"), (0, "T", "hc0"), (0, "T-k", "hc1")):
        found = e.newey_west_ols(y=y, X=x, intercept=True, lags=lags, denominator=denominator)
        assert found["se"] == pytest.approx(SM[key], rel=1e-12), key
        assert found["coef"] == pytest.approx(SM["coef"], rel=1e-12)
        assert found["tstat"] == pytest.approx(np.divide(SM["coef"], SM[key]), rel=1e-12), key
    design = np.column_stack([np.ones(60), x])
    resid = e.ols(y=y, X=x, intercept=True)["resid"]
    lag = np.abs(np.subtract.outer(np.arange(60), np.arange(60)))
    weights = np.where(lag <= 5, 1 - lag / 6, 0.0) * np.outer(resid, resid)
    bread = np.linalg.inv(design.T @ design)
    assert e.newey_west_ols(y=y, X=x, intercept=True, lags=5, denominator="T")["cov"] == pytest.approx(
        bread @ (design.T @ weights @ design) @ bread, rel=1e-11)
    for bad, name in ((dict(lags=-1), "lags"), (dict(lags=2.0), "lags"), (dict(denominator="n"), "denominator")):
        with pytest.raises(ValueError, match=name):
            e.newey_west_ols(**(dict(y=y, X=x, intercept=True, lags=2, denominator="T") | bad))


def test_rolling_ols_is_ols_on_each_trailing_window():
    y, x = regression_data()
    coef = e.rolling_ols(y=y, X=x, intercept=True, window=20)
    assert coef.shape == (60, 3) and np.isnan(coef[:19]).all()
    for t in (19, 37, 59):
        assert coef[t] == pytest.approx(e.ols(y=y[t - 19:t + 1], X=x[t - 19:t + 1], intercept=True)["coef"], rel=1e-10)
    for window in (3, 61, 20.0):
        with pytest.raises(ValueError, match="window"):
            e.rolling_ols(y=y, X=x, intercept=True, window=window)


def panel():
    """24 periods by 15 assets, returns linear in two characteristics with period-varying premia; RandomState(5)."""
    rng = np.random.RandomState(5)
    chars = rng.normal(size=(24, 15, 2))
    premia = np.column_stack([0.01 + 0.02 * rng.normal(size=24), 0.005 + 0.01 * rng.normal(size=24)])
    returns = 0.002 + np.einsum("tnk,tk->tn", chars, premia) + 0.03 * rng.normal(size=(24, 15))
    return returns, chars


def test_fama_macbeth_averages_the_cross_sections_and_its_errors_are_the_slopes_own():
    """Fama and MacBeth (1973): one OLS per period; the coefficient is the slopes' mean and, at 0 lags with 'T-k',
    its error is std(lambda, ddof=1) / sqrt(T) (Cochrane, Asset Pricing, 12.3); with lags, Newey-West's on the slope
    series, the regression of each slope on a constant. An asset with a NaN leaves its period; a period with too few
    assets is left out."""
    y, x = panel()
    fm = qf.fama_macbeth(y=y, X=x, intercept=True, lags=0, denominator="T-k", correction="none",
                         factor_covariance=None)
    slopes = np.array([np.linalg.lstsq(np.column_stack([np.ones(15), x[t]]), y[t], rcond=None)[0] for t in range(24)])
    assert fm["slopes"] == pytest.approx(slopes, rel=1e-10) and fm["n_periods"] == 24
    assert fm["coef"] == pytest.approx(slopes.mean(axis=0), rel=1e-12)
    assert fm["se"] == pytest.approx(slopes.std(axis=0, ddof=1) / math.sqrt(24), rel=1e-12)
    assert fm["tstat"] == pytest.approx(slopes.mean(axis=0) / (slopes.std(axis=0, ddof=1) / math.sqrt(24)), rel=1e-12)
    for lags, denominator in ((3, "T"), (2, "T-k")):
        nw = qf.fama_macbeth(y=y, X=x, intercept=True, lags=lags, denominator=denominator, correction="none",
                             factor_covariance=None)
        for j in range(3):
            series = e.newey_west_ols(y=slopes[:, j], X=np.ones((24, 0)), intercept=True, lags=lags,
                                      denominator=denominator)
            assert nw["se"][j] == pytest.approx(series["se"][0], rel=1e-10), (lags, j)
    holes = y.copy()
    holes[3, 4] = np.nan
    holes[7, :12] = np.nan                          # 3 assets left for 3 coefficients: no residual, left out
    gapped = qf.fama_macbeth(y=holes, X=x, intercept=True, lags=0, denominator="T-k", correction="none",
                             factor_covariance=None)
    assert gapped["n_periods"] == 23 and np.isnan(gapped["slopes"][7]).all()
    keep = np.arange(15) != 4
    assert gapped["slopes"][3] == pytest.approx(np.linalg.lstsq(np.column_stack([np.ones(14), x[3][keep]]),
                                                                y[3][keep], rcond=None)[0], rel=1e-10)
    one = qf.fama_macbeth(y=y, X=x[..., 0], intercept=False, lags=0, denominator="T", correction="none",
                          factor_covariance=None)
    assert one["coef"].shape == (1,)


# ------------------------------------------------------------------------------------------------ event study

DAYS = np.arange(-45, 6)
ESTIMATION, EVENT = (-45, -8), (-2, 3)


def events(n, seed, shift):
    """n events in event time: market returns and asset returns from a market model with an abnormal `shift` on the
    event days; RandomState(seed)."""
    rng = np.random.RandomState(seed)
    market = 0.01 * rng.normal(size=(n, DAYS.size))
    alpha, beta = 0.0005 * rng.normal(size=(n, 1)), 1 + 0.3 * rng.normal(size=(n, 1))
    asset = alpha + beta * market + 0.015 * rng.normal(size=(n, DAYS.size)) + shift * (DAYS >= -2) * (DAYS <= 3)
    return asset, market


def test_the_event_study_fits_each_market_model_and_cumulates_its_abnormal_returns():
    """The market model's alpha and beta are each event's OLS line over the estimation window (numpy.polyfit); AR is
    the return less the line on the event days, CAR its sum, AAR the cross-event mean, CAAR the sum of AAR (= the mean
    CAR). A NaN in an estimation window drops that day; one in the event window is refused."""
    asset, market = events(6, 3, 0.004)
    asset[2, 5] = np.nan
    study = qf.market_model_event_study(asset_returns=asset, market_returns=market, event_time=DAYS,
                                       estimation_window=ESTIMATION, event_window=EVENT)
    window, during = (DAYS >= -45) & (DAYS <= -8), (DAYS >= -2) & (DAYS <= 3)
    for i in range(6):
        keep = window & np.isfinite(asset[i])
        beta, alpha = np.polyfit(market[i][keep], asset[i][keep], 1)
        assert (study["alpha"][i], study["beta"][i]) == pytest.approx((alpha, beta), rel=1e-9)
        assert study["ar"][i] == pytest.approx(asset[i][during] - alpha - beta * market[i][during], abs=1e-14)
    assert study["n_estimation"].tolist() == [38, 38, 37, 38, 38, 38]
    assert study["event_days"].tolist() == [-2, -1, 0, 1, 2, 3]
    assert study["car"] == pytest.approx(study["ar"].sum(axis=1)) and study["aar"] == pytest.approx(study["ar"].mean(0))
    assert study["caar"] == pytest.approx(study["car"].mean())
    asset[1, DAYS == 1] = np.nan
    with pytest.raises(ValueError, match="event window"):
        qf.market_model_event_study(asset_returns=asset, market_returns=market, event_time=DAYS,
                                   estimation_window=ESTIMATION, event_window=EVENT)
    with pytest.raises(ValueError, match="estimation window first"):
        qf.market_model_event_study(asset_returns=asset, market_returns=market, event_time=DAYS,
                                   estimation_window=(-5, 2), event_window=EVENT)


def test_patell_standardizes_by_the_prediction_error_and_bmp_is_the_t_test_of_the_scars():
    """Patell (1976): each SAR divides AR by s sqrt(1 + x0' (X'X)^{-1} x0), the OLS prediction error of a new point,
    which is s sqrt(C) with C = 1 + 1/L + (Rm - mean)^2 / Sxx; patell_z = sum SCAR / sqrt(sum (L - 2)/(L - 4)), SCAR the
    sum of SARs over sqrt(K). BMP (Boehmer, Musumeci and Poulsen 1991) is the one-sample t statistic of the SCARs
    (scipy.stats.ttest_1samp)."""
    asset, market = events(8, 11, 0.0)
    study = qf.market_model_event_study(asset_returns=asset, market_returns=market, event_time=DAYS,
                                       estimation_window=ESTIMATION, event_window=EVENT)
    window, during = (DAYS >= -45) & (DAYS <= -8), (DAYS >= -2) & (DAYS <= 3)
    for i in range(8):
        design = np.column_stack([np.ones(38), market[i][window]])
        coef, ssr = np.linalg.lstsq(design, asset[i][window], rcond=None)[:2]
        s2 = ssr[0] / 36
        new = np.column_stack([np.ones(6), market[i][during]])
        forecast = s2 * (1 + np.einsum("ij,jk,ik->i", new, np.linalg.inv(design.T @ design), new))
        assert study["sar"][i] == pytest.approx((asset[i][during] - new @ coef) / np.sqrt(forecast), rel=1e-10)
        assert study["sigma"][i] == pytest.approx(math.sqrt(s2), rel=1e-12)
    assert study["scar"] == pytest.approx(study["sar"].sum(axis=1) / math.sqrt(6))
    assert study["patell_z"] == pytest.approx(study["scar"].sum() / math.sqrt(8 * 36 / 34), rel=1e-12)
    assert study["bmp_t"] == pytest.approx(stats.ttest_1samp(study["scar"], 0.0).statistic, rel=1e-12)
    day0 = qf.market_model_event_study(asset_returns=asset, market_returns=market, event_time=DAYS,
                                      estimation_window=ESTIMATION, event_window=(0, 0))
    assert day0["patell_z"] == pytest.approx(study["sar"][:, 2].sum() / math.sqrt(8 * 36 / 34), rel=1e-12)
    single = qf.market_model_event_study(asset_returns=asset[0], market_returns=market[0], event_time=DAYS,
                                        estimation_window=ESTIMATION, event_window=EVENT)
    assert math.isnan(single["bmp_t"]) and single["car"] == pytest.approx(study["car"][:1])
    shifted = qf.market_model_event_study(asset_returns=asset + 0.01 * (DAYS == 0), market_returns=market,
                                         event_time=DAYS, estimation_window=ESTIMATION, event_window=EVENT)
    assert shifted["car"] == pytest.approx(study["car"] + 0.01) and shifted["patell_z"] > study["patell_z"]


@pytest.mark.slow
def test_a_one_day_sar_is_students_t_with_l_minus_2_degrees_under_the_null():
    """Under the market model with normal errors a one-day SAR is exactly t(L - 2) (the regression prediction
    interval), variance (L - 2)/(L - 4): 4,000 events, the variance and the 5% two-sided tail within 4 SE."""
    asset, market = events(4000, 17, 0.0)
    sar = qf.market_model_event_study(asset_returns=asset, market_returns=market, event_time=DAYS,
                                     estimation_window=ESTIMATION, event_window=(0, 0))["sar"][:, 0]
    assert sar.var() == pytest.approx(36 / 34, abs=4 * math.sqrt(2.25 / 4000) * 36 / 34)
    tail = np.mean(np.abs(sar) > stats.t.ppf(0.975, 36))
    assert tail == pytest.approx(0.05, abs=4 * math.sqrt(0.05 * 0.95 / 4000))


# ------------------------------------------------------------------------------------------------ PCA, GRS


def test_pca_diagonalizes_the_named_matrix_and_fixes_each_sign():
    """Jolliffe (2002): V' C V = diag(lambda) in decreasing order, V orthonormal, the scores' covariance diag(lambda)
    with the same ddof, X_c = scores V'; correlation PCA is covariance PCA of the standardized columns; the two sign
    rules hold column by column, on data where numpy's own eigenvectors break each rule somewhere."""
    rng = np.random.RandomState(9)
    mix = rng.normal(size=(4, 4)) * [1.0, 0.5, 3.0, 0.2]
    data = rng.normal(size=(80, 4)) @ mix + [1.0, -2.0, 0.5, 3.0]
    raw_cov, raw_corr = np.linalg.eigh(np.cov(data.T))[1], np.linalg.eigh(np.corrcoef(data.T))[1]
    assert np.any(raw_cov[np.abs(raw_cov).argmax(axis=0), np.arange(4)] < 0) and np.any(raw_corr.sum(axis=0) < 0)
    for ddof in (0, 1):
        found = e.pca(data=data, data_kind="panel", matrix="covariance", ddof=ddof, sign="max_abs_positive")
        v, lam = found["loadings"], found["variances"]
        assert np.all(np.diff(lam) <= 0) and v.T @ v == pytest.approx(np.eye(4), abs=1e-12)
        assert v.T @ np.cov(data.T, ddof=ddof) @ v == pytest.approx(np.diag(lam), abs=1e-10)
        assert np.cov(found["scores"].T, ddof=ddof) == pytest.approx(np.diag(lam), abs=1e-10)
        assert found["scores"] @ v.T == pytest.approx(data - data.mean(axis=0), abs=1e-10)
        assert found["explained"].sum() == pytest.approx(1.0) and found["mean"] == pytest.approx(data.mean(axis=0))
        top = np.abs(v).argmax(axis=0)
        assert np.all(v[top, np.arange(4)] > 0)
    corr = e.pca(data=data, data_kind="panel", matrix="correlation", ddof=1, sign="sum_positive")
    assert corr["variances"] == pytest.approx(np.sort(np.linalg.eigvalsh(np.corrcoef(data.T)))[::-1], rel=1e-10)
    assert np.all(corr["loadings"].sum(axis=0) > 0) and corr["variances"].sum() == pytest.approx(4.0)
    standard = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    again = e.pca(data=standard, data_kind="panel", matrix="covariance", ddof=1, sign="sum_positive")
    assert again["loadings"] == pytest.approx(corr["loadings"], abs=1e-10)
    for bad, name in ((dict(matrix="cov"), "matrix"), (dict(sign="first"), "sign"), (dict(ddof=2), "ddof")):
        with pytest.raises(ValueError, match=name):
            e.pca(**(dict(data=data, data_kind="panel", matrix="covariance", ddof=1, sign="sum_positive") | bad))


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_pca_refuses_an_indefinite_matrix(matrix):
    with pytest.raises(ValueError, match="positive semidefinite"):
        e.pca(data=[[1.0, 2.0], [2.0, 1.0]], data_kind="matrix", matrix=matrix, ddof=None, sign="sum_positive")


def grs_data(t, n, lf, seed):
    rng = np.random.RandomState(seed)
    factors = 0.005 + 0.04 * rng.normal(size=(t, lf))
    returns = factors @ rng.uniform(0.5, 1.5, size=(lf, n)) + 0.02 * rng.normal(size=(t, n))
    return returns, factors


def test_grs_for_one_asset_is_the_squared_alpha_t_and_its_two_forms_agree():
    """With N = 1 the GRS statistic is the square of the intercept's OLS t statistic (F(1, T - 1 - L) = t^2), the mle
    form; the unbiased form differs only by Omega's T - 1 against T (GRS 1989; Cochrane, Asset Pricing, 12.1)."""
    returns, factors = grs_data(40, 1, 2, 2)
    returns = returns + 0.006
    grs = qf.grs_test(asset_excess_returns=returns, factor_excess_returns=factors, covariance="mle")
    fit = e.ols(y=returns[:, 0], X=factors, intercept=True)
    assert grs["statistic"] == pytest.approx(fit["tstat"][0] ** 2, rel=1e-10) and grs["df"] == (1, 37)
    assert grs["p_value"] == pytest.approx(2 * stats.t.sf(abs(fit["tstat"][0]), 37), rel=1e-9)
    returns, factors = grs_data(50, 5, 3, 4)
    mle = qf.grs_test(asset_excess_returns=returns, factor_excess_returns=factors, covariance="mle")
    unbiased = qf.grs_test(asset_excess_returns=returns, factor_excess_returns=factors, covariance="unbiased")
    mu = factors.mean(axis=0)
    q = mu @ np.linalg.solve(np.cov(factors.T, ddof=0), mu)
    assert unbiased["statistic"] == pytest.approx(mle["statistic"] * (1 + q) / (1 + q * 49 / 50), rel=1e-11)
    assert mle["alpha"][3] == pytest.approx(e.ols(y=returns[:, 3], X=factors, intercept=True)["coef"][0], rel=1e-10)
    with pytest.raises(ValueError, match="more periods"):
        qf.grs_test(asset_excess_returns=returns[:8], factor_excess_returns=factors[:8], covariance="mle")


@pytest.mark.slow
def test_grs_is_exactly_f_under_normal_errors():
    """GRS (1989): under zero alphas and normal errors the mle form is F(N, T - N - L) for any T: at T = 24, N = 4,
    L = 2 over 3,000 samples, its mean d2 / (d2 - 2) and its 5% tail within 4 SE."""
    stat = np.array([qf.grs_test(asset_excess_returns=r, factor_excess_returns=f, covariance="mle")["statistic"]
                     for r, f in (grs_data(24, 4, 2, 1000 + s) for s in range(3000))])
    d1, d2 = 4, 18
    var = 2 * d2 ** 2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4))
    assert stat.mean() == pytest.approx(d2 / (d2 - 2), abs=4 * math.sqrt(var / 3000))
    assert np.mean(stat > stats.f.ppf(0.95, d1, d2)) == pytest.approx(0.05, abs=4 * math.sqrt(0.0475 / 3000))


# ------------------------------------------------------------------------------------------------ Black-Litterman

SIGMA = np.array([[0.040, 0.012, 0.006], [0.012, 0.090, 0.018], [0.006, 0.018, 0.0225]])
WEIGHTS = np.array([0.5, 0.2, 0.3])


def test_equilibrium_returns_are_the_reverse_optimization():
    """Black and Litterman (1992): pi = delta Sigma w, the returns at which w is the mean-variance optimum, so
    Sigma^{-1} pi / delta gives the weights back."""
    pi = qf.equilibrium_returns(covariance=SIGMA, market_weights=WEIGHTS, risk_aversion=3.0)
    assert pi == pytest.approx(3.0 * SIGMA @ WEIGHTS) and np.linalg.solve(SIGMA, pi) / 3.0 == pytest.approx(WEIGHTS)
    with pytest.raises(ValueError, match="risk_aversion"):
        qf.equilibrium_returns(covariance=SIGMA, market_weights=WEIGHTS, risk_aversion=0.0)


def test_black_litterman_is_theils_mixed_estimate_and_meets_its_limits():
    """He and Litterman (1999): the posterior mean is the precision-weighted form [(tau S)^-1 + P'O^-1 P]^-1
    [(tau S)^-1 pi + P'O^-1 Q], the GLS estimate on the stacked prior and views (Theil's mixed estimation), and M is
    that system's covariance; exact views (Omega = 0) hold exactly; vague ones leave the prior; one view on an
    uncorrelated asset is the precision-weighted average of prior and view."""
    pi = qf.equilibrium_returns(covariance=SIGMA, market_weights=WEIGHTS, risk_aversion=2.5)
    tau, p = 0.05, np.array([[1.0, -1.0, 0.0], [0.0, 0.5, 0.5]])
    q, omega = np.array([0.02, 0.05]), np.diag([0.0004, 0.0009])
    bl = qf.black_litterman(prior_returns=pi, covariance=SIGMA, tau=tau, pick_matrix=p, view_returns=q,
                           view_covariance=omega)
    precision = np.linalg.inv(tau * SIGMA) + p.T @ np.linalg.inv(omega) @ p
    assert bl["mean_covariance"] == pytest.approx(np.linalg.inv(precision), rel=1e-10)
    assert bl["mean"] == pytest.approx(np.linalg.solve(precision, np.linalg.solve(tau * SIGMA, pi)
                                                       + p.T @ np.linalg.solve(omega, q)), rel=1e-10)
    stacked = np.vstack([np.eye(3), p])
    noise = np.block([[tau * SIGMA, np.zeros((3, 2))], [np.zeros((2, 3)), omega]])
    gls = np.linalg.solve(stacked.T @ np.linalg.solve(noise, stacked), stacked.T @ np.linalg.solve(noise,
                                                                                                  np.r_[pi, q]))
    assert bl["mean"] == pytest.approx(gls, rel=1e-10) and bl["covariance"] == pytest.approx(SIGMA +
                                                                                              bl["mean_covariance"])
    assert qf.black_litterman(prior_returns=pi, covariance=SIGMA, tau=tau, pick_matrix=p, view_returns=q,
                             view_covariance=[0.0004, 0.0009])["mean"] == pytest.approx(bl["mean"])
    exact = qf.black_litterman(prior_returns=pi, covariance=SIGMA, tau=tau, pick_matrix=p, view_returns=q,
                              view_covariance=np.zeros((2, 2)))
    assert p @ exact["mean"] == pytest.approx(q, abs=1e-14)
    vague = qf.black_litterman(prior_returns=pi, covariance=SIGMA, tau=tau, pick_matrix=p, view_returns=q,
                              view_covariance=np.eye(2) * 1e12)
    assert vague["mean"] == pytest.approx(pi, abs=1e-12)
    diagonal = np.diag([0.04, 0.09, 0.0225])
    one = qf.black_litterman(prior_returns=pi, covariance=diagonal, tau=0.1, pick_matrix=[[0.0, 1.0, 0.0]],
                            view_returns=[0.12], view_covariance=[0.003])
    weight = (1 / (0.1 * 0.09)) / (1 / (0.1 * 0.09) + 1 / 0.003)
    assert one["mean"] == pytest.approx([pi[0], weight * pi[1] + (1 - weight) * 0.12, pi[2]], rel=1e-12)
    with pytest.raises(ValueError, match="shapes"):
        qf.black_litterman(prior_returns=pi, covariance=SIGMA, tau=tau, pick_matrix=p[:, :2], view_returns=q,
                          view_covariance=omega)


# ------------------------------------------------------------------------------------------------ attribution


def test_brinson_fachler_splits_the_active_return_and_its_allocation_is_against_the_benchmark():
    """Brinson and Fachler (1985): allocation (wp - wb)(rb - Rb), selection wb (rp - rb), interaction (wp - wb)
    (rp - rb), summing to Rp - Rb; the allocation total equals Brinson-Hood-Beebower's (wp - wb) rb, though not
    segment by segment; equal weights leave no allocation, equal returns no selection or interaction."""
    wp, wb = np.array([0.40, 0.30, 0.25, 0.05]), np.array([0.40, 0.20, 0.35, 0.05])
    rp, rb = np.array([0.20, -0.05, 0.06, 0.01]), np.array([0.10, -0.04, 0.08, 0.01])
    bf = qf.brinson_fachler(portfolio_weights=wp, benchmark_weights=wb, portfolio_returns=rp, benchmark_returns=rb)
    total_b = 0.40 * 0.10 + 0.20 * -0.04 + 0.35 * 0.08 + 0.05 * 0.01
    assert bf["allocation"] == pytest.approx((wp - wb) * (rb - total_b), abs=1e-15)
    assert bf["selection"] == pytest.approx(wb * (rp - rb)) and bf["interaction"] == pytest.approx((wp - wb) * (rp - rb))
    assert bf["total_allocation"] + bf["total_selection"] + bf["total_interaction"] == pytest.approx(bf["active_return"])
    assert bf["active_return"] == pytest.approx(wp @ rp - total_b)
    bhb = (wp - wb) * rb
    assert bf["total_allocation"] == pytest.approx(bhb.sum()) and not np.allclose(bf["allocation"], bhb)
    same = qf.brinson_fachler(portfolio_weights=wb, benchmark_weights=wb, portfolio_returns=rp, benchmark_returns=rb)
    assert same["allocation"] == pytest.approx(np.zeros(4)) and same["total_interaction"] == 0.0
    matched = qf.brinson_fachler(portfolio_weights=wp, benchmark_weights=wb, portfolio_returns=rb, benchmark_returns=rb)
    assert matched["total_selection"] == 0.0 and matched["total_interaction"] == 0.0
    with pytest.raises(ValueError, match="sum to 1"):
        qf.brinson_fachler(portfolio_weights=wp * 100, benchmark_weights=wb, portfolio_returns=rp, benchmark_returns=rb)


def statsmodels_values():
    """SM recomputed with statsmodels: `scratch/lib2/venv/bin/python tests/test_qf_econometrics.py`."""
    import statsmodels.api as sm
    y, x = regression_data()
    with_const = sm.add_constant(x)
    plain = sm.OLS(y, with_const).fit()
    hac = {flag: sm.OLS(y, with_const).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": flag}).bse
           for flag in (False, True)}
    origin = sm.OLS(y, x).fit()
    return {"coef": plain.params.tolist(), "se": plain.bse.tolist(), "r2": plain.rsquared, "sigma2": plain.scale,
            "hac3": hac[False].tolist(), "hac3_corrected": hac[True].tolist(),
            "hc0": sm.OLS(y, with_const).fit(cov_type="HC0").bse.tolist(),
            "hc1": sm.OLS(y, with_const).fit(cov_type="HC1").bse.tolist(), "origin_r2": origin.rsquared,
            "origin_coef": origin.params.tolist(), "origin_se": origin.bse.tolist()}


if __name__ == "__main__":
    fresh = statsmodels_values()
    print("SM frozen values match statsmodels:", all(np.allclose(fresh[k], SM[k], rtol=1e-13, atol=0) for k in SM))
