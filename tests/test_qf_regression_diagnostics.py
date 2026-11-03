"""Lane lib-econometrics1 (runs/lanes/lib-econometrics1-prompt.md): qf.ols_influence, qf.newey_west_cov and
qf.cross_sectional_r2, and the two variants of held functions, qf.pca on a covariance or correlation matrix and
qf.block_bootstrap's iid kind and statistic. Each against statsmodels 0.15.0 on this file's own series (frozen in
`STATSMODELS`; `reference_values()` recomputes them where statsmodels imports, the slow test compares) and against the
textbook identities: leverages summing to p and every deletion statistic by refitting without the row (Belsley, Kuh and
Welsch 1980; Cook 1977); the VIF as the inverse correlation matrix's diagonal; L = 0 the plain covariance and each
diagonal T times newey_west_ols's on a constant (Newey and West 1987); W = I the OLS R^2 (Cochrane 2005, 12.2). No
unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401  (first: it puts agent/ on the path outside pytest too)
import qf

pytestmark = pytest.mark.fast


def design():
    """Twelve rows, two regressors and an outlier, from the legacy generator (a frozen stream)."""
    rng = np.random.RandomState(31)
    x = np.column_stack([rng.normal(size=12), 0.4 * rng.normal(size=12) + np.linspace(0.0, 1.0, 12)])
    y = 0.3 + x @ [1.0, -0.6] + 0.3 * rng.normal(size=12)
    y[4] += 2.0
    return y, x


def factor_series():
    """Sixty dates of three autocorrelated series (an MA(1) of a mixed normal), the legacy generator."""
    rng = np.random.RandomState(32)
    e = rng.normal(size=(61, 3)) @ [[1.0, 0.3, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]
    return 0.01 + 0.02 * (e[1:] + 0.5 * e[:-1])


Y, X = design()
SERIES = factor_series()
# statsmodels 0.15.0: OLSInfluence and variance_inflation_factor on sm.add_constant(X); S_hac_simple(x - mean, L) / T
STATSMODELS = {
    "leverage": [0.22998283481963244, 0.19161547406315682, 0.10057175553908086, 0.15664778599603543, 0.21553239960565573, 0.15004037441681245, 0.1470936993721764, 0.8351192852744515, 0.13318796471273056, 0.2989796778579495, 0.4250639747330785, 0.11616477360924003],
    "cooks_distance": [0.1004369638376541, 0.07033797514063989, 0.011571497957554579, 0.011347064872413718, 0.7716878933560988, 0.006614818209361717, 0.0001784142546747429, 0.8833377487210995, 0.0032511992637942384, 0.021714334528319165, 0.040675055936706526, 0.0009074317882397339],
    "student_external": [-1.0049624409599092, -0.9371083057524426, -0.5346224440210346, -0.4077894293747017, 10.837537625272189, -0.3181030289820175, -0.052532450605676476, -0.7026899997520887, 0.2383818199248969, 0.37163703879855076, 0.38658967929225974, -0.13584388848028287],
    "dfbetas_row4": [5.367088494602977, 0.7697584603704798, -4.441239012186103],
    "vif": [1.0562017300110533, 1.0562017300110533],
    "hac_L2": [[0.0007230840950896727, 0.00030235001457217805, -7.454610279867443e-05], [0.00030235001457217805, 0.0006754147711070623, 9.876338606745019e-05], [-7.454610279867443e-05, 9.876338606745019e-05, 0.0009249144257909799]],
}


def reference_values():
    """STATSMODELS again, from statsmodels itself (a Python where it imports: the lane's scratch/lib-econometrics1/refenv)."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
    from statsmodels.stats.sandwich_covariance import S_hac_simple
    exog = sm.add_constant(X)
    influence = OLSInfluence(sm.OLS(Y, exog).fit())
    u = SERIES - SERIES.mean(axis=0)
    return {"leverage": influence.hat_matrix_diag.tolist(), "cooks_distance": influence.cooks_distance[0].tolist(),
            "student_external": influence.resid_studentized_external.tolist(),
            "dfbetas_row4": influence.dfbetas[4].tolist(),
            "vif": [variance_inflation_factor(exog, j) for j in (1, 2)],
            "hac_L2": (S_hac_simple(u, nlags=2) / SERIES.shape[0]).tolist()}


@pytest.mark.slow
def test_the_frozen_values_are_statsmodels_own():
    pytest.importorskip("statsmodels")
    found = reference_values()
    for key, value in STATSMODELS.items():
        assert np.asarray(found[key]) == pytest.approx(np.asarray(value), rel=1e-12, abs=1e-15), key


# ------------------------------------------------------------------------------------------------ influence and VIF


def test_influence_matches_statsmodels_olsinfluence_and_its_vif():
    found = qf.ols_influence(y=Y, X=X, intercept=True)
    assert found["leverage"] == pytest.approx(STATSMODELS["leverage"], rel=1e-12)
    assert found["cooks_distance"] == pytest.approx(STATSMODELS["cooks_distance"], rel=1e-10)
    assert found["student_external"] == pytest.approx(STATSMODELS["student_external"], rel=1e-10)
    assert found["dfbetas"][4] == pytest.approx(STATSMODELS["dfbetas_row4"], rel=1e-10)
    assert found["vif"] == pytest.approx(STATSMODELS["vif"], rel=1e-12)
    assert np.argmax(np.abs(found["student_external"])) == 4 and found["p"] == 3


@pytest.mark.parametrize("intercept", [True, False])
def test_each_deletion_statistic_is_the_regression_refitted_without_the_row(intercept):
    """Belsley, Kuh and Welsch (1980), 2.1: b - b_(i), s_(i), the external t and DFFITS by refitting with row i
    deleted; Cook's D in Cook's (1977) own form (b - b_(i))' X'X (b - b_(i)) / (p s^2); the leverages sum to p."""
    found = qf.ols_influence(y=Y, X=X, intercept=intercept)
    full = qf.ols(y=Y, X=X, intercept=intercept)
    design_matrix = np.column_stack([np.ones(12), X]) if intercept else X
    p = design_matrix.shape[1]
    assert found["leverage"].sum() == pytest.approx(p, rel=1e-12) and found["sigma2"] == pytest.approx(full["sigma2"])
    for i in range(12):
        keep = np.arange(12) != i
        less = qf.ols(y=Y[keep], X=X[keep], intercept=intercept)
        change = full["coef"] - less["coef"]
        assert found["dfbeta"][i] == pytest.approx(change, rel=1e-9, abs=1e-12)
        assert found["sigma_deleted"][i] == pytest.approx(math.sqrt(less["sigma2"]), rel=1e-10)
        predicted = Y[i] - design_matrix[i] @ less["coef"]
        h = found["leverage"][i]
        assert found["student_external"][i] == pytest.approx(predicted * (1 - h) / (math.sqrt(less["sigma2"] * (1 - h))),
                                                             rel=1e-9)
        assert found["cooks_distance"][i] == pytest.approx(
            change @ design_matrix.T @ design_matrix @ change / (p * full["sigma2"]), rel=1e-9)
        assert found["dffits"][i] == pytest.approx(design_matrix[i] @ change / (math.sqrt(less["sigma2"] * h)), rel=1e-9)
        assert found["dfbetas"][i] == pytest.approx(change / np.sqrt(less["sigma2"] * np.diag(np.linalg.inv(
            design_matrix.T @ design_matrix))), rel=1e-9)


def test_the_vif_is_the_inverse_correlation_matrix_diagonal_and_one_for_uncorrelated_regressors():
    rng = np.random.RandomState(33)
    z = rng.normal(size=(50, 3))
    z[:, 2] += 0.9 * z[:, 0]
    for intercept in (True, False):
        found = qf.ols_influence(y=rng.normal(size=50), X=z, intercept=intercept)["vif"]
        assert found == pytest.approx(np.diag(np.linalg.inv(np.corrcoef(z.T))), rel=1e-10)
    orthogonal = np.linalg.qr(rng.normal(size=(20, 2)) - 0.0)[0]
    orthogonal -= orthogonal.mean(axis=0)
    orthogonal[:, 1] -= orthogonal[:, 0] * (orthogonal[:, 0] @ orthogonal[:, 1]) / (orthogonal[:, 0] @ orthogonal[:, 0])
    assert qf.ols_influence(y=rng.normal(size=20), X=orthogonal, intercept=True)["vif"] == pytest.approx([1.0, 1.0])
    assert qf.ols_influence(y=Y, X=X[:, 0], intercept=True)["vif"] == pytest.approx([1.0])


def test_influence_refuses_a_row_that_fits_itself_and_too_few_rows():
    x = np.r_[np.zeros(6), 1.0]
    with pytest.raises(ValueError, match="leverage 1"):
        qf.ols_influence(y=np.arange(7.0), X=x, intercept=True)
    with pytest.raises(ValueError, match="p \\+ 2"):
        qf.ols_influence(y=[1.0, 2.0, 3.0, 5.0], X=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], intercept=True)


# ------------------------------------------------------------------------------------------------ Newey-West matrix


def test_newey_west_cov_is_statsmodels_hac_and_the_plain_covariance_at_lag_zero():
    found = qf.newey_west_cov(data=SERIES, lags=2, denominator="T", demean=True, scale="long_run")
    assert found["cov"] == pytest.approx(np.array(STATSMODELS["hac_L2"]), rel=1e-12)
    for denominator, ddof in (("T", 0), ("T-1", 1)):
        zero = qf.newey_west_cov(data=SERIES, lags=0, denominator=denominator, demean=True, scale="long_run")
        assert zero["cov"] == pytest.approx(np.cov(SERIES.T, ddof=ddof), rel=1e-12)
    assert qf.newey_west_cov(data=SERIES, lags=0, denominator="T-lag", demean=True, scale="long_run")["cov"] == \
        pytest.approx(np.cov(SERIES.T, ddof=0), rel=1e-12)


@pytest.mark.parametrize("lags", [0, 1, 4])
@pytest.mark.parametrize("denominator, ols_denominator", [("T", "T"), ("T-1", "T-k")])
def test_each_diagonal_is_newey_west_ols_on_a_constant(lags, denominator, ols_denominator):
    found = qf.newey_west_cov(data=SERIES, lags=lags, denominator=denominator, demean=True, scale="mean")
    for j in range(3):
        alone = qf.newey_west_ols(y=SERIES[:, j], X=np.empty((60, 0)), intercept=True, lags=lags,
                                  denominator=ols_denominator)
        assert found["cov"][j, j] == pytest.approx(alone["cov"][0, 0], rel=1e-12)
        assert found["se"][j] == pytest.approx(alone["se"][0], rel=1e-12)
        assert found["tstat"][j] == pytest.approx(alone["tstat"][0], rel=1e-10)


def test_the_long_run_covariance_is_positive_semidefinite_and_its_wald_test_reads_its_mean():
    rng = np.random.RandomState(34)
    for lags in (1, 3, 8):
        data = rng.normal(size=(40, 4)).cumsum(axis=0) * 0.1 + rng.normal(size=(40, 4))
        long_run = qf.newey_west_cov(data=data, lags=lags, denominator="T", demean=True, scale="long_run")
        assert np.linalg.eigvalsh(long_run["cov"]).min() > -1e-12
        mean = qf.newey_west_cov(data=data, lags=lags, denominator="T", demean=True, scale="mean")
        assert mean["cov"] == pytest.approx(long_run["cov"] / 40, rel=1e-12)
        assert long_run["wald"] == pytest.approx(mean["mean"] @ np.linalg.solve(mean["cov"], mean["mean"]))
    one = qf.newey_west_cov(data=SERIES[:, 0], lags=3, denominator="T", demean=True, scale="mean")
    assert one["p_value"] == pytest.approx(math.erfc(abs(one["tstat"][0]) / math.sqrt(2)), rel=1e-10)
    lagged = qf.newey_west_cov(data=SERIES, lags=3, denominator="T-lag", demean=True, scale="long_run")["cov"]
    by_hand = sum((1 - lag / 4) * (lambda g: g + g.T)((SERIES - SERIES.mean(0))[lag:].T
                                                      @ (SERIES - SERIES.mean(0))[:60 - lag] / (60 - lag))
                  for lag in range(1, 4)) + np.cov(SERIES.T, ddof=0)
    assert lagged == pytest.approx(by_hand, rel=1e-12)


def test_without_demeaning_the_raw_products_are_used():
    raw = qf.newey_west_cov(data=SERIES, lags=0, denominator="T", demean=False, scale="long_run")["cov"]
    assert raw == pytest.approx(SERIES.T @ SERIES / 60, rel=1e-12)
    with pytest.raises(ValueError, match="denominator"):
        qf.newey_west_cov(data=SERIES, lags=1, denominator="T-k", demean=True, scale="mean")


def test_newey_west_returns_a_singular_covariance_without_a_joint_wald_test():
    data = np.column_stack([SERIES[:, 0], SERIES[:, 0]])
    found = qf.newey_west_cov(data=data, lags=0, denominator="T", demean=True, scale="long_run")
    assert found["cov"] == pytest.approx(np.cov(data.T, ddof=0), rel=1e-12)
    assert np.isnan(found["wald"]) and np.isnan(found["p_value"])


# ------------------------------------------------------------------------------------------------ cross-sectional R^2


def test_the_ols_cross_sectional_r2_is_ols_on_mean_returns_and_betas():
    rng = np.random.RandomState(35)
    betas = rng.uniform(0.5, 1.5, size=(15, 2))
    means = 0.002 + betas @ [0.006, -0.002] + 0.001 * rng.normal(size=15)
    for intercept in (True, False):
        found = qf.cross_sectional_r2(mean_returns=means, betas=betas, method="ols", cov=None, intercept=intercept)
        plain = qf.ols(y=means, X=betas, intercept=intercept)
        assert found["r2"] == pytest.approx(plain["r2"], rel=1e-12)
        assert found["gamma"] == pytest.approx(plain["coef"], rel=1e-10)
        assert found["errors"] == pytest.approx(plain["resid"], abs=1e-15)


def test_the_gls_r2_is_the_weighted_fit_against_the_weighted_constant_and_ols_under_a_scalar_covariance():
    """Cochrane (2005), 12.2; Kan, Robotti and Shanken (2013): gamma = (B'WB)^-1 B'WR, which is OLS on the whitened
    regression (L'B, L'R), W = LL'; R^2 = 1 - Q / Q0, Q0 the weighted residual of the constant alone. With cov
    proportional to the identity the GLS R^2 is the OLS one; a mean return in the span of the betas gives 1."""
    rng = np.random.RandomState(36)
    betas = rng.uniform(0.5, 1.5, size=(10, 2))
    root = rng.normal(size=(10, 10)) * 0.01
    cov = root @ root.T + 1e-4 * np.eye(10)
    means = 0.001 + betas @ [0.004, 0.002] + 0.002 * rng.normal(size=10)
    found = qf.cross_sectional_r2(mean_returns=means, betas=betas, method="gls", cov=cov, intercept=True)
    whiten = np.linalg.cholesky(np.linalg.inv(cov))
    b = np.column_stack([np.ones(10), betas])
    gamma = np.linalg.lstsq(whiten.T @ b, whiten.T @ means, rcond=None)[0]
    assert found["gamma"] == pytest.approx(gamma, rel=1e-9)
    weighted = lambda e: e @ np.linalg.solve(cov, e)   # noqa: E731
    constant = np.linalg.lstsq(whiten.T @ np.ones((10, 1)), whiten.T @ means, rcond=None)[0][0]
    assert found["r2"] == pytest.approx(1 - weighted(means - b @ gamma) / weighted(means - constant), rel=1e-10)
    scalar = qf.cross_sectional_r2(mean_returns=means, betas=betas, method="gls", cov=3e-4 * np.eye(10), intercept=True)
    ols = qf.cross_sectional_r2(mean_returns=means, betas=betas, method="ols", cov=None, intercept=True)
    assert scalar["r2"] == pytest.approx(ols["r2"], rel=1e-12)
    exact = qf.cross_sectional_r2(mean_returns=b @ [0.001, 0.003, 0.002], betas=betas, method="gls", cov=cov,
                                  intercept=True)
    assert exact["r2"] == pytest.approx(1.0, abs=1e-12) and 0 <= found["r2"] <= 1


def test_the_cross_sectional_r2_refuses_a_mismatched_weighting():
    with pytest.raises(ValueError, match="cov is for method='gls'"):
        qf.cross_sectional_r2(mean_returns=[0.01, 0.02, 0.015], betas=[1.0, 1.2, 0.9], method="ols",
                              cov=np.eye(3), intercept=True)
    with pytest.raises(ValueError, match="N x N"):
        qf.cross_sectional_r2(mean_returns=[0.01, 0.02, 0.015], betas=[1.0, 1.2, 0.9], method="gls", cov=None,
                              intercept=True)


@pytest.mark.parametrize("cov", [np.diag([1.0, 1.0, -1.0]), np.diag([1.0, 1.0, 0.0])])
def test_the_gls_r2_refuses_an_indefinite_or_singular_covariance(cov):
    with pytest.raises(ValueError, match="positive definite"):
        qf.cross_sectional_r2(mean_returns=[0.01, 0.02, 0.015], betas=[1.0, 1.2, 0.9], method="gls", cov=cov,
                              intercept=True)


# ------------------------------------------------------------------------------------------------ the variants


def test_pca_of_a_matrix_is_pca_of_the_panel_it_came_from():
    rng = np.random.RandomState(37)
    data = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 4)) * [1.0, 0.5, 2.0, 0.2]
    for matrix in ("covariance", "correlation"):
        panel = qf.pca(data=data, data_kind="panel", matrix=matrix, ddof=1, sign="sum_positive")
        given = qf.pca(data=np.cov(data.T, ddof=1), data_kind="matrix", matrix=matrix, ddof=None, sign="sum_positive")
        assert given["loadings"] == pytest.approx(panel["loadings"], abs=1e-10)
        assert given["variances"] == pytest.approx(panel["variances"], rel=1e-10)
        assert given["scores"] is None and given["mean"] is None
        assert given["scale"] == pytest.approx(panel["scale"], rel=1e-12)
    corr = qf.pca(data=np.corrcoef(data.T), data_kind="matrix", matrix="covariance", ddof=None, sign="sum_positive")
    assert corr["variances"].sum() == pytest.approx(4.0) and np.all(corr["loadings"].sum(axis=0) > 0)
    with pytest.raises(ValueError, match="symmetric"):
        qf.pca(data=[[1.0, 0.2], [0.3, 1.0]], data_kind="matrix", matrix="covariance", ddof=None, sign="sum_positive")
    with pytest.raises(ValueError, match="data_kind"):
        qf.pca(data=data, data_kind="returns", matrix="covariance", ddof=1, sign="sum_positive")


def test_the_marchenko_pastur_edge_bounds_pure_noise():
    """The edge (1 + sqrt(N/T))^2 on a correlation matrix of independent series (Marchenko and Pastur 1967): the
    largest eigenvalue of pure noise sits at it, within a few per cent at this size."""
    rng = np.random.RandomState(38)
    t, n = 2000, 100
    top = qf.pca(data=rng.normal(size=(t, n)), data_kind="panel", matrix="correlation", ddof=1,
                 sign="sum_positive")["variances"][0]
    assert top == pytest.approx((1 + math.sqrt(n / t)) ** 2, rel=0.05)


def test_the_iid_bootstrap_draws_every_position_uniformly_and_a_statistic_is_evaluated_on_each_draw():
    """Efron and Tibshirani (1993), ch. 6: the bootstrap standard error of a mean is s sqrt((n - 1)/n) / sqrt(n)
    (s with ddof 1), here within the draws' own sampling error."""
    ix = qf.block_bootstrap(n_obs=20, kind="iid", block_length=1, n_draws=5000, rng=3, statistic=None)
    assert ix.shape == (5000, 20) and ix.min() == 0 and ix.max() == 19
    share = np.bincount(ix.ravel(), minlength=20) / ix.size
    assert np.allclose(share, 1 / 20, atol=4 * math.sqrt(1 / 20 / ix.size))
    r = np.random.RandomState(39).normal(0.001, 0.02, size=50)
    means = qf.block_bootstrap(n_obs=50, kind="iid", block_length=None, n_draws=4000, rng=4,
                               statistic=lambda positions: r[positions].mean())
    assert means.shape == (4000,)
    assert means.std() == pytest.approx(r.std(ddof=0) / math.sqrt(50), rel=0.05)
    again = qf.block_bootstrap(n_obs=50, kind="iid", block_length=None, n_draws=4000, rng=4, statistic=None)
    assert means == pytest.approx(r[again].mean(axis=1), rel=1e-12)
    low, high = np.percentile(means, [2.5, 97.5])
    assert low < r.mean() < high


def test_the_held_kinds_draw_as_before_and_a_statistic_needs_a_function():
    """The base library's draws (lane libgaps1's code, n 9, l 3, 2 draws, seed 5), unchanged by the iid kind."""
    before = {"moving": [[4, 5, 6, 5, 6, 7, 0, 1, 2], [5, 6, 7, 3, 4, 5, 3, 4, 5]],
              "circular": [[6, 7, 8, 7, 8, 0, 0, 1, 2], [7, 8, 0, 4, 5, 6, 4, 5, 6]],
              "stationary": [[5, 6, 7, 0, 8, 0, 1, 2, 3], [7, 8, 0, 1, 2, 3, 4, 5, 6]]}
    for kind, draws in before.items():
        assert qf.block_bootstrap(n_obs=9, kind=kind, block_length=3.0, n_draws=2, rng=5,
                                  statistic=None).tolist() == draws
        assert qf.block_bootstrap(n_obs=9, kind=kind, block_length=3.0, n_draws=2, rng=5,
                                  statistic=lambda ix: float(ix[0])).tolist() == [row[0] for row in draws]
    with pytest.raises(ValueError, match="statistic"):
        qf.block_bootstrap(n_obs=9, kind="iid", block_length=1, n_draws=2, rng=5, statistic="mean")
    with pytest.raises(ValueError, match="block_length must be 1 or None"):
        qf.block_bootstrap(n_obs=9, kind="iid", block_length=3, n_draws=2, rng=5, statistic=None)


if __name__ == "__main__":
    print(reference_values())
