"""agent/qf/factors.py's additions (lane lib-signals1; docs/research/library-syllabus-1011.md rows 71 and 72,
docs/research/fresh-gaps-1012.md rows 8 and 9, the Shanken line): the Fama-French 2 x 3 sort worked by hand on a
twelve-stock panel (Fama and French 1993); the Barra regression's identities (X'We = 0, W = I gives OLS, the industry
constraint holds, the country factor is the cap-weighted market under cap weights: Menchero); the Pesaran-Yamagata
statistic written out from its eq. (44) and near N(0, 1) on simulated nulls; the split-sample corrections written out,
exact on a noise-free panel, and the jackknife's and the instrumented estimate's bias below the two-pass estimate's on
a simulated panel; Shanken's correction from Cochrane's 12.2 form. Panels are this file's own; no unit's values appear
(rule 8)."""
import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf

pytestmark = pytest.mark.fast

# ------------------------------------------------------------------------------------------------ the double sort

SIZE = np.arange(1.0, 13.0)                                     # stock i's size is i + 1: median 6.5
BM = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0, 0.05, 0.35])
RET = np.array([0.01, 0.03, -0.02, 0.0, 0.02, 0.01, 0.04, -0.01, 0.0, 0.05, -0.03, 0.02])


def sort2x3(first=SIZE, second=BM, returns=RET, **changes):
    args = dict(first_breakpoints=[0.5], second_breakpoints=[0.3, 0.7], sort="independent", weighting="equal",
                universe=None, rebalance=None, quantile_method="linear")
    return qf.double_sort(first=np.atleast_2d(first), second=np.atleast_2d(second), returns=np.atleast_2d(returns),
                          **{**args, **changes})


def test_the_2x3_independent_sort_worked_by_hand():
    """Size at its median (6.5): stocks 0 to 5 small. Book-to-market's 30th and 70th percentiles over all twelve,
    numpy's linear quantiles, 0.315 and 0.67: low {0, 3, 5, 10}, high {1, 4, 6, 9}, the rest medium. SMB = the small
    portfolios' mean less the big ones'; HML = the two high portfolios' mean less the two low ones'."""
    cells = {(0, 0): [0, 3, 5], (0, 1): [2], (0, 2): [1, 4], (1, 0): [10], (1, 1): [7, 8, 11], (1, 2): [6, 9]}
    found = sort2x3()
    for (i, j), members in cells.items():
        assert found["portfolio_returns"][0, i, j] == pytest.approx(RET[members].mean(), abs=1e-15), (i, j)
        assert found["counts"][0, i, j] == len(members)
    p = {key: RET[members].mean() for key, members in cells.items()}
    smb = (p[0, 0] + p[0, 1] + p[0, 2]) / 3 - (p[1, 0] + p[1, 1] + p[1, 2]) / 3
    hml = (p[0, 2] + p[1, 2]) / 2 - (p[0, 0] + p[1, 0]) / 2
    assert found["smb"][0] == pytest.approx(smb, abs=1e-15) and found["hml"][0] == pytest.approx(hml, abs=1e-15)
    assert found["first_group"][0].tolist() == [0] * 6 + [1] * 6
    odd = sort2x3(first=SIZE[:5], second=BM[:5], returns=RET[:5])
    assert odd["first_group"][0].tolist() == [0, 0, 0, 1, 1]         # the median stock (size 3) goes below: right-closed


def test_the_sequential_sort_splits_each_size_group_at_its_own_percentiles():
    """Within the small stocks book-to-market's 30th and 70th percentiles are 0.25 and 0.6; within the big 0.375 and
    0.7: two stocks a cell."""
    found = sort2x3(sort="sequential")
    assert (found["counts"][0] == 2).all()
    assert found["second_group"][0].tolist() == [0, 2, 1, 1, 2, 0, 2, 1, 1, 2, 0, 0]


def test_the_breakpoints_come_from_the_universe_and_groups_hold_between_rebalances_and_caps_weight():
    universe = np.arange(12) < 8                                  # the first eight set the breakpoints (NYSE-like)
    found = sort2x3(universe=np.atleast_2d(universe))
    cut = np.quantile(SIZE[universe], 0.5)
    low, high = np.quantile(BM[universe], [0.3, 0.7])
    assert found["first_group"][0].tolist() == (SIZE > cut).astype(int).tolist()
    assert found["second_group"][0].tolist() == ((BM > low).astype(int) + (BM > high)).tolist()
    panel = np.vstack([SIZE, SIZE[::-1], SIZE])                   # sizes reverse on row 1: held, not re-sorted
    held = qf.double_sort(first=panel, second=np.vstack([BM] * 3), returns=np.vstack([RET] * 3),
                          first_breakpoints=[0.5], second_breakpoints=[0.3, 0.7], sort="independent",
                          weighting="equal", universe=None, rebalance=[True, False, True], quantile_method="linear")
    assert (held["first_group"][1] == held["first_group"][0]).all()
    caps = np.linspace(1.0, 3.0, 12)
    weighted = sort2x3(weighting=np.atleast_2d(caps))
    members = [7, 8, 11]
    assert weighted["portfolio_returns"][0, 1, 1] == pytest.approx(caps[members] @ RET[members] / caps[members].sum())


# ------------------------------------------------------------------------------------------------ Barra's regression


def test_breakpoint_quantiles_follow_the_named_percentile_rule():
    """Hyndman and Fan (1996), sample quantiles: on [1, 2, 3, 4], the linear median is 2.5 and the higher median 3;
    right-closed bins consequently place stock 3 in different groups. Inverted CDF selects an order statistic."""
    for method, groups in (("linear", [0, 0, 1, 1]), ("higher", [0, 0, 0, 1]),
                           ("inverted_cdf", [0, 0, 1, 1])):
        found = sort2x3(first=SIZE[:4], second=BM[:4], returns=RET[:4], quantile_method=method)
        assert found["first_group"][0].tolist() == groups
    with pytest.raises(ValueError, match="quantile_method"):
        sort2x3(quantile_method="unknown")

RNG = np.random.default_rng(72)
N_STOCKS = 30
INDUSTRY = np.repeat(np.arange(3), 10)
CAP = RNG.lognormal(7.0, 1.0, size=N_STOCKS)


def exposures(style):
    dummies = (INDUSTRY[:, None] == np.arange(3)).astype(float)
    return np.column_stack([np.ones(N_STOCKS), dummies, style])


def test_weights_of_one_give_ols_and_the_residuals_are_orthogonal_under_any_weights():
    x = np.column_stack([np.ones(N_STOCKS), RNG.normal(size=(N_STOCKS, 2))])
    y = RNG.normal(0.0, 0.02, size=N_STOCKS)
    ols = qf.cross_sectional_wls(returns=y, exposures=x, weights="equal", industries=None, industry_weights=None)
    assert ols["factor_returns"] == pytest.approx(np.linalg.lstsq(x, y, rcond=None)[0], abs=1e-15)
    w = np.sqrt(CAP)
    wls = qf.cross_sectional_wls(returns=y, exposures=x, weights=w, industries=None, industry_weights=None)
    assert x.T @ (w * wls["residuals"]) == pytest.approx(np.zeros(3), abs=1e-14)


def test_the_industry_constraint_holds_and_with_cap_weights_the_country_factor_is_the_cap_weighted_market():
    """Menchero's constrained regression: a country column and every industry dummy, the industries' factor returns
    summing to zero weighted by their caps. Where the regression weights are the caps and the style has zero
    cap-weighted mean, the country factor return is the cap-weighted market return."""
    style = RNG.normal(size=N_STOCKS)
    style -= CAP @ style / CAP.sum()
    x = exposures(style)
    y = 0.01 + 0.02 * (INDUSTRY == 1) + 0.005 * style + RNG.normal(0.0, 0.01, size=N_STOCKS)
    found = qf.cross_sectional_wls(returns=y, exposures=x, weights=CAP, industries=[1, 2, 3], industry_weights=CAP)
    b = found["factor_returns"]
    shares = CAP @ x[:, 1:4]
    assert shares @ b[1:4] == pytest.approx(0.0, abs=1e-15 * shares.sum())
    assert b[0] == pytest.approx(CAP @ y / CAP.sum(), abs=1e-14)
    assert CAP @ found["residuals"] == pytest.approx(0.0, abs=1e-10)
    rooted = qf.cross_sectional_wls(returns=y, exposures=x, weights=np.sqrt(CAP), industries=[1, 2, 3],
                                    industry_weights=CAP)
    assert shares @ rooted["factor_returns"][1:4] == pytest.approx(0.0, abs=1e-15 * shares.sum())
    with pytest.raises(ValueError, match="collinear"):
        qf.cross_sectional_wls(returns=y, exposures=x, weights=CAP, industries=None, industry_weights=None)


def test_a_panel_runs_date_by_date_and_a_nan_leaves_its_stock_out():
    style = RNG.normal(size=(2, N_STOCKS))
    x = np.stack([exposures(style[0]), exposures(style[1])])
    y = RNG.normal(0.0, 0.02, size=(2, N_STOCKS))
    y[1, 4] = np.nan
    found = qf.cross_sectional_wls(returns=y, exposures=x, weights=np.sqrt(CAP), industries=[1, 2, 3],
                                   industry_weights=CAP)
    alone = qf.cross_sectional_wls(returns=np.delete(y[1], 4), exposures=np.delete(x[1], 4, axis=0),
                                   weights=np.sqrt(np.delete(CAP, 4)), industries=[1, 2, 3],
                                   industry_weights=np.delete(CAP, 4))
    assert found["factor_returns"][1] == pytest.approx(alone["factor_returns"], abs=1e-15)
    assert np.isnan(found["residuals"][1, 4]) and found["factor_returns"].shape == (2, 5)


def test_exactly_identified_wls_returns_the_unique_fit():
    """The WLS normal equations, Grinold and Kahn: a full-rank square design determines factor returns exactly."""
    found = qf.cross_sectional_wls(returns=[1.0, 3.0], exposures=[[1.0, 0.0], [1.0, 1.0]],
                                   weights="equal", industries=None, industry_weights=None)
    assert found["factor_returns"] == pytest.approx([1.0, 2.0], abs=1e-14)
    assert found["residuals"] == pytest.approx([0.0, 0.0], abs=1e-14)


# ------------------------------------------------------------------------------------------------ the HDA test


def panel(t, n, alphas=0.0, seed=8, correlated=0.0):
    rng = np.random.default_rng(seed)
    f = rng.normal(0.005, 0.04, size=t)
    beta = rng.normal(1.0, 0.3, size=n)
    common = rng.normal(size=(t, 1)) * correlated
    errors = (rng.standard_t(8, size=(t, n)) + common) * 0.05
    return alphas + np.outer(f, beta) + errors, f


def test_the_t_ratios_are_each_assets_ols_intercept_t_and_the_statistics_are_their_formulas():
    """Pesaran and Yamagata (2024), eq. (23) and (44): v = T - m - 1, t_i the OLS t-ratio of alpha_i."""
    r, f = panel(40, 25, correlated=0.8)
    v = 40 - 1 - 1
    t = np.array([qf.ols(y=r[:, i], X=f, intercept=True)["tstat"][0] for i in range(25)])
    normal = qf.hda_test(asset_excess_returns=r, factor_excess_returns=f, version="normal", threshold_p=None,
                         threshold_power=None)
    assert normal["tstats"] == pytest.approx(t, rel=1e-10)
    assert normal["statistic"] == pytest.approx((np.sum(t ** 2) - 25) / np.sqrt(50), rel=1e-10)
    centre, root = v / (v - 2), np.sqrt(2 * (v - 1) / (v - 4))
    student = qf.hda_test(asset_excess_returns=r, factor_excess_returns=f, version="student_t", threshold_p=None,
                          threshold_power=None)
    assert student["statistic"] == pytest.approx(np.sum(t ** 2 - centre) / np.sqrt(25) / (centre * root), rel=1e-10)
    design = np.column_stack([np.ones(40), f])
    resid = r - design @ np.linalg.lstsq(design, r, rcond=None)[0]
    corr = np.corrcoef(resid, rowvar=False)[np.triu_indices(25, 1)]
    from scipy.stats import norm
    kept = corr[np.sqrt(v) * np.abs(corr) > norm.ppf(1 - 0.10 / (2 * 24))]
    rho2 = np.sum(kept ** 2) * 2 / (25 * 24)
    py = qf.hda_test(asset_excess_returns=r, factor_excess_returns=f, version="pesaran_yamagata_2017", threshold_p=0.10,
                     threshold_power=None)
    assert kept.size > 0 and py["rho2"] == pytest.approx(rho2, rel=1e-12)
    assert py["statistic"] == pytest.approx(student["statistic"] / np.sqrt(1 + 24 * rho2), rel=1e-10)
    assert py["p_value"] == pytest.approx(norm.sf(py["statistic"]), rel=1e-12)


def test_the_published_and_working_paper_thresholds_are_distinct():
    """Pesaran-Yamagata (2024), eq. (43): f(N)=N^delta; York 17/04 (2017), eq. (42): f(N)=N-1.
    Orthogonal sine/cosine errors set sqrt(v)*rho=1.8 between their cutoffs at N=2, p=.1, delta=1."""
    j = np.arange(10)
    f = 0.01 * np.cos(4 * np.pi * j / 10)
    u, z = np.sin(2 * np.pi * j / 10), np.cos(2 * np.pi * j / 10)
    rho = 1.8 / np.sqrt(8)
    errors = 0.02 * np.column_stack([u, rho * u + np.sqrt(1 - rho ** 2) * z])
    r = np.array([0.004, 0.006]) + f[:, None] * np.array([0.7, 1.1]) + errors
    working = qf.hda_test(asset_excess_returns=r, factor_excess_returns=f, version="pesaran_yamagata_2017",
                          threshold_p=0.1, threshold_power=None)
    published = qf.hda_test(asset_excess_returns=r, factor_excess_returns=f, version="pesaran_yamagata",
                            threshold_p=0.1, threshold_power=1)
    assert working["tstats"] == pytest.approx([0.8, 1.2], abs=1e-14)
    assert working["rho2"] == pytest.approx(0.405) and published["rho2"] == 0.0
    assert working["statistic"] == pytest.approx(-0.140302527371, abs=5e-13)
    assert published["statistic"] == pytest.approx(-0.166304368124, abs=5e-13)
    with pytest.raises(ValueError, match="threshold_power"):
        qf.hda_test(asset_excess_returns=r, factor_excess_returns=f, version="pesaran_yamagata",
                    threshold_p=0.1, threshold_power=None)


@pytest.mark.slow
def test_under_zero_alphas_the_student_version_is_near_standard_normal_and_alphas_push_it_up():
    """500 simulated nulls, T = 60, N = 100, Student-t errors: the mean within 0.15 of 0 and the spread within 0.15 of
    1; the version that takes the t-ratios as normal sits above 0 by N (v/(v - 2) - 1) / sqrt(2N) on average."""
    draws = {"student_t": [], "normal": []}
    for seed in range(500):
        r, f = panel(60, 100, seed=seed)
        for version in draws:
            draws[version].append(qf.hda_test(asset_excess_returns=r, factor_excess_returns=f, version=version,
                                              threshold_p=None, threshold_power=None)["statistic"])
    student, normal = np.array(draws["student_t"]), np.array(draws["normal"])
    assert abs(student.mean()) < 0.15 and abs(student.std() - 1.0) < 0.15
    assert normal.mean() == pytest.approx(100 * (58 / 56 - 1) / np.sqrt(200), abs=0.15)
    r, f = panel(60, 100, alphas=0.02)
    assert qf.hda_test(asset_excess_returns=r, factor_excess_returns=f, version="student_t",
                       threshold_p=None, threshold_power=None)["p_value"] < 1e-6
    with pytest.raises(ValueError, match="T - m - 1 > 4"):
        qf.hda_test(asset_excess_returns=r[:5], factor_excess_returns=f[:5], version="student_t", threshold_p=None,
                    threshold_power=None)


# ------------------------------------------------------------------------------------------------ split samples


def two_pass(r, f, means, intercept, instruments=None):
    design = np.column_stack([np.ones(r.shape[0]), f])
    betas = np.linalg.lstsq(design, r, rcond=None)[0][1]
    x = np.column_stack([np.ones(betas.size), betas]) if intercept else betas[:, None]
    if instruments is None:
        return np.linalg.lstsq(x, means, rcond=None)[0]
    z = np.column_stack([np.ones(betas.size), instruments]) if intercept else instruments[:, None]
    return np.linalg.solve(z.T @ x, z.T @ means)


def jackknifed(r, f, method, split="halves", intercept=True):
    return qf.split_sample_jackknife(returns=r, factors=f, intercept=intercept, method=method, split=split)


def test_each_method_written_out_on_a_panel():
    r, f = panel(40, 20)
    one, two = np.arange(20), np.arange(20, 40)
    full = two_pass(r, f, r.mean(0), True)
    a, b = two_pass(r[one], f[one], r[one].mean(0), True), two_pass(r[two], f[two], r[two].mean(0), True)
    found = jackknifed(r, f, "jackknife")
    assert found["full"] == pytest.approx(full, rel=1e-10) and found["premia"] == pytest.approx(2 * full - (a + b) / 2)
    design = lambda rows: np.linalg.lstsq(np.column_stack([np.ones(rows.size), f[rows]]), r[rows], rcond=None)[0][1]
    beta_one, beta_two = design(one), design(two)
    iv = (two_pass(r[two], f[two], r.mean(0), True, beta_one) + two_pass(r[one], f[one], r.mean(0), True, beta_two)) / 2
    assert jackknifed(r, f, "instrumented")["premia"] == pytest.approx(iv, rel=1e-10)
    crossed = (two_pass(r[one], f[one], r[two].mean(0), True) + two_pass(r[two], f[two], r[one].mean(0), True)) / 2
    assert jackknifed(r, f, "cross")["premia"] == pytest.approx(crossed, rel=1e-10)
    odd = jackknifed(r, f, "jackknife", split="odd_even")
    a, b = (two_pass(r[rows], f[rows], r[rows].mean(0), True) for rows in (np.arange(0, 40, 2), np.arange(1, 40, 2)))
    assert odd["halves"] == pytest.approx(np.array([a, b]), rel=1e-10)


def test_on_a_noise_free_panel_every_method_returns_the_factors_mean():
    rng = np.random.default_rng(3)
    f = rng.normal(0.01, 0.04, size=24)
    r = np.outer(f, rng.normal(1.0, 0.5, size=12))
    for method in ("jackknife", "instrumented", "cross"):
        found = jackknifed(r, f, method, intercept=False)
        assert found["premia"] == pytest.approx([f.mean()], abs=1e-13), method


def test_odd_panel_jackknife_uses_length_weights_and_both_adjacent_splits():
    """Dhaene and Jochmans (2015), eqs. (2.2), (2.4): unequal subpanels use their length/T weights;
    consecutive odd panels average the two adjacent partitions. Noiseless traded factors retain their mean."""
    f = np.array([0.01, 0.02, -0.01, 0.03, 0.08, -0.02, 0.06])
    r = np.outer(f, [0.5, 1.0, 1.5])
    for split in ("halves", "odd_even"):
        assert jackknifed(r, f, "jackknife", split=split, intercept=False)["premia"] == pytest.approx([f.mean()])
    noisy, factors = panel(41, 20)
    full = two_pass(noisy, factors, noisy.mean(0), True)
    splits = []
    for cut in (20, 21):
        rows = np.arange(cut), np.arange(cut, 41)
        splits.append(sum(len(h) / 41 * two_pass(noisy[h], factors[h], noisy[h].mean(0), True) for h in rows))
    found = jackknifed(noisy, factors, "jackknife")
    assert found["premia"] == pytest.approx(2 * full - np.mean(splits, axis=0), rel=1e-10)


@pytest.mark.slow
def test_on_a_simulated_panel_the_jackknife_and_the_instruments_take_out_the_two_pass_bias():
    """1,000 panels of 50 assets over 60 periods, one traded factor, betas around 1 with errors of vol 10%: the
    two-pass estimate falls short of the factor's own mean (errors in variables); the jackknife's and the
    instrumented estimate's mean error is within three standard errors of 0 and a quarter of the two-pass's; crossing
    halves shrinks it further."""
    rng = np.random.default_rng(9)
    error = {"full": [], "jackknife": [], "instrumented": [], "cross": []}
    for _ in range(1000):
        f = rng.normal(0.01, 0.04, size=60)
        r = np.outer(f, rng.normal(1.0, 0.5, size=50)) + rng.normal(0.0, 0.10, size=(60, 50))
        for method in ("jackknife", "instrumented", "cross"):
            found = jackknifed(r, f, method, intercept=False)
            error[method].append(found["premia"][0] - f.mean())
        error["full"].append(found["full"][0] - f.mean())
    bias = {key: np.mean(value) for key, value in error.items()}
    se = {key: np.std(value) / np.sqrt(1000) for key, value in error.items()}
    assert bias["full"] < -5 * se["full"]
    for method in ("jackknife", "instrumented"):
        assert abs(bias[method]) < 3 * se[method] and abs(bias[method]) < abs(bias["full"]) / 4, method
    assert bias["cross"] < bias["full"]


# ------------------------------------------------------------------------------------------------ Shanken


def test_shankens_correction_is_cochranes_form_on_the_fama_macbeth_variance():
    """Shanken (1992); Cochrane, Asset Pricing (2005), 12.2: V = (1 + c)(V_FM - S_f/T) + S_f/T on the factor rows,
    (1 + c) V_FM on the intercept's, c = lambda' S_f^{-1} lambda over the factor premia; the multiplier form is
    (1 + c) V_FM throughout. One factor of monthly premium 0.5% and vol 4.5% gives c = (0.005/0.045)^2."""
    rng = np.random.default_rng(11)
    t, n = 48, 20
    beta = rng.normal(1.0, 0.4, size=(n, 2))
    factors = rng.normal([0.005, 0.003], [0.045, 0.03], size=(t, 2))
    y = factors @ beta.T + rng.normal(0.0, 0.05, size=(t, n))
    x = np.broadcast_to(beta, (t, n, 2))
    cov = np.cov(factors, rowvar=False)
    plain = qf.fama_macbeth(y=y, X=x, intercept=True, lags=0, denominator="T", correction="none",
                            factor_covariance=None)
    found = qf.fama_macbeth(y=y, X=x, intercept=True, lags=0, denominator="T", correction="shanken",
                            factor_covariance=cov)
    lam = plain["coef"][1:]
    c = lam @ np.linalg.solve(cov, lam)
    own = np.r_[0.0, np.diag(cov)] / t
    assert found["multiplier"] == pytest.approx(1 + c, rel=1e-12) and plain["multiplier"] == 1.0
    assert found["se"] ** 2 == pytest.approx((1 + c) * (plain["se"] ** 2 - own) + own, rel=1e-12)
    assert found["coef"] == pytest.approx(plain["coef"], abs=0) and found["tstat"] == pytest.approx(
        plain["coef"] / found["se"])
    scaled = qf.fama_macbeth(y=y, X=x, intercept=True, lags=0, denominator="T", correction="shanken_multiplier",
                             factor_covariance=cov)
    assert scaled["se"] == pytest.approx(plain["se"] * np.sqrt(1 + c), rel=1e-12)
    assert (0.005 / 0.045) ** 2 == pytest.approx(0.012346, abs=1e-6)
    with pytest.raises(ValueError, match="None with correction 'none'"):
        qf.fama_macbeth(y=y, X=x, intercept=True, lags=0, denominator="T", correction="none", factor_covariance=cov)
    with pytest.raises(ValueError, match="2 x 2"):
        qf.fama_macbeth(y=y, X=x, intercept=True, lags=0, denominator="T", correction="shanken",
                        factor_covariance=np.eye(3))
