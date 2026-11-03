"""agent/qf/volatility.py (runs/lanes/lib1-prompt.md): every function against a worked example (Hull's EWMA and
GARCH(1,1) examples), a published formula written out from its paper (Parkinson, Garman-Klass, Rogers-Satchell,
Yang-Zhang), an identity (Gatheral's local variance from an SVI surface for Dupire, the GARCH likelihood, RV's
expectation, the OU discretisation), Monte Carlo on simulated paths, or an independent implementation as the oracle
(QuantLib 1.43's ParkinsonSigma, GarmanKlassSigma5 and SviSmileSection, numpy.polyfit). The GARCH fit wraps the arch
package; its tests need arch (in the image, not in every test environment) and run in the slow tier. No unit's
instruction values, checks, expected values or data appear."""
import math
import statistics

import numpy as np
import pytest

import qf
import qf.volatility as vol
from test_qf_conventions import card_problems, run_examples

pytestmark = pytest.mark.fast


def test_the_cards_follow_the_format_and_every_example_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert card_problems(vol) == []
    run_examples(vol, tmp_path, skip=("garch11_fit",))


# ------------------------------------------------------------------------------------------------ EWMA and GARCH


def test_ewma_volatility_gives_hulls_example_and_annualises_by_the_root():
    """Hull, "The EWMA model": lambda 0.90, yesterday's vol 1%, yesterday's move 2%: today's vol sqrt(0.9 x 0.0001 +
    0.1 x 0.0004) = 1.14%. The path is the RiskMetrics recursion on squared returns, times sqrt(P)."""
    assert vol.ewma_volatility([0.02], lam=0.9, seed=0.0001, nan="raise", periods_per_year=1) == pytest.approx(
        [math.sqrt(0.00013)], rel=1e-14)
    r = np.random.default_rng(1).normal(0, 0.01, 30)
    lam, t = 0.94, np.arange(30)
    variance = [lam ** (k + 1) * r[0] ** 2 + (1 - lam) * np.sum(lam ** (k - t[:k + 1]) * r[:k + 1] ** 2) for k in t]
    assert vol.ewma_volatility(r, lam=lam, seed="first", nan="raise", periods_per_year=252) == pytest.approx(
        np.sqrt(np.array(variance) * 252), rel=1e-13)
    with pytest.raises(ValueError):
        vol.ewma_volatility(r, lam=lam, seed="first", nan="raise", periods_per_year=0)


def test_garch_variance_gives_hulls_example_and_both_start_rules():
    """Hull, "The GARCH(1,1) model": omega 0.000002, alpha 0.13, beta 0.86, yesterday's vol 1.6% and move 1%: today's
    variance 0.000002 + 0.13 x 0.0001 + 0.86 x 0.000256 = 0.00023516, vol 1.53%; the long-run variance omega / (1 -
    alpha - beta) = 0.0002, 1.4% a day. 'backcast' starts at omega + (alpha + beta) v, 'sigma2_0' at v."""
    path, nxt = vol.garch11_variance([0.01], mu=0.0, omega=2e-6, alpha=0.13, beta=0.86, start_variance=0.000256,
                                     start="sigma2_0")
    assert path.tolist() == [0.000256] and nxt == pytest.approx(0.00023516, rel=1e-13)
    r = np.array([0.012, -0.004, 0.021, -0.017, 0.003])
    for start, first in (("sigma2_0", 1e-4), ("backcast", 2e-6 + 0.99 * 1e-4)):
        path, nxt = vol.garch11_variance(r, mu=0.001, omega=2e-6, alpha=0.13, beta=0.86, start_variance=1e-4,
                                         start=start)
        expected = [first]
        for e in (r - 0.001):
            expected.append(2e-6 + 0.13 * e * e + 0.86 * expected[-1])
        assert path == pytest.approx(expected[:-1], rel=1e-14) and nxt == pytest.approx(expected[-1], rel=1e-14)
    with pytest.raises(ValueError):
        vol.garch11_variance(r, mu=0.0, omega=2e-6, alpha=-0.1, beta=0.86, start_variance=1e-4, start="sigma2_0")
    with pytest.raises(ValueError):
        vol.garch11_variance(r, mu=0.0, omega=2e-6, alpha=0.1, beta=0.86, start_variance=1e-4, start="arch")


def test_garch_forecasts_follow_hulls_mean_reversion_formula():
    """Hull, "Using GARCH(1,1) to forecast future volatility": E[sigma^2_{n+h}] = V_L + (alpha + beta)^h (sigma^2_n -
    V_L), here counted from the one-step forecast v_1; at alpha + beta = 1 the forecast grows by omega a step."""
    omega, alpha, beta, v1 = 2e-6, 0.13, 0.86, 0.0003
    v_long = omega / (1 - alpha - beta)
    h = np.arange(25)
    assert vol.garch11_forecast(v1, omega=omega, alpha=alpha, beta=beta, horizon=25) == pytest.approx(
        v_long + (alpha + beta) ** h * (v1 - v_long), rel=1e-12)
    assert vol.garch11_forecast(v1, omega=omega, alpha=0.1, beta=0.9, horizon=5) == pytest.approx(v1 + h[:5] * omega)
    with pytest.raises(ValueError):
        vol.garch11_forecast(v1, omega=omega, alpha=alpha, beta=beta, horizon=0)


def simulate_garch(n, omega, alpha, beta, mu, seed):
    rng = np.random.default_rng(seed)
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = mu + math.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * (r[t] - mu) ** 2 + beta * s2
    return r


@pytest.mark.slow
def test_garch_fit_is_its_own_recursion_and_likelihood_and_recovers_a_simulated_path(tmp_path, monkeypatch):
    """Bollerslev (1986) through arch: the fitted conditional vol is garch11_variance on the fitted parameters from
    the backcast reported (in the returns' units, so the scale is undone), the log-likelihood is the normal GARCH
    likelihood of the returns themselves, the fit beats the true parameters' likelihood, the scale does not move the
    estimates, and a simulated path's parameters come back."""
    pytest.importorskip("arch")
    monkeypatch.chdir(tmp_path)
    run_examples(vol, tmp_path, skip=tuple(n for n in vol.__all__ if n != "garch11_fit"))
    omega, alpha, beta, mu = 2e-6, 0.08, 0.90, 0.0004
    r = simulate_garch(3000, omega, alpha, beta, mu, seed=2)
    fits = {}
    for mean, start in (("constant", "arch"), ("zero", "sample"), ("constant", 1.5e-4)):
        fit = vol.garch11_fit(r, mean=mean, dist="normal", scale=100.0, start_variance=start, periods_per_year=252)
        fits[mean, start] = fit
        path, nxt = vol.garch11_variance(r, mu=fit["mu"], omega=fit["omega"], alpha=fit["alpha"], beta=fit["beta"],
                                         start_variance=fit["backcast"], start="backcast")
        assert fit["conditional_vol"] == pytest.approx(np.sqrt(path), rel=1e-9)
        assert fit["conditional_vol_annual"] == pytest.approx(np.sqrt(path * 252), rel=1e-9)
        assert (fit["next_vol"], fit["next_vol_annual"]) == pytest.approx((math.sqrt(nxt), math.sqrt(nxt * 252)))
        e = r - fit["mu"]
        loglik = -0.5 * np.sum(np.log(2 * math.pi * path) + e * e / path)
        assert fit["loglik"] == pytest.approx(loglik, rel=1e-9) and fit["converged"]
        true_path, _ = vol.garch11_variance(r, mu=mu if mean == "constant" else 0.0, omega=omega, alpha=alpha,
                                            beta=beta, start_variance=fit["backcast"], start="backcast")
        e_true = r - (mu if mean == "constant" else 0.0)
        assert fit["loglik"] >= -0.5 * np.sum(np.log(2 * math.pi * true_path) + e_true ** 2 / true_path)
    assert fits["zero", "sample"]["backcast"] == pytest.approx(np.mean(r ** 2), rel=1e-12)
    assert fits["constant", 1.5e-4]["backcast"] == pytest.approx(1.5e-4, rel=1e-12)
    weights = 0.94 ** np.arange(75)
    assert fits["constant", "arch"]["backcast"] == pytest.approx(
        np.sum(weights * (r[:75] - r.mean()) ** 2) / weights.sum(), rel=1e-12)
    fit = fits["constant", "arch"]
    assert (fit["alpha"], fit["beta"]) == pytest.approx((alpha, beta), abs=0.04)
    assert fit["mu"] == pytest.approx(mu, abs=4 * math.sqrt(omega / (1 - alpha - beta) / 3000))
    assert math.isnan(fit["nu"]) and fit["long_run_variance"] == pytest.approx(
        fit["omega"] / (1 - fit["persistence"]))
    rescaled = vol.garch11_fit(r, mean="constant", dist="normal", scale=10.0, start_variance="arch",
                               periods_per_year=252)
    assert (rescaled["omega"], rescaled["alpha"], rescaled["beta"]) == pytest.approx(
        (fit["omega"], fit["alpha"], fit["beta"]), rel=2e-3)
    student = vol.garch11_fit(r, mean="zero", dist="t", scale=100.0, start_variance="arch", periods_per_year=252)
    assert student["nu"] > 8 and student["loglik"] >= fits["zero", "sample"]["loglik"] - 5.0
    with pytest.raises(ValueError):
        vol.garch11_fit(r, mean="ar1", dist="normal", scale=100.0, start_variance="arch", periods_per_year=252)


# ------------------------------------------------------------------------------------------------ range estimators


BARS = np.array([[100.0, 101.3, 99.2, 100.8], [100.5, 102.0, 100.1, 101.9], [102.3, 102.9, 100.7, 101.0],
                 [100.9, 101.4, 99.0, 99.6], [99.9, 100.6, 98.8, 100.4], [100.2, 101.9, 100.0, 101.7]])


def test_range_estimators_are_their_papers_formulas_on_hand_bars():
    """Parkinson (1980): (ln H/L)^2 / (4 ln 2); Garman and Klass (1980) sigma-hat-5: 0.5 (ln H/L)^2 - (2 ln 2 - 1)
    (ln C/O)^2; Rogers and Satchell (1991): ln(H/C) ln(H/O) + ln(L/C) ln(L/O); each averaged over the bars and
    annualised by sqrt(P). Close-to-close is the sample standard deviation of log returns (statistics.stdev)."""
    o, h, l, c = BARS.T
    u, d, x = np.log(h / o), np.log(l / o), np.log(c / o)
    parkinson = np.mean((u - d) ** 2) / (4 * math.log(2))
    gk = np.mean(0.5 * (u - d) ** 2 - (2 * math.log(2) - 1) * x ** 2)
    rs = np.mean(u * (u - x) + d * (d - x))
    assert vol.parkinson_vol(h, l, periods_per_year=252) == pytest.approx(math.sqrt(252 * parkinson), rel=1e-14)
    assert vol.garman_klass_vol(o, h, l, c, periods_per_year=252) == pytest.approx(math.sqrt(252 * gk), rel=1e-14)
    assert vol.rogers_satchell_vol(o, h, l, c, periods_per_year=1) == pytest.approx(math.sqrt(rs), rel=1e-14)
    returns = np.diff(np.log(c))
    assert vol.close_to_close_vol(c, ddof=1, periods_per_year=52) == pytest.approx(
        statistics.stdev(returns) * math.sqrt(52), rel=1e-14)
    assert vol.close_to_close_vol(c, ddof=0, periods_per_year=1) == pytest.approx(statistics.pstdev(returns))
    with pytest.raises(ValueError, match="low <= open"):
        vol.garman_klass_vol(o, h, l, c * 1.05, periods_per_year=252)


def test_a_bar_without_wicks_has_zero_rogers_satchell_and_known_parkinson_and_garman_klass_values():
    """A bar whose high and low are its open and close: Rogers-Satchell's two products vanish, Parkinson is
    c^2 / (4 ln 2) and Garman-Klass (1.5 - 2 ln 2) c^2, c = ln(C/O)."""
    o, c = np.array([100.0, 103.0]), np.array([103.0, 99.0])
    h, l = np.maximum(o, c), np.minimum(o, c)
    x2 = np.log(c / o) ** 2
    assert vol.rogers_satchell_vol(o, h, l, c, periods_per_year=1) == 0.0
    assert vol.parkinson_vol(h, l, periods_per_year=1) == pytest.approx(math.sqrt(np.mean(x2) / (4 * math.log(2))))
    assert vol.garman_klass_vol(o, h, l, c, periods_per_year=1) == pytest.approx(
        math.sqrt((1.5 - 2 * math.log(2)) * np.mean(x2)), rel=1e-13)


def test_yang_zhang_combines_its_three_variances_with_the_minimum_variance_k():
    """Yang and Zhang (2000), eq. for sigma^2_YZ with alpha = 1.34: over the n = N - 1 bars with a previous close,
    V_o + k V_c + (1 - k) V_RS, V_o and V_c sample variances (ddof 1), k = 0.34 / (1.34 + (n + 1)/(n - 1)). Without
    overnight gaps or wicks only k V_c is left."""
    o, h, l, c = BARS.T
    n = len(o) - 1
    overnight, intraday = np.log(o[1:] / c[:-1]), np.log(c[1:] / o[1:])
    rs = np.mean(np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) + np.log(l[1:] / c[1:]) * np.log(l[1:] / o[1:]))
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    expected = np.var(overnight, ddof=1) + k * np.var(intraday, ddof=1) + (1 - k) * rs
    assert vol.yang_zhang_vol(o, h, l, c, periods_per_year=252) == pytest.approx(math.sqrt(252 * expected), rel=1e-14)
    closes = 100 * np.exp(np.cumsum([0.01, -0.02, 0.015, 0.004, -0.011]))
    opens = np.r_[100.0, closes[:-1]]
    flat = vol.yang_zhang_vol(opens, np.maximum(opens, closes), np.minimum(opens, closes), closes, periods_per_year=1)
    kk = 0.34 / (1.34 + 5 / 3)
    assert flat == pytest.approx(math.sqrt(kk * np.var(np.log(closes[1:] / opens[1:]), ddof=1)), rel=1e-13)
    with pytest.raises(ValueError):
        vol.yang_zhang_vol(o[:2], h[:2], l[:2], c[:2], periods_per_year=252)


def test_parkinson_and_garman_klass_agree_with_quantlib():
    """QuantLib 1.43's ParkinsonSigma and GarmanKlassSigma5 as the oracle: per bar sqrt(point / yearFraction), so the
    mean of their squares times the bar's year fraction is our per-bar variance."""
    ql = pytest.importorskip("QuantLib")
    rng = np.random.default_rng(3)
    o = 100 * np.exp(rng.normal(0, 0.01, 40))
    c = o * np.exp(rng.normal(0, 0.012, 40))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.006, 40)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.006, 40)))
    dates = [ql.Date(2, 1, 2024) + i for i in range(40)]
    bars = ql.IntervalPriceVector()
    for values in zip(o, h, l, c):
        o_, h_, l_, c_ = map(float, values)
        bars.push_back(ql.IntervalPrice(o_, c_, h_, l_))          # QuantLib's order: open, close, high, low
    series = ql.IntervalPriceTimeSeries(dates, bars)
    for estimator, ours in ((ql.ParkinsonSigma(1 / 252), vol.parkinson_vol(h, l, periods_per_year=252)),
                            (ql.GarmanKlassSigma5(1 / 252), vol.garman_klass_vol(o, h, l, c, periods_per_year=252))):
        theirs = np.array(list(estimator.calculate(series).values()))
        assert ours == pytest.approx(math.sqrt(np.mean(theirs ** 2)), rel=1e-12)


@pytest.mark.slow
def test_the_range_estimators_are_unbiased_on_simulated_brownian_bars():
    """Parkinson (1980) and Garman and Klass (1980) for a driftless Brownian motion, Rogers and Satchell (1991) with
    drift, Yang and Zhang (2000) with drift and opening gaps: the mean estimate is the bar's variance (the gap's added
    for Yang-Zhang). Each bar is 1,600 steps of the motion; the discretely sampled high and low miss the true range
    by O(m^-1/2), removed by Richardson's extrapolation from 400 to 1,600 steps on the same paths."""
    rng = np.random.default_rng(4)
    bars, steps, sigma, gap = 6000, 1600, 0.02, 0.01

    def estimates(drift):
        paths = np.cumsum(rng.normal(drift / steps, sigma / math.sqrt(steps), (bars, steps)), axis=1)
        gaps = rng.normal(0, gap, bars)
        found = {}
        for m in (400, 1600):
            sub = np.concatenate([np.zeros((bars, 1)), paths[:, steps // m - 1::steps // m]], axis=1)
            log_open = math.log(100.0) + np.cumsum(gaps) + np.r_[0.0, np.cumsum(sub[:-1, -1])]
            o = np.exp(log_open)
            c, h, l = o * np.exp(sub[:, -1]), o * np.exp(sub.max(axis=1)), o * np.exp(sub.min(axis=1))
            found[m] = {"parkinson": vol.parkinson_vol(h, l, periods_per_year=1) ** 2,
                        "gk": vol.garman_klass_vol(o, h, l, c, periods_per_year=1) ** 2,
                        "rs": vol.rogers_satchell_vol(o, h, l, c, periods_per_year=1) ** 2,
                        "yz": vol.yang_zhang_vol(o, h, l, c, periods_per_year=1) ** 2}
        return {k: 2 * found[1600][k] - found[400][k] for k in found[1600]}
    flat = estimates(0.0)
    assert flat["parkinson"] == pytest.approx(sigma ** 2, rel=0.03)
    assert flat["gk"] == pytest.approx(sigma ** 2, rel=0.03)
    trending = estimates(0.03)
    assert trending["rs"] == pytest.approx(sigma ** 2, rel=0.03)
    assert trending["yz"] == pytest.approx(sigma ** 2 + gap ** 2, rel=0.04)
    assert trending["gk"] != pytest.approx(sigma ** 2, rel=0.03)


# ------------------------------------------------------------------------------------------------ realized variance


def test_realized_variance_sums_sampled_squared_returns_and_is_unbiased_for_brownian_paths():
    """Andersen, Bollerslev, Diebold and Labys (2001): RV over a grid is the sum of squared log returns, E[RV] =
    sigma^2 T on any grid of a Brownian path; Zhang, Mykland and Ait-Sahalia (2005): RV-avg is the mean over the
    step offset grids, each spanning from its first point to its last, so its expectation is sigma^2 times that
    span. Hand sums on a short path; Monte Carlo over 500 simulated days at four standard errors."""
    p = np.log([100.0, 100.4, 99.9, 100.7, 100.2, 101.0, 100.6])
    assert qf.realized_variance(p, step=1, subsample="first") == pytest.approx(np.sum(np.diff(p) ** 2), rel=1e-14)
    assert qf.realized_variance(p, step=2, subsample="first") == pytest.approx(
        (p[2] - p[0]) ** 2 + (p[4] - p[2]) ** 2 + (p[6] - p[4]) ** 2, rel=1e-14)
    assert qf.realized_variance(p, step=2, subsample="average") == pytest.approx(
        ((p[2] - p[0]) ** 2 + (p[4] - p[2]) ** 2 + (p[6] - p[4]) ** 2 + (p[3] - p[1]) ** 2 + (p[5] - p[3]) ** 2) / 2,
        rel=1e-14)
    days = np.cumsum(np.random.default_rng(5).normal(0, 0.01 / math.sqrt(390), (500, 391)), axis=1)
    per_step = 1e-4 / 390
    for step, rule in ((1, "first"), (5, "first"), (13, "average"), (7, "average")):
        spans = [(390 - o) // step * step for o in range(step)] if rule == "average" else [390 // step * step]
        rvs = np.array([qf.realized_variance(day, step=step, subsample=rule) for day in days])
        assert abs(rvs.mean() - per_step * np.mean(spans)) < 4 * rvs.std() / math.sqrt(500), (step, rule)
    for bad in (0, 7, 1.0):
        with pytest.raises(ValueError):
            qf.realized_variance(p, step=bad, subsample="first")


# ------------------------------------------------------------------------------------------------ mean reversion


def test_ar1_fit_is_least_squares_with_its_ddof_and_half_life():
    """Hamilton 8.1: the AR(1) by OLS, against numpy.polyfit; the residual variance SSR / (n - 1 - ddof); the
    half-life h solves b^h = 1/2."""
    x = np.array([0.50, 0.42, 0.47, 0.31, 0.38, 0.22, 0.30, 0.19, 0.26, 0.12, 0.20])
    slope, intercept = np.polyfit(x[:-1], x[1:], 1)
    resid = x[1:] - intercept - slope * x[:-1]
    for ddof in (0, 2):
        fit = qf.ar1_fit(x, ddof=ddof)
        assert (fit["slope"], fit["intercept"]) == pytest.approx((slope, intercept), rel=1e-12)
        assert fit["resid_var"] == pytest.approx(np.sum(resid ** 2) / (10 - ddof), rel=1e-12)
        assert fit["slope"] ** fit["half_life_periods"] == pytest.approx(0.5, rel=1e-12) and fit["n"] == 10
        assert fit["resid"] == pytest.approx(resid, abs=1e-15) and fit["resid"].size == 10
    assert qf.ar1_fit(np.arange(10.0) ** 1.5, ddof=2)["half_life_periods"] == math.inf
    assert math.isnan(qf.ar1_fit([1.0, -1.0, 1.1, -0.9, 1.0, -1.2], ddof=2)["half_life_periods"])
    with pytest.raises(ValueError):
        qf.ar1_fit(x, ddof=10)


def test_ou_fit_inverts_the_exact_discretisation_and_recovers_a_simulated_path():
    """Brigo and Mercurio 3.2.1: ou_fit is qf.ou_from_ar1 on the AR(1) fit, half-life ln 2 / kappa; on an exact OU
    simulation (kappa 3, theta 0.5, sigma 0.3, daily over 40 years) kappa, theta and sigma come back within four
    standard errors of the AR(1) estimates' sampling spread."""
    rng = np.random.default_rng(6)
    kappa, theta, sigma, dt, n = 3.0, 0.5, 0.3, 1 / 252, 10080
    b = math.exp(-kappa * dt)
    sd = sigma * math.sqrt((1 - b * b) / (2 * kappa))
    x = np.empty(n)
    x[0] = theta
    shocks = rng.standard_normal(n) * sd
    for t in range(1, n):
        x[t] = theta + (x[t - 1] - theta) * b + shocks[t]
    fit = qf.ou_fit(x, dt=dt, ddof=2, scheme="exact")
    ar = qf.ar1_fit(x, ddof=2)
    assert (fit["kappa"], fit["theta"], fit["sigma"]) == pytest.approx(
        qf.ou_from_ar1(ar["intercept"], ar["slope"], ar["resid_var"], dt), rel=1e-14)
    assert fit["half_life"] == pytest.approx(math.log(2) / fit["kappa"], rel=1e-14)
    assert fit["half_life_periods"] == pytest.approx(fit["half_life"] / dt, rel=1e-9)
    assert fit["kappa"] == pytest.approx(kappa, abs=4 * math.sqrt(2 * kappa / (n * dt)))
    assert fit["theta"] == pytest.approx(theta, abs=4 * sigma / math.sqrt(2 * kappa) / math.sqrt(kappa * n * dt / 2))
    assert fit["sigma"] == pytest.approx(sigma, rel=4 / math.sqrt(2 * n))
    assert fit["resid"] == pytest.approx(ar["resid"], abs=0) and fit["resid"].size == n - 1


def test_ou_fit_under_euler_is_the_regression_of_the_change_on_the_level():
    """The Euler form dx = kappa (theta - x) dt + sigma dW read as Delta x = alpha + beta x + e: kappa = -beta / dt,
    theta = -alpha / beta, sigma = sd(e) / sqrt(dt), the regression by numpy.polyfit of Delta x on x; the residuals
    are the exact scheme's (one regression); kappa differs from the exact scheme's -ln(1 + beta) / dt, more so the
    larger kappa dt."""
    rng = np.random.default_rng(16)
    x = np.empty(300)
    x[0] = 0.4
    for t in range(1, 300):
        x[t] = x[t - 1] + 6.0 * (0.2 - x[t - 1]) * 0.02 + 0.3 * math.sqrt(0.02) * rng.standard_normal()
    beta, alpha = np.polyfit(x[:-1], np.diff(x), 1)
    resid = np.diff(x) - alpha - beta * x[:-1]
    euler = qf.ou_fit(x, dt=0.02, ddof=2, scheme="euler")
    assert euler["kappa"] == pytest.approx(-beta / 0.02, rel=1e-10)
    assert euler["theta"] == pytest.approx(-alpha / beta, rel=1e-10)
    assert euler["sigma"] == pytest.approx(math.sqrt(np.sum(resid ** 2) / (299 - 2) / 0.02), rel=1e-10)
    assert euler["resid"] == pytest.approx(resid, abs=1e-14)
    assert euler["half_life"] == pytest.approx(math.log(2) / euler["kappa"])
    assert euler["half_life_periods"] == pytest.approx(euler["half_life"] / 0.02)
    exact = qf.ou_fit(x, dt=0.02, ddof=2, scheme="exact")
    assert exact["kappa"] == pytest.approx(-math.log(1 + beta) / 0.02, rel=1e-10) and exact["kappa"] > euler["kappa"]
    assert np.array_equal(exact["resid"], euler["resid"]) and exact["theta"] == pytest.approx(euler["theta"])
    with pytest.raises(ValueError, match="scheme"):
        qf.ou_fit(x, dt=0.02, ddof=2, scheme="milstein")
    with pytest.raises(ValueError, match="below 1"):                  # a trending path: no mean reversion
        qf.ou_fit(np.arange(20.0) ** 1.5, dt=0.02, ddof=2, scheme="euler")


# ------------------------------------------------------------------------------------------------ surfaces


def test_svi_has_gatherals_vertex_and_wings():
    """Gatheral (2004): w(m) = a + b sigma; the minimum a + b sigma sqrt(1 - rho^2) at m - rho sigma / sqrt(1 -
    rho^2); wings of slope b (1 -/+ rho)."""
    a, b, rho, m, s = 0.02, 0.15, -0.35, 0.05, 0.25
    w = lambda k: qf.svi_total_variance(k, a=a, b=b, rho=rho, m=m, sigma=s)  # noqa: E731
    assert w(m) == pytest.approx(a + b * s, rel=1e-15)
    vertex = m - rho * s / math.sqrt(1 - rho * rho)
    grid = np.linspace(vertex - 1e-3, vertex + 1e-3, 2001)
    assert grid[np.argmin(w(grid))] == pytest.approx(vertex, abs=2e-6)
    assert w(vertex) == pytest.approx(a + b * s * math.sqrt(1 - rho * rho), rel=1e-14)
    assert (w(60.0) - w(50.0)) / 10 == pytest.approx(b * (1 + rho), rel=1e-4)
    assert (w(-60.0) - w(-50.0)) / 10 == pytest.approx(b * (1 - rho), rel=1e-4)
    assert type(w(0.1)) is float and w(np.array([0.1, 0.2])).shape == (2,)
    with pytest.raises(ValueError):
        qf.svi_total_variance(0.1, a=a, b=b, rho=1.0, m=m, sigma=s)


def test_svi_agrees_with_quantlibs_svi_smile_section():
    """QuantLib 1.43's SviSmileSection(T, F, [a, b, sigma, rho, m]) as the oracle: its variance at strike K is the
    total variance at k = ln(K / F)."""
    ql = pytest.importorskip("QuantLib")
    a, b, rho, m, s = 0.02, 0.15, -0.35, 0.05, 0.25
    section = ql.SviSmileSection(1.5, 100.0, [a, b, s, rho, m])
    for strike in (60.0, 90.0, 100.0, 125.0, 180.0):
        assert qf.svi_total_variance(math.log(strike / 100.0), a=a, b=b, rho=rho, m=m, sigma=s) == pytest.approx(
            section.variance(strike), rel=1e-13)


def test_svi_fit_recovers_the_parameters_of_an_exact_smile():
    """Least squares on exact SVI total variances returns the parameters that made them."""
    k = np.linspace(-0.6, 0.5, 23)
    truth = dict(a=0.015, b=0.12, rho=-0.45, m=0.03, sigma=0.18)
    w = qf.svi_total_variance(k, **truth)
    fit = qf.svi_fit(k, w, initial={"a": 0.01, "b": 0.2, "rho": 0.0, "m": 0.0, "sigma": 0.1})
    assert {key: fit[key] for key in truth} == pytest.approx(truth, rel=1e-6)
    assert fit["rmse"] < 1e-10 and fit["converged"]
    with pytest.raises(ValueError):
        qf.svi_fit(k, w, initial={"a": 0.01, "b": 0.2, "rho": 0.0, "m": 0.0})
    with pytest.raises(ValueError):
        qf.svi_fit(k, w, initial={"a": 0.01, "b": -0.2, "rho": 0.0, "m": 0.0, "sigma": 0.1})


def calls_from_total_variance(total, strikes, maturities, spot, r, q):
    """Black-Scholes calls on a grid from a total-variance surface w(y, T), y = ln(K / F_T)."""
    forward = spot * np.exp((r - q) * maturities)[:, None]
    y = np.log(strikes[None, :] / forward)
    w = total(y, maturities[:, None])
    return qf.bs_price("call", spot, strikes[None, :], maturities[:, None], r, q, np.sqrt(w / maturities[:, None]))


def test_dupire_recovers_a_flat_vol_and_an_instantaneous_term_structure():
    """Dupire (1994): a flat implied vol is its own local vol; with sigma_imp^2 T = a T + c T^2 (no smile) the local
    variance is the instantaneous a + 2 c T at every strike. Interior points of a fine grid, to the finite
    differences' error (strikes 80 to 130, maturities 0.5 to 1.975); the edges are NaN."""
    strikes, maturities = np.linspace(50.0, 160.0, 111), np.linspace(0.2, 2.0, 73)
    flat = calls_from_total_variance(lambda y, t: 0.04 * t + 0 * y, strikes, maturities, 100.0, 0.03, 0.01)
    local = qf.dupire_local_vol(flat, strikes=strikes, maturities=maturities, rate=0.03, dividend_yield=0.01)
    assert np.isnan(local[0]).all() and np.isnan(local[-1]).all() and np.isnan(local[:, [0, -1]]).all()
    core = local[12:-1, 30:-30]
    assert core == pytest.approx(np.full(core.shape, 0.2), abs=3e-4)
    term = calls_from_total_variance(lambda y, t: 0.03 * t + 0.01 * t * t + 0 * y, strikes, maturities, 100.0,
                                     0.02, 0.0)
    local = qf.dupire_local_vol(term, strikes=strikes, maturities=maturities, rate=0.02, dividend_yield=0.0)
    instantaneous = np.sqrt(0.03 + 0.02 * maturities[12:-1])[:, None]
    assert local[12:-1, 30:-30] == pytest.approx(np.broadcast_to(instantaneous, (60, 51)), abs=5e-4)
    uneven_k, uneven_t = np.geomspace(50.0, 160.0, 121), np.sqrt(np.linspace(0.04, 4.0, 80))
    term = calls_from_total_variance(lambda y, t: 0.03 * t + 0.01 * t * t + 0 * y, uneven_k, uneven_t, 100.0,
                                     0.02, 0.01)
    local = qf.dupire_local_vol(term, strikes=uneven_k, maturities=uneven_t, rate=0.02, dividend_yield=0.01)
    rows, cols = (uneven_t >= 0.5) & (uneven_t < 1.99), (uneven_k >= 80) & (uneven_k <= 130)
    assert local[rows][:, cols] == pytest.approx(np.broadcast_to(np.sqrt(0.03 + 0.02 * uneven_t[rows])[:, None],
                                                                 (rows.sum(), cols.sum())), abs=5e-4)
    with pytest.raises(ValueError):
        qf.dupire_local_vol(flat, strikes=strikes[::-1], maturities=maturities, rate=0.03, dividend_yield=0.01)


def test_dupire_on_an_svi_surface_is_gatherals_local_variance_in_total_variance():
    """Gatheral, The Volatility Surface (2006), eq. 1.10: with w(y, T) the total implied variance at y = ln(K/F_T),
    sigma_loc^2 = w_T / [1 - (y/w) w_y + (1/4)(-1/4 - 1/w + y^2/w^2) w_y^2 + w_yy / 2]. Calls priced from w = T
    svi(y) with rates and a yield; Dupire's finite differences in strike and maturity on the prices match it."""
    a, b, rho, m, s = 0.03, 0.08, -0.4, 0.0, 0.3
    svi = lambda y: a + b * (rho * (y - m) + np.sqrt((y - m) ** 2 + s * s))  # noqa: E731
    d1 = lambda y: b * (rho + (y - m) / np.sqrt((y - m) ** 2 + s * s))  # noqa: E731
    d2 = lambda y: b * s * s / ((y - m) ** 2 + s * s) ** 1.5  # noqa: E731
    spot, r, q = 100.0, 0.03, 0.015
    strikes, maturities = np.linspace(60.0, 150.0, 181), np.linspace(0.5, 1.5, 41)
    calls = calls_from_total_variance(lambda y, t: t * svi(y), strikes, maturities, spot, r, q)
    local = qf.dupire_local_vol(calls, strikes=strikes, maturities=maturities, rate=r, dividend_yield=q)
    t = maturities[:, None]
    y = np.log(strikes[None, :] / (spot * np.exp((r - q) * t)))
    w, w_y, w_yy = t * svi(y), t * d1(y), t * d2(y)
    gatheral = svi(y) / (1 - y / w * w_y + 0.25 * (-0.25 - 1 / w + y * y / (w * w)) * w_y ** 2 + 0.5 * w_yy)
    inner = (slice(1, -1), slice(40, -40))
    assert local[inner] == pytest.approx(np.sqrt(gatheral[inner]), rel=2e-3)
