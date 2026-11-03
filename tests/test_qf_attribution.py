"""agent/qf/metrics.py's additions (lane lib-signals1; docs/research/library-syllabus-1011.md rows 61 and 104, the
census's two variants): Carino's linked effects adding to the active return in each of its forms (Carino 1999) over
Brinson-Fachler's per-period effects; the Sharpe ratio's standard error on Lo's (2002) Table 1, the scale factor of
his eq. (20) on his Table 2, the GMM error at no lags being the moment form, and the deflated Sharpe ratio of Bailey
and Lopez de Prado's (2014) example, 0.9004 at 100 trials and 0.9505 at 46; Sortino's compounded numerator and the
rebalance path with weights held fixed, each written out; the sharpe_ratio card's claim on scipy's and pandas'
moments. Returns are this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf

pytestmark = pytest.mark.fast

WP = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]])
WB = np.array([[0.4, 0.4, 0.2], [0.4, 0.3, 0.3], [0.5, 0.3, 0.2]])
RP = np.array([[0.04, -0.02, 0.01], [-0.03, 0.02, 0.05], [0.02, 0.01, -0.01]])
RB = np.array([[0.03, -0.01, 0.015], [-0.02, 0.025, 0.03], [0.015, 0.012, 0.0]])


def brinson_by_period():
    rows = [qf.brinson_fachler(portfolio_weights=WP[t], benchmark_weights=WB[t], portfolio_returns=RP[t],
                               benchmark_returns=RB[t]) for t in range(3)]
    effects = np.array([[row["total_allocation"], row["total_selection"], row["total_interaction"]] for row in rows])
    return (WP * RP).sum(axis=1), (WB * RB).sum(axis=1), effects


def test_carinos_linked_effects_add_to_the_active_return_in_each_form():
    rp, rb, effects = brinson_by_period()
    total_p, total_b = np.prod(1 + rp) - 1, np.prod(1 + rb) - 1
    targets = {"arithmetic": total_p - total_b, "log": math.log((1 + total_p) / (1 + total_b)),
               "geometric": (1 + total_p) / (1 + total_b) - 1}
    for active, target in targets.items():
        found = qf.carino_linking(portfolio_returns=rp, benchmark_returns=rb, effects=effects, active=active)
        assert found["linked"].sum() == pytest.approx(target, abs=1e-15), active
        assert found["active_return"] == pytest.approx(target, abs=1e-15)
    k_t = (np.log1p(rp) - np.log1p(rb)) / (rp - rb)
    k = (math.log1p(total_p) - math.log1p(total_b)) / (total_p - total_b)
    found = qf.carino_linking(portfolio_returns=rp, benchmark_returns=rb, effects=effects, active="arithmetic")
    assert found["coefficients"] == pytest.approx(k_t / k, rel=1e-14)
    assert found["linked"] == pytest.approx((k_t / k) @ effects, rel=1e-14)


def test_one_period_links_to_itself_and_equal_returns_take_the_limit():
    rp, rb, effects = brinson_by_period()
    one = qf.carino_linking(portfolio_returns=rp[:1], benchmark_returns=rb[:1], effects=effects[:1],
                            active="arithmetic")
    assert one["coefficients"] == pytest.approx([1.0], abs=1e-15) and one["linked"] == pytest.approx(effects[0])
    same = qf.carino_linking(portfolio_returns=[0.02, 0.01], benchmark_returns=[0.02, -0.01],
                             effects=[[0.0, 0.0], [0.01, 0.01]], active="log")
    assert same["coefficients"][0] == pytest.approx(1 / 1.02, rel=1e-14)


# ------------------------------------------------------------------------------------------------ Sharpe inference


def scaled(sr, n, ddof=0):
    """n returns whose mean over their std (ddof) is exactly sr: normal quantiles standardised, shifted by sr."""
    z = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    z = (z - z.mean()) / z.std(ddof=ddof)
    return 0.01 * (sr + z)


def infer(returns, **changes):
    args = dict(periods_per_year=12, ddof=0, moments="normal", dependence="iid", lags=None, benchmark_sharpe=0.0,
                trials=None, trials_variance=None)
    return qf.sharpe_inference(returns=returns, **{**args, **changes})


@pytest.mark.parametrize("sr, n, se", [(1.0, 60, 0.158), (2.0, 12, 0.500), (0.5, 250, 0.067), (3.0, 500, 0.105),
                                       (1.5, 125, 0.130)])
def test_the_standard_error_is_lo_2002_table_1(sr, n, se):
    found = infer(scaled(sr, n))
    assert found["sharpe"] == pytest.approx(sr, rel=1e-12)
    assert found["se"] == pytest.approx(se, abs=5e-4)
    assert found["se_annual"] == pytest.approx(found["se"] * math.sqrt(12)) and math.isnan(found["dsr"])


def test_the_autocorrelated_scale_factor_is_lo_eq_20_and_matches_his_table_2_under_ar1():
    """Lo's eq. (20), eta(q) = q / sqrt(q + 2 sum_{k<q} (q - k) rho_k): at q = 12 and AR(1) autocorrelations rho^k it
    is 2.88 at rho = 0.2 and 4.17 at rho = -0.2 (Table 2); the function takes the series' own autocorrelations."""
    def eta(q, rho):
        return q / math.sqrt(q + 2 * sum((q - k) * rho[k - 1] for k in range(1, q)))
    assert eta(12, [0.2 ** k for k in range(1, 12)]) == pytest.approx(2.88, abs=5e-3)
    assert eta(12, [(-0.2) ** k for k in range(1, 12)]) == pytest.approx(4.17, abs=5e-3)
    rng = np.random.default_rng(61)
    r = np.zeros(240)
    for t in range(1, 240):
        r[t] = 0.3 * r[t - 1] + rng.normal(0.004, 0.02)
    centred = r - r.mean()
    rho = [centred[k:] @ centred[:-k] / (centred @ centred) for k in range(1, 12)]
    found = infer(r, dependence="autocorrelated", lags=3)
    assert found["sharpe_annual"] / found["sharpe"] == pytest.approx(eta(12, rho), rel=1e-12)


def test_the_gmm_error_at_no_lags_is_the_moment_form_on_the_sample_moments():
    """Lo's GMM variance with no lags is 1 - g3 SR + (g4 - 1)/4 SR^2 on the biased sample moments (Mertens 2002)."""
    r = np.random.default_rng(5).standard_t(5, size=200) * 0.02 + 0.003
    gmm = infer(r, dependence="autocorrelated", lags=0)
    moment = infer(r, moments="sample")
    assert gmm["se"] == pytest.approx(moment["se"], rel=1e-12)
    assert moment["skew"] == pytest.approx(stats.skew(r)) and moment["kurtosis"] == pytest.approx(
        stats.kurtosis(r, fisher=False))
    adjusted = infer(r, moments="sample_unbiased")
    g3, g4, sr = stats.skew(r, bias=False), stats.kurtosis(r, fisher=False, bias=False), adjusted["sharpe"]
    assert (adjusted["skew"], adjusted["kurtosis"]) == pytest.approx((g3, g4), rel=1e-12)
    assert adjusted["se"] == pytest.approx(math.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr * sr) / 200), rel=1e-12)
    phi = np.column_stack([r - r.mean(), (r - r.mean()) ** 2 - r.var()])
    long_run = phi.T @ phi / 200
    for lag in (1, 2):
        cross = phi[lag:].T @ phi[:-lag] / 200
        long_run += (1 - lag / 3) * (cross + cross.T)                   # Newey-West's Bartlett weights at two lags
    gradient = np.array([1 / r.std(), -r.mean() / (2 * r.var() ** 1.5)])
    assert infer(r, dependence="autocorrelated", lags=2)["se"] == pytest.approx(
        math.sqrt(gradient @ long_run @ gradient / 200), rel=1e-12)


def test_autocorrelated_inference_does_not_report_iid_probabilities():
    """Lo (2002), annual Sharpe scale and its joint delta-method error differ from the iid PSR assumption.
    A dependent estimate must not label the iid benchmark probability as its own."""
    rng = np.random.default_rng(1972)
    r = np.zeros(240)
    for t in range(1, r.size):
        r[t] = 0.6 * r[t - 1] + rng.normal(0.004, 0.02)
    estimate = infer(r, dependence="autocorrelated", lags=3)
    found = infer(r, dependence="autocorrelated", lags=3, benchmark_sharpe=estimate["sharpe_annual"],
                  trials=10, trials_variance=0.5)
    assert math.isnan(found["psr"]) and math.isnan(found["dsr"])
    assert np.isfinite(found["se"]) and np.isfinite(found["sr0"])


def test_the_deflated_sharpe_ratio_is_bailey_and_lopez_de_prados_example():
    """An annual SR of 2.5 over 1,250 daily returns (250 a year), skewness -3 and kurtosis 10, after 100 trials whose
    annual Sharpe ratios have variance 1/2: DSR 0.9004, the expected maximum 1.789 a year (0.1132 a day); after 46
    trials 0.9505. PSR against 0 is the probability the true SR is positive."""
    r = scaled(2.5 / math.sqrt(250), 1250)
    for trials, dsr in ((100, 0.9004), (46, 0.9505)):
        found = infer(r, periods_per_year=250, moments=(-3.0, 10.0), trials=trials, trials_variance=0.5)
        assert found["dsr"] == pytest.approx(dsr, abs=5e-5), trials
    found = infer(r, periods_per_year=250, moments=(-3.0, 10.0), trials=100, trials_variance=0.5)
    assert found["sr0"] == pytest.approx(1.7894, abs=5e-5) and found["sr0"] / math.sqrt(250) == pytest.approx(
        0.1132, abs=5e-5)
    sr = 2.5 / math.sqrt(250)
    assert found["psr"] == pytest.approx(stats.norm.cdf(sr * math.sqrt(1249) / math.sqrt(1 + 3 * sr + 9 / 4 * sr ** 2)))
    short = infer(scaled(0.2, 24), benchmark_sharpe=0.12 * math.sqrt(12))    # SR* of 0.12 a month, 24 months
    assert short["psr"] == pytest.approx(stats.norm.cdf(0.08 * math.sqrt(23) / math.sqrt(1 + 0.02)), rel=1e-12)


# ------------------------------------------------------------------------------------------------ the variants

MONTHLY = np.array([0.012, -0.004, 0.021, 0.008, -0.017, 0.015, 0.003, -0.009, 0.026, 0.011, -0.002, 0.007])


def test_sortinos_compounded_numerator_is_the_annual_growth_less_the_effective_annual_rate():
    rf = 0.001
    below = np.minimum(MONTHLY - rf, 0.0)
    downside = math.sqrt(np.sum(below ** 2) / 12)
    growth = np.prod(1 + MONTHLY) ** (12 / 12) - 1 - ((1 + rf) ** 12 - 1)
    found = qf.sortino_ratio(returns=MONTHLY, risk_free=rf, risk_free_basis="per_period", periods_per_year=12, ddof=0,
                             numerator="compounded")
    assert found == pytest.approx(growth / (downside * math.sqrt(12)), rel=1e-12)
    yearly = qf.sortino_ratio(returns=MONTHLY, risk_free=0.03, risk_free_basis="annual_compound", periods_per_year=12,
                              ddof=0, numerator="compounded")
    monthly_rf = 1.03 ** (1 / 12) - 1
    below = np.minimum(MONTHLY - monthly_rf, 0.0)
    assert yearly == pytest.approx((np.prod(1 + MONTHLY) - 1 - 0.03) / (math.sqrt(np.sum(below ** 2) / 12)
                                                                         * math.sqrt(12)), rel=1e-12)


def test_weights_held_fixed_earn_the_target_mix_every_bar_and_trade_only_when_it_moves():
    r = np.array([[0.10, -0.10], [0.05, 0.02], [-0.03, 0.04], [0.02, 0.01]])
    fixed = qf.rebalance_path(returns=r, weights=[0.5, 0.5], rebalance=[True, False, False, False],
                              initial_value=1000.0, cash_return=0.0, timing="start", fee_order="before",
                              turnover="two_way", turnover_base="drifted", cost_rate=0.001, start="invested",
                              between_rebalances="fixed")
    assert fixed["gross_returns"] == pytest.approx(r @ [0.5, 0.5], abs=1e-15)
    assert fixed["drifted"] == pytest.approx(np.full((4, 2), 0.5), abs=0) and fixed["turnover"].sum() == 0.0
    moved = qf.rebalance_path(returns=r, weights=[[0.5, 0.5], [0.5, 0.5], [0.7, 0.3], [0.7, 0.3]],
                              rebalance=[True, True, True, True], initial_value=1000.0, cash_return=0.0,
                              timing="start", fee_order="before", turnover="two_way", turnover_base="drifted",
                              cost_rate=0.0, start="invested", between_rebalances="fixed")
    assert moved["turnover"].tolist() == pytest.approx([0.0, 0.0, 0.4, 0.0], abs=1e-15)


def test_the_sharpe_cards_moment_line_scipy_biased_and_pandas_adjusted():
    r = np.random.default_rng(2).normal(size=30)
    series = pd.Series(r)
    assert stats.skew(r, bias=False) == pytest.approx(series.skew(), rel=1e-12)
    assert stats.kurtosis(r, fisher=True, bias=False) == pytest.approx(series.kurt(), rel=1e-12)
    assert stats.skew(r) != pytest.approx(series.skew(), rel=1e-6)
    doc = qf.sharpe_ratio.__doc__
    assert all(term in doc for term in ("Kelly's fraction", "Treynor's ratio", "Jensen's alpha", "M^2"))
