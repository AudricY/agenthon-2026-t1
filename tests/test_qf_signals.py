"""agent/qf/signals.py (lane lib-signals1; docs/research/library-syllabus-1011.md row 41, docs/research/fresh-gaps-1012.md
row 5): winsorizing against pandas' clip at quantiles and scipy's mstats.winsorize; the cross-sectional z-score's
moments each date, equal and cap-weighted, and Barra's standardise, truncate, standardise written out; ranks against
pandas' rank and their sum N(N + 1)/2; residual momentum's identities (a stock that is an exact mix of the factors scores
zero; with no factors it is plain 12-1 momentum) and one regression written out. The panels are this file's own; no
unit's values appear (rule 8)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import mstats

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf

pytestmark = pytest.mark.fast

RNG = np.random.default_rng(41)
PANEL = RNG.standard_t(3, size=(6, 40))          # six dates of forty stocks, heavy tails
PANEL[2, 5] = np.nan
CAPS = RNG.lognormal(8.0, 1.5, size=(6, 40))


def zscore(signal, **changes):
    args = dict(weights="equal", scale="equal", ddof=1, clip="none", limits=None, rounds=1, quantile_method=None)
    return qf.cross_sectional_zscore(signal=signal, **{**args, **changes})


# ------------------------------------------------------------------------------------------------ winsorize


def test_quantile_clipping_is_pandas_clip_at_its_quantiles_each_date_and_a_nan_stays():
    found = qf.winsorize(values=PANEL, clip="quantile", limits=(0.05, 0.95), ddof=None, quantile_method="linear")
    for t in range(PANEL.shape[0]):
        row = pd.Series(PANEL[t])
        expected = row.clip(row.quantile(0.05), row.quantile(0.95)).to_numpy()
        assert np.allclose(found[t], expected, equal_nan=True, rtol=0, atol=1e-15), t
    assert np.isnan(found[2, 5]) and found.shape == PANEL.shape


def test_count_clipping_is_scipys_mstats_winsorize():
    """scipy.stats.mstats.winsorize(x, limits=(a, b)) replaces the floor(a n) smallest and floor(b n) largest values
    by the next value inward."""
    x = PANEL[0]
    for lower, upper in ((0.05, 0.05), (0.1, 0.2), (0.0, 0.25), (0.07, 0.13)):    # 2.8 and 5.2 of 40: floors
        expected = np.asarray(mstats.winsorize(x, limits=(lower, upper)))
        found = qf.winsorize(values=x, clip="count", limits=(lower, upper), ddof=None, quantile_method=None)
        assert found == pytest.approx(expected, abs=0), (lower, upper)


def test_sd_clipping_is_at_the_mean_less_and_plus_k_standard_deviations():
    x = PANEL[1]
    for ddof in (0, 1):
        m, s = x.mean(), x.std(ddof=ddof)
        found = qf.winsorize(values=x, clip="sd", limits=(2.0, 1.5), ddof=ddof, quantile_method=None)
        assert found == pytest.approx(np.clip(x, m - 2.0 * s, m + 1.5 * s), abs=1e-15)
    with pytest.raises(ValueError, match="ddof"):
        qf.winsorize(values=x, clip="sd", limits=(3.0, 3.0), ddof=None, quantile_method=None)
    with pytest.raises(ValueError, match="quantile levels"):
        qf.winsorize(values=x, clip="quantile", limits=(0.99, 0.01), ddof=None, quantile_method="linear")


def test_count_clipping_floors_even_immediately_below_an_integer_tail_count():
    """SciPy mstats.winsorize's inclusive=True rule truncates share * n, without rounding up near an integer."""
    x = np.arange(10.0)
    share = np.nextafter(0.2, 0.0)
    expected = np.asarray(mstats.winsorize(x, limits=(share, share)))
    found = qf.winsorize(values=x, clip="count", limits=(share, share), ddof=None, quantile_method=None)
    assert found == pytest.approx(expected, abs=0) and found[0] == 1.0 and found[-1] == 8.0


def test_quantile_clipping_and_zscores_respect_the_named_percentile_rule():
    """Hyndman and Fan (1996): the lower and higher sample quantiles select different order statistics.
    A z-score after clipping must preserve the named clipping rule."""
    x = np.array([0.0, 1.0, 2.0, 10.0])
    for method, clipped in (("lower", [0.0, 1.0, 2.0, 2.0]), ("higher", [1.0, 1.0, 2.0, 10.0])):
        found = qf.winsorize(values=x, clip="quantile", limits=(0.25, 0.75), ddof=None, quantile_method=method)
        assert found == pytest.approx(clipped, abs=0)
        expected = (np.asarray(clipped) - np.mean(clipped)) / np.std(clipped, ddof=1)
        assert zscore(x, clip="quantile", limits=(0.25, 0.75), quantile_method=method) == pytest.approx(expected)
    with pytest.raises(ValueError, match="quantile_method"):
        zscore(x, quantile_method="linear")


# ------------------------------------------------------------------------------------------------ z-scores


def test_z_scores_have_mean_0_and_sd_1_each_date_in_either_ddof():
    for ddof in (0, 1):
        z = zscore(PANEL, ddof=ddof)
        for t in range(PANEL.shape[0]):
            row = z[t][np.isfinite(z[t])]
            assert row.mean() == pytest.approx(0.0, abs=1e-14) and row.std(ddof=ddof) == pytest.approx(1.0, abs=1e-14)
    assert np.isnan(zscore(PANEL)[2, 5])
    one = zscore(PANEL[0])
    assert one.shape == (40,) and one == pytest.approx(zscore(PANEL)[0], abs=0)


def test_a_cap_weighted_mean_is_0_under_cap_weights_and_the_sd_is_the_equal_or_the_weighted_one():
    """The Barra handbooks' standardisation: the cap-weighted mean taken out, the equal-weighted standard deviation
    divided by; under 'weighted' the cap-weighted standard deviation is 1 instead."""
    z = zscore(PANEL, weights=CAPS)
    zw = zscore(PANEL, weights=CAPS, scale="weighted", ddof=0)
    for t in range(PANEL.shape[0]):
        kept = np.isfinite(PANEL[t])
        x, w = PANEL[t, kept], CAPS[t, kept]
        assert w @ z[t, kept] / w.sum() == pytest.approx(0.0, abs=1e-12)
        assert z[t, kept] == pytest.approx((x - w @ x / w.sum()) / x.std(ddof=1), abs=1e-12)
        assert w @ zw[t, kept] == pytest.approx(0.0, abs=1e-9)
        assert np.sqrt(w @ zw[t, kept] ** 2 / w.sum()) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="ddof must be 0"):
        zscore(PANEL, weights=CAPS, scale="weighted", ddof=1)


def test_one_round_is_standardise_truncate_at_k_standardise_again_written_out():
    """Barra's two standardisations with a truncation between them, written out, against one round of 'sd' clipping;
    a second round truncates and standardises the result once more."""
    k, x = 1.5, PANEL[0]
    first = (x - x.mean()) / x.std(ddof=1)
    cut = np.clip(first, -k, k)
    barra = (cut - cut.mean()) / cut.std(ddof=1)
    once = zscore(x, clip="sd", limits=(k, k), rounds=1)
    assert once == pytest.approx(barra, abs=1e-12) and np.abs(first).max() > k
    again = np.clip(barra, -k, k)
    assert zscore(x, clip="sd", limits=(k, k), rounds=2) == pytest.approx((again - again.mean()) / again.std(ddof=1),
                                                                          abs=1e-12)


def test_zero_total_mean_weights_are_refused():
    """A weighted mean divides by the total weight, so an all-zero cross-section cannot be standardised."""
    with pytest.raises(ValueError, match="positive sum"):
        zscore([1.0, 2.0, 3.0], weights=[0.0, 0.0, 0.0])


def test_quantile_clipping_then_standardising_is_winsorize_then_the_z_score():
    w = qf.winsorize(values=PANEL, clip="quantile", limits=(0.01, 0.99), ddof=None, quantile_method="linear")
    assert zscore(PANEL, clip="quantile", limits=(0.01, 0.99), quantile_method="linear") == pytest.approx(
        zscore(w), abs=1e-12, nan_ok=True)


# ------------------------------------------------------------------------------------------------ ranks


@pytest.mark.parametrize("ties, pandas_method", [("average", "average"), ("min", "min"), ("max", "max"),
                                                 ("dense", "dense"), ("ordinal", "first")])
def test_ranks_and_percentiles_are_pandas_rank_each_date(ties, pandas_method):
    x = np.round(PANEL, 0)                       # ties
    ranks = qf.cross_sectional_rank(signal=x, ties=ties, form="rank")
    pct = qf.cross_sectional_rank(signal=x, ties=ties, form="percentile")
    frame = pd.DataFrame(x)
    assert np.allclose(ranks, frame.rank(axis=1, method=pandas_method).to_numpy(), equal_nan=True, rtol=0, atol=0)
    assert np.allclose(pct, frame.rank(axis=1, method=pandas_method, pct=True).to_numpy(), equal_nan=True, rtol=0,
                       atol=1e-15)


def test_n_ranks_sum_to_n_n_plus_1_over_2_and_normal_scores_are_symmetric():
    x = np.round(PANEL[0], 0)
    n = x.size
    for ties in ("average", "ordinal"):
        assert qf.cross_sectional_rank(signal=x, ties=ties, form="rank").sum() == pytest.approx(n * (n + 1) / 2)
    scores = qf.cross_sectional_rank(signal=PANEL[1], ties="average", form="normal_score")
    assert scores.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.sort(scores) == pytest.approx(stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n), abs=1e-14)
    unit = qf.cross_sectional_rank(signal=PANEL[1], ties="average", form="zero_one")
    assert unit.min() == 0.0 and unit.max() == 1.0


def test_the_cross_sectional_correlation_of_two_signals_is_spearmans_as_the_pearson_of_ranks():
    a, b = PANEL[0], PANEL[0] + RNG.normal(size=40)
    ra, rb = (qf.cross_sectional_rank(signal=s, ties="average", form="rank") for s in (a, b))
    assert np.corrcoef(ra, rb)[0, 1] == pytest.approx(stats.spearmanr(a, b)[0], abs=1e-12)


# ------------------------------------------------------------------------------------------------ residual momentum

T, N = 30, 8
FACTORS = RNG.normal(0.005, 0.04, size=(T, 3))
NOISE = RNG.normal(0.0, 0.03, size=(T, N))
BETAS = RNG.normal(0.8, 0.4, size=(3, N))


def momentum(returns, factors, **changes):
    args = dict(risk_free=0.0, regression_window=24, formation_window=11, skip=1, intercept="dropped",
                aggregate="compound", ddof=None)
    return qf.residual_momentum(returns=returns, factors=factors, **{**args, **changes})


def test_a_stock_that_is_an_exact_mix_of_the_factors_scores_zero():
    returns = 0.002 + FACTORS @ BETAS + NOISE
    returns[:, 0] = 0.001 + FACTORS @ BETAS[:, 0]
    for aggregate in ("compound", "sum"):
        found = momentum(returns, FACTORS, aggregate=aggregate)
        assert np.all(np.isnan(found[:23])) and found[23:, 0] == pytest.approx(0.0, abs=1e-13), aggregate
        assert np.all(np.abs(found[23:, 1:]) > 0)
    kept = momentum(returns, FACTORS, intercept="kept", aggregate="sum")
    assert kept[23:, 0] == pytest.approx(11 * 0.001, abs=1e-13)        # the intercept left in: eleven of it


def test_with_no_factors_and_no_intercept_it_is_plain_12_1_momentum_from_prices():
    returns = 0.002 + NOISE
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    found = momentum(returns, None, regression_window=12, formation_window=11, skip=1, intercept="none")
    for t in range(12, T):
        assert found[t] == pytest.approx(prices[t - 1] / prices[t - 12] - 1.0, rel=1e-12), t


def test_one_score_written_out_from_its_regression_and_the_excess_returns():
    """Row t: the excess returns over rows t - 23 to t regressed on an intercept and the factors; the residuals of rows
    t - 11 to t - 1 (formation 11, skip 1) summed over their standard deviation (Blitz, Huij and Martens)."""
    returns = 0.002 + FACTORS @ BETAS + NOISE
    rf = np.linspace(0.0001, 0.0009, T)
    found = momentum(returns, FACTORS, risk_free=rf, aggregate="sum_over_sd", ddof=1)
    t, stock = 27, 3
    rows = slice(t - 23, t + 1)
    design = np.column_stack([np.ones(24), FACTORS[rows]])
    y = returns[rows, stock] - rf[rows]
    resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    e = resid[12:23]
    assert found[t, stock] == pytest.approx(e.sum() / e.std(ddof=1), rel=1e-10)
    returns[10, stock] = np.nan
    gapped = momentum(returns, FACTORS)
    assert np.isnan(gapped[10:10 + 24, stock]).all() and np.isfinite(gapped[10 + 24:, stock]).all()
    with pytest.raises(ValueError, match="inside the regression window"):
        momentum(returns, FACTORS, regression_window=10)
