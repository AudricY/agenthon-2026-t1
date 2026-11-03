"""agent/qf/metrics.py's information_coefficient (runs/lanes/lib6-prompt.md, rank 7 of
docs/research/library-syllabus-1011.md): each date's IC against scipy's Spearman and Pearson, the forward return's
lag and horizon written out, and the Newey-West t-statistic against a long-run variance computed here (Grinold and
Kahn ch. 6; Newey and West 1987). The panels are drawn here from a seed; no unit's values appear (rule 8)."""
import numpy as np
import pytest
from scipy import stats

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf
import qf.metrics as m

pytestmark = pytest.mark.fast

RNG = np.random.default_rng(20261011)
T, N = 40, 12
SIGNAL = RNG.standard_normal((T, N))
RETURNS = 0.01 * RNG.standard_normal((T, N))
RETURNS[1:] += 0.004 * SIGNAL[:-1]                     # the signal predicts next period's return a little
SIGNAL[3, :8] = np.nan                                 # a date left with four names
RETURNS[10, 2] = np.nan


def ic(**changes):
    args = dict(signal=SIGNAL, returns=RETURNS, method="spearman", lag=1, horizon=1, return_kind="simple",
                min_names=5, ddof=1, t_stat="iid", lags=None)
    return qf.information_coefficient(**{**args, **changes})


@pytest.mark.parametrize("method, reference", [("spearman", stats.spearmanr), ("pearson", stats.pearsonr)])
def test_each_dates_ic_is_scipys_correlation_with_the_next_periods_return(method, reference):
    found = ic(method=method)
    for t in range(T - 1):
        kept = np.isfinite(SIGNAL[t]) & np.isfinite(RETURNS[t + 1])
        if kept.sum() < 5:
            assert np.isnan(found["ic"][t])
            continue
        assert found["ic"][t] == pytest.approx(reference(SIGNAL[t, kept], RETURNS[t + 1, kept])[0], abs=1e-12)
    assert np.isnan(found["ic"][3]) and np.isnan(found["ic"][-1])        # four names; no next period
    used = found["ic"][np.isfinite(found["ic"])]
    assert found["n_dates"] == used.size == T - 2
    assert found["mean"] == pytest.approx(used.mean()) and found["std"] == pytest.approx(used.std(ddof=1))
    assert found["icir"] == pytest.approx(used.mean() / used.std(ddof=1))
    assert found["t_stat"] == pytest.approx(used.mean() / (used.std(ddof=1) / np.sqrt(used.size)))


def test_the_forward_return_over_a_lag_and_a_horizon_compounds_or_sums():
    """lag 2, horizon 3: the signal at t against (1 + r_{t+2})(1 + r_{t+3})(1 + r_{t+4}) - 1, or the log returns'
    sum."""
    t = 5
    window = RETURNS[t + 2:t + 5]
    kept = np.isfinite(SIGNAL[t]) & np.all(np.isfinite(window), axis=0)
    simple = np.prod(1.0 + window[:, kept], axis=0) - 1.0
    assert ic(lag=2, horizon=3, method="pearson")["ic"][t] == pytest.approx(
        stats.pearsonr(SIGNAL[t, kept], simple)[0], abs=1e-12)
    assert ic(lag=2, horizon=3, method="pearson", return_kind="log")["ic"][t] == pytest.approx(
        stats.pearsonr(SIGNAL[t, kept], window[:, kept].sum(axis=0))[0], abs=1e-12)
    assert np.all(np.isnan(ic(lag=2, horizon=3)["ic"][-4:]))
    assert ic(lag=0)["ic"][7] == pytest.approx(stats.spearmanr(SIGNAL[7], RETURNS[7])[0], abs=1e-12)


def test_a_signal_against_itself_has_an_ic_of_one_and_its_negative_minus_one():
    """Grinold and Kahn: the IC of a perfect forecast is 1; flipping the signal flips it."""
    clean = np.nan_to_num(SIGNAL)
    returns = np.vstack([np.zeros(N), clean[:-1] * 0.01])                 # next period's return is the signal
    for method in ("spearman", "pearson"):
        found = ic(signal=clean, returns=returns, method=method)
        assert found["ic"][:-1] == pytest.approx(1.0, abs=1e-12)
        assert ic(signal=-clean, returns=returns, method=method)["mean"] == pytest.approx(-1.0, abs=1e-12)


def test_spearman_averages_tied_ranks():
    signal = np.array([[1.0, 1.0, 2.0, 3.0, 3.0, 4.0]] * 4)
    returns = np.array([[0.01, 0.03, 0.02, 0.02, 0.05, 0.04]] * 4)
    found = qf.information_coefficient(signal, returns, "spearman", 0, 1, "simple", 3, 0, "iid", None)
    assert found["ic"][0] == pytest.approx(stats.spearmanr(signal[0], returns[0])[0], abs=1e-12)


@pytest.mark.parametrize("lags", [0, 1, 4])
def test_the_newey_west_t_statistic_uses_the_bartlett_long_run_variance(lags):
    """V = gamma_0 + 2 sum_{l=1}^{L} (1 - l/(L + 1)) gamma_l, gamma_l = sum_t (x_t - m)(x_{t-l} - m) / n, written as
    loops here; t = mean / sqrt(V / n). At L = 0 it is the variance over n."""
    found = ic(t_stat="newey_west", lags=lags, horizon=3)
    x = found["ic"][np.isfinite(found["ic"])]
    n, mean = x.size, x.mean()
    gamma = [sum((x[t] - mean) * (x[t - k] - mean) for t in range(k, n)) / n for k in range(lags + 1)]
    variance = gamma[0] + 2 * sum((1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))
    assert found["t_stat"] == pytest.approx(mean / np.sqrt(variance / n), rel=1e-12)
    assert found["std"] == pytest.approx(x.std(ddof=1))


@pytest.mark.parametrize("bad, match", [
    (dict(method="kendall"), "method"), (dict(return_kind="excess"), "return_kind"), (dict(t_stat="hac"), "t_stat"),
    (dict(t_stat="newey_west"), "lags"), (dict(lags=2), "lags"), (dict(lag=-1), "lag"), (dict(horizon=0), "horizon"),
    (dict(min_names=2), "min_names"), (dict(ddof=2), "ddof"), (dict(returns=RETURNS[:, :5]), "T x N")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        ic(**bad)
