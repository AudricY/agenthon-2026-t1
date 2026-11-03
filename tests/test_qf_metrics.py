"""agent/qf/metrics.py, the backtest metrics shelf (runs/lanes/lib2-prompt.md): every function against its defining
formula computed here another way (plain Python's statistics module, explicit loops, a cash-and-shares simulation), a
worked example, or the identity the method must satisfy (Maillard, Roncalli and Teiletche's two-asset and uncorrelated
risk parity), each source named in its test. No unit's instruction values, checks, expected values or data appear
here (rules 8 and 9): the parameters are chosen freely."""
import datetime
import math
import statistics

import numpy as np
import pytest

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf
import qf.metrics as m

pytestmark = pytest.mark.fast

MONTHLY = [0.012, -0.004, 0.021, 0.008, -0.017, 0.015, 0.003, -0.009, 0.026, 0.011, -0.002, 0.007]


def test_the_metrics_cards_keep_the_format_and_their_examples_run():
    check_cards(m)


# ------------------------------------------------------------------------------------------------ Sharpe, Sortino


def test_sharpe_ratio_is_the_mean_excess_over_its_deviation_in_each_basis_and_ddof():
    """Sharpe (1994): sqrt(P) mean(r - rf) / std(r - rf), the standard deviation from Python's statistics module
    (stdev for ddof 1, pstdev for ddof 0); each basis converts the annual rate to the per-period one it names."""
    rf_year = 0.03
    per_period = {"annual_simple": rf_year / 12, "annual_compound": 1.03 ** (1 / 12) - 1,
                  "annual_continuous": math.exp(rf_year / 12) - 1}
    for basis, rf in per_period.items():
        excess = [r - rf for r in MONTHLY]
        for ddof, spread in ((1, statistics.stdev(excess)), (0, statistics.pstdev(excess))):
            expected = math.sqrt(12) * statistics.fmean(excess) / spread
            assert m.sharpe_ratio(returns=MONTHLY, risk_free=rf_year, risk_free_basis=basis, periods_per_year=12,
                                  ddof=ddof) == pytest.approx(expected, rel=1e-12), (basis, ddof)
            assert m.sharpe_ratio(returns=MONTHLY, risk_free=rf, risk_free_basis="per_period", periods_per_year=12,
                                  ddof=ddof) == pytest.approx(expected, rel=1e-12)
    varying = np.linspace(0.001, 0.003, 12)
    excess = [r - f for r, f in zip(MONTHLY, varying)]
    assert m.sharpe_ratio(returns=MONTHLY, risk_free=varying, risk_free_basis="per_period", periods_per_year=12,
                          ddof=1) == pytest.approx(math.sqrt(12) * statistics.fmean(excess) / statistics.stdev(excess))
    for bad, name in ((dict(risk_free_basis="annual"), "risk_free_basis"), (dict(ddof=2), "ddof"),
                      (dict(returns=[0.01, 0.01, 0.01]), "dispersion"), (dict(periods_per_year=0), "periods_per_year"),
                      (dict(risk_free=[0.0, 0.0]), "risk_free"), (dict(returns=[0.01, float("nan")]), "returns")):
        with pytest.raises(ValueError, match=name):
            m.sharpe_ratio(**(dict(returns=MONTHLY, risk_free=0.0, risk_free_basis="per_period", periods_per_year=12,
                                   ddof=1) | bad))


def test_sortino_ratio_averages_the_shortfall_over_every_period():
    """Sortino and Price (1994): the downside deviation sqrt(sum min(r - T, 0)^2 / (n - ddof)) runs over all n
    periods; written out here as a loop. A series that never falls below the target is refused."""
    target = 0.002
    shortfall = sum(min(r - target, 0.0) ** 2 for r in MONTHLY)
    for ddof in (0, 1):
        expected = math.sqrt(12) * (statistics.fmean(MONTHLY) - target) / math.sqrt(shortfall / (12 - ddof))
        assert m.sortino_ratio(returns=MONTHLY, risk_free=target, risk_free_basis="per_period", periods_per_year=12,
                               ddof=ddof, numerator="arithmetic") == pytest.approx(expected, rel=1e-12)
    losers_only = math.sqrt(shortfall / sum(r < target for r in MONTHLY))
    assert m.sortino_ratio(returns=MONTHLY, risk_free=target, risk_free_basis="per_period", periods_per_year=12,
                           ddof=0, numerator="arithmetic") != pytest.approx(math.sqrt(12) * (statistics.fmean(MONTHLY) - target) / losers_only)
    with pytest.raises(ValueError, match="unbounded"):
        m.sortino_ratio(returns=[0.01, 0.02, 0.0], risk_free=0.0, risk_free_basis="per_period", periods_per_year=12,
                        ddof=0, numerator="arithmetic")


# ------------------------------------------------------------------------------------------------ drawdowns, CAGR


def test_max_drawdown_dates_finds_the_peak_trough_and_recovery():
    """A worked series: highs 120 and then 130; the fall from 130 to 80 is 50/130, the deepest; the series regains
    130 at its last point. A flat top keeps the last date at the high; a series that never regains its high has no
    recovery; a rising series has no drawdown."""
    values = [100.0, 120.0, 90.0, 95.0, 130.0, 130.0, 80.0, 85.0, 131.0]
    dates = [f"2024-0{i + 1}-28" for i in range(9)]
    found = m.max_drawdown_dates(values=values, dates=dates)
    assert found["max_drawdown"] == pytest.approx(50 / 130)
    assert (found["peak"], found["trough"], found["recovery"]) == ("2024-06-28", "2024-07-28", "2024-09-28")
    assert found["drawdowns"] == pytest.approx([0, 0, 0.25, 25 / 120, 0, 0, 50 / 130, 45 / 130, 0])
    lost = m.max_drawdown_dates(values=values[:-1], dates=list(range(8)))
    assert (lost["peak"], lost["trough"], lost["recovery"]) == (5, 6, None)
    days = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    shallow = m.max_drawdown_dates(values=[10.0, 9.0, 10.0], dates=days)
    assert shallow["trough"] == datetime.date(2024, 1, 3) and shallow["recovery"] == datetime.date(2024, 1, 4)
    assert type(shallow["trough"]) is datetime.date     # a plain date, which json and str() take as one
    assert m.max_drawdown_dates(values=[1.0, 2.0, 3.0], dates=[0, 1, 2])["peak"] is None
    with pytest.raises(ValueError, match="levels"):
        m.max_drawdown_dates(values=[0.01, -0.02, 0.03], dates=[0, 1, 2])
    with pytest.raises(ValueError, match="dates"):
        m.max_drawdown_dates(values=[1.0, 2.0], dates=[0])


def test_cagr_compounds_and_the_arithmetic_mean_does_not():
    """A constant return g a period compounds to (1 + g)^P - 1 a year; doubling over 24 months is sqrt(2) - 1 a
    year; the arithmetic annual mean is P times the mean, and compounding that mean exceeds the CAGR when returns
    vary (the arithmetic-geometric mean inequality on 1 + r)."""
    assert m.cagr(returns=[0.01] * 36, periods_per_year=12) == pytest.approx(1.01 ** 12 - 1, rel=1e-13)
    doubling = [2 ** (1 / 24) - 1] * 24
    assert m.cagr(returns=doubling, periods_per_year=12) == pytest.approx(math.sqrt(2) - 1, rel=1e-13)
    assert m.annualized_mean_return(returns=MONTHLY, periods_per_year=12) == pytest.approx(12 * statistics.fmean(MONTHLY))
    growth = math.prod(1 + r for r in MONTHLY)
    assert m.cagr(returns=MONTHLY, periods_per_year=12) == pytest.approx(growth - 1, rel=1e-13)
    assert m.cagr(returns=MONTHLY * 2, periods_per_year=252) == pytest.approx(growth ** (2 * 252 / 24) - 1, rel=1e-12)
    assert m.cagr(returns=MONTHLY, periods_per_year=12) < (1 + statistics.fmean(MONTHLY)) ** 12 - 1
    with pytest.raises(ValueError, match="above -1"):
        m.cagr(returns=[0.1, -1.0], periods_per_year=12)


def test_calmar_ratio_divides_the_named_annual_return_by_the_drawdown_from_the_start():
    """Young (1991): the annual return over the maximum drawdown; the wealth starts at 1 before the first return, so
    the opening loss of 4% is the drawdown here (a series starting at 1 - 0.04 would see only the later 3%)."""
    returns = [-0.04, 0.02, 0.03, -0.03, 0.05, 0.01]
    wealth = np.cumprod([1.0] + [1 + r for r in returns])
    worst = max(1 - w / max(wealth[:i + 1]) for i, w in enumerate(wealth))
    assert worst == pytest.approx(0.04)
    for kind, annual in (("cagr", wealth[-1] ** (12 / 6) - 1), ("arithmetic", 12 * statistics.fmean(returns))):
        assert m.calmar_ratio(returns=returns, periods_per_year=12, annual_return=kind) == pytest.approx(annual / worst)
    with pytest.raises(ValueError, match="never draw down"):
        m.calmar_ratio(returns=[0.01, 0.02], periods_per_year=12, annual_return="cagr")
    with pytest.raises(ValueError, match="annual_return"):
        m.calmar_ratio(returns=returns, periods_per_year=12, annual_return="mean")


# ------------------------------------------------------------------------------------------------ backtests

CLOSE = np.array([50.0, 50.5, 49.8, 51.2, 52.0, 51.1, 50.4, 51.9, 53.0, 52.2])
OPEN = np.array([49.9, 50.2, 50.6, 50.1, 51.5, 51.8, 50.9, 50.7, 52.4, 53.3])
LONG_FLAT = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=float)


def shares_and_cash(signals, close, open_, lag):
    """An independent long-only simulation in shares and cash: the signal at close t fills at the open of bar
    t + lag with all the wealth (fractional shares), valued at each close."""
    cash, shares, wealth = 1.0, 0.0, [1.0]
    for t in range(1, len(close)):
        target = signals[t - lag] if t >= lag else 0.0
        value = cash + shares * open_[t]
        if (shares > 0) != (target > 0):
            shares, cash = (value / open_[t], 0.0) if target > 0 else (0.0, value)
        wealth.append(cash + shares * close[t])
    return np.array(wealth[1:]) / np.array(wealth[:-1]) - 1.0


def test_close_execution_is_the_lagged_signal_times_the_bars_return():
    """'close' with lag L: bar t earns s_{t-L} (C_t / C_{t-1} - 1), the shift(L) backtest, written as a loop here."""
    signals = np.array([1, 1, -1, 0, 0.5, 1, -1, -1, 0, 1], dtype=float)
    for lag in (1, 2):
        found = m.backtest_signals(signals=signals, close=CLOSE, open=None, lag=lag, execution="close", cost_bps=0.0)
        expected = [0.0] + [(signals[t - lag] if t >= lag else 0.0) * (CLOSE[t] / CLOSE[t - 1] - 1)
                            for t in range(1, 10)]
        assert found["returns"] == pytest.approx(expected, abs=1e-15)
        assert found["positions"][:lag].tolist() == [0.0] * lag and found["positions"][lag:].tolist() == \
            signals[:-lag].tolist()


def test_next_open_execution_is_a_cash_and_shares_backtest_and_costs_are_per_turn():
    """A long-only signal filled at the next open is the share-and-cash simulation above, bar by bar. A cost of c bps
    takes c / 10,000 times the turnover from each bar's return; a round trip on a flat price costs 2c bps; a flip from
    long to short turns over 2; a short loses what the price gains in the session."""
    for lag, turns in ((1, 4), (2, 3)):
        found = m.backtest_signals(signals=LONG_FLAT, close=CLOSE, open=OPEN, lag=lag, execution="next_open",
                                   cost_bps=0.0)
        assert found["returns"][1:] == pytest.approx(shares_and_cash(LONG_FLAT, CLOSE, OPEN, lag), abs=1e-14), lag
        assert found["returns"][0] == 0.0
        costly = m.backtest_signals(signals=LONG_FLAT, close=CLOSE, open=OPEN, lag=lag, execution="next_open",
                                    cost_bps=7.5)
        assert costly["returns"] == pytest.approx(found["returns"] - 7.5e-4 * found["turnover"], abs=1e-15)
        assert found["turnover"].sum() == turns and costly["costs"].sum() == pytest.approx(7.5e-4 * turns)
        assert costly["gross"] == pytest.approx(found["returns"], abs=1e-15)
    flat = np.full(5, 20.0)
    trip = m.backtest_signals(signals=[1, 0, 0, 0, 0], close=flat, open=flat, lag=1, execution="next_open", cost_bps=10.0)
    assert np.prod(1 + trip["returns"]) == pytest.approx((1 - 0.001) ** 2, rel=1e-12)
    assert trip["turnover"].tolist() == [0, 1, 1, 0, 0]
    flip = m.backtest_signals(signals=[1, -1, -1], close=[10.0, 10.0, 10.0], open=None, lag=1, execution="close",
                              cost_bps=0.0)
    assert flip["turnover"].tolist() == [0.0, 1.0, 2.0]
    short = m.backtest_signals(signals=[-1, -1, -1], close=[10.0, 10.0, 11.0], open=[10.0, 10.0, 10.0], lag=1,
                               execution="next_open", cost_bps=0.0)
    assert short["returns"].tolist() == pytest.approx([0.0, 0.0, -0.1])
    base = dict(signals=LONG_FLAT, close=CLOSE, open=OPEN, lag=1, execution="next_open", cost_bps=0.0)
    for bad, name in ((dict(lag=0), "lag"), (dict(lag=1.0), "lag"), (dict(execution="vwap"), "execution"),
                      (dict(open=None), "open"), (dict(cost_bps=-1.0), "cost_bps"),
                      (dict(signals=[np.nan] + [1.0] * 9), "signals"), (dict(close=CLOSE[:5]), "close")):
        with pytest.raises(ValueError, match=name):
            m.backtest_signals(**(base | bad))


# ------------------------------------------------------------------------------------------------ sizing


def test_vol_target_leverage_is_the_target_over_the_trailing_window_vol_capped():
    """Each leverage is target / (stdev of the window ending at t, times sqrt(P)), the window's deviation from the
    statistics module; the cap binds where vol is low, and a still window takes the cap. At lag 0, the unlagged series
    (lane libconv1's lag: tests/test_qf_libconv1.py)."""
    r = [0.012, -0.018, 0.004, 0.022, 0.0, 0.0, 0.0, 0.0, 0.015, -0.011]
    lever = m.vol_target_leverage(returns=r, target_vol=0.15, lookback=4, periods_per_year=252, ddof=1,
                                  max_leverage=3.0, lag=0)
    assert np.isnan(lever[:3]).all()
    for t in range(3, 10):
        spread = statistics.stdev(r[t - 3:t + 1]) * math.sqrt(252)
        expected = 3.0 if spread == 0 else min(0.15 / spread, 3.0)
        assert lever[t] == pytest.approx(expected, rel=1e-12), t
    assert lever[7] == 3.0
    population = m.vol_target_leverage(returns=r, target_vol=0.15, lookback=4, periods_per_year=252, ddof=0,
                                       max_leverage=np.inf, lag=0)
    assert population[3] == pytest.approx(0.15 / (statistics.pstdev(r[0:4]) * math.sqrt(252)), rel=1e-12)
    for bad in (1, 11, 4.0):
        with pytest.raises(ValueError, match="lookback"):
            m.vol_target_leverage(returns=r, target_vol=0.15, lookback=bad, periods_per_year=252, ddof=1,
                                  max_leverage=3.0, lag=0)


def test_inverse_volatility_weights_are_the_normalised_inverses():
    assert qf.inverse_volatility_weights(vols=[0.1, 0.2, 0.4]) == pytest.approx([4 / 7, 2 / 7, 1 / 7])
    rows = qf.inverse_volatility_weights(vols=[[1.0, 1.0], [1.0, 3.0]])
    assert rows == pytest.approx(np.array([[0.5, 0.5], [0.75, 0.25]]))
    with pytest.raises(ValueError, match="vols"):
        qf.inverse_volatility_weights(vols=[0.1, 0.0])


def test_equal_risk_contribution_weights_equalise_the_contributions_and_meet_the_closed_forms():
    """Maillard, Roncalli and Teiletche (2010): the risk contributions w_i (Sigma w)_i are equal, the weights positive
    and summing to 1 (which makes the solution unique); uncorrelated assets, and two assets at any correlation, give
    the inverse-volatility weights; scaling Sigma leaves the weights unchanged."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(6, 6))
    sigma = a @ a.T / 6 + np.diag(rng.uniform(0.01, 0.2, size=6))
    w = qf.equal_risk_contribution_weights(covariance=sigma)
    contributions = w * (sigma @ w)
    assert w.sum() == pytest.approx(1.0) and np.all(w > 0)
    assert np.ptp(contributions) < 1e-11 * contributions.mean()
    assert qf.equal_risk_contribution_weights(covariance=sigma * 250) == pytest.approx(w, rel=1e-10)
    vols = np.array([0.1, 0.25, 0.4])
    assert qf.equal_risk_contribution_weights(covariance=np.diag(vols ** 2)) == pytest.approx(
        qf.inverse_volatility_weights(vols=vols), rel=1e-10)
    for rho in (-0.6, 0.0, 0.85):
        pair = np.array([[0.04, rho * 0.2 * 0.5], [rho * 0.2 * 0.5, 0.25]])
        assert qf.equal_risk_contribution_weights(covariance=pair) == pytest.approx([0.5 / 0.7, 0.2 / 0.7], rel=1e-10)
    with pytest.raises(ValueError, match="symmetric"):
        qf.equal_risk_contribution_weights(covariance=[[1.0, 0.2], [0.1, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        qf.equal_risk_contribution_weights(covariance=[[1.0, 2.0], [2.0, 1.0]])


# ------------------------------------------------------------------------------------------------ sorts and projections


def test_quantile_portfolios_written_out_on_six_assets():
    """Six assets, three groups by rank: the two lowest signals, the middle two, the top two; equal-weighted group
    returns 0, 0.01 and 0.035, the spread 0.035; value weights 1 and 3 in the top group give (0.04 + 3 * 0.03) / 4."""
    signal, returns = [0.5, -1.2, 0.3, 2.0, -0.4, 1.1], [0.02, -0.01, 0.00, 0.04, 0.01, 0.03]
    found = qf.quantile_portfolios(signal, returns, 3, "rank", "equal")
    assert list(found["groups"]) == [1, 0, 1, 2, 0, 2]
    assert found["group_returns"] == pytest.approx([0.0, 0.01, 0.035], abs=1e-15)
    assert found["spread"] == pytest.approx(0.035)
    assert list(found["weights"]) == [0.0, -0.5, 0.0, 0.5, -0.5, 0.5]
    caps = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
    valued = qf.quantile_portfolios(signal, returns, 3, "rank", caps)
    assert valued["group_returns"][2] == pytest.approx((0.04 + 3 * 0.03) / 4) and valued["weights"][5] == 0.75
    ties = qf.quantile_portfolios([1.0, 1.0, 1.0, 2.0], [0.01, 0.02, 0.03, 0.04], 2, "rank", "equal")
    assert list(ties["groups"]) == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="repeat"):
        qf.quantile_portfolios([1.0, 1.0, 1.0, 2.0], [0.01, 0.02, 0.03, 0.04], 2, "quantile", "equal")


def test_quantile_portfolios_group_as_pandas_qcut_by_period_and_leave_out_missing_assets():
    """pandas.qcut(signal, n) for 'quantile' and qcut of the ordinal ranks for 'rank', period by period; an asset
    whose signal or return is NaN is in no group that period (-1) and in no weight."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(31)
    signal, returns = rng.standard_normal((4, 21)), rng.standard_normal((4, 21)) * 0.02     # 21: edges on values
    signal[1, 5], returns[2, 7] = np.nan, np.nan
    for breakpoints in ("quantile", "rank"):
        found = qf.quantile_portfolios(signal, returns, 5, breakpoints, "equal")
        for t in range(4):
            kept = np.isfinite(signal[t]) & np.isfinite(returns[t])
            values = pd.Series(signal[t][kept])
            keys = values if breakpoints == "quantile" else values.rank(method="first")
            expected = pd.qcut(keys, 5, labels=False).to_numpy()
            assert list(found["groups"][t][kept]) == list(expected), (breakpoints, t)
            assert np.all(found["groups"][t][~kept] == -1) and np.all(found["weights"][t][~kept] == 0)
            means = pd.Series(returns[t][kept]).groupby(expected).mean().to_numpy()
            assert found["group_returns"][t] == pytest.approx(means, rel=1e-13)
            assert found["spread"][t] == pytest.approx(means[-1] - means[0], rel=1e-12)
        assert np.allclose(found["weights"].sum(axis=1), 0.0) and np.allclose(np.abs(found["weights"]).sum(axis=1), 2.0)


def test_neutralize_written_out_and_as_the_least_squares_residual():
    """Dollar neutrality subtracts the mean weight: [0.5, 0.3, -0.2] becomes [0.3, 0.1, -0.4]. With betas too the
    result has X'w = 0, is the residual of w regressed on X (numpy's lstsq), is unchanged when neutralized again,
    and a repeated column changes nothing (the pseudo-inverse)."""
    assert qf.neutralize([0.5, 0.3, -0.2], [1.0, 1.0, 1.0]) == pytest.approx([0.3, 0.1, -0.4], abs=1e-15)
    rng = np.random.default_rng(91)
    w, beta = rng.standard_normal(12), 1 + 0.3 * rng.standard_normal(12)
    x = np.column_stack([np.ones(12), beta])
    found = qf.neutralize(w, x)
    assert x.T @ found == pytest.approx([0.0, 0.0], abs=1e-13)
    assert found == pytest.approx(w - x @ np.linalg.lstsq(x, w, rcond=None)[0], abs=1e-13)
    assert qf.neutralize(found, x) == pytest.approx(found, abs=1e-13)
    assert qf.neutralize(w, np.column_stack([x, x[:, 1]])) == pytest.approx(found, abs=1e-12)
    other = found + 0.01 * (rng.standard_normal(12) @ (np.eye(12) - x @ np.linalg.pinv(x)))     # still neutral
    assert np.sum((w - found) ** 2) < np.sum((w - other) ** 2)
    with pytest.raises(ValueError, match="N x k"):
        qf.neutralize(w, np.ones((11, 2)))
