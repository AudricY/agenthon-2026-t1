"""qf.ima_capital, qf.dcc_fit, qf.block_bootstrap and qf.variance_swap_strike (lane libgaps1; docs/research/
library-syllabus-1011.md rows 46 and 53, and the holdout read's IMA and DCC gaps): the Basel internal-models capital
charge, Engle's DCC(1,1) correlations, the moving, circular and stationary bootstraps, and the variance-swap strike
by static replication. References: the Basel texts' formula and traffic-light table, closed-form identities (DCC at
a = b = 0 is the constant correlation; the likelihood forms differ by constants; a circular or stationary resample's
positions are uniform; Black-Scholes prices replicate vol^2), and simulated series' parameters recovered. Every input
is this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast


def capital(var, svar, m=3.0, ms=3.0, exceptions=None, window=60, rule="basel"):
    return qf.ima_capital(var=var, stressed_var=svar, multiplier=m, stressed_multiplier=ms, exceptions=exceptions,
                          window=window, rule=rule)


def test_the_basel_charge_is_the_larger_of_the_latest_and_the_multiplied_average_for_each():
    """Basel Committee (2009), 718(Lxxvi): c = max(VaR_{t-1}, m_c VaR_avg) + max(sVaR_{t-1}, m_s sVaR_avg), the
    averages over the last 60 days. A calm book takes the multiplied averages; a VaR that jumps on the last day
    above three times its average takes the latest; 'last' and 'average' scale one or the other."""
    var = np.r_[np.full(100, 2.0), np.full(59, 2.0), 2.6]
    svar = np.r_[np.full(59, 5.0), 5.3]
    found = capital(var, svar)
    var_avg, svar_avg = (59 * 2.0 + 2.6) / 60, (59 * 5.0 + 5.3) / 60
    assert found["var_component"] == pytest.approx(3.0 * var_avg) and found["svar_component"] == pytest.approx(
        3.0 * svar_avg)
    assert found["capital"] == pytest.approx(3.0 * var_avg + 3.0 * svar_avg)
    jump = capital(np.r_[np.full(59, 1.0), 9.0], 4.0)
    assert jump["var_component"] == 9.0 and jump["svar_component"] == 12.0
    assert capital(var, svar, rule="last")["capital"] == pytest.approx(3.0 * 2.6 + 3.0 * 5.3)
    assert capital(var, svar, m=3.5, rule="average")["capital"] == pytest.approx(3.5 * var_avg + 3.0 * svar_avg)
    assert capital(var, None)["svar_component"] == 0.0 and capital(var, None)["stressed_multiplier"] is None


def test_the_traffic_light_sets_the_multiplier_from_the_exceptions():
    """Basel Committee (1996), Supervisory framework for the use of backtesting, Table 2: green 0 to 4 exceptions
    in 250 days, plus factor 0; yellow 5 to 9, 0.40, 0.50, 0.65, 0.75, 0.85; red 10 or more, 1.00."""
    table = {0: (3.00, "green"), 4: (3.00, "green"), 5: (3.40, "yellow"), 6: (3.50, "yellow"), 7: (3.65, "yellow"),
             8: (3.75, "yellow"), 9: (3.85, "yellow"), 10: (4.00, "red"), 14: (4.00, "red")}
    for count, (m, zone) in table.items():
        found = capital(2.0, 4.0, m="traffic_light", ms="traffic_light", exceptions=count, rule="last")
        assert found["multiplier"] == pytest.approx(m) and found["stressed_multiplier"] == pytest.approx(m)
        assert found["zone"] == zone and found["capital"] == pytest.approx(6.0 * m)


def test_the_capital_charge_refuses_a_multiplier_below_three_and_a_light_without_exceptions():
    with pytest.raises(ValueError, match="at least 3"):
        capital(2.0, 4.0, m=2.5)
    with pytest.raises(ValueError, match="exceptions"):
        capital(2.0, 4.0, m="traffic_light")
    with pytest.raises(ValueError, match="rule"):
        capital(2.0, 4.0, rule="max")


# --------------------------------------------------------------------------------------------- DCC

C = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])


def dcc_series(t, a, b, seed):
    """Standardised residuals from a DCC(1,1) of unconditional correlation C (Engle 2002's recursion)."""
    rng, q, z = np.random.default_rng(seed), C.copy(), np.empty((t, 3))
    for k in range(t):
        if k:
            q = (1 - a - b) * C + a * np.outer(z[k - 1], z[k - 1]) + b * q
        scale = 1 / np.sqrt(np.diag(q))
        z[k] = np.linalg.cholesky(q * np.outer(scale, scale)) @ rng.standard_normal(3)
    return z


def test_dcc_at_zero_is_the_constant_correlation_and_the_likelihoods_differ_by_constants():
    """At a = b = 0, Q_t = Qbar: every R_t is Qbar scaled to unit diagonal (Bollerslev's CCC). The 'gaussian' form
    adds -T N ln(2 pi)/2 to 'quadratic', 'engle' adds sum z'z/2; a and b do not move with the form."""
    z = np.random.default_rng(4).standard_normal((300, 3)) @ np.linalg.cholesky(C).T
    found = {form: qf.dcc_fit(residuals=z, start=(0.02, 0.9), likelihood=form) for form in ("engle", "quadratic",
                                                                                           "gaussian")}
    assert found["engle"]["a"] == found["gaussian"]["a"] and found["engle"]["b"] == found["quadratic"]["b"]
    assert found["gaussian"]["log_likelihood"] - found["quadratic"]["log_likelihood"] == pytest.approx(
        -0.5 * 300 * 3 * math.log(2 * math.pi))
    assert found["engle"]["log_likelihood"] - found["quadratic"]["log_likelihood"] == pytest.approx(
        0.5 * float(np.sum(z * z)))
    from qf.volatility import _dcc_path
    qbar = z.T @ z / 300                     # Engle (2002): the residuals' second moments, uncentred, unscaled
    assert np.array_equal(found["engle"]["qbar"], qbar)
    corr, _, _ = _dcc_path(z, 0.0, 0.0, qbar)
    scale = 1 / np.sqrt(np.diag(qbar))
    assert np.allclose(corr, qbar * np.outer(scale, scale), atol=1e-15)


@pytest.mark.slow
def test_dcc_recovers_a_simulated_series_and_finds_no_dynamics_in_constant_correlation():
    """A DCC(1,1) series of 1,500 days with a = 0.06, b = 0.90 (this file's own) is fitted within 0.03 and 0.06; a
    constant-correlation series gives a near 0. The correlations are R_t from the information to t - 1."""
    fit = qf.dcc_fit(residuals=dcc_series(1500, 0.06, 0.90, 5), start=(0.03, 0.9), likelihood="engle")
    assert fit["a"] == pytest.approx(0.06, abs=0.03) and fit["b"] == pytest.approx(0.90, abs=0.06)
    assert fit["persistence"] < 1 and fit["converged"] and fit["correlations"].shape == (1500, 3, 3)
    flat = qf.dcc_fit(residuals=dcc_series(800, 0.0, 0.0, 6), start=(0.03, 0.9), likelihood="engle")
    assert flat["a"] < 0.02


def test_dcc_refuses_a_start_outside_the_stationary_region():
    with pytest.raises(ValueError, match="a0 \\+ b0"):
        qf.dcc_fit(residuals=np.ones((10, 2)), start=(0.5, 0.6), likelihood="engle")


# --------------------------------------------------------------------------------------------- bootstrap


def test_the_circular_and_stationary_bootstraps_draw_every_position_uniformly():
    """Politis and Romano (1992, 1994): each resampled position is uniform on the series, so a resampled mean is
    unbiased for the sample mean; the moving block bootstrap under-draws the ends. Mean block length 1/p."""
    for kind in ("circular", "stationary"):
        ix = qf.block_bootstrap(n_obs=40, kind=kind, block_length=6.0, n_draws=20_000, rng=3, statistic=None)
        share = np.bincount(ix.ravel(), minlength=40) / ix.size
        assert np.allclose(share, 1 / 40, atol=4 * math.sqrt(1 / 40 / ix.size) * 6), kind
    moving = qf.block_bootstrap(n_obs=40, kind="moving", block_length=6.0, n_draws=20_000, rng=3, statistic=None)
    share = np.bincount(moving.ravel(), minlength=40) / moving.size
    assert share[0] < 0.5 / 40 and share[20] > 1 / 40
    ix = qf.block_bootstrap(n_obs=1000, kind="stationary", block_length=8.0, n_draws=200, rng=4, statistic=None)
    steps = np.diff(ix, axis=1)
    restarts = np.mean((steps != 1) & (steps != -999))
    assert 1 / restarts == pytest.approx(8.0, rel=0.05)


def test_the_blocks_hold_consecutive_positions_and_the_seed_repeats_the_draw():
    ix = qf.block_bootstrap(n_obs=30, kind="moving", block_length=5.0, n_draws=50, rng=7, statistic=None)
    blocks = ix.reshape(50, 6, 5)
    assert np.all(np.diff(blocks, axis=2) == 1) and np.all(blocks[:, :, 0] <= 25)
    circular = qf.block_bootstrap(n_obs=30, kind="circular", block_length=5.0, n_draws=50, rng=7, statistic=None).reshape(50, 6, 5)
    assert np.all(np.diff(circular, axis=2) % 30 == 1)
    assert np.array_equal(qf.block_bootstrap(n_obs=30, kind="stationary", block_length=5.0, n_draws=5, rng=9, statistic=None),
                          qf.block_bootstrap(n_obs=30, kind="stationary", block_length=5.0, n_draws=5, rng=9, statistic=None))
    with pytest.raises(ValueError, match="block_length"):
        qf.block_bootstrap(n_obs=30, kind="moving", block_length=40.0, n_draws=5, rng=9, statistic=None)


# --------------------------------------------------------------------------------------------- variance swaps


def strip(strikes, s=100.0, r=0.03, q=0.01, t=0.5, vol=0.25):
    k = np.asarray(strikes, dtype=float)
    return k, qf.bs_price("call", s, k, t, r, q, vol), qf.bs_price("put", s, k, t, r, q, vol), s * math.exp((r - q) * t)


def test_a_flat_smile_replicates_vol_squared_in_every_form_and_the_two_published_ones_agree_as_the_grid_fines():
    """Demeterfi, Derman, Kamal and Zou (1999): under Black-Scholes the log contract, so the fair variance, is vol^2.
    A dense and wide strip gives it in each form; the exact boundary term (2/T)[ln(F/K0) - (F/K0 - 1)] and Cboe's
    -(1/T)(F/K0 - 1)^2 differ by O((F/K0 - 1)^3)/T, so the replication and the VIX form close as the grid fines."""
    k, calls, puts, forward = strip(np.arange(5.0, 400.0, 0.5))
    for form in ("replication", "vix", "forward"):
        found = qf.variance_swap_strike(strikes=k, call_prices=calls, put_prices=puts, forward=forward, rate=0.03,
                                        maturity=0.5, form=form)
        assert found["variance"] == pytest.approx(0.0625, abs=2e-5) and found["vol"] == pytest.approx(0.25, abs=5e-5)
    gaps = []
    for step in (20.0, 10.0, 1.0):                         # K0 90, 100, 101 under F = 101.005
        k, calls, puts, forward = strip(np.arange(10.0, 400.0, step))
        both = [qf.variance_swap_strike(strikes=k, call_prices=calls, put_prices=puts, forward=forward, rate=0.03,
                                        maturity=0.5, form=form)["variance"] for form in ("replication", "vix")]
        gaps.append(abs(both[0] - both[1]))
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 1e-9


def test_the_vix_form_reproduces_cboes_worked_example():
    """Cboe Volatility Index Mathematics Methodology, the worked example of 26 Sep 2022 (near term): T1 = 34,484 /
    525,600 = 0.0656088, R1 0.031664% (0.000317 in its arithmetic), F1 = 1965 + e^{R1 T1}(21.05 - 23.15) = 1962.89996,
    K0 = 1960, the price at K0 the mean (24.25 + 21.30)/2 = 22.775, the 1370 put (the strip's lowest, dK 5) contributing
    dK/K^2 e^{RT} Q = 0.0000005328, and the term (1/T1)(F1/K0 - 1)^2 = 0.00003337. The strip below is the example's
    shown rows (its elided strikes aside), so the variance is checked as the formula on them."""
    t, r = 34_484 / 525_600, 0.000317
    forward = 1965 + np.exp(r * t) * (21.05 - 23.15)
    assert t == pytest.approx(0.0656088, abs=5e-8) and forward == pytest.approx(1962.89996, abs=5e-6)
    k = np.array([1370.0, 1375.0, 1380.0, 1950.0, 1955.0, 1960.0, 1965.0, 1970.0, 2095.0, 2100.0, 2125.0])
    puts = np.array([0.2, 0.125, 0.15, 18.25, 19.75, 21.30, np.nan, np.nan, np.nan, np.nan, np.nan])
    calls = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 24.25, 21.05, 18.1, 0.2, 0.1, 0.1])
    found = qf.variance_swap_strike(strikes=k, call_prices=calls, put_prices=puts, forward=forward, rate=r,
                                    maturity=t, form="vix")
    assert found["boundary_strike"] == 1960.0 and found["otm_prices"][5] == pytest.approx(22.775)
    assert found["weights"][0] * found["otm_prices"][0] * t / 2 == pytest.approx(0.0000005328, abs=5e-11)
    term = (forward / 1960.0 - 1) ** 2 / t
    assert term == pytest.approx(0.00003337, abs=5e-9)
    assert found["variance"] == pytest.approx(float(found["weights"] @ found["otm_prices"]) - term, abs=1e-15)
    exact = qf.variance_swap_strike(strikes=k, call_prices=calls, put_prices=puts, forward=forward, rate=r,
                                    maturity=t, form="replication")["variance"]
    assert exact - found["variance"] == pytest.approx(2 / t * (np.log(forward / 1960) - forward / 1960 + 1) + term,
                                                      abs=1e-15)


def test_the_weights_are_the_trapezoids_and_k0_takes_its_term():
    """dK_i half the gap of the neighbours, one-sided at the ends (Cboe: the next-term 1325 put's dK is (1350 -
    1275)/2 = 37.5); K0 the highest strike at or below F, its price the mean of the call and the put; 'forward'
    splits the strip at F itself with no term, so on a strike at F the forms agree."""
    edges = qf.variance_swap_strike(strikes=[1275.0, 1325.0, 1350.0], call_prices=[np.nan] * 3,
                                    put_prices=[0.075, 0.15, 0.15], forward=1962.4, rate=0.0003, maturity=0.0855289,
                                    form="forward")
    assert edges["weights"][1] * 0.0855289 / 2 / np.exp(0.0003 * 0.0855289) * 1325.0 ** 2 == pytest.approx(37.5)
    k = np.array([80.0, 90.0, 95.0, 100.0, 110.0, 130.0])
    calls = np.array([22.0, 13.0, 9.0, 6.0, 2.5, 0.4])
    puts = np.array([0.6, 1.8, 3.1, 5.0, 11.0, 29.0])
    found = qf.variance_swap_strike(strikes=k, call_prices=calls, put_prices=puts, forward=103.0, rate=0.02,
                                    maturity=0.25, form="vix")
    dk = np.array([10.0, 7.5, 5.0, 7.5, 15.0, 20.0])
    q = np.array([0.6, 1.8, 3.1, 5.5, 2.5, 0.4])
    expected = 2 * math.exp(0.02 * 0.25) / 0.25 * np.sum(dk / k ** 2 * q) - (103.0 / 100.0 - 1) ** 2 / 0.25
    assert found["variance"] == pytest.approx(expected, rel=1e-14) and found["boundary_strike"] == 100.0
    plain = qf.variance_swap_strike(strikes=k, call_prices=calls, put_prices=puts, forward=103.0, rate=0.02,
                                    maturity=0.25, form="forward")
    assert np.array_equal(plain["otm_prices"], [0.6, 1.8, 3.1, 5.0, 2.5, 0.4])
    on_f = qf.variance_swap_strike(strikes=[90.0, 100.0, 110.0], call_prices=[11.0, 5.0, 1.5],
                                   put_prices=[1.2, 4.0, 10.0], forward=100.0, rate=0.0, maturity=0.5, form="forward")
    assert np.array_equal(on_f["otm_prices"], [1.2, 5.0, 1.5])          # a strike at F takes the call
    k2, calls2, puts2, _ = strip(np.arange(50.0, 200.0, 5.0), q=0.03)
    at_strike = dict(strikes=k2, call_prices=calls2, put_prices=puts2, forward=100.0, rate=0.03, maturity=0.5)
    on = qf.variance_swap_strike(**at_strike, form="forward")["variance"]
    for form in ("vix", "replication"):
        assert qf.variance_swap_strike(**at_strike, form=form)["variance"] == pytest.approx(on, abs=1e-6)
