"""qf.correlated_gbm_paths, qf.local_vol_paths and qf.barrier_mc_price (lane libgaps1; docs/research/
library-syllabus-1011.md row 68 and the holdout read's local-vol barrier): correlated paths by the Cholesky factor
(Glasserman 2.3.3), Euler paths under a local-volatility surface, and barrier options on simulated paths, monitored
at the dates or continuously by the Brownian bridge (Glasserman 6.4). References: closed forms the simulations must
meet within their standard errors (qf.bs_price, qf.barrier_price, the first-passage probability qf.hit_probability,
Broadie, Glasserman and Kou's shift), and exact identities (a constant local vol is GBM; in + out = vanilla). Every
input is this file's own; no unit's values appear (rule 8)."""
import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast

RHO = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]])


def correlated(n_paths, rng, antithetic, times=(0.5, 1.0), form="arithmetic"):
    return qf.correlated_gbm_paths(spots=[100.0, 50.0, 20.0], drifts=[0.05, 0.02, -0.01], vols=[0.2, 0.3, 0.45],
                                   correlation=RHO, times=list(times), n_paths=n_paths, rng=rng, drift_form=form,
                                   antithetic=antithetic)


def test_correlated_paths_carry_the_correlation_the_means_and_the_antithetic_mirror():
    """Glasserman (2004) 2.3.3: W = L Z with L L' = C gives corr(W_i, W_k) = C_ik; each asset's price has mean
    S e^{mu t} ('arithmetic'); an antithetic path's log increment is 2 nu dt less its pair's; a seed repeats a run."""
    paths = correlated(200_000, 1, False)
    logs = np.diff(np.log(np.concatenate([np.tile([100.0, 50.0, 20.0], (200_000, 1, 1)), paths], axis=1)), axis=1)
    for step in range(2):
        found = np.corrcoef(logs[:, step, :].T)
        assert np.allclose(found, RHO, atol=4 / np.sqrt(200_000) * 2)
    means = paths[:, -1, :].mean(axis=0)
    errors = paths[:, -1, :].std(axis=0, ddof=1) / np.sqrt(200_000)
    expected = np.array([100.0, 50.0, 20.0]) * np.exp(np.array([0.05, 0.02, -0.01]))
    assert np.all(np.abs(means - expected) < 4 * errors)
    mirrored = correlated(10, 4, True, times=(0.25,))
    nu = np.array([0.05, 0.02, -0.01]) - 0.5 * np.array([0.2, 0.3, 0.45]) ** 2
    first, second = np.log(mirrored[:5, 0] / [100.0, 50.0, 20.0]), np.log(mirrored[5:, 0] / [100.0, 50.0, 20.0])
    assert np.allclose(first + second, 2 * nu * 0.25, atol=1e-12)
    assert np.array_equal(correlated(10, 4, True, times=(0.25,)), mirrored)
    logged = correlated(100_000, 2, True, times=(1.0,), form="log")
    assert np.allclose(np.log(logged[:, -1, :] / [100.0, 50.0, 20.0]).mean(axis=0), [0.05, 0.02, -0.01], atol=1e-12)


def test_correlated_paths_refuse_a_matrix_that_is_not_a_correlation():
    for bad in ([[1.0, 1.2], [1.2, 1.0]], [[1.0, 0.5], [0.4, 1.0]], [[2.0, 0.0], [0.0, 1.0]]):
        with pytest.raises(ValueError):
            qf.correlated_gbm_paths(spots=[1.0, 1.0], drifts=[0.0, 0.0], vols=[0.1, 0.1], correlation=bad, times=[1.0],
                                    n_paths=4, rng=1, drift_form="log", antithetic=False)


def test_a_constant_local_vol_is_gbm_exactly():
    """Under 'log_euler' a constant vol steps ln S exactly: one substep and the same draws give qf.gbm_paths's
    paths (risk-neutral drift r - q)."""
    times = np.linspace(0.0, 1.0, 13)
    gbm = qf.gbm_paths(100.0, 0.03 - 0.01, 0.25, times, 1000, 5, "arithmetic", True)
    local = qf.local_vol_paths(spot=100.0, local_vol=lambda t, s: 0.25, times=times, substeps=1, rate=0.03,
                               dividend_yield=0.01, n_paths=1000, rng=5, scheme="log_euler", antithetic=True)
    assert np.allclose(local, gbm, rtol=1e-12, atol=0)


@pytest.mark.parametrize("paths, substeps", [(20_000, 50), pytest.param(100_000, 200, marks=pytest.mark.slow)])
def test_a_vol_of_time_alone_prices_europeans_at_the_root_mean_square_vol(paths, substeps):
    """With vol(t) deterministic, ln S_T is normal with variance int_0^T vol(t)^2 dt: a European is qf.bs_price at
    sqrt(that / T). 'log_euler' and 'euler' (whose E[S_T] is the forward) both meet it within their errors."""
    def vol(t, s):
        return 0.15 + 0.2 * t
    t_end = 1.0
    rms = np.sqrt(((0.15 + 0.2 * t_end) ** 3 - 0.15 ** 3) / (3 * 0.2) / t_end)
    exact = qf.bs_price("call", 100.0, 105.0, t_end, 0.04, 0.0, rms)
    for scheme in ("log_euler", "euler"):
        found = qf.local_vol_paths(spot=100.0, local_vol=vol, times=[t_end], substeps=substeps, rate=0.04,
                                   dividend_yield=0.0, n_paths=paths, rng=8, scheme=scheme, antithetic=True)
        payoff = np.exp(-0.04 * t_end) * np.maximum(found[:, -1] - 105.0, 0.0)
        pairs = 0.5 * (payoff[:paths // 2] + payoff[paths // 2:])
        assert abs(pairs.mean() - exact) < 3 * pairs.std(ddof=1) / np.sqrt(pairs.size) + 0.02, scheme


def test_local_vol_paths_read_the_surface_at_each_steps_start_and_refuse_a_negative_vol():
    seen = []
    qf.local_vol_paths(spot=100.0, local_vol=lambda t, s: seen.append(t) or 0.2, times=[0.5, 1.0], substeps=2,
                       rate=0.0, dividend_yield=0.0, n_paths=2, rng=1, scheme="euler", antithetic=False)
    assert seen == pytest.approx([0.0, 0.25, 0.5, 0.75])
    with pytest.raises(ValueError, match="negative"):
        qf.local_vol_paths(spot=100.0, local_vol=lambda t, s: -0.1, times=[1.0], substeps=1, rate=0.0,
                           dividend_yield=0.0, n_paths=2, rng=1, scheme="euler", antithetic=False)


GBM = dict(spot=100.0, rate=0.05, q=0.01, vol=0.25)


def barrier(paths, times, option, kind, strike, level, monitoring, vol=None):
    return qf.barrier_mc_price(paths=paths, times=times, option=option, barrier_type=kind, strike=strike,
                               barrier=level, rate=GBM["rate"], monitoring=monitoring, vol=vol, antithetic=True)


def test_the_bridge_meets_the_continuous_closed_form_from_quarterly_dates():
    """Between dates the bridge's crossing probability is exact for GBM, so four quarterly dates price a continuously
    monitored barrier (qf.barrier_price) within the standard error, and the hit probability is the first-passage
    probability of ln S (qf.hit_probability); the dates alone miss most crossings. Monitoring starts at the first
    column: the spot's, at time 0."""
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    paths = qf.gbm_paths(100.0, 0.04, 0.25, times, 200_000, 11, "arithmetic", True)
    for option, kind, k, h in (("call", "down-and-out", 100.0, 85.0), ("put", "up-and-out", 105.0, 120.0),
                               ("call", "down-and-in", 95.0, 90.0)):
        found = barrier(paths, times, option, kind, k, h, "bridge", 0.25)
        closed = qf.barrier_price(option, kind, 100.0, k, h, 0.0, "hit", 1.0, 0.05, 0.01, 0.25)["price"]
        assert abs(found["price"] - closed) < 3.5 * found["std_error"], (option, kind)
        hit = qf.hit_probability(np.log(h / 100.0), 1.0, 0.04 - 0.5 * 0.25 ** 2, 0.25)
        assert found["hit_probability"] == pytest.approx(hit, abs=0.004)
        dates = barrier(paths, times, option, kind, k, h, "discrete")
        assert dates["hit_probability"] < found["hit_probability"] - 0.02


def test_daily_monitoring_meets_the_shifted_closed_form_and_in_plus_out_is_the_vanilla():
    """Broadie, Glasserman and Kou (1997): a barrier monitored every dt prices as the continuous one with H moved away
    from the spot by e^{0.5826 vol sqrt(dt)}. In + out is the vanilla on every path, so their prices add up to it
    exactly; the vanilla on the paths meets qf.bs_price."""
    times = np.arange(1, 127) / 252
    paths = qf.gbm_paths(100.0, 0.04, 0.25, times, 40_000, 12, "arithmetic", True)
    shift = np.exp(0.5826 * 0.25 * np.sqrt(1 / 252))
    out_ = barrier(paths, times, "call", "down-and-out", 100.0, 90.0, "discrete")
    in_ = barrier(paths, times, "call", "down-and-in", 100.0, 90.0, "discrete")
    closed = qf.barrier_price("call", "down-and-out", 100.0, 100.0, 90.0 / shift, 0.0, "hit", 0.5, 0.05, 0.01,
                              0.25)["price"]
    assert abs(out_["price"] - closed) < 3 * out_["std_error"]
    assert out_["price"] + in_["price"] == pytest.approx(out_["vanilla"], abs=1e-12)
    assert abs(out_["vanilla"] - qf.bs_price("call", 100.0, 100.0, 0.5, 0.05, 0.01, 0.25)) < \
        3 * out_["vanilla_std_error"]
    assert out_["price"] <= out_["vanilla"] and 0.0 < out_["hit_probability"] < 1.0


@pytest.mark.parametrize("paths, substeps", [(16_000, 2), pytest.param(60_000, 4, marks=pytest.mark.slow)])
def test_a_local_vol_barrier_is_priced_on_local_vol_paths(paths, substeps):
    """The two compose: a down-and-out under a constant local vol is the GBM barrier, the bridge reading the step's
    vol; a surface at or above the flat vol everywhere (higher at low spots) knocks out more often, the same draws."""
    times = np.linspace(0.0, 1.0, 53)
    flat = qf.local_vol_paths(spot=100.0, local_vol=lambda t, s: 0.25, times=times, substeps=substeps, rate=0.05,
                              dividend_yield=0.01, n_paths=paths, rng=21, scheme="log_euler", antithetic=True)
    found = barrier(flat, times, "call", "down-and-out", 100.0, 85.0, "bridge", 0.25)
    closed = qf.barrier_price("call", "down-and-out", 100.0, 100.0, 85.0, 0.0, "hit", 1.0, 0.05, 0.01, 0.25)["price"]
    assert abs(found["price"] - closed) < 3.5 * found["std_error"]
    skewed = qf.local_vol_paths(spot=100.0, local_vol=lambda t, s: np.clip(0.25 * (100.0 / s) ** 2, 0.25, 0.6),
                                times=times, substeps=substeps, rate=0.05, dividend_yield=0.01, n_paths=paths, rng=21,
                                scheme="log_euler", antithetic=True)
    assert barrier(skewed, times, "call", "down-and-out", 100.0, 85.0, "discrete")["hit_probability"] > \
        barrier(flat, times, "call", "down-and-out", 100.0, 85.0, "discrete")["hit_probability"]


def test_antithetic_pairs_are_averaged_before_the_error_and_the_price_is_every_paths_mean():
    """Glasserman (2004) 4.2: an antithetic pair is one draw, so the error is the pair averages' standard deviation
    over the root of the pairs; the price is the mean over every path either way."""
    times = [0.0, 0.5, 1.0]
    paths = qf.gbm_paths(100.0, 0.03, 0.2, times, 1000, 2, "arithmetic", True)
    found = barrier(paths, times, "put", "up-and-out", 100.0, 125.0, "discrete")
    payoff = np.exp(-0.05) * np.maximum(100.0 - paths[:, -1], 0.0) * (paths.max(axis=1) < 125.0)
    pairs = 0.5 * (payoff[:500] + payoff[500:])
    assert found["price"] == pytest.approx(payoff.mean(), abs=1e-12)
    assert found["std_error"] == pytest.approx(pairs.std(ddof=1) / np.sqrt(500), abs=1e-12)


def test_barrier_mc_refuses_a_vol_under_discrete_and_odd_antithetic_paths():
    paths = np.full((3, 2), 100.0)
    with pytest.raises(ValueError, match="vol is None"):
        qf.barrier_mc_price(paths=paths, times=[0.5, 1.0], option="call", barrier_type="down-and-out", strike=100.0,
                            barrier=90.0, rate=0.0, monitoring="discrete", vol=0.2, antithetic=False)
    with pytest.raises(ValueError, match="even"):
        qf.barrier_mc_price(paths=paths, times=[0.5, 1.0], option="call", barrier_type="down-and-out", strike=100.0,
                            barrier=90.0, rate=0.0, monitoring="discrete", vol=None, antithetic=True)
