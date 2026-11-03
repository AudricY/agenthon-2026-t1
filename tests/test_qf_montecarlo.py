"""agent/qf/montecarlo.py (runs/lanes/lib5-prompt.md): each function against the closed form its estimate converges
to, at a tolerance set by its own standard error, or an identity: bump-and-revalue Greeks against Black-Scholes's
(Hull, "The Greek letters"), exactly on bs_price and within the error of the bumped payoffs on a Monte Carlo pricer
with common random numbers (Glasserman 2004, 7.1). Seeds are fixed; every parameter is chosen freely, no unit's."""
import math

import numpy as np
import pytest

import qf
import qf.montecarlo as MC
from test_qf_bs import check_cards

pytestmark = pytest.mark.fast

POINT = dict(spot=100.0, strike=105.0, maturity=0.75, rate=0.03, dividend_yield=0.01, vol=0.25)


def test_the_montecarlo_cards_keep_the_format_and_their_examples_run():
    check_cards(MC)


def bs(option="call"):
    return lambda spot, strike, maturity, rate, dividend_yield, vol: qf.bs_price(option, spot, strike, maturity, rate,
                                                                                dividend_yield, vol)


def test_bumping_bs_price_gives_its_closed_form_greeks_by_every_scheme():
    """Central differences agree with bs_greeks to O(h^2), one-sided ones to O(h); theta is minus the maturity's."""
    exact = qf.bs.bs_greeks(option="call", **POINT, vega_per=1.0, theta_per="year", rho_per=1.0)
    bumps = {"spot": (1e-4, "relative"), "vol": (1e-4, "absolute"), "maturity": (1e-5, "absolute"),
             "rate": (1e-5, "absolute")}
    central = MC.bump_greeks(bs(), POINT, bumps, "central")
    assert central["price"] == pytest.approx(exact["price"], rel=1e-14)
    assert central["first"]["spot"] == pytest.approx(exact["delta"], abs=1e-8)
    assert central["second"]["spot"] == pytest.approx(exact["gamma"], abs=1e-6)
    assert central["first"]["vol"] == pytest.approx(exact["vega"], abs=1e-6)
    assert -central["first"]["maturity"] == pytest.approx(exact["theta"], abs=1e-6)
    assert central["first"]["rate"] == pytest.approx(exact["rho"], abs=1e-6)
    for scheme in ("forward", "backward"):
        one_sided = MC.bump_greeks(bs(), POINT, {"spot": (1e-4, "relative")}, scheme)
        sign = 1.0 if scheme == "forward" else -1.0
        h = 1e-4 * POINT["spot"]
        # the leading error is +- (h/2) V'' for the first derivative and +- h V''' for the second
        assert one_sided["first"]["spot"] - exact["delta"] == pytest.approx(sign * h / 2 * exact["gamma"], rel=1e-2)
        assert one_sided["second"]["spot"] == pytest.approx(exact["gamma"], abs=2 * h * 1e-3)


def monte_carlo(seed, n=40_000):
    """A European call by Monte Carlo on terminal GBM draws, its generator reseeded on every call (common random
    numbers), and its discounted payoffs for the per-path differences."""
    def payoffs(spot, strike, maturity, rate, dividend_yield, vol):
        z = np.random.default_rng(seed).standard_normal(n)
        terminal = spot * np.exp((rate - dividend_yield - vol * vol / 2) * maturity + vol * math.sqrt(maturity) * z)
        return math.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    return payoffs, lambda **point: float(np.mean(payoffs(**point)))


def test_a_monte_carlo_pricer_with_common_random_numbers_bumps_to_the_closed_form_within_its_error():
    """Delta, gamma and vega of a seeded Monte Carlo call against Black-Scholes's, each within four standard errors
    of the bumped estimator: the mean of the per-path differences and its sample error."""
    exact = qf.bs.bs_greeks(option="call", **POINT, vega_per=1.0, theta_per="year", rho_per=1.0)
    payoffs, price = monte_carlo(7)
    h_spot, h_vol = 0.01 * POINT["spot"], 0.002
    found = MC.bump_greeks(price, POINT, {"spot": (0.01, "relative"), "vol": (h_vol, "absolute")}, "central")
    def moved(**shift):
        return payoffs(**{**POINT, **{key: POINT[key] + value for key, value in shift.items()}})
    paths = {"delta": (moved(spot=h_spot) - moved(spot=-h_spot)) / (2 * h_spot),
             "gamma": (moved(spot=h_spot) - 2 * payoffs(**POINT) + moved(spot=-h_spot)) / h_spot ** 2,
             "vega": (moved(vol=h_vol) - moved(vol=-h_vol)) / (2 * h_vol)}
    said = {"delta": found["first"]["spot"], "gamma": found["second"]["spot"], "vega": found["first"]["vol"]}
    for greek, per_path in paths.items():
        assert said[greek] == pytest.approx(per_path.mean(), rel=1e-12), greek     # the same draws at every bump
        error = per_path.std(ddof=1) / math.sqrt(per_path.size)
        assert abs(said[greek] - exact[greek]) < 4 * error, (greek, said[greek], exact[greek], error)


def test_fresh_draws_at_every_bump_make_the_gamma_noise():
    """The card's warning: a pricer drawing anew on each call differences away the price's error, not the noise;
    its gamma here misses by more than ten of the common-random-number estimator's errors."""
    exact = qf.bs.bs_greeks(option="call", **POINT, vega_per=1.0, theta_per="year", rho_per=1.0)["gamma"]
    calls = iter(range(100, 200))
    def fresh(**point):
        return monte_carlo(next(calls))[1](**point)
    found = MC.bump_greeks(fresh, POINT, {"spot": (0.01, "relative")}, "central")["second"]["spot"]
    payoffs, _ = monte_carlo(7)
    h = 0.01 * POINT["spot"]
    per_path = (payoffs(**{**POINT, "spot": 100.0 + h}) - 2 * payoffs(**POINT)
                + payoffs(**{**POINT, "spot": 100.0 - h})) / h ** 2
    assert abs(found - exact) > 10 * per_path.std(ddof=1) / math.sqrt(per_path.size)


def test_bump_greeks_refuses_what_it_cannot_bump():
    with pytest.raises(ValueError, match="scheme"):
        MC.bump_greeks(bs(), POINT, {"spot": (0.01, "relative")}, "complex_step")
    with pytest.raises(ValueError, match="params does not hold"):
        MC.bump_greeks(bs(), POINT, {"sigma": (0.01, "absolute")}, "central")
    with pytest.raises(ValueError, match="form"):
        MC.bump_greeks(bs(), POINT, {"spot": (0.01, "percent")}, "central")
    with pytest.raises(ValueError, match="zero"):
        MC.bump_greeks(bs(), {**POINT, "dividend_yield": 0.0}, {"dividend_yield": (0.01, "relative")}, "central")
    with pytest.raises(ValueError, match="pair"):
        MC.bump_greeks(bs(), POINT, {"spot": 0.01}, "central")


# ------------------------------------------------------------------------------------------------ OU paths

OU = dict(x0=0.8, kappa=1.5, theta=0.2, sigma=0.4, dt=0.02, n_steps=40)


def within(sample, mean, variance, k=4.0):
    """The sample's mean and variance each within k standard errors of the closed form's: the mean's error
    sqrt(var / n), the variance's sqrt((m4 - var^2) / n) from the sample's own fourth central moment."""
    n = sample.size
    centred = sample - sample.mean()
    m4 = np.mean(centred ** 4)
    return (abs(sample.mean() - mean) < k * math.sqrt(variance / n)
            and abs(sample.var(ddof=1) - variance) < k * math.sqrt((m4 - variance ** 2) / n))


def test_exact_ou_paths_have_the_transition_laws_moments_at_every_horizon():
    """Glasserman (2004), 3.3: the exact OU step; mean theta + (x0 - theta) e^{-kappa t}, variance sigma^2 (1 -
    e^{-2 kappa t}) / (2 kappa), checked at three horizons (qf.ou_jump_moments with no jumps)."""
    x = MC.ou_paths(**OU, n_paths=20_000, rng=np.random.default_rng(3), scheme="exact", jump_rate=0.0,
                    jump_mean=0.0, jump_std=0.0, jump_count="poisson")
    assert x.shape == (20_000, 41) and np.all(x[:, 0] == 0.8)
    for k in (1, 10, 40):
        mean, variance = qf.ou_jump_moments(0.8, 1.5, 0.2, 0.4, 0.0, 0.0, 0.0, k * 0.02)
        assert within(x[:, k], mean, variance), k


def test_exact_ou_paths_with_poisson_jumps_have_the_jump_diffusions_exact_moments():
    """Cont and Tankov (2004), 6.3, with each jump decayed from its time in the step: the moments are
    qf.ou_jump_moments' at every horizon, the jumps' (lambda (mu^2 + s^2)) in the variance."""
    jumps = dict(jump_rate=6.0, jump_mean=-0.15, jump_std=0.2)
    x = MC.ou_paths(**OU, n_paths=20_000, rng=np.random.default_rng(4), scheme="exact", **jumps, jump_count="poisson")
    for k in (5, 40):
        mean, variance = qf.ou_jump_moments(0.8, 1.5, 0.2, 0.4, 6.0, -0.15, 0.2, k * 0.02)
        assert within(x[:, k], mean, variance), k
    coarse = MC.ou_paths(**dict(OU, dt=0.5, n_steps=2), n_paths=20_000, rng=np.random.default_rng(8), scheme="exact",
                         **jumps, jump_count="poisson")       # kappa dt 0.75: a jump undecayed in its step would show
    for k in (1, 2):
        assert within(coarse[:, k], *qf.ou_jump_moments(0.8, 1.5, 0.2, 0.4, 6.0, -0.15, 0.2, k * 0.5)), k
    pure = MC.ou_paths(**OU, n_paths=20_000, rng=np.random.default_rng(4), scheme="exact", jump_rate=0.0,
                       jump_mean=-0.15, jump_std=0.2, jump_count="poisson")
    assert not within(pure[:, 40], *qf.ou_jump_moments(0.8, 1.5, 0.2, 0.4, 6.0, -0.15, 0.2, 0.8))


def recursion(c, drift, noise_var, jump_mean, jump_var, steps, x0):
    """The moments of X_{k+1} = c X_k + drift + noise + jump by the recursion, step by step."""
    mean, variance, found = x0, 0.0, [(x0, 0.0)]
    for _ in range(steps):
        mean, variance = c * mean + drift + jump_mean, c * c * variance + noise_var + jump_var
        found.append((mean, variance))
    return found


def test_euler_ou_paths_have_their_discrete_recursions_moments_with_and_without_bernoulli_jumps():
    """The Euler step X + kappa (theta - X) dt + sigma sqrt(dt) Z, and at most one N(mu, s^2) jump a step with
    probability p = lambda dt: mean p mu, variance p (mu^2 + s^2) - p^2 mu^2 a step."""
    c, drift, noise = 1 - 1.5 * 0.02, 1.5 * 0.2 * 0.02, 0.4 ** 2 * 0.02
    x = MC.ou_paths(**OU, n_paths=20_000, rng=np.random.default_rng(5), scheme="euler", jump_rate=0.0,
                    jump_mean=0.0, jump_std=0.0, jump_count="bernoulli")
    moments = recursion(c, drift, noise, 0.0, 0.0, 40, 0.8)
    assert all(within(x[:, k], *moments[k]) for k in (1, 20, 40))
    p, mu, s = 6.0 * 0.02, -0.15, 0.2
    x = MC.ou_paths(**OU, n_paths=20_000, rng=np.random.default_rng(6), scheme="euler", jump_rate=6.0,
                    jump_mean=mu, jump_std=s, jump_count="bernoulli")
    moments = recursion(c, drift, noise, p * mu, p * (mu * mu + s * s) - (p * mu) ** 2, 40, 0.8)
    assert all(within(x[:, k], *moments[k]) for k in (1, 20, 40))
    exact = qf.ou_jump_moments(0.8, 1.5, 0.2, 0.4, 6.0, mu, s, 0.8)
    assert moments[40] != pytest.approx(exact, rel=1e-3)        # Euler's moments are its own, near the SDE's


def test_ou_paths_draw_the_normals_first_reproduce_from_a_seed_and_refuse_bad_arguments():
    common = dict(OU, n_paths=50, scheme="exact", jump_mean=0.0, jump_std=0.1, jump_count="poisson")
    first = MC.ou_paths(**common, rng=9, jump_rate=0.0)
    assert np.array_equal(first, MC.ou_paths(**common, rng=np.random.default_rng(9), jump_rate=0.0))
    z = np.random.default_rng(9).standard_normal((50, 40))
    b = math.exp(-1.5 * 0.02)
    by_hand = 0.2 + (0.8 - 0.2) * b + 0.4 * math.sqrt((1 - b * b) / 3.0) * z[:, 0]
    assert first[:, 1] == pytest.approx(by_hand, abs=1e-15)
    for bad, match in ((dict(scheme="milstein"), "scheme"), (dict(jump_count="hawkes"), "jump_count"),
                       (dict(rng="seed"), "rng"), (dict(n_steps=0), "n_steps"), (dict(jump_rate=-1.0), "negative")):
        with pytest.raises(ValueError, match=match):
            MC.ou_paths(**{**common, "rng": 1, "jump_rate": 1.0, **bad})
    with pytest.raises(ValueError, match="not exceed 1"):
        MC.ou_paths(**{**common, "rng": 1, "jump_rate": 60.0, "jump_count": "bernoulli"})


# ------------------------------------------------------------------------------------------------ GBM paths


def test_gbm_paths_have_the_lognormal_laws_moments_with_the_ito_term_taken_once():
    """Glasserman (2004), 3.2.1: under 'arithmetic' E[S_t] = S e^{mu t} and Var = S^2 e^{2 mu t} (e^{vol^2 t} - 1);
    under 'log' E[ln S_t] = ln S + nu t, Var = vol^2 t; each within four standard errors, and the two forms the same
    paths where nu = mu - vol^2/2 (the Ito term applied once, never twice)."""
    times = [0.1, 0.5, 1.0, 2.0]
    s, mu, vol = 50.0, 0.08, 0.3
    paths = MC.gbm_paths(s, mu, vol, times, 40_000, np.random.default_rng(41), "arithmetic", False)
    assert paths.shape == (40_000, 4)
    for j, t in enumerate(times):
        assert within(paths[:, j], s * math.exp(mu * t), s * s * math.exp(2 * mu * t) * math.expm1(vol * vol * t)), t
    logged = np.log(MC.gbm_paths(s, mu - vol * vol / 2, vol, times, 40_000, np.random.default_rng(41), "log", False))
    assert np.allclose(np.log(paths), logged, rtol=0, atol=1e-12)
    for j, t in enumerate(times):
        assert within(logged[:, j], math.log(s) + (mu - vol * vol / 2) * t, vol * vol * t), t
    assert np.cov(logged[:, 1], logged[:, 3])[0, 1] == pytest.approx(vol * vol * 0.5, rel=0.05)    # vol^2 min(s, t)
    twice = MC.gbm_paths(s, mu - vol * vol / 2, vol, times, 40_000, np.random.default_rng(41), "arithmetic", False)
    assert not within(twice[:, 3], s * math.exp(mu * 2.0), s * s * math.exp(4 * mu) * math.expm1(vol * vol * 2.0))


def test_gbm_paths_mirror_antithetic_pairs_and_start_at_the_spot_at_time_zero():
    paths = MC.gbm_paths(100.0, 0.05, 0.2, [0.0, 0.5, 1.0], 6, 3, "log", True)
    assert np.all(paths[:, 0] == 100.0)
    logs = np.log(paths / 100.0)
    assert logs[:3] + logs[3:] == pytest.approx(np.tile(2 * 0.05 * np.array([0.0, 0.5, 1.0]), (3, 1)), abs=1e-14)
    z = np.random.default_rng(3).standard_normal((3, 3))
    assert logs[:3, 2] == pytest.approx(0.05 + 0.2 * (math.sqrt(0.5) * z[:, 1] + math.sqrt(0.5) * z[:, 2]), abs=1e-14)
    for bad, match in ((dict(n_paths=5), "even"), (dict(times=[0.5, 0.25]), "increasing"),
                       (dict(drift_form="ito"), "drift_form"), (dict(antithetic="yes"), "antithetic")):
        args = dict(spot=100.0, drift=0.05, vol=0.2, times=[0.5, 1.0], n_paths=4, rng=1, drift_form="log",
                    antithetic=True) | bad
        with pytest.raises(ValueError, match=match):
            MC.gbm_paths(**args)


# ------------------------------------------------------------------------------------------------ the estimate


def test_mc_price_is_the_discounted_mean_its_error_the_samples_and_its_interval_covers_at_its_level():
    """Glasserman (2004), 1.1: the error is D s / sqrt(n) with s numpy's std at the ddof; over 1,000 independent
    batches the batch prices spread as the error says (their standard deviation within 10% of the mean error), and
    the 90% intervals cover the true Black-Scholes price on 90% of them (within three binomial errors)."""
    payoffs = np.array([0.0, 3.2, 0.0, 7.5, 1.1, 0.0, 4.4, 2.0])
    found = MC.mc_price(payoffs, 0.97, 1, 0.95)
    assert found["price"] == pytest.approx(0.97 * payoffs.mean(), rel=1e-15) and found["n"] == 8
    assert found["std_error"] == pytest.approx(0.97 * payoffs.std(ddof=1) / math.sqrt(8), rel=1e-15)
    assert found["high"] - found["price"] == pytest.approx(1.959963984540054 * found["std_error"], rel=1e-14)
    assert MC.mc_price(payoffs, 0.97, 0, 0.95)["std_error"] == pytest.approx(0.97 * payoffs.std() / math.sqrt(8))
    exact = qf.bs_price("call", 100.0, 105.0, 0.75, 0.03, 0.01, 0.25)
    z = np.random.default_rng(51).standard_normal((2_000, 1_000))
    terminal = 100.0 * np.exp((0.03 - 0.01 - 0.25 ** 2 / 2) * 0.75 + 0.25 * math.sqrt(0.75) * z)
    batches = MC.mc_price(np.maximum(terminal - 105.0, 0.0), math.exp(-0.03 * 0.75), 1, 0.90)
    assert batches["price"].shape == (1_000,)
    assert np.std(batches["price"], ddof=1) == pytest.approx(np.mean(batches["std_error"]), rel=0.1)
    covered = np.mean((batches["low"] <= exact) & (exact <= batches["high"]))
    assert abs(covered - 0.90) < 3 * math.sqrt(0.9 * 0.1 / 1_000)
    for bad, match in ((dict(ddof=2), "ddof"), (dict(confidence=95), "confidence"), (dict(payoffs=[1.0]), "payoffs"),
                       (dict(discount_factor=0.0), "discount_factor")):
        with pytest.raises(ValueError, match=match):
            MC.mc_price(**dict(payoffs=payoffs, discount_factor=0.97, ddof=1, confidence=0.95) | bad)


# ------------------------------------------------------------------------------------------------ first hits


def test_first_hit_times_written_out():
    paths = [[100.0, 104.0, 111.0, 108.0], [100.0, 97.0, 99.0, 101.0], [112.0, 90.0, 95.0, 96.0]]
    times = [0.0, 0.25, 0.5, 0.75]
    assert list(MC.first_hit_times(paths, times, 110.0, "up")) == [0.5, math.inf, 0.0]
    assert list(MC.first_hit_times(paths, times, 97.0, "down")) == [math.inf, 0.25, 0.25]
    assert list(MC.first_hit_times(paths, times, [104.0, 101.0, 200.0], "up")) == [0.25, 0.75, math.inf]
    with pytest.raises(ValueError, match="direction"):
        MC.first_hit_times(paths, times, 110.0, "across")
    with pytest.raises(ValueError, match="n_times"):
        MC.first_hit_times(paths, times[:3], 110.0, "up")


def test_first_hit_times_of_brownian_paths_follow_the_reflection_principle_with_the_discrete_correction():
    """X_t = drift t + vol W_t on a grid of dt: the share hit by t is qf.hit_probability's for the level moved out by
    0.5826 vol sqrt(dt) (Broadie, Glasserman and Kou 1997), within four binomial errors at three horizons, and below
    the continuous probability at the level itself; down-crossings mirror up-crossings of -X."""
    n, steps, horizon, drift, vol, level = 20_000, 200, 1.0, 0.1, 0.3, 0.25
    dt = horizon / steps
    rng = np.random.default_rng(81)
    x = np.cumsum(drift * dt + vol * math.sqrt(dt) * rng.standard_normal((n, steps)), axis=1)
    x = np.hstack([np.zeros((n, 1)), x])
    times = np.arange(steps + 1) * dt
    hits = MC.first_hit_times(x, times, level, "up")
    shifted = level + 0.5826 * vol * math.sqrt(dt)
    for t in (0.25, 0.5, 1.0):
        share, p = np.mean(hits <= t + 1e-12), qf.hit_probability(shifted, t, drift, vol)
        assert abs(share - p) < 4 * math.sqrt(p * (1 - p) / n), t
    assert np.mean(hits <= 1.0) < qf.hit_probability(level, 1.0, drift, vol)
    mirrored = MC.first_hit_times(-x, times, -level, "down")
    assert np.array_equal(mirrored, hits)
