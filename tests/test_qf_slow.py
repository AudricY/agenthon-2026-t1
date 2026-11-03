"""agent/qf against Monte Carlo, each simulation exact for the model it checks (no time-discretisation bias): the
Asian prices on simulated monthly averages, the barrier digitals with the Brownian bridge's crossing probability,
the running extrema by the bridge's maximum given the endpoint, the OU-with-jumps moments by its exact transition,
the AR(1) inversion and the copula densities by fitting simulated samples, and Baum-Welch by recovering a simulated
chain. Seeded; the tolerances are four standard errors or the stated relative band. No unit's values or data."""
import math

import numpy as np
import pytest
from scipy.stats import kendalltau

import qf.exotics
import qf

pytestmark = pytest.mark.slow


def mean_se(sample):
    return float(np.mean(sample)), float(np.std(sample, ddof=1) / math.sqrt(len(sample)))


def test_asian_prices_against_simulated_monthly_averages():
    """Geometric exact within 4 SE of the simulated geometric payoff; Levy and both Curran forms within 1% of the
    simulated arithmetic price, Curran's root at or below it (a lower bound) and above the geometric."""
    rng = np.random.default_rng(20261008)
    s, t, r, q, v, n, paths = 100.0, 1.0, 0.03, 0.01, 0.3, 12, 400_000
    times = np.arange(1, n + 1) * t / n
    steps = rng.standard_normal((paths, n)) * v * math.sqrt(t / n) + (r - q - v * v / 2) * t / n
    prices = s * np.exp(np.cumsum(steps, axis=1))
    arith, geo = prices.mean(axis=1), np.exp(np.log(prices).mean(axis=1))
    for k in (85.0, 100.0, 115.0):
        mc_geo = mean_se(math.exp(-r * t) * np.maximum(geo - k, 0.0))
        mc_arith = mean_se(math.exp(-r * t) * np.maximum(arith - k, 0.0))
        found = {m: qf.asian_price(m, "call", s, k, times, t, r, q, v) for m in qf.exotics.METHODS}
        assert abs(found["geometric"] - mc_geo[0]) < 4 * mc_geo[1], k
        for method in ("levy", "curran_taylor", "curran_root"):
            assert found[method] == pytest.approx(mc_arith[0], rel=0.01, abs=4 * mc_arith[1]), (method, k)
        assert found["curran_root"] < mc_arith[0] + 4 * mc_arith[1] and found["geometric"] < found["curran_root"]


def test_barrier_digitals_against_the_brownian_bridges_crossing_probability():
    """Each step's chance of crossing the barrier between its two simulated ends is exp(-2 (x_i - h)(x_{i+1} - h) /
    (vol^2 dt)) for the log path (Glasserman, Monte Carlo Methods in Financial Engineering, 6.4), so the survival
    weight is exact for continuous monitoring at any step count."""
    rng = np.random.default_rng(8)
    s, t, r, q, v, steps, paths = 100.0, 0.5, 0.04, 0.02, 0.35, 25, 200_000
    dt = t / steps
    for payoff in ("cash", "asset"):
        drift = r - q + (v * v / 2 if payoff == "asset" else -v * v / 2)    # the share measure for an asset payoff
        log_path = np.concatenate([np.zeros((paths, 1)), np.cumsum(drift * dt + v * math.sqrt(dt)
                                                                   * rng.standard_normal((paths, steps)), axis=1)], 1)
        scale = s * math.exp(-q * t) if payoff == "asset" else math.exp(-r * t)
        for barrier_type, option, h, k in (("down-and-out", "call", 85.0, 95.0), ("down-and-out", "put", 85.0, 105.0),
                                           ("up-and-out", "put", 118.0, 104.0), ("up-and-out", "call", 118.0, 98.0)):
            gap = log_path - math.log(h / s)
            cross = np.exp(-2 * gap[:, :-1] * gap[:, 1:] / (v * v * dt))
            survive = np.prod(np.where(gap[:, :-1] * gap[:, 1:] > 0, 1 - cross, 0.0), axis=1)
            final = s * np.exp(log_path[:, -1])
            paid = (final > k) if option == "call" else (final < k)
            mc = mean_se(scale * paid * survive)
            found = qf.barrier_digital_price(payoff, option, barrier_type, s, k, h, t, r, q, v)
            assert abs(found - mc[0]) < 4 * mc[1], (payoff, barrier_type, option)


def test_running_extrema_against_the_bridges_maximum_given_the_endpoint():
    """Given X_T, the maximum of the bridge is (X_T + sqrt(X_T^2 - 2 vol^2 T ln U))/2 (Glasserman 6.4; Shreve II,
    3.7), the minimum (X_T - sqrt(...))/2, drift or not: an exact draw of (X_T, M_T) and (X_T, m_T)."""
    rng = np.random.default_rng(5)
    t, d, v, n = 1.3, -0.04, 0.27, 1_000_000
    x = d * t + v * math.sqrt(t) * rng.standard_normal(n)
    spread = np.sqrt(x * x - 2 * v * v * t * np.log(rng.random(n)))
    top, bottom = (x + spread) / 2, (x - spread) / 2
    for level in (0.1, 0.35):
        mc = mean_se(top >= level)
        assert abs(qf.hit_probability(level, t, d, v) - mc[0]) < 4 * mc[1]
    for level in (-0.1, -0.4):
        mc = mean_se(bottom <= level)
        assert abs(qf.hit_probability(level, t, d, v) - mc[0]) < 4 * mc[1]
    for xx, m in ((0.0, 0.2), (-0.15, 0.05)):
        mc = mean_se((x <= xx) & (top <= m))
        assert abs(qf.terminal_and_max_cdf(xx, m, t, d, v) - mc[0]) < 4 * mc[1]


def test_ou_jump_moments_against_the_exact_transition():
    """X_h = x0 e^{-kh} + theta (1 - e^{-kh}) + sigma sqrt((1 - e^{-2kh})/(2k)) Z + sum_jumps J e^{-k(h - s)}, the
    jump count Poisson(lambda h) and its times uniform on [0, h]: an exact draw (Cont and Tankov 2004, 15.1)."""
    rng = np.random.default_rng(13)
    x0, kappa, theta, sigma, lam, mu_j, s_j, n = 0.9, 1.4, 0.2, 0.5, 9.0, 0.06, 0.1, 400_000
    for h in (0.2, 1.0, 3.0):
        counts = rng.poisson(lam * h, n)
        owner = np.repeat(np.arange(n), counts)
        decayed = rng.normal(mu_j, s_j, owner.size) * np.exp(-kappa * h * rng.random(owner.size))
        jumps = np.bincount(owner, weights=decayed, minlength=n)
        x = (x0 * math.exp(-kappa * h) + theta * (1 - math.exp(-kappa * h))
             + sigma * math.sqrt((1 - math.exp(-2 * kappa * h)) / (2 * kappa)) * rng.standard_normal(n) + jumps)
        mean, var = qf.ou_jump_moments(x0, kappa, theta, sigma, lam, mu_j, s_j, h)
        assert abs(x.mean() - mean) < 4 * x.std() / math.sqrt(n)
        assert x.var() == pytest.approx(var, rel=0.01)


def test_ou_from_ar1_recovers_a_simulated_paths_parameters():
    """An exactly simulated OU path regressed by OLS on its lag, inverted by ou_from_ar1."""
    rng = np.random.default_rng(21)
    kappa, theta, sigma, dt, n = 3.0, 0.5, 0.4, 1 / 252, 200_000
    b = math.exp(-kappa * dt)
    shocks = sigma * math.sqrt((1 - b * b) / (2 * kappa)) * rng.standard_normal(n)
    x = np.empty(n + 1)
    x[0] = theta
    for i in range(n):
        x[i + 1] = theta * (1 - b) + b * x[i] + shocks[i]
    slope, intercept = np.polyfit(x[:-1], x[1:], 1)
    resid = x[1:] - (intercept + slope * x[:-1])
    found = qf.ou_from_ar1(intercept, slope, resid.var(ddof=2), dt)
    assert found[0] == pytest.approx(kappa, rel=0.15) and found[1] == pytest.approx(theta, abs=0.03)
    assert found[2] == pytest.approx(sigma, rel=0.01)


def test_the_copula_densities_fit_their_own_samples():
    """Simulated pairs (the Gaussian and t copulas through their normal mixture; Clayton by conditional inversion,
    Nelsen 4.3; Gumbel by Marshall and Olkin's positive-stable frailty): the sample's Kendall tau inverted lands on
    the true parameter, and the log-likelihood peaks there on a grid."""
    rng = np.random.default_rng(34)
    n = 6000
    from scipy.special import ndtr, stdtr
    z = rng.standard_normal((n, 2))
    rho = 0.55
    x = z[:, 0], rho * z[:, 0] + math.sqrt(1 - rho * rho) * z[:, 1]
    w = np.sqrt(4.0 / rng.chisquare(4.0, n))
    samples = {"gaussian": (ndtr(x[0]), ndtr(x[1])), "t": (stdtr(4.0, x[0] * w), stdtr(4.0, x[1] * w))}
    theta = 2.0
    u, e = rng.random(n), rng.random(n)
    samples["clayton"] = (u, ((e ** (-theta / (1 + theta)) - 1) * u ** -theta + 1) ** (-1 / theta))
    alpha = 1 / 1.6           # Gumbel 1.6: a positive alpha-stable frailty (Chambers, Mallows and Stuck)
    a, b = rng.uniform(0, math.pi, n), rng.exponential(1.0, n)
    frailty = (np.sin(alpha * a) / np.sin(a) ** (1 / alpha)) * (np.sin((1 - alpha) * a) / b) ** ((1 - alpha) / alpha)
    samples["gumbel"] = tuple(np.exp(-(rng.exponential(1.0, n) / frailty) ** alpha) for _ in range(2))
    truth = {"gaussian": rho, "t": rho, "clayton": theta, "gumbel": 1.6}
    for family, (uu, vv) in samples.items():
        tau = kendalltau(uu, vv).statistic
        assert qf.copula_param_from_tau(family, tau) == pytest.approx(truth[family], rel=0.06), family
    grid = {"gaussian": (qf.gaussian_copula_logpdf, np.linspace(0.4, 0.7, 31)),
            "clayton": (qf.clayton_copula_logpdf, np.linspace(1.5, 2.5, 41)),
            "gumbel": (qf.gumbel_copula_logpdf, np.linspace(1.3, 1.9, 31))}
    for family, (logpdf, params) in grid.items():
        uu, vv = samples[family]
        best = params[np.argmax([np.sum(logpdf(uu, vv, p)) for p in params])]
        assert best == pytest.approx(truth[family], rel=0.06), family
    uu, vv = samples["t"]
    nus = np.arange(2.0, 15.0, 0.5)
    best = nus[np.argmax([np.sum(qf.t_copula_logpdf(uu, vv, rho, nu)) for nu in nus])]
    assert 3.0 <= best <= 6.0
    assert np.sum(qf.t_copula_logpdf(uu, vv, rho, 4.0)) > np.sum(qf.gaussian_copula_logpdf(uu, vv, rho))


def test_baum_welch_recovers_a_simulated_two_state_chain():
    """Two regimes with known means, deviations and transitions: Baum-Welch from rough starting values recovers them
    and labels most days by their posterior."""
    rng = np.random.default_rng(55)
    a = np.array([[0.97, 0.03], [0.05, 0.95]])
    n, states = 5000, [0]
    for _ in range(n - 1):
        states.append(int(rng.random() < a[states[-1], 1]))
    states = np.array(states)
    x = np.where(states == 0, rng.normal(0.05, 0.6, n), rng.normal(-0.2, 1.8, n))
    fit = qf.hmm_baum_welch(x, [0.5, -0.5], [1.0, 1.5], [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5], max_iter=500,
                            tol=1e-8, stop="loglik", update_initial=True)
    assert fit["converged"]
    assert fit["means"] == pytest.approx([0.05, -0.2], abs=0.08)
    assert fit["stds"] == pytest.approx([0.6, 1.8], rel=0.06)
    assert fit["transition"] == pytest.approx(a, abs=0.02)
    assert np.mean((fit["posteriors"][:, 1] > 0.5) == (states == 1)) > 0.9
