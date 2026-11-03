"""qf.montecarlo's estimators (runs/lanes/lib-montecarlo1-prompt.md; docs/research/library-syllabus-1011.md ranks 32,
69, 70, 82, 108, 113 and the owner's in633): the payoff pricer against qf.mc_price and Black-Scholes; Kemna and
Vorst's control variate on a monthly arithmetic Asian against QuantLib 1.43's MCDiscreteArithmeticAPEngine with its
geometric control (6.154108 +- 0.000343); importance sampling of a deep out-of-the-money digital (Glasserman 4.6);
stratified and Latin hypercube normals against their identities (Glasserman 4.3, 4.4); Sobol points against
scipy.stats.qmc.Sobol and Black-Scholes; the quantile, ratio and batch-means errors against exact normal samples
(Glasserman 1.1, 9.1); the random tree and the dual bounds bracketing a Bermudan put on a fine lattice built here
(Broadie and Glasserman 1997; Andersen and Broadie 2004; Glasserman 8.3 to 8.7). Seeds fixed; no unit's values."""
import math

import numpy as np
import pytest
from scipy import stats
from scipy.stats import qmc

import qf

pytestmark = pytest.mark.fast

CALL = dict(spot=100.0, strike=100.0, maturity=1.0, rate=0.03, dividend_yield=0.0, vol=0.2)


def bs_call():
    return qf.bs_price("call", **CALL)


def terminal(n, seed, antithetic):
    return qf.gbm_paths(100.0, 0.03, 0.2, [1.0], n, seed, "arithmetic", antithetic)


def call_payoff(s):
    return np.maximum(s[:, -1] - 100.0, 0.0)


def test_the_payoff_pricer_is_mc_price_on_the_same_payoffs():
    paths = terminal(4_000, 1, False)
    found = qf.mc_payoff_price(paths, call_payoff, None, None, False, None, None, math.exp(-0.03), 1, 0.95)
    plain = qf.mc_price(call_payoff(paths), math.exp(-0.03), 1, 0.95)
    assert all(found[key] == pytest.approx(plain[key], rel=1e-14) for key in ("price", "std_error", "low", "high"))
    assert found["n"] == 4_000 and math.isnan(found["beta"])
    pairs = terminal(4_000, 1, True)
    paired = qf.mc_payoff_price(pairs, call_payoff, None, None, True, None, None, math.exp(-0.03), 0, 0.9)
    averaged = 0.5 * (call_payoff(pairs)[:2_000] + call_payoff(pairs)[2_000:])
    assert paired["std_error"] == pytest.approx(qf.mc_price(averaged, math.exp(-0.03), 0, 0.9)["std_error"])
    assert paired["n"] == 2_000


def test_the_payoff_pricer_draws_from_a_generator_and_converges_to_black_scholes_with_and_without_a_control():
    def draw(n, rng):
        return qf.gbm_paths(100.0, 0.03, 0.2, [0.5, 1.0], n, rng, "arithmetic", True)

    plain = qf.mc_payoff_price(draw, call_payoff, 40_000, 5, True, None, None, math.exp(-0.03), 1, 0.95)
    assert abs(plain["price"] - bs_call()) < 4 * plain["std_error"]
    controlled = qf.mc_payoff_price(draw, call_payoff, 40_000, 5, True, lambda s: s[:, -1], 100.0 * math.exp(0.03),
                                    math.exp(-0.03), 1, 0.95)
    assert abs(controlled["price"] - bs_call()) < 4 * controlled["std_error"]
    assert controlled["std_error"] < plain["std_error"] / 2 and controlled["beta"] > 0


def test_the_payoff_pricer_names_the_arguments_it_does_not_use():
    with pytest.raises(ValueError, match="None where paths is an array"):
        qf.mc_payoff_price(terminal(10, 1, False), call_payoff, 10, None, False, None, None, 1.0, 1, 0.95)
    with pytest.raises(ValueError, match="control_mean"):
        qf.mc_payoff_price(terminal(10, 1, False), call_payoff, None, None, False, None, 1.0, 1.0, 1, 0.95)


QL_TIMES = [0.0821917808219178, 0.16712328767123288, 0.2493150684931507, 0.33424657534246577, 0.41643835616438357,
            0.4986301369863014, 0.5835616438356165, 0.6657534246575343, 0.7506849315068493, 0.8328767123287671,
            0.9178082191780822, 1.0]           # QuantLib's monthly fixings, Actual/365 from 2024-01-15


def test_kemna_vorsts_control_variate_prices_the_arithmetic_asian_as_quantlibs_monte_carlo_does():
    """QuantLib 1.43, MCDiscreteArithmeticAPEngine (pseudorandom, the geometric control, 2^20 samples, seed 42): S = K
    = 100, r = 5%, vol 20%, twelve monthly fixings to a year: 6.154108358747 with error 0.000343; the control's mean
    is the exact discrete geometric price (QuantLib's analytic 5.938216292950, qf.asian_price 'geometric')."""
    geometric = qf.asian_price("geometric", "call", 100.0, 100.0, QL_TIMES, 1.0, 0.05, 0.0, 0.2)
    assert geometric == pytest.approx(5.938216292949869, rel=1e-9)
    paths = qf.gbm_paths(100.0, 0.05, 0.2, QL_TIMES, 40_000, 4, "arithmetic", False)
    d = math.exp(-0.05)
    arithmetic = d * np.maximum(paths.mean(axis=1) - 100.0, 0.0)
    control = d * np.maximum(np.exp(np.log(paths).mean(axis=1)) - 100.0, 0.0)
    found = qf.control_variate(arithmetic, control, geometric, "estimated", 1, 0.95)
    assert abs(found["estimate"] - 6.154108358747484) < 4 * math.hypot(found["std_error"], 0.000343)
    assert found["variance_ratio"] < 0.005 and 0.9 < found["beta"] < 1.2 and found["n"] == 40_000
    plain = arithmetic.std(ddof=1) / math.sqrt(arithmetic.size)
    assert found["std_error"] < plain / 10


def test_a_control_that_explains_the_draws_exactly_gives_their_mean_exactly():
    rng = np.random.default_rng(3)
    c = rng.standard_normal(500)
    found = qf.control_variate(2.0 + 3.0 * c, c, 0.0, "estimated", 1, 0.95)
    assert found["estimate"] == pytest.approx(2.0, abs=1e-12) and found["std_error"] < 1e-12
    assert found["beta"] == pytest.approx(3.0)
    zero = qf.control_variate(2.0 + 3.0 * c, c, 0.0, 0.0, 1, 0.95)
    assert zero["estimate"] == pytest.approx((2.0 + 3.0 * c).mean()) and zero["variance_ratio"] == pytest.approx(1.0)


def test_with_two_controls_the_variance_falls_by_one_minus_r_squared():
    rng = np.random.default_rng(4)
    c = rng.standard_normal((2_000, 2))
    y = 1.0 + c @ [0.5, -0.8] + 0.3 * rng.standard_normal(2_000)
    found = qf.control_variate(y, c, [0.0, 0.0], "estimated", 1, 0.95)
    fit = np.linalg.lstsq(np.c_[np.ones(2_000), c], y, rcond=None)[0]
    r2 = 1 - np.var(y - np.c_[np.ones(2_000), c] @ fit) / np.var(y)
    assert found["variance_ratio"] == pytest.approx(1 - r2, rel=1e-9) and np.allclose(found["beta"], fit[1:])


def test_importance_sampling_prices_a_deep_out_of_the_money_digital_with_a_fraction_of_the_variance():
    """A cash digital paying above 160 (S 100, r 3%, vol 20%, a year): the shift to the draw where it starts paying
    leaves the estimate unbiased and divides the variance by far more than ten (Glasserman 4.6)."""
    k = 160.0
    edge = (math.log(k / 100.0) - (0.03 - 0.02) * 1.0) / 0.2

    def pays(z):
        return math.exp(-0.03) * (100.0 * np.exp(0.01 + 0.2 * z[:, 0]) > k)

    exact = qf.digital_price("cash", "call", 100.0, k, 1.0, 0.03, 0.0, 0.2)
    shifted = qf.importance_shift(edge, 40_000, 1, 7)
    weighted = pays(shifted["normals"]) * shifted["weights"]
    plain = pays(np.random.default_rng(7).standard_normal((40_000, 1)))
    assert abs(weighted.mean() - exact) < 4 * weighted.std() / math.sqrt(weighted.size)
    assert plain.var() / weighted.var() > 20


def test_importance_weights_have_mean_one_and_undo_the_shift():
    found = qf.importance_shift([0.5, -0.3], 100_000, 2, 2)
    w, z = found["weights"], found["normals"]
    assert abs(w.mean() - 1.0) < 4 * w.std() / math.sqrt(w.size)
    assert np.all(np.abs((w[:, None] * z).mean(axis=0)) < 4 * (w[:, None] * z).std(axis=0) / math.sqrt(w.size))
    assert np.all(qf.importance_shift(0.0, 10, 3, 1)["weights"] == 1.0)
    assert np.array_equal(qf.importance_shift(0.0, 10, 3, 1)["normals"], np.random.default_rng(1).standard_normal((10, 3)))


def truncated_normal_variance(a, b):
    """Var(Z | a < Z < b) for a standard normal (the moments of the truncated normal)."""
    p = stats.norm.cdf(b) - stats.norm.cdf(a)
    pa, pb = stats.norm.pdf(a), stats.norm.pdf(b)
    ta, tb = (a * pa if np.isfinite(a) else 0.0), (b * pb if np.isfinite(b) else 0.0)
    return 1 + (ta - tb) / p - ((pa - pb) / p) ** 2


def test_stratified_draws_fill_each_stratum_and_their_error_is_the_strata_formula():
    """Proportional allocation, f(z) = z: each of the ten strata holds its tenth of the draws inside its quantiles,
    the estimate is near zero, and std_error is near sqrt(sum p_k^2 sigma_k^2 / n_k), sigma_k^2 the truncated
    normal's variance (Glasserman 4.3)."""
    found = qf.stratified_normals(5_000, 10, 2, "stratified", "proportional", None, lambda z: z[:, 0], 3, 1)
    edges = stats.norm.ppf(np.linspace(0, 1, 11))
    assert np.array_equal(np.bincount(found["stratum"]), np.full(10, 500))
    first = found["normals"][:, 0]
    assert np.all((first > edges[found["stratum"]]) & (first < edges[found["stratum"] + 1]))
    exact = math.sqrt(sum(0.01 * truncated_normal_variance(edges[k], edges[k + 1]) / 500 for k in range(10)))
    assert found["std_error"] == pytest.approx(exact, rel=0.1) and abs(found["estimate"]) < 4 * exact
    assert np.allclose(found["weights"], 1 / 5_000) and abs(found["normals"][:, 1].mean()) < 0.06


def test_stratification_estimates_a_known_mean_and_neyman_allocation_follows_the_stds():
    found = qf.stratified_normals(8_000, 20, 1, "stratified", "proportional", None, lambda z: np.exp(0.5 * z[:, 0]), 5, 1)
    assert abs(found["estimate"] - math.exp(0.125)) < 4 * found["std_error"]
    stds = np.r_[np.full(10, 1.0), np.full(10, 3.0)]
    neyman = qf.stratified_normals(4_000, 20, 1, "stratified", "optimal", stds, lambda z: z[:, 0], 5, 1)
    counts = np.bincount(neyman["stratum"])
    assert counts.sum() == 4_000 and np.all(counts >= 2) and abs(counts[10:].mean() / counts[:10].mean() - 3) < 0.05
    assert np.sum(neyman["weights"]) == pytest.approx(1.0)


def test_a_latin_hypercube_puts_one_draw_in_each_cell_of_every_margin():
    found = qf.stratified_normals(1_000, None, 3, "latin_hypercube", None, None, lambda z: z.sum(axis=1), 6, 1)
    cells = np.floor(stats.norm.cdf(found["normals"]) * 1_000).astype(int)
    assert all(np.array_equal(np.sort(cells[:, j]), np.arange(1_000)) for j in range(3))
    assert abs(found["estimate"]) < 0.01 and math.isnan(found["std_error"])


def test_unscrambled_sobol_normals_are_scipys_points_after_the_origin():
    found = qf.sobol_normals(8, 3, False, "sequential", 1, None, None, 1, 0.95)["normals"]
    points = qmc.Sobol(3, scramble=False).random_base2(4)[1:9]
    assert found.shape == (1, 8, 3) and np.allclose(found[0], stats.norm.ppf(points))


def test_scrambled_sobol_prices_a_call_near_black_scholes_with_a_tiny_error():
    def payoff(z):
        return math.exp(-0.03) * np.maximum(100.0 * np.exp(0.03 - 0.02 + 0.2 * z[:, 0]) - 100.0, 0.0)

    found = qf.sobol_normals(4_096, 1, True, "sequential", 16, payoff, 7, 1, 0.95)
    means = [payoff(each).mean() for each in found["normals"]]
    assert found["estimate"] == pytest.approx(np.mean(means), rel=1e-12)
    assert found["std_error"] == pytest.approx(np.std(means, ddof=1) / 4.0, rel=1e-9)
    assert abs(found["estimate"] - bs_call()) < 4 * found["std_error"] + 1e-4
    assert found["std_error"] < 14.7 / math.sqrt(16 * 4_096) / 10 and found["low"] < found["estimate"] < found["high"]


def test_the_brownian_bridge_order_gives_independent_increments_ending_at_the_first_coordinate():
    """W(steps) = sqrt(steps) z_0: each point's increments sum to sqrt(steps) times its first coordinate."""
    plain = qf.sobol_normals(1_024, 8, True, "sequential", 1, None, 2, 1, 0.95)["normals"][0]
    bridge = qf.sobol_normals(1_024, 8, True, "brownian_bridge", 1, None, 2, 1, 0.95)["normals"][0]
    assert np.allclose(bridge.sum(axis=1), math.sqrt(8) * plain[:, 0])
    assert np.allclose(bridge.var(axis=0), 1.0, atol=0.1) and np.abs(np.corrcoef(bridge.T) - np.eye(8)).max() < 0.1


def test_the_quantile_error_on_an_exact_normal_sample_is_its_formula():
    """Glasserman 9.1: the p-quantile's standard error sqrt(p (1 - p) / n) / f(q); on N(0, 1) samples the spread of
    the 0.99-quantile over independent samples matches it, the density known or estimated."""
    rng = np.random.default_rng(2)
    density = float(stats.norm.pdf(stats.norm.ppf(0.99)))
    exact = math.sqrt(0.99 * 0.01 / 5_000) / density
    known = qf.estimate_error("quantile", rng.standard_normal(5_000), None, 0.99, density, None, None, 0.95)
    assert known["std_error"] == pytest.approx(exact, rel=1e-12)
    spread = np.std([np.quantile(rng.standard_normal(5_000), 0.99, method="inverted_cdf") for _ in range(300)])
    assert spread == pytest.approx(exact, rel=0.15)
    kernel = qf.estimate_error("quantile", rng.standard_normal(5_000), None, 0.99, "kernel", None, None, 0.95)
    assert kernel["std_error"] == pytest.approx(exact, rel=0.25)
    sample = np.arange(1.0, 101.0)
    assert qf.estimate_error("quantile", sample, None, 0.95, 1.0, None, None, 0.95)["estimate"] == 95.0


def test_the_ratio_error_vanishes_for_a_fixed_ratio_and_matches_resampling_otherwise():
    rng = np.random.default_rng(5)
    x = 1.0 + rng.random(400)
    exact = qf.estimate_error("ratio", 2.5 * x, x, None, None, None, 1, 0.95)
    assert exact["estimate"] == pytest.approx(2.5) and exact["std_error"] < 1e-12
    y = 2.0 * x + rng.standard_normal(400)
    found = qf.estimate_error("ratio", y, x, None, None, None, 1, 0.95)
    again = [np.mean(2.0 * xs + rng.standard_normal(400)) / np.mean(xs) for xs in 1.0 + rng.random((400, 400))]
    assert found["std_error"] == pytest.approx(np.std(again), rel=0.2)


def test_batch_means_on_an_ar1_series_gives_the_long_run_error():
    """For x_t = 0.8 x_{t-1} + e_t the mean's error is sqrt(Var(x) (1 + rho) / (1 - rho) / n); batch means recover it
    where the plain iid error understates it threefold."""
    rng = np.random.default_rng(6)
    e, x = rng.standard_normal(60_000), np.zeros(60_000)
    for t in range(1, 60_000):
        x[t] = 0.8 * x[t - 1] + e[t]
    found = qf.estimate_error("batch_means", x, None, None, None, 30, 1, 0.95)
    exact = math.sqrt(1 / (1 - 0.64) * (1.8 / 0.2) / 60_000)
    assert found["std_error"] == pytest.approx(exact, rel=0.3) and found["std_error"] > 2.5 * x.std() / math.sqrt(x.size)


def bermudan_lattice(payoff, dates, per):
    """A CRR lattice for S 100, r 5%, vol 20%, exercise only at `dates` (equally spaced), `per` steps between them."""
    n = per * len(dates)
    dt = dates[-1] / n
    u, disc = math.exp(0.2 * math.sqrt(dt)), math.exp(-0.05 * dt)
    p = (math.exp(0.05 * dt) - 1 / u) / (u - 1 / u)
    v = payoff(100.0 * u ** (n - 2.0 * np.arange(n + 1)))
    for i in range(n - 1, -1, -1):
        v = disc * (p * v[:-1] + (1 - p) * v[1:])
        if i and i % per == 0:
            v = np.maximum(v, payoff(100.0 * u ** (i - 2.0 * np.arange(i + 1))))
    return v[0]


def put(s):
    return np.maximum(100.0 - s, 0.0)


def test_the_random_trees_low_and_high_estimators_bracket_a_bermudan_put():
    dates = [1 / 3, 2 / 3, 1.0]
    exact = bermudan_lattice(put, dates, 600)
    found = qf.american_bounds("random_tree", put, 100.0, 0.05, 0.0, 0.2, dates, 200, 16, None, None, None, 1, 0.95)
    assert found["lower"] <= exact <= found["upper"] and found["low"] < found["high"]
    assert found["low_std_error"] > 0 and found["n"] == 200
    one = qf.american_bounds("random_tree", put, 100.0, 0.05, 0.0, 0.2, [1.0], 400, 25, None, None, None, 3, 0.95)
    assert one["low"] == one["high"]
    assert abs(one["low"] - qf.bs_price("put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2)) < 4 * one["low_std_error"]


def random_tree_by_hand(trees, branching, dates, seed):
    """Broadie and Glasserman's (1997) high and low estimators, node by node, on the trees the function draws (each
    level's normals in one draw, a node's children together)."""
    rng = np.random.default_rng(seed)
    steps = np.diff(np.r_[0.0, dates])
    levels, s = [], np.full((trees, 1), 100.0)
    for h in steps:
        s = np.repeat(s, branching, axis=1) * np.exp((0.05 - 0.02 - 0.02) * h + 0.2 * math.sqrt(h)
                                                      * rng.standard_normal((trees, s.shape[1] * branching)))
        levels.append(s)
    highs, lows = [], []
    for tree in range(trees):
        def values(level, node):
            here = put(np.array([levels[level][tree, node]]))[0]
            if level == len(dates) - 1:
                return here, here
            disc = math.exp(-0.05 * steps[level + 1])
            kids = [values(level + 1, node * branching + c) for c in range(branching)]
            hi = max(here, disc * np.mean([k[0] for k in kids]))
            lo_kids = [disc * k[1] for k in kids]
            lo = np.mean([here if here > 0 and here >= (sum(lo_kids) - lo_kids[i]) / (branching - 1) else lo_kids[i]
                          for i in range(branching)])
            return hi, lo
        kids = [values(0, c) for c in range(branching)]
        disc = math.exp(-0.05 * steps[0])
        highs.append(disc * np.mean([k[0] for k in kids]))
        lows.append(disc * np.mean([k[1] for k in kids]))
    return np.mean(highs), np.mean(lows)


def test_the_random_trees_estimators_are_broadie_and_glassermans_node_by_node():
    dates = [0.25, 0.5, 0.75]
    found = qf.american_bounds("random_tree", put, 100.0, 0.05, 0.02, 0.2, dates, 3, 4, None, None, None, 11, 0.95)
    high, low = random_tree_by_hand(3, 4, dates, 11)
    assert found["high"] == pytest.approx(high, rel=1e-12) and found["low"] == pytest.approx(low, rel=1e-12)


def test_the_dual_bound_brackets_a_bermudan_put_and_names_its_regression_arguments():
    dates = [0.25, 0.5, 0.75, 1.0]
    exact = bermudan_lattice(put, dates, 400)
    found = qf.american_bounds("dual", put, 100.0, 0.05, 0.0, 0.2, dates, 1_500, None, 30, "polynomial", 2, 4, 0.95)
    assert found["lower"] <= exact <= found["upper"] and found["high"] >= found["low"] - 3 * found["low_std_error"]
    with pytest.raises(ValueError, match="basis and degree"):
        qf.american_bounds("dual", put, 100.0, 0.05, 0.0, 0.2, dates, 10, 5, 4, None, None, 1, 0.95)


@pytest.mark.slow
def test_the_bounds_bracket_a_ten_date_bermudan_and_a_call_without_dividends():
    dates = [k / 10 for k in range(1, 11)]
    exact = bermudan_lattice(put, dates, 200)
    found = qf.american_bounds("dual", put, 100.0, 0.05, 0.0, 0.2, dates, 4_000, None, 100, "laguerre", 3, 2, 0.95)
    assert found["lower"] <= exact <= found["upper"] and found["high"] - exact < 0.3

    def call(s):
        return np.maximum(s - 100.0, 0.0)

    tree = qf.american_bounds("random_tree", call, 100.0, 0.05, 0.0, 0.2, [0.5, 1.0], 2_000, 50, None, None, None, 5, 0.95)
    assert tree["lower"] <= qf.bs_price("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2) <= tree["upper"]


def quantlib_values():
    """The frozen values above, recomputed where QuantLib is installed (1.43 gave them): the discrete geometric price
    (AnalyticDiscreteGeometricAveragePriceAsianEngine) and the arithmetic one by MCDiscreteArithmeticAPEngine with the
    geometric control (pseudorandom, 2^20 samples, seed 42), monthly fixings Actual/365 from 2024-01-15."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    count = ql.Actual365Fixed()
    dates = [today + ql.Period(int(round(365 * k / 12)), ql.Days) for k in range(1, 13)]
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(100.0)), ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, count)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.2, count)))
    payoff, exercise = ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ql.EuropeanExercise(dates[-1])
    geometric = ql.DiscreteAveragingAsianOption(ql.Average.Geometric, 1.0, 0, dates, payoff, exercise)
    geometric.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process))
    arithmetic = ql.DiscreteAveragingAsianOption(ql.Average.Arithmetic, 0.0, 0, dates, payoff, exercise)
    arithmetic.setPricingEngine(ql.MCDiscreteArithmeticAPEngine(process, "pseudorandom", brownianBridge=False,
                                                                antitheticVariate=False, controlVariate=True,
                                                                requiredSamples=2 ** 20, seed=42))
    return {"times": [count.yearFraction(today, d) for d in dates], "geometric": geometric.NPV(),
            "arithmetic": arithmetic.NPV(), "error": arithmetic.errorEstimate()}


if __name__ == "__main__":
    print(quantlib_values())
