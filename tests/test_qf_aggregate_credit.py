"""qf.aggregate_loss, qf.creditrisk_plus and qf.creditmetrics (lane libgaps1; docs/research/library-syllabus-1011.md
rows 89 and 90, and the holdout read's compound-Poisson gap): the compound Poisson aggregate loss by FFT or Panjer's
recursion, CreditRisk+'s gamma-mixed Poisson by the recursion over exposure bands, and CreditMetrics' migration
simulation. References: the definitions (a direct convolution of the Poisson mixture), closed forms (E[S] = lambda
E[X], Var[S] = lambda E[X^2]; CreditRisk+'s mean and variance and its negative binomial count; the exponential
severity's Erlang mixture), the CreditMetrics Technical Document's BBB thresholds and the bivariate normal joint
default. Every input is this file's own or a textbook's; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import optimize, stats

import qf
from qf.aggregate import SET_POINTS, _var_es

pytestmark = pytest.mark.fast


def aggregate(law, lam, h, n, rule="rounding", method="fft", alphas=(0.99,), es="beyond", counts=None, cover=0.995):
    frequency, parameters = counts or ("poisson", {"lambda": lam})
    return qf.aggregate_loss(severity=law, frequency=frequency, frequency_parameters=parameters, step=h, points=n,
                             cover=cover, discretisation=rule, method=method, alphas=list(alphas), es_rule=es)


def test_panjer_and_fft_agree_on_one_grid_for_every_severity():
    """Panjer's recursion is exact on the grid; the FFT is exact but for the mass past n h wrapping round, negligible
    when the grid passes the far tail (Klugman, Panjer and Willmot 9.8)."""
    for law, lam, h in ((stats.gamma(a=3.0, scale=200.0), 15.0, 25.0), (stats.lognorm(s=0.9, scale=300.0), 8.0, 25.0),
                        (stats.pareto(b=3.5, scale=150.0), 10.0, 20.0)):
        fft, panjer = aggregate(law, lam, h, 2048), aggregate(law, lam, h, 2048, method="panjer")
        assert np.max(np.abs(fft["pmf"] - panjer["pmf"])) < 1e-9
        assert np.allclose(fft["var"], panjer["var"]) and np.allclose(fft["es"], panjer["es"], rtol=1e-9)


COUNTS = {   # each (a, b, 0) count with its law's pmf (scipy's parameters: nbinom's p = 1/(1 + beta))
    "poisson": ({"lambda": 1.3}, stats.poisson(1.3).pmf),
    "negative_binomial": ({"r": 2.5, "beta": 0.8}, stats.nbinom(2.5, 1 / 1.8).pmf),
    "binomial": ({"m": 6, "q": 0.35}, stats.binom(6, 0.35).pmf),
}


@pytest.mark.parametrize("frequency", sorted(COUNTS))
def test_the_recursion_is_the_count_mixture_of_convolutions(frequency):
    """The definition, by brute force: P(S = x) = sum_n P(N = n) f^{*n}(x), f^{*n} the n-fold convolution of the
    severity (Klugman, Panjer and Willmot 9.6, the (a, b, 0) class's recursion). A claim of 0, 1, 2 or 3 units on a
    unit grid ('upper' puts each atom on its own point, the atom at 0 testing g_0 = P_N(f_0)): both methods equal the
    direct sum for each count, the binomial's finite support included."""
    law = stats.rv_discrete(values=([0, 1, 2, 3], [0.1, 0.45, 0.3, 0.15]))
    f = np.array([0.1, 0.45, 0.3, 0.15])
    parameters, pmf = COUNTS[frequency]
    direct, power = np.zeros(64), np.array([1.0])
    for n in range(60):
        direct[:min(power.size, 64)] += pmf(n) * power[:64]
        power = np.convolve(power, f)
    for method in ("fft", "panjer"):
        found = aggregate(law, None, 1.0, 64, rule="upper", method=method, counts=(frequency, parameters))
        assert np.allclose(found["pmf"], direct, atol=1e-14), method
    lower, power = np.zeros(64), np.array([1.0])        # 'lower' moves each atom down a step, the one at 0 kept
    for n in range(60):
        lower[:min(power.size, 64)] += pmf(n) * power[:64]
        power = np.convolve(power, [0.55, 0.3, 0.15])
    found = aggregate(law, None, 1.0, 64, rule="lower", method="panjer", counts=(frequency, parameters))
    assert np.allclose(found["pmf"], lower, atol=1e-14)


def test_each_counts_moments_carry_to_the_aggregate():
    """E[S] = E[N] E[X], Var[S] = E[N] Var[X] + Var[N] E[X]^2 (Klugman, Panjer and Willmot 9.3), the counts'
    moments lambda, lambda; r beta, r beta (1 + beta); m q, m q (1 - q)."""
    law = stats.gamma(a=2.0, scale=5.0)
    for frequency, (parameters, pmf) in COUNTS.items():
        found = aggregate(law, None, 0.25, 1024, counts=(frequency, parameters), method="panjer")
        mean_n = sum(n * pmf(n) for n in range(200))
        var_n = sum(n * n * pmf(n) for n in range(200)) - mean_n ** 2
        assert found["mean_exact"] == pytest.approx(mean_n * 10.0)
        assert found["variance_exact"] == pytest.approx(mean_n * 50.0 + var_n * 10.0 ** 2)
        assert found["mean"] == pytest.approx(found["mean_exact"], rel=1e-4)
    with pytest.raises(ValueError, match="'r', 'beta'"):
        aggregate(law, None, 1.0, 64, counts=("negative_binomial", {"lambda": 2.0}))


def test_the_moments_are_lambda_times_the_severitys():
    """E[S] = lambda E[X] and Var[S] = lambda E[X^2] (Klugman, Panjer and Willmot 9.3): exactly for the grid's own
    severity, and the grid's moments near the law's (mean_exact, variance_exact) for a fine step."""
    law = stats.gamma(a=2.5, scale=80.0)
    found = aggregate(law, 20.0, 2.0, 8192, method="panjer")
    grid = found["grid"]
    cuts = law.cdf(grid + 1.0)
    f = np.diff(np.r_[0.0, cuts])
    assert found["mean"] == pytest.approx(20.0 * f @ grid, rel=1e-10)
    assert found["variance"] == pytest.approx(20.0 * f @ grid ** 2, rel=1e-8)
    assert found["mean_exact"] == pytest.approx(20.0 * 200.0) and found["variance_exact"] == pytest.approx(
        20.0 * (200.0 ** 2 + 2.5 * 80.0 ** 2))
    assert found["mean"] == pytest.approx(found["mean_exact"], rel=1e-4)


def test_an_exponential_severity_meets_its_erlang_mixture():
    """Closed form: with X exponential of mean theta, P(S <= x) = e^{-lambda} + sum_{n>=1} Pois(n; lambda)
    Gamma(n, theta).cdf(x). 'rounding' on a step of theta/400 lands within 1e-3; 'lower' and 'upper' bracket it (on a
    grid ten times the mean, past the tail the FFT would wrap round)."""
    lam, theta, h = 4.0, 100.0, 0.25
    x = np.array([100.0, 400.0, 800.0, 1500.0])
    exact = math.exp(-lam) + sum(stats.poisson.pmf(n, lam) * stats.gamma.cdf(x, n, scale=theta) for n in range(1, 80))
    at = (x / h).astype(int)
    found = {rule: aggregate(stats.expon(scale=theta), lam, h, 16384, rule=rule)["cdf"][at]
             for rule in ("rounding", "lower", "upper")}
    assert np.allclose(found["rounding"], exact, atol=1e-3)
    assert np.all(found["upper"] <= exact + 1e-12) and np.all(found["lower"] >= exact - 1e-12)


def test_the_plain_pareto_is_the_lomax_shifted_and_numpy_draws_the_lomax():
    """A plain Pareto of minimum x_m is x_m plus a Lomax of scale x_m; the aggregates' means differ by lambda x_m.
    numpy's Generator.pareto(a) draws the Lomax of scale 1 (Kolmogorov-Smirnov), so x_m (1 + draws) is the plain."""
    a, xm, lam = 3.0, 200.0, 6.0
    plain, lomax = aggregate(stats.pareto(b=a, scale=xm), lam, 10.0, 4096), aggregate(stats.lomax(c=a, scale=xm), lam,
                                                                                         10.0, 4096)
    assert plain["mean_exact"] - lomax["mean_exact"] == pytest.approx(lam * xm)
    draws = np.random.default_rng(3).pareto(a, 20_000)
    assert stats.kstest(draws, stats.lomax(c=a).cdf).pvalue > 0.01
    assert stats.kstest(xm * (1.0 + draws), stats.pareto(b=a, scale=xm).cdf).pvalue > 0.01
    assert stats.kstest(draws, stats.pareto(b=a).cdf).pvalue < 1e-6


def test_var_and_es_follow_their_named_rules_on_a_small_distribution():
    """VaR at 0.9 of P = (0.5, 0.3, 0.15, 0.05) on 0..3 is 2 (the CDF reaches 0.95 there); ES 'beyond' E[S | S > 2]
    = 3, 'at_or_beyond' (2 x 0.15 + 3 x 0.05)/0.2 = 2.25, 'exact' (3 x 0.05 + 2 x (0.95 - 0.9))/0.1 = 2.5."""
    grid, pmf = np.arange(4.0), np.array([0.5, 0.3, 0.15, 0.05])
    for rule, expected in (("beyond", 3.0), ("at_or_beyond", 2.25), ("exact", 2.5)):
        var, es = _var_es(grid, pmf, [0.9], rule)
        assert var[0] == 2.0 and es[0] == pytest.approx(expected), rule
    assert _var_es(grid, pmf, [0.8], "beyond")[0][0] == 1.0      # the CDF reaches 0.8 exactly at 1
    summed = np.array([0.7, 0.1, 0.1, 0.1])                       # its sum to 1 is 0.7999999999999999 in floats
    assert np.cumsum(summed)[1] < 0.8 and _var_es(grid, summed, [0.8], "beyond")[0][0] == 1.0



# -------------------------------------------------------------------------------- the grid's coverage (libfix1)


def covered(law, lam, cover, step=None, points=None, alphas=(0.9, 0.98), es="exact", method="fft"):
    return qf.aggregate_loss(severity=law, frequency="poisson", frequency_parameters={"lambda": lam}, step=step,
                             points=points, cover=cover, discretisation="rounding", method=method,
                             alphas=list(alphas), es_rule=es)


def test_a_poisson_exponential_aggregate_on_the_grid_cover_sets_meets_its_closed_forms():
    """Closed forms (Klugman, Panjer and Willmot 9.3 and 9.6): with X exponential of mean theta, E[e^{-sS}] =
    exp(lambda (1/(1 + theta s) - 1)); P(S <= x) = e^{-lambda} + sum_n Pois(n; lambda) Gamma(n, theta).cdf(x); and
    E[S; S > x] = sum_n Pois(n; lambda) n theta Gamma(n + 1, theta).sf(x), so ES 'exact' at alpha is that at VaR over
    1 - alpha. On the grid set from cover (step and points None), the grid's transform, VaR (to a step), ES and the
    cover's quantile (the tilted FFT search, Grubel and Hermesmeier 1999) meet them."""
    lam, theta, cover = 3.0, 100.0, 0.9995
    found = covered(stats.expon(scale=theta), lam, cover)
    for s in (1 / 500, 1 / 100, 1 / 25):
        assert found["pmf"] @ np.exp(-s * found["grid"]) == pytest.approx(math.exp(lam * (1 / (1 + theta * s) - 1)),
                                                                             rel=1e-5)
    n = np.arange(1, 120)
    weights = stats.poisson.pmf(n, lam)

    def cdf(x):
        return math.exp(-lam) + weights @ stats.gamma.cdf(x, n, scale=theta)

    for alpha, var, es in zip((0.9, 0.98), found["var"], found["es"]):
        exact = optimize.brentq(lambda x: cdf(x) - alpha, 1e-9, 1e5)
        assert abs(var - exact) <= found["step"]
        assert es == pytest.approx(weights @ (n * theta * stats.gamma.sf(exact, n + 1, scale=theta)) / (1 - alpha),
                                   rel=1e-5)
    reach = optimize.brentq(lambda x: cdf(x) - cover, 1e-9, 1e5)
    assert found["cover_quantile"] == pytest.approx(reach, rel=2e-3)
    assert found["points"] * found["step"] == pytest.approx(2.0 * found["cover_quantile"])


def test_a_poisson_lognormal_aggregate_on_the_grid_cover_sets_meets_monte_carlo():
    """The FFT on the grid cover sets against a seeded Monte Carlo of the same compound Poisson (N Poisson, then the
    claims): VaR and ES 'beyond' within a few standard errors, and the mass the simulation puts past the cover's
    quantile is 1 - cover."""
    law, lam, cover = stats.lognorm(s=1.1, scale=250.0), 4.0, 0.9995
    found = covered(law, lam, cover, alphas=(0.9, 0.975), es="beyond")
    draws = np.random.default_rng(11)
    counts = draws.poisson(lam, 400_000)
    claims = law.rvs(size=int(counts.sum()), random_state=draws)
    sums = np.bincount(np.repeat(np.arange(counts.size), counts), weights=claims, minlength=counts.size)
    for alpha, var, es in zip((0.9, 0.975), found["var"], found["es"]):
        level = np.quantile(sums, alpha, method="inverted_cdf")
        assert var == pytest.approx(level, rel=0.015) and es == pytest.approx(sums[sums > level].mean(), rel=0.02)
    assert abs(np.mean(sums > found["cover_quantile"]) - (1 - cover)) < 4 * math.sqrt(cover * (1 - cover) / 400_000)


def test_a_given_grid_short_of_cover_is_refused_and_one_past_it_is_the_plain_fft():
    """A grid whose end n h falls under the cover's quantile is refused, the shortfall said as a ratio; one past it
    gives g = ifft(P_N(fft(f))) on the given grid, as before the coverage was an argument."""
    law = stats.lognorm(s=1.1, scale=250.0)
    with pytest.raises(ValueError, match=r"reaches 0\.\d\d of the aggregate's 0\.9995 quantile \(cover\)"):
        covered(law, 4.0, 0.9995, step=10.0, points=256)
    found = covered(law, 4.0, 0.999, step=20.0, points=4096)
    f = qf.discretize_distribution(law=law, step=20.0, points=4096, method="rounding")["pmf"]
    assert np.array_equal(found["pmf"], np.maximum(np.fft.ifft(np.exp(4.0 * (np.fft.fft(f) - 1.0))).real, 0.0))
    assert found["step"] == 20.0 and found["points"] == 4096 and found["cover_quantile"] < 20.0 * 4096
    reach = found["cover_quantile"]
    with pytest.raises(ValueError, match=r"reaches 0\.90 of"):
        covered(law, 4.0, 0.999, step=0.9 * reach / 1024, points=1024)
    assert covered(law, 4.0, 0.999, step=1.02 * reach / 1024, points=1024)["cover_quantile"] == reach


def test_cover_sets_the_step_or_the_points_the_other_given_and_lies_above_every_alpha():
    law = stats.gamma(a=2.0, scale=30.0)
    by_points = covered(law, 5.0, 0.999, points=1024)
    assert by_points["points"] == 1024 and by_points["step"] * 1024 == pytest.approx(2 * by_points["cover_quantile"])
    by_step = covered(law, 5.0, 0.999, step=0.5, method="panjer")
    n = by_step["points"]
    assert by_step["step"] == 0.5 and n & (n - 1) == 0 and n * 0.5 >= 2 * by_step["cover_quantile"] > n * 0.25
    for cover in (0.95, 0.98, 1.0):
        with pytest.raises(ValueError, match="cover must lie"):
            covered(law, 5.0, cover)
    assert covered(law, 5.0, 0.999, method="panjer")["points"] == SET_POINTS["panjer"] < SET_POINTS["fft"]
    with pytest.raises(ValueError, match="a step this small"):
        covered(law, 5.0, 0.999, step=1e-4, method="panjer")


def test_the_covers_quantile_is_read_to_a_fine_step_where_the_mean_starts_the_search_far_above_it():
    """An atom at zero holding most of the mass (lambda 0.3, P(N = 0) = e^-0.3): the quantile just past it is far
    under E[S], where the search starts, and is still read to about 1/2,000 of itself (the Erlang mixture's)."""
    lam, theta, cover = 0.3, 100.0, 0.742
    n = np.arange(1, 60)
    weights = stats.poisson.pmf(n, lam)
    exact = optimize.brentq(lambda x: math.exp(-lam) + weights @ stats.gamma.cdf(x, n, scale=theta) - cover, 1e-9,
                            1e4)
    found = covered(stats.expon(scale=theta), lam, cover, alphas=(0.5,))["cover_quantile"]
    assert exact < lam * theta / 50 and found == pytest.approx(exact, rel=1 / 2048)


# --------------------------------------------------------------------------------------------- CreditRisk+


def plus(exposure, pd, unit, banding="up", preserve="expected_loss", rate_vol=0.0, points=200, alphas=(0.99,),
         es="beyond"):
    return qf.creditrisk_plus(exposure=exposure, pd=pd, unit=unit, banding=banding, preserve=preserve,
                              rate_vol=rate_vol, points=points, alphas=list(alphas), es_rule=es)


def test_creditrisk_plus_counts_defaults_as_a_negative_binomial_on_one_band():
    """CSFB (1997), A10: a gamma-mixed Poisson count of mean mu and rate std w mu is negative binomial (alpha =
    1/w^2, p = mu w^2 / (1 + mu w^2)); with every loss one band it is the loss distribution, exactly."""
    found = plus(np.full(12, 10.0), np.full(12, 0.04), 10.0, rate_vol=0.7, points=60)
    mu, w = 0.48, 0.7
    expected = stats.nbinom(1 / w ** 2, 1 / (1 + mu * w * w)).pmf(np.arange(60))
    assert np.allclose(found["pmf"], expected, atol=1e-15)


def test_creditrisk_plus_mean_and_variance_are_csfbs():
    """CSFB (1997), A7 and A9: mean L sum mu_i nu_i, the expected loss sum p_i E_i under 'expected_loss';
    variance L^2 (sum mu_i nu_i^2 + w^2 (sum mu_i nu_i)^2); the grid distribution's moments equal them."""
    e, p = np.array([120.0, 80.0, 250.0, 60.0, 150.0, 35.0]), np.array([0.02, 0.05, 0.01, 0.08, 0.03, 0.12])
    for rate_vol in (0.0, 0.5, 1.2):
        found = plus(e, p, 20.0, rate_vol=rate_vol, points=600)
        assert found["mean"] == pytest.approx(float(p @ e), rel=1e-12)
        nu, mu = found["bands"], found["intensities"]
        variance = 400.0 * (float(mu @ nu ** 2) + rate_vol ** 2 * float(mu @ nu) ** 2)
        assert found["variance"] == pytest.approx(variance, rel=1e-12)
        assert found["pmf"] @ found["grid"] == pytest.approx(found["mean"], rel=1e-9)
        assert found["pmf"] @ found["grid"] ** 2 - found["mean"] ** 2 == pytest.approx(variance, rel=1e-6)
        assert np.all(found["es"] >= found["var"])


def test_creditrisk_plus_without_rate_vol_is_the_poisson_sum_and_the_bands_follow_their_rule():
    """rate_vol 0: independent Poisson defaults, the loss a sum of nu_i times Poisson(mu_i) counts, convolved
    directly. 'up' rounds E/L up, 'nearest' to the nearest band (at least 1); 'default_probability' keeps mu_i =
    p_i."""
    e, p = np.array([25.0, 47.0, 10.0, 4.0]), np.array([0.1, 0.2, 0.05, 0.3])
    found = plus(e, p, 10.0, banding="nearest", preserve="default_probability", points=80)
    assert list(found["bands"]) == [2, 5, 1, 1] and np.array_equal(found["intensities"], p)
    direct = np.zeros(80)
    direct[0] = 1.0
    for nu, mu in zip(found["bands"], p):
        one = np.zeros(80)
        one[::nu] = stats.poisson.pmf(np.arange(-(-80 // nu)), mu)[:one[::nu].size]
        direct = np.convolve(direct, one)[:80]
    assert np.allclose(found["pmf"], direct, atol=1e-15)
    assert list(plus(e, p, 10.0, banding="up", points=80)["bands"]) == [3, 5, 1, 1]


# --------------------------------------------------------------------------------------------- CreditMetrics

BBB = [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0012, 0.0018]   # CreditMetrics TD (1997), a BBB's row


def test_the_thresholds_are_the_technical_documents_for_a_bbb_bond():
    """Gupton, Finger and Bhatia, CreditMetrics Technical Document (1997), ch. 8: a BBB's one-year row (AAA 0.02%,
    AA 0.33%, A 5.95%, BBB 86.93%, BB 5.30%, B 1.17%, CCC 0.12%, default 0.18%) gives the asset-return thresholds
    -2.91 (default), -2.75 (CCC), -2.18 (B), -1.49 (BB), 1.53 (BBB), 2.70 (A) and 3.54 (AA)."""
    found = qf.creditmetrics(transition=[BBB], initial=[0], values=np.ones((1, 8)), rho=0.2, n=10,
                             rng=np.random.default_rng(1))
    assert np.round(found["thresholds"][0, :7], 2).tolist() == [-2.91, -2.75, -2.18, -1.49, 1.53, 2.70, 3.54]
    assert found["thresholds"][0, -1] == math.inf


def test_each_obligor_migrates_by_its_row_and_two_default_together_by_the_bivariate_normal():
    """The thresholds keep each row's probabilities (they sum to one); two obligors of asset correlation rho
    default together with Phi_2(z_D, z_D; rho) (CreditMetrics TD, ch. 8); within the simulation's error."""
    row = [0.05, 0.75, 0.12, 0.08]
    n = 400_000
    found = qf.creditmetrics(transition=[row], initial=[0, 0], values=np.zeros((2, 4)), rho=0.4, n=n,
                             rng=np.random.default_rng(6))
    states = found["states"]
    frequency = np.bincount(states[:, 0], minlength=4) / n
    assert np.allclose(frequency, row, atol=4 * np.sqrt(0.25 / n))
    z = stats.norm.ppf(0.08)
    joint = stats.multivariate_normal([0.0, 0.0], [[1.0, 0.4], [0.4, 1.0]]).cdf([z, z])
    seen = np.mean((states[:, 0] == 3) & (states[:, 1] == 3))
    assert abs(seen - joint) < 4 * np.sqrt(joint / n)


def test_the_portfolio_value_averages_the_revalued_states_and_the_draws_follow_their_order():
    """E[V] = sum_i sum_s p_is v_is, within the simulation's error; the factor is drawn first, so a generator handed
    over after the factor's n normals gives the same idiosyncratic draws."""
    rows = [[0.9, 0.08, 0.02], [0.05, 0.85, 0.10]]
    values = np.array([[101.0, 99.0, 40.0], [102.0, 98.0, 45.0], [100.5, 99.5, 35.0]])
    initial = [0, 1, 1]
    found = qf.creditmetrics(transition=rows, initial=initial, values=values, rho=0.25, n=200_000,
                             rng=np.random.default_rng(8))
    expected = sum(np.dot(rows[r], values[i]) for i, r in enumerate(initial))
    assert abs(found["mean"] - expected) < 4 * found["std"] / math.sqrt(200_000)
    rng = np.random.default_rng(9)
    first = qf.creditmetrics(transition=rows, initial=initial, values=values, rho=0.0, n=50, rng=rng)
    again = np.random.default_rng(9)
    again.standard_normal(50)
    noise = again.standard_normal((50, 3))
    cuts = first["thresholds"]
    worst = np.array([[np.searchsorted(cuts[r], noise[k, i], side="right") for i, r in enumerate(initial)]
                      for k in range(50)])
    assert np.array_equal(first["states"], 2 - worst)


def test_creditmetrics_refuses_percent_rows_and_mismatched_values():
    with pytest.raises(ValueError, match="percent"):
        qf.creditmetrics(transition=[[90.0, 10.0]], initial=[0], values=[[1.0, 0.5]], rho=0.1, n=5,
                         rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="values"):
        qf.creditmetrics(transition=[[0.9, 0.1]], initial=[0, 0], values=[[1.0, 0.5]], rho=0.1, n=5,
                         rng=np.random.default_rng(1))
