"""qf.density_expectation, qf.bivariate_normal_cdf, qf.multivariate_normal_cdf and qf.noncentral_chi2 (lane
lib-primitives1; the owner's in633): an expectation over a density by quadrature, the normal laws in two and more
dimensions, and the noncentral chi-squared law plain or as the CIR short rate's and the CEV price's. References: the
normal's moments and moment generating function (E[e^{sZ}] = e^{s^2/2}, E[Z^4] = 3), Gauss-Hermite's exactness below
degree 2n (Abramowitz and Stegun 25.4.46) and the lognormal call's expectation against qf.bs_price; M(0, 0; rho) = 1/4
+ arcsin(rho)/(2 pi), the product at rho = 0 and Owen's (1956) T-function form (scipy.special.owens_t); the trivariate
orthant 1/8 + sum arcsin(rho_ij)/(4 pi); the central chi-squared at noncentrality 0, the CIR rate's mean theta + (r_0
- theta) e^{-kappa T} (Cox, Ingersoll and Ross 1985); Schroder's CEV call (Hull 27.1) against the CEV law's own
density, which holds its mass and the forward. Every input is this file's own or a textbook's; no unit's values
appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import integrate, optimize, stats
from scipy.special import ive, owens_t

import qf

pytestmark = pytest.mark.fast

N01 = stats.norm(loc=0.0, scale=1.0)


def expect(function, law, method, limits=None, points=None, tolerance=None):
    return qf.density_expectation(function=function, law=law, method=method, limits=limits, points=points,
                                  tolerance=tolerance)


def test_gauss_hermite_holds_the_normals_moments_and_its_exactness_below_degree_2n():
    """E[e^{sZ}] = e^{s^2/2}; E[Z^4] = 3 exactly with 3 nodes (degree 4 < 6) and not with 2 (2 nodes give 1); a
    normal of mean 1 and std 2: E[X^2] = 1 + 4, the nodes scaled by sqrt(2) sigma."""
    assert expect(lambda z: np.exp(0.3 * z), N01, "gauss_hermite", points=20) == pytest.approx(math.exp(0.045),
                                                                                                rel=1e-14)
    assert expect(lambda z: z ** 4, N01, "gauss_hermite", points=3) == pytest.approx(3.0, rel=1e-14)
    assert expect(lambda z: z ** 4, N01, "gauss_hermite", points=2) == pytest.approx(1.0, rel=1e-14)
    assert expect(lambda x: x ** 2, stats.norm(loc=1.0, scale=2.0), "gauss_hermite", points=4) == pytest.approx(5.0)
    assert expect(lambda x: 7.0, N01, "gauss_hermite", points=5) == pytest.approx(7.0, rel=1e-14)


def test_the_adaptive_and_legendre_rules_integrate_over_a_law_a_density_or_nothing():
    """E[Z^4] = 3 over the law's support and over finite limits; a density written as a function, f(x) = 2x on (0,
    1), E[X] = 2/3; no law: the plain integral of x^2 over (0, 1) is 1/3; the overflow of g where f is 0 ignored."""
    assert expect(lambda z: z ** 4, N01, "adaptive", tolerance=1e-12) == pytest.approx(3.0, rel=1e-10)
    assert expect(lambda z: z ** 4, N01, "gauss_legendre", limits=(-12.0, 12.0), points=80) == pytest.approx(3.0,
                                                                                                           rel=1e-12)
    assert expect(lambda x: x, lambda x: 2.0 * x, "adaptive", limits=(0.0, 1.0), tolerance=1e-12) == pytest.approx(
        2.0 / 3.0, rel=1e-12)
    assert expect(lambda x: x ** 2, None, "gauss_legendre", limits=(0.0, 1.0), points=3) == pytest.approx(1 / 3,
                                                                                                        rel=1e-14)
    assert expect(lambda z: np.exp(800.0 * z), lambda x: float(abs(x) < 0.5), "adaptive", limits=(-1.0, 1.0),
                  tolerance=1e-10) == pytest.approx(math.sinh(400.0) / 400.0, rel=1e-8)


def test_the_lognormal_calls_discounted_expectation_is_bs_price():
    s, k, t, r, q, v = 100.0, 105.0, 0.75, 0.04, 0.01, 0.25

    def payoff(z):
        return math.exp(-r * t) * np.maximum(s * np.exp((r - q - v * v / 2) * t + v * math.sqrt(t) * z) - k, 0.0)

    found = expect(payoff, N01, "adaptive", tolerance=1e-12)
    assert found == pytest.approx(qf.bs_price(option="call", spot=s, strike=k, maturity=t, rate=r, dividend_yield=q,
                                              vol=v), rel=1e-10)


def test_density_expectation_refuses_a_rule_without_its_inputs():
    for change, message in (({"law": stats.t(df=4)}, "normal law"), ({"limits": (0.0, 1.0)}, "normal law"),
                            ({"tolerance": 1e-8}, "tolerance=None"), ({"method": "simpson"}, "must be one of"),
                            ({"method": "gauss_legendre", "limits": (0.0, np.inf)}, "finite limits"),
                            ({"method": "adaptive", "points": None, "tolerance": None}, "takes tolerance"),
                            ({"method": "adaptive", "points": None, "tolerance": 1e-8, "law": lambda x: 1.0},
                             "limits=None needs")):
        with pytest.raises(ValueError, match=message):
            qf.density_expectation(**{"function": lambda z: z, "law": N01, "method": "gauss_hermite", "limits": None,
                                      "points": 5, "tolerance": None, **change})


def owen(h, k, rho):
    """M(h, k; rho) by Owen's T function (Owen 1956), h and k not 0."""
    root = math.sqrt(1 - rho * rho)
    beta = 0.0 if h * k > 0 or (h * k == 0 and h + k >= 0) else 0.5
    return (0.5 * stats.norm.cdf(h) + 0.5 * stats.norm.cdf(k) - owens_t(h, (k - rho * h) / (h * root))
            - owens_t(k, (h - rho * k) / (k * root)) - beta)


@pytest.mark.parametrize("h, k, rho", [(0.5, -0.2, 0.3), (-1.0, 1.5, -0.7), (1.2, 0.8, 0.95), (-2.0, -2.5, 0.5),
                                       (0.3, 0.3, -0.99)])
def test_the_bivariate_normal_is_owens_t_form(h, k, rho):
    assert qf.bivariate_normal_cdf(h=h, k=k, rho=rho) == pytest.approx(owen(h, k, rho), abs=1e-13)


def test_the_bivariate_normals_closed_forms_and_limits():
    """M(0, 0; rho) = 1/4 + arcsin(rho)/(2 pi); rho = 0 the product N(h) N(k); rho = 1 N(min), rho = -1 max(N(h) +
    N(k) - 1, 0); arrays broadcast; |rho| > 1 refused."""
    rhos = np.array([-0.9, -0.3, 0.0, 0.4, 0.8])
    assert qf.bivariate_normal_cdf(h=0.0, k=0.0, rho=rhos) == pytest.approx(0.25 + np.arcsin(rhos) / (2 * np.pi),
                                                                            abs=1e-14)
    assert qf.bivariate_normal_cdf(h=0.7, k=-1.1, rho=0.0) == pytest.approx(stats.norm.cdf(0.7) * stats.norm.cdf(-1.1),
                                                                            abs=1e-14)
    assert qf.bivariate_normal_cdf(h=0.7, k=-1.1, rho=1.0) == pytest.approx(stats.norm.cdf(-1.1), abs=1e-15)
    assert qf.bivariate_normal_cdf(h=0.7, k=-0.1, rho=-1.0) == pytest.approx(
        stats.norm.cdf(0.7) + stats.norm.cdf(-0.1) - 1.0, abs=1e-15)
    assert qf.bivariate_normal_cdf(h=-0.7, k=-0.1, rho=-1.0) == 0.0
    assert isinstance(qf.bivariate_normal_cdf(h=0.1, k=0.2, rho=0.3), float)
    with pytest.raises(ValueError, match="between -1 and 1"):
        qf.bivariate_normal_cdf(h=0.0, k=0.0, rho=1.2)


@pytest.mark.parametrize("h, k, expected", [(np.inf, 0.0, 0.5), (np.inf, np.inf, 1.0),
                                           (-np.inf, 0.0, 0.0), (10000.0, 0.0, 0.5)])
def test_bivariate_normal_unrestricted_and_distant_limits(h, k, expected):
    """A normal variable's unrestricted bound marginalizes it; an extreme positive bound has negligible tail."""
    assert qf.bivariate_normal_cdf(h=h, k=k, rho=0.3) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("rho", [-0.99999999, 0.99999999])
def test_bivariate_normal_resolves_nearly_perfect_correlation(rho):
    """Owen's T form and the orthant arcsin identity remain exact near the singular limits."""
    assert qf.bivariate_normal_cdf(h=0.0, k=0.0, rho=rho) == pytest.approx(
        0.25 + math.asin(rho) / (2 * math.pi), abs=1e-14)
    assert qf.bivariate_normal_cdf(h=0.3, k=0.3, rho=rho) == pytest.approx(owen(0.3, 0.3, rho), abs=1e-13)


@pytest.mark.parametrize("variance", [0.0, -1.0])
def test_one_dimensional_normal_cdf_refuses_invalid_covariance(variance):
    with pytest.raises(ValueError, match="variance"):
        qf.multivariate_normal_cdf(upper=[0.0], mean=[0.0], matrix=[[variance]], matrix_kind="covariance",
                                   tolerance=1e-6, seed=7)


def test_the_trivariate_orthant_and_the_two_dimensional_case():
    """P(Z_1, Z_2, Z_3 <= 0) = 1/8 + (arcsin rho_12 + arcsin rho_13 + arcsin rho_23)/(4 pi), within the tolerance;
    two variables equal bivariate_normal_cdf; one, the normal CDF."""
    R = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    found = qf.multivariate_normal_cdf(upper=[0.0, 0.0, 0.0], mean=[0.0, 0.0, 0.0], matrix=R,
                                       matrix_kind="correlation", tolerance=1e-6, seed=1)
    assert found == pytest.approx(0.125 + (np.arcsin(0.5) + np.arcsin(0.2) + np.arcsin(0.3)) / (4 * np.pi), abs=5e-6)
    two = qf.multivariate_normal_cdf(upper=[0.5, -0.2], mean=[0.0, 0.0], matrix=[[1.0, 0.3], [0.3, 1.0]],
                                     matrix_kind="correlation", tolerance=1e-8, seed=1)
    assert two == pytest.approx(qf.bivariate_normal_cdf(h=0.5, k=-0.2, rho=0.3), abs=1e-7)
    assert qf.multivariate_normal_cdf(upper=[1.0], mean=[0.5], matrix=[[4.0]], matrix_kind="covariance",
                                      tolerance=1e-8, seed=0) == pytest.approx(stats.norm.cdf(0.25), abs=1e-15)


def test_a_covariance_matrix_is_the_correlation_one_on_standardised_limits():
    sd, mean = np.array([0.2, 1.5, 3.0]), np.array([0.1, -0.4, 2.0])
    R = np.array([[1.0, -0.4, 0.25], [-0.4, 1.0, 0.1], [0.25, 0.1, 1.0]])
    b = np.array([0.3, 0.5, 5.0])
    cov = qf.multivariate_normal_cdf(upper=b, mean=mean, matrix=np.outer(sd, sd) * R, matrix_kind="covariance",
                                     tolerance=1e-5, seed=3)
    corr = qf.multivariate_normal_cdf(upper=(b - mean) / sd, mean=[0.0, 0.0, 0.0], matrix=R,
                                      matrix_kind="correlation", tolerance=1e-5, seed=3)
    assert cov == pytest.approx(corr, abs=3e-5)
    for change, message in (({"matrix": [[2.0, 0.1], [0.1, 1.0]]}, "ones on its diagonal"),
                            ({"matrix": [[1.0, 0.1], [0.2, 1.0]]}, "symmetric"), ({"mean": [0.0]}, "entries"),
                            ({"matrix_kind": "variance"}, "must be one of")):
        with pytest.raises(ValueError, match=message):
            qf.multivariate_normal_cdf(**{"upper": [0.0, 0.0], "mean": [0.0, 0.0], "matrix": [[1.0, 0.1], [0.1, 1.0]],
                                          "matrix_kind": "correlation", "tolerance": 1e-6, "seed": 1, **change})


def law(what, x, form, parameters):
    return qf.noncentral_chi2(what=what, x=x, form=form, parameters=parameters)


def test_the_standard_form_is_scipys_ncx2_and_at_noncentrality_0_the_chi_squared():
    x = np.array([0.5, 2.0, 7.5])
    standard = {"df": 3.0, "noncentrality": 0.0}
    assert law("cdf", x, "standard", standard) == pytest.approx(stats.chi2.cdf(x, 3.0), rel=1e-12)
    shifted = {"df": 4.0, "noncentrality": 2.5}
    assert law("pdf", x, "standard", shifted) == pytest.approx(stats.ncx2.pdf(x, 4.0, 2.5), rel=1e-12)
    assert law("cdf", x, "standard", shifted) + law("sf", x, "standard", shifted) == pytest.approx(np.ones(3))
    assert law("cdf", law("quantile", 0.3, "standard", shifted), "standard", shifted) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="'df', 'noncentrality'"):
        law("cdf", 1.0, "standard", {"df": 3.0})
    with pytest.raises(ValueError, match="a probability"):
        law("quantile", 1.5, "standard", shifted)


CIR = {"kappa": 0.8, "theta": 0.05, "sigma": 0.12, "rate": 0.03, "horizon": 2.0}


def test_the_cir_forms_law_has_the_cir_mean_and_its_quantile_inverts_its_cdf():
    """E[r_T] = theta + (r_0 - theta) e^{-kappa T}, the density integrating to 1; 2c r_T's own law (c = 2 kappa /
    (sigma^2 (1 - e^{-kappa T}))) at 2c x."""
    density = lambda x: law("pdf", x, "cir", CIR)       # noqa: E731
    mass = integrate.quad(density, 0.0, np.inf)[0]
    mean = integrate.quad(lambda x: x * density(x), 0.0, np.inf)[0]
    assert mass == pytest.approx(1.0, abs=1e-9)
    assert mean == pytest.approx(0.05 + (0.03 - 0.05) * math.exp(-0.8 * 2.0), rel=1e-8)
    c = 2 * 0.8 / (0.12 ** 2 * -math.expm1(-0.8 * 2.0))
    standard = {"df": 4 * 0.8 * 0.05 / 0.12 ** 2, "noncentrality": 2 * c * 0.03 * math.exp(-1.6)}
    assert law("cdf", 0.04, "cir", CIR) == pytest.approx(law("cdf", 2 * c * 0.04, "standard", standard), rel=1e-14)
    u = np.array([0.05, 0.5, 0.95])
    assert law("cdf", law("quantile", u, "cir", CIR), "cir", CIR) == pytest.approx(u, rel=1e-10)


def cev(beta):
    return {"spot": 100.0, "rate": 0.05, "dividend_yield": 0.01, "sigma": 0.2 * 100.0 ** (1 - beta), "beta": beta,
            "maturity": 1.0}


def schroder_call(parameters, strike):
    """Schroder/Hull below beta 1; Lindsay and Brecher (2012), eq. 21's payoff expectation above beta 1."""
    s, r, q, sigma, beta, t = (parameters[name] for name in ("spot", "rate", "dividend_yield", "sigma", "beta",
                                                             "maturity"))
    y = 2 * (r - q) * (beta - 1) * t
    v = sigma ** 2 * t * (math.expm1(y) / y if y != 0 else 1.0)
    a = (strike * math.exp(-(r - q) * t)) ** (2 * (1 - beta)) / ((1 - beta) ** 2 * v)
    b, c = 1 / (1 - beta), s ** (2 * (1 - beta)) / ((1 - beta) ** 2 * v)

    def F(z, df, nc):
        return law("cdf", z, "standard", {"df": df, "noncentrality": nc})
    if beta < 1:
        return s * math.exp(-q * t) * (1 - F(a, b + 2, c)) - strike * math.exp(-r * t) * F(c, b, a)
    return s * math.exp(-q * t) * (F(c, -b, 0) - F(c, -b, a)) - strike * math.exp(-r * t) * F(a, 2 - b, c)


def test_the_cev_forms_law_prices_schroders_call():
    """The fast twin of the case below: one elasticity, one strike."""
    p = cev(0.5)
    found = math.exp(-0.05) * integrate.quad(lambda x: (x - 100.0) * law("pdf", x, "cev", p), 100.0, np.inf)[0]
    assert found == pytest.approx(schroder_call(p, 100.0), rel=1e-9)


def test_superlinear_cev_has_the_payoff_expectation_and_martingale_defect():
    """Lindsay and Brecher (2012), eqs. 18 and 21: beta 2, S0=sigma=T=1 and zero carry is reciprocal BES(3).
    Its mean is 2 Phi(1)-1; the parity solution exceeds the payoff expectation by 2 Phi(-1)."""
    p = {"spot": 1.0, "rate": 0.0, "dividend_yield": 0.0, "sigma": 1.0, "beta": 2.0, "maturity": 1.0}
    density = lambda x: law("pdf", x, "cev", p)
    mean = integrate.quad(lambda x: x * density(x), 0.0, np.inf)[0]
    call = integrate.quad(lambda x: (x - 1.0) * density(x), 1.0, np.inf)[0]
    assert mean == pytest.approx(2.0 * stats.norm.cdf(1.0) - 1.0, abs=1e-12)
    assert call == pytest.approx(schroder_call(p, 1.0), abs=1e-12)
    assert call == pytest.approx(0.07314106992168894, abs=1e-12)


def test_superlinear_cev_tiny_quantiles_against_the_reciprocal_bessel_normal_form():
    """Reciprocal BES(3) radial-tail probability, by integrating the shifted normal radial density; Lindsay and
    Brecher's squared-Bessel representation. The normal formula avoids subtracting a probability from one."""
    p = {"spot": 1.0, "rate": 0.0, "dividend_yield": 0.0, "sigma": 1.0, "beta": 2.0, "maturity": 1.0}

    def probability(x):
        y = 1.0 / x
        return stats.norm.sf(y - 1.0) + stats.norm.sf(y + 1.0) + stats.norm.pdf(y - 1.0) - stats.norm.pdf(y + 1.0)

    for u in (1e-18, 1e-20):
        expected = optimize.brentq(lambda x: probability(x) - u, 0.05, 0.2, xtol=1e-14)
        found = law("quantile", u, "cev", p)
        assert found == pytest.approx(expected, rel=1e-11)
        assert law("cdf", found, "cev", p) == pytest.approx(u, rel=1e-11, abs=0.0)


def test_square_root_cev_tail_density_against_the_bessel_transition():
    """Schroder's beta 1/2 process with sigma 2 is squared BES(0). Its positive transition density is
    .5 sqrt(S0/x) exp(-(S0+x)/2) I_1(sqrt(S0*x)), independently of a CDF subtraction."""
    p = {"spot": 100.0, "rate": 0.0, "dividend_yield": 0.0, "sigma": 2.0, "beta": 0.5, "maturity": 1.0}
    x = np.array([0.001, 0.1, 1.0])
    z = np.sqrt(100.0 * x)
    expected = 0.5 * np.sqrt(100.0 / x) * np.exp(-(100.0 + x) / 2.0 + z) * ive(1, z)
    assert law("pdf", x, "cev", p) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.slow
@pytest.mark.parametrize("beta", [0.5, 0.8, 1.3])
def test_the_cev_forms_law_prices_schroders_call_holds_its_mass_and_the_forward(beta):
    """The risk-neutral law of S_T under the CEV process: its density (with the mass absorbed at 0 below beta 1)
    sums to 1, its mean equals the forward below beta 1 and has the strict-local-martingale defect above it
    (Lindsay and Brecher, eq. 18), and its quantile inverts its CDF."""
    p = cev(beta)
    density = lambda x: law("pdf", x, "cev", p)        # noqa: E731
    atom = law("cdf", 0.0, "cev", p) if beta < 1 else 0.0
    assert integrate.quad(density, 0.0, np.inf, limit=200)[0] + atom == pytest.approx(1.0, abs=1e-9)
    mean = 100.0 * math.exp(0.04)
    if beta > 1:
        y = 2 * 0.04 * (beta - 1)
        v = p["sigma"] ** 2 * math.expm1(y) / (2 * 0.04 * (beta - 1))
        c = 100.0 ** (2 * (1 - beta)) / ((1 - beta) ** 2 * v)
        mean *= stats.chi2.cdf(c, 1 / (beta - 1))
    assert integrate.quad(lambda x: x * density(x), 0.0, np.inf, limit=200)[0] == pytest.approx(mean, rel=1e-9)
    for strike in (80.0, 100.0, 125.0):
        found = math.exp(-0.05) * integrate.quad(lambda x: (x - strike) * density(x), strike, np.inf, limit=200)[0]
        assert found == pytest.approx(schroder_call(p, strike), rel=1e-9), strike
    u = np.array([0.1, 0.5, 0.9])
    assert law("cdf", law("quantile", u, "cev", p), "cev", p) == pytest.approx(u, rel=1e-9)
    assert law("sf", 90.0, "cev", p) == pytest.approx(1.0 - law("cdf", 90.0, "cev", p), abs=1e-15)


@pytest.mark.parametrize("beta", [0.5, 1.3])
def test_the_cev_forms_quantile_inverts_its_cdf_either_side_of_beta_1(beta):
    p = cev(beta)
    u = np.array([0.1, 0.5, 0.9])
    assert law("cdf", law("quantile", u, "cev", p), "cev", p) == pytest.approx(u, rel=1e-9)
    assert law("sf", 90.0, "cev", p) == pytest.approx(1.0 - law("cdf", 90.0, "cev", p), abs=1e-15)
    assert law("pdf", 90.0, "cev", p) == pytest.approx((law("cdf", 90.0 + 1e-4, "cev", p)
                                                        - law("cdf", 90.0 - 1e-4, "cev", p)) / 2e-4, rel=1e-6)


def test_the_cev_forms_mass_at_zero_and_its_refusals():
    """Below beta 1 the price is absorbed at 0 with probability the chi-squared's survival at c with b degrees of
    freedom: a quantile below it is 0; beta 1 is the lognormal law and refused."""
    p = {**cev(0.5), "sigma": 8.0}
    atom = law("cdf", 0.0, "cev", p)
    assert 0.001 < atom < 0.5 and law("quantile", atom / 2, "cev", p) == 0.0
    assert law("cdf", -1.0, "cev", p) == 0.0 and law("pdf", 0.0, "cev", p) == 0.0
    with pytest.raises(ValueError, match="beta"):
        law("cdf", 100.0, "cev", cev(1.0))


def test_the_cir_forms_density_over_consecutive_pairs_is_the_transition_density():
    """The fast twin of the fit below: the 'cir' form's density with the previous rates as an array is, pair by
    pair, 2c times the noncentral chi-squared density at 2c r_next, df 4 kappa theta / sigma^2, noncentrality 2c
    r_prev e^{-kappa dt} (Cox, Ingersoll and Ross 1985)."""
    dt, p = 1 / 52, {"kappa": 1.5, "theta": 0.04, "sigma": 0.08}
    prev, nxt = np.array([0.03, 0.035, 0.041, 0.038]), np.array([0.035, 0.041, 0.038, 0.036])
    c = 2 * 1.5 / (0.08 ** 2 * -math.expm1(-1.5 * dt))
    want = 2 * c * stats.ncx2.pdf(2 * c * nxt, 4 * 1.5 * 0.04 / 0.08 ** 2, 2 * c * prev * math.exp(-1.5 * dt))
    assert law("pdf", nxt, "cir", {**p, "rate": prev, "horizon": dt}) == pytest.approx(want, rel=1e-13)


@pytest.mark.slow
def test_the_cir_transition_likelihood_through_mle_fit_is_cir_fits_exact_mle():
    """A CIR path drawn exactly, each rate the 'cir' form's quantile of a uniform given the one before; the
    likelihood of its consecutive pairs by the 'cir' form's density (the previous rates as an array) maximised by
    mle_fit is cir_fit's 'mle_exact' (Cox, Ingersoll and Ross 1985; Brigo and Mercurio 3.2.3)."""
    dt, true = 1 / 52, {"kappa": 1.5, "theta": 0.04, "sigma": 0.08}
    rng = np.random.default_rng(11)
    path = [0.03]
    for u in rng.random(400):
        path.append(float(law("quantile", u, "cir", {**true, "rate": path[-1], "horizon": dt})))
    path = np.array(path)

    def transition(p, nxt, prev):
        return np.log(law("pdf", nxt, "cir", {**p, "rate": prev, "horizon": dt}))

    fit = qf.mle_fit(log_density=transition, data=path, form="transitions", starts=dict(true),
                     transforms={"kappa": "positive", "theta": "positive", "sigma": "positive"}, optimiser="bfgs",
                     errors="none", checks={"feller": lambda p: 2 * p["kappa"] * p["theta"] >= p["sigma"] ** 2})
    held = qf.cir_fit(rates=path, dt=dt, method="mle_exact",
                      bounds={"kappa": (0.01, 10.0), "theta": (0.001, 0.2), "sigma": (0.005, 0.5)}, starts=4, rng=5)
    for name in ("kappa", "theta", "sigma"):
        assert fit["params"][name] == pytest.approx(held[name], rel=1e-4), name
    assert fit["loglik"] == pytest.approx(held["log_likelihood"], rel=1e-9)
    assert fit["flags"] == {"feller": held["feller_ratio"] >= 1}
