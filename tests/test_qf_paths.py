"""qf.paths (runs/lanes/lib-montecarlo1-prompt.md; docs/research/library-syllabus-1011.md ranks 45, 48, 115, 85): exact
and truncated CIR paths against the transition's moments (Cox, Ingersoll and Ross 1985; Glasserman 2004, 3.4) and
the noncentral chi-square law; Heston paths against Andersen's (2008) test cases, priced by the library's Fourier
inversion and QuantLib 1.43's AnalyticHestonEngine alike; Euler and Milstein against the identities of GBM (the
Euler mean, Milstein's strong order; Glasserman 6.2); LIBOR market model paths repricing the discount bonds and
Black's caplet (Glasserman 3.7; Brigo and Mercurio ch. 6). The moved generators resolve on both shelves. Seeds fixed,
every parameter chosen freely, no unit's."""
import math

import numpy as np
import pytest
from scipy import stats

import qf
import qf.montecarlo as MC

pytestmark = pytest.mark.fast

CIR = dict(x0=0.03, kappa=0.5, theta=0.04, sigma=0.1)


def cir_moments(x0, kappa, theta, sigma, h):
    """The square-root diffusion's conditional mean and variance (Cox, Ingersoll and Ross 1985, eq. 19)."""
    e = math.exp(-kappa * h)
    return theta + (x0 - theta) * e, x0 * sigma ** 2 / kappa * (e - e * e) + theta * sigma ** 2 / (2 * kappa) * (1 - e) ** 2


def test_moved_generators_are_the_same_functions_on_paths_and_on_montecarlo():
    for name in ("gbm_paths", "ou_paths", "correlated_gbm_paths", "local_vol_paths"):
        assert getattr(qf, name) is getattr(qf.paths, name) is getattr(MC, name), name
        assert getattr(qf, name).__module__ == "qf.paths" and name not in MC.__all__, name


def test_exact_cir_matches_the_transitions_mean_and_variance_and_its_noncentral_chi_square_law():
    h = 1.0
    draws = qf.cir_paths(**CIR, times=[h], n_paths=40_000, rng=11, scheme="exact")
    x = draws["paths"][:, 0]
    mean, var = cir_moments(**CIR, h=h)
    assert abs(x.mean() - mean) < 4 * math.sqrt(var / x.size)
    assert x.var() == pytest.approx(var, rel=0.04)
    c = CIR["sigma"] ** 2 * (1 - math.exp(-CIR["kappa"] * h)) / (4 * CIR["kappa"])
    law = stats.ncx2(4 * CIR["kappa"] * CIR["theta"] / CIR["sigma"] ** 2, CIR["x0"] * math.exp(-CIR["kappa"] * h) / c,
                     scale=c)
    assert stats.kstest(x[:5000], law.cdf).pvalue > 0.001
    assert draws["feller"] is True and qf.cir_paths(0.03, 0.5, 0.04, 0.3, [1.0], 4, 1, "exact")["feller"] is False
    assert qf.cir_paths(0.03, 0.5, 0.04, 0.17, [1.0], 4, 1, "exact")["feller"] is True        # 0.04 >= 0.0289
    assert qf.cir_paths(0.03, 0.5, 0.04, 0.21, [1.0], 4, 1, "exact")["feller"] is False       # 0.04 < 0.0441


def test_exact_cir_is_exact_at_any_spacing_and_its_first_column_is_x0_at_time_zero():
    """Two half steps give the one step's law: the mean and variance at 1 from a path through 0.5."""
    found = qf.cir_paths(**CIR, times=[0.0, 0.5, 1.0], n_paths=40_000, rng=3, scheme="exact")["paths"]
    assert np.all(found[:, 0] == CIR["x0"]) and np.all(found >= 0)
    mean, var = cir_moments(**CIR, h=1.0)
    assert abs(found[:, 2].mean() - mean) < 4 * math.sqrt(var / 40_000)
    assert found[:, 2].var() == pytest.approx(var, rel=0.04)


def test_full_truncation_stays_non_negative_and_its_mean_converges_where_feller_fails():
    rough = dict(x0=0.04, kappa=1.0, theta=0.04, sigma=0.6)                  # 2 kappa theta < sigma^2
    found = qf.cir_paths(**rough, times=np.linspace(0.01, 1.0, 100), n_paths=10_000, rng=5, scheme="full_truncation")
    assert np.all(found["paths"] >= 0) and found["feller"] is False
    mean, var = cir_moments(**rough, h=1.0)
    assert abs(found["paths"][:, -1].mean() - mean) < 4 * math.sqrt(var / 10_000) + 2e-3


def test_cir_refuses_a_negative_start_and_names_its_schemes():
    with pytest.raises(ValueError, match="x0"):
        qf.cir_paths(-0.01, 0.5, 0.04, 0.1, [1.0], 4, 1, "exact")
    with pytest.raises(ValueError, match="'exact', 'full_truncation'"):
        qf.cir_paths(0.03, 0.5, 0.04, 0.1, [1.0], 4, 1, "euler")


HESTON = dict(spot=100.0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7, rate=0.03, dividend_yield=0.01)


def heston_price(spot, strike, maturity, rate, dividend_yield, v0, kappa, theta, sigma, rho):
    return qf.charfn_price(charfn=lambda u: qf.heston_charfn(u, spot, maturity, rate, dividend_yield, v0, kappa, theta,
                                                             sigma, rho, False),
                           option="call", spot=spot, strike=strike, maturity=maturity, rate=rate,
                           dividend_yield=dividend_yield, quadrature="gauss_legendre", upper=200.0, nodes=512)["price"]


def test_andersens_cases_priced_by_the_library_agree_with_his_table_and_quantlib():
    """Andersen (2008), Journal of Computational Finance 11(3), the cases' at-the-money calls: case I (kappa 0.5,
    theta 0.04, sigma 1, rho -0.9, T 10) 13.0847 and case II (kappa 0.3, sigma 0.9, rho -0.5, T 15) 16.6492, S = K =
    100, r = 0, v0 = 0.04; QuantLib 1.43's AnalyticHestonEngine gives 13.084670136992 and 16.649222920359."""
    one = heston_price(100.0, 100.0, 10.0, 0.0, 0.0, 0.04, 0.5, 0.04, 1.0, -0.9)
    two = heston_price(100.0, 100.0, 15.0, 0.0, 0.0, 0.04, 0.3, 0.04, 0.9, -0.5)
    assert one == pytest.approx(13.084670136992312, abs=1e-6) and round(one, 4) == 13.0847
    assert two == pytest.approx(16.649222920358667, abs=1e-6) and round(two, 4) == 16.6492


@pytest.mark.parametrize("scheme, psi_c", [("qe", 1.5), ("full_truncation", None)])
def test_heston_paths_keep_the_forward_and_price_a_call_near_the_fourier_price(scheme, psi_c):
    found = qf.heston_paths(**HESTON, times=[0.5, 1.0], substeps=16, n_paths=20_000, rng=7, scheme=scheme, psi_c=psi_c)
    end = found["spot"][:, -1]
    assert np.all(found["variance"] >= 0) and found["spot"].shape == (20_000, 2)
    forward = 100.0 * math.exp(0.02)
    assert abs(end.mean() - forward) < 4 * end.std() / math.sqrt(end.size)
    pay = math.exp(-0.03) * np.maximum(end - 100.0, 0.0)
    exact = heston_price(100.0, 100.0, 1.0, 0.03, 0.01, 0.04, 1.5, 0.04, 0.5, -0.7)
    assert abs(pay.mean() - exact) < 4 * pay.std() / math.sqrt(pay.size) + 0.05


@pytest.mark.parametrize("scheme, psi_c", [("qe", 1.5), ("full_truncation", None)])
def test_a_short_steps_price_and_variance_shocks_carry_rho(scheme, psi_c):
    """Over one short step from v0 the log price's change and the variance's change are correlated by rho (exactly
    in law for the truncated Euler step, to first order in the step for QE)."""
    for rho in (-0.7, 0.4):
        found = qf.heston_paths(**{**HESTON, "rho": rho}, times=[0.0, 0.01], substeps=1, n_paths=20_000, rng=6,
                                scheme=scheme, psi_c=psi_c)
        moves = np.log(found["spot"][:, 1]), found["variance"][:, 1]
        assert np.corrcoef(*moves)[0, 1] == pytest.approx(rho, abs=0.06)


def test_qe_matches_the_variances_conditional_mean_at_every_step():
    """Andersen's QE draws v(t + h) with the exact conditional mean theta + (v - theta) e^{-kappa h}; from v0 its mean
    at T is theta + (v0 - theta) e^{-kappa T} whatever the step."""
    p = dict(HESTON, v0=0.09)
    found = qf.heston_paths(**p, times=[1.0], substeps=4, n_paths=40_000, rng=2, scheme="qe", psi_c=1.5)["variance"][:, 0]
    mean = 0.04 + (0.09 - 0.04) * math.exp(-1.5)
    assert abs(found.mean() - mean) < 4 * found.std() / math.sqrt(found.size)


def test_heston_with_a_tiny_vol_of_variance_prices_as_black_scholes_at_the_mean_variance():
    p = dict(HESTON, sigma=1e-4, v0=0.09, kappa=2.0, rho=0.0)
    found = qf.heston_paths(**p, times=[1.0], substeps=8, n_paths=20_000, rng=4, scheme="qe", psi_c=1.5)["spot"][:, 0]
    mean_var = 0.04 + (0.09 - 0.04) * (1 - math.exp(-2.0)) / 2.0
    pay = math.exp(-0.03) * np.maximum(found - 100.0, 0.0)
    exact = qf.bs_price("call", 100.0, 100.0, 1.0, 0.03, 0.01, math.sqrt(mean_var))
    assert abs(pay.mean() - exact) < 4 * pay.std() / math.sqrt(pay.size)


def test_heston_names_psi_c_by_its_scheme():
    with pytest.raises(ValueError, match="psi_c"):
        qf.heston_paths(**HESTON, times=[1.0], substeps=2, n_paths=4, rng=1, scheme="qe", psi_c=None)
    with pytest.raises(ValueError, match="psi_c is None"):
        qf.heston_paths(**HESTON, times=[1.0], substeps=2, n_paths=4, rng=1, scheme="full_truncation", psi_c=1.5)


MU, VOL = 0.05, 0.4


def gbm_sde(scheme, steps, paths, seed):
    times = np.linspace(1.0 / steps, 1.0, steps)
    slope = (lambda t, x: VOL + 0.0 * x) if scheme == "milstein" else None
    found = qf.sde_paths(100.0, lambda t, x: MU * x, lambda t, x: VOL * x, slope, times, 1, paths, seed, scheme)
    w = np.cumsum(np.random.default_rng(seed).standard_normal((paths, steps)) * math.sqrt(1.0 / steps), axis=1)
    return found, 100.0 * np.exp((MU - VOL ** 2 / 2) + VOL * w[:, -1])


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_on_gbm_both_schemes_mean_is_x0_times_one_plus_mu_h_to_the_steps(scheme):
    """E[X_{k+1} | X_k] = X_k (1 + mu h) under both schemes (the Milstein term has mean zero), so E[X_n] = x0 (1 + mu
    h)^n exactly (Glasserman 6.2)."""
    found, _ = gbm_sde(scheme, 16, 40_000, 3)
    end = found[:, -1]
    assert abs(end.mean() - 100.0 * (1 + MU / 16) ** 16) < 4 * end.std() / math.sqrt(end.size)


def test_milsteins_strong_error_falls_as_the_step_and_eulers_as_its_root():
    """The mean absolute error at T against the exact GBM on the same Brownian path: a step a quarter as long cuts
    Euler's by about two and Milstein's by about four (Kloeden and Platen 10.3; Glasserman 6.2)."""
    errors = {}
    for scheme in ("euler", "milstein"):
        errors[scheme] = []
        for steps in (16, 64):
            found, exact = gbm_sde(scheme, steps, 4_000, 21)
            errors[scheme].append(np.mean(np.abs(found[:, -1] - exact)))
    euler, milstein = (errors[s][0] / errors[s][1] for s in ("euler", "milstein"))
    assert 1.6 < euler < 2.5 and 3.2 < milstein < 4.8
    assert errors["milstein"][1] < errors["euler"][1] / 5


def test_milstein_is_euler_where_the_diffusion_does_not_depend_on_x():
    args = dict(x0=0.5, drift=lambda t, x: 2.0 * (0.2 - x), diffusion=lambda t, x: 0.3 + 0.0 * x, times=[0.0, 0.5, 1.0],
                substeps=10, n_paths=500, rng=8)
    euler = qf.sde_paths(**args, diffusion_derivative=None, scheme="euler")
    milstein = qf.sde_paths(**args, diffusion_derivative=lambda t, x: 0.0 * x, scheme="milstein")
    assert np.array_equal(euler, milstein) and np.all(euler[:, 0] == 0.5)


FORWARDS = [0.03, 0.032, 0.034, 0.035, 0.036]
ACCRUALS = [0.5] * 5
CORR = [[math.exp(-0.1 * abs(i - j)) for j in range(5)] for i in range(5)]


def bonds():
    return np.r_[1.0, np.cumprod(1.0 / (1.0 + np.array(ACCRUALS) * np.array(FORWARDS)))]


@pytest.mark.parametrize("measure", ["spot", "terminal"])
@pytest.mark.parametrize("corrected", [False, True])
def test_lmm_deflators_reprice_the_discount_bonds(measure, corrected):
    """Glasserman (2004) 3.7: mean(deflator[:, k]) = P(0, T_k) = prod_{j < k} 1 / (1 + d_j L_j(0)), within the
    simulation's error and the log-Euler step's bias."""
    found = qf.lmm_paths(FORWARDS, ACCRUALS, [0.25] * 5, CORR, 1, 20_000, 3, measure, corrected)
    deflator = found["deflator"]
    error = deflator.std(axis=0) / math.sqrt(deflator.shape[0])
    assert np.all(np.abs(deflator.mean(axis=0) - bonds()) <= 4 * error + 2e-5)
    assert np.allclose(found["dates"], np.r_[0.0, np.cumsum(ACCRUALS)])
    assert np.all(found["forwards"][:, :, 0] == FORWARDS[0])                    # L_0 fixed today


def test_lmm_deflators_are_the_numeraires_ratio_read_off_the_paths_forwards():
    """Spot: 1 / prod_{j < k} (1 + d_j L_j(T_j)); terminal: P(0, T_M) / P(T_k, T_M) with P(T_k, T_M) = prod_{j >= k}
    1 / (1 + d_j L_j(T_k)), each forward read at the date (Glasserman 3.7)."""
    d = np.array(ACCRUALS)
    spot = qf.lmm_paths(FORWARDS, ACCRUALS, [0.25] * 5, CORR, 2, 50, 2, "spot", True)
    fixed = spot["forwards"][:, np.arange(5), np.arange(5)]
    assert np.allclose(spot["deflator"], 1.0 / np.cumprod(np.c_[np.ones(50), 1.0 + d * fixed], axis=1))
    terminal = qf.lmm_paths(FORWARDS, ACCRUALS, [0.25] * 5, CORR, 2, 50, 2, "terminal", True)
    for k in range(6):
        to_end = np.prod(1.0 / (1.0 + d[k:] * terminal["forwards"][:, k, k:]), axis=1)
        assert np.allclose(terminal["deflator"][:, k], bonds()[-1] / to_end)


def test_lmm_known_columns_are_exact_and_a_zero_vol_leaves_the_forwards_alone():
    spot = qf.lmm_paths(FORWARDS, ACCRUALS, [0.25] * 5, CORR, 2, 200, 1, "spot", False)
    terminal = qf.lmm_paths(FORWARDS, ACCRUALS, [0.25] * 5, CORR, 2, 200, 1, "terminal", False)
    assert np.allclose(spot["deflator"][:, 1], bonds()[1]) and np.allclose(terminal["deflator"][:, -1], bonds()[-1])
    flat = qf.lmm_paths(FORWARDS, ACCRUALS, [0.0] * 5, CORR, 2, 10, 1, "spot", True)
    assert np.allclose(flat["forwards"], np.array(FORWARDS)) and np.allclose(flat["deflator"], bonds())


@pytest.mark.slow
def test_lmm_prices_a_caplet_at_blacks_formula():
    """A caplet on L_3 fixing at T_3, paid d_3 max(L_3(T_3) - K, 0) at T_4: Black's formula with vol 0.25 and the bond
    P(0, T_4) (the LIBOR market model's consistency with Black, Brigo and Mercurio ch. 6), within the error."""
    found = qf.lmm_paths(FORWARDS, ACCRUALS, [0.25] * 5, CORR, 2, 40_000, 9, "spot", True)
    pay = 0.5 * np.maximum(found["forwards"][:, 3, 3] - 0.035, 0.0) * found["deflator"][:, 4]
    black = 0.5 * qf.black76_price("call", FORWARDS[3], 0.035, 1.5, 0.25, bonds()[4])
    assert abs(pay.mean() - black) < 4 * pay.std() / math.sqrt(pay.size) + 1e-6


def one_step_by_hand(forwards, accruals, vols, corr, measure, corrected, z):
    """The first period's log-Euler step written out from Glasserman's (2004) drifts, forward by forward."""
    size, h = len(forwards), accruals[0]
    w = np.linalg.cholesky(np.array(corr)) @ z

    def drift(f, i):
        terms = range(1, i + 1) if measure == "spot" else range(i + 1, size)
        total = sum(accruals[j] * f[j] * vols[j] * corr[i][j] / (1 + accruals[j] * f[j]) for j in terms)
        return (1.0 if measure == "spot" else -1.0) * vols[i] * total

    guess = [forwards[0]] + [forwards[i] * math.exp((drift(forwards, i) - vols[i] ** 2 / 2) * h
                                                   + vols[i] * math.sqrt(h) * w[i]) for i in range(1, size)]
    out = [forwards[0]]
    for i in range(1, size):
        mu = 0.5 * (drift(forwards, i) + drift(guess, i)) if corrected else drift(forwards, i)
        out.append(forwards[i] * math.exp((mu - vols[i] ** 2 / 2) * h + vols[i] * math.sqrt(h) * w[i]))
    return out


@pytest.mark.parametrize("measure", ["spot", "terminal"])
@pytest.mark.parametrize("corrected", [False, True])
def test_lmms_first_step_is_the_log_euler_step_written_out(measure, corrected):
    forwards, accruals, vols = [0.03, 0.04, 0.05], [0.5, 0.5, 0.5], [0.2, 0.3, 0.25]
    corr = [[1.0, 0.5, 0.3], [0.5, 1.0, 0.6], [0.3, 0.6, 1.0]]
    found = qf.lmm_paths(forwards, accruals, vols, corr, 1, 1, 5, measure, corrected)["forwards"][0, 1]
    z = np.random.default_rng(5).standard_normal((1, 2, 3))[0, 0]
    assert np.allclose(found, one_step_by_hand(forwards, accruals, vols, corr, measure, corrected, z), rtol=1e-13)


def test_lmm_takes_a_vol_by_period_and_refuses_a_bad_correlation():
    vols = np.full((5, 5), 0.2)
    vols[2:] = 0.3
    found = qf.lmm_paths(FORWARDS, ACCRUALS, vols, CORR, 1, 50, 4, "terminal", True)
    assert found["forwards"].shape == (50, 6, 5)
    with pytest.raises(ValueError, match="Cholesky"):
        qf.lmm_paths(FORWARDS[:2], ACCRUALS[:2], [0.2, 0.2], [[1.0, 1.0], [1.0, 1.0]], 1, 4, 1, "spot", False)


@pytest.mark.slow
def test_exact_cir_paths_price_the_cir_bond():
    """E[exp(-int_0^T r dt)] by the trapezoid on fine exact paths against the closed form (qf.short_rate_bond_price,
    Cox, Ingersoll and Ross 1985)."""
    times = np.linspace(0.0, 2.0, 401)
    paths = qf.cir_paths(**CIR, times=times, n_paths=20_000, rng=12, scheme="exact")["paths"]
    discount = np.exp(-np.trapezoid(paths, times, axis=1))
    exact = qf.short_rate_bond_price("cir", "none", 0.03, 2.0, 0.5, 0.04, 0.1, None, None)
    exact = exact["price"] if isinstance(exact, dict) else exact
    assert abs(discount.mean() - exact) < 4 * discount.std() / math.sqrt(discount.size) + 2e-5


@pytest.mark.slow
def test_qe_prices_andersens_case_one_at_an_eighth_of_a_year():
    """Andersen (2008) case I at the step 1/8: QE's call within its error of 13.0847 (its bias there is small)."""
    found = qf.heston_paths(100.0, 0.04, 0.5, 0.04, 1.0, -0.9, 0.0, 0.0, [10.0], 80, 100_000, 8, "qe", 1.5)["spot"][:, 0]
    pay = np.maximum(found - 100.0, 0.0)
    assert abs(pay.mean() - 13.0847) < 4 * pay.std() / math.sqrt(pay.size) + 0.05


def quantlib_values():
    """The frozen values above, recomputed where QuantLib is installed (1.43 gave them): Andersen's cases I and II by
    AnalyticHestonEngine, S = K = 100, r = q = 0, Actual/365 from 2024-01-15."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    found = []
    for kappa, theta, sigma, rho, years in ((0.5, 0.04, 1.0, -0.9, 10), (0.3, 0.04, 0.9, -0.5, 15)):
        process = ql.HestonProcess(flat, flat, ql.QuoteHandle(ql.SimpleQuote(100.0)), 0.04, kappa, theta, sigma, rho)
        option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
                                  ql.EuropeanExercise(today + ql.Period(int(years * 365), ql.Days)))
        option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process), 1e-10, 100000))
        found.append(option.NPV())
    return found


if __name__ == "__main__":
    print(quantlib_values())
