"""qf.mle_fit, qf.moments_fit and qf.least_squares_fit (lane lib-primitives1; the owner's in633): a fit of any model
from a log-density, moment conditions or residuals the caller writes as a function. References: the normal's,
exponential's and Bernoulli's maximum-likelihood closed forms and their asymptotic errors (Casella and Berger 7.2.2
and 10.1), the exponential's outer-product and sandwich errors by hand; the Ornstein-Uhlenbeck transition likelihood
against qf.ou_fit's exact scheme; a gamma law's method-of-moments estimates (Rice, Mathematical Statistics and Data
Analysis, 8.4); OLS by its moments against qf.ols, its errors White's HC0 and qf.newey_west_ols' HAC; the NIST StRD
Misra1a certified values (both starts); weighted least squares' normal equations. Every input is this file's own or
a textbook's; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import stats

import qf

pytestmark = pytest.mark.fast

RNG = np.random.default_rng(20261012)
NORMAL = RNG.normal(1.5, 2.0, 400)
EXPONENTIAL = RNG.exponential(0.8, 300)


def normal_fit(errors, optimiser="bfgs"):
    return qf.mle_fit(log_density=lambda p, x: stats.norm.logpdf(x, p["mu"], np.sqrt(p["var"])), data=NORMAL,
                      form="observations", starts={"mu": 0.0, "var": 1.0},
                      transforms={"mu": "free", "var": "positive"}, optimiser=optimiser, errors=errors, checks=None)


def exponential_fit(errors):
    return qf.mle_fit(log_density=lambda p, x: np.log(p["rate"]) - p["rate"] * x, data=EXPONENTIAL,
                      form="observations", starts={"rate": 1.0}, transforms={"rate": "positive"}, optimiser="bfgs",
                      errors=errors, checks=None)


def test_the_normal_mle_is_the_mean_and_the_ddof_0_variance_with_their_asymptotic_errors():
    """mu-hat = x-bar, sigma^2-hat = mean (x - x-bar)^2; errors sigma/sqrt(n) and sigma^2 sqrt(2/n) (the inverse
    Fisher information); l = -n/2 (ln(2 pi sigma^2) + 1), AIC = 2k - 2l, BIC = k ln n - 2l."""
    n, mean, var = NORMAL.size, NORMAL.mean(), NORMAL.var()
    fit = normal_fit("hessian")
    assert fit["params"]["mu"] == pytest.approx(mean, rel=1e-8)
    assert fit["params"]["var"] == pytest.approx(var, rel=1e-8)
    assert fit["std_errors"]["mu"] == pytest.approx(math.sqrt(var / n), rel=1e-6)
    assert fit["std_errors"]["var"] == pytest.approx(var * math.sqrt(2.0 / n), rel=1e-6)
    loglik = -n / 2 * (math.log(2 * math.pi * var) + 1)
    assert fit["loglik"] == pytest.approx(loglik, rel=1e-12)
    assert fit["aic"] == pytest.approx(4 - 2 * loglik, rel=1e-12)
    assert fit["bic"] == pytest.approx(2 * math.log(n) - 2 * loglik, rel=1e-12)
    assert fit["n_obs"] == n and fit["converged"] and fit["flags"] == {}
    assert fit["covariance"][0, 1] == pytest.approx(0.0, abs=1e-6 * var / n)


@pytest.mark.parametrize("optimiser", ["nelder_mead", "powell"])
def test_each_optimiser_reaches_the_same_normal_mle(optimiser):
    fit = normal_fit("none", optimiser)
    assert fit["params"]["mu"] == pytest.approx(NORMAL.mean(), rel=1e-6)
    assert fit["params"]["var"] == pytest.approx(NORMAL.var(), rel=1e-6)
    assert np.all(np.isnan(fit["covariance"])) and fit["converged"]


def test_the_exponential_rate_and_its_three_kinds_of_error():
    """lambda-hat = 1/x-bar. Minus the Hessian n/lambda^2 gives lambda/sqrt(n); the scores' outer products B = sum
    (1/lambda - x_i)^2 give 1/sqrt(B); the sandwich H^-1 B H^-1 gives lambda^2 sqrt(B)/n."""
    n = EXPONENTIAL.size
    rate = 1.0 / EXPONENTIAL.mean()
    outer = float(np.sum((1.0 / rate - EXPONENTIAL) ** 2))
    expected = {"hessian": rate / math.sqrt(n), "outer_product": 1.0 / math.sqrt(outer),
                "sandwich": rate ** 2 * math.sqrt(outer) / n}
    for errors, se in expected.items():
        fit = exponential_fit(errors)
        assert fit["params"]["rate"] == pytest.approx(rate, rel=1e-8), errors
        assert fit["std_errors"]["rate"] == pytest.approx(se, rel=1e-6), errors
    assert len({round(value, 6) for value in expected.values()}) == 3


def test_the_ornstein_uhlenbeck_transition_likelihood_is_ou_fits_exact_scheme():
    """The exact OU transition, X_{t+dt} | X_t normal with mean theta + (X_t - theta) e^{-kappa dt} and variance
    sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa), is the AR(1)'s Gaussian likelihood reparameterised: its maximum is
    ou_fit's exact scheme with the ddof-0 residual variance. Feller's condition returned as a flag."""
    dt, kappa, theta, sigma = 1 / 52, 1.2, 0.05, 0.02
    path = [0.03]
    shocks = RNG.standard_normal(500)
    for z in shocks:
        mean = theta + (path[-1] - theta) * math.exp(-kappa * dt)
        path.append(mean + sigma * math.sqrt(-math.expm1(-2 * kappa * dt) / (2 * kappa)) * z)
    path = np.array(path)

    def transition(p, nxt, prev):
        mean = p["theta"] + (prev - p["theta"]) * np.exp(-p["kappa"] * dt)
        sd = p["sigma"] * np.sqrt(-np.expm1(-2 * p["kappa"] * dt) / (2 * p["kappa"]))
        return stats.norm.logpdf(nxt, mean, sd)

    fit = qf.mle_fit(log_density=transition, data=path, form="transitions",
                     starts={"kappa": 1.0, "theta": 0.04, "sigma": 0.03},
                     transforms={"kappa": "positive", "theta": "free", "sigma": "positive"}, optimiser="bfgs",
                     errors="hessian", checks={"feller": lambda p: 2 * p["kappa"] * p["theta"] >= p["sigma"] ** 2,
                                               "fast": lambda p: p["kappa"] > 10.0})
    held = qf.ou_fit(path, dt=dt, ddof=0, scheme="exact")
    for name in ("kappa", "theta", "sigma"):
        assert fit["params"][name] == pytest.approx(held[name], rel=1e-6), name
    assert fit["n_obs"] == path.size - 1 and fit["flags"] == {"feller": True, "fast": False}
    assert all(value > 0 for value in fit["std_errors"].values())


def test_a_probability_between_0_and_1_and_an_interval_transform_give_the_bernoulli_mle():
    """p-hat = the share of ones, error sqrt(p (1 - p) / n), whether p is optimised through the logit or an
    interval's; a start outside its bound is refused."""
    x = (RNG.random(250) < 0.3).astype(float)
    p_hat = x.mean()
    for transform in ("unit_interval", (0.1, 0.9)):
        fit = qf.mle_fit(log_density=lambda p, x: x * np.log(p["p"]) + (1 - x) * np.log1p(-p["p"]), data=x,
                         form="observations", starts={"p": 0.5}, transforms={"p": transform}, optimiser="bfgs",
                         errors="hessian", checks=None)
        assert fit["params"]["p"] == pytest.approx(p_hat, rel=1e-7), transform
        assert fit["std_errors"]["p"] == pytest.approx(math.sqrt(p_hat * (1 - p_hat) / x.size), rel=1e-5)
    with pytest.raises(ValueError, match="strictly inside"):
        qf.mle_fit(log_density=lambda p, x: x * 0.0, data=x, form="observations", starts={"p": 0.95},
                   transforms={"p": (0.1, 0.9)}, optimiser="bfgs", errors="none", checks=None)


def test_mle_fit_refuses_what_it_cannot_read():
    good = {"log_density": lambda p, x: stats.norm.logpdf(x, p["mu"], 1.0), "data": NORMAL, "form": "observations",
            "starts": {"mu": 0.0}, "transforms": {"mu": "free"}, "optimiser": "bfgs", "errors": "none",
            "checks": None}
    for change, message in (({"log_density": lambda p, x: float(np.sum(x))}, "one log-density a term"),
                            ({"transforms": {}}, "naming every parameter"),
                            ({"transforms": {"mu": "bounded"}}, "must be one of"),
                            ({"form": "pairs"}, "form must be one of"),
                            ({"errors": "robust"}, "errors must be one of"),
                            ({"form": "transitions", "data": [1.0, 2.0]}, "at least three"),
                            ({"checks": [len]}, "checks must be None")):
        with pytest.raises(ValueError, match=message):
            qf.mle_fit(**{**good, **change})


def gamma_moments(p, x):
    return np.column_stack([x - p["shape"] * p["scale"], (x - x.mean()) ** 2 - p["shape"] * p["scale"] ** 2])


@pytest.mark.parametrize("weighting", ["identity", "two_step"])
def test_a_gamma_laws_moment_estimates_are_the_closed_forms(weighting):
    """Exactly identified: shape x-bar^2 / s^2 and scale s^2 / x-bar, s^2 the ddof-0 variance (Rice 8.4), whatever
    the weighting; Hansen's J is zero with no degree of freedom under 'two_step', nan under 'identity'."""
    x = RNG.gamma(2.5, 1.6, 600)
    fit = qf.moments_fit(moments=gamma_moments, data=x, starts={"shape": 1.0, "scale": 1.0},
                         transforms={"shape": "positive", "scale": "positive"}, weighting=weighting, lags=0,
                         optimiser="bfgs")
    assert fit["params"]["shape"] == pytest.approx(x.mean() ** 2 / x.var(), rel=1e-6)
    assert fit["params"]["scale"] == pytest.approx(x.var() / x.mean(), rel=1e-6)
    assert fit["j_dof"] == 0 and math.isnan(fit["j_pvalue"]) and fit["n_obs"] == x.size and fit["converged"]
    assert (fit["j_stat"] == pytest.approx(0.0, abs=1e-9)) if weighting == "two_step" else math.isnan(fit["j_stat"])


def test_ols_by_its_moments_is_ols_with_whites_and_newey_wests_errors():
    """g_i = x_i (y_i - x_i' b): exactly identified, b = qf.ols's; under 'identity' with lags 0 the errors are White's
    HC0, (X'X)^-1 sum e_i^2 x_i x_i' (X'X)^-1, and with lags L Newey-West's (qf.newey_west_ols, denominator 'T')."""
    n = 200
    z = RNG.normal(size=n)
    y = 0.3 + 1.7 * z + RNG.normal(size=n) * (0.5 + 0.5 * np.abs(z))
    X = np.column_stack([np.ones(n), z])

    def conditions(p, data):
        return data[:, 1:] * (data[:, 0] - data[:, 1:] @ np.array([p["a"], p["b"]]))[:, None]

    for lags in (0, 3):
        fit = qf.moments_fit(moments=conditions, data=np.column_stack([y, X]), starts={"a": 0.0, "b": 1.0},
                             transforms={"a": "free", "b": "free"}, weighting="identity", lags=lags, optimiser="bfgs")
        held = qf.ols(y=y, X=z, intercept=True)
        assert [fit["params"]["a"], fit["params"]["b"]] == pytest.approx(list(held["coef"]), rel=1e-7)
        hac = qf.newey_west_ols(y=y, X=z, intercept=True, lags=lags, denominator="T")
        assert [fit["std_errors"]["a"], fit["std_errors"]["b"]] == pytest.approx(list(hac["se"]), rel=1e-5), lags
    e = y - X @ held["coef"]
    bread = np.linalg.inv(X.T @ X)
    white = np.sqrt(np.diag(bread @ (X.T * e ** 2) @ X @ bread))
    assert list(hac["se"]) != pytest.approx(list(white), rel=1e-3)        # lags 3 differ from HC0
    fit0 = qf.moments_fit(moments=conditions, data=np.column_stack([y, X]), starts={"a": 0.0, "b": 1.0},
                          transforms={"a": "free", "b": "free"}, weighting="identity", lags=0, optimiser="bfgs")
    assert [fit0["std_errors"]["a"], fit0["std_errors"]["b"]] == pytest.approx(list(white), rel=1e-5)


def test_an_overidentified_poisson_fit_takes_the_efficient_weight_and_hansens_j():
    """lambda from E[x - lambda] = 0 and E[(x - lambda)^2 - lambda] = 0 (mean and variance both lambda): two-step
    errors (G'S^-1 G)^-1 / n at the estimate, J = n g-bar' W g-bar with one degree of freedom, its chi-squared
    p-value."""
    x = RNG.poisson(4.0, 500).astype(float)

    def conditions(p, data):
        return np.column_stack([data - p["lam"], (data - p["lam"]) ** 2 - p["lam"]])

    fit = qf.moments_fit(moments=conditions, data=x, starts={"lam": 3.0}, transforms={"lam": "positive"},
                         weighting="two_step", lags=0, optimiser="bfgs")
    lam = fit["params"]["lam"]
    g = conditions({"lam": lam}, x)
    s = g.T @ g / x.size
    G = np.array([[-1.0], [-2.0 * (x - lam).mean() - 1.0]])
    assert fit["std_errors"]["lam"] == pytest.approx(math.sqrt(1.0 / (G.T @ np.linalg.inv(s) @ G)[0, 0] / x.size),
                                                     rel=1e-4)
    assert fit["j_dof"] == 1 and fit["j_stat"] == pytest.approx(x.size * fit["objective"], rel=1e-12)
    assert fit["j_pvalue"] == pytest.approx(stats.chi2.sf(fit["j_stat"], 1), rel=1e-12)
    assert 0.0 < fit["j_pvalue"] < 1.0 and min(x.mean(), x.var()) - 0.05 < lam < max(x.mean(), x.var()) + 0.05
    from scipy import optimize        # the two steps by hand: W = I, then W = S^-1 at the first step's estimate

    def criterion(value, weight):
        mean = conditions({"lam": value}, x).mean(axis=0)
        return mean @ weight @ mean

    first = optimize.minimize_scalar(lambda v: criterion(v, np.eye(2)), bounds=(1.0, 8.0), method="bounded",
                                     options={"xatol": 1e-12}).x
    g1 = conditions({"lam": first}, x)
    weight = np.linalg.inv(g1.T @ g1 / x.size)
    second = optimize.minimize_scalar(lambda v: criterion(v, weight), bounds=(1.0, 8.0), method="bounded",
                                      options={"xatol": 1e-12}).x
    assert abs(second - first) > 1e-3 and lam == pytest.approx(second, rel=1e-7)
    assert fit["j_stat"] == pytest.approx(x.size * criterion(second, weight), rel=1e-6)
    with pytest.raises(ValueError, match="cannot identify"):
        qf.moments_fit(moments=lambda p, d: d - p["a"] - p["b"], data=x, starts={"a": 1.0, "b": 1.0},
                       transforms={"a": "free", "b": "free"}, weighting="identity", lags=0, optimiser="bfgs")


MISRA_Y = np.array([10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78])
MISRA_X = np.array([77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0])


@pytest.mark.parametrize("start", [{"b1": 500.0, "b2": 1e-4}, {"b1": 250.0, "b2": 5e-4}])
def test_misra1a_gives_nists_certified_values(start):
    """NIST StRD nonlinear regression, Misra1a (lower difficulty), y = b1 (1 - e^{-b2 x}): certified b1 =
    2.3894212918E+02 (sd 2.7070075241E+00), b2 = 5.5015643181E-04 (sd 7.2668688436E-06), residual sum of squares
    1.2455138894E-01, 12 degrees of freedom; from either of NIST's starts."""
    fit = qf.least_squares_fit(residuals=lambda p: MISRA_Y - p["b1"] * (1 - np.exp(-p["b2"] * MISRA_X)),
                               starts=start, bounds=None, weights=None, loss="linear", loss_scale=1.0)
    assert fit["params"]["b1"] == pytest.approx(2.3894212918e02, rel=1e-8)
    assert fit["params"]["b2"] == pytest.approx(5.5015643181e-04, rel=1e-8)
    assert fit["std_errors"]["b1"] == pytest.approx(2.7070075241, rel=1e-7)
    assert fit["std_errors"]["b2"] == pytest.approx(7.2668688436e-06, rel=1e-7)
    assert fit["sse"] == pytest.approx(1.2455138894e-01, rel=1e-9)
    assert fit["dof"] == 12 and fit["converged"] and fit["residuals"].shape == MISRA_Y.shape


def test_a_nelson_siegel_curve_by_least_squares_is_nelson_siegel_fits():
    """The same yields, the same model (qf.nelson_siegel_yield, the decay dividing T) and the same start: the minimum
    nelson_siegel_fit reaches (L-BFGS-B), or a sum of squares no larger, its decay bounded above 0."""
    t = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    yields = qf.nelson_siegel_yield(variant="nelson_siegel", maturities=t, params=[0.045, -0.015, 0.01, 2.0],
                                    decay_form="time") + RNG.normal(size=t.size) * 2e-4
    start = [0.04, -0.01, 0.0, 1.5]
    held = qf.nelson_siegel_fit(variant="nelson_siegel", maturities=t, yields=yields, initial=start, decay_form="time",
                                method="L-BFGS-B", options={"ftol": 0.0, "gtol": 1e-14, "maxiter": 20000}, fixed_decays=None)
    names = ("b0", "b1", "b2", "tau")
    fit = qf.least_squares_fit(
        residuals=lambda p: qf.nelson_siegel_yield(variant="nelson_siegel", maturities=t,
                                                   params=[p[name] for name in names], decay_form="time") - yields,
        starts=dict(zip(names, start)), bounds={"tau": (1e-6, np.inf)}, weights=None, loss="linear", loss_scale=1.0)
    assert [fit["params"][name] for name in names] == pytest.approx(list(held["params"]), rel=1e-4)
    assert held["sse"] * (1 - 1e-6) < fit["sse"] <= held["sse"] * (1 + 1e-12) and fit["dof"] == t.size - 4


def test_weights_give_weighted_least_squares_and_a_bound_holds():
    """A model linear in its parameters with weights w: the normal equations (X'WX) b = X'Wy and their errors
    s^2 (X'WX)^-1, s^2 = sum w e^2 / (n - k); a bound below the free solution holds the parameter at it."""
    t = np.linspace(0.0, 5.0, 30)
    y = 2.0 + 0.7 * t + RNG.normal(size=t.size) * 0.3
    w = 1.0 + t
    X = np.column_stack([np.ones_like(t), t])
    b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    e = y - X @ b
    cov = (w * e ** 2).sum() / (t.size - 2) * np.linalg.inv(X.T @ (w[:, None] * X))
    fit = qf.least_squares_fit(residuals=lambda p: p["a"] + p["b"] * t - y, starts={"a": 1.0, "b": 1.0},
                               bounds=None, weights=w, loss="linear", loss_scale=1.0)
    assert [fit["params"]["a"], fit["params"]["b"]] == pytest.approx(list(b), rel=1e-8)
    assert [fit["std_errors"]["a"], fit["std_errors"]["b"]] == pytest.approx(list(np.sqrt(np.diag(cov))), rel=1e-6)
    assert fit["sse"] == pytest.approx((w * e ** 2).sum(), rel=1e-10)
    assert fit["residuals"] == pytest.approx(X @ b - y, rel=1e-7, abs=1e-9)
    held = qf.least_squares_fit(residuals=lambda p: p["a"] + p["b"] * t - y, starts={"a": 1.0, "b": 0.5},
                                bounds={"b": (0.0, 0.6)}, weights=None, loss="linear", loss_scale=1.0)
    assert held["params"]["b"] == pytest.approx(0.6, abs=1e-8)


def test_a_robust_loss_discounts_an_outlier_that_moves_least_squares():
    t = np.linspace(0.0, 1.0, 21)
    y = 1.0 + 2.0 * t
    y[10] += 5.0
    fits = {loss: qf.least_squares_fit(residuals=lambda p: p["a"] + p["b"] * t - y, starts={"a": 0.5, "b": 1.0},
                                       bounds=None, weights=None, loss=loss, loss_scale=0.1)
            for loss in ("linear", "soft_l1", "huber", "cauchy")}
    assert abs(fits["linear"]["params"]["a"] - 1.0) > 0.2
    assert all(abs(fits[loss]["params"]["a"] - 1.0) < 0.1 for loss in ("soft_l1", "huber", "cauchy"))
    for change, message in (({"weights": [1.0, 2.0]}, "as long as the residuals"), ({"loss": "l2"}, "must be one of"),
                            ({"starts": {"a": 2.0, "b": 1.0}, "bounds": {"a": (0.0, 1.0)}}, "strictly inside")):
        with pytest.raises(ValueError, match=message):
            qf.least_squares_fit(**{"residuals": lambda p: p["a"] + p["b"] * t - y, "starts": {"a": 0.5, "b": 1.0},
                                    "bounds": None, "weights": None, "loss": "linear", "loss_scale": 1.0, **change})


@pytest.mark.parametrize("upper", [False, True])
def test_least_squares_derivatives_stay_in_the_residuals_domain(upper):
    """A square-root residual has its exact zero at a = target^2 (or 1 - target^2). Bound-aware differences
    must keep every residual evaluation in [0, 1], including the covariance at a near-bound optimum."""
    y = np.array([0.8e-4, 1e-4, 1.2e-4])
    target = y.mean()

    def residuals(p):
        assert 0.0 <= p["a"] <= 1.0
        return np.sqrt(1.0 - p["a"] if upper else p["a"]) - y

    fit = qf.least_squares_fit(residuals=residuals, starts={"a": 0.5}, bounds={"a": (0.0, 1.0)},
                               weights=None, loss="linear", loss_scale=1.0)
    assert fit["params"]["a"] == pytest.approx(1.0 - target ** 2 if upper else target ** 2, abs=1e-11)
    assert fit["converged"] and np.all(np.isfinite(fit["covariance"]))
    expected_error = 2.0 * target * y.std(ddof=1) / np.sqrt(y.size)
    assert fit["std_errors"]["a"] == pytest.approx(expected_error, rel=1e-3)
