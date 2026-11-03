"""qf.short_rate_bond_price and qf.cir_fit (lane libgaps1; docs/research/library-syllabus-1011.md row 44 and the
holdout read's CIR gap): Vasicek's and Cox, Ingersoll and Ross's zero-coupon bonds and bond options in closed form,
and the CIR fit by exact maximum likelihood. References: QuantLib 1.43's Vasicek and CoxIngersollRoss models (frozen
in `QL`; `__main__` recomputes them), closed-form identities (both models' sigma -> 0 limit; Vasicek's option is
Hull-White's on Vasicek's own discount factors; parity), the noncentral chi-square density's normalisation, and a
simulated path's parameters recovered. Every input is this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

import qf

pytestmark = pytest.mark.fast

QL = [   # QuantLib 1.43: model.discountBond(0, tau, r0) and model.discountBondOption(type, K, T, S) (`quantlib_values`)
    (("cir", "none", 0.03, 0.3, 0.04, 0.08, 1.0, None, None), 0.9691515360558345),
    (("cir", "none", 0.03, 0.3, 0.04, 0.08, 10.0, None, None), 0.69642509488653),
    (("cir", "call", 0.03, 0.3, 0.04, 0.08, 5.0, 1.0, 0.85), 0.02134442579054463),
    (("cir", "put", 0.03, 0.3, 0.04, 0.08, 5.0, 1.0, 0.85), 0.0035353485126115025),
    (("cir", "call", 0.05, 0.8, 0.03, 0.15, 3.0, 2.0, 0.96), 0.010343909177482513),
    (("cir", "put", 0.05, 0.8, 0.03, 0.15, 3.0, 2.0, 0.96), 0.002747342479341386),
    (("vasicek", "none", 0.03, 0.3, 0.04, 0.08, 5.0, None, None), 0.8832344737623052),
    (("vasicek", "none", 0.05, 0.8, 0.03, 0.15, 10.0, None, None), 0.8334724697460046),
    (("vasicek", "call", 0.03, 0.3, 0.04, 0.08, 5.0, 1.0, 0.85), 0.08927558945812442),
    (("vasicek", "put", 0.05, 0.8, 0.03, 0.15, 3.0, 2.0, 0.96), 0.02031573575753709),
]


def bond(model, option, r, kappa, theta, sigma, maturity, expiry=None, strike=None):
    return qf.short_rate_bond_price(model=model, option=option, short_rate=r, maturity=maturity, kappa=kappa,
                                    theta=theta, sigma=sigma, expiry=expiry, strike=strike)


def test_bonds_and_bond_options_equal_quantlibs_models():
    for (model, option, r, kappa, theta, sigma, maturity, expiry, strike), expected in QL:
        found = bond(model, option, r, kappa, theta, sigma, maturity, expiry, strike)["price"]
        assert found == pytest.approx(expected, abs=1e-12), (model, option, maturity)


def test_both_models_tend_to_the_deterministic_rate_as_the_vol_vanishes():
    """With sigma -> 0 the short rate follows r(s) = theta + (r - theta) e^{-kappa s}, so P = exp(-theta tau - (r -
    theta)(1 - e^{-kappa tau})/kappa): CIR's and Vasicek's closed forms both meet it (Brigo and Mercurio 3.2)."""
    r, kappa, theta, tau = 0.02, 0.4, 0.05, 7.0
    limit = math.exp(-theta * tau - (r - theta) * (1 - math.exp(-kappa * tau)) / kappa)
    for model in ("cir", "vasicek"):
        found = bond(model, "none", r, kappa, theta, 1e-7, tau)
        assert found["price"] == pytest.approx(limit, rel=1e-10) and found["zero_rate"] == pytest.approx(
            -math.log(limit) / tau, rel=1e-9)
        assert found["b"] == pytest.approx((1 - math.exp(-kappa * tau)) / kappa, rel=1e-9)


def test_vasicek_options_are_hull_whites_on_vasicek_discount_factors():
    """Hull-White's bond option depends on the curve only through P(0, T) and P(0, S) (Jamshidian 1989), so on
    Vasicek's own factors it is Vasicek's option exactly (qf.hull_white_zcb_option)."""
    r, kappa, theta, sigma, t, s = 0.035, 0.5, 0.045, 0.012, 1.5, 4.0
    p_t, p_s = (bond("vasicek", "none", r, kappa, theta, sigma, x)["price"] for x in (t, s))
    for option, k in (("call", 0.90), ("put", 0.93)):
        found = bond("vasicek", option, r, kappa, theta, sigma, s, t, k)
        hw = qf.hull_white_zcb_option(option, t, s, k, kappa, sigma, p_t, p_s)
        assert found["price"] == pytest.approx(hw, abs=1e-14) and found["bond"] == p_s and found["discount"] == p_t


def test_bond_options_keep_parity_and_a_strike_past_cirs_ceiling_is_worthless():
    """call - put = P(0, S) - K P(0, T) in both models; under CIR P(T, S) <= A(S - T) (rates stay at or above 0), so
    a call struck above A(S - T) is worth nothing and its put is the forward's intrinsic value."""
    for model in ("cir", "vasicek"):
        call, put = (bond(model, o, 0.04, 0.6, 0.05, 0.1, 6.0, 2.0, 0.85) for o in ("call", "put"))
        assert call["price"] - put["price"] == pytest.approx(call["bond"] - 0.85 * call["discount"], abs=1e-12)
    ceiling = qf.short_rate_bond_price(model="cir", option="none", short_rate=0.0, maturity=4.0, kappa=0.6, theta=0.05,
                                       sigma=0.1, expiry=None, strike=None)["a"]
    above = bond("cir", "call", 0.04, 0.6, 0.05, 0.1, 6.0, 2.0, ceiling * 1.01)
    assert above["price"] == 0.0


def cir_path(n, dt, kappa, theta, sigma, r0, seed):
    """The exact CIR transition (Cox, Ingersoll and Ross 1985): r_{t+dt} = Y / (2c), Y noncentral chi-square."""
    rng, c = np.random.default_rng(seed), 2 * kappa / (sigma ** 2 * (1 - math.exp(-kappa * dt)))
    df, path = 4 * kappa * theta / sigma ** 2, [r0]
    for _ in range(n):
        path.append(rng.noncentral_chisquare(df, 2 * c * path[-1] * math.exp(-kappa * dt)) / (2 * c))
    return np.array(path)


def test_the_transition_density_integrates_to_one():
    kappa, theta, sigma, dt, now = 0.7, 0.04, 0.1, 1 / 52, 0.03
    c = 2 * kappa / (sigma * sigma * -math.expm1(-kappa * dt))

    def density(x):
        return 2 * c * stats.ncx2.pdf(2 * c * x, 4 * kappa * theta / sigma ** 2, 2 * c * now * math.exp(-kappa * dt))
    assert integrate.quad(density, 0, 1, points=[now], limit=200)[0] == pytest.approx(1.0, abs=1e-9)


def test_the_fit_recovers_an_exactly_simulated_path():
    """Three thousand weekly rates from known parameters (exact transitions): the fit lands near them, its likelihood
    at least the true parameters', and the Feller ratio is 2 kappa theta / sigma^2."""
    kappa, theta, sigma, dt = 0.8, 0.045, 0.09, 1 / 52
    rates = cir_path(3000, dt, kappa, theta, sigma, 0.03, 31)
    fit = qf.cir_fit(rates=rates, dt=dt, method="mle_exact", bounds={"kappa": (0.05, 4.0), "theta": (0.005, 0.15), "sigma": (0.01, 0.4)},
                     starts=3, rng=2)
    assert fit["kappa"] == pytest.approx(kappa, rel=0.35) and fit["theta"] == pytest.approx(theta, rel=0.15)
    assert fit["sigma"] == pytest.approx(sigma, rel=0.05) and fit["converged"]
    c = 2 * kappa / (sigma ** 2 * -math.expm1(-kappa * dt))
    truth = np.sum(np.log(2 * c) + stats.ncx2.logpdf(2 * c * rates[1:], 4 * kappa * theta / sigma ** 2,
                                                       2 * c * rates[:-1] * math.exp(-kappa * dt)))
    assert fit["log_likelihood"] >= truth - 1e-6
    assert fit["feller_ratio"] == pytest.approx(2 * fit["kappa"] * fit["theta"] / fit["sigma"] ** 2)


def test_the_fit_on_a_short_path_and_its_refusals():
    rates = cir_path(400, 1 / 52, 1.0, 0.05, 0.1, 0.05, 3)
    fit = qf.cir_fit(rates=rates, dt=1 / 52, method="mle_exact", bounds={"kappa": (0.1, 3.0), "theta": (0.01, 0.1), "sigma": (0.02, 0.3)},
                     starts=2, rng=1)
    assert 0.02 < fit["sigma"] < 0.3 and np.isfinite(fit["log_likelihood"])
    with pytest.raises(ValueError, match="positive"):
        qf.cir_fit(rates=[0.03, 0.0, 0.02, 0.01], dt=1 / 52, method="mle_exact", bounds={"kappa": (0.1, 1), "theta": (0.01, 0.1),
                                                               "sigma": (0.02, 0.3)}, starts=1, rng=1)
    with pytest.raises(ValueError, match="bounds"):
        qf.cir_fit(rates=rates, dt=1 / 52, method="mle_exact", bounds={"kappa": (0.1, 1)}, starts=1, rng=1)


def test_the_fit_keeps_the_best_start(monkeypatch):
    """Each start's optimum is kept only where its likelihood beats the best so far (worse, better, worse: the
    second)."""
    import types
    found = iter([5.0, 3.0, 4.0])

    def fake(loss, start, method, bounds):
        value = next(found)
        return types.SimpleNamespace(x=np.array([0.5, 0.04, 0.05 + value / 100]), fun=value, success=True)
    monkeypatch.setattr(qf.shortrate.optimize, "minimize", fake)
    fit = qf.cir_fit(rates=cir_path(50, 1 / 52, 1.0, 0.05, 0.1, 0.05, 3), dt=1 / 52, method="mle_exact",
                     bounds={"kappa": (0.1, 3.0), "theta": (0.01, 0.1), "sigma": (0.02, 0.3)}, starts=3, rng=1)
    assert fit["log_likelihood"] == -3.0 and fit["sigma"] == pytest.approx(0.08)


@pytest.mark.parametrize("n, slack", [(5_000, 2.0), pytest.param(20_000, 1.0, marks=pytest.mark.slow)])
def test_the_euler_regression_recovers_a_fine_path_and_carries_its_step_bias_on_a_coarse_one(n, slack):
    """Brigo and Mercurio (2006) 3.2.3's Euler scheme as a regression of (r' - r)/sqrt(r) on 1/sqrt(r) and sqrt(r):
    on exactly simulated daily rates it lands near the true parameters and near the exact MLE; on half-yearly rates
    its kappa is the exact transition's (1 - e^{-kappa dt})/dt, not kappa (E[r' - r | r] = (theta - r)(1 -
    e^{-kappa dt})), while theta = -b1/b2 stays unbiased: the bias falls with the step."""
    kappa, theta, sigma = 0.9, 0.04, 0.08
    fine = cir_path(n, 1 / 252, kappa, theta, sigma, 0.03, 41)
    euler = qf.cir_fit(rates=fine, dt=1 / 252, method="ols_euler", bounds=None, starts=None, rng=None)
    exact = qf.cir_fit(rates=fine, dt=1 / 252, method="mle_exact",
                       bounds={"kappa": (0.05, 4.0), "theta": (0.005, 0.15), "sigma": (0.01, 0.4)}, starts=2, rng=3)
    assert euler["sigma"] == pytest.approx(sigma, rel=0.02 * slack) and euler["theta"] == pytest.approx(
        theta, rel=0.15 * slack)
    assert euler["kappa"] == pytest.approx(exact["kappa"], rel=0.05 * slack) and euler["theta"] == pytest.approx(
        exact["theta"], rel=0.02 * slack)
    assert euler["log_likelihood"] <= exact["log_likelihood"] + 1e-6
    coarse = cir_path(n, 0.5, kappa, theta, sigma, 0.03, 43)
    found = qf.cir_fit(rates=coarse, dt=0.5, method="ols_euler", bounds=None, starts=None, rng=None)
    assert found["kappa"] == pytest.approx(-np.expm1(-kappa * 0.5) / 0.5, rel=0.05 * slack)
    assert found["kappa"] < 0.85 * kappa and found["theta"] == pytest.approx(theta, rel=0.05 * slack)
    with pytest.raises(ValueError, match="None"):
        qf.cir_fit(rates=coarse, dt=0.5, method="ols_euler", bounds={"kappa": (0.1, 1.0)}, starts=None, rng=None)


def test_the_euler_regression_is_its_least_squares_by_hand():
    """The definition on a short series: slopes by the normal equations of y = (r' - r)/sqrt(r) on 1/sqrt(r) and
    sqrt(r), kappa = -b2/dt, theta = -b1/b2, sigma^2 = SSE/(n - 2)/dt."""
    r, dt = np.array([0.030, 0.0312, 0.0305, 0.0298, 0.0310, 0.0321, 0.0309, 0.0302]), 1 / 52
    x = np.column_stack([1 / np.sqrt(r[:-1]), np.sqrt(r[:-1])])
    y = np.diff(r) / np.sqrt(r[:-1])
    b = np.linalg.solve(x.T @ x, x.T @ y)
    e = y - x @ b
    fit = qf.cir_fit(rates=r, dt=dt, method="ols_euler", bounds=None, starts=None, rng=None)
    assert fit["kappa"] == pytest.approx(-b[1] / dt, rel=1e-10) and fit["theta"] == pytest.approx(-b[0] / b[1],
                                                                                                   rel=1e-10)
    assert fit["sigma"] == pytest.approx(np.sqrt(e @ e / (y.size - 2) / dt), rel=1e-10)


def quantlib_values() -> list:
    """The frozen `QL` values recomputed by QuantLib (scratch: a Python with QuantLib 1.43)."""
    import QuantLib as ql
    found = []
    for args, _ in QL:
        model, option, r, kappa, theta, sigma, maturity, expiry, strike = args
        made = ql.CoxIngersollRoss(r, theta, kappa, sigma) if model == "cir" else ql.Vasicek(r, kappa, theta, sigma)
        if option == "none":
            found.append((args, made.discountBond(0.0, maturity, r)))
        else:
            kind = ql.Option.Call if option == "call" else ql.Option.Put
            found.append((args, made.discountBondOption(kind, strike, expiry, maturity)))
    return found


if __name__ == "__main__":
    print("QL frozen values match QuantLib:", np.allclose([v for _, v in quantlib_values()], [v for _, v in QL],
                                                          rtol=1e-14, atol=0))
