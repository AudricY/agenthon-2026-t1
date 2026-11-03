"""qf.merton_jump_price and qf.merton_jump_fit (lane libgaps1; docs/research/library-syllabus-1011.md row 111): the
Merton (1976) jump diffusion's Poisson-weighted sum of Black-Scholes prices, and the maximum-likelihood fit of a
return series to its Poisson mixture of normals. References: zero intensity is Black-Scholes; the series equals the
Gil-Pelaez inversion (qf.charfn_price) of Merton's characteristic function, an independent route; put-call parity;
the density integrates to one; a simulated series' parameters are recovered. Every input is this file's own; no
unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import integrate

import qf
from qf import jumps

pytestmark = pytest.mark.fast


def price(option, s, k, t, r, q, v, lam, mu, delta, terms=60):
    return qf.merton_jump_price(option=option, spot=s, strike=k, maturity=t, rate=r, dividend_yield=q, vol=v,
                                intensity=lam, jump_mean=mu, jump_vol=delta, terms=terms)


def merton_charfn(s, t, r, q, v, lam, mu, delta):
    """Merton's characteristic function of ln S_T under the pricing measure (Merton 1976; Cont and Tankov 2004,
    4.4.1): the diffusion's with the drift compensated by lambda k, times the compound Poisson's."""
    k = math.exp(mu + 0.5 * delta * delta) - 1.0

    def phi(u):
        drift = math.log(s) + (r - q - lam * k - 0.5 * v * v) * t
        return np.exp(1j * u * drift - 0.5 * v * v * u * u * t + lam * t * (np.exp(1j * u * mu - 0.5 * delta ** 2
                                                                                    * u * u) - 1.0))
    return phi


def test_zero_intensity_or_zero_jumps_is_black_scholes():
    for option in ("call", "put"):
        bs = qf.bs_price(option, 100.0, 95.0, 0.7, 0.04, 0.01, 0.22)
        assert price(option, 100.0, 95.0, 0.7, 0.04, 0.01, 0.22, 0.0, -0.1, 0.2) == pytest.approx(bs, abs=1e-14)
        assert price(option, 100.0, 95.0, 0.7, 0.04, 0.01, 0.22, 3.0, 0.0, 0.0) == pytest.approx(bs, abs=1e-12)


def test_the_series_equals_the_characteristic_function_inversion():
    """Two routes to one price: Merton's sum of Black-Scholes prices conditional on n jumps, weighted by lambda' T's
    Poisson law, and the Gil-Pelaez inversion of the model's characteristic function. Negative and positive mean
    jumps, across strikes; a lambda T-weighted sum with r_n would miss by the k != 0 terms."""
    for lam, mu, delta in ((1.5, -0.08, 0.12), (4.0, 0.03, 0.06)):
        phi = merton_charfn(100.0, 0.75, 0.03, 0.01, 0.18, lam, mu, delta)
        for k in (80.0, 100.0, 125.0):
            gil = qf.charfn_price(charfn=phi, option="call", spot=100.0, strike=k, maturity=0.75, rate=0.03,
                                  dividend_yield=0.01, quadrature="quad", upper=400.0, nodes=400)["price"]
            assert price("call", 100.0, k, 0.75, 0.03, 0.01, 0.18, lam, mu, delta) == pytest.approx(gil, abs=1e-7)
    lam, mu, delta, t = 1.5, -0.08, 0.12, 0.75
    k_bar = math.exp(mu + 0.5 * delta * delta) - 1.0
    wrong = sum(math.exp(-lam * t) * (lam * t) ** n / math.factorial(n) * qf.bs_price(
        "call", 100.0, 100.0, t, 0.03 - lam * k_bar + n * math.log1p(k_bar) / t, 0.01,
        math.sqrt(0.18 ** 2 + n * delta * delta / t)) for n in range(60))
    assert abs(wrong - price("call", 100.0, 100.0, t, 0.03, 0.01, 0.18, lam, mu, delta)) > 1e-3


def test_put_call_parity_holds_and_jumps_raise_the_wings():
    """c - p = S e^{-qT} - K e^{-rT} under any jump law; with mean-zero jumps the out-of-the-money options are worth
    more than Black-Scholes at the same diffusion vol (fat tails)."""
    for k in (70.0, 100.0, 140.0):
        call, put = (price(o, 100.0, k, 1.2, 0.05, 0.02, 0.2, 2.0, -0.05, 0.1) for o in ("call", "put"))
        assert call - put == pytest.approx(100.0 * math.exp(-0.02 * 1.2) - k * math.exp(-0.05 * 1.2), abs=1e-10)
    assert price("put", 100.0, 70.0, 0.5, 0.03, 0.0, 0.2, 2.0, 0.0, 0.15) > qf.bs_price("put", 100.0, 70.0, 0.5, 0.03,
                                                                                           0.0, 0.2)


def test_the_return_density_is_a_density():
    for lam, mu, delta in ((0.0, 0.0, 0.1), (30.0, -0.02, 0.05)):
        total = integrate.quad(lambda x: math.exp(jumps._log_density(np.array([x]), 1 / 252, 0.05, 0.2, lam, mu,
                                                                     delta, 40)[0]), -1.0, 1.0, limit=400)[0]
        assert total == pytest.approx(1.0, abs=1e-9)


BOUNDS = {"vol": (0.02, 0.6), "intensity": (0.0, 60.0), "jump_mean": (-0.15, 0.15), "jump_vol": (0.005, 0.2)}


def simulate(n, dt, drift, vol, lam, mu, delta, seed):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam * dt, n)
    jumps_ = rng.normal(mu * counts, delta * np.sqrt(counts))
    return drift * dt + vol * math.sqrt(dt) * rng.standard_normal(n) + jumps_


@pytest.mark.slow
def test_the_fit_recovers_a_simulated_series_and_beats_the_true_parameters_likelihood():
    """Ten thousand daily returns from known parameters (Ball and Torous 1985's likelihood): the fit lands near
    them, and its likelihood is at least the true parameters' (it is the maximum)."""
    dt, true = 1 / 252, dict(drift=0.06, vol=0.15, lam=25.0, mu=-0.02, delta=0.03)
    x = simulate(10_000, dt, true["drift"], true["vol"], true["lam"], true["mu"], true["delta"], 17)
    fit = qf.merton_jump_fit(returns=x, dt=dt, terms=8, drift="fit", bounds={**BOUNDS, "drift": (-1.0, 1.0)},
                             starts=3, rng=5)
    assert fit["vol"] == pytest.approx(true["vol"], rel=0.1) and fit["intensity"] == pytest.approx(true["lam"],
                                                                                                    rel=0.35)
    assert fit["jump_mean"] == pytest.approx(true["mu"], abs=0.01) and fit["jump_vol"] == pytest.approx(true["delta"],
                                                                                                         abs=0.01)
    at_truth = float(jumps._log_density(x, dt, true["drift"], true["vol"], true["lam"], true["mu"], true["delta"],
                                        8).sum())
    assert fit["log_likelihood"] >= at_truth - 1e-6 and fit["converged"]
    assert fit["total_vol"] == pytest.approx(math.sqrt(fit["vol"] ** 2 + fit["intensity"] * (
        fit["jump_mean"] ** 2 + fit["jump_vol"] ** 2)), abs=1e-15)


def test_without_jumps_the_fit_is_the_normal_mle_and_a_fixed_mu_compensates_the_drift():
    """Intensity held at 0: the likelihood is the normal's, maximised by vol^2 = var(x)/dt (ddof 0) and m dt =
    mean(x). With drift a number mu, m = mu - vol^2/2 - lambda k is reported (Merton's compensated drift)."""
    dt = 1 / 252
    x = simulate(1500, dt, 0.08, 0.2, 0.0, 0.0, 0.01, 23)
    bounds = {**BOUNDS, "intensity": (0.0, 1e-12), "drift": (-2.0, 2.0)}
    fit = qf.merton_jump_fit(returns=x, dt=dt, terms=2, drift="fit", bounds=bounds, starts=2, rng=1)
    assert fit["vol"] == pytest.approx(x.std() / math.sqrt(dt), rel=1e-4)
    assert fit["drift"] * dt == pytest.approx(x.mean(), abs=1e-6)
    normal = -0.5 * x.size * (math.log(2 * math.pi * x.var()) + 1.0)
    assert fit["log_likelihood"] == pytest.approx(normal, abs=1e-4)
    fixed = qf.merton_jump_fit(returns=x, dt=dt, terms=4, drift=0.1, bounds=BOUNDS, starts=2, rng=2)
    k = math.expm1(fixed["jump_mean"] + 0.5 * fixed["jump_vol"] ** 2)
    assert fixed["drift"] == pytest.approx(0.1 - 0.5 * fixed["vol"] ** 2 - fixed["intensity"] * k, abs=1e-12)
    # total_vol^2 dt is the fitted density's own variance, vol^2 dt + lambda dt (jump_mean^2 + jump_vol^2)
    grid = np.linspace(-0.3, 0.3, 60_001)
    density = np.exp(jumps._log_density(grid, dt, fixed["drift"], fixed["vol"], fixed["intensity"],
                                        fixed["jump_mean"], fixed["jump_vol"], 40))
    mean = np.trapezoid(grid * density, grid)
    assert fixed["total_vol"] ** 2 * dt == pytest.approx(np.trapezoid(grid ** 2 * density, grid) - mean ** 2, rel=1e-9)


def test_the_fit_refuses_bounds_that_do_not_name_its_parameters():
    x = simulate(100, 1 / 252, 0.0, 0.2, 0.0, 0.0, 0.01, 1)
    with pytest.raises(ValueError, match="bounds"):
        qf.merton_jump_fit(returns=x, dt=1 / 252, terms=4, drift="fit", bounds=BOUNDS, starts=1, rng=1)
    with pytest.raises(ValueError, match="drift"):
        qf.merton_jump_fit(returns=x, dt=1 / 252, terms=4, drift="free", bounds=BOUNDS, starts=1, rng=1)


def test_the_fit_keeps_the_best_start(monkeypatch):
    """Each start's optimum is kept only where its likelihood beats the best so far: three starts whose optima are
    worse, better, worse give the second."""
    import types
    found = iter([5.0, 3.0, 4.0])

    def fake(loss, start, method, bounds):
        value = next(found)
        return types.SimpleNamespace(x=np.array([0.1 + value / 100, 2.0, -0.01, 0.02, 0.0]), fun=value, success=True)
    monkeypatch.setattr(jumps.optimize, "minimize", fake)
    x = simulate(100, 1 / 252, 0.0, 0.2, 0.0, 0.0, 0.01, 1)
    fit = qf.merton_jump_fit(returns=x, dt=1 / 252, terms=2, drift="fit", bounds={**BOUNDS, "drift": (-1.0, 1.0)},
                             starts=3, rng=1)
    assert fit["log_likelihood"] == -3.0 and fit["vol"] == pytest.approx(0.13)
