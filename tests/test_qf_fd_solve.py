"""qf.fd_solve, the finite-difference solver of any one-factor pricing PDE (lane lib-primitives1; the owner's in633).
References: Black-Scholes coefficients against qf.bs_fd_price on the same grid (European and American) and against
qf.bs_price; the Vasicek bond PDE against qf.short_rate_bond_price; the heat equation's exact solution e^{-pi^2 T}
sin(pi x), with Dirichlet ends, and e^{-pi^2 T/4} cos(pi x/2) with a zero slope, converging at second order under
Crank-Nicolson; a source term's exact integral; the CEV PDE against Schroder's formula through qf.noncentral_chi2
(Hull 27.1). Every input is this file's own or a textbook's; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast


def solve(**change):
    base = {"diffusion": lambda x, t: 0.5 * 0.2 ** 2 * x ** 2, "drift": lambda x, t: 0.04 * x,
            "rate": lambda x, t: 0.05, "source": None, "payoff": lambda x: np.maximum(x - 100.0, 0.0),
            "grid": np.linspace(0.0, 400.0, 201), "maturity": 1.0, "time_steps": 100, "scheme": "crank_nicolson",
            "rannacher_steps": 2, "lower": "linear", "upper": "linear", "exercise": "european",
            "exercise_value": None, "at": 100.0}
    return qf.fd_solve(**{**base, **change})


@pytest.mark.parametrize("scheme, exercise", [("crank_nicolson", "european"), ("implicit", "european"),
                                              ("crank_nicolson", "american_projection")])
def test_black_scholes_coefficients_are_bs_fd_price_on_the_same_grid(scheme, exercise):
    """a = vol^2 S^2/2, b = (r - q) S, rate r on nodes uniform in S, both ends linear: bs_fd_price's march, so its
    price and delta; American by projection on each step."""
    put = exercise != "european"
    payoff = (lambda x: np.maximum(100.0 - x, 0.0)) if put else (lambda x: np.maximum(x - 100.0, 0.0))
    found = solve(diffusion=lambda x, t: 0.5 * 0.3 ** 2 * x ** 2, drift=lambda x, t: (0.05 - 0.01) * x,
                  payoff=payoff, grid=np.linspace(0.0, 400.0, 401), time_steps=150, scheme=scheme,
                  exercise="american" if put else "european",
                  exercise_value=(lambda x, t: np.maximum(100.0 - x, 0.0)) if put else None)
    held = qf.bs_fd_price(option="put" if put else "call", exercise=exercise, spot=100.0, strike=100.0,
                          maturity=1.0, rate=0.05, dividend_yield=0.01, vol=0.3, grid="price", lower=0.0, upper=400.0,
                          space_steps=400, time_steps=150, scheme=scheme, boundary="linear", rannacher_steps=2)
    assert found["value"] == pytest.approx(held["price"], rel=1e-12)
    assert found["delta"] == pytest.approx(held["delta"], rel=1e-10)
    assert found["values"] == pytest.approx(held["values"], rel=1e-10, abs=1e-12)


def test_black_scholes_coefficients_on_a_grid_dense_at_the_strike_give_bs_price():
    nodes = np.r_[0.0, 100.0 * np.exp(np.linspace(math.log(0.05), math.log(5.0), 401))]
    found = solve(grid=nodes, time_steps=200, lower=("value", 0.0))
    exact = qf.bs_price(option="call", spot=100.0, strike=100.0, maturity=1.0, rate=0.05, dividend_yield=0.01,
                        vol=0.2)
    assert found["value"] == pytest.approx(exact, abs=3e-3)
    d1 = (0.05 - 0.01 + 0.5 * 0.2 ** 2) / 0.2
    assert found["delta"] == pytest.approx(math.exp(-0.01) * 0.5 * math.erfc(-d1 / math.sqrt(2)), abs=2e-3)


def test_the_vasicek_bond_pde_is_the_closed_form_bond():
    """x the short rate: a = sigma^2/2, b = kappa (theta - x), rate x, payoff 1."""
    kappa, theta, sigma, r0, t = 0.3, 0.05, 0.02, 0.03, 5.0
    found = solve(diffusion=lambda x, s: 0.5 * sigma ** 2, drift=lambda x, s: kappa * (theta - x),
                  rate=lambda x, s: x, payoff=lambda x: np.ones_like(x), grid=np.linspace(-0.25, 0.35, 121),
                  maturity=t, time_steps=100, rannacher_steps=0, at=r0)
    held = qf.short_rate_bond_price(model="vasicek", option="none", short_rate=r0, maturity=t, kappa=kappa,
                                    theta=theta, sigma=sigma, expiry=None, strike=None)
    assert found["value"] == pytest.approx(held["price"], rel=1e-6)
    assert found["delta"] == pytest.approx(-held["b"] * held["price"], rel=1e-4)


def heat(cells, steps, scheme="crank_nicolson", lower=("value", 0.0), payoff=lambda x: np.sin(np.pi * x)):
    return solve(diffusion=lambda x, t: 1.0, drift=lambda x, t: 0.0, rate=lambda x, t: 0.0, payoff=payoff,
                 grid=np.linspace(0.0, 1.0, cells + 1), maturity=0.1, time_steps=steps, scheme=scheme,
                 rannacher_steps=0, lower=lower, upper=("value", 0.0), at=0.3)


def test_the_heat_equation_converges_to_its_exact_solution_at_second_order():
    """V_t + V_xx = 0, V(x, T) = sin(pi x), V = 0 at both ends: V(x, 0) = e^{-pi^2 T} sin(pi x); halving the steps
    of x and t quarters the error under Crank-Nicolson."""
    exact = math.exp(-math.pi ** 2 * 0.1) * math.sin(math.pi * 0.3)
    errors = [abs(heat(cells, cells)["value"] - exact) for cells in (40, 80)]
    assert errors[0] < 2e-4 and errors[1] == pytest.approx(errors[0] / 4, rel=0.05)
    explicit = heat(50, 600, scheme="explicit")
    assert explicit["value"] == pytest.approx(exact, abs=3e-4)
    with pytest.raises(ValueError, match="explicit scheme is unstable"):
        heat(50, 100, scheme="explicit")


def test_a_slope_boundary_and_a_source_term():
    """V_x = 0 at 0 and V = 0 at 1 with V(x, T) = cos(pi x / 2): V(x, 0) = e^{-pi^2 T/4} cos(pi x/2). V_t + f = 0
    with f = t: V(x, 0) = payoff + T^2 / 2 (the source a function of time, exact under the theta scheme)."""
    found = heat(200, 200, lower=("slope", 0.0), payoff=lambda x: np.cos(np.pi * x / 2))
    assert found["value"] == pytest.approx(math.exp(-math.pi ** 2 / 40) * math.cos(0.15 * math.pi), abs=5e-6)
    sourced = solve(diffusion=lambda x, t: 0.0, drift=lambda x, t: 0.0, rate=lambda x, t: 0.0,
                    source=lambda x, t: t + 0.0 * x, payoff=lambda x: x, grid=np.linspace(0.0, 1.0, 11),
                    maturity=2.0, time_steps=10, rannacher_steps=0, at=0.5)
    assert sourced["value"] == pytest.approx(0.5 + 2.0, rel=1e-14)
    sloped = solve(diffusion=lambda x, t: 0.0, drift=lambda x, t: 0.0, rate=lambda x, t: 0.0, payoff=lambda x: 3 * x,
                   grid=np.linspace(0.0, 1.0, 11), lower=("slope", lambda t: 3.0), upper=("slope", 3.0), at=0.55)
    assert sloped["value"] == pytest.approx(1.65, rel=1e-12) and sloped["delta"] == pytest.approx(3.0, rel=1e-10)


def test_a_boundary_value_moving_in_time_is_taken_at_each_step():
    """V = x^2 + 2 (T - t) solves V_t + V_xx = 0; with its own values at both ends, moving in time, the theta scheme
    is exact on a quadratic in x and linear in t."""
    found = solve(diffusion=lambda x, t: 1.0, drift=lambda x, t: 0.0, rate=lambda x, t: 0.0, payoff=lambda x: x * x,
                  grid=np.linspace(0.0, 1.0, 21), maturity=0.5, time_steps=10, rannacher_steps=0,
                  lower=("value", lambda t: 2.0 * (0.5 - t)), upper=("value", lambda t: 1.0 + 2.0 * (0.5 - t)), at=0.3)
    assert found["value"] == pytest.approx(0.09 + 1.0, rel=1e-12)
    assert found["delta"] == pytest.approx(0.6, rel=1e-10) and found["gamma"] == pytest.approx(2.0, rel=1e-10)


def test_a_slope_at_either_end_and_the_linear_end_on_uneven_nodes_are_exact_on_quadratics():
    """V = x^2 + 2 (T - t) has V_x = 2 at 1; V = (x - 1)^2 + 2 (T - t) has V_x = -2 at 0: each given as a slope, the
    other end its moving value, the one-sided differences exact on a quadratic. A linear V (diffusion leaves it
    unchanged) on uneven nodes with both ends 'linear' stays itself near an end."""
    common = {"diffusion": lambda x, t: 1.0, "drift": lambda x, t: 0.0, "rate": lambda x, t: 0.0,
              "grid": np.linspace(0.0, 1.0, 21), "maturity": 0.5, "time_steps": 10, "rannacher_steps": 0, "at": 0.3}
    upper = solve(**common, payoff=lambda x: x * x, lower=("value", lambda t: 2.0 * (0.5 - t)), upper=("slope", 2.0))
    assert upper["value"] == pytest.approx(0.09 + 1.0, rel=1e-11)
    lower = solve(**common, payoff=lambda x: (x - 1.0) ** 2, lower=("slope", lambda t: -2.0),
                  upper=("value", lambda t: 2.0 * (0.5 - t)))
    assert lower["value"] == pytest.approx(0.49 + 1.0, rel=1e-11)
    uneven = solve(**{**common, "grid": np.linspace(0.0, 1.0, 11) ** 2, "at": 0.02}, payoff=lambda x: 2.0 * x + 1.0,
                   lower="linear", upper="linear")
    assert uneven["value"] == pytest.approx(1.04, rel=1e-12) and uneven["delta"] == pytest.approx(2.0, rel=1e-10)


def test_the_implicit_scheme_takes_the_source_at_the_earlier_date():
    """V_t + f = 0, f = t: the implicit step adds dt f(t_n) at the date it solves for, so V(0) = payoff + dt^2 N (N -
    1)/2, the left sum; Crank-Nicolson's average is the exact T^2/2."""
    sourced = solve(diffusion=lambda x, t: 0.0, drift=lambda x, t: 0.0, rate=lambda x, t: 0.0,
                    source=lambda x, t: t + 0.0 * x, payoff=lambda x: x, grid=np.linspace(0.0, 1.0, 11),
                    maturity=2.0, time_steps=10, scheme="implicit", rannacher_steps=0, at=0.5)
    assert sourced["value"] == pytest.approx(0.5 + 0.2 ** 2 * 45, rel=1e-13)


def test_the_cev_pde_is_schroders_closed_form():
    """dS = (r - q) S dt + sigma S^beta dW, beta 0.5: the PDE with a = sigma^2 S^{2 beta}/2 and an absorbing 0
    against the call by the noncentral chi-squared (Hull 27.1), its law qf.noncentral_chi2's 'cev' form."""
    beta, sigma, k = 0.5, 2.0, 95.0
    p = {"spot": 100.0, "rate": 0.05, "dividend_yield": 0.01, "sigma": sigma, "beta": beta, "maturity": 1.0}
    found = solve(diffusion=lambda x, t: 0.5 * sigma ** 2 * x ** (2 * beta), payoff=lambda x: np.maximum(x - k, 0.0),
                  grid=np.linspace(0.0, 400.0, 401), time_steps=100, lower=("value", 0.0))
    _, carry, t, _, scale, b, c = qf.primitives._cev(p)
    a = (k * math.exp(-carry * t)) ** (2 * (1 - beta)) / scale

    def F(z, df, nc):
        return qf.noncentral_chi2(what="cdf", x=z, form="standard", parameters={"df": df, "noncentrality": nc})
    call = 100.0 * math.exp(-0.01) * (1 - F(a, b + 2, c)) - k * math.exp(-0.05) * F(c, b, a)
    assert found["value"] == pytest.approx(call, rel=2e-4)


def test_fd_solve_refuses_what_it_cannot_march():
    for change, message in (({"grid": np.array([0.0, 2.0, 1.0, 3.0, 4.0])}, "increasing"),
                            ({"at": 500.0}, "inside the grid"), ({"exercise": "american"}, "needs exercise_value"),
                            ({"lower": ("value",)}, "not enough values|'linear' alone"),
                            ({"lower": "dirichlet"}, "must be one of"), ({"upper": ("linear", 0.0)}, "'linear' alone"),
                            ({"rannacher_steps": 200}, "must not exceed"), ({"scheme": "theta"}, "must be one of")):
        with pytest.raises(ValueError, match=message):
            solve(**change)


def test_explicit_diffusion_stability_also_checks_the_drift():
    """Tavella and Randall, central differences: a positive diffusion-only diagonal cannot stabilize an explicit
    convection stencil with negative neighbour weights. This positive Gaussian payoff otherwise explodes."""
    with pytest.raises(ValueError, match="unstable"):
        solve(diffusion=lambda x, t: 0.01, drift=lambda x, t: 20.0, rate=lambda x, t: 0.0,
              payoff=lambda x: np.exp(-100.0 * (x - 0.5) ** 2), grid=np.linspace(0.0, 1.0, 101),
              maturity=0.04, time_steps=10, scheme="explicit", rannacher_steps=0,
              lower=("value", 0.0), upper=("value", 0.0), at=0.5)


def test_absorbing_boundary_applies_at_the_terminal_date():
    """An absorbing heat boundary fixes the terminal endpoint; measure-zero payoff values there cannot affect
    interior prices. The one-step Crank-Nicolson system with zero endpoints gives 84/181 at the first node."""
    point = dict(diffusion=lambda x, t: 1.0, drift=lambda x, t: 0.0, rate=lambda x, t: 0.0,
                 grid=np.linspace(0.0, 1.0, 11), maturity=0.01, time_steps=1, rannacher_steps=0,
                 lower=("value", 0.0), upper=("value", 0.0), at=0.1)
    raw = solve(payoff=lambda x: np.ones_like(x), **point)
    clipped = solve(payoff=lambda x: ((x > 0.0) & (x < 1.0)).astype(float), **point)
    assert np.array_equal(raw["values"], clipped["values"])
    assert raw["value"] == pytest.approx(84.0 / 181.0, abs=1e-13)
