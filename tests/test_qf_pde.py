"""agent/qf/pde.py, the finite-difference shelf (runs/lanes/lib6-prompt.md, rank 1 of
docs/research/library-syllabus-1011.md): the Black-Scholes PDE on a grid against `qf.bs_price` and `qf.bs.bs_greeks`
at a tolerance the grid sets, against Hull's worked finite-difference values for an American put and a many-step
binomial tree written here. The inputs are textbook examples or chosen freely; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf
import qf.bs as bs
import qf.pde as pde

pytestmark = pytest.mark.fast

CALL = dict(spot=100.0, strike=100.0, maturity=0.5, rate=0.03, dividend_yield=0.0, vol=0.2)
LOG = dict(grid="log", lower=100.0 / 8, upper=800.0)     # the spot and strike on a node for every even M
HULL = dict(spot=50.0, strike=50.0, maturity=5 / 12, rate=0.10, dividend_yield=0.0, vol=0.40)


def fd(option, exercise, args, grid, m, n, scheme, boundary="dirichlet", rannacher=0):
    return pde.bs_fd_price(option=option, exercise=exercise, **args, **grid, space_steps=m, time_steps=n,
                           scheme=scheme, boundary=boundary, rannacher_steps=rannacher)


def crr_american_put(s, k, t, r, q, vol, steps):
    """Cox, Ross and Rubinstein's tree with early exercise at every node (Hull, "Binomial trees")."""
    dt = t / steps
    u = math.exp(vol * math.sqrt(dt))
    p, disc = (math.exp((r - q) * dt) - 1 / u) / (u - 1 / u), math.exp(-r * dt)
    value = np.maximum(k - s * u ** np.arange(steps, -steps - 1, -2.0), 0.0)
    for i in range(steps - 1, -1, -1):
        value = np.maximum(disc * (p * value[:-1] + (1 - p) * value[1:]), k - s * u ** np.arange(i, -i - 1, -2.0))
    return float(value[0])


def test_the_pde_card_keeps_the_format_and_its_example_runs():
    check_cards(pde)


def test_crank_nicolson_converges_to_black_scholes_at_second_order_and_richardson_removes_the_leading_error():
    """Wilmott, ch. 78: Crank-Nicolson's error falls by four as dS and dt halve; with Rannacher's start the kink at
    the strike does not spoil it. Richardson's (4 F_h - F_2h) / 3 then leaves almost nothing."""
    exact = qf.bs_price("call", 100.0, 100.0, 0.5, 0.03, 0.0, 0.2)
    errors = [fd("call", "european", CALL, LOG, m, m // 2, "crank_nicolson", rannacher=2)["price"] - exact
              for m in (200, 400, 800)]
    assert abs(errors[-1]) < 1.5e-3 and errors[0] / errors[1] == pytest.approx(4.0, rel=0.02)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.02)
    fine, coarse = errors[2] + exact, errors[1] + exact
    assert abs((4 * fine - coarse) / 3 - exact) < 2e-5


def test_the_greeks_read_off_the_grid_are_black_scholes():
    """Hull, "The Greek letters": delta, gamma and theta (per year) of the European call."""
    greeks = bs.bs_greeks(option="call", **CALL, vega_per=1.0, theta_per="year", rho_per=1.0)
    found = fd("call", "european", CALL, LOG, 800, 400, "crank_nicolson", rannacher=2)
    assert found["delta"] == pytest.approx(greeks["delta"], abs=2e-5)
    assert found["gamma"] == pytest.approx(greeks["gamma"], abs=1e-5)
    assert found["theta"] == pytest.approx(greeks["theta"], abs=1e-2)
    put = bs.bs_greeks(option="put", **CALL, vega_per=1.0, theta_per="year", rho_per=1.0)
    grid = dict(grid="price", lower=0.0, upper=400.0)
    found = fd("put", "european", CALL, grid, 800, 400, "crank_nicolson", rannacher=2)
    assert found["price"] == pytest.approx(put["price"], abs=2e-3)
    assert found["delta"] == pytest.approx(put["delta"], abs=2e-4) and found["gamma"] == pytest.approx(put["gamma"],
                                                                                                    abs=1e-4)


def test_gamma_off_a_node_is_read_between_the_second_differences_either_side():
    """The lane's review: the quadratic through the three nearest nodes carries the centre node's second difference,
    an error of order dS where the spot falls between nodes (the card's own example grid); read between the second
    differences of the nodes either side, gamma off a node is as close as on one."""
    gamma = bs.bs_greeks(option="call", **CALL, vega_per=1.0, theta_per="year", rho_per=1.0)["gamma"]
    for grid in (dict(grid="log", lower=10.0, upper=800.0), dict(grid="price", lower=0.0, upper=397.0)):
        found = fd("call", "european", CALL, grid, 1000, 500, "crank_nicolson", "linear", 4)
        assert found["gamma"] == pytest.approx(gamma, abs=1e-5), grid

def test_rannachers_implicit_start_removes_crank_nicolsons_gamma_oscillation():
    """Pooley, Vetzal and Forsyth (2003): with a long time step on a fine grid, Crank-Nicolson's gamma at the strike
    oscillates; two implicit steps first damp it."""
    gamma = bs.bs_greeks(option="call", **CALL, vega_per=1.0, theta_per="year", rho_per=1.0)["gamma"]
    plain = fd("call", "european", CALL, LOG, 800, 10, "crank_nicolson")["gamma"]
    smoothed = fd("call", "european", CALL, LOG, 800, 10, "crank_nicolson", rannacher=2)["gamma"]
    assert abs(plain - gamma) > 0.1 and abs(smoothed - gamma) < 1e-3


def test_hulls_american_put_on_his_finite_difference_grids():
    """Hull, "Finite difference methods": an American put, S0 = K = 50, r = 10%, vol = 40%, five months, dS = 5 on
    [0, 100], ten steps of half a month: the implicit method gives 4.07 and the explicit 4.26. Hull's implicit and
    explicit methods solve each step and then take the larger of the value and the payoff."""
    grid = dict(grid="price", lower=0.0, upper=100.0)
    implicit = fd("put", "american_projection", HULL, grid, 20, 10, "implicit")["price"]
    explicit = fd("put", "american_projection", HULL, grid, 20, 10, "explicit")["price"]
    assert implicit == pytest.approx(4.07, abs=0.005) and explicit == pytest.approx(4.26, abs=0.005)


def test_an_american_put_on_a_fine_grid_is_the_binomial_trees():
    """The same put on a fine log grid, by projection and by the exact complementarity problem, against Cox, Ross and
    Rubinstein's tree of 2,000 steps (Hull's converged 4.28); the American exceeds the European."""
    grid = dict(grid="log", lower=2.0, upper=1250.0)
    tree = crr_american_put(50.0, 50.0, 5 / 12, 0.10, 0.0, 0.40, 2000)
    lcp = fd("put", "american_lcp", HULL, grid, 800, 400, "crank_nicolson", rannacher=2)
    projection = fd("put", "american_projection", HULL, grid, 800, 400, "crank_nicolson", rannacher=2)
    european = qf.bs_price("put", 50.0, 50.0, 5 / 12, 0.10, 0.0, 0.40)
    assert lcp["price"] == pytest.approx(tree, abs=1e-3) and projection["price"] == pytest.approx(tree, abs=1.5e-3)
    assert tree == pytest.approx(4.28, abs=0.005) and lcp["price"] > european + 0.2
    assert np.all(lcp["values"] >= np.maximum(50.0 - lcp["nodes"], 0.0) - 1e-12)


def test_an_american_call_without_a_dividend_is_the_european_and_with_one_is_worth_more():
    """Merton (1973): never exercise a call early on a stock that pays nothing; with a yield above the rate the
    early exercise has value."""
    european = fd("call", "european", CALL, LOG, 400, 200, "crank_nicolson", rannacher=2)["price"]
    american = fd("call", "american_lcp", CALL, LOG, 400, 200, "crank_nicolson", rannacher=2)["price"]
    assert american == pytest.approx(european, abs=1e-10)
    paying = {**CALL, "dividend_yield": 0.08}
    assert (fd("call", "american_lcp", paying, LOG, 400, 200, "crank_nicolson", rannacher=2)["price"]
            > fd("call", "european", paying, LOG, 400, 200, "crank_nicolson", rannacher=2)["price"] + 0.05)


def test_the_boundaries_agree_far_away_and_differ_near_the_spot():
    """V_SS = 0 at a distant edge and the discounted intrinsic there are the same condition to many places; on a
    narrow grid they differ."""
    wide = [fd("put", "european", CALL, LOG, 400, 200, "crank_nicolson", boundary, 2)["price"]
            for boundary in ("dirichlet", "linear")]
    narrow = [fd("put", "european", CALL, dict(grid="log", lower=80.0, upper=125.0), 200, 200, "crank_nicolson",
                 boundary, 2)["price"] for boundary in ("dirichlet", "linear")]
    assert wide[0] == pytest.approx(wide[1], abs=1e-9) and abs(narrow[0] - narrow[1]) > 5e-4


def test_the_dirichlet_edges_are_the_discounted_intrinsic_and_a_spot_between_nodes_is_read_off_the_quadratic():
    """The values the grid holds at its ends are the stated boundary (Hull: K e^-rT for a European put at S = 0); a
    spot between two nodes is priced and hedged as Black-Scholes prices and hedges it there."""
    found = fd("put", "european", CALL, dict(grid="price", lower=0.0, upper=250.0), 500, 200, "crank_nicolson",
               rannacher=2)
    assert found["values"][0] == pytest.approx(100.0 * math.exp(-0.03 * 0.5), abs=1e-12)
    assert found["values"][-1] == 0.0
    call = fd("call", "european", CALL, dict(grid="log", lower=20.0, upper=500.0), 600, 300, "crank_nicolson",
              rannacher=2)
    assert call["values"][-1] == pytest.approx(500.0 - 100.0 * math.exp(-0.03 * 0.5), abs=1e-12)
    off = {**CALL, "spot": 101.3}
    found = fd("call", "european", off, LOG, 400, 200, "crank_nicolson", rannacher=2)
    greeks = bs.bs_greeks(option="call", **off, vega_per=1.0, theta_per="year", rho_per=1.0)
    assert found["price"] == pytest.approx(greeks["price"], abs=5e-3)
    assert found["delta"] == pytest.approx(greeks["delta"], abs=2e-4)


def test_an_explicit_grid_converges_where_it_is_stable_and_one_that_diverges_is_refused():
    exact = qf.bs_price("put", 100.0, 100.0, 0.5, 0.03, 0.0, 0.2)
    stable = fd("put", "european", CALL, LOG, 200, 1000, "explicit")["price"]
    assert stable == pytest.approx(exact, abs=0.02)
    with pytest.raises(ValueError, match="diverged"):
        fd("put", "european", CALL, dict(grid="price", lower=0.0, upper=400.0), 400, 50, "explicit")


@pytest.mark.parametrize("bad, match", [
    (dict(scheme="cn"), "scheme"), (dict(exercise="bermudan"), "exercise"), (dict(grid="sinh"), "grid"),
    (dict(lower=0.0), "lower"), (dict(upper=90.0), "lower"), (dict(space_steps=2), "space_steps"),
    (dict(rannacher_steps=300), "rannacher_steps"), (dict(boundary="neumann"), "boundary")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(option="put", exercise="european", **CALL, **LOG, space_steps=100, time_steps=50,
                scheme="crank_nicolson", boundary="dirichlet", rannacher_steps=2)
    with pytest.raises(ValueError, match=match):
        pde.bs_fd_price(**{**args, **bad})
