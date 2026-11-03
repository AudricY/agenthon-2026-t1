"""Finite-difference PDE solvers: explicit, implicit, Crank-Nicolson, American exercise; any one-dimensional PDE.

`import qf`. Rates and the dividend yield are continuously compounded and annual, times in years, vol annual. A
function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from ._common import Result, choice, integer, positive
from .primitives import _guard

__all__ = ["bs_fd_price", "fd_solve"]

OPTIONS = ("call", "put")
EXERCISES = ("european", "american_projection", "american_lcp")
GRIDS = ("price", "log")
SCHEMES = ("explicit", "implicit", "crank_nicolson")
BOUNDARIES = ("dirichlet", "linear")
THETA = {"explicit": 0.0, "implicit": 1.0, "crank_nicolson": 0.5}
BANDS = (2, 2)                  # a linear boundary row reaches two nodes in: the system is pentadiagonal at its ends
LCP_PASSES = 200                # policy iteration settles in fewer passes than the grid has nodes; a cap, not a rule
EDGES = ("value", "slope", "linear")
FD_EXERCISES = ("european", "american")


def _edges(option, american, s, k, tau, r, q):
    """The asymptotic values at the grid's ends: the discounted forward intrinsic, the immediate one too if American."""
    forward = s * np.exp(-q * tau) - k * np.exp(-r * tau)
    value = np.maximum(forward if option == "call" else -forward, 0.0)
    if american:
        value = np.maximum(value, np.maximum(s - k if option == "call" else k - s, 0.0))
    return value


def _banded(lower, diagonal, upper, first, last):
    """The (2, 2) band of a matrix with the tridiagonal interior given and rows 0 and M as `first`, `last`
    (three coefficients each: row 0 on nodes 0, 1, 2; row M on nodes M, M - 1, M - 2)."""
    m = diagonal.size - 1
    band = np.zeros((5, m + 1))
    band[2] = diagonal
    band[1, 1:] = upper[:-1]          # A[j, j + 1]
    band[3, :-1] = lower[1:]          # A[j + 1, j]
    band[2, 0], band[1, 1], band[0, 2] = first
    band[2, m], band[3, m - 1], band[4, m - 2] = last
    return band


def _apply(band, v):
    """The banded matrix times v."""
    m = v.size
    found = band[2] * v
    found[:-1] += band[1, 1:] * v[1:]
    found[:-2] += band[0, 2:] * v[2:]
    found[1:] += band[3, :-1] * v[:-1]
    found[2:] += band[4, :-2] * v[:-2]
    return found


def _lcp(band, rhs, payoff, start):
    """min(A V - b, V - g) = 0 on the interior rows, by policy iteration from `start` (Howard's algorithm: each pass
    takes, row by row, the condition that is the smaller at the last iterate, and solves the system it makes)."""
    v, active = start, None
    for _ in range(LCP_PASSES):
        chosen = (v - payoff) < (_apply(band, v) - rhs)
        chosen[[0, -1]] = False
        if active is not None and np.array_equal(chosen, active):
            return v
        active = chosen
        system, target = band.copy(), np.where(active, payoff, rhs)
        rows = np.flatnonzero(active)
        for offset in range(5):           # row j of the band sits at band[2 + i - j, j] for column j = i + 2 - offset
            columns = rows + 2 - offset
            keep = (columns >= 0) & (columns < v.size)
            system[offset, columns[keep]] = 0.0
        system[2, rows] = 1.0
        v = solve_banded(BANDS, system, target)
    raise ValueError("american_lcp: the exercise boundary did not settle")


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def bs_fd_price(option, exercise, spot, strike, maturity, rate, dividend_yield, vol, grid, lower, upper,
                space_steps, time_steps, scheme, boundary, rannacher_steps):
    """Black-Scholes PDE by finite differences, American exercise by projection.

    V_t + 1/2 vol^2 S^2 V_SS + (r - q) S V_S - r V = 0 solved backward from the payoff in tau = T - t on M + 1 nodes of
    S (or ln S) and N steps: (I - theta dtau L) V^{n+1} = (I + (1 - theta) dtau L) V^n, L central differences, theta 0
    explicit, 1 implicit, 1/2 Crank-Nicolson; the first `rannacher_steps` steps implicit. American: V >= payoff at
    every step. Delta from the quadratic at the spot, gamma interpolated between nodes, theta from the last step.
    Richardson: prices F at step h and C at 2h give (2^p F - C) / (2^p - 1), p = 2 Crank-Nicolson, 1 implicit. The
    explicit scheme is stable only where dtau vol^2 S^2 / dS^2 <= 1 on every node. Any other one-factor PDE (its own
    drift, diffusion, rate, payoff or boundaries: a short rate, CEV, local vol) is qf.fd_solve.

    conventions:
    - option: 'call' or 'put'; exercise: 'european', 'american_projection' (each step solved, then max with the payoff,
      Hull's) or 'american_lcp' (each step's linear complementarity problem solved exactly, the PSOR limit)
    - grid: 'price' (nodes uniform in S, lower may be 0) or 'log' (uniform in ln S, lower > 0); lower < spot < upper
    - space_steps: M intervals; time_steps: N steps; rannacher_steps: implicit steps that start the march (0 for none)
    - scheme: 'explicit', 'implicit' or 'crank_nicolson'
    - boundary: 'dirichlet' (the discounted intrinsic e^-qt S - e^-rt K at both ends, at least the payoff if
      American) or 'linear' (V linear in S through the last three nodes: V_SS = 0)
    returns: a dict: price at the spot; delta, gamma (per unit S), theta (per year, dV/dt); nodes, the grid's S
      values; values, V at today on the nodes
    source: Hull, "Finite difference methods"; Wilmott (2006), ch. 77-78; Rannacher (1984); Tavella and Randall (2000)
    aliases: Crank-Nicolson, finite-difference PDE, explicit finite difference, implicit finite difference, Black-Scholes PDE, PDE solver, Rannacher, PSOR, [finite differences]
    example:
        qf.bs_fd_price(option="put", exercise="american_lcp", spot=100.0, strike=100.0, maturity=1.0, rate=0.05,
                       dividend_yield=0.0, vol=0.3, grid="log", lower=10.0, upper=400.0, space_steps=200,
                       time_steps=100, scheme="crank_nicolson", boundary="dirichlet", rannacher_steps=2)
    """
    choice("option", option, OPTIONS)
    choice("exercise", exercise, EXERCISES)
    choice("grid", grid, GRIDS)
    choice("scheme", scheme, SCHEMES)
    choice("boundary", boundary, BOUNDARIES)
    s0, k, t, sigma = (float(positive(name, value)) for name, value in
                       (("spot", spot), ("strike", strike), ("maturity", maturity), ("vol", vol)))
    r, q = float(rate), float(dividend_yield)
    m, n = integer("space_steps", space_steps, 3), integer("time_steps", time_steps, 1)
    smoothing = integer("rannacher_steps", rannacher_steps, 0)
    if smoothing > n:
        raise ValueError("rannacher_steps must not exceed time_steps")
    low, high = float(lower), float(upper)
    if not (low < s0 < high) or low < 0 or (grid == "log" and low <= 0):
        raise ValueError("lower < spot < upper must hold, lower >= 0 ('price') or lower > 0 ('log')")
    american = exercise != "european"
    if grid == "price":
        nodes = np.linspace(low, high, m + 1)
        h = nodes[1] - nodes[0]
        s2, drift = sigma ** 2 * nodes ** 2 / h ** 2, (r - q) * nodes / h
    else:
        y = np.linspace(np.log(low), np.log(high), m + 1)
        nodes, h = np.exp(y), y[1] - y[0]
        s2 = np.full(m + 1, sigma ** 2 / h ** 2)
        drift = np.full(m + 1, (r - q - 0.5 * sigma ** 2) / h)
    a, b, c = 0.5 * s2 - 0.5 * drift, -s2 - r, 0.5 * s2 + 0.5 * drift      # (L V)_j = a V_{j-1} + b V_j + c V_{j+1}
    payoff = np.maximum(nodes - k if option == "call" else k - nodes, 0.0)
    dtau = t / n
    bound = 2.0 * (high + k)        # no call is worth more than S, no put more than K: past this the march diverged
    first_ratio = (nodes[1] - nodes[0]) / (nodes[2] - nodes[1])
    last_ratio = (nodes[m] - nodes[m - 1]) / (nodes[m - 1] - nodes[m - 2])
    linear_rows = ((1.0, -(1.0 + first_ratio), first_ratio), (1.0, -(1.0 + last_ratio), last_ratio))
    v, previous = payoff.copy(), payoff.copy()
    for step in range(n):
        theta = 1.0 if step < smoothing else THETA[scheme]
        tau = (step + 1) * dtau
        rhs = v.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            rhs[1:-1] += (1.0 - theta) * dtau * (a[1:-1] * v[:-2] + b[1:-1] * v[1:-1] + c[1:-1] * v[2:])
        if boundary == "dirichlet":
            rows = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
            rhs[0], rhs[-1] = _edges(option, american, nodes[[0, -1]], k, tau, r, q)
        else:
            rows, rhs[0], rhs[-1] = linear_rows, 0.0, 0.0
        band = _banded(-theta * dtau * a, 1.0 - theta * dtau * b, -theta * dtau * c, *rows)
        if not np.all(np.abs(rhs) <= bound):
            raise ValueError(f"the explicit steps diverged: dtau vol^2 S^2 / dS^2 reaches {dtau * s2.max():.3g} on the "
                             "grid, above 1; take more time_steps or another scheme")
        previous, v = v, solve_banded(BANDS, band, rhs)
        if exercise == "american_projection":
            v = np.maximum(v, payoff)
        elif exercise == "american_lcp":
            v = _lcp(band, rhs, payoff, np.maximum(v, payoff))
    at = np.log(s0) if grid == "log" else s0
    axis = np.log(nodes) if grid == "log" else nodes
    centre = int(np.clip(np.rint((at - axis[0]) / h), 1, m - 1))
    u = (at - axis[centre]) / h
    below = int(np.clip(np.floor((at - axis[0]) / h), 1, m - 2))
    w = (at - axis[below]) / h

    def read(values):
        first = (values[centre + 1] - values[centre - 1]) / 2.0
        second = values[centre + 1] - 2.0 * values[centre] + values[centre - 1]
        either = np.diff(values[below - 1:below + 3], 2)        # the second differences at the nodes either side
        return (values[centre] + u * first + 0.5 * u * u * second, (first + u * second) / h,
                ((1.0 - w) * either[0] + w * either[1]) / h ** 2)

    price, dv, d2v = read(v)
    if grid == "log":
        dv, d2v = dv / s0, (d2v - dv) / s0 ** 2
    return Result({"price": float(price), "delta": float(dv), "gamma": float(d2v),
                   "theta": float(-(price - read(previous)[0]) / dtau), "nodes": nodes, "values": v})


def _edge(name, given):
    """A boundary as its kind and its value, a function of t ('linear': none)."""
    kind, value = (given, None) if isinstance(given, str) else tuple(given)
    choice(name, kind, EDGES)
    if (kind == "linear") != (value is None):
        raise ValueError(f"{name}: 'linear' alone, or ('value', g) or ('slope', s) with g or s a number or a function "
                         "of t")
    return kind, (value if callable(value) else (lambda t, value=value: value))


def _row(kind, nodes, end):
    """A boundary row's three coefficients on the end node and the next two in (`_banded`'s order)."""
    if kind == "value":
        return 1.0, 0.0, 0.0
    near, far = abs(nodes[1 if end == 0 else -2] - nodes[0 if end == 0 else -1]), abs(
        nodes[2 if end == 0 else -3] - nodes[1 if end == 0 else -2])
    if kind == "linear":
        ratio = near / far
        return 1.0, -(1.0 + ratio), ratio
    sign = -1.0 if end == 0 else 1.0      # V_x by the one-sided quadratic through the end node and the next two in
    return (sign * (2.0 * near + far) / (near * (near + far)), -sign * (near + far) / (near * far),
            sign * near / (far * (near + far)))


@_guard(time=('maturity',))
def fd_solve(diffusion, drift, rate, source, payoff, grid, maturity, time_steps, scheme, rannacher_steps, lower, upper,
             exercise, exercise_value, at):
    """One-dimensional PDE solver: V_t + a V_xx + b V_x - r V + f = 0 for any a, b, r, f (Feynman-Kac).

    From V(x, T) = payoff back to t = 0 in N theta-scheme steps on the nodes of x (uniform or not): (I - theta dt
    L_n) V^n = (I + (1 - theta) dt L_{n+1}) V^{n+1} + dt (theta f_n + (1 - theta) f_{n+1}), L V = a V_xx + b V_x
    - r V by central differences; theta 0 explicit (nonnegative neighbour weights and 1 + dt L_jj >= 0),
    1 implicit, 1/2 Crank-Nicolson,
    the first rannacher_steps implicit. American: V = max(V, exercise value) each step. Value, V_x, V_xx at `at`
    from the quadratic through the nearest nodes.

    Used for: barrier options on a grid (('value', 0) at the barrier), short-rate bond PDEs (x the rate: a =
    sigma^2/2, b = kappa (theta - x), r = x), CEV and local-vol PDEs, convertibles. Black-Scholes is bs_fd_price.

    conventions:
    - diffusion, drift, rate, source: a, b, r, f, functions of (x, t), x the nodes' array, t years from today;
      source None for none
    - payoff: V at maturity, a function of x or an array on the nodes; grid: the nodes, increasing
    - maturity: T, years; time_steps: N; scheme: 'explicit', 'implicit' or 'crank_nicolson'; rannacher_steps: 0
      for none
    - lower, upper: 'linear' (V_xx = 0), ('value', g) or ('slope', s) (V_x = s), g and s numbers or functions of t
    - exercise: 'european' or 'american'; exercise_value: None, or a function of (x, t)
    - at: the x the value is read at (the spot, today's rate)
    returns: a dict: value, delta (V_x) and gamma (V_xx) at `at`; nodes; values (V today on the nodes)
    source: Wilmott (2006), ch. 77-78; Tavella and Randall (2000), ch. 3 and 4; Rannacher (1984)
    aliases: Feynman-Kac PDE, one-dimensional PDE, CEV PDE
    example:
        import qf, numpy as np
        # Result: value 10.44050109.
        qf.fd_solve(diffusion=lambda x, t: 0.5 * 0.2 ** 2 * x ** 2, drift=lambda x, t: 0.05 * x, rate=lambda x, t: 0.05, source=None, payoff=lambda x: np.maximum(x - 100.0, 0.0), grid=np.linspace(0.0, 400.0, 201), maturity=1.0, time_steps=100, scheme="crank_nicolson", rannacher_steps=2, lower=("value", 0.0), upper="linear", exercise="european", exercise_value=None, at=100.0)
    """
    choice("scheme", scheme, SCHEMES)
    choice("exercise", exercise, FD_EXERCISES)
    nodes = np.asarray(grid, dtype=float).ravel()
    if nodes.size < 4 or np.any(np.diff(nodes) <= 0):
        raise ValueError("grid must be increasing nodes, at least four")
    t_end = float(positive("maturity", maturity))
    n = integer("time_steps", time_steps, 1)
    smoothing = integer("rannacher_steps", rannacher_steps, 0)
    if smoothing > n:
        raise ValueError("rannacher_steps must not exceed time_steps")
    x0 = float(at)
    if not nodes[0] < x0 < nodes[-1]:
        raise ValueError("at must lie inside the grid")
    low_kind, low_value = _edge("lower", lower)
    high_kind, high_value = _edge("upper", upper)
    if (exercise == "american") != (exercise_value is not None):
        raise ValueError("'american' needs exercise_value, a function of (x, t); 'european' takes None")
    m = nodes.size - 1
    left, right = nodes[1:-1] - nodes[:-2], nodes[2:] - nodes[1:-1]
    second = (2.0 / (left * (left + right)), -2.0 / (left * right), 2.0 / (right * (left + right)))
    first = (-right / (left * (left + right)), (right - left) / (left * right), left / (right * (left + right)))

    def coefficients(t):
        """(L V)_j = lo_j V_{j-1} + mid_j V_j + hi_j V_{j+1} on the interior nodes, and f there."""
        x = nodes[1:-1]
        a = np.broadcast_to(np.asarray(diffusion(x, t), dtype=float), x.shape)
        b = np.broadcast_to(np.asarray(drift(x, t), dtype=float), x.shape)
        r = np.broadcast_to(np.asarray(rate(x, t), dtype=float), x.shape)
        f = np.zeros(x.shape) if source is None else np.broadcast_to(np.asarray(source(x, t), dtype=float), x.shape)
        return a * second[0] + b * first[0], a * second[1] + b * first[1] - r, a * second[2] + b * first[2], f, a, r

    v = np.asarray(payoff(nodes) if callable(payoff) else payoff, dtype=float)
    v = np.broadcast_to(v, nodes.shape).astype(float)
    if low_kind == "value":
        v[0] = float(low_value(t_end))
    if high_kind == "value":
        v[-1] = float(high_value(t_end))
    rows = (_row(low_kind, nodes, 0), _row(high_kind, nodes, -1))
    dt = t_end / n
    later = coefficients(t_end)
    for step in range(n):
        t_now = t_end - (step + 1) * dt
        theta = 1.0 if step < smoothing else THETA[scheme]
        now = coefficients(t_now)
        if theta == 0.0 and (np.any(later[0] < -1e-12) or np.any(later[2] < -1e-12)
                             or np.any(1.0 + dt * later[1] < -1e-12)):
            raise ValueError("the explicit scheme is unstable here: the central stencil is not monotone; take "
                             "a finer grid, more time_steps or another scheme")
        rhs = np.zeros(nodes.size)
        rhs[1:-1] = v[1:-1] + (1.0 - theta) * dt * (later[0] * v[:-2] + later[1] * v[1:-1] + later[2] * v[2:]) + dt * (
            theta * now[3] + (1.0 - theta) * later[3])
        rhs[0] = 0.0 if low_kind == "linear" else float(low_value(t_now))
        rhs[-1] = 0.0 if high_kind == "linear" else float(high_value(t_now))
        lo, mid, hi = np.zeros(m + 1), np.ones(m + 1), np.zeros(m + 1)
        lo[1:-1], mid[1:-1], hi[1:-1] = -theta * dt * now[0], 1.0 - theta * dt * now[1], -theta * dt * now[2]
        v = solve_banded(BANDS, _banded(lo, mid, hi, *rows), rhs)
        if exercise == "american":
            v = np.maximum(v, np.broadcast_to(np.asarray(exercise_value(nodes, t_now), dtype=float), nodes.shape))
        if not np.all(np.isfinite(v)):
            raise ValueError("the march diverged; take more time_steps, a finer grid or another scheme")
        later = now
    centre = int(np.clip(np.argmin(np.abs(nodes - x0)), 1, m - 1))
    xs, ys = nodes[centre - 1:centre + 2], v[centre - 1:centre + 2]
    d1, d2 = (ys[1] - ys[0]) / (xs[1] - xs[0]), (ys[2] - ys[1]) / (xs[2] - xs[1])
    curvature = 2.0 * (d2 - d1) / (xs[2] - xs[0])
    slope = d1 + 0.5 * curvature * (2.0 * x0 - xs[0] - xs[1])
    value = ys[0] + d1 * (x0 - xs[0]) + 0.5 * curvature * (x0 - xs[0]) * (x0 - xs[1])
    return Result({"value": float(value), "delta": float(slope), "gamma": float(curvature), "nodes": nodes,
                   "values": v})
