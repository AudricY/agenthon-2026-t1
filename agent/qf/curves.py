"""Zero curves: par, bond and multi-curve bootstraps, Nelson-Siegel, discount factors, extrapolation, forward rates.

`import qf`. Times are year fractions from today under the caller's day count (cash-flow times, maturities,
fixings); zero rates are continuously compounded unless a function names another basis; discount factors are
P(0, t). A scalar result is a float; a function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares, minimize, newton

from ._common import Result, choice, out, positive
from .primitives import _guard

__all__ = ["bootstrap_bond_prices", "bootstrap_multicurve", "bootstrap_par_curve", "curve_discount",
           "curve_discount_extrapolated", "forward_rates", "nelson_siegel_fit", "nelson_siegel_yield",
           "par_key_rate_durations"]

INTERPOLATIONS = ("linear_zero", "log_linear_discount", "cubic_log_discount")
SOLVERS = ("brentq", "newton")
EXTRAPOLATIONS = ("flat_zero", "flat_forward", "linear_zero", "refuse")
BOND_METHODS = ("sequential", "least_squares")
VARIANTS = {"nelson_siegel": (3, 1), "svensson": (4, 2)}    # betas, decays
DECAY_FORMS = ("rate", "time")
FIT_METHODS = ("L-BFGS-B", "Nelder-Mead", "Powell", "TNC", "SLSQP")
DECAY_FLOOR = 1e-6
INSTRUMENT_FIELDS = {"deposit": ("rate", "start", "end", "accrual"), "fra": ("rate", "start", "end", "accrual"),
                     "future": ("price", "start", "end", "accrual", "convexity"),
                     "ois": ("rate", "start", "times", "accruals"),
                     "swap": ("rate", "start", "fixed_times", "fixed_accruals", "float_times")}
DISCOUNT_KINDS = ("deposit", "ois")
PROJECTION_KINDS = ("deposit", "fra", "future", "swap")
REPRICED = 1e-12                # every instrument's value per unit notional at its quote, once the curve is built


# ------------------------------------------------------------------------------------------------ curves


def _increasing(name, values):
    array = positive(name, values).ravel()
    if np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def _interpolation_weights(times, nodes, interpolation):
    """W with -ln D(times) = W @ g, g = -ln D(nodes): every interpolation here is linear in g. g(0) = 0;
    'linear_zero' holds the first node's zero rate below it."""
    t = np.asarray(times, dtype=float)
    unit = np.eye(nodes.size)
    if interpolation == "linear_zero":
        return np.column_stack([t * np.interp(t, nodes, unit[j] / nodes) for j in range(nodes.size)])
    knots = np.r_[0.0, nodes]
    if interpolation == "log_linear_discount":
        return np.column_stack([np.interp(t, knots, np.r_[0.0, unit[j]]) for j in range(nodes.size)])
    return np.column_stack([CubicSpline(knots, np.r_[0.0, unit[j]], bc_type="not-a-knot")(t)
                            for j in range(nodes.size)])


def _par_schedule(maturity, frequency):
    """Payment times T - j/f > 0 in increasing order and their accruals; the first period is the stub."""
    periods = int(np.ceil(maturity * frequency - 1e-9))
    times = maturity - np.arange(periods)[::-1] / frequency
    return times, np.diff(np.r_[0.0, times])


@_guard(rate=('par_rates',), time=('maturities',))
def bootstrap_par_curve(maturities, par_rates, frequency, interpolation, solver):
    """Zero curve bootstrapped from par yields or swap rates, every quote repriced to par.

    Instrument k pays c_k a_j at T_k - j/f > 0 (a_j the period's accrual, the first a short stub where T_k f is not a
    whole number) and 1 at T_k; its node solves sum_j c_k a_j D(t_j) + D(T_k) = 1 by the root-finder named, D read
    off the interpolation through D(0) = 1 and the nodes. A maturity of one period or less is a deposit,
    D = 1 / (1 + c T). Interpolations: 'linear_zero', zero rates linear (the first node's rate below it);
    'log_linear_discount', ln D linear (flat forwards); 'cubic_log_discount', a not-a-knot cubic spline on -ln D = z T
    through 0. The cubic is global: its nodes are solved again in order until every quote reprices to 1e-13 on the
    final curve, as QuantLib's iterative bootstrap does.

    conventions:
    - par_rates: decimals with frequency f coupons a year (a swap's fixed leg, a bond's coupon); maturities in years
    - interpolation: 'linear_zero', 'log_linear_discount' or 'cubic_log_discount', as above; the cubic's nodes
      re-solved until all reprice, which a one-pass bootstrap (each node solved once, in order) is not
    - solver: 'brentq' (on the node's zero rate in [-1, 1]) or 'newton' (the secant from the par rate); same root
    returns: a dict: discount_factors and zero_rates (continuously compounded, -ln D / T) at the
      maturities
    source: Hagan and West (2006), "Interpolation methods for curve construction", Applied Math. Finance 13(2)
    aliases: bootstrap, bootstrapping, [zero curve], par yield curve, [swap curve], [discount factors], [spot rates], deposits
    example:
        qf.bootstrap_par_curve(maturities=[0.5, 1.0, 2.0, 3.0, 5.0],
                               par_rates=[0.040, 0.042, 0.043, 0.044, 0.045], frequency=2,
                               interpolation="log_linear_discount", solver="brentq")
    """
    choice("interpolation", interpolation, INTERPOLATIONS)
    choice("solver", solver, SOLVERS)
    nodes = _increasing("maturities", maturities)
    coupons = np.asarray(par_rates, dtype=float).ravel()
    if coupons.shape != nodes.shape or not np.all(np.isfinite(coupons)):
        raise ValueError("par_rates must hold one finite rate per maturity")
    f = _whole_frequency(frequency)
    instruments = []
    for k, maturity in enumerate(nodes):
        times, accruals = _par_schedule(maturity, f)
        flows = coupons[k] * accruals
        flows[-1] += 1.0
        instruments.append((flows, _interpolation_weights(times, nodes, interpolation)))
    g = coupons * nodes
    for _ in range(200):
        for k, (flows, weights) in enumerate(instruments):
            rest = weights @ np.where(np.arange(nodes.size) == k, 0.0, g)

            def gap(zero, k=k, flows=flows, weights=weights, rest=rest):
                return flows @ np.exp(-(rest + weights[:, k] * zero * nodes[k])) - 1.0

            try:
                zero = (brentq(gap, -1.0, 1.0, xtol=1e-16, rtol=4.5e-15) if solver == "brentq"
                        else newton(gap, x0=g[k] / nodes[k], tol=1e-15, maxiter=100))
            except (ValueError, RuntimeError) as failure:
                raise ValueError(f"par_rates: no zero rate reprices the maturity-{nodes[k]} instrument ({failure})")
            g[k] = zero * nodes[k]
        if max(abs(flows @ np.exp(-(weights @ g)) - 1.0) for flows, weights in instruments) <= 1e-13:
            break
    else:
        raise ValueError("par_rates: the cubic bootstrap did not settle in 200 passes")
    return Result({"discount_factors": np.exp(-g), "zero_rates": g / nodes})

@_guard(rate=('par_rates',), time=('times', 'maturities', 'key_maturities',))
def par_key_rate_durations(cash_flows, times, maturities, par_rates, frequency, interpolation, key_maturities, shift,
                           bump, difference):
    """Key-rate durations on the par curve: each key's par rate bumped, the zero curve bootstrapped again, repriced.

    PV = sum_i CF_i D(t_i), D the curve qf.bootstrap_par_curve builds from the par rates, read between its nodes as
    curve_discount reads it. Key k's bump adds bump s_k(T) to each par rate: 'node' the key's maturity alone, 'tent'
    Ho's (1992) shift over the key maturities (1 at key k, linear to 0 at its neighbours, flat beyond the end keys, the
    tents summing to a parallel shift); the curve is bootstrapped again and the flows repriced. KRD_k = -(PV_up - PV)
    / (bump PV) ('forward') or -(PV_up - PV_down) / (2 bump PV) ('central'). Zero rates bumped in place, with no
    bootstrap: qf.key_rate_durations.

    conventions:
    - shift: 'node' or 'tent' over the par maturities; bump: in rate units (0.0001 is 1 bp); difference: 'forward' or
      'central'
    - maturities, par_rates, frequency, interpolation: qf.bootstrap_par_curve's (par yields or swap rates, f coupons
      a year); key_maturities: among the maturities
    - cash_flows at times (years), none past the last maturity (the curve does not reach it)
    returns: a dict: durations (one per key; times price the dollar KRD, times price / 10,000 the KR01); price (PV on
      the curve)
    source: Ho (1992), "Key rate durations: measures of interest rate risks", J. Fixed Income 2(2); Tuckman and Serrat,
      Fixed Income Securities (2011), ch. 5 (key rates on par yields)
    aliases: par key rate duration, par-curve key rate durations, key rate durations on par yields, par KRD, [par rate shock], [rebootstrap]
    example:
        # Returns price 101.379, durations near [0.0042, 0.0085, 2.7338]: the 3-year par rate carries the bond.
        qf.par_key_rate_durations(cash_flows=[5.0, 5.0, 105.0], times=[1.0, 2.0, 3.0], maturities=[1.0, 2.0, 3.0], par_rates=[0.04, 0.042, 0.045], frequency=1, interpolation="linear_zero", key_maturities=[1.0, 2.0, 3.0], shift="node", bump=0.0001, difference="central")
    """
    choice("shift", shift, ("node", "tent"))
    choice("difference", difference, ("forward", "central"))
    choice("interpolation", interpolation, INTERPOLATIONS)
    cf, t = _flows(cash_flows, times)
    nodes = _increasing("maturities", maturities)
    par = np.asarray(par_rates, dtype=float).ravel()
    if par.shape != nodes.shape or not np.all(np.isfinite(par)):
        raise ValueError("par_rates must hold one finite rate per maturity")
    keys = _increasing("key_maturities", key_maturities)
    if not np.all(np.isin(keys, nodes)):
        raise ValueError("key_maturities must be among the maturities")
    if np.any(t > nodes[-1] * (1 + 1e-12)):
        raise ValueError("times must not pass the last maturity: the par curve does not reach them")
    size = float(positive("bump", bump))
    weights = _interpolation_weights(t, nodes, interpolation)

    def value(rates):
        curve = bootstrap_par_curve(nodes, rates, frequency, interpolation, "brentq")
        return float(cf @ np.exp(weights @ np.log(curve["discount_factors"])))

    base = value(par)
    if not base > 0:
        raise ValueError("cash_flows must have a positive value on the curve")
    durations = np.empty(keys.size)
    for k, key in enumerate(keys):
        s = (nodes == key).astype(float) if shift == "node" else np.interp(nodes, keys, np.eye(keys.size)[k])
        up = value(par + size * s)
        down = base if difference == "forward" else value(par - size * s)
        durations[k] = -(up - down) / ((1.0 if difference == "forward" else 2.0) * size * base)
    return Result({"durations": durations, "price": base})


@_guard(time=('node_times',), price=('prices',))
def bootstrap_bond_prices(cash_flows, times, prices, accrued, method, node_times, weights, interpolation, compounding):
    """Zero curve from coupon-bond prices: sequential bootstrap or least squares in price, the zero rates' compounding named.

    Each bond's dirty price P_i is sum_j c_ij D(t_ij), D read off the interpolation through D(0) = 1 and the nodes.
    'sequential': a node at each bond's maturity, solved in maturity order so each bond reprices exactly (the
    cubic's nodes re-solved until all reprice); 'least_squares': nodes at
    node_times, their zero rates minimising sum_i w_i (P_hat_i - P_i)^2 (scipy.optimize.least_squares), fewer nodes
    than bonds giving a smooth fit; weights 1 / D_i^2 (D_i the modified duration) make the errors about yield errors.

    The nodes are zero rates in the named compounding: 'linear_zero' holds them linear in time inside the solve (spot
    rates linear in semi-annual compounding: 2), D(t) = (1 + z(t)/k)^{-k t}; elsewhere it names the rates returned.

    conventions:
    - cash_flows, times: one array per bond, its coupons and principal and their times in years from settlement;
      prices: per the flows' face; accrued: one per bond, added to clean prices, or None (prices dirty)
    - method: 'sequential' (node_times and weights None; the maturities distinct) or 'least_squares' (node_times
      increasing, the last at or past every flow; weights one per bond, or 'equal')
    - interpolation: 'linear_zero', 'log_linear_discount' or 'cubic_log_discount', as bootstrap_par_curve
    - compounding: the zero rates' own: 'continuous', 'simple' or periods a year (2 semi-annual spot rates)
    returns: a dict: node_times; discount_factors and zero_rates (in the named compounding) at them; fitted_prices
      (dirty, one per bond); residuals (fitted less market, dirty)
    source: Tuckman and Serrat (2012), Fixed Income Securities, ch. 2 and 21; Hagan and West (2006)
    aliases: bootstrap from bond prices, coupon bond bootstrap, zero curve from bond prices, discount factors from bond prices, bond stripping, yield curve fitting, least-squares yield curve, real yield curve, [zero-coupon curve], [spot curve]
    example:
        qf.bootstrap_bond_prices(cash_flows=[[102.0], [2.5, 102.5], [3.0, 3.0, 3.0, 103.0]], times=[[0.5], [0.5, 1.0], [0.5, 1.0, 1.5, 2.0]],
            prices=[100.9, 101.2, 103.4], accrued=None, method="sequential", node_times=None, weights=None,
            interpolation="log_linear_discount", compounding="continuous")
    """
    choice("method", method, BOND_METHODS)
    choice("interpolation", interpolation, INTERPOLATIONS)
    rule = _zero_compounding(compounding)
    if len(cash_flows) != len(times) or len(cash_flows) == 0:
        raise ValueError("cash_flows and times must hold one array per bond")
    bonds = [_flows(cf, t) for cf, t in zip(cash_flows, times)]
    if any(np.any(t <= 0) for _, t in bonds):
        raise ValueError("every cash flow's time must be after settlement (positive)")
    dirty = np.asarray(prices, dtype=float).ravel()
    if dirty.size != len(bonds) or not np.all(np.isfinite(dirty)):
        raise ValueError("prices must hold one finite price per bond")
    if accrued is not None:
        extra = np.asarray(accrued, dtype=float).ravel()
        if extra.shape != dirty.shape:
            raise ValueError("accrued must hold one amount per bond, or be None for dirty prices")
        dirty = dirty + extra
    ends = np.array([t.max() for _, t in bonds])
    if method == "sequential":
        if node_times is not None or weights is not None:
            raise ValueError("node_times and weights are for 'least_squares': pass None")
        nodes = np.sort(ends)
        if np.any(np.diff(nodes) <= 0):
            raise ValueError("'sequential' needs one bond per maturity; bonds sharing one want 'least_squares'")
        order = np.argsort(ends)
    else:
        nodes = _increasing("node_times", node_times)
        if np.any(ends > nodes[-1] * (1 + 1e-12)):
            raise ValueError("node_times must reach every bond's last flow (no extrapolation)")
        w = np.ones(len(bonds)) if isinstance(weights, str) and weights == "equal" else positive("weights", weights)
        if w.shape != dirty.shape:
            raise ValueError("weights must be 'equal' or one positive weight per bond")
    unit = np.eye(nodes.size)

    def reader(t):
        """D at the times t as a function of the nodes' zero rates z under the compounding."""
        if interpolation == "linear_zero":
            linear = np.column_stack([np.interp(t, nodes, unit[j]) for j in range(nodes.size)])
            return lambda z: _zero_discounts(linear @ z, t, rule)[0]
        logs = _interpolation_weights(t, nodes, interpolation)
        return lambda z: np.exp(logs @ np.log(_zero_discounts(z, nodes, rule)[0]))

    design = [(cf, reader(t)) for cf, t in bonds]

    def priced(z):
        with np.errstate(invalid="ignore", over="ignore"):
            return np.array([cf @ read(z) for cf, read in design])

    if method == "sequential":
        z = np.full(nodes.size, 0.03)
        for _ in range(200):
            for k, i in enumerate(order):
                cf, read = design[i]
                low = -1.0 if rule == "continuous" else -0.999 / nodes[-1] if rule == "simple" else -0.999 * rule

                def gap(zero, k=k, cf=cf, read=read):
                    trial = z.copy()
                    trial[k] = zero
                    return cf @ read(trial) - dirty[i]

                try:
                    z[k] = brentq(gap, max(low, -1.0), 1.0, xtol=1e-16, rtol=4.5e-15)
                except ValueError as failure:
                    raise ValueError(f"prices: no zero rate reprices the bond maturing at {nodes[k]:g} ({failure})")
            if np.max(np.abs(priced(z) - dirty) / np.abs(dirty)) <= 1e-13:
                break
        else:
            raise ValueError("prices: the cubic bootstrap did not settle in 200 passes")
    else:
        root = np.sqrt(w)
        fit = least_squares(lambda x: root * (priced(x) - dirty), np.full(nodes.size, 0.03), method="trf",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=10000)
        z = fit.x
    fitted = priced(z)
    return Result({"node_times": nodes, "discount_factors": _zero_discounts(z, nodes, rule)[0], "zero_rates": z,
                   "fitted_prices": fitted, "residuals": fitted - dirty})


@_guard(time=('times', 'node_times',))
def curve_discount(times, node_times, node_discounts, interpolation):
    """Discount factors at any maturities off a curve's nodes, the interpolation named (zero rate -ln D / t).

    -ln D(t) is read from the nodes (and D(0) = 1) exactly as bootstrap_par_curve reads it: 'linear_zero' linear in
    the zero rate -ln D / t (the first node's rate below it), 'log_linear_discount' linear in -ln D, and
    'cubic_log_discount' a not-a-knot cubic spline on -ln D through 0. Times beyond the last node are refused (no
    extrapolation).

    A commodity swap's fixed price is the discount-weighted mean of the forwards on its fixing dates, sum_i D(t_i)
    F(t_i) / sum_i D(t_i). The roll yield is the futures curve's slope, ln(F_near / F_far) / (T_far - T_near) a year
    (positive in backwardation).

    conventions:
    - node_times, node_discounts: the curve's nodes, e.g. bootstrap_par_curve's maturities and discount_factors
    - interpolation: the curve's own ('linear_zero', 'log_linear_discount', 'cubic_log_discount')
    - times: years from today, 0 to the last node; the zero rate at t is -ln D(t) / t (continuously compounded)
    returns: discount factors at the times: a float, or an array for arrays
    source: Hagan and West (2006), "Interpolation methods for curve construction", Applied Math. Finance 13(2)
    aliases: [discount curve], interpolate discount factors, zero curve interpolation, log-linear discount, [curve lookup]
    example:
        qf.curve_discount(times=[0.25, 1.5, 4.0], node_times=[0.5, 1.0, 2.0, 5.0],
                          node_discounts=[0.98, 0.96, 0.92, 0.80],
                          interpolation="log_linear_discount")
    """
    choice("interpolation", interpolation, INTERPOLATIONS)
    nodes = _increasing("node_times", node_times)
    g = -np.log(positive("node_discounts", node_discounts).ravel())
    if g.shape != nodes.shape:
        raise ValueError("node_discounts must hold one discount factor per node time")
    t = np.asarray(times, dtype=float)
    if np.any(t < 0) or np.any(t > nodes[-1] * (1 + 1e-12)):
        raise ValueError("times must lie between 0 and the last node (no extrapolation)")
    return out(np.exp(-(_interpolation_weights(t.ravel(), nodes, interpolation) @ g)).reshape(t.shape))

@_guard(time=('times', 'node_times',))
def curve_discount_extrapolated(times, node_times, node_discounts, interpolation, extrapolation):
    """Discount factors off a curve's nodes, the extrapolation past the last node named: flat or linear.

    Up to the last node t_N, curve_discount's (the interpolation named; below the first node 'linear_zero' and
    'log_linear_discount' hold its zero rate, the cubic runs from D(0) = 1). Past t_N, with g = -ln D and z = g / t:
    'flat_zero' holds the last zero rate, D(t) = exp(-z_N t); 'flat_forward' holds the last segment's forward, g(t) =
    g_N + f (t - t_N), f = (g_N - g_{N-1}) / (t_N - t_{N-1}) (g_1 / t_1 for one node); 'linear_zero' extends the zero
    rate's line through the last two nodes (flat for one node); 'refuse' refuses such a time, as curve_discount does.

    conventions:
    - extrapolation: 'flat_zero', 'flat_forward', 'linear_zero' or 'refuse', past the last node only
    - interpolation: the curve's own inside ('linear_zero', 'log_linear_discount' or 'cubic_log_discount')
    - node_times, node_discounts: the curve's nodes; times: years from today, 0 or more
    returns: discount factors at the times: a float, or an array for arrays
    source: Hagan and West (2006), "Interpolation methods for curve construction", Applied Math. Finance 13(2)
    aliases: curve extrapolation, flat zero-rate extrapolation, flat forward extrapolation, zero rate extrapolation, extrapolate discount factors, [extrapolation]
    example:
        # Returns [0.980199, 0.95123]: past 2 years the last zero rate, 1%, is held.
        qf.curve_discount_extrapolated(times=[2.0, 5.0], node_times=[1.0, 2.0], node_discounts=[0.99104, 0.980199], interpolation="log_linear_discount", extrapolation="flat_zero")
    """
    choice("interpolation", interpolation, INTERPOLATIONS)
    choice("extrapolation", extrapolation, EXTRAPOLATIONS)
    nodes = _increasing("node_times", node_times)
    g = -np.log(positive("node_discounts", node_discounts).ravel())
    if g.shape != nodes.shape:
        raise ValueError("node_discounts must hold one discount factor per node time")
    t = np.asarray(times, dtype=float)
    flat = t.ravel()
    if not np.all(np.isfinite(flat)) or np.any(flat < 0):
        raise ValueError("times must be finite years from today, 0 or more")
    beyond = flat > nodes[-1] * (1 + 1e-12)
    if beyond.any() and extrapolation == "refuse":
        raise ValueError("times past the last node are refused under extrapolation 'refuse'")
    found = np.empty(flat.size)
    found[~beyond] = _interpolation_weights(flat[~beyond], nodes, interpolation) @ g
    if beyond.any():
        late, last, g_last = flat[beyond], nodes[-1], g[-1]
        if extrapolation == "flat_zero":
            found[beyond] = g_last / last * late
        elif extrapolation == "flat_forward":
            forward = (g_last - g[-2]) / (last - nodes[-2]) if nodes.size > 1 else g_last / last
            found[beyond] = g_last + forward * (late - last)
        else:
            zeros = g / nodes
            slope = (zeros[-1] - zeros[-2]) / (last - nodes[-2]) if nodes.size > 1 else 0.0
            found[beyond] = (zeros[-1] + slope * (late - last)) * late
    return out(np.exp(-found).reshape(t.shape))


def _instrument(entry, kinds, where, lag):
    """One quote as (kind, fields, the times it reads, its node): the fields checked and made floats or arrays."""
    if not isinstance(entry, dict) or "kind" not in entry:
        raise ValueError(f"each of {where} must be a dict with a 'kind'")
    kind = choice(f"an instrument's kind in {where}", entry["kind"], kinds)
    fields = INSTRUMENT_FIELDS[kind]
    unknown, missing = sorted(set(entry) - set(fields) - {"kind"}), [f for f in fields if f not in entry]
    if unknown or missing:
        raise ValueError(f"a {kind!r} in {where} takes kind, {', '.join(fields)}: missing {missing}, unknown {unknown}")
    found = {name: (np.asarray(entry[name], dtype=float).ravel() if name.endswith(("times", "accruals"))
                    else float(entry[name])) for name in fields}
    start = found["start"]
    if kind in ("deposit", "fra", "future"):
        if not (0 <= start < found["end"]) or not found["accrual"] > 0:
            raise ValueError(f"a {kind!r} in {where} needs 0 <= start < end and a positive accrual")
        return kind, found, found["end"]
    ends = found["times" if kind == "ois" else "fixed_times"]
    accruals = found["accruals" if kind == "ois" else "fixed_accruals"]
    legs = [ends] if kind == "ois" else [ends, found["float_times"]]
    if (ends.shape != accruals.shape or np.any(~(accruals > 0)) or start < 0
            or any(leg.size == 0 or np.any(np.diff(np.r_[start, leg]) <= 0) for leg in legs)):
        raise ValueError(f"a {kind!r} in {where} needs increasing times after its start, one positive accrual each")
    return kind, found, (ends[-1] + lag if kind == "ois" else max(ends[-1], found["float_times"][-1]))


def _quote_gap(kind, found, own, discount, lag):
    """The instrument's value per unit notional at its quote under the curves `own` and `discount` (callables of
    times giving discount factors); zero when the curves reprice it."""
    if kind in ("deposit", "fra", "future"):
        rate = found["rate"] if kind != "future" else (100.0 - found["price"]) / 100.0 - found["convexity"]
        start, end = own(np.array([found["start"], found["end"]]))
        return start / (end * (1.0 + rate * found["accrual"])) - 1.0
    if kind == "ois":
        ends = found["times"]
        bounds, paid = own(np.r_[found["start"], ends]), own(ends + lag)
        return found["rate"] * found["accruals"] @ paid - (bounds[:-1] / bounds[1:] - 1.0) @ paid
    projected = own(np.r_[found["start"], found["float_times"]])
    return (found["rate"] * found["fixed_accruals"] @ discount(found["fixed_times"])
            - (projected[:-1] / projected[1:] - 1.0) @ discount(found["float_times"]))


def _bootstrap(quotes, interpolation, discount, lag, where):
    """The nodes and -ln D at them that reprice every quote, each node solved in order on its zero rate, the pass
    repeated until all reprice (one pass for a local interpolation)."""
    quotes = sorted(quotes, key=lambda quote: quote[2])
    nodes = np.array([quote[2] for quote in quotes])
    if np.any(np.diff(nodes) <= 0):
        raise ValueError(f"two instruments in {where} end on the same date: one node each")
    guess = [quote[1].get("rate", (100.0 - quote[1].get("price", 97.0)) / 100.0) for quote in quotes]
    g = np.asarray(guess, dtype=float) * nodes
    reads = {}

    def curve(values):
        def read(t):
            key = tuple(np.round(t, 14))
            if key not in reads:
                reads[key] = _interpolation_weights(t, nodes, interpolation)
            return np.exp(-(reads[key] @ values))
        return read

    def gap(k, values):
        own = curve(values)
        return _quote_gap(quotes[k][0], quotes[k][1], own, discount or own, lag)

    for _ in range(200):
        for k in range(nodes.size):
            def solve(zero, k=k):
                trial = g.copy()
                trial[k] = zero * nodes[k]
                return gap(k, trial)
            try:
                g[k] = brentq(solve, -1.0, 1.0, xtol=1e-16, rtol=4.5e-15) * nodes[k]
            except ValueError as failure:
                raise ValueError(f"{where}: no zero rate reprices the instrument ending at {nodes[k]} ({failure})")
        if max(abs(gap(k, g)) for k in range(nodes.size)) <= REPRICED:
            return nodes, g
    raise ValueError(f"{where}: the cubic bootstrap did not settle in 200 passes")


def bootstrap_multicurve(ois_instruments, projection_instruments, ois_payment_lag, interpolation):
    """Multi-curve bootstrap: OIS discounting and a projection curve.

    The discount curve D from ois_instruments: an OIS swaps K tau_i for D(T_{i-1}) / D(T_i) - 1 (the compounded
    overnight rate), both paid at T_i + lag; a deposit D(start) / D(end) = 1 + K tau. Then the projection curve P: a
    deposit or FRA P(start) / P(end) = 1 + K tau; a future the same at K = (100 - price)/100 - convexity; a swap
    K sum_i tau_i D(T_i) = sum_j (P(t_{j-1}) / P(t_j) - 1) D(t_j), projected off P, discounted off D. Each
    instrument's last payment is a node, solved in order until all price to zero (a joint solve gives the same
    curves). ois_instruments None: one curve projects and discounts (the single-curve bootstrap).

    conventions:
    - ois_instruments: dicts {"kind": "ois", "rate", "start", "times", "accruals"} (periods' ends) or
      {"kind": "deposit", "rate", "start", "end", "accrual"}; or None
    - projection_instruments: dicts of kind "deposit" or "fra" (rate, start, end, accrual), "future" (price, start,
      end, accrual, convexity) or "swap" (rate, start, fixed_times, fixed_accruals, float_times: the tenor's ends)
    - times in years from today; accruals each leg's year fractions by its day count; rates decimal, simple
    - ois_payment_lag: years from an OIS period's end to its payment (0.0 for none)
    - interpolation: 'linear_zero', 'log_linear_discount' or 'cubic_log_discount', both curves, as curve_discount's
    returns: a dict: discount_times, discount_factors (D at its nodes); projection_times, projection_factors (P's
      pseudo-discount factors); read either with qf.curve_discount
    source: Ametrano and Bianchetti (2013), multiple curve bootstrapping; Henrard (2014), the multi-curve framework
    aliases: multi-curve, multi-curve bootstrap, dual curve, dual-curve bootstrap, OIS discounting, OIS curve, projection curve, forwarding curve, [OIS], [basis]
    example:
        qf.bootstrap_multicurve(ois_instruments=[{"kind": "ois", "rate": 0.03, "start": 0.0, "times": [1.0, 2.0],
            "accruals": [1.0, 1.0]}], projection_instruments=[{"kind": "fra", "rate": 0.032, "start": 0.0, "end": 1.0,
            "accrual": 1.0}, {"kind": "swap", "rate": 0.034, "start": 0.0, "fixed_times": [1.0, 2.0], "fixed_accruals": [1.0, 1.0], "float_times": [1.0, 2.0]}], ois_payment_lag=0.0, interpolation="linear_zero")
    """
    choice("interpolation", interpolation, INTERPOLATIONS)
    lag = float(ois_payment_lag)
    if not lag >= 0:
        raise ValueError("ois_payment_lag must not be negative")
    if not isinstance(projection_instruments, (list, tuple)) or not projection_instruments:
        raise ValueError("projection_instruments must be a list of instrument dicts")
    discount = None
    if ois_instruments is not None:
        if not isinstance(ois_instruments, (list, tuple)) or not ois_instruments:
            raise ValueError("ois_instruments must be a list of instrument dicts, or None for one curve")
        quotes = [_instrument(entry, DISCOUNT_KINDS, "ois_instruments", lag) for entry in ois_instruments]
        d_nodes, d_g = _bootstrap(quotes, interpolation, None, lag, "ois_instruments")

        def discount(t):
            if np.any(t > d_nodes[-1] * (1 + 1e-12)):
                raise ValueError("the OIS curve must reach every projection instrument's payment (no extrapolation)")
            return np.exp(-(_interpolation_weights(t, d_nodes, interpolation) @ d_g))
    quotes = [_instrument(entry, PROJECTION_KINDS, "projection_instruments", lag) for entry in projection_instruments]
    p_nodes, p_g = _bootstrap(quotes, interpolation, discount, lag, "projection_instruments")
    if discount is None:
        d_nodes, d_g = p_nodes, p_g
    return Result({"discount_times": d_nodes, "discount_factors": np.exp(-d_g), "projection_times": p_nodes,
                   "projection_factors": np.exp(-p_g)})


def _curve(name, curve, interpolation):
    """A curve given as (node_times, node_discounts), as a reader of discount factors at times from 0 to its last
    node, by the interpolation named (curve_discount's)."""
    try:
        times, factors = curve
    except (TypeError, ValueError):
        times = factors = None
    if np.ndim(times) != 1 or np.ndim(factors) != 1:
        raise ValueError(f"{name} must be a pair (node_times, node_discounts) of equal-length sequences")
    nodes = _increasing(f"{name}'s node_times", times)
    g = -np.log(positive(f"{name}'s node_discounts", factors).ravel())
    if g.shape != nodes.shape:
        raise ValueError(f"{name} must hold one discount factor per node time")

    def read(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if np.any(t < 0) or np.any(t > nodes[-1] * (1 + 1e-12)):
            raise ValueError(f"{name}: every time must lie between 0 and its last node (no extrapolation)")
        return np.exp(-(_interpolation_weights(t, nodes, interpolation) @ g))
    return read


def _ns(t, params, variant, decay_form):
    """The Nelson-Siegel or Svensson yield at times t, without argument checks."""
    betas, decays = VARIANTS[variant]
    yields = params[0] + np.zeros_like(t)
    for j in range(decays):
        x = t * params[betas + j] if decay_form == "rate" else t / params[betas + j]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(x > 0, -np.expm1(-x) / x, 1.0)
        if j == 0:
            yields = yields + params[1] * slope
        yields = yields + params[2 + j] * (slope - np.exp(-x))
    return yields


def _ns_params(name, variant, params):
    p = np.asarray(params, dtype=float).ravel()
    betas, decays = VARIANTS[variant]
    if p.size != betas + decays or not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must be {betas + decays} finite numbers for {variant}")
    return p


def nelson_siegel_yield(variant, maturities, params, decay_form):
    """Nelson-Siegel or Svensson yield curve at the maturities from its parameters.

    x = lambda T ('rate') or T / tau ('time'); L1 = (1 - e^{-x}) / x, L2 = L1 - e^{-x} (L1 = 1, L2 = 0 at T = 0);
    nelson_siegel: y(T) = b0 + b1 L1(x1) + b2 L2(x1), params [b0, b1, b2, d1];
    svensson:      y(T) = b0 + b1 L1(x1) + b2 L2(x1) + b3 L2(x2), params [b0, b1, b2, b3, d1, d2].
    y is the average of the instantaneous forward b0 + b1 e^{-x} + b2 x e^{-x} (+ b3 x2 e^{-x2}) from 0 to T.

    conventions:
    - decay_form: 'rate' if the decay multiplies T (Diebold and Li's lambda), 'time' if it divides T (Svensson's tau)
    - params: in the order above, decays positive; yields on the basis fitted (continuous zero rates in the papers)
    - variant: 'nelson_siegel' (4 parameters) or 'svensson' (6)
    returns: the yields at the maturities, in params' units (decimals for decimal betas)
    source: Nelson and Siegel (1987), J. Business 60(4); Svensson (1994), NBER WP 4871; Diebold and Li (2006)
    aliases: Nelson-Siegel, Svensson, Nelson-Siegel-Svensson, NSS, NS curve, parametric yield curve, Diebold-Li
    example:
        qf.nelson_siegel_yield(variant="svensson", maturities=[1.0, 2.0, 5.0, 10.0],
                               params=[0.045, -0.01, 0.005, -0.004, 1.7, 6.0],
                               decay_form="time")
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of 'nelson_siegel', 'svensson', not {variant!r}")
    choice("decay_form", decay_form, DECAY_FORMS)
    p = _ns_params("params", variant, params)
    if np.any(p[VARIANTS[variant][0]:] <= 0):
        raise ValueError("params: the decays must be positive")
    t = np.asarray(maturities, dtype=float)
    if np.any(t < 0):
        raise ValueError("maturities must not be negative")
    return out(_ns(t, p, variant, decay_form))


@_guard(time=('maturities',), curve=(('maturities', 'yields'),))
def nelson_siegel_fit(variant, maturities, yields, initial, decay_form, method, options, fixed_decays):
    """Least-squares Nelson-Siegel or Svensson fit: the parameters and the fitted yields, the decays fitted or given.

    Minimizes SSE = sum_i (y(T_i; params) - y_i)^2: decays fitted by scipy.optimize.minimize from `initial` (the start
    picks the minimum; L-BFGS-B on decimals wants tiny ftol, gtol), or fixed (Diebold-Li) and the betas exact by OLS.

    conventions:
    - yields: the data's own scale, percent or decimals: the betas and rmse come back in it (betas in percent want
      percent yields); annual rates, -ln(D)/t, not rates times their times
    - fixed_decays: None (fitted) or the decay(s) held fixed, in decay_form's form; then initial, method, options None
    - initial: the start in nelson_siegel_yield's order; method: 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'TNC' or
      'SLSQP'; options: scipy's dict ({} for defaults); decay_form: 'rate' or 'time', as nelson_siegel_yield
    returns: a dict: params (the model's order, fixed decays included); fitted (the yields at the maturities); rmse
      (sqrt(SSE / n)); sse; success (the optimizer's flag; True for fixed decays)
    source: Nelson and Siegel (1987), J. Business 60(4); Svensson (1994); Diebold and Li (2006), J. Econometrics 130
    aliases: Nelson-Siegel fit, Nelson-Siegel curve, Svensson fit, [fit yield curve], NSS fit, Diebold-Li fit, [curve fitting], [term structure fit]
    example:
        qf.nelson_siegel_fit(variant="nelson_siegel", maturities=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
                             yields=[0.040, 0.041, 0.042, 0.043, 0.044, 0.045], initial=[0.045, -0.005, 0.0, 1.0],
                             decay_form="time", method="L-BFGS-B", options={"ftol": 0.0, "gtol": 1e-12}, fixed_decays=None)
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of 'nelson_siegel', 'svensson', not {variant!r}")
    choice("decay_form", decay_form, DECAY_FORMS)
    t = np.asarray(maturities, dtype=float).ravel()
    y = np.asarray(yields, dtype=float).ravel()
    if y.shape != t.shape or not (np.all(np.isfinite(y)) and np.all(t >= 0)):
        raise ValueError("maturities and yields must be finite, one yield per maturity")
    betas, decays = VARIANTS[variant]
    if fixed_decays is not None:
        if initial is not None or method is not None or options is not None:
            raise ValueError("with fixed_decays the betas are solved exactly: initial, method and options are None")
        held = np.asarray(fixed_decays, dtype=float).ravel()
        if held.size != decays or np.any(~(held > 0)):
            raise ValueError(f"fixed_decays must be {decays} positive decay(s) for {variant}")
        basis = np.column_stack([_ns(t, np.r_[np.eye(betas)[j], held], variant, decay_form) for j in range(betas)])
        found = np.linalg.lstsq(basis, y, rcond=None)[0]
        params = np.r_[found, held]
        fitted = _ns(t, params, variant, decay_form)
        return Result({"params": params, "fitted": fitted, "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
                       "sse": float(np.sum((fitted - y) ** 2)), "success": True})
    choice("method", method, FIT_METHODS)
    start = _ns_params("initial", variant, initial)
    if np.any(start[betas:] <= 0):
        raise ValueError("initial: the decays must be positive")
    if not isinstance(options, dict):
        raise ValueError("options must be a dict of scipy.optimize.minimize options ({} for its defaults)")

    def sse(params):
        return float(np.sum((_ns(t, params, variant, decay_form) - y) ** 2))

    found = minimize(sse, start, method=method, bounds=[(None, None)] * betas + [(DECAY_FLOOR, None)] * decays,
                     options=dict(options))
    fitted = _ns(t, found.x, variant, decay_form)
    return Result({"params": found.x, "fitted": fitted, "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
                   "sse": float(found.fun), "success": bool(found.success)})


# ------------------------------------------------------------------------------------------------ bonds


def _whole_frequency(frequency):
    if not isinstance(frequency, (int, np.integer)) or isinstance(frequency, bool) or frequency < 1:
        raise ValueError(f"frequency must be a whole number of periods a year (1, 2, 4, 12), not {frequency!r}")
    return int(frequency)


def _flows(cash_flows, times):
    cf, t = np.asarray(cash_flows, dtype=float).ravel(), np.asarray(times, dtype=float).ravel()
    if cf.shape != t.shape or cf.size == 0 or not (np.all(np.isfinite(cf)) and np.all(t >= 0)):
        raise ValueError("cash_flows and times must be finite, one time (years, not negative) per cash flow")
    return cf, t


def _zero_compounding(compounding):
    """'continuous', 'simple' or a whole number of periods a year."""
    if compounding in ("continuous", "simple"):
        return compounding
    if not isinstance(compounding, (int, np.integer)) or isinstance(compounding, bool) or compounding < 1:
        raise ValueError("compounding must be 'continuous', 'simple' or a whole number of periods a year (1, 2, 12), "
                         f"not {compounding!r}")
    return int(compounding)


def _zero_discounts(z, t, compounding):
    """D(t) from zero rates z under the compounding, with dD/dz and d2D/dz2 (the parallel shift's derivatives)."""
    if compounding == "continuous":
        d = np.exp(-z * t)
        return d, -t * d, t * t * d
    if compounding == "simple":
        d = 1.0 / (1.0 + z * t)
        return d, -t * d * d, 2.0 * t * t * d ** 3
    base = 1.0 + z / compounding
    if np.any(~(base > 0)):
        raise ValueError(f"zero rates must exceed -{compounding} for compounding {compounding}")
    d = base ** (-compounding * t)
    return d, -t * d / base, t * (t + 1.0 / compounding) * d / base ** 2


@_guard(rate=('zero_start', 'zero_end',), time=('start', 'end',), curve=(('start', 'zero_start'), ('end', 'zero_end'),))
def forward_rates(start, end, zero_start, zero_end, compounding):
    """Forward rate between two maturities from their zero rates, the compounding named.

    The rate for [t1, t2] that D(t2) = D(t1) F(t1, t2) implies, quoted in the zero rates' own compounding:
    continuous (z2 t2 - z1 t1) / (t2 - t1); k a year k [((1 + z2/k)^{k t2} / (1 + z1/k)^{k t1})^{1/(k (t2 - t1))} - 1];
    simple [(1 + z2 t2) / (1 + z1 t1) - 1] / (t2 - t1). At t1 = 0 it is z2. Arrays broadcast: a strip is one call.

    An FRA's fair rate is the simple forward over its accrual tau, (D(t1) / D(t2) - 1) / tau; long at K it pays
    N tau (F - K) at t2, worth N tau (F - K) D(t2) today (settled at t1, that over 1 + F tau).

    conventions:
    - start, end: t1 < t2 in years from today (t1 = 0 allowed)
    - zero_start, zero_end: the zero rates to t1 and to t2 in the named compounding: annual rates, -ln(D)/t, not rates
      times their times
    - compounding: 'continuous', 'simple' or periods a year (1, 2, 4, 12), of the zero rates and of the forward
    returns: the forward rate as a decimal: a float, or an array for arrays
    source: Hull, Options, Futures, and Other Derivatives, "Forward rates"; Tuckman and Serrat (2011), ch. 2
    aliases: forward rate, forward rates, forward-forward rate, forward rates from zero rates
    example:
        qf.forward_rates(start=[1.0, 2.0, 3.0], end=[2.0, 3.0, 4.0], zero_start=[0.03, 0.04, 0.046],
                         zero_end=[0.04, 0.046, 0.05], compounding="continuous")
    """
    rule = _zero_compounding(compounding)
    t1, t2, z1, z2 = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (start, end, zero_start, zero_end)))
    if np.any(t1 < 0) or np.any(~(t2 > t1)):
        raise ValueError("start must be at least 0 and end after it")
    ratio = _zero_discounts(z2, t2, rule)[0] / _zero_discounts(z1, t1, rule)[0]
    span = t2 - t1
    if rule == "continuous":
        return out(-np.log(ratio) / span)
    if rule == "simple":
        return out((1.0 / ratio - 1.0) / span)
    return out(rule * np.expm1(-np.log(ratio) / (rule * span)))
