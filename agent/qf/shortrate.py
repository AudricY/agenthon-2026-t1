"""Hull-White, Ho-Lee, Vasicek, CIR, CIR++, G2++ and BDT models: zero-coupon bonds, swaptions, caplets, trees, OAS.

`import qf`. Times are year fractions from today under the caller's day count (cash-flow times, maturities,
fixings); zero rates and short rates are continuously compounded unless a function names another basis; discount
factors are P(0, t). A scalar result is a float; a function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from scipy.optimize import brentq, least_squares
from scipy.special import roots_hermite

from ._common import Result, choice, integer, ndtr, out, positive
from .curves import INTERPOLATIONS, _curve, _increasing, _zero_compounding, curve_discount
from .options import black76_price
from .ou import _cir_loglik
from .primitives import _guard, _stream

__all__ = ["cir_fit", "cir_pp_bond_price", "g2pp_swaption", "g2pp_zcb", "hull_white_calibrate",
           "hull_white_caplet", "hull_white_zcb", "hull_white_zcb_option", "jamshidian_swaption", "short_rate_bond_price",
           "short_rate_tree", "vasicek_fit_curve"]

HW_KINDS = {"caplet": ("option", "strike", "fixing", "payment", "accrual", "vol"),
            "swaption": ("option", "strike", "expiry", "payment_times", "accruals", "vol")}
HW_LOSSES = ("price", "vol")
BLACK_HIGH = 5.0                # a Black vol is sought in (0, this]; a price with none counts 1.0 off in vol
SIDES = ("payer", "receiver")
TREE_MODELS = ("hull_white", "ho_lee", "black_karasinski", "bdt")
DISCRETIZATIONS = ("exact", "euler")
BRANCHINGS = ("hull_white", "nearest")
TREE_COMPOUNDINGS = ("continuous", "simple")
TREE_EXERCISES = ("none", "callable", "puttable", "call", "put")
EXERCISE_FLOWS = ("after", "from")
JMAX_FACTOR = 0.184             # Hull and White (1994): jmax the smallest integer above 0.184 / (a dt)
ON_STEP = 1e-9                  # a time within this many steps of a step is on it
QUADRATURES = ("gauss_hermite", "adaptive")
G2_SPAN = 12.0                  # 'adaptive' integrates x over its mean plus or minus this many of its deviations


# ------------------------------------------------------------------------------------------------ Hull-White


def _hw(a, sigma):
    return float(positive("a", a)), float(positive("sigma", sigma))


def _b(a, tau):
    return -np.expm1(-a * tau) / a


@_guard(rate=('short_rate', 'forward_t',), time=('t', 'maturity',))
def hull_white_zcb(short_rate, t, maturity, a, sigma, discount_t, discount_maturity, forward_t):
    """Hull-White zero-coupon bond price P(t, T) at short rate r(t), fitted to today's curve.

    P(t, T) = A(t, T) e^{-B(t, T) r(t)},  B = (1 - e^{-a (T - t)}) / a,
    A = P(0, T) / P(0, t) exp(B f(0, t) - sigma^2 / (4a) (1 - e^{-2at}) B^2),
    for dr = (theta(t) - a r) dt + sigma dW with theta fitted to the curve; at t = 0, r = f(0, 0) it is P(0, T).

    theta(t) = df(0, t)/dt + a f(0, t) + sigma^2 / (2a) (1 - e^{-2at}); a bond's vol is sigma B(t, T), the term
    structure of vols; under the T-forward measure the drift is less sigma^2 B(t, T). Ho-Lee is a -> 0 (B = T - t).

    conventions:
    - discount_t, discount_maturity: today's P(0, t) and P(0, T); P(0, 0) = 1
    - forward_t: the instantaneous forward f(0, t) = -d ln P(0, t)/dt, continuously compounded (f(0, 0) = r(0))
    - a: mean reversion, sigma: the short rate's absolute (normal) vol, both annual and positive; t, maturity in years
    returns: a float, the bond price P(t, T) per unit face
    source: Hull and White (1990), Review of Financial Studies 3(4); Brigo and Mercurio, Interest Rate Models, 3.3
    aliases: Hull-White, Hull White, HW zero-coupon bond, affine bond price, extended Vasicek, HW1F, A(t,T) B(t,T)
    example:
        qf.hull_white_zcb(short_rate=0.048, t=1.0, maturity=4.0, a=0.06, sigma=0.0125,
                          discount_t=0.9512, discount_maturity=0.8187,
                          forward_t=0.05)
    """
    a, sigma = _hw(a, sigma)
    t, big_t = np.asarray(t, dtype=float), np.asarray(maturity, dtype=float)
    if np.any(t < 0) or np.any(big_t < t):
        raise ValueError("t and maturity must satisfy 0 <= t <= maturity")
    p_t, p_big = positive("discount_t", discount_t), positive("discount_maturity", discount_maturity)
    b = _b(a, big_t - t)
    log_a = (np.log(p_big / p_t) + b * np.asarray(forward_t, dtype=float)
             - sigma ** 2 / (4.0 * a) * -np.expm1(-2.0 * a * t) * b * b)
    return out(np.exp(log_a - b * np.asarray(short_rate, dtype=float)))


@_guard(time=('expiry', 'maturity',), price=('strike',))
def hull_white_zcb_option(option, expiry, maturity, strike, a, sigma, discount_expiry, discount_maturity):
    """Hull-White price of a European option expiring at T on the zero-coupon bond maturing at S.

    sigma_p = sigma sqrt((1 - e^{-2aT}) / (2a)) B(T, S),  h = ln(P(0, S) / (P(0, T) X)) / sigma_p + sigma_p / 2,
    call ZBC = P(0, S) N(h) - X P(0, T) N(h - sigma_p),  put ZBP = X P(0, T) N(sigma_p - h) - P(0, S) N(-h).

    conventions:
    - strike: X, the bond's price at expiry per 1 of face; expiry T < maturity S in years
    - discount_expiry, discount_maturity: today's P(0, T) and P(0, S); a, sigma as hull_white_zcb
    returns: a float per unit face of the bond
    source: Jamshidian (1989), J. Finance 44(1); Brigo and Mercurio, Interest Rate Models (2006), (3.40)-(3.41)
    aliases: zero-coupon bond option, ZBC, ZBP, bond option Hull-White, discount bond option, HW bond option
    example:
        qf.hull_white_zcb_option(option="put", expiry=1.0, maturity=2.0, strike=0.95, a=0.06, sigma=0.0125,
                                 discount_expiry=0.9512,
                                 discount_maturity=0.9048)
    """
    choice("option", option, ("call", "put"))
    a, sigma = _hw(a, sigma)
    big_t, s = positive("expiry", expiry), np.asarray(maturity, dtype=float)
    if np.any(s <= big_t):
        raise ValueError("maturity must come after expiry")
    x = positive("strike", strike)
    p_t, p_s = positive("discount_expiry", discount_expiry), positive("discount_maturity", discount_maturity)
    vol = sigma * np.sqrt(-np.expm1(-2.0 * a * big_t) / (2.0 * a)) * _b(a, s - big_t)
    h = np.log(p_s / (p_t * x)) / vol + 0.5 * vol
    if option == "call":
        return out(p_s * ndtr(h) - x * p_t * ndtr(h - vol))
    return out(x * p_t * ndtr(vol - h) - p_s * ndtr(-h))


@_guard(rate=('strike',), time=('fixing', 'payment',))
def hull_white_caplet(option, strike, fixing, payment, accrual, a, sigma, discount_fixing, discount_payment):
    """Hull-White caplet or floorlet per unit notional, as (1 + K tau) zero-coupon bond options.

    caplet = (1 + K tau) ZBP(0, T1, T2, X), floorlet = (1 + K tau) ZBC(0, T1, T2, X), X = 1 / (1 + K tau): the
    payoff tau (L - K)^+ at T2 on the rate L fixed at T1 is worth (1 + K tau)(X - P(T1, T2))^+ at T1.

    conventions:
    - fixing: T1, the rate's reset and the option's expiry; payment: T2, when it pays (both years from today)
    - accrual: tau, the period's year fraction under the rate's day count; strike: K, simply compounded like L
    - discount_fixing, discount_payment: P(0, T1), P(0, T2); a, sigma as hull_white_zcb
    returns: a float per unit notional
    source: Hull and White (1990); Brigo and Mercurio, Interest Rate Models (2006), 3.3.2; Hull, ch. 32
    aliases: Hull-White caplet, HW caplet, caplet Hull-White, floorlet, cap pricing Hull-White, caplet as bond put
    example:
        qf.hull_white_caplet(option="caplet", strike=0.05, fixing=1.0, payment=1.25, accrual=0.25, a=0.06,
                             sigma=0.0125, discount_fixing=0.9512,
                             discount_payment=0.9394)
    """
    choice("option", option, ("caplet", "floorlet"))
    k, tau = np.asarray(strike, dtype=float), positive("accrual", accrual)
    face = 1.0 + k * tau
    if np.any(face <= 0):
        raise ValueError("strike must keep 1 + strike * accrual positive")
    bond = hull_white_zcb_option("put" if option == "caplet" else "call", fixing, payment, 1.0 / face, a, sigma,
                                 discount_fixing, discount_payment)
    return out(face * np.asarray(bond))


@_guard(rate=('strike', 'forward_expiry',), time=('expiry', 'payment_times',))
def jamshidian_swaption(option, strike, expiry, payment_times, accruals, a, sigma, discount_expiry, discount_payments,
                        forward_expiry):
    """Hull-White European swaption price per unit notional by Jamshidian's decomposition.

    The fixed leg with the notional is a coupon bond, c_i = K tau_i, plus 1 at T_n. r* solves sum_i c_i P(T0, T_i, r*)
    = 1 (hull_white_zcb at t = T0); X_i = P(T0, T_i, r*); payer = sum_i c_i ZBP(0, T0, T_i, X_i), receiver = sum_i
    c_i ZBC(0, T0, T_i, X_i).

    conventions:
    - option: 'payer' (pay fixed: a put on the coupon bond) or 'receiver' (a call); strike: the fixed rate K
    - payment_times: the fixed leg's T_i after expiry T0; accruals: tau_i under the fixed leg's day count
    - discount_expiry, discount_payments: P(0, T0), P(0, T_i); forward_expiry: f(0, T0), as hull_white_zcb's forward_t
    returns: a dict: price (per unit notional); r_star (the short rate at expiry that prices the fixed
      leg at par); strikes (the X_i, one per payment)
    source: Jamshidian (1989), "An exact bond option formula", J. Finance 44(1); Brigo and Mercurio (2006), 3.11.1
    aliases: Jamshidian, Jamshidian decomposition, Jamshidian trick, swaption Hull-White, HW swaption, [European swaption]
    example:
        qf.jamshidian_swaption(option="payer", strike=0.05, expiry=1.0, payment_times=[2.0, 3.0, 4.0],
                               accruals=[1.0, 1.0, 1.0], a=0.06, sigma=0.0125, discount_expiry=0.9512,
                               discount_payments=[0.9048, 0.8607, 0.8187], forward_expiry=0.05)
    """
    choice("option", option, ("payer", "receiver"))
    big_t = float(positive("expiry", expiry))
    times = _increasing("payment_times", payment_times)
    tau = positive("accruals", accruals).ravel()
    p_i = positive("discount_payments", discount_payments).ravel()
    if tau.shape != times.shape or p_i.shape != times.shape:
        raise ValueError("payment_times, accruals and discount_payments must have the same length")
    if times[0] <= big_t:
        raise ValueError("payment_times must come after expiry")
    coupons = float(strike) * tau
    coupons[-1] += 1.0
    if np.any(coupons <= 0):
        raise ValueError("strike must keep every coupon positive")

    def bond(rate):
        return float(coupons @ np.asarray(hull_white_zcb(rate, big_t, times, a, sigma, discount_expiry, p_i,
                                                         forward_expiry))) - 1.0

    low, high = -1.0, 1.0
    while bond(low) * bond(high) > 0 and high < 64.0:
        low, high = 2.0 * low, 2.0 * high
    r_star = brentq(bond, low, high, xtol=1e-16, rtol=4.5e-15)
    strikes = np.asarray(hull_white_zcb(r_star, big_t, times, a, sigma, discount_expiry, p_i, forward_expiry))
    legs = hull_white_zcb_option("put" if option == "payer" else "call", big_t, times, strikes, a, sigma,
                                 discount_expiry, p_i)
    return Result({"price": float(coupons @ np.asarray(legs)), "r_star": float(r_star), "strikes": strikes})


def _steps(name, times, dt):
    """Each time's step number, refused where a time is not a whole number of dt after today."""
    steps = np.asarray(times if times is not None else [], dtype=float).ravel() / dt
    whole = np.rint(steps)
    if (steps.size == 0 or not np.all(np.isfinite(steps)) or np.any(whole < 1)
            or np.any(np.abs(steps - whole) > ON_STEP * np.maximum(whole, 1.0))):
        raise ValueError(f"{name} must lie on the tree's steps: whole multiples of dt after today")
    return whole.astype(int)


def _lattice(model, a, sigma, dt, steps, discretization, branching):
    """The tree's state x on nodes j dx: each step's lowest and highest node (spans, steps + 1 of them) and, for each
    step's nodes, the node its middle branch reaches and the down, middle and up probabilities (moves)."""
    if model == "bdt":            # ln r +- sigma sqrt(dt) with 1/2 each: the nodes of one parity a step
        half = [np.full(2 * i + 1, 0.5) for i in range(steps)]
        return (sigma * np.sqrt(dt), [(-i, i) for i in range(steps + 1)],
                [(np.arange(-i, i + 1), p, np.zeros(p.size), p) for i, p in enumerate(half)])
    if a > 0 and discretization == "exact":
        m, v = np.expm1(-a * dt), sigma ** 2 * -np.expm1(-2.0 * a * dt) / (2.0 * a)
    else:
        m, v = -a * dt, sigma ** 2 * dt
    top = int(np.floor(JMAX_FACTOR / (a * dt))) + 1 if a > 0 else None
    spans, moves = [(0, 0)], []
    for _ in range(steps):
        j = np.arange(spans[-1][0], spans[-1][1] + 1)
        if branching == "nearest":
            k = np.floor(j * (1.0 + m) + 0.5).astype(int)
        else:
            k = j if top is None else np.where(j >= top, j - 1, np.where(j <= -top, j + 1, j))
        e = j * (1.0 + m) - k                                 # the mean's distance from the middle branch, in dx
        moves.append((k, (1.0 + 3.0 * e * e - 3.0 * e) / 6.0, (2.0 - 3.0 * e * e) / 3.0,
                      (1.0 + 3.0 * e * e + 3.0 * e) / 6.0))
        spans.append((int(k.min()) - 1, int(k.max()) + 1))
    return np.sqrt(3.0 * v), spans, moves


def _node_discounts(model, alpha, x, spread, dt, compounding):
    rate = np.exp(alpha + x) if model in ("black_karasinski", "bdt") else alpha + x
    with np.errstate(over="ignore"):
        return np.exp(-(rate + spread) * dt) if compounding == "continuous" else 1.0 / (1.0 + (rate + spread) * dt)


def _fit(model, dx, spans, moves, dt, zeros, compounding):
    """alpha at each step, so that the Arrow-Debreu prices Q reprice each zero-coupon bond: sum_j Q_ij d(alpha_i + x_j)
    = P(0, t_{i+1}), Q carried forward through the branches."""
    q, alphas = np.array([1.0]), []
    normal = model in ("hull_white", "ho_lee")
    for i, (k, down, middle, up) in enumerate(moves):
        x = np.arange(spans[i][0], spans[i][1] + 1) * dx

        def gap(alpha):
            return q @ _node_discounts(model, alpha, x, 0.0, dt, compounding) / zeros[i] - 1.0

        if normal and compounding == "continuous":
            alpha = (np.log(q @ np.exp(-x * dt)) - np.log(zeros[i])) / dt
        else:
            lo, hi = (-1.0, 1.0) if normal else (-30.0, 3.0)
            if gap(lo) * gap(hi) > 0:
                raise ValueError(f"no shift at step {i} reprices the curve's discount factor at {(i + 1) * dt:g} "
                                 "(a lognormal model needs positive forward rates)")
            alpha = brentq(gap, lo, hi, xtol=1e-15, rtol=4.5e-15)
        alphas.append(alpha)
        carried = q * _node_discounts(model, alpha, x, 0.0, dt, compounding)
        q = np.zeros(spans[i + 1][1] - spans[i + 1][0] + 1)
        for shift, weight in ((-1, down), (0, middle), (1, up)):
            np.add.at(q, k + shift - spans[i + 1][0], carried * weight)
    return np.array(alphas)


@_guard(time=('flow_times', 'exercise_times',), price=('market_price',))
def short_rate_tree(model, mean_reversion, sigma, dt, curve, interpolation, discretization, branching, compounding,
                    flow_times, flows, exercise, exercise_times, strikes, exercise_flows, oas, market_price):
    """Short-rate trees fitted to the curve: callable bonds, Bermudans, OAS.

    Hull-White r = alpha_i + x, Black-Karasinski ln r = alpha_i + x, x a trinomial OU (reversion a, dx = sqrt(3 V),
    V its step variance); Ho-Lee that at a = 0; Black-Derman-Toy ln r +- sigma sqrt(dt), 1/2 each. alpha_i makes the
    Arrow-Debreu prices reprice each zero: sum_j Q_ij d_ij = P(0, t_{i+1}). Backward, d_ij = e^{-(r + s) dt} or
    1 / (1 + (r + s) dt), s the OAS; on an exercise date a callable is min(value, strike), a puttable max, a call
    max(continuation, flows - strike), a put max(continuation, strike - flows). A Bermudan payer swaption is a 'put'
    on the fixed leg's coupon bond struck at 1.

    conventions:
    - model: 'hull_white', 'ho_lee', 'black_karasinski' or 'bdt'; mean_reversion: a (0.0 for 'ho_lee', 'bdt');
      sigma: of r, or of ln r ('black_karasinski', 'bdt')
    - dt: the step (flow and exercise times whole steps); curve: (node_times, node_discounts), by interpolation
    - discretization: 'exact' (the OU's step moments) or 'euler' (Hull's -a dt, sigma^2 dt); branching:
      'hull_white' (edges at jmax, the least integer above 0.184 / (a dt)) or 'nearest'; both unused for 'bdt'
    - compounding: a node's rate over its step: 'continuous' or 'simple'
    - exercise: 'none', 'callable', 'puttable', 'call' or 'put'; exercise_times, strikes: None for 'none'
    - exercise_flows: 'after' (a flow on the date is paid anyway) or 'from' (the exercise takes it)
    - oas: s, or None to solve it from market_price (else None)
    returns: a dict: price; straight (no exercise); oas; shifts (alpha_i, a step each)
    source: Hull and White (1994); Black, Derman and Toy (1990); Hull, "Interest rate trees"
    aliases: rate tree, Hull-White tree, Black-Derman-Toy, Black-Karasinski, Ho-Lee, Arrow-Debreu, callable bond, puttable bond, Bermudan swaption, option-adjusted spread, OAS, BDT tree
    example:
        qf.short_rate_tree(model="hull_white", mean_reversion=0.1, sigma=0.01, dt=0.5, curve=([1.0, 5.0], [0.96, 0.8]),
            interpolation="linear_zero", discretization="exact", branching="hull_white", compounding="continuous", flow_times=[1.0, 2.0],
            flows=[5.0, 105.0], exercise="callable", exercise_times=[1.0], strikes=[100.0], exercise_flows="after", oas=0.0, market_price=None)
    """
    choice("model", model, TREE_MODELS)
    choice("interpolation", interpolation, INTERPOLATIONS)
    choice("discretization", discretization, DISCRETIZATIONS)
    choice("branching", branching, BRANCHINGS)
    choice("compounding", compounding, TREE_COMPOUNDINGS)
    choice("exercise", exercise, TREE_EXERCISES)
    choice("exercise_flows", exercise_flows, EXERCISE_FLOWS)
    a, vol, step = float(mean_reversion), float(positive("sigma", sigma)), float(positive("dt", dt))
    if (model in ("ho_lee", "bdt") and a != 0) or (model in ("hull_white", "black_karasinski") and not a > 0):
        raise ValueError("mean_reversion must be 0.0 for 'ho_lee' and 'bdt' and positive for the others")
    paid = _steps("flow_times", flow_times, step)
    amounts = np.asarray(flows, dtype=float).ravel()
    if amounts.shape != paid.shape or not np.all(np.isfinite(amounts)):
        raise ValueError("flows must hold one finite amount per flow time")
    if exercise == "none":
        if exercise_times is not None or strikes is not None:
            raise ValueError("exercise_times and strikes are None with exercise 'none'")
        dates, prices = np.array([], dtype=int), np.array([])
    else:
        dates = _steps("exercise_times", exercise_times, step)
        if np.any(np.diff(dates) <= 0):
            raise ValueError("exercise_times must be increasing")
        prices = np.asarray(strikes, dtype=float).ravel()
        if prices.shape != dates.shape or not np.all(np.isfinite(prices)):
            raise ValueError("strikes must hold one price per exercise time")
    if (oas is None) == (market_price is None):
        raise ValueError("give oas (the spread) or market_price (to solve it), the other None")
    n = int(max(paid.max(), dates.max(initial=0)))
    zeros = _curve("curve", curve, interpolation)(step * np.arange(1, n + 1))
    dx, spans, moves = _lattice(model, a, vol, step, n, discretization, branching)
    alphas = _fit(model, dx, spans, moves, step, zeros, compounding)
    cash = np.zeros(n + 1)
    np.add.at(cash, paid, amounts)
    struck = dict(zip(dates.tolist(), prices.tolist()))
    whole = exercise_flows == "from"

    def value(spread):
        """(price, straight) at the spread: backward from the last step, each date's flow and exercise applied."""
        size = spans[n][1] - spans[n][0] + 1
        bond, held, option = np.zeros(size), np.zeros(size), np.zeros(size)
        for i in range(n, 0, -1):
            flow, strike = cash[i], struck.get(i)
            if strike is None or exercise == "none":
                held = held + flow
            elif exercise in ("callable", "puttable"):
                bound = np.minimum if exercise == "callable" else np.maximum
                held = bound(held + flow, strike) if whole else bound(held, strike) + flow
            else:
                under = bond + flow if whole else bond
                option = np.maximum(option, under - strike if exercise == "call" else strike - under)
            bond = bond + flow
            k, down, middle, up = moves[i - 1]
            low = spans[i][0]
            d = _node_discounts(model, alphas[i - 1], np.arange(spans[i - 1][0], spans[i - 1][1] + 1) * dx, spread,
                                step, compounding)
            bond, held, option = (d * (down * v[k - 1 - low] + middle * v[k - low] + up * v[k + 1 - low])
                                  for v in (bond, held, option))
        return float((option if exercise in ("call", "put") else held)[0]), float(bond[0])

    if market_price is None:
        spread = float(oas)
    else:
        target, lo, hi = float(market_price), -0.05, 0.05
        while (value(lo)[0] - target) * (value(hi)[0] - target) > 0 and hi < 1.0:
            lo, hi = 2.0 * lo, 2.0 * hi
        try:
            spread = brentq(lambda s: value(s)[0] - target, lo, hi, xtol=1e-15, rtol=4.5e-15)
        except ValueError:
            raise ValueError("market_price: no spread within +-1 reprices it") from None
    price, straight = value(spread)
    return Result({"price": price, "straight": straight, "oas": float(spread), "shifts": alphas})


def _hw_instrument(entry, curve):
    """(market price from its Black vol, a callable of a Black vol, the Hull-White price at a and sigma)."""
    if entry["kind"] == "caplet":
        t1, t2, tau = float(entry["fixing"]), float(entry["payment"]), float(entry["accrual"])
        d1, d2 = curve(t1), curve(t2)
        forward = (d1 / d2 - 1.0) / tau
        option = "call" if entry["option"] == "caplet" else "put"

        def black(vol):
            return tau * float(black76_price(option, forward, entry["strike"], t1, vol, d2))

        return black, lambda x, y: float(hull_white_caplet(entry["option"], entry["strike"], t1, t2, tau, x, y, d1, d2))
    t0 = float(entry["expiry"])
    times, tau = _increasing("payment_times", entry["payment_times"]), positive("accruals", entry["accruals"]).ravel()
    d0, dis = curve(t0), np.array([curve(t) for t in times])
    annuity = float(tau @ dis)
    rate = (d0 - dis[-1]) / annuity
    option = "call" if entry["option"] == "payer" else "put"
    step = 1e-4 * t0
    forward = np.log(curve(t0 - step) / curve(t0)) / step

    def black(vol):
        return float(black76_price(option, rate, entry["strike"], t0, vol, annuity))

    return black, lambda x, y: jamshidian_swaption(entry["option"], entry["strike"], t0, times, tau, x, y, d0, dis,
                                                    forward)["price"]


@_guard(time=('curve_times',))
def hull_white_calibrate(instruments, curve_times, curve_discounts, interpolation, loss, weights, initial, bounds):
    """Hull-White calibration of (a, sigma) to caplets or swaptions quoted in Black vols.

    Each instrument's market price is Black-76's at its quoted vol on the curve (a caplet tau P(0, T2) Black(F, K,
    T1), F = (P(T1)/P(T2) - 1)/tau; a swaption A Black(S, K, T0), A = sum tau_i P(0, T_i), S = (P(T0) - P(T_n))/A);
    its model price qf.hull_white_caplet or qf.jamshidian_swaption on the same curve. Least squares
    (scipy.optimize.least_squares, 'trf', within the bounds) over sum_i w_i (model_i - market_i)^2, in price, or in
    Black vol for 'vol' (the model price's implied vol, solved). Prices made from a known (a, sigma) return them.

    conventions:
    - instruments: dicts, {'kind': 'caplet', 'option': 'caplet' or 'floorlet', 'strike', 'fixing', 'payment',
      'accrual', 'vol'} or {'kind': 'swaption', 'option': 'payer' or 'receiver', 'strike', 'expiry',
      'payment_times', 'accruals', 'vol'}; times in years, strikes and vols decimals (Black's lognormal vol)
    - curve_times, curve_discounts, interpolation: the single curve, as qf.curve_discount reads it
    - loss: 'price' or 'vol'; weights: one per instrument, or 'equal'
    - initial: {'a': ..., 'sigma': ...}; bounds: {'a': (low, high), 'sigma': (low, high)}
    returns: a dict: a; sigma; rmse (unweighted, in the loss's units); converged (the optimiser's flag);
      market_prices and model_prices (per unit notional, one per instrument)
    source: Brigo and Mercurio (2006), Interest Rate Models, 3.3; Hull, "Interest rate derivatives: short-rate models"
    aliases: Hull-White calibration, calibrate Hull-White, HW calibration, calibrate to swaptions, calibrate to caplets, short-rate model calibration, mean reversion and volatility calibration
    example:
        qf.hull_white_calibrate(instruments=[{"kind": "caplet", "option": "caplet", "strike": 0.04, "fixing": 1.0, "payment": 1.5, "accrual": 0.5, "vol": 0.25},
            {"kind": "caplet", "option": "caplet", "strike": 0.04, "fixing": 3.0, "payment": 3.5, "accrual": 0.5, "vol": 0.21}], curve_times=[1.0, 5.0],
            curve_discounts=[0.96, 0.82], interpolation="linear_zero", loss="price", weights="equal", initial={"a": 0.05, "sigma": 0.01}, bounds={"a": (1e-4, 1.0), "sigma": (1e-4, 0.1)})
    """
    choice("loss", loss, HW_LOSSES)
    choice("interpolation", interpolation, INTERPOLATIONS)
    nodes, values = _increasing("curve_times", curve_times), positive("curve_discounts", curve_discounts).ravel()
    if values.shape != nodes.shape:
        raise ValueError("curve_discounts must hold one discount factor per curve time")

    def curve(t):
        return float(curve_discount(t, nodes, values, interpolation))

    if not instruments:
        raise ValueError("instruments must hold at least one caplet or swaption")
    for entry in instruments:
        fields = HW_KINDS.get(entry.get("kind")) if isinstance(entry, dict) else None
        if fields is None or set(entry) != {"kind", *fields}:
            raise ValueError("each instrument must be a dict of 'kind' ('caplet' or 'swaption') and exactly its fields: "
                             + "; ".join(f"{k}: {', '.join(v)}" for k, v in HW_KINDS.items()))
        choice("option", entry["option"], ("caplet", "floorlet") if entry["kind"] == "caplet" else SIDES)
    built = [_hw_instrument(entry, curve) for entry in instruments]
    quoted = np.array([float(positive("vol", entry["vol"])) for entry in instruments])
    market = np.array([black(vol) for (black, _), vol in zip(built, quoted)])
    n = len(instruments)
    w = np.ones(n) if isinstance(weights, str) and weights == "equal" else positive("weights", weights).ravel()
    if w.size != n:
        raise ValueError("weights must be 'equal' or one positive weight per instrument")
    if not (isinstance(initial, dict) and set(initial) == {"a", "sigma"} and isinstance(bounds, dict)
            and set(bounds) == {"a", "sigma"}):
        raise ValueError("initial and bounds must be dicts of exactly 'a' and 'sigma'")
    low = np.array([float(bounds["a"][0]), float(bounds["sigma"][0])])
    high = np.array([float(bounds["a"][1]), float(bounds["sigma"][1])])
    start = np.array([float(initial["a"]), float(initial["sigma"])])
    if not (np.all(low > 0) and np.all(low < high) and np.all(start >= low) and np.all(start <= high)):
        raise ValueError("bounds must be positive (low, high) with low < high, and initial inside them")

    def implied(black, price):
        try:
            return brentq(lambda vol: black(vol) - price, 1e-8, BLACK_HIGH, xtol=1e-15, rtol=1e-15)
        except ValueError:
            return np.nan

    def residuals(x):
        prices = np.array([model(x[0], x[1]) for _, model in built])
        if loss == "price":
            return prices - market
        vols = np.array([implied(black, price) for (black, _), price in zip(built, prices)])
        return np.where(np.isfinite(vols), vols - quoted, 1.0)

    root = np.sqrt(w)
    fit = least_squares(lambda x: root * residuals(x), start, bounds=(low, high), method="trf", xtol=1e-15,
                        ftol=1e-15, gtol=1e-15, max_nfev=2000)
    return Result({"a": float(fit.x[0]), "sigma": float(fit.x[1]),
                   "rmse": float(np.sqrt(np.mean(residuals(fit.x) ** 2))), "converged": bool(fit.success),
                   "market_prices": market, "model_prices": np.array([model(*fit.x) for _, model in built])})

SHORT_RATE_MODELS = ("vasicek", "cir")
BOND_OPTIONS = ("none", "call", "put")
CIR_METHODS = ("mle_exact", "ols_euler")


def _affine(model, tau, kappa, theta, sigma):
    """A(tau) and B(tau) of P = A e^{-B r} under Vasicek or CIR."""
    if model == "vasicek":
        b = -np.expm1(-kappa * tau) / kappa
        log_a = (theta - sigma * sigma / (2 * kappa * kappa)) * (b - tau) - sigma * sigma * b * b / (4 * kappa)
        return np.exp(log_a), b
    h = np.sqrt(kappa * kappa + 2.0 * sigma * sigma)
    gap = 2.0 * sigma * sigma / (h + kappa)        # h - kappa, free of cancellation as sigma falls
    grow = np.expm1(h * tau)
    # ln[2h e^{(kappa+h)tau/2} / (2h + (kappa+h)grow)], each part of order sigma^2
    log_ratio = np.log1p(gap / (h + kappa)) - 0.5 * gap * tau - np.log1p(gap * np.exp(-h * tau) / (h + kappa))
    return np.exp(2.0 * kappa * theta / (sigma * sigma) * log_ratio), 2.0 * grow / (2.0 * h + (kappa + h) * grow)


@_guard(rate=('short_rate',), time=('maturity', 'expiry',), price=('strike',))
def short_rate_bond_price(model, option, short_rate, maturity, kappa, theta, sigma, expiry, strike):
    """Vasicek or CIR (Cox-Ingersoll-Ross) zero-coupon bond price, or an option on that bond.

    P(t, t + tau) = A(tau) e^{-B(tau) r}. Vasicek (dr = kappa (theta - r) dt + sigma dW): B = (1 - e^{-kappa tau}) /
    kappa, A = exp[(theta - sigma^2/(2 kappa^2))(B - tau) - sigma^2 B^2/(4 kappa)]. CIR (dr = kappa (theta - r) dt +
    sigma sqrt(r) dW; r stays positive where 2 kappa theta >= sigma^2, Feller): h = sqrt(kappa^2 + 2 sigma^2), B =
    2(e^{h tau} - 1)/(2h + (kappa + h)(e^{h tau} - 1)), A = [2h e^{(kappa + h) tau/2}/(2h + (kappa + h)(e^{h tau} -
    1))]^{2 kappa theta/sigma^2}. The yield is -ln(P)/tau. An option expiring at T on the bond maturing at S:
    Vasicek's Jamshidian formula, P(0,S) N(h) - K P(0,T) N(h - s_p), s_p = sigma B(S - T) sqrt((1 - e^{-2 kappa
    T})/(2 kappa)); CIR's by the noncentral chi-square law of r_T. As sigma goes to 0 both give exp(-theta tau - (r -
    theta)(1 - e^{-kappa tau})/kappa).

    The parameters are risk-neutral; a fit on history (qf.cir_fit) gives real-world ones. With a market price of risk
    lambda, dW^Q = dW^P + lambda dt (Vasicek) or lambda sqrt(r) dt (CIR): Vasicek's theta is theta - lambda sigma /
    kappa, CIR's kappa + lambda sigma and kappa theta / (kappa + lambda sigma).

    conventions:
    - model: 'vasicek' or 'cir'; option: 'none' (the bond), 'call' or 'put' (on the bond)
    - short_rate: r today, a decimal, continuously compounded; maturity: the bond's tau (S for an option), years
    - kappa: the mean-reversion speed; theta: the long-run level; sigma: the short rate's vol (CIR: of sqrt(r) dW)
    - expiry: the option's T < maturity, and strike: K per unit face; both None for 'none'
    returns: a dict: price; with 'none' zero_rate (the yield -ln(P)/tau, not -ln(P)), a and b (A, B); with an option
      bond (P(0, S)) and discount (P(0, T))
    source: Vasicek (1977), J. Financial Economics 5; Cox, Ingersoll and Ross (1985), Econometrica 53(2); Jamshidian
      (1989), J. Finance 44(1); Brigo and Mercurio, Interest Rate Models (2006), 3.2.1 and 3.2.3
    aliases: Vasicek bond price, Vasicek model, CIR bond price, CIR model, Cox-Ingersoll-Ross, CIR zero-coupon bond, CIR bond option, Vasicek bond option, [Feller condition]
    example:
        qf.short_rate_bond_price(model="cir", option="none", short_rate=0.03, maturity=5.0, kappa=0.3,
                                 theta=0.04, sigma=0.08, expiry=None,
                                 strike=None)
    """
    choice("model", model, SHORT_RATE_MODELS)
    choice("option", option, BOND_OPTIONS)
    r, s = float(short_rate), float(positive("maturity", maturity))
    kappa, theta, sigma = float(positive("kappa", kappa)), float(theta), float(positive("sigma", sigma))
    if model == "cir" and (r < 0 or theta <= 0):
        raise ValueError("CIR needs short_rate >= 0 and theta > 0")
    a, b = _affine(model, s, kappa, theta, sigma)
    bond = float(a * np.exp(-b * r))
    if option == "none":
        if expiry is not None or strike is not None:
            raise ValueError("expiry and strike are None for the bond itself (option 'none')")
        return Result({"price": bond, "zero_rate": float(-np.log(bond) / s), "a": float(a), "b": float(b)})
    t, k = float(positive("expiry", expiry)), float(positive("strike", strike))
    if not t < s:
        raise ValueError("expiry must come before the bond's maturity")
    a_t, b_t = _affine(model, t, kappa, theta, sigma)
    discount = float(a_t * np.exp(-b_t * r))
    if model == "vasicek":
        _, b_ts = _affine(model, s - t, kappa, theta, sigma)
        sd = sigma * b_ts * np.sqrt(-np.expm1(-2.0 * kappa * t) / (2.0 * kappa))
        h = np.log(bond / (discount * k)) / sd + 0.5 * sd
        call = bond * ndtr(h) - k * discount * ndtr(h - sd)
    else:
        a_ts, b_ts = _affine(model, s - t, kappa, theta, sigma)
        h = np.sqrt(kappa * kappa + 2.0 * sigma * sigma)
        rho = 2.0 * h / (sigma * sigma * np.expm1(h * t))
        psi = (kappa + h) / (sigma * sigma)
        cut = np.log(a_ts / k) / b_ts           # r_T below it puts the option in the money (none past K = A)
        df = 4.0 * kappa * theta / (sigma * sigma)
        centre = 2.0 * rho * rho * r * np.exp(h * t)
        call = (bond * stats.ncx2.cdf(2.0 * cut * (rho + psi + b_ts), df, centre / (rho + psi + b_ts))
                - k * discount * stats.ncx2.cdf(2.0 * cut * (rho + psi), df, centre / (rho + psi)))
    price = call if option == "call" else call - bond + k * discount
    return Result({"price": float(price), "bond": bond, "discount": discount})


@_guard(rate=('rates',))
def cir_fit(rates, dt, method, bounds, starts, rng):
    """CIR short-rate fit: exact MLE (noncentral chi-square) or Euler least squares; at given values, cir_log_likelihood.

    'mle_exact': with c = 2 kappa / (sigma^2 (1 - e^{-kappa dt})), 2c r_{t+dt} given r_t is noncentral chi-square of
    4 kappa theta / sigma^2 degrees of freedom and noncentrality 2c r_t e^{-kappa dt}; the fit maximises the sum of
    ln(2c) + ln f_ncx2(2c r_{t+dt}) over consecutive pairs by L-BFGS-B within the bounds from each random start, the
    highest kept. 'ols_euler': the Euler step r_{t+dt} - r_t = kappa (theta - r_t) dt + sigma sqrt(r_t dt) e,
    divided by sqrt(r_t), is a regression of (r_{t+dt} - r_t)/sqrt(r_t) on 1/sqrt(r_t) and sqrt(r_t) without
    intercept: slopes b1 = kappa theta dt and b2 = -kappa dt, so kappa = -b2/dt, theta = -b1/b2 and sigma the
    residuals' std (sum of squares over n - 2) over sqrt(dt); its kappa tends to (1 - e^{-kappa dt})/dt on exact
    data, the exact one as dt falls. log_likelihood is the exact transitions' at the estimates either way;
    feller_ratio = 2 kappa theta / sigma^2 (at least 1: r stays positive). Prices: qf.short_rate_bond_price.

    conventions:
    - rates: the short rate series, decimals (percent / 100), oldest first, every one positive
    - dt: the spacing in years (1/252 for business days, 1/365 for calendar days)
    - method: 'mle_exact' or 'ols_euler'
    - bounds: a dict of (low, high) for 'kappa', 'theta', 'sigma', each above 0 ('mle_exact'; None for 'ols_euler');
      a point (low equal to high) is refused: at given values the likelihood is cir_log_likelihood
    - starts: the random starts, an int of at least 1, uniform within the bounds, drawn kappa, theta, sigma
    - rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream) ('mle_exact'; starts and rng
      None for 'ols_euler')
    returns: a dict: kappa; theta; sigma; log_likelihood; feller_ratio; converged
    source: Cox, Ingersoll and Ross (1985), Econometrica 53(2); Brigo and Mercurio, Interest Rate Models (2006),
      3.2.3; Kladivko (2007)
    aliases: CIR calibration, CIR fit, CIR maximum likelihood, CIR MLE, noncentral chi-square likelihood, CIR transition density, CIR OLS, square-root process fit
    example:
        qf.cir_fit(rates=[0.031, 0.0312, 0.0309, 0.0315, 0.0318, 0.0314, 0.0311, 0.0316, 0.0320, 0.0317, 0.0322],
                   dt=1 / 252, method="mle_exact", bounds={"kappa": (0.05, 3.0), "theta": (0.005, 0.15),
                   "sigma": (0.005, 0.4)}, starts=3, rng=1)
    """
    choice("method", method, CIR_METHODS)
    x = np.asarray(rates, dtype=float).ravel()
    if x.size < 4 or np.any(~(x > 0)):
        raise ValueError("rates must hold four or more positive decimals (percent / 100)")
    step = float(positive("dt", dt))
    now, nxt = x[:-1], x[1:]

    def loglik(kappa, theta, sigma):
        return _cir_loglik(now, nxt, step, kappa, theta, sigma)

    if method == "ols_euler":
        if bounds is not None or starts is not None or rng is not None:
            raise ValueError("'ols_euler' takes bounds, starts and rng None")
        root = np.sqrt(now)
        design = np.column_stack([1.0 / root, root])
        slopes, *_ = np.linalg.lstsq(design, (nxt - now) / root, rcond=None)
        kappa = -slopes[1] / step
        if not kappa > 0:
            raise ValueError("the Euler regression gives no mean reversion (kappa <= 0): the series is not CIR-like")
        theta = -slopes[0] / slopes[1]
        residuals = (nxt - now) / root - design @ slopes
        sigma = float(np.sqrt(residuals @ residuals / (residuals.size - 2) / step))
        found = loglik(kappa, theta, sigma) if theta > 0 else float("nan")
        return Result({"kappa": float(kappa), "theta": float(theta), "sigma": sigma, "log_likelihood": found,
                       "feller_ratio": 2.0 * kappa * theta / (sigma * sigma), "converged": True})
    n_starts = integer("starts", starts, 1)
    if not isinstance(bounds, dict) or set(bounds) != {"kappa", "theta", "sigma"}:
        raise ValueError("bounds must be a dict of (low, high) for exactly kappa, theta, sigma")
    box = [tuple(float(v) for v in bounds[name]) for name in ("kappa", "theta", "sigma")]
    fixed = [name for name, (low, high) in zip(("kappa", "theta", "sigma"), box) if low == high]
    if fixed:
        raise ValueError(f"bounds fix {', '.join(fixed)} at a point (low equal to high): cir_fit fits within (low, "
                         "high); the CIR log-likelihood at given values is qf.cir_log_likelihood(rates, dt, kappa, "
                         "theta, sigma)")
    if any(not 0 < low < high for low, high in box):
        raise ValueError("each bound must be (low, high) with 0 < low < high")
    rng = _stream(rng)

    def loss(theta_):
        total = loglik(*theta_)
        return -total if np.isfinite(total) else 1e300

    best = None
    for _ in range(n_starts):
        found = optimize.minimize(loss, np.array([rng.uniform(low, high) for low, high in box]), method="L-BFGS-B",
                                  bounds=box)
        if best is None or found.fun < best.fun:
            best = found
    kappa, theta, sigma = (float(v) for v in best.x)
    return Result({"kappa": kappa, "theta": theta, "sigma": sigma, "log_likelihood": float(-best.fun),
                   "feller_ratio": 2.0 * kappa * theta / (sigma * sigma), "converged": bool(best.success)})


# ------------------------------------------------------------------------------------------------ G2++


def _g2_params(a, b, sigma, eta, rho, least_vol):
    """(a, b, sigma, eta, rho) checked: reversions positive, vols at least `least_vol` (0 allowed for the bond), |rho|
    at most 1 and below it where both vols are positive."""
    a, b = float(positive("a", a)), float(positive("b", b))
    sigma, eta, rho = float(sigma), float(eta), float(rho)
    if not (sigma >= least_vol and eta >= least_vol and sigma > 0):
        raise ValueError("sigma must be positive and eta " + ("positive" if least_vol > 0 else "0 or more"))
    if not abs(rho) <= 1.0 or (eta > 0 and abs(rho) == 1.0 and least_vol > 0):
        raise ValueError("rho must lie in [-1, 1] (strictly inside for a swaption)")
    return a, b, sigma, eta, rho


def _g2_v(tau, a, b, sigma, eta, rho):
    """V(t, T), tau = T - t: the variance of the integral of x + y over [t, T] (Brigo and Mercurio (4.10))."""
    tau = np.asarray(tau, dtype=float)
    own_a = tau + 2.0 * np.expm1(-a * tau) / a - np.expm1(-2.0 * a * tau) / (2.0 * a)
    own_b = tau + 2.0 * np.expm1(-b * tau) / b - np.expm1(-2.0 * b * tau) / (2.0 * b)
    cross = tau + np.expm1(-a * tau) / a + np.expm1(-b * tau) / b - np.expm1(-(a + b) * tau) / (a + b)
    return sigma ** 2 / a ** 2 * own_a + eta ** 2 / b ** 2 * own_b + 2.0 * rho * sigma * eta / (a * b) * cross


@_guard(time=('t', 'maturity',))
def g2pp_zcb(x, y, t, maturity, a, b, sigma, eta, rho, discount_t, discount_maturity):
    """G2++ (two-factor Hull-White) zero-coupon bond price P(t, T) at the states x(t), y(t), fitted to today's curve.

    r(t) = x(t) + y(t) + phi(t), dx = -a x dt + sigma dW1, dy = -b y dt + eta dW2, dW1 dW2 = rho dt, x(0) = y(0) = 0,
    phi fitted so that P(0, T) is today's curve: P(t, T) = P(0, T) / P(0, t) exp(1/2 [V(t, T) - V(0, T) + V(0, t)]
    - B_a x(t) - B_b y(t)), B_a = (1 - e^{-a (T - t)}) / a, B_b alike in b, V the variance of the integral of x + y.
    At t = 0 it is P(0, T); eta = 0 is Hull-White with r = x + phi.

    The instantaneous forwards' correlation: corr(df(t, T1), df(t, T2)) = (sigma^2 E_a1 E_a2 + eta^2 E_b1 E_b2 + rho
    sigma eta (E_a1 E_b2 + E_b1 E_a2)) / sqrt((sigma^2 E_a1^2 + eta^2 E_b1^2 + 2 rho sigma eta E_a1 E_b1)(the same at
    T2)), E_a1 = e^{-a (T1 - t)}: two factors can give it below one, which one factor cannot.

    conventions:
    - x, y: the two factors at t (0 at t = 0), not the short rate; t <= maturity in years
    - a, b: the factors' mean reversions; sigma, eta: their absolute (normal) vols; rho: their correlation
    - discount_t, discount_maturity: today's P(0, t) and P(0, T) (1.0 for t = 0)
    returns: a float, the bond price per unit face (an array for arrays)
    source: Brigo and Mercurio, Interest Rate Models (2006), 4.2, (4.14) and (4.10); Hull and White (1994)
    aliases: G2++, G2 model, two-factor Hull-White, two-factor Gaussian model, G2++ bond price, two-factor short rate
    example:
        qf.g2pp_zcb(x=0.01, y=-0.005, t=1.0, maturity=5.0, a=0.5, b=0.05, sigma=0.01, eta=0.008, rho=-0.7,
                    discount_t=0.97, discount_maturity=0.85)  # 0.87678226 per unit face
    """
    a, b, sigma, eta, rho = _g2_params(a, b, sigma, eta, rho, 0.0)
    t, big_t = np.asarray(t, dtype=float), np.asarray(maturity, dtype=float)
    if np.any(t < 0) or np.any(big_t < t):
        raise ValueError("t and maturity must satisfy 0 <= t <= maturity")
    p_t, p_big = positive("discount_t", discount_t), positive("discount_maturity", discount_maturity)
    tau = big_t - t
    v = _g2_v(tau, a, b, sigma, eta, rho) - _g2_v(big_t, a, b, sigma, eta, rho) + _g2_v(t, a, b, sigma, eta, rho)
    return out(p_big / p_t * np.exp(0.5 * v + np.expm1(-a * tau) / a * np.asarray(x, dtype=float)
                                    + np.expm1(-b * tau) / b * np.asarray(y, dtype=float)))


@_guard(rate=('strike',), time=('expiry', 'payment_times',))
def g2pp_swaption(option, strike, expiry, payment_times, accruals, a, b, sigma, eta, rho, discount_expiry,
                  discount_payments, quadrature, nodes):
    """G2++ (two-factor Hull-White) European swaption price per unit notional, by Brigo and Mercurio's integral.

    The fixed leg with the notional is a coupon bond, c_i = K tau_i, plus 1 at T_n. Under the T-forward measure x(T)
    is normal (mean mu_x, deviation s_x); given x, the bond sum is 1 at y = y_bar(x), solved; then price = w P(0, T)
    int phi(x) [N(-w h1(x)) - sum_i lambda_i(x) e^{kappa_i(x)} N(-w h2_i(x))] dx, w = 1 payer, -1 receiver, h1 = (y_bar
    - mu_y) / (s_y sqrt(1 - r^2)) - r (x - mu_x) / (s_x sqrt(1 - r^2)), h2_i = h1 + B_b(T, T_i) s_y sqrt(1 - r^2), r
    the factors' correlation at T (Brigo and Mercurio 4.2.5). One payment is a caplet (payer) or floorlet: a zero-bond
    put struck at X is the payer at K = (1/X - 1)/tau, divided by 1 + K tau. eta -> 0 gives Hull-White's Jamshidian
    price.

    conventions:
    - option: 'payer' (pay fixed) or 'receiver'; strike: the fixed rate K, a decimal
    - payment_times: the fixed leg's T_i after expiry T; accruals: tau_i under the fixed leg's day count
    - discount_expiry, discount_payments: today's P(0, T) and P(0, T_i); a, b, sigma, eta, rho as g2pp_zcb
    - quadrature: 'gauss_hermite' (nodes: an int, the Gauss-Hermite nodes in x) or 'adaptive' (scipy's quad over
      the mean plus or minus twelve deviations; nodes None), the surer where eta is tiny (the integrand nears a step)
    returns: a float, the price per unit notional
    source: Brigo and Mercurio, Interest Rate Models (2006), 4.2.5, (4.31); Jamshidian (1989) for eta = 0
    aliases: G2++ swaption, two-factor swaption, G2 swaption, two-factor Hull-White swaption, G2++ caplet
    example:
        qf.g2pp_swaption(option="payer", strike=0.04, expiry=1.0, payment_times=[2.0, 3.0, 4.0],
                         accruals=[1.0, 1.0, 1.0], a=0.5, b=0.05, sigma=0.01, eta=0.008, rho=-0.7,
                         discount_expiry=0.96, discount_payments=[0.92, 0.88, 0.85], quadrature="gauss_hermite",
                         nodes=64)  # 0.00805715 per unit notional
    """
    choice("option", option, SIDES)
    choice("quadrature", quadrature, QUADRATURES)
    a, b, sigma, eta, rho = _g2_params(a, b, sigma, eta, rho, 1e-300)
    big_t = float(positive("expiry", expiry))
    times = _increasing("payment_times", payment_times)
    tau = positive("accruals", accruals).ravel()
    p_i = positive("discount_payments", discount_payments).ravel()
    p_t = float(positive("discount_expiry", discount_expiry))
    if tau.shape != times.shape or p_i.shape != times.shape:
        raise ValueError("payment_times, accruals and discount_payments must have the same length")
    if times[0] <= big_t:
        raise ValueError("payment_times must come after expiry")
    coupons = float(strike) * tau
    coupons[-1] += 1.0
    if np.any(coupons <= 0):
        raise ValueError("strike must keep every coupon positive")
    w = 1.0 if option == "payer" else -1.0
    span = times - big_t
    big_a = p_i / p_t * np.exp(0.5 * (_g2_v(span, a, b, sigma, eta, rho) - _g2_v(times, a, b, sigma, eta, rho)
                                      + _g2_v(big_t, a, b, sigma, eta, rho)))
    b_a, b_b = -np.expm1(-a * span) / a, -np.expm1(-b * span) / b
    ea, eb, eab = -np.expm1(-a * big_t), -np.expm1(-b * big_t), -np.expm1(-(a + b) * big_t)
    cross = rho * sigma * eta
    mu_x = (-(sigma ** 2 / a ** 2 + cross / (a * b)) * ea + sigma ** 2 / (2 * a ** 2) * -np.expm1(-2 * a * big_t)
            + cross / (b * (a + b)) * eab)
    mu_y = (-(eta ** 2 / b ** 2 + cross / (a * b)) * eb + eta ** 2 / (2 * b ** 2) * -np.expm1(-2 * b * big_t)
            + cross / (a * (a + b)) * eab)
    s_x = sigma * np.sqrt(-np.expm1(-2 * a * big_t) / (2 * a))
    s_y = eta * np.sqrt(-np.expm1(-2 * b * big_t) / (2 * b))
    r_xy = rho * sigma * eta / ((a + b) * s_x * s_y) * eab
    root = np.sqrt(1.0 - r_xy * r_xy)

    def bracket(x):
        """y_bar(x): the y at which the coupon bond is worth 1 at T given x(T) = x (the sum falls as y rises)."""
        lam = coupons * big_a * np.exp(-b_a * x)

        def gap(y):
            return float(lam @ np.exp(-b_b * y)) - 1.0
        low, high = mu_y - s_y, mu_y + s_y
        for _ in range(200):
            if gap(low) > 0 > gap(high):
                break
            low, high = low - (high - low), high + (high - low)
        return brentq(gap, low, high, xtol=1e-15, rtol=4.5e-15), lam

    def inner(x):
        y_bar, lam = bracket(x)
        h1 = (y_bar - mu_y) / (s_y * root) - r_xy * (x - mu_x) / (s_x * root)
        h2 = h1 + b_b * s_y * root
        kappa = -b_b * (mu_y - 0.5 * root * root * s_y * s_y * b_b + r_xy * s_y * (x - mu_x) / s_x)
        return float(ndtr(-w * h1) - lam @ (np.exp(kappa) * ndtr(-w * h2)))

    if quadrature == "gauss_hermite":
        count = integer("nodes", nodes, 2)
        u, weights = roots_hermite(count)
        total = float(weights @ np.array([inner(mu_x + np.sqrt(2.0) * s_x * z) for z in u])) / np.sqrt(np.pi)
    else:
        if nodes is not None:
            raise ValueError("nodes is None for 'adaptive'")
        from scipy import integrate
        total = integrate.quad(lambda x: np.exp(-0.5 * ((x - mu_x) / s_x) ** 2) / (s_x * np.sqrt(2 * np.pi)) * inner(x),
                               mu_x - G2_SPAN * s_x, mu_x + G2_SPAN * s_x, epsabs=1e-14, epsrel=1e-12, limit=200)[0]
    return float(w * p_t * total)


# ------------------------------------------------------------------------------------------------ CIR++, Vasicek fit


@_guard(time=('t', 'maturity', 'expiry',), price=('strike',))
def cir_pp_bond_price(option, factor, t, maturity, expiry, strike, x0, kappa, theta, sigma, discount_t,
                      discount_maturity, discount_expiry):
    """CIR++ (CIR with a shift that reprices today's curve) zero-coupon bond price at time t, or a bond option.

    r(t) = x(t) + phi(t), x a CIR process (dx = kappa (theta - x) dt + sigma sqrt(x) dW, x(0) = x0) and phi the
    deterministic shift that makes every bond reprice today's curve P^M: phi(t) = f^M(0, t) - f^CIR(0, t). With
    Phi(t, T) = P^M(0, T) P^CIR(0, t) / (P^M(0, t) P^CIR(0, T)), the bond is P(t, T) = Phi(t, T) P^CIR(t, T; x(t)) and
    an option expiring at T on the bond maturing at S is Phi(t, S) times CIR's option struck at X / Phi(T, S) (a
    shifted strike); CIR's bonds and options are qf.short_rate_bond_price's. A zero shift (a curve that is CIR's own)
    gives CIR exactly. Feller: 2 kappa theta >= sigma^2 keeps x positive.

    conventions:
    - option: 'none' (the bond), 'call' or 'put' (on the bond, European)
    - factor: x(t), the CIR part of the short rate at t (r(t) less phi(t)); x0 at t = 0
    - t: the valuation time (0.0 for today); maturity: the bond's S; expiry: the option's T (t < T < S) and strike: X
      per unit face, both None for 'none'
    - x0, kappa, theta, sigma: CIR's start, mean-reversion speed, long-run level and vol (of sqrt(x) dW)
    - discount_t, discount_maturity, discount_expiry: today's market P^M(0, t), P^M(0, S), P^M(0, T) (discount_expiry
      None for 'none')
    returns: a dict: price; shift_factor (Phi(t, S), the curve's discount over CIR's from t to S); feller_ratio (2
      kappa theta / sigma^2: at least 1 keeps x positive)
    source: Brigo and Mercurio, Interest Rate Models (2006), 3.9, (3.77)-(3.81); Brigo and Mercurio (2001), Finance
      and Stochastics 5
    aliases: CIR++, shifted CIR, extended CIR, CIR fitted to the curve
    example:
        qf.cir_pp_bond_price(option="call", factor=0.03, t=0.0, maturity=5.0, expiry=1.0, strike=0.85, x0=0.03,
                             kappa=0.3, theta=0.04, sigma=0.08, discount_t=1.0, discount_maturity=0.86,
                             discount_expiry=0.97)  # price 0.03668246; shift_factor 1.02187783; feller_ratio 3.75
    """
    choice("option", option, BOND_OPTIONS)
    now, s = float(t), float(positive("maturity", maturity))
    if not 0 <= now < s:
        raise ValueError("t and maturity must satisfy 0 <= t < maturity")
    params = {"kappa": kappa, "theta": theta, "sigma": sigma}

    def cir(r, horizon, which="none", expiry=None, strike=None):
        return short_rate_bond_price("cir", which, r, horizon, expiry=expiry, strike=strike, **params)["price"]

    def model_discount(u):
        return 1.0 if u == 0 else cir(float(x0), u)

    p_t, p_s = float(positive("discount_t", discount_t)), float(positive("discount_maturity", discount_maturity))
    if now == 0 and p_t != 1.0:
        raise ValueError("discount_t is P(0, 0) = 1.0 at t = 0")

    def phi(start, end, d_start, d_end):
        return d_end / d_start * model_discount(start) / model_discount(end)

    shift = phi(now, s, p_t, p_s)
    feller = 2.0 * float(kappa) * float(theta) / float(sigma) ** 2
    if option == "none":
        if expiry is not None or strike is not None or discount_expiry is not None:
            raise ValueError("expiry, strike and discount_expiry are None for the bond itself (option 'none')")
        return Result({"price": shift * cir(float(factor), s - now), "shift_factor": shift, "feller_ratio": feller})
    big_t = float(positive("expiry", expiry))
    if not now < big_t < s:
        raise ValueError("expiry must lie between t and the bond's maturity")
    p_big = float(positive("discount_expiry", discount_expiry))
    shifted = float(positive("strike", strike)) / phi(big_t, s, p_big, p_s)
    price = shift * cir(float(factor), s - now, option, big_t - now, shifted)
    return Result({"price": price, "shift_factor": shift, "feller_ratio": feller})


@_guard(rate=('yields', 'short_rate',), time=('maturities',), curve=(('maturities', 'yields'),))
def vasicek_fit_curve(maturities, yields, compounding, short_rate, weights, initial):
    """Vasicek fitted to a yield curve: kappa, theta, sigma (and r0) by least squares in the model's yield.

    The model's zero-coupon bond is P(tau) = A(tau) e^{-B(tau) r0} (qf.short_rate_bond_price, 'vasicek'); its yield
    under the yields' compounding is -ln(P)/tau (continuous), k (P^{-1/(k tau)} - 1) (k a year) or (1/P - 1)/tau
    (simple). The fit minimises sum_i w_i (model_i - yield_i)^2 (scipy.optimize.least_squares, 'trf', kappa and sigma
    kept positive), r0 fitted or held fixed. The parameters are risk-neutral (a curve fit), not a fit on history; a
    curve made from known parameters returns them.

    conventions:
    - maturities: years, positive; yields: decimals (percent / 100), one per maturity: annual rates, -ln(D)/t, not rates
      times their times
    - compounding: the yields' own: 'continuous', 'simple' or periods a year (1 annual, 2 semi-annual)
    - short_rate: None to fit r0 with the others, or today's short rate (a decimal) held fixed
    - weights: 'equal' or one positive weight per maturity
    - initial: the start, {'kappa': ..., 'theta': ..., 'sigma': ...} and 'short_rate' where it is fitted
    returns: a dict: kappa; theta; sigma; short_rate (fitted or the one given); fitted (the model's yields); rmse
      (unweighted); converged (the optimiser's flag)
    source: Vasicek (1977), J. Financial Economics 5; Brigo and Mercurio, Interest Rate Models (2006), 3.2.1
    aliases: Vasicek calibration, Vasicek fit, calibrate Vasicek, Vasicek curve fit
    example:
        qf.vasicek_fit_curve(maturities=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
                             yields=[0.031, 0.032, 0.034, 0.0355, 0.0375, 0.039, 0.0405], compounding="continuous",
                             short_rate=None, weights="equal",
                             initial={"kappa": 0.3, "theta": 0.05, "sigma": 0.01, "short_rate": 0.03})
        # kappa 0.255976; theta 0.0509335; sigma 0.0279094; short_rate 0.0296952; rmse 0.0000584442
    """
    rule = _zero_compounding(compounding)
    tau = positive("maturities", maturities).ravel()
    y = np.asarray(yields, dtype=float).ravel()
    if y.shape != tau.shape or not np.all(np.isfinite(y)):
        raise ValueError("yields must hold one finite decimal per maturity")
    w = np.ones(tau.size) if isinstance(weights, str) and weights == "equal" else positive("weights", weights).ravel()
    if w.size != tau.size:
        raise ValueError("weights must be 'equal' or one positive weight per maturity")
    names = ("kappa", "theta", "sigma") + (("short_rate",) if short_rate is None else ())
    if not isinstance(initial, dict) or set(initial) != set(names):
        raise ValueError(f"initial must be a dict of exactly {', '.join(names)}")
    start = np.array([float(initial[name]) for name in names])
    if not (start[0] > 0 and start[2] > 0):
        raise ValueError("initial kappa and sigma must be positive")

    def model(params):
        kappa, theta, sigma = params[:3]
        r0 = params[3] if short_rate is None else float(short_rate)
        big_a, big_b = _affine("vasicek", tau, kappa, theta, sigma)
        price = big_a * np.exp(-big_b * r0)
        if rule == "continuous":
            return -np.log(price) / tau
        if rule == "simple":
            return (1.0 / price - 1.0) / tau
        return rule * np.expm1(-np.log(price) / (rule * tau))

    low = np.array([1e-8, -np.inf, 1e-10] + ([-np.inf] if short_rate is None else []))
    root = np.sqrt(w)
    fit = least_squares(lambda p: root * (model(p) - y), start, bounds=(low, np.inf), method="trf", x_scale="jac",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    fitted = model(fit.x)
    return Result({"kappa": float(fit.x[0]), "theta": float(fit.x[1]), "sigma": float(fit.x[2]),
                   "short_rate": float(fit.x[3]) if short_rate is None else float(short_rate), "fitted": fitted,
                   "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))), "converged": bool(fit.success)})
