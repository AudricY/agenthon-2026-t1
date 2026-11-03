"""Vol smiles and surfaces: SVI, SABR, Dupire local vol, static arbitrage, Breeden-Litzenberger density, interpolation.

An SVI slice is the total implied variance w(k) = sigma^2 T of one maturity in log-moneyness k = ln(K / F); rates
are continuously compounded and annual, times in years, a local volatility annual. A function with several results
returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize

from ._common import Result, choice, ndtr, out, positive
from .conventions import _finite
from .options import bs_price
from .volatility import _series
from .primitives import _guard

__all__ = ["arbitrage_checks", "dupire_local_vol", "dupire_local_vol_grid", "risk_neutral_density", "sabr_fit",
           "sabr_vol", "svi_fit", "svi_local_vol", "svi_total_variance", "total_variance_interp"]

SVI_KEYS = ("a", "b", "rho", "m", "sigma")
T_DERIVATIVES = ("forward_difference", "central_difference")
ORIENTATIONS = ("maturity_rows", "strike_rows")
EDGE_RULES = ("nan", "one_sided", "second_order")
SECOND_DERIVATIVES = ("stencil", "gradient_twice")


# ------------------------------------------------------------------------------------------------ surfaces


def _svi(k, a, b, rho, m, sigma):
    d = k - m
    return a + b * (rho * d + np.sqrt(d * d + sigma * sigma))


def svi_total_variance(k, *, a, b, rho, m, sigma):
    """Raw SVI total variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)).

    k = ln(K / F) the log-moneyness against the forward; the implied vol is sqrt(w / T). The wings are linear with
    slopes b (1 - rho) and b (1 + rho); the minimum a + b sigma sqrt(1 - rho^2) sits at k = m - rho sigma /
    sqrt(1 - rho^2).

    Lee's moment formula: no wing's slope in total variance exceeds 2, so b (1 + |rho|) <= 2. Gyongy's mimicking
    theorem: the local vol sigma_loc^2(K, T) = E[sigma_T^2 | S_T = K] gives the surface's European prices
    (qf.svi_local_vol).
    conventions:
        a: the level
        b: the angle between the wings, at least 0
        rho: the rotation, inside (-1, 1)
        m: the horizontal shift
        sigma: the smoothness at the vertex, above 0
    returns: a float, or an array for an array k
    source: Gatheral (2004), "A parsimonious arbitrage-free implied volatility parametrization", Global Derivatives;
        Gatheral and Jacquier (2014), "Arbitrage-free SVI volatility surfaces", Quantitative Finance 14(1)
    aliases: SVI, raw SVI, stochastic volatility inspired, total implied variance, SVI parametrization
    example:
        import numpy as np, qf
        k = np.log(np.array([80.0, 100.0, 120.0]) / 100.0)
        qf.svi_total_variance(k, a=0.02, b=0.1, rho=-0.4, m=0.0, sigma=0.2)
    """
    if not (b >= 0 and -1 < rho < 1 and sigma > 0):
        raise ValueError("SVI needs b >= 0, -1 < rho < 1 and sigma > 0")
    return out(_svi(np.asarray(k, dtype=float), float(a), float(b), float(rho), float(m), float(sigma)))


def svi_fit(k, total_variance, *, initial):
    """Least-squares raw SVI fit to total implied variances, from a named starting point.

    Minimises sum (w_svi(k_i) - w_i)^2 (unweighted, in total variance) by scipy.optimize.least_squares ('trf')
    within b >= 0, |rho| < 1, sigma > 0, from `initial`; a + b sigma sqrt(1 - rho^2) >= 0 and the absence of
    butterfly arbitrage are not imposed (check them). w_i = implied vol^2 T.
    conventions:
        initial: a dict with the five starting values a, b, rho, m, sigma (inside the bounds)
        the optimiser: scipy.optimize.least_squares; a task naming another fits svi_total_variance with it
    returns: a dict: a, b, rho, m, sigma (raw SVI, in total variance); rmse (in total variance); converged (the
        optimizer's flag: False is not a fit)
    source: Gatheral (2004), Global Derivatives; Gatheral and Jacquier (2014), Quantitative Finance 14(1)
    aliases: SVI fit, calibrate SVI, SVI calibration, fit SVI smile
    example:
        import numpy as np, qf
        k = np.linspace(-0.4, 0.4, 9); w = qf.svi_total_variance(k, a=0.02, b=0.1, rho=-0.4, m=0.05, sigma=0.2)
        qf.svi_fit(k, w, initial={"a": 0.01, "b": 0.2, "rho": 0.0, "m": 0.0, "sigma": 0.1})
    """
    if not isinstance(initial, dict) or set(initial) != set(SVI_KEYS):
        raise ValueError(f"initial must be a dict with exactly the keys {list(SVI_KEYS)}")
    kk, w = _series("k", k, least=5), _series("total_variance", total_variance, least=5)
    if kk.size != w.size:
        raise ValueError("k and total_variance must have equal lengths")
    start = np.array([float(initial[key]) for key in SVI_KEYS])
    lower = np.array([-np.inf, 0.0, -1.0 + 1e-9, -np.inf, 1e-10])
    upper = np.array([np.inf, np.inf, 1.0 - 1e-9, np.inf, np.inf])
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("initial must satisfy b >= 0, -1 < rho < 1 and sigma > 0")
    found = optimize.least_squares(lambda p: _svi(kk, *p) - w, start, bounds=(lower, upper), method="trf",
                                   xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    return Result({**dict(zip(SVI_KEYS, map(float, found.x))), "rmse": float(np.sqrt(np.mean(found.fun ** 2))),
                   "converged": bool(found.success)})


def _svi_parts(k, a, b, rho, m, sigma):
    """w, dw/dk and d2w/dk2 of one raw SVI slice at k."""
    d = k - m
    root = np.sqrt(d * d + sigma * sigma)
    return a + b * (rho * d + root), b * (rho + d / root), b * sigma * sigma / root ** 3


@_guard(rate=('rate', 'dividend_yield',), time=('maturity', 'maturities',), price=('strike', 'spot',))
def svi_local_vol(strike, *, spot, maturity, rate, dividend_yield, maturities, a, b, rho, m, sigma, t_derivative):
    """Local volatility from an SVI surface by Gatheral's formula in total variance.

    sigma_loc^2(K, T) = w_T / (1 - (k/w) w_k + (1/4)(-1/4 - 1/w + k^2/w^2) w_k^2 + w_kk / 2), k = ln(K / F_T), F_T =
    S e^{(r - q) T}, w(k, T) the total implied variance. The slices are raw SVI (svi_total_variance), one per
    maturity, each in its own forward's k; between two, w is linear in T at fixed k, so w_k and w_kk are the slices'
    exact derivatives mixed alike. w_T: 'forward_difference' the slope of w to the next slice (the last slice's to
    the one before); 'central_difference', at a slice, the three-point derivative across its neighbours; between
    slices both are the interval's slope. numpy.nan where w_T or the denominator is not positive (calendar or
    butterfly arbitrage). A flat surface, a = vol^2 T, b = 0, returns vol; on smooth prices dupire_local_vol agrees.
    conventions:
        spot: S, for the forward
        maturity: T, within the slices' maturities
        rate: r, continuously compounded, for the forward
        dividend_yield: q, continuously compounded, for the forward
        maturities: the slices' increasing maturities, at least two
        a: each slice's level (raw SVI, svi_total_variance's), one per slice
        b: each slice's wing angle, at least 0
        rho: each slice's rotation, inside (-1, 1)
        m: each slice's shift
        sigma: each slice's smoothness, above 0
        t_derivative: 'forward_difference' or 'central_difference'
    returns: a dict: local_vol, local_variance, k, w, w_k, w_kk, w_t (each shaped like strike)
    source: Gatheral (2006), The Volatility Surface, eq. 1.10; Gatheral and Jacquier (2014), Quantitative Finance 14(1)
    aliases: SVI local volatility, local volatility from SVI, Gatheral local variance, local variance from total implied variance, local vol from implied vol surface
    example:
        import qf
        K = [90.0, 100.0, 110.0]
        qf.svi_local_vol(K, spot=100.0, maturity=0.75, rate=0.03, dividend_yield=0.01, maturities=[0.5, 1.0], a=[0.01, 0.02], b=[0.1, 0.1], rho=[-0.4, -0.4], m=[0.0, 0.0], sigma=[0.2, 0.25], t_derivative="central_difference")
    """
    choice("t_derivative", t_derivative, T_DERIVATIVES)
    k_strike = positive("strike", strike)
    s, t = float(positive("spot", spot)), float(positive("maturity", maturity))
    times = _series("maturities", maturities)
    if np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("maturities must be positive and strictly increasing, at least two")
    slices = [np.asarray(v, dtype=float).ravel() for v in (a, b, rho, m, sigma)]
    if any(v.shape != times.shape for v in slices):
        raise ValueError("a, b, rho, m and sigma must hold one value per maturity")
    if np.any(slices[1] < 0) or np.any(np.abs(slices[2]) >= 1) or np.any(slices[4] <= 0):
        raise ValueError("SVI needs b >= 0, -1 < rho < 1 and sigma > 0 on every slice")
    if not times[0] <= t <= times[-1]:
        raise ValueError("maturity must lie within the slices' maturities (no extrapolation)")
    k = np.log(k_strike / (s * math.exp((float(rate) - float(dividend_yield)) * t)))
    parts = [_svi_parts(k, *(v[i] for v in slices)) for i in range(times.size)]
    i = min(int(np.searchsorted(times, t, side="right")) - 1, times.size - 2)
    theta = (t - times[i]) / (times[i + 1] - times[i])
    w, w_k, w_kk = ((1 - theta) * low + theta * high for low, high in zip(parts[i], parts[i + 1]))
    w_t = (parts[i + 1][0] - parts[i][0]) / (times[i + 1] - times[i])
    if t_derivative == "central_difference" and t in times[1:-1]:
        j = int(np.searchsorted(times, t))
        lo, hi = times[j] - times[j - 1], times[j + 1] - times[j]
        w_t = (lo * lo * parts[j + 1][0] - hi * hi * parts[j - 1][0] + (hi * hi - lo * lo) * parts[j][0]) / (
            lo * hi * (lo + hi))
    denominator = 1 - k / w * w_k + 0.25 * (-0.25 - 1 / w + k * k / (w * w)) * w_k * w_k + 0.5 * w_kk
    with np.errstate(invalid="ignore", divide="ignore"):
        local = np.where((w_t > 0) & (denominator > 0), w_t / denominator, np.nan)
    return Result({"local_vol": out(np.sqrt(local)), "local_variance": out(local), "k": out(k), "w": out(w),
                   "w_k": out(w_k), "w_kk": out(w_kk), "w_t": out(w_t)})


def _first_and_second(values, grid, axis):
    """Central differences on a non-uniform grid along `axis`: (f', f'') at the interior points, NaN at the ends."""
    f = np.moveaxis(values, axis, 0)
    lo, hi = np.diff(grid)[:-1], np.diff(grid)[1:]
    shape = (-1,) + (1,) * (f.ndim - 1)
    lo, hi = lo.reshape(shape), hi.reshape(shape)
    first, second = np.full(f.shape, np.nan), np.full(f.shape, np.nan)
    first[1:-1] = (lo * lo * f[2:] - hi * hi * f[:-2] + (hi * hi - lo * lo) * f[1:-1]) / (lo * hi * (lo + hi))
    second[1:-1] = 2.0 * (lo * f[2:] - (lo + hi) * f[1:-1] + hi * f[:-2]) / (lo * hi * (lo + hi))
    return np.moveaxis(first, 0, axis), np.moveaxis(second, 0, axis)


@_guard(rate=('rate', 'dividend_yield',), time=('maturities',), price=('strikes',))
def dupire_local_vol(calls, *, strikes, maturities, rate, dividend_yield):
    """Dupire's local volatility from European call prices on a strike-maturity grid.

    sigma_loc^2(K, T) = [dC/dT + (r - q) K dC/dK + q C] / [0.5 K^2 d2C/dK2], the derivatives by second-order central
    differences on the (possibly uneven) grid; edge strikes and maturities, and points where the numerator or
    denominator is not positive (an arbitrage or noise), are NaN. calls[i, j] is the call of maturity i and strike j.
    From a fitted SVI per expiry the analytic route is Gatheral's local variance in w(k, T) and its k-derivatives
    (The Volatility Surface, 1.10): a task asking analytic derivatives wants that, not this grid.
    conventions:
        strikes: increasing positive strikes, one per column (strike rows, one-sided edges: dupire_local_vol_grid)
        maturities: increasing positive maturities in years, one per row
        rate: r, continuously compounded, annual
        dividend_yield: q, continuously compounded, annual
    returns: an array of local vols shaped like calls
    source: Dupire (1994), "Pricing with a smile", Risk 7(1); Gatheral, The Volatility Surface (2006), 1.2
    aliases: Dupire, local volatility, local vol, Dupire formula, local volatility surface
    example:
        import numpy as np, qf
        K, T = np.linspace(60.0, 140.0, 41), np.linspace(0.25, 2.0, 8)
        qf.dupire_local_vol(qf.bs_price("call", 100.0, K, T[:, None], 0.03, 0.01, 0.2), strikes=K, maturities=T, rate=0.03, dividend_yield=0.01)
    """
    k = _series("strikes", strikes, least=3)
    t = _series("maturities", maturities, least=3)
    if np.any(np.diff(k) <= 0) or np.any(k <= 0) or np.any(np.diff(t) <= 0) or np.any(t <= 0):
        raise ValueError("strikes and maturities must be positive and strictly increasing")
    c = _finite("calls", calls)
    if c.shape != (t.size, k.size):
        raise ValueError(f"calls must be maturities x strikes, {t.size} x {k.size}, not {c.shape}")
    r, q = float(rate), float(dividend_yield)
    c_t, _ = _first_and_second(c, t, 0)
    c_k, c_kk = _first_and_second(c, k, 1)
    numerator = c_t + (r - q) * k * c_k + q * c
    denominator = 0.5 * k * k * c_kk
    with np.errstate(invalid="ignore", divide="ignore"):
        local = np.where((numerator > 0) & (denominator > 0), numerator / denominator, np.nan)
    return np.sqrt(local)


@_guard(rate=('rate', 'dividend_yield',), time=('maturities',), price=('strikes',))
def dupire_local_vol_grid(calls, strikes, maturities, rate, dividend_yield, orientation, edges, second_derivative):
    """Dupire local vol on a call grid, its orientation, edge differences and second derivative named.

    sigma_loc^2 = [C_T + (r - q) K C_K + q C] / [0.5 K^2 C_KK], dupire_local_vol's formula. Inside the grid, C_T and C_K
    are second-order central differences on the (possibly uneven) grid, numpy.gradient's; at the edge rows and
    columns 'nan' leaves them NaN (dupire_local_vol's), 'one_sided' takes the forward or backward difference
    (numpy.gradient's edge_order=1), 'second_order' the one-sided second-order difference (edge_order=2). C_KK:
    'stencil' the three-point second difference (NaN at the edge strikes), 'gradient_twice'
    numpy.gradient(numpy.gradient(C, K), K) under the edge rule (not with 'nan'). The local variance is returned as
    computed, so a task's floor, cap or small-denominator rule applies to it; local_vol is its root where numerator
    and denominator are positive, NaN elsewhere.

    conventions:
    - orientation: 'maturity_rows' (calls[i, j] maturity i, strike j: dupire_local_vol's) or 'strike_rows' (calls[i, j]
      strike i, maturity j); every result comes back in the same orientation
    - edges: 'nan', 'one_sided' or 'second_order'; second_derivative: 'stencil' or 'gradient_twice'
    - strikes, maturities: positive and increasing; rate, dividend_yield: r and q, continuously compounded, annual
    returns: a dict: local_vol; local_variance (unfloored); c_t, c_k, c_kk (the derivatives); each shaped like calls
    source: Dupire (1994), "Pricing with a smile", Risk 7(1); Gatheral, The Volatility Surface (2006), 1.2;
      Fornberg (1988), Mathematics of Computation 51 (numpy.gradient's differences on uneven grids)
    aliases: Dupire local vol grid, local volatility from call prices, Dupire finite differences, strike rows, [numpy.gradient], [edge_order]
    example:
        # Returns local_vol near 0.2 off a flat 0.2 surface, NaN on the edge strikes (strikes in rows).
        import numpy as np, qf
        K, T = np.linspace(80.0, 120.0, 9), np.array([0.25, 0.5, 1.0, 1.5])
        qf.dupire_local_vol_grid(calls=qf.bs_price("call", 100.0, K[:, None], T, 0.02, 0.0, 0.2), strikes=K, maturities=T, rate=0.02, dividend_yield=0.0, orientation="strike_rows", edges="one_sided", second_derivative="stencil")
    """
    choice("orientation", orientation, ORIENTATIONS)
    choice("edges", edges, EDGE_RULES)
    choice("second_derivative", second_derivative, SECOND_DERIVATIVES)
    if edges == "nan" and second_derivative == "gradient_twice":
        raise ValueError("second_derivative 'gradient_twice' needs an edge rule: edges 'one_sided' or 'second_order'")
    k = _series("strikes", strikes, least=3)
    t = _series("maturities", maturities, least=3)
    if np.any(np.diff(k) <= 0) or np.any(k <= 0) or np.any(np.diff(t) <= 0) or np.any(t <= 0):
        raise ValueError("strikes and maturities must be positive and strictly increasing")
    given = _finite("calls", calls)
    c = given.T if orientation == "strike_rows" and given.ndim == 2 else given
    if c.shape != (t.size, k.size):
        wanted = (k.size, t.size) if orientation == "strike_rows" else (t.size, k.size)
        raise ValueError(f"calls must be {wanted[0]} x {wanted[1]} under orientation {orientation!r}, not {given.shape}")
    if edges == "nan":
        c_t, _ = _first_and_second(c, t, 0)
        c_k, c_kk = _first_and_second(c, k, 1)
    else:
        order = 1 if edges == "one_sided" else 2
        c_t, c_k = np.gradient(c, t, axis=0, edge_order=order), np.gradient(c, k, axis=1, edge_order=order)
        c_kk = (_first_and_second(c, k, 1)[1] if second_derivative == "stencil"
                else np.gradient(c_k, k, axis=1, edge_order=order))
    r, q = float(rate), float(dividend_yield)
    numerator = c_t + (r - q) * k * c_k + q * c
    denominator = 0.5 * k * k * c_kk
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = numerator / denominator
        local = np.where((numerator > 0) & (denominator > 0), np.sqrt(np.abs(variance)), np.nan)
    turn = (lambda a: a.T) if orientation == "strike_rows" else (lambda a: a)
    return Result({"local_vol": turn(local), "local_variance": turn(variance), "c_t": turn(c_t), "c_k": turn(c_k),
                   "c_kk": turn(c_kk)})


# ------------------------------------------------------------------------------------------------ SABR

VOL_TYPES = ("lognormal", "normal")
ALPHA_RULES = ("atm", "free")
SABR_LOSSES = ("vol", "price")


def _sabr(k, f, t, alpha, beta, rho, nu, vol_type):
    """Hagan et al.'s SABR implied vol at strikes k (shift already added), the at-the-money limit taken in series."""
    one = 1.0 - beta
    fk = f * k
    log_m = np.log(f / k)
    z = nu / alpha * fk ** (0.5 * one) * log_m
    root = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(z) > 1e-7, z / np.log((root + z - rho) / (1.0 - rho)),
                         1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0)
    wings = 1.0 + one * one * log_m ** 2 / 24.0 + one ** 4 * log_m ** 4 / 1920.0
    cross = rho * beta * nu * alpha / (4.0 * fk ** (0.5 * one)) + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    if vol_type == "lognormal":
        level = alpha / (fk ** (0.5 * one) * wings)
        return level * ratio * (1.0 + (one * one * alpha * alpha / (24.0 * fk ** one) + cross) * t)
    level = alpha * fk ** (0.5 * beta) * (1.0 + log_m ** 2 / 24.0 + log_m ** 4 / 1920.0) / wings
    return level * ratio * (1.0 + (-beta * (2.0 - beta) * alpha * alpha / (24.0 * fk ** one) + cross) * t)


def _sabr_check(beta, rho, nu, alpha=1.0):
    if not (0.0 <= beta <= 1.0 and -1.0 < rho < 1.0 and nu >= 0.0 and alpha > 0.0):
        raise ValueError("SABR needs 0 <= beta <= 1, -1 < rho < 1, nu >= 0 and alpha > 0")


@_guard(time=('expiry',))
def sabr_vol(strike, forward, expiry, alpha, beta, rho, nu, shift, vol_type):
    """SABR implied vol by Hagan's expansion (lognormal or normal), shifted SABR for negative rates.

    dF = alpha_t F^beta dW, d alpha_t = nu alpha_t dZ, dW dZ = rho dt (F + shift in place of F where shifted). With
    z = (nu/alpha) (FK)^{(1 - beta)/2} ln(F/K), x(z) = ln[(sqrt(1 - 2 rho z + z^2) + z - rho)/(1 - rho)], L = ln(F/K):
    lognormal: alpha / {(FK)^{(1-beta)/2} [1 + (1-beta)^2 L^2/24 + (1-beta)^4 L^4/1920]} (z/x(z)) {1 + [(1-beta)^2
      alpha^2 / (24 (FK)^{1-beta}) + rho beta nu alpha / (4 (FK)^{(1-beta)/2}) + (2 - 3 rho^2) nu^2/24] T};
    normal: alpha (FK)^{beta/2} [1 + L^2/24 + L^4/1920] / [1 + (1-beta)^2 L^2/24 + (1-beta)^4 L^4/1920] (z/x(z)) {1 +
      [-beta (2 - beta) alpha^2 / (24 (FK)^{1-beta}) + rho beta nu alpha / (4 (FK)^{(1-beta)/2}) + (2 - 3 rho^2)
      nu^2/24] T};
    at the money z/x(z) -> 1 (a series near it). The leading ATM backbone is alpha F^(beta - 1) for lognormal
    vols, alpha F^beta for normal vols; the expansion also includes the finite-expiry correction above.

    conventions:
    - strike, forward: K and F, rates in decimals; shift: s added to both (0 for plain SABR; 0.01 lets F, K > -1%)
    - expiry: T in years; alpha: the vol level (F^(1 - beta) times the ATM vol, roughly); beta: in [0, 1], fixed by
      the user (1 lognormal, 0 normal, 0.5 common for rates); rho: in (-1, 1); nu: the vol of vol, >= 0
    - vol_type: 'lognormal' (a Black vol, qf.black76_price) or 'normal' (a Bachelier vol, qf.bachelier_price)
    returns: the implied vol, a float or an array shaped like strike
    source: Hagan, Kumar, Lesniewski and Woodward (2002), "Managing smile risk", Wilmott, eqs. (2.17) and (B.69);
      West (2005), "Calibration of the SABR model in illiquid markets", Applied Mathematical Finance 12(4)
    aliases: SABR, SABR model, Hagan formula, shifted SABR
    example:
        import qf
        qf.sabr_vol(strike=[0.02, 0.03, 0.04], forward=0.03, expiry=1.0, alpha=0.035,
                    beta=0.5, rho=-0.3, nu=0.4, shift=0.0,
                    vol_type="lognormal")  # result: [0.26183855, 0.20387662, 0.18387927]
    """
    choice("vol_type", vol_type, VOL_TYPES)
    _sabr_check(float(beta), float(rho), float(nu), float(alpha))
    s = float(shift)
    k = positive("strike + shift", np.asarray(strike, dtype=float) + s)
    f = float(positive("forward + shift", float(forward) + s))
    t = float(positive("expiry", expiry))
    return out(_sabr(k, f, t, float(alpha), float(beta), float(rho), float(nu), vol_type))


def _atm_alpha(atm_vol, f, t, beta, rho, nu, vol_type):
    """The smallest positive alpha whose SABR ATM vol is atm_vol: the ATM cubic (West 2005)."""
    one = 1.0 - beta
    if vol_type == "lognormal":
        coefficients = [one * one * t / (24.0 * f ** (2.0 * one)), rho * beta * nu * t / (4.0 * f ** one),
                        1.0 + (2.0 - 3.0 * rho * rho) * nu * nu * t / 24.0, -atm_vol * f ** one]
    else:
        coefficients = [-beta * (2.0 - beta) * t / (24.0 * f ** (2.0 * one)) * f ** beta,
                        rho * beta * nu * t / (4.0 * f ** one) * f ** beta,
                        (1.0 + (2.0 - 3.0 * rho * rho) * nu * nu * t / 24.0) * f ** beta, -atm_vol]
    roots = np.roots(coefficients)
    real = roots[(np.abs(roots.imag) < 1e-10 * np.abs(roots)) & (roots.real > 0)].real
    if real.size == 0:
        raise ValueError("no positive alpha gives this ATM vol")
    return float(real.min())


@_guard(time=('expiry',))
def sabr_fit(strikes, vols, forward, expiry, beta, shift, vol_type, alpha_rule, atm_vol, loss, weights, initial):
    """SABR calibration to a smile (West's ATM cubic).

    Minimises sum_i w_i e_i^2 by scipy.optimize.least_squares ('trf', rho in (-1, 1), nu and alpha positive), e_i the
    gap between qf.sabr_vol and the market vol at strike i ('vol') or between their undiscounted out-of-the-money
    prices (Black-76 for lognormal, Bachelier for normal vols; calls at K >= F, puts below) ('price'). 'atm': alpha
    is the smallest positive root of the ATM cubic, (1 - beta)^2 T / (24 F^{2 - 2 beta}) alpha^3 + rho beta nu T /
    (4 F^{1 - beta}) alpha^2 + [1 + (2 - 3 rho^2) nu^2 T / 24] alpha - atm_vol F^{1 - beta} = 0 (lognormal; its
    normal twin), so the ATM vol is matched exactly; 'free' fits alpha with the rest.

    conventions:
    - strikes, vols: the smile, one market vol per strike, of vol_type ('lognormal' or 'normal'); forward F; expiry T
    - beta: fixed in [0, 1], not fitted; shift: added to F and K (0 for plain SABR)
    - alpha_rule: 'atm' (atm_vol given) or 'free' (atm_vol None); loss: 'vol' or 'price'
    - weights: one per strike, or 'equal'; initial: a dict of the start values, rho and nu (and alpha under 'free')
    returns: a dict: alpha, beta, rho, nu; rmse (unweighted, in vol); converged (the optimiser's flag); model (the
      fitted vols at strikes)
    source: Hagan, Kumar, Lesniewski and Woodward (2002), Wilmott; West (2005), Applied Mathematical Finance 12(4)
    aliases: SABR calibration, calibrate SABR
    example:
        import qf
        qf.sabr_fit(strikes=[0.02, 0.025, 0.03, 0.035, 0.04], vols=[0.31, 0.27, 0.245, 0.235, 0.233],
                    forward=0.03, expiry=1.0, beta=0.5, shift=0.0, vol_type="lognormal",
                    alpha_rule="atm", atm_vol=0.245, loss="vol", weights="equal",
                    initial={"rho": 0.0, "nu": 0.5})  # result: alpha 0.04154701, rho -0.15395652, nu 0.53800526
    """
    choice("vol_type", vol_type, VOL_TYPES)
    choice("alpha_rule", alpha_rule, ALPHA_RULES)
    choice("loss", loss, SABR_LOSSES)
    s, b = float(shift), float(beta)
    k = positive("strikes + shift", np.asarray(strikes, dtype=float).ravel() + s)
    market = positive("vols", vols).ravel()
    f, t = float(positive("forward + shift", float(forward) + s)), float(positive("expiry", expiry))
    if k.size != market.size or k.size < 3:
        raise ValueError("strikes and vols must hold one vol per strike, three or more")
    w = np.ones(k.size) if isinstance(weights, str) and weights == "equal" else positive("weights", weights).ravel()
    if w.size != k.size:
        raise ValueError("weights must be 'equal' or one positive weight per strike")
    names = ("rho", "nu") if alpha_rule == "atm" else ("alpha", "rho", "nu")
    if not isinstance(initial, dict) or set(initial) != set(names):
        raise ValueError(f"initial must be a dict of exactly {', '.join(names)}")
    if (alpha_rule == "atm") != (atm_vol is not None):
        raise ValueError("atm_vol is the market ATM vol under alpha_rule 'atm', and None under 'free'")
    _sabr_check(b, 0.0, 0.0)
    target = float(positive("atm_vol", atm_vol)) if alpha_rule == "atm" else None
    calls = k >= f

    def premium(vol):
        sd = vol * np.sqrt(t)
        if vol_type == "normal":
            d = (f - k) / sd
            return np.where(calls, (f - k) * ndtr(d), (k - f) * ndtr(-d)) + sd * np.exp(-0.5 * d * d) / np.sqrt(
                2.0 * np.pi)
        d1 = (np.log(f / k) + 0.5 * sd * sd) / sd
        return np.where(calls, f * ndtr(d1) - k * ndtr(d1 - sd), k * ndtr(sd - d1) - f * ndtr(-d1))

    def unpack(x):
        if alpha_rule == "atm":
            return _atm_alpha(target, f, t, b, x[0], x[1], vol_type), x[0], x[1]
        return x[0], x[1], x[2]

    reference = premium(market) if loss == "price" else market

    def residuals(x):
        try:
            alpha, rho, nu = unpack(x)
        except ValueError:
            return np.full(k.size, 1e3)
        model = _sabr(k, f, t, alpha, b, rho, nu, vol_type)
        return np.sqrt(w) * ((premium(model) if loss == "price" else model) - reference)

    start = np.array([float(initial[name]) for name in names])
    low = np.array([1e-8 if name == "alpha" else (-0.9999 if name == "rho" else 1e-8) for name in names])
    high = np.array([np.inf if name != "rho" else 0.9999 for name in names])
    if np.any(start < low) or np.any(start > high):
        raise ValueError("initial must hold rho in (-1, 1) and positive nu and alpha")
    fit = optimize.least_squares(residuals, start, bounds=(low, high), method="trf", xtol=1e-15, ftol=1e-15,
                                 gtol=1e-15, max_nfev=5000)
    alpha, rho, nu = unpack(fit.x)
    model = _sabr(k, f, t, alpha, b, rho, nu, vol_type)
    return Result({"alpha": float(alpha), "beta": b, "rho": float(rho), "nu": float(nu),
                   "rmse": float(np.sqrt(np.mean((model - market) ** 2))), "converged": bool(fit.success),
                   "model": model})


# ------------------------------------------------------------------------------------------------ densities, checks

DENSITY_INPUTS = ("prices", "vols")


@_guard(rate=('rate', 'dividend_yield',), time=('maturity',), price=('strikes', 'spot',))
def risk_neutral_density(strikes, values, maturity, rate, inputs, spot, dividend_yield):
    """Risk-neutral density by Breeden-Litzenberger.

    q(K) = e^{rT} d2C/dK2, the density of S_T under the pricing measure; its CDF P(S_T <= K) = 1 + e^{rT} dC/dK;
    -dC/dK is the price of a digital call paying 1 (a cash-or-nothing). The derivatives are the three-point central
    differences on the strikes' own (uneven) grid; the end strikes are NaN. From implied vols the calls are priced
    first by qf.bs_price. On a strike grid fine and wide enough the density integrates to 1.

    conventions:
    - strikes: increasing; inputs: 'prices' (values are European call prices there, today's) or 'vols' (values are
      Black-Scholes implied vols; spot and dividend_yield then given, None for 'prices')
    - maturity: T in years; rate: r continuously compounded (the e^{rT} undoes the discounting)
    returns: a dict: density (q(K)); cdf (P(S_T <= K)); digital (-dC/dK, a discounted digital call); each shaped like
      strikes, NaN at the two end strikes
    source: Breeden and Litzenberger (1978), J. Business 51(4); Gatheral (2006), The Volatility Surface, ch. 1; Hull,
      "Determining implied risk-neutral distributions from volatility smiles"
    aliases: risk-neutral density, Breeden-Litzenberger, state price density
    example:
        import qf
        qf.risk_neutral_density(strikes=[80.0, 90.0, 100.0, 110.0, 120.0], values=[0.25, 0.22, 0.2, 0.19, 0.19],
                                maturity=0.5, rate=0.03, inputs="vols", spot=100.0,
                                dividend_yield=0.0)  # result: density [NaN, 0.01872963, 0.02845497, 0.02422646, NaN]
    """
    choice("inputs", inputs, DENSITY_INPUTS)
    k = _series("strikes", strikes, least=3)
    if np.any(np.diff(k) <= 0) or k[0] <= 0:
        raise ValueError("strikes must be positive and strictly increasing")
    given = np.asarray(values, dtype=float).ravel()
    if given.shape != k.shape:
        raise ValueError("values must hold one call price or vol per strike")
    t, r = float(positive("maturity", maturity)), float(rate)
    if inputs == "vols":
        if spot is None or dividend_yield is None:
            raise ValueError("inputs 'vols' takes spot and dividend_yield")
        calls = np.asarray(bs_price("call", float(positive("spot", spot)), k, t, r, float(dividend_yield),
                                    positive("values", given)), dtype=float)
    else:
        if spot is not None or dividend_yield is not None:
            raise ValueError("inputs 'prices' takes spot=None and dividend_yield=None")
        calls = given
    first, second = _first_and_second(calls[None, :], k, 1)
    growth = np.exp(r * t)
    return Result({"density": growth * second[0], "cdf": 1.0 + growth * first[0], "digital": -first[0]})


INTERP_VARIABLES = ("total_variance", "variance", "vol")
EXTRAPOLATIONS = ("flat_vol", "flat_forward", "none")


@_guard(time=('maturity', 'maturities',))
def total_variance_interp(maturity, maturities, vols, variable, extrapolation):
    """Total-variance interpolation of a smile across expiries.

    At a fixed log-moneyness (or delta), w(T) = w_i + (w_{i+1} - w_i)(T - T_i)/(T_{i+1} - T_i) between expiries
    T_i < T <= T_{i+1}, vol(T) = sqrt(w(T)/T): the forward variance (w_{i+1} - w_i)/(T_{i+1} - T_i) is flat
    between them, and it is >= 0 where the quotes have no calendar arbitrage (NaN forward vol where it is not).
    'variance' takes vol^2 linear in T, 'vol' vol itself, instead. Outside the expiries: 'flat_vol' holds the end
    vol; 'flat_forward' holds the first vol before the first expiry and the last interval's forward variance after
    the last; 'none' refuses.

    conventions:
    - maturity: the target T (a number or an array), years; maturities: the quoted expiries, increasing
    - vols: one vol per expiry, or a 2-D array, one row per expiry (one column per fixed log-moneyness or delta)
    - variable: 'total_variance', 'variance' or 'vol' (what is linear in T); extrapolation: 'flat_vol',
      'flat_forward' or 'none'
    returns: a dict: vol and total_variance at maturity; forward_vol (sqrt of the forward variance of the interval
      holding it, NaN outside the expiries or where negative); each shaped like maturity (times the columns)
    source: Gatheral (2006), The Volatility Surface, ch. 3; Castagna (2010), FX Options and Smile Risk, ch. 6
    aliases: total variance interpolation
    example:
        import qf
        qf.total_variance_interp(maturity=0.75, maturities=[0.5, 1.0, 2.0], vols=[0.2, 0.22, 0.21],
                                 variable="total_variance", extrapolation="flat_vol")  # result: vol 0.21354157, total_variance 0.0342, forward_vol 0.23832751
    """
    choice("variable", variable, INTERP_VARIABLES)
    choice("extrapolation", extrapolation, EXTRAPOLATIONS)
    times = _series("maturities", maturities, least=2)
    if np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("maturities must be positive and strictly increasing")
    sigma = positive("vols", vols)
    if sigma.shape[0] != times.size or sigma.ndim > 2:
        raise ValueError("vols must hold one vol (or one row) per maturity")
    target = positive("maturity", maturity)
    flat_t = target.ravel()
    if extrapolation == "none" and (np.any(flat_t < times[0]) or np.any(flat_t > times[-1])):
        raise ValueError("maturity lies outside the maturities and extrapolation is 'none'")
    grid = times.reshape((-1,) + (1,) * (sigma.ndim - 1))
    level = {"total_variance": sigma * sigma * grid, "variance": sigma * sigma, "vol": sigma}[variable]
    w_nodes = sigma * sigma * grid
    i = np.clip(np.searchsorted(times, flat_t, side="left") - 1, 0, times.size - 2)
    span = (times[i + 1] - times[i]).reshape((-1,) + (1,) * (sigma.ndim - 1))
    theta = ((flat_t - times[i]).reshape(span.shape)) / span
    mixed = (1.0 - theta) * level[i] + theta * level[i + 1]
    tt = flat_t.reshape(span.shape)
    w = {"total_variance": mixed, "variance": mixed * tt, "vol": mixed * mixed * tt}[variable]
    forward_var = (w_nodes[i + 1] - w_nodes[i]) / span
    before, after = (flat_t < times[0]).reshape(span.shape), (flat_t > times[-1]).reshape(span.shape)
    first_w, last_w = sigma[0] * sigma[0] * tt, sigma[-1] * sigma[-1] * tt
    if extrapolation == "flat_forward":
        tail = w_nodes[-1] + (w_nodes[-1] - w_nodes[-2]) / (times[-1] - times[-2]) * (tt - times[-1])
    else:
        tail = last_w
    w = np.where(before, first_w, np.where(after, tail, w))
    with np.errstate(invalid="ignore"):
        forward = np.where(before | after | (forward_var < 0), np.nan, np.sqrt(np.maximum(forward_var, 0.0)))
        vol = np.sqrt(np.where(w >= 0, w, np.nan) / tt)
    shape = target.shape + sigma.shape[1:]
    return Result({"vol": out(vol.reshape(shape)), "total_variance": out(w.reshape(shape)),
                   "forward_vol": out(forward.reshape(shape))})


CHECK_INPUTS = ("calls", "forward_calls", "total_variance")


def _rows(name, grid, shape):
    """The grid as one row per maturity: a 1-D grid shared by every row, or a 2-D one, each row increasing."""
    g = _finite(name, grid)
    g = np.broadcast_to(g, shape) if g.ndim == 1 else g
    if g.shape != shape or np.any(np.diff(g, axis=1) <= 0):
        raise ValueError(f"{name} must be increasing, one row per maturity (or one row for all), shaped like values")
    return g


@_guard(time=('maturities',))
def arbitrage_checks(inputs, grid, maturities, values, forwards, discount_factors, tolerance):
    """Static arbitrage checks of a smile or surface: butterfly and calendar arbitrage (Gatheral-Jacquier).

    Call prices, made forward (C/DF): max(F - K, 0) <= C <= F; nonincreasing in K, slope at least -1; convex, each
    butterfly's density 2 (s_i - s_{i-1}) / (K_{i+1} - K_{i-1}) >= 0 on the strikes' uneven spacing (s the
    slopes); calendar: C/F at a fixed log-moneyness k = ln(K/F) does not fall as T grows. Total variance w(k) =
    vol^2 T: w >= 0; Gatheral and Jacquier's g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2 >= 0 (uneven
    central differences); w nondecreasing in T at each k. A later maturity is read at the earlier one's
    moneyness by linear interpolation, inside its range: calls in K/F, total variance in ln(K/F).

    The calendar check is the forward variance's sign, (w_2 - w_1)/(T_2 - T_1) >= 0. Puts go in as calls by
    parity first, C = P + DF (F - K).

    conventions:
    - inputs: 'calls' (today's prices, discount_factors one per maturity), 'forward_calls' (prices already divided by
      the discount factor; discount_factors None) or 'total_variance' (values are w at log-moneyness)
    - grid: strikes K (prices) or log-moneyness k = ln(K/F) (total variance), one increasing row for all maturities
      or one a maturity; values: one row a maturity, maturities increasing; forwards: F, one a maturity
    - tolerance: the slack allowed, in the values' units (0 for a strict check)
    returns: a dict: arbitrage_free; bounds, monotonic, butterfly (True where that check fails, shaped like values);
      calendar (True at a point the next maturity falls below); density (the butterfly's, or g(k); NaN at the ends)
    source: Carr and Madan (2005), "A note on sufficient conditions for no arbitrage", Finance Research Letters 2;
      Gatheral and Jacquier (2014), "Arbitrage-free SVI volatility surfaces", Quantitative Finance 14(1)
    aliases: static arbitrage, butterfly arbitrage, [calendar arbitrage], Gatheral-Jacquier
    example:
        import qf
        qf.arbitrage_checks(inputs="forward_calls", grid=[90.0, 100.0, 110.0], maturities=[0.5, 1.0],
                            values=[[11.8, 5.6, 2.2], [13.6, 8.0, 4.3]], forwards=[100.0, 100.0],
                            discount_factors=None, tolerance=0.0)  # result: arbitrage_free True
    """
    choice("inputs", inputs, CHECK_INPUTS)
    v = _finite("values", values)
    v = v[None, :] if v.ndim == 1 else v
    times = positive("maturities", _finite("maturities", maturities)).ravel()
    f = positive("forwards", _finite("forwards", forwards)).ravel()
    if v.ndim != 2 or v.shape[0] != times.size or f.size != times.size or v.shape[1] < 3:
        raise ValueError("values must be one row per maturity, three or more points a row; forwards one per maturity")
    if np.any(np.diff(times) <= 0):
        raise ValueError("maturities must be strictly increasing")
    tol = float(_finite("tolerance", tolerance))
    if tol < 0:
        raise ValueError("tolerance must not be negative")
    if (inputs == "calls") != (discount_factors is not None):
        raise ValueError("discount_factors is one per maturity for 'calls' and None otherwise")
    g = _rows("grid", grid, v.shape)
    if inputs == "calls":
        df = positive("discount_factors", _finite("discount_factors", discount_factors)).ravel()
        if df.size != times.size:
            raise ValueError("discount_factors must hold one factor per maturity")
        v = v / df[:, None]
    bounds = np.zeros(v.shape, dtype=bool)
    monotonic = np.zeros(v.shape, dtype=bool)
    butterfly = np.zeros(v.shape, dtype=bool)
    calendar = np.zeros(v.shape, dtype=bool)
    density = np.full(v.shape, np.nan)
    if inputs == "total_variance":
        bounds = v < -tol
        w_k, w_kk = _uneven_derivatives(v, g)
        with np.errstate(divide="ignore", invalid="ignore"):
            density = (1.0 - g * w_k / (2.0 * v)) ** 2 - 0.25 * w_k * w_k * (1.0 / v + 0.25) + 0.5 * w_kk
        butterfly = np.nan_to_num(density, nan=0.0) < -tol
        level, k_rows = v, g
    else:
        k = positive("grid", g)
        bounds = (v < np.maximum(f[:, None] - k, 0.0) - tol) | (v > f[:, None] + tol)
        slope = np.diff(v, axis=1) / np.diff(k, axis=1)
        monotonic[:, 1:] = (slope > tol) | (slope < -1.0 - tol)
        density[:, 1:-1] = 2.0 * np.diff(slope, axis=1) / (k[:, 2:] - k[:, :-2])
        butterfly[:, 1:-1] = np.diff(slope, axis=1) < -tol
        level, k_rows = v / f[:, None], k / f[:, None]
    for j in range(1, v.shape[0]):
        inside = (k_rows[j - 1] >= k_rows[j, 0]) & (k_rows[j - 1] <= k_rows[j, -1])
        later = np.interp(k_rows[j - 1], k_rows[j], level[j])
        scale = 1.0 if inputs == "total_variance" else 1.0 / f[j - 1]
        calendar[j - 1] = inside & (later < level[j - 1] - tol * scale)
    found = bool(not (bounds.any() or monotonic.any() or butterfly.any() or calendar.any()))
    return Result({"arbitrage_free": found, "bounds": bounds, "monotonic": monotonic, "butterfly": butterfly,
                   "calendar": calendar, "density": density})


def _uneven_derivatives(values, grid):
    """First and second derivatives along each row by three-point central differences on its own uneven grid, NaN at
    the ends."""
    lo, hi = np.diff(grid, axis=1)[:, :-1], np.diff(grid, axis=1)[:, 1:]
    f0, f1, f2 = values[:, :-2], values[:, 1:-1], values[:, 2:]
    first, second = np.full(values.shape, np.nan), np.full(values.shape, np.nan)
    first[:, 1:-1] = (lo * lo * f2 - hi * hi * f0 + (hi * hi - lo * lo) * f1) / (lo * hi * (lo + hi))
    second[:, 1:-1] = 2.0 * (lo * f2 - (lo + hi) * f1 + hi * f0) / (lo * hi * (lo + hi))
    return first, second
