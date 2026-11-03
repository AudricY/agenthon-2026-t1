"""Fourier pricing from a characteristic function (Gil-Pelaez, FFT, COS); Heston, Bates and Merton jump models.

`import qf`. A characteristic function is phi(u) = E[exp(i u ln S_T)] under the pricing measure, a callable
of a complex numpy array (the pricer also takes it at u - i). Rates are continuously compounded and annual, times in
years. Several results come back as a dict with named keys.
"""
from __future__ import annotations

import functools

import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from ._common import Result, choice, out, positive
from .impvol import implied_vol
from .primitives import _guard

__all__ = ["charfn_price", "fft_price", "heston_calibrate", "heston_charfn", "jump_charfn"]

OPTIONS = ("call", "put")
QUADRATURES = ("gauss_legendre", "quad")
LOSSES = ("price", "implied_vol")
FELLER = ("free", "enforce")
HESTON = ("v0", "kappa", "theta", "sigma", "rho")
FFT_METHODS = ("carr_madan", "cos")
FFT_WEIGHTS = ("simpson", "trapezoid")
CUMULANT_STEP = 1e-3            # the central differences that read ln S_T's first two cumulants off the charfn
NO_VOL = 1.0                    # the residual of a model price with no implied vol (outside the bounds), in vol
FORWARD_TOLERANCE = 1e-6        # phi(-i) against S e^{(r-q)T}, relative: a wrong drift is refused, not priced


@functools.lru_cache(maxsize=32)
def _gauss_legendre(count):
    """Gauss-Legendre nodes and weights on (-1, 1), kept: each count's eigenproblem is solved once."""
    x, w = np.polynomial.legendre.leggauss(count)
    x.flags.writeable = w.flags.writeable = False
    return x, w


def _nodes(nodes):
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer)) or nodes < 2:
        raise ValueError("nodes must be an int of at least 2")
    return int(nodes)


@_guard(rate=('rate', 'dividend_yield',), time=('maturity',), price=('spot', 'strike',))
def charfn_price(charfn, option, spot, strike, maturity, rate, dividend_yield, quadrature, upper, nodes):
    """European option price from the characteristic function of ln S_T, by Gil-Pelaez inversion.

    P2 = 1/2 + (1/pi) int_0^upper Re[e^{-iu ln K} phi(u) / (iu)] du, P(S_T > K) under the pricing measure; P1 the
    same with phi(u - i) / phi(-i), under the share measure. call = S e^{-qT} P1 - K e^{-rT} P2, put = K e^{-rT}
    (1 - P2) - S e^{-qT} (1 - P1). phi(-i) = E[S_T] must be the forward S e^{(r-q)T}: a charfn whose drift is not is
    refused. A model of several variance factors is a product of characteristic functions (qf.heston_charfn).

    conventions:
    - charfn: phi(u) of a complex array u, elementwise, spot and maturity inside it (one maturity a call)
    - quadrature: 'gauss_legendre' (nodes points on (0, upper), phi taken once for every strike) or 'quad'
      (scipy.integrate.quad on (0, upper) with at most nodes subintervals, a strike at a time)
    - upper: where the integral is cut (200 suits annual parameters; phi must have decayed there)
    - spot, maturity, rate, dividend_yield: numbers, as bs_price; strike a number or an array
    returns: a dict: price; p1 and p2, the call's exercise probabilities under the share and the pricing measure;
      each a float, or an array shaped like strike
    source: Gil-Pelaez (1951), Biometrika 38; Heston (1993), Review of Financial Studies 6(2); Gatheral (2006), ch. 2
    aliases: characteristic function pricing, Gil-Pelaez, Fourier inversion, Fourier pricing, characteristic function
    example:
        qf.charfn_price(charfn=lambda u: qf.heston_charfn(u, 100.0, 1.0, 0.03, 0.0, 0.04, 1.5, 0.04,
                        0.5, -0.7, False), option="call", spot=100.0, strike=[90.0, 100.0], maturity=1.0,
                        rate=0.03, dividend_yield=0.0, quadrature="gauss_legendre", upper=200.0, nodes=256)
    """
    choice("option", option, OPTIONS)
    choice("quadrature", quadrature, QUADRATURES)
    if not callable(charfn):
        raise ValueError("charfn must be a callable phi(u) of a complex numpy array")
    s, t = float(positive("spot", spot)), float(positive("maturity", maturity))
    k = positive("strike", strike)
    r, q = float(rate), float(dividend_yield)
    cut, count = float(positive("upper", upper)), _nodes(nodes)
    forward = complex(np.asarray(charfn(np.array([-1j])), dtype=complex).ravel()[0])
    expected = s * np.exp((r - q) * t)
    if not abs(forward - expected) <= FORWARD_TOLERANCE * expected:
        raise ValueError(f"charfn(-i) = E[S_T] is {forward:.6g}, not the forward S e^((r-q)T) = {expected:.6g}: "
                         "its drift or its spot is not the pricer's")
    log_k = np.log(k).ravel()
    if quadrature == "gauss_legendre":
        x, w = _gauss_legendre(count)
        u, w = 0.5 * cut * (x + 1.0), 0.5 * cut * w
        phi2 = np.asarray(charfn(u.astype(complex)), dtype=complex)
        phi1 = np.asarray(charfn(u - 1j), dtype=complex) / forward
        kernel = np.exp(-1j * np.multiply.outer(log_k, u)) / (1j * u)
        p1, p2 = 0.5 + (kernel * phi1).real @ w / np.pi, 0.5 + (kernel * phi2).real @ w / np.pi
    else:
        def probability(shift, scale, lk):
            def integrand(v):
                at = np.array([v - shift])
                return float((np.exp(-1j * v * lk) * np.asarray(charfn(at), dtype=complex)[0] / scale / (1j * v)).real)
            return 0.5 + integrate.quad(integrand, 0.0, cut, limit=count)[0] / np.pi

        p1 = np.array([probability(1j, forward, lk) for lk in log_k])
        p2 = np.array([probability(0.0, 1.0, lk) for lk in log_k])
    p1, p2 = p1.reshape(k.shape), p2.reshape(k.shape)
    share, cash = s * np.exp(-q * t), k * np.exp(-r * t)
    price = share * p1 - cash * p2 if option == "call" else cash * (1.0 - p2) - share * (1.0 - p1)
    return Result({"price": out(price), "p1": out(p1), "p2": out(p2)})


@_guard(rate=('rate', 'dividend_yield',), time=('maturity',), price=('spot',))
def heston_charfn(u, spot, maturity, rate, dividend_yield, v0, kappa, theta, sigma, rho, factor_only):
    """Heston's characteristic function of ln S_T (stable form), or its variance factor alone.

    xi = kappa - sigma rho i u, d = sqrt(xi^2 + sigma^2 (u^2 + i u)), g = (xi - d)/(xi + d);
    C = (kappa theta / sigma^2) [(xi - d) T - 2 ln((1 - g e^{-dT}) / (1 - g))], D = (xi - d)/sigma^2 (1 - e^{-dT}) /
    (1 - g e^{-dT}); phi(u) = exp(i u (ln S + (r - q) T) + C + D v0), and the factor alone exp(C + D v0).
    This is Albrecher et al.'s form of Heston's: the same function, with no jump of the complex logarithm's branch at
    long maturities. dv = kappa (theta - v) dt + sigma sqrt(v) dW_v, d ln S = (r - q - v/2) dt + sqrt(v) dW_S, corr
    rho. A two-factor model (Christoffersen, Heston and Jacobs) is one full charfn times the other's factor alone.

    Feller: 2 kappa theta >= sigma^2 keeps v from 0. E[v_t] = theta + (v0 - theta) e^{-kappa t}; Var[v_t] = v0 sigma^2
    (e^{-kappa t} - e^{-2 kappa t}) / kappa + theta sigma^2 (1 - e^{-kappa t})^2 / (2 kappa); the variance-swap strike
    (annual variance over T) is theta + (v0 - theta)(1 - e^{-kappa T}) / (kappa T).

    conventions:
    - u: a number or a complex numpy array
    - v0, theta: variances (a vol of 20% is 0.04); kappa per year; sigma the vol of variance, positive; rho in [-1, 1]
    - factor_only: False for phi of ln S_T; True for the variance factor exp(C + D v0) alone, spot, rate and
      dividend_yield then unused
    returns: phi(u), a complex number or a complex array shaped like u
    source: Heston (1993), Review of Financial Studies 6(2); Albrecher, Mayer, Schoutens and Tistaert (2007), Wilmott
    aliases: Heston, Heston model, Heston characteristic function, stochastic volatility model, two-factor Heston
    example:
        qf.heston_charfn(u=[0.5, 1.0], spot=100.0, maturity=1.0, rate=0.03, dividend_yield=0.0,
                         v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7,
                         factor_only=False)
    """
    z = np.asarray(u, dtype=complex)
    t = float(positive("maturity", maturity))
    kappa, sigma = float(positive("kappa", kappa)), float(positive("sigma", sigma))
    v0, theta, rho = float(v0), float(theta), float(rho)
    if v0 < 0 or theta < 0 or not -1.0 <= rho <= 1.0:
        raise ValueError("v0 and theta must not be negative and rho must lie in [-1, 1]")
    if not isinstance(factor_only, (bool, np.bool_)):
        raise ValueError("factor_only must be True or False")
    iu = 1j * z
    xi = kappa - sigma * rho * iu
    d = np.sqrt(xi * xi + sigma * sigma * (z * z + iu))
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (xi - d) / (xi + d)
        decay = np.exp(-d * t)
        c = kappa * theta / sigma ** 2 * ((xi - d) * t - 2.0 * np.log((1.0 - g * decay) / (1.0 - g)))
        dv = (xi - d) / sigma ** 2 * (1.0 - decay) / (1.0 - g * decay)
    flat = np.isclose(z * z + iu, 0.0, rtol=0.0, atol=1e-300)     # u = 0 and u = -i: the factor is 1 exactly
    factor = np.where(flat, 1.0 + 0j, np.exp(np.where(flat, 0.0, c + dv * v0)))
    if factor_only:
        return complex(factor) if factor.ndim == 0 else factor
    drift = iu * (np.log(float(positive("spot", spot))) + (float(rate) - float(dividend_yield)) * t)
    found = np.exp(drift) * factor
    return complex(found) if found.ndim == 0 else found


JUMP_MODELS = {"merton": ("vol", "intensity", "jump_mean", "jump_vol"),
               "bates": ("v0", "kappa", "theta", "sigma", "rho", "intensity", "jump_mean", "jump_vol"),
               "variance_gamma": ("sigma", "nu", "theta")}


@_guard(rate=('rate', 'dividend_yield',), time=('maturity',), price=('spot',))
def jump_charfn(u, model, spot, maturity, rate, dividend_yield, parameters):
    """Jump models' characteristic function: Merton, Bates, variance gamma.

    phi(u) = E[e^{iu ln S_T}], the drift compensated so that E[S_T] = S e^{(r - q) T} (qf.charfn_price checks it):
    'merton': exp(iu (ln S + (r - q - lambda k - vol^2/2) T) - vol^2 u^2 T/2 + lambda T (e^{iu m - d^2 u^2/2} - 1)),
      k = e^{m + d^2/2} - 1, jumps ln J ~ N(m, d^2) at intensity lambda a year;
    'bates': Heston's (qf.heston_charfn) times exp(lambda T (e^{iu m - d^2 u^2/2} - 1) - iu lambda k T);
    'variance_gamma' (Madan, Carr and Chang): exp(iu (ln S + (r - q + omega) T)) (1 - iu theta nu + sigma^2 nu u^2/2)^{-T/nu},
      omega = ln(1 - theta nu - sigma^2 nu / 2) / nu.
    Priced by qf.charfn_price or qf.fft_price; Merton's equals qf.merton_jump_price; no jumps give Black-Scholes and
    Heston.

    conventions:
    - u: a number or a complex numpy array; model: 'merton', 'bates' or 'variance_gamma'
    - parameters: a dict, by model: merton vol, intensity, jump_mean, jump_vol; bates v0, kappa, theta, sigma, rho
      (Heston's, variances) and intensity, jump_mean, jump_vol; variance_gamma sigma, nu (the gamma clock's variance
      rate), theta (its drift); intensity lambda in jumps a year, jump_mean and jump_vol of ln J
    - spot S; maturity T in years; rate r and dividend_yield q continuously compounded
    returns: phi(u), a complex number or a complex array shaped like u
    source: Merton (1976), J. Financial Economics 3; Bates (1996), Review of Financial Studies 9(1); Madan, Carr and
      Chang (1998), European Finance Review 2; Gatheral (2006), ch. 5
    aliases: Bates, Bates model, variance gamma, VG, SVJ, [Bates charfn]
    example:
        import qf
        qf.jump_charfn(u=[0.5, 1.0], model="variance_gamma", spot=100.0, maturity=1.0, rate=0.05,
                       dividend_yield=0.0, parameters={"sigma": 0.2, "nu": 0.2, "theta": -0.15})  # result: [-0.67497246+0.73031234j, -0.07644289-0.97506130j]
    """
    choice("model", model, tuple(JUMP_MODELS))
    names = JUMP_MODELS[model]
    if not isinstance(parameters, dict) or set(parameters) != set(names):
        raise ValueError(f"parameters must be a dict of exactly {', '.join(names)} for {model!r}")
    p = {name: float(parameters[name]) for name in names}
    z = np.asarray(u, dtype=complex)
    s, t = float(positive("spot", spot)), float(positive("maturity", maturity))
    iu, carry = 1j * z, float(rate) - float(dividend_yield)
    if model == "variance_gamma":
        sigma, nu, theta = p["sigma"], p["nu"], p["theta"]
        if not (sigma > 0 and nu > 0) or not 1.0 - theta * nu - 0.5 * sigma * sigma * nu > 0:
            raise ValueError("variance gamma needs sigma > 0, nu > 0 and 1 - theta nu - sigma^2 nu / 2 > 0")
        omega = np.log(1.0 - theta * nu - 0.5 * sigma * sigma * nu) / nu
        found = np.exp(iu * (np.log(s) + (carry + omega) * t)) * (1.0 - iu * theta * nu + 0.5 * sigma * sigma * nu
                                                                  * z * z) ** (-t / nu)
        return complex(found) if found.ndim == 0 else found
    lam, mean, spread = p["intensity"], p["jump_mean"], p["jump_vol"]
    if lam < 0 or spread < 0:
        raise ValueError("intensity and jump_vol must not be negative")
    k = np.expm1(mean + 0.5 * spread * spread)
    jumps = np.exp(lam * t * (np.exp(iu * mean - 0.5 * spread * spread * z * z) - 1.0) - iu * lam * k * t)
    if model == "merton":
        vol = float(positive("vol", p["vol"]))
        found = np.exp(iu * (np.log(s) + (carry - 0.5 * vol * vol) * t) - 0.5 * vol * vol * z * z * t) * jumps
    else:
        found = heston_charfn(z, s, t, rate, dividend_yield, p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"],
                              False) * jumps
    return complex(found) if np.ndim(found) == 0 else found


@_guard(rate=('rate', 'dividend_yield',), time=('maturities',), price=('spot', 'strikes',))
def heston_calibrate(options, strikes, maturities, market, spot, rate, dividend_yield, loss, weights, initial, bounds,
                     feller, upper, nodes):
    """Heston calibration to a smile: least squares over (v0, kappa, theta, sigma, rho).

    Minimises sum_i w_i (model_i - market_i)^2 by scipy.optimize.least_squares ('trf', within the bounds), each
    model price by qf.charfn_price on qf.heston_charfn (Gauss-Legendre); 'implied_vol' compares
    Black-Scholes implied vols, the model's by qf.implied_vol's bisection (a price with none counts 1.0 off).
    'enforce' fits sigma as s sqrt(2 kappa theta), s in (0, 1], so Feller's 2 kappa theta >= sigma^2 holds (sigma's
    own bounds unused); 'free' does not. Real smiles have local minima: the start matters (Cui et al. 2017).

    conventions:
    - options: 'call' or 'put', or one per quote; strikes, maturities: one per quote (years)
    - market: one per quote, prices for loss 'price', Black-Scholes implied vols for 'implied_vol'
    - spot, rate, dividend_yield: as qf.bs_price; weights: one per quote (1/vega^2 makes price errors vol errors),
      or 'equal'
    - initial: a dict of v0, kappa, theta, sigma, rho; bounds: a dict of each one's (low, high)
    - feller: 'free' or 'enforce'; upper, nodes: the quadrature, as charfn_price (200 and 128 suit annual data)
    returns: a dict: v0, kappa, theta, sigma, rho; rmse (unweighted, in the loss's units); converged (the optimiser's
      flag); feller (2 kappa theta >= sigma^2); model (one per quote, prices or vols)
    source: Heston (1993); Gatheral (2006), ch. 3; Cui, del Bano Rollin and Germano (2017), EJOR 263(2)
    aliases: Heston calibration, calibrate Heston, Heston model calibration, stochastic volatility calibration, fit Heston to the smile, Heston parameters
    example:
        qf.heston_calibrate(options="call", strikes=[90.0, 100.0, 110.0, 90.0, 100.0, 110.0], maturities=[0.5] * 3 + [1.0] * 3,
            market=[13.19, 6.37, 2.08, 15.6, 9.1, 4.41], spot=100.0, rate=0.02, dividend_yield=0.0, loss="price", weights="equal",
            initial=dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.6), bounds=dict(v0=(1e-4, 1), kappa=(1e-3, 10), theta=(1e-4, 1), sigma=(1e-3, 3), rho=(-0.99, 0.99)), feller="free", upper=200.0, nodes=128)
    """
    choice("loss", loss, LOSSES)
    choice("feller", feller, FELLER)
    k = positive("strikes", strikes).ravel()
    t = positive("maturities", maturities).ravel()
    target = np.asarray(market, dtype=float).ravel()
    kinds = np.full(k.size, options, dtype=object) if isinstance(options, str) else np.asarray(options, dtype=object)
    if not (k.size == t.size == target.size == kinds.size) or not np.all(np.isfinite(target)):
        raise ValueError("options (or one for all), strikes, maturities and market must hold one entry per quote")
    for kind in set(kinds):
        choice("options", kind, OPTIONS)
    w = np.ones(k.size) if isinstance(weights, str) and weights == "equal" else positive("weights", weights).ravel()
    if w.size != k.size:
        raise ValueError("weights must be 'equal' or one positive weight per quote")
    if not (isinstance(initial, dict) and set(initial) == set(HESTON) and isinstance(bounds, dict)
            and set(bounds) == set(HESTON)):
        raise ValueError(f"initial and bounds must be dicts of exactly {', '.join(HESTON)}")
    s, r, q = float(positive("spot", spot)), float(rate), float(dividend_yield)
    cut, count = float(positive("upper", upper)), _nodes(nodes)
    groups = [(maturity, kind, (t == maturity) & (kinds == kind)) for maturity in np.unique(t) for kind in OPTIONS
              if np.any((t == maturity) & (kinds == kind))]

    def heston(x):
        return Result(zip(HESTON, x if feller == "free" else (*x[:3], x[3] * np.sqrt(2 * x[1] * x[2]), x[4])))

    def model(x):
        p = heston(x)
        found = np.empty(k.size)
        for maturity, kind, chosen in groups:
            def charfn(u, maturity=maturity):
                return heston_charfn(u, s, maturity, r, q, p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"],
                                     False)
            found[chosen] = charfn_price(charfn, kind, s, k[chosen], maturity, r, q, "gauss_legendre", cut,
                                         count)["price"]
        if loss == "implied_vol":
            vols = np.empty(k.size)
            for kind in set(kinds):
                chosen = kinds == kind
                vols[chosen] = implied_vol(kind, found[chosen], s, k[chosen], t[chosen], r, q, "bisection")
            found = np.where(np.isfinite(vols), vols, target + NO_VOL)
        return found

    low = np.array([float(bounds[name][0]) for name in HESTON])
    high = np.array([float(bounds[name][1]) for name in HESTON])
    start = np.array([float(initial[name]) for name in HESTON])
    if feller == "enforce":
        low[3], high[3] = 1e-6, 1.0
        start[3] = min(max(start[3] / np.sqrt(2 * start[1] * start[2]), 1e-6), 1.0)
    if np.any(low >= high) or np.any(start < low) or np.any(start > high) or low[4] < -1 or high[4] > 1:
        raise ValueError("each bound must be (low, high) with low < high, rho's within [-1, 1], and initial inside")
    root = np.sqrt(w)
    fit = least_squares(lambda x: root * (model(x) - target), start, bounds=(low, high), method="trf", xtol=1e-15,
                        ftol=1e-15, gtol=1e-15, max_nfev=5000)
    found = heston(fit.x)
    fitted = model(fit.x)
    return Result({**{name: float(value) for name, value in found.items()},
                   "rmse": float(np.sqrt(np.mean((fitted - target) ** 2))), "converged": bool(fit.success),
                   "feller": bool(2 * found["kappa"] * found["theta"] >= found["sigma"] ** 2 * (1 - 1e-12)),
                   "model": fitted})


def _carr_madan(charfn, log_s, log_k, t, r, points, spacing, damping, weights):
    """Call prices at log strikes by Carr and Madan's FFT, the grid centred on ln S, a cubic spline between."""
    v = spacing * np.arange(points)
    step = 2.0 * np.pi / (points * spacing)
    start = log_s - 0.5 * points * step
    psi = (np.exp(-r * t) * np.asarray(charfn(v - (damping + 1.0) * 1j), dtype=complex)
           / (damping * damping + damping - v * v + 1j * (2.0 * damping + 1.0) * v))
    if weights == "simpson":
        w = spacing / 3.0 * (3.0 + (-1.0) ** (np.arange(points) + 1))
        w[0] = spacing / 3.0
    else:
        w = np.full(points, spacing)
        w[0] = spacing / 2.0
    grid = start + step * np.arange(points)
    calls = np.exp(-damping * grid) / np.pi * np.fft.fft(np.exp(-1j * v * start) * psi * w).real
    if np.any(log_k < grid[1]) or np.any(log_k > grid[-2]):
        raise ValueError("a strike lies off the FFT's log-strike grid: widen it (more points or a smaller spacing)")
    return CubicSpline(grid, calls)(log_k)


def _cos_puts(charfn, spot, strikes, t, r, points, truncation):
    """Put prices by Fang and Oosterlee's cosine expansion of ln(S_T / S)'s density on c1 +- L sqrt(c2)."""
    log_s = np.log(spot)

    def phi(u):
        return np.asarray(charfn(np.asarray(u, dtype=complex)), dtype=complex) * np.exp(-1j * np.asarray(u) * log_s)

    h = CUMULANT_STEP
    up, down = np.log(phi(np.array([h, -h])))
    c1 = (up.imag - down.imag) / (2.0 * h)
    c2 = -(up.real + down.real) / (h * h)
    if not c2 > 0:
        raise ValueError("the charfn's variance of ln S_T, read off it, is not positive")
    a, b = c1 - truncation * np.sqrt(c2), c1 + truncation * np.sqrt(c2)
    u = np.arange(points) * np.pi / (b - a)
    terms = (phi(u) * np.exp(-1j * u * a)).real
    terms[0] *= 0.5
    d = np.clip(np.log(strikes / spot), a, b)[:, None]
    span = d - a
    chi = (np.cos(u * span) * np.exp(d) - np.exp(a) + u * np.sin(u * span) * np.exp(d)) / (1.0 + u * u)
    psi = np.where(u > 0, np.sin(u * span) / np.where(u > 0, u, 1.0), span)
    return np.exp(-r * t) * 2.0 / (b - a) * ((strikes[:, None] * psi - spot * chi) @ terms)


@_guard(rate=('rate', 'dividend_yield',), time=('maturity',), price=('spot', 'strike',))
def fft_price(charfn, option, spot, strike, maturity, rate, dividend_yield, method, points, spacing, damping,
              weights, truncation):
    """European prices from a characteristic function by Carr-Madan's FFT or the COS method.

    phi(u) = E[e^{iu ln S_T}], as qf.charfn_price takes it. 'carr_madan': the damped call's transform
    psi(v) = e^{-rT} phi(v - (alpha + 1) i) / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v), summed by FFT on v_j =
    eta j with Simpson's or the trapezoid's weights onto log strikes ln S + lambda (u - N/2), lambda eta = 2 pi / N,
    then C(k) = e^{-alpha k} / pi sum, a cubic spline in ln K between the grid's points. 'cos': the density of
    ln(S_T/S) as a cosine series of N terms on [a, b] = c1 +- L sqrt(c2), c1 and c2 its cumulants read off phi; a put
    priced by the series (Fang and Oosterlee 2008). The other option by put-call parity in both. Each converges to
    charfn_price's Gil-Pelaez price and, for a lognormal phi, to bs_price.

    conventions:
    - charfn: phi of ln S_T (spot and maturity inside it), of a complex numpy array; option: 'call' or 'put'
    - spot, strike (a number or an array), maturity, rate, dividend_yield: as qf.bs_price
    - method: 'carr_madan' (points N a power of 2, spacing eta, damping alpha above 0, weights 'simpson' or
      'trapezoid'; truncation None) or 'cos' (points the terms N, truncation L; spacing, damping, weights None)
    returns: the prices, a float or an array shaped like strike
    source: Carr and Madan (1999), J. Computational Finance 2(4); Fang and Oosterlee (2008), SIAM J. Sci. Comput. 31(2)
    aliases: Carr-Madan, FFT option pricing, fast Fourier transform pricing, COS method, Fourier-cosine, Fang-Oosterlee, damped call transform
    example:
        qf.fft_price(charfn=lambda u: qf.heston_charfn(u, 100.0, 1.0, 0.03, 0.0, 0.04, 1.5, 0.04, 0.5, -0.7, False), option="call",
            spot=100.0, strike=[90.0, 100.0, 110.0], maturity=1.0, rate=0.03, dividend_yield=0.0, method="cos", points=256, spacing=None,
            damping=None, weights=None, truncation=20.0)
    """
    choice("option", option, OPTIONS)
    choice("method", method, FFT_METHODS)
    if not callable(charfn):
        raise ValueError("charfn must be a callable phi(u) of a complex numpy array")
    s, t = float(positive("spot", spot)), float(positive("maturity", maturity))
    k = positive("strike", strike)
    r, q = float(rate), float(dividend_yield)
    n = _nodes(points)
    flat = k.ravel()
    if method == "carr_madan":
        if truncation is not None:
            raise ValueError("truncation is for 'cos': pass None")
        choice("weights", weights, FFT_WEIGHTS)
        if n & (n - 1):
            raise ValueError("points must be a power of 2 for the FFT")
        eta, alpha = float(positive("spacing", spacing)), float(positive("damping", damping))
        calls = _carr_madan(charfn, np.log(s), np.log(flat), t, r, n, eta, alpha, weights)
        prices = calls if option == "call" else calls - s * np.exp(-q * t) + flat * np.exp(-r * t)
    else:
        if spacing is not None or damping is not None or weights is not None:
            raise ValueError("spacing, damping and weights are for 'carr_madan': pass None")
        puts = _cos_puts(charfn, s, flat, t, r, n, float(positive("truncation", truncation)))
        prices = puts if option == "put" else puts + s * np.exp(-q * t) - flat * np.exp(-r * t)
    return out(np.asarray(prices).reshape(k.shape))
