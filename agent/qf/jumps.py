"""Jump diffusion (Merton jump): option prices by the Poisson-weighted Black-Scholes sum, a return series' MLE fit.

`import qf`. Merton's (1976) model dS/S = (mu - lambda k) dt + vol dW + (J - 1) dN: N Poisson of intensity lambda a
year, ln J ~ N(jump_mean, jump_vol^2), k = E[J - 1] = e^{jump_mean + jump_vol^2/2} - 1. Rates continuously
compounded and annual, times in years. Merton's structural credit model (equity as a call on the assets) is
qf.merton_calibrate on `credit`, not this.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

from ._common import Result, choice, integer, out, positive
from .options import OPTIONS, bs_price
from .primitives import _guard, _stream

__all__ = ["merton_jump_fit", "merton_jump_price"]

FIT_DRIFTS = ("fit",)
PARAMETERS = ("vol", "intensity", "jump_mean", "jump_vol")


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('maturity',), price=('spot', 'strike',))
def merton_jump_price(option, spot, strike, maturity, rate, dividend_yield, vol, intensity, jump_mean, jump_vol,
                      terms):
    """Merton jump-diffusion option price: the Poisson-weighted sum of Black-Scholes prices (lognormal jumps).

    With k = e^{jump_mean + jump_vol^2/2} - 1 and lambda' = lambda (1 + k): V = sum_{n=0}^{N} e^{-lambda' T}
    (lambda' T)^n / n! BS(S, K, T, r_n, q, vol_n), vol_n^2 = vol^2 + n jump_vol^2 / T, r_n = r - lambda k + n ln(1 +
    k) / T, the n-th term the price given n jumps. The weights are lambda' T's, not lambda T's: the two agree only
    where k = 0. intensity 0 (or jump_vol and jump_mean 0) gives qf.bs_price; put-call parity holds as under
    Black-Scholes. The sum stops at n = terms; a few times lambda' T terms past lambda' T leave a negligible tail.

    conventions:
    - option: 'call' or 'put'; spot S, strike K; rate r, dividend_yield q continuously compounded and annual;
      maturity T in years
    - vol: the diffusion's annual vol; intensity: lambda, jumps a year; jump_mean, jump_vol: ln J's mean and std
    - terms: N, the last n summed, an int of at least 0
    returns: the price today, a float or an array
    source: Merton (1976), "Option pricing when underlying stock returns are discontinuous", J. Financial Economics
      3; Hull, Options, Futures, and Other Derivatives, "Merton's mixed jump-diffusion model"
    aliases: Merton jump diffusion, Merton jump-diffusion model, Merton jump model, Merton jump, jump-diffusion option price, Merton 1976, Merton series, Poisson-weighted sum, [jump diffusion], [lognormal jumps]
    example:
        qf.merton_jump_price(option="call", spot=100.0, strike=100.0, maturity=0.75, rate=0.04, dividend_yield=0.0,
                             vol=0.2, intensity=1.0, jump_mean=-0.1, jump_vol=0.15,
                             terms=40)
    """
    choice("option", option, OPTIONS)
    n_max = integer("terms", terms, 0)
    t = positive("maturity", maturity)
    lam, delta = np.asarray(intensity, dtype=float), np.asarray(jump_vol, dtype=float)
    if np.any(lam < 0) or np.any(delta < 0):
        raise ValueError("intensity and jump_vol must not be negative")
    k = np.expm1(np.asarray(jump_mean, dtype=float) + 0.5 * delta * delta)
    lam_prime = lam * (1.0 + k) * t
    total = 0.0
    for n in range(n_max + 1):
        weight = stats.poisson.pmf(n, lam_prime)
        vol_n = np.sqrt(np.asarray(vol, dtype=float) ** 2 + n * delta * delta / t)
        rate_n = np.asarray(rate, dtype=float) - lam * k + n * np.log1p(k) / t
        total = total + weight * np.asarray(bs_price(option, spot, strike, t, rate_n, dividend_yield, vol_n))
    return out(total)


def _log_density(x, dt, drift, vol, lam, mu_j, delta, n_max):
    """ln sum_n Poisson(n; lambda dt) phi(x; drift dt + n mu_j, vol^2 dt + n delta^2), by log-sum-exp."""
    n = np.arange(n_max + 1)[:, None]
    log_weight = -lam * dt + n * math.log(lam * dt) - gammaln(n + 1) if lam > 0 else np.where(n == 0, 0.0, -np.inf)
    terms = log_weight + stats.norm.logpdf(x[None, :], drift * dt + n * mu_j, np.sqrt(vol * vol * dt + n * delta ** 2))
    top = terms.max(axis=0)
    return top + np.log(np.exp(terms - top).sum(axis=0))


def merton_jump_fit(returns, dt, terms, drift, bounds, starts, rng):
    """Merton jump-diffusion fit of a log-return series by maximum likelihood (a Poisson mixture of normals).

    Each period's log return x has the density p(x) = sum_{n=0}^{N} e^{-lambda dt} (lambda dt)^n / n! phi(x; m dt +
    n jump_mean, vol^2 dt + n jump_vol^2), m the diffusion's log drift a year; the fit maximises sum ln p(x_i) by
    L-BFGS-B within the bounds from each start, and keeps the highest. drift 'fit' estimates m; a number mu holds the
    price's expected return a year fixed, m = mu - vol^2/2 - lambda k, k = e^{jump_mean + jump_vol^2/2} - 1 (a sample
    estimate of mu is the caller's to choose). intensity 0 gives the normal likelihood. Prices under the fit, with
    the risk-neutral rate for the drift: qf.merton_jump_price.

    conventions:
    - returns: log returns ln(S_t/S_{t-1}), oldest first; dt: one period in years (1/252 for daily)
    - terms: N, the most jumps a period summed (an int of at least 1)
    - drift: 'fit' or a number mu (the annual expected return of the price, held fixed)
    - bounds: a dict of (low, high) for 'vol', 'intensity', 'jump_mean', 'jump_vol' (and 'drift' under 'fit'), each
      parameter annual but the jump's
    - starts: the random starts, an int of at least 1, each uniform within the bounds, drawn in the parameters' order
    - rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream)
    returns: a dict: vol; intensity; jump_mean; jump_vol; drift (m); log_likelihood; total_vol (sqrt(vol^2 +
      lambda (jump_mean^2 + jump_vol^2)), the annual vol of the returns); converged
    source: Merton (1976), J. Financial Economics 3; Ball and Torous (1985), J. Finance 40(1) (the likelihood);
      Press (1967), J. Business 40(3)
    aliases: Merton jump diffusion, Merton jump-diffusion model, Merton jump, Merton jump diffusion calibration, jump diffusion calibration, jump-diffusion MLE, jump diffusion likelihood, Merton jump likelihood, Merton jump fit, [jump diffusion], [Poisson mixture of normals]
    example:
        import numpy as np, qf
        x = np.random.default_rng(3).standard_t(4, 1000) * 0.01
        qf.merton_jump_fit(returns=x, dt=1 / 252, terms=8, drift="fit", bounds={"vol": (0.05, 0.8), "intensity": (0.0, 30.0), "jump_mean": (-0.2, 0.2), "jump_vol": (0.01, 0.25), "drift": (-1.0, 1.0)}, starts=2, rng=1)
    """
    x = np.asarray(returns, dtype=float).ravel()
    if x.size < 10 or not np.all(np.isfinite(x)):
        raise ValueError("returns must hold ten or more finite log returns")
    step, n_max = float(positive("dt", dt)), integer("terms", terms, 1)
    n_starts = integer("starts", starts, 1)
    if isinstance(drift, str):
        choice("drift", drift, FIT_DRIFTS)
        names = (*PARAMETERS, "drift")
    else:
        names, mu = PARAMETERS, float(drift)
    if not isinstance(bounds, dict) or set(bounds) != set(names):
        raise ValueError(f"bounds must be a dict of (low, high) for exactly {', '.join(names)}")
    box = [tuple(float(b) for b in bounds[name]) for name in names]
    if any(not low < high for low, high in box) or box[0][0] <= 0 or box[1][0] < 0 or box[3][0] <= 0:
        raise ValueError("each bound must be (low, high) with low < high; vol and jump_vol above 0, intensity >= 0")
    rng = _stream(rng)

    def unpacked(theta):
        vol, lam, mu_j, delta = theta[:4]
        m = theta[4] if len(names) == 5 else mu - 0.5 * vol * vol - lam * math.expm1(mu_j + 0.5 * delta * delta)
        return vol, lam, mu_j, delta, m

    def loss(theta):
        vol, lam, mu_j, delta, m = unpacked(theta)
        return -float(_log_density(x, step, m, vol, lam, mu_j, delta, n_max).sum())

    best = None
    for _ in range(n_starts):
        start = np.array([rng.uniform(low, high) for low, high in box])
        found = optimize.minimize(loss, start, method="L-BFGS-B", bounds=box)
        if best is None or found.fun < best.fun:
            best = found
    vol, lam, mu_j, delta, m = unpacked(best.x)
    return Result({"vol": float(vol), "intensity": float(lam), "jump_mean": float(mu_j), "jump_vol": float(delta),
                   "drift": float(m), "log_likelihood": float(-best.fun),
                   "total_vol": float(math.sqrt(vol * vol + lam * (mu_j * mu_j + delta * delta))),
                   "converged": bool(best.success)})
