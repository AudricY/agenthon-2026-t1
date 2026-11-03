"""Simulated paths: GBM, correlated GBM, Dupire, OU, CIR, Heston, Euler and Milstein schemes, LIBOR (LMM).

`import qf`. A path array is n_paths x n_times, a row a path; times in years; `rng` a numpy.random.Generator or an
int seed (numpy.random.default_rng(seed)), or numpy's legacy stream, a numpy.random.RandomState or None (the global one
numpy.random.seed(seed) sets; from one seed the two streams draw different samples), whose draws each function names:
a seed reproduces a run. A payoff on the paths is priced by qf.mc_payoff_price; several results come back as a dict
with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from ._common import Result, choice, positive
from .primitives import _guard, _stream

__all__ = ["cir_paths", "correlated_gbm_paths", "gbm_paths", "heston_paths", "lmm_paths", "local_vol_paths",
           "ou_paths", "sde_paths"]

OU_SCHEMES = ("exact", "euler")
JUMP_COUNTS = ("poisson", "bernoulli")
DRIFT_FORMS = ("arithmetic", "log")
LOCAL_VOL_SCHEMES = ("log_euler", "euler")
CIR_SCHEMES = ("exact", "full_truncation")
HESTON_SCHEMES = ("full_truncation", "qe")
SDE_SCHEMES = ("euler", "milstein")
MEASURES = ("spot", "terminal")


def _count(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return int(value)


def _normals(rng, paths, shape, antithetic):
    """One draw of standard normals (paths, *shape), the second half mirroring the first where `antithetic`."""
    if not isinstance(antithetic, (bool, np.bool_)):
        raise ValueError("antithetic must be True or False")
    if antithetic:
        if paths % 2:
            raise ValueError("n_paths must be even for antithetic pairs")
        half = rng.standard_normal((paths // 2, *shape))
        return np.concatenate([half, -half])
    return rng.standard_normal((paths, *shape))


def _times(times):
    t = np.asarray(times, dtype=float).ravel()
    if t.size == 0 or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be increasing years, from 0 on")
    return t


@_guard(vol=('vol',), time=('times',), price=('spot',))
def gbm_paths(spot, drift, vol, times, n_paths, rng, drift_form, antithetic):
    """Monte Carlo paths of geometric Brownian motion at given times, the drift's form named.

    S(t_j) = S(t_{j-1}) exp(nu (t_j - t_{j-1}) + vol sqrt(t_j - t_{j-1}) Z_j), exact at every time, any spacing.
    'arithmetic' reads `drift` as mu in dS = mu S dt + vol S dW and sets nu = mu - vol^2/2 (risk-neutral mu = r - q);
    'log' reads it as nu, the drift of ln S with the -vol^2/2 already in. The Ito term is taken once: by this form or
    by the caller, never both. E[S_t] = S e^{mu t}; E[ln S_t] = ln S + nu t.

    conventions:
    - times: increasing years from today (only a first 0 gives the spot's column)
    - drift_form: 'arithmetic' (mu; the vol^2/2 subtracted here) or 'log' (nu = mu - vol^2/2; nothing subtracted)
    - antithetic: True mirrors the first half's normals in the second half (n_paths even), False draws them all
    - rng: a Generator or int seed (default_rng), or a RandomState or None (the legacy stream np.random.seed sets);
      one draw of normals, (n_paths, or n_paths / 2, by len(times))
    returns: an n_paths x len(times) array, column j the price at times[j]
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 3.2.1 and 4.2; Hull, ch. "Wiener processes"
    aliases: geometric Brownian motion simulation, GBM simulation, GBM paths, simulate price paths, antithetic variates
    example:
        qf.gbm_paths(spot=100.0, drift=0.05, vol=0.2, times=[0.25, 0.5, 0.75, 1.0],
                     n_paths=1000, rng=1, drift_form="arithmetic",
                     antithetic=True)
    """
    choice("drift_form", drift_form, DRIFT_FORMS)
    rng = _stream(rng)
    s, sigma, mu = float(positive("spot", spot)), float(vol), float(drift)
    if sigma < 0:
        raise ValueError("vol must not be negative")
    t = np.asarray(times, dtype=float).ravel()
    if t.size == 0 or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be increasing years, from 0 on")
    paths = _count("n_paths", n_paths)
    if not isinstance(antithetic, (bool, np.bool_)):
        raise ValueError("antithetic must be True or False")
    if antithetic:
        if paths % 2:
            raise ValueError("n_paths must be even for antithetic pairs")
        half = rng.standard_normal((paths // 2, t.size))
        z = np.vstack([half, -half])
    else:
        z = rng.standard_normal((paths, t.size))
    nu = mu - 0.5 * sigma * sigma if drift_form == "arithmetic" else mu
    step = np.diff(np.r_[0.0, t])
    return s * np.exp(np.cumsum(nu * step + sigma * np.sqrt(step) * z, axis=1))


@_guard(vol=('vols',), time=('times',), price=('spots',))
def correlated_gbm_paths(spots, drifts, vols, correlation, times, n_paths, rng, drift_form, antithetic):
    """Correlated GBM paths of several assets, normals correlated by Cholesky, the drift's form named.

    For asset i, S_i(t_j) = S_i(t_{j-1}) exp(nu_i dt_j + vol_i sqrt(dt_j) W_i), W = L Z with L the lower Cholesky
    factor of the correlation matrix (L L' = C) and Z independent standard normals, so corr(W_i, W_k) = C_ik; exact at
    every time, any spacing. drift_form as qf.gbm_paths: 'arithmetic' reads drifts as mu_i (nu_i = mu_i - vol_i^2/2,
    risk-neutral mu_i = r - q_i), 'log' as nu_i. A basket, spread or rainbow payoff is a function of the last column.

    conventions:
    - spots, drifts, vols: one value per asset; correlation: the assets' symmetric positive-definite matrix
    - times: increasing years from today (only a first 0 gives the spots' column)
    - drift_form: 'arithmetic' or 'log'; antithetic: True mirrors the first half's normals in the second (n_paths even)
    - rng: a Generator or int seed (default_rng), or a RandomState or None (the legacy stream np.random.seed sets);
      one draw of normals (n_paths, or n_paths / 2, by len(times) by assets), each row's assets then correlated by L
    returns: an n_paths x len(times) x assets array, [p, j, i] asset i's price at times[j] on path p
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 2.3.3 and 3.2.3; Hull, "Correlated
      stochastic processes"
    aliases: correlated GBM, correlated paths, correlated Brownian motions, Cholesky simulation, two-asset simulation, basket simulation
    example:
        qf.correlated_gbm_paths(spots=[100.0, 50.0], drifts=[0.05, 0.05], vols=[0.2, 0.3],
                                correlation=[[1.0, 0.6], [0.6, 1.0]], times=[0.5, 1.0], n_paths=1000, rng=7,
                                drift_form="arithmetic", antithetic=False)
    """
    choice("drift_form", drift_form, DRIFT_FORMS)
    rng = _stream(rng)
    s = positive("spots", spots).ravel()
    mu, sigma = np.asarray(drifts, dtype=float).ravel(), np.asarray(vols, dtype=float).ravel()
    c = np.asarray(correlation, dtype=float)
    if s.size < 2 or mu.shape != s.shape or sigma.shape != s.shape or c.shape != (s.size, s.size):
        raise ValueError("spots, drifts and vols need one value per asset (two or more) and correlation their matrix")
    if np.any(sigma < 0) or not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0) or np.any(np.abs(c) > 1):
        raise ValueError("vols must not be negative; correlation must be symmetric with a unit diagonal")
    try:
        lower = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as error:
        raise ValueError("correlation is not positive definite (no Cholesky factor)") from error
    t = _times(times)
    z = _normals(rng, _count("n_paths", n_paths), (t.size, s.size), antithetic) @ lower.T
    nu = mu - 0.5 * sigma * sigma if drift_form == "arithmetic" else mu
    step = np.diff(np.r_[0.0, t])[None, :, None]
    return s * np.exp(np.cumsum(nu * step + sigma * np.sqrt(step) * z, axis=1))


@_guard(rate=('rate', 'dividend_yield',), time=('times',), price=('spot',))
def local_vol_paths(spot, local_vol, times, substeps, rate, dividend_yield, n_paths, rng, scheme, antithetic):
    """Local-vol Monte Carlo paths: log-Euler or Euler steps under a surface vol(t, S), the drift r - q.

    dS = (r - q) S dt + vol(t, S) S dW, each interval to the next date split into `substeps` equal steps h and the
    vol read at the step's start (t_k, S_k): 'log_euler' ln S += (r - q - vol^2/2) h + vol sqrt(h) Z (positive
    paths), 'euler' S *= 1 + (r - q) h + vol sqrt(h) Z. A constant vol under 'log_euler' is qf.gbm_paths exactly (one
    substep, the same draws); a vol of time alone prices Europeans as qf.bs_price at the root mean square vol. The
    surface is the caller's, e.g. Dupire's from qf.dupire_local_vol on a grid, interpolated and clipped to it. Barrier
    and vanilla payoffs on the paths: qf.barrier_mc_price.

    conventions:
    - local_vol: a callable vol(t, S) of a time (years) and an array of spots, annual vols out (a grid's
      interpolation and extrapolation are the caller's)
    - times: increasing years from today (only a first 0 gives the spot's column); the monitoring or payment dates
    - substeps: steps per interval between dates, an int of at least 1
    - rate r, dividend_yield q: continuously compounded and annual
    - scheme: 'log_euler' or 'euler'; antithetic: True mirrors the first half's normals in the second (n_paths even)
    - rng: a Generator or int seed (default_rng), or a RandomState or None (the legacy stream np.random.seed sets);
      one draw of normals (n_paths, or n_paths / 2, by len(times) x substeps)
    returns: an n_paths x len(times) array, column j the price at times[j]
    source: Dupire (1994), "Pricing with a smile", Risk 7(1); Glasserman, Monte Carlo Methods in Financial Engineering
      (2004), 6.1 (Euler schemes); Gatheral, The Volatility Surface (2006), ch. 1
    aliases: local vol Monte Carlo, local vol simulation, local vol paths, Dupire Monte Carlo, local-vol MC
    example:
        qf.local_vol_paths(spot=100.0, local_vol=lambda t, s: 0.2 + 0.1 * (s / 100.0 - 1.0) ** 2, times=[0.5, 1.0],
                           substeps=50, rate=0.03, dividend_yield=0.0, n_paths=2000, rng=11,
                           scheme="log_euler", antithetic=True)
    """
    choice("scheme", scheme, LOCAL_VOL_SCHEMES)
    rng = _stream(rng)
    if not callable(local_vol):
        raise ValueError("local_vol must be a callable vol(t, S) of a time and an array of spots")
    s0, t = float(positive("spot", spot)), _times(times)
    m, paths = _count("substeps", substeps), _count("n_paths", n_paths)
    carry = float(rate) - float(dividend_yield)
    z = _normals(rng, paths, (t.size * m,), antithetic)
    out_paths = np.empty((paths, t.size))
    s, now, k = np.full(paths, s0), 0.0, 0
    for j, end in enumerate(t):
        h = (end - now) / m
        for _ in range(m):
            if h > 0:
                vol = np.broadcast_to(np.asarray(local_vol(now, s), dtype=float), s.shape)
                if np.any(~(vol >= 0)):
                    raise ValueError(f"local_vol gave a negative or NaN vol at t = {now}")
                if scheme == "log_euler":
                    s = s * np.exp((carry - 0.5 * vol * vol) * h + vol * np.sqrt(h) * z[:, k])
                else:
                    s = s * (1.0 + carry * h + vol * np.sqrt(h) * z[:, k])
                now += h
            k += 1
        now = end
        out_paths[:, j] = s
    return out_paths


def ou_paths(x0, kappa, theta, sigma, dt, n_steps, n_paths, rng, scheme, jump_rate, jump_mean, jump_std, jump_count):
    """Monte Carlo paths of an OU process, exact or Euler, with optional compound-Poisson jumps.

    dX = kappa (theta - X) dt + sigma dW + J dN, J ~ N(jump_mean, jump_std^2), N Poisson of rate lambda. 'exact':
    X_{k+1} = theta + (X_k - theta) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) Z, each jump
    decayed by e^{-kappa (dt - s)} from its time s, uniform in the step; 'euler': X_{k+1} = X_k + kappa (theta -
    X_k) dt + sigma sqrt(dt) Z + the step's jumps undecayed. jump_count 'poisson' draws Poisson(lambda dt) jumps a
    step, 'bernoulli' at most one, with probability lambda dt. 'exact' with 'poisson' is the exact law at every step
    (its moments are qf.ou_jump_moments); jump_rate 0 is the pure OU. A geometric mean-reverting model's price is S
    = e^X.

    conventions:
    - dt: the step, in years for annual parameters; n_steps steps, so n_steps + 1 columns from x0
    - scheme: 'euler' where the task says Euler or Euler-Maruyama, or writes the step X + kappa (theta - X) dt + sigma
      sqrt(dt) Z; 'exact' where it says exact simulation, the exact discretisation or the AR(1) transition
    - jump_count: 'poisson' or 'bernoulli' (lambda dt <= 1)
    - jump_rate, jump_mean, jump_std: lambda per year, the sizes' mean and standard deviation in X's units
    - rng: a Generator or int seed (default_rng), or a RandomState or None (the legacy stream np.random.seed sets);
      the normals (n_paths, n_steps) first, then the jumps' counts, sizes, times
    returns: an n_paths x (n_steps + 1) array, column k the value at time k dt, column 0 x0 (the
      paths after the first step: [:, 1:])
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 3.3; Cont and Tankov (2004), 6.3
    aliases: Ornstein-Uhlenbeck simulation, OU simulation, OU paths, mean-reverting jump diffusion simulation
    example:
        qf.ou_paths(x0=0.5, kappa=2.0, theta=0.2, sigma=0.3, dt=1 / 252, n_steps=252, n_paths=1000,
                    rng=1, scheme="euler", jump_rate=1.5, jump_mean=-0.05,
                    jump_std=0.1, jump_count="poisson")  # result: {'shape': (1000, 253)}
    """
    choice("scheme", scheme, OU_SCHEMES)
    choice("jump_count", jump_count, JUMP_COUNTS)
    rng = _stream(rng)
    kappa, step = float(positive("kappa", kappa)), float(positive("dt", dt))
    steps, paths = _count("n_steps", n_steps), _count("n_paths", n_paths)
    theta, sigma, lam = float(theta), float(sigma), float(jump_rate)
    mu_j, s_j = float(jump_mean), float(jump_std)
    if sigma < 0 or lam < 0 or s_j < 0:
        raise ValueError("sigma, jump_rate and jump_std must not be negative")
    if jump_count == "bernoulli" and lam * step > 1:
        raise ValueError("jump_rate * dt must not exceed 1, a step's jump probability ('bernoulli')")
    z = rng.standard_normal((paths, steps))
    if scheme == "exact":
        b = np.exp(-kappa * step)
        a, scale = theta * (1.0 - b), sigma * np.sqrt(-np.expm1(-2.0 * kappa * step) / (2.0 * kappa))
    else:
        b, a, scale = 1.0 - kappa * step, kappa * theta * step, sigma * np.sqrt(step)
    noise = scale * z
    if lam > 0:
        counts = (rng.poisson(lam * step, (paths, steps)) if jump_count == "poisson"
                  else (rng.random((paths, steps)) < lam * step).astype(np.int64))
        total = int(counts.sum())
        sizes = rng.normal(mu_j, s_j, total)
        if scheme == "exact":
            sizes = sizes * np.exp(-kappa * step * rng.random(total))     # (dt - s)/dt is uniform as s is
        cells = np.repeat(np.arange(paths * steps), counts.ravel())
        noise = noise + np.bincount(cells, weights=sizes, minlength=paths * steps).reshape(paths, steps)
    x = np.empty((paths, steps + 1))
    x[:, 0] = float(x0)
    for k in range(steps):
        x[:, k + 1] = a + b * x[:, k] + noise[:, k]
    return x


@_guard(time=('times',))
def cir_paths(x0, kappa, theta, sigma, times, n_paths, rng, scheme):
    """CIR (Cox-Ingersoll-Ross) paths: the exact noncentral transition, or truncated Euler.

    dX = kappa (theta - X) dt + sigma sqrt(X) dW, X never negative: a short rate, a default intensity or a variance.
    'exact' draws X(t + h) from its law given X(t), at any spacing: X(t + h) = c Y with Y noncentral chi-square of d =
    4 kappa theta / sigma^2 degrees of freedom and noncentrality X(t) e^{-kappa h} / c, c = sigma^2 (1 - e^{-kappa h})
    / (4 kappa). 'full_truncation' steps Euler on a shadow value x, using x+ = max(x, 0) in the drift and the
    diffusion, x + kappa (theta - x+) h + sigma sqrt(x+ h) Z, and returns x+ (Lord, Koekkoek and van Dijk): biased,
    the bias falling with the step. E[X(t + h) | X(t)] = theta + (X(t) - theta) e^{-kappa h}. The Feller condition
    2 kappa theta >= sigma^2 keeps X away from zero. A bond off the paths: mean of exp(-integral of X) by qf.mc_price.

    conventions:
    - x0: today's value; kappa: the speed of mean reversion; theta: the long-run level; sigma: the volatility of the
      square root (sigma sqrt(X) the diffusion); all annual
    - times: increasing years from today (only a first 0 gives x0's column); 'full_truncation' takes one Euler step
      from each time to the next, so pass a fine grid
    - scheme: 'exact' (the noncentral chi-square transition) or 'full_truncation' (Euler with max(x, 0))
    - rng: a Generator or int seed (default_rng), or a RandomState or None (the legacy stream np.random.seed sets);
      'exact' one noncentral chi-square draw a time for every path, in order; 'full_truncation' one draw of normals
      (n_paths, len(times))
    returns: a dict: paths (n_paths x len(times), column j the value at times[j]); feller (True where 2 kappa theta
      >= sigma^2)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 3.4; Cox, Ingersoll and Ross (1985),
      Econometrica 53(2); Lord, Koekkoek and van Dijk (2010), Quantitative Finance 10(2)
    aliases: CIR paths, Cox-Ingersoll-Ross paths, noncentral chi-square transition, [Feller condition]
    example:
        qf.cir_paths(x0=0.03, kappa=0.5, theta=0.04, sigma=0.1, times=[0.25, 0.5, 0.75, 1.0],
                     n_paths=1000, rng=1,
                     scheme="exact")
        # result: {'paths.shape': (1000, 4), 'feller': True}
    """
    choice("scheme", scheme, CIR_SCHEMES)
    rng = _stream(rng)
    k, level, s = float(positive("kappa", kappa)), float(positive("theta", theta)), float(positive("sigma", sigma))
    x = float(x0)
    if not x >= 0:
        raise ValueError("x0 must not be negative")
    t, paths = _times(times), _count("n_paths", n_paths)
    steps = np.diff(np.r_[0.0, t])
    found, now = np.empty((paths, t.size)), np.full(paths, x)
    if scheme == "exact":
        dof = 4.0 * k * level / (s * s)
        for j, h in enumerate(steps):
            if h > 0:
                decay = np.exp(-k * h)
                c = s * s * -np.expm1(-k * h) / (4.0 * k)
                now = c * rng.noncentral_chisquare(dof, now * decay / c)
            found[:, j] = now
    else:
        z = rng.standard_normal((paths, t.size))
        for j, h in enumerate(steps):
            if h > 0:
                plus = np.maximum(now, 0.0)
                now = now + k * (level - plus) * h + s * np.sqrt(plus * h) * z[:, j]
            found[:, j] = np.maximum(now, 0.0)
    return Result({"paths": found, "feller": bool(2.0 * k * level >= s * s)})


@_guard(rate=('rate', 'dividend_yield',), time=('times',), price=('spot',))
def heston_paths(spot, v0, kappa, theta, sigma, rho, rate, dividend_yield, times, substeps, n_paths, rng, scheme,
                 psi_c):
    """Heston paths by Andersen's quadratic-exponential (QE) scheme or truncated Euler.

    dS = (r - q) S dt + sqrt(v) S dW1, dv = kappa (theta - v) dt + sigma sqrt(v) dW2, corr(dW1, dW2) = rho; each interval
    between times cut into `substeps` steps h. 'full_truncation' (Lord, Koekkoek and van Dijk): v+ = max(v, 0) in the
    drift and diffusion, ln S += (r - q - v+/2) h + sqrt(v+ h) Z1, v += kappa (theta - v+) h + sigma sqrt(v+ h) (rho Z1 +
    sqrt(1 - rho^2) Z2), v+ reported. 'qe' (Andersen 2008, quadratic exponential): v(t + h) matched to its exact law's
    mean m and variance s^2, a (b + Z)^2 where psi = s^2 / m^2 <= psi_c, else zero or an exponential; ln S += (r - q) h
    + K0 + K1 v(t) + K2 v(t + h) + sqrt(K3 v(t) + K4 v(t + h)) Z, central weights 1/2, no martingale correction. Both
    keep the variance non-negative. A European on the last column prices toward qf.charfn_price with qf.heston_charfn.

    conventions:
    - spot; v0: today's variance (a vol of 20% is 0.04); kappa, theta: the variance's speed and long-run level; sigma:
      the vol of variance; rho: the correlation of the price's and the variance's shocks; all annual
    - rate r, dividend_yield q: continuously compounded and annual, the price's drift r - q
    - times: increasing years from today (only a first 0 gives today's column); substeps: steps per interval, an int
    - scheme: 'full_truncation' (Euler with max(v, 0)) or 'qe' (Andersen's quadratic exponential)
    - psi_c: under 'qe' the branches' switch, in [1, 2] (Andersen's 1.5); None under 'full_truncation'
    - rng: a Generator or int seed (default_rng), or a RandomState or None (the legacy stream np.random.seed sets);
      'full_truncation' normals (n_paths, steps, 2); 'qe' uniforms (n_paths, steps) for the variance, then normals
      (n_paths, steps) for the price
    returns: a dict: spot (n_paths x len(times), column j the price at times[j]); variance (the same shape)
    source: Andersen (2008), Journal of Computational Finance 11(3); Lord, Koekkoek and van Dijk (2010)
    aliases: Heston paths, Heston simulation, QE scheme, Andersen QE
    example:
        qf.heston_paths(spot=100.0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7, rate=0.03, dividend_yield=0.0,
                        times=[0.5, 1.0], substeps=50, n_paths=2000, rng=3,
                        scheme="qe", psi_c=1.5)
        # result: {'spot.shape': (2000, 2), 'variance.shape': (2000, 2)}
    """
    choice("scheme", scheme, HESTON_SCHEMES)
    rng = _stream(rng)
    s0, var0 = float(positive("spot", spot)), float(v0)
    k, level, s = float(positive("kappa", kappa)), float(positive("theta", theta)), float(positive("sigma", sigma))
    corr = float(rho)
    if not (var0 >= 0 and -1.0 <= corr <= 1.0):
        raise ValueError("v0 must not be negative and rho must lie in [-1, 1]")
    if scheme == "qe":
        if psi_c is None or not 1.0 <= float(psi_c) <= 2.0:
            raise ValueError("psi_c must be a number in [1, 2] under 'qe' (Andersen's 1.5)")
    elif psi_c is not None:
        raise ValueError("psi_c is None under 'full_truncation'")
    t, m, paths = _times(times), _count("substeps", substeps), _count("n_paths", n_paths)
    carry = float(rate) - float(dividend_yield)
    total = t.size * m
    if scheme == "qe":
        uniform, z = rng.random((paths, total)), rng.standard_normal((paths, total))
    else:
        draws = rng.standard_normal((paths, total, 2))
    prices, variances = np.empty((paths, t.size)), np.empty((paths, t.size))
    log_s, v, now, step = np.full(paths, np.log(s0)), np.full(paths, var0), 0.0, 0
    for j, end in enumerate(t):
        h = (end - now) / m
        for _ in range(m):
            if h > 0:
                if scheme == "qe":
                    log_s, v = _qe_step(log_s, v, h, k, level, s, corr, carry, float(psi_c), uniform[:, step],
                                        z[:, step])
                else:
                    plus = np.maximum(v, 0.0)
                    root = np.sqrt(plus * h)
                    z1, z2 = draws[:, step, 0], draws[:, step, 1]
                    log_s = log_s + (carry - 0.5 * plus) * h + root * z1
                    v = v + k * (level - plus) * h + s * root * (corr * z1 + np.sqrt(1.0 - corr * corr) * z2)
                now += h
            step += 1
        now = end
        prices[:, j], variances[:, j] = np.exp(log_s), np.maximum(v, 0.0)
    return Result({"spot": prices, "variance": variances})


def _qe_step(log_s, v, h, kappa, theta, sigma, rho, carry, psi_c, uniform, z):
    """One step of Andersen's QE scheme: the variance by its quadratic or exponential branch, then ln S."""
    decay = np.exp(-kappa * h)
    mean = theta + (v - theta) * decay
    var = v * sigma * sigma * decay * -np.expm1(-kappa * h) / kappa + theta * sigma * sigma * np.expm1(-kappa * h) ** 2 \
        / (2.0 * kappa)
    psi = var / (mean * mean)
    quadratic = psi <= psi_c
    nxt = np.empty_like(v)
    q = psi[quadratic]
    b2 = 2.0 / q - 1.0 + np.sqrt(2.0 / q) * np.sqrt(2.0 / q - 1.0)
    nxt[quadratic] = mean[quadratic] / (1.0 + b2) * (np.sqrt(b2) + ndtri(np.maximum(uniform[quadratic], 1e-300))) ** 2
    e = ~quadratic
    p = (psi[e] - 1.0) / (psi[e] + 1.0)
    beta = (1.0 - p) / mean[e]
    u = uniform[e]
    nxt[e] = np.where(u <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - u, 1e-300)) / beta)
    k0 = -rho * kappa * theta * h / sigma
    k1 = 0.5 * h * (kappa * rho / sigma - 0.5) - rho / sigma
    k2 = 0.5 * h * (kappa * rho / sigma - 0.5) + rho / sigma
    k3 = k4 = 0.5 * h * (1.0 - rho * rho)
    log_s = log_s + carry * h + k0 + k1 * v + k2 * nxt + np.sqrt(k3 * v + k4 * nxt) * z
    return log_s, nxt


@_guard(time=('times',))
def sde_paths(x0, drift, diffusion, diffusion_derivative, times, substeps, n_paths, rng, scheme):
    """SDE paths by the Euler-Maruyama or Milstein scheme, for any drift a(t, X) and diffusion b(t, X).

    Each interval between times split into `substeps` equal steps h: 'euler' X + a h + b sqrt(h) Z (strong order
    1/2, weak order 1); 'milstein' adds (1/2) b b' h (Z^2 - 1), b' the diffusion's derivative in x (strong order 1).
    On GBM (a = mu x, b = vol x, b' = vol) both schemes' means are x0 (1 + mu h)^n after n steps. Where b does not
    depend on x (b' = 0) the two schemes are the same. For GBM, OU or CIR exactly: qf.gbm_paths, qf.ou_paths,
    qf.cir_paths.

    conventions:
    - x0: today's value; drift a(t, x) and diffusion b(t, x): callables of a time in years and an array of values,
      annual, giving an array; diffusion_derivative: under 'milstein' the callable b'(t, x) = db/dx, None under
      'euler'
    - times: increasing years from today (only a first 0 gives x0's column); substeps: steps per interval, an int
    - scheme: 'euler' (Euler-Maruyama) or 'milstein'
    - rng: a Generator or int seed (default_rng), or a RandomState or None (the legacy stream np.random.seed sets);
      one draw of normals (n_paths, len(times) x substeps), the steps in order
    returns: an n_paths x len(times) array, column j the value at times[j]
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 6.1 and 6.2; Kloeden and Platen,
      Numerical Solution of Stochastic Differential Equations (1992), 10.3
    aliases: [Euler-Maruyama], [Euler scheme], Milstein scheme, SDE discretization, SDE simulation
    example:
        qf.sde_paths(x0=100.0, drift=lambda t, x: 0.05 * x, diffusion=lambda t, x: 0.2 * x,
                     diffusion_derivative=lambda t, x: 0.2 + 0.0 * x, times=[0.5, 1.0], substeps=50, n_paths=1000,
                     rng=4, scheme="milstein")
        # result: {'shape': (1000, 2)}
    """
    choice("scheme", scheme, SDE_SCHEMES)
    rng = _stream(rng)
    if not (callable(drift) and callable(diffusion)):
        raise ValueError("drift and diffusion must be callables a(t, x) and b(t, x)")
    if scheme == "milstein" and not callable(diffusion_derivative):
        raise ValueError("diffusion_derivative must be the callable b'(t, x) under 'milstein'")
    if scheme == "euler" and diffusion_derivative is not None:
        raise ValueError("diffusion_derivative is None under 'euler'")
    t, m, paths = _times(times), _count("substeps", substeps), _count("n_paths", n_paths)
    z = rng.standard_normal((paths, t.size * m))
    found = np.empty((paths, t.size))
    x, now, k = np.full(paths, float(x0)), 0.0, 0
    for j, end in enumerate(t):
        h = (end - now) / m
        for _ in range(m):
            if h > 0:
                a = np.broadcast_to(np.asarray(drift(now, x), dtype=float), x.shape)
                b = np.broadcast_to(np.asarray(diffusion(now, x), dtype=float), x.shape)
                shock = z[:, k]
                step = x + a * h + b * np.sqrt(h) * shock
                if scheme == "milstein":
                    slope = np.broadcast_to(np.asarray(diffusion_derivative(now, x), dtype=float), x.shape)
                    step = step + 0.5 * b * slope * h * (shock * shock - 1.0)
                x = step
                now += h
            k += 1
        now = end
        found[:, j] = x
    return found


@_guard(rate=('forwards',), vol=('vols',))
def lmm_paths(forwards, accruals, vols, correlation, substeps, n_paths, rng, measure, predictor_corrector):
    """LIBOR market model (LMM, BGM) paths by log-Euler under a named numeraire.

    Tenor dates T_0 = 0 < T_1 < ... < T_M, accruals d_i = T_{i+1} - T_i; forward L_i covers [T_i, T_{i+1}] and fixes at
    T_i. Each period is cut into `substeps` steps h; each live forward steps ln L_i += (mu_i - vol_i^2/2) h + vol_i
    sqrt(h) W_i, W correlated by the correlation's Cholesky factor. The measure sets the drift: 'spot' (the rolling
    money-market numeraire) mu_i = vol_i sum_{j = n..i} d_j L_j vol_j rho_ij / (1 + d_j L_j), n the next tenor date's
    index; 'terminal' (the T_M bond) mu_i = -vol_i sum_{j = i+1..M-1} of the same terms. predictor_corrector averages the
    drifts at the step's start and at its predicted end. A payment X at T_k is worth mean(X deflator[:, k]) today;
    mean(deflator[:, k]) reprices prod_{j < k} 1 / (1 + d_j L_j(0)) within the scheme's error.

    conventions:
    - forwards: today's forward rates L_0 .. L_{M-1}, simply compounded over their accruals (L_0 the period starting
      today); accruals: the M periods' year fractions
    - vols: each forward's lognormal vol, M values, or an M x M array whose row k holds the vols over period k;
      correlation: the forwards' M x M correlation matrix
    - substeps: steps per period, an int; measure: 'spot' or 'terminal'; predictor_corrector: True or False
    - rng: a Generator or int seed (default_rng), or a RandomState or None (the legacy stream np.random.seed sets);
      one draw of normals (n_paths, (M - 1) x substeps, M), the steps in order
    returns: a dict: forwards (n_paths x (M + 1) x M, [p, k, i] forward i at tenor date k on path p, frozen after its
      fixing); deflator (n_paths x (M + 1), numeraire(0) / numeraire(T_k) on each path); dates (T_0 .. T_M)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 3.7; Brigo and Mercurio, Interest Rate
      Models (2006), ch. 6
    aliases: LIBOR market model, LMM, BGM model, Brace-Gatarek-Musiela, predictor-corrector
    example:
        qf.lmm_paths(forwards=[0.03, 0.032, 0.034, 0.035], accruals=[0.5, 0.5, 0.5, 0.5], vols=[0.2, 0.2, 0.2, 0.2],
                     correlation=[[1.0, 0.9, 0.8, 0.7], [0.9, 1.0, 0.9, 0.8], [0.8, 0.9, 1.0, 0.9], [0.7, 0.8, 0.9, 1.0]],
                     substeps=4, n_paths=1000, rng=6, measure="spot", predictor_corrector=True)
        # result: {'forwards.shape': (1000, 5, 4), 'deflator.shape': (1000, 5)}
    """
    choice("measure", measure, MEASURES)
    rng = _stream(rng)
    start = positive("forwards", forwards).ravel()
    delta = positive("accruals", accruals).ravel()
    size = start.size
    if delta.shape != start.shape:
        raise ValueError("forwards and accruals need one value per period")
    sigma = np.asarray(vols, dtype=float)
    sigma = np.broadcast_to(sigma, (size, size)) if sigma.ndim == 1 else sigma
    c = np.asarray(correlation, dtype=float)
    if sigma.shape != (size, size) or np.any(sigma < 0) or c.shape != (size, size):
        raise ValueError("vols must be M values or an M x M array, not negative, and correlation M x M")
    if not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
        raise ValueError("correlation must be symmetric with a unit diagonal")
    try:
        lower = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as error:
        raise ValueError("correlation is not positive definite (no Cholesky factor)") from error
    if not isinstance(predictor_corrector, (bool, np.bool_)):
        raise ValueError("predictor_corrector must be True or False")
    m, paths = _count("substeps", substeps), _count("n_paths", n_paths)
    z = rng.standard_normal((paths, max(size - 1, 0) * m, size)) @ lower.T
    dates = np.r_[0.0, np.cumsum(delta)]
    found = np.empty((paths, size + 1, size))
    now = np.tile(start, (paths, 1))
    found[:, 0] = now
    step = 0
    for k in range(size):
        if k < size - 1:
            h = delta[k] / m
            vol = sigma[k]
            live = np.arange(size) > k
            for _ in range(m):
                shock = vol * np.sqrt(h) * z[:, step]
                first = _lmm_drift(now, delta, vol, c, k, measure)
                guess = now * np.exp((first - 0.5 * vol * vol) * h + shock)
                drift = 0.5 * (first + _lmm_drift(guess, delta, vol, c, k, measure)) if predictor_corrector else first
                now = np.where(live, now * np.exp((drift - 0.5 * vol * vol) * h + shock), now)
                step += 1
        found[:, k + 1] = now
    growth = 1.0 + delta * np.diagonal(found[:, :size], axis1=1, axis2=2)      # 1 + d_j L_j(T_j) on each path
    if measure == "spot":
        deflator = 1.0 / np.cumprod(np.c_[np.ones(paths), growth], axis=1)
    else:
        bond_end = np.prod(1.0 / (1.0 + delta * start))
        deflator = np.empty((paths, size + 1))
        for k in range(size + 1):
            deflator[:, k] = bond_end * np.prod(1.0 + delta[k:] * found[:, k, k:], axis=1)
    return Result({"forwards": found, "deflator": deflator, "dates": dates})


def _lmm_drift(forwards, delta, vol, correlation, period, measure):
    """Each forward's drift in period `period` (from T_period to the next date) under the spot or terminal measure;
    a forward fixed by then gets zero."""
    size = delta.size
    weight = delta * forwards * vol / (1.0 + delta * forwards)          # d_j L_j vol_j / (1 + d_j L_j)
    index = np.arange(size)
    if measure == "spot":
        mask = (index[None, :] > period) & (index[None, :] <= index[:, None])      # j from the next date to i
        sign = 1.0
    else:
        mask = index[None, :] > index[:, None]                                      # j from i + 1 to M - 1
        sign = -1.0
    terms = weight @ (correlation * mask).T                                          # [p, i] sum_j rho_ij w_pj
    return np.where(index > period, sign * vol * terms, 0.0)
