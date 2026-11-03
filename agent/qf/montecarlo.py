"""Monte Carlo estimates: control variates, importance and stratified sampling, Sobol, barriers, Longstaff-Schwartz.

`import qf`. A path array is n_paths x n_times, a row a path (qf.gbm_paths and the other generators); times in years;
`rng` a numpy.random.Generator or an int seed (numpy.random.default_rng(seed)), or numpy's legacy stream, a
numpy.random.RandomState or None (the global one numpy.random.seed(seed) sets; from one seed the two streams draw
different samples), whose draws each function names: a seed reproduces a run. An estimate comes with its standard
error and interval; several results come back as a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import gaussian_kde, qmc

from ._common import Result, choice, out, positive
from .greeks import bump_greeks, likelihood_ratio_greeks, pathwise_greeks  # noqa: F401 - moved, qf.montecarlo.<name> kept
from .paths import _count, _times
from .primitives import _guard, _stream
from .paths import correlated_gbm_paths, gbm_paths, local_vol_paths, ou_paths  # noqa: F401 - moved, as above

__all__ = ["american_bounds", "barrier_mc_price", "control_variate", "estimate_error", "first_hit_times",
           "importance_shift", "longstaff_schwartz", "mc_payoff_price", "mc_price", "sobol_normals",
           "stratified_normals"]

DIRECTIONS = ("up", "down")
BASES = ("polynomial", "laguerre", "weighted_laguerre")
OPTIONS = ("call", "put")
MONITORS = ("discrete", "bridge")
BARRIER_TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")
BETAS = ("estimated",)
STRATA_KINDS = ("stratified", "latin_hypercube")
ALLOCATIONS = ("proportional", "optimal")
ORDERS = ("sequential", "brownian_bridge")
ESTIMATORS = ("quantile", "ratio", "batch_means")
DENSITIES = ("kernel",)
BOUND_METHODS = ("random_tree", "dual")


def _level(confidence):
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1), e.g. 0.95")
    return ndtri(0.5 + level / 2.0)


def _ddof(ddof):
    if isinstance(ddof, (bool, np.bool_)) or ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, not {ddof!r}")
    return int(ddof)


@_guard(level=('confidence',))
def mc_price(payoffs, discount_factor, ddof, confidence):
    """Monte Carlo estimate: the discounted mean payoff, its standard error (ddof named) and interval.

    price = D mean(X); std_error = D s / sqrt(n), s^2 = sum (X - mean)^2 / (n - ddof); interval price +- z std_error,
    z = Phi^-1((1 + confidence) / 2). Average antithetic pairs first.

    Price half-width h needs n = (z D s / h)^2. Efficiency: variance times time. Inverse transform: X = F^-1(U),
    U uniform. Conditional Monte Carlo uses E[X | Y] to reduce variance. Euler: weak order (means) 1, strong
    (paths) 1/2; Milstein: strong 1.

    conventions:
    - payoffs: undiscounted draws at one date (n values, or n x k estimates)
    - discount_factor: D; 1.0 for discounted payoffs
    - ddof: 1 (the unbiased sample variance) or 0
    - confidence: the interval's level, inside (0, 1)
    returns: a dict: price, std_error, low and high (interval), floats or one per column; n (paths)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 1.1 (the estimator and its error)
    aliases: Monte Carlo price, Monte Carlo estimate, Monte Carlo standard error, Monte Carlo confidence interval
    example:
        qf.mc_price(payoffs=[0.0, 3.2, 0.0, 7.5, 1.1, 0.0, 4.4, 2.0],
                    discount_factor=0.97, ddof=1,
                    confidence=0.95)
    """
    x = np.asarray(payoffs, dtype=float)
    if x.ndim not in (1, 2) or x.shape[0] < 2 or not np.all(np.isfinite(x)):
        raise ValueError("payoffs must be n finite values (or n x k), n at least 2")
    if ddof not in (0, 1) or isinstance(ddof, bool):
        raise ValueError(f"ddof must be 0 or 1, not {ddof!r}")
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1), e.g. 0.95")
    d = float(positive("discount_factor", discount_factor))
    n = x.shape[0]
    price = d * x.mean(axis=0)
    error = d * x.std(axis=0, ddof=int(ddof)) / np.sqrt(n)
    z = ndtri(0.5 + level / 2.0)
    return Result({"price": out(price), "std_error": out(error), "low": out(price - z * error),
                   "high": out(price + z * error), "n": n})


@_guard(level=('confidence',))
def mc_payoff_price(paths, payoff, n_paths, rng, antithetic, control, control_mean, discount_factor, ddof, confidence):
    """Monte Carlo price of any payoff on simulated paths, antithetic or with a control variate.

    The paths are an array (from qf.gbm_paths, qf.correlated_gbm_paths, qf.local_vol_paths, qf.heston_paths) or drawn
    by a callable paths(n_paths, rng); payoff(paths) gives each path's payoff X, paid at one date: price = D mean(X),
    the error and interval as qf.mc_price. Antithetic paths (the second half mirroring the first) are averaged in pairs
    first. With a control, a callable C(paths) of known mean E[C]: D mean(X - beta (C - E[C])), beta = Cov(X, C) /
    Var(C) on the same paths (qf.control_variate).

    What it is used for: any payoff no function prices, in a few lines of numpy on the paths: a path-dependent
    payoff (an average, a maximum), a basket on correlated paths, local-vol or Heston paths.

    conventions:
    - paths: an array of paths (n_paths x n_times, or x assets), or a callable paths(n_paths, rng) drawing them
    - payoff: a callable of the path array, one undiscounted payoff a path (lambda s: (s[:, -1] - 100.0).clip(0.0))
    - n_paths, rng: the paths, and the rng the callable gets as given (any a qf sampler takes); both None with an
      array
    - antithetic: True where the second half of the paths mirrors the first (pairs averaged), else False
    - control: None, or a callable of the path array, the control's value a path; control_mean: its known mean E[C],
      undiscounted, exact for the simulated dates (None without a control)
    - discount_factor: D, e.g. exp(-r T); 1.0 for payoffs already discounted
    - ddof: 1 (the unbiased sample variance) or 0; confidence: the interval's level, e.g. 0.95
    returns: a dict: price, std_error, low and high (the interval), n (the independent draws: pairs where antithetic),
      beta (the control's coefficient, nan without one)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 1.1 and 4.1
    aliases: path-dependent payoff, exotic payoff Monte Carlo
    example:
        qf.mc_payoff_price(paths=qf.gbm_paths(100.0, 0.03, 0.2, [0.25, 0.5, 0.75, 1.0], 2000, 1, "arithmetic", True),
                           payoff=lambda s: (s.mean(axis=1) - 100.0).clip(0.0), n_paths=None, rng=None, antithetic=True,
                           control=None, control_mean=None, discount_factor=0.97, ddof=1, confidence=0.95)
        # result: {'price': 6.39627, 'std_error': 0.15844}
    """
    if callable(paths):
        drawn = np.asarray(paths(_count("n_paths", n_paths), rng), dtype=float)
    elif n_paths is not None or rng is not None:
        raise ValueError("n_paths and rng are None where paths is an array (they feed a callable paths(n_paths, rng))")
    else:
        drawn = np.asarray(paths, dtype=float)
    if drawn.ndim < 2 or drawn.shape[0] < 2:
        raise ValueError("paths must be at least two paths, a row each")
    if not callable(payoff):
        raise ValueError("payoff must be a callable of the path array")
    values = np.asarray(payoff(drawn), dtype=float).reshape(-1)
    if values.size != drawn.shape[0]:
        raise ValueError("payoff must give one value a path")
    controls = None
    if control is None:
        if control_mean is not None:
            raise ValueError("control_mean is None without a control")
    else:
        if not callable(control) or control_mean is None:
            raise ValueError("control must be a callable of the path array, with its known mean control_mean")
        controls = np.asarray(control(drawn), dtype=float).reshape(values.size, -1)
    if not isinstance(antithetic, (bool, np.bool_)):
        raise ValueError("antithetic must be True or False")
    if antithetic:
        if values.size % 2:
            raise ValueError("antithetic pairs need an even number of paths")
        half = values.size // 2
        values = 0.5 * (values[:half] + values[half:])
        controls = None if controls is None else 0.5 * (controls[:half] + controls[half:])
    d = float(positive("discount_factor", discount_factor))
    if controls is None:
        found = mc_price(payoffs=values, discount_factor=d, ddof=ddof, confidence=confidence)
        return Result({**found, "beta": float("nan")})
    fit = control_variate(samples=values, controls=controls, control_means=control_mean, beta="estimated", ddof=ddof,
                          confidence=confidence)
    return Result({"price": d * fit["estimate"], "std_error": d * fit["std_error"], "low": d * fit["low"],
                   "high": d * fit["high"], "n": fit["n"], "beta": fit["beta"]})


@_guard(level=('confidence',))
def control_variate(samples, controls, control_means, beta, ddof, confidence):
    """Control variate estimate, Y - b (C - E[C]), b fitted or given; Kemna-Vorst's geometric Asian control.

    For draws Y_i and controls C_i of known mean E[C], estimate = mean(Y_i - beta (C_i - E[C])). beta 'estimated' is
    the least-squares slope Cov(Y, C) / Var(C) on the same draws (several controls: the regression of Y on them), the
    choice that minimises the variance; a number fixes it beforehand. The variance falls by the factor 1 - rho^2
    (1 - R^2 for several controls); variance_ratio is that factor on the sample, the adjusted draws' variance over
    Y's. std_error is s / sqrt(n), s^2 the adjusted draws' squared deviations over n - ddof (beta's own estimation
    left out, small for large n). Kemna and Vorst's control for an arithmetic-average Asian option is the geometric
    average's payoff on the same paths, its mean the exact qf.asian_price 'geometric' at the same dates.

    conventions:
    - samples: the n draws Y (payoffs, discounted or not)
    - controls: the n control values C on the same draws, or an n x k array of k controls
    - control_means: E[C], a number or k numbers, in the samples' units and exact for the simulated dates (a monthly
      geometric average's mean, not the continuous average's)
    - beta: 'estimated' (the least-squares slope on these draws) or a number (k numbers) fixed beforehand
    - ddof: 1 (the unbiased sample variance) or 0; confidence: the interval's level, e.g. 0.95
    returns: a dict: estimate, std_error, low and high (the interval), beta (a float, or k), variance_ratio (the
      adjusted variance over the plain one), n
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 4.1; Kemna and Vorst (1990), Journal of
      Banking and Finance 14(1); Boyle, Broadie and Glasserman (1997), Journal of Economic Dynamics and Control 21
    aliases: control variate, control variates, Kemna-Vorst control variate, geometric control variate
    example:
        qf.control_variate(samples=[4.1, 0.0, 7.3, 2.2, 0.0, 5.6, 1.4, 3.0], controls=[3.9, 0.0, 6.8, 2.0, 0.0, 5.1, 1.2, 2.7],
                           control_means=2.6, beta="estimated",
                           ddof=1, confidence=0.95)
        # result: {'estimate': 2.82926, 'beta': 1.07322}
    """
    y = np.asarray(samples, dtype=float).ravel()
    c = np.asarray(controls, dtype=float)
    n = y.size
    c = c[:, None] if c.ndim == 1 else c
    if n < 3 or c.ndim != 2 or c.shape[0] != n or not (np.all(np.isfinite(y)) and np.all(np.isfinite(c))):
        raise ValueError("samples must be n finite draws (three or more) and controls n values (or n x k)")
    mean = np.broadcast_to(np.asarray(control_means, dtype=float).ravel(), (c.shape[1],))
    ddof = _ddof(ddof)
    z = _level(confidence)
    centred = c - c.mean(axis=0)
    if isinstance(beta, str):
        choice("beta", beta, BETAS)
        slope = np.linalg.lstsq(centred, y - y.mean(), rcond=None)[0]
    else:
        slope = np.broadcast_to(np.asarray(beta, dtype=float).ravel(), (c.shape[1],))
    adjusted = y - (c - mean) @ slope
    estimate, error = adjusted.mean(), adjusted.std(ddof=ddof) / np.sqrt(n)
    plain = y.var(ddof=ddof)
    return Result({"estimate": float(estimate), "std_error": float(error), "low": float(estimate - z * error),
                   "high": float(estimate + z * error), "beta": out(slope if slope.size > 1 else slope[0]),
                   "variance_ratio": float(adjusted.var(ddof=ddof) / plain) if plain > 0 else float("nan"), "n": n})


def importance_shift(shift, n_draws, dimension, rng):
    """Importance sampling by exponential tilting: shifted normals, each with its likelihood weight.

    Draws Z ~ N(mu, I) in `dimension` coordinates and the weight w = exp(-mu . Z + |mu|^2 / 2), the density of N(0, I)
    over that of N(mu, I) at Z, so mean(f(Z) w) estimates E[f(X)], X standard normal, without bias. A shift toward
    where f pays (a rare event, a deep out-of-the-money option) cuts the variance: for a payoff of one terminal
    normal, a shift near the draw where the option starts to pay. Pass f(Z) w as the payoffs to qf.mc_price. The
    weights' mean is one and mean(w Z) is zero, a check on any run.

    conventions:
    - shift: mu, in standard deviations of the plain normal, one number for every coordinate or one a coordinate
    - n_draws: the draws, an int; dimension: the coordinates of a draw, an int (1 for one terminal normal, the
      steps for a path)
    - rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); one draw of standard normals
      (n_draws, dimension), then shifted by mu
    returns: a dict: normals (n_draws x dimension, the shifted draws); weights (the n_draws likelihood ratios)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 4.6.1; Boyle, Broadie and Glasserman
      (1997), Journal of Economic Dynamics and Control 21
    aliases: importance sampling, exponential tilting, rare-event sampling
    example:
        qf.importance_shift(shift=2.0, n_draws=1000, dimension=1,
                            rng=1)
        # result: {'normals.shape': (1000, 1), 'weights.shape': (1000,)}
    """
    rng = _stream(rng)
    count, dim = _count("n_draws", n_draws), _count("dimension", dimension)
    mu = np.broadcast_to(np.asarray(shift, dtype=float).ravel(), (dim,)) if np.size(shift) in (1, dim) else None
    if mu is None or not np.all(np.isfinite(mu)):
        raise ValueError("shift must be one finite number, or one a coordinate")
    z = rng.standard_normal((count, dim)) + mu
    return Result({"normals": z, "weights": np.exp(-z @ mu + 0.5 * float(mu @ mu))})


def stratified_normals(n_draws, strata, dimension, kind, allocation, stratum_stds, integrand, rng, ddof):
    """Stratified sampling and Latin hypercube sampling (LHS) of normals, with the stratified estimate of an integrand.

    'stratified': the first coordinate's probability cut into `strata` equal strata, p_k = 1 / strata, n_k draws in
    stratum k at Phi^-1((k + U) / strata), the other coordinates plain normals; the estimate sum_k p_k mean_k(f) has
    variance sum_k p_k^2 s_k^2 / n_k (s_k^2 the stratum's variance), never above plain Monte Carlo's at proportional
    allocation. 'proportional' takes n_k = n p_k; 'optimal' (Neyman) n_k in proportion to p_k s_k from a pilot's
    stratum_stds, two a stratum at least. 'latin_hypercube' (McKay, Beckman and Conover) cuts every coordinate into
    n_draws strata, one draw each, the coordinates paired by random permutations: the estimate is the plain mean, its
    error needs independent hypercubes (std_error nan). The estimate is sum(weights f).

    conventions:
    - n_draws: an int (a multiple of strata under 'proportional'); strata: the first coordinate's strata, an int (None
      for 'latin_hypercube'); dimension: the coordinates a draw
    - kind: 'stratified' or 'latin_hypercube'; allocation: 'proportional' or 'optimal' (None for the hypercube)
    - stratum_stds: under 'optimal' each stratum's standard deviation of the integrand from a pilot run, else None
    - integrand: None (the draws alone), or a callable of the n_draws x dimension array, one value a draw
    - rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); the stratified coordinate's
      uniforms, then the other normals; the hypercube's uniforms, then a permutation a coordinate
    - ddof: 1 or 0, each stratum's variance (1 wants two a stratum)
    returns: a dict: normals (n_draws x dimension); stratum (each draw's stratum of the first coordinate); weights (p_k /
      n_k a draw); estimate and std_error (nan without an integrand)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 4.3 and 4.4; McKay, Beckman and Conover
      (1979), Technometrics 21(2)
    aliases: stratified sampling, Latin hypercube sampling, [LHS], Neyman allocation, stratification
    example:
        qf.stratified_normals(n_draws=1000, strata=10, dimension=1, kind="stratified", allocation="proportional",
                              stratum_stds=None, integrand=lambda z: (z[:, 0] > 1.5) * 1.0, rng=2, ddof=1)
        # result: {'estimate': 0.068, 'std_error': 0.00468826}
    """
    choice("kind", kind, STRATA_KINDS)
    rng = _stream(rng)
    count, dim = _count("n_draws", n_draws), _count("dimension", dimension)
    ddof = _ddof(ddof)
    if integrand is not None and not callable(integrand):
        raise ValueError("integrand must be None or a callable of the draws")
    if kind == "latin_hypercube":
        if strata is not None or allocation is not None or stratum_stds is not None:
            raise ValueError("strata, allocation and stratum_stds are None under 'latin_hypercube'")
        u = rng.random((count, dim))
        cells = np.column_stack([rng.permutation(count) for _ in range(dim)])
        z = ndtri((cells + u) / count)
        stratum, weights = cells[:, 0], np.full(count, 1.0 / count)
        if integrand is None:
            return Result({"normals": z, "stratum": stratum, "weights": weights, "estimate": float("nan"),
                           "std_error": float("nan")})
        values = _values(integrand, z)
        return Result({"normals": z, "stratum": stratum, "weights": weights, "estimate": float(values.mean()),
                       "std_error": float("nan")})
    choice("allocation", allocation, ALLOCATIONS)
    k = _count("strata", strata)
    p = np.full(k, 1.0 / k)
    if allocation == "proportional":
        if stratum_stds is not None or count % k:
            raise ValueError("'proportional' takes n_draws a multiple of strata and stratum_stds None")
        sizes = np.full(k, count // k)
    else:
        spread = np.asarray(stratum_stds, dtype=float).ravel() if stratum_stds is not None else np.array([])
        if spread.shape != (k,) or np.any(~(spread >= 0)) or count < 2 * k:
            raise ValueError("'optimal' needs stratum_stds, one non-negative value a stratum, and two draws a stratum")
        share = p * spread
        share = share / share.sum() if share.sum() > 0 else p
        extra = (count - 2 * k) * share
        sizes = 2 + np.floor(extra).astype(int)
        left = count - sizes.sum()
        sizes[np.argsort(-(extra - np.floor(extra)), kind="stable")[:left]] += 1
    stratum = np.repeat(np.arange(k), sizes)
    u = rng.random((count, 1))
    first = ndtri((stratum[:, None] + u) / k)
    z = np.hstack([first, rng.standard_normal((count, dim - 1))])
    weights = (p / sizes)[stratum]
    if integrand is None:
        return Result({"normals": z, "stratum": stratum, "weights": weights, "estimate": float("nan"),
                       "std_error": float("nan")})
    values = _values(integrand, z)
    means = np.bincount(stratum, weights=values, minlength=k) / sizes
    if ddof and np.any(sizes < 2):
        raise ValueError("ddof 1 needs two draws a stratum")
    squares = np.bincount(stratum, weights=(values - means[stratum]) ** 2, minlength=k)
    variance = np.sum(p * p * squares / (sizes - ddof) / sizes)
    return Result({"normals": z, "stratum": stratum, "weights": weights, "estimate": float(p @ means),
                   "std_error": float(np.sqrt(variance))})


def _values(integrand, z):
    values = np.asarray(integrand(z), dtype=float).reshape(-1)
    if values.size != z.shape[0] or not np.all(np.isfinite(values)):
        raise ValueError("integrand must give one finite value a draw")
    return values


@_guard(level=('confidence',))
def sobol_normals(n_points, dimension, scramble, order, replications, integrand, rng, ddof, confidence):
    """Sobol sequence normals (QMC), scrambled for independent replications, in Brownian bridge construction.

    Sobol's low-discrepancy points u in [0, 1)^dimension (scipy.stats.qmc.Sobol), mapped to standard normals by
    Phi^-1(u); the error can fall near 1/n where plain Monte Carlo's falls as 1/sqrt(n). A power of two preserves the
    scrambled sequence's balance; the plain sequence skips its origin and loses that balance. Scrambled points (a
    random linear matrix scramble and digital shift) are each uniform, so independent replications give an unbiased
    estimate, the mean of their means, and an error, their standard deviation over sqrt(replications). The first coordinates are the most even: 'brownian_bridge' turns each
    point into the increments of a path of `dimension` equal steps built by the Brownian bridge, the first coordinate
    setting the path's end, the next its middle; 'sequential' feeds coordinate j to step j.

    conventions:
    - n_points: a power of two; dimension: the coordinates a point (a path's steps; 1 for one terminal draw)
    - scramble: True (scipy's scrambled Sobol) or False (the plain sequence, its first point, the origin, skipped;
      one replication)
    - order: 'sequential' or 'brownian_bridge' (the columns a path's standard normal increments, equal steps)
    - replications: independent scrambles, an int (two or more for an error)
    - integrand: None (the points alone), or a callable of the n_points x dimension normals, one value a point
    - rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream), for the scrambling; None
      with scramble False
    - ddof: 1 or 0 (the replications' variance); confidence: the interval's level, e.g. 0.95
    returns: a dict: normals (replications x n_points x dimension); estimate, std_error, low and high (from the
      replications' means; nan without an integrand, std_error nan with one replication)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), ch. 5 (5.5, the Brownian bridge); Owen
      (1997); scipy.stats.qmc.Sobol
    aliases: Sobol sequence, QMC, low-discrepancy sequence, scrambled Sobol
    example:
        qf.sobol_normals(n_points=1024, dimension=1, scramble=True, order="sequential", replications=8,
                         integrand=lambda z: z[:, 0] ** 2, rng=3, ddof=1, confidence=0.95)
        # result: {'estimate': 1.000384, 'std_error': 0.00070360}
    """
    choice("order", order, ORDERS)
    count, dim, reps = _count("n_points", n_points), _count("dimension", dimension), _count("replications",
                                                                                              replications)
    if count & (count - 1):
        raise ValueError("n_points must be a power of two (the sequence's balance holds at powers of two)")
    if not isinstance(scramble, (bool, np.bool_)):
        raise ValueError("scramble must be True or False")
    if integrand is not None and not callable(integrand):
        raise ValueError("integrand must be None or a callable of the normals")
    if scramble:
        rng = _stream(rng)
        seed = getattr(rng, "legacy", rng)      # scipy draws from a RandomState itself, not from its wrapper
        seeds = [qmc.Sobol(dim, scramble=True, seed=seed).random_base2(int(np.log2(count))) for _ in range(reps)]
    else:
        if rng is not None or reps != 1:
            raise ValueError("an unscrambled sequence is one replication, rng None")
        seeds = [qmc.Sobol(dim, scramble=False).random_base2(int(np.log2(count)) + 1)[1:count + 1]]
    z = ndtri(np.clip(np.stack(seeds), 1e-300, 1.0 - 1e-16))
    if order == "brownian_bridge":
        z = _bridge(z)
    found = {"normals": z, "estimate": float("nan"), "std_error": float("nan"), "low": float("nan"),
             "high": float("nan")}
    if integrand is None:
        return Result(found)
    means = np.array([_values(integrand, each).mean() for each in z])
    estimate = means.mean()
    error = means.std(ddof=_ddof(ddof)) / np.sqrt(reps) if reps > 1 else float("nan")
    level = _level(confidence)
    return Result({**found, "estimate": float(estimate), "std_error": float(error),
                   "low": float(estimate - level * error), "high": float(estimate + level * error)})


def _bridge(z):
    """Each point's coordinates as the standard normal increments of a path of equal steps built by the Brownian
    bridge: the first coordinate sets W at the last step, the next the midpoint, then the midpoints of the halves."""
    steps = z.shape[-1]
    w = np.zeros(z.shape[:-1] + (steps + 1,))
    w[..., steps] = np.sqrt(steps) * z[..., 0]
    queue, used = [(0, steps)], 1
    while queue:
        left, right = queue.pop(0)
        if right - left < 2:
            continue
        mid = (left + right) // 2
        a, b = mid - left, right - mid
        w[..., mid] = (b * w[..., left] + a * w[..., right]) / (a + b) + np.sqrt(a * b / (a + b)) * z[..., used]
        used += 1
        queue += [(left, mid), (mid, right)]
    return np.diff(w, axis=-1)


@_guard(level=('p', 'confidence',))
def estimate_error(estimator, samples, denominators, p, density, batches, ddof, confidence):
    """Batch means for autocorrelated draws, and the uncertainty of a quantile (by its density) or a quotient.

    'quantile': the p-quantile of the samples, numpy's 'inverted_cdf' (the ceil(n p)-th smallest), with std_error
    sqrt(p (1 - p) / n) / f(q), f the density at the quantile (a VaR's error from simulation). 'ratio': mean(Y) /
    mean(X) (a conditional expectation, a likelihood-ratio estimate) with the delta method's std_error sqrt((s_Y^2 - 2
    R s_XY + R^2 s_X^2) / n) / |mean(X)|. 'batch_means': the mean of draws that are correlated (a Markov chain, a long
    path's steps), the samples cut into equal consecutive batches whose means are near independent: std_error the
    batch means' standard deviation over sqrt(batches). The interval is estimate +- z std_error, z = Phi^-1((1 +
    confidence) / 2).

    conventions:
    - estimator: 'quantile', 'ratio' or 'batch_means'; samples: the n draws (Y for 'ratio')
    - denominators: under 'ratio' the n draws X on the same paths, else None
    - p: under 'quantile' the probability, inside (0, 1), else None; density: the density at the quantile, a number
      where it is known (a normal sample's), or 'kernel' (a Gaussian kernel estimate, Scott's bandwidth); None else
    - batches: under 'batch_means' the batches, an int dividing n (20 to 30 is usual), else None
    - ddof: 1 or 0 (the variances of 'ratio' and 'batch_means'), None under 'quantile'; confidence: e.g. 0.95
    returns: a dict: estimate, std_error, low and high (the interval)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 1.1.3 (the ratio), 9.1.1 (the quantile);
      Asmussen and Glynn, Stochastic Simulation (2007), IV.5 (batch means)
    aliases: batch means
    example:
        qf.estimate_error(estimator="quantile", samples=[-1.2, 0.3, 2.1, -0.4, 0.9, -2.3, 1.5, 0.1, -0.8, 0.6],
                          denominators=None, p=0.9, density=0.1755, batches=None,
                          ddof=None, confidence=0.95)
        # result: {'estimate': 1.5, 'std_error': 0.540560}
    """
    choice("estimator", estimator, ESTIMATORS)
    y = np.asarray(samples, dtype=float).ravel()
    n = y.size
    if n < 2 or not np.all(np.isfinite(y)):
        raise ValueError("samples must be two or more finite draws")
    z = _level(confidence)
    unused = {"quantile": ("denominators", "batches"), "ratio": ("p", "density", "batches"),
              "batch_means": ("denominators", "p", "density")}[estimator]
    given = {"denominators": denominators, "p": p, "density": density, "batches": batches}
    if any(given[name] is not None for name in unused):
        raise ValueError(f"{', '.join(unused)} are None under {estimator!r}")
    if estimator == "quantile":
        if ddof is not None:
            raise ValueError("ddof is None under 'quantile'")
        prob = float(p)
        if not 0.0 < prob < 1.0:
            raise ValueError("p must lie strictly inside (0, 1)")
        estimate = float(np.quantile(y, prob, method="inverted_cdf"))
        if isinstance(density, str):
            choice("density", density, DENSITIES)
            f = float(gaussian_kde(y)(estimate)[0])
        else:
            f = float(positive("density", density))
        error = np.sqrt(prob * (1.0 - prob) / n) / f
    elif estimator == "ratio":
        x = np.asarray(denominators, dtype=float).ravel()
        if x.shape != y.shape or not np.all(np.isfinite(x)) or x.mean() == 0:
            raise ValueError("denominators must be n finite draws with a non-zero mean")
        d = _ddof(ddof)
        estimate = y.mean() / x.mean()
        cov = np.cov(y, x, ddof=d)
        error = np.sqrt(max(cov[0, 0] - 2.0 * estimate * cov[0, 1] + estimate ** 2 * cov[1, 1], 0.0) / n) / abs(x.mean())
    else:
        b = _count("batches", batches)
        if b < 2 or n % b:
            raise ValueError("batches must be an int of at least 2 dividing the samples")
        means = y.reshape(b, n // b).mean(axis=1)
        estimate, error = y.mean(), means.std(ddof=_ddof(ddof)) / np.sqrt(b)
    return Result({"estimate": float(estimate), "std_error": float(error), "low": float(estimate - z * error),
                   "high": float(estimate + z * error)})


@_guard(time=('times',))
def first_hit_times(paths, times, level, direction):
    """First time each simulated path reaches a level, infinity where it never does.

    Row by row, the first times[j] whose value is at or above the level ('up') or at or below it ('down'); numpy.inf
    where none is. A column at time 0 counts: a path starting beyond the level hits at once. The monitoring is
    discrete: a crossing between two times that comes back is missed, so the share hit falls short of the continuous
    probability (qf.hit_probability), by about that of a level moved out by 0.5826 vol sqrt(dt) (Broadie, Glasserman
    and Kou's correction, vol in the paths' log or level units).

    conventions:
    - paths: n_paths x n_times values (as gbm_paths or ou_paths return them); times: their n_times increasing times
    - level: the barrier in the paths' units, a number or one per path
    - direction: 'up' (the first value at or above the level) or 'down' (at or below it)
    returns: an array of n_paths first-hit times, numpy.inf where the path never reaches the level
    source: Glasserman (2004), 6.4; Broadie, Glasserman and Kou (1997), Mathematical Finance 7(4)
    aliases: first passage time, first hitting time, first-passage simulation, barrier hitting time, [stopping time]
    example:
        qf.first_hit_times(paths=[[100.0, 104.0, 111.0], [100.0, 97.0, 99.0]],
                           times=[0.0, 0.5, 1.0], level=110.0,
                           direction="up")
    """
    choice("direction", direction, DIRECTIONS)
    x = np.asarray(paths, dtype=float)
    t = np.asarray(times, dtype=float).ravel()
    if x.ndim != 2 or x.shape[1] != t.size or np.any(np.diff(t) <= 0):
        raise ValueError("paths must be n_paths x n_times with times their n_times increasing times")
    barrier = np.asarray(level, dtype=float)
    if barrier.ndim:
        barrier = barrier.reshape(-1, 1)
    hit = x >= barrier if direction == "up" else x <= barrier
    first = np.argmax(hit, axis=1)
    return np.where(hit[np.arange(x.shape[0]), first], t[first], np.inf)


def _basis(x, basis, degree):
    """The regressors at x: 'polynomial' 1, x, ..., x^d; 'laguerre' L_0(x), ..., L_d(x); 'weighted_laguerre' 1 and
    e^{-x/2} L_0(x), ..., e^{-x/2} L_{d-1}(x) (Longstaff and Schwartz's)."""
    if basis == "polynomial":
        return np.vander(x, degree + 1, increasing=True)
    count = degree + 1 if basis == "laguerre" else degree
    columns = [np.polynomial.laguerre.lagval(x, np.eye(count)[k]) for k in range(count)]
    if basis == "laguerre":
        return np.column_stack(columns)
    return np.column_stack([np.ones_like(x)] + [np.exp(-x / 2) * column for column in columns])


@_guard(rate=('rate',), time=('times',))
def longstaff_schwartz(paths, times, payoff, rate, basis, degree, scale, itm_only, pricing_paths):
    """Least-squares Monte Carlo (Longstaff-Schwartz) for American and Bermudan exercise.

    Every time after times[0] is an exercise date. Backward from the last, at each date regress the realised cash
    flow, discounted to that date, on basis functions of x = S/scale over the paths in the money (or all), and
    exercise where the payoff is positive and exceeds the fitted continuation. The price is the mean of each path's
    cash flow at its first exercise, discounted to times[0], on pricing_paths under the fitted rule (fresh paths:
    an estimate biased low, the rule being suboptimal) or on the paths themselves. A finer grid of dates tends to the
    American price; at times[0] none is exercised, so an American's price is max(payoff(S0), price).

    conventions:
    - paths: n_paths x n_times underlying values at times (as gbm_paths returns), times[0] today; payoff: a
      callable of an array of values, the exercise value (lambda s: numpy.maximum(K - s, 0.0) for a put)
    - rate: r, continuously compounded, discounting between dates
    - basis: 'polynomial' (1, x, ..., x^d), 'laguerre' (L_0 to L_d) or 'weighted_laguerre' (1 and e^{-x/2} L_0 to
      L_{d-1}, Longstaff and Schwartz's); degree: d; scale: x = S/scale, the strike say (only the weighted basis
      depends on it)
    - itm_only: True regresses on the paths in the money alone (Longstaff and Schwartz), False on all
    - pricing_paths: None, or independent paths on the same times for the price (out of sample)
    returns: a dict: price; std_error (of the mean, ddof 1, the paths independent draws); coefficients (a row per exercise date but the last, nan
      where none was regressed); exercise_times (each pricing path's, numpy.inf where never exercised)
    source: Longstaff and Schwartz (2001), Review of Financial Studies 14(1); Glasserman (2004), 8.6
    aliases: Longstaff-Schwartz, least-squares Monte Carlo, LSM, LSMC, regression-based Monte Carlo, American option Monte Carlo, Bermudan option Monte Carlo, [early exercise]
    example:
        qf.longstaff_schwartz(paths=qf.gbm_paths(36.0, 0.06, 0.2, [k / 50 for k in range(51)], 2000, 1, "arithmetic", True),
            times=[k / 50 for k in range(51)], payoff=lambda s: (40.0 - s).clip(0.0), rate=0.06, basis="weighted_laguerre",
            degree=3, scale=40.0, itm_only=True, pricing_paths=None)
    """
    choice("basis", basis, BASES)
    x = np.asarray(paths, dtype=float)
    t = np.asarray(times, dtype=float).ravel()
    if x.ndim != 2 or x.shape[1] != t.size or t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("paths must be n_paths x n_times with times their n_times increasing times, at least two")
    if not callable(payoff):
        raise ValueError("payoff must be a callable of an array of underlying values")
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be an int of at least 1")
    if not isinstance(itm_only, (bool, np.bool_)):
        raise ValueError("itm_only must be True or False")
    r, unit = float(rate), float(positive("scale", scale))
    priced = x if pricing_paths is None else np.asarray(pricing_paths, dtype=float)
    if priced.ndim != 2 or priced.shape[1] != t.size:
        raise ValueError("pricing_paths must be n x n_times on the same times, or None")
    last = t.size - 1
    cash, when = np.asarray(payoff(x[:, last]), dtype=float), np.full(x.shape[0], last)
    beta = np.full((last, int(degree) + 1), np.nan)
    for j in range(last - 1, 0, -1):
        value = np.asarray(payoff(x[:, j]), dtype=float)
        keep = value > 0 if itm_only else np.ones(value.size, dtype=bool)
        if keep.sum() <= degree:
            continue
        design = _basis(x[keep, j] / unit, basis, int(degree))
        target = cash[keep] * np.exp(-r * (t[when[keep]] - t[j]))
        beta[j] = np.linalg.lstsq(design, target, rcond=None)[0]
        stop = np.zeros(value.size, dtype=bool)
        stop[keep] = (value[keep] > 0) & (value[keep] > design @ beta[j])
        cash[stop], when[stop] = value[stop], j
    flows, stopped = np.zeros(priced.shape[0]), np.full(priced.shape[0], np.inf)
    alive = np.ones(priced.shape[0], dtype=bool)
    for j in range(1, last + 1):
        value = np.asarray(payoff(priced[:, j]), dtype=float)
        if j == last:
            stop = alive & (value > 0)
        elif np.isnan(beta[j, 0]):
            continue
        else:
            stop = alive & (value > 0) & (value > _basis(priced[:, j] / unit, basis, int(degree)) @ beta[j])
        flows[stop], stopped[stop], alive = value[stop] * np.exp(-r * (t[j] - t[0])), t[j], alive & ~stop
    return Result({"price": float(flows.mean()), "std_error": float(flows.std(ddof=1) / np.sqrt(flows.size)),
                   "coefficients": beta[1:], "exercise_times": stopped})


@_guard(rate=('rate', 'dividend_yield',), vol=('vol',), time=('exercise_times',), price=('spot',),
        level=('confidence',))
def american_bounds(method, payoff, spot, rate, dividend_yield, vol, exercise_times, n_paths, branching, inner_paths,
                    basis, degree, rng, confidence):
    """American and Bermudan price bounds: the Broadie-Glasserman random tree, or Andersen-Broadie duality.

    Black-Scholes paths of drift r - q, exercise at exercise_times (not today, as qf.longstaff_schwartz: an American is
    worth max(payoff(spot), the bounds)). 'random_tree' (Broadie and Glasserman): n_paths trees of `branching` children
    a node; the high estimator takes max(exercise, the children's discounted mean), the low one decides each child's
    exercise on the other children and values it on that child. 'dual' (Andersen and Broadie): a Longstaff-Schwartz
    policy's price on fresh paths is the low bound; its values, each continuation from inner_paths sub-paths, build a
    martingale M, and high = E[max over dates of (discounted payoff - M)].

    conventions:
    - method: 'random_tree' or 'dual'; payoff: a callable of prices, the exercise value (lambda s: (100.0 -
      s).clip(0.0) for a put)
    - spot, rate r, dividend_yield q, vol: Black-Scholes inputs, annual; exercise_times: increasing years after today
    - n_paths: the trees, or under 'dual' each fresh set of paths (the regression's, the low bound's, the high's)
    - branching: under 'random_tree' the children a node (branching ** dates nodes a tree), else None
    - inner_paths: under 'dual' the sub-paths for a continuation value, else None
    - basis, degree: under 'dual' the regression's, as qf.longstaff_schwartz (on spot / today's spot), else None
    - rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream)
    - confidence: the interval's level, e.g. 0.95
    returns: a dict: low and high (the estimates), low_std_error and high_std_error, lower and upper (the interval's
      ends), n
    source: Glasserman (2004), 8.3 and 8.7; Broadie and Glasserman (1997), JEDC 21; Andersen and Broadie (2004)
    aliases: Broadie-Glasserman random tree, Andersen-Broadie, Bermudan bounds
    example:
        qf.american_bounds(method="random_tree", payoff=lambda s: (100.0 - s).clip(0.0), spot=100.0, rate=0.05,
                           dividend_yield=0.0, vol=0.2, exercise_times=[1 / 3, 2 / 3, 1.0], n_paths=100, branching=10,
                           inner_paths=None, basis=None, degree=None, rng=1, confidence=0.95)
        # result: {'low': 5.978125, 'high': 6.518449}
    """
    choice("method", method, BOUND_METHODS)
    rng = _stream(rng)
    s0, v = float(positive("spot", spot)), float(positive("vol", vol))
    r, carry = float(rate), float(rate) - float(dividend_yield)
    t = np.asarray(exercise_times, dtype=float).ravel()
    if t.size == 0 or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("exercise_times must be increasing years after today")
    if not callable(payoff):
        raise ValueError("payoff must be a callable of an array of prices")
    count, z = _count("n_paths", n_paths), _level(confidence)
    steps = np.diff(np.r_[0.0, t])

    def grow(s, h, normals):
        return s * np.exp((carry - 0.5 * v * v) * h + v * np.sqrt(h) * normals)

    def worth(s):
        return np.asarray(payoff(s), dtype=float).reshape(np.shape(s))

    if method == "random_tree":
        if inner_paths is not None or basis is not None or degree is not None:
            raise ValueError("inner_paths, basis and degree are None under 'random_tree'")
        b = _count("branching", branching)
        if b < 2 or count * b ** t.size > 20_000_000:
            raise ValueError("branching must be at least 2, and trees x branching ** dates at most 20 million nodes")
        levels, s = [], np.full((count, 1), s0)
        for h in steps:
            s = grow(np.repeat(s, b, axis=1), h, rng.standard_normal((count, s.shape[1] * b)))
            levels.append(s)
        high = low = worth(levels[-1])
        for j in range(t.size - 1, -1, -1):
            disc = np.exp(-r * steps[j])
            hi = disc * high.reshape(count, -1, b)
            lo = disc * low.reshape(count, -1, b)
            if j == 0:
                high, low = hi.mean(axis=2), lo.mean(axis=2)
                break
            now = worth(levels[j - 1])[:, :, None]
            others = (lo.sum(axis=2, keepdims=True) - lo) / (b - 1)
            high = np.maximum(now[:, :, 0], hi.mean(axis=2))
            low = np.where((now > 0) & (now >= others), now, lo).mean(axis=2)
        high, low = high.ravel(), low.ravel()
    else:
        if branching is not None or basis is None or degree is None:
            raise ValueError("'dual' takes basis and degree, and branching None")
        choice("basis", basis, BASES)
        inner = _count("inner_paths", inner_paths)
        times = np.r_[0.0, t]

        def paths(n):
            return np.cumprod(np.c_[np.full(n, s0), np.exp((carry - 0.5 * v * v) * steps + v * np.sqrt(steps)
                                                             * rng.standard_normal((n, t.size)))], axis=1)

        fit = longstaff_schwartz(paths=paths(count), times=times, payoff=payoff, rate=r, basis=basis, degree=degree,
                                 scale=s0, itm_only=True, pricing_paths=None)
        beta = np.vstack([fit["coefficients"], np.full((1, int(degree) + 1), np.nan)])   # a row a date, the last none
        low = _policy_cash(paths(count), times, payoff, r, beta, basis, int(degree), s0, 1)
        outer = paths(count)
        exercise_value = np.exp(-r * t) * worth(outer[:, 1:])
        held = np.zeros((count, t.size))            # each date's continuation value under the policy, discounted
        for j in range(1, t.size):
            normals = rng.standard_normal((count, inner, t.size - j))
            grown = outer[:, j, None, None] * np.exp(np.cumsum((carry - 0.5 * v * v) * steps[j:] + v
                                                               * np.sqrt(steps[j:]) * normals, axis=2))
            sub = np.concatenate([np.repeat(outer[:, None, :j + 1], inner, axis=1), grown], axis=2)
            cash = _policy_cash(sub.reshape(count * inner, -1), times, payoff, r, beta, basis, int(degree), s0, j + 1)
            held[:, j - 1] = cash.reshape(count, inner).mean(axis=1)
        stop = _stops(outer, times, payoff, beta, basis, int(degree), s0)
        value = np.where(stop, exercise_value, held)                   # L_j, the policy's value at each date
        martingale = np.cumsum(value, axis=1) - np.c_[np.zeros(count), np.cumsum(held[:, :-1], axis=1)]
        high = value[:, 0] + np.max(exercise_value - martingale, axis=1)
    lo_se, hi_se = low.std(ddof=1) / np.sqrt(low.size), high.std(ddof=1) / np.sqrt(high.size)
    lo_mean, hi_mean = float(low.mean()), float(high.mean())
    return Result({"low": lo_mean, "high": hi_mean, "low_std_error": float(lo_se), "high_std_error": float(hi_se),
                   "lower": float(lo_mean - z * lo_se), "upper": float(hi_mean + z * hi_se), "n": int(low.size)})


def _stops(paths, times, payoff, beta, basis, degree, scale):
    """Where the fitted policy exercises at each date after today: in the money and above the fitted continuation (at
    the last date, in the money); an n_paths x dates array of booleans."""
    found = np.zeros((paths.shape[0], times.size - 1), dtype=bool)
    for j in range(1, times.size):
        value = np.asarray(payoff(paths[:, j]), dtype=float)
        if j == times.size - 1:
            found[:, j - 1] = value > 0
        elif not np.isnan(beta[j - 1, 0]):
            found[:, j - 1] = (value > 0) & (value > _basis(paths[:, j] / scale, basis, degree) @ beta[j - 1])
    return found


def _policy_cash(paths, times, payoff, rate, beta, basis, degree, scale, first):
    """Each path's cash flow under the fitted policy from date `first` on, discounted to today."""
    stop = _stops(paths, times, payoff, beta, basis, degree, scale)[:, first - 1:]
    hit = stop.any(axis=1)
    when = first + np.argmax(stop, axis=1)
    value = np.asarray(payoff(paths[np.arange(paths.shape[0]), when]), dtype=float)
    return np.where(hit, value * np.exp(-rate * times[when]), 0.0)


@_guard(rate=('rate',), vol=('vol',), time=('times',), price=('strike', 'barrier',))
def barrier_mc_price(paths, times, option, barrier_type, strike, barrier, rate, monitoring, vol, antithetic):
    """Barrier price by Monte Carlo on simulated paths: discretely monitored, or continuously by the Brownian bridge.

    The payoff max(+-(S_T - K), 0) at the last date T = times[-1], paid if a knock-out never hit the barrier H (an in:
    if it did). 'discrete' checks H at every column (the monitoring dates). 'bridge' monitors continuously: between
    dates the path crosses H with the Brownian bridge's probability exp(-2 ln(S_j/H) ln(S_{j+1}/H)/(vol^2 dt)) (1
    where an end is past H), a knock-out paying its payoff times the steps' survivals. Prices are discounted by
    e^{-rT}; the vanilla on the same paths is the control (barrier <= vanilla, in + out = vanilla). Against the
    closed form (qf.barrier_price): 'bridge' converges to it; 'discrete' at interval dt to it with H moved away from
    the spot by e^{0.5826 vol sqrt(dt)} (Broadie-Glasserman-Kou).

    conventions:
    - paths: n_paths x len(times), as qf.gbm_paths or qf.local_vol_paths give them; times: their dates in years,
      monitoring from the first column on (times from 0, the spot's column, to monitor from today)
    - option: 'call' or 'put'; barrier_type: 'down-and-out', 'down-and-in', 'up-and-out' or 'up-and-in'
    - strike: K; barrier: H; rate: r continuously compounded, for e^{-rT}
    - monitoring: 'discrete' (vol None) or 'bridge' (vol: the annual vol of each step, a number or n_paths x
      (len(times) - 1))
    - antithetic: True where the second half of the paths mirrors the first (pairs averaged for the errors)
    returns: a dict: price; std_error; vanilla (the same paths' European price); vanilla_std_error;
      hit_probability (of touching H)
    source: Glasserman, Monte Carlo Methods in Financial Engineering (2004), 6.4 (the bridge's crossing
      probability); Broadie, Glasserman and Kou (1997), Mathematical Finance 7(4); Hull, "Barrier options"
    aliases: barrier Monte Carlo, Monte Carlo barrier, discretely monitored barrier, discrete barrier, Brownian bridge barrier, barrier hit probability
    example:
        qf.barrier_mc_price(paths=qf.gbm_paths(100.0, 0.03, 0.2, [0.25, 0.5, 0.75, 1.0], 2000, 5, "arithmetic", True),
                            times=[0.25, 0.5, 0.75, 1.0], option="call", barrier_type="down-and-out", strike=100.0,
                            barrier=85.0, rate=0.03, monitoring="discrete", vol=None, antithetic=True)
    """
    choice("option", option, OPTIONS)
    choice("barrier_type", barrier_type, BARRIER_TYPES)
    choice("monitoring", monitoring, MONITORS)
    s = np.asarray(paths, dtype=float)
    t = _times(times)
    if s.ndim != 2 or s.shape[1] != t.size or np.any(~(s > 0)):
        raise ValueError("paths must be n_paths x len(times), every price positive")
    k, h = float(positive("strike", strike)), float(positive("barrier", barrier))
    if not isinstance(antithetic, (bool, np.bool_)) or (antithetic and s.shape[0] % 2):
        raise ValueError("antithetic must be True or False, and True needs an even number of paths")
    down = barrier_type.startswith("down")
    beyond = s <= h if down else s >= h
    if monitoring == "discrete":
        if vol is not None:
            raise ValueError("vol is None under 'discrete' monitoring")
        survive = (~beyond.any(axis=1)).astype(float)
    else:
        if t.size < 2:
            raise ValueError("'bridge' needs two dates or more")
        v = np.broadcast_to(np.asarray(vol, dtype=float), (s.shape[0], t.size - 1))
        if np.any(~(v > 0)):
            raise ValueError("vol must be positive under 'bridge'")
        logs = np.log(s / h)
        cross = np.exp(-2.0 * logs[:, :-1] * logs[:, 1:] / (v * v * np.diff(t)))
        cross = np.where(beyond[:, :-1] | beyond[:, 1:], 1.0, np.minimum(cross, 1.0))
        survive = np.prod(1.0 - cross, axis=1)            # a step with an end past H crosses surely
    vanilla = np.maximum((1.0 if option == "call" else -1.0) * (s[:, -1] - k), 0.0) * np.exp(-float(rate) * t[-1])
    knocked = 1.0 - survive
    payoff = vanilla * (survive if barrier_type.endswith("out") else knocked)

    def estimate(values):
        if antithetic:
            half = values.size // 2
            values = 0.5 * (values[:half] + values[half:])
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))

    price, error = estimate(payoff)
    plain, plain_error = estimate(vanilla)
    return Result({"price": price, "std_error": error, "vanilla": plain, "vanilla_std_error": plain_error,
                   "hit_probability": float(knocked.mean())})
