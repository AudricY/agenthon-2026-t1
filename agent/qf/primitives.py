"""The pieces when no function fits: likelihood, GMM, quadrature, bivariate and noncentral laws, Panjer, lattices, roots.

`import qf`. The pieces a method no other shelf holds is built from (the owner's in633): a fit of any model from a
log-density, moment conditions or residuals the caller writes as a function; an expectation by quadrature; the
bivariate, multivariate and noncentral chi-squared laws; the Panjer and lattice recursions; a bracketing root for an
implied parameter; a distribution discretised onto a grid. Where a function on another shelf computes the quantity
(cir_fit, implied_vol, binomial_price, aggregate_loss), that function comes first. Parameters pass by name, as a
dict; a function with several results returns a dict with named keys.
"""
from __future__ import annotations

import functools
import inspect
import sys
import warnings

import numpy as np
from scipy import integrate, optimize, stats
from scipy.special import expit, logit

from ._common import Result, bivariate_normal_cdf as _bivariate, choice, integer, ndtr, out, positive

__all__ = ["bivariate_normal_cdf", "density_expectation", "discretize_distribution", "implied_parameter",
           "lattice_backward", "least_squares_fit", "mle_fit", "moments_fit", "multivariate_normal_cdf",
           "noncentral_chi2", "ols_lstsq", "panjer_recursion"]

FORMS = ("observations", "transitions")
TRANSFORMS = ("free", "positive", "unit_interval")
OPTIMISERS = ("bfgs", "nelder_mead", "powell")
ERRORS = ("hessian", "outer_product", "sandwich", "none")
WEIGHTINGS = ("identity", "two_step")
LOSSES = ("linear", "soft_l1", "huber", "cauchy")
SINGULAR_POLICIES = ("min_norm", "refuse")
QUADRATURES = ("gauss_hermite", "gauss_legendre", "adaptive")
MATRICES = ("covariance", "correlation")
LAW_VALUES = ("cdf", "sf", "pdf", "quantile")
LAW_FORMS = ("standard", "cir", "cev")
LAW_PARAMETERS = {"standard": ("df", "noncentrality"), "cir": ("kappa", "theta", "sigma", "rate", "horizon"),
                  "cev": ("spot", "rate", "dividend_yield", "sigma", "beta", "maturity")}
COUNTINGS = ("poisson", "negative_binomial", "binomial", "ab0")
COUNTING_PARAMETERS = {"poisson": ("lambda",), "negative_binomial": ("r", "beta"), "binomial": ("m", "q"),
                       "ab0": ("a", "b", "p0")}
BRANCHINGS = ("binomial", "trinomial")
EXERCISES = ("european", "american", "bermudan")
EXERCISERS = ("holder", "issuer")
EXPANSIONS = ("none", "up", "down", "both")
ROOTS = ("brent", "bisection")
DISCRETISATIONS = ("rounding", "lower", "upper", "moment_matching")
GRADIENT_STEP = 6e-6            # central differences: about the cube root of the double's epsilon, relative to the
#                                 value or, nearer zero, to its start's size (`_scale`)
HESSIAN_STEP = 1.2e-4           # second differences: about its fourth root
EXPANSION_PASSES = 60           # a bracket doubles at most this often looking for the sign change
SPAN_NODES = 20                 # Gauss-Legendre nodes a span for the limited expected values of 'moment_matching'
SERIES_TERMS = 100_000          # the (a, b, 0) series for g_0 stops here or where its terms vanish


# ------------------------------------------------------------------------------------------------- the fits


def _maps(names, transforms):
    """Each parameter's map from the optimiser's free value u: theta(u), u(theta) and dtheta/du at theta."""
    if not isinstance(transforms, dict) or set(transforms) != set(names):
        raise ValueError(f"transforms must be a dict naming every parameter: {', '.join(map(repr, names))}")
    found = []
    for name in names:
        kind = transforms[name]
        if isinstance(kind, str):
            choice(f"transforms[{name!r}]", kind, TRANSFORMS)
            if kind == "free":
                found.append((lambda u: u, lambda t: t, lambda t: 1.0))
            elif kind == "positive":
                found.append((np.exp, np.log, lambda t: t))
            else:
                found.append((expit, logit, lambda t: t * (1.0 - t)))
            continue
        low, high = (float(end) for end in kind)
        if not low < high:
            raise ValueError(f"transforms[{name!r}]: an interval (low, high) needs low < high")
        found.append((lambda u, low=low, high=high: low + (high - low) * expit(u),
                      lambda t, low=low, high=high: logit((t - low) / (high - low)),
                      lambda t, low=low, high=high: (t - low) * (high - t) / (high - low)))
    return found


def _starts(starts):
    if not isinstance(starts, dict) or not starts:
        raise ValueError("starts must be a dict of each parameter's name and its start value")
    return list(starts), np.array([float(value) for value in starts.values()])


def _free(maps, theta):
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.array([to_u(t) for (_, to_u, _), t in zip(maps, theta)], dtype=float)
    if not np.all(np.isfinite(u)):
        raise ValueError("every start value must lie strictly inside its transform's bounds")
    return u


def _theta(maps, u):
    return np.array([to_theta(x) for (to_theta, _, _), x in zip(maps, u)], dtype=float)


def _jacobian(maps, theta):
    return np.array([slope(t) for (_, _, slope), t in zip(maps, theta)], dtype=float)


def _scale(u0):
    """Each value's size for its difference steps: its start's magnitude, 1 where the start is 0."""
    return np.where(u0 != 0.0, np.abs(u0), 1.0)


def _gradient(f, u, scale):
    """Central differences of f (a scalar or an array) at u, the last axis the parameter's."""
    steps = GRADIENT_STEP * np.maximum(np.abs(u), scale)
    columns = []
    for j, step in enumerate(steps):
        e = np.zeros_like(u)
        e[j] = step
        columns.append((np.asarray(f(u + e), dtype=float) - np.asarray(f(u - e), dtype=float)) / (2.0 * step))
    return np.stack(columns, axis=-1)


def _hessian(f, u, scale):
    """Second central differences of the scalar f at u."""
    steps = HESSIAN_STEP * np.maximum(np.abs(u), scale)
    k = u.size
    found = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = steps[i], steps[j]
            found[i, j] = found[j, i] = (f(u + ei + ej) - f(u + ei - ej) - f(u - ei + ej) + f(u - ei - ej)) / (
                4.0 * steps[i] * steps[j])
    return found


def _minimise(objective, u0, optimiser, scale):
    def safe(u):
        value = float(objective(u))
        return value if np.isfinite(value) else np.inf

    if optimiser == "bfgs":
        found = optimize.minimize(safe, u0, jac=lambda u: _gradient(safe, u, scale), method="BFGS",
                                  options={"gtol": 1e-9, "maxiter": 20_000})
    elif optimiser == "nelder_mead":
        found = optimize.minimize(safe, u0, method="Nelder-Mead",
                                  options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 40_000, "maxfev": 40_000,
                                           "adaptive": True})
    else:
        found = optimize.minimize(safe, u0, method="Powell", options={"xtol": 1e-10, "ftol": 1e-14, "maxiter": 40_000})
    return found


def _named(names, values):
    return Result({name: float(value) for name, value in zip(names, values)})


def _flags(checks, params):
    if checks is None:
        return Result()
    if not isinstance(checks, dict):
        raise ValueError("checks must be None or a dict of a name and a function of the parameters' dict")
    return Result({name: bool(check(params)) for name, check in checks.items()})


def mle_fit(log_density, data, form, starts, transforms, optimiser, errors, checks):
    """Maximum likelihood (MLE) of any model from its log-density, with AIC and BIC.

    theta-hat maximises l(theta) = sum_i ln f(x_i; theta), each parameter optimised in a free value u (u itself;
    e^u, positive; 1/(1 + e^-u), in (0, 1); an interval alike) and mapped back. Errors: 'hessian' minus the Hessian
    of l, inverted; 'outer_product' the summed outer products of each term's score, inverted (BHHH); 'sandwich'
    H^-1 B H^-1 (robust); each mapped by the delta method. AIC = 2k - 2l, BIC = k ln n - 2l.

    Used for: a short-rate model (CIR, Vasicek) from a rate series by its transition density, a jump diffusion, a
    Hawkes process, GEV and GPD tails, ARMA and state-space likelihoods, any return density. Held fits come first:
    cir_fit, ou_fit, merton_jump_fit, garch11_fit, t_copula_fit, gpd_fit. Errors by resampling: refit on
    block_bootstrap's draws.

    conventions:
    - log_density: a function of (params, x) giving each observation's ln f, an array (params a dict by name);
      with 'transitions', of (params, x_next, x_previous) over consecutive pairs of the series
    - data: the observations, an array (with 'transitions' the series in time order)
    - form: 'observations' (independent draws) or 'transitions' (a Markov series: a transition density)
    - starts: a dict of each parameter's name and start value (its order kept in the results)
    - transforms: a dict, each parameter's bound: 'free', 'positive', 'unit_interval' or a (low, high) interval
    - optimiser: 'bfgs', 'nelder_mead' or 'powell'; errors: 'hessian', 'outer_product', 'sandwich' or 'none'
    - checks: None, or a dict of a name and a function of params giving True or False (Feller's 2 kappa theta >=
      sigma^2), each returned in flags
    returns: a dict: params, std_errors (by name); covariance; loglik; aic; bic; n_obs; converged; message; flags
    source: Casella and Berger, Statistical Inference (2002), 7.2.2 and 10.1; White (1982), Econometrica 50(1)
    aliases: likelihood maximisation, [MLE]
    example:
        import qf, numpy as np, scipy.stats as st
        x = np.array([0.8, 1.9, 0.4, 1.1, 2.7, 0.6, 1.4, 0.9])
        # Result: params rate 0.81632653.
        qf.mle_fit(log_density=lambda p, x: st.expon.logpdf(x, scale=1.0 / p["rate"]), data=x, form="observations", starts={"rate": 1.0}, transforms={"rate": "positive"}, optimiser="bfgs", errors="hessian", checks=None)
    """
    choice("form", form, FORMS)
    choice("optimiser", optimiser, OPTIMISERS)
    choice("errors", errors, ERRORS)
    names, theta0 = _starts(starts)
    maps = _maps(names, transforms)
    x = np.asarray(data, dtype=float)
    if form == "transitions":
        if x.shape[0] < 3:
            raise ValueError("form 'transitions' needs a series of at least three values")

        def terms(theta):
            return np.asarray(log_density(_named(names, theta), x[1:], x[:-1]), dtype=float)
    else:
        def terms(theta):
            return np.asarray(log_density(_named(names, theta), x), dtype=float)

    n = x.shape[0] - (form == "transitions")
    first = terms(theta0)
    if first.ndim != 1 or first.size != n:
        raise ValueError(f"log_density must give one log-density a term, an array of {n} (it gave shape "
                         f"{first.shape})")

    def per_term(u):
        return terms(_theta(maps, u))

    def objective(u):
        return -float(np.sum(per_term(u))) / n

    u0 = _free(maps, theta0)
    scale = _scale(u0)
    found = _minimise(objective, u0, optimiser, scale)
    u = found.x
    theta = _theta(maps, u)
    loglik = float(np.sum(per_term(u)))
    k = len(names)
    covariance = np.full((k, k), np.nan)
    if errors != "none":
        hessian = -_hessian(lambda v: float(np.sum(per_term(v))), u, scale)
        scores = _gradient(per_term, u, scale)
        outer = scores.T @ scores
        with np.errstate(all="ignore"):
            try:
                if errors == "hessian":
                    inner = np.linalg.inv(hessian)
                elif errors == "outer_product":
                    inner = np.linalg.inv(outer)
                else:
                    bread = np.linalg.inv(hessian)
                    inner = bread @ outer @ bread
            except np.linalg.LinAlgError:
                inner = np.full((k, k), np.nan)
        slope = _jacobian(maps, theta)
        covariance = slope[:, None] * inner * slope[None, :]
    params = _named(names, theta)
    gradient = float(np.max(np.abs(_gradient(objective, u, scale))))
    return Result({"params": params, "std_errors": _named(names, np.sqrt(np.diag(covariance))),
                   "covariance": covariance, "loglik": loglik, "aic": 2.0 * k - 2.0 * loglik,
                   "bic": k * np.log(n) - 2.0 * loglik, "n_obs": int(n),
                   "converged": bool(found.success or gradient < 1e-7), "message": str(found.message),
                   "flags": _flags(checks, params)})


def _long_run(g, lags):
    """Newey-West's long-run covariance of the rows of g, uncentred, Bartlett weights 1 - l/(L + 1)."""
    n = g.shape[0]
    found = g.T @ g / n
    for lag in range(1, lags + 1):
        gamma = g[lag:].T @ g[:-lag] / n
        found = found + (1.0 - lag / (lags + 1.0)) * (gamma + gamma.T)
    return found


def moments_fit(moments, data, starts, transforms, weighting, lags, optimiser):
    """GMM (generalized method of moments) from any model's orthogonality conditions: two-step weighting, Hansen's J.

    theta-hat minimises g-bar(theta)' W g-bar(theta), g-bar the mean over the observations of the moment conditions
    g_i(theta) (expected zero at the true theta). 'identity': W = I, errors (G'G)^-1 G'SG (G'G)^-1 / n; 'two_step':
    a first fit at W = I, then W = S^-1 at it and a second fit, asymptotic errors (G'S^-1 G)^-1 / n (S recomputed
    at the final fit) and Hansen's J = n times
    the minimised criterion, chi-squared with m - k degrees of freedom. G = d g-bar / d theta; S the long-run
    covariance of g_i, uncentred, Newey-West with Bartlett weights over `lags` (0: the plain mean of g_i g_i').
    Exactly identified (m = k): the moments solved, whatever W, J zero.

    Used for: matching a model's moments to data (a jump diffusion's mean, variance, skew and kurtosis; an OU
    process; a gamma or Pareto law), asset-pricing moment conditions (an Euler equation, a stochastic discount
    factor). Parameters each optimised in a free value as mle_fit's transforms say.

    conventions:
    - moments: a function of (params, data) giving the moment conditions, an array with a row an observation and
      a column a condition (params a dict by name)
    - starts: a dict of each parameter's name and start value; transforms: as mle_fit's ('free', 'positive',
      'unit_interval' or a (low, high) interval)
    - weighting: 'identity' or 'two_step' (Hansen's efficient two-step)
    - lags: the Newey-West lags of S (0 for independent observations)
    - optimiser: 'bfgs', 'nelder_mead' or 'powell'
    returns: a dict: params and std_errors (dicts by name); covariance; objective (the minimised criterion);
      j_stat, j_dof and j_pvalue (nan under 'identity'); n_obs; converged; message
    source: Hansen (1982), Econometrica 50(4); Cochrane, Asset Pricing (2005), ch. 10 and 11
    aliases: generalized method of moments, [GMM], [Hansen's J]
    example:
        import qf, numpy as np
        x = np.array([1.2, 0.7, 2.9, 1.8, 0.4, 2.2, 1.1, 3.4])
        # Result: params shape 2.97401363, scale 0.57582117.
        qf.moments_fit(moments=lambda p, x: np.column_stack([x - p["shape"] * p["scale"], (x - x.mean()) ** 2 - p["shape"] * p["scale"] ** 2]), data=x, starts={"shape": 1.0, "scale": 1.0}, transforms={"shape": "positive", "scale": "positive"}, weighting="identity", lags=0, optimiser="bfgs")
    """
    choice("weighting", weighting, WEIGHTINGS)
    choice("optimiser", optimiser, OPTIMISERS)
    lags = integer("lags", lags, 0)
    names, theta0 = _starts(starts)
    maps = _maps(names, transforms)
    k = len(names)

    def conditions(u):
        found = np.asarray(moments(_named(names, _theta(maps, u)), data), dtype=float)
        return found[:, None] if found.ndim == 1 else found

    g0 = conditions(_free(maps, theta0))
    n, m = g0.shape
    if m < k:
        raise ValueError(f"{m} moment condition(s) cannot identify {k} parameter(s)")

    def criterion(u, weight):
        mean = conditions(u).mean(axis=0)
        return float(mean @ weight @ mean)

    u0 = _free(maps, theta0)
    scale = _scale(u0)
    found = _minimise(lambda u: criterion(u, np.eye(m)), u0, optimiser, scale)
    weight = np.eye(m)
    if weighting == "two_step":
        weight = np.linalg.inv(_long_run(conditions(found.x), lags))
        found = _minimise(lambda u: criterion(u, weight), found.x, optimiser, scale)
    u = found.x
    theta = _theta(maps, u)
    g = conditions(u)
    s = _long_run(g, lags)
    slope_u = _gradient(lambda v: conditions(v).mean(axis=0), u, scale)        # m x k
    with np.errstate(all="ignore"):
        try:
            if weighting == "two_step":
                inner = np.linalg.inv(slope_u.T @ np.linalg.inv(s) @ slope_u) / n
            else:
                bread = np.linalg.inv(slope_u.T @ slope_u)
                inner = bread @ slope_u.T @ s @ slope_u @ bread / n
        except np.linalg.LinAlgError:
            inner = np.full((k, k), np.nan)
    slope = _jacobian(maps, theta)
    covariance = slope[:, None] * inner * slope[None, :]
    objective = criterion(u, weight)
    j_stat, j_dof, j_pvalue = np.nan, m - k, np.nan
    if weighting == "two_step":
        j_stat = n * objective
        j_pvalue = float(stats.chi2.sf(j_stat, j_dof)) if j_dof > 0 else np.nan
    gradient = float(np.max(np.abs(_gradient(lambda v: criterion(v, weight), u, scale))))
    return Result({"params": _named(names, theta), "std_errors": _named(names, np.sqrt(np.diag(covariance))),
                   "covariance": covariance, "objective": objective, "j_stat": float(j_stat), "j_dof": int(j_dof),
                   "j_pvalue": float(j_pvalue), "n_obs": int(n),
                   "converged": bool(found.success or gradient < 1e-9), "message": str(found.message)})


def least_squares_fit(residuals, starts, bounds, weights, loss, loss_scale):
    """Nonlinear least squares (NLS) of any model coded by the caller, by a trust region.

    theta-hat minimises sum_i rho(w_i r_i(theta)^2), r_i the residuals the caller writes (model minus target, or
    the reverse), rho the loss ('linear': the plain sum of squares), by scipy's trust-region reflective method
    within the bounds. Errors s^2 (J'WJ)^-1, J = dr/dtheta at the optimum (differences kept inside the bounds), s^2 = sum_i
    w_i r_i^2 / (n - k), the usual linearized regression errors; under a robust loss the same formula is a guide.

    Used for: calibrating any model to prices, vols or yields when no calibration is held: SABR to a smile,
    Vasicek or a short-rate model to a curve, a vol term structure, a parametric smile, a decay or a growth curve.
    Held calibrations come first: nelson_siegel_fit, svi_fit, heston_calibrate, hull_white_calibrate.

    conventions:
    - residuals: a function of the params dict giving the residuals, an array
    - starts: a dict of each parameter's name and start value, inside the bounds
    - bounds: None (no bounds), or a dict of a name and its (low, high), numpy.inf for an open side
    - weights: None (equal), or w_i an array as long as the residuals (a vega or an inverse variance)
    - loss: 'linear' (least squares), 'soft_l1', 'huber' or 'cauchy' (robust, scipy's)
    - loss_scale: the residual size where a robust loss turns (scipy's f_scale; 'linear' ignores it, pass 1.0)
    returns: a dict: params and std_errors (dicts by name); covariance; residuals (unweighted, at the fit); sse
      (sum w r^2); dof (n - k); converged; message
    source: NIST StRD nonlinear regression (Misra1a); Bates and Watts, Nonlinear Regression Analysis (1988), 2.3
    aliases: trust-region reflective, [NLS]
    example:
        import qf, numpy as np
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([8.2, 6.6, 5.5, 4.5, 3.6, 3.0])
        # Result: params level 9.99364520, decay 0.20156209.
        qf.least_squares_fit(residuals=lambda p: p["level"] * np.exp(-p["decay"] * t) - y, starts={"level": 5.0, "decay": 0.1}, bounds=None, weights=None, loss="linear", loss_scale=1.0)
    """
    choice("loss", loss, LOSSES)
    names, theta0 = _starts(starts)
    low, high = np.full(len(names), -np.inf), np.full(len(names), np.inf)
    if bounds is not None:
        if not isinstance(bounds, dict) or set(bounds) - set(names):
            raise ValueError("bounds must be None or a dict of parameter names and (low, high)")
        for j, name in enumerate(names):
            if name in bounds:
                low[j], high[j] = (float(end) for end in bounds[name])
    if not np.all((low < theta0) & (theta0 < high)):
        raise ValueError("every start value must lie strictly inside its bounds")
    scale = float(positive("loss_scale", loss_scale))
    first = np.asarray(residuals(_named(names, theta0)), dtype=float).ravel()
    root = np.ones(first.size) if weights is None else np.sqrt(np.asarray(positive("weights", weights), dtype=float))
    if root.shape != first.shape:
        raise ValueError(f"weights must be as long as the residuals ({first.size})")

    def weighted(theta):
        return root * np.asarray(residuals(_named(names, theta)), dtype=float).ravel()

    sizes = _scale(theta0)

    def jacobian(theta):
        steps = GRADIENT_STEP * np.maximum(np.abs(theta), sizes)
        columns = []
        for j, step in enumerate(steps):
            left, right = theta[j] - low[j], high[j] - theta[j]
            room = min(left, right)
            if step > room and room > 0.0:
                local_step = GRADIENT_STEP * room
                if theta[j] + local_step != theta[j] and theta[j] - local_step != theta[j]:
                    step = local_step
            e = np.zeros_like(theta)
            if step <= room:
                e[j] = step
                column = (weighted(theta + e) - weighted(theta - e)) / (2.0 * step)
            else:
                direction = 1.0 if right >= left else -1.0
                step = min(step, max(left, right) / 2.0)
                e[j] = direction * step
                column = direction * (-3.0 * weighted(theta) + 4.0 * weighted(theta + e)
                                      - weighted(theta + 2.0 * e)) / (2.0 * step)
            columns.append(column)
        return np.stack(columns, axis=-1)

    found = optimize.least_squares(weighted, theta0, jac=jacobian, bounds=(low, high), method="trf", loss=loss,
                                   f_scale=scale, x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                   max_nfev=20_000)
    theta = found.x
    n, k = first.size, len(names)
    fitted = weighted(theta)
    sse = float(fitted @ fitted)
    jac = jacobian(theta)
    with np.errstate(all="ignore"):
        try:
            covariance = sse / (n - k) * np.linalg.inv(jac.T @ jac) if n > k else np.full((k, k), np.nan)
        except np.linalg.LinAlgError:
            covariance = np.full((k, k), np.nan)
    return Result({"params": _named(names, theta), "std_errors": _named(names, np.sqrt(np.diag(covariance))),
                   "covariance": covariance, "residuals": fitted / root, "sse": sse, "dof": int(n - k),
                   "converged": bool(found.success), "message": str(found.message)})


def ols_lstsq(y, X, intercept, singular):
    """OLS on a design that may be singular: the minimum-norm (pseudo-inverse) solution, or refused.

    b = X^+ y, the Moore-Penrose pseudo-inverse's solution and numpy.linalg.lstsq's (rcond=None: singular values below
    eps max(n, p) times the largest taken as zero), the least-squares fit of smallest |b|. With full column rank it is
    qf.ols's b, and qf.ols comes first there (its errors and t need the full rank); with rank r < p the fitted values
    are still y's projection and collinear columns share their coefficient (two equal columns take half each).
    e = y - X b, s^2 = e'e / (n - r), R^2 as qf.ols's (centred with an intercept).

    Used for: a cross-sectional or rolling regression whose design can lose rank (a constant or repeated column, a
    dummy for every group beside the intercept), a task that names the pseudo-inverse or lstsq; qf.ols where X has
    full rank.

    conventions:
    - singular: 'min_norm' (the pseudo-inverse's solution where X lacks full column rank) or 'refuse' (a ValueError
      there, as qf.ols)
    - intercept: True puts a ones column first (coef[0] the constant), False fits through the origin
    - y: (n,) and X: (n, k) or (n,), finite (drop rows with NaN first), with more rows than X's rank
    returns: a dict: coef (one per column, the constant first with an intercept); resid and fitted (one per row);
      rank (r); dof (n - r); sigma2 (s^2); r2
    source: Golub and Van Loan, Matrix Computations (2013), 5.5 (rank-deficient least squares); Penrose (1955),
      "A generalized inverse for matrices", Proc. Cambridge Philosophical Society 51(3)
    aliases: pseudo-inverse regression, Moore-Penrose pseudo-inverse, minimum-norm least squares, lstsq regression, singular design matrix, rank-deficient regression
    example:
        # Result: coef [0.0, 0.5, 0.5] (to rounding), the two equal columns sharing the slope 1.0.
        import qf
        qf.ols_lstsq(y=[1.0, 2.0, 3.0, 4.0], X=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], intercept=True, singular="min_norm")
    """
    choice("singular", singular, SINGULAR_POLICIES)
    if not isinstance(intercept, (bool, np.bool_)):
        raise ValueError(f"intercept must be True or False, not {intercept!r}")
    target, design = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    design = design[:, None] if design.ndim == 1 else design
    if target.ndim != 1 or design.ndim != 2 or design.shape[0] != target.size:
        raise ValueError("y must be (n,) and X (n, k) or (n,) with the same n")
    if intercept:
        design = np.column_stack([np.ones(target.size), design])
    if design.shape[1] == 0 or not (np.all(np.isfinite(target)) and np.all(np.isfinite(design))):
        raise ValueError("y and X must be finite (drop the rows with NaN first) and X must have a column")
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    rank = int(rank)
    if rank < design.shape[1] and singular == "refuse":
        raise ValueError(f"X has rank {rank} below its {design.shape[1]} columns: singular='min_norm' takes the "
                         "pseudo-inverse's solution")
    if target.size <= rank:
        raise ValueError("y needs more rows than X's rank")
    resid = target - design @ coef
    dof = target.size - rank
    total = np.sum((target - target.mean()) ** 2) if intercept else np.sum(target * target)
    return Result({"coef": coef, "resid": resid, "fitted": target - resid, "rank": rank, "dof": dof,
                   "sigma2": float(resid @ resid / dof),
                   "r2": float(1.0 - resid @ resid / total) if total > 0 else float("nan")})


# ------------------------------------------------------------------------------------------ quadrature and laws

def density_expectation(function, law, method, limits, points, tolerance):
    """Gauss-Hermite quadrature, E[g(Z)] of a normal Z by its nodes; Gauss-Legendre or adaptive quadrature over a density.

    'gauss_hermite': X normal (mu, sigma), E[g(X)] = sum_i (w_i / sqrt(pi)) g(mu + sqrt(2) sigma t_i), t_i and w_i
    the physicists' Hermite nodes and weights (numpy's hermgauss; the probabilists' nodes s_i = sqrt(2) t_i with
    weights w_i sqrt(2) give the same sum), exact for g a polynomial of degree below 2 n. 'gauss_legendre': the
    integral of g f over (low, high) by n Legendre nodes; 'adaptive': scipy's quad over the limits, infinite ends
    allowed. f is the law's density; with no law, the plain integral of g.

    Used for: a one-factor model's expectation over its normal factor (a large-portfolio loss and its expected
    shortfall, tranche losses, a conditional default probability averaged), a replication integral (CMS, a variance
    swap's strip), a swaption integral (G2++), a spectral risk measure, any option payoff under a density held as a
    function. Over many factors or a path, simulate instead: mc_payoff_price on the montecarlo shelf.

    conventions:
    - function: g, a function of x (numpy arrays in, an array out, for the Gauss rules)
    - law: a scipy.stats frozen law (normal for 'gauss_hermite': norm(loc=mu, scale=sigma)), a density f(x) as a
      function, or None (the plain integral of g)
    - method: 'gauss_hermite', 'gauss_legendre' or 'adaptive'
    - limits: None (the whole line; the law's support under 'adaptive'), or (low, high), finite for
      'gauss_legendre'
    - points: n, the nodes of a Gauss rule (None for 'adaptive'); tolerance: the absolute and relative error of
      'adaptive' (None for a Gauss rule)
    returns: a float, the expectation (or the integral)
    source: Abramowitz and Stegun (1964), 25.4.29 and 25.4.46; Glasserman, Monte Carlo Methods (2003), 9.4
    aliases: Gauss-Hermite quadrature, Gauss-Hermite nodes, [Gauss-Legendre quadrature], [adaptive quadrature]
    example:
        import qf, numpy as np, scipy.stats as st
        # Result: 1.02020134.
        qf.density_expectation(function=lambda z: np.exp(0.2 * z), law=st.norm(loc=0.0, scale=1.0), method="gauss_hermite", limits=None, points=20, tolerance=None)
    """
    choice("method", method, QUADRATURES)
    if method == "adaptive":
        if points is not None or tolerance is None:
            raise ValueError("'adaptive' takes tolerance and points=None")
        tol = float(positive("tolerance", tolerance))
    else:
        if tolerance is not None:
            raise ValueError(f"{method!r} takes points and tolerance=None")
        n = integer("points", points, 1)
    if method == "gauss_hermite":
        if limits is not None or getattr(getattr(law, "dist", None), "name", "") != "norm":
            raise ValueError("'gauss_hermite' takes a normal law (scipy.stats.norm(loc=, scale=)) and limits=None")
        t, w = np.polynomial.hermite.hermgauss(n)
        x = float(law.mean()) + np.sqrt(2.0) * float(law.std()) * t
        return float(w @ np.broadcast_to(np.asarray(function(x), dtype=float), x.shape) / np.sqrt(np.pi))
    density = (law.pdf if hasattr(law, "pdf") else law) if law is not None else (lambda x: 1.0)
    if limits is None:
        if not hasattr(law, "support"):
            raise ValueError("limits=None needs a scipy.stats frozen law (its support); give (low, high)")
        low, high = (float(end) for end in law.support())
    else:
        low, high = (float(end) for end in limits)
    if not low < high:
        raise ValueError("limits must be (low, high) with low < high")
    if method == "gauss_legendre":
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError("'gauss_legendre' needs finite limits")
        t, w = np.polynomial.legendre.leggauss(n)
        x = 0.5 * (high - low) * t + 0.5 * (high + low)
        weight = np.broadcast_to(np.asarray(density(x), dtype=float), x.shape)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.where(weight == 0.0, 0.0, np.asarray(function(x), dtype=float) * weight)
        return float(0.5 * (high - low) * (w @ values))

    def integrand(x):           # g f, 0 where f is: g may overflow where the law has no mass
        weight = float(density(x))
        if weight == 0.0:
            return 0.0
        with np.errstate(over="ignore"):
            return float(function(x)) * weight
    return float(integrate.quad(integrand, low, high, epsabs=tol, epsrel=tol, limit=500)[0])


def bivariate_normal_cdf(h, k, rho):
    """Bivariate normal M(h, k; rho) = P(X <= h, Y <= k), by quadrature of its defining integral.

    M(h, k; rho) = int_{-inf}^h phi(x) N((k - rho x) / sqrt(1 - rho^2)) dx, by adaptive quadrature (abs(rho) < 1),
    with angular correlation quadrature near singularity or in the tails;
    rho = 1 gives N(min(h, k)) and rho = -1 max(N(h) + N(k) - 1, 0). M(0, 0; rho) = 1/4 + arcsin(rho)/(2 pi);
    rho = 0 gives N(h) N(k). Arrays broadcast.

    Used for: two-asset options (rainbow on the max or min, exchange), partial-time barriers, simple and complex
    choosers, compound options and Roll-Geske-Whaley, a large-portfolio expected shortfall, the joint default of two
    names. Three or more variables: multivariate_normal_cdf.

    conventions:
    - h, k: the upper limits in standard deviations (a variable of mean m and std s at x: h = (x - m)/s)
    - rho: the correlation, between -1 and 1
    returns: a float (an array where the inputs are arrays)
    source: Hull, Options, Futures, and Other Derivatives, Technical Note 5 (the definition); Owen (1956), its T form
    aliases: M(a, b; rho)
    example:
        import qf
        # Result: 0.33202625.
        qf.bivariate_normal_cdf(h=0.5, k=-0.2, rho=0.3)
    """
    h, k, rho = np.broadcast_arrays(np.asarray(h, dtype=float), np.asarray(k, dtype=float),
                                    np.asarray(rho, dtype=float))
    if np.any(np.isnan(rho)) or np.any(np.abs(rho) > 1.0):
        raise ValueError("rho must lie between -1 and 1")
    found = np.empty(h.shape)
    for at in np.ndindex(h.shape):
        a, b, r = float(h[at]), float(k[at]), float(rho[at])
        if np.isnan(a) or np.isnan(b):
            found[at] = np.nan
        elif a == -np.inf or b == -np.inf:
            found[at] = 0.0
        elif a == np.inf:
            found[at] = ndtr(b)
        elif b == np.inf:
            found[at] = ndtr(a)
        elif r == 1.0:
            found[at] = ndtr(min(a, b))
        elif r == -1.0:
            found[at] = max(ndtr(a) + ndtr(b) - 1.0, 0.0)
        elif r == 0.0:
            found[at] = ndtr(a) * ndtr(b)
        elif a == b == 0.0:
            found[at] = 0.25 + np.arcsin(r) / (2.0 * np.pi)
        elif abs(r) >= 0.99 or max(abs(a), abs(b)) >= 8.0:
            a, b = np.clip([a, b], -40.0, 40.0)

            def angular(theta):
                sine = np.sin(theta)
                exponent = -0.25 * ((a + b) ** 2 / (1.0 + sine) + (a - b) ** 2 / (1.0 - sine))
                return np.exp(exponent) / (2.0 * np.pi)

            correction = integrate.quad(angular, 0.0, np.arcsin(r), epsabs=1e-13, epsrel=1e-13)[0]
            found[at] = np.clip(ndtr(a) * ndtr(b) + correction, 0.0, min(ndtr(a), ndtr(b)))
        else:
            found[at] = _bivariate(a, b, r)
    return out(found)


def multivariate_normal_cdf(upper, mean, matrix, matrix_kind, tolerance, seed):
    """Trivariate and higher multivariate normal P(X <= b) by Genz's algorithm (scipy).

    X ~ N(mean, Sigma); P(X <= b) by Genz's algorithm (scipy.stats.multivariate_normal.cdf), randomised quasi-Monte
    Carlo to the absolute error `tolerance`, its points drawn from `seed`. Two variables equal bivariate_normal_cdf;
    the trivariate orthant P(all <= 0) of standard normals is 1/8 + sum_{i<j} arcsin(rho_ij)/(4 pi).

    Used for: options on the maximum or minimum of three or more assets, the joint default of several names, a
    multivariate probability no closed form gives.

    conventions:
    - upper: b, the upper limits, one a variable (numpy.inf for none)
    - mean: the variables' means (zeros for standard normals)
    - matrix and matrix_kind: 'covariance' (Sigma) or 'correlation' (unit variances, so the limits are in standard
      deviations from the means)
    - tolerance: the absolute error (scipy's abseps); seed: an int, the quasi-Monte Carlo's draws
    returns: a float
    source: Genz (1992), Journal of Computational and Graphical Statistics 1(2); Genz and Bretz (2009)
    aliases: trivariate normal, Genz algorithm
    example:
        import qf, numpy as np
        # Result: about 0.37860432, within the integration tolerance.
        qf.multivariate_normal_cdf(upper=[0.5, 0.0, 1.0], mean=[0.0, 0.0, 0.0], matrix=np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]), matrix_kind="correlation", tolerance=1e-6, seed=7)
    """
    choice("matrix_kind", matrix_kind, MATRICES)
    b = np.atleast_1d(np.asarray(upper, dtype=float))
    mu = np.atleast_1d(np.asarray(mean, dtype=float))
    sigma = np.atleast_2d(np.asarray(matrix, dtype=float))
    d = b.size
    if mu.shape != (d,) or sigma.shape != (d, d):
        raise ValueError(f"mean must have {d} entries and matrix be {d} by {d}, as upper")
    if not np.allclose(sigma, sigma.T):
        raise ValueError("matrix must be symmetric")
    if matrix_kind == "correlation" and not np.allclose(np.diag(sigma), 1.0):
        raise ValueError("a correlation matrix has ones on its diagonal")
    tol = float(positive("tolerance", tolerance))
    seed = integer("seed", seed, 0)
    if d == 1:
        positive("variance", sigma[0, 0])
        return float(ndtr((b[0] - mu[0]) / np.sqrt(sigma[0, 0])))
    return float(stats.multivariate_normal.cdf(b, mean=mu, cov=sigma, abseps=tol, releps=0.0,
                                               rng=np.random.default_rng(seed)))


def _cev(parameters):
    """Schroder's a(K)'s scale, b and c under the CEV process dS = (r - q) S dt + sigma S^beta dW (Hull 27.1)."""
    s = float(positive("spot", parameters["spot"]))
    r, q = float(parameters["rate"]), float(parameters["dividend_yield"])
    sigma, beta = float(positive("sigma", parameters["sigma"])), float(parameters["beta"])
    t = float(positive("maturity", parameters["maturity"]))
    if beta == 1.0 or beta < 0.0:
        raise ValueError("beta must be at least 0 and not 1 (beta 1 is Black-Scholes' lognormal law)")
    y = 2.0 * (r - q) * (beta - 1.0) * t
    v = sigma * sigma * t * (np.expm1(y) / y if y != 0.0 else 1.0)       # sigma^2 (e^y - 1) / (2 (r - q)(beta - 1))
    scale = (1.0 - beta) ** 2 * v
    return s, r - q, t, beta, scale, 1.0 / (1.0 - beta), s ** (2.0 * (1.0 - beta)) / scale


def noncentral_chi2(what, x, form, parameters):
    """Noncentral chi-squared law (ncx2), plain or as a CIR rate's or a CEV price's (Schroder's a, b, c).

    'standard': Z ~ chi'^2(df, lambda), scipy's ncx2. 'cir': the short rate r_T of dr = kappa (theta - r) dt + sigma
    sqrt(r) dW, 2c r_T ~ chi'^2(4 kappa theta / sigma^2, 2c r_0 e^{-kappa T}), c = 2 kappa / (sigma^2 (1 -
    e^{-kappa T})), mean theta + (r_0 - theta) e^{-kappa T}. 'cev': the price S_T of dS = (r - q) S dt + sigma
    S^beta dW, risk-neutral, by Schroder's a = (K e^{-(r - q) T})^{2(1 - beta)} / ((1 - beta)^2 v), b = 1/(1 -
    beta), c = S_0^{2(1 - beta)} / ((1 - beta)^2 v), v = sigma^2 (e^{2(r - q)(beta - 1) T} - 1) / (2 (r - q)(beta
    - 1)) (sigma^2 T at r = q): below beta 1 P(S_T > K) = F(c; b, a), the mass at 0 absorbed; above it P(S_T > K)
    = F(a; 2 - b, c). F(z; k, lambda) is the 'standard' CDF. Call = S_0 e^{-qT} (1 - F(a; b + 2, c)) - K e^{-rT}
    F(c; b, a) below beta 1; above, S_0 e^{-qT} (F(c; -b, 0) - F(c; -b, a)) - K e^{-rT} F(a; 2 - b, c).
    Above beta 1 the discounted price is a strict local martingale: E[S_T] = S_0 e^{(r-q)T} F(c; -b, 0).
    Replacing F(c; -b, 0) by 1 gives the put-call-parity solution, exceeding the payoff expectation by the defect.

    Used for: CEV calls from the formulas above, CIR bond options and transition likelihoods (mle_fit; held
    cir_fit), exact CIR sampling (quantile of uniforms), a CIR horizon quantile.

    conventions:
    - what: 'cdf' P(<= x), 'sf' P(> x), 'pdf' the density at x, or 'quantile' (x a probability)
    - form: 'standard' (x the chi-squared variable), 'cir' (x the short rate at the horizon) or 'cev' (x the
      price at maturity)
    - parameters: a dict: 'standard' {'df', 'noncentrality'}; 'cir' {'kappa', 'theta', 'sigma', 'rate' (r_0
      today), 'horizon' (T, years)}; 'cev' {'spot', 'rate', 'dividend_yield', 'sigma', 'beta' (exponent on S),
      'maturity'}; 'cir' rate may be an array beside x (previous rates for consecutive-pair transitions)
    returns: a float (an array where x is an array)
    source: Johnson, Kotz and Balakrishnan (1995), ch. 29; Cox, Ingersoll and Ross (1985); Schroder (1989);
      Lindsay and Brecher (2012), Applied Mathematical Finance, eqs. 18, 20-21
    aliases: Schroder CEV formula, [ncx2]
    example:
        import qf
        # Result: 0.69980364.
        qf.noncentral_chi2(what="cdf", x=0.04, form="cir", parameters={"kappa": 0.5, "theta": 0.04, "sigma": 0.1, "rate": 0.03, "horizon": 1.0})
    """
    choice("what", what, LAW_VALUES)
    choice("form", form, LAW_FORMS)
    if not isinstance(parameters, dict) or set(parameters) != set(LAW_PARAMETERS[form]):
        raise ValueError(f"parameters for {form!r} must be a dict of exactly "
                         f"{', '.join(map(repr, LAW_PARAMETERS[form]))}")
    x = np.asarray(x, dtype=float)
    if what == "quantile" and np.any((x < 0) | (x > 1)):
        raise ValueError("a quantile's x is a probability, between 0 and 1")
    if form == "standard":
        law = stats.ncx2(float(positive("df", parameters["df"])), float(parameters["noncentrality"]))
        if float(parameters["noncentrality"]) < 0:
            raise ValueError("noncentrality must not be negative")
        return out({"cdf": law.cdf, "sf": law.sf, "pdf": law.pdf, "quantile": law.ppf}[what](x))
    if form == "cir":
        kappa, theta, sigma, horizon = (float(positive(name, parameters[name]))
                                        for name in ("kappa", "theta", "sigma", "horizon"))
        r0 = np.asarray(parameters["rate"], dtype=float)
        if np.any(r0 < 0):
            raise ValueError("rate (r_0) must not be negative")
        c = 2.0 * kappa / (sigma * sigma * -np.expm1(-kappa * horizon))
        law = stats.ncx2(4.0 * kappa * theta / (sigma * sigma), 2.0 * c * r0 * np.exp(-kappa * horizon))
        if what == "quantile":
            return out(law.ppf(x) / (2.0 * c))
        if what == "pdf":
            return out(2.0 * c * law.pdf(2.0 * c * x))
        return out((law.cdf if what == "cdf" else law.sf)(2.0 * c * x))
    s, carry, t, beta, scale, b, c = _cev(parameters)

    def a_of(level):
        with np.errstate(divide="ignore"):
            return (level * np.exp(-carry * t)) ** (2.0 * (1.0 - beta)) / scale

    if beta < 1.0:
        def above(level):           # P(S_T > level): the mass absorbed at 0 lies below every positive level
            level = np.asarray(level, dtype=float)
            return np.where(level > 0, stats.ncx2.cdf(c, b, a_of(np.maximum(level, 1e-300))),
                            np.where(level == 0, stats.chi2.cdf(c, b), 1.0))

        def below(level):
            level = np.asarray(level, dtype=float)
            return np.where(level > 0, stats.ncx2.sf(c, b, a_of(np.maximum(level, 1e-300))),
                            np.where(level == 0, stats.chi2.sf(c, b), 0.0))
        if what == "pdf":
            a = a_of(np.maximum(x, 1e-300))
            slope = 2.0 * (1.0 - beta) * a / np.maximum(x, 1e-300)
            found = stats.ncx2.pdf(c, b + 2.0, a) * slope
            return out(np.where(x > 0, found, 0.0))
        if what == "quantile":
            atom = float(stats.chi2.sf(c, b))
            mean = s * np.exp(carry * t)

            def one(u):
                if u <= atom:
                    return 0.0
                if u >= 1.0:
                    return np.inf
                high = mean
                while float(below(high)) < u:
                    high *= 2.0
                return optimize.brentq(lambda level: float(below(level)) - u, 0.0, high, xtol=1e-13 * high,
                                       rtol=1e-14)
            return out(np.vectorize(one, otypes=[float])(x))
        return out(above(x) if what == "sf" else below(x))
    law = stats.ncx2(2.0 - b, c)
    if what == "quantile":
        a = law.isf(x)
        return out((a * scale) ** (1.0 / (2.0 * (1.0 - beta))) * np.exp(carry * t))
    a = a_of(np.maximum(x, 1e-300))
    if what == "pdf":
        slope = 2.0 * (beta - 1.0) * a / np.maximum(x, 1e-300)
        return out(np.where(x > 0, law.pdf(a) * slope, 0.0))
    found = np.where(x > 0, law.cdf(a), 1.0)
    return out(found if what == "sf" else np.where(x > 0, law.sf(a), 0.0))


# -------------------------------------------------------------------------------------------- the recursions


def _counts(counting, parameters, argument="parameters"):
    """The count's (a, b), its probability generating function P(z), its mean and variance (Loss Models, 6.5)."""
    if not isinstance(parameters, dict) or set(parameters) != set(COUNTING_PARAMETERS[counting]):
        raise ValueError(f"{argument} for {counting!r} must be a dict of exactly "
                         f"{', '.join(map(repr, COUNTING_PARAMETERS[counting]))}")
    if counting == "poisson":
        lam = float(parameters["lambda"])
        if not np.isfinite(lam) or not lam >= 0:
            raise ValueError("lambda must be finite and not negative")
        return 0.0, lam, lambda z: np.exp(lam * (z - 1.0)), lam, lam
    if counting == "negative_binomial":
        r, beta = float(positive("r", parameters["r"])), float(positive("beta", parameters["beta"]))
        return (beta / (1 + beta), (r - 1) * beta / (1 + beta), lambda z: (1.0 - beta * (z - 1.0)) ** -r, r * beta,
                r * beta * (1 + beta))
    if counting == "binomial":
        m, q = integer("m", parameters["m"], 1), float(parameters["q"])
        if not 0.0 < q < 1.0:
            raise ValueError("q must lie strictly inside (0, 1)")
        return -q / (1 - q), (m + 1) * q / (1 - q), lambda z: (1.0 + q * (z - 1.0)) ** m, m * q, m * q * (1 - q)
    a, b, p0 = float(parameters["a"]), float(parameters["b"]), float(parameters["p0"])
    if not (np.isfinite(a) and np.isfinite(b) and a < 1.0 and 0.0 < p0 <= 1.0):
        raise ValueError("'ab0' needs a < 1 and 0 < p0 <= 1")
    if a == 0.0:
        log_p0 = -b
        valid = b >= 0.0
    elif a > 0.0:
        log_p0 = (1.0 + b / a) * np.log1p(-a)
        valid = a + b >= 0.0
    else:
        m = -b / a - 1.0
        log_p0 = -m * np.log1p(-a)
        valid = m >= 0.0 and np.isclose(m, np.rint(m), rtol=0.0, atol=1e-9)
    if not valid or not np.isclose(np.log(p0), log_p0, rtol=0.0, atol=1e-9):
        raise ValueError("'ab0' must define a nonnegative, normalized count distribution")

    def generating(z):          # sum_k p_k z^k, p_k = (a + b/k) p_{k-1} from p_0
        z = float(np.real(z))
        total = term = p0
        for k in range(1, SERIES_TERMS):
            term *= (a + b / k) * z
            total += term
            if term == 0.0 or (abs(term) <= 1e-17 * abs(total) and abs((a + b / (k + 1)) * z) < 1.0):
                break
        return total
    return a, b, generating, (a + b) / (1 - a), (a + b) / (1 - a) ** 2


def _panjer(a, b, generating, f):
    """Panjer's recursion on the severity f over its grid: g_0 = P(f_0), g_k = sum_{j<=k} (a + b j/k) f_j g_{k-j}
    / (1 - a f_0)."""
    n = f.size
    g = np.empty(n)
    g[0] = float(np.real(generating(f[0])))
    if g[0] == 0.0:
        raise ValueError("the initial probability underflowed; split the count and convolve the partial sums")
    j = np.arange(n)
    for k in range(1, n):
        g[k] = ((a + b * j[1:k + 1] / k) * f[1:k + 1]) @ g[k - 1::-1][:k] / (1.0 - a * f[0])
    return g


def panjer_recursion(counting, parameters, severity, points):
    """Panjer recursion for compound Poisson, negative binomial or binomial sums of a discretised severity.

    S = X_1 + ... + X_N, X_i on the grid 0, h, 2h, ... with probabilities f_j, N of the (a, b, 0) class (p_k = (a
    + b/k) p_{k-1}): g_0 = P_N(f_0), P_N the count's generating function, and g_k = (1/(1 - a f_0)) sum_{j=1}^k (a +
    b j/k) f_j g_{k-j}. Poisson a = 0, b = lambda; negative binomial a = beta/(1 + beta), b = (r - 1) beta/(1 +
    beta); binomial a = -q/(1 - q), b = (m + 1) q/(1 - q). Exact on the grid; mean E[N] E[X], variance E[N] Var[X]
    + Var[N] E[X]^2. If the Poisson initial probability underflows, split its intensity and self-convolve.

    Used for: aggregate claim and loss distributions from a severity already on a grid (discretize_distribution
    puts a continuous one there), CreditRisk+'s loss in exposure bands, any compound sum. Held: aggregate_loss
    (the severity's law to VaR and ES in one call, by this recursion or FFT), creditrisk_plus.

    conventions:
    - counting: 'poisson', 'negative_binomial', 'binomial', or 'ab0' (the class's a and b with P(N = 0) as p0)
    - parameters: a dict: {'lambda'}; {'r', 'beta'} (mean r beta, numpy's and scipy's p = 1/(1 + beta)); {'m',
      'q'}; or {'a', 'b', 'p0'}
    - severity: f_j = P(X = j h), the claim size's probabilities on the grid from 0 (a shorter array is padded
      with zeros)
    - points: the grid's length n for S
    returns: an array, g_k = P(S = k h) for k below n
    source: Panjer (1981), ASTIN Bulletin 12(1); Klugman, Panjer and Willmot, Loss Models (2019), 9.6
    aliases: Panjer algorithm, [(a, b, 0) recursion]
    example:
        import qf
        # Result starts [0.04978707, 0.07468060, 0.10081881].
        qf.panjer_recursion(counting="poisson", parameters={"lambda": 3.0}, severity=[0.0, 0.5, 0.3, 0.2], points=40)
    """
    choice("counting", counting, COUNTINGS)
    a, b, generating, _, _ = _counts(counting, parameters)
    n = integer("points", points, 1)
    given = np.asarray(severity, dtype=float).ravel()
    if given.size == 0 or np.any(given < 0) or given.sum() > 1.0 + 1e-9:
        raise ValueError("severity must be probabilities on the grid, none negative, their sum at most 1")
    f = np.zeros(n)
    f[:min(n, given.size)] = given[:n]
    if counting == "poisson" and float(parameters["lambda"]) * (1.0 - f[0]) > 700.0:
        passes = int(np.ceil(np.log2(float(parameters["lambda"]) * (1.0 - f[0]) / 300.0)))
        a, b, generating, _, _ = _counts("poisson", {"lambda": float(parameters["lambda"]) / 2.0 ** passes})
        g = _panjer(a, b, generating, f)
        for _ in range(passes):
            g = np.convolve(g, g)[:n]
        return g
    return _panjer(a, b, generating, f)


def _layered(argument, given, sizes, width=None):
    """A per-step input as one array a step. Probabilities (`width` branches): one row for every node of every step,
    or a list by step of rows or arrays; a discount or a flow: a number for every node, or a list by step of a
    number or an array of the step's nodes."""
    steps = len(sizes) - 1
    if width is None:
        by_step = isinstance(given, (list, tuple)) or np.ndim(given) >= 1
    else:
        by_step = (isinstance(given, (list, tuple)) and len(given) > 0 and np.ndim(given[0]) >= 1) or (
            isinstance(given, np.ndarray) and given.ndim >= 2)
    if by_step and len(given) != steps:
        raise ValueError(f"{argument}: a list by step needs one entry a step ({steps}), not {len(given)}")
    layers = ([np.asarray(layer, dtype=float) for layer in given] if by_step
              else [np.asarray(given, dtype=float)] * steps)
    found = []
    for i, layer in enumerate(layers):
        shape = (sizes[i],) if width is None else (sizes[i], width)
        try:
            found.append(np.broadcast_to(layer, shape))
        except ValueError:
            raise ValueError(f"{argument} at step {i} must fit {shape} (the step's nodes"
                             f"{'' if width is None else ', its branches'})") from None
    return found


def lattice_backward(terminal, branching, probabilities, discount, cash_flows, exercise, exercise_value,
                     exercise_steps, exerciser, paid_on_exercise):
    """Backward induction on any recombining tree (binomial, trinomial, own branching): American, Bermudan, callable.

    V_i = D_i sum_b p_b V_{i+1}(child b) + c_i back from the last date, D_i the node's one-step discount and c_i
    its cash flow; where exercise is allowed the holder takes the max with the exercise value E_i (the issuer the
    min: a callable bond). Nodes count from the lowest: binomial step i has i + 1, node j going to j + 1 (up) and
    j (down); trinomial 2i + 1, node j to j + 2, j + 1 and j.

    Used for: a callable or puttable bond on a rate tree not fitted to a curve, a convertible, American or Bermudan
    exercise on dynamics no function holds. Held: binomial_price (stock trees), short_rate_tree (fitted rate trees).

    conventions:
    - terminal: the last date's values, lowest node first, its last flow included
    - branching: 'binomial', 'trinomial', or a list by step of integer arrays (a row a node: the next step's nodes)
    - probabilities: in the branching's order (up, down; up, middle, down): one row for all, or a list by step
    - discount: D, a number, or a list by step (e^{-r dt} by node on a rate tree)
    - cash_flows: None, or a list by step from step 0 of each node's flow
    - exercise: 'european', 'american' or 'bermudan' (at exercise_steps, else None); exercise_value: None, or a
      function of the step i giving E at its nodes (or a list by step)
    - exerciser: 'holder' (max) or 'issuer' (min); paid_on_exercise: True where the step's flow is paid as well (a
      coupon, then the call), False where the exercise value replaces it
    returns: a dict: value; values and exercised (by step: the node values, where exercise is taken)
    source: Cox, Ross and Rubinstein (1979); Hull, Options, Futures, and Other Derivatives, 21.4 and 32.4
    aliases: recombining tree, [backward induction], [trinomial tree]
    example:
        import qf, numpy as np
        s = 100.0 * 1.1 ** np.arange(-2, 3, 2)
        # Result: value 4.5.
        qf.lattice_backward(terminal=np.maximum(100.0 - s, 0.0), branching="binomial", probabilities=[0.5, 0.5], discount=0.99, cash_flows=None, exercise="american", exercise_value=lambda i: np.maximum(100.0 - 100.0 * 1.1 ** np.arange(-i, i + 1, 2), 0.0), exercise_steps=None, exerciser="holder", paid_on_exercise=True)
    """
    choice("exercise", exercise, EXERCISES)
    choice("exerciser", exerciser, EXERCISERS)
    last = np.asarray(terminal, dtype=float).ravel()
    if isinstance(branching, str):
        choice("branching", branching, BRANCHINGS)
        width = 2 if branching == "binomial" else 3
        steps = last.size - 1 if width == 2 else (last.size - 1) // 2
        if steps < 1 or (width == 3 and last.size % 2 == 0):
            raise ValueError(f"a {branching} terminal layer has {'n + 1' if width == 2 else '2n + 1'} nodes, n >= 1")
        sizes = [i + 1 if width == 2 else 2 * i + 1 for i in range(steps + 1)]
        children = [np.arange(size)[:, None] + np.arange(width - 1, -1, -1)[None, :] for size in sizes[:-1]]
    else:
        children = [np.asarray(layer, dtype=float) for layer in branching]
        if not children or any(layer.ndim != 2 or layer.size == 0 or not np.all(np.isfinite(layer))
                               or np.any(layer != np.floor(layer)) for layer in children):
            raise ValueError("branching must have nonempty matrices of integer node indices")
        children = [layer.astype(int) for layer in children]
        steps, width = len(children), children[0].shape[1]
        sizes = [layer.shape[0] for layer in children] + [last.size]
        if sizes[0] != 1:
            raise ValueError("branching must start with one root node")
        for i, layer in enumerate(children):
            if layer.ndim != 2 or layer.shape[1] != width or layer.min() < 0 or layer.max() >= sizes[i + 1]:
                raise ValueError(f"branching at step {i}: a row a node, each a node of step {i + 1}")
    probability = _layered("probabilities", probabilities, sizes, width)
    for i, layer in enumerate(probability):
        if np.any(layer < -1e-12) or not np.allclose(layer.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError(f"probabilities at step {i}: each node's must be non-negative and sum to 1")
    discounts = _layered("discount", discount, sizes)
    flows = _layered("cash_flows", 0.0 if cash_flows is None else cash_flows, sizes)
    if exercise == "european":
        allowed = set()
        if exercise_value is not None or exercise_steps is not None:
            raise ValueError("'european' takes exercise_value=None and exercise_steps=None")
    else:
        if exercise_value is None:
            raise ValueError(f"{exercise!r} needs exercise_value")
        if exercise == "american":
            if exercise_steps is not None:
                raise ValueError("'american' takes exercise_steps=None (every step)")
            allowed = set(range(steps))
        else:
            allowed = {integer("exercise_steps", step, 0) for step in exercise_steps}
            if not allowed or max(allowed) >= steps:
                raise ValueError(f"exercise_steps must name steps from 0 to {steps - 1} (the last date's is terminal)")
    value = last
    values, exercised = [None] * (steps + 1), [np.zeros(last.size, dtype=bool)] * (steps + 1)
    values[steps] = value
    for i in range(steps - 1, -1, -1):
        held = discounts[i] * np.sum(probability[i] * value[children[i]], axis=1)
        if i in allowed:
            worth = exercise_value(i) if callable(exercise_value) else exercise_value[i]
            worth = np.broadcast_to(np.asarray(worth, dtype=float), (sizes[i],))
            base = held if paid_on_exercise else held + flows[i]
            better = worth > base if exerciser == "holder" else worth < base
            value = np.where(better, worth, base) + (flows[i] if paid_on_exercise else 0.0)
            exercised[i] = better
        else:
            value = held + flows[i]
            exercised[i] = np.zeros(sizes[i], dtype=bool)
        values[i] = value
    return Result({"value": float(values[0][0]), "values": values, "exercised": exercised})


# ---------------------------------------------------------------------------------------- roots and grids


def implied_parameter(pricer, target, bracket, expand, method, tolerance):
    """Any pricer inverted for its input at a given price, by Brent's bracketing root or bisection.

    theta with price(theta) = target, for price monotone in theta on the bracket (low, high): Brent's method or
    bisection once the bracket's ends give prices either side of the target; `expand` extends each named end by
    the current width until they do. Where no value reaches the target (a price outside the model's range, a
    quote below intrinsic), found is False and value nan.

    Used for: an implied vol under any model (normal or Bachelier, CEV, SABR, Heston), an option-adjusted spread or
    a discount margin, a base correlation, an implied hazard rate or default probability, a breakeven vol, a yield.
    Held: implied_vol (Black-Scholes-Merton), bond_yield (a bond's yield to maturity).

    conventions:
    - pricer: a function of the one parameter giving the price (other inputs fixed inside it)
    - target: the quoted price to match
    - bracket: (low, high), where the root is looked for
    - expand: 'none', 'up' (the high end moved out), 'down' or 'both', each named end extended by the current width
    - method: 'brent' (Brent 1973) or 'bisection'
    - tolerance: the parameter's absolute error
    returns: a dict: value; found (True where the target was bracketed and solved); error (price at value less
      target); bracket (the one solved in); iterations
    source: Brent, Algorithms for Minimization without Derivatives (1973), ch. 4; Press et al., Numerical Recipes
      (2007), 9.3
    aliases: bracketing root, [Brent root]
    example:
        import qf
        # Result: value 0.19998445, found True.
        qf.implied_parameter(pricer=lambda v: qf.bs_price(option="call", spot=100.0, strike=100.0, maturity=1.0, rate=0.05, dividend_yield=0.0, vol=v), target=10.45, bracket=(0.01, 1.0), expand="up", method="brent", tolerance=1e-10)
    """
    choice("expand", expand, EXPANSIONS)
    choice("method", method, ROOTS)
    low, high = (float(end) for end in bracket)
    if not low < high:
        raise ValueError("bracket must be (low, high) with low < high")
    tol, target = float(positive("tolerance", tolerance)), float(target)

    def gap(level):
        return float(pricer(level)) - target

    at_low, at_high, passes = gap(low), gap(high), 0
    while expand != "none" and passes < EXPANSION_PASSES and not at_low * at_high <= 0:
        width = high - low
        if expand in ("up", "both"):
            high, at_high = high + width, gap(high + width)
        if expand in ("down", "both"):
            low, at_low = low - width, gap(low - width)
        passes += 1
    found, value, iterations = True, np.nan, 0
    if at_low == 0.0:
        value = low
    elif at_high == 0.0:
        value = high
    elif not at_low * at_high < 0:
        found = False
    else:
        solve = optimize.brentq if method == "brent" else optimize.bisect
        value, report = solve(gap, low, high, xtol=tol, maxiter=1_000, full_output=True)
        iterations = int(report.iterations)
    return Result({"value": float(value), "found": found, "error": gap(value) if found else np.nan,
                   "bracket": (low, high), "iterations": iterations})


def discretize_distribution(law, step, points, method):
    """Discretised severity: a scipy law's probabilities on a lattice, by rounding or local moment matching.

    On x_j = j h, j < n: 'rounding' f_0 = F(h/2), f_j = F((j + 1/2) h) - F((j - 1/2) h); 'lower' f_j = F((j + 1) h)
    - F(j h) (each span's mass at its lower end, F(0) added at 0); 'upper' f_0 = F(0), f_j = F(j h) - F((j - 1) h)
    (at its upper end); 'moment_matching' local matching of the mean span by span, f_0 = 1 - E[X ^ h]/h, f_j = (2
    E[X ^ j h] - E[X ^ (j - 1) h] - E[X ^ (j + 1) h])/h, E[X ^ u] the limited expected value (X >= 0 only), so the
    grid's mean is the law's but for the tail past the grid. 'lower' and 'upper' bracket the law (stochastically
    below and above); rounding's mean converges as h falls.

    Used for: severities for aggregate losses (panjer_recursion, or an FFT convolution of the grid), a loss or
    return density put on a grid, a lattice's terminal law. Held: aggregate_loss discretises its severity this way.

    conventions:
    - law: a scipy.stats frozen law (e.g. lognorm(s=sigma, scale=exp(mu)), gamma(a=, scale=), pareto(b=, scale=))
    - step: h, the grid's spacing; points: n, the grid's length
    - method: 'rounding', 'lower', 'upper' or 'moment_matching'
    returns: a dict: grid (j h); pmf (f_j); tail_mass (the law's mass past the grid)
    source: Klugman, Panjer and Willmot, Loss Models (2019), 9.6.5; Gerber (1982), Insurance: Mathematics and
      Economics 1(1)
    aliases: local moment matching
    example:
        import qf, scipy.stats as st
        # Result: pmf starts [0.00467884, 0.03225747, 0.05326770], tail_mass about 0.
        qf.discretize_distribution(law=st.gamma(a=2.0, scale=50.0), step=10.0, points=200, method="rounding")
    """
    choice("method", method, DISCRETISATIONS)
    if not hasattr(law, "cdf"):
        raise ValueError("law must be a scipy.stats frozen distribution (with .cdf)")
    h, n = float(positive("step", step)), integer("points", points, 2)
    grid = h * np.arange(n)
    if method == "rounding":
        cuts = law.cdf(h * (np.arange(n) + 0.5))
        f = np.diff(np.r_[0.0, cuts])
    elif method == "lower":
        f = np.diff(law.cdf(h * np.arange(n + 1)))
        f[0] += law.cdf(0.0)
        cuts = np.cumsum(f)
    elif method == "upper":
        f = np.diff(np.r_[0.0, law.cdf(grid)])
        cuts = np.cumsum(f)
    else:
        if law.cdf(0.0) > 0.0 and float(law.support()[0]) < 0.0:
            raise ValueError("'moment_matching' needs a law on X >= 0")
        t, w = np.polynomial.legendre.leggauss(SPAN_NODES)
        nodes = h * (np.arange(n)[:, None] + 0.5 * (t[None, :] + 1.0))
        spans = 0.5 * h * (law.sf(nodes) @ w)                       # int over each span of S(x) dx
        probabilities = [1e-12, 1e-9, 1e-6, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                         0.9, 0.95, 0.99, 0.999, 1 - 1e-4, 1 - 1e-6, 1 - 1e-9, 1 - 1e-12]
        breaks = np.unique(np.r_[law.support(), law.ppf(probabilities)])
        breaks = breaks[np.isfinite(breaks)]
        for j in range(n):
            start, end = j * h, (j + 1) * h
            inner = breaks[(breaks > start) & (breaks < end)]
            if inner.size:
                cuts = np.r_[start, inner, end]
                low, high = cuts[:-1, None], cuts[1:, None]
                xs = 0.5 * (high - low) * t[None, :] + 0.5 * (high + low)
                spans[j] = 0.5 * (high - low)[:, 0] @ (law.sf(xs) @ w)
        limited = np.r_[0.0, np.cumsum(spans)]                      # E[X ^ j h], j = 0 .. n
        f = np.empty(n)
        f[0] = 1.0 - limited[1] / h
        f[1:] = (2.0 * limited[1:n] - limited[:n - 1] - limited[2:]) / h
        cuts = np.cumsum(f)
    return Result({"grid": grid, "pmf": f, "tail_mass": float(max(1.0 - cuts[-1], 0.0))})


# ------------------------------------------------------------------------------------ random streams and checks


class _Legacy:
    """numpy's legacy stream (a numpy.random.RandomState) drawn as a Generator is drawn: the methods both have pass
    through to it, and a Generator's `integers` is its `randint` (both exclusive of `high`)."""

    __slots__ = ("legacy",)

    def __init__(self, legacy):
        self.legacy = legacy

    def integers(self, low, high=None, size=None):
        return self.legacy.randint(low, high, size)

    def __getattr__(self, name):
        return getattr(self.legacy, name)


def _stream(rng):
    """The stream a sampler draws from: a numpy.random.Generator as it is; an int seed (0 or more) as
    numpy.random.default_rng(seed); a numpy.random.RandomState, numpy's legacy stream, as it is (`_Legacy`, which
    draws from it and advances it); None as numpy's global legacy stream, the one numpy.random.seed(seed) sets. The
    two kinds of stream give different draws from the same seed."""
    if isinstance(rng, (np.random.Generator, _Legacy)):
        return rng
    if isinstance(rng, np.random.RandomState):
        return _Legacy(rng)
    if rng is None:
        return _Legacy(np.random.mtrand._rand)
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_)) and rng >= 0:
        return np.random.default_rng(int(rng))
    raise ValueError("rng must be a numpy.random.Generator or an int seed (numpy.random.default_rng(seed)), or a "
                     "numpy.random.RandomState or None (numpy's legacy stream, as numpy.random.seed(seed) sets it)")


# Shared wording for the guards and the convention entries added with them.
WORDS = {
    "rate": "a decimal annual rate",
    "vol": "a decimal annual volatility",
    "time": "in years",
    "price": "a positive price",
    "level": "a probability level strictly between zero and one",
    "curve": "annual rates, -ln(D)/t, not rates times their times",
}
RATE_LIMIT = 1.0                # a decimal annual rate above this (100% a year) is read as a percent: a percent slip
#                                 is positive; negative rates, including a formula's adjusted rates, are allowed
VOL_LIMIT = 10.0                # a decimal annual volatility past this (1,000% a year) is read as a percent
YEARS_LIMIT = 100.0             # a time in years past this (a century bond's life) is read as days
CURVE_POINTS = 3                # a curve's rates are read as -ln D(t) on at least this many maturities,
CURVE_SPAN = 8.0                # the longest at least this many times the shortest,
CURVE_SPREAD = 1.5              # each rate over its maturity within this ratio of the others (in proportion to it),
CURVE_FLOOR = 0.1               # and the longest's rate at least this: rates in proportion to their maturities
#                                 may also be a steep annual-rate curve: shape alone cannot distinguish the units
GUARD_DEPTH = 6                 # the frames above a call searched for a qf or scipy caller (`_probed`)


def _numbers(value):
    """`value` as a float array, or None where it is not numbers (a callable, a string, a ragged list): the
    function's own checks say what is wrong with those."""
    if value is None or callable(value) or isinstance(value, (str, bytes, dict)):
        return None
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _refuse(function, name, kind, said):
    head = f"{name}{said}" if said.startswith("=") else f"{name} {said}"
    raise ValueError(f"{head}: qf.{function} takes {name} {'' if kind == 'time' else 'as '}{WORDS[kind]}")


def _check_rate(function, name, array):
    wrong = array > RATE_LIMIT
    if np.any(wrong):
        _refuse(function, name, "rate", "looks like a percent")


def _check_vol(function, name, array):
    if np.any(array < 0):
        _refuse(function, name, "vol", "must not be negative")
    if np.any(array > VOL_LIMIT):
        _refuse(function, name, "vol", "looks like a percent")


def _check_time(function, name, array):
    if np.any(array < 0):
        _refuse(function, name, "time", "must not be negative")
    days = np.isfinite(array) & (array > YEARS_LIMIT)
    if np.any(days):
        _refuse(function, name, "time", "looks like days")


def _check_price(function, name, array):
    if np.any(array <= 0):
        _refuse(function, name, "price", "must be positive")


def _check_level(function, name, array):
    wrong = ~((array > 0) & (array < 1)) & ~np.isnan(array)
    if np.any(wrong):
        percent = wrong & (array > 1) & (array < 100)
        _refuse(function, name, "level", "looks like a percent" if np.any(percent) else
                "must lie strictly between zero and one")


def _check_curve(function, pairs):
    """Rates in proportion to their maturities, possibly -ln D(t) (`CURVE_POINTS`, `CURVE_SPAN`, `CURVE_SPREAD`,
    `CURVE_FLOOR`): each pair's (time, rate) points pooled, the times above zero; decimals only (a curve in percent
    is the function's own business)."""
    times, rates = [], []
    for (_, t), (_, z) in pairs:
        if t is None or z is None:
            return
        try:
            t, z = np.broadcast_arrays(t, z)
        except ValueError:
            return
        times.append(t.ravel())
        rates.append(z.ravel())
    t, z = np.concatenate(times), np.concatenate(rates)
    keep = np.isfinite(t) & np.isfinite(z) & (t > 0)
    t, z = t[keep], z[keep]
    if t.size < CURVE_POINTS or np.any(z <= 0) or np.any(z > RATE_LIMIT) or t.max() < CURVE_SPAN * t.min():
        return
    ratio = z / t
    if (ratio.max() <= CURVE_SPREAD * ratio.min() and z[np.argmax(t)] >= CURVE_FLOOR
            and np.unique(t).size >= CURVE_POINTS):
        names = " and ".join(dict.fromkeys(z_name for _, (z_name, _) in pairs))
        times = " and ".join(dict.fromkeys(t_name for (t_name, _), _ in pairs))
        warnings.warn(f"{names} grow in proportion to {times} and may be -ln D(t) (a rate times its time): "
                      f"qf.{function} takes them as {WORDS['curve']}", RuntimeWarning, stacklevel=3)


_CHECKS = {"rate": _check_rate, "vol": _check_vol, "time": _check_time, "price": _check_price,
           "level": _check_level}
_SCALAR_HOLDS = {"rate": lambda x: x <= RATE_LIMIT, "vol": lambda x: 0.0 <= x <= VOL_LIMIT,
                 "time": lambda x: 0.0 <= x <= YEARS_LIMIT, "price": lambda x: x > 0.0,
                 "level": lambda x: 0.0 < x < 1.0}    # each kind's check on a float, True where it would pass


def _probed(frame):
    """Whether a call comes from inside the library or a scipy routine (`GUARD_DEPTH` frames up): the values another
    qf function, a root search or a quadrature passes are trial points, not the caller's inputs."""
    for _ in range(GUARD_DEPTH):
        if frame is None:
            return False
        if frame.f_globals.get("__name__", "").startswith(("qf.", "scipy.")):
            return True
        frame = frame.f_back
    return False


def _guard(curve=(), **kinds):
    """A function's arguments checked by their kind (`WORDS`) after it returns, so a refusal of its own speaks first
    and the guard speaks where it would have returned figures: each refusal names the function's convention and the
    likely slip, under the guard's policy (a rate above `RATE_LIMIT`; a volatility
    negative or past `VOL_LIMIT`; a time negative or past `YEARS_LIMIT`; a price not positive; a level outside (0,
    1)). A curve's rates in proportion to their maturities get a convention warning (`_check_curve`), since a
    steep annual-rate curve can have the same shape. None, a callable and what is not numbers
    pass, and so does NaN; a call from inside qf or scipy is not checked (`_probed`)."""
    def wrap(function):
        names = list(inspect.signature(function).parameters)
        checks = [(names.index(name), name, kind, _CHECKS[kind]) for kind, given in kinds.items() for name in given]
        pairs = [((names.index(t), t), (names.index(z), z)) for t, z in curve]

        def value(args, kwargs, at, name):
            return args[at] if at < len(args) else kwargs.get(name)

        @functools.wraps(function)
        def guarded(*args, **kwargs):
            found = function(*args, **kwargs)
            if _probed(sys._getframe(1)):
                return found
            for at, name, kind, check in checks:
                given = value(args, kwargs, at, name)
                if type(given) is float and _SCALAR_HOLDS[kind](given):
                    continue                            # the common case, a plain number its convention holds
                array = _numbers(given)
                if array is not None and array.size:
                    check(function.__name__, name, array)
            if pairs:
                _check_curve(function.__name__, [((t, _numbers(value(args, kwargs, at_t, t))),
                                                  (z, _numbers(value(args, kwargs, at_z, z))))
                                                 for (at_t, t), (at_z, z) in pairs])
            return found
        return guarded
    return wrap
