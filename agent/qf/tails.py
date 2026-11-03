"""Extreme value theory: Hill estimator, GPD peaks-over-threshold VaR and ES, GEV block maxima, return levels.

`losses` are losses-positive (pass -returns for a return series' left tail). A VaR or ES takes `alpha`, the
confidence level (0.99), and `sign`: 'loss_positive' reports a loss of 0.02 as 0.02, 'return_negative' as -0.02. A
function with several results returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize
from scipy.special import gamma as gamma_function

from ._common import Result, choice, integer
from .conventions import SIGNS, _finite, _tail
from .primitives import _guard

__all__ = ["gev_fit", "gpd_errors", "gpd_fit", "gpd_var_es", "hill_estimator"]

GEV_METHODS = ("mle", "pwm")
GEV_SHAPES = ("free", "gumbel")
EULER_GAMMA = 0.5772156649015329
SMALL_XI = 1e-9                 # a shape below this in size is the exponential (GPD) or Gumbel (GEV) limit
INFORMATION = ("expected", "observed")
SERIES_XI = 1e-4                # below this |xi| the observed information's 1/xi terms go by their series


def hill_estimator(losses, *, k):
    """Hill's tail-index estimator from the k largest losses, with its standard error.

    With X_(1) >= X_(2) >= ... the losses in descending order: xi = (1/k) sum_{i=1}^k ln X_(i) - ln X_(k+1), the
    tail index alpha = 1/xi (P(X > x) ~ x^-alpha), standard errors xi/sqrt(k) and alpha/sqrt(k). The threshold is the
    (k+1)-th largest loss, which must be positive. Losses positive: pass -returns for a return series' left tail.

    Weissman's estimate of a quantile beyond the data from it: x_p = X_(k+1) (k / (n p))^xi for a tail probability
    p = 1 - alpha below k/n, n the sample size. The mean-excess function e(u) = sum (X_i - u) 1{X_i > u} / #{X_i > u},
    the mean excess over u, is linear in u for a GPD tail with xi < 1 (slope xi/(1 - xi), rising when xi > 0): its plot picks k
    here and the threshold of qf.gpd_fit.
    conventions:
        k: the number of upper order statistics used, 1 <= k < n; the threshold is X_(k+1), the (k+1)-th largest
            loss, and xi = mean(ln X_(i)) - ln X_(k+1) over the k largest losses (positive: pass -returns)
    returns: a dict: xi, alpha, se_xi, se_alpha, threshold (X_(k+1)), k
    source: Hill (1975), "A simple general approach to inference about the tail of a distribution", Annals of
        Statistics 3(5); Embrechts, Kluppelberg and Mikosch, Modelling Extremal Events (1997), 6.4.2 (Hill, and
        Weissman's quantile); McNeil, Frey and Embrechts (2005), 7.2.2 (the mean-excess plot)
    aliases: Hill estimator, Hill tail index, tail index, Hill plot, Weissman estimator
    example:
        import numpy as np, qf
        losses = np.random.default_rng(7).pareto(3.0, 2000) + 1.0
        qf.hill_estimator(losses, k=200)
    """
    x = np.sort(_finite("losses", losses).ravel())[::-1]
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k < x.size:
        raise ValueError(f"k must be an int with 1 <= k < {x.size} (the sample size)")
    k = int(k)
    if not x[k] > 0:
        raise ValueError("the threshold X_(k+1) must be positive: take k below the count of positive losses")
    xi = float(np.mean(np.log(x[:k])) - math.log(x[k]))
    tail_index = math.inf if xi == 0 else 1.0 / xi
    return Result({"xi": xi, "alpha": tail_index, "se_xi": xi / math.sqrt(k), "se_alpha": tail_index / math.sqrt(k),
                   "threshold": float(x[k]), "k": k})


def _gpd_nll(params, y):
    xi, log_beta = params
    beta = math.exp(log_beta)
    z = xi * y / beta
    if np.any(z <= -1.0):
        return math.inf
    if abs(xi) < 1e-10:
        return y.size * log_beta + float(np.sum(y)) / beta
    return y.size * log_beta + (1.0 + 1.0 / xi) * float(np.sum(np.log1p(z)))


def _gpd_score(params, y):
    xi, beta = params
    z = 1.0 + xi * y / beta
    d_xi = np.sum(np.log(z)) / xi ** 2 - (1.0 + 1.0 / xi) * np.sum(y / z) / beta
    d_beta = -y.size / beta + (1.0 + xi) * np.sum(y / z) / beta ** 2
    return np.array([d_xi, d_beta])


def gpd_fit(losses, *, threshold):
    """Smith's GPD maximum-likelihood fit of the losses above a threshold, with analytic errors.

    The excesses y = x - u of the losses x > u (strictly above) are fitted to G(y) = 1 - (1 + xi y / beta)^(-1/xi)
    by maximum likelihood, ln L = -n ln beta - (1 + 1/xi) sum ln(1 + xi y / beta) (the exponential at xi = 0). For xi
    > -1/2 the inverse Fisher information gives Var(xi) = (1 + xi)^2 / n, Var(beta) = 2 beta^2 (1 + xi) / n and
    Cov = -beta (1 + xi) / n (NaN standard errors at or below -1/2): the expected information's errors; the observed
    Fisher information's (the likelihood's Hessian) are qf.gpd_errors. Losses positive. The tail's VaR and ES from
    the fit are qf.gpd_var_es.
    conventions:
        losses: the whole sample of losses (positive), not the excesses: x - u is taken here
        threshold: u; only losses strictly above it are excesses
    returns: a dict: xi (the shape); beta (the scale, in the losses' units); se_xi, se_beta and cov_xi_beta;
        exceedances (the count above u); threshold (u); loglik
    source: Smith (1987), "Estimating tails of probability distributions", Annals of Statistics 15(3); Hosking and
        Wallis (1987), Technometrics 29(3); McNeil, Frey and Embrechts, Quantitative Risk Management (2005), 7.2
    aliases: GPD, generalized Pareto, generalised Pareto distribution, peaks over threshold, POT, Smith GPD
    example:
        import numpy as np, qf
        losses = np.random.default_rng(7).standard_t(4, 5000)
        qf.gpd_fit(losses, threshold=2.0)
    """
    x = _finite("losses", losses).ravel()
    u = float(threshold)
    if not math.isfinite(u):
        raise ValueError("threshold must be a finite number")
    y = x[x > u] - u
    n = y.size
    if n < 3:
        raise ValueError(f"only {n} losses lie above the threshold; the fit needs at least 3")
    mean, var = float(np.mean(y)), float(np.var(y, ddof=1))
    xi0 = min(max(0.5 * (1.0 - mean * mean / var), -0.4), 0.9) if var > 0 else 0.0
    beta0 = mean * (1.0 - xi0)
    found = optimize.minimize(_gpd_nll, [xi0, math.log(beta0)], args=(y,), method="Nelder-Mead",
                              options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 20000, "maxfev": 40000})
    xi, beta = float(found.x[0]), math.exp(found.x[1])
    if abs(xi) > 1e-6:
        # Newton-type polish of the score: Nelder-Mead stops near 1e-5 in the score, the root at rounding
        polished = optimize.root(_gpd_score, [xi, beta], args=(y,), method="hybr", options={"xtol": 1e-14})
        x_new, beta_new = float(polished.x[0]), float(polished.x[1])
        if beta_new > 0 and abs(x_new) > 1e-10 and math.isfinite(x_new) and \
                np.linalg.norm(_gpd_score([x_new, beta_new], y)) < np.linalg.norm(_gpd_score([xi, beta], y)) and \
                _gpd_nll([x_new, math.log(beta_new)], y) <= found.fun + 1e-9 * max(1.0, abs(found.fun)):
            xi, beta = x_new, beta_new
    regular = xi > -0.5
    return Result({"xi": xi, "beta": beta, "se_xi": (1.0 + xi) / math.sqrt(n) if regular else math.nan,
                   "se_beta": beta * math.sqrt(2.0 * (1.0 + xi) / n) if regular else math.nan,
                   "cov_xi_beta": -beta * (1.0 + xi) / n if regular else math.nan, "exceedances": n, "threshold": u,
                   "loglik": -_gpd_nll([xi, math.log(beta)], y)})

def _gpd_information(y, xi, beta):
    """Minus the GPD log-likelihood's Hessian in (xi, beta) at the excesses y: the observed information. The two
    terms with 1/xi go by their series below SERIES_XI, where the closed forms cancel."""
    a = y / beta
    z = 1.0 + xi * a
    n = y.size
    l_bb = n / beta ** 2 - (1.0 + xi) / beta ** 2 * (np.sum(a / z) + np.sum(a / z ** 2))
    if abs(xi) >= SERIES_XI:
        l_xx = -2.0 / xi ** 3 * np.sum(np.log1p(xi * a)) + 2.0 / xi ** 2 * np.sum(a / z) + \
            (1.0 + 1.0 / xi) * np.sum(a * a / z ** 2)
        l_xb = (-np.sum(a / z) + (1.0 + xi) * np.sum(a / z ** 2)) / (xi * beta)
    else:
        s1, s2, s3, s4 = (np.sum(a ** k) for k in (1, 2, 3, 4))
        l_xx = s2 - 2.0 * s3 / 3.0 + xi * (1.5 * s4 - 2.0 * s3)
        l_xb = (s1 - s2 + 2.0 * xi * (s3 - s2)) / beta
    return -np.array([[l_xx, l_xb], [l_xb, l_bb]])


def gpd_errors(losses, threshold, xi, beta, information):
    """GPD standard errors at a fit, by the expected (Smith's analytic) or the observed Fisher information.

    'expected': Smith's (1987) inverse Fisher information over the n excesses, Var(xi) = (1 + xi)^2 / n, Var(beta) =
    2 beta^2 (1 + xi) / n, Cov = -beta (1 + xi) / n, for xi > -1/2 (NaN at or below): gpd_fit's own se_xi and
    se_beta. 'observed': the inverse of minus the log-likelihood's Hessian in (xi, beta) at the values given, from
    the excesses themselves, in closed form (what a task's "observed Fisher information", or a numerical Hessian of
    its likelihood at the MLE, gives); the two agree as n grows. A variance the inverse leaves negative (not a
    maximum) is NaN.

    conventions:
    - information: 'expected' (Smith's analytic, gpd_fit's) or 'observed' (the Hessian's, at these xi and beta)
    - losses: the whole sample (positive losses), not the excesses; threshold: u, the excesses those strictly above
    - xi, beta: the shape and scale (gpd_fit's), every excess inside the support (1 + xi y / beta > 0)
    returns: a dict: se_xi; se_beta; cov_xi_beta; covariance (2 x 2, xi then beta); exceedances (n)
    source: Smith (1987), "Estimating tails of probability distributions", Annals of Statistics 15(3); Efron and
      Hinkley (1978), Biometrika 65(3) (observed against expected information)
    aliases: GPD standard errors, observed Fisher information, expected Fisher information, GPD observed information, [Hessian standard errors]
    example:
        # Returns se_xi 0.3981 (the observed information's; 0.4899 the expected's) for xi 0.2, beta 1.0, six excesses.
        qf.gpd_errors(losses=[0.5, 1.2, 1.8, 2.6, 3.1, 4.4, 6.0], threshold=1.0, xi=0.2, beta=1.0, information="observed")
    """
    choice("information", information, INFORMATION)
    x = _finite("losses", losses).ravel()
    u, shape, scale = float(threshold), float(xi), float(beta)
    if not (math.isfinite(u) and math.isfinite(shape) and math.isfinite(scale) and scale > 0):
        raise ValueError("threshold and xi must be finite numbers and beta a positive one")
    y = x[x > u] - u
    n = y.size
    if n < 2:
        raise ValueError(f"only {n} losses lie above the threshold; the errors need at least 2")
    if np.any(1.0 + shape * y / scale <= 0):
        raise ValueError("an excess lies outside the GPD's support at these xi and beta (1 + xi y / beta <= 0)")
    if information == "expected":
        if shape <= -0.5:
            cov = np.full((2, 2), np.nan)
        else:
            cov = np.array([[(1.0 + shape) ** 2, -scale * (1.0 + shape)],
                            [-scale * (1.0 + shape), 2.0 * scale ** 2 * (1.0 + shape)]]) / n
    else:
        cov = np.linalg.inv(_gpd_information(y, shape, scale))
    variances = np.diag(cov)
    se = np.where(variances > 0, np.sqrt(np.abs(variances)), np.nan)
    return Result({"se_xi": float(se[0]), "se_beta": float(se[1]), "cov_xi_beta": float(cov[0, 1]),
                   "covariance": cov, "exceedances": n})


@_guard(level=('alpha',))
def gpd_var_es(*, threshold, xi, beta, n, exceedances, alpha, sign):
    """VaR and ES by peaks over threshold: the tail beyond u fitted as a GPD (McNeil-Frey's estimator).

    With N_u of the n losses above the threshold u, and the GPD's shape xi and scale beta fitted to their excesses
    (qf.gpd_fit), the tail P(X > x) = (N_u / n)(1 + xi (x - u) / beta)^(-1/xi) gives
    VaR = u + (beta / xi) [((n / N_u)(1 - alpha))^(-xi) - 1] and ES = VaR / (1 - xi) + (beta - xi u) / (1 - xi); at
    xi = 0 (the exponential tail) VaR = u - beta ln((n / N_u)(1 - alpha)) and ES = VaR + beta. The ES needs xi < 1
    (infinite beyond); the level must lie beyond the threshold, 1 - alpha below N_u / n.
    conventions:
        threshold: u, in the losses' units (losses positive)
        xi: the GPD shape (0 the exponential tail); beta: the GPD scale, above 0
        n: the sample size; exceedances: N_u, the losses strictly above u (gpd_fit's exceedances)
        alpha: the confidence level, e.g. 0.99 (the tail is 1 - alpha)
        sign: 'loss_positive' (VaR and ES of a loss as positive numbers) or 'return_negative' (their negatives)
    returns: (var, es)
    source: McNeil and Frey (2000), Journal of Empirical Finance 7; McNeil, Frey and Embrechts, Quantitative Risk
        Management (2005), 7.2.3, equations (7.18) and (7.19)
    aliases: GPD VaR, POT VaR, peaks over threshold VaR, GPD ES, EVT VaR, extreme value VaR
    example:
        import qf
        qf.gpd_var_es(threshold=2.0, xi=0.25, beta=0.6, n=1000, exceedances=50, alpha=0.99, sign="loss_positive")
        # (VaR, ES) = (3.18883707, 4.38511610)
    """
    choice("sign", sign, SIGNS)
    u, shape, scale = float(threshold), float(xi), float(beta)
    if not (math.isfinite(u) and math.isfinite(shape) and scale > 0):
        raise ValueError("threshold and xi must be finite numbers and beta positive")
    size, above = integer("n", n, 1), integer("exceedances", exceedances, 1)
    if above > size:
        raise ValueError("exceedances cannot exceed n")
    tail = _tail(alpha)
    if not tail < above / size:
        raise ValueError(f"1 - alpha = {tail:g} is not beyond the threshold: it must lie below N_u / n = "
                         f"{above / size:g}")
    log_t = math.log(size / above * tail)                       # negative: the level lies beyond u
    if abs(shape) < SMALL_XI:
        var, es = u - scale * log_t, u - scale * log_t + scale
    else:
        var = u + scale * math.expm1(-shape * log_t) / shape
        es = math.inf if shape >= 1.0 else var / (1.0 - shape) + (scale - shape * u) / (1.0 - shape)
    return (var, es) if sign == "loss_positive" else (-var, -es)


def _block_maxima(x, block_size):
    if block_size is None:
        return x
    size = integer("block_size", block_size, 2)
    blocks = x.size // size
    if blocks < 3:
        raise ValueError(f"{x.size} losses make {blocks} whole blocks of {size}; the fit needs at least 3")
    return x[:blocks * size].reshape(blocks, size).max(axis=1)


def _gev_nll(mu, sigma, xi, m):
    """The GEV's negative log-likelihood of the maxima m; inf outside the support."""
    if not sigma > 0:
        return math.inf
    z = (m - mu) / sigma
    if abs(xi) < SMALL_XI:
        return float(m.size * math.log(sigma) + np.sum(z) + np.sum(np.exp(-z)))
    t = 1.0 + xi * z
    if np.any(t <= 0):
        return math.inf
    return float(m.size * math.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(t)) + np.sum(t ** (-1.0 / xi)))


def _gev_pwm(m, gumbel):
    """Hosking, Wallis and Wood's probability-weighted moments: (mu, sigma, xi)."""
    s, n = np.sort(m), m.size
    j = np.arange(n, dtype=float)
    b0 = float(s.mean())
    b1 = float(np.sum(j / (n - 1) * s) / n)
    b2 = float(np.sum(j * (j - 1) / ((n - 1) * (n - 2)) * s) / n)
    if gumbel:
        sigma = (2.0 * b1 - b0) / math.log(2.0)
        return b0 - EULER_GAMMA * sigma, sigma, 0.0
    c = (2.0 * b1 - b0) / (3.0 * b2 - b0) - math.log(2.0) / math.log(3.0)
    k = 7.8590 * c + 2.9554 * c * c                             # Hosking's k = -xi
    if abs(k) < SMALL_XI:
        sigma = (2.0 * b1 - b0) / math.log(2.0)
        return b0 - EULER_GAMMA * sigma, sigma, 0.0
    g = gamma_function(1.0 + k)
    sigma = (2.0 * b1 - b0) * k / (g * (1.0 - 2.0 ** -k))
    return b0 + sigma * (g - 1.0) / k, sigma, -k


def gev_fit(losses, *, block_size, method, shape, return_period):
    """GEV fit of block maxima (generalised extreme value) by likelihood or PWM; the return level of a k-block period.

    The maxima of consecutive blocks of `block_size` losses (an incomplete last block dropped) are fitted to
    H(x) = exp(-(1 + xi (x - mu) / sigma)^(-1/xi)), the Gumbel exp(-exp(-(x - mu) / sigma)) at xi = 0: xi > 0 a
    heavy (Frechet) tail, xi < 0 a bounded (Weibull) one. scipy.stats.genextreme writes the shape the other way,
    c = -xi: genextreme(c=-xi, loc=mu, scale=sigma). The k-block return level, the level one block's maximum exceeds
    once in k blocks on average, is H^-1(1 - 1/k) = mu + (sigma / xi)[(-ln(1 - 1/k))^(-xi) - 1] (mu - sigma
    ln(-ln(1 - 1/k)) at xi = 0).
    conventions:
        losses: the losses, oldest first (positive: pass -returns), or the block maxima themselves
        block_size: the losses a block holds (21 trading days for monthly maxima), or None where losses are maxima
        method: 'mle' (maximum likelihood) or 'pwm' (Hosking, Wallis and Wood's probability-weighted moments)
        shape: 'free' (xi fitted) or 'gumbel' (xi = 0 held)
        return_period: k, in blocks (10 for the ten-block level), or None for no return level
    returns: a dict: xi, mu, sigma; return_level (None without a return_period); maxima (the fitted values); blocks;
        loglik
    source: McNeil, Frey and Embrechts, Quantitative Risk Management (2005), 7.1.3-7.1.4 (block maxima, return
        levels); Hosking, Wallis and Wood (1985), Technometrics 27(3) (PWM); Embrechts, Kluppelberg and Mikosch (1997)
    aliases: GEV, generalized extreme value, block maxima, return level, Gumbel fit, Frechet, Fisher-Tippett
    example:
        import qf
        maxima = [2.1, 3.4, 2.7, 5.0, 2.9, 3.8, 2.4, 4.1, 3.1, 2.6]
        qf.gev_fit(losses=maxima, block_size=None, method="mle", shape="free", return_period=10)
        # xi = 0.08334057, mu = 2.79694963, sigma = 0.62055826; return_level = 4.33297210
    """
    choice("method", method, GEV_METHODS)
    choice("shape", shape, GEV_SHAPES)
    m = _block_maxima(_finite("losses", losses).ravel(), block_size)
    if m.size < 3 or np.ptp(m) == 0:
        raise ValueError("the fit needs at least 3 maxima that are not all equal")
    gumbel = shape == "gumbel"
    mu, sigma, xi = _gev_pwm(m, gumbel)
    if method == "mle":
        if gumbel:
            cost = lambda p: _gev_nll(p[0], math.exp(p[1]), 0.0, m)  # noqa: E731
            start = [mu, math.log(sigma)]
        else:
            cost = lambda p: _gev_nll(p[0], math.exp(p[1]), p[2], m)  # noqa: E731
            xi = min(max(xi, -0.45), 0.45)
            start = [mu, math.log(sigma), xi]
            if not math.isfinite(cost(start)):
                start = [mu, math.log(sigma), 0.0]
        options = {"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000, "maxfev": 80000}
        found = optimize.minimize(cost, start, method="Nelder-Mead", options=options)
        found = optimize.minimize(cost, found.x, method="Nelder-Mead", options=options)    # a restart polishes
        mu, sigma, xi = float(found.x[0]), math.exp(found.x[1]), 0.0 if gumbel else float(found.x[2])
    level = None
    if return_period is not None:
        k = float(return_period)
        if not k > 1:
            raise ValueError("return_period must exceed 1 block, or be None")
        y = -math.log1p(-1.0 / k)
        level = mu - sigma * math.log(y) if abs(xi) < SMALL_XI else mu + sigma * math.expm1(-xi * math.log(y)) / xi
    return Result({"xi": xi, "mu": mu, "sigma": sigma, "return_level": level, "maxima": m, "blocks": int(m.size),
                   "loglik": -_gev_nll(mu, sigma, xi, m)})
