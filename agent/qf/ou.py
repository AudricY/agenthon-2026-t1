"""Mean reversion: AR(1) and Ornstein-Uhlenbeck fits, half-life, OU moments with jumps.

The Ornstein-Uhlenbeck process with compound-Poisson jumps: its conditional and stationary moments; and the
continuous-time parameters from an AR(1) fitted to equally spaced observations, by the exact discretisation or Euler's
as each function names.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ._common import Result, choice, out, positive
from .primitives import _guard
from .volatility import LN2, _int, _series

__all__ = ["ar1_fit", "cir_log_likelihood", "ou_fit", "ou_from_ar1", "ou_jump_moments"]

OU_SCHEMES = ("exact", "euler")


def ou_jump_moments(x0, kappa, theta, sigma, jump_rate, jump_mean, jump_std, horizon):
    """Mean and variance of X_{t+h} given X_t = x0 for dX = kappa (theta - X) dt + sigma dW + J dN,
    N a Poisson process of intensity `jump_rate` lambda and J ~ i.i.d. with mean `jump_mean` mu_J and standard
    deviation `jump_std` s_J (any law with these two moments; normal in the usual model), independent of W:

    E = x0 e^{-kappa h} + (theta + lambda mu_J / kappa)(1 - e^{-kappa h}),
    Var = (sigma^2 + lambda (mu_J^2 + s_J^2)) (1 - e^{-2 kappa h}) / (2 kappa).
    Each jump decays at the same rate kappa as the diffusion: the jumps' contributions are damped, never linear in h.
    `horizon` h = numpy.inf gives the stationary moments, theta + lambda mu_J / kappa and (sigma^2 + lambda (mu_J^2 +
    s_J^2)) / (2 kappa); lambda = 0 the pure OU's. X is whatever the model makes mean-revert: the level itself, or its
    logarithm in the geometric mean-reverting model (S = e^X), where E[S] and Var[S] under a lognormal approximation are
    e^{E + Var/2} and (e^{Var} - 1) e^{2E + Var}. Rates per unit of time (per year with annual parameters).
    returns: (mean, variance) of X_{t+h} given X_t = x0, both with the jumps: mean x0 e^{-kappa h} + (theta +
        lambda mu_J / kappa)(1 - e^{-kappa h}), variance (sigma^2 + lambda (mu_J^2 + s_J^2))(1 - e^{-2 kappa h}) /
        (2 kappa). Jumps revert with the state at kappa: never lambda mu_J h or lambda (mu_J^2 + s_J^2) h. With jumps
        the stationary variance is (sigma^2 + lambda (mu_J^2 + s_J^2)) / (2 kappa), above the pure OU's sigma^2 /
        (2 kappa); no conditional variance exceeds it.
    Source: Campbell's theorem on the shot-noise integral of e^{-kappa (h - s)} dJ_s; Cont and Tankov, Financial
    Modelling with Jump Processes (2004), 15.1 (OU processes driven by Levy processes); Uhlenbeck and Ornstein (1930).
    aliases: mean-reverting jump diffusion, OU with jumps, jump-diffusion moments, geometric mean-reverting jump diffusion, conditional mean and variance under jumps, OU jump moments, OU-jump, OU process with jumps, Ornstein-Uhlenbeck process with Poisson jumps
    example:
        qf.ou_jump_moments(x0=0.5, kappa=2.0, theta=0.2, sigma=0.3, jump_rate=1.5, jump_mean=-0.05,
                           jump_std=0.1, horizon=0.5)
    """
    kappa = positive("kappa", kappa)
    h = np.asarray(horizon, dtype=float)
    lam, mu_j, s_j = (np.asarray(a, dtype=float) for a in (jump_rate, jump_mean, jump_std))
    if np.any(h < 0) or np.any(lam < 0) or np.any(s_j < 0):
        raise ValueError("horizon, jump_rate and jump_std must not be negative")
    decay = np.exp(-kappa * h)
    mean = np.asarray(x0, dtype=float) * decay + (np.asarray(theta, dtype=float) + lam * mu_j / kappa) * (1.0 - decay)
    rate = np.asarray(sigma, dtype=float) ** 2 + lam * (mu_j * mu_j + s_j * s_j)
    var = rate * (1.0 - decay * decay) / (2.0 * kappa)
    return out(mean), out(var)


def ou_from_ar1(intercept, slope, resid_var, dt):
    """(kappa, theta, sigma) of dX = kappa (theta - X) dt + sigma dW from X_{t+dt} = a + b X_t + e fitted by OLS,
    inverting the OU's exact discretisation b = e^{-kappa dt}, a = theta (1 - b), Var(e) = sigma^2 (1 - b^2)/(2 kappa):

    kappa = -ln(b)/dt,  theta = a/(1 - b),  sigma = sqrt(2 kappa Var(e) / (1 - b^2)),  for 0 < b < 1.
    (The Euler scheme's kappa = (1 - b)/dt and sigma = sqrt(Var(e)/dt) are its small-dt limits, another estimator: a
    task stating them wants them.) `dt` in years for annual parameters (1/252 for
    trading days); `resid_var` the residual variance as the caller estimated it (its ddof is the caller's).
    returns: (kappa, theta, sigma): kappa per unit of dt's time, theta in X's units, sigma per square root of dt's
        unit. No jumps: an AR(1) on a path with jumps puts their variance in sigma (qf.ou_jump_moments has them).
    Source: the exact transition law of the OU process; Brigo and Mercurio, Interest Rate Models (2006), 3.2.1 (the
    Vasicek model's exact discretisation); Gillespie (1996), Physical Review E 54.
    aliases: Ornstein-Uhlenbeck from AR(1), OU calibration, AR(1) to OU, [mean reversion speed]
    example:
        qf.ou_from_ar1(intercept=0.02, slope=0.95, resid_var=0.0004, dt=1 / 252)
    """
    b = np.asarray(slope, dtype=float)
    if np.any(~((b > 0) & (b < 1))):
        raise ValueError("slope must lie strictly inside (0, 1) for a mean-reverting fit")
    dt = positive("dt", dt)
    kappa = -np.log(b) / dt
    theta = np.asarray(intercept, dtype=float) / (1.0 - b)
    sigma = np.sqrt(2.0 * kappa * np.asarray(resid_var, dtype=float) / (1.0 - b * b))
    return out(kappa), out(theta), out(sigma)


# ------------------------------------------------------------------------------------------------ mean reversion


def ar1_fit(x, *, ddof):
    """AR(1) fit and half-life of mean reversion: OLS of x_{t+1} on x_t, the residual ddof named.

    b = sum (x_t - xbar)(x_{t+1} - ybar) / sum (x_t - xbar)^2 over the n - 1 pairs, a = ybar - b xbar, the residual
    variance SSR / (n - 1 - ddof) (ddof 2 the unbiased OLS s^2, 0 the conditional MLE). Half-life in periods -ln 2 /
    ln b for 0 < b < 1 (inf for b >= 1, nan for b <= 0); the regression Delta x = c + lambda x_{t-1} gives b = 1 +
    lambda, and -ln 2 / lambda is its small-lambda approximation.
    conventions:
        ddof: the residual variance's degrees of freedom removed, 0 or 2 (any int below n - 1)
    returns: a dict: intercept and slope (a, b); resid_var (per period, x's units squared); half_life_periods (in
        observations); n (the pairs); resid, the n - 1 residuals x_{t+1} - a - b x_t (Delta x's too, same fit)
    source: Hamilton, Time Series Analysis (1994), 8.1 (OLS for an AR(1)); Chan, Algorithmic Trading (2013),
        the half-life of mean reversion
    aliases: AR(1), AR1, autoregression, [half-life], half life of mean reversion, [mean reversion speed]
    example:
        import qf
        spread = [0.50, 0.42, 0.47, 0.31, 0.38, 0.22, 0.30, 0.19]
        qf.ar1_fit(spread, ddof=2)
    """
    series = _series("x", x, least=3)
    y, z = series[1:], series[:-1]
    m = y.size
    removed = _int("ddof", ddof, 0, m)
    dz = z - z.mean()
    denominator = float(dz @ dz)
    if not denominator > 0:
        raise ValueError("x is constant: no slope exists")
    slope = float(dz @ (y - y.mean())) / denominator
    intercept = float(y.mean() - slope * z.mean())
    resid = y - intercept - slope * z
    if 0 < slope < 1:
        half_life = -LN2 / math.log(slope)
    else:
        half_life = math.inf if slope >= 1 else math.nan
    return Result({"intercept": intercept, "slope": slope, "resid_var": float(resid @ resid) / (m - removed),
                   "half_life_periods": half_life, "n": m, "resid": resid})


def ou_fit(x, *, dt, ddof, scheme):
    """Ornstein-Uhlenbeck speed, level, vol and half-life of a path by an AR(1) regression, exact or Euler scheme named.

    qf.ar1_fit, x_{t+1} = a + b x_t + e, then the scheme's inversion: 'exact' (qf.ou_from_ar1's exact
    discretisation, 0 < b < 1) kappa = -ln b / dt, theta = a / (1 - b), sigma = sqrt(2 kappa Var(e) / (1 - b^2));
    'euler' (Delta x = alpha + beta x + e, beta = b - 1 < 0) kappa = -beta / dt, theta = -alpha / beta, sigma = sd(e)
    / sqrt(dt). half_life = ln 2 / kappa in dt's time unit, half_life_periods in observations. The residuals come
    back for a jump filter or a test of them.
    conventions:
        dt: the spacing in years (1/252 daily) or in whatever unit kappa and the half-life should be in
        ddof: the residual variance's ddof, 0 or 2 (any int below n - 1, as ar1_fit)
        scheme: 'exact' (the exact discretisation) or 'euler' (b = 1 + beta of the Euler regression of Delta x on x)
    returns: a dict: kappa (per unit of dt's time); theta (x's units); sigma (per square root of dt's unit);
        half_life (ln 2 / kappa, in dt's unit); half_life_periods (in observations); intercept, slope, resid_var and
        resid (the AR(1)'s). No jumps: on a path with jumps their variance enters sigma (qf.ou_jump_moments)
    source: Uhlenbeck and Ornstein (1930), Physical Review 36; Brigo and Mercurio, Interest Rate Models (2006),
        3.2.1 (the exact discretisation)
    aliases: Ornstein-Uhlenbeck, OU fit, OU calibration, [mean-reverting process], Vasicek calibration, OU half-life
    example:
        import qf
        spread = [0.50, 0.42, 0.47, 0.31, 0.38, 0.22, 0.30, 0.19]
        qf.ou_fit(spread, dt=1 / 252, ddof=2, scheme="exact")
    """
    choice("scheme", scheme, OU_SCHEMES)
    fit = ar1_fit(x, ddof=ddof)
    step = float(positive("dt", dt))
    if scheme == "exact":
        kappa, theta, sigma = ou_from_ar1(fit["intercept"], fit["slope"], fit["resid_var"], step)
    else:
        if not fit["slope"] < 1:
            raise ValueError("the AR(1) slope must lie below 1 (beta < 0) for a mean-reverting Euler fit")
        kappa = (1.0 - fit["slope"]) / step
        theta, sigma = fit["intercept"] / (1.0 - fit["slope"]), math.sqrt(fit["resid_var"] / step)
    return Result({"kappa": kappa, "theta": theta, "sigma": sigma, "half_life": LN2 / kappa,
                   "half_life_periods": LN2 / (kappa * step), "intercept": fit["intercept"], "slope": fit["slope"],
                   "resid_var": fit["resid_var"], "resid": fit["resid"]})


def _cir_loglik(now, nxt, step, kappa, theta, sigma):
    """The CIR exact transitions' log-likelihood, sum of ln(2c) + ln f_ncx2(2c r_{t+dt}) over the pairs (now, nxt)."""
    c = 2.0 * kappa / (sigma * sigma * -np.expm1(-kappa * step))
    return float(np.sum(np.log(2.0 * c) + stats.ncx2.logpdf(2.0 * c * nxt, 4.0 * kappa * theta / (sigma * sigma),
                                                              2.0 * c * now * np.exp(-kappa * step))))


@_guard(rate=('rates', 'theta'))
def cir_log_likelihood(rates, dt, kappa, theta, sigma):
    """CIR log-likelihood at given parameters: the noncentral chi-square transitions that cir_fit maximises.

    With c = 2 kappa / (sigma^2 (1 - e^{-kappa dt})), 2c r_{t+dt} given r_t is noncentral chi-square of
    4 kappa theta / sigma^2 degrees of freedom and noncentrality 2c r_t e^{-kappa dt}; the log-likelihood is the sum
    of ln(2c) + ln f_ncx2(2c r_{t+dt}) over consecutive pairs, the first rate given. At cir_fit's estimates it is
    cir_fit's log_likelihood. A grid, an optimiser of the caller's own (minimising its negative) or a profile with a
    parameter held calls this, never cir_fit with point bounds.

    conventions:
    - rates: the short-rate series, decimals (percent / 100), oldest first, every one positive, two or more
    - dt: the spacing in years (1/252 for business days, 1/365 for calendar days)
    - kappa, theta, sigma: the mean-reversion speed, the long-run level (a decimal) and the volatility, each above 0
    returns: a float, the sum of the transitions' log-densities
    source: Cox, Ingersoll and Ross (1985), Econometrica 53(2); Brigo and Mercurio, Interest Rate Models (2006),
      3.2.3; Kladivko (2007)
    aliases: CIR log-likelihood, CIR likelihood, CIR loglik
    example:
        qf.cir_log_likelihood(rates=[0.031, 0.0312, 0.0309, 0.0315, 0.0318, 0.0314], dt=1 / 252, kappa=0.8,
                              theta=0.035, sigma=0.05)
    """
    x = np.asarray(rates, dtype=float).ravel()
    if x.size < 2 or np.any(~(x > 0)):
        raise ValueError("rates must hold two or more positive decimals (percent / 100)")
    step = float(positive("dt", dt))
    kappa, theta, sigma = (float(positive(name, value)) for name, value in
                           (("kappa", kappa), ("theta", theta), ("sigma", sigma)))
    return _cir_loglik(x[:-1], x[1:], step, kappa, theta, sigma)
