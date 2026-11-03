"""Copulas: densities, samplers (joint or conditional) and fits; parameter from Kendall's tau; tail dependence.

`u` and `v` are pseudo-observations in (0, 1), arrays that broadcast; the log-likelihood of a sample is the sum of
the log-density over its pairs. A family's AIC is 2k - 2 loglik and BIC k ln(n) - 2 loglik, k its parameter count.
A sampler names its random generator's draws; a function with several results returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, stats
from scipy.special import gammaln, ndtr, ndtri, stdtr, stdtrit

from ._common import Result, choice, out, positive
from .primitives import _stream

__all__ = ["archimedean_copula_sample", "clayton_copula_logpdf", "conditional_copula_sample", "copula_param_from_tau",
           "gaussian_copula_fit", "gaussian_copula_logpdf", "gaussian_copula_sample", "gumbel_copula_logpdf",
           "t_copula_fit", "t_copula_logpdf", "t_copula_sample", "tail_dependence"]

FAMILIES = ("gaussian", "t", "clayton", "gumbel", "frank")
T_FITS = ("mle", "tau")
TAILS = ("upper", "lower")
ARCHIMEDEAN = ("clayton", "gumbel", "frank")
INVERSES = ("closed_form", "numerical")
BISECTIONS = 90                 # halvings of [-40, 40] in the normal scale: below 1e-25 wide, past double precision


def _draws(n, rng):
    """The draws' count and the stream they come from (primitives._stream)."""
    stream = _stream(rng)
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive int")
    return int(n), stream


def _unit(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if np.any(~((u > 0) & (u < 1))) or np.any(~((v > 0) & (v < 1))):
        raise ValueError("u and v must lie strictly inside (0, 1)")
    return u, v


def _rho(rho):
    rho = np.asarray(rho, dtype=float)
    if np.any(~(np.abs(rho) < 1)):
        raise ValueError("rho must lie strictly inside (-1, 1)")
    return rho


def gaussian_copula_logpdf(u, v, rho):
    """ln c(u, v) of the Gaussian copula with correlation rho, x = N^{-1}(u), y = N^{-1}(v):

    ln c = -ln(1 - rho^2)/2 - (rho^2 (x^2 + y^2) - 2 rho x y) / (2 (1 - rho^2)).
    Source: Cherubini, Luciano and Vecchiato, Copula Methods in Finance (2004), the Gaussian copula's density;
    McNeil, Frey and Embrechts, Quantitative Risk Management (2005), 5.1.
    aliases: Gaussian copula density, Gaussian copula log-likelihood, Gaussian copula
    returns: ln c at each (u, v) pair: a float, or an array for arrays
    example:
        qf.gaussian_copula_logpdf(u=[0.2, 0.7], v=[0.3, 0.6], rho=0.5)
    """
    u, v = _unit(u, v)
    rho = _rho(rho)
    x, y = ndtri(u), ndtri(v)
    one = 1.0 - rho * rho
    return out(-0.5 * np.log(one) - (rho * rho * (x * x + y * y) - 2.0 * rho * x * y) / (2.0 * one))


def t_copula_logpdf(u, v, rho, nu):
    """ln c(u, v) of the Student t copula with correlation rho and nu > 0 degrees of freedom,
    x = t_nu^{-1}(u), y = t_nu^{-1}(v): the bivariate t density over the product of its t marginals' densities,

    ln c = lnG((nu+2)/2) + lnG(nu/2) - 2 lnG((nu+1)/2) - ln(1 - rho^2)/2
           - (nu+2)/2 ln(1 + (x^2 - 2 rho x y + y^2) / (nu (1 - rho^2)))
           + (nu+1)/2 [ln(1 + x^2/nu) + ln(1 + y^2/nu)],     lnG the log-gamma function.
    The marginals' log-densities are subtracted, never added. Source: Demarta and McNeil (2005), "The t copula and
    related copulas", International Statistical Review 73(1), eq. (2.2); McNeil, Frey and Embrechts (2005), 5.1.
    aliases: t copula density, Student t copula, t-copula log-likelihood, t copula
    returns: ln c at each (u, v) pair: a float, or an array for arrays
    example:
        qf.t_copula_logpdf(u=[0.2, 0.7], v=[0.3, 0.6], rho=0.5, nu=4.0)
    """
    u, v = _unit(u, v)
    rho = _rho(rho)
    nu = np.asarray(nu, dtype=float)
    if np.any(~(nu > 0)):
        raise ValueError("nu must be positive")
    x, y = stdtrit(nu, u), stdtrit(nu, v)
    one = 1.0 - rho * rho
    head = gammaln((nu + 2.0) / 2.0) + gammaln(nu / 2.0) - 2.0 * gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(one)
    joint = -(nu + 2.0) / 2.0 * np.log1p((x * x - 2.0 * rho * x * y + y * y) / (nu * one))
    margins = (nu + 1.0) / 2.0 * (np.log1p(x * x / nu) + np.log1p(y * y / nu))
    return out(head + joint + margins)


def clayton_copula_logpdf(u, v, theta):
    """ln c(u, v) of the Clayton copula, theta > 0, C(u, v) = (u^-theta + v^-theta - 1)^(-1/theta):

    ln c = ln(1 + theta) - (1 + theta)(ln u + ln v) - (2 + 1/theta) ln(u^-theta + v^-theta - 1).
    Lower-tail dependence 2^(-1/theta). Source: Nelsen, An Introduction to Copulas (2006), 4.2, family (4.2.1);
    Clayton (1978), Biometrika 65.
    aliases: Clayton copula, Clayton density
    returns: ln c at each (u, v) pair: a float, or an array for arrays
    example:
        qf.clayton_copula_logpdf(u=[0.2, 0.7], v=[0.3, 0.6], theta=2.0)
    """
    u, v = _unit(u, v)
    theta = np.asarray(theta, dtype=float)
    if np.any(~(theta > 0)):
        raise ValueError("theta must be positive")
    lu, lv = np.log(u), np.log(v)
    inner = np.exp(-theta * lu) + np.exp(-theta * lv) - 1.0
    return out(np.log1p(theta) - (1.0 + theta) * (lu + lv) - (2.0 + 1.0 / theta) * np.log(inner))


def gumbel_copula_logpdf(u, v, theta):
    """ln c(u, v) of the Gumbel (Gumbel-Hougaard) copula, theta >= 1,
    C(u, v) = exp(-A), A = (x^theta + y^theta)^(1/theta), x = -ln u, y = -ln v:

    ln c = -A + x + y + (theta - 1)(ln x + ln y) + (1/theta - 2) ln(x^theta + y^theta) + ln(A + theta - 1).
    Upper-tail dependence 2 - 2^(1/theta). Source: Nelsen, An Introduction to Copulas (2006), 4.2, family (4.2.4);
    Cherubini, Luciano and Vecchiato (2004), the Gumbel copula's density.
    aliases: Gumbel copula, Gumbel density
    returns: ln c at each (u, v) pair: a float, or an array for arrays
    example:
        qf.gumbel_copula_logpdf(u=[0.2, 0.7], v=[0.3, 0.6], theta=2.0)
    """
    u, v = _unit(u, v)
    theta = np.asarray(theta, dtype=float)
    if np.any(~(theta >= 1)):
        raise ValueError("theta must be at least 1")
    x, y = -np.log(u), -np.log(v)
    total = x ** theta + y ** theta
    a = total ** (1.0 / theta)
    return out(-a + x + y + (theta - 1.0) * (np.log(x) + np.log(y)) + (1.0 / theta - 2.0) * np.log(total)
               + np.log(a + theta - 1.0))


def _frank_tau(theta):
    """Kendall's tau of the Frank copula, 1 - (4 / theta)(1 - D1(theta)), D1 the first Debye function."""
    debye = integrate.quad(lambda x: x / np.expm1(x) if x != 0 else 1.0, 0.0, theta, epsabs=1e-14, epsrel=1e-13)[0]
    return 1.0 - 4.0 / theta * (1.0 - debye / theta)


def _frank_theta(tau):
    """Frank's theta for a Kendall's tau in (-1, 1): tau is odd in theta and increasing; 0 at independence."""
    if tau == 0:
        return 0.0
    size = abs(tau)
    high = 4.0 / (1.0 - size) + 10.0      # tau = 1 - 4/theta + O(1/theta^2) as theta grows
    root = optimize.brentq(lambda theta: _frank_tau(theta) - size, 1e-9, high, xtol=1e-14, rtol=1e-15)
    return float(np.copysign(root, tau))


def copula_param_from_tau(family, tau):
    """The copula parameter that gives Kendall's tau:
    'gaussian' and 't' rho = sin(pi tau / 2) (any elliptical copula); 'clayton' theta = 2 tau / (1 - tau) (tau > 0);
    'gumbel' theta = 1 / (1 - tau) (tau >= 0); 'frank' the root of tau = 1 - (4 / theta)(1 - D1(theta)), D1(theta) =
    (1/theta) int_0^theta x / (e^x - 1) dx the Debye function (any tau; theta 0 at tau 0). The t copula's nu is not
    set by tau (fit it by likelihood with rho fixed).
    Source: Lindskog, McNeil and Schmock (2003), "Kendall's tau for elliptical distributions"; for the Archimedean
    ones, tau = 1 + 4 int_0^1 phi/phi' (Nelsen, An Introduction to Copulas (2006), 5.1.1, corollary 5.1.4, and
    example 5.4 for Frank's).
    aliases: [Kendall's tau], Kendall tau to rho, tau to theta, rank correlation to copula parameter
    returns: rho ('gaussian', 't') or theta ('clayton', 'gumbel', 'frank'): a float, or an array for an array tau
    example:
        qf.copula_param_from_tau(family="clayton", tau=0.4)
    """
    choice("family", family, FAMILIES)
    tau = np.asarray(tau, dtype=float)
    if np.any(~(np.abs(tau) < 1)):
        raise ValueError("tau must lie strictly inside (-1, 1)")
    if family in ("gaussian", "t"):
        return out(np.sin(np.pi * tau / 2.0))
    if family == "clayton":
        if np.any(~(tau > 0)):
            raise ValueError("the Clayton copula needs tau > 0")
        return out(2.0 * tau / (1.0 - tau))
    if family == "frank":
        return out(np.vectorize(_frank_theta, otypes=[float])(tau))
    if np.any(~(tau >= 0)):
        raise ValueError("the Gumbel copula needs tau >= 0")
    return out(1.0 / (1.0 - tau))


# ------------------------------------------------------------------------------------------------ copulas


def _correlation(corr):
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 2:
        raise ValueError("corr must be a square matrix of at least 2 x 2")
    if not (np.allclose(c, c.T, atol=1e-12) and np.allclose(np.diag(c), 1.0, atol=1e-12)):
        raise ValueError("corr must be symmetric with a unit diagonal")
    try:
        return np.linalg.cholesky(c)
    except np.linalg.LinAlgError as error:
        raise ValueError("corr must be positive definite") from error


def gaussian_copula_sample(corr, *, n, rng):
    """Draw n points of a Gaussian copula with correlation matrix corr, uniform margins.

    U = Phi(Z L'), L the Cholesky factor of corr, Z an n x d matrix of standard normals drawn by
    rng.standard_normal((n, d)). Map to any margins with their inverse CDFs (scipy.stats.<law>.ppf(U)). The
    densities are qf.gaussian_copula_logpdf (bivariate).
    conventions:
        n: the number of draws
        rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); the draws are its next n d
            normals
    returns: an n x d array in (0, 1)
    source: McNeil, Frey and Embrechts, Quantitative Risk Management (2005), algorithm 5.9; Cherubini, Luciano and
        Vecchiato, Copula Methods in Finance (2004), 6.6
    aliases: [Gaussian copula], [normal copula], Gaussian copula simulation, sample Gaussian copula, Gaussian copula sample
    example:
        import numpy as np, qf
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        qf.gaussian_copula_sample(corr, n=1000, rng=np.random.default_rng(1))
    """
    lower = _correlation(corr)
    n, rng = _draws(n, rng)
    return ndtr(rng.standard_normal((n, lower.shape[0])) @ lower.T)


def t_copula_sample(corr, *, nu, n, rng):
    """Draw n points of a Student t copula with correlation corr and nu degrees of freedom.

    X = Z L' / sqrt(W / nu), Z an n x d matrix of standard normals (rng.standard_normal((n, d))) and then W a
    chi-square(nu) per row (rng.chisquare(nu, n)), the same W for every column of a row; U = t_nu(X). The densities
    are qf.t_copula_logpdf (bivariate).
    conventions:
        nu: the degrees of freedom, above 0
        n: the number of draws
        rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); normals first, then the
            chi-squares
    returns: an n x d array in (0, 1)
    source: Demarta and McNeil (2005), "The t copula and related copulas", International Statistical Review 73(1);
        McNeil, Frey and Embrechts, Quantitative Risk Management (2005), algorithm 5.10
    aliases: [t copula], [Student t copula], Student-t copula simulation, sample t copula, t copula sample
    example:
        import numpy as np, qf
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        qf.t_copula_sample(corr, nu=4.0, n=1000, rng=np.random.default_rng(1))
    """
    lower = _correlation(corr)
    nu = float(positive("nu", nu))
    n, rng = _draws(n, rng)
    z = rng.standard_normal((n, lower.shape[0])) @ lower.T
    w = rng.chisquare(nu, n)
    return stdtr(nu, z / np.sqrt(w / nu)[:, None])


def _positive_stable(alpha, n, rng):
    """n draws of the positive alpha-stable law with Laplace transform exp(-s^alpha), 0 < alpha < 1, by Kanter's
    representation: (A(U) / E)^((1 - alpha) / alpha), U ~ U(0, pi) then E ~ Exp(1), A(u) = [sin(alpha u)^alpha
    sin((1 - alpha) u)^(1 - alpha) / sin(u)]^(1 / (1 - alpha))."""
    u, e = rng.uniform(0.0, np.pi, n), rng.standard_exponential(n)
    a = (np.sin(alpha * u) ** alpha * np.sin((1.0 - alpha) * u) ** (1.0 - alpha) / np.sin(u)) ** (1.0 / (1.0 - alpha))
    return (a / e) ** ((1.0 - alpha) / alpha)


def archimedean_copula_sample(family, *, theta, dim, n, rng):
    """Draw n points of a Clayton, Gumbel or Frank copula by the Marshall-Olkin algorithm.

    V is the frailty whose Laplace transform psi is the generator's inverse, E_1..E_d independent Exp(1), and U_j =
    psi(E_j / V): 'clayton' V ~ Gamma(1/theta), psi(s) = (1 + s)^(-1/theta), theta > 0; 'gumbel' V positive stable
    of index 1/theta (Kanter's representation), psi(s) = exp(-s^(1/theta)), theta >= 1 (1 is independence);
    'frank' V logarithmic of p = 1 - e^-theta, psi(s) = -ln(1 - p e^-s) / theta, theta > 0 (Marshall-Olkin needs
    positive dependence). Kendall's tau: theta / (theta + 2), 1 - 1/theta, Frank's by qf.copula_param_from_tau;
    densities: qf.clayton_copula_logpdf, qf.gumbel_copula_logpdf. Not u = phi^-1(phi(v) w): no sampler of these.
    conventions:
        theta: the family's parameter (qf.copula_param_from_tau)
        dim: 2 for a pair
        n: the draws
        rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); V first (Gumbel's U then
            E), then the E_j (n x dim)
    returns: an n x dim array in (0, 1), uniform margins
    source: Marshall and Olkin (1988), JASA 83(403); McNeil, Frey and Embrechts, Quantitative Risk Management (2005),
        algorithm 5.48; Kanter (1975), Annals of Probability 3(4)
    aliases: [Clayton copula], [Gumbel copula], Frank copula, Archimedean copula, Marshall-Olkin, Clayton copula simulation, Gumbel copula simulation, Frank copula simulation, Clayton copula sample, Gumbel copula sample
    example:
        import qf
        theta = 3.0
        qf.archimedean_copula_sample("clayton", theta=theta, dim=2, n=5000, rng=1)
    """
    choice("family", family, ARCHIMEDEAN)
    theta = float(theta)
    if not (theta >= 1 if family == "gumbel" else theta > 0):
        raise ValueError(f"theta must be {'at least 1' if family == 'gumbel' else 'positive'} for the {family} copula")
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)) or dim < 2:
        raise ValueError("dim must be an int of at least 2")
    n, rng = _draws(n, rng)
    if family == "clayton":
        v = rng.gamma(1.0 / theta, 1.0, n)
    elif family == "gumbel":
        v = np.ones(n) if theta == 1.0 else _positive_stable(1.0 / theta, n, rng)
    else:
        v = rng.logseries(-np.expm1(-theta), n).astype(float)
    s = rng.standard_exponential((n, int(dim))) / v[:, None]
    if family == "clayton":
        return (1.0 + s) ** (-1.0 / theta)
    if family == "gumbel":
        return np.exp(-s ** (1.0 / theta))
    return -np.log1p(np.expm1(-theta) * np.exp(-s)) / theta


def _pair(u, v):
    u, v = np.asarray(u, dtype=float).ravel(), np.asarray(v, dtype=float).ravel()
    if u.size != v.size or u.size < 3:
        raise ValueError("u and v must be equal-length samples of at least 3 pseudo-observations")
    return u, v


def _scalar_max(f, lo, hi, points):
    """The maximiser of f on [lo, hi]: the best of a grid, then a bounded Brent search between its neighbours."""
    grid = np.linspace(lo, hi, points)
    values = np.array([f(x) for x in grid])
    best = int(np.argmax(values))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, points - 1)]
    found = optimize.minimize_scalar(lambda x: -f(x), bounds=(left, right), method="bounded",
                                     options={"xatol": 1e-10})
    return float(found.x) if -found.fun >= values[best] else float(grid[best])


def _scores(loglik, k, n):
    return Result({"loglik": loglik, "aic": 2.0 * k - 2.0 * loglik, "bic": k * math.log(n) - 2.0 * loglik, "n": n})


def gaussian_copula_fit(u, v):
    """Maximum-likelihood correlation of a bivariate Gaussian copula from pseudo-observations.

    rho maximises sum_i ln c(u_i, v_i; rho) with c the density qf.gaussian_copula_logpdf (a grid on (-0.999, 0.999),
    then Brent). AIC = 2k - 2 ln L and BIC = k ln n - 2 ln L with k = 1. Pseudo-observations from data x: u =
    scipy.stats.rankdata(x) / (n + 1).
    conventions:
        (none: u and v are the two margins' pseudo-observations in (0, 1))
    returns: a dict: rho, loglik, aic, bic, n
    source: Genest, Ghoudi and Rivest (1995), Biometrika 82(3) (maximum pseudo-likelihood); Cherubini, Luciano and
        Vecchiato, Copula Methods in Finance (2004), 5.3
    aliases: fit Gaussian copula, Gaussian copula MLE, normal copula fit, copula maximum likelihood
    example:
        import numpy as np, qf
        uv = qf.gaussian_copula_sample(np.array([[1.0, 0.5], [0.5, 1.0]]), n=500, rng=np.random.default_rng(2))
        qf.gaussian_copula_fit(uv[:, 0], uv[:, 1])
    """
    u, v = _pair(u, v)
    loglik = lambda rho: float(np.sum(gaussian_copula_logpdf(u, v, rho)))  # noqa: E731
    rho = _scalar_max(loglik, -0.999, 0.999, 81)
    return Result({"rho": rho, **_scores(loglik(rho), 1, u.size)})


def t_copula_fit(u, v, *, method):
    """Fit a bivariate t copula's rho and nu: joint MLE, or rho from Kendall's tau then nu.

    'mle' maximises sum_i ln c(u_i, v_i; rho, nu) over both (qf.t_copula_logpdf; Nelder-Mead on (atanh rho, ln nu)
    from the 'tau' fit); 'tau' sets rho = sin(pi tau / 2) from Kendall's tau (scipy.stats.kendalltau) and maximises
    over nu alone. nu is searched in [1, 1000]; a fit near 1000 says the data favour the Gaussian copula. AIC and
    BIC with k = 2.
    conventions:
        method: 'mle' (joint maximum likelihood) or 'tau' (rho from Kendall's tau, then nu by likelihood)
    returns: a dict: rho, nu, loglik, aic, bic, n
    source: Demarta and McNeil (2005), International Statistical Review 73(1); McNeil, Frey and Embrechts,
        Quantitative Risk Management (2005), 5.5.3 (the tau-then-likelihood fit)
    aliases: fit t copula, t copula MLE, Student t copula fit, copula degrees of freedom
    example:
        import numpy as np, qf
        uv = qf.t_copula_sample(np.array([[1.0, 0.5], [0.5, 1.0]]), nu=5.0, n=500, rng=np.random.default_rng(2))
        qf.t_copula_fit(uv[:, 0], uv[:, 1], method="tau")
    """
    choice("method", method, T_FITS)
    u, v = _pair(u, v)
    loglik = lambda rho, nu: float(np.sum(t_copula_logpdf(u, v, rho, nu)))  # noqa: E731
    rho = math.sin(math.pi * float(stats.kendalltau(u, v).statistic) / 2.0)
    rho = min(max(rho, -0.999), 0.999)
    log_nu = _scalar_max(lambda x: loglik(rho, math.exp(x)), 0.0, math.log(1000.0), 25)
    if method == "mle":
        def cost(p):
            if abs(p[0]) > 7.6 or not 0.0 <= p[1] <= math.log(1000.0):
                return math.inf
            return -loglik(math.tanh(p[0]), math.exp(p[1]))
        found = optimize.minimize(cost, [math.atanh(rho), log_nu], method="Nelder-Mead",
                                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 4000})
        rho, log_nu = math.tanh(found.x[0]), float(found.x[1])
    nu = math.exp(log_nu)
    return Result({"rho": rho, "nu": nu, **_scores(loglik(rho, nu), 2, u.size)})


# ------------------------------------------------------------------------------------------ tails and conditionals


def tail_dependence(u, v, *, q, tail):
    """Empirical upper or lower tail-dependence coefficient of a pair of uniforms at a quantile.

    'upper': #{u > q and v > q} / #{u > q}, the estimate of P(V > q | U > q); 'lower': #{u <= q and v <= q} / #{u <=
    q}, of P(V <= q | U <= q). The coefficient is the limit q -> 1 (upper) or q -> 0 (lower): at a finite q
    independence gives 1 - q (upper) or q (lower), a comonotonic pair 1. Pseudo-observations from data are ranks / (n
    + 1) (scipy.stats.rankdata(x) / (n + 1)). Closed forms: Clayton's lower 2^(-1/theta), Gumbel's upper 2 -
    2^(1/theta), the Gaussian's 0 (rho < 1), the t copula's 2 t_{nu+1}(-sqrt((nu + 1)(1 - rho) / (1 + rho))).
    conventions:
        q: the tail's own level in (0, 1): for 'lower' a small level (0.05 for the worst 5%: counts u <= q), for
            'upper' a level near 1 (0.95 for the top 5%: counts u > q); never an upper level on a lower-tail call
        tail: 'upper' (both large together) or 'lower' (both small together: joint crashes of returns)
    returns: a float in [0, 1]; nan where no u lies beyond q
    source: Joe, Multivariate Models and Dependence Concepts (1997), 2.1.10; Frahm, Junker and Schmidt (2005),
        "Estimating the tail-dependence coefficient", Insurance: Mathematics and Economics 37(1)
    aliases: tail dependence, tail dependence coefficient, upper tail dependence, lower tail dependence, tail coefficient, [tail dependence coeff]
    example:
        import numpy as np, qf
        u, v = qf.gaussian_copula_sample([[1.0, 0.7], [0.7, 1.0]], n=5000, rng=np.random.default_rng(1)).T
        qf.tail_dependence(u, v, q=0.05, tail="lower")  # .44958
    """
    choice("tail", tail, TAILS)
    u, v = _pair(u, v)
    level = float(q)
    if not 0.0 < level < 1.0:
        raise ValueError(f"q must lie strictly inside (0, 1), not {level}")
    beyond_u, beyond_v = (u > level, v > level) if tail == "upper" else (u <= level, v <= level)
    count = int(beyond_u.sum())
    return float(np.sum(beyond_u & beyond_v) / count) if count else math.nan


def _h(u, v, family, p, nu):
    """C_{2|1}(v | u) = dC(u, v)/du, the law of V given U = u."""
    if family == "gaussian":
        return ndtr((ndtri(v) - p * ndtri(u)) / math.sqrt(1.0 - p * p))
    if family == "t":
        x, y = stdtrit(nu, u), stdtrit(nu, v)
        return stdtr(nu + 1.0, (y - p * x) / np.sqrt((nu + x * x) * (1.0 - p * p) / (nu + 1.0)))
    if family == "clayton":
        return u ** (-p - 1.0) * (u ** -p + v ** -p - 1.0) ** (-1.0 / p - 1.0)
    if family == "gumbel":
        x, y = -np.log(u), -np.log(v)
        a = (x ** p + y ** p) ** (1.0 / p)
        return np.exp(-a) * a ** (1.0 - p) * x ** (p - 1.0) / u
    if p == 0:
        return v + 0.0 * u
    eu, ev = np.expm1(-p * u), np.expm1(-p * v)
    return (eu + 1.0) * ev / (math.expm1(-p) + eu * ev)


def _h_inverse(u, w, family, p, nu):
    """The closed-form inverse of _h in v, where the family has one."""
    if family == "gaussian":
        return ndtr(p * ndtri(u) + math.sqrt(1.0 - p * p) * ndtri(w))
    if family == "t":
        x = stdtrit(nu, u)
        return stdtr(nu, p * x + stdtrit(nu + 1.0, w) * np.sqrt((nu + x * x) * (1.0 - p * p) / (nu + 1.0)))
    if family == "clayton":
        return (1.0 + u ** -p * (w ** (-p / (1.0 + p)) - 1.0)) ** (-1.0 / p)
    if family == "frank":
        if p == 0:
            return w + 0.0 * u
        return -np.log1p(w * math.expm1(-p) / (w + (1.0 - w) * np.exp(-p * u))) / p
    raise ValueError("the Gumbel copula's conditional law has no closed-form inverse: inverse 'numerical'")


def conditional_copula_sample(u, *, family, parameter, nu, inverse, w, rng):
    """Conditional copula sampling: V given U = u by inverting the copula's conditional law, closed form or numerical.

    C_{2|1}(v | u) = dC(u, v)/du is the law of V given U = u; with W uniform and independent of U, V = C_{2|1}^-1(W |
    U) has the copula C with U. Closed forms: Gaussian Phi(rho x + sqrt(1 - rho^2) Phi^-1(w)), x = Phi^-1(u); t with
    x = t_nu^-1(u), t_nu(rho x + t_{nu+1}^-1(w) sqrt((nu + x^2)(1 - rho^2)/(nu + 1))); Clayton (1 + u^-theta
    (w^(-theta/(1 + theta)) - 1))^(-1/theta); Frank -ln(1 + w (e^-theta - 1)/(w + (1 - w) e^(-theta u)))/theta. The
    Gumbel's is numerical alone: the root in v of C_{2|1}(v | u) = w (bisection). The families are exchangeable, so U
    given V = v is the same call with v in u's place. Pairs from a joint draw are qf.gaussian_copula_sample and the
    other samplers.
    conventions:
        u: the conditioning margin's uniforms in (0, 1) (a number or an array)
        family: 'gaussian', 't', 'clayton', 'gumbel' or 'frank'
        parameter: rho for 'gaussian' and 't' (inside (-1, 1)); theta for 'clayton' (above 0), 'gumbel' (at least 1)
            or 'frank' (any; 0 independence); from Kendall's tau by qf.copula_param_from_tau
        nu: the t copula's degrees of freedom, above 0; None for the other families
        inverse: 'closed_form' (not for 'gumbel') or 'numerical' (the h-function's root, any family)
        w: independent uniforms shaped as u (the task's own), or None to draw them with rng
        rng: a Generator or int seed (default_rng), or numpy's legacy stream (a RandomState), drawing w
            (rng.uniform(size=u's shape)); or None where w is given
    returns: v, uniforms shaped as u: a float, or an array for an array
    source: Nelsen, An Introduction to Copulas (2006), 2.9 (conditional sampling) and 4.2; Aas, Czado, Frigessi and
        Bakken (2009), Insurance: Mathematics and Economics 44 (the h-functions); Cherubini, Luciano and Vecchiato
        (2004), 6.2
    aliases: conditional copula sampling, conditional sampling, h-function inversion, inverse h-function
    example:
        import qf
        qf.conditional_copula_sample(u=[0.1, 0.5, 0.9], family="clayton", parameter=2.0, nu=None, inverse="closed_form", w=[0.3, 0.6, 0.8], rng=None)
        # [0.08975044, 0.61746227, 0.91362451]
    """
    choice("family", family, FAMILIES)
    choice("inverse", inverse, INVERSES)
    u = np.asarray(u, dtype=float)
    if np.any(~((u > 0) & (u < 1))):
        raise ValueError("u must lie strictly inside (0, 1)")
    p = float(parameter)
    if family in ("gaussian", "t"):
        _rho(p)
    elif not (p > 0 if family == "clayton" else p >= 1 if family == "gumbel" else math.isfinite(p)):
        raise ValueError(f"parameter (theta) out of range for the {family} copula")
    if family == "t":
        nu = float(positive("nu", nu))
    elif nu is not None:
        raise ValueError("nu is for the t copula: pass None")
    if (w is None) == (rng is None):
        raise ValueError("pass the uniforms w, or rng to draw them: exactly one, the other None")
    if w is None:
        w = _stream(rng).uniform(size=u.shape)
    w = np.broadcast_to(np.asarray(w, dtype=float), u.shape)
    if np.any(~((w > 0) & (w < 1))):
        raise ValueError("w must lie strictly inside (0, 1)")
    if inverse == "closed_form":
        return out(_h_inverse(u, w, family, p, nu))
    low, high = np.full(u.shape, -40.0), np.full(u.shape, 40.0)
    for _ in range(BISECTIONS):
        middle = 0.5 * (low + high)
        below = _h(u, ndtr(middle), family, p, nu) < w
        low, high = np.where(below, middle, low), np.where(below, high, middle)
    return out(ndtr(0.5 * (low + high)))
