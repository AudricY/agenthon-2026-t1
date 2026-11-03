"""What every qf module shares: the normal CDF, the bivariate normal CDF, argument checks and the return of a scalar
as a float."""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import log_ndtr, ndtr

__all__ = ["Result", "bivariate_normal_cdf", "choice", "integer", "log_ndtr", "ndtr", "out", "positive"]


class Result(dict):
    """A function's dict of results: a key it does not hold raises a KeyError naming every key it does, so a guessed
    key ('t_stat' for 'tstat') is said, never silently missed."""

    def __missing__(self, key):
        raise KeyError(f"{key!r} is not a key of this result; its keys are {', '.join(map(repr, self))}")


def out(value):
    """A 0-d result as a Python float, an array as it is."""
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


def choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    """`value` where it is one of `allowed`, else a ValueError naming them."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, allowed))}, not {value!r}")
    return value


def positive(name: str, value) -> np.ndarray:
    """`value` as a float array, refused where any element is not positive."""
    array = np.asarray(value, dtype=float)
    if np.any(~(array > 0)):
        raise ValueError(f"{name} must be positive")
    return array


def integer(name: str, value, least: int) -> int:
    """`value` as an int where it is an int (not a bool) of at least `least`, else a ValueError naming it."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < least:
        raise ValueError(f"{name} must be an int of at least {least}")
    return int(value)


def bivariate_normal_cdf(h, k, rho):
    """M(h, k; rho) = P(X <= h, Y <= k) for standard normals of correlation rho, |rho| < 1, by its definition
    int_{-inf}^h phi(x) N((k - rho x) / sqrt(1 - rho^2)) dx (adaptive quadrature from -40, where phi is below
    1e-300)."""
    s = np.sqrt(1.0 - rho * rho)

    def integrand(x):
        return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi) * ndtr((k - rho * x) / s)

    found = integrate.quad(integrand, -40.0, h, limit=200, epsabs=1e-15, epsrel=1e-13)[0] if h > -40.0 else 0.0
    return float(min(max(found, 0.0), 1.0))
