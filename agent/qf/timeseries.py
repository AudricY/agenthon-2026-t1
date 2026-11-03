"""Time-series models: ARMA with the ACF, the Kalman filter and smoother, seasonal decomposition.

`import qf`. A series comes oldest first, one value a period; a state-space model's matrices are numpy arrays (a
list of lists will do). A function with several results returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import linalg, optimize, signal

from ._common import Result, choice, integer

__all__ = ["arma_fit", "kalman_filter", "seasonal_decompose"]

LIKELIHOODS = ("exact", "conditional")
MEANS = ("constant", "zero")
ACF_DENOMINATORS = ("n", "n-k")
STARTS = ("first_state", "before_first")
SMOOTHERS = ("none", "rts")
DECOMPOSITIONS = ("additive", "multiplicative")
STEADY = 1e-13                  # the ARMA filter's covariance this close to its limit: the plain recursion from there


# ------------------------------------------------------------------------------------------------ ARMA


def _coefficients(partials):
    """AR coefficients from partial autocorrelations, by the Durbin-Levinson recursion."""
    phi = np.zeros(0)
    for r in partials:
        phi = np.append(phi - r * phi[::-1], r)
    return phi


def _partials(phi):
    """The partial autocorrelations of AR coefficients, the Durbin-Levinson recursion run backwards."""
    phi, found = np.asarray(phi, dtype=float), np.zeros(len(phi))
    for k in range(len(phi), 0, -1):
        r = phi[k - 1]
        found[k - 1] = r
        if k > 1:
            phi = (phi[:k - 1] + r * phi[:k - 1][::-1]) / (1.0 - r * r)
    return found


def _constrained(u):
    """Unconstrained numbers to stationary AR coefficients (Monahan 1984; statsmodels' transform)."""
    return _coefficients(u / np.sqrt(1.0 + u * u))


def _unconstrained(phi):
    r = np.clip(_partials(phi), -0.99, 0.99)
    return r / np.sqrt(1.0 - r * r)


def _split(params, p, q, constant):
    """(mean, phi, theta) from the optimiser's vector: the mean, then the AR and MA parts unconstrained."""
    mean = params[0] if constant else 0.0
    rest = params[1:] if constant else params
    return mean, _constrained(rest[:p]), -_constrained(rest[p:p + q])


def _arma_filter(phi, theta):
    """lfilter's (b, a) for the ARMA residual recursion e_t = y_t - sum phi_i y_{t-i} - sum theta_j e_{t-j}."""
    return np.r_[1.0, -phi], np.r_[1.0, theta]


def _conditional_residuals(y, phi, theta):
    """The residuals for t = p .. n - 1, the first p values given and the earlier residuals 0."""
    p = phi.size
    if p == 0 and theta.size == 0:
        return y.copy()
    b, a = _arma_filter(phi, theta)
    zi = signal.lfiltic(b, a, np.zeros(theta.size), y[p - 1::-1] if p else np.zeros(0))
    return signal.lfilter(b, a, y[p:], zi=zi)[0]


def _innovations(y, phi, theta):
    """The exact Gaussian one-step errors v_t and their variances f_t (in units of the shock variance) of a zero-mean
    ARMA, by the Kalman filter on Harvey's state space from the stationary covariance; once the covariance reaches
    its limit the filter is the residual recursion, run from there by lfilter. Also return the next state's
    conditional mean and covariance, in units of the shock variance."""
    n, p, q = y.size, phi.size, theta.size
    if p == 0 and q == 0:
        return y.copy(), np.ones(n), np.zeros(1), np.ones((1, 1))
    r = max(p, q + 1)
    tm = np.zeros((r, r))
    tm[:p, 0] = phi
    tm[:-1, 1:] += np.eye(r - 1)
    rv = np.zeros(r)
    rv[0], rv[1:q + 1] = 1.0, theta
    limit = np.outer(rv, rv)
    cov, state = linalg.solve_discrete_lyapunov(tm, limit), np.zeros(r)
    v, f = np.empty(n), np.ones(n)
    for t in range(n):
        v[t], f[t] = y[t] - state[0], cov[0, 0]
        gain = tm @ cov[:, 0] / f[t]
        state = tm @ state + gain * v[t]
        cov = tm @ cov @ tm.T + limit - np.outer(gain, gain) * f[t]
        if t + 1 >= r and t + 1 < n and np.max(np.abs(cov - limit)) < STEADY:
            b, a = _arma_filter(phi, theta)
            zi = signal.lfiltic(b, a, v[t::-1][:q] if q else np.zeros(0), y[t::-1][:p] if p else np.zeros(0))
            v[t + 1:] = signal.lfilter(b, a, y[t + 1:], zi=zi)[0]
            state = np.array([sum(phi[j] * y[n - 1 - j + i] for j in range(i, p)) +
                              sum(theta[j] * v[n - 1 - j + i] for j in range(i, q)) for i in range(r)])
            cov = limit
            break
    return v, f, state, cov


def _exact_loglik(y, phi, theta):
    """The exact Gaussian log-likelihood with the shock variance at its maximum, and that variance."""
    v, f, _, _ = _innovations(y, phi, theta)
    sigma2 = float(np.mean(v * v / f))
    return -0.5 * y.size * (math.log(2.0 * math.pi) + math.log(sigma2) + 1.0) - 0.5 * float(np.sum(np.log(f))), sigma2


def _acf(x, lags, denominator):
    """The sample autocorrelations 0 .. lags around the sample mean, and the partial ones by Durbin-Levinson."""
    d = x - x.mean()
    n = d.size
    gamma = np.array([d[k:] @ d[:n - k] for k in range(lags + 1)])
    gamma = gamma / (n if denominator == "n" else n - np.arange(lags + 1))
    rho = gamma / gamma[0]
    partial, phi = np.ones(lags + 1), np.zeros(0)
    for k in range(1, lags + 1):
        r = (rho[k] - phi @ rho[k - 1:0:-1]) / (1.0 - phi @ rho[1:k])
        phi = np.append(phi - r * phi[::-1], r)
        partial[k] = r
    return rho, partial


def _psi(phi, theta, steps):
    """The MA(infinity) weights psi_0 .. psi_{steps-1}: psi_0 = 1, psi_j = theta_j + sum_i phi_i psi_{j-i}."""
    psi = np.zeros(steps)
    psi[0] = 1.0
    for j in range(1, steps):
        psi[j] = (theta[j - 1] if j <= theta.size else 0.0) + sum(phi[i - 1] * psi[j - i]
                                                                    for i in range(1, min(j, phi.size) + 1))
    return psi


def arma_fit(series, order, likelihood, mean, acf_lags, acf_denominator, horizon):
    """ARMA(p, q) estimation, exact or conditional MLE; the ACF and PACF; forecasts and their variance.

    x_t - mu = sum_{i<=p} phi_i (x_{t-i} - mu) + e_t + sum_{j<=q} theta_j e_{t-j}, e_t ~ N(0, sigma^2), stationary and
    invertible. 'exact': the full Gaussian likelihood by the Kalman filter (statsmodels' ARIMA(order=(p, 0, q)) to
    rounding); 'conditional': least squares for t > p, earlier shocks 0 (for an AR(p), OLS on the lags). Forecasts by
    the conditional state for 'exact', retaining uncertainty in the presample shocks; 'conditional' uses the
    recursion and variance sigma^2 sum_{j<h} psi_j^2 (psi the MA(infinity) weights). The ACF rho_k = gamma_k /
    gamma_0 around the sample mean, the PACF by Durbin-Levinson; an MA(1) has rho_1 = theta / (1 + theta^2).

    Ljung-Box: Q = n (n + 2) sum_{k<=m} rho_k^2 / (n - k) of the residuals, chi-square, m - p - q degrees
    (statsmodels' acorr_ljungbox(resid, lags=[m], model_df=p + q)); ARCH-LM: n R^2 of e_t^2 on m of its lags,
    chi-square, m degrees (het_arch(resid, nlags=m)).

    conventions:
    - series: the values, oldest first, NaN refused
    - order: (p, q), the AR and MA orders
    - likelihood: 'exact' (full maximum likelihood) or 'conditional' (conditional sum of squares)
    - mean: 'constant' estimates mu (statsmodels' trend 'c', its const the mean, not an intercept); 'zero' fixes mu = 0
    - acf_lags: the lags of the ACF and PACF returned; acf_denominator: 'n' (gamma_k divides by n, statsmodels'
      acf and its pacf 'ywm') or 'n-k' (by n - k, adjusted=True and pacf 'ywadjusted')
    - horizon: the forecast steps after the last value, an int of 1 or more
    returns: a dict: ar (phi); ma (theta); mean (mu); sigma2; loglik; aic and bic (sigma2 counted); resid (the
      one-step errors, NaN for the first p under 'conditional'); nobs; acf and pacf (lags 0 .. acf_lags); forecast,
      forecast_var and psi (one per step ahead)
    source: Hamilton, Time Series Analysis (1994), ch. 3-5 and 13.4; Tsay, Analysis of Financial Time Series (2010),
      ch. 2; Ljung and Box (1978); Engle (1982)
    aliases: ARMA, ARIMA, ARMA model, MA(1), AR(2), Box-Jenkins, ACF, PACF, Ljung-Box, ARCH-LM, [autocorrelation]
    example:
        qf.arma_fit(series=[0.5, 0.8, 0.3, -0.2, 0.1, 0.6, 0.9, 0.4, -0.1, -0.3, 0.2, 0.5], order=(1, 0),
                    likelihood="exact", mean="constant", acf_lags=3, acf_denominator="n",
                    horizon=2)  # forecast approximately [0.391236, 0.350380]
    """
    choice("likelihood", likelihood, LIKELIHOODS)
    choice("mean", mean, MEANS)
    choice("acf_denominator", acf_denominator, ACF_DENOMINATORS)
    x = np.asarray(series, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("series must be finite (drop the NaN first)")
    if len(order) != 2:
        raise ValueError("order must be (p, q)")
    p, q = integer("p", order[0], 0), integer("q", order[1], 0)
    lags, steps = integer("acf_lags", acf_lags, 0), integer("horizon", horizon, 1)
    constant = mean == "constant"
    if x.size < p + q + int(constant) + 3 or acf_lags >= x.size:
        raise ValueError("series is too short for the order (or acf_lags)")
    centre = x.mean() if constant else 0.0
    start = [centre] if constant else []
    if p:
        lagged = np.column_stack([x[p - i:x.size - i] - centre for i in range(1, p + 1)])
        guess = np.linalg.lstsq(lagged, x[p:] - centre, rcond=None)[0]
        start += list(_unconstrained(guess))
    start = np.array(start + [0.0] * q)

    def squares(params):
        mu, phi, theta = _split(params, p, q, constant)
        return _conditional_residuals(x - mu, phi, theta)

    if start.size:
        start = optimize.least_squares(squares, start, xtol=1e-14, ftol=1e-14, gtol=1e-14).x
    if likelihood == "exact" and start.size:
        found = optimize.minimize(lambda params: -_exact_loglik(x - _split(params, p, q, constant)[0],
                                                               *_split(params, p, q, constant)[1:])[0],
                                  start, method="BFGS", options={"gtol": 1e-8})
        start = found.x
    mu, phi, theta = _split(start, p, q, constant)
    y = x - mu
    if likelihood == "exact":
        loglik, sigma2 = _exact_loglik(y, phi, theta)
        resid, _, state, cov = _innovations(y, phi, theta)
        nobs = x.size
    else:
        part = _conditional_residuals(y, phi, theta)
        nobs = part.size
        sigma2 = float(part @ part / nobs)
        loglik = -0.5 * nobs * (math.log(2.0 * math.pi * sigma2) + 1.0)
        resid = np.r_[np.full(p, np.nan), part]
    forecasts, errors = np.r_[y, np.zeros(steps)], np.r_[np.nan_to_num(resid), np.zeros(steps)]
    for h in range(steps):
        t = x.size + h
        forecasts[t] = sum(phi[i - 1] * forecasts[t - i] for i in range(1, p + 1)) + sum(
            theta[j - 1] * errors[t - j] for j in range(1, q + 1))
    psi = _psi(phi, theta, steps)
    forecast_variance = sigma2 * np.cumsum(psi * psi)
    if likelihood == "exact":
        r = max(p, q + 1)
        tm = np.zeros((r, r))
        tm[:p, 0] = phi
        tm[:-1, 1:] += np.eye(r - 1)
        rv = np.r_[1.0, theta, np.zeros(r - q - 1)]
        for h in range(steps):
            forecasts[x.size + h], forecast_variance[h] = state[0], sigma2 * cov[0, 0]
            state, cov = tm @ state, tm @ cov @ tm.T + np.outer(rv, rv)
    acf, pacf = _acf(x, lags, acf_denominator)
    k = p + q + int(constant) + 1
    return Result({"ar": phi, "ma": theta, "mean": float(mu), "sigma2": sigma2, "loglik": loglik,
                   "aic": -2.0 * loglik + 2.0 * k, "bic": -2.0 * loglik + k * math.log(nobs), "resid": resid,
                   "nobs": nobs, "acf": acf, "pacf": pacf, "forecast": forecasts[x.size:] + mu,
                   "forecast_var": forecast_variance, "psi": psi})


# ------------------------------------------------------------------------------------------------ Kalman


def _matrix(name, value, shape):
    """`value` as a finite array of `shape`; a scalar or a vector of the right size is reshaped, nothing else is."""
    found = np.asarray(value, dtype=float)
    if found.shape != shape and found.size == math.prod(shape) and (found.ndim <= 1 or max(shape) == found.size):
        found = found.reshape(shape)
    if found.shape != shape or not np.all(np.isfinite(found)):
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return found


def _covariance(name, value, shape):
    found = _matrix(name, value, shape)
    if not np.allclose(found, found.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be a symmetric covariance matrix")
    found = (found + found.T) / 2.0
    if np.linalg.eigvalsh(found).min() < -1e-12 * max(1.0, np.max(np.abs(found))):
        raise ValueError(f"{name} must be a positive semidefinite covariance matrix")
    return found


def kalman_filter(observations, transition, observation_matrix, state_cov, observation_cov, initial_state,
                  initial_cov, start, smoother):
    """Kalman filter and Rauch-Tung-Striebel smoother of a state-space model, with its log-likelihood.

    State x_t = F x_{t-1} + w_t, w ~ N(0, Q); observation y_t = H_t x_t + v_t, v ~ N(0, R). Predict a = F a, P = F P F'
    + Q; innovation v_t = y_t - H a, S = H P H' + R; gain K = P H' S^{-1}; update a + K v_t, P - K S K'. Log-likelihood
    -1/2 sum_t [ln det S_t + v_t' S_t^{-1} v_t + n_t ln 2 pi], n_t values observed (a NaN: no update). Smoother: C =
    P_f F' P_pred^{-1}, smoothed = filtered + C (smoothed next - predicted next). A time-varying hedge ratio or beta:
    the coefficients the state, F = I, H_t the regressors at t.

    conventions:
    - observations: (T, n), or (T,) for one series; NaN where a value is missing
    - transition: F (m x m); observation_matrix: H (n x m), or (T, n, m) when it changes with t ((T, m) for one
      series: the regressors at each date)
    - state_cov: Q (m x m); observation_cov: R (n x n); variances, not standard deviations
    - initial_state, initial_cov: the state's mean (m,) and covariance (m x m) at the start
    - start: 'first_state' (they are the first observation's state, updated by it with no prediction first;
      statsmodels, pykalman) or 'before_first' (the state a period before it, predicted once first; Hamilton's)
    - smoother: 'none' or 'rts' (Rauch-Tung-Striebel, the states given all the data)
    returns: a dict: predicted_state, predicted_cov (before each observation); filtered_state, filtered_cov (after);
      innovations (T x n, NaN where missing); innovation_cov; gain (T x m x n); loglik and loglik_no_constant (without
      the ln 2 pi terms); next_state, next_cov (after the last); smoothed_state, smoothed_cov (None with 'none')
    source: Harvey, Forecasting, Structural Time Series Models and the Kalman Filter (1989), ch. 3; Durbin and
      Koopman, Time Series Analysis by State Space Methods (2012), ch. 4; Rauch, Tung and Striebel (1965)
    aliases: Kalman filter, Kalman smoother, RTS smoother, Rauch-Tung-Striebel, state space model, time-varying beta
    example:
        qf.kalman_filter(observations=[1.1, 0.9, 1.3, 1.2, 1.0], transition=[[1.0]], observation_matrix=[[1.0]],
                         state_cov=[[0.01]], observation_cov=[[0.1]], initial_state=[1.0], initial_cov=[[1.0]],
                         start="first_state", smoother="rts")  # final filtered state approximately 1.097570
    """
    choice("start", start, STARTS)
    choice("smoother", smoother, SMOOTHERS)
    y = np.asarray(observations, dtype=float)
    y = y[:, None] if y.ndim == 1 else y
    if y.ndim != 2 or y.shape[0] < 1 or np.any(np.isinf(y)):
        raise ValueError("observations must be (T, n) or (T,), NaN where missing")
    t_count, n = y.shape
    f = np.asarray(transition, dtype=float)
    m = f.shape[0] if f.ndim == 2 else 1
    f = _matrix("transition", f, (m, m))
    h = np.asarray(observation_matrix, dtype=float)
    if n == 1 and h.ndim == 2 and h.shape[0] == t_count > 1:
        h = h[:, None, :]                               # one series' regressors, a row a date
    h = _matrix("observation_matrix", h, (t_count, n, m)) if h.ndim == 3 else np.broadcast_to(
        _matrix("observation_matrix", h, (n, m)), (t_count, n, m))
    q = _covariance("state_cov", state_cov, (m, m))
    r = _covariance("observation_cov", observation_cov, (n, n))
    a, p = _matrix("initial_state", initial_state, (m,)), _covariance("initial_cov", initial_cov, (m, m))
    if start == "before_first":
        a, p = f @ a, f @ p @ f.T + q
    shape = {"state": (t_count, m), "cov": (t_count, m, m)}
    predicted, predicted_cov = np.empty(shape["state"]), np.empty(shape["cov"])
    filtered, filtered_cov = np.empty(shape["state"]), np.empty(shape["cov"])
    innovations, innovation_cov = np.full((t_count, n), np.nan), np.full((t_count, n, n), np.nan)
    gain = np.zeros((t_count, m, n))
    kernel, observed = 0.0, 0
    for t in range(t_count):
        predicted[t], predicted_cov[t] = a, p
        keep = np.isfinite(y[t])
        if keep.any():
            ht = h[t][keep]
            s = ht @ p @ ht.T + r[np.ix_(keep, keep)]
            v = y[t, keep] - ht @ a
            k = np.linalg.solve(s, ht @ p).T
            a, p = a + k @ v, p - k @ s @ k.T
            p = (p + p.T) / 2.0
            kernel -= 0.5 * (np.linalg.slogdet(s)[1] + v @ np.linalg.solve(s, v))
            observed += int(keep.sum())
            innovations[t, keep], innovation_cov[t][np.ix_(keep, keep)], gain[t][:, keep] = v, s, k
        filtered[t], filtered_cov[t] = a, p
        a, p = f @ a, f @ p @ f.T + q
    found = {"predicted_state": predicted, "predicted_cov": predicted_cov, "filtered_state": filtered,
             "filtered_cov": filtered_cov, "innovations": innovations, "innovation_cov": innovation_cov,
             "gain": gain, "loglik": kernel - 0.5 * observed * math.log(2.0 * math.pi),
             "loglik_no_constant": kernel, "next_state": a, "next_cov": p, "smoothed_state": None,
             "smoothed_cov": None}
    if smoother == "rts":
        smoothed, smoothed_cov = filtered.copy(), filtered_cov.copy()
        for t in range(t_count - 2, -1, -1):
            c = filtered_cov[t] @ f.T @ linalg.pinvh(predicted_cov[t + 1])
            smoothed[t] = filtered[t] + c @ (smoothed[t + 1] - predicted[t + 1])
            smoothed_cov[t] = filtered_cov[t] + c @ (smoothed_cov[t + 1] - predicted_cov[t + 1]) @ c.T
        found["smoothed_state"], found["smoothed_cov"] = smoothed, smoothed_cov
    return Result(found)


# ------------------------------------------------------------------------------------------------ seasonality


def seasonal_decompose(series, period, model):
    """Classical seasonal decomposition (additive or multiplicative): trend, seasonal indices, seasonally adjusted.

    The trend is the centred moving average over one period, the 2 x m average for an even period m (weights 1/(2m) at
    the two ends, 1/m between) and the plain m-term one for an odd m; NaN for the half period at each end. Detrended
    x - T (additive) or x / T (multiplicative); each season's factor is the mean of its detrended values (the
    positions i, i + m, ... from the first observation), normalised to sum to 0 (additive) or to average 1, summing to
    m (multiplicative). Seasonal S repeats the factors; residual x - T - S or x / (T S); seasonally adjusted x - S or
    x / S. statsmodels' seasonal_decompose to rounding.

    conventions:
    - series: the values, oldest first, at least two full periods, NaN refused; positive for 'multiplicative'
    - period: m, the observations in one seasonal cycle (12 for monthly data with a yearly season, 4 quarterly)
    - model: 'additive' (x = T + S + residual) or 'multiplicative' (x = T S residual)
    returns: a dict: trend (NaN at the ends); seasonal (one per observation); factors (m, the first observation's
      season first); resid; adjusted (the seasonally adjusted series)
    source: Hyndman and Athanasopoulos, Forecasting: Principles and Practice (2021), 3.4; Brockwell and Davis,
      Introduction to Time Series and Forecasting (2016), 1.5; statsmodels' seasonal_decompose
    aliases: seasonal decomposition, classical decomposition, seasonal adjustment, seasonal indices, seasonal_decompose, [seasonality]
    example:
        qf.seasonal_decompose(series=[10.0, 12.0, 14.0, 11.0, 10.5, 12.6, 14.4, 11.3, 11.0, 13.1, 15.0, 11.9],
                              period=4,
                              model="additive")  # factors [-1.55625, 0.425, 2.15625, -1.025]
    """
    choice("model", model, DECOMPOSITIONS)
    x = np.asarray(series, dtype=float).ravel()
    m = integer("period", period, 2)
    if not np.all(np.isfinite(x)) or x.size < 2 * m:
        raise ValueError("series must be finite with at least two full periods")
    if model == "multiplicative" and np.any(x <= 0):
        raise ValueError("a multiplicative decomposition needs positive values")
    weights = np.r_[0.5, np.ones(m - 1), 0.5] / m if m % 2 == 0 else np.ones(m) / m
    half = weights.size // 2
    trend = np.full(x.size, np.nan)
    trend[half:x.size - half] = np.convolve(x, weights, mode="valid")
    detrended = x / trend if model == "multiplicative" else x - trend
    factors = np.array([np.nanmean(detrended[i::m]) for i in range(m)])
    factors = factors / factors.mean() if model == "multiplicative" else factors - factors.mean()
    seasonal = np.resize(factors, x.size)
    if model == "multiplicative":
        resid, adjusted = x / (trend * seasonal), x / seasonal
    else:
        resid, adjusted = x - trend - seasonal, x - seasonal
    return Result({"trend": trend, "seasonal": seasonal, "factors": factors, "resid": resid, "adjusted": adjusted})
