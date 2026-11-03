"""Commodity and weather: HDD, CDD, Schwartz-Smith, gas storage, CAR temperature, Samuelson effect.

`import qf`. Times in years unless a function names its own unit (a temperature model's days); temperatures in the
unit of the data, its reference temperature in the same one. Several results come back as a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares, linprog

from ._common import Result, choice, out, positive
from .primitives import _guard

__all__ = ["car_temperature", "degree_days", "samuelson_vol_fit", "schwartz_smith", "storage_intrinsic"]

DAILY_MEANS = ("given", "max_min")
INDEXES = ("hdd", "cdd", "cat")
WEATHER_PAYOFFS = ("call", "put", "swap")
PREMIA = ("subtracted", "added")
CAR_FORMS = ("car", "ar")
VOL_FORMS = ("two_factor", "one_factor")


def degree_days(temperatures, daily_mean, reference, index, option, strike, tick, cap, discount_factor, ddof):
    """Heating and cooling degree days (HDD, CDD) or CAT, and a capped weather option's burn price.

    HDD = the sum over the days of max(T_ref - T, 0), CDD = the sum of max(T - T_ref, 0), CAT = the sum of T, T a
    day's mean temperature: a day at the reference adds nothing. The payoff on the index I: 'call' tick max(I - K,
    0), 'put' tick max(K - I, 0), 'swap' tick (I - K), limited to cap. The burn price is D times the mean payoff
    over past years, a row of temperatures a year; std_error D s / sqrt(years). Detrending is the caller's.

    conventions:
    - temperatures: one period's days, or years x days (a row a year's period); under 'max_min' each day a (maximum,
      minimum) pair on the last axis
    - daily_mean: 'given' (the temperatures are the daily means) or 'max_min' ((maximum + minimum) / 2, the CME's)
    - reference: the base temperature in the temperatures' own unit (65 in Fahrenheit and 18 in Celsius are usual)
    - index: 'hdd' (heating degree days), 'cdd' (cooling degree days) or 'cat' (cumulative average temperature)
    - option: 'call', 'put' or 'swap'; strike: K in index units; tick: the money an index unit pays
    - cap: the payoff's limit in money (a swap's either way), or None for no cap
    - discount_factor: D to the payment date; 1.0 for none
    - ddof: 1 or 0 (the payoffs' standard deviation across years)
    returns: a dict: index and payoff (one a year, or a float for one period); price (D times the mean payoff);
      std_error (nan for one period)
    source: Hull, Options, Futures, and Other Derivatives, "Weather derivatives"; Jewson and Brix, Weather Derivative
      Valuation (2005) (burn analysis)
    aliases: heating degree days, cooling degree days, HDD, CDD, degree day option, weather derivative
    example:
        qf.degree_days(temperatures=[[60.0, 50.0, 66.0, 41.0], [62.0, 55.0, 70.0, 48.0]], daily_mean="given",
                       reference=65.0, index="hdd", option="call", strike=30.0, tick=20.0, cap=None,
                       discount_factor=0.99, ddof=1)
        # result: {'index': [44.0, 30.0], 'payoff': [280.0, 0.0], 'price': 138.6}
    """
    choice("daily_mean", daily_mean, DAILY_MEANS)
    choice("index", index, INDEXES)
    choice("option", option, WEATHER_PAYOFFS)
    temps = np.asarray(temperatures, dtype=float)
    if daily_mean == "max_min":
        if temps.ndim < 2 or temps.shape[-1] != 2:
            raise ValueError("under 'max_min' each day is a (maximum, minimum) pair on the last axis")
        temps = temps.mean(axis=-1)
    if temps.ndim not in (1, 2) or temps.shape[-1] < 1 or not np.all(np.isfinite(temps)):
        raise ValueError("temperatures must be one period's days or years x days, finite")
    base = float(reference)
    if index == "hdd":
        found = np.maximum(base - temps, 0.0).sum(axis=-1)
    elif index == "cdd":
        found = np.maximum(temps - base, 0.0).sum(axis=-1)
    else:
        found = temps.sum(axis=-1)
    k, unit = float(strike), float(positive("tick", tick))
    pay = unit * {"call": np.maximum(found - k, 0.0), "put": np.maximum(k - found, 0.0), "swap": found - k}[option]
    if cap is not None:
        limit = float(positive("cap", cap))
        pay = np.clip(pay, -limit, limit)
    if isinstance(ddof, (bool, np.bool_)) or ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, not {ddof!r}")
    d = float(positive("discount_factor", discount_factor))
    years = np.atleast_1d(pay)
    error = d * years.std(ddof=int(ddof)) / np.sqrt(years.size) if years.size > 1 else float("nan")
    return Result({"index": out(found), "payoff": out(pay), "price": float(d * years.mean()),
                   "std_error": float(error)})


@_guard(time=('maturities',))
def schwartz_smith(chi, xi, maturities, kappa, sigma_chi, sigma_xi, rho, mu_xi, lambda_chi, lambda_xi, premia):
    """Schwartz-Smith commodity futures prices, their log volatility and the Samuelson effect.

    ln S = chi + xi: chi the short-term deviation, an OU dchi = -kappa chi dt + sigma_chi dW_chi reverting to zero;
    xi the long-term level, a Brownian motion dxi = mu_xi dt + sigma_xi dW_xi; corr(dW_chi, dW_xi) = rho. Under the
    pricing measure the drifts carry the risk premia: -kappa chi - lambda_chi and mu_xi - lambda_xi ('subtracted',
    Schwartz and Smith's own). The futures price for maturity T is ln F = e^{-kappa T} chi + xi + A(T), A(T) = mu* T -
    (1 - e^{-kappa T}) lambda_chi / kappa + (1/2) [(1 - e^{-2 kappa T}) sigma_chi^2 / (2 kappa) + sigma_xi^2 T + 2 (1 -
    e^{-kappa T}) rho sigma_chi sigma_xi / kappa], mu* = mu_xi - lambda_xi. The futures' log volatility with tau to
    run is sqrt(sigma_chi^2 e^{-2 kappa tau} + sigma_xi^2 + 2 rho sigma_chi sigma_xi e^{-kappa tau}): high near
    expiry when rho >= 0, tending to sigma_xi far out (the Samuelson effect; negative correlation can give a
    nonmonotone curve). As kappa grows, F tends to the one-factor forward
    e^{xi + (mu* + sigma_xi^2 / 2) T}.

    conventions:
    - chi, xi: today's short-term deviation and long-term level of ln S (from a Kalman filter or a fit)
    - maturities: the futures' maturities in years from today
    - kappa, sigma_chi, sigma_xi, rho, mu_xi: the real-world parameters, annual; lambda_chi, lambda_xi: the risk
      premia
    - premia: 'subtracted' (pricing drifts -kappa chi - lambda_chi and mu_xi - lambda_xi, the paper's) or 'added'
      (the same with + lambda, a source writing the premia with the other sign)
    returns: a dict: forwards (the futures price a maturity); vol (the futures' log volatility a maturity, annual);
      variance (the variance of ln S_T a maturity); spot (e^{chi + xi})
    source: Schwartz and Smith (2000), "Short-term variations and long-term dynamics in commodity prices",
      Management Science 46(7), equations (9) and (10)
    aliases: Schwartz-Smith, Schwartz and Smith model
    example:
        qf.schwartz_smith(chi=0.1, xi=4.0, maturities=[0.25, 0.5, 1.0, 2.0], kappa=1.5, sigma_chi=0.3, sigma_xi=0.15,
                          rho=0.3, mu_xi=0.0, lambda_chi=0.1, lambda_xi=0.02,
                          premia="subtracted")
        # result: {'forwards': [57.76730, 55.92927, 53.67872, 51.81325]}
    """
    choice("premia", premia, PREMIA)
    t = np.asarray(maturities, dtype=float)
    if np.any(~(t >= 0)):
        raise ValueError("maturities must not be negative")
    k = float(positive("kappa", kappa))
    s_chi, s_xi, corr = float(sigma_chi), float(sigma_xi), float(rho)
    if s_chi < 0 or s_xi < 0 or not -1.0 <= corr <= 1.0:
        raise ValueError("sigma_chi and sigma_xi must not be negative and rho must lie in [-1, 1]")
    sign = -1.0 if premia == "subtracted" else 1.0
    drift = float(mu_xi) + sign * float(lambda_xi)
    decay = -np.expm1(-k * t)                                       # 1 - e^{-kappa T}
    variance = -np.expm1(-2.0 * k * t) * s_chi ** 2 / (2.0 * k) + s_xi ** 2 * t + 2.0 * decay * corr * s_chi * s_xi / k
    shift = drift * t + sign * decay * float(lambda_chi) / k + 0.5 * variance
    forwards = np.exp(np.exp(-k * t) * float(chi) + float(xi) + shift)
    vol = np.sqrt(s_chi ** 2 * np.exp(-2.0 * k * t) + s_xi ** 2 + 2.0 * corr * s_chi * s_xi * np.exp(-k * t))
    return Result({"forwards": out(forwards), "vol": out(vol), "variance": out(variance),
                   "spot": float(np.exp(float(chi) + float(xi)))})


def storage_intrinsic(forwards, discount_factors, capacity, injection_rate, withdrawal_rate, injection_cost,
                      withdrawal_cost, initial, final):
    """Gas storage intrinsic value: the injection and withdrawal plan of greatest value on the forward strip.

    For periods t with forward F_t and discount factor D_t: choose injections i_t and withdrawals w_t to maximise sum_t
    D_t [(F_t - c_w) w_t - (F_t + c_i) i_t], the inventory after each period I_t = initial + sum_{s <= t} (i_s - w_s)
    held between 0 and the capacity, each period's volumes within the rates, and the last inventory equal to `final`
    (scipy.optimize.linprog, HiGHS). The intrinsic value locks in the curve's spreads; the extrinsic value (the option
    to re-optimise as the curve moves) is not in it. With zero initial and final inventory, a flat curve, equal
    discounts and nonnegative costs give zero value.

    conventions:
    - forwards: the forward price of each delivery period, in money a volume unit; discount_factors: D_t for each
      period's cash flow
    - capacity: the most the storage holds, in volume units
    - injection_rate, withdrawal_rate: the most injected or withdrawn in a period, a number or one a period
    - injection_cost, withdrawal_cost: the money a volume unit injected or withdrawn, beside the gas's price
    - initial: the inventory at the start; final: the inventory required after the last period
    returns: a dict: value (the intrinsic value, discounted); injections, withdrawals and inventory (one a period, the
      inventory at each period's end)
    source: Eydeland and Wolyniec, Energy and Power Risk Management (2003), ch. 6 (storage); Clewlow and Strickland,
      Energy Derivatives (2000)
    aliases: gas storage, storage intrinsic value, natural gas storage
    example:
        qf.storage_intrinsic(forwards=[3.0, 2.8, 3.4, 4.1], discount_factors=[0.99, 0.98, 0.97, 0.96], capacity=100.0,
                             injection_rate=50.0, withdrawal_rate=60.0, injection_cost=0.05, withdrawal_cost=0.05,
                             initial=0.0, final=0.0)
        # result: {'value': 72.635, 'inventory': [50.0, 100.0, 60.0, 0.0]}
    """
    f = np.asarray(forwards, dtype=float).ravel()
    n = f.size
    d = positive("discount_factors", discount_factors).ravel()
    if n < 1 or d.shape != f.shape or not np.all(np.isfinite(f)):
        raise ValueError("forwards and discount_factors need one finite value a period")
    room = float(positive("capacity", capacity))
    rate_in = np.broadcast_to(np.asarray(injection_rate, dtype=float), (n,))
    rate_out = np.broadcast_to(np.asarray(withdrawal_rate, dtype=float), (n,))
    start, end = float(initial), float(final)
    if np.any(~(rate_in >= 0)) or np.any(~(rate_out >= 0)) or not (0 <= start <= room and 0 <= end <= room):
        raise ValueError("the rates must not be negative and initial and final must lie within the capacity")
    cost = np.r_[d * (f + float(injection_cost)), -d * (f - float(withdrawal_cost))]
    lower = np.tril(np.ones((n, n)))
    upper_rows = np.hstack([lower, -lower])                    # I_t - initial
    result = linprog(cost, A_ub=np.vstack([upper_rows, -upper_rows]), b_ub=np.r_[np.full(n, room - start),
                                                                                  np.full(n, start)],
                     A_eq=np.r_[np.ones(n), -np.ones(n)][None, :], b_eq=[end - start],
                     bounds=list(zip(np.zeros(2 * n), np.r_[rate_in, rate_out])), method="highs")
    if result.status != 0:
        raise ValueError(f"no schedule meets the constraints ({result.message})")
    inject, withdraw = result.x[:n], result.x[n:]
    return Result({"value": float(-result.fun) + 0.0, "injections": inject, "withdrawals": withdraw,
                   "inventory": start + np.cumsum(inject - withdraw)})


def car_temperature(coefficients, form, state, seasonal_mean, horizons, sigma):
    """Temperature forecast and its variance under Benth's CAR(p) autoregression.

    T(t) = Lambda(t) + X_1(t), Lambda the seasonal mean, X the state of dX = A X dt + e_p sigma dB, A the companion
    matrix (ones above the diagonal, last row -alpha_p, ..., -alpha_1). Ahead by h: E[T(t + h)] = Lambda(t + h) + e_1'
    e^{A h} X(t), the variance sigma^2 int_0^h (e_1' e^{A u} e_p)^2 du (by Van Loan's block exponential). CAR(1) is the
    Ornstein-Uhlenbeck process, mean X e^{-alpha h}, variance sigma^2 (1 - e^{-2 alpha h}) / (2 alpha). form 'ar'
    takes the AR(p) coefficients b of the deseasonalised daily series, x_{t+p} = b_1 x_{t+p-1} + ... + b_p x_t + e,
    and maps them by the Euler step of one time unit: the alphas are the coefficients of Q(1 + y), Q(E) = E^p - b_1
    E^{p-1} - ... - b_p (for p = 3: 3 - b_1, 3 - 2 b_1 - b_2, 1 - b_1 - b_2 - b_3).

    conventions:
    - coefficients: alpha_1 .. alpha_p ('car') or the AR(p) b_1 .. b_p ('ar'), in the time unit of the data (a day)
    - form: 'car' or 'ar'
    - state: X(t), p values: today's deviation from the seasonal mean first, then its time derivatives (from the last
      deviations d: d_t, d_t - d_{t-1}, d_t - 2 d_{t-1} + d_{t-2} for p = 3)
    - seasonal_mean: Lambda at each horizon's date, one value a horizon (the caller's seasonal fit)
    - horizons: the times ahead, in the coefficients' unit; sigma: the noise's volatility in that unit
    returns: a dict: expected and variance (one a horizon); alphas (the CAR coefficients); stationary (True where
      every eigenvalue of A has a negative real part)
    source: Benth, Saltyte Benth and Koekebakker, Stochastic Modelling of Electricity and Related Markets (2008), ch.
      10; Benth and Saltyte Benth (2007), "The volatility of temperature and pricing of weather derivatives",
      Quantitative Finance 7(5); Lopez Cabrera (2010), Weather Risk Management: CAT Bonds and Weather Derivatives,
      equation (4.34) (the corrected AR-to-CAR map); Van Loan (1978)
    aliases: CAR temperature model, CAR(3), Benth temperature model, CAR(p) autoregressive
    example:
        qf.car_temperature(coefficients=[0.95, -0.25, 0.08], form="ar", state=[2.0, 0.5, -0.1],
                           seasonal_mean=[50.0, 51.0, 52.0], horizons=[1.0, 5.0, 10.0],
                           sigma=3.0)
        # result: {'expected': [52.35889, 52.43896, 52.44283], 'alphas': [2.05, 1.35, 0.22]}
    """
    choice("form", form, CAR_FORMS)
    given = np.asarray(coefficients, dtype=float).ravel()
    p = given.size
    if p < 1 or not np.all(np.isfinite(given)):
        raise ValueError("coefficients must be p finite values")
    if form == "ar":
        q = np.polynomial.Polynomial(np.r_[-given[::-1], 1.0])       # Q(E) = E^p - b_1 E^{p-1} - ... - b_p
        alphas = q(np.polynomial.Polynomial([1.0, 1.0])).coef[::-1][1:]      # Q(1 + y), leading 1 dropped
    else:
        alphas = given
    a = np.zeros((p, p))
    a[np.arange(p - 1), np.arange(1, p)] = 1.0
    a[-1] = -alphas[::-1]
    x = np.asarray(state, dtype=float).ravel()
    h = np.asarray(horizons, dtype=float).ravel()
    mean = np.asarray(seasonal_mean, dtype=float).ravel()
    if x.size != p or mean.shape != h.shape or np.any(~(h >= 0)):
        raise ValueError("state needs p values, and seasonal_mean one value a horizon, horizons not negative")
    s = float(sigma)
    if s < 0:
        raise ValueError("sigma must not be negative")
    noise = np.zeros((p, p))
    noise[-1, -1] = 1.0
    expected, variance = np.empty(h.size), np.empty(h.size)
    for i, ahead in enumerate(h):
        # A short block exponential avoids growing -A over a long stationary horizon.
        squarings = max(0, int(np.ceil(np.log2(max(1.0, np.linalg.norm(a, 1) * ahead)))))
        step = ahead / 2.0 ** squarings
        block = expm(np.block([[-a, noise], [np.zeros((p, p)), a.T]]) * step)
        grown = block[p:, p:].T
        covariance = grown @ block[:p, p:]
        for _ in range(squarings):
            covariance += grown @ covariance @ grown.T
            grown = grown @ grown
        expected[i] = mean[i] + grown[0] @ x
        variance[i] = s * s * covariance[0, 0]
    return Result({"expected": expected, "variance": np.maximum(variance, 0.0), "alphas": alphas,
                   "stationary": bool(np.all(np.linalg.eigvals(a).real < 0))})


@_guard(vol=('vols',), time=('maturities',))
def samuelson_vol_fit(maturities, vols, form):
    """Samuelson effect: commodity futures vols decaying with time to expiry, fitted.

    The futures' volatility rises as expiry nears (the Samuelson effect). 'two_factor': sigma(tau) = sigma_inf sqrt(1 +
    a e^{-2 kappa tau}), the long-run vol sigma_inf, a the short-term factor's weight (the two-factor model with
    uncorrelated factors: a = sigma_chi^2 / sigma_xi^2); 'one_factor': sigma(tau) = sigma e^{-kappa tau} (Schwartz's
    one-factor model). Least squares on the vols (scipy.optimize.least_squares, from several starts, the best kept).
    The vols are the caller's, annualised from contract closes (qf.close_to_close_vol); grouping by time to maturity
    and excluding roll days belong to the caller.

    conventions:
    - maturities: each vol's time to maturity in years; vols: the annualised vols there (decimals)
    - form: 'two_factor' (sigma_inf sqrt(1 + a e^{-2 kappa tau})) or 'one_factor' (sigma e^{-kappa tau});
      a and kappa are nonnegative, allowing a flat term structure
    returns: a dict: sigma (sigma_inf, or the one-factor sigma), a (nan under 'one_factor'), kappa, fitted (the model's
      vols at the maturities), rmse
    source: Clewlow and Strickland, Energy Derivatives: Pricing and Risk Management (2000), ch. 6; Samuelson (1965);
      Schwartz (1997), Journal of Finance 52(3)
    aliases: Samuelson effect
    example:
        qf.samuelson_vol_fit(maturities=[0.1, 0.25, 0.5, 1.0, 2.0], vols=[0.48, 0.41, 0.34, 0.28, 0.25],
                             form="two_factor")
        # result: {'sigma': 0.251312, 'a': 3.47438, 'kappa': 1.42303, 'rmse': 0.00299498}
    """
    choice("form", form, VOL_FORMS)
    tau = np.asarray(maturities, dtype=float).ravel()
    v = np.asarray(vols, dtype=float).ravel()
    need = 3 if form == "two_factor" else 2
    if tau.shape != v.shape or tau.size < need or np.any(~(tau >= 0)) or np.any(~(v > 0)):
        raise ValueError(f"maturities and vols need {need} or more points, maturities not negative, vols positive")

    def model(x):
        if form == "two_factor":
            return x[0] * np.sqrt(1.0 + x[1] * np.exp(-2.0 * x[2] * tau))
        return x[0] * np.exp(-x[1] * tau)

    scale = max(float(tau.mean()), 1e-6)
    longest, shortest = v[np.argmax(tau)], v[np.argmin(tau)]
    best = None
    for speed in (0.25 / scale, 1.0 / scale, 4.0 / scale):
        if form == "two_factor":
            start = [longest, max((shortest / longest) ** 2 - 1.0, 0.0), speed]
            bounds = ([1e-12, 0.0, 0.0], [np.inf, np.inf, np.inf])
        else:
            start, bounds = [shortest, speed], ([1e-12, 0.0], [np.inf, np.inf])
        fit = least_squares(lambda x: model(x) - v, start, bounds=bounds, x_scale="jac")
        if best is None or fit.cost < best.cost:
            best = fit
    x = best.x
    fitted = model(x)
    sigma, a, kappa = (x[0], x[1], x[2]) if form == "two_factor" else (x[0], float("nan"), x[1])
    return Result({"sigma": float(sigma), "a": float(a), "kappa": float(kappa), "fitted": fitted,
                   "rmse": float(np.sqrt(np.mean((fitted - v) ** 2)))})
