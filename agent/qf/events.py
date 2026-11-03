"""Event studies: abnormal returns, CAR, CAAR, Patell, BMP and Corrado tests.

`import qf`. Returns come as (N, D) per event, aligned so column j is event_time[j], whole event days; a window
is (first, last), inclusive. A function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

from ._common import Result, choice, out

__all__ = ["corrado_rank_test", "market_model_event_study"]

RANK_STATISTICS = ("day", "window")
RANK_VERSIONS = ("corrado", "corrado_zivney")


def market_model_event_study(asset_returns, market_returns, event_time, estimation_window, event_window):
    """Market-model event study: abnormal returns, CAR, CAAR, and the Patell and BMP statistics.

    For event i, OLS of R on R_m over the estimation window's L_i days: alpha_i, beta_i, s_i^2 = SSR / (L_i - 2).
    AR_it = R_it - alpha_i - beta_i R_mt over the K event-window days; CAR_i = sum_t AR_it; AAR_t = mean_i AR_it;
    CAAR = sum_t AAR_t. Patell (1976): SAR_it = AR_it / (s_i sqrt(C_it)), C_it = 1 + 1/L_i + (R_mt - mean R_m)^2 /
    sum (R_m - mean R_m)^2 over the estimation window; SCAR_i = sum_t SAR_it / sqrt(K);
    patell_z = sum_i SCAR_i / sqrt(sum_i (L_i - 2)/(L_i - 4)); bmp_t = mean(SCAR) / (std(SCAR, ddof=1) / sqrt(N))
    (Boehmer, Musumeci and Poulsen; numpy.nan with one event).

    Two groups' abnormal returns compared: Welch's t, scipy.stats.ttest_ind(a, b, equal_var=False); Mann-Whitney's U,
    scipy.stats.mannwhitneyu(a, b, alternative='two-sided').

    Standardized unexpected earnings: SUE = (EPS - its expected value) / the std of the past forecast errors (or the
    price, as the task scales it).

    conventions:
    - asset_returns, market_returns: (N, D) per event, aligned so column j is event_time[j] (D days; 1-d for one event)
    - estimation_window, event_window: (first, last) event days, inclusive, the estimation window wholly before
    - a NaN in the estimation window drops that day (L_i counts the rest, at least 5); in the event window it is
      refused: drop such an event first where the task skips it
    - the window's SCAR sums the daily standardized returns (Patell); Corrado's rank test is corrado_rank_test
    returns: a dict: alpha, beta, sigma (the residual sd) and n_estimation, one per event; event_days;
      ar and sar (events x event days); car and scar (one per event); aar (one per event day);
      caar, patell_z and bmp_t (floats); returns in the inputs' units; not the estimation residuals (a rank
      test needs them: refit each event by ols)
    source: Patell (1976), J. Accounting Research 14(2); Boehmer, Musumeci and Poulsen (1991), J. Financial Economics 30
    aliases: event study, market model, abnormal returns, [CAR], CAAR, Patell test, BMP test, standardized abnormal return, [Patell BMP test]
    example:
        qf.market_model_event_study(asset_returns=[[0.01, -0.02, 0.01, 0.0, -0.01, 0.03, 0.02]],
            market_returns=[[0.01, -0.01, 0.01, 0.0, -0.01, 0.01, 0.0]], event_time=[-6, -5, -4, -3, -2, 0, 1],
            estimation_window=(-6, -2), event_window=(0, 1))
    """
    r = np.atleast_2d(np.asarray(asset_returns, dtype=float))
    rm = np.atleast_2d(np.asarray(market_returns, dtype=float))
    days = np.asarray(event_time)
    if r.ndim != 2 or rm.shape != r.shape or days.shape != (r.shape[1],):
        raise ValueError("asset_returns and market_returns must be (N, D) with event_time's D days")
    if not np.issubdtype(days.dtype, np.integer) or np.any(np.diff(days) <= 0):
        raise ValueError("event_time must be increasing whole event days")
    (e0, e1), (w0, w1) = estimation_window, event_window
    if not (e0 <= e1 < w0 <= w1):
        raise ValueError("estimation_window and event_window must be (first, last) with the estimation window first")
    estimation, event = (days >= e0) & (days <= e1), (days >= w0) & (days <= w1)
    if not event.any():
        raise ValueError("event_window holds none of event_time's days")
    n, k = r.shape[0], int(event.sum())
    alpha, beta, sigma, length = np.empty(n), np.empty(n), np.empty(n), np.empty(n, dtype=int)
    ar, sar = np.empty((n, k)), np.empty((n, k))
    for i in range(n):
        keep = estimation & np.isfinite(r[i]) & np.isfinite(rm[i])
        x, y = rm[i][keep], r[i][keep]
        if x.size < 5:
            raise ValueError(f"event {i} has {x.size} estimation days; at least 5 are needed")
        if not (np.all(np.isfinite(r[i][event])) and np.all(np.isfinite(rm[i][event]))):
            raise ValueError(f"event {i} has a NaN in its event window")
        spread = np.sum((x - x.mean()) ** 2)
        beta[i] = np.sum((x - x.mean()) * (y - y.mean())) / spread
        alpha[i] = y.mean() - beta[i] * x.mean()
        resid = y - alpha[i] - beta[i] * x
        length[i] = x.size
        sigma[i] = np.sqrt(resid @ resid / (x.size - 2))
        ar[i] = r[i][event] - alpha[i] - beta[i] * rm[i][event]
        sar[i] = ar[i] / (sigma[i] * np.sqrt(1.0 + 1.0 / x.size + (rm[i][event] - x.mean()) ** 2 / spread))
    car, aar = ar.sum(axis=1), ar.mean(axis=0)
    scar = sar.sum(axis=1) / np.sqrt(k)
    patell = float(scar.sum() / np.sqrt(np.sum((length - 2.0) / (length - 4.0))))
    bmp = float(scar.mean() / (scar.std(ddof=1) / np.sqrt(n))) if n > 1 else float("nan")
    return Result({"alpha": alpha, "beta": beta, "sigma": sigma, "n_estimation": length, "event_days": days[event],
                   "ar": ar, "car": car, "aar": aar, "caar": float(aar.sum()), "sar": sar, "scar": scar,
                   "patell_z": patell, "bmp_t": bmp})


def corrado_rank_test(abnormal_returns, event_time, estimation_window, event_window, statistic, version):
    """Corrado's nonparametric rank test of event-day abnormal returns, the version named.

    Each event's abnormal returns over the estimation and event windows together (its M_i days with a return) are
    ranked, ties by their average rank. 'corrado' (1989): d_it = K_it - (M_i + 1)/2 and x_t = the mean of d_it over
    the N_t events with a return that day; 'corrado_zivney' (1992): U_it = K_it / (M_i + 1) and x_t = sum (U_it -
    1/2) / sqrt(N_t). S = sqrt(mean of x_t^2 over every day of both windows); 'day' gives x_t / S for each event day,
    'window' sum x_t over the event window / (sqrt(L_W) S) (Cowan 1992). Near standard normal under the null; the
    p-value two-sided. With no return missing the versions agree.

    conventions:
    - abnormal_returns: (N, D) per event, aligned so column j is event_time[j], over the estimation window and the
      event window (the market model's residuals and abnormal returns: market_model_event_study keeps the latter
      only); numpy.nan where a return is missing
    - event_time: D increasing whole event days; estimation_window, event_window: (first, last), inclusive, the
      estimation window wholly before
    - statistic: 'day' or 'window'; version: 'corrado' or 'corrado_zivney'
    returns: a dict: statistic and p_value (one per event day for 'day', floats for 'window'); event_days; x (x_t
      over both windows' days); s (S)
    source: Corrado (1989), JFE 23; Corrado and Zivney (1992), JFQA 27(3); Campbell, Lo and MacKinlay (1997), 4.6
    aliases: Corrado rank test, rank test, nonparametric event study test, Corrado-Zivney, Cowan rank test
    example:
        qf.corrado_rank_test(abnormal_returns=[[0.01, -0.02, 0.005, -0.004, 0.012, 0.03], [-0.006, 0.011, -0.002, 0.007, -0.01, 0.024]],
            event_time=[-5, -4, -3, -2, -1, 0], estimation_window=(-5, -1), event_window=(0, 0), statistic="day",
            version="corrado")
    """
    choice("statistic", statistic, RANK_STATISTICS)
    choice("version", version, RANK_VERSIONS)
    a = np.atleast_2d(np.asarray(abnormal_returns, dtype=float))
    days = np.asarray(event_time)
    if a.ndim != 2 or days.shape != (a.shape[1],) or not np.issubdtype(days.dtype, np.integer) or np.any(
            np.diff(days) <= 0):
        raise ValueError("abnormal_returns must be (N, D) with event_time's D increasing whole event days")
    (e0, e1), (w0, w1) = estimation_window, event_window
    if not (e0 <= e1 < w0 <= w1):
        raise ValueError("estimation_window and event_window must be (first, last) with the estimation window first")
    used = ((days >= e0) & (days <= e1)) | ((days >= w0) & (days <= w1))
    event = (days[used] >= w0) & (days[used] <= w1)
    r = a[:, used]
    if not event.any():
        raise ValueError("event_window holds none of event_time's days")
    present = np.isfinite(r)
    counts = present.sum(axis=1)
    if np.any(counts < 2):
        raise ValueError("every event needs returns on at least two of the windows' days")
    ranks = np.full(r.shape, np.nan)
    for i in range(r.shape[0]):
        ranks[i, present[i]] = rankdata(r[i, present[i]], method="average")
    with np.errstate(invalid="ignore"):
        if version == "corrado":
            x = np.nanmean(ranks - (counts[:, None] + 1) / 2.0, axis=0)
        else:
            x = np.nansum(ranks / (counts[:, None] + 1) - 0.5, axis=0) / np.sqrt(present.sum(axis=0))
    if np.any(~np.isfinite(x)):
        raise ValueError("every day of the windows needs a return from at least one event")
    sd = float(np.sqrt(np.mean(x * x)))
    found = x[event] / sd if statistic == "day" else float(x[event].sum() / (np.sqrt(event.sum()) * sd))
    return Result({"statistic": out(found), "p_value": out(2 * norm.sf(np.abs(found))),
                   "event_days": days[used][event], "x": x, "s": sd})
