"""Regulatory capital: Basel internal-models charge with its multipliers, FRTB sensitivities and liquidity-horizon ES.

Every VaR, ES and capital figure is a loss, positive. A function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np

from ._common import Result, choice
from .conventions import _finite

__all__ = ["frtb_es", "frtb_sbm", "ima_capital"]

CAPITAL_RULES = ("basel", "last", "average")
TRAFFIC_LIGHT = "traffic_light"
PLUS_FACTORS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.50, 0.65, 0.75, 0.85)   # exceptions 0..9 in 250 days; 10+: 1.00
SCENARIOS = ("low", "medium", "high", "max")
ALTERNATIVES = ("if_negative", "always")


def ima_capital(*, var, stressed_var, multiplier, stressed_multiplier, exceptions, window, rule):
    """Basel internal-models (IMA) capital charge: VaR and stressed VaR, their multipliers, the 60-day averaging rule.

    'basel' (Basel 2.5): capital = max(VaR_{t-1}, m_c VaR_avg) + max(sVaR_{t-1}, m_s sVaR_avg), the averages over
    the last `window` days (60) and the last values the latest; 'last' m_c VaR_{t-1} + m_s sVaR_{t-1}; 'average' m_c
    VaR_avg + m_s sVaR_avg. Each VaR a 10-day 99% loss, positive (a 1-day VaR times sqrt(10) under the square-root
    rule); without stressed VaR (Basel II's 1996 rule) the second term is 0. The multipliers are at least 3: a number,
    or 'traffic_light', 3 plus the backtest's plus factor for the exceptions of the last 250 days, 0 in the green
    zone (0 to 4), 0.40, 0.50, 0.65, 0.75, 0.85 for 5 to 9 (yellow), 1 from 10 (red); qf.var_backtest counts them.
    conventions:
        var: the VaR series, oldest first (or one number), losses positive
        stressed_var: the stressed VaR series alike, or None
        multiplier, stressed_multiplier: m_c and m_s, each a number or 'traffic_light'
        exceptions: the backtesting exceptions in the last 250 days ('traffic_light'), else None
        window: the days averaged, the last `window` values (60 in Basel)
        rule: 'basel', 'last' or 'average'
    returns: a dict: capital; var_component and svar_component (the two terms); multiplier and stressed_multiplier
        (as applied); zone ('green', 'yellow' or 'red', or None without exceptions)
    source: Basel Committee, Revisions to the Basel II market risk framework (2009), paragraph 718(Lxxvi); Basel
        Committee, Supervisory framework for the use of backtesting (1996), Table 2; Hull, Risk Management and
        Financial Institutions, "Basel 2.5"
    aliases: Basel capital charge, IMA capital charge, internal models approach, stressed VaR, Basel 2.5, Basel multiplier, traffic light approach, regulatory capital charge
    example:
        import qf
        v, s = [1.9, 2.1, 2.0, 2.2], [4.8, 5.1, 5.0, 4.9]
        qf.ima_capital(var=v, stressed_var=s, multiplier=3.0, stressed_multiplier=3.0, exceptions=None, window=60, rule="basel")
    """
    choice("rule", rule, CAPITAL_RULES)
    days = int(window) if not isinstance(window, (bool, np.bool_)) and float(window) == int(window) else 0
    if days < 1:
        raise ValueError("window must be a positive int (60 days in Basel)")
    if exceptions is not None:
        if isinstance(exceptions, (bool, np.bool_)) or int(exceptions) != exceptions or exceptions < 0:
            raise ValueError("exceptions must be a non-negative int, or None")
        exceptions = int(exceptions)
        zone = "green" if exceptions <= 4 else "yellow" if exceptions <= 9 else "red"
        plus = PLUS_FACTORS[exceptions] if exceptions < 10 else 1.0
    else:
        zone = plus = None

    def factor(name, value):
        if isinstance(value, str):
            choice(name, value, (TRAFFIC_LIGHT,))
            if plus is None:
                raise ValueError(f"{name} 'traffic_light' needs the exceptions count")
            return 3.0 + plus
        if not float(value) >= 3.0:
            raise ValueError(f"{name} must be at least 3 (Basel's floor), or 'traffic_light'")
        return float(value)

    def term(series, m):
        values = np.atleast_1d(np.asarray(series, dtype=float))
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("a VaR series must hold finite losses, positive")
        last, average = values[-1], values[-days:].mean()
        return float(max(last, m * average) if rule == "basel" else m * last if rule == "last" else m * average)

    m_c = factor("multiplier", multiplier)
    m_s = factor("stressed_multiplier", stressed_multiplier) if stressed_var is not None else None
    first = term(var, m_c)
    second = term(stressed_var, m_s) if stressed_var is not None else 0.0
    return Result({"capital": first + second, "var_component": first, "svar_component": second,
                   "multiplier": m_c, "stressed_multiplier": m_s, "zone": zone})


def _scenario(corr, scenario):
    """MAR21.6: 'high' 1.25 rho capped at 1, 'low' max(2 rho - 1, 0.75 rho), 'medium' as given."""
    if scenario == "high":
        return np.minimum(1.25 * corr, 1.0)
    if scenario == "low":
        return np.maximum(2.0 * corr - 1.0, 0.75 * corr)
    return corr


def _sbm(ws, labels, groups, rho, gamma, scenario, alternative):
    """(the charge, K_b, S_b) under one correlation scenario."""
    r, g = _scenario(rho, scenario), _scenario(gamma, scenario)
    kb, sb = np.empty(len(labels)), np.empty(len(labels))
    for b, members in enumerate(groups):
        w = ws[members]
        inner = r[np.ix_(members, members)].copy()
        np.fill_diagonal(inner, 1.0)
        kb[b] = np.sqrt(max(0.0, float(w @ inner @ w)))
        sb[b] = w.sum()
    cross = g.copy()
    np.fill_diagonal(cross, 0.0)
    capped = np.clip(sb, -kb, kb)
    total = float(kb @ kb + sb @ cross @ sb)
    if alternative == "always" or total < 0:
        sb = capped
        total = float(kb @ kb + sb @ cross @ sb)
    return float(np.sqrt(max(total, 0.0))), kb, sb


def _square(name, value, size):
    m = _finite(name, value)
    if m.ndim == 0:
        m = np.full((size, size), float(m))
    if m.shape != (size, size) or not np.allclose(m, m.T) or np.any(np.abs(m) > 1):
        raise ValueError(f"{name} must be one number or a symmetric {size} x {size} matrix of correlations")
    return m.astype(float)


def frtb_sbm(*, sensitivities, risk_weights, buckets, rho, gamma, scenario, alternative):
    """FRTB SBM charge (MAR21): weighted sensitivities netted in buckets and across buckets at each correlation scenario.

    Each weighted sensitivity WS_k = RW_k s_k; within bucket b, K_b = sqrt(max(0, sum_k WS_k^2 + sum_{k != l} rho_kl
    WS_k WS_l)); across buckets, charge = sqrt(sum_b K_b^2 + sum_{b != c} gamma_bc S_b S_c) with S_b = sum_k WS_k in
    b, and where that sum is negative the alternative S_b = max(min(sum_k WS_k, K_b), -K_b). The scenarios scale
    every rho and gamma: 'medium' as given, 'high' 1.25 rho capped at 1, 'low' max(2 rho - 1, 0.75 rho). The SBM
    charge is the largest scenario after summing delta, vega and curvature over the risk classes per scenario
    (MAR21.7): sum `scenarios`' entries across classes before taking the largest. Curvature is not this formula.
    conventions:
        sensitivities: s_k, the net sensitivity per risk factor (currency per unit move of the factor)
        risk_weights: RW_k per factor (one number for all), as fractions (0.015, not 1.5%)
        buckets: each factor's bucket label (numbers or strings)
        rho: the within-bucket correlation, one number or a factors x factors matrix (pairs across buckets unused)
        gamma: the across-bucket correlation, one number or a matrix over the buckets in sorted label order
        scenario: 'low', 'medium', 'high' or 'max' (the largest of the three for this class alone)
        alternative: 'if_negative' (Basel's: the capped S_b only where the sum is negative) or 'always'
    returns: a dict: charge; scenarios (the charge under 'low', 'medium' and 'high'); kb and sb (per bucket, under
        the scenario reported); buckets (their labels, sorted)
    source: Basel Committee, Minimum capital requirements for market risk (2019), MAR21.4 (aggregation), MAR21.6
        (the correlation scenarios) and MAR21.7
    aliases: FRTB, FRTB SBM, FRTB standardised approach
    example:
        import qf
        # Charge 44.34524 (the high scenario).
        qf.frtb_sbm(sensitivities=[100.0, -40.0, 60.0], risk_weights=[0.3, 0.3, 0.5], buckets=[1, 1, 2], rho=0.2, gamma=0.15, scenario="max", alternative="if_negative")
    """
    choice("scenario", scenario, SCENARIOS)
    choice("alternative", alternative, ALTERNATIVES)
    s = _finite("sensitivities", sensitivities).ravel()
    rw = np.broadcast_to(_finite("risk_weights", risk_weights), s.shape).astype(float)
    if np.any(rw < 0):
        raise ValueError("risk_weights must not be negative")
    labels_of = list(np.asarray(buckets).ravel())
    if len(labels_of) != s.size:
        raise ValueError("buckets must hold one label per sensitivity")
    labels = sorted(set(labels_of))
    groups = [np.array([k for k, label in enumerate(labels_of) if label == b]) for b in labels]
    r = _square("rho", rho, s.size)
    g = _square("gamma", gamma, len(labels))
    ws = rw * s
    found = {name: _sbm(ws, labels, groups, r, g, name, alternative) for name in SCENARIOS[:3]}
    pick = max(SCENARIOS[:3], key=lambda name: found[name][0]) if scenario == "max" else scenario
    charge, kb, sb = found[pick]
    return Result({"charge": charge, "scenarios": {name: found[name][0] for name in SCENARIOS[:3]}, "kb": kb, "sb": sb,
                   "buckets": labels})


def frtb_es(*, es, horizons, base_horizon, full_current, reduced_current):
    """FRTB liquidity-horizon ES (internal models), scaled by the reduced-set stress ratio floored at 1.

    ES = sqrt(sum_j (ES_T(P, j) sqrt((LH_j - LH_{j-1}) / T))^2), LH_0 = 0: ES_T(P, 1) is the base-horizon ES of every
    risk factor's shock, ES_T(P, j) that of the factors whose liquidity horizon is at least LH_j alone (the others
    held). Basel's horizons are 10, 20, 40, 60 and 120 days on a base T of 10, so the first term is ES_T(P) itself.
    The stress calibration scales the result (the reduced set's stressed ES) by the current full-set ES over the
    current reduced-set ES, that ratio floored at 1.
    conventions:
        es: ES_T(P, j) per liquidity horizon bucket, the first over all risk factors, losses positive
        horizons: LH_j in days, increasing (Basel's [10, 20, 40, 60, 120])
        base_horizon: T in days (10 in Basel)
        full_current, reduced_current: ES_F,C and ES_R,C, the current ES of the full and of the reduced set of
            risk factors, or both None for no stress scaling
    returns: a dict: es (the liquidity-adjusted ES, scaled where asked); unscaled; ratio (the stress scaling applied,
        1 without)
    source: Basel Committee, Minimum capital requirements for market risk (2019), MAR33.4 (the liquidity horizons)
        and MAR33.5-33.6 (the reduced set, the ratio floored at 1); Hull, Risk Management and Financial Institutions,
        "Fundamental review of the trading book"
    aliases: FRTB ES, liquidity horizon ES, FRTB IMA, FRTB internal models
    example:
        import qf
        # ES 16.79286; ratio 1.0.
        qf.frtb_es(es=[12.0, 8.0, 5.0, 3.0, 1.0], horizons=[10, 20, 40, 60, 120], base_horizon=10, full_current=None, reduced_current=None)
    """
    values = _finite("es", es).ravel()
    lh = _finite("horizons", horizons).ravel()
    t = float(base_horizon)
    if lh.shape != values.shape or np.any(np.diff(lh) <= 0) or lh[0] <= 0 or not t > 0:
        raise ValueError("horizons must be increasing positive days, one per ES, and base_horizon positive")
    if np.any(values < 0):
        raise ValueError("es must be losses, positive")
    steps = np.diff(np.r_[0.0, lh])
    unscaled = float(np.sqrt(np.sum((values * np.sqrt(steps / t)) ** 2)))
    if (full_current is None) != (reduced_current is None):
        raise ValueError("full_current and reduced_current: both numbers, or both None")
    ratio = 1.0
    if full_current is not None:
        full, reduced = float(full_current), float(reduced_current)
        if not (full >= 0 and reduced > 0):
            raise ValueError("full_current must be a loss and reduced_current a positive loss")
        ratio = max(1.0, full / reduced)
    return Result({"es": unscaled * ratio, "unscaled": unscaled, "ratio": ratio})
