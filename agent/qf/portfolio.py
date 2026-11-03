"""Portfolio weights: Markowitz (constrained), Black-Litterman, inverse-vol, ERC, Euler contributions, Barra risk, costs.

`import qf`. A covariance's horizon is the results' (an annual covariance gives annual figures). A scalar result
is a float; a function with several results returns a dict with named keys.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize

from ._common import Result, choice, out, positive
from .conventions import QUANTILE_METHODS, _finite, _tail
from .risk import ES_RULES
from .primitives import _guard

__all__ = ["black_litterman", "cost_aware_rebalance", "equal_risk_contribution_weights", "equilibrium_returns",
           "euler_es_contributions", "euler_vol_contributions", "factor_risk", "inverse_volatility_weights",
           "marginal_vol_contributions", "mean_variance"]

OBJECTIVES = ("utility", "min_variance", "target_return", "max_sharpe")
PENALTIES = ("half", "whole")
BUDGETS = ("full", "zero", "none")
COST_FORMS = ("linear", "quadratic")


def equilibrium_returns(covariance, market_weights, risk_aversion):
    """Implied equilibrium excess returns by reverse optimization: pi = delta Sigma w.

    pi = delta Sigma w_mkt: the expected excess returns at which the market portfolio is mean-variance optimal for a
    risk aversion delta, often (E[r_m] - r_f) / sigma_m^2.

    conventions:
    - covariance: of excess returns, on the horizon the returns are wanted in (annual covariance gives annual pi)
    - market_weights: capitalization weights summing to 1; risk_aversion: delta, positive, the caller's to state
    returns: an array of implied excess returns pi, one per asset, per the covariance's period
    source: Black and Litterman (1992), Financial Analysts Journal 48(5); He and Litterman (1999), Goldman Sachs
    aliases: reverse optimization, implied equilibrium returns, equilibrium excess returns, CAPM prior, [implied returns]
    example:
        qf.equilibrium_returns(covariance=[[0.04, 0.012], [0.012, 0.09]],
                               market_weights=[0.6, 0.4],
                               risk_aversion=2.5)
    """
    sigma = np.asarray(covariance, dtype=float)
    w = np.asarray(market_weights, dtype=float)
    if sigma.ndim != 2 or sigma.shape != (w.size, w.size):
        raise ValueError("covariance must be (n, n) for the n market_weights")
    return sigma @ w * float(positive("risk_aversion", risk_aversion))


def black_litterman(prior_returns, covariance, tau, pick_matrix, view_returns, view_covariance):
    """Black-Litterman posterior returns and covariance from a prior and uncertain views.

    mu = pi + tau Sigma P' (P tau Sigma P' + Omega)^{-1} (Q - P pi)
       = [(tau Sigma)^{-1} + P' Omega^{-1} P]^{-1} [(tau Sigma)^{-1} pi + P' Omega^{-1} Q],
    M = tau Sigma - tau Sigma P' (P tau Sigma P' + Omega)^{-1} P tau Sigma (the posterior covariance of the mean),
    covariance = Sigma + M (He and Litterman's for returns). Omega = 0 makes the views exact.

    conventions:
    - prior_returns: pi (n,), e.g. equilibrium_returns; covariance: Sigma (n, n) on the same horizon
    - tau: the prior's scale, positive (0.025 to 1 in the literature: the caller's)
    - pick_matrix: P (K, n), one row per view; view_returns: Q (K,); view_covariance: Omega (K, K), or (K,) variances
    returns: a dict: mean (the posterior expected returns); covariance (Sigma + M, for returns);
      mean_covariance (M, the posterior covariance of the mean)
    source: Black and Litterman (1992), Financial Analysts Journal 48(5); He and Litterman (1999); Idzorek (2005)
    aliases: Black-Litterman, Black Litterman, BL model, [posterior returns], views, investor views, BL posterior
    example:
        qf.black_litterman(prior_returns=[0.05, 0.07], covariance=[[0.04, 0.012], [0.012, 0.09]],
                           tau=0.05, pick_matrix=[[1.0, -1.0]], view_returns=[0.01],
                           view_covariance=[0.0004])
    """
    pi = np.asarray(prior_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    p = np.atleast_2d(np.asarray(pick_matrix, dtype=float))
    q = np.atleast_1d(np.asarray(view_returns, dtype=float))
    omega = np.asarray(view_covariance, dtype=float)
    omega = np.diag(omega) if omega.ndim == 1 else omega
    n, views = pi.size, q.size
    if sigma.shape != (n, n) or p.shape != (views, n) or omega.shape != (views, views):
        raise ValueError("shapes: prior_returns (n,), covariance (n, n), pick_matrix (K, n), view_returns (K,), "
                         "view_covariance (K, K) or (K,)")
    if np.any(np.diag(omega) < 0):
        raise ValueError("view_covariance must not hold a negative variance")
    scaled = float(positive("tau", tau)) * sigma
    gain = np.linalg.solve(p @ scaled @ p.T + omega, p @ scaled).T
    mean_cov = scaled - gain @ p @ scaled
    return Result({"mean": pi + gain @ (q - p @ pi), "covariance": sigma + mean_cov, "mean_covariance": mean_cov})


def inverse_volatility_weights(vols):
    """Inverse-volatility weights: each asset's 1/vol as a share of the sum of 1/vol.

    w_i = (1 / sigma_i) / sum_j (1 / sigma_j), along the last axis (a 2-d input gives one weight row per row of vols).
    Equal to equal_risk_contribution_weights when the assets are uncorrelated, and for two assets at any correlation.

    conventions:
    - vols: positive, on one scale (daily or annual alike: the scale cancels); estimate them as the task says
    - long-only, fully invested; no leverage or vol target here (vol_target_leverage scales the whole)
    returns: an array of weights summing to 1 along the last axis
    source: Maillard, Roncalli and Teiletche (2010), "The properties of equally weighted risk contribution portfolios"
    aliases: inverse volatility, inverse vol weights, naive risk parity, 1/vol weighting, volatility parity
    example:
        qf.inverse_volatility_weights(
            vols=[0.10, 0.20, 0.40]
        )
    """
    inverse = 1.0 / positive("vols", vols)
    return out(inverse / np.sum(inverse, axis=-1, keepdims=True))


def equal_risk_contribution_weights(covariance):
    """Equal-risk-contribution (ERC) weights: every asset's share of the variance equal (Maillard-Roncalli-Teiletche).

    w_i (Sigma w)_i = w' Sigma w / n for every i, w > 0, sum w = 1: the unique long-only solution, found as the
    minimum of y' Sigma y / 2 - (1/n) sum ln y_i (Spinu 2013) by cyclical coordinate descent, each step the positive
    root y_i = [-(Sigma y)_{-i} + sqrt((Sigma y)_{-i}^2 + 4 Sigma_ii / n)] / (2 Sigma_ii), then w = y / sum y; to a
    relative spread of the contributions below 1e-12.

    Not inverse-volatility weights (1/vol, often called naive risk parity: inverse_volatility_weights), which ignore
    the correlations; the two agree for uncorrelated assets, equal correlations or two assets. A task's "risk parity" names either:
    its own words (the covariance, or each vol alone) decide.

    conventions:
    - covariance: symmetric positive definite, on any time scale (it cancels); not a correlation matrix
    - long-only, fully invested; equal budgets
    returns: an array of weights, one per asset, summing to 1
    source: Maillard, Roncalli and Teiletche (2010), J. Portfolio Management 36(4); Spinu (2013), arXiv:1311.4057
    aliases: equal risk contribution, ERC, ERC weights, Maillard Roncalli Teiletche, [risk parity], [risk budgeting]
    example:
        qf.equal_risk_contribution_weights(
            covariance=[[0.04, 0.006, 0.0], [0.006, 0.09, 0.012], [0.0, 0.012, 0.16]]
        )
    """
    sigma = np.asarray(covariance, dtype=float)
    n = sigma.shape[0] if sigma.ndim == 2 else 0
    if sigma.ndim != 2 or sigma.shape != (n, n) or n == 0 or not np.allclose(sigma, sigma.T, rtol=1e-12, atol=0):
        raise ValueError("covariance must be a square symmetric matrix")
    if np.any(np.linalg.eigvalsh(sigma) <= 0):
        raise ValueError("covariance must be positive definite")
    diag = np.diag(sigma).copy()
    y = 1.0 / np.sqrt(diag) / np.sqrt(n)
    for _ in range(100_000):
        for i in range(n):
            rest = sigma[i] @ y - diag[i] * y[i]
            y[i] = (-rest + np.sqrt(rest * rest + 4.0 * diag[i] / n)) / (2.0 * diag[i])
        contributions = y * (sigma @ y)
        if np.ptp(contributions) <= 1e-12 * np.mean(contributions):
            break
    return y / np.sum(y)


def _vol_marginals(weights, cov):
    """The weights, the marginals (Sigma w)_i / sigma_p and sigma_p = sqrt(w' Sigma w), checked."""
    w = _finite("weights", weights).ravel()
    s = _finite("cov", cov)
    if s.shape != (w.size, w.size) or not np.allclose(s, s.T):
        raise ValueError("cov must be a symmetric matrix matching the weights")
    sigma_w = s @ w
    sigma = math.sqrt(float(w @ sigma_w))
    if not sigma > 0:
        raise ValueError("the portfolio variance is 0")
    return w, sigma_w / sigma, sigma


def euler_vol_contributions(weights, *, cov):
    """Euler contributions to portfolio vol, RC_i = w_i x the marginal (Sigma w)_i / sigma_p, summing to sigma_p.

    sigma_p = sqrt(w' Sigma w); the marginal MCTR_i = d sigma_p / d w_i = (Sigma w)_i / sigma_p is
    qf.marginal_vol_contributions; RC_i = w_i MCTR_i (Euler's theorem for a homogeneous measure), so the RC sum to
    sigma_p. Under a zero-mean normal model the VaR and ES contributions are these times z_alpha and
    phi(z_alpha) / (1 - alpha). Percentage contributions are RC / sigma_p.
    conventions:
        the result: already times the weights, not the marginal; multiplied by the weights again it counts them twice
        cov: the covariance matrix of the asset returns (its period sets sigma_p's)
    returns: an array of contributions, one per asset, summing to sigma_p
    source: Tasche (2000), "Risk contributions and performance measurement", TU Munchen; Litterman (1996),
        "Hot spots and hedges", Journal of Portfolio Management 23
    aliases: Euler allocation, [risk contribution], component VaR, [risk decomposition]
    example:
        import numpy as np, qf
        w, cov = np.array([0.5, 0.3, 0.2]), np.array([[0.04, 0.006, 0.002], [0.006, 0.09, -0.003], [0.002, -0.003, 0.01]])
        qf.euler_vol_contributions(w, cov=cov)   # w * qf.marginal_vol_contributions(w, cov); sums to sqrt(w @ cov @ w)
    """
    w, marginal, _ = _vol_marginals(weights, cov)
    return w * marginal


def marginal_vol_contributions(weights, cov):
    """Marginal contributions to risk MCTR_i = (Sigma w)_i / sigma_p = d sigma_p / d w_i, not yet times the weights.

    sigma_p = sqrt(w' Sigma w). The contribution RC_i = w_i MCTR_i is qf.euler_vol_contributions: the RC sum to
    sigma_p, the MCTR alone do not. Percentage contributions are w_i MCTR_i / sigma_p; under a zero-mean normal model
    the marginal VaR is z_alpha MCTR_i.

    conventions:
    - weights: the portfolio's, one per asset (the marginal is sigma_p's derivative at them)
    - cov: the covariance matrix of the asset returns (its period sets sigma_p's and the marginals')
    returns: an array of marginals (Sigma w)_i / sigma_p, one per asset; w times it is the contributions
    source: Litterman (1996), "Hot spots and hedges", Journal of Portfolio Management 23; Tasche (2000), "Risk
      contributions and performance measurement", TU Munchen
    aliases: marginal contribution to risk, MCTR, marginal contribution to volatility
    example:
        import numpy as np, qf
        w, cov = np.array([0.5, 0.3, 0.2]), np.array([[0.04, 0.006, 0.002], [0.006, 0.09, -0.003], [0.002, -0.003, 0.01]])
        contributions = qf.euler_vol_contributions(w, cov=cov)   # w * the marginal, summing to sqrt(w @ cov @ w)
        qf.marginal_vol_contributions(weights=w, cov=cov)
    """
    return _vol_marginals(weights, cov)[1]


@_guard(level=('alpha',))
def euler_es_contributions(losses, *, alpha, method, es_rule):
    """Euler ES contributions from simulated scenario losses by position: E[L_i | L in the tail].

    L = sum_i L_i per scenario (rows scenarios, columns positions, losses positive). The VaR is the alpha sample
    quantile of L by `method` (on losses 'higher' is the more extreme neighbour); the tail is L >= VaR
    ('at_or_beyond') or L > VaR ('beyond'); each contribution is the mean of that position's losses over the tail
    scenarios, and they sum to the ES, the tail mean of L.

    No contribution exceeds its position's own ES over as many worst scenarios (subadditivity): the gap is the
    position's diversification benefit.
    conventions:
        alpha: the confidence level, e.g. 0.99
        method: 'linear', 'lower', 'higher', 'nearest', 'midpoint', 'hazen', 'weibull' or 'inverted_cdf'
        es_rule: 'at_or_beyond' or 'beyond'
    returns: a dict: contributions (an array, one per position), var, es, tail_scenarios
    source: Tasche (2002), "Expected shortfall and beyond", Journal of Banking and Finance 26(7); Glasserman
        (2005), "Measuring marginal risk contributions in credit portfolios", Journal of Computational Finance 9(2)
    aliases: Euler ES allocation, ES contribution, CVaR contribution, expected shortfall allocation, marginal ES
    example:
        import numpy as np, qf
        losses = np.random.default_rng(3).exponential([1.0, 2.0, 0.5], size=(10000, 3))
        qf.euler_es_contributions(losses, alpha=0.99, method="linear", es_rule="at_or_beyond")
    """
    choice("method", method, QUANTILE_METHODS)
    choice("es_rule", es_rule, ES_RULES[:2])
    matrix = _finite("losses", losses)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("losses must be a scenarios x positions matrix")
    level = 1.0 - _tail(alpha)
    total = matrix.sum(axis=1)
    var = float(np.quantile(total, level, method=method))
    tail = total >= var if es_rule == "at_or_beyond" else total > var
    if not np.any(tail):
        raise ValueError("no scenario lies beyond the VaR; es_rule 'at_or_beyond' has one")
    return Result({"contributions": matrix[tail].mean(axis=0), "var": var, "es": float(total[tail].mean()),
                   "tail_scenarios": int(tail.sum())})


def _qp(q, c, a, b, g, h, tol):
    """min 1/2 x'Qx - c'x subject to Ax = b and Gx <= h, Q positive definite: the primal active-set method (Nocedal
    and Wright, Algorithm 16.3) from a feasible point linprog finds; each step solves its equality problem exactly."""
    n = q.shape[0]
    start = optimize.linprog(np.zeros(n), A_ub=g if len(h) else None, b_ub=h if len(h) else None,
                             A_eq=a if len(b) else None, b_eq=b if len(b) else None, bounds=[(None, None)] * n,
                             method="highs")
    if start.status != 0:
        raise ValueError("the constraints admit no portfolio (the budget, the target and the bounds conflict)")
    x = start.x
    scale = max(1.0, float(np.max(np.abs(h)))) if len(h) else 1.0
    working = [i for i in range(len(h)) if g[i] @ x >= h[i] - tol * scale]
    for _ in range(50 * (n + len(h) + 1)):
        rows = np.vstack([a, g[working]]) if len(b) or working else np.zeros((0, n))
        gradient = q @ x - c
        kkt = np.block([[q, rows.T], [rows, np.zeros((rows.shape[0], rows.shape[0]))]])
        solved = np.linalg.lstsq(kkt, np.concatenate([-gradient, np.zeros(rows.shape[0])]), rcond=None)[0]
        step = solved[:n]
        if np.max(np.abs(step)) <= tol * max(1.0, float(np.max(np.abs(x)))):
            multipliers = solved[n + len(b):]
            if not working or multipliers.min() >= -tol:
                return x
            working.pop(int(np.argmin(multipliers)))
            continue
        alpha, blocking = 1.0, None
        for i in range(len(h)):
            if i not in working and g[i] @ step > tol:
                room = (h[i] - g[i] @ x) / (g[i] @ step)
                if room < alpha:
                    alpha, blocking = max(room, 0.0), i
        x = x + alpha * step
        if blocking is not None:
            working.append(blocking)
    raise ValueError("the quadratic program did not converge: loosen tolerance")


def mean_variance(expected_returns, covariance, objective, risk_aversion, risk_penalty, target_return, risk_free,
                  budget, long_only, caps, tolerance):
    """Mean-variance (Markowitz) optimization: utility, minimum variance, target return or tangency, with constraints.

    With e = mu - r_f the expected excess returns and the rest of the budget in the risk-free asset, 'utility'
    maximises w'e - (lambda/2) w' Sigma w ('half', Merton's: unconstrained w = Sigma^{-1} e / lambda) or w'e - lambda
    w' Sigma w ('whole', Grinold and Kahn's: Sigma^{-1} e / (2 lambda)); 'min_variance' minimises w' Sigma w fully
    invested; 'target_return' minimises it at r_f + w'e = target; 'max_sharpe' is the tangency portfolio, w
    proportional to Sigma^{-1} e fully invested. Equality constraints by Lagrange; long-only and caps by quadratic
    programming (an active-set method, exact at its solution).

    conventions:
    - expected_returns: mu (N), covariance: Sigma (N x N), on one horizon (annual mu with an annual Sigma)
    - objective: 'utility', 'min_variance', 'target_return' or 'max_sharpe'
    - risk_aversion: lambda and risk_penalty 'half' or 'whole' with 'utility'; None otherwise
    - target_return: the portfolio's expected return with 'target_return'; None otherwise
    - risk_free: r_f, subtracted from mu (0.0 where mu is already excess)
    - budget: 'full' (weights sum to 1), 'zero' (dollar neutral) or 'none' (the rest in cash); 'min_variance' and
      'max_sharpe' take 'full'
    - long_only: True for w >= 0; caps: None, or a number or N bounds on |w_i| (0.1 for a 10% position limit)
    - tolerance: the solver's, e.g. 1e-10
    returns: a dict: weights; expected_return (r_f + w'e); volatility; sharpe ((expected_return - r_f) / volatility);
      cash (1 - sum w)
    source: Markowitz (1952), J. Finance 7(1); Merton (1972), JFQA 7(4); Grinold and Kahn (2000), ch. 14
    aliases: mean-variance optimization, Markowitz portfolio, efficient frontier, tangency portfolio, maximum Sharpe portfolio, minimum variance portfolio, global minimum variance
    example:
        # weights = [0.438095, 0.142857, 0.419048]; expected_return = 0.0744762.
        qf.mean_variance(expected_returns=[0.08, 0.10, 0.06], covariance=[[0.04, 0.01, 0.0], [0.01, 0.09, 0.02],
                         [0.0, 0.02, 0.0225]], objective="utility", risk_aversion=3.0, risk_penalty="half",
                         target_return=None, risk_free=0.02, budget="full", long_only=True, caps=None, tolerance=1e-10)
    """
    choice("objective", objective, OBJECTIVES)
    choice("budget", budget, BUDGETS)
    mu = np.asarray(expected_returns, dtype=float).ravel()
    sigma = np.asarray(covariance, dtype=float)
    n = mu.size
    if sigma.shape != (n, n) or not np.allclose(sigma, sigma.T) or not np.all(np.isfinite(sigma)):
        raise ValueError("covariance must be a symmetric N x N matrix for the N expected_returns")
    if np.any(np.linalg.eigvalsh(sigma) <= 0):
        raise ValueError("covariance must be positive definite")
    if (objective == "utility") != (risk_aversion is not None) or (objective == "utility") != (risk_penalty is not None):
        raise ValueError("risk_aversion and risk_penalty: given with 'utility', None otherwise")
    if (objective == "target_return") != (target_return is not None):
        raise ValueError("target_return: given with 'target_return', None otherwise")
    if objective in ("min_variance", "max_sharpe") and budget != "full":
        raise ValueError(f"'{objective}' needs budget 'full' (with no budget the answer is all cash)")
    if not isinstance(long_only, (bool, np.bool_)):
        raise ValueError("long_only must be True or False")
    tol = float(positive("tolerance", tolerance))
    rf = float(risk_free)
    excess = mu - rf
    ones = np.ones(n)
    limit = None if caps is None else np.broadcast_to(positive("caps", caps), (n,)).astype(float)
    if objective == "max_sharpe":
        rows, bounds = [], []          # y = w / k, k = sum y: min y' Sigma y / 2 at e'y = 1; bounds scale with sum y
        if long_only:
            rows.append(-np.eye(n))
        if limit is not None:
            rows.append(np.eye(n) - np.outer(limit, ones))
            if not long_only:
                rows.append(-np.eye(n) - np.outer(limit, ones))
        g = np.vstack(rows) if rows else np.zeros((0, n))
        y = _qp(sigma, np.zeros(n), excess[None, :], np.array([1.0]), g, np.zeros(g.shape[0]), tol)
        if not y.sum() > 0:
            raise ValueError("no fully invested portfolio has a positive Sharpe ratio here")
        w = y / y.sum()
    else:
        if objective == "utility":
            choice("risk_penalty", risk_penalty, PENALTIES)
            kappa = float(positive("risk_aversion", risk_aversion)) * (1.0 if risk_penalty == "half" else 2.0)
            q, c = kappa * sigma, excess
        else:
            q, c = sigma, np.zeros(n)
        a_rows, b_rows = [], []
        if objective == "target_return":
            a_rows.append(excess)
            b_rows.append(float(target_return) - rf)
        if budget != "none":
            a_rows.append(ones)
            b_rows.append(1.0 if budget == "full" else 0.0)
        g_rows, h_rows = [], []
        if long_only:
            g_rows.append(-np.eye(n))
            h_rows.append(np.zeros(n))
        if limit is not None:
            g_rows.append(np.eye(n))
            h_rows.append(limit)
            if not long_only:
                g_rows.append(-np.eye(n))
                h_rows.append(limit)
        a = np.vstack(a_rows) if a_rows else np.zeros((0, n))
        g = np.vstack(g_rows) if g_rows else np.zeros((0, n))
        w = _qp(q, c, a, np.array(b_rows), g, np.concatenate(h_rows) if h_rows else np.zeros(0), tol)
    vol = float(np.sqrt(w @ sigma @ w))
    expected = rf + float(w @ excess)
    return Result({"weights": w, "expected_return": expected, "volatility": vol,
                   "sharpe": (expected - rf) / vol if vol > 0 else np.nan, "cash": 1.0 - float(w.sum())})


def factor_risk(weights, exposures, factor_covariance, specific_variance, periods_per_year, factor_groups):
    """Factor risk decomposition: factor and specific variance, Euler contributions by factor and group.

    x = B'w (the portfolio's factor exposures), variance = x'Fx + w'Dw (factor part plus specific part, D the
    specific variances), volatility its root. Euler contributions to the volatility, summing to it: by factor x_k (Fx)_k
    / vol, the specific part w_i (Dw)_i / vol by asset, by asset w_i ((BFB' + D)w)_i / vol; a group's is its factors'
    sum. A contribution over the volatility is its share of the risk, the same as its share of the variance.

    conventions:
    - weights: w (N); exposures: B, N assets by k factors (rows assets)
    - factor_covariance: F (k x k); specific_variance: D's diagonal (N variances: square a specific vol first) or N x N
    - periods_per_year: multiplies F and D, the variances (252 for daily ones, 12 for monthly; 1 for annual ones)
    - factor_groups: None, or one label per factor ('style', 'industry'), whose contributions are summed
    returns: a dict: volatility, variance, factor_variance, specific_variance (the portfolio's), factor_share (factor
      over total variance); factor_exposures (x); factor_contributions (k), specific_contributions (N),
      asset_contributions (N); group_contributions (label: contribution; empty with None)
    source: Grinold and Kahn (2000), Active Portfolio Management, ch. 3; Menchero and Davis (2011), J. Portfolio Management 37(3)
    aliases: factor risk decomposition, factor risk model, risk attribution, factor and specific risk, idiosyncratic risk, Barra risk decomposition, [risk decomposition]
    example:
        # variance = 0.0480074; volatility = 0.219106.
        qf.factor_risk(weights=[0.5, 0.3, 0.2], exposures=[[1.0, 0.2], [0.8, -0.5], [1.2, 0.1]],
                       factor_covariance=[[0.04, 0.002], [0.002, 0.01]], specific_variance=[0.02, 0.03, 0.05],
                       periods_per_year=1, factor_groups=["market", "style"])
    """
    w = _finite("weights", weights).ravel()
    b = _finite("exposures", exposures)
    b = b[:, None] if b.ndim == 1 else b
    f = np.atleast_2d(_finite("factor_covariance", factor_covariance))
    d = _finite("specific_variance", specific_variance)
    n, k = w.size, f.shape[0]
    if b.shape != (n, k) or f.shape != (k, k) or not np.allclose(f, f.T):
        raise ValueError("exposures must be N x k (rows assets) and factor_covariance a symmetric k x k matrix")
    d = np.diag(d) if d.ndim == 1 else d
    if d.shape != (n, n) or np.any(np.diag(d) < 0):
        raise ValueError("specific_variance must be N variances (or N x N), not negative")
    scale = float(positive("periods_per_year", periods_per_year))
    f, d = f * scale, d * scale
    x = b.T @ w
    factor_var, specific_var = float(x @ f @ x), float(w @ d @ w)
    variance = factor_var + specific_var
    if not variance > 0:
        raise ValueError("the portfolio's variance is 0")
    vol = np.sqrt(variance)
    by_factor = x * (f @ x) / vol
    groups = {}
    if factor_groups is not None:
        labels = list(factor_groups)
        if len(labels) != k:
            raise ValueError("factor_groups must hold one label per factor")
        for label, contribution in zip(labels, by_factor):
            groups[label] = groups.get(label, 0.0) + float(contribution)
    return Result({"volatility": float(vol), "variance": variance, "factor_variance": factor_var,
                   "specific_variance": specific_var, "factor_share": factor_var / variance, "factor_exposures": x,
                   "factor_contributions": by_factor, "specific_contributions": w * (d @ w) / vol,
                   "asset_contributions": w * ((b @ f @ b.T + d) @ w) / vol, "group_contributions": groups})


def _descend(q, alpha, start, linear, quadratic, tol):
    """Coordinate descent on t = h - h0 for min 1/2 h'Qh - alpha'h + sum linear_i |t_i| + quadratic_i t_i^2: each
    coordinate's exact minimiser, a soft threshold (a no-trade band where linear > 0)."""
    t = np.zeros_like(start)
    moved = q @ start - alpha
    diag = np.diag(q)
    for _ in range(200_000):
        largest = 0.0
        for i in range(t.size):
            slope = moved[i] - diag[i] * t[i]
            new = -np.sign(slope) * max(abs(slope) - linear[i], 0.0) / (diag[i] + 2.0 * quadratic[i])
            change = new - t[i]
            if change:
                moved += q[:, i] * change
                t[i] = new
                largest = max(largest, abs(change))
        if largest <= tol:
            return t
    raise ValueError("the coordinate descent did not converge: loosen tolerance")


def cost_aware_rebalance(holdings, alpha, covariance, risk_aversion, risk_penalty, cost, cost_form, turnover_limit,
                         tolerance):
    """Cost-aware rebalancing: the trades worth their transaction cost, a no-trade band, an optional turnover limit.

    From the current holdings h0, the new holdings h maximise alpha'h - (lambda/2) h' Sigma h ('half') or alpha'h -
    lambda h' Sigma h ('whole', Grinold and Kahn's) less the cost of the trade t = h - h0: 'linear' sum c_i |t_i| or
    'quadratic' sum c_i t_i^2, under an optional turnover limit sum |t_i| <= L. With linear costs asset i is not traded
    while its marginal value alpha_i - (risk's gradient)_i lies within -/+ c_i: the no-trade band. Zero cost gives the
    unconstrained optimum, Sigma^{-1} alpha / lambda ('half'). No budget: cash takes the trades' sum.

    conventions:
    - holdings: h0 (N), as weights; alpha: expected (active or residual) returns (N); covariance: Sigma (N x N)
    - risk_aversion: lambda; risk_penalty: 'half' or 'whole'
    - cost: c, a number or N (0.001 is 10 bps of each unit traded for 'linear'); cost_form: 'linear' or 'quadratic'
    - turnover_limit: None, or L, the most sum |h - h0| may reach (a limit binds by a shadow cost per unit traded)
    - tolerance: the descent's, the largest change of a holding in a sweep, e.g. 1e-12
    returns: a dict: holdings; trades (h - h0); turnover (sum |trades|); cost; objective (alpha'h less the risk term
      and the cost)
    source: Grinold and Kahn (2000), Active Portfolio Management, ch. 16; Garleanu and Pedersen (2013), J. Finance 68(6)
    aliases: cost-aware rebalancing, rebalancing with transaction costs, no-trade band, no-trade region, turnover constraint, turnover-constrained optimization
    example:
        # holdings = [0.250598, 0.0976096, -0.264542]; turnover = 0.916335.
        qf.cost_aware_rebalance(holdings=[0.3, 0.3, 0.4], alpha=[0.02, 0.01, -0.01],
                                covariance=[[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.0225]],
                                risk_aversion=2.0, risk_penalty="half", cost=0.002, cost_form="linear",
                                turnover_limit=None, tolerance=1e-12)
    """
    choice("risk_penalty", risk_penalty, PENALTIES)
    choice("cost_form", cost_form, COST_FORMS)
    h0 = _finite("holdings", holdings).ravel()
    a = _finite("alpha", alpha).ravel()
    sigma = _finite("covariance", covariance)
    n = h0.size
    if a.size != n or sigma.shape != (n, n) or not np.allclose(sigma, sigma.T):
        raise ValueError("holdings and alpha must be N each and covariance a symmetric N x N matrix")
    if np.any(np.linalg.eigvalsh(sigma) <= 0):
        raise ValueError("covariance must be positive definite")
    q = sigma * float(positive("risk_aversion", risk_aversion)) * (1.0 if risk_penalty == "half" else 2.0)
    c = np.broadcast_to(_finite("cost", cost), (n,)).astype(float)
    if np.any(c < 0):
        raise ValueError("cost must not be negative")
    tol = float(positive("tolerance", tolerance))
    zero = np.zeros(n)

    def solve(shadow):
        if cost_form == "linear":
            return _descend(q, a, h0, c + shadow, zero, tol)
        return _descend(q, a, h0, zero + shadow, c, tol)

    t = solve(0.0)
    if turnover_limit is not None:
        cap = float(turnover_limit)
        if not cap >= 0:
            raise ValueError("turnover_limit must be None or a number of at least 0")
        if np.abs(t).sum() > cap:
            low, high = 0.0, float(np.max(np.abs(a - q @ h0))) + 1.0
            for _ in range(200):
                middle = 0.5 * (low + high)
                t = solve(middle)
                low, high = (middle, high) if np.abs(t).sum() > cap else (low, middle)
                if high - low <= tol * max(1.0, high):
                    break
            t = solve(high)
    h = h0 + t
    spent = float(c @ np.abs(t)) if cost_form == "linear" else float(c @ (t * t))
    value = float(a @ h - 0.5 * h @ q @ h - spent)
    return Result({"holdings": h, "trades": t, "turnover": float(np.abs(t).sum()), "cost": spent, "objective": value})
