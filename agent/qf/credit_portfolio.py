"""Portfolio credit: Vasicek loss law and ES, Gordy IRB and its correlation, granularity, CreditRisk+, CDO tranches.

`import qf`. Default probabilities and losses given default are fractions; a quantile's level is the confidence
(Basel's 0.999). Rating migrations and CreditMetrics are on the `migrations` shelf (their old paths here still
work). A function with several results returns a dict with named keys.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri, stdtrit

from ._common import Result, bivariate_normal_cdf, choice, integer, out, positive
from .aggregate import _var_es
from .conventions import _finite
from .copulas import _draws
from .migrations import cohort_transition_matrix, creditmetrics, generator_from_transition  # noqa: F401 - moved
from .risk import ES_RULES
from .primitives import _guard

__all__ = ["basel_irb_correlation", "creditrisk_plus", "factor_copula_losses", "gordy_capital",
           "granularity_adjustment", "tranche_loss", "vasicek_loss_cdf", "vasicek_loss_es", "vasicek_loss_quantile"]

EXPECTED_LOSS = ("subtract", "include")
FACTOR_COPULAS = ("gaussian", "t")
GA_VERSIONS = ("gordy_2003", "gordy_lutkebohmert")
ASSET_CLASSES = ("corporate", "sme", "large_financial", "mortgage", "revolving", "other_retail")
POOLS = ("large", "finite")
GAUSS_HERMITE_MOST = 300        # numpy's hermegauss weights overflow to NaN a little past this


def _unit_interval(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~((array > 0) & (array < 1))):
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return array


def vasicek_loss_cdf(x, *, pd, rho):
    """Vasicek's large-portfolio default-rate distribution P(L <= x), one factor.

    P(L <= x) = Phi((sqrt(1 - rho) Phi^-1(x) - Phi^-1(pd)) / sqrt(rho)) for the default-rate fraction x of an
    infinitely granular homogeneous portfolio whose asset values share one normal factor with correlation rho
    (a loss fraction of LGD x where losses carry a constant LGD). x = 0 gives 0, x = 1 gives 1.
    conventions:
        pd: the unconditional default probability, inside (0, 1)
        rho: the asset correlation, inside (0, 1)
    returns: a float, or an array for an array
    source: Vasicek (2002), "The distribution of loan portfolio value", Risk 15(12); Vasicek (1987), "Probability of
        loss on loan portfolio", KMV
    aliases: Vasicek, Vasicek distribution, large homogeneous portfolio, LHP, one-factor Gaussian copula, ASRF
    example:
        import qf
        x = 0.05
        qf.vasicek_loss_cdf(x, pd=0.02, rho=0.15)
    """
    pd, rho = _unit_interval("pd", pd), _unit_interval("rho", rho)
    x = np.asarray(x, dtype=float)
    if np.any(~((x >= 0) & (x <= 1))):
        raise ValueError("x must lie inside [0, 1] (a loss fraction)")
    with np.errstate(divide="ignore"):
        return out(ndtr((np.sqrt(1.0 - rho) * ndtri(x) - ndtri(pd)) / np.sqrt(rho)))


@_guard(level=('q',))
def vasicek_loss_quantile(q, *, pd, rho):
    """Vasicek's loss quantile: the worst-case (conditional) default rate at confidence q.

    L_q = Phi((Phi^-1(pd) + sqrt(rho) Phi^-1(q)) / sqrt(1 - rho)), the inverse of qf.vasicek_loss_cdf: the
    conditional default probability at the factor's (1 - q) quantile (Basel's WCDR at q = 0.999). A portfolio's
    Vasicek VaR is sum_i EAD_i LGD_i L_q(pd_i, rho_i), each name at its own pd and rho (arrays broadcast).
    conventions:
        pd: the unconditional default probability, inside (0, 1)
        rho: the asset correlation, inside (0, 1)
    returns: a float, or an array for an array
    source: Vasicek (2002), "The distribution of loan portfolio value", Risk 15(12); Gordy (2003), Journal of
        Financial Intermediation 12(3)
    aliases: worst case default rate, WCDR, Vasicek quantile, credit VaR, conditional default probability
    example:
        import qf
        q = 0.999
        qf.vasicek_loss_quantile(q, pd=0.02, rho=0.15)
    """
    q, pd, rho = _unit_interval("q", q), _unit_interval("pd", pd), _unit_interval("rho", rho)
    return out(ndtr((ndtri(pd) + np.sqrt(rho) * ndtri(q)) / np.sqrt(1.0 - rho)))


@_guard(level=('q',))
def vasicek_loss_es(q, *, pd, rho, lgd):
    """Vasicek ES (large homogeneous pool): the default rate's average beyond its q-quantile, LGD times it for loss.

    The default rate of an infinitely granular homogeneous portfolio is p(Z) = Phi((Phi^-1(pd) - sqrt(rho) Z) /
    sqrt(1 - rho)), Z the normal factor; beyond its q-quantile (qf.vasicek_loss_quantile) lie the factor's worst
    1 - q, so ES_q = E[p(Z) | Z <= Phi^-1(1 - q)] = Phi2(Phi^-1(pd), Phi^-1(1 - q); sqrt(rho)) / (1 - q), Phi2 the
    bivariate normal CDF (qf.bivariate_normal_cdf); times a constant LGD for the loss fraction. It is the mean of the
    quantile over the tail levels, (1 / (1 - q)) int_q^1 L_u du, at least the quantile; rho -> 0 gives pd.
    conventions:
        q: the confidence level, e.g. 0.999 (the tail is 1 - q)
        pd: the unconditional default probability, inside (0, 1)
        rho: the asset correlation, inside (0, 1)
        lgd: the loss given default as a fraction (1 for the default rate itself)
    returns: a float, or an array for an array (the ES as a fraction of the portfolio)
    source: Vasicek (2002), "The distribution of loan portfolio value", Risk 15(12); McNeil, Frey and Embrechts,
        Quantitative Risk Management (2005), 8.4 (one-factor models); Acerbi and Tasche (2002) (ES as the quantile's
        tail mean)
    aliases: Vasicek ES, LHP ES, Vasicek CVaR, large homogeneous portfolio ES
    example:
        import qf
        qf.vasicek_loss_es(q=0.999, pd=0.02, rho=0.15, lgd=0.45)
        # ES about 0.094232 of the portfolio
    """
    q, pd, rho = _unit_interval("q", q), _unit_interval("pd", pd), _unit_interval("rho", rho)
    lgd = np.asarray(lgd, dtype=float)
    if np.any(~((lgd >= 0) & (lgd <= 1))):
        raise ValueError("lgd must lie inside [0, 1], a fraction (0.45, not 45); 1 for the default rate")
    tail = np.vectorize(lambda a, b, r: bivariate_normal_cdf(float(ndtri(a)), float(ndtri(1.0 - b)), float(np.sqrt(r))),
                        otypes=[float])(pd, q, rho)
    return out(lgd * tail / (1.0 - q))


@_guard(time=('maturity',), level=('q',))
def gordy_capital(*, pd, lgd, rho, q, expected_loss, maturity):
    """Gordy's ASRF capital per unit of exposure (the Basel IRB formula), its options named.

    K = LGD [Phi((Phi^-1(PD) + sqrt(rho) Phi^-1(q)) / sqrt(1 - rho)) - PD] with expected_loss 'subtract' (Basel's
    unexpected-loss charge), without the - PD for 'include'; times the maturity adjustment (1 + (M - 2.5) b) / (1 -
    1.5 b), b = (0.11852 - 0.05478 ln PD)^2, where `maturity` M is a number. RWA = 12.5 K EAD (Basel II then scales
    by 1.06). Basel's corporate rho = 0.12 f + 0.24 (1 - f), f = (1 - e^(-50 PD)) / (1 - e^(-50)) (every class's is
    qf.basel_irb_correlation). A heterogeneous portfolio's capital is the exposure-weighted sum sum_i EAD_i K_i, each
    name at its own pd, lgd, rho and maturity (arrays broadcast; the example), and its RWA 12.5 sum_i EAD_i K_i.

    A PD's usual scoring input, Altman's Z-score: the original Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5 (X1 to
    X3 working capital, retained earnings and EBIT over total assets; X4 market equity over total liabilities; X5
    sales over total assets), distress below 1.81, safe above 2.99; Z' (private firms, X4 book equity) 0.717, 0.847,
    3.107, 0.420, 0.998, zones 1.23 and 2.90; Z'' (non-manufacturers, no X5) 6.56, 3.26, 6.72, 1.05, zones 1.10 and
    2.60.
    conventions:
        pd: the default probability, inside (0, 1)
        lgd: the loss given default, a fraction inside [0, 1]
        rho: the asset correlation, inside (0, 1)
        q: the confidence level, 0.999 in Basel II
        expected_loss: 'subtract' (unexpected loss only) or 'include'
        maturity: M in years for Basel's maturity adjustment, or None for none
    returns: a float, or an array for arrays
    source: Gordy (2003), "A risk-factor model foundation for ratings-based bank capital rules", Journal of
        Financial Intermediation 12(3); Basel Committee (2005), An Explanatory Note on the Basel II IRB Risk Weight
        Functions; Basel II (2006), paragraph 272; Altman (1968), Journal of Finance 23(4), and Altman (2000)
    aliases: Gordy, IRB formula, Basel IRB, ASRF, [capital requirement], [risk weight], [regulatory capital]
    example:
        import numpy as np, qf
        ead, pd = np.array([100.0, 250.0, 80.0]), np.array([0.01, 0.02, 0.05])
        ead @ qf.gordy_capital(pd=pd, lgd=0.45, rho=0.19, q=0.999, expected_loss="subtract", maturity=2.5)
    """
    choice("expected_loss", expected_loss, EXPECTED_LOSS)
    pd = _unit_interval("pd", pd)
    lgd = np.asarray(lgd, dtype=float)
    if np.any(~((lgd >= 0) & (lgd <= 1))):
        raise ValueError("lgd must lie inside [0, 1], a fraction (0.45, not 45)")
    worst = np.asarray(vasicek_loss_quantile(q, pd=pd, rho=rho))
    capital = lgd * (worst - pd if expected_loss == "subtract" else worst)
    if maturity is None:
        return out(capital)
    m = np.asarray(maturity, dtype=float)
    if np.any(~(m > 0)):
        raise ValueError("maturity must be positive years, or None for no adjustment")
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    return out(capital * (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b))


def basel_irb_correlation(*, pd, asset_class, sales):
    """Basel IRB correlation R by exposure class: corporate (SME, large bank), mortgage, revolving and other retail.

    With f_k(PD) = (1 - e^(-k PD)) / (1 - e^(-k)): 'corporate' (also sovereigns and banks) R = 0.12 f_50 + 0.24 (1 -
    f_50), from 0.24 at PD near 0 to 0.12 as PD grows; 'sme' that less 0.04 (1 - (S - 5) / 45), S the annual sales
    in millions of euros held within [5, 50]; 'large_financial' 1.25 times the corporate R (Basel III); 'mortgage'
    (residential) 0.15; 'revolving' (qualifying revolving retail) 0.04; 'other_retail' 0.03 f_35 + 0.16 (1 - f_35).
    The R that qf.gordy_capital's rho takes.
    conventions:
        pd: the obligor's one-year default probability, inside (0, 1) (Basel floors it at 0.0003 for corporates)
        asset_class: 'corporate', 'sme', 'large_financial', 'mortgage', 'revolving' or 'other_retail'
        sales: the SME's annual sales in millions of euros for 'sme'; None for the other classes
    returns: a float, or an array for an array
    source: Basel Committee, International Convergence of Capital Measurement and Capital Standards (2006),
        paragraphs 272 (corporate), 273 (SME), 328 (mortgage), 329 (revolving) and 330 (other retail); Basel III
        (2011), paragraph 102 (the 1.25 multiplier for large financial institutions)
    aliases: Basel IRB correlation, [IRB asset correlation], asset value correlation, AVC, SME adjustment
    example:
        import qf
        qf.basel_irb_correlation(pd=0.01, asset_class="corporate", sales=None)
        # correlation about 0.192784
    """
    choice("asset_class", asset_class, ASSET_CLASSES)
    pd = _unit_interval("pd", pd)
    if (sales is None) != (asset_class != "sme"):
        raise ValueError("sales: the SME's sales in millions of euros for 'sme', None for every other class")

    def blend(k, low, high):
        f = -np.expm1(-k * pd) / -np.expm1(-k)
        return low * f + high * (1.0 - f)

    if asset_class == "mortgage":
        found = np.full(pd.shape, 0.15)
    elif asset_class == "revolving":
        found = np.full(pd.shape, 0.04)
    elif asset_class == "other_retail":
        found = blend(35.0, 0.03, 0.16)
    else:
        found = blend(50.0, 0.12, 0.24)
        if asset_class == "large_financial":
            found = 1.25 * found
        elif asset_class == "sme":
            size = np.clip(np.asarray(sales, dtype=float), 5.0, 50.0)
            found = found - 0.04 * (1.0 - (size - 5.0) / 45.0)
    return out(found)


@_guard(time=('maturity',), level=('q',))
def granularity_adjustment(*, exposure, pd, lgd, lgd_variance, rho, q, version, xi, maturity):
    """Granularity adjustment of the ASRF quantile for a finite portfolio, in either version.

    The first-order add-on to the infinitely granular loss quantile for name concentration, a fraction of the total
    exposure, s_i = EAD_i / sum EAD. 'gordy_2003' (Gordy 2003; Martin and Wilde 2002), in Vasicek's one-factor model:
    GA = [sigma^2 (y + mu''/mu') - (sigma^2)'] / (2 mu') at y = Phi^-1(q), mu(y) = sum s_i ELGD_i p_i(y), sigma^2(y)
    = sum s_i^2 [(ELGD_i^2 + V_i) p_i - ELGD_i^2 p_i^2], p_i(y) = Phi((Phi^-1(PD_i) + sqrt(rho_i) y) /
    sqrt(1 - rho_i)). 'gordy_lutkebohmert' (2013), CreditRisk+'s gamma factor fitted to IRB capital: GA = 1/(2 K*)
    sum s_i^2 [delta C_i (K_i + R_i) + delta (K_i + R_i)^2 V_i / ELGD_i^2 - K_i (C_i + 2 (K_i + R_i) V_i /
    ELGD_i^2)], K_i gordy_capital's ('subtract'), R_i = ELGD_i PD_i, C_i = (ELGD_i^2 + V_i) / ELGD_i, K* = sum s_i
    K_i, delta = (x_q - 1)(xi + (1 - xi)/x_q), x_q the q-quantile of a gamma of mean 1 and variance 1/xi (4.83 at
    0.999, xi 0.25). V_i is the LGD's variance. Both fall as 1/n for n equal names.
    conventions:
        exposure: EAD per name, positive (the shares are over the names given)
        pd: per name, inside (0, 1)
        lgd: ELGD per name, inside (0, 1]
        lgd_variance: V_i per name (0 for fixed LGDs; Gordy and Lutkebohmert's regulatory 0.25 ELGD (1 - ELGD))
        rho: the asset correlation per name, as gordy_capital
        q: the confidence level, 0.999 in Basel
        version: 'gordy_2003' or 'gordy_lutkebohmert'
        xi: the gamma factor's precision ('gordy_lutkebohmert', 0.25; None for 'gordy_2003')
        maturity: gordy_capital's M for K_i ('gordy_lutkebohmert', or None; None for 'gordy_2003')
    returns: a float, the add-on to capital per unit of total exposure
    source: Gordy (2003), J. Financial Intermediation 12(3); Gordy and Lutkebohmert (2013), IJCB 9(3); Lutkebohmert (2009)
    aliases: granularity adjustment, name concentration, single-name concentration, Gordy-Lutkebohmert, granularity add-on
    example:
        import qf
        ead = [40.0, 25.0, 15.0, 10.0, 10.0]
        qf.granularity_adjustment(exposure=ead, pd=[0.01, 0.02, 0.005, 0.03, 0.01], lgd=[0.45] * 5, lgd_variance=[0.0] * 5, rho=0.15, q=0.999, version="gordy_2003", xi=None, maturity=None)
    """
    choice("version", version, GA_VERSIONS)
    ead = positive("exposure", exposure).ravel()
    p_d = _unit_interval("pd", pd).ravel() * np.ones(ead.size)
    elgd = np.asarray(lgd, dtype=float).ravel() * np.ones(ead.size)
    vlgd = np.asarray(lgd_variance, dtype=float).ravel() * np.ones(ead.size)
    corr = _unit_interval("rho", rho).ravel() * np.ones(ead.size)
    if np.any(~((elgd > 0) & (elgd <= 1))) or np.any(vlgd < 0) or np.any(vlgd > elgd * (1 - elgd) + 1e-15):
        raise ValueError("lgd must lie in (0, 1] and lgd_variance in [0, lgd (1 - lgd)], the most a fraction allows")
    share = ead / ead.sum()
    if version == "gordy_2003":
        if xi is not None or maturity is not None:
            raise ValueError("xi and maturity are for 'gordy_lutkebohmert': pass None")
        y = float(ndtri(_unit_interval("q", q)))
        z = (ndtri(p_d) + np.sqrt(corr) * y) / np.sqrt(1 - corr)
        p, slope = ndtr(z), stats.norm.pdf(z) * np.sqrt(corr / (1 - corr))
        bend = -z * stats.norm.pdf(z) * corr / (1 - corr)
        mu1, mu2 = np.sum(share * elgd * slope), np.sum(share * elgd * bend)
        second = share * share * (elgd * elgd + vlgd)
        var = np.sum(second * p - share * share * elgd * elgd * p * p)
        var1 = np.sum(second * slope - 2 * share * share * elgd * elgd * p * slope)
        return float((var * (y + mu2 / mu1) - var1) / (2 * mu1))
    precision = float(positive("xi", xi))
    x_q = stats.gamma(a=precision, scale=1 / precision).ppf(float(q))
    delta = (x_q - 1) * (precision + (1 - precision) / x_q)
    k = np.asarray(gordy_capital(pd=p_d, lgd=elgd, rho=corr, q=q, expected_loss="subtract", maturity=maturity))
    r_i, c_i, spread = elgd * p_d, (elgd * elgd + vlgd) / elgd, vlgd / (elgd * elgd)
    total = k + r_i
    terms = delta * c_i * total + delta * total * total * spread - k * (c_i + 2 * total * spread)
    return float(np.sum(share * share * terms) / (2 * np.sum(share * k)))


def factor_copula_losses(*, pd, lgd, exposure, loading, factor_corr, factor_index, copula, nu, n, rng):
    """Default losses simulated by a multi-factor Gaussian or Student t copula, per obligor.

    Obligor i's latent X_i = a_i Z_k(i) + sqrt(1 - a_i^2) e_i, the sector factors Z ~ N(0, factor_corr) and e_i
    independent N(0, 1); 'gaussian' defaults where X_i < Phi^-1(pd_i), 't' divides a scenario's every X_i by one
    sqrt(W / nu), W ~ chi-square(nu), and defaults where X_i < t_nu^-1(pd_i); a default loses exposure_i lgd_i. An
    asset correlation rho_i (Basel's, CreditMetrics') is a_i^2; two obligors' is a_i a_j factor_corr[k(i), k(j)]. The
    mean loss is sum_i exposure_i lgd_i pd_i under either copula; under the t joint defaults cluster. The losses by
    obligor go to qf.euler_es_contributions as they are; the portfolio's VaR and ES are
    qf.historical_var_es of -portfolio_losses (gains-positive).
    conventions:
        pd: each obligor's default probability over the horizon, inside (0, 1)
        lgd: the loss given default, fractions inside [0, 1]
        exposure: the exposure at default, in currency
        loading: a_i in [-1, 1], the square root of the obligor's asset correlation
        factor_corr: the sector factors' correlation matrix, k x k ([[1.0]] for one factor)
        factor_index: each obligor's sector, an int in [0, k)
        copula: 'gaussian' or 't'
        nu: the t copula's degrees of freedom, above 0; None for 'gaussian'
        n: the number of scenarios
        rng: a Generator or int seed (default_rng), or a RandomState (numpy's legacy stream); the factors (n x k) first,
            then the e_i (n x m), then W (n, for 't')
    returns: a dict: obligor_losses (n x m), portfolio_losses (n), defaults (n x m booleans)
    source: Gupton, Finger and Bhatia, CreditMetrics Technical Document (1997); Li (2000), Journal of Fixed Income
        9(4); Glasserman and Li (2005), Management Science 51(11)
    aliases: factor copula, Gaussian factor copula, t copula default simulation, multi-factor default model, credit portfolio simulation, default simulation
    example:
        import numpy as np, qf
        pd, lgd, ead, a = [0.01, 0.02, 0.05], [0.45, 0.45, 0.6], [100.0, 50.0, 80.0], [0.4, 0.5, 0.3]
        qf.factor_copula_losses(pd=pd, lgd=lgd, exposure=ead, loading=a, factor_corr=[[1.0, 0.3], [0.3, 1.0]], factor_index=[0, 0, 1], copula="t", nu=5.0, n=10000, rng=np.random.default_rng(1))
    """
    choice("copula", copula, FACTOR_COPULAS)
    p = _unit_interval("pd", pd).ravel()
    m = p.size
    lgd, ead, a = (np.broadcast_to(_finite(name, value), (m,)).astype(float) for name, value in
                   (("lgd", lgd), ("exposure", exposure), ("loading", loading)))
    if np.any((lgd < 0) | (lgd > 1)) or np.any(ead < 0) or np.any(np.abs(a) > 1):
        raise ValueError("lgd must lie in [0, 1], exposure must not be negative and loading must lie in [-1, 1]")
    corr = _finite("factor_corr", factor_corr)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or not np.allclose(corr, corr.T, atol=1e-12) or \
            not np.allclose(np.diag(corr), 1.0, atol=1e-12):
        raise ValueError("factor_corr must be a symmetric k x k matrix with a unit diagonal ([[1.0]] for one factor)")
    try:
        lower = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as error:
        raise ValueError("factor_corr must be positive definite") from error
    sector = np.asarray(factor_index)
    if sector.shape != (m,) or not np.issubdtype(sector.dtype, np.integer) or np.any((sector < 0) |
                                                                                   (sector >= corr.shape[0])):
        raise ValueError(f"factor_index must hold one int sector in [0, {corr.shape[0]}) per obligor")
    if copula == "gaussian":
        if nu is not None:
            raise ValueError("nu must be None for the Gaussian copula")
        threshold = ndtri(p)
    else:
        nu = float(positive("nu", nu))
        threshold = stdtrit(nu, p)
    n, rng = _draws(n, rng)
    factors = rng.standard_normal((n, corr.shape[0])) @ lower.T
    latent = a * factors[:, sector] + np.sqrt(1.0 - a * a) * rng.standard_normal((n, m))
    if copula == "t":
        latent /= np.sqrt(rng.chisquare(nu, n) / nu)[:, None]
    defaults = latent < threshold
    losses = defaults * (ead * lgd)
    return Result({"obligor_losses": losses, "portfolio_losses": losses.sum(axis=1), "defaults": defaults})


def _min_loss(cap, pd, recovery, rho):
    """E[min(L, cap)] for the large pool's loss L = (1 - R) p(Z): cap Phi(z) + (1 - R)(pd - Phi2(c, z; sqrt(rho))),
    z the factor level where L reaches cap."""
    if cap <= 0:
        return 0.0
    severity = 1.0 - recovery
    if cap >= severity:
        return severity * pd
    c = float(ndtri(pd))
    z = (c - np.sqrt(1.0 - rho) * float(ndtri(cap / severity))) / np.sqrt(rho)
    return cap * float(ndtr(z)) + severity * (pd - bivariate_normal_cdf(c, z, float(np.sqrt(rho))))


def tranche_loss(*, attachment, detachment, pd, recovery, rho, pool, names, nodes):
    """CDO tranche loss (attachment to detachment) under a Gaussian copula: large-pool Vasicek or finite binomial pool.

    The tranche [A, D] loses min(max(L - A, 0), D - A) of the pool loss L, so E = E[min(L, D)] - E[min(L, A)],
    reported over the tranche's notional D - A and over the pool's. Given the factor Z each name defaults with
    p(Z) = Phi((Phi^-1(pd) - sqrt(rho) Z) / sqrt(1 - rho)) and loses 1 - R. 'large': L = (1 - R) p(Z), in closed form
    with the bivariate normal (the integral of 1 - qf.vasicek_loss_cdf over [A, D]); 'finite': n equal names, the
    defaults binomial(n, p(Z)) given Z, mixed over Z by Gauss-Hermite nodes, the default count's law returned. Tranches
    across the capital structure add to the pool's (1 - R) pd; the equity tranche's loss falls as rho rises. Base
    correlation is the rho at which the equity tranche [0, D] matches a quoted loss: a root in rho
    (scipy.optimize.brentq).
    conventions:
        attachment, detachment: A and D, fractions of the pool's notional, 0 <= A < D <= 1
        pd: each name's default probability to the horizon (1 - e^(-lambda T) from a hazard)
        recovery: R, a fraction; rho: the factor correlation, inside (0, 1)
        pool: 'large' (infinitely many names) or 'finite'
        names: the pool's count of equal names for 'finite'; None for 'large'
        nodes: Gauss-Hermite nodes for 'finite' (64 to 128 for a pool of about a hundred names; a pool of thousands
            is near 'large'); None for 'large'
    returns: a dict: expected_loss (over the tranche's notional); expected_loss_pool (over the pool's); pool_loss
        ((1 - R) pd); default_distribution (P(k defaults), k = 0..n, for 'finite'; None for 'large')
    source: Li (2000), "On default correlation: a copula function approach", Journal of Fixed Income 9(4); Vasicek
        (2002); O'Kane, Modelling Single-name and Multi-name Credit Derivatives (2008), ch. 16-18; Hull, Options,
        Futures, and Other Derivatives, "Synthetic CDOs"
    aliases: tranche loss, CDO tranche, synthetic CDO, CDO, Gaussian copula CDO, mezzanine tranche, large homogeneous pool
    example:
        import qf
        qf.tranche_loss(attachment=0.03, detachment=0.07, pd=0.05, recovery=0.4, rho=0.3, pool="large", names=None, nodes=None)
        # expected_loss about 0.195847 of the tranche's notional
    """
    choice("pool", pool, POOLS)
    a, d = float(attachment), float(detachment)
    if not 0.0 <= a < d <= 1.0:
        raise ValueError("attachment and detachment must satisfy 0 <= A < D <= 1, fractions of the pool")
    p = float(_unit_interval("pd", pd))
    r, corr = float(recovery), float(_unit_interval("rho", rho))
    if not 0.0 <= r < 1.0:
        raise ValueError("recovery must lie in [0, 1), a fraction")
    distribution = None
    if pool == "large":
        if names is not None or nodes is not None:
            raise ValueError("names and nodes are for 'finite': pass None")
        loss = _min_loss(d, p, r, corr) - _min_loss(a, p, r, corr)
    else:
        count, points = integer("names", names, 1), integer("nodes", nodes, 2)
        if points > GAUSS_HERMITE_MOST:
            raise ValueError(f"nodes must be at most {GAUSS_HERMITE_MOST} (numpy's Gauss-Hermite weights overflow past it)")
        z, weight = np.polynomial.hermite_e.hermegauss(points)
        weight = weight / weight.sum()
        conditional = ndtr((ndtri(p) - np.sqrt(corr) * z) / np.sqrt(1.0 - corr))
        k = np.arange(count + 1)
        distribution = weight @ stats.binom.pmf(k[None, :], count, conditional[:, None])
        pool_loss = k * (1.0 - r) / count
        loss = float(distribution @ np.clip(pool_loss - a, 0.0, d - a))
    return Result({"expected_loss": loss / (d - a), "expected_loss_pool": loss, "pool_loss": (1.0 - r) * p,
                   "default_distribution": distribution})


BANDINGS = ("up", "nearest")
PRESERVED = ("expected_loss", "default_probability")


@_guard(level=('alphas',))
def creditrisk_plus(*, exposure, pd, unit, banding, preserve, rate_vol, points, alphas, es_rule):
    """CreditRisk+ loss distribution: exposure bands and gamma-mixed Poisson defaults by the Panjer recursion.

    Each loss E_i (exposure net of recovery) is banded to nu_i = E_i / L rounded ('up' or 'nearest', at least 1),
    and its default intensity mu_i kept at its PD ('default_probability') or scaled to keep its expected loss, mu_i =
    p_i E_i / (nu_i L) ('expected_loss', CSFB's rule). With mu = sum mu_i, the sector's default rate a gamma of mean
    1 and std w = rate_vol (alpha = 1/w^2, beta = mu w^2, p = beta / (1 + beta)), the loss in units has generating
    function ((1 - p) / (1 - p P(z)))^alpha, P(z) = sum mu_i z^{nu_i} / mu, so A_0 = (1 - p)^alpha and A_n = (p / n)
    sum_{nu_i <= n} (mu_i / mu) (n - nu_i + alpha nu_i) A_{n - nu_i}; rate_vol 0 is independent Poisson defaults, A_0
    = e^{-mu}, A_n = (1/n) sum mu_i nu_i A_{n - nu_i}. The mean is L sum mu_i nu_i (the expected loss under
    'expected_loss'), the variance L^2 (sum mu_i nu_i^2 + w^2 (sum mu_i nu_i)^2). VaR and ES as qf.aggregate_loss's.
    conventions:
        exposure: each obligor's loss given default in currency (exposure times LGD); pd: each default probability
        unit: L, the band width in currency; banding: 'up' or 'nearest'
        preserve: 'expected_loss' (CSFB) or 'default_probability'
        rate_vol: the default rate's std over its mean, one sector (0: Poisson defaults)
        points: the grid's length in units (losses 0 .. (points - 1) L)
        alphas: confidence levels inside (0, 1); es_rule: 'beyond', 'at_or_beyond' or 'exact'
    returns: a dict: grid (losses in currency); pmf; cdf; var and es (one per alpha); mean; variance; bands (nu_i);
        intensities (mu_i); tail_mass (the probability beyond the grid)
    source: Credit Suisse First Boston, CreditRisk+: A Credit Risk Management Framework (1997), A4 and A10; Gundlach
        and Lehrbass (2004), CreditRisk+ in the Banking Industry, ch. 2 and 5
    aliases: CreditRisk+, CreditRisk plus, CSFB CreditRisk+, Panjer recursion credit, gamma-mixed Poisson, exposure bands
    example:
        import qf
        e, p = [120.0, 80.0, 250.0, 60.0, 150.0], [0.02, 0.05, 0.01, 0.08, 0.03]
        qf.creditrisk_plus(exposure=e, pd=p, unit=20.0, banding="up", preserve="expected_loss", rate_vol=0.5, points=200, alphas=[0.99], es_rule="beyond")
    """
    choice("banding", banding, BANDINGS)
    choice("preserve", preserve, PRESERVED)
    choice("es_rule", es_rule, ES_RULES)
    loss = positive("exposure", exposure).ravel()
    p = _unit_interval("pd", pd).ravel()
    if p.shape != loss.shape:
        raise ValueError("exposure and pd need one value per obligor")
    width, size = float(positive("unit", unit)), integer("points", points, 2)
    w = float(rate_vol)
    if not w >= 0:
        raise ValueError("rate_vol must not be negative")
    nu = np.maximum(np.ceil(loss / width - 1e-12) if banding == "up" else np.rint(loss / width), 1.0).astype(int)
    mu = p * loss / (nu * width) if preserve == "expected_loss" else p.copy()
    total = mu.sum()
    a = np.zeros(size)
    weights = np.zeros(size)
    np.add.at(weights, nu[nu < size], mu[nu < size])     # each band's intensity by its width
    reach = np.flatnonzero(weights)
    if w == 0:
        a[0] = np.exp(-total)
        for k in range(1, size):
            j = reach[reach <= k]
            a[k] = (weights[j] * j) @ a[k - j] / k
    else:
        alpha = 1.0 / (w * w)
        q = total * w * w / (1.0 + total * w * w)
        a[0] = (1.0 - q) ** alpha
        for k in range(1, size):
            j = reach[reach <= k]
            a[k] = q / k * (weights[j] / total * (k - j + alpha * j)) @ a[k - j]
    grid = width * np.arange(size)
    var, es = _var_es(grid, a, [float(x) for x in np.atleast_1d(alphas)], es_rule)
    first = float(np.sum(mu * nu))
    return Result({"grid": grid, "pmf": a, "cdf": np.cumsum(a), "var": var, "es": es, "mean": width * first,
                   "variance": width * width * (float(np.sum(mu * nu * nu)) + w * w * first * first),
                   "bands": nu, "intensities": mu, "tail_mass": float(max(1.0 - a.sum(), 0.0))})
