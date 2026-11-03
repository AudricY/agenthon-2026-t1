"""qf: vetted implementations of standard quantitative-finance methods (numpy and scipy only). Rates are continuously compounded and annual, times in years, vol annual, option 'call' or 'put'; arrays broadcast; help(qf.<name>) gives each formula, convention and source.
- bs_price(option, spot, strike, maturity, rate, dividend_yield, vol): Black-Scholes-Merton European price
- implied_vol(option, price, spot, strike, maturity, rate, dividend_yield, method): its exact inverse, by method 'bisection' or 'newton'
- forward_start_price(option, spot, alpha, start, maturity, rate, dividend_yield, vol): strike alpha*S(start) set at start (Rubinstein); a cliquet is a sum of these
- digital_price(payoff, option, spot, strike, maturity, rate, dividend_yield, vol): payoff 'cash' (pays 1) or 'asset' (pays S_T)
- barrier_digital_price(payoff, option, barrier_type, spot, strike, barrier, maturity, rate, dividend_yield, vol): barrier_type 'down-and-out', 'down-and-in', 'up-and-out' or 'up-and-in', continuous monitoring (reflection principle)
- asian_price(method, option, spot, strike, times, maturity, rate, dividend_yield, vol): average over the monitoring times; method 'geometric' (exact), 'levy' (Levy 1992), 'curran_taylor' or 'curran_root' (Curran 1994, its K-hat by first-order expansion or solved)
- hit_probability(level, horizon, drift, vol): P(X_t = drift*t + vol*W_t reaches level by horizon): running max (level > 0) or min (level < 0); the first-passage CDF
- first_passage_pdf(level, t, drift, vol): first-passage density (inverse Gaussian)
- first_passage_mean(level, drift, vol): expected first-passage time (inf if the drift points away)
- terminal_and_max_cdf(x, m, horizon, drift, vol): P(X_T <= x and max X <= m)
- gaussian_copula_logpdf(u, v, rho): Gaussian copula log-density
- t_copula_logpdf(u, v, rho, nu): Student t copula log-density
- clayton_copula_logpdf(u, v, theta): Clayton copula log-density
- gumbel_copula_logpdf(u, v, theta): Gumbel copula log-density
- copula_param_from_tau(family, tau): rho or theta from Kendall's tau; family 'gaussian', 't', 'clayton', 'gumbel' or 'frank'
- ou_jump_moments(x0, kappa, theta, sigma, jump_rate, jump_mean, jump_std, horizon): (mean, variance) of dX = kappa*(theta - X)dt + sigma*dW + J*dN; horizon=numpy.inf gives the stationary ones
- ou_from_ar1(intercept, slope, resid_var, dt): (kappa, theta, sigma) from an AR(1) fit, by the exact discretisation
- brenner_subrahmanyam_vol(call_price, underlying_pv, maturity): ATM implied-vol approximation; underlying_pv = S*exp(-q*T)
- li_atm_vol(call_price, underlying_pv, maturity): Li (2005) ATM implied-vol approximation
- corrado_miller_vol(call_price, underlying_pv, strike_pv, maturity, variant): implied-vol approximation for any moneyness; strike_pv = K*exp(-r*T); variant 'corrado_miller' or 'hallerbach'
- bandi_russell_noise_var(log_returns): microstructure-noise variance, sum r^2 / (2n) of the highest-frequency returns
- noise_corrected_rv(rv, n_returns, noise_var): rv - 2 * n_returns * noise_var
- bipower_variation(log_returns, scale): sum |r_j||r_{j-1}| times scale 'none', 'pi/2' or 'pi/2*n/(n-1)'
- hmm_smooth(x, means, stds, transition, initial): Gaussian-HMM forward-backward, scaled: (loglik, posteriors as probabilities, expected transition counts)
- hmm_baum_welch(x, means, stds, transition, initial, max_iter, tol, stop, update_initial): Baum-Welch EM; stop 'means' or 'loglik'; a dict of the fit
Shelves added 10-08 (import qf; then qf.<shelf>.<function>, each with help(): conventions (JSON and CSV I/O, returns, annualising, day counts, compounding, EWMA, std, quantiles, drawdown), risk (VaR and ES by seven methods, backtests, Hill and GPD tails, copula samplers and fits, Vasicek and Gordy, Euler contributions, migrations), volatility (EWMA, GARCH through arch, the range estimators, realized variance, AR(1) and OU fits, SVI, Dupire), bs (Greeks with their units named, Black-76, parity forwards, escrowed dividends, Reiner-Rubinstein barrier options with a rebate), rates (bootstrap, Nelson-Siegel and Svensson, yield and risk measures, key rates, immunisation, Hull-White, Jamshidian, caps, CIP, cross rates), econometrics (OLS, Newey-West, rolling OLS, Fama-MacBeth, event studies, PCA, GRS, Black-Litterman, Brinson-Fachler), metrics (Sharpe, Sortino, drawdown dates, CAGR, Calmar, signal backtests, vol targeting, risk parity); and qf.impvol.li_nonatm_vol, Li's implied-vol approximation at any moneyness. Every convention is a required argument.
"""
# Every function is qf.<name>, the same object as qf.<shelf>.<name> (lane names1): every shelf has an `__all__` and is
# imported whole; tests/test_qf_names.py holds the bare names unique and every function exported. `bs` and `asian`
# are import modules since the re-shelving (lane libshelves1), so their functions' old paths still resolve; a
# folded function (docs/research/library-design-1011.md) is out of its shelf's `__all__` and the index, imported
# here by name so qf.<name> still calls it.
from . import asian, bs  # noqa: F401 - the import modules
from .aggregate import *  # noqa: F403
from .backtests import *  # noqa: F403
from .barriers import *  # noqa: F403
from .bonds import *  # noqa: F403
from .capital import *  # noqa: F403
from .clocks import *  # noqa: F403
from .commodity import *  # noqa: F403
from .conventions import *  # noqa: F403
from .copulas import *  # noqa: F403
from .covariance import *  # noqa: F403
from .credit import *  # noqa: F403
from .credit_portfolio import *  # noqa: F403
from .curves import *  # noqa: F403
from .econometrics import *  # noqa: F403
from .events import *  # noqa: F403
from .execution import *  # noqa: F403
from .exotics import *  # noqa: F403
from .factors import *  # noqa: F403
from .fourier import *  # noqa: F403
from .fx import *  # noqa: F403
from .greeks import *  # noqa: F403
from .hmm import *  # noqa: F403
from .impvol import *  # noqa: F403
from .inflation import *  # noqa: F403
from .jumps import *  # noqa: F403
from .ledgers import *  # noqa: F403
from .metrics import *  # noqa: F403
from .microstructure import *  # noqa: F403
from .migrations import *  # noqa: F403
from .montecarlo import *  # noqa: F403
from .multiasset import *  # noqa: F403
from .options import *  # noqa: F403
from .ou import *  # noqa: F403
from .passage import *  # noqa: F403
from .paths import *  # noqa: F403
from .pde import *  # noqa: F403
from .portfolio import *  # noqa: F403
from .primitives import *  # noqa: F403
from .rates import *  # noqa: F403
from .realized import *  # noqa: F403
from .risk import *  # noqa: F403
from .shortrate import *  # noqa: F403
from .signals import *  # noqa: F403
from .surfaces import *  # noqa: F403
from .tails import *  # noqa: F403
from .text import *  # noqa: F403
from .timeseries import *  # noqa: F403
from .volatility import *  # noqa: F403
from .econometrics import rolling_ols  # noqa: F401 - folded: a line on ols's card
from .metrics import annualized_mean_return  # noqa: F401 - folded: a line on annualize's card

__all__ = sorted(name for name, value in list(globals().items())     # every function bound above from a shelf
                 if getattr(value, "__module__", "").startswith(f"{__name__}."))
