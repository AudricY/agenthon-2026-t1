"""agent/qf, the method library (runs/lanes/qf-prompt.md): every function against a value published in a named
textbook or paper, or an identity the method must satisfy, each source named in its test. No unit's instruction
values, checks, expected values or data appear here (rules 8 and 9): the parameters are textbook examples or chosen
freely. The Monte Carlo cross-checks are in tests/test_qf_slow.py; these stay under a second of CPU together."""
import inspect
import itertools
import math

import numpy as np
import pytest
from scipy.special import stdtrit

import qf.barriers
import qf.exotics
import qf
from conftest import ROOT
from test_qf_conventions import OPTIONAL

pytestmark = pytest.mark.fast

Phi = np.vectorize(lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0)))


def phi(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------------------------------------------ the package


def test_the_list_names_its_functions_with_their_signatures_and_no_unit():
    """qf.__doc__'s list (what N9's system prompt carries: the first library's functions; every function is exported
    since lane names1, tests/test_qf_names.py) names exported functions once each, each with its own signature, every
    one stating its formula's source; no function takes a default; nothing in the package names a unit (rules 8
    and 9)."""
    listed = [line[2:] for line in qf.__doc__.splitlines() if line.startswith("- ")]
    names = [entry.split("(")[0] for entry in listed]
    assert len(names) == len(set(names)) >= 25 and set(names) <= set(qf.__all__)
    for entry in listed:
        name = entry.split("(")[0]
        assert entry.startswith(name + str(inspect.signature(getattr(qf, name))) + ": "), name
        assert "Source:" in getattr(qf, name).__doc__, name
    for path in (ROOT / "agent/qf").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "t1-" not in text and "/app/" not in text and "open(" not in text, path.name
    for name in qf.__all__:   # no default argument anywhere: every convention is the caller's to state (the writers'
        parameters = inspect.signature(getattr(qf, name)).parameters.values()     # `decimals` alone, lane contract1)
        assert all(p.default is inspect.Parameter.empty or p.default is None and p.name in OPTIONAL.get(name, ())
                   for p in parameters), name


def test_a_scalar_comes_back_a_float_an_array_an_array_and_a_bad_argument_is_refused():
    """What a script writes to JSON must be a plain number (json.dumps refuses a 0-d array); names are checked."""
    assert type(qf.bs_price("call", 100.0, 95.0, 1.0, 0.02, 0.0, 0.2)) is float
    assert qf.bs_price("call", 100.0, [95.0, 105.0], 1.0, 0.02, 0.0, 0.2).shape == (2,)
    for bad in (dict(spot=0.0), dict(spot=-1.0), dict(vol=0.0)):
        with pytest.raises(ValueError):
            qf.barrier_digital_price("cash", "call", "down-and-out", **{**dict(spot=100.0, strike=100.0, barrier=90.0,
                                     maturity=1.0, rate=0.02, dividend_yield=0.0, vol=0.2), **bad})
    with pytest.raises(ValueError, match="'cash', 'asset'"):
        qf.digital_price("bond", "call", 100.0, 95.0, 1.0, 0.02, 0.0, 0.2)


# ------------------------------------------------------------------------------------------------ Black-Scholes


def test_bs_price_gives_hulls_worked_example_parity_and_the_yield_as_a_discounted_spot():
    """Hull, Options, Futures, and Other Derivatives, the Black-Scholes-Merton worked example: S0 = 42, K = 40,
    r = 10%, vol = 20%, T = 0.5 gives c = 4.76 and p = 0.81. Put-call parity with a yield, c - p = S e^{-qT} -
    K e^{-rT}, and Merton's rule that a yield q prices as the spot S e^{-qT} with no yield."""
    assert qf.bs_price("call", 42.0, 40.0, 0.5, 0.10, 0.0, 0.20) == pytest.approx(4.76, abs=0.005)
    assert qf.bs_price("put", 42.0, 40.0, 0.5, 0.10, 0.0, 0.20) == pytest.approx(0.81, abs=0.005)
    s, k, t, r, q, v = 100.0, np.array([70.0, 95.0, 100.0, 130.0]), 1.5, 0.03, 0.045, 0.35
    call, put = qf.bs_price("call", s, k, t, r, q, v), qf.bs_price("put", s, k, t, r, q, v)
    assert call - put == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-12)
    assert call == pytest.approx(qf.bs_price("call", s * math.exp(-q * t), k, t, r, 0.0, v), rel=1e-12)
    assert qf.bs_price("call", s, 90.0, t, r, q, 0.0) == pytest.approx(s * math.exp(-q * t) - 90.0 * math.exp(-r * t))
    assert qf.bs_price("put", s, 90.0, t, r, q, 0.0) == 0.0
    with pytest.raises(ValueError):
        qf.bs_price("straddle", s, 90.0, t, r, q, v)


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_implied_vol_inverts_bs_price_and_refuses_a_price_outside_the_bounds(method):
    """Hull, "Implied volatilities": the vol that reproduces the price, by either method; none exists outside the
    no-arbitrage bounds."""
    vols = np.array([0.08, 0.2, 0.7, 1.8])
    for option, k in itertools.product(("call", "put"), (80.0, 100.0, 125.0)):
        price = qf.bs_price(option, 100.0, k, 0.75, 0.02, 0.01, vols)
        assert qf.implied_vol(option, price, 100.0, k, 0.75, 0.02, 0.01, method) == pytest.approx(vols, rel=1e-9)
    fs = 100.0 * math.exp(-0.01 * 0.75)
    assert math.isnan(qf.implied_vol("call", fs * 1.01, 100.0, 100.0, 0.75, 0.02, 0.01, method))
    assert math.isnan(qf.implied_vol("put", 0.0, 100.0, 100.0, 0.75, 0.02, 0.01, method))


def test_newton_implied_vol_starts_at_the_inflection_and_agrees_with_bisection_across_the_grid():
    """Manaster and Koehler (1982): from sqrt(2 |ln(S~/K~)| / T), the price's inflection in vol, Newton's iterates
    converge for any European price; at the money forward, where that start is 0, Brenner and Subrahmanyam's. On a
    grid of strikes, maturities and vols both methods give the vol back to 1e-10 wherever |d1| < 3; Newton gives
    one wherever |d1| < 5, and nan, not a wrong vol, where vega vanishes (a time value of 1e-40 and below), which
    bisection still inverts."""
    strikes, maturities = np.array([40.0, 70.0, 95.0, 100.0, 105.0, 140.0, 250.0]), np.array([0.02, 0.5, 3.0])
    k, t, vol = np.meshgrid(strikes, maturities, [0.1, 0.45, 1.2], indexing="ij")
    d1 = (np.log(100.0 / k) + (0.03 - 0.01 + vol * vol / 2) * t) / (vol * np.sqrt(t))
    for option in ("call", "put"):
        price = qf.bs_price(option, 100.0, k, t, 0.03, 0.01, vol)
        newton = qf.implied_vol(option, price, 100.0, k, t, 0.03, 0.01, "newton")
        bisection = qf.implied_vol(option, price, 100.0, k, t, 0.03, 0.01, "bisection")
        near = np.abs(d1) < 3
        assert np.abs(newton - vol)[near].max() < 1e-10 and np.abs(bisection - vol)[near].max() < 1e-10
        assert np.all(np.isfinite(newton[np.abs(d1) < 5])) and np.all(np.isnan(newton) | (np.abs(newton - vol) < 1e-7))
        far = (k == (250.0 if option == "call" else 40.0)) & (t == 0.02) & (vol == 0.45)     # a value near 1e-47
        assert np.isnan(newton[far]).all() and np.isfinite(bisection[far]).all()
    price = qf.bs_price("call", 100.0, 100.0, 0.5, 0.02, 0.02, 0.3)        # S~ = K~ exactly: the start is 0
    assert qf.implied_vol("call", price, 100.0, 100.0, 0.5, 0.02, 0.02, "newton") == pytest.approx(0.3, rel=1e-12)
    with pytest.raises(ValueError, match="method"):
        qf.implied_vol("call", price, 100.0, 100.0, 0.5, 0.03, 0.01, "brent")


def test_forward_start_price_gives_haugs_example_and_reduces_to_a_vanilla_at_start_zero():
    """Haug, The Complete Guide to Option Pricing Formulas (2007), "Forward start options" (Rubinstein 1990): S = 60,
    alpha = 1.1, start 0.25, T = 1, r = 8%, q = 4% (b = 4%), vol = 30% gives a call of 4.4064 (a remembered value, to
    four places). The test does not rest on it: S_T / S_t1 is independent of S_t1, so the price is e^{-rT} E[S_t1]
    E[(R - alpha)^+], R = S_T / S_t1 lognormal, which a quadrature on normal scores gives without any closed form
    (4.406454). Started today it is the vanilla struck at alpha S; at its start the call less the put is
    S_t1 (e^{-q(T-t1)} - alpha e^{-r(T-t1)})."""
    assert qf.forward_start_price("call", 60.0, 1.1, 0.25, 1.0, 0.08, 0.04, 0.30) == pytest.approx(4.4064, abs=1e-4)
    s, a, t1, t, r, q, v = 60.0, 1.1, 0.25, 1.0, 0.08, 0.04, 0.30
    m, sd = (r - q - v * v / 2) * (t - t1), v * math.sqrt(t - t1)
    z = np.linspace((math.log(a) - m) / sd, 12.0, 200001)              # R above alpha: a smooth integrand
    tail = np.trapezoid((np.exp(m + sd * z) - a) * phi(z), z)
    assert qf.forward_start_price("call", s, a, t1, t, r, q, v) == pytest.approx(
        math.exp(-r * t) * s * math.exp((r - q) * t1) * tail, rel=1e-8)
    for option in ("call", "put"):
        assert qf.forward_start_price(option, 80.0, 0.9, 0.0, 2.0, 0.03, 0.02, 0.25) == pytest.approx(
            qf.bs_price(option, 80.0, 72.0, 2.0, 0.03, 0.02, 0.25), rel=1e-12)
    s, a, t1, t, r, q = 80.0, 1.05, 0.5, 2.0, 0.03, 0.02
    call, put = (qf.forward_start_price(o, s, a, t1, t, r, q, 0.25) for o in ("call", "put"))
    gap = call - put
    assert gap == pytest.approx(s * math.exp(-q * t1) * (math.exp(-q * (t - t1)) - a * math.exp(-r * (t - t1))))
    with pytest.raises(ValueError):
        qf.forward_start_price("call", s, a, 3.0, t, r, q, 0.25)


def test_digital_prices_add_up_and_the_cash_digital_is_minus_the_calls_strike_derivative():
    """Reiner and Rubinstein (1991); Hull, "Binary options": cash call + cash put = e^{-rT}, asset call + asset put =
    S e^{-qT}, the call = asset call - K cash call; and Breeden and Litzenberger's first derivative, the cash call =
    -dC/dK (a strike spread's limit), the cash put = dP/dK."""
    s, k, t, r, q, v = 100.0, np.array([80.0, 100.0, 125.0]), 0.8, 0.03, 0.02, 0.3
    cc, cp = qf.digital_price("cash", "call", s, k, t, r, q, v), qf.digital_price("cash", "put", s, k, t, r, q, v)
    ac, ap = qf.digital_price("asset", "call", s, k, t, r, q, v), qf.digital_price("asset", "put", s, k, t, r, q, v)
    assert cc + cp == pytest.approx(np.full(3, math.exp(-r * t)), abs=1e-14)
    assert ac + ap == pytest.approx(np.full(3, s * math.exp(-q * t)), abs=1e-12)
    assert ac - k * cc == pytest.approx(qf.bs_price("call", s, k, t, r, q, v), abs=1e-12)
    h = 1e-4
    slope = (qf.bs_price("call", s, k + h, t, r, q, v) - qf.bs_price("call", s, k - h, t, r, q, v)) / (2 * h)
    assert cc == pytest.approx(-slope, abs=1e-7)
    slope = (qf.bs_price("put", s, k + h, t, r, q, v) - qf.bs_price("put", s, k - h, t, r, q, v)) / (2 * h)
    assert cp == pytest.approx(slope, abs=1e-7)
    assert qf.digital_price("cash", "call", s, 90.0, t, r, q, 0.0) == pytest.approx(math.exp(-r * t))


def hull_barrier(kind, s, k, h, t, r, q, v):
    """Hull, Options, Futures, and Other Derivatives, "Barrier options" (after Merton 1973 and Reiner and Rubinstein
    1991): the eight single-barrier calls and puts, continuously monitored, no rebate."""
    sd = v * math.sqrt(t)
    lam = (r - q + v * v / 2) / (v * v)
    y = math.log(h * h / (s * k)) / sd + lam * sd
    x1 = math.log(s / h) / sd + lam * sd
    y1 = math.log(h / s) / sd + lam * sd
    es, ek, p2, p22 = s * math.exp(-q * t), k * math.exp(-r * t), (h / s) ** (2 * lam), (h / s) ** (2 * lam - 2)
    N = lambda x: float(Phi(x))  # noqa: E731
    c = qf.bs_price("call", s, k, t, r, q, v)
    p = qf.bs_price("put", s, k, t, r, q, v)
    if kind == "down-and-in call":
        if h <= k:
            return es * p2 * N(y) - ek * p22 * N(y - sd)
        return c - hull_barrier("down-and-out call", s, k, h, t, r, q, v)
    if kind == "down-and-out call":
        if h <= k:
            return c - hull_barrier("down-and-in call", s, k, h, t, r, q, v)
        return es * N(x1) - ek * N(x1 - sd) - es * p2 * N(y1) + ek * p22 * N(y1 - sd)
    if kind == "up-and-in call":
        if h <= k:
            return c
        return (es * N(x1) - ek * N(x1 - sd) - es * p2 * (N(-y) - N(-y1)) + ek * p22 * (N(-y + sd) - N(-y1 + sd)))
    if kind == "up-and-out call":
        return c - hull_barrier("up-and-in call", s, k, h, t, r, q, v)
    if kind == "up-and-in put":
        if h >= k:
            return -es * p2 * N(-y) + ek * p22 * N(-y + sd)
        return p - hull_barrier("up-and-out put", s, k, h, t, r, q, v)
    if kind == "up-and-out put":
        if h >= k:
            return p - hull_barrier("up-and-in put", s, k, h, t, r, q, v)
        return -es * N(-x1) + ek * N(-x1 + sd) + es * p2 * N(-y1) - ek * p22 * N(-y1 + sd)
    if kind == "down-and-in put":
        if h >= k:
            return p
        return (-es * N(-x1) + ek * N(-x1 + sd) + es * p2 * (N(y) - N(y1)) - ek * p22 * (N(y - sd) - N(y1 - sd)))
    return p - hull_barrier("down-and-in put", s, k, h, t, r, q, v)     # down-and-out put


BARRIER_CASES = [(kind, k, h) for kind in qf.barriers.BARRIERS
                 for k, h in ((90.0, 80.0), (110.0, 80.0), (85.0, 92.0), (90.0, 120.0), (110.0, 120.0), (115.0, 108.0))
                 if (h < 100.0) == kind.startswith("down")]


@pytest.mark.parametrize("barrier_type, k, h", BARRIER_CASES)
def test_barrier_digitals_assemble_hulls_barrier_options_and_are_minus_their_strike_derivative(barrier_type, k, h):
    """The asset digital less K cash digitals is Hull's barrier option of the same kind, and the cash digital is the
    barrier option's strike derivative (-dC/dK for a call, dP/dK for a put): two checks against Hull's closed forms,
    with the barrier on either side of the strike for each direction."""
    s, t, r, q, v = 100.0, 0.75, 0.04, 0.015, 0.28
    for option in ("call", "put"):
        kind = f"{barrier_type} {option}"
        cash = qf.barrier_digital_price("cash", option, barrier_type, s, k, h, t, r, q, v)
        asset = qf.barrier_digital_price("asset", option, barrier_type, s, k, h, t, r, q, v)
        sign = 1.0 if option == "call" else -1.0
        assert sign * (asset - k * cash) == pytest.approx(hull_barrier(kind, s, k, h, t, r, q, v), abs=1e-10), kind
        dk = 1e-4
        slope = (hull_barrier(kind, s, k + dk, h, t, r, q, v) - hull_barrier(kind, s, k - dk, h, t, r, q, v)) / (2 * dk)
        assert cash == pytest.approx(-sign * slope, abs=2e-7), kind


def test_a_knock_in_and_its_knock_out_make_the_vanilla_digital_and_a_crossed_barrier_has_knocked():
    """In-out parity (Reiner and Rubinstein 1991): knock-in + knock-out = the vanilla digital; a barrier already
    crossed today leaves the out worth 0 and the in the vanilla."""
    s, k, t, r, q, v = 100.0, np.array([85.0, 100.0, 115.0]), 0.6, 0.02, 0.01, 0.4
    for payoff, option, (way, h) in itertools.product(qf.exotics.PAYOFFS, ("call", "put"), (("down", 90.0),
                                                                                           ("up", 110.0))):
        knocked = qf.barrier_digital_price(payoff, option, f"{way}-and-in", s, k, h, t, r, q, v)
        kept = qf.barrier_digital_price(payoff, option, f"{way}-and-out", s, k, h, t, r, q, v)
        assert knocked + kept == pytest.approx(qf.digital_price(payoff, option, s, k, t, r, q, v), abs=1e-13)
        assert np.all(kept >= 0) and np.all(knocked >= -1e-15)
    vanilla = qf.digital_price("cash", "call", s, 95.0, t, r, q, v)
    assert qf.barrier_digital_price("cash", "call", "down-and-out", s, 95.0, 101.0, t, r, q, v) == 0.0
    assert qf.barrier_digital_price("cash", "call", "down-and-in", s, 95.0, 101.0, t, r, q, v) == pytest.approx(vanilla)
    assert qf.barrier_digital_price("cash", "call", "up-and-out", s, 95.0, 99.0, t, r, q, v) == 0.0


# ------------------------------------------------------------------------------------------------ Asian options

TIMES12 = np.arange(1, 13) / 12.0


def test_every_asian_method_on_one_date_at_maturity_is_black_scholes_and_puts_follow_the_averages_parity():
    """With one monitoring date at T the average is S_T: the geometric, Levy's and Curran's prices are Black-Scholes
    (Haug 2007, "Asian options"). Puts come from the average's parity, call - put = e^{-rT}(E[average] - K)."""
    for method, option in itertools.product(qf.exotics.METHODS, ("call", "put")):
        assert qf.asian_price(method, option, 100.0, 104.0, [0.9], 0.9, 0.035, 0.01, 0.3) == pytest.approx(
            qf.bs_price(option, 100.0, 104.0, 0.9, 0.035, 0.01, 0.3), rel=1e-11), (method, option)
    s, k, t, r, q, v = 100.0, 97.0, 1.0, 0.035, 0.01, 0.3
    m1 = np.mean(s * np.exp((r - q) * TIMES12))
    mu = math.log(s) + (r - q - v * v / 2) * TIMES12
    eg = math.exp(mu.mean() + 0.5 * v * v * np.minimum.outer(TIMES12, TIMES12).sum() / 144)
    for method in qf.exotics.METHODS:
        call, put = (qf.asian_price(method, o, s, k, TIMES12, t, r, q, v) for o in ("call", "put"))
        forward = eg if method == "geometric" else m1
        assert call - put == pytest.approx(math.exp(-r * t) * (forward - k), abs=1e-12), method


def test_the_geometric_and_levy_prices_reach_their_continuous_average_closed_forms():
    """Densely monitored, the discrete prices reach the continuous-average formulas: Kemna and Vorst (1990) for the
    geometric average (vol/sqrt(3), drift (b - vol^2/6)/2); Levy (1992) with the continuous average's moments,
    M1 = S (e^{bT} - 1)/(bT) and M2 = 2 S^2 / (T^2 (b + vol^2)) [(e^{(2b + vol^2)T} - 1)/(2b + vol^2) - (e^{bT} - 1)/b],
    the double integral of E[S_s S_t] (Haug 2007, "Asian options"). The discrete error is O(1/n), so the limit is read
    by Richardson's extrapolation from n and 2n dates, 2 P(2n) - P(n)."""
    s, k, t, r, q, v = 100.0, 95.0, 1.0, 0.04, 0.01, 0.25
    b = r - q

    def limit(method):
        p1, p2 = (qf.asian_price(method, "call", s, k, np.arange(1, n + 1) * t / n, t, r, q, v) for n in (300, 600))
        return 2 * p2 - p1

    va, ba = v / math.sqrt(3.0), 0.5 * (b - v * v / 6.0)
    d1 = (math.log(s / k) + (ba + 0.5 * va * va) * t) / (va * math.sqrt(t))
    kemna_vorst = (s * math.exp((ba - r) * t) * float(Phi(d1))
                   - k * math.exp(-r * t) * float(Phi(d1 - va * math.sqrt(t))))
    assert limit("geometric") == pytest.approx(kemna_vorst, rel=2e-5)
    m1 = s * (math.exp(b * t) - 1.0) / (b * t)
    m2 = 2 * s * s / (t * t * (b + v * v)) * ((math.exp((2 * b + v * v) * t) - 1) / (2 * b + v * v)
                                             - (math.exp(b * t) - 1) / b)
    w = math.sqrt(math.log(m2) - 2 * math.log(m1))
    d1 = (math.log(m1 / k) + 0.5 * w * w) / w
    levy = math.exp(-r * t) * (m1 * float(Phi(d1)) - k * float(Phi(d1 - w)))
    assert limit("levy") == pytest.approx(levy, rel=2e-5)


def test_curran_is_the_integral_of_the_conditional_average_above_its_k_hat():
    """Curran (1994): e^{-rT} int over ln G above ln K-hat of (E[A | G] - K), the conditional law of the ln S(t_i)
    given ln G read here from their joint covariance vol^2 min(t_i, t_j) and integrated on a grid. With K-hat the
    root of E[A | G = K-hat] = K the integrand is (E[A|G] - K)^+, so the root's price is the largest over thresholds
    (the Taylor K-hat's lies below it) and a lower bound of the true price (conditional Jensen), above the geometric
    price (A >= G). The first-order K-hat = 2K - E[A | G = K]; where it falls below 0 the call is e^{-rT}(M1 - K)."""
    r, q = 0.03, 0.0
    for s, k, t, v, close in ((100.0, 105.0, 1.0, 0.35, 1e-5), (100.0, 40.0, 4.0, 1.0, 1e-3)):
        times = TIMES12 * t
        n, b = times.size, r - q
        cov = v * v * np.minimum.outer(times, times)
        mu = math.log(s) + (b - v * v / 2) * times
        weights = np.full(n, 1.0 / n)
        mu_g, var_g, c_xg = weights @ mu, weights @ cov @ weights, cov @ weights
        def given(y):
            return np.exp(mu[None, :] + np.outer(y - mu_g, c_xg / var_g)
                          + 0.5 * (np.diag(cov) - c_xg ** 2 / var_g)).mean(axis=1)

        def above(low):
            """e^{-rT} int_low^inf (E[A | ln G = y] - K) density(y) dy, on a grid starting at `low`."""
            y = np.linspace(low, mu_g + 12 * math.sqrt(var_g), 40001)
            density = np.exp(-0.5 * (y - mu_g) ** 2 / var_g) / math.sqrt(2 * math.pi * var_g)
            return math.exp(-r * t) * np.trapezoid((given(y) - k) * density, y)

        grid = np.linspace(mu_g - 12 * math.sqrt(var_g), mu_g + 12 * math.sqrt(var_g), 40001)
        crossing = float(np.interp(k, given(grid), grid))       # E[A | G] increases: its root by interpolation
        root = qf.asian_price("curran_root", "call", s, k, times, t, r, q, v)
        assert root == pytest.approx(above(crossing), rel=1e-7)
        k_hat = 2 * k - float(given(np.array([math.log(k)]))[0])
        taylor = above(math.log(k_hat))
        found = qf.asian_price("curran_taylor", "call", s, k, times, t, r, q, v)
        assert found == pytest.approx(taylor, rel=1e-6) and found <= root
        assert found == pytest.approx(root, rel=close)
        assert qf.asian_price("geometric", "call", s, k, times, t, r, q, v) < root
    assert root - found > 1e-4 * root                               # the two K-hats differ where vol^2 T is large
    assert qf.asian_price("curran_root", "call", s, [80.0, k], times, t, r, q, v)[1] == pytest.approx(root)
    deep = qf.asian_price("curran_taylor", "call", s, 5.0, times, t, r, q, v)       # K-hat below 0
    assert deep == pytest.approx(math.exp(-r * t) * (np.mean(s * np.exp(b * times)) - 5.0), rel=1e-12)
    assert qf.asian_price("levy", "call", 100.0, 105.0, TIMES12, 1.0, r, q, 0.35) == pytest.approx(
        qf.asian_price("curran_root", "call", 100.0, 105.0, TIMES12, 1.0, r, q, 0.35), rel=0.02)


# ------------------------------------------------------------------------------------------------ first passage


def test_hit_probability_is_the_reflection_principles_and_the_integral_of_the_first_passage_density():
    """With no drift, P(max W >= b) = 2 N(-b/(vol sqrt T)) (the reflection principle; Shreve, Stochastic Calculus for
    Finance II, 3.7). With drift, the first-passage density (the inverse Gaussian) integrates to the hit probability,
    for a level above and below 0; drift away from the level leaves the defective total e^{2 d |b| / vol^2}."""
    assert qf.hit_probability(0.2, 1.5, 0.0, 0.3) == pytest.approx(2 * float(Phi(-0.2 / (0.3 * math.sqrt(1.5)))))
    assert qf.hit_probability(0.0, 1.0, 0.1, 0.2) == 1.0
    grid = np.linspace(0.0, 2.0, 20001)[1:]
    for level, drift in ((0.15, 0.05), (0.15, -0.12), (-0.1, 0.08), (-0.25, -0.3)):
        mass = np.trapezoid(np.concatenate([[0.0], qf.first_passage_pdf(level, grid, drift, 0.25)]),
                            np.concatenate([[0.0], grid]))
        assert mass == pytest.approx(qf.hit_probability(level, 2.0, drift, 0.25), abs=2e-6), (level, drift)
    toward = 0.12 * np.sign(0.15)
    long = np.linspace(0.0, 400.0, 400001)[1:]
    away = np.trapezoid(qf.first_passage_pdf(0.15, long, -toward, 0.25), long)
    assert away == pytest.approx(math.exp(-2 * toward * 0.15 / 0.25 ** 2), abs=1e-4)
    assert qf.first_passage_pdf(0.15, 0.0, 0.1, 0.25) == 0.0
    # a level a hair above 0, where the two reflection terms, each rounded, sum to 1 + 2^-52 unclipped
    near = qf.hit_probability([1.8032614982113497e-19, 1.2731743243043884e-17, 2.0433139590982746e-19],
                              [0.020167791681543177, 86.10733533961918, 27.657713421895522],
                              [0.38776509767107453, 0.01186197814431578, 0.057820415154294494],
                              [0.04170392330081793, 0.0957632365446124, 0.2604659006348866])
    assert np.all(near <= 1.0) and near == pytest.approx(1.0)


def test_first_passage_mean_is_the_inverse_gaussians_mean():
    """E[tau_b] = |b| / d for a drift d toward the level (the inverse Gaussian's mean), numerically int t f(t) dt;
    infinite with no drift or drift away."""
    grid = np.linspace(0.0, 250.0, 500001)[1:]
    for level, drift in ((0.2, 0.15), (-0.3, -0.2)):
        assert np.trapezoid(grid * qf.first_passage_pdf(level, grid, drift, 0.3), grid) == pytest.approx(
            qf.first_passage_mean(level, drift, 0.3), rel=1e-5)
        assert qf.first_passage_mean(level, drift, 0.3) == pytest.approx(abs(level) / abs(drift))
    assert qf.first_passage_mean(0.2, 0.0, 0.3) == math.inf and qf.first_passage_mean(0.2, -0.1, 0.3) == math.inf


def test_terminal_and_max_cdf_differentiates_to_the_joint_density_and_closes_on_the_maximums_law():
    """The joint density of (X_T, M_T), f(x, m) = 2 (2m - x) / (vol^3 T sqrt(2 pi T)) exp(d x / vol^2 - d^2 T /
    (2 vol^2) - (2m - x)^2 / (2 vol^2 T)) for m >= max(x, 0) (Shreve, Stochastic Calculus for Finance II, theorem 7.2.1,
    with drift by Girsanov), is the CDF's mixed derivative; at x = m the CDF is P(M_T <= m)."""
    t, d, v = 0.8, 0.07, 0.3
    for x, m in ((-0.1, 0.2), (0.05, 0.3), (0.1, 0.12)):
        e = 1e-4
        mixed = (qf.terminal_and_max_cdf(x + e, m + e, t, d, v) - qf.terminal_and_max_cdf(x + e, m - e, t, d, v)
                 - qf.terminal_and_max_cdf(x - e, m + e, t, d, v) + qf.terminal_and_max_cdf(x - e, m - e, t, d, v))
        joint = (2 * (2 * m - x) / (v ** 3 * t * math.sqrt(2 * math.pi * t))
                 * math.exp(d * x / v ** 2 - d * d * t / (2 * v * v) - (2 * m - x) ** 2 / (2 * v * v * t)))
        assert mixed / (4 * e * e) == pytest.approx(joint, rel=1e-4), (x, m)
    assert qf.terminal_and_max_cdf(0.25, 0.25, t, d, v) == pytest.approx(1 - qf.hit_probability(0.25, t, d, v))
    assert qf.terminal_and_max_cdf(0.5, 0.25, t, d, v) == pytest.approx(1 - qf.hit_probability(0.25, t, d, v))
    plain = float(Phi((-0.1 - d * t) / (v * math.sqrt(t))))
    assert qf.terminal_and_max_cdf(-0.1, 50.0, t, d, v) == pytest.approx(plain)
    assert qf.terminal_and_max_cdf(-0.1, -0.05, t, d, v) == 0.0


# ------------------------------------------------------------------------------------------------ copulas

Z = np.linspace(-9.0, 9.0, 3601)
DENSITIES = [("gaussian", lambda u, v: qf.gaussian_copula_logpdf(u, v, 0.6)),
             ("t", lambda u, v: qf.t_copula_logpdf(u, v, -0.4, 4.5)),
             ("clayton", lambda u, v: qf.clayton_copula_logpdf(u, v, 2.5)),
             ("gumbel", lambda u, v: qf.gumbel_copula_logpdf(u, v, 1.8))]


@pytest.mark.parametrize("family, logpdf", DENSITIES)
def test_each_copula_density_integrates_to_one_along_every_margin(family, logpdf):
    """A copula density's every conditional slice integrates to one: int_0^1 c(u, v) dv = 1 for each u (uniform
    margins; Nelsen, An Introduction to Copulas, 2.2), on normal scores v = N(z), dv = phi(z) dz, which reach the
    corners."""
    v = Phi(Z)
    inside = (v > 0) & (v < 1)
    for u in (0.03, 0.4, 0.9):
        mass = np.trapezoid(np.exp(logpdf(u, v[inside])) * phi(Z[inside]), Z[inside])
        assert mass == pytest.approx(1.0, abs=2e-4), (family, u)


def test_the_elliptical_copula_densities_are_their_joint_density_over_the_margins():
    """Sklar's theorem (Nelsen, 2.10): c(u, v) = f(x, y) / (f(x) f(y)) at the margins' quantiles, with the bivariate
    normal and the bivariate t (McNeil, Frey and Embrechts, Quantitative Risk Management, 3.2) written out here."""
    u, w = np.array([0.1, 0.5, 0.97]), np.array([0.8, 0.3, 0.95])
    rho = 0.35
    x, y = (np.array([math.sqrt(2) * _erfinv(2 * a - 1) for a in side]) for side in (u, w))
    normal2 = (np.exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho)))
               / (2 * math.pi * math.sqrt(1 - rho ** 2)))
    assert np.exp(qf.gaussian_copula_logpdf(u, w, rho)) == pytest.approx(normal2 / (phi(x) * phi(y)), rel=1e-6)
    nu = 3.5
    x, y = stdtrit(nu, u), stdtrit(nu, w)
    t2 = (math.gamma((nu + 2) / 2) / (math.gamma(nu / 2) * nu * math.pi * math.sqrt(1 - rho * rho))
          * (1 + (x * x - 2 * rho * x * y + y * y) / (nu * (1 - rho * rho))) ** (-(nu + 2) / 2))
    head = math.gamma((nu + 1) / 2) / (math.gamma(nu / 2) * math.sqrt(nu * math.pi))
    t1 = lambda z: head * (1 + z * z / nu) ** (-(nu + 1) / 2)  # noqa: E731
    assert np.exp(qf.t_copula_logpdf(u, w, rho, nu)) == pytest.approx(t2 / (t1(x) * t1(y)), rel=1e-10)


def _erfinv(y):
    """The inverse error function by bisection on math.erf: the test's own normal quantile."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.erf(mid) < y else (lo, mid)
    return 0.5 * (lo + hi)


def test_the_archimedean_densities_are_their_copulas_mixed_derivatives():
    """c = d^2 C / du dv, by central differences of Nelsen's copulas: Clayton C = (u^-t + v^-t - 1)^(-1/t) and Gumbel
    C = exp(-((-ln u)^t + (-ln v)^t)^(1/t)) (Nelsen, An Introduction to Copulas, table 4.1, families 1 and 4)."""
    clayton = lambda u, v, t: (u ** -t + v ** -t - 1) ** (-1 / t)  # noqa: E731
    gumbel = lambda u, v, t: math.exp(-((-math.log(u)) ** t + (-math.log(v)) ** t) ** (1 / t))  # noqa: E731
    for cdf, logpdf, theta in ((clayton, qf.clayton_copula_logpdf, 1.7), (gumbel, qf.gumbel_copula_logpdf, 2.2)):
        for u, v in ((0.2, 0.7), (0.55, 0.5), (0.9, 0.15)):
            e = 1e-4
            mixed = (cdf(u + e, v + e, theta) - cdf(u + e, v - e, theta) - cdf(u - e, v + e, theta)
                     + cdf(u - e, v - e, theta)) / (4 * e * e)
            assert math.exp(logpdf(u, v, theta)) == pytest.approx(mixed, rel=1e-5), (logpdf.__name__, u, v)


def test_copula_param_from_tau_inverts_kendalls_tau_of_each_family():
    """Kendall's tau of an elliptical copula is (2/pi) arcsin(rho) (Lindskog, McNeil and Schmock 2003); of an
    Archimedean copula with generator phi, 1 + 4 int_0^1 phi/phi' (Nelsen, theorem 5.1.4), integrated here with
    Clayton's phi = (t^-theta - 1)/theta and Gumbel's (-ln t)^theta."""
    for rho in (-0.6, 0.2, 0.85):
        tau = 2 / math.pi * math.asin(rho)
        assert qf.copula_param_from_tau("gaussian", tau) == pytest.approx(rho)
        assert qf.copula_param_from_tau("t", tau) == pytest.approx(rho)
    t = np.linspace(0.0, 1.0, 200001)[1:-1]
    for tau in (0.15, 0.5, 0.7):
        theta = qf.copula_param_from_tau("clayton", tau)
        ratio = -(t ** -theta - 1) / theta * t ** (theta + 1)
        assert 1 + 4 * np.trapezoid(ratio, t) == pytest.approx(tau, abs=1e-5)
        theta = qf.copula_param_from_tau("gumbel", tau)
        ratio = t * np.log(t) / theta
        assert 1 + 4 * np.trapezoid(ratio, t) == pytest.approx(tau, abs=1e-5)
    for tau in (-0.45, 0.15, 0.5, 0.7):     # Frank's phi = -ln((e^-theta t - 1) / (e^-theta - 1)), any tau
        theta = qf.copula_param_from_tau("frank", tau)
        phi = -np.log(np.expm1(-theta * t) / np.expm1(-theta))
        ratio = phi * np.expm1(-theta * t) / (theta * np.exp(-theta * t))
        assert 1 + 4 * np.trapezoid(ratio, t) == pytest.approx(tau, abs=1e-5)
    assert qf.copula_param_from_tau("frank", 0.0) == 0.0
    assert qf.copula_param_from_tau("frank", [0.3, -0.3]) == pytest.approx(np.array([1, -1]) * qf.copula_param_from_tau(
        "frank", 0.3), rel=1e-14)
    with pytest.raises(ValueError):
        qf.copula_param_from_tau("clayton", -0.2)


# ------------------------------------------------------------------------------------------------ OU with jumps


def test_ou_jump_moments_solve_the_sdes_moment_equations_and_settle_on_the_stationary_law():
    """From dX = kappa (theta - X) dt + sigma dW + J dN (Cont and Tankov 2004, 15.1): the mean solves dm/dh = kappa
    (theta - m) + lambda mu_J and the variance dV/dh = -2 kappa V + sigma^2 + lambda (mu_J^2 + s_J^2), from m = x0 and
    V = 0 (the jumps' variance rate lambda E[J^2], Levy-Khintchine). Checked by central differences along h; h = inf
    and a long horizon give the stationary moments; with lambda = 0 the OU's (Uhlenbeck and Ornstein 1930)."""
    args = dict(x0=1.3, kappa=2.5, theta=0.4, sigma=0.6, jump_rate=7.0, jump_mean=-0.08, jump_std=0.15)
    ex2 = 0.08 ** 2 + 0.15 ** 2
    assert qf.ou_jump_moments(horizon=0.0, **args) == (pytest.approx(1.3), pytest.approx(0.0))
    for h in (0.01, 0.3, 1.7):
        e = 1e-6
        (m0, v0), (m1, v1), (mm, vm) = (qf.ou_jump_moments(horizon=x, **args) for x in (h - e, h + e, h))
        assert (m1 - m0) / (2 * e) == pytest.approx(2.5 * (0.4 - mm) + 7.0 * -0.08, rel=1e-6)
        assert (v1 - v0) / (2 * e) == pytest.approx(-2 * 2.5 * vm + 0.36 + 7.0 * ex2, rel=1e-6)
    stationary = qf.ou_jump_moments(horizon=np.inf, **args)
    assert stationary == (pytest.approx(0.4 + 7.0 * -0.08 / 2.5), pytest.approx((0.36 + 7.0 * ex2) / 5.0))
    assert qf.ou_jump_moments(horizon=40.0, **args) == (pytest.approx(stationary[0]), pytest.approx(stationary[1]))
    plain = qf.ou_jump_moments(**{**args, "jump_rate": 0.0}, horizon=0.5)
    assert plain == (pytest.approx(0.4 + 0.9 * math.exp(-1.25)), pytest.approx(0.36 * (1 - math.exp(-2.5)) / 5.0))


def test_ou_from_ar1_inverts_the_exact_discretisation():
    """The OU's exact transition (Brigo and Mercurio, Interest Rate Models, 3.2.1): b = e^{-kappa dt}, a = theta
    (1 - b), Var(e) = sigma^2 (1 - e^{-2 kappa dt})/(2 kappa), inverted back to (kappa, theta, sigma); at small dt the
    Euler reading sqrt(Var(e)/dt) is its limit."""
    for kappa, theta, sigma, dt in ((0.8, 0.03, 0.01, 1 / 52), (6.0, -1.2, 0.9, 0.25)):
        b = math.exp(-kappa * dt)
        found = qf.ou_from_ar1(theta * (1 - b), b, sigma ** 2 * (1 - b * b) / (2 * kappa), dt)
        assert found == (pytest.approx(kappa), pytest.approx(theta), pytest.approx(sigma))
    b = math.exp(-0.5 * 1e-4)
    resid = 0.2 ** 2 * (1 - b * b) / (2 * 0.5)
    assert qf.ou_from_ar1(0.0, b, resid, 1e-4)[2] == pytest.approx(math.sqrt(resid / 1e-4), rel=1e-4)
    with pytest.raises(ValueError):
        qf.ou_from_ar1(0.1, 1.02, 0.01, 1 / 252)


# ------------------------------------------------------------------------------------- implied-vol approximations


def atm_forward_call(v):
    """A call at the money forward (S~ = K~ = 1) whose total volatility vol sqrt(T) is v: C = 2 N(v/2) - 1."""
    return 2 * float(Phi(v / 2)) - 1


def test_brenner_subrahmanyam_is_first_order_and_li_solves_the_cubic():
    """At the money forward, C/S~ = 2N(v/2) - 1 = (v - v^3/24 + ...)/sqrt(2 pi): Brenner and Subrahmanyam (1988)
    keep the first term, so their error is -v^3/24 to leading order; Li (2005) solves alpha = v - v^3/24, so his v
    satisfies the cubic exactly and errs by the series' next term, v^5/640 to leading order (2N(x) - 1 = sqrt(2/pi)
    (x - x^3/6 + x^5/40 - ...)); past arccos's domain Li's formula has no value."""
    for v in (0.05, 0.1, 0.2):
        c = atm_forward_call(v)
        found = qf.brenner_subrahmanyam_vol(c, 1.0, 1.0)
        assert (found - v) / (-v ** 3 / 24) == pytest.approx(1.0, abs=0.01 + v)
        assert qf.brenner_subrahmanyam_vol(3 * c, 3.0, 4.0) == pytest.approx(found / 2)
    for v in (0.1, 0.3, 0.6):
        c = atm_forward_call(v)
        found = qf.li_atm_vol(c, 1.0, 1.0)
        assert found - found ** 3 / 24 == pytest.approx(math.sqrt(2 * math.pi) * c, abs=1e-12)
        assert found - v == pytest.approx(v ** 5 / 640, rel=0.05)
        assert qf.li_atm_vol(5 * c, 5.0, 2.25) == pytest.approx(found / 1.5)
    assert math.isnan(qf.li_atm_vol(atm_forward_call(4.0), 1.0, 1.0))


def test_corrado_miller_and_hallerbach_are_brenner_subrahmanyam_at_the_money_and_close_off_it():
    """At S~ = K~ both reduce to Brenner and Subrahmanyam (Corrado and Miller 1996). Off the money, on calls priced
    by bs_price within 8% of the forward and a total volatility vol sqrt(T) up to 0.5, both invert within 0.01 of
    vol, and Hallerbach's (2004) refinement errs less than Corrado and Miller's at the worst point, as his paper
    reports."""
    c = atm_forward_call(0.3)
    for variant in qf.impvol.VARIANTS:
        found = qf.corrado_miller_vol(c, 1.0, 1.0, 1.0, variant)
        assert found == pytest.approx(qf.brenner_subrahmanyam_vol(c, 1.0, 1.0))
    worst = {}
    for variant in qf.impvol.VARIANTS:
        errors = []
        for vol, t, m in itertools.product((0.15, 0.25, 0.35), (0.5, 1.0, 2.0), (0.92, 0.97, 1.03, 1.08)):
            s, r, q = 100.0, 0.03, 0.01
            k = m * s * math.exp((r - q) * t)
            price = qf.bs_price("call", s, k, t, r, q, vol)
            found = qf.corrado_miller_vol(price, s * math.exp(-q * t), k * math.exp(-r * t), t, variant)
            errors.append(abs(found - vol))
        worst[variant] = max(errors)
    assert worst["corrado_miller"] < 0.01 and worst["hallerbach"] < 0.01
    assert worst["hallerbach"] < worst["corrado_miller"]


# ------------------------------------------------------------------------------------------------ realized measures


def noisy_days(days, n, iv, noise_sd, seed):
    """Log prices of `days` days: Brownian motion of integrated variance iv over n steps plus i.i.d. noise."""
    rng = np.random.default_rng(seed)
    efficient = np.cumsum(rng.normal(0.0, math.sqrt(iv / n), (days, n + 1)), axis=1)
    return efficient + rng.normal(0.0, noise_sd, (days, n + 1))


def test_the_noise_variance_and_the_corrected_rv_follow_e_rv_equals_iv_plus_2n_noise():
    """Bandi and Russell (2006): with i.i.d. noise of variance w2, E[RV_n] = IV + 2 n w2, so sum r^2/(2n) at the
    highest frequency has mean w2 + IV/(2n), and RV_n - 2 n w2 is unbiased for IV at a sparse n where the raw RV is
    not (Zhang, Mykland and Ait-Sahalia 2005)."""
    iv, sd = 1e-4, 4e-4
    prices = noisy_days(40, 4680, iv, sd, seed=7)
    found = [qf.bandi_russell_noise_var(np.diff(day)) for day in prices]
    assert np.mean(found) == pytest.approx(sd * sd + iv / (2 * 4680), rel=0.01)
    sparse = prices[:, ::60]
    raw = np.sum(np.diff(sparse, axis=1) ** 2, axis=1)
    corrected = qf.noise_corrected_rv(raw, 78, sd * sd)
    se = np.std(corrected) / math.sqrt(40)
    assert abs(np.mean(corrected) - iv) < 3.5 * se and abs(np.mean(raw) - iv) > 6 * se
    assert qf.noise_corrected_rv(2e-4, [10, 100], 1e-7) == pytest.approx([1.98e-4, 1.8e-4])


def test_bipower_variation_estimates_the_integrated_variance_through_a_jump():
    """Barndorff-Nielsen and Shephard (2004): (pi/2) sum |r_j||r_{j-1}| -> IV, a jump changing it not at all in the
    limit while the RV takes the jump's square; the raw sum and the n/(n-1) factor of Huang and Tauchen (2005)."""
    assert qf.bipower_variation([0.01, -0.02, 0.03], "none") == pytest.approx(0.0008)
    assert qf.bipower_variation([0.01, -0.02, 0.03], "pi/2") == pytest.approx(0.0008 * math.pi / 2)
    assert qf.bipower_variation([0.01, -0.02, 0.03], "pi/2*n/(n-1)") == pytest.approx(0.0008 * math.pi / 2 * 1.5)
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, math.sqrt(1e-4 / 20000), 20000)
    r[9000] += 0.02
    assert qf.bipower_variation(r, "pi/2") == pytest.approx(1e-4, rel=0.04)
    assert np.sum(r * r) == pytest.approx(1e-4 + 4e-4, rel=0.04)
    with pytest.raises(ValueError):
        qf.bipower_variation(r, "two")


# ------------------------------------------------------------------------------------------------ hidden Markov models

HMM_ARGS = dict(means=[-0.4, 0.7], stds=[0.5, 1.1], transition=[[0.85, 0.15], [0.3, 0.7]], initial=[0.6, 0.4])


def brute_force(x, means, stds, transition, initial):
    """Every state path's joint probability summed: P(x), each P(s_t = k | x) and each P(s_t = i, s_{t+1} = j | x)."""
    means, stds, a, p = map(np.asarray, (means, stds, transition, initial))
    k = means.size
    dens = np.exp(-0.5 * ((np.asarray(x)[:, None] - means) / stds) ** 2) / (stds * math.sqrt(2 * math.pi))
    total, post, pair = 0.0, np.zeros((len(x), k)), np.zeros((k, k))
    for path in itertools.product(range(k), repeat=len(x)):
        joint = p[path[0]] * dens[0, path[0]]
        for t in range(1, len(x)):
            joint *= a[path[t - 1], path[t]] * dens[t, path[t]]
        total += joint
        for t, state in enumerate(path):
            post[t, state] += joint
        for t in range(len(x) - 1):
            pair[path[t], path[t + 1]] += joint
    return math.log(total), post / total, pair / total


def test_hmm_smooth_is_the_sum_over_every_state_path():
    """Rabiner (1989), III-A: forward-backward computes P(x) and the posteriors without enumerating the K^T paths;
    here they are enumerated, two states over seven observations and three over four."""
    x = [0.3, -1.1, 0.9, 2.2, -0.2, 0.05, 1.4]
    loglik, post, pair = qf.hmm_smooth(x, **HMM_ARGS)
    want = brute_force(x, **HMM_ARGS)
    assert loglik == pytest.approx(want[0], rel=1e-12)
    assert post == pytest.approx(want[1], abs=1e-12) and pair == pytest.approx(want[2], abs=1e-12)
    three = dict(means=[-1.0, 0.0, 1.5], stds=[0.4, 0.8, 0.6], initial=[0.2, 0.5, 0.3],
                 transition=[[0.7, 0.2, 0.1], [0.25, 0.5, 0.25], [0.05, 0.15, 0.8]])
    found, want = qf.hmm_smooth(x[:4], **three), brute_force(x[:4], **three)
    for got, wanted in zip(found, want):
        assert got == pytest.approx(wanted, rel=1e-10, abs=1e-14)


def test_baum_welch_makes_rabiners_m_step_raises_the_likelihood_and_stops_by_its_rule():
    """Rabiner (1989), III-C and IV-A: one iteration's re-estimates from the E-step's posteriors; EM never lowers the
    likelihood (Baum et al. 1970; Dempster, Laird and Rubin 1977); the stop rules and the kept initial distribution."""
    rng = np.random.default_rng(3)
    states = (np.cumsum(rng.random(300) < 0.08) % 2).astype(int)
    x = np.where(states == 0, rng.normal(-0.3, 0.6, 300), rng.normal(0.9, 1.2, 300))
    _, gamma, xi = qf.hmm_smooth(x, **HMM_ARGS)
    one = qf.hmm_baum_welch(x, **HMM_ARGS, max_iter=1, tol=1e-12, stop="means", update_initial=True)
    means = gamma.T @ x / gamma.sum(0)
    assert one["means"] == pytest.approx(means, rel=1e-12)
    assert one["stds"] == pytest.approx(np.sqrt((gamma * (x[:, None] - means) ** 2).sum(0) / gamma.sum(0)), rel=1e-12)
    assert one["transition"] == pytest.approx(xi / xi.sum(1, keepdims=True), rel=1e-12)
    assert one["initial"] == pytest.approx(gamma[0]) and (one["iterations"], one["converged"]) == (1, False)
    assert one["loglik"] == pytest.approx(qf.hmm_smooth(x, one["means"], one["stds"], one["transition"],
                                                        one["initial"])[0])
    logliks = [qf.hmm_baum_welch(x, **HMM_ARGS, max_iter=n, tol=1e-12, stop="loglik", update_initial=False)["loglik"]
               for n in range(1, 5)]
    assert all(b >= a - 1e-9 for a, b in zip(logliks, logliks[1:])) and logliks[-1] > logliks[0]
    rises = np.diff([qf.hmm_smooth(x, **HMM_ARGS)[0]] + logliks)      # each E-step's log-likelihood over the last
    tol = float(rises[2]) * 1.01                                          # met first by the fourth iteration's E-step
    assert np.all(rises[:2] >= tol)
    by_loglik = qf.hmm_baum_welch(x, **HMM_ARGS, max_iter=50, tol=tol, stop="loglik", update_initial=False)
    assert (by_loglik["iterations"], by_loglik["converged"]) == (4, True)
    kept = qf.hmm_baum_welch(x, **HMM_ARGS, max_iter=200, tol=1e-3, stop="means", update_initial=False)
    assert kept["converged"] and kept["iterations"] < 200 and list(kept["initial"]) == HMM_ARGS["initial"]
    loose = qf.hmm_baum_welch(x, **HMM_ARGS, max_iter=50, tol=10.0, stop="means", update_initial=False)
    assert (loose["iterations"], loose["converged"]) == (1, True)
    with pytest.raises(ValueError):
        qf.hmm_smooth(x, means=[0, 1], stds=[1, 1], transition=[[0.5, 0.6], [0.5, 0.5]], initial=[0.5, 0.5])
    # The chain cannot leave state 0; its Gaussian log density remains finite even at a distant observation.
    loglik, post, pair = qf.hmm_smooth([0.0, 100.0], means=[0, 100], stds=[1, 1],
                                      transition=[[1, 0], [0, 1]], initial=[1, 0])
    assert loglik == pytest.approx(-math.log(2 * math.pi) - 5000)
    assert post == pytest.approx(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert pair == pytest.approx(np.array([[1.0, 0.0], [0.0, 0.0]]))
