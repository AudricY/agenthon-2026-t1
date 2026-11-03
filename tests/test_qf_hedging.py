"""qf.delta_hedge_pnl (runs/lanes/lib-montecarlo1-prompt.md; docs/research/library-syllabus-1011.md rank 34): one
hedge worked by hand in the layout of Hull's delta-hedging tables ("Delta hedging", Options, Futures, and Other
Derivatives); the identity that a self-financing hedge's present value averages premium - Black-Scholes price on
risk-neutral paths at any rebalancing frequency (cash dividends by the escrowed model too); put-call parity path by
path; the cost, side, settlement and frequency conventions against each other; and, slow, Derman and Kamal's (1999)
sqrt(pi/4) vol vega / sqrt(N) spread on Hull's example option (S 49, K 50, r 5%, vol 20%, 20 weeks, drift 13%) with its
1/sqrt(N) fall (Boyle and Emanuel 1980). Seeds fixed; no unit's values."""
import math

import numpy as np
import pytest
from scipy.stats import norm

import qf

pytestmark = pytest.mark.fast

S0, K, R, Q, VOL, T = 49.0, 50.0, 0.05, 0.02, 0.2, 20 / 52


def hedge(paths, times, option="call", premium=None, every=1, cost=0.0, rule="proportional", compounding="continuous",
          settlement="cash", position="short", dividends=None, vol=VOL, q=Q, rate=R):
    paid = qf.bs_price(option, S0, K, T, rate, q, VOL) if premium is None else premium
    return qf.delta_hedge_pnl(paths=paths, times=times, option=option, strike=K, rate=rate, financing_rate=rate,
                              dividend_yield=q,
                              cash_dividends=dividends, hedge_vol=vol, premium=paid, position=position,
                              rebalance_every=every, cost=cost, cost_rule=rule, compounding=compounding,
                              settlement=settlement)


def risk_neutral(steps, n, seed, q=Q):
    times = np.linspace(0.0, T, steps + 1)
    return qf.gbm_paths(S0, R - q, VOL, times, n, seed, "arithmetic", False), times


@pytest.mark.parametrize("prices, published_cost", [
    ([49.00, 48.12, 47.37, 50.25, 51.75, 53.12, 53.00, 51.87, 51.38, 53.00, 49.88, 48.50, 49.88, 50.37,
      52.13, 51.88, 52.87, 54.87, 54.62, 55.87, 57.25], 263_300.0),
    ([49.00, 49.75, 52.00, 50.00, 48.38, 48.25, 48.75, 49.63, 48.25, 48.25, 51.12, 51.50, 49.88, 49.88,
      48.75, 47.50, 48.00, 46.25, 48.13, 46.63, 48.12], 256_600.0),
])
def test_hulls_published_weekly_hedging_costs(prices, published_cost):
    """Hull, Risk Management and Financial Institutions, 6th ed. (2023), Tables 15.2–15.3, pp. 325–326:
    https://elibrary.nib.edu.kh/book/BNK-0016.pdf . These are the textbook's paths, not a task's data. The tables
    round deltas to three decimals and cash to $100; full-precision deltas and continuous interest differ by less
    than $100 per 100,000 options. A zero premium makes minus terminal P&L the hedge's cost."""
    found = qf.delta_hedge_pnl([prices], np.arange(21) / 52, "call", 50.0, 0.05, 0.05, 0.0, None, 0.2, 0.0,
                               "short", 1, 0.0, "proportional", "continuous", "physical")
    assert -100_000 * found["pnl"][0] == pytest.approx(published_cost, abs=100.0)


def test_one_step_hedge_worked_by_hand():
    """Sold at 10.4506 (Black-Scholes, S = K = 100, r 5%, vol 20%, a year), hedged with N(0.35) = 0.636831 shares;
    a year on at S = 110 the cash is (10.4506 - 63.6831) e^{0.05} + 0.636831 x 110 - 10, with 0.1% proportional costs
    on the purchase (63.6831 x 0.001) and on the sale (70.0514 x 0.001)."""
    premium = qf.bs_price("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2)
    delta = norm.cdf(0.35)
    found = qf.delta_hedge_pnl(paths=[[100.0, 110.0]], times=[0.0, 1.0], option="call", strike=100.0, rate=0.05,
                               financing_rate=0.05,
                               dividend_yield=0.0, cash_dividends=None, hedge_vol=0.2, premium=premium, position="short",
                               rebalance_every=1, cost=0.001, cost_rule="proportional", compounding="continuous",
                               settlement="cash")
    buy, sell = 0.001 * delta * 100.0, 0.001 * delta * 110.0
    by_hand = (premium - delta * 100.0 - buy) * math.exp(0.05) + delta * 110.0 - sell - 10.0
    assert found["pnl"][0] == pytest.approx(by_hand, rel=1e-12)
    assert found["costs"][0] == pytest.approx(buy + sell, rel=1e-12) and found["rebalances"] == 0
    assert found["present_value"][0] == pytest.approx(by_hand * math.exp(-0.05), rel=1e-12)


def test_one_step_hedge_with_a_cash_dividend_worked_by_hand():
    """A dividend of 2 at half a year: the delta is Black-Scholes's on 100 - 2 e^{-0.05 x 0.5} (the escrowed spot),
    the dividend is paid on the shares held and earns interest until expiry, and the close of 98 is after the drop."""
    premium = 9.0
    escrowed = 100.0 - 2.0 * math.exp(-0.025)
    delta = norm.cdf((math.log(escrowed / 100.0) + 0.07) / 0.2)
    found = qf.delta_hedge_pnl(paths=[[100.0, 98.0]], times=[0.0, 1.0], option="call", strike=100.0, rate=0.05,
                               financing_rate=0.05,
                               dividend_yield=0.0, cash_dividends=[(0.5, 2.0)], hedge_vol=0.2, premium=premium,
                               position="short", rebalance_every=1, cost=0.0, cost_rule="proportional",
                               compounding="continuous", settlement="cash")
    by_hand = (premium - delta * 100.0) * math.exp(0.05) + delta * 2.0 * math.exp(0.025) + delta * 98.0
    assert found["pnl"][0] == pytest.approx(by_hand, rel=1e-12) and found["dividends"][0] == pytest.approx(2 * delta)


@pytest.mark.parametrize("compounding", ["continuous", "simple"])
def test_cash_account_uses_its_financing_rate_and_receives_each_dividend_on_its_payment_date(compounding):
    """Hull's cash-account bookkeeping, derived directly: initial cash earns its financing rate for the year;
    the dividend at 0.4 earns that rate for the remaining 0.6. The escrowed Black-Scholes delta still uses the
    pricing rate 0.03. This is independent of the implementation's cash-account loop."""
    rate, financing, dividend, payment, premium = 0.03, 0.08, 2.0, 0.4, 10.0
    escrowed = 100.0 - dividend * math.exp(-rate * payment)
    delta = norm.cdf((math.log(escrowed / 100.0) + rate + 0.2 ** 2 / 2) / 0.2)
    initial_cash = premium - delta * 100.0
    year_growth = math.exp(financing) if compounding == "continuous" else 1.0 + financing
    dividend_growth = math.exp(financing * 0.6) if compounding == "continuous" else 1.0 + financing * 0.6
    by_hand = initial_cash * year_growth + delta * dividend * dividend_growth + delta * 110.0 - 10.0
    found = qf.delta_hedge_pnl([[100.0, 110.0]], [0.0, 1.0], "call", 100.0, rate, financing, 0.0,
                               [(payment, dividend)], 0.2, premium, "short", 1, 0.0, "proportional", compounding,
                               "cash")
    assert found["pnl"][0] == pytest.approx(by_hand, rel=1e-12)
    assert found["present_value"][0] == pytest.approx(by_hand / year_growth, rel=1e-12)
    assert found["dividends"][0] == pytest.approx(delta * dividend, rel=1e-12)


def test_adding_a_dividend_date_without_a_trade_preserves_the_continuously_financed_hedge():
    """A self-financing portfolio's value depends on cash-flow dates and trade dates; adding an observation
    without trading must leave it alone. Compare a coarse grid with the payment explicitly observed."""
    args = dict(option="call", strike=100.0, rate=0.05, financing_rate=0.05, dividend_yield=0.0,
                cash_dividends=[(0.5, 2.0)], hedge_vol=0.2, premium=9.0, position="short", rebalance_every=3,
                cost=0.0, cost_rule="proportional", compounding="continuous", settlement="cash")
    coarse = qf.delta_hedge_pnl([[100.0, 98.0]], [0.0, 1.0], **args)
    observed = qf.delta_hedge_pnl([[100.0, 98.0, 98.0]], [0.0, 0.5, 1.0], **args)
    assert coarse["pnl"][0] == pytest.approx(observed["pnl"][0], rel=1e-12)
    assert coarse["rebalances"] == observed["rebalances"] == 0


@pytest.mark.parametrize("every", [1, 4])
@pytest.mark.parametrize("option", ["call", "put"])
def test_a_self_financing_hedge_averages_premium_less_black_scholes_on_risk_neutral_paths(option, every):
    """Discounted gains of a self-financing hedge are a martingale, so E[PV] = premium - BS price exactly, for any
    rebalancing dates and any hedge vol (the dividends reinvested at the yield, interest continuous)."""
    paths, times = risk_neutral(20, 20_000, 9)
    found = hedge(paths, times, option, premium=qf.bs_price(option, S0, K, T, R, Q, VOL) + 0.5, every=every, vol=0.3)
    assert abs(found["mean"] - 0.5) < 4 * found["std"] / math.sqrt(paths.shape[0])
    assert found["rebalances"] == (19 if every == 1 else 4)


def test_short_call_and_short_put_hedges_have_the_same_pnl_on_every_path():
    """Put-call parity: the call's hedge less the put's is e^{-q tau} shares against a short forward, self-financing
    with the yield reinvested, so the two P&Ls agree path by path when each is sold at its Black-Scholes price."""
    paths, times = risk_neutral(10, 200, 4)
    call, put = hedge(paths, times, "call"), hedge(paths, times, "put")
    assert np.allclose(call["pnl"], put["pnl"], atol=1e-9)


def test_long_is_minus_short_and_cash_settlement_is_physical_without_costs():
    paths, times = risk_neutral(10, 300, 5)
    short, long_ = hedge(paths, times), hedge(paths, times, position="long")
    assert np.allclose(long_["pnl"], -short["pnl"], atol=1e-12)
    assert np.allclose(hedge(paths, times, settlement="physical")["pnl"], short["pnl"], atol=1e-9)


def test_costs_come_off_the_pnl_exactly_and_the_side_sets_the_rate():
    """At a zero rate a cost paid is a cost lost: pnl without costs less pnl with them is the costs. Buying pays the
    first rate of the pair and selling the second; 'per_share' charges the rate a share."""
    paths, times = risk_neutral(10, 300, 6, q=0.0)
    free = hedge(paths, times, rate=0.0, q=0.0, premium=2.0)
    for cost, rule in ((0.002, "proportional"), ((0.001, 0.003), "proportional"), (0.01, "per_share")):
        paid = hedge(paths, times, rate=0.0, q=0.0, premium=2.0, cost=cost, rule=rule)
        assert np.allclose(free["pnl"] - paid["pnl"], paid["costs"], atol=1e-12)
    one = qf.delta_hedge_pnl([[100.0, 120.0]], [0.0, 1.0], "call", 100.0, 0.0, 0.0, 0.0, None, 0.2, 8.0, "short", 1,
                             (0.001, 0.003), "proportional", "continuous", "cash")
    delta = norm.cdf(0.1)
    assert one["costs"][0] == pytest.approx(0.001 * delta * 100.0 + 0.003 * delta * 120.0, rel=1e-12)
    physical = qf.delta_hedge_pnl([[100.0, 120.0]], [0.0, 1.0], "call", 100.0, 0.0, 0.0, 0.0, None, 0.2, 8.0, "short", 1,
                                  0.001, "proportional", "continuous", "physical")
    assert physical["costs"][0] == pytest.approx(0.001 * delta * 100.0 + 0.001 * (1 - delta) * 120.0, rel=1e-12)


def test_rebalancing_every_kth_date_is_the_hedge_on_every_kth_date_without_a_yield():
    paths, times = risk_neutral(40, 100, 7, q=0.0)
    every = hedge(paths, times, every=4, q=0.0)
    coarse = hedge(paths[:, ::4], times[::4], q=0.0)
    assert np.allclose(every["pnl"], coarse["pnl"], atol=1e-10) and every["rebalances"] == coarse["rebalances"] == 9


def test_simple_compounding_grows_the_cash_by_one_plus_r_h():
    found = qf.delta_hedge_pnl([[100.0, 100.0]], [0.0, 1.0], "call", 100.0, 0.05, 0.05, 0.0, None, 0.2, 10.0,
                               "short", 1, 0.0,
                               "proportional", "simple", "cash")
    delta = norm.cdf(0.35)
    assert found["pnl"][0] == pytest.approx((10.0 - delta * 100.0) * 1.05 + delta * 100.0, rel=1e-12)
    assert found["present_value"][0] == pytest.approx(found["pnl"][0] / 1.05, rel=1e-12)


def test_cash_dividends_hedged_by_the_escrowed_delta_average_premium_less_the_escrowed_price():
    """A dividend D at t_d: the price is X + PV(D) before t_d and X after, X a GBM of drift r (the escrowed model);
    the call is worth Black-Scholes on X(0), and the escrowed hedge, dividends credited on the shares held, is
    self-financing: E[PV] = premium - that price."""
    dividend, when, steps, n = 1.0, 0.2, 40, 20_000
    times = np.linspace(0.0, T, steps + 1)
    x0 = S0 - dividend * math.exp(-R * when)
    x = qf.gbm_paths(x0, R, VOL, times, n, 13, "arithmetic", False)
    paths = x + np.where(times < when, dividend * np.exp(-R * (when - times)), 0.0)
    price = qf.bs_price("call", x0, K, T, R, 0.0, VOL)
    found = qf.delta_hedge_pnl(paths, times, "call", K, R, R, 0.0, [(when, dividend)], VOL, price + 0.25, "short", 1, 0.0,
                               "proportional", "continuous", "cash")
    assert abs(found["mean"] - 0.25) < 4 * found["std"] / math.sqrt(n)
    held = norm.cdf((math.log(x0 / K) + (R + VOL ** 2 / 2) * T) / (VOL * math.sqrt(T)))
    assert found["dividends"].mean() == pytest.approx(held * dividend, rel=0.15)


def test_the_conventions_refuse_what_they_do_not_name():
    paths, times = risk_neutral(4, 4, 1)
    with pytest.raises(ValueError, match="hedge_vol"):
        hedge(paths, times, vol=[0.2, 0.2])
    with pytest.raises(ValueError, match="cost"):
        hedge(paths, times, cost=-0.001)
    with pytest.raises(ValueError, match="rebalance_every"):
        hedge(paths, times, every=0)
    with pytest.raises(ValueError, match="'cash', 'physical'"):
        hedge(paths, times, settlement="net")


@pytest.mark.slow
def test_the_hedges_spread_is_derman_and_kamals_and_halves_as_rebalancing_quadruples():
    """Hull's example option hedged on real-world paths (drift 13%): the present value's standard deviation is about
    sqrt(pi/4) vol vega / sqrt(N) (Derman and Kamal 1999), and N four times as many rebalancings halve it."""
    vega = qf.bs_greeks("call", S0, K, T, R, 0.0, VOL, 1.0, "year", 1.0)["vega"]
    spread = {}
    for steps in (20, 80):
        times = np.linspace(0.0, T, steps + 1)
        paths = qf.gbm_paths(S0, 0.13, VOL, times, 40_000, 2, "arithmetic", False)
        spread[steps] = hedge(paths, times, q=0.0)["std"]
        assert spread[steps] == pytest.approx(math.sqrt(math.pi / 4) * VOL * vega / math.sqrt(steps), rel=0.06)
    assert spread[20] / spread[80] == pytest.approx(2.0, rel=0.05)
