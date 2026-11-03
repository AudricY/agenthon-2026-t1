"""Lane lib-econometrics1 (runs/lanes/lib-econometrics1-prompt.md): qf.implementation_shortfall and qf.impact_cost.
Perold (1988): the shortfall is the paper portfolio's return less the real one's, its parts (delay, execution,
opportunity, fees) summing to it under both of the field's splits (Kissell 2013, ch. 3; Wagner and Edwards 1993),
worked by hand on this file's own order; the square-root law (Toth et al. 2011; Almgren et al. 2005): the impact at
one day's volume is Y sigma and doubling the order multiplies it by sqrt 2. No unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401  (first: it puts agent/ on the path outside pytest too)
import qf

pytestmark = pytest.mark.fast

ORDER = dict(order_quantity=2000, decision_price=25.00, arrival_price=25.05, fill_prices=[25.10, 25.12, 25.20],
             fill_quantities=[600, 500, 400], close_price=25.30, fees=12.0)


def test_a_buy_worked_by_hand():
    """Q = 2,000 decided at 25.00, arriving at 25.05; 1,500 filled (600 at 25.10, 500 at 25.12, 400 at 25.20, an
    average of 37,700 / 1,500 = 25.1333); 500 unfilled at the 25.30 close; 12 of fees. Paper: 2,000 x 25.00 =
    50,000. execution = 600 x 0.05 + 500 x 0.07 + 400 x 0.15 = 125; on the whole order (Kissell): delay = 2,000 x
    0.05 = 100, opportunity = 500 x 0.25 = 125; on the filled part (Perold): delay = 1,500 x 0.05 = 75, opportunity =
    500 x 0.30 = 150; total = 200 + 150 + 12 = 362 either way, 72.4 bps. Slippage to the arrival: (25.1333 - 25.05) /
    25.05 = 33.3 bps."""
    order = qf.implementation_shortfall(side="buy", **ORDER, delay_on="order", benchmarks={"arrival": 25.05})
    filled = qf.implementation_shortfall(side="buy", **ORDER, delay_on="filled", benchmarks={"arrival": 25.05})
    assert (order["execution"], order["delay"], order["opportunity"]) == pytest.approx((125.0, 100.0, 125.0))
    assert (filled["execution"], filled["delay"], filled["opportunity"]) == pytest.approx((125.0, 75.0, 150.0))
    for found in (order, filled):
        assert found["total"] == pytest.approx(362.0) and found["total_bps"] == pytest.approx(72.4)
        assert found["paper_value"] == 50_000.0 and found["filled"] == 1500.0 and found["fill_rate"] == 0.75
        assert found["average_price"] == pytest.approx(37_700 / 1_500)
        assert found["slippage_bps"]["arrival"] == pytest.approx((37_700 / 1_500 - 25.05) / 25.05 * 1e4)
        assert found["fees_bps"] == pytest.approx(12.0 / 50_000 * 1e4)


@pytest.mark.parametrize("delay_on", ["order", "filled"])
def test_the_parts_sum_to_the_paper_less_the_real_return_on_either_side(delay_on):
    rng = np.random.RandomState(81)
    for _ in range(25):
        side = rng.choice(["buy", "sell"])
        q = float(rng.randint(100, 5000))
        sizes = rng.uniform(0, 1, size=rng.randint(0, 6))
        sizes = np.floor(sizes / max(sizes.sum(), 1) * q * rng.uniform(0.2, 1.0))
        prices = 40 + rng.normal(size=sizes.size)
        decision, arrival, close = 40 + rng.normal(size=3)
        fees = float(rng.uniform(0, 20))
        found = qf.implementation_shortfall(side=side, order_quantity=q, decision_price=decision,
                                            arrival_price=arrival, fill_prices=prices, fill_quantities=sizes,
                                            close_price=close, fees=fees, delay_on=delay_on, benchmarks={})
        sign = 1 if side == "buy" else -1
        paper = sign * (sizes @ (prices - decision) + (q - sizes.sum()) * (close - decision)) + fees
        assert found["total"] == pytest.approx(paper, rel=1e-12, abs=1e-9)
        parts = found["delay"] + found["execution"] + found["opportunity"] + found["fees"]
        assert parts == pytest.approx(found["total"], rel=1e-12, abs=1e-9)
        assert found["total_bps"] == pytest.approx(found["total"] / (q * decision) * 1e4)


def test_a_sell_mirrors_a_buy_and_the_edges_of_filling():
    sell = qf.implementation_shortfall(side="sell", order_quantity=1000, decision_price=50.0, arrival_price=49.9,
                                       fill_prices=[49.8], fill_quantities=[1000], close_price=49.0, fees=0.0,
                                       delay_on="order", benchmarks={"vwap": 49.85})
    assert sell["delay"] == pytest.approx(100.0) and sell["execution"] == pytest.approx(100.0)
    assert sell["opportunity"] == 0.0 and sell["total"] == pytest.approx(200.0)
    assert sell["slippage_bps"]["vwap"] == pytest.approx((49.85 - 49.8) / 49.85 * 1e4)
    none = qf.implementation_shortfall(side="buy", order_quantity=500, decision_price=10.0, arrival_price=10.0,
                                       fill_prices=[], fill_quantities=[], close_price=10.4, fees=0.0,
                                       delay_on="filled", benchmarks={})
    assert none["filled"] == 0.0 and math.isnan(none["average_price"]) and none["execution"] == 0.0
    assert none["total"] == pytest.approx(500 * 0.4) and none["slippage_bps"] == {}
    with pytest.raises(ValueError, match="no more than order_quantity"):
        qf.implementation_shortfall(side="buy", order_quantity=100, decision_price=10.0, arrival_price=10.0,
                                    fill_prices=[10.0], fill_quantities=[150], close_price=10.0, fees=0.0,
                                    delay_on="order", benchmarks={})
    with pytest.raises(ValueError, match="side"):
        qf.implementation_shortfall(side="long", **ORDER, delay_on="order", benchmarks={})


def test_the_square_root_law_at_one_days_volume_and_at_twice_the_order():
    one_day = qf.impact_cost(quantity=1e6, adv=1e6, daily_vol=0.02, spread=0.0, coefficient=0.8, exponent=0.5,
                             price=10.0, participation=None)
    assert one_day["impact"] == pytest.approx(0.8 * 0.02) and math.isnan(one_day["liquidation_days"])
    base = qf.impact_cost(quantity=20_000, adv=1e6, daily_vol=0.02, spread=0.0008, coefficient=1.0, exponent=0.5,
                          price=40.0, participation=0.1)
    double = qf.impact_cost(quantity=40_000, adv=1e6, daily_vol=0.02, spread=0.0008, coefficient=1.0, exponent=0.5,
                            price=40.0, participation=0.1)
    assert double["impact"] / base["impact"] == pytest.approx(math.sqrt(2.0))
    assert base["spread_cost"] == pytest.approx(0.0004) and base["total"] == pytest.approx(0.0004 + 0.02 * math.sqrt(0.02))
    assert base["total_bps"] == pytest.approx(base["total"] * 1e4)
    assert base["cost"] == pytest.approx(base["total"] * 20_000 * 40.0) and base["liquidation_days"] == pytest.approx(0.2)
    three_fifths = qf.impact_cost(quantity=20_000, adv=1e6, daily_vol=0.02, spread=0.0, coefficient=1.0, exponent=0.6,
                                  price=40.0, participation=None)
    assert three_fifths["impact"] == pytest.approx(0.02 * 0.02 ** 0.6) and three_fifths["impact"] < base["impact"]
    with pytest.raises(ValueError, match="exponent"):
        qf.impact_cost(quantity=1, adv=1, daily_vol=0.02, spread=0.0, coefficient=1.0, exponent=0.0, price=1.0,
                       participation=None)
