"""agent/qf/metrics.py's rebalance_path (runs/lanes/lib6-prompt.md, rank 5 of docs/research/library-syllabus-1011.md):
the self-financing identity on toys with no costs and with costs, buy and hold, the drift and turnover worked by hand
(Bacon 2008; Grinold and Kahn ch. 16). The returns and weights are this file's own; no unit's values appear (rule 8)."""
import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf.metrics as m

pytestmark = pytest.mark.fast

R = np.array([[0.10, -0.10], [0.05, 0.02], [-0.03, 0.04], [0.02, 0.01]])


def path(**changes):
    args = dict(returns=R, weights=[0.5, 0.5], rebalance=[True, True, True, True], initial_value=1000.0,
                cash_return=0.0, timing="start", fee_order="before", turnover="two_way", turnover_base="drifted",
                cost_rate=0.0, start="invested", between_rebalances="drift")
    return m.rebalance_path(**{**args, **changes})


def test_with_no_cost_the_value_is_the_product_of_the_gross_returns_and_a_constant_mix_earns_w_r_each_bar():
    found = path()
    assert found["gross_returns"] == pytest.approx(R @ [0.5, 0.5], abs=1e-15)
    assert found["values"][-1] == pytest.approx(1000.0 * np.prod(1.0 + R @ [0.5, 0.5]), abs=1e-10)
    assert found["returns"] == pytest.approx(found["gross_returns"], abs=1e-15)


def test_one_trade_then_holding_is_buy_and_hold():
    """V_T = V_0 sum_i w_i prod_t (1 + r_ti): the weights drift and no trade follows."""
    found = path(weights=[0.3, 0.7], rebalance=[True, False, False, False], start="cash")
    assert found["values"][-1] == pytest.approx(1000.0 * (np.array([0.3, 0.7]) @ np.prod(1.0 + R, axis=0)), abs=1e-10)
    assert found["turnover"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_the_drift_and_the_turnover_by_hand():
    """50/50 earning +10% and -10%: the book is worth 1,000 still, the weights 55/45; trading back to 50/50 is a
    turnover of 0.10 two-way, 0.05 one-way, against the drifted weights; against the last target, none."""
    found = path()
    assert found["drifted"][0] == pytest.approx([0.55, 0.45], abs=1e-15)
    assert found["turnover"][1] == pytest.approx(0.10, abs=1e-15)
    assert path(turnover="one_way")["turnover"][1] == pytest.approx(0.05, abs=1e-15)
    assert path(turnover_base="target")["turnover"][1] == 0.0
    moving = path(weights=[[0.5, 0.5], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]], turnover_base="target")
    assert moving["turnover"] == pytest.approx([0.0, 0.6, 1.2, 0.0], abs=1e-15)   # each against the last target
    held = path(rebalance=[True, False, True, False])
    assert held["weights"][1] == pytest.approx([0.55, 0.45], abs=1e-15)
    expected = held["drifted"][1]
    assert held["turnover"][2] == pytest.approx(np.abs(np.array([0.5, 0.5]) - expected).sum(), abs=1e-15)


def test_costs_by_hand_before_and_after_the_return_and_the_self_financing_identity():
    """10 bps a unit of two-way turnover on the trade back to 50/50 (0.10): 0.10 of value. Before the bar's return
    the book loses it and then earns 3.5%; after, it earns 3.5% on the whole and pays it from the result."""
    before, after = path(cost_rate=0.001), path(cost_rate=0.001, fee_order="after")
    v0 = 1000.0
    assert before["costs"][1] == pytest.approx(0.1, abs=1e-12)
    assert before["values"][1] == pytest.approx(v0 * (1 - 0.0001) * 1.035, abs=1e-10)
    assert after["values"][1] == pytest.approx(v0 * 1.035 - 0.1, abs=1e-10)
    for found in (before, after):         # V_t = V_{t-1} (1 + r_p) - cost_t (1 + r_p if before)
        previous = np.r_[1000.0, found["values"][:-1]]
        grown = 1.0 + found["gross_returns"]
        paid = found["costs"] * (grown if found is before else 1.0)
        assert found["values"] == pytest.approx(previous * grown - paid, abs=1e-10)


def test_a_cost_after_the_return_comes_out_of_every_holding_pro_rata():
    """The lane's review: a fee paid after the bar's return leaves the drifted weights as they were, so the next bar
    earns them, as share-count accounting with the fee taken from each holding in proportion has it."""
    found = path(weights=[0.6, 0.3], rebalance=[True, False, False, False], start="cash", cost_rate=0.004,
                 fee_order="after", cash_return=0.0005)
    held, cash = np.array([600.0, 300.0]), 100.0
    for t, row in enumerate(R):
        held, cash = held * (1.0 + row), cash * 1.0005
        total = held.sum() + cash
        scale = (total - (0.004 * 0.9 * 1000.0 if t == 0 else 0.0)) / total
        held, cash = held * scale, cash * scale
        assert found["values"][t] == pytest.approx(held.sum() + cash, abs=1e-10)


def test_the_first_trade_from_cash_is_charged_and_cash_earns_its_rate():
    found = path(start="cash", cost_rate=0.002, weights=[0.4, 0.4], cash_return=0.001)
    assert found["turnover"][0] == pytest.approx(0.8, abs=1e-15) and found["costs"][0] == pytest.approx(1.6, abs=1e-12)
    assert found["gross_returns"][0] == pytest.approx(0.4 * 0.10 - 0.4 * 0.10 + 0.2 * 0.001, abs=1e-15)


def test_trading_at_the_bars_end_is_trading_at_the_next_bars_start():
    targets = np.array([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
    late = path(weights=targets, timing="end", start="cash", cost_rate=0.001)
    early = path(weights=np.vstack([[0.0, 0.0], targets[:-1]]), rebalance=[False, True, True, True], start="cash",
                 cost_rate=0.001)
    assert late["values"] == pytest.approx(early["values"], abs=1e-12)
    assert late["gross_returns"][0] == 0.0 and late["weights"][1] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("bad, match", [
    (dict(timing="close"), "timing"), (dict(fee_order="during"), "fee_order"), (dict(turnover="half"), "turnover"),
    (dict(turnover_base="prior"), "turnover_base"), (dict(start="empty"), "start"),
    (dict(rebalance=[1, 1, 1, 1]), "booleans"), (dict(weights=[0.5, 0.3, 0.2]), "weights"),
    (dict(cost_rate=-0.001), "cost_rate"), (dict(returns=R[:, :1] - 2.0), "returns")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        path(**bad)
