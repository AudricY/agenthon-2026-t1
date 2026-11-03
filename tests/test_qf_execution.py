"""agent/qf/execution.py (runs/lanes/lib6-prompt.md, rank 4 of docs/research/library-syllabus-1011.md): TWAP and VWAP
schedules by the sum identity and Hamilton's largest-remainder rule (Balinski and Young) on small cases worked by
hand. The orders and profiles are this file's own; no unit's values appear (rule 8)."""
import itertools

import numpy as np
import pytest

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf.execution as ex

pytestmark = pytest.mark.fast


def test_the_execution_card_keeps_the_format_and_its_example_runs():
    check_cards(ex)


def test_hamiltons_method_by_hand():
    """10,000 shares in three equal slices of round lots: quotas of 33 1/3 lots, one lot left over, to the earliest or
    the latest; 1,000 shares on a 45/45/10 profile: quotas 4.5, 4.5 and 1 lots, the leftover lot to the first of the
    two halves, while flooring with the rest to the last gives 400, 400, 200."""
    twap = ex.schedule("twap", 10000, 3, None, 100, "largest_remainder", "earliest")
    assert twap["slices"].tolist() == [3400.0, 3300.0, 3300.0]
    assert twap["quotas"] == pytest.approx([10000 / 3] * 3) and twap["cumulative"].tolist() == [3400, 6700, 10000]
    assert ex.schedule("twap", 10000, 3, None, 100, "largest_remainder", "latest")["slices"].tolist() == [3300, 3300,
                                                                                                       3400]
    profile = [45.0, 45.0, 10.0]
    assert ex.schedule("vwap", 1000, 3, profile, 100, "largest_remainder", "earliest")["slices"].tolist() == [500, 400,
                                                                                                           100]
    assert ex.schedule("vwap", 1000, 3, profile, 100, "largest_remainder", "latest")["slices"].tolist() == [400, 500,
                                                                                                         100]
    assert ex.schedule("vwap", 1000, 3, profile, 100, "floor_last", "earliest")["slices"].tolist() == [400, 400, 200]
    found = ex.schedule("vwap", 1000, 3, profile, 100, "none", "earliest")
    assert found["slices"] == pytest.approx([450.0, 450.0, 100.0]) and found["weights"] == pytest.approx([.45, .45, .1])


@pytest.mark.parametrize("seed", range(6))
def test_the_slices_sum_to_the_order_and_each_lies_within_a_lot_of_its_quota(seed):
    """Hamilton's method meets the quota: every slice is its quota's floor or ceiling in lots, and the slices sum to
    the order; the remainders handed a lot are never smaller than one passed over."""
    rng = np.random.default_rng(seed)
    n, lot = int(rng.integers(2, 30)), int(rng.choice([1, 10, 100]))
    order = lot * int(rng.integers(n, 400))
    profile = rng.gamma(2.0, 1.0, n)
    found = ex.schedule("vwap", order, n, profile, lot, "largest_remainder", "earliest")
    lots, quotas = found["slices"] / lot, found["quotas"] / lot
    assert found["slices"].sum() == order and np.all(np.abs(lots - quotas) < 1.0)
    assert np.all((lots == np.floor(quotas)) | (lots == np.ceil(quotas)))
    raised = lots > np.floor(quotas + 1e-9)
    if raised.any() and (~raised).any():
        assert (quotas - np.floor(quotas))[raised].min() >= (quotas - np.floor(quotas))[~raised].max() - 1e-9


def test_a_quota_a_rounding_error_below_a_whole_lot_is_that_lot():
    """A profile of 2.8, 0.1 and 0.1 over 30 lots has quotas of 28, 1 and 1 lots, the first a hair under 28 in floating
    point: flooring it as it stands would move a lot to the last interval."""
    found = ex.schedule("vwap", 3000, 3, [2.8, 0.1, 0.1], 100, "floor_last", "earliest")
    assert found["slices"].tolist() == [2800.0, 100.0, 100.0]


def test_a_flat_profile_is_twap():
    for rounding, tie in itertools.product(("largest_remainder", "floor_last", "none"), ("earliest", "latest")):
        flat = ex.schedule("vwap", 2500, 7, [5.0] * 7, 100, rounding, tie)
        assert flat["slices"].tolist() == ex.schedule("twap", 2500, 7, None, 100, rounding, tie)["slices"].tolist()


@pytest.mark.parametrize("bad, match", [
    (dict(kind="pov"), "kind"), (dict(rounding="nearest"), "rounding"), (dict(tie="random"), "tie"),
    (dict(quantity=1050), "whole number of lots"), (dict(profile=[1.0, 2.0]), "profile"),
    (dict(profile=[1.0, -1.0, 2.0]), "profile"), (dict(kind="twap"), "profile"), (dict(intervals=0), "intervals")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(kind="vwap", quantity=1000, intervals=3, profile=[1.0, 2.0, 3.0], lot=100, rounding="largest_remainder",
                tie="earliest")
    with pytest.raises(ValueError, match=match):
        ex.schedule(**{**args, **bad})
