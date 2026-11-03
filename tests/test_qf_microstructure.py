"""agent/qf/microstructure.py (runs/lanes/lib6-prompt.md, rank 3 of docs/research/library-syllabus-1011.md): trade
signing and the spread measures on a small quote-and-trade set written out here, each figure worked by hand from
Lee and Ready's (1991) rule and Huang and Stoll's (1996) definitions. The quotes and trades are this file's own; no
unit's values appear (rule 8)."""
import numpy as np
import pytest

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf.microstructure as ms

pytestmark = pytest.mark.fast

QUOTES = dict(quote_times=[0.0, 10.0, 20.0, 30.0], bids=[10.00, 10.02, 10.10, 10.00], asks=[10.10, 10.08, 10.20, 10.04])
#             mids 10.05, 10.05, 10.15, 10.02; spreads 0.10, 0.06, 0.10, 0.04
TRADES = dict(trade_times=[5.0, 12.0, 20.0, 25.0, 31.0], trade_prices=[10.10, 10.05, 10.10, 10.15, 10.04],
              trade_sizes=[100.0, 200.0, 300.0, 100.0, 300.0])


def measures(**changes):
    args = dict(**TRADES, **QUOTES, sign_rule="lee_ready", signs=None, quote_lag=0.0, quote_match="at_or_before",
                horizon=10.0, scale="price", weighting="equal")
    return ms.spread_measures(**{**args, **changes})


def test_the_microstructure_card_keeps_the_format_and_its_example_runs():
    check_cards(ms)


def test_lee_and_readys_rule_by_hand():
    """Trade 1 at the ask of the quote at 0 (mid 10.05): a buy. Trade 2 at the mid of the quote at 10: the tick test,
    10.10 to 10.05, a sell. Trade 3 at 20 meets the quote stamped at 20 (mid 10.15): below it, a sell; with quotes
    strictly before, the quote at 10 (mid 10.05): above it, a buy. Trade 4 at the mid of 10.15: an uptick, a buy.
    Trade 5 above the mid 10.02: a buy."""
    assert measures()["sign"].tolist() == [1.0, -1.0, -1.0, 1.0, 1.0]
    assert measures(quote_match="before")["sign"].tolist() == [1.0, -1.0, 1.0, 1.0, 1.0]
    assert measures()["midpoint"] == pytest.approx([10.05, 10.05, 10.15, 10.15, 10.02], abs=1e-12)
    assert measures(quote_lag=5.0)["midpoint"] == pytest.approx([10.05, 10.05, 10.05, 10.15, 10.15], abs=1e-12)
    assert measures(sign_rule="quote")["sign"].tolist() == [1.0, 0.0, -1.0, 0.0, 1.0]
    assert measures(sign_rule="tick")["sign"].tolist() == [0.0, -1.0, 1.0, 1.0, -1.0]


def test_the_tick_test_carries_a_zero_tick():
    """Lee and Ready: a zero tick takes the direction of the last price change; before any change, no sign."""
    prices = [10.0, 10.0, 10.1, 10.1, 10.05, 10.05, 10.05]
    found = ms.spread_measures(list(range(7)), prices, [1.0] * 7, [0.0], [9.0], [11.0], "tick", None, 0.0,
                               "at_or_before", 0.0, "price", "equal")
    assert found["sign"].tolist() == [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, -1.0]


def test_the_spreads_by_hand_and_their_identities():
    """Huang and Stoll: quoted = ask - bid; effective = 2 D (P - M); realized = 2 D (P - M at t + 10); impact their
    difference. A buy at the ask has its effective spread equal to the quoted."""
    found = measures()
    assert found["quoted"] == pytest.approx([0.10, 0.06, 0.10, 0.10, 0.04], abs=1e-12)
    assert found["effective"] == pytest.approx([0.10, 0.0, 0.10, 0.0, 0.04], abs=1e-12)
    #     the midpoints 10 later: 15 -> 10.05, 22 -> 10.15, 30 -> 10.02, 35 -> 10.02, 41 -> 10.02 (the last stands)
    assert found["realized"] == pytest.approx([0.10, 0.20, -0.16, 0.26, 0.04], abs=1e-12)
    assert found["impact"] == pytest.approx(found["effective"] - found["realized"], abs=1e-15)
    assert found["effective"][0] == pytest.approx(found["quoted"][0], abs=1e-12)
    assert found["quoted_mean"] == pytest.approx(0.08, abs=1e-12)
    assert found["effective_mean"] == pytest.approx(0.24 / 5, abs=1e-12)
    sized = measures(weighting="size")
    assert sized["effective_mean"] == pytest.approx((100 * 0.10 + 300 * 0.10 + 300 * 0.04) / 1000, abs=1e-12)
    dollar = measures(weighting="dollar")
    weights = np.array(TRADES["trade_sizes"]) * np.array(TRADES["trade_prices"])
    assert dollar["realized_mean"] == pytest.approx(weights @ found["realized"] / weights.sum(), abs=1e-12)
    assert found["quoted_time_weighted"] == pytest.approx((0.10 + 0.06 + 0.10) / 3, abs=1e-12)


def test_the_scales_divide_by_the_trades_midpoint():
    price, bps, relative = measures(), measures(scale="bps"), measures(scale="relative")
    mids = np.array([10.05, 10.05, 10.15, 10.15, 10.02])
    assert relative["effective"] == pytest.approx(price["effective"] / mids, abs=1e-15)
    assert bps["realized"] == pytest.approx(1e4 * price["realized"] / mids, abs=1e-10)
    assert bps["quoted_time_weighted"] == pytest.approx(1e4 * (0.10 / 10.05 + 0.06 / 10.05 + 0.10 / 10.15) / 3,
                                                        abs=1e-10)


def test_given_signs_are_used_and_unsigned_trades_leave_the_means():
    found = measures(sign_rule="given", signs=[1, 1, 0, -1, 1])
    assert found["effective"] == pytest.approx([0.10, 0.0, 0.0, 0.0, 0.04], abs=1e-12)
    assert found["effective_mean"] == pytest.approx(0.14 / 4, abs=1e-12)
    early = measures(trade_times=[-1.0, 12.0, 20.0, 25.0, 31.0])        # before any quote: no midpoint, no sign
    assert np.isnan(early["midpoint"][0]) and early["sign"][0] == 0 and np.isnan(early["effective"][0])


@pytest.mark.parametrize("bad, match", [
    (dict(sign_rule="bulk"), "sign_rule"), (dict(signs=[1, 1, 1, 1, 1]), "signs"), (dict(sign_rule="given"), "signs"),
    (dict(scale="pct"), "scale"), (dict(weighting="volume"), "weighting"), (dict(quote_match="after"), "quote_match"),
    (dict(trade_times=[5.0, 4.0, 20.0, 25.0, 31.0]), "order"), (dict(bids=[10.2, 10.02, 10.10, 10.00]), "bid"),
    (dict(horizon=-1.0), "horizon")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        measures(**bad)
