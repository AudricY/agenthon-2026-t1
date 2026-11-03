"""qf.microstructure.match_orders (runs/lanes/lib7-prompt.md, rank 26 of docs/research/library-syllabus-1011.md): a
small book written out event by event (price then time priority, sweeps, partial fills, cancels, a market order's
rest cancelled, the self-trade rules) and the book's invariants on a random stream (Gould et al. 2013). Every order
is this file's own; no unit's values appear (rule 8)."""
import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf.microstructure as ms

pytestmark = pytest.mark.fast


def limit(i, side, price, quantity, owner=None):
    event = {"type": "limit", "id": i, "side": side, "price": price, "quantity": quantity}
    return {**event, "owner": owner} if owner is not None else event


def market(i, side, quantity, owner=None):
    event = {"type": "market", "id": i, "side": side, "quantity": quantity}
    return {**event, "owner": owner} if owner is not None else event


def cancel(i):
    return {"type": "cancel", "id": i}


def test_a_small_book_written_out():
    """Asks 100.1 x 200 (a), 100.1 x 100 (b), 100.2 x 300 (c); bids 99.9 x 150 (d), 99.8 x 100 (e). A market buy of
    250 takes a whole and 50 of b (time priority at 100.1); a buy limit at 100.2 for 400 takes b's 50 and c's 300 and
    rests 50 at 100.2; a sell limit at 99.8 for 300 meets that 50, then d's 150, then e's 100; cancels of e (filled)
    and of an unknown id are rejected."""
    events = [limit("a", "sell", 100.1, 200), limit("b", "sell", 100.1, 100), limit("c", "sell", 100.2, 300),
              limit("d", "buy", 99.9, 150), limit("e", "buy", 99.8, 100), market("m", "buy", 250),
              limit("f", "buy", 100.2, 400), limit("g", "sell", 99.8, 300), cancel("e"), cancel("zz")]
    found = ms.match_orders(events, 0.1, "allow")
    expected = [("m", "a", 100.1, 200), ("m", "b", 100.1, 50), ("f", "b", 100.1, 50), ("f", "c", 100.2, 300),
                ("f", "g", 100.2, 50), ("d", "g", 99.9, 150), ("e", "g", 99.8, 100)]
    assert [(t["buy_id"], t["sell_id"], t["price"], t["quantity"]) for t in found["trades"]] == expected
    assert [t["aggressor"] for t in found["trades"]] == ["buy"] * 4 + ["sell"] * 3
    assert [t["event"] for t in found["trades"]] == [5, 5, 6, 6, 7, 7, 7]
    assert found["bids"] == [] and found["asks"] == [] and found["rejected"] == ["e", "zz"]
    assert found["best_ask"].tolist()[:6] == [100.1, 100.1, 100.1, 100.1, 100.1, 100.1]
    assert found["best_bid"].tolist()[3:7] == [99.9, 99.9, 99.9, 100.2]


def test_a_market_orders_rest_is_cancelled_and_a_limits_rest_queues_behind_its_level():
    events = [limit(1, "sell", 10.0, 5), market(2, "buy", 8), limit(3, "buy", 9.9, 4), limit(4, "buy", 9.9, 6),
              limit(5, "buy", 9.95, 1), cancel(3)]
    found = ms.match_orders(events, None, "allow")
    assert [(t["buy_id"], t["quantity"]) for t in found["trades"]] == [(2, 5.0)]
    assert found["bids"] == [(5, 9.95, 1.0), (4, 9.9, 6.0)] and found["asks"] == []
    assert np.isnan(found["best_ask"][-1]) and found["best_bid"][-1] == 9.95


@pytest.mark.parametrize("rule, trades, asks", [
    ("allow", [(3, 1, 5.0)], [(2, 10.1, 5.0)]),
    ("cancel_resting", [(3, 2, 5.0)], []),
    ("cancel_incoming", [], [(1, 10.0, 5.0), (2, 10.1, 5.0)])])
def test_the_self_trade_rules(rule, trades, asks):
    """Owner X rests 5 at 10.0 and owner Y 5 at 10.1; X buys 5 at 10.1."""
    events = [limit(1, "sell", 10.0, 5, "X"), limit(2, "sell", 10.1, 5, "Y"), limit(3, "buy", 10.1, 5, "X")]
    found = ms.match_orders(events, None, rule)
    assert [(t["buy_id"], t["sell_id"], t["quantity"]) for t in found["trades"]] == trades
    assert found["asks"] == asks and found["bids"] == []


def test_the_books_invariants_on_a_random_stream():
    """After every event the best bid is below the best ask; every quantity submitted is traded (on both sides),
    resting, cancelled while resting, or a market order's or a self-trade's rest."""
    rng = np.random.default_rng(8)
    events, submitted = [], 0.0
    for i in range(600):
        draw = rng.random()
        if draw < 0.1 and i:
            events.append(cancel(int(rng.integers(0, i))))
        else:
            side = "buy" if rng.random() < 0.5 else "sell"
            size = float(rng.integers(1, 20))
            submitted += size
            if draw < 0.25:
                events.append(market(i, side, size))
            else:
                events.append(limit(i, side, round(100 + 0.1 * rng.integers(-15, 16), 1), size))
    found = ms.match_orders(events, 0.1, "allow")
    both = ~(np.isnan(found["best_bid"]) | np.isnan(found["best_ask"]))
    assert np.all(found["best_bid"][both] < found["best_ask"][both])
    traded = sum(t["quantity"] for t in found["trades"])
    assert sum(q for _, _, q in found["bids"] + found["asks"]) <= submitted - 2 * traded + 1e-9
    prices = {(e["id"]): e.get("price") for e in events if e["type"] == "limit"}
    for t in found["trades"]:
        resting = t["sell_id"] if t["aggressor"] == "buy" else t["buy_id"]
        assert t["price"] == prices[resting]


@pytest.mark.parametrize("bad, match", [
    ([{"type": "modify", "id": 1}], "must be a dict"), ([limit(1, "buy", 100.05, 1)], "multiple of the tick"),
    ([limit(1, "hold", 100.0, 1)], "side"), ([market(1, "buy", 0)], "quantity")])
def test_a_bad_event_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        ms.match_orders(bad, 0.1, "allow")
    with pytest.raises(ValueError, match="self_trade"):
        ms.match_orders([], 0.1, "decrement")
