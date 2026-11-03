"""qf.credit.cva (runs/lanes/lib7-prompt.md, rank 18 of docs/research/library-syllabus-1011.md): the charge on a
one-period case by hand, a constant exposure under a flat hazard in closed form, the zero and linear cases, netting
before the positive part, the bilateral and first-to-default weights and the CSA's collateral, each written out
here (Gregory 2015, ch. 14). Every input is this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf.credit as cr

pytestmark = pytest.mark.fast

TIMES = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
D = np.exp(-0.03 * TIMES)
SC, SO = np.exp(-0.02 * TIMES), np.exp(-0.01 * TIMES)
PATHS = np.random.default_rng(9).normal(0.0, 1.0, (500, 5)).cumsum(axis=1) * np.sqrt(0.5)
PATHS[:, 0] = 0.0


def charge(**changes):
    args = dict(values=PATHS, times=TIMES, discount_factors=D, counterparty_survival=SC, counterparty_recovery=0.4,
                own_survival=None, own_recovery=None, timing="end", kind="unilateral", collateral=None)
    return cr.cva(**{**args, **changes})


def test_one_period_by_hand():
    """Two paths ending at 3 and -1: EE = 1.5, NEE = 0.5; CVA = (1 - R_c) D EE (1 - S_c), DVA alike."""
    found = cr.cva([[0.0, 3.0], [0.0, -1.0]], [0.0, 1.0], [1.0, 0.95], [1.0, 0.97], 0.4, [1.0, 0.99], 0.25, "end",
                   "bilateral", None)
    assert found["ee"].tolist() == [0.0, 1.5] and found["nee"].tolist() == [0.0, 0.5]
    assert found["cva"] == pytest.approx(0.6 * 0.95 * 1.5 * 0.03, rel=1e-14)
    assert found["dva"] == pytest.approx(0.75 * 0.95 * 0.5 * 0.01, rel=1e-14)
    assert found["bcva"] == pytest.approx(found["cva"] - found["dva"], rel=1e-14)


def test_a_constant_exposure_under_a_flat_hazard_is_the_closed_sum():
    values = np.full((3, 5), 2.5)
    found = charge(values=values)
    assert found["cva"] == pytest.approx(0.6 * 2.5 * np.sum(D[1:] * (SC[:-1] - SC[1:])), rel=1e-14)
    mid = charge(values=values, timing="midpoint")
    assert mid["cva"] == pytest.approx(0.6 * 2.5 * np.sum(0.5 * (D[:-1] + D[1:]) * (SC[:-1] - SC[1:])), rel=1e-14)


def test_the_charge_vanishes_without_loss_or_default_and_is_linear_in_the_loss():
    assert charge(counterparty_recovery=1.0)["cva"] == 0.0
    assert charge(kind="bilateral", own_survival=SO, own_recovery=1.0)["dva"] == 0.0
    assert charge(counterparty_survival=np.ones(5))["cva"] == 0.0
    assert charge(counterparty_recovery=0.0)["cva"] == pytest.approx(charge()["cva"] / 0.6, rel=1e-14)
    held = D * charge()["ee"]
    assert charge(timing="midpoint")["cva"] == pytest.approx(
        0.6 * np.sum(0.5 * (held[:-1] + held[1:]) * (SC[:-1] - SC[1:])), rel=1e-14)


def test_the_trades_are_netted_before_the_positive_part():
    rng = np.random.default_rng(3)
    trades = rng.normal(0.0, 1.0, (400, 5, 3))
    trades[:, 0, :] = 0.0
    netted = charge(values=trades)
    assert netted["ee"] == pytest.approx(np.maximum(trades.sum(axis=2), 0.0).mean(axis=0), rel=1e-14)
    alone = sum(charge(values=trades[:, :, j])["cva"] for j in range(3))
    assert netted["cva"] < alone


def test_first_to_default_weights_each_leg_by_the_others_survival():
    both = charge(kind="bilateral", own_survival=SO, own_recovery=0.4)
    first = charge(kind="first_to_default", own_survival=SO, own_recovery=0.4)
    ee, nee = both["ee"], both["nee"]
    assert both["cva"] == pytest.approx(charge()["cva"], rel=1e-15)
    assert first["cva"] == pytest.approx(0.6 * np.sum(D[1:] * ee[1:] * (SC[:-1] - SC[1:]) * SO[:-1]), rel=1e-14)
    assert first["dva"] == pytest.approx(0.6 * np.sum(D[1:] * nee[1:] * (SO[:-1] - SO[1:]) * SC[:-1]), rel=1e-14)
    assert first["cva"] < both["cva"] and first["dva"] < both["dva"]


def test_the_csa_holds_the_value_past_the_threshold_as_of_the_lag():
    """A path 0, 5, -4, 2, -6 with H = 1, M = 0.5, a lag of one step: collateral 0, 0, 4, -3, 1 (sign(V) (|V| - H)^+
    a step before, posted both ways), so V - C = 0, 5, -8, 5, -7; a lag of none and H = 0 leaves nothing exposed."""
    path = np.array([[0.0, 5.0, -4.0, 2.0, -6.0]])
    found = charge(values=path, collateral={"threshold": 1.0, "minimum_transfer": 0.5, "lag": 1})
    assert found["exposure"].tolist() == [[0.0, 5.0, 0.0, 5.0, 0.0]]
    assert found["nee"].tolist() == [0.0, 0.0, 8.0, 0.0, 7.0]
    small = charge(values=np.array([[0.0, 1.3, 1.3, 1.3, 1.3]]),
                   collateral={"threshold": 1.0, "minimum_transfer": 0.5, "lag": 0})
    assert small["exposure"].tolist() == [[0.0, 1.3, 1.3, 1.3, 1.3]]
    none = charge(collateral={"threshold": 0.0, "minimum_transfer": 0.0, "lag": 0})
    assert none["cva"] == 0.0 and np.all(none["exposure"] == 0.0)
    pfe = np.quantile(charge()["exposure"], 0.95, axis=0)
    assert np.all(pfe >= charge()["ee"])


def test_negative_rates_discount_above_one_and_a_survival_above_one_is_refused():
    """The review of lane lib7: under a negative rate D(t) > 1 is a discount factor like any other; a survival is not."""
    above = np.exp(0.005 * TIMES)
    held = above * charge(discount_factors=above)["ee"]
    assert charge(discount_factors=above)["cva"] == pytest.approx(0.6 * np.sum(held[1:] * (SC[:-1] - SC[1:])),
                                                                  rel=1e-14)
    with pytest.raises(ValueError, match=r"counterparty_survival must hold one value in \(0, 1\]"):
        charge(counterparty_survival=np.exp(0.01 * TIMES))
    with pytest.raises(ValueError, match=r"recovery must lie in \[0, 1\]"):
        charge(counterparty_recovery=1.2)


@pytest.mark.parametrize("bad, match", [
    (dict(kind="one_way"), "kind"), (dict(timing="start"), "timing"), (dict(own_recovery=0.4), "None"),
    (dict(kind="bilateral"), "own_survival"), (dict(counterparty_survival=SC[1:]), "counterparty_survival"),
    (dict(times=TIMES + 0.1), "times"), (dict(collateral={"threshold": 1.0}), "collateral"),
    (dict(discount_factors=D * 1.1), "discount_factors")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        charge(**bad)
