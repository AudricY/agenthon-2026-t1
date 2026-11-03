"""Lane lib-risk1 (runs/lanes/lib-risk1-prompt.md), the `backtests` and `capital` shelves: Acerbi and Szekely's Z1 and
Z2 by hand on a short history and zero in expectation under the model (Acerbi and Szekely 2014, "Backtesting expected
shortfall", Risk); the FRTB sensitivities-based charge by MAR21.4 and MAR21.6 written out by hand, one factor its
weighted sensitivity, the S_b alternative where the sum turns negative; the liquidity-horizon ES by MAR33.4 and the
reduced-set ratio floored at 1 (MAR33.6); the moved var_backtest and ima_capital still `qf.risk.`. Synthetic values
alone; no unit's."""
import math

import numpy as np
import pytest

import qf
import qf.risk

pytestmark = pytest.mark.fast

LOSS = "loss_positive"
HISTORY = np.array([-0.03, 0.01, -0.025, 0.005, -0.01, 0.002, 0.004, -0.015, 0.008, 0.0])


def test_the_z_statistics_by_hand_on_a_short_history():
    """VaR 0.02, ES 0.028 at 97.5%: the hits are -0.03 and -0.025 (strictly beyond -VaR). Z1 = mean(X/ES) + 1 over the
    hits; Z2 = sum X I / (T p ES) + 1."""
    z1 = qf.es_backtest(returns=HISTORY, var=0.02, es=0.028, alpha=0.975, statistic="z1", sign=LOSS, null_draws=None)
    z2 = qf.es_backtest(returns=HISTORY, var=0.02, es=0.028, alpha=0.975, statistic="z2", sign=LOSS, null_draws=None)
    assert z1["exceptions"] == z2["exceptions"] == 2 and z1["observations"] == 10
    assert z1["statistic"] == pytest.approx((-0.03 / 0.028 - 0.025 / 0.028) / 2 + 1, rel=1e-14)
    assert z2["statistic"] == pytest.approx(-0.055 / (10 * 0.025 * 0.028) + 1, rel=1e-14)
    assert math.isnan(z2["p_value"]) and math.isnan(z2["null_mean"])
    negative = qf.es_backtest(returns=HISTORY, var=-0.02, es=-0.028, alpha=0.975, statistic="z2",
                              sign="return_negative", null_draws=None)
    assert negative["statistic"] == z2["statistic"]
    per_day = qf.es_backtest(returns=HISTORY, var=np.full(10, 0.02), es=np.r_[0.028, 0.028, 0.035, [0.028] * 7],
                             alpha=0.975, statistic="z2", sign=LOSS, null_draws=None)
    assert per_day["statistic"] == pytest.approx((-0.03 / 0.028 - 0.025 / 0.035) / 0.25 + 1, rel=1e-14)


def test_under_the_model_both_statistics_are_zero_in_expectation_and_an_understated_risk_is_rejected():
    """Acerbi and Szekely (2014): E[Z2] = 0 and E[Z1 | N > 0] = 0 when the forecast law is the truth. Normal returns of
    std 0.01 forecast by their own normal VaR and ES; the same forecasts on returns 1.5 times as wide give negative
    statistics and small simulated p-values."""
    var, es = qf.normal_var_es(mu=0.0, sigma=0.01, alpha=0.975, sign=LOSS)
    null = np.random.default_rng(2).standard_normal((4000, 250)) * 0.01
    for statistic, bound in (("z2", 0.03), ("z1", 0.03)):
        values = [qf.es_backtest(returns=row, var=var, es=es, alpha=0.975, statistic=statistic, sign=LOSS,
                                 null_draws=None)["statistic"] for row in null[:600]]
        assert abs(np.nanmean(values)) < bound, statistic
        found = qf.es_backtest(returns=null[0], var=var, es=es, alpha=0.975, statistic=statistic, sign=LOSS,
                               null_draws=null)
        assert abs(found["null_mean"]) < bound
    wide = np.random.default_rng(5).standard_normal(250) * 0.015
    for statistic in ("z1", "z2"):
        found = qf.es_backtest(returns=wide, var=var, es=es, alpha=0.975, statistic=statistic, sign=LOSS,
                               null_draws=null)
        assert found["statistic"] < 0 and found["p_value"] < 0.05, statistic


@pytest.mark.parametrize("bad", [dict(es=0.01), dict(statistic="z3"), dict(null_draws=np.zeros((5, 9))),
                                 dict(es=-0.028)])
def test_the_es_backtest_refuses_an_es_below_its_var_and_a_null_of_another_length(bad):
    args = dict(returns=HISTORY, var=0.02, es=0.028, alpha=0.975, statistic="z2", sign=LOSS, null_draws=None) | bad
    with pytest.raises(ValueError):
        qf.es_backtest(**args)


def test_var_backtest_moved_to_backtests_and_still_resolves_on_risk():
    assert qf.risk.var_backtest is qf.var_backtest is qf.backtests.var_backtest
    assert qf.risk.ima_capital is qf.ima_capital is qf.capital.ima_capital


# ------------------------------------------------------------------------------------------------ FRTB


def by_hand(ws, groups, rho, gamma):
    """MAR21.4 written out for a scalar rho and gamma: K_b within each bucket, then across, the S_b alternative."""
    k, s = [], []
    for members in groups:
        w = [ws[i] for i in members]
        inner = sum(a * a for a in w) + sum(rho * a * b for i, a in enumerate(w) for j, b in enumerate(w) if i != j)
        k.append(math.sqrt(max(0.0, inner)))
        s.append(sum(w))
    total = sum(x * x for x in k) + sum(gamma * s[b] * s[c] for b in range(len(s)) for c in range(len(s)) if b != c)
    if total < 0:
        s = [max(min(x, kb), -kb) for x, kb in zip(s, k)]
        total = sum(x * x for x in k) + sum(gamma * s[b] * s[c] for b in range(len(s)) for c in range(len(s)) if b != c)
    return math.sqrt(max(total, 0.0))


def test_the_sensitivities_based_charge_by_hand_in_each_correlation_scenario():
    """MAR21.6: 'high' 1.25 rho capped at 1, 'low' max(2 rho - 1, 0.75 rho), each for rho and gamma; MAR21.7: the
    largest."""
    s, rw, buckets = [100.0, -40.0, 60.0], [0.3, 0.3, 0.5], [1, 1, 2]
    found = qf.frtb_sbm(sensitivities=s, risk_weights=rw, buckets=buckets, rho=0.2, gamma=0.15, scenario="max",
                        alternative="if_negative")
    ws = [30.0, -12.0, 30.0]
    expected = {"medium": by_hand(ws, [[0, 1], [2]], 0.2, 0.15),
                "high": by_hand(ws, [[0, 1], [2]], 0.25, 0.1875),
                "low": by_hand(ws, [[0, 1], [2]], 0.15, 0.1125)}
    assert found["scenarios"] == pytest.approx(expected, rel=1e-14)
    assert found["charge"] == pytest.approx(max(expected.values()), rel=1e-14)
    assert expected["medium"] == pytest.approx(math.sqrt(1962.0), rel=1e-14)
    medium = qf.frtb_sbm(sensitivities=s, risk_weights=rw, buckets=buckets, rho=0.2, gamma=0.15, scenario="medium",
                         alternative="if_negative")
    assert medium["charge"] == pytest.approx(math.sqrt(1962.0), rel=1e-14)
    assert medium["kb"] == pytest.approx([30.0, 30.0]) and medium["sb"] == pytest.approx([18.0, 30.0])
    assert medium["buckets"] == [1, 2]


def test_one_factor_gives_its_weighted_sensitivity_and_a_matrix_rho_reads_within_buckets_alone():
    """BCBS MAR21: a single risk factor's charge is |RW s|; rho given as a matrix uses its within-bucket entries."""
    one = qf.frtb_sbm(sensitivities=[-250.0], risk_weights=0.016, buckets=["USD"], rho=0.5, gamma=0.3, scenario="max",
                      alternative="if_negative")
    assert one["charge"] == pytest.approx(4.0, rel=1e-14)
    matrix = [[1.0, 0.2, 0.99], [0.2, 1.0, 0.99], [0.99, 0.99, 1.0]]
    a = qf.frtb_sbm(sensitivities=[100.0, -40.0, 60.0], risk_weights=[0.3, 0.3, 0.5], buckets=[1, 1, 2], rho=matrix,
                    gamma=[[1.0, 0.15], [0.15, 1.0]], scenario="medium", alternative="if_negative")
    b = qf.frtb_sbm(sensitivities=[100.0, -40.0, 60.0], risk_weights=[0.3, 0.3, 0.5], buckets=[1, 1, 2], rho=0.2,
                    gamma=0.15, scenario="medium", alternative="if_negative")
    assert a["charge"] == pytest.approx(b["charge"], rel=1e-14)


def test_the_alternative_caps_s_b_where_the_sum_turns_negative():
    """Two buckets of two uncorrelated factors each, the buckets of opposite sign and gamma 0.9: S_b = +-20 exceeds K_b
    = sqrt(200), the sum 400 - 720 is negative, and the capped S_b = +-sqrt(200) gives 400 - 360."""
    args = dict(sensitivities=[10.0, 10.0, -10.0, -10.0], risk_weights=1.0, buckets=["a", "a", "b", "b"], rho=0.0,
                gamma=0.9, scenario="medium")
    found = qf.frtb_sbm(**args, alternative="if_negative")
    assert found["charge"] == pytest.approx(math.sqrt(40.0), rel=1e-12)
    assert found["charge"] == pytest.approx(by_hand([10.0, 10.0, -10.0, -10.0], [[0, 1], [2, 3]], 0.0, 0.9), rel=1e-12)
    assert found["sb"] == pytest.approx([math.sqrt(200.0), -math.sqrt(200.0)])
    positive = dict(args, gamma=0.1)
    assert qf.frtb_sbm(**positive, alternative="if_negative")["charge"] == pytest.approx(math.sqrt(400 - 80.0))
    assert qf.frtb_sbm(**positive, alternative="always")["charge"] == pytest.approx(math.sqrt(400 - 40.0))


def test_the_liquidity_adjusted_es_by_mar33_and_the_floored_stress_ratio():
    """MAR33.4: ES = sqrt(sum_j (ES_j sqrt((LH_j - LH_{j-1}) / T))^2) on Basel's horizons 10, 20, 40, 60 and 120 days;
    one class at the base horizon is its own ES; MAR33.6: times ES_F,C / ES_R,C, that ratio floored at 1."""
    es, lh = [12.0, 8.0, 5.0, 3.0, 1.0], [10, 20, 40, 60, 120]
    found = qf.frtb_es(es=es, horizons=lh, base_horizon=10, full_current=None, reduced_current=None)
    assert found["es"] == pytest.approx(math.sqrt(144 + 64 + 25 * 2 + 9 * 2 + 1 * 6), rel=1e-14)
    assert found["ratio"] == 1.0 and found["unscaled"] == found["es"]
    assert qf.frtb_es(es=[12.0], horizons=[10], base_horizon=10, full_current=None,
                      reduced_current=None)["es"] == pytest.approx(12.0, rel=1e-15)
    scaled = qf.frtb_es(es=es, horizons=lh, base_horizon=10, full_current=10.0, reduced_current=8.0)
    assert scaled["ratio"] == 1.25 and scaled["es"] == pytest.approx(1.25 * found["es"], rel=1e-14)
    floored = qf.frtb_es(es=es, horizons=lh, base_horizon=10, full_current=7.0, reduced_current=8.0)
    assert floored["ratio"] == 1.0 and floored["es"] == found["es"]
    with pytest.raises(ValueError):
        qf.frtb_es(es=es, horizons=lh, base_horizon=10, full_current=7.0, reduced_current=None)
    with pytest.raises(ValueError):
        qf.frtb_es(es=es, horizons=[10, 20, 20, 60, 120], base_horizon=10, full_current=None, reduced_current=None)
