"""qf.econometrics.corrado_rank_test (runs/lanes/lib7-prompt.md, rank 27 of docs/research/library-syllabus-1011.md): a
small event set ranked by hand, ties averaged, the two versions equal without a missing return and apart with one
(Corrado and Zivney's standardised ranks by hand), the window statistic, and near-standard-normal statistics on
simulated nulls (Corrado 1989; Campbell, Lo and MacKinlay 1997, 4.6). Every return is this file's own; no unit's
values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf
import qf.econometrics as e

pytestmark = pytest.mark.fast

DAYS = [-4, -3, -2, -1, 0, 1]
A = np.array([[0.010, -0.020, 0.004, 0.000, 0.030, -0.010],
              [-0.005, 0.012, 0.012, -0.002, 0.020, 0.001]])


def test_a_small_event_set_ranked_by_hand():
    """Event 1's ranks over its six days 5, 1, 4, 3, 6, 2; event 2's 1, 4.5, 4.5, 2, 6, 3 (a tie averaged); centred
    at 3.5 and averaged over the two events: x = -0.5, -0.75, 0.75, -1, 2.5, -1; S = sqrt(mean x^2); day 0's x / S
    and day 1's."""
    found = qf.corrado_rank_test(A, DAYS, (-4, -1), (0, 1), "day", "corrado")
    x = np.array([-0.5, -0.75, 0.75, -1.0, 2.5, -1.0])
    s = math.sqrt(np.mean(x * x))
    assert found["x"] == pytest.approx(x) and found["s"] == pytest.approx(s)
    assert found["statistic"] == pytest.approx([2.5 / s, -1.0 / s], rel=1e-14)
    assert found["p_value"] == pytest.approx(2 * norm.sf(np.abs([2.5 / s, -1.0 / s])), rel=1e-12)
    assert found["event_days"].tolist() == [0, 1]
    window = qf.corrado_rank_test(A, DAYS, (-4, -1), (0, 1), "window", "corrado")
    assert window["statistic"] == pytest.approx(1.5 / (math.sqrt(2) * s), rel=1e-14)


def test_the_versions_agree_without_a_gap_and_part_with_one():
    """With event 2's day -1 return missing: its five rank 1, 3.5, 3.5, 5, 2, U = K / 6 (M = 5), event 1's U = K / 7;
    each day's sum over the events present scaled by 1/sqrt(N_t)."""
    one = qf.corrado_rank_test(A, DAYS, (-4, -1), (0, 1), "day", "corrado")
    two = qf.corrado_rank_test(A, DAYS, (-4, -1), (0, 1), "day", "corrado_zivney")
    assert one["statistic"] == pytest.approx(two["statistic"], rel=1e-13)
    gap = A.copy()
    gap[1, 3] = np.nan
    found = qf.corrado_rank_test(gap, DAYS, (-4, -1), (0, 1), "day", "corrado_zivney")
    k1 = np.array([5, 1, 4, 3, 6, 2]) / 7
    k2 = np.array([1, 3.5, 3.5, np.nan, 5, 2]) / 6
    x = np.array([(k1[t] - 0.5 + (0 if np.isnan(k2[t]) else k2[t] - 0.5)) / math.sqrt(1 if np.isnan(k2[t]) else 2)
                  for t in range(6)])
    s = math.sqrt(np.mean(x * x))
    assert found["x"] == pytest.approx(x, rel=1e-13) and found["statistic"] == pytest.approx(x[4:] / s, rel=1e-13)
    assert qf.corrado_rank_test(gap, DAYS, (-4, -1), (0, 1), "day", "corrado")["statistic"] != pytest.approx(
        found["statistic"])


def test_near_standard_normal_on_simulated_nulls_and_large_with_an_event_day_shock():
    rng = np.random.default_rng(14)
    days = list(range(-60, 1))
    stats = [qf.corrado_rank_test(rng.standard_normal((20, 61)), days, (-60, -1), (0, 0), "day", "corrado")
             ["statistic"][0] for _ in range(200)]
    assert abs(np.mean(stats)) < 0.15 and 0.85 < np.std(stats) < 1.15
    shocked = rng.standard_normal((20, 61))
    shocked[:, -1] += 2.0
    assert qf.corrado_rank_test(shocked, days, (-60, -1), (0, 0), "day", "corrado")["statistic"][0] > 4


@pytest.mark.parametrize("bad, match", [
    (dict(statistic="cumulative"), "statistic"), (dict(version="kolari_pynnonen"), "version"),
    (dict(event_window=(-2, 0)), "estimation window first"), (dict(event_time=[0, 1, 2, 3, 4, 4]), "event days"),
    (dict(event_window=(5, 6)), "none of")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(abnormal_returns=A, event_time=DAYS, estimation_window=(-4, -1), event_window=(0, 1), statistic="day",
                version="corrado")
    with pytest.raises(ValueError, match=match):
        qf.corrado_rank_test(**{**args, **bad})
