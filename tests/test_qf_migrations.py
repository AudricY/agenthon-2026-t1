"""Lane lib-risk1 (runs/lanes/lib-risk1-prompt.md), the `migrations` shelf: Lando and Skodeberg's (2002) duration
estimator on a history built by hand (each exposure and count written out), the Aalen-Johansen matrix against the
cohort matrix where every obligor is watched throughout (Aalen and Johansen 1978), the estimate against a simulated
chain's own generator, censoring, the mean time to default the card writes, and the homogeneity test the cohort
card names (scipy's chi2_contingency without the continuity correction against the statistic by hand). Synthetic
histories alone; no unit's."""
import numpy as np
import pytest
from scipy import linalg, stats

import qf

pytestmark = pytest.mark.fast

STATES = ["A", "B", "D"]
HAND = [[(0.0, "A"), (1.5, "B")], [(0.0, "B"), (0.7, "D")], [(0.0, "A")], [(0.0, "B"), (1.2, "A")]]


def test_the_duration_estimate_on_a_hand_built_history():
    """Watched to 2.0: A for 1.5 + 2.0 + 0.8 = 4.3 years and B for 0.5 + 0.7 + 1.2 = 2.4 (default absorbs); moves A->B,
    B->D, B->A once each. q_AB = 1/4.3, q_BA = q_BD = 1/2.4; A, never seen to default, still has a default rate in
    expm(Q)."""
    found = qf.generator_mle(histories=HAND, states=STATES, absorbing=["D"], censored=[], end=2.0, method="duration",
                             horizon=1.0)
    q = np.array([[-1 / 4.3, 1 / 4.3, 0.0], [1 / 2.4, -2 / 2.4, 1 / 2.4], [0.0, 0.0, 0.0]])
    assert found["exposure"] == pytest.approx([4.3, 2.4, 0.0], rel=1e-14)
    assert found["counts"].tolist() == [[0, 1, 0], [1, 0, 1], [0, 0, 0]]
    assert found["generator"] == pytest.approx(q, rel=1e-14)
    assert found["matrix"] == pytest.approx(linalg.expm(q), rel=1e-12)
    two = qf.generator_mle(histories=HAND, states=STATES, absorbing=["D"], censored=[], end=2.0, method="duration",
                           horizon=2.0)
    assert two["matrix"] == pytest.approx(linalg.expm(2.0 * q), rel=1e-12)
    assert found["matrix"][0, 2] > 0 and found["generator"].sum(axis=1) == pytest.approx(0.0, abs=1e-15)


def test_censoring_stops_the_watch_without_a_move():
    found = qf.generator_mle(histories=[[(0.0, "A"), (1.0, "NR")], [(0.0, "A"), (0.5, "B")]], states=STATES,
                             absorbing=["D"], censored=["NR"], end=2.0, method="duration", horizon=1.0)
    assert found["exposure"] == pytest.approx([1.5, 1.5, 0.0]) and found["counts"][0, 1] == 1
    with pytest.raises(ValueError):
        qf.generator_mle(histories=[[(0.0, "A"), (1.0, "NR")]], states=STATES, absorbing=["D"], censored=[], end=2.0,
                         method="duration", horizon=1.0)


def test_the_aalen_johansen_product_on_the_hand_built_history():
    """Aalen and Johansen (1978): the product over the move times of I + dA, dA_ij = moves i -> j over those in i just
    before. At 0.7 (B -> D) two are in A and two in B; at 1.2 (B -> A) one is in B; at 1.5 (A -> B) three are in A."""
    found = qf.generator_mle(histories=HAND, states=STATES, absorbing=["D"], censored=[], end=2.0,
                             method="aalen_johansen", horizon=2.0)
    steps = [np.array([[1, 0, 0], [0, 0.5, 0.5], [0, 0, 1.0]]), np.array([[1, 0, 0], [1.0, 0, 0], [0, 0, 1.0]]),
             np.array([[2 / 3, 1 / 3, 0], [0, 1, 0], [0, 0, 1.0]])]
    assert found["matrix"] == pytest.approx(steps[0] @ steps[1] @ steps[2], abs=1e-15)
    short = qf.generator_mle(histories=HAND, states=STATES, absorbing=["D"], censored=[], end=2.0,
                             method="aalen_johansen", horizon=1.0)
    assert short["matrix"] == pytest.approx(steps[0], abs=1e-15)


def test_aalen_johansen_is_the_cohort_matrix_where_obligors_only_default_and_are_watched_throughout():
    """With complete watching and default the only move, each row is Kaplan-Meier's, which is the cohort fraction."""
    rng = np.random.default_rng(3)
    histories, start, end = [], [], []
    for k in range(60):
        rating = "A" if k % 2 else "B"
        when = rng.exponential(4.0 if rating == "A" else 1.5)
        histories.append([(0.0, rating)] + ([(when, "D")] if when <= 1.0 else []))
        start.append(rating)
        end.append("D" if when <= 1.0 else rating)
    found = qf.generator_mle(histories=histories, states=STATES, absorbing=["D"], censored=[], end=1.0,
                             method="aalen_johansen", horizon=1.0)
    cohort, _ = qf.cohort_transition_matrix(start, end, states=STATES, absorbing=["D"])
    assert found["matrix"] == pytest.approx(cohort, abs=1e-12) and 0 < cohort[1, 2] < 1


Q = np.array([[-0.15, 0.12, 0.03], [0.1, -0.3, 0.2], [0.0, 0.0, 0.0]])


def simulated(q, obligors, seed):
    """Each obligor's history of a continuous-time chain on A, B, D over 4 years, D absorbing."""
    rng = np.random.default_rng(seed)
    histories = []
    for _ in range(obligors):
        state, time, path = int(rng.integers(0, 2)), 0.0, []
        path.append((0.0, STATES[state]))
        while state != 2:
            time += rng.exponential(1 / -q[state, state])
            if time > 4.0:
                break
            state = int(rng.choice(3, p=np.maximum(q[state], 0) / -q[state, state]))
            path.append((time, STATES[state]))
        histories.append(path)
    return histories, rng


def test_the_estimate_is_near_a_simulated_chains_generator():
    histories, _ = simulated(Q, 1500, 8)
    found = qf.generator_mle(histories=histories, states=STATES, absorbing=["D"], censored=[], end=4.0,
                             method="duration", horizon=1.0)
    assert found["generator"] == pytest.approx(Q, abs=0.03)


@pytest.mark.slow
def test_the_estimate_recovers_a_simulated_chains_generator_and_the_mean_time_to_default():
    """A continuous-time chain on A, B, D (default absorbing) simulated for 5,000 obligors over 4 years: the duration
    estimate is within sampling error of Q; the card's mean time to default, solve(-Q_T, 1), is the simulated mean."""
    q = Q
    rng = np.random.default_rng(8)
    histories = []
    for _ in range(5000):
        state, time, path = int(rng.integers(0, 2)), 0.0, []
        path.append((0.0, STATES[state]))
        while state != 2:
            time += rng.exponential(1 / -q[state, state])
            if time > 4.0:
                break
            state = int(rng.choice(3, p=np.maximum(q[state], 0) / -q[state, state]))
            path.append((time, STATES[state]))
        histories.append(path)
    found = qf.generator_mle(histories=histories, states=STATES, absorbing=["D"], censored=[], end=4.0,
                             method="duration", horizon=1.0)
    assert found["generator"] == pytest.approx(q, abs=0.015)
    mean = np.linalg.solve(-q[:-1, :-1], np.ones(2))
    times = []
    for start in (0, 1):
        for _ in range(3000):
            state, time = start, 0.0
            while state != 2:
                time += rng.exponential(1 / -q[state, state])
                state = int(rng.choice(3, p=np.maximum(q[state], 0) / -q[state, state]))
            times.append(time)
    assert np.mean(times[:3000]) == pytest.approx(mean[0], rel=0.05)
    assert np.mean(times[3000:]) == pytest.approx(mean[1], rel=0.05)


def test_the_homogeneity_test_the_cohort_card_names_is_pearsons_statistic_without_correction():
    """Two periods' moves out of one rating, an all-zero end-state column dropped first (cohort card's line): scipy's
    chi2_contingency(table, correction=False) is sum (O - E)^2 / E with E from the margins."""
    table = np.array([[80, 15, 5, 0], [70, 22, 8, 0]])
    table = table[:, table.sum(axis=0) > 0]
    statistic, p, dof, expected = stats.chi2_contingency(table, correction=False)
    by_hand = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    assert statistic == pytest.approx(np.sum((table - by_hand) ** 2 / by_hand), rel=1e-12) and dof == 2
    assert "chi2_contingency(table, correction=False)" in qf.cohort_transition_matrix.__doc__
