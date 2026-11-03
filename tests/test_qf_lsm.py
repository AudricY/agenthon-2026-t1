"""qf.montecarlo.longstaff_schwartz (runs/lanes/lib7-prompt.md, rank 13 of docs/research/library-syllabus-1011.md):
the American put of Longstaff and Schwartz (2001), Table 1, against a binomial tree with exercise on the same dates
(Bermudan) and at every step (American), within the estimate's standard errors; the bases' span and the in-sample
rule. Every input is the paper's or this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf.montecarlo as MC

pytestmark = pytest.mark.fast

S0, K, R, VOL, T, DATES = 36.0, 40.0, 0.06, 0.2, 1.0, 50
TIMES = np.linspace(0.0, T, DATES + 1)


def put(s):
    return np.maximum(K - s, 0.0)


def binomial_put(steps, exercise_every):
    """CRR tree (Cox, Ross and Rubinstein 1979), exercise allowed every `exercise_every` steps (1: American)."""
    dt = T / steps
    u = math.exp(VOL * math.sqrt(dt))
    p = (math.exp(R * dt) - 1 / u) / (u - 1 / u)
    s = S0 * u ** (2.0 * np.arange(steps + 1) - steps)
    v = put(s)
    for n in range(steps - 1, -1, -1):
        s = S0 * u ** (2.0 * np.arange(n + 1) - n)
        v = math.exp(-R * dt) * (p * v[1:] + (1 - p) * v[:-1])
        if n % exercise_every == 0 and n > 0:
            v = np.maximum(v, put(s))
    return float(v[0])


def lsm(paths, basis="weighted_laguerre", degree=3, itm_only=True, pricing_paths=None):
    return MC.longstaff_schwartz(paths, TIMES, put, R, basis, degree, K, itm_only, pricing_paths)


PATHS = MC.gbm_paths(S0, R, VOL, TIMES, 20000, 2001, "arithmetic", False)
FRESH = MC.gbm_paths(S0, R, VOL, TIMES, 20000, 2002, "arithmetic", False)


def test_the_put_of_table_1_meets_the_tree_with_the_same_exercise_dates():
    """S = 36, K = 40, r = 6%, vol 20%, T = 1, 50 dates a year (Longstaff and Schwartz, Table 1: 4.472, s.e. 0.010,
    beside a finite-difference 4.478). The tree with exercise on the same 50 dates gives 4.4780, the paper's
    finite-difference figure to its digits; exercise at every step 4.4867, the American, above it. The estimate lies
    within three standard errors of the first, in sample and on fresh paths, and of the paper's."""
    bermudan = binomial_put(2500, 50)
    american = binomial_put(2500, 1)
    assert bermudan == pytest.approx(4.478, abs=5e-4) and american == pytest.approx(4.4867, abs=5e-4)
    inside, outside = lsm(PATHS), lsm(PATHS, pricing_paths=FRESH)
    for found in (inside, outside):
        assert abs(found["price"] - bermudan) < 3 * found["std_error"] and found["std_error"] < 0.025
    assert abs(inside["price"] - 4.472) < 3 * (inside["std_error"] + 0.010)


EIGHT = np.array([[1.00, 1.09, 1.08, 1.34], [1.00, 1.16, 1.26, 1.54], [1.00, 1.22, 1.07, 1.03],
                  [1.00, 0.93, 0.97, 0.92], [1.00, 1.11, 1.56, 1.52], [1.00, 0.76, 0.77, 0.90],
                  [1.00, 0.92, 0.84, 1.01], [1.00, 0.88, 1.22, 1.34]])


def eight_put(s):
    return np.maximum(1.10 - s, 0.0)


def test_longstaff_and_schwartzs_eight_path_example():
    """Their Section 1: eight paths, K = 1.10, r = 6% a period, exercise at 1, 2, 3; regressions on 1, X, X^2 over
    the paths in the money: E[Y|X] = -1.070 + 2.983 X - 1.813 X^2 at time 2, 2.038 - 3.335 X + 1.356 X^2 at time 1;
    paths 4, 6, 7, 8 exercise at 1 and path 3 at 3; the value 0.1144."""
    found = MC.longstaff_schwartz(EIGHT, [0.0, 1.0, 2.0, 3.0], eight_put, 0.06, "polynomial", 2, 1.0, True, None)
    assert found["coefficients"][1] == pytest.approx([-1.070, 2.983, -1.813], abs=1e-3)   # the paper truncates
    assert found["coefficients"][0] == pytest.approx([2.038, -3.335, 1.356], abs=5e-4)
    assert found["price"] == pytest.approx(0.1144, abs=5e-5)
    assert found["exercise_times"].tolist() == [np.inf, np.inf, 3.0, 1.0, np.inf, 1.0, 1.0, 1.0]


def test_the_weighted_laguerre_regression_by_hand_at_the_last_date():
    """At time 2 of the eight paths: the cash flow at 3 discounted one period, regressed on 1 and e^{-x/2} L_k(x),
    L_0 = 1, L_1 = 1 - x, L_2 = 1 - 2x + x^2 / 2, x = S / 1.10, over the paths in the money."""
    found = MC.longstaff_schwartz(EIGHT, [0.0, 1.0, 2.0, 3.0], eight_put, 0.06, "weighted_laguerre", 3, 1.10, True,
                                  None)
    s2 = EIGHT[:, 2]
    keep = eight_put(s2) > 0
    x = s2[keep] / 1.10
    w = np.exp(-x / 2)
    design = np.column_stack([np.ones_like(x), w, w * (1 - x), w * (1 - 2 * x + x * x / 2)])
    expected = np.linalg.lstsq(design, eight_put(EIGHT[keep, 3]) * math.exp(-0.06), rcond=None)[0]
    assert found["coefficients"][1] == pytest.approx(expected, rel=1e-9)


def by_hand(paths, times, itm_only, degree):
    """The paper's backward recursion written out path by path: a cash-flow row and its date per path; at each date
    from the last but one down, the regression of each kept path's cash flow discounted to that date on 1, S, ...,
    S^d, then exercise where the payoff is positive and above the fit."""
    n, m = paths.shape
    cash = [eight_put(paths[i, -1]) for i in range(n)]
    date = [m - 1] * n
    rows = {}
    for j in range(m - 2, 0, -1):
        kept = [i for i in range(n) if eight_put(paths[i, j]) > 0 or not itm_only]
        if len(kept) <= degree:
            continue
        design = np.array([[paths[i, j] ** p for p in range(degree + 1)] for i in kept])
        target = np.array([cash[i] * math.exp(-0.06 * (times[date[i]] - times[j])) for i in kept])
        rows[j] = beta = np.linalg.lstsq(design, target, rcond=None)[0]
        for i, x in zip(kept, design):
            if eight_put(paths[i, j]) > 0 and eight_put(paths[i, j]) > x @ beta:
                cash[i], date[i] = eight_put(paths[i, j]), j
    return rows


@pytest.mark.parametrize("itm_only", [True, False])
def test_the_regressions_are_the_recursion_written_out(itm_only):
    times = np.linspace(0.0, 1.5, 7)
    paths = MC.gbm_paths(1.0, 0.06, 0.25, times, 300, 32, "arithmetic", False)
    found = MC.longstaff_schwartz(paths, times, eight_put, 0.06, "polynomial", 3, 1.0, itm_only, None)
    expected = by_hand(paths, times, itm_only, 3)
    for j in range(1, 6):
        assert found["coefficients"][j - 1] == pytest.approx(expected[j], rel=1e-8, abs=1e-10)


def test_a_date_with_no_more_paths_in_the_money_than_coefficients_is_not_regressed():
    """At degree 5 the eight paths' five in the money at each date fit no regression: none is exercised early, and
    the value is the European's on the paths."""
    found = MC.longstaff_schwartz(EIGHT, [0.0, 1.0, 2.0, 3.0], eight_put, 0.06, "polynomial", 5, 1.0, True, None)
    assert np.all(np.isnan(found["coefficients"]))
    assert found["price"] == pytest.approx(math.exp(-0.18) * eight_put(EIGHT[:, 3]).mean(), rel=1e-14)


def test_the_polynomial_and_laguerre_bases_span_one_space_so_give_one_price():
    a, b = lsm(PATHS[:5000], "polynomial", 3), lsm(PATHS[:5000], "laguerre", 3)
    assert a["price"] == pytest.approx(b["price"], rel=1e-9)
    assert lsm(PATHS[:5000], "polynomial", 3)["price"] == pytest.approx(MC.longstaff_schwartz(
        PATHS[:5000], TIMES, put, R, "polynomial", 3, 1.0, True, None)["price"], rel=1e-9)


def test_the_in_sample_rule_is_the_backward_inductions_cash_flow_and_exercise_is_only_in_the_money():
    """Priced again on its own paths through pricing_paths, the in-sample estimate is unchanged; every exercise is
    where the payoff is positive; a path never in the money pays nothing."""
    found = lsm(PATHS[:4000])
    assert lsm(PATHS[:4000], pricing_paths=PATHS[:4000])["price"] == found["price"]
    stopped = np.isfinite(found["exercise_times"])
    columns = np.searchsorted(TIMES, found["exercise_times"][stopped])
    assert np.all(put(PATHS[:4000][stopped, columns]) > 0)
    never = np.all(put(PATHS[:4000][:, 1:]) == 0, axis=1)
    assert np.all(~stopped[never])
    assert found["coefficients"].shape == (DATES - 1, 4)


def test_regressing_on_every_path_moves_the_price_and_a_european_is_never_exercised_early():
    every = lsm(PATHS[:5000], itm_only=False)
    stopped = np.isfinite(every["exercise_times"])
    assert np.all(put(PATHS[:5000][stopped, np.searchsorted(TIMES, every["exercise_times"][stopped])]) > 0)
    assert abs(every["price"] - binomial_put(2500, 50)) < 3 * every["std_error"] + 0.03
    european = MC.longstaff_schwartz(PATHS[:, [0, -1]], TIMES[[0, -1]], put, R, "polynomial", 2, K, True, None)
    assert european["price"] == pytest.approx(math.exp(-R * T) * put(PATHS[:, -1]).mean(), rel=1e-14)
    assert np.all(np.isnan(european["coefficients"]))


@pytest.mark.parametrize("bad, match", [(dict(basis="hermite"), "basis"), (dict(degree=0), "degree"),
                                        (dict(itm_only="yes"), "itm_only"), (dict(scale=0.0), "scale"),
                                        (dict(pricing_paths=np.ones((3, 4))), "pricing_paths")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(paths=PATHS[:100], times=TIMES, payoff=put, rate=R, basis="polynomial", degree=2, scale=K,
                itm_only=True, pricing_paths=None)
    with pytest.raises(ValueError, match=match):
        MC.longstaff_schwartz(**{**args, **bad})
