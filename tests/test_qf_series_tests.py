"""Lane lib-econometrics1 (runs/lanes/lib-econometrics1-prompt.md): qf.variance_ratio, qf.johansen and
qf.multiple_testing. Against the arch package 8.0.0's VarianceRatio and statsmodels 0.15.0's coint_johansen and
multipletests on this file's own series (frozen in `PACKAGES`; `reference_values()` recomputes them where both import,
the slow test compares), and the identities: VR(2) = 1 + rho_1 less its end terms, exactly (Lo and MacKinlay 1988);
Johansen's eigenvalues as the squared canonical correlations of the differences and the lagged levels, and the
critical values' first row the chi-square(1) quantiles (MacKinnon, Haug and Michelis 1999); one test's p unchanged
and the step-down and step-up orderings (Holm 1979; Benjamini and Hochberg 1995). No unit's values appear (rule 8)."""
import itertools
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401  (first: it puts agent/ on the path outside pytest too)
import qf

pytestmark = pytest.mark.fast


def walk():
    """201 log prices of a random walk with drift, the legacy generator (a frozen stream)."""
    rng = np.random.RandomState(41)
    return np.log(50.0) + np.cumsum(np.r_[0.0, 0.0004 + 0.01 * rng.normal(size=200) * (1 + 0.5 * (np.arange(200) > 120))])


def levels():
    """150 dates of three series, two sharing a random-walk trend, the legacy generator."""
    rng = np.random.RandomState(42)
    common = np.cumsum(rng.normal(size=150))
    return np.column_stack([common + 0.5 * rng.normal(size=150), 0.5 * common + 0.4 * rng.normal(size=150),
                            np.cumsum(rng.normal(size=150))])


WALK, LEVELS = walk(), levels()
P_VALUES = [0.21, 0.012, 0.6, 0.039, 0.0004, 0.045, 0.31, 0.07, 0.95, 0.041, 0.2, 0.009]
VR_CASES = [(True, True, "robust", True), (True, True, "homoskedastic", True), (True, False, "robust", False),
            (True, False, "homoskedastic", True), (False, False, "homoskedastic", True),
            (False, False, "homoskedastic", False)]
CORRECTIONS = {"bonferroni": "bonferroni", "sidak": "sidak", "holm": "holm", "hochberg": "simes-hochberg",
               "benjamini_hochberg": "fdr_bh", "benjamini_yekutieli": "fdr_by"}
PACKAGES = {   # arch 8.0.0 and statsmodels 0.15.0 on WALK, LEVELS and P_VALUES (reference_values)
    ('vr', 2, True, True, 'robust', True): [0.9869452331410621, -0.157468070726232, 0.8748759698027282],
    ('vr', 2, True, True, 'homoskedastic', True): [0.9869452331410621, -0.1846222834552881, 0.8535253067270605],
    ('vr', 2, True, False, 'robust', False): [0.9782690386759053, -0.26147810734768695, 0.7937238302265017],
    ('vr', 2, True, False, 'homoskedastic', True): [0.9770757808096515, -0.324197416858044, 0.7457885776109321],
    ('vr', 2, False, False, 'homoskedastic', True): [1.0710735089172934, 0.7071724848379222, 0.4794592953384149],
    ('vr', 2, False, False, 'homoskedastic', False): [1.071979286449423, 0.7161848574907472, 0.4738772118924457],
    ('vr', 4, True, True, 'robust', True): [1.0219147507102697, 0.14766593274307413, 0.8826064185651017],
    ('vr', 4, True, True, 'homoskedastic', True): [1.0219147507102697, 0.16565994406671364, 0.8684245704138169],
    ('vr', 4, True, False, 'robust', False): [0.9947984005944247, -0.034985413427946414, 0.972091372151751],
    ('vr', 4, True, False, 'homoskedastic', True): [0.9914113656890686, -0.0649239728240029, 0.9482345333047806],
    ('vr', 4, False, False, 'homoskedastic', True): [1.2184779960326573, 1.2487057573802418, 0.21177271430554412],
    ('vr', 4, False, False, 'homoskedastic', False): [1.22092765766932, 1.2627067398355962, 0.20669459097821097],
    ('johansen', 'none', 1): [[66.62924483463999, 3.6835885986393393, 0.5517121297658734], [62.94565623600066, 3.131876468873466, 0.5517121297658734], [0.3464318639518708, 0.020938997628428175, 0.0037208450970865493]],
    ('johansen', 'none', 2): [[47.935623560745874, 3.9673270905107816, 0.7698491607551504], [43.9682964702351, 3.1974779297556313, 0.7698491607551504], [0.25851775145804456, 0.021516691498657824, 0.005223379582407515]],
    ('johansen', 'constant', 1): [[70.61918318597598, 6.036470229480283, 1.9757327904791577], [64.5827129564957, 4.060737439001126, 1.9757327904791577], [0.35362127290874235, 0.027064428304092218, 0.013260835878826161]],
    ('johansen', 'constant', 2): [[50.76382042786422, 6.318991530401804, 1.8753485710908393], [44.44482889746241, 4.4436429593109645, 1.8753485710908393], [0.26091753539873724, 0.029776540763312754, 0.012676441662799501]],
    ('johansen', 'linear_trend', 1): [[74.29981072200938, 9.602441548735483, 3.7400639813240657], [64.6973691732739, 5.862377567411418, 3.7400639813240657], [0.3541218312849868, 0.0388364135258572, 0.024954071145678255]],
    ('johansen', 'linear_trend', 2): [[53.49346788529164, 8.396345607553766, 3.7789097060983075], [45.097122277737874, 4.617435901455457, 3.7789097060983075], [0.26418985209653445, 0.03092292411856038, 0.025379260463094334]],
    ('tests', 'bonferroni'): [[1.0, 0.14400000000000002, 1.0, 0.46799999999999997, 0.0048000000000000004, 0.54, 1.0, 0.8400000000000001, 1.0, 0.492, 1.0, 0.10799999999999998], [False, False, False, False, True, False, False, False, False, False, False, False]],
    ('tests', 'sidak'): [[0.9409084889683258, 0.13486609002406186, 0.999983222784, 0.37958733903447234, 0.004789454067336107, 0.424506452007523, 0.9883536700772226, 0.5814037025206293, 0.9999999999999998, 0.39490538768386496, 0.931280523264, 0.1028111785845234], [False, False, False, False, True, False, False, False, False, False, False, False]],
    ('tests', 'holm'): [[1.0, 0.12, 1.0, 0.351, 0.0048000000000000004, 0.351, 1.0, 0.42000000000000004, 1.0, 0.351, 1.0, 0.09899999999999999], [False, False, False, False, True, False, False, False, False, False, False, False]],
    ('tests', 'hochberg'): [[0.84, 0.12, 0.95, 0.315, 0.0048000000000000004, 0.315, 0.9299999999999999, 0.42000000000000004, 0.95, 0.315, 0.84, 0.09899999999999999], [False, False, False, False, True, False, False, False, False, False, False, False]],
    ('tests', 'benjamini_hochberg'): [[0.27999999999999997, 0.048, 0.6545454545454545, 0.09, 0.0048000000000000004, 0.09, 0.372, 0.12000000000000001, 0.95, 0.09, 0.27999999999999997, 0.048], [False, True, False, False, True, False, False, False, False, False, False, True]],
    ('tests', 'benjamini_yekutieli'): [[0.8688989898989898, 0.14895411255411256, 1.0, 0.27928896103896106, 0.014895411255411257, 0.27928896103896106, 1.0, 0.3723852813852814, 1.0, 0.27928896103896106, 0.8688989898989898, 0.14895411255411256], [False, False, False, False, True, False, False, False, False, False, False, False]],
}


def reference_values():
    """PACKAGES again, from arch and statsmodels (the lane's scratch/lib-econometrics1/refenv)."""
    from arch.unitroot import VarianceRatio
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    found = {}
    for q in (2, 4):
        for overlap, debiased, z, demean in VR_CASES:
            y = WALK if overlap else WALK[:-1]
            vr = VarianceRatio(y, lags=q, trend="c" if demean else "n", debiased=debiased, robust=z == "robust",
                               overlap=overlap)
            found[("vr", q, overlap, debiased, z, demean)] = [float(vr.vr), float(vr.stat), float(vr.pvalue)]
    for det, case in ((-1, "none"), (0, "constant"), (1, "linear_trend")):
        for k in (1, 2):
            result = coint_johansen(LEVELS, det, k)
            found[("johansen", case, k)] = [result.lr1.tolist(), result.lr2.tolist(), np.real(result.eig).tolist()]
    for name, method in CORRECTIONS.items():
        reject, adjusted, _, _ = multipletests(P_VALUES, alpha=0.05, method=method)
        found[("tests", name)] = [adjusted.tolist(), reject.tolist()]
    return found


@pytest.mark.slow
def test_the_frozen_values_are_the_packages_own():
    pytest.importorskip("arch")
    pytest.importorskip("statsmodels")
    found = reference_values()
    assert set(found) == set(PACKAGES)
    for key, value in PACKAGES.items():
        if key[0] == "tests":
            assert found[key][1] == value[1] and found[key][0] == pytest.approx(value[0], rel=1e-12), key
        else:
            assert np.asarray(found[key], dtype=float) == pytest.approx(np.asarray(value, dtype=float), rel=1e-11), key


# ------------------------------------------------------------------------------------------------ variance ratio


@pytest.mark.parametrize("q", [2, 4])
@pytest.mark.parametrize("overlapping, bias_correction, z, demean", VR_CASES)
def test_the_variance_ratio_is_archs(q, overlapping, bias_correction, z, demean):
    y = WALK if overlapping else WALK[:-1]
    found = qf.variance_ratio(log_prices=y, q=q, overlapping=overlapping, bias_correction=bias_correction, z=z,
                              demean=demean)
    vr, statistic, p_value = PACKAGES[("vr", q, overlapping, bias_correction, z, demean)]
    assert found["vr"] == pytest.approx(vr, rel=1e-12)
    assert found["z"] == pytest.approx(statistic, rel=1e-10, abs=1e-12)
    assert found["p_value"] == pytest.approx(p_value, rel=1e-10)
    assert found["variance_q"] / (q * found["variance_one"]) == pytest.approx(found["vr"], rel=1e-12)


def test_vr2_is_one_plus_the_first_autocorrelation_less_its_end_terms():
    """Overlapping, no bias correction: sigma_c^2(2) sums the nq - 1 two-period changes, so VR(2) = 1 + C/S -
    (e_1^2 + e_n^2)/(2S), S = sum e_t^2 and C = sum e_t e_{t-1} of the demeaned returns: Lo and MacKinlay's VR(2) =
    1 + rho_1 in the limit."""
    e = np.diff(WALK) - (WALK[-1] - WALK[0]) / (WALK.size - 1)
    s, c = e @ e, e[1:] @ e[:-1]
    found = qf.variance_ratio(log_prices=WALK, q=2, overlapping=True, bias_correction=False, z="homoskedastic",
                              demean=True)
    assert found["vr"] == pytest.approx(1 + c / s - (e[0] ** 2 + e[-1] ** 2) / (2 * s), rel=1e-13)


def test_a_random_walk_has_a_ratio_near_one_and_autocorrelation_moves_it_either_way():
    rng = np.random.RandomState(43)
    shocks = rng.normal(size=20001) * 0.01
    for phi, side in ((0.0, 0), (0.3, 1), (-0.3, -1)):
        r = shocks.copy()
        r[1:] = shocks[1:] + phi * shocks[:-1]
        found = qf.variance_ratio(log_prices=np.r_[0.0, np.cumsum(r)], q=2, overlapping=True, bias_correction=True,
                                  z="robust", demean=True)
        target = 1 + phi / (1 + phi * phi)          # an MA(1)'s rho_1 = theta / (1 + theta^2)
        assert found["vr"] == pytest.approx(target, abs=0.03)
        assert (side == 0 and abs(found["z"]) < 3) or np.sign(found["z"]) == side


def test_the_variance_ratio_refuses_what_lo_and_mackinlay_do_not_define():
    with pytest.raises(ValueError, match="non-overlapping"):
        qf.variance_ratio(log_prices=WALK, q=2, overlapping=False, bias_correction=True, z="homoskedastic",
                          demean=True)
    with pytest.raises(ValueError, match="non-overlapping"):
        qf.variance_ratio(log_prices=WALK, q=2, overlapping=False, bias_correction=False, z="robust", demean=True)
    with pytest.raises(ValueError, match="q must be"):
        qf.variance_ratio(log_prices=WALK, q=1, overlapping=True, bias_correction=True, z="robust", demean=True)


# ------------------------------------------------------------------------------------------------ Johansen


@pytest.mark.parametrize("case", ["none", "constant", "linear_trend"])
@pytest.mark.parametrize("lags", [1, 2])
def test_johansen_is_statsmodels_coint_johansen(case, lags):
    found = qf.johansen(levels=LEVELS, deterministic=case, lagged_differences=lags)
    trace, largest, eigenvalues = PACKAGES[("johansen", case, lags)]
    assert found["trace"] == pytest.approx(trace, rel=1e-9)
    assert found["max_eigen"] == pytest.approx(largest, rel=1e-9)
    assert found["eigenvalues"] == pytest.approx(eigenvalues, rel=1e-9)
    assert found["trace"][-1] == pytest.approx(found["max_eigen"][-1], rel=1e-12)
    assert np.all(np.diff(found["eigenvalues"]) <= 0) and np.all(found["trace"] >= found["max_eigen"] - 1e-9)


@pytest.mark.parametrize("case, order", [("none", -1), ("constant", 0)])
def test_with_no_lagged_differences_the_eigenvalues_are_the_squared_canonical_correlations(case, order):
    """Johansen (1991): with k = 0 the eigenvalues are the squared canonical correlations of dx_t and x_{t-1} (each
    demeaned under 'constant'). statsmodels 0.15.0 pairs dx_t with x_t at k_ar_diff = 0 and differs here."""
    dx, lagged = np.diff(LEVELS, axis=0), LEVELS[:-1]
    if order == 0:
        dx, lagged = dx - dx.mean(axis=0), lagged - lagged.mean(axis=0)
    n = dx.shape[0]
    s00, s11, s01 = dx.T @ dx / n, lagged.T @ lagged / n, dx.T @ lagged / n
    squared = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(s11, s01.T) @ np.linalg.solve(s00, s01))))[::-1]
    found = qf.johansen(levels=LEVELS, deterministic=case, lagged_differences=0)
    assert found["eigenvalues"] == pytest.approx(squared, rel=1e-10)
    assert found["trace"][0] == pytest.approx(-n * np.sum(np.log(1 - squared)), rel=1e-10)
    v = found["vectors"]
    assert v.T @ s11 @ v == pytest.approx(np.eye(3), abs=1e-10)


def test_johansen_finds_the_shared_trend_and_its_vector():
    found = qf.johansen(levels=LEVELS, deterministic="constant", lagged_differences=1)
    assert found["rank_trace"] == 1 and found["rank_max_eigen"] == 1
    vector = found["vectors"][:, 0]
    assert vector[1] / vector[0] == pytest.approx(-2.0, rel=0.1) and abs(vector[2] / vector[0]) < 0.1
    assert found["trace_critical"][-1] == pytest.approx([2.7055, 3.8415, 6.6349])      # chi-square(1)'s quantiles
    assert qf.johansen(levels=LEVELS, deterministic="none", lagged_differences=1)["trace_critical"][-1] == \
        pytest.approx([2.9762, 4.1296, 6.9406])
    independent = np.cumsum(np.random.RandomState(44).normal(size=(300, 2)), axis=0)
    assert qf.johansen(levels=independent, deterministic="constant", lagged_differences=1)["rank_trace"] == 0
    with pytest.raises(ValueError, match="at most twelve series"):
        qf.johansen(levels=np.cumsum(np.random.RandomState(45).normal(size=(400, 13)), axis=0),
                    deterministic="constant", lagged_differences=1)


def test_the_rank_is_read_at_95_per_cent():
    """Two independent walks whose trace statistic lies between the 90% and the 95% critical values: no
    cointegration at 95% (rank 0), though the 90% value would reject."""
    walks = np.cumsum(np.random.RandomState(411).normal(size=(120, 2)), axis=0)
    found = qf.johansen(levels=walks, deterministic="constant", lagged_differences=1)
    low, high, _ = found["trace_critical"][0]
    assert low < found["trace"][0] < high and found["rank_trace"] == 0


# ------------------------------------------------------------------------------------------------ multiple testing


@pytest.mark.parametrize("method", list(CORRECTIONS))
def test_multiple_testing_is_statsmodels_multipletests(method):
    found = qf.multiple_testing(p_values=P_VALUES, method=method, level=0.05)
    adjusted, reject = PACKAGES[("tests", method)]
    assert found["adjusted"] == pytest.approx(adjusted, rel=1e-12)
    assert found["reject"].tolist() == reject and found["n_rejected"] == sum(reject)


@pytest.mark.parametrize("method", list(CORRECTIONS))
def test_one_test_leaves_its_p_value_unchanged(method):
    found = qf.multiple_testing(p_values=[0.037], method=method, level=0.05)
    assert found["adjusted"] == pytest.approx([0.037], rel=1e-12) and found["reject"].tolist() == [True]


def test_the_corrections_keep_their_textbook_order_and_the_input_order():
    """Hochberg's step-up never exceeds Holm's step-down, which never exceeds Bonferroni's; Benjamini-Yekutieli is
    Benjamini-Hochberg times sum 1/j before the cap; every adjusted p-value keeps the order of the raw ones; and the
    result follows the input's order, not the sorted one."""
    rng = np.random.RandomState(46)
    for _ in range(20):
        p = rng.uniform(0, 0.3, size=15) ** 2
        get = {name: qf.multiple_testing(p_values=p, method=name, level=0.1)["adjusted"] for name in CORRECTIONS}
        assert np.all(get["hochberg"] <= get["holm"] + 1e-15) and np.all(get["holm"] <= get["bonferroni"] + 1e-15)
        assert np.all(get["sidak"] <= get["bonferroni"] + 1e-15)
        harmonic = sum(1 / j for j in range(1, 16))
        assert get["benjamini_yekutieli"] == pytest.approx(np.minimum(1.0, get["benjamini_hochberg"] * harmonic),
                                                           rel=1e-12)
        order = np.argsort(p)
        for name, adjusted in get.items():
            assert np.all(np.diff(adjusted[order]) >= -1e-15), name
    reversed_ = qf.multiple_testing(p_values=P_VALUES[::-1], method="holm", level=0.05)["adjusted"]
    assert reversed_[::-1] == pytest.approx(qf.multiple_testing(p_values=P_VALUES, method="holm",
                                                                level=0.05)["adjusted"])
    with pytest.raises(ValueError, match="inside"):
        qf.multiple_testing(p_values=[0.2, 1.3], method="holm", level=0.05)


if __name__ == "__main__":
    for key, value in reference_values().items():
        print(f"    {key!r}: {value!r},")
