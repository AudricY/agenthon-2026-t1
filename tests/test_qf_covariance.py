"""Lane lib-risk1 (runs/lanes/lib-risk1-prompt.md), the `covariance` shelf: Ledoit-Wolf's scaled-identity shrinkage
against scikit-learn 1.9.1's LedoitWolf (frozen on a small panel, and live where scikit-learn imports), the
constant-correlation target against the "Honey, I shrunk" formulas written out term by term; the nearest
correlation matrix against Higham's (2002) worked example (section 4: the tridiagonal 2, -1 matrix of order 4).
Synthetic panels alone; no unit's."""
import math

import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast

PANEL = np.array([[0.010, 0.020, -0.010], [-0.020, 0.010, 0.000], [0.015, -0.005, 0.010], [0.000, 0.012, -0.004],
                  [-0.010, -0.020, 0.020], [0.007, 0.003, -0.012], [-0.004, 0.009, 0.006], [0.012, -0.011, 0.001]])
SKLEARN = {   # sklearn.covariance.LedoitWolf(assume_centered=...).fit(PANEL), scikit-learn 1.9.1
    False: (0.9330701853037917, [[0.0001268467648851169, -3.722970942476585e-07, -2.173127420917513e-06],
                                 [-3.722970942476585e-07, 0.00012867060233558855, -5.988126858601385e-06],
                                 [-2.173127420917513e-06, -5.988126858601385e-06, 0.00012484200777929452]]),
    True: (0.9591688518959136, [[0.00012960968831946098, -1.1228565728623753e-07, -1.2555578042006565e-06],
                                [-1.1228565728623753e-07, 0.00013086524612366163, -3.5267904174904623e-06],
                                [-1.2555578042006565e-06, -3.5267904174904623e-06, 0.00012840006555687742]])}


@pytest.mark.parametrize("centered", [False, True])
def test_the_scaled_identity_target_is_scikit_learns_ledoit_wolf(centered):
    found = qf.ledoit_wolf(returns=PANEL, target="scaled_identity", demean=not centered, ddof=0)
    shrinkage, covariance = SKLEARN[centered]
    assert found["shrinkage"] == pytest.approx(shrinkage, rel=1e-12)
    assert found["covariance"] == pytest.approx(np.array(covariance), rel=1e-11)


@pytest.mark.slow     # scikit-learn's import alone takes a quarter second
def test_the_scaled_identity_target_agrees_with_scikit_learn_live_on_a_larger_panel():
    sklearn = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 8)) @ rng.standard_normal((8, 8)) * 0.01
    found = qf.ledoit_wolf(returns=x, target="scaled_identity", demean=True, ddof=0)
    fit = sklearn.LedoitWolf().fit(x)
    assert found["shrinkage"] == pytest.approx(fit.shrinkage_, rel=1e-10)
    assert found["covariance"] == pytest.approx(fit.covariance_, rel=1e-10, abs=1e-18)


def honey(y, ddof):
    """Ledoit and Wolf (2004), "Honey, I shrunk the sample covariance matrix", appendix B, term by term with loops
    (demeaned panel, n = T - ddof). At ddof 0 pi and theta take the paper's defining forms, mean (y_i y_j - s_ij)^2
    and mean (y_i^2 - s_ii)(y_i y_j - s_ij); at ddof 1 the shortcut forms of Ledoit and Wolf's 2020 code (covCor),
    sum y_i^2 y_j^2 / n - s_ij^2 and sum y_i^3 y_j / n - s_ii s_ij, which differ from the defining ones at n = T - 1."""
    y = y - y.mean(axis=0)
    t, p = y.shape
    n = t - ddof
    s = [[sum(y[k, i] * y[k, j] for k in range(t)) / n for j in range(p)] for i in range(p)]
    r = [[s[i][j] / math.sqrt(s[i][i] * s[j][j]) for j in range(p)] for i in range(p)]
    rbar = sum(r[i][j] for i in range(p) for j in range(p) if i != j) / (p * (p - 1))
    f = [[s[i][i] if i == j else rbar * math.sqrt(s[i][i] * s[j][j]) for j in range(p)] for i in range(p)]
    if ddof == 0:
        pi = [[sum((y[k, i] * y[k, j] - s[i][j]) ** 2 for k in range(t)) / n for j in range(p)] for i in range(p)]
        theta_ii = [[sum((y[k, i] ** 2 - s[i][i]) * (y[k, i] * y[k, j] - s[i][j]) for k in range(t)) / n
                     for j in range(p)] for i in range(p)]
    else:
        pi = [[sum(y[k, i] ** 2 * y[k, j] ** 2 for k in range(t)) / n - s[i][j] ** 2 for j in range(p)]
              for i in range(p)]
        theta_ii = [[sum(y[k, i] ** 3 * y[k, j] for k in range(t)) / n - s[i][i] * s[i][j] for j in range(p)]
                    for i in range(p)]
    rho = sum(pi[i][i] for i in range(p)) + sum(
        rbar / 2 * (math.sqrt(s[j][j] / s[i][i]) * theta_ii[i][j] + math.sqrt(s[i][i] / s[j][j]) * theta_ii[j][i])
        for i in range(p) for j in range(p) if i != j)
    gamma = sum((f[i][j] - s[i][j]) ** 2 for i in range(p) for j in range(p))
    kappa = (sum(map(sum, pi)) - rho) / gamma
    delta = max(0.0, min(1.0, kappa / n))
    return delta, np.array([[delta * f[i][j] + (1 - delta) * s[i][j] for j in range(p)] for i in range(p)])


@pytest.mark.parametrize("ddof", [0, 1])
def test_the_constant_correlation_target_is_the_honey_papers_estimator(ddof):
    """theta_ii,ij = mean of (y_i^2 - s_ii)(y_i y_j - s_ij), which is mean of y_i^3 y_j - s_ii s_ij on a demeaned panel
    (the lane's form); ddof 1 is the same estimator at n = T - 1 (Ledoit and Wolf's 2020 code)."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((30, 4)) @ np.array([[1, 0.5, 0.2, 0], [0, 1, 0.3, 0.1], [0, 0, 1, 0.4], [0, 0, 0, 1]])
    delta, sigma = honey(x * 0.01, ddof)
    found = qf.ledoit_wolf(returns=x * 0.01, target="constant_correlation", demean=True, ddof=ddof)
    assert found["shrinkage"] == pytest.approx(delta, rel=1e-10) and 0 <= delta <= 1
    assert found["covariance"] == pytest.approx(sigma, rel=1e-10)
    assert np.diag(found["target"]) == pytest.approx(np.diag(found["sample"]), rel=1e-15)


def test_the_intensity_falls_as_the_sample_grows_and_lies_in_the_unit_interval():
    rng = np.random.default_rng(6)
    mix = np.array([[1.0, 0.8, 0.0, 0.0, 0.3], [0.0, 1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 2.0, 0.7, 0.0],
                    [0.0, 0.0, 0.0, 1.0, -0.6], [0.0, 0.0, 0.0, 0.0, 0.5]])
    values = [qf.ledoit_wolf(returns=rng.standard_normal((t, 5)) @ mix * 0.01, target=target, demean=True,
                             ddof=0)["shrinkage"] for target in ("scaled_identity", "constant_correlation")
              for t in (20, 2000)]
    assert all(0 <= v <= 1 for v in values) and values[1] < values[0] and values[3] < values[2]


def test_higham_s_example_nearest_correlation_matrix():
    """Higham (2002), IMA Journal of Numerical Analysis 22(3), section 4: the nearest correlation matrix to
    tridiag(-1, 2, -1) of order 4 is [1 -0.8084 0.1916 0.1068; -0.8084 1 -0.6562 0.1916; ...] (to the four decimals
    printed there; the eight here are the converged iteration's, as Higham's algorithm 3.3 reproduces them)."""
    a = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]], dtype=float)
    found = qf.nearest_correlation(matrix=a, method="higham", floor=0.0, weights=None, max_iterations=1000, tol=1e-12)
    expected = np.array([[1.0, -0.8084125, 0.1915875, 0.10677505], [-0.8084125, 1.0, -0.65623269, 0.1915875],
                         [0.1915875, -0.65623269, 1.0, -0.8084125], [0.10677505, 0.1915875, -0.8084125, 1.0]])
    assert found["correlation"] == pytest.approx(expected, abs=1e-8)
    assert found["min_eigenvalue"] > -1e-10 and found["distance"] == pytest.approx(np.linalg.norm(expected - a), abs=1e-7)


def test_each_method_returns_a_correlation_matrix_and_higham_is_nearer_than_clipping():
    a = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]])
    assert np.linalg.eigvalsh(a).min() < 0
    clip = qf.nearest_correlation(matrix=a, method="clip", floor=0.0, weights=None, max_iterations=None, tol=None)
    near = qf.nearest_correlation(matrix=a, method="higham", floor=0.0, weights=None, max_iterations=200, tol=1e-12)
    for found in (clip, near):
        c = found["correlation"]
        assert np.allclose(c, c.T) and np.all(np.diag(c) == 1.0) and found["min_eigenvalue"] > -1e-10
    assert near["distance"] < clip["distance"] and clip["iterations"] == 0
    floored = qf.nearest_correlation(matrix=a, method="higham", floor=1e-3, weights=None, max_iterations=500,
                                     tol=1e-12)
    assert floored["min_eigenvalue"] > 5e-4
    ok = np.array([[1.0, 0.3], [0.3, 1.0]])
    assert qf.nearest_correlation(matrix=ok, method="higham", floor=0.0, weights=None, max_iterations=50,
                                  tol=1e-12)["correlation"] == pytest.approx(ok, abs=1e-14)


def test_weights_keep_the_heavily_weighted_rows_closer():
    a = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]])
    plain = qf.nearest_correlation(matrix=a, method="higham", floor=0.0, weights=None, max_iterations=500, tol=1e-13)
    heavy = qf.nearest_correlation(matrix=a, method="higham", floor=0.0, weights=[100.0, 100.0, 1.0],
                                   max_iterations=2000, tol=1e-13)
    assert abs(heavy["correlation"][0, 1] - 0.9) < abs(plain["correlation"][0, 1] - 0.9)
    with pytest.raises(ValueError):
        qf.nearest_correlation(matrix=a, method="clip", floor=0.0, weights=None, max_iterations=100, tol=None)


def test_higham_floor_is_in_the_original_matrix_and_is_independent_of_weight_scale():
    """Higham (2002), theorem 3.2: weighted PSD projection is a congruence. For a floor f, project A - f I
    onto that cone and add f I, so the constraint is X >= f I rather than W^1/2 X W^1/2 >= f I."""
    a = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]])
    plain = qf.nearest_correlation(matrix=a, method="higham", floor=0.1, weights=None, max_iterations=2000,
                                  tol=1e-13)
    for weights in ([0.1, 0.1, 0.1], [100.0, 100.0, 100.0]):
        weighted = qf.nearest_correlation(matrix=a, method="higham", floor=0.1, weights=weights,
                                         max_iterations=2000, tol=1e-13)
        assert weighted["correlation"] == pytest.approx(plain["correlation"], abs=1e-12)
    heavy = qf.nearest_correlation(matrix=a, method="higham", floor=0.1, weights=[100.0, 100.0, 1.0],
                                  max_iterations=2000, tol=1e-13)
    assert heavy["min_eigenvalue"] >= 0.1 - 1e-11
    assert abs(heavy["correlation"][0, 1] - 0.9) < abs(plain["correlation"][0, 1] - 0.9)


def test_higham_refuses_an_unconverged_iterate_instead_of_returning_an_indefinite_matrix():
    a = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]])
    with pytest.raises(ValueError, match="has not converged"):
        qf.nearest_correlation(matrix=a, method="higham", floor=0.0, weights=None, max_iterations=1, tol=1e-12)
    with pytest.raises(ValueError, match="finite and positive"):
        qf.nearest_correlation(matrix=a, method="higham", floor=0.0, weights=None, max_iterations=50, tol=math.inf)
