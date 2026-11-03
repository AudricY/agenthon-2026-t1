"""Lane lib-econometrics1 (runs/lanes/lib-econometrics1-prompt.md): qf.hmm_filter and qf.gjr_garch_fit. The filter
against its definition, P(s_t | x_1 .. x_t) summed over every state path (Rabiner 1989, III-A), and against
hmm_smooth on the data to date (the last filtered row is the last smoothed one); the asymmetric GARCH fits against the
arch package 8.0.0's arch_model (vol 'GARCH' with o=1, and 'EGARCH') on this file's own simulated series (frozen in
`ARCH`; `reference_values()` recomputes them where arch imports, the slow test compares): the same likelihood and
backcast, the same optimum to the optimisers' tolerance; GJR at gamma = 0 is garch11_variance's recursion (Glosten,
Jagannathan and Runkle 1993; Nelson 1991). No unit's values appear (rule 8)."""
import itertools
import math

import numpy as np
import pytest
from scipy import special

from test_qf_bs import check_cards  # noqa: F401  (first: it puts agent/ on the path outside pytest too)
import qf
from qf import volatility as V

pytestmark = pytest.mark.fast

HMM = dict(means=[0.0, 1.0, -0.5], stds=[0.3, 0.4, 0.6],
           transition=[[0.8, 0.15, 0.05], [0.1, 0.85, 0.05], [0.2, 0.2, 0.6]], initial=[0.5, 0.3, 0.2])
X = [0.1, -0.2, 0.9, 1.3, 0.4, -0.8, -0.3]


def brute_force_filter(x, means, stds, transition, initial):
    """P(s_t = k | x_1 .. x_t) for each t by summing the joint density over every path of states."""
    a, k = np.asarray(transition), len(means)
    density = lambda value, state: math.exp(-0.5 * ((value - means[state]) / stds[state]) ** 2) / stds[state]  # noqa
    rows = []
    for t in range(1, len(x) + 1):
        weight = np.zeros(k)
        for path in itertools.product(range(k), repeat=t):
            w = initial[path[0]] * density(x[0], path[0])
            for i in range(1, t):
                w *= a[path[i - 1], path[i]] * density(x[i], path[i])
            weight[path[-1]] += w
        rows.append(weight / weight.sum())
    return np.array(rows)


def test_the_filter_is_its_definition_and_the_smoother_on_the_data_to_date():
    found = qf.hmm_filter(x=X, kind="filtered", **HMM)
    assert found["probabilities"] == pytest.approx(brute_force_filter(X, **HMM), rel=1e-11)
    loglik, smoothed, _ = qf.hmm_smooth(X, **HMM)
    assert found["probabilities"][-1] == pytest.approx(smoothed[-1], rel=1e-12)
    assert found["loglik"] == pytest.approx(loglik, rel=1e-13)
    for t in range(1, len(X)):            # hmm_smooth takes two observations or more
        assert found["probabilities"][t] == pytest.approx(qf.hmm_smooth(X[:t + 1], **HMM)[1][-1], rel=1e-11)
    assert not np.allclose(found["probabilities"][:-1], smoothed[:-1])      # the smoother looks ahead
    assert found["probabilities"].sum(axis=1) == pytest.approx(np.ones(len(X)))


def test_the_predicted_probabilities_are_the_last_filtered_moved_one_step():
    filtered = qf.hmm_filter(x=X, kind="filtered", **HMM)
    predicted = qf.hmm_filter(x=X, kind="predicted", **HMM)
    a = np.asarray(HMM["transition"])
    assert predicted["probabilities"][0] == pytest.approx(HMM["initial"])
    assert predicted["probabilities"][1:] == pytest.approx(filtered["probabilities"][:-1] @ a, rel=1e-12)
    assert predicted["next"] == pytest.approx(filtered["probabilities"][-1] @ a, rel=1e-12)
    assert predicted["loglik"] == filtered["loglik"]
    with pytest.raises(ValueError, match="kind"):
        qf.hmm_filter(x=X, kind="smoothed", **HMM)


def test_a_long_series_does_not_underflow():
    rng = np.random.RandomState(61)
    x = np.r_[rng.normal(0, 0.3, 4000), rng.normal(1.0, 0.4, 4000)]
    found = qf.hmm_filter(x=x, kind="filtered", **HMM)["probabilities"]
    assert np.all(np.isfinite(found)) and found[3000].argmax() == 0 and found[7000].argmax() == 1


def test_a_tiny_reachable_state_can_recover_and_an_unreachable_state_does_not_shift_the_density():
    """Rabiner's path sum in log space: an identity transition permits just the two constant-state paths,
    retaining the small path's density even when its displayed posterior rounds to zero."""
    x = np.array([0.0, 100.0, 0.0])
    model = dict(means=[0.0, 100.0], stds=[1.0, 1.0], transition=np.eye(2), initial=[0.5, 0.5])
    found = qf.hmm_filter(x=x, kind="filtered", **model)
    state_log_density = -0.5 * ((x[:, None] - model["means"]) ** 2 + math.log(2 * math.pi))
    paths = math.log(0.5) + np.cumsum(state_log_density, axis=0)
    prefix_density = special.logsumexp(paths, axis=1)
    probabilities = np.exp(paths - prefix_density[:, None])
    assert found["probabilities"] == pytest.approx(probabilities, abs=1e-12)
    assert found["probabilities"][1] == pytest.approx([0.5, 0.5], abs=1e-12)
    assert found["loglik"] == pytest.approx(prefix_density[-1], abs=1e-10)
    loglik, smoothed, transitions = qf.hmm_smooth(x=x, **model)
    assert loglik == pytest.approx(prefix_density[-1], abs=1e-10)
    assert smoothed == pytest.approx(np.tile(probabilities[-1], (x.size, 1)), abs=1e-12)
    assert transitions == pytest.approx(np.diag((x.size - 1) * probabilities[-1]), abs=1e-12)
    fixed = qf.hmm_filter(x=[100.0, 100.0], kind="filtered", **(model | {"initial": [1.0, 0.0]}))
    assert fixed["loglik"] == pytest.approx(-10000 - math.log(2 * math.pi), abs=1e-10)
    assert fixed["probabilities"] == pytest.approx(np.array([[1.0, 0.0], [1.0, 0.0]]))


# ------------------------------------------------------------------------------------------------ GJR and EGARCH


def gjr_series(n, seed, student):
    """A GJR-GARCH(1,1) path, omega 2e-6, alpha 0.03, gamma 0.10, beta 0.88, mu 3e-4; standardised t(6) shocks where
    `student`; the legacy generator (a frozen stream)."""
    rng = np.random.RandomState(seed)
    r, v = np.empty(n), 2e-6 / 0.04
    for t in range(n):
        z = rng.standard_t(6) * math.sqrt(4 / 6) if student else rng.normal()
        e = math.sqrt(v) * z
        r[t] = 3e-4 + e
        v = 2e-6 + (0.03 + 0.10 * (e < 0)) * e * e + 0.88 * v
    return r


def egarch_series(n, seed):
    """An EGARCH(1,1) path, omega -0.2, alpha 0.12, gamma -0.08, beta 0.98, normal shocks."""
    rng = np.random.RandomState(seed)
    r, lv = np.empty(n), -0.2 / 0.02
    for t in range(n):
        z = rng.normal()
        r[t] = math.exp(lv / 2) * z
        lv = -0.2 + 0.12 * (abs(z) - math.sqrt(2 / math.pi)) - 0.08 * z + 0.98 * lv
    return r


SERIES = {("gjr", "normal"): gjr_series(1000, 62, False), ("gjr", "t"): gjr_series(1000, 63, True),
          ("egarch", "normal"): egarch_series(1000, 64)}
CASES = [("gjr", "normal", "constant"), ("gjr", "normal", "zero"), ("gjr", "t", "constant"),
         ("egarch", "normal", "zero")]
ARCH = {   # arch 8.0.0 on SERIES (reference_values), in the returns' units
    ('gjr', 'normal', 'constant'): {'mu': 0.00033855278235664317, 'omega': 1.5543282647395791e-06, 'alpha': 0.05914871190453905, 'gamma': 0.09881635306085422, 'beta': 0.8617933044465433, 'nu': math.nan, 'loglik': 3615.023266846495, 'last_vol': 0.0074765869139499655},
    ('gjr', 'normal', 'zero'): {'mu': 0.0, 'omega': 1.5550115700419393e-06, 'alpha': 0.05160384408674679, 'gamma': 0.11688512009716605, 'beta': 0.8653191208146463, 'nu': math.nan, 'loglik': 3613.4750217084506, 'last_vol': 0.0074782776322912855},
    ('gjr', 't', 'constant'): {'mu': 0.0003062264210955462, 'omega': 3.1649550721650407e-07, 'alpha': 2.2505615608562733e-18, 'gamma': 0.0643213721557368, 'beta': 0.9624997485436891, 'nu': 7.834854285202401, 'loglik': 3735.2680194789996, 'last_vol': 0.006084941426660343},
    ('egarch', 'normal', 'zero'): {'mu': 0.0, 'omega': -0.1944345486453129, 'alpha': 0.10735876648704541, 'gamma': -0.08524065140570658, 'beta': 0.9805603425869948, 'nu': math.nan, 'loglik': 3403.277229653443, 'last_vol': 0.007057392324928022},
}


def reference_values():
    """ARCH again, from arch's fits at scale 100 (the lane's scratch/lib-econometrics1/refenv), in the returns' units."""
    from arch import arch_model
    found = {}
    for model, dist, mean in CASES:
        r = SERIES[(model, dist)]
        fit = arch_model(r * 100, mean="Zero" if mean == "zero" else "Constant",
                         vol="GARCH" if model == "gjr" else "EGARCH", p=1, o=1, q=1,
                         dist="normal" if dist == "normal" else "studentst", rescale=False).fit(disp="off")
        p = fit.params
        omega = p["omega"] / 1e4 if model == "gjr" else p["omega"] - 2 * (1 - p["beta[1]"]) * math.log(100)
        found[(model, dist, mean)] = {"mu": p.get("mu", 0.0) / 100, "omega": omega, "alpha": p["alpha[1]"],
                                      "gamma": p["gamma[1]"], "beta": p["beta[1]"],
                                      "nu": p.get("nu", math.nan), "loglik": fit.loglikelihood + r.size * math.log(100),
                                      "last_vol": fit.conditional_volatility[-1] / 100}
    return found


@pytest.mark.slow
def test_the_frozen_values_are_archs_own():
    pytest.importorskip("arch")
    found = reference_values()
    for key, value in ARCH.items():
        model, dist, _ = key
        scale = float(np.std(SERIES[(model, dist)]))
        # BLAS summation order changes SLSQP's final iterates: a single thread moves nu by 0.000128 while
        # loglik moves by less than 1e-8. Preserve the tight objective check and compare fitted parameters in
        # their natural units. The fixed-parameter density/recursion test below stays tighter.
        tolerances = {"mu": 1e-6 * scale, "omega": 1e-6 * scale**2 if model == "gjr" else 1e-6,
                      "alpha": 1e-6, "gamma": 1e-6, "beta": 1e-6, "nu": 2e-4,
                      "last_vol": 1e-6 * scale, "loglik": 1e-7}
        for part, number in value.items():
            assert found[key][part] == pytest.approx(number, rel=0, abs=tolerances[part], nan_ok=True), (key, part)


@pytest.mark.parametrize("model, dist, mean", [case for case in CASES if case[0] == "gjr"])
def test_the_gjr_fit_is_archs(model, dist, mean):
    found = qf.gjr_garch_fit(returns=SERIES[(model, dist)], model=model, dist=dist, mean=mean,
                             start_variance="arch")
    want = ARCH[(model, dist, mean)]
    assert found["converged"] and found["loglik"] >= want["loglik"] - 1e-6
    assert found["loglik"] == pytest.approx(want["loglik"], abs=1e-3)
    for part in ("alpha", "gamma", "beta"):
        assert found[part] == pytest.approx(want[part], abs=3e-3), part
    assert found["omega"] == pytest.approx(want["omega"], rel=0.05)
    assert found["mu"] == pytest.approx(want["mu"], abs=2e-5)
    if dist == "t":
        assert found["nu"] == pytest.approx(want["nu"], rel=0.02)
    assert found["conditional_vol"][-1] == pytest.approx(want["last_vol"], rel=1e-3)
    assert found["gamma"] > 0 and found["persistence"] == pytest.approx(
        found["alpha"] + found["gamma"] / 2 + found["beta"])


def test_the_egarch_fit_is_archs():
    model, dist, mean = CASES[-1]
    found = qf.gjr_garch_fit(returns=SERIES[(model, dist)], model=model, dist=dist, mean=mean, start_variance="arch")
    want = ARCH[(model, dist, mean)]
    assert found["converged"] and found["loglik"] >= want["loglik"] - 1e-6
    for part in ("omega", "alpha", "gamma", "beta"):
        assert found[part] == pytest.approx(want[part], abs=3e-3), part
    assert found["gamma"] < 0 and found["persistence"] == found["beta"]
    assert math.isnan(found["long_run_variance"])
    geometric = math.exp(found["omega"] / (1 - found["beta"]))
    assert found["geometric_variance"] == pytest.approx(geometric, rel=1e-12)
    # Nelson's log-variance shocks have zero normal mean but positive variance; their exponential mean exceeds 1.
    alpha, gamma = found["alpha"], found["gamma"]
    moment = math.exp(-alpha * math.sqrt(2 / math.pi)) * (
        math.exp((alpha + gamma) ** 2 / 2) * special.ndtr(alpha + gamma) +
        math.exp((alpha - gamma) ** 2 / 2) * special.ndtr(alpha - gamma))
    assert geometric * moment > geometric


def test_the_egarch_likelihood_and_recursion_are_archs_at_archs_parameters():
    """The likelihood a fit maximises, at arch's optimum, is arch's log-likelihood (the backcast arch's own: the
    0.94-weighted mean of the first 75 squared residuals at the sample mean, its log for EGARCH)."""
    for (model, dist, mean), want in ARCH.items():
        r = SERIES[(model, dist)]
        residual = r - (r.mean() if mean == "constant" else 0.0)
        weights = 0.94 ** np.arange(75)
        backcast = float(np.sum(weights * residual[:75] ** 2) / weights.sum())
        e = r - want["mu"]
        variance = V._asymmetric_variance(e, model, want["omega"], want["alpha"], want["gamma"], want["beta"],
                                          backcast)
        loglik = V._asymmetric_loglik(e, variance, None if dist == "normal" else want["nu"])
        assert loglik == pytest.approx(want["loglik"], rel=1e-10), (model, dist, mean)
        assert math.sqrt(variance[-1]) == pytest.approx(want["last_vol"], rel=1e-9)


def test_gjr_at_gamma_zero_is_the_garch11_recursion_and_bad_news_raises_the_variance_more():
    r = SERIES[("gjr", "normal")]
    gjr = V._asymmetric_variance(r, "gjr", 2e-6, 0.05, 0.0, 0.9, 1e-4)
    garch, _ = qf.garch11_variance(r, mu=0.0, omega=2e-6, alpha=0.05, beta=0.9, start_variance=1e-4,
                                   start="backcast")
    assert gjr == pytest.approx(garch, rel=1e-12)
    up = V._asymmetric_variance(np.array([0.02, 0.0]), "gjr", 2e-6, 0.05, 0.1, 0.9, 1e-4)
    down = V._asymmetric_variance(np.array([-0.02, 0.0]), "gjr", 2e-6, 0.05, 0.1, 0.9, 1e-4)
    assert down[1] - up[1] == pytest.approx(0.1 * 0.02 ** 2, rel=1e-12)
    found = qf.gjr_garch_fit(returns=r, model="gjr", dist="normal", mean="zero", start_variance="sample")
    e = r
    following = found["omega"] + (found["alpha"] + found["gamma"] * (e[-1] < 0)) * e[-1] ** 2 + found["beta"] * \
        found["conditional_vol"][-1] ** 2
    assert found["next_vol"] == pytest.approx(math.sqrt(following), rel=1e-10)
    assert found["backcast"] == pytest.approx(np.mean(r ** 2), rel=1e-12)
    with pytest.raises(ValueError, match="model"):
        qf.gjr_garch_fit(returns=r, model="tgarch", dist="normal", mean="zero", start_variance="arch")


if __name__ == "__main__":
    for key, value in reference_values().items():
        print(f"    {key!r}: {value!r},")
