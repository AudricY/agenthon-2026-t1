"""Every sampler's `rng` (lane libfix1; docs/research/conventions-1012.md, the CreditMetrics read): a
numpy.random.Generator or an int seed as before, byte for byte, and numpy's legacy stream too, a numpy.random.RandomState
or None (the global stream numpy.random.seed(seed) sets), through one helper (qf.primitives._stream).

- The pins: each sampler's output under a Generator seeded 7, digested (`digest`) on the library before the change
  (lane libfix1's base, 46f59335), equals its output now; an int seed is the Generator it seeds.
- The legacy stream: under a RandomState seeded 7 each sampler draws what its card names, in that order, so the stream
  is left where the same draws taken directly from a twin RandomState leave it; where the output is a plain transform
  of the draws (the copulas, CreditMetrics, the bootstrap, GBM, the Hawkes thinning), the output is computed by hand
  from the twin's draws. None draws from numpy's global stream: after numpy.random.seed(7) it gives what a
  RandomState(7) gives.
Every input is this file's own; no unit's values appear (rule 8)."""
import ast
import hashlib
import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

import qf
from conftest import ROOT

pytestmark = pytest.mark.fast

CORR = [[1.0, 0.4], [0.4, 1.0]]
TRANSITION = [[0.85, 0.10, 0.05], [0.10, 0.80, 0.10]]
VALUES = [[103.0, 100.0, 45.0], [103.0, 100.0, 45.0], [104.0, 99.0, 50.0]]
MERTON_BOX = {"vol": (0.1, 0.5), "intensity": (0.0, 10.0), "jump_mean": (-0.1, 0.1), "jump_vol": (0.02, 0.2),
              "drift": (-0.5, 0.5)}
CIR_BOX = {"kappa": (0.1, 2.0), "theta": (0.01, 0.1), "sigma": (0.01, 0.3)}
SERIES = np.array([0.021, 0.0213, 0.0209, 0.0216, 0.0219, 0.0214, 0.0212, 0.0218])
RETURNS = np.array([0.004, -0.012, 0.008, 0.001, -0.003, 0.015, -0.007, 0.002, 0.006, -0.010, 0.003, 0.009])


def _cir_exact(r, paths, x0, kappa, theta, sigma, times):
    now, last, dof = np.full(paths, x0), 0.0, 4.0 * kappa * theta / sigma ** 2
    found = []
    for t in times:
        h = t - last
        c = sigma ** 2 * -math.expm1(-kappa * h) / (4.0 * kappa)
        now = c * r.noncentral_chisquare(dof, now * math.exp(-kappa * h) / c)
        found.append(now)
        last = t
    return np.column_stack(found)


def _hawkes(r, mu, alpha, beta, start, end):
    now, excite, found = start, 0.0, []
    while True:
        bound = mu + alpha * excite
        gap = r.exponential(1.0 / bound)
        now += gap
        if now > end:
            return np.array(found)
        excite *= math.exp(-beta * gap)
        if r.random() * bound <= mu + alpha * excite:
            found.append(now)
            excite += 1.0


def _ou_draws(r):
    r.standard_normal((4, 5))
    counts = r.poisson(2.0 * 0.1, (4, 5))
    total = int(counts.sum())
    r.normal(0.05, 0.1, total)
    r.random(total)


def _stratified_lhs(r):
    r.random((8, 2))
    for _ in range(2):
        r.permutation(8)


def _sobol(r):
    for _ in range(2):
        qmc.Sobol(2, scramble=True, seed=r).random_base2(3)


def _gumbel(r):
    r.uniform(0.0, np.pi, 6)
    r.standard_exponential(6)
    r.standard_exponential((6, 2))


# name: (the call under `rng`, the draws its card names taken from a twin stream). A sampler with several branches
# has a row a branch, `<function>/<branch>`.
ROWS = {
    "gbm_paths": (lambda rng: qf.gbm_paths(spot=50.0, drift=0.04, vol=0.3, times=[0.25, 1.0], n_paths=6, rng=rng,
                                           drift_form="arithmetic", antithetic=True),
                  lambda r: r.standard_normal((3, 2))),
    "correlated_gbm_paths": (lambda rng: qf.correlated_gbm_paths(spots=[50.0, 20.0], drifts=[0.04, 0.01],
                                                                 vols=[0.3, 0.2], correlation=CORR, times=[0.5, 1.0],
                                                                 n_paths=5, rng=rng, drift_form="log",
                                                                 antithetic=False),
                             lambda r: r.standard_normal((5, 2, 2))),
    "local_vol_paths": (lambda rng: qf.local_vol_paths(spot=50.0, local_vol=lambda t, s: 0.25 + 0.0 * s,
                                                       times=[0.5, 1.0], substeps=3, rate=0.02, dividend_yield=0.0,
                                                       n_paths=4, rng=rng, scheme="log_euler", antithetic=True),
                        lambda r: r.standard_normal((2, 6))),
    "ou_paths": (lambda rng: qf.ou_paths(x0=0.1, kappa=1.5, theta=0.3, sigma=0.2, dt=0.1, n_steps=5, n_paths=4,
                                         rng=rng, scheme="exact", jump_rate=2.0, jump_mean=0.05, jump_std=0.1,
                                         jump_count="poisson"),
                 _ou_draws),
    "cir_paths": (lambda rng: qf.cir_paths(x0=0.02, kappa=0.8, theta=0.03, sigma=0.1, times=[0.25, 0.75], n_paths=4,
                                           rng=rng, scheme="exact"),
                  lambda r: _cir_exact(r, 4, 0.02, 0.8, 0.03, 0.1, [0.25, 0.75])),
    "heston_paths": (lambda rng: qf.heston_paths(spot=50.0, v0=0.05, kappa=1.2, theta=0.06, sigma=0.4, rho=-0.5,
                                                 rate=0.02, dividend_yield=0.0, times=[0.5, 1.0], substeps=2,
                                                 n_paths=3, rng=rng, scheme="qe", psi_c=1.5),
                     lambda r: (r.random((3, 4)), r.standard_normal((3, 4)))),
    "sde_paths": (lambda rng: qf.sde_paths(x0=1.0, drift=lambda t, x: 0.1 * x, diffusion=lambda t, x: 0.3 * x,
                                           diffusion_derivative=lambda t, x: 0.3 + 0.0 * x, times=[0.5, 1.0],
                                           substeps=2, n_paths=3, rng=rng, scheme="milstein"),
                  lambda r: r.standard_normal((3, 4))),
    "lmm_paths": (lambda rng: qf.lmm_paths(forwards=[0.02, 0.025, 0.03], accruals=[0.5, 0.5, 0.5],
                                           vols=[0.15, 0.15, 0.15],
                                           correlation=[[1.0, 0.8, 0.6], [0.8, 1.0, 0.8], [0.6, 0.8, 1.0]],
                                           substeps=2, n_paths=3, rng=rng, measure="spot", predictor_corrector=True),
                  lambda r: r.standard_normal((3, 4, 3))),
    "mc_payoff_price": (lambda rng: qf.mc_payoff_price(
        paths=lambda n, g: qf.gbm_paths(50.0, 0.02, 0.3, [0.5, 1.0], n, g, "arithmetic", False),
        payoff=lambda s: (s[:, -1] - 50.0).clip(0.0), n_paths=6, rng=rng, antithetic=False, control=None,
        control_mean=None, discount_factor=0.98, ddof=1, confidence=0.9),
                        lambda r: r.standard_normal((6, 2))),
    "importance_shift": (lambda rng: qf.importance_shift(shift=1.0, n_draws=5, dimension=2, rng=rng),
                         lambda r: r.standard_normal((5, 2))),
    "stratified_normals/stratified": (lambda rng: qf.stratified_normals(
        n_draws=8, strata=4, dimension=2, kind="stratified", allocation="proportional", stratum_stds=None,
        integrand=lambda z: z[:, 0] ** 2, rng=rng, ddof=1),
                                      lambda r: (r.random((8, 1)), r.standard_normal((8, 1)))),
    "stratified_normals/latin_hypercube": (lambda rng: qf.stratified_normals(
        n_draws=8, strata=None, dimension=2, kind="latin_hypercube", allocation=None, stratum_stds=None,
        integrand=None, rng=rng, ddof=1),
                                           _stratified_lhs),
    "sobol_normals": (lambda rng: qf.sobol_normals(n_points=8, dimension=2, scramble=True, order="sequential",
                                                   replications=2, integrand=lambda z: z[:, 0] ** 2, rng=rng, ddof=1,
                                                   confidence=0.9),
                      _sobol),
    "american_bounds": (lambda rng: qf.american_bounds(method="random_tree", payoff=lambda s: (50.0 - s).clip(0.0),
                                                       spot=50.0, rate=0.04, dividend_yield=0.0, vol=0.3,
                                                       exercise_times=[0.5, 1.0], n_paths=3, branching=2,
                                                       inner_paths=None, basis=None, degree=None, rng=rng,
                                                       confidence=0.9),
                        lambda r: (r.standard_normal((3, 2)), r.standard_normal((3, 4)))),
    "gaussian_copula_sample": (lambda rng: qf.gaussian_copula_sample(CORR, n=5, rng=rng),
                               lambda r: r.standard_normal((5, 2))),
    "t_copula_sample": (lambda rng: qf.t_copula_sample(CORR, nu=4.0, n=5, rng=rng),
                        lambda r: (r.standard_normal((5, 2)), r.chisquare(4.0, 5))),
    "archimedean_copula_sample/clayton": (lambda rng: qf.archimedean_copula_sample("clayton", theta=2.0, dim=2, n=6,
                                                                                   rng=rng),
                                          lambda r: (r.gamma(0.5, 1.0, 6), r.standard_exponential((6, 2)))),
    "archimedean_copula_sample/gumbel": (lambda rng: qf.archimedean_copula_sample("gumbel", theta=2.0, dim=2, n=6,
                                                                                  rng=rng),
                                         _gumbel),
    "archimedean_copula_sample/frank": (lambda rng: qf.archimedean_copula_sample("frank", theta=3.0, dim=2, n=6,
                                                                                 rng=rng),
                                        lambda r: (r.logseries(-math.expm1(-3.0), 6), r.standard_exponential((6, 2)))),
    "conditional_copula_sample": (lambda rng: qf.conditional_copula_sample(u=[0.2, 0.5, 0.7], family="clayton",
                                                                           parameter=1.5, nu=None,
                                                                           inverse="closed_form", w=None, rng=rng),
                                  lambda r: r.uniform(size=(3,))),
    "factor_copula_losses/gaussian": (lambda rng: qf.factor_copula_losses(
        pd=[0.05, 0.1, 0.2], lgd=[0.5, 0.5, 0.4], exposure=[10.0, 20.0, 5.0], loading=[0.4, 0.5, 0.3],
        factor_corr=CORR, factor_index=[0, 0, 1], copula="gaussian", nu=None, n=6, rng=rng),
                                      lambda r: (r.standard_normal((6, 2)), r.standard_normal((6, 3)))),
    "factor_copula_losses/t": (lambda rng: qf.factor_copula_losses(
        pd=[0.05, 0.1, 0.2], lgd=[0.5, 0.5, 0.4], exposure=[10.0, 20.0, 5.0], loading=[0.4, 0.5, 0.3],
        factor_corr=CORR, factor_index=[0, 0, 1], copula="t", nu=6.0, n=6, rng=rng),
                               lambda r: (r.standard_normal((6, 2)), r.standard_normal((6, 3)), r.chisquare(6.0, 6))),
    "creditmetrics": (lambda rng: qf.creditmetrics(transition=TRANSITION, initial=[0, 1, 1], values=VALUES, rho=0.3,
                                                   n=7, rng=rng),
                      lambda r: (r.standard_normal(7), r.standard_normal((7, 3)))),
    "delta_gamma_var": (lambda rng: qf.delta_gamma_var(delta=[40.0, -10.0], gamma=[[-5.0, 1.0], [1.0, 2.0]],
                                                       covariance=[[4e-4, 1e-4], [1e-4, 9e-4]], horizon=1, alpha=0.9,
                                                       method="simulation", n=50, rng=rng, sign="loss_positive"),
                        lambda r: r.standard_normal((50, 2))),
    "copula_loss_simulation": (lambda rng: qf.copula_loss_simulation(
        copula="gaussian", corr=CORR, nu=None, marginals=[("normal", 0.0, 0.01), ("normal", 0.0, 0.02)],
        sigma_kind="std", weights=[0.5, 0.5], n=40, rng=rng, alpha=0.9, sign="loss_positive"),
                               lambda r: r.standard_normal((40, 2))),
    "block_bootstrap/iid": (lambda rng: qf.block_bootstrap(n_obs=9, kind="iid", block_length=None, n_draws=3, rng=rng,
                                                           statistic=None),
                            lambda r: r.randint(0, 9, (3, 9))),
    "block_bootstrap/stationary": (lambda rng: qf.block_bootstrap(n_obs=9, kind="stationary", block_length=3.0,
                                                                  n_draws=3, rng=rng, statistic=None),
                                   lambda r: (r.random((3, 9)), r.randint(0, 9, (3, 9)))),
    "block_bootstrap/moving": (lambda rng: qf.block_bootstrap(n_obs=9, kind="moving", block_length=3.0, n_draws=3,
                                                              rng=rng, statistic=None),
                               lambda r: r.randint(0, 7, (3, 3))),
    "merton_jump_fit": (lambda rng: qf.merton_jump_fit(returns=RETURNS, dt=1 / 252, terms=3, drift="fit",
                                                       bounds=MERTON_BOX, starts=1, rng=rng),
                        lambda r: [r.uniform(low, high) for low, high in MERTON_BOX.values()]),
    "hawkes_simulate": (lambda rng: qf.hawkes_simulate(mu=0.8, alpha=0.5, beta=1.5, start=0.0, end=6.0, rng=rng),
                        lambda r: _hawkes(r, 0.8, 0.5, 1.5, 0.0, 6.0)),
    "cir_fit": (lambda rng: qf.cir_fit(rates=SERIES, dt=1 / 252, method="mle_exact", bounds=CIR_BOX, starts=1,
                                       rng=rng),
                lambda r: [r.uniform(low, high) for low, high in CIR_BOX.values()]),
}

# The digests under numpy.random.default_rng(7), taken on the library before the stream helper (lane libfix1's base):
# the outputs there and here are byte for byte the same.
PINS = {
    "american_bounds": "a1b0153370fe86582bf4bfe1",
    "archimedean_copula_sample/clayton": "f8249e774975199ff76dedde",
    "archimedean_copula_sample/frank": "554826cc3b0026e29dee0dc3",
    "archimedean_copula_sample/gumbel": "1d6010b99e31ff8129b21d49",
    "block_bootstrap/iid": "38cee4d2597e9d696456613f",
    "block_bootstrap/moving": "3f35b3510c6e7b3fe105aae1",
    "block_bootstrap/stationary": "38cc98c412c3c9177113ae02",
    "cir_fit": "183f3566556a6dc3167740a8",
    "cir_paths": "89c8212a52b4f534328252d7",
    "conditional_copula_sample": "0d88fc18669c66f6e783dd83",
    "copula_loss_simulation": "6b4f80783b7926000ca51eb4",
    "correlated_gbm_paths": "f7e542e63e4da6f6263e3da4",
    "creditmetrics": "5d39fa9ab7c88613575a1a19",
    "delta_gamma_var": "63b21608bf5d4dc953d26699",
    "factor_copula_losses/gaussian": "4f271093fe2e2954e28c09dd",
    "factor_copula_losses/t": "3c0c7a2b0680c3de0a003659",
    "gaussian_copula_sample": "8bc1986e3592a7fd1ccdccbc",
    "gbm_paths": "02ec990b50956e1faf81a54a",
    "hawkes_simulate": "5dfab49faf2b2761003545bb",
    "heston_paths": "0fd7f6f23ed93671c3a2100f",
    "importance_shift": "6af678db7a4eb3d093cb809c",
    "lmm_paths": "d7c1e5d80bb47231e3aeabc4",
    "local_vol_paths": "53648c92e771238015dbc26f",
    "mc_payoff_price": "68277f44a8d7d9360ff313bc",
    "merton_jump_fit": "bac219d197536a1725609a63",
    "ou_paths": "20db68ceee7a03c42419a1b1",
    "sde_paths": "97050b07209d0d298efeab5a",
    "sobol_normals": "42b9a833cc153557d4ae93e9",
    "stratified_normals/latin_hypercube": "3bc5c0532eae5c664a65497e",
    "stratified_normals/stratified": "30832550299de376ed4c38f8",
    "t_copula_sample": "dfcb81e849742bd748e4a658",
}


def digest(value) -> str:
    """A sha256 of the output's every array (dtype, shape and bytes), a dict's keys in order."""
    found = hashlib.sha256()

    def feed(item):
        if isinstance(item, dict):
            for key in sorted(item):
                found.update(str(key).encode())
                feed(item[key])
        elif isinstance(item, (list, tuple)):
            for each in item:
                feed(each)
        else:
            array = np.ascontiguousarray(np.asarray(item))
            found.update(f"{array.dtype}{array.shape}".encode())
            found.update(array.tobytes())
    feed(value)
    return found.hexdigest()[:24]


def sampled() -> set[str]:
    """Every library function with an `rng` argument, read from the source."""
    names = set()
    for path in sorted((ROOT / "agent" / "qf").glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if (isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
                    and "rng" in [a.arg for a in node.args.args + node.args.kwonlyargs]):
                names.add(node.name)
    return names


def test_the_table_holds_every_sampler_and_its_pin():
    """A sampler added later joins the table, or this fails: every function taking `rng` has a row, and every row a
    pin."""
    assert {name.split("/")[0] for name in ROWS} == sampled()
    assert set(PINS) == set(ROWS)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_a_generator_seeded_seven_gives_the_pinned_output_and_an_int_seed_the_same(name):
    call, _ = ROWS[name]
    assert digest(call(np.random.default_rng(7))) == PINS[name]
    assert digest(call(7)) == PINS[name]


@pytest.mark.parametrize("name", sorted(ROWS))
def test_the_legacy_stream_is_drawn_as_the_card_names(name):
    """Under RandomState(7) the sampler leaves the stream where its card's draws, taken directly, leave a twin;
    numpy.random.seed(7) and None give the same output (None for the global stream; conditional_copula_sample's None
    says the uniforms are given)."""
    call, draws = ROWS[name]
    stream, twin = np.random.RandomState(7), np.random.RandomState(7)
    found = call(stream)
    draws(twin)
    mine, theirs = stream.get_state(), twin.get_state()
    assert np.array_equal(mine[1], theirs[1]) and mine[2:] == theirs[2:]
    assert digest(found) == digest(call(np.random.RandomState(7)))
    assert digest(found) != PINS[name]                  # the other stream: another sample
    if name == "conditional_copula_sample":
        return
    saved = np.random.get_state()
    try:
        np.random.seed(7)
        assert digest(call(None)) == digest(found)
    finally:
        np.random.set_state(saved)


def test_the_plain_transforms_by_hand_from_the_legacy_draws():
    """The output from the twin's draws by each card's formula: GBM's exp of the cumulated steps, the Gaussian and t
    copulas' N(Z L') and t_nu(Z L' / sqrt(W / nu)), CreditMetrics' Z = sqrt(rho) M + sqrt(1 - rho) e against the
    worst-first thresholds, the bootstrap's positions, the conditional sample's uniforms, the Hawkes thinning."""
    r = np.random.RandomState(7)
    half = r.standard_normal((3, 2))
    z, step = np.vstack([half, -half]), np.diff([0.0, 0.25, 1.0])
    expected = 50.0 * np.exp(np.cumsum((0.04 - 0.045) * step + 0.3 * np.sqrt(step) * z, axis=1))
    assert ROWS["gbm_paths"][0](np.random.RandomState(7)) == pytest.approx(expected, rel=1e-14)
    lower = np.linalg.cholesky(np.array(CORR))
    r = np.random.RandomState(7)
    assert qf.gaussian_copula_sample(CORR, n=5, rng=np.random.RandomState(7)) == pytest.approx(
        ndtr(r.standard_normal((5, 2)) @ lower.T), rel=1e-14)
    r = np.random.RandomState(7)
    x, w = r.standard_normal((5, 2)) @ lower.T, r.chisquare(4.0, 5)
    assert qf.t_copula_sample(CORR, nu=4.0, n=5, rng=np.random.RandomState(7)) == pytest.approx(
        stats.t.cdf(x / np.sqrt(w / 4.0)[:, None], 4.0), rel=1e-12)
    r = np.random.RandomState(7)
    factor, noise = r.standard_normal(7), r.standard_normal((7, 3))
    latent = math.sqrt(0.3) * factor[:, None] + math.sqrt(0.7) * noise
    cuts = ndtri(np.cumsum(np.array(TRANSITION)[:, ::-1], axis=1))   # default, the middle state, the best: worst first
    rows = [0, 1, 1]
    states = np.array([[2 - int(np.searchsorted(cuts[rows[i]][:2], latent[k, i], side="right")) for i in range(3)]
                       for k in range(7)])
    found = qf.creditmetrics(transition=TRANSITION, initial=rows, values=VALUES, rho=0.3, n=7,
                             rng=np.random.RandomState(7))
    assert np.array_equal(found["states"], states)
    assert found["portfolio_values"] == pytest.approx([sum(VALUES[i][states[k, i]] for i in range(3))
                                                       for k in range(7)], rel=1e-15)
    assert np.array_equal(ROWS["block_bootstrap/iid"][0](np.random.RandomState(7)),
                          np.random.RandomState(7).randint(0, 9, (3, 9)))
    given = qf.conditional_copula_sample(u=[0.2, 0.5, 0.7], family="clayton", parameter=1.5, nu=None,
                                         inverse="closed_form", w=np.random.RandomState(7).uniform(size=(3,)),
                                         rng=None)
    assert np.array_equal(ROWS["conditional_copula_sample"][0](np.random.RandomState(7)), given)
    assert np.array_equal(ROWS["hawkes_simulate"][0](np.random.RandomState(7)),
                          _hawkes(np.random.RandomState(7), 0.8, 0.5, 1.5, 0.0, 6.0))


def test_a_stream_the_helper_does_not_take_is_refused_by_name():
    for bad in (4.2, "7", -1, True):
        with pytest.raises(ValueError, match="rng must be a numpy.random.Generator or an int seed"):
            qf.gaussian_copula_sample(CORR, n=5, rng=bad)
