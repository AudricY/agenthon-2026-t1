"""qf.panjer_recursion, qf.discretize_distribution, qf.lattice_backward and qf.implied_parameter (lane
lib-primitives1; the owner's in633). References: Panjer's recursion by hand on a small compound Poisson (Klugman,
Panjer and Willmot 9.6's form), the definition by brute-force convolution for each (a, b, 0) count, the compound
moments lambda E[X] and lambda E[X^2], and qf.aggregate_loss's own recursion; the discretisations' definitions, the
mean that local moment matching keeps (Loss Models 9.6.5) and the bracket lower <= rounding <= upper; a CRR tree
against qf.binomial_price, and Hull's Hull-White trinomial tree (a = 0.1, sigma = 0.01, dt = 1 year: alpha 3.824%,
5.205%, 6.252%, Options, Futures, and Other Derivatives, the tree's illustration) repricing its curve, and a callable
bond on a finer such tree against qf.short_rate_tree; inverting qf.bs_price and qf.bond_price against qf.implied_vol
and qf.bond_yield (Brent 1973). Every input is this file's own or a textbook's; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import stats

import qf

pytestmark = pytest.mark.fast


def test_panjers_recursion_by_hand_on_a_small_compound_poisson():
    """lambda = 3, claims of 1 or 2 with probabilities 0.6 and 0.4: g_0 = e^-3, g_1 = 3 (0.6) g_0, g_2 = (3/2)(0.6
    g_1 + 2 (0.4) g_0)."""
    g = qf.panjer_recursion(counting="poisson", parameters={"lambda": 3.0}, severity=[0.0, 0.6, 0.4], points=5)
    g0 = math.exp(-3.0)
    g1 = 3.0 * 0.6 * g0
    assert g[:3] == pytest.approx([g0, g1, 1.5 * (0.6 * g1 + 0.8 * g0)], rel=1e-14)
    assert g.shape == (5,)


COUNTS = {   # each (a, b, 0) count, its law's pmf, and its a, b and p0 for 'ab0'
    "poisson": ({"lambda": 1.3}, stats.poisson(1.3).pmf, (0.0, 1.3, math.exp(-1.3))),
    "negative_binomial": ({"r": 2.5, "beta": 0.8}, stats.nbinom(2.5, 1 / 1.8).pmf,
                          (0.8 / 1.8, 1.5 * 0.8 / 1.8, 1.8 ** -2.5)),
    "binomial": ({"m": 6, "q": 0.35}, stats.binom(6, 0.35).pmf, (-0.35 / 0.65, 7 * 0.35 / 0.65, 0.65 ** 6)),
}


@pytest.mark.parametrize("counting", sorted(COUNTS))
def test_the_recursion_is_the_count_mixture_of_convolutions_for_each_count_and_its_a_b_form(counting):
    """P(S = x) = sum_n P(N = n) f^{*n}(x) (Loss Models 9.6), with an atom at 0 (g_0 = P_N(f_0)); the same from
    the count's a, b and p0 ('ab0')."""
    f = np.array([0.1, 0.45, 0.3, 0.15])
    parameters, pmf, (a, b, p0) = COUNTS[counting]
    direct, power = np.zeros(64), np.array([1.0])
    for n in range(60):
        direct[:min(power.size, 64)] += pmf(n) * power[:64]
        power = np.convolve(power, f)
    found = qf.panjer_recursion(counting=counting, parameters=parameters, severity=f, points=64)
    assert found == pytest.approx(direct, abs=1e-14)
    by_ab = qf.panjer_recursion(counting="ab0", parameters={"a": a, "b": b, "p0": p0}, severity=f, points=64)
    assert by_ab == pytest.approx(direct, abs=1e-13)


def test_the_compound_poissons_moments_and_aggregate_losss_own_recursion():
    """E[S] = lambda E[X], Var[S] = lambda E[X^2]; aggregate_loss by 'panjer' on the severity's rounding grid is
    this recursion on discretize_distribution's grid."""
    f = np.zeros(400)
    f[1:6] = [0.3, 0.25, 0.2, 0.15, 0.1]
    g = qf.panjer_recursion(counting="poisson", parameters={"lambda": 4.0}, severity=f, points=400)
    k = np.arange(400)
    mean_x, square_x = f @ k, f @ k ** 2
    assert g.sum() == pytest.approx(1.0, abs=1e-12)
    assert g @ k == pytest.approx(4.0 * mean_x, rel=1e-12)
    assert g @ k ** 2 - (g @ k) ** 2 == pytest.approx(4.0 * square_x, rel=1e-10)
    law = stats.gamma(a=2.0, scale=40.0)
    grid = qf.discretize_distribution(law=law, step=10.0, points=64, method="rounding")
    held = qf.aggregate_loss(severity=law, frequency="negative_binomial", frequency_parameters={"r": 3.0, "beta": 1.5},
                             step=10.0, points=64, cover=0.8, discretisation="rounding", method="panjer",
                             alphas=[0.5], es_rule="beyond")      # the grid cut short of the tail, its cover too
    found = qf.panjer_recursion(counting="negative_binomial", parameters={"r": 3.0, "beta": 1.5},
                                severity=grid["pmf"], points=64)
    assert np.array_equal(found, held["pmf"]) and grid["tail_mass"] == held["tail_mass"] > 1e-7


def test_panjer_recursion_pads_a_short_severity_and_refuses_bad_inputs():
    short = qf.panjer_recursion(counting="poisson", parameters={"lambda": 2.0}, severity=[0.2, 0.8], points=30)
    padded = qf.panjer_recursion(counting="poisson", parameters={"lambda": 2.0}, severity=[0.2, 0.8] + [0.0] * 28,
                                 points=30)
    assert np.array_equal(short, padded)
    for change, message in (({"severity": [0.5, 0.7]}, "sum at most 1"), ({"severity": [-0.1, 1.1]}, "none negative"),
                            ({"parameters": {"lam": 2.0}}, "'lambda'"), ({"counting": "geometric"}, "must be one of"),
                            ({"counting": "ab0", "parameters": {"a": 1.2, "b": 0.0, "p0": 0.5}}, "a < 1")):
        with pytest.raises(ValueError, match=message):
            qf.panjer_recursion(**{"counting": "poisson", "parameters": {"lambda": 2.0}, "severity": [0.2, 0.8],
                                   "points": 30, **change})


EXPONENTIAL = stats.expon(scale=30.0)


def test_poisson_unit_claims_survive_initial_probability_underflow():
    """Loss Models (2019), 9.6.2: split a Poisson intensity then self-convolve. Unit claims make the aggregate
    Poisson itself; its log-PMF reference remains representable near the mean even when exp(-lambda) is zero."""
    g = qf.panjer_recursion(counting="poisson", parameters={"lambda": 800.0}, severity=[0.0, 1.0], points=1200)
    assert g == pytest.approx(stats.poisson(800.0).pmf(np.arange(1200)), abs=2e-14)
    assert g.sum() == pytest.approx(1.0, abs=1e-12)


def test_ab0_parameters_must_describe_a_probability_distribution():
    for p in ({"a": 0.0, "b": 1.0, "p0": 0.9}, {"a": 0.0, "b": -1.0, "p0": 1.0},
              {"a": -1.0, "b": 2.5, "p0": 2.0 ** -1.5}):
        with pytest.raises(ValueError, match="normalized count"):
            qf.panjer_recursion(counting="ab0", parameters=p, severity=[0.0, 1.0], points=10)


@pytest.mark.parametrize("law", [stats.uniform(loc=0.001, scale=0.001), stats.gamma(a=1e6, scale=1e-7)])
def test_local_moment_matching_preserves_a_law_concentrated_inside_one_span(law):
    """Loss Models, local moment matching: with negligible tail past the grid, its mean equals the law's.
    Uniform and gamma means are loc + scale/2 and shape*scale; a wide grid must still sample narrow mass."""
    result = qf.discretize_distribution(law=law, step=1.0, points=3, method="moment_matching")
    assert result["grid"] @ result["pmf"] == pytest.approx(law.mean(), abs=1e-12)
    assert result["pmf"].sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("branching", [[], [[[1.9, 0.9]]], [[[0, 1], [0, 1]]], [[0, 1]]])
def test_custom_branching_requires_integer_indices_from_one_root(branching):
    with pytest.raises(ValueError, match="branching"):
        qf.lattice_backward(terminal=[80.0, 120.0], branching=branching, probabilities=[0.5, 0.5], discount=0.99,
                            cash_flows=None, exercise="european", exercise_value=None, exercise_steps=None,
                            exerciser="holder", paid_on_exercise=True)


def test_each_discretisation_is_its_definition():
    """On x_j = j h: rounding F((j + 1/2) h) - F((j - 1/2) h), f_0 = F(h/2); lower F((j + 1) h) - F(j h); upper
    F(j h) - F((j - 1) h), f_0 = F(0); each with the mass past the grid as tail_mass."""
    h, n = 5.0, 40
    F = EXPONENTIAL.cdf
    j = np.arange(n)
    want = {"rounding": np.r_[F(h / 2), F((j[1:] + 0.5) * h) - F((j[1:] - 0.5) * h)],
            "lower": F((j + 1) * h) - F(j * h), "upper": np.r_[F(0.0), F(j[1:] * h) - F((j[1:] - 1) * h)]}
    for method, f in want.items():
        found = qf.discretize_distribution(law=EXPONENTIAL, step=h, points=n, method=method)
        assert found["pmf"] == pytest.approx(f, abs=1e-15), method
        assert found["grid"] == pytest.approx(h * j) and found["tail_mass"] == pytest.approx(1 - f.sum(), abs=1e-14)


def test_local_moment_matching_keeps_the_mean_and_the_others_bracket_it():
    """Matching the first moment span by span: the grid's mean is E[X] but for the tail past the grid, the
    probabilities the limited expected values' second differences (Loss Models 9.6.5); 'lower' sits below the mean
    and 'upper' above, rounding's within h^2."""
    law = stats.gamma(a=2.0, scale=50.0)
    means = {}
    for method in ("rounding", "lower", "upper", "moment_matching"):
        found = qf.discretize_distribution(law=law, step=10.0, points=400, method=method)
        means[method] = found["grid"] @ found["pmf"]
        assert found["pmf"].sum() + found["tail_mass"] == pytest.approx(1.0, abs=1e-12) and found["pmf"].min() >= 0
    assert means["moment_matching"] == pytest.approx(100.0, rel=1e-12)
    assert means["lower"] < means["rounding"] < means["upper"]
    assert abs(means["rounding"] - 100.0) < 1e-2 and means["upper"] - means["lower"] == pytest.approx(10.0, rel=1e-6)
    h, n = 7.0, 30
    found = qf.discretize_distribution(law=EXPONENTIAL, step=h, points=n, method="moment_matching")
    limited = 30.0 * (1 - np.exp(-np.arange(n + 1) * h / 30.0))          # E[X ^ u] of the exponential, by hand
    assert found["pmf"][0] == pytest.approx(1 - limited[1] / h, rel=1e-12)
    assert found["pmf"][1:] == pytest.approx((2 * limited[1:n] - limited[:n - 1] - limited[2:]) / h, rel=1e-10)
    with pytest.raises(ValueError, match="X >= 0"):
        qf.discretize_distribution(law=stats.norm(), step=0.5, points=20, method="moment_matching")


def crr_nodes(spot, up, i):
    return spot * up ** np.arange(-i, i + 1, 2)


@pytest.mark.parametrize("exercise", ["european", "american"])
def test_a_crr_tree_on_the_lattice_is_binomial_price(exercise):
    """Cox, Ross and Rubinstein's tree (u = e^{vol sqrt(dt)}, p = (e^{(r - q) dt} - d)/(u - d)) backward on the
    lattice: binomial_price's European and American puts."""
    s, k, t, r, q, vol, n = 50.0, 52.0, 2.0, 0.05, 0.01, 0.3, 120
    dt = t / n
    up = math.exp(vol * math.sqrt(dt))
    p = (math.exp((r - q) * dt) - 1 / up) / (up - 1 / up)
    found = qf.lattice_backward(terminal=np.maximum(k - crr_nodes(s, up, n), 0.0), branching="binomial",
                                probabilities=[p, 1 - p], discount=math.exp(-r * dt), cash_flows=None,
                                exercise=exercise,
                                exercise_value=None if exercise == "european" else
                                (lambda i: np.maximum(k - crr_nodes(s, up, i), 0.0)),
                                exercise_steps=None, exerciser="holder", paid_on_exercise=True)
    held = qf.binomial_price(option="put", exercise=exercise, tree="crr", spot=s, strike=k, maturity=t, rate=r,
                             dividend_yield=q, vol=vol, steps=n)
    assert found["value"] == pytest.approx(held["price"], rel=1e-12)
    assert len(found["values"]) == n + 1 and found["values"][1].shape == (2,)
    assert found["exercised"][0].shape == (1,) and (exercise == "american") == bool(np.any(np.concatenate(
        found["exercised"])))


def test_an_american_put_deep_in_the_money_is_exercised_today():
    """K = 100, S = 50 and a high rate: holding loses the interest on the strike, so the value is the intrinsic K - S
    at the first node, exercised there."""
    s, k, r, n, dt = 50.0, 100.0, 0.10, 20, 0.05
    up = math.exp(0.2 * math.sqrt(dt))
    p = (math.exp(r * dt) - 1 / up) / (up - 1 / up)
    found = qf.lattice_backward(terminal=np.maximum(k - crr_nodes(s, up, n), 0.0), branching="binomial",
                                probabilities=[p, 1 - p], discount=math.exp(-r * dt), cash_flows=None,
                                exercise="american", exercise_value=lambda i: np.maximum(k - crr_nodes(s, up, i), 0.0),
                                exercise_steps=None, exerciser="holder", paid_on_exercise=True)
    assert found["value"] == pytest.approx(k - s, rel=1e-15) and found["exercised"][0][0]


def test_one_step_is_the_discounted_expectation_and_the_exercise_rules():
    """One binomial step: V = D (p V_up + (1 - p) V_down) + c; a holder's exercise the max, an issuer's the min;
    paid_on_exercise adds the step's flow to the exercise value, or the exercise value replaces it."""
    common = {"terminal": [90.0, 110.0], "branching": "binomial", "probabilities": [0.6, 0.4], "discount": 0.95}
    held = 0.95 * (0.6 * 110.0 + 0.4 * 90.0)
    plain = qf.lattice_backward(**common, cash_flows=[[3.0]], exercise="european", exercise_value=None,
                                exercise_steps=None, exerciser="holder", paid_on_exercise=True)
    assert plain["value"] == pytest.approx(held + 3.0, rel=1e-15)
    cases = {("holder", True): 3.0 + max(held, 99.0), ("holder", False): max(held + 3.0, 99.0),
             ("issuer", True): 3.0 + min(held, 95.0), ("issuer", False): min(held + 3.0, 95.0)}
    for (exerciser, paid), want in cases.items():
        found = qf.lattice_backward(**common, cash_flows=[3.0], exercise="bermudan",
                                    exercise_value=[99.0 if exerciser == "holder" else 95.0], exercise_steps=[0],
                                    exerciser=exerciser, paid_on_exercise=paid)
        assert found["value"] == pytest.approx(want, rel=1e-15), (exerciser, paid)


def test_a_trinomial_lattice_is_the_explicit_branching_and_a_martingale():
    """A symmetric trinomial walk x_{i+1} = x_i + (1, 0, -1) with probabilities (1/6, 2/3, 1/6): a payoff linear in
    the level is worth its level today, discounted; the same lattice written as explicit branching agrees."""
    n = 6
    levels = np.arange(-n, n + 1, dtype=float)
    kwargs = {"probabilities": [1 / 6, 2 / 3, 1 / 6], "discount": 0.99, "cash_flows": None, "exercise": "european",
              "exercise_value": None, "exercise_steps": None, "exerciser": "holder", "paid_on_exercise": True}
    found = qf.lattice_backward(terminal=10.0 + levels, branching="trinomial", **kwargs)
    assert found["value"] == pytest.approx(10.0 * 0.99 ** n, rel=1e-14)
    children = [np.arange(2 * i + 1)[:, None] + np.array([2, 1, 0])[None, :] for i in range(n)]
    explicit = qf.lattice_backward(terminal=np.maximum(levels - 1.0, 0.0), branching=children, **kwargs)
    same = qf.lattice_backward(terminal=np.maximum(levels - 1.0, 0.0), branching="trinomial", **kwargs)
    assert explicit["value"] == pytest.approx(same["value"], rel=1e-15)


def hull_white_tree(a, sigma, dt, zeros):
    """Hull's Hull-White trinomial tree written out (Options, Futures, and Other Derivatives, the tree's
    construction): dx = sigma sqrt(3 dt), M = -a dt, j_max the least integer above 0.184 / (a dt), Hull's three
    branchings and their probabilities, alpha_m from the Arrow-Debreu prices Q (continuous compounding). Returns
    each step's children (node indices), probabilities and one-step discounts, and the alphas."""
    steps, dx, m = len(zeros), sigma * math.sqrt(3 * dt), -a * dt
    jmax = math.floor(0.184 / (a * dt)) + 1
    width = [min(i, jmax) for i in range(steps + 1)]
    q, alphas, children, probabilities, discounts = np.array([1.0]), [], [], [], []
    for i in range(steps):
        js = np.arange(-width[i], width[i] + 1)
        alpha = math.log(q @ np.exp(-js * dx * dt) / zeros[i]) / dt
        kids, probs = [], []
        for j in js:
            jm = j * j * m * m
            if j == jmax:
                kids.append((j, j - 1, j - 2))
                probs.append((7 / 6 + (jm + 3 * j * m) / 2, -1 / 3 - jm - 2 * j * m, 1 / 6 + (jm + j * m) / 2))
            elif j == -jmax:
                kids.append((j + 2, j + 1, j))
                probs.append((1 / 6 + (jm - j * m) / 2, -1 / 3 - jm + 2 * j * m, 7 / 6 + (jm - 3 * j * m) / 2))
            else:
                kids.append((j + 1, j, j - 1))
                probs.append((1 / 6 + (jm + j * m) / 2, 2 / 3 - jm, 1 / 6 + (jm - j * m) / 2))
        kids, probs = np.array(kids) + width[i + 1], np.array(probs)
        discount = np.exp(-(alpha + js * dx) * dt)
        nxt = np.zeros(2 * width[i + 1] + 1)
        for column in range(3):
            np.add.at(nxt, kids[:, column], q * discount * probs[:, column])
        q = nxt
        alphas.append(alpha)
        children.append(kids)
        probabilities.append(probs)
        discounts.append(discount)
    return children, probabilities, discounts, alphas


def test_hulls_trinomial_tree_reprices_its_curve_and_gives_hulls_alphas():
    """Hull's illustration: a = 0.1, sigma = 0.01, dt = 1, zero rates 3.824%, 4.512%, 5.086% (continuously
    compounded) give alpha 3.824%, 5.205%, 6.252%; each zero-coupon bond backward on the lattice is the curve's
    discount factor."""
    zeros = [math.exp(-0.03824), math.exp(-0.04512 * 2), math.exp(-0.05086 * 3)]
    children, probabilities, discounts, alphas = hull_white_tree(0.1, 0.01, 1.0, zeros)
    assert [round(alpha, 5) for alpha in alphas] == [0.03824, 0.05205, 0.06252]
    for maturity in (1, 2, 3):
        found = qf.lattice_backward(terminal=np.ones(children[maturity - 1].max() + 1), branching=children[:maturity],
                                    probabilities=probabilities[:maturity], discount=discounts[:maturity],
                                    cash_flows=None, exercise="european", exercise_value=None, exercise_steps=None,
                                    exerciser="holder", paid_on_exercise=True)
        assert found["value"] == pytest.approx(zeros[maturity - 1], rel=1e-13), maturity


def test_a_callable_bond_on_hulls_tree_is_short_rate_trees():
    """A bond paying 5 a year for five years and 100 at the end, callable at 100 on its coupon dates from year 2 (the
    coupon paid, then the call), on a quarterly Hull-White tree (Hull's branching and Euler moments): the lattice's
    price is short_rate_tree's, the issuer's call below the straight bond and both below a holder's put."""
    dt, years = 0.25, 5
    steps = int(years / dt)
    times = dt * np.arange(1, steps + 1)
    zeros = np.exp(-(0.03 + 0.004 * times) * times)
    children, probabilities, discounts, _ = hull_white_tree(0.1, 0.012, dt, zeros)
    flows = [np.full(children[i].shape[0], 5.0 if i and i % 4 == 0 else 0.0) for i in range(steps)]
    common = {"terminal": np.full(children[-1].max() + 1, 105.0), "branching": children,
              "probabilities": probabilities, "discount": discounts, "cash_flows": flows, "paid_on_exercise": True}
    calls = [8, 12, 16]
    callable_bond = qf.lattice_backward(**common, exercise="bermudan", exercise_value=lambda i: 100.0,
                                        exercise_steps=calls, exerciser="issuer")
    straight = qf.lattice_backward(**common, exercise="european", exercise_value=None, exercise_steps=None,
                                   exerciser="issuer")
    held = qf.short_rate_tree(model="hull_white", mean_reversion=0.1, sigma=0.012, dt=dt,
                              curve=(times, zeros), interpolation="linear_zero", discretization="euler",
                              branching="hull_white", compounding="continuous", flow_times=[1.0, 2.0, 3.0, 4.0, 5.0],
                              flows=[5.0, 5.0, 5.0, 5.0, 105.0], exercise="callable", exercise_times=[2.0, 3.0, 4.0],
                              strikes=[100.0, 100.0, 100.0], exercise_flows="after", oas=0.0, market_price=None)
    assert callable_bond["value"] == pytest.approx(held["price"], rel=1e-12)
    assert straight["value"] == pytest.approx(held["straight"], rel=1e-12)
    puttable = qf.lattice_backward(**common, exercise="bermudan", exercise_value=lambda i: 100.0,
                                   exercise_steps=calls, exerciser="holder")
    assert callable_bond["value"] < straight["value"] < puttable["value"]


def test_lattice_backward_refuses_what_does_not_fit():
    base = {"terminal": [1.0, 2.0, 3.0], "branching": "binomial", "probabilities": [0.5, 0.5], "discount": 1.0,
            "cash_flows": None, "exercise": "european", "exercise_value": None, "exercise_steps": None,
            "exerciser": "holder", "paid_on_exercise": True}
    for change, message in (({"probabilities": [0.6, 0.6]}, "sum to 1"),
                            ({"branching": "trinomial", "terminal": [1.0, 2.0]}, "nodes"),
                            ({"exercise": "american"}, "needs exercise_value"),
                            ({"exercise": "bermudan", "exercise_value": [0.0, 0.0], "exercise_steps": [2]},
                             "exercise_steps must name"),
                            ({"cash_flows": [1.0]}, "one entry a step"), ({"exerciser": "writer"}, "must be one of"),
                            ({"discount": [[0.9, 0.9, 0.9], 0.9]}, "must fit")):
        with pytest.raises(ValueError, match=message):
            qf.lattice_backward(**{**base, **change})


BS = {"option": "call", "spot": 100.0, "strike": 110.0, "maturity": 0.5, "rate": 0.03, "dividend_yield": 0.01}


@pytest.mark.parametrize("method", ["brent", "bisection"])
def test_inverting_bs_price_is_implied_vol_and_inverting_bond_price_is_bond_yield(method):
    """The bracket (0.05, 0.1) misses a 27% vol: 'up' doubles it until it holds the root."""
    price = qf.bs_price(**BS, vol=0.27)
    found = qf.implied_parameter(pricer=lambda v: qf.bs_price(**BS, vol=v), target=price, bracket=(0.05, 0.1),
                                 expand="up", method=method, tolerance=1e-13)
    held = qf.implied_vol(option="call", price=price, spot=100.0, strike=110.0, maturity=0.5, rate=0.03,
                          dividend_yield=0.01, method="bisection")
    assert found["value"] == pytest.approx(held, abs=1e-11) and found["found"] and abs(found["error"]) < 1e-9
    assert found["bracket"][0] == 0.05 and found["bracket"][1] > 0.27 and found["iterations"] > 0
    flows, times = [5.0, 5.0, 105.0], [1.0, 2.0, 3.0]
    bond = qf.bond_price(cash_flows=flows, times=times, ytm=0.061, frequency=2)
    found = qf.implied_parameter(pricer=lambda y: qf.bond_price(cash_flows=flows, times=times, ytm=y, frequency=2),
                                 target=bond, bracket=(0.0, 0.2), expand="none", method=method, tolerance=1e-13)
    assert found["value"] == pytest.approx(qf.bond_yield(price=bond, cash_flows=flows, times=times, frequency=2),
                                           abs=1e-11)


def test_brents_method_takes_fewer_steps_than_bisection():
    price = qf.bs_price(**BS, vol=0.27)
    steps = {method: qf.implied_parameter(pricer=lambda v: qf.bs_price(**BS, vol=v), target=price, bracket=(0.01, 2.0),
                                          expand="none", method=method, tolerance=1e-12)["iterations"]
             for method in ("brent", "bisection")}
    assert steps["brent"] < steps["bisection"] / 2 and steps["bisection"] >= 35


def test_the_bracket_expands_on_the_side_named_and_a_price_out_of_reach_is_not_found():
    """A price below the call's intrinsic value is no vol's; 'down' reaches a root below the bracket, 'both' either
    side, 'none' leaves it; a root at an end is that end."""
    line = lambda x: 2.0 * x - 3.0       # noqa: E731
    assert qf.implied_parameter(pricer=line, target=0.0, bracket=(4.0, 5.0), expand="down", method="brent",
                                tolerance=1e-14)["value"] == pytest.approx(1.5, abs=1e-12)
    assert qf.implied_parameter(pricer=line, target=0.0, bracket=(10.0, 11.0), expand="both", method="bisection",
                                tolerance=1e-14)["value"] == pytest.approx(1.5, abs=1e-12)
    missed = qf.implied_parameter(pricer=line, target=0.0, bracket=(4.0, 5.0), expand="up", method="brent",
                                  tolerance=1e-12)
    assert not missed["found"] and math.isnan(missed["value"]) and math.isnan(missed["error"])
    assert qf.implied_parameter(pricer=line, target=0.0, bracket=(1.5, 2.0), expand="none", method="brent",
                                tolerance=1e-12)["value"] == 1.5
    intrinsic = 100.0 - 90.0 * math.exp(-0.03)
    below = qf.implied_parameter(pricer=lambda v: qf.bs_price(option="call", spot=100.0, strike=90.0, maturity=1.0,
                                                              rate=0.03, dividend_yield=0.0, vol=v),
                                 target=intrinsic - 0.5, bracket=(0.01, 1.0), expand="up", method="brent",
                                 tolerance=1e-12)
    assert not below["found"]
    with pytest.raises(ValueError, match="low < high"):
        qf.implied_parameter(pricer=line, target=0.0, bracket=(2.0, 1.0), expand="none", method="brent",
                             tolerance=1e-12)
