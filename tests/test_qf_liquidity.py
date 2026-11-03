"""Lane lib-econometrics1 (runs/lanes/lib-econometrics1-prompt.md): qf.roll_spread, qf.book_imbalance,
qf.order_flow_imbalance, qf.information_share, qf.hawkes_fit and qf.hawkes_simulate. Roll (1984): the bounce's
covariance -s^2/4 returns s, pandas' own lagged covariance the estimate; Cont, Kukanov and Stoikov (2014): each case
of the order-flow definition worked by hand; the depth imbalance inside [-1, 1] and 0 for a symmetric book; Hasbrouck
(1995): the shares sum to 1 at every ordering, Baillie, Booth, Tse and Zabotina's (2002) two-venue closed form, a
leading venue's share near 1; Hawkes (1971) and Ozaki (1979): the likelihood's recursion against the intensity summed
event by event, the expected count from an empty start, a simulated sample's parameters returned. No unit's values
appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401  (first: it puts agent/ on the path outside pytest too)
import qf

pytestmark = pytest.mark.fast


# ------------------------------------------------------------------------------------------------ Roll


def test_roll_recovers_the_spread_of_a_random_bounce():
    """Roll's model: P_t = m_t + (s/2) q_t, q_t = +-1 with equal odds and independent, m_t a random walk; Cov(dP_t,
    dP_{t-1}) = -s^2/4 (Hasbrouck 2007, 3.2: -c^2 with c the half spread)."""
    rng = np.random.RandomState(71)
    n, s = 200_000, 0.10
    prices = 100.0 + np.cumsum(0.02 * rng.normal(size=n)) + s / 2 * rng.choice([-1.0, 1.0], size=n)
    found = qf.roll_spread(prices=prices, changes="price", ddof=1, positive_cov="nan")
    assert found["spread"] == pytest.approx(s, rel=0.01) and found["covariance"] == pytest.approx(-s * s / 4, rel=0.02)
    relative = qf.roll_spread(prices=prices, changes="log", ddof=1, positive_cov="nan")
    assert relative["spread"] == pytest.approx(s * np.mean(1 / prices), rel=0.02)     # s / P, a fraction of the price


def test_roll_is_pandas_lagged_covariance_and_a_strict_alternation_gives_twice_the_spread():
    """The pairs' covariance is pandas' Series.cov(Series.shift(1)) (ddof 1). A strictly alternating bounce is not
    Roll's model: its changes are +-s by turns, the covariance -s^2, the estimate 2s."""
    pandas = pytest.importorskip("pandas")
    rng = np.random.RandomState(72)
    prices = 50.0 + np.cumsum(0.01 * rng.normal(size=300)) + 0.025 * rng.choice([-1.0, 1.0], size=300)
    d = pandas.Series(np.diff(prices))
    covariance = d.cov(d.shift(1))
    found = qf.roll_spread(prices=prices, changes="price", ddof=1, positive_cov="nan")
    assert found["covariance"] == pytest.approx(covariance, rel=1e-12) and found["n_pairs"] == 298
    assert found["spread"] == pytest.approx(2 * math.sqrt(-covariance), rel=1e-12)
    alternating = 20.0 + 0.05 * (-1.0) ** np.arange(12)          # ten pairs: each lagged series' mean is 0
    exact = qf.roll_spread(prices=alternating, changes="price", ddof=0, positive_cov="nan")
    assert exact["spread"] == pytest.approx(2 * 0.10, rel=1e-12)


def test_roll_has_no_spread_where_the_changes_are_not_negatively_correlated():
    trending = 10.0 + np.cumsum(np.r_[0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4])
    assert qf.roll_spread(prices=trending, changes="price", ddof=1, positive_cov="zero")["spread"] == 0.0
    assert math.isnan(qf.roll_spread(prices=trending, changes="price", ddof=1, positive_cov="nan")["spread"])
    with pytest.raises(ValueError, match="positive_cov"):
        qf.roll_spread(prices=trending, changes="price", ddof=1, positive_cov="abs")


# ------------------------------------------------------------------------------------------------ the book


def book(rows):
    return {name: np.array([row[i] for row in rows], dtype=float) for i, name in
            enumerate(("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"))}


def test_the_depth_imbalance_by_hand_symmetric_and_bounded():
    two = book([([99.9, 99.8], [500, 300], [100.0, 100.1], [300, 400]),
                ([99.9, 99.8], [200, 200], [100.0, 100.1], [200, 200]),
                ([99.9, 99.8], [0, 0], [100.0, 100.1], [100, 50])])
    found = qf.book_imbalance(**two, weights=[1.0, 1.0])
    assert found["imbalance"] == pytest.approx([(800 - 700) / 1500, 0.0, -1.0])
    best = qf.book_imbalance(**two, weights=[1.0])
    assert best["imbalance"] == pytest.approx([200 / 800, 0.0, -1.0])
    decaying = qf.book_imbalance(**two, weights=[1.0, 0.5])
    assert decaying["imbalance"][0] == pytest.approx((650 - 500) / 1150)
    assert found["weighted_mid"][0] == pytest.approx((100.0 * 500 + 99.9 * 300) / 800)
    assert found["weighted_mid"][1] == pytest.approx(found["mid"][1]) and found["mid"][0] == pytest.approx(99.95)
    rng = np.random.RandomState(73)
    sizes = rng.uniform(0, 1000, size=(50, 5))
    random = qf.book_imbalance(bid_prices=np.tile(100 - np.arange(5) * 0.1, (50, 1)), bid_sizes=sizes,
                               ask_prices=np.tile(100.1 + np.arange(5) * 0.1, (50, 1)),
                               ask_sizes=rng.uniform(0, 1000, size=(50, 5)), weights=np.exp(-0.5 * np.arange(5)))
    assert np.all(np.abs(random["imbalance"]) <= 1)
    mirror = qf.book_imbalance(bid_prices=[99.0], bid_sizes=[700], ask_prices=[101.0], ask_sizes=[700], weights=[1.0])
    assert mirror["imbalance"] == pytest.approx([0.0]) and mirror["weighted_mid"] == pytest.approx([100.0])


def test_the_order_flow_imbalance_by_cont_kukanov_and_stoikovs_cases():
    """Each snapshot change worked from the definition: a bid that rises adds its new size, one that stays adds its
    change in size, one that falls takes off its old size; the ask the other way."""
    snapshots = book([(100.0, 500, 100.2, 400),
                      (100.1, 300, 100.2, 450),     # bid up: +300; ask same: -(450 - 400)
                      (100.1, 350, 100.1, 200),     # bid same: +50; ask down: -200
                      (100.0, 600, 100.1, 150),     # bid down: -350; ask same: -(150 - 200)
                      (100.0, 600, 100.3, 900)])    # bid same: 0; ask up: +150
    found = qf.order_flow_imbalance(**snapshots, weights=[1.0])
    assert found["ofi"] == pytest.approx([300 - 50, 50 - 200, -350 + 50, 0 + 150])
    assert found["cumulative"] == pytest.approx(np.cumsum(found["ofi"]))
    still = book([(100.0, 500, 100.1, 400), (100.0, 620, 100.1, 380)])
    assert qf.order_flow_imbalance(**still, weights=[1.0])["ofi"] == pytest.approx([120 + 20])
    both = book([(100.0, 500, 100.1, 400), (100.0, 600, 100.1, 500)])
    assert qf.order_flow_imbalance(**both, weights=[1.0])["ofi"] == pytest.approx([0.0])


def test_the_multi_level_flow_weighs_each_levels_own_flow():
    levels = {"bid_prices": [[100.0, 99.9], [100.1, 100.0]], "bid_sizes": [[500, 400], [300, 450]],
              "ask_prices": [[100.2, 100.3], [100.2, 100.3]], "ask_sizes": [[400, 600], [350, 700]]}
    found = qf.order_flow_imbalance(**levels, weights=[1.0, 0.5])
    first, second = 300 - (350 - 400), 450 - (700 - 600)
    assert found["by_level"] == pytest.approx(np.array([[first, second]]))
    assert found["ofi"] == pytest.approx([first + 0.5 * second])
    with pytest.raises(ValueError, match="levels the weights name"):
        qf.order_flow_imbalance(**levels, weights=[1.0, 1.0, 1.0])


# ------------------------------------------------------------------------------------------------ information share


def two_venues(n, seed, lead_noise, follow_noise):
    """Venue 1 quotes the efficient price with small noise; venue 2 quotes it a period late."""
    rng = np.random.RandomState(seed)
    efficient = np.cumsum(0.01 * rng.normal(size=n + 1))
    return np.column_stack([efficient[1:] + lead_noise * rng.normal(size=n),
                            efficient[:-1] + follow_noise * rng.normal(size=n)])


def test_the_shares_sum_to_one_at_every_ordering_and_the_bounds_contain_them():
    rng = np.random.RandomState(74)
    efficient = np.cumsum(0.01 * rng.normal(size=801))
    prices = np.column_stack([efficient[1:], efficient[:-1] + 0.004 * rng.normal(size=800),
                              0.5 * (efficient[1:] + efficient[:-1]) + 0.003 * rng.normal(size=800)])
    found = qf.information_share(prices=prices, lags=2, constant=True)
    assert found["shares"].sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all(found["lower"] <= found["shares"] + 1e-12) and np.all(found["shares"] <= found["upper"] + 1e-12)
    reordered = qf.information_share(prices=prices[:, [2, 0, 1]], lags=2, constant=True)
    assert reordered["shares"].sum() == pytest.approx(1.0, rel=1e-12)
    assert reordered["upper"] == pytest.approx(found["upper"][[2, 0, 1]], rel=1e-9)
    assert found["component_share"].sum() == pytest.approx(1.0) and found["upper"][0] > 0.8


def test_two_venues_follow_baillie_booth_tse_and_zabotinas_closed_form():
    """BBTZ (2002), eq. 16-17: with gamma = alpha_perp scaled to sum to 1 and M the lower Cholesky factor of Omega,
    IS_1 = (g1 m11 + g2 m12)^2 / ((g1 m11 + g2 m12)^2 + (g2 m22)^2); the VECM's alpha and Omega are OLS's on [1,
    p1 - p2, the lagged changes]."""
    prices = two_venues(1500, 75, 0.002, 0.003)
    found = qf.information_share(prices=prices, lags=1, constant=True)
    dp = np.diff(prices, axis=0)
    design = np.column_stack([np.ones(dp.shape[0] - 1), prices[1:-1, 0] - prices[1:-1, 1], dp[:-1]])
    coef = np.linalg.lstsq(design, dp[1:], rcond=None)[0]
    errors = dp[1:] - design @ coef
    omega = errors.T @ errors / errors.shape[0]
    assert found["alpha"][:, 0] == pytest.approx(coef[1], rel=1e-9) and found["omega"] == pytest.approx(omega)
    a1, a2 = coef[1]
    g1, g2 = a2 / (a2 - a1), -a1 / (a2 - a1)
    (m11, _), (m12, m22) = np.linalg.cholesky(omega)
    first = (g1 * m11 + g2 * m12) ** 2 / ((g1 * m11 + g2 * m12) ** 2 + (g2 * m22) ** 2)
    assert found["shares"][0] == pytest.approx(first, rel=1e-9)
    assert found["component_share"] == pytest.approx([g1, g2], rel=1e-9)
    assert found["upper"][0] + found["lower"][1] == pytest.approx(1.0) and found["lower"][0] > 0.9


def test_the_information_share_refuses_one_venue_or_seven():
    with pytest.raises(ValueError, match="2 to 6 venues"):
        qf.information_share(prices=np.ones((50, 1)), lags=1, constant=True)


# ------------------------------------------------------------------------------------------------ Hawkes


def direct_loglik(times, start, end, mu, alpha, beta):
    """The Hawkes log-likelihood summed event by event, the intensity's every term written out."""
    intensity = [mu + alpha * sum(math.exp(-beta * (t - s)) for s in times[:i]) for i, t in enumerate(times)]
    compensator = mu * (end - start) + alpha / beta * sum(1 - math.exp(-beta * (end - t)) for t in times)
    return sum(math.log(x) for x in intensity) - compensator, intensity


def test_the_hawkes_likelihood_is_the_intensity_summed_event_by_event():
    times = qf.hawkes_simulate(mu=0.8, alpha=1.2, beta=3.0, start=0.0, end=40.0, rng=76)
    found = qf.hawkes_fit(times=times, start=0.0, end=40.0, initial=(0.5, 0.5, 2.0))
    loglik, intensity = direct_loglik(list(times), 0.0, 40.0, found["mu"], found["alpha"], found["beta"])
    assert found["loglik"] == pytest.approx(loglik, rel=1e-12)
    assert found["intensity"] == pytest.approx(intensity, rel=1e-12)
    assert found["branching_ratio"] == pytest.approx(found["alpha"] / found["beta"]) and found["branching_ratio"] < 1
    assert found["mean_intensity"] == pytest.approx(found["mu"] / (1 - found["branching_ratio"]))
    for mu, alpha, beta in ((found["mu"] * 1.1, found["alpha"], found["beta"]),
                            (found["mu"], found["alpha"] * 0.9, found["beta"] * 1.05)):
        assert direct_loglik(list(times), 0.0, 40.0, mu, alpha, beta)[0] < found["loglik"]
    integrated = sum(found["residuals"]) + found["mu"] * times[0]
    assert integrated == pytest.approx(found["mu"] * (times[-1]) + found["alpha"] / found["beta"] * sum(
        1 - math.exp(-found["beta"] * (times[-1] - t)) for t in times[:-1]), rel=1e-10)


def test_the_fit_does_not_depend_on_where_the_clock_starts():
    """Moving every time and the window by a constant moves nothing: the same parameters and log-likelihood."""
    times = qf.hawkes_simulate(mu=0.8, alpha=1.2, beta=3.0, start=0.0, end=40.0, rng=78)
    here = qf.hawkes_fit(times=times, start=0.0, end=40.0, initial=(0.5, 0.5, 2.0))
    later = qf.hawkes_fit(times=times + 100.0, start=100.0, end=140.0, initial=(0.5, 0.5, 2.0))
    assert later["loglik"] == pytest.approx(here["loglik"], rel=1e-8)
    assert (later["mu"], later["alpha"], later["beta"]) == pytest.approx((here["mu"], here["alpha"], here["beta"]),
                                                                         rel=1e-5)
    window = qf.hawkes_fit(times=times + 100.0, start=95.0, end=140.0, initial=(0.5, 0.5, 2.0))
    assert window["mu"] < here["mu"]           # five more quiet time units lower the baseline


@pytest.mark.parametrize("times", [[1.0, 1.0], [1.0, 2.0, 2.0]])
def test_the_hawkes_fit_refuses_simultaneous_events(times):
    """A simple Hawkes process has strictly ordered event times; a duplicate has no valid likelihood here."""
    with pytest.raises(ValueError, match="increasing event times"):
        qf.hawkes_fit(times=times, start=0.0, end=5.0, initial=(0.5, 0.8, 2.0))


@pytest.mark.parametrize("start, end", [(float("nan"), 5.0), (float("-inf"), 5.0),
                                        (0.0, float("nan")), (0.0, float("inf"))])
def test_both_hawkes_functions_refuse_nonfinite_observation_windows(start, end):
    with pytest.raises(ValueError, match="start, end"):
        qf.hawkes_fit(times=[1.0, 2.0], start=start, end=end, initial=(0.5, 0.8, 2.0))
    with pytest.raises(ValueError, match="start and end must be finite"):
        qf.hawkes_simulate(mu=0.5, alpha=0.8, beta=2.0, start=start, end=end, rng=7)


@pytest.mark.parametrize("initial", [(float("inf"), 0.8, 2.0), (0.5, float("nan"), 2.0),
                                     (0.5, 0.8, float("inf"))])
def test_the_hawkes_fit_refuses_nonfinite_initial_parameters(initial):
    with pytest.raises(ValueError, match="initial"):
        qf.hawkes_fit(times=[1.0, 2.0], start=0.0, end=5.0, initial=initial)


@pytest.mark.parametrize("mu, alpha, beta", [(float("inf"), 0.8, 2.0), (0.5, float("nan"), 2.0),
                                           (0.5, 0.8, float("inf"))])
def test_the_hawkes_simulator_refuses_nonfinite_parameters(mu, alpha, beta):
    with pytest.raises(ValueError, match="mu must be positive"):
        qf.hawkes_simulate(mu=mu, alpha=alpha, beta=beta, start=0.0, end=5.0, rng=7)


def test_the_simulated_count_has_the_hawkes_mean_from_an_empty_start():
    mu, alpha, beta, horizon = 0.5, 0.8, 2.0, 30.0
    counts = np.array([qf.hawkes_simulate(mu=mu, alpha=alpha, beta=beta, start=0.0, end=horizon, rng=seed).size
                       for seed in range(400)])
    n = alpha / beta
    expected = mu * horizon / (1 - n) - mu * n * (1 - math.exp(-(beta - alpha) * horizon)) / ((beta - alpha) * (1 - n))
    assert counts.mean() == pytest.approx(expected, abs=4 * counts.std() / math.sqrt(counts.size))
    again = qf.hawkes_simulate(mu=mu, alpha=alpha, beta=beta, start=0.0, end=horizon, rng=3)
    assert np.array_equal(again, qf.hawkes_simulate(mu=mu, alpha=alpha, beta=beta, start=0.0, end=horizon, rng=3))
    assert np.all(np.diff(again) > 0) and again[0] > 0 and again[-1] < horizon
    with pytest.raises(ValueError, match="alpha < beta"):
        qf.hawkes_simulate(mu=mu, alpha=2.0, beta=2.0, start=0.0, end=horizon, rng=3)


@pytest.mark.slow
def test_a_long_simulated_sample_returns_its_parameters():
    times = qf.hawkes_simulate(mu=0.5, alpha=0.8, beta=2.0, start=0.0, end=6000.0, rng=77)
    found = qf.hawkes_fit(times=times, start=0.0, end=6000.0, initial=(1.0, 0.5, 1.0))
    assert found["mu"] == pytest.approx(0.5, rel=0.1) and found["branching_ratio"] == pytest.approx(0.4, abs=0.04)
    assert found["beta"] == pytest.approx(2.0, rel=0.15) and np.mean(found["residuals"]) == pytest.approx(1.0, abs=0.03)
