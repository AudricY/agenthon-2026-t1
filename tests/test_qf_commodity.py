"""qf.commodity (runs/lanes/lib-montecarlo1-prompt.md; docs/research/library-syllabus-1011.md ranks 43, 99, 105, 110,
122): degree days by Hull's worked examples ("Weather derivatives": a day of 68 and 44 is 9 HDD; a cumulative-HDD
call struck at 700 paying 10,000 a degree day pays 1.2 million at 820 and caps at 1.5 million, a 700-850 call
spread), its identities and burn pricing; Schwartz and Smith's (2000) futures price against the mean and variance
of ln S_T integrated here, and its limits; storage worked by hand (Eydeland and Wolyniec's two-period arbitrage);
CAR(1) as the Ornstein-Uhlenbeck process (qf.ou_jump_moments) and CAR(3)'s variance by quadrature; the Samuelson
fit returning the parameters a curve was built from (Clewlow and Strickland). No unit's values."""
import math

import numpy as np
import pytest
from scipy import integrate
from scipy.linalg import expm

import qf

pytestmark = pytest.mark.fast


def test_hulls_daily_degree_days_from_the_maximum_and_minimum():
    """Hull: a maximum of 68 and a minimum of 44 average 56 F; HDD = max(65 - 56, 0) = 9, CDD = 0."""
    hdd = qf.degree_days([[68.0, 44.0]], "max_min", 65.0, "hdd", "swap", 0.0, 1.0, None, 1.0, 1)
    cdd = qf.degree_days([[68.0, 44.0]], "max_min", 65.0, "cdd", "swap", 0.0, 1.0, None, 1.0, 1)
    assert hdd["index"] == 9.0 and cdd["index"] == 0.0 and hdd["payoff"] == 9.0 and math.isnan(hdd["std_error"])


def test_hulls_cumulative_hdd_call_and_its_cap():
    """Hull: a call on the month's cumulative HDD struck at 700 paying 10,000 a degree day pays 1.2 million at 820;
    capped at 1.5 million it is a 700 call less an 850 call."""
    days = np.full(30, 65.0 - 820.0 / 30)
    found = qf.degree_days(days, "given", 65.0, "hdd", "call", 700.0, 10_000.0, 1.5e6, 1.0, 1)
    assert found["index"] == pytest.approx(820.0) and found["payoff"] == pytest.approx(1.2e6)
    for total in (650.0, 760.0, 900.0, 1_200.0):
        each = np.full(30, 65.0 - total / 30)
        capped = qf.degree_days(each, "given", 65.0, "hdd", "call", 700.0, 10_000.0, 1.5e6, 1.0, 1)["payoff"]
        spread = 10_000.0 * (max(total - 700.0, 0) - max(total - 850.0, 0))
        assert capped == pytest.approx(spread)


def test_a_day_at_the_reference_adds_nothing_and_hdd_less_cdd_is_days_times_the_reference_less_cat():
    temps = np.array([[60.0, 65.0, 72.0, 50.0, 65.0], [66.0, 64.0, 65.0, 80.0, 41.0]])
    hdd, cdd, cat = (qf.degree_days(temps, "given", 65.0, index, "swap", 0.0, 1.0, None, 1.0, 1)["index"]
                     for index in ("hdd", "cdd", "cat"))
    assert np.allclose(hdd - cdd, 5 * 65.0 - cat) and np.allclose(hdd, [20.0, 25.0]) and np.allclose(cdd, [7.0, 16.0])
    at = qf.degree_days(np.full(10, 18.0), "given", 18.0, "hdd", "swap", 0.0, 1.0, None, 1.0, 1)
    assert at["index"] == 0.0


def test_burn_pricing_is_the_discounted_mean_payoff_over_the_years():
    years = np.array([[60.0, 50.0, 66.0], [62.0, 55.0, 70.0], [58.0, 45.0, 64.0]])       # HDD 20, 13, 28
    found = qf.degree_days(years, "given", 65.0, "hdd", "put", 25.0, 20.0, None, 0.98, 1)
    assert np.allclose(found["payoff"], [100.0, 240.0, 0.0])
    assert found["price"] == pytest.approx(0.98 * 340.0 / 3)
    assert found["std_error"] == pytest.approx(0.98 * np.std([100.0, 240.0, 0.0], ddof=1) / math.sqrt(3))
    swap = qf.degree_days(years, "given", 65.0, "hdd", "swap", 25.0, 20.0, 150.0, 1.0, 0)
    assert np.allclose(swap["payoff"], [-100.0, -150.0, 60.0])


SS = dict(kappa=1.5, sigma_chi=0.3, sigma_xi=0.15, rho=0.3, mu_xi=0.01, lambda_chi=0.1, lambda_xi=0.02)


def test_schwartz_smith_futures_are_the_pricing_measures_mean_of_the_spot():
    """F(0, T) = E*[S_T] = exp(E*[ln S_T] + Var[ln S_T] / 2), E*[ln S_T] = e^{-kappa T} chi - (1 - e^{-kappa T})
    lambda_chi / kappa + xi + (mu_xi - lambda_xi) T and the variance the integral of the two factors' instantaneous
    variance: Schwartz and Smith (2000), equation (9), checked by quadrature."""
    chi, xi, k = 0.1, 4.0, SS["kappa"]
    for t in (0.25, 1.0, 3.0):
        found = qf.schwartz_smith(chi, xi, [t], **SS, premia="subtracted")
        mean = math.exp(-k * t) * chi - SS["lambda_chi"] / k * (1 - math.exp(-k * t)) + xi + (0.01 - 0.02) * t
        var = integrate.quad(lambda s: (0.3 * math.exp(-k * (t - s))) ** 2 + 0.15 ** 2
                             + 2 * 0.3 * 0.3 * 0.15 * math.exp(-k * (t - s)), 0.0, t)[0]
        assert found["forwards"] == pytest.approx(math.exp(mean + var / 2), rel=1e-12)
        assert found["variance"] == pytest.approx(var, rel=1e-10)
    added = qf.schwartz_smith(chi, xi, [1.0], **SS, premia="added")
    flipped = qf.schwartz_smith(chi, xi, [1.0], **{**SS, "lambda_chi": -0.1, "lambda_xi": -0.02}, premia="subtracted")
    assert added["forwards"] == pytest.approx(flipped["forwards"], rel=1e-14)


def test_schwartz_smith_limits_the_spot_today_the_one_factor_forward_and_the_samuelson_vol():
    today = qf.schwartz_smith(0.2, 3.5, [0.0], **SS, premia="subtracted")
    assert today["forwards"] == pytest.approx(math.exp(3.7)) == today["spot"]
    fast = qf.schwartz_smith(0.3, 4.0, [1.0, 3.0], **{**SS, "kappa": 1e7}, premia="subtracted")
    assert np.allclose(fast["forwards"], np.exp(4.0 + (0.01 - 0.02 + 0.15 ** 2 / 2) * np.array([1.0, 3.0])), rtol=1e-6)
    vol = qf.schwartz_smith(0.0, 4.0, [0.0, 50.0], **SS, premia="subtracted")["vol"]
    assert vol[0] == pytest.approx(math.sqrt(0.09 + 0.0225 + 2 * 0.3 * 0.3 * 0.15)) and vol[1] == pytest.approx(0.15)


def test_storage_buys_cheap_and_sells_dear_as_worked_by_hand():
    """Two periods at 3 and 5, a capacity of 10 at rates of 10, costs of 0.1 each way: inject 10 at 3.1, withdraw 10
    at 4.9: worth 18 undiscounted. This independent example uses Eydeland and Wolyniec's storage-arbitrage model."""
    found = qf.storage_intrinsic([3.0, 5.0], [1.0, 1.0], 10.0, 10.0, 10.0, 0.1, 0.1, 0.0, 0.0)
    assert found["value"] == pytest.approx(18.0) and np.allclose(found["injections"], [10.0, 0.0])
    assert np.allclose(found["withdrawals"], [0.0, 10.0]) and np.allclose(found["inventory"], [10.0, 0.0])
    discounted = qf.storage_intrinsic([3.0, 5.0], [0.99, 0.97], 10.0, 10.0, 10.0, 0.1, 0.1, 0.0, 0.0)
    assert discounted["value"] == pytest.approx(10 * (4.9 * 0.97 - 3.1 * 0.99))


def test_storage_is_worth_nothing_on_a_flat_curve_and_meets_its_limits():
    assert qf.storage_intrinsic([3.0] * 4, [1.0] * 4, 10.0, 5.0, 5.0, 0.05, 0.05, 0.0, 0.0)["value"] == 0.0
    slow = qf.storage_intrinsic([2.0, 2.0, 6.0], [1.0] * 3, 100.0, 5.0, 20.0, 0.0, 0.0, 0.0, 0.0)
    assert slow["value"] == pytest.approx(40.0) and np.all(slow["inventory"] <= 100.0)
    full = qf.storage_intrinsic([4.0, 3.0], [1.0, 1.0], 10.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0)
    assert full["value"] == pytest.approx(10.0) and full["inventory"][-1] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="no schedule"):
        qf.storage_intrinsic([3.0, 5.0], [1.0, 1.0], 10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0)


def test_storage_pays_for_the_inventory_required_at_the_end():
    """An independent two-period cash balance in Eydeland and Wolyniec's storage model: buy ten at 3.1,
    sell six at 4.9 and retain four, so cash value is 6*4.9-10*3.1=-1.6."""
    found = qf.storage_intrinsic([3.0, 5.0], [1.0, 1.0], 10.0, 10.0, 10.0, 0.1, 0.1, 0.0, 4.0)
    assert found["value"] == pytest.approx(-1.6)
    assert np.allclose(found["injections"], [10.0, 0.0])
    assert np.allclose(found["withdrawals"], [0.0, 6.0])
    assert np.allclose(found["inventory"], [10.0, 4.0])


def test_car_one_is_the_ornstein_uhlenbeck_process():
    """Benth, Saltyte Benth and Koekebakker (2008): CAR(1) is dX = -alpha X dt + sigma dB, mean X e^{-alpha h} and
    variance sigma^2 (1 - e^{-2 alpha h}) / (2 alpha), qf.ou_jump_moments without jumps."""
    found = qf.car_temperature([0.2], "car", [3.0], [10.0, 12.0], [1.0, 5.0], 2.0)
    for i, h in enumerate((1.0, 5.0)):
        mean, var = qf.ou_jump_moments(3.0, 0.2, 0.0, 2.0, 0.0, 0.0, 0.0, h)
        assert found["expected"][i] == pytest.approx([10.0, 12.0][i] + mean)
        assert found["variance"][i] == pytest.approx(var)
    assert found["stationary"] and np.allclose(found["alphas"], [0.2])


def test_car_three_from_an_ar_three_by_the_euler_map_and_its_variance_by_quadrature():
    """The AR(3) x_{t+3} = b1 x_{t+2} + b2 x_{t+1} + b3 x_t is the CAR(3)'s Euler step of one unit: I + A has the
    AR's characteristic polynomial; the variance sigma^2 int_0^h (e_1' e^{A u} e_3)^2 du by quadrature."""
    b = [0.95, -0.25, 0.08]
    found = qf.car_temperature(b, "ar", [2.0, 0.5, -0.1], [0.0, 0.0], [1.0, 5.0], 3.0)
    assert np.allclose(found["alphas"], [3 - 0.95, 3 - 2 * 0.95 + 0.25, 1 - 0.95 + 0.25 - 0.08])
    a = np.zeros((3, 3))
    a[0, 1] = a[1, 2] = 1.0
    a[2] = -found["alphas"][::-1]
    assert np.allclose(np.poly(np.eye(3) + a), [1.0, -0.95, 0.25, -0.08])
    for i, h in enumerate((1.0, 5.0)):
        var = 9.0 * integrate.quad(lambda u: expm(a * u)[0, 2] ** 2, 0.0, h)[0]
        assert found["variance"][i] == pytest.approx(var, rel=1e-8)
        assert found["expected"][i] == pytest.approx(expm(a * h)[0] @ [2.0, 0.5, -0.1])
    assert found["stationary"]


def test_car_stationary_variance_stays_finite_at_long_weather_horizons():
    """The covariance semigroup reaches its Lyapunov limit; CAR(3)'s stationary P11 is
    sigma^2 alpha1 / (2 alpha3 (alpha1 alpha2 - alpha3))."""
    one = qf.car_temperature([0.2], "car", [3.0], [10.0], [5_000.0], 2.0)
    assert one["expected"][0] == pytest.approx(10.0)
    assert one["variance"][0] == pytest.approx(10.0, rel=1e-12)
    alphas = [2.05, 1.35, 0.22]
    three = qf.car_temperature(alphas, "car", [2.0, 0.5, -0.1], [0.0, 0.0], [100.0, 365.0], 3.0)
    limit = 9.0 * alphas[0] / (2.0 * alphas[2] * (alphas[0] * alphas[1] - alphas[2]))
    assert np.allclose(three["variance"], limit, rtol=1e-10)
    assert np.all(np.abs(three["expected"]) < 1e-8)


def test_car_zero_reversion_is_brownian_at_short_and_long_horizons():
    """At alpha zero CAR(1) is X(t+h)=X(t)+sigma(B(t+h)-B(t)), with variance sigma^2 h."""
    horizons = [0.0, 1e-9, 10_000.0]
    found = qf.car_temperature([0.0], "car", [3.0], [10.0] * 3, horizons, 2.0)
    assert np.allclose(found["expected"], 13.0)
    assert np.allclose(found["variance"], 4.0 * np.array(horizons), rtol=1e-12, atol=0.0)
    assert not found["stationary"]


def test_samuelson_fit_returns_the_parameters_its_curve_was_built_from():
    tau = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0])
    two = qf.samuelson_vol_fit(tau, 0.25 * np.sqrt(1 + 2.0 * np.exp(-3.0 * tau)), "two_factor")
    assert (two["sigma"], two["a"], two["kappa"]) == pytest.approx((0.25, 2.0, 1.5), rel=1e-6) and two["rmse"] < 1e-10
    one = qf.samuelson_vol_fit(tau, 0.4 * np.exp(-0.7 * tau), "one_factor")
    assert (one["sigma"], one["kappa"]) == pytest.approx((0.4, 0.7), rel=1e-6) and math.isnan(one["a"])
    noisy = 0.3 * np.sqrt(1 + 1.0 * np.exp(-2.0 * tau)) * (1 + 0.01 * np.random.default_rng(1).standard_normal(7))
    fit = qf.samuelson_vol_fit(tau, noisy, "two_factor")
    assert fit["sigma"] == pytest.approx(0.3, rel=0.05) and fit["rmse"] < 0.01


@pytest.mark.parametrize("form", ["one_factor", "two_factor"])
def test_samuelson_fit_keeps_nonnegative_factor_variance_and_reversion(form):
    """An increasing observed curve cannot be a mean-reverting Samuelson model: its closest nonincreasing fit is
    flat, rather than a negative speed or a negative factor variance."""
    tau = np.array([0.1, 0.3, 0.6, 1.0, 2.0, 3.0])
    vols = 0.2 * np.exp(0.3 * tau) if form == "one_factor" else 0.4 * np.sqrt(1.0 - 0.6 * np.exp(-tau))
    fit = qf.samuelson_vol_fit(tau, vols, form)
    assert fit["kappa"] >= 0.0
    if form == "two_factor":
        assert fit["a"] >= 0.0
    assert np.allclose(fit["fitted"], vols.mean(), rtol=1e-7)
    assert fit["rmse"] == pytest.approx(np.std(vols), rel=1e-7)
