"""Lane lib-econometrics1 (runs/lanes/lib-econometrics1-prompt.md): the timeseries shelf, qf.arma_fit,
qf.kalman_filter and qf.seasonal_decompose. Against statsmodels 0.15.0's ARIMA, acf, pacf, UnobservedComponents and
seasonal_decompose on this file's own series (frozen in `STATSMODELS`; `reference_values()` recomputes them where
statsmodels imports, the slow test compares), and exact identities: the exact ARMA likelihood is the multivariate
normal density of the data under the model's autocovariances (Hamilton 1994, 5.2); an MA(1)'s rho_1 = theta / (1 +
theta^2) and an ARMA(1,1)'s psi_j = (phi + theta) phi^(j-1); the Kalman likelihood is the data's joint normal density
and the smoothed states the normal conditional means (Durbin and Koopman 2012, 4.3); a local level's gain settles at
its closed form (Harvey 1989, 3.3); a linear trend plus a fixed pattern decomposes exactly (Hyndman and
Athanasopoulos 2021, 3.4). No unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import linalg, stats

from test_qf_bs import check_cards  # noqa: F401  (first: it puts agent/ on the path outside pytest too)
import qf

pytestmark = pytest.mark.fast


def arma_series(n, phi, theta, mean, seed):
    """An ARMA(1,1) path after a burn-in of 200, the legacy generator (a frozen stream)."""
    e = np.random.RandomState(seed).normal(size=n + 200)
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t] + theta * e[t - 1]
    return mean + x[200:]


def seasonal_series():
    rng = np.random.RandomState(51)
    return 20.0 + 0.1 * np.arange(36) + np.tile([1.5, -0.5, 2.0, -3.0], 9) + 0.4 * rng.normal(size=36)


def level_series():
    rng = np.random.RandomState(52)
    return np.cumsum(0.3 * rng.normal(size=40)) + rng.normal(size=40)


SERIES = {(1, 0): arma_series(200, 0.6, 0.0, 1.0, 53), (0, 1): arma_series(200, 0.0, 0.5, 1.0, 54),
          (1, 1): arma_series(200, 0.7, -0.3, 1.0, 55)}
SEASONAL, LEVEL = seasonal_series(), level_series()
STATSMODELS = {   # statsmodels 0.15.0 on SERIES, LEVEL and SEASONAL (reference_values)
    ('arma', 1, 0): {'mean': 0.8381490715805259, 'ar': [0.7326022207199528], 'ma': [], 'sigma2': 1.0278522944692887, 'loglik': -286.9213162069134, 'aic': 579.8426324138268, 'forecast': [-0.07401989779688278, 0.16989205894280568, 0.34858250011045044], 'forecast_var': [1.0278522944692887, 1.579506802212997, 1.8755830940610236]},
    ('arma', 0, 1): {'mean': 0.8730473176251163, 'ar': [], 'ma': [0.4748789392006936], 'sigma2': 1.0921825609535516, 'loglik': -292.73429147535376, 'aic': 591.4685829507075, 'forecast': [0.8520528935250966, 0.8730473176251163, 0.8730473176251163], 'forecast_var': [1.0921825609535516, 1.3384806578062887, 1.3384806578062887]},
    ('arma', 1, 1): {'mean': 0.8915170668162605, 'ar': [0.631577609899659], 'ma': [-0.21150040645998575], 'sigma2': 0.8405338573763625, 'loglik': -266.5509816834476, 'aic': 541.1019633668952, 'forecast': [1.9744160198999832, 1.575451799367721, 1.3234749305284745], 'forecast_var': [0.8405338573763625, 0.9888585441956057, 1.0480238196553096]},
    'acf': {'n': [1.0, 0.4581663702480529, 0.2934299381981219, 0.12238284202561342, 0.11284313766163187, 0.06077131637790411, 0.0020335484103420013], 'n-k': [1.0, 0.4604687138171386, 0.2963938769677999, 0.12424654012752628, 0.11514605883839986, 0.06232955525938883, 0.0020964416601463934], 'pacf n': [1.0, 0.4581663702480529, 0.10570212795789204, -0.05923216106570077, 0.0654228617909837, -0.007846302225697231, -0.055796671652987455], 'pacf n-k': [1.0, 0.4604687138171386, 0.10706320589546704, -0.06023703994612208, 0.06690005727127447, -0.00799337271970115, -0.057747319781220934]},
    'local level': {'llf_obs': [-1.786765975382373, -1.254068344408401, -1.2539036797633676, -1.342470076522846, -1.1102373767080302, -1.4884419001787572, -1.0858425608267006, -3.2825599839598754, -1.1115722840054247, -3.5888372459679623, -1.633102375375176, -2.4153470506041335, -1.1973230051880885, -1.1120361343817993, -1.1066955511170524, -1.0692858083433, -1.216528359539509, -1.1566247720654563, -1.4128813982157111, -1.2567235959430865, -1.0840482988082711, -1.4356039785179937, -1.172194498185452, -1.6890584720262116, -1.7223430665008572, -1.0686013354325499, -2.1612398798137398, -1.1391147540445212, -1.1199813872076474, -1.4059603028295786, -2.7189052864527605, -1.0692079556472933, -1.1098218225901277, -1.0683840141477778, -1.4062217744426977, -1.2051055584511459, -1.2220983768883036, -1.079453626770575, -1.7207051315949615, -1.8352391784625008], 'filtered': [0.6355267973485934, 0.7543395765798782, 0.5415412415991696, 0.7990397293294589, 0.7264480117719418, 0.43636465985159956, 0.48458300818007144, -0.15893155690288413, -0.24637652596386067, 0.43111217769696225, 0.7508517640920269, 1.2440920474943278, 1.3965029658665202, 1.3078927798706024, 1.390914476278102, 1.403498943129556, 1.2401929787887296, 1.3662247828811296, 1.1172014070520524, 0.9330759769835613, 0.9861779562579593, 0.7290777747097134, 0.5923797899501059, 0.9266283514936664, 0.5835347578785575, 0.577247344296451, 1.020772502098693, 1.1336084802280795, 1.2299824991257917, 0.9834786238011366, 1.5285421668571897, 1.540737825873033, 1.454370989757543, 1.4537191904204463, 1.7003185050454814, 1.543441792512737, 1.3771017485371297, 1.4217442751720832, 1.764407705229711, 1.3928779557089292], 'smoothed': [0.6046622315295993, 0.6011899678749625, 0.5719194640090005, 0.5795278891140018, 0.5158053377633234, 0.4494567528608684, 0.45377103132753155, 0.44336101147640605, 0.6496676311751839, 0.9589290017803185, 1.141861542602972, 1.2776904475498787, 1.2893766019600763, 1.2520901470213543, 1.2326600315134593, 1.1775452743670776, 1.0988440312292553, 1.0496080730740949, 0.9393173870555518, 0.8773518385735324, 0.8579402391304598, 0.8132681498024383, 0.8425962763691024, 0.9297605596772096, 0.9308516826078378, 1.051841617016394, 1.217169427631014, 1.2855855059741876, 1.3385276434007545, 1.3763400179832026, 1.5131957262093774, 1.50784969815547, 1.4963929142092551, 1.5110315131508212, 1.5309966175544392, 1.4720122839051972, 1.4471293698669223, 1.4715239278479781, 1.4888649798190383, 1.3928779557089292]},
    ('decompose', 'additive', 4): {'trend': [math.nan, math.nan, 20.190674805322416, 20.302190650597726, 20.313833452437283, 20.347098755991414, 20.43816777115509, 20.647439794948408, 20.871788292740383, 20.98952973164058, 21.068638141724673, 21.030453212168045, 21.035737600173157, 21.15677821631448, 21.284741735815267, 21.301279425045895, 21.306953196492298, 21.401635364182347, 21.603397806941754, 21.90011478736177, 22.21666276561372, 22.46724339191695, 22.647786666464825, 22.806404231380952, 22.847312463973097, 22.868663598637923, 22.79917993697835, 22.634642304192823, 22.52806974748469, 22.57946672563279, 22.818828995847586, 23.09457164005092, 23.414587474878946, 23.5699154661863, math.nan, math.nan], 'seasonal': [1.4308895796131114, -0.32825882810275997, 1.9697285479074598, -3.0723592994178115]},
    ('decompose', 'additive', 3): {'trend': [math.nan, 21.24298933286903, 19.700271093165895, 20.4653117378654, 19.55880690774636, 21.459428278528804, 19.900962682623437, 20.627225803227493, 20.286509755747264, 22.13089752852516, 20.69979199392172, 20.951592083332578, 20.385249927449273, 22.155378339576686, 20.91035138177634, 21.419135425203876, 20.625818118571942, 22.31527676226154, 21.062169301403713, 22.072966523965462, 21.553751262128944, 23.560750022521532, 22.120284267744243, 23.008643235233077, 22.190718459573944, 23.93730035055959, 22.332739092167785, 22.77692362573593, 21.855317642990364, 23.364450897401216, 22.31968664840467, 23.281140580578416, 22.86177240165782, 24.584984803390178, 23.102049996652273, math.nan], 'seasonal': [-0.2292228132370599, 0.003969227958051685, 0.22525358527900824]},
    ('decompose', 'multiplicative', 4): {'trend': [math.nan, math.nan, 20.190674805322416, 20.302190650597726, 20.313833452437283, 20.347098755991414, 20.43816777115509, 20.647439794948408, 20.871788292740383, 20.98952973164058, 21.068638141724673, 21.030453212168045, 21.035737600173157, 21.15677821631448, 21.284741735815267, 21.301279425045895, 21.306953196492298, 21.401635364182347, 21.603397806941754, 21.90011478736177, 22.21666276561372, 22.46724339191695, 22.647786666464825, 22.806404231380952, 22.847312463973097, 22.868663598637923, 22.79917993697835, 22.634642304192823, 22.52806974748469, 22.57946672563279, 22.818828995847586, 23.09457164005092, 23.414587474878946, 23.5699154661863, math.nan, math.nan], 'seasonal': [1.0655222512881752, 0.9852679226768396, 1.0915180707656118, 0.8576917552693735]},
    ('decompose', 'multiplicative', 3): {'trend': [math.nan, 21.24298933286903, 19.700271093165895, 20.4653117378654, 19.55880690774636, 21.459428278528804, 19.900962682623437, 20.627225803227493, 20.286509755747264, 22.13089752852516, 20.69979199392172, 20.951592083332578, 20.385249927449273, 22.155378339576686, 20.91035138177634, 21.419135425203876, 20.625818118571942, 22.31527676226154, 21.062169301403713, 22.072966523965462, 21.553751262128944, 23.560750022521532, 22.120284267744243, 23.008643235233077, 22.190718459573944, 23.93730035055959, 22.332739092167785, 22.77692362573593, 21.855317642990364, 23.364450897401216, 22.31968664840467, 23.281140580578416, 22.86177240165782, 24.584984803390178, 23.102049996652273, math.nan], 'seasonal': [0.9895343259087686, 0.9997559125446348, 1.0107097615465963]},
}


def reference_values():
    """STATSMODELS again, from statsmodels (the lane's scratch/lib-econometrics1/refenv)."""
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf, pacf
    found = {}
    for (p, q), x in SERIES.items():
        fit = ARIMA(x, order=(p, 0, q), trend="c").fit()
        forecast = fit.get_forecast(3)
        found[("arma", p, q)] = {"mean": float(fit.params[0]), "ar": fit.arparams.tolist(),
                                 "ma": fit.maparams.tolist(), "sigma2": float(fit.params[-1]),
                                 "loglik": float(fit.llf), "aic": float(fit.aic),
                                 "forecast": forecast.predicted_mean.tolist(),
                                 "forecast_var": forecast.var_pred_mean.tolist()}
    x = SERIES[(1, 1)]
    found["acf"] = {"n": acf(x, nlags=6).tolist(), "n-k": acf(x, nlags=6, adjusted=True).tolist(),
                    "pacf n": pacf(x, nlags=6, method="ywm").tolist(),
                    "pacf n-k": pacf(x, nlags=6, method="ywadjusted").tolist()}
    model = sm.tsa.UnobservedComponents(LEVEL, level="local level")
    model.initialize_known(np.array([0.0]), np.array([[4.0]]))
    smoothed = model.smooth([1.0, 0.09])
    found["local level"] = {"llf_obs": smoothed.llf_obs.tolist(), "filtered": smoothed.filtered_state[0].tolist(),
                            "smoothed": smoothed.smoothed_state[0].tolist()}
    for model_name in ("additive", "multiplicative"):
        for period in (4, 3):
            result = seasonal_decompose(SEASONAL, model=model_name, period=period)
            found[("decompose", model_name, period)] = {"trend": result.trend.tolist(),
                                                        "seasonal": result.seasonal[:period].tolist()}
    return found


@pytest.mark.slow
def test_the_frozen_values_are_statsmodels_own():
    pytest.importorskip("statsmodels")
    found = reference_values()
    for key, value in STATSMODELS.items():
        for part, numbers in value.items():
            assert np.asarray(found[key][part]) == pytest.approx(np.asarray(numbers), rel=1e-10, abs=1e-12,
                                                                 nan_ok=True), (key, part)


# ------------------------------------------------------------------------------------------------ ARMA


@pytest.mark.parametrize("order", [(1, 0), (0, 1), (1, 1)])
def test_the_exact_arma_fit_is_statsmodels_arima(order):
    """statsmodels' ARIMA(order=(p, 0, q), trend='c') maximises the same exact likelihood (its llf equals this one at
    either optimum to rounding); the two optimisers agree to their tolerances."""
    found = qf.arma_fit(series=SERIES[order], order=order, likelihood="exact", mean="constant", acf_lags=2,
                        acf_denominator="n", horizon=3)
    want = STATSMODELS[("arma",) + order]
    assert found["loglik"] == pytest.approx(want["loglik"], abs=1e-5)
    assert found["aic"] == pytest.approx(want["aic"], abs=1e-4)
    assert found["mean"] == pytest.approx(want["mean"], abs=5e-4)
    assert found["ar"] == pytest.approx(want["ar"], abs=5e-4) and found["ma"] == pytest.approx(want["ma"], abs=5e-4)
    assert found["sigma2"] == pytest.approx(want["sigma2"], rel=1e-4)
    assert found["forecast"] == pytest.approx(want["forecast"], abs=5e-4)
    assert found["forecast_var"] == pytest.approx(want["forecast_var"], rel=1e-4)


def autocovariances(phi, theta, sigma2, lags):
    """An ARMA(1,1)'s autocovariances gamma_0 .. gamma_lags from its psi weights (Hamilton 1994, 3.6)."""
    psi = np.r_[1.0, (phi + theta) * phi ** np.arange(4000)]
    return np.array([sigma2 * psi[:psi.size - k] @ psi[k:] for k in range(lags + 1)])


def test_the_exact_likelihood_is_the_multivariate_normal_density_of_the_data():
    x = SERIES[(1, 1)][:40]
    fit = qf.arma_fit(series=x, order=(1, 1), likelihood="exact", mean="constant", acf_lags=1,
                      acf_denominator="n", horizon=1)
    gamma = autocovariances(fit["ar"][0], fit["ma"][0], fit["sigma2"], 39)
    density = stats.multivariate_normal(np.full(40, fit["mean"]), linalg.toeplitz(gamma)).logpdf(x)
    assert fit["loglik"] == pytest.approx(density, rel=1e-10)


def test_an_arma_11s_psi_weights_and_its_forecast_variance():
    found = qf.arma_fit(series=SERIES[(1, 1)], order=(1, 1), likelihood="conditional", mean="constant",
                        acf_lags=1, acf_denominator="n", horizon=6)
    phi, theta = found["ar"][0], found["ma"][0]
    assert found["psi"] == pytest.approx(np.r_[1.0, (phi + theta) * phi ** np.arange(5)], rel=1e-12)
    assert found["forecast_var"] == pytest.approx(found["sigma2"] * np.cumsum(found["psi"] ** 2), rel=1e-12)
    far = qf.arma_fit(series=SERIES[(1, 1)], order=(1, 1), likelihood="conditional", mean="constant",
                      acf_lags=1, acf_denominator="n", horizon=200)
    assert far["forecast"][-1] == pytest.approx(far["mean"], abs=1e-9)
    assert far["forecast_var"][-1] == pytest.approx(autocovariances(phi, theta, far["sigma2"], 0)[0], rel=1e-9)


def test_a_short_exact_ma1_forecast_conditions_on_the_finite_observation_history():
    """Hamilton (1994), 3.6: gamma_0 = sigma2 (1 + theta^2), gamma_1 = sigma2 theta. The future given the
    observed sample is a block-normal conditional, including uncertainty about the presample shock."""
    x = np.array([0.5, 0.8, 0.3, -0.2, 0.1, 0.6])
    horizon = 3
    found = qf.arma_fit(series=x, order=(0, 1), likelihood="exact", mean="zero", acf_lags=1,
                        acf_denominator="n", horizon=horizon)
    theta, sigma2 = found["ma"][0], found["sigma2"]
    gamma = np.r_[sigma2 * (1 + theta * theta), sigma2 * theta, np.zeros(x.size + horizon - 2)]
    joint_cov = linalg.toeplitz(gamma)
    observed = joint_cov[:x.size, :x.size]
    cross = joint_cov[x.size:, :x.size]
    mean = cross @ np.linalg.solve(observed, x)
    covariance = joint_cov[x.size:, x.size:] - cross @ np.linalg.solve(observed, cross.T)
    assert found["forecast"] == pytest.approx(mean, rel=1e-11, abs=1e-12)
    assert found["forecast_var"] == pytest.approx(np.diag(covariance), rel=1e-11)
    assert found["forecast_var"][0] > sigma2 * 1.1


def test_the_conditional_fit_of_an_ar_is_ols_on_its_lags():
    x = SERIES[(1, 0)]
    for p in (1, 2):
        found = qf.arma_fit(series=x, order=(p, 0), likelihood="conditional", mean="constant", acf_lags=1,
                            acf_denominator="n", horizon=1)
        lags = np.column_stack([x[p - i:x.size - i] for i in range(1, p + 1)])
        plain = qf.ols(y=x[p:], X=lags, intercept=True)
        assert found["ar"] == pytest.approx(plain["coef"][1:], rel=1e-7)
        assert found["mean"] * (1 - found["ar"].sum()) == pytest.approx(plain["coef"][0], rel=1e-7)
        assert found["sigma2"] == pytest.approx(np.mean(plain["resid"] ** 2), rel=1e-9)
        assert found["nobs"] == x.size - p and np.all(np.isnan(found["resid"][:p]))
    ar1 = qf.ar1_fit(x, ddof=0)
    one = qf.arma_fit(series=x, order=(1, 0), likelihood="conditional", mean="constant", acf_lags=1,
                      acf_denominator="n", horizon=1)
    assert one["ar"][0] == pytest.approx(ar1["slope"], rel=1e-7) and one["sigma2"] == pytest.approx(ar1["resid_var"],
                                                                                                rel=1e-9)


def test_the_acf_and_pacf_are_statsmodels_and_an_ma1_has_its_closed_form_rho1():
    x = SERIES[(1, 1)]
    want = STATSMODELS["acf"]
    for denominator in ("n", "n-k"):
        found = qf.arma_fit(series=x, order=(0, 0), likelihood="exact", mean="constant", acf_lags=6,
                            acf_denominator=denominator, horizon=1)
        assert found["acf"] == pytest.approx(want[denominator], rel=1e-12)
        assert found["pacf"] == pytest.approx(want[f"pacf {denominator}"], rel=1e-10)
    e = np.random.RandomState(56).normal(size=40001)
    long = qf.arma_fit(series=e[1:] + 0.6 * e[:-1], order=(0, 0), likelihood="exact", mean="constant", acf_lags=2,
                       acf_denominator="n", horizon=1)
    rho = 0.6 / 1.36                        # Bartlett's standard errors: rho_1's sqrt((1 - 3 rho^2 + 4 rho^4)/n),
    se1, se2 = math.sqrt((1 - 3 * rho ** 2 + 4 * rho ** 4) / 40000), math.sqrt((1 + 2 * rho ** 2) / 40000)  # rho_2's
    assert long["acf"][1] == pytest.approx(rho, abs=4 * se1) and abs(long["acf"][2]) < 4 * se2
    assert long["sigma2"] == pytest.approx(1.36, rel=0.02) and long["mean"] == pytest.approx(np.mean(e[1:] + 0.6 * e[:-1]))


def test_arma_fit_refuses_what_it_cannot_fit():
    with pytest.raises(ValueError, match="order"):
        qf.arma_fit(series=SERIES[(1, 0)], order=(1,), likelihood="exact", mean="constant", acf_lags=2,
                    acf_denominator="n", horizon=1)
    with pytest.raises(ValueError, match="likelihood"):
        qf.arma_fit(series=SERIES[(1, 0)], order=(1, 0), likelihood="css", mean="constant", acf_lags=2,
                    acf_denominator="n", horizon=1)


# ------------------------------------------------------------------------------------------------ Kalman


def local_level(y, start, smoother, initial=(0.0, 4.0)):
    return qf.kalman_filter(observations=y, transition=[[1.0]], observation_matrix=[[1.0]], state_cov=[[0.09]],
                            observation_cov=[[1.0]], initial_state=[initial[0]], initial_cov=[[initial[1]]],
                            start=start, smoother=smoother)


def test_the_local_level_is_statsmodels_unobserved_components():
    """Per observation statsmodels' llf_obs; its llf drops the first observation (loglikelihood_burn 1), so the sum
    is compared in full against the joint normal density below, not against its llf."""
    found = local_level(LEVEL, "first_state", "rts")
    want = STATSMODELS["local level"]
    v, s = found["innovations"][:, 0], found["innovation_cov"][:, 0, 0]
    assert -0.5 * (np.log(2 * np.pi * s) + v * v / s) == pytest.approx(want["llf_obs"], rel=1e-9)
    assert found["filtered_state"][:, 0] == pytest.approx(want["filtered"], rel=1e-8, abs=1e-9)
    assert found["smoothed_state"][:, 0] == pytest.approx(want["smoothed"], rel=1e-8, abs=1e-9)


def joint(transition, observation, q, r, a0, p0, steps):
    """The joint normal of the states x_1 .. x_T and the observations y_1 .. y_T, x_1 ~ N(a0, p0) (first_state)."""
    m, n = len(a0), observation[0].shape[0]
    mean_x, cov_x = [np.asarray(a0, dtype=float)], {}
    big = np.zeros((steps * m, steps * m))
    powers = [np.eye(m)]
    for _ in range(steps):
        powers.append(transition @ powers[-1])
    for i in range(steps):
        for j in range(steps):
            k = min(i, j)
            block = powers[i - k] @ (powers[k] @ p0 @ powers[k].T + sum(powers[k - 1 - l] @ q @ powers[k - 1 - l].T
                                                                        for l in range(k))) @ powers[j - k].T
            big[i * m:(i + 1) * m, j * m:(j + 1) * m] = block
    mean_x = np.concatenate([powers[i] @ a0 for i in range(steps)])
    h = linalg.block_diag(*observation)
    cov_y = h @ big @ h.T + np.kron(np.eye(steps), r)
    return mean_x, big, h @ mean_x, cov_y, big @ h.T


@pytest.mark.parametrize("start", ["first_state", "before_first"])
def test_the_kalman_likelihood_and_smoother_are_the_joint_normals_density_and_conditional_mean(start):
    """A time-varying regression (a two-state random-walk beta, H_t = [1, x_t]): the likelihood is the observations'
    joint normal density, the RTS means and covariances the states' normal conditional ones given every observation."""
    rng = np.random.RandomState(57)
    steps = 12
    regressors = np.column_stack([np.ones(steps), rng.normal(size=steps)])
    y = regressors @ [0.1, 1.2] + 0.3 * rng.normal(size=steps)
    f, q, r = np.eye(2), np.diag([1e-3, 4e-3]), np.array([[0.09]])
    a0, p0 = np.array([0.0, 1.0]), np.diag([0.5, 0.5])
    found = qf.kalman_filter(observations=y, transition=f, observation_matrix=regressors, state_cov=q,
                             observation_cov=r, initial_state=a0, initial_cov=p0, start=start, smoother="rts")
    if start == "before_first":
        a0, p0 = f @ a0, f @ p0 @ f.T + q
    mean_x, cov_x, mean_y, cov_y, cross = joint(f, [row[None, :] for row in regressors], q, r, a0, p0, steps)
    assert found["loglik"] == pytest.approx(stats.multivariate_normal(mean_y, cov_y).logpdf(y), rel=1e-10)
    assert found["loglik_no_constant"] == pytest.approx(found["loglik"] + 0.5 * steps * math.log(2 * math.pi))
    conditional = mean_x + cross @ np.linalg.solve(cov_y, y - mean_y)
    assert found["smoothed_state"].ravel() == pytest.approx(conditional, rel=1e-9, abs=1e-12)
    spread = cov_x - cross @ np.linalg.solve(cov_y, cross.T)
    for t in range(steps):
        assert found["smoothed_cov"][t] == pytest.approx(spread[2 * t:2 * t + 2, 2 * t:2 * t + 2], rel=1e-8, abs=1e-12)
    assert found["smoothed_state"][-1] == pytest.approx(found["filtered_state"][-1], rel=1e-12)


def test_a_mean_reverting_two_state_model_is_its_joint_normal_too():
    """F not the identity (a stationary VAR(1) state seen through two noisy series): the likelihood and the smoother
    against the joint normal, as above, and the filter's last state against the normal conditional mean on the data."""
    rng = np.random.RandomState(58)
    steps = 10
    f = np.array([[0.8, 0.1], [-0.2, 0.6]])
    h = np.array([[1.0, 0.0], [0.5, 1.0]])
    q, r = np.array([[0.05, 0.01], [0.01, 0.03]]), np.diag([0.2, 0.1])
    a0, p0 = np.array([0.3, -0.1]), np.eye(2) * 0.4
    y = rng.normal(size=(steps, 2))
    found = qf.kalman_filter(observations=y, transition=f, observation_matrix=h, state_cov=q, observation_cov=r,
                             initial_state=a0, initial_cov=p0, start="first_state", smoother="rts")
    mean_x, cov_x, mean_y, cov_y, cross = joint(f, [h] * steps, q, r, a0, p0, steps)
    assert found["loglik"] == pytest.approx(stats.multivariate_normal(mean_y, cov_y).logpdf(y.ravel()), rel=1e-10)
    conditional = mean_x + cross @ np.linalg.solve(cov_y, y.ravel() - mean_y)
    assert found["smoothed_state"].ravel() == pytest.approx(conditional, rel=1e-9, abs=1e-12)
    assert found["next_state"] == pytest.approx(f @ found["filtered_state"][-1], rel=1e-12)


def test_the_rts_smoother_handles_a_deterministic_state_and_refuses_invalid_covariance():
    """Normal conditioning remains defined when a state is known exactly: only the observation covariance
    requires inversion. The deterministic coordinate's conditional covariance is zero."""
    f, h = np.diag([0.8, 1.0]), np.array([[1.0, 1.0]])
    q, r = np.diag([0.0, 0.1]), np.array([[0.2]])
    a0, p0 = np.array([0.5, 1.0]), np.diag([0.0, 0.3])
    y = np.array([2.0, 1.5, 2.2, 1.9])
    found = qf.kalman_filter(observations=y, transition=f, observation_matrix=h, state_cov=q,
                             observation_cov=r, initial_state=a0, initial_cov=p0, start="first_state",
                             smoother="rts")
    mean_x, cov_x, mean_y, cov_y, cross = joint(f, [h] * y.size, q, r, a0, p0, y.size)
    conditional = mean_x + cross @ np.linalg.solve(cov_y, y - mean_y)
    spread = cov_x - cross @ np.linalg.solve(cov_y, cross.T)
    assert found["smoothed_state"].ravel() == pytest.approx(conditional, rel=1e-11)
    for t in range(y.size):
        assert found["smoothed_cov"][t] == pytest.approx(spread[2 * t:2 * t + 2, 2 * t:2 * t + 2], abs=1e-12)
    for invalid in ([[0.3, 0.1], [0.0, 0.3]], [[0.3, 0.0], [0.0, -0.1]]):
        with pytest.raises(ValueError, match="covariance"):
            qf.kalman_filter(observations=y, transition=f, observation_matrix=h, state_cov=q,
                             observation_cov=r, initial_state=a0, initial_cov=invalid, start="first_state",
                             smoother="none")


def test_a_local_levels_gain_settles_at_its_closed_form_and_a_missing_value_carries_the_prediction():
    """Harvey (1989), 3.3: the predicted variance P solves P^2 - qP - qr = 0, the gain K = P / (P + r)."""
    q, r = 0.09, 1.0
    found = local_level(np.r_[LEVEL, LEVEL], "first_state", "none")
    steady = (q + math.sqrt(q * q + 4 * q * r)) / 2
    assert found["gain"][-1, 0, 0] == pytest.approx(steady / (steady + r), rel=1e-10)
    assert found["predicted_cov"][-1, 0, 0] == pytest.approx(steady, rel=1e-10)
    gappy = LEVEL.copy()
    gappy[[5, 6, 20]] = np.nan
    missing = local_level(gappy, "first_state", "none")
    assert missing["filtered_state"][5] == pytest.approx(missing["predicted_state"][5])
    assert missing["predicted_cov"][7, 0, 0] == pytest.approx(missing["filtered_cov"][5, 0, 0] + 2 * q)
    keep = ~np.isnan(gappy)
    times = np.arange(40)
    cov = 4.0 + q * np.minimum.outer(times, times) + np.eye(40) * r
    density = stats.multivariate_normal(np.zeros(keep.sum()), cov[np.ix_(keep, keep)]).logpdf(gappy[keep])
    assert missing["loglik"] == pytest.approx(density, rel=1e-10) and np.all(np.isnan(missing["innovations"][[5, 6, 20]]))


def test_the_two_starts_differ_by_one_prediction_and_a_transposed_matrix_is_refused():
    before = local_level(LEVEL, "before_first", "none", initial=(0.0, 4.0))
    first = local_level(LEVEL, "first_state", "none", initial=(0.0, 4.09))
    assert before["filtered_state"] == pytest.approx(first["filtered_state"], rel=1e-12)
    assert before["loglik"] == pytest.approx(first["loglik"], rel=1e-12)
    with pytest.raises(ValueError, match="shape"):
        qf.kalman_filter(observations=np.zeros((5, 2)), transition=np.eye(3), observation_matrix=np.ones((3, 2)),
                         state_cov=np.eye(3), observation_cov=np.eye(2), initial_state=np.zeros(3),
                         initial_cov=np.eye(3), start="first_state", smoother="none")


# ------------------------------------------------------------------------------------------------ seasonality


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
@pytest.mark.parametrize("period", [4, 3])
def test_the_decomposition_is_statsmodels_and_its_factors_sum_to_zero_or_to_the_period(model, period):
    found = qf.seasonal_decompose(series=SEASONAL, period=period, model=model)
    want = STATSMODELS[("decompose", model, period)]
    assert found["trend"] == pytest.approx(np.array(want["trend"]), rel=1e-12, nan_ok=True)
    assert found["factors"] == pytest.approx(want["seasonal"], rel=1e-12, abs=1e-12)
    assert found["factors"].sum() == pytest.approx(0.0 if model == "additive" else period, abs=1e-12)
    rebuilt = (found["trend"] + found["seasonal"] + found["resid"] if model == "additive"
               else found["trend"] * found["seasonal"] * found["resid"])
    assert rebuilt[np.isfinite(rebuilt)] == pytest.approx(SEASONAL[np.isfinite(rebuilt)], rel=1e-12)
    assert np.isnan(found["trend"][:period // 2]).all() and np.isfinite(found["trend"][period // 2:-(period // 2)]).all()


def test_a_linear_trend_plus_a_fixed_pattern_decomposes_exactly():
    pattern = np.array([1.0, -2.0, 0.5, 0.5])
    for model in ("additive", "multiplicative"):
        if model == "additive":
            series = 10.0 + 0.3 * np.arange(24) + np.tile(pattern, 6)
        else:
            series = (10.0 + 0.3 * np.arange(24)) * np.tile(1.0 + pattern / 10, 6)
        found = qf.seasonal_decompose(series=series, period=4, model=model)
        if model == "additive":
            assert found["factors"] == pytest.approx(pattern, abs=1e-12)
            assert found["adjusted"] == pytest.approx(10.0 + 0.3 * np.arange(24), rel=1e-12)
        assert np.nanmax(np.abs(found["trend"] - (10.0 + 0.3 * np.arange(24)))) < (1e-12 if model == "additive" else 0.05)
    with pytest.raises(ValueError, match="two full periods"):
        qf.seasonal_decompose(series=SEASONAL[:7], period=4, model="additive")


if __name__ == "__main__":
    for key, value in reference_values().items():
        print(f"    {key!r}: {value!r},")
