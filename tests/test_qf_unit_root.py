"""qf.econometrics.adf_test and engle_granger (runs/lanes/lib7-prompt.md, rank 17 of
docs/research/library-syllabus-1011.md): against statsmodels 0.15.0's adfuller and coint on series of this file's
own, every trend and lag rule (the values frozen below: `scratch/lib2/venv/bin/python tests/test_qf_unit_root.py`
prints them again, and the slow test recomputes them where statsmodels imports); the Dickey-Fuller regression
written out; MacKinnon's tables at their published landmarks. No unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf.econometrics as e
from qf import _mackinnon

pytestmark = pytest.mark.fast

nan = math.nan


def series():
    out = {}
    rng = np.random.default_rng(101)
    out["walk"] = 100 + np.cumsum(rng.standard_normal(250))
    rng = np.random.default_rng(102)
    shocks, x = rng.standard_normal(200), np.zeros(200)
    for t in range(1, 200):
        x[t] = 0.7 * x[t - 1] + shocks[t]
    out["ar1"] = x
    rng = np.random.default_rng(103)
    shocks, x = rng.standard_normal(180), np.zeros(180)
    for t in range(1, 180):
        x[t] = 0.5 * x[t - 1] + shocks[t]
    out["trend"] = 0.05 * np.arange(180) + x
    rng = np.random.default_rng(104)
    out["short"] = np.cumsum(rng.standard_normal(40))
    return out


def pairs():
    rng = np.random.default_rng(201)
    x = np.cumsum(rng.standard_normal(300))
    shocks, u = rng.standard_normal(300), np.zeros(300)
    for t in range(1, 300):
        u[t] = 0.6 * u[t - 1] + shocks[t]
    z = np.cumsum(rng.standard_normal(300))
    return {"cointegrated": (1.0 + 2.0 * x + u, x), "independent": (np.cumsum(rng.standard_normal(300)), z),
            "two": (0.5 + x - 0.8 * z + u, np.column_stack([x, z]))}


SERIES, PAIRS = series(), pairs()
# (series, trend, lag rule, max lag): (statistic, p-value, lag, nobs, critical values 1%, 5%, 10%)
ADF = {
    ('walk', 'none', 'aic', 6):
        (-1.2283833563641373, 0.20127973903913576, 4, 245, (-2.5749261391087046, -1.9421502633766543, -1.6157794081377657)),
    ('walk', 'none', 'bic', 6):
        (-1.0699710517252632, 0.2569401905546619, 0, 249, (-2.57477761552233, -1.9421309657332253, -1.615795463888424)),
    ('walk', 'none', 't_stat', 6):
        (-1.2283833563641373, 0.20127973903913576, 4, 245, (-2.5749261391087046, -1.9421502633766543, -1.6157794081377657)),
    ('walk', 'none', 'fixed', 6):
        (-1.5451949939438139, 0.11494271887683716, 6, 243, (-2.5750022466087485, -1.94216016038016, -1.6157711900523155)),
    ('walk', 'constant', 'aic', 6):
        (-0.6838001060518426, 0.850901171555028, 0, 249, (-3.4568881317725864, -2.8732185133016057, -2.5729936189738876)),
    ('walk', 'constant', 'bic', 6):
        (-0.6838001060518426, 0.850901171555028, 0, 249, (-3.4568881317725864, -2.8732185133016057, -2.5729936189738876)),
    ('walk', 'constant', 't_stat', 6):
        (-0.2781101265502812, 0.9285478852089474, 4, 245, (-3.4573260719088132, -2.873410402808354, -2.573095980841316)),
    ('walk', 'constant', 'fixed', 6):
        (-0.11711231545514292, 0.9477071807293301, 6, 243, (-3.4575505077947746, -2.8735087323013526, -2.573148434859185)),
    ('walk', 'constant_trend', 'aic', 6):
        (-3.217332968672065, 0.08100289187602011, 0, 249, (-3.9955950298398477, -3.428270807162781, -3.1374986908197946)),
    ('walk', 'constant_trend', 'bic', 6):
        (-3.217332968672065, 0.08100289187602011, 0, 249, (-3.9955950298398477, -3.428270807162781, -3.1374986908197946)),
    ('walk', 'constant_trend', 't_stat', 6):
        (-3.1881821295065347, 0.08678238587244091, 4, 245, (-3.996204153626465, -3.428563622657226, -3.1376703806237196)),
    ('walk', 'constant_trend', 'fixed', 6):
        (-2.725195719094372, 0.2257706902802642, 6, 243, (-3.9965163357243862, -3.428713677357446, -3.137758359138435)),
    ('ar1', 'none', 'aic', 6):
        (-6.490788407193224, 8.155957705111391e-10, 0, 199, (-2.5770667644756444, -1.942430759336949, -1.615550641718986)),
    ('ar1', 'none', 'bic', 6):
        (-6.490788407193224, 8.155957705111391e-10, 0, 199, (-2.5770667644756444, -1.942430759336949, -1.615550641718986)),
    ('ar1', 'none', 't_stat', 6):
        (-6.490788407193224, 8.155957705111391e-10, 0, 199, (-2.5770667644756444, -1.942430759336949, -1.615550641718986)),
    ('ar1', 'none', 'fixed', 6):
        (-4.302409922156168, 2.1606046979444775e-05, 6, 193, (-2.577421827700072, -1.9424777047114803, -1.6155131670732337)),
    ('ar1', 'constant', 'aic', 6):
        (-6.592611341691443, 7.052709825082816e-09, 0, 199, (-3.4636447617687436, -2.8761761179270766, -2.57457158581854)),
    ('ar1', 'constant', 'bic', 6):
        (-6.592611341691443, 7.052709825082816e-09, 0, 199, (-3.4636447617687436, -2.8761761179270766, -2.57457158581854)),
    ('ar1', 'constant', 't_stat', 6):
        (-6.592611341691443, 7.052709825082816e-09, 0, 199, (-3.4636447617687436, -2.8761761179270766, -2.57457158581854)),
    ('ar1', 'constant', 'fixed', 6):
        (-4.545327853251974, 0.00016270193547254826, 6, 193, (-3.4646940755442612, -2.8766348847254934, -2.5748163958763994)),
    ('ar1', 'constant_trend', 'aic', 6):
        (-6.671040367553972, 7.72918081738121e-08, 0, 199, (-4.004997848936356, -3.4327862452981046, -3.140144918368515)),
    ('ar1', 'constant_trend', 'bic', 6):
        (-6.671040367553972, 7.72918081738121e-08, 0, 199, (-4.004997848936356, -3.4327862452981046, -3.140144918368515)),
    ('ar1', 'constant_trend', 't_stat', 6):
        (-6.671040367553972, 7.72918081738121e-08, 0, 199, (-4.004997848936356, -3.4327862452981046, -3.140144918368515)),
    ('ar1', 'constant_trend', 'fixed', 6):
        (-4.5983585791736195, 0.0010376031334776894, 6, 193, (-4.006459103160539, -3.4334870817591234, -3.1405553762127636)),
    ('trend', 'none', 'aic', 6):
        (0.8832362854273378, 0.8987059959524639, 6, 173, (-2.578784886230746, -1.942659003649678, -1.615370522479309)),
    ('trend', 'none', 'bic', 6):
        (0.8832362854273378, 0.8987059959524639, 6, 173, (-2.578784886230746, -1.942659003649678, -1.615370522479309)),
    ('trend', 'none', 't_stat', 6):
        (0.8832362854273378, 0.8987059959524639, 6, 173, (-2.578784886230746, -1.942659003649678, -1.615370522479309)),
    ('trend', 'none', 'fixed', 6):
        (0.8832362854273378, 0.8987059959524639, 6, 173, (-2.578784886230746, -1.942659003649678, -1.615370522479309)),
    ('trend', 'constant', 'aic', 6):
        (-1.5065538564973944, 0.5302667843430492, 6, 173, (-3.4687256239864017, -2.8783961376954363, -2.57575634100705)),
    ('trend', 'constant', 'bic', 6):
        (-1.5065538564973944, 0.5302667843430492, 6, 173, (-3.4687256239864017, -2.8783961376954363, -2.57575634100705)),
    ('trend', 'constant', 't_stat', 6):
        (-1.5065538564973944, 0.5302667843430492, 6, 173, (-3.4687256239864017, -2.8783961376954363, -2.57575634100705)),
    ('trend', 'constant', 'fixed', 6):
        (-1.5065538564973944, 0.5302667843430492, 6, 173, (-3.4687256239864017, -2.8783961376954363, -2.57575634100705)),
    ('trend', 'constant_trend', 'aic', 6):
        (-8.022814797698048, 6.32787145422618e-11, 0, 179, (-4.01025660682481, -3.4353072992738527, -3.1416210941585145)),
    ('trend', 'constant_trend', 'bic', 6):
        (-8.022814797698048, 6.32787145422618e-11, 0, 179, (-4.01025660682481, -3.4353072992738527, -3.1416210941585145)),
    ('trend', 'constant_trend', 't_stat', 6):
        (-3.7217065593433976, 0.020967018276260213, 6, 173, (-4.012075815845091, -3.4361787125348875, -3.1421311308149904)),
    ('trend', 'constant_trend', 'fixed', 6):
        (-3.7217065593433976, 0.020967018276260213, 6, 173, (-4.012075815845091, -3.4361787125348875, -3.1421311308149904)),
    ('short', 'none', 'aic', 3):
        (1.853434608586102, 0.9855121336072262, 1, 38, (-2.6270886149584487, -1.9498297383000438, -1.6112477883073333)),
    ('short', 'none', 'bic', 3):
        (1.3306687768099035, 0.9533863169447742, 0, 39, (-2.6254528205128205, -1.9495731822856084, -1.6113665095500598)),
    ('short', 'none', 't_stat', 3):
        (1.853434608586102, 0.9855121336072262, 1, 38, (-2.6270886149584487, -1.9498297383000438, -1.6112477883073333)),
    ('short', 'none', 'fixed', 3):
        (1.5996806571584117, 0.9733552567556493, 3, 36, (-2.6306441666666665, -1.950388344478738, -1.610992719478738)),
    ('short', 'constant', 'aic', 3):
        (-0.37776048794754863, 0.9137462838179651, 2, 37, (-3.6209175221605827, -2.9435394610388332, -2.6104002410518627)),
    ('short', 'constant', 'bic', 3):
        (-0.4308124513492815, 0.9048256982388874, 1, 38, (-3.6155091011809297, -2.941262357486514, -2.6091995013850418)),
    ('short', 'constant', 't_stat', 3):
        (-0.37776048794754863, 0.9137462838179651, 2, 37, (-3.6209175221605827, -2.9435394610388332, -2.6104002410518627)),
    ('short', 'constant', 'fixed', 3):
        (-0.5714274700204873, 0.8773084347205421, 3, 36, (-3.626651907578875, -2.9459512825788754, -2.6116707716049383)),
    ('short', 'constant_trend', 'aic', 3):
        (-3.2882469784093193, 0.06820731366163843, 0, 39, (-4.211852690200441, -3.5297701024966703, -3.196305257168867)),
    ('short', 'constant_trend', 'bic', 3):
        (-3.2882469784093193, 0.06820731366163843, 0, 39, (-4.211852690200441, -3.5297701024966703, -3.196305257168867)),
    ('short', 'constant_trend', 't_stat', 3):
        (-3.2882469784093193, 0.06820731366163843, 0, 39, (-4.211852690200441, -3.5297701024966703, -3.196305257168867)),
    ('short', 'constant_trend', 'fixed', 3):
        (-1.7179201296817384, 0.7429293374061299, 3, 36, (-4.235055592421125, -3.540390300068587, -3.2023804526748973)),
}
# (pair, trend, lag rule, max lag): (statistic, p-value, critical values 1%, 5%, 10%)
EG = {
    ('cointegrated', 'none', 'aic', 5):
        (-9.36014955365332, 8.408504274904676e-16, (nan, nan, nan)),
    ('cointegrated', 'none', 'bic', 5):
        (-9.36014955365332, 8.408504274904676e-16, (nan, nan, nan)),
    ('cointegrated', 'none', 't_stat', 5):
        (-9.36014955365332, 8.408504274904676e-16, (nan, nan, nan)),
    ('cointegrated', 'none', 'fixed', 5):
        (-5.9643588023326535, 1.7401924274772918e-07, (nan, nan, nan)),
    ('cointegrated', 'constant', 'aic', 5):
        (-9.697019805056103, 1.4508667010363245e-15, (-3.933443446270176, -3.35664143611369, -3.0586650401002227)),
    ('cointegrated', 'constant', 'bic', 5):
        (-9.697019805056103, 1.4508667010363245e-15, (-3.933443446270176, -3.35664143611369, -3.0586650401002227)),
    ('cointegrated', 'constant', 't_stat', 5):
        (-9.697019805056103, 1.4508667010363245e-15, (-3.933443446270176, -3.35664143611369, -3.0586650401002227)),
    ('cointegrated', 'constant', 'fixed', 5):
        (-6.28676991828767, 3.810658828690253e-07, (-3.933443446270176, -3.35664143611369, -3.0586650401002227)),
    ('cointegrated', 'constant_trend', 'aic', 5):
        (-9.701791812075543, 1.8612778314468176e-15, (-4.379653537656178, -3.812513081173589, -3.520077444241213)),
    ('cointegrated', 'constant_trend', 'bic', 5):
        (-9.701791812075543, 1.8612778314468176e-15, (-4.379653537656178, -3.812513081173589, -3.520077444241213)),
    ('cointegrated', 'constant_trend', 't_stat', 5):
        (-9.701791812075543, 1.8612778314468176e-15, (-4.379653537656178, -3.812513081173589, -3.520077444241213)),
    ('cointegrated', 'constant_trend', 'fixed', 5):
        (-6.285571224452915, 2.2470130208944965e-06, (-4.379653537656178, -3.812513081173589, -3.520077444241213)),
    ('independent', 'none', 'aic', 5):
        (-1.0708447195464206, 0.6713731838853135, (nan, nan, nan)),
    ('independent', 'none', 'bic', 5):
        (-1.0708447195464206, 0.6713731838853135, (nan, nan, nan)),
    ('independent', 'none', 't_stat', 5):
        (-1.1969137596685253, 0.6150847516186674, (nan, nan, nan)),
    ('independent', 'none', 'fixed', 5):
        (-1.1615891386339836, 0.6312768664580163, (nan, nan, nan)),
    ('independent', 'constant', 'aic', 5):
        (-2.2256606053477532, 0.4111730204329311, (-3.933443446270176, -3.35664143611369, -3.0586650401002227)),
    ('independent', 'constant', 'bic', 5):
        (-2.2256606053477532, 0.4111730204329311, (-3.933443446270176, -3.35664143611369, -3.0586650401002227)),
    ('independent', 'constant', 't_stat', 5):
        (-2.2256606053477532, 0.4111730204329311, (-3.933443446270176, -3.35664143611369, -3.0586650401002227)),
    ('independent', 'constant', 'fixed', 5):
        (-1.8115539155368425, 0.6240931496290584, (-3.933443446270176, -3.35664143611369, -3.0586650401002227)),
    ('independent', 'constant_trend', 'aic', 5):
        (-2.195112075253517, 0.6825663063948574, (-4.379653537656178, -3.812513081173589, -3.520077444241213)),
    ('independent', 'constant_trend', 'bic', 5):
        (-2.195112075253517, 0.6825663063948574, (-4.379653537656178, -3.812513081173589, -3.520077444241213)),
    ('independent', 'constant_trend', 't_stat', 5):
        (-2.195112075253517, 0.6825663063948574, (-4.379653537656178, -3.812513081173589, -3.520077444241213)),
    ('independent', 'constant_trend', 'fixed', 5):
        (-1.7899222240014894, 0.8483774158225308, (-4.379653537656178, -3.812513081173589, -3.520077444241213)),
    ('two', 'none', 'aic', 5):
        (-9.74399269291268, 2.328327443278465e-16, (nan, nan, nan)),
    ('two', 'none', 'bic', 5):
        (-9.74399269291268, 2.328327443278465e-16, (nan, nan, nan)),
    ('two', 'none', 't_stat', 5):
        (-9.74399269291268, 2.328327443278465e-16, (nan, nan, nan)),
    ('two', 'none', 'fixed', 5):
        (-6.34387490741443, 2.10420807595832e-07, (nan, nan, nan)),
    ('two', 'constant', 'aic', 5):
        (-9.744011978846194, 6.134654166827609e-16, (-4.342388459874095, -3.769419803746219, -3.4730051999418348)),
    ('two', 'constant', 'bic', 5):
        (-9.744011978846194, 6.134654166827609e-16, (-4.342388459874095, -3.769419803746219, -3.4730051999418348)),
    ('two', 'constant', 't_stat', 5):
        (-9.744011978846194, 6.134654166827609e-16, (-4.342388459874095, -3.769419803746219, -3.4730051999418348)),
    ('two', 'constant', 'fixed', 5):
        (-6.336748123928711, 1.4806312445746916e-06, (-4.342388459874095, -3.769419803746219, -3.4730051999418348)),
    ('two', 'constant_trend', 'aic', 5):
        (-9.743606642766732, 2.1267711462325958e-15, (-4.726374969047992, -4.158883223504754, -3.8655459380243813)),
    ('two', 'constant_trend', 'bic', 5):
        (-9.743606642766732, 2.1267711462325958e-15, (-4.726374969047992, -4.158883223504754, -3.8655459380243813)),
    ('two', 'constant_trend', 't_stat', 5):
        (-9.743606642766732, 2.1267711462325958e-15, (-4.726374969047992, -4.158883223504754, -3.8655459380243813)),
    ('two', 'constant_trend', 'fixed', 5):
        (-6.332714655953498, 8.114133935069457e-06, (-4.726374969047992, -4.158883223504754, -3.8655459380243813)),
}


def test_the_unit_root_cards_keep_the_format_and_their_examples_run():
    check_cards(e)


@pytest.mark.parametrize("key", list(ADF))
def test_the_adf_test_is_statsmodels_adfuller(key):
    name, trend, rule, most = key
    statistic, p_value, lag, nobs, critical = ADF[key]
    found = e.adf_test(SERIES[name], trend, most, rule)
    assert (found["lag"], found["nobs"]) == (lag, nobs)
    assert found["statistic"] == pytest.approx(statistic, rel=1e-10)
    assert found["p_value"] == pytest.approx(p_value, rel=1e-9, abs=1e-15)
    assert list(found["critical_values"].values()) == pytest.approx(critical, rel=1e-12)


@pytest.mark.parametrize("key", list(EG))
def test_engle_granger_is_statsmodels_coint(key):
    name, trend, rule, most = key
    statistic, p_value, critical = EG[key]
    y, x = PAIRS[name]
    found = e.engle_granger(y, x, trend, most, rule)
    assert found["statistic"] == pytest.approx(statistic, rel=1e-10)
    assert found["p_value"] == pytest.approx(p_value, rel=1e-9, abs=1e-15)
    assert list(found["critical_values"].values()) == pytest.approx(critical, rel=1e-12, nan_ok=True)


def test_the_dickey_fuller_regression_written_out():
    """At a fixed lag: dy_t on 1, y_{t-1}, dy_{t-1}, dy_{t-2} over every change the lags allow, OLS by hand, the t of
    y_{t-1} with s^2 = SSR / (n - k); and the cointegrating regression's coefficients by numpy's lstsq."""
    y = SERIES["ar1"]
    dy = np.diff(y)
    rows = np.arange(2, dy.size)
    x = np.column_stack([np.ones(rows.size), y[rows], dy[rows - 1], dy[rows - 2]])
    coef = np.linalg.solve(x.T @ x, x.T @ dy[rows])
    resid = dy[rows] - x @ coef
    se = np.sqrt(resid @ resid / (rows.size - 4) * np.linalg.inv(x.T @ x)[1, 1])
    found = e.adf_test(y, "constant", 2, "fixed")
    assert found["statistic"] == pytest.approx(coef[1] / se, rel=1e-12) and found["gamma"] == pytest.approx(coef[1])
    assert found["nobs"] == rows.size == y.size - 3
    target, regressors = PAIRS["two"]
    eg = e.engle_granger(target, regressors, "constant_trend", 2, "fixed")
    design = np.column_stack([np.ones(300), np.arange(1.0, 301), regressors])
    assert eg["coefficients"] == pytest.approx(np.linalg.lstsq(design, target, rcond=None)[0], rel=1e-10)
    assert eg["residuals"] == pytest.approx(target - design @ eg["coefficients"], abs=1e-10)


def test_mackinnons_tables_at_their_landmarks():
    """The Dickey-Fuller 5% values with a constant, -2.86 asymptotically and with a trend -3.41 (MacKinnon 2010,
    Table 2); a statistic at the asymptotic 5% value has p near 0.05 and at the 1% value near 0.01 (MacKinnon 1994)."""
    big = 1e15
    assert _mackinnon.critical_values("constant", 1, big)["5%"] == pytest.approx(-2.86154, abs=1e-5)
    assert _mackinnon.critical_values("constant_trend", 1, big)["5%"] == pytest.approx(-3.41049, abs=1e-5)
    assert _mackinnon.critical_values("constant", 2, big)["5%"] == pytest.approx(-3.33613, abs=1e-5)
    for trend in ("constant", "constant_trend"):
        values = _mackinnon.critical_values(trend, 1, big)
        assert _mackinnon.p_value(values["5%"], trend, 1) == pytest.approx(0.05, abs=2e-3)
        assert _mackinnon.p_value(values["1%"], trend, 1) == pytest.approx(0.01, abs=1e-3)
    assert _mackinnon.p_value(-30.0, "constant", 1) == 0.0 and _mackinnon.p_value(3.0, "constant", 1) == 1.0


def test_a_walk_keeps_its_unit_root_and_stationary_series_reject_it():
    assert e.adf_test(SERIES["walk"], "constant", 6, "aic")["p_value"] > 0.1
    assert e.adf_test(SERIES["ar1"], "constant", 6, "aic")["p_value"] < 0.01
    assert e.adf_test(SERIES["trend"], "constant_trend", 6, "aic")["p_value"] < 0.01
    assert e.engle_granger(*PAIRS["cointegrated"], "constant", 5, "aic")["p_value"] < 0.01
    assert e.engle_granger(*PAIRS["independent"], "constant", 5, "aic")["p_value"] > 0.1


@pytest.mark.parametrize("bad, match", [(dict(trend="drift"), "trend"), (dict(lag_rule="hqic"), "lag_rule"),
                                        (dict(max_lag=200), "max_lag"), (dict(max_lag=1.5), "max_lag")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        e.adf_test(**{**dict(series=SERIES["walk"], trend="constant", max_lag=4, lag_rule="aic"), **bad})
    with pytest.raises(ValueError, match="linear function"):
        e.engle_granger(2.0 * SERIES["walk"] + 1.0, SERIES["walk"], "constant", 2, "aic")


def statsmodels_values():
    import warnings
    from statsmodels.tsa.stattools import adfuller, coint
    warnings.simplefilter("ignore")
    trends = {"none": "n", "constant": "c", "constant_trend": "ct"}
    rules = {"aic": "AIC", "bic": "BIC", "t_stat": "t-stat", "fixed": None}
    adf, eg = {}, {}
    for (name, trend, rule, most) in ADF:
        stat, p, lag, nobs, crit = adfuller(SERIES[name], maxlag=most, regression=trends[trend], autolag=rules[rule],
                                            result_object=False)[:5]
        adf[(name, trend, rule, most)] = (stat, p, lag, nobs, (crit["1%"], crit["5%"], crit["10%"]))
    for (name, trend, rule, most) in EG:
        stat, p, crit = coint(*PAIRS[name], trend=trends[trend], maxlag=most,
                              autolag=None if rule == "fixed" else rule.replace("_", "-"))[:3]
        eg[(name, trend, rule, most)] = (stat, p, tuple(crit))
    return adf, eg


@pytest.mark.slow
def test_the_frozen_values_are_statsmodels():
    pytest.importorskip("statsmodels")
    adf, eg = statsmodels_values()
    for key, value in adf.items():
        assert value[2:4] == ADF[key][2:4] and value[:2] == pytest.approx(ADF[key][:2], rel=1e-12, abs=1e-15)
    for key, value in eg.items():
        assert value[:2] == pytest.approx(EG[key][:2], rel=1e-12, abs=1e-15)


if __name__ == "__main__":
    print(statsmodels_values())
