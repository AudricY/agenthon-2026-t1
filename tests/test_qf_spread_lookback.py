"""qf.spread_price and qf.lookback_price (lane libgaps1; docs/research/library-syllabus-1011.md rows 112 and 49):
Kirk's spread approximation and Margrabe's exchange option; Goldman, Sosin and Gatto's floating and Conze and
Viswanathan's fixed lookbacks, with Broadie, Glasserman and Kou's shift for discrete monitoring. References: Haug's
worked example, QuantLib 1.43's engines (frozen in `QL`, `__main__` recomputes them), the closed-form identities
(Kirk at K = 0 is Margrabe; parity; a fixed lookback struck at the extremum against the floating one), and Monte
Carlo on this file's own parameters; no unit's values appear (rule 8)."""
import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast

QL = {   # QuantLib 1.43, Actual365Fixed, days from 2024-01-15 (`quantlib_values`)
    "lookback": [
        (("call", "floating", 120.0, None, 100.0, 182, 0.1, 0.04, 0.3), 26.26557673264839),
        (("put", "floating", 100.0, None, 110.0, 365, 0.05, 0.0, 0.25), 20.06690389441921),
        (("call", "fixed", 100.0, 105.0, 100.0, 182, 0.1, 0.0, 0.3), 15.819497681991297),
        (("call", "fixed", 100.0, 95.0, 110.0, 274, 0.06, 0.02, 0.2), 22.382448958702305),
        (("put", "fixed", 100.0, 95.0, 100.0, 365, 0.05, 0.03, 0.35), 18.56269195887674),
        (("put", "fixed", 100.0, 105.0, 90.0, 182, 0.06, 0.02, 0.25), 19.221396851505283)],
    "margrabe": [   # AnalyticEuropeanMargrabeEngine, one unit of each asset
        ((22.0, 20.0, 36, 0.1, 0.06, 0.04, 0.2, 0.25, -0.5), 2.2779900223199068),
        ((100.0, 95.0, 365, 0.05, 0.02, 0.0, 0.3, 0.2, 0.4), 12.562734721605779)],
    "kirk": [   # KirkEngine on two Black processes (futures prices)
        ((122.0, 120.0, 3.0, 36, 0.1, 0.2, 0.2, -0.5, "call"), 4.717823188421923),
        ((50.0, 45.0, -2.0, 182, 0.03, 0.4, 0.3, 0.8, "put"), 0.7908279409489826)],
}


def spread(method, option, s1, s2, k, t, r, q1, q2, v1, v2, rho):
    return qf.spread_price(method=method, option=option, spot1=s1, spot2=s2, strike=k, maturity=t, rate=r,
                           dividend_yield1=q1, dividend_yield2=q2, vol1=v1, vol2=v2, correlation=rho)


def lookback(option, kind, s, k, extreme, t, r, q, v, monitoring="continuous"):
    return qf.lookback_price(option=option, strike_type=kind, spot=s, strike=k, extreme=extreme, maturity=t, rate=r,
                             dividend_yield=q, vol=v, monitoring=monitoring)


def test_kirk_gives_haugs_worked_example():
    """Haug, The Complete Guide to Option Pricing Formulas (2007), the spread option approximation (Kirk 1995): a call
    on the spread of two futures, F1 28, F2 20, K 7, three months, r 5%, vols 29% and 36%, correlation 0.42: 2.1670."""
    found = spread("kirk", "call", 28.0, 20.0, 7.0, 0.25, 0.05, 0.05, 0.05, 0.29, 0.36, 0.42)
    assert found["price"] == pytest.approx(2.1670, abs=5e-5)


def test_kirk_and_margrabe_equal_quantlibs_engines():
    for (f1, f2, k, days, r, v1, v2, rho, option), expected in QL["kirk"]:
        assert spread("kirk", option, f1, f2, k, days / 365, r, r, r, v1, v2, rho)["price"] == \
            pytest.approx(expected, abs=1e-12)
    for (s1, s2, days, r, q1, q2, v1, v2, rho), expected in QL["margrabe"]:
        assert spread("margrabe", "call", s1, s2, 0.0, days / 365, r, q1, q2, v1, v2, rho)["price"] == \
            pytest.approx(expected, abs=1e-12)


def test_kirk_at_zero_strike_is_margrabe_and_both_keep_parity():
    """Kirk's effective vol at K = 0 is Margrabe's sqrt(v1^2 + v2^2 - 2 rho v1 v2) and the prices agree exactly
    (Haug 2007); a call less a put is e^{-rT}(F1 - F2 - K); the exchange option to receive S2 for S1 is the call
    with the assets swapped."""
    point = (105.0, 98.0, 0.9, 0.03, 0.01, 0.02, 0.32, 0.21, 0.35)
    kirk = spread("kirk", "call", *point[:2], 0.0, *point[2:])
    margrabe = spread("margrabe", "call", *point[:2], 0.0, *point[2:])
    assert kirk["price"] == pytest.approx(margrabe["price"], abs=1e-12)
    assert margrabe["vol"] == pytest.approx(np.sqrt(0.32 ** 2 + 0.21 ** 2 - 2 * 0.35 * 0.32 * 0.21), abs=1e-15)
    s1, s2, t, r, q1, q2 = 105.0, 98.0, 0.9, 0.03, 0.01, 0.02
    forwards = s1 * np.exp((r - q1) * t) - s2 * np.exp((r - q2) * t)
    for method, k in (("kirk", 4.0), ("kirk", -3.0), ("margrabe", 0.0)):
        call, put = (spread(method, o, s1, s2, k, t, r, q1, q2, 0.32, 0.21, 0.35)["price"] for o in ("call", "put"))
        assert call - put == pytest.approx(np.exp(-r * t) * (forwards - k), abs=1e-12), (method, k)
    swapped = spread("margrabe", "call", s2, s1, 0.0, t, r, q2, q1, 0.21, 0.32, 0.35)["price"]
    assert spread("margrabe", "put", s1, s2, 0.0, t, r, q1, q2, 0.32, 0.21, 0.35)["price"] == \
        pytest.approx(swapped, abs=1e-12)


def test_kirk_meets_a_correlated_monte_carlo_and_margrabe_falls_with_correlation():
    """Two correlated GBMs (qf.correlated_gbm_paths, 200,000 antithetic paths): the spread call's discounted mean
    payoff within three standard errors plus Kirk's own approximation error (small at a small strike); the exchange
    option's value falls as the correlation rises (Margrabe's vol shrinks)."""
    s, q, v, rho, t, r, k = [110.0, 100.0], [0.01, 0.0], [0.3, 0.25], 0.6, 0.75, 0.04, 5.0
    paths = qf.correlated_gbm_paths(spots=s, drifts=[r - q[0], r - q[1]], vols=v,
                                    correlation=[[1.0, rho], [rho, 1.0]], times=[t], n_paths=200_000, rng=9,
                                    drift_form="arithmetic", antithetic=True)
    payoff = np.exp(-r * t) * np.maximum(paths[:, -1, 0] - paths[:, -1, 1] - k, 0.0)
    pairs = 0.5 * (payoff[:100_000] + payoff[100_000:])
    error = pairs.std(ddof=1) / np.sqrt(pairs.size)
    kirk = spread("kirk", "call", 110.0, 100.0, k, t, r, 0.01, 0.0, 0.3, 0.25, rho)["price"]
    assert abs(kirk - pairs.mean()) < 3 * error + 0.005 * kirk
    values = [spread("margrabe", "call", 100.0, 100.0, 0.0, 1.0, 0.03, 0.0, 0.0, 0.3, 0.3, c)["price"]
              for c in (-0.9, -0.5, 0.0, 0.5, 0.9)]
    assert all(np.diff(values) < 0)


def test_spread_refuses_a_margrabe_strike_and_an_unfit_kirk():
    with pytest.raises(ValueError, match="kirk"):
        spread("margrabe", "call", 100.0, 95.0, 2.0, 1.0, 0.03, 0.0, 0.0, 0.3, 0.2, 0.4)
    with pytest.raises(ValueError, match="F2 \\+ K"):
        spread("kirk", "call", 100.0, 95.0, -200.0, 1.0, 0.03, 0.0, 0.0, 0.3, 0.2, 0.4)


def test_lookbacks_equal_quantlibs_continuous_engines():
    for (option, kind, s, k, extreme, days, r, q, v), expected in QL["lookback"]:
        assert lookback(option, kind, s, k, extreme, days / 365, r, q, v) == pytest.approx(expected, abs=1e-10)


def test_a_fixed_lookback_struck_at_the_extremum_is_the_floating_one_plus_the_forward_gap():
    """Closed-form identities of the payoffs: max(M_T - M, 0) = M_T - M and M_T - S_T differ by S_T - M, so the
    fixed call struck at the observed maximum M is the floating put plus S e^{-qT} - M e^{-rT}; the fixed put at
    the minimum m the floating call less S e^{-qT} - m e^{-rT}. Below the extremum the fixed call adds
    e^{-rT}(M - K)."""
    s, t, r, q, v = 100.0, 0.8, 0.04, 0.015, 0.27
    for big in (100.0, 112.0):
        fixed = lookback("call", "fixed", s, big, big, t, r, q, v)
        floating = lookback("put", "floating", s, None, big, t, r, q, v)
        assert fixed == pytest.approx(floating + s * np.exp(-q * t) - big * np.exp(-r * t), abs=1e-10)
        assert lookback("call", "fixed", s, big - 7.0, big, t, r, q, v) == \
            pytest.approx(fixed + 7.0 * np.exp(-r * t), abs=1e-10)
    for small in (100.0, 91.0):
        fixed = lookback("put", "fixed", s, small, small, t, r, q, v)
        floating = lookback("call", "floating", s, None, small, t, r, q, v)
        assert fixed == pytest.approx(floating - s * np.exp(-q * t) + small * np.exp(-r * t), abs=1e-10)


def test_the_zero_carry_branch_is_the_limit_and_a_lookback_is_worth_more_than_the_vanilla():
    """At b = r - q = 0 Haug's term sigma^2/(2b) [...] is 0/0: the branch takes its limit, met by b = +-1e-6 to
    their own error; a fixed lookback call is worth at least the vanilla call at its strike (M_T >= S_T)."""
    for option, kind, k, extreme in (("call", "floating", None, 95.0), ("put", "floating", None, 108.0),
                                     ("call", "fixed", 104.0, 100.0), ("put", "fixed", 96.0, 100.0)):
        at = lookback(option, kind, 100.0, k, extreme, 1.0, 0.05, 0.05, 0.3)
        for b in (1e-6, -1e-6):
            assert lookback(option, kind, 100.0, k, extreme, 1.0, 0.05, 0.05 - b, 0.3) == pytest.approx(at, abs=1e-4)
    for k in (90.0, 100.0, 115.0):
        assert lookback("call", "fixed", 100.0, k, 100.0, 0.5, 0.03, 0.0, 0.2) >= \
            qf.bs_price("call", 100.0, k, 0.5, 0.03, 0.0, 0.2)


def test_the_discrete_shift_meets_a_daily_monitored_monte_carlo():
    """Broadie, Glasserman and Kou (1999): a lookback monitored every dt is the continuous one with the extremum
    shifted by e^{+-0.5826 vol sqrt(dt)}. 40,000 antithetic GBM paths monitored daily over half a year (today
    monitored), new and seasoned (an extremum observed before): every price within three standard errors, the
    continuous one well outside them; dt to 0 is the continuous price."""
    times = np.arange(1, 127) / 252
    paths = qf.gbm_paths(100.0, 0.04, 0.3, times, 40_000, 3, "arithmetic", True)
    full = np.concatenate([np.full((paths.shape[0], 1), 100.0), paths], axis=1)
    cases = (("call", "fixed", 105.0, 100.0, lambda p: np.maximum(p.max(1) - 105.0, 0.0)),
             ("put", "floating", None, 100.0, lambda p: p.max(1) - p[:, -1]),
             ("call", "floating", None, 100.0, lambda p: p[:, -1] - p.min(1)),
             ("put", "fixed", 95.0, 100.0, lambda p: np.maximum(95.0 - p.min(1), 0.0)),
             ("call", "floating", None, 97.0, lambda p: p[:, -1] - np.minimum(p.min(1), 97.0)),   # seasoned
             ("put", "floating", None, 103.0, lambda p: np.maximum(p.max(1), 103.0) - p[:, -1]),
             ("call", "fixed", 98.0, 104.0, lambda p: np.maximum(np.maximum(p.max(1), 104.0) - 98.0, 0.0)),
             ("put", "fixed", 102.0, 96.0, lambda p: np.maximum(102.0 - np.minimum(p.min(1), 96.0), 0.0)))
    for option, kind, k, extreme, payoff in cases:
        x = np.exp(-0.05 * 0.5) * payoff(full)
        x = 0.5 * (x[:20_000] + x[20_000:])
        error = x.std(ddof=1) / np.sqrt(x.size)
        shifted = lookback(option, kind, 100.0, k, extreme, 0.5, 0.05, 0.01, 0.3, monitoring=1 / 252)
        continuous = lookback(option, kind, 100.0, k, extreme, 0.5, 0.05, 0.01, 0.3)
        assert abs(shifted - x.mean()) < 3 * error and abs(continuous - x.mean()) > 5 * error, (option, kind)
        assert lookback(option, kind, 100.0, k, extreme, 0.5, 0.05, 0.01, 0.3, monitoring=1e-20) == \
            pytest.approx(continuous, abs=1e-6)


def test_lookback_refuses_an_extremum_on_the_wrong_side_and_a_strike_on_a_floating_one():
    with pytest.raises(ValueError, match="running maximum"):
        lookback("call", "fixed", 100.0, 100.0, 95.0, 1.0, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError, match="strike None"):
        lookback("call", "floating", 100.0, 100.0, 95.0, 1.0, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError, match="monitoring"):
        lookback("put", "floating", 100.0, None, 105.0, 1.0, 0.05, 0.0, 0.2, monitoring="daily")


def quantlib_values() -> dict:
    """The frozen `QL` values recomputed by QuantLib (scratch: a Python with QuantLib 1.43)."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    count = ql.Actual365Fixed()

    def handle(rate):
        return ql.YieldTermStructureHandle(ql.FlatForward(today, rate, count))

    def vol(v):
        return ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), v, count))

    def process(s, r, q, v):
        return ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(s)), handle(q), handle(r), vol(v))

    def kind(option):
        return ql.Option.Call if option == "call" else ql.Option.Put

    found = {"lookback": [], "margrabe": [], "kirk": []}
    for args, _ in QL["lookback"]:
        option, style, s, k, extreme, days, r, q, v = args
        exercise = ql.EuropeanExercise(today + days)
        if style == "floating":
            made = ql.ContinuousFloatingLookbackOption(extreme, ql.FloatingTypePayoff(kind(option)), exercise)
            made.setPricingEngine(ql.AnalyticContinuousFloatingLookbackEngine(process(s, r, q, v)))
        else:
            made = ql.ContinuousFixedLookbackOption(extreme, ql.PlainVanillaPayoff(kind(option), k), exercise)
            made.setPricingEngine(ql.AnalyticContinuousFixedLookbackEngine(process(s, r, q, v)))
        found["lookback"].append((args, made.NPV()))
    for args, _ in QL["margrabe"]:
        s1, s2, days, r, q1, q2, v1, v2, rho = args
        made = ql.MargrabeOption(1, 1, ql.EuropeanExercise(today + days))
        made.setPricingEngine(ql.AnalyticEuropeanMargrabeEngine(process(s1, r, q1, v1), process(s2, r, q2, v2), rho))
        found["margrabe"].append((args, made.NPV()))
    for args, _ in QL["kirk"]:
        f1, f2, k, days, r, v1, v2, rho, option = args
        legs = [ql.BlackProcess(ql.QuoteHandle(ql.SimpleQuote(f)), handle(r), vol(v)) for f, v in ((f1, v1), (f2, v2))]
        made = ql.BasketOption(ql.SpreadBasketPayoff(ql.PlainVanillaPayoff(kind(option), k)),
                               ql.EuropeanExercise(today + days))
        made.setPricingEngine(ql.KirkEngine(legs[0], legs[1], rho))
        found["kirk"].append((args, made.NPV()))
    return found


if __name__ == "__main__":
    fresh = quantlib_values()
    print("QL frozen values match QuantLib:", all(
        np.allclose([v for _, v in fresh[key]], [v for _, v in QL[key]], rtol=1e-14, atol=0) for key in QL))
