"""Lane lib-rates1's rate derivatives on the rates shelf (runs/lanes/lib-rates1-prompt.md; syllabus rows 47, 73, 83,
84, 85): the futures convexity adjustment, the compounded overnight rate, caplet vol stripping, the CMS convexity
adjustment and the LIBOR market model's vols. References: Hull's Example 6.4 (Options, Futures, and Other
Derivatives); the Federal Reserve Bank of New York's published SOFR, SOFR Averages and SOFR Index (2020, below);
closed-form identities: Hull-White's futures rate as the expectation of the period's rate at E[r] (Brigo and
Mercurio (3.37)), the round trip of caplet stripping, the linear swap-rate model's CMS rate in closed form (E^A[S^2]
under Black or Bachelier), the abcd integral in closed form (Rebonato 2002). Every input is this file's own or the
published data named; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy import integrate
from scipy.optimize import brentq

import qf

pytestmark = pytest.mark.fast


# ------------------------------------------------------------------------------------------------ futures convexity


def adjust(**changes):
    args = dict(model="ho_lee", futures_price=94.0, start=8.0, end=8.25, sigma=0.012, mean_reversion=0.0,
                quote_basis="simple_act360")
    return qf.futures_convexity_adjustment(**{**args, **changes})


def test_hulls_eurodollar_example():
    """Hull, Example 6.4: a quote of 94 is 6% ACT/360 quarterly, 6.038% continuous ACT/365; sigma 0.012, t1 = 8, t2 =
    8.25 adjust it by 0.475%, a forward of 5.563%."""
    found = adjust()
    assert found["futures_rate"] == pytest.approx(0.06, abs=1e-15, rel=0)
    assert round(found["futures_continuous"] * 100, 3) == 6.038 and round(found["adjustment"] * 100, 3) == 0.475
    assert round(found["forward_continuous"] * 100, 3) == 5.563
    assert found["futures_continuous"] == pytest.approx(365 / 90 * math.log(1.015), rel=1e-14)
    back = found["forward_rate"]
    assert 365 / 90 * math.log1p(back * 90 / 360) == pytest.approx(found["forward_continuous"], rel=1e-13)
    plain = adjust(quote_basis="continuous")
    assert plain["futures_continuous"] == plain["forward_rate"] + plain["adjustment"] == pytest.approx(0.06)
    assert plain["adjustment"] == pytest.approx(0.5 * 0.012 ** 2 * 8 * 8.25, rel=1e-14)


def test_hull_white_is_the_expected_rate_at_the_mean_short_rate_and_tends_to_ho_lee():
    """Hull-White's period rate R = -ln P(t1, t2) / tau is linear in r(t1), so its expectation is R at E[r(t1)] = f(0,
    t1) + sigma^2 / (2 a^2) (1 - e^{-a t1})^2 (Brigo and Mercurio (3.37)); the forward is ln(P(0, t1) / P(0, t2)) / tau.
    On a flat curve their gap is the adjustment, by qf.hull_white_zcb; a -> 0 gives Ho-Lee's; a zero vol adds nothing."""
    a, sigma, f, t1, t2 = 0.08, 0.011, 0.04, 3.0, 3.25
    mean = f + sigma ** 2 / (2 * a ** 2) * (1 - math.exp(-a * t1)) ** 2
    bond = qf.hull_white_zcb(short_rate=mean, t=t1, maturity=t2, a=a, sigma=sigma, discount_t=math.exp(-f * t1),
                             discount_maturity=math.exp(-f * t2), forward_t=f)
    expected = -math.log(bond) / (t2 - t1) - f
    found = adjust(model="hull_white", start=t1, end=t2, sigma=sigma, mean_reversion=a, quote_basis="continuous")
    assert found["adjustment"] == pytest.approx(expected, rel=1e-10)
    near = adjust(model="hull_white", mean_reversion=1e-7)["adjustment"]
    assert near == pytest.approx(adjust()["adjustment"], rel=1e-5)
    assert adjust(sigma=0.0)["adjustment"] == 0.0 and adjust(model="hull_white", mean_reversion=0.1, sigma=0.0)[
        "adjustment"] == 0.0
    assert adjust(model="hull_white", mean_reversion=0.1)["adjustment"] < adjust()["adjustment"]   # reversion damps it


@pytest.mark.parametrize("bad, match", [(dict(model="ho_lee", mean_reversion=0.1), "mean_reversion"),
                                        (dict(model="hull_white", mean_reversion=0.0), "mean_reversion"),
                                        (dict(start=1.0, end=1.0), "start"), (dict(quote_basis="act360"), "quote_basis")])
def test_futures_convexity_refuses_a_bad_argument_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        adjust(**bad)


# ------------------------------------------------------------------------------------------------ overnight


SOFR = {   # FRBNY SOFR in percent by date, 2020 (markets.newyorkfed.org, retrieved 10-12): Good Friday and Memorial Day
    #        had none
    "03-13": 1.1, "03-16": 0.26, "03-17": 0.54, "03-18": 0.1, "03-19": 0.06, "03-20": 0.04, "03-23": 0.02,
    "03-24": 0.01, "03-25": 0.01, "03-26": 0.01, "03-27": 0.01, "03-30": 0.01, "03-31": 0.01, "04-01": 0.01,
    "04-02": 0.01, "04-03": 0.01, "04-06": 0.01, "04-07": 0.01, "04-08": 0.01, "04-09": 0.01,
    "05-15": 0.05, "05-18": 0.04, "05-19": 0.04, "05-20": 0.01, "05-21": 0.02, "05-22": 0.04, "05-26": 0.06,
    "05-27": 0.06, "05-28": 0.06, "05-29": 0.06, "06-01": 0.06, "06-02": 0.07, "06-03": 0.06, "06-04": 0.07,
    "06-05": 0.07, "06-08": 0.07, "06-09": 0.08, "06-10": 0.07, "06-11": 0.08, "06-12": 0.08, "06-15": 0.09,
    "06-16": 0.09, "06-17": 0.09, "06-18": 0.09, "06-19": 0.09, "06-22": 0.08, "06-23": 0.07, "06-24": 0.08,
    "06-25": 0.09, "06-26": 0.08, "06-29": 0.08}
AVERAGES = {"2020-04-13": 0.11667, "2020-06-15": 0.058, "2020-06-30": 0.078}   # FRBNY 30-day SOFR Average, percent
INDEX = {"2020-06-01": 1.04142634, "2020-06-30": 1.04149230}                    # FRBNY SOFR Index


def compounded(start, end, **changes):
    dates = sorted(f"2020-{key}" for key in SOFR)
    args = dict(fixing_dates=dates, fixings=[SOFR[d[5:]] / 100 for d in dates], start=start, end=end,
                day_count="ACT/360", lookback=0, lockout=0, observation="lag")
    return qf.compounded_overnight_rate(**{**args, **changes})


@pytest.mark.parametrize("published", sorted(AVERAGES))
def test_the_fed_s_30_day_sofr_averages(published):
    """The 30-day SOFR Average on a date is SOFR compounded over the 30 calendar days before it, a weekend or holiday
    at the start taking the preceding business day's rate, published to five decimals in percent."""
    end = np.datetime64(published)
    found = compounded(str(end - np.timedelta64(30, "D")), published)
    assert found["days"] == 30 and round(found["rate"] * 100, 5) == AVERAGES[published]


def test_the_sofr_index_ratio_is_the_growth():
    found = compounded("2020-06-01", "2020-06-30")
    assert found["growth"] == pytest.approx(INDEX["2020-06-30"] / INDEX["2020-06-01"], abs=2e-8, rel=0)   # 8-decimal index
    assert found["rate"] == pytest.approx((found["growth"] - 1) * 360 / 29, rel=1e-14)


def test_a_constant_rate_compounds_daily_and_a_weekend_counts_three_days_at_simple_interest():
    dates = ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15"]
    flat = qf.compounded_overnight_rate(fixing_dates=dates[:4], fixings=[0.05] * 4, start="2024-01-08",
                                        end="2024-01-12", day_count="ACT/360", lookback=0, lockout=0, observation="lag")
    assert flat["rate"] == pytest.approx(((1 + 0.05 / 360) ** 4 - 1) * 360 / 4, rel=1e-14)
    week = qf.compounded_overnight_rate(fixing_dates=dates, fixings=[0.05] * 6, start="2024-01-08", end="2024-01-16",
                                        day_count="ACT/365F", lookback=0, lockout=0, observation="lag")
    assert week["days"] == 8 and week["growth"] == pytest.approx((1 + 0.05 / 365) ** 5 * (1 + 3 * 0.05 / 365))


DAYS = ["2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06",
        "2024-03-07", "2024-03-08", "2024-03-11"]
RATES = [0.0500, 0.0501, 0.0502, 0.0503, 0.0504, 0.0505, 0.0506, 0.0507, 0.0508, 0.0509, 0.0510]


def by_hand(rates, weights, days, basis=360.0):
    return (np.prod(1 + np.asarray(rates) * np.asarray(weights) / basis) - 1) * basis / days


def window(**changes):
    args = dict(fixing_dates=DAYS, fixings=RATES, start="2024-03-01", end="2024-03-08", day_count="ACT/360",
                lookback=0, lockout=0, observation="lag")
    return qf.compounded_overnight_rate(**{**args, **changes})["rate"]


def test_a_lookback_takes_the_rate_two_business_days_earlier_weighted_by_the_interest_period():
    """[03-01, 03-08): business days 03-01 (Friday, three days), 03-04 to 03-07. Two days back: 02-28, 02-29, 03-01,
    03-04, 03-05's rates on those weights."""
    assert window() == pytest.approx(by_hand(RATES[4:9], [3, 1, 1, 1, 1], 7), rel=1e-14)
    assert window(lookback=2) == pytest.approx(by_hand(RATES[2:7], [3, 1, 1, 1, 1], 7), rel=1e-14)


def test_an_observation_shift_moves_the_period_and_its_weights():
    """Shifted two business days: [02-28, 03-06), whose days weight the rates: 02-28, 02-29 one each, 03-01 three,
    03-04 and 03-05 one; D is the shifted period's 7 days. It differs from the lag, whose Friday rate is 02-28's."""
    found = window(lookback=2, observation="shift")
    assert found == pytest.approx(by_hand(RATES[2:7], [1, 1, 3, 1, 1], 7), rel=1e-14)
    assert found != pytest.approx(window(lookback=2), rel=1e-9)
    assert window(lookback=0, observation="shift") == window()


def test_a_lockout_holds_the_last_business_days_at_the_cut_off_rate():
    """Lockout 2: the period's last two business days (03-06, 03-07) both take 03-06's rate, the rate cut-off date;
    a lockout of 1 changes nothing."""
    assert window(lockout=2) == pytest.approx(by_hand([RATES[4], RATES[5], RATES[6], RATES[7], RATES[7]],
                                                       [3, 1, 1, 1, 1], 7), rel=1e-14)
    assert window(lockout=1) == window()
    assert window(lockout=3, lookback=1) == pytest.approx(by_hand([RATES[3], RATES[4], RATES[5], RATES[5], RATES[5]],
                                                                   [3, 1, 1, 1, 1], 7), rel=1e-14)


@pytest.mark.parametrize("bad, match", [(dict(lookback=9), "lookback"), (dict(day_count="ACT/365"), "day_count"),
                                        (dict(observation="shifted"), "observation"), (dict(lockout=6), "lockout"),
                                        (dict(start="2024-02-20"), "fixing_dates"), (dict(lookback=-1), "lookback")])
def test_the_overnight_rate_refuses_a_bad_argument_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        window(**bad)


# ------------------------------------------------------------------------------------------------ caplet stripping


TIMES = np.arange(0.0, 5.0, 0.5)
TAU = np.full(TIMES.size, 0.5)
FORWARDS = 0.03 + 0.004 * np.sqrt(TIMES)
DISCOUNTS = np.exp(-0.032 * (TIMES + TAU))
CAPS = np.array([1.0, 2.0, 3.0, 5.0])


def caplet(kind, forward, strike, expiry, vol, tau, discount):
    if kind == "black":
        return tau * qf.black76_price(option="call", forward=forward, strike=strike, expiry=expiry, vol=vol,
                                      discount_factor=discount)
    width = vol * math.sqrt(expiry)
    d = (forward - strike) / width
    return tau * discount * ((forward - strike) * 0.5 * math.erfc(-d / math.sqrt(2)) + width * math.exp(-d * d / 2)
                             / math.sqrt(2 * math.pi))


def flat_vols(kind, vols, strikes, used):
    """Each cap's flat vol for caplet vols `vols` (one per used caplet), solved here by Brent."""
    found = []
    for cap, strike in zip(CAPS, strikes):
        members = [i for i in used if TIMES[i] + TAU[i] <= cap + 1e-12]
        price = sum(caplet(kind, FORWARDS[i], strike, TIMES[i], vols[used.index(i)], TAU[i], DISCOUNTS[i])
                    for i in members)
        found.append(brentq(lambda v: sum(caplet(kind, FORWARDS[i], strike, TIMES[i], v, TAU[i], DISCOUNTS[i])
                                           for i in members) - price, 1e-6, 3.0, xtol=1e-15))
    return np.array(found)


def strip(kind, flat, strikes, interpolation, first="excluded"):
    return qf.caplet_vol_strip(cap_maturities=CAPS, cap_vols=flat, strikes=strikes, fixing_times=TIMES, accruals=TAU,
                               forwards=FORWARDS, discount_factors=DISCOUNTS, first_caplet=first,
                               interpolation=interpolation, vol_kind=kind)


@pytest.mark.parametrize("kind, scale", [("black", 1.0), ("normal", 0.006)])
def test_stripping_round_trips_piecewise_constant_caplet_vols(kind, scale):
    """Caplet vols constant within each cap's new segment price the caps; their flat vols, solved here, strip back to
    them (Hull, "Caps and floors": flat and spot vols)."""
    used = list(range(1, TIMES.size))
    segment = np.searchsorted(CAPS, TIMES[used] + TAU[used] - 1e-12)
    vols = scale * np.array([0.25, 0.22, 0.20, 0.18])[segment]
    strikes = np.array([0.031, 0.033, 0.034, 0.035])
    found = strip(kind, flat_vols(kind, vols, strikes, used), strikes, "constant")
    assert found["caplet_vols"] == pytest.approx(vols, rel=1e-9) and found["fixing_times"] == pytest.approx(TIMES[1:])


def test_stripping_round_trips_vols_linear_between_cap_maturities():
    used = list(range(1, TIMES.size))
    knots = {1.0: 0.21}            # the first cap's caplets share one vol; after, linear to each segment's last
    vols, last_t, last_v = [], None, None
    for i in used:
        end = TIMES[i] + TAU[i]
        cap = CAPS[np.searchsorted(CAPS, end - 1e-12)]
        if cap == 1.0:
            vols.append(knots[1.0])
            last_t, last_v = TIMES[i], knots[1.0]
    ends = {2.0: 0.19, 3.0: 0.24, 5.0: 0.2}
    for cap, value in ends.items():
        new = [i for i in used if CAPS[np.searchsorted(CAPS, TIMES[i] + TAU[i] - 1e-12)] == cap]
        stop = TIMES[new[-1]]
        vols += [last_v + (value - last_v) * (TIMES[i] - last_t) / (stop - last_t) for i in new]
        last_t, last_v = stop, value
    strikes = np.full(4, 0.033)
    found = strip("black", flat_vols("black", np.array(vols), strikes, used), strikes, "linear")
    assert found["caplet_vols"] == pytest.approx(vols, rel=1e-9)


def test_constant_caplet_vols_give_equal_flat_vols_and_strip_to_themselves():
    used = list(range(1, TIMES.size))
    flat = flat_vols("black", np.full(len(used), 0.23), np.full(4, 0.034), used)
    assert flat == pytest.approx(0.23, rel=1e-12)
    found = strip("black", np.full(4, 0.23), 0.034, "linear")
    assert found["caplet_vols"] == pytest.approx(0.23, rel=1e-12)
    every = strip("black", np.full(4, 0.23), 0.025, "constant", first="included")
    assert every["fixing_times"][0] == 0.0 and every["caplet_vols"][1:] == pytest.approx(0.23, rel=1e-12)
    intrinsic = TAU[0] * DISCOUNTS[0] * (FORWARDS[0] - 0.025)     # the period fixed today, in the money
    assert every["cap_prices"] - strip("black", np.full(4, 0.23), 0.025, "constant")["cap_prices"] == pytest.approx(
        intrinsic, rel=1e-12)


def test_a_cap_adding_no_caplet_is_refused():
    with pytest.raises(ValueError, match="adds no caplet"):
        qf.caplet_vol_strip(cap_maturities=[1.0, 1.2], cap_vols=[0.2, 0.2], strikes=0.03, fixing_times=TIMES,
                            accruals=TAU, forwards=FORWARDS, discount_factors=DISCOUNTS, first_caplet="excluded",
                            interpolation="constant", vol_kind="black")


# ------------------------------------------------------------------------------------------------ CMS


def cms(**changes):
    args = dict(method="replication", mapping="linear", forward_swap_rate=0.04, annuity=4.3, discount_payment=0.95,
                expiry=2.0, payment_time=2.0, fixed_periods=5, frequency=1, vol=0.25, vol_kind="black")
    return qf.cms_convexity_adjustment(**{**args, **changes})


def test_the_linear_mapping_is_exact_in_closed_form_under_black_and_bachelier():
    """Linear swap-rate model: a(S) = a1 S + a2, a2 = q/n, a(S0) = P(0, t_p)/A(0), so the CMS rate is S0 + (A0/P) a1
    Var^A(S), Var = S0^2 (e^{v^2 T} - 1) for Black and v^2 T for Bachelier (Hagan 2003; Andersen and Piterbarg 16.6);
    the replication's integrals and Hagan's approximation both give it."""
    s0, a0, p, t, n, q = 0.04, 4.3, 0.95, 2.0, 5, 1
    a1 = (p / a0 - q / n) / s0
    for kind, vol, variance in (("black", 0.25, s0 ** 2 * math.expm1(0.25 ** 2 * t)), ("normal", 0.008, 0.008 ** 2 * t)):
        expected = s0 + a0 / p * a1 * variance
        for method in ("replication", "approximation"):
            found = cms(method=method, vol=vol, vol_kind=kind)
            assert found["cms_rate"] == pytest.approx(expected, rel=1e-9) and found["adjustment"] == pytest.approx(
                expected - s0, rel=1e-6), (kind, method)


def test_a_constant_mapping_or_a_vanishing_vol_adds_nothing():
    """a(S) constant where P(0, t_p)/A(0) = q/n: no adjustment; a tiny vol leaves the forward."""
    assert abs(cms(discount_payment=4.3 / 5)["adjustment"]) < 1e-15
    for mapping in ("linear", "standard"):
        assert abs(cms(mapping=mapping, vol=1e-6)["adjustment"]) < 1e-12, mapping


@pytest.mark.parametrize("method", ["replication", "approximation"])
@pytest.mark.parametrize("mapping", ["linear", "standard"])
@pytest.mark.parametrize("kind", ["black", "normal"])
def test_zero_vol_has_no_cms_adjustment(method, mapping, kind):
    """Hagan's replication formula: a deterministic swap rate is unchanged under any measure."""
    for vol in (0.0, ([0.01, 0.04, 0.1], [0.0, 0.0, 0.0])):
        found = cms(method=method, mapping=mapping, vol_kind=kind, vol=vol)
        assert found["cms_rate"] == 0.04 and found["adjustment"] == 0.0


@pytest.mark.parametrize("vol", [-0.01, float("nan"), float("inf"),
                                  ([0.01, 0.04], [0.0, -0.01]), ([0.01, float("nan")], [0.1, 0.1])])
def test_cms_rejects_nonfinite_or_negative_vol(vol):
    with pytest.raises(ValueError, match="vol"):
        cms(vol=vol)


def hagan_g(s, n=5, q=1, lag=0.0):
    """Hagan's standard (flat-yield) mapping, the payment bond over the annuity as a function of the swap rate."""
    return s * (1 + s / q) ** (-lag) / (1 - (1 + s / q) ** (-n))


@pytest.mark.parametrize("lag, discount", [(0.0, 0.95), (0.5, 0.93)])
def test_hagans_standard_mapping_replicates_the_annuity_measure_expectation(lag, discount):
    """Replication is the annuity-measure expectation (A0/P) E^A[S a(S)], a(S) = G(S) P / (A0 G(S0)), S lognormal
    under the annuity measure: integrated here against its density. Hagan's approximation keeps G's first derivative
    alone, S0 + G'(S0)/G(S0) Var^A(S); a payment after the fixing takes some of the adjustment back (Hagan 2003)."""
    s0, a0, t, vol = 0.04, 4.3, 2.0, 0.25
    g0 = hagan_g(s0, lag=lag)
    width = vol * math.sqrt(t)

    def density(s):
        return math.exp(-0.5 * ((math.log(s / s0) + 0.5 * width ** 2) / width) ** 2) / (s * width * math.sqrt(2 * math.pi))

    mean = integrate.quad(lambda s: s * hagan_g(s, lag=lag) / g0 * density(s), 1e-9, 2.0, epsabs=1e-15, epsrel=1e-13,
                          limit=400)[0]
    found = cms(mapping="standard", payment_time=t + lag, discount_payment=discount)
    assert found["cms_rate"] == pytest.approx(mean, rel=1e-9) and found["adjustment"] > 0
    slope = (hagan_g(s0 + 1e-6, lag=lag) - hagan_g(s0 - 1e-6, lag=lag)) / 2e-6 / g0
    approx = cms(method="approximation", mapping="standard", payment_time=t + lag, discount_payment=discount)
    assert approx["cms_rate"] == pytest.approx(s0 + slope * s0 ** 2 * math.expm1(vol ** 2 * t), rel=1e-9)
    assert approx["adjustment"] == pytest.approx(found["adjustment"], rel=0.1)
    if lag:
        assert found["adjustment"] < cms(mapping="standard")["adjustment"]


def test_a_flat_smile_given_by_strikes_is_the_flat_vol_and_a_skew_moves_the_rate():
    flat = cms(mapping="standard")["cms_rate"]
    assert cms(mapping="standard", vol=([0.01, 0.04, 0.1], [0.25, 0.25, 0.25]))["cms_rate"] == pytest.approx(flat,
                                                                                                           rel=1e-12)
    assert cms(mapping="standard", vol=([0.01, 0.04, 0.1], [0.25, 0.25, 0.35]))["cms_rate"] > flat


# ------------------------------------------------------------------------------------------------ LMM


FIX = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
ABCD = {"a": 0.05, "b": 0.09, "c": 0.44, "d": 0.11}


def abcd_integral(a, b, c, d, horizon):
    """int_0^T [(a + b u) e^{-c u} + d]^2 du in closed form (Rebonato 2002, the abcd integral)."""
    k = 2 * c
    square = (a * a / k + 2 * a * b / k ** 2 + 2 * b * b / k ** 3
              - math.exp(-k * horizon) * ((a + b * horizon) ** 2 / k + 2 * b * (a + b * horizon) / k ** 2 + 2 * b * b
                                          / k ** 3))
    linear = a / c + b / c ** 2 - math.exp(-c * horizon) * ((a + b * horizon) / c + b / c ** 2)
    return square + 2 * d * linear + d * d * horizon


def lmm(**changes):
    args = dict(fixing_times=FIX, accruals=np.ones(5), forwards=[0.04, 0.042, 0.044, 0.045, 0.046],
                discount_factors=np.exp(-0.043 * (FIX + 1)), abcd=ABCD, correlation=0.1, swaption=(2.0, 5.0))
    return qf.lmm_vols(**{**args, **changes})


def test_caplet_vols_integrate_the_abcd_vol_back():
    found = lmm()["caplet_vols"]
    expected = [math.sqrt(abcd_integral(*ABCD.values(), t) / t) for t in FIX]
    assert found == pytest.approx(expected, rel=1e-13)
    flat = lmm(abcd={"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.2})
    assert flat["caplet_vols"] == pytest.approx(0.2, rel=1e-14)


def test_rebonatos_swaption_vol_by_quadrature_and_its_limits():
    """Rebonato's formula computed here by adaptive quadrature; one forward gives its caplet vol; flat vols perfectly
    correlated give that vol; a decorrelating beta lowers it."""
    found = lmm()
    f = np.array([0.04, 0.042, 0.044, 0.045, 0.046])
    p = np.exp(-0.043 * (FIX + 1))
    span = [1, 2, 3]
    w = p[span] / p[span].sum()
    s = w @ f[span]
    a, b, c, d = ABCD.values()
    total = 0.0
    for i in span:
        for j in span:
            moment = integrate.quad(lambda t: ((a + b * (FIX[i] - t)) * math.exp(-c * (FIX[i] - t)) + d)
                                    * ((a + b * (FIX[j] - t)) * math.exp(-c * (FIX[j] - t)) + d), 0, 2.0,
                                    epsabs=1e-15, epsrel=1e-13)[0]
            total += w[span.index(i)] * w[span.index(j)] * f[i] * f[j] * math.exp(-0.1 * abs(FIX[i] - FIX[j])) * moment
    assert found["swap_rate"] == pytest.approx(s, rel=1e-14)
    assert found["swaption_vol"] == pytest.approx(math.sqrt(total / s ** 2 / 2.0), rel=1e-12)
    one = lmm(swaption=(3.0, 4.0))
    assert one["swaption_vol"] == pytest.approx(one["caplet_vols"][2], rel=1e-13)
    flat = lmm(abcd={"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.2}, correlation=np.ones((5, 5)))
    assert flat["swaption_vol"] == pytest.approx(0.2, rel=1e-13)
    assert lmm(correlation=1.0)["swaption_vol"] < found["swaption_vol"] < lmm(correlation=0.0)["swaption_vol"]


@pytest.mark.parametrize("bad, match", [(dict(swaption=(2.5, 5.0)), "swaption"), (dict(abcd={"a": 0.1}), "abcd"),
                                        (dict(correlation=np.eye(4)), "correlation")])
def test_lmm_refuses_a_bad_argument_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        lmm(**bad)


@pytest.mark.parametrize("correlation", [-0.1, float("nan"), float("inf"),
                                         np.full((5, 5), float("nan")),
                                         np.array([[1.0, 1.1], [1.1, 1.0]]),
                                         np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])])
def test_lmm_rejects_invalid_correlation(correlation):
    """A covariance cannot have negative variance; the final matrix has eigenvalue -0.8."""
    changes = {}
    if np.ndim(correlation) == 2 and correlation.shape[0] != 5:
        count = correlation.shape[0]
        changes = dict(fixing_times=FIX[:count], accruals=np.ones(count), forwards=np.full(count, 0.04),
                       discount_factors=np.exp(-0.04 * (FIX[:count] + 1)), swaption=None)
    with pytest.raises(ValueError, match="correlation"):
        lmm(correlation=correlation, **changes)


def test_lmm_accepts_a_singular_correlation_matrix():
    """Identical Brownian drivers give the valid, singular, all-ones correlation matrix."""
    found = lmm(correlation=np.ones((5, 5)))
    assert found["swaption_vol"] == pytest.approx(lmm(correlation=0.0)["swaption_vol"], rel=1e-14)


@pytest.mark.parametrize("rho", [-1.0, -1.0 - 5e-13])
def test_lmm_offsetting_forward_exposures_have_zero_swaption_vol(rho):
    """Rebonato: equal weighted vols with opposite drivers cancel; tolerated correlation roundoff cannot make a NaN."""
    found = qf.lmm_vols(fixing_times=[1.0, 2.0], accruals=[1.0, 1.0], forwards=[0.04, 0.04],
                       discount_factors=[1.0, 1.0], abcd={"a": 0.0, "b": 0.0, "c": 0.44, "d": 0.2},
                       correlation=[[1.0, rho], [rho, 1.0]], swaption=(1.0, 3.0))
    assert found["swaption_vol"] == pytest.approx(0.0, rel=0, abs=1e-12)
