"""agent/qf/fx.py (runs/lanes/lib7-prompt.md, ranks 12 and 22 of docs/research/library-syllabus-1011.md): the four
delta conventions of Reiswich and Wystup (2010) against the Garman-Kohlhagen price differentiated here and their own
identities, strikes from deltas by round trip, and the smile from ATM, risk-reversal and butterfly quotes by round
trip from a known smile. Every input is this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from test_qf_bs import check_cards  # first: it puts agent/ on the path outside pytest too
import qf
import qf.fx as fx

pytestmark = pytest.mark.fast

PAIR = dict(spot=1.10, maturity=0.75, domestic_rate=0.045, foreign_rate=0.03)
TYPES = ("spot", "forward", "spot_premium_adjusted", "forward_premium_adjusted")


def forward_of(spot, maturity, domestic_rate, foreign_rate):
    return spot * math.exp((domestic_rate - foreign_rate) * maturity)


def gk(option, strike, vol, spot=PAIR["spot"]):
    """Garman-Kohlhagen: Black-Scholes with the foreign rate as the yield, DOM per unit of FOR."""
    return qf.bs_price(option, spot, strike, PAIR["maturity"], PAIR["domestic_rate"], PAIR["foreign_rate"], vol)


def test_the_fx_cards_keep_the_format_and_their_examples_run():
    check_cards(fx)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("strike", [0.95, 1.10, 1.21])
def test_each_delta_is_its_definition_on_the_garman_kohlhagen_price(option, strike):
    """Spot delta dV/dS (a central difference); forward delta e^{r_f T} times it; premium-adjusted spot delta the
    spot delta less V/S; premium-adjusted forward delta e^{r_f T} times that."""
    vol, h = 0.11, 1e-5
    spot = (gk(option, strike, vol, PAIR["spot"] + h) - gk(option, strike, vol, PAIR["spot"] - h)) / (2 * h)
    grow = math.exp(PAIR["foreign_rate"] * PAIR["maturity"])
    expected = {"spot": spot, "forward": spot * grow, "spot_premium_adjusted": spot - gk(option, strike, vol) / PAIR["spot"]}
    expected["forward_premium_adjusted"] = expected["spot_premium_adjusted"] * grow
    for kind in TYPES:
        assert fx.fx_delta(option, strike=strike, vol=vol, delta_type=kind, **PAIR) == pytest.approx(expected[kind],
                                                                                                    abs=1e-9)


def test_the_premium_adjusted_spot_delta_discounts_k_over_s_at_the_domestic_rate():
    """phi e^{-r_f T} (K/F) N(phi d2) = phi e^{-r_d T} (K/S) N(phi d2) (Reiswich and Wystup, 2010, eq. 9)."""
    k, vol, t = 1.17, 0.09, PAIR["maturity"]
    f = forward_of(**PAIR)
    d2 = (math.log(f / k) - vol * vol * t / 2) / (vol * math.sqrt(t))
    found = fx.fx_delta("call", strike=k, vol=vol, delta_type="spot_premium_adjusted", **PAIR)
    assert found == pytest.approx(math.exp(-PAIR["domestic_rate"] * t) * k / PAIR["spot"] * norm.cdf(d2), rel=1e-13)


def test_call_less_put_and_the_delta_neutral_strikes():
    vol, t = 0.1, PAIR["maturity"]
    f = forward_of(**PAIR)
    k = np.array([0.98, 1.1, 1.2])

    def both(kind, strike):
        return (fx.fx_delta("call", strike=strike, vol=vol, delta_type=kind, **PAIR),
                fx.fx_delta("put", strike=strike, vol=vol, delta_type=kind, **PAIR))

    call, put = both("forward", k)
    assert call - put == pytest.approx(np.ones(3), abs=1e-15)
    call, put = both("spot", k)
    assert call - put == pytest.approx(np.full(3, math.exp(-PAIR["foreign_rate"] * t)), abs=1e-15)
    call, put = both("forward_premium_adjusted", k)
    assert call - put == pytest.approx(k / f, abs=1e-15)
    for kind, strike in [("spot", f * math.exp(vol * vol * t / 2)), ("forward", f * math.exp(vol * vol * t / 2)),
                         ("spot_premium_adjusted", f * math.exp(-vol * vol * t / 2)),
                         ("forward_premium_adjusted", f * math.exp(-vol * vol * t / 2))]:
        assert sum(both(kind, strike)) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("bad, match", [(dict(delta_type="premium"), "delta_type"), (dict(option="straddle"), "option"),
                                        (dict(vol=0.0), "vol"), (dict(strike=-1.0), "strike")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        fx.fx_delta(**{**dict(option="call", strike=1.1, vol=0.1, delta_type="spot", **PAIR), **bad})


# ------------------------------------------------------------------------------------------------ strike from delta


def smile(strike):
    """A smile of this file's own: a parabola in log-moneyness, a skew down to the right."""
    x = math.log(strike / forward_of(**PAIR))
    return 0.09 - 0.05 * x + 0.6 * x * x


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("option, delta", [("call", 0.1), ("call", 0.25), ("call", 0.45), ("put", -0.1),
                                           ("put", -0.25), ("put", -0.45)])
def test_the_strike_gives_its_delta_back_flat_and_on_a_smile(kind, option, delta):
    for vol in (0.095, smile):
        k = fx.strike_from_delta(option, delta, vol=vol, delta_type=kind, **PAIR)
        at = vol(k) if callable(vol) else vol
        assert fx.fx_delta(option, strike=k, vol=at, delta_type=kind, **PAIR) == pytest.approx(delta, abs=1e-12)


def test_the_forward_and_spot_strikes_are_reiswich_and_wystups_closed_form():
    vol, t = 0.095, PAIR["maturity"]
    f, sd = forward_of(**PAIR), vol * math.sqrt(t)
    for option, phi, delta in [("call", 1, 0.25), ("put", -1, -0.1)]:
        expected = f * math.exp(-phi * norm.ppf(phi * delta) * sd + sd * sd / 2)
        assert fx.strike_from_delta(option, delta, vol=vol, delta_type="forward", **PAIR) == pytest.approx(expected,
                                                                                                         rel=1e-14)
        spot = delta * math.exp(-PAIR["foreign_rate"] * t)
        assert fx.strike_from_delta(option, spot, vol=vol, delta_type="spot", **PAIR) == pytest.approx(expected,
                                                                                                     rel=1e-13)


def test_a_premium_adjusted_call_delta_takes_the_larger_of_its_two_strikes():
    """(K/F) N(d2) rises then falls in K: at 0.25 two strikes give it; the larger lies between the maximum, where
    vol sqrt(T) N(d2) = n(d2) (Reiswich and Wystup, 2010, section 3), and the unadjusted strike."""
    vol, t = 0.3, 2.0
    pair = {**PAIR, "maturity": t}
    f, sd = forward_of(**pair), vol * math.sqrt(t)
    k = fx.strike_from_delta("call", 0.25, vol=vol, delta_type="forward_premium_adjusted", **pair)
    unadjusted = fx.strike_from_delta("call", 0.25, vol=vol, delta_type="forward", **pair)
    from scipy.optimize import brentq
    d2_peak = brentq(lambda d2: sd * norm.cdf(d2) - norm.pdf(d2), -10, 10)
    peak = f * math.exp(-d2_peak * sd - sd * sd / 2)
    smaller = brentq(lambda x: x / f * norm.cdf((math.log(f / x) - sd * sd / 2) / sd) - 0.25, 1e-6 * f, peak)
    assert smaller < peak < k < unadjusted
    assert fx.fx_delta("call", strike=smaller, vol=vol, delta_type="forward_premium_adjusted", **pair) == \
        pytest.approx(0.25, abs=1e-12)
    top = peak / f * norm.cdf(d2_peak)
    with pytest.raises(ValueError, match="maximum"):
        fx.strike_from_delta("call", top + 1e-4, vol=vol, delta_type="forward_premium_adjusted", **pair)


@pytest.mark.parametrize("bad, match", [(dict(delta=0.25, option="put"), "not a put"), (dict(delta=1.2), "not a call"),
                                        (dict(vol=lambda k: -0.1, delta_type="forward_premium_adjusted"), "vol"),
                                        (dict(delta_type="forward_adjusted"), "delta_type")])
def test_strike_from_delta_refuses_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        fx.strike_from_delta(**{**dict(option="call", delta=0.25, vol=0.1, delta_type="spot", **PAIR), **bad})


# ------------------------------------------------------------------------------------------------ the smile


KNOWN = (0.6, -0.05, 0.09)       # a smile of this file's own: vol = 0.09 - 0.05 x + 0.6 x^2, x = ln(K/F)


def known_vol(strike):
    x = math.log(strike / forward_of(**PAIR))
    return KNOWN[2] + KNOWN[1] * x + KNOWN[0] * x * x


def quotes_of(delta_type, atm="delta_neutral", delta=0.25):
    """The known smile's ATM vol (its delta-neutral strike a fixed point), pillars, RR and smile butterfly."""
    f, t = forward_of(**PAIR), PAIR["maturity"]
    k_atm = f
    if atm == "delta_neutral":
        for _ in range(100):
            k_atm = f * math.exp((-0.5 if "premium" in delta_type else 0.5) * known_vol(k_atm) ** 2 * t)
    elif atm == "spot":
        k_atm = PAIR["spot"]
    k_c = fx.strike_from_delta("call", delta, vol=known_vol, delta_type=delta_type, **PAIR)
    k_p = fx.strike_from_delta("put", -delta, vol=known_vol, delta_type=delta_type, **PAIR)
    vol_atm, vol_c, vol_p = known_vol(k_atm), known_vol(k_c), known_vol(k_p)
    return dict(k_atm=k_atm, k_c=k_c, k_p=k_p, atm_vol=vol_atm, rr=vol_c - vol_p, bf=(vol_c + vol_p) / 2 - vol_atm)


def market_strangle_of(quotes, delta_type, delta=0.25):
    """The market strangle's butterfly the known smile implies: one vol for both legs whose own-delta strikes price
    the strangle as the smile prices those strikes."""
    from scipy.optimize import brentq

    def gap(fly):
        vol = quotes["atm_vol"] + fly
        k_c = fx.strike_from_delta("call", delta, vol=vol, delta_type=delta_type, **PAIR)
        k_p = fx.strike_from_delta("put", -delta, vol=vol, delta_type=delta_type, **PAIR)
        flat = gk("call", k_c, vol) + gk("put", k_p, vol)
        return flat - gk("call", k_c, known_vol(k_c)) - gk("put", k_p, known_vol(k_p))

    return brentq(gap, -0.02, 0.05, xtol=1e-15)


@pytest.mark.parametrize("delta_type", TYPES)
@pytest.mark.parametrize("butterfly_type", ["smile_strangle", "market_strangle"])
def test_quotes_made_from_a_known_smile_return_it(delta_type, butterfly_type):
    quotes = quotes_of(delta_type)
    fly = quotes["bf"] if butterfly_type == "smile_strangle" else market_strangle_of(quotes, delta_type)
    reads = np.array([0.95, 1.05, 1.12, 1.2, 1.3])
    found = fx.smile_from_quotes(**PAIR, atm_vol=quotes["atm_vol"], risk_reversal=quotes["rr"], butterfly=fly,
                                 delta=0.25, delta_type=delta_type, atm="delta_neutral", butterfly_type=butterfly_type,
                                 strikes=reads)
    assert (found["call_strike"], found["put_strike"]) == pytest.approx((quotes["k_c"], quotes["k_p"]), rel=1e-10)
    assert found["atm_strike"] == pytest.approx(quotes["k_atm"], rel=1e-10)
    assert found["smile_butterfly"] == pytest.approx(quotes["bf"], abs=1e-11)
    assert found["vols"] == pytest.approx([known_vol(k) for k in reads], abs=1e-10)
    assert found["call_vol"] - found["put_vol"] == pytest.approx(quotes["rr"], abs=1e-14)


def test_the_pillars_sit_at_their_deltas_and_the_atm_rules():
    found = fx.smile_from_quotes(**PAIR, atm_vol=0.1, risk_reversal=0.015, butterfly=0.004, delta=0.1,
                                 delta_type="forward", atm="forward", butterfly_type="smile_strangle", strikes=None)
    assert (found["call_vol"], found["put_vol"]) == pytest.approx((0.1 + 0.004 + 0.0075, 0.1 + 0.004 - 0.0075))
    assert fx.fx_delta("call", strike=found["call_strike"], vol=found["call_vol"], delta_type="forward",
                       **PAIR) == pytest.approx(0.1, abs=1e-12)
    assert fx.fx_delta("put", strike=found["put_strike"], vol=found["put_vol"], delta_type="forward",
                       **PAIR) == pytest.approx(-0.1, abs=1e-12)
    assert found["atm_strike"] == pytest.approx(forward_of(**PAIR)) and found["vols"] is None
    spot = fx.smile_from_quotes(**PAIR, atm_vol=0.1, risk_reversal=0.0, butterfly=0.0, delta=0.25, delta_type="spot",
                                atm="spot", butterfly_type="smile_strangle", strikes=[0.9, 1.3])
    assert spot["atm_strike"] == PAIR["spot"] and spot["vols"] == pytest.approx([0.1, 0.1], abs=1e-14)


@pytest.mark.parametrize("bad, match", [(dict(atm="delta_neutral_straddle"), "atm"), (dict(delta=25), "delta"),
                                        (dict(butterfly_type="broker"), "butterfly_type"),
                                        (dict(risk_reversal=-0.3), "not positive")])
def test_the_smile_refuses_by_name(bad, match):
    args = dict(**PAIR, atm_vol=0.1, risk_reversal=-0.01, butterfly=0.003, delta=0.25, delta_type="spot",
                atm="delta_neutral", butterfly_type="smile_strangle", strikes=None)
    with pytest.raises(ValueError, match=match):
        fx.smile_from_quotes(**{**args, **bad})
