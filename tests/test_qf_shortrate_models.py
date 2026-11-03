"""Lane lib-rates1's short-rate shelf (runs/lanes/lib-rates1-prompt.md; syllabus rows 87, 106, 107): the G2++ bond and
swaption, the CIR++ bond and bond option, the Vasicek fit to a yield curve, and the eight functions moved from rates.
References: QuantLib 1.43's G2 model (discountBond, G2SwaptionEngine) and ExtendedCoxIngersollRoss
(discountBondOption), frozen in `QL_*` (`quantlib_values` recomputes them in a Python with QuantLib); closed-form
identities: G2++ at eta = 0 is Hull-White (qf.hull_white_zcb, qf.jamshidian_swaption), a one-payment swaption is
G2++'s zero-bond put in closed form (Brigo and Mercurio 4.2.4), payer less receiver is the forward swap, CIR++ on
CIR's own curve is CIR (qf.short_rate_bond_price), and a curve made from known Vasicek parameters fits back to them.
Every input is this file's own; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest
from scipy.stats import norm

import qf
import qf.rates
import qf.shortrate

pytestmark = pytest.mark.fast

G2 = dict(a=0.5, b=0.05, sigma=0.01, eta=0.008, rho=-0.7)
QL_G2_BONDS = [   # (t, T, x, y, P(0, t), P(0, T)): G2(curve, a, sigma, b, eta, rho).discountBond(t, T, [x, y])
    ((1.0, 5.0, 0.01, -0.005, 0.9772731062493097, 0.8564503252025419), 0.8768610785791477),
    ((2.0, 10.0, -0.02, 0.015, 0.9493444723771842, 0.7047266685898254), 0.6978813743792132),
    ((0.5, 0.75, 0.003, 0.001, 0.9895522435635838, 0.9836373840843852), 0.993073493841643)]
QL_G2_SWAPTIONS = [   # (option, strike, expiry, payment times, accruals, P(0, T), P(0, T_i)): G2SwaptionEngine(10, 400)
    (("payer", 0.03, 1.0027397260273974, [2.0027397260273974, 3.0027397260273974, 4.002739726027397],
      [1.0, 1.0, 1.0], 0.9772009047898221, [0.9492612458252067, 0.9193607270875851, 0.8886315281928122]),
     0.009377593850869309),
    (("receiver", 0.03, 1.0027397260273974, [2.0027397260273974, 3.0027397260273974, 4.002739726027397],
      [1.0, 1.0, 1.0], 0.9772009047898221, [0.9492612458252067, 0.9193607270875851, 0.8886315281928122]),
     0.0035258222870275025),
    (("payer", 0.035, 2.0027397260273974, [3.0027397260273974, 4.002739726027397, 5.005479452054795,
                                           6.005479452054795, 7.005479452054795],
      [1.0, 1.0, 1.0027397260273974, 1.0, 1.0], 0.9492612458252067,
      [0.9193607270875851, 0.8886315281928122, 0.8562697165751528, 0.825162169836638, 0.7935959535182495]),
     0.016496174978956286),
    (("receiver", 0.04, 5.005479452054795, [6.005479452054795, 7.005479452054795], [1.0, 1.0], 0.8562697165751528,
      [0.825162169836638, 0.7935959535182495]), 0.009384073585449727)]
CIR = dict(x0=0.03, kappa=0.3, theta=0.04, sigma=0.08)
QL_CIRPP = [   # (option, strike, T, S, P(0, T), P(0, S)): ExtendedCoxIngersollRoss(curve, theta, k, sigma, x0)
    (("call", 0.85, 1.0, 5.0, 0.9772731062493097, 0.8564503252025419), 0.028007945379621257),
    (("put", 0.85, 1.0, 5.0, 0.9772731062493097, 0.8564503252025419), 0.002239760488992548),
    (("call", 0.9, 2.0, 4.0, 0.9493444723771842, 0.8887179359605768), 0.03513519441868529),
    (("put", 0.78, 3.0, 10.0, 0.9194463701578058, 0.7047266685898254), 0.02006130899364289)]


def swaption(option, strike, expiry, times, accruals, p0, pi, quadrature="gauss_hermite", nodes=64, **params):
    return qf.g2pp_swaption(option=option, strike=strike, expiry=expiry, payment_times=times, accruals=accruals,
                            **{**G2, **params}, discount_expiry=p0, discount_payments=pi, quadrature=quadrature,
                            nodes=nodes)


def test_g2pp_bonds_equal_quantlibs():
    for (t, big_t, x, y, p_t, p_big), expected in QL_G2_BONDS:
        found = qf.g2pp_zcb(x=x, y=y, t=t, maturity=big_t, **G2, discount_t=p_t, discount_maturity=p_big)
        assert found == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize("quadrature, nodes", [("gauss_hermite", 64), ("adaptive", None)])
def test_g2pp_swaptions_equal_quantlibs(quadrature, nodes):
    """QuantLib integrates by Simpson's rule on 400 intervals over ten deviations: the two agree to its error."""
    for args, expected in QL_G2_SWAPTIONS:
        assert swaption(*args, quadrature=quadrature, nodes=nodes) == pytest.approx(expected, rel=0, abs=2e-12), args[:3]


def test_g2pp_reprices_the_curve_and_is_hull_white_without_its_second_factor():
    """At t = 0 the bond is P(0, T). With eta = 0, r = x + phi, phi(t) = f(0, t) + sigma^2 / (2 a^2) (1 - e^{-a t})^2,
    so G2++'s bond is qf.hull_white_zcb at that short rate."""
    assert qf.g2pp_zcb(x=0.0, y=0.0, t=0.0, maturity=7.0, **G2, discount_t=1.0, discount_maturity=0.8) == \
        pytest.approx(0.8, rel=1e-15)
    a, sigma, f, t, big_t, x = 0.5, 0.01, 0.03, 2.0, 6.0, 0.012
    one = {**G2, "eta": 0.0}
    found = qf.g2pp_zcb(x=x, y=0.0, t=t, maturity=big_t, **one, discount_t=math.exp(-f * t),
                        discount_maturity=math.exp(-f * big_t))
    r = x + f + sigma ** 2 / (2 * a ** 2) * (1 - math.exp(-a * t)) ** 2
    hw = qf.hull_white_zcb(short_rate=r, t=t, maturity=big_t, a=a, sigma=sigma, discount_t=math.exp(-f * t),
                           discount_maturity=math.exp(-f * big_t), forward_t=f)
    assert found == pytest.approx(hw, rel=1e-13)


def test_a_g2pp_swaption_tends_to_jamshidians_as_eta_vanishes_and_parity_holds():
    f, expiry, times = 0.03, 1.0, np.array([2.0, 3.0, 4.0])
    p0, pi = math.exp(-f * expiry), np.exp(-f * times)
    hw = qf.jamshidian_swaption(option="payer", strike=0.032, expiry=expiry, payment_times=times, accruals=np.ones(3),
                                a=0.5, sigma=0.01, discount_expiry=p0, discount_payments=pi, forward_expiry=f)["price"]
    near = swaption("payer", 0.032, expiry, times, np.ones(3), p0, pi, "adaptive", None, eta=1e-6, rho=0.0)
    assert near == pytest.approx(hw, rel=1e-7)          # the gap falls as eta^2; the integrand nears a step in x
    for args, _ in QL_G2_SWAPTIONS[:3]:
        _, strike, t0, ts, tau, p_t, p_i = args
        coupons = strike * np.asarray(tau)
        coupons[-1] += 1
        gap = swaption("payer", *args[1:]) - swaption("receiver", *args[1:])
        assert gap == pytest.approx(p_t - coupons @ np.asarray(p_i), abs=1e-14)


def test_a_one_payment_payer_is_the_closed_form_zero_bond_put():
    """Brigo and Mercurio 4.2.4: a zero-bond option in G2++ is Black's with Sigma^2 = sigma^2/(2a^3)(1 - e^{-a
    tau})^2 (1 - e^{-2aT}) + eta^2/(2b^3)(...)^2(...) + 2 rho sigma eta/(ab(a+b))(1 - e^{-a tau})(1 - e^{-b tau})
    (1 - e^{-(a+b)T}); the caplet is (1 + K tau) ZBP(X = 1/(1 + K tau))."""
    a, b, sigma, eta, rho = G2.values()
    big_t, s, k, tau = 1.5, 2.0, 0.04, 0.5
    p_t, p_s = math.exp(-0.03 * big_t), math.exp(-0.031 * s)
    gap = s - big_t
    var = (sigma ** 2 / (2 * a ** 3) * (1 - math.exp(-a * gap)) ** 2 * (1 - math.exp(-2 * a * big_t))
           + eta ** 2 / (2 * b ** 3) * (1 - math.exp(-b * gap)) ** 2 * (1 - math.exp(-2 * b * big_t))
           + 2 * rho * sigma * eta / (a * b * (a + b)) * (1 - math.exp(-a * gap)) * (1 - math.exp(-b * gap))
           * (1 - math.exp(-(a + b) * big_t)))
    width, strike = math.sqrt(var), 1 / (1 + k * tau)
    h = math.log(p_s / (strike * p_t)) / width + width / 2
    put = strike * p_t * norm.cdf(width - h) - p_s * norm.cdf(-h)
    assert swaption("payer", k, big_t, [s], [tau], p_t, [p_s]) == pytest.approx((1 + k * tau) * put, rel=1e-12)


def cirpp(option, factor, t, maturity, expiry, strike, d_t, d_s, d_big):
    return qf.cir_pp_bond_price(option=option, factor=factor, t=t, maturity=maturity, expiry=expiry, strike=strike,
                                **CIR, discount_t=d_t, discount_maturity=d_s, discount_expiry=d_big)


def test_cir_pp_bond_options_equal_quantlibs():
    for (option, strike, big_t, s, p_t, p_s), expected in QL_CIRPP:
        found = cirpp(option, CIR["x0"], 0.0, s, big_t, strike, 1.0, p_s, p_t)
        assert found["price"] == pytest.approx(expected, rel=0, abs=1e-12) and found["feller_ratio"] == pytest.approx(3.75)


def cir(option, r, maturity, expiry=None, strike=None):
    return qf.short_rate_bond_price(model="cir", option=option, short_rate=r, maturity=maturity, kappa=0.3,
                                    theta=0.04, sigma=0.08, expiry=expiry, strike=strike)["price"]


def test_cir_pp_on_cirs_own_curve_is_cir_and_today_it_reprices_the_market_curve():
    """A curve that is CIR's own needs no shift: bonds and options are CIR's, today and at a later time t from the
    factor then; any curve is repriced at t = 0; call less put is P(0, S) - K P(0, T) (Brigo and Mercurio 3.9)."""
    own = {u: cir("none", 0.03, u) for u in (1.0, 2.0, 5.0)}
    found = cirpp("put", 0.03, 0.0, 5.0, 2.0, 0.86, 1.0, own[5.0], own[2.0])
    assert found["price"] == pytest.approx(cir("put", 0.03, 5.0, 2.0, 0.86), rel=1e-13)
    assert found["shift_factor"] == pytest.approx(1.0, rel=1e-14)
    later = cirpp("none", 0.045, 1.0, 5.0, None, None, own[1.0], own[5.0], None)
    assert later["price"] == pytest.approx(cir("none", 0.045, 4.0), rel=1e-13)
    assert cirpp("none", 0.03, 0.0, 5.0, None, None, 1.0, 0.82, None)["price"] == pytest.approx(0.82, rel=1e-14)
    call = cirpp("call", 0.03, 0.0, 5.0, 1.0, 0.85, 1.0, 0.86, 0.97)["price"]
    put = cirpp("put", 0.03, 0.0, 5.0, 1.0, 0.85, 1.0, 0.86, 0.97)["price"]
    assert call - put == pytest.approx(0.86 - 0.85 * 0.97, abs=1e-14)


@pytest.mark.parametrize("bad, match", [(dict(option="none", expiry=1.0), "None for the bond"),
                                        (dict(option="call", expiry=6.0), "expiry"), (dict(t=-1.0), "t and maturity"),
                                        (dict(option="swap"), "option")])
def test_cir_pp_refuses_a_bad_argument_by_name(bad, match):
    args = dict(option="call", factor=0.03, t=0.0, maturity=5.0, expiry=1.0, strike=0.85, discount_t=1.0,
                discount_maturity=0.86, discount_expiry=0.97)
    if args.get("option") == "call" and bad.get("option") == "none":
        args.update(strike=None, discount_expiry=None)
    with pytest.raises(ValueError, match=match):
        qf.cir_pp_bond_price(**{**args, **CIR, **bad})


VAS = dict(kappa=0.35, theta=0.05, sigma=0.015)
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def vasicek_yields(r0, compounding):
    price = np.array([qf.short_rate_bond_price(model="vasicek", option="none", short_rate=r0, maturity=m, **VAS,
                                               expiry=None, strike=None)["price"] for m in MATURITIES])
    if compounding == "continuous":
        return -np.log(price) / MATURITIES
    if compounding == "simple":
        return (1 / price - 1) / MATURITIES
    return compounding * (price ** (-1 / (compounding * MATURITIES)) - 1)


@pytest.mark.parametrize("compounding", ["continuous", 2, "simple"])
def test_a_vasicek_curve_fits_back_to_its_parameters(compounding):
    """Brigo and Mercurio 3.2.1: yields made from known parameters, in each compounding, return them."""
    found = qf.vasicek_fit_curve(maturities=MATURITIES, yields=vasicek_yields(0.025, compounding),
                                 compounding=compounding, short_rate=None, weights="equal",
                                 initial={"kappa": 0.2, "theta": 0.04, "sigma": 0.01, "short_rate": 0.03})
    assert [found[k] for k in ("kappa", "theta", "sigma", "short_rate")] == pytest.approx(
        [0.35, 0.05, 0.015, 0.025], rel=1e-6) and found["rmse"] < 1e-12 and found["converged"]


def test_a_fixed_short_rate_is_held_and_weights_move_a_misfit():
    exact = vasicek_yields(0.025, "continuous")
    held = qf.vasicek_fit_curve(maturities=MATURITIES, yields=exact, compounding="continuous", short_rate=0.025,
                                weights=np.linspace(1, 2, 10), initial={"kappa": 0.2, "theta": 0.04, "sigma": 0.01})
    assert held["short_rate"] == 0.025 and [held[k] for k in VAS] == pytest.approx(list(VAS.values()), rel=1e-6)
    bent = exact + 0.0005 * np.sin(np.arange(10))
    plain, heavy = (qf.vasicek_fit_curve(maturities=MATURITIES, yields=bent, compounding="continuous", short_rate=None,
                                         weights=w, initial={**VAS, "short_rate": 0.025})
                    for w in ("equal", np.r_[50.0, np.ones(9)]))
    assert abs(heavy["fitted"][0] - bent[0]) < abs(plain["fitted"][0] - bent[0])


MOVED = ("hull_white_zcb", "hull_white_zcb_option", "hull_white_caplet", "jamshidian_swaption", "hull_white_calibrate",
         "short_rate_tree", "short_rate_bond_price", "cir_fit")


def test_the_moved_functions_keep_both_paths_and_sit_on_the_new_shelf():
    from conftest import ROOT
    from n11 import library as L
    lib = L.scan(ROOT / "agent" / "qf")
    for name in MOVED:
        function = getattr(qf, name)
        assert getattr(qf.rates, name) is function is getattr(qf.shortrate, name), name
        assert lib.find(name).shelf == "shortrate" and name not in qf.rates.__all__, name
    assert len(lib.shelf("shortrate").functions) == 12 and len(lib.shelf("rates").functions) == 7


def quantlib_values() -> dict:
    """The frozen QuantLib values recomputed (a Python with QuantLib 1.43): the curve is a zero curve of continuous
    rates linear between dated nodes, ACT/365F, from 2020-01-15; the swaptions' fixed legs annual and unadjusted."""
    import QuantLib as ql
    today = ql.Date(15, ql.January, 2020)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    nodes = [today + ql.Period(m, ql.Months) for m in (0, 6, 12, 24, 36, 60, 84, 120)]
    curve = ql.ZeroCurve(nodes, [0.02, 0.021, 0.023, 0.026, 0.028, 0.031, 0.033, 0.035], dc, ql.NullCalendar(),
                         ql.Linear(), ql.Continuous)
    curve.enableExtrapolation()
    handle = ql.YieldTermStructureHandle(curve)
    g2 = ql.G2(handle, G2["a"], G2["sigma"], G2["b"], G2["eta"], G2["rho"])
    bonds = [g2.discountBond(t, big_t, ql.Array([x, y])) for (t, big_t, x, y, _, _), _ in QL_G2_BONDS]
    swaptions = []
    for (option, strike, *_), _ in QL_G2_SWAPTIONS:
        years, tenor = {0.03: (1, 3), 0.035: (2, 5), 0.04: (5, 2)}[strike]
        start = today + ql.Period(years, ql.Years)
        end = start + ql.Period(tenor, ql.Years)
        fixed = ql.Schedule(start, end, ql.Period(ql.Annual), ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
                            ql.DateGeneration.Forward, False)
        floating = ql.Schedule(start, end, ql.Period(ql.Semiannual), ql.NullCalendar(), ql.Unadjusted,
                               ql.Unadjusted, ql.DateGeneration.Forward, False)
        index = ql.IborIndex("flat", ql.Period(6, ql.Months), 0, ql.USDCurrency(), ql.NullCalendar(), ql.Unadjusted,
                             False, dc, handle)
        kind = ql.VanillaSwap.Payer if option == "payer" else ql.VanillaSwap.Receiver
        made = ql.Swaption(ql.VanillaSwap(kind, 1.0, fixed, strike, dc, floating, index, 0.0, dc),
                           ql.EuropeanExercise(start))
        made.setPricingEngine(ql.G2SwaptionEngine(g2, 10.0, 400))
        swaptions.append(made.NPV())
    ecir = ql.ExtendedCoxIngersollRoss(handle, CIR["theta"], CIR["kappa"], CIR["sigma"], CIR["x0"])
    options = [ecir.discountBondOption(ql.Option.Call if o == "call" else ql.Option.Put, k, big_t, s)
               for (o, k, big_t, s, _, _), _ in QL_CIRPP]
    return {"bonds": bonds, "swaptions": swaptions, "options": options}


if __name__ == "__main__":
    found = quantlib_values()
    print("frozen values match QuantLib:",
          np.allclose(found["bonds"], [v for _, v in QL_G2_BONDS], rtol=1e-14, atol=0),
          np.allclose(found["swaptions"], [v for _, v in QL_G2_SWAPTIONS], rtol=1e-14, atol=0),
          np.allclose(found["options"], [v for _, v in QL_CIRPP], rtol=1e-14, atol=0))
