"""Lane lib-risk1 (runs/lanes/lib-risk1-prompt.md), the `credit` shelf's new bonds and cva's midpoint reading: the
defaultable bond against QuantLib 1.43's RiskyBondEngine (frozen), against Lando's closed forms for a zero-coupon bond
under each recovery (Credit Risk Modeling, 2004, 5.7) and at zero hazard the riskless price; the Tsiveriotis-Fernandes
convertible against Black-Scholes in its no-spread limit, a two-step tree written out by hand, and its bounds; cva
with exposure and discount given at each interval's midpoint (Hull's) by hand. Synthetic values alone; no unit's."""
import math

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import qf

pytestmark = pytest.mark.fast

FLAT = dict(interpolation="log_linear_discount")


def bond(**kw):
    t = kw.pop("payment_times")
    r = kw.pop("r")
    args = dict(payment_times=t, discount_times=[t[-1]], discount_factors=[math.exp(-r * t[-1])], **FLAT) | kw
    return qf.risky_bond_price(**args)


def test_the_midpoint_recovery_of_face_is_quantlibs_risky_bond_engine():
    """QuantLib 1.43, RiskyBondEngine on a FixedRateBond: 100 face, 6% coupons (Actual365Fixed accrual) on periods of
    364 days from 15 January 2025, a flat 4% continuous curve and a flat 3% hazard, recovery 40%: NPV
    99.96974872104282. The engine pays recovery at each coupon period's midpoint date; on 364-day periods that date is
    the period's time midpoint (scratch/lib-risk1/tools/ql_risky2.py)."""
    t = np.arange(1, 6) * 364 / 365
    found = bond(payment_times=t, r=0.04, coupons=np.full(5, 6.0 * 364 / 365), face=100.0, hazard_times=[t[-1]],
                 hazard_rates=[0.03], recovery_rate=0.4, recovery="face", recovery_timing="midpoint")
    assert found["price"] == pytest.approx(99.96974872104282, rel=1e-13)


@pytest.mark.parametrize("recovery", ["face", "treasury", "market_value"])
def test_a_zero_coupon_bond_under_each_recovery_is_landos_closed_form(recovery):
    """Lando (2004), 5.7, flat r and lambda, maturity T, recovery R: of face paid at default e^{-(r+lambda)T} + R lambda
    (1 - e^{-(r+lambda)T}) / (r + lambda); of treasury e^{-rT} (S + R (1 - S)); of market value e^{-(r + (1 - R)
    lambda) T} (Duffie and Singleton 1999)."""
    r, lam, rec, t = 0.04, 0.03, 0.4, 5.0
    found = bond(payment_times=[t], r=r, coupons=[0.0], face=1.0, hazard_times=[t], hazard_rates=[lam],
                 recovery_rate=rec, recovery=recovery, recovery_timing="at_default")
    s = math.exp(-lam * t)
    expected = {"face": math.exp(-(r + lam) * t) + rec * lam / (r + lam) * (1 - math.exp(-(r + lam) * t)),
                "treasury": math.exp(-r * t) * (s + rec * (1 - s)),
                "market_value": math.exp(-(r + (1 - rec) * lam) * t)}[recovery]
    assert found["price"] == pytest.approx(expected, rel=1e-13)
    assert found["riskless_price"] == pytest.approx(math.exp(-r * t), rel=1e-14)
    spread = -math.log(found["price"] / found["riskless_price"]) / t
    if recovery == "market_value":
        assert spread == pytest.approx((1 - rec) * lam, rel=1e-12)


def test_zero_hazard_is_the_riskless_price_and_the_timings_order_as_the_discounting_says():
    t = [0.5, 1.0, 1.5, 2.0]
    args = dict(payment_times=t, r=0.05, coupons=[3.0] * 4, face=100.0, hazard_times=[1.0, 2.0])
    riskless = sum(3.0 * math.exp(-0.05 * x) for x in t) + 100 * math.exp(-0.1)
    for recovery in ("face", "treasury", "market_value"):
        found = bond(**args, hazard_rates=[0.0, 0.0], recovery_rate=0.4, recovery=recovery,
                     recovery_timing="at_default")
        assert found["price"] == pytest.approx(riskless, rel=1e-14) and found["expected_loss"] == pytest.approx(0, abs=1e-12)
    prices = [bond(**args, hazard_rates=[0.02, 0.05], recovery_rate=0.4, recovery="face", recovery_timing=timing)["price"]
              for timing in ("at_default", "midpoint", "period_end")]
    assert prices[2] < prices[1] and abs(prices[0] - prices[1]) < 1e-3 * prices[0]
    full = bond(**args, hazard_rates=[0.02, 0.05], recovery_rate=1.0, recovery="treasury", recovery_timing="at_default")
    assert full["price"] == pytest.approx(riskless, rel=1e-14)
    s = bond(**args, hazard_rates=[0.02, 0.05], recovery_rate=0.4, recovery="face", recovery_timing="at_default")
    assert s["survival"] == pytest.approx([math.exp(-0.01), math.exp(-0.02), math.exp(-0.02 - 0.025),
                                           math.exp(-0.02 - 0.05)], rel=1e-14)


def test_the_exact_recovery_integral_is_the_limit_of_ever_finer_midpoints():
    t = np.array([1.0, 2.0, 3.0])
    exact = bond(payment_times=t, r=0.03, coupons=[0.0] * 3, face=1.0, hazard_times=[1.0, 3.0], hazard_rates=[0.04, 0.1],
                 recovery_rate=0.5, recovery="face", recovery_timing="at_default")["price"]
    fine = np.linspace(0.001, 3.0, 3000)
    mid = bond(payment_times=fine, r=0.03, coupons=np.zeros(fine.size), face=1.0, hazard_times=[1.0, 3.0],
               hazard_rates=[0.04, 0.1], recovery_rate=0.5, recovery="face", recovery_timing="midpoint")["price"]
    assert mid == pytest.approx(exact, rel=1e-8)


@pytest.mark.parametrize("interpolation", ["log_linear_discount", "linear_zero", "cubic_log_discount"])
def test_recovery_at_default_integrates_the_selected_nonflat_discount_curve(interpolation):
    """Lando (2004), 5.7: recovery of face is R F integral D(t) lambda(t) S(t) dt. Independent Gauss-Legendre
    quadrature uses the selected curve's defining interpolation, including curved log discounts."""
    knots = np.array([1.0, 3.0, 5.0])
    zeros = np.array([0.01, 0.05, 0.10])
    log_discounts = -zeros * knots
    x, weights = np.polynomial.legendre.leggauss(40)
    curve = CubicSpline(np.r_[0.0, knots], np.r_[0.0, log_discounts])
    lost = 0.0
    for start, stop in zip([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 5.0]):
        times = start + (x + 1.0) * (stop - start) / 2.0
        if interpolation == "linear_zero":
            discount = np.exp(-times * np.interp(times, knots, zeros))
        elif interpolation == "log_linear_discount":
            discount = np.exp(np.interp(times, np.r_[0.0, knots], np.r_[0.0, log_discounts]))
        else:
            discount = np.exp(curve(times))
        intensity = 0.03 if stop <= 2.0 else 0.1
        survival = np.exp(-0.03 * np.minimum(times, 2.0) - 0.1 * np.maximum(times - 2.0, 0.0))
        lost += (stop - start) / 2.0 * float(weights @ (discount * intensity * survival))
    expected = 100.0 * math.exp(-0.5 - 0.36) + 40.0 * lost
    found = qf.risky_bond_price(payment_times=[5.0], coupons=[0.0], face=100.0, hazard_times=[2.0, 5.0],
                                hazard_rates=[0.03, 0.1], recovery_rate=0.4, recovery="face",
                                recovery_timing="at_default", discount_times=knots,
                                discount_factors=np.exp(log_discounts), interpolation=interpolation)
    assert found["price"] == pytest.approx(expected, rel=1e-13)


# ------------------------------------------------------------------------------------------------ convertibles


CB = dict(spot=50.0, vol=0.3, rate=0.05, dividend_yield=0.0, maturity=2.0, face=100.0, conversion_ratio=2.0,
          last_coupon_time=0.0, price_basis="clean")


def test_without_spread_or_coupons_a_convertible_is_a_bond_and_a_black_scholes_call():
    """max(F, ratio S_T) = F + ratio max(S_T - F/ratio, 0); on a share paying nothing early conversion never pays, so
    the tree converges to F e^{-rT} + ratio C_BS(K = F/ratio) (Ingersoll 1977)."""
    found = qf.convertible_tree(**CB, credit_spread=0.0, coupon_times=[], coupons=[], calls=None, puts=None,
                                steps=1600)
    limit = 100 * math.exp(-0.1) + 2 * qf.bs_price("call", 50.0, 50.0, 2.0, 0.05, 0.0, 0.3)
    assert found["price"] == pytest.approx(limit, abs=5e-3)


def test_a_two_step_tree_by_hand():
    """Tsiveriotis and Fernandes (1998): U discounted at r + s, V - U at r, conversion where parity beats holding; the
    share's dividend yield q in the tree's drift."""
    s0, sigma, r, q, s, t = 100.0, 0.2, 0.05, 0.02, 0.03, 1.0
    dt = t / 2
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    nodes = {(2, 2): s0 * u * u, (2, 1): s0, (2, 0): s0 * d * d}
    value, cash = {}, {}
    for j in range(3):
        parity = nodes[(2, j)]
        value[(2, j)], cash[(2, j)] = (parity, 0.0) if parity > 100.0 else (100.0, 100.0)
    for i in (1, 0):
        for j in range(i + 1):
            up, down = (i + 1, j + 1), (i + 1, j)
            equity = math.exp(-r * dt) * (p * (value[up] - cash[up]) + (1 - p) * (value[down] - cash[down]))
            cash[(i, j)] = math.exp(-(r + s) * dt) * (p * cash[up] + (1 - p) * cash[down])
            value[(i, j)] = equity + cash[(i, j)]
            parity = s0 * u ** j * d ** (i - j)
            if parity > value[(i, j)]:
                value[(i, j)], cash[(i, j)] = parity, 0.0
    found = qf.convertible_tree(spot=s0, vol=sigma, rate=r, dividend_yield=q, credit_spread=s, maturity=t,
                                face=100.0, conversion_ratio=1.0, coupon_times=[], coupons=[], last_coupon_time=0.0,
                                calls=None, puts=None, price_basis="clean", steps=2)
    assert found["price"] == pytest.approx(value[(0, 0)], rel=1e-14)
    assert found["cash_part"] == pytest.approx(cash[(0, 0)], rel=1e-14)
    assert found["delta"] == pytest.approx((value[(1, 1)] - value[(1, 0)]) / (s0 * u - s0 * d), rel=1e-14)


def test_the_node_rules_by_hand_a_coupon_forgone_on_conversion_and_a_call_answered_by_converting():
    """One step of vol 0.1. From 95 the up node's shares are worth 104.99: the holder takes face plus the last coupon,
    105, since converting forgoes the coupon. Today a call at 105 with shares worth 110 (the down node in cash) is
    answered by converting (V = parity, U = 0); with shares worth 100 and holding worth more than 102, a call at 102 is
    taken in cash."""
    args = dict(vol=0.1, rate=0.05, dividend_yield=0.0, credit_spread=0.02, maturity=1.0, face=100.0,
                conversion_ratio=1.0, coupon_times=[1.0], coupons=[5.0], last_coupon_time=0.0, puts=None,
                price_basis="dirty", steps=1)
    held = qf.convertible_tree(spot=95.0, calls=None, **args)
    assert held["price"] == pytest.approx(105.0 * math.exp(-0.07), rel=1e-14) and held["equity_part"] == 0.0
    converted = qf.convertible_tree(spot=110.0, calls=[(0.0, 0.0, 105.0)], **args)
    assert converted["price"] == 110.0 and converted["cash_part"] == 0.0
    cash = qf.convertible_tree(spot=100.0, calls=[(0.0, 0.0, 102.0)], **args)
    assert cash["price"] == 102.0 and cash["cash_part"] == 102.0


@pytest.mark.parametrize("coupon_time", [0.5, 1.0])
def test_a_dirty_call_on_a_coupon_date_caps_the_total_cash_payment(coupon_time):
    """Tsiveriotis-Fernandes (1998), p. 99: add coupons then reapply the exercise constraints. A dirty call of 103
    includes the coupon, while a clean call adds its accrued coupon; the stock stays far below conversion."""
    args = dict(spot=1.0, vol=0.1, rate=0.05, dividend_yield=0.0, credit_spread=0.02, maturity=1.0, face=100.0,
                conversion_ratio=1.0, coupon_times=[coupon_time, 1.0] if coupon_time < 1.0 else [1.0],
                coupons=[5.0, 5.0] if coupon_time < 1.0 else [5.0], last_coupon_time=0.0,
                calls=[(coupon_time, coupon_time, 103.0)], puts=None, steps=2)
    dirty = qf.convertible_tree(**args, price_basis="dirty")
    clean = qf.convertible_tree(**args, price_basis="clean")
    assert dirty["price"] == pytest.approx(103.0 * math.exp(-0.07 * coupon_time), rel=1e-14)
    floor = 105.0 * math.exp(-0.07) + (5.0 * math.exp(-0.035) if coupon_time < 1.0 else 0.0)
    assert clean["price"] == pytest.approx(floor, rel=1e-14)


@pytest.mark.parametrize("price_basis, payment", [("dirty", 110.0), ("clean", 115.0)])
def test_a_put_on_a_coupon_date_is_applied_to_the_price_in_its_stated_basis(price_basis, payment):
    """Tsiveriotis-Fernandes (1998), p. 99: the dirty put already includes the coupon, a clean put adds it."""
    found = qf.convertible_tree(spot=1.0, vol=0.1, rate=0.05, dividend_yield=0.0, credit_spread=0.02, maturity=1.0,
                                face=100.0, conversion_ratio=1.0, coupon_times=[0.5, 1.0], coupons=[5.0, 5.0],
                                last_coupon_time=0.0, calls=None, puts=[(0.5, 0.5, 110.0)], price_basis=price_basis,
                                steps=2)
    assert found["price"] == pytest.approx(payment * math.exp(-0.035), rel=1e-14)


def test_the_price_bounds_and_how_calls_puts_spread_and_the_price_basis_move_it():
    base = dict(CB, coupon_times=[0.5, 1.0, 1.5, 2.0], coupons=[2.0] * 4, steps=400)
    plain = qf.convertible_tree(**base, credit_spread=0.02, calls=None, puts=None)
    assert plain["price"] >= max(plain["parity"], plain["bond_floor"])
    assert plain["bond_floor"] == pytest.approx(100 * math.exp(-0.07 * 2) + sum(2.0 * math.exp(-0.07 * x)
                                                                                 for x in (0.5, 1.0, 1.5, 2.0)))
    assert plain["conversion_premium"] == pytest.approx(plain["price"] / 100.0 - 1)
    wider = qf.convertible_tree(**base, credit_spread=0.06, calls=None, puts=None)
    called = qf.convertible_tree(**base, credit_spread=0.02, calls=[(0.5, 2.0, 103.0)], puts=None)
    put = qf.convertible_tree(**base, credit_spread=0.02, calls=None, puts=[(1.0, 1.0, 101.0)])
    assert wider["price"] < plain["price"] and called["price"] < plain["price"] < put["price"]
    dirty = qf.convertible_tree(**dict(base, price_basis="dirty"), credit_spread=0.02, calls=[(0.5, 2.0, 103.0)],
                                puts=None)
    assert dirty["price"] < called["price"]
    deep = qf.convertible_tree(**dict(base, spot=1.0), credit_spread=0.02, calls=None, puts=None)
    assert deep["price"] == pytest.approx(deep["bond_floor"], rel=1e-9)
    high = qf.convertible_tree(**dict(base, spot=500.0), credit_spread=0.02, calls=None, puts=None)
    waiting = sum(2.0 * math.exp(-0.07 * x) for x in (0.5, 1.0, 1.5))     # converted at maturity, last coupon forgone
    assert high["price"] == pytest.approx(high["parity"] + waiting, rel=1e-9)     # the lowest nodes stay cash
    assert high["cash_part"] == pytest.approx(waiting, rel=1e-6)


@pytest.mark.slow
def test_the_trees_delta_settles_and_is_the_prices_slope():
    """A bump's value on a tree jumps as nodes cross the coupon steps, so the slope is read on a fine tree."""
    base = dict(CB, coupon_times=[0.5, 1.0, 1.5, 2.0], coupons=[2.0] * 4, steps=400)
    plain = qf.convertible_tree(**base, credit_spread=0.02, calls=None, puts=None)
    fine = dict(base, steps=2000)
    up = qf.convertible_tree(**dict(fine, spot=50.5), credit_spread=0.02, calls=None, puts=None)["price"]
    down = qf.convertible_tree(**dict(fine, spot=49.5), credit_spread=0.02, calls=None, puts=None)["price"]
    finer = qf.convertible_tree(**fine, credit_spread=0.02, calls=None, puts=None)["delta"]
    assert plain["delta"] == pytest.approx(finer, abs=1e-3) and finer == pytest.approx(up - down, rel=0.01)


# ------------------------------------------------------------------------------------------------ cva


def test_cva_with_exposure_and_discount_given_at_each_intervals_midpoint():
    """Hull, "Credit value adjustment": CVA = (1 - R) sum_i v_i q_i, v_i the discounted expected exposure at the
    midpoint of interval i and q_i its default probability; here written out by hand on three paths."""
    values = np.array([[2.0, -1.0], [0.5, 3.0], [1.0, 0.5]])
    times, d_mid, s_c = [0.0, 0.5, 1.0], [0.99, 0.97], [1.0, 0.99, 0.975]
    found = qf.cva(values=values, times=times, discount_factors=d_mid, counterparty_survival=s_c,
                   counterparty_recovery=0.4, own_survival=None, own_recovery=None, timing="given_at_midpoints",
                   kind="unilateral", collateral=None)
    ee = np.maximum(values, 0).mean(axis=0)
    expected = 0.6 * (d_mid[0] * ee[0] * (1.0 - 0.99) + d_mid[1] * ee[1] * (0.99 - 0.975))
    assert found["cva"] == pytest.approx(expected, rel=1e-14) and found["ee"] == pytest.approx(ee)
    s_o = [1.0, 0.995, 0.985]
    both = qf.cva(values=values, times=times, discount_factors=d_mid, counterparty_survival=s_c,
                  counterparty_recovery=0.4, own_survival=s_o, own_recovery=0.3, timing="given_at_midpoints",
                  kind="first_to_default", collateral=None)
    nee = np.maximum(-values, 0).mean(axis=0)
    assert both["cva"] == pytest.approx(0.6 * (0.99 * ee[0] * 0.01 * 1.0 + 0.97 * ee[1] * 0.015 * 0.995), rel=1e-14)
    assert both["dva"] == pytest.approx(0.7 * (0.99 * nee[0] * 0.005 * 1.0 + 0.97 * nee[1] * 0.01 * 0.99), rel=1e-14)
    with pytest.raises(ValueError):
        qf.cva(values=values, times=times, discount_factors=[1.0, 0.99, 0.97], counterparty_survival=s_c,
               counterparty_recovery=0.4, own_survival=None, own_recovery=None, timing="given_at_midpoints",
               kind="unilateral", collateral=None)


def test_a_constant_exposure_gives_the_same_charge_read_at_ends_midpoints_or_given_midpoints_of_flat_discounting():
    s_c = [1.0, 0.98, 0.95]
    common = dict(times=[0.0, 1.0, 2.0], counterparty_survival=s_c, counterparty_recovery=0.4, own_survival=None,
                  own_recovery=None, kind="unilateral", collateral=None)
    end = qf.cva(values=np.full((2, 3), 5.0), discount_factors=[1.0, 1.0, 1.0], timing="end", **common)["cva"]
    mid = qf.cva(values=np.full((2, 3), 5.0), discount_factors=[1.0, 1.0, 1.0], timing="midpoint", **common)["cva"]
    given = qf.cva(values=np.full((2, 2), 5.0), discount_factors=[1.0, 1.0], timing="given_at_midpoints",
                   **common)["cva"]
    assert end == pytest.approx(0.6 * 5.0 * 0.05) and mid == pytest.approx(end) and given == pytest.approx(end)
