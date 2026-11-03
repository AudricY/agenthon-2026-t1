"""qf.impvol.li_nonatm_vol (runs/lanes/lib3-prompt.md): Li's (2005) implied-vol approximation at any moneyness, as
Orlando and Taglialatela (2017) restate it, against its own algebra (each branch the root of its equation, the switch
at rho = 1.4, li_atm_vol at the money), the library's Brenner-Subrahmanyam and Corrado-Miller approximations, and an
exact implied vol (qf.implied_vol, bisection to the price's precision) over a grid of moneyness, vols and maturities.
Every parameter is chosen freely; no unit's value appears."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import qf  # noqa: E402
import qf.impvol as iv  # noqa: E402
from test_qf_bs import check_card  # noqa: E402

pytestmark = pytest.mark.fast

VOLS = np.array([0.1, 0.2, 0.3, 0.45, 0.6])[:, None, None]
TIMES = np.array([0.1, 0.25, 0.5, 1.0, 2.0])[None, :, None]
ETA = np.exp(np.linspace(-0.3, 0.3, 61))[None, None, :]      # K~ / S~
S = 100.0


def grid():
    """Calls priced by qf.bs_price on the grid (r = q = 0, so S~ = S and K~ = K), broadcast to one shape."""
    price = qf.bs_price("call", S, S * ETA, TIMES, 0.0, 0.0, VOLS)
    full = np.ones_like(price)
    return price, S * ETA * full, TIMES * full, VOLS * full


def test_the_card_keeps_the_library_format_and_its_example_runs():
    check_card("li_nonatm_vol")


def test_at_the_money_it_is_li_atm_vol_and_closer_than_brenner_subrahmanyam_and_corrado_miller():
    """At K~ = S~ (eta = 1, rho = 0) the cubic branch is li_atm_vol's formula on the same price, exactly; its error
    there is below Brenner and Subrahmanyam's first-order one, which Corrado and Miller's reduces to at the money."""
    price = qf.bs_price("call", S, S, TIMES, 0.02, 0.02, VOLS)        # r = q: the money forward is the spot
    pv = S * np.exp(-0.02 * TIMES) * np.ones_like(price)
    li = iv.li_nonatm_vol(price, pv, pv, TIMES * np.ones_like(price))
    assert li == pytest.approx(qf.li_atm_vol(price, pv, TIMES * np.ones_like(price)), rel=1e-14)
    bs = qf.brenner_subrahmanyam_vol(price, pv, TIMES)
    cm = qf.corrado_miller_vol(price, pv, pv, TIMES, "corrado_miller")
    assert cm == pytest.approx(bs, rel=1e-13)
    assert np.all(np.abs(li - VOLS) < np.abs(bs - VOLS)) and np.max(np.abs(li - VOLS)) < 1e-3


def test_each_branch_is_the_root_of_its_own_equation_and_the_switch_is_at_rho_1_4():
    """With c = C / S~, eta = K~ / S~, alpha = sqrt(2 pi)(2c + eta - 1)/(1 + eta), rho = |eta - 1| / c^2 and
    v = vol sqrt(T): where rho <= 1.4, alpha = v - v^3/24 (the cubic); where rho > 1.4, v^2 - alpha v + (eta - 1)^2 /
    (1 + eta) = 0 with v the larger root. The grid holds both branches and points either side of the switch."""
    price, strike, t, _ = grid()
    v = iv.li_nonatm_vol(price, S, strike, t) * np.sqrt(t)
    c, eta = price / S, strike / S
    alpha = math.sqrt(2.0 * math.pi) * (2.0 * c + eta - 1.0) / (1.0 + eta)
    rho = np.abs(eta - 1.0) / c ** 2
    cubic, quadratic = np.isfinite(v) & (rho <= 1.4), np.isfinite(v) & (rho > 1.4)
    assert cubic.sum() > 100 and quadratic.sum() > 500
    assert np.sum(cubic & (rho > 1.2)) > 10 and np.sum(quadratic & (rho < 1.6)) > 10
    assert np.max(np.abs(alpha - (v - v ** 3 / 24.0))[cubic]) < 1e-12
    constant = (eta - 1.0) ** 2 / (1.0 + eta)
    assert np.max(np.abs(v * v - alpha * v + constant)[quadratic]) < 1e-12
    assert np.all((v >= alpha / 2.0)[quadratic])


def test_against_the_exact_implied_vol_it_is_close_near_the_money_and_beats_corrado_miller_on_the_grid():
    """Against qf.implied_vol (the exact inversion), with v = vol sqrt(T) <= 0.5: within 2% of the vol where
    |ln(K~/S~)| <= v/4 and within 5% where it is <= v/2 (defined throughout); and over the grid, where both are defined, a smaller mean and median error than Corrado and
    Miller's (the comparison Li and the reviews report). Where it is undefined (far from the money for its time value)
    it says so with nan, where the exact inversion still gives a vol."""
    price, strike, t, vol = grid()
    li = iv.li_nonatm_vol(price, S, strike, t)
    exact = qf.implied_vol("call", price, S, strike, t, 0.0, 0.0, "bisection")
    cm = qf.corrado_miller_vol(price, S, strike, t, "corrado_miller")
    v, away = vol * np.sqrt(t), np.abs(np.log(strike / S))
    relative = np.abs(li - exact) / exact
    for share, bound in ((0.25, 0.02), (0.5, 0.05)):
        near = (away <= share * v) & (v <= 0.5)
        assert near.sum() > 150 and np.all(np.isfinite(li[near])) and np.max(relative[near]) < bound
    both = np.isfinite(li) & (cm > 0)
    error, cm_error = np.abs(li - exact)[both], np.abs(cm - exact)[both]
    assert both.sum() > 900 and error.mean() < cm_error.mean() and np.median(error) < np.median(cm_error)
    undefined = ~np.isfinite(li)
    assert np.sum(undefined & np.isfinite(exact) & (exact > 0.05)) > 100      # the exact inversion is defined there


def test_a_put_enters_by_parity_and_arrays_broadcast():
    """C = P + S~ - K~ (continuous rates): the put's price gives the call's vol; a scalar gives a float."""
    s_pv, k_pv, t = 100.0 * math.exp(-0.01 * 0.5), 104.0 * math.exp(-0.03 * 0.5), 0.5
    put = qf.bs_price("put", 100.0, 104.0, t, 0.03, 0.01, 0.25)
    call = qf.bs_price("call", 100.0, 104.0, t, 0.03, 0.01, 0.25)
    assert iv.li_nonatm_vol(put + s_pv - k_pv, s_pv, k_pv, t) == pytest.approx(iv.li_nonatm_vol(call, s_pv, k_pv, t),
                                                                                rel=1e-12)
    assert type(iv.li_nonatm_vol(call, s_pv, k_pv, t)) is float
    many = iv.li_nonatm_vol(np.array([call, call]), s_pv, np.array([k_pv, k_pv]), t)
    assert many.shape == (2,) and many[0] == iv.li_nonatm_vol(call, s_pv, k_pv, t)
    for bad in (dict(underlying_pv=0.0), dict(strike_pv=-1.0), dict(maturity=0.0)):
        with pytest.raises(ValueError):
            iv.li_nonatm_vol(**{**dict(call_price=call, underlying_pv=s_pv, strike_pv=k_pv, maturity=t), **bad})
