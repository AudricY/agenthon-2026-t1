"""Lane lib-rates1's inflation shelf (runs/lanes/lib-rates1-prompt.md; syllabus row 79): the reference CPI and index
ratio, the implied CPI curve from zero-coupon inflation swaps, and the year-on-year forward with its Jarrow-Yildirim
convexity adjustment. References: 31 CFR 356 Appendix B's worked example (the Ref CPI for April 15, 1996 from the
January and February 1996 CPI-U, 154.63333, and the index ratio on April 16, 1.00011); closed-form identities: the
quoted swaps reprice, a zero breakeven gives a flat index (Brigo and Mercurio ch. 15), the adjustment's two written
forms agree, and (slow) the adjustment against a Monte Carlo of the Jarrow-Yildirim dynamics themselves. Every input is
this file's own or the regulation's; no unit's values appear (rule 8)."""
import math

import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast

CPI_1996 = {"1996-01": 154.4, "1996-02": 154.9}       # CPI-U, as 31 CFR 356 Appendix B's example gives them


def ref(dates, **changes):
    args = dict(dates=dates, cpi=CPI_1996, lag_months=3, interpolation="daily", base_cpi=None, rounding="treasury")
    return qf.reference_cpi(**{**args, **changes})


def test_the_treasurys_worked_example():
    """Ref CPI April 15, 1996 = 154.4 + (14/30)(154.9 - 154.4), truncated to six decimals and rounded to five,
    154.63333; on April 16 the index ratio to it is 154.65000 / 154.63333, 1.00011."""
    assert ref("1996-04-15")["reference_cpi"] == 154.63333
    found = ref("1996-04-16", base_cpi=154.63333)
    assert found["reference_cpi"] == 154.65 and found["index_ratio"] == 1.00011
    plain = ref("1996-04-15", rounding="none")["reference_cpi"]
    assert plain == pytest.approx(154.4 + 14 / 30 * 0.5, rel=1e-15) and plain != 154.63333


def test_the_first_of_the_month_is_the_lagged_month_and_monthly_holds_it_all_month():
    found = ref(["1996-04-01", "1996-04-30"], rounding="none")["reference_cpi"]
    assert found[0] == 154.4 and found[1] == pytest.approx(154.4 + 29 / 30 * 0.5, rel=1e-15)
    assert ref(["1996-04-01", "1996-04-30"], interpolation="monthly")["reference_cpi"] == pytest.approx([154.4, 154.4])
    with pytest.raises(ValueError, match="1996-03"):
        ref("1996-05-10")
    with pytest.raises(ValueError, match="interpolation"):
        ref("1996-04-10", interpolation="weekly")


SWAPS = dict(base_index=250.0, maturities=[1.0, 2.0, 5.0, 10.0], swap_rates=[0.021, 0.0225, 0.024, 0.026])


@pytest.mark.parametrize("interpolation", ["log_linear", "linear_rate"])
def test_the_quoted_swaps_reprice_and_a_zero_breakeven_is_a_flat_index(interpolation):
    found = qf.implied_cpi_curve(**SWAPS, times=SWAPS["maturities"], interpolation=interpolation)
    k, t = np.array(SWAPS["swap_rates"]), np.array(SWAPS["maturities"])
    assert found["index"] == pytest.approx(250.0 * (1 + k) ** t, rel=1e-14)
    assert found["zero_rates"] == pytest.approx(k, rel=1e-12)
    flat = qf.implied_cpi_curve(base_index=250.0, maturities=[1.0, 5.0], swap_rates=[0.0, 0.0], times=[0.0, 0.5, 3.0],
                                interpolation=interpolation)
    assert flat["index"] == pytest.approx(250.0, rel=1e-15)


def test_log_linear_holds_the_forward_inflation_constant_and_linear_rate_reads_the_swap_rate():
    mid = qf.implied_cpi_curve(**SWAPS, times=[3.5], interpolation="log_linear")["index"]
    assert mid == pytest.approx(250.0 * math.sqrt(1.0225 ** 2 * 1.024 ** 5), rel=1e-14)
    linear = qf.implied_cpi_curve(**SWAPS, times=[3.5, 0.5], interpolation="linear_rate")
    assert linear["zero_rates"] == pytest.approx([0.02325, 0.021], rel=1e-12)
    with pytest.raises(ValueError, match="extrapolation"):
        qf.implied_cpi_curve(**SWAPS, times=[11.0], interpolation="log_linear")


JY = dict(nominal_reversion=0.08, nominal_vol=0.009, real_reversion=0.12, real_vol=0.007, index_vol=0.015,
          rho_nominal_real=0.6, rho_real_index=0.25)


def yoy(starts, ends, i_start, i_end, **changes):
    return qf.yoy_convexity_adjustment(start_times=starts, end_times=ends, index_start=i_start, index_end=i_end,
                                       **{**JY, **changes})


def test_no_real_rate_vol_or_a_period_starting_today_needs_no_adjustment():
    flat = yoy([1.0, 4.0], [2.0, 5.0], [100.0, 106.0], [102.0, 108.0], real_vol=0.0)
    assert flat["convexity_factors"] == pytest.approx(1.0) and flat["yoy_rates"] == pytest.approx(flat["naive_rates"])
    today = yoy(0.0, 1.0, 100.0, 102.0)
    assert today["convexity_factors"] == pytest.approx(1.0, abs=1e-15) and today["yoy_rates"] == pytest.approx(0.02)


def test_the_adjustment_is_the_real_bonds_forward_measure_shift():
    """C = -B_r(T_{i-1}, T_i) m, m the mean shift of the real rate at T_{i-1} from the real to the nominal
    T_{i-1}-forward measure: -rho_rI s_I s_r B_r - rho_nr s_n s_r (B_r - B_{n+r}) / a_n + s_r^2 B_r^2 / 2, B_x =
    B_x(0, T_{i-1}). The card's form (Brigo and Mercurio 15.4) is the same quantity written otherwise."""
    a_n, s_n, a_r, s_r, s_i, rho_nr, rho_ri = JY.values()
    t0, t1 = 3.0, 4.0
    b_r = (1 - math.exp(-a_r * t0)) / a_r
    b_nr = (1 - math.exp(-(a_n + a_r) * t0)) / (a_n + a_r)
    m = -rho_ri * s_i * s_r * b_r - rho_nr * s_n * s_r * (b_r - b_nr) / a_n + s_r ** 2 * b_r ** 2 / 2
    c = -(1 - math.exp(-a_r * (t1 - t0))) / a_r * m
    found = yoy(t0, t1, 110.0, 112.5)
    assert found["convexity_factors"] == pytest.approx(math.exp(c), rel=1e-14)
    assert found["yoy_rates"] == pytest.approx(112.5 / 110.0 * math.exp(c) - 1, rel=1e-14)


@pytest.mark.slow
def test_the_adjustment_against_a_monte_carlo_of_the_jarrow_yildirim_dynamics():
    """Under the nominal risk-neutral measure: n and r Ornstein-Uhlenbeck (constant drifts, so the model's own curves
    are Vasicek's), r's drift less rho_rI s_I s_r, and dI/I = (n - r) dt + s_I dW_I. The YoY payoff discounted by
    exp(-int n) over P_n(0, T_i) is the forward measure's E[I(T_i)/I(T_{i-1})]; the implied index ratio is the
    model's own P_r(0, T_i) P_n(0, T_{i-1}) / (P_r(0, T_{i-1}) P_n(0, T_i)). Exact Gaussian steps for n and r,
    trapezoid integrals, antithetic draws, the discount factor (mean P_n(0, T_i) known) as a control variate; a long
    horizon and a large real-rate vol make the adjustment many standard errors wide, and the estimate meets it."""
    jy = dict(nominal_reversion=0.05, nominal_vol=0.01, real_reversion=0.05, real_vol=0.02, index_vol=0.02,
              rho_nominal_real=0.5, rho_real_index=0.4)
    a_n, s_n, a_r, s_r, s_i, rho_nr, rho_ri = jy.values()
    rho_ni, theta_n, theta_r, n0, r0 = 0.1, 0.035, 0.012, 0.03, 0.01
    t0, t1, steps, paths = 8.0, 9.0, 360, 100_000
    dt = t1 / steps
    chol = np.linalg.cholesky(np.array([[1, rho_nr, rho_ni], [rho_nr, 1, rho_ri], [rho_ni, rho_ri, 1.0]]))
    rng = np.random.default_rng(20261012)
    n, r = np.full(2 * paths, n0), np.full(2 * paths, r0)
    log_i, integral_n, at_start = np.zeros(2 * paths), np.zeros(2 * paths), None
    drift_r = theta_r - rho_ri * s_i * s_r / a_r
    keep_n, keep_r = math.exp(-a_n * dt), math.exp(-a_r * dt)
    width_n = s_n * math.sqrt(-math.expm1(-2 * a_n * dt) / (2 * a_n))
    width_r = s_r * math.sqrt(-math.expm1(-2 * a_r * dt) / (2 * a_r))
    for k in range(steps):
        z = rng.standard_normal((paths, 3)) @ chol.T
        z = np.vstack([z, -z])
        n_next = theta_n + (n - theta_n) * keep_n + width_n * z[:, 0]
        r_next = drift_r + (r - drift_r) * keep_r + width_r * z[:, 1]
        log_i += 0.5 * ((n - r) + (n_next - r_next)) * dt - 0.5 * s_i ** 2 * dt + s_i * math.sqrt(dt) * z[:, 2]
        integral_n += 0.5 * (n + n_next) * dt
        n, r = n_next, r_next
        if k + 1 == round(t0 / dt):
            at_start = log_i.copy()

    def vasicek(kappa, theta, sigma, x0, tau):
        return qf.short_rate_bond_price(model="vasicek", option="none", short_rate=x0, maturity=tau, kappa=kappa,
                                        theta=theta, sigma=sigma, expiry=None, strike=None)["price"]

    p_n1 = vasicek(a_n, theta_n, s_n, n0, t1)
    discount = np.exp(-integral_n) / p_n1
    sample = discount * np.exp(log_i - at_start)
    beta = np.cov(sample, discount)[0, 1] / discount.var()
    adjusted = sample - beta * (discount - 1.0)
    paired = 0.5 * (adjusted[:paths] + adjusted[paths:])
    estimate, error = paired.mean(), paired.std(ddof=1) / math.sqrt(paths)
    naive = (vasicek(a_r, theta_r, s_r, r0, t1) * vasicek(a_n, theta_n, s_n, n0, t0)
             / (vasicek(a_r, theta_r, s_r, r0, t0) * p_n1))
    found = qf.yoy_convexity_adjustment(start_times=t0, end_times=t1, index_start=1.0, index_end=naive, **jy)
    assert abs(estimate - (found["yoy_rates"] + 1)) < 4 * error
    assert abs(naive * found["convexity_factors"] - naive) > 20 * error      # the adjustment itself, resolved
