"""Lane lib-rates1's variants of two held curves functions (runs/lanes/lib-rates1-prompt.md and lib-cards-addendum.md):
qf.bootstrap_bond_prices takes the zero rates' compounding, so a curve linear in semi-annual spot rates is solved
inside the bootstrap (the fresh-dev census's variant); qf.nelson_siegel_fit takes fixed decays, the betas by exact
linear least squares (Diebold and Li 2006 fix lambda), and its card names the yields' scale (the misuse audit). The
references are closed-form identities: bonds priced off a curve this file builds come back as that curve; the fixed-decay
betas are ordinary least squares on the Nelson-Siegel loadings and scale with the yields. Every input is this file's
own; no unit's values appear (rule 8)."""
import numpy as np
import pytest

import qf

pytestmark = pytest.mark.fast

NODES = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
SEMI = np.array([0.031, 0.033, 0.036, 0.038, 0.041])      # semi-annual spot rates at the nodes


def semi_annual_discount(t):
    """D(t) with semi-annual spot rates linear in time between the nodes, flat below the first."""
    z = np.interp(t, NODES, SEMI)
    return (1 + z / 2) ** (-2 * np.asarray(t))


def bonds():
    flows, times = [], []
    for maturity, coupon in zip(NODES, [0.03, 0.035, 0.04, 0.02, 0.05]):
        t = np.arange(0.5, maturity + 1e-9, 0.5)
        cf = np.full(t.size, 100 * coupon / 2)
        cf[-1] += 100
        flows.append(cf), times.append(t)
    return flows, times


def bootstrap(compounding, interpolation="linear_zero"):
    flows, times = bonds()
    prices = [cf @ semi_annual_discount(t) for cf, t in zip(flows, times)]
    return qf.bootstrap_bond_prices(cash_flows=flows, times=times, prices=prices, accrued=None, method="sequential",
                                    node_times=None, weights=None, interpolation=interpolation, compounding=compounding)


def test_a_curve_linear_in_semi_annual_spot_rates_is_solved_back_inside_the_bootstrap():
    found = bootstrap(2)
    assert found["zero_rates"] == pytest.approx(SEMI, rel=1e-12) and np.abs(found["residuals"]).max() < 1e-10
    assert found["discount_factors"] == pytest.approx(semi_annual_discount(NODES), rel=1e-13)
    other = bootstrap("continuous")      # linear in continuous rates is another curve: the 3-year bond still prices
    assert np.abs(other["residuals"]).max() < 1e-10
    assert other["discount_factors"][3] != pytest.approx(found["discount_factors"][3], rel=1e-9)


@pytest.mark.parametrize("interpolation", ["log_linear_discount", "cubic_log_discount"])
def test_elsewhere_the_compounding_names_the_zero_rates_returned(interpolation):
    continuous, semi = bootstrap("continuous", interpolation), bootstrap(2, interpolation)
    assert semi["discount_factors"] == pytest.approx(continuous["discount_factors"], rel=1e-11)
    assert semi["zero_rates"] == pytest.approx(2 * np.expm1(continuous["zero_rates"] / 2), rel=1e-10)
    simple = bootstrap("simple", interpolation)
    assert simple["zero_rates"] == pytest.approx(np.expm1(continuous["zero_rates"] * NODES) / NODES, rel=1e-10)


MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def loadings(tau, decay):
    x = MATURITIES / decay
    slope = -np.expm1(-x) / x
    return np.column_stack([np.ones_like(x), slope, slope - np.exp(-x)])


def fit(yields, **changes):
    args = dict(variant="nelson_siegel", maturities=MATURITIES, yields=yields, initial=None, decay_form="time",
                method=None, options=None, fixed_decays=[1.8])
    return qf.nelson_siegel_fit(**{**args, **changes})


def test_fixed_decay_betas_are_least_squares_on_the_loadings_and_scale_with_the_yields():
    yields = 0.04 + 0.01 * np.log1p(MATURITIES) / 4 + 0.001 * np.sin(MATURITIES)
    design = loadings(MATURITIES, 1.8)
    expected = np.linalg.lstsq(design, yields, rcond=None)[0]
    found = fit(yields)
    assert found["params"] == pytest.approx([*expected, 1.8], rel=1e-12) and found["success"]
    assert found["rmse"] == pytest.approx(np.sqrt(np.mean((design @ expected - yields) ** 2)), rel=1e-12)
    percent = fit(100 * yields)
    assert percent["params"][:3] == pytest.approx(100 * found["params"][:3], rel=1e-12)
    assert percent["rmse"] == pytest.approx(100 * found["rmse"], rel=1e-10)


def test_fixing_the_decay_at_the_full_fits_optimum_gives_its_betas():
    exact = qf.nelson_siegel_yield(variant="nelson_siegel", maturities=MATURITIES, params=[0.045, -0.012, 0.008, 2.2],
                                   decay_form="time")
    held = fit(exact, fixed_decays=[2.2])
    assert held["params"] == pytest.approx([0.045, -0.012, 0.008, 2.2], rel=1e-10) and held["rmse"] < 1e-14
    svensson = fit(exact, variant="svensson", fixed_decays=[2.2, 9.0])
    assert svensson["params"][4:] == pytest.approx([2.2, 9.0]) and svensson["rmse"] < 1e-12


@pytest.mark.parametrize("bad, match", [(dict(initial=[0.04, -0.01, 0.0, 1.0]), "None"),
                                        (dict(fixed_decays=[1.0, 2.0]), "fixed_decays"),
                                        (dict(fixed_decays=[-1.0]), "fixed_decays")])
def test_fixed_decays_refuse_a_bad_argument_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        fit(0.04 + np.zeros(10), **bad)
