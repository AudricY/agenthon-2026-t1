"""Cold-review references constructed from Hallerbach (2004), ERS-2004-054, equations 20–21.

Inputs are synthetic, independently chosen; the expected values use the rearranged equation 21.
"""
import math

import pytest

import qf

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("price,spot_pv,strike_pv,expiry,expected", [
    (12.0, 100.0, 103.0, 0.75, 0.38342842190861315),
    (9.0, 102.0, 97.0, 1.25, 0.13939346704696212),
])
def test_hallerbach_unadjusted_equation_20(price, spot_pv, strike_pv, expiry, expected):
    actual = qf.corrado_miller_vol(price, spot_pv, strike_pv, expiry, "hallerbach_unadjusted")
    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("variant", ["corrado_miller", "hallerbach", "hallerbach_unadjusted"])
def test_all_quadratic_estimators_have_the_published_atm_limit(variant):
    expected = math.sqrt(2.0 * math.pi / 0.75) * 12.0 / 100.0
    assert qf.corrado_miller_vol(12.0, 100.0, 100.0, 0.75, variant) == pytest.approx(expected, rel=1e-14)
