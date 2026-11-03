"""qf.risk.granularity_adjustment (runs/lanes/lib7-prompt.md, rank 25 of docs/research/library-syllabus-1011.md): each
version against the adjustment's defining property, the first-order move of the loss quantile when a conditional
variance sigma^2(factor) is added to the infinitely granular loss mu(factor), computed here by quadrature (Gordy 2003,
Martin and Wilde 2002; Gordy and Lutkebohmert 2013's CreditRisk+ factor); delta = 4.83 at 99.9% (Gordy and
Lutkebohmert's); their Table 4; the 1/n fall. Every portfolio is this file's own or the paper's; no unit's values
appear (rule 8)."""
import numpy as np
import pytest
from scipy import stats
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from test_qf_conventions import card_problems  # noqa: F401 - the shelf's card check is test_qf_risk's
import qf
import qf.risk as risk

pytestmark = pytest.mark.fast

EAD = np.array([30.0, 20.0, 15.0, 12.0, 10.0, 8.0, 5.0])
PD = np.array([0.01, 0.02, 0.005, 0.03, 0.015, 0.01, 0.04])
LGD = np.array([0.45, 0.4, 0.6, 0.35, 0.45, 0.5, 0.25])
VLGD = 0.25 * LGD * (1 - LGD)
RHO = np.array([0.12, 0.15, 0.2, 0.12, 0.18, 0.14, 0.1])
Q = 0.999


def adjustment(version, **changes):
    args = dict(exposure=EAD, pd=PD, lgd=LGD, lgd_variance=VLGD, rho=RHO, q=Q, version=version,
                xi=0.25 if version == "gordy_lutkebohmert" else None, maturity=None)
    return qf.granularity_adjustment(**{**args, **changes})


def quantile_shift(mu, var, density, grid, centre, eps):
    """(VaR of mu(F) + sqrt(eps) sigma(F) Z - VaR of mu(F)) / eps: P(L <= l) by the trapezoid over the factor's grid
    plus its mass below the grid, the quantile by brentq (mu increasing in the factor); at eps and 2 eps, Richardson's
    2 D(eps) - D(2 eps) takes the next order out."""
    step = np.full(grid.size, grid[1] - grid[0])
    step[[0, -1]] /= 2
    weights = step * density(grid)
    m = mu(grid)
    base = float(mu(np.array([centre(None)]))[0])

    def shift(e):
        s = np.sqrt(e * var(grid))

        def below(level):
            return centre(grid[0]) + np.sum(weights * ndtr((level - m) / s)) - Q

        return (brentq(below, base - 0.05, base + 0.05, xtol=1e-15) - base) / e

    return 2 * shift(eps) - shift(2 * eps)


def test_gordy_2003_is_the_first_order_shift_of_the_vasicek_quantile():
    share = EAD / EAD.sum()

    def p(y):
        return ndtr((ndtri(PD)[:, None] + np.sqrt(RHO)[:, None] * y) / np.sqrt(1 - RHO)[:, None])

    mu = lambda y: (share * LGD) @ p(y)
    var = lambda y: (share ** 2 * (LGD ** 2 + VLGD)) @ p(y) - (share ** 2 * LGD ** 2) @ p(y) ** 2
    y_q = ndtri(Q)
    grid = np.linspace(y_q - 4.0, y_q + 4.0, 160001)
    centre = lambda low: ndtr(low) if low is not None else y_q
    shift = quantile_shift(mu, var, stats.norm.pdf, grid, centre, 1e-3)
    assert adjustment("gordy_2003") == pytest.approx(shift, rel=2e-4)


def test_gordy_lutkebohmert_is_the_first_order_shift_under_creditrisk_plus():
    """X a gamma of mean 1 and variance 4; p_i(x) = PD_i (1 - w_i + w_i x), w_i fitted so ELGD_i p_i(x_q) = K_i + R_i;
    the conditional variance sum s_i^2 [(ELGD_i^2 + V_i) p_i + V_i p_i^2] (a Poisson count of a random LGD)."""
    share = EAD / EAD.sum()
    gamma = stats.gamma(a=0.25, scale=4.0)
    x_q = gamma.ppf(Q)
    k = qf.gordy_capital(pd=PD, lgd=LGD, rho=RHO, q=Q, expected_loss="subtract", maturity=None)
    w = k / (LGD * PD * (x_q - 1))

    def p(x):
        return PD[:, None] * (1 - w[:, None] + w[:, None] * x)

    mu = lambda x: (share * LGD) @ p(x)
    var = lambda x: (share ** 2 * (LGD ** 2 + VLGD)) @ p(x) + (share ** 2 * VLGD) @ p(x) ** 2
    grid = np.linspace(x_q - 8.0, x_q + 8.0, 160001)
    centre = lambda low: gamma.cdf(low) if low is not None else x_q
    shift = quantile_shift(mu, var, gamma.pdf, grid, centre, 1e-3)
    assert adjustment("gordy_lutkebohmert") == pytest.approx(shift, rel=2e-4)
    assert (x_q - 1) * (0.25 + 0.75 / x_q) == pytest.approx(4.83, abs=5e-3)


@pytest.mark.parametrize("version", ["gordy_2003", "gordy_lutkebohmert"])
def test_n_equal_names_fall_as_one_over_n_and_one_name_adds(version):
    one = dict(exposure=[1.0], pd=[0.02], lgd=[0.45], lgd_variance=[0.0], rho=[0.15])
    single = adjustment(version, **one)
    assert single > 0
    for n in (10, 250):
        many = {key: np.repeat(value, n) for key, value in one.items()}
        assert adjustment(version, **many) == pytest.approx(single / n, rel=1e-12)
    assert adjustment(version) < adjustment(version, lgd_variance=VLGD * 2)


@pytest.mark.parametrize("pd, printed", [(0.01, [10.79, 14.38, 19.41, 62.13, 277.62]),
                                         (0.04, [12.34, 16.45, 22.21, 71.08, 317.64])])
def test_gordy_lutkebohmert_reproduces_their_table_4(pd, printed):
    """Gordy and Lutkebohmert (2013), IJCB 9(3), Table 4, the exact GA of their eq. (14) in basis points: 1,000 loans
    of exposure j^p (p = 0, 1, 2, 10, 50), E[LGD] 45%, V[LGD] 0.25 E[LGD](1 - E[LGD]) (their eq. 16), M 2.5,
    q 0.999, xi 0.125, the IRB corporate correlation at the PD (the review of lane lib7)."""
    weight = (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
    rho = 0.12 * weight + 0.24 * (1 - weight)
    names = np.arange(1, 1001, dtype=float)
    found = [1e4 * qf.granularity_adjustment(exposure=names ** p, pd=pd, lgd=0.45, lgd_variance=0.25 * 0.45 * 0.55,
                                               rho=rho, q=0.999, version="gordy_lutkebohmert", xi=0.125, maturity=2.5)
             for p in (0, 1, 2, 10, 50)]
    assert np.round(found, 2).tolist() == printed


def test_the_maturity_adjustment_moves_only_gordy_lutkebohmerts_capital():
    assert adjustment("gordy_lutkebohmert", maturity=2.5) != adjustment("gordy_lutkebohmert")
    with pytest.raises(ValueError, match="None"):
        adjustment("gordy_2003", maturity=2.5)


@pytest.mark.parametrize("bad, match", [
    (dict(pd=0.0), "pd"), (dict(lgd_variance=LGD), "lgd_variance"),
    (dict(exposure=-EAD), "exposure"), (dict(xi=0.25), "None")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        adjustment("gordy_2003", **bad)
    with pytest.raises(ValueError, match="version"):
        adjustment("basel")
