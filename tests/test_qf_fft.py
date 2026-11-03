"""qf.fourier.fft_price (runs/lanes/lib7-prompt.md, rank 20 of docs/research/library-syllabus-1011.md): Carr-Madan's
FFT and the COS method against qf.bs_price on a lognormal characteristic function and against qf.fourier.charfn_price
on Heston's; the FFT's sum at a grid strike written out. Every input is this file's own; no unit's values appear
(rule 8)."""
import numpy as np
import pytest

from test_qf_bs import check_cards  # noqa: F401 - first: it puts agent/ on the path outside pytest too
import qf
import qf.fourier as F

pytestmark = pytest.mark.fast

S, T, R, Q, VOL = 100.0, 0.8, 0.03, 0.01, 0.25
K = np.array([60.0, 80.0, 95.0, 100.0, 110.0, 130.0, 160.0])


def lognormal(u):
    return np.exp(1j * u * (np.log(S) + (R - Q - VOL * VOL / 2) * T) - VOL * VOL * u * u * T / 2)


def heston(u):
    return F.heston_charfn(u, S, T, R, Q, 0.04, 1.5, 0.04, 0.5, -0.7, False)


def cm(charfn, option, points=4096, spacing=0.25, damping=1.5, weights="simpson", strike=K):
    return F.fft_price(charfn, option, S, strike, T, R, Q, "carr_madan", points, spacing, damping, weights, None)


def cos(charfn, option, points=128, truncation=12.0, strike=K):
    return F.fft_price(charfn, option, S, strike, T, R, Q, "cos", points, None, None, None, truncation)


@pytest.mark.parametrize("option", ["call", "put"])
def test_both_methods_give_black_scholes_on_a_lognormal_charfn(option):
    exact = qf.bs_price(option, S, K, T, R, Q, VOL)
    assert cos(lognormal, option) == pytest.approx(exact, abs=1e-12)
    assert cm(lognormal, option) == pytest.approx(exact, abs=5e-7)
    assert cm(lognormal, option, weights="trapezoid") == pytest.approx(exact, abs=5e-8)


def test_both_methods_give_the_gil_pelaez_price_on_hestons_charfn():
    """COS converges as L and N grow (L = 12 on c2 alone leaves 5e-6 of the tails out; 20 leaves 4e-10); the FFT
    as its grid widens and refines."""
    exact = F.charfn_price(heston, "call", S, K, T, R, Q, "gauss_legendre", 200.0, 512)["price"]   # quad agrees to 1e-11
    assert cos(heston, "call", 1024, 20.0) == pytest.approx(exact, abs=1e-9)
    assert np.max(np.abs(cos(heston, "call", 256, 12.0) - exact)) > 1e-6
    assert cm(heston, "call", 16384, 0.1) == pytest.approx(exact, abs=2e-8)


def test_the_fft_at_a_grid_strike_is_the_sum_written_out():
    """At K = S, ln K the grid's middle point: e^{-alpha k} / pi Re sum_j e^{-i v_j k} psi(v_j) w_j, Simpson's
    w_j = eta/3 (1, 4, 2, 4, ..., 2, 4)."""
    n, eta, alpha = 1024, 0.2, 1.25
    v = eta * np.arange(n)
    psi = np.exp(-R * T) * heston(v - (alpha + 1) * 1j) / (alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v)
    w = eta / 3 * np.where(np.arange(n) % 2 == 1, 4.0, 2.0)
    w[0] = eta / 3
    k = np.log(S)
    expected = np.exp(-alpha * k) / np.pi * np.sum(np.exp(-1j * v * k) * psi * w).real
    assert cm(heston, "call", n, eta, alpha, strike=S) == pytest.approx(expected, rel=1e-13)


def test_call_less_put_is_the_forward_less_the_strike():
    for price in (cm, cos):
        gap = price(heston, "call") - price(heston, "put")
        assert gap == pytest.approx(S * np.exp(-Q * T) - K * np.exp(-R * T), abs=1e-12)


@pytest.mark.parametrize("bad, match", [
    (dict(method="lewis"), "method"), (dict(points=1000), "power of 2"), (dict(weights="midpoint"), "weights"),
    (dict(truncation=10.0), "truncation"), (dict(method="cos", truncation=10.0), "carr_madan"),
    (dict(strike=[1e9]), "off the FFT"), (dict(damping=0.0), "damping")])
def test_a_bad_argument_is_refused_by_name(bad, match):
    args = dict(charfn=heston, option="call", spot=S, strike=K, maturity=T, rate=R, dividend_yield=Q,
                method="carr_madan", points=1024, spacing=0.25, damping=1.5, weights="simpson", truncation=None)
    with pytest.raises(ValueError, match=match):
        F.fft_price(**{**args, **bad})
