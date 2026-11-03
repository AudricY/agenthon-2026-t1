"""agent/qf/bs.py, the Black-Scholes core shelf (runs/lanes/lib2-prompt.md): every function against a worked example
in Hull, QuantLib 1.43's engines, or an identity the formula must satisfy, each source named in its test. The
QuantLib values are frozen in `QL`; running this file under a Python with QuantLib 1.43 recomputes and compares them
(scratch/lib2/venv is the lane's). No unit's instruction values, checks, expected values or data appear here (rules 8
and 9): the parameters are textbook examples or chosen freely.

`check_cards` is the card format every lib2 shelf keeps, shared by the four lib2 test files."""
import ast
import inspect
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.fast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import qf  # noqa: E402
import qf.bs  # noqa: E402
from conftest import FIRST_LINE, folded, shelved_before  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CARD_SECTIONS = ("conventions:", "returns: ", "source: ", "aliases: ", "example:")


def check_cards(module):
    """The card (runs/lanes/lib2-prompt.md, "The docstring is the card"): where the module is a shelf (an import module
    holds no function of its own, lane libshelves1), its first line names the shelf within `FIRST_LINE` characters and
    `__all__` is exactly its public functions but the folded; and every function the shelf held before the
    re-shelving (`shelved_before`, wherever it lives now) has its card as `check_card` reads it, in a module that
    names no unit, loads no file and writes none."""
    public = sorted(name for name, obj in vars(module).items() if inspect.isfunction(obj)
                    and not name.startswith("_") and obj.__module__ == module.__name__)
    if public:
        assert module.__doc__.splitlines()[0] and len(module.__doc__.splitlines()[0]) <= FIRST_LINE
        assert sorted(module.__all__) == sorted(set(public) - folded())
    names = shelved_before(module.__name__.rpartition(".")[2])
    assert names, module.__name__
    for path in {Path(sys.modules[getattr(qf, name).__module__].__file__) for name in names}:
        text = path.read_text(encoding="utf-8")
        assert "t1-" not in text and "/app/" not in text and "open(" not in text and "savetxt" not in text
    for name in names:
        check_card(name)


def check_card(name):
    """One function's card, `qf.<name>`: no parameter has a default; the docstring is line 1 (at most `FIRST_LINE`
    characters), a blank,
    the formula, `conventions:` with a `- ` line each (a continuation indented under it), `returns:` (lane lib3:
    the result, and every key of a dict result, named), `source:` with optional indented continuation lines,
    `aliases:` on one line, and `example:` ending in a bare `qf.<name>(...)` call naming every parameter by keyword.
    Callback examples may first import their dependencies and set up their data."""
    function = getattr(qf, name)
    parameters = inspect.signature(function).parameters
    assert all(p.default is inspect.Parameter.empty and p.kind is p.POSITIONAL_OR_KEYWORD
               for p in parameters.values()), name
    lines = inspect.cleandoc(function.__doc__).splitlines()
    assert 0 < len(lines[0]) <= FIRST_LINE and lines[1] == "", name
    conventions = lines.index("conventions:")
    returns, source, aliases, example = (next(i for i, line in enumerate(lines) if line.startswith(head))
                                         for head in CARD_SECTIONS[1:])
    assert 2 < conventions < returns - 1 and returns < source < aliases and example == aliases + 1, name
    assert all(line.startswith("  ") for line in lines[source + 1:aliases]), name
    listed, answer = lines[conventions + 1:returns], lines[returns:source]
    assert listed[0].startswith("- ") and all(line.startswith(("- ", "  ")) for line in listed), name
    assert all(line.startswith("  ") for line in answer[1:]), name
    code = textwrap.dedent("\n".join(lines[example + 1:]))
    tree = ast.parse(code)
    assert tree.body and isinstance(tree.body[-1], ast.Expr), name
    call = tree.body[-1].value
    assert isinstance(call, ast.Call) and not call.args and ast.unparse(call.func) == f"qf.{name}", name
    assert [keyword.arg for keyword in call.keywords] == list(parameters), name
    scope = {"qf": qf}
    exec(compile(ast.Module(tree.body[:-1], []), name, "exec"), scope)
    result = eval(compile(ast.Expression(call), name, "eval"), scope)
    if isinstance(result, dict):
        said = " ".join(answer)
        assert "a dict" in said and all(re.search(rf"(?<![\w.]){key}(?!\w)", said) for key in result), (name, said)


def test_the_bs_cards_keep_the_format_and_their_examples_run():
    check_cards(qf.bs)


# ------------------------------------------------------------------------------------------------ the Greeks

HULL = dict(spot=49.0, strike=50.0, maturity=20 / 52, rate=0.05, dividend_yield=0.0, vol=0.2)


def test_bs_greeks_give_hulls_worked_example_in_every_unit():
    """Hull, Options, Futures, and Other Derivatives, "The Greek letters": a call with S0 = 49, K = 50, r = 5%,
    vol = 20%, T = 20 weeks has delta 0.522, gamma 0.066, vega 12.1 (per unit of vol, 0.121 per point), theta -4.31
    a year (-0.0118 per calendar day, -0.0171 per trading day) and rho 8.91 (0.0891 per point)."""
    raw = qf.bs.bs_greeks(option="call", **HULL, vega_per=1.0, theta_per="year", rho_per=1.0)
    assert raw["price"] == pytest.approx(2.40, abs=0.005)
    assert raw["delta"] == pytest.approx(0.522, abs=5e-4) and raw["gamma"] == pytest.approx(0.066, abs=5e-4)
    assert raw["vega"] == pytest.approx(12.1, abs=0.05) and raw["theta"] == pytest.approx(-4.31, abs=0.005)
    assert raw["rho"] == pytest.approx(8.91, abs=0.005)
    scaled = qf.bs.bs_greeks(option="call", **HULL, vega_per=0.01, theta_per="calendar_day", rho_per=0.01)
    assert scaled["vega"] == pytest.approx(0.121, abs=5e-4) and scaled["rho"] == pytest.approx(0.0891, abs=5e-5)
    assert scaled["theta"] == pytest.approx(-0.0118, abs=5e-5)
    assert scaled["delta"] == raw["delta"] and scaled["gamma"] == raw["gamma"]
    trading = qf.bs.bs_greeks(option="call", **HULL, vega_per=1.0, theta_per="trading_day", rho_per=1.0)
    assert trading["theta"] == pytest.approx(-0.0171, abs=5e-5)
    assert trading["theta"] * 252 == pytest.approx(raw["theta"]) and scaled["theta"] * 365 == pytest.approx(raw["theta"])


QL = {   # QuantLib 1.43, AnalyticEuropeanEngine: S 105, K 100, 292 days ACT/365, r 4.5%, q 2%, vol 27% (see __main__)
    "call": dict(price=13.420684679719756, delta=0.6470452197637716, gamma=0.014260684182929608,
                 vega=33.96039331322856, theta=-6.825379262899813, theta_day=-0.018699669213424145,
                 rho=43.61525071638103),
    "put": dict(price=6.551345422227133, delta=-0.33708210029151314, gamma=0.014260684182929608,
                vega=33.96039331322856, theta=-4.55116531434189, theta_day=-0.01246894606669011,
                rho=-33.5559727622688),
    # blackFormula(type, strike 3.2, forward 3.5, stdDev 0.3 sqrt(1.75), discount 0.93)
    "black76": dict(call=0.6418117818855162, put=0.3628117818855163),
    # AnalyticDividendEuropeanEngine: S 52, K 50, 300 days, r 3.5%, vol 32%, cash 1.2 at 50 days, 1.4 at 230, 2.0 at 400
    "escrow": dict(call=(6.089572708130165, 0.581265229704543), put=(5.235429896049431, -0.41873477029545697)),
}
QL_POINT = dict(spot=105.0, strike=100.0, maturity=292 / 365, rate=0.045, dividend_yield=0.02, vol=0.27)
ESCROW = dict(spot=52.0, strike=50.0, maturity=300 / 365, rate=0.035, vol=0.32, dividend_amounts=[1.2, 1.4, 2.0],
              dividend_times=[50 / 365, 230 / 365, 400 / 365])


@pytest.mark.parametrize("option", ["call", "put"])
def test_bs_greeks_equal_quantlibs_analytic_european_engine(option):
    """QuantLib 1.43's AnalyticEuropeanEngine on a BlackScholesMertonProcess: its vega and rho are per unit, its theta
    per year and thetaPerDay per calendar day (/365), the units vega_per=1.0, rho_per=1.0 and theta_per name."""
    greeks = qf.bs.bs_greeks(option=option, **QL_POINT, vega_per=1.0, theta_per="year", rho_per=1.0)
    for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
        assert greeks[key] == pytest.approx(QL[option][key], rel=1e-12), key
    day = qf.bs.bs_greeks(option=option, **QL_POINT, vega_per=1.0, theta_per="calendar_day", rho_per=1.0)
    assert day["theta"] == pytest.approx(QL[option]["theta_day"], rel=1e-12)


def test_bs_greeks_are_the_derivatives_of_bs_price():
    """Each Greek is the derivative of qf.bs_price it names, by central differences: delta and gamma in spot, vega in
    vol, theta as minus the derivative in maturity, rho in the rate; calls and puts, with a yield, across moneyness."""
    s, k, t, r, q, v = 100.0, np.array([70.0, 95.0, 100.0, 140.0]), 1.3, 0.03, 0.045, 0.35
    for option in ("call", "put"):
        g = qf.bs.bs_greeks(option=option, spot=s, strike=k, maturity=t, rate=r, dividend_yield=q, vol=v,
                            vega_per=1.0, theta_per="year", rho_per=1.0)
        def price(**bump):
            point = dict(spot=s, strike=k, maturity=t, rate=r, dividend_yield=q, vol=v) | bump
            return qf.bs_price(option, *point.values())
        h = 1e-4
        assert g["delta"] == pytest.approx((price(spot=s + h) - price(spot=s - h)) / (2 * h), abs=1e-8)
        assert g["gamma"] == pytest.approx((price(spot=s + h) - 2 * price() + price(spot=s - h)) / h ** 2, abs=1e-5)
        assert g["vega"] == pytest.approx((price(vol=v + h) - price(vol=v - h)) / (2 * h), abs=1e-6)
        assert g["theta"] == pytest.approx(-(price(maturity=t + h) - price(maturity=t - h)) / (2 * h), abs=1e-6)
        assert g["rho"] == pytest.approx((price(rate=r + h) - price(rate=r - h)) / (2 * h), abs=1e-6)
        assert g["price"] == pytest.approx(price(), abs=0)


def test_a_unit_or_input_outside_its_list_is_refused_by_name():
    base = dict(option="call", **HULL, vega_per=0.01, theta_per="year", rho_per=1.0)
    for bad, name in ((dict(vega_per=100.0), "vega_per"), (dict(theta_per="day"), "theta_per"),
                      (dict(rho_per=0.1), "rho_per"), (dict(maturity=0.0), "maturity"), (dict(vol=0.0), "vol"),
                      (dict(option="straddle"), "option")):
        with pytest.raises(ValueError, match=name):
            qf.bs.bs_greeks(**(base | bad))


# ------------------------------------------------------------------------------------------------ Black-76


def test_black76_gives_hulls_futures_option_example_parity_and_quantlibs_formula():
    """Hull, "Black's model": a European put on a futures price with F0 = 20, K = 20, r = 9%, T = 4 months,
    vol = 25% is worth 1.12. Parity c - p = DF (F - K); on F = S e^{(r - q)T} and DF = e^{-rT} it is Black-Scholes-
    Merton (qf.bs_price); at zero vol the discounted intrinsic; QuantLib 1.43's blackFormula at a second point."""
    put = qf.bs.black76_price(option="put", forward=20.0, strike=20.0, expiry=4 / 12, vol=0.25,
                              discount_factor=math.exp(-0.09 * 4 / 12))
    assert put == pytest.approx(1.12, abs=0.005)
    f, k, t, v, df = 3.5, np.array([2.5, 3.2, 4.4]), 1.75, 0.3, 0.93
    call = qf.bs.black76_price(option="call", forward=f, strike=k, expiry=t, vol=v, discount_factor=df)
    put = qf.bs.black76_price(option="put", forward=f, strike=k, expiry=t, vol=v, discount_factor=df)
    assert call - put == pytest.approx(df * (f - k), abs=1e-14)
    assert call[1] == pytest.approx(QL["black76"]["call"], rel=1e-13) and put[1] == pytest.approx(QL["black76"]["put"],
                                                                                                    rel=1e-13)
    s, r, q = 80.0, 0.05, 0.02
    assert qf.bs.black76_price(option="call", forward=s * math.exp((r - q) * t), strike=k, expiry=t, vol=v,
                               discount_factor=math.exp(-r * t)) == pytest.approx(qf.bs_price("call", s, k, t, r, q, v),
                                                                                  rel=1e-12)
    assert qf.bs.black76_price(option="call", forward=f, strike=3.0, expiry=0.0, vol=v, discount_factor=df) == \
        pytest.approx(df * 0.5)
    with pytest.raises(ValueError, match="discount_factor"):
        qf.bs.black76_price(option="call", forward=f, strike=3.0, expiry=t, vol=v, discount_factor=0.0)


# ------------------------------------------------------------------------------------------------ parity


def test_parity_forward_on_model_quotes_is_the_carry_forward_and_its_spread_is_the_quotes():
    """Stoll (1969); Hull, "Put-call parity": on quotes at the model price (bid = ask) the synthetic forward is
    S e^{(r - q)T} and the implied yield is q; widening each quote by its half-spread moves the bid down and the ask up
    by the spreads' sum over DF; a crossed quote is refused."""
    s, k, t, r, q, v = 100.0, np.array([80.0, 100.0, 125.0]), 0.75, 0.04, 0.015, 0.3
    c, p = qf.bs_price("call", s, k, t, r, q, v), qf.bs_price("put", s, k, t, r, q, v)
    flat = qf.bs.parity_forward(call_bid=c, call_ask=c, put_bid=p, put_ask=p, strike=k, spot=s, rate=r, maturity=t)
    assert flat["bid"] == pytest.approx(np.full(3, s * math.exp((r - q) * t)), rel=1e-13)
    assert flat["ask"] == pytest.approx(flat["bid"]) and flat["implied_yield"] == pytest.approx(np.full(3, q), abs=1e-12)
    hc, hp = np.array([0.05, 0.1, 0.2]), np.array([0.15, 0.1, 0.05])
    wide = qf.bs.parity_forward(call_bid=c - hc, call_ask=c + hc, put_bid=p - hp, put_ask=p + hp, strike=k, spot=s,
                                rate=r, maturity=t)
    df = math.exp(-r * t)
    assert wide["bid"] == pytest.approx(flat["bid"] - (hc + hp) / df, rel=1e-13)
    assert wide["ask"] == pytest.approx(flat["ask"] + (hc + hp) / df, rel=1e-13)
    assert wide["mid"] == pytest.approx(flat["mid"], rel=1e-13)
    with pytest.raises(ValueError, match="crossed"):
        qf.bs.parity_forward(call_bid=5.0, call_ask=4.9, put_bid=1.0, put_ask=1.1, strike=100.0, spot=100.0, rate=r,
                             maturity=t)


def test_parity_rate_and_forward_recovers_the_discount_and_forward_and_two_strikes_are_the_box():
    """C - P = DF (F - K) on Black-Scholes-Merton prices over five strikes: the line's slope is -e^{-rT} and its
    intercept DF S e^{(r - q)T}, residuals 0; two strikes are the box spread's DF; one strike is refused."""
    s, k, t, r, q, v = 100.0, np.array([70.0, 90.0, 100.0, 115.0, 150.0]), 1.25, 0.035, 0.01, 0.25
    c, p = qf.bs_price("call", s, k, t, r, q, v), qf.bs_price("put", s, k, t, r, q, v)
    fit = qf.bs.parity_rate_and_forward(strikes=k, call_prices=c, put_prices=p, maturity=t)
    assert fit["discount_factor"] == pytest.approx(math.exp(-r * t), rel=1e-13)
    assert fit["rate"] == pytest.approx(r, rel=1e-11) and fit["forward"] == pytest.approx(s * math.exp((r - q) * t))
    assert np.max(np.abs(fit["residuals"])) < 1e-12
    noisy = qf.bs.parity_rate_and_forward(strikes=k, call_prices=c + [0.02, -0.01, 0.0, 0.01, -0.02], put_prices=p,
                                          maturity=t)
    assert noisy["residuals"].sum() == pytest.approx(0.0, abs=1e-12) and np.max(np.abs(noisy["residuals"])) > 1e-3
    box = qf.bs.parity_rate_and_forward(strikes=[95.0, 110.0], call_prices=[9.0, 2.5], put_prices=[3.0, 11.0],
                                        maturity=0.5)
    assert box["discount_factor"] == pytest.approx(((9.0 - 3.0) - (2.5 - 11.0)) / 15.0)
    with pytest.raises(ValueError, match="two distinct strikes"):
        qf.bs.parity_rate_and_forward(strikes=[100.0, 100.0], call_prices=[5.0, 5.0], put_prices=[4.0, 4.0],
                                      maturity=0.5)


# ------------------------------------------------------------------------------------------------ dividends


def test_escrowed_dividend_option_gives_hulls_example_and_quantlibs_dividend_engine():
    """Hull, "Dividends": S0 = 40, K = 40, r = 9%, vol = 30%, T = 6 months, dividends of 0.50 in two and five months:
    the dividends' present value is 0.9742 and the call 3.67. QuantLib 1.43's AnalyticDividendEuropeanEngine (the same model:
    spot less the dividends' PV) at a second point, price and delta, with a dividend after expiry it ignores."""
    hull = qf.bs.escrowed_dividend_option(option="call", spot=40.0, strike=40.0, maturity=0.5, rate=0.09, vol=0.3,
                                          dividend_amounts=[0.5, 0.5], dividend_times=[2 / 12, 5 / 12])
    assert 40.0 - hull["escrowed_spot"] == pytest.approx(0.9742, abs=5e-5) and hull["price"] == pytest.approx(3.67,
                                                                                                          abs=0.005)
    for option in ("call", "put"):
        found = qf.bs.escrowed_dividend_option(option=option, **ESCROW)
        price, delta = QL["escrow"][option]
        assert found["price"] == pytest.approx(price, rel=1e-12) and found["delta"] == pytest.approx(delta, rel=1e-12)


def test_escrowed_delta_is_the_spot_derivative_and_only_the_remaining_dividends_count():
    """delta = dV/dS by central differences; with no dividend left (none, all paid, all after expiry) the price is
    Black-Scholes with no yield; parity holds on the escrowed spot, C - P = S* - K e^{-rT}."""
    h = 1e-4
    for option in ("call", "put"):
        up = qf.bs.escrowed_dividend_option(option=option, **(ESCROW | dict(spot=52.0 + h)))["price"]
        down = qf.bs.escrowed_dividend_option(option=option, **(ESCROW | dict(spot=52.0 - h)))["price"]
        assert qf.bs.escrowed_dividend_option(option=option, **ESCROW)["delta"] == pytest.approx((up - down) / (2 * h),
                                                                                              abs=1e-8)
    plain = qf.bs_price("call", 52.0, 50.0, 300 / 365, 0.035, 0.0, 0.32)
    for amounts, times in (([], []), ([1.0, 2.0], [-0.1, 0.0]), ([1.0], [0.9])):
        found = qf.bs.escrowed_dividend_option(option="call", **(ESCROW | dict(dividend_amounts=amounts,
                                                                              dividend_times=times)))
        assert found["price"] == pytest.approx(plain, rel=1e-13) and found["escrowed_spot"] == 52.0
    call, put = (qf.bs.escrowed_dividend_option(option=o, **ESCROW) for o in ("call", "put"))
    assert call["price"] - put["price"] == pytest.approx(call["escrowed_spot"] - 50.0 * math.exp(-0.035 * 300 / 365))
    with pytest.raises(ValueError, match="present value"):
        qf.bs.escrowed_dividend_option(option="call", **(ESCROW | dict(dividend_amounts=[60.0], dividend_times=[0.5])))


def quantlib_values():
    """QL recomputed with QuantLib 1.43: `scratch/lib2/venv/bin/python tests/test_qf_bs.py`."""
    import QuantLib as ql
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()

    def process(s, r, q, v):
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(s)), ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), v, dc)))

    values = {}
    for kind, name in ((ql.Option.Call, "call"), (ql.Option.Put, "put")):
        option = ql.VanillaOption(ql.PlainVanillaPayoff(kind, 100.0), ql.EuropeanExercise(today + 292))
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process(105.0, 0.045, 0.02, 0.27)))
        values[name] = dict(price=option.NPV(), delta=option.delta(), gamma=option.gamma(), vega=option.vega(),
                            theta=option.theta(), theta_day=option.thetaPerDay(), rho=option.rho())
    values["black76"] = {name: ql.blackFormula(kind, 3.2, 3.5, 0.3 * math.sqrt(1.75), 0.93)
                         for kind, name in ((ql.Option.Call, "call"), (ql.Option.Put, "put"))}
    dividends = ql.DividendVector([today + 50, today + 230, today + 400], [1.2, 1.4, 2.0])
    values["escrow"] = {}
    for kind, name in ((ql.Option.Call, "call"), (ql.Option.Put, "put")):
        option = ql.VanillaOption(ql.PlainVanillaPayoff(kind, 50.0), ql.EuropeanExercise(today + 300))
        option.setPricingEngine(ql.AnalyticDividendEuropeanEngine(process(52.0, 0.035, 0.0, 0.32), dividends))
        values["escrow"][name] = (option.NPV(), option.delta())
    return values


if __name__ == "__main__":
    fresh = quantlib_values()
    print("QL frozen values match QuantLib:", fresh == QL or all(
        np.allclose(np.ravel(list(fresh[key].values())), np.ravel(list(QL[key].values())), rtol=1e-14, atol=0)
        for key in QL))


def test_black76_greeks_are_bs_greeks_on_the_forward_with_the_yield_at_the_rate():
    """Lane lib6's line on bs_greeks (Black 1976; Hull, "Black's model"): the price and delta on a futures price F
    are Black-Scholes-Merton's with S = F and q = r."""
    f, k, t, rate, vol = 52.0, 50.0, 0.75, 0.04, 0.3
    greeks = qf.bs.bs_greeks(option="call", spot=f, strike=k, maturity=t, rate=rate, dividend_yield=rate, vol=vol,
                             vega_per=1.0, theta_per="year", rho_per=1.0)
    assert greeks["price"] == pytest.approx(qf.bs.black76_price("call", f, k, t, vol, math.exp(-rate * t)), abs=1e-12)
    bumped = [qf.bs.black76_price("call", f + h, k, t, vol, math.exp(-rate * t)) for h in (1e-5, -1e-5)]
    assert greeks["delta"] == pytest.approx((bumped[0] - bumped[1]) / 2e-5, abs=1e-8)

    def black(forward=f, maturity=t, r=rate, sigma=vol):
        return qf.bs.black76_price("call", forward, k, maturity, sigma, math.exp(-r * maturity))

    h = 1e-4         # the lane's review: gamma, vega and theta are Black-76's too; its rho, F held, is -T times the price
    assert greeks["gamma"] == pytest.approx((black(f + h) - 2 * black() + black(f - h)) / h ** 2, abs=1e-6)
    assert greeks["vega"] == pytest.approx((black(sigma=vol + h) - black(sigma=vol - h)) / (2 * h), abs=1e-6)
    assert greeks["theta"] == pytest.approx(-(black(maturity=t + h) - black(maturity=t - h)) / (2 * h), abs=1e-6)
    rho = (black(r=rate + h) - black(r=rate - h)) / (2 * h)
    assert rho == pytest.approx(-t * greeks["price"], abs=1e-7) and abs(greeks["rho"] - rho) > 1.0


def test_bs_prices_keep_the_no_arbitrage_bounds():
    """Lane lib6's line on bs_price (Hull, "Properties of stock options")."""
    s, t, rate, q = 100.0, 1.3, 0.04, 0.02
    k = np.linspace(40.0, 220.0, 19)[:, None]
    vol = np.array([0.01, 0.2, 0.8, 3.0])[None, :]
    call, put = qf.bs_price("call", s, k, t, rate, q, vol), qf.bs_price("put", s, k, t, rate, q, vol)
    fs, fk = s * math.exp(-q * t), k * math.exp(-rate * t)
    assert np.all(call >= np.maximum(fs - fk, 0.0) - 1e-12) and np.all(call <= fs + 1e-12)
    assert np.all(put >= np.maximum(fk - fs, 0.0) - 1e-12) and np.all(put <= fk + 1e-12)
