"""qf.fx_value_dates and qf.fx_swap_outrights (lane libgaps1; docs/research/library-syllabus-1011.md row 58 and the
holdout read's FX settlement gap): spot dates by business days on both calendars, tenors by months or days from spot
with the end-of-month rule and following or modified following, the NDF fixing date; outrights and FX swaps on the
far leg's side, the near leg's named. References: ISDA 2006's business day conventions applied to calendars of this
file's own (weekends and holidays named in each test), FIX's swap points, and the arithmetic of points; no unit's
dates, pairs or quotes appear (rule 8)."""
import pytest

import qf

pytestmark = pytest.mark.fast

NONE = {"EUR": [], "USD": []}


def dates(trade, lag, tenor, holidays=NONE, adjustment="modified_following", eom=False, rule="joint", fixing=None):
    return qf.fx_value_dates(trade_date=trade, spot_lag=lag, tenor=tenor, holidays=holidays, adjustment=adjustment,
                             end_of_month=eom, lag_rule=rule, fixing_lag=fixing)


def test_spot_counts_business_days_open_in_both_calendars():
    """T + 2 from a Wednesday is the Friday; from a Thursday the Monday (the weekend skipped); a holiday in either
    currency is no business day ('joint'), so a Friday holiday in one and a Monday in the other push spot to Tuesday.
    T + 1 (USD/CAD's lag) from a Friday is the Monday."""
    assert dates("2025-06-11", 2, None)["spot_date"] == "2025-06-13"
    assert dates("2025-06-12", 2, None)["spot_date"] == "2025-06-16"
    held = {"EUR": ["2025-06-13"], "USD": ["2025-06-16"]}
    assert dates("2025-06-11", 2, None, held)["spot_date"] == "2025-06-17"
    assert dates("2025-06-13", 1, None)["spot_date"] == "2025-06-16"


def test_the_usd_rule_lets_an_intermediate_usd_holiday_pass():
    """Under 'usd_rule' the lag counts the non-USD calendar's days and the spot date then moves on to a day open in
    all: a USD holiday on T + 1 does not delay spot; one on the spot date itself does."""
    held = {"EUR": [], "USD": ["2025-06-12"]}
    assert dates("2025-06-11", 2, None, held, rule="joint")["spot_date"] == "2025-06-16"
    assert dates("2025-06-11", 2, None, held, rule="usd_rule")["spot_date"] == "2025-06-13"
    on_spot = {"EUR": [], "USD": ["2025-06-13"]}
    assert dates("2025-06-11", 2, None, on_spot, rule="usd_rule")["spot_date"] == "2025-06-16"


def test_month_tenors_roll_modified_following_and_keep_the_month():
    """A month from spot is the same day of the month; on a Saturday 'following' goes to Monday, 'modified_following'
    to the Monday unless that leaves the month, then back to the Friday (ISDA 2006, 4.12)."""
    assert dates("2025-05-27", 2, "1M")["delivery_date"] == "2025-06-30"     # spot Thu 29 May; 29 Jun is a Sunday
    late = dates("2025-05-27", 2, "1M", adjustment="following")
    assert late["delivery_date"] == "2025-06-30" and late["days_from_spot"] == 32
    month_end = dates("2025-03-27", 2, "1M")                                   # spot Mon 31 Mar; 30 Apr is a Wed
    assert month_end["spot_date"] == "2025-03-31" and month_end["delivery_date"] == "2025-04-30"
    sat = dates("2025-04-28", 2, "5M")                                         # spot Wed 30 Apr; 30 Sep is a Tue
    assert sat["delivery_date"] == "2025-09-30"
    friday = dates("2025-08-27", 2, "1M", adjustment="modified_following")    # spot Fri 29 Aug; 29 Sep is a Mon
    assert friday["delivery_date"] == "2025-09-29"
    edge = dates("2025-10-29", 2, "1M", adjustment="modified_following")      # spot Fri 31 Oct; 30 Nov is a Sun
    assert edge["delivery_date"] == "2025-11-28"
    assert dates("2025-10-29", 2, "1M", adjustment="following")["delivery_date"] == "2025-12-01"


def test_the_end_of_month_rule_holds_a_spot_on_its_months_last_business_day_to_month_ends():
    """End of month: a spot on its month's last business day delivers on the target month's last business day; off
    it, the day of the month is kept (clipped to a shorter month)."""
    spot_on_end = "2025-04-28"                                                  # spot Wed 30 Apr, April's last
    assert dates(spot_on_end, 2, "1M", eom=True)["delivery_date"] == "2025-05-30"   # 31 May is a Saturday
    assert dates(spot_on_end, 2, "1M", eom=False)["delivery_date"] == "2025-05-30"
    assert dates(spot_on_end, 2, "2M", eom=True)["delivery_date"] == "2025-06-30"
    assert dates(spot_on_end, 2, "2M", eom=False)["delivery_date"] == "2025-06-30"
    feb = dates("2025-02-26", 2, "1M", eom=True)                                # spot Fri 28 Feb, its last
    assert feb["delivery_date"] == "2025-03-31"
    assert dates("2025-02-26", 2, "1M", eom=False)["delivery_date"] == "2025-03-28"
    assert dates("2025-01-27", 2, "1M", eom=False)["delivery_date"] == "2025-02-28"   # 29 Jan + 1M: clipped, Fri
    assert dates("2025-04-11", 2, "1M", eom=True)["delivery_date"] == "2025-05-15"    # spot 15 Apr: off the end


def test_week_and_day_tenors_count_calendar_days_and_the_ndf_fixes_before_settlement():
    """'6W' is 42 calendar days from spot, rolled; an NDF fixes fixing_lag business days before its settlement, on
    the joint calendar (a holiday in between pushes the fixing back)."""
    found = dates("2025-06-11", 2, "6W", fixing=2)
    assert found["delivery_date"] == "2025-07-25" and found["days_from_spot"] == 42
    assert found["fixing_date"] == "2025-07-23"
    held = {"EUR": [], "BRL": ["2025-07-24"]}
    assert dates("2025-06-11", 2, "6W", held, fixing=2)["fixing_date"] == "2025-07-22"
    assert dates("2025-06-11", 2, "10D")["delivery_date"] == "2025-06-23"      # 23 Jun is a Monday
    assert dates("2025-05-27", 2, "2D")["delivery_date"] == "2025-05-30"       # Sat 31 May: back, in the month
    assert dates("2025-05-27", 2, "2D", adjustment="following")["delivery_date"] == "2025-06-02"


def test_fx_value_dates_refuses_an_unread_tenor_and_holidays_not_by_currency():
    with pytest.raises(ValueError, match="tenor"):
        dates("2025-06-11", 2, "3 months")
    with pytest.raises(ValueError, match="dict"):
        qf.fx_value_dates(trade_date="2025-06-11", spot_lag=2, tenor=None, holidays=["2025-06-12"],
                          adjustment="following", end_of_month=False, lag_rule="joint", fixing_lag=None)


def test_outrights_take_the_far_legs_side_and_the_near_legs_named_side():
    """An outright is spot + points / multiplier; the far leg is on its direction's side: selling the base forward
    takes the bid points, buying it the offer. 'same' puts the near leg on that side too; 'opposite' on the other, as
    FIX's BidSwapPoints (1065) and OfferSwapPoints (1066) define a swap's points: far bid - near offer and far offer -
    near bid. Discount points (negative) subtract; JPY's multiplier is 100; a spot start (near points 0) is one price
    either way."""
    def swap(far_leg, near_side, near=(2.0, 2.5), far=(60.0, 62.0), spot=1.2000, scale=10_000.0):
        return qf.fx_swap_outrights(spot=spot, near_points=near, far_points=far, point_multiplier=scale,
                                    far_leg=far_leg, near_side=near_side)
    sell = swap("sell_base", "same")
    assert sell["side"] == "bid" and sell["near_quote"] == "bid" and sell["near_outright"] == pytest.approx(1.2002)
    assert sell["far_outright"] == pytest.approx(1.2060) and sell["swap_points"] == pytest.approx(58.0)
    buy = swap("buy_base", "same")
    assert buy["side"] == "offer" and buy["far_outright"] == pytest.approx(1.2062) and buy["swap_points"] == 59.5
    bid = swap("sell_base", "opposite")
    assert bid["side"] == "bid" and bid["near_quote"] == "offer" and bid["swap_points"] == pytest.approx(60.0 - 2.5)
    assert bid["near_outright"] == pytest.approx(1.20025) and bid["far_outright"] == pytest.approx(1.2060)
    offer = swap("buy_base", "opposite")
    assert offer["near_quote"] == "bid" and offer["swap_points"] == pytest.approx(62.0 - 2.0)
    assert offer["near_outright"] == pytest.approx(1.2002) and offer["far_outright"] == pytest.approx(1.2062)
    for far_leg in ("sell_base", "buy_base"):
        assert swap(far_leg, "opposite", near=(0.0, 0.0)) == swap(far_leg, "same", near=(0.0, 0.0)) | {
            "near_quote": swap(far_leg, "opposite", near=(0.0, 0.0))["near_quote"]}
    yen = swap("sell_base", "opposite", near=(0.0, 0.0), far=(-210.0, -205.0), spot=150.00, scale=100.0)
    assert yen["far_outright"] == pytest.approx(147.90)
    with pytest.raises(ValueError, match="bid at or below"):
        swap("sell_base", "same", near=(2.5, 2.0))
    with pytest.raises(ValueError, match="near_side"):
        swap("sell_base", "far")
