from datetime import datetime, timedelta, timezone

import pytest

from rift_domain.timeparse import Reason, TimeParseResult, cn_to_int, parse_time_expr

# Thursday afternoon, as in the DEV_SPEC L4 example.
NOW = datetime(2026, 10, 1, 14, 0)
CURRENT = datetime(2026, 10, 2, 20, 0)  # an already-agreed booking time: tomorrow 20:00


def dt(month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute)


# --- absolute expressions ---

ABSOLUTE = [
    ("明晚八点", dt(10, 2, 20)),
    ("今晚八点", dt(10, 1, 20)),
    ("今晚8点半", dt(10, 1, 20, 30)),
    ("明天晚上八点", dt(10, 2, 20)),
    ("明天下午三点", dt(10, 2, 15)),
    ("明天上午十点", dt(10, 2, 10)),
    ("后天晚上9点", dt(10, 3, 21)),
    ("大后天下午两点", dt(10, 4, 14)),
    ("明天20:30", dt(10, 2, 20, 30)),
    ("明天20点", dt(10, 2, 20)),
    ("今天下午四点一刻", dt(10, 1, 16, 15)),
    ("今晚七点三刻", dt(10, 1, 19, 45)),
    ("明晚8点15分", dt(10, 2, 20, 15)),
    ("明晚八点零五", dt(10, 2, 20, 5)),
    ("明晚十点二十", dt(10, 2, 22, 20)),
    ("明晚十一点", dt(10, 2, 23)),
    ("周六晚上八点", dt(10, 3, 20)),
    ("星期天下午两点", dt(10, 4, 14)),
    ("礼拜六晚上9点", dt(10, 3, 21)),
    ("下周一晚上八点", dt(10, 5, 20)),
    ("下周六晚上八点", dt(10, 10, 20)),
    ("下个星期三下午3点", dt(10, 7, 15)),
    ("下下周一晚上八点", dt(10, 12, 20)),
    ("这周四晚上八点", dt(10, 1, 20)),
    ("周四晚上八点", dt(10, 1, 20)),  # today is Thursday and it is still ahead
    ("周四上午十点", dt(10, 8, 10)),  # today's 10:00 has passed -> next Thursday
    ("10月3号晚上八点", dt(10, 3, 20)),
    ("十月三日晚上八点", dt(10, 3, 20)),
    ("3号晚上8点", dt(10, 3, 20)),
    ("5日下午3点", dt(10, 5, 15)),
    ("2026-10-02 20:00", dt(10, 2, 20)),
    ("明天凌晨一点", dt(10, 2, 1)),
    ("今晚十二点", dt(10, 2, 0)),
    ("今晚一点", dt(10, 2, 1)),
    ("明天中午12点", dt(10, 2, 12)),
    ("明天中午一点", dt(10, 2, 13)),
    ("明天傍晚六点", dt(10, 2, 18)),
    ("明早九点", dt(10, 2, 9)),
    ("晚上八点", dt(10, 1, 20)),
    ("八点", dt(10, 1, 20)),  # 08:00 already passed, only 20:00 is ahead
    ("下午三点", dt(10, 1, 15)),
    ("上午十点", dt(10, 2, 10)),  # passed today -> tomorrow
    ("21点", dt(10, 1, 21)),
    ("今天八点", dt(10, 1, 20)),  # explicit day, only 20:00 is still ahead
    ("明晚八点左右", dt(10, 2, 20)),
    ("明晚８点", dt(10, 2, 20)),  # full-width digit
    ("那就明晚八点吧", dt(10, 2, 20)),
    ("约在明天晚上 8 点", dt(10, 2, 20)),
    ("一小时后", dt(10, 1, 15)),
    ("半小时之后", dt(10, 1, 14, 30)),
    ("30分钟后", dt(10, 1, 14, 30)),
]


@pytest.mark.parametrize(("expr", "expected"), ABSOLUTE)
def test_absolute_expressions(expr: str, expected: datetime) -> None:
    result = parse_time_expr(expr, NOW)
    assert result == TimeParseResult(value=expected), expr
    assert result.ok


# --- relative to the current booking time ---

RELATIVE = [
    ("晚一小时", dt(10, 2, 21)),
    ("改成晚一小时", dt(10, 2, 21)),
    ("再晚一个小时", dt(10, 2, 21)),
    ("提前半小时", dt(10, 2, 19, 30)),
    ("早一小时", dt(10, 2, 19)),
    ("推迟一个半小时", dt(10, 2, 21, 30)),
    ("晚30分钟", dt(10, 2, 20, 30)),
    ("往后推两个小时", dt(10, 2, 22)),
    ("提早半个小时", dt(10, 2, 19, 30)),
    ("改成九点", dt(10, 2, 21)),  # keeps current's date, picks the PM closest to 20:00
    ("改成九点半", dt(10, 2, 21, 30)),
    ("改到后天", dt(10, 3, 20)),  # date only: keep current's clock time
    ("改到周六", dt(10, 3, 20)),
    ("换成后天晚上", dt(10, 3, 20)),
    ("改到晚上十点", dt(10, 2, 22)),
]


@pytest.mark.parametrize(("expr", "expected"), RELATIVE)
def test_relative_to_current(expr: str, expected: datetime) -> None:
    assert parse_time_expr(expr, NOW, current=CURRENT) == TimeParseResult(value=expected), expr


def test_relative_without_current_is_unresolvable() -> None:
    result = parse_time_expr("晚一小时", NOW)
    assert result == TimeParseResult(value=None, ambiguous=False, reason=Reason.NO_CURRENT)
    assert not result.ok


# --- ambiguous: never guess the hour ---

AMBIGUOUS = [
    ("晚上", Reason.AMBIGUOUS_HOUR),
    ("明天", Reason.AMBIGUOUS_HOUR),
    ("明晚", Reason.AMBIGUOUS_HOUR),
    ("周六下午", Reason.AMBIGUOUS_HOUR),
    ("下周三", Reason.AMBIGUOUS_HOUR),
    ("明天八点", Reason.AMBIGUOUS_AMPM),
    ("周六3点", Reason.AMBIGUOUS_AMPM),
    ("找时间", Reason.VAGUE),
    ("有空再说", Reason.VAGUE),
    ("随时都行", Reason.VAGUE),
    ("明天晚点", Reason.VAGUE),
    ("周末", Reason.AMBIGUOUS_DATE),
    ("周末晚上八点", Reason.AMBIGUOUS_DATE),
    ("明晚后天九点", Reason.AMBIGUOUS_DATE),
    ("明天周六晚上八点", Reason.AMBIGUOUS_DATE),
]


@pytest.mark.parametrize(("expr", "reason"), AMBIGUOUS)
def test_ambiguous_expressions(expr: str, reason: str) -> None:
    assert parse_time_expr(expr, NOW) == TimeParseResult(value=None, ambiguous=True, reason=reason)


def test_period_mismatch_with_current_stays_ambiguous() -> None:
    # current is 20:00; "明天上午" gives no hour and 20:00 is not a morning hour
    result = parse_time_expr("明天上午", NOW, current=CURRENT)
    assert result.ambiguous and result.reason == Reason.AMBIGUOUS_HOUR


# --- past / invalid / unrecognized ---


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("今天上午十点", dt(10, 1, 10)),
        ("昨天晚上八点", dt(9, 30, 20)),
        ("本周二晚上八点", dt(9, 29, 20)),
        ("今天上午八点", dt(10, 1, 8)),
    ],
)
def test_past_times(expr: str, value: datetime) -> None:
    result = parse_time_expr(expr, NOW)
    assert result == TimeParseResult(value=value, reason=Reason.PAST)
    assert not result.ok


def test_past_relative_shift() -> None:
    early = parse_time_expr("提前八个小时", NOW, current=datetime(2026, 10, 1, 20))
    assert early == TimeParseResult(value=dt(10, 1, 12), reason=Reason.PAST)


@pytest.mark.parametrize(
    ("expr", "reason"),
    [
        ("", Reason.EMPTY),
        ("   ", Reason.EMPTY),
        ("abc", Reason.UNRECOGNIZED),
        ("去网吧", Reason.UNRECOGNIZED),
        ("提前两天", Reason.UNRECOGNIZED),
        ("明天25点", Reason.INVALID),
        ("明晚8点70分", Reason.INVALID),
        ("10月32号晚上八点", Reason.INVALID),
        ("2026-02-30 20:00", Reason.INVALID),
        ("十二十号晚上八点", Reason.INVALID),
        ("明天十二十点", Reason.INVALID),
        ("明晚八点十二十分", Reason.INVALID),
        ("十二十小时后", Reason.INVALID),
        ("吧", Reason.EMPTY),
    ],
)
def test_unparseable(expr: str, reason: str) -> None:
    assert parse_time_expr(expr, NOW) == TimeParseResult(value=None, reason=reason)


def test_invalid_shift_amount() -> None:
    result = parse_time_expr("晚十二十小时", NOW, current=CURRENT)
    assert result == TimeParseResult(value=None, reason=Reason.INVALID)


def test_next_month_without_that_day_is_invalid() -> None:
    # Jan 31st: "30号" has passed this month and February has no 30th
    result = parse_time_expr("30号晚上八点", datetime(2026, 1, 31, 10))
    assert result == TimeParseResult(value=None, reason=Reason.INVALID)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [("深夜十一点", dt(10, 1, 23)), ("晚上20点", dt(10, 1, 20)), ("半夜十二点", dt(10, 2, 0))],
)
def test_late_night_periods(expr: str, expected: datetime) -> None:
    assert parse_time_expr(expr, NOW).value == expected


def test_explicit_past_day_without_period_is_past() -> None:
    result = parse_time_expr("昨天八点", NOW)
    assert result == TimeParseResult(value=datetime(2026, 9, 30, 20), reason=Reason.PAST)


def test_same_weekday_ampm_rolls_to_next_week_then_stays_ambiguous() -> None:
    thursday_night = datetime(2026, 10, 1, 21)
    result = parse_time_expr("周四八点", thursday_night)
    assert result.reason == Reason.AMBIGUOUS_AMPM


def test_none_expression() -> None:
    assert parse_time_expr(None, NOW).reason == Reason.EMPTY


# --- calendar rollovers ---


def test_day_of_month_before_today_rolls_to_next_month() -> None:
    mid_month = datetime(2026, 10, 15, 14)
    assert parse_time_expr("1号晚上八点", mid_month).value == datetime(2026, 11, 1, 20)
    assert parse_time_expr("1号晚上八点", NOW).value == datetime(2026, 10, 1, 20)  # today


def test_month_day_before_today_rolls_to_next_year() -> None:
    assert parse_time_expr("1月5号晚上八点", NOW).value == datetime(2027, 1, 5, 20)


def test_december_day_rolls_into_january() -> None:
    now = datetime(2026, 12, 20, 10)
    assert parse_time_expr("3号晚上八点", now).value == datetime(2027, 1, 3, 20)


def test_late_evening_time_rolls_to_tomorrow() -> None:
    now = datetime(2026, 10, 1, 21, 0)
    assert parse_time_expr("晚上八点", now).value == datetime(2026, 10, 2, 20)
    assert parse_time_expr("八点", now).reason == Reason.AMBIGUOUS_AMPM


def test_timezone_is_preserved() -> None:
    tz = timezone(timedelta(hours=8))
    result = parse_time_expr("明晚八点", NOW.replace(tzinfo=tz))
    assert result.value == datetime(2026, 10, 2, 20, tzinfo=tz)


# --- numerals ---


@pytest.mark.parametrize(
    ("text", "value"),
    [("8", 8), ("20", 20), ("八", 8), ("两", 2), ("十", 10), ("十一", 11), ("二十", 20),
     ("二十三", 23), ("零五", 5), ("四十五", 45), ("十二十", None), ("abc", None), ("", None)],
)  # fmt: skip
def test_cn_to_int(text: str, value: int | None) -> None:
    assert cn_to_int(text) == value
