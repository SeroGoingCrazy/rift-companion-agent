"""Chinese time-expression parser.

``parse_time_expr(expr, now, current)`` resolves what the slot extractor copied verbatim
(``start_time_expr``) into an absolute datetime. It never guesses an hour:

* Vague expressions (找时间 / 晚上 / 明天 / 周末) -> ``ambiguous=True``, ``value=None``.
* Clock hours 1-11 without a period word (上午/下午/晚上...) are resolved only when
  exactly one of ``h`` / ``h+12`` is still in the future; otherwise ambiguous.
* Relative edits (晚一小时 / 提前半小时) shift ``current``; without it -> ``no_current``.
* When editing (``current`` given) and the expression has no date, ``current``'s date is
  kept and AM/PM is picked closest to ``current`` (改成九点 at 20:00 -> 21:00).
* Resolved datetimes before ``now`` come back with ``reason="past"`` (value kept for the
  reply).

Supported: 今天/明天/后天/大后天/昨天, 今晚/明晚/今早/明早, 周X/星期X/礼拜X with
这/本/下/下下, X月Y日, Y号, YYYY-MM-DD, 凌晨/早上/上午/中午/下午/傍晚/晚上/半夜,
X点/X点半/X点一刻/X点三刻/X点Y分/HH:MM, Chinese or Arabic numerals, X小时后.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Final

# --- result ------------------------------------------------------------------------------


class Reason:
    EMPTY: Final = "empty"
    UNRECOGNIZED: Final = "unrecognized"
    INVALID: Final = "invalid_time"
    VAGUE: Final = "vague"
    AMBIGUOUS_DATE: Final = "ambiguous_date"
    AMBIGUOUS_HOUR: Final = "ambiguous_hour"
    AMBIGUOUS_AMPM: Final = "ambiguous_ampm"
    NO_CURRENT: Final = "no_current"
    PAST: Final = "past"


@dataclass(frozen=True)
class TimeParseResult:
    value: datetime | None
    ambiguous: bool = False
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None and self.reason is None


def _fail(reason: str, *, ambiguous: bool = False) -> TimeParseResult:
    return TimeParseResult(value=None, ambiguous=ambiguous, reason=reason)


# --- numerals ----------------------------------------------------------------------------

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}  # fmt: skip
_NUM = r"[0-9零〇一二两三四五六七八九十]+"


def cn_to_int(text: str) -> int | None:
    """Arabic or Chinese numerals up to 99 (十五, 二十三, 零五, 12)."""
    if text.isdigit():
        return int(text)
    if not text or any(ch not in _CN_DIGITS and ch != "十" for ch in text):
        return None
    if "十" not in text:
        value = 0
        for ch in text:
            value = value * 10 + _CN_DIGITS[ch]
        return value
    tens_part, _, ones_part = text.partition("十")
    if len(tens_part) > 1 or len(ones_part) > 1:
        return None
    tens = _CN_DIGITS[tens_part] if tens_part else 1
    ones = _CN_DIGITS[ones_part] if ones_part else 0
    return tens * 10 + ones


def _quantity(qty: str, half: bool) -> float | None:
    if qty == "半":
        return 0.5
    value = cn_to_int(qty)
    if value is None:
        return None
    return value + (0.5 if half else 0.0)


# --- vocabulary --------------------------------------------------------------------------

_FULL_WIDTH = str.maketrans("０１２３４５６７８９：", "0123456789:")
_PREFIXES = ("那就改成", "那就改到", "改成", "改到", "改为", "换成", "换到", "调到", "调成",
             "挪到", "那就", "定在", "定到", "约在", "约到", "就", "约", "在")  # fmt: skip
_FILLERS = ("左右", "前后", "的时候", "开始", "吧", "呢", "哈", "啊", "呀", "。", "！", "!",
            "~", "～", "，", ",")  # fmt: skip
_VAGUE = ("找时间", "有空", "随时", "都行", "随便", "看情况", "改天", "过几天", "最近", "待定",
          "不确定", "回头", "晚点", "早点", "什么时候")  # fmt: skip

_UNIT = r"(?P<unit>小时|钟头|分钟|分)"
_QTY = r"(?P<qty>[0-9零一二两三四五六七八九十半]+?)个?(?P<half>半)?"
_SHIFT_RE = re.compile(
    r"^再?(?P<dir>往后推|往后挪|往后|推迟|延后|延迟|推后|晚|往前推|往前挪|往前|提前|提早|早)"
    + _QTY + _UNIT + r"$"
)  # fmt: skip
_LATER_DIRS = frozenset({"往后推", "往后挪", "往后", "推迟", "延后", "延迟", "推后", "晚"})
_AFTER_RE = re.compile(r"^" + _QTY + _UNIT + r"(?:以|之)?后$")

# Day words; combined forms carry a period as well.
_DAY_WORDS: tuple[tuple[str, int, str | None], ...] = (
    ("大后天", 3, None), ("后天", 2, None), ("明天", 1, None), ("明日", 1, None),
    ("明儿", 1, None), ("今天", 0, None), ("今日", 0, None), ("今儿", 0, None),
    ("昨天", -1, None), ("明晚", 1, "night"), ("明早", 1, "morning"), ("明晨", 1, "morning"),
    ("今晚", 0, "night"), ("今早", 0, "morning"), ("今夜", 0, "night"), ("明夜", 1, "night"),
)  # fmt: skip
_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
             "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}  # fmt: skip
_WEEKDAY_RE = re.compile(
    r"(?P<pre>下下个?|下个?|这个?|本)?(?:周|星期|礼拜)(?P<d>[一二三四五六日天1-7])"
)
_ISO_DATE_RE = re.compile(r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})")
_MONTH_DAY_RE = re.compile(rf"(?:(?P<m>{_NUM})月)?(?P<d>{_NUM})(?:日|号)")

# Period words, longest first so 下午 wins over 午 and 晚上 over 晚.
_PERIODS: tuple[tuple[str, str], ...] = (
    ("凌晨", "early"), ("清晨", "morning"), ("早上", "morning"), ("早晨", "morning"),
    ("上午", "morning"), ("中午", "noon"), ("下午", "afternoon"), ("傍晚", "evening"),
    ("晚上", "night"), ("夜里", "night"), ("夜晚", "night"), ("半夜", "late"),
    ("深夜", "late"), ("午夜", "late"), ("晚", "night"), ("早", "morning"),
)  # fmt: skip
#: Hours of day (0-23, may wrap past midnight) that count as inside each period.
_PERIOD_HOURS: dict[str, frozenset[int]] = {
    "early": frozenset(range(0, 6)),
    "morning": frozenset(range(5, 12)),
    "noon": frozenset(range(11, 14)),
    "afternoon": frozenset(range(12, 19)),
    "evening": frozenset(range(17, 21)),
    "night": frozenset([*range(18, 24), *range(0, 5)]),
    "late": frozenset([22, 23, *range(0, 5)]),
}

_HHMM_RE = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2})")
_CLOCK_RE = re.compile(
    rf"(?P<h>{_NUM})(?:点|时)钟?(?:(?P<half>半)|(?P<q1>一刻)|(?P<q3>三刻)|(?P<m>{_NUM})分?)?"
)


# --- parsing -----------------------------------------------------------------------------


@dataclass
class _Parts:
    day: date | None = None
    same_weekday_unprefixed: bool = False
    period: str | None = None
    hour: int | None = None
    minute: int = 0
    weekend: bool = False


def _normalize(expr: str) -> str:
    text = re.sub(r"\s+", "", expr.translate(_FULL_WIDTH))
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                text = text[len(prefix) :]
                changed = True
    for filler in _FILLERS:
        text = text.replace(filler, "")
    return text


def _take(pattern: re.Pattern[str], text: str) -> tuple[re.Match[str] | None, str]:
    match = pattern.search(text)
    if match is None:
        return None, text
    return match, text[: match.start()] + text[match.end() :]


def _parse_date(text: str, today: date, parts: _Parts) -> tuple[str, str | None]:
    """Extract the date into ``parts``; returns (remaining text, error reason)."""
    if "周末" in text:
        parts.weekend = True
        return text.replace("周末", ""), None

    for word, offset, period in _DAY_WORDS:
        if word in text:
            parts.day = today + timedelta(days=offset)
            parts.period = period
            return text.replace(word, "", 1), None

    match, rest = _take(_WEEKDAY_RE, text)
    if match:
        target = _WEEKDAYS[match["d"]]
        prefix = (match["pre"] or "").rstrip("个")
        monday = today - timedelta(days=today.weekday())
        if prefix in ("这", "本"):
            parts.day = monday + timedelta(days=target)
        elif prefix == "下":
            parts.day = monday + timedelta(days=7 + target)
        elif prefix == "下下":
            parts.day = monday + timedelta(days=14 + target)
        else:
            parts.day = today + timedelta(days=(target - today.weekday()) % 7)
            parts.same_weekday_unprefixed = parts.day == today
        return rest, None

    match, rest = _take(_ISO_DATE_RE, text)
    if match:
        try:
            parts.day = date(int(match["y"]), int(match["m"]), int(match["d"]))
        except ValueError:
            return rest, Reason.INVALID
        return rest, None

    match, rest = _take(_MONTH_DAY_RE, text)
    if match:
        day = cn_to_int(match["d"])
        month = cn_to_int(match["m"]) if match["m"] else today.month
        if day is None or month is None:
            return rest, Reason.INVALID
        try:
            candidate = date(today.year, month, day)
        except ValueError:
            return rest, Reason.INVALID
        if candidate < today:
            # "3号" late in the month means next month; "1月5日" in October means next year.
            if match["m"]:
                candidate = candidate.replace(year=today.year + 1)
            else:
                year, month = (today.year + 1, 1) if today.month == 12 else (today.year, month + 1)
                try:
                    candidate = date(year, month, day)
                except ValueError:
                    return rest, Reason.INVALID
        parts.day = candidate
        return rest, None
    return text, None


def _parse_period(text: str, parts: _Parts) -> str:
    for word, period in _PERIODS:
        if word in text:
            parts.period = parts.period or period
            return text.replace(word, "", 1)
    return text


def _parse_clock(text: str, parts: _Parts) -> tuple[str, str | None]:
    match, rest = _take(_HHMM_RE, text)
    if match:
        parts.hour, parts.minute = int(match["h"]), int(match["m"])
    else:
        match, rest = _take(_CLOCK_RE, text)
        if match is None:
            return text, None
        hour = cn_to_int(match["h"])
        if hour is None:
            return rest, Reason.INVALID
        parts.hour = hour
        if match["half"]:
            parts.minute = 30
        elif match["q1"]:
            parts.minute = 15
        elif match["q3"]:
            parts.minute = 45
        elif match["m"]:
            minute = cn_to_int(match["m"])
            if minute is None:
                return rest, Reason.INVALID
            parts.minute = minute
    if not (0 <= parts.hour <= 24 and 0 <= parts.minute <= 59):
        return rest, Reason.INVALID
    return rest, None


def _apply_period(hour: int, period: str) -> int:
    """Convert a spoken hour to 0-24+ given its period; >= 24 means the next day."""
    if period == "early":
        return 0 if hour == 12 else hour
    if period == "morning":
        return hour
    if period == "noon":
        return hour + 12 if 1 <= hour <= 5 else hour
    if period in ("afternoon", "evening"):
        return hour + 12 if hour < 12 else hour
    # night / late: 八点 -> 20, 十二点 -> 24 (midnight), 一点 -> 25 (1am next day)
    if period == "night" and 6 <= hour <= 11:
        return hour + 12
    if period == "late" and hour == 11:
        return 23
    if hour == 12:
        return 24
    if 0 < hour <= 5:
        return hour + 24
    return hour


def _at(day: date, hour: int, minute: int, now: datetime) -> datetime:
    """``day`` at ``hour``:``minute`` where hour may be >= 24 (rolls into the next day)."""
    extra_days, hour = divmod(hour, 24)
    return datetime.combine(day + timedelta(days=extra_days), time(hour, minute), now.tzinfo)


def _finish(value: datetime, now: datetime) -> TimeParseResult:
    if value < now:
        return TimeParseResult(value=value, reason=Reason.PAST)
    return TimeParseResult(value=value)


def _shift(match: re.Match[str], base: datetime, sign: int) -> TimeParseResult | None:
    amount = _quantity(match["qty"], bool(match["half"]))
    if amount is None:
        return None
    unit = match["unit"]
    delta = timedelta(hours=amount) if unit in ("小时", "钟头") else timedelta(minutes=amount)
    return TimeParseResult(value=base + sign * delta)


def parse_time_expr(
    expr: str | None, now: datetime, current: datetime | None = None
) -> TimeParseResult:
    """Resolve a Chinese time expression; see the module docstring for the rules."""
    if expr is None or not expr.strip():
        return _fail(Reason.EMPTY)
    text = _normalize(expr)
    if not text:
        return _fail(Reason.EMPTY)

    # Relative edits of the current booking time.
    shift = _SHIFT_RE.match(text)
    if shift:
        if current is None:
            return _fail(Reason.NO_CURRENT)
        sign = 1 if shift["dir"] in _LATER_DIRS else -1
        shifted = _shift(shift, current, sign)
        return _finish(shifted.value, now) if shifted and shifted.value else _fail(Reason.INVALID)
    after = _AFTER_RE.match(text)
    if after:
        result = _shift(after, now.replace(second=0, microsecond=0), 1)
        return result if result else _fail(Reason.INVALID)

    parts = _Parts()
    rest, error = _parse_date(text, now.date(), parts)
    if error:
        return _fail(error)
    rest, error = _parse_clock(rest, parts)
    if error:
        return _fail(error)
    rest = _parse_period(rest, parts)

    if parts.hour is None and any(word in text for word in _VAGUE):
        return _fail(Reason.VAGUE, ambiguous=True)
    if parts.weekend:
        return _fail(Reason.AMBIGUOUS_DATE, ambiguous=True)
    if parts.day is None and parts.hour is None and parts.period is None:
        return _fail(Reason.UNRECOGNIZED)

    if parts.hour is None:
        return _resolve_without_hour(parts, now, current)
    return _resolve_with_hour(parts, now, current)


def _resolve_without_hour(
    parts: _Parts, now: datetime, current: datetime | None
) -> TimeParseResult:
    """Date and/or period only: reuse ``current``'s clock time when it fits, else ambiguous."""
    if current is not None and parts.day is not None:
        fits = parts.period is None or current.hour in _PERIOD_HOURS[parts.period]
        if fits:
            return _finish(_at(parts.day, current.hour, current.minute, now), now)
    return _fail(Reason.AMBIGUOUS_HOUR, ambiguous=True)


def _resolve_with_hour(parts: _Parts, now: datetime, current: datetime | None) -> TimeParseResult:
    assert parts.hour is not None
    hour, minute = parts.hour, parts.minute
    explicit_day = parts.day is not None
    day = parts.day or (current.date() if current is not None else now.date())

    if parts.period is not None:
        candidates = [_apply_period(hour, parts.period)]
    elif 1 <= hour <= 11:
        candidates = [hour, hour + 12]
    else:
        candidates = [hour]

    if len(candidates) == 1:
        value = _at(day, candidates[0], minute, now)
        if not explicit_day and current is None and value < now:
            value += timedelta(days=1)  # "晚上八点" said at 21:00 -> tomorrow
        elif parts.same_weekday_unprefixed and value < now:
            value += timedelta(days=7)  # "周四上午十点" said on Thursday afternoon
        return _finish(value, now)

    # AM/PM unknown.
    options = [_at(day, h, minute, now) for h in candidates]
    if current is not None and not explicit_day:
        return _finish(min(options, key=lambda v: abs(v - current)), now)
    future = [v for v in options if v >= now]
    if not future and not explicit_day:
        options = [v + timedelta(days=1) for v in options]
        future = options
    if not future and parts.same_weekday_unprefixed:
        future = [v + timedelta(days=7) for v in options]
    if len(future) == 1:
        return _finish(future[0], now)
    if not future:
        return TimeParseResult(value=options[-1], reason=Reason.PAST)
    return _fail(Reason.AMBIGUOUS_AMPM, ambiguous=True)
