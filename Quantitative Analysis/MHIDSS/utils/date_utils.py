"""날짜 처리 유틸리티."""

from datetime import date, timedelta


def years_before(reference: str, years: int) -> str:
    ref = date.fromisoformat(reference)
    return date(ref.year - years, ref.month, ref.day).isoformat()


def days_before(reference: str, days: int) -> str:
    return (date.fromisoformat(reference) - timedelta(days=days)).isoformat()


def trading_days_between(start: str, end: str) -> int:
    """대략적인 거래일 수 계산 (주말 제외)."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    delta = (e - s).days
    weeks = delta // 7
    remainder = delta % 7
    # 시작일의 요일(0=월요일) 기준
    weekday = s.weekday()
    extra_weekdays = sum(1 for i in range(remainder) if (weekday + i) % 7 < 5)
    return weeks * 5 + extra_weekdays
