"""Module 5: Macro Environment Score (0-100).

5 FRED indicators, each 0-20 points:
  1. 10Y rate direction      — falling = favorable for equities
  2. Yield curve             — 10Y-2Y spread positive & widening
  3. Dollar strength (DXY)  — falling = risk-on
  4. Inflation trend (CPI)  — falling YoY = rate-cut expectations
  5. Liquidity (M2)          — rising = ample liquidity
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(fred: pd.DataFrame, market: pd.DataFrame) -> dict:
    """
    Args:
        fred:   FRED DataFrame (columns: DGS10, DGS2, CPIAUCSL, M2SL, SPREAD_10Y2Y, …)
        market: market DataFrame (column: DX-Y.NYB for DXY)

    Returns dict with:
        score       — int 0-100
        components  — dict of component name → score (0-20)
        details     — dict of component name → (value, description)
        label       — str description of macro environment
    """
    components: dict[str, float] = {}
    details: dict[str, dict] = {}

    # 1. 10Y rate direction (20 pts) — 20-day change; falling = 20, rising = 0
    components["rate_direction"] = _score_rate_direction(fred, details)

    # 2. Yield curve (20 pts) — spread level; > 0.5% & widening = 20
    components["yield_curve"] = _score_yield_curve(fred, details)

    # 3. Dollar strength (20 pts) — DXY 20-day change; falling = 20
    components["dollar"] = _score_dollar(market, details)

    # 4. Inflation trend (20 pts) — CPI YoY direction; falling = 20
    components["inflation"] = _score_inflation(fred, details)

    # 5. Liquidity (20 pts) — M2 YoY growth; > 5% = 20, < 0% = 0
    components["liquidity"] = _score_liquidity(fred, details)

    total = int(round(sum(components.values())))
    total = max(0, min(100, total))

    return {
        "score": total,
        "components": components,
        "details": details,
        "label": _label(total),
    }


# ── Component scorers ─────────────────────────────────────────────────────────

def _score_rate_direction(fred: pd.DataFrame,
                          details: dict) -> float:
    col = "DGS10"
    if col not in fred.columns or fred[col].dropna().empty:
        details["rate_direction"] = {"value": None, "desc": "데이터 없음"}
        return 10.0  # neutral

    series = fred[col].dropna()
    change_20d = float(series.iloc[-1] - series.iloc[max(-21, -len(series))])

    # Linear: -1pp change → 20pts, +1pp change → 0pts
    score = float(np.clip(20 - change_20d * 20, 0, 20))
    details["rate_direction"] = {
        "value": round(change_20d, 3),
        "desc": f"10Y 금리 20일 변화 {change_20d:+.3f}%p",
    }
    return score


def _score_yield_curve(fred: pd.DataFrame, details: dict) -> float:
    col = "SPREAD_10Y2Y"
    if col not in fred.columns or fred[col].dropna().empty:
        details["yield_curve"] = {"value": None, "desc": "데이터 없음"}
        return 10.0

    series = fred[col].dropna()
    spread = float(series.iloc[-1])
    spread_prev = float(series.iloc[max(-21, -len(series))])
    widening = spread - spread_prev

    level_score = float(np.clip((spread + 1) / 2 * 10, 0, 10))   # -1~+1pp → 0~10
    direction_score = float(np.clip(widening * 20 + 5, 0, 10))     # expanding bonus

    score = level_score + direction_score
    details["yield_curve"] = {
        "value": round(spread, 3),
        "desc": f"10Y-2Y 스프레드 {spread:+.3f}%p ({'+' if widening >= 0 else ''}{widening:.3f}%p 변화)",
    }
    return score


def _score_dollar(market: pd.DataFrame, details: dict) -> float:
    col = "DX-Y.NYB"
    if market.empty or col not in market.columns:
        details["dollar"] = {"value": None, "desc": "데이터 없음"}
        return 10.0

    series = market[col].dropna()
    if len(series) < 2:
        details["dollar"] = {"value": None, "desc": "데이터 부족"}
        return 10.0

    current = float(series.iloc[-1])
    prev = float(series.iloc[max(-21, -len(series))])
    pct_change = (current - prev) / prev * 100

    # DXY falling = good for risk assets → high score
    score = float(np.clip(10 - pct_change * 5, 0, 20))
    details["dollar"] = {
        "value": round(pct_change, 2),
        "desc": f"DXY 20일 변화 {pct_change:+.2f}%",
    }
    return score


def _score_inflation(fred: pd.DataFrame, details: dict) -> float:
    col = "CPIAUCSL"
    if col not in fred.columns or fred[col].dropna().empty:
        details["inflation"] = {"value": None, "desc": "데이터 없음"}
        return 10.0

    series = fred[col].dropna()
    if len(series) < 13:
        details["inflation"] = {"value": None, "desc": "데이터 부족"}
        return 10.0

    # YoY CPI change
    cpi_now = float(series.iloc[-1])
    cpi_12m = float(series.iloc[-13])
    yoy = (cpi_now - cpi_12m) / cpi_12m * 100

    # YoY trend (compare to 6 months ago)
    cpi_6m = float(series.iloc[-7]) if len(series) >= 7 else cpi_12m
    yoy_6m = (cpi_now - cpi_6m) / cpi_6m * 200  # annualized proxy
    falling = yoy_6m < yoy  # most recent pace is below YoY average

    base_score = float(np.clip((5 - yoy) * 2, 0, 10))  # CPI < 2% → ~20, > 7% → 0
    trend_bonus = 10.0 if falling else 0.0
    score = min(base_score + trend_bonus, 20.0)

    details["inflation"] = {
        "value": round(yoy, 2),
        "desc": f"CPI YoY {yoy:.2f}% ({'하락' if falling else '상승'} 추세)",
    }
    return score


def _score_liquidity(fred: pd.DataFrame, details: dict) -> float:
    col = "M2SL"
    if col not in fred.columns or fred[col].dropna().empty:
        details["liquidity"] = {"value": None, "desc": "데이터 없음"}
        return 10.0

    series = fred[col].dropna()
    if len(series) < 13:
        details["liquidity"] = {"value": None, "desc": "데이터 부족"}
        return 10.0

    m2_now = float(series.iloc[-1])
    m2_12m = float(series.iloc[-13])
    growth = (m2_now - m2_12m) / m2_12m * 100

    # 5%+ growth = 20, 0% = 10, <-5% = 0
    score = float(np.clip((growth + 5) / 10 * 20, 0, 20))
    details["liquidity"] = {
        "value": round(growth, 2),
        "desc": f"M2 YoY 증가율 {growth:+.2f}%",
    }
    return score


def _label(score: int) -> str:
    if score >= 80:
        return "위험자산 매우 유리 (Full Risk-On)"
    if score >= 60:
        return "위험자산 유리"
    if score >= 40:
        return "중립"
    return "위험자산 불리 (Risk-Off)"
