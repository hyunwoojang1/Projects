"""Module 3: Fear & Greed Index (CNN 방식 근사, 롤링 퍼센타일 랭크 기반).

CNN과의 백테스트 결과 고정 선형 변환이 +20점 과낙관 편향을 일으켜,
모든 컴포넌트를 252거래일 롤링 퍼센타일 랭크로 재설계.

CNN 7개 컴포넌트 → 우리 구현:
  1. Market Momentum      — SPY / SPY_125MA 이탈률 → 퍼센타일 랭크
  2. Stock Price Strength — 섹터 ETF 52주 신고가/신저가 비율 → 퍼센타일 랭크
  3. Stock Price Breadth  — 섹터 ETF 중 50MA 위 비율 → 퍼센타일 랭크
  4. Put/Call (proxy)     — VIX 5일 MA → 퍼센타일 랭크 역전 (실제 P/C 데이터 없음)
  5. Market Volatility    — VIX / VIX_50MA → 퍼센타일 랭크 역전 (CNN 동일)
  6. Safe Haven Demand    — TLT - SPY 20일 수익률 차이 → 퍼센타일 랭크 역전
  7. Junk Bond Demand     — HYG - IEF 20일 수익률 차이 → 퍼센타일 랭크

퍼센타일 랭크 공식:
  score = (과거 RANK_WINDOW일 중 현재값보다 낮은 날 비율) × 100
  역전(invert): score = 100 - score   (높은 값 = 공포인 지표)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_TICKERS = ["XLK", "XLE", "XLF", "XLV", "XLI",
                  "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]

RANK_WINDOW = 252   # 퍼센타일 랭크 기준 거래일 (≈1년)
MIN_HISTORY = 130   # 최소 필요 데이터


def compute(market: pd.DataFrame, fred: pd.DataFrame) -> dict:
    """
    Returns:
        score       — int 0-100
        components  — {name: score 0-100}
        label       — 극단적 공포 / 공포 / 중립 / 탐욕 / 극단적 탐욕
        week_ago    — 5거래일 전 점수
        change      — 이번 주 대비 변화
    """
    components: dict[str, float] = {
        "시장 모멘텀":      _market_momentum(market),
        "주가 강도":        _price_strength(market),
        "주가 폭":          _price_breadth(market),
        "풋/콜 비율":       _put_call_proxy(market),
        "시장 변동성":      _volatility(market),
        "안전자산 수요":    _safe_haven(market),
        "정크본드 수요":    _junk_bond(market),
    }

    score = float(np.mean(list(components.values())))
    score = max(0.0, min(100.0, score))

    week_ago = _score_at_offset(market, fred, offset=5)

    return {
        "score": int(round(score)),
        "components": {k: int(round(v)) for k, v in components.items()},
        "label": _label(score),
        "week_ago": int(round(week_ago)) if week_ago is not None else None,
        "change": int(round(score - week_ago)) if week_ago is not None else None,
    }


# ── 퍼센타일 랭크 헬퍼 ───────────────────────────────────────────────────────

def _pct_rank(series: pd.Series, invert: bool = False,
              window: int = RANK_WINDOW) -> float:
    """시리즈의 마지막 값을 과거 window일 기준 퍼센타일 랭크로 변환.
    look-ahead 방지: 마지막 값은 현재, history는 그 이전 window일만 사용.
    """
    clean = series.dropna()
    if len(clean) < max(window // 4, 30):
        return 50.0

    current = float(clean.iloc[-1])
    # 현재 포함 안 하고 과거 window일만 참조
    history = clean.iloc[-(window + 1):-1]
    if history.empty:
        return 50.0

    rank = float((history < current).mean()) * 100.0
    return max(0.0, min(100.0, 100.0 - rank if invert else rank))


def _pct_change_n(series: pd.Series, n: int) -> float | None:
    clean = series.dropna()
    if len(clean) < n + 1:
        return None
    return float(clean.iloc[-1] / clean.iloc[-n - 1] - 1)


# ── 7개 컴포넌트 ─────────────────────────────────────────────────────────────

def _market_momentum(market: pd.DataFrame) -> float:
    """SPY / SPY_125MA 이탈률 → 퍼센타일 랭크."""
    col = "SPY"
    if col not in market.columns or len(market[col].dropna()) < MIN_HISTORY:
        return 50.0

    spy = market[col].dropna()
    # 이탈률 시리즈: (t일 SPY - t일 125MA) / t일 125MA
    deviation = spy / spy.rolling(125).mean() - 1.0
    return _pct_rank(deviation)


def _price_strength(market: pd.DataFrame) -> float:
    """섹터 ETF 52주 신고가 / (신고가 + 신저가) 비율 → 퍼센타일 랭크.
    벡터화: 각 날짜별 highs/lows를 rolling으로 동시 계산.
    """
    sectors = [t for t in SECTOR_TICKERS if t in market.columns]
    if not sectors or len(market) < MIN_HISTORY:
        return 50.0

    window = 252
    near_highs = pd.DataFrame(index=market.index)
    near_lows = pd.DataFrame(index=market.index)

    for t in sectors:
        s = market[t].dropna().reindex(market.index)
        roll_max = s.rolling(window).max()
        roll_min = s.rolling(window).min()
        near_highs[t] = (s >= roll_max * 0.98).astype(float)
        near_lows[t] = (s <= roll_min * 1.02).astype(float)

    h = near_highs.sum(axis=1)
    l = near_lows.sum(axis=1)
    total = h + l
    ratio = (h / total).where(total > 0, 0.5)
    return _pct_rank(ratio)


def _price_breadth(market: pd.DataFrame) -> float:
    """섹터 ETF 중 50MA 위에 있는 비율 → 퍼센타일 랭크.
    CNN McClellan Volume Summation Index 대체 프록시.
    벡터화: rolling(50).mean() 동시 계산.
    """
    sectors = [t for t in SECTOR_TICKERS if t in market.columns]
    if not sectors or len(market) < MIN_HISTORY:
        return 50.0

    above = pd.DataFrame(index=market.index)
    for t in sectors:
        s = market[t].dropna().reindex(market.index)
        ma50 = s.rolling(50).mean()
        above[t] = (s > ma50).astype(float)

    fraction = above.mean(axis=1).dropna()
    return _pct_rank(fraction)


def _put_call_proxy(market: pd.DataFrame) -> float:
    """VIX 5일 평균 → 퍼센타일 랭크 역전.
    실제 Put/Call 비율 데이터 없음 — VIX 단기 MA로 대체.
    낮은 VIX → 탐욕(높은 점수), 높은 VIX → 공포(낮은 점수).
    """
    col = "^VIX"
    if col not in market.columns or len(market[col].dropna()) < MIN_HISTORY:
        return 50.0

    vix = market[col].dropna()
    vix_5d = vix.rolling(5).mean().dropna()
    return _pct_rank(vix_5d, invert=True)


def _volatility(market: pd.DataFrame) -> float:
    """VIX / VIX_50MA 비율 → 퍼센타일 랭크 역전.
    CNN과 동일하게 50일 MA 기준 사용.
    비율이 높을수록(VIX > MA) 공포 → 역전.
    """
    col = "^VIX"
    if col not in market.columns or len(market[col].dropna()) < 55:
        return 50.0

    vix = market[col].dropna()
    ratio = vix / vix.rolling(50).mean()
    ratio = ratio.dropna()
    return _pct_rank(ratio, invert=True)


def _safe_haven(market: pd.DataFrame) -> float:
    """TLT - SPY 20일 수익률 차이 → 퍼센타일 랭크 역전.
    TLT 아웃퍼폼(양수) = 안전자산 선호 = 공포.
    """
    tlt, spy = "TLT", "SPY"
    if tlt not in market.columns or spy not in market.columns:
        return 50.0
    if len(market) < MIN_HISTORY:
        return 50.0

    tlt_ret = market[tlt].dropna().pct_change(20)
    spy_ret = market[spy].dropna().pct_change(20)
    diff = (tlt_ret - spy_ret).dropna()
    return _pct_rank(diff, invert=True)


def _junk_bond(market: pd.DataFrame) -> float:
    """HYG - IEF 20일 수익률 차이 → 퍼센타일 랭크.
    HYG 아웃퍼폼(양수) = 하이일드 선호 = 탐욕.
    """
    hyg, ief = "HYG", "IEF"
    if hyg not in market.columns or ief not in market.columns:
        return 50.0
    if len(market) < MIN_HISTORY:
        return 50.0

    hyg_ret = market[hyg].dropna().pct_change(20)
    ief_ret = market[ief].dropna().pct_change(20)
    diff = (hyg_ret - ief_ret).dropna()
    return _pct_rank(diff)


# ── Label & Offset ────────────────────────────────────────────────────────────

def _label(score: float) -> str:
    if score < 25:
        return "극단적 공포 (Extreme Fear)"
    if score < 45:
        return "공포 (Fear)"
    if score < 56:
        return "중립 (Neutral)"
    if score < 75:
        return "탐욕 (Greed)"
    return "극단적 탐욕 (Extreme Greed)"


def _score_at_offset(market: pd.DataFrame, fred: pd.DataFrame,
                     offset: int = 5) -> float | None:
    """지난주(offset 거래일 전) 점수 계산."""
    if market.empty or len(market) < offset + MIN_HISTORY:
        return None
    shifted = market.iloc[:-offset].copy()
    try:
        return float(compute(shifted, fred)["score"])
    except Exception:
        return None
