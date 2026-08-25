"""Module 4: Sector Rotation Signal.

11 SPDR sector ETFs ranked by weighted multi-timeframe momentum vs SPY.
Momentum weights: 1M 50%, 3M 30%, 6M 20%.
Output: top 3 / bottom 3 sectors, rotation direction, business cycle position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_NAMES = {
    "XLK": "기술 (Technology)",
    "XLE": "에너지 (Energy)",
    "XLF": "금융 (Financials)",
    "XLV": "헬스케어 (Health Care)",
    "XLI": "산업재 (Industrials)",
    "XLY": "임의소비재 (Consumer Discr.)",
    "XLP": "필수소비재 (Consumer Staples)",
    "XLB": "소재 (Materials)",
    "XLRE": "부동산 (Real Estate)",
    "XLU": "유틸리티 (Utilities)",
    "XLC": "커뮤니케이션 (Communication)",
}

# Weights: 1M, 3M, 6M
MOMENTUM_WEIGHTS = {21: 0.50, 63: 0.30, 126: 0.20}

# Business cycle sectors: which sectors typically lead each phase
CYCLE_PROFILE = {
    "확장기": ["XLK", "XLY", "XLF", "XLI"],
    "후기확장": ["XLE", "XLB", "XLI", "XLF"],
    "수축기": ["XLP", "XLV", "XLU", "XLRE"],
    "회복기": ["XLF", "XLY", "XLK", "XLC"],
}


def compute(market: pd.DataFrame) -> dict:
    """
    Args:
        market: market DataFrame with sector ETF and SPY columns

    Returns dict with:
        scores        — {ticker: weighted_zscore}
        ranked        — list of (ticker, name, zscore) sorted desc
        top3          — list of top 3 tickers
        bottom3       — list of bottom 3 tickers
        rotation_dir  — "방어 → 성장" / "성장 → 방어" / "혼조"
        cycle_phase   — estimated business cycle phase
        momentum_data — raw momentum by horizon
    """
    sectors = [t for t in SECTOR_NAMES if t in market.columns]
    spy = "SPY"

    if not sectors or spy not in market.columns:
        return _empty_result()

    scores: dict[str, float] = {}
    momentum_data: dict[str, dict] = {}

    for ticker in sectors:
        raw_scores: dict[int, float] = {}
        for window, weight in MOMENTUM_WEIGHTS.items():
            rel = _relative_momentum(market[ticker], market[spy], window)
            raw_scores[window] = rel if rel is not None else 0.0
        weighted = sum(MOMENTUM_WEIGHTS[w] * raw_scores[w] for w in MOMENTUM_WEIGHTS)
        scores[ticker] = weighted
        momentum_data[ticker] = {
            "1M": round(raw_scores[21] * 100, 2),
            "3M": round(raw_scores[63] * 100, 2),
            "6M": round(raw_scores[126] * 100, 2),
        }

    # Z-score normalization
    values = np.array(list(scores.values()))
    if values.std() > 0:
        zscores = {t: float((scores[t] - values.mean()) / values.std())
                   for t in sectors}
    else:
        zscores = {t: 0.0 for t in sectors}

    ranked = sorted(zscores.items(), key=lambda x: x[1], reverse=True)
    top3 = [t for t, _ in ranked[:3]]
    bottom3 = [t for t, _ in ranked[-3:]]

    ranked_full = [(t, SECTOR_NAMES[t], round(z, 3)) for t, z in ranked]

    return {
        "scores": zscores,
        "ranked": ranked_full,
        "top3": top3,
        "bottom3": bottom3,
        "rotation_dir": _rotation_direction(top3, bottom3),
        "cycle_phase": _estimate_cycle(top3),
        "momentum_data": momentum_data,
    }


def _relative_momentum(sector: pd.Series, spy: pd.Series,
                        window: int) -> float | None:
    sector_clean = sector.dropna()
    spy_clean = spy.dropna()
    common = sector_clean.index.intersection(spy_clean.index)
    if len(common) < window + 1:
        return None

    sector_ret = float(sector_clean.loc[common].iloc[-1] /
                       sector_clean.loc[common].iloc[-window - 1] - 1)
    spy_ret = float(spy_clean.loc[common].iloc[-1] /
                    spy_clean.loc[common].iloc[-window - 1] - 1)
    return sector_ret - spy_ret


def _rotation_direction(top3: list[str], bottom3: list[str]) -> str:
    defensive = {"XLP", "XLV", "XLU", "XLRE"}
    growth = {"XLK", "XLY", "XLC", "XLF"}

    top_defensive = sum(1 for t in top3 if t in defensive)
    top_growth = sum(1 for t in top3 if t in growth)

    if top_growth >= 2 and top_defensive == 0:
        return "방어 → 성장 (위험선호 강화)"
    if top_defensive >= 2 and top_growth == 0:
        return "성장 → 방어 (위험회피 강화)"
    return "혼조 (섹터 간 분산)"


def _estimate_cycle(top3: list[str]) -> str:
    scores = {phase: 0 for phase in CYCLE_PROFILE}
    for phase, sectors in CYCLE_PROFILE.items():
        for t in top3:
            if t in sectors:
                scores[phase] += 1
    best = max(scores, key=lambda p: scores[p])
    return best if scores[best] > 0 else "불확실"


def _empty_result() -> dict:
    return {
        "scores": {}, "ranked": [], "top3": [], "bottom3": [],
        "rotation_dir": "데이터 없음", "cycle_phase": "불확실",
        "momentum_data": {},
    }
