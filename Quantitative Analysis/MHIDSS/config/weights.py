"""시계열별 가중치 행렬 레지스트리.

v2.0 변경 사항:
- Short=일봉, Mid=주봉, Long=월봉 해상도 분리에 따른 가중치 재조정
- Long-term 기술적 가중치: 0.05 → 0.25 (월봉 RSI/MACD/SMA는 유효한 장기 신호)
- Mid-term 기술적 가중치: 0.35 → 0.40 (주봉이 일봉보다 잡음이 적음)
"""

from typing import Literal

Horizon = Literal["short", "mid", "long"]

# ── 그룹 간 가중치 (합계 = 1.0) ──────────────────────────────────────────────
# Short (일봉): 단기 가격 모멘텀이 지배 → Tech 70%
# Mid  (주봉): 주봉 기술적 신호 비중 상향 → Tech 40%
# Long (월봉): 월봉 기술적 지표가 의미 있어 Tech 25%로 상향,
#              Macro+Fund는 여전히 75%로 지배
HORIZON_GROUP_WEIGHTS: dict[Horizon, dict[str, float]] = {
    "short": {"macro": 0.20, "fundamental": 0.10, "technical": 0.70},
    "mid":   {"macro": 0.30, "fundamental": 0.30, "technical": 0.40},
    "long":  {"macro": 0.40, "fundamental": 0.35, "technical": 0.25},
}

# ── 매크로 그룹 내 지표별 가중치 (합계 = 1.0) ────────────────────────────────
MACRO_INDICATOR_WEIGHTS: dict[Horizon, dict[str, float]] = {
    "short": {
        "YIELD_CURVE_SPREAD": 0.15,
        "FEDFUNDS":           0.15,
        "CPIAUCSL":           0.10,
        "PCEPILFE":           0.10,
        "CREDIT_SPREAD":      0.20,
        "UNRATE":             0.10,
        "M2SL":               0.05,
        "ICSA":               0.15,
    },
    "mid": {
        "YIELD_CURVE_SPREAD": 0.20,
        "FEDFUNDS":           0.20,
        "CPIAUCSL":           0.15,
        "PCEPILFE":           0.10,
        "CREDIT_SPREAD":      0.15,
        "UNRATE":             0.10,
        "M2SL":               0.05,
        "ICSA":               0.05,
    },
    "long": {
        "YIELD_CURVE_SPREAD": 0.25,
        "FEDFUNDS":           0.20,
        "CPIAUCSL":           0.15,
        "PCEPILFE":           0.10,
        "CREDIT_SPREAD":      0.15,
        "UNRATE":             0.10,
        "M2SL":               0.05,
        "ICSA":               0.00,
    },
}

# ── 펀더멘탈 그룹 내 지표별 가중치 (합계 = 1.0) ──────────────────────────────
FUNDAMENTAL_INDICATOR_WEIGHTS: dict[Horizon, dict[str, float]] = {
    "short": {
        "eps_change_rate": 0.30,
        "roe":             0.15,
        "fcf_yield":       0.15,
        "pbr":             0.10,
        "revenue_growth":  0.15,
        "de_ratio":        0.10,
        "earnings_yield":  0.05,
    },
    "mid": {
        "eps_change_rate": 0.25,
        "roe":             0.20,
        "fcf_yield":       0.20,
        "pbr":             0.15,
        "revenue_growth":  0.10,
        "de_ratio":        0.05,
        "earnings_yield":  0.05,
    },
    "long": {
        "eps_change_rate": 0.15,
        "roe":             0.25,
        "fcf_yield":       0.25,
        "pbr":             0.20,
        "revenue_growth":  0.10,
        "de_ratio":        0.05,
        "earnings_yield":  0.00,
    },
}

# ── 기술적 그룹 내 지표별 가중치 (합계 = 1.0) ────────────────────────────────
#
# Short (일봉): 오버솔드/오버보트 + 단기 모멘텀 중심
# Mid  (주봉): SMA 추세 비중 상향 (주봉 SMA20/100은 핵심 지표)
# Long (월봉): 추세(SMA10/40) + MACD + OBV 중심
#              월봉 Stochastic/Bollinger %B는 노이즈가 많아 제외
TECHNICAL_INDICATOR_WEIGHTS: dict[Horizon, dict[str, float]] = {
    "short": {
        "rsi_14":         0.20,
        "macd_histogram": 0.20,
        "sma_ratio":      0.10,
        "stoch_k":        0.15,
        "bb_pct_b":       0.15,
        "obv_slope":      0.10,
        "atr_norm":       0.05,
        "roc":            0.05,
    },
    "mid": {
        "rsi_14":         0.15,
        "macd_histogram": 0.20,
        "sma_ratio":      0.25,   # 주봉 SMA20/100 비중 상향
        "stoch_k":        0.10,
        "bb_pct_b":       0.10,
        "obv_slope":      0.15,   # 주봉 OBV slope 상향
        "atr_norm":       0.00,   # 주봉에서 변동성 제거
        "roc":            0.05,
    },
    "long": {
        # 월봉에서 추세/모멘텀 지표가 의미 있음
        "rsi_14":         0.15,   # 월봉 RSI: 구조적 과매수/과매도 판단
        "macd_histogram": 0.20,   # 월봉 MACD: 장기 모멘텀 전환점
        "sma_ratio":      0.30,   # 월봉 SMA10/40: 장기 추세의 핵심
        "stoch_k":        0.00,   # 월봉에서 Stochastic은 노이즈
        "bb_pct_b":       0.00,   # 월봉 Bollinger 의미 약함
        "obv_slope":      0.20,   # 월봉 OBV: 기관 매집/분산 확인
        "atr_norm":       0.05,   # 월봉 변동성 (소폭 반영)
        "roc":            0.10,   # 6개월 ROC
    },
}

# 버전 태그 (리포트 메타데이터에 포함)
WEIGHT_VERSION = "v2.0.0"
