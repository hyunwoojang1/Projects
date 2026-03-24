"""지표별 정규화 전략 매핑 (해상도별 기술적 파라미터 포함)."""

from dataclasses import dataclass, field
from typing import Literal

NormMethod = Literal["minmax", "zscore", "percentile"]


@dataclass(frozen=True)
class NormConfig:
    method: NormMethod
    invert: bool = False
    window_years: int = 10
    fixed_min: float | None = None
    fixed_max: float | None = None


@dataclass(frozen=True)
class TechIndicatorParams:
    """해상도별 기술적 지표 파라미터."""
    rsi_length: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    sma_fast: int       # SMA ratio 분자
    sma_slow: int       # SMA ratio 분모
    bb_length: int
    stoch_k: int
    stoch_d: int
    obv_slope_window: int
    atr_length: int
    roc_length: int
    min_bars: int       # 이 해상도에서 지표 계산에 필요한 최소 봉 수


# ── 해상도별 기술적 지표 파라미터 ────────────────────────────────────────────
TECH_PARAMS: dict[str, TechIndicatorParams] = {
    "daily": TechIndicatorParams(
        rsi_length=14,
        macd_fast=12, macd_slow=26, macd_signal=9,
        sma_fast=50,  sma_slow=200,
        bb_length=20,
        stoch_k=14,   stoch_d=3,
        obv_slope_window=20,
        atr_length=14,
        roc_length=10,
        min_bars=200,   # SMA200 계산 기준
    ),
    "weekly": TechIndicatorParams(
        rsi_length=14,
        macd_fast=12, macd_slow=26, macd_signal=9,
        sma_fast=20,  sma_slow=100,
        bb_length=20,
        stoch_k=14,   stoch_d=3,
        obv_slope_window=20,
        atr_length=14,
        roc_length=10,
        min_bars=100,   # SMA100 계산 기준
    ),
    "monthly": TechIndicatorParams(
        rsi_length=14,
        macd_fast=6,  macd_slow=12, macd_signal=9,
        sma_fast=10,  sma_slow=40,
        bb_length=20,
        stoch_k=14,   stoch_d=3,
        obv_slope_window=12,
        atr_length=12,
        roc_length=6,
        min_bars=40,    # SMA40 계산 기준
    ),
}

# ── 매크로 지표 정규화 설정 ───────────────────────────────────────────────────
MACRO_NORM: dict[str, NormConfig] = {
    "FEDFUNDS":           NormConfig("zscore",     invert=True,  window_years=10),
    "DGS10":              NormConfig("zscore",     invert=True,  window_years=10),
    "DGS2":               NormConfig("zscore",     invert=True,  window_years=10),
    "YIELD_CURVE_SPREAD": NormConfig("minmax",     invert=False, fixed_min=-3.0, fixed_max=4.0),
    "CPIAUCSL":           NormConfig("minmax",     invert=True,  fixed_min=0.0,  fixed_max=10.0),
    "PCEPILFE":           NormConfig("minmax",     invert=True,  fixed_min=0.0,  fixed_max=8.0),
    "UNRATE":             NormConfig("zscore",     invert=True,  window_years=10),
    "ICSA":               NormConfig("zscore",     invert=True,  window_years=10),
    "M2SL":               NormConfig("zscore",     invert=False, window_years=10),
    "CREDIT_SPREAD":      NormConfig("minmax",     invert=True,  fixed_min=0.0,  fixed_max=5.0),
}

# ── 펀더멘탈 지표 정규화 설정 ─────────────────────────────────────────────────
FUNDAMENTAL_NORM: dict[str, NormConfig] = {
    "pbr":             NormConfig("percentile", invert=True,  window_years=5),
    "eps_change_rate": NormConfig("zscore",     invert=False, window_years=5),
    "roe":             NormConfig("percentile", invert=False, window_years=5),
    "fcf_yield":       NormConfig("percentile", invert=False, window_years=5),
    "de_ratio":        NormConfig("percentile", invert=True,  window_years=5),
    "revenue_growth":  NormConfig("zscore",     invert=False, window_years=5),
    "earnings_yield":  NormConfig("percentile", invert=False, window_years=5),
}

# ── 기술적 지표 정규화 설정 (해상도 공통 — 파라미터는 TECH_PARAMS 참조) ─────
TECHNICAL_NORM: dict[str, NormConfig] = {
    "rsi_14":         NormConfig("minmax",  invert=False, fixed_min=0.0,  fixed_max=100.0),
    "macd_histogram": NormConfig("zscore",  invert=False, window_years=2),
    "sma_ratio":      NormConfig("minmax",  invert=False, fixed_min=0.85, fixed_max=1.15),
    "stoch_k":        NormConfig("minmax",  invert=False, fixed_min=0.0,  fixed_max=100.0),
    "bb_pct_b":       NormConfig("minmax",  invert=False, fixed_min=0.0,  fixed_max=1.0),
    "obv_slope":      NormConfig("zscore",  invert=False, window_years=1),
    "atr_norm":       NormConfig("zscore",  invert=True,  window_years=2),
    "roc":            NormConfig("zscore",  invert=False, window_years=2),
}

# RSI V자형 비선형 스코어링 (oversold=고점수, overbought=저점수)
RSI_NONLINEAR = True

# ── GICS 섹터 코드 매핑 ────────────────────────────────────────────────────────
GICS_SECTOR_NAMES: dict[str, str] = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# ── 섹터별 펀더멘탈 지표 가중치 ────────────────────────────────────────────────
# 근거: Barra USE4, AQR QMJ, Novy-Marx(2013), Peters&Taylor(2017), Ehsani et al.(2023)
# 키: GICS sector code (문자열)
# 값: 각 지표 가중치 (합산 = 1.0)
# Financials: D/E·FCF 제거(부채가 영업도구), PBR 핵심
# Utilities/RE: PBR·D/E 제거, earnings_yield·FCF 중시
# IT/Comm: PBR 비중 최소화(무형자산 왜곡), 성장·수익성 중시
SECTOR_FUNDAMENTAL_WEIGHTS: dict[str, dict[str, float]] = {
    "45": {  # Information Technology
        "roe": 0.20, "eps_change_rate": 0.25, "revenue_growth": 0.25,
        "fcf_yield": 0.15, "pbr": 0.10, "de_ratio": 0.05, "earnings_yield": 0.00,
    },
    "50": {  # Communication Services
        "roe": 0.20, "eps_change_rate": 0.20, "revenue_growth": 0.25,
        "fcf_yield": 0.20, "pbr": 0.10, "de_ratio": 0.05, "earnings_yield": 0.00,
    },
    "25": {  # Consumer Discretionary
        "roe": 0.20, "eps_change_rate": 0.20, "revenue_growth": 0.20,
        "fcf_yield": 0.15, "pbr": 0.10, "de_ratio": 0.10, "earnings_yield": 0.05,
    },
    "30": {  # Consumer Staples
        "roe": 0.20, "eps_change_rate": 0.15, "revenue_growth": 0.15,
        "fcf_yield": 0.20, "pbr": 0.10, "de_ratio": 0.10, "earnings_yield": 0.10,
    },
    "35": {  # Health Care
        "roe": 0.20, "eps_change_rate": 0.25, "revenue_growth": 0.20,
        "fcf_yield": 0.20, "pbr": 0.05, "de_ratio": 0.05, "earnings_yield": 0.05,
    },
    "20": {  # Industrials
        "roe": 0.25, "eps_change_rate": 0.20, "revenue_growth": 0.15,
        "fcf_yield": 0.20, "pbr": 0.10, "de_ratio": 0.10, "earnings_yield": 0.00,
    },
    "10": {  # Energy
        "roe": 0.20, "eps_change_rate": 0.10, "revenue_growth": 0.15,
        "fcf_yield": 0.25, "pbr": 0.10, "de_ratio": 0.15, "earnings_yield": 0.05,
    },
    "15": {  # Materials
        "roe": 0.20, "eps_change_rate": 0.10, "revenue_growth": 0.15,
        "fcf_yield": 0.25, "pbr": 0.10, "de_ratio": 0.15, "earnings_yield": 0.05,
    },
    "40": {  # Financials — D/E·FCF 완전 제거, P/B 핵심
        "roe": 0.35, "eps_change_rate": 0.20, "revenue_growth": 0.15,
        "fcf_yield": 0.00, "pbr": 0.30, "de_ratio": 0.00, "earnings_yield": 0.00,
    },
    "55": {  # Utilities — PBR·D/E 제거, earnings_yield·FCF 중심
        "roe": 0.20, "eps_change_rate": 0.10, "revenue_growth": 0.10,
        "fcf_yield": 0.30, "pbr": 0.00, "de_ratio": 0.00, "earnings_yield": 0.30,
    },
    "60": {  # Real Estate — PBR·D/E 제거, FCF(≈FFO proxy)·earnings_yield 중심
        "roe": 0.20, "eps_change_rate": 0.10, "revenue_growth": 0.10,
        "fcf_yield": 0.35, "pbr": 0.00, "de_ratio": 0.00, "earnings_yield": 0.25,
    },
}

# 알 수 없는 섹터 폴백
DEFAULT_FUNDAMENTAL_WEIGHTS: dict[str, float] = {
    "roe": 0.20, "eps_change_rate": 0.20, "revenue_growth": 0.20,
    "fcf_yield": 0.15, "pbr": 0.10, "de_ratio": 0.10, "earnings_yield": 0.05,
}
