"""공유 pytest 픽스처."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_series_100():
    """0~100 범위의 100개 샘플 시리즈."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2015-01-01", periods=100, freq="ME")
    return pd.Series(rng.uniform(0, 100, 100), index=idx)


@pytest.fixture
def sample_rate_series():
    """금리 시뮬레이션: 0~10% 범위."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2005-01-01", periods=240, freq="ME")
    return pd.Series(rng.uniform(0, 10, 240), index=idx)


@pytest.fixture
def sample_macro_scores():
    return {
        "FEDFUNDS": 40.0,
        "DGS10": 35.0,
        "DGS2": 38.0,
        "YIELD_CURVE_SPREAD": 55.0,
        "CPIAUCSL": 45.0,
        "PCEPILFE": 50.0,
        "UNRATE": 60.0,
        "ICSA": 55.0,
        "M2SL": 65.0,
        "CREDIT_SPREAD": 40.0,
    }


@pytest.fixture
def sample_fundamental_scores():
    return {
        "pbr": 50.0,
        "eps_change_rate": 60.0,
        "roe": 65.0,
        "fcf_yield": 70.0,
        "de_ratio": 55.0,
        "revenue_growth": 60.0,
        "earnings_yield": 65.0,
    }


@pytest.fixture
def sample_technical_scores():
    return {
        "rsi_14": 45.0,
        "macd_histogram": 60.0,
        "sma_ratio": 55.0,
        "stoch_k": 40.0,
        "bb_pct_b": 50.0,
        "obv_slope": 60.0,
        "atr_norm": 45.0,
        "roc": 55.0,
    }
