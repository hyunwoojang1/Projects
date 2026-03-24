"""전체 파이프라인 통합 테스트 (mock 데이터)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from engine.horizons.base import HorizonResult


def _mock_macro_df():
    idx = pd.date_range("2005-01-01", periods=240, freq="ME")
    rng = np.random.default_rng(42)
    cols = ["FEDFUNDS", "DGS10", "DGS2", "YIELD_CURVE_SPREAD",
            "CPIAUCSL", "PCEPILFE", "UNRATE", "ICSA", "M2SL", "CREDIT_SPREAD"]
    return pd.DataFrame({c: rng.uniform(0, 10, 240) for c in cols}, index=idx)


def _mock_tech_df():
    idx = pd.date_range("2023-01-01", periods=504, freq="B")
    rng = np.random.default_rng(0)
    cols = ["rsi_14", "macd_histogram", "sma_ratio", "stoch_k",
            "bb_pct_b", "obv_slope", "atr_norm", "roc"]
    return pd.DataFrame({c: rng.uniform(20, 80, 504) for c in cols}, index=idx)


def test_horizons_return_valid_results(sample_macro_scores, sample_fundamental_scores, sample_technical_scores):
    from engine.horizons.short_term import ShortTermHorizon
    from engine.horizons.mid_term import MidTermHorizon
    from engine.horizons.long_term import LongTermHorizon

    for HCls in [ShortTermHorizon, MidTermHorizon, LongTermHorizon]:
        result = HCls().compute(sample_macro_scores, sample_fundamental_scores, sample_technical_scores)
        assert isinstance(result, HorizonResult)
        assert 0 <= result.entry_score <= 100
        assert result.signal in {"STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"}
