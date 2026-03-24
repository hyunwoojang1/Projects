"""스코어러 단위 테스트."""

import numpy as np
import polars as pl
import pytest

from engine.scorers.macro_scorer import MacroScorer
from engine.scorers.technical_scorer import TechnicalScorer, _rsi_nonlinear


class TestRSINonlinear:
    def test_oversold(self):
        assert _rsi_nonlinear(30.0) == pytest.approx(100.0)

    def test_overbought(self):
        assert _rsi_nonlinear(70.0) == pytest.approx(0.0)

    def test_neutral(self):
        assert _rsi_nonlinear(50.0) == pytest.approx(50.0)

    def test_extreme_oversold(self):
        assert _rsi_nonlinear(10.0) == pytest.approx(100.0)

    def test_extreme_overbought(self):
        assert _rsi_nonlinear(90.0) == pytest.approx(0.0)


class TestMacroScorer:
    def _make_historical(self) -> pl.DataFrame:
        rng = np.random.default_rng(42)
        n = 240
        dates = pl.date_range(
            pl.date(2005, 1, 1), pl.date(2024, 12, 31), interval="1mo", eager=True
        ).head(n)
        return pl.DataFrame(
            {
                "date": dates,
                "FEDFUNDS": rng.uniform(0, 8, n).tolist(),
                "UNRATE": rng.uniform(3, 12, n).tolist(),
                "CPIAUCSL": rng.uniform(-1, 9, n).tolist(),
                "PCEPILFE": rng.uniform(0, 7, n).tolist(),
                "M2SL": rng.uniform(-5, 15, n).tolist(),
                "CREDIT_SPREAD": rng.uniform(0.5, 4, n).tolist(),
                "YIELD_CURVE_SPREAD": rng.uniform(-2, 3, n).tolist(),
                "ICSA": rng.uniform(200000, 800000, n).tolist(),
                "DGS10": rng.uniform(0.5, 5, n).tolist(),
                "DGS2": rng.uniform(0.1, 4.5, n).tolist(),
            }
        )

    def test_scores_in_range(self):
        historical = self._make_historical()
        scorer = MacroScorer(historical_df=historical)
        indicator_cols = [c for c in historical.columns if c != "date"]
        last_row = historical.tail(1)
        raw = {col: float(last_row[col][0]) for col in indicator_cols}
        scores = scorer.score(raw, as_of_date="2025-01-01")
        for k, v in scores.items():
            assert 0.0 <= v <= 100.0, f"{k} 점수 범위 초과: {v}"
