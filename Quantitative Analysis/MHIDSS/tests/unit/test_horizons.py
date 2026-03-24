"""시계열 호라이즌 단위 테스트."""

import pytest

from engine.horizons.short_term import ShortTermHorizon
from engine.horizons.mid_term import MidTermHorizon
from engine.horizons.long_term import LongTermHorizon

VALID_SIGNALS = {"STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"}


def _all_scores(val: float) -> tuple[dict, dict, dict]:
    macro = {k: val for k in ["FEDFUNDS", "DGS10", "DGS2", "YIELD_CURVE_SPREAD",
                               "CPIAUCSL", "PCEPILFE", "UNRATE", "ICSA", "M2SL", "CREDIT_SPREAD"]}
    fund = {k: val for k in ["pbr", "eps_change_rate", "roe", "fcf_yield",
                              "de_ratio", "revenue_growth", "earnings_yield"]}
    tech = {k: val for k in ["rsi_14", "macd_histogram", "sma_ratio", "stoch_k",
                              "bb_pct_b", "obv_slope", "atr_norm", "roc"]}
    return macro, fund, tech


class TestHorizons:
    @pytest.mark.parametrize("HCls", [ShortTermHorizon, MidTermHorizon, LongTermHorizon])
    def test_entry_score_in_range(self, HCls, sample_macro_scores, sample_fundamental_scores, sample_technical_scores):
        result = HCls().compute(sample_macro_scores, sample_fundamental_scores, sample_technical_scores)
        assert 0.0 <= result.entry_score <= 100.0

    @pytest.mark.parametrize("HCls", [ShortTermHorizon, MidTermHorizon, LongTermHorizon])
    def test_signal_is_valid(self, HCls, sample_macro_scores, sample_fundamental_scores, sample_technical_scores):
        result = HCls().compute(sample_macro_scores, sample_fundamental_scores, sample_technical_scores)
        assert result.signal in VALID_SIGNALS

    def test_all_100_produces_strong_buy(self):
        macro, fund, tech = _all_scores(100.0)
        result = ShortTermHorizon().compute(macro, fund, tech)
        assert result.signal in {"STRONG_BUY", "BUY"}
        assert result.entry_score >= 55.0

    def test_all_0_produces_strong_sell(self):
        macro, fund, tech = _all_scores(0.0)
        result = ShortTermHorizon().compute(macro, fund, tech)
        assert result.signal in {"STRONG_SELL", "SELL"}
        assert result.entry_score <= 45.0

    def test_short_term_dominated_by_technical(self, sample_macro_scores, sample_fundamental_scores):
        low_tech = {k: 10.0 for k in ["rsi_14", "macd_histogram", "sma_ratio", "stoch_k",
                                       "bb_pct_b", "obv_slope", "atr_norm", "roc"]}
        high_tech = {k: 90.0 for k in low_tech}
        r_low = ShortTermHorizon().compute(sample_macro_scores, sample_fundamental_scores, low_tech)
        r_high = ShortTermHorizon().compute(sample_macro_scores, sample_fundamental_scores, high_tech)
        assert r_high.entry_score > r_low.entry_score

    def test_long_term_dominated_by_macro(self, sample_fundamental_scores, sample_technical_scores):
        low_macro = {k: 10.0 for k in ["FEDFUNDS", "DGS10", "DGS2", "YIELD_CURVE_SPREAD",
                                        "CPIAUCSL", "PCEPILFE", "UNRATE", "ICSA", "M2SL", "CREDIT_SPREAD"]}
        high_macro = {k: 90.0 for k in low_macro}
        r_low = LongTermHorizon().compute(low_macro, sample_fundamental_scores, sample_technical_scores)
        r_high = LongTermHorizon().compute(high_macro, sample_fundamental_scores, sample_technical_scores)
        assert r_high.entry_score > r_low.entry_score

    def test_group_scores_present(self, sample_macro_scores, sample_fundamental_scores, sample_technical_scores):
        result = MidTermHorizon().compute(sample_macro_scores, sample_fundamental_scores, sample_technical_scores)
        assert {"macro", "fundamental", "technical"} == set(result.group_scores.keys())
