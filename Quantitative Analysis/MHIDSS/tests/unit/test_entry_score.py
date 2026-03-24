"""EntryScoreEngine 통합 단위 테스트 (mock 데이터 사용)."""

import pytest
from engine.horizons.base import classify_signal


class TestClassifySignal:
    def test_strong_buy(self):
        assert classify_signal(75.0) == "STRONG_BUY"

    def test_buy(self):
        assert classify_signal(60.0) == "BUY"

    def test_neutral(self):
        assert classify_signal(50.0) == "NEUTRAL"

    def test_sell(self):
        assert classify_signal(35.0) == "SELL"

    def test_strong_sell(self):
        assert classify_signal(20.0) == "STRONG_SELL"

    def test_boundary_strong_buy(self):
        assert classify_signal(70.0) == "STRONG_BUY"

    def test_boundary_buy(self):
        assert classify_signal(55.0) == "BUY"

    def test_boundary_neutral(self):
        assert classify_signal(45.0) == "NEUTRAL"

    def test_boundary_sell(self):
        assert classify_signal(30.0) == "SELL"
