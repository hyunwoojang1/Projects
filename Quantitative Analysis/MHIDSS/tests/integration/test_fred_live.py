"""FRED API 실제 연결 테스트 (FRED_API_KEY 환경변수 필요)."""

import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"),
    reason="FRED_API_KEY 환경변수 미설정 — 실제 연결 테스트 건너뜀",
)


def test_fred_connection():
    from data.fetchers.fred_fetcher import FREDFetcher
    fetcher = FREDFetcher()
    assert fetcher.validate_connection() is True


def test_fred_fetch_fedfunds():
    from data.fetchers.fred_fetcher import FREDFetcher
    fetcher = FREDFetcher()
    df = fetcher.fetch(["FEDFUNDS"], start_date="2020-01-01", end_date="2021-01-01")
    assert "FEDFUNDS" in df.columns
    assert len(df) > 0
    assert df["FEDFUNDS"].notna().any()


def test_fred_derived_yield_curve():
    from data.fetchers.fred_fetcher import FREDFetcher
    from config.fred_series import YIELD_CURVE_SPREAD
    fetcher = FREDFetcher()
    df = fetcher.fetch([YIELD_CURVE_SPREAD], start_date="2020-01-01", end_date="2021-01-01")
    assert YIELD_CURVE_SPREAD in df.columns
