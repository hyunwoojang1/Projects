"""EntryScoreEngine — 전체 파이프라인 오케스트레이터."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl

from config import fred_series as fs
from config.settings import (
    CACHE_DIR,
    CACHE_TTL_HOURS_FRED,
    CACHE_TTL_HOURS_TECHNICAL,
    CACHE_TTL_HOURS_WRDS,
)
from data.cache.disk_cache import DiskCache
from data.fetchers.fred_fetcher import FREDFetcher
from data.fetchers.technical_fetcher import TechnicalFetcher
from data.fetchers.wrds_fetcher import WRDSFetcher
from engine.horizons.base import HorizonResult
from engine.horizons.long_term import LongTermHorizon
from engine.horizons.mid_term import MidTermHorizon
from engine.horizons.short_term import ShortTermHorizon
from engine.scorers.fundamental_scorer import FundamentalScorer
from engine.scorers.macro_scorer import MacroScorer
from engine.scorers.technical_scorer import TechnicalScorer


class EntryScoreEngine:
    LOOKBACK_MACRO_YEARS = 20
    LOOKBACK_FUND_YEARS  = 5
    LOOKBACK_TECH_DAYS   = 1500   # 월봉 40개 + 여유

    def __init__(self) -> None:
        fred_cache = DiskCache(CACHE_DIR / "fred", ttl_hours=CACHE_TTL_HOURS_FRED)
        wrds_cache = DiskCache(CACHE_DIR / "wrds", ttl_hours=CACHE_TTL_HOURS_WRDS)
        tech_cache = DiskCache(CACHE_DIR / "tech", ttl_hours=CACHE_TTL_HOURS_TECHNICAL)

        self._fred_fetcher = FREDFetcher(cache=fred_cache)
        self._wrds_fetcher = WRDSFetcher(cache=wrds_cache)
        self._tech_fetcher = TechnicalFetcher(cache=tech_cache)

        self._short_horizon = ShortTermHorizon()
        self._mid_horizon   = MidTermHorizon()
        self._long_horizon  = LongTermHorizon()

    def run(
        self,
        ticker: str,
        as_of_date: str | None = None,
    ) -> dict[str, HorizonResult]:
        as_of = as_of_date or date.today().isoformat()

        macro_start = _years_before(as_of, self.LOOKBACK_MACRO_YEARS)
        fund_start  = _years_before(as_of, self.LOOKBACK_FUND_YEARS)
        tech_start  = _days_before(as_of, self.LOOKBACK_TECH_DAYS)

        # ── 1. 데이터 수집 ────────────────────────────────────────────────────
        macro_df = self._fred_fetcher.fetch(
            fs.FETCH_SERIES + list(fs.DERIVED_SERIES.keys()),
            start_date=macro_start,
            end_date=as_of,
        )
        macro_df = self._fred_fetcher.compute_yoy(
            macro_df, [fs.CPIAUCSL, fs.PCEPILFE, fs.M2SL]
        )

        # 전체 시장 재무 데이터 (gsector, tic 포함) — 섹터 z-score 모집단
        all_fund_df = self._wrds_fetcher.fetch(
            [], start_date=fund_start, end_date=as_of
        )

        tech_dfs: dict[str, pl.DataFrame] = self._tech_fetcher.fetch_all_resolutions(
            ticker, start_date=tech_start, end_date=as_of
        )

        # ── 2. 원시값 추출 ────────────────────────────────────────────────────
        macro_raw = _latest_values(macro_df, as_of)

        # 티커의 재무값 + GICS 섹터 코드
        ticker_fund_raw, sector_code = self._wrds_fetcher.get_ticker_fundamentals(
            ticker, all_fund_df, as_of
        )

        # 섹터 내 최신값 분포 (z-score 분모)
        sector_latest_df = self._wrds_fetcher.get_sector_latest(
            sector_code, all_fund_df, as_of
        )

        daily_raw   = _latest_values(tech_dfs.get("daily",   pl.DataFrame()), as_of)
        weekly_raw  = _latest_values(tech_dfs.get("weekly",  pl.DataFrame()), as_of)
        monthly_raw = _latest_values(tech_dfs.get("monthly", pl.DataFrame()), as_of)

        # ── 3. 스코어링 ──────────────────────────────────────────────────────
        macro_scorer = MacroScorer(historical_df=macro_df)

        # 섹터 내 z-score 기반 FundamentalScorer
        fund_scorer = FundamentalScorer(
            sector_latest_df=sector_latest_df,
            sector_code=sector_code,
        )

        daily_tech_scorer   = TechnicalScorer(historical_df=tech_dfs.get("daily",   pl.DataFrame()))
        weekly_tech_scorer  = TechnicalScorer(historical_df=tech_dfs.get("weekly",  pl.DataFrame()))
        monthly_tech_scorer = TechnicalScorer(historical_df=tech_dfs.get("monthly", pl.DataFrame()))

        macro_scores = macro_scorer.score(macro_raw, as_of)
        fund_scores  = fund_scorer.score(ticker_fund_raw, as_of)

        short_tech_scores   = daily_tech_scorer.score(daily_raw,   as_of)
        mid_tech_scores     = weekly_tech_scorer.score(weekly_raw,  as_of)
        long_tech_scores    = monthly_tech_scorer.score(monthly_raw, as_of)

        # ── 4. 시계열별 Entry Score ───────────────────────────────────────────
        short_result = self._short_horizon.compute(macro_scores, fund_scores, short_tech_scores, as_of)
        mid_result   = self._mid_horizon.compute(  macro_scores, fund_scores, mid_tech_scores,   as_of)
        long_result  = self._long_horizon.compute( macro_scores, fund_scores, long_tech_scores,  as_of)

        # 섹터 정보를 결과에 추가
        sector_info = f"{sector_code}:{fund_scorer.sector_name}(n={fund_scorer.sector_size})"
        for result in (short_result, mid_result, long_result):
            result.indicator_scores["_sector"] = sector_info  # type: ignore[assignment]

        return {
            "short": short_result,
            "mid":   mid_result,
            "long":  long_result,
        }


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def _years_before(reference: str, years: int) -> str:
    ref = date.fromisoformat(reference)
    return date(ref.year - years, ref.month, ref.day).isoformat()


def _days_before(reference: str, days: int) -> str:
    return (date.fromisoformat(reference) - timedelta(days=days)).isoformat()


def _latest_values(df: pl.DataFrame, as_of: str) -> dict[str, float]:
    """as_of 이전 가장 최근 행에서 컬럼별 float 값을 추출."""
    if df.is_empty() or "date" not in df.columns:
        return {}
    subset = df.filter(pl.col("date").cast(pl.Utf8) <= as_of)
    if subset.is_empty():
        return {}
    latest = subset.tail(1)
    result: dict[str, float] = {}
    for col in df.columns:
        if col == "date":
            continue
        val = latest[col][0]
        try:
            fval = float(val)  # type: ignore[arg-type]
            if math.isfinite(fval):
                result[col] = fval
        except (TypeError, ValueError):
            pass
    return result


def _dict_to_pl_df(values: dict[str, float], as_of: str) -> pl.DataFrame:
    """단일 스냅샷 딕셔너리를 Polars DataFrame으로 변환."""
    row: dict[str, object] = {"date": as_of}
    row.update(values)
    return pl.DataFrame([row])
