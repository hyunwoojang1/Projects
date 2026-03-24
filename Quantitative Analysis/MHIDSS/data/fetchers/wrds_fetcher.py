"""WRDS Compustat 데이터 fetcher (Polars 출력).

섹터 내 z-score 기반 펀더멘탈 스코어링을 위해:
- comp.funda + comp.company JOIN → gsector, tic 포함
- fetch()     : 전체 시장 데이터 (섹터 분포 기준 모집단)
- get_ticker_fundamentals() : 특정 티커의 최신 재무값
- get_sector_latest()       : 섹터 내 각 기업 최신값 (z-score 분모)
"""

from __future__ import annotations

import warnings

import pandas as pd
import polars as pl

from config.settings import WRDS_USERNAME
from config import wrds_fields as wf
from data.cache.disk_cache import DiskCache
from .base import BaseFetcher

FUND_METRICS = ["pbr", "roe", "fcf_yield", "de_ratio",
                "earnings_yield", "eps_change_rate", "revenue_growth"]


class WRDSFetcher(BaseFetcher):
    def __init__(self, cache: DiskCache | None = None) -> None:
        self._cache = cache
        self._db = None

    def _get_db(self):
        if self._db is None:
            import wrds
            self._db = wrds.Connection(wrds_username=WRDS_USERNAME)
        return self._db

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def fetch(
        self,
        identifiers: list[str],
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """전체 시장 재무 데이터 수집 (gsector, tic 포함).

        섹터 z-score 계산의 모집단으로 사용된다.
        """
        if not WRDS_USERNAME:
            return pl.DataFrame()

        cache_key = f"wrds_funda_{start_date}_{end_date}"
        if self._cache and (cached := self._cache.get(cache_key)) is not None:
            return cached

        try:
            fields = ", ".join(wf.KEY_FIELDS + wf.COMPANY_FIELDS + wf.FINANCIAL_FIELDS)
            filters = " AND ".join(f"f.{k} = '{v}'" for k, v in wf.STANDARD_FILTERS.items())
            query = f"""
                SELECT {fields}
                FROM {wf.FUNDA_TABLE} f
                LEFT JOIN {wf.COMPANY_TABLE} c ON f.gvkey = c.gvkey
                WHERE {filters}
                  AND f.datadate >= '{start_date}'
                  AND f.datadate <= '{end_date}'
                ORDER BY f.gvkey, f.datadate
            """
            db = self._get_db()
            pd_df = db.raw_sql(query, date_cols=["datadate"])
            # 컬럼명에서 테이블 접두사 제거 (f.gvkey → gvkey)
            pd_df.columns = [c.split(".")[-1] for c in pd_df.columns]
            pd_df = _compute_derived(pd_df)

            result = pl.from_pandas(pd_df)
            if self._cache:
                self._cache.set(cache_key, result)
            return result

        except Exception as e:
            warnings.warn(f"WRDS fetch 실패 — 펀더멘탈 스코어 생략: {e}")
            return pl.DataFrame()

    def get_ticker_fundamentals(
        self,
        ticker: str,
        all_data: pl.DataFrame,
        as_of_date: str,
    ) -> tuple[dict[str, float], str]:
        """티커의 최신 재무값과 GICS 섹터 코드를 반환.

        Returns:
            (raw_values, gsector_code)
            raw_values: 지표명 → float 딕셔너리
            gsector_code: GICS 섹터 코드 문자열 (예: "45")
        """
        if all_data.is_empty():
            return {}, ""

        # 티커 필터 + as_of_date 이전 데이터
        ticker_df = (
            all_data
            .filter(pl.col("tic") == ticker)
            .filter(pl.col("datadate").cast(pl.Utf8) <= as_of_date)
            .sort("datadate")
        )
        if ticker_df.is_empty():
            return {}, ""

        latest = ticker_df.tail(1)
        gsector = str(latest["gsector"][0] or "")

        raw: dict[str, float] = {}
        for metric in FUND_METRICS:
            if metric in latest.columns:
                val = latest[metric][0]
                try:
                    fval = float(val)  # type: ignore[arg-type]
                    import math
                    if not math.isnan(fval) and not math.isinf(fval):
                        raw[metric] = fval
                except (TypeError, ValueError):
                    pass

        return raw, gsector

    def get_sector_latest(
        self,
        gsector: str,
        all_data: pl.DataFrame,
        as_of_date: str,
    ) -> pl.DataFrame:
        """섹터 내 모든 기업의 point-in-time 최신값 반환 (z-score 분모).

        각 기업의 as_of_date 이전 가장 최근 회계연도 값을 선택한다.
        """
        if all_data.is_empty() or not gsector:
            return pl.DataFrame()

        sector_hist = (
            all_data
            .filter(pl.col("gsector").cast(pl.Utf8) == gsector)
            .filter(pl.col("datadate").cast(pl.Utf8) <= as_of_date)
        )
        if sector_hist.is_empty():
            return pl.DataFrame()

        # 기업별 최신 관측치 1개씩 선택
        sector_latest = (
            sector_hist
            .sort("datadate")
            .group_by("gvkey")
            .agg([pl.last(m) for m in FUND_METRICS if m in sector_hist.columns])
        )
        return sector_latest

    def validate_connection(self) -> bool:
        try:
            self._get_db().raw_sql("SELECT 1")
            return True
        except Exception:
            return False

    # ── 하위 호환성 유지 ──────────────────────────────────────────────────────
    def aggregate_market(self, df: pl.DataFrame, as_of_date: str) -> dict[str, float]:
        """(레거시) 시장 중위값 반환 — 섹터 z-score 방식으로 대체됨."""
        if df.is_empty():
            return {}
        pd_df = df.to_pandas()
        pd_df = pd_df[pd_df["datadate"].astype(str) <= as_of_date]
        latest = pd_df.sort_values("datadate").groupby("gvkey").last()
        return {c: float(latest[c].median()) for c in FUND_METRICS if c in latest.columns}


# ── 파생 지표 계산 (모듈 수준 함수) ────────────────────────────────────────────

def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"prcc_f", "ceq", "csho"}.issubset(df.columns):
        book_ps = df["ceq"] / df["csho"].replace(0, float("nan"))
        df["pbr"] = df["prcc_f"] / book_ps.replace(0, float("nan"))

    if {"ni", "ceq"}.issubset(df.columns):
        df["roe"] = df["ni"] / df["ceq"].replace(0, float("nan"))

    if {"oancf", "capx", "mkvalt"}.issubset(df.columns):
        df["fcf_yield"] = (df["oancf"] - df["capx"]) / df["mkvalt"].replace(0, float("nan"))

    if {"dltt", "dlc", "ceq"}.issubset(df.columns):
        df["de_ratio"] = (df["dltt"] + df["dlc"]) / df["ceq"].replace(0, float("nan"))

    if {"epsfx", "prcc_f"}.issubset(df.columns):
        df["earnings_yield"] = df["epsfx"] / df["prcc_f"].replace(0, float("nan"))

    if "epsfx" in df.columns:
        df = df.sort_values(["gvkey", "datadate"])
        df["eps_change_rate"] = df.groupby("gvkey")["epsfx"].pct_change(4)

    if "sale" in df.columns:
        df["revenue_growth"] = df.groupby("gvkey")["sale"].pct_change(4)

    return df
