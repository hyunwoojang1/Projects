"""FRED API 데이터 fetcher (Polars 출력)."""

import polars as pl
import pandas as pd

from config import fred_series as fs
from config.settings import FRED_API_KEY
from data.cache.disk_cache import DiskCache
from .base import BaseFetcher


class FREDFetcher(BaseFetcher):
    def __init__(self, cache: DiskCache | None = None) -> None:
        self._cache = cache
        self._fred = None

    def _get_client(self):
        if self._fred is None:
            from fredapi import Fred
            self._fred = Fred(api_key=FRED_API_KEY)
        return self._fred

    def fetch(
        self,
        identifiers: list[str],
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        cache_key = f"fred_{'_'.join(sorted(identifiers))}_{start_date}_{end_date}"
        if self._cache and (cached := self._cache.get(cache_key)) is not None:
            return cached

        fred = self._get_client()
        raw_ids = [i for i in identifiers if i not in fs.DERIVED_SERIES]
        needed = set(raw_ids)
        for did in identifiers:
            if did in fs.DERIVED_SERIES:
                a, b, _ = fs.DERIVED_SERIES[did]
                needed.update([a, b])

        frames: dict[str, pd.Series] = {}
        for sid in needed:
            frames[sid] = fred.get_series(
                sid, observation_start=start_date, observation_end=end_date
            )

        # 파생 지표 계산
        for did in identifiers:
            if did in fs.DERIVED_SERIES:
                a, b, op = fs.DERIVED_SERIES[did]
                if op == "subtract":
                    frames[did] = frames[a] - frames[b]

        # pandas → Polars 변환
        cols = {sid: frames[sid] for sid in identifiers if sid in frames}
        pd_df = pd.DataFrame(cols)
        pd_df.index.name = "date"
        pd_df = pd_df.reset_index()
        pd_df["date"] = pd.to_datetime(pd_df["date"])

        result = pl.from_pandas(pd_df).sort("date")
        if self._cache:
            self._cache.set(cache_key, result)
        return result

    def compute_yoy(self, df: pl.DataFrame, series_ids: list[str]) -> pl.DataFrame:
        """YoY 변화율 계산 (월간 데이터 기준 12개월 전 대비 %)."""
        result = df.clone()
        for sid in series_ids:
            if sid in result.columns:
                col = result[sid]
                lagged = col.shift(12)
                yoy = (col - lagged) / lagged.abs() * 100
                result = result.with_columns(yoy.alias(sid))
        return result

    def validate_connection(self) -> bool:
        try:
            self._get_client().get_series("FEDFUNDS", observation_start="2020-01-01")
            return True
        except Exception:
            return False
