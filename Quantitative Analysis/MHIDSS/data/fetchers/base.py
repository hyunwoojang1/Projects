"""데이터 fetcher 기본 인터페이스 및 DataResolution 열거형."""

from abc import ABC, abstractmethod
from enum import Enum

import polars as pl


class DataResolution(str, Enum):
    """데이터 해상도 (시계열별 봉 단위)."""
    DAILY   = "daily"    # Short-term: 일봉
    WEEKLY  = "weekly"   # Mid-term:   주봉
    MONTHLY = "monthly"  # Long-term:  월봉


class BaseFetcher(ABC):
    @abstractmethod
    def fetch(
        self,
        identifiers: list[str],
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """'date' 컬럼 + 지표별 컬럼을 가진 Polars DataFrame 반환."""
        ...

    @abstractmethod
    def validate_connection(self) -> bool:
        """데이터 소스 연결 가능 여부 확인."""
        ...
