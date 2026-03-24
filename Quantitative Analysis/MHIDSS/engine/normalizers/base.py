"""정규화기 기본 인터페이스 — numpy/Polars 양쪽 지원."""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import polars as pl


def _to_numpy(data: "pl.Series | np.ndarray | list") -> np.ndarray:
    """다양한 입력을 NaN 제거된 float numpy 배열로 변환."""
    if isinstance(data, pl.Series):
        arr = data.drop_nulls().cast(pl.Float64).to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data[~np.isnan(data.astype(float))]
    else:
        arr = np.array(data, dtype=float)
        arr = arr[~np.isnan(arr)]
    return arr.astype(float)


class BaseNormalizer(ABC):
    """fit()은 as_of_date 이전 과거 데이터만, transform()은 임의 값에 적용."""

    def __init__(self, invert: bool = False) -> None:
        self.invert = invert
        self._fitted = False

    @abstractmethod
    def fit(self, historical: "pl.Series | np.ndarray | list") -> "BaseNormalizer":
        """과거 데이터로 파라미터 학습. as_of_date 이전 데이터만 전달할 것."""
        ...

    @abstractmethod
    def _transform_value(self, value: float) -> float:
        """단일 float 값을 [0, 100]으로 변환 (반전 전)."""
        ...

    def transform(self, value: float) -> float:
        if not self._fitted:
            raise RuntimeError("transform() 호출 전 fit()을 먼저 실행하세요.")
        if np.isnan(value):
            return float("nan")
        score = self._transform_value(float(value))
        return 100.0 - score if self.invert else score

    def transform_series(self, series: "pl.Series | np.ndarray") -> np.ndarray:
        """시리즈 전체를 변환해 numpy 배열로 반환."""
        if isinstance(series, pl.Series):
            arr = series.cast(pl.Float64).to_numpy()
        else:
            arr = np.asarray(series, dtype=float)
        return np.array([self.transform(v) for v in arr])

    def fit_transform(self, historical: "pl.Series | np.ndarray") -> np.ndarray:
        return self.fit(historical).transform_series(historical)
