"""Percentile Rank 정규화: 히스토리 내 상대 순위 → [0, 100]."""

from __future__ import annotations
import numpy as np
from .base import BaseNormalizer, _to_numpy


class PercentileRankNormalizer(BaseNormalizer):
    def __init__(self, invert: bool = False) -> None:
        super().__init__(invert)
        self._sorted: np.ndarray = np.array([])

    def fit(self, historical) -> "PercentileRankNormalizer":
        arr = _to_numpy(historical)
        if len(arr) == 0:
            raise ValueError("fit()에 전달된 데이터가 비어 있습니다.")
        self._sorted = np.sort(arr)
        self._fitted = True
        return self

    def _transform_value(self, value: float) -> float:
        n = len(self._sorted)
        rank = int(np.searchsorted(self._sorted, value, side="right"))
        return float(rank / n * 100.0)
