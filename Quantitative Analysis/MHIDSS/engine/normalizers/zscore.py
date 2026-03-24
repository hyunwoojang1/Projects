"""Z-Score 정규화: z ∈ [-3, 3] → [0, 100]."""

from __future__ import annotations
import numpy as np
from .base import BaseNormalizer, _to_numpy


class ZScoreNormalizer(BaseNormalizer):
    def __init__(self, invert: bool = False, window_years: int = 10) -> None:
        super().__init__(invert)
        self.window_years = window_years
        self._mean: float = 0.0
        self._std: float = 1.0

    def fit(self, historical) -> "ZScoreNormalizer":
        arr = _to_numpy(historical)
        if len(arr) == 0:
            raise ValueError("fit()에 전달된 데이터가 비어 있습니다.")
        self._mean = float(arr.mean())
        self._std  = float(arr.std())
        if self._std < 1e-9:
            self._std = 1e-9
        self._fitted = True
        return self

    def _transform_value(self, value: float) -> float:
        z = (value - self._mean) / self._std
        score = (z + 3.0) / 6.0 * 100.0
        return float(np.clip(score, 0.0, 100.0))
