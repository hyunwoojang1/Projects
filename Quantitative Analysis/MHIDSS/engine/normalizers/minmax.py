"""MinMax 정규화: [min, max] → [0, 100]."""

from __future__ import annotations
import numpy as np
from .base import BaseNormalizer, _to_numpy


class MinMaxNormalizer(BaseNormalizer):
    def __init__(
        self,
        invert: bool = False,
        fixed_min: float | None = None,
        fixed_max: float | None = None,
    ) -> None:
        super().__init__(invert)
        self._fixed_min = fixed_min
        self._fixed_max = fixed_max
        self._min: float = 0.0
        self._max: float = 1.0

    def fit(self, historical) -> "MinMaxNormalizer":
        arr = _to_numpy(historical)
        if len(arr) == 0:
            raise ValueError("fit()에 전달된 데이터가 비어 있습니다.")
        self._min = self._fixed_min if self._fixed_min is not None else float(arr.min())
        self._max = self._fixed_max if self._fixed_max is not None else float(arr.max())
        if self._min >= self._max:
            self._max = self._min + 1e-9
        self._fitted = True
        return self

    def _transform_value(self, value: float) -> float:
        score = (value - self._min) / (self._max - self._min) * 100.0
        return float(np.clip(score, 0.0, 100.0))
