"""수학 유틸리티."""

from __future__ import annotations

import numpy as np


def clip_score(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, value)))


def rolling_linear_slope(values: np.ndarray, window: int) -> np.ndarray:
    """롤링 윈도우 내 선형 기울기. pandas-ta 내부 계산을 위해 numpy 배열로 처리."""
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = values[i - window + 1 : i + 1]
        if len(chunk) >= 2:
            result[i] = float(np.polyfit(range(len(chunk)), chunk, 1)[0])
    return result


def redistribute_weights(
    weights: dict[str, float],
    missing_keys: list[str],
) -> dict[str, float]:
    """누락된 지표의 가중치를 나머지에 비례 재분배."""
    available = {k: v for k, v in weights.items() if k not in missing_keys}
    total = sum(available.values())
    if total == 0:
        return available
    return {k: v / total for k, v in available.items()}
