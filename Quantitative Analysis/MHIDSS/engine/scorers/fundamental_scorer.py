"""펀더멘탈 스코어러 — GICS 섹터 내 z-score 기반.

설계 원칙 (Barra USE4 / AQR QMJ / Ehsani et al. 2023 기반):
1. 섹터 내(within-sector) z-score: 성장주/가치주 스타일 bias 제거
2. 섹터별 지표 가중치: 금융(PBR 핵심), IT(성장·수익성 중심), 유틸(FCF·수익수율) 등
3. Point-in-time: as_of_date 이전 데이터만 사용 (look-ahead bias 차단)
4. 데이터 부족 처리: 지표 없으면 NaN → 가중치 재분배
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from config.normalization import (
    SECTOR_FUNDAMENTAL_WEIGHTS,
    DEFAULT_FUNDAMENTAL_WEIGHTS,
    GICS_SECTOR_NAMES,
    FUNDAMENTAL_NORM,
)
from .base import BaseScorer

# 섹터 z-score에 필요한 최소 기업 수
_MIN_SECTOR_SIZE = 10
# z-score 클리핑 범위 → [0, 100] 변환
_Z_CLIP = 3.0


class FundamentalScorer(BaseScorer):
    def __init__(
        self,
        sector_latest_df: pl.DataFrame,
        sector_code: str,
        historical_df: pl.DataFrame | None = None,  # 하위 호환성
    ) -> None:
        """
        Args:
            sector_latest_df: 섹터 내 모든 기업의 point-in-time 최신값
                              (WRDSFetcher.get_sector_latest() 반환값)
            sector_code:      GICS 섹터 코드 (예: "45")
            historical_df:    레거시 파라미터 — 무시됨
        """
        self._sector_df = sector_latest_df
        self._sector_code = sector_code
        self._weights = SECTOR_FUNDAMENTAL_WEIGHTS.get(sector_code, DEFAULT_FUNDAMENTAL_WEIGHTS)
        self._sector_name = GICS_SECTOR_NAMES.get(sector_code, f"Unknown({sector_code})")

        # 섹터별 각 지표의 분포 캐시 {metric: (mean, std)}
        self._dist_cache: dict[str, tuple[float, float]] = {}

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def score(self, raw_values: dict[str, float], as_of_date: str = "") -> dict[str, float]:
        """티커의 원시 재무값 → 섹터 내 z-score 기반 [0,100] 점수 딕셔너리."""
        scores: dict[str, float] = {}

        for metric, weight in self._weights.items():
            if weight == 0.0:
                continue  # 이 섹터에서 사용하지 않는 지표

            value = raw_values.get(metric)
            if value is None or not math.isfinite(value):
                scores[metric] = float("nan")
                continue

            try:
                scores[metric] = self._zscore_to_100(metric, value)
            except Exception:
                scores[metric] = float("nan")

        return scores

    def composite_score(self, raw_values: dict[str, float], as_of_date: str = "") -> float:
        """섹터 가중치 적용 후 단일 복합 점수 [0,100] 반환."""
        indicator_scores = self.score(raw_values, as_of_date)
        return _weighted_mean(indicator_scores, self._weights)

    @property
    def sector_name(self) -> str:
        return self._sector_name

    @property
    def sector_size(self) -> int:
        """섹터 내 기업 수 (z-score 모집단 크기)."""
        return len(self._sector_df) if not self._sector_df.is_empty() else 0

    # ── 내부 구현 ──────────────────────────────────────────────────────────────

    def _get_distribution(self, metric: str) -> tuple[float, float] | None:
        """섹터 내 지표 분포 (mean, std) — 캐시 적용."""
        if metric in self._dist_cache:
            return self._dist_cache[metric]

        if self._sector_df.is_empty() or metric not in self._sector_df.columns:
            return None

        if self.sector_size < _MIN_SECTOR_SIZE:
            return None  # 섹터 표본 부족

        vals = self._sector_df[metric].drop_nulls()
        # 극단값 윈저화 (1%~99% 범위로 클리핑) → 이상치가 분포를 왜곡하는 것 방지
        if len(vals) < _MIN_SECTOR_SIZE:
            return None

        arr = vals.to_numpy().astype(float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < _MIN_SECTOR_SIZE:
            return None

        p1, p99 = np.percentile(arr, [1, 99])
        arr_w = np.clip(arr, p1, p99)

        mean = float(arr_w.mean())
        std  = float(arr_w.std())
        if std < 1e-9:
            return None  # 분산 없음 → 의미 없는 z-score

        self._dist_cache[metric] = (mean, std)
        return mean, std

    def _zscore_to_100(self, metric: str, value: float) -> float:
        """값을 섹터 내 z-score → [0,100]으로 변환."""
        dist = self._get_distribution(metric)
        if dist is None:
            return float("nan")

        mean, std = dist
        z = (value - mean) / std
        z_clipped = float(np.clip(z, -_Z_CLIP, _Z_CLIP))
        score = (z_clipped + _Z_CLIP) / (2 * _Z_CLIP) * 100.0

        # 방향 반전 (높을수록 나쁜 지표: pbr, de_ratio)
        norm_cfg = FUNDAMENTAL_NORM.get(metric)
        if norm_cfg and norm_cfg.invert:
            score = 100.0 - score

        return float(score)


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _weighted_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    """NaN 지표를 건너뛰고 나머지 가중치를 비례 재분배하여 평균 계산."""
    total_w = 0.0
    total_s = 0.0
    for metric, w in weights.items():
        if w == 0.0:
            continue
        v = scores.get(metric)
        if v is None or not math.isfinite(v):
            continue
        total_s += v * w
        total_w += w
    return total_s / total_w if total_w > 0.0 else float("nan")
