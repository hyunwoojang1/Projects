"""정규화기 단위 테스트."""

import numpy as np
import pandas as pd
import pytest

from engine.normalizers.minmax import MinMaxNormalizer
from engine.normalizers.zscore import ZScoreNormalizer
from engine.normalizers.percentile import PercentileRankNormalizer


class TestMinMaxNormalizer:
    def test_basic_range(self):
        norm = MinMaxNormalizer()
        data = pd.Series([0.0, 50.0, 100.0])
        norm.fit(data)
        assert norm.transform(0.0) == pytest.approx(0.0)
        assert norm.transform(100.0) == pytest.approx(100.0)
        assert norm.transform(50.0) == pytest.approx(50.0)

    def test_clip_out_of_range(self):
        norm = MinMaxNormalizer(fixed_min=0.0, fixed_max=10.0)
        norm.fit(pd.Series([2.0, 8.0]))
        assert norm.transform(20.0) == pytest.approx(100.0)
        assert norm.transform(-5.0) == pytest.approx(0.0)

    def test_invert(self):
        norm = MinMaxNormalizer(invert=True, fixed_min=0.0, fixed_max=100.0)
        norm.fit(pd.Series([0.0, 100.0]))
        assert norm.transform(0.0) == pytest.approx(100.0)
        assert norm.transform(100.0) == pytest.approx(0.0)

    def test_empty_series_raises(self):
        norm = MinMaxNormalizer()
        with pytest.raises(ValueError):
            norm.fit(pd.Series([], dtype=float))

    def test_transform_before_fit_raises(self):
        norm = MinMaxNormalizer()
        with pytest.raises(RuntimeError):
            norm.transform(50.0)

    def test_nan_passthrough(self):
        norm = MinMaxNormalizer()
        norm.fit(pd.Series([0.0, 100.0]))
        assert np.isnan(norm.transform(float("nan")))


class TestZScoreNormalizer:
    def test_mean_maps_to_50(self):
        data = pd.Series(range(100), dtype=float)
        norm = ZScoreNormalizer()
        norm.fit(data)
        score = norm.transform(float(data.mean()))
        assert 40 < score < 60  # z=0 → 50

    def test_extreme_values_clipped(self):
        data = pd.Series(range(100), dtype=float)
        norm = ZScoreNormalizer()
        norm.fit(data)
        assert norm.transform(1000.0) == pytest.approx(100.0)
        assert norm.transform(-1000.0) == pytest.approx(0.0)

    def test_invert(self):
        data = pd.Series(range(100), dtype=float)
        norm_normal = ZScoreNormalizer(invert=False)
        norm_invert = ZScoreNormalizer(invert=True)
        norm_normal.fit(data)
        norm_invert.fit(data)
        s1 = norm_normal.transform(float(data.mean()))
        s2 = norm_invert.transform(float(data.mean()))
        assert s1 + s2 == pytest.approx(100.0, abs=1.0)


class TestPercentileRankNormalizer:
    def test_median_near_50(self):
        data = pd.Series(range(101), dtype=float)
        norm = PercentileRankNormalizer()
        norm.fit(data)
        score = norm.transform(50.0)
        assert 45 < score < 55

    def test_min_near_0(self):
        data = pd.Series(range(1, 101), dtype=float)
        norm = PercentileRankNormalizer()
        norm.fit(data)
        assert norm.transform(0.5) == pytest.approx(0.0)

    def test_max_near_100(self):
        data = pd.Series(range(1, 101), dtype=float)
        norm = PercentileRankNormalizer()
        norm.fit(data)
        assert norm.transform(200.0) == pytest.approx(100.0)
