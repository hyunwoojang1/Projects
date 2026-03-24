"""가격/기술적 지표 fetcher — 해상도(일/주/월봉)별 분리 계산 (Polars 출력).

데이터 부족 처리 전략:
- 각 해상도별 `min_bars` 기준 미달 시 해당 지표를 null 처리
- null 지표는 Scorer에서 가중치 재분배로 처리
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from config.normalization import TECH_PARAMS, TechIndicatorParams
from config.settings import PRICE_DATA_SOURCE
from data.cache.disk_cache import DiskCache
from .base import BaseFetcher, DataResolution

warnings.filterwarnings("ignore", category=FutureWarning)


class TechnicalFetcher(BaseFetcher):
    def __init__(self, cache: DiskCache | None = None) -> None:
        self._cache = cache

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def fetch(
        self,
        identifiers: list[str],
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """일봉 기준 지표 DataFrame 반환 (기본 인터페이스 충족용)."""
        ticker = identifiers[0] if identifiers else "SPY"
        return self.fetch_by_resolution(ticker, start_date, end_date, DataResolution.DAILY)

    def fetch_all_resolutions(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, pl.DataFrame]:
        """Short/Mid/Long 3개 해상도의 지표 DataFrame을 한번에 반환."""
        cache_key = f"tech_all_{ticker}_{start_date}_{end_date}"
        if self._cache and (cached := self._cache.get(cache_key)) is not None:
            return _split_resolution_cache(cached)

        daily_prices = self._download(ticker, start_date, end_date)

        result: dict[str, pl.DataFrame] = {}
        for res in DataResolution:
            prices = self._resample(daily_prices, res)
            result[res.value] = self._compute_indicators(prices, res)

        if self._cache:
            combined = _merge_resolution_cache(result)
            self._cache.set(cache_key, combined)

        return result

    def fetch_by_resolution(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        resolution: DataResolution,
    ) -> pl.DataFrame:
        all_res = self.fetch_all_resolutions(ticker, start_date, end_date)
        return all_res.get(resolution.value, pl.DataFrame())

    def validate_connection(self) -> bool:
        try:
            import yfinance as yf
            t = yf.Ticker("SPY")
            return t.history(period="1d").shape[0] > 0
        except Exception:
            return False

    # ── 내부 구현 ──────────────────────────────────────────────────────────────

    def _download(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        if PRICE_DATA_SOURCE == "yfinance":
            import yfinance as yf
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            return df.reset_index()
        raise NotImplementedError(f"지원하지 않는 PRICE_DATA_SOURCE: {PRICE_DATA_SOURCE}")

    def _resample(self, df: pd.DataFrame, resolution: DataResolution) -> pd.DataFrame:
        """일봉 OHLCV를 주봉/월봉으로 리샘플링."""
        if resolution == DataResolution.DAILY:
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        rule = "W-FRI" if resolution == DataResolution.WEEKLY else "ME"
        resampled = df.resample(rule).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna(subset=["close"])

        return resampled.reset_index()

    def _compute_indicators(
        self, df: pd.DataFrame, resolution: DataResolution
    ) -> pl.DataFrame:
        """ta 라이브러리로 지표 계산 후 Polars로 변환."""
        import ta

        params = TECH_PARAMS[resolution.value]
        n = len(df)

        result = pd.DataFrame({"date": df["date"] if "date" in df.columns else df.index})
        result["date"] = pd.to_datetime(result["date"])

        close  = pd.Series(df["close"].values,  dtype=float)
        high   = pd.Series(df["high"].values,   dtype=float)
        low    = pd.Series(df["low"].values,    dtype=float)
        volume = pd.Series(df["volume"].values, dtype=float)

        # ── RSI ──────────────────────────────────────────────────────────────
        if n >= params.rsi_length + 1:
            result["rsi_14"] = ta.momentum.RSIIndicator(
                close, window=params.rsi_length
            ).rsi().values
        else:
            result["rsi_14"] = np.nan

        # ── MACD Histogram ───────────────────────────────────────────────────
        if n >= params.macd_slow + params.macd_signal:
            result["macd_histogram"] = ta.trend.MACD(
                close,
                window_slow=params.macd_slow,
                window_fast=params.macd_fast,
                window_sign=params.macd_signal,
            ).macd_diff().values
        else:
            result["macd_histogram"] = np.nan

        # ── SMA Ratio (fast/slow) ────────────────────────────────────────────
        if n >= params.sma_slow:
            sma_fast = ta.trend.SMAIndicator(close, window=params.sma_fast).sma_indicator()
            sma_slow = ta.trend.SMAIndicator(close, window=params.sma_slow).sma_indicator()
            result["sma_ratio"] = (sma_fast / sma_slow.replace(0, np.nan)).values
        else:
            result["sma_ratio"] = np.nan

        # ── Bollinger Band %B ────────────────────────────────────────────────
        if n >= params.bb_length:
            result["bb_pct_b"] = ta.volatility.BollingerBands(
                close, window=params.bb_length, window_dev=2
            ).bollinger_pband().values
        else:
            result["bb_pct_b"] = np.nan

        # ── Stochastic %K ────────────────────────────────────────────────────
        if n >= params.stoch_k + params.stoch_d:
            result["stoch_k"] = ta.momentum.StochasticOscillator(
                high, low, close,
                window=params.stoch_k,
                smooth_window=params.stoch_d,
            ).stoch().values
        else:
            result["stoch_k"] = np.nan

        # ── OBV Slope ────────────────────────────────────────────────────────
        if n >= params.obv_slope_window + 1:
            obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            result["obv_slope"] = obv.rolling(params.obv_slope_window).apply(
                lambda x: float(np.polyfit(range(len(x)), x, 1)[0]),
                raw=True,
            ).values
        else:
            result["obv_slope"] = np.nan

        # ── ATR (정규화: ATR / Close) ────────────────────────────────────────
        if n >= params.atr_length + 1:
            atr = ta.volatility.AverageTrueRange(
                high, low, close, window=params.atr_length
            ).average_true_range()
            result["atr_norm"] = (atr / close.replace(0, np.nan)).values
        else:
            result["atr_norm"] = np.nan

        # ── ROC ──────────────────────────────────────────────────────────────
        if n >= params.roc_length + 1:
            result["roc"] = ta.momentum.ROCIndicator(
                close, window=params.roc_length
            ).roc().values
        else:
            result["roc"] = np.nan

        return pl.from_pandas(result).sort("date")


# ── 캐시 직렬화 헬퍼 ──────────────────────────────────────────────────────────

def _merge_resolution_cache(data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """3개 해상도 DataFrame을 하나의 DataFrame으로 병합 (resolution 컬럼 추가)."""
    frames = []
    for res, df in data.items():
        frames.append(df.with_columns(pl.lit(res).alias("_resolution")))
    return pl.concat(frames) if frames else pl.DataFrame()


def _split_resolution_cache(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """병합된 캐시 DataFrame을 해상도별로 분리."""
    if "_resolution" not in df.columns:
        return {"daily": df}
    result = {}
    for res in df["_resolution"].unique().to_list():
        result[res] = df.filter(pl.col("_resolution") == res).drop("_resolution")
    return result
