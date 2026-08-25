"""Central data collector — yfinance, FRED, WRDS.

Provides a single DataCollector.collect() call that returns all data
needed by the analysis modules, with parquet-based disk caching.
"""

from __future__ import annotations

import math
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")
WRDS_USERNAME: str = os.environ.get("WRDS_USERNAME", "")
WRDS_PASSWORD: str = os.environ.get("WRDS_PASSWORD", "")

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache"))
CACHE_TTL_MARKET = int(os.environ.get("CACHE_TTL_HOURS_MARKET", "4"))
CACHE_TTL_FRED = int(os.environ.get("CACHE_TTL_HOURS_FRED", "24"))
CACHE_TTL_WRDS = int(os.environ.get("CACHE_TTL_HOURS_WRDS", "168"))

MARKET_TICKERS = {
    "qqq": "QQQ",
    "qld": "QLD",
    "tqqq": "TQQQ",
    "ndx": "^NDX",    # 나스닥100 지수 — QQQ 1999 이전 대체용
    "spy": "SPY",
    "vix": "^VIX",
    "tlt": "TLT",
    "gld": "GLD",
    "dxy": "DX-Y.NYB",
    "hyg": "HYG",
    "ief": "IEF",
}

SECTOR_TICKERS = {
    "XLK": "Technology",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discr.",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication",
}

FRED_SERIES = {
    "DGS10": "10Y Treasury Rate",
    "DGS2": "2Y Treasury Rate",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI",
    "M2SL": "M2 Money Supply",
}


# ── Disk cache (parquet-based, TTL-enforced) ──────────────────────────────────

class _DiskCache:
    def __init__(self, cache_dir: Path, ttl_hours: int = 24) -> None:
        self._dir = cache_dir
        self._ttl = timedelta(hours=ttl_hours)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in key)
        return self._dir / f"{safe}.parquet"

    def get(self, key: str) -> pd.DataFrame | None:
        path = self._path(key)
        if not path.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > self._ttl:
            path.unlink(missing_ok=True)
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        self._path(key).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._path(key))


# ── DataCollector ─────────────────────────────────────────────────────────────

class DataCollector:
    """Fetches and caches all market, macro, and fundamental data."""

    def __init__(self) -> None:
        self._market_cache = _DiskCache(CACHE_DIR / "market", CACHE_TTL_MARKET)
        self._fred_cache = _DiskCache(CACHE_DIR / "fred", CACHE_TTL_FRED)
        self._wrds_cache = _DiskCache(CACHE_DIR / "wrds", CACHE_TTL_WRDS)
        self._wrds_db = None

    # ── Public interface ───────────────────────────────────────────────────────

    def collect(self, lookback_years: int = 10) -> dict:
        """Returns all data needed by analysis modules.

        Keys:
            market   — daily OHLCV for MARKET_TICKERS + SECTOR_TICKERS
            fred     — macro series from FRED
            factors  — fundamental factor data from WRDS (may be empty)
        """
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

        market = self._fetch_market(start, end)
        fred = self._fetch_fred(start, end)
        factors = self._fetch_wrds_factors(start, end)

        return {"market": market, "fred": fred, "factors": factors,
                "start": start, "end": end}

    # ── Market data (yfinance) ─────────────────────────────────────────────────

    def _fetch_market(self, start: str, end: str) -> pd.DataFrame:
        cache_key = f"market_{start}_{end}"
        cached = self._market_cache.get(cache_key)
        if cached is not None:
            return cached

        all_tickers = list(MARKET_TICKERS.values()) + list(SECTOR_TICKERS.keys())
        try:
            raw = yf.download(all_tickers, start=start, end=end,
                              auto_adjust=True, progress=False)
            close = raw["Close"].copy()
            volume = raw["Volume"].copy() if "Volume" in raw else pd.DataFrame()

            # Flatten multi-index columns if necessary
            if isinstance(close.columns, pd.MultiIndex):
                close.columns = close.columns.get_level_values(0)

            close.index = pd.to_datetime(close.index)
            close = close.ffill().dropna(how="all")

            result = close.copy()
            if not volume.empty:
                if isinstance(volume.columns, pd.MultiIndex):
                    volume.columns = volume.columns.get_level_values(0)
                for col in volume.columns:
                    result[f"{col}_vol"] = volume[col].reindex(result.index).ffill()

            self._market_cache.set(cache_key, result)
            return result

        except Exception as e:
            warnings.warn(f"Market data fetch failed: {e}")
            return pd.DataFrame()

    # ── FRED macro data ────────────────────────────────────────────────────────

    def _fetch_fred(self, start: str, end: str) -> pd.DataFrame:
        cache_key = f"fred_{'_'.join(sorted(FRED_SERIES))}_{start}_{end}"
        cached = self._fred_cache.get(cache_key)
        if cached is not None:
            return cached

        if not FRED_API_KEY:
            warnings.warn("FRED_API_KEY not set — macro data skipped.")
            return pd.DataFrame()

        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            frames: dict[str, pd.Series] = {}
            for sid in FRED_SERIES:
                frames[sid] = fred.get_series(sid, observation_start=start,
                                              observation_end=end)

            # Yield curve spread
            if "DGS10" in frames and "DGS2" in frames:
                frames["SPREAD_10Y2Y"] = frames["DGS10"] - frames["DGS2"]

            df = pd.DataFrame(frames)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().ffill()

            self._fred_cache.set(cache_key, df)
            return df

        except Exception as e:
            warnings.warn(f"FRED fetch failed: {e}")
            return pd.DataFrame()

    # ── WRDS fundamental factor data ───────────────────────────────────────────

    def _fetch_wrds_factors(self, start: str, end: str) -> pd.DataFrame:
        """Fetches annual fundamentals from Compustat for factor construction.

        Returns a DataFrame with columns:
            gvkey, datadate, tic, gsector,
            pbr, gpa, roe, mkvalt, earnings_surprise
        """
        # WRDS_PASSWORD 미설정 시 대화형 프롬프트 차단 — 즉시 스킵
        if not WRDS_USERNAME or not WRDS_PASSWORD:
            return pd.DataFrame()

        cache_key = f"wrds_factors_{start}_{end}"
        cached = self._wrds_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            db = self._get_wrds_db()
            # Annual fundamentals: value (PBR) + quality (GP/A per Novy-Marx 2013)
            annual_query = """
                SELECT
                    f.gvkey, f.datadate, f.tic, f.fyear,
                    c.gsector,
                    f.prcc_f, f.csho, f.ceq,
                    f.ni, f.gp, f.at, f.mkvalt,
                    f.sale, f.cogs
                FROM comp.funda f
                LEFT JOIN comp.company c ON f.gvkey = c.gvkey
                WHERE f.indfmt  = 'INDL'
                  AND f.consol  = 'C'
                  AND f.popsrc  = 'D'
                  AND f.datafmt = 'STD'
                  AND f.datadate >= %(start)s
                  AND f.datadate <= %(end)s
                  AND f.exchg IN (11, 14)
                ORDER BY f.gvkey, f.datadate
            """
            pd_df = db.raw_sql(annual_query, params={"start": start, "end": end},
                               date_cols=["datadate"])
            pd_df.columns = [c.split(".")[-1] for c in pd_df.columns]
            pd_df = _derive_factor_metrics(pd_df)

            # Quarterly earnings surprise (naive: QoQ EPS change)
            quarterly_query = """
                SELECT gvkey, rdq, epsfxq
                FROM comp.fundq
                WHERE indfmt  = 'INDL'
                  AND consol  = 'C'
                  AND popsrc  = 'D'
                  AND datafmt = 'STD'
                  AND rdq     >= %(start)s
                  AND rdq     <= %(end)s
                ORDER BY gvkey, rdq
            """
            q_df = db.raw_sql(quarterly_query, params={"start": start, "end": end},
                              date_cols=["rdq"])
            q_df = q_df.sort_values(["gvkey", "rdq"])
            q_df["eps_lag"] = q_df.groupby("gvkey")["epsfxq"].shift(4)
            q_df["earnings_surprise"] = (
                (q_df["epsfxq"] - q_df["eps_lag"]) / q_df["eps_lag"].abs()
            ).clip(-3, 3)

            # Aggregate quarterly surprise to annual
            q_agg = (q_df.dropna(subset=["earnings_surprise"])
                        .groupby("gvkey")["earnings_surprise"]
                        .mean()
                        .reset_index())
            pd_df = pd_df.merge(q_agg, on="gvkey", how="left")

            self._wrds_cache.set(cache_key, pd_df)
            return pd_df

        except Exception as e:
            warnings.warn(f"WRDS fetch failed — factor scores skipped: {e}")
            return pd.DataFrame()

    # ── Aggregate factor returns (for regime model) ────────────────────────────

    def compute_factor_returns(self, factors: pd.DataFrame,
                               market: pd.DataFrame) -> pd.DataFrame:
        """Constructs monthly long-short factor portfolio returns.

        Returns DataFrame with columns:
            date, momentum, value, quality, size
        """
        if factors.empty or market.empty:
            return pd.DataFrame()

        try:
            monthly = market.resample("ME").last()
            monthly_ret = monthly.pct_change()

            # Momentum: 12-1 month price momentum (yfinance-based, QQQ proxy)
            spy_col = "SPY" if "SPY" in monthly_ret.columns else monthly_ret.columns[0]
            momentum_12_1 = monthly_ret[spy_col].rolling(11).sum().shift(1)

            # Factor scores from WRDS (latest available, market-level aggregate)
            latest = factors.sort_values("datadate").groupby("gvkey").last()
            value_score = -latest["pbr"].median() if "pbr" in latest else np.nan
            quality_score = latest["gpa"].median() if "gpa" in latest else np.nan
            size_score = -np.log(latest["mkvalt"].median()) if "mkvalt" in latest else np.nan

            # Build a daily time series (returns are market-level for now)
            result = pd.DataFrame(index=monthly_ret.index)
            result["momentum"] = momentum_12_1
            result["value"] = np.nan  # will be filled with WRDS cross-section signals
            result["quality"] = np.nan
            result["size"] = np.nan

            if not np.isnan(value_score):
                result["value"] = value_score
            if not np.isnan(quality_score):
                result["quality"] = quality_score
            if not np.isnan(size_score):
                result["size"] = size_score

            return result.dropna(how="all")

        except Exception as e:
            warnings.warn(f"Factor return computation failed: {e}")
            return pd.DataFrame()

    # ── WRDS lazy connection ───────────────────────────────────────────────────

    def _get_wrds_db(self):
        if self._wrds_db is None:
            import wrds
            kwargs: dict = {"wrds_username": WRDS_USERNAME}
            if WRDS_PASSWORD:
                kwargs["wrds_password"] = WRDS_PASSWORD
            self._wrds_db = wrds.Connection(**kwargs)
        return self._wrds_db


# ── Derived fundamental metrics ───────────────────────────────────────────────

def _derive_factor_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PBR = price / book value per share
    if {"prcc_f", "ceq", "csho"}.issubset(df.columns):
        bvps = df["ceq"] / df["csho"].replace(0, np.nan)
        df["pbr"] = df["prcc_f"] / bvps.replace(0, np.nan)

    # GP/A = gross profit / total assets (Novy-Marx 2013 quality factor)
    # Use gp directly if available; otherwise compute from sale - cogs
    if "gp" not in df.columns or df["gp"].isna().all():
        if {"sale", "cogs"}.issubset(df.columns):
            df["gp"] = df["sale"] - df["cogs"]
    if {"gp", "at"}.issubset(df.columns):
        df["gpa"] = df["gp"] / df["at"].replace(0, np.nan)

    # ROE = net income / common equity
    if {"ni", "ceq"}.issubset(df.columns):
        df["roe"] = df["ni"] / df["ceq"].replace(0, np.nan)

    # Clamp extreme values
    for col in ["pbr", "gpa", "roe"]:
        if col in df.columns:
            low, high = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(low, high)

    return df


# ── Market snapshot (today's values) ─────────────────────────────────────────

def get_latest_snapshot(market: pd.DataFrame) -> dict[str, float]:
    """Returns the most recent close price for each ticker."""
    if market.empty:
        return {}
    last_row = market.iloc[-1]
    return {col: float(val) for col, val in last_row.items()
            if isinstance(val, (int, float)) and not math.isnan(val)}
