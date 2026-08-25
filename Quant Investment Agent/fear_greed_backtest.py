"""공포탐욕지수 백테스트 — 자체 모델 vs CNN 실제값 비교.

10년 (2015~2024) 주 단위로 자체 점수를 계산하고,
CNN Fear & Greed 공개 API의 실제값과 MAE / 상관계수로 비교.

Look-ahead 방지: 각 날짜 t에서 market.iloc[:t] 슬라이스만 사용.

실행:
  python fear_greed_backtest.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent


# ── CNN 공개 API ──────────────────────────────────────────────────────────────

def fetch_cnn_history(start: str = "2015-01-01") -> pd.Series:
    """CNN Fear & Greed 공개 API에서 일별 점수 받아오기.
    반환: DatetimeIndex → score(float) Series
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://edition.cnn.com/markets/fear-and-greed",
        "Origin": "https://edition.cnn.com",
    }
    print(f"CNN API 요청 중: {url}")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    records = data.get("fear_and_greed_historical", {}).get("data", [])

    rows = []
    for r in records:
        ts = r.get("x")
        score = r.get("y")
        if ts is not None and score is not None:
            # timestamp는 ms 단위
            date = pd.Timestamp(ts, unit="ms").normalize()
            rows.append({"date": date, "cnn_score": float(score)})

    if not rows:
        raise ValueError("CNN API 응답에 데이터 없음")

    df = pd.DataFrame(rows).set_index("date")["cnn_score"]
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    print(f"CNN 데이터: {len(df)}일 ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


# ── 자체 모델 롤링 계산 ───────────────────────────────────────────────────────

def compute_rolling_scores(market: pd.DataFrame, fred: pd.DataFrame,
                           sample_dates: pd.DatetimeIndex) -> pd.Series:
    """각 날짜에서 look-ahead 없이 자체 점수 계산.
    market.iloc[:t+1] 슬라이스만 사용 → 미래 데이터 차단.
    """
    from modules.fear_greed import compute

    results = {}
    total = len(sample_dates)
    for i, date in enumerate(sample_dates):
        if (i + 1) % 20 == 0:
            print(f"  계산 중... {i+1}/{total} ({date.date()})")

        # look-ahead 방지: date까지의 데이터만 슬라이스
        mkt_slice = market.loc[:date]
        fred_slice = fred.loc[:date] if not fred.empty else fred

        if len(mkt_slice) < 130:  # 최소 데이터 부족
            continue

        try:
            result = compute(mkt_slice, fred_slice)
            results[date] = result["score"]
        except Exception as e:
            print(f"    [{date.date()}] 계산 실패: {e}")

    return pd.Series(results, name="model_score")


# ── 통계 분석 ─────────────────────────────────────────────────────────────────

def analyze(model: pd.Series, cnn: pd.Series) -> dict:
    """두 시리즈의 공통 날짜 기준으로 비교 통계 산출."""
    # 타임존 통일 (timezone-naive)
    model = model.copy()
    cnn = cnn.copy()
    model.index = pd.DatetimeIndex(model.index).tz_localize(None).normalize()
    cnn.index = pd.DatetimeIndex(cnn.index).tz_localize(None).normalize()
    common = model.index.intersection(cnn.index)
    m = model.loc[common]
    c = cnn.loc[common]

    diff = m - c
    mae = float(diff.abs().mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    corr = float(m.corr(c))
    bias = float(diff.mean())  # 양수 = 모델이 과낙관

    # 레이블 일치율 (극단적공포/공포/중립/탐욕/극단탐욕)
    def label(s):
        if s < 25: return "극단공포"
        if s < 45: return "공포"
        if s < 56: return "중립"
        if s < 75: return "탐욕"
        return "극단탐욕"

    match = sum(label(mv) == label(cv) for mv, cv in zip(m, c))
    label_acc = match / len(m) if len(m) > 0 else 0.0

    # 연도별 MAE
    dti = pd.DatetimeIndex(common)
    yearly = {}
    for year in sorted(set(dti.year)):
        idx = dti[dti.year == year]
        if len(idx) < 5:
            continue
        yearly[int(year)] = round(float((model.loc[idx] - cnn.loc[idx]).abs().mean()), 1)

    return {
        "n": len(common),
        "period": f"{common[0].date()} ~ {common[-1].date()}",
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "correlation": round(corr, 4),
        "bias": round(bias, 2),
        "label_accuracy": round(label_acc * 100, 1),
        "yearly_mae": yearly,
        "max_diff": round(float(diff.abs().max()), 1),
        "pct_within_10": round(float((diff.abs() <= 10).mean() * 100), 1),
        "pct_within_20": round(float((diff.abs() <= 20).mean() * 100), 1),
    }


def print_report(stats: dict) -> None:
    print("\n" + "=" * 55)
    print("  공포탐욕지수 백테스트 결과 (자체 모델 vs CNN)")
    print("=" * 55)
    print(f"  기간         : {stats['period']}")
    print(f"  비교 샘플 수 : {stats['n']}개 (주 단위)")
    print()
    print(f"  MAE          : {stats['mae']}점  (오차 평균)")
    print(f"  RMSE         : {stats['rmse']}점")
    print(f"  상관계수     : {stats['correlation']}  (1.0 = 완벽)")
    print(f"  편향(Bias)   : {stats['bias']:+.1f}점  ({'과낙관' if stats['bias'] > 0 else '과비관'})")
    print()
    print(f"  레이블 일치율 : {stats['label_accuracy']}%  (같은 공포/중립/탐욕 구간)")
    print(f"  ±10점 이내   : {stats['pct_within_10']}%")
    print(f"  ±20점 이내   : {stats['pct_within_20']}%")
    print(f"  최대 오차    : {stats['max_diff']}점")
    print()
    print("  연도별 MAE:")
    for year, mae in stats["yearly_mae"].items():
        bar = "#" * int(mae / 3)
        print(f"    {year}: {mae:5.1f}점  {bar}")
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== 공포탐욕지수 백테스트 시작 ===\n")

    # 1. 시장 데이터 로드
    print("[1] 시장 데이터 로드...")
    from modules.data_collector import DataCollector
    dc = DataCollector()
    data = dc.collect(lookback_years=11)
    market = data["market"]
    fred = data["fred"]
    print(f"  시장: {len(market)}rows ({market.index[0].date()} ~ {market.index[-1].date()})")

    # 2. CNN 데이터 로드
    print("\n[2] CNN Fear & Greed 데이터 로드...")
    try:
        cnn = fetch_cnn_history("2015-01-01")
    except Exception as e:
        print(f"  CNN API 실패: {e}")
        return

    # 3. 비교할 날짜 샘플링 (주 1회, 금요일 기준)
    print("\n[3] 롤링 백테스트 계산 중 (look-ahead 없음)...")
    # CNN API는 최근 약 1년치만 제공 — cnn 데이터 시작일 기준으로 맞춤
    start = max(pd.Timestamp("2015-06-01"), cnn.index[0])
    end = market.index[-1]

    all_dates = market.index[(market.index >= start) & (market.index <= end)]
    # 주 1회 샘플링: 각 달력 주의 마지막 실제 거래일
    # 주 1회: 매주 금요일(또는 금요일 이전 마지막 거래일) 추출
    df_dates = pd.Series(1, index=all_dates)
    weekly_last = df_dates.resample("W-FRI").last().dropna()
    sample_dates = pd.DatetimeIndex(
        all_dates[all_dates.isin(weekly_last.index)]
        if all_dates.isin(weekly_last.index).any()
        else all_dates[::5]  # fallback: 5거래일마다
    ).normalize()
    print(f"  샘플 날짜: {len(sample_dates)}개 ({sample_dates[0].date() if len(sample_dates) else '없음'} ~ {sample_dates[-1].date() if len(sample_dates) else ''})")

    model_scores = compute_rolling_scores(market, fred, sample_dates)

    # 4. 통계 분석
    print("\n[4] 통계 분석...")
    stats = analyze(model_scores, cnn)
    print_report(stats)

    # 5. 결과 저장
    out_path = ROOT / "cache" / "fear_greed_backtest.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
