"""CSV 리포트 포맷터."""

from pathlib import Path

import pandas as pd

from engine.horizons.base import HorizonResult


class CSVFormatter:
    def write(
        self,
        ticker: str,
        as_of_date: str,
        results: dict[str, HorizonResult],
        output_dir: Path,
    ) -> Path:
        rows = []
        for horizon_name, r in results.items():
            for ind_id, score in r.indicator_scores.items():
                rows.append({
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "horizon": horizon_name,
                    "entry_score": r.entry_score,
                    "signal": r.signal,
                    "indicator": ind_id,
                    "normalized_score": score,
                })

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{ticker}_{as_of_date}_scores.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path
