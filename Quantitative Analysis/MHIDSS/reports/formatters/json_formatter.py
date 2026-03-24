"""JSON 리포트 포맷터."""

import json
from datetime import datetime
from pathlib import Path

from engine.horizons.base import HorizonResult


class JSONFormatter:
    def write(
        self,
        ticker: str,
        as_of_date: str,
        results: dict[str, HorizonResult],
        output_dir: Path,
    ) -> Path:
        payload = {
            "metadata": {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "system_version": "0.1.0",
            },
            "horizons": {
                name: {
                    "entry_score": r.entry_score,
                    "signal": r.signal,
                    "group_scores": r.group_scores,
                    "indicator_scores": r.indicator_scores,
                    "missing_indicators": r.missing_indicators,
                    "weight_version": r.weight_version,
                }
                for name, r in results.items()
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{ticker}_{as_of_date}_report.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return path
