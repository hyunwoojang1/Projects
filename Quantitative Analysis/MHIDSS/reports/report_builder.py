"""리포트 빌더 — 포맷터 선택 및 오케스트레이션."""

from pathlib import Path

from engine.horizons.base import HorizonResult
from .formatters.json_formatter import JSONFormatter
from .formatters.csv_formatter import CSVFormatter
from .formatters.html_formatter import HTMLFormatter


class ReportBuilder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self._formatters = {
            "json": JSONFormatter(),
            "csv": CSVFormatter(),
            "html": HTMLFormatter(),
        }

    def build(
        self,
        ticker: str,
        as_of_date: str,
        results: dict[str, HorizonResult],
        formats: list[str],
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for fmt in formats:
            fmt = fmt.strip().lower()
            if fmt in self._formatters:
                paths[fmt] = self._formatters[fmt].write(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    results=results,
                    output_dir=self.output_dir,
                )
        return paths
