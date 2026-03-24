"""MHIDSS CLI 진입점."""

from __future__ import annotations

import json
import webbrowser
from datetime import date
from pathlib import Path
from typing import Optional

import sys
import io

import typer
from rich.console import Console
from rich.table import Table

from engine.entry_score import EntryScoreEngine
from engine.horizons.base import HorizonResult
from reports.report_builder import ReportBuilder

# Windows 터미널 UTF-8 강제 설정
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

app = typer.Typer(help="Multi-Horizon Investment Decision Support System")
console = Console(legacy_windows=False, highlight=False)

SIGNAL_COLORS = {
    "STRONG_BUY": "bold green",
    "BUY": "green",
    "NEUTRAL": "yellow",
    "SELL": "red",
    "STRONG_SELL": "bold red",
}


def _resolve_ticker(query: str) -> tuple[str, str]:
    """회사명 또는 티커 → (ticker, company_name) 반환.

    - 이미 티커처럼 보이면 (≤6자, 영문자만) 그대로 반환
    - 아니면 yfinance Search로 검색 후 첫 번째 주식(EQUITY) 결과 반환
    - 실패 시 원본 문자열을 대문자로 반환
    """
    import yfinance as yf

    # 사용자가 이미 대문자로 입력했으면 티커로 간주 (AAPL, MSFT, SPY 등)
    stripped = query.strip()
    cleaned = stripped.upper()
    if stripped == cleaned and cleaned.isalpha() and len(cleaned) <= 6:
        return cleaned, cleaned

    # 회사명으로 검색
    try:
        results = yf.Search(query, max_results=5).quotes
        for r in results:
            if r.get("quoteType") in ("EQUITY", "ETF") and "." not in r.get("symbol", ""):
                symbol = r["symbol"].upper()
                name = r.get("longname") or r.get("shortname") or symbol
                return symbol, name
    except Exception:
        pass

    return cleaned, cleaned


@app.command()
def run(
    queries: list[str] = typer.Argument(..., help="티커 또는 회사명 (예: AAPL '애플' 'Microsoft' NVDA)"),
    as_of: Optional[str] = typer.Option(None, "--date", "-d", help="분석 기준일 (YYYY-MM-DD, 기본: 오늘)"),
    horizon: Optional[str] = typer.Option(None, "--horizon", "-h", help="특정 시계열만 출력 (short|mid|long)"),
    output_format: str = typer.Option("html", "--format", "-f", help="출력 형식 (json,csv,html)"),
    output_dir: Path = typer.Option(Path("./output"), "--output-dir", "-o"),
    no_browser: bool = typer.Option(False, "--no-browser", help="브라우저 자동 오픈 비활성화"),
) -> None:
    """티커 또는 회사명으로 Short/Mid/Long 시계열 대쉬보드를 생성하고 브라우저로 엽니다."""
    as_of_date = as_of or date.today().isoformat()
    formats = [f.strip() for f in output_format.split(",")]
    builder = ReportBuilder(output_dir=output_dir)
    engine = EntryScoreEngine()

    for query in queries:
        with console.status(f"[{query}] 티커 검색 중..."):
            ticker, company = _resolve_ticker(query)

        if ticker != query.strip().upper():
            console.print(f"  [dim]{query}[/dim] → [cyan]{ticker}[/cyan] ({company})")

        console.print(f"\n[bold]MHIDSS[/bold] | 티커: [cyan]{ticker}[/cyan] | 기준일: [cyan]{as_of_date}[/cyan]")

        with console.status(f"[{ticker}] 데이터 수집 및 스코어 계산 중..."):
            results = engine.run(ticker=ticker, as_of_date=as_of_date)

        if horizon:
            results = {k: v for k, v in results.items() if k == horizon}

        _print_table(results)

        paths = builder.build(ticker=ticker, as_of_date=as_of_date, results=results, formats=formats)
        for fmt, path in paths.items():
            console.print(f"  [dim]Saved:[/dim] {path}")
            if fmt == "html" and not no_browser:
                webbrowser.open(path.resolve().as_uri())

    if len(queries) > 1:
        console.print(f"\n[green]{len(queries)} dashboards generated.[/green]")

    _print_legend()


@app.command()
def validate_config() -> None:
    """Validate environment variables and config files."""
    from config import settings  # validation runs on import
    console.print("[green]Config loaded successfully.[/green]")


@app.command()
def check_connections() -> None:
    """Check connectivity to all external data sources."""
    from data.fetchers.fred_fetcher import FREDFetcher
    from data.fetchers.wrds_fetcher import WRDSFetcher
    from data.fetchers.technical_fetcher import TechnicalFetcher

    for name, fetcher in [("FRED", FREDFetcher()), ("WRDS", WRDSFetcher()), ("Technical (yfinance)", TechnicalFetcher())]:
        ok = fetcher.validate_connection()
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {name:25s} {status}")


@app.command()
def clear_cache(
    older_than: str = typer.Option("7d", help="Expiry threshold (e.g. 7d, 24h)"),
) -> None:
    """Clear expired cache files."""
    from config.settings import CACHE_DIR
    from data.cache.disk_cache import DiskCache

    hours = _parse_duration(older_than)
    cache = DiskCache(CACHE_DIR, ttl_hours=hours)
    removed = cache.clear_expired()
    console.print(f"[green]Removed {removed} expired cache file(s).[/green]")


def _print_table(results: dict[str, HorizonResult]) -> None:
    table = Table(title="Entry Score Summary", show_lines=True)
    table.add_column("Horizon", style="bold")
    table.add_column("Entry Score", justify="center")
    table.add_column("Signal", justify="center")
    table.add_column("Macro", justify="right")
    table.add_column("Fundamental", justify="right")
    table.add_column("Technical", justify="right")

    order = ["short", "mid", "long"]
    labels = {"short": "Short (1-4W)", "mid": "Mid (1-6M)", "long": "Long (6-24M)"}

    for h in order:
        if h not in results:
            continue
        r = results[h]
        color = SIGNAL_COLORS.get(r.signal, "white")
        table.add_row(
            labels[h],
            f"{r.entry_score:.1f}",
            f"[{color}]{r.signal}[/{color}]",
            f"{r.group_scores.get('macro', 0):.1f}",
            f"{r.group_scores.get('fundamental', 0):.1f}",
            f"{r.group_scores.get('technical', 0):.1f}",
        )
    console.print(table)

    first = next(iter(results.values()), None)
    if first and "_sector" in first.indicator_scores:
        sector_info = first.indicator_scores["_sector"]
        console.print(f"  [dim]Sector: {sector_info}[/dim]")


def _print_legend() -> None:
    """Print score reference and signal guide."""
    from rich.panel import Panel

    signal_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    signal_table.add_column("Signal", style="bold")
    signal_table.add_column("Score Range", justify="center")
    signal_table.add_column("Meaning")
    signal_table.add_row("[bold green]STRONG_BUY[/bold green]", "70 – 100", "All indicators favorable across groups.")
    signal_table.add_row("[green]BUY[/green]",                  "55 – 69",  "Majority of indicators lean positive.")
    signal_table.add_row("[yellow]NEUTRAL[/yellow]",            "45 – 54",  "Mixed signals; directional bias unclear.")
    signal_table.add_row("[red]SELL[/red]",                     "30 – 44",  "Majority of indicators lean negative.")
    signal_table.add_row("[bold red]STRONG_SELL[/bold red]",    " 0 – 29",  "All indicators unfavorable across groups.")

    weight_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    weight_table.add_column("Horizon", style="bold")
    weight_table.add_column("Resolution")
    weight_table.add_column("Macro", justify="center")
    weight_table.add_column("Fundamental", justify="center")
    weight_table.add_column("Technical", justify="center")
    weight_table.add_row("Short (1-4W)", "Daily",   "20%", "10%", "[bold]70%[/bold]")
    weight_table.add_row("Mid   (1-6M)", "Weekly",  "35%", "30%", "[bold]35%[/bold]")
    weight_table.add_row("Long (6-24M)", "Monthly", "[bold]50%[/bold]", "[bold]45%[/bold]", "5%")

    console.print()
    console.print(Panel(
        signal_table,
        title="[bold]Score Reference  (0 – 100)[/bold]",
        subtitle="Each indicator normalized to 0–100, then weighted average applied",
        border_style="dim",
        padding=(0, 1),
    ))
    console.print(Panel(
        weight_table,
        title="[bold]Group Weights by Horizon[/bold]",
        subtitle="Longer horizons shift weight toward Macro & Fundamental",
        border_style="dim",
        padding=(0, 1),
    ))


def _parse_duration(s: str) -> int:
    if s.endswith("d"):
        return int(s[:-1]) * 24
    if s.endswith("h"):
        return int(s[:-1])
    return int(s)


if __name__ == "__main__":
    app()
