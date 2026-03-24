"""HTML dashboard formatter — trading terminal style."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from engine.horizons.base import HorizonResult, classify_signal


# ── Stock price fetch ─────────────────────────────────────────────────────────

def _fetch_price(ticker: str) -> tuple[float | None, bool]:
    """Return (price, is_realtime). Returns (None, False) on failure."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        state = info.get("marketState", "CLOSED")
        if state == "REGULAR":
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            return float(price), True
        else:
            price = info.get("regularMarketPreviousClose") or info.get("previousClose")
            return float(price), False
    except Exception:
        return None, False


# ── Score → CSS class ─────────────────────────────────────────────────────────

def _signal_class(score: float) -> str:
    sig = classify_signal(score)
    return {
        "STRONG_BUY": "strong-buy",
        "BUY": "buy",
        "NEUTRAL": "neutral",
        "SELL": "sell",
        "STRONG_SELL": "strong-sell",
    }.get(sig, "neutral")


def _signal_label(score: float) -> str:
    sig = classify_signal(score)
    return {
        "STRONG_BUY": "STRONG BUY",
        "BUY": "BUY",
        "NEUTRAL": "NEUTRAL",
        "SELL": "SELL",
        "STRONG_SELL": "STRONG SELL",
    }.get(sig, "NEUTRAL")


# ── HTML template ─────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg: #0d1117;
  --bg2: #161b22;
  --bg3: #21262d;
  --border: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --strong-buy: #1a7c3e;
  --strong-buy-bg: #0d2b1a;
  --buy: #2ea043;
  --buy-bg: #0d2818;
  --neutral: #d29922;
  --neutral-bg: #2d2208;
  --sell: #da3633;
  --sell-bg: #2d0f0e;
  --strong-sell: #b91c1c;
  --strong-sell-bg: #200b0b;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  min-height: 100vh;
  padding: 24px;
}

.dashboard {
  max-width: 960px;
  margin: 0 auto;
}

/* Header */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 28px;
  margin-bottom: 20px;
}

.header-left {
  display: flex;
  align-items: baseline;
  gap: 14px;
}

.ticker-symbol {
  font-size: 32px;
  font-weight: 700;
  letter-spacing: 1px;
  color: #58a6ff;
}

.sector-tag {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 12px;
  color: var(--muted);
}

.header-right {
  text-align: right;
}

.price-block {
  display: flex;
  align-items: baseline;
  gap: 8px;
  justify-content: flex-end;
}

.price-value {
  font-size: 28px;
  font-weight: 600;
  color: #e6edf3;
}

.price-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
}

.price-badge.realtime {
  background: #1a3a1a;
  color: #4caf50;
  border: 1px solid #2ea043;
}

.price-badge.close {
  background: #1a1a2e;
  color: #8b949e;
  border: 1px solid var(--border);
}

.date-line {
  color: var(--muted);
  font-size: 13px;
  margin-top: 4px;
}

/* 3x3 grid */
.grid-section {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 20px;
}

.grid-title {
  padding: 14px 20px;
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  border-bottom: 1px solid var(--border);
  background: var(--bg3);
}

.score-grid {
  width: 100%;
  border-collapse: collapse;
}

.score-grid th {
  padding: 12px 16px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--muted);
  background: var(--bg3);
  border: 1px solid var(--border);
}

.score-grid th.row-header {
  text-align: left;
  width: 130px;
  font-size: 13px;
  color: var(--text);
}

.score-grid th.col-header { text-align: center; }

.score-grid th.col-header .col-label {
  display: block;
  font-size: 14px;
  color: var(--text);
  font-weight: 700;
  margin-bottom: 2px;
}

.score-grid th.col-header .col-sub {
  display: block;
  font-size: 11px;
  color: var(--muted);
  font-weight: 400;
}

.score-cell {
  border: 1px solid var(--border);
  padding: 16px;
  text-align: center;
  vertical-align: middle;
}

.score-cell .score-num {
  font-size: 26px;
  font-weight: 700;
  line-height: 1;
  margin-bottom: 6px;
}

.score-cell .sig-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  padding: 3px 10px;
  border-radius: 4px;
  display: inline-block;
  margin-bottom: 10px;
}

.score-cell .bar-track {
  background: rgba(255,255,255,0.08);
  border-radius: 4px;
  height: 5px;
  overflow: hidden;
}

.score-cell .bar-fill {
  height: 5px;
  border-radius: 4px;
  transition: width 0.6s ease;
}

/* Cell colors */
.strong-buy .score-num { color: #56d364; }
.strong-buy .sig-label { background: var(--strong-buy-bg); color: #56d364; border: 1px solid var(--strong-buy); }
.strong-buy .bar-fill  { background: #56d364; }

.buy .score-num { color: #3fb950; }
.buy .sig-label { background: var(--buy-bg); color: #3fb950; border: 1px solid var(--buy); }
.buy .bar-fill  { background: #3fb950; }

.neutral .score-num { color: #d29922; }
.neutral .sig-label { background: var(--neutral-bg); color: #d29922; border: 1px solid var(--neutral); }
.neutral .bar-fill  { background: #d29922; }

.sell .score-num { color: #f85149; }
.sell .sig-label { background: var(--sell-bg); color: #f85149; border: 1px solid var(--sell); }
.sell .bar-fill  { background: #f85149; }

.strong-sell .score-num { color: #ff7b72; }
.strong-sell .sig-label { background: var(--strong-sell-bg); color: #ff7b72; border: 1px solid var(--strong-sell); }
.strong-sell .bar-fill  { background: #ff7b72; }

/* Entry score cards */
.entry-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 20px;
}

.entry-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px 20px;
  text-align: center;
  position: relative;
  overflow: hidden;
}

.entry-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
}

.entry-card.strong-buy::before { background: #56d364; }
.entry-card.buy::before         { background: #3fb950; }
.entry-card.neutral::before     { background: #d29922; }
.entry-card.sell::before        { background: #f85149; }
.entry-card.strong-sell::before { background: #ff7b72; }

.entry-card .horizon-label {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 4px;
}

.entry-card .horizon-sub {
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 16px;
}

.entry-card .entry-score-num {
  font-size: 52px;
  font-weight: 700;
  line-height: 1;
  margin-bottom: 12px;
}

.entry-card.strong-buy .entry-score-num { color: #56d364; }
.entry-card.buy .entry-score-num         { color: #3fb950; }
.entry-card.neutral .entry-score-num     { color: #d29922; }
.entry-card.sell .entry-score-num        { color: #f85149; }
.entry-card.strong-sell .entry-score-num { color: #ff7b72; }

.entry-card .entry-signal {
  font-size: 13px;
  font-weight: 700;
  padding: 5px 16px;
  border-radius: 6px;
  display: inline-block;
  margin-bottom: 16px;
}

.entry-card.strong-buy .entry-signal { background: var(--strong-buy-bg); color: #56d364; border: 1px solid var(--strong-buy); }
.entry-card.buy .entry-signal         { background: var(--buy-bg); color: #3fb950; border: 1px solid var(--buy); }
.entry-card.neutral .entry-signal     { background: var(--neutral-bg); color: #d29922; border: 1px solid var(--neutral); }
.entry-card.sell .entry-signal        { background: var(--sell-bg); color: #f85149; border: 1px solid var(--sell); }
.entry-card.strong-sell .entry-signal { background: var(--strong-sell-bg); color: #ff7b72; border: 1px solid var(--strong-sell); }

.entry-card .weight-bar {
  display: flex;
  height: 6px;
  border-radius: 4px;
  overflow: hidden;
  margin-top: 16px;
  gap: 2px;
}

.weight-bar .wb-macro { background: #58a6ff; }
.weight-bar .wb-fund  { background: #d2a8ff; }
.weight-bar .wb-tech  { background: #ffa657; }

.entry-card .weight-legend {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 8px;
  font-size: 11px;
  color: var(--muted);
}

.weight-legend span::before {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: middle;
  content: "";
}

.weight-legend .wl-macro::before { background: #58a6ff; }
.weight-legend .wl-fund::before  { background: #d2a8ff; }
.weight-legend .wl-tech::before  { background: #ffa657; }

/* Score legend */
.legend-section {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 20px;
}

.legend-title {
  padding: 14px 20px;
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  border-bottom: 1px solid var(--border);
  background: var(--bg3);
}

.legend-body {
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.legend-intro {
  font-size: 13px;
  color: var(--muted);
  line-height: 1.6;
}

.legend-scale {
  display: flex;
  align-items: stretch;
  height: 10px;
  border-radius: 6px;
  overflow: hidden;
  margin: 4px 0 2px;
}

.legend-scale .ls-strong-sell { background: #ff7b72; flex: 30; }
.legend-scale .ls-sell        { background: #f85149; flex: 15; }
.legend-scale .ls-neutral     { background: #d29922; flex: 10; }
.legend-scale .ls-buy         { background: #3fb950; flex: 15; }
.legend-scale .ls-strong-buy  { background: #56d364; flex: 30; }

.legend-scale-labels {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 12px;
}

.legend-rows {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.legend-row {
  display: flex;
  align-items: center;
  gap: 14px;
}

.legend-row .lr-badge {
  font-size: 12px;
  font-weight: 700;
  padding: 4px 14px;
  border-radius: 5px;
  min-width: 110px;
  text-align: center;
}

.lr-badge.strong-buy  { background: var(--strong-buy-bg); color: #56d364; border: 1px solid var(--strong-buy); }
.lr-badge.buy         { background: var(--buy-bg);        color: #3fb950; border: 1px solid var(--buy); }
.lr-badge.neutral     { background: var(--neutral-bg);    color: #d29922; border: 1px solid var(--neutral); }
.lr-badge.sell        { background: var(--sell-bg);       color: #f85149; border: 1px solid var(--sell); }
.lr-badge.strong-sell { background: var(--strong-sell-bg);color: #ff7b72; border: 1px solid var(--strong-sell); }

.legend-row .lr-range {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  min-width: 70px;
}

.legend-row .lr-desc {
  font-size: 13px;
  color: var(--muted);
}

/* Footer */
footer {
  text-align: center;
  color: var(--muted);
  font-size: 12px;
  padding: 12px 0;
  border-top: 1px solid var(--border);
}
"""

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MHIDSS · {ticker}</title>
<style>{css}</style>
</head>
<body>
<div class="dashboard">

  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <div class="ticker-symbol">{ticker}</div>
      {sector_tag}
    </div>
    <div class="header-right">
      <div class="price-block">
        <div class="price-value">{price_str}</div>
        <span class="price-badge {price_badge_cls}">{price_badge_txt}</span>
      </div>
      <div class="date-line">As of {date_str}</div>
    </div>
  </div>

  <!-- 3x3 Analysis Grid -->
  <div class="grid-section">
    <div class="grid-title">Analysis View — Group &times; Horizon</div>
    <table class="score-grid">
      <thead>
        <tr>
          <th class="row-header"></th>
          <th class="col-header">
            <span class="col-label">SHORT</span>
            <span class="col-sub">1 – 4 weeks</span>
          </th>
          <th class="col-header">
            <span class="col-label">MID</span>
            <span class="col-sub">1 – 6 months</span>
          </th>
          <th class="col-header">
            <span class="col-label">LONG</span>
            <span class="col-sub">6 – 24 months</span>
          </th>
        </tr>
      </thead>
      <tbody>
        {grid_rows}
      </tbody>
    </table>
  </div>

  <!-- Entry Score Cards -->
  <div class="entry-cards">
    {entry_cards}
  </div>

  <!-- Score Legend -->
  <div class="legend-section">
    <div class="legend-title">Score Guide</div>
    <div class="legend-body">
      <div class="legend-intro">
        All scores are normalized to a <strong>0 – 100</strong> scale using within-sector z-scores.
        Each group (Fundamental, Macro, Technical) is scored independently, then combined with
        horizon-specific weights into a single <strong>Entry Score</strong>.
      </div>

      <div>
        <div class="legend-scale">
          <div class="ls-strong-sell"></div>
          <div class="ls-sell"></div>
          <div class="ls-neutral"></div>
          <div class="ls-buy"></div>
          <div class="ls-strong-buy"></div>
        </div>
        <div class="legend-scale-labels">
          <span>0</span>
          <span>30</span>
          <span>45</span>
          <span>55</span>
          <span>70</span>
          <span>100</span>
        </div>
      </div>

      <div class="legend-rows">
        <div class="legend-row">
          <div class="lr-badge strong-buy">STRONG BUY</div>
          <div class="lr-range">70 – 100</div>
          <div class="lr-desc">All indicators are favorable across groups.</div>
        </div>
        <div class="legend-row">
          <div class="lr-badge buy">BUY</div>
          <div class="lr-range">55 – 69</div>
          <div class="lr-desc">Majority of indicators lean positive.</div>
        </div>
        <div class="legend-row">
          <div class="lr-badge neutral">NEUTRAL</div>
          <div class="lr-range">45 – 54</div>
          <div class="lr-desc">Mixed signals; directional bias unclear.</div>
        </div>
        <div class="legend-row">
          <div class="lr-badge sell">SELL</div>
          <div class="lr-range">30 – 44</div>
          <div class="lr-desc">Majority of indicators lean negative.</div>
        </div>
        <div class="legend-row">
          <div class="lr-badge strong-sell">STRONG SELL</div>
          <div class="lr-range">0 – 29</div>
          <div class="lr-desc">All indicators are unfavorable across groups.</div>
        </div>
      </div>
    </div>
  </div>

  <footer>MHIDSS &nbsp;|&nbsp; Weight version: {weight_version} &nbsp;|&nbsp; Generated: {generated_at}</footer>
</div>
</body>
</html>"""

# Horizon group weights (for weight-bar visualization)
_HORIZON_WEIGHTS = {
    "short": {"macro": 0.20, "fundamental": 0.10, "technical": 0.70},
    "mid":   {"macro": 0.35, "fundamental": 0.30, "technical": 0.35},
    "long":  {"macro": 0.50, "fundamental": 0.45, "technical": 0.05},
}

_HORIZON_LABELS = {
    "short": ("SHORT", "1 – 4 weeks"),
    "mid":   ("MID",   "1 – 6 months"),
    "long":  ("LONG",  "6 – 24 months"),
}

_GROUP_LABELS = {
    "fundamental": "Fundamental",
    "macro":       "Macro",
    "technical":   "Technical",
}

_SIGNAL_EN = {
    "STRONG_BUY":  "STRONG BUY",
    "BUY":         "BUY",
    "NEUTRAL":     "NEUTRAL",
    "SELL":        "SELL",
    "STRONG_SELL": "STRONG SELL",
}


class HTMLFormatter:
    def write(
        self,
        ticker: str,
        as_of_date: str,
        results: dict[str, HorizonResult],
        output_dir: Path,
    ) -> Path:
        price, is_realtime = _fetch_price(ticker)

        if price is not None:
            price_str = f"${price:,.2f}"
            price_badge_cls = "realtime" if is_realtime else "close"
            price_badge_txt = "Live" if is_realtime else "Close"
        else:
            price_str = "—"
            price_badge_cls = "close"
            price_badge_txt = "N/A"

        # Sector info
        sector_info = ""
        first = next(iter(results.values()), None)
        if first and "_sector" in first.indicator_scores:
            raw = str(first.indicator_scores["_sector"])
            sector_info = raw.split("(")[0].strip()

        sector_tag = (
            f'<div class="sector-tag">{sector_info}</div>' if sector_info else ""
        )

        grid_rows   = _build_grid_rows(results)
        entry_cards = _build_entry_cards(results)

        weight_version = first.weight_version if first else ""
        generated_at   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        html = _HTML.format(
            css=_CSS,
            ticker=ticker,
            sector_tag=sector_tag,
            price_str=price_str,
            price_badge_cls=price_badge_cls,
            price_badge_txt=price_badge_txt,
            date_str=as_of_date,
            grid_rows=grid_rows,
            entry_cards=entry_cards,
            weight_version=weight_version,
            generated_at=generated_at,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{ticker}_{as_of_date}_dashboard.html"
        path.write_text(html, encoding="utf-8")
        return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_cell(score: float | None) -> str:
    if score is None or score != score:  # NaN guard
        return '<td class="score-cell neutral"><div class="score-num" style="color:var(--muted)">—</div></td>'
    cls = _signal_class(score)
    lbl = _signal_label(score)
    pct = min(max(score, 0), 100)
    return f"""<td class="score-cell {cls}">
      <div class="score-num">{score:.1f}</div>
      <div class="sig-label">{lbl}</div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
    </td>"""


def _build_grid_rows(results: dict[str, HorizonResult]) -> str:
    horizons = ["short", "mid", "long"]
    groups   = ["fundamental", "macro", "technical"]
    rows_html = []
    for group in groups:
        cells = []
        for h in horizons:
            r = results.get(h)
            score = r.group_scores.get(group) if r else None
            cells.append(_score_cell(score))
        rows_html.append(
            f"<tr><th class='row-header'>{_GROUP_LABELS[group]}</th>{''.join(cells)}</tr>"
        )
    return "\n".join(rows_html)


def _build_entry_cards(results: dict[str, HorizonResult]) -> str:
    horizons = ["short", "mid", "long"]
    cards = []
    for h in horizons:
        r = results.get(h)
        if r is None:
            continue
        cls    = _signal_class(r.entry_score)
        sig_en = _SIGNAL_EN.get(r.signal, r.signal)
        label, sub = _HORIZON_LABELS[h]
        w = _HORIZON_WEIGHTS[h]

        wb_macro = w["macro"] * 100
        wb_fund  = w["fundamental"] * 100
        wb_tech  = w["technical"] * 100

        cards.append(f"""
<div class="entry-card {cls}">
  <div class="horizon-label">{label}</div>
  <div class="horizon-sub">{sub}</div>
  <div class="entry-score-num">{r.entry_score:.1f}</div>
  <div class="entry-signal">{sig_en}</div>
  <div class="weight-bar">
    <div class="wb-macro" style="flex:{wb_macro}"></div>
    <div class="wb-fund"  style="flex:{wb_fund}"></div>
    <div class="wb-tech"  style="flex:{wb_tech}"></div>
  </div>
  <div class="weight-legend">
    <span class="wl-macro">Macro {int(wb_macro)}%</span>
    <span class="wl-fund">Fundamental {int(wb_fund)}%</span>
    <span class="wl-tech">Technical {int(wb_tech)}%</span>
  </div>
</div>""")
    return "\n".join(cards)
