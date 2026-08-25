"""이메일 발송 모듈. Gmail SMTP 사용."""

from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SENDER   = os.environ.get("EMAIL_SENDER", "")
PASSWORD = os.environ.get("EMAIL_APP_PASSWORD", "")
RECIPIENT = os.environ.get("EMAIL_RECIPIENT", "")


def send_report(analysis: dict, claude_opinion: str) -> dict:
    """분석 결과를 HTML 이메일로 발송."""
    if not all([SENDER, PASSWORD, RECIPIENT]):
        return {"status": "skipped", "reason": "EMAIL_SENDER / EMAIL_APP_PASSWORD / EMAIL_RECIPIENT 미설정"}

    date_str = analysis.get("date", datetime.now().strftime("%Y-%m-%d"))
    qqq      = analysis.get("qqq_signal", {})
    regime   = analysis.get("regime", {})
    fg       = analysis.get("fear_greed", {})
    sector   = analysis.get("sector", {})
    macro    = analysis.get("macro", {})
    claude   = analysis.get("claude", {})

    # claude_opinion 인자가 있으면 우선 사용
    opinion_text = claude_opinion or claude.get("raw", "") or claude.get("reasoning", "")
    final_opinion = claude_opinion or claude.get("opinion", "분석 중")
    summary_text  = claude.get("summary", "")

    html = _build_html(
        date_str, qqq, regime, fg, sector, macro,
        final_opinion, summary_text, opinion_text,
    )

    msg = MIMEMultipart("related")
    msg["Subject"] = f"[퀀트 리포트] {date_str} — TQQQ {final_opinion}"
    msg["From"]    = SENDER
    msg["To"]      = RECIPIENT

    msg.attach(MIMEText(html, "html", "utf-8"))

    # 카드 이미지 첨부
    report_dir = Path("output") / f"report_{date_str.replace('-', '')}"
    attached = 0
    if report_dir.exists():
        for i, img_path in enumerate(sorted(report_dir.glob("card_*.png"))):
            with open(img_path, "rb") as f:
                img = MIMEImage(f.read())
            img.add_header("Content-ID", f"<card_{i}>")
            img.add_header("Content-Disposition", "inline", filename=img_path.name)
            msg.attach(img)
            attached += 1

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER, PASSWORD)
            smtp.sendmail(SENDER, RECIPIENT, msg.as_string())
        return {"status": "success", "recipient": RECIPIENT, "cards_attached": attached}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def _build_html(date_str, qqq, regime, fg, sector, macro,
                opinion, summary, reasoning) -> str:
    signal_color = {
        "강력매수": "#16a34a",
        "관망":     "#d97706",
        "회피":     "#dc2626",
    }.get(opinion, "#6b7280")

    qqq_label = qqq.get("signal_label", "—")
    regime_label = regime.get("state_label", "—")
    fg_score = fg.get("score", "—")
    fg_label = fg.get("label", "—")
    top3 = ", ".join(sector.get("top3", []))
    macro_score = macro.get("score", "—")
    macro_label = macro.get("label", "—")
    conf = qqq.get("confidence", 0)

    # 어날로그 테이블
    analogs = qqq.get("analogs", [])
    analog_rows = ""
    for a in analogs[:8]:
        t1 = f"{a['tqqq_1y_pct']}%" if a.get("tqqq_1y_pct") is not None else "N/A"
        t3 = f"{a['tqqq_3y_pct']}%" if a.get("tqqq_3y_pct") is not None else "N/A"
        dd = f"{a['drawdown_from_ath_pct']}%" if a.get("drawdown_from_ath_pct") is not None else "—"
        t1_color = "#16a34a" if a.get("tqqq_1y_pct", 0) > 0 else "#dc2626"
        analog_rows += f"""
        <tr>
          <td>{a['date']}</td>
          <td>{a.get('vix','—')}</td>
          <td>{dd}</td>
          <td style="color:{t1_color};font-weight:600">{t1}</td>
          <td>{t3}</td>
        </tr>"""

    # 예상 수익률
    exp = qqq.get("expected_returns", {})
    exp_1y = exp.get("tqqq_1y_median_pct", "—")
    exp_1y_range = exp.get("tqqq_1y_range_pct", ["—", "—"])
    exp_3y = exp.get("tqqq_3y_median_pct", "—")

    reasoning_html = reasoning.replace("\n", "<br>") if reasoning else "—"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: -apple-system, 'Malgun Gothic', sans-serif; background:#f8fafc; margin:0; padding:20px; color:#1e293b; }}
  .container {{ max-width:680px; margin:0 auto; }}
  .header {{ background:linear-gradient(135deg,#1e293b,#334155); color:white; padding:28px; border-radius:12px 12px 0 0; }}
  .header h1 {{ margin:0 0 4px; font-size:22px; }}
  .header p {{ margin:0; opacity:.7; font-size:14px; }}
  .opinion-box {{ background:{signal_color}; color:white; padding:20px 28px; text-align:center; }}
  .opinion-box .label {{ font-size:32px; font-weight:700; }}
  .opinion-box .summary {{ margin-top:6px; font-size:14px; opacity:.9; }}
  .section {{ background:white; padding:24px 28px; border-bottom:1px solid #e2e8f0; }}
  .section h2 {{ margin:0 0 16px; font-size:15px; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }}
  .signals {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .signal-card {{ background:#f1f5f9; padding:14px; border-radius:8px; }}
  .signal-card .name {{ font-size:12px; color:#64748b; }}
  .signal-card .value {{ font-size:16px; font-weight:600; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#f1f5f9; padding:8px 12px; text-align:left; color:#64748b; font-weight:600; }}
  td {{ padding:8px 12px; border-bottom:1px solid #f1f5f9; }}
  .reasoning {{ background:#fffbeb; border-left:4px solid #f59e0b; padding:16px; border-radius:0 8px 8px 0; font-size:14px; line-height:1.7; }}
  .footer {{ background:#f1f5f9; padding:16px 28px; border-radius:0 0 12px 12px; font-size:12px; color:#94a3b8; text-align:center; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>퀀트 투자 리포트</h1>
    <p>{date_str} · TQQQ 장기 진입 분석</p>
  </div>

  <div class="opinion-box">
    <div class="label">{opinion}</div>
    <div class="summary">{summary}</div>
  </div>

  <div class="section">
    <h2>신호 요약</h2>
    <div class="signals">
      <div class="signal-card">
        <div class="name">QQQ 타이밍 모델</div>
        <div class="value">{qqq_label} ({conf:.0%})</div>
      </div>
      <div class="signal-card">
        <div class="name">시장 국면 (HMM)</div>
        <div class="value">{regime_label}</div>
      </div>
      <div class="signal-card">
        <div class="name">공포탐욕지수</div>
        <div class="value">{fg_score} / 100 ({fg_label})</div>
      </div>
      <div class="signal-card">
        <div class="name">매크로 점수</div>
        <div class="value">{macro_score}점 ({macro_label})</div>
      </div>
      <div class="signal-card">
        <div class="name">강세 섹터 TOP3</div>
        <div class="value">{top3}</div>
      </div>
      <div class="signal-card">
        <div class="name">예상 TQQQ 1Y (중앙값)</div>
        <div class="value">{exp_1y}% ({exp_1y_range[0]}~{exp_1y_range[1]}%)</div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Claude 종합 분석</h2>
    <div class="reasoning">{reasoning_html}</div>
  </div>

  <div class="section">
    <h2>유사 과거 구간 (TOP 8)</h2>
    <table>
      <thead><tr><th>날짜</th><th>VIX</th><th>ATH 낙폭</th><th>TQQQ 1Y</th><th>TQQQ 3Y</th></tr></thead>
      <tbody>{analog_rows}</tbody>
    </table>
    <p style="font-size:12px;color:#94a3b8;margin-top:8px">
      예상 3Y 중앙값: {exp_3y}%
    </p>
  </div>

  <div class="footer">
    이 리포트는 투자 참고용입니다. 투자 결정의 책임은 본인에게 있습니다.<br>
    Quant Signal Agent · {date_str}
  </div>
</div>
</body>
</html>"""
