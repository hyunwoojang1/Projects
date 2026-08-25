"""Claude Code CLI를 subprocess로 호출해 신호를 종합 분석.

별도 API 키 불필요 — 현재 로그인된 Claude Code 계정 사용.
"""

from __future__ import annotations

import json
import subprocess
import warnings
from datetime import datetime


def analyze(signals: dict) -> dict:
    """모든 모듈 신호를 Claude에게 넘겨 종합 의견을 받는다.

    Returns:
        {
            "opinion":    str,   # 최종 의견 (매수/관망/회피)
            "summary":    str,   # 한 줄 요약
            "reasoning":  str,   # 상세 분석
            "risks":      str,   # 주요 리스크
            "raw":        str,   # Claude 전체 응답
        }
    """
    prompt = _build_prompt(signals)
    raw = _call_claude(prompt)
    if raw is None:
        return _fallback(signals)
    return _parse(raw)


# ── Prompt ─────────────────────────────────────────────────────────────────────

def _build_prompt(signals: dict) -> str:
    date_str = signals.get("date", datetime.now().strftime("%Y-%m-%d"))

    qqq  = signals.get("qqq_signal", {})
    reg  = signals.get("regime", {})
    fg   = signals.get("fear_greed", {})
    sec  = signals.get("sector", {})
    mac  = signals.get("macro", {})

    analogs = qqq.get("analogs", [])
    analog_lines = ""
    for a in analogs[:5]:
        t1 = f"{a['tqqq_1y_pct']}%" if a.get("tqqq_1y_pct") is not None else "N/A"
        t3 = f"{a['tqqq_3y_pct']}%" if a.get("tqqq_3y_pct") is not None else "N/A"
        dd = f"{a['drawdown_from_ath_pct']}%" if a.get("drawdown_from_ath_pct") is not None else "N/A"
        analog_lines += (
            f"  - {a['date']}  ATH낙폭:{dd}  VIX:{a.get('vix','?')}"
            f"  → 1Y:{t1}  3Y:{t3}\n"
        )

    exp = qqq.get("expected_returns", {})
    exp_1y = exp.get("tqqq_1y_median_pct", "N/A")
    exp_1y_range = exp.get("tqqq_1y_range_pct", ["N/A", "N/A"])
    exp_3y = exp.get("tqqq_3y_median_pct", "N/A")

    return f"""당신은 퀀트 투자 분석 AI입니다.
아래는 {date_str} 기준으로 여러 모델이 산출한 투자 신호입니다.
현재 QLD를 보유 중이며, TQQQ 추가 진입 여부를 판단해야 합니다.

━━━ 신호 요약 ━━━
• QQQ 타이밍 모델 : {qqq.get('signal_label', '?')} (신뢰도 {qqq.get('confidence', 0):.1%})
  - XGBoost  : {qqq.get('models', {}).get('xgboost', '?')}
  - LightGBM : {qqq.get('models', {}).get('lightgbm', '?')}
  - LSTM     : {qqq.get('models', {}).get('lstm', '?')}
  - Transformer: {qqq.get('models', {}).get('transformer', '?')}

• 시장 국면 (HMM) : {reg.get('state_label', '?')}
  - 확률: {json.dumps(reg.get('state_probs', {}), ensure_ascii=False)}
  - 전략: {reg.get('strategy', '?')}

• 공포탐욕 지수  : {fg.get('score', '?')} / 100 ({fg.get('label', '?')})
  - 1주일 전: {fg.get('week_ago', '?')}  변화: {fg.get('change', '?')}

• 섹터 로테이션  : 강세 TOP3 = {sec.get('top3', [])}
  - 경기 국면: {sec.get('phase', '?')}

• 매크로 점수    : {mac.get('score', '?')}점 ({mac.get('label', '?')})

━━━ 유사 과거 구간 (TOP 5) ━━━
{analog_lines}
유사 구간 기반 예상 수익률:
  TQQQ 1Y 중앙값: {exp_1y}%  (범위: {exp_1y_range[0]}% ~ {exp_1y_range[1]}%)
  TQQQ 3Y 중앙값: {exp_3y}%

━━━ 분석 요청 ━━━
다음 형식으로 정확히 답해주세요 (한국어):

[최종의견]
매수 / 관망 / 회피 중 하나

[한줄요약]
한 문장으로

[상세분석]
- 신호 간 충돌이 있다면 해석
- 유사 과거 구간과 현재 비교
- 지금 TQQQ 진입이 좋은/나쁜 이유

[주요리스크]
- 리스크 요인 2~3개"""


# ── Claude CLI 호출 ────────────────────────────────────────────────────────────

def _call_claude(prompt: str) -> str | None:
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--print"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        if result.returncode != 0:
            warnings.warn(f"Claude CLI 오류: {result.stderr[:200]}")
            return None
        return result.stdout.strip()
    except FileNotFoundError:
        warnings.warn("claude CLI를 찾을 수 없음 — PATH 확인 필요")
        return None
    except subprocess.TimeoutExpired:
        warnings.warn("Claude CLI 타임아웃 (120초)")
        return None
    except Exception as e:
        warnings.warn(f"Claude CLI 호출 실패: {e}")
        return None


# ── 응답 파싱 ──────────────────────────────────────────────────────────────────

def _parse(raw: str) -> dict:
    sections = {"opinion": "", "summary": "", "reasoning": "", "risks": ""}

    mapping = {
        "[최종의견]": "opinion",
        "[한줄요약]": "summary",
        "[상세분석]": "reasoning",
        "[주요리스크]": "risks",
    }

    current = None
    lines = raw.splitlines()
    buf: list[str] = []

    def _flush():
        if current:
            sections[current] = "\n".join(buf).strip()

    for line in lines:
        matched = next((v for k, v in mapping.items() if k in line), None)
        if matched:
            _flush()
            current = matched
            buf = []
        else:
            if current:
                buf.append(line)

    _flush()

    return {**sections, "raw": raw}


def _fallback(signals: dict) -> dict:
    """Claude 호출 실패 시 규칙 기반 최소 의견."""
    qqq_label = signals.get("qqq_signal", {}).get("signal_label", "")
    regime_label = signals.get("regime", {}).get("state_label", "")

    if "Bear" in regime_label and "강력" in qqq_label:
        opinion = "관망"
        summary = "QQQ 모델은 매수, HMM은 Bear — 충돌로 인해 관망 권장"
    elif "강력" in qqq_label:
        opinion = "매수"
        summary = "TQQQ 진입 신호 감지"
    elif "회피" in qqq_label:
        opinion = "회피"
        summary = "하락 위험 — QLD 유지"
    else:
        opinion = "관망"
        summary = "신호 불확실 — 추가 조정 대기"

    return {
        "opinion":   opinion,
        "summary":   summary,
        "reasoning": "(Claude CLI 호출 실패 — 규칙 기반 fallback)",
        "risks":     "Claude 분석 불가",
        "raw":       "",
    }
