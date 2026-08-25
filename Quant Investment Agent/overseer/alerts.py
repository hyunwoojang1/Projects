"""알림 시스템 — 카카오톡(메인) + 슬랙(보조).

카카오톡: 모든 레벨 (INFO / WARNING / CRITICAL / SUCCESS)
슬랙:     중요 레벨만 (CRITICAL / SUCCESS)

카카오톡 토큰 관리:
  - access_token 만료(401) 시 refresh_token으로 자동 갱신
  - 갱신된 토큰은 .env에 즉시 저장 (다음 실행에도 유효)
  - refresh_token도 새 값이 오면 함께 업데이트

실행 테스트:
  python overseer/alerts.py --test
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

load_dotenv(ENV_PATH)

KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_MEMO_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

ALERT_EMOJI = {
    "INFO":     "💬",
    "WARNING":  "⚠️",
    "CRITICAL": "🚨",
    "SUCCESS":  "✅",
}

SLACK_LEVELS = {"CRITICAL", "SUCCESS"}


class AlertSystem:
    """중앙 알림 시스템. 모든 알림은 이 클래스를 통해 발송."""

    def send(self, level: str, message: str,
             details: list[str] | None = None) -> None:
        emoji = ALERT_EMOJI.get(level, "📌")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        text = f"{emoji} 퀀트 Agent\n📅 {ts}\n\n{message}"
        if details:
            text += "\n\n상세:\n" + "\n".join(f"  • {d}" for d in details)

        _send_kakao(text)

        if level in SLACK_LEVELS:
            _send_slack(text)

    # ── 시나리오별 헬퍼 ────────────────────────────────────────────────────────

    def notify_start(self, date_str: str) -> None:
        self.send("INFO", f"분석 시작 — {date_str}\n데이터 수집 중...")

    def notify_warning(self, module: str, warnings: list[str]) -> None:
        self.send("WARNING", f"{module} 경고 발생 — 실행 계속", warnings)

    def notify_critical_halt(self, module: str, flags: list[str]) -> None:
        self.send("CRITICAL",
                  f"🛑 전체 중단\n원인 모듈: {module}\n수동 확인 필요",
                  flags)

    def notify_upload_blocked(self, reasons: list[str]) -> None:
        self.send("CRITICAL", "🛑 업로드 차단 — 수동 확인 필요", reasons)

    def notify_upload_success(self, fg_score: int, signal: str) -> None:
        self.send("SUCCESS",
                  f"인스타 업로드 완료\nQQQ 신호: {signal}\n공포탐욕: {fg_score}점")

    def notify_daily_summary(self, log: dict) -> None:
        warnings = log.get("warnings", [])
        self.send("INFO",
                  f"일일 요약\n실행시간: {log.get('runtime_seconds', 0):.0f}초\n"
                  f"경고 수: {len(warnings)}개\n업로드: {log.get('upload_status', '?')}",
                  warnings if warnings else None)

    def notify_analysis(self, analysis: dict) -> None:
        """전체 분석 결과를 상세하게 카카오톡으로 전송."""
        date = analysis.get("date", "")
        qqq_sig = analysis.get("qqq_signal", {})
        regime = analysis.get("regime", {})
        fg = analysis.get("fear_greed", {})
        sector = analysis.get("sector", {})
        macro = analysis.get("macro", {})
        claude = analysis.get("claude", {})

        signal = qqq_sig.get("signal_label", "?")
        conf = qqq_sig.get("confidence", 0)
        models = qqq_sig.get("models", {})

        # 개별 모델 의견
        model_lines = [f"  {m}: {v}" for m, v in models.items()] if models else []

        # 섹터 이름 매핑
        sector_names = {
            "XLK": "XLK (기술/Tech)",
            "XLE": "XLE (에너지/Energy)",
            "XLF": "XLF (금융/Financials)",
            "XLV": "XLV (헬스케어/Healthcare)",
            "XLI": "XLI (산업재/Industrials)",
            "XLY": "XLY (임의소비재/Cons.Discr.)",
            "XLP": "XLP (필수소비재/Cons.Staples)",
            "XLB": "XLB (소재/Materials)",
            "XLRE": "XLRE (부동산/Real Estate)",
            "XLU": "XLU (유틸리티/Utilities)",
            "XLC": "XLC (커뮤니케이션/Comm.)",
        }
        top3 = [sector_names.get(t, t) for t in sector.get("top3", [])]
        bottom3 = [sector_names.get(t, t) for t in sector.get("bottom3", [])]

        # 공포탐욕 컴포넌트
        fg_comps = fg.get("components", {})
        comp_names = {
            "market_momentum": "시장 모멘텀",
            "price_strength": "주가 강도",
            "price_breadth": "주가 폭",
            "put_call": "풋/콜 비율",
            "volatility": "시장 변동성",
            "safe_haven": "안전자산 수요",
            "junk_bond": "정크본드 수요",
        }
        comp_lines = [f"  {comp_names.get(k, k)}: {v}점"
                      for k, v in fg_comps.items()]

        # 매크로 컴포넌트
        macro_comps = macro.get("components", {})
        macro_lines = [f"  {k}: {v}점" for k, v in macro_comps.items()]

        # 아날로그 (유사 과거 구간)
        analogs = qqq_sig.get("analogs", [])
        analog_lines = []
        for a in analogs[:3]:
            d = a.get("date", "")[:7]
            sim = a.get("similarity_pct", 0)
            r1y = a.get("tqqq_1y_return", None)
            r1y_str = f"{r1y:+.0f}%" if r1y is not None else "?"
            analog_lines.append(f"  {d} (유사도 {sim:.0f}%) → 1Y: {r1y_str}")

        text = (
            f"📊 퀀트 시그널 — {date}\n"
            f"{'─'*30}\n\n"
            f"【QQQ 타이밍 신호】\n"
            f"  {signal}  신뢰도 {conf:.0%}\n"
        )
        if model_lines:
            text += "\n".join(model_lines) + "\n"

        text += (
            f"\n【시장 레짐 (HMM)】\n"
            f"  {regime.get('state_label', '?')}\n"
            f"  {regime.get('regime_description', '')}\n"
            f"\n【공포탐욕지수】\n"
            f"  {fg.get('score', '?')}점 — {fg.get('label', '?')}\n"
            f"  지난주 대비 {fg.get('change', 0):+d}점\n"
        )
        if comp_lines:
            text += "\n".join(comp_lines) + "\n"

        text += (
            f"\n【섹터 로테이션】\n"
            f"  강세 ▲ {' | '.join(top3)}\n"
            f"  약세 ▼ {' | '.join(bottom3)}\n"
            f"  사이클: {sector.get('cycle_phase', '?')}\n"
            f"\n【매크로 환경】\n"
            f"  {macro.get('score', '?')}점 — {macro.get('label', '?')}\n"
        )
        if macro_lines:
            text += "\n".join(macro_lines) + "\n"

        if analog_lines:
            text += f"\n【과거 유사 구간 TOP3】\n" + "\n".join(analog_lines) + "\n"

        text += (
            f"\n【Claude 종합 의견】\n"
            f"  {claude.get('opinion', '?')}\n"
            f"  {claude.get('summary', '')}\n"
            f"\n{'─'*30}\n"
            f"⚠️ 투자 권유 아님. 참고용 데이터 분석."
        )

        _send_kakao(text[:1000])


# ── 카카오 토큰 관리 ──────────────────────────────────────────────────────────

def _get_kakao_env() -> tuple[str, str, str]:
    """현재 .env에서 카카오 토큰 3종 읽기 (실행 시마다 최신값 반영)."""
    load_dotenv(ENV_PATH, override=True)
    return (
        os.environ.get("KAKAO_REST_API_KEY", ""),
        os.environ.get("KAKAO_ACCESS_TOKEN", ""),
        os.environ.get("KAKAO_REFRESH_TOKEN", ""),
    )


def _update_env(key: str, value: str) -> None:
    """`.env` 파일에서 특정 키의 값을 in-place로 교체."""
    content = ENV_PATH.read_text(encoding="utf-8")
    pattern = re.compile(rf"^({re.escape(key)}=).*$", re.MULTILINE)
    if pattern.search(content):
        new_content = pattern.sub(rf"\g<1>{value}", content)
    else:
        new_content = content.rstrip("\n") + f"\n{key}={value}\n"
    ENV_PATH.write_text(new_content, encoding="utf-8")
    os.environ[key] = value  # 현재 프로세스에도 즉시 반영


def _refresh_kakao_token() -> str | None:
    """refresh_token으로 새 access_token을 발급받고 .env에 저장.
    성공 시 새 access_token 반환, 실패 시 None 반환.
    """
    rest_api_key, _, refresh_token = _get_kakao_env()
    if not refresh_token or not rest_api_key:
        print("[KAKAO] refresh_token 또는 REST_API_KEY 미설정 — 갱신 불가")
        return None

    try:
        resp = requests.post(
            KAKAO_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
            data={
                "grant_type": "refresh_token",
                "client_id": rest_api_key,
                "refresh_token": refresh_token,
            },
            timeout=10,
        )
        body = resp.json()

        if resp.status_code != 200 or "access_token" not in body:
            print(f"[KAKAO] 토큰 갱신 실패: {resp.status_code} {body}")
            return None

        new_access = body["access_token"]
        _update_env("KAKAO_ACCESS_TOKEN", new_access)
        print("[KAKAO] access_token 갱신 완료")

        if "refresh_token" in body:
            _update_env("KAKAO_REFRESH_TOKEN", body["refresh_token"])
            print("[KAKAO] refresh_token도 갱신 완료")

        return new_access

    except requests.RequestException as e:
        print(f"[KAKAO] 토큰 갱신 요청 실패: {e}")
        return None


# ── 카카오톡 메시지 발송 ──────────────────────────────────────────────────────

def _do_send(access_token: str, text: str) -> requests.Response:
    """실제 카카오 API 호출 (토큰 갱신 없이 1회)."""
    template = {
        "object_type": "text",
        "text": text[:1000],
        "link": {"web_url": "https://developers.kakao.com",
                 "mobile_web_url": "https://developers.kakao.com"},
        "button_title": "확인",
    }
    return requests.post(
        KAKAO_MEMO_URL,
        headers={"Authorization": f"Bearer {access_token}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"template_object": json.dumps(template, ensure_ascii=False)},
        timeout=10,
    )


def _send_kakao(text: str) -> bool:
    """카카오톡 나에게 보내기. 401 시 토큰 자동 갱신 후 1회 재시도."""
    _, access_token, _ = _get_kakao_env()

    if not access_token:
        print(f"[KAKAO 미설정] {text[:80]}...")
        return False

    try:
        resp = _do_send(access_token, text)

        if resp.status_code == 200:
            return True

        if resp.status_code == 401:
            print("[KAKAO] 토큰 만료 — 자동 갱신 시도...")
            new_token = _refresh_kakao_token()
            if not new_token:
                print("[KAKAO] 갱신 실패 — 메시지 전송 포기")
                return False
            retry = _do_send(new_token, text)
            if retry.status_code == 200:
                print("[KAKAO] 갱신 후 재전송 성공")
                return True
            print(f"[KAKAO] 재전송 실패: {retry.status_code} {retry.text[:200]}")
            return False

        print(f"[KAKAO 오류] HTTP {resp.status_code}: {resp.text[:200]}")
        return False

    except requests.RequestException as e:
        print(f"[KAKAO 실패] {e}")
        return False


# ── 슬랙 메시지 발송 ──────────────────────────────────────────────────────────

def _send_slack(text: str) -> bool:
    if not SLACK_WEBHOOK_URL:
        print(f"[SLACK 미설정] {text[:80]}...")
        return False

    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": text},
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException as e:
        print(f"[SLACK 실패] {e}")
        return False


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="카카오/슬랙 테스트 발송")
    args = parser.parse_args()

    if args.test:
        alert = AlertSystem()
        print("카카오톡 테스트 발송 중...")
        alert.send("INFO", "✅ 퀀트 Agent 알림 테스트\n카카오톡 연결 정상")
        print("완료. 카카오톡에서 메시지를 확인하세요.")
