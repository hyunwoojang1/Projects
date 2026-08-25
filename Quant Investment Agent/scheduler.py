"""매일 오전 8시 자동 실행 스크립트.

Windows Task Scheduler가 이 파일을 호출합니다.
분석 → Claude 의견 → 이메일 발송까지 전부 자동.

등록 방법:
  python scheduler.py --register   # Task Scheduler에 등록
  python scheduler.py              # 수동 즉시 실행 (테스트)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "scheduler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def run() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log.info(f"=== 자동 분석 시작: {date_str} ===")
    t0 = time.time()

    # ── 1. 데이터 수집 ──────────────────────────────────────────────────────────
    log.info("[1/7] 데이터 수집...")
    from modules.data_collector import DataCollector
    dc      = DataCollector()
    data    = dc.collect(lookback_years=30)
    market  = data["market"]
    fred    = data["fred"]
    factors = data["factors"]
    factor_returns = dc.compute_factor_returns(factors, market)
    log.info(f"  시장 {len(market)}행, FRED {len(fred)}행")

    # ── 2. QQQ 타이밍 신호 ──────────────────────────────────────────────────────
    log.info("[2/7] QQQ 신호 분석...")
    from modules import qqq_model
    qqq_sig = _safe(lambda: qqq_model.predict(market, fred, factors), "qqq_model")
    log.info(f"  → {qqq_sig.get('signal_label','?')} ({qqq_sig.get('confidence',0):.1%})")

    # ── 3. 시장 국면 ────────────────────────────────────────────────────────────
    log.info("[3/7] HMM 국면 분류...")
    from modules import regime_model
    regime = _safe(lambda: regime_model.predict(market, factor_returns), "regime")
    log.info(f"  → {regime.get('state_label','?')}")

    # ── 4. 공포탐욕 ─────────────────────────────────────────────────────────────
    log.info("[4/7] 공포탐욕지수...")
    from modules import fear_greed
    fg = _safe(lambda: fear_greed.compute(market, fred), "fear_greed")
    log.info(f"  → {fg.get('score','?')} ({fg.get('label','?')})")

    # ── 5. 섹터 / 매크로 ────────────────────────────────────────────────────────
    log.info("[5/7] 섹터 · 매크로...")
    from modules import sector_rotation, macro_score
    sector = _safe(lambda: sector_rotation.compute(market), "sector")
    macro  = _safe(lambda: macro_score.compute(fred, market), "macro")
    log.info(f"  섹터 TOP3: {sector.get('top3',[])}  매크로: {macro.get('score','?')}점")

    # ── 6. Claude 종합 분석 ──────────────────────────────────────────────────────
    log.info("[6/7] Claude 종합 분석...")
    from modules import claude_analyst
    signals = {
        "date": date_str,
        "qqq_signal": qqq_sig,
        "regime":     regime,
        "fear_greed": fg,
        "sector":     sector,
        "macro":      macro,
    }
    claude_result = _safe(lambda: claude_analyst.analyze(signals), "claude_analyst",
                          fallback={"opinion": "관망", "summary": "분석 실패", "raw": ""})
    log.info(f"  → Claude: {claude_result.get('opinion','?')} — {claude_result.get('summary','')}")

    # ── 7. 카드 생성 + 이메일 발송 ──────────────────────────────────────────────
    log.info("[7/7] 카드 생성 + 이메일 발송...")
    analysis = {
        "date":       date_str,
        "qqq_signal": qqq_sig,
        "regime":     regime,
        "fear_greed": fg,
        "sector":     sector,
        "macro":      macro,
        "claude":     claude_result,
    }

    # 카드 생성
    try:
        import yaml
        from report.card_generator import CardGenerator
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        gen = CardGenerator(cfg.get("design", {}), ROOT / "output")
        cards = gen.generate_all(analysis, date_str)
        log.info(f"  카드 {len(cards)}장 생성")
    except Exception as e:
        log.warning(f"  카드 생성 실패: {e}")

    # 카카오톡 알림
    from overseer.alerts import AlertSystem
    alert = AlertSystem()
    _safe(lambda: alert.notify_analysis(analysis), "kakao_alert")
    log.info("  카카오톡 알림 발송 완료")

    elapsed = round(time.time() - t0, 1)
    log.info(f"=== 완료 ({elapsed}초) ===")


def _safe(fn, name: str, fallback: dict | None = None):
    try:
        return fn()
    except Exception as e:
        log.warning(f"  [{name}] 실패: {e}")
        return fallback or {}


# ── Windows Task Scheduler 등록 ────────────────────────────────────────────────

def register_task() -> None:
    python_exe = sys.executable
    script     = str(ROOT / "scheduler.py")

    cmd = [
        "schtasks", "/create",
        "/tn",  "Quant_Investment_Agent",
        "/tr",  f'"{python_exe}" "{script}"',
        "/sc",  "DAILY",
        "/st",  "08:00",
        "/rl",  "HIGHEST",
        "/f",                        # 이미 있으면 덮어씀
    ]

    log.info("Task Scheduler 등록 중...")
    log.info(f"  명령어: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log.info("  ✓ 등록 완료 — 매일 오전 08:00 자동 실행")
        log.info("  확인: 작업 스케줄러 → '퀀트 투자 에이전트' 검색")
    else:
        log.error(f"  등록 실패: {result.stderr}")
        log.info("  수동 등록 방법:")
        log.info(f"    1. 작업 스케줄러 열기")
        log.info(f"    2. 새 작업 만들기")
        log.info(f"    3. 프로그램: {python_exe}")
        log.info(f"    4. 인수:     {script}")
        log.info(f"    5. 트리거:   매일 08:00")


if __name__ == "__main__":
    if "--register" in sys.argv:
        register_task()
    else:
        run()
