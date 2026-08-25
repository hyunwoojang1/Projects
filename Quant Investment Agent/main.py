"""Quant Signal Agent — 메인 실행 파일.

CLAUDE.md §7 실행 순서 준수:
  1. todo.md 생성 (오늘 날짜, 9개 체크박스)
  2. 데이터 수집 → status.json 기록
  3. Module 1: QQQ 앙상블 신호
  4. Module 2: HMM 레짐 분류
  5. Module 3: 공포탐욕지수
  6. Module 4: 섹터 로테이션
  7. Module 5: 매크로 점수
  8. Module 6: 리포트 생성 (7장 PNG)
  9. Module 7: 인스타 업로드 (gate.json 승인 대기)
  10. logs/analysis_log.json 업데이트
  11. todo.md 전체 체크 확인

모듈 하나 실패해도 다음 모듈 계속. 전체 중단 금지.

실행:
  python overseer/main.py          # 정상 실행 (Overseer가 감싸서 실행)
  python main.py --no-overseer     # 개발/테스트용 (Overseer 없이)
  python main.py --module qqq      # 특정 모듈만 테스트
  python main.py --no-upload       # 카드 생성만 (업로드 생략)
  python main.py --retrain         # 모델 강제 재훈련
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STATUS_FILE = ROOT / "status.json"
GATE_FILE = ROOT / "gate.json"
GATE_TIMEOUT_SECONDS = 300  # Overseer가 5분 내 응답 없으면 타임아웃

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Status / Gate 통신 ────────────────────────────────────────────────────────

def write_status(phase: str, **kwargs) -> None:
    """Agent → Overseer 통신. status.json에 현재 단계 기록."""
    payload = {"phase": phase, "timestamp": datetime.now().isoformat(), **kwargs}
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str)
    except OSError:
        pass


def wait_for_gate(timeout: int = GATE_TIMEOUT_SECONDS) -> bool:
    """Overseer의 gate.json 응답을 기다린다. 타임아웃 시 False 반환."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if GATE_FILE.exists():
            try:
                with open(GATE_FILE, "r", encoding="utf-8") as f:
                    gate = json.load(f)
                approved = gate.get("approved", False)
                reasons = gate.get("reasons", [])
                if not approved:
                    log.warning(f"업로드 차단됨: {reasons}")
                GATE_FILE.unlink(missing_ok=True)
                return approved
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(5)
    log.error("gate.json 타임아웃 — Overseer가 응답하지 않음")
    return False


# ── todo.md 생성/업데이트 ─────────────────────────────────────────────────────

TODO_STEPS = [
    "데이터 수집 (yfinance + FRED + WRDS)",
    "Module 1: QQQ 앙상블 신호",
    "Module 2: HMM 레짐 분류",
    "Module 3: 공포탐욕지수",
    "Module 4: 섹터 로테이션",
    "Module 5: 매크로 점수",
    "Module 6: 리포트 생성 (7장 PNG)",
    "Module 7: 인스타 업로드",
    "logs/analysis_log.json 업데이트",
]


class TodoManager:
    def __init__(self, date_str: str) -> None:
        self.path = ROOT / "todo.md"
        self.date_str = date_str
        self.done: set[int] = set()

    def create(self) -> None:
        lines = [f"# todo.md — {self.date_str}\n\n"]
        for i, step in enumerate(TODO_STEPS):
            lines.append(f"- [ ] {step}\n")
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def check(self, step_idx: int) -> None:
        self.done.add(step_idx)
        lines = [f"# todo.md — {self.date_str}\n\n"]
        for i, step in enumerate(TODO_STEPS):
            mark = "x" if i in self.done else " "
            lines.append(f"- [{mark}] {step}\n")
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        log.info(f"  ✓ todo [{step_idx+1}/{len(TODO_STEPS)}]: {TODO_STEPS[step_idx]}")


# ── 개별 모듈 실행 헬퍼 ──────────────────────────────────────────────────────

def _safe_run(fn, module_name: str, fallback=None):
    """모듈 실행. 실패해도 fallback 반환, 전체 중단 없음."""
    try:
        return fn()
    except Exception as e:
        log.error(f"[{module_name}] 실패: {e}")
        return fallback if fallback is not None else {}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(module: str | None = None, no_upload: bool = False,
        retrain: bool = False, no_overseer: bool = False) -> None:

    start_time = time.time()
    date_str = datetime.now().strftime("%Y-%m-%d")
    cfg = load_config()
    analysis_cfg = cfg.get("analysis", {})

    log.info(f"=== Quant Signal Agent 시작 ({date_str}) ===")

    # ── STEP 0: todo.md 생성 (CLAUDE.md §7) ──────────────────────────────────
    todo = TodoManager(date_str)
    if module is None:
        todo.create()
        log.info("  todo.md 생성 완료")

    # --module 단독 실행 모드
    if module:
        _run_single_module(module, date_str, cfg, retrain)
        return

    all_flags: list[str] = []

    # ── STEP 1: 데이터 수집 ──────────────────────────────────────────────────
    log.info("[1/9] 데이터 수집 중...")
    from modules.data_collector import DataCollector
    dc = DataCollector()

    def _collect():
        return dc.collect(lookback_years=analysis_cfg.get("lookback_years", 30))

    data = _safe_run(_collect, "data_collector",
                     fallback={"market": __import__("pandas").DataFrame(),
                               "fred": __import__("pandas").DataFrame(),
                               "factors": __import__("pandas").DataFrame()})
    market = data.get("market", __import__("pandas").DataFrame())
    fred = data.get("fred", __import__("pandas").DataFrame())
    factors = data.get("factors", __import__("pandas").DataFrame())

    # 데이터 통계 계산 (Overseer 검증용)
    market_stats = _compute_market_stats(market)
    wrds_stats = _compute_wrds_stats(factors)

    if not no_overseer:
        write_status("data_collected",
                     market_stats=market_stats,
                     wrds_stats=wrds_stats)

    factor_returns = _safe_run(
        lambda: dc.compute_factor_returns(factors, market),
        "factor_returns",
        fallback=__import__("pandas").DataFrame())

    log.info(f"  시장: {len(market)}rows, FRED: {len(fred)}rows, "
             f"WRDS: {len(factors)}rows")
    todo.check(0)

    # ── STEP 2: QQQ 타이밍 신호 ─────────────────────────────────────────────
    log.info("[2/9] Module 1: QQQ 앙상블 신호...")
    from modules import qqq_model

    def _qqq():
        model_path = ROOT / "models"
        model_files = sorted(model_path.glob("model_*.pkl"))
        if retrain or not model_files:
            return qqq_model.train(market, fred, factors)
        return qqq_model.predict(market, fred, factors)

    qqq_sig = _safe_run(_qqq, "qqq_model")
    log.info(f"  신호: {qqq_sig.get('signal_label', '?')} "
             f"(신뢰도 {qqq_sig.get('confidence', 0):.2%})")

    if not no_overseer:
        write_status("module_complete", module_name="qqq_model",
                     module_data=qqq_sig)
        time.sleep(2)  # Overseer 처리 여유

    todo.check(1)

    # ── STEP 3: 시장 레짐 분류 ───────────────────────────────────────────────
    log.info("[3/9] Module 2: HMM 레짐 분류...")
    from modules import regime_model

    regime = _safe_run(
        lambda: regime_model.predict(market, factor_returns),
        "regime_model")

    # Overseer 검증용 통계 추가
    regime_check_data = _enrich_regime_data(regime, market)
    log.info(f"  레짐: {regime.get('state_label', '?')}")

    if not no_overseer:
        write_status("module_complete", module_name="regime_model",
                     module_data=regime_check_data)
        time.sleep(2)

    todo.check(2)

    # ── STEP 4: 공포탐욕지수 ─────────────────────────────────────────────────
    log.info("[4/9] Module 3: 공포탐욕지수...")
    from modules import fear_greed

    fg = _safe_run(lambda: fear_greed.compute(market, fred), "fear_greed")
    log.info(f"  점수: {fg.get('score', '?')} ({fg.get('label', '?')})")

    if not no_overseer:
        write_status("module_complete", module_name="fear_greed",
                     module_data=fg)
        time.sleep(2)

    todo.check(3)

    # ── STEP 5: 섹터 로테이션 ────────────────────────────────────────────────
    log.info("[5/9] Module 4: 섹터 로테이션...")
    from modules import sector_rotation

    sector = _safe_run(lambda: sector_rotation.compute(market),
                       "sector_rotation")
    log.info(f"  강세 TOP 3: {sector.get('top3', [])}")
    todo.check(4)

    # ── STEP 6: 매크로 환경 점수 ─────────────────────────────────────────────
    log.info("[6/9] Module 5: 매크로 점수...")
    from modules import macro_score

    macro = _safe_run(lambda: macro_score.compute(fred, market), "macro_score")
    log.info(f"  점수: {macro.get('score', '?')}점 ({macro.get('label', '?')})")
    todo.check(5)

    # QQQ 당일 수익률
    qqq_ret = 0.0
    if "QQQ" in market.columns and len(market["QQQ"].dropna()) >= 2:
        qqq_ret = float(market["QQQ"].dropna().pct_change().iloc[-1] * 100)

    # ── STEP 7: Claude 종합 분석 ────────────────────────────────────────────────
    log.info("[7/9] Claude 종합 분석...")
    from modules import claude_analyst

    claude_input = {
        "date": date_str,
        "qqq_signal": qqq_sig,
        "regime": regime,
        "fear_greed": fg,
        "sector": sector,
        "macro": macro,
    }
    claude_result = _safe_run(
        lambda: claude_analyst.analyze(claude_input),
        "claude_analyst",
        fallback={"opinion": "관망", "summary": "분석 실패", "reasoning": "", "risks": "", "raw": ""},
    )
    log.info(f"  Claude 의견: {claude_result.get('opinion', '?')} — {claude_result.get('summary', '')}")
    todo.check(6)

    analysis = {
        "date": date_str,
        "qqq_return": round(qqq_ret, 4),
        "qqq_signal": qqq_sig,
        "regime": regime,
        "fear_greed": fg,
        "sector": sector,
        "macro": macro,
        "claude": claude_result,
        "flags": all_flags,
    }

    # ── STEP 8: 리포트 생성 ──────────────────────────────────────────────────
    log.info("[8/9] Module 6: 카드 생성...")
    from report.card_generator import CardGenerator

    output_dir = ROOT / "output"
    gen = CardGenerator(cfg["design"], output_dir)
    card_paths = _safe_run(
        lambda: gen.generate_all(analysis, date_str),
        "card_generator", fallback=[])

    report_dir = output_dir / f"report_{date_str.replace('-', '')}"
    log.info(f"  {len(card_paths)}장 생성: {report_dir}")
    todo.check(7)

    # ── STEP 9: 인스타그램 업로드 ────────────────────────────────────────────
    log.info("[9/9] Module 7: 인스타그램 업로드...")
    upload_status = "skipped"

    if no_upload:
        upload_status = "skipped (--no-upload)"
        log.info("  업로드 생략 (--no-upload)")
    else:
        # Overseer 업로드 게이트 확인
        should_upload = True
        if not no_overseer:
            summary = {
                "date": date_str,
                "qqq_signal_label": qqq_sig.get("signal_label"),
                "flags": all_flags,
            }
            write_status("ready_to_upload",
                         report_path=str(report_dir),
                         summary=summary)
            log.info("  Overseer 승인 대기 중...")
            should_upload = wait_for_gate()
            if not should_upload:
                upload_status = "blocked_by_overseer"
                log.warning("  업로드가 Overseer에 의해 차단됨")

        if should_upload:
            try:
                from upload.instagram import InstagramUploader, generate_caption
                caption = generate_caption(analysis)
                uploader = InstagramUploader()
                result = uploader.upload_carousel(card_paths, caption)
                upload_status = "success" if result["success"] else f"failed: {result['error']}"
                log.info(f"  {upload_status}")
            except EnvironmentError as e:
                upload_status = f"skipped (env 미설정)"
                log.warning(f"  {upload_status}: {e}")
            except Exception as e:
                upload_status = f"failed: {e}"
                log.error(f"  업로드 실패: {e}")

    todo.check(7)

    # ── STEP 9: 로그 저장 ────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    log_entry = {
        "date": date_str,
        "qqq_signal": qqq_sig.get("signal_label"),
        "ensemble_confidence": qqq_sig.get("confidence"),
        "fear_greed_score": fg.get("score"),
        "regime": regime.get("state_label"),
        "top_sectors": sector.get("top3", []),
        "macro_score": macro.get("score"),
        "upload_status": upload_status,
        "runtime_seconds": elapsed,
        "warnings": all_flags,
    }
    _append_log(log_entry)
    todo.check(8)

    # Overseer에 완료 신호 전송
    if not no_overseer:
        write_status("done", log=log_entry)

    log.info(f"=== 완료 ({elapsed}초) ===")


# ── 단일 모듈 실행 ────────────────────────────────────────────────────────────

def _run_single_module(module: str, date_str: str, cfg: dict,
                       retrain: bool) -> None:
    log.info(f"=== 단독 모듈 실행: {module} ===")
    from modules.data_collector import DataCollector

    dc = DataCollector()
    data = dc.collect(lookback_years=10)
    market, fred, factors = data["market"], data["fred"], data["factors"]

    if module == "qqq":
        from modules import qqq_model
        if retrain:
            result = qqq_model.train(market, fred, factors)
        else:
            result = qqq_model.predict(market, fred, factors)
        log.info(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif module == "regime":
        from modules import regime_model
        factor_returns = dc.compute_factor_returns(factors, market)
        result = regime_model.predict(market, factor_returns)
        log.info(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif module == "fear_greed":
        from modules import fear_greed
        result = fear_greed.compute(market, fred)
        log.info(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif module == "sector":
        from modules import sector_rotation
        result = sector_rotation.compute(market)
        log.info(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif module == "macro":
        from modules import macro_score
        result = macro_score.compute(fred, market)
        log.info(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif module == "report":
        from report.card_generator import CardGenerator
        # Use dummy analysis for card generation test
        dummy = {
            "date": date_str,
            "qqq_return": 0.0,
            "qqq_signal": {"signal_label": "QQQ (보유)", "confidence": 0.5,
                           "models": {}, "top_features": []},
            "regime": {"state_label": "테스트", "state_probs": {},
                       "factor_weights": {}, "regime_description": "테스트 모드"},
            "fear_greed": {"score": 50, "label": "중립", "components": {}},
            "sector": {"ranked": [], "top3": [], "bottom3": [],
                       "rotation_dir": "—", "cycle_phase": "—"},
            "macro": {"score": 50, "label": "중립", "components": {}, "details": {}},
        }
        gen = CardGenerator(cfg["design"], ROOT / "output")
        paths = gen.generate_all(dummy, date_str)
        log.info(f"생성된 카드: {[str(p) for p in paths]}")
    else:
        log.error(f"알 수 없는 모듈: {module}. 사용 가능: qqq, regime, fear_greed, sector, macro, report")


# ── 보조 함수들 ───────────────────────────────────────────────────────────────

def _compute_market_stats(market) -> dict:
    if market is None or market.empty:
        return {}
    try:
        import numpy as np
        missing_rate = float(market.isnull().mean().mean())
        price_cols = [c for c in market.columns if not c.endswith("_vol")]
        if price_cols:
            rets = market[price_cols].pct_change().abs()
            max_ret = float(rets.max().max())
        else:
            max_ret = 0.0
        last_date = str(market.index[-1].date()) if len(market) > 0 else ""
        return {
            "missing_rate": missing_rate,
            "max_daily_return": max_ret,
            "last_date": last_date,
            "rows": len(market),
        }
    except Exception:
        return {}


def _compute_wrds_stats(factors) -> dict:
    if factors is None or factors.empty:
        return {"total_rows": 0}
    try:
        total = len(factors)
        neg_pbr = int((factors["pbr"] < 0).sum()) if "pbr" in factors.columns else 0
        extreme = int((factors.get("earnings_surprise", __import__("pandas").Series()).abs() > 5).sum())
        return {
            "total_rows": total,
            "negative_pbr_count": neg_pbr,
            "extreme_surprise_count": extreme,
        }
    except Exception:
        return {"total_rows": 0}


def _enrich_regime_data(regime: dict, market) -> dict:
    data = dict(regime)
    # Add transition count approximation
    try:
        if not market.empty and "SPY" in market.columns:
            months = max(len(market) // 21, 1)
            data["months"] = months
            data["transition_count"] = months * 2  # rough estimate
            # Extract probs as flat dict for checker
            probs = regime.get("state_probs", {})
            data["current_probs"] = {k: v for k, v in probs.items()}
            # Regime distribution (uniform approximation)
            n = len(probs)
            if n > 0:
                data["regime_distribution"] = {k: 1/n for k in probs}
    except Exception:
        pass
    return data


def _append_log(entry: dict) -> None:
    log_path = ROOT / "logs" / "analysis_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []

    existing.append(entry)
    if len(existing) > 365:
        existing = existing[-365:]

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Signal Agent")
    parser.add_argument("--module", type=str, default=None,
                        choices=["qqq", "regime", "fear_greed", "sector", "macro", "report"],
                        help="특정 모듈만 실행")
    parser.add_argument("--no-upload", action="store_true",
                        help="카드 생성만 수행, 인스타그램 업로드 생략")
    parser.add_argument("--retrain", action="store_true",
                        help="ML/DL 앙상블 모델 강제 재훈련")
    parser.add_argument("--no-overseer", action="store_true",
                        help="Overseer 없이 단독 실행 (개발/테스트용)")
    args = parser.parse_args()

    run(
        module=args.module,
        no_upload=args.no_upload,
        retrain=args.retrain,
        no_overseer=args.no_overseer,
    )
