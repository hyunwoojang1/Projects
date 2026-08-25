"""Overseer 메인 — Agent와 완전히 독립된 감독관 프로세스.

Agent를 서브프로세스로 실행하고 status.json을 폴링하여 각 단계를 검증.
치명적 플래그 감지 시 Agent 프로세스를 즉시 종료하고 카카오톡 + 슬랙으로 알림.

실행:
  python overseer/main.py                   # 전체 실행 (Agent 감싸서)
  python overseer/main.py --dry-run         # 검증만 (업로드 차단)

통신 파일:
  status.json  ← Agent가 기록, Overseer가 읽음
  gate.json    ← Overseer가 기록, Agent가 읽음
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = ROOT / "status.json"
GATE_FILE = ROOT / "gate.json"

POLL_INTERVAL_SECONDS = 10
MAX_WAIT_MINUTES = 60  # Agent 전체 실행 최대 시간


def main(dry_run: bool = False) -> None:
    from overseer.alerts import AlertSystem
    from overseer.checkers import (
        DataQualityChecker, FearGreedChecker, ModelSanityChecker,
        RegimeSanityChecker, UploadGate,
    )

    alert = AlertSystem()
    date_str = datetime.now().strftime("%Y-%m-%d")

    # 이전 통신 파일 초기화
    STATUS_FILE.unlink(missing_ok=True)
    GATE_FILE.unlink(missing_ok=True)

    alert.notify_start(date_str)

    # Agent 서브프로세스 실행
    agent_cmd = [sys.executable, str(ROOT / "main.py")]
    if dry_run:
        agent_cmd.append("--no-upload")
    print(f"[OVERSEER] Agent 실행: {' '.join(agent_cmd)}")

    agent_proc = subprocess.Popen(agent_cmd, cwd=str(ROOT))

    warnings_log: list[str] = []
    start_time = time.time()
    yesterday_fg_score: int | None = _load_yesterday_fg_score()

    def terminate(module: str, flags: list[str]) -> None:
        print(f"[OVERSEER] 🚨 전체 중단 — 원인: {module}")
        agent_proc.terminate()
        alert.notify_critical_halt(module, flags)

    try:
        while True:
            # 타임아웃 체크
            elapsed = time.time() - start_time
            if elapsed > MAX_WAIT_MINUTES * 60:
                terminate("타임아웃", [f"Agent가 {MAX_WAIT_MINUTES}분 안에 완료되지 않음"])
                return

            # Agent가 이미 종료되었는지 확인
            rc = agent_proc.poll()
            if rc is not None and not STATUS_FILE.exists():
                print(f"[OVERSEER] Agent 비정상 종료 (rc={rc})")
                alert.notify_critical_halt("Agent 프로세스", [f"비정상 종료 코드: {rc}"])
                return

            status = _read_json(STATUS_FILE)
            if not status:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            phase = status.get("phase", "")
            print(f"[OVERSEER] Phase: {phase}")

            # ── Check 1: 데이터 수집 완료 ──────────────────────────────────────
            if phase == "data_collected":
                flags = DataQualityChecker().check_all(status)
                critical = [f for f in flags if f.startswith("🚨")]
                warnings = [f for f in flags if f.startswith("⚠️")]

                if warnings:
                    warnings_log.extend(warnings)
                    alert.notify_warning("데이터 수집", warnings)

                if critical:
                    terminate("데이터 수집", critical)
                    return

                _clear_status()

            # ── Check 2: 모듈 완료 ────────────────────────────────────────────
            elif phase == "module_complete":
                module_name = status.get("module_name", "?")
                module_flags = _check_module(
                    module_name, status,
                    ModelSanityChecker(), RegimeSanityChecker(),
                    FearGreedChecker(), yesterday_fg_score)

                critical = [f for f in module_flags if f.startswith("🚨")]
                warnings = [f for f in module_flags if f.startswith("⚠️")]

                if warnings:
                    warnings_log.extend(warnings)
                    alert.notify_warning(module_name, warnings)

                if critical:
                    terminate(module_name, critical)
                    return

                _clear_status()

            # ── Check 3: 업로드 게이트 ────────────────────────────────────────
            elif phase == "ready_to_upload":
                report_path = status.get("report_path", "")
                summary = status.get("summary", {})

                if dry_run:
                    print("[OVERSEER] --dry-run: 업로드 차단")
                    _write_gate(approved=False, reasons=["dry-run 모드"])
                    _clear_status()
                    continue

                approved, reasons = UploadGate().approve(report_path, summary)
                _write_gate(approved=approved, reasons=reasons)

                if not approved:
                    alert.notify_upload_blocked(reasons)
                    # Agent가 gate.json 읽고 스스로 종료할 것 — 기다림
                    time.sleep(30)
                    if agent_proc.poll() is None:
                        agent_proc.terminate()
                    return

                _clear_status()

            # ── Done ──────────────────────────────────────────────────────────
            elif phase == "done":
                log = status.get("log", {})
                log["warnings"] = warnings_log

                # 업로드 성공 알림
                if log.get("upload_status") == "success":
                    alert.notify_upload_success(
                        fg_score=log.get("fear_greed_score", 0),
                        signal=log.get("qqq_signal", "?"))

                alert.notify_daily_summary(log)
                print(f"[OVERSEER] ✅ 완료 ({time.time() - start_time:.0f}초)")
                return

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("[OVERSEER] 수동 중단")
        agent_proc.terminate()


# ── Module-specific checks ────────────────────────────────────────────────────

def _check_module(module_name: str, status: dict,
                  model_checker: "ModelSanityChecker",
                  regime_checker: "RegimeSanityChecker",
                  fg_checker: "FearGreedChecker",
                  yesterday_fg: int | None) -> list[str]:
    data = status.get("module_data", {})

    if module_name == "qqq_model":
        return model_checker.check(data)
    if module_name == "regime_model":
        return regime_checker.check(data)
    if module_name == "fear_greed":
        return fg_checker.check(data, yesterday_fg)
    return []


# ── File utilities ────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_gate(approved: bool, reasons: list[str]) -> None:
    with open(GATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"approved": approved, "reasons": reasons}, f, ensure_ascii=False)
    print(f"[OVERSEER] gate.json 기록: approved={approved}, reasons={reasons}")


def _clear_status() -> None:
    """Status를 neutral로 초기화해 다음 단계를 기다린다."""
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump({"phase": "waiting"}, f)
    except OSError:
        pass


def _load_yesterday_fg_score() -> int | None:
    log_path = ROOT / "logs" / "analysis_log.json"
    if not log_path.exists():
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
        if isinstance(logs, list) and logs:
            return logs[-1].get("fear_greed_score")
    except Exception:
        pass
    return None


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Quant Signal Overseer")
    parser.add_argument("--dry-run", action="store_true",
                        help="업로드 없이 검증만 수행")
    args = parser.parse_args()

    main(dry_run=args.dry_run)
