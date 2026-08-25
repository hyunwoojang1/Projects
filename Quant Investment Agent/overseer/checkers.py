"""Overseer 검증 클래스 — OVERSEER.md 명세 구현.

Agent와 독립된 프로세스에서 status.json을 읽어 검증한다.
Agent 코드를 직접 호출하거나 수정하지 않는다.

플래그 규칙:
  ⚠️ 경고  — 로그만 기록, 실행 계속
  🚨 치명적 — 즉시 전체 중단 + 카카오톡(메인) + 슬랙(보조)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path


# ── Data Quality Checker ──────────────────────────────────────────────────────

class DataQualityChecker:
    """Check 1: 데이터 수집 직후 품질 검증."""

    MISSING_RATE_WARN = 0.10       # 10% 초과 → 경고
    DAILY_RETURN_WARN = 0.50       # 50% 초과 → 경고 (스플릿 의심)
    MAX_DATE_LAG_DAYS = 3          # 3일 이상 지연 → 경고
    MAX_ZERO_VOLUME_DAYS = 5
    NEGATIVE_PBR_RATIO = 0.05      # 5% 초과 → 경고
    EXTREME_SURPRISE_RATIO = 0.01  # 1% 초과 → 경고

    def check_all(self, status: dict) -> list[str]:
        flags: list[str] = []
        flags.extend(self._check_market(status.get("market_stats", {})))
        flags.extend(self._check_wrds(status.get("wrds_stats", {})))
        return flags

    def _check_market(self, stats: dict) -> list[str]:
        flags = []
        if not stats:
            return flags

        missing_rate = stats.get("missing_rate", 0.0)
        if missing_rate > self.MISSING_RATE_WARN:
            flags.append(f"⚠️ 가격 데이터 결측 {missing_rate:.1%} — 해당 날짜 제외 후 계속")

        max_ret = stats.get("max_daily_return", 0.0)
        if max_ret > self.DAILY_RETURN_WARN:
            flags.append(f"⚠️ 일일 수익률 극단값 {max_ret:.1%} — 스플릿/배당 의심")

        last_date_str = stats.get("last_date", "")
        if last_date_str:
            try:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                lag = (datetime.now() - last_date).days
                if lag > self.MAX_DATE_LAG_DAYS:
                    flags.append(f"⚠️ 데이터 최신성 문제 — 마지막 날짜 {last_date_str} (API 장애 의심)")
            except ValueError:
                pass

        zero_vol = stats.get("zero_volume_days", 0)
        if zero_vol > self.MAX_ZERO_VOLUME_DAYS:
            flags.append(f"⚠️ 거래량 0 발생 {zero_vol}일 — 데이터 확인 필요")

        return flags

    def _check_wrds(self, stats: dict) -> list[str]:
        flags = []
        if not stats:
            return flags

        total = stats.get("total_rows", 0)
        if total == 0:
            return flags  # WRDS 미설정 시 정상

        neg_pbr = stats.get("negative_pbr_count", 0)
        if total > 0 and neg_pbr / total > self.NEGATIVE_PBR_RATIO:
            flags.append(f"⚠️ PBR 음수 {neg_pbr}개 ({neg_pbr/total:.1%}) — 자본잠식 기업 필터링 확인")

        extreme = stats.get("extreme_surprise_count", 0)
        if total > 0 and extreme / total > self.EXTREME_SURPRISE_RATIO:
            flags.append(f"⚠️ 어닝서프라이즈 극단값 {extreme}개 — 데이터 오류 가능성")

        return flags


# ── Model Sanity Checker ──────────────────────────────────────────────────────

class ModelSanityChecker:
    """Check 2-1: QQQ 앙상블 모델 결과 검증."""

    LIMITS = {
        "max_sharpe": 3.0,
        "max_annual_return": 200.0,   # % — look-ahead bias 의심 기준
        "max_confidence": 0.95,
        "max_tqqq_ratio": 0.70,
    }

    def check(self, model_output: dict) -> list[str]:
        flags = []

        sharpe = model_output.get("strategy_sharpe", 0.0)
        if sharpe and sharpe > self.LIMITS["max_sharpe"]:
            flags.append(f"⚠️ 샤프지수 {sharpe:.2f} — 과적합 의심")

        # Annual return estimation from backtest
        mdd = abs(model_output.get("strategy_mdd", 0.0))
        if mdd < 0.02 and sharpe > 2.0:
            flags.append("⚠️ MDD < 2% with Sharpe > 2 — look-ahead bias 가능성")

        confidence = model_output.get("confidence", 0.0)
        if confidence and confidence > self.LIMITS["max_confidence"]:
            flags.append(f"⚠️ 앙상블 신뢰도 {confidence:.2f} — 과적합 의심")

        # Detect if all 4 models agree identically (diversity check)
        models_dict = model_output.get("models", {})
        if models_dict:
            unique_preds = set(models_dict.values())
            if len(unique_preds) == 1:
                flags.append("⚠️ 4개 모델 전부 동일 신호 — 다양성 부족 (과적합 가능)")

        return flags


# ── Regime Sanity Checker ─────────────────────────────────────────────────────

class RegimeSanityChecker:
    """Check 2-2: HMM 레짐 분류 결과 검증."""

    MAX_TRANSITIONS_PER_MONTH = 8
    MAX_REGIME_RATIO = 0.90
    PROB_SUM_TOLERANCE = 0.01

    def check(self, hmm_output: dict) -> list[str]:
        flags = []

        # 레짐 전환 빈도
        t_count = hmm_output.get("transition_count", 0)
        months = hmm_output.get("months", 1)
        tpm = t_count / months if months > 0 else 0
        if tpm > self.MAX_TRANSITIONS_PER_MONTH:
            flags.append(f"⚠️ 레짐 전환 과다 {tpm:.1f}회/월 — HMM 상태 수 재검토 필요")

        # 레짐 분포 극단 편향
        dist = hmm_output.get("regime_distribution", {})
        if dist:
            max_ratio = max(dist.values())
            if max_ratio > self.MAX_REGIME_RATIO:
                flags.append(f"⚠️ 레짐 분포 극단 편향 ({max_ratio:.0%}) — 데이터 확인 필요")

        # 확률 합 검증 (코드 버그 탐지)
        probs = hmm_output.get("current_probs", {})
        if probs:
            prob_sum = sum(probs.values())
            if abs(prob_sum - 1.0) > self.PROB_SUM_TOLERANCE:
                flags.append(f"🚨 확률 합 오류: {prob_sum:.4f} ≠ 1.0 — 코드 버그")

        return flags


# ── Fear & Greed Checker ──────────────────────────────────────────────────────

class FearGreedChecker:
    """Check 2-3: 공포탐욕지수 이상치 검증."""

    MAX_DAILY_CHANGE = 30

    def check(self, fg_output: dict, yesterday_score: int | None) -> list[str]:
        flags = []

        score = fg_output.get("score", 50)

        if yesterday_score is not None:
            daily_change = abs(score - yesterday_score)
            if daily_change > self.MAX_DAILY_CHANGE:
                flags.append(
                    f"⚠️ 공포탐욕지수 급변: {daily_change:.0f}점 "
                    f"({yesterday_score} → {score}) — 데이터 확인 필요")

        components = fg_output.get("components", {})
        missing = [k for k, v in components.items() if v is None]
        if missing:
            flags.append(f"⚠️ 누락 구성지표: {missing}")

        return flags


# ── Upload Gate ───────────────────────────────────────────────────────────────

class UploadGate:
    """Check 3: 업로드 최종 승인 관문.

    하나라도 실패하면 업로드 차단 + 카카오톡 알림.
    """

    MIN_IMAGE_SIZE_BYTES = 100_000  # 100KB
    EXPECTED_CARD_COUNT = 7

    def approve(self, report_path: str,
                summary: dict) -> tuple[bool, list[str]]:
        gates = [
            self._check_images_exist(report_path),
            self._check_image_size(report_path),
            self._check_no_critical_flags(summary),
            self._check_disclaimer_present(report_path),
            self._check_date_correct(report_path, summary),
            self._check_signal_not_empty(summary),
        ]

        failures = [reason for ok, reason in gates if not ok]

        if failures:
            return False, failures
        return True, []

    def _check_images_exist(self, path: str) -> tuple[bool, str]:
        for i in range(1, self.EXPECTED_CARD_COUNT + 1):
            p = Path(path) / f"card_0{i}.png"
            if not p.exists():
                return False, f"card_0{i}.png 없음"
        return True, ""

    def _check_image_size(self, path: str) -> tuple[bool, str]:
        for i in range(1, self.EXPECTED_CARD_COUNT + 1):
            p = Path(path) / f"card_0{i}.png"
            if not p.exists():
                return False, f"card_0{i}.png 없음"
            size = p.stat().st_size
            if size < self.MIN_IMAGE_SIZE_BYTES:
                return False, f"card_0{i}.png 크기 비정상 ({size:,} bytes < 100KB)"
        return True, ""

    def _check_no_critical_flags(self, summary: dict) -> tuple[bool, str]:
        critical = [f for f in summary.get("flags", []) if f.startswith("🚨")]
        if critical:
            return False, f"치명적 플래그 존재: {critical}"
        return True, ""

    def _check_disclaimer_present(self, path: str) -> tuple[bool, str]:
        # card_07 is the disclaimer card
        p = Path(path) / "card_07.png"
        if not p.exists():
            return False, "면책 고지 카드(card_07.png) 없음"
        return True, ""

    def _check_date_correct(self, path: str,
                             summary: dict) -> tuple[bool, str]:
        today = datetime.now().strftime("%Y%m%d")
        report_dir = Path(path).name  # e.g., "report_20260502"
        if today not in report_dir:
            return False, f"날짜 불일치: 폴더={report_dir}, 오늘={today}"

        summary_date = summary.get("date", "").replace("-", "")
        if summary_date and summary_date != today:
            return False, f"분석 날짜 불일치: summary={summary_date}, 오늘={today}"

        return True, ""

    def _check_signal_not_empty(self, summary: dict) -> tuple[bool, str]:
        signal = summary.get("qqq_signal_label")
        if not signal:
            return False, "QQQ 신호가 None — 분석 실패"
        return True, ""
