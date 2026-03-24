"""환경 변수 기반 중앙 설정 로더."""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()


def _get(key: str, default: str | None = None) -> str:
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"필수 환경 변수 누락: {key}. .env 파일을 확인하세요.")
    return value


def _get_optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _get_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


# ── API 자격증명 ──────────────────────────────────────────────────────────────
FRED_API_KEY: str = _get("FRED_API_KEY")
# WRDS는 기관 구독 필요 — 없으면 펀더멘탈 스코어 생략
WRDS_USERNAME: str = _get_optional("WRDS_USERNAME", "")
WRDS_PASSWORD: str = _get_optional("WRDS_PASSWORD", "")

# ── 가격 데이터 소스 ──────────────────────────────────────────────────────────
PRICE_DATA_SOURCE: str = _get_optional("PRICE_DATA_SOURCE", "yfinance")
PRICE_DATA_API_KEY: str = _get_optional("PRICE_DATA_API_KEY", "")

# ── 캐시 ─────────────────────────────────────────────────────────────────────
CACHE_DIR: Path = Path(_get_optional("CACHE_DIR", "./data/.cache"))
CACHE_TTL_HOURS_FRED: int = _get_int("CACHE_TTL_HOURS_FRED", 24)
CACHE_TTL_HOURS_WRDS: int = _get_int("CACHE_TTL_HOURS_WRDS", 168)
CACHE_TTL_HOURS_TECHNICAL: int = _get_int("CACHE_TTL_HOURS_TECHNICAL", 4)

# ── 출력 ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path(_get_optional("OUTPUT_DIR", "./output"))
OUTPUT_FORMATS: list[str] = _get_optional("OUTPUT_FORMAT", "json,html").split(",")

# ── 정규화 ───────────────────────────────────────────────────────────────────
NORMALIZATION_WINDOW_YEARS: int = _get_int("NORMALIZATION_WINDOW_YEARS", 10)

# ── 시그널 임계값 ─────────────────────────────────────────────────────────────
SIGNAL_THRESHOLD_STRONG_BUY: float = _get_float("SIGNAL_THRESHOLD_STRONG_BUY", 70.0)
SIGNAL_THRESHOLD_BUY: float = _get_float("SIGNAL_THRESHOLD_BUY", 55.0)
SIGNAL_THRESHOLD_NEUTRAL: float = _get_float("SIGNAL_THRESHOLD_NEUTRAL", 45.0)
SIGNAL_THRESHOLD_SELL: float = _get_float("SIGNAL_THRESHOLD_SELL", 30.0)

# ── 로깅 ─────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get_optional("LOG_LEVEL", "INFO")
LOG_FORMAT: str = _get_optional("LOG_FORMAT", "text")
