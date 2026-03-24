"""BaseHorizon 인터페이스 및 HorizonResult 데이터클래스."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class HorizonResult:
    horizon: str                                      # "short" | "mid" | "long"
    entry_score: float                                # [0, 100]
    signal: str                                       # STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL
    resolution: str = ""                              # "daily" | "weekly" | "monthly"
    group_scores: dict[str, float] = field(default_factory=dict)
    indicator_scores: dict[str, float] = field(default_factory=dict)
    missing_indicators: list[str] = field(default_factory=list)
    as_of_date: str = ""
    weight_version: str = ""


def classify_signal(
    score: float,
    threshold_strong_buy: float = 70.0,
    threshold_buy: float = 55.0,
    threshold_neutral: float = 45.0,
    threshold_sell: float = 30.0,
) -> str:
    if score >= threshold_strong_buy:
        return "STRONG_BUY"
    if score >= threshold_buy:
        return "BUY"
    if score >= threshold_neutral:
        return "NEUTRAL"
    if score >= threshold_sell:
        return "SELL"
    return "STRONG_SELL"


class BaseHorizon(ABC):
    @abstractmethod
    def compute(
        self,
        macro_scores: dict[str, float],
        fundamental_scores: dict[str, float],
        technical_scores: dict[str, float],
        as_of_date: str = "",
    ) -> HorizonResult:
        ...
