"""스코어러 기본 인터페이스."""

from abc import ABC, abstractmethod


class BaseScorer(ABC):
    @abstractmethod
    def score(self, raw_values: dict[str, float], as_of_date: str) -> dict[str, float]:
        """원시 지표값을 받아 각 지표의 [0, 100] 점수 딕셔너리 반환."""
        ...
