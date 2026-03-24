"""펀더멘탈 데이터 스냅샷 모델."""

from dataclasses import dataclass, field


@dataclass
class FundamentalSnapshot:
    as_of_date: str
    universe: str = "SP500"
    values: dict[str, float] = field(default_factory=dict)  # 시장 중위값
    source_dates: dict[str, str] = field(default_factory=dict)
