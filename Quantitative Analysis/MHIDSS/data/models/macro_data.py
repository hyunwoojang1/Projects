"""매크로 데이터 스냅샷 모델."""

from dataclasses import dataclass, field


@dataclass
class MacroSnapshot:
    as_of_date: str
    values: dict[str, float] = field(default_factory=dict)
    source_dates: dict[str, str] = field(default_factory=dict)  # 지표별 실제 데이터 날짜
