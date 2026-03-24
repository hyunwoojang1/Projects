"""Long-Term Horizon (6-24개월): Macro 50% + Fundamental 45% 가중."""

from config.weights import (
    FUNDAMENTAL_INDICATOR_WEIGHTS,
    HORIZON_GROUP_WEIGHTS,
    MACRO_INDICATOR_WEIGHTS,
    TECHNICAL_INDICATOR_WEIGHTS,
    WEIGHT_VERSION,
)

from .base import BaseHorizon, HorizonResult, classify_signal
from .short_term import _group_score

_HORIZON = "long"


class LongTermHorizon(BaseHorizon):
    def compute(
        self,
        macro_scores: dict[str, float],
        fundamental_scores: dict[str, float],
        technical_scores: dict[str, float],
        as_of_date: str = "",
    ) -> HorizonResult:
        gw = HORIZON_GROUP_WEIGHTS[_HORIZON]

        g_m = _group_score(macro_scores, MACRO_INDICATOR_WEIGHTS[_HORIZON])
        g_f = _group_score(fundamental_scores, FUNDAMENTAL_INDICATOR_WEIGHTS[_HORIZON])
        g_t = _group_score(technical_scores, TECHNICAL_INDICATOR_WEIGHTS[_HORIZON])

        entry_score = gw["macro"] * g_m + gw["fundamental"] * g_f + gw["technical"] * g_t
        all_scores = {**macro_scores, **fundamental_scores, **technical_scores}
        all_keys = (
            list(MACRO_INDICATOR_WEIGHTS[_HORIZON])
            + list(FUNDAMENTAL_INDICATOR_WEIGHTS[_HORIZON])
            + list(TECHNICAL_INDICATOR_WEIGHTS[_HORIZON])
        )
        missing = [k for k in all_keys if all_scores.get(k) is None]

        return HorizonResult(
            horizon=_HORIZON,
            entry_score=round(entry_score, 2),
            signal=classify_signal(entry_score),
            resolution="monthly",
            group_scores={"macro": round(g_m, 2), "fundamental": round(g_f, 2), "technical": round(g_t, 2)},
            indicator_scores={k: round(v, 2) for k, v in all_scores.items() if v is not None},
            missing_indicators=missing,
            as_of_date=as_of_date,
            weight_version=WEIGHT_VERSION,
        )
