# gpe_core2/rule_optimizer.py
"""
Hint-driven rule-selection engine.

`select_rules(data, hints)` 는
- `data` : Python object (dict / list / scalar …)
- `hints`: Optional[Mapping[str, Any]] ; GlassBox 가 넣어준 통계 정보

반환값은 encode 단계에서 적용할 압축 Rule 리스트.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

__all__ = ["select_rules"]


# ――― rule constants (추후 utils.constants 로 이동 가능) ―――
CONST_RULE = "OP_CONST"          # 고정 상수
RANGE_RULE = "OP_RANGE"          # 숫자 범위
VEC_RANGE_RULE = "OP_VECTOR_RANGE"  # S-4 에서 구현


def _is_constant_field(values: list[Any], cardinality_hint: int | None) -> bool:
    """모든 값이 1~`cardinality_hint` 개 이내라면 CONST_RULE 적용."""
    unique_cnt = len({*values})
    if cardinality_hint is not None:
        return unique_cnt <= cardinality_hint
    return unique_cnt == 1


def select_rules(data: Any, hints: Mapping[str, Any] | None = None) -> List[str]:
    """
    Very-light heuristic v0:

    • dict → key 단위로 필드별 분석
    • list[int/float] + hints["is_vector"] → VEC_RANGE_RULE 후보
    • 숫자·bool 등 스칼라 반복 → CONST_RULE

    Returns ordered list of rule-codes to try.
    """
    rules: List[str] = []

    if hints is None:
        hints = {}

    # Vector hint
    if hints.get("is_vector"):
        rules.append(VEC_RANGE_RULE)

    # Constant hint (cardinality)
    card_hint = hints.get("cardinality")  # e.g., 3
    if isinstance(data, list):
        if _is_constant_field(data, card_hint):
            rules.append(CONST_RULE)
        else:
            rules.append(RANGE_RULE)
    elif isinstance(data, (int, float, bool, str)):
        # scalar field
        rules.append(CONST_RULE)
    elif isinstance(data, dict):
        # dict → check values collectively
        values = list(data.values())
        if _is_constant_field(values, card_hint):
            rules.append(CONST_RULE)

    # fallback
    if not rules:
        rules.append(RANGE_RULE)
    return rules
