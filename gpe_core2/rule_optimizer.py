# gpe_core2/rule_optimizer.py
"""
Hint-driven + pattern-scanning Rule Optimizer  (v2-lite)

▪ GlassBox 가 넘겨준 hints 로 “빠른 결정”
▪ 부족할 경우 v1 휴리스틱으로 패턴 탐색
▪ 범위: Constant / Numeric Range / VectorRange  (S-4 확장 예정)
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, List

__all__ = ["select_rules", "RuleOptimizerLite"]

# ── Rule opcode 상수 ──────────────────────────────────────────────────────────
OP_CONST = "OP_CONST"                # 동일 값 반복
OP_RANGE = "OP_RANGE"                # 연속 숫자
OP_VEC_RANGE = "OP_VECTOR_RANGE"     # 연속·근접 벡터(S-4)

# ── 내부 유틸 ─────────────────────────────────────────────────────────────────
def _unique_count(seq: Sequence[Any]) -> int:
    try:
        return len(set(seq))
    except TypeError:
        # 리스트·딕셔너리처럼 unhashable 값이 섞인 경우
        return len({repr(v) for v in seq})


def _is_const(seq: Sequence[Any], threshold: int) -> bool:
    return _unique_count(seq) <= threshold


def _is_numeric_range(seq: Sequence[Any], min_len: int = 5) -> bool:
    """strict ↑↓ 1 step range 검출"""
    if len(seq) < min_len or not all(isinstance(v, (int, float)) for v in seq):
        return False
    start = seq[0]
    for i, v in enumerate(seq[1:], start=1):
        if v != start + i:
            return False
    return True


# ── 핵심 Optimizer 클래스 ────────────────────────────────────────────────────
class RuleOptimizerLite:
    def __init__(
        self,
        constant_threshold: int = 3,
        range_min_length: int = 5,
    ) -> None:
        self.constant_threshold = constant_threshold
        self.range_min_length = range_min_length

    # public -----------------------------------------------------------------
    def select_rules(
        self,
        data: Any,
        hints: Mapping[str, Any] | None = None,
    ) -> List[str]:
        """
        1) 힌트 기반 빠른 판단
        2) dict / list 패턴 스캔
        3) fallback: OP_RANGE
        """
        if hints is None:
            hints = {}

        # ── 1. 힌트 기반 ────────────────────────────────────
        if hints.get("is_vector"):
            return [OP_VEC_RANGE]

        if card := hints.get("cardinality"):
            if card <= self.constant_threshold:
                return [OP_CONST]

        if hints.get("range_span") and hints["range_span"] >= self.range_min_length:
            return [OP_RANGE]

        # ── 2. 데이터 패턴 스캔 ─────────────────────────────
        rules: List[str] = []
        if isinstance(data, list):
            if _is_const(data, self.constant_threshold):
                rules.append(OP_CONST)
            elif _is_numeric_range(data, self.range_min_length):
                rules.append(OP_RANGE)

        elif isinstance(data, dict):
            if _is_const(list(data.values()), self.constant_threshold):
                rules.append(OP_CONST)

        elif isinstance(data, (int, float, bool, str)):
            rules.append(OP_CONST)

        # ── 3. fallback ───────────────────────────────────
        if not rules:
            rules.append(OP_RANGE)

        return rules


# ── procedural facade (v1 호환) ───────────────────────────────────────────────
_optimizer_singleton = RuleOptimizerLite()


def select_rules(
    data: Any,
    hints: Mapping[str, Any] | None = None,
) -> List[str]:
    """Functional wrapper → 기존 encode() 코드와 인터페이스 동일."""
    return _optimizer_singleton.select_rules(data, hints)
