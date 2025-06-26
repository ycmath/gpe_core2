# gpe_core2/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# ──────────────────────────────────────────────────────────────
# 1. AST
# ──────────────────────────────────────────────────────────────
@dataclass
class ASTNode:
    instance_id: str
    class_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)


 ── Rule base & 파생 ──────────────────────────────────
@dataclass
class BaseRule:
    opcode: str                               # ← non-default
    params: Dict[str, Any] = field(default_factory=dict)
    # class_name 필드 ❌없음❌  << 반드시 제거되어야 함


# -- 2-1. AST 조작 규칙 --------------------------------------------------------
@dataclass
class InstantiateRule(BaseRule):
    """
    새 노드 인스턴스 생성 규칙
    """
    class_name: str           # non-default
    instance_id: str          # non-default
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppendChildRule(BaseRule):
    parent_id: str
    child_id: str
    attribute_name: str = "children"


@dataclass
class RepeatRule(BaseRule):
    count: int
    instruction: BaseRule


# -- 2-2. 최적화 규칙 ----------------------------------------------------------
@dataclass
class ConstantRule(BaseRule):
    value_id: str
    value: Any
    references: List[str] = field(default_factory=list)

    # opcode 고정값 – BaseRule(opcode, params)를 오버라이드
    opcode: str = field(init=False, default="CONSTANT")


@dataclass
class RangeRule(BaseRule):
    instance_ids: List[str]
    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float] = 1

    opcode: str = field(init=False, default="RANGE")


@dataclass
class TemplateRule(BaseRule):
    template_id: str
    structure: Dict[str, Any]
    variable_keys: List[str]
    instances: List[Dict[str, Any]]

    opcode: str = field(init=False, default="TEMPLATE")


@dataclass
class CompactListRule(BaseRule):
    parent_id: str
    length: int
    default_value: Any
    exceptions: List[Tuple[int, Any]] = field(default_factory=list)

    opcode: str = field(init=False, default="COMPACT_LIST")


# ──────────────────────────────────────────────────────────────
# 3. Attention Seed (간이 정의)
# ──────────────────────────────────────────────────────────────
@dataclass
class AttentionSeed:
    rules: List[BaseRule]


# ──────────────────────────────────────────────────────────────
# 4. GPE Payload
# ──────────────────────────────────────────────────────────────
@dataclass
class GpePayload:
    """
    통합 v2 Payload
    • generative_payload : 실제 전송 바이트
    • payload_type       : "json" / "vector" / …
    • rules              : 적용된 Rule 목록
    • metadata           : 캐싱 여부 등
    • fallback_payload   : 레거시 JSON
    """
    generative_payload: bytes                                # non-default

    # optional / meta
    payload_type: str = "json"
    rules: List[BaseRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_payload: Optional[Dict[str, Any]] = None
