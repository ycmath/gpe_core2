# gpe_core2/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# ── 공통 AST node ─────────────────────────────────────
@dataclass
class ASTNode:
    instance_id: str
    class_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

# ── Rule base & 파생 ──────────────────────────────────
@dataclass
class BaseRule:
    """v1 규칙과 호환되는 최소 필드 집합"""
    opcode: str
    class_name: str                      # ← default 없음이 먼저!
    params: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class InstantiateRule(BaseRule):
    class_name: str
    instance_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AppendChildRule(BaseRule):
    parent_id: str
    child_id: str
    attribute_name: str = "children"

@dataclass
class RepeatRule(BaseRule):
    count: int
    instruction: BaseRule  # could be list

@dataclass
class AttentionSeed:
    rules: List[BaseRule]

# ── Payload ───────────────────────────────────────────
@dataclass
class GpePayload:
    """통합 v2 Payload 객체"""
    generative_payload: bytes

    # optional/meta
    payload_type: str = "json"
    rules: List[BaseRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_payload: Optional[Dict[str, Any]] = None
    
# ===== 최적화 규칙 클래스 =====

@dataclass
class ConstantRule(BaseRule):
    """반복 상수 값을 하나의 규칙으로 묶음"""
    op_code: str = field(default="CONSTANT", init=False)
    value_id: str
    value: Any
    references: List[str] = field(default_factory=list)

@dataclass
class TemplateRule(BaseRule):
    """구조는 동일·값만 다른 객체 묶음"""
    op_code: str = field(default="TEMPLATE", init=False)
    template_id: str
    structure: Dict[str, Any]
    variable_keys: List[str]
    instances: List[Dict[str, Any]]

@dataclass
class RangeRule(BaseRule):
    """연속 숫자 시퀀스"""
    op_code: str = field(default="RANGE", init=False)
    instance_ids: List[str]
    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float] = 1

@dataclass
class CompactListRule(BaseRule):
    """희소 리스트를 기본값+예외로 압축"""
    op_code: str = field(default="COMPACT_LIST", init=False)
    parent_id: str
    length: int
    default_value: Any
    exceptions: List[Tuple[int, Any]] = field(default_factory=list)






