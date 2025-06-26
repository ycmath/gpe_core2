# gpe_core2/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass
class ASTNode:
    id: str
    type: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BaseRule:   # 이미 정의돼 있다면 생략
    opcode: str
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


@dataclass
class GpePayload:
    """
    Canonical payload object for GPE-v2.

    ▸ `generative_payload`  : 인코더가 생성한 바이트 스트림
    ▸ `payload_type`        : (선택) "json", "vector", "mixed" … 등 확장 가능
    ▸ `rules`               : 적용된 Rule 목록 (디코더·디버깅용)
    ▸ `metadata`            : 캐싱 여부, 압축률 등 부가 메타정보
    ▸ `fallback_payload`    : 구버전 디코더 호환용 raw JSON
    """
    generative_payload: bytes

    # --- Optional / meta ---
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






