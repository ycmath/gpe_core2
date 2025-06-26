# gpe_core2/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# ──────────────────────────────────────────────────────────────
# 1. AST
@dataclass
class ASTNode:
    instance_id: str
    class_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

    # ── v1 호환 별칭 ───────────────────────
    @property
    def id(self) -> str:          # node.id
        return self.instance_id

    @property
    def type(self) -> str:        # node.type
        return self.class_name

# ──────────────────────────────────────────────────────────────
# 2. Rule Base & Derivatives
# ──────────────────────────────────────────────────────────────

@dataclass
class BaseRule:
    opcode: str                         # non-default
    params: Dict[str, Any] = field(default_factory=dict)  # ← 다시 default 허용


# -- 2-1. AST 조작 규칙 --------------------------------------------------------
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
    instruction: BaseRule


# -- 2-2. 최적화 규칙 ----------------------------------------------------------
@dataclass
class ConstantRule(BaseRule):
    value_id: str
    value: Any
    references: List[str] = field(default_factory=list)

    opcode: str = field(init=False, default="OP_CONST")


@dataclass
class RangeRule(BaseRule):
    instance_ids: List[str]
    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float] = 1

    opcode: str = field(init=False, default="OP_RANGE")


@dataclass
class TemplateRule(BaseRule):
    template_id: str
    structure: Dict[str, Any]
    variable_keys: List[str]
    instances: List[Dict[str, Any]]

    opcode: str = field(init=False, default="OP_TEMPLATE")


@dataclass
class CompactListRule(BaseRule):
    parent_id: str
    length: int
    default_value: Any
    exceptions: List[Tuple[int, Any]] = field(default_factory=list)

    opcode: str = field(init=False, default="OP_COMPACT_LIST")


# ──────────────────────────────────────────────────────────────
# 3. Attention Seed (stub)
# ──────────────────────────────────────────────────────────────
@dataclass
class AttentionSeed:
    rules: List[BaseRule]


# ──────────────────────────────────────────────────────────────
# 4. GPE Payload
# ──────────────────────────────────────────────────────────────
@dataclass
class GpePayload:
    """Unified v2 Payload object."""
    generative_payload: Any                       # bytes | dict

    # optional / meta
    payload_type: str = "json"
    rules: List[BaseRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_payload: Optional[Dict[str, Any]] = None
