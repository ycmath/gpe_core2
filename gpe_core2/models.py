# gpe_core2/models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class AttentionSeed:
    """Stub for transformer-style seed."""
    data: Any

@dataclass
class BaseRule:
    """Generic rule container for encoder."""
    opcode: str
    params: Dict[str, Any]

@dataclass
class GpePayload:
    """Wrapper returned by encoder."""
    generative_payload: bytes
    metadata: Dict[str, Any] = None
    rules: List[BaseRule] = None
