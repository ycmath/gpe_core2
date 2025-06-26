# gpe_core2/ast_builder.py
"""
Minimal stub so that encoder.base can import ASTBuilder.
Replace with full logic later if needed.
"""

from __future__ import annotations
from typing import Any

class ASTNode(dict):
    """Simple dict-based AST node."""
    pass

class ASTBuilder:
    """
    Very-light builder: wraps input data in a root node.
    Real v1 logic (tokenization, dedup, etc.) can be ported later.
    """

    @staticmethod
    def build(data: Any) -> ASTNode:
        return ASTNode({"root": data})
