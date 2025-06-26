"""
Thin compatibility shim so that legacy
`from gpe_core1_compat.ast_builder import ASTBuilder`
continues to work on v2 code-base.
"""
from importlib import import_module

def __getattr__(name: str):
    # 모든 심볼을 gpe_core2.* 쪽으로 포워딩
    return getattr(import_module(f"gpe_core2.{name}"), name)  # type: ignore
