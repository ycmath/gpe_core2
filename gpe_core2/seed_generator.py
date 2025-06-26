# gpe_core2/seed_generator.py
from __future__ import annotations
from typing import Any, List
from gpe_core2.models import AttentionSeed

class SeedGenerator:
    """Very-light toy seed generator."""
    @staticmethod
    def generate(ast: Any) -> List[AttentionSeed]:
        # 실제 v1 로직은 AST → 토큰/시드 변환이지만,
        # 여기서는 AST 전체를 하나의 시드로 래핑
        return [AttentionSeed(data=ast)]
