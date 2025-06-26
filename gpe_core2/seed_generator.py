# gpe_core2/seed_generator.py  (수정/교체)

from __future__ import annotations

from typing import Dict, List, Set

from .models import (
    ASTNode,
    AttentionSeed,
    BaseRule,
    InstantiateRule,
    AppendChildRule,
    RepeatRule,
)


class SeedGenerator:
    """Generate rule trees with **nested RepeatRule** support."""

    # ────────────────────────────────────────────────────────────
    def __init__(self, nodes: Dict[str, ASTNode], groups: Dict[str, List[str]]):
        self.nodes = nodes          # AST node table
        self.groups = groups        # hash → [node_ids]

        # node_id → 그 노드가 속한 반복 그룹(List[str])
        self._grp_of: Dict[str, List[str]] = {
            nid: lst for lst in groups.values() for nid in lst
        }
        self._seen_repeat: Set[str] = set()  # 이미 RepeatRule을 만든 그룹 리더

    # ────────────────────────────────────────────────────────────
    def generate(self, root_id: str | None = None) -> List[AttentionSeed]:
        """Return *one* AttentionSeed covering the whole AST."""
        if root_id is None:
            root_id = min(self.nodes)  # lexicographically smallest
        rules = self._emit(root_id)
        return [AttentionSeed(rules=rules)]

    # ────────────────────────────────────────────────────────────
    def _emit(self, nid: str) -> List[BaseRule]:
        """
        그룹 리더라면 RepeatRule 한 방으로 압축,
        아니면 평소대로 서브트리 방출.
        """
        if (
            nid in self._grp_of
            and self._grp_of[nid][0] == nid        # 그룹의 첫 번째=리더
            and nid not in self._seen_repeat
        ):
            grp = self._grp_of[nid]
            self._seen_repeat.update(grp)
            template_rules = self._emit_subtree(nid, emit_repeat=False)
            return [
                RepeatRule(
                    opcode="REPEAT",
                    params={},
                    count=len(grp),
                    instruction=template_rules,
                )
            ]
        else:
            # 일반 노드
            return self._emit_subtree(nid, emit_repeat=True)

    # ────────────────────────────────────────────────────────────
    def _emit_subtree(self, nid: str, *, emit_repeat: bool) -> List[BaseRule]:
        """emit_repeat=False → 이미 Repeat 컨텍스트 안임."""
        node = self.nodes[nid]

        rules: List[BaseRule] = [
            InstantiateRule(
                opcode="NEW",
                params={},
                class_name=node.class_name,      # ASTNode 별칭 사용
                instance_id=node.instance_id,
                attributes={
                    k: v for k, v in node.attributes.items() if k != "hash"
                },
            )
        ]

        for child_id in node.children:
            # 이미 RepeatRule로 처리될 자식은 스킵
            leader = self._grp_of.get(child_id, [None])[0]
            if leader != child_id and leader in self._seen_repeat:
                continue

            # 재귀
            if emit_repeat:
                rules.extend(self._emit(child_id))
            else:
                rules.extend(self._emit_subtree(child_id, emit_repeat=False))

            rules.append(
                AppendChildRule(
                    opcode="APPEND",
                    params={},
                    parent_id=nid,
                    child_id=child_id,
                )
            )

        return rules
