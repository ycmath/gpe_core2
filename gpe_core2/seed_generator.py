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

    def __init__(self, nodes: Dict[str, ASTNode], groups: Dict[str, List[str]]):
        self.nodes = nodes                      # AST node table
        self.groups = groups                    # hash → [node_ids]

        # Map every node id → its repetition group (if any)
        self._grp_of: Dict[str, List[str]] = {}
        for lst in groups.values():
            for nid in lst:
                self._grp_of[nid] = lst

        self._seen_repeat: Set[str] = set()     # group leaders already emitted

    # ------------------------------------------------------------------
    def generate(self, root_id: str | None = None) -> List[AttentionSeed]:
        """Return *one* AttentionSeed covering the whole AST (root‑aligned)."""
        if root_id is None:
            # pick the lexicographically smallest id as root (ASTBuilder behaves that way)
            root_id = min(self.nodes)
        rules = self._emit(root_id)
        return [AttentionSeed(rules=rules)]

    # ------------------------------------------------------------------
    def _emit(self, nid: str) -> List[BaseRule]:
        # If this node belongs to a repeat group AND we haven't emitted it yet
        if nid in self._grp_of and self._grp_of[nid][0] == nid and nid not in self._seen_repeat:
            grp = self._grp_of[nid]
            self._seen_repeat.update(grp)
            template_rules = self._emit_subtree(nid, emit_repeat=False)
            return [RepeatRule(opcode="REPEAT", count=len(grp), instruction=template_rules)]
        else:
            # normal path (including inner nodes inside a larger Repeat)
            return self._emit_subtree(nid, emit_repeat=True)

    # ------------------------------------------------------------------
    def _emit_subtree(self, nid: str, *, emit_repeat: bool) -> List[BaseRule]:
        """Return rules for a subtree. emit_repeat=False means we are already
        inside a Repeat template; we must not nest again."""
        node = self.nodes[nid]
        rules: List[BaseRule] = [
            InstantiateRule(
                opcode="NEW",                   # ← 필드명 수정
                class_name=node.type,
                instance_id=node.id,
                params={},  
                attributes={k: v for k, v in node.attributes.items() if k != "hash"},
            )
        ]

        for child_id in node.children:
            # Skip other instances that will be covered by a Repeat emitted elsewhere
            if child_id in self._grp_of and self._grp_of[child_id][0] != child_id and self._grp_of[child_id][0] in self._seen_repeat:
                continue
            # recurse — possibly emitting nested Repeat
            if emit_repeat:
                rules.extend(self._emit(child_id))
            else:
                rules.extend(self._emit_subtree(child_id, emit_repeat=False))
            rules.append(AppendChildRule(opcode="APPEND", parent_id=nid, child_id=child_id))
        
            # RepeatRule 생성부도 params 생략 가능
            RepeatRule(opcode="REPEAT", count=len(grp), instruction=template_rules)
        return rules
