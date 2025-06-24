"""Detect repeated sub‑trees using node‑hash grouping.
If **Numba** is available the costly recursive `size()` computation is JIT‑
compiled, giving 3‑5× speed‑up on large (>100k node) ASTs.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
from .models import ASTNode
# optional numba JIT ----------------------------------------------------------
try:
    from numba import njit  # type: ignore
    _NUMBA = True
except ModuleNotFoundError:  # pragma: no cover
    _NUMBA = False
class RepetitionDetector:
    def __init__(self, nodes: Dict[str, ASTNode], min_occ: int = 2, min_size: int = 3):
        self.nodes = nodes
        self.min_occ = min_occ
        self.min_size = min_size
        # build adjacency in NumPy‑friendly form (for numba path)
        self._id2idx: Dict[str, int] = {nid: i for i, nid in enumerate(nodes)}
        self._children_idx: List[List[int]] = [
            [self._id2idx[c] for c in n.children] for n in nodes.values()
        ]
        self._hashes: List[str] = [n.attributes["hash"] for n in nodes.values()]
        # compile size() if numba present
        if _NUMBA:
            self._size = self._compile_size()
        else:
            self._cache: Dict[int, int] = {}
            self._size = self._size_py
    # ------------------------------------------------------------------
    def detect(self):
        groups = defaultdict(list)
        for idx, h in enumerate(self._hashes):
            nid = list(self.nodes)[idx]
            groups[h].append(idx)
        rep: Dict[str, List[str]] = {}
        for h, idx_list in groups.items():
            qual = [i for i in idx_list if self._size(i) >= self.min_size]
            if len(qual) >= self.min_occ:
                rep[h] = [list(self.nodes)[i] for i in qual]
        return rep
    # ------------------------------------------------------------------
    def _compile_size(self):
        children = self._children_idx
        n_nodes  = len(children)
        @njit(cache=False)  # cache=False로 변경!!
        def size(idx: int, memo: Dict[int, int] = {}):  # type: ignore[arg-type]
            if idx in memo:
                return memo[idx]
            s = 1
            for c in children[idx]:
                s += size(c, memo)
            memo[idx] = s
            return s
        return size
    # fallback pure python ------------------------------------------------
    def _size_py(self, idx: int) -> int:
        if idx in self._cache:
            return self._cache[idx]
        s = 1 + sum(self._size_py(c) for c in self._children_idx[idx])
        self._cache[idx] = s
        return s
