from itertools import count
import json, hashlib
from typing import Any, Dict
from gpe_core2.models import ASTNode

try:
    import xxhash  # type: ignore
    _fast_hash = lambda b: xxhash.xxh3_128_digest(b)
except ModuleNotFoundError:
    try:
        import mmh3  # type: ignore
        _fast_hash = lambda b: mmh3.hash_bytes(b, seed=0)
    except ModuleNotFoundError:
        _fast_hash = lambda b: hashlib.sha1(b).digest()

class ASTBuilder:
    def __init__(self):
        self.nodes: dict[str, ASTNode] = {}   # ← dict
        self._ctr = itertools.count()

    def build(self, obj):
        return self._visit(obj, None)
    
    # ------------------------------------------------------------------
    def _visit(self, obj: Any, parent: str | None) -> str:
        nid = f"n{next(self._ctr):08d}"
        ntype = (
            "dict" if isinstance(obj, dict) else
            "list" if isinstance(obj, list) else
            type(obj).__name__
        )
        pay = (
            json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
            if isinstance(obj, (dict, list)) else repr(obj).encode()
        )
        h = _fast_hash(pay).hex()[:32]
        node = ASTNode(
            instance_id=nid,
            class_name=ntype,
            attributes={"hash": h, "value": None},
            children=[],
        )
        self.nodes[nid] = node     # ← dict 저장
        if isinstance(obj, dict):
            for k, v in obj.items():
                cid = self._visit(v, nid)
                node.children.append(cid)
                self.nodes[cid].attributes["key"] = k
        elif isinstance(obj, list):
            for v in obj:
                cid = self._visit(v, nid)
                node.children.append(cid)
        else:
            node.attributes["value"] = obj
        return nid
