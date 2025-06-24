"""
hybrid_flatten_meta(seeds)  →  dict(op, mask_a, pool16_a, …, meta_cls, meta_key)
- meta_cls :  uint16 배열  (NEW 행마다 class_name 사전 index)
- meta_key :  uint32 배열  (APPEND child 행마다 key 사전 index)
- lut_cls   :  List[str]   index → class_name
- lut_key   :  List[str]   index → dict-key string
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any
from .vectorizer_hybrid import build_pools
from .vectorizer import vectorize_seeds, OP_NEW, OP_APPEND

def hybrid_flatten_meta(seeds: List[Dict[str, Any]]):
    op, a, b, _ = vectorize_seeds(seeds)

    meta_cls = np.full(op.shape, 0xFFFF, np.uint16)   # default sentinel
    meta_key = np.full(op.shape, 0xFFFFFFFF, np.uint32)

    lut_cls: List[str] = []
    lut_key: List[str] = []

    def add(cls: str, table: List[str]) -> int:
        try:
            return table.index(cls)
        except ValueError:
            table.append(cls)
            return len(table)-1

    row = 0
    for s in seeds:
        for r in s["rules"]:
            if r["op_code"] == "NEW":
                idx = add(r["class_name"], lut_cls)
                meta_cls[row] = idx
            elif r["op_code"] == "APPEND":
                k = r.get("attribute_name") or r.get("key")
                idx = add(k, lut_key)
                meta_key[row] = idx
            row += 1

    masks_a, p16a, p32a = build_pools(a)
    masks_b, p16b, p32b = build_pools(b)

    return dict(
        op=op,
        mask_a=masks_a, mask_b=masks_b,
        pool16_a=p16a, pool32_a=p32a,
        pool16_b=p16b, pool32_b=p32b,
        meta_cls=meta_cls,
        meta_key=meta_key,
        lut_cls=lut_cls,
        lut_key=lut_key,
    )
