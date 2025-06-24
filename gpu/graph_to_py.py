"""
cupy_graph_to_py
────────────────
assemble_graph.cu 결과 버퍼(d_type / d_head / d_next / d_key)와
룩업 테이블(lut_cls, lut_key_blob, lut_key_off)을 받아
파이썬 객체 그래프(dict / list / custom)를 재구성한다.
"""
from __future__ import annotations
from typing import List, Any, Dict

import cupy as cp
import numpy as np


def cupy_graph_to_py(
    d_type: cp.ndarray,
    d_head: cp.ndarray,
    d_next: cp.ndarray,
    d_key: cp.ndarray,
    lut_cls: List[str],
    lut_key_blob: bytes,
    lut_key_off: np.ndarray,
) -> Dict[int, Any]:
    n = int(d_type.size)
    objs: List[Any] = [None] * n

    # 1) allocate node shells
    for vid in range(n):
        t = int(d_type[vid])
        if t == 0:
            objs[vid] = {}
        elif t == 1:
            objs[vid] = []
        else:
            objs[vid] = {"__class__": lut_cls[t]}

    # 2) second pass: attach children
    for parent in range(n):
        child = int(d_head[parent])
        while child != 0xFFFFFFFF:
            pobj = objs[parent]
            if isinstance(pobj, list):
                pobj.append(objs[child])
            else:
                k_idx = int(d_key[child])
                # key off 배열 끝에 sentinel offset 포함
                k = lut_key_blob[lut_key_off[k_idx] : lut_key_off[k_idx + 1]].decode()
                pobj[k] = objs[child]
            child = int(d_next[child])

    return objs
