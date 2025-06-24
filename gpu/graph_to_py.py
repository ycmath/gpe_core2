"""
cupy_graph_to_py  (GPU child-list → 파이썬 객체)
───────────────────────────────────────────────
* d_key  : child 별 “4-byte 해시” (커널 v2에서 계산)
* key_blob / key_off : GPU → Host 복사된 전체 key 문자열 풀
"""
from __future__ import annotations
from typing import List, Any, Dict

import cupy as cp
import numpy as np


def cupy_graph_to_py(
    d_type: cp.ndarray,
    d_head: cp.ndarray,
    d_next: cp.ndarray,
    d_key:  cp.ndarray,
    lut_cls: List[str],
    key_blob: bytes,
    key_off: np.ndarray,
) -> Dict[int, Any]:
    n = int(d_type.size)
    objs: List[Any] = [None] * n
    key_cache: Dict[int, str] = {}

    # 1) allocate shells
    for vid in range(n):
        t = int(d_type[vid])
        objs[vid] = (
            {} if t == 0 else
            [] if t == 1 else
            {"__class__": lut_cls[t]}
        )

    # 2) children attach
    for parent in range(n):
        child = int(d_head[parent])
        while child != 0xFFFFFFFF:
            pobj = objs[parent]
            if isinstance(pobj, list):
                pobj.append(objs[child])
            else:
                h = int(d_key[child])
                if h not in key_cache:
                    # 해시 충돌 가능 → 선형 탐색으로 실제 key 찾기
                    # (dict-size 보통 적어 오버헤드 미미)
                    for i in range(len(key_off) - 1):
                        if int.from_bytes(key_blob[key_off[i] : key_off[i] + 4], "little") == h:
                            key_cache[h] = key_blob[key_off[i] : key_off[i + 1]].decode()
                            break
                k = key_cache[h]
                pobj[k] = objs[child]
            child = int(d_next[child])

    return objs

