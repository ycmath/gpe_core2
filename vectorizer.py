import numpy as np
from typing import List, Tuple, Dict, Any

OP_NEW, OP_APPEND, OP_REPEAT_BEG, OP_REPEAT_END = range(4)

def vectorize_seeds(seeds: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idmap: Dict[str, int] = {}
    nxt = 0
    rows: List[Tuple[int, int, int, int]] = []

    def idx(s: str) -> int:
        nonlocal nxt
        if s not in idmap:
            idmap[s] = nxt
            nxt += 1
        return idmap[s]

    def emit(r):
        oc = r["op_code"]
        if oc == "NEW":
            rows.append((OP_NEW, idx(r["instance_id"]), 0, 0))
        elif oc == "APPEND":
            rows.append((OP_APPEND, idx(r["parent_id"]), idx(r["child_id"]), 0))
        elif oc == "REPEAT":
            rows.append((OP_REPEAT_BEG, r["count"], 0, 0))
            for sub in r["instruction"]:
                emit(sub)
            rows.append((OP_REPEAT_END, 0, 0, 0))

    for seed in seeds:
        for rule in seed["rules"]:
            emit(rule)

    arr = np.asarray(rows, dtype=np.int32)
    return arr[:, 0].astype(np.uint8), arr[:, 1], arr[:, 2], arr[:, 3]
