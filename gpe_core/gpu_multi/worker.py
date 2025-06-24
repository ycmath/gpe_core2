# ─────────────────────────────────────────────────────────────
# Ray 액터: 각 GPU 디바이스에서 ID-remap → assemble_graph 까지 수행
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
import ray
import cupy as cp
import numpy as np
from typing import Dict, Any, Tuple

from ..gpu.id_remap_opt import run_remap
from ..gpu.assemble_graph import gpu_assemble


@ray.remote(num_gpus=1)
class GPEWorker:
    def __init__(self, device_id: int):
        cp.cuda.Device(device_id).use()
        self.dev = device_id

    # ------------------------------------------------------------------
    def process_chunk(
        self,
        chunk: Dict[str, Any],
        rows: slice,
        ids_a_full: np.ndarray,
        ids_b_full: np.ndarray,
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        sub = {
            k: (v[rows] if k.startswith(("op", "mask", "meta")) else v)
            for k, v in chunk.items()
        }
        ids_a = ids_a_full[rows]
        ids_b = ids_b_full[rows]

        # GPU로 변환만 수행
        ida_dev = cp.asarray(ids_a, dtype=cp.uint32)
        idb_dev = cp.asarray(ids_b, dtype=cp.uint32)

        # gpu_assemble에 remap된 id 바로 전달
        d_type, d_head, d_next, d_key = gpu_assemble(sub, ids_a, ids_b)
        return d_type, d_head, d_next, d_key
