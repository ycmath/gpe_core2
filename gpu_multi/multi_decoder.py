"""
multi_decoder.py
────────────────
Ray actor-pool을 이용해 여러 GPU 디바이스에 Seed 파티션을 분배하고,
ID remap → assemble_graph → host-merge 까지 수행한다.
"""
from __future__ import annotations
import json
import math
import numpy as np
import ray
from typing import Any, Dict, List

from ..models import GpePayload
from ..vectorizer_hybrid_meta import hybrid_flatten_meta
from ..gpu.id_remap_opt import run_remap
from .assemble_graph import gpu_assemble
from .graph_to_py import cupy_graph_to_py
from .worker import GPEWorker


class GPEDecoderGPU_Ray:
    """Multi-GPU decoder orchestrated by Ray actors."""

    def __init__(self, num_gpus: int | None = None, vram_frac: float = 0.65):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.num_gpus = num_gpus or max(1, len(ray.get_gpu_ids()))
        self.vram_frac = vram_frac
        self.workers = [
            GPEWorker.options(num_gpus=1, resources={"GPU": 1}).remote(i)
            for i in range(self.num_gpus)
        ]

    # ------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # fallback
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])

        chunk = hybrid_flatten_meta(payload.generative_payload["seeds"])
        n_rows = chunk["op"].size

        # 1) 단일 GPU로 ID remap 먼저 수행 (fast & memory-light)
        ids_a_full = np.empty(n_rows, np.uint32)
        ids_b_full = np.empty(n_rows, np.uint32)
        a, b = run_remap(chunk)
        ids_a_full[:] = a
        ids_b_full[:] = b

        # 2) 파티션 계산
        rows_per = math.ceil(n_rows / self.num_gpus)
        ranges = [slice(i, min(i + rows_per, n_rows)) for i in range(0, n_rows, rows_per)]

        # 3) 각 GPU actor에 작업 분배
        futs = [
            w.process_chunk.remote(chunk, rows, ids_a_full, ids_b_full)
            for w, rows in zip(self.workers, ranges)
        ]
        parts = ray.get(futs)  # List[Tuple[d_type, d_head, d_next, d_key]]

        # 4) Host-side merge & Python 객체 재구성
        #    — 키 LUT 준비
        lut_key_blob = "|".join(chunk["lut_key"]).encode() + b"|"
        lut_key_off = np.fromiter(
            (0, *np.cumsum([len(k) + 1 for k in chunk["lut_key"]])),
            dtype=np.uint32,
        )

        objs_global: Dict[int, Any] = {}
        offset = 0
        for (d_type, d_head, d_next, d_key), rows in zip(parts, ranges):
            local_objs = cupy_graph_to_py(
                d_type, d_head, d_next, d_key,
                chunk["lut_cls"],
                lut_key_blob,
                lut_key_off,
            )
            # local_objs 키 = 0..len(rows)-1 → global id = offset + idx
            for idx, obj in local_objs.items():
                objs_global[offset + idx] = obj
            offset += rows.stop - rows.start

        root_text = payload.generative_payload["root_id"]
        root_idx = int(root_text[1:], 10)
        return objs_global[root_idx]
