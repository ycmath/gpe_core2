from __future__ import annotations
import json
import numpy as np
import cupy as cp
from typing import Any

from ..models import GpePayload
from ..vectorizer_hybrid_meta import hybrid_flatten_meta, OP_NEW, OP_APPEND
from ..decoder import GPEDecoder          # ← 추가
from .id_remap_opt import run_remap
from .assemble_graph import gpu_assemble
from .graph_to_py import cupy_graph_to_py


class GPEDecoderGPUStreamFull:
    """Full GPU decode: ID remap + CUDA graph assemble + Host copy-back."""

    def __init__(self, vram_frac: float = 0.7):
        self.vram_frac = vram_frac
    # ------------------------------------------------------------------
    def _auto_rows(self, n_rows: int, itemsize: int) -> int:
        """Return rows_per chunk size based on free VRAM."""
        free, _ = cp.cuda.runtime.memGetInfo()
        budget  = int(free * self.vram_frac)
        rows    = max(int(budget // (itemsize * 4)), 128_000)
        while rows > 128_000 and rows * itemsize * 4 > budget:
            rows //= 2
        return rows
    # ------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # fallback 우선
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])

        # v1.1(flat rule list) → GPU-flatten 미지원 → CPU 디코더 우회
        seeds_field = payload.generative_payload["seeds"]
        if seeds_field and isinstance(seeds_field, list) and "op_code" in seeds_field[0]:
            return GPEDecoder().decode(payload)

        # v1.0 : seeds = [{"rules":[…]}, …]  → hybrid flatten
        chunk = hybrid_flatten_meta(seeds_field)

        # 1) GPU ID remap
        op = chunk["op"]
        n_rows = op.size
        free, _ = cp.cuda.runtime.memGetInfo()
        rows_per = self._auto_rows(n_rows, op.itemsize)

        ids_a = np.empty(n_rows, np.uint32)
        ids_b = np.empty(n_rows, np.uint32)

        def ranges(total, step):
            s = 0
            while s < total:
                e = min(s + step, total)
                yield range(s, e)
                s = e

        for r in ranges(n_rows, rows_per):
            sub = {k: (v[r] if k.startswith(("op", "mask", "meta")) else v)
                   for k, v in chunk.items()}
            a, b = run_remap(sub)
            ids_a[r], ids_b[r] = a, b

        # 2) GPU graph assembly
        d_type, d_head, d_next, d_key = gpu_assemble(chunk, ids_a, ids_b)

        # key LUT 준비 (host copy once)
        key_blob = "|".join(chunk["lut_key"]).encode() + b"|"
        key_off  = np.fromiter(
            (0, *np.cumsum([len(k) + 1 for k in chunk["lut_key"]])),
            dtype=np.uint32,
        )

        d_type, d_head, d_next, d_key = gpu_assemble(chunk, ids_a, ids_b)

        objs = cupy_graph_to_py(
            d_type, d_head, d_next, d_key,
            chunk["lut_cls"],
            key_blob,
            key_off,
        )

        root_text = payload.generative_payload["root_id"]   # "n00000xxx"
        root_idx = int(root_text[1:], 10)
        return objs[root_idx]


## lut_key_blob 간단화: 키를 | 구분자로 이어붙여 offset 계산 → 필요 시 UTF-8 safe concat으로 개선 가능.
## 이제 gpu-full 백엔드는 Host Python 루프 없이 GPU → Host 단일 전송만으로 완전 복원이 이루어집니다.
