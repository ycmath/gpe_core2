from __future__ import annotations

import cupy as cp
import numpy as np
import json
from typing import Iterable, Any, Dict

from ..models import GpePayload
from ..vectorizer_hybrid import hybrid_flatten
from .id_remap_opt import run_remap
from ..decoder import GPEDecoder


class GPEDecoderGPUStream:
    """GPU remap + CPU 객체 재구성 (단순 스트리밍)."""
    def __init__(self, vram_frac: float = 0.7):
        self.vram_frac = vram_frac
        
    def _auto_rows(self, n_rows: int, itemsize: int) -> int:
        """GPU free-mem 기준 rows_per 자동 조정."""
        free, _ = cp.cuda.runtime.memGetInfo()
        budget  = int(free * self.vram_frac)
        rows    = max(int(budget // (itemsize * 4)), 128_000)
        while rows > 128_000 and rows * itemsize * 4 > budget:
            rows //= 2
        return rows

    # ------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # 0) fallback JSON 우선
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])
            
        # ── v1.1 (flat rule list) 은 GPU-remap 아직 미지원 → CPU 경로 ──
        seeds_field = payload.generative_payload["seeds"]
        if seeds_field and isinstance(seeds_field, list) and "op_code" in seeds_field[0]:
            # 단순 포워딩 – GPEDecoder 는 CONSTANT / RANGE / COMPACT_LIST 지원
            return GPEDecoder().decode(payload)
            
        # 1) seeds → hybrid 배열(flatten)
        chunk = hybrid_flatten(seeds_field)

        op   = chunk["op"]
        total = op.size

        # 2) chunk 크기 계산 (GPU VRAM 70 % 정도 사용)
        free, _ = cp.cuda.runtime.memGetInfo()
        rows_per = self._auto_rows(total, op.itemsize)

        def ranges(n: int, step: int) -> Iterable[range]:
            s = 0
            while s < n:
                e = min(s + step, n)
                yield range(s, e)
                s = e

        ids_a = np.empty(total, dtype=np.uint32)
        ids_b = np.empty(total, dtype=np.uint32)

        # 3) GPU remap — chunk 단위
        for r in ranges(total, rows_per):
            sub = {k: (v[r] if k.startswith(("op", "mask")) else v)
                   for k, v in chunk.items()}
            a, b = run_remap(sub)
            ids_a[r], ids_b[r] = a, b

        # 4) CPU 디코더로 최종 객체 그래프 재구성
        #    (ids 배열은 payload 에 임시 주입)
        payload.generative_payload["hybrid_ids_a"] = ids_a.tolist()
        payload.generative_payload["hybrid_ids_b"] = ids_b.tolist()

        return GPEDecoder().decode(payload)
