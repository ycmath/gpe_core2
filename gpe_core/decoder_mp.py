import multiprocessing as mp
import json
from typing import Dict, Any
from .decoder import GPEDecoder
from .models import GpePayload
from math import ceil

class GPEDecoderMP(GPEDecoder):
    """Seed-level multiprocessing decoder (pickle IPC)."""
    def __init__(self, processes: int | None = None):
        self.processes = processes or max(mp.cpu_count() - 1, 1)

    def decode(self, payload: GpePayload):
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])

        raw = payload.generative_payload["seeds"]

        # ── v1.1 (flat rule list) → 싱글-프로세스 fallback ──
        if raw and "op_code" in raw[0]:
            return GPEDecoder(use_numba=False).decode(payload)

        # ── v1.0 (seed list) → 다중 프로세스 ──
        chunk = ceil(len(raw) / self.processes)
        chunks = [s["rules"] for s in raw]              # 시드별 rule 리스트
        with mp.Pool(self.processes) as pool:
            results = pool.map(_decode_chunk, chunks)

        objs, meta = {}, {}
        for o, m in results:
            objs.update(o)
            meta.update(m)
        return objs[payload.generative_payload["root_id"]]
        
# 워커 함수 (Numba OFF)
def _decode_chunk(rules):
    dec = GPEDecoder(use_numba=False)
    o, m = {}, {}
    for r in rules:
        dec._apply_py(r, o, m)
    return o, m
