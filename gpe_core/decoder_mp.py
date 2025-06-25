import multiprocessing as mp
import json
from typing import Dict, Any
from .decoder import GPEDecoder
from .models import GpePayload
from math import ceil

class GPEDecoderMP(GPEDecoder):
    """Seed‑level multiprocessing decoder (pickle IPC)."""
    def __init__(self, processes: int | None = None):
        self.processes = processes or max(mp.cpu_count() - 1, 1)
    
    def decode(self, payload: GpePayload):
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])
        raw = payload.generative_payload["seeds"]

        # v1.0 = [{"rules":[…]}, …] ;  v1.1 = [{"op_code":…}, …]
        if raw and "op_code" in raw[0]:
            # flat list ⇒ 블록 단위로 나눠서 병렬 처리
            chunk = ceil(len(raw) / self.processes)
            chunks = [raw[i : i + chunk] for i in range(0, len(raw), chunk)]
            target = self._decode_chunk        # 새 worker
        else:
            chunks = [s["rules"] for s in raw]  # v1.0: 시드별 rules
            target = self._decode_chunk

        with mp.Pool(self.processes) as pool:
            results = pool.map(target, chunks)
        objs, meta = {}, {}
        for o, m in results:
            objs.update(o); meta.update(m)
        return objs[payload.generative_payload["root_id"]]
    
    @staticmethod
    def _decode_chunk(rules):
        dec = GPEDecoder(use_numba=False)
        o, m = {}, {}
        for r in rules:
            dec._apply_py(r, o, m)
        return o, m
