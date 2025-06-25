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

        # v1.1 은 의존성 보존을 위해 싱글-프로세스 처리
        if raw and "op_code" in raw[0]:
            dec = GPEDecoder(use_numba=False)
            obj = dec.decode(payload)
            return obj
        else:+        # v1.1 은 의존성 보존을 위해 싱글-프로세스 처리
        if raw and "op_code" in raw[0]:
            dec = GPEDecoder(use_numba=False)
            obj = dec.decode(payload)
            return obj
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
