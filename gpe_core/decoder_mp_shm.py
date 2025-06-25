from multiprocessing import shared_memory, get_context
import multiprocessing as mp
from .json_util import dumps
from .decoder import GPEDecoder
from .models import GpePayload
from typing import Dict, Any
import numpy as np, orjson, json
from math import ceil

class GPEDecoderMP_ShMem(GPEDecoder):
    """Shared-memory fan-in decoder (zero-copy IPC)."""
    def __init__(self, processes: int | None = None):
        self.processes = processes or max(mp.cpu_count() - 1, 1)

    def decode(self, payload: GpePayload):
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])

        raw = payload.generative_payload["seeds"]

        # v1.1 → 싱글-프로세스 fallback
        if raw and "op_code" in raw[0]:
            return GPEDecoder(use_numba=False).decode(payload)

        # v1.0 → 병렬 + shared-memory
        chunks = [s["rules"] for s in raw]
        ctx = get_context("spawn")
        with ctx.Pool(self.processes) as pool:
            infos = pool.map(_worker, chunks)

        objs, meta = {}, {}
        for name, size in infos:
            shm  = shared_memory.SharedMemory(name=name)
            try:
                payload_bytes = bytes(shm.buf[:size])   # 빠른 복사 → 독립 메모리
            finally:
                shm.close(); shm.unlink()               # view 없이 안전히 닫기

            o, m = orjson.loads(payload_bytes)
            objs.update(o); meta.update(m)
        return objs[payload.generative_payload["root_id"]]


def _worker(rules):
    dec = GPEDecoder(use_numba=False)
    o, m = {}, {}
    for r in rules:
        dec._apply_py(r, o, m)
    payload = orjson.dumps((o, m))
    shm = shared_memory.SharedMemory(create=True, size=len(payload))
    shm.buf[:len(payload)] = payload
    return shm.name, len(payload)
