from multiprocessing import shared_memory, get_context
import multiprocessing as mp
from .json_util import dumps
from .decoder import GPEDecoder
from .models import GpePayload
from typing import Dict, Any
import numpy as np, orjson, json

class GPEDecoderMP_ShMem(GPEDecoder):
    """Shared‑memory fan‑in decoder (zero‑copy IPC)."""
    def __init__(self, processes: int | None = None):
        self.processes = processes or max(mp.cpu_count() - 1, 1)
    
    def decode(self, payload: GpePayload):
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])
        seeds = payload.generative_payload["seeds"]
        ctx = get_context("spawn")
        with ctx.Pool(self.processes) as pool:
            infos = pool.map(self._worker, seeds)
        objs, meta = {}, {}
        for name, size in infos:
            shm = shared_memory.SharedMemory(name=name)
            view = memoryview(shm.buf)[:size]
            o, m = orjson.loads(view)
            objs.update(o); meta.update(m)
            shm.close(); shm.unlink()
        return objs[payload.generative_payload["root_id"]]
    
    @staticmethod
    def _worker(seed: Dict[str, Any]):
        dec = GPEDecoder(); o, m = {}, {}
        for r in seed["rules"]:
            dec._apply_py(r, o, m)
        payload = orjson.dumps((o, m))
        shm = shared_memory.SharedMemory(create=True, size=len(payload))
        shm.buf[: len(payload)] = payload
        return shm.name, len(payload)
