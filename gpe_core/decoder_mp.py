import multiprocessing as mp
import json
from typing import Dict, Any
from .decoder import GPEDecoder
from .models import GpePayload

class GPEDecoderMP(GPEDecoder):
    """Seed‑level multiprocessing decoder (pickle IPC)."""
    def __init__(self, processes: int | None = None):
        self.processes = processes or max(mp.cpu_count() - 1, 1)
    
    def decode(self, payload: GpePayload):
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])
        seeds = payload.generative_payload["seeds"]
        with mp.Pool(self.processes) as pool:
            results = pool.map(self._decode_seed, seeds)
        objs, meta = {}, {}
        for o, m in results:
            objs.update(o); meta.update(m)
        return objs[payload.generative_payload["root_id"]]
    
    @staticmethod
    def _decode_seed(seed: Dict[str, Any]):
        dec = GPEDecoder()
        o, m = {}, {}
        for r in seed["rules"]:
            dec._apply_py(r, o, m)
        return o, m
