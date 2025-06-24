# ===== gpe_core project — FINAL CONSOLIDATED CODE (v1) =====
# Python 3.11 │ MIT License
# ─────────────────────────────────────────────────────────────
# ❶ DATA MODEL / AST / VECTORIZER
# ❷ REPETITION DETECTOR / SEED GENERATOR
# ❸ ENCODER / DECODER (CPU · MP · SHM)
# ❹ JSON util (orjson fallback)
# ❺ CLI 툴 (gpe encode / decode / bench)
# ─────────────────────────────────────────────────────────────
# (GPU·Hybrid·Stream 디코더 파일은 별도 모듈에서 유지—필요 시 확장)

################################################################################
# gpe_core/models.py
################################################################################
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ASTNode:
    id: str
    type: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BaseRule:
    op_code: str

@dataclass
class InstantiateRule(BaseRule):
    class_name: str
    instance_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AppendChildRule(BaseRule):
    parent_id: str
    child_id: str
    attribute_name: str = "children"

@dataclass
class RepeatRule(BaseRule):
    count: int
    instruction: BaseRule  # could be list

@dataclass
class AttentionSeed:
    rules: List[BaseRule]

@dataclass
class GpePayload:
    payload_type: str
    generative_payload: Dict[str, Any]
    fallback_payload: Optional[Dict[str, Any]] = None

################################################################################
# gpe_core/ast_builder.py
################################################################################
from itertools import count
import json, hashlib
from typing import Any, Dict

try:
    import xxhash  # type: ignore
    _fast_hash = lambda b: xxhash.xxh3_128_digest(b)
except ModuleNotFoundError:
    try:
        import mmh3  # type: ignore
        _fast_hash = lambda b: mmh3.hash_bytes(b, seed=0)
    except ModuleNotFoundError:
        _fast_hash = lambda b: hashlib.sha1(b).digest()

class ASTBuilder:
    _ctr = count()

    def __init__(self):
        self.nodes: Dict[str, ASTNode] = {}

    def build(self, obj: Any) -> str:
        return self._visit(obj, None)

    # ------------------------------------------------------------------
    def _visit(self, obj: Any, parent: str | None) -> str:
        nid = f"n{next(self._ctr):08d}"
        ntype = (
            "dict" if isinstance(obj, dict) else
            "list" if isinstance(obj, list) else
            type(obj).__name__
        )
        pay = (
            json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
            if isinstance(obj, (dict, list)) else repr(obj).encode()
        )
        h = _fast_hash(pay).hex()[:32]
        node = ASTNode(id=nid, type=ntype, parent_id=parent,
                       attributes={"hash": h, "value": None})
        self.nodes[nid] = node
        if isinstance(obj, dict):
            for k, v in obj.items():
                cid = self._visit(v, nid)
                node.children.append(cid)
                self.nodes[cid].attributes["key"] = k
        elif isinstance(obj, list):
            for v in obj:
                cid = self._visit(v, nid)
                node.children.append(cid)
        else:
            node.attributes["value"] = obj
        return nid

################################################################################
# gpe_core/vectorizer.py
################################################################################
import numpy as np
from typing import List, Tuple, Dict, Any

OP_NEW, OP_APPEND, OP_REPEAT_BEG, OP_REPEAT_END = range(4)

def vectorize_seeds(seeds: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idmap: Dict[str, int] = {}
    nxt = 0
    rows: List[Tuple[int, int, int, int]] = []

    def idx(s: str) -> int:
        nonlocal nxt
        if s not in idmap:
            idmap[s] = nxt
            nxt += 1
        return idmap[s]

    def emit(r):
        oc = r["op_code"]
        if oc == "NEW":
            rows.append((OP_NEW, idx(r["instance_id"]), 0, 0))
        elif oc == "APPEND":
            rows.append((OP_APPEND, idx(r["parent_id"]), idx(r["child_id"]), 0))
        elif oc == "REPEAT":
            rows.append((OP_REPEAT_BEG, r["count"], 0, 0))
            for sub in r["instruction"]:
                emit(sub)
            rows.append((OP_REPEAT_END, 0, 0, 0))

    for seed in seeds:
        for rule in seed["rules"]:
            emit(rule)

    arr = np.asarray(rows, dtype=np.int32)
    return arr[:, 0].astype(np.uint8), arr[:, 1], arr[:, 2], arr[:, 3]

################################################################################
# gpe_core/repetition_detector.py
################################################################################
from collections import defaultdict

class RepetitionDetector:
    def __init__(self, nodes: Dict[str, ASTNode], min_occ: int = 2, min_size: int = 3):
        self.nodes = nodes
        self.min_occ = min_occ
        self.min_size = min_size
        self._cache: Dict[str, int] = {}

    def detect(self):
        groups = defaultdict(list)
        for nid, n in self.nodes.items():
            groups[n.attributes["hash"]].append(nid)

        def size(nid: str) -> int:
            if nid in self._cache:
                return self._cache[nid]
            s = 1 + sum(size(c) for c in self.nodes[nid].children)
            self._cache[nid] = s
            return s

        rep: Dict[str, List[str]] = {}
        for h, ids in groups.items():
            qual = [i for i in ids if size(i) >= self.min_size]
            if len(qual) >= self.min_occ:
                rep[h] = qual
        return rep

################################################################################
# gpe_core/seed_generator.py
################################################################################
from typing import List

class SeedGenerator:
    def __init__(self, nodes: Dict[str, ASTNode], groups: Dict[str, List[str]]):
        self.nodes = nodes
        self.groups = groups

    def generate(self):
        seeds: List[AttentionSeed] = []
        for roots in self.groups.values():
            tmpl = self._emit(roots[0])
            seeds.append(
                AttentionSeed(
                    rules=[RepeatRule(op_code="REPEAT", count=len(roots), instruction=tmpl)]
                )
            )
        return seeds

    def _emit(self, rid: str):
        n = self.nodes[rid]
        rules: List[BaseRule] = [
            InstantiateRule(
                op_code="NEW",
                class_name=n.type,
                instance_id=n.id,
                attributes={k: v for k, v in n.attributes.items() if k != "hash"},
            )
        ]
        for c in n.children:
            rules.extend(self._emit(c))
            rules.append(AppendChildRule(op_code="APPEND", parent_id=n.id, child_id=c))
        return rules

################################################################################
# gpe_core/json_util.py
################################################################################
import json
try:
    import orjson  # type: ignore
    def dumps(o):
        return orjson.dumps(o, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY).decode()
    backend = lambda: "orjson"
except ModuleNotFoundError:
    dumps = lambda o: json.dumps(o, separators=(",", ":"))
    backend = lambda: "stdlib"

################################################################################
# gpe_core/encoder.py
################################################################################
from dataclasses import asdict

class GPEEncoder:
    def __init__(self, include_fallback: bool = True):
        self.include_fallback = include_fallback

    def encode(self, data):
        builder = ASTBuilder()
        root_id = builder.build(data)
        rep = RepetitionDetector(builder.nodes).detect()
        seeds = SeedGenerator(builder.nodes, rep).generate()
        gen = {
            "version": "gpe.v1",
            "root_id": root_id,
            "seeds": [asdict(s) for s in seeds],
        }
        fb = dumps(data) if self.include_fallback else None
        return GpePayload(
            payload_type="gpe.v1",
            generative_payload=gen,
            fallback_payload={"json": fb} if fb else None,
        )

################################################################################
# gpe_core/decoder.py (CPU)
################################################################################
import json, copy
from typing import Dict, Any

class GPEDecoder:
    """Single‑process reference decoder (fallback‑first)."""

    # ------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # 0) fallback JSON 우선
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])

        # 1) generative payload 실행
        objs: Dict[str, Any] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        for seed in payload.generative_payload["seeds"]:
            for rule in seed["rules"]:
                self._apply(rule, objs, meta)

        root_id = payload.generative_payload["root_id"]
        if root_id not in objs:
            raise ValueError(f"Root id {root_id} missing after decode")
        return objs[root_id]

    # ------------------------------------------------------------------
    def _apply(self, r: Dict[str, Any], o: Dict[str, Any], m: Dict[str, Dict[str, Any]]):
        op = r["op_code"]
        if op == "NEW":
            vid, cls = r["instance_id"], r["class_name"]
            attrs = r.get("attributes", {})
            if "value" in attrs:
                o[vid] = attrs["value"]
            elif cls == "dict":
                o[vid] = {}
            elif cls == "list":
                o[vid] = []
            else:
                o[vid] = {"__class__": cls}
            m[vid] = attrs
        elif op == "APPEND":
            p, c = r["parent_id"], r["child_id"]
            parent, child = o[p], o[c]
            if isinstance(parent, list):
                parent.append(child)
            else:
                key = m[c].get("key")
                parent[key] = child
        elif op == "REPEAT":
            for _ in range(r["count"]):
                tmpl = copy.deepcopy(r["instruction"])
                for rule in tmpl:
                    self._apply(rule, o, m)
        else:
            raise ValueError(f"Unknown op {op}")

################################################################################
# gpe_core/decoder_mp.py (multiprocessing)
################################################################################
import multiprocessing as mp

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
            dec._apply(r, o, m)
        return o, m

################################################################################
# gpe_core/decoder_mp_shm.py (multiprocessing + shared_memory)
################################################################################
from multiprocessing import shared_memory, get_context
from ..json_util import dumps
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
            dec._apply(r, o, m)
        payload = orjson.dumps((o, m))
        shm = shared_memory.SharedMemory(create=True, size=len(payload))
        shm.buf[: len(payload)] = payload
        return shm.name, len(payload)

################################################################################
# gpe_core/cli.py
################################################################################
"""Command‑line interface: gpe encode / decode / bench"""
from __future__ import annotations
from pathlib import Path
import argparse, time, json, sys
from typing import Dict

from .encoder import GPEEncoder
from .models import GpePayload
from .decoder import GPEDecoder
from .decoder_mp import GPEDecoderMP
from .decoder_mp_shm import GPEDecoderMP_ShMem

# Optional GPU backend
BACKENDS: Dict[str, type] = {
    "cpu": GPEDecoder,
    "mp": GPEDecoderMP,
    "mp-shm": GPEDecoderMP_ShMem,
}
try:
    from .gpu.stream_decoder import GPEDecoderGPUStream  # type: ignore
    BACKENDS["gpu-stream"] = GPEDecoderGPUStream
except Exception:
    pass

# -------------------------------------------------------------------------

def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _dump_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

# -------------------------------------------------------------------------

def _get_decoder(name: str):
    if name not in BACKENDS:
        sys.exit(f"❌  unknown backend '{name}' (choose from {', '.join(BACKENDS)})")
    return BACKENDS[name]()

# -------------------------------------------------------------------------

def cmd_encode(ns):
    data = _load_json(ns.input)
    enc  = GPEEncoder(include_fallback=not ns.no_fallback)
    t0 = time.perf_counter(); payload = enc.encode(data); enc_ms = (time.perf_counter()-t0)*1000
    print(f"✓ encoded in {enc_ms:.2f} ms → {ns.output}")
    obj = payload.generative_payload if ns.strip else payload.__dict__
    _dump_json(obj, ns.output)


def cmd_decode(ns):
    raw = _load_json(ns.input)
    if "generative_payload" in raw:
        payload = GpePayload(**raw)  # type: ignore[arg-type]
    else:
        payload = GpePayload(payload_type="gpe.v1", generative_payload={}, fallback_payload={"json": json.dumps(raw)})

    dec = _get_decoder(ns.backend)
    t0 = time.perf_counter(); data = dec.decode(payload); dec_ms = (time.perf_counter()-t0)*1000
    print(f"✓ decoded in {dec_ms:.2f} ms using backend '{ns.backend}' → {ns.output}")
    _dump_json(data, ns.output)


def cmd_bench(ns):
    from random import randint
    data = [{"x": randint(0,9), "y": [randint(0,9) for _ in range(5)]} for _ in range(ns.n)]
    enc = GPEEncoder(include_fallback=False)
    payload = enc.encode(data)

    dec = _get_decoder(ns.backend)
    t0 = time.perf_counter(); enc.encode(data); enc_ms = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter(); dec.decode(payload); dec_ms = (time.perf_counter()-t0)*1000
    print(f"n={ns.n:,} | encode {enc_ms:.2f} ms | decode({ns.backend}) {dec_ms:.2f} ms")

# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(prog="gpe", description="GPE encode/decode toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # encode
    sp = sub.add_parser("encode", help="JSON → GPE payload")
    sp.add_argument("--input",  "-i", type=Path, required=True)
    sp.add_argument("--output", "-o", type=Path, required=True)
    sp.add_argument("--no-fallback", action="store_true", help="omit raw JSON fallback")
    sp.add_argument("--strip",        action="store_true", help="save only generative_payload")
    sp.set_defaults(func=cmd_encode)

    # decode
    sp = sub.add_parser("decode", help="GPE payload → JSON")
    sp.add_argument("--input",  "-i", type=Path, required=True)
    sp.add_argument("--output", "-o", type=Path, required=True)
    sp.add_argument("--backend","-b", default="cpu", choices=list(BACKENDS))
    sp.set_defaults(func=cmd_decode)

    # bench
    sp = sub.add_parser("bench",  help="quick encode/decode benchmark")
    sp.add_argument("--n", type=int, default=10000, help="synthetic record count")
    sp.add_argument("--backend","-b", default="cpu", choices=list(BACKENDS))
    sp.set_defaults(func=cmd_bench)

    ns = ap.parse_args(); ns.func(ns)

if __name__ == "__main__":
    main()


################################################################################
# gpe_core/vectorizer_hybrid.py
################################################################################
"""Hybrid vectorizer: uint16 + uint32 pool 레이아웃 생성."""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any

from .vectorizer import vectorize_seeds, OP_NEW, OP_APPEND, OP_REPEAT_BEG, OP_REPEAT_END

def build_pools(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (ids < 65_536).astype(np.uint8)
    pool16 = ids[mask == 1].astype(np.uint16, copy=False)
    pool32 = ids[mask == 0].astype(np.uint32, copy=False)
    return mask, pool16, pool32

def hybrid_flatten(seeds: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    op, a, b, _ = vectorize_seeds(seeds)
    m_a, p16_a, p32_a = build_pools(a)
    m_b, p16_b, p32_b = build_pools(b)
    return dict(
        op=op,
        mask_a=m_a, mask_b=m_b,
        pool16_a=p16_a, pool32_a=p32_a,
        pool16_b=p16_b, pool32_b=p32_b,
    )

################################################################################
# gpe_core/gpu/id_remap_opt.cu
################################################################################

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

extern "C" __global__
void id_remap_opt(const uint8_t*  __restrict__ op,
                  const uint8_t*  __restrict__ mask16_a,
                  const uint8_t*  __restrict__ mask16_b,
                  const uint16_t* __restrict__ pool16_a,
                  const uint32_t* __restrict__ pool32_a,
                  const uint16_t* __restrict__ pool16_b,
                  const uint32_t* __restrict__ pool32_b,
                  uint32_t*       __restrict__ out_a,
                  uint32_t*       __restrict__ out_b,
                  int32_t n16_a, int32_t n32_a,
                  int32_t n16_b, int32_t n32_b,
                  int32_t n_rows)
{
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage tmp;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;

    int bit_a = mask16_a[i];
    int bit_b = mask16_b[i];

    int pref_a; Scan(tmp).ExclusiveSum(bit_a, pref_a);
    int pref_b; Scan(tmp).ExclusiveSum(bit_b, pref_b);

    int idx16a = pref_a;
    int idx32a = i - idx16a;
    int idx16b = pref_b;
    int idx32b = i - idx16b;

    uint32_t ida = bit_a ? static_cast<uint32_t>(pool16_a[idx16a])
                         : pool32_a[idx32a];
    uint32_t idb = bit_b ? static_cast<uint32_t>(pool16_b[idx16b])
                         : pool32_b[idx32b];

    if (op[i] == 0u) {                 // NEW
        out_a[i] = ida;
    } else if (op[i] == 1u) {          // APPEND
        out_a[i] = ida;
        out_b[i] = idb;
    }
}

################################################################################
# gpe_core/gpu/id_remap_opt.py
################################################################################

import cupy as cp
from pathlib import Path
import numpy as np

_SRC = (Path(__file__).with_suffix('.cu')).read_text()
_KERNEL = cp.RawKernel(_SRC, "id_remap_opt",
                       options=("-I/usr/local/cuda/include", "-use_fast_math", "-O3"))

def run_remap(chunk: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    op      = cp.asarray(chunk['op'],       dtype=cp.uint8)
    mask_a  = cp.asarray(chunk['mask_a'],   dtype=cp.uint8)
    mask_b  = cp.asarray(chunk['mask_b'],   dtype=cp.uint8)
    pool16a = cp.asarray(chunk['pool16_a'], dtype=cp.uint16)
    pool32a = cp.asarray(chunk['pool32_a'], dtype=cp.uint32)
    pool16b = cp.asarray(chunk['pool16_b'], dtype=cp.uint16)
    pool32b = cp.asarray(chunk['pool32_b'], dtype=cp.uint32)

    out_a = cp.empty(op.size, dtype=cp.uint32)
    out_b = cp.empty(op.size, dtype=cp.uint32)

    threads = 256
    blocks  = (op.size + threads - 1) // threads
    _KERNEL((blocks,), (threads,),
            (op, mask_a, mask_b,
             pool16a, pool32a, pool16b, pool32b,
             out_a, out_b,
             np.int32(pool16a.size), np.int32(pool32a.size),
             np.int32(pool16b.size), np.int32(pool32b.size),
             np.int32(op.size)))
    return cp.asnumpy(out_a), cp.asnumpy(out_b)

################################################################################
# gpe_core/gpu/stream_decoder.py
################################################################################

from __future__ import annotations
import cupy as cp, numpy as np, json
from typing import List, Dict, Any, Iterable
from ..models import GpePayload
from ..vectorizer_hybrid import hybrid_flatten
from .id_remap_opt import run_remap


class GPEDecoderGPUStream:
    """Chunk-stream GPU decoder (no CPU fallback hack)."""

    def __init__(self, vram_frac: float = 0.7):
        self.vram_frac = vram_frac

    # ------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # ── 0) fallback 우선 ────────────────────────────────────────────
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])

        # ── 1) flatten → hybrid arrays ────────────────────────────────
        seeds = payload.generative_payload["seeds"]
        chunk = hybrid_flatten(seeds)
        op  = chunk["op"]
        tot = op.size

        # ── 2) GPU remap (chunked) ────────────────────────────────────
        free, _ = cp.cuda.runtime.memGetInfo()
        rows_per = max(int((free * self.vram_frac)//(op.itemsize*4)), 256_000)

        def ranges(n, step) -> Iterable[range]:
            s = 0
            while s < n:
                e = min(s+step, n)
                yield range(s, e)
                s = e

        ids_a = np.empty(tot, dtype=np.uint32)
        ids_b = np.empty(tot, dtype=np.uint32)

        for r in ranges(tot, rows_per):
            sub = {k: (v[r] if k.startswith(("op", "mask")) else v)
                   for k, v in chunk.items()}
            a, b = run_remap(sub)
            ids_a[r], ids_b[r] = a, b

        # ── 3) 배열 → 객체 그래프 재구성 ────────────────────────────────
        objs: Dict[int, Any]  = {}
        meta: Dict[int, Dict[str, Any]] = {}

        for i in range(tot):
            code = int(op[i])
            if code == 0:            # NEW
                vid = ids_a[i]
                objs[vid] = {}       # 임시, 타입 정보 생략(필요시 class_name 처리)
            elif code == 1:          # APPEND
                p = ids_a[i]; c = ids_b[i]
                parent, child = objs[p], objs[c]
                if isinstance(parent, list):
                    parent.append(child)
                else:
                    # dict 부모라 가정 → 키 이름은 없으므로 인덱스로 보관
                    parent[str(len(parent))] = child
            # REPEAT_BEG / END 는 flatten 토큰이므로 스킵

        root = payload.generative_payload["root_id"]
        root_idx = int(root.replace("n", ""), 10)  # n00000001 → 1
        return objs[root_idx]





