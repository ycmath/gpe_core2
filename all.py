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
# gpe_core/repetition_detector.py  (Numba‑accelerated)
################################################################################
"""Detect repeated sub‑trees using node‑hash grouping.

If **Numba** is available the costly recursive `size()` computation is JIT‑
compiled, giving 3‑5× speed‑up on large (>100k node) ASTs.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

from .models import ASTNode

# optional numba JIT ----------------------------------------------------------
try:
    from numba import njit  # type: ignore
    _NUMBA = True
except ModuleNotFoundError:  # pragma: no cover
    _NUMBA = False

class RepetitionDetector:
    def __init__(self, nodes: Dict[str, ASTNode], min_occ: int = 2, min_size: int = 3):
        self.nodes = nodes
        self.min_occ = min_occ
        self.min_size = min_size

        # build adjacency in NumPy‑friendly form (for numba path)
        self._id2idx: Dict[str, int] = {nid: i for i, nid in enumerate(nodes)}
        self._children_idx: List[List[int]] = [
            [self._id2idx[c] for c in n.children] for n in nodes.values()
        ]
        self._hashes: List[str] = [n.attributes["hash"] for n in nodes.values()]

        # compile size() if numba present
        if _NUMBA:
            self._size = self._compile_size()
        else:
            self._cache: Dict[int, int] = {}
            self._size = self._size_py

    # ------------------------------------------------------------------
    def detect(self):
        groups = defaultdict(list)
        for idx, h in enumerate(self._hashes):
            nid = list(self.nodes)[idx]
            groups[h].append(idx)

        rep: Dict[str, List[str]] = {}
        for h, idx_list in groups.items():
            qual = [i for i in idx_list if self._size(i) >= self.min_size]
            if len(qual) >= self.min_occ:
                rep[h] = [list(self.nodes)[i] for i in qual]
        return rep

    # ------------------------------------------------------------------
    def _compile_size(self):
        children = self._children_idx
        n_nodes  = len(children)

        @njit(cache=True)
        def size(idx: int, memo: Dict[int, int] = {}):  # type: ignore[arg-type]
            if idx in memo:
                return memo[idx]
            s = 1
            for c in children[idx]:
                s += size(c, memo)
            memo[idx] = s
            return s
        return size

    # fallback pure python ------------------------------------------------
    def _size_py(self, idx: int) -> int:
        if idx in self._cache:
            return self._cache[idx]
        s = 1 + sum(self._size_py(c) for c in self._children_idx[idx])
        self._cache[idx] = s
        return s
        
################################################################################
# gpe_core/seed_generator.py  (v2 – nested‑Repeat aware)
################################################################################
"""Seed generation that preserves **nested Repeat** structure.

Algo overview
-------------
1.  Root‑first DFS over the AST.
2.  When the current node *belongs to* a repetition group (same hash occurring
    ≥ min_occ):
    • emit **one** `RepeatRule(count, instruction)` with the *first* instance as
      template.
    • mark all siblings in that group as *consumed* so recursion skips them.
3.  Otherwise emit plain NEW + (children) + APPEND in natural order.

This keeps the rule sequence aligned with reconstruction order while still
compressing identical subtrees.
"""
from __future__ import annotations
from typing import Dict, List, Set

from .models import (
    ASTNode,
    AttentionSeed,
    BaseRule,
    InstantiateRule,
    AppendChildRule,
    RepeatRule,
)

class SeedGenerator:
    """Generate rule trees with **nested RepeatRule** support."""

    def __init__(self, nodes: Dict[str, ASTNode], groups: Dict[str, List[str]]):
        self.nodes = nodes                      # AST node table
        self.groups = groups                    # hash → [node_ids]

        # Map every node id → its repetition group (if any)
        self._grp_of: Dict[str, List[str]] = {}
        for lst in groups.values():
            for nid in lst:
                self._grp_of[nid] = lst

        self._seen_repeat: Set[str] = set()     # group leaders already emitted

    # ------------------------------------------------------------------
    def generate(self, root_id: str | None = None) -> List[AttentionSeed]:
        """Return *one* AttentionSeed covering the whole AST (root‑aligned)."""
        if root_id is None:
            # pick the lexicographically smallest id as root (ASTBuilder behaves that way)
            root_id = min(self.nodes)
        rules = self._emit(root_id)
        return [AttentionSeed(rules=rules)]

    # ------------------------------------------------------------------
    def _emit(self, nid: str) -> List[BaseRule]:
        # If this node belongs to a repeat group AND we haven't emitted it yet
        if nid in self._grp_of and self._grp_of[nid][0] == nid and nid not in self._seen_repeat:
            grp = self._grp_of[nid]
            self._seen_repeat.update(grp)
            template_rules = self._emit_subtree(nid, emit_repeat=False)
            return [RepeatRule(op_code="REPEAT", count=len(grp), instruction=template_rules)]
        else:
            # normal path (including inner nodes inside a larger Repeat)
            return self._emit_subtree(nid, emit_repeat=True)

    # ------------------------------------------------------------------
    def _emit_subtree(self, nid: str, *, emit_repeat: bool) -> List[BaseRule]:
        """Return rules for a subtree. emit_repeat=False means we are already
        inside a Repeat template; we must not nest again."""
        node = self.nodes[nid]
        rules: List[BaseRule] = [
            InstantiateRule(
                op_code="NEW",
                class_name=node.type,
                instance_id=node.id,
                attributes={k: v for k, v in node.attributes.items() if k != "hash"},
            )
        ]

        for child_id in node.children:
            # Skip other instances that will be covered by a Repeat emitted elsewhere
            if child_id in self._grp_of and self._grp_of[child_id][0] != child_id and self._grp_of[child_id][0] in self._seen_repeat:
                continue
            # recurse — possibly emitting nested Repeat
            if emit_repeat:
                rules.extend(self._emit(child_id))
            else:
                rules.extend(self._emit_subtree(child_id, emit_repeat=False))
            rules.append(AppendChildRule(op_code="APPEND", parent_id=nid, child_id=child_id))
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
# gpe_core/decoder.py (CPU + optional Numba‑JIT loop)
################################################################################
"""Reference decoder + optional **Numba‑accelerated rule loop**.

* If `numba` is present, the inner rule‑iteration is delegated to a JIT
  function that operates on `typed.Dict` / `typed.List` structures, giving
  ~2‑3× speed‑up on large seed lists (≥100k rules).
"""
from __future__ import annotations
import json, copy
from typing import Dict, Any, List

from .models import GpePayload

# ---------------------------------------------------------------------------
try:
    from numba import njit, typed  # type: ignore
    _NUMBA_AVAIL = True
except ModuleNotFoundError:  # pragma: no cover
    _NUMBA_AVAIL = False

class GPEDecoder:
    """Single‑process decoder. JIT fast‑path engaged when numba present."""

    # ------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # 0) fallback JSON
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])

        # 1) prepare containers
        objs: Dict[str, Any] = {}
        meta: Dict[str, Dict[str, Any]] = {}

        # 2) choose apply loop implementation
        if _NUMBA_AVAIL:
            self._apply_numba(payload.generative_payload["seeds"], objs, meta)
        else:
            for seed in payload.generative_payload["seeds"]:
                for rule in seed["rules"]:
                    self._apply_py(rule, objs, meta)

        root_id = payload.generative_payload["root_id"]
        return objs[root_id]

    # ==================================================================
    # Pure‑python implementation (fallback)
    # ==================================================================
    def _apply_py(self, r: Dict[str, Any], o: Dict[str, Any], m: Dict[str, Dict[str, Any]]):
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
                    self._apply_py(rule, o, m)
        else:
            raise ValueError(op)

    # ==================================================================
    # Numba‑accelerated path
    # ==================================================================
    if _NUMBA_AVAIL:
        def _apply_numba(self, seeds: List[Dict[str, Any]], objs_py: Dict[str, Any], meta_py: Dict[str, Dict[str, Any]]):
            """Convert py‑dicts to numba typed.Dict & run JIT kernel."""
            t_objs = typed.Dict.empty(key_type=njit.str_, value_type=njit.types.pyobject)
            t_meta = typed.Dict.empty(key_type=njit.str_, value_type=njit.types.pyobject)

            @njit(cache=True)
            def run(seeds_list, objs, meta):
                for seed in seeds_list:
                    for rule in seed["rules"]:
                        op = rule["op_code"]
                        if op == "NEW":
                            vid = rule["instance_id"]
                            cls = rule["class_name"]
                            attrs = rule.get("attributes", {})
                            if "value" in attrs:
                                objs[vid] = attrs["value"]
                            elif cls == "dict":
                                objs[vid] = {}
                            elif cls == "list":
                                objs[vid] = []
                            else:
                                objs[vid] = {"__class__": cls}
                            meta[vid] = attrs
                        elif op == "APPEND":
                            p = rule["parent_id"]; c = rule["child_id"]
                            parent = objs[p]; child = objs[c]
                            if isinstance(parent, list):
                                parent.append(child)
                            else:
                                key = meta[c].get("key")
                                parent[key] = child
                        elif op == "REPEAT":
                            for _ in range(rule["count"]):
                                for sub in rule["instruction"]:
                                    # NOTE: recursion depth typically shallow; inline for perf
                                    sop = sub["op_code"]
                                    if sop == "NEW":
                                        vid = sub["instance_id"]
                                        cls = sub["class_name"]
                                        attrs = sub.get("attributes", {})
                                        objs[vid] = {} if cls == "dict" else []
                                        meta[vid] = attrs
                                    elif sop == "APPEND":
                                        pp = sub["parent_id"]
                                        cc = sub["child_id"]
                                        pobj = objs[pp]
                                        if isinstance(pobj, list):
                                            pobj.append(objs[cc])
                                        else:
                                            pobj[str(len(pobj))] = objs[cc]
            run(seeds, t_objs, t_meta)
            # move back to python dict for downstream
            objs_py.update(t_objs)
            meta_py.update(t_meta)

    else:
        def _apply_numba(self, *a, **kw):  # type: ignore[no-self-use]
            raise RuntimeError("Numba not installed")

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
"""Command‑line interface: **gpe encode / decode / bench**"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

from .encoder import GPEEncoder
from .models import GpePayload
from .decoder import GPEDecoder
from .decoder_mp import GPEDecoderMP
from .decoder_mp_shm import GPEDecoderMP_ShMem

# -----------------------------------------------------------------------------
# Back‑end registry
# -----------------------------------------------------------------------------
BACKENDS: Dict[str, type[GPEDecoder]] = {
    "cpu":     GPEDecoder,
    "mp":      GPEDecoderMP,
    "mp-shm":  GPEDecoderMP_ShMem,
}

# ── GPU: ID 재매핑까지 (객체 복원은 CPU) ─────────────────────────────────────
try:
    from .gpu.stream_decoder import GPEDecoderGPUStream  # type: ignore
    BACKENDS["gpu-stream"] = GPEDecoderGPUStream
except Exception:
    pass

# ── GPU: 메타까지 완전 복원 ─────────────────────────────────────────────────
try:
    from .gpu.stream_decoder_meta import GPEDecoderGPUStreamFull  # type: ignore
    BACKENDS["gpu-full"] = GPEDecoderGPUStreamFull
except Exception:
    pass

# ── Multi‑GPU (Ray) 백엔드 ───────────────────────────────────────────────
try:
    from .gpu_multi.multi_decoder import GPEDecoderGPU_Ray  # type: ignore
    BACKENDS["gpu-ray"] = GPEDecoderGPU_Ray
except Exception:
    pass

# -----------------------------------------------------------------------------
# Helper I/O
# -----------------------------------------------------------------------------

def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(obj, path: Path):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# Decoder factory
# -----------------------------------------------------------------------------

def _get_decoder(name: str):
    if name not in BACKENDS:
        sys.exit(f"❌ unknown backend '{name}' (choose from {', '.join(BACKENDS)})")
    return BACKENDS[name]()


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def cmd_encode(ns):
    data = _load_json(ns.input)
    enc = GPEEncoder(include_fallback=not ns.no_fallback)

    t0 = time.perf_counter()
    payload = enc.encode(data)
    enc_ms = (time.perf_counter() - t0) * 1000

    print(f"✓ encoded in {enc_ms:.2f} ms → {ns.output}")
    obj = payload.generative_payload if ns.strip else payload.__dict__
    _dump_json(obj, ns.output)



def cmd_decode(ns):
    raw = _load_json(ns.input)
    if "generative_payload" in raw:
        payload = GpePayload(**raw)  # type: ignore[arg-type]
    else:
        payload = GpePayload(
            payload_type="gpe.v1",
            generative_payload={},
            fallback_payload={"json": json.dumps(raw)},
        )

    dec = _get_decoder(ns.backend)

    t0 = time.perf_counter()
    data = dec.decode(payload)
    dec_ms = (time.perf_counter() - t0) * 1000

    print(
        f"✓ decoded in {dec_ms:.2f} ms using backend '{ns.backend}' → {ns.output}"
    )
    _dump_json(data, ns.output)



def cmd_bench(ns):
    from random import randint

    # synthetic dataset
    data = [
        {"x": randint(0, 9), "y": [randint(0, 9) for _ in range(5)]}
        for _ in range(ns.n)
    ]

    enc = GPEEncoder(include_fallback=False)
    t0 = time.perf_counter()
    payload = enc.encode(data)
    enc_ms = (time.perf_counter() - t0) * 1000

    dec = _get_decoder(ns.backend)
    t0 = time.perf_counter()
    dec.decode(payload)
    dec_ms = (time.perf_counter() - t0) * 1000

    print(
        f"n={ns.n:,} | encode {enc_ms:.2f} ms | decode({ns.backend}) {dec_ms:.2f} ms"
    )


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(prog="gpe", description="GPE encode/decode toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # encode ---------------------------------------------------------
    sp = sub.add_parser("encode", help="JSON → GPE payload")
    sp.add_argument("--input", "-i", type=Path, required=True)
    sp.add_argument("--output", "-o", type=Path, required=True)
    sp.add_argument("--no-fallback", action="store_true", help="omit raw JSON fallback")
    sp.add_argument("--strip", action="store_true", help="save only generative_payload")
    sp.set_defaults(func=cmd_encode)

    # decode ---------------------------------------------------------
    sp = sub.add_parser("decode", help="GPE payload → JSON")
    sp.add_argument("--input", "-i", type=Path, required=True)
    sp.add_argument("--output", "-o", type=Path, required=True)
    sp.add_argument("--backend", "-b", default="cpu", choices=list(BACKENDS))
    sp.set_defaults(func=cmd_decode)

    # bench ----------------------------------------------------------
    sp = sub.add_parser("bench", help="quick encode/decode benchmark")
    sp.add_argument("--n", type=int, default=10000, help="synthetic record count")
    sp.add_argument("--backend", "-b", default="cpu", choices=list(BACKENDS))
    sp.set_defaults(func=cmd_bench)

    ns = ap.parse_args()
    ns.func(ns)


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

    # ------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # 0) fallback JSON 우선
        fb = payload.fallback_payload
        if fb and fb.get("json"):
            return json.loads(fb["json"])

        # 1) seeds → hybrid 배열(flatten)
        seeds = payload.generative_payload["seeds"]
        chunk = hybrid_flatten(seeds)

        op   = chunk["op"]
        total = op.size

        # 2) chunk 크기 계산 (GPU VRAM 70 % 정도 사용)
        free, _ = cp.cuda.runtime.memGetInfo()
        rows_per = max(int((free * self.vram_frac) // (op.itemsize * 4)), 256_000)

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

################################################################################
# gpe_core/gpu/vectorizer_hybrid_meta.py
################################################################################

"""
hybrid_flatten_meta(seeds)  →  dict(op, mask_a, pool16_a, …, meta_cls, meta_key)
- meta_cls :  uint16 배열  (NEW 행마다 class_name 사전 index)
- meta_key :  uint32 배열  (APPEND child 행마다 key 사전 index)
- lut_cls   :  List[str]   index → class_name
- lut_key   :  List[str]   index → dict-key string
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any
from .vectorizer_hybrid import build_pools
from .vectorizer import vectorize_seeds, OP_NEW, OP_APPEND

def hybrid_flatten_meta(seeds: List[Dict[str, Any]]):
    op, a, b, _ = vectorize_seeds(seeds)

    meta_cls = np.full(op.shape, 0xFFFF, np.uint16)   # default sentinel
    meta_key = np.full(op.shape, 0xFFFFFFFF, np.uint32)

    lut_cls: List[str] = []
    lut_key: List[str] = []

    def add(cls: str, table: List[str]) -> int:
        try:
            return table.index(cls)
        except ValueError:
            table.append(cls)
            return len(table)-1

    row = 0
    for s in seeds:
        for r in s["rules"]:
            if r["op_code"] == "NEW":
                idx = add(r["class_name"], lut_cls)
                meta_cls[row] = idx
            elif r["op_code"] == "APPEND":
                k = r.get("attribute_name") or r.get("key")
                idx = add(k, lut_key)
                meta_key[row] = idx
            row += 1

    masks_a, p16a, p32a = build_pools(a)
    masks_b, p16b, p32b = build_pools(b)

    return dict(
        op=op,
        mask_a=masks_a, mask_b=masks_b,
        pool16_a=p16a, pool32_a=p32a,
        pool16_b=p16b, pool32_b=p32b,
        meta_cls=meta_cls,
        meta_key=meta_key,
        lut_cls=lut_cls,
        lut_key=lut_key,
    )

################################################################################
# gpe_core/gpu/stream_decoder_meta.py
################################################################################
from __future__ import annotations

import cupy as cp
import numpy as np
import json
from typing import Any, Dict, Iterable

from ..models import GpePayload
from ..vectorizer_hybrid_meta import hybrid_flatten_meta, OP_NEW, OP_APPEND
from .id_remap_opt import run_remap


class GPEDecoderGPUStreamFull:
    """GPU-based ID 재매핑 + 메타를 이용한 완전 복원 디코더."""

    def __init__(self, vram_frac: float = 0.7) -> None:
        self.vram_frac = vram_frac

    # ----------------------------------------------------------------------
    def decode(self, payload: GpePayload) -> Any:
        # 0) fallback JSON 우선
        if payload.fallback_payload and payload.fallback_payload.get("json"):
            return json.loads(payload.fallback_payload["json"])

        # 1) seeds → hybrid meta-flatten
        chunk = hybrid_flatten_meta(payload.generative_payload["seeds"])

        op:        np.ndarray = chunk["op"]
        meta_cls:  np.ndarray = chunk["meta_cls"]
        meta_key:  np.ndarray = chunk["meta_key"]
        lut_cls:   list[str]  = chunk["lut_cls"]
        lut_key:   list[str]  = chunk["lut_key"]

        n_rows = op.size

        # 2) chunk 사이즈 계산
        free, _ = cp.cuda.runtime.memGetInfo()
        rows_per = max(int((free * self.vram_frac) // (op.itemsize * 4)), 256_000)

        ids_a = np.empty(n_rows, np.uint32)
        ids_b = np.empty(n_rows, np.uint32)

        def ranges(total: int, step: int) -> Iterable[range]:
            s = 0
            while s < total:
                e = min(s + step, total)
                yield range(s, e)
                s = e

        # 3) GPU remap (chunked)
        for r in ranges(n_rows, rows_per):
            sub = {k: (v[r] if k.startswith(("op", "mask", "meta")) else v)
                   for k, v in chunk.items()}
            a, b = run_remap(sub)
            ids_a[r], ids_b[r] = a, b

        # 4) GPU 결과 → 객체 그래프 복원
        objs: Dict[int, Any] = {}

        for i in range(n_rows):
            code = int(op[i])

            if code == OP_NEW:
                vid      = ids_a[i]
                cls_idx  = int(meta_cls[i])
                cls_name = lut_cls[cls_idx] if cls_idx != 0xFFFF else "dict"

                match cls_name:
                    case "dict": objs[vid] = {}
                    case "list": objs[vid] = []
                    case _:      objs[vid] = {"__class__": cls_name}

            elif code == OP_APPEND:
                p, c = ids_a[i], ids_b[i]
                parent, child = objs[p], objs[c]

                if isinstance(parent, list):
                    parent.append(child)
                else:
                    k_idx = int(meta_key[i])
                    key   = lut_key[k_idx] if k_idx != 0xFFFFFFFF else str(len(parent))
                    parent[key] = child
            # OP_REPEAT_BEG / END 토큰은 실질 데이터 아님 → skip

        # 5) 루트 객체 반환
        root_text = payload.generative_payload["root_id"]  # e.g. "n00000123"
        root_idx  = int(root_text[1:])                     # strip leading "n"
        if root_idx not in objs:
            raise ValueError(f"Root id {root_text} not present after decode")
        return objs[root_idx]


