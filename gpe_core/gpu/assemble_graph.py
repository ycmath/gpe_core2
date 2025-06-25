from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import cupy as cp
import numpy as np

# ── CUDA 커널 로더 ───────────────────────────────────────────
def _compile(src_path: str, func: str):
    src = (Path(__file__).with_name(src_path)).read_text()
    try:
        # 가장 보수적으로 옵션을 아예 주지 않는다
        return cp.RawKernel(src, func)          # options=() ← 기본
    except cp.cuda.compiler.CompileException as e:
        raise RuntimeError(f"CUDA compile failed for {src_path}") from e

def _load_kernel():
    try:
        return _compile("assemble_graph_v2.cu", "assemble_graph_v2")
    except Exception:
        return _compile("assemble_graph.cu",    "assemble_graph")

_KERNEL = _load_kernel()


# ── 래퍼 함수 ────────────────────────────────────────────────────────────
def gpu_assemble(
    chunk: Dict[str, Any],
    ids_a: np.ndarray,
    ids_b: np.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Assemble GPU child-lists; returns (d_type, d_head, d_next, d_key) in GPU mem.
    """
    n = chunk["op"].size

    d_op       = cp.asarray(chunk["op"],       dtype=cp.uint8)
    d_meta_cls = cp.asarray(chunk["meta_cls"], dtype=cp.uint16)
    d_meta_key = cp.asarray(chunk["meta_key"], dtype=cp.uint32)
    d_ida      = cp.asarray(ids_a,             dtype=cp.uint32)
    d_idb      = cp.asarray(ids_b,             dtype=cp.uint32)
    # key LUT (bytes) → uint8 GPU buffer  ─ CuPy 12.x safe path
    key_blob = "|".join(chunk["lut_key"]).encode() + b"|"
    d_blob   = cp.asarray(                 # bytes → NumPy → CuPy
        np.frombuffer(key_blob, dtype=np.uint8)
    )
    key_off  = np.fromiter(
        (0, *np.cumsum([len(k)+1 for k in chunk["lut_key"]])),
        dtype=np.uint32,
    )
    d_off = cp.asarray(key_off, dtype=cp.uint32)
    d_type = cp.empty(n,           dtype=cp.uint8)
    d_head = cp.full(n, 0xFFFFFFFF, dtype=cp.uint32)
    d_next = cp.full(n, 0xFFFFFFFF, dtype=cp.uint32)
    d_key  = cp.full(n, 0xFFFFFFFF, dtype=cp.uint32)

    threads = 256
    blocks  = (n + threads - 1) // threads
# 커널 호출 인자에 추가
    _KERNEL(
        (blocks,), (threads,),
        (
            d_op, d_ida, d_idb, d_meta_cls, d_meta_key,
            d_blob, d_off,          # <── new
            d_type, d_head, d_next, d_key,
            np.int32(n),
        ),
    )
    return d_type, d_head, d_next, d_key



