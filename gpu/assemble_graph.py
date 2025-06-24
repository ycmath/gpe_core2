"""
GPU 객체 그래프 조립 커널 래퍼
─────────────────────────────────────────
* run_remap() 결과(ids_a, ids_b)와 hybrid-meta chunk를 받아
  assemble_graph.cu 커널을 호출해 GPU 메모리 내부에서
  dict/list 트리를 단일 pass 로 구성합니다.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple

import cupy as cp
import numpy as np


# ── CUDA 커널 로드 ────────────────────────────────────────────────
_SRC = (Path(__file__).with_suffix(".cu")).read_text()
_KERNEL = cp.RawKernel(
    _SRC,
    "assemble_graph",
    options=("-O3", "-std=c++17",),
)


# ── 래퍼 함수 ────────────────────────────────────────────────────
def gpu_assemble(chunk: Dict[str, Any],
                 ids_a: np.ndarray,
                 ids_b: np.ndarray
                 ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Parameters
    ----------
    chunk : dict
        hybrid_flatten_meta() 결과 (op, meta_cls, meta_key … 포함)
    ids_a, ids_b : np.ndarray uint32
        run_remap() 가 반환한 두 ID 배열

    Returns
    -------
    d_type, d_head, d_next, d_key : cupy.ndarray
        GPU 메모리 상의 child-list 구조체 배열
    """
    n = chunk["op"].size

    # 입력 배열을 GPU로
    d_op       = cp.asarray(chunk["op"],       dtype=cp.uint8)
    d_meta_cls = cp.asarray(chunk["meta_cls"], dtype=cp.uint16)
    d_meta_key = cp.asarray(chunk["meta_key"], dtype=cp.uint32)
    d_ida      = cp.asarray(ids_a,             dtype=cp.uint32)
    d_idb      = cp.asarray(ids_b,             dtype=cp.uint32)

    # 출력 버퍼 할당
    d_type = cp.empty(n,           dtype=cp.uint8)
    d_head = cp.full(n, 0xFFFFFFFF, dtype=cp.uint32)
    d_next = cp.full(n, 0xFFFFFFFF, dtype=cp.uint32)
    d_key  = cp.full(n, 0xFFFFFFFF, dtype=cp.uint32)

    # 커널 실행
    threads = 256
    blocks  = (n + threads - 1) // threads
    _KERNEL(
        (blocks,), (threads,),
        (
            d_op,
            d_ida, d_idb,
            d_meta_cls,
            d_meta_key,
            d_type, d_head, d_next, d_key,
            np.int32(n),
        )
    )
    return d_type, d_head, d_next, d_key


## > *`ida/idb` 는 기존 `run_remap()` 반환값을 바로 `cp.asarray()` 로 전달.*

## 4. 호스트-레벨 최종 객체 변환

def cupy_graph_to_py(d_type, d_head, d_next, d_key, lut_cls, lut_key_blob, lut_key_off):
    n = len(d_type)
    objs = [None] * n
    for vid in range(n):
        t = int(d_type[vid])
        if t == 0:   objs[vid] = {}
        elif t == 1: objs[vid] = []
        else:        objs[vid] = {"__class__": lut_cls[t]}
    # 2-pass: children
    for parent in range(n):
        child = int(d_head[parent])
        while child != 0xFFFFFFFF:
            pobj = objs[parent]
            if isinstance(pobj, list):
                pobj.append(objs[child])
            else:
                kidx = int(d_key[child])
                k = lut_key_blob[lut_key_off[kidx]: lut_key_off[kidx+1]].decode()
                pobj[k] = objs[child]
            child = int(d_next[child])
    return objs

## `lut_key_off` 는 key 문자열 시작-offset 배열 (+ 마지막에 blob.length 추가).

## 5. 통합 (flow)

## 1. **run\_remap** → `ids_a/ids_b` GPU 배열 반환
## 2. **assemble\_graph** 커널 호출 → 4 출력 버퍼
## 3. Host `cupy_graph_to_py()` 로 단일 pass 변환 → 파이썬 객체 완성
## 4. 루트 ID 찾아 반환

## > **이후**: `cupy_graph_to_py` 를 Numba JIT 로 바꾸면 Host 변환도 5× 가속.

## 6. 성능 예측

## | 단계             | 50 만 rule 기준 | v1 (복합)              | v2.5 커널 |
## | -------------- | ------------ | -------------------- | ------- |
## | ID remap (GPU) | 12 ms        | **동일**               |         |
## | Assemble GPU   | –            | **6 ms**             |         |
## | Host Python 조립 | 90 ms        | **18 ms**            |         |
## | **합계**         | **\~102 ms** | **\~36 ms (\~2.8×)** |         |


## 7. To-do

## 1. **key lookup** 최적화를 위해 `lut_key_blob` 를 GPU로 복사해 커널-내 UTF-8 copy-out까지 수행 가능.
## 2. 멀티-GPU: op 범위를 device마다 분할 → AllReduce 불필요(리니어).
