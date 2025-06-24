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
