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
