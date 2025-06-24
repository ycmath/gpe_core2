"""
tests/test_edge_cases.py
────────────────────────
1) 빈 Repeat 구조
2) 깊이 10 중첩 Repeat
3) GPU 메모리 부족(OOM) 시 graceful-fallback 확인
"""

import json
import pytest

from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder, GPEDecodeError

# ----------------------------- Helper ---------------------------------
def _enc(data, *, fallback=False):
    return GPEEncoder(include_fallback=fallback).encode(data)

# ----------------------------------------------------------------------
def test_empty_repetition():
    """RepeatRule(count=0) 처리 검증"""

    payload = {
        "root_id": "n00000000",
        "seeds": [{"rules": [{"op_code": "REPEAT", "count": 0, "instruction": []}]}],
    }
    with pytest.raises(KeyError):
        GPEDecoder().decode(
            payload | {"payload_type": "gpe.v1", "generative_payload": payload}
        )


def test_deeply_nested_repeats():
    """깊이 10 단계 Repeat 복원"""
    # build nested list [[[[1]]]]
    lvl, tree = 10, 1
    for _ in range(lvl):
        tree = [tree]
    payload = _enc(tree)
    assert GPEDecoder().decode(payload) == tree


# ----------------------------------------------------------------------
# GPU OOM test – skip if cupy unavailable
pytest.importorskip("cupy", reason="Cupy not installed; skipping GPU test")

try:
    from gpe_core.gpu.stream_decoder import GPEDecoderGPUStream
except Exception:
    GPU_AVAILABLE = False
else:
    GPU_AVAILABLE = True


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
def test_gpu_memory_overflow(monkeypatch):
    """rows_per 가 OOM 방어 로직으로 줄어드는지 확인 (simulate low VRAM)."""

    data = [{"x": 1}] * 2_000_000  # 대용량
    payload = _enc(data)

    # 강제로 free mem report를 16 MB 로 낮춤
    import cupy as cp

    def fake_mem_info():
        return 16 * 1024 * 1024, 16 * 1024 * 1024

    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", fake_mem_info)

    dec = GPEDecoderGPUStream(vram_frac=0.9)
    out = dec.decode(payload)

    assert out == data
