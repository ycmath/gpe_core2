"""
tests/perf_bench.py
-------------------

단순 벤치마크용 테스트.

* 5만 레코드(≈ 0.6 MB) JSON 을 만들어
  - encode 1회
  - decode 1회
* 백엔드별 벽시계 시간(ms)을 측정해 출력
* CPU-Numba ON 은 CPU-Numba OFF 보다 **최소 1.5×** 빨라야 통과
  (Numba 미설치 환경이면 비교 항목 skip)

pytest 사용 방법
----------------
$ pytest -k perf -s
"""

import time
import json
import random
import importlib

import pytest

from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder, _NUMBA_AVAIL
from gpe_core.decoder import GPEDecoder as CPUDec
from gpe_core.decoder_mp import GPEDecoderMP
from gpe_core.decoder_mp_shm import GPEDecoderMP_ShMem

BACKENDS = {
    "cpu": CPUDec,
    "mp": GPEDecoderMP,
    "mp-shm": GPEDecoderMP_ShMem,
}

# optional GPU back-ends
for name in ("gpu-stream", "gpu-full"):
    try:
        dec_cls = importlib.import_module(f"gpe_core.gpu.stream_decoder{'' if name=='gpu-stream' else '_meta'}").__dict__[{
            "gpu-stream": "GPEDecoderGPUStream",
            "gpu-full": "GPEDecoderGPUStreamFull",
        }[name]]
        BACKENDS[name] = dec_cls
    except Exception:
        pass


def _make_dataset(n=50_000):
    rand = random.Random(0)
    return [
        {"x": rand.randint(0, 9),
         "y": [rand.randint(0, 9) for _ in range(5)]}
        for _ in range(n)
    ]


@pytest.mark.perf
@pytest.mark.parametrize("backend", list(BACKENDS))
def test_perf(backend):
    data = _make_dataset()
    enc = GPEEncoder(include_fallback=False)
    dec = BACKENDS[backend]()

    t0 = time.perf_counter()
    payload = enc.encode(data)
    enc_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dec.decode(payload)
    dec_ms = (time.perf_counter() - t0) * 1000

    print(f"[perf] backend={backend:8}  encode={enc_ms:7.1f} ms  decode={dec_ms:7.1f} ms")

    # 간단 성능 어썰션: Numba ON 은 OFF 대비 ≥1.5× 빨라야 한다
    if backend == "cpu" and _NUMBA_AVAIL:
        dec_nojit = GPEDecoder()
        dec_nojit._apply_numba = dec_nojit._apply_py  # 강제로 JIT 경로 끔
        t0 = time.perf_counter()
        dec_nojit.decode(payload)
        nojit_ms = (time.perf_counter() - t0) * 1000
        assert dec_ms <= nojit_ms / 1.5, f"Numba path too slow: {dec_ms:.1f} vs {nojit_ms:.1f}"
