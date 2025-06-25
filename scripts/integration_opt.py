#!/usr/bin/env python
"""
Step 6  ▸ 통합 검증 스크립트
---------------------------------
• v1.0 / v1.1 인코딩-디코딩 round-trip
• Numba ON / OFF 모두 실행
• 3종 데이터세트(단순·연속·희소) 자동 생성
---------------------------------
Usage:  python scripts/integration_opt.py
"""
import os, sys, json, random
from time import perf_counter
from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder

CASES = {
    "const":  [1]*32,
    "range":  list(range(40)),
    "sparse": [9 if i in (5,17) else 0 for i in range(40)],
}

def run_case(flag: str):
    os.environ["GPE_USE_NUMBA"] = flag
    enc = GPEEncoder(enable_optimization=True, include_fallback=False)
    dec = GPEDecoder()
    for name, data in CASES.items():
        blob = enc.encode(data)
        assert blob.generative_payload["version"] == "gpe.v1.1"
        assert dec.decode(blob) == data, f"Mismatch ({flag}) {name}"
    print(f"✔  v1.1 round-trip OK  (NUMBA={flag})")

if __name__ == "__main__":
    try:
        for f in ("false", "true"):
            run_case(f)
        # v1.0 호환 확인
        enc = GPEEncoder(enable_optimization=False, include_fallback=False)
        dec = GPEDecoder()
        raw = enc.encode(CASES["range"])
        assert raw.generative_payload["version"] == "gpe.v1.0"
        assert dec.decode(raw) == CASES["range"]
        print("✔  v1.0 호환 OK")
    except AssertionError as e:
        print("❌", e); sys.exit(1)
