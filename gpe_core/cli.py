"""Command‑line interface: **gpe encode / decode / bench**"""
from __future__ import annotations

import argparse, json, sys
from typing import Optional
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
    enc = GPEEncoder(
        include_fallback=not ns.no_fallback,
        enable_optimization=ns.opt,          # ← NEW
    )

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
    """Benchmark encode → decode with optional progress bar."""
    from random import randint
    try:
        from tqdm import tqdm  # type: ignore
        _tqdm = tqdm  # real progress bar
    except ModuleNotFoundError:  # pragma: no cover
        def _tqdm(iterable=None, **kw):
            return iterable  # no‑op iterator

    # synthetic dataset
    data = [
        {"x": randint(0, 9), "y": [randint(0, 9) for _ in range(5)]}
        for _ in range(ns.n)
    ]

    enc = GPEEncoder(
        include_fallback=False,
        enable_optimization=ns.opt,
    )
    dec = _get_decoder(ns.backend)

    steps = ["encode", "decode"]
    for step in _tqdm(steps, desc="Benchmark", disable=not ns.progress):
        if step == "encode":
            t0 = time.perf_counter()
            payload = enc.encode(data)
            enc_ms = (time.perf_counter() - t0) * 1000
        else:
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
    g_opt = sp.add_mutually_exclusive_group()
    g_opt.add_argument("--opt",     dest="opt", action="store_true",
                       help="enable rule-optimizer (v1.1, default)")
    g_opt.add_argument("--no-opt",  dest="opt", action="store_false",
                       help="disable optimizer (v1.0 payload)")
    sp.set_defaults(opt=True)
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
    sp.add_argument("--progress", action="store_true", help="show progress bar with tqdm")
    g2 = sp.add_mutually_exclusive_group()
    g2.add_argument("--opt",    dest="opt", action="store_true",
                    help="benchmark v1.1 encoder (default)")
    g2.add_argument("--no-opt", dest="opt", action="store_false",
                    help="benchmark v1.0 encoder")
    sp.set_defaults(opt=True)
    sp.set_defaults(func=cmd_bench)

    ns = ap.parse_args()
    ns.func(ns)

if __name__ == "__main__":
    main()


## tqdm 가 설치되어 있으면 실제 진행률 바, 없으면 no-op.
## 사용법:
## bash
## gpe bench --n 20000 --progress
## tqdm 미설치 환경에서도 문제없이 작동합니다.
