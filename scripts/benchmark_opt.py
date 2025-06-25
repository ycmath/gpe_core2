#!/usr/bin/env python
"""
Step 7 ▸ v1.0 vs v1.1 성능 벤치마크 & 시각화

$ python scripts/benchmark_opt.py -b gpu-stream --rows 10000 50000
"""

from __future__ import annotations
import argparse, time, csv, random, os
from pathlib import Path
import orjson as json
import matplotlib.pyplot as plt

from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder
from gpe_core.cli     import _get_decoder          # backend factory

BENCH_DIR = Path("benchmarks"); BENCH_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────────
def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return (time.perf_counter() - t0) * 1000, out


def _bench_once(data, backend: str = "cpu", repeat: int = 3) -> dict[str, float]:
    enc0 = GPEEncoder(enable_optimization=False, include_fallback=False)
    enc1 = GPEEncoder(enable_optimization=True,  include_fallback=False)

    dec  = GPEDecoder() if backend == "cpu" else _get_decoder(backend)

    t_e0, blob0 = _timed(enc0.encode, data)
    t_e1, blob1 = _timed(enc1.encode, data)

    # ★ 디코드 타임은 repeat 회 평균
    t_d0 = sum(_timed(dec.decode, blob0)[0] for _ in range(repeat)) / repeat
    t_d1 = sum(_timed(dec.decode, blob1)[0] for _ in range(repeat)) / repeat

    rules0 = sum(len(s["rules"]) for s in blob0.generative_payload["seeds"])
    rules1 = len(blob1.generative_payload["seeds"])         # v1.1 flat list
    size0  = len(json.dumps(blob0.generative_payload, separators=(",",":")))
    size1  = len(json.dumps(blob1.generative_payload, separators=(",",":")))

    stats  = blob1.generative_payload.get("optimization", {})
    return dict(
        rules_v10 = rules0,           rules_v11 = rules1,
        bytes_v10 = size0,            bytes_v11 = size1,
        enc_ms_v10 = t_e0,            enc_ms_v11 = t_e1,
        dec_ms_v10 = t_d0,            dec_ms_v11 = t_d1,
        backend = backend,
        **stats,
    )


def _plot(csv_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    x  = df["rows"]
    plt.figure(figsize=(6,4))
    plt.plot(x, df["bytes_v10"]/1024, "-o", label="v1.0 size (KB)")
    plt.plot(x, df["bytes_v11"]/1024, "-o", label="v1.1 size (KB)")
    plt.xlabel("Records"); plt.ylabel("Payload size (KB)")
    plt.title("Payload size vs record count"); plt.legend(); plt.tight_layout()
    out = csv_path.with_suffix(".png"); plt.savefig(out); print("➜", out)


# ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="GPE v1.0 vs v1.1 benchmark")
    ap.add_argument("-i", "--input", help="input JSON file (omit → synthetic)")
    ap.add_argument("-b", "--backend", default="cpu",
                    choices=["cpu", "mp", "gpu-stream"])
    ap.add_argument("--rows", type=int, nargs="+", default=[100_000],
                    help="synthetic rows when --input omitted")
    ap.add_argument("--repeat", type=int, default=3,
                    help="decode timing repeat count")
    args = ap.parse_args()

    results: list[dict[str,float]] = []
    for n in args.rows:
        if args.input:
            with open(args.input, "rb") as fp:
                data = json.loads(fp.read())
        else:                               # ★ 레코드 수별 synthetic
            data = [random.randint(0, 9) for _ in range(n)]

        bench_row          = _bench_once(data, args.backend, args.repeat)
        bench_row["rows"]  = n
        results.append(bench_row)

    # CSV 저장
    csv_path = BENCH_DIR / f"bench_{args.backend}.csv"
    with csv_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print("✓ CSV saved →", csv_path)

    _plot(csv_path)


if __name__ == "__main__":
    main()
