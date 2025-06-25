#!/usr/bin/env python
"""
Step 7 ▸ 성능 벤치마크 + 시각화
---------------------------------
• 입력 JSON → ① v1.0 , ② v1.1(optimizer) 로 인코딩 후
  - 규칙 수, 바이트 크기, 인코드/디코드 시간(ms) 비교
• 결과를 `bench_results.csv` 와 `bench_plot.png` 로 저장
---------------------------------
Usage:  python scripts/benchmark_opt.py sample.json
"""
import json, argparse, time, csv
from pathlib import Path
import matplotlib.pyplot as plt

from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder

def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return (time.perf_counter() - t0)*1000, out

def bench(data):
    enc0 = GPEEncoder(enable_optimization=False, include_fallback=False)
    enc1 = GPEEncoder(enable_optimization=True,  include_fallback=False)
    dec   = GPEDecoder()

    t_e0, blob0 = timed(enc0.encode, data)
    t_d0, _     = timed(dec.decode, blob0)
    t_e1, blob1 = timed(enc1.encode, data)
    t_d1, _     = timed(dec.decode, blob1)

    rules0 = sum(len(s["rules"]) for s in blob0.generative_payload["seeds"])
    rules1 = len(blob1.generative_payload["seeds"])           # v1.1 은 flat list
    size0  = len(json.dumps(blob0.generative_payload))
    size1  = len(json.dumps(blob1.generative_payload))

    return {
        "rules_v1.0": rules0, "rules_v1.1": rules1,
        "bytes_v1.0": size0,  "bytes_v1.1": size1,
        "enc_ms_v1.0": t_e0,  "enc_ms_v1.1": t_e1,
        "dec_ms_v1.0": t_d0,  "dec_ms_v1.1": t_d1,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPE v1.0 vs v1.1 성능 벤치마크")
    ap.add_argument("-i", "--input", help="input JSON file (omit → synthetic)")
    ap.add_argument("-o", "--out-prefix", default="bench", help="output file prefix")
    ap.add_argument("--rows", type=int, default=200_000,
                    help="synthetic rows when --input omitted")
    args = ap.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    else:
        from random import randint
        data = [randint(0, 9) for _ in range(args.rows)]
    res  = bench(data)

    # CSV 저장
    csv_path = f"{args.out_prefix}_results.csv"
    png_path = f"{args.out_prefix}_plot.png"

    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(res.keys()); w.writerow(res.values())

    # Plot 저장 (압축률, 규칙 감소율)
    labels = ["Rule count", "Payload bytes"]
    v0 = [res["rules_v1.0"], res["bytes_v1.0"]]
    v1 = [res["rules_v1.1"], res["bytes_v1.1"]]
    x  = range(len(labels))
    plt.figure(figsize=(6,4))
    plt.bar(x, v0, label="v1.0")
    plt.bar(x, v1, label="v1.1", bottom=v0, alpha=0.6)
    plt.xticks(x, labels); plt.ylabel("count / bytes")
    plt.title("GPE Optimizer Gain"); plt.legend()
    plt.tight_layout(); plt.savefig(png_path)
    print(f"✓  Results saved → {csv_path}, {png_path}")
