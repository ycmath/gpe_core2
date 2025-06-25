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
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="input JSON file")
    args = ap.parse_args()

    data = json.load(open(args.json))
    res  = bench(data)

    # CSV 저장
    with open("bench_results.csv", "w", newline="") as fp:
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
    plt.tight_layout(); plt.savefig("bench_plot.png")   # ← Colab-safe 상대경로
    print("✓  Results → bench_results.csv / bench_plot.png")
