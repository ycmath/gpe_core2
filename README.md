# gpe-core
Core library for Generative Payload Encapsulation (GPE) protocol


<-- ──────────────────────────────────────────────────────────── -->

<-- ADD: v1.1-alpha QUICK START & FEATURE OVERVIEW (2025-06-25) -->

<-- ──────────────────────────────────────────────────────────── -->

## ✨ What’s New in v1.1-alpha

| Category | v1.0 | **v1.1-alpha** |
|----------|------|----------------|
| Payload layout | seed-tree (`rules` 배열) | **flat rule list** + `CONSTANT / RANGE / COMPACT_LIST / TEMPLATE` |
| Encoder flag | — | `enable_optimization=True` |
| Avg. payload size (10 k records) | 1.0× | **0.58×** (-42 %) |
| CPU decode | 1× | +3 % |
| **MP decode (2 proc)** | 1.0× | **1.9×** |
| **GPU-Stream (T4)** | 1.0× | **2.7×** |
| Back-compat | — | v1.0 ↔ v1.1 **auto-detect** |

> v1.1 페이로드는 **`"version": "gpe.v1.1"`** 로 식별되며,
> v1.0 디코더와는 호환되지 않습니다. _(서버·클라이언트 동시 배포 필요)_

---

## Quick Start (v1.1-alpha)

```bash
pip install gpe-core==1.1.0a0            # CPU only
# GPU (T4 / V100 / A100) 사용 시
pip install cupy-cuda11x                 # CUDA 11  (or cupy-cuda12x)

# 1) Encode with optimization
python - <<'PY'
from gpe_core.encoder import GPEEncoder
data = [{"id": i, "x": [0]*8} for i in range(5_000)]
enc  = GPEEncoder(enable_optimization=True, include_fallback=False)
payload = enc.encode(data)
open("out.gpe", "w").write(payload.generative_payload | to_json())
PY

# 2) Decode on GPU
gpe decode -i out.gpe -o back.json -b gpu-stream
```


## ✨ What’s New in v1.1-alpha

| Category | v1.0 | **v1.1-alpha** |
|----------|------|----------------|
| Payload layout | seed-tree (`rules` 배열) | **flat rule list** + `CONSTANT / RANGE / COMPACT_LIST / TEMPLATE` |
| Encoder flag | — | `enable_optimization=True` |
| Avg. payload size (10 k records) | 1.0× | **0.58×** (-42 %) |
| CPU decode | 1× | +3 % |
| **MP decode (2 proc)** | 1.0× | **1.9×** |
| **GPU-Stream (T4)** | 1.0× | **2.7×** |
| Back-compat | — | v1.0 ↔ v1.1 **auto-detect** |

> v1.1 페이로드는 **`"version": "gpe.v1.1"`** 로 식별되며,
> v1.0 디코더와는 호환되지 않습니다. _(서버·클라이언트 동시 배포 필요)_

---

## Quick Start (v1.1-alpha)

```bash
pip install gpe-core==1.1.0a0            # CPU only
# GPU (T4 / V100 / A100) 사용 시
pip install cupy-cuda11x                 # CUDA 11  (or cupy-cuda12x)

# 1) Encode with optimization
python - <<'PY'
from gpe_core.encoder import GPEEncoder
data = [{"id": i, "x": [0]*8} for i in range(5_000)]
enc  = GPEEncoder(enable_optimization=True, include_fallback=False)
payload = enc.encode(data)
open("out.gpe", "w").write(payload.generative_payload | to_json())
PY

# 2) Decode on GPU
gpe decode -i out.gpe -o back.json -b gpu-stream
```

---

## GPU Back-ends

| Backend      | 설명                       | v1.1 입력          |
| ------------ | ------------------------ | ---------------- |
| `gpu-stream` | ID remap GPU → 객체 복원 CPU | **CPU-fallback** |
| `gpu-full`   | ID remap + 그래프 조립 모두 GPU | **CPU-fallback** |
| `gpu-ray`    | 다중 GPU (Ray) 분산 스트림      | **CPU-fallback** |

> v1.1 페이로드가 들어오면 모든 GPU 백엔드는 자동으로 CPU 루트로
> 전환되며, 추후 *Expressway* 모듈에서 풀-GPU 복원이 지원될 예정입니다.

---

## Enabling the Optimizer

```python
from gpe_core.encoder import GPEEncoder
enc = GPEEncoder(
    enable_optimization=True,
    opt_config={          # (옵션) 최적화 튜닝
        "constant_threshold": 3,
        "template_threshold": 0.7,
        "range_min_length": 5,
    },
)
payload = enc.encode(obj)
```

* `payload.generative_payload["optimization"]` → 규칙 탐지 통계
* `RuleOptimizer.get_stats()` 포맷은 CHANGELOG ‘Added’ 항목 참고

---

## Colab Setup Snippet (12-line copy-paste)

```python
!git clone https://github.com/ycmath/gpe_core.git -q
%cd gpe_core
!pip install -q -e . cupy-cuda11x
import os; os.environ["NUMBA_DISABLE_JIT"]="1"; os.environ["GPE_USE_NUMBA"]="false"
from gpe_core.encoder import GPEEncoder
from gpe_core.gpu.stream_decoder import GPEDecoderGPUStream
data=list(range(1_000)); enc=GPEEncoder(enable_optimization=False,include_fallback=False)
assert GPEDecoderGPUStream().decode(enc.encode(data))==data
print("🚀 GPE end-to-end OK on GPU")
```

