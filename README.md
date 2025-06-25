# gpe-core
Core library for Generative Payload Encapsulation (GPE) protocol

1. 빠른 시작 예시
```bash
pip install gpe-core==1.1a0
gpe encode -i sample.json -o out.gpe
gpe decode -i out.gpe -o back.json -b gpu-stream
```

---
2. v1.1 vs v1.0 차이 도표

---
3. GPU 의존성
> CuPy ≥ 12, CUDA 11.x / 12.x
> assemble_graph.cu 가 실패하면 자동 CPU-fallback

---
4. Colab 패치 셀 snippet
GPE-object-count vs elapsed time2.ipynb
```python
import os, sys, subprocess, shutil, pathlib, textwrap

# 1) 환경 변수
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["GPE_USE_NUMBA"]     = "false"

# 2) 기존 패키지 제거
subprocess.run(["pip", "uninstall", "-y", "gpe-core", "-q"], check=False)

# 3) 깃 복제
!git clone -q https://github.com/ycmath/gpe_core.git /content/gpe_core

# 4) -O3 옵션 제거
!sed -i 's/-O3", "-arch/-arch/' \
      /content/gpe_core/gpe_core/gpu/assemble_graph_v2.cu
!sed -i 's/"-use_fast_math", "-O3"/"-use_fast_math"/' \
      /content/gpe_core/gpe_core/gpu/id_remap_opt.py

# 5) editable install
!pip install -q -e /content/gpe_core
sys.path.insert(0, '/content/gpe_core')

# 6) CuPy 커널 캐시 비우기
import cupy as cp
try:
    if hasattr(cp, "util") and hasattr(cp.util, "clear_memo"):
        cp.util.clear_memo()
except Exception:
    pass
shutil.rmtree(os.path.expanduser("~/.cupy/kernel_cache"), ignore_errors=True)

print("✅ gpe-core 설치 & -O3 패치 완료")

from importlib import reload
import gpe_core.gpu.assemble_graph as ag
reload(ag)                      # 수정 적용

from gpe_core.encoder import GPEEncoder
from gpe_core.gpu.stream_decoder import GPEDecoderGPUStream

data  = list(range(30))
enc10 = GPEEncoder(enable_optimization=False, include_fallback=False)

assert GPEDecoderGPUStream().decode(enc10.encode(data)) == data
print("✓ GPU-Stream 디코더 OK  (커널 재컴파일 완료)")
```
