# gpe_core/utils_profile.py
# ------------------------------------------------------------
"""Lightweight 프로파일링 헬퍼.

사용 예:
    from gpe_core.utils_profile import profile_section

    with profile_section("encode"):
        payload = enc.encode(data)

활성화 방법:
    • 환경변수 GPE_PROFILE=1
    • 또는 코드에서 utils_profile.PROFILING_ENABLED = True
"""

import os
import cProfile
import pstats
from contextlib import contextmanager
from io import StringIO
from time import perf_counter

PROFILING_ENABLED: bool = bool(int(os.getenv("GPE_PROFILE", "0")))


@contextmanager
def profile_section(name: str):
    if not PROFILING_ENABLED:
        yield
        return

    pr = cProfile.Profile()
    pr.enable()
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # 상위 20개 함수
        print(f"\n=== [{name}] {dt*1000:.1f} ms  Profile ===")
        print(s.getvalue())


################################################################################
## 간단 통합 예시

## from gpe_core.utils_profile import profile_section

## def cmd_bench(ns):
##     ...
##     with profile_section("encode"):
##         payload = enc.encode(data)
##     with profile_section("decode"):
##         dec.decode(payload)
        
## GPE_PROFILE=1 gpe bench ... 처럼 실행하면 섹션별 누적 함수 시간이 콘솔에 출력됩니다.
## 기본(환경변수 없을 때)은 오버헤드 없이 작동합니다.
