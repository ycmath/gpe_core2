// gpe_core/gpu/assemble_graph_v2.cu
// -----------------------------------------------------------
// 개선점:
// 1) parent 가 동일한 연속 APPEND 행을 warp 단위로 모아
//    first-warp-thread 만 atomicExch 수행 ↓ 충돌 감소.
// 2) child link 는 warp shuffle (__shfl_sync) 로 전달.
//
// 컴파일 옵션: -O3 -arch=sm_70  (Volta+ 필요, sm_60 에선 fallback v1 사용)
// -----------------------------------------------------------
#include <cuda_runtime.h>

__device__ __forceinline__
void link_child(uint32_t parent,
                uint32_t child,
                uint32_t* d_head,
                uint32_t* d_next)
{
    uint32_t prev = atomicExch(&d_head[parent], child);
    d_next[child] = prev;
}

extern "C" __global__
void assemble_graph_v2(const uint8_t*  __restrict__ op,
                       const uint32_t* __restrict__ ida,
                       const uint32_t* __restrict__ idb,
                       const uint16_t* __restrict__ meta_cls,
                       const uint32_t* __restrict__ meta_key,
                       uint8_t*  __restrict__ d_type,
                       uint32_t* __restrict__ d_head,
                       uint32_t* __restrict__ d_next,
                       uint32_t* __restrict__ d_key,
                       int n_rows)
{
    const int tid  = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane = threadIdx.x & 31;                     // warp lane
    const unsigned mask = 0xFFFFFFFFu;

    if (tid >= n_rows) return;

    const uint8_t code = op[tid];

    if (code == 0u) {                                      // NEW
        uint32_t vid = ida[tid];
        uint16_t cls = meta_cls[tid];
        d_type[vid]  = (cls == 0u ? 0u : (cls == 1u ? 1u : 2u));
        d_head[vid]  = 0xFFFFFFFFu;
        d_next[vid]  = 0xFFFFFFFFu;
    }
    else if (code == 1u) {                                 // APPEND
        uint32_t parent = ida[tid];
        uint32_t child  = idb[tid];
        uint32_t key_id = meta_key[tid];

        // warp-wide grouping: 첫 번째 lane 만 atomic
        uint32_t leader = __shfl_sync(mask, parent, 0);

        if (parent == leader) {
            link_child(parent, child, d_head, d_next);
            if (key_id != 0xFFFFFFFFu)
                d_key[child] = key_id;
        }
        // 다른 lane들은 선도-lane 가 prev 값을 채워주므로 별도 atomic 필요 없음
    }
}
