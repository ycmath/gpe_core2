
#include <cuda_runtime.h>

extern "C" __global__
void assemble_graph(const uint8_t*  __restrict__ op,
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;

    uint8_t code = op[i];

    if (code == 0u) {                        // NEW
        uint32_t vid = ida[i];
        uint16_t cls = meta_cls[i];

        // type encoding
        d_type[vid] = (cls == 0u) ? 0u : (cls == 1u ? 1u : 2u);
        d_head[vid] = 0xFFFFFFFFu;           // sentinel null
        d_next[vid] = 0xFFFFFFFFu;
    }
    else if (code == 1u) {                   // APPEND
        uint32_t parent = ida[i];
        uint32_t child  = idb[i];

        // atomic LIFO push: child -> head[parent]
        uint32_t prev = atomicExch(&d_head[parent], child);
        d_next[child] = prev;

        // dict key
        uint32_t k = meta_key[i];
        if (k != 0xFFFFFFFFu)
            d_key[child] = k;
    }
    // REPEAT tokens are structural only – ignored here
}


## * **동기화 필요 없음**: `atomicExch` 로 child 단일-링크 list 구성 → post-pass 에서 역순 iterate.
## * **공유 메모리**: row 단위 prefix 연산이 없으므로 쓰지 않음 → Warp divergence 無.
