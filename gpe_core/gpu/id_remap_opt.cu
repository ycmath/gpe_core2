#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

extern "C" __global__
void id_remap_opt(const uint8_t*  __restrict__ op,
                  const uint8_t*  __restrict__ mask16_a,
                  const uint8_t*  __restrict__ mask16_b,
                  const uint16_t* __restrict__ pool16_a,
                  const uint32_t* __restrict__ pool32_a,
                  const uint16_t* __restrict__ pool16_b,
                  const uint32_t* __restrict__ pool32_b,
                  uint32_t*       __restrict__ out_a,
                  uint32_t*       __restrict__ out_b,
                  int32_t n16_a, int32_t n32_a,
                  int32_t n16_b, int32_t n32_b,
                  int32_t n_rows)
{
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage tmp;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;

    int bit_a = mask16_a[i];
    int bit_b = mask16_b[i];

    int pref_a; Scan(tmp).ExclusiveSum(bit_a, pref_a);
    int pref_b; Scan(tmp).ExclusiveSum(bit_b, pref_b);

    int idx16a = pref_a;
    int idx32a = i - idx16a;
    int idx16b = pref_b;
    int idx32b = i - idx16b;

    uint32_t ida = bit_a ? static_cast<uint32_t>(pool16_a[idx16a])
                         : pool32_a[idx32a];
    uint32_t idb = bit_b ? static_cast<uint32_t>(pool16_b[idx16b])
                         : pool32_b[idx32b];

    if (op[i] == 0u) {                 // NEW
        out_a[i] = ida;
    } else if (op[i] == 1u) {          // APPEND
        out_a[i] = ida;
        out_b[i] = idb;
    }
}
