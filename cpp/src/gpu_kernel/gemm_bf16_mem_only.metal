// ═══════════════════════════════════════════════════════════════════════════════
// gemm_bf16_mem_only — Memory-bandwidth isolate
// ═══════════════════════════════════════════════════════════════════════════════
// Identical global→threadgroup fetch to gemm_bf16.  NO simdgroup_matrix math.
// Writes one byte to output to prevent DCE.
// Measures: global memory read throughput (GB/s) achievable by the fetch pattern.
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM=128, BN=128, BK=32;

kernel void gemm_bf16_mem_only(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float*       C      [[buffer(2)]],
    constant uint&       M      [[buffer(3)]],
    constant uint&       N      [[buffer(4)]],
    constant uint&       Kdim   [[buffer(5)]],
    constant bool&       trA    [[buffer(6)]],
    constant bool&       trB    [[buffer(7)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 li    [[thread_position_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  ln_id [[thread_index_in_simdgroup]]
) {
    uint tid = li.x;

    threadgroup bfloat shA[BM * BK];
    threadgroup bfloat shB[BK * BN];

    uint tr = BM * tg_id.y, tc = BN * tg_id.x;

    // ── Inner K-loop: FETCH ONLY, no math ──
    for (uint kb = 0; kb < Kdim; kb += BK) {
        for (uint i = 0; i < 2; ++i) {
            uint base = (tid * 2 + i) * 4;
            uint r = base / BK, c = base % BK;
            uint gr = tr + r, gc = kb + c;
            if (gr < M && gc < Kdim) {
                float4 v = *((device const float4*)(trA ? &A[gc*M+gr] : &A[gr*Kdim+gc]));
                shA[r*BK+c+0] = (bfloat)v[0]; shA[r*BK+c+1] = (bfloat)v[1];
                shA[r*BK+c+2] = (bfloat)v[2]; shA[r*BK+c+3] = (bfloat)v[3];
            }
        }
        for (uint i = 0; i < 2; ++i) {
            uint base = (tid * 2 + i) * 4;
            uint r = base / BN, c = base % BN;
            uint gr = kb + r, gc = tc + c;
            if (gr < Kdim && gc < N) {
                float4 v = *((device const float4*)(trB ? &B[gc*Kdim+gr] : &B[gr*N+gc]));
                shB[r*BN+c+0] = (bfloat)v[0]; shB[r*BN+c+1] = (bfloat)v[1];
                shB[r*BN+c+2] = (bfloat)v[2]; shB[r*BN+c+3] = (bfloat)v[3];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // NO MATH — just barrier to simulate compute timing
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Single dummy write to prevent DCE ──
    if (tid == 0 && tg_id.x == 0 && tg_id.y == 0) {
        C[0] = (float)shA[0] + (float)shB[0];
    }
}
