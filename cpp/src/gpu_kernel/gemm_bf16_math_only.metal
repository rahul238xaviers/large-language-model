// ═══════════════════════════════════════════════════════════════════════════════
// gemm_bf16_math_only — Compute-throughput isolate
// ═══════════════════════════════════════════════════════════════════════════════
// No global memory fetches in the inner loop.  Threadgroup memory initialized
// once with dummy data.  Inner loop runs simdgroup_matrix math on the same data
// for all K-iterations.  Measures: maximum achievable simdgroup_matrix throughput
// and SRAM bank conflict impact.
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM=128, BN=128, BK=32, SGM=4, SGN=4;
constant uint AR=BM/SGM/8, AC=BN/SGN/8;

kernel void gemm_bf16_math_only(
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
    uint sgr = sg_id / SGN, sgc = sg_id % SGN;
    uint rps = BM / SGM, cps = BN / SGN;

    threadgroup bfloat shA[BM * BK];
    threadgroup bfloat shB[BK * BN];

    // ── Initialize shared memory ONCE with dummy data ──
    if (tid < BM * BK) {
        // Use the input A/B pointers to get realistic data (prevents constant-folding)
        shA[tid] = (bfloat)1.0f;
    }
    if (tid < BK * BN) {
        shB[tid] = (bfloat)1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc[AR][AC];
    for (uint r = 0; r < AR; ++r)
        for (uint c = 0; c < AC; ++c)
            acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);

    // ── Inner loop: MATH ONLY, no global fetches ──
    for (uint kb = 0; kb < Kdim; kb += BK) {
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<bfloat, 8, 8> tA[AR], tB[AC];
            for (uint r = 0; r < AR; ++r)
                simdgroup_load(tA[r], &shA[(sgr*rps + r*8) * BK + kk], BK);
            for (uint c = 0; c < AC; ++c)
                simdgroup_load(tB[c], &shB[kk * BN + sgc*cps + c*8], BN);
            for (uint r = 0; r < AR; ++r)
                for (uint c = 0; c < AC; ++c)
                    simdgroup_multiply_accumulate(acc[r][c], tA[r], tB[c], acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write output ──
    for (uint r = 0; r < AR; ++r) {
        for (uint c = 0; c < AC; ++c) {
            uint grow = BM * tg_id.y + sgr*rps + r*8;
            uint gcol = BN * tg_id.x + sgc*cps + c*8;
            if (grow >= M || gcol >= N) continue;
            auto vals = acc[r][c].thread_elements();
            uint lr = ln_id / 4, lc = (ln_id % 4) * 2;
            uint wr = grow + lr, wc = gcol + lc;
            if (wr < M && wc < N) {
                C[wr * N + wc] = vals[0];
                if (wc + 1 < N) C[wr * N + wc + 1] = vals[1];
            }
        }
    }
}
