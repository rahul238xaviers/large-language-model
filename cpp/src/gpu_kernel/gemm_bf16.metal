// ═══════════════════════════════════════════════════════════════════════════════
// gemm_bf16 — FP32 input → BF16 shared → BF16×BF16→FP32 accum → FP32 output
// ═══════════════════════════════════════════════════════════════════════════════
// Reads FP32 from Tensor storage, explicitly converts to BF16 in shared memory
// (preserving the upper 16 bits which IS the BF16 representation), computes
// via simdgroup_matrix<bfloat,8,8> × <bfloat,8,8> → <float,8,8>, writes FP32.
//
// Tiles: BM=128, BN=128, BK=32  →  16 KB shared  →  25% of 64 KB
// Threadgroup: 256 threads, 8 SIMD groups (4×2 grid)
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM=128, BN=128, BK=32;

kernel void gemm_bf16(
    device const float* A      [[buffer(0)]],  // FP32 input (Tensor storage)
    device const float* B      [[buffer(1)]],  // FP32 input
    device float*       C      [[buffer(2)]],  // FP32 output
    constant uint&       M      [[buffer(3)]],
    constant uint&       N      [[buffer(4)]],
    constant uint&       Kdim   [[buffer(5)]],
    constant bool&       trA    [[buffer(6)]],
    constant bool&       trB    [[buffer(7)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  ln_id [[thread_index_in_simdgroup]]
) {
    const uint SGM=4, SGN=2;
    uint sgr = sg_id / SGN, sgc = sg_id % SGN;

    threadgroup bfloat shA[BM * BK];
    threadgroup bfloat shB[BK * BN];

    simdgroup_matrix<float, 8, 8> acc[4][8];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 8; ++c)
            acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);

    uint tr = BM * tg_id.y, tc = BN * tg_id.x;

    for (uint kb = 0; kb < Kdim; kb += BK) {
        // Load A: FP32 → BF16 (explicit cast, preserves upper 16 bits)
        for (uint i = lid.x; i < BM * BK; i += 256) {
            uint r = i / BK, c = i % BK;
            uint gr = tr + r, gc = kb + c;
            float val = (gr < M && gc < Kdim)
                ? (trA ? A[gc * M + gr] : A[gr * Kdim + gc])
                : 0.0f;
            shA[i] = (bfloat)val;  // FP32 → BF16 truncation
        }
        // Load B: FP32 → BF16
        for (uint i = lid.x; i < BK * BN; i += 256) {
            uint r = i / BN, c = i % BN;
            uint gr = kb + r, gc = tc + c;
            float val = (gr < Kdim && gc < N)
                ? (trB ? B[gc * Kdim + gr] : B[gr * N + gc])
                : 0.0f;
            shB[i] = (bfloat)val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<bfloat, 8, 8> tA[4], tB[8];
            for (int r = 0; r < 4; ++r)
                simdgroup_load(tA[r], &shA[(sgr*32 + r*8) * BK + kk], BK);
            for (int c = 0; c < 8; ++c)
                simdgroup_load(tB[c], &shB[kk * BN + sgc*64 + c*8], BN);
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 8; ++c)
                    simdgroup_multiply_accumulate(acc[r][c], tA[r], tB[c], acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write FP32 output
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 8; ++c) {
            uint grow = tr + sgr * 32 + r * 8;
            uint gcol = tc + sgc * 64 + c * 8;
            if (grow >= M || gcol >= N) continue;
            auto vals = acc[r][c].thread_elements();
            uint lr = ln_id / 4, lc = (ln_id % 4) * 2;
            uint wr = grow + lr, wc = gcol + lc;
            if (wr < M && wc < N) {
                C[wr * N + wc] = vals[0];       // FP32 accumulator → FP32 output
                if (wc + 1 < N) C[wr * N + wc + 1] = vals[1];
            }
        }
    }
}
