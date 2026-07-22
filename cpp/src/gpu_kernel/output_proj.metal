// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT PROJECTION — BF16×BF16 → FP32 accum → BF16/FP32 output
// ═══════════════════════════════════════════════════════════════════════════════
// Forward:  Y = X @ W      [32768,1024]×[1024,100352] → [32768,100352] BF16
// Backward dX: dX = dY @ W^T  [32768,100352]×[100352,1024] → [32768,1024] FP32
// Backward dW: dW = X^T @ dY  [1024,32768]×[32768,100352] → [1024,100352] FP32
//
// TILES: BM=128, BN=128, BK=32
// Threadgroup: 256 threads (8 SIMD, 4×2 grid), 88 regs/thread, 16 KB shared
// simdgroup_matrix<bfloat,8,8> × <bfloat,8,8> → <float,8,8> accum
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM = 128, BN = 128, BK = 32;

kernel void output_proj(
    device const bfloat* A      [[buffer(0)]],
    device const bfloat* B      [[buffer(1)]],
    device void*         C      [[buffer(2)]],
    constant uint&       M      [[buffer(3)]],
    constant uint&       N      [[buffer(4)]],
    constant uint&       Kdim   [[buffer(5)]],
    constant bool&       trA    [[buffer(6)]],
    constant bool&       trB    [[buffer(7)]],
    constant bool&       is_fwd [[buffer(8)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  ln_id [[thread_index_in_simdgroup]]
) {
    const uint SG_M = 4, SG_N = 2;
    uint sgr = sg_id / SG_N, sgc = sg_id % SG_N;

    threadgroup bfloat shA[BM * BK];
    threadgroup bfloat shB[BK * BN];

    simdgroup_matrix<float, 8, 8> acc[4][8];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 8; ++c)
            acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);

    uint tr = BM * tg_id.y, tc = BN * tg_id.x;

    for (uint kb = 0; kb < Kdim; kb += BK) {
        // Load A tile (256 threads × 16 = 4096 = BM×BK BF16)
        uint th = lid.x;
        for (uint i = th; i < BM * BK; i += 256) {
            uint r = i / BK, c = i % BK;
            uint gr = tr + r, gc = kb + c;
            shA[i] = (gr < M && gc < Kdim)
                ? (trA ? (bfloat)A[gc * M + gr] : (bfloat)A[gr * Kdim + gc])
                : (bfloat)0;
        }
        // Load B tile
        for (uint i = th; i < BK * BN; i += 256) {
            uint r = i / BN, c = i % BN;
            uint gr = kb + r, gc = tc + c;
            shB[i] = (gr < Kdim && gc < N)
                ? (trB ? (bfloat)B[gc * Kdim + gr] : (bfloat)B[gr * N + gc])
                : (bfloat)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: 4×8 simdgroup_mat_mul per SIMD group
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<bfloat, 8, 8> tA[4], tB[8];
            for (int r = 0; r < 4; ++r)
                simdgroup_load(tA[r], &shA[(sgr*32 + r*8) * BK + kk], BK);
            for (int c = 0; c < 8; ++c)
                simdgroup_load(tB[c], &shB[kk * BN + sgc*64 + c*8], BN);
#pragma clang loop unroll(full)
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 8; ++c)
                    simdgroup_multiply_accumulate(acc[r][c], tA[r], tB[c], acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write output ──
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 8; ++c) {
            uint grow = tr + sgr * 32 + r * 8;
            uint gcol = tc + sgc * 64 + c * 8;
            if (grow >= M || gcol >= N) continue;

            // simdgroup_matrix<float,8,8> on Apple Silicon:
            // Thread ln_id holds element at (ln_id/4, (ln_id%4)*2 + element_index)
            auto vals = acc[r][c].thread_elements();
            uint lr = ln_id / 4;
            uint lc = (ln_id % 4) * 2;
            uint wr = grow + lr;
            uint wc = gcol + lc;

            if (wr < M && wc < N) {
                if (is_fwd) {
                    // BF16 output with FP32→BF16 rounding
                    ((device bfloat*)C)[wr * N + wc] = (bfloat)vals[0];
                    if (wc + 1 < N)
                        ((device bfloat*)C)[wr * N + wc + 1] = (bfloat)vals[1];
                } else {
                    // FP32 gradient output
                    ((device float*)C)[wr * N + wc] = vals[0];
                    if (wc + 1 < N)
                        ((device float*)C)[wr * N + wc + 1] = vals[1];
                }
            }
        }
    }
}
