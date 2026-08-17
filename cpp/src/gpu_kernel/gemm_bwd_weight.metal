// ═══════════════════════════════════════════════════════════════════════════════
// gemm_bwd_weight — Coalesced weight-gradient GEMM: C[M][N] += A^T[M][K] × B[K][N]
// ═══════════════════════════════════════════════════════════════════════════════
// Computes weight gradients dW = x^T @ dy, where A (activations) is stored
// row-major as K×M ([tokens][features]) and B (upstream grads) as K×N.
// The contraction K is the TOKEN dimension (e.g. 32768), while the output M×N
// is weight-sized.  Streaming over K in BK-sized tiles with coalesced loads of
// both A and B keeps this memory-bound shape near bandwidth, instead of the
// strided transA path in gemm_bf16 that was ~10x slower.
//
// Tile: BM=64, BN=128, BK=32, SGM=2, SGN=4 (8 SIMD groups / 256 threads).
// Shared: shA 4 KB + shB 8 KB = 12 KB (double-buffered across K-tiles via
// barriers).
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM=64, BN=128, BK=32;
constant uint SGM=2, SGN=4;
constant uint AR=BM/(SGM*8), AC=BN/(SGN*8);
constant uint RPS=BM/SGM, CPS=BN/SGN;

kernel void gemm_bwd_weight(
    device const bfloat* A      [[buffer(0)]],  // K×M row-major
    device const bfloat* B      [[buffer(1)]],  // K×N row-major
    device bfloat*       C      [[buffer(2)]],  // M×N accumulator (adds)
    constant uint&       M      [[buffer(3)]],
    constant uint&       N      [[buffer(4)]],
    constant uint&       Kdim   [[buffer(5)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 li    [[thread_position_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  ln_id [[thread_index_in_simdgroup]]
) {
    const uint tid = li.x;
    const uint sgr = sg_id / SGN, sgc = sg_id % SGN;

    threadgroup bfloat shA[2][BM * BK];
    threadgroup bfloat shB[2][BK * BN];

    simdgroup_matrix<float, 8, 8> acc[AR][AC];
    for (uint r = 0; r < AR; ++r)
        for (uint c = 0; c < AC; ++c)
            acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint tr = BM * tg_id.y;
    const uint tc = BN * tg_id.x;

    auto load_tile = [&](uint slot, uint kb) {
        // ── Coalesced A load ── A is K×M: elem A[(kb+c)*M + (tr+r)].
        // Consecutive threads map to consecutive r (M dim), contiguous in memory.
        for (uint idx = tid; idx < BM * BK; idx += 256) {
            const uint r = idx % BM;
            const uint c = idx / BM;
            const uint gc = kb + c;
            const uint gr = tr + r;
            shA[slot][r * BK + c] = (gc < Kdim && gr < M) ? A[gc * M + gr] : (bfloat)0.0f;
        }
        // ── Coalesced B load ── B is K×N: elem B[(kb+r)*N + (tc+c)].
        for (uint idx = tid; idx < BK * BN; idx += 256) {
            const uint r = idx / BN;
            const uint c = idx % BN;
            const uint gr = kb + r;
            const uint gc = tc + c;
            shB[slot][r * BN + c] = (gr < Kdim && gc < N) ? B[gr * N + gc] : (bfloat)0.0f;
        }
    };

    auto compute_from = [&](uint slot) {
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<bfloat, 8, 8> tA[AR], tB[AC];
            for (uint r = 0; r < AR; ++r)
                simdgroup_load(tA[r], &shA[slot][(sgr * RPS + r * 8) * BK + kk], BK);
            for (uint c = 0; c < AC; ++c)
                simdgroup_load(tB[c], &shB[slot][kk * BN + sgc * CPS + c * 8], BN);
            for (uint r = 0; r < AR; ++r)
                for (uint c = 0; c < AC; ++c)
                    simdgroup_multiply_accumulate(acc[r][c], tA[r], tB[c], acc[r][c]);
        }
    };

    const uint nK = (Kdim + BK - 1) / BK;

    // Software prefetch: load tiles 0 and 1 up front so global loads are in
    // flight while the first compute runs (overlaps memory latency).
    load_tile(0, 0);
    if (nK > 1) load_tile(1, BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < nK; ++k) {
        compute_from(k % 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k + 2 < nK) {
            load_tile(k % 2, (k + 2) * BK);  // refill the slot just consumed
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Accumulate acc into global C ──
    for (uint r = 0; r < AR; ++r) {
        for (uint c = 0; c < AC; ++c) {
            const uint grow = tr + sgr * RPS + r * 8;
            const uint gcol = tc + sgc * CPS + c * 8;
            if (grow >= M || gcol >= N) continue;
            auto vals = acc[r][c].thread_elements();
            // Apple simdgroup_matrix 8x8 lane layout (Morton order)
            const uint lr = (ln_id / 4 >> 2) * 4 + (ln_id / 2) % 4;
            const uint lc = (ln_id / 4 & 2) * 2 + (ln_id % 2) * 2;
            const uint wr = grow + lr;
            const uint wc = gcol + lc;
            if (wr < M && wc < N) {
                C[wr * N + wc] = (bfloat)((float)C[wr * N + wc] + vals[0]);
                if (wc + 1 < N)
                    C[wr * N + wc + 1] = (bfloat)((float)C[wr * N + wc + 1] + vals[1]);
            }
        }
    }
}
