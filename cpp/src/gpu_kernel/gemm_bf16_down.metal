// ═══════════════════════════════════════════════════════════════════════════════
// gemm_bf16_down — Small-batch bf16 GEMM for down-projection backward
// ═══════════════════════════════════════════════════════════════════════════════
// C[M][N] = A[M][K] × B_tr[N][K]  (B stored transposed: N×K row-major)
//
// Specialized: BM=32 (matches M=32 exactly), BN=128, BK=32, SGM=2, SGN=4
// Key optimization: coalesced B load — consecutive threads stride along K
//   (stride 1 in memory) instead of N (stride K = 1024 bf16).
// Shared: 20 KB  (shA 4 KB + shB 16 KB, double-buffered)
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM=32, BN=128, BK=32, SGM=2, SGN=4;
constant uint AR=BM/SGM/8, AC=BN/SGN/8;
constant uint RPS=BM/SGM, CPS=BN/SGN;

kernel void gemm_bf16_down(
    device const bfloat* A      [[buffer(0)]],
    device const bfloat* B      [[buffer(1)]],
    device bfloat*       C      [[buffer(2)]],
    constant uint&       M      [[buffer(3)]],
    constant uint&       N      [[buffer(4)]],
    constant uint&       Kdim   [[buffer(5)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 li    [[thread_position_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  ln_id [[thread_index_in_simdgroup]]
) {
    uint tid = li.x;
    uint sgr = sg_id / SGN, sgc = sg_id % SGN;

    threadgroup bfloat shA[2][BM * BK];
    threadgroup bfloat shB[2][BK * BN];

    simdgroup_matrix<float, 8, 8> acc[AR][AC];
    for (uint r = 0; r < AR; ++r)
        for (uint c = 0; c < AC; ++c)
            acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);

    uint tr = BM * tg_id.y;
    uint tc = BN * tg_id.x;

    auto load_A = [&](uint slot, uint kb) {
        uint base = tid * 4;
        for (uint j = 0; j < 4 && base + j < BM * BK; ++j) {
            uint rr = (base + j) / BK;
            uint cc = (base + j) % BK;
            uint grr = tr + rr;
            uint gcc = kb + cc;
            shA[slot][base + j] = (bfloat)((grr < M && gcc < Kdim) ? A[grr * Kdim + gcc] : 0.0f);
        }
    };

    // ── Coalesced B load (trB=true) ──
    // B_stored is N×K row-major.  Tile B_logical[kb..kb+BK-1][tc..tc+BN-1] =
    // B_stored[tc..tc+BN-1][kb..kb+BK-1].
    // 8 consecutive threads load 1 column × 32 rows → 4 BF16 each → 1 cache line.
    auto load_B = [&](uint slot, uint kb) {
        uint cols_per_load = 256 / (BK / 4);
        uint iters = (BN + cols_per_load - 1) / cols_per_load;
        for (uint i = 0; i < iters; ++i) {
            uint col_group = tid / (BK / 4);
            uint row_group = tid % (BK / 4);
            uint c = i * cols_per_load + col_group;
            uint r = row_group * 4;
            if (c >= BN) break;
            uint gc = tc + c;
            for (uint j = 0; j < 4 && r + j < BK; ++j) {
                uint grr = kb + r + j;
                shB[slot][(r + j) * BN + c] = (bfloat)((grr < Kdim && gc < N)
                    ? B[gc * Kdim + grr] : 0.0f);
            }
        }
    };

    auto compute = [&](uint slot) {
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

    uint NK = (Kdim + BK - 1) / BK;

    load_A(0, 0);
    load_B(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < NK - 1; ++k) {
        compute(k % 2);
        load_A((k + 1) % 2, (k + 1) * BK);
        load_B((k + 1) % 2, (k + 1) * BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    compute((NK - 1) % 2);

    for (uint r = 0; r < AR; ++r) {
        for (uint c = 0; c < AC; ++c) {
            uint grow = tr + sgr * RPS + r * 8;
            uint gcol = tc + sgc * CPS + c * 8;
            if (grow >= M || gcol >= N) continue;
            auto vals = acc[r][c].thread_elements();
            // Apple simdgroup_matrix 8x8 lane layout (Morton order)
            uint lr = (ln_id/4 >> 2)*4 + (ln_id/2) % 4;
            uint lc = (ln_id/4 & 2)*2 + (ln_id%2)*2;
            uint wr = grow + lr, wc = gcol + lc;
            if (wr < M && wc < N) {
                C[wr * N + wc] = (bfloat)vals[0];
                if (wc + 1 < N) C[wr * N + wc + 1] = (bfloat)vals[1];
            }
        }
    }
}
