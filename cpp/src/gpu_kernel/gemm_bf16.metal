// ═══════════════════════════════════════════════════════════════════════════════
// gemm_bf16 — FP32 DRAM, BF16 shared, FP32 accumulators
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM=64, BN=128, BK=32, SGM=2, SGN=4;
constant uint NO_LOADS = 0;
constant uint AR=BM/SGM/8, AC=BN/SGN/8;

kernel void gemm_bf16(
    device const bfloat* A      [[buffer(0)]],
    device const bfloat* B      [[buffer(1)]],
    device bfloat*       C      [[buffer(2)]],
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

    threadgroup bfloat shA[2][BM * BK];
    threadgroup bfloat shB[2][BK * BN];

    simdgroup_matrix<float, 8, 8> acc[AR][AC];
    for (uint r = 0; r < AR; ++r)
        for (uint c = 0; c < AC; ++c)
            acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);

    uint tr = BM * tg_id.y, tc = BN * tg_id.x;

    auto load_tile = [&](uint slot, uint kb) {
        for (uint i = 0; i < 4; ++i) {
            uint base = (tid * 4 + i) * 4;
            for (uint j = 0; j < 4 && base+j < BM*BK; ++j) {
                uint rr = (base+j)/BK, cc = (base+j)%BK;
                uint grr = tr+rr, gcc = kb+cc;
                shA[slot][base+j] = (grr<M && gcc<Kdim)
                    ? ((NO_LOADS == 1) ? (bfloat)0.01h : A[trA ? (gcc*M+grr) : (grr*Kdim+gcc)])
                    : (bfloat)0;
            }
        }
        for (uint i = 0; i < 4; ++i) {
            uint base = (tid * 4 + i) * 4;
            for (uint j = 0; j < 4 && base+j < BK*BN; ++j) {
                uint rr = (base+j)/BN, cc = (base+j)%BN;
                uint grr = kb+rr, gcc = tc+cc;
                shB[slot][base+j] = (grr<Kdim && gcc<N)
                    ? ((NO_LOADS == 1) ? (bfloat)0.01h : B[trB ? (gcc*Kdim+grr) : (grr*N+gcc)])
                    : (bfloat)0;
            }
        }
    };

    auto compute_from = [&](uint slot) {
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<bfloat, 8, 8> tA[AR], tB[AC];
            for (uint r = 0; r < AR; ++r)
                simdgroup_load(tA[r], &shA[slot][(sgr*rps + r*8) * BK + kk], BK);
            for (uint c = 0; c < AC; ++c)
                simdgroup_load(tB[c], &shB[slot][kk * BN + sgc*cps + c*8], BN);
            for (uint r = 0; r < AR; ++r)
                for (uint c = 0; c < AC; ++c)
                    simdgroup_multiply_accumulate(acc[r][c], tA[r], tB[c], acc[r][c]);
        }
    };

    uint NK = (Kdim + BK - 1) / BK;

    // Software prefetch: load tile 0 and tile 1 up front so global loads are
    // in flight while the first compute runs (overlaps memory latency).
    load_tile(0, 0);
    if (NK > 1) load_tile(1, BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < NK; ++k) {
        compute_from(k % 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k + 2 < NK) {
            load_tile(k % 2, (k + 2) * BK);  // refill the slot we just consumed
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint r = 0; r < AR; ++r) {
        for (uint c = 0; c < AC; ++c) {
            uint grow = tr + sgr * rps + r * 8;
            uint gcol = tc + sgc * cps + c * 8;
            if (grow >= M || gcol >= N) continue;
            auto vals = acc[r][c].thread_elements();
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
