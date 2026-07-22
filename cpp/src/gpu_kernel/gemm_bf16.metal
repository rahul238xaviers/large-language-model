// ═══════════════════════════════════════════════════════════════════════════════
// gemm_bf16 — FP32 DRAM, BF16 shared, FP32 accumulators
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM=128, BN=128, BK=32, SGM=4, SGN=4;
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
        for (uint i = 0; i < 2; ++i) {
            uint base = (tid * 2 + i) * 4;
            uint r = base / BK, c = base % BK;
            uint gr = tr + r, gc = kb + c;
            if (gr < M && gc < Kdim) {
                float4 v = *((device const float4*)(trA ? &A[gc*M+gr] : &A[gr*Kdim+gc]));
                shA[slot][r*BK+c+0] = (bfloat)v[0]; shA[slot][r*BK+c+1] = (bfloat)v[1];
                shA[slot][r*BK+c+2] = (bfloat)v[2]; shA[slot][r*BK+c+3] = (bfloat)v[3];
            } else {
                for (uint j = 0; j < 4 && base+j < BM*BK; ++j) {
                    uint rr = (base+j)/BK, cc = (base+j)%BK;
                    uint grr = tr+rr, gcc = kb+cc;
                    shA[slot][base+j] = (grr<M && gcc<Kdim)
                        ? (bfloat)(trA ? A[gcc*M+grr] : A[grr*Kdim+gcc])
                        : (bfloat)0;
                }
            }
        }
        for (uint i = 0; i < 2; ++i) {
            uint base = (tid * 2 + i) * 4;
            uint r = base / BN, c = base % BN;
            uint gr = kb + r, gc = tc + c;
            if (gr < Kdim && gc < N) {
                float4 v = *((device const float4*)(trB ? &B[gc*Kdim+gr] : &B[gr*N+gc]));
                shB[slot][r*BN+c+0] = (bfloat)v[0]; shB[slot][r*BN+c+1] = (bfloat)v[1];
                shB[slot][r*BN+c+2] = (bfloat)v[2]; shB[slot][r*BN+c+3] = (bfloat)v[3];
            } else {
                for (uint j = 0; j < 4 && base+j < BK*BN; ++j) {
                    uint rr = (base+j)/BN, cc = (base+j)%BN;
                    uint grr = kb+rr, gcc = tc+cc;
                    shB[slot][base+j] = (grr<Kdim && gcc<N)
                        ? (bfloat)(trB ? B[gcc*Kdim+grr] : B[grr*N+gcc])
                        : (bfloat)0;
                }
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

    load_tile(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < NK - 1; ++k) {
        compute_from(k % 2);
        load_tile((k + 1) % 2, (k + 1) * BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    compute_from((NK - 1) % 2);

    for (uint r = 0; r < AR; ++r) {
        for (uint c = 0; c < AC; ++c) {
            uint grow = tr + sgr * rps + r * 8;
            uint gcol = tc + sgc * cps + c * 8;
            if (grow >= M || gcol >= N) continue;
            auto vals = acc[r][c].thread_elements();
            uint lr = ln_id / 4, lc = (ln_id % 4) * 2;
            uint wr = grow + lr, wc = gcol + lc;
            if (wr < M && wc < N) {
                C[wr * N + wc] = (bfloat)vals[0];
                if (wc + 1 < N) C[wr * N + wc + 1] = (bfloat)vals[1];
            }
        }
    }
}
