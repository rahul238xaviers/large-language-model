// ═══════════════════════════════════════════════════════════════════════════════
// fused_swiglu_gemm — SwiGLU activation fused into down-projection GEMM
// ═══════════════════════════════════════════════════════════════════════════════
// Computes: C[M,N] = SwiGLU(gate_proj[M,K], up_proj[M,K]) @ w_down[K,N]
//
// Activation computed on-the-fly in registers during tile load:
//   shA[row][col] = SwiGLU(gate_proj[row][k], up_proj[row][k])
//                = gate * sigmoid(gate) * up
// The simdgroup_matrix pipelined GEMM is completely unchanged.
//
// Eliminates 360 MB DRAM write + 180 MB read per layer = 8.64 GB/step.
// Register impact: ~8 extra registers per thread (well within 128 limit).
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint BM=128, BN=128, BK=32, SGM=4, SGN=4;
constant uint AR=BM/SGM/8, AC=BN/SGN/8;

kernel void fused_swiglu_gemm(
    device const bfloat* gate_proj [[buffer(0)]],  // [M, K]
    device const bfloat* up_proj   [[buffer(1)]],  // [M, K]
    device const bfloat* B         [[buffer(2)]],  // [K, N]  w_down
    device bfloat*       C         [[buffer(3)]],  // [M, N]  ffn_out
    constant uint&       M         [[buffer(4)]],
    constant uint&       N         [[buffer(5)]],
    constant uint&       Kdim      [[buffer(6)]],
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
        // ── Left-hand matrix: compute SwiGLU on the fly ──
        for (uint i = 0; i < 2; ++i) {
            uint base = (tid * 2 + i) * 4;
            uint r = base / BK, c = base % BK;
            uint gr = tr + r, gc = kb + c;
            // Vector path only when the full bfloat4 stays inside the row;
            // otherwise fall through to the bounds-checked scalar path.  A
            // bfloat4 load at gc near Kdim would read past the row and, on the
            // last row, past the buffer (GPU page fault).
            if (gr < M && gc < Kdim && gc + 4 <= Kdim) {
                bfloat4 gv = *((device const bfloat4*)(&gate_proj[gr * Kdim + gc]));
                bfloat4 uv = *((device const bfloat4*)(&up_proj[gr * Kdim + gc]));
                float4 g = float4((float)gv[0], (float)gv[1], (float)gv[2], (float)gv[3]);
                float4 u = float4((float)uv[0], (float)uv[1], (float)uv[2], (float)uv[3]);
                float4 sig = 1.0f / (1.0f + exp(-g));
                float4 ac = g * sig * u;
                shA[slot][r*BK+c+0] = (bfloat)ac[0]; shA[slot][r*BK+c+1] = (bfloat)ac[1];
                shA[slot][r*BK+c+2] = (bfloat)ac[2]; shA[slot][r*BK+c+3] = (bfloat)ac[3];
            } else {
                for (uint j = 0; j < 4 && base+j < BM*BK; ++j) {
                    uint rr = (base+j)/BK, cc = (base+j)%BK;
                    uint grr = tr+rr, gcc = kb+cc;
                    bfloat ac = (bfloat)0;
                    if (grr < M && gcc < Kdim) {
                        float g = (float)gate_proj[grr * Kdim + gcc];
                        float u = (float)up_proj[grr * Kdim + gcc];
                        float sig = 1.0f / (1.0f + exp(-g));
                        ac = (bfloat)(g * sig * u);
                    }
                    shA[slot][base+j] = ac;
                }
            }
        }
        // ── Right-hand matrix B: standard tile load (unchanged) ──
        for (uint i = 0; i < 2; ++i) {
            uint base = (tid * 2 + i) * 4;
            uint r = base / BN, c = base % BN;
            uint gr = kb + r, gc = tc + c;
            // Same tail guard: bfloat4 must fit within the row (and buffer).
            if (gr < Kdim && gc < N && gc + 4 <= N) {
                bfloat4 v = *((device const bfloat4*)(&B[gr * N + gc]));
                shB[slot][r*BN+c+0] = v[0]; shB[slot][r*BN+c+1] = v[1];
                shB[slot][r*BN+c+2] = v[2]; shB[slot][r*BN+c+3] = v[3];
            } else {
                for (uint j = 0; j < 4 && base+j < BK*BN; ++j) {
                    uint rr = (base+j)/BN, cc = (base+j)%BN;
                    uint grr = kb+rr, gcc = tc+cc;
                    shB[slot][base+j] = (grr<Kdim && gcc<N) ? B[grr*N+gcc] : (bfloat)0;
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
