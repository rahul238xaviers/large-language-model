// ═══════════════════════════════════════════════════════════════════════════════
// fused_attn_bwd — Tile-coalesced FlashAttention-2 backward
// ═══════════════════════════════════════════════════════════════════════════════
// Two-pass:
//   Phase 1 — online softmax: compute m (row max), l (row sum), sp (rowsum P*dP)
//   Phase 2 — gradients: dQ, dK, dV using stored m/l/sp
//
// Grid: [B, nH, 1]  — one TG per (batch,query_head).  512 TGs × 256 threads.
// Tiling: Br=32, Bc=32 (was Br=32, Bc=16 → 2× fewer inner iterations)
//
// Shared memory (29 KB = 90% of 32 KB):
//   Qs [Br×HD]  BF16   4 KB  |  dOs[Br×HD]  BF16   4 KB
//   Ks [Bc×HD]  BF16   4 KB  |  Vs [Bc×HD]  BF16   4 KB
//   mx [Br]     FP32 128 B   |  sm [Br]     FP32 128 B
//   sp [Br]     FP32 128 B   |  dKa[Bc×HD]  FP32   8 KB
//   dVa[Bc×HD]  FP32   8 KB
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint S=1024, HD=64, Br=32, Bc=32;
constant uint NQ=S/Br, NK=S/Bc;
constant uint WL=32; // warp/lane size

kernel void fused_attn_bwd(
    device const float* Q      [[buffer(0)]],
    device const float* K      [[buffer(1)]],
    device const float* V      [[buffer(2)]],
    device const float* dO     [[buffer(3)]],
    device float*        dQ    [[buffer(4)]],
    device float*        dK    [[buffer(5)]],
    device float*        dV    [[buffer(6)]],
    constant uint*  n_heads_ptr [[buffer(7)]],
    constant uint*  n_kv_ptr    [[buffer(8)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 li_p  [[thread_position_in_threadgroup]]
) {
    uint b = tg_id.x, h = tg_id.y;
    uint n_heads = *n_heads_ptr, n_kv = *n_kv_ptr;
    if (b >= 32 || h >= n_heads) return;
    uint kv_h = h / (n_heads / n_kv);
    float scale = 1.0f / sqrt((float)HD);

    uint li = li_p.x;
    uint sg_id = li / WL, ln_id = li % WL;
    uint QR = Br / 8; // 4 Q-rows per warp

    // ── Shared memory ──
    threadgroup bfloat Qs [2048];   // 32×64 BF16  (4 KB)
    threadgroup bfloat dOs[2048];   // 32×64 BF16
    threadgroup bfloat Ks [2048];   // 32×64 BF16
    threadgroup bfloat Vs [2048];   // 32×64 BF16
    threadgroup float  mx [32];     // 32 FP32     (128 B)
    threadgroup float  sm [32];     // 32 FP32
    threadgroup float  sp [32];     // 32 FP32
    threadgroup bfloat dKa[2048];   // 32×64 BF16  (4 KB)
    threadgroup bfloat dVa[2048];   // 32×64 BF16
    // Total: 28,672 B = 28 KB

    uint q_off  = (b * n_heads + h) * S * HD;
    uint kv_off = (b * n_kv + kv_h)  * S * HD;

    // dQ accumulator in registers
    float dQa[4][HD];  // max 4 Q-rows per warp
    for (uint qr = 0; qr < QR; ++qr)
        for (uint d = ln_id; d < HD; d += WL) dQa[qr][d] = 0.0f;

    // ── Outer loop: Q tiles ──
    for (uint qt = 0; qt < NQ; ++qt) {
        uint q_base = qt * Br;
        uint gqr0 = q_base + sg_id * QR;

        // ── Load Q, dO (coalesced float4 → shared BF16) ──
        for (uint i = li; i < Br*HD; i += 256) {
            uint r = i/HD, c = i%HD, gr = q_base + r;
            if (gr < S) {
                Qs [r*HD+c] = (bfloat)Q [q_off + gr*HD + c];
                dOs[r*HD+c] = (bfloat)dO[q_off + gr*HD + c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (li < Br) { mx[li] = -INFINITY; sm[li] = 0.0f; sp[li] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ══════════════════════════════════════════════════════════════════
        // PHASE 1 — online softmax (K-tiles up to qt, causal)
        // ══════════════════════════════════════════════════════════════════
        for (uint kt = 0; kt <= qt && kt < NK; ++kt) {
            uint k_base = kt * Bc;

            // Load K,V (float4 → shared BF16)
            for (uint i = li; i < Bc*HD; i += 256) {
                uint r = i/HD, c = i%HD, gr = k_base + r;
                if (gr < S) {
                    Ks[r*HD+c] = (bfloat)K[kv_off + gr*HD + c];
                    Vs[r*HD+c] = (bfloat)V[kv_off + gr*HD + c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint qr = 0; qr < QR; ++qr) {
                uint gqr = gqr0 + qr;
                if (gqr >= S) continue;

                // S[j] = Q[gqr]·K[j] / √d  (lane = column j)
                uint kj = k_base + ln_id;
                float s_val = -INFINITY;
                if (kj <= gqr && kj < S) {
                    s_val = 0.0f;
                    for (uint d = 0; d < HD; ++d)
                        s_val += (float)Qs[(sg_id*QR+qr)*HD + d] * (float)Ks[ln_id*HD + d];
                    s_val *= scale;
                }

                // Warp rowmax → global max
                float wmax = s_val;
                for (uint m = 16; m > 0; m >>= 1) wmax = max(wmax, simd_shuffle_down(wmax, m));
                float gmax = simd_shuffle(wmax, 0);

                // Update mx[qr] = max(mx[qr], gmax) over all warps
                // Since only one warp writes to mx[qr], no contention
                // But multiple warps handle DIFFERENT qr values
                uint abs_qr = sg_id * QR + qr;  // absolute Q-row index in this tile
                float cur_mx = mx[abs_qr];
                if (gmax > cur_mx) mx[abs_qr] = gmax;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float new_mx = mx[abs_qr];
                float rescale = exp(cur_mx - new_mx);

                // P[j], dP[j]
                float p_val  = (kj <= gqr && kj < S) ? exp(s_val - new_mx) : 0.0f;
                float dP_val = 0.0f;
                if (kj <= gqr && kj < S)
                    for (uint d = 0; d < HD; ++d)
                        dP_val += (float)dOs[(sg_id*QR+qr)*HD + d] * (float)Vs[ln_id*HD + d];

                // Warp sum of P for this row
                float p_sum = p_val;
                for (uint m = 16; m > 0; m >>= 1) p_sum += simd_shuffle_down(p_sum, m);
                float tile_l = simd_shuffle(p_sum, 0);

                // Warp sum of P*dP
                float pd_sum = p_val * dP_val;
                for (uint m = 16; m > 0; m >>= 1) pd_sum += simd_shuffle_down(pd_sum, m);
                float tile_sp = simd_shuffle(pd_sum, 0);

                if (ln_id == 0) {
                    sm[abs_qr] = cur_mx > -INFINITY ? sm[abs_qr] * rescale + tile_l : tile_l;
                    sp[abs_qr] = cur_mx > -INFINITY ? sp[abs_qr] * rescale + tile_sp : tile_sp;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE 2 — gradients (replay K-tiles with m/l/sp from Phase 1)
        // ══════════════════════════════════════════════════════════════════
        for (uint i = li; i < Bc*HD; i += 256) { dKa[i] = (bfloat)0; dVa[i] = (bfloat)0; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt <= qt && kt < NK; ++kt) {
            uint k_base = kt * Bc;

            // Reload K,V
            for (uint i = li; i < Bc*HD; i += 256) {
                uint r = i/HD, c = i%HD, gr = k_base + r;
                if (gr < S) {
                    Ks[r*HD+c] = (bfloat)K[kv_off + gr*HD + c];
                    Vs[r*HD+c] = (bfloat)V[kv_off + gr*HD + c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint qr = 0; qr < QR; ++qr) {
                uint gqr = gqr0 + qr;
                if (gqr >= S) continue;

                uint abs_qr = sg_id * QR + qr;
                float q_mx = mx[abs_qr];
                float q_sm = sm[abs_qr];
                float q_sp = sp[abs_qr];

                uint kj = k_base + ln_id;
                float s_val = -INFINITY;
                if (kj <= gqr && kj < S) {
                    s_val = 0.0f;
                    for (uint d = 0; d < HD; ++d)
                        s_val += (float)Qs[(sg_id*QR+qr)*HD + d] * (float)Ks[ln_id*HD + d];
                    s_val *= scale;
                }

                float p_val = (kj <= gqr && kj < S) ? exp(s_val - q_mx) / q_sm : 0.0f;

                float dP_val = 0.0f;
                if (kj <= gqr && kj < S)
                    for (uint d = 0; d < HD; ++d)
                        dP_val += (float)dOs[(sg_id*QR+qr)*HD + d] * (float)Vs[ln_id*HD + d];

                float dS = p_val * (dP_val - q_sp);
                float dSs = dS * scale;

                // dQ[gqr,:] += dSs × K[j,:]
                if (kj <= gqr && kj < S)
                    for (uint d = ln_id; d < HD; d += WL)
                        dQa[qr][d] += dSs * (float)Ks[ln_id*HD + d];

                // Tile-local dK[j,:] += dSs × Q[gqr,:]
                // Tile-local dV[j,:] += P[j] × dO[gqr,:]
                if (kj <= gqr && kj < S)
                    for (uint d = ln_id; d < HD; d += WL) {
                        float dk_cur = (float)dKa[ln_id*HD + d];
                        float dv_cur = (float)dVa[ln_id*HD + d];
                        dk_cur += dSs * (float)Qs[(sg_id*QR+qr)*HD + d];
                        dv_cur += p_val * (float)dOs[(sg_id*QR+qr)*HD + d];
                        dKa[ln_id*HD + d] = (bfloat)dk_cur;
                        dVa[ln_id*HD + d] = (bfloat)dv_cur;
                    }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── Flush tile dK/dV → global (coalesced, each thread → unique element) ──
            for (uint i = li; i < Bc*HD; i += 256) {
                uint r = i / HD, c = i % HD;
                uint gkr = k_base + r;
                if (gkr < S) {
                    float dk_v = (float)dKa[i];
                    float dv_v = (float)dVa[i];
                    if (dk_v != 0.0f)
                        atomic_fetch_add_explicit(
                            (device atomic_float*)&dK[kv_off + gkr*HD + c],
                            dk_v, memory_order_relaxed);
                    if (dv_v != 0.0f)
                        atomic_fetch_add_explicit(
                            (device atomic_float*)&dV[kv_off + gkr*HD + c],
                            dv_v, memory_order_relaxed);
                    dKa[i] = (bfloat)0.0f; dVa[i] = (bfloat)0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Flush dQ to global ──
        for (uint qr = 0; qr < QR; ++qr) {
            uint gqr = gqr0 + qr;
            if (gqr >= S) continue;
            for (uint d = ln_id; d < HD; d += WL)
                dQ[q_off + gqr*HD + d] += dQa[qr][d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
