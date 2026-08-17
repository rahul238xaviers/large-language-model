// ═══════════════════════════════════════════════════════════════════════════════
// fused_attn_bwd — Tile-coalesced FlashAttention-2 backward
// ═══════════════════════════════════════════════════════════════════════════════
// Fix: dQa moved from registers to shared memory (eliminates 256-register spill)
// Grid: [B, nH, 1]  — one TG per (batch,query_head).  512 TGs × 256 threads.
// Tiling: Br=32, Bc=32
// Shared memory (32 KB = 50% of 64 KB):
//   Qs/dOs/Ks/Vs/mx/sm/sp: 24 KB
//   dKa/dVa/sh_dQa: 12 KB
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

constant uint S=1024, HD=64, Br=32, Bc=32;
constant uint NQ=S/Br, NK=S/Bc;
constant uint WL=32;

kernel void fused_attn_bwd(
    device const bfloat* Q      [[buffer(0)]],
    device const bfloat* K      [[buffer(1)]],
    device const bfloat* V      [[buffer(2)]],
    device const bfloat* dO     [[buffer(3)]],
    device float*         dQ    [[buffer(4)]],
    device float*         dK    [[buffer(5)]],
    device float*         dV    [[buffer(6)]],
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
    uint QR = Br / 8;

    // ── Shared memory ──
    threadgroup bfloat Qs [2048];     // 4 KB
    threadgroup bfloat dOs[2048];     // 4 KB
    threadgroup bfloat Ks [2048];     // 4 KB
    threadgroup bfloat Vs [2048];     // 4 KB
    threadgroup float  mx [32];       // 128 B
    threadgroup float  sm [32];       // 128 B
    threadgroup float  sp [32];       // 128 B
    // dK/dV are accumulated directly into global FP32 buffers via device
    // atomics (threadgroup atomics unsupported in this MSL version).
    threadgroup bfloat sh_dQa[2048];  // 4 KB (dQ accumulator, was register)
    // Total: 32,768 B = 32 KB

    uint q_off  = (b * n_heads + h) * S * HD;
    uint kv_off = (b * n_kv + kv_h)  * S * HD;

    for (uint qt = 0; qt < NQ; ++qt) {
        uint q_base = qt * Br;
        uint gqr0 = q_base + sg_id * QR;

        // Zero sh_dQa for this Q-tile
        for (uint i = li; i < Br*HD; i += 256) sh_dQa[i] = (bfloat)0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Load Q, dO ──
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
        // PHASE 1 — online softmax
        // ══════════════════════════════════════════════════════════════════
        for (uint kt = 0; kt <= qt && kt < NK; ++kt) {
            uint k_base = kt * Bc;

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

                uint kj = k_base + ln_id;
                float s_val = -INFINITY;
                if (kj <= gqr && kj < S) {
                    s_val = 0.0f;
                    for (uint d = 0; d < HD; ++d)
                        s_val += (float)Qs[(sg_id*QR+qr)*HD + d] * (float)Ks[ln_id*HD + d];
                    s_val *= scale;
                }

                float wmax = s_val;
                for (uint m = 16; m > 0; m >>= 1) wmax = max(wmax, simd_shuffle_down(wmax, m));
                float gmax = simd_shuffle(wmax, 0);

                uint abs_qr = sg_id * QR + qr;
                float cur_mx = mx[abs_qr];
                float new_mx = gmax > cur_mx ? gmax : cur_mx;
                mx[abs_qr] = new_mx;

                float rescale = exp(cur_mx - new_mx);

                float p_val  = (kj <= gqr && kj < S) ? exp(s_val - new_mx) : 0.0f;
                float dP_val = 0.0f;
                if (kj <= gqr && kj < S)
                    for (uint d = 0; d < HD; ++d)
                        dP_val += (float)dOs[(sg_id*QR+qr)*HD + d] * (float)Vs[ln_id*HD + d];

                float p_sum = p_val;
                for (uint m = 16; m > 0; m >>= 1) p_sum += simd_shuffle_down(p_sum, m);
                float tile_l = simd_shuffle(p_sum, 0);

                float pd_sum = p_val * dP_val;
                for (uint m = 16; m > 0; m >>= 1) pd_sum += simd_shuffle_down(pd_sum, m);
                float tile_sp = simd_shuffle(pd_sum, 0);

                if (ln_id == 0) {
                    sm[abs_qr] = cur_mx > -INFINITY ? sm[abs_qr] * rescale + tile_l : tile_l;
                    sp[abs_qr] = cur_mx > -INFINITY ? sp[abs_qr] * rescale + tile_sp : tile_sp;
                }
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE 2 — gradients
        // ══════════════════════════════════════════════════════════════════
        for (uint kt = 0; kt <= qt && kt < NK; ++kt) {
            uint k_base = kt * Bc;

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

                // Accumulate dQ in shared memory (bfloat tile-local, was float register)
                if (kj <= gqr && kj < S)
                    for (uint d = ln_id; d < HD; d += WL) {
                        float dq_cur = (float)sh_dQa[(sg_id*QR+qr)*HD + d];
                        dq_cur += dSs * (float)Ks[ln_id*HD + d];
                        sh_dQa[(sg_id*QR+qr)*HD + d] = (bfloat)dq_cur;
                    }

                // dK/dV: device atomic adds into the global FP32 buffers.
                // Correct across all 8 warps + Q-tiles (no shared-memory race).
                if (kj <= gqr && kj < S)
                    for (uint d = ln_id; d < HD; d += WL) {
                        float dk_add = dSs * (float)Qs[(sg_id*QR+qr)*HD + d];
                        float dv_add = p_val * (float)dOs[(sg_id*QR+qr)*HD + d];
                        if (dk_add != 0.0f)
                            atomic_fetch_add_explicit(
                                (device atomic_float*)&dK[kv_off + kj*HD + d],
                                dk_add, memory_order_relaxed);
                        if (dv_add != 0.0f)
                            atomic_fetch_add_explicit(
                                (device atomic_float*)&dV[kv_off + kj*HD + d],
                                dv_add, memory_order_relaxed);
                    }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Flush dQ from shared memory to global (FP32 buffer) ──
        for (uint qr = 0; qr < QR; ++qr) {
            uint gqr = gqr0 + qr;
            if (gqr >= S) continue;
            for (uint d = ln_id; d < HD; d += WL)
                dQ[q_off + gqr*HD + d] += (float)sh_dQa[(sg_id*QR+qr)*HD + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
