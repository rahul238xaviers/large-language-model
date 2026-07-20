// ==============================================================================
// KERNEL: gemm_backward
// WHAT: Performs transposed Matrix Multiplication: C [M x N] += A^T [M x K] * B [K x N]
// WHY:  In backward propagation, weight gradients are computed as:
//       grad_W [hidden, out] += activation.T [hidden, batch*seq] * grad_output [batch*seq, out]
//       Matrix A is stored in memory as [K x M]. This kernel reads A transposed.
// ==============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

kernel void gemm_backward(
    device const float* A [[buffer(0)]], // Shape [K x M] in memory, accessed transposed as [M x K]
    device const float* B [[buffer(1)]], // Shape [K x N] in memory
    device float*       C [[buffer(2)]], // Shape [M x N] output gradient accumulator
    constant uint3&  dims [[buffer(3)]], // x = M, y = N, z = K
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_A[TILE_M][TILE_K];
    threadgroup float shared_B[TILE_K][TILE_N];

    simdgroup_matrix<float, 8, 8> accum[8][4];
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint tile_start_col = TILE_N * tg_id.x;
    uint tile_start_row = TILE_M * tg_id.y;
    uint thread_linear_id = simd_id * 32 + lane_id;

    for (uint k = 0; k < dims.z; k += TILE_K) {
        // Load A transposed into shared_A [64 x 32]
        // A in global memory is [K x M]. A^T[row, col] = A[col * M + row]
        for (uint i = 0; i < 32; ++i) {
            uint load_idx = thread_linear_id + i * 64;
            uint load_row = load_idx / 32;
            uint load_col = load_idx % 32;

            uint global_row = tile_start_row + load_row;
            uint global_col = k + load_col;

            if (global_row < dims.x && global_col < dims.z) {
                // Read transposed: global_col is row in A, global_row is col in A
                shared_A[load_row][load_col] = A[global_col * dims.x + global_row];
            } else {
                shared_A[load_row][load_col] = 0.0f;
            }
        }

        // Load B [32 x 64] into shared_B
        for (uint i = 0; i < 32; ++i) {
            uint load_idx = thread_linear_id + i * 64;
            uint load_row = load_idx / 64;
            uint load_col = load_idx % 64;

            uint global_row = k + load_row;
            uint global_col = tile_start_col + load_col;

            if (global_row < dims.z && global_col < dims.y) {
                shared_B[load_row][load_col] = B[global_row * dims.y + global_col];
            } else {
                shared_B[load_row][load_col] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_step = 0; k_step < TILE_K; k_step += 8) {
            simdgroup_matrix<float, 8, 8> mat_A[8];
            simdgroup_matrix<float, 8, 8> mat_B[4];

            for (int r = 0; r < 8; ++r) {
                simdgroup_load(mat_A[r], &shared_A[r * 8][k_step], TILE_K, ulong2(0));
            }
            for (int c = 0; c < 4; ++c) {
                simdgroup_load(mat_B[c], &shared_B[k_step][simd_id * 32 + c * 8], TILE_N, ulong2(0));
            }
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 4; ++c) {
                    simdgroup_multiply_accumulate(accum[r][c], mat_A[r], mat_B[c], accum[r][c]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Accumulate results into Matrix C [M x N] in global memory
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            uint global_row = tile_start_row + r * 8;
            uint global_col = tile_start_col + simd_id * 32 + c * 8;

            simdgroup_matrix<float, 8, 8> mat_C;
            simdgroup_load(mat_C, &C[global_row * dims.y + global_col], dims.y, ulong2(0));
            simdgroup_multiply_accumulate(accum[r][c], accum[r][c], simdgroup_matrix<float, 8, 8>(1.0f), mat_C);
            simdgroup_store(accum[r][c], &C[global_row * dims.y + global_col], dims.y, ulong2(0));
        }
    }
}
