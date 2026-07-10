// ==============================================================================
// TECHNICAL SPECIFICATION & ARCHITECTURAL REFERENCE: FUSED SwiGLU FFN GEMM KERNEL
// ==============================================================================
//
// 1. WHAT IS SwiGLU?
//    SwiGLU is an activation function used in modern LLMs (like LLaMA).
//    It is defined as: SwiGLU(x) = (x * W_gate) * sigmoid(x * W_gate) * (x * W_up)
//    where Swish(gate) = gate * sigmoid(gate) = gate / (1.0 + exp(-gate)).
//    So, SwiGLU(gate, up) = (gate / (1.0 + exp(-gate))) * up.
//
// 2. FUSION BENEFIT:
//    Normally, computing SwiGLU requires performing two separate matrix multiplications
//    (x * W_gate and x * W_up), writing both wide intermediate projection tensors to VRAM,
//    and then launching an element-wise kernel to load them back and compute the activation.
//    By fusing both multiplications and the activation directly inside the registers of this
//    kernel, we reduce memory traffic by 2x (only writing the final activated output back to RAM),
//    completely eliminating kernel launch overhead and intermediate VRAM read/writes.
//
// 3. TILE BLOCKING & MEMORY LAYOUT:
//    - Input A: [M x K] (row-major)
//    - Weight B_gate: [K x N] (row-major)
//    - Weight B_up: [K x N] (row-major)
//    - Output C: [M x N] (row-major)
//    We use static local caches (Shared Threadgroup Memory) of size:
//    - shared_A: 128 x 32 floats (16 KB)
//    - shared_B_gate: 32 x 128 floats (16 KB)
//    - shared_B_up: 32 x 128 floats (16 KB)
//    Total shared memory footprint = 48 KB, which fits within the 64 KB L1 cache limit.
// ==============================================================================

#include <metal_stdlib>
using namespace metal;

#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

kernel void gemm_ffn(
    device const float* A           [[buffer(0)]], // Matrix A [M x K] (row-major)
    device const float* B_gate      [[buffer(1)]], // Matrix B_gate [K x N] (row-major)
    device const float* B_up        [[buffer(2)]], // Matrix B_up [K x N] (row-major)
    device float*       C           [[buffer(3)]], // Matrix C [M x N] (row-major)
    constant uint3&     dims        [[buffer(4)]], // dims.x = M, dims.y = N, dims.z = K
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint2 lid     [[thread_position_in_threadgroup]],
    uint  simd_id [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]])
{
  uint M = dims.x;
  uint N = dims.y;
  uint K = dims.z;

  // 1. Allocate Shared Memory (Threadgroup Memory)
  threadgroup float shared_A[TILE_M][TILE_K];
  threadgroup float shared_B_gate[TILE_K][TILE_N];
  threadgroup float shared_B_up[TILE_K][TILE_N];

  // Arrange the 4 SIMDgroups in a 2x2 grid to compute the 128x128 tile
  uint sg_row = simd_id / 2; // 0 or 1
  uint sg_col = simd_id % 2; // 0 or 1

  // 2. Initialize Registers to Accumulate Outputs (Doubled accumulators)
  simdgroup_matrix<float, 8, 8> accum_gate[8][8];
  simdgroup_matrix<float, 8, 8> accum_up[8][8];

  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      accum_gate[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
      accum_up[r][c]   = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  uint tile_row_start = tg_id.y * 128;
  uint tile_col_start = tg_id.x * 128;

  // Flatten the thread ID (0..127) inside the threadgroup for collaborative loading
  uint thread_linear_id = simd_id * 32 + lane_id;

  // 3. Slide our window along the K dimension
  for (uint k_offset = 0; k_offset < K; k_offset += 32) {
    
    // COLLABORATIVE LOAD: Load 128x32 slice of A into shared_A
    for (uint i = 0; i < 32; ++i) {
      uint load_idx = thread_linear_id + i * 128; // 128 threads work cleanly in line strides
      uint load_row = load_idx / 32;
      uint load_col = load_idx % 32;

      uint global_row = tile_row_start + load_row;
      uint global_col = k_offset + load_col;

      if (global_row < M && global_col < K) {
        shared_A[load_row][load_col] = A[global_row * K + global_col];
      } else {
        shared_A[load_row][load_col] = 0.0f;
      }
    }

    // COLLABORATIVE LOAD: Load 32x128 slice of B_gate into shared_B_gate
    for (uint i = 0; i < 32; ++i) {
      uint load_idx = thread_linear_id + i * 128;
      uint load_row = load_idx / 128;
      uint load_col = load_idx % 128;

      uint global_row = k_offset + load_row;
      uint global_col = tile_col_start + load_col;

      if (global_row < K && global_col < N) {
        shared_B_gate[load_row][load_col] = B_gate[global_row * N + global_col];
      } else {
        shared_B_gate[load_row][load_col] = 0.0f;
      }
    }

    // COLLABORATIVE LOAD: Load 32x128 slice of B_up into shared_B_up
    for (uint i = 0; i < 32; ++i) {
      uint load_idx = thread_linear_id + i * 128;
      uint load_row = load_idx / 128;
      uint load_col = load_idx % 128;

      uint global_row = k_offset + load_row;
      uint global_col = tile_col_start + load_col;

      if (global_row < K && global_col < N) {
        shared_B_up[load_row][load_col] = B_up[global_row * N + global_col];
      } else {
        shared_B_up[load_row][load_col] = 0.0f;
      }
    }

    // Handshake execution barrier
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Compute Inner Dot Products using hardware matrix units
    for (uint k_step = 0; k_step < 32; k_step += 8) {
      for (int c = 0; c < 8; ++c) {
        simdgroup_matrix<float, 8, 8> tile_B_gate, tile_B_up;
        simdgroup_load(tile_B_gate, &shared_B_gate[k_step][sg_col * 64 + c * 8], 128, ulong2(0));
        simdgroup_load(tile_B_up, &shared_B_up[k_step][sg_col * 64 + c * 8], 128, ulong2(0));

        for (int r = 0; r < 8; ++r) {
          simdgroup_matrix<float, 8, 8> tile_A;
          simdgroup_load(tile_A, &shared_A[sg_row * 64 + r * 8][k_step], 32, ulong2(0));

          simdgroup_multiply_accumulate(accum_gate[r][c], tile_A, tile_B_gate, accum_gate[r][c]);
          simdgroup_multiply_accumulate(accum_up[r][c], tile_A, tile_B_up, accum_up[r][c]);
        }
      }
    }

    // Handshake execution barrier
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // 5. In-register SwiGLU Activation and VRAM Write-Back
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      
      auto gate_elements = accum_gate[r][c].thread_elements();
      auto up_elements   = accum_up[r][c].thread_elements();

      // Apply SwiGLU and write back directly to the actual matrix elements
      // NOTE: simdgroup_matrix<float,8,8> owns exactly 2 floats per lane
      for (ushort i = 0; i < 2; ++i) {
        float gate_val = gate_elements[i];
        float up_val   = up_elements[i];
        accum_gate[r][c].thread_elements()[i] = (gate_val / (1.0f + exp(-gate_val))) * up_val;
      }

      uint global_row = tile_row_start + sg_row * 64 + r * 8;
      uint global_col = tile_col_start + sg_col * 64 + c * 8;

      if (global_row < M && global_col < N) {
        // Store the result directly into Matrix C using simdgroup_store
        simdgroup_store(accum_gate[r][c], &C[global_row * N + global_col], N, ulong2(0));
      }
    }
  }
}
