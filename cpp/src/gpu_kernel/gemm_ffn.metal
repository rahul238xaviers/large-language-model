#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// SPRINT 6b: HIGH-PERFORMANCE CUSTOM GEMM KERNEL FOR FFN PROJECTION
// ==============================================================================
// 
// WHAT: A tiled Matrix Multiplication kernel utilizing Apple Silicon's
//       simdgroup_matrix (8x8 hardware matrix cores) and Shared Threadgroup Memory.
// WHY: General-purpose GEMM kernels spend too much time on boundary checks and non-optimal
//      memory access. This kernel is custom-tailored for the FFN Projection shape:
//      Matrix A is [M, H] (tall/narrow), Matrix B is [H, 4H] (wide), producing C [M, 4H].
//
// TILING ARCHITECTURE:
// - Threadgroup Tile: 128 x 128 output elements of Matrix C.
// - Threadgroup Memory (Shared L1 Cache):
//   - shared_A: 128 x 32 elements (16 KB)
//   - shared_B: 32 x 128 elements (16 KB)
//   - Total shared memory used = 32 KB (fits perfectly inside the 32KB/64KB GPU core L1 limits).
// - Threadgroup Size: 128 threads (organized as 4 SIMD groups of 32 threads each).
// - Sub-tile: Each of the 4 SIMD groups computes a 64x64 segment of the 128x128 tile.
//   - Within the 64x64 segment, the 32 threads use hardware units to multiply 8x8 matrices.
// ==============================================================================

kernel void gemm_ffn(
    device const float* A       [[buffer(0)]], // Matrix A [M x K] (row-major)
    device const float* B       [[buffer(1)]], // Matrix B [K x N] (row-major)
    device float*       C       [[buffer(2)]], // Matrix C [M x N] (row-major)
    constant uint3&     dims    [[buffer(3)]], // dims.x = M, dims.y = N, dims.z = K
    uint2 tg_id   [[threadgroup_position_in_grid]], // Index of the 128x128 tile we are calculating
    uint2 lid     [[thread_position_in_threadgroup]], // Thread coordinates inside the threadgroup (0..31, 0..3)
    uint  simd_id [[simdgroup_index_in_threadgroup]], // SIMD group index (0, 1, 2, or 3)
    uint  lane_id [[thread_index_in_simdgroup]]) // Thread index inside the SIMD group (0..31)
{
  uint M = dims.x;
  uint N = dims.y;
  uint K = dims.z;

  // 1. Allocate Shared Threadgroup Memory (our whiteboard)
  // We use two buffers to store the current 128x32 slice of A and 32x128 slice of B.
  threadgroup float shared_A[128][32];
  threadgroup float shared_B[32][128];

  // 2. Define Sub-tile Mapping for each SIMD group
  // We arrange the 4 SIMD groups (0, 1, 2, 3) as a 2x2 grid to compute the 128x128 tile:
  // - SIMD 0: Top-Left [64 x 64]
  // - SIMD 1: Top-Right [64 x 64]
  // - SIMD 2: Bottom-Left [64 x 64]
  // - SIMD 3: Bottom-Right [64 x 64]
  uint sg_row = simd_id / 2; // 0 or 1
  uint sg_col = simd_id % 2; // 0 or 1

  // 3. Initialize Registers to Accumulate the Output
  // Each SIMD group is responsible for a 64x64 sub-tile of C.
  // We divide the 64x64 sub-tile into an 8x8 grid of 8x8 matrix blocks.
  // Each block is held in registers using the simdgroup_matrix helper.
  simdgroup_matrix<float, 8, 8> accum[8][8];
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  // Find the global row/column offset for this threadgroup tile
  uint tile_row_start = tg_id.y * 128;
  uint tile_col_start = tg_id.x * 128;

  // Flatten the thread ID within the threadgroup (from 0 to 127) for collaborative loading
  uint thread_linear_id = simd_id * 32 + lane_id;

  // 4. Slide our window along the K dimension
  for (uint k_offset = 0; k_offset < K; k_offset += 32) {
    
    // COLLABORATIVE LOAD: Load 128x32 slice of A into shared_A
    // Since we have 128 threads and 128x32 = 4096 elements to load:
    // Each thread loads exactly 32 elements. To keep memory reads coalesced,
    // threads load elements in contiguous strides.
    for (uint i = 0; i < 32; ++i) {
      // Linear index of the element this thread is loading
      uint load_idx = thread_linear_id + i * 128;
      uint load_row = load_idx / 32;
      uint load_col = load_idx % 32;

      uint global_row = tile_row_start + load_row;
      uint global_col = k_offset + load_col;

      if (global_row < M && global_col < K) {
        shared_A[load_row][load_col] = A[global_row * K + global_col];
      } else {
        shared_A[load_row][load_col] = 0.0f; // Padding out of bounds
      }
    }

    // COLLABORATIVE LOAD: Load 32x128 slice of B into shared_B
    // We have 128 threads and 32x128 = 4096 elements.
    for (uint i = 0; i < 32; ++i) {
      uint load_idx = thread_linear_id + i * 128;
      uint load_row = load_idx / 128;
      uint load_col = load_idx % 128;

      uint global_row = k_offset + load_row;
      uint global_col = tile_col_start + load_col;

      if (global_row < K && global_col < N) {
        shared_B[load_row][load_col] = B[global_row * N + global_col];
      } else {
        shared_B[load_row][load_col] = 0.0f; // Padding out of bounds
      }
    }

    // Synchronize to make sure all threads have finished loading A and B slices
    // before any thread starts computing with them.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Compute Inner Dot Products using hardware 8x8 matrix units
    // Loop over the K-slice width (32 elements). We move in steps of 8.
    for (uint k_step = 0; k_step < 32; k_step += 8) {
      
      // Load 8x8 slices of A and B from shared memory into registers
      simdgroup_matrix<float, 8, 8> mat_A[8];
      simdgroup_matrix<float, 8, 8> mat_B[8];

      // Load 8 rows of A for this SIMD group's rows
      for (int r = 0; r < 8; ++r) {
        // Load an 8x8 matrix from shared_A
        // Arguments: destination register, source pointer, stride, offset
        simdgroup_load(mat_A[r], &shared_A[sg_row * 64 + r * 8][k_step], 32, ulong2(0));
      }

      // Load 8 columns of B for this SIMD group's columns
      for (int c = 0; c < 8; ++c) {
        // Load an 8x8 matrix from shared_B
        simdgroup_load(mat_B[c], &shared_B[k_step][sg_col * 64 + c * 8], 128, ulong2(0));
      }

      // Perform hardware matrix multiply-accumulate (Tensor core equivalents)
      // Accumulates: accum[r][c] += mat_A[r] * mat_B[c]
      for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
          simdgroup_multiply_accumulate(accum[r][c], mat_A[r], mat_B[c], accum[r][c]);
        }
      }
    }

    // Synchronize again to make sure all threads have finished reading from shared memory
    // before we overwrite it with the next K-slice in the next iteration.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // 6. Write final outputs back to Unified RAM (Matrix C)
  // Each SIMD group stores its 64x64 sub-tile of accumulators.
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      uint global_row = tile_row_start + sg_row * 64 + r * 8;
      uint global_col = tile_col_start + sg_col * 64 + c * 8;

      // Make sure we are within the boundaries of Matrix C
      if (global_row < M && global_col < N) {
        // Store the 8x8 matrix result directly into global RAM at C
        simdgroup_store(accum[r][c], &C[global_row * N + global_col], N, ulong2(0));
      }
    }
  }
}
