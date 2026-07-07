// ==============================================================================
// TECHNICAL SPECIFICATION & ARCHITECTURAL REFERENCE: PROJECTION GEMM KERNEL
// ==============================================================================
//
// 1. WHAT IS GEMM?
//    GEMM stands for General Matrix Multiplication. In this kernel, we compute
//    C = A * B, where Matrix A represents input activations [M x K], Matrix B 
//    represents projection weights [K x N], and Matrix C is the output [M x N].
//    All matrices are stored in row-major contiguous memory.
//
// 2. WHO CALLS THIS KERNEL AND WHY?
//    This kernel is invoked by the host C++ engine via the Objective-C++ bridge
//    (MetalBridge.mm). We use it instead of Apple's MPS (Metal Performance Shaders)
//    to achieve zero-overhead execution and fine-grained control over cache sizes,
//    allowing us to tile specifically for the projection shape [M, H] x [H, H].
//
// 3. HOW DOES THIS FIT IN LLM TRAINING?
//    In a Transformer architecture, the projection layers (such as QKV projections 
//    and output projection layers) constitute a massive portion of the training
//    workload. This kernel accelerates both the forward inference/training projections
//    and the backward gradient updates by keeping intermediate data local to the core.
//
// 4. WHAT IS MSL AND HOW IS IT TRIGGERED?
//    MSL (Metal Shading Language) is a unified GPU programming language based on C++14.
//    At build time, MSL source files are compiled by CMake (via xcrun) into `.air` 
//    intermediate bytecode and bundled into a `default.metallib` binary archive.
//    At runtime, the host CPU loads this library, compiles it into binary machine 
//    instructions (MTLComputePipelineState), sets the GPU pointer parameters, 
//    and dispatches a grid of threads to trigger execution.
//
// 5. GPU HARDWARE STRUCTURE (Apple Silicon M3 Ultra):
//    - The GPU features 80 physical GPU Cores.
//    - Each GPU Core contains 4 physical SIMD units (ALUs).
//    - Each SIMD unit executes a "SIMDgroup" of exactly 32 threads in physical lockstep.
//    - At any given clock cycle, each core executes 4 SIMDgroups (128 threads) 
//      physically in parallel, while managing a queue of up to 32 SIMDgroups (1024 threads)
//      to hide memory latency.
//
// 6. MATRIX LAYOUT & WORKLOAD PARTITIONING:
//    - The output Matrix C [M x N] is partitioned into 2D tiles of size 64 x 64.
//    - A single Threadgroup of 64 threads is assigned to compute one 64 x 64 tile.
//    - To avoid slow VRAM reads, we allocate 16 KB of Shared Threadgroup Memory (L1 cache) 
//      divided into shared_A [64 x 32] and shared_B [32 x 64].
//    - Threads load contiguous blocks of 32 floats (128 bytes) in parallel, matching the
//      GPU's memory coalescing bus width.
//
// 7. ARITHMETIC PERFORMANCE (Per Clock Cycle):
//    - Inside each SIMDgroup, the threads execute Fused Multiply-Add (FMA) instructions
//      (d = a * b + c) directly on the physical hardware matrix cores (AMX/Tensor cores).
//    - Each FMA instruction performs 2 floating-point operations (1 multiply, 1 add) 
//      in a single clock cycle, maximizing hardware occupancy.
//
// 8. THE LIFECYCLE OF THE WORKLOAD (Thread & SIMDgroup Workflow):
//    - Step A: The GPU Global Scheduler assigns a Threadgroup (64 threads) to a Core.
//    - Step B: The Core Scheduler splits the 64 threads into SIMDgroup 0 (threads 0-31)
//              and SIMDgroup 1 (threads 32-63).
//    - Step C (The K-Loop): We iterate along the inner dimension K in chunks of 32:
//      - The 64 threads collaboratively load data from RAM into shared_A and shared_B.
//      - A threadgroup_barrier forces all execution to halt until L1 memory is populated.
//      - SIMDgroup 0 loads data from L1 and computes the left half of the tile (64x32).
//      - SIMDgroup 1 loads data from L1 and computes the right half of the tile (64x32).
//      - Both SIMDgroups run matrix math inside their registers.
//      - A second barrier ensures all math reading from L1 is complete before L1 is overwritten.
//    - Step D: Once all K iterations are complete, the threads store their register matrices
//              directly back into global RAM at C.
// ==============================================================================

#include<metal_stdlib>
using namespace metal;

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

#define SIMD_MAX_THREADS 32

kernel void gemm_proj(
    device const float* A       [[buffer(0)]], // Matrix A [M x K] (row-major)
    device const float* B       [[buffer(1)]], // Matrix B [K x N] (row-major)
    device float*       C       [[buffer(2)]], // Matrix C [M x N] (row-major)
    constant uint3&     dims    [[buffer(3)]], // dims.x = M, dims.y = N, dims.z = K
    uint2 tg_id   [[threadgroup_position_in_grid]], // Index of the 64x64 tile we are calculating
    uint2 lid     [[thread_position_in_threadgroup]], // Thread coordinates inside the threadgroup (0..31, 0..3)
    uint  simd_id [[simdgroup_index_in_threadgroup]], // SIMD group index (0, 1, 2, or 3)
    uint  lane_id [[thread_index_in_simdgroup]]) // Thread index inside the SIMD group (0..31)){
    {

    threadgroup float shared_A[TILE_M][TILE_K];
    threadgroup float shared_B[TILE_K][TILE_N];

    float accum_array[TILE_M] = {0.0f};

    uint tile_start_col = TILE_N * tg_id.x;
    uint tile_start_row = TILE_M * tg_id.y;

    uint thread_col = simd_id * SIMD_MAX_THREADS + lane_id ;

    uint global_col = tile_start_col + thread_col;
    uint global_row = tile_start_row + thread_col;


    //Load the data from the buffer to the threadgroup memory
    // we are loading the matrix in the tile size of TILE_MxTILE_K for A and TILE_KxTILE_N for B
    // A matrix is loaded in the row major order
    // B matrix is loaded in the column major order
    //Example would be :- if M=128, N=256, K=512, threads = 64
    // 64 thread would mean 2 SIMD groups in Apple Silicon.
    for(uint k = 0; k < dims.z; k+= TILE_K) {
      for(uint i =0; i < TILE_K; i++) { 
        if(k + i >= dims.z || dims.x <= global_row) {
          shared_A[thread_col][i]  = 0.0f;
        }
        else{ 
          shared_A[thread_col][i] = A[(tile_start_row + thread_col) * dims.z  +  k + i];
        }
      }

      for(uint i = 0; i < TILE_K; i++) {

         if(k + i >= dims.z || dims.y <= global_col) {
           shared_B[i][thread_col] = 0.0f;

         }
         else{
           shared_B[i][thread_col] = B[(k + i )* dims.y + global_col];
         }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } 

    for(size_t i = 0; i < TILE_K ; i++) {
      for(size_t j = 0; j < TILE_M ; j++) {
        accum_array[j]+=shared_A[j][i] * shared_B[i][thread_col];
      }
    }

    

    }

    
