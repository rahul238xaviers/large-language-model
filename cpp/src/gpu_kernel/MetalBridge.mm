/**
 * @file MetalBridge.mm
 * @brief Implementation of the C++ to Metal GPU runtime bridge.
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Orchestrates GPU memory buffers creation, manages pipeline compute states,
 * and submits execution dispatches to Apple Silicon hardware (using MPS matrix
 * cores).
 */

#include "gpu_kernel/MetalBridge.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstddef>
#include <iostream>
#import <simd/simd.h>
#include <unordered_map>

namespace metal_bridge {

// device (MTLDevice): Represents the physical GPU hardware.
// Think of it as the parent manager of all GPU memory and pipeline allocations.
static id<MTLDevice> device = nil;

// commandQueue (MTLCommandQueue): A thread-safe FIFO (first-in-first-out)
// queue. This is the conveyor belt that carries instruction packets (command
// buffers) from CPU to GPU.
static id<MTLCommandQueue> commandQueue = nil;

// pipelineStateFFN (MTLComputePipelineState): The compiled microcode for our
// tiled FFN GEMM.
static id<MTLComputePipelineState> pipelineStateFFN = nil;
static id<MTLComputePipelineState> pipelineStateProj = nil;
static id<MTLComputePipelineState> pipelineStateGQA = nil;
static id<MTLComputePipelineState> pipelineStateRMSForward = nil;
static id<MTLComputePipelineState> pipelineStateRMSBackwardDX = nil;
static id<MTLComputePipelineState> pipelineStateSwiGLUBackward = nil;
static id<MTLComputePipelineState> pipelineStateRoPEBackward = nil;
static id<MTLComputePipelineState> pipelineStateGQABackward = nil;
static id<MTLComputePipelineState> pipelineStateAdamW = nil;
static id<MTLComputePipelineState> pipelineStateResidualAdd = nil;
static id<MTLComputePipelineState> pipelineStateEmbeddingFwd = nil;
static id<MTLComputePipelineState> pipelineStateCrossEntropy = nil;
static id<MTLComputePipelineState> pipelineStateGEMMBackward = nil;
static id<MTLComputePipelineState> pipelineStateGEMMProjTransB = nil;
static id<MTLComputePipelineState> pipelineStateReshape4D = nil;
static id<MTLComputePipelineState> pipelineStateReshape3D = nil;
static id<MTLComputePipelineState> pipelineStateRoPEForward = nil;

static float last_step_loss = 0.0f;
static bool initialized = false;

static id<MTLCommandBuffer> activeCmdBuffer = nil;
static id<MTLComputeCommandEncoder> activeEncoder = nil;
static bool batchActive = false;

// Legacy buffer cache (unused — kept for compatibility with existing clear() calls)
struct CachedBuffer {
  id<MTLBuffer> buf;
  bool is_persistent;
};
static std::unordered_map<const void *, CachedBuffer> bufferCache;

// Size-bucketed pool for intermediate activation buffers (recycled step to step)
static std::unordered_map<size_t, std::vector<id<MTLBuffer>>> sizePool;

// persistent weight/optimizer state cache (lives forever)
static std::unordered_map<const void *, id<MTLBuffer>> weightCache;

// step-local activation cache (cleared each training step)
static std::unordered_map<const void *, id<MTLBuffer>> stepCache;

// buffers allocated during the current step (for recycling at step end)
static std::vector<std::pair<size_t, id<MTLBuffer>>> activeAllocationsThisStep;

struct CopyBackTask {
  void *dest;
  id<MTLBuffer> src;
  size_t size;
};
static std::vector<CopyBackTask> copyBackQueue;

static void run_copy_back_tasks() {
  for (const auto &task : copyBackQueue) {
    memcpy(task.dest, [task.src contents], task.size);
  }
  copyBackQueue.clear();
}

static id<MTLBuffer> get_or_create_buffer(const void *ptr, size_t bytes,
                                          bool is_write = false, bool is_persistent = false,
                                          bool is_host_input = false);

// Profiling statistics definitions
double accum_gpu_time_ms = 0.0;
double accum_cpu_time_ms = 0.0;
size_t count_gpu_calls = 0;
size_t count_cpu_calls = 0;

/**
 * @brief Check if the GPU device and compile pipeline states are fully loaded
 * and operational.
 *
 * @return true If available.
 * @return false Otherwise.
 */
bool is_available() {
  return (device != nil) && (pipelineStateFFN != nil) &&
         (pipelineStateProj != nil) && (pipelineStateGQA != nil);
}

/**
 * @brief Reset accumulated GPU and CPU execution profiling timers and counts to
 * 0.
 */
void reset_profile_stats() {
  accum_gpu_time_ms = 0.0;
  accum_cpu_time_ms = 0.0;
  count_gpu_calls = 0;
  count_cpu_calls = 0;
}

/**
 * @brief Searches for the Apple GPU device and compiles the Metal shader
 * libraries.
 */
void initialize() {
  if (initialized)
    return;
  // 1. MTLCreateSystemDefaultDevice()
  // WHAT: A system-level function that searches the Apple Silicon chip for the
  // GPU core block
  //       and initializes a driver interface for it.
  // WHY: This interface is required to create all other Metal objects (queues,
  // buffers, libraries).
  device = MTLCreateSystemDefaultDevice();
  if (!device) {
    std::cerr << "Failed to find a Metal GPU device!" << std::endl;
    return;
  }
  // [device.name UTF8String] calls the 'name' property on the device object
  // (which returns an NSString like @"Apple M5") and converts it to a standard
  // C++ char* pointer.
  std::cout << "Metal GPU initialized: " << [device.name UTF8String]
            << std::endl;

  // 2. [device newCommandQueue]
  // WHAT: A method called on the device object that allocates a command queue
  // structure. WHY: The GPU is asynchronous. We cannot send instructions
  // directly to the GPU cores.
  //      We must submit them to this queue, which handles the hardware
  //      execution.
  // C++ equivalent: commandQueue = device->newCommandQueue();
  commandQueue = [device newCommandQueue];

  NSError *error = nil;

  // 3. [device newDefaultLibrary]
  // WHAT: Searches the application's running bundle/directory for a
  // pre-compiled
  //       shader library file named 'default.metallib' and loads it into
  //       memory.
  // WHY: A shader library contains all of our compiled GPU functions. We must
  // load it
  //      into RAM first so that we can look inside it and locate the specific
  //      functions we want.
  id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];

  // 4. [device newLibraryWithURL:error:]
  // WHAT: Loads the compiled binary library from a URL path in the filesystem
  // instead of a plain string. WHY: Apple deprecated the string-based
  // 'newLibraryWithFile' method in macOS 13 to enforce the use of URLs,
  //      which are more secure and support modern sandboxing and network paths
  //      uniformly.
  // PARAMETERS:
  // - libraryURL: An NSURL object representing the file path
  // 'default.metallib'.
  // - error: Pointer to our NSError object to capture compilation/loading
  // diagnostics. C++ Translation: NSURL* libraryURL = [NSURL
  // fileURLWithPath:@"default.metallib"]; defaultLibrary =
  // device->newLibraryWithURL(libraryURL, &error);
  if (!defaultLibrary) {
    NSURL *libraryURL = [NSURL fileURLWithPath:@"default.metallib"];
    defaultLibrary = [device newLibraryWithURL:libraryURL error:&error];
  }
  if (!defaultLibrary) {
    std::cerr << "Failed to load default.metallib! Error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
    return;
  }

  // 7. [defaultLibrary newFunctionWithName:@"gemm_ffn"]
  // WHAT: Finds the FFN GEMM kernel in our compiled shader library.
  id<MTLFunction> ffnFunction =
      [defaultLibrary newFunctionWithName:@"gemm_ffn"];
  if (!ffnFunction) {
    std::cerr << "Failed to find kernel function 'gemm_ffn' in library!"
              << std::endl;
    return;
  }

  id<MTLFunction> projFunction =
      [defaultLibrary newFunctionWithName:@"gemm_proj"];
  if (!projFunction) {
    std::cerr << "Failed to find kernel function 'gemm_proj' in library!"
              << std::endl;
    return;
  }

  // 8. Compile the FFN kernel function into pipelineStateFFN
  pipelineStateFFN = [device newComputePipelineStateWithFunction:ffnFunction
                                                           error:&error];
  if (!pipelineStateFFN) {
    std::cerr << "Failed to compile pipeline state for 'gemm_ffn'! Error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'gemm_ffn'!"
            << std::endl;

  pipelineStateProj = [device newComputePipelineStateWithFunction:projFunction
                                                            error:&error];
  if (!pipelineStateProj) {
    std::cerr << "Failed to compile pipeline state for 'gemm_proj'! Error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'gemm_proj'!"
            << std::endl;

  id<MTLFunction> gqaFunction =
      [defaultLibrary newFunctionWithName:@"gemm_gqa"];
  if (!gqaFunction) {
    std::cerr << "Failed to find kernel function 'gemm_gqa' in library!"
              << std::endl;
    return;
  }
  pipelineStateGQA = [device newComputePipelineStateWithFunction:gqaFunction
                                                           error:&error];
  if (!pipelineStateGQA) {
    std::cerr << "Failed to compile pipeline state for 'gemm_gqa'! Error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'gemm_gqa'!"
            << std::endl;

  id<MTLFunction> rmsForwardFunc =
      [defaultLibrary newFunctionWithName:@"rms_norm_forward"];
  if (!rmsForwardFunc) {
    std::cerr << "Failed to find kernel function 'rms_norm_forward' in library!"
              << std::endl;
    return;
  }
  pipelineStateRMSForward =
      [device newComputePipelineStateWithFunction:rmsForwardFunc error:&error];
  if (!pipelineStateRMSForward) {
    std::cerr
        << "Failed to compile pipeline state for 'rms_norm_forward'! Error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout
      << "Metal Pipeline State compiled successfully for 'rms_norm_forward'!"
      << std::endl;

  id<MTLFunction> rmsBackwardDXFunc =
      [defaultLibrary newFunctionWithName:@"rms_norm_backward_dx"];
  if (!rmsBackwardDXFunc) {
    std::cerr
        << "Failed to find kernel function 'rms_norm_backward_dx' in library!"
        << std::endl;
    return;
  }
  pipelineStateRMSBackwardDX =
      [device newComputePipelineStateWithFunction:rmsBackwardDXFunc
                                            error:&error];
  if (!pipelineStateRMSBackwardDX) {
    std::cerr << "Failed to compile pipeline state for 'rms_norm_backward_dx'! "
                 "Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for "
               "'rms_norm_backward_dx'!"
            << std::endl;

  id<MTLFunction> swigluBackwardFunc =
      [defaultLibrary newFunctionWithName:@"swiglu_backward"];
  if (!swigluBackwardFunc) {
    std::cerr << "Failed to find kernel function 'swiglu_backward' in library!"
              << std::endl;
    return;
  }
  pipelineStateSwiGLUBackward =
      [device newComputePipelineStateWithFunction:swigluBackwardFunc
                                            error:&error];
  if (!pipelineStateSwiGLUBackward) {
    std::cerr << "Failed to compile pipeline state for 'swiglu_backward'! "
                 "Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for "
               "'swiglu_backward'!"
            << std::endl;

  id<MTLFunction> ropeBackwardFunc =
      [defaultLibrary newFunctionWithName:@"rope_backward"];
  if (!ropeBackwardFunc) {
    std::cerr << "Failed to find kernel function 'rope_backward' in library!"
              << std::endl;
    return;
  }
  pipelineStateRoPEBackward =
      [device newComputePipelineStateWithFunction:ropeBackwardFunc
                                            error:&error];
  if (!pipelineStateRoPEBackward) {
    std::cerr << "Failed to compile pipeline state for 'rope_backward'! "
                 "Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for "
               "'rope_backward'!"
            << std::endl;

  id<MTLFunction> gqaBackwardFunc =
      [defaultLibrary newFunctionWithName:@"gqa_backward"];
  if (!gqaBackwardFunc) {
    std::cerr << "Failed to find kernel function 'gqa_backward' in library!"
              << std::endl;
    return;
  }
  pipelineStateGQABackward =
      [device newComputePipelineStateWithFunction:gqaBackwardFunc
                                            error:&error];
  if (!pipelineStateGQABackward) {
    std::cerr << "Failed to compile pipeline state for 'gqa_backward'! "
                 "Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for "
               "'gqa_backward'!"
            << std::endl;

  id<MTLFunction> adamwFunc =
      [defaultLibrary newFunctionWithName:@"adamw_step"];
  if (!adamwFunc) {
    std::cerr << "Failed to find kernel function 'adamw_step' in library!"
              << std::endl;
    return;
  }
  pipelineStateAdamW =
      [device newComputePipelineStateWithFunction:adamwFunc
                                            error:&error];
  if (!pipelineStateAdamW) {
    std::cerr << "Failed to compile pipeline state for 'adamw_step'! "
                 "Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for "
               "'adamw_step'!"
            << std::endl;

  id<MTLFunction> residualAddFunc = [defaultLibrary newFunctionWithName:@"residual_add"];
  if (!residualAddFunc) {
    std::cerr << "Failed to find kernel function 'residual_add' in library!" << std::endl;
    return;
  }
  pipelineStateResidualAdd = [device newComputePipelineStateWithFunction:residualAddFunc error:&error];
  if (!pipelineStateResidualAdd) {
    std::cerr << "Failed to compile pipeline state for 'residual_add'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'residual_add'!" << std::endl;

  id<MTLFunction> embeddingFwdFunc = [defaultLibrary newFunctionWithName:@"embedding_forward"];
  if (!embeddingFwdFunc) {
    std::cerr << "Failed to find kernel function 'embedding_forward' in library!" << std::endl;
    return;
  }
  pipelineStateEmbeddingFwd = [device newComputePipelineStateWithFunction:embeddingFwdFunc error:&error];
  if (!pipelineStateEmbeddingFwd) {
    std::cerr << "Failed to compile pipeline state for 'embedding_forward'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'embedding_forward'!" << std::endl;

  id<MTLFunction> crossEntropyFunc = [defaultLibrary newFunctionWithName:@"cross_entropy"];
  if (!crossEntropyFunc) {
    std::cerr << "Failed to find kernel function 'cross_entropy' in library!" << std::endl;
    return;
  }
  pipelineStateCrossEntropy = [device newComputePipelineStateWithFunction:crossEntropyFunc error:&error];
  if (!pipelineStateCrossEntropy) {
    std::cerr << "Failed to compile pipeline state for 'cross_entropy'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'cross_entropy'!" << std::endl;

  id<MTLFunction> gemmBackwardFunc = [defaultLibrary newFunctionWithName:@"gemm_backward"];
  if (!gemmBackwardFunc) {
    std::cerr << "Failed to find kernel function 'gemm_backward' in library!" << std::endl;
    return;
  }
  pipelineStateGEMMBackward = [device newComputePipelineStateWithFunction:gemmBackwardFunc error:&error];
  if (!pipelineStateGEMMBackward) {
    std::cerr << "Failed to compile pipeline state for 'gemm_backward'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'gemm_backward'!" << std::endl;

  id<MTLFunction> gemmProjTransBFunc = [defaultLibrary newFunctionWithName:@"gemm_proj_trans_b"];
  if (!gemmProjTransBFunc) {
    std::cerr << "Failed to find kernel function 'gemm_proj_trans_b' in library!" << std::endl;
    return;
  }
  pipelineStateGEMMProjTransB = [device newComputePipelineStateWithFunction:gemmProjTransBFunc error:&error];
  if (!pipelineStateGEMMProjTransB) {
    std::cerr << "Failed to compile pipeline state for 'gemm_proj_trans_b'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'gemm_proj_trans_b'!" << std::endl;

  id<MTLFunction> reshape4DFunc = [defaultLibrary newFunctionWithName:@"reshape_to_4d"];
  if (!reshape4DFunc) {
    std::cerr << "Failed to find kernel function 'reshape_to_4d' in library!" << std::endl;
    return;
  }
  pipelineStateReshape4D = [device newComputePipelineStateWithFunction:reshape4DFunc error:&error];
  if (!pipelineStateReshape4D) {
    std::cerr << "Failed to compile pipeline state for 'reshape_to_4d'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'reshape_to_4d'!" << std::endl;

  id<MTLFunction> reshape3DFunc = [defaultLibrary newFunctionWithName:@"reshape_to_3d"];
  if (!reshape3DFunc) {
    std::cerr << "Failed to find kernel function 'reshape_to_3d' in library!" << std::endl;
    return;
  }
  pipelineStateReshape3D = [device newComputePipelineStateWithFunction:reshape3DFunc error:&error];
  if (!pipelineStateReshape3D) {
    std::cerr << "Failed to compile pipeline state for 'reshape_to_3d'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'reshape_to_3d'!" << std::endl;

  id<MTLFunction> ropeForwardFunc = [defaultLibrary newFunctionWithName:@"rope_forward"];
  if (!ropeForwardFunc) {
    std::cerr << "Failed to find kernel function 'rope_forward' in library!" << std::endl;
    return;
  }
  pipelineStateRoPEForward = [device newComputePipelineStateWithFunction:ropeForwardFunc error:&error];
  if (!pipelineStateRoPEForward) {
    std::cerr << "Failed to compile pipeline state for 'rope_forward'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'rope_forward'!" << std::endl;

  initialized = true;
}

void gemm_ffn(const float *a, const float *b_gate, const float *b_up, float *c,
              size_t M, size_t N, size_t K) {
  if (M == 0 || N == 0 || K == 0)
    return;

  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  id<MTLBuffer> bufferA = get_or_create_buffer(a, bytesA);
  id<MTLBuffer> bufferB_gate = get_or_create_buffer(b_gate, bytesB, false, true);
  id<MTLBuffer> bufferB_up = get_or_create_buffer(b_up, bytesB, false, true);
  id<MTLBuffer> bufferC = get_or_create_buffer(c, bytesC, true);

  if (!bufferA || !bufferB_gate || !bufferB_up || !bufferC) {
    std::cerr << "Failed to allocate Metal GEMM buffers for gemm_ffn!"
              << std::endl;
    return;
  }

  simd::uint3 dimensions = {(uint32_t)M, (uint32_t)N, (uint32_t)K};

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateFFN];
    [activeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferB_gate offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferB_up offset:0 atIndex:2];
    [activeEncoder setBuffer:bufferC offset:0 atIndex:3];
    [activeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:4];

    MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);

    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateFFN];
      [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferB_gate offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferB_up offset:0 atIndex:2];
      [computeEncoder setBuffer:bufferC offset:0 atIndex:3];
      [computeEncoder setBytes:&dimensions
                        length:sizeof(simd::uint3)
                       atIndex:4];

      MTLSize threadgroupsPerGrid =
          MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);

      MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];

      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}

void gemm_proj(const float *a, const float *b, float *c, size_t M, size_t N,
               size_t K) {
  if (M == 0 || N == 0 || K == 0)
    return;

  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  id<MTLBuffer> bufferA = get_or_create_buffer(a, bytesA);
  id<MTLBuffer> bufferB = get_or_create_buffer(b, bytesB, false, true);
  id<MTLBuffer> bufferC = get_or_create_buffer(c, bytesC, true);

  if (!bufferA || !bufferB || !bufferC) {
    std::cerr << "Failed to allocate Metal buffers for gemm_proj!" << std::endl;
    return;
  }

  simd::uint3 dimensions = {(uint32_t)M, (uint32_t)N, (uint32_t)K};

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateProj];
    [activeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferC offset:0 atIndex:2];
    [activeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];

    MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);

    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState:pipelineStateProj];
      [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
      [computeEncoder setBytes:&dimensions
                        length:sizeof(simd::uint3)
                       atIndex:3];

      MTLSize threadgroupsPerGrid =
          MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
      MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);

      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];

      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}

void gemm_gqa(const GQAParams &gqa_params, const float *q, const float *k,
              const float *v, float *out_gqa) {

  size_t bytesQ = gqa_params.batch * gqa_params.seq_len * gqa_params.n_q_heads *
                  gqa_params.head_dim * sizeof(float);
  size_t bytesK = gqa_params.batch * gqa_params.seq_len *
                  gqa_params.n_kv_heads * gqa_params.head_dim * sizeof(float);
  size_t bytesV = gqa_params.batch * gqa_params.seq_len *
                  gqa_params.n_kv_heads * gqa_params.head_dim * sizeof(float);
  size_t bytesOut = gqa_params.batch * gqa_params.seq_len *
                    gqa_params.n_q_heads * gqa_params.head_dim * sizeof(float);

  id<MTLBuffer> bufferQ = get_or_create_buffer(q, bytesQ);
  id<MTLBuffer> bufferK = get_or_create_buffer(k, bytesK);
  id<MTLBuffer> bufferV = get_or_create_buffer(v, bytesV);
  id<MTLBuffer> bufferOut = get_or_create_buffer(out_gqa, bytesOut, true);

  if (!bufferQ || !bufferK || !bufferV || !bufferOut) {
    std::cerr << "Failed to allocate Metal buffers for gemm_gqa!" << std::endl;
    return;
  }

  MTLSize threadsPerGrid =
      MTLSizeMake(gqa_params.batch, gqa_params.seq_len, gqa_params.n_q_heads);
  MTLSize threadsPerThreadgroup = MTLSizeMake(1, 32, 1);

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateGQA];
    [activeEncoder setBuffer:bufferQ offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferK offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferV offset:0 atIndex:2];
    [activeEncoder setBuffer:bufferOut offset:0 atIndex:3];
    [activeEncoder setBytes:&gqa_params length:sizeof(GQAParams) atIndex:4];

    [activeEncoder dispatchThreads:threadsPerGrid
             threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState:pipelineStateGQA];
      [computeEncoder setBuffer:bufferQ offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferK offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferV offset:0 atIndex:2];
      [computeEncoder setBuffer:bufferOut offset:0 atIndex:3];
      [computeEncoder setBytes:&gqa_params length:sizeof(GQAParams) atIndex:4];

      [computeEncoder dispatchThreads:threadsPerGrid
                threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];

      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}
static id<MTLBuffer> get_or_create_buffer(const void *ptr, size_t bytes,
                                          bool is_write,
                                          bool is_persistent,
                                          bool is_host_input) {
  if (ptr == nullptr || bytes == 0)
    return nil;

  // 1. Persistent weight / optimizer-state / gradient-accumulator buffers
  if (is_persistent) {
    auto it = weightCache.find(ptr);
    if (it != weightCache.end()) {
      if (is_write) {
        memset([it->second contents], 0, bytes);
        copyBackQueue.push_back({const_cast<void *>(ptr), it->second, bytes});
      }
      return it->second;
    }
    id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buf && !is_write) {
      memcpy([buf contents], ptr, bytes);
    }
    if (buf && is_write) {
      memset([buf contents], 0, bytes);
      copyBackQueue.push_back({const_cast<void *>(ptr), buf, bytes});
    }
    if (buf) {
      CFRetain((__bridge CFTypeRef)buf);
      weightCache[ptr] = buf;
    }
    return buf;
  }

  // 2. Check activation caches — weightCache first, then stepCache
  auto weight_it = weightCache.find(ptr);
  if (weight_it != weightCache.end()) {
    stepCache[ptr] = weight_it->second;
    // Even for read, the weight was uploaded at creation time
    return weight_it->second;
  }

  auto step_it = stepCache.find(ptr);
  if (step_it != stepCache.end()) {
    if (is_write) {
      // Reuse the cached buffer but zero it for fresh accumulation
      memset([step_it->second contents], 0, bytes);
      // Register a copy-back so the CPU tensor is updated after GPU execution
      copyBackQueue.push_back({const_cast<void *>(ptr), step_it->second, bytes});
    } else if (is_host_input) {
      // Host-input flag forces an upload even if cached (e.g., token IDs change each step)
      memcpy([step_it->second contents], ptr, bytes);
    }
    return step_it->second;
  }

  // 3. Allocate a buffer from the size pool (or create new)
  id<MTLBuffer> buf = nil;
  auto &bucket = sizePool[bytes];
  if (!bucket.empty()) {
    buf = bucket.back();
    bucket.pop_back();
  } else {
    buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buf) {
      CFRetain((__bridge CFTypeRef)buf);
    }
  }

  if (buf) {
    activeAllocationsThisStep.push_back({bytes, buf});

    if (is_write) {
      // GPU will fully overwrite this buffer; zero it first
      memset([buf contents], 0, bytes);
      // Register copy-back so the CPU tensor is updated after GPU executes
      copyBackQueue.push_back({const_cast<void *>(ptr), buf, bytes});
    } else if (is_host_input) {
      // Upload CPU data to the new GPU buffer
      memcpy([buf contents], ptr, bytes);
    } else {
      // First GPU read of this tensor: upload current CPU contents
      // This ensures the GPU sees the correct data even if the tensor
      // was produced by CPU operations (reshape, transpose, add, etc.)
      memcpy([buf contents], ptr, bytes);
    }
    stepCache[ptr] = buf;
  }

  return buf;
}

void rms_norm_forward(const float *input, float *output, const float *weight,
                      float eps, size_t num_rows, size_t dims) {
  if (num_rows == 0 || dims == 0)
    return;

  size_t bytesIn = num_rows * dims * sizeof(float);
  size_t bytesOut = num_rows * dims * sizeof(float);
  size_t bytesW = dims * sizeof(float);

  id<MTLBuffer> bufferIn = get_or_create_buffer(input, bytesIn);
  id<MTLBuffer> bufferOut = get_or_create_buffer(output, bytesOut, true);
  id<MTLBuffer> bufferW = get_or_create_buffer(weight, bytesW, false, true);

  if (!bufferIn || !bufferOut || !bufferW) {
    std::cerr << "Failed to allocate Metal buffers for rms_norm_forward!"
              << std::endl;
    return;
  }

  uint dims_val = (uint)dims;

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateRMSForward];
    [activeEncoder setBuffer:bufferIn offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferOut offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferW offset:0 atIndex:2];
    [activeEncoder setBytes:&eps length:sizeof(float) atIndex:3];
    [activeEncoder setBytes:&dims_val length:sizeof(uint) atIndex:4];

    MTLSize threadgroupsPerGrid = MTLSizeMake(num_rows, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);

    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState:pipelineStateRMSForward];
      [computeEncoder setBuffer:bufferIn offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferOut offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferW offset:0 atIndex:2];
      [computeEncoder setBytes:&eps length:sizeof(float) atIndex:3];
      [computeEncoder setBytes:&dims_val length:sizeof(uint) atIndex:4];

      MTLSize threadgroupsPerGrid = MTLSizeMake(num_rows, 1, 1);
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);

      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];

      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}

void rms_norm_backward(const float *grad_output, const float *input,
                       const float *weight, float *grad_input,
                       float *grad_weight, float eps, size_t num_rows,
                       size_t dims) {
  if (num_rows == 0 || dims == 0)
    return;

  size_t bytesIn = num_rows * dims * sizeof(float);
  size_t bytesW = dims * sizeof(float);

  id<MTLBuffer> bufferGradOutput = get_or_create_buffer(grad_output, bytesIn);
  id<MTLBuffer> bufferInput = get_or_create_buffer(input, bytesIn);
  id<MTLBuffer> bufferWeight = get_or_create_buffer(weight, bytesW, false, true);
  id<MTLBuffer> bufferGradInput =
      get_or_create_buffer(grad_input, bytesIn, true);
  id<MTLBuffer> bufferGradWeight =
      get_or_create_buffer(grad_weight, bytesW, true, true);

  if (!bufferGradOutput || !bufferInput || !bufferWeight || !bufferGradInput ||
      !bufferGradWeight) {
    std::cerr << "Failed to allocate Metal buffers for rms_norm_backward!"
              << std::endl;
    return;
  }

  uint dims_val = (uint)dims;

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateRMSBackwardDX];
    [activeEncoder setBuffer:bufferGradOutput offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferInput offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferWeight offset:0 atIndex:2];
    [activeEncoder setBuffer:bufferGradInput offset:0 atIndex:3];
    [activeEncoder setBuffer:bufferGradWeight offset:0 atIndex:4];
    [activeEncoder setBytes:&eps length:sizeof(float) atIndex:5];
    [activeEncoder setBytes:&dims_val length:sizeof(uint) atIndex:6];

    MTLSize threadgroupsPerGrid1 = MTLSizeMake(num_rows, 1, 1);
    MTLSize threadsPerThreadgroup1 = MTLSizeMake(256, 1, 1);

    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid1
                  threadsPerThreadgroup:threadsPerThreadgroup1];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState:pipelineStateRMSBackwardDX];
      [computeEncoder setBuffer:bufferGradOutput offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferInput offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferWeight offset:0 atIndex:2];
      [computeEncoder setBuffer:bufferGradInput offset:0 atIndex:3];
      [computeEncoder setBuffer:bufferGradWeight offset:0 atIndex:4];
      [computeEncoder setBytes:&eps length:sizeof(float) atIndex:5];
      [computeEncoder setBytes:&dims_val length:sizeof(uint) atIndex:6];

      MTLSize threadgroupsPerGrid1 = MTLSizeMake(num_rows, 1, 1);
      MTLSize threadsPerThreadgroup1 = MTLSizeMake(256, 1, 1);

      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid1
                     threadsPerThreadgroup:threadsPerThreadgroup1];

      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}

void swiglu_backward(const float *grad_output, const float *gate,
                     const float *up, float *grad_gate, float *grad_up,
                     size_t n) {
  if (n == 0)
    return;
  size_t bytesIn = n * sizeof(float);
  size_t bytesOut = n * sizeof(float);
  id<MTLBuffer> bufferGradOutput = get_or_create_buffer(grad_output, bytesIn);
  id<MTLBuffer> bufferGate = get_or_create_buffer(gate, bytesIn);
  id<MTLBuffer> bufferUp = get_or_create_buffer(up, bytesIn);
  id<MTLBuffer> bufferGradGate =
      get_or_create_buffer(grad_gate, bytesOut, true);
  id<MTLBuffer> bufferGradUp = get_or_create_buffer(grad_up, bytesOut, true);
  if (!bufferGradOutput || !bufferGate || !bufferUp || !bufferGradGate ||
      !bufferGradUp) {
    std::cerr << "Failed to allocate Metal buffers for swiglu_backward!"
              << std::endl;
    return;
  }
  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateSwiGLUBackward];
    [activeEncoder setBuffer:bufferGradOutput offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferGate offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferUp offset:0 atIndex:2];
    [activeEncoder setBuffer:bufferGradGate offset:0 atIndex:3];
    [activeEncoder setBuffer:bufferGradUp offset:0 atIndex:4];

    uint n_val = (uint)n;
    [activeEncoder setBytes:&n_val length:sizeof(uint) atIndex:5];

    MTLSize threadgroupsPerGrid = MTLSizeMake((n + 255) / 256, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateSwiGLUBackward];
      [computeEncoder setBuffer:bufferGradOutput offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferGate offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferUp offset:0 atIndex:2];
      [computeEncoder setBuffer:bufferGradGate offset:0 atIndex:3];
      [computeEncoder setBuffer:bufferGradUp offset:0 atIndex:4];

      uint n_val = (uint)n;
      [computeEncoder setBytes:&n_val length:sizeof(uint) atIndex:5];

      MTLSize threadgroupsPerGrid = MTLSizeMake((n + 255) / 256, 1, 1);
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}

void rope_backward(float *grad, const float *cos_table, const float *sin_table,
                   size_t batch, size_t heads, size_t seq_len,
                   size_t head_dim) {
  if (batch == 0 || heads == 0 || seq_len == 0 || head_dim == 0)
    return;

  size_t half_dim = head_dim / 2;
  size_t total_pairs = batch * heads * seq_len * half_dim;
  size_t bytesGrad = batch * heads * seq_len * head_dim * sizeof(float);
  size_t bytesTable = seq_len * half_dim * sizeof(float);

  id<MTLBuffer> bufferGrad = get_or_create_buffer(grad, bytesGrad, true);
  id<MTLBuffer> bufferCos = get_or_create_buffer(cos_table, bytesTable, false, true);
  id<MTLBuffer> bufferSin = get_or_create_buffer(sin_table, bytesTable, false, true);

  if (!bufferGrad || !bufferCos || !bufferSin) {
    std::cerr << "Failed to allocate Metal buffers for rope_backward!"
              << std::endl;
    return;
  }

  uint b_val = (uint)batch;
  uint h_val = (uint)heads;
  uint s_val = (uint)seq_len;
  uint d_val = (uint)head_dim;

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateRoPEBackward];
    [activeEncoder setBuffer:bufferGrad offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferCos offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferSin offset:0 atIndex:2];
    [activeEncoder setBytes:&b_val length:sizeof(uint) atIndex:3];
    [activeEncoder setBytes:&h_val length:sizeof(uint) atIndex:4];
    [activeEncoder setBytes:&s_val length:sizeof(uint) atIndex:5];
    [activeEncoder setBytes:&d_val length:sizeof(uint) atIndex:6];

    MTLSize threadgroupsPerGrid = MTLSizeMake((total_pairs + 255) / 256, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateRoPEBackward];
      [computeEncoder setBuffer:bufferGrad offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferCos offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferSin offset:0 atIndex:2];
      [computeEncoder setBytes:&b_val length:sizeof(uint) atIndex:3];
      [computeEncoder setBytes:&h_val length:sizeof(uint) atIndex:4];
      [computeEncoder setBytes:&s_val length:sizeof(uint) atIndex:5];
      [computeEncoder setBytes:&d_val length:sizeof(uint) atIndex:6];

      MTLSize threadgroupsPerGrid = MTLSizeMake((total_pairs + 255) / 256, 1, 1);
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}

void gqa_backward(const GQABackwardParams &params,
                  const float *Q, const float *K, const float *V,
                  const float *grad_attn_output,
                  float *grad_Q, float *grad_K, float *grad_V) {
  size_t bytesQ = params.batch * params.n_q_heads * params.seq_len *
                  params.head_dim * sizeof(float);
  size_t bytesKV = params.batch * params.n_kv_heads * params.seq_len *
                   params.head_dim * sizeof(float);
  size_t bytesGradOut = params.batch * params.seq_len * params.n_q_heads *
                        params.head_dim * sizeof(float);

  id<MTLBuffer> bufferQ = get_or_create_buffer(Q, bytesQ);
  id<MTLBuffer> bufferK = get_or_create_buffer(K, bytesKV);
  id<MTLBuffer> bufferV = get_or_create_buffer(V, bytesKV);
  id<MTLBuffer> bufferGradOut = get_or_create_buffer(grad_attn_output, bytesGradOut);
  id<MTLBuffer> bufferGradQ = get_or_create_buffer(grad_Q, bytesQ, true);
  id<MTLBuffer> bufferGradK = get_or_create_buffer(grad_K, bytesKV, true);
  id<MTLBuffer> bufferGradV = get_or_create_buffer(grad_V, bytesKV, true);

  if (!bufferQ || !bufferK || !bufferV || !bufferGradOut ||
      !bufferGradQ || !bufferGradK || !bufferGradV) {
    std::cerr << "Failed to allocate Metal buffers for gqa_backward!"
              << std::endl;
    return;
  }

  // WHAT: Metal shader expects the same struct layout as GQABackwardParams.
  // We pass the C++ struct directly via setBytes since the layouts match.
  struct {
    uint32_t batch;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t seq_len;
    uint32_t head_dim;
  } gpu_params = {params.batch, params.n_q_heads, params.n_kv_heads,
                  params.seq_len, params.head_dim};

  MTLSize threadsPerGrid =
      MTLSizeMake(params.batch, params.n_q_heads, params.seq_len);
  MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateGQABackward];
    [activeEncoder setBuffer:bufferQ offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferK offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferV offset:0 atIndex:2];
    [activeEncoder setBuffer:bufferGradOut offset:0 atIndex:3];
    [activeEncoder setBuffer:bufferGradQ offset:0 atIndex:4];
    [activeEncoder setBuffer:bufferGradK offset:0 atIndex:5];
    [activeEncoder setBuffer:bufferGradV offset:0 atIndex:6];
    [activeEncoder setBytes:&gpu_params length:sizeof(gpu_params) atIndex:7];

    [activeEncoder dispatchThreads:threadsPerGrid
             threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState:pipelineStateGQABackward];
      [computeEncoder setBuffer:bufferQ offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferK offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferV offset:0 atIndex:2];
      [computeEncoder setBuffer:bufferGradOut offset:0 atIndex:3];
      [computeEncoder setBuffer:bufferGradQ offset:0 atIndex:4];
      [computeEncoder setBuffer:bufferGradK offset:0 atIndex:5];
      [computeEncoder setBuffer:bufferGradV offset:0 atIndex:6];
      [computeEncoder setBytes:&gpu_params length:sizeof(gpu_params) atIndex:7];

      [computeEncoder dispatchThreads:threadsPerGrid
               threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];

      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}

void adamw_step(float *param, const float *grad, float *m, float *v,
                const AdamWStepParams &params) {
  if (params.n == 0)
    return;

  size_t bytes = params.n * sizeof(float);

  id<MTLBuffer> bufferParam = get_or_create_buffer(param, bytes, true, true);
  id<MTLBuffer> bufferGrad = get_or_create_buffer(grad, bytes, false, false);
  id<MTLBuffer> bufferM = get_or_create_buffer(m, bytes, true, true);
  id<MTLBuffer> bufferV = get_or_create_buffer(v, bytes, true, true);

  if (!bufferParam || !bufferGrad || !bufferM || !bufferV) {
    std::cerr << "Failed to allocate Metal buffers for adamw_step!"
              << std::endl;
    return;
  }

  // WHAT: Metal shader expects AdamWParams struct with float fields + uint n.
  struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2;
    uint32_t n;
  } gpu_params = {params.lr, params.beta1, params.beta2, params.eps,
                  params.weight_decay, params.bias_correction1,
                  params.bias_correction2, params.n};

  MTLSize threadgroupsPerGrid = MTLSizeMake((params.n + 255) / 256, 1, 1);
  MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateAdamW];
    [activeEncoder setBuffer:bufferParam offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferGrad offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferM offset:0 atIndex:2];
    [activeEncoder setBuffer:bufferV offset:0 atIndex:3];
    [activeEncoder setBytes:&gpu_params length:sizeof(gpu_params) atIndex:4];

    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder =
          [cmdBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState:pipelineStateAdamW];
      [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
      [computeEncoder setBuffer:bufferGrad offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferM offset:0 atIndex:2];
      [computeEncoder setBuffer:bufferV offset:0 atIndex:3];
      [computeEncoder setBytes:&gpu_params length:sizeof(gpu_params) atIndex:4];

      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];

      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
      bufferCache.clear();
    }
  }
}

void residual_add(float *a, const float *b, size_t n) {
  // WHAT: Guard: nothing to add for empty tensors.
  if (n == 0 || a == nullptr || b == nullptr) return;

  size_t bytes = n * sizeof(float);

  id<MTLBuffer> bufA = get_or_create_buffer(a, bytes, true);
  id<MTLBuffer> bufB = get_or_create_buffer(b, bytes, false);
  uint32_t un = static_cast<uint32_t>(n);

  if (!bufA || !bufB) {
    std::cerr << "[residual_add] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateResidualAdd];
    [activeEncoder setBuffer:bufA offset:0 atIndex:0];
    [activeEncoder setBuffer:bufB offset:0 atIndex:1];
    [activeEncoder setBytes:&un length:sizeof(uint32_t) atIndex:2];

    MTLSize threads = MTLSizeMake(256, 1, 1);
    MTLSize grid    = MTLSizeMake((n + 255) / 256, 1, 1);
    [activeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateResidualAdd];
      [computeEncoder setBuffer:bufA offset:0 atIndex:0];
      [computeEncoder setBuffer:bufB offset:0 atIndex:1];
      [computeEncoder setBytes:&un length:sizeof(uint32_t) atIndex:2];
      MTLSize threads = MTLSizeMake(256, 1, 1);
      MTLSize grid    = MTLSizeMake((n + 255) / 256, 1, 1);
      [computeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void embedding_forward(const uint32_t *token_ids, const float *embedding_table,
                       float *output, size_t total_tokens, size_t hidden_dim,
                       size_t vocab_size) {
  // WHAT: Guard: nothing to embed for empty input.
  if (total_tokens == 0 || hidden_dim == 0) return;

  size_t bytes_ids   = total_tokens * sizeof(uint32_t);
  size_t bytes_emb   = vocab_size * hidden_dim * sizeof(float);
  size_t bytes_out   = total_tokens * hidden_dim * sizeof(float);

  // WHAT: Upload token IDs (read-only input, host_input=true) and embedding table (read-only weight, persistent=true).
  // WHY:  token_ids change every step. embedding_table is persistent (is_persistent=true).
  id<MTLBuffer> bufIds  = get_or_create_buffer(token_ids,       bytes_ids, false, false, true);
  id<MTLBuffer> bufEmb  = get_or_create_buffer(embedding_table, bytes_emb, false, true,  false);
  // WHAT: Output buffer — GPU writes h from scratch. No upload needed (is_write=true).
  id<MTLBuffer> bufOut  = get_or_create_buffer(output,          bytes_out, true,  false, false);

  if (!bufIds || !bufEmb || !bufOut) {
    std::cerr << "[embedding_forward] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  uint32_t u_hidden  = static_cast<uint32_t>(hidden_dim);
  uint32_t u_tokens  = static_cast<uint32_t>(total_tokens);
  size_t   n_threads = total_tokens * hidden_dim;  // one thread per float element

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateEmbeddingFwd];
    [activeEncoder setBuffer:bufIds  offset:0 atIndex:0];
    [activeEncoder setBuffer:bufEmb  offset:0 atIndex:1];
    [activeEncoder setBuffer:bufOut  offset:0 atIndex:2];
    [activeEncoder setBytes:&u_hidden length:sizeof(uint32_t) atIndex:3];
    [activeEncoder setBytes:&u_tokens length:sizeof(uint32_t) atIndex:4];

    MTLSize threads = MTLSizeMake(256, 1, 1);
    MTLSize grid    = MTLSizeMake((n_threads + 255) / 256, 1, 1);
    [activeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateEmbeddingFwd];
      [computeEncoder setBuffer:bufIds  offset:0 atIndex:0];
      [computeEncoder setBuffer:bufEmb  offset:0 atIndex:1];
      [computeEncoder setBuffer:bufOut  offset:0 atIndex:2];
      [computeEncoder setBytes:&u_hidden length:sizeof(uint32_t) atIndex:3];
      [computeEncoder setBytes:&u_tokens length:sizeof(uint32_t) atIndex:4];
      MTLSize threads = MTLSizeMake(256, 1, 1);
      MTLSize grid    = MTLSizeMake((n_threads + 255) / 256, 1, 1);
      [computeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void cross_entropy(const float *logits, const uint32_t *targets, float *loss_out,
                   float *grad_logits, size_t total_tokens, size_t vocab_size) {
  if (total_tokens == 0 || vocab_size == 0) return;

  size_t bytes_logits = total_tokens * vocab_size * sizeof(float);
  size_t bytes_targets = total_tokens * sizeof(uint32_t);
  size_t bytes_loss = sizeof(float);

  // Use the static last_step_loss as the persistent write target (stable address,
  // no stack-dangling risk). Sync back to the caller's loss_out after execution.
  last_step_loss = 0.0f;
  id<MTLBuffer> bufLogits  = get_or_create_buffer(logits,          bytes_logits,  false, false, false);
  id<MTLBuffer> bufTargets = get_or_create_buffer(targets,         bytes_targets, false, false, true);
  id<MTLBuffer> bufLoss    = get_or_create_buffer(&last_step_loss, bytes_loss,    true,  true,  false);
  id<MTLBuffer> bufGrad    = get_or_create_buffer(grad_logits,     bytes_logits,  true,  false, false);

  if (!bufLogits || !bufTargets || !bufLoss || !bufGrad) {
    std::cerr << "[cross_entropy] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  uint32_t u_vocab  = static_cast<uint32_t>(vocab_size);
  uint32_t u_tokens = static_cast<uint32_t>(total_tokens);

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateCrossEntropy];
    [activeEncoder setBuffer:bufLogits  offset:0 atIndex:0];
    [activeEncoder setBuffer:bufTargets offset:0 atIndex:1];
    [activeEncoder setBuffer:bufLoss    offset:0 atIndex:2];
    [activeEncoder setBuffer:bufGrad    offset:0 atIndex:3];
    [activeEncoder setBytes:&u_vocab  length:sizeof(uint32_t) atIndex:4];
    [activeEncoder setBytes:&u_tokens length:sizeof(uint32_t) atIndex:5];

    MTLSize threadsPerTG = MTLSizeMake(256, 1, 1);
    MTLSize tgGrid       = MTLSizeMake(1, total_tokens, 1);
    [activeEncoder dispatchThreadgroups:tgGrid threadsPerThreadgroup:threadsPerTG];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState:pipelineStateCrossEntropy];
      [computeEncoder setBuffer:bufLogits  offset:0 atIndex:0];
      [computeEncoder setBuffer:bufTargets offset:0 atIndex:1];
      [computeEncoder setBuffer:bufLoss    offset:0 atIndex:2];
      [computeEncoder setBuffer:bufGrad    offset:0 atIndex:3];
      [computeEncoder setBytes:&u_vocab  length:sizeof(uint32_t) atIndex:4];
      [computeEncoder setBytes:&u_tokens length:sizeof(uint32_t) atIndex:5];

      MTLSize threadsPerTG = MTLSizeMake(256, 1, 1);
      MTLSize tgGrid       = MTLSizeMake(1, total_tokens, 1);
      [computeEncoder dispatchThreadgroups:tgGrid threadsPerThreadgroup:threadsPerTG];

      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
    }
  }

  // Sync the CPU-side loss value to the caller's pointer
  *loss_out = last_step_loss;
}

void gemm_backward(const float *a_transposed, const float *b, float *c,
                   size_t M, size_t N, size_t K) {
  if (M == 0 || N == 0 || K == 0) return;

  size_t bytesA = K * M * sizeof(float); // A in memory is [K x M]
  size_t bytesB = K * N * sizeof(float); // B in memory is [K x N]
  size_t bytesC = M * N * sizeof(float); // C in memory is [M x N]

  id<MTLBuffer> bufA = get_or_create_buffer(a_transposed, bytesA, false, false);
  id<MTLBuffer> bufB = get_or_create_buffer(b,            bytesB, false, false);
  id<MTLBuffer> bufC = get_or_create_buffer(c,            bytesC, true,  true);

  if (!bufA || !bufB || !bufC) {
    std::cerr << "[gemm_backward] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  simd::uint3 dimensions = {(uint32_t)M, (uint32_t)N, (uint32_t)K};

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateGEMMBackward];
    [activeEncoder setBuffer:bufA offset:0 atIndex:0];
    [activeEncoder setBuffer:bufB offset:0 atIndex:1];
    [activeEncoder setBuffer:bufC offset:0 atIndex:2];
    [activeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];

    MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateGEMMBackward];
      [computeEncoder setBuffer:bufA offset:0 atIndex:0];
      [computeEncoder setBuffer:bufB offset:0 atIndex:1];
      [computeEncoder setBuffer:bufC offset:0 atIndex:2];
      [computeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];
      MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);
      MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                    threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void gemm_proj_trans_b(const float *a, const float *b_transposed, float *c,
                       size_t M, size_t N, size_t K) {
  if (M == 0 || N == 0 || K == 0) return;

  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = N * K * sizeof(float); // B in memory is [N x K]
  size_t bytesC = M * N * sizeof(float);

  id<MTLBuffer> bufA = get_or_create_buffer(a,            bytesA, false, false);
  id<MTLBuffer> bufB = get_or_create_buffer(b_transposed, bytesB, false, true);
  id<MTLBuffer> bufC = get_or_create_buffer(c,            bytesC, true,  false);

  if (!bufA || !bufB || !bufC) {
    std::cerr << "[gemm_proj_trans_b] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  simd::uint3 dimensions = {(uint32_t)M, (uint32_t)N, (uint32_t)K};

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateGEMMProjTransB];
    [activeEncoder setBuffer:bufA offset:0 atIndex:0];
    [activeEncoder setBuffer:bufB offset:0 atIndex:1];
    [activeEncoder setBuffer:bufC offset:0 atIndex:2];
    [activeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];

    MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
    [activeEncoder dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateGEMMProjTransB];
      [computeEncoder setBuffer:bufA offset:0 atIndex:0];
      [computeEncoder setBuffer:bufB offset:0 atIndex:1];
      [computeEncoder setBuffer:bufC offset:0 atIndex:2];
      [computeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];
      MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);
      MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                    threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void reshape_to_4d(const float *src, float *dst,
                   size_t batch, size_t n_heads, size_t seq_len, size_t head_dim) {
  size_t total = batch * n_heads * seq_len * head_dim;
  if (total == 0) return;

  size_t bytesSrc = (batch * seq_len * n_heads * head_dim) * sizeof(float);
  size_t bytesDst = total * sizeof(float);

  id<MTLBuffer> bufSrc = get_or_create_buffer(src, bytesSrc, false, false, false);
  id<MTLBuffer> bufDst = get_or_create_buffer(dst, bytesDst, true, false, false);

  if (!bufSrc || !bufDst) {
    std::cerr << "[reshape_to_4d] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  uint u_batch   = (uint)batch;
  uint u_n_heads = (uint)n_heads;
  uint u_seq_len = (uint)seq_len;
  uint u_hd      = (uint)head_dim;

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateReshape4D];
    [activeEncoder setBuffer:bufSrc offset:0 atIndex:0];
    [activeEncoder setBuffer:bufDst offset:0 atIndex:1];
    [activeEncoder setBytes:&u_batch   length:sizeof(uint) atIndex:2];
    [activeEncoder setBytes:&u_n_heads length:sizeof(uint) atIndex:3];
    [activeEncoder setBytes:&u_seq_len length:sizeof(uint) atIndex:4];
    [activeEncoder setBytes:&u_hd      length:sizeof(uint) atIndex:5];
    MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);
    [activeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateReshape4D];
      [computeEncoder setBuffer:bufSrc offset:0 atIndex:0];
      [computeEncoder setBuffer:bufDst offset:0 atIndex:1];
      [computeEncoder setBytes:&u_batch   length:sizeof(uint) atIndex:2];
      [computeEncoder setBytes:&u_n_heads length:sizeof(uint) atIndex:3];
      [computeEncoder setBytes:&u_seq_len length:sizeof(uint) atIndex:4];
      [computeEncoder setBytes:&u_hd      length:sizeof(uint) atIndex:5];
      MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
      MTLSize tg   = MTLSizeMake(256, 1, 1);
      [computeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void reshape_to_3d(const float *src, float *dst,
                   size_t batch, size_t n_heads, size_t seq_len, size_t head_dim) {
  size_t total = batch * n_heads * seq_len * head_dim;
  if (total == 0) return;

  size_t bytesSrc = total * sizeof(float);
  size_t bytesDst = (batch * seq_len * n_heads * head_dim) * sizeof(float);

  id<MTLBuffer> bufSrc = get_or_create_buffer(src, bytesSrc, false, false, false);
  id<MTLBuffer> bufDst = get_or_create_buffer(dst, bytesDst, true, false, false);

  if (!bufSrc || !bufDst) {
    std::cerr << "[reshape_to_3d] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  uint u_batch   = (uint)batch;
  uint u_n_heads = (uint)n_heads;
  uint u_seq_len = (uint)seq_len;
  uint u_hd      = (uint)head_dim;

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateReshape3D];
    [activeEncoder setBuffer:bufSrc offset:0 atIndex:0];
    [activeEncoder setBuffer:bufDst offset:0 atIndex:1];
    [activeEncoder setBytes:&u_batch   length:sizeof(uint) atIndex:2];
    [activeEncoder setBytes:&u_n_heads length:sizeof(uint) atIndex:3];
    [activeEncoder setBytes:&u_seq_len length:sizeof(uint) atIndex:4];
    [activeEncoder setBytes:&u_hd      length:sizeof(uint) atIndex:5];
    MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);
    [activeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateReshape3D];
      [computeEncoder setBuffer:bufSrc offset:0 atIndex:0];
      [computeEncoder setBuffer:bufDst offset:0 atIndex:1];
      [computeEncoder setBytes:&u_batch   length:sizeof(uint) atIndex:2];
      [computeEncoder setBytes:&u_n_heads length:sizeof(uint) atIndex:3];
      [computeEncoder setBytes:&u_seq_len length:sizeof(uint) atIndex:4];
      [computeEncoder setBytes:&u_hd      length:sizeof(uint) atIndex:5];
      MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
      MTLSize tg   = MTLSizeMake(256, 1, 1);
      [computeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void rope_forward(float *q, float *k,
                  const float *cos_table, const float *sin_table,
                  size_t batch, size_t q_heads, size_t kv_heads,
                  size_t seq_len, size_t head_dim) {
  size_t half_dim = head_dim / 2;

  size_t bytesQ  = batch * q_heads * seq_len * head_dim * sizeof(float);
  size_t bytesK  = batch * kv_heads * seq_len * head_dim * sizeof(float);
  size_t bytesT  = seq_len * half_dim * sizeof(float);

  id<MTLBuffer> bufQ   = get_or_create_buffer(q,          bytesQ, true,  false, false);
  id<MTLBuffer> bufK   = get_or_create_buffer(k,          bytesK, true,  false, false);
  id<MTLBuffer> bufCos = get_or_create_buffer(cos_table,  bytesT, false, true,  false);
  id<MTLBuffer> bufSin = get_or_create_buffer(sin_table,  bytesT, false, true,  false);

  if (!bufQ || !bufK || !bufCos || !bufSin) {
    std::cerr << "[rope_forward] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  uint u_batch   = (uint)batch;
  uint u_qheads  = (uint)q_heads;
  uint u_kvheads = (uint)kv_heads;
  uint u_seqlen  = (uint)seq_len;
  uint u_hd      = (uint)head_dim;

  size_t total_pairs_q  = batch * q_heads * seq_len * half_dim;
  size_t total_pairs_k  = batch * kv_heads * seq_len * half_dim;
  size_t total_pairs = total_pairs_q + total_pairs_k;

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateRoPEForward];
    [activeEncoder setBuffer:bufQ   offset:0 atIndex:0];
    [activeEncoder setBuffer:bufK   offset:0 atIndex:1];
    [activeEncoder setBuffer:bufCos offset:0 atIndex:2];
    [activeEncoder setBuffer:bufSin offset:0 atIndex:3];
    [activeEncoder setBytes:&u_batch   length:sizeof(uint) atIndex:4];
    [activeEncoder setBytes:&u_qheads  length:sizeof(uint) atIndex:5];
    [activeEncoder setBytes:&u_kvheads length:sizeof(uint) atIndex:6];
    [activeEncoder setBytes:&u_seqlen  length:sizeof(uint) atIndex:7];
    [activeEncoder setBytes:&u_hd      length:sizeof(uint) atIndex:8];
    MTLSize grid = MTLSizeMake((total_pairs + 255) / 256, 1, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);
    [activeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState:pipelineStateRoPEForward];
      [computeEncoder setBuffer:bufQ   offset:0 atIndex:0];
      [computeEncoder setBuffer:bufK   offset:0 atIndex:1];
      [computeEncoder setBuffer:bufCos offset:0 atIndex:2];
      [computeEncoder setBuffer:bufSin offset:0 atIndex:3];
      [computeEncoder setBytes:&u_batch   length:sizeof(uint) atIndex:4];
      [computeEncoder setBytes:&u_qheads  length:sizeof(uint) atIndex:5];
      [computeEncoder setBytes:&u_kvheads length:sizeof(uint) atIndex:6];
      [computeEncoder setBytes:&u_seqlen  length:sizeof(uint) atIndex:7];
      [computeEncoder setBytes:&u_hd      length:sizeof(uint) atIndex:8];
      MTLSize grid = MTLSizeMake((total_pairs + 255) / 256, 1, 1);
      MTLSize tg   = MTLSizeMake(256, 1, 1);
      [computeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [computeEncoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void clear_step_cache() {
  // Return all buffers allocated this step to the size pool for reuse
  for (const auto &pair : activeAllocationsThisStep) {
    sizePool[pair.first].push_back(pair.second);
  }
  activeAllocationsThisStep.clear();
  stepCache.clear();
}

void reconcile_buffers() {
  run_copy_back_tasks();
  clear_step_cache();
}

void start_batch() {
  if (batchActive)
    return;
  stepCache.clear();
  activeAllocationsThisStep.clear();
  activeCmdBuffer = [commandQueue commandBuffer];
  activeEncoder = [activeCmdBuffer computeCommandEncoder];
  batchActive = true;
}

void commit_batch() {
  if (!batchActive)
    return;
  [activeEncoder endEncoding];
  activeEncoder = nil;

  std::cout << "[GPU-EXEC] Dispatched batch to Apple Silicon hardware... GPU executing..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  [activeCmdBuffer commit];
  [activeCmdBuffer waitUntilCompleted];

  auto end = std::chrono::high_resolution_clock::now();
  double gpu_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
  accum_gpu_time_ms += gpu_time;
  count_gpu_calls++;

  std::cout << "[GPU-EXEC] GPU Hardware finished execution in " << gpu_time << " ms." << std::endl;

  activeCmdBuffer = nil;

  // Run copy-back tasks for unaligned writes
  run_copy_back_tasks();

  // Return all intermediate activation buffers from this step back to sizePool buckets
  for (const auto &pair : activeAllocationsThisStep) {
    sizePool[pair.first].push_back(pair.second);
  }
  activeAllocationsThisStep.clear();

  batchActive = false;
  stepCache.clear();
}

float get_last_loss() {
  return last_step_loss;
}

void execute_in_autoreleasepool(std::function<void()> func) {
  @autoreleasepool {
    func();
  }
}

} // namespace metal_bridge
