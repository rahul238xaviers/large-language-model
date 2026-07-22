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
#include "gemm_profiler.hpp"
#define ACCELERATE_NEW_LAPACK
#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstddef>
#include <iostream>
#import <simd/simd.h>
#include <unordered_map>

// Pull in Tensor and PagedBuffer OUTSIDE the metal_bridge namespace so that
// ::PagedBuffer is the same type in all translation units.
#include "Tensor.hpp"

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
static id<MTLComputePipelineState> pipelineStateFlashAttnFwd = nil;
static id<MTLComputePipelineState> pipelineStateGEMMBackward = nil;
static id<MTLComputePipelineState> pipelineStateGEMMProjTransB = nil;
static id<MTLComputePipelineState> pipelineStateReshape4D = nil;
static id<MTLComputePipelineState> pipelineStateReshape3D = nil;
static id<MTLComputePipelineState> pipelineStateRoPEForward = nil;
static id<MTLComputePipelineState> pipelineStateGQAScores = nil;
static id<MTLComputePipelineState> pipelineStateAttnSoftmax = nil;
static id<MTLComputePipelineState> pipelineStateAttnDS = nil;
static id<MTLComputePipelineState> pipelineStateSwiGLUForward = nil;
static id<MTLComputePipelineState> pipelineStateFusedAttnBwd = nil;
static id<MTLComputePipelineState> pipelineStateGEMMBF16 = nil;
static id<MTLComputePipelineState> pipelineStateFusedAddNorm = nil;
static id<MTLComputePipelineState> pipelineStateFusedBackwardAddNorm = nil;

static float last_step_loss = 0.0f;
static bool initialized = false;

// ── Manual autorelease pool for the training step ──
// Every dispatch function in the batchActive path creates Objective-C objects
// (MTLCommandBuffer, MTLComputeCommandEncoder, MTLBuffer from get_or_create_buffer)
// that are autoreleased.  Without a draining pool, they accumulate for the
// entire program lifetime, thrashing the allocator.
// We create the pool in begin_scope() and drain it in end_scope().
static NSAutoreleasePool* stepPool = nil;

// ── Single command buffer for the entire step (Multi-Encoder architecture) ──
// begin_scope creates one command buffer.  Every dispatch function creates a
// new encoder from it, encodes one kernel, and immediately ends the encoder.
// end_scope commits once and waits.  This gives the GPU hardware scheduler
// maximum flexibility (each encoder is an independent scheduling unit) while
// the CPU pays exactly one XNU commit/wait syscall per step.
static id<MTLCommandBuffer> batchCommandBuffer = nil;

// ── Persistent pointer-to-gpu_wrapper map ──
// Maps float* data pointers to their PagedBuffer::gpu_wrapper_ location.
// Populated when a PagedBuffer allocates memory, never cleared.
// Enables O(1) buffer lookup from raw float* without per-step caches.
#include <unordered_map>
static std::unordered_map<const float*, void**> ptr_wrapper_map;

// Register a PagedBuffer's gpu_wrapper location (called from PagedBuffer::grow)
void register_gpu_wrapper(const float* ptr, void** wrapper_loc) {
  ptr_wrapper_map[ptr] = wrapper_loc;
}
void unregister_gpu_wrapper(const float* ptr) {
  ptr_wrapper_map.erase(ptr);
}

// Look up the gpu_wrapper for a raw float pointer (O(1), persistent)
static void** find_gpu_wrapper(const void* ptr) {
  auto it = ptr_wrapper_map.find(static_cast<const float*>(ptr));
  return it != ptr_wrapper_map.end() ? it->second : nullptr;
}

// PagedBuffer::gpu_release_fn is set during initialize() below.

// Minimal persistent cache for non-PagedBuffer pointers (e.g. &last_step_loss for loss)
// These are created once and reused for the program lifetime.
static std::unordered_map<const void*, id<MTLBuffer>> persistent_fallback_cache;

// Size-bucketed pool for intermediate activation buffers (recycled step to step)
static std::unordered_map<size_t, std::vector<id<MTLBuffer>>> sizePool;

// Buffers allocated for non-PagedBuffer pointers (fallback path, recycled at end_scope)
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

// ensure_encoder removed — each dispatch creates its own encoder from batchCommandBuffer.

// Debug guard: if this fires, a kernel is encoding outside begin_scope().
// In a correctly scoped step, all kernel encodes happen between begin_scope()
// and end_scope().  Any else-block execution means a sync pipeline stall.
#ifndef NDEBUG
#define ASSERT_BATCHED(msg) if (!batchActive) { std::cerr << "[BATCH] " << msg << " outside scope — sync fallback!\n"; }
#else
#define ASSERT_BATCHED(msg)
#endif

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
         (pipelineStateProj != nil) && (pipelineStateGQA != nil) &&
         (pipelineStateFusedAttnBwd != nil) && (pipelineStateFlashAttnFwd != nil) &&
         (pipelineStateFusedAddNorm != nil) &&
         (pipelineStateFusedBackwardAddNorm != nil);
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

  id<MTLFunction> flashFwdFunc = [defaultLibrary newFunctionWithName:@"flash_attn_fwd"];
  if (!flashFwdFunc) {
    std::cerr << "Failed to find kernel function 'flash_attn_fwd' in library!" << std::endl;
    return;
  }
  pipelineStateFlashAttnFwd = [device newComputePipelineStateWithFunction:flashFwdFunc error:&error];
  if (!pipelineStateFlashAttnFwd) {
    std::cerr << "Failed to compile pipeline state for 'flash_attn_fwd'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'flash_attn_fwd'!" << std::endl;

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

  id<MTLFunction> gqaScoresFunc = [defaultLibrary newFunctionWithName:@"gqa_scores"];
  if (!gqaScoresFunc) {
    std::cerr << "Failed to find kernel function 'gqa_scores' in library!" << std::endl;
    return;
  }
  pipelineStateGQAScores = [device newComputePipelineStateWithFunction:gqaScoresFunc error:&error];
  if (!pipelineStateGQAScores) {
    std::cerr << "Failed to compile pipeline state for 'gqa_scores'! Error: "
              << [[error localizedDescription] UTF8String] << std::endl;
    return;
  }
  std::cout << "Metal Pipeline State compiled successfully for 'gqa_scores'!" << std::endl;

  id<MTLFunction> attnSoftmaxFunc = [defaultLibrary newFunctionWithName:@"attn_softmax"];
  if (!attnSoftmaxFunc) { std::cerr << "Failed to find 'attn_softmax'\n"; return; }
  pipelineStateAttnSoftmax = [device newComputePipelineStateWithFunction:attnSoftmaxFunc error:&error];
  if (!pipelineStateAttnSoftmax) { std::cerr << "Failed to compile 'attn_softmax'\n"; return; }
  std::cout << "Compiled 'attn_softmax'!" << std::endl;

  id<MTLFunction> attnDSFunc = [defaultLibrary newFunctionWithName:@"attn_ds"];
  if (!attnDSFunc) { std::cerr << "Failed to find 'attn_ds'\n"; return; }
  pipelineStateAttnDS = [device newComputePipelineStateWithFunction:attnDSFunc error:&error];
  if (!pipelineStateAttnDS) { std::cerr << "Failed to compile 'attn_ds'\n"; return; }
  std::cout << "Compiled 'attn_ds'!" << std::endl;

  id<MTLFunction> swigluFwdFunc = [defaultLibrary newFunctionWithName:@"swiglu_forward"];
  if (!swigluFwdFunc) { std::cerr << "Failed to find 'swiglu_forward'\n"; return; }
  pipelineStateSwiGLUForward = [device newComputePipelineStateWithFunction:swigluFwdFunc error:&error];
  if (!pipelineStateSwiGLUForward) { std::cerr << "Failed to compile 'swiglu_forward'\n"; return; }
  std::cout << "Compiled 'swiglu_forward'!" << std::endl;

  id<MTLFunction> fusedBwdFunc = [defaultLibrary newFunctionWithName:@"fused_attn_bwd"];
  if (!fusedBwdFunc) {
    NSURL *libURL = [NSURL fileURLWithPath:@"default.metallib"];
    NSError *libErr = nil;
    id<MTLLibrary> testLib = [device newLibraryWithURL:libURL error:&libErr];
    id<MTLFunction> testFn = [testLib newFunctionWithName:@"fused_attn_bwd"];
    std::cerr << "[DIAG] lib=" << (testLib != nil) << " fn=" << (testFn != nil)
              << " err=" << (libErr ? [[libErr localizedDescription] UTF8String] : "nil")
              << std::endl;
    std::cerr << "Failed to find 'fused_attn_bwd'\n"; return;
  }
  pipelineStateFusedAttnBwd = [device newComputePipelineStateWithFunction:fusedBwdFunc error:&error];
  if (!pipelineStateFusedAttnBwd) { std::cerr << "Failed to compile 'fused_attn_bwd'\n"; return; }
  std::cout << "Compiled 'fused_attn_bwd'!" << std::endl;

  id<MTLFunction> gemmBF16Func = [defaultLibrary newFunctionWithName:@"gemm_bf16"];
  if (!gemmBF16Func) { std::cerr << "Failed to find 'gemm_bf16'\n"; return; }
  pipelineStateGEMMBF16 = [device newComputePipelineStateWithFunction:gemmBF16Func error:&error];
  if (!pipelineStateGEMMBF16) { std::cerr << "Failed to compile 'gemm_bf16'\n"; return; }
  std::cout << "Compiled 'gemm_bf16'!" << std::endl;

  id<MTLFunction> fusedAddNormFunc = [defaultLibrary newFunctionWithName:@"fused_add_norm"];
  if (!fusedAddNormFunc) { std::cerr << "Failed to find 'fused_add_norm'\n"; return; }
  pipelineStateFusedAddNorm = [device newComputePipelineStateWithFunction:fusedAddNormFunc error:&error];
  if (!pipelineStateFusedAddNorm) { std::cerr << "Failed to compile 'fused_add_norm'\n"; return; }
  std::cout << "Compiled 'fused_add_norm'!" << std::endl;

  id<MTLFunction> fusedBwdAddNormFunc = [defaultLibrary newFunctionWithName:@"fused_backward_add_norm"];
  if (!fusedBwdAddNormFunc) { std::cerr << "Failed to find 'fused_backward_add_norm'\n"; return; }
  pipelineStateFusedBackwardAddNorm = [device newComputePipelineStateWithFunction:fusedBwdAddNormFunc error:&error];
  if (!pipelineStateFusedBackwardAddNorm) { std::cerr << "Failed to compile 'fused_backward_add_norm'\n"; return; }
  std::cout << "Compiled 'fused_backward_add_norm'!" << std::endl;

  // Wire up the gpu_wrapper release function so PagedBuffer destructors
  // can call CFRelease without including ObjC headers.
  PagedBuffer::gpu_release_fn = [](void* wrapper) {
    if (wrapper) {
      CFRelease((__bridge CFTypeRef)wrapper);
    }
  };

  // ── Diagnostic dump: pipeline state static metrics ────────────────
  if (getenv("GPU_PROFILE")) {
    auto print_pipeline = [](id<MTLComputePipelineState> ps, const char* name) {
      if (!ps) { printf("  %-30s  NOT COMPILED\n", name); return; }
      NSUInteger tgMem = ps.staticThreadgroupMemoryLength;
      NSUInteger maxThreads = ps.maxTotalThreadsPerThreadgroup;
      NSUInteger execWidth = ps.threadExecutionWidth;
      printf("  %-30s  tgMem=%4lu B  maxThreads=%4lu  execWidth=%lu\n",
             name, (unsigned long)tgMem, (unsigned long)maxThreads, (unsigned long)execWidth);
    };
    printf("\n=== GPU PIPELINE STATE DIAGNOSTICS ===\n");
    print_pipeline(pipelineStateGEMMBF16,    "gemm_bf16");
    print_pipeline(pipelineStateFusedAttnBwd,"fused_attn_bwd");
    print_pipeline(pipelineStateFlashAttnFwd,"flash_attn_fwd");
    print_pipeline(pipelineStateRMSForward,  "rms_norm_forward");
    print_pipeline(pipelineStateRMSBackwardDX,"rms_norm_backward");
    print_pipeline(pipelineStateResidualAdd, "residual_add");
    print_pipeline(pipelineStateSwiGLUForward,"swiglu_forward");
    print_pipeline(pipelineStateSwiGLUBackward,"swiglu_backward");
    print_pipeline(pipelineStateAdamW,       "adamw_step");
    print_pipeline(pipelineStateRoPEForward, "rope_forward");
    print_pipeline(pipelineStateRoPEBackward,"rope_backward");
    print_pipeline(pipelineStateCrossEntropy,"cross_entropy");
    print_pipeline(pipelineStateEmbeddingFwd,"embedding_forward");
    print_pipeline(pipelineStateGQABackward, "gqa_backward");
    print_pipeline(pipelineStateGQAScores,   "gqa_scores");
    print_pipeline(pipelineStateAttnSoftmax, "attn_softmax");
    print_pipeline(pipelineStateAttnDS,      "attn_ds");
    print_pipeline(pipelineStateGEMMBackward,"gemm_backward");
    print_pipeline(pipelineStateGEMMProjTransB,"gemm_proj_trans_b");
    print_pipeline(pipelineStateReshape4D,   "reshape_to_4d");
    print_pipeline(pipelineStateReshape3D,   "reshape_to_3d");
    printf("=== END DIAGNOSTICS ===\n\n");
  }

  initialized = true;
}

void gemm_ffn(const float *a, const float *b_gate, const float *b_up, float *c,
              size_t M, size_t N, size_t K) {
  if (M == 0 || N == 0 || K == 0)
    return;

  size_t bytesA = M * K * sizeof(__bf16);
  size_t bytesB = K * N * sizeof(__bf16);
  size_t bytesC = M * N * sizeof(__bf16);

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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateFFN];
    [e__ setBuffer:bufferA offset:0 atIndex:0];
    [e__ setBuffer:bufferB_gate offset:0 atIndex:1];
    [e__ setBuffer:bufferB_up offset:0 atIndex:2];
    [e__ setBuffer:bufferC offset:0 atIndex:3];
    [e__ setBytes:&dimensions length:sizeof(simd::uint3) atIndex:4];

    MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);

    [e__ dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateFFN];
      [e__ setBuffer:bufferA offset:0 atIndex:0];
      [e__ setBuffer:bufferB_gate offset:0 atIndex:1];
      [e__ setBuffer:bufferB_up offset:0 atIndex:2];
      [e__ setBuffer:bufferC offset:0 atIndex:3];
      [e__ setBytes:&dimensions
                        length:sizeof(simd::uint3)
                       atIndex:4];

      MTLSize threadgroupsPerGrid =
          MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);

      MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
      [e__ dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
      // end encoding handled above
    }
  }
}

// Use Accelerate cblas_sgemm for all standard GEMMs — runs on the AMX coprocessor
// which delivers 10+ TFLOPS on M3 Ultra.  Custom Metal kernels are reserved for
// fused/specialized operations (gemm_ffn, gemm_gqa) where AMX can't help.

void gemm_proj(const float *a, const float *b, float *c, size_t M, size_t N,
               size_t K) {
  if (M == 0 || N == 0 || K == 0) return;
  auto start = std::chrono::high_resolution_clock::now();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)M, (int)N, (int)K,
              1.0f, a, (int)K, b, (int)N, 0.0f, c, (int)N);
  auto end = std::chrono::high_resolution_clock::now();
  accum_cpu_time_ms += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  count_cpu_calls++;
}

void gemm_gqa(const GQAParams &gqa_params, const float *q, const float *k,
              const float *v, float *out_gqa) {

  size_t bytesQ = gqa_params.batch * gqa_params.seq_len * gqa_params.n_q_heads *
                  gqa_params.head_dim * sizeof(__bf16);
  size_t bytesK = gqa_params.batch * gqa_params.seq_len *
                  gqa_params.n_kv_heads * gqa_params.head_dim * sizeof(__bf16);
  size_t bytesV = gqa_params.batch * gqa_params.seq_len *
                  gqa_params.n_kv_heads * gqa_params.head_dim * sizeof(__bf16);
  size_t bytesOut = gqa_params.batch * gqa_params.seq_len *
                    gqa_params.n_q_heads * gqa_params.head_dim * sizeof(__bf16);

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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateGQA];
    [e__ setBuffer:bufferQ offset:0 atIndex:0];
    [e__ setBuffer:bufferK offset:0 atIndex:1];
    [e__ setBuffer:bufferV offset:0 atIndex:2];
    [e__ setBuffer:bufferOut offset:0 atIndex:3];
    [e__ setBytes:&gqa_params length:sizeof(GQAParams) atIndex:4];

    [e__ dispatchThreads:threadsPerGrid
             threadsPerThreadgroup:threadsPerThreadgroup];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];

      [e__ setComputePipelineState:pipelineStateGQA];
      [e__ setBuffer:bufferQ offset:0 atIndex:0];
      [e__ setBuffer:bufferK offset:0 atIndex:1];
      [e__ setBuffer:bufferV offset:0 atIndex:2];
      [e__ setBuffer:bufferOut offset:0 atIndex:3];
      [e__ setBytes:&gqa_params length:sizeof(GQAParams) atIndex:4];

      [e__ dispatchThreads:threadsPerGrid
                threadsPerThreadgroup:threadsPerThreadgroup];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
      // end encoding handled above
    }
  }
}
static id<MTLBuffer> get_or_create_buffer(const void *ptr, size_t bytes,
                                          bool is_write,
                                          bool is_persistent,
                                          bool is_host_input) {
  if (ptr == nullptr || bytes == 0)
    return nil;

  // ── O(0) fast path: PagedBuffer owns a gpu_wrapper ────────────────
  void** wrapper_loc = find_gpu_wrapper(ptr);
  if (wrapper_loc) {
    id<MTLBuffer> buf = *wrapper_loc ? (__bridge id<MTLBuffer>)*wrapper_loc : nil;
    if (!buf) {
      // First GPU access for this PagedBuffer — create the Metal wrapper
      buf = [device newBufferWithBytesNoCopy:(void*)ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                 deallocator:nil];
      if (buf) {
        CFRetain((__bridge CFTypeRef)buf);
        *wrapper_loc = (void*)buf;
      }
    }
    if (buf && is_write && is_host_input) {
      // Host-input flag: upload CPU data even if buffer exists
      memcpy([buf contents], ptr, bytes);
    }
    return buf;
  }

  // ── Fallback: pointer not owned by any PagedBuffer ─────────────────
  // Check persistent fallback cache first (avoids re-creating buffers for
  // static pointers like &last_step_loss).
  {
    auto fb_it = persistent_fallback_cache.find(ptr);
    if (fb_it != persistent_fallback_cache.end()) return fb_it->second;
  }

  bool is_page_aligned = (((uintptr_t)ptr) & 0x3FFF) == 0;
  id<MTLBuffer> buf = nil;
  
  if (is_page_aligned) {
    buf = [device newBufferWithBytesNoCopy:(void*)ptr
                                    length:bytes
                                   options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                               deallocator:nil];
    if (buf) CFRetain((__bridge CFTypeRef)buf);
  }
  
  if (!buf) {
    auto &bucket = sizePool[bytes];
    if (!bucket.empty()) { buf = bucket.back(); bucket.pop_back(); }
    else { buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked]; CFRetain((__bridge CFTypeRef)buf); }
  }

  if (buf) {
    persistent_fallback_cache[ptr] = buf;
    activeAllocationsThisStep.push_back({bytes, buf});

    if (is_write) {
      if (!is_page_aligned)
        copyBackQueue.push_back({const_cast<void *>(ptr), buf, bytes});
    } else if (is_host_input) {
      if (!is_page_aligned) memcpy([buf contents], ptr, bytes);
    } else {
      if (!is_page_aligned) memcpy([buf contents], ptr, bytes);
    }
  }

  return buf;
}

void rms_norm_forward(const float *input, float *output, const float *weight,
                      float eps, size_t num_rows, size_t dims) {
  if (num_rows == 0 || dims == 0)
    return;

  size_t bytesIn = num_rows * dims * sizeof(__bf16);
  size_t bytesOut = num_rows * dims * sizeof(__bf16);
  size_t bytesW = dims * sizeof(__bf16);

  id<MTLBuffer> bufferIn = get_or_create_buffer(input, bytesIn);
  id<MTLBuffer> bufferOut = get_or_create_buffer(output, bytesOut, true);
  id<MTLBuffer> bufferW = get_or_create_buffer(weight, bytesW, false, true);

  if (!bufferIn || !bufferOut || !bufferW) {
    std::cerr << "Failed to allocate Metal buffers for rms_norm_forward!"
              << std::endl;
    return;
  }

  uint dims_val = (uint)dims;

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateRMSForward];
    [e__ setBuffer:bufferIn offset:0 atIndex:0];
    [e__ setBuffer:bufferOut offset:0 atIndex:1];
    [e__ setBuffer:bufferW offset:0 atIndex:2];
    [e__ setBytes:&eps length:sizeof(float) atIndex:3];
    [e__ setBytes:&dims_val length:sizeof(uint) atIndex:4];

    MTLSize threadgroupsPerGrid = MTLSizeMake(num_rows, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);

    [e__ dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];

      [e__ setComputePipelineState:pipelineStateRMSForward];
      [e__ setBuffer:bufferIn offset:0 atIndex:0];
      [e__ setBuffer:bufferOut offset:0 atIndex:1];
      [e__ setBuffer:bufferW offset:0 atIndex:2];
      [e__ setBytes:&eps length:sizeof(float) atIndex:3];
      [e__ setBytes:&dims_val length:sizeof(uint) atIndex:4];

      MTLSize threadgroupsPerGrid = MTLSizeMake(num_rows, 1, 1);
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);

      [e__ dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
      // end encoding handled above
    }
  }
}

void rms_norm_backward(const float *grad_output, const float *input,
                       const float *weight, float *grad_input,
                       float *grad_weight, float eps, size_t num_rows,
                       size_t dims) {
  if (num_rows == 0 || dims == 0)
    return;

  size_t bytesIn = num_rows * dims * sizeof(__bf16);
  size_t bytesW = dims * sizeof(__bf16);

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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateRMSBackwardDX];
    [e__ setBuffer:bufferGradOutput offset:0 atIndex:0];
    [e__ setBuffer:bufferInput offset:0 atIndex:1];
    [e__ setBuffer:bufferWeight offset:0 atIndex:2];
    [e__ setBuffer:bufferGradInput offset:0 atIndex:3];
    [e__ setBuffer:bufferGradWeight offset:0 atIndex:4];
    [e__ setBytes:&eps length:sizeof(float) atIndex:5];
    [e__ setBytes:&dims_val length:sizeof(uint) atIndex:6];

    MTLSize threadgroupsPerGrid1 = MTLSizeMake(num_rows, 1, 1);
    MTLSize threadsPerThreadgroup1 = MTLSizeMake(256, 1, 1);

    [e__ dispatchThreadgroups:threadgroupsPerGrid1
                  threadsPerThreadgroup:threadsPerThreadgroup1];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];

      [e__ setComputePipelineState:pipelineStateRMSBackwardDX];
      [e__ setBuffer:bufferGradOutput offset:0 atIndex:0];
      [e__ setBuffer:bufferInput offset:0 atIndex:1];
      [e__ setBuffer:bufferWeight offset:0 atIndex:2];
      [e__ setBuffer:bufferGradInput offset:0 atIndex:3];
      [e__ setBuffer:bufferGradWeight offset:0 atIndex:4];
      [e__ setBytes:&eps length:sizeof(float) atIndex:5];
      [e__ setBytes:&dims_val length:sizeof(uint) atIndex:6];

      MTLSize threadgroupsPerGrid1 = MTLSizeMake(num_rows, 1, 1);
      MTLSize threadsPerThreadgroup1 = MTLSizeMake(256, 1, 1);

      [e__ dispatchThreadgroups:threadgroupsPerGrid1
                     threadsPerThreadgroup:threadsPerThreadgroup1];

      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
      // end encoding handled above
    }
  }
}

void swiglu_backward(const float *grad_output, const float *gate,
                     const float *up, float *grad_gate, float *grad_up,
                     size_t n) {
  if (n == 0)
    return;
  size_t bytesIn = n * sizeof(__bf16);
  size_t bytesOut = n * sizeof(__bf16);
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
  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateSwiGLUBackward];
    [e__ setBuffer:bufferGradOutput offset:0 atIndex:0];
    [e__ setBuffer:bufferGate offset:0 atIndex:1];
    [e__ setBuffer:bufferUp offset:0 atIndex:2];
    [e__ setBuffer:bufferGradGate offset:0 atIndex:3];
    [e__ setBuffer:bufferGradUp offset:0 atIndex:4];

    uint n_val = (uint)n;
    [e__ setBytes:&n_val length:sizeof(uint) atIndex:5];

    MTLSize threadgroupsPerGrid = MTLSizeMake((n + 255) / 256, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
    [e__ dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateSwiGLUBackward];
      [e__ setBuffer:bufferGradOutput offset:0 atIndex:0];
      [e__ setBuffer:bufferGate offset:0 atIndex:1];
      [e__ setBuffer:bufferUp offset:0 atIndex:2];
      [e__ setBuffer:bufferGradGate offset:0 atIndex:3];
      [e__ setBuffer:bufferGradUp offset:0 atIndex:4];

      uint n_val = (uint)n;
      [e__ setBytes:&n_val length:sizeof(uint) atIndex:5];

      MTLSize threadgroupsPerGrid = MTLSizeMake((n + 255) / 256, 1, 1);
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      [e__ dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
      // end encoding handled above
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
  size_t bytesGrad = batch * heads * seq_len * head_dim * sizeof(__bf16);
  size_t bytesTable = seq_len * half_dim * sizeof(__bf16);

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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateRoPEBackward];
    [e__ setBuffer:bufferGrad offset:0 atIndex:0];
    [e__ setBuffer:bufferCos offset:0 atIndex:1];
    [e__ setBuffer:bufferSin offset:0 atIndex:2];
    [e__ setBytes:&b_val length:sizeof(uint) atIndex:3];
    [e__ setBytes:&h_val length:sizeof(uint) atIndex:4];
    [e__ setBytes:&s_val length:sizeof(uint) atIndex:5];
    [e__ setBytes:&d_val length:sizeof(uint) atIndex:6];

    MTLSize threadgroupsPerGrid = MTLSizeMake((total_pairs + 255) / 256, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
    [e__ dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateRoPEBackward];
      [e__ setBuffer:bufferGrad offset:0 atIndex:0];
      [e__ setBuffer:bufferCos offset:0 atIndex:1];
      [e__ setBuffer:bufferSin offset:0 atIndex:2];
      [e__ setBytes:&b_val length:sizeof(uint) atIndex:3];
      [e__ setBytes:&h_val length:sizeof(uint) atIndex:4];
      [e__ setBytes:&s_val length:sizeof(uint) atIndex:5];
      [e__ setBytes:&d_val length:sizeof(uint) atIndex:6];

      MTLSize threadgroupsPerGrid = MTLSizeMake((total_pairs + 255) / 256, 1, 1);
      MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
      [e__ dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
      // end encoding handled above
    }
  }
}

void gqa_scores(const float *Q, const float *K, float *scores,
                size_t batch, size_t n_heads, size_t n_kv,
                size_t seq_len, size_t head_dim) {
  size_t bytesQ = batch * n_heads * seq_len * head_dim * sizeof(__bf16);
  size_t bytesK = batch * n_kv * seq_len * head_dim * sizeof(__bf16);
  size_t bytesScores = batch * n_heads * seq_len * seq_len * sizeof(__bf16);

  id<MTLBuffer> bufQ = get_or_create_buffer(Q, bytesQ);
  id<MTLBuffer> bufK = get_or_create_buffer(K, bytesK);
  id<MTLBuffer> bufS = get_or_create_buffer(scores, bytesScores, true);

  if (!bufQ || !bufK || !bufS) {
    std::cerr << "[gqa_scores] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  uint u_batch = (uint)batch, u_nh = (uint)n_heads, u_nkv = (uint)n_kv;
  uint u_seq = (uint)seq_len, u_hd = (uint)head_dim;

  MTLSize grid = MTLSizeMake(batch, n_heads, seq_len * seq_len);
  MTLSize tg = MTLSizeMake(1, 1, 256);

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateGQAScores];
    [e__ setBuffer:bufQ offset:0 atIndex:0];
    [e__ setBuffer:bufK offset:0 atIndex:1];
    [e__ setBuffer:bufS offset:0 atIndex:2];
    [e__ setBytes:&u_batch length:sizeof(uint) atIndex:3];
    [e__ setBytes:&u_nh    length:sizeof(uint) atIndex:4];
    [e__ setBytes:&u_nkv   length:sizeof(uint) atIndex:5];
    [e__ setBytes:&u_seq   length:sizeof(uint) atIndex:6];
    [e__ setBytes:&u_hd    length:sizeof(uint) atIndex:7];
    [e__ dispatchThreads:grid threadsPerThreadgroup:tg];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pipelineStateGQAScores];
      [enc setBuffer:bufQ offset:0 atIndex:0];
      [enc setBuffer:bufK offset:0 atIndex:1];
      [enc setBuffer:bufS offset:0 atIndex:2];
      [enc setBytes:&u_batch length:sizeof(uint) atIndex:3];
      [enc setBytes:&u_nh    length:sizeof(uint) atIndex:4];
      [enc setBytes:&u_nkv   length:sizeof(uint) atIndex:5];
      [enc setBytes:&u_seq   length:sizeof(uint) atIndex:6];
      [enc setBytes:&u_hd    length:sizeof(uint) atIndex:7];
      [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
      [cb commit];
      // no FIFO tracking — single command buffer
      [cb waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void attn_softmax(const float *scores, float *probs,
                  size_t batch, size_t n_heads, size_t seq_len) {
  size_t bytes = batch * n_heads * seq_len * seq_len * sizeof(__bf16);
  id<MTLBuffer> bufS = get_or_create_buffer(scores, bytes);
  id<MTLBuffer> bufP = get_or_create_buffer(probs, bytes, true);
  if (!bufS || !bufP) { std::cerr << "[attn_softmax] buffer alloc failed\n"; return; }
  uint ub = (uint)batch, unh = (uint)n_heads, us = (uint)seq_len;
  MTLSize grid = MTLSizeMake(batch, n_heads, seq_len);
  MTLSize tg = MTLSizeMake(1, 1, 256);
  auto enc = ^(id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:pipelineStateAttnSoftmax];
    [e setBuffer:bufS offset:0 atIndex:0];
    [e setBuffer:bufP offset:0 atIndex:1];
    [e setBytes:&ub  length:sizeof(uint) atIndex:2];
    [e setBytes:&unh length:sizeof(uint) atIndex:3];
    [e setBytes:&us  length:sizeof(uint) atIndex:4];
    [e dispatchThreads:grid threadsPerThreadgroup:tg];
  };
  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) { id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder]; enc(e__); [e__ endEncoding]; }
  else { @autoreleasepool {
    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    enc(e); [e endEncoding];
      [cb commit];
      // no FIFO tracking — single command buffer
      [cb waitUntilCompleted];
      run_copy_back_tasks();
  }}
}

void attn_ds(const float *probs, const float *dP_row, float *dS_out,
             size_t batch, size_t n_heads, size_t seq_len) {
  size_t bytes = batch * n_heads * seq_len * seq_len * sizeof(__bf16);
  id<MTLBuffer> bufP = get_or_create_buffer(probs, bytes);
  id<MTLBuffer> bufD = get_or_create_buffer(dP_row, bytes);
  id<MTLBuffer> bufO = get_or_create_buffer(dS_out, bytes, true);
  if (!bufP || !bufD || !bufO) { std::cerr << "[attn_ds] buffer alloc failed\n"; return; }
  uint ub=(uint)batch, unh=(uint)n_heads, us=(uint)seq_len;
  MTLSize grid = MTLSizeMake(batch, n_heads, seq_len);
  MTLSize tg = MTLSizeMake(1, 1, 256);
  auto enc = ^(id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:pipelineStateAttnDS];
    [e setBuffer:bufP offset:0 atIndex:0];
    [e setBuffer:bufD offset:0 atIndex:1];
    [e setBuffer:bufO offset:0 atIndex:2];
    [e setBytes:&ub  length:sizeof(uint) atIndex:3];
    [e setBytes:&unh length:sizeof(uint) atIndex:4];
    [e setBytes:&us  length:sizeof(uint) atIndex:5];
    [e dispatchThreads:grid threadsPerThreadgroup:tg];
  };
  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) { id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder]; enc(e__); [e__ endEncoding]; }
  else { @autoreleasepool {
    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    enc(e); [e endEncoding];
      [cb commit];
      // no FIFO tracking — single command buffer
      [cb waitUntilCompleted];
      run_copy_back_tasks();
  }}
}

void swiglu_forward(const float *gate, const float *up, float *out, size_t n) {
  if (n == 0) return;
  size_t bytes = n * sizeof(__bf16);
  id<MTLBuffer> bufG = get_or_create_buffer(gate, bytes);
  id<MTLBuffer> bufU = get_or_create_buffer(up, bytes);
  id<MTLBuffer> bufO = get_or_create_buffer(out, bytes, true);
  if (!bufG || !bufU || !bufO) { std::cerr << "[swiglu_forward] buffer alloc failed\n"; return; }
  uint un = (uint)n;
  MTLSize grid = MTLSizeMake((n + 255) / 256, 1, 1);
  MTLSize tg = MTLSizeMake(256, 1, 1);
  auto enc = ^(id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:pipelineStateSwiGLUForward];
    [e setBuffer:bufG offset:0 atIndex:0];
    [e setBuffer:bufU offset:0 atIndex:1];
    [e setBuffer:bufO offset:0 atIndex:2];
    [e setBytes:&un length:sizeof(uint) atIndex:3];
    [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  };
  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) { id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder]; enc(e__); [e__ endEncoding]; }
  else { @autoreleasepool {
    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    enc(e); [e endEncoding];
      [cb commit];
      // no FIFO tracking — single command buffer
      [cb waitUntilCompleted];
      run_copy_back_tasks();
  }}
}

void fused_attn_bwd(const void *Q, const void *K, const void *V,
                     const void *dO, void *dQ, void *dK, void *dV,
                     size_t batch, size_t n_heads, size_t n_kv,
                     size_t seq_len, size_t head_dim) {
  size_t bytesQ = batch * n_heads * seq_len * head_dim * sizeof(__bf16);
  size_t bytesKV = batch * n_kv * seq_len * head_dim * sizeof(__bf16);
  size_t bytesDO = batch * seq_len * n_heads * head_dim * sizeof(__bf16);

  id<MTLBuffer> bQ = get_or_create_buffer(Q, bytesQ);
  id<MTLBuffer> bK = get_or_create_buffer(K, bytesKV);
  id<MTLBuffer> bV = get_or_create_buffer(V, bytesKV);
  id<MTLBuffer> bdO = get_or_create_buffer(dO, bytesDO);
  id<MTLBuffer> bdQ = get_or_create_buffer(dQ, bytesQ, true);
  id<MTLBuffer> bdK = get_or_create_buffer(dK, bytesKV, true);
  id<MTLBuffer> bdV = get_or_create_buffer(dV, bytesKV, true);
  if (!bQ || !bK || !bV || !bdO || !bdQ || !bdK || !bdV) {
    std::cerr << "[fused_attn_bwd] buffer alloc failed\n"; return;
  }
  uint32_t u_nh = (uint32_t)n_heads, u_nkv = (uint32_t)n_kv;
  MTLSize grid = MTLSizeMake(batch, n_heads, 1);
  MTLSize tg = MTLSizeMake(256, 1, 1);
  auto enc = ^(id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:pipelineStateFusedAttnBwd];
    [e setBuffer:bQ offset:0 atIndex:0]; [e setBuffer:bK offset:0 atIndex:1];
    [e setBuffer:bV offset:0 atIndex:2]; [e setBuffer:bdO offset:0 atIndex:3];
    [e setBuffer:bdQ offset:0 atIndex:4]; [e setBuffer:bdK offset:0 atIndex:5];
    [e setBuffer:bdV offset:0 atIndex:6];
    [e setBytes:&u_nh length:sizeof(uint32_t) atIndex:7];
    [e setBytes:&u_nkv length:sizeof(uint32_t) atIndex:8];
    [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  };
  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) { id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder]; enc(e__); [e__ endEncoding]; }
  else { @autoreleasepool {
    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    enc(e); [e endEncoding];
      [cb commit];
      // no FIFO tracking — single command buffer
      [cb waitUntilCompleted];
      run_copy_back_tasks();
  }}
}

void flash_attn_fwd(const void *Q, const void *K, const void *V, void *O,
                     size_t batch, size_t n_heads, size_t n_kv,
                     size_t seq_len, size_t head_dim) {
  size_t bytesQ = batch * n_heads * seq_len * head_dim * sizeof(__bf16);
  size_t bytesKV = batch * n_kv * seq_len * head_dim * sizeof(__bf16);
  size_t bytesO  = batch * n_heads * seq_len * head_dim * sizeof(__bf16);

  id<MTLBuffer> bQ = get_or_create_buffer(Q, bytesQ);
  id<MTLBuffer> bK = get_or_create_buffer(K, bytesKV);
  id<MTLBuffer> bV = get_or_create_buffer(V, bytesKV);
  id<MTLBuffer> bO = get_or_create_buffer(O, bytesO, true);
  if (!bQ || !bK || !bV || !bO) {
    std::cerr << "[flash_attn_fwd] buffer alloc failed\n"; return;
  }
  uint32_t u_nh = (uint32_t)n_heads, u_nkv = (uint32_t)n_kv;
  MTLSize grid = MTLSizeMake(batch, n_heads, 1);
  MTLSize tg = MTLSizeMake(256, 1, 1);
  auto enc = ^(id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:pipelineStateFlashAttnFwd];
    [e setBuffer:bQ offset:0 atIndex:0];
    [e setBuffer:bK offset:0 atIndex:1];
    [e setBuffer:bV offset:0 atIndex:2];
    [e setBuffer:bO offset:0 atIndex:3];
    [e setBytes:&u_nh  length:sizeof(uint32_t) atIndex:4];
    [e setBytes:&u_nkv length:sizeof(uint32_t) atIndex:5];
    [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  };
  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) { id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder]; enc(e__); [e__ endEncoding]; }
  else { @autoreleasepool {
    id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    enc(e); [e endEncoding];
      [cb commit];
      // no FIFO tracking — single command buffer
      [cb waitUntilCompleted];
      run_copy_back_tasks();
  }}
}

void gemm_bf16(const void *A, const void *B, void *C,
                 size_t M, size_t N, size_t K,
                 bool transA, bool transB, bool is_forward) {
  auto prof_start = std::chrono::high_resolution_clock::now();

  size_t bytesA = (transA ? K : M) * (transA ? M : K) * sizeof(__bf16);
  size_t bytesB = (transB ? N : K) * (transB ? K : N) * sizeof(__bf16);
  size_t bytesC = M * N * sizeof(__bf16);

  id<MTLBuffer> bA = get_or_create_buffer(A, bytesA);
  id<MTLBuffer> bB = get_or_create_buffer(B, bytesB);
  id<MTLBuffer> bC = get_or_create_buffer(C, bytesC, true);
  if (!bA || !bB || !bC) { std::cerr << "[gemm_bf16] buffer alloc failed\n"; return; }

  uint uM = (uint)M, uN = (uint)N, uK = (uint)K;
  // BF16 → buffer expects bfloat. Tensor stores float.  The kernel casts.
  // We pass the raw float pointers and the kernel reads as bfloat via cast.

  MTLSize grid = MTLSizeMake((N + 127) / 128, (M + 127) / 128, 1);
  MTLSize tg  = MTLSizeMake(256, 1, 1);

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateGEMMBF16];
    [e__ setBuffer:bA offset:0 atIndex:0];
    [e__ setBuffer:bB offset:0 atIndex:1];
    [e__ setBuffer:bC offset:0 atIndex:2];
    [e__ setBytes:&uM length:sizeof(uint) atIndex:3];
    [e__ setBytes:&uN length:sizeof(uint) atIndex:4];
    [e__ setBytes:&uK length:sizeof(uint) atIndex:5];
    [e__ setBytes:&transA length:sizeof(bool) atIndex:6];
    [e__ setBytes:&transB length:sizeof(bool) atIndex:7];
    [e__ setBytes:&is_forward length:sizeof(bool) atIndex:8];
    [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pipelineStateGEMMBF16];
      [enc setBuffer:bA offset:0 atIndex:0];
      [enc setBuffer:bB offset:0 atIndex:1];
      [enc setBuffer:bC offset:0 atIndex:2];
      [enc setBytes:&uM length:sizeof(uint) atIndex:3];
      [enc setBytes:&uN length:sizeof(uint) atIndex:4];
      [enc setBytes:&uK length:sizeof(uint) atIndex:5];
      [enc setBytes:&transA length:sizeof(bool) atIndex:6];
      [enc setBytes:&transB length:sizeof(bool) atIndex:7];
      [enc setBytes:&is_forward length:sizeof(bool) atIndex:8];
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cb commit];
      // Async FIFO tracking — no waitUntilCompleted
      // no FIFO tracking — single command buffer
      [cb waitUntilCompleted];
      run_copy_back_tasks();
    }
  }

  // Record profiling data
  auto prof_end = std::chrono::high_resolution_clock::now();
  double prof_ms = std::chrono::duration<double, std::milli>(prof_end - prof_start).count();
  GemmProfiler::instance().record(M, N, K, transA, transB, prof_ms);
}

void gqa_backward(const GQABackwardParams &params,
                  const float *Q, const float *K, const float *V,
                  const float *grad_attn_output,
                  float *grad_Q, float *grad_K, float *grad_V,
                  const float *precomputed_scores) {
  size_t bytesQ = params.batch * params.n_q_heads * params.seq_len *
                  params.head_dim * sizeof(__bf16);
  size_t bytesKV = params.batch * params.n_kv_heads * params.seq_len *
                   params.head_dim * sizeof(__bf16);
  size_t bytesGradOut = params.batch * params.seq_len * params.n_q_heads *
                        params.head_dim * sizeof(__bf16);
  size_t bytesScores = params.batch * params.n_q_heads *
                       params.seq_len * params.seq_len * sizeof(__bf16);

  id<MTLBuffer> bufferQ = get_or_create_buffer(Q, bytesQ);
  id<MTLBuffer> bufferK = get_or_create_buffer(K, bytesKV);
  id<MTLBuffer> bufferV = get_or_create_buffer(V, bytesKV);
  id<MTLBuffer> bufferGradOut = get_or_create_buffer(grad_attn_output, bytesGradOut);
  id<MTLBuffer> bufferGradQ = get_or_create_buffer(grad_Q, bytesQ, true);
  id<MTLBuffer> bufferGradK = get_or_create_buffer(grad_K, bytesKV, true);
  id<MTLBuffer> bufferGradV = get_or_create_buffer(grad_V, bytesKV, true);
  id<MTLBuffer> bufferScores = precomputed_scores ?
      get_or_create_buffer(precomputed_scores, bytesScores) : nil;

  if (!bufferQ || !bufferK || !bufferV || !bufferGradOut ||
      !bufferGradQ || !bufferGradK || !bufferGradV) {
    std::cerr << "Failed to allocate Metal buffers for gqa_backward!"
              << std::endl;
    return;
  }

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
  uint tg_size_z = 32;
  MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, tg_size_z);

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateGQABackward];
    [e__ setBuffer:bufferQ offset:0 atIndex:0];
    [e__ setBuffer:bufferK offset:0 atIndex:1];
    [e__ setBuffer:bufferV offset:0 atIndex:2];
    [e__ setBuffer:bufferGradOut offset:0 atIndex:3];
    [e__ setBuffer:bufferGradQ offset:0 atIndex:4];
    [e__ setBuffer:bufferGradK offset:0 atIndex:5];
    [e__ setBuffer:bufferGradV offset:0 atIndex:6];
    [e__ setBytes:&gpu_params length:sizeof(gpu_params) atIndex:7];
    if (bufferScores)
      [e__ setBuffer:bufferScores offset:0 atIndex:8];

    [e__ dispatchThreads:threadsPerGrid
             threadsPerThreadgroup:threadsPerThreadgroup];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];

      [e__ setComputePipelineState:pipelineStateGQABackward];
      [e__ setBuffer:bufferQ offset:0 atIndex:0];
      [e__ setBuffer:bufferK offset:0 atIndex:1];
      [e__ setBuffer:bufferV offset:0 atIndex:2];
      [e__ setBuffer:bufferGradOut offset:0 atIndex:3];
      [e__ setBuffer:bufferGradQ offset:0 atIndex:4];
      [e__ setBuffer:bufferGradK offset:0 atIndex:5];
      [e__ setBuffer:bufferGradV offset:0 atIndex:6];
      [e__ setBytes:&gpu_params length:sizeof(gpu_params) atIndex:7];
      if (bufferScores)
        [e__ setBuffer:bufferScores offset:0 atIndex:8];

      [e__ dispatchThreads:threadsPerGrid
               threadsPerThreadgroup:threadsPerThreadgroup];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
      // end encoding handled above
    }
  }
}

void adamw_step(float *param, const float *grad, float *m, float *v,
                const AdamWStepParams &params) {
  if (params.n == 0)
    return;

  size_t bytesBF16 = params.n * sizeof(__bf16);
  size_t bytesFP32 = params.n * sizeof(float);

  id<MTLBuffer> bufferParam = get_or_create_buffer(param, bytesBF16, true, true);
  id<MTLBuffer> bufferGrad = get_or_create_buffer(grad, bytesBF16, false, false);
  id<MTLBuffer> bufferM = get_or_create_buffer(m, bytesFP32, true, true);
  id<MTLBuffer> bufferV = get_or_create_buffer(v, bytesFP32, true, true);

  if (!bufferParam || !bufferGrad || !bufferM || !bufferV) {
    std::cerr << "[adamw_step] buf fail: param=" << (bufferParam!=nil)
              << " grad=" << (bufferGrad!=nil) << " m=" << (bufferM!=nil)
              << " v=" << (bufferV!=nil) << " n=" << params.n
              << " bf16=" << bytesBF16 << " fp32=" << bytesFP32 << std::endl;
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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateAdamW];
    [e__ setBuffer:bufferParam offset:0 atIndex:0];
    [e__ setBuffer:bufferGrad offset:0 atIndex:1];
    [e__ setBuffer:bufferM offset:0 atIndex:2];
    [e__ setBuffer:bufferV offset:0 atIndex:3];
    [e__ setBytes:&gpu_params length:sizeof(gpu_params) atIndex:4];

    [e__ dispatchThreadgroups:threadgroupsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];

      [e__ setComputePipelineState:pipelineStateAdamW];
      [e__ setBuffer:bufferParam offset:0 atIndex:0];
      [e__ setBuffer:bufferGrad offset:0 atIndex:1];
      [e__ setBuffer:bufferM offset:0 atIndex:2];
      [e__ setBuffer:bufferV offset:0 atIndex:3];
      [e__ setBytes:&gpu_params length:sizeof(gpu_params) atIndex:4];

      [e__ dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
      // end encoding handled above
    }
  }
}

void residual_add(float *a, const float *b, size_t n) {
  // WHAT: Guard: nothing to add for empty tensors.
  if (n == 0 || a == nullptr || b == nullptr) return;

  size_t bytes = n * sizeof(__bf16);

  id<MTLBuffer> bufA = get_or_create_buffer(a, bytes, true);
  id<MTLBuffer> bufB = get_or_create_buffer(b, bytes, false);
  uint32_t un = static_cast<uint32_t>(n);

  if (!bufA || !bufB) {
    std::cerr << "[residual_add] Failed to allocate Metal buffers." << std::endl;
    return;
  }

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateResidualAdd];
    [e__ setBuffer:bufA offset:0 atIndex:0];
    [e__ setBuffer:bufB offset:0 atIndex:1];
    [e__ setBytes:&un length:sizeof(uint32_t) atIndex:2];

    MTLSize threads = MTLSizeMake(256, 1, 1);
    MTLSize grid    = MTLSizeMake((n + 255) / 256, 1, 1);
    [e__ dispatchThreadgroups:grid threadsPerThreadgroup:threads];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateResidualAdd];
      [e__ setBuffer:bufA offset:0 atIndex:0];
      [e__ setBuffer:bufB offset:0 atIndex:1];
      [e__ setBytes:&un length:sizeof(uint32_t) atIndex:2];
      MTLSize threads = MTLSizeMake(256, 1, 1);
      MTLSize grid    = MTLSizeMake((n + 255) / 256, 1, 1);
      [e__ dispatchThreadgroups:grid threadsPerThreadgroup:threads];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void fused_add_norm(float *x_residual, const float *residual,
                    const float *weight, float *output,
                    size_t num_rows, size_t D, float eps) {
  if (num_rows == 0 || D == 0) return;
  size_t bytes = num_rows * D * sizeof(__bf16);
  size_t bytesW = D * sizeof(__bf16);

  id<MTLBuffer> bufX = get_or_create_buffer(x_residual, bytes, true);
  id<MTLBuffer> bufR = get_or_create_buffer(residual,   bytes, false);
  id<MTLBuffer> bufW = get_or_create_buffer(weight,     bytesW, false, true);
  id<MTLBuffer> bufO = get_or_create_buffer(output,     bytes, true);
  if (!bufX || !bufR || !bufW || !bufO) {
    std::cerr << "[fused_add_norm] buffer alloc failed\n"; return;
  }
  uint32_t uD = (uint32_t)D;
  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateFusedAddNorm];
    [e__ setBuffer:bufX offset:0 atIndex:0];
    [e__ setBuffer:bufR offset:0 atIndex:1];
    [e__ setBuffer:bufW offset:0 atIndex:2];
    [e__ setBuffer:bufO offset:0 atIndex:3];
    [e__ setBytes:&uD  length:sizeof(uint32_t) atIndex:4];
    [e__ setBytes:&eps length:sizeof(float) atIndex:5];
    MTLSize tg = MTLSizeMake(256, 1, 1);
    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:pipelineStateFusedAddNorm];
      [e setBuffer:bufX offset:0 atIndex:0];
      [e setBuffer:bufR offset:0 atIndex:1];
      [e setBuffer:bufW offset:0 atIndex:2];
      [e setBuffer:bufO offset:0 atIndex:3];
      [e setBytes:&uD  length:sizeof(uint32_t) atIndex:4];
      [e setBytes:&eps length:sizeof(float) atIndex:5];
      MTLSize tg = MTLSizeMake(256, 1, 1);
      MTLSize grid = MTLSizeMake(num_rows, 1, 1);
      [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [e endEncoding]; [cb commit];
      // no FIFO tracking — single command buffer
      [cb waitUntilCompleted]; run_copy_back_tasks();
    }
  }
}

void fused_backward_add_norm(const void *grad_output, const void *input,
                              const void *weight, const void *residual,
                              void *grad_input,
                              size_t num_rows, size_t D, float eps) {
  if (num_rows == 0 || D == 0) return;
  size_t bytes = num_rows * D * sizeof(__bf16);
  size_t bytesW = D * sizeof(__bf16);

  id<MTLBuffer> bGO  = get_or_create_buffer(grad_output, bytes,  false, false, false);
  id<MTLBuffer> bIn  = get_or_create_buffer(input,      bytes,  false, false, false);
  id<MTLBuffer> bW   = get_or_create_buffer(weight,     bytesW, false, true,  false);
  id<MTLBuffer> bRes = get_or_create_buffer(residual,   bytes,  false, false, false);
  id<MTLBuffer> bGI  = get_or_create_buffer(grad_input, bytes,  true,  false, false);
  if (!bGO || !bIn || !bW || !bRes || !bGI) {
    std::cerr << "[fused_backward_add_norm] buffer alloc failed\n"; return;
  }
  uint32_t uD = (uint32_t)D;

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateFusedBackwardAddNorm];
    [e__ setBuffer:bGO  offset:0 atIndex:0];
    [e__ setBuffer:bIn   offset:0 atIndex:1];
    [e__ setBuffer:bW    offset:0 atIndex:2];
    [e__ setBuffer:bRes  offset:0 atIndex:3];
    [e__ setBuffer:bGI   offset:0 atIndex:4];
    [e__ setBytes:&uD   length:sizeof(uint32_t) atIndex:5];
    [e__ setBytes:&eps  length:sizeof(float) atIndex:6];
    MTLSize tg = MTLSizeMake(256, 1, 1);
    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateFusedBackwardAddNorm];
      [e__ setBuffer:bGO  offset:0 atIndex:0];
      [e__ setBuffer:bIn   offset:0 atIndex:1];
      [e__ setBuffer:bW    offset:0 atIndex:2];
      [e__ setBuffer:bRes  offset:0 atIndex:3];
      [e__ setBuffer:bGI   offset:0 atIndex:4];
      [e__ setBytes:&uD   length:sizeof(uint32_t) atIndex:5];
      [e__ setBytes:&eps  length:sizeof(float) atIndex:6];
      MTLSize tg = MTLSizeMake(256, 1, 1);
      MTLSize grid = MTLSizeMake(num_rows, 1, 1);
      [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [e__ endEncoding];
      [cb2__ commit];
      [cb2__ waitUntilCompleted];
    }
  }
}

void embedding_forward(const uint32_t *token_ids, const float *embedding_table,
                       float *output, size_t total_tokens, size_t hidden_dim,
                       size_t vocab_size) {
  // WHAT: Guard: nothing to embed for empty input.
  if (total_tokens == 0 || hidden_dim == 0) return;

  size_t bytes_ids   = total_tokens * sizeof(uint32_t);
  size_t bytes_emb   = vocab_size * hidden_dim * sizeof(__bf16);
  size_t bytes_out   = total_tokens * hidden_dim * sizeof(__bf16);

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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateEmbeddingFwd];
    [e__ setBuffer:bufIds  offset:0 atIndex:0];
    [e__ setBuffer:bufEmb  offset:0 atIndex:1];
    [e__ setBuffer:bufOut  offset:0 atIndex:2];
    [e__ setBytes:&u_hidden length:sizeof(uint32_t) atIndex:3];
    [e__ setBytes:&u_tokens length:sizeof(uint32_t) atIndex:4];

    MTLSize threads = MTLSizeMake(256, 1, 1);
    MTLSize grid    = MTLSizeMake((n_threads + 255) / 256, 1, 1);
    [e__ dispatchThreadgroups:grid threadsPerThreadgroup:threads];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateEmbeddingFwd];
      [e__ setBuffer:bufIds  offset:0 atIndex:0];
      [e__ setBuffer:bufEmb  offset:0 atIndex:1];
      [e__ setBuffer:bufOut  offset:0 atIndex:2];
      [e__ setBytes:&u_hidden length:sizeof(uint32_t) atIndex:3];
      [e__ setBytes:&u_tokens length:sizeof(uint32_t) atIndex:4];
      MTLSize threads = MTLSizeMake(256, 1, 1);
      MTLSize grid    = MTLSizeMake((n_threads + 255) / 256, 1, 1);
      [e__ dispatchThreadgroups:grid threadsPerThreadgroup:threads];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void cross_entropy(const void *logits, const uint32_t *targets, float *loss_out,
                   float *grad_logits, size_t total_tokens, size_t vocab_size) {
  if (total_tokens == 0 || vocab_size == 0) return;

  size_t bytes_logits = total_tokens * vocab_size * sizeof(__bf16);
  size_t bytes_targets = total_tokens * sizeof(uint32_t);
  size_t bytes_loss = sizeof(__bf16);

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

  // Explicitly zero the loss buffer — get_or_create_buffer for persistent
  // writes does NOT zero on reuse (only on first creation).  Without this,
  // the GPU's atomic_fetch_add would accumulate on top of stale loss data.
  memset([bufLoss contents], 0, bytes_loss);


  uint32_t u_vocab  = static_cast<uint32_t>(vocab_size);
  uint32_t u_tokens = static_cast<uint32_t>(total_tokens);

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateCrossEntropy];
    [e__ setBuffer:bufLogits  offset:0 atIndex:0];
    [e__ setBuffer:bufTargets offset:0 atIndex:1];
    [e__ setBuffer:bufLoss    offset:0 atIndex:2];
    [e__ setBuffer:bufGrad    offset:0 atIndex:3];
    [e__ setBytes:&u_vocab  length:sizeof(uint32_t) atIndex:4];
    [e__ setBytes:&u_tokens length:sizeof(uint32_t) atIndex:5];

    MTLSize threadsPerTG = MTLSizeMake(256, 1, 1);
    MTLSize tgGrid       = MTLSizeMake(1, total_tokens, 1);
    [e__ dispatchThreadgroups:tgGrid threadsPerThreadgroup:threadsPerTG];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];

      [e__ setComputePipelineState:pipelineStateCrossEntropy];
      [e__ setBuffer:bufLogits  offset:0 atIndex:0];
      [e__ setBuffer:bufTargets offset:0 atIndex:1];
      [e__ setBuffer:bufLoss    offset:0 atIndex:2];
      [e__ setBuffer:bufGrad    offset:0 atIndex:3];
      [e__ setBytes:&u_vocab  length:sizeof(uint32_t) atIndex:4];
      [e__ setBytes:&u_tokens length:sizeof(uint32_t) atIndex:5];

      MTLSize threadsPerTG = MTLSizeMake(256, 1, 1);
      MTLSize tgGrid       = MTLSizeMake(1, total_tokens, 1);
      [e__ dispatchThreadgroups:tgGrid threadsPerThreadgroup:threadsPerTG];

      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
    }
  }

  // IN BATCH MODE: GPU hasn't executed yet.  Do NOT write to *loss_out —
  // that would store 0.  The caller reads get_last_loss() after end_scope().
  // IN SYNC MODE: GPU already executed (waitUntilCompleted + run_copy_back_tasks
  // above), so last_step_loss is valid.
  // loss is read at end_scope — no inline write needed
  if (false) {
  }
}

void gemm_backward(const float *a_transposed, const float *b, float *c,
                   size_t M, size_t N, size_t K) {
  if (M == 0 || N == 0 || K == 0) return;
  memset(c, 0, M * N * sizeof(__bf16));
  metal_bridge::gemm_bf16(a_transposed, b, c, M, N, K, true, false);
}

void gemm_proj_trans_b(const float *a, const float *b_transposed, float *c,
                       size_t M, size_t N, size_t K) {
  if (M == 0 || N == 0 || K == 0) return;
  metal_bridge::gemm_bf16(a, b_transposed, c, M, N, K, false, true);
}

void reshape_to_4d(const float *src, float *dst,
                   size_t batch, size_t n_heads, size_t seq_len, size_t head_dim) {
  size_t total = batch * n_heads * seq_len * head_dim;
  if (total == 0) return;

  size_t bytesSrc = (batch * seq_len * n_heads * head_dim) * sizeof(__bf16);
  size_t bytesDst = total * sizeof(__bf16);

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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateReshape4D];
    [e__ setBuffer:bufSrc offset:0 atIndex:0];
    [e__ setBuffer:bufDst offset:0 atIndex:1];
    [e__ setBytes:&u_batch   length:sizeof(uint) atIndex:2];
    [e__ setBytes:&u_n_heads length:sizeof(uint) atIndex:3];
    [e__ setBytes:&u_seq_len length:sizeof(uint) atIndex:4];
    [e__ setBytes:&u_hd      length:sizeof(uint) atIndex:5];
    MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);
    [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateReshape4D];
      [e__ setBuffer:bufSrc offset:0 atIndex:0];
      [e__ setBuffer:bufDst offset:0 atIndex:1];
      [e__ setBytes:&u_batch   length:sizeof(uint) atIndex:2];
      [e__ setBytes:&u_n_heads length:sizeof(uint) atIndex:3];
      [e__ setBytes:&u_seq_len length:sizeof(uint) atIndex:4];
      [e__ setBytes:&u_hd      length:sizeof(uint) atIndex:5];
      MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
      MTLSize tg   = MTLSizeMake(256, 1, 1);
      [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void reshape_to_3d(const float *src, float *dst,
                   size_t batch, size_t n_heads, size_t seq_len, size_t head_dim) {
  size_t total = batch * n_heads * seq_len * head_dim;
  if (total == 0) return;

  size_t bytesSrc = total * sizeof(__bf16);
  size_t bytesDst = (batch * seq_len * n_heads * head_dim) * sizeof(__bf16);

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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateReshape3D];
    [e__ setBuffer:bufSrc offset:0 atIndex:0];
    [e__ setBuffer:bufDst offset:0 atIndex:1];
    [e__ setBytes:&u_batch   length:sizeof(uint) atIndex:2];
    [e__ setBytes:&u_n_heads length:sizeof(uint) atIndex:3];
    [e__ setBytes:&u_seq_len length:sizeof(uint) atIndex:4];
    [e__ setBytes:&u_hd      length:sizeof(uint) atIndex:5];
    MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);
    [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateReshape3D];
      [e__ setBuffer:bufSrc offset:0 atIndex:0];
      [e__ setBuffer:bufDst offset:0 atIndex:1];
      [e__ setBytes:&u_batch   length:sizeof(uint) atIndex:2];
      [e__ setBytes:&u_n_heads length:sizeof(uint) atIndex:3];
      [e__ setBytes:&u_seq_len length:sizeof(uint) atIndex:4];
      [e__ setBytes:&u_hd      length:sizeof(uint) atIndex:5];
      MTLSize grid = MTLSizeMake((total + 255) / 256, 1, 1);
      MTLSize tg   = MTLSizeMake(256, 1, 1);
      [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void rope_forward(float *q, float *k,
                  const float *cos_table, const float *sin_table,
                  size_t batch, size_t q_heads, size_t kv_heads,
                  size_t seq_len, size_t head_dim) {
  size_t half_dim = head_dim / 2;

  size_t bytesQ  = batch * q_heads * seq_len * head_dim * sizeof(__bf16);
  size_t bytesK  = batch * kv_heads * seq_len * head_dim * sizeof(__bf16);
  size_t bytesT  = seq_len * half_dim * sizeof(__bf16);

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

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLComputeCommandEncoder> e__ = [cb__ computeCommandEncoder];
    [e__ setComputePipelineState:pipelineStateRoPEForward];
    [e__ setBuffer:bufQ   offset:0 atIndex:0];
    [e__ setBuffer:bufK   offset:0 atIndex:1];
    [e__ setBuffer:bufCos offset:0 atIndex:2];
    [e__ setBuffer:bufSin offset:0 atIndex:3];
    [e__ setBytes:&u_batch   length:sizeof(uint) atIndex:4];
    [e__ setBytes:&u_qheads  length:sizeof(uint) atIndex:5];
    [e__ setBytes:&u_kvheads length:sizeof(uint) atIndex:6];
    [e__ setBytes:&u_seqlen  length:sizeof(uint) atIndex:7];
    [e__ setBytes:&u_hd      length:sizeof(uint) atIndex:8];
    MTLSize grid = MTLSizeMake((total_pairs + 255) / 256, 1, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);
    [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [e__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb2__ = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> e__ = [cb2__ computeCommandEncoder];
      [e__ setComputePipelineState:pipelineStateRoPEForward];
      [e__ setBuffer:bufQ   offset:0 atIndex:0];
      [e__ setBuffer:bufK   offset:0 atIndex:1];
      [e__ setBuffer:bufCos offset:0 atIndex:2];
      [e__ setBuffer:bufSin offset:0 atIndex:3];
      [e__ setBytes:&u_batch   length:sizeof(uint) atIndex:4];
      [e__ setBytes:&u_qheads  length:sizeof(uint) atIndex:5];
      [e__ setBytes:&u_kvheads length:sizeof(uint) atIndex:6];
      [e__ setBytes:&u_seqlen  length:sizeof(uint) atIndex:7];
      [e__ setBytes:&u_hd      length:sizeof(uint) atIndex:8];
      MTLSize grid = MTLSizeMake((total_pairs + 255) / 256, 1, 1);
      MTLSize tg   = MTLSizeMake(256, 1, 1);
      [e__ dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [e__ endEncoding];
      [cb2__ commit];
      // no FIFO tracking — single command buffer
      [cb2__ waitUntilCompleted];
      run_copy_back_tasks();
    }
  }
}

void reconcile_buffers() {
  run_copy_back_tasks();
}

void copy_buffer_async(float *dst, size_t dst_bytes, const float *src, size_t src_bytes) {
  if (!dst || !src || dst_bytes == 0) return;
  id<MTLBuffer> bufDst = get_or_create_buffer(dst, dst_bytes, true, false, false);
  id<MTLBuffer> bufSrc = get_or_create_buffer(src, src_bytes, false, false, false);
  if (!bufDst || !bufSrc) { std::cerr << "[copy_buffer] buffer alloc failed\n"; return; }

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLBlitCommandEncoder> blit__ = [cb__ blitCommandEncoder];
    [blit__ copyFromBuffer:bufSrc sourceOffset:0 toBuffer:bufDst destinationOffset:0 size:std::min(dst_bytes, src_bytes)];
    [blit__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromBuffer:bufSrc sourceOffset:0 toBuffer:bufDst destinationOffset:0 size:std::min(dst_bytes, src_bytes)];
      [blit endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }
}

void fill_zero_async(float *data, size_t bytes) {
  if (data == nullptr || bytes == 0) return;
  id<MTLBuffer> buf = get_or_create_buffer(data, bytes, true, false, false);
  if (!buf) { std::cerr << "[fill_zero] buffer alloc failed\n"; return; }

  id<MTLCommandBuffer> cb__ = batchCommandBuffer;
  if (cb__) {
    id<MTLBlitCommandEncoder> blit__ = [cb__ blitCommandEncoder];
    [blit__ fillBuffer:buf range:NSMakeRange(0, bytes) value:0];
    [blit__ endEncoding];
  } else {
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit fillBuffer:buf range:NSMakeRange(0, bytes) value:0];
      [blit endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }
}

void begin_scope() {
  // Drain any leftover pool from a failed scope
  if (stepPool != nil) { [stepPool drain]; stepPool = nil; }
  stepPool = [[NSAutoreleasePool alloc] init];

  // ── Single command buffer for the entire step ──
  // Each dispatch function creates a new encoder from this buffer, encodes one
  // kernel, and immediately ends the encoder.  At end_scope we commit once.
  if (batchCommandBuffer != nil) { CFRelease((__bridge CFTypeRef)batchCommandBuffer); }
  batchCommandBuffer = [commandQueue commandBuffer];
  if (batchCommandBuffer) CFRetain((__bridge CFTypeRef)batchCommandBuffer);
}

void end_scope() {
  if (batchCommandBuffer == nil) {
    if (stepPool != nil) { [stepPool drain]; stepPool = nil; }
    return;
  }

  std::cout << "[GPU-EXEC] Synchronizing... waiting for last buffer..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  // Single commit + wait — one XNU syscall for ALL 386 kernels
  [batchCommandBuffer commit];
  [batchCommandBuffer waitUntilCompleted];
  CFRelease((__bridge CFTypeRef)batchCommandBuffer);
  batchCommandBuffer = nil;

  auto end = std::chrono::high_resolution_clock::now();
  double gpu_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
  accum_gpu_time_ms += gpu_time;
  count_gpu_calls++;

  std::cout << "[GPU-EXEC] GPU Hardware finished execution in " << gpu_time << " ms." << std::endl;

  // Read loss from the persistent GPU buffer (coherent after waitUntilCompleted)
  id<MTLBuffer> loss_read_buf = get_or_create_buffer(&last_step_loss, sizeof(__bf16), true, false, false);
  if (loss_read_buf) {
    last_step_loss = *(const float*)[loss_read_buf contents];
  }
  run_copy_back_tasks();

  // Return size-pool buffers for reuse
  for (const auto &pair : activeAllocationsThisStep)
    sizePool[pair.first].push_back(pair.second);
  activeAllocationsThisStep.clear();

  // Drain the step-level autorelease pool
  if (stepPool != nil) { [stepPool drain]; stepPool = nil; }
}

// Backward compat aliases — don't call these directly (use begin_scope/end_scope)

float get_last_loss() {
  id<MTLBuffer> loss_read_buf = get_or_create_buffer(&last_step_loss, sizeof(__bf16), true, false, false);
  if (loss_read_buf) {
    last_step_loss = *(const float*)[loss_read_buf contents];
  } else {
    fprintf(stderr, "[LOSS] loss buffer not found, returning %f\n", last_step_loss);
  }
  return last_step_loss;
}

 void execute_in_autoreleasepool(std::function<void()> func) {
   @autoreleasepool {
     func();
   }
 }

 // ── Programmatic GPU trace capture ─────────────────────────────────────────
 // Set CAPTURE_METAL_TRACE=1 to generate step_profile.gputrace for Xcode.
 // The trace captures one full training step (begin_scope → end_scope).

 // Persistent capture state
 static MTLCaptureManager* g_captureManager = nil;
 static bool g_trace_active = false;

 static bool is_trace_requested() {
   static bool checked = false;
   static bool enabled = false;
   if (!checked) {
     checked = true;
     const char* env = getenv("CAPTURE_METAL_TRACE");
     enabled = env && env[0] == '1';
   }
   return enabled;
 }

 void start_step_trace() {
   if (!is_trace_requested()) return;
   if (g_trace_active) return;
   if (!device) { std::cerr << "[TRACE] device not initialized\n"; return; }

   g_captureManager = [MTLCaptureManager sharedCaptureManager];
   if (!g_captureManager) { std::cerr << "[TRACE] sharedCaptureManager is nil\n"; return; }

   MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
   desc.captureObject = device;
   desc.destination = MTLCaptureDestinationGPUTraceDocument;

   NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
   NSString* path = [cwd stringByAppendingPathComponent:@"step_profile.gputrace"];
   desc.outputURL = [NSURL fileURLWithPath:path];

   NSError* error = nil;
   BOOL ok = [g_captureManager startCaptureWithDescriptor:desc error:&error];
   [desc release];

   if (!ok) {
     std::cerr << "[TRACE] Failed to start GPU trace: "
               << (error ? [[error localizedDescription] UTF8String] : "unknown")
               << std::endl;
     g_trace_active = false;
     return;
   }

   g_trace_active = true;
   std::cout << "[TRACE] Capturing to " << [path UTF8String] << std::endl;
 }

 void stop_step_trace() {
   if (!g_trace_active) return;
   [g_captureManager stopCapture];
   g_trace_active = false;
   std::cout << "[TRACE] Capture saved to step_profile.gputrace" << std::endl;
 }

 } // namespace metal_bridge
