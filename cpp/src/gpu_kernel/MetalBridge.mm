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

static bool initialized = false;

static id<MTLCommandBuffer> activeCmdBuffer = nil;
static id<MTLComputeCommandEncoder> activeEncoder = nil;
static bool batchActive = false;

// Address-to-Buffer cache mapping raw host pointers to persistent MTLBuffers
static std::unordered_map<const void *, id<MTLBuffer>> bufferCache;

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
                                          bool is_write = false);

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
  id<MTLBuffer> bufferB_gate = get_or_create_buffer(b_gate, bytesB);
  id<MTLBuffer> bufferB_up = get_or_create_buffer(b_up, bytesB);
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
  id<MTLBuffer> bufferB = get_or_create_buffer(b, bytesB);
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
                                          bool is_write) {
  if (ptr == nullptr || bytes == 0)
    return nil;

  auto it = bufferCache.find(ptr);
  if (it != bufferCache.end()) {
    return it->second;
  }

  id<MTLBuffer> buf = nil;

  // Check if both pointer and size are 16KB page-aligned
  bool is_aligned = (((uintptr_t)ptr % 16384) == 0) && ((bytes % 16384) == 0);

  if (is_aligned) {
    buf = [device newBufferWithBytesNoCopy:(void *)ptr
                                    length:bytes
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
  }

  // If not aligned, or if NoCopy allocation failed, fall back to copy
  // allocation
  if (!buf) {
    buf = [device newBufferWithBytes:(void *)ptr
                              length:bytes
                             options:MTLResourceStorageModeShared];

    // If it was copy-allocated and we intend to write to it, register a
    // copy-back task
    if (buf && is_write) {
      copyBackQueue.push_back({(void *)ptr, buf, bytes});
    }
  }

  if (buf) {
    bufferCache[ptr] = buf;
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
  id<MTLBuffer> bufferW = get_or_create_buffer(weight, bytesW);

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
  id<MTLBuffer> bufferWeight = get_or_create_buffer(weight, bytesW);
  id<MTLBuffer> bufferGradInput =
      get_or_create_buffer(grad_input, bytesIn, true);
  id<MTLBuffer> bufferGradWeight =
      get_or_create_buffer(grad_weight, bytesW, true);

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
  id<MTLBuffer> bufferCos = get_or_create_buffer(cos_table, bytesTable);
  id<MTLBuffer> bufferSin = get_or_create_buffer(sin_table, bytesTable);

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

  id<MTLBuffer> bufferParam = get_or_create_buffer(param, bytes, true);
  id<MTLBuffer> bufferGrad = get_or_create_buffer(grad, bytes);
  id<MTLBuffer> bufferM = get_or_create_buffer(m, bytes, true);
  id<MTLBuffer> bufferV = get_or_create_buffer(v, bytes, true);

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

void start_batch() {
  if (batchActive)
    return;
  activeCmdBuffer = [commandQueue commandBuffer];
  activeEncoder = [activeCmdBuffer computeCommandEncoder];
  batchActive = true;
}

void commit_batch() {
  if (!batchActive)
    return;
  [activeEncoder endEncoding];
  activeEncoder = nil;

  [activeCmdBuffer commit];
  [activeCmdBuffer waitUntilCompleted];
  activeCmdBuffer = nil;

  // Run copy-back tasks for unaligned writes
  run_copy_back_tasks();

  batchActive = false;
  bufferCache.clear();
}

} // namespace metal_bridge
