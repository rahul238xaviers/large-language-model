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

static bool initialized = false;

static id<MTLCommandBuffer> activeCmdBuffer = nil;
static id<MTLComputeCommandEncoder> activeEncoder = nil;
static bool batchActive = false;

// Address-to-Buffer cache mapping raw host pointers to persistent MTLBuffers
static std::unordered_map<const void*, id<MTLBuffer>> bufferCache;

static id<MTLBuffer> get_or_create_buffer(const void* ptr, size_t bytes);

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

  initialized = true;
}

void gemm_ffn(const float *a, const float *b, float *c, size_t M, size_t N,
              size_t K) {
  if (M == 0 || N == 0 || K == 0)
    return;

  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  id<MTLBuffer> bufferA = get_or_create_buffer(a, bytesA);
  id<MTLBuffer> bufferB = get_or_create_buffer(b, bytesB);
  id<MTLBuffer> bufferC = get_or_create_buffer(c, bytesC);

  if (!bufferA || !bufferB || !bufferC) {
    std::cerr << "Failed to allocate Metal GEMM buffers!" << std::endl;
    return;
  }

  simd::uint3 dimensions = {(uint32_t)M, (uint32_t)N, (uint32_t)K};

  if (batchActive) {
    [activeEncoder setComputePipelineState:pipelineStateFFN];
    [activeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [activeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [activeEncoder setBuffer:bufferC offset:0 atIndex:2];
    [activeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];

    MTLSize threadgroupsPerGrid =
        MTLSizeMake((N + 127) / 128, (M + 127) / 128, 1);
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
      [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
      [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
      [computeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];

      MTLSize threadgroupsPerGrid =
          MTLSizeMake((N + 127) / 128, (M + 127) / 128, 1);

      MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];

      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
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
  id<MTLBuffer> bufferC = get_or_create_buffer(c, bytesC);

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
      [computeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];

      MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
      MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);

      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                     threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];

      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
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
  id<MTLBuffer> bufferOut = get_or_create_buffer(out_gqa, bytesOut);

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
    }
  }
}
static id<MTLBuffer> get_or_create_buffer(const void* ptr, size_t bytes) {
  if (ptr == nullptr || bytes == 0) return nil;
  
  auto it = bufferCache.find(ptr);
  if (it != bufferCache.end()) {
    return it->second;
  }
  
  id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:(void*)ptr
                                                length:bytes
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
  if (buf) {
    bufferCache[ptr] = buf;
  }
  return buf;
}

void start_batch() {
  if (batchActive) return;
  activeCmdBuffer = [commandQueue commandBuffer];
  activeEncoder = [activeCmdBuffer computeCommandEncoder];
  batchActive = true;
}

void commit_batch() {
  if (!batchActive) return;
  [activeEncoder endEncoding];
  activeEncoder = nil;
  
  [activeCmdBuffer commit];
  [activeCmdBuffer waitUntilCompleted];
  activeCmdBuffer = nil;
  
  batchActive = false;
  bufferCache.clear();
}

} // namespace metal_bridge
