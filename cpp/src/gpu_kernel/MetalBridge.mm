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

static bool initialized = false;

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
         (pipelineStateProj != nil);
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

  initialized = true;
}

void gemm_ffn(const float *a, const float *b, float *c, size_t M, size_t N,
              size_t K) {
  // WHAT: Safety guard checking if any dimension is zero.
  // WHY: A matrix multiplication with size 0 has no work to do, and would cause
  // division/allocation errors.
  if (M == 0 || N == 0 || K == 0)
    return;

  // WHAT: Calculates the byte sizes of our three matrices.
  // WHY: Zero-copy buffer allocation operates on raw byte counts.
  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  // WHAT: Creates a zero-copy pointer to CPU Matrix A.
  // WHY: Unified memory allows the GPU to read A directly from CPU RAM without
  // copying.
  id<MTLBuffer> bufferA =
      [device newBufferWithBytesNoCopy:(void *)a
                                length:bytesA
                               options:MTLResourceStorageModeShared
                           deallocator:nil];

  // WHAT: Creates a zero-copy pointer to CPU Matrix B.
  // WHY: Unified memory allows the GPU to read B directly from CPU RAM without
  // copying.
  id<MTLBuffer> bufferB =
      [device newBufferWithBytesNoCopy:(void *)b
                                length:bytesB
                               options:MTLResourceStorageModeShared
                           deallocator:nil];

  // WHAT: Creates a zero-copy pointer to CPU Matrix C.
  // WHY: Unified memory allows the GPU to write results directly into CPU RAM.
  id<MTLBuffer> bufferC =
      [device newBufferWithBytesNoCopy:(void *)c
                                length:bytesC
                               options:MTLResourceStorageModeShared
                           deallocator:nil];

  // WHAT: Validates all buffer allocations succeeded.
  // WHY: To prevent accessing null references, which crashes the GPU.
  if (!bufferA || !bufferB || !bufferC) {
    std::cerr << "Failed to allocate Metal GEMM buffers!" << std::endl;
    return;
  }

  // WHAT: Execute MPS Matrix Multiplication within an autorelease pool.
  // WHY: Metal Performance Shaders are objective-C classes that allocate
  // temporary wrappers. Wrapping
  //      them in @autoreleasepool ensures they are immediately freed, avoiding
  //      memory growth.
  @autoreleasepool {
    // WHAT: Creates a command buffer packet (cmdBuffer) for the GPU.
    // WHY: To bundle all GEMM configuration commands into a single
    // transmission.
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];

    simd::uint3 dimensions = {(uint32_t)M, (uint32_t)N, (uint32_t)K};

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

    // WHAT: Submits the command buffer packet to the commandQueue conveyor
    // belt. WHY: Triggers the GPU hardware to execute the tasks.
    [cmdBuffer commit];

    // WHAT: Blocks the C++ thread until the GPU finishes calculating the GEMM.
    // WHY: Guarantees the output matrix C contains the results before we read
    // it on the CPU.
    [cmdBuffer waitUntilCompleted];
  }
}

void gemm_proj(const float *a, const float *b, float *c, size_t M, size_t N,
               size_t K) {
  // WHAT: Safety guard checking if any dimension is zero.
  // WHY: A matrix multiplication with size 0 has no work to do, and would cause
  // division/allocation errors.
  if (M == 0 || N == 0 || K == 0)
    return;

  // WHAT: Calculates the byte sizes of our three matrices.
  // WHY: Zero-copy buffer allocation operates on raw byte counts.
  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  // WHAT: Creates a zero-copy pointer to CPU Matrix A.
  // WHY: Unified memory allows the GPU to read A directly from CPU RAM without
  // copying.
  id<MTLBuffer> bufferA =
      [device newBufferWithBytesNoCopy:(void *)a
                                length:bytesA
                               options:MTLResourceStorageModeShared
                           deallocator:nil];
  // WHAT: Creates a zero-copy pointer to CPU Matrix B.
  // WHY: Unified memory allows the GPU to read B directly from CPU RAM without
  // copying.
  id<MTLBuffer> bufferB =
      [device newBufferWithBytesNoCopy:(void *)b
                                length:bytesB
                               options:MTLResourceStorageModeShared
                           deallocator:nil];
  // WHAT: Creates a zero-copy pointer to CPU Matrix C.
  // WHY: Unified memory allows the GPU to write results directly into CPU RAM.
  id<MTLBuffer> bufferC =
      [device newBufferWithBytesNoCopy:(void *)c
                                length:bytesC
                               options:MTLResourceStorageModeShared
                           deallocator:nil];

  // WHAT: Checks if buffer allocation failed.
  // WHY: If the GPU runs out of memory, buffers will be nil, and we must abort
  // gracefully.
  if (!bufferA || !bufferB || !bufferC) {
    std::cerr << "Failed to allocate Metal buffers for gemm_proj!" << std::endl;
    return;
  }

  // WHAT: Packages the matrix dimensions into a Metal-compatible struct.
  // WHY: The MSL kernel expects dimensions as a SIMD vector for efficient
  // thread indexing.
  simd::uint3 dimensions = {(uint32_t)M, (uint32_t)N, (uint32_t)K};

  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder =
        [cmdBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pipelineStateProj];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
    [computeEncoder setBytes:&dimensions length:sizeof(simd::uint3) atIndex:3];

    // WHAT: Calculates the number of threadgroup blocks needed to cover the
    // output matrix.
    // WHY: We divide the total rows (M) and columns (N) by our tile size (64)
    // to determine the grid dimensions.
    MTLSize threadgroupsPerGrid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);

    // WHAT: Defines the computational unit: a block of 64 threads working in
    // lockstep.
    // WHY: Hardcoded to 64 to match the MSL kernel's SIMDwidth (32) plus
    // its parallel SIMDgroup structure (2 groups of 32).
    MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);

    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];

    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
  }
}

} // namespace metal_bridge
