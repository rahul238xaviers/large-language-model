/**
 * @file MetalBridge.mm
 * @brief Implementation of the C++ to Metal GPU runtime bridge.
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Orchestrates GPU memory buffers creation, manages pipeline compute states,
 * and submits execution dispatches to Apple Silicon hardware (using MPS matrix cores).
 */

#include "gpu_kernel/MetalBridge.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <simd/simd.h>
#include <cstddef>
#include <iostream>

namespace metal_bridge {

// device (MTLDevice): Represents the physical GPU hardware.
// Think of it as the parent manager of all GPU memory and pipeline allocations.
static id<MTLDevice> device = nil;

// commandQueue (MTLCommandQueue): A thread-safe FIFO (first-in-first-out)
// queue. This is the conveyor belt that carries instruction packets (command
// buffers) from CPU to GPU.
static id<MTLCommandQueue> commandQueue = nil;



// pipelineStateFFN (MTLComputePipelineState): The compiled microcode for our tiled FFN GEMM.
static id<MTLComputePipelineState> pipelineStateFFN = nil;

static bool initialized = false;

// Profiling statistics definitions
double accum_gpu_time_ms = 0.0;
double accum_cpu_time_ms = 0.0;
size_t count_gpu_calls = 0;
size_t count_cpu_calls = 0;

/**
 * @brief Check if the GPU device and compile pipeline states are fully loaded and operational.
 *
 * @return true If available.
 * @return false Otherwise.
 */
bool is_available() {
  return (device != nil) && (pipelineStateFFN != nil);
}

/**
 * @brief Reset accumulated GPU and CPU execution profiling timers and counts to 0.
 */
void reset_profile_stats() {
  accum_gpu_time_ms = 0.0;
  accum_cpu_time_ms = 0.0;
  count_gpu_calls = 0;
  count_cpu_calls = 0;
}

/**
 * @brief Searches for the Apple GPU device and compiles the Metal shader libraries.
 */
void initialize() {
  if (initialized) return;
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
  initialized = true;
}



void gemm_ffn(const float *a, const float *b, float *c, size_t M, size_t N, size_t K) {
  // WHAT: Safety guard checking if any dimension is zero.
  // WHY: A matrix multiplication with size 0 has no work to do, and would cause division/allocation errors.
  if (M == 0 || N == 0 || K == 0) return;

  // WHAT: Calculates the byte sizes of our three matrices.
  // WHY: Zero-copy buffer allocation operates on raw byte counts.
  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  // WHAT: Creates a zero-copy pointer to CPU Matrix A.
  // WHY: Unified memory allows the GPU to read A directly from CPU RAM without copying.
  id<MTLBuffer> bufferA = [device newBufferWithBytesNoCopy:(void *)a length:bytesA options:MTLResourceStorageModeShared deallocator:nil];

  // WHAT: Creates a zero-copy pointer to CPU Matrix B.
  // WHY: Unified memory allows the GPU to read B directly from CPU RAM without copying.
  id<MTLBuffer> bufferB = [device newBufferWithBytesNoCopy:(void *)b length:bytesB options:MTLResourceStorageModeShared deallocator:nil];

  // WHAT: Creates a zero-copy pointer to CPU Matrix C.
  // WHY: Unified memory allows the GPU to write results directly into CPU RAM.
  id<MTLBuffer> bufferC = [device newBufferWithBytesNoCopy:(void *)c length:bytesC options:MTLResourceStorageModeShared deallocator:nil];

  // WHAT: Validates all buffer allocations succeeded.
  // WHY: To prevent accessing null references, which crashes the GPU.
  if (!bufferA || !bufferB || !bufferC) {
    std::cerr << "Failed to allocate Metal GEMM buffers!" << std::endl;
    return;
  }

  // WHAT: Execute MPS Matrix Multiplication within an autorelease pool.
  // WHY: Metal Performance Shaders are objective-C classes that allocate temporary wrappers. Wrapping
  //      them in @autoreleasepool ensures they are immediately freed, avoiding memory growth.
  @autoreleasepool {
    // WHAT: Creates a command buffer packet (cmdBuffer) for the GPU.
    // WHY: To bundle all GEMM configuration commands into a single transmission.
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];

    // WHAT: Defines the shape and row stride of Matrix A.
    // WHY: Tells MPS the row-major format and byte offsets for reading input A.
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                      columns:K
                                                                     rowBytes:K * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
    
    // WHAT: Defines the shape and row stride of Matrix B.
    // WHY: Tells MPS the row-major format and byte offsets for reading input B.
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                      columns:N
                                                                     rowBytes:N * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
    
    // WHAT: Defines the shape and row stride of Matrix C.
    // WHY: Tells MPS where and how to write the output matrix C.
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                      columns:N
                                                                     rowBytes:N * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];

    // WHAT: Instantiates an MPSMatrix wrapper around bufferA.
    // WHY: Connects the raw Metal buffer data to the MPS matrix math object.
    MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];

    // WHAT: Instantiates an MPSMatrix wrapper around bufferB.
    // WHY: Connects the raw Metal buffer data to the MPS matrix math object.
    MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];

    // WHAT: Instantiates an MPSMatrix wrapper around bufferC.
    // WHY: Connects the raw Metal buffer data to the MPS matrix math object.
    MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

    // WHAT: Instantiates the high-performance MPS Matrix Multiplication kernel.
    // WHY: Instructs Apple's hardware matrix engines to perform C = A @ B.
    MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                        transposeLeft:NO
                                                                       transposeRight:NO
                                                                           resultRows:M
                                                                        resultColumns:N
                                                                      interiorColumns:K
                                                                                alpha:1.0f
                                                                                 beta:0.0f];

    // WHAT: Encodes the matrix multiplication execution command into the command buffer.
    // WHY: Queues the task to be run by the GPU shader cores.
    [matmul encodeToCommandBuffer:cmdBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];

    // WHAT: Submits the command buffer packet to the commandQueue conveyor belt.
    // WHY: Triggers the GPU hardware to execute the tasks.
    [cmdBuffer commit];

    // WHAT: Blocks the C++ thread until the GPU finishes calculating the GEMM.
    // WHY: Guarantees the output matrix C contains the results before we read it on the CPU.
    [cmdBuffer waitUntilCompleted];
  }
}

} // namespace metal_bridge
