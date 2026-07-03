#include <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdlib>
#include <iostream>

id<MTLBuffer> createBufferWithBytesWithoutCopy(id<MTLDevice> device,
                                              float *host_ptr,
                                              size_t size_in_bytes) {
  id<MTLBuffer> buffer = [device
      newBufferWithBytesNoCopy:host_ptr
                        length:size_in_bytes
                       options:MTLResourceStorageModeShared
                   deallocator:^(void *pointer, NSUInteger length) {
                     fprintf(stderr, "Buffer deallocated %p\n", pointer);
                     free(pointer);
                   }];

  return buffer;
}

int main() {
  @autoreleasepool {
    NSError *error = nil;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    NSURL *shaderURL = [NSURL fileURLWithPath:@"build/05_vector_add.metallib"];
    id<MTLLibrary> library = [device newLibraryWithURL:shaderURL error:&error];

    if (library == nil) {
      NSString *localisedError = error.localizedDescription;
      const char *errorString = localisedError.UTF8String;
      fprintf(stderr, "Failed to load library: %s\n", errorString);
      return 1;
    }

    id<MTLFunction> vectorAdd = [library newFunctionWithName:@"vector_add"];
    if (vectorAdd == nil) {
      fprintf(stderr, "Failed to find vector_add function\n");
      return 1;
    }

    id<MTLComputePipelineState> pipeLineState =
        [device newComputePipelineStateWithFunction:vectorAdd error:&error];

    if (pipeLineState == nil) {
      NSString *localisedError = error.localizedDescription;
      const char *errorString = localisedError.UTF8String;
      fprintf(stderr, "Failed to create compute pipeline state: %s\n", errorString);
      return 1;
    }

    const size_t num_elements = 1003520; // Aligned to 16 KB page boundary for floats
    const size_t bytes = sizeof(float) * num_elements;

    float *host_A = nullptr;
    float *host_B = nullptr;
    float *host_C = nullptr;

    int allocate_A = posix_memalign((void **)&host_A, 16384, bytes);
    int allocate_B = posix_memalign((void **)&host_B, 16384, bytes);
    int allocate_C = posix_memalign((void **)&host_C, 16384, bytes);

    if (allocate_A != 0 || allocate_B != 0 || allocate_C != 0) {
      fprintf(stderr, "Failed to allocate memory for host arrays\n");
      return 1;
    }

    for (size_t i = 0; i < num_elements; i++) {
      host_A[i] = (float)i;
      host_B[i] = (float)i * 1.5f;
      host_C[i] = 0.0f;
    }

    id<MTLBuffer> A_buffer = createBufferWithBytesWithoutCopy(device, host_A, bytes);
    id<MTLBuffer> B_buffer = createBufferWithBytesWithoutCopy(device, host_B, bytes);
    id<MTLBuffer> C_buffer = createBufferWithBytesWithoutCopy(device, host_C, bytes);

    if (A_buffer == nil || B_buffer == nil || C_buffer == nil) {
      fprintf(stderr, "Failed to wrap host pointers in Metal buffers\n");
      return 1;
    }

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeLineState];
    [encoder setBuffer:A_buffer offset:0 atIndex:0];
    [encoder setBuffer:B_buffer offset:0 atIndex:1];
    [encoder setBuffer:C_buffer offset:0 atIndex:2];

    uint32_t size = num_elements;
    [encoder setBytes:&size length:sizeof(size) atIndex:3];

    MTLSize gridSize = MTLSizeMake(num_elements, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(pipeLineState.threadExecutionWidth, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Verification
    for (size_t i = 0; i < num_elements; i++) {
      float expected = host_A[i] + host_B[i];
      if (std::abs(host_C[i] - expected) > 1e-5) {
        fprintf(stderr, "Error at index %zu: Expected %f, got %f\n", i, expected, host_C[i]);
        return 1;
      }
    }

    fprintf(stdout, "Vector Add Verification Successful!!\n");
  }
  return 0;
}
