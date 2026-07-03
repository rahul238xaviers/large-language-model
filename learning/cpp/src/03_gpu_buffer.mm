#import <Metal/Metal.h>
#include <cstdlib>
#include <iostream>

int main() {

  @autoreleasepool {

    void *my_ptr = nullptr;

    int success = posix_memalign(&my_ptr, 16384, 32768);

    if (success == 0) {
      std::cerr << "Allocation passed" << std::endl;
    }
    std::cout << "Memory address: " << my_ptr << std::endl;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    id<MTLBuffer> allocatedBuffer = [device
        newBufferWithBytesNoCopy:my_ptr
                          length:32768
                         options:MTLResourceStorageModeShared
                     deallocator:^(void *_Nonnull pointer, NSUInteger length) {
                       std::cout << "The deallocator block executed"
                                 << std::endl;
                       if (pointer == my_ptr) {
                         std::cerr
                             << "Test successfull the pointer object has same "
                                "memory address "
                             << my_ptr << std::endl;
                         free(my_ptr);
                       }
                     }];

    std::cout << "Allocated buffer: " << allocatedBuffer << std::endl;

    void *gpu_buffer_pointer = [allocatedBuffer contents];

    std::cout << "GPU Buffer Pointer: " << gpu_buffer_pointer << std::endl;
    [allocatedBuffer release];
    allocatedBuffer = nil;
  }

  return 0;
}
