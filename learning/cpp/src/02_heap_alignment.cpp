#include <cstdlib>
#include <iostream>

int main() {

  void *my_ptr = nullptr;
  // posix_memalign allocates memory in heap that is aligned to the specified
  // boundary
  // Parameters: 1. address of pointer to be allocated
  //             2. alignment boundary
  //             3. size of memory to be allocated
  // For page aligned which is 16KB it helps in doing a buffer without copy to
  // GPU.
  // 16384 = 16 KB
  // 32768 = 32 KB
  int success_fail = posix_memalign(&my_ptr, 16384, 32768);
  if (success_fail == 0) {
    std::cerr << "Memory allocation Passed" << std::endl;
  }

  std::cout << "Memory address " << my_ptr << std::endl;
  free(my_ptr);
  return 0;
}