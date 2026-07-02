#include <cstdlib>
#include <iostream>

int main() {

  void *my_ptr = nullptr;

  int success_fail = posix_memalign(&my_ptr, 16384, 32768);
  if (success_fail == 0) {
    std::cerr << "Memory allocation Passed" << std::endl;
  }

  std::cout << "Memory address " << my_ptr << std::endl;
  free(my_ptr);
  return 0;
}