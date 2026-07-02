#include <cstdint>
#include <iostream>

int main() {

  float first_val = 1.0f;
  float second_val = 2.0f;

  float *first_pointer = &first_val;
  float *second_pointer = &second_val;

  std::cout << "Address of first_val:" << first_pointer << std::endl;
  std::cout << "Address of second_val:" << second_pointer << std::endl;

  uintptr_t addr1 = reinterpret_cast<uintptr_t>(first_pointer);
  uintptr_t addr2 = reinterpret_cast<uintptr_t>(second_pointer);

  intptr_t diff = (addr2 - addr1);

  std::cout << "Diff:" << abs(diff) << std::endl;
  std::cout << "Address of first_val:" << addr1 << std::endl;
  std::cout << "Address of second_val:" << addr2 << std::endl;
  return 0;
}