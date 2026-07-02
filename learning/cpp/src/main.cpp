#include <iostream>

int main() {

  std::cout << "size of char:" << sizeof(char) << std::endl;
  std::cout << "size of float:" << sizeof(float) << std::endl;

  std::cout << "Align of char:" << alignof(char) << std::endl;
  std::cout << "Align of float:" << alignof(float) << std::endl;

  return 0;
}