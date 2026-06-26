#pragma once
#include "Tensor.hpp"

namespace activatations {

void silu_(Tensor &x);
Tensor swiglu(const Tensor &gate, const Tensor &up);
} // namespace activatations