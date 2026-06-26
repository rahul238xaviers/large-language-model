#include "Activations.hpp"
#include "Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace activatations {

void silu_(Tensor &x) {
  std::transform(x.data().begin(), x.data().end(), x.data().begin(),
                 [](const float val) { return val * 1 / (1 + exp(-val)); });
}

Tensor swiglu(const Tensor &gate, const Tensor &up) {

  if (gate.shape() != up.shape()) {
    throw std::invalid_argument("Dimension mismatch for SwiGLU activation");
  }
  Tensor gated = gate;
  silu_(gated);
  gated.mul_(up);
  return gated;
}
} // namespace activatations
